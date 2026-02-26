/*
 * masterV3_hybrid_full.ino — Toy Holography Master (Non-Blocking, Full Peripherals)
 *
 * Keeps: OLED (U8g2 SH1106), RTC DS3231, SD logging (HSPI), BME280, NeoPixel, buttons
 * Adds: PC bidirectional Serial control + robust slave parsing (long + compact)
 *
 * Slave protocols supported:
 * 1) LONG (legacy):  "O,<tick>,<bmask>,<loss>,<noise>,<seed>\n"
 * 2) COMPACT (recommended): "O,%08lX,%04X,%04X,%02X\n"
 *     -> tickHex, bmaskHex, lossHex, noiseByte (0..255)
 *
 * Master -> Slave commands:
 * - Legacy tick: "T,<tick>,<mode>,<noise>\n"
 * - Optional params (if your slave supports it): "P,<noise>,<bias>,<kp>,<mode>\n"
 *
 * Serial commands (PC -> Master), line based:
 *   @HELLO
 *   @SET  N=0.20 M=1            (N=noise 0..1, M=mode)
 *   @PARAM N=0.20 B=5 K=60 M=1  (broadcast P command to slaves if supported)
 *   @GET  K=200 STRIDE=1 BURN=20
 *   @STOP
 *
 * Data output to PC during batch:
 *   O,<tick>,<slaveIndex>,<bmask>,<loss>,<noise>,<seedOr0>,<fmt>
 *     fmt: 0=invalid, 1=long, 2=compact
 *
 * Constraints:
 * - No while(1){}
 * - No return in setup() to abort
 * - Always continue init, use flags, retry in loop()
 */

#include <Wire.h>
#include <U8g2lib.h>
#include <RTClib.h>
#include <Adafruit_NeoPixel.h>
#include "FS.h"
#include "SD.h"
#include "SPI.h"
#include <Adafruit_Sensor.h>
#include <Adafruit_BME280.h>

// ============================================================================
// PIN DEFINITIONS (KEEP AS CURRENT WORKING HARDWARE)
// ============================================================================

// Buttons
static constexpr int BTNA_PIN = 1;
static constexpr int BTNB_PIN = 2;

// NeoPixel
static constexpr int RGB_LED_PIN = 48;
static constexpr uint8_t NUM_RGB_LEDS = 1;

// SD (HSPI)
static constexpr int SD_CS_PIN   = 7;
static constexpr int SD_MOSI_PIN = 6;
static constexpr int SD_MISO_PIN = 4;
static constexpr int SD_SCLK_PIN = 5;

// ============================================================================
// I2C ADDRESSES
// ============================================================================
static constexpr uint8_t OLED_ADDR = 0x3C;
static constexpr uint8_t RTC_ADDR  = 0x68;
static constexpr uint8_t BME_ADDR  = 0x76;

// Slaves
static constexpr uint8_t SLAVE_ADDRESSES[] = { 0x10, 0x11, 0x12, 0x13 };
static constexpr uint8_t NUM_SLAVES = sizeof(SLAVE_ADDRESSES) / sizeof(SLAVE_ADDRESSES[0]);

// ============================================================================
// I2C SETTINGS
// ============================================================================
static constexpr uint32_t I2C_CLOCK_SPEED = 100000;
static constexpr uint16_t I2C_TIMEOUT_MS  = 200; // Increased for stability
static constexpr uint8_t  I2C_MAX_RX_BYTES = 32;

// ============================================================================
// TIMING (NON-BLOCKING SCHEDULER)
// ============================================================================
static constexpr uint32_t TICK_PERIOD_MS       = 1500;  // free-running tick if no PC batch
static constexpr uint32_t MIN_BATCH_PERIOD_MS  = 100;   // THROTTLE: Max 10 ticks/sec (very stable)
static constexpr uint32_t DISPLAY_PERIOD_MS    = 200;
static constexpr uint32_t HEALTH_PERIOD_MS     = 2000;
static constexpr uint32_t SD_RETRY_PERIOD_MS   = 10000;
static constexpr uint32_t BME_RETRY_PERIOD_MS  = 10000;
static constexpr uint32_t RTC_RETRY_PERIOD_MS  = 10000;
static constexpr uint32_t OLED_RETRY_PERIOD_MS = 10000;
static constexpr uint8_t  SLAVE_READS_PER_LOOP = 1; // Back to 1 for stability

// Buttons debounce (non-blocking)
static constexpr uint32_t BTN_DEBOUNCE_MS = 180;

// ============================================================================
// OLED (U8g2) — SH1106 1.3" 128x64
// ============================================================================
U8G2_SH1106_128X64_NONAME_F_HW_I2C display(U8G2_R0, U8X8_PIN_NONE);

// ============================================================================
// MODULES
// ============================================================================
RTC_DS3231 rtc;
Adafruit_NeoPixel rgbLed(NUM_RGB_LEDS, RGB_LED_PIN, NEO_GRB + NEO_KHZ800);
SPIClass spiSD(HSPI);
Adafruit_BME280 bme;

// ============================================================================
// GLOBAL FLAGS (NO ABORT IN SETUP)
// ============================================================================
static bool i2cOK  = false;
static bool oledOK = false;
static bool rtcOK  = false;
static bool sdOK   = false;
static bool bmeOK  = false;

// ============================================================================
// SLAVE DATA MODEL
// ============================================================================
struct SlaveObs {
  uint32_t tick = 0;
  uint16_t bmask = 0;
  uint16_t loss = 0;
  float noise = NAN;     // for long format
  uint8_t noiseByte = 0; // for compact format
  uint32_t seed = 0;
  uint8_t fmt = 0;       // 0 invalid, 1 long, 2 compact
  bool valid = false;
  uint32_t lastRxMs = 0;
};

static SlaveObs slaveObs[NUM_SLAVES];

// ============================================================================
// TOY BULK RECONSTRUCTION
// ============================================================================
static constexpr uint8_t N_LOGICALS = 6;
static constexpr uint8_t LOGICAL_REQ[N_LOGICALS] = {
  0b0001,
  0b0011,
  0b0110,
  0b1100,
  0b0101,
  0b1110
};

static uint8_t boundary_mask = 0;
static uint8_t bulk_mask = 0;
static float recon_ratio = 0.0f;

// ============================================================================
// ENV DATA
// ============================================================================
static float currentTemp = NAN;
static float currentHum  = NAN;

// ============================================================================
// STATE MACHINE
// ============================================================================
static uint32_t tickCounter = 0;
static uint32_t lastTickMs = 0;
static uint32_t lastDisplayMs = 0;
static uint32_t lastHealthMs = 0;
static uint32_t lastSdRetryMs = 0;
static uint32_t lastBmeRetryMs = 0;
static uint32_t lastRtcRetryMs = 0;
static uint32_t lastOledRetryMs = 0;

// History
static uint8_t reconHistory[70];
static uint8_t historyIdx = 0;

// Polling
static uint8_t pollIndex = 0;
static uint8_t slavesPolledThisTick = 0;
static bool tickInProgress = false;

// Simulation parameters
static uint8_t simMode = 1;
static float globalNoise = 0.05f; // 0..1

// Hybrid params (optional broadcast P)
static int8_t  globalBias = 0;    // -127..127
static uint8_t globalKp   = 0;    // 0..255

// SD
static constexpr const char* SD_FILENAME = "/datalog.csv";

// Batch driver (PC requested)
static bool batchActive = false;
static uint16_t batchRemaining = 0;
static uint16_t batchStride = 1;
static uint16_t batchBurn = 0;
static uint16_t strideCounter = 0;

// ============================================================================
// LED UTILS
// ============================================================================
static void setLed(uint8_t r, uint8_t g, uint8_t b) {
  rgbLed.setPixelColor(0, rgbLed.Color(r, g, b));
  rgbLed.show();
}
static void setLedIdle()    { setLed(0, 0, 50); }
static void setLedPolling() { setLed(255, 255, 0); }
static void setLedOk()      { setLed(0, 255, 0); }
static void setLedWarn()    { setLed(255, 128, 0); }
static void setLedErr()     { setLed(255, 0, 0); }

// ============================================================================
// UTILITIES
// ============================================================================
static inline uint8_t popcount8(uint8_t x) {
  uint8_t c = 0;
  while (x) { c += (x & 1); x >>= 1; }
  return c;
}

static inline float clamp01(float x) {
  if (x < 0.0f) return 0.0f;
  if (x > 1.0f) return 1.0f;
  return x;
}

// Timestamp string
static bool getTimestamp(char* out, size_t outLen) {
  if (rtcOK) {
    DateTime now = rtc.now();
    snprintf(out, outLen, "%04d-%02d-%02d %02d:%02d:%02d",
             now.year(), now.month(), now.day(),
             now.hour(), now.minute(), now.second());
    return true;
  }
  snprintf(out, outLen, "ms:%lu", (unsigned long)millis());
  return false;
}

// ============================================================================
// INIT HELPERS (NEVER ABORT)
// ============================================================================
static void initI2C() {
  // Keep default pins for your master board (as in your current code)
  Wire.begin();
  Wire.setClock(I2C_CLOCK_SPEED);
  Wire.setTimeOut(I2C_TIMEOUT_MS);
  i2cOK = true;
}

static void initOLED() {
  if (!oledOK) {
    display.begin();
    oledOK = true;
  }
}

static void initRTC() {
  if (!rtcOK) {
    if (rtc.begin()) rtcOK = true;
  }
}

static void initSD() {
  if (!sdOK) {
    spiSD.begin(SD_SCLK_PIN, SD_MISO_PIN, SD_MOSI_PIN, SD_CS_PIN);
    if (SD.begin(SD_CS_PIN, spiSD)) {
      sdOK = true;

      if (!SD.exists(SD_FILENAME)) {
        File f = SD.open(SD_FILENAME, FILE_WRITE);
        if (f) {
          f.println("iso_time,tick,active_slaves,boundary_mask,bulk_mask,recon_ratio,loss_sum,noise_avg,temp,hum,mode,noise,bias,kp");
          f.close();
        }
      }
    }
  }
}

static void initBME() {
  if (!bmeOK) {
    if (bme.begin(BME_ADDR)) bmeOK = true;
  }
}

// ============================================================================
// I2C PROTOCOL
// ============================================================================
static void sendTickToSlave(uint8_t addr, uint32_t tick, uint8_t mode, float noise) {
  char msg[32];
  snprintf(msg, sizeof(msg), "T,%lu,%u,%.2f\n",
           (unsigned long)tick, (unsigned)mode, (double)noise);

  Wire.beginTransmission(addr);
  Wire.write((const uint8_t*)msg, (uint8_t)strlen(msg));
  uint8_t err = Wire.endTransmission();
  if (err != 0) {
    // not fatal
    // Serial.printf("[I2C] T->0x%02X err=%u\n", addr, (unsigned)err);
  }
}

static void sendParamsToSlave(uint8_t addr, float noise, int8_t bias, uint8_t kp, uint8_t mode) {
  // Optional; only works if slave supports 'P'
  char msg[32];
  snprintf(msg, sizeof(msg), "P,%.2f,%d,%u,%u\n",
           (double)clamp01(noise), (int)bias, (unsigned)kp, (unsigned)mode);

  Wire.beginTransmission(addr);
  Wire.write((const uint8_t*)msg, (uint8_t)strlen(msg));
  uint8_t err = Wire.endTransmission();
  if (err != 0) {
    // Serial.printf("[I2C] P->0x%02X err=%u\n", addr, (unsigned)err);
  }
}

static bool parseCompactObs(const char* buf,
                            uint32_t expectedTick,
                            uint32_t* outTick,
                            uint16_t* outBmask,
                            uint16_t* outLoss,
                            uint8_t* outNoiseByte) {
  // "O,%08lX,%04X,%04X,%02X"
  char prefix = 0;
  unsigned long tickHex = 0;
  unsigned bmaskHex = 0, lossHex = 0, noiseHex = 0;

  int ok = sscanf(buf, "%c,%lx,%x,%x,%x", &prefix, &tickHex, &bmaskHex, &lossHex, &noiseHex);
  if (ok < 5 || prefix != 'O') return false;

  if ((uint32_t)tickHex != expectedTick) return false;

  *outTick = (uint32_t)tickHex;
  *outBmask = (uint16_t)(bmaskHex & 0xFFFFu);
  *outLoss  = (uint16_t)(lossHex  & 0xFFFFu);
  *outNoiseByte = (uint8_t)(noiseHex & 0xFFu);
  return true;
}

static bool parseLongObs(const char* buf,
                         uint32_t expectedTick,
                         uint32_t* outTick,
                         uint16_t* outBmask,
                         uint16_t* outLoss,
                         float* outNoise,
                         uint32_t* outSeed) {
  // "O,<tick>,<bmask>,<loss>,<noise>,<seed>"
  char prefix = 0;
  unsigned long tick = 0;
  unsigned int bmask = 0;
  unsigned int loss = 0;
  float noise = 0.0f;
  unsigned long seed = 0;

  int parsed = sscanf(buf, "%c,%lu,%u,%u,%f,%lu", &prefix, &tick, &bmask, &loss, &noise, &seed);
  if (parsed < 6 || prefix != 'O') return false;

  if ((uint32_t)tick != expectedTick) return false;

  *outTick = (uint32_t)tick;
  *outBmask = (uint16_t)bmask;
  *outLoss  = (uint16_t)loss;
  *outNoise = noise;
  *outSeed  = (uint32_t)seed;
  return true;
}

static bool readObsFromSlave(uint8_t idx, uint8_t addr, uint32_t expectedTick) {
  uint8_t n = Wire.requestFrom(addr, I2C_MAX_RX_BYTES);
  if (n == 0) return false;

  char buf[I2C_MAX_RX_BYTES + 1];
  uint8_t i = 0;
  while (Wire.available() && i < I2C_MAX_RX_BYTES) {
    buf[i++] = (char)Wire.read();
  }
  buf[i] = '\0';

  uint32_t t = 0;
  uint16_t bm = 0, ls = 0;
  uint8_t nb = 0;
  float nz = NAN;
  uint32_t sd = 0;

  // Try COMPACT first (more robust)
  if (parseCompactObs(buf, expectedTick, &t, &bm, &ls, &nb)) {
    slaveObs[idx].tick = t;
    slaveObs[idx].bmask = bm;
    slaveObs[idx].loss = ls;
    slaveObs[idx].noiseByte = nb;
    slaveObs[idx].noise = (float)nb / 255.0f;
    slaveObs[idx].seed = 0;
    slaveObs[idx].fmt = 2;
    slaveObs[idx].valid = true;
    slaveObs[idx].lastRxMs = millis();
    return true;
  }

  // Then try LONG (legacy)
  if (parseLongObs(buf, expectedTick, &t, &bm, &ls, &nz, &sd)) {
    slaveObs[idx].tick = t;
    slaveObs[idx].bmask = bm;
    slaveObs[idx].loss = ls;
    slaveObs[idx].noise = nz;
    slaveObs[idx].noiseByte = (uint8_t)(clamp01(nz) * 255.0f + 0.5f);
    slaveObs[idx].seed = sd;
    slaveObs[idx].fmt = 1;
    slaveObs[idx].valid = true;
    slaveObs[idx].lastRxMs = millis();
    return true;
  }

  // Parse failed (likely truncation)
  // Serial.printf("[I2C] Parse fail 0x%02X: '%s'\n", addr, buf);
  return false;
}

// ============================================================================
// BULK RECONSTRUCTION
// ============================================================================
static void computeBulkFromBoundary() {
  bulk_mask = 0;
  for (uint8_t li = 0; li < N_LOGICALS; li++) {
    uint8_t req = LOGICAL_REQ[li];
    if ((boundary_mask & req) == req) {
      bulk_mask |= (1u << li);
    }
  }

  uint8_t reconCount = popcount8(bulk_mask);
  recon_ratio = (N_LOGICALS > 0) ? (100.0f * reconCount / (float)N_LOGICALS) : 0.0f;

  reconHistory[historyIdx] = (uint8_t)recon_ratio;
  historyIdx = (historyIdx + 1) % (sizeof(reconHistory) / sizeof(reconHistory[0]));
}

// ============================================================================
// SD LOGGING
// ============================================================================
static void logToSD() {
  if (!sdOK) return;

  File f = SD.open(SD_FILENAME, FILE_APPEND);
  if (!f) {
    Serial.println("[SD] Failed to open datalog.csv for append");
    sdOK = false;
    return;
  }

  char ts[32];
  getTimestamp(ts, sizeof(ts));

  uint8_t active = popcount8(boundary_mask);

  uint32_t lossSum = 0;
  float noiseSum = 0.0f;
  uint8_t validCount = 0;

  for (uint8_t i = 0; i < NUM_SLAVES; i++) {
    if (slaveObs[i].valid) {
      lossSum += slaveObs[i].loss;
      noiseSum += slaveObs[i].noise;
      validCount++;
    }
  }
  float noiseAvg = (validCount > 0) ? (noiseSum / validCount) : NAN;

  f.printf("%s,%lu,%u,0x%02X,0x%02X,%.2f,%lu,%.3f,%.2f,%.2f,%u,%.2f,%d,%u\n",
           ts,
           (unsigned long)tickCounter,
           (unsigned)active,
           (unsigned)boundary_mask,
           (unsigned)bulk_mask,
           (double)recon_ratio,
           (unsigned long)lossSum,
           (double)noiseAvg,
           (double)currentTemp,
           (double)currentHum,
           (unsigned)simMode,
           (double)globalNoise,
           (int)globalBias,
           (unsigned)globalKp);

  f.close();
}

// ============================================================================
// OLED DRAW
// ============================================================================
static void drawOLED() {
  if (!oledOK) return;

  display.clearBuffer();

  display.setFont(u8g2_font_6x10_tr);
  char line1[32];
  snprintf(line1, sizeof(line1), "T:%lu", (unsigned long)tickCounter);
  display.drawStr(0, 10, line1);

  display.setFont(u8g2_font_4x6_tr);
  for (uint8_t i = 0; i < NUM_SLAVES && i < 4; i++) {
    int x = 45 + i * 15;
    display.drawStr(x, 10, slaveObs[i].valid ? "OK" : "X");
  }

  display.setFont(u8g2_font_6x10_tr);
  display.drawStr(0, 24, "Bulk");

  int barX = 28, barY = 14, barMaxW = 98, barH = 12;
  display.drawFrame(barX, barY, barMaxW, barH);

  int barW = (int)((recon_ratio / 100.0f) * (barMaxW - 2));
  if (barW < 0) barW = 0;
  if (barW > (barMaxW - 2)) barW = (barMaxW - 2);
  display.drawBox(barX + 1, barY + 1, barW, barH - 2);

  char ratioStr[16];
  snprintf(ratioStr, sizeof(ratioStr), "%.0f%%", (double)recon_ratio);
  int strWidth = display.getStrWidth(ratioStr);
  int textX = barX + (barMaxW - strWidth) / 2;
  int textY = barY + 9;
  display.setDrawColor(2);
  display.setFontMode(1);
  display.drawStr(textX, textY, ratioStr);
  display.setDrawColor(1);
  display.setFontMode(0);

  int baseX = 0, baseY = 32, cellW = 20, cellH = 10;
  for (uint8_t i = 0; i < 4; i++) {
    int cx = baseX + (i % 2) * (cellW + 2);
    int cy = baseY + (i / 2) * (cellH + 2);
    display.drawFrame(cx, cy, cellW, cellH);

    if (i < NUM_SLAVES && slaveObs[i].valid) {
      float n = slaveObs[i].noise;
      n = clamp01(n);
      int fillW = (int)((1.0f - n) * (cellW - 2));
      if (fillW < 0) fillW = 0;
      display.drawBox(cx + 1, cy + 1, fillW, cellH - 2);
    }
  }

  int graphX = 60, graphW = 68, graphY = 32, graphH = 20;
  display.drawFrame(graphX, graphY, graphW, graphH);

  int count = sizeof(reconHistory) / sizeof(reconHistory[0]);
  for (int i = 0; i < graphW - 2; i++) {
    int idx = (historyIdx - 1 - i);
    while (idx < 0) idx += count;
    idx = idx % count;

    uint8_t val = reconHistory[idx];
    int pixelH = (int)((val / 100.0f) * (graphH - 2));
    if (pixelH > (graphH - 2)) pixelH = graphH - 2;

    if (i < count) {
      display.drawVLine(graphX + graphW - 2 - i,
                        graphY + graphH - 1 - pixelH,
                        pixelH);
    }
  }

  display.setFont(u8g2_font_4x6_tr);
  char footer[32];
  snprintf(footer, sizeof(footer), "B:%02X K:%02X N:%.2f", (unsigned)boundary_mask, (unsigned)bulk_mask, (double)globalNoise);
  display.drawStr(0, 62, footer);

  char timeStr[16];
  if (rtcOK) {
    DateTime now = rtc.now();
    snprintf(timeStr, sizeof(timeStr), "%02d:%02d:%02d", now.hour(), now.minute(), now.second());
  } else {
    snprintf(timeStr, sizeof(timeStr), "--:--:--");
  }
  display.drawStr(75, 62, timeStr);

  display.sendBuffer();
}

// ============================================================================
// HEALTH / SERIAL REPORT
// ============================================================================
static void printHealth() {
  Serial.println("=== MasterV3 Health ===");
  Serial.printf("I2C:%s OLED:%s RTC:%s SD:%s BME:%s BATCH:%u REM:%u\n",
                i2cOK ? "OK" : "X",
                oledOK ? "OK" : "X",
                rtcOK ? "OK" : "X",
                sdOK ? "OK" : "X",
                bmeOK ? "OK" : "X",
                (unsigned)batchActive,
                (unsigned)batchRemaining);

  Serial.printf("Tick:%lu BoundaryMask:0x%02X BulkMask:0x%02X Recon:%.1f%% Mode:%u Noise:%.2f Bias:%d Kp:%u\n",
                (unsigned long)tickCounter,
                (unsigned)boundary_mask,
                (unsigned)bulk_mask,
                (double)recon_ratio,
                (unsigned)simMode,
                (double)globalNoise,
                (int)globalBias,
                (unsigned)globalKp);

  for (uint8_t i = 0; i < NUM_SLAVES; i++) {
    Serial.printf("Slave[%u] 0x%02X valid:%u fmt:%u loss:%u noise:%.3f lastRx:%lums\n",
                  (unsigned)i,
                  SLAVE_ADDRESSES[i],
                  (unsigned)slaveObs[i].valid,
                  (unsigned)slaveObs[i].fmt,
                  (unsigned)slaveObs[i].loss,
                  (double)slaveObs[i].noise,
                  (unsigned long)(millis() - slaveObs[i].lastRxMs));
  }
}

// ============================================================================
// TICK CONTROL
// ============================================================================
static void startNewTick() {
  tickCounter++;
  tickInProgress = true;
  slavesPolledThisTick = 0;
  pollIndex = 0;

  for (uint8_t i = 0; i < NUM_SLAVES; i++) slaveObs[i].valid = false;

  // Broadcast tick
  for (uint8_t i = 0; i < NUM_SLAVES; i++) {
    sendTickToSlave(SLAVE_ADDRESSES[i], tickCounter, simMode, globalNoise);
  }

  setLedPolling();
}

static void closeTick() {
  boundary_mask = 0;
  for (uint8_t i = 0; i < NUM_SLAVES; i++) {
    if (slaveObs[i].valid) boundary_mask |= (1u << i);
  }

  computeBulkFromBoundary();

  uint8_t validCount = popcount8(boundary_mask);
  if (validCount == NUM_SLAVES) setLedOk();
  else if (validCount == 0) setLedErr();
  else setLedWarn();

  if (bmeOK) {
    currentTemp = bme.readTemperature();
    currentHum  = bme.readHumidity();
  }

  logToSD();

  tickInProgress = false;
  setLedIdle();
}

static void pollSlavesStep() {
  uint8_t reads = 0;

  while (reads < SLAVE_READS_PER_LOOP && slavesPolledThisTick < NUM_SLAVES) {
    uint8_t idx = pollIndex;
    uint8_t addr = SLAVE_ADDRESSES[idx];

    // Read with robustness delay
    (void)readObsFromSlave(idx, addr, tickCounter);
    delay(4); // Short breather

    pollIndex++;
    slavesPolledThisTick++;
    reads++;
  }

  if (slavesPolledThisTick >= NUM_SLAVES) {
     // Serial.println("[DBG] Polling Complete -> Closing Tick");
     closeTick();
  }
}



// ============================================================================
// SERIAL COMMAND PARSER (PC CONTROL)
// ============================================================================
static char serialLine[160];
static uint16_t serialLen = 0;

static void serialResetLine() {
  serialLen = 0;
  serialLine[0] = 0;
}

static void serialWriteHello() {
  Serial.printf("@HELLO OK FW=MasterV3Full SLAVES=%u\n", (unsigned)NUM_SLAVES);
}

static void serialBroadcastParamsIfSupported() {
  // Not fatal if slaves don't support 'P'
  for (uint8_t i = 0; i < NUM_SLAVES; i++) {
    sendParamsToSlave(SLAVE_ADDRESSES[i], globalNoise, globalBias, globalKp, simMode);
  }
}

static void serialHandleSet(const char* s) {
  // @SET N=0.20 M=1
  const char* p = s;
  while (*p) {
    if (strncmp(p, "N=", 2) == 0) globalNoise = clamp01(atof(p + 2));
    else if (strncmp(p, "M=", 2) == 0) simMode = (uint8_t)atoi(p + 2);

    while (*p && *p != ' ') p++;
    while (*p == ' ') p++;
  }
  Serial.printf("@ACK SET N=%.2f M=%u\n", (double)globalNoise, (unsigned)simMode);
}

static void serialHandleParam(const char* s) {
  // @PARAM N=0.20 B=5 K=60 M=1
  const char* p = s;
  while (*p) {
    if (strncmp(p, "N=", 2) == 0) globalNoise = clamp01(atof(p + 2));
    else if (strncmp(p, "B=", 2) == 0) {
      int b = atoi(p + 2);
      if (b < -127) b = -127;
      if (b > 127)  b = 127;
      globalBias = (int8_t)b;
    } else if (strncmp(p, "K=", 2) == 0) {
      int k = atoi(p + 2);
      if (k < 0) k = 0;
      if (k > 255) k = 255;
      globalKp = (uint8_t)k;
    } else if (strncmp(p, "M=", 2) == 0) {
      int m = atoi(p + 2);
      if (m < 0) m = 0;
      if (m > 255) m = 255;
      simMode = (uint8_t)m;
    }

    while (*p && *p != ' ') p++;
    while (*p == ' ') p++;
  }

  serialBroadcastParamsIfSupported();
  Serial.printf("@ACK PARAM N=%.2f B=%d K=%u M=%u\n",
                (double)globalNoise, (int)globalBias, (unsigned)globalKp, (unsigned)simMode);
}

static void serialHandleGet(const char* s) {
  // @GET K=200 STRIDE=1 BURN=20
  int K = 200, STR = 1, BURN = 0;
  const char* p = s;
  while (*p) {
    if (strncmp(p, "K=", 2) == 0) K = atoi(p + 2);
    else if (strncmp(p, "STRIDE=", 7) == 0) STR = atoi(p + 7);
    else if (strncmp(p, "BURN=", 5) == 0) BURN = atoi(p + 5);

    while (*p && *p != ' ') p++;
    while (*p == ' ') p++;
  }

  if (K < 1) K = 1;
  if (K > 2000) K = 2000;
  if (STR < 1) STR = 1;
  if (STR > 1000) STR = 1000;
  if (BURN < 0) BURN = 0;
  if (BURN > 5000) BURN = 5000;

  batchActive = true;
  batchRemaining = (uint16_t)K;
  batchStride = (uint16_t)STR;
  batchBurn = (uint16_t)BURN;
  strideCounter = 0;

  Serial.printf("@BATCH K=%u STRIDE=%u BURN=%u N=%.2f M=%u B=%d Kp=%u\n",
                (unsigned)batchRemaining,
                (unsigned)batchStride,
                (unsigned)batchBurn,
                (double)globalNoise,
                (unsigned)simMode,
                (int)globalBias,
                (unsigned)globalKp);
}

static void serialHandleStop() {
  batchActive = false;
  batchRemaining = 0;
  Serial.println("@ACK STOP");
}

static void serialParseLine(char* line) {
  while (*line == ' ' || *line == '\t') line++;
  if (*line == 0) return;

  if (strncmp(line, "@HELLO", 6) == 0) serialWriteHello();
  else if (strncmp(line, "@SET", 4) == 0) serialHandleSet(line + 4);
  else if (strncmp(line, "@PARAM", 6) == 0) serialHandleParam(line + 6);
  else if (strncmp(line, "@GET", 4) == 0) serialHandleGet(line + 4);
  else if (strncmp(line, "@STOP", 5) == 0) serialHandleStop();
  else Serial.printf("@ERR UNKNOWN '%s'\n", line);
}

static void serialStep() {
  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\r') continue;
    if (c == '\n') {
      serialLine[serialLen] = 0;
      serialParseLine(serialLine);
      serialResetLine();
    } else {
      if (serialLen < sizeof(serialLine) - 1) serialLine[serialLen++] = c;
      else {
        serialResetLine();
        Serial.println("@ERR LINE_OVERFLOW");
      }
    }
  }
}

// ============================================================================
// BUTTONS (NON-BLOCKING DEBOUNCE)
// ============================================================================
static uint32_t lastBtnAMs = 0;
static uint32_t lastBtnBMs = 0;

static void buttonsStep(uint32_t nowMs) {
  if (digitalRead(BTNA_PIN) == LOW) {
    if (nowMs - lastBtnAMs >= BTN_DEBOUNCE_MS) {
      lastBtnAMs = nowMs;
      globalNoise = clamp01(globalNoise + 0.01f);
      Serial.printf("[BTN] A -> noise=%.2f\n", (double)globalNoise);
    }
  }
  if (digitalRead(BTNB_PIN) == LOW) {
    if (nowMs - lastBtnBMs >= BTN_DEBOUNCE_MS) {
      lastBtnBMs = nowMs;
      globalNoise = clamp01(globalNoise - 0.01f);
      Serial.printf("[BTN] B -> noise=%.2f\n", (double)globalNoise);
    }
  }
}

// ============================================================================
// SETUP / LOOP
// ============================================================================
void setup() {
  Serial.begin(115200);
  Serial.println("\n=== MasterV3 Full (Hybrid Ready) ===");

  pinMode(BTNA_PIN, INPUT_PULLUP);
  pinMode(BTNB_PIN, INPUT_PULLUP);

  rgbLed.begin();
  rgbLed.setBrightness(50);
  rgbLed.show();
  setLedIdle();

  // Init (never abort)
  initI2C();
  initOLED();
  initRTC();
  initSD();
  initBME();

  lastTickMs = millis();
  lastDisplayMs = millis();
  lastHealthMs = millis();
  serialResetLine();

  Serial.println("[MasterV3] Setup complete (non-blocking).");
  Serial.println("Commands: @HELLO | @SET N=0.20 M=1 | @PARAM N=0.2 B=5 K=60 M=1 | @GET K=200 STRIDE=1 BURN=20 | @STOP");
}

void loop() {
  uint32_t nowMs = millis();

  serialStep();
  buttonsStep(nowMs);

  // Batch driver has priority over free-running tick rate
  if (batchActive) {
    if (!tickInProgress) {
        // RATE LIMIT: Don't hammer the bus too hard
        if (nowMs - lastTickMs >= MIN_BATCH_PERIOD_MS) {
            
            // Not currently waiting for slaves. Decide what to do next.
            if (batchBurn > 0) {
                startNewTick();
                lastTickMs = nowMs; 
            } else {
                // Normal sampling or stride
                startNewTick();
                lastTickMs = nowMs;
            }
        }
    }
  }

  // Common Polling Logic (for both Batch and Free-running)
  if (tickInProgress) {
      pollSlavesStep();
      
      // If poll finished this cycle (tick just closed)
      if (!tickInProgress && batchActive) {
          // DEBUG: Trace why batch is stuck
          Serial.printf("[BATCH_DBG] Burn=%u Stride=%u Rem=%u\n", (unsigned)batchBurn, (unsigned)strideCounter, (unsigned)batchRemaining);

          if (batchBurn > 0) {
              batchBurn--;
          } else {
              strideCounter++;
              if (strideCounter >= batchStride) {
                  strideCounter = 0;
                  // Emit one sample (one line per slave)
                  for (uint8_t i = 0; i < NUM_SLAVES; i++) {
                      Serial.printf("O,%lu,%u,%u,%u,%.3f,%lu,%u\n",
                                    (unsigned long)tickCounter,
                                    (unsigned)i,
                                    (unsigned)(slaveObs[i].valid ? slaveObs[i].bmask : 0),
                                    (unsigned)(slaveObs[i].valid ? slaveObs[i].loss : 0),
                                    (double)(slaveObs[i].valid ? slaveObs[i].noise : 0.0f),
                                    (unsigned long)(slaveObs[i].valid ? slaveObs[i].seed : 0),
                                    (unsigned)(slaveObs[i].valid ? slaveObs[i].fmt : 0));
                  }
                  
                  if (batchRemaining > 0) batchRemaining--;
                  if (batchRemaining == 0) {
                      batchActive = false;
                      Serial.printf("@DONE LASTTICK=%lu\n", (unsigned long)tickCounter);
                  }
              }
          }
      }
      // NO RETURN HERE -> Fall through to OLED/Health
  } else if (!batchActive && (nowMs - lastTickMs >= TICK_PERIOD_MS)) {
      // Free running mode start (only if not batch)
      lastTickMs = nowMs;
      startNewTick();
  }

  // Duplicate free-running scheduler removed.
  // The block above (lines 920-924) handles it correctly with !batchActive check.



  if (oledOK && (nowMs - lastDisplayMs >= DISPLAY_PERIOD_MS)) {
    lastDisplayMs = nowMs;
    drawOLED();
  }

  if (nowMs - lastHealthMs >= HEALTH_PERIOD_MS) {
    lastHealthMs = nowMs;
    printHealth();
  }

  // Retry init (non-spam)
  if (!sdOK && (nowMs - lastSdRetryMs >= SD_RETRY_PERIOD_MS)) {
    lastSdRetryMs = nowMs;
    Serial.println("[SD] Retry init...");
    initSD();
  }
  if (!bmeOK && (nowMs - lastBmeRetryMs >= BME_RETRY_PERIOD_MS)) {
    lastBmeRetryMs = nowMs;
    Serial.println("[BME] Retry init...");
    initBME();
  }
  if (!rtcOK && (nowMs - lastRtcRetryMs >= RTC_RETRY_PERIOD_MS)) {
    lastRtcRetryMs = nowMs;
    Serial.println("[RTC] Retry init...");
    initRTC();
  }
  if (!oledOK && (nowMs - lastOledRetryMs >= OLED_RETRY_PERIOD_MS)) {
    lastOledRetryMs = nowMs;
    Serial.println("[OLED] Retry init...");
    initOLED();
  }
}
