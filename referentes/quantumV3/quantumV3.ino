/*
 * cuantumsV3_compact.ino — Boundary Node (Non-Blocking, Bidirectional, Compact Observables)
 *
 * I2C RX supported:
 *  - "T,<tick>,<mode>,<noise>\n"     (legacy tick)
 *  - "P,<noise>,<bias>,<kp>,<mode>\n" (set params)
 *
 * I2C TX (always <= 32 bytes):
 *  - "O,%08lX,%04X,%04X,%02X\n"
 *     tickHex (32-bit), bmask(16), loss(16), noiseByte(0..255)
 *
 * Constraints: no while(1), no return in setup, callbacks are light.
 */

#include <Arduino.h>
#include <Wire.h>

// ============================ CONFIG ============================
#define SLAVE_ADDRESS 0x10  // CHANGE PER NODE: 0x10,0x11,0x12,0x13...

// Pick a pin that is NOT used by SDA/SCL. Avoid 6/7 if those are I2C.
// V3 Spec: Force pin 7 for visual feedback
static constexpr int LED_PIN = 7;

static constexpr uint8_t I2C_MAX_RX = 32;
static constexpr uint32_t WORK_STEP_PERIOD_MS = 20;
static constexpr uint32_t HEALTH_PRINT_MS     = 3000;
static constexpr uint16_t LED_PULSE_MS        = 30;   // visible pulse

// ============================ FLAGS =============================
static bool i2cOK = false;
static bool ledOK = false;
static volatile bool ledTriggered = false; // Trigger from ISR

// ============================ BUFFERS ===========================
static volatile bool cmdPending = false;
static volatile uint8_t rxLen = 0;
static char rxBuf[I2C_MAX_RX + 1];

static char txBuf[32]; // guaranteed short
static volatile uint8_t txLen = 0;

// ============================ STATE =============================
static uint32_t currentTick = 0;
static uint8_t  currentMode = 1;

// Params (bidirectional)
static float  paramNoise = 0.05f;  // T
static int8_t paramBias  = 0;      // B
static uint8_t paramKp   = 0;      // coupling

// Toy boundary
static uint16_t boundaryState = 0xACE1u;
static uint16_t bmask = 0;
static uint16_t lossCount = 0;
static uint32_t prng = 0;

// Scheduler
static uint32_t lastWorkMs = 0;
static uint32_t lastHealthMs = 0;
static uint32_t ledOffAtMs = 0;

// ============================ UTILS =============================
static inline float clamp01(float x) {
  if (x < 0.0f) return 0.0f;
  if (x > 1.0f) return 1.0f;
  return x;
}

static inline uint16_t lfsr16_step(uint16_t s) {
  uint16_t lsb = s & 1u;
  s >>= 1u;
  if (lsb) s ^= 0xB400u;
  return s;
}

static inline uint32_t prng_step(uint32_t x) {
  // xorshift32
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  return x;
}

static inline uint8_t noiseToByte(float n) {
  n = clamp01(n);
  int v = (int)(n * 255.0f + 0.5f);
  if (v < 0) v = 0;
  if (v > 255) v = 255;
  return (uint8_t)v;
}

static void pulseLed() {
  if (!ledOK) return;
  digitalWrite(LED_PIN, HIGH);
  ledOffAtMs = millis() + LED_PULSE_MS;
}

// ============================ I2C CALLBACKS =====================
static void onReceiveEvent(int howMany) {
  uint8_t i = 0;
  while (Wire.available() && i < I2C_MAX_RX) {
    rxBuf[i++] = (char)Wire.read();
  }
  rxBuf[i] = '\0';
  rxLen = i;
  cmdPending = true;
  ledTriggered = true; // Flag for loop
}

static void onRequestEvent() {
  Wire.write((const uint8_t*)txBuf, (uint8_t)strnlen(txBuf, sizeof(txBuf)));
  ledTriggered = true; // Flag for loop
}

// ============================ PARSERS ===========================
static bool parseTickLegacy(const char* s, uint32_t* outTick, uint8_t* outMode, float* outNoise) {
  // "T,<tick>,<mode>,<noise>"
  char prefix = 0;
  unsigned long t = 0;
  unsigned int m = 0;
  float n = 0.0f;
  int parsed = sscanf(s, "%c,%lu,%u,%f", &prefix, &t, &m, &n);
  if (parsed < 4 || prefix != 'T') return false;
  *outTick = (uint32_t)t;
  *outMode = (uint8_t)m;
  *outNoise = n;
  return true;
}

static bool parseParams(const char* s, float* outNoise, int* outBias, int* outKp, int* outMode) {
  // "P,<noise>,<bias>,<kp>,<mode>"
  char prefix = 0;
  float n = 0.0f;
  int b = 0, k = 0, m = 0;
  int parsed = sscanf(s, "%c,%f,%d,%d,%d", &prefix, &n, &b, &k, &m);
  if (parsed < 5 || prefix != 'P') return false;
  *outNoise = n;
  *outBias = b;
  *outKp = k;
  *outMode = m;
  return true;
}

// ============================ DYNAMICS ==========================
static void updateBoundaryState() {
  boundaryState = lfsr16_step(boundaryState);

  prng = prng_step(prng);
  uint16_t r = (uint16_t)(prng >> 16);

  float n = clamp01(paramNoise);

  // Bias: push toward parity outcome (toy). Stronger bias => more stable bmask bits.
  float b = (float)abs((int)paramBias) / 127.0f;

  // Coupling: introduces correlations by copying bit neighborhoods
  uint8_t copies = (uint8_t)(paramKp / 32); // 0..7

  // Noise-driven flips
  uint8_t flips = 0;
  if (n > 0.70f) flips = 3;
  else if (n > 0.35f) flips = 2;
  else if (n > 0.10f) flips = 1;

  for (uint8_t i = 0; i < flips; i++) {
    uint8_t bit = (r + i * 7) & 0x0F;
    boundaryState ^= (1u << bit);
  }

  // Coupling copies
  for (uint8_t c = 0; c < copies; c++) {
    uint8_t src = (uint8_t)((r + 3 * c) & 0x0F);
    uint8_t dst = (uint8_t)((src + 1 + c) & 0x0F);
    bool v = (boundaryState >> src) & 1u;
    boundaryState = (uint16_t)((boundaryState & ~(1u << dst)) | ((uint16_t)v << dst));
  }

  // Compute group parities -> 4-bit observable
  uint16_t s = boundaryState;
  uint8_t g0 = __builtin_parity((unsigned)(s & 0x000Fu));
  uint8_t g1 = __builtin_parity((unsigned)((s >> 4) & 0x000Fu));
  uint8_t g2 = __builtin_parity((unsigned)((s >> 8) & 0x000Fu));
  uint8_t g3 = __builtin_parity((unsigned)((s >> 12) & 0x000Fu));

  uint16_t newMask = (uint16_t)((g0 << 0) | (g1 << 1) | (g2 << 2) | (g3 << 3));

  // Bias “freezes” mask occasionally
  if (b > 0.0f) {
    prng = prng_step(prng);
    float u = (float)(prng & 0xFFFFu) / 65535.0f;
    if (u < b * (1.0f - n)) {
      // keep previous bmask
      newMask = bmask;
    }
  }

  bmask = newMask;

  if (n > 0.60f && ((r & 0x0030u) == 0x0030u)) {
    lossCount++;
  }
}

// ============================ RESPONSE ==========================
static void buildResponseCompact() {
  // Always <= 32 bytes:
  // "O,%08lX,%04X,%04X,%02X\n" => 2 + 1 + 8 + 1 + 4 + 1 + 4 + 1 + 2 + 1 = 26 chars
  uint8_t nB = noiseToByte(paramNoise);
  snprintf(txBuf, sizeof(txBuf),
           "O,%08lX,%04X,%04X,%02X\n",
           (unsigned long)currentTick,
           (unsigned)bmask,
           (unsigned)lossCount,
           (unsigned)nB);
}

// ============================ SETUP / LOOP ======================
static uint32_t lastI2CRetryMs = 0;

static void initI2C() {
  if (i2cOK) return;
  if (millis() - lastI2CRetryMs < 2000) return;
  lastI2CRetryMs = millis();

  Wire.begin(SLAVE_ADDRESS);
  Wire.onReceive(onReceiveEvent);
  Wire.onRequest(onRequestEvent);
  i2cOK = true;

  Serial.print("[INIT] I2C addr 0x");
  Serial.println(SLAVE_ADDRESS, HEX);
}

void setup() {
  Serial.begin(115200);
  Serial.println("\n=== cuantumsV3_compact ===");
  Serial.print("I2C addr: 0x"); Serial.println(SLAVE_ADDRESS, HEX);

  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);
  ledOK = true;

  // Seed PRNG
  prng = (uint32_t)esp_random();
  boundaryState ^= (uint16_t)prng;

  buildResponseCompact();
  initI2C();

  Serial.println("[READY]");
}

void loop() {
  uint32_t now = millis();

  // LED Logic (Safe handling from ISR)
  if (ledTriggered) {
    ledTriggered = false;
    pulseLed(); // Sets ledOffAtMs
  }

  // LED auto-off
  if (ledOK && digitalRead(LED_PIN) == HIGH && (int32_t)(now - ledOffAtMs) >= 0) {
    digitalWrite(LED_PIN, LOW);
  }

  initI2C();

  if (cmdPending) {
    cmdPending = false;

    // Try P first
    float n = 0.0f;
    int b = 0, k = 0, m = 0;
    if (parseParams(rxBuf, &n, &b, &k, &m)) {
      paramNoise = clamp01(n);
      if (b < -127) b = -127;
      if (b > 127)  b = 127;
      paramBias = (int8_t)b;
      if (k < 0) k = 0;
      if (k > 255) k = 255;
      paramKp = (uint8_t)k;
      if (m < 0) m = 0;
      if (m > 255) m = 255;
      currentMode = (uint8_t)m;

      lossCount = 0; // optional reset on param update
      buildResponseCompact();

    } else {
      // Legacy T tick
      uint32_t t = 0;
      uint8_t md = 0;
      float tn = 0.0f;
      if (parseTickLegacy(rxBuf, &t, &md, &tn)) {
        currentTick = t;
        currentMode = md;
        paramNoise = clamp01(tn);

        updateBoundaryState();
        buildResponseCompact();
      } else {
        Serial.print("[I2C] Bad cmd: ");
        Serial.println(rxBuf);
      }
    }
  }

  // Background evolution (non-blocking)
  if (now - lastWorkMs >= WORK_STEP_PERIOD_MS) {
    lastWorkMs = now;
    if (currentTick > 0) {
      updateBoundaryState();
      buildResponseCompact();
    }
  }

  // Health
  if (now - lastHealthMs >= HEALTH_PRINT_MS) {
    lastHealthMs = now;
    Serial.printf("[HEALTH] tick=%lu bmask=0x%X loss=%u noise=%.2f bias=%d kp=%u mode=%u i2c=%u\n",
                  (unsigned long)currentTick,
                  (unsigned)bmask,
                  (unsigned)lossCount,
                  (double)paramNoise,
                  (int)paramBias,
                  (unsigned)paramKp,
                  (unsigned)currentMode,
                  (unsigned)i2cOK);
  }
}
