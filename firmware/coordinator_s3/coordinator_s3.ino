/*
 * cuantum_coordinator_s3.ino — CuantumWiki Tone Detection Coordinator
 *
 * ESP32-S3 — Master I2C + Serial Bridge
 *
 * Adapted from masterV3.ino reference for the CuantumWiki tone
 * detection architecture.
 *
 * Architecture:
 *   Web App <-> Serial (JSONL) <-> This S3 <-> I2C <-> 4x C3 Workers
 *
 * Serial Protocol (PC/Web <-> S3):
 *   TX from Web:
 * {"type":"infer","req_id":"r-001","model":"tone-v1","tokens":[12,45,77],"max_len":64,"route":"auto"}
 *   RX to Web:
 * {"type":"result","req_id":"r-001","label":"neutral","confidence":0.91,"node":"c3-2","latency_ms":38}
 *   RX errors:
 * {"type":"error","req_id":"r-001","code":"NODE_TIMEOUT","message":"worker no
 * respondió"}
 *
 * I2C Protocol (S3 <-> C3):
 *   S3 -> C3:  "I,<req_id>,<num_tokens>,<token0>,<token1>,...\n"  (chunked if
 * needed) C3 -> S3:  "R,<req_id>,<label_idx>,<confidence_x100>,<latency_ms>\n"
 *
 * Peripherals: OLED SH1106, RTC DS3231, SD (HSPI), BME280, NeoPixel, 2 buttons
 *
 * Constraints:
 * - No while(1){}
 * - No return in setup() to abort
 * - Always continue init, use flags, retry in loop()
 */

#include "FS.h"
#include "SD.h"
#include "SPI.h"
#include <Adafruit_BME280.h>
#include <Adafruit_NeoPixel.h>
#include <Adafruit_Sensor.h>
#include <ArduinoJson.h>
#include <RTClib.h>
#include <U8g2lib.h>
#include <Wire.h>

// ============================================================================
// PIN DEFINITIONS
// ============================================================================
static constexpr int BTNA_PIN = 1;
static constexpr int BTNB_PIN = 2;
static constexpr int RGB_LED_PIN = 48;
static constexpr uint8_t NUM_RGB_LEDS = 1;
static constexpr int SD_CS_PIN = 7;
static constexpr int SD_MOSI_PIN = 6;
static constexpr int SD_MISO_PIN = 4;
static constexpr int SD_SCLK_PIN = 5;

// ============================================================================
// I2C CONFIG
// ============================================================================
static constexpr uint8_t OLED_ADDR = 0x3C;
static constexpr uint8_t RTC_ADDR = 0x68;
static constexpr uint8_t BME_ADDR = 0x76;

static constexpr uint8_t SLAVE_ADDRESSES[] = {0x10, 0x11, 0x12, 0x13};
static constexpr uint8_t NUM_SLAVES =
    sizeof(SLAVE_ADDRESSES) / sizeof(SLAVE_ADDRESSES[0]);
static constexpr uint32_t I2C_CLOCK_SPEED = 100000;
static constexpr uint16_t I2C_TIMEOUT_MS = 500;
static constexpr uint8_t I2C_MAX_RX_BYTES = 32;

// ============================================================================
// TIMING
// ============================================================================
static constexpr uint32_t DISPLAY_PERIOD_MS = 250;
static constexpr uint32_t HEALTH_PERIOD_MS = 3000;
static constexpr uint32_t SD_RETRY_PERIOD_MS = 10000;
static constexpr uint32_t BME_RETRY_PERIOD_MS = 10000;
static constexpr uint32_t RTC_RETRY_PERIOD_MS = 10000;
static constexpr uint32_t OLED_RETRY_PERIOD_MS = 10000;
static constexpr uint32_t PING_PERIOD_MS = 5000;
static constexpr uint32_t INFER_TIMEOUT_MS = 5000;
static constexpr uint32_t BTN_DEBOUNCE_MS = 180;

// ============================================================================
// LABEL DEFINITIONS
// ============================================================================
static const char *LABELS[] = {"neutral", "positivo", "urgente", "molesto",
                               "formal"};
static constexpr uint8_t NUM_LABELS = 5;

// ============================================================================
// MODULES
// ============================================================================
U8G2_SH1106_128X64_NONAME_F_HW_I2C display(U8G2_R0, U8X8_PIN_NONE);
RTC_DS3231 rtc;
Adafruit_NeoPixel rgbLed(NUM_RGB_LEDS, RGB_LED_PIN, NEO_GRB + NEO_KHZ800);
SPIClass spiSD(HSPI);
Adafruit_BME280 bme;

// ============================================================================
// FLAGS
// ============================================================================
static bool i2cOK = false;
static bool oledOK = false;
static bool rtcOK = false;
static bool sdOK = false;
static bool bmeOK = false;

// ============================================================================
// NODE STATE
// ============================================================================
struct NodeState {
  bool alive = false;
  uint32_t lastPingMs = 0;
  uint32_t lastSeenMs = 0;
  uint16_t totalInfers = 0;
  uint16_t totalErrors = 0;
  uint16_t avgLatency = 0;
  char lastLabel[12] = "";
};
static NodeState nodeState[NUM_SLAVES];

// ============================================================================
// REQUEST QUEUE
// ============================================================================
struct InferRequest {
  bool active = false;
  char reqId[16] = "";
  char model[16] = "";
  int16_t tokens[128];
  uint8_t numTokens = 0;
  uint8_t maxLen = 64;
  char route[8] = "auto";
  uint8_t assignedNode = 0xFF;
  uint32_t sentAtMs = 0;
  bool waitingResponse = false;
};
static constexpr uint8_t MAX_QUEUE = 8;
static InferRequest requestQueue[MAX_QUEUE];
static uint8_t roundRobinIdx = 0;

// ============================================================================
// ENVIRONMENT
// ============================================================================
static float currentTemp = NAN;
static float currentHum = NAN;

// ============================================================================
// TIMING STATE
// ============================================================================
static uint32_t lastDisplayMs = 0;
static uint32_t lastHealthMs = 0;
static uint32_t lastPingMs = 0;
static uint32_t lastSdRetryMs = 0;
static uint32_t lastBmeRetryMs = 0;
static uint32_t lastRtcRetryMs = 0;
static uint32_t lastOledRetryMs = 0;
static uint32_t totalInfers = 0;
static uint32_t totalErrors = 0;

// Buttons
static uint32_t lastBtnAMs = 0;
static uint32_t lastBtnBMs = 0;
static uint8_t displayPage = 0;

// ============================================================================
// SERIAL LINE BUFFER
// ============================================================================
static char serialLine[512];
static uint16_t serialLen = 0;

// SD
static constexpr const char *SD_FILENAME = "/infer_log.csv";

// ============================================================================
// LED UTILS
// ============================================================================
static void setLed(uint8_t r, uint8_t g, uint8_t b) {
  rgbLed.setPixelColor(0, rgbLed.Color(r, g, b));
  rgbLed.show();
}
static void setLedIdle() { setLed(0, 0, 50); }
static void setLedBusy() { setLed(255, 255, 0); }
static void setLedOk() { setLed(0, 255, 0); }
static void setLedErr() { setLed(255, 0, 0); }

// ============================================================================
// UTILITIES
// ============================================================================
static bool getTimestamp(char *out, size_t outLen) {
  if (rtcOK) {
    DateTime now = rtc.now();
    snprintf(out, outLen, "%04d-%02d-%02d %02d:%02d:%02d", now.year(),
             now.month(), now.day(), now.hour(), now.minute(), now.second());
    return true;
  }
  snprintf(out, outLen, "ms:%lu", (unsigned long)millis());
  return false;
}

// ============================================================================
// INIT HELPERS
// ============================================================================
static void initI2C() {
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
    if (rtc.begin())
      rtcOK = true;
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
          f.println(
              "timestamp,req_id,label,confidence,node,latency_ms,temp,hum");
          f.close();
        }
      }
    }
  }
}

static void initBME() {
  if (!bmeOK) {
    if (bme.begin(BME_ADDR))
      bmeOK = true;
  }
}

// ============================================================================
// ROUTING — Round-Robin with health check
// ============================================================================
static uint8_t selectNode(const char *route) {
  // Specific node requested
  if (route[0] == 'c' && route[1] == '3' && route[2] == '-') {
    uint8_t idx = (uint8_t)(route[3] - '0');
    if (idx < NUM_SLAVES)
      return idx;
  }

  // Auto: round-robin, skip dead nodes (max 1 full rotation)
  for (uint8_t tries = 0; tries < NUM_SLAVES; tries++) {
    uint8_t idx = roundRobinIdx;
    roundRobinIdx = (roundRobinIdx + 1) % NUM_SLAVES;

    // Skip nodes that haven't responded recently (if we've pinged them)
    if (nodeState[idx].alive || nodeState[idx].lastPingMs == 0) {
      return idx;
    }
  }

  // Fallback: just use the next one
  uint8_t idx = roundRobinIdx;
  roundRobinIdx = (roundRobinIdx + 1) % NUM_SLAVES;
  return idx;
}

// ============================================================================
// I2C COMMUNICATION — Send tokens to worker
// ============================================================================
static bool sendTokensToWorker(uint8_t nodeIdx, const InferRequest *req) {
  uint8_t addr = SLAVE_ADDRESSES[nodeIdx];

  // Build command: "I,<reqId>,<numTokens>,<t0>,<t1>,...\n"
  // I2C max write is 32 bytes, so we send a compact header first,
  // then the tokens follow in a second transmission if needed.

  // Header: "I,<reqId>,<numTokens>\n"
  char hdr[32];
  snprintf(hdr, sizeof(hdr), "I,%s,%u\n", req->reqId, (unsigned)req->numTokens);

  Wire.beginTransmission(addr);
  Wire.write((const uint8_t *)hdr, (uint8_t)strlen(hdr));
  uint8_t err = Wire.endTransmission();

  if (err != 0)
    return false;

  // Send tokens in chunks of 16 (as binary int16)
  // Each chunk: 16 tokens * 2 bytes = 32 bytes (I2C max)
  for (uint8_t offset = 0; offset < req->numTokens; offset += 16) {
    uint8_t chunkLen = req->numTokens - offset;
    if (chunkLen > 16)
      chunkLen = 16;

    Wire.beginTransmission(addr);
    for (uint8_t i = 0; i < chunkLen; i++) {
      int16_t tok = req->tokens[offset + i];
      Wire.write((uint8_t)(tok & 0xFF));
      Wire.write((uint8_t)((tok >> 8) & 0xFF));
    }
    err = Wire.endTransmission();
    if (err != 0)
      return false;

    delay(10); // I2C breather — C3 needs time to process in ISR
  }

  return true;
}

// ============================================================================
// I2C COMMUNICATION — Read result from worker
// ============================================================================
static bool readResultFromWorker(uint8_t nodeIdx, char *outLabel,
                                 float *outConf, uint16_t *outLatency) {
  uint8_t addr = SLAVE_ADDRESSES[nodeIdx];
  uint8_t n = Wire.requestFrom(addr, I2C_MAX_RX_BYTES);
  if (n == 0)
    return false;

  char buf[I2C_MAX_RX_BYTES + 1];
  uint8_t i = 0;
  while (Wire.available() && i < I2C_MAX_RX_BYTES) {
    buf[i++] = (char)Wire.read();
  }
  buf[i] = '\0';

  // Parse: "R,<reqId>,<labelIdx>,<conf_x100>,<latency_ms>\n"
  char prefix = 0;
  char rxReqId[16] = "";
  unsigned int labelIdx = 0;
  unsigned int confX100 = 0;
  unsigned int latMs = 0;

  int parsed = sscanf(buf, "%c,%[^,],%u,%u,%u", &prefix, rxReqId, &labelIdx,
                      &confX100, &latMs);
  if (parsed < 5 || prefix != 'R')
    return false;

  if (labelIdx >= NUM_LABELS)
    labelIdx = 0;
  strncpy(outLabel, LABELS[labelIdx], 11);
  outLabel[11] = '\0';
  *outConf = (float)confX100 / 100.0f;
  *outLatency = (uint16_t)latMs;
  return true;
}

// ============================================================================
// SERIAL — Parse JSONL from PC/Web
// ============================================================================
static void serialResetLine() {
  serialLen = 0;
  serialLine[0] = 0;
}

static void handleSerialJSON(const char *line) {
  StaticJsonDocument<1024> doc;
  DeserializationError err = deserializeJson(doc, line);
  if (err) {
    Serial.println("{\"type\":\"error\",\"code\":\"PARSE_ERROR\",\"message\":"
                   "\"JSON inválido\"}");
    return;
  }

  const char *type = doc["type"] | "";

  if (strcmp(type, "infer") == 0) {
    // Find free slot in queue
    int slot = -1;
    for (int i = 0; i < MAX_QUEUE; i++) {
      if (!requestQueue[i].active) {
        slot = i;
        break;
      }
    }

    if (slot < 0) {
      Serial.printf("{\"type\":\"error\",\"req_id\":\"%s\",\"code\":\"QUEUE_"
                    "FULL\",\"message\":\"cola llena\"}\n",
                    doc["req_id"] | "?");
      return;
    }

    InferRequest *req = &requestQueue[slot];
    req->active = true;
    strncpy(req->reqId, doc["req_id"] | "?", 15);
    req->reqId[15] = '\0';
    strncpy(req->model, doc["model"] | "tone-v1", 15);
    req->model[15] = '\0';
    strncpy(req->route, doc["route"] | "auto", 7);
    req->route[7] = '\0';
    req->maxLen = doc["max_len"] | 64;

    // Parse tokens array
    JsonArray tokArr = doc["tokens"].as<JsonArray>();
    req->numTokens = 0;
    for (JsonVariant v : tokArr) {
      if (req->numTokens >= 128)
        break;
      req->tokens[req->numTokens++] = (int16_t)v.as<int>();
    }

    req->assignedNode = 0xFF;
    req->waitingResponse = false;
    req->sentAtMs = 0;

  } else if (strcmp(type, "hello") == 0) {
    Serial.printf("{\"type\":\"hello\",\"fw\":\"CuantumCoordinator\","
                  "\"version\":\"1.0\",\"slaves\":%u,\"labels\":[\"neutral\","
                  "\"positivo\",\"urgente\",\"molesto\",\"formal\"]}\n",
                  (unsigned)NUM_SLAVES);

  } else if (strcmp(type, "ping") == 0) {
    Serial.printf("{\"type\":\"pong\",\"uptime_ms\":%lu,\"nodes\":[",
                  (unsigned long)millis());
    for (uint8_t i = 0; i < NUM_SLAVES; i++) {
      Serial.printf("{\"id\":\"c3-%u\",\"alive\":%s,\"infers\":%u,\"errors\":%"
                    "u,\"avg_ms\":%u}%s",
                    (unsigned)i, nodeState[i].alive ? "true" : "false",
                    (unsigned)nodeState[i].totalInfers,
                    (unsigned)nodeState[i].totalErrors,
                    (unsigned)nodeState[i].avgLatency,
                    i < NUM_SLAVES - 1 ? "," : "");
    }
    Serial.println("]}");

  } else {
    Serial.printf("{\"type\":\"error\",\"code\":\"UNKNOWN_TYPE\",\"message\":"
                  "\"tipo '%s' desconocido\"}\n",
                  type);
  }
}

static void serialStep() {
  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\r')
      continue;
    if (c == '\n') {
      serialLine[serialLen] = 0;
      if (serialLen > 0)
        handleSerialJSON(serialLine);
      serialResetLine();
    } else {
      if (serialLen < sizeof(serialLine) - 1)
        serialLine[serialLen++] = c;
      else {
        serialResetLine();
        Serial.println("{\"type\":\"error\",\"code\":\"LINE_OVERFLOW\","
                       "\"message\":\"línea serial demasiado larga\"}");
      }
    }
  }
}

// ============================================================================
// REQUEST PROCESSING
// ============================================================================
static void processQueue() {
  for (uint8_t qi = 0; qi < MAX_QUEUE; qi++) {
    InferRequest *req = &requestQueue[qi];
    if (!req->active)
      continue;

    // Step 1: Assign node and send tokens
    if (!req->waitingResponse && req->assignedNode == 0xFF) {
      uint8_t node = selectNode(req->route);
      req->assignedNode = node;

      setLedBusy();
      bool ok = sendTokensToWorker(node, req);
      if (ok) {
        req->waitingResponse = true;
        req->sentAtMs = millis();
      } else {
        // Send failed — try next node or report error
        nodeState[node].totalErrors++;
        nodeState[node].alive = false;

        // Try one more node
        uint8_t alt = selectNode("auto");
        if (alt != node) {
          req->assignedNode = alt;
          ok = sendTokensToWorker(alt, req);
          if (ok) {
            req->waitingResponse = true;
            req->sentAtMs = millis();
          } else {
            // Both failed
            Serial.printf(
                "{\"type\":\"error\",\"req_id\":\"%s\",\"code\":\"SEND_"
                "FAILED\",\"message\":\"no se pudo enviar a workers\"}\n",
                req->reqId);
            req->active = false;
            totalErrors++;
            setLedErr();
          }
        } else {
          Serial.printf("{\"type\":\"error\",\"req_id\":\"%s\",\"code\":\"NO_"
                        "NODES\",\"message\":\"sin nodos disponibles\"}\n",
                        req->reqId);
          req->active = false;
          totalErrors++;
          setLedErr();
        }
      }
    }

    // Step 2: Poll for result
    if (req->active && req->waitingResponse) {
      uint32_t elapsed = millis() - req->sentAtMs;

      // Give worker time to compute (at least 50ms)
      if (elapsed < 50)
        continue;

      char label[12] = "";
      float confidence = 0.0f;
      uint16_t latency = 0;

      // Try reading result (polled statelessly without blocking)
      bool gotResult =
          readResultFromWorker(req->assignedNode, label, &confidence, &latency);

      if (gotResult) {
        // Success!
        uint8_t ni = req->assignedNode;
        nodeState[ni].alive = true;
        nodeState[ni].lastSeenMs = millis();
        nodeState[ni].totalInfers++;
        nodeState[ni].avgLatency =
            nodeState[ni].avgLatency == 0
                ? latency
                : (uint16_t)((nodeState[ni].avgLatency + latency) / 2);
        strncpy(nodeState[ni].lastLabel, label, 11);

        // Send result to PC/Web
        Serial.printf(
            "{\"type\":\"result\",\"req_id\":\"%s\",\"label\":\"%s\","
            "\"confidence\":%.2f,\"node\":\"c3-%u\",\"latency_ms\":%u}\n",
            req->reqId, label, (double)confidence, (unsigned)ni,
            (unsigned)latency);

        // Log to SD
        logInferToSD(req->reqId, label, confidence, ni, latency);

        totalInfers++;
        req->active = false;
        setLedOk();

      } else if (elapsed > INFER_TIMEOUT_MS) {
        // Timeout
        nodeState[req->assignedNode].totalErrors++;
        nodeState[req->assignedNode].alive = false;

        Serial.printf("{\"type\":\"error\",\"req_id\":\"%s\",\"code\":\"NODE_"
                      "TIMEOUT\",\"message\":\"worker c3-%u no respondió\"}\n",
                      req->reqId, (unsigned)req->assignedNode);

        totalErrors++;
        req->active = false;
        setLedErr();
      }
    }
  }
}

// ============================================================================
// SD LOGGING
// ============================================================================
static void logInferToSD(const char *reqId, const char *label, float confidence,
                         uint8_t node, uint16_t latency) {
  if (!sdOK)
    return;

  File f = SD.open(SD_FILENAME, FILE_APPEND);
  if (!f) {
    sdOK = false;
    return;
  }

  char ts[32];
  getTimestamp(ts, sizeof(ts));

  f.printf("%s,%s,%s,%.2f,c3-%u,%u,%.1f,%.1f\n", ts, reqId, label,
           (double)confidence, (unsigned)node, (unsigned)latency,
           (double)currentTemp, (double)currentHum);
  f.close();
}

// ============================================================================
// PING WORKERS
// ============================================================================
static void pingWorkers() {
  for (uint8_t i = 0; i < NUM_SLAVES; i++) {
    // Simple I2C ping: just try to address the slave
    Wire.beginTransmission(SLAVE_ADDRESSES[i]);
    uint8_t err = Wire.endTransmission();
    nodeState[i].lastPingMs = millis();

    if (err == 0) {
      nodeState[i].alive = true;
      nodeState[i].lastSeenMs = millis();
    } else {
      // Only mark dead after consecutive failures
      if (millis() - nodeState[i].lastSeenMs > PING_PERIOD_MS * 3) {
        nodeState[i].alive = false;
      }
    }
  }
}

// ============================================================================
// OLED DISPLAY
// ============================================================================
static void drawOLED() {
  if (!oledOK)
    return;

  display.clearBuffer();
  display.setFont(u8g2_font_6x10_tr);

  if (displayPage == 0) {
    // Page 0: Overview
    display.drawStr(0, 10, "CuantumWiki");

    display.setFont(u8g2_font_4x6_tr);
    char line[32];
    snprintf(line, sizeof(line), "Infers:%lu Err:%lu",
             (unsigned long)totalInfers, (unsigned long)totalErrors);
    display.drawStr(0, 22, line);

    // Node status
    for (uint8_t i = 0; i < NUM_SLAVES && i < 4; i++) {
      int x = i * 32;
      int y = 30;
      snprintf(line, sizeof(line), "C%u:%s", (unsigned)i,
               nodeState[i].alive ? "OK" : "X");
      display.drawStr(x, y, line);

      if (nodeState[i].lastLabel[0]) {
        display.drawStr(x, y + 8, nodeState[i].lastLabel);
      }
    }

    // Queue usage
    uint8_t queueUsed = 0;
    for (uint8_t i = 0; i < MAX_QUEUE; i++) {
      if (requestQueue[i].active)
        queueUsed++;
    }
    snprintf(line, sizeof(line), "Q:%u/%u", (unsigned)queueUsed,
             (unsigned)MAX_QUEUE);
    display.drawStr(0, 52, line);

    // Time
    char timeStr[16];
    if (rtcOK) {
      DateTime now = rtc.now();
      snprintf(timeStr, sizeof(timeStr), "%02d:%02d:%02d", now.hour(),
               now.minute(), now.second());
    } else {
      snprintf(timeStr, sizeof(timeStr), "--:--:--");
    }
    display.drawStr(75, 62, timeStr);

    // Env
    if (bmeOK) {
      snprintf(line, sizeof(line), "%.1fC %.0f%%", (double)currentTemp,
               (double)currentHum);
      display.drawStr(0, 62, line);
    }

  } else {
    // Page 1: Node details
    display.drawStr(0, 10, "Nodes Detail");
    display.setFont(u8g2_font_4x6_tr);

    for (uint8_t i = 0; i < NUM_SLAVES && i < 4; i++) {
      int y = 20 + i * 12;
      char line[48];
      snprintf(line, sizeof(line), "C%u %s inf:%u err:%u avg:%ums", (unsigned)i,
               nodeState[i].alive ? "ON " : "OFF",
               (unsigned)nodeState[i].totalInfers,
               (unsigned)nodeState[i].totalErrors,
               (unsigned)nodeState[i].avgLatency);
      display.drawStr(0, y, line);
    }
  }

  display.sendBuffer();
}

// ============================================================================
// BUTTONS
// ============================================================================
static void buttonsStep(uint32_t nowMs) {
  if (digitalRead(BTNA_PIN) == LOW) {
    if (nowMs - lastBtnAMs >= BTN_DEBOUNCE_MS) {
      lastBtnAMs = nowMs;
      displayPage = (displayPage + 1) % 2;
    }
  }
}

// ============================================================================
// HEALTH REPORT (to serial log)
// ============================================================================
static void printHealth() {
  Serial.printf(
      "{\"type\":\"health\",\"uptime_ms\":%lu,\"infers\":%lu,\"errors\":%lu,"
      "\"i2c\":%s,\"oled\":%s,\"rtc\":%s,\"sd\":%s,\"bme\":%s}\n",
      (unsigned long)millis(), (unsigned long)totalInfers,
      (unsigned long)totalErrors, i2cOK ? "true" : "false",
      oledOK ? "true" : "false", rtcOK ? "true" : "false",
      sdOK ? "true" : "false", bmeOK ? "true" : "false");
}

// ============================================================================
// SETUP
// ============================================================================
void setup() {
  Serial.begin(115200);
  Serial.println(
      "{\"type\":\"boot\",\"fw\":\"CuantumCoordinator\",\"version\":\"1.0\"}");

  pinMode(BTNA_PIN, INPUT_PULLUP);
  pinMode(BTNB_PIN, INPUT_PULLUP);

  rgbLed.begin();
  rgbLed.setBrightness(50);
  rgbLed.show();
  setLedIdle();

  // Init all (never abort)
  initI2C();
  initOLED();
  initRTC();
  initSD();
  initBME();

  lastDisplayMs = millis();
  lastHealthMs = millis();
  lastPingMs = millis();
  serialResetLine();

  Serial.println(
      "{\"type\":\"ready\",\"message\":\"CuantumWiki Coordinator listo\"}");
}

// ============================================================================
// LOOP
// ============================================================================
void loop() {
  uint32_t nowMs = millis();

  // 1. Read serial commands from Web/PC
  serialStep();

  // 2. Process inference queue
  processQueue();

  // 3. Buttons
  buttonsStep(nowMs);

  // 4. Periodic ping workers
  if (nowMs - lastPingMs >= PING_PERIOD_MS) {
    lastPingMs = nowMs;
    pingWorkers();
  }

  // 5. Update display
  if (oledOK && (nowMs - lastDisplayMs >= DISPLAY_PERIOD_MS)) {
    lastDisplayMs = nowMs;
    drawOLED();
  }

  // 6. Health report
  if (nowMs - lastHealthMs >= HEALTH_PERIOD_MS) {
    lastHealthMs = nowMs;
    if (bmeOK) {
      currentTemp = bme.readTemperature();
      currentHum = bme.readHumidity();
    }
    printHealth();
  }

  // 7. Retry failed inits
  if (!sdOK && (nowMs - lastSdRetryMs >= SD_RETRY_PERIOD_MS)) {
    lastSdRetryMs = nowMs;
    initSD();
  }
  if (!bmeOK && (nowMs - lastBmeRetryMs >= BME_RETRY_PERIOD_MS)) {
    lastBmeRetryMs = nowMs;
    initBME();
  }
  if (!rtcOK && (nowMs - lastRtcRetryMs >= RTC_RETRY_PERIOD_MS)) {
    lastRtcRetryMs = nowMs;
    initRTC();
  }
  if (!oledOK && (nowMs - lastOledRetryMs >= OLED_RETRY_PERIOD_MS)) {
    lastOledRetryMs = nowMs;
    initOLED();
  }

  // LED idle if no active requests
  bool anyActive = false;
  for (uint8_t i = 0; i < MAX_QUEUE; i++) {
    if (requestQueue[i].active) {
      anyActive = true;
      break;
    }
  }
  if (!anyActive)
    setLedIdle();
}
