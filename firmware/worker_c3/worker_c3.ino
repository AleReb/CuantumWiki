/*
 * cuantum_worker_c3.ino — CuantumWiki Tone Detection Worker
 *
 * ESP32-C3 — I2C Slave + Manual Neural Network Inference
 *
 * Architecture:
 *   S3 Coordinator -> I2C -> This C3 Worker -> inference -> I2C response
 *
 * Model: Embedding(595,16) → GlobalAvgPool → Dense(32,relu) → Dense(5,softmax)
 * Weights loaded from model_weights.h (exported by train_tone_model.py)
 * NO TFLite dependency — pure C forward pass!
 *
 * I2C Protocol:
 *   RX from S3:  "I,<req_id>,<num_tokens>\n" + binary token data (int16 LE)
 *   TX to S3:    "R,<req_id>,<label_idx>,<confidence_x100>,<latency_ms>\n"
 *
 * Labels: 0=neutral, 1=positivo, 2=urgente, 3=molesto, 4=formal
 *
 * IMPORTANT: Change SLAVE_ADDRESS for each node!
 *   Node 0: 0x10   Node 1: 0x11   Node 2: 0x12   Node 3: 0x13
 */

#include "model_weights.h" // Auto-generated: emb_weights, dense1/2 weights+bias
#include <Arduino.h>
#include <Wire.h>
#include <math.h>


// ========================= CONFIG =========================
// *** CHANGE THIS PER NODE *** 10 11 12 13
#define SLAVE_ADDRESS 0x13

static constexpr int LED_PIN = 7;
static constexpr uint8_t I2C_MAX_RX = 32;
static constexpr uint32_t HEALTH_PRINT_MS = 3000;
static constexpr uint16_t LED_PULSE_MS = 30;
static constexpr uint8_t MAX_TOKENS = 128;

static const char *LABELS[] = {"neutral", "positivo", "urgente", "molesto",
                               "formal"};

// ========================= STATE ==========================
static bool i2cOK = false;
static bool modelLoaded = false;
static volatile bool ledTriggered = false;

// I2C receive state machine
static volatile uint8_t rxPhase = 0;
static char reqId[16] = "";
static uint8_t expectedTokens = 0;
static uint8_t receivedTokens = 0;
static int16_t tokenBuffer[MAX_TOKENS];

static volatile bool cmdReady = false;
static char rxBuf[I2C_MAX_RX + 1];
static volatile uint8_t rxLen = 0;
static char txBuf[32];

static bool inferenceReady = false;
static uint32_t inferStartMs = 0;

static uint8_t resultLabelIdx = 0;
static uint16_t resultConfX100 = 0;
static uint16_t resultLatencyMs = 0;
static char resultReqId[16] = "";
static bool resultAvailable = false;

static uint32_t totalInfers = 0;
static uint32_t totalErrors = 0;
static uint32_t lastHealthMs = 0;
static uint32_t lastI2CRetryMs = 0;
static uint32_t ledOffAtMs = 0;

// ========================= LED ============================
static void pulseLed() {
  digitalWrite(LED_PIN, HIGH);
  ledOffAtMs = millis() + LED_PULSE_MS;
}

// ========================= NEURAL NET FORWARD PASS ========
/*
 * Manual implementation of:
 *   1. Embedding lookup → (numTokens, EMBED_DIM)
 *   2. GlobalAveragePooling1D → (EMBED_DIM,)
 *   3. Dense(HIDDEN_DIM, relu) → (HIDDEN_DIM,)
 *   4. Dense(NUM_LABELS, softmax) → (NUM_LABELS,)
 *
 * All weights come from model_weights.h
 */

static void initModel() {
  modelLoaded = true;
  Serial.printf(
      "[MODEL] Manual NN loaded (vocab=%d, emb=%d, hidden=%d, labels=%d)\n",
      MODEL_VOCAB_SIZE, MODEL_EMBED_DIM, MODEL_HIDDEN_DIM, MODEL_NUM_LABELS);
}

static void runInference(const int16_t *tokens, uint8_t numTokens,
                         uint8_t *outLabel, uint16_t *outConf) {

  // ── Step 1: Embedding + GlobalAveragePooling ──
  // Average all token embeddings into a single vector
  float pooled[MODEL_EMBED_DIM];
  memset(pooled, 0, sizeof(pooled));

  uint8_t validCount = 0;
  for (uint8_t t = 0; t < numTokens; t++) {
    int16_t tok = tokens[t];
    if (tok <= 0 || tok >= MODEL_VOCAB_SIZE)
      continue; // Skip PAD and OOV

    for (uint8_t d = 0; d < MODEL_EMBED_DIM; d++) {
      pooled[d] += emb_weights[tok][d];
    }
    validCount++;
  }

  // Average
  if (validCount > 0) {
    float inv = 1.0f / (float)validCount;
    for (uint8_t d = 0; d < MODEL_EMBED_DIM; d++) {
      pooled[d] *= inv;
    }
  }

  // ── Step 2: Dense1 (EMBED_DIM → HIDDEN_DIM, ReLU) ──
  float hidden[MODEL_HIDDEN_DIM];
  for (uint8_t h = 0; h < MODEL_HIDDEN_DIM; h++) {
    float sum = dense1_bias[h];
    for (uint8_t d = 0; d < MODEL_EMBED_DIM; d++) {
      sum += pooled[d] * dense1_weights[d][h];
    }
    // ReLU
    hidden[h] = (sum > 0.0f) ? sum : 0.0f;
  }

  // ── Step 3: Dense2 (HIDDEN_DIM → NUM_LABELS, Softmax) ──
  float logits[MODEL_NUM_LABELS];
  float maxLogit = -1e9f;
  for (uint8_t l = 0; l < MODEL_NUM_LABELS; l++) {
    float sum = dense2_bias[l];
    for (uint8_t h = 0; h < MODEL_HIDDEN_DIM; h++) {
      sum += hidden[h] * dense2_weights[h][l];
    }
    logits[l] = sum;
    if (sum > maxLogit)
      maxLogit = sum;
  }

  // Softmax (numerically stable)
  float expSum = 0.0f;
  float probs[MODEL_NUM_LABELS];
  for (uint8_t l = 0; l < MODEL_NUM_LABELS; l++) {
    probs[l] = expf(logits[l] - maxLogit);
    expSum += probs[l];
  }

  // Find argmax and confidence
  uint8_t bestIdx = 0;
  float bestProb = 0.0f;
  for (uint8_t l = 0; l < MODEL_NUM_LABELS; l++) {
    probs[l] /= expSum;
    if (probs[l] > bestProb) {
      bestProb = probs[l];
      bestIdx = l;
    }
  }

  *outLabel = bestIdx;
  *outConf = (uint16_t)(bestProb * 100.0f + 0.5f);
  if (*outConf > 99)
    *outConf = 99;
  if (*outConf < 1)
    *outConf = 1;
}

// ========================= I2C CALLBACKS ==================
static void onReceiveEvent(int howMany) {
  uint8_t i = 0;

  if (rxPhase == 0 || rxPhase == 1) {
    while (Wire.available() && i < I2C_MAX_RX) {
      rxBuf[i++] = (char)Wire.read();
    }
    rxBuf[i] = '\0';
    rxLen = i;
    cmdReady = true;
  } else if (rxPhase == 2) {
    while (Wire.available() >= 2 && receivedTokens < expectedTokens) {
      uint8_t lo = Wire.read();
      uint8_t hi = Wire.read();
      tokenBuffer[receivedTokens++] = (int16_t)((hi << 8) | lo);
    }
    while (Wire.available())
      Wire.read();

    if (receivedTokens >= expectedTokens) {
      inferenceReady = true;
      rxPhase = 0;
    }
  }

  ledTriggered = true;
}

static void onRequestEvent() {
  Wire.write((const uint8_t *)txBuf, (uint8_t)strnlen(txBuf, sizeof(txBuf)));
  ledTriggered = true;
}

// ========================= PARSERS ========================
static bool parseInferHeader(const char *s) {
  char prefix = 0;
  char rid[16] = "";
  unsigned int nt = 0;

  int parsed = sscanf(s, "%c,%[^,],%u", &prefix, rid, &nt);
  if (parsed < 3 || prefix != 'I')
    return false;

  strncpy(reqId, rid, 15);
  reqId[15] = '\0';
  expectedTokens = (uint8_t)(nt > MAX_TOKENS ? MAX_TOKENS : nt);
  receivedTokens = 0;
  rxPhase = 2;
  return true;
}

// ========================= RESPONSE =======================
static void buildResponse() {
  snprintf(txBuf, sizeof(txBuf), "R,%s,%u,%u,%u\n", resultReqId,
           (unsigned)resultLabelIdx, (unsigned)resultConfX100,
           (unsigned)resultLatencyMs);
}

static void buildIdleResponse() {
  snprintf(txBuf, sizeof(txBuf), "P,OK,%lu\n", (unsigned long)millis());
}

// ========================= I2C INIT =======================
static void initI2C() {
  if (i2cOK)
    return;
  if (millis() - lastI2CRetryMs < 2000)
    return;
  lastI2CRetryMs = millis();

  Wire.begin(SLAVE_ADDRESS);
  Wire.onReceive(onReceiveEvent);
  Wire.onRequest(onRequestEvent);
  i2cOK = true;

  Serial.print("[INIT] I2C addr 0x");
  Serial.println(SLAVE_ADDRESS, HEX);
}

// ========================= SETUP ==========================
void setup() {
  Serial.begin(115200);
  Serial.println("\n=== CuantumWiki Worker C3 ===");
  Serial.print("I2C addr: 0x");
  Serial.println(SLAVE_ADDRESS, HEX);

  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  initI2C();
  initModel();
  buildIdleResponse();

  Serial.println("[READY]");
}

// ========================= LOOP ===========================
void loop() {
  uint32_t now = millis();

  if (ledTriggered) {
    ledTriggered = false;
    pulseLed();
  }

  if (digitalRead(LED_PIN) == HIGH && (int32_t)(now - ledOffAtMs) >= 0) {
    digitalWrite(LED_PIN, LOW);
  }

  initI2C();

  if (cmdReady) {
    cmdReady = false;
    if (parseInferHeader(rxBuf)) {
      inferStartMs = millis();
      resultAvailable = false;
      Serial.printf("[RX] Infer req=%s tokens=%u\n", reqId,
                    (unsigned)expectedTokens);
      if (expectedTokens == 0) {
        inferenceReady = true;
        rxPhase = 0;
      }
    } else {
      Serial.print("[RX] Unknown: ");
      Serial.println(rxBuf);
    }
  }

  if (inferenceReady && modelLoaded) {
    inferenceReady = false;

    Serial.printf("[INFER] Running on %u tokens...\n",
                  (unsigned)receivedTokens);
    uint32_t t0 = micros();

    uint8_t labelIdx = 0;
    uint16_t confX100 = 0;
    runInference(tokenBuffer, receivedTokens, &labelIdx, &confX100);

    uint32_t elapsed_us = micros() - t0;
    uint16_t latency = (uint16_t)(elapsed_us / 1000);
    if (latency == 0)
      latency = 1;

    resultLabelIdx = labelIdx;
    resultConfX100 = confX100;
    resultLatencyMs = latency;
    strncpy(resultReqId, reqId, 15);
    resultReqId[15] = '\0';
    resultAvailable = true;

    buildResponse();
    totalInfers++;

    Serial.printf("[RESULT] %s conf=%u%% latency=%ums\n",
                  LABELS[resultLabelIdx], (unsigned)resultConfX100,
                  (unsigned)resultLatencyMs);
  }

  if (now - lastHealthMs >= HEALTH_PRINT_MS) {
    lastHealthMs = now;
    Serial.printf(
        "[HEALTH] addr=0x%02X infers=%lu errors=%lu model=%s i2c=%u\n",
        SLAVE_ADDRESS, (unsigned long)totalInfers, (unsigned long)totalErrors,
        modelLoaded ? "loaded" : "none", (unsigned)i2cOK);
  }
}
