// ANN_sim.ino - Arduino sketch simulating ANN behavior onboard
// Parses JSON commands over Serial and computes a forward pass

#include <ArduinoJson.h>
#include <math.h>

// === User configuration ===
const uint8_t num_layers = 3;               // total number of layers (including input & output)
const uint8_t layers[num_layers] = {2, 3, 1}; // layer sizes

// Maximum neurons per layer (for static allocation)
const uint8_t MAX_NEURONS = 3;
const uint8_t MAX_LAYERS  = 3;

// === Activation function selection ===
float relu(float x) { return x > 0 ? x : 0; }
float tanh_act(float x) { return tanh(x); }
float linear(float x) { return x; }

// Assign activation functions to aliases
#define HIDDEN_ACTIVATION tanh_act
#define OUTPUT_ACTIVATION linear

// === Data storage ===
float W[MAX_LAYERS][MAX_NEURONS][MAX_NEURONS];
float b[MAX_LAYERS][MAX_NEURONS];
float I_vec[MAX_NEURONS];

// Forward pass
void forward_once(const float* X, float* Y) {
  static float A[MAX_NEURONS], Z[MAX_NEURONS];
  for (uint8_t i = 0; i < layers[0]; i++) A[i] = X[i];

  for (uint8_t L = 1; L < num_layers; L++) {
    uint8_t prev_n = layers[L-1], cur_n = layers[L];
    for (uint8_t j = 0; j < cur_n; j++) {
      float sum = b[L][j];
      for (uint8_t i = 0; i < prev_n; i++) sum += A[i] * W[L][i][j];
      Z[j] = (L < num_layers) ? HIDDEN_ACTIVATION(sum) : OUTPUT_ACTIVATION(sum);
    }
    for (uint8_t j = 0; j < cur_n; j++) A[j] = Z[j];
  }
  for (uint8_t j = 0; j < layers[num_layers-1]; j++) Y[j] = A[j];
}

const int LED_PIN = LED_BUILTIN;

bool is_ready = false;

void setup() {
  Serial.begin(9600);
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);
}

void loop() {
    if (!is_ready && Serial.available()) {
    String str = Serial.readStringUntil('\n');
    DynamicJsonDocument hd(256);
    if (deserializeJson(hd, str) == DeserializationError::Ok) {
      const char* c = hd["cmd"];
      if (c && String(c) == "handshake") {
        Serial.println("{\"status\":\"ready\"}");
        while (Serial.available()) Serial.read();
        is_ready = true;
        digitalWrite(LED_PIN, HIGH); 
      }
    }
    return;
  }
    if (is_ready && Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    processJson(cmd);
  }
}

void processJson(const String &json) {
  DynamicJsonDocument doc(4096);
  auto err = deserializeJson(doc, json);
  if (err) {
      DynamicJsonDocument r(64); 
      r["error"]="JSON parse error"; 
      r["received"] = json;
      serializeJson(r, Serial); Serial.println(); return;
  }
  const char* cmd = doc["cmd"];
  if (!cmd) {
      DynamicJsonDocument r(64); 
      r["error"]="No cmd specified";
      r["received"] = json;
      serializeJson(r, Serial); Serial.println(); return;
  }
  String cmdType(cmd);

  if (cmdType == "set_weights_and_biases") {
    JsonObject Wobj = doc["W"].as<JsonObject>();
    JsonObject bobj = doc["b"].as<JsonObject>();
    for (auto kv : Wobj) {
      uint8_t L = atoi(kv.key().c_str());
      if (L >= num_layers) continue;
      JsonArray rows = kv.value().as<JsonArray>();
      for (uint8_t i = 0; i < rows.size() && i < layers[L-1]; i++) {
        JsonArray cols = rows[i].as<JsonArray>();
        for (uint8_t j = 0; j < cols.size() && j < layers[L]; j++) {
          W[L][i][j] = cols[j].as<float>();
        }
      }
    }
    for (auto kv : bobj) {
      uint8_t L = atoi(kv.key().c_str());
      if (L >= num_layers) continue;
      JsonArray biases = kv.value().as<JsonArray>();
      for (uint8_t j = 0; j < biases.size() && j < layers[L]; j++) {
        b[L][j] = biases[j].as<float>();
      }
    }
    Serial.println("{\"status\":\"OK\"}");
  }

  else if (cmdType == "forward") {
    JsonArray inputs = doc["input"].as<JsonArray>();
    float X[MAX_NEURONS] = {0}, Y[MAX_NEURONS] = {0};
    for (uint8_t i = 0; i < inputs.size() && i < layers[0]; i++) {
      X[i] = inputs[0][i].as<float>(); // doess not support batching change the code to fit 
    }
    forward_once(X, Y);
    DynamicJsonDocument r(512);
    JsonArray out = r.createNestedArray("output");
    for (uint8_t j = 0; j < layers[num_layers-1]; j++) {
      out.add(Y[j]);
    }
    serializeJson(r, Serial);
    Serial.println();
  }

  else if (cmdType == "set") {
    const char* param = doc["param"];
    float value = doc["value"];
    bool ok = false;
    if (param[0] == 'W') {
      int L, R, C;
      if (sscanf(param+1, "%d:%d:%d", &L, &R, &C) == 3
          && L < num_layers && R < layers[L-1] && C < layers[L]) {
        W[L][R][C] = value;
        ok = true;
      }
    } else if (param[0] == 'b') {
      int L, C;
      if (sscanf(param+1, "%d:%d", &L, &C) == 2
          && L < num_layers && C < layers[L]) {
        b[L][C] = value;
        ok = true;
      }
    } else if (param[0] == 'I') {
      int idx;
      if (sscanf(param+1, "%d", &idx) == 1 && idx < layers[0]) {
        I_vec[idx] = value;
        ok = true;
      }
    }
    Serial.println(ok ? "{\"status\":\"OK\"}" : "{\"error\":\"Invalid param\"}");
  }

  else if (cmdType == "read") {
    int idx = doc["index"];
    float Y[MAX_NEURONS] = {0};
    forward_once(I_vec, Y);
    if (idx >= 0 && idx < layers[num_layers-1]) {
      DynamicJsonDocument r(64);
      r["value"] = Y[idx];
      serializeJson(r, Serial);
      Serial.println();
    } else {
      Serial.println("{\"error\":\"Invalid output index\"}");
    }
  }

  else {
      DynamicJsonDocument r(64); r["error"]="Unknown command"; r["received"] = json;
      serializeJson(r,Serial); Serial.println();
  }
}


