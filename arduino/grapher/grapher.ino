#include <Wire.h>
#include <Adafruit_MCP4725.h>
#include <ArduinoJson.h>

#define TCA9548A_ADDR 0x70   // I2C multiplexer address
#define NUM_DACS       5     // Number of DACs connected
#define Vref           5.0   // DAC/ADC reference voltage

Adafruit_MCP4725 dac[NUM_DACS];

// Selects the channel on the TCA9548A I2C multiplexer
void selectTCAChannel(uint8_t channel) {
    if (channel > 7) return;
    Wire.beginTransmission(TCA9548A_ADDR);
    Wire.write(1 << channel);
    Wire.endTransmission();
}

//=========================
// Pin Mapping Functions
//=========================

struct InputMap { uint8_t index, channel; };
InputMap input_map[] = {{0,0},{1,1},{2,2},{3,3},{5,5}};;
uint8_t I(uint8_t idx) {
    for (auto &i : input_map) if (i.index==idx) return i.channel;
    return 255;
}

struct OutputMap { uint8_t index, channel; };
OutputMap output_map[] = {{0,0},{1,1},{2,2},{3,3},{5,5}};
uint8_t O(uint8_t idx) {
    for (auto &o : output_map) if (o.index==idx) return o.channel;
    return 255;
}

//=========================
// DAC Write Function
//=========================
// Sets the specified DAC channel to a given voltage (0–Vref)
// Returns true if success, false if invalid channel or voltage out of range
bool setVoltage(uint8_t channel, float voltage) {
    if (channel >= NUM_DACS) return false;
    if (voltage < 0.0 || voltage > Vref) return false;
    uint16_t value = (voltage / Vref) * 4095;
    selectTCAChannel(channel);
    dac[channel].setVoltage(value, false);
    return true;
}

//=========================
// Read Outputs
//=========================
// Reads outputs in sorted order of index
void readAllOutputs(JsonArray &outer) {
    uint8_t max_index = 0;
    for (auto &o : output_map) if (o.index > max_index) max_index = o.index;
    for (uint8_t idx = 0; idx <= max_index; idx++) {
        uint8_t channel = O(idx);
        if (channel < 6) {
            int value = analogRead(channel);
            float voltage = value * Vref / 1023.0;
            outer.add(voltage);
        } else {
            outer.add(nullptr);
        }
    }
}

const int LED_PIN = LED_BUILTIN;

bool is_ready = false;

//=========================
// Setup
//=========================
void setup() {
    Serial.begin(9600);
    Wire.begin();
    bool ok = true;
    for (uint8_t i = 0; i < NUM_DACS; i++) {
        selectTCAChannel(i);
        if (!dac[i].begin(0x62)) {
            ok = false;
            DynamicJsonDocument err(64);
            err["error"] = "DAC init failed on channel " + String(i);
            serializeJson(err, Serial);
            Serial.println();
        }
    }
    if (!ok) {
        while (1); // Halt if any DAC failed
    }
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

//=========================
// JSON Processing
//=========================
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

        // Forward propagation: always batch (nested arrays)
    else if (strcmp(cmd, "graph")==0) {
        JsonArray batch = doc["input"].as<JsonArray>();
        DynamicJsonDocument r(4096);
        JsonArray outOuter = r.createNestedArray("output");
        
        // For each input vector in batch
        for (JsonVariant arrVar : batch) {
            JsonArray arr = arrVar.as<JsonArray>();
            // Set each input DAC
            for (size_t i = 0; i < arr.size(); i++) {
                float v = arr[i].as<float>();
                if (!setVoltage(I(i), v)) {
                    DynamicJsonDocument e(64);
                    e["error"] = "Invalid input";
                    e["received"] = json;
                    serializeJson(e, Serial);
                    Serial.println();
                    return;
                }
            }
            // Read outputs for this input
            JsonArray inner = outOuter.createNestedArray();
            readAllOutputs(inner);
        }
        // Send batched output
        serializeJson(r, Serial);
        Serial.println();
        return;
    }

    else if (strcmp(cmd, "set")==0) {
        const char* param = doc["param"];
        float value = doc["value"];
        bool ok2=false;
        if (param && param[0]=='I'){
            int idx;
            if (sscanf(param+1, "%d", &idx)==1)
                ok2=setVoltage(I(idx), value);
        }
        DynamicJsonDocument r(64);
        r[ok2?"status":"error"] = ok2?"OK":"Invalid param or value";
        serializeJson(r,Serial);Serial.println();
        return;
    }

    else if (strcmp(cmd, "read")==0) {
        int idx = doc["index"];
        uint8_t ch = O(idx);
        DynamicJsonDocument r(64);
        if (ch<6) {
            int v=analogRead(ch);
            r["value"] = v * Vref / 1023.0;
        } else r["error"]="Invalid output index"; 
        serializeJson(r,Serial);Serial.println();
        return;
    }

    else {
        DynamicJsonDocument r(64); r["error"]="Unknown command"; r["received"] = json;
        serializeJson(r,Serial); Serial.println();
    }
}
