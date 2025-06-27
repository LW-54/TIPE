#include <Wire.h>
#include <Adafruit_MCP4725.h>
#include <ArduinoJson.h>

#define Vref      5.0
// I2C addresses for the two TCA9548A multiplexers
#define MUX_ADDR1 0x70
#define MUX_ADDR2 0x71

// Number of DACs per multiplexer
#define DAC_PER_MUX 7 //6
// Total number of DACs (flat array size)
#define TOTAL_DACS (DAC_PER_MUX * 2)

// Flat array of 14 DAC objects (indices 0–13)
Adafruit_MCP4725 dac[TOTAL_DACS];

/**
 * @brief Selects a channel on a TCA9548A I2C multiplexer.
 * 
 * @param muxAddr The I2C address of the multiplexer.
 * @param channel The channel to select (0-7).
 */
void tcaselect(uint8_t muxAddr, uint8_t channel) {
  if (channel > 7) return;          // Invalid channel
  Wire.beginTransmission(muxAddr);
  Wire.write(1 << channel);
  Wire.endTransmission();
}

//=========================
// Pin Mapping Functions
//=========================

struct InputMap { uint8_t index, channel; };
InputMap input_map[] = {{0,0},{1,1},{2,2},{3,3},{4,4},{5,5},{6,6},{7,7},{8,8},{9,9},{10,10},{11,11},{12,12},{13,13}};
/**
 * @brief Gets the DAC channel for a given input.
 * 
 * @param idx The index of the input.
 * @return The DAC channel for the input, or 255 if not found.
 */
uint8_t I(uint8_t idx) {
    for (auto &i : input_map) if (i.index==idx) return i.channel;
    return 255;
}

struct OutputMap { uint8_t index, channel; };
OutputMap output_map[] = {{0,0},{1,1},{2,2},{3,3},{4,4},{5,5}};
/**
 * @brief Gets the ADC channel for a given output.
 * 
 * @param idx The index of the output.
 * @return The ADC channel for the output, or 255 if not found.
 */
uint8_t O(uint8_t idx) {
    for (auto &o : output_map) if (o.index==idx) return o.channel;
    return 255;
}

//=========================
// DAC Write Function
//=========================
/**
 * @brief Sets the voltage of a DAC.
 * 
 * @param index The index of the DAC to set.
 * @param voltage The voltage to set the DAC to.
 * @return True if the voltage was set successfully, false otherwise.
 */
bool setVoltage(uint8_t index, float voltage) {
    if (index >= TOTAL_DACS) return false;
    if (voltage < 0.0 || voltage > Vref) return false;
    uint16_t value = (voltage / Vref) * 4095;
    // Determine multiplexer and channel from index
    uint8_t muxAddr;
    uint8_t channel;
    if (index < DAC_PER_MUX) {
    muxAddr = MUX_ADDR1;          // Use first multiplexer
    channel = index;              // Channels 0–6
    } else {
    muxAddr = MUX_ADDR2;          // Use second multiplexer
    channel = index - DAC_PER_MUX; // Channels 0–6 for indices 7–13
    }

    // Select the appropriate channel and set DAC output
    tcaselect(muxAddr, channel);
    dac[index].setVoltage(value, false);  // Write voltage to the DAC

    // Deselect (disable) all channels on this multiplexer by writing 0
    Wire.beginTransmission(muxAddr);
    Wire.write(0);
    Wire.endTransmission();
    return true;
}

//=========================
// Read Outputs
//=========================
/**
 * @brief Reads all the outputs from the ADC and adds them to a JSON array.
 * 
 * @param outer The JSON array to add the outputs to.
 */
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

    // Initialize 7 DACs on first multiplexer (address 0x70)
    for (uint8_t ch = 0; ch < DAC_PER_MUX; ch++) {
        tcaselect(MUX_ADDR1, ch);        // Enable channel ch on first MUX
        if (!dac[ch].begin(0x62)) {
            ok = false;
            DynamicJsonDocument err(64);
            err["error"] = "DAC init failed on channel " + String(ch) + " MUX " + String(MUX_ADDR1);
            serializeJson(err, Serial);
            Serial.println();
        }           // Default address for MCP4725 (A0=GND)
    }
    // Initialize 7 DACs on second multiplexer (address 0x71)
    for (uint8_t ch = 0; ch < DAC_PER_MUX; ch++) {
        tcaselect(MUX_ADDR2, ch);        // Enable channel ch on second MUX
        if (!dac[DAC_PER_MUX + ch].begin(0x62)) {
            ok = false;
            DynamicJsonDocument err(64);
            err["error"] = "DAC init failed on channel " + String(DAC_PER_MUX + ch) + " MUX " + String(MUX_ADDR2);
            serializeJson(err, Serial);
            Serial.println();
        }           // Default address for MCP4725 (A0=GND)
    }
    
    if (!ok) {
        while (1); // Halt if any DAC failed
    }

    // Deselect all channels on both multiplexers (write 0 to disable all)
    Wire.beginTransmission(MUX_ADDR1);
    Wire.write(0);
    Wire.endTransmission();
    Wire.beginTransmission(MUX_ADDR2);
    Wire.write(0);
    Wire.endTransmission();

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
/**
 * @brief Processes a JSON command received over serial.
 * 
 * @param json The JSON string to process.
 */
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
        } else {r["error"]="Invalid output index"; r["received"] = json;}
        serializeJson(r,Serial);Serial.println();
        return;
    }

    else {
        DynamicJsonDocument r(64); r["error"]="Unknown command"; r["received"] = json;
        serializeJson(r,Serial); Serial.println();
    }
}