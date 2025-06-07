#include <Wire.h>
#include <Adafruit_MCP4725.h>
#include <ArduinoJson.h>

#define Vref      5.0
#define BYTES     512
// I2C addresses for the two TCA9548A multiplexers
#define MUX_ADDR1 0x70
#define MUX_ADDR2 0x71

// Number of DACs per multiplexer
#define DAC_PER_MUX 6
// Total number of DACs (flat array size)
#define TOTAL_DACS (DAC_PER_MUX * 2)

// Flat array of 14 DAC objects (indices 0–13)
Adafruit_MCP4725 dac[TOTAL_DACS];

/**
 * Selects a channel (0–7) on the given TCA9548A.
 * Writes a byte with only that bit set to the multiplexer’s control register.
 * E.g., write (1 << channel) to select that channel:contentReference[oaicite:3]{index=3}.
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
struct Weight { uint8_t layer, row, col, channel; };
Weight weight_map[] = {
    {1,0,0,6},
    {1,0,1,9},
    {1,0,2,11},
    {2,0,0,4},
    {2,1,0,5},
    {2,2,0,3},
};
uint8_t W(uint8_t layer, uint8_t row, uint8_t col) {
    for (auto &w : weight_map) if (w.layer==layer && w.row==row && w.col==col) return w.channel;
    return 255;
}

struct Bias { uint8_t layer, row, channel; };
Bias bias_map[] = {
    {1,0,6},
    {1,1,8},
    {1,3,10},
    {2,0,1},
};
uint8_t b(uint8_t layer, uint8_t row) {
    for (auto &B : bias_map) if (B.layer==layer && B.row==row) return B.channel;
    return 255;
}

struct InputMap { uint8_t index, channel; };
InputMap input_map[] = {
    {0,2},
};
uint8_t I(uint8_t idx) {
    for (auto &i : input_map) if (i.index==idx) return i.channel;
    return 255;
}

struct OutputMap { uint8_t index, channel; };
OutputMap output_map[] = {{0,0},{1,1},{2,2},{3,3},{4,4},{5,5}};
uint8_t O(uint8_t idx) {
    for (auto &o : output_map) if (o.index==idx) return o.channel;
    return 255;
}

//=========================
// DAC Write Function
//=========================
// Sets the specified DAC channel to a given voltage (0–Vref)
// Returns true if success, false if invalid channel or voltage out of range
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
    Serial.begin(115200);
    Wire.begin();
    bool ok = true;

    // Initialize 7 DACs on first multiplexer (address 0x70)
    for (uint8_t ch = 0; ch < DAC_PER_MUX; ch++) {
        tcaselect(MUX_ADDR1, ch);        // Enable channel ch on first MUX
        if (!dac[ch].begin(0x62)) {
            ok = false;
            DynamicJsonDocument err(BYTES);
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
            DynamicJsonDocument err(BYTES);
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
    DynamicJsonDocument hd(BYTES);
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
    String json = Serial.readStringUntil('\n');
    processJson(json);
  }
}

//=========================
// JSON Processing
//=========================
void processJson(const String &json) {
    DynamicJsonDocument doc(BYTES);
    auto err = deserializeJson(doc, json);
    if (err) {
        DynamicJsonDocument r(BYTES); 
        r["error"]="JSON parse error"; 
        r["received"] = json.substring(0,64);
        serializeJson(r, Serial); Serial.println(); return;
    }
    const char* cmd = doc["cmd"];
    if (!cmd) {
        DynamicJsonDocument r(BYTES); 
        r["error"]="No cmd specified";
        r["received"] = json.substring(0,64);
        serializeJson(r, Serial); Serial.println(); return;
    }

    if (strcmp(cmd, "set_weights_and_biases")==0) {
        for (JsonPair kv : doc["W"].as<JsonObject>()) {
            int layer = atoi(kv.key().c_str());
            for (size_t i=0;i<kv.value().size();i++){
                for (size_t j=0;j<kv.value()[i].size();j++){
                    float val = kv.value()[i][j];
                    if (!setVoltage(W(layer,i,j), val + Vref/2.0)){
                        DynamicJsonDocument r(BYTES);
                        r["error"] = "Invalid W";
                        r["received"] = json.substring(0,64);
                        serializeJson(r,Serial);Serial.println(); return;
                    }
                }
            }
        }
        for (JsonPair kv : doc["b"].as<JsonObject>()) {
            int layer = atoi(kv.key().c_str());
            JsonArray arr = kv.value().as<JsonArray>();
            JsonArray vals = arr;
            if (arr.size()>0 && arr[0].is<JsonArray>()) vals = arr[0].as<JsonArray>();
            for (size_t i=0;i<vals.size();i++){
                float val = vals[i];
                if (!setVoltage(b(layer,i), val + Vref/2.0)){
                    DynamicJsonDocument r(BYTES);
                    r["error"] = "Invalid b";
                    r["received"] = json.substring(0,64);
                    serializeJson(r,Serial);Serial.println(); return;
                }
            }
        }
        DynamicJsonDocument r(BYTES); r["status"]="OK";
        serializeJson(r,Serial); Serial.println();
        return;
    }

        // Forward propagation: always batch (nested arrays)
    else if (strcmp(cmd, "forward")==0) {
        JsonArray batch = doc["input"].as<JsonArray>();
        DynamicJsonDocument r(BYTES);
        JsonArray outOuter = r.createNestedArray("output");
        
        // For each input vector in batch
        for (JsonVariant arrVar : batch) {
            JsonArray arr = arrVar.as<JsonArray>();
            // Set each input DAC
            for (size_t i = 0; i < arr.size(); i++) {
                float v = arr[i].as<float>();
                if (!setVoltage(I(i), v)) {
                    DynamicJsonDocument e(BYTES);
                    e["error"] = "Invalid input";
                    e["received"] = json.substring(0,64);
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
        if (param && param[0]=='W'){
            int L,R,C;
            if (sscanf(param+1, "%d:%d:%d", &L,&R,&C)==3)
                ok2=setVoltage(W(L,R,C), value);
        } else if (param && param[0]=='b'){
            int L,C;
            if (sscanf(param+1, "%d:%d", &L,&C)==2)
                ok2=setVoltage(b(L,C), value);
        } else if (param && param[0]=='I'){
            int idx;
            if (sscanf(param+1, "%d", &idx)==1)
                ok2=setVoltage(I(idx), value);
        }
        DynamicJsonDocument r(BYTES);
        r[ok2?"status":"error"] = ok2?"OK":"Invalid param or value";
        serializeJson(r,Serial);Serial.println();
        return;
    }

    else if (strcmp(cmd, "read")==0) {
        int idx = doc["index"];
        uint8_t ch = O(idx);
        DynamicJsonDocument r(BYTES);
        if (ch<6) {
            int v=analogRead(ch);
            r["value"] = v * Vref / 1023.0;
        } else r["error"]="Invalid output index"; 
        serializeJson(r,Serial);Serial.println();
        return;
    }

    else {
        DynamicJsonDocument r(BYTES); r["error"]="Unknown command"; r["received"] = json.substring(0,64);
        serializeJson(r,Serial); Serial.println();
    }
}



/*
Comands for attaching to WSL2
-----------------------------
usbipd list
usbipd bind --busid 1-2
usbipd attach --wsl --busid 1-2
lsusb
usbipd detach --busid 1-2
wsl --shutdown
*/

// move uses to applications folder
// get arduino setup to work again
// write ANN_sim.ino
// search for problem datasets that can be solved by extremly small Neural networks
// test sim on found problem or perceptron chien chat
// reajust circuit to current code and vice versa 
// tools for giving data the right shape for the circuit and problem
// tools in python and arduino for graphing different components 

