/*
 * pressure_monitor_INA219.ino  Ver 1.4
 *
 * Real-time pump pressure estimation using TI INA219 current sensor
 * and a Deep Neural Network deployed on Sony Spresense.
 *
 * Processing pipeline:
 *   INA219 (512 samples, I2C 400kHz)
 *   → DC mean + Peak-to-Peak + Hamming FFT (512-point)
 *   → DNN input: 257 dimensions [DC, P-P, FFT bin1–255]
 *   → DNNRT inference
 *   → Estimated pressure [MPa] → Serial Monitor output
 *
 * Sensor: TI INA219 (I2C address 0x40)
 * Model:  257 → 128 → 64 → 32 → 1 (R²=0.9992, MAE=0.0024 MPa)
 *
 * Required files on SD card:
 *   model.nnb  … Trained model exported from Neural Network Console
 *
 * Critical: The Neural Network was trained on raw (non-normalized) values.
 *           Do NOT normalize DNN inputs at inference time.
 *           INA219 register config (0x398F) must match training exactly.
 *
 * Version history:
 *   Ver 1.0: Initial release
 *   Ver 1.1: Added pump-stop detection
 *   Ver 1.2: Fixed INA219 register setting (0x398F); added I2C 400kHz
 *   Ver 1.3: Added display output (dual-core version)
 *   Ver 1.4: Optimized data collection; removed unused micros() calls
 */

#include <Wire.h>
#include <Adafruit_INA219.h>
#include <FFT.h>
#include <SDHCI.h>
#include <DNNRT.h>

// ============================================
// Configuration parameters
// ============================================
#define FFT_LEN            512    // FFT sample count
#define PRESSURE_THRESHOLD 0.300  // Over-pressure threshold [MPa]
#define STOP_THRESHOLD     30.0   // Pump-stop detection threshold [mA]

// Measurement parameters — must match data collection settings exactly
// SCALE_FACTOR: normalizes AC-coupled signal to [-1.0, 1.0] for q15_t conversion
const float SCALE_FACTOR      = 1500.0;  // AC coupling scale factor [mA]
// CORRECTION_FACTOR: compensates INA219 reading offset (verified by oscilloscope)
const float CORRECTION_FACTOR = 1.50;    // Oscilloscope-calibrated correction factor

// Critical: DNN was trained on raw values — do NOT normalize at inference time
// Incorrect:  dnnbuf[0] = normalize(dc_mean);   // Will degrade accuracy
// Correct:    dnnbuf[0] = dc_mean;               // Pass raw mA value directly

// ============================================
// Global objects
// ============================================
FFTClass<1, FFT_LEN> FFT;
SDClass SD;
DNNRT   dnnrt;

Adafruit_INA219 ina219;

// Buffers
q15_t fft_input[FFT_LEN];
float fft_output[FFT_LEN];
float raw_samples[FFT_LEN];

// Measurement results (shared across functions)
float g_dc_mean      = 0;
float g_peak_to_peak = 0;

// Zero-current offset (calibrated at startup with DC power OFF)
float ZERO_OFFSET = 0;

// INA219 register definitions
// Config register 0x00 target value: 0x398F
#define INA219_REG_CONFIG                       (0x00)
#define INA219_CONFIG_BVOLTAGERANGE_32V         (0x2000)  // Bus voltage range: 32V
#define INA219_CONFIG_GAIN_8_320MV              (0x1800)  // PGA gain: 8, ±320mV
#define INA219_CONFIG_BADCRES_10BIT_1S_148US    (0x0080)  // Bus ADC: 10-bit, 148us
#define INA219_CONFIG_SADCRES_10BIT_1S_148US    (0x0008)  // Shunt ADC: 10-bit, 148us
#define INA219_CONFIG_MODE_SANDBVOLT_CONTINUOUS (0x0007)  // Continuous measurement


// ============================================
// setup
// ============================================
void setup() {
  Serial.begin(115200);
  while (!Serial) {
    ;  // Wait for Serial connection
  }

  Serial.println("========================================");
  Serial.println("  Pump Pressure Estimation System");
  Serial.println("  Sensor: TI INA219  |  Model: DNN");
  Serial.println("========================================");
  Serial.println();

  // SD card initialization
  Serial.println("Initializing SD card...");
  while (!SD.begin()) {
    Serial.println("Insert SD card");
    delay(1000);
  }
  Serial.println("SD card OK");
  Serial.println();

  // Load DNN model from SD card
  Serial.println("Initializing DNNRT...");
  File nnbfile = SD.open("model.nnb");
  if (!nnbfile) {
    Serial.println("ERROR: model.nnb not found on SD card");
    while (1);
  }

  int ret = dnnrt.begin(nnbfile);
  if (ret < 0) {
    Serial.print("ERROR: DNNRT begin failed: ");
    Serial.println(ret);
    while (1);
  }
  Serial.println("DNNRT initialized");
  Serial.println();

  // INA219 initialization
  Serial.println("Initializing INA219...");
  if (!ina219.begin()) {
    Serial.println("ERROR: Failed to find INA219 chip");
    while (1);
  }

  Wire.setClock(400000);  // I2C 400kHz — mandatory for consistent sampling rate
  ina219.setCalibration_32V_2A();

  delay(10);

  // Configure INA219 for 10-bit ADC (fast sampling mode)
  // Target register value: 0x398F — must match training configuration
  uint16_t config = INA219_CONFIG_BVOLTAGERANGE_32V         |
                    INA219_CONFIG_GAIN_8_320MV              |
                    INA219_CONFIG_BADCRES_10BIT_1S_148US    |
                    INA219_CONFIG_SADCRES_10BIT_1S_148US    |
                    INA219_CONFIG_MODE_SANDBVOLT_CONTINUOUS;

  Wire.beginTransmission(0x40);          // INA219 default I2C address
  Wire.write(INA219_REG_CONFIG);
  Wire.write((config >> 8) & 0xFF);      // Upper byte: 0x39
  Wire.write(config & 0xFF);             // Lower byte: 0x8F
  Wire.endTransmission();

  delay(10);

  Serial.println("INA219 initialized (10-bit ADC, 400kHz I2C)");
  Serial.println();

  // FFT initialization: Hamming window, mono, 512 samples
  FFT.begin(WindowHamming, 1, FFT_LEN/2);
  Serial.println("FFT initialized (Hamming window, 512 samples)");
  Serial.println();

  // Zero offset calibration
  // Measures sensor baseline with DC power OFF to eliminate offset error
  Serial.println("=== Zero Offset Calibration ===");
  Serial.println("Make sure DC power is OFF");
  Serial.println("Calibrating in 3 seconds...");
  delay(3000);

  calibrateZeroOffset();

  Serial.print("Zero offset: ");
  Serial.print(ZERO_OFFSET, 2);
  Serial.println(" mA");
  Serial.println();

  // Print configuration summary
  Serial.println("=== Configuration ===");
  Serial.print("FFT length: ");
  Serial.println(FFT_LEN);
  Serial.print("DNN input dimensions: 257 (DC + P-P + FFT bin1-255)");
  Serial.println();
  Serial.print("Pressure threshold: ");
  Serial.print(PRESSURE_THRESHOLD, 3);
  Serial.println(" MPa");
  Serial.print("Stop threshold: ");
  Serial.print(STOP_THRESHOLD, 1);
  Serial.println(" mA");
  Serial.print("Correction factor: ");
  Serial.println(CORRECTION_FACTOR);
  Serial.println();

  Serial.println("========================================");
  Serial.println("  System Ready");
  Serial.println("  Starting pressure estimation...");
  Serial.println("========================================");
  Serial.println();

  // Print CSV header
  Serial.println("DC(mA), P-P(mA), Pressure(MPa), Status");
}


// ============================================
// loop
// ============================================
void loop() {
  // Run inference
  float estimated_pressure = estimatePressure();

  // Output result in CSV format
  Serial.print(g_dc_mean, 1);
  Serial.print(", ");
  Serial.print(g_peak_to_peak, 1);
  Serial.print(", ");
  Serial.print(estimated_pressure, 4);
  Serial.print(", ");

  // Status determination (priority: Stop > Over Pressure > OK)
  if (g_dc_mean < STOP_THRESHOLD) {
    Serial.println("Stop");           // Pump not running
  } else if (estimated_pressure >= PRESSURE_THRESHOLD) {
    Serial.println("OVER PRESSURE!"); // Pressure exceeds threshold
  } else {
    Serial.println("OK");
  }
}


// ============================================
// Zero offset calibration
// Average 200 INA219 readings with DC power OFF
// ============================================
void calibrateZeroOffset() {
  float zero_sum = 0;
  for (int i = 0; i < 200; i++) {
    zero_sum += ina219.getCurrent_mA();
    delay(1);
  }
  ZERO_OFFSET = zero_sum / 200.0;
}


// ============================================
// Main pressure estimation function
// Returns estimated discharge pressure [MPa]
// ============================================
float estimatePressure() {
  // Step 1: Acquire 512 samples from INA219
  sampleData();

  // Step 2: Compute DC mean and peak-to-peak amplitude
  float dc_mean = 0;
  float min_val = raw_samples[0];
  float max_val = raw_samples[0];

  for (int i = 0; i < FFT_LEN; i++) {
    dc_mean += raw_samples[i];
    if (raw_samples[i] < min_val) min_val = raw_samples[i];
    if (raw_samples[i] > max_val) max_val = raw_samples[i];
  }
  dc_mean /= FFT_LEN;

  float peak_to_peak = max_val - min_val;

  // Store in globals for access from loop()
  g_dc_mean      = dc_mean;
  g_peak_to_peak = peak_to_peak;

  // Step 3: AC coupling and FFT data preparation
  prepareFFTData(dc_mean);

  // Step 4: Run FFT
  FFT.put(fft_input, FFT_LEN);
  FFT.get(fft_output, 0);

  // Step 5: Prepare DNN input (257 dimensions)
  // Input layout: [0]=DC mean, [1]=Peak-to-Peak, [2-256]=FFT bin1-255
  // IMPORTANT: pass raw (non-normalized) values — matches training data format
  DNNVariable input(257);
  float *dnnbuf = input.data();

  dnnbuf[0] = dc_mean;       // DC mean current [mA] (raw)
  dnnbuf[1] = peak_to_peak;  // Peak-to-peak amplitude [mA] (raw)

  // FFT spectrum: bin1–bin255 (bin0 DC component excluded)
  for (int i = 1; i < FFT_LEN/2; i++) {
    dnnbuf[i + 1] = fft_output[i];
  }

  // Step 6: Run DNNRT inference
  dnnrt.inputVariable(input, 0);
  dnnrt.forward();
  DNNVariable output = dnnrt.outputVariable(0);

  // Step 7: Return estimated pressure [MPa]
  float estimated_pressure = output[0];

  return estimated_pressure;
}


// ============================================
// Acquire 512 current samples from INA219
// Applies zero-offset correction and oscilloscope correction factor
// ============================================
void sampleData() {
  for (int i = 0; i < FFT_LEN; i++) {
    float raw    = ina219.getCurrent_mA() - ZERO_OFFSET;
    raw_samples[i] = raw * CORRECTION_FACTOR;
  }
}


// ============================================
// Prepare FFT input: AC coupling + q15_t conversion
// Subtracts DC mean (AC coupling), normalizes to [-1.0, 1.0],
// clips to valid range, and converts to q15_t for FFT library
// ============================================
void prepareFFTData(float dc_mean) {
  for (int i = 0; i < FFT_LEN; i++) {
    float ac_value   = raw_samples[i] - dc_mean;
    float normalized = ac_value / SCALE_FACTOR;

    // Clip to valid q15_t input range
    if (normalized >  1.0f) normalized =  1.0f;
    if (normalized < -1.0f) normalized = -1.0f;

    // Convert to q15_t format required by FFT library
    fft_input[i] = (q15_t)(normalized * 32767.0f);
  }
}
