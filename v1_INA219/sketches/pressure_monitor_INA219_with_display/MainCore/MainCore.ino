/*
 * MainCore.ino  pressure_monitor_INA219_with_display
 *
 * MainCore responsibilities:
 *   Acquires motor current via TI INA219, performs FFT analysis and
 *   DNN inference for pressure estimation, then sends results to SubCore1
 *   for ILI9341 display rendering.
 *
 * Processing pipeline:
 *   INA219 (512 samples, I2C 400kHz)
 *   → DC mean + Peak-to-Peak + Hamming FFT (512-point)
 *   → DNN input: 257 dimensions [DC, P-P, FFT bin1–255]
 *   → DNNRT inference → Estimated pressure [MPa]
 *   → MP.Send to SubCore1 (normalized FFT spectrum + pressure + status)
 *
 * Sensor: TI INA219 (I2C address 0x40)
 * Model:  257 → 128 → 64 → 32 → 1 (R²=0.9992, MAE=0.0024 MPa)
 *
 * Inter-core communication:
 *   display_data[0–254] : Normalized FFT spectrum (0.0–1.0, 255 bins)
 *   display_data[255]   : Estimated pressure [MPa]
 *   msgid               : 100=STOP, 101=OK, 102=OVER PRESSURE
 *
 * Required files on SD card:
 *   model.nnb  … Trained DNN model (root of SD)
 *
 * Note: When uploading via Arduino IDE,
 *       select Tools → Core → MainCore.
 *
 * Version history:
 *   Ver 1.0: Initial release
 *   Ver 1.1: Added pump-stop detection; removed millis() output
 *   Ver 1.2: Fixed INA219 register setting (0x398F); added I2C 400kHz
 *   Ver 1.3: Added dual-core display support
 */

#ifdef SUBCORE
#error "Core selection is wrong!! Select MainCore."
#endif

#include <Wire.h>
#include <Adafruit_INA219.h>
#include <FFT.h>
#include <SDHCI.h>
#include <DNNRT.h>
#include <MP.h>
#include <MPMutex.h>

// ============================================
// Configuration parameters
// ============================================
#define FFT_LEN            512    // FFT sample count
#define PRESSURE_THRESHOLD 0.300  // Over-pressure threshold [MPa]
#define STOP_THRESHOLD     30.0   // Pump-stop detection threshold [mA]

// Measurement parameters — must match data collection settings exactly
const float SCALE_FACTOR      = 1500.0;  // AC coupling scale factor [mA]
const float CORRECTION_FACTOR = 1.50;    // Oscilloscope-calibrated correction factor

// Display parameters
#define DISPLAY_SAMPLES 255  // Number of FFT bins sent to SubCore1
#define MAX_AVG_COUNT   4    // Frame count for moving-average normalization

// ============================================
// Global objects
// ============================================
FFTClass<1, FFT_LEN> FFT;
SDClass SD;
DNNRT   dnnrt;
Adafruit_INA219 ina219;

// Inter-core communication
MPMutex    mutex(MP_MUTEX_ID0);
const int  subcore = 1;

// Buffers
q15_t fft_input[FFT_LEN];
float fft_output[FFT_LEN];
float raw_samples[FFT_LEN];

// Measurement results (shared across functions)
float g_dc_mean      = 0;
float g_peak_to_peak = 0;

// Zero-current offset (calibrated at startup with DC power OFF)
float ZERO_OFFSET = 0;

// Display data buffer: [0–254] = normalized FFT spectrum, [255] = pressure
static float display_data[DISPLAY_SAMPLES + 1];

// Message IDs for inter-core status communication
static const int8_t MSG_STOP = 100;  // Pump not running
static const int8_t MSG_OK   = 101;  // Normal operation
static const int8_t MSG_OVER = 102;  // Over-pressure detected


// ============================================
// Moving average of FFT maximum value
// Used to normalize the spectrum for stable display
// Averaging over MAX_AVG_COUNT frames prevents display flicker
// ============================================
float getMaxSpectrumAverage(float* spectrum, int len) {
  static float max_history[MAX_AVG_COUNT] = {0};
  static int   history_index = 0;

  // Find maximum value in current FFT spectrum
  float current_max = 0;
  for (int i = 0; i < len; i++) {
    if (spectrum[i] > current_max) {
      current_max = spectrum[i];
    }
  }

  // Store in circular buffer
  max_history[history_index] = current_max;
  history_index = (history_index + 1) % MAX_AVG_COUNT;

  // Return moving average
  float sum = 0;
  for (int i = 0; i < MAX_AVG_COUNT; i++) {
    sum += max_history[i];
  }

  return sum / MAX_AVG_COUNT;
}


// ============================================
// setup
// ============================================
void setup() {
  Serial.begin(115200);
  while (!Serial) { delay(100); }

  Serial.println("========================================");
  Serial.println("  Pressure Estimation System");
  Serial.println("  Sensor: TI INA219  |  Display: ILI9341");
  Serial.println("========================================");
  Serial.println();

  // SD card initialization
  Serial.print("SD card initialize...");
  while (!SD.begin()) {
    Serial.println("Insert SD card");
    delay(1000);
  }
  Serial.println("OK");

  // Load DNN model from SD card
  Serial.print("DNNRT initialize...");
  File nnbfile = SD.open("model.nnb");
  if (!nnbfile) {
    Serial.println("NG");
    Serial.println("Error: model.nnb not found");
    while (1);
  }

  int ret = dnnrt.begin(nnbfile);
  if (ret < 0) {
    Serial.println("NG");
    Serial.print("DNNRT begin failed: ");
    Serial.println(ret);
    while (1);
  }
  Serial.println("OK");

  // INA219 initialization
  Serial.print("INA219 initialize...");
  if (!ina219.begin()) {
    Serial.println("NG");
    Serial.println("Error: Failed to find INA219 chip");
    while (1);
  }

  Wire.setClock(400000);  // I2C 400kHz — mandatory for consistent sampling rate
  ina219.setCalibration_32V_2A();

  delay(10);

  // Configure INA219 for 10-bit ADC (fast sampling mode)
  // Register 0x00 value: 0x398F — must match training configuration
  uint16_t config = 0x2000 |  // Bus voltage range: 32V
                    0x1800 |  // PGA gain: 8, ±320mV
                    0x0080 |  // Bus ADC: 10-bit, 1 sample, 148us
                    0x0008 |  // Shunt ADC: 10-bit, 1 sample, 148us
                    0x0007;   // Continuous shunt and bus measurement

  Wire.beginTransmission(0x40);       // INA219 default I2C address
  Wire.write(0x00);                   // Config register
  Wire.write((config >> 8) & 0xFF);   // Upper byte: 0x39
  Wire.write(config & 0xFF);          // Lower byte: 0x8F
  Wire.endTransmission();

  delay(10);

  Serial.println("INA219 initialized (10-bit ADC, 400kHz I2C)");
  Serial.println();

  // FFT initialization: Hamming window, mono, 512 samples
  FFT.begin(WindowHamming, 1, FFT_LEN/2);
  Serial.println("FFT initialized (Hamming window, 512 samples)");
  Serial.println();

  // Zero offset calibration
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

  // Start SubCore1 (blocks until SubCore1 setup() completes)
  Serial.println("Starting SubCore1...");
  MP.begin(subcore);
  Serial.println("SubCore1 started");
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

  // Determine status (priority: Stop > Over Pressure > OK)
  int8_t msgid;
  if (g_dc_mean < STOP_THRESHOLD) {
    msgid = MSG_STOP;
  } else if (estimated_pressure >= PRESSURE_THRESHOLD) {
    msgid = MSG_OVER;
  } else {
    msgid = MSG_OK;
  }

  // Output result in CSV format
  Serial.print(g_dc_mean, 1);
  Serial.print(", ");
  Serial.print(g_peak_to_peak, 1);
  Serial.print(", ");
  Serial.print(estimated_pressure, 4);
  Serial.print(", ");

  if (msgid == MSG_STOP) {
    Serial.println("Stop");
  } else if (msgid == MSG_OVER) {
    Serial.println("OVER PRESSURE!");
  } else {
    Serial.println("OK");
  }

  // Prepare normalized display data for SubCore1
  prepareDisplayData(estimated_pressure);

  // Send to SubCore1 (skip if SubCore1 is busy rendering)
  int ret = mutex.Trylock();
  if (ret == 0) {
    ret = MP.Send(msgid, &display_data, subcore);
    if (ret < 0) {
      Serial.println("MP.Send Error");
    }
    mutex.Unlock();
  }
}


// ============================================
// Prepare normalized FFT spectrum for SubCore1 display
// Normalizes FFT spectrum to [0.0, 1.0] using moving-average maximum
// Stores pressure value at display_data[255]
// ============================================
void prepareDisplayData(float pressure) {
  // Compute moving-average maximum for stable normalization
  float max_spectrum = getMaxSpectrumAverage(fft_output, FFT_LEN/2);

  // Prevent division by zero
  if (max_spectrum < 1.0) max_spectrum = 1.0;

  // Normalize FFT spectrum: bin1–bin255 (255 bins)
  for (int i = 0; i < DISPLAY_SAMPLES; i++) {
    display_data[i] = fft_output[i + 1] / max_spectrum;

    // Clip to [0.0, 1.0]
    if (display_data[i] > 1.0f) display_data[i] = 1.0f;
    if (display_data[i] < 0.0f) display_data[i] = 0.0f;
  }

  // Store pressure value as the last element
  display_data[DISPLAY_SAMPLES] = pressure;
}


// ============================================
// Zero offset calibration
// Averages 200 INA219 readings with DC power OFF
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
  // IMPORTANT: pass raw (non-normalized) values — matches training data format
  DNNVariable input(257);
  float *dnnbuf = input.data();

  dnnbuf[0] = dc_mean;       // DC mean current [mA] — index 0
  dnnbuf[1] = peak_to_peak;  // Peak-to-peak amplitude [mA] — index 1

  // FFT spectrum: bin1–bin255 — indices 2 through 256
  // bin0 (DC component) is excluded — matches training data format
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
    float raw      = ina219.getCurrent_mA() - ZERO_OFFSET;
    raw_samples[i] = raw * CORRECTION_FACTOR;
  }
}


// ============================================
// Prepare FFT input: AC coupling + q15_t conversion
// Subtracts DC mean, normalizes to [-1.0, 1.0], clips, converts to q15_t
// ============================================
void prepareFFTData(float dc_mean) {
  for (int i = 0; i < FFT_LEN; i++) {
    float ac_value   = raw_samples[i] - dc_mean;
    float normalized = ac_value / SCALE_FACTOR;

    // Clip to valid range
    if (normalized >  1.0f) normalized =  1.0f;
    if (normalized < -1.0f) normalized = -1.0f;

    // Convert to q15_t format required by FFT library
    fft_input[i] = (q15_t)(normalized * 32767.0f);
  }
}
