/*
 * ina219_data_collector.ino  Ver 1.1
 *
 * FFT-based current data collector using TI INA219 current sensor.
 * Collects motor current samples, applies FFT analysis, and saves
 * the results to SD card in CSV format for DNN model training.
 *
 * Features:
 *   - Acquires 512 current samples from INA219 via I2C (400kHz)
 *   - Applies zero-offset calibration at startup
 *   - Computes DC mean, peak-to-peak, and Hamming-windowed FFT
 *   - Saves results as CSV: dc_mean, peak_to_peak, bin1, ..., bin255
 *   - Auto-collects SAMPLES_PER_SESSION files per key press
 *
 * Sensor: TI INA219 (I2C address 0x40)
 *   Config register: 0x398F
 *   ADC resolution: 10-bit (148us/sample) — required for fast sampling
 *   I2C clock: 400kHz (mandatory — do not change)
 *
 * CSV format:
 *   dc_mean, peak_to_peak, bin1, bin2, ..., bin255
 *   (257 columns total: 2 statistics + 255 FFT bins)
 *
 * Important: INA219 register settings (0x398F) must be identical
 *            between data collection and inference. Any change will
 *            degrade model accuracy significantly.
 */

#include <Wire.h>
#include <Adafruit_INA219.h>
#include <FFT.h>
#include <SDHCI.h>

// ============================================
// File naming configuration (change per measurement condition)
// Format: S{suction_pressure}D{discharge_pressure}
// Example: "S0.002D0.400" = suction 0.002 MPa, discharge 0.400 MPa
// ============================================
const char* FILE_PREFIX = "S0.002D0.400";
// ============================================

#define FFT_LEN             512   // FFT sample count
#define SAMPLES_PER_SESSION 100   // Files collected per key press
#define SAMPLE_INTERVAL_MS  100   // Wait time between samples (ms)

FFTClass<1, FFT_LEN> FFT;
SDClass SD;

Adafruit_INA219 ina219;
q15_t  fft_input[FFT_LEN];
float  fft_output[FFT_LEN];
float  raw_samples[FFT_LEN];

// Measurement parameters
// SCALE_FACTOR: normalizes AC-coupled current to [-1.0, 1.0] range for FFT input (q15_t)
const float SCALE_FACTOR      = 1500.0;  // AC coupling scale factor [mA]
// CORRECTION_FACTOR: compensates INA219 measurement offset verified against oscilloscope
const float CORRECTION_FACTOR = 1.50;    // Oscilloscope-calibrated correction factor
float       ZERO_OFFSET        = 0;      // Zero-current offset (calibrated at startup)

// INA219 register definitions
#define INA219_REG_CONFIG                       (0x00)
#define INA219_CONFIG_BVOLTAGERANGE_32V         (0x2000)  // Bus voltage range: 32V
#define INA219_CONFIG_GAIN_8_320MV              (0x1800)  // PGA gain: 8, ±320mV
#define INA219_CONFIG_BADCRES_10BIT_1S_148US    (0x0080)  // Bus ADC: 10-bit, 1 sample, 148us
#define INA219_CONFIG_SADCRES_10BIT_1S_148US    (0x0008)  // Shunt ADC: 10-bit, 1 sample, 148us
#define INA219_CONFIG_MODE_SANDBVOLT_CONTINUOUS (0x0007)  // Continuous shunt and bus measurement

// File sequence counter
int file_counter = 0;


void setup() {
  Serial.begin(115200);
  while (!Serial) {
    ;  // Wait for Serial connection
  }

  Serial.println("=== INA219 Data Collector ===");
  Serial.println("");

  // SD card initialization
  Serial.println("Initializing SD card...");
  while (!SD.begin()) {
    Serial.println("Insert SD card and press any key");
    while (!Serial.available());
    Serial.read();
  }
  Serial.println("SD card initialized successfully");
  Serial.println("");

  // INA219 initialization
  Serial.println("Initializing INA219...");
  ina219.begin();
  Wire.setClock(400000);  // I2C 400kHz — mandatory for consistent sampling rate
  ina219.setCalibration_32V_2A();

  delay(10);

  // Configure INA219 for 10-bit ADC (148us/sample — enables fast sampling)
  // Register 0x00 value: 0x398F
  // This setting must match exactly between data collection and inference
  uint16_t config = INA219_CONFIG_BVOLTAGERANGE_32V         |
                    INA219_CONFIG_GAIN_8_320MV              |
                    INA219_CONFIG_BADCRES_10BIT_1S_148US    |
                    INA219_CONFIG_SADCRES_10BIT_1S_148US    |
                    INA219_CONFIG_MODE_SANDBVOLT_CONTINUOUS;

  Wire.beginTransmission(0x40);          // INA219 default I2C address
  Wire.write(INA219_REG_CONFIG);         // Config register
  Wire.write((config >> 8) & 0xFF);      // Upper byte: 0x39
  Wire.write(config & 0xFF);             // Lower byte: 0x8F
  Wire.endTransmission();

  delay(10);

  Serial.println("INA219 initialized (10-bit ADC, 32V_2A, 400kHz I2C)");
  Serial.println("");

  // FFT initialization: Hamming window, mono, 512 samples
  FFT.begin(WindowHamming, 1, FFT_LEN/2);
  Serial.println("FFT initialized (Hamming window, 512 samples)");
  Serial.println("");

  // Zero offset calibration
  // Turn off DC power before this step to measure sensor baseline
  Serial.println("=== Zero Offset Calibration ===");
  Serial.println("Make sure DC power is OFF");
  Serial.println("Calibrating in 3 seconds...");
  delay(3000);

  calibrateZeroOffset();

  Serial.print("Zero offset: ");
  Serial.print(ZERO_OFFSET, 2);
  Serial.println(" mA");
  Serial.println("");

  // Print configuration summary
  Serial.println("=== Configuration ===");
  Serial.print("File prefix: ");
  Serial.println(FILE_PREFIX);
  Serial.print("Samples per session: ");
  Serial.println(SAMPLES_PER_SESSION);
  Serial.print("Sample interval: ");
  Serial.print(SAMPLE_INTERVAL_MS);
  Serial.println(" ms");
  Serial.print("Correction factor: ");
  Serial.println(CORRECTION_FACTOR);
  Serial.print("Scale factor: ");
  Serial.print(SCALE_FACTOR);
  Serial.println(" mA");
  Serial.println("");

  // Determine next available file number
  findNextFileNumber();
  Serial.print("Next file will be: ");
  Serial.print(FILE_PREFIX);
  Serial.print("_");
  Serial.print(file_counter, DEC);
  Serial.println(".csv");
  Serial.println("");

  Serial.println("=== Ready ===");
  Serial.println("Press any key to start collecting samples");
  Serial.println("");
}


void loop() {
  // Wait for key input
  if (Serial.available() > 0) {
    Serial.read();  // Consume input character

    Serial.println("===========================================");
    Serial.print("Starting data collection: ");
    Serial.print(SAMPLES_PER_SESSION);
    Serial.println(" samples");
    Serial.println("===========================================");
    Serial.println("");

    for (int i = 0; i < SAMPLES_PER_SESSION; i++) {
      // Print progress every 10 samples
      if (i % 10 == 0) {
        Serial.print("Progress: ");
        Serial.print(i);
        Serial.print(" / ");
        Serial.println(SAMPLES_PER_SESSION);
      }

      collectAndSaveData();

      // Wait between samples (skip wait after the last sample)
      if (i < SAMPLES_PER_SESSION - 1) {
        delay(SAMPLE_INTERVAL_MS);
      }
    }

    Serial.println("");
    Serial.println("===========================================");
    Serial.println("Data collection completed!");
    Serial.print("Saved ");
    Serial.print(SAMPLES_PER_SESSION);
    Serial.println(" samples");
    Serial.println("===========================================");
    Serial.println("");
    Serial.println("Press any key to collect another session");
    Serial.println("");
  }
}


// Calibrate zero offset by averaging 200 samples with DC power OFF
void calibrateZeroOffset() {
  float zero_sum = 0;
  for (int i = 0; i < 200; i++) {
    zero_sum += ina219.getCurrent_mA();
    delay(1);
  }
  ZERO_OFFSET = zero_sum / 200.0;
}


// Scan existing files to determine the next available sequence number
void findNextFileNumber() {
  char filename[32];
  file_counter = 1;

  while (true) {
    sprintf(filename, "%s_%03d.csv", FILE_PREFIX, file_counter);
    if (!SD.exists(filename)) {
      break;  // Found a filename that does not yet exist
    }
    file_counter++;

    // Safety limit to prevent infinite loop
    if (file_counter > 9999) {
      Serial.println("Warning: File counter exceeded 9999");
      break;
    }
  }
}


// Collect one session of data and save to SD card
void collectAndSaveData() {
  // Sample 512 readings from INA219 with correction factor applied
  for (int i = 0; i < FFT_LEN; i++) {
    float raw = ina219.getCurrent_mA() - ZERO_OFFSET;
    raw_samples[i] = raw * CORRECTION_FACTOR;
  }

  // Compute DC mean and peak-to-peak amplitude
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

  // AC coupling: subtract DC mean, normalize to [-1.0, 1.0], convert to q15_t for FFT
  for (int i = 0; i < FFT_LEN; i++) {
    float ac_value  = raw_samples[i] - dc_mean;
    float normalized = ac_value / SCALE_FACTOR;

    // Clip to valid range
    if (normalized >  1.0f) normalized =  1.0f;
    if (normalized < -1.0f) normalized = -1.0f;

    fft_input[i] = (q15_t)(normalized * 32767.0f);
  }

  // Run FFT (Hamming window applied internally)
  FFT.put(fft_input, FFT_LEN);
  FFT.get(fft_output, 0);

  // Save results to SD card
  saveDataToSD(dc_mean, peak_to_peak, fft_output);
}


// Save one sample to SD card as CSV
// Format: dc_mean, peak_to_peak, bin1, bin2, ..., bin255
// Note: bin0 (DC component) is excluded — matches DNN training data format
void saveDataToSD(float dc_mean, float peak_to_peak, float* spectrum) {
  char filename[32];
  sprintf(filename, "%s_%03d.csv", FILE_PREFIX, file_counter);

  // Open file in append mode
  File dataFile = SD.open(filename, FILE_WRITE);

  if (!dataFile) {
    Serial.print("Error: Cannot open file: ");
    Serial.println(filename);
    return;
  }

  // Write DC mean and peak-to-peak
  dataFile.print(dc_mean, 2);
  dataFile.print(",");
  dataFile.print(peak_to_peak, 2);

  // Write FFT spectrum: bin1 to bin255 (bin0 DC component excluded)
  for (int i = 1; i < FFT_LEN/2; i++) {
    dataFile.print(",");
    dataFile.print(spectrum[i], 2);
  }
  dataFile.println();

  dataFile.close();

  // Increment file counter for next sample
  file_counter++;
}
