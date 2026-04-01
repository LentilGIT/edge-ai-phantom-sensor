/*
 * pcm_waveform_recorder.ino
 *
 * Records PCM waveform input from the MICA pin to SD card.
 *
 * Features:
 *   - Acquires 48kHz PCM data using Audio.h
 *   - Captures one frame per key press and saves to SD card as CSV
 *   - Output format: int16_t raw values, single-column CSV, no header
 *   - Filename: pcm_48k_001.csv (includes sampling rate in name)
 *   - Previews the first 16 samples on Serial Monitor after saving
 *
 * Connections:
 *   - MICA ← Voltage divider circuit (LT6105 VOUT → 10kΩ → MICA / 1kΩ → GND)
 *   - SD card required (Spresense extension board)
 *
 * How to change sampling rate:
 *   - Set SAMPLE_RATE to AS_SAMPLINGRATE_48000 or AS_SAMPLINGRATE_192000
 *   - Update FILE_PREFIX to match (e.g. "pcm_48k_" or "pcm_192k_")
 *   - For 192kHz, the SRC DSP file must be present in /mnt/sd0/BIN/ on the SD card
 *
 * Ver 1.0: Initial release
 */

#include <Audio.h>
#include <SDHCI.h>

// ============================================================
// Configuration (change sampling rate here)
// ============================================================
#define SAMPLE_RATE   AS_SAMPLINGRATE_48000   // 48kHz or 192kHz
#define FILE_PREFIX   "pcm_48k_"              // Output filename prefix

// ============================================================
// Constants (no changes needed)
// ============================================================
// Number of samples captured per trigger
// Audio.h processes internally in units of 768 samples; multiples of 768 are recommended.
// 1024 is not a multiple of 768 but is filled by repeated readFrames calls.
#define CAPTURE_SAMPLES  1024

// PCM uses int16_t (2 bytes per sample)
#define BYTES_PER_SAMPLE  sizeof(int16_t)
#define CAPTURE_BYTES     (CAPTURE_SAMPLES * BYTES_PER_SAMPLE)

// Number of samples to preview on Serial Monitor
#define PREVIEW_SAMPLES   16

// ============================================================
// Global variables
// ============================================================
SDClass  SD;
AudioClass *theAudio = AudioClass::getInstance();

// PCM data buffer (used as int16_t)
int16_t  pcm_buffer[CAPTURE_SAMPLES];

int file_counter = 1;  // File sequence number


// ============================================================
// setup
// ============================================================
void setup() {
  Serial.begin(115200);
  while (!Serial) { ; }

  Serial.println("========================================");
  Serial.println(" PCM Waveform Recorder  Ver 1.0");
  Serial.println("========================================");
  Serial.println("");

  // --- SD card initialization ---
  Serial.print("Initializing SD card... ");
  while (!SD.begin()) {
    Serial.println("NG  Insert SD card and press any key.");
    while (!Serial.available()) { ; }
    Serial.read();
  }
  Serial.println("OK");

  // --- Scan existing files to determine next sequence number ---
  findNextFileNumber();
  Serial.print("Next file: ");
  Serial.print(FILE_PREFIX);
  Serial.print(file_counter);
  Serial.println(".csv");
  Serial.println("");

  // --- Audio initialization ---
  Serial.print("Init Audio recorder... ");
  theAudio->begin();
  theAudio->setRecorderMode(AS_SETRECDR_STS_INPUTDEVICE_MIC);

  // PCM recording, DSP path, sampling rate, mono
  int err = theAudio->initRecorder(
                AS_CODECTYPE_PCM,
                "/mnt/sd0/BIN",   // Required even for 48kHz
                SAMPLE_RATE,
                AS_CHANNEL_MONO);

  if (err != AUDIOLIB_ECODE_OK) {
    Serial.print("NG  error=");
    Serial.println(err);
    Serial.println("Check: SRC DSP file in /mnt/sd0/BIN/ for 192kHz");
    while (1) { ; }
  }
  Serial.println("OK");

  // --- Start recording (DSP begins hardware sampling from here) ---
  theAudio->startRecorder();
  Serial.println("Recorder started.");
  Serial.println("");

  // --- Print configuration ---
  printConfig();

  Serial.println("Press any key to capture.");
  Serial.println("");
}


// ============================================================
// loop
// ============================================================
//
// Design notes:
// After startRecorder(), the DSP continuously samples at 48kHz.
// If readFrames is not called, the internal FIFO overflows and
// triggers an Attention error, corrupting the data. Therefore:
//   - While waiting for key input (IDLE): discard data via readFrames
//     to prevent overflow
//   - After key input (CAPTURING): accumulate data into pcm_buffer
//     and save
// This is managed as a state machine.
//
// States:
//   IDLE      : Waiting for key input. Continuously discarding data via readFrames.
//   CAPTURING : Accumulating data into pcm_buffer.
//   SAVING    : Writing to SD card and displaying preview.
// ============================================================

enum State { IDLE, CAPTURING, SAVING };

void loop() {
  static State    state      = IDLE;
  static uint32_t total_read = 0;
  static uint8_t *write_ptr  = nullptr;

  // Discard buffer used during IDLE to prevent FIFO overflow
  static char discard_buf[CAPTURE_BYTES];

  // buffering_time: time required to accumulate 768 samples (internal unit)
  // 768 / 48000 * 1000 = 16ms. Calculated from CAPTURE_SAMPLES with some margin.
  static const uint32_t buffering_time =
      (uint32_t)CAPTURE_SAMPLES * 1000 / SAMPLE_RATE;

  // --- Detect key input → transition from IDLE to CAPTURING ---
  if (Serial.available() > 0) {
    Serial.read();
    if (state == IDLE) {
      Serial.println("----------------------------------------");
      Serial.println("Capturing...");
      memset(pcm_buffer, 0, sizeof(pcm_buffer));
      total_read = 0;
      write_ptr  = (uint8_t*)pcm_buffer;
      state      = CAPTURING;
    }
  }

  // --- State-based processing ---
  switch (state) {

    case IDLE: {
      // Continuously discard data to prevent FIFO overflow
      uint32_t read_size = 0;
      int ret = theAudio->readFrames(
                    discard_buf, sizeof(discard_buf), &read_size);
      if (ret != AUDIOLIB_ECODE_OK &&
          ret != AUDIOLIB_ECODE_INSUFFICIENT_BUFFER_AREA) {
        Serial.print("readFrames error (IDLE): ");
        Serial.println(ret);
      }
      // Wait briefly if no data is available
      if (read_size == 0) delay(buffering_time);
      break;
    }

    case CAPTURING: {
      uint32_t read_size = 0;
      uint32_t remain    = CAPTURE_BYTES - total_read;

      int ret = theAudio->readFrames(
                    (char*)write_ptr, remain, &read_size);

      if (ret != AUDIOLIB_ECODE_OK &&
          ret != AUDIOLIB_ECODE_INSUFFICIENT_BUFFER_AREA) {
        Serial.print("readFrames error (CAPTURING): ");
        Serial.println(ret);
        state = IDLE;
        Serial.println("Capture failed. Press any key to retry.");
        break;
      }

      total_read += read_size;
      write_ptr  += read_size;

      if (total_read < CAPTURE_BYTES) {
        // Not enough data yet — wait and retry
        delay(buffering_time);
      } else {
        // Capture complete → transition to SAVING
        Serial.print("Captured: ");
        Serial.print(total_read / BYTES_PER_SAMPLE);
        Serial.println(" samples  OK");
        state = SAVING;
      }
      break;
    }

    case SAVING: {
      // Save to SD and show preview (no timing impact since capture is complete)
      saveToSD();
      printPreview();
      Serial.println("----------------------------------------");
      Serial.println("Press any key to capture again.");
      Serial.println("");
      state = IDLE;
      break;
    }
  }
}


// ============================================================
// Save to SD card
// Writes int16_t raw values as single-column CSV (no header)
// ============================================================
void saveToSD() {
  char filename[32];
  sprintf(filename, "%s%03d.csv", FILE_PREFIX, file_counter);

  File f = SD.open(filename, FILE_WRITE);
  if (!f) {
    Serial.print("Error: Cannot open ");
    Serial.println(filename);
    return;
  }

  for (int i = 0; i < CAPTURE_SAMPLES; i++) {
    f.println(pcm_buffer[i]);
  }
  f.close();

  Serial.print("Saved: ");
  Serial.println(filename);

  file_counter++;
}


// ============================================================
// Preview first 16 samples on Serial Monitor
// ============================================================
void printPreview() {
  Serial.println("--- Preview (first 16 samples, int16_t raw) ---");
  for (int i = 0; i < PREVIEW_SAMPLES; i++) {
    Serial.print("  [");
    if (i < 10) Serial.print("0");
    Serial.print(i);
    Serial.print("] ");
    Serial.println(pcm_buffer[i]);
  }
  Serial.println("------------------------------------------------");
}


// ============================================================
// Scan existing files to determine the next sequence number
// ============================================================
void findNextFileNumber() {
  char filename[32];
  file_counter = 1;
  while (file_counter <= 9999) {
    sprintf(filename, "%s%03d.csv", FILE_PREFIX, file_counter);
    if (!SD.exists(filename)) break;
    file_counter++;
  }
}


// ============================================================
// Print current configuration to Serial Monitor
// ============================================================
void printConfig() {
  Serial.println("=== Configuration ===");
  Serial.print("  Sample rate  : ");
  Serial.print(SAMPLE_RATE);
  Serial.println(" Hz");
  Serial.print("  Capture pts  : ");
  Serial.println(CAPTURE_SAMPLES);
  Serial.print("  Time window  : ");
  Serial.print((float)CAPTURE_SAMPLES / SAMPLE_RATE * 1000.0, 2);
  Serial.println(" ms");
  Serial.print("  Freq. resol. : ");
  Serial.print((float)SAMPLE_RATE / CAPTURE_SAMPLES, 1);
  Serial.println(" Hz/bin");
  Serial.print("  Nyquist      : ");
  Serial.print(SAMPLE_RATE / 2 / 1000);
  Serial.println(" kHz");
  Serial.print("  File prefix  : ");
  Serial.println(FILE_PREFIX);
  Serial.println("=====================");
  Serial.println("");
}
