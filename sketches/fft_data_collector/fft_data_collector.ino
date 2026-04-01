/*
 * fft_data_collector.ino
 *
 * Based on fft_datacollection.ino with the following changes:
 *   - No averaging filter
 *   - No overlap (FFT_LEN/2 → 0)
 *   - No maxSpectrum normalization
 *   - Saved bins: FFT_LEN/8 (128) → FFT_LEN/2 (512)
 *   - Trigger: 1-second timer → key input
 *   - Filename: data000.csv → fft_48k_001.csv (auto-incremented)
 *
 * Output format:
 *   - 512 rows × 1 column CSV (float, 6 decimal places)
 *   - bin0 (0Hz) to bin511 (23953Hz)
 *   - Frequency resolution: 48000 / 1024 = 46.875 Hz/bin
 *
 * Ver 1.0: Initial release
 */

#include <Audio.h>
#include <FFT.h>
#include <SDHCI.h>
SDClass SD;

#define FFT_LEN     1024
#define MAX_FILES   100          // Maximum number of files to save

char FILE_PREFIX[32] = {};       // Input from Serial Monitor
int  gCounter = 0;               // File sequence number (reset when FILE_PREFIX is updated)

// Initialize FFT: mono, 1024 samples
FFTClass<AS_CHANNEL_MONO, FFT_LEN> FFT;

AudioClass* theAudio = AudioClass::getInstance();

void setup() {
  Serial.begin(115200);

  // Wait for SD card insertion
  while (!SD.begin()) { Serial.println("Insert SD card"); }

  // Hanning window, mono, no overlap
  FFT.begin(WindowHanning, AS_CHANNEL_MONO, 0);

  Serial.println("Init Audio Recorder");
  theAudio->begin();
  // Set input source to microphone
  theAudio->setRecorderMode(AS_SETRECDR_STS_INPUTDEVICE_MIC);
  // Recorder configuration:
  //   Format: PCM (16-bit raw data)
  //   DSP codec path on SD card (/mnt/sd0/BIN)
  //   Sampling rate: 48000 Hz, Mono
  int err = theAudio->initRecorder(AS_CODECTYPE_PCM,
              "/mnt/sd0/BIN", AS_SAMPLINGRATE_48000, AS_CHANNEL_MONO);
  if (err != AUDIOLIB_ECODE_OK) {
    Serial.println("Recorder initialize error");
    while (1);
  }

  // Read FILE_PREFIX from Serial Monitor
  // Note: called before startRecorder() to prevent FIFO overflow
  inputFilePrefix();

  Serial.println("Start Recorder");
  theAudio->startRecorder(); // Start recording after FILE_PREFIX is confirmed

  Serial.println("Press any key to capture.");
}


void loop() {
  static const uint32_t buffering_time =
      FFT_LEN * 1000 / AS_SAMPLINGRATE_48000;
  static const uint32_t buffer_size = FFT_LEN * sizeof(int16_t);
  static char buff[buffer_size];
  static float pDst[FFT_LEN];
  uint32_t read_size;

  int ret = theAudio->readFrames(buff, buffer_size, &read_size);  // ret is also reused inside the for loop
  if (ret != AUDIOLIB_ECODE_OK &&
      ret != AUDIOLIB_ECODE_INSUFFICIENT_BUFFER_AREA) {
    Serial.println("Error err = " + String(ret));
    theAudio->stopRecorder();
    while (1);
  }

  if (read_size < buffer_size) {
    delay(buffering_time);
    return;
  }

  FFT.put((q15_t*)buff, FFT_LEN); // Run FFT
  FFT.get(pDst, 0);               // Retrieve FFT result

  // On key input, start consecutive capture of MAX_FILES files
  if (Serial.available() > 0) {
    Serial.read();
    Serial.println("Start capturing " + String(MAX_FILES) + " files...");

    for (int n = 0; n < MAX_FILES; n++) {
      Serial.print("[" + String(n + 1) + "/" + String(MAX_FILES) + "] ");

      // Wait until one full frame is available
      uint32_t read_size = 0;
      do {
        ret = theAudio->readFrames(buff, buffer_size, &read_size);
        if (ret != AUDIOLIB_ECODE_OK &&
            ret != AUDIOLIB_ECODE_INSUFFICIENT_BUFFER_AREA) {
          Serial.println("Error err = " + String(ret));
          theAudio->stopRecorder();
          while (1);
        }
        if (read_size < buffer_size) delay(buffering_time);
      } while (read_size < buffer_size);

      FFT.put((q15_t*)buff, FFT_LEN);
      FFT.get(pDst, 0);

      theAudio->stopRecorder(); // Stop recording
      saveData(pDst, FFT_LEN / 2, MAX_FILES); // Save 512 bins
      theAudio->startRecorder(); // Resume recording
    }

    Serial.println("All " + String(MAX_FILES) + " files captured.");

    // Prompt for next FILE_PREFIX before returning to key-wait state
    theAudio->stopRecorder(); // Stop to prevent FIFO overflow during input wait
    inputFilePrefix();
    theAudio->startRecorder();

    Serial.println("Press any key to capture.");
  }
}


// Reads FILE_PREFIX from Serial and resets gCounter
void inputFilePrefix() {
  Serial.println("Enter FILE_PREFIX then press Enter:");
  Serial.print("> ");
  readLineFromSerial(FILE_PREFIX, sizeof(FILE_PREFIX));
  Serial.println("FILE_PREFIX set to: " + String(FILE_PREFIX));
  Serial.println("");
  gCounter = 0; // Reset sequence number for the new FILE_PREFIX
}


// Reads one line from Serial Monitor (confirmed by Enter key)
void readLineFromSerial(char* buf, int bufSize) {
  int idx = 0;
  memset(buf, 0, bufSize);
  while (true) {
    if (Serial.available() > 0) {
      char c = Serial.read();
      if (c == '\n' || c == '\r') {
        if (idx > 0) break;      // Confirm on Enter
      } else if (idx < bufSize - 1) {
        buf[idx++] = c;
        Serial.print(c);         // Echo input character
      }
    }
  }
  Serial.println();              // New line
}


// Save FFT data to SD card as CSV
void saveData(float* pDst, int dsize, int quantity) {
  char filename[32] = {};

  // Return without saving if the maximum file count has been reached
  if (gCounter >= quantity) {
    Serial.println("Data accumulated");
    return;
  }

  // Open file for writing
  // FILE_PREFIX can be changed via Serial Monitor at startup
  sprintf(filename, "%s%03d.csv", FILE_PREFIX, gCounter++);
  // Remove existing file if present
  if (SD.exists(filename)) SD.remove(filename);
  // Open file
  File myFile = SD.open(filename, FILE_WRITE);
  // Write data
  for (int i = 0; i < dsize; ++i) {
    myFile.println(String(pDst[i], 6));
  }
  myFile.close();  // Close file
  Serial.println("Data saved as " + String(filename));
}
