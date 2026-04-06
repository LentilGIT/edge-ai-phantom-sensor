/*
 * pressure_monitor.ino  Ver 1.3
 *
 * Applies FFT to the current waveform acquired via LT6105,
 * then runs real-time pressure estimation using DNNRT.
 * Estimated pressure is displayed on Serial Monitor.
 *
 * Processing pipeline:
 *   MICA pin (48kHz) → FFT (1024-point, Hanning window, no overlap)
 *   → bin1–320 (320 dimensions) → DNNRT inference → Estimated pressure [MPa]
 *   → Output to Serial Monitor
 *
 * Required files on SD card:
 *   model.nnb  … Trained model exported from Neural Network Console (root of SD)
 *   BIN/       … Spresense Audio DSP codec files (root of SD)
 *
 * Frequency resolution : 48000 / 1024 = 46.875 Hz/bin
 * bin1–320            : 46.9Hz – 15000Hz
 * Frame duration      : 1024 / 48000 ≈ 21.3 ms
 *
 * Version history:
 *   Ver 1.0: Initial release (bin0–511, 512-dimensional input)
 *   Ver 1.1: Reduced Audio buffer size (workaround for DNNRT error -16)
 *   Ver 1.2: Changed to bin1–320 (320-dimensional input), updated model to 128→64→32→1
 *   Ver 1.3: Restored default Audio buffer; moved DNNRT initialization after Audio init
 */

#include <Audio.h>
#include <FFT.h>
#include <SDHCI.h>
#include <DNNRT.h>

/* ===== Constants ===== */
#define FFT_LEN       1024           // FFT size
#define BIN_START     1              // First input bin (bin0 DC component excluded)
#define BIN_END       320            // Last input bin
#define DNN_INPUT     (BIN_END - BIN_START + 1)  // = 320 dimensions
#define MODEL_FILE    "model.nnb"    // Model filename on SD card root

/* ===== Global objects ===== */
SDClass     SD;
DNNRT       dnnrt;
AudioClass* theAudio = AudioClass::getInstance();

// Hanning window, mono, 1024-point FFT
FFTClass<AS_CHANNEL_MONO, FFT_LEN> FFT;

// DNNRT input variable (320 dimensions: bin1–bin320)
DNNVariable input(DNN_INPUT);

/* ===== setup ===== */
void setup() {
  Serial.begin(115200);

  // ----- SD card initialization -----
  while (!SD.begin()) {
    Serial.println("[ERROR] Insert SD card");
    delay(500);
  }
  Serial.println("[OK] SD card mounted");

  // ----- FFT initialization -----
  // Hanning window, mono, no overlap (matches training configuration)
  FFT.begin(WindowHanning, AS_CHANNEL_MONO, 0);
  Serial.println("[OK] FFT initialized (Hanning, no overlap)");

  // ----- Audio initialization -----
  theAudio->begin();
  theAudio->setRecorderMode(AS_SETRECDR_STS_INPUTDEVICE_MIC);

  int ret = theAudio->initRecorder(
    AS_CODECTYPE_PCM,
    "/mnt/sd0/BIN",
    AS_SAMPLINGRATE_48000,
    AS_CHANNEL_MONO
  );
  if (ret != AUDIOLIB_ECODE_OK) {
    Serial.println("[ERROR] Audio init failed: " + String(ret));
    while (1);
  }
  Serial.println("[OK] Audio initialized (48kHz, Mono)");

  // ----- Load model -----
  File nnbfile = SD.open(MODEL_FILE);
  if (!nnbfile) {
    Serial.println("[ERROR] " + String(MODEL_FILE) + " not found");
    while (1);
  }
  int ret2 = dnnrt.begin(nnbfile);
  if (ret2 < 0) {
    Serial.println("[ERROR] DNNRT begin failed: " + String(ret2));
    while (1);
  }
  Serial.println("[OK] Model loaded: " + String(MODEL_FILE));

  theAudio->startRecorder();
  Serial.println("[OK] Recording started");
  Serial.println("----------------------------------------");
  Serial.println("  bin: " + String(BIN_START) + " - " + String(BIN_END)
               + "  (" + String(BIN_START * 48000 / FFT_LEN) + "Hz"
               + " - " + String(BIN_END   * 48000 / FFT_LEN) + "Hz)");
  Serial.println("  DNN input: " + String(DNN_INPUT) + " dimensions");
  Serial.println("  Pressure estimation running...");
  Serial.println("----------------------------------------");
}

/* ===== loop ===== */
void loop() {
  static const uint32_t buffer_size    = FFT_LEN * sizeof(int16_t); // 2048 bytes
  static const uint32_t buffering_time = FFT_LEN * 1000 / AS_SAMPLINGRATE_48000; // 21ms
  static char  buff[FFT_LEN * sizeof(int16_t)];
  static float fft_result[FFT_LEN];
  uint32_t read_size = 0;

  // ----- Read one frame of audio data -----
  int ret = theAudio->readFrames(buff, buffer_size, &read_size);

  if (ret != AUDIOLIB_ECODE_OK &&
      ret != AUDIOLIB_ECODE_INSUFFICIENT_BUFFER_AREA) {
    Serial.println("[ERROR] readFrames: " + String(ret));
    theAudio->stopRecorder();
    while (1);
  }

  // Wait and return if insufficient data has accumulated
  if (read_size < buffer_size) {
    delay(buffering_time);
    return;
  }

  // ----- FFT computation -----
  // Apply FFT to 1024 samples and retrieve 512-bin amplitude spectrum
  FFT.put((q15_t*)buff, FFT_LEN);
  FFT.get(fft_result, 0);

  // ----- Prepare DNNRT input (bin1–bin320) -----
  // Exclude bin0 (0Hz, DC component); pass bin1–320 (320 dimensions) to DNNRT
  // Must match the bin range used during training
  float* input_buf = input.data();
  for (int i = 0; i < DNN_INPUT; i++) {
    input_buf[i] = fft_result[BIN_START + i];  // fft_result[1] – fft_result[320]
  }

  // ----- DNNRT inference -----
  dnnrt.inputVariable(input, 0);
  dnnrt.forward();
  DNNVariable output = dnnrt.outputVariable(0);

  float pressure = output[0];  // Estimated pressure [MPa]

  // ----- Output to Serial Monitor -----
  Serial.print("Pressure: ");
  Serial.print(pressure, 3);
  Serial.println(" MPa");
}
