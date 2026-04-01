/*
 * MainCore.ino  pressure_monitor_with_display  Ver 1.1
 *
 * Processing pipeline:
 *   MICA pin (48kHz) → FFT (1024-point, Hanning window, no overlap)
 *   → bin1–320 (320 dimensions) → DNNRT inference → Estimated pressure
 *   → Transfer to SubCore1 (pressure + FFT data) via MP → ILI9341 display
 *
 * SubCore transfer rules (based on "Getting Started with Low-Power Edge AI on SPRESENSE"):
 *   ① Mutex.Trylock() → if it fails (SubCore is rendering), return and
 *      continue reading audio → no SEND_INTERVAL needed to prevent buffer overflow
 *   ② Copy data to a static variable in loop() via memcpy before passing the pointer
 *      → prevents MainCore from overwriting memory while SubCore is rendering
 *
 * Required files on SD card:
 *   model.nnb  … Trained model (root of SD)
 *   BIN/       … Audio DSP codec files (root of SD)
 *
 * Memory setting (Arduino IDE → Tools → Memory):
 *   MainCore Memory: set to 1024KB or higher
 *
 * Version history:
 *   Ver 1.0: Initial release
 *   Ver 1.1: Changed sendData to static variable in loop(); removed SEND_INTERVAL
 *   Ver 1.2: Moved MP.begin() before startRecorder() to prevent Audio buffer overflow
 *            during SubCore's setupLcd()
 */

#ifdef SUBCORE
#error "Core selection is wrong!! Select MainCore."
#endif

#include <Audio.h>
#include <FFT.h>
#include <SDHCI.h>
#include <DNNRT.h>
#include <MP.h>
#include <MPMutex.h>

/* ===== Constants ===== */
#define FFT_LEN     1024
#define BIN_START   1              // Exclude bin0 (DC component)
#define BIN_END     320
#define DNN_INPUT   (BIN_END - BIN_START + 1)  // = 320 dimensions
#define MODEL_FILE  "model.nnb"

/* ===== Inter-core communication ===== */
const int subcore = 1;
MPMutex mutex(MP_MUTEX_ID0);

// Data structure shared between MainCore and SubCore
// Note: DNN_INPUT definition must match SubCore1/SubCore1.ino exactly
struct DisplayData {
  float pressure;             // Estimated pressure [MPa]
  float fft_data[DNN_INPUT];  // FFT amplitude bin1–320
};

/* ===== Global objects ===== */
SDClass     SD;
DNNRT       dnnrt;
AudioClass* theAudio = AudioClass::getInstance();
FFTClass<AS_CHANNEL_MONO, FFT_LEN> FFT;
DNNVariable input(DNN_INPUT);

/* ===== setup ===== */
void setup() {
  Serial.begin(115200);

  // SD card initialization
  while (!SD.begin()) {
    Serial.println("[ERROR] Insert SD card");
    delay(500);
  }
  Serial.println("[OK] SD card mounted");

  // FFT initialization (Hanning window, mono, no overlap — matches training configuration)
  FFT.begin(WindowHanning, AS_CHANNEL_MONO, 0);
  Serial.println("[OK] FFT initialized");

  // Audio initialization
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

  // Load DNNRT model (after Audio initialization)
  File nnbfile = SD.open(MODEL_FILE);
  if (!nnbfile) {
    Serial.println("[ERROR] model.nnb not found");
    while (1);
  }
  int ret2 = dnnrt.begin(nnbfile);
  if (ret2 < 0) {
    Serial.println("[ERROR] DNNRT begin failed: " + String(ret2));
    while (1);
  }
  Serial.println("[OK] Model loaded: " + String(MODEL_FILE));

  // Start SubCore before beginning recording
  // Note: MP.begin() blocks until SubCore's setup() completes.
  //       Must be called before startRecorder() to prevent Audio buffer overflow
  //       during SubCore's setupLcd()
  MP.begin(subcore);
  Serial.println("[OK] SubCore1 started");

  // Start recording (after SubCore is ready)
  theAudio->startRecorder();
  Serial.println("[OK] Recording started");
  Serial.println("----------------------------------------");
  Serial.println("  Pressure estimation + Display running");
  Serial.println("----------------------------------------");
}

/* ===== loop ===== */
void loop() {
  static const uint32_t buffer_size    = FFT_LEN * sizeof(int16_t);
  static const uint32_t buffering_time = FFT_LEN * 1000 / AS_SAMPLINGRATE_48000;
  static char  buff[FFT_LEN * sizeof(int16_t)];
  static float fft_result[FFT_LEN];

  // Send buffer to SubCore (static address fixed; safely copied via memcpy)
  static DisplayData sendData;

  uint32_t read_size = 0;

  // ----- Read audio data -----
  int ret = theAudio->readFrames(buff, buffer_size, &read_size);
  if (ret != AUDIOLIB_ECODE_OK &&
      ret != AUDIOLIB_ECODE_INSUFFICIENT_BUFFER_AREA) {
    Serial.println("[ERROR] readFrames: " + String(ret));
    theAudio->stopRecorder();
    while (1);
  }
  if (read_size < buffer_size) {
    delay(buffering_time);
    return;
  }

  // ----- FFT computation (1024-point → 512-bin amplitude spectrum) -----
  FFT.put((q15_t*)buff, FFT_LEN);
  FFT.get(fft_result, 0);

  // ----- Prepare DNNRT input (bin1–320 → 320 dimensions) -----
  float* input_buf = input.data();
  for (int i = 0; i < DNN_INPUT; i++) {
    input_buf[i] = fft_result[BIN_START + i];
  }

  // ----- DNNRT inference -----
  dnnrt.inputVariable(input, 0);
  dnnrt.forward();
  DNNVariable output = dnnrt.outputVariable(0);
  float pressure = output[0];

  // Output to Serial Monitor
  Serial.print("Pressure: ");
  Serial.print(pressure, 3);
  Serial.println(" MPa");

  // ----- Transfer to SubCore -----
  // Skip if SubCore is currently rendering (Mutex acquisition failed)
  // → readFrames continues to be called, preventing Audio buffer overflow
  ret = mutex.Trylock();
  if (ret != 0) return;

  // Copy data to static buffer before passing the pointer
  // → safe even if fft_result is overwritten while SubCore is rendering
  sendData.pressure = pressure;
  memcpy(sendData.fft_data, &fft_result[BIN_START], DNN_INPUT * sizeof(float));

  ret = MP.Send(0, &sendData, subcore);
  if (ret < 0) Serial.println("[WARN] MP.Send failed: " + String(ret));

  mutex.Unlock();
}
