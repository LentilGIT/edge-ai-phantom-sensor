/*
 * SubCore1.ino  pressure_monitor_with_display
 *
 * SubCore1 responsibilities:
 *   Receives FFT data and estimated pressure from MainCore,
 *   then renders the FFT spectrum and pressure value on the ILI9341 display.
 *
 * Note: When uploading via Arduino IDE,
 *       select Tools → Core → SubCore1.
 */

#ifndef SUBCORE
#error "Core selection is wrong!! Select SubCore1."
#endif

#include <MP.h>
#include <MPMutex.h>
MPMutex mutex(MP_MUTEX_ID0);

// Data structure must match MainCore.ino exactly
#define DNN_INPUT 320
struct DisplayData {
  float pressure;
  float fft_data[DNN_INPUT];
};

// Include display utility (LCD initialization and rendering functions)
#include "displayUtil_Ver1_3.h"

void setup() {
  setupLcd();   // Initialize LCD and render initial screen
  MP.begin();   // Notify MainCore that SubCore1 is ready
}

void loop() {
  int8_t       msgid;
  DisplayData* recvData;  // Pointer to MainCore's static variable

  // Receive data from MainCore (returns immediately if no data available)
  int ret = MP.Recv(&msgid, &recvData);
  if (ret < 0) return;

  // Acquire Mutex before updating display
  do {
    ret = mutex.Trylock();
  } while (ret != 0);

  // Render FFT spectrum
  showSpectrum(recvData->fft_data);

  // Render estimated pressure
  showPressure(recvData->pressure);

  mutex.Unlock();  // Release Mutex back to MainCore
}
