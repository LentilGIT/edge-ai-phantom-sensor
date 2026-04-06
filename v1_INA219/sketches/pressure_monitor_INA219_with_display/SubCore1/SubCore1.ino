/*
 * SubCore1.ino  pressure_monitor_INA219_with_display
 *
 * SubCore1 responsibilities:
 *   Receives FFT spectrum, pressure value, and status from MainCore,
 *   then renders the FFT bar graph and pressure/status text on the
 *   ILI9341 display (240×320px, portrait orientation).
 *
 * Received data format from MainCore:
 *   buff[0–254] : Normalized FFT spectrum (0.0–1.0, 255 bins)
 *   buff[255]   : Estimated pressure [MPa]
 *   msgid       : 100=STOP, 101=OK, 102=OVER PRESSURE
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

// Include display utility (ILI9341 initialization and rendering functions)
#include "displayUtil_INA219.h"


void setup() {
  setupLcd();  // Initialize ILI9341 display
  MP.begin();  // Notify MainCore that SubCore1 is ready
}


void loop() {
  int     ret;
  int8_t  msgid;
  float  *buff;  // Pointer to MainCore's display_data array

  // Receive data from MainCore (returns immediately if no data available)
  ret = MP.Recv(&msgid, &buff);
  if (ret < 0) return;

  // Acquire Mutex before updating display
  // (prevents MainCore from overwriting buff while rendering)
  do {
    ret = mutex.Trylock();
  } while (ret != 0);

  // Convert msgid to status code for showSpectrum()
  // msgid: 100=STOP, 101=OK, 102=OVER PRESSURE
  // status:   0=STOP,   1=OK,   2=OVER PRESSURE
  uint8_t status = 0;
  if      (msgid == 100) { status = 0; }  // STOP
  else if (msgid == 101) { status = 1; }  // OK
  else if (msgid == 102) { status = 2; }  // OVER PRESSURE

  // Render FFT spectrum bar graph and pressure/status text
  // buff[0–254]: normalized spectrum
  // buff[255]:   pressure value [MPa]
  showSpectrum(buff, status);

  // Release Mutex back to MainCore
  mutex.Unlock();
}
