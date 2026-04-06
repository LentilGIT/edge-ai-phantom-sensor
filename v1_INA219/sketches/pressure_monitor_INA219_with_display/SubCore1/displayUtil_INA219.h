/*
 * displayUtil_INA219.h  Ver 1.1
 *
 * ILI9341 display control utility for the INA219-based pressure estimation system.
 * Renders FFT spectrum as a vertical bar graph and displays pressure or status text.
 *
 * Layout (portrait orientation, rotation=2: 240×320px):
 *
 *   The display uses a dual-rotation approach:
 *   - rotation=2 (portrait) for the FFT bar graph (full 320px height)
 *   - rotation=3 (landscape) for text rendering (coordinate system rotates)
 *
 *   FFT spectrum area:
 *     X: 40–239 (GRAPH_WIDTH = 200px)
 *     Y: 0–319  (GRAPH_HEIGHT = 320px, full height)
 *     255 bins mapped vertically (low frequency at top, high at bottom)
 *     Bars extend horizontally (left = 0, right = full amplitude)
 *     Color: Blue
 *
 *   Pressure/status text area (rotation=3 coordinate system):
 *     Position: (TX=35, TY=210)
 *     Text size: 2
 *     STOP / OVER PRESSURE → Red
 *     Pressure value → Blue ("Pressure X.XX MPa")
 *
 * Version history:
 *   Ver 1.0: Initial release — portrait layout, 255-bin FFT display
 *   Ver 1.1: Increased graph height to 320px (full screen utilization)
 */

#ifndef DISPLAY_UTIL_H
#define DISPLAY_UTIL_H

#include "Adafruit_GFX.h"
#include "Adafruit_ILI9341.h"

// ============================================
// Display pin configuration
// ============================================
#define TFT_DC  9
#define TFT_CS  10

Adafruit_ILI9341 display = Adafruit_ILI9341(TFT_CS, TFT_DC);

// ============================================
// Text display position (in rotation=3 coordinate system)
// ============================================
#define TX 35   // Text X position [px]
#define TY 210  // Text Y position [px]

// ============================================
// Bar graph configuration
// ============================================
#define GRAPH_WIDTH  200  // Bar graph width [px]
#define GRAPH_HEIGHT 320  // Bar graph height [px] (full screen)

#define GX 40  // Graph origin X [px]
#define GY 0   // Graph origin Y [px]

#define SAMPLES 255  // Number of FFT bins to render


// ============================================
// LCD initialization
// ============================================
void setupLcd() {
  display.begin();
  display.setRotation(3);             // Landscape — used for text cursor positioning
  display.fillScreen(ILI9341_BLACK);  // Clear screen to black
  display.setRotation(2);             // Portrait — used for bar graph rendering
}


// ============================================
// Render FFT spectrum and pressure/status text
//
// Parameters:
//   data   : float array
//            data[0–254] — normalized FFT spectrum (0.0–1.0)
//            data[255]   — estimated pressure [MPa]
//   status : display mode
//            0 = STOP         (pump not running — red text)
//            1 = OK           (normal — show pressure value in blue)
//            2 = OVER PRESSURE (threshold exceeded — red text)
// ============================================
void showSpectrum(float *data, uint8_t status) {
  static uint16_t frameBuf[GRAPH_HEIGHT][GRAPH_WIDTH];

  // Build frame buffer: vertical bar graph
  // Each row i corresponds to FFT bin i (up to SAMPLES=255)
  // Bar extends horizontally: 0.0–1.0 → 1–201 pixels
  for (int i = 0; i < GRAPH_HEIGHT; ++i) {
    int val;

    if (i < SAMPLES) {
      val = data[i] * GRAPH_WIDTH + 1;  // Normalize: 0.0–1.0 → 1–201 px
    } else {
      val = 0;  // No data beyond SAMPLES bins
    }

    // Clip to graph width
    val = (val > GRAPH_WIDTH) ? GRAPH_WIDTH : val;

    for (int j = 0; j < GRAPH_WIDTH; ++j) {
      // Bar region: Blue; background: Dark grey
      frameBuf[i][j] = (j > val) ? ILI9341_DARKGREY : ILI9341_BLUE;
    }
  }

  // Transfer frame buffer to display (portrait mode, rotation=2)
  display.drawRGBBitmap(GX, GY, (uint16_t*)frameBuf, GRAPH_WIDTH, GRAPH_HEIGHT);

  // Switch to landscape mode for text rendering
  display.setRotation(3);

  // Clear previous text area
  display.fillRect(TX - 5, TY - 5, 220, 25, ILI9341_BLACK);

  // Render text based on status
  display.setCursor(TX, TY);
  display.setTextSize(2);

  if (status == 0) {
    // Pump not running
    display.setTextColor(ILI9341_RED);
    display.println("STOP");

  } else if (status == 2) {
    // Pressure exceeded threshold
    display.setTextColor(ILI9341_RED);
    display.println("OVER PRESSURE");

  } else {
    // Normal operation: display pressure value
    display.setTextColor(ILI9341_BLUE);

    float pressure = data[255];  // Pressure stored at index 255

    // Format: "Pressure X.XX MPa"
    display.print("Pressure ");
    display.print(pressure, 2);  // 2 decimal places
    display.println(" MPa");
  }

  // Restore portrait mode for next bar graph render
  display.setRotation(2);
}

#endif // DISPLAY_UTIL_H
