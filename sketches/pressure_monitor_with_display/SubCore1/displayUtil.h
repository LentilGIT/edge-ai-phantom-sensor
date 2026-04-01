/*
 * displayUtil.h  Ver 1.3
 *
 * ILI9341 display control utility for Sony Spresense.
 *
 * Layout (landscape, rotation=3: 320×240px):
 *
 *   x=0                              x=319
 *   +----------------------------------+  y=0
 *   |  DNN Pressure Predictor          |  y=4–19   textSize=2
 *   +----------------------------------+  y=22  (separator line)
 *   |  FFT Spectrum                    |  y=24–191  (168px)
 *   |  Left=low freq (47Hz)            |
 *   |  Right=high freq (15kHz)         |
 *   |  Cyan = 1–1.5kHz band            |
 *   +----------------------------------+  y=194 (separator line)
 *   |  Pred. Pressure: 0.07 MPa        |  y=207  textSize=2
 *   +----------------------------------+  y=239
 *
 * Version history:
 *   Ver 1.0: Portrait orientation (rotation=2)
 *   Ver 1.1: Switched to landscape (rotation=3)
 *   Ver 1.2: Label and pressure value displayed side by side; textSize=2
 *   Ver 1.3: Added title; unified label to "Predictor"
 *            Shortened to "Pred. Pressure: X.XX MPa" to fit on one line
 */

#include "Adafruit_GFX.h"
#include "Adafruit_ILI9341.h"

/* ===== Display configuration ===== */
#define TFT_DC  9
#define TFT_CS  10
Adafruit_ILI9341 display = Adafruit_ILI9341(TFT_CS, TFT_DC);

/* ===== Screen dimensions (rotation=3, landscape) ===== */
#define LCD_W  320
#define LCD_H  240

/* ===== Title configuration ===== */
#define TITLE_STR  "DNN Pressure Predictor"
#define TITLE_X    5
#define TITLE_Y    4
#define TITLE_SEP_Y  22   // Separator line below title

/* ===== Spectrum graph configuration ===== */
#define GRAPH_X  0
#define GRAPH_Y  24           // Offset down to accommodate title
#define GRAPH_W  320
#define GRAPH_H  168          // (194-24-2) px
#define SAMPLES  320

/* ===== Pressure display configuration ===== */
#define SPEC_SEP_Y  194   // Separator line below spectrum
#define PRES_X      5
#define PRES_Y      207

/* ===== Spectrum scale ===== */
#define MAX_SPECTRUM  25.0f

/* ===== LCD initialization ===== */
void setupLcd() {
  display.begin();
  display.setRotation(3);            // Landscape (320×240)
  display.fillScreen(ILI9341_BLACK);

  // Title
  display.setTextColor(ILI9341_CYAN);
  display.setTextSize(2);
  display.setCursor(TITLE_X, TITLE_Y);
  display.print(TITLE_STR);

  // Separator line below title
  display.drawFastHLine(0, TITLE_SEP_Y, LCD_W, ILI9341_DARKGREY);

  // Separator line below spectrum
  display.drawFastHLine(0, SPEC_SEP_Y, LCD_W, ILI9341_DARKGREY);
}

/* ===== FFT spectrum display ===== */
// data: amplitude array for bin1–bin320 (320 elements)
// Left edge = low frequency (bin1 = 47Hz), right edge = high frequency (bin320 = 15kHz)
// The 1.1–1.5kHz band (dominant pressure-correlated band identified by SHAP analysis)
// is highlighted in cyan.
void showSpectrum(float *data) {
  static uint16_t frameBuf[GRAPH_H][GRAPH_W];

  for (int x = 0; x < GRAPH_W; ++x) {
    int bin_idx = (x < SAMPLES) ? x : SAMPLES - 1;

    int bar_h = (int)(data[bin_idx] / MAX_SPECTRUM * GRAPH_H);
    bar_h = (bar_h < 0) ? 0 : (bar_h > GRAPH_H) ? GRAPH_H : bar_h;

    for (int y = 0; y < GRAPH_H; ++y) {
      if (y >= (GRAPH_H - bar_h)) {
        // Highlight bin25–31 (approx. 1172–1453Hz) in cyan
        // This band shows the highest SHAP contribution to pressure estimation
        frameBuf[y][x] = (bin_idx >= 24 && bin_idx <= 31)
                          ? ILI9341_CYAN : ILI9341_WHITE;
      } else {
        frameBuf[y][x] = ILI9341_BLACK;
      }
    }
  }

  display.drawRGBBitmap(GRAPH_X, GRAPH_Y,
    (uint16_t*)frameBuf, GRAPH_W, GRAPH_H);
}

/* ===== Pressure text display ===== */
void showPressure(float pressure) {
  // Clear pressure display area
  display.fillRect(0, SPEC_SEP_Y + 1, LCD_W, LCD_H - SPEC_SEP_Y - 1,
                   ILI9341_BLACK);

  // Color-code pressure value by range
  uint16_t color;
  if (pressure < 0.10f) {
    color = ILI9341_GREEN;   // Low pressure
  } else if (pressure < 0.14f) {
    color = ILI9341_YELLOW;  // Medium pressure
  } else {
    color = ILI9341_RED;     // High pressure
  }

  // "Pred. Pressure: X.XX MPa" (textSize=2: approx. 276px — fits within 320px)
  char buf[32];
  snprintf(buf, sizeof(buf), "Pred. Pressure: %.2f MPa", pressure);

  display.setTextColor(color);
  display.setTextSize(2);
  display.setCursor(PRES_X, PRES_Y);
  display.print(buf);
}
