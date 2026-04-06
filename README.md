# edge-ai-phantom-sensor

Real-time pump pressure estimation using only motor current signals and 
edge AI — eliminating the need for dedicated sensors in deployment.

## Project Overview

This repository documents the full development journey of a sensorless 
pump pressure estimation system, from initial prototype to deployed 
edge AI solution.

The core idea: pump discharge pressure can be inferred in real time 
using only motor current signals — without installing any dedicated 
pressure sensors or flow meters. This is especially valuable in compact 
equipment where physical sensor installation is impractical.

## Two Implementation Versions

### v1_INA219 — Initial Prototype (TI INA219)

- **Sensor**: Texas Instruments INA219 (I2C interface, 0x40)
- **Input**: 257-dimensional (DC mean + Peak-to-Peak + FFT bin1–255)
- **Model**: DNN 257→128→64→32→1
- **Result**: R²=0.9992 under controlled conditions
- **Limitation**: DC mean and Peak-to-Peak dominate predictions 
  (confirmed by SHAP analysis) — these features drift with motor 
  temperature, causing accuracy degradation during extended operation

### v2_LT6105 — Primary Version (Analog Devices LT6105)

- **Sensor**: LT6105 current sense amplifier via Spresense MIC_A pin
- **Input**: 320-dimensional (FFT bin1–320 only — no DC or P-P)
- **Model**: DNN 320→128→64→32→1
- **Result**: R²=0.9946, RMSE=3.18 kPa across diverse conditions
- **Improvement**: FFT frequency features are temperature-stable, 
  making this version robust across cold-start and warm-running 
  conditions

## Why the Transition from v1 to v2?

SHAP analysis on v1 revealed the root cause of its limitations:

| Version | Top Feature | SHAP Value | Temperature Stable? |
|---------|------------|------------|-------------------|
| v1 INA219 | Peak-to-Peak [mA] | 0.046 | ❌ No |
| v1 INA219 | DC mean [mA] | 0.035 | ❌ No |
| v2 LT6105 | 1.172 kHz (FFT) | 0.163 | ✅ Yes |
| v2 LT6105 | 1.125 kHz (FFT) | 0.134 | ✅ Yes |

v1 achieved high accuracy under controlled conditions but relied on 
DC current and amplitude features that drift with motor temperature. 
v2 was redesigned to use only FFT frequency features — validated by 
SHAP to be temperature-invariant — achieving robust performance across 
real operating conditions.

## Why Motor Current?

| Signal | Problem |
|--------|---------|
| Acoustic | Vulnerable to ambient noise in industrial environments |
| Vibration | Accuracy depends heavily on sensor placement |
| **Motor Current** | **Always measured at the power supply — no installation variability** |

Critically, the **frequency domain** of the current signal is stable 
across temperature changes, whereas amplitude and DC current drift 
significantly — making FFT-based features far more robust for 
long-term deployment.

## Repository Structure

edge-ai-phantom-sensor/
├── v1_INA219/
│   ├── sketches/          — Arduino C++ firmware (INA219 version)
│   ├── analysis/          — SHAP analysis, model weights, result plot
│   └── data/              — INA219_valid.zip (320 CSV files)
│
└── v2_LT6105/
├── sketches/          — Arduino C++ firmware (LT6105 version)
├── analysis/          — SHAP analysis (320bin & 512bin), model weights
└── data/              — TEST5.zip (1,500 CSV files)

## v2_LT6105 — Technical Details

### Hardware

| Component | Part | Role |
|-----------|------|------|
| Microcontroller | Sony Spresense (Arm Cortex-M4F) | FFT + DNN inference + display |
| Current Sensor | LT6105 | Motor current measurement (1V/1A output) |
| Display | ILI9341 (SPI) | Real-time pressure display |
| Storage | SD Card | FFT data logging |

### Technical Stack

| Layer | Technology |
|-------|-----------|
| Signal Acquisition | Motor current via LT6105, sampled at 48kHz |
| Feature Extraction | FFT (1024-point, Hanning window) — bin1–320 (47Hz–15kHz) |
| Model Training | Sony Neural Network Console (NNC) |
| Model | DNN 320→128→64→32→1 |
| Explainability | SHAP analysis (KernelExplainer), Weight magnitude analysis |
| Firmware | Arduino IDE (C++) |

### Model Performance

| Metric | v1 INA219 | v2 LT6105 |
|--------|-----------|-----------|
| R² | 0.9992 (controlled) | **0.9946 (diverse conditions)** |
| RMSE | 3.1 kPa | **3.18 kPa** |
| MAE | 2.4 kPa | **2.12 kPa** |
| Model Size | ~50 KB | **206 KB** |
| Temperature Robust | ❌ | **✅** |

### Key Finding: Weight Importance ≠ Prediction Contribution

| Analysis Method | Dominant Band | Interpretation |
|----------------|--------------|----------------|
| Weight Magnitude | **14–15kHz** | Strong inter-neuron connections |
| SHAP Analysis | **1.1–1.5kHz** | Dominant actual contribution to output |

**Lesson:** SHAP analysis should always be examined before finalizing 
model input ranges. Weight magnitude and prediction contribution reveal 
fundamentally different aspects of model behavior.

### Data Collection Design (v2)

**Measurement range:** 0.03–0.17 MPa (15 levels, 0.01 MPa steps)  
**Total files:** 7,500 (15 pressures × 5 rounds × 100 files)  
**Train / Validation split:** TEST1–4 (6,000 files) / TEST5 (1,500 files)

Temperature drift was identified as the primary cause of accuracy 
degradation in v1. Data collection for v2 was deliberately designed 
to mix cold-start and warm-running conditions, allowing the model to 
learn temperature-invariant features.

## Inference Without TensorFlow

Due to incompatibility between Python 3.14 / Windows 11 and TensorFlow, 
inference validation is implemented using NumPy only. Weights are 
extracted directly from the `.pb` file using `google.protobuf` and 
saved as `.npz` files — no TensorFlow dependency required at inference 
time.

## Motivation

Japan's industrial sector faces an accelerating labor shortage driven 
by demographic decline. Edge AI and remote monitoring represent the 
most scalable solution — yet adoption across factory floors remains 
remarkably low.

This project was conceived as a proof of concept: that meaningful 
physical quantities can be inferred from signals already present in 
any pumping system, without additional sensor hardware. If this 
approach scales, it could significantly reduce the cost and complexity 
of industrial condition monitoring — including in compact equipment 
where physical sensor installation has traditionally been impractical.

## Future Work

- LTE integration with AWS IoT Core → API Gateway → Lambda → S3
  for remote monitoring (LTE-to-AWS pipeline separately validated
  in a related project)
- Extension to flow rate estimation (dual-output model)
- Closed-loop constant flow control using estimated flow rate as
  feedback signal
- Closed-loop constant pressure control using estimated discharge
  pressure as feedback signal
- Generalization across pump models and operating conditions

## Development

- Independently designed and developed outside of working hours
- Active development period: approximately 6 months
  (v1 INA219 prototype: December 2024;
   v2 LT6105 deployed version: completed 2025)
- Initial concept explored approximately 3 years prior;
  revived and completed after identifying the core technical barrier

## Libraries & References

- [Spresense FFT Library](https://github.com/sonydevworld/spresense-arduino-compatible/blob/master/Arduino15/packages/SPRESENSE/hardware/spresense/1.0.0/libraries/SignalProcessing/src/FFT.h)
- [Spresense Audio Library](https://github.com/sonydevworld/spresense-arduino-compatible/tree/master/Arduino15/packages/SPRESENSE/hardware/spresense/1.0.0/libraries/Audio)
- [Spresense SDHCI Library](https://github.com/sonydevworld/spresense-arduino-compatible/tree/master/Arduino15/packages/SPRESENSE/hardware/spresense/1.0.0/libraries/SDHCI)
- [TI INA219 Product Page](https://www.ti.com/product/INA219)

## Acknowledgements

Special thanks to the engineers at Sony Semiconductor Solutions for 
their technical guidance on the Spresense platform and for their 
encouragement to pursue edge AI development in industrial applications. 
Their feedback was instrumental in validating the direction of this work.

## License

MIT License
