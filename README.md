# edge-ai-phantom-sensor

Real-time pump pressure and flow rate estimation using only motor current 
signals and edge AI — eliminating the need for dedicated sensors in 
deployment.

## Overview

This project demonstrates that pump discharge pressure can be inferred 
in real time using only motor current signals — without installing any 
dedicated pressure sensors or flow meters.

Traditional sensor-based monitoring requires physical installation of 
pressure gauges and flow meters at each pump, which adds cost, 
maintenance burden, and installation complexity. This is especially 
challenging in compact equipment where physical space for sensor 
installation is severely limited. This system replaces dedicated sensors 
with a single current measurement and an edge AI model running entirely 
on a microcontroller — making it applicable even where conventional 
sensor installation is impractical.

## Why Motor Current?

Three signal types were evaluated as input candidates:

| Signal | Problem |
|--------|---------|
| Acoustic | Vulnerable to ambient noise in industrial environments |
| Vibration | Accuracy depends heavily on sensor placement |
| **Motor Current** | **Always measured at the power supply — no installation variability** |

Motor current was selected for its reproducibility and universality. 
Critically, the **frequency domain** of the current signal is stable across 
temperature changes, whereas amplitude and DC current drift significantly 
with motor temperature — making FFT-based features far more robust than 
raw current values for long-term deployment.

## System Architecture
```
Pump Motor Current
        │
        ▼
[LT6105 Current Sensor] → [Voltage Divider] → [Sony Spresense MIC_A Pin]
                                                        │
                                               FFT (1024-point, Hanning)
                                               48kHz sampling
                                               bin1–320 (47Hz–15kHz)
                                                        │
                                               Deep Neural Network
                                               (320→128→64→32→1)
                                                        │
                                             Real-time Pressure Output
                                                   [MPa]
                                                        │
                                            ILI9341 Display (SPI)
```

## Hardware

| Component | Part | Role |
|-----------|------|------|
| Microcontroller | Sony Spresense (Arm Cortex-M4F) | FFT + DNN inference + display |
| Current Sensor | LT6105 | Motor current measurement (1V/1A output) |
| Display | ILI9341 (SPI) | Real-time pressure display |
| Storage | SD Card | FFT data logging |

## Technical Stack

| Layer | Technology |
|-------|-----------|
| Hardware | Sony Spresense (Arm Cortex-M4F) |
| Signal Acquisition | Motor current via LT6105, sampled at 48kHz |
| Feature Extraction | FFT (1024-point, Hanning window) — bin1–320 (47Hz–15kHz) |
| Model Training | Sony Neural Network Console (NNC) |
| Model | Deep Neural Network (DNN) 320→128→64→32→1 |
| Explainability | SHAP analysis (KernelExplainer), Weight magnitude analysis |
| Firmware | Arduino IDE (C++) |

## Model Performance

| Metric | Value |
|--------|-------|
| R² | **0.9946** |
| RMSE | 3.18 kPa |
| MAE | 2.12 kPa |
| Max Error | 12.75 kPa |
| Model Size | 206 KB |
| Inference | Real-time, on-device |
| Sensor requirement at deployment | None |

## Key Finding: Weight Importance ≠ Prediction Contribution

One of the most important insights from this project came from comparing 
two analysis methods:

| Analysis Method | Dominant Band | Interpretation |
|----------------|--------------|----------------|
| Weight Magnitude | **14–15kHz** | Strong inter-neuron connections |
| SHAP Analysis | **1.1–1.5kHz** | Dominant actual contribution to output |

The 14–15kHz band (hypothesized to be the motor driver PWM switching 
frequency) shows large weights but small actual output contribution due 
to weak signal amplitude. The 1.1–1.5kHz band (pump rotation frequency) 
is the primary physical driver of pressure estimation — but this would 
not have been identified from weight analysis alone.

**Lesson:** SHAP analysis should always be examined before finalizing 
model input ranges. Weight magnitude and prediction contribution reveal 
fundamentally different aspects of model behavior.

## Data Collection Design

**Measurement range:** 0.03–0.17 MPa (15 levels, 0.01 MPa steps)  
**Total files:** 7,500 (15 pressures × 5 rounds × 100 files)  
**Train / Validation split:** TEST1–4 (6,000 files) / TEST5 (1,500 files)

Temperature drift was identified as the primary cause of accuracy 
degradation in earlier models. Data collection was deliberately designed 
to mix cold-start and warm-running conditions across rounds, allowing the 
model to learn temperature-invariant features.

## Data

FFT CSV data used for SHAP analysis is included as `data/TEST5.zip`.  
Unzip before running the analysis scripts:
```
data/
└── TEST5/
    └── *.csv
```

The Python scripts assume the following relative path:
```python
CSV_FOLDER = "../data/TEST5"
```

## Inference Without TensorFlow

Due to incompatibility between Python 3.14 / Windows 11 and TensorFlow, 
inference validation is implemented using NumPy only. Weights are 
extracted directly from the `.pb` file using `google.protobuf` and 
saved as `.npz` files — no TensorFlow dependency required at inference 
time.

## Motivation

Japan's industrial sector faces an accelerating labor shortage driven by 
demographic decline. Edge AI and remote monitoring represent the most 
scalable solution — yet adoption across factory floors remains remarkably 
low.

This project was conceived as a proof of concept: that meaningful 
physical quantities can be inferred from signals already present in any 
pumping system, without additional sensor hardware. If this approach 
scales, it could significantly reduce the cost and complexity of 
industrial condition monitoring — including in compact equipment where 
physical sensor installation has traditionally been impractical.

## Future Work

- LTE integration with AWS IoT Core → API Gateway → Lambda → S3 
  for remote monitoring (LTE-to-AWS pipeline separately validated 
  in a related project)
- Extension to flow rate estimation (dual-output model)
- Closed-loop constant flow control using estimated flow rate as 
  feedback signal — eliminating the need for a dedicated flow meter 
  in control applications
- Closed-loop constant pressure control using estimated discharge 
  pressure as feedback signal — enabling sensorless pressure regulation
- Generalization across pump models and operating conditions

## Development

- Independently designed and developed outside of working hours
- Active development period: approximately 6 months
- (Initial concept explored approximately 3 years prior;
  revived and completed after identifying the core technical barrier)

## Libraries & References

This project is built on the following Sony Spresense libraries:

- [Spresense FFT Library](https://github.com/sonydevworld/spresense-arduino-compatible/blob/master/Arduino15/packages/SPRESENSE/hardware/spresense/1.0.0/libraries/SignalProcessing/src/FFT.h)
- [Spresense Audio Library](https://developer.spresense.sony-semicon.com/spresense-api-references-arduino/Audio_8h_source)
- [Spresense SDHCI Library](https://github.com/sonydevworld/spresense-arduino-compatible/blob/master/Arduino15/packages/SPRESENSE/hardware/spresense/1.0.0/libraries/SDHCI/src/SDHCI.h)

## Acknowledgements

Special thanks to the engineers at Sony Semiconductor Solutions for 
their technical guidance on the Spresense platform and for their 
encouragement to pursue edge AI development in industrial applications. 
Their feedback was instrumental in validating the direction of this work.

## License

MIT License
