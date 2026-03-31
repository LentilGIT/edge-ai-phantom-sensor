# edge-ai-phantom-sensor

Real-time pump pressure and flow rate estimation using only motor current signals and edge AI — eliminating the need for dedicated sensors in deployment.

## Overview

This project demonstrates that pump discharge pressure and flow rate can be inferred in real time using only motor current signals — without installing any dedicated pressure sensors or flow meters.

Traditional sensor-based monitoring requires physical installation of pressure gauges and flow meters at each pump, which adds cost, maintenance burden, and installation complexity. This system replaces them with a single current measurement and an edge AI model.

## Why Motor Current?

Three signal types were evaluated as input candidates:

| Signal | Problem |
|--------|---------|
| Acoustic | Vulnerable to ambient noise in industrial environments |
| Vibration | Accuracy depends heavily on sensor placement |
| **Motor Current** | **Always measured at the power supply — no installation variability** |

Motor current was selected for its reproducibility and universality. Any pump with an electric motor already has current flowing through a measurable point, making this approach applicable without hardware modifications.

## System Architecture
```
Motor Current Signal
        │
        ▼
[Current Probe] → [Arduino-compatible Microcontroller]
                          │
                    FFT Feature Extraction
                    (Dominant band: 1,400 Hz)
                          │
                    Deep Neural Network
                    (257 → 128 → 64 → 32 → 1)
                          │
                  Real-time Inference
                  Pressure & Flow Rate
                          │
                    LTE Communication
                          │
                    AWS IoT Core
                    → API Gateway
                    → Lambda
                    → S3
```

## Technical Stack

| Layer | Technology |
|-------|-----------|
| Hardware | Arduino-compatible microcontroller (Arm Cortex-M4F) |
| Signal Acquisition | Motor current via current probe, sampled at 48 kHz |
| Feature Extraction | FFT — dominant frequency band identified at 1,400 Hz |
| Model | Deep Neural Network (DNN) |
| Explainability | SHAP analysis, Integrated Gradients |
| Cloud | AWS IoT Core → API Gateway → Lambda → S3 |
| Connectivity | LTE |

## Results

| Metric | Value |
|--------|-------|
| R² (pressure & flow estimation) | **0.9945** |
| Inference | Real-time, on-device |
| Sensor requirement at deployment | None |

SHAP analysis confirmed that the 1,400 Hz frequency band is the dominant contributor to model predictions, providing physical interpretability alongside predictive accuracy.

## Motivation

Japan's industrial sector faces an accelerating labor shortage driven by demographic decline. Edge AI and remote monitoring represent the most scalable solution — yet adoption across factory floors remains remarkably low.

This project was conceived as a proof of concept: that meaningful physical quantities can be inferred from signals already present in any pumping system, without additional sensor hardware. If this approach scales, it could significantly reduce the cost and complexity of industrial condition monitoring.

## Development

- Independently designed and developed outside of working hours
- Total active development period: approximately 6 months
- Conceptual development: approximately 3 years

## License

MIT License
