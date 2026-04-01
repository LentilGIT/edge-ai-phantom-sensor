"""
shap_analysis_512bin.py

SHAP-based input importance analysis for the 512-bin comparison DNN model.

Model architecture : 512 -> 256 -> 128 -> 32 -> 1
Input range        : FFT bin0 – bin511 (0Hz – 23.95kHz)
Sampling rate      : 48kHz, 1024-point Hanning FFT
Frequency resolution: 46.875 Hz/bin

This script is used as a comparison against the deployed 320-bin model
(shap_analysis_320bin.py). Both models show consistent SHAP results:
the 1.1–1.5kHz band dominates actual prediction contribution in both cases,
confirming that the model has learned physically meaningful features
regardless of input dimensionality.

Key finding:
    Despite the 14–15kHz band showing large weights in weight magnitude
    analysis, SHAP analysis confirms that the 1.1–1.5kHz band
    (pump rotation frequency) is the primary driver of pressure estimation
    in both the 320-bin and 512-bin models.

Dependencies:
    pip install numpy matplotlib shap

Usage:
    1. Place model_weights_512bin.npz in the same folder as this script.
    2. Set CSV_FOLDER to the directory containing FFT CSV files (TEST5 data).
    3. Run: python shap_analysis_512bin.py
    4. Output: shap_result_512bin.png saved in the same folder.
"""

import numpy as np
import matplotlib.pyplot as plt
import shap
import os
import glob

# ── Configuration ─────────────────────────────────────────────────────────────
NPZ_PATH   = "model_weights_512bin.npz"   # Path to extracted model weights
CSV_FOLDER = "../data/TEST5"              # Path to FFT CSV data folder
MAX_FILES  = 1500                         # Maximum number of CSV files to load
# ──────────────────────────────────────────────────────────────────────────────

# Load model weights from .npz
w = np.load(NPZ_PATH)

def relu(x):
    return np.maximum(0, x)

def batch_norm(x, mean, var, gamma, beta, eps=1e-3):
    """Batch normalization: gamma * (x - mean) / sqrt(var + eps) + beta"""
    return gamma * (x - mean) / np.sqrt(var + eps) + beta

def predict(X):
    """
    TensorFlow-free inference using NumPy.
    Architecture: 512 -> 256 (BN+ReLU) -> 128 (BN+ReLU) -> 32 (ReLU) -> 1
    Input X shape: (n_samples, 512)
    Output shape : (n_samples, 1)
    """
    X = np.array(X, dtype=np.float32)
    # Layer 1: 512 -> 256 with BatchNorm and ReLU
    h = X @ w['W1'] + w['b1']
    h = batch_norm(h, w['bn1_mean'], w['bn1_var'], w['bn1_gamma'], w['bn1_beta'])
    h = relu(h)
    # Layer 2: 256 -> 128 with BatchNorm and ReLU
    h = h @ w['W2'] + w['b2']
    h = batch_norm(h, w['bn2_mean'], w['bn2_var'], w['bn2_gamma'], w['bn2_beta'])
    h = relu(h)
    # Layer 3: 128 -> 32 with ReLU
    h = h @ w['W3'] + w['b3']
    h = relu(h)
    # Layer 4: 32 -> 1 (regression output)
    h = h @ w['W4'] + w['b4']
    return h

# Load CSV files (bin0–511: full spectrum including DC component)
csv_files = sorted(glob.glob(os.path.join(CSV_FOLDER, "*.csv")))[:MAX_FILES]
print(f"Loading {len(csv_files)} files...")

data = []
for f in csv_files:
    d = np.loadtxt(f)
    data.append(d[0:512])  # bin0 to bin511 (full 512-bin spectrum)
X = np.array(data, dtype=np.float32)
print(f"Data shape: {X.shape}")

# Inference test — output should be in range 0.03–0.17 MPa
pred = predict(X[:5])
print(f"Inference test: {pred.flatten()} MPa")

# SHAP computation (may take several minutes)
print("\nComputing SHAP values...")
print("  Background: k-means with 20 clusters")
print("  Samples for explanation: 50 files x 500 iterations")
explainer  = shap.KernelExplainer(predict, shap.kmeans(X, 20))
shap_values = explainer.shap_values(X[:50], nsamples=500)
shap_arr   = np.array(shap_values).squeeze()

# Frequency axis: bin0–511 → 0Hz–23.95kHz
freq_res = 48000 / 1024          # 46.875 Hz/bin
freqs    = np.arange(0, 512) * freq_res / 1000  # kHz

mean_abs_shap = np.mean(np.abs(shap_arr), axis=0)
top10_idx     = np.argsort(mean_abs_shap)[::-1][:10]

print("\n=== Global Top 10 bins by SHAP importance ===")
for rank, idx in enumerate(top10_idx, 1):
    print(f"  Rank {rank:2d}  {freqs[idx]:.3f} kHz  SHAP = {mean_abs_shap[idx]:.6f}")

# Plot
plt.figure(figsize=(14, 6))
plt.plot(freqs, mean_abs_shap, color='steelblue', lw=0.8, label='Mean |SHAP value|')

# Highlight key frequency bands
plt.axvspan(1.1,  1.5,  alpha=0.15, color='red',    label='1.1–1.5 kHz  (pump rotation band)')
plt.axvspan(14.0, 15.0, alpha=0.15, color='orange',  label='14–15 kHz   (high weight zone)')

# Mark Top 10 bins
for idx in top10_idx:
    plt.plot(freqs[idx], mean_abs_shap[idx], 'ro', ms=7)
    plt.annotate(f'{freqs[idx]:.2f} kHz',
                 xy=(freqs[idx], mean_abs_shap[idx]),
                 xytext=(2, 3), textcoords='offset points',
                 fontsize=8, color='darkred')

plt.title('SHAP Value Analysis — Input Importance (bin0–511, 0Hz–23.95kHz)\n'
          'Comparison Model: 512→256→128→32→1  R²=0.9945  RMSE=3.20kPa')
plt.xlabel('Frequency (kHz)')
plt.ylabel('Mean |SHAP value|')
plt.legend(fontsize=9)
plt.grid(True, alpha=0.3)
plt.tight_layout()

output_path = 'shap_result_512bin.png'
plt.savefig(output_path, dpi=150)
plt.show()
print(f"\nSaved: {output_path}")
