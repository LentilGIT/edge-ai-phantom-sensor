"""
shap_analysis_INA219.py

SHAP-based input importance analysis for the INA219-based DNN model.

Model architecture : 257 -> 128 -> 64 -> 32 -> 1
Input layout       :
  index 0          : DC mean current [mA]  (raw, not normalized)
  index 1          : Peak-to-Peak amplitude [mA]  (raw, not normalized)
  index 2–256      : FFT bin1–bin255 (Hamming window, 512-point FFT)

Sensor             : TI INA219 (I2C 400kHz, 10-bit ADC, config 0x398F)
FFT                : 512-point Hamming window
Model performance  : R²=0.9992, MAE=0.0024 MPa

This script loads pre-extracted model weights (.npz) and INA219 CSV data,
computes SHAP values using KernelExplainer, and visualizes which inputs
actually contribute to pressure estimation.

Key question:
    The INA219 model includes DC mean and Peak-to-Peak as inputs.
    These are known to be temperature-sensitive (drift with motor temperature).
    SHAP analysis will reveal how much they actually contribute vs FFT features.

Dependencies:
    pip install numpy matplotlib shap

Usage:
    1. Place model_weights_INA219.npz in the same folder as this script.
    2. Set CSV_FOLDER to the directory containing INA219 CSV files.
    3. Run: python shap_analysis_INA219.py
    4. Output: shap_result_INA219.png saved in the same folder.

CSV format expected:
    dc_mean, peak_to_peak, bin1, bin2, ..., bin255
    (257 columns: 2 statistics + 255 FFT bins, raw values, no normalization)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import shap
import os
import glob

# ── Configuration ─────────────────────────────────────────────────────────────
NPZ_PATH   = r"D:\OneDrive\edge-ai-phantom-sensor\v1_INA219\analysis\model_weights_INA219.npz"  # Path to extracted model weights
CSV_FOLDER = r"D:\OneDrive\edge-ai-phantom-sensor\v1_INA219\data\INA219_valid"            # Path to INA219 CSV data folder
MAX_FILES  = 320                         # Maximum number of CSV files to load
# ──────────────────────────────────────────────────────────────────────────────

# Load model weights
w = np.load(NPZ_PATH)

def relu(x):
    return np.maximum(0, x)

def batch_norm(x, mean, var, gamma, beta, eps=1e-3):
    """Batch normalization: gamma * (x - mean) / sqrt(var + eps) + beta"""
    return gamma * (x - mean) / np.sqrt(var + eps) + beta

def predict(X):
    """
    TensorFlow-free inference using NumPy.
    Architecture: 257 -> 128 (BN+ReLU) -> 64 (BN+ReLU) -> 32 (ReLU) -> 1

    Input layout:
      X[:, 0]     : DC mean [mA]
      X[:, 1]     : Peak-to-Peak [mA]
      X[:, 2-256] : FFT bin1-255

    IMPORTANT: pass raw (non-normalized) values — matches training data format
    """
    X = np.array(X, dtype=np.float32)
    h = X @ w['W1'] + w['b1']
    h = batch_norm(h, w['bn1_mean'], w['bn1_var'], w['bn1_gamma'], w['bn1_beta'])
    h = relu(h)
    h = h @ w['W2'] + w['b2']
    h = batch_norm(h, w['bn2_mean'], w['bn2_var'], w['bn2_gamma'], w['bn2_beta'])
    h = relu(h)
    h = h @ w['W3'] + w['b3']
    h = relu(h)
    h = h @ w['W4'] + w['b4']
    return h

# Load CSV files (257 columns: dc_mean, peak_to_peak, bin1..bin255)
csv_files = sorted(glob.glob(os.path.join(CSV_FOLDER, "*.csv")))[:MAX_FILES]
print(f"Loading {len(csv_files)} files...")

data = []
for f in csv_files:
    d = np.loadtxt(f, delimiter=',')
    # Each row: [dc_mean, peak_to_peak, bin1, ..., bin255]
    data.append(d)
X = np.array(data, dtype=np.float32)
print(f"Data shape: {X.shape}")  # Expected: (n_files, 257)

# Inference test — output should be in pressure range of training data
pred = predict(X[:5])
print(f"Inference test: {pred.flatten()} MPa")

# SHAP computation
print("\nComputing SHAP values...")
print("  Background: k-means with 20 clusters")
print("  Samples for explanation: 50 files x 500 iterations")
explainer   = shap.KernelExplainer(predict, shap.kmeans(X, 20))
shap_values = explainer.shap_values(X[:50], nsamples=500)
shap_arr    = np.array(shap_values).squeeze()

mean_abs_shap = np.mean(np.abs(shap_arr), axis=0)  # shape: (257,)

# Input labels:
# index 0: DC mean, index 1: Peak-to-Peak, index 2-256: FFT bin1-255
labels = ['DC mean', 'Peak-to-Peak'] + [f'bin{i}' for i in range(1, 256)]

top10_idx = np.argsort(mean_abs_shap)[::-1][:10]

print("\n=== Global Top 10 inputs by SHAP importance ===")
for rank, idx in enumerate(top10_idx, 1):
    label = labels[idx]
    print(f"  Rank {rank:2d}  [{idx:3d}] {label:15s}  SHAP = {mean_abs_shap[idx]:.6f}")

# Separate SHAP values for plotting
shap_dc_pp  = mean_abs_shap[:2]    # DC mean, Peak-to-Peak
shap_fft    = mean_abs_shap[2:]    # FFT bin1-255 (253 values... wait, bin1-255 = 255 values = index 2-256)

# Note: INA219 effective sampling rate via I2C 400kHz
# Actual sample rate depends on INA219 ADC setting (10-bit: 148us/sample)
# Effective rate ≈ 1 / 148us ≈ 6.76 kHz
# FFT bin frequency: bin_n = n * sample_rate / FFT_LEN
EFFECTIVE_SAMPLE_RATE = 6757  # Hz (1 / 148e-6)
freq_res = EFFECTIVE_SAMPLE_RATE / 512  # Hz/bin ≈ 13.2 Hz/bin
freqs_fft = np.arange(1, 256) * freq_res / 1000  # kHz, bin1-255

# ── Plot ──────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 10))
gs  = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1, 2], hspace=0.4, wspace=0.35)
fig.suptitle('SHAP Value Analysis — INA219 DNN Model\n'
             'Architecture: 257→128→64→32→1  R²=0.9992  MAE=0.0024 MPa',
             fontsize=13, fontweight='bold')

# ── Panel 1: DC mean & Peak-to-Peak SHAP ─────────────────────────────────────
ax0 = fig.add_subplot(gs[0, 0])
bars = ax0.bar(['DC mean\n[mA]', 'Peak-to-Peak\n[mA]'],
               shap_dc_pp,
               color=['steelblue', 'salmon'],
               edgecolor='grey', linewidth=0.5)
for bar, val in zip(bars, shap_dc_pp):
    ax0.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001,
             f'{val:.5f}', ha='center', va='bottom', fontsize=9)
ax0.set_title('Non-FFT Features\n(DC mean & Peak-to-Peak)', fontsize=10)
ax0.set_ylabel('Mean |SHAP value|')
ax0.grid(True, alpha=0.3, axis='y')

# ── Panel 2: Top10 bar chart ──────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 1])
top10_labels = [labels[i] for i in top10_idx]
top10_shap   = [mean_abs_shap[i] for i in top10_idx]
colors = ['salmon' if i < 2 else 'steelblue' for i in top10_idx]
bars2 = ax1.barh(range(10), top10_shap[::-1], color=colors[::-1],
                  edgecolor='grey', linewidth=0.5)
ax1.set_yticks(range(10))
ax1.set_yticklabels([f'[{top10_idx[9-i]}] {top10_labels[9-i]}' for i in range(10)],
                     fontsize=8)
ax1.set_title('Global Top 10 Inputs\n(Red = non-FFT feature)', fontsize=10)
ax1.set_xlabel('Mean |SHAP value|')
ax1.grid(True, alpha=0.3, axis='x')

# ── Panel 3: FFT spectrum SHAP (full range) ───────────────────────────────────
ax2 = fig.add_subplot(gs[1, :])
ax2.plot(freqs_fft, shap_fft, color='steelblue', lw=0.8, label='Mean |SHAP| (FFT bins)')

# Mark Top10 FFT bins (exclude non-FFT indices 0,1)
fft_top10_idx = [i for i in top10_idx if i >= 2][:10]
for idx in fft_top10_idx:
    fft_bin_pos = idx - 2  # position in shap_fft array
    freq = freqs_fft[fft_bin_pos]
    val  = mean_abs_shap[idx]
    ax2.plot(freq, val, 'ro', ms=7)
    ax2.annotate(f'{freq:.2f}kHz',
                 xy=(freq, val),
                 xytext=(2, 3), textcoords='offset points',
                 fontsize=8, color='darkred')

# Reference lines for DC mean and P-P SHAP levels
ax2.axhline(y=shap_dc_pp[0], color='orange', lw=1, linestyle='--',
            label=f'DC mean SHAP = {shap_dc_pp[0]:.5f}')
ax2.axhline(y=shap_dc_pp[1], color='salmon', lw=1, linestyle='--',
            label=f'Peak-to-Peak SHAP = {shap_dc_pp[1]:.5f}')

ax2.set_title('FFT Bin SHAP Values (bin1–255)\n'
              f'Effective sample rate ≈ {EFFECTIVE_SAMPLE_RATE:.0f} Hz  |  '
              f'Freq resolution ≈ {freq_res:.1f} Hz/bin',
              fontsize=10)
ax2.set_xlabel('Frequency (kHz)')
ax2.set_ylabel('Mean |SHAP value|')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

output_path = 'shap_result_INA219.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"\nSaved: {output_path}")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n=== Summary ===")
print(f"DC mean SHAP     : {shap_dc_pp[0]:.6f}")
print(f"Peak-to-Peak SHAP: {shap_dc_pp[1]:.6f}")
print(f"FFT max SHAP     : {shap_fft.max():.6f}  (at {freqs_fft[shap_fft.argmax()]:.3f} kHz)")
print(f"FFT mean SHAP    : {shap_fft.mean():.6f}")
