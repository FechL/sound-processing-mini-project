from pathlib import Path 
import json 
import math 
import os 
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.io import wavfile 
from scipy.signal import resample_poly 
  
ROOT = Path(__file__).resolve().parent 
CONFIG_PATH = ROOT / "config/project_config.json" 
STATUS_PATH = ROOT / "project_status.json" 
  
# ------------------------------------------------------------ 
# 1. VALIDASI HANDOFF MINI PROJECT 1 
# ------------------------------------------------------------ 
with open(CONFIG_PATH, "r", encoding="utf-8") as f: 
    config = json.load(f) 
with open(STATUS_PATH, "r", encoding="utf-8") as f: 
    status = json.load(f) 
  
if status.get("status") != "ready_for_mp02": 
    raise RuntimeError("Mini Project 1 belum berstatus ready_for_mp02") 
  
input_wav = ROOT / config["reference_file"] 
if not input_wav.exists(): 
    raise FileNotFoundError(input_wav) 
  
fs, raw = wavfile.read(input_wav) 
if raw.ndim > 1: 
    raw = raw.mean(axis=1) 
  
# Konversi input PCM menjadi float sekitar [-1, 1] 
if raw.dtype == np.uint8: 
    x = (raw.astype(np.float32) - 128.0) / 128.0 
elif np.issubdtype(raw.dtype, np.integer): 
    scale = float(max(abs(np.iinfo(raw.dtype).min), np.iinfo(raw.dtype).max)) 
    x = raw.astype(np.float32) / scale 
else: 
    x = raw.astype(np.float32) 
x = np.clip(x, -1.0, 1.0) 
  
OUT = ROOT / "data/processed" 
FIG = ROOT / "results/figures" 
MET = ROOT / "results/metrics" 
CFG = ROOT / "config" 
for p in [OUT, FIG, MET, CFG]: 
    p.mkdir(parents=True, exist_ok=True) 
  
# ------------------------------------------------------------ 
# 2. RESAMPLING: 16 kHz -> 8 kHz 
# ------------------------------------------------------------ 
target_fs = 8000 
g = math.gcd(int(fs), target_fs) 
x_8k = resample_poly(x, target_fs // g, int(fs) // g) 
x_8k16 = np.int16(np.clip(x_8k, -1, 1) * 32767) 
path_8k16 = OUT / "mp02_nyala_8k16.wav" 
wavfile.write(path_8k16, target_fs, x_8k16) 
  
# ------------------------------------------------------------ 
# 3. KUANTISASI 8-BIT PADA 16 kHz 
# WAV 8-bit menggunakan unsigned integer 0..255 
# ------------------------------------------------------------ 
def quantize_u8(signal): 
    return np.round((np.clip(signal, -1, 1) + 1.0) * 127.5).astype(np.uint8) 
  
def dequantize_u8(q): 
    return q.astype(np.float32) / 127.5 - 1.0 
  
q_16k8 = quantize_u8(x) 
path_16k8 = OUT / "mp02_nyala_16k8.wav" 
wavfile.write(path_16k8, int(fs), q_16k8) 
  
q_8k8 = quantize_u8(x_8k) 
path_8k8 = OUT / "mp02_nyala_8k8.wav" 
wavfile.write(path_8k8, target_fs, q_8k8) 
  
# ------------------------------------------------------------ 
# 4. METRIK SEDERHANA 
# ------------------------------------------------------------ 
def snr_db(reference, test): 
    n = min(len(reference), len(test)) 
    reference = reference[:n] 
    test = test[:n] 
    noise = reference - test 
    ps = np.mean(reference ** 2) + 1e-12 
    pn = np.mean(noise ** 2) + 1e-12 
    return float(10.0 * np.log10(ps / pn)) 
  
x_q8 = dequantize_u8(q_16k8) 
quant_snr = snr_db(x, x_q8) 
  
# Resample 8 kHz kembali ke fs asli untuk perbandingan numerik kasar 
g2 = math.gcd(target_fs, int(fs)) 
x_8k_back = resample_poly(x_8k, int(fs) // g2, target_fs // g2) 
resample_snr = snr_db(x, x_8k_back) 
  
files = { 
    "16k16_baseline": input_wav, 
    "8k16": path_8k16, 
    "16k8": path_16k8, 
    "8k8": path_8k8, 
} 
sizes = {k: os.path.getsize(v) for k, v in files.items()} 
base_size = sizes["16k16_baseline"] 
reduction = { 
    k: float(100.0 * (1.0 - v / base_size)) 
    for k, v in sizes.items() 
} 
  
metrics = { 
    "source_file": str(input_wav.relative_to(ROOT)), 
    "source_sample_rate_hz": int(fs), 
    "duration_s": float(len(x) / fs), 
    "file_size_bytes": sizes, 
    "size_reduction_percent_vs_baseline": reduction, 
    "quantization_8bit_snr_db": quant_snr, 
    "resample_8k_roundtrip_snr_db": resample_snr, 
    "selected_format": { 
        "sample_rate_hz": 16000, 
        "channels": 1, 
        "bit_depth": 16, 
        "reason": "Baseline proyek: menjaga informasi ucapan dan memudahkan analisis DSP/AI berikutnya" 
    } 
} 
with open(MET / "mp02_sampling_quantization.json", "w", encoding="utf-8") as f: 
    json.dump(metrics, f, indent=2) 
  
selected = { 
    "sample_rate_hz": 16000, 
    "channels": 1, 
    "bit_depth": 16, 
    "reference_file": config["reference_file"], 
    "selected_by_mini_project": 2 
} 
with open(CFG / "audio_format_selected.json", "w", encoding="utf-8") as f: 
    json.dump(selected, f, indent=2) 
  
# Simpan juga keputusan ke project_config tanpa menghapus konfigurasi lama 
config["audio_format_selected"] = selected 
with open(CONFIG_PATH, "w", encoding="utf-8") as f: 
    json.dump(config, f, indent=2) 
  
# ------------------------------------------------------------ 
# 5. VISUALISASI RINGKAS 
# ------------------------------------------------------------ 
fig, ax = plt.subplots(3, 1, figsize=(9, 8)) 
  
# Waveform pendek agar langkah kuantisasi terlihat 
n_show = min(int(0.04 * fs), len(x)) 
t = np.arange(n_show) / fs * 1000.0 
ax[0].plot(t, x[:n_show], label="16-bit", linewidth=1.0) 
ax[0].step(t, x_q8[:n_show], where="mid", label="8-bit", linewidth=0.8) 
ax[0].set_title("Efek Kuantisasi pada Potongan Waveform") 
ax[0].set_xlabel("Waktu (ms)") 
ax[0].set_ylabel("Amplitudo") 
ax[0].legend() 
ax[0].grid(alpha=0.25) 
  
# Spektrum baseline dan hasil 8 kHz 
def spectrum(sig, sr): 
    w = np.hanning(len(sig)) 
    X = np.abs(np.fft.rfft(sig * w)) 
    f = np.fft.rfftfreq(len(sig), 1.0 / sr) 
    Xdb = 20 * np.log10(X / (np.max(X) + 1e-12) + 1e-12) 
    return f, Xdb 
  
f16, s16 = spectrum(x, fs) 
f8, s8 = spectrum(x_8k, target_fs) 
ax[1].plot(f16, s16, label="16 kHz") 
ax[1].plot(f8, s8, label="8 kHz") 
ax[1].set_xlim(0, fs / 2) 
ax[1].set_ylim(-80, 5) 
ax[1].set_title("Spektrum: Batas Nyquist Berubah saat Downsampling") 
ax[1].set_xlabel("Frekuensi (Hz)") 
ax[1].set_ylabel("Magnitudo relatif (dB)") 
ax[1].legend() 
ax[1].grid(alpha=0.25) 
  
labels = ["16k/16", "8k/16", "16k/8", "8k/8"] 
values = [sizes[k] / 1024 for k in ["16k16_baseline", "8k16", "16k8", "8k8"]] 
ax[2].bar(labels, values) 
ax[2].set_title("Perbandingan Ukuran File") 
ax[2].set_ylabel("Ukuran (KiB)") 
ax[2].grid(axis="y", alpha=0.25) 
  
fig.tight_layout() 
fig.savefig(FIG / "mp02_sampling_quantization.png", dpi=160) 
plt.close(fig) 
  
# ------------------------------------------------------------ 
# 6. HANDOFF KE MINI PROJECT 3 
# ------------------------------------------------------------ 
status = { 
    "last_completed_mini_project": 2, 
    "next_mini_project": 3, 
    "next_input": config["reference_file"], 
    "audio_format": "config/audio_format_selected.json", 
    "metrics": "results/metrics/mp02_sampling_quantization.json", 
    "status": "ready_for_mp03" 
} 
with open(STATUS_PATH, "w", encoding="utf-8") as f: 
    json.dump(status, f, indent=2) 
  
print("Mini Project 2 selesai") 
print("SNR kuantisasi 8-bit      :", round(quant_snr, 2), "dB") 
print("SNR resampling 8 kHz      :", round(resample_snr, 2), "dB") 
print("Format dipilih            : 16 kHz / mono / 16-bit") 
print("Status berikutnya         : ready_for_mp03") 