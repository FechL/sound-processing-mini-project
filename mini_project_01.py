from pathlib import Path 
import json 
import numpy as np 
import matplotlib.pyplot as plt 
import sounddevice as sd 
from scipy.io import wavfile 
  
# ------------------------------------------------------------  
# MINI PROJECT 1 - INISIALISASI PROYEK KEYWORD SPOTTING 
# ------------------------------------------------------------  
ROOT = Path(__file__).resolve().parent 
DIRS = [ 
    "config", "data/raw", "data/processed", "features", 
    "models", "results/figures", "results/metrics", "firmware" 
] 
for folder in DIRS: 
    (ROOT / folder).mkdir(parents=True, exist_ok=True) 
  
CONFIG = { 
    "project_name": "embedded_keyword_spotting",  
    "labels": ["nyala", "mati", "naik", "turun"], 
    "sample_rate_hz": 16000, 
    "channels": 1, 
    "bit_depth": 16, 
    "record_duration_s": 2.0, 
    "reference_file": "data/raw/nyala_001.wav" 
} 
  
config_path = ROOT / "config/project_config.json"  
with open(config_path, "w", encoding="utf-8") as f: 
    json.dump(CONFIG, f, indent=2) 
  
fs = CONFIG["sample_rate_hz"]  
duration = CONFIG["record_duration_s"] 
output_wav = ROOT / CONFIG["reference_file"] 
  
print("Mini Project 1 - Keyword Spotting Embedded") 
print("Katakan kata: NYALA") 
input("Tekan ENTER ketika siap merekam...") 
  
# sounddevice menghasilkan float32 pada rentang sekitar -1 sampai +1 
recording = sd.rec( 
    int(duration * fs), samplerate=fs, channels=1, dtype="float32" 
) 
sd.wait() 
x = recording[:, 0] 
  
# Pemeriksaan kualitas dasar sebelum konversi ke PCM-16 
peak = float(np.max(np.abs(x))) 
rms = float(np.sqrt(np.mean(x ** 2))) 
clipping_ratio = float(np.mean(np.abs(x) >= 0.99)) 
  
# Konversi float32 [-1,1] ke PCM signed 16-bit 
x_pcm16 = np.int16(np.clip(x, -1.0, 1.0) * 32767) 
wavfile.write(output_wav, fs, x_pcm16) 
  
metadata = { 
    "file": str(output_wav.relative_to(ROOT)), 
    "sample_rate_hz": fs, 
    "channels": 1, 
    "bit_depth": 16, 
    "duration_s": len(x) / fs, 
    "num_samples": int(len(x)), 
    "peak_normalized": peak, 
    "rms_normalized": rms, 
    "clipping_ratio": clipping_ratio, 
    "quality_check": "PASS" if peak > 0.02 and clipping_ratio == 0.0 else "CHECK" 
} 
  
metadata_path = ROOT / "results/metrics/mp01_audio_metadata.json" 
with open(metadata_path, "w", encoding="utf-8") as f: 
    json.dump(metadata, f, indent=2) 
  
# Waveform untuk inspeksi visual 
axis_t = np.arange(len(x)) / fs 
plt.figure(figsize=(9, 3)) 
plt.plot(axis_t, x, linewidth=0.8) 
plt.xlabel("Waktu (s)") 
plt.ylabel("Amplitudo ternormalisasi") 
plt.title("Mini Project 1 - Waveform Kata 'nyala'") 
plt.grid(alpha=0.25) 
plt.tight_layout() 
figure_path = ROOT / "results/figures/mp01_waveform.png" 
plt.savefig(figure_path, dpi=160) 
plt.close() 
  
status = { 
    "last_completed_mini_project": 1, 
    "next_mini_project": 2, 
    "next_input": CONFIG["reference_file"], 
    "config": "config/project_config.json", 
    "status": "ready_for_mp02" 
} 
with open(ROOT / "project_status.json", "w", encoding="utf-8") as f: 
    json.dump(status, f, indent=2) 
  
print("\nSelesai.") 
print("Audio     :", output_wav) 
print("Metadata  :", metadata_path)  
print("Waveform  :", figure_path) 
print("Peak      :", round(peak, 4)) 
print("RMS       :", round(rms, 4)) 
print("Clipping  :", clipping_ratio) 
print("Check     :", metadata["quality_check"]) 