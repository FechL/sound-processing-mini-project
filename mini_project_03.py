from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import stft

ROOT = Path(__file__).resolve().parent
STATUS_PATH = ROOT / "project_status.json"
FORMAT_PATH = ROOT / "config/audio_format_selected.json"
INPUT_WAV = ROOT / "data/raw/nyala_001.wav"

# 1. Validasi handoff Mini Project 2
with open(STATUS_PATH, "r", encoding="utf-8") as f:
    status = json.load(f)
if status.get("status") != "ready_for_mp03":
    raise RuntimeError("Mini Project 2 belum berstatus ready_for_mp03")

with open(FORMAT_PATH, "r", encoding="utf-8") as f:
    audio_format = json.load(f)

fs, raw = wavfile.read(INPUT_WAV)
if raw.ndim > 1:
    raw = raw.mean(axis=1)

if np.issubdtype(raw.dtype, np.integer):
    scale = float(max(abs(np.iinfo(raw.dtype).min), np.iinfo(raw.dtype).max))
    x = raw.astype(np.float32) / scale
else:
    x = raw.astype(np.float32)

if int(fs) != int(audio_format["sample_rate_hz"]):
    raise ValueError("Sampling rate WAV tidak sama dengan audio_format_selected.json")

# 2. Parameter spektral baku proyek
frame_ms = 25.0
hop_ms = 10.0
n_frame = int(round(frame_ms * fs / 1000.0))
n_hop = int(round(hop_ms * fs / 1000.0))
nfft = 512
window_name = "hann"
delta_f = fs / nfft

spectral_config = {
    "sample_rate_hz": int(fs),
    "frame_ms": frame_ms,
    "hop_ms": hop_ms,
    "frame_samples": n_frame,
    "hop_samples": n_hop,
    "window": window_name,
    "nfft": nfft,
    "frequency_resolution_hz": float(delta_f)
}
(ROOT / "config").mkdir(exist_ok=True)
with open(ROOT / "config/spectral_config.json", "w", encoding="utf-8") as f:
    json.dump(spectral_config, f, indent=2)

# 3. FFT pada frame dengan energi RMS terbesar
starts = np.arange(0, max(1, len(x) - n_frame + 1), n_hop)
if len(starts) == 0:
    starts = np.array([0])
energies = []
for s in starts:
    frame = x[s:s+n_frame]
    if len(frame) < n_frame:
        frame = np.pad(frame, (0, n_frame-len(frame)))
    energies.append(np.sqrt(np.mean(frame**2) + 1e-12))
start = int(starts[int(np.argmax(energies))])
frame = x[start:start+n_frame]
if len(frame) < n_frame:
    frame = np.pad(frame, (0, n_frame-len(frame)))
win = np.hanning(n_frame)
X = np.fft.rfft(frame * win, n=nfft)
freq = np.fft.rfftfreq(nfft, d=1.0/fs)
mag = np.abs(X)

# Abaikan DC untuk mencari puncak dominan
valid = (freq >= 50) & (freq <= min(4000, fs/2))
idx_valid = np.where(valid)[0]
top = idx_valid[np.argsort(mag[idx_valid])[-5:]][::-1]
dominant_hz = [float(freq[i]) for i in top]

# 4. STFT seluruh rekaman
f_stft, t_stft, Z = stft(
    x, fs=fs, window="hann", nperseg=n_frame,
    noverlap=n_frame-n_hop, nfft=nfft,
    boundary=None, padded=False
)
mag_stft = np.abs(Z).astype(np.float32)

# 5. Simpan array untuk Mini Project berikutnya
feat_dir = ROOT / "features/intermediate"
fig_dir = ROOT / "results/figures"
met_dir = ROOT / "results/metrics"
for p in [feat_dir, fig_dir, met_dir]:
    p.mkdir(parents=True, exist_ok=True)

np.savez_compressed(
    feat_dir / "mp03_reference_stft.npz",
    frequencies_hz=f_stft.astype(np.float32),
    times_s=t_stft.astype(np.float32),
    magnitude=mag_stft
)

# 6. Gambar waveform + spectrum baseline
fig, ax = plt.subplots(2, 1, figsize=(9, 6))
time = np.arange(len(x)) / fs
ax[0].plot(time, x, linewidth=0.8)
ax[0].set(title="Waveform audio referensi", xlabel="Waktu (s)", ylabel="Amplitudo")
ax[0].grid(alpha=0.2)
ax[1].plot(freq, 20*np.log10(mag/(mag.max()+1e-12)+1e-12), linewidth=0.9)
ax[1].set_xlim(0, 4000)
ax[1].set_ylim(-100, 3)
ax[1].set(title="Spectrum frame energi terbesar", xlabel="Frekuensi (Hz)", ylabel="Magnitudo (dB)")
ax[1].grid(alpha=0.2)
fig.tight_layout()
fig.savefig(fig_dir / "mp03_time_frequency.png", dpi=180)
plt.close(fig)

# 7. Gambar spektrogram
S_db = 20*np.log10(mag_stft + 1e-6)
fig, ax = plt.subplots(figsize=(9, 4.8))
mesh = ax.pcolormesh(t_stft, f_stft, S_db, shading="auto")
ax.set_ylim(0, 4000)
ax.set(title="STFT audio referensi", xlabel="Waktu (s)", ylabel="Frekuensi (Hz)")
fig.colorbar(mesh, ax=ax, label="Magnitudo (dB)")
fig.tight_layout()
fig.savefig(fig_dir / "mp03_spectrogram.png", dpi=180)
plt.close(fig)

# 8. Simpan metrik
metrics = {
    "duration_s": float(len(x)/fs),
    "sample_rate_hz": int(fs),
    "frame_samples": n_frame,
    "hop_samples": n_hop,
    "nfft": nfft,
    "frequency_resolution_hz": float(delta_f),
    "number_of_stft_frames": int(len(t_stft)),
    "dominant_frequencies_hz": dominant_hz
}
with open(met_dir / "mp03_fft_stft.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

# 9. Handoff ke Mini Project 4
status.update({
    "status": "ready_for_mp04",
    "last_completed_mini_project": 3,
    "spectral_config": "config/spectral_config.json",
    "stft_reference": "features/intermediate/mp03_reference_stft.npz"
})
with open(STATUS_PATH, "w", encoding="utf-8") as f:
    json.dump(status, f, indent=2)

print("Mini Project 3 selesai")
print("Δf =", delta_f, "Hz")
print("Frekuensi dominan =", dominant_hz)
print("Status = ready_for_mp04")
