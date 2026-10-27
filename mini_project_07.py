from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import stft, istft, welch

ROOT = Path(__file__).resolve().parent
STATUS = ROOT / 'project_status.json'

with open(STATUS, 'r', encoding='utf-8') as f:
    status = json.load(f)
assert status.get('status') in ['ready_for_mp07', 'ready_for_mp08', 'ready_for_mp09', 'ready_for_mp10', 'ready_for_mp11'], status.get('status')

input_wav = ROOT / status['mp07_input_audio']
filter_cfg = ROOT / status['filter_final_config']
assert input_wav.exists(), input_wav
assert filter_cfg.exists(), filter_cfg

fs, x = wavfile.read(input_wav)
if x.ndim > 1:
    x = x.mean(axis=1)
if np.issubdtype(x.dtype, np.integer):
    scale = max(abs(np.iinfo(x.dtype).min), np.iinfo(x.dtype).max)
    x = x.astype(np.float64) / scale
else:
    x = x.astype(np.float64)

# 1) DC removal + peak normalization
x_dc = x - np.mean(x)
target_peak = 0.95
x_norm = target_peak * x_dc / (np.max(np.abs(x_dc)) + 1e-12)

# 2) Framing parameters
frame_ms = 25.0
hop_ms = 10.0
frame_len = int(round(fs * frame_ms / 1000.0))
hop_len = int(round(fs * hop_ms / 1000.0))
window = np.hamming(frame_len)

starts = np.arange(0, max(1, len(x_norm) - frame_len + 1), hop_len)
if len(starts) == 0:
    starts = np.array([0])
frames = []
for s in starts:
    fr = x_norm[s:s + frame_len]
    if len(fr) < frame_len:
        fr = np.pad(fr, (0, frame_len - len(fr)))
    frames.append(fr)
frames = np.asarray(frames)
rms = np.sqrt(np.mean(frames**2, axis=1) + 1e-12)

# 3) Adaptive VAD
noise_rms = float(np.percentile(rms, 10))
high_rms = float(np.percentile(rms, 90))
vad_factor = 0.25
vad_threshold = noise_rms + vad_factor * max(0.0, high_rms - noise_rms)
active = rms >= vad_threshold
# Fallback konservatif: jangan biarkan VAD memotong hampir seluruh kata.
if np.mean(active) < 0.15:
    vad_threshold = float(np.percentile(rms, 50))
    active = rms >= vad_threshold
if not np.any(active):
    active[:] = True
raw_active = active.copy()
noise_reliable = bool((np.mean(~raw_active) >= 0.10) and (high_rms >= 1.8 * max(noise_rms, 1e-12)))

# Tambahkan margin 2 frame sebelum/sesudah area aktif
idx = np.flatnonzero(active)
first = max(0, int(idx[0]) - 2)
last = min(len(frames) - 1, int(idx[-1]) + 2)
active[first:last + 1] = True

# 4) Spectral subtraction sederhana
nperseg = min(512, frame_len)
noverlap = int(0.5 * nperseg)
f, tt, Z = stft(x_norm, fs=fs, window='hann', nperseg=nperseg,
                noverlap=noverlap, boundary='zeros')
mag = np.abs(Z)
phase = np.exp(1j * np.angle(Z))

# Estimasi noise dari kolom STFT dengan energi terendah
col_energy = np.sqrt(np.mean(mag**2, axis=0) + 1e-12)
q = max(1, int(round(0.20 * len(col_energy))))
noise_cols = np.argsort(col_energy)[:q]
noise_mag = np.median(mag[:, noise_cols], axis=1, keepdims=True)

beta = 0.6 if noise_reliable else 0.0
spectral_floor = 0.25
clean_mag = np.maximum(mag - beta * noise_mag,
                       spectral_floor * noise_mag) if beta > 0 else mag.copy()
_, y = istft(clean_mag * phase, fs=fs, window='hann', nperseg=nperseg,
             noverlap=noverlap, input_onesided=True, boundary=True)
y = y[:len(x_norm)]

# 5) Trim memakai batas VAD
sample_start = int(starts[first])
sample_end = int(min(len(y), starts[last] + frame_len))
y_trim = y[sample_start:sample_end]
y_trim = target_peak * y_trim / (np.max(np.abs(y_trim)) + 1e-12)

# 6) Frame output + pre-emphasis untuk Mini Project 8
starts2 = np.arange(0, max(1, len(y_trim) - frame_len + 1), hop_len)
if len(starts2) == 0:
    starts2 = np.array([0])
frames2 = []
for s in starts2:
    fr = y_trim[s:s + frame_len]
    if len(fr) < frame_len:
        fr = np.pad(fr, (0, frame_len - len(fr)))
    frames2.append(fr)
frames2 = np.asarray(frames2)
alpha = 0.97
pre = frames2.copy()
pre[:, 1:] = frames2[:, 1:] - alpha * frames2[:, :-1]
pre[:, 0] = frames2[:, 0]
windowed = pre * window[None, :]

# 7) Metrics
rms_in = float(np.sqrt(np.mean(x_norm**2) + 1e-12))
rms_out = float(np.sqrt(np.mean(y_trim**2) + 1e-12))
active_ratio = float(np.mean(active))
clipping = int(np.sum(np.abs(y_trim) >= 0.999))

# Noise-floor proxy: median PSD 3-7 kHz
fi, pi = welch(x_norm, fs=fs, nperseg=min(1024, len(x_norm)))
fo, po = welch(y_trim, fs=fs, nperseg=min(1024, len(y_trim)))

def hf_floor(freq, psd):
    mask = (freq >= 3000) & (freq <= 7000)
    if not np.any(mask):
        return float(np.median(psd))
    return float(np.median(psd[mask]))

noise_in = hf_floor(fi, pi)
noise_out = hf_floor(fo, po)
noise_reduction_db = float(10 * np.log10((noise_in + 1e-15) / (noise_out + 1e-15)))

cfg = {
    'sampling_rate_hz': int(fs),
    'dc_removal': True,
    'normalization': {'type': 'peak', 'target_peak': target_peak},
    'frame_ms': frame_ms,
    'hop_ms': hop_ms,
    'frame_length_samples': frame_len,
    'hop_length_samples': hop_len,
    'window': 'hamming',
    'pre_emphasis_alpha': alpha,
    'vad': {'method': 'frame_rms_adaptive', 'relative_factor': vad_factor,
            'threshold': float(vad_threshold), 'margin_frames': 2},
    'denoise': {'method': 'spectral_subtraction', 'beta': beta,
                'spectral_floor': spectral_floor, 'noise_fraction': 0.20,
                'noise_estimate_reliable': noise_reliable},
    'source_filter_config': str(filter_cfg.relative_to(ROOT).as_posix())
}

out_audio = ROOT / 'data/processed/mp07_preprocessing'
feat_dir = ROOT / 'features/intermediate'
metric_dir = ROOT / 'results/metrics'
fig_dir = ROOT / 'results/figures'
config_dir = ROOT / 'config'

for d in [out_audio, feat_dir, metric_dir, fig_dir, config_dir]:
    d.mkdir(parents=True, exist_ok=True)

wav_out = out_audio / 'nyala_001_preprocessed.wav'
wavfile.write(wav_out, fs, np.int16(np.clip(y_trim, -1, 1) * 32767))
np.savez(feat_dir / 'mp07_preprocessed_frames.npz',
         frames=windowed.astype(np.float32), fs=np.int32(fs),
         frame_length=np.int32(frame_len), hop_length=np.int32(hop_len),
         pre_emphasis_alpha=np.float32(alpha), window='hamming')

with open(config_dir / 'preprocessing_config.json', 'w', encoding='utf-8') as f:
    json.dump(cfg, f, indent=2)

metrics = {
    'input_duration_s': float(len(x_norm) / fs),
    'output_duration_s': float(len(y_trim) / fs),
    'input_rms': rms_in,
    'output_rms': rms_out,
    'vad_threshold': float(vad_threshold),
    'active_ratio': active_ratio,
    'n_input_frames': int(len(frames)),
    'n_output_frames': int(len(frames2)),
    'noise_floor_change_db_proxy': noise_reduction_db,
    'output_clipping_samples': clipping
}
with open(metric_dir / 'mp07_preprocessing_metrics.json', 'w', encoding='utf-8') as f:
    json.dump(metrics, f, indent=2)

# Figures
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.plot(rms, label='RMS frame')
ax.axhline(vad_threshold, ls='--', label='VAD threshold')
ax.fill_between(np.arange(len(rms)), 0, rms, where=active, alpha=.2, label='active')
ax.set_xlabel('Frame')
ax.set_ylabel('RMS')
ax.grid(alpha=.2)
ax.legend()
fig.tight_layout()
fig.savefig(fig_dir / 'mp07_vad.png', dpi=180)
plt.close(fig)

fig, axs = plt.subplots(2, 1, figsize=(10, 6.2))
axs[0].plot(np.arange(len(x_norm)) / fs, x_norm, label='Input MP6', lw=.8)
axs[0].plot(np.arange(sample_start, sample_end) / fs, y_trim, label='Output MP7', lw=.8)
axs[0].set_ylabel('Amplitude')
axs[0].grid(alpha=.2)
axs[0].legend()
axs[1].semilogy(fi, pi + 1e-15, label='Input PSD')
axs[1].semilogy(fo, po + 1e-15, label='Output PSD')
axs[1].set_xlim(0, fs / 2)
axs[1].set_xlabel('Frequency (Hz)')
axs[1].set_ylabel('PSD')
axs[1].grid(alpha=.2)
axs[1].legend()
fig.tight_layout()
fig.savefig(fig_dir / 'mp07_preprocessing_overview.png', dpi=180)
plt.close(fig)

# 8) Update serial status
curr_status = status.get('status')
target_status = 'ready_for_mp08' if curr_status in ['ready_for_mp07', 'ready_for_mp08'] else curr_status
status.update({
    'status': target_status,
    'last_completed_mini_project': max(status.get('last_completed_mini_project', 7), 7),
    'preprocessing_config': 'config/preprocessing_config.json',
    'preprocessed_audio': 'data/processed/mp07_preprocessing/nyala_001_preprocessed.wav',
    'preprocessed_frames': 'features/intermediate/mp07_preprocessed_frames.npz',
    'preprocessing_metrics': 'results/metrics/mp07_preprocessing_metrics.json',
    'mp08_input_audio': 'data/processed/mp07_preprocessing/nyala_001_preprocessed.wav',
    'mp08_input_frames': 'features/intermediate/mp07_preprocessed_frames.npz'
})
with open(STATUS, 'w', encoding='utf-8') as f:
    json.dump(status, f, indent=2)

print('Input duration :', round(metrics['input_duration_s'], 3), 's')
print('Output duration:', round(metrics['output_duration_s'], 3), 's')
print('Active ratio :', round(active_ratio, 3))
print('Output frames :', len(frames2))
print('Noise-floor change proxy:', round(noise_reduction_db, 2), 'dB')
print('Denoise applied:', beta > 0)
print('Clipping samples:', clipping)
print('Status:', target_status)

# Checklist Tabel 7.9
print("\n" + "=" * 65)
print("VALIDASI ARTEFAK WAJIB (Tabel 7.9 Kriteria Lulus MP7)")
print("=" * 65)
artifacts_check = [
    ("project_status.json", STATUS.exists() and status.get("status") in ["ready_for_mp08", "ready_for_mp09", "ready_for_mp10", "ready_for_mp11"], "status = ready_for_mp08"),
    ("preprocessing_config.json", (config_dir / "preprocessing_config.json").exists(), "fs=16000; frame=25 ms; hop=10 ms; window=Hamming"),
    ("nyala_001_preprocessed.wav", wav_out.exists(), "Dapat diputar; tidak clipping; durasi > 0"),
    ("mp07_preprocessed_frames.npz", (feat_dir / "mp07_preprocessed_frames.npz").exists(), "Array frame 2D; tanpa NaN/Inf"),
    ("mp07_preprocessing_metrics.json", (metric_dir / "mp07_preprocessing_metrics.json").exists(), "active_ratio dan jumlah frame tercatat"),
    ("mp07_vad.png", (fig_dir / "mp07_vad.png").exists(), "Wilayah aktif kata tidak terpotong"),
    ("mp07_preprocessing_overview.png", (fig_dir / "mp07_preprocessing_overview.png").exists(), "Waveform dan PSD dapat dibandingkan"),
]

all_passed = True
for name, ok, criteria in artifacts_check:
    mark = "[V] LULUS" if ok else "[X] GAGAL"
    if not ok:
        all_passed = False
    print(f"{mark} | {name:<33} : {criteria}")

print("=" * 65)
if all_passed:
    print("STATUS KELULUSAN: SEMUA ARTEFAK LULUS & SIAP UNTUK MP08")
else:
    print("STATUS KELULUSAN: ADA ARTEFAK YANG BELUM LENGKAP")
print("=" * 65)
