"""Read-only frozen classical INT8 runtime; no training or acceptance writes."""
from pathlib import Path
import hashlib
import json
import numpy as np
from scipy.signal import lfilter
from scipy.fft import dct


class FrozenClassifier:
    def __init__(self, root):
        self.root = Path(root)
        read = lambda name: json.loads((self.root / name).read_text())
        manifest = read('deployment/mp15_manifest.json')
        self.model_name = manifest['selected_model']
        if self.model_name not in ('classical_svm_int8', 'classical_logreg_int8'):
            raise RuntimeError('Mode ini memerlukan model klasik INT8 dari MP15.')
        for name, expected in manifest['sha256'].items():
            if hashlib.sha256((self.root / name).read_bytes()).hexdigest() != expected:
                raise RuntimeError(f'Frozen artifact mismatch: {name}')
        self.fs = int(manifest['sample_rate_hz'])
        self.pre = read('config/preprocessing_config.json')
        self.feat = read('config/feature_config.json')
        self.window_samples = round(self.fs * float(read('config/dataset_config.json')['record_duration_s']))
        self.n = int(self.pre['frame_length_samples'])
        self.hop = int(self.pre['hop_length_samples'])
        with np.load(self.root / 'features/intermediate/mp06_filter_final.npz') as f:
            self.b, self.a = f['b'].astype(float), f['a'].astype(float)
        with np.load(self.root / 'models/mp15_classical_int8.npz', allow_pickle=True) as p:
            self.selected = p['selected_indices'].astype(int)
            self.mean, self.scale = p['selected_mean'].astype(float), p['selected_scale'].astype(float)
            self.xscale = float(p['input_quant_scale'][0])
            self.weight = p['weight_q'].astype(np.int32)
            self.wscale, self.bias = p['weight_scales'].astype(float), p['bias'].astype(float)
            self.classes = p['classes'].astype(str).tolist()
        if self.classes != manifest['classes'] or self.fs != 16000:
            raise RuntimeError('Class order atau sample rate paket tidak sesuai.')
        self.freq = np.fft.rfftfreq(self.n, 1 / self.fs)
        mel = self.feat['mel']
        lo = 2595 * np.log10(1 + mel['fmin_hz'] / 700)
        hi = 2595 * np.log10(1 + mel['fmax_hz'] / 700)
        hz = 700 * (10 ** (np.linspace(lo, hi, mel['n_mels'] + 2) / 2595) - 1)
        bins = np.floor((self.n + 1) * hz / self.fs).astype(int)
        self.fb = np.zeros((mel['n_mels'], self.n // 2 + 1))
        for i in range(1, mel['n_mels'] + 1):
            left, center, right = bins[i - 1:i + 2]
            center = max(center, left + 1); right = max(right, center + 1)
            for k in range(left, min(center, self.fb.shape[1])):
                self.fb[i - 1, k] = (k - left) / (center - left)
            for k in range(center, min(right, self.fb.shape[1])):
                self.fb[i - 1, k] = (right - k) / (right - center)

    def preprocess(self, signal):
        y = np.asarray(signal, dtype=float)
        if y.ndim != 1 or not y.size or not np.isfinite(y).all():
            raise ValueError('Audio harus mono, non-empty dan finite.')
        y = lfilter(self.b, self.a, y)
        y -= np.mean(y)
        y *= self.pre['normalization']['target_peak'] / (np.max(np.abs(y)) + 1e-12)
        if len(y) < self.n: y = np.pad(y, (0, self.n - len(y)))
        starts = np.arange(0, len(y) - self.n + 1, self.hop)
        frames = np.array([y[s:s + self.n] for s in starts])
        rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
        active = np.where(rms >= self.pre['vad']['relative_factor'] * np.max(rms))[0]
        if len(active):
            margin = int(self.pre['vad'].get('margin_frames', 2))
            first = max(0, int(active[0]) - margin)
            last = min(len(frames) - 1, int(active[-1]) + margin)
            y = y[starts[first]:min(len(y), starts[last] + self.n)]
        if len(y) < self.n: y = np.pad(y, (0, self.n - len(y)))
        frames = np.array([y[s:s + self.n] for s in range(0, len(y) - self.n + 1, self.hop)])
        frames = np.concatenate([frames[:, :1], frames[:, 1:] - self.pre['pre_emphasis_alpha'] * frames[:, :-1]], axis=1)
        return frames * np.hamming(self.n)[None, :]

    def pitch(self, frame):
        x = frame - np.mean(frame)
        ac = np.correlate(x, x, mode='full')[len(x) - 1:]
        if ac[0] <= 1e-12: return 0.0
        ac /= ac[0] + 1e-12
        lo = max(1, int(self.fs / self.feat['pitch']['max_hz']))
        hi = min(len(ac) - 1, int(self.fs / self.feat['pitch']['min_hz']))
        if hi <= lo: return 0.0
        lag = int(np.argmax(ac[lo:hi + 1]) + lo)
        return float(self.fs / lag) if ac[lag] >= self.feat['pitch']['confidence_threshold'] else 0.0

    def extract39(self, signal):
        frames = self.preprocess(signal)
        rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
        zcr = np.mean(np.abs(np.diff(np.signbit(frames), axis=1)), axis=1)
        power = (np.abs(np.fft.rfft(frames, axis=1)) + 1e-12) ** 2
        den = np.sum(power, axis=1) + 1e-12
        centroid = np.sum(power * self.freq[None, :], axis=1) / den
        bandwidth = np.sqrt(np.sum(power * (self.freq[None, :] - centroid[:, None]) ** 2, axis=1) / den)
        cum = np.cumsum(power, axis=1)
        roll = np.array([self.freq[min(np.searchsorted(cum[i], .85 * cum[i, -1]), len(self.freq) - 1)] for i in range(len(frames))])
        pitch = np.array([self.pitch(f) for f in frames]); voiced = pitch > 0
        mfcc = dct(np.log(np.maximum(power @ self.fb.T, 1e-12)), type=2, axis=1, norm='ortho')[:, :self.feat['mfcc']['n_mfcc']]
        values = []
        for j in range(mfcc.shape[1]): values.extend([float(mfcc[:, j].mean()), float(mfcc[:, j].std())])
        for a in [rms, zcr, centroid, bandwidth, roll]: values.extend([float(a.mean()), float(a.std())])
        pv = pitch[voiced]
        values.extend([float(pv.mean()) if len(pv) else 0, float(pv.std()) if len(pv) else 0, float(voiced.mean())])
        result = np.asarray(values, dtype=np.float32)
        if result.shape != (39,) or not np.isfinite(result).all():
            raise RuntimeError('Feature vector tidak valid.')
        return result

    def predict_features(self, features):
        z = (features[self.selected] - self.mean) / self.scale
        quant = np.clip(np.round(z / self.xscale), -127, 127).astype(np.int32)
        scores = (self.weight @ quant).astype(float) * (self.wscale * self.xscale) + self.bias
        return self.classes[int(np.argmax(scores))], scores

    def predict(self, signal):
        return self.predict_features(self.extract39(signal))
