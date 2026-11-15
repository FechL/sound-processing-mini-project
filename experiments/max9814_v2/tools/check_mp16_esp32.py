"""Compile the portable ESP32 frontend on the host and compare it with frozen MP9 data."""
from pathlib import Path
import argparse
import ctypes
import json
import shutil
import subprocess
import tempfile
import numpy as np
from scipy.io import wavfile


def check(root, firmware, output=None):
    if not shutil.which('cc') or not shutil.which('c++'):
        raise RuntimeError('Host check membutuhkan compiler C dan C++.')
    with tempfile.TemporaryDirectory(prefix='mp16_esp32_check_') as folder:
        tmp = Path(folder)
        include = firmware / 'include'
        kiss = firmware / 'lib/kissfft'
        wrapper = tmp / 'wrapper.cpp'
        wrapper.write_text('''
#include "kws_frontend.h"
#include "mp15_model.h"
extern "C" int extract_pcm(const int16_t *pcm, int count, float *features, float *scores, int full) {
    static KwsFrontend frontend;
    if (!frontend.begin()) return -2;
    KwsResult result;
    if (!frontend.extract(pcm, count, result, full != 0)) return -3;
    for (int j = 0; j < 39; ++j) features[j] = result.features[j];
    return MP15_predict39(features, scores);
}
''', encoding='utf-8')
        flags = ['-O2', '-fPIC', '-I'+str(include), '-I'+str(kiss)]
        objects = []
        for name in ['kiss_fft', 'kiss_fftr']:
            obj = tmp / f'{name}.o'
            subprocess.run(['cc', *flags, '-c', str(kiss / f'{name}.c'), '-o', str(obj)], check=True)
            objects.append(str(obj))
        library = tmp / 'frontend.so'
        subprocess.run(['c++', '-std=c++11', *flags, '-shared', str(wrapper),
                        str(firmware / 'src/kws_frontend.cpp'), *objects, '-o', str(library)], check=True)
        lib = ctypes.CDLL(str(library))
        lib.extract_pcm.argtypes = [ctypes.POINTER(ctypes.c_int16), ctypes.c_int,
                                   ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int]
        lib.extract_pcm.restype = ctypes.c_int
        with np.load(root / 'models/mp15_classical_int8.npz', allow_pickle=True) as data:
            selected = data['selected_indices'].astype(int)
            mean = data['selected_mean'].astype(float)
            scale = data['selected_scale'].astype(float)
            qscale = float(data['input_quant_scale'][0])
            weight = data['weight_q'].astype(np.int32)
            wscale = data['weight_scales'].astype(float)
            bias = data['bias'].astype(float)
            classes = data['classes'].astype(str)
        with np.load(root / 'features/dataset/mp09_features_39d.npz', allow_pickle=True) as data:
            ids = data['sample_ids'].astype(str)
            reference = data['X'].copy()
        rows = []
        for sid, features_py in zip(ids, reference):
            fs, pcm = wavfile.read(root / 'data/raw/mp09_dataset' / f'{sid}.wav')
            if fs != 16000 or pcm.dtype != np.int16 or pcm.ndim != 1:
                raise RuntimeError(f'WAV tidak cocok: {sid}')
            pcm = np.ascontiguousarray(pcm)
            f = np.zeros(39, dtype=np.float32)
            score = np.zeros(len(classes), dtype=np.float32)
            pred = lib.extract_pcm(pcm.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)), len(pcm),
                f.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), score.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), 1)
            quant = np.clip(np.round(((features_py[selected] - mean) / scale) / qscale), -127, 127).astype(np.int32)
            reference_scores = (weight @ quant) * (wscale * qscale) + bias
            predicted_py = int(np.argmax(reference_scores))
            selected_error = float(np.max(np.abs(f[selected] - features_py[selected])))
            # Also execute the optimized live path that omits unused pitch calculations.
            live = lib.extract_pcm(pcm.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)), len(pcm),
                f.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), score.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), 0)
            live_error = float(np.max(np.abs(f[selected] - features_py[selected])))
            rows.append({'sample_id': sid, 'max_selected_feature_abs_error': selected_error,
                         'max_live_selected_feature_abs_error': live_error,
                         'python_prediction': str(classes[predicted_py]),
                         'cpp_prediction': str(classes[pred]) if pred >= 0 else 'ERROR',
                         'prediction_matches': pred == predicted_py and live == predicted_py,
                         'max_score_abs_error': float(np.max(np.abs(score - reference_scores)))})
        passed = all(r['prediction_matches'] and r['max_selected_feature_abs_error'] < 0.01
                     and r['max_live_selected_feature_abs_error'] < 0.01 for r in rows)
        stress = []
        rng = np.random.default_rng(2026)
        t = np.arange(19200) / 16000
        signals = {'silence': np.zeros(19200),
                   'noise': rng.normal(0, 0.1, 19200),
                   'dc_noise': 0.4 + rng.normal(0, 0.03, 19200)}
        for hz in (50, 300, 1000, 6000):
            signals[f'tone_{hz}_noise'] = 0.3 * np.sin(2 * np.pi * hz * t) + rng.normal(0, 0.01, 19200)
        for name, signal in signals.items():
            pcm = np.ascontiguousarray(np.round(np.clip(signal, -1, 1) * 32767).astype(np.int16))
            reference_f, live_f = np.zeros(39, np.float32), np.zeros(39, np.float32)
            scores = np.zeros(len(classes), np.float32)
            predictions = []
            for full, features in ((1, reference_f), (0, live_f)):
                predictions.append(lib.extract_pcm(pcm.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)), len(pcm),
                    features.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), scores.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), full))
            error = float(np.max(np.abs(reference_f[selected] - live_f[selected])))
            stress.append({'signal': name, 'max_live_selected_feature_abs_error': error,
                           'pass': bool(np.isfinite(live_f).all() and error < 0.01
                                        and predictions[0] == predictions[1] and predictions[0] >= 0)})
        passed = passed and all(row['pass'] for row in stress)
        package = json.loads((include / 'package_manifest.json').read_text())
        report = {'scope': 'host_compiled_firmware_equivalence', 'n_samples': len(rows),
                  'package_id': package['package_id'], 'selected_model': package['selected_model'],
                  'selected_feature_atol': 0.01, 'all_pass': passed,
                  'device_benchmark_performed': False, 'rows': rows, 'numerical_stress_checks': stress}
        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(report, indent=2))
        if not passed:
            raise SystemExit('Firmware numerical equivalence failed; do not flash this package.')
        return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--firmware', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    check(args.root.resolve(), args.firmware or args.root / 'firmware/esp32', args.output)
