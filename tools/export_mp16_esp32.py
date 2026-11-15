"""Export the frozen MP15 linear classifier and MP9 frontend for ESP32-WROOM."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import numpy as np
from scipy.signal import firwin, tf2sos


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def c_array(name, values, ctype='double'):
    suffix = 'f' if ctype == 'float' else ''
    values = ', '.join(f'{float(v):.17e}{suffix}' for v in np.asarray(values).ravel())
    return f'static const {ctype} {name}[] = {{{values}}};\n'


def export(root, output):
    def read(name):
        return json.loads((root / name).read_text(encoding='utf-8'))
    deployment = read('config/deployment_config.json')
    manifest = read('deployment/mp15_manifest.json')
    selected = manifest['selected_model']
    if selected not in ('classical_logreg_int8', 'classical_svm_int8'):
        raise RuntimeError('Firmware ini membutuhkan paket linear INT8 dari MP15.')
    if selected != deployment['selected_deployment_model']:
        raise RuntimeError('Manifest dan deployment config tidak sama.')
    for name, expected in manifest['sha256'].items():
        if digest(root / name) != expected:
            raise RuntimeError(f'Paket MP15 tidak konsisten: {name}. Ekspor ulang melalui MP15.')
    project = read('config/project_config.json')
    pre = read('config/preprocessing_config.json')
    features = read('config/feature_config.json')
    dataset = read('config/dataset_config.json')
    rt = read('config/realtime_architecture_config.json')
    with np.load(root / deployment['classical_package'], allow_pickle=True) as model:
        classes = model['classes'].astype(str).tolist()
        selected_indices = model['selected_indices'].astype(int)
        if model['weight_q'].shape != (4, len(selected_indices)):
            raise RuntimeError('Dimensi model tidak cocok dengan empat kelas.')
    fs = int(project['sample_rate_hz'])
    frame = int(pre['frame_length_samples'])
    hop = int(pre['hop_length_samples'])
    count = int(round(float(dataset['record_duration_s']) * fs))
    if (fs, frame, hop, features['mfcc']['n_mfcc'], features['mel']['n_mels']) != (16000, 400, 160, 13, 26):
        raise RuntimeError('Kontrak firmware memerlukan 16 kHz, frame 400, hop 160, 13 MFCC, 26 Mel.')
    if count > 19200 or count < frame or rt['inference_stride_frames'] != 10:
        raise RuntimeError('Window/stride tidak didukung firmware ini.')
    with np.load(root / 'features/intermediate/mp06_filter_final.npz', allow_pickle=True) as filt:
        b = filt['b'].astype(float)
        a = filt['a'].astype(float)
    if len(a) != 5 or len(b) != 5:
        raise RuntimeError('Frontend saat ini membutuhkan filter MP6 orde empat.')
    b, a = b / a[0], a / a[0]
    input_names = list(manifest['sha256']) + ['config/dataset_config.json',
        'firmware/esp32/src/main.cpp', 'firmware/esp32/src/kws_frontend.cpp',
        'firmware/esp32/include/kws_frontend.h', 'firmware/esp32/include/board_config.h',
        'firmware/esp32/include/noise_gate.h',
        'firmware/esp32/platformio.ini', 'firmware/esp32/verify_package.py']
    hashes = {name: digest(root / name) for name in input_names}
    package_id = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()[:16]
    definitions = {
        'KWS_SAMPLE_RATE': fs, 'KWS_FRAME_SAMPLES': frame, 'KWS_HOP_SAMPLES': hop,
        'KWS_SERIAL_BAUD': 460800,
        'KWS_WINDOW_SAMPLES': count, 'KWS_MAX_FRAMES': 1 + (count - frame) // hop,
        'KWS_MEL_BANDS': 26, 'KWS_MFCC_COUNT': 13, 'KWS_CLASS_COUNT': len(classes),
        'KWS_FILTER_ORDER': 4, 'KWS_VAD_MARGIN': int(pre['vad'].get('margin_frames', 2)),
        'KWS_FILTER_SECTIONS': 2,
        'KWS_MFCC_ONLY': int(all(i < 26 for i in selected_indices)),
        'KWS_PITCH_LAG_MIN': max(1, int(fs / features['pitch']['max_hz'])),
        'KWS_PITCH_LAG_MAX': min(frame - 1, int(fs / features['pitch']['min_hz'])),
        'KWS_NEED_PITCH': int(any(i >= 36 for i in selected_indices)),
        'KWS_STRIDE_SAMPLES': hop * rt['inference_stride_frames'],
        'KWS_MEL_MIN_HZ': float(features['mel']['fmin_hz']),
        'KWS_MEL_MAX_HZ': float(features['mel']['fmax_hz']),
        'KWS_TARGET_PEAK': float(pre['normalization']['target_peak']),
        'KWS_PREEMPH': float(pre['pre_emphasis_alpha']),
        'KWS_VAD_FACTOR': float(pre['vad']['relative_factor']),
        'KWS_PITCH_CONFIDENCE': float(features['pitch']['confidence_threshold']),
        'KWS_ROLLOFF': float(features['spectral']['rolloff_fraction']),
    }
    output.mkdir(parents=True, exist_ok=True)
    header = '#pragma once\n// Generated from frozen MP9/MP15 artifacts.\n'
    header += ''.join(f'#define {name} {value}\n' for name, value in definitions.items())
    header += f'#define KWS_PACKAGE_ID "{package_id}"\n#define KWS_MODEL_NAME "{selected}"\n'
    header += 'static const char *const KWS_CLASSES[] = {' + ', '.join(json.dumps(c) for c in classes) + '};\n'
    header += c_array('KWS_FILTER_B', b) + c_array('KWS_FILTER_A', a)
    header += c_array('KWS_FILTER_SOS', tf2sos(b, a), 'float')
    (output / 'generated_frontend.h').write_text(header, encoding='utf-8')
    # The ESP32 ADC DMA minimum is 20 kHz; acquire at 32 kHz, low-pass, then decimate by two.
    decimator = firwin(63, 7200, fs=32000, window='hamming')
    (output / 'adc_decimator.h').write_text('#pragma once\n#define KWS_DECIMATOR_TAPS 63\n' +
                                          c_array('KWS_DECIMATOR', decimator, 'float'), encoding='utf-8')
    shutil.copy2(root / deployment['classical_c_header'], output / 'mp15_model.h')
    metadata = {
        'package_id': package_id, 'selected_model': selected, 'classes': classes,
        'serial_baud': definitions['KWS_SERIAL_BAUD'],
        'input_sha256': hashes, 'sample_rate_hz': fs, 'adc_sample_rate_hz': 32000,
        'utterance_samples': count, 'inference_stride_samples': definitions['KWS_STRIDE_SAMPLES'],
        'feature_scope': 'full_39_except_unused_pitch_in_live_mode',
        'physical_deployment_complete': False,
    }
    metadata['generated_sha256'] = {name: digest(output / name) for name in
        ['generated_frontend.h', 'adc_decimator.h', 'mp15_model.h']}
    (output / 'package_manifest.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(f'ESP32 package {package_id}: {selected}, window={count}, classes={classes}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    export(root, args.output_dir or root / 'firmware/esp32/include')
