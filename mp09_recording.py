"""Isolated MP9 experiments and continuous MAX9814 serial acquisition."""
from pathlib import Path
import json
import re
import shutil
import threading
import time

import numpy as np
from scipy.signal import firwin, lfilter
from tools.test_max9814_live import FrameParser, decode_adc


def experiment_path(root, name):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', name):
        raise ValueError('Nama dataset: 1-64 huruf/angka, underscore atau tanda minus; bukan path.')
    return Path(root) / 'experiments' / name


def prepare_experiment(root, name):
    root = Path(root)
    target = experiment_path(root, name)
    target.mkdir(parents=True, exist_ok=False)
    try:
        for source in root.glob('*.py'):
            shutil.copy2(source, target / source.name)
        for name in ('requirements.txt', 'README.md'):
            if (root / name).is_file():
                shutil.copy2(root / name, target / name)
        downstream = {'recording_session.json', 'dataset_config.json', 'classifier_config.json', 'deep_learning_config.json',
                      'deployment_config.json', 'evaluation_protocol.json', 'feature_reduction_config.json',
                      'realtime_architecture_config.json', 'system_integration_config.json'}
        for source in (root / 'config').glob('*.json'):
            if source.name not in downstream:
                dest = target / 'config' / source.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, dest)
        for folder in ('features/intermediate', 'results/metrics', 'results/figures', 'firmware'):
            for source in (root / folder).glob('mp0[1-8]_*'):
                if source.is_file():
                    dest = target / folder / source.name
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, dest)
        if (root / 'tools').is_dir():
            shutil.copytree(root / 'tools', target / 'tools', ignore=shutil.ignore_patterns('__pycache__'))
        for folder in ('firmware/esp32', 'firmware/mic_test'):
            if (root / folder).is_dir():
                shutil.copytree(root / folder, target / folder,
                    ignore=shutil.ignore_patterns('.pio', '__pycache__', 'generated_frontend.h',
                        'package_manifest.json', 'mp15_model.h', 'adc_decimator.h'))
        status = json.loads((root / 'project_status.json').read_text())
        for key in list(status):
            if re.match(r'mp(?:09|[1-9][0-9])(?:_|$)', key):
                status.pop(key)
        status.update(status='ready_for_mp09', last_completed_mini_project=8,
                      next_mini_project=9, next_input='data/raw/mp09_dataset',
                      completed_mini_projects=[n for n in status.get('completed_mini_projects', []) if n < 9],
                      experiment_source=str(root.resolve()))
        (target / 'project_status.json').write_text(json.dumps(status, indent=2))
    except Exception:
        # Only remove the new, exclusively-created experiment if setup failed.
        shutil.rmtree(target)
        raise
    return target


class ESP32Microphone:
    """Read mic_test continuously; condition audio like MP16 acquisition."""
    def __init__(self, port):
        import serial
        self.device = serial.Serial(port, 921600, timeout=0.05)
        self.device.dtr = False
        self.device.rts = False
        self.stop = threading.Event()
        self.ready = threading.Event()
        self.lock = threading.Lock()
        self.error = None
        self.chunks = None
        self.count = 0
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()
        if not self.ready.wait(8) or self.error:
            self.close()
            raise RuntimeError(self.error or 'Tidak ada audio ESP32. Upload firmware/mic_test dan tutup serial monitor lain.')

    def _read(self):
        parser = FrameParser()
        alpha = 0.0004
        bdc, adc = [1 - alpha, -(1 - alpha)], [1, -(1 - alpha)]
        fir = firwin(63, 7200, fs=32000, window='hamming')
        dc_state, fir_state = None, np.zeros(62)
        seen, phase = 0, 1
        previous = None
        try:
            while not self.stop.is_set():
                for sequence, flags, payload in parser.feed(self.device.read(2048)):
                    if flags or (previous is not None and sequence != (previous + 1) & 0xffffffff):
                        raise RuntimeError('Stream ADC putus/overflow/error; rekaman tidak disimpan. Periksa koneksi USB dan firmware mic_test.')
                    previous = sequence
                    raw, invalid = decode_adc(payload)
                    if invalid:
                        raise RuntimeError('Channel ADC tidak cocok; gunakan GPIO35 dan firmware mic_test.')
                    if not raw.size:
                        continue
                    x = raw / 2048.0
                    if dc_state is None:
                        dc_state = np.array([-(1 - alpha) * x[0]])
                    centered, dc_state = lfilter(bdc, adc, x, zi=dc_state)
                    filtered, fir_state = lfilter(fir, [1.0], centered, zi=fir_state)
                    pcm = filtered[phase::2].astype(np.float32)
                    phase = (phase - len(filtered)) % 2
                    seen += len(raw)
                    if seen <= 32000:
                        continue
                    self.ready.set()
                    with self.lock:
                        if self.chunks is not None:
                            self.chunks.append(pcm)
                            self.count += len(pcm)
        except Exception as error:
            self.error = str(error)
            self.ready.set()

    def record(self, count, sample_rate):
        if sample_rate != 16000:
            raise ValueError('Jalur MAX9814 memerlukan sample rate proyek 16000 Hz.')
        with self.lock:
            self.chunks, self.count = [], 0
        deadline = time.monotonic() + count / sample_rate + 5
        try:
            while time.monotonic() < deadline:
                if self.error:
                    raise RuntimeError(self.error)
                with self.lock:
                    if self.count >= count:
                        return np.concatenate(self.chunks)[:count]
                time.sleep(0.01)
            raise RuntimeError('Timeout rekaman ESP32; tidak ada WAV yang disimpan.')
        finally:
            with self.lock:
                self.chunks = None

    def close(self):
        self.stop.set()
        self.thread.join()
        self.device.close()
