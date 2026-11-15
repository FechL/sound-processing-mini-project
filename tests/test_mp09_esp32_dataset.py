"""Exercise MP9's new experiment mode without using a physical microphone."""
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
import json
import runpy
import shutil
import sys
import tempfile
import threading
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np
from scipy.io import wavfile
import mp09_recording

ROOT = Path(__file__).resolve().parents[1]


class NewDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for relative in ['mini_project_09.py', 'mp09_recording.py',
                         'config/project_config.json', 'config/preprocessing_config.json',
                         'config/feature_config.json', 'features/intermediate/mp06_filter_final.npz']:
            target = self.root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, target)
        self.status = self.root / 'project_status.json'
        self.status.write_text(json.dumps({'status': 'project_complete', 'mp16': {'old': True},
            'completed_mini_projects': [9, 10, 11, 12, 13, 14, 15, 16]}))
        self.old = self.root / 'data/raw/mp09_dataset/nyala_b01.wav'
        self.old.parent.mkdir(parents=True)
        self.old.write_bytes(b'original dataset must remain untouched')
        self.script = self.root / 'mini_project_09.py'

    def run_script(self, args):
        with patch.object(sys, 'argv', [str(self.script), *args]), \
             redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            return runpy.run_path(str(self.script), run_name='__main__')

    def test_snapshot_does_not_copy_frozen_downstream(self):
        frozen = self.root / 'results/metrics/mp13_test_open_lock.json'
        frozen.parent.mkdir(parents=True)
        frozen.write_text('{}')
        before = self.status.read_bytes()
        target = mp09_recording.prepare_experiment(self.root, 'max9814_v2')
        state = json.loads((target / 'project_status.json').read_text())
        self.assertEqual(state['status'], 'ready_for_mp09')
        self.assertNotIn('mp16', state)
        self.assertFalse((target / 'data/raw/mp09_dataset').exists())
        self.assertFalse((target / 'results/metrics/mp13_test_open_lock.json').exists())
        self.assertTrue((target / 'features/intermediate/mp06_filter_final.npz').exists())
        self.assertEqual(self.status.read_bytes(), before)

    def test_name_cannot_escape_or_overwrite_existing_experiment(self):
        with self.assertRaises(ValueError):
            mp09_recording.prepare_experiment(self.root, '../old')
        mp09_recording.prepare_experiment(self.root, 'test')
        with self.assertRaises(FileExistsError):
            mp09_recording.prepare_experiment(self.root, 'test')

    def test_full_serial_dataset_preserves_original_project(self):
        before = self.status.read_bytes()
        t = np.arange(19200) / 16000
        source = Mock()
        source.record.return_value = (0.2 * np.sin(2 * np.pi * 400 * t)).astype(np.float32)
        with patch.object(mp09_recording, 'ESP32Microphone', return_value=source), \
             patch('time.sleep'), patch('builtins.input', side_effect=['', 's'] * 24):
            self.run_script(['--record', '--new-dataset', 'serial_v2', '--port', 'mock'])
        target = self.root / 'experiments/serial_v2'
        self.assertEqual(source.record.call_count, 24)
        self.assertEqual(len(list((target / 'data/raw/mp09_dataset').glob('*.wav'))), 24)
        fs, audio = wavfile.read(target / 'data/raw/mp09_dataset/nyala_b01.wav')
        self.assertEqual((fs, audio.shape, audio.dtype), (16000, (19200,), np.dtype('int16')))
        state = json.loads((target / 'project_status.json').read_text())
        self.assertEqual(state['status'], 'ready_for_mp10')
        config = json.loads((target / 'config/dataset_config.json').read_text())
        self.assertEqual(config['recording_source'], 'esp32_max9814_gpio35')
        self.assertEqual(self.status.read_bytes(), before)
        self.assertEqual(self.old.read_bytes(), b'original dataset must remain untouched')
        with self.assertRaises(SystemExit):
            self.run_script(['--record', '--dataset', 'serial_v2', '--resume', '--port', 'mock'])

    def test_resume_skips_valid_existing_wav(self):
        target = mp09_recording.prepare_experiment(self.root, 'partial')
        path = target / 'data/raw/mp09_dataset/nyala_b01.wav'
        path.parent.mkdir(parents=True)
        wavfile.write(path, 16000, np.zeros(19200, np.int16))
        original = path.read_bytes()
        source = Mock()
        source.record.return_value = np.full(19200, 0.03, np.float32)
        with patch.object(mp09_recording, 'ESP32Microphone', return_value=source), \
             patch('time.sleep'), patch('builtins.input', side_effect=['', 's'] * 23):
            self.run_script(['--record', '--dataset', 'partial', '--resume', '--port', 'mock'])
        self.assertEqual(source.record.call_count, 23)
        self.assertEqual(path.read_bytes(), original)

    def test_original_completed_dataset_is_still_protected(self):
        with self.assertRaises(SystemExit):
            self.run_script(['--record', '--port', 'mock'])
        self.assertEqual(json.loads(self.status.read_text())['status'], 'project_complete')


class SerialAcquisitionTests(unittest.TestCase):
    def test_raw_serial_is_conditioned_to_16khz(self):
        from tools.test_max9814_live import HEADER, MAGIC
        stopped = threading.Event()
        device = Mock()
        sequence = 0
        def read(size):
            nonlocal sequence
            stopped.wait(0.001)
            index = np.arange(320) + sequence * 320
            raw = np.round(2048 + 200 * np.sin(2 * np.pi * 400 * index / 32000)).astype('<u2')
            payload = (raw | 0x7000).astype('<u2').tobytes()
            packet = HEADER.pack(MAGIC, sequence, len(payload), 0) + payload
            sequence += 1
            return packet
        device.read.side_effect = read
        module = types.SimpleNamespace(Serial=Mock(return_value=device))
        with patch.dict(sys.modules, {'serial': module}):
            source = mp09_recording.ESP32Microphone('fake')
            try:
                audio = source.record(1600, 16000)
                self.assertEqual(audio.shape, (1600,))
                self.assertTrue(np.isfinite(audio).all())
                self.assertGreater(float(np.std(audio)), 0.01)
            finally:
                stopped.set()
                source.close()
        device.close.assert_called_once()

    def test_adc_failure_aborts_acquisition(self):
        from tools.test_max9814_live import HEADER, MAGIC
        device = Mock()
        device.read.return_value = HEADER.pack(MAGIC, 0, 0, 4)
        module = types.SimpleNamespace(Serial=Mock(return_value=device))
        with patch.dict(sys.modules, {'serial': module}), self.assertRaises(RuntimeError):
            mp09_recording.ESP32Microphone('fake')
        device.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
