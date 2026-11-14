"""Test MP9 retry against a temporary dataset and a mocked microphone."""
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
from pathlib import Path
import hashlib
import json
import runpy
import shutil
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np
from scipy.io import wavfile

ROOT = Path(__file__).resolve().parents[1]


class RetryRecordingTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.script = self.root / 'mini_project_09.py'
        shutil.copy2(ROOT / 'mini_project_09.py', self.script)
        for relative in ['config/project_config.json', 'config/preprocessing_config.json',
                         'config/feature_config.json', 'features/intermediate/mp06_filter_final.npz']:
            dest = self.root / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, dest)
        (self.root / 'project_status.json').write_text(json.dumps({'status': 'ready_for_mp09'}))
        self.raw = self.root / 'data/raw/mp09_dataset'
        self.raw.mkdir(parents=True)
        t = np.arange(19200) / 16000
        for block in range(1, 7):
            for index, label in enumerate(['nyala', 'mati', 'naik', 'turun']):
                y = 0.2 * np.sin(2 * np.pi * (140 + 20 * index + block) * t)
                wavfile.write(self.raw / f'{label}_b{block:02d}.wav', 16000,
                              np.round(y * 32767).astype(np.int16))

    def run_script(self, arguments):
        with patch.object(sys, 'argv', [str(self.script), *arguments]), redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            return runpy.run_path(str(self.script), run_name='__main__')

    def hashes(self):
        return {p.stem: hashlib.sha256(p.read_bytes()).hexdigest() for p in self.raw.glob('*.wav')}

    def test_retry_changes_only_requested_wavs_and_preserves_split(self):
        self.run_script([])
        before = self.hashes()
        split_path = self.root / 'splits/mp09_split.json'
        split_before = split_path.read_bytes()
        dataset_path = self.root / 'features/dataset/mp09_features_39d.npz'
        with np.load(dataset_path, allow_pickle=True) as data:
            ids = data['sample_ids'].astype(str)
            features_before = data['X'].copy()
        t = np.arange(19200) / 16000
        audio = (0.3 * np.sin(2 * np.pi * 700 * t)).astype(np.float32)[:, None]
        microphone = types.SimpleNamespace(rec=Mock(return_value=audio), wait=Mock(), play=Mock())
        with patch.dict(sys.modules, {'sounddevice': microphone}), patch('time.sleep'), patch('builtins.input', side_effect=['', 'd', 'u', '', 's', '', 's']):
            self.run_script(['--record', '--retry', 'naik_b04', 'naik_b05'])
        after = self.hashes()
        changed = {sid for sid in before if before[sid] != after[sid]}
        self.assertEqual(changed, {'naik_b04', 'naik_b05'})
        self.assertEqual(microphone.rec.call_count, 3)
        microphone.play.assert_called_once()
        self.assertEqual(split_before, split_path.read_bytes())
        with np.load(dataset_path, allow_pickle=True) as data:
            other = ~np.isin(ids, list(changed))
            np.testing.assert_array_equal(features_before[other], data['X'][other])
            self.assertTrue(np.any(features_before[~other] != data['X'][~other]))
        status = json.loads((self.root / 'project_status.json').read_text())
        self.assertEqual(status['status'], 'ready_for_mp10')
        self.assertEqual(status['last_completed_mini_project'], 9)
        self.assertEqual(status['next_mini_project'], 10)
        backups = list((self.root / 'data/raw/archives').glob('mp09_retry_*'))
        self.assertEqual(len(backups), 1)
        for sid in changed:
            archived = backups[0] / 'data/raw/mp09_dataset' / f'{sid}.wav'
            self.assertEqual(hashlib.sha256(archived.read_bytes()).hexdigest(), before[sid])

    def test_unknown_sample_is_rejected_without_changing_recordings(self):
        before = self.hashes()
        with self.assertRaises(SystemExit) as error:
            self.run_script(['--record', '--retry', 'naik_b99'])
        self.assertEqual(error.exception.code, 2)
        self.assertEqual(before, self.hashes())

    def test_missing_other_sample_does_not_generate_demo_audio(self):
        missing = self.raw / 'mati_b01.wav'
        missing.unlink()
        with self.assertRaises(SystemExit) as error:
            self.run_script(['--record', '--retry', 'naik_b04'])
        self.assertEqual(error.exception.code, 2)
        self.assertFalse(missing.exists())

    def test_retry_requires_record_flag(self):
        with self.assertRaises(SystemExit) as error:
            self.run_script(['--retry', 'naik_b04'])
        self.assertEqual(error.exception.code, 2)


if __name__ == '__main__':
    unittest.main()
