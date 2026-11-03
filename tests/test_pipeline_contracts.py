"""Regression checks for frozen splits, artifact integrity and CNN input geometry."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import kws_common
from kws_models import TinyKWSNet, load_logmel


class SplitContractTests(unittest.TestCase):
    def setUp(self):
        self.data = {
            'sample_ids': np.array(['a', 'b', 'c']),
            'groups': np.array(['g1', 'g2', 'g3']),
            'train_mask': np.array([True, False, False]),
            'val_mask': np.array([False, True, False]),
            'test_mask': np.array([False, False, True]),
        }
        self.split = {'train_ids': ['a'], 'validation_ids': ['b'], 'test_ids': ['c']}

    def test_original_split_is_accepted(self):
        kws_common.validate_split(self.data, self.split)

    def test_mask_overlap_is_rejected(self):
        self.data['test_mask'][0] = True
        with self.assertRaises(RuntimeError):
            kws_common.validate_split(self.data, self.split)

    def test_group_leakage_is_rejected(self):
        self.data['groups'][1] = 'g1'
        with self.assertRaises(RuntimeError):
            kws_common.validate_split(self.data, self.split)

    def test_changed_test_identity_is_rejected(self):
        self.split['test_ids'] = ['a']
        with self.assertRaises(RuntimeError):
            kws_common.validate_split(self.data, self.split)


class FrozenArtifactTests(unittest.TestCase):
    def test_changed_model_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            model = root / 'model.bin'
            model.write_bytes(b'frozen weights')
            audit = root / 'results/metrics/mp13_pretest_artifact_hashes.json'
            audit.parent.mkdir(parents=True)
            audit.write_text(json.dumps({'model.bin': kws_common.sha256(model)}))
            with patch.object(kws_common, 'ROOT', root):
                kws_common.verify_frozen()
                model.write_bytes(b'changed weights')
                with self.assertRaises(RuntimeError):
                    kws_common.verify_frozen()


class LogMelContractTests(unittest.TestCase):
    def test_short_recording_uses_minimum_value_padding(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / 'features/logmel/mp09'
            path.mkdir(parents=True)
            x = np.arange(52, dtype=np.float32).reshape(2, 26) - 20
            np.savez(path / 'sample.npz', log_mel=x, label='nyala')
            padded, label = load_logmel(root, 'sample')
            self.assertEqual(label, 'nyala')
            self.assertEqual(padded.shape, (96, 26))
            np.testing.assert_array_equal(padded[:2], x)
            self.assertTrue(np.all(padded[2:] == x.min()))

    def test_cnn_matches_frozen_input_and_parameter_contract(self):
        torch.set_num_threads(1)
        net = TinyKWSNet(4).eval()
        with torch.inference_mode():
            result = net(torch.zeros(2, 1, 96, 26))
        self.assertEqual(tuple(result.shape), (2, 4))
        self.assertEqual(sum(p.numel() for p in net.parameters()), 1340)


if __name__ == '__main__':
    unittest.main()
