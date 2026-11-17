from pathlib import Path
import io
import json
import queue
import sys
import types
import unittest
import numpy as np
from scipy.io import wavfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.laptop_kws_runtime import FrozenClassifier
from tools.laptop_kws_led import LedClient, SpeechCapture


class FrozenLaptopRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.model = FrozenClassifier(ROOT)

    def test_pcm_extraction_matches_existing_frozen_features_and_int8_predictions(self):
        with np.load(ROOT / 'features/dataset/mp09_features_39d.npz', allow_pickle=True) as d:
            for sid, features in zip(d['sample_ids'].astype(str), d['X']):
                with self.subTest(sample=sid):
                    fs, pcm = wavfile.read(ROOT / 'data/raw/mp09_dataset' / f'{sid}.wav')
                    self.assertEqual(fs, self.model.fs)
                    x = pcm.astype(float) / 32767
                    actual = self.model.extract39(x)
                    np.testing.assert_allclose(actual[self.model.selected], features[self.model.selected], atol=.01, rtol=0)
                    self.assertEqual(self.model.predict(x)[0], self.model.predict_features(features)[0])

    def test_invalid_audio_rejected(self):
        for x in [np.array([]), np.zeros((10, 2)), np.array([np.nan])]:
            with self.assertRaises(ValueError): self.model.predict(x)

    def test_silence_features_are_finite(self):
        self.assertTrue(np.isfinite(self.model.extract39(np.zeros(self.model.window_samples))).all())


class SpeechCaptureTests(unittest.TestCase):
    def make_gate(self):
        gate = SpeechCapture(16000, 19200)
        for _ in range(40): self.assertIsNone(gate.feed(np.zeros(800)))
        return gate

    def test_calibration_and_quiet_do_not_generate_commands(self):
        gate = self.make_gate()
        self.assertTrue(gate.ready)
        for _ in range(100): self.assertIsNone(gate.feed(np.zeros(800)))

    def test_fixed_context_and_silence_rearm(self):
        gate = self.make_gate()
        for _ in range(4): gate.feed(np.zeros(800))
        audio = None
        for _ in range(30):
            result = gate.feed(np.ones(800) * .1)
            if result is not None: audio = result
        self.assertIsNotNone(audio)
        self.assertEqual(audio.shape, (19200,))
        self.assertFalse(gate.ready)
        for _ in range(3): gate.feed(np.zeros(800))
        self.assertTrue(gate.ready)


class LedClientTests(unittest.TestCase):
    def test_identity_ack_heartbeat_and_safe_close(self):
        incoming = queue.Queue(); commands = []

        class Device:
            def write(self, data):
                text = data.decode().strip()
                if text == 'HELLO': incoming.put(b'{"type":"identity","mode":"led_only","protocol":1}\n')
                elif text.startswith('CMD '):
                    _, sequence, command = text.split()
                    commands.append(command)
                    incoming.put((json.dumps({'type':'ack','sequence':int(sequence),
                        'command':command, 'led_on':command=='nyala','led_level':3})+'\n').encode())
                return len(data)
            def read_until(self, *args):
                try: return incoming.get(timeout=.01)
                except queue.Empty: return b''

        log = io.StringIO(); client = LedClient(Device(), log)
        try:
            client.connect()
            self.assertTrue(client.command('nyala')['led_on'])
            self.assertEqual(client.command('naik')['command'], 'naik')
            with self.assertRaises(ValueError): client.command('invalid')
        finally: client.close()
        self.assertEqual(commands, ['nyala', 'naik', 'mati'])
        self.assertFalse(client.thread.is_alive())
        self.assertIn('"type": "ack"', log.getvalue())
