from pathlib import Path
import contextlib
import importlib.util
import io
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import patch
import wave

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('inmp_live', ROOT / 'tools/test_inmp441_live.py')
mic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mic)


def packet(sequence=0, payload=None, flags=0):
    if payload is None:
        payload = np.array([0x10000000, -0x10000000] * 80, dtype='<i4').tobytes()
    return mic.HEADER.pack(mic.MAGIC, sequence, len(payload), flags) + payload


class INMP441StreamTests(unittest.TestCase):
    def test_fragmented_packets_and_boot_noise(self):
        p = mic.FrameParser()
        data = b'ESP32 boot\n' + packet(8)
        frames = []
        for i in range(0, len(data), 7):
            frames.extend(p.feed(data[i:i + 7]))
        self.assertEqual(frames, [(8, 0, packet()[16:])])

    def test_multiple_frames_and_fatal(self):
        result = mic.FrameParser().feed(packet(2) + packet(3, b'', 4))
        self.assertEqual(result, [(2, 0, packet()[16:]), (3, 4, b'')])

    def test_invalid_headers_resynchronize(self):
        for size, flags in [(644, 0), (3, 0), (0, 0), (4, 8)]:
            invalid = mic.HEADER.pack(mic.MAGIC, 0, size, flags)
            self.assertEqual(mic.FrameParser().feed(invalid + packet(1)), [(1, 0, packet()[16:])])

    def test_wrong_firmware_and_long_noise_use_bounded_buffer(self):
        p = mic.FrameParser()
        self.assertEqual(p.feed(b'MIC9814!' + b'x' * 10000), [])
        self.assertLessEqual(len(p.buffer), 7)

    def test_signed_24bit_scaling_and_unused_low_bits(self):
        words = np.array([0, 0x7fffff00, -0x80000000, -256, 0x100000ff], dtype='<i4')
        actual = mic.decode_pcm(words.tobytes())
        np.testing.assert_allclose(actual, [0, 1 - 1 / 8388608, -1, -1 / 8388608, 0.125], rtol=0, atol=1e-8)
        self.assertEqual(actual.dtype, np.float32)
        self.assertEqual(mic.decode_pcm(b'').size, 0)
        with self.assertRaises(ValueError):
            mic.decode_pcm(b'x')

    def test_playback_partial_reads_and_silence_on_underflow(self):
        b = mic.PlaybackBuffer(8)
        self.assertEqual(b.push(np.array([1, 2, 3])), 0)
        out = np.zeros((2, 1), np.float32)
        self.assertFalse(b.read_into(out))
        self.assertEqual(out[:, 0].tolist(), [1, 2])
        self.assertTrue(b.read_into(out))
        self.assertEqual(out[:, 0].tolist(), [3, 0])
        self.assertEqual(b.available, 0)

    def test_playback_drops_old_audio_instead_of_accumulating_latency(self):
        b = mic.PlaybackBuffer(4)
        b.push(np.array([1, 2, 3]))
        self.assertEqual(b.push(np.array([4, 5, 6])), 3)
        self.assertEqual(b.push(np.arange(10)), 9)
        self.assertLessEqual(b.available, 4)
        out = np.zeros((4, 1), np.float32)
        self.assertFalse(b.read_into(out))
        self.assertEqual(out[:, 0].tolist(), [6, 7, 8, 9])

    def test_sounddevice_error_has_conda_workaround(self):
        with patch.object(mic.importlib, 'import_module', side_effect=OSError('GLIBCXX_3.4.32 not found')), \
             patch.object(mic.Path, 'exists', return_value=True), \
             self.assertRaises(SystemExit) as caught:
            mic.load_sounddevice()
        self.assertIn('LD_PRELOAD=', str(caught.exception))
        self.assertIn('tanpa --listen', str(caught.exception))

    def test_missing_playback_dependency_is_optional(self):
        with patch.object(mic.importlib, 'import_module', side_effect=ImportError('sounddevice')), \
             self.assertRaises(SystemExit) as caught:
            mic.load_sounddevice()
        self.assertIn('PortAudio', str(caught.exception))

    def test_headless_serial_capture_finalizes_wav_without_sounddevice(self):
        class FakeSerial:
            def __init__(self, *args, **kwargs):
                self.sequence = 0
                self.closed = False

            def __enter__(self):
                return self

            def __exit__(self, *args):
                self.closed = True

            def read(self, size):
                time.sleep(0.002)
                result = packet(self.sequence)
                self.sequence += 1
                return result

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'capture.wav'
            serial = types.SimpleNamespace(Serial=FakeSerial)
            argv = ['test_inmp441_live.py', '--port', 'FAKE', '--no-plot', '--seconds', '0.03', '--save', str(path)]
            with patch.dict(sys.modules, serial=serial), patch.object(sys, 'argv', argv), \
                 patch.object(mic, 'load_sounddevice', side_effect=AssertionError('must not load playback')), \
                 contextlib.redirect_stdout(io.StringIO()):
                mic.main()
            with wave.open(str(path)) as wav:
                self.assertEqual((wav.getnchannels(), wav.getsampwidth(), wav.getframerate()), (1, 2, 16000))
                self.assertGreater(wav.getnframes(), 0)
                self.assertEqual(len(wav.readframes(wav.getnframes())), wav.getnframes() * 2)

    def test_existing_wav_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'existing.wav'
            path.write_bytes(b'preserve')
            with patch.object(sys, 'argv', ['test', '--port', 'FAKE', '--save', str(path)]), \
                 contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                mic.main()
            self.assertEqual(path.read_bytes(), b'preserve')


if __name__ == '__main__':
    unittest.main()
