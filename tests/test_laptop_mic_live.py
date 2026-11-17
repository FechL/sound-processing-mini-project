from pathlib import Path
import contextlib
import importlib.util
import io
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import wave

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('laptop_live', ROOT / 'tools/test_laptop_mic_live.py')
mic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mic)


class LaptopMicTests(unittest.TestCase):
    def test_device_can_be_numeric_or_name(self):
        self.assertEqual(mic.device_selector('2'), 2)
        self.assertEqual(mic.device_selector('Built-in'), 'Built-in')

    def test_capture_copies_input_buffer(self):
        c = mic.AudioCapture(1)
        data = np.array([[.25], [-.25]], np.float32)
        c.capture(data, 2, None, False)
        data.fill(0)
        np.testing.assert_array_equal(c.blocks.get_nowait(), [.25, -.25])

    def test_full_queue_drops_oldest(self):
        c = mic.AudioCapture(1, max_blocks=1)
        c.capture(np.array([[.25]], np.float32), 1, None, False)
        c.capture(np.array([[.5]], np.float32), 1, None, False)
        self.assertEqual(c.blocks.get_nowait().tolist(), [.5])
        self.assertEqual(c.counters['host_dropped_blocks'], 1)

    def test_playback_gain_does_not_change_saved_input(self):
        c = mic.AudioCapture(4)
        x = np.array([[.5], [-.5]], np.float32)
        out = np.zeros((2, 2), np.float32)
        c.monitor(x, out, 2, None, False)
        np.testing.assert_array_equal(out, [[1, 1], [-1, -1]])
        np.testing.assert_array_equal(c.blocks.get_nowait(), [.5, -.5])

    def test_callback_status_is_counted(self):
        c = mic.AudioCapture(1)
        c.capture(np.zeros((2, 1)), 2, None, types.SimpleNamespace(input_overflow=True, output_underflow=False))
        self.assertEqual(c.counters['status_warnings'], 1)
        self.assertEqual(c.counters['input_overflows'], 1)

    def test_pcm16_clips_without_wrapping(self):
        result = np.frombuffer(mic.pcm16(np.array([-2, -1, 0, .5, 1, 2])), dtype='<i2')
        self.assertEqual(result.tolist(), [-32767, -32767, 0, 16384, 32767, 32767])

    def test_levels_and_silence_are_finite(self):
        self.assertEqual(mic.levels(np.zeros(10))['peak'], 0)
        self.assertTrue(np.isfinite(mic.levels(np.zeros(10))['rms_dbfs']))
        self.assertAlmostEqual(mic.levels(np.ones(10) * .5)['rms_dbfs'], -6.020599913)
        self.assertAlmostEqual(mic.levels(np.array([0, 1, -1]))['clipping_percent'], 200 / 3)

    def test_conda_dependency_error(self):
        with patch.object(mic.importlib, 'import_module', side_effect=OSError('GLIBCXX_3.4.32')), \
             patch.object(mic.Path, 'exists', return_value=True), self.assertRaises(SystemExit) as caught:
            mic.load_sounddevice()
        self.assertIn('LD_PRELOAD=', str(caught.exception))
        self.assertIn('termasuk tanpa --listen', str(caught.exception))

    def test_list_devices_does_not_record(self):
        sd = types.SimpleNamespace(query_devices=lambda: 'fake devices')
        with patch.object(mic, 'load_sounddevice', return_value=sd), \
             patch.object(sys, 'argv', ['test', '--list-devices']), \
             contextlib.redirect_stdout(io.StringIO()) as stdout:
            mic.main()
        self.assertIn('fake devices', stdout.getvalue())

    def fake_sounddevice(self):
        streams = []

        class FakeStream:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.active = True
                self.closed = False
                streams.append(self)

            def __enter__(self):
                block = np.array([[.25], [-.25]] * 80, np.float32)
                if isinstance(self.kwargs['channels'], tuple):
                    out = np.empty((160, self.kwargs['channels'][1]), np.float32)
                    self.kwargs['callback'](block, out, 160, None, False)
                else:
                    self.kwargs['callback'](block, 160, None, False)
                return self

            def __exit__(self, *args):
                self.active = False

            def close(self):
                self.closed = True

        def info(device=None, kind=None):
            return {'name': 'Fake laptop mic', 'default_samplerate': 48000,
                    'max_input_channels': 1, 'max_output_channels': 2}

        return types.SimpleNamespace(query_devices=info, check_input_settings=lambda **kw: None,
            check_output_settings=lambda **kw: None, InputStream=FakeStream, Stream=FakeStream), streams

    def test_headless_capture_saves_raw_wav_and_closes_stream(self):
        sd, streams = self.fake_sounddevice()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'mic.wav'
            argv = ['test', '--no-plot', '--seconds', '.01', '--save', str(path)]
            with patch.object(mic, 'load_sounddevice', return_value=sd), \
                 patch.object(sys, 'argv', argv), contextlib.redirect_stdout(io.StringIO()):
                mic.main()
            with wave.open(str(path)) as w:
                self.assertEqual((w.getnchannels(), w.getsampwidth(), w.getframerate()), (1, 2, 48000))
                values = np.frombuffer(w.readframes(w.getnframes()), dtype='<i2')
                self.assertEqual(values[:2].tolist(), [8192, -8192])
            self.assertTrue(streams[0].closed)

    def test_listen_supports_selected_stereo_output(self):
        sd, streams = self.fake_sounddevice()
        argv = ['test', '--no-plot', '--seconds', '.01', '--listen', '--input-device', '2', '--output-device', '3']
        with patch.object(mic, 'load_sounddevice', return_value=sd), \
             patch.object(sys, 'argv', argv), contextlib.redirect_stdout(io.StringIO()):
            mic.main()
        self.assertEqual(streams[0].kwargs['device'], (2, 3))
        self.assertEqual(streams[0].kwargs['channels'], (1, 2))

    def test_invalid_options_do_not_open_audio(self):
        for flags in [['--gain', 'nan'], ['--seconds', '-1'], ['--sample-rate', '0'], ['--output-device', '2']]:
            with self.subTest(flags=flags), patch.object(sys, 'argv', ['test', *flags]), \
                 patch.object(mic, 'load_sounddevice', side_effect=AssertionError('must not open')), \
                 contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                mic.main()

    def test_existing_wav_is_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'old.wav'
            path.write_bytes(b'preserve')
            with patch.object(sys, 'argv', ['test', '--save', str(path)]), \
                 contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                mic.main()
            self.assertEqual(path.read_bytes(), b'preserve')


if __name__ == '__main__':
    unittest.main()
