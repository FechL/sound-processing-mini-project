from pathlib import Path
import importlib.util
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('mic_live', ROOT / 'tools/test_max9814_live.py')
mic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mic)


class MicrophoneStreamTests(unittest.TestCase):
    def test_audio_dependency_error_has_actionable_message(self):
        error = OSError("miniconda3/lib/libstdc++.so.6: GLIBCXX_3.4.32 not found")
        with patch.object(mic.importlib, 'import_module', side_effect=error), \
             patch.object(mic.Path, 'exists', return_value=True), \
             self.assertRaises(SystemExit) as caught:
            mic.load_sounddevice()
        self.assertIn('LD_PRELOAD=', str(caught.exception))
        self.assertIn('tanpa --listen', str(caught.exception))

    def test_missing_sounddevice_explains_optional_playback(self):
        with patch.object(mic.importlib, 'import_module', side_effect=ImportError('No sounddevice')), \
             self.assertRaises(SystemExit) as caught:
            mic.load_sounddevice()
        self.assertIn('PortAudio', str(caught.exception))
        self.assertNotIn('LD_PRELOAD=', str(caught.exception))

    def packet(self, sequence=0, payload=b'\x00\x78', flags=0):
        return mic.HEADER.pack(mic.MAGIC, sequence, len(payload), flags) + payload

    def test_fragmented_frames_and_boot_text(self):
        parser = mic.FrameParser()
        packet = self.packet(8)
        self.assertEqual(parser.feed(b'ESP32 boot\n' + packet[:5]), [])
        self.assertEqual(parser.feed(packet[5:17]), [])
        self.assertEqual(parser.feed(packet[17:]), [(8, 0, b'\x00\x78')])

    def test_multiple_packets_and_error_frame(self):
        frames = mic.FrameParser().feed(self.packet(2) + self.packet(3, b'', 4))
        self.assertEqual(frames, [(2, 0, b'\x00\x78'), (3, 4, b'')])

    def test_bad_header_resynchronizes(self):
        invalid = mic.HEADER.pack(mic.MAGIC, 0, 641, 0)
        self.assertEqual(mic.FrameParser().feed(invalid + self.packet()), [(0, 0, b'\x00\x78')])

    def test_channel_and_12bit_values(self):
        raw, invalid = mic.decode_adc(b'\x00\x70\xff\x7f\x00\x68')
        self.assertEqual(raw.tolist(), [0, 4095])
        self.assertEqual(invalid, 1)


if __name__ == '__main__':
    unittest.main()
