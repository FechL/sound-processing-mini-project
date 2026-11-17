from pathlib import Path
import importlib.util
import sys
import shutil
import struct
import subprocess
import tempfile
import unittest
import zlib
import contextlib
import io
import json
import queue
import types
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('laptop_bridge', ROOT / 'tools/stream_laptop_to_esp32.py')
bridge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bridge)


class LaptopESP32PacketTests(unittest.TestCase):
    def test_wire_format(self):
        p = bridge.packet(7, np.zeros(800))
        self.assertEqual(len(p), 1620)
        self.assertEqual(bridge.HEADER.unpack(p[:16]), (b'LPCM2!!!', 7, 800, 0))
        self.assertEqual(p[16:-4], bytes(1600))
        self.assertEqual(struct.unpack('<I', p[-4:])[0], zlib.crc32(p[:-4]))

    def test_signed_pcm_clips_and_preserves_order(self):
        samples = np.zeros(800); samples[:5] = [-2, -.5, 0, .5, 2]
        words = np.frombuffer(bridge.packet(0, samples)[16:-4], dtype='<i2')
        self.assertEqual(words[:5].tolist(), [-32767, -16384, 0, 16384, 32767])

    def test_sequence_wrap(self):
        self.assertEqual(bridge.HEADER.unpack(bridge.packet(2**32, np.zeros(800))[:16])[1], 0)

    def test_wrong_audio_shape_is_rejected(self):
        for x in [np.zeros(799), np.zeros((800, 1)), np.full(800, np.nan)]:
            with self.assertRaises(ValueError):
                bridge.packet(0, x)

    def test_boot_prefix_is_not_dumped_and_json_is_recovered(self):
        self.assertEqual(bridge.decode_log_line(b'\xffxxx{\"type\":\"boot\"}\n'), {'type': 'boot'})
        self.assertIsNone(bridge.decode_log_line(b'\xffxxxx\n'))
        self.assertIsNone(bridge.decode_log_line(b'{broken'))

    def test_host_identity_audio_ack_and_saved_summary(self):
        incoming = queue.Queue()
        writes = []

        class Device:
            def __init__(self, *args, **kwargs): pass
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read_until(self, *args):
                try: return incoming.get(timeout=.01)
                except queue.Empty: return b''
            def write(self, data):
                magic, sequence, count, kind = bridge.HEADER.unpack(data[:16])
                assert magic == bridge.MAGIC
                assert struct.unpack('<I', data[-4:])[0] == zlib.crc32(data[:-4])
                writes.append(kind)
                row = ({'type': 'identity', 'input': 'laptop_usb_pcm16',
                        'processing': 'esp32', 'protocol': 2} if kind == 1 else
                       {'type': 'ack', 'sequence': sequence})
                incoming.put((json.dumps(row) + '\n').encode())
                return len(data)

        class Input:
            def __init__(self, **kw): self.callback = kw['callback']
            def __enter__(self):
                self.callback(np.zeros((800, 1)), 800, None, types.SimpleNamespace(input_overflow=False))
                return self
            def __exit__(self, *args): pass

        sd = types.SimpleNamespace(InputStream=Input, check_input_settings=lambda **kw: None)
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(bridge, '__file__', str(Path(folder) / 'tools/bridge.py')), \
                 patch.object(bridge, 'load_sounddevice', return_value=sd), \
                 patch.dict(sys.modules, serial=types.SimpleNamespace(Serial=Device)), \
                 patch.object(sys, 'argv', ['test', '--seconds', '.02']), \
                 contextlib.redirect_stdout(io.StringIO()):
                bridge.main()
            summary = json.loads(next(Path(folder).glob('results/device/*/host_summary.json')).read_text())
            self.assertEqual(summary['protocol'], 2)
            self.assertEqual(summary['counters']['sent_blocks'], 1)
            self.assertEqual(summary['failures'], [])
            self.assertEqual(writes, [1, 0])


@unittest.skipUnless(shutil.which('g++'), 'native parser tests require g++')
class FirmwareParserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory()
        root = Path(cls.folder.name)
        source = root / 'parser.cpp'
        source.write_text('''#include <iostream>
#include "pcm_protocol.h"
int main() {
 PcmParser p; char c;
 while (std::cin.get(c)) if (p.push(static_cast<uint8_t>(c)))
   std::cout << p.kind() << " " << p.sequence() << " " << p.count() << " "
             << (p.count() ? p.sample(0) : 0) << "\\n";
 std::cout << "bad " << p.bad_frames << "\\n";
}
''')
        cls.binary = root / 'parser'
        subprocess.run(['g++', '-std=c++11', '-Wall', '-Wextra', '-Werror',
                        '-I', str(ROOT / 'firmware/esp32_laptop_mic/include'),
                        str(source), '-o', str(cls.binary)], check=True, capture_output=True)

    @classmethod
    def tearDownClass(cls): cls.folder.cleanup()

    def parse(self, data):
        return subprocess.run([str(self.binary)], input=data, capture_output=True, check=True).stdout.decode().splitlines()

    def test_cross_language_crc_and_identity(self):
        x = np.zeros(800); x[0] = -.5
        rows = self.parse(bridge.frame(0, 0, 1) + bridge.packet(9, x))
        self.assertEqual(rows, ['1 0 0 0', '0 9 800 -16384', 'bad 0'])

    def test_question_bytes_and_boot_noise_never_create_commands(self):
        rows = self.parse(b'???boot garbage???' + bridge.packet(1, np.zeros(800)))
        self.assertEqual(rows, ['0 1 800 0', 'bad 0'])

    def test_corruption_and_dropped_byte_resynchronize(self):
        p = bytearray(bridge.packet(3, np.zeros(800))); p[100] ^= 1
        valid = bridge.packet(4, np.zeros(800))
        for invalid in [bytes(p), bytes(p[:100] + p[101:])]:
            rows = self.parse(invalid + valid)
            self.assertEqual(rows[0], '0 4 800 0')
            self.assertGreaterEqual(int(rows[-1].split()[1]), 1)

    def test_invalid_length_and_long_noise_are_bounded(self):
        bad = bridge.HEADER.pack(bridge.MAGIC, 0, 65535, 0)
        rows = self.parse(b'x' * 100000 + bad + bridge.packet(1, np.zeros(800)))
        self.assertEqual(rows[0], '0 1 800 0')

    def test_pcm_can_contain_magic_and_question_without_resynchronization(self):
        data = b'?LPCM2!!!?' + bytes(1590)
        rows = self.parse(bridge.frame(5, 800, 0, data))
        self.assertEqual(len(rows), 2)
        self.assertTrue(rows[0].startswith('0 5 800 '))
        self.assertEqual(rows[-1], 'bad 0')
