"""Compile the actual frontend and exercise non-MFCC live statistics."""
from pathlib import Path
import ctypes
import shutil
import subprocess
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which('cc') and shutil.which('c++'), 'Host C/C++ compiler required')
class NonPitchFrontendTests(unittest.TestCase):
    def test_live_bandwidth_matches_reference(self):
        firmware = ROOT / 'firmware/esp32'
        kiss = firmware / 'lib/kissfft'
        with tempfile.TemporaryDirectory() as folder:
            tmp = Path(folder)
            shutil.copy2(firmware / 'include/kws_frontend.h', tmp / 'kws_frontend.h')
            header = (firmware / 'include/generated_frontend.h').read_text()
            # Force the mixed-feature path regardless of the current root model.
            header += '\n#undef KWS_MFCC_ONLY\n#define KWS_MFCC_ONLY 0\n#undef KWS_NEED_PITCH\n#define KWS_NEED_PITCH 0\n'
            (tmp / 'generated_frontend.h').write_text(header)
            (tmp / 'wrapper.cpp').write_text('''
#include "kws_frontend.h"
extern "C" int run(const int16_t *pcm, float *features, int full) {
    static KwsFrontend frontend;
    if (!frontend.begin()) return 0;
    KwsResult result;
    if (!frontend.extract(pcm, KWS_WINDOW_SAMPLES, result, full != 0)) return 0;
    for (int i = 0; i < 39; ++i) features[i] = result.features[i];
    return 1;
}
''')
            flags = ['-O2', '-fPIC', '-I' + str(tmp), '-I' + str(kiss)]
            objects = []
            for name in ('kiss_fft', 'kiss_fftr'):
                obj = tmp / (name + '.o')
                subprocess.run(['cc', *flags, '-c', str(kiss / (name + '.c')), '-o', str(obj)], check=True)
                objects.append(str(obj))
            library = tmp / 'frontend.so'
            subprocess.run(['c++', '-std=c++11', *flags, '-shared',
                str(tmp / 'wrapper.cpp'), str(firmware / 'src/kws_frontend.cpp'),
                *objects, '-o', str(library)], check=True)
            lib = ctypes.CDLL(str(library))
            lib.run.argtypes = [ctypes.POINTER(ctypes.c_int16), ctypes.POINTER(ctypes.c_float), ctypes.c_int]
            rng = np.random.default_rng(2026)
            t = np.arange(19200) / 16000
            signals = [np.zeros(19200), rng.normal(0, 0.1, 19200),
                       0.3 * np.sin(2 * np.pi * 300 * t) + rng.normal(0, 0.01, 19200)]
            for index, signal in enumerate(signals):
                pcm = np.round(np.clip(signal, -1, 1) * 32767).astype(np.int16)
                outputs = []
                for full in (1, 0):
                    features = np.zeros(39, np.float32)
                    self.assertEqual(lib.run(pcm.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
                        features.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), full), 1)
                    outputs.append(features)
                with self.subTest(signal=index):
                    self.assertTrue(np.isfinite(outputs[1]).all())
                    np.testing.assert_allclose(outputs[0][:36], outputs[1][:36], atol=0.01, rtol=0)
                    # Nonzero bandwidth is required; MFCC-only output cannot pass.
                    self.assertGreater(float(outputs[1][32]), 0)


if __name__ == '__main__':
    unittest.main()
