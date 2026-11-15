from pathlib import Path
import ctypes
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which('c++'), 'Host C++ compiler required')
class NoiseGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory()
        tmp = Path(cls.folder.name)
        source = tmp / 'gate.cpp'
        source.write_text('''
#include "noise_gate.h"
static NoiseGate gate;
extern "C" void reset() { gate.reset(); }
extern "C" void update(float rms, int valid) { gate.update(rms, valid != 0); }
extern "C" int ready() { return gate.ready(); }
extern "C" int active() { return gate.active(); }
extern "C" int calibration_count() { return KWS_NOISE_CALIBRATION_WINDOWS; }
extern "C" float noise() { return gate.noise(); }
''')
        lib = tmp / 'gate.so'
        subprocess.run(['c++', '-std=c++11', '-shared', '-fPIC', '-O2',
                        '-I' + str(ROOT / 'firmware/esp32/include'), str(source), '-o', str(lib)], check=True)
        cls.lib = ctypes.CDLL(str(lib))
        cls.lib.update.argtypes = [ctypes.c_float, ctypes.c_int]
        cls.lib.noise.restype = ctypes.c_float

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def setUp(self):
        self.lib.reset()

    def calibrate(self):
        for _ in range(self.lib.calibration_count()):
            self.lib.update(0.03, 1)

    def test_no_event_activity_during_calibration(self):
        for _ in range(self.lib.calibration_count() - 1):
            self.lib.update(0.03, 1)
            self.assertFalse(self.lib.ready())
            self.assertFalse(self.lib.active())
        self.lib.update(0.03, 1)
        self.assertTrue(self.lib.ready())
        self.assertFalse(self.lib.active())

    def test_nonzero_ambient_can_release_speech_gate(self):
        self.calibrate()
        self.lib.update(0.04, 1)
        self.assertFalse(self.lib.active())
        self.lib.update(0.12, 1)
        self.assertTrue(self.lib.active())
        self.lib.update(0.06, 1)
        self.assertTrue(self.lib.active())
        self.lib.update(0.04, 1)
        self.assertFalse(self.lib.active())

    def test_invalid_audio_does_not_calibrate_and_reset_clears_state(self):
        for _ in range(self.lib.calibration_count() * 2):
            self.lib.update(float('nan'), 1)
            self.lib.update(1.0, 0)
        self.assertFalse(self.lib.ready())
        self.calibrate()
        self.lib.update(0.12, 1)
        self.assertTrue(self.lib.active())
        self.lib.reset()
        self.assertFalse(self.lib.ready())
        self.assertFalse(self.lib.active())

    def test_median_rejects_minority_transient_noise(self):
        count = self.lib.calibration_count()
        for index in range(count):
            self.lib.update(1.0 if index < count // 4 else 0.03, 1)
        self.assertAlmostEqual(self.lib.noise(), 0.03, places=6)

    def test_high_startup_floor_recovers_after_quieter_audio(self):
        for _ in range(self.lib.calibration_count()):
            self.lib.update(0.124, 1)
        self.lib.update(0.20, 1)
        self.assertFalse(self.lib.active())
        for _ in range(30):
            self.lib.update(0.03, 1)
        self.assertLess(self.lib.noise(), 0.04)
        self.lib.update(0.20, 1)
        self.assertTrue(self.lib.active())

    def test_speech_and_louder_input_do_not_raise_noise_floor(self):
        self.calibrate()
        baseline = self.lib.noise()
        for _ in range(100):
            self.lib.update(0.20, 1)
        self.assertTrue(self.lib.active())
        self.assertAlmostEqual(self.lib.noise(), baseline, places=6)

    def test_variable_quiet_noise_does_not_ratchet_floor_down(self):
        for _ in range(self.lib.calibration_count()):
            self.lib.update(0.04, 1)
        for _ in range(40):
            for rms in (0.01, 0.04, 0.05, 0.03):
                self.lib.update(rms, 1)
        self.assertGreater(self.lib.noise(), 0.025)
        self.assertLess(self.lib.noise(), 0.045)
        self.lib.update(0.15, 1)
        self.assertTrue(self.lib.active())
        self.lib.update(0.04, 1)
        self.assertFalse(self.lib.active())

    def test_quiet_noise_can_follow_a_modest_increase(self):
        self.calibrate()
        for _ in range(30):
            self.lib.update(0.04, 1)
        self.assertFalse(self.lib.active())
        self.assertGreater(self.lib.noise(), 0.038)


if __name__ == '__main__':
    unittest.main()
