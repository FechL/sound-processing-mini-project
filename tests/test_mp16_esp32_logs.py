from pathlib import Path
import importlib.util
import contextlib
import io
import json
import sys
import tempfile
import threading
import queue
from unittest.mock import MagicMock, patch
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('esp32_collect', ROOT / 'tools/collect_mp16_esp32.py')
collector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collector)


class DeviceAcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.package = {'package_id': 'test', 'selected_model': 'classical_logreg_int8',
                        'sample_rate_hz': 16000, 'inference_stride_samples': 1600,
                        'classes': ['mati', 'naik', 'nyala', 'turun']}
        self.rows = [{'type': 'boot', 'package': 'test', 'model': 'classical_logreg_int8'}]
        for label in self.package['classes']:
            self.rows.append({'type': 'event', 'expected_label': label, 'predicted_class': label,
                              'pipeline_latency_ms': 20, 'adc_rate_hz': 32000,
                              'overflow_count': 0, 'adc_errors': 0, 'missed_strides': 0,
                              'minimum_free_heap_bytes': 40000})

    def test_complete_labelled_demo_passes(self):
        report = collector.summarize(self.rows, self.package, True)
        self.assertTrue(report['required_hardware_acceptance_pass'])
        self.assertEqual(report['event_accuracy'], 1)

    def test_empty_log_cannot_pass(self):
        report = collector.summarize([], self.package, True)
        self.assertFalse(report['required_hardware_acceptance_pass'])
        self.assertFalse(report['device_benchmark_performed'])

    def test_demo_requires_identity_and_audio_windows(self):
        self.assertTrue(collector.ready_for_demo(self.rows, self.package))
        self.assertFalse(collector.ready_for_demo(self.rows[:1], self.package))
        self.assertFalse(collector.ready_for_demo([], self.package))

    def test_fatal_prevents_demo_even_after_audio_started(self):
        self.rows.append({'type': 'fatal', 'error': 'frontend_or_adc_init_failed'})
        self.assertFalse(collector.ready_for_demo(self.rows, self.package))
        report = collector.summarize(self.rows, self.package, True)
        self.assertFalse(report['required_hardware_acceptance_pass'])

    def test_wrong_firmware_prevents_demo(self):
        self.rows[0]['package'] = 'old-firmware'
        self.assertFalse(collector.ready_for_demo(self.rows, self.package))

    def test_calibration_must_finish_and_gate_must_be_armed(self):
        self.rows[0]['requires_noise_calibration'] = True
        self.assertFalse(collector.ready_for_demo(self.rows, self.package))
        self.rows[-1].update(gate_ready=True, speech_active=False, event_armed=False)
        self.assertFalse(collector.ready_for_demo(self.rows, self.package))
        self.rows[-1]['event_armed'] = True
        self.assertTrue(collector.ready_for_demo(self.rows, self.package))
        self.rows[-1]['speech_active'] = True
        self.assertFalse(collector.ready_for_demo(self.rows, self.package))

    def test_missing_calibration_cannot_pass_acceptance(self):
        self.rows[0]['requires_noise_calibration'] = True
        report = collector.summarize(self.rows, self.package, True)
        self.assertFalse(report['acceptance_checks']['noise_calibration_complete'])
        self.assertFalse(report['required_hardware_acceptance_pass'])

    def test_demo_waits_for_firmware_readiness_indicator(self):
        self.rows[0]['requires_noise_calibration'] = True
        self.rows[-1].update(gate_ready=True, speech_active=False, event_armed=True,
                             ready_for_command=False)
        self.assertFalse(collector.ready_for_demo(self.rows, self.package))
        self.rows[-1]['ready_for_command'] = True
        self.assertTrue(collector.ready_for_demo(self.rows, self.package))

    def test_startup_fatal_saves_report_without_command_prompts(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest = root / 'firmware/esp32/include/package_manifest.json'
            manifest.parent.mkdir(parents=True)
            manifest.write_text(json.dumps({**self.package, 'input_sha256': {}}))
            serial_module = MagicMock()
            device = serial_module.Serial.return_value.__enter__.return_value
            device.readline.return_value = b'{"type":"fatal","error":"adc_init_failed"}\n'
            argv = ['collect', '--root', folder, '--port', 'fake', '--demo']
            with patch.dict(sys.modules, {'serial': serial_module}), \
                 patch.object(sys, 'argv', argv), \
                 patch('builtins.input') as prompt, \
                 contextlib.redirect_stdout(io.StringIO()), \
                 self.assertRaises(SystemExit):
                collector.main()
            prompt.assert_not_called()
            device.readline.assert_called_once()
            reports = list((root / 'results/device').glob('*/acceptance_metrics.json'))
            self.assertEqual(len(reports), 1)
            report = json.loads(reports[0].read_text())
            self.assertFalse(report['required_hardware_acceptance_pass'])
            self.assertFalse(report['acceptance_checks']['no_device_fatal'])

    def test_serial_is_drained_without_active_collection(self):
        incoming = queue.Queue()
        recorded = threading.Event()
        device, log = MagicMock(), MagicMock()
        def readline():
            try:
                return incoming.get(timeout=0.01)
            except queue.Empty:
                return b''
        device.readline.side_effect = readline
        log.flush.side_effect = recorded.set
        packets = []
        reader = collector.SerialReader(device, log, packets)
        reader.thread.start()
        try:
            incoming.put(b'{"type":"window","pipeline_latency_ms":20}\n')
            self.assertTrue(recorded.wait(2))
            self.assertEqual(len(packets), 1)
            self.assertIsNone(packets[0]['expected_label'])
            device.reset_input_buffer.assert_not_called()
        finally:
            reader.close()

    def test_continuous_mode_has_no_prompts_and_saves_on_ctrl_c(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest = root / 'firmware/esp32/include/package_manifest.json'
            manifest.parent.mkdir(parents=True)
            manifest.write_text(json.dumps({**self.package, 'input_sha256': {}}))
            serial_module = MagicMock()
            reader = MagicMock()
            reader.collect.side_effect = KeyboardInterrupt
            def create_reader(device, log, packets):
                packets.append({'type': 'boot', 'package': 'test', 'model': 'classical_logreg_int8'})
                return reader
            argv = ['collect', '--root', folder, '--port', 'fake', '--continuous']
            with patch.dict(sys.modules, {'serial': serial_module}), \
                 patch.object(sys, 'argv', argv), \
                 patch.object(collector, 'SerialReader', side_effect=create_reader), \
                 patch.object(collector, 'ready_for_demo', return_value=True), \
                 patch('builtins.input') as prompt, \
                 contextlib.redirect_stdout(io.StringIO()):
                collector.main()
            prompt.assert_not_called()
            reader.collect.assert_called_once_with(None)
            reader.close.assert_called_once()
            report_path = next((root / 'results/device').glob('*/acceptance_metrics.json'))
            report = json.loads(report_path.read_text())
            self.assertEqual(report['run_mode'], 'continuous_unlabelled')
            self.assertIsNone(report['event_accuracy'])
            self.assertFalse(report['demo_requested'])
            self.assertFalse(report['required_hardware_acceptance_pass'])

    def test_continuous_and_labelled_demo_cannot_be_combined(self):
        argv = ['collect', '--port', 'fake', '--continuous', '--demo']
        with patch.object(sys, 'argv', argv), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            collector.main()

    def test_unlabelled_monitoring_is_not_full_acceptance(self):
        report = collector.summarize(self.rows, self.package, False)
        self.assertFalse(report['required_hardware_acceptance_pass'])

    def test_wrong_package_cannot_pass(self):
        self.rows[0]['package'] = 'old-model'
        self.assertFalse(collector.summarize(self.rows, self.package, True)['required_hardware_acceptance_pass'])

    def test_overflow_cannot_pass(self):
        self.rows[-1]['overflow_count'] = 1
        self.assertFalse(collector.summarize(self.rows, self.package, True)['required_hardware_acceptance_pass'])

    def test_deadline_miss_cannot_pass(self):
        self.rows[-1]['pipeline_latency_ms'] = 200
        self.assertFalse(collector.summarize(self.rows, self.package, True)['required_hardware_acceptance_pass'])

    def test_wrong_command_does_not_count_as_detected(self):
        self.rows[-1]['predicted_class'] = 'nyala'
        report = collector.summarize(self.rows, self.package, True)
        self.assertFalse(report['required_hardware_acceptance_pass'])
        self.assertNotIn('turun', report['correctly_detected_commands'])


if __name__ == '__main__':
    unittest.main()
