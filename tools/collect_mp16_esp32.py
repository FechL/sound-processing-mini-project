"""Collect real ESP32 JSON logs and optionally run the four-command live demo."""
from pathlib import Path
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import threading
import time
import numpy as np


def summarize(packets, package, demo_requested):
    boots = [p for p in packets if p.get('type') == 'boot']
    windows = [p for p in packets if p.get('type') in ('window', 'event')]
    events = [p for p in windows if p.get('type') == 'event']
    labelled = [p for p in events if p.get('expected_label')]
    latencies = [float(p['pipeline_latency_ms']) for p in windows if 'pipeline_latency_ms' in p]
    rates = [float(p['adc_rate_hz']) for p in windows if float(p.get('adc_rate_hz', 0)) > 0]
    observed = {p['expected_label'] for p in labelled if p.get('predicted_class') == p['expected_label']}
    overflow = max([int(p.get('overflow_count', 0)) for p in packets] or [0])
    errors = max([int(p.get('adc_errors', 0)) for p in packets] or [0])
    missed = max([int(p.get('missed_strides', 0)) for p in windows] or [0])
    p95 = float(np.percentile(latencies, 95)) if latencies else None
    stride_ms = package['inference_stride_samples'] / package['sample_rate_hz'] * 1000
    checks = {
        'matching_firmware_package': bool(boots) and all(p.get('package') == package['package_id'] and
            p.get('model') == package['selected_model'] for p in boots),
        'device_logs_present': bool(windows),
        'no_device_fatal': not any(p.get('type') == 'fatal' for p in packets),
        'no_adc_overflow': bool(windows) and overflow == 0,
        'no_adc_errors': bool(windows) and errors == 0,
        'no_missed_inference_strides': bool(windows) and missed == 0,
        'p95_faster_than_stride': p95 is not None and p95 < stride_ms,
        'adc_rate_within_2_percent': bool(rates) and all(abs(r - 32000) <= 640 for r in rates),
        'four_labelled_commands_detected': demo_requested and observed == set(package['classes']),
    }
    calibration_required = any(p.get('requires_noise_calibration') for p in boots)
    if calibration_required:
        checks['noise_calibration_complete'] = any(p.get('gate_ready') for p in windows)
    return {
        'scope': 'esp32_wroom_live_hardware', 'package_id': package['package_id'],
        'selected_model': package['selected_model'], 'device_benchmark_performed': bool(windows),
        'live_microphone_tested': bool(windows), 'demo_requested': demo_requested,
        'n_windows': len(windows), 'n_events': len(events),
        'event_accuracy': float(np.mean([p.get('predicted_class') == p['expected_label'] for p in labelled])) if labelled else None,
        'correctly_detected_commands': sorted(observed),
        'p95_pipeline_latency_ms': p95, 'inference_stride_ms': stride_ms,
        'max_overflow_count': overflow, 'max_adc_errors': errors, 'max_missed_strides': missed,
        'minimum_free_heap_bytes': min([int(p['minimum_free_heap_bytes']) for p in windows if 'minimum_free_heap_bytes' in p] or [0]),
        'firmware_flash_bytes': max([int(p.get('firmware_flash_bytes', 0)) for p in boots] or [0]),
        'power_mw': None,
        'detection_diagnostics': {
            'calibration_required': calibration_required,
            'last_noise_rms': windows[-1].get('noise_rms') if windows else None,
            'last_speech_on_rms': windows[-1].get('speech_on_rms') if windows else None,
            'last_speech_off_rms': windows[-1].get('speech_off_rms') if windows else None,
            'n_speech_active_windows': sum(bool(p.get('speech_active')) for p in windows),
        },
        'acceptance_checks': checks,
        'required_hardware_acceptance_pass': all(checks.values()),
        'note': 'Event accuracy is a live demonstration result, not a replacement for frozen MP13 test metrics. Measure power separately.',
    }


def ready_for_demo(packets, package):
    report = summarize(packets, package, False)
    checks = report['acceptance_checks']
    basic = all(checks[name] for name in
                ('matching_firmware_package', 'device_logs_present', 'no_device_fatal'))
    if not basic:
        return False
    if any(p.get('requires_noise_calibration') for p in packets if p.get('type') == 'boot'):
        windows = [p for p in packets if p.get('type') in ('window', 'event')]
        latest = windows[-1]
        return bool(latest.get('gate_ready') and latest.get('event_armed') and not latest.get('speech_active')
                    and latest.get('ready_for_command', True))
    return True


class SerialReader:
    """Drain serial even while input() waits, so device logging never blocks."""
    def __init__(self, device, log, packets):
        self.device, self.log, self.packets = device, log, packets
        self.expected = None
        self.stopped = threading.Event()
        self.failed = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.show_status = False
        self.last_gate_state = None
        self.last_gate_notice = 0

    def record(self, row):
        row['host_timestamp_utc'] = datetime.now(timezone.utc).isoformat()
        row['expected_label'] = self.expected
        self.packets.append(row)
        self.log.write(json.dumps(row) + '\n')
        self.log.flush()
        if row.get('type') in ('event', 'fatal', 'boot'):
            print(row)
        if self.show_status and row.get('gate_ready') and row.get('type') in ('window', 'event'):
            ready = row.get('ready_for_command', row.get('event_armed') and not row.get('speech_active'))
            state = ('SIAP' if ready else 'MENDETEKSI UCAPAN' if row.get('event_armed') and row.get('speech_active') else
                     'JEDA SINGKAT' if row.get('event_armed') and not row.get('speech_active') else 'TUNGGU HENING')
            now = time.monotonic()
            if state != self.last_gate_state or now - self.last_gate_notice >= 5:
                print(f'[{state}] RMS={row.get("audio_rms", 0):.3f} '
                      f'noise={row.get("noise_rms", 0):.3f} '
                      f'ambang masuk={row.get("speech_on_rms", 0):.3f}')
                self.last_gate_state, self.last_gate_notice = state, now
        if row.get('type') == 'fatal':
            self.failed.set()

    def run(self):
        try:
            while not self.stopped.is_set() and not self.failed.is_set():
                line = self.device.readline().decode('utf-8', errors='replace').strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    print(line)
                    continue
                if isinstance(row, dict):
                    self.record(row)
        except Exception as error:
            self.record({'type': 'fatal', 'error': 'serial_reader_failed', 'detail': str(error)})

    def collect(self, duration, expected=None):
        self.expected = expected
        try:
            return not self.failed.wait(duration)
        finally:
            self.expected = None

    def close(self):
        self.stopped.set()
        self.thread.join()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', required=True, help='Contoh: /dev/ttyUSB0')
    parser.add_argument('--seconds', type=float, default=30, help='Durasi logging atau durasi tiap kata pada --demo')
    parser.add_argument('--demo', action='store_true', help='Minta empat command dan beri label log langsung')
    parser.add_argument('--continuous', action='store_true', help='Monitor bebas tanpa prompt/batas waktu; Ctrl+C untuk berhenti')
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    if not np.isfinite(args.seconds) or args.seconds <= 0:
        parser.error('--seconds harus positif.')
    if args.demo and args.continuous:
        parser.error('--continuous tidak boleh digabung dengan --demo.')
    import serial
    root = args.root.resolve()
    manifest_path = root / 'firmware/esp32/include/package_manifest.json'
    package = json.loads(manifest_path.read_text())
    for name, expected in package['input_sha256'].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Ekspor firmware sudah usang: {name}')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output = root / 'results/device' / ('mp16_esp32_' + stamp)
    output.mkdir(parents=True)
    packets = []
    with serial.Serial(args.port, package.get('serial_baud', 115200), timeout=0.25) as device, (output / 'serial.jsonl').open('w', encoding='utf-8') as log:
        device.dtr = False
        device.rts = False
        reader = SerialReader(device, log, packets)
        reader.show_status = not args.demo
        device.write(b'?\n')
        reader.thread.start()
        try:
            print('Diam dahulu: menunggu board, pemanasan ADC dan kalibrasi noise (sekitar 7 detik)...')
            deadline = time.monotonic() + 12
            while time.monotonic() < deadline and not reader.failed.is_set():
                if ready_for_demo(list(packets), package):
                    break
                reader.failed.wait(0.1)
            if not ready_for_demo(list(packets), package):
                print('Board belum siap atau paket tidak cocok; demo dihentikan. '
                      'Periksa fatal log dan upload firmware terbaru.')
            elif args.demo:
                for label in ['nyala', 'naik', 'turun', 'mati']:
                    if reader.failed.is_set():
                        break
                    input(f'Tekan ENTER lalu ucapkan "{label}" sekali; beri jeda hening antar-kata. ')
                    if not reader.collect(args.seconds, label):
                        print('Board melaporkan fatal; sisa demo dibatalkan.')
                        break
            else:
                print('Mode bebas: ucapkan nyala/mati/naik/turun dalam urutan apa saja. '
                      'Beri jeda hening; Ctrl+C hanya menghentikan monitor, bukan firmware.')
                reader.collect(None if args.continuous else args.seconds)
        except KeyboardInterrupt:
            print('\nMonitor dihentikan; log disimpan. ESP32 tetap bekerja selama mendapat daya.')
        finally:
            reader.close()
    report = summarize(packets, package, args.demo)
    report['run_mode'] = 'labelled_demo' if args.demo else 'continuous_unlabelled' if args.continuous else 'timed_unlabelled'
    report['serial_log'] = (output / 'serial.jsonl').relative_to(root).as_posix()
    report['created_at_utc'] = datetime.now(timezone.utc).isoformat()
    (output / 'acceptance_metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    events = [p for p in packets if p.get('type') == 'event']
    columns = ['host_timestamp_utc', 'timestamp_ms', 'expected_label', 'predicted_class', 'score',
               'pipeline_latency_ms', 'overflow_count', 'led_on', 'led_level']
    with (output / 'events.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
        writer.writeheader(); writer.writerows(events)
    print(json.dumps(report, indent=2))
    print('Hasil:', output)
    if not args.demo:
        print('Monitor tanpa label: akurasi dan acceptance empat perintah tidak dinilai.')
    if args.demo and not report['required_hardware_acceptance_pass']:
        raise SystemExit('Acceptance perangkat belum lulus. Periksa log; jangan retune menggunakan frozen test.')


if __name__ == '__main__':
    main()
