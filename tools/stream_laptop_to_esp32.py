"""Laptop captures PCM only; ESP32 runs preprocessing, classification and LEDs."""
from pathlib import Path
import argparse
from datetime import datetime, timezone
import json
import queue
import struct
import sys
import threading
import time
import zlib

import numpy as np

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.test_laptop_mic_live import load_sounddevice, device_selector

HEADER = struct.Struct('<8sIHH')
MAGIC = b'LPCM2!!!'
BAUD = 921600
PROTOCOL = 2
SAMPLE_RATE = 16000
BLOCK_SAMPLES = 800


def frame(sequence, count, kind, payload=b''):
    data = HEADER.pack(MAGIC, sequence & 0xffffffff, count, kind) + payload
    return data + struct.pack('<I', zlib.crc32(data))


def packet(sequence, samples):
    samples = np.asarray(samples)
    if samples.shape != (BLOCK_SAMPLES,):
        raise ValueError('USB PCM packet requires 800 mono samples (50 ms).')
    if not np.isfinite(samples).all():
        raise ValueError('Audio contains NaN/Inf.')
    pcm = np.round(np.clip(samples, -1, 1) * 32767).astype('<i2')
    return frame(sequence, BLOCK_SAMPLES, 0, pcm.tobytes())


def decode_log_line(data):
    """Boot ROM runs at another baud; recover JSON after a garbled prefix."""
    line = data.decode('utf-8', errors='replace').strip()
    at = line.find('{')
    if at < 0:
        return None
    try:
        row = json.loads(line[at:])
    except json.JSONDecodeError:
        return None
    return row if isinstance(row, dict) else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', default='/dev/ttyUSB0')
    parser.add_argument('--input-device', type=device_selector)
    parser.add_argument('--list-devices', action='store_true')
    parser.add_argument('--seconds', type=float, default=0, help='0 = sampai Ctrl+C')
    args = parser.parse_args()
    if not np.isfinite(args.seconds) or args.seconds < 0:
        parser.error('--seconds harus >= 0')
    sd = load_sounddevice()
    if args.list_devices:
        print(sd.query_devices()); return
    import serial
    sd.check_input_settings(device=args.input_device, channels=1, dtype='float32', samplerate=SAMPLE_RATE)
    root = Path(__file__).resolve().parents[1]
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output = root / 'results/device' / ('laptop_input_esp32_' + stamp)
    output.mkdir(parents=True)
    blocks = queue.Queue(maxsize=4)
    stopped = threading.Event()
    identified = threading.Event()
    acknowledged = threading.Event()
    expected_ack = [None]
    failure = []
    counters = {'captured_blocks': 0, 'host_drops': 0, 'input_overflows': 0, 'sent_blocks': 0}

    def capture(indata, frames, timing, status):
        if status.input_overflow:
            counters['input_overflows'] += 1
            # Sequence gap makes the board discard discontinuous context.
            counters['captured_blocks'] += 1
        sequence = counters['captured_blocks']
        counters['captured_blocks'] += 1
        try:
            blocks.put_nowait((sequence, indata[:, 0].copy()))
        except queue.Full:
            counters['host_drops'] += 1

    with serial.Serial(args.port, BAUD, timeout=.2, write_timeout=1) as device, \
         (output / 'serial.jsonl').open('w') as log:
        device.dtr = False; device.rts = False

        def read_logs():
            try:
                last_ready = None
                identity_shown = False
                last_diagnostic = 0
                while not stopped.is_set():
                    data = device.read_until(b'\n', 2048)
                    if not data:
                        continue
                    row = decode_log_line(data)
                    if row is None:
                        continue
                    if row.get('type') == 'ack':
                        if row.get('sequence') == expected_ack[0]:
                            acknowledged.set()
                        continue
                    row['host_timestamp_utc'] = datetime.now(timezone.utc).isoformat()
                    log.write(json.dumps(row) + '\n'); log.flush()
                    if row.get('type') in ('event', 'fatal'):
                        print(row)
                    if row.get('type') in ('boot', 'identity'):
                        if row.get('input') != 'laptop_usb_pcm16' or row.get('processing') != 'esp32' or row.get('protocol') != PROTOCOL:
                            failure.append('Firmware/protokol salah. Upload ulang firmware/esp32_laptop_mic versi 2.'); stopped.set()
                        else:
                            identified.set()
                            if not identity_shown:
                                print(row); identity_shown = True
                    ready = row.get('ready_for_command')
                    if ready is not None and ready != last_ready:
                        print('[SIAP]' if ready else '[MEMPROSES / TUNGGU HENING]')
                        last_ready = ready
                    if row.get('type') == 'fatal':
                        failure.append(str(row)); stopped.set()
                    if row.get('type') == 'health' and time.monotonic() - last_diagnostic >= 3:
                        print(f'[USB] paket={row.get("received_packets")} error={row.get("transport_errors")} '
                              f'crc={row.get("crc_errors")} gap={row.get("sequence_gaps")} '
                              f'input_overflow={counters["input_overflows"]} host_drop={counters["host_drops"]}')
                        last_diagnostic = time.monotonic()
            except Exception as error:
                failure.append(str(error)); stopped.set()

        reader = threading.Thread(target=read_logs, daemon=True)
        reader.start()
        try:
            deadline = time.monotonic() + 8
            while not identified.is_set() and not stopped.is_set() and time.monotonic() < deadline:
                device.write(frame(0, 0, 1))
                identified.wait(.5)
            if not identified.is_set():
                raise RuntimeError('Firmware laptop-PCM belum terdeteksi. Upload firmware/esp32_laptop_mic dahulu.')
            print('Input mikrofon laptop -> USB -> ESP32. Diam sekitar 7 detik untuk kalibrasi, lalu nyala/mati/naik/turun. Ctrl+C berhenti.')
            with sd.InputStream(device=args.input_device, samplerate=SAMPLE_RATE,
                                blocksize=BLOCK_SAMPLES, latency='high', channels=1, dtype='float32', callback=capture):
                started = last_audio = time.monotonic()
                while not stopped.is_set():
                    if args.seconds and time.monotonic() - started >= args.seconds:
                        break
                    try:
                        sequence, samples = blocks.get(timeout=.1)
                    except queue.Empty:
                        if time.monotonic() - last_audio > 5:
                            raise RuntimeError('Tidak ada audio laptop selama 5 detik.')
                        continue
                    last_audio = time.monotonic()
                    data = packet(sequence, samples)
                    expected_ack[0] = sequence & 0xffffffff
                    acknowledged.clear()
                    if device.write(data) != len(data):
                        raise RuntimeError('Serial write tidak lengkap; stream dihentikan.')
                    if not acknowledged.wait(1):
                        raise RuntimeError('ESP32 tidak mengakui paket audio selama 1 detik. Periksa kabel/port dan upload firmware protokol 2.')
                    counters['sent_blocks'] += 1
        except KeyboardInterrupt:
            print('\nDihentikan.')
        except Exception as error:
            failure.append(str(error))
            raise
        finally:
            stopped.set(); reader.join()
            (output / 'host_summary.json').write_text(json.dumps({
                'input': 'laptop_microphone', 'processing': 'esp32',
                'sample_rate_hz': SAMPLE_RATE, 'counters': counters, 'failures': failure,
                'protocol': PROTOCOL, 'serial_baud': BAUD,
                'acceptance_evaluated': False,
                'note': 'Different microphone from frozen MAX9814 dataset; accuracy not established.',
            }, indent=2))
            print('Log:', output)
        if failure:
            raise RuntimeError(failure[0])


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        raise SystemExit(f'Stream laptop -> ESP32 gagal: {error}') from None
