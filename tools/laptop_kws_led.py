"""Microphone, frozen preprocessing and inference on laptop; ESP32 controls LEDs only."""
from pathlib import Path
import argparse
from collections import deque
from datetime import datetime, timezone
import json
import queue
import sys
import threading
import time
import numpy as np

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.test_laptop_mic_live import load_sounddevice, device_selector
from tools.laptop_kws_runtime import FrozenClassifier


class LedClient:
    def __init__(self, device, log):
        self.device, self.log = device, log
        self.stop = threading.Event(); self.identified = threading.Event()
        self.ack = threading.Event(); self.pending = None; self.response = None
        self.sequence = 0; self.error = None
        self.write_lock = threading.Lock()
        self.thread = threading.Thread(target=self.read, daemon=True)

    def write(self, line):
        data = (line + '\n').encode('ascii')
        with self.write_lock:
            if self.device.write(data) != len(data): raise RuntimeError('Serial write tidak lengkap.')

    def read(self):
        ping = 0
        try:
            while not self.stop.is_set():
                if time.monotonic() - ping > 1:
                    self.write('PING'); ping = time.monotonic()
                line = self.device.read_until(b'\n', 2048).decode(errors='replace')
                at = line.find('{')
                if at < 0: continue
                try: row = json.loads(line[at:])
                except json.JSONDecodeError: continue
                if not isinstance(row, dict): continue
                row['host_timestamp_utc'] = datetime.now(timezone.utc).isoformat()
                self.log.write(json.dumps(row) + '\n'); self.log.flush()
                if row.get('type') == 'identity':
                    if row.get('mode') != 'led_only' or row.get('protocol') != 1:
                        raise RuntimeError('Firmware salah; upload firmware/esp32_led_only.')
                    self.identified.set()
                if row.get('type') == 'ack' and row.get('sequence') == self.pending:
                    self.response = row; self.ack.set()
        except Exception as error:
            self.error = str(error); self.stop.set()

    def connect(self):
        self.thread.start()
        end = time.monotonic() + 8
        while time.monotonic() < end and not self.stop.is_set():
            self.write('HELLO')
            if self.identified.wait(.5): return
        raise RuntimeError(self.error or 'Firmware LED-only tidak terdeteksi. Upload dahulu.')

    def command(self, label):
        if label not in ('nyala', 'mati', 'naik', 'turun'): raise ValueError('Unknown command')
        if self.error: raise RuntimeError(self.error)
        self.sequence = (self.sequence + 1) & 0xffffffff
        self.pending = self.sequence; self.response = None; self.ack.clear()
        self.write(f'CMD {self.sequence} {label}')
        if not self.ack.wait(1.5): raise RuntimeError(self.error or 'ESP32 tidak mengakui perintah LED.')
        return self.response

    def close(self):
        try:
            if self.identified.is_set() and not self.error:
                self.command('mati'); self.write('READY 0')
        finally:
            self.stop.set()
            if self.thread.is_alive(): self.thread.join()


class SpeechCapture:
    """Operational energy gate only; fixed 1.2 s context, no model retuning."""
    def __init__(self, fs, count):
        self.fs, self.count = fs, count
        self.calibration = []; self.noise = 0.0
        self.pre = deque(maxlen=4); self.parts = None
        self.quiet = 0; self.armed = True

    def feed(self, block):
        rms = float(np.sqrt(np.mean(np.asarray(block, dtype=float) ** 2)))
        if len(self.calibration) < 40:
            self.calibration.append(rms)
            if len(self.calibration) == 40: self.noise = float(np.median(self.calibration))
            return None
        onset, release = max(.008, self.noise * 3), max(.008, self.noise * 1.5)
        if self.parts is not None:
            self.parts.append(block.copy())
            if sum(len(x) for x in self.parts) >= self.count:
                result = np.concatenate(self.parts)[:self.count]
                self.parts = None; self.armed = False; self.quiet = 0; self.pre.clear()
                return result
        elif self.armed and rms >= onset:
            self.parts = [*self.pre, block.copy()]; self.pre.clear()
        else:
            self.pre.append(block.copy())
            if rms < release:
                self.quiet += 1
                if self.quiet >= 3: self.armed = True
            else: self.quiet = 0
            if rms < onset: self.noise += .1 * (rms - self.noise)
        return None

    @property
    def ready(self):
        return len(self.calibration) == 40 and self.armed and self.parts is None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', default='/dev/ttyUSB0')
    parser.add_argument('--input-device', type=device_selector)
    parser.add_argument('--list-devices', action='store_true')
    parser.add_argument('--continuous', action='store_true', help='Tanpa ENTER; kalibrasi noise dahulu')
    parser.add_argument('--seconds', type=float, default=0, help='Batas durasi mode continuous; 0 = sampai Ctrl+C')
    args = parser.parse_args()
    if not np.isfinite(args.seconds) or args.seconds < 0: parser.error('--seconds harus >= 0')
    if args.seconds and not args.continuous: parser.error('--seconds membutuhkan --continuous')
    sd = load_sounddevice()
    if args.list_devices: print(sd.query_devices()); return
    import serial
    root = Path(__file__).resolve().parents[1]
    model = FrozenClassifier(root)
    sd.check_input_settings(device=args.input_device, channels=1, dtype='float32', samplerate=model.fs)
    output = root / 'results/device' / ('laptop_model_led_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    output.mkdir(parents=True)
    with serial.Serial(args.port, 115200, timeout=.1, write_timeout=1) as device, \
         (output / 'serial.jsonl').open('w') as serial_log, (output / 'predictions.jsonl').open('w') as predictions:
        device.dtr = False; device.rts = False
        client = LedClient(device, serial_log)

        def classify(audio):
            client.write('READY 0')
            clip = float(np.mean(np.abs(audio) >= .99))
            rms = float(np.sqrt(np.mean(np.asarray(audio, dtype=float) ** 2)))
            row = {'processing': 'laptop', 'model': model.model_name, 'audio_rms': rms,
                   'clipping_ratio': clip, 'host_timestamp_utc': datetime.now(timezone.utc).isoformat()}
            if clip > .01 or rms < .001:
                row.update(type='rejected', reason='clipping' if clip > .01 else 'too_quiet')
                print('Audio ditolak:', row['reason'])
            else:
                start = time.perf_counter()
                label, scores = model.predict(audio)
                elapsed = (time.perf_counter() - start) * 1000
                state = client.command(label)
                row.update(type='prediction', predicted_class=label, scores=scores.tolist(),
                           pipeline_latency_ms=elapsed, device_state=state)
                print(f'{label} | laptop {elapsed:.2f} ms | LED {"ON" if state["led_on"] else "OFF"} level {state["led_level"]}')
            predictions.write(json.dumps(row) + '\n'); predictions.flush()

        try:
            client.connect()
            print(f'Mikrofon + preprocessing + model: LAPTOP ({model.model_name}). ESP32: LED saja.')
            if not args.continuous:
                while not client.stop.is_set():
                    client.write('READY 1')
                    input('ENTER lalu ucapkan satu kata: nyala/mati/naik/turun (Ctrl+C berhenti)... ')
                    client.write('READY 0')
                    audio = sd.rec(model.window_samples, samplerate=model.fs, channels=1,
                                   dtype='float32', device=args.input_device)
                    sd.wait(); classify(audio[:, 0])
            else:
                blocks = queue.Queue(maxsize=8)
                dropped = threading.Event()

                def capture(indata, frames, timing, status):
                    if status.input_overflow: dropped.set()
                    try: blocks.put_nowait(indata[:, 0].copy())
                    except queue.Full: dropped.set()

                gate = SpeechCapture(model.fs, model.window_samples)
                last_ready = None; started = time.monotonic()
                print('Diam sekitar 2 detik untuk kalibrasi, lalu tunggu [SIAP].')
                with sd.InputStream(samplerate=model.fs, blocksize=model.fs // 20, channels=1,
                                    dtype='float32', device=args.input_device, latency='high', callback=capture):
                    last_audio = time.monotonic()
                    while not client.stop.is_set():
                        if args.seconds and time.monotonic() - started >= args.seconds: break
                        if dropped.is_set():
                            dropped.clear(); gate = SpeechCapture(model.fs, model.window_samples)
                            while not blocks.empty():
                                try: blocks.get_nowait()
                                except queue.Empty: break
                            client.command('mati'); client.write('READY 0')
                            last_ready = False
                            print('Audio drop: LED OFF, kalibrasi ulang.')
                        try: block = blocks.get(timeout=.1)
                        except queue.Empty:
                            if time.monotonic() - last_audio > 5: raise RuntimeError('Tidak ada audio mikrofon selama 5 detik.')
                            continue
                        last_audio = time.monotonic()
                        audio = gate.feed(block)
                        if gate.ready != last_ready:
                            last_ready = gate.ready; client.write('READY 1' if gate.ready else 'READY 0')
                            print('[SIAP]' if gate.ready else '[MEREKAM / TUNGGU HENING]')
                        if audio is not None: classify(audio)
            if client.error: raise RuntimeError(client.error)
        except (KeyboardInterrupt, EOFError): print('\nDihentikan.')
        finally:
            client.close()
            print('Log:', output)
            print('Mode laptop-model ini bukan pengujian acceptance ESP32 inference MP16.')


if __name__ == '__main__':
    try: main()
    except Exception as error: raise SystemExit(f'Laptop KWS LED gagal: {error}') from None
