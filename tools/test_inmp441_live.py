"""Listen to INMP441 over ESP32 I2S/USB; optional waveform, meter and WAV capture."""
from pathlib import Path
import argparse
from collections import deque
import contextlib
import importlib
import queue
import shlex
import struct
import sys
import threading
import time
import wave

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi

MAGIC = b'INMP441!'
HEADER = struct.Struct('<8sIHH')
SAMPLE_RATE = 16000
BAUD = 921600
BLOCK_SAMPLES = 160


def load_sounddevice():
    try:
        return importlib.import_module('sounddevice')
    except (ImportError, OSError) as error:
        message = f'Playback tidak dapat dibuka: {error}\n'
        cpp = Path('/usr/lib/x86_64-linux-gnu/libstdc++.so.6')
        if 'GLIBCXX_' in str(error) and cpp.exists():
            message += ('Library C++ Conda tidak cocok dengan PortAudio sistem. Coba:\n'
                        f'LD_PRELOAD={cpp} {shlex.join([sys.executable, *sys.argv])}\n')
        message += 'Periksa sounddevice/PortAudio, atau jalankan tanpa --listen.'
        raise SystemExit(message) from None


class FrameParser:
    def __init__(self):
        self.buffer = bytearray()

    def feed(self, data):
        self.buffer.extend(data)
        frames = []
        while True:
            start = self.buffer.find(MAGIC)
            if start < 0:
                self.buffer[:] = self.buffer[-(len(MAGIC) - 1):]
                break
            if start:
                del self.buffer[:start]
            if len(self.buffer) < HEADER.size:
                break
            _, sequence, size, flags = HEADER.unpack_from(self.buffer)
            if size > BLOCK_SAMPLES * 4 or size % 4 or flags & ~7 or (size == 0 and not flags):
                del self.buffer[0]
                continue
            end = HEADER.size + size
            if len(self.buffer) < end:
                break
            frames.append((sequence, flags, bytes(self.buffer[HEADER.size:end])))
            del self.buffer[:end]
        return frames


def decode_pcm(payload):
    """Signed 24-bit INMP441 values occupy the upper bits of each 32-bit word."""
    if len(payload) % 4:
        raise ValueError('Payload INMP441 harus kelipatan empat byte.')
    values = np.frombuffer(payload, dtype='<i4') >> 8
    return values.astype(np.float32) / 8388608.0


class PlaybackBuffer:
    def __init__(self, max_samples=SAMPLE_RATE // 10):
        if max_samples <= 0:
            raise ValueError('Playback buffer harus positif.')
        self.max_samples = max_samples
        self.blocks = deque()
        self.available = 0
        self.lock = threading.Lock()

    def push(self, samples):
        samples = np.asarray(samples, dtype=np.float32)
        dropped = max(0, samples.size - self.max_samples)
        samples = samples[-self.max_samples:]
        with self.lock:
            # Discard oldest audio on a slow host, keeping live monitoring bounded.
            while self.blocks and self.available + samples.size > self.max_samples:
                block = self.blocks.popleft()
                self.available -= block.size
                dropped += block.size
            if samples.size:
                self.blocks.append(samples.copy())
                self.available += samples.size
        return dropped

    def read_into(self, outdata):
        outdata.fill(0)
        at = 0
        with self.lock:
            while self.blocks and at < len(outdata):
                block = self.blocks.popleft()
                count = min(block.size, len(outdata) - at)
                outdata[at:at + count, 0] = block[:count]
                self.available -= count
                at += count
                if count < block.size:
                    self.blocks.appendleft(block[count:])
        return at < len(outdata)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', required=True)
    parser.add_argument('--listen', action='store_true', help='Putar audio di komputer; gunakan headphone')
    parser.add_argument('--gain', type=float, default=4.0, help='Gain playback saja, >0 sampai 64; default 4')
    parser.add_argument('--output-device', type=int)
    parser.add_argument('--no-plot', action='store_true', help='Meter terminal tanpa jendela grafik')
    parser.add_argument('--seconds', type=float, default=0, help='0 = sampai Ctrl+C')
    parser.add_argument('--save', type=Path, help='WAV PCM16 mono 16 kHz, high-pass tanpa gain playback')
    args = parser.parse_args()
    if not np.isfinite(args.gain) or not 0 < args.gain <= 64:
        parser.error('--gain harus > 0 dan <= 64')
    if not np.isfinite(args.seconds) or args.seconds < 0:
        parser.error('--seconds harus >= 0')
    if args.save and args.save.exists():
        parser.error('--save sudah ada; pilih nama baru agar rekaman tidak tertimpa')
    sd = load_sounddevice() if args.listen else None
    import serial

    packets = queue.Queue(maxsize=30)
    audio = PlaybackBuffer()
    stopped = threading.Event()
    failures = []
    counters = {'host_dropped_packets': 0, 'playback_dropped_samples': 0,
                'playback_underflows': 0, 'output_status_warnings': 0}

    def read_serial(device):
        framing = FrameParser()
        try:
            while not stopped.is_set():
                for frame in framing.feed(device.read(1024)):
                    if packets.full():
                        try:
                            packets.get_nowait()
                            counters['host_dropped_packets'] += 1
                        except queue.Empty:
                            pass
                    packets.put_nowait(frame)
        except Exception as error:
            failures.append(str(error))
            stopped.set()

    def playback(outdata, frames, timing, status):
        if status:
            counters['output_status_warnings'] += 1
        if audio.read_into(outdata):
            counters['playback_underflows'] += 1

    with contextlib.ExitStack() as stack:
        output = None
        if args.save:
            args.save.parent.mkdir(parents=True, exist_ok=True)
            output = stack.enter_context(wave.open(stack.enter_context(args.save.open('xb')), 'wb'))
            output.setparams((1, 2, SAMPLE_RATE, 0, 'NONE', 'not compressed'))
        if args.listen:
            stack.enter_context(sd.OutputStream(samplerate=SAMPLE_RATE, channels=1,
                dtype='float32', device=args.output_device, callback=playback))
            print('Playback headphone aktif. Mulai dengan volume komputer rendah.')
        plt = None
        if not args.no_plot:
            import matplotlib.pyplot as plt
            plt.ion()
            fig, ax = plt.subplots(figsize=(10, 4))
            history_size = SAMPLE_RATE // 5
            line, = ax.plot(np.arange(history_size) / SAMPLE_RATE, np.zeros(history_size), linewidth=0.8)
            ax.set(xlabel='Waktu (s)', ylabel='Amplitudo digital (tanpa gain)',
                   ylim=(-0.1, 0.1), title='INMP441 live - I2S GPIO26/27/33')
            ax.grid(alpha=0.2)
            plt.show(block=False)
            stack.callback(plt.close, fig)
        device = stack.enter_context(serial.Serial(args.port, BAUD, timeout=0.05))
        device.dtr = False
        device.rts = False
        reader = threading.Thread(target=read_serial, args=(device,), daemon=True)
        reader.start()

        def close_reader():
            stopped.set()
            reader.join()

        stack.callback(close_reader)
        history = deque(maxlen=SAMPLE_RATE // 5)
        highpass = butter(2, 40, 'highpass', fs=SAMPLE_RATE, output='sos')
        state = None
        started = last_frame = last_print = time.monotonic()
        samples = overflow = errors = missing = resets = 0
        previous = None
        print('INMP441 live: bicara dekat mikrofon. Ctrl+C untuk berhenti.')
        try:
            while not stopped.is_set():
                now = time.monotonic()
                if args.seconds and now - started >= args.seconds:
                    break
                try:
                    sequence, flags, payload = packets.get(timeout=0.03)
                except queue.Empty:
                    if now - last_frame > 5:
                        raise RuntimeError('Tidak ada paket INMP441 selama 5 detik. Upload firmware/inmp441_test, cek port dan baud 921600.')
                else:
                    last_frame = now
                    discontinuity = False
                    if previous is not None:
                        gap = (sequence - previous - 1) & 0xffffffff
                        if gap < 0x80000000:
                            missing += gap
                        else:
                            resets += 1
                        discontinuity = gap != 0
                    previous = sequence
                    if flags & 4:
                        raise RuntimeError('Inisialisasi I2S gagal di ESP32. Periksa firmware dan pin.')
                    overflow += bool(flags & 1)
                    errors += bool(flags & 2)
                    raw = decode_pcm(payload)
                    if raw.size:
                        samples += raw.size
                        history.extend(raw.tolist())
                        if state is None or flags or discontinuity:
                            state = sosfilt_zi(highpass) * raw[0]
                        signal, state = sosfilt(highpass, raw, zi=state)
                        if output:
                            pcm = np.round(np.clip(signal, -1, 1) * 32767).astype('<i2')
                            output.writeframesraw(pcm.tobytes())
                        if args.listen:
                            counters['playback_dropped_samples'] += audio.push(np.clip(signal * args.gain, -1, 1))
                    elif flags:
                        state = None
                if now - last_print >= 0.25 and history:
                    data = np.asarray(history)
                    rms = float(np.std(data))
                    dbfs = 20 * np.log10(max(rms, 1e-9))
                    peak = float(np.max(np.abs(data)))
                    clip = float(np.mean(np.abs(data) >= 0.99)) * 100
                    meter = '#' * min(40, max(0, int((dbfs + 80) / 2)))
                    print(f'RMS={dbfs:6.1f} dBFS peak={peak:.5f} clip={clip:.2f}% '
                          f'overflow={overflow} errors={errors} missing={missing} '
                          f'host_drop={counters["host_dropped_packets"]} [{meter:<40}]')
                    if plt:
                        values = np.zeros(history_size)
                        values[-len(data):] = data
                        line.set_ydata(values)
                        limit = max(0.1, peak * 1.1)
                        ax.set_ylim(-limit, limit)
                        ax.set_title(f'INMP441 I2S | RMS {dbfs:.1f} dBFS | peak {peak:.5f}')
                        fig.canvas.draw_idle()
                    last_print = now
                if plt:
                    plt.pause(0.001)
                    if not plt.fignum_exists(fig.number):
                        break
        except KeyboardInterrupt:
            pass
        stopped.set()
        reader.join()
        if failures:
            raise RuntimeError(failures[0])
        if samples == 0:
            raise RuntimeError('Tidak ada sampel INMP441. Pastikan firmware diagnosis sudah diupload.')
        print(f'Selesai: {samples} sampel; {counters}; overflow={overflow}, '
              f'errors={errors}, missing={missing}, resets={resets}.')
        if args.save:
            print('WAV:', args.save.resolve())
        print('Ini tes mikrofon, bukan klasifikasi atau acceptance MP16.')


if __name__ == '__main__':
    try:
        main()
    except (OSError, RuntimeError) as error:
        raise SystemExit(f'Tes INMP441 gagal: {error}') from None

