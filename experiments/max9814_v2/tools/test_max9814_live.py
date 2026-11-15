"""Live raw-ADC waveform, level meter, optional headphone playback and WAV capture."""
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

MAGIC = b'MIC9814!'
HEADER = struct.Struct('<8sIHH')
SAMPLE_RATE = 32000


def load_sounddevice():
    try:
        return importlib.import_module('sounddevice')
    except (ImportError, OSError) as error:
        message = f'Playback tidak dapat dibuka: {error}\n'
        system_cpp = Path('/usr/lib/x86_64-linux-gnu/libstdc++.so.6')
        if 'GLIBCXX_' in str(error) and system_cpp.exists():
            command = shlex.join([sys.executable, *sys.argv])
            message += ('Library C++ Conda tidak cocok dengan PortAudio/JACK sistem.\n'
                        'Coba untuk proses ini saja, tanpa mengubah Conda:\n'
                        f'LD_PRELOAD={system_cpp} {command}\n')
        else:
            message += 'Periksa instalasi sounddevice dan library PortAudio sistem.\n'
        message += ('Tes mikrofon tetap bisa dijalankan tanpa --listen. '
                    'Gunakan --no-plot untuk meter terminal saja.')
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
                self.buffer[:] = self.buffer[-7:]
                break
            if start:
                del self.buffer[:start]
            if len(self.buffer) < HEADER.size:
                break
            _, sequence, size, flags = HEADER.unpack_from(self.buffer)
            if size > 640 or size % 2 or flags & ~7:
                del self.buffer[0]
                continue
            end = HEADER.size + size
            if len(self.buffer) < end:
                break
            frames.append((sequence, flags, bytes(self.buffer[HEADER.size:end])))
            del self.buffer[:end]
        return frames


def decode_adc(payload):
    words = np.frombuffer(payload, dtype='<u2')
    valid = (words >> 12) == 7
    return (words[valid] & 4095).astype(np.float32), int(np.count_nonzero(~valid))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', required=True)
    parser.add_argument('--listen', action='store_true', help='Playback live; gunakan headphone')
    parser.add_argument('--gain', type=float, default=0.5, help='Gain playback saja, default 0.5')
    parser.add_argument('--output-device', type=int)
    parser.add_argument('--no-plot', action='store_true', help='Meter terminal tanpa jendela grafik')
    parser.add_argument('--seconds', type=float, default=0, help='0 = sampai Ctrl+C')
    parser.add_argument('--save', type=Path, help='Simpan audio ADC high-pass sebagai WAV mono 32 kHz')
    args = parser.parse_args()
    if not np.isfinite(args.gain) or not 0 < args.gain <= 4:
        parser.error('--gain harus > 0 dan <= 4')
    if not np.isfinite(args.seconds) or args.seconds < 0:
        parser.error('--seconds harus >= 0')
    if args.save and args.save.exists():
        parser.error('--save sudah ada; pilih nama baru agar rekaman tidak tertimpa')
    sd = load_sounddevice() if args.listen else None
    import serial
    packets = queue.Queue(maxsize=200)
    audio = queue.Queue(maxsize=100)
    stopped = threading.Event()
    failures = []
    counters = {'host_drops': 0, 'playback_underflows': 0}

    def read_serial(device):
        framing = FrameParser()
        try:
            while not stopped.is_set():
                data = device.read(2048)
                for frame in framing.feed(data):
                    try:
                        packets.put_nowait(frame)
                    except queue.Full:
                        counters['host_drops'] += 1
        except Exception as error:
            failures.append(str(error))
            stopped.set()

    pending = np.empty(0, np.float32)
    def playback(outdata, frames, timing, status):
        nonlocal pending
        outdata.fill(0)
        at = 0
        while at < frames:
            if not pending.size:
                try:
                    pending = audio.get_nowait()
                except queue.Empty:
                    counters['playback_underflows'] += 1
                    break
            count = min(frames - at, pending.size)
            outdata[at:at + count, 0] = pending[:count]
            pending = pending[count:]
            at += count

    with contextlib.ExitStack() as stack:
        output = None
        if args.save:
            args.save.parent.mkdir(parents=True, exist_ok=True)
            output = stack.enter_context(wave.open(str(args.save), 'wb'))
            output.setparams((1, 2, SAMPLE_RATE, 0, 'NONE', 'not compressed'))
        if args.listen:
            stack.enter_context(sd.OutputStream(samplerate=SAMPLE_RATE, channels=1,
                dtype='float32', device=args.output_device, callback=playback))
            print('Playback aktif: gunakan headphone dan mulai dengan volume rendah.')
        if not args.no_plot:
            import matplotlib.pyplot as plt
            plt.ion()
            fig, ax = plt.subplots(figsize=(10, 4))
            line, = ax.plot(np.arange(6400) / SAMPLE_RATE, np.zeros(6400), linewidth=0.8)
            ax.set(xlabel='Waktu (s)', ylabel='ADC count (0-4095)', ylim=(0, 4095),
                   title='MAX9814 live - GPIO35')
            ax.grid(alpha=0.2)
            plt.show(block=False)
        device = stack.enter_context(serial.Serial(args.port, 921600, timeout=0.05))
        device.dtr = False
        device.rts = False
        reader = threading.Thread(target=read_serial, args=(device,), daemon=True)
        reader.start()
        def close_reader():
            stopped.set()
            reader.join()
        stack.callback(close_reader)
        raw_history = deque(maxlen=6400)
        filter_sos = butter(2, 40, 'highpass', fs=SAMPLE_RATE, output='sos')
        state = None
        started = last_frame = last_print = time.monotonic()
        samples = invalid = overflow = errors = missing = 0
        previous = None
        print('Tes live aktif. Diam beberapa detik, lalu bicara atau tepuk tangan. Ctrl+C untuk berhenti.')
        try:
            while not stopped.is_set():
                now = time.monotonic()
                if args.seconds and now - started >= args.seconds:
                    break
                try:
                    sequence, flags, payload = packets.get(timeout=0.03)
                except queue.Empty:
                    if now - last_frame > 5:
                        raise RuntimeError('Tidak ada paket audio selama 5 detik. Upload firmware mic_test, cek port dan baud 921600.')
                else:
                    last_frame = now
                    if previous is not None:
                        missing += (sequence - previous - 1) & 0xffffffff
                    previous = sequence
                    if flags & 4:
                        raise RuntimeError('Inisialisasi ADC di board gagal.')
                    overflow += bool(flags & 1)
                    errors += bool(flags & 2)
                    raw, bad = decode_adc(payload)
                    invalid += bad
                    if raw.size:
                        samples += raw.size
                        raw_history.extend(raw.tolist())
                        x = raw / 2048.0
                        if state is None or flags or bad:
                            state = sosfilt_zi(filter_sos) * x[0]
                        signal, state = sosfilt(filter_sos, x, zi=state)
                        if output:
                            output.writeframesraw(np.round(np.clip(signal, -1, 1) * 32767).astype('<i2').tobytes())
                        if args.listen:
                            try:
                                audio.put_nowait(np.clip(signal * args.gain, -1, 1).astype(np.float32))
                            except queue.Full:
                                counters['host_drops'] += 1
                if now - last_print >= 0.25 and raw_history:
                    data = np.asarray(raw_history)
                    rms = float(np.std(data))
                    rail = float(np.mean((data <= 8) | (data >= 4087))) * 100
                    meter = '#' * min(40, int(rms / 15))
                    print(f'ADC min={data.min():4.0f} mean={data.mean():6.1f} max={data.max():4.0f} '
                          f'AC_RMS={rms:6.1f} rail={rail:5.1f}% overflow={overflow} '
                          f'errors={errors} missing={missing} invalid={invalid} [{meter:<40}]')
                    if not args.no_plot:
                        values = np.zeros(6400)
                        values[-len(data):] = data
                        line.set_ydata(values)
                        ax.set_title(f'MAX9814 GPIO35 | RMS {rms:.1f} counts | rail {rail:.1f}%')
                        fig.canvas.draw_idle()
                    last_print = now
                if not args.no_plot:
                    plt.pause(0.001)
                    if not plt.fignum_exists(fig.number):
                        break
        except KeyboardInterrupt:
            pass
        finally:
            if not args.no_plot:
                plt.close(fig)
        if failures:
            raise RuntimeError(failures[0])
        if samples == 0:
            raise RuntimeError('Tidak ada sampel ADC1 channel 7 yang valid.')
        print(f'Selesai: {samples} sampel; {counters}; overflow={overflow}, errors={errors}, missing={missing}.')
        if args.save:
            print('WAV:', args.save.resolve())
        print('Ini diagnosis mikrofon, bukan pengukuran akurasi atau acceptance MP16.')


if __name__ == '__main__':
    main()
