"""Live laptop/USB microphone meter, waveform, optional headphone monitor and WAV."""
from pathlib import Path
import argparse
from collections import deque
import contextlib
import importlib
import queue
import shlex
import sys
import time
import wave

import numpy as np


def load_sounddevice():
    try:
        return importlib.import_module('sounddevice')
    except (ImportError, OSError) as error:
        message = f'Audio laptop tidak dapat dibuka: {error}\n'
        cpp = Path('/usr/lib/x86_64-linux-gnu/libstdc++.so.6')
        if 'GLIBCXX_' in str(error) and cpp.exists():
            message += ('Library C++ Conda tidak cocok dengan PortAudio sistem. Coba:\n'
                        f'LD_PRELOAD={cpp} {shlex.join([sys.executable, *sys.argv])}\n')
        message += 'Program ini membutuhkan sounddevice dan library PortAudio, termasuk tanpa --listen.'
        raise SystemExit(message) from None


def device_selector(value):
    try:
        return int(value)
    except ValueError:
        return value


def pcm16(samples):
    return np.round(np.clip(samples, -1, 1) * 32767).astype('<i2').tobytes()


def levels(samples):
    samples = np.asarray(samples, dtype=np.float64)
    rms = float(np.sqrt(np.mean(samples ** 2)))
    return {
        'rms_dbfs': float(20 * np.log10(max(rms, 1e-9))),
        'peak': float(np.max(np.abs(samples))),
        'clipping_percent': float(100 * np.mean(np.abs(samples) >= 0.99)),
    }


class AudioCapture:
    def __init__(self, gain, max_blocks=25):
        self.gain = gain
        self.blocks = queue.Queue(maxsize=max_blocks)
        self.counters = {'host_dropped_blocks': 0, 'input_overflows': 0,
                         'output_underflows': 0, 'status_warnings': 0}

    def capture(self, indata, frames, timing, status):
        if status:
            self.counters['status_warnings'] += 1
            self.counters['input_overflows'] += bool(status.input_overflow)
            self.counters['output_underflows'] += bool(status.output_underflow)
        block = indata[:, 0].copy()
        try:
            self.blocks.put_nowait(block)
        except queue.Full:
            try:
                self.blocks.get_nowait()
                self.counters['host_dropped_blocks'] += 1
            except queue.Empty:
                pass
            self.blocks.put_nowait(block)

    def monitor(self, indata, outdata, frames, timing, status):
        # Playback never waits for plotting or disk IO; mono broadcasts to both ears.
        np.multiply(indata[:, :1], self.gain, out=outdata)
        np.clip(outdata, -1, 1, out=outdata)
        self.capture(indata, frames, timing, status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--list-devices', action='store_true', help='Daftar perangkat audio, tanpa merekam')
    parser.add_argument('--input-device', type=device_selector, help='ID atau potongan nama mikrofon; default mikrofon OS')
    parser.add_argument('--output-device', type=device_selector, help='ID atau nama headphone untuk --listen')
    parser.add_argument('--listen', action='store_true', help='Monitor suara ke headphone; hindari speaker')
    parser.add_argument('--gain', type=float, default=1.0, help='Gain playback saja, >0 sampai 16; default 1')
    parser.add_argument('--sample-rate', type=int, help='Hz; default sample rate native mikrofon')
    parser.add_argument('--no-plot', action='store_true', help='Meter terminal saja')
    parser.add_argument('--seconds', type=float, default=0, help='0 = sampai Ctrl+C')
    parser.add_argument('--save', type=Path, help='Simpan input mono PCM16 tanpa gain playback')
    args = parser.parse_args()
    if not np.isfinite(args.gain) or not 0 < args.gain <= 16:
        parser.error('--gain harus > 0 dan <= 16')
    if not np.isfinite(args.seconds) or args.seconds < 0:
        parser.error('--seconds harus >= 0')
    if args.sample_rate is not None and not 8000 <= args.sample_rate <= 192000:
        parser.error('--sample-rate harus 8000 sampai 192000 Hz')
    if args.output_device is not None and not args.listen and not args.list_devices:
        parser.error('--output-device membutuhkan --listen')
    if args.save and args.save.exists() and not args.list_devices:
        parser.error('--save sudah ada; pilih nama baru agar rekaman tidak tertimpa')
    sd = load_sounddevice()
    if args.list_devices:
        print(sd.query_devices())
        return
    try:
        info = sd.query_devices(args.input_device, 'input')
        fs = args.sample_rate or int(round(info['default_samplerate']))
        sd.check_input_settings(device=args.input_device, channels=1, dtype='float32', samplerate=fs)
        output_channels = 1
        if args.listen:
            output_info = sd.query_devices(args.output_device, 'output')
            output_channels = min(2, int(output_info['max_output_channels']))
            sd.check_output_settings(device=args.output_device, channels=output_channels,
                                     dtype='float32', samplerate=fs)
    except Exception as error:
        raise RuntimeError(f'Perangkat/rate audio tidak cocok: {error}. '
                           'Gunakan --list-devices, pilih --input-device/--output-device, '
                           'atau coba --sample-rate 48000 / tanpa --listen.') from None
    capture = AudioCapture(args.gain)
    kwargs = dict(samplerate=fs, blocksize=max(1, fs // 50), dtype='float32')
    if args.listen:
        stream = sd.Stream(device=(args.input_device, args.output_device),
                           channels=(1, output_channels), callback=capture.monitor, **kwargs)
    else:
        stream = sd.InputStream(device=args.input_device, channels=1, callback=capture.capture, **kwargs)
    with contextlib.ExitStack() as stack:
        # Close the stream even if figure/WAV initialization fails before entering it.
        stack.callback(stream.close)
        plt = None
        history_size = max(1, fs // 5)
        if not args.no_plot:
            import matplotlib.pyplot as plt
            plt.ion()
            fig, ax = plt.subplots(figsize=(10, 4))
            line, = ax.plot(np.arange(history_size) / fs, np.zeros(history_size), linewidth=0.8)
            ax.set(xlabel='Waktu (s)', ylabel='Amplitudo input (tanpa gain)',
                   ylim=(-1, 1), title='Mikrofon laptop live')
            ax.grid(alpha=0.2)
            plt.show(block=False)
            stack.callback(plt.close, fig)
        output = None
        if args.save:
            args.save.parent.mkdir(parents=True, exist_ok=True)
            output = stack.enter_context(wave.open(stack.enter_context(args.save.open('xb')), 'wb'))
            output.setparams((1, 2, fs, 0, 'NONE', 'not compressed'))
        print(f'Input: {info["name"]} | mono {fs} Hz | Ctrl+C untuk berhenti.')
        if args.listen:
            print('Playback aktif: gunakan headphone dan volume komputer rendah.')
        history = deque(maxlen=history_size)
        samples = 0
        try:
            with stream:
                started = last_audio = last_print = time.monotonic()
                while True:
                    now = time.monotonic()
                    if args.seconds and now - started >= args.seconds:
                        break
                    try:
                        block = capture.blocks.get(timeout=0.05)
                    except queue.Empty:
                        if not stream.active:
                            raise RuntimeError('Stream audio berhenti. Periksa perangkat dan izin mikrofon.')
                        if now - last_audio > 5:
                            raise RuntimeError('Tidak ada blok audio selama 5 detik. Periksa izin mikrofon dan input OS.')
                    else:
                        samples += block.size
                        history.extend(block.tolist())
                        last_audio = now
                        if output:
                            output.writeframesraw(pcm16(block))
                    if now - last_print >= 0.25 and history:
                        data = np.asarray(history)
                        value = levels(data)
                        meter = '#' * min(40, max(0, int((value['rms_dbfs'] + 80) / 2)))
                        print(f'RMS={value["rms_dbfs"]:6.1f} dBFS peak={value["peak"]:.4f} '
                              f'clip={value["clipping_percent"]:.2f}% '
                              f'drop={capture.counters["host_dropped_blocks"]} [{meter:<40}]')
                        if plt:
                            values = np.zeros(history_size)
                            values[-len(data):] = data
                            line.set_ydata(values)
                            ax.set_title(f'{info["name"]} | RMS {value["rms_dbfs"]:.1f} dBFS')
                            fig.canvas.draw_idle()
                        last_print = now
                    if plt:
                        plt.pause(0.001)
                        if not plt.fignum_exists(fig.number):
                            break
        except KeyboardInterrupt:
            pass
        if samples == 0:
            raise RuntimeError('Tidak ada sampel yang diterima dari mikrofon.')
        print(f'Selesai: {samples} sampel ({samples / fs:.2f} s audio); {capture.counters}.')
        if args.save:
            print('WAV:', args.save.resolve())
        print('Ini tes input laptop, bukan perekaman dataset atau acceptance MP16.')


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        raise SystemExit(f'Tes mikrofon laptop gagal: {error}') from None

