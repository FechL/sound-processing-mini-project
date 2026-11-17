# Tes mikrofon laptop live

Program `test_laptop_mic_live.py` memakai mikrofon komputer (internal/USB), bukan
ESP32, MAX9814 atau INMP441. Tidak perlu upload firmware atau port serial.
Program tidak mengubah dataset, model, konfigurasi, atau status MP1--16.

Dari root proyek:

```bash
cd /home/aliy/Coding/Myrepos/sound-processing-mini-project
.venv/bin/python tools/test_laptop_mic_live.py
```

Secara default: grafik, meter RMS dBFS/peak/clipping, mono, sample rate native
perangkat input OS. Pilih mikrofon internal pada pengaturan OS atau gunakan
`--list-devices` dan `--input-device ID` (nama perangkat juga boleh).
Headset yang terpasang bisa menjadi input default, jadi cek nama input yang
dicetak program. dBFS adalah level digital, bukan dB SPL terkalibrasi.

```bash
.venv/bin/python tools/test_laptop_mic_live.py --list-devices
.venv/bin/python tools/test_laptop_mic_live.py --input-device 2 --no-plot
```

Untuk mendengar suara sendiri secara live, gunakan headphone dan volume rendah:

```bash
.venv/bin/python tools/test_laptop_mic_live.py --listen
```

`--output-device ID` memilih headphone; default mengikuti perangkat output OS.
`--gain 1` adalah default playback; ubah perlahan bila perlu (maksimum 16).
Gain playback tidak mengubah meter atau file WAV. Tanpa `--listen`, tidak ada
audio keluar. Jangan memakai speaker dekat mikrofon karena feedback.

Jika konflik C++ Conda/PortAudio/JACK muncul pada mesin Linux x86-64 ini:

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 .venv/bin/python tools/test_laptop_mic_live.py --listen
```

Tanpa listen pun program membutuhkan sounddevice/PortAudio untuk merekam input.
Dependensi Python: numpy, matplotlib, sounddevice (sudah ada di requirements.txt).
Mikrofon harus diizinkan oleh OS dan tidak dimute.

## Rekaman opsional

```bash
.venv/bin/python tools/test_laptop_mic_live.py --no-plot --seconds 10 --save results/laptop_mic/mic_001.wav
```

WAV mono PCM16, tanpa peak normalization, high-pass, atau gain playback.
File lama tidak ditimpa. `--seconds 0` berarti sampai Ctrl+C; tutup grafik juga
menghentikan tes. Program menutup stream dan memfinalisasi header WAV saat keluar.
Buffer callback dibatasi; counter host_dropped_blocks/input_overflows menunjukkan
rekaman tidak kontinu bila host/perangkat tertinggal. Tidak ada pengisian gap palsu.

Jika perlu 16 kHz, tambahkan `--sample-rate 16000`, bila perangkat mendukung.
Jika rate/output tidak cocok, pilih perangkat lain atau coba 48000 Hz. Tidak ada
resampling tersembunyi di program; WAV memakai rate yang diminta ke PortAudio.
Ini diagnosis input laptop, bukan pengukuran akurasi model atau acceptance MP16.

[Dokumentasi stream sounddevice](https://python-sounddevice.readthedocs.io/en/latest/api/streams.html)
