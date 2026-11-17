# Tes live INMP441 pada ESP32-WROOM/DevKit

Firmware ini terpisah dari tes analog MAX9814 dan klasifikasi MP16.
Pastikan modul benar-benar INMP441 (I2S), bukan mikrofon analog atau PDM.
Matikan daya sebelum merangkai. Jangan memberi 5 V ke VDD mikrofon.

| INMP441 | ESP32 |
| --- | --- |
| VDD / VCC | 3V3 |
| GND | GND |
| SCK / BCLK | GPIO26 |
| WS / LRCLK | GPIO27 |
| SD / DOUT | GPIO33 |
| L/R | GND (kanal kiri) |

L/R jangan dibiarkan mengambang. Jika breakout memiliki CHIPEN/EN, sambungkan
ke 3V3. Breakout biasanya sudah memiliki kapasitor bypass; bila tidak, pasang
100 nF dekat VDD-GND. Datasheet merekomendasikan pulldown 100 kohm pada SD;
firmware juga mengaktifkan pulldown internal ESP32. Gunakan kabel pendek dan
pastikan lubang mikrofon di bagian bawah tidak tertutup.

## Upload

Tutup semua serial monitor dan collector, lalu jalankan dari root proyek:

```bash
cd /home/aliy/Coding/Myrepos/sound-processing-mini-project
~/.platformio/penv/bin/pio run -d firmware/inmp441_test -t upload --upload-port /dev/ttyUSB0
```

Upload mengganti program di board, bukan model atau artefak MP16 di komputer.
Tidak ada kendali LED atau model pengenalan kata pada firmware diagnosis ini.

## Mendengarkan live

Gunakan headphone dan volume komputer rendah agar tidak terjadi feedback.
Suara diputar di komputer, bukan di ESP32.

```bash
.venv/bin/python tools/test_inmp441_live.py --port /dev/ttyUSB0 --listen
```

Jika terkena konflik library C++ Conda/PortAudio pada komputer Linux ini:

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 .venv/bin/python tools/test_inmp441_live.py --port /dev/ttyUSB0 --listen
```

`--no-plot` menonaktifkan grafik. Tanpa `--listen`, tidak ada playback dan
sounddevice/PortAudio tidak dimuat. `--gain 4` adalah default playback;
naikkan perlahan, misalnya `--gain 8`, bila terlalu pelan. Gain tidak mengubah
meter atau rekaman WAV. Jangan menjadikan gain playback sebagai kalibrasi SPL.
`--output-device ID` memilih perangkat audio lain (lihat `python -m sounddevice`).

## Simpan WAV dan meter

```bash
.venv/bin/python tools/test_inmp441_live.py --port /dev/ttyUSB0 --no-plot --seconds 15 --save results/inmp441_test/mic_001.wav
```

Ctrl+C atau tutup grafik untuk berhenti. File WAV yang sudah ada tidak ditimpa.
WAV mono PCM16 nominal 16 kHz memakai high-pass 40 Hz, tanpa gain playback.
Grafik dan meter menunjukkan amplitudo digital sebelum high-pass/gain.
Nilai RMS menggunakan 20*log10(RMS / digital-full-scale), bukan dB SPL.
Sinyal sangat pelan/flat: periksa L/R, SD, common ground dan suplai terlebih dahulu.
L/R=3V3 memerlukan SELECTED_SLOT=1 di src/main.cpp dan upload ulang; default 0
untuk kanal kiri. Nilai zero tidak otomatis membuktikan mikrofon sehat.

Firmware menerima dua slot 32-bit pada 16 kHz (BCLK = 64*16000 = 1,024 MHz)
dan mengirim slot kiri sebagai signed 24-bit left-aligned dalam word 32-bit.
Serial 921600 baud: header little-endian `<8sIHH`, magic `INMP441!`, sequence,
payload byte count (maksimum 640), flags. Payload berisi maksimum 160 sampel.
Flags: 1 = DMA RX overflow, 2 = read/DMA error, 4 = init fatal.
Tidak kompatibel dengan stream ADC MAX9814 atau collector JSON MP16.

Periksa overflow/errors/missing/host_drop sebelum menilai kualitas audio.
Buffer host dibatasi; audio tertua dibuang bila host tertinggal. WAV tidak
mengisi gap dengan sampel buatan: rekaman dengan drop bukan bukti timing kontinu.
Tidak ada peak normalization, auto-gain, klasifikasi, atau perubahan status MP16.
Dependensi minimum: `pip install -r firmware/inmp441_test/requirements-host.txt`.

## Kembali ke MP16 MAX9814

```bash
~/.platformio/penv/bin/pio run -d firmware/esp32 -t upload --upload-port /dev/ttyUSB0
```

Firmware MP16 saat ini tetap membutuhkan MAX9814 OUT -> GPIO35. INMP441 tidak
otomatis menggantikan MAX9814 pada model tersebut; frontend dan akurasi perlu
divalidasi terpisah sebelum migrasi perangkat/dataset.

## Referensi

- [INMP441 datasheet, TDK](https://product.tdk.com/system/files/dam/doc/product/sw_piezo/mic/mems-mic/data_sheet/inmp441.pdf)
- [I2S driver ESP-IDF 4.4.7](https://docs.espressif.com/projects/esp-idf/en/v4.4.7/esp32/api-reference/peripherals/i2s.html)

