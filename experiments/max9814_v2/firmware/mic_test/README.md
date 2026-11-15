# Tes live MAX9814 pada ESP32-WROOM/DevKit

Firmware diagnosis terpisah dari MP16. MAX9814 VDD -> 3V3, GND -> GND,
OUT -> GPIO35 / ADC1 channel 7. Matikan daya sebelum mengubah wiring.
Tidak ada klasifikasi atau kendali LED di firmware ini.

## Upload firmware diagnosis

Dari root repository, tutup collector dan semua serial monitor:

```bash
~/.platformio/penv/bin/pio run -d firmware/mic_test -t upload --upload-port /dev/ttyUSB0
```

Upload mengganti program yang sedang tersimpan pada ESP32, tetapi tidak
mengubah kode, model atau artefak MP16 di komputer.

## Monitor live

```bash
.venv/bin/python tools/test_max9814_live.py --port /dev/ttyUSB0
```

Grafik memperlihatkan nilai ADC mentah 0-4095. Meter terminal menampilkan
minimum, rata-rata/bias DC, maksimum, RMS AC (satuan count ADC), persentase
rail, dan error transport. Diam 3-5 detik, lalu bicara dan tepuk tangan.
Perubahan bentuk gelombang dan RMS harus mengikuti suara. Bias DC nonnol
normal untuk modul mikrofon analog; RMS tidak sama dengan rata-rata DC.
Sinyal nyaris tetap, rail besar, atau noise kuat yang tidak mengikuti suara
perlu diperiksa lewat wiring, suplai, gain modul, dan audio yang didengar.
Angka ini bukan dB SPL atau tegangan terkalibrasi; tidak ada ambang PASS
otomatis berdasarkan satu nilai RMS.

Tanpa jendela grafik, gunakan `--no-plot`. Ctrl+C atau tutup grafik untuk
berhenti; `--seconds 15` membatasi durasi. Baud data adalah 921600,
bukan 115200. Host menerima sampel ADC mentah pada nominal 32 kHz, tanpa
model, VAD, peak normalization, maupun filter MP6.

## Mendengarkan live

Gunakan headphone, bukan speaker dekat mikrofon, untuk mencegah feedback.
Mulai dengan volume komputer rendah:

```bash
.venv/bin/python tools/test_max9814_live.py --port /dev/ttyUSB0 --listen --no-plot
```

Playback memakai high-pass 40 Hz untuk membuang bias DC. `--gain 0.5`
adalah default playback saja, bukan gain MAX9814 atau perubahan data model.
Jika perangkat output tidak mendukung mono 32 kHz, jalankan tanpa `--listen`
dan periksa WAV. Perangkat output lain dapat dipilih dengan
`--output-device ID`, sesuai daftar `python -m sounddevice`.

Jika Conda memunculkan error `GLIBCXX_3.4.32 not found` dari `libjack` atau
PortAudio, coba library C++ sistem untuk proses ini saja:

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 .venv/bin/python tools/test_max9814_live.py --port /dev/ttyUSB0 --listen --no-plot
```

Path di atas khusus Linux x86-64 pada komputer proyek ini. Tidak perlu
menghapus library Conda atau mengubah konfigurasi shell global. Tanpa
`--listen`, program tidak memuat sounddevice/PortAudio.

Opsional menyimpan audio high-pass, tanpa gain playback:

```bash
.venv/bin/python tools/test_max9814_live.py --port /dev/ttyUSB0 --no-plot --seconds 15 --save results/mic_test/mic_001.wav
```

Dependensi host mengikuti environment proyek: numpy, scipy, matplotlib,
pyserial, dan sounddevice (yang terakhir hanya untuk `--listen`).
WAV memakai nominal 32 kHz; laju ADC aktual bisa sedikit berbeda. Periksa
counter overflow/missing/host_drops sebelum menyimpulkan kualitas suara.
Program ini tidak membuat status MP16 lulus dan tidak menambah rekaman ke
dataset training/test secara otomatis.

## Kembali ke MP16

```bash
~/.platformio/penv/bin/pio run -d firmware/esp32 -t upload --upload-port /dev/ttyUSB0
```

Firmware mic_test tidak kompatibel dengan collector JSON MP16.
