# MP16: ESP32-WROOM/DevKit + MAX9814 + LED

Firmware ini menjalankan pengenalan empat command di ESP32. Komputer hanya
dipakai untuk upload dan pengumpulan log. Paket model linear INT8 dari MP15
dipakai apa adanya, tanpa retraining atau pengubahan scaler/urutan fitur.

## Wiring

Matikan suplai ketika memasang kabel. Semua komponen memakai ground bersama.

| Komponen | Sambungan |
| --- | --- |
| MAX9814 VDD | ESP32 3V3 |
| MAX9814 GND | ESP32 GND |
| MAX9814 OUT | GPIO35 / ADC1 channel 7 |
| LED user bawaan board (indikator SIAP) | GPIO2, sambungan internal board |
| LED eksternal, anode (+) | GPIO25 melalui resistor seri 330 ohm |
| LED eksternal, cathode (-) | GND |

Default memakai satu LED indikator biasa (misalnya merah 3/5 mm) dan resistor
330 ohm, minimal 1/4 watt. Rangkai seri: GPIO25 -> resistor -> anode LED (+),
cathode LED (-) -> GND. Anode umumnya kaki panjang; cathode kaki pendek/sisi
datar kemasan. Periksa datasheet bila kaki sudah dipotong atau penanda berbeda.
Resistor tidak berpolaritas dan boleh berada di sisi anode maupun cathode,
asalkan tetap seri. Jangan memasang LED langsung tanpa resistor atau menghubungkan
anode ke 5V/3V3 untuk rangkaian active-HIGH ini. Gunakan ground bersama MAX9814.
Firmware memakai KWS_LED_GPIO=25 dan KWS_LED_ACTIVE_LOW=0 untuk output perintah.
LED user bawaan GPIO2 dipakai terpisah sebagai indikator SIAP: ON setelah
kalibrasi, gate armed, hening mencukupi, dan cooldown selesai; OFF selama belum
siap atau stream audio berhenti. Tidak perlu kabel tambahan jika board memiliki
LED user GPIO2. LED power tidak dapat dipakai sebagai indikator ini; beberapa
varian DevKit hanya memiliki LED power. Untuk LED user aktif LOW, atur
KWS_READY_LED_ACTIVE_LOW=1. Jangan menambahkan beban HIGH pada GPIO2 saat boot.
Log ready_for_command dan tampilan SIAP memakai kondisi yang sama dengan
indikator bawaan. LED eksternal tidak berubah ketika indikator SIAP berganti.
Sambungan GPIO25 ini untuk LED indikator kecil, bukan LED daya, strip LED,
lampu atau relay yang memerlukan driver terpisah.
MAX9814 beroutput analog;
tidak memakai koneksi mikrofon I2S digital. Gunakan suplai modul 3,3 V dan pastikan
tegangan OUT sesuai rentang ADC ESP32. Kabel audio sebaiknya pendek. Pin GAIN/AR
dapat memakai default modul untuk pengujian awal; catat setelannya pada laporan.

## Build dan Upload

PlatformIO Core sudah tersedia pada komputer ini. Jalankan dari root repository:

```bash
.venv/bin/python tools/export_mp16_esp32.py
.venv/bin/python tools/check_mp16_esp32.py --output results/metrics/mp16_esp32_equivalence.json
~/.platformio/penv/bin/pio run -d firmware/esp32
~/.platformio/penv/bin/pio run -d firmware/esp32 -t upload --upload-port /dev/ttyUSB0
~/.platformio/penv/bin/pio device monitor -p /dev/ttyUSB0 -b 460800
```

Sesuaikan port USB. PlatformIO memakai board `esp32dev`, Arduino ESP32 2.0.17
dan platform versi 7.1.3. Toolchain menghasilkan `.pio/build/esp32dev/firmware.bin`.
Jika upload berhenti pada Connecting, tahan BOOT ketika mulai menghubungkan lalu
lepaskan setelah proses penulisan dimulai. Jangan mengedit header model beku.

Ekspor diperlukan lagi setelah menjalankan MP15 dengan paket baru. Pre-build
guard menolak header/model/config yang tidak cocok dengan SHA-256 ekspor.
Perubahan pin atau polaritas LED juga memerlukan ekspor ulang sebelum build.
KISS FFT 131.1.0 disertakan dalam `lib/kissfft` beserta lisensinya agar FFT
400 sampel sama dengan training, bukan diganti ke 512 sampel.

## Perilaku

Sesudah reset, diam sekitar tujuh detik untuk pemanasan ADC, pengisian buffer,
dan kalibrasi ambient 40 window. LED tetap OFF selama kalibrasi. Median RMS
ambient menjadi baseline awal; speech gate memakai onset 3x
dan release 1,5x baseline (dengan batas minimum RMS 0,008). Hysteresis ini
memungkinkan noise lantai yang tidak nol kembali dianggap hening. Ini bukan
training classifier dan tidak memakai label demo atau frozen test. Jika ada
ucapan/noise keras selama kalibrasi atau lingkungan berubah, reset dan ulangi
di keadaan hening. Jangan menaikkan ambang untuk sekadar membuat acceptance PASS.
Baseline mengikuti RMS secara bertahap (EMA alpha 0,1 per window) hanya ketika
gate tidak aktif dan RMS masih di bawah onset. Keputusan onset dilakukan
sebelum adaptasi; ucapan aktif tidak menaikkan ambangnya sendiri. Pelacakan
dua arah ini menghindari baseline yang terus merosot menuju minimum noise
sehingga pelepasan menjadi terlalu sulit. Jika noise berubah mendadak hingga
melewati onset terus-menerus, reset dan kalibrasi ulang di keadaan hening.
Ucapkan satu command, lalu beri jeda hening setidaknya dua detik.

| Command | Aksi LED |
| --- | --- |
| nyala | ON pada level kecerahan tersimpan |
| mati | OFF |
| naik | Tambah satu level, maksimum 5 |
| turun | Kurangi satu level, minimum 1 |

Level awal 3; rentang level 1-5; LED mulai OFF. `mati` memadamkan LED tanpa
mengubah level tersimpan. `naik/turun` mempertahankan keadaan ON/OFF,
jadi nyalakan LED dahulu untuk mengamati perubahan kecerahan. Tiga prediksi
konsisten diperlukan untuk satu event. Event berikutnya menunggu jeda hening
dan cooldown agar satu ucapan tidak menaikkan level berkali-kali.
Level 1-5 memakai duty PWM nonlinier 4, 16, 48, 120, 255 dari 255, bukan
langkah linear. Persentase duty kira-kira 1,6%, 6,3%, 18,8%, 47,1%, 100%;
ini bukan persentase kecerahan yang dipersepsikan mata. Level 1 lebih redup
dan level 5 tetap penuh. Jika log level berubah tetapi cahaya tidak berubah,
periksa GPIO25 -> resistor seri 330 ohm -> anode LED -> cathode -> GND;
jangan menyuplai LED langsung dari pin 3V3/5V. Evaluasi dengan mata, bukan
auto-exposure kamera yang dapat menyamarkan perbedaan.
Gate ambient RMS/clipping dan konsensus adalah kebijakan operasi perangkat, bukan model
unknown/silence terlatih. Suara di luar empat kelas masih bisa salah dikenali.

## Audio dan Kesetaraan

ADC DMA ESP32 klasik bekerja pada 32 kHz. Sinyal dikurangi bias DC, dilewatkan
FIR anti-alias 63 tap, lalu diturunkan menjadi 16 kHz. Penyimpanan PCM16 tidak
menambah resolusi fisik ADC 12-bit. Filter acquisition ini berbeda dari filter
MP6; filter MP6 tetap diterapkan sesudahnya sesuai frontend training.

Window klasik sepanjang 1,2 detik mengikuti dataset MP9, diperbarui tiap 100 ms
sesuai stride MP14. Jalur referensi menjalankan filter MP6 dengan koefisien double,
DC removal, peak normalization, VAD/trimming, pre-emphasis, Hamming, FFT 400,
log-Mel, MFCC dan scaler/model INT8 beku. Perhitungan pitch dilewati saat live
hanya jika fitur tersebut tidak dipakai model; fitur terpilih tidak berubah.
Ini berbeda dari context CNN 96 frame; pipeline klasik mengikuti utterance MP9.
Jika fitur terpilih tidak memakai pitch, jalur live memakai float pada FPU,
representasi dua biquad/SOS dari transfer function MP6 yang sama, rolling energy
untuk VAD, serta statistik MFCC. Jika model juga memakai RMS, ZCR, centroid,
bandwidth atau rolloff, statistik non-pitch tersebut ikut dihitung dengan
definisi yang sama. Model yang memilih bandwidth tidak lagi jatuh ke jalur
referensi double hanya karena fiturnya bukan MFCC. Pitch tetap memakai jalur
referensi; jangan mengklaim jalur pitch sudah memenuhi deadline ESP32.
Bobot/scaler/fitur terpilih tetap
beku. Pemeriksaan host membandingkan fitur dan prediksi jalur live maupun
referensi, termasuk sinyal silence/noise/DC/nada. Toleransi fitur tetap 0,01;
keberhasilan pemeriksaan ini bukan jaminan deadline 100 ms di perangkat.

Capture berjalan di core 0; ekstraksi/inferensi di core 1. Buffer besar memakai
heap internal karena batas linker DRAM statis ESP32. Log free/minimum heap adalah
ukuran runtime; persentase RAM dari build hanya menghitung alokasi statis.

## Pengujian Perangkat MP16

### Pemakaian bebas tanpa urutan perintah

Firmware mengenali perintah secara terus-menerus, tidak memerlukan prompt atau
komputer untuk inferensi. Setelah diberi daya, diam selama kalibrasi sekitar
tujuh detik, lalu ucapkan nyala/mati/naik/turun dengan urutan bebas dan jeda
hening minimal dua detik antar-ucapan. LED eksternal GPIO25 mengikuti perintah;
naik/turun mengubah level tersimpan tanpa mengubah status ON/OFF.

Untuk melihat event dan menyimpan log tanpa batas waktu, jalankan dari root
proyek eksperimen yang bersangkutan:

```bash
python tools/collect_mp16_esp32.py --port /dev/ttyUSB0 --continuous
```

Tidak ada prompt ENTER atau label target pada mode ini. Ctrl+C menghentikan
monitor dan menyimpan log, tetapi tidak menghentikan firmware ESP32. Event
tanpa label tidak dipakai untuk mengklaim akurasi atau kelulusan demo empat
perintah. Jangan gabungkan --continuous dengan --demo. Firmware tidak perlu
di-upload ulang hanya untuk mengganti mode collector.
Monitor menampilkan SIAP, MENDETEKSI UCAPAN atau TUNGGU HENING beserta RMS dan
ambang saat ini. Ucapkan perintah setelah SIAP; beri jeda sampai kembali SIAP
sebelum ucapan berikutnya. Ini tetap sistem empat keyword terpisah, bukan
pengenal percakapan atau rangkaian kata tanpa jeda.

Tutup serial monitor sebelum menjalankan collector, kemudian:

```bash
.venv/bin/python -m pip install -r firmware/esp32/requirements-host.txt
.venv/bin/python tools/collect_mp16_esp32.py --port /dev/ttyUSB0 --demo --seconds 5
```

Collector meminta empat command dan menyimpan JSONL, events CSV, serta
acceptance metrics di `results/device/mp16_esp32_<timestamp>/`. Pemeriksaan
mencakup identitas paket, empat command dengan label yang benar, overflow/error
ADC, missed stride, laju sampling, p95 latency, minimum heap dan ukuran firmware.
Pastikan perubahan LED yang terlihat sesuai log; daya diukur terpisah dengan alat.
Pembaca serial berjalan di thread latar belakang, termasuk saat menunggu ENTER;
buffer serial tidak dibuang antar-kata. Jangan membuka serial monitor lain
bersamaan. Log bertipe `boot` juga dapat merupakan balasan identitas `?`, bukan
bukti board reset. Collector membatalkan demo jika belum ada window audio,
paket tidak cocok, atau board mengirim fatal.
Collector menunggu kalibrasi selesai dan gate kembali armed sebelum meminta
ucapan. Log window mencantumkan gate_ready, noise_rms, speech_on_rms,
speech_off_rms, speech_active, event_armed dan detection_reason. Baud firmware
MP16 sekarang 460800 agar tambahan diagnostik tidak menahan inferensi; collector
membaca baud dari manifest paket. Firmware mic_test tetap memakai 921600.

Host equivalence adalah pemeriksaan implementasi, bukan benchmark ESP32.
Keberhasilan kompilasi tidak menandai deployment fisik selesai. Manifest MP16
lama tetap menjadi hasil replay; lampirkan hasil collector dan pengamatan LED
sebagai bukti tambahan pengujian perangkat. Jangan mengubah model berdasarkan
hasil frozen test; catat kegagalan mikrofon sebagai error analysis.

Referensi: [MAX9814](https://www.analog.com/en/products/max9814.html),
[ADC DMA ESP-IDF 4.4](https://docs.espressif.com/projects/esp-idf/en/v4.4.7/esp32/api-reference/peripherals/adc.html),
[KISS FFT](https://github.com/mborgerding/kissfft/tree/131.1.0).
