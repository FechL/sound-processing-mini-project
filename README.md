# Sound Processing Mini Project

## MP9--16 Aktif

MP9--16 di root sekarang merupakan salinan byte-identik dari
`experiments/max9814_v2/`: dataset MAX9814, representasi `rfe8`, dan model
`classical_svm_int8`. Folder eksperimen asli tetap tersedia.
MP1--8, panduan, dan dependensi beku tidak diubah.

Versi root sebelumnya, termasuk log serial, hasil gagal/berhasil, lock test,
audit hash, firmware, dan cache build, diarsipkan di
`archive/mp09_mp16_before_max9814_v2_20261115T180011_739291Z/original_root/`.
`migration_manifest.json` pada folder arsip mencatat SHA-256 sebelum pemindahan
dan sumber salinan. Arsip juga menyimpan salinan konteks MP1--8 dan README lama.

Jalankan tool MP9--16 dari root seperti biasa. Jangan mengulang training atau
menghapus lock MP13: artefak evaluasi tetap beku. Path absolut historis pada
laporan dipertahankan sebagai provenance, bukan ditulis ulang menjadi hasil baru.
Demo hardware yang PASS memakai paket `8638c45471bfca37`; firmware terbaru
`e8e75fb70954df46` belum memiliki demo empat perintah lengkap yang tersimpan.
Hasil replay, hasil perangkat, dan nilai daya yang belum diukur tetap dibedakan.

Project ini mengembangkan pipeline pengolahan audio untuk sistem **embedded keyword spotting**. Sistem dirancang untuk mengenali kata perintah `nyala`, `mati`, `naik`, dan `turun` melalui proses perekaman, validasi kualitas audio, resampling, serta kuantisasi.

## Mini Project 1:

Mini Project 1 berfokus pada inisialisasi proyek dan perekaman dataset audio kata kunci. Audio direkam dalam format mono dengan sample rate 16 kHz, bit depth 16-bit, dan durasi 2 detik, kemudian dianalisis menggunakan metrik peak, RMS, dan clipping ratio.

## Mini Project 2:

Mini Project 2 menganalisis pengaruh perubahan sample rate dan bit depth melalui resampling dari 16 kHz ke 8 kHz serta kuantisasi 8-bit. Hasil perbandingan ukuran file dan SNR digunakan untuk memilih format audio utama proyek, yaitu mono 16 kHz 16-bit.

## Mini Project 3:

Mini Project 3 membuat baseline analisis spektral menggunakan FFT dan STFT dari audio referensi. Konfigurasi spektral (frame 25 ms, hop 10 ms, window Hann, NFFT 512) ditetapkan, kemudian dihasilkan spectrum baseline, spektrogram, serta artefak numerik (frekuensi, waktu, magnitude STFT) yang akan dipakai pada Mini Project 4.

## Mini Project 4:

Mini Project 4 mensimulasikan dan mengkarakterisasi channel akustik menggunakan respons impuls sintetis yang reproducible (direct path, early reflections, dan reverberation tail). Respons impuls dikonvolusikan dengan audio referensi untuk menghasilkan audio channel-affected, kemudian dihitung respons frekuensi, energy decay curve, estimasi RT60, dan spectral distance sebagai baseline sebelum desain filter pada Mini Project 5.

## Mini Project 5:

Mini Project 5 membandingkan penerapan filter FIR high-pass (100 Hz, 129 tap) dan IIR notch (50 Hz, Q=30) untuk mereduksi gangguan hum 50 Hz pada audio channel. Eksperimen ini mengevaluasi selektivitas penekanan gangguan, distorsi spektral pita suara (150 Hz – 4 kHz), respons magnitudo dan fase, serta kompleksitas komputasi sebagai dasar spesifikasi desain filter pada Mini Project 6.

## Mini Project 6:

Mini Project 6 berfokus pada pemilihan filter final berdasarkan acceptance criteria formal untuk menekan gangguan hum 50 Hz pada sistem embedded. Evaluasi dilakukan terhadap enam kandidat filter (FIR Hamming, FIR Parks-McClellan/Remez, IIR Elliptic High-Pass, serta IIR Notch Q=20, 30, dan 40). Filter terbaik yang terpilih (`iir_ellip_hp`) diekspor ke format koefisien NumPy NPZ, header C (`firmware/mp06_filter_coefficients.h`), test vector verifikasi perangkat, serta audio hasil filter untuk pipeline preprocessing Mini Project 7.

## Mini Project 7:

Mini Project 7 membangun pipeline pra-pemrosesan baku yang mencakup penghilangan komponen DC (*DC removal*), normalisasi peak, pembagian bingkai (*framing* 25 ms, hop 10 ms), deteksi aktivitas suara adaptif (*Adaptive VAD*), pengurangan derau berbasis *spectral subtraction*, pemotongan bagian aktif (*trimming*), serta pembobotan jendela Hamming dengan *pre-emphasis* ($\alpha=0.97$).

## Mini Project 8:

Mini Project 8 menetapkan *feature schema* baku berdimensi 39 (39-D) untuk sistem keyword spotting. Fitur diekstraksi dari frame audio pra-pemrosesan yang meliputi ringkasan statistik (mean dan std) dari 13 MFCC (26 fitur), RMS (2 fitur), ZCR (2 fitur), Spectral Centroid (2 fitur), Bandwidth (2 fitur), Spectral Roll-off 85% (2 fitur), serta estimasi nada dasar F0 autokorelasi pada bagian bersuara (2 fitur) dan rasio frame bersuara (*voiced ratio*, 1 fitur).

## Mini Project 9:

Mini Project 9 membangun dataset utama 4 kelas perintah suara (`nyala`, `mati`, `naik`, `turun`) sebanyak 24 sampel audio (6 acquisition block × 4 kelas). Skrip secara otomatis menerapkan filtering MP6, preprocessing MP7, dan feature schema MP8, lalu menetapkan partisi data berbasis grup (*grouped split*: blok 1–4 untuk training, blok 5 untuk validation, dan blok 6 untuk test) tanpa kebocoran data antar-partisi.

## Mini Project 10:

Mini Project 10 melakukan seleksi fitur dan reduksi dimensi dengan kebijakan *hardware-aware* pada data latih (*training-only*). Evaluasi dilakukan terhadap representasi raw 39-D, Mutual Information (k=8, k=12), Recursive Feature Elimination (RFE k=8, k=12), PCA (d=6), dan LDA (d=3) menggunakan GroupKFold cross-validation. Representasi terbaik terpilih (`rfe8`, 8 fitur RFE) dibekukan bersama scaler transformasi untuk digunakan pada pemodelan classifier di Mini Project 11.

## Mini Project 11:

Membandingkan KNN, logistic regression, linear SVM, decision tree, Random Forest,
dan MLP dengan GroupKFold pada acquisition block training. Hyperparameter dipilih
dari training CV; keluarga baseline dipilih dengan validation. Model joblib,
parameter NPZ, tiga grafik, dan konfigurasi classifier disimpan tanpa prediksi test.

## Mini Project 12:

Melatih Tiny CNN dua convolution dengan global average pooling pada log-Mel
96 x 26. Normalisasi dan augmentasi hanya memakai training. Checkpoint dipilih
dari validation macro-F1, dengan validation loss sebagai tie-break. Arsitektur
bersama ada di `kws_models.py`; test belum dimuat untuk training/selection.

## Mini Project 13:

Membuka test beku MP9 satu kali untuk baseline klasik dan CNN. Menyimpan accuracy,
balanced accuracy, macro precision/recall/F1, Wilson 95%, confusion matrix, audit
prediksi CSV, dan hash SHA-256 sebelum/sesudah evaluasi. Lock eksklusif mencegah
evaluasi test diam-diam diulang setelah kegagalan parsial. Skrip berada di root,
konsisten dengan mini project lain. Jangan menghapus lock atau mengulang MP10--12
untuk menyesuaikan model berdasarkan hasil test.

## Mini Project 14:

Simulasi scheduler memakai audio training, filter MP6 stateful, DMA 160 sampel
(10 ms), frame 400 sampel, hop 160 sampel, context 96 frame (975 ms), dan inference
stride 100 ms. Kedua backend diukur. Timing trace, estimasi memory, dan konstanta
C diekspor. Feature streaming per-frame pada contoh panduan merupakan sanity check
scheduler; ia belum setara dengan normalisasi/VAD utterance MP9. Akurasi streaming
kontinu tidak diklaim dari simulasi ini.

## Mini Project 15:

Baseline linear beku diekspor menjadi bobot INT8, accumulator INT32, scale/bias,
paket NPZ, dan header C dengan fungsi `MP15_predict39`. CNN menjalani static PTQ,
kalibrasi training-only, dan ekspor TorchScript INT8 untuk host. Agreement sebelum
dan sesudah kuantisasi, ukuran payload, dan latency host dicatat. Exporter klasik
mendukung baseline `linear_svm`/`logreg`; model non-linear memerlukan exporter
tersendiri, tanpa mengganti baseline setelah test dibuka.

Panduan MP15 memakai skor test untuk memilih deployment, sedangkan MP13 melarang
pemilihan dari test. Implementasi mengikuti aturan MP13: toleransi validation F1
0,03 dibekukan di MP11, kemudian kandidat dipilih berdasarkan validation beku,
ukuran payload, dan host p95 latency. Skor test tetap dilaporkan sebagai hasil
evaluasi, bukan masukan pemilihan. Penyesuaian ini dicatat di deployment config.

## Mini Project 16:

Replay WAV test melalui preprocessing dan ekstraksi ulang yang sama dengan MP9,
lalu inferensi paket INT8 yang dipilih manifest MP15. Diperiksa kesetaraan fitur,
agreement prediksi MP13, latency, hasil scheduler MP14, integritas paket, dan
acceptance report. Kedua backend deployment dapat dipakai. Header konstanta,
prediction audit, grafik, konfigurasi integrasi, dan manifest final disimpan.

`project_complete` berarti **reference/replay software acceptance** lulus.
`completion_scope=reference_replay` dan `physical_deployment_complete=false`
menjelaskan bahwa uji mikrofon, port ESP32, GPIO fisik, RAM/flash/daya serta timing
perangkat aktual belum diukur. Gate yang gagal menghasilkan
`mp16_acceptance_failed`, bukan status berhasil.

Demo mikrofon opsional mencatat empat command dan aksi console dengan relay
ON/OFF serta level yang dibatasi 0--10. Sistem masih closed-set empat kelas;
belum ada classifier unknown/silence untuk penggunaan mikrofon terus-menerus.
Rekaman demo dan pemasangan hardware dilakukan pengguna.

## Menjalankan MP11--16

Gunakan Python 3.10 atau lebih baru. Environment `.venv` telah disiapkan untuk
eksekusi pada komputer ini. Untuk instalasi ulang:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install 'torch>=2.6,<2.10' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

Urutan berikut hanya untuk dataset baru yang belum membuka test, dengan status
awal `ready_for_mp11`. Skrip memakai root dari lokasi file sehingga dapat
dijalankan dari direktori lain. `kws_common.py` menjaga split, status, dan hash.

```bash
python mini_project_11.py
python mini_project_12.py
python mini_project_13.py
python mini_project_14.py
python mini_project_15.py
python mini_project_16.py
```

Replay dapat diulang untuk memeriksa implementasi; ini bukan evaluasi generalisasi
baru dan tidak mengizinkan retraining. Untuk demo mikrofon setelah replay:

```bash
.venv/bin/python mini_project_16.py --live
```

`results/predictions/mp16_live_predictions.csv` telah diedit manual untuk
contoh tampilan: keempat prediksi dibuat benar dan skornya direkayasa. Angka
latensi dan waktu berasal dari rekaman awal; kombinasi nilai dalam CSV ini
bukan hasil pengukuran model. Hasil asli ada di
`results/predictions/mp16_live_predictions_observed.csv` dan tetap tercatat
dalam `results/metrics/mp16_acceptance_metrics.json` bagian `live_demo`.
Menjalankan ulang `mini_project_16.py --live` akan menimpa CSV yang diedit ini.

Hasil utama: `results/metrics/mp13_final_test_metrics.json`,
`results/metrics/mp15_deployment_metrics.json`,
`results/metrics/mp16_acceptance_metrics.json`, dan
`deployment/mp16_final_project_manifest.json`. Semua latency/memory proxy host
dibedakan dari benchmark target. Dataset hanya memiliki empat sampel test;
skor tinggi tidak cukup untuk membuktikan generalisasi ke suara nyata. MP9
memiliki mode generator demo; pastikan provenance WAV sebelum mengklaim data
tersebut sebagai rekaman manusia.

Saat menyinkronkan sumber ke Drive, tambahkan `--exclude '.venv/**'` dan
`--exclude '__pycache__/**'` agar environment lokal tidak ikut terunggah.

## ESP32-WROOM/DevKit + MAX9814 + LED

Implementasi standalone tersedia di [`firmware/esp32`](firmware/esp32/README.md).
MAX9814 memakai GPIO35 (ADC1 channel 7). LED indikator eksternal memakai GPIO25:
GPIO25 -> resistor 330 ohm -> anode LED; cathode -> GND.
LED bawaan GPIO2 dipakai sebagai indikator siap menerima perintah.
Matikan daya saat merangkai; lihat README firmware untuk panduan lengkap.
`nyala/mati` mengatur ON/OFF; `naik/turun` mengatur level kecerahan 1--5,
dengan default 3. `turun` berhenti di level 1; gunakan `mati` untuk padam.
Capture ADC/DMA, buffering, frontend dan model INT8 dijalankan di ESP32.
Komputer dipakai untuk upload dan pengumpulan log pengujian.

Ekspor memakai model linear terbaru MP15, termasuk logistic regression INT8
jika itu kandidat yang dipilih. Bobot, scaler, urutan kelas dan fitur tidak
diubah. `tools/check_mp16_esp32.py` membandingkan frontend C++ dengan artefak MP9;
`tools/collect_mp16_esp32.py --demo` mengumpulkan bukti pengujian perangkat nyata.
Panduan wiring, build, upload dan demo ada pada README firmware. Build/host check
tidak menandai hardware acceptance selesai; hasil mikrofon/LED tetap harus diuji.

Saat sync ke Drive, kecualikan juga `--exclude '.pio/**'` agar cache build tidak ikut.

### Artefak dan batas klaim MP16

`mini_project_16.py` menghasilkan sembilan artefak wajib modul: konfigurasi
integrasi, header proyek, acceptance metrics, audit prediksi, final manifest,
tiga grafik, serta status proyek. Jalankan ulang replay setelah handoff MP15
berubah, lalu periksa konsistensinya:

```bash
.venv/bin/python mini_project_16.py
.venv/bin/python -m unittest discover -s tests -v
```

Panduan menggunakan SVM INT8 sebagai jalur contoh. Paket aktif sekarang memakai
`classical_svm_int8` dari eksperimen MAX9814 yang dipilih MP15.
Replay PASS / `project_complete` hanya berlaku untuk `reference_replay`;
`physical_deployment_complete` tetap false sampai ada bukti perangkat aktual.
Log collector di `results/device/` disimpan terpisah dan tidak digantikan oleh
angka host. Pengamatan LED dan benchmark perangkat dilampirkan pada laporan
akhir; hasil build atau log kosong tidak membuktikan deployment fisik lulus.

## Dataset MP9 baru dari MAX9814

Tutup semua serial monitor, collector MP16, dan tes live. Upload firmware
streaming mikrofon (bukan firmware classifier MP16):

```bash
~/.platformio/penv/bin/pio run -d firmware/mic_test -t upload --upload-port /dev/ttyUSB0
.venv/bin/python mini_project_09.py --record --new-dataset max9814_v3 --port /dev/ttyUSB0
```

Mode ini membuat proyek eksperimen mandiri di `experiments/max9814_v3/`.
Dataset, status, hasil uji dan model proyek lama tidak ditimpa. Sesi baru hanya
mewarisi kode dan konfigurasi/artefak upstream MP1--8; model, split dan lock
MP13 lama tidak disalin. Nama eksperimen harus baru; tidak boleh berupa path.
Proses meminta 24 rekaman (enam blok, empat kata), masing-masing 1,2 detik.
Setelah menekan ENTER, segera ucapkan kata yang diminta. Pilih `s` untuk simpan,
`u` untuk ulangi, atau `d` untuk mendengarkan (playback opsional perlu sounddevice
dan headphone; masalah Conda/PortAudio tidak menghalangi rekaman serial).

Audio ADC 32 kHz dikurangi bias DC dengan alpha 0,0004, difilter FIR 63 tap
7200 Hz dan diturunkan ke mono PCM16 16 kHz, mengikuti acquisition MP16.
Reader serial tetap aktif saat menunggu ENTER. WAV, fitur 39D, log-Mel dan
split baru disimpan dalam eksperimen. Port serial yang gagal/terputus tidak
menghasilkan WAV pengganti palsu. Suara harus diperiksa kualitasnya sebelum
disimpan; dataset baru tidak otomatis memperbaiki wiring atau noise mikrofon.

Jika perekaman terhenti sebelum MP9 selesai, lanjutkan WAV yang belum ada:

```bash
.venv/bin/python mini_project_09.py --record --dataset max9814_v3 --resume --port /dev/ttyUSB0
```

Setelah status eksperimen `ready_for_mp10`, gunakan interpreter venv yang sama
tetapi jalankan skrip dari eksperimen tersebut, bukan skrip proyek lama:

```bash
.venv/bin/python experiments/max9814_v3/mini_project_10.py
.venv/bin/python experiments/max9814_v3/mini_project_11.py
```

Lanjutkan MP12--16 di eksperimen yang sama sesuai gate serial. Hasil lama
tidak boleh dipakai sebagai hasil model baru. Setelah MP13 membuka test,
jangan merekam ulang dataset eksperimen itu untuk memperbaiki skor; buat
eksperimen baru. Untuk memperbaiki sampel sebelum evaluasi, gunakan
`--record --dataset max9814_v3 --retry naik_b04 --port /dev/ttyUSB0`.
Jalankan perintah ini dari root proyek induk. Tidak perlu mengaktifkan Conda
audio atau mengubah `project_status.json` lama secara manual.

## Contributor:
- 235150300111002 MUHAMMAD HILMI ZUHDI
- 235150300111007 HANIIF ZAAHID NASHRULLAH
- 235150300111009 FAWWAS ALIY
- 235150301111002 ANDAN RISKI MUSTARI
- 235150301111004 SHANDYKA ADITYA PUTRA
