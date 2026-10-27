# Sound Processing Mini Project

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

Mini Project 10 melakukan seleksi fitur dan reduksi dimensi dengan kebijakan *hardware-aware* pada data latih (*training-only*). Evaluasi dilakukan terhadap representasi raw 39-D, Mutual Information (k=8, k=12), Recursive Feature Elimination (RFE k=8, k=12), PCA (d=6), dan LDA (d=3) menggunakan GroupKFold cross-validation. Representasi terbaik terpilih (`mi8`, 8 fitur Mutual Information) dibekukan bersama scaler transformasi untuk digunakan pada pemodelan classifier di Mini Project 11.

## Contributor:
- 235150300111002 MUHAMMAD HILMI ZUHDI
- 235150300111007 HANIIF ZAAHID NASHRULLAH
- 235150300111009 FAWWAS ALIY
- 235150301111002 ANDAN RISKI MUSTARI
- 235150301111004 SHANDYKA ADITYA PUTRA