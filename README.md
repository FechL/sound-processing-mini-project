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

## Contributor:
- 235150300111002 MUHAMMAD HILMI ZUHDI
- 235150300111007 HANIIF ZAAHID NASHRULLAH
- 235150300111009 FAWWAS ALIY
- 235150301111002 ANDAN RISKI MUSTARI
- 235150301111004 SHANDYKA ADITYA PUTRA