#include "kws_frontend.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>

static const double PI = 3.14159265358979323846;
static double hz_to_mel(double f) { return 2595.0 * std::log10(1.0 + f / 700.0); }
static double mel_to_hz(double m) { return 700.0 * (std::pow(10.0, m / 2595.0) - 1.0); }

KwsFrontend::KwsFrontend() : fft_(nullptr) {}
KwsFrontend::~KwsFrontend() { if (fft_) free(fft_); }

bool KwsFrontend::begin() {
    if (fft_) return true;
    fft_ = kiss_fftr_alloc(KWS_FRAME_SAMPLES, 0, nullptr, nullptr);
    if (!fft_) return false;
    for (int j = 0; j < KWS_FRAME_SAMPLES; ++j)
        window_[j] = 0.54 - 0.46 * std::cos(2.0 * PI * j / (KWS_FRAME_SAMPLES - 1));
    int bins[KWS_MEL_BANDS + 2];
    const double low = hz_to_mel(KWS_MEL_MIN_HZ), high = hz_to_mel(KWS_MEL_MAX_HZ);
    for (int j = 0; j < KWS_MEL_BANDS + 2; ++j)
        bins[j] = static_cast<int>(std::floor((KWS_FRAME_SAMPLES + 1) *
            mel_to_hz(low + (high - low) * j / (KWS_MEL_BANDS + 1)) / KWS_SAMPLE_RATE));
    std::memset(mel_, 0, sizeof(mel_));
    for (int m = 0; m < KWS_MEL_BANDS; ++m) {
        const int left = bins[m], center = std::max(bins[m + 1], left + 1);
        const int right = std::max(bins[m + 2], center + 1);
        mel_start_[m] = left;
        mel_end_[m] = std::min(right, KWS_FRAME_SAMPLES / 2 + 1);
        for (int k = left; k < center && k <= KWS_FRAME_SAMPLES / 2; ++k)
            mel_[m][k] = static_cast<float>(k - left) / (center - left);
        for (int k = center; k < right && k <= KWS_FRAME_SAMPLES / 2; ++k)
            mel_[m][k] = static_cast<float>(right - k) / (right - center);
    }
    for (int j = 0; j < KWS_MFCC_COUNT; ++j)
        for (int m = 0; m < KWS_MEL_BANDS; ++m)
            dct_[j][m] = std::cos(PI * j * (m + 0.5) / KWS_MEL_BANDS) *
                std::sqrt((j == 0 ? 1.0 : 2.0) / KWS_MEL_BANDS);
    for (int j = 0; j < KWS_MFCC_COUNT; ++j)
        for (int m = 0; m < KWS_MEL_BANDS; ++m)
            live_dct_[j][m] = static_cast<float>(dct_[j][m]);
    return true;
}

bool KwsFrontend::extract(const int16_t *pcm, size_t count, KwsResult &out, bool all_features) {
    if (!fft_ || !pcm || count < KWS_FRAME_SAMPLES || count > KWS_WINDOW_SAMPLES) return false;
    if (!all_features && !KWS_NEED_PITCH) return extract_live(pcm, count, out);
    std::memset(&out, 0, sizeof(out));
    out.predicted = -1;
    double state[KWS_FILTER_ORDER] = {}, sum = 0, raw_sum = 0, raw_sq = 0;
    unsigned clipped = 0;
    for (size_t i = 0; i < count; ++i) {
        const double x = pcm[i] / 32767.0;
        raw_sum += x; raw_sq += x * x;
        if (std::abs(x) >= 0.99) ++clipped;
        const double y = KWS_FILTER_B[0] * x + state[0];
        for (int j = 0; j < KWS_FILTER_ORDER - 1; ++j)
            state[j] = state[j + 1] + KWS_FILTER_B[j + 1] * x - KWS_FILTER_A[j + 1] * y;
        state[KWS_FILTER_ORDER - 1] = KWS_FILTER_B[KWS_FILTER_ORDER] * x - KWS_FILTER_A[KWS_FILTER_ORDER] * y;
        wave_[i] = y;
        sum += wave_[i];
    }
    out.raw_rms = std::sqrt(std::max(0.0, raw_sq / count - std::pow(raw_sum / count, 2)));
    out.clipping_ratio = static_cast<float>(clipped) / count;
    const double mean = sum / count;
    double peak = 0;
    for (size_t i = 0; i < count; ++i) peak = std::max(peak, std::abs(wave_[i] - mean));
    const double gain = KWS_TARGET_PEAK / (peak + 1e-12);
    for (size_t i = 0; i < count; ++i) wave_[i] = (wave_[i] - mean) * gain;
    const int frames = 1 + (count - KWS_FRAME_SAMPLES) / KWS_HOP_SAMPLES;
    double rms[KWS_MAX_FRAMES], max_rms = 0;
    for (int f = 0; f < frames; ++f) {
        double sq = 0;
        for (int j = 0; j < KWS_FRAME_SAMPLES; ++j) {
            const double x = wave_[f * KWS_HOP_SAMPLES + j]; sq += x * x;
        }
        rms[f] = std::sqrt(sq / KWS_FRAME_SAMPLES + 1e-12);
        max_rms = std::max(max_rms, rms[f]);
    }
    int first = frames, last = -1;
    for (int f = 0; f < frames; ++f) if (rms[f] >= KWS_VAD_FACTOR * max_rms) {
        first = std::min(first, f); last = f;
    }
    if (last < 0) { first = 0; last = frames - 1; }
    first = std::max(0, first - KWS_VAD_MARGIN);
    last = std::min(frames - 1, last + KWS_VAD_MARGIN);
    double sums[19] = {}, squares[19] = {}, pitch_sum = 0, pitch_sq = 0;
    unsigned voiced_count = 0;
    const int n_frames = last - first + 1;
    for (int f = first; f <= last; ++f) {
        const int start = f * KWS_HOP_SAMPLES;
        double sq = 0, frame_mean = 0;
        unsigned crossings = 0;
        for (int j = 0; j < KWS_FRAME_SAMPLES; ++j) {
            const double x = wave_[start + j] - (j ? KWS_PREEMPH * wave_[start + j - 1] : 0);
            frame_[j] = x * window_[j];
            sq += static_cast<double>(frame_[j]) * frame_[j];
            frame_mean += frame_[j];
            if (j && std::signbit(frame_[j]) != std::signbit(frame_[j - 1])) ++crossings;
        }
        kiss_fftr(fft_, frame_, spectrum_);
        double power[KWS_FRAME_SAMPLES / 2 + 1], total = 0, weighted = 0;
        for (int k = 0; k <= KWS_FRAME_SAMPLES / 2; ++k) {
            const double magnitude = std::hypot(spectrum_[k].r, spectrum_[k].i) + 1e-12;
            power[k] = magnitude * magnitude;
            total += power[k]; weighted += power[k] * k * KWS_SAMPLE_RATE / KWS_FRAME_SAMPLES;
        }
        const double centroid = weighted / (total + 1e-12);
        double variance = 0, cumulative = 0, rolloff = 0;
        bool reached = false;
        for (int k = 0; k <= KWS_FRAME_SAMPLES / 2; ++k) {
            const double hz = static_cast<double>(k) * KWS_SAMPLE_RATE / KWS_FRAME_SAMPLES;
            variance += power[k] * (hz - centroid) * (hz - centroid);
            cumulative += power[k];
            if (!reached && cumulative >= KWS_ROLLOFF * total) { rolloff = hz; reached = true; }
        }
        double logmel[KWS_MEL_BANDS], values[19] = {};
        for (int m = 0; m < KWS_MEL_BANDS; ++m) {
            double energy = 0;
            for (int k = mel_start_[m]; k < mel_end_[m]; ++k) energy += power[k] * mel_[m][k];
            logmel[m] = std::log(std::max(energy, 1e-12));
        }
        for (int j = 0; j < KWS_MFCC_COUNT; ++j)
            for (int m = 0; m < KWS_MEL_BANDS; ++m) values[j] += dct_[j][m] * logmel[m];
        values[13] = std::sqrt(sq / KWS_FRAME_SAMPLES + 1e-12);
        values[14] = static_cast<double>(crossings) / (KWS_FRAME_SAMPLES - 1);
        values[15] = centroid; values[16] = std::sqrt(variance / (total + 1e-12)); values[17] = rolloff;
        const double fm = frame_mean / KWS_FRAME_SAMPLES;
        double ac0 = 0;
        for (int j = 0; j < KWS_FRAME_SAMPLES; ++j) ac0 += (frame_[j] - fm) * (frame_[j] - fm);
        double best_ac = -1; int best_lag = 0;
        if ((all_features || KWS_NEED_PITCH) && ac0 > 1e-12) {
            for (int lag = KWS_PITCH_LAG_MIN; lag <= KWS_PITCH_LAG_MAX; ++lag) {
                double ac = 0;
                for (int j = 0; j < KWS_FRAME_SAMPLES - lag; ++j)
                    ac += (frame_[j] - fm) * (frame_[j + lag] - fm);
                if (ac > best_ac) { best_ac = ac; best_lag = lag; }
            }
            if (best_lag && best_ac / (ac0 + 1e-12) >= KWS_PITCH_CONFIDENCE) {
                const double pitch = static_cast<double>(KWS_SAMPLE_RATE) / best_lag;
                pitch_sum += pitch; pitch_sq += pitch * pitch; ++voiced_count;
            }
        }
        for (int j = 0; j < 18; ++j) { sums[j] += values[j]; squares[j] += values[j] * values[j]; }
    }
    for (int j = 0; j < 18; ++j) {
        const double m = sums[j] / n_frames;
        out.features[2 * j] = m;
        out.features[2 * j + 1] = std::sqrt(std::max(0.0, squares[j] / n_frames - m * m));
    }
    if (voiced_count) {
        const double p = pitch_sum / voiced_count;
        out.features[36] = p;
        out.features[37] = std::sqrt(std::max(0.0, pitch_sq / voiced_count - p * p));
    }
    out.features[38] = static_cast<float>(voiced_count) / n_frames;
    out.frame_count = n_frames;
    for (float v : out.features) if (!std::isfinite(v)) return false;
    return true;
}

// Same frozen transfer function and non-pitch features, evaluated in float for the
// ESP32 FPU. The full double path above remains available for equivalence checks.
bool KwsFrontend::extract_live(const int16_t *pcm, size_t count, KwsResult &out) {
    std::memset(&out, 0, sizeof(out));
    out.predicted = -1;
    float state[KWS_FILTER_SECTIONS][2] = {};
    float sum = 0, raw_sum = 0, raw_sq = 0;
    unsigned clipped = 0;
    for (size_t i = 0; i < count; ++i) {
        float x = static_cast<float>(pcm[i]) * (1.0f / 32767.0f);
        raw_sum += x; raw_sq += x * x;
        if (std::abs(x) >= 0.99f) ++clipped;
        for (int s = 0; s < KWS_FILTER_SECTIONS; ++s) {
            const float *c = KWS_FILTER_SOS + 6 * s;
            const float y = c[0] * x + state[s][0];
            state[s][0] = c[1] * x - c[4] * y + state[s][1];
            state[s][1] = c[2] * x - c[5] * y;
            x = y;
        }
        wave_[i] = x; sum += x;
    }
    const float raw_mean = raw_sum / count;
    out.raw_rms = std::sqrt(std::max(0.0f, raw_sq / count - raw_mean * raw_mean));
    out.clipping_ratio = static_cast<float>(clipped) / count;
    const float mean = sum / count;
    float peak = 0;
    for (size_t i = 0; i < count; ++i) peak = std::max(peak, std::abs(wave_[i] - mean));
    const float gain = static_cast<float>(KWS_TARGET_PEAK) / (peak + 1e-12f);
    for (size_t i = 0; i < count; ++i) wave_[i] = (wave_[i] - mean) * gain;
    const int frames = 1 + (count - KWS_FRAME_SAMPLES) / KWS_HOP_SAMPLES;
    // Rolling energy avoids re-summing the overlap for each VAD frame.
    float energy[KWS_MAX_FRAMES], sq = 0, maximum = 0;
    for (int j = 0; j < KWS_FRAME_SAMPLES; ++j) sq += wave_[j] * wave_[j];
    for (int f = 0; f < frames; ++f) {
        if (f) {
            const int previous = (f - 1) * KWS_HOP_SAMPLES;
            for (int j = 0; j < KWS_HOP_SAMPLES; ++j) {
                const float removed = wave_[previous + j];
                const float added = wave_[previous + KWS_FRAME_SAMPLES + j];
                sq += added * added - removed * removed;
            }
        }
        energy[f] = std::max(sq, 0.0f) / KWS_FRAME_SAMPLES + 1e-12f;
        maximum = std::max(maximum, energy[f]);
    }
    const float threshold = static_cast<float>(KWS_VAD_FACTOR * KWS_VAD_FACTOR) * maximum;
    int first = frames, last = -1;
    for (int f = 0; f < frames; ++f) if (energy[f] >= threshold) {
        first = std::min(first, f); last = f;
    }
    if (last < 0) { first = 0; last = frames - 1; }
    first = std::max(0, first - KWS_VAD_MARGIN);
    last = std::min(frames - 1, last + KWS_VAD_MARGIN);
    constexpr int STATS = KWS_MFCC_ONLY ? KWS_MFCC_COUNT : 18;
    float means[STATS] = {}, m2[STATS] = {};
    int n = 0;
    for (int f = first; f <= last; ++f) {
        const int start = f * KWS_HOP_SAMPLES;
        for (int j = 0; j < KWS_FRAME_SAMPLES; ++j)
            frame_[j] = (wave_[start + j] - (j ? static_cast<float>(KWS_PREEMPH) *
                         wave_[start + j - 1] : 0.0f)) * window_[j];
        kiss_fftr(fft_, frame_, spectrum_);
        float power[KWS_FRAME_SAMPLES / 2 + 1], logmel[KWS_MEL_BANDS];
        float values[STATS] = {};
        for (int k = 0; k <= KWS_FRAME_SAMPLES / 2; ++k) {
            const float re = spectrum_[k].r, im = spectrum_[k].i;
            const float magnitude = std::sqrt(re * re + im * im) + 1e-12f;
            power[k] = magnitude * magnitude;
        }
        if (!KWS_MFCC_ONLY) {
            float frame_sq = 0, total = 0, weighted = 0;
            unsigned crossings = 0;
            for (int j = 0; j < KWS_FRAME_SAMPLES; ++j) {
                frame_sq += frame_[j] * frame_[j];
                if (j && std::signbit(frame_[j]) != std::signbit(frame_[j - 1])) ++crossings;
            }
            for (int k = 0; k <= KWS_FRAME_SAMPLES / 2; ++k) {
                const float hz = static_cast<float>(k * KWS_SAMPLE_RATE) / KWS_FRAME_SAMPLES;
                total += power[k];
                weighted += power[k] * hz;
            }
            const float centroid = weighted / (total + 1e-12f);
            float variance = 0, cumulative = 0, rolloff = 0;
            bool reached = false;
            for (int k = 0; k <= KWS_FRAME_SAMPLES / 2; ++k) {
                const float hz = static_cast<float>(k * KWS_SAMPLE_RATE) / KWS_FRAME_SAMPLES;
                const float delta = hz - centroid;
                variance += power[k] * delta * delta;
                cumulative += power[k];
                if (!reached && cumulative >= static_cast<float>(KWS_ROLLOFF) * total) {
                    rolloff = hz; reached = true;
                }
            }
            values[13] = std::sqrt(frame_sq / KWS_FRAME_SAMPLES + 1e-12f);
            values[14] = static_cast<float>(crossings) / (KWS_FRAME_SAMPLES - 1);
            values[15] = centroid;
            values[16] = std::sqrt(variance / (total + 1e-12f));
            values[17] = rolloff;
        }
        for (int m = 0; m < KWS_MEL_BANDS; ++m) {
            float e = 0;
            for (int k = mel_start_[m]; k < mel_end_[m]; ++k) e += power[k] * mel_[m][k];
            logmel[m] = std::log(std::max(e, 1e-12f));
        }
        ++n;
        for (int j = 0; j < KWS_MFCC_COUNT; ++j) {
            for (int m = 0; m < KWS_MEL_BANDS; ++m) values[j] += live_dct_[j][m] * logmel[m];
        }
        for (int j = 0; j < STATS; ++j) {
            const float value = values[j];
            // Welford variance avoids cancellation in low-variance MFCCs.
            const float delta = value - means[j];
            means[j] += delta / n;
            m2[j] += delta * (value - means[j]);
        }
    }
    for (int j = 0; j < STATS; ++j) {
        out.features[2 * j] = means[j];
        out.features[2 * j + 1] = std::sqrt(std::max(0.0f, m2[j] / n));
    }
    out.frame_count = n;
    for (float v : out.features) if (!std::isfinite(v)) return false;
    return true;
}
