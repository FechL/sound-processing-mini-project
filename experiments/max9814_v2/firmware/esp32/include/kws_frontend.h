#pragma once
#include <stddef.h>
#include <stdint.h>
#include "generated_frontend.h"
#include "kiss_fftr.h"

struct KwsResult {
    float features[39];
    float scores[KWS_CLASS_COUNT];
    int predicted;
    unsigned frame_count;
    float raw_rms;
    float clipping_ratio;
};

// Utterance frontend follows MP9/MP16, including the 400-point FFT.
class KwsFrontend {
public:
    KwsFrontend();
    ~KwsFrontend();
    bool begin();
    bool extract(const int16_t *pcm, size_t count, KwsResult &result, bool all_features = false);
private:
    bool extract_live(const int16_t *pcm, size_t count, KwsResult &result);
    kiss_fftr_cfg fft_;
    float wave_[KWS_WINDOW_SAMPLES];
    float frame_[KWS_FRAME_SAMPLES];
    kiss_fft_cpx spectrum_[KWS_FRAME_SAMPLES / 2 + 1];
    float window_[KWS_FRAME_SAMPLES];
    float mel_[KWS_MEL_BANDS][KWS_FRAME_SAMPLES / 2 + 1];
    int mel_start_[KWS_MEL_BANDS], mel_end_[KWS_MEL_BANDS];
    double dct_[KWS_MFCC_COUNT][KWS_MEL_BANDS];
    float live_dct_[KWS_MFCC_COUNT][KWS_MEL_BANDS];
};
