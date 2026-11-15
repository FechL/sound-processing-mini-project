#pragma once
#include <algorithm>
#include <cmath>
#include "board_config.h"

// Ambient calibration is an operating policy, not classifier/model training.
class NoiseGate {
public:
    void reset() { count_ = 0; noise_ = 0; active_ = false; }
    bool ready() const { return count_ == KWS_NOISE_CALIBRATION_WINDOWS; }
    bool active() const { return active_; }
    float noise() const { return noise_; }
    float onset() const { return std::max(KWS_MIN_AUDIO_RMS, noise_ * KWS_SPEECH_ON_RATIO); }
    float release() const { return std::max(KWS_MIN_AUDIO_RMS, noise_ * KWS_SPEECH_OFF_RATIO); }
    void update(float rms, bool valid) {
        if (!valid || !std::isfinite(rms) || rms < 0) { active_ = false; return; }
        if (!ready()) {
            levels_[count_++] = rms;
            if (ready()) {
                std::sort(levels_, levels_ + KWS_NOISE_CALIBRATION_WINDOWS);
                const unsigned middle = KWS_NOISE_CALIBRATION_WINDOWS / 2;
                noise_ = KWS_NOISE_CALIBRATION_WINDOWS % 2 ? levels_[middle] :
                         (levels_[middle - 1] + levels_[middle]) * 0.5f;
            }
            return;
        }
        if (active_) {
            if (rms < release()) active_ = false;
        } else {
            // Decide onset before adapting, so a speech burst cannot raise its
            // own threshold. Track quiet observations in both directions;
            // a minimum-only estimator drifts down and can prevent release.
            if (rms >= onset()) active_ = true;
            else noise_ += KWS_NOISE_TRACK_ALPHA * (rms - noise_);
        }
    }
private:
    float levels_[KWS_NOISE_CALIBRATION_WINDOWS] = {};
    unsigned count_ = 0;
    float noise_ = 0;
    bool active_ = false;
};
