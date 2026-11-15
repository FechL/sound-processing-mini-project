#pragma once

// External indicator LED: GPIO25 -> 330 ohm -> anode, cathode -> GND.
#define KWS_LED_GPIO 25
#define KWS_READY_LED_GPIO 2
#define KWS_READY_LED_ACTIVE_LOW 0
#define KWS_AUDIO_STALE_MS 300
#define KWS_LED_ACTIVE_LOW 0
#define KWS_LED_CHANNEL 0
#define KWS_LED_LEVEL_MIN 1
#define KWS_LED_LEVEL_MAX 5
#define KWS_LED_INITIAL_LEVEL 3
// Nonlinear PWM steps keep the low levels visibly dimmer than full brightness.
static constexpr unsigned KWS_LED_DUTY_BY_LEVEL[] = {0, 4, 16, 48, 120, 255};
// MAX9814 OUT -> GPIO35 (ADC1 channel 7), VDD -> 3V3, GND -> GND.
#define KWS_ADC_GPIO 35
#define KWS_ADC_CHANNEL 7
#define KWS_ADC_SAMPLE_RATE 32000
#define KWS_DMA_BYTES 640
#define KWS_ADC_POOL_BYTES 8192
#define KWS_WARMUP_ADC_SAMPLES 32000
// Hardware event gates, independent of frozen classifier parameters.
#define KWS_MIN_AUDIO_RMS 0.008f
#define KWS_NOISE_CALIBRATION_WINDOWS 40
#define KWS_SPEECH_ON_RATIO 3.0f
#define KWS_SPEECH_OFF_RATIO 1.5f
#define KWS_NOISE_TRACK_ALPHA 0.1f
#define KWS_MAX_CLIPPING_RATIO 0.01f
#define KWS_REQUIRED_CONSENSUS 3
#define KWS_EVENT_COOLDOWN_MS 1500
#define KWS_QUIET_WINDOWS_TO_REARM 3
