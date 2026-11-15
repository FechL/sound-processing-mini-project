#include <Arduino.h>
#include <driver/adc.h>
#include <esp_timer.h>
#include <esp_heap_caps.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <new>
#include "board_config.h"
#include "adc_decimator.h"
#include "kws_frontend.h"
#include "mp15_model.h"
#include "noise_gate.h"

SET_LOOP_TASK_STACK_SIZE(16384);
static_assert(KWS_LED_LEVEL_MIN > 0 && KWS_LED_LEVEL_MIN <= KWS_LED_INITIAL_LEVEL &&
              KWS_LED_INITIAL_LEVEL <= KWS_LED_LEVEL_MAX, "Invalid LED brightness range");
static_assert(sizeof(KWS_LED_DUTY_BY_LEVEL) / sizeof(KWS_LED_DUTY_BY_LEVEL[0]) ==
              KWS_LED_LEVEL_MAX + 1, "LED PWM table does not match level range");
static_assert(KWS_LED_DUTY_BY_LEVEL[KWS_LED_LEVEL_MAX] == 255, "Top LED level must be full duty");
static_assert(KWS_READY_LED_GPIO != KWS_LED_GPIO && KWS_READY_LED_GPIO != KWS_ADC_GPIO,
              "Ready indicator must use a separate GPIO");

static KwsFrontend *frontend = nullptr;
static int16_t *ring_pcm = nullptr, *snapshot = nullptr;
static portMUX_TYPE ring_mux = portMUX_INITIALIZER_UNLOCKED;
static size_t write_pos = 0, filled = 0;
static uint32_t sample_sequence = 0, overflow_count = 0, adc_errors = 0, stream_epoch = 0;
static uint32_t adc_samples = 0, adc_rail_samples = 0;
static TaskHandle_t inference_task = nullptr;
static int led_level = KWS_LED_INITIAL_LEVEL;
static bool led_on = false;
static unsigned led_pwm = 0;
static bool ready_led_on = false;
static NoiseGate noise_gate;

static void report_identity() {
    Serial.printf("{\"type\":\"boot\",\"requires_noise_calibration\":true,\"model\":\"%s\",\"package\":\"%s\",\"adc_gpio\":%d,\"led_gpio\":%d,\"ready_led_gpio\":%d,\"sample_rate_hz\":%d,\"adc_sample_rate_hz\":%d,\"window_samples\":%d,\"firmware_flash_bytes\":%u}\n",
                  KWS_MODEL_NAME, KWS_PACKAGE_ID, KWS_ADC_GPIO, KWS_LED_GPIO, KWS_READY_LED_GPIO, KWS_SAMPLE_RATE,
                  KWS_ADC_SAMPLE_RATE, KWS_WINDOW_SAMPLES, ESP.getSketchSize());
}

static void ready_led_update(bool ready) {
    ready_led_on = ready;
    digitalWrite(KWS_READY_LED_GPIO, (ready != static_cast<bool>(KWS_READY_LED_ACTIVE_LOW)) ? HIGH : LOW);
}

static void led_update() {
    led_pwm = led_on ? KWS_LED_DUTY_BY_LEVEL[led_level] : 0;
    ledcWrite(KWS_LED_CHANNEL, KWS_LED_ACTIVE_LOW ? 255 - led_pwm : led_pwm);
}

static void apply_command(int predicted) {
    const char *label = KWS_CLASSES[predicted];
    if (!strcmp(label, "nyala")) led_on = true;
    else if (!strcmp(label, "mati")) led_on = false;
    else if (!strcmp(label, "naik")) led_level = std::min(KWS_LED_LEVEL_MAX, led_level + 1);
    else if (!strcmp(label, "turun")) led_level = std::max(KWS_LED_LEVEL_MIN, led_level - 1);
    led_update();
}

static void reset_ring() {
    portENTER_CRITICAL(&ring_mux);
    filled = 0; write_pos = 0; ++stream_epoch;
    portEXIT_CRITICAL(&ring_mux);
}

static bool start_adc() {
    adc_digi_init_config_t init = {};
    init.max_store_buf_size = KWS_ADC_POOL_BYTES;
    init.conv_num_each_intr = KWS_DMA_BYTES;
    init.adc1_chan_mask = 1u << KWS_ADC_CHANNEL;
    if (adc_digi_initialize(&init) != ESP_OK) return false;
    adc_digi_pattern_config_t pattern = {};
    pattern.atten = ADC_ATTEN_DB_12;
    pattern.channel = KWS_ADC_CHANNEL;
    // DMA patterns use a zero-based unit index, unlike adc_unit_t.
    pattern.unit = 0;
    pattern.bit_width = SOC_ADC_DIGI_MAX_BITWIDTH;
    adc_digi_configuration_t config = {};
    // Required by the classic ESP32 ADC DMA driver (ESP-IDF 4.4).
    config.conv_limit_en = true;
    config.conv_limit_num = 250;
    config.sample_freq_hz = KWS_ADC_SAMPLE_RATE;
    config.conv_mode = ADC_CONV_SINGLE_UNIT_1;
    config.format = ADC_DIGI_OUTPUT_FORMAT_TYPE1;
    config.pattern_num = 1;
    config.adc_pattern = &pattern;
    if (adc_digi_controller_configure(&config) != ESP_OK || adc_digi_start() != ESP_OK) {
        adc_digi_deinitialize(); return false;
    }
    return true;
}

static void capture_task(void *) {
    alignas(4) uint8_t dma[KWS_DMA_BYTES];
    float delay_line[KWS_DECIMATOR_TAPS] = {};
    int16_t block[KWS_HOP_SAMPLES];
    int delay_pos = 0, phase = 0, block_fill = 0;
    uint32_t warmup = 0;
    float dc = 0;
    bool first_sample = true;
    while (true) {
        uint32_t bytes = 0;
        const esp_err_t err = adc_digi_read_bytes(dma, sizeof(dma), &bytes, 1000);
        if (err != ESP_OK) {
            portENTER_CRITICAL(&ring_mux);
            if (err == ESP_ERR_INVALID_STATE) ++overflow_count;
            else ++adc_errors;
            portEXIT_CRITICAL(&ring_mux);
            reset_ring();
            memset(delay_line, 0, sizeof(delay_line));
            block_fill = 0; phase = 0; warmup = 0; first_sample = true;
            vTaskDelay(pdMS_TO_TICKS(1));
            continue;
        }
        for (uint32_t offset = 0; offset + sizeof(adc_digi_output_data_t) <= bytes;
             offset += sizeof(adc_digi_output_data_t)) {
            const auto *sample = reinterpret_cast<const adc_digi_output_data_t *>(dma + offset);
            if (sample->type1.channel != KWS_ADC_CHANNEL) continue;
            const int raw = sample->type1.data;
            portENTER_CRITICAL(&ring_mux);
            ++adc_samples;
            if (raw <= 8 || raw >= 4087) ++adc_rail_samples;
            portEXIT_CRITICAL(&ring_mux);
            if (first_sample) { dc = raw; first_sample = false; }
            dc += 0.0004f * (raw - dc);
            delay_line[delay_pos] = (raw - dc) / 2048.0f;
            float filtered = 0;
            // Only compute the FIR on samples that survive the 2:1 decimator.
            phase ^= 1;
            if (!phase) {
                for (int j = 0; j < KWS_DECIMATOR_TAPS; ++j) {
                    int at = delay_pos - j;
                    if (at < 0) at += KWS_DECIMATOR_TAPS;
                    filtered += KWS_DECIMATOR[j] * delay_line[at];
                }
            }
            delay_pos = (delay_pos + 1) % KWS_DECIMATOR_TAPS;
            if (warmup < KWS_WARMUP_ADC_SAMPLES) { ++warmup; continue; }
            if (phase) continue;
            filtered = std::max(-1.0f, std::min(1.0f, filtered));
            block[block_fill++] = static_cast<int16_t>(lrintf(filtered * 32767));
            if (block_fill == KWS_HOP_SAMPLES) {
                portENTER_CRITICAL(&ring_mux);
                for (int j = 0; j < KWS_HOP_SAMPLES; ++j) {
                    ring_pcm[write_pos] = block[j];
                    write_pos = (write_pos + 1) % KWS_WINDOW_SAMPLES;
                }
                filled = std::min(static_cast<size_t>(KWS_WINDOW_SAMPLES), filled + KWS_HOP_SAMPLES);
                sample_sequence += KWS_HOP_SAMPLES;
                portEXIT_CRITICAL(&ring_mux);
                block_fill = 0;
                xTaskNotifyGive(inference_task);
            }
        }
    }
}

void setup() {
    Serial.begin(KWS_SERIAL_BAUD);
    pinMode(KWS_READY_LED_GPIO, OUTPUT);
    ready_led_update(false);
    ledcSetup(KWS_LED_CHANNEL, 5000, 8);
    ledcAttachPin(KWS_LED_GPIO, KWS_LED_CHANNEL);
    led_update();
    // Large workspaces use internal heap; ESP32's statically linked DRAM is limited.
    frontend = new(std::nothrow) KwsFrontend;
    ring_pcm = static_cast<int16_t *>(heap_caps_malloc(KWS_WINDOW_SAMPLES * sizeof(int16_t), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
    snapshot = static_cast<int16_t *>(heap_caps_malloc(KWS_WINDOW_SAMPLES * sizeof(int16_t), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
    if (!frontend || !ring_pcm || !snapshot || !frontend->begin() || !start_adc()) {
        Serial.println("{\"type\":\"fatal\",\"error\":\"frontend_or_adc_init_failed\"}");
        while (true) delay(1000);
    }
    report_identity();
    inference_task = xTaskGetCurrentTaskHandle();
    if (xTaskCreatePinnedToCore(capture_task, "audio_adc", 4096, nullptr, 3, nullptr, 0) != pdPASS) {
        Serial.println("{\"type\":\"fatal\",\"error\":\"capture_task_failed\"}");
        while (true) delay(1000);
    }
}

void loop() {
    while (Serial.available()) if (Serial.read() == '?') report_identity();
    static uint32_t previous_sequence = 0, previous_epoch = 0, last_event_ms = 0;
    static uint32_t missed_strides = 0, consecutive = 0, quiet = 0;
    static int previous_class = -1;
    static bool armed = false;
    static uint32_t rate_start_us = micros(), rate_start_samples = 0;
    static float measured_adc_rate = 0;
    static uint32_t last_health_ms = 0;
    static uint32_t observed_sequence = 0, last_audio_ms = 0;
    ulTaskNotifyTake(pdTRUE, pdMS_TO_TICKS(100));
    uint32_t sequence, epoch, overflows, errors, raw_count, rail_count;
    bool ready, buffer_full;
    portENTER_CRITICAL(&ring_mux);
    sequence = sample_sequence; epoch = stream_epoch;
    overflows = overflow_count; errors = adc_errors;
    raw_count = adc_samples; rail_count = adc_rail_samples;
    buffer_full = filled == KWS_WINDOW_SAMPLES;
    ready = buffer_full && sequence - previous_sequence >= KWS_STRIDE_SAMPLES;
    if (ready) for (size_t j = 0; j < KWS_WINDOW_SAMPLES; ++j)
        snapshot[j] = ring_pcm[(write_pos + j) % KWS_WINDOW_SAMPLES];
    portEXIT_CRITICAL(&ring_mux);
    if (sequence != observed_sequence) {
        observed_sequence = sequence;
        last_audio_ms = millis();
    }
    if (epoch != previous_epoch) {
        previous_epoch = epoch; previous_class = -1; consecutive = 0; quiet = 0; armed = false;
        noise_gate.reset();
        ready_led_update(false);
        led_on = false; led_update();
    }
    const uint32_t now_us = micros();
    if (now_us - rate_start_us >= 1000000) {
        measured_adc_rate = static_cast<float>(raw_count - rate_start_samples) * 1000000.0f / (now_us - rate_start_us);
        rate_start_us = now_us; rate_start_samples = raw_count;
    }
    if (!ready) {
        if (!buffer_full || millis() - last_audio_ms >= KWS_AUDIO_STALE_MS)
            ready_led_update(false);
        if (millis() - last_health_ms < 1000) return;
        last_health_ms = millis();
        Serial.printf("{\"type\":\"health\",\"timestamp_ms\":%lu,\"overflow_count\":%lu,\"adc_errors\":%lu,\"adc_rate_hz\":%.1f,\"free_heap_bytes\":%u}\n",
                      static_cast<unsigned long>(millis()), static_cast<unsigned long>(overflows),
                      static_cast<unsigned long>(errors), measured_adc_rate, ESP.getFreeHeap());
        return;
    }
    if (previous_sequence && sequence - previous_sequence > KWS_STRIDE_SAMPLES)
        missed_strides += (sequence - previous_sequence) / KWS_STRIDE_SAMPLES - 1;
    previous_sequence = sequence;
    const int64_t started = esp_timer_get_time();
    KwsResult result = {};
    bool valid = frontend->extract(snapshot, KWS_WINDOW_SAMPLES, result);
    const int64_t feature_done = esp_timer_get_time();
    if (valid) result.predicted = MP15_predict39(result.features, result.scores);
    const float pipeline_ms = (esp_timer_get_time() - started) / 1000.0f;
    const bool audio_valid = valid && result.clipping_ratio <= KWS_MAX_CLIPPING_RATIO;
    noise_gate.update(result.raw_rms, audio_valid);
    const bool voiced = audio_valid && result.predicted >= 0 && noise_gate.ready() && noise_gate.active();
    portENTER_CRITICAL(&ring_mux);
    const bool uninterrupted = stream_epoch == epoch;
    portEXIT_CRITICAL(&ring_mux);
    bool event = false;
    if (!voiced || !uninterrupted) {
        consecutive = 0; previous_class = -1;
        if (!noise_gate.ready()) { quiet = 0; armed = false; }
        else if (++quiet >= KWS_QUIET_WINDOWS_TO_REARM) armed = true;
    } else {
        quiet = 0;
        consecutive = previous_class == result.predicted ? consecutive + 1 : 1;
        previous_class = result.predicted;
        if (armed && consecutive >= KWS_REQUIRED_CONSENSUS &&
            (!last_event_ms || millis() - last_event_ms >= KWS_EVENT_COOLDOWN_MS)) {
            apply_command(result.predicted); event = true; armed = false; last_event_ms = millis();
        }
    }
    const bool command_ready = noise_gate.ready() && armed && quiet >= KWS_QUIET_WINDOWS_TO_REARM &&
                               !noise_gate.active() && audio_valid && uninterrupted &&
                               millis() - last_audio_ms < KWS_AUDIO_STALE_MS &&
                               (!last_event_ms || millis() - last_event_ms >= KWS_EVENT_COOLDOWN_MS);
    ready_led_update(command_ready);
    const char *label = voiced && uninterrupted ? KWS_CLASSES[result.predicted] : "unknown";
    const char *reason = !uninterrupted ? "stream_reset" : !valid ? "invalid_features" :
                         !noise_gate.ready() ? "calibrating_noise" : !audio_valid ? "clipping" :
                         !noise_gate.active() ? "below_speech_gate" : !armed && !event ? "wait_for_quiet" :
                         event ? "event" : "wait_for_consensus_or_cooldown";
    float score = result.predicted >= 0 ? result.scores[result.predicted] : 0;
    Serial.printf("{\"type\":\"%s\",\"timestamp_ms\":%lu,\"predicted_class\":\"%s\",\"score\":%.6f,\"feature_ms\":%.3f,\"pipeline_latency_ms\":%.3f,\"audio_rms\":%.5f,\"clipping_ratio\":%.5f,\"overflow_count\":%lu,\"adc_errors\":%lu,\"missed_strides\":%lu,\"adc_rate_hz\":%.1f,\"adc_rail_samples\":%lu,\"free_heap_bytes\":%u,\"minimum_free_heap_bytes\":%u,\"led_on\":%s,\"led_level\":%d,\"gate_ready\":%s,\"noise_rms\":%.5f,\"speech_on_rms\":%.5f,\"speech_off_rms\":%.5f,\"speech_active\":%s,\"event_armed\":%s,\"ready_for_command\":%s,\"detection_reason\":\"%s\"}\n",
                  event ? "event" : "window", static_cast<unsigned long>(millis()), label, score,
                  (feature_done - started) / 1000.0f, pipeline_ms, result.raw_rms, result.clipping_ratio,
                  static_cast<unsigned long>(overflows), static_cast<unsigned long>(errors),
                  static_cast<unsigned long>(missed_strides), measured_adc_rate, static_cast<unsigned long>(rail_count),
                  ESP.getFreeHeap(), ESP.getMinFreeHeap(), led_on ? "true" : "false", led_level,
                  noise_gate.ready() ? "true" : "false", noise_gate.noise(), noise_gate.onset(), noise_gate.release(),
                  noise_gate.active() ? "true" : "false", armed ? "true" : "false",
                  ready_led_on ? "true" : "false", reason);
}
