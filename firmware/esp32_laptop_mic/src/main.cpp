#include <Arduino.h>
#include <esp_timer.h>
#include <algorithm>
#include <cstring>
#include <new>
#include "board_config.h"
#include "kws_frontend.h"
#include "mp15_model.h"
#include "noise_gate.h"
#include "pcm_protocol.h"

SET_LOOP_TASK_STACK_SIZE(16384);
static KwsFrontend *frontend;
static int16_t *ring_pcm, *snapshot;
static portMUX_TYPE mux = portMUX_INITIALIZER_UNLOCKED;
static size_t position = 0, filled = 0;
static uint32_t sample_count = 0, stream_epoch = 0, last_packet_ms = 0, transport_errors = 0;
static uint32_t crc_errors = 0, sequence_gaps = 0, received_packets = 0;
static constexpr unsigned USB_BAUD = 921600;
static constexpr unsigned PROTOCOL = 2;
static TaskHandle_t infer_task;
static bool led_on = false;
static int level = KWS_LED_INITIAL_LEVEL;

static void report_identity(const char *type = "identity") {
    Serial.printf("{\"type\":\"%s\",\"input\":\"laptop_usb_pcm16\",\"processing\":\"esp32\",\"protocol\":%u,\"baud\":%u,\"base_package\":\"%s\",\"model\":\"%s\",\"sample_rate_hz\":16000,\"led_gpio\":%d}\n", type, PROTOCOL, USB_BAUD, KWS_PACKAGE_ID, KWS_MODEL_NAME, KWS_LED_GPIO);
}

static void update_led() {
    unsigned duty = led_on ? KWS_LED_DUTY_BY_LEVEL[level] : 0;
    ledcWrite(KWS_LED_CHANNEL, KWS_LED_ACTIVE_LOW ? 255 - duty : duty);
}

static void ready_led(bool on) {
    digitalWrite(KWS_READY_LED_GPIO, on != bool(KWS_READY_LED_ACTIVE_LOW) ? HIGH : LOW);
}

static void reset_audio() {
    portENTER_CRITICAL(&mux);
    filled = position = 0;
    ++stream_epoch;
    portEXIT_CRITICAL(&mux);
    xTaskNotifyGive(infer_task);
}

static void receive_audio(void *) {
    PcmParser parser;
    uint32_t expected = 0;
    uint32_t observed_bad = 0;
    bool have_sequence = false, timed_out = false;
    while (true) {
        if (!Serial.available()) {
            if (!timed_out && millis() - last_packet_ms > 500) {
                if (have_sequence) reset_audio();
                parser.reset(); have_sequence = false; timed_out = true;
            }
            vTaskDelay(1); continue;
        }
        const bool complete = parser.push(Serial.read());
        if (parser.bad_frames != observed_bad) {
            const uint32_t delta = parser.bad_frames - observed_bad;
            observed_bad = parser.bad_frames; crc_errors += delta; transport_errors += delta;
            reset_audio(); have_sequence = false;
        }
        if (!complete) continue;
        if (parser.kind() == PcmParser::IDENTITY) { report_identity(); continue; }
        const uint32_t sequence = parser.sequence();
        const size_t count = parser.count();
        if (have_sequence && sequence != expected) { ++transport_errors; ++sequence_gaps; reset_audio(); }
        expected = sequence + 1; have_sequence = true; timed_out = false;
        portENTER_CRITICAL(&mux);
        for (size_t j = 0; j < count; ++j) {
            ring_pcm[position] = parser.sample(j);
            position = (position + 1) % KWS_WINDOW_SAMPLES;
        }
        filled = std::min(size_t(KWS_WINDOW_SAMPLES), filled + count);
        sample_count += count; last_packet_ms = millis(); ++received_packets;
        portEXIT_CRITICAL(&mux);
        xTaskNotifyGive(infer_task);
        Serial.printf("{\"type\":\"ack\",\"sequence\":%lu}\n", static_cast<unsigned long>(sequence));
    }
}

void setup() {
    Serial.setRxBufferSize(8192);
    Serial.begin(USB_BAUD);
    pinMode(KWS_READY_LED_GPIO, OUTPUT); ready_led(false);
    ledcSetup(KWS_LED_CHANNEL, 5000, 8);
    ledcAttachPin(KWS_LED_GPIO, KWS_LED_CHANNEL); update_led();
    frontend = new(std::nothrow) KwsFrontend;
    ring_pcm = static_cast<int16_t *>(malloc(KWS_WINDOW_SAMPLES * sizeof(int16_t)));
    snapshot = static_cast<int16_t *>(malloc(KWS_WINDOW_SAMPLES * sizeof(int16_t)));
    if (!frontend || !ring_pcm || !snapshot || !frontend->begin()) {
        Serial.println("{\"type\":\"fatal\",\"error\":\"allocation_or_frontend_failed\"}");
        while (true) delay(1000);
    }
    infer_task = xTaskGetCurrentTaskHandle();
    if (xTaskCreatePinnedToCore(receive_audio, "usb_pcm", 4096, nullptr, 3, nullptr, 0) != pdPASS) {
        Serial.println("{\"type\":\"fatal\",\"error\":\"receiver_failed\"}");
        while (true) delay(1000);
    }
    report_identity("boot");
}

void loop() {
    static NoiseGate gate;
    static uint32_t previous = 0, epoch_seen = 0, last_event = 0;
    static unsigned quiet = 0, consecutive = 0;
    static int last_class = -1;
    static bool armed = false;
    static uint32_t last_health = 0;
    ulTaskNotifyTake(pdTRUE, pdMS_TO_TICKS(100));
    uint32_t count, epoch, last;
    bool full;
    portENTER_CRITICAL(&mux);
    count = sample_count; epoch = stream_epoch; last = last_packet_ms;
    full = filled == KWS_WINDOW_SAMPLES;
    if (full && count - previous >= KWS_STRIDE_SAMPLES)
        for (size_t i = 0; i < KWS_WINDOW_SAMPLES; ++i)
            snapshot[i] = ring_pcm[(position + i) % KWS_WINDOW_SAMPLES];
    portEXIT_CRITICAL(&mux);
    if (millis() - last_health >= 1000) {
        last_health = millis();
        Serial.printf("{\"type\":\"health\",\"protocol\":2,\"received_packets\":%lu,\"buffer_samples\":%u,\"transport_errors\":%lu,\"crc_errors\":%lu,\"sequence_gaps\":%lu,\"stream_epoch\":%lu}\n", static_cast<unsigned long>(received_packets), static_cast<unsigned>(full ? KWS_WINDOW_SAMPLES : 0), static_cast<unsigned long>(transport_errors), static_cast<unsigned long>(crc_errors), static_cast<unsigned long>(sequence_gaps), static_cast<unsigned long>(epoch));
    }
    if (epoch != epoch_seen) {
        epoch_seen = epoch; gate.reset(); armed = false; quiet = consecutive = 0;
        last_class = -1; led_on = false; update_led(); ready_led(false);
        previous = count;
        return;
    }
    if (!full || millis() - last > 300) { ready_led(false); return; }
    if (count - previous < KWS_STRIDE_SAMPLES) return;
    previous = count;
    int64_t started = esp_timer_get_time();
    KwsResult result = {};
    bool valid = frontend->extract(snapshot, KWS_WINDOW_SAMPLES, result);
    if (valid) result.predicted = MP15_predict39(result.features, result.scores);
    float elapsed = (esp_timer_get_time() - started) / 1000.0f;
    portENTER_CRITICAL(&mux);
    bool uninterrupted = stream_epoch == epoch && millis() - last_packet_ms < 300;
    portEXIT_CRITICAL(&mux);
    bool clean = valid && result.clipping_ratio <= KWS_MAX_CLIPPING_RATIO && uninterrupted;
    gate.update(result.raw_rms, clean);
    bool voiced = clean && gate.ready() && gate.active() && result.predicted >= 0;
    bool event = false;
    if (!voiced) {
        consecutive = 0; last_class = -1;
        if (!gate.ready() || !clean) { quiet = 0; armed = false; }
        else if (++quiet >= KWS_QUIET_WINDOWS_TO_REARM) armed = true;
    } else {
        quiet = 0;
        consecutive = last_class == result.predicted ? consecutive + 1 : 1;
        last_class = result.predicted;
        if (armed && consecutive >= KWS_REQUIRED_CONSENSUS &&
            (!last_event || millis() - last_event >= KWS_EVENT_COOLDOWN_MS)) {
            const char *label = KWS_CLASSES[result.predicted];
            if (!strcmp(label,"nyala")) led_on = true;
            else if (!strcmp(label,"mati")) led_on = false;
            else if (!strcmp(label,"naik")) level = std::min(KWS_LED_LEVEL_MAX, level + 1);
            else if (!strcmp(label,"turun")) level = std::max(KWS_LED_LEVEL_MIN, level - 1);
            update_led(); event = true; armed = false; last_event = millis();
        }
    }
    bool ready = gate.ready() && armed && !gate.active() && clean &&
                 quiet >= KWS_QUIET_WINDOWS_TO_REARM &&
                 (!last_event || millis() - last_event >= KWS_EVENT_COOLDOWN_MS);
    ready_led(ready);
    Serial.printf("{\"type\":\"%s\",\"predicted_class\":\"%s\",\"processing\":\"esp32\",\"pipeline_latency_ms\":%.3f,\"audio_rms\":%.5f,\"gate_ready\":%s,\"ready_for_command\":%s,\"led_on\":%s,\"led_level\":%d,\"transport_errors\":%lu}\n", event ? "event" : "window", voiced ? KWS_CLASSES[result.predicted] : "unknown", elapsed, result.raw_rms, gate.ready() ? "true":"false", ready ? "true":"false", led_on ? "true":"false", level, static_cast<unsigned long>(transport_errors));
}
