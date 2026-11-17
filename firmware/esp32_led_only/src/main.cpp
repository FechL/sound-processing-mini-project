#include <Arduino.h>
#include <algorithm>
#include <cstring>
#include "board_config.h"

static bool on = false;
static int level = KWS_LED_INITIAL_LEVEL;
static char line[80];
static unsigned used = 0;
static bool discard_line = false, have_sequence = false;
static uint32_t last_seen = 0, last_sequence = 0;

static void update_led() {
    unsigned duty = on ? KWS_LED_DUTY_BY_LEVEL[level] : 0;
    ledcWrite(KWS_LED_CHANNEL, KWS_LED_ACTIVE_LOW ? 255 - duty : duty);
}

static void ready(bool value) {
    digitalWrite(KWS_READY_LED_GPIO, value != bool(KWS_READY_LED_ACTIVE_LOW) ? HIGH : LOW);
}

static void identity() {
    Serial.printf("{\"type\":\"identity\",\"mode\":\"led_only\",\"protocol\":1,\"led_gpio\":%d,\"ready_led_gpio\":%d}\n", KWS_LED_GPIO, KWS_READY_LED_GPIO);
}

static void handle_line() {
    if (!strcmp(line, "HELLO")) {
        have_sequence = false; last_seen = millis(); identity(); return;
    }
    if (!strcmp(line, "PING")) { last_seen = millis(); return; }
    if (!strcmp(line, "READY 1")) { last_seen = millis(); ready(true); return; }
    if (!strcmp(line, "READY 0")) { last_seen = millis(); ready(false); return; }
    unsigned long sequence;
    char command[12], extra;
    if (sscanf(line, "CMD %lu %11s %c", &sequence, command, &extra) != 2 || sequence > UINT32_MAX) return;
    if (strcmp(command,"nyala") && strcmp(command,"mati") && strcmp(command,"naik") && strcmp(command,"turun")) return;
    last_seen = millis();
    if (!have_sequence || sequence != last_sequence) {
        if (!strcmp(command,"nyala")) on = true;
        else if (!strcmp(command,"mati")) on = false;
        else if (!strcmp(command,"naik")) level = std::min(KWS_LED_LEVEL_MAX, level + 1);
        else level = std::max(KWS_LED_LEVEL_MIN, level - 1);
        last_sequence = sequence; have_sequence = true; update_led();
    }
    Serial.printf("{\"type\":\"ack\",\"sequence\":%lu,\"command\":\"%s\",\"led_on\":%s,\"led_level\":%d}\n", sequence, command, on ? "true":"false", level);
}

void setup() {
    Serial.begin(115200);
    pinMode(KWS_READY_LED_GPIO, OUTPUT); ready(false);
    ledcSetup(KWS_LED_CHANNEL, 5000, 8);
    ledcAttachPin(KWS_LED_GPIO, KWS_LED_CHANNEL); update_led();
    last_seen = millis(); identity();
}

void loop() {
    while (Serial.available()) {
        char c = Serial.read();
        if (c == '\r') continue;
        if (c == '\n') {
            if (!discard_line && used) { line[used] = 0; handle_line(); }
            used = 0; discard_line = false;
        } else if (!discard_line) {
            if (used + 1 < sizeof(line)) line[used++] = c;
            else { used = 0; discard_line = true; }
        }
    }
    if (millis() - last_seen > 3000) { on = false; update_led(); ready(false); }
    delay(1);
}
