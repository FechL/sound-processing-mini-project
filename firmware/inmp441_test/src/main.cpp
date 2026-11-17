#include <Arduino.h>
#include <driver/gpio.h>
#include <driver/i2s.h>

static constexpr i2s_port_t PORT = I2S_NUM_0;
static constexpr int BCLK_GPIO = 26;
static constexpr int WS_GPIO = 27;
static constexpr int DATA_GPIO = 33;
static constexpr unsigned SAMPLE_RATE = 16000;
static constexpr unsigned BLOCK_SAMPLES = 160;
// L/R -> GND selects the left slot; use 1 only for L/R -> 3V3.
static constexpr unsigned SELECTED_SLOT = 0;
static QueueHandle_t events = nullptr;
static uint32_t sequence = 0;

static void packet(const int32_t *data, uint16_t bytes, uint16_t flags) {
    uint8_t header[16] = {'I', 'N', 'M', 'P', '4', '4', '1', '!'};
    const uint32_t id = sequence++;
    memcpy(header + 8, &id, sizeof(id));
    memcpy(header + 12, &bytes, sizeof(bytes));
    memcpy(header + 14, &flags, sizeof(flags));
    Serial.write(header, sizeof(header));
    if (bytes) Serial.write(reinterpret_cast<const uint8_t *>(data), bytes);
}

void setup() {
    Serial.begin(921600);
    i2s_config_t config = {};
    config.mode = static_cast<i2s_mode_t>(I2S_MODE_MASTER | I2S_MODE_RX);
    config.sample_rate = SAMPLE_RATE;
    config.bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT;
    config.bits_per_chan = I2S_BITS_PER_CHAN_32BIT;
    // Two 32-bit slots keep BCLK at 64 * WS, as required by INMP441.
    config.channel_format = I2S_CHANNEL_FMT_RIGHT_LEFT;
    config.communication_format = I2S_COMM_FORMAT_STAND_I2S;
    config.intr_alloc_flags = ESP_INTR_FLAG_LEVEL1;
    config.dma_buf_count = 8;
    config.dma_buf_len = BLOCK_SAMPLES;
    config.use_apll = false;
    esp_err_t error = i2s_driver_install(PORT, &config, 16, &events);
    if (error == ESP_OK) {
        i2s_pin_config_t pins = {};
        pins.mck_io_num = I2S_PIN_NO_CHANGE;
        pins.bck_io_num = BCLK_GPIO;
        pins.ws_io_num = WS_GPIO;
        pins.data_out_num = I2S_PIN_NO_CHANGE;
        pins.data_in_num = DATA_GPIO;
        error = i2s_set_pin(PORT, &pins);
        gpio_pulldown_en(static_cast<gpio_num_t>(DATA_GPIO));
    }
    if (error != ESP_OK) {
        while (true) { packet(nullptr, 0, 4); delay(1000); }
    }
    // Let the microphone settle while draining DMA instead of creating a backlog.
    int32_t warmup[BLOCK_SAMPLES * 2];
    for (unsigned i = 0; i < 30; ++i) {
        size_t received = 0;
        i2s_read(PORT, warmup, sizeof(warmup), &received, pdMS_TO_TICKS(100));
    }
    i2s_event_t event;
    while (xQueueReceive(events, &event, 0) == pdTRUE) {}
}

void loop() {
    int32_t stereo[BLOCK_SAMPLES * 2];
    int32_t mono[BLOCK_SAMPLES];
    size_t bytes = 0;
    const esp_err_t error = i2s_read(PORT, stereo, sizeof(stereo), &bytes, pdMS_TO_TICKS(1000));
    uint16_t flags = 0;
    i2s_event_t event;
    while (xQueueReceive(events, &event, 0) == pdTRUE) {
        if (event.type == I2S_EVENT_RX_Q_OVF) flags |= 1;
        if (event.type == I2S_EVENT_DMA_ERROR) flags |= 2;
    }
    if (error != ESP_OK || bytes == 0 || bytes % (2 * sizeof(int32_t))) {
        packet(nullptr, 0, flags | 2);
        delay(1);
        return;
    }
    const size_t count = bytes / (2 * sizeof(int32_t));
    for (size_t i = 0; i < count; ++i) {
        // Preserve signed, left-aligned 24-bit audio; the last eight bits are high-Z.
        mono[i] = static_cast<int32_t>(static_cast<uint32_t>(stereo[2 * i + SELECTED_SLOT]) & 0xffffff00u);
    }
    packet(mono, count * sizeof(int32_t), flags);
}

