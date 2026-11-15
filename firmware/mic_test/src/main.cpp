#include <Arduino.h>
#include <driver/adc.h>
#include <soc/soc_caps.h>

static constexpr unsigned CHANNEL = 7;  // MAX9814 OUT -> GPIO35.
static constexpr unsigned BLOCK_BYTES = 640;
static uint32_t sequence = 0;

static void packet(const uint8_t *data, uint16_t bytes, uint16_t flags) {
    // Little-endian wire header: magic[8], sequence:u32, bytes:u16, flags:u16.
    uint8_t header[16] = {'M', 'I', 'C', '9', '8', '1', '4', '!'};
    const uint32_t id = sequence++;
    memcpy(header + 8, &id, 4);
    memcpy(header + 12, &bytes, 2);
    memcpy(header + 14, &flags, 2);
    Serial.write(header, sizeof(header));
    if (bytes) Serial.write(data, bytes);
}

void setup() {
    Serial.begin(921600);
    adc_digi_init_config_t init = {};
    init.max_store_buf_size = 16384;
    init.conv_num_each_intr = BLOCK_BYTES;
    init.adc1_chan_mask = 1u << CHANNEL;
    esp_err_t error = adc_digi_initialize(&init);
    if (error == ESP_OK) {
        adc_digi_pattern_config_t pattern = {};
        pattern.atten = ADC_ATTEN_DB_12;
        pattern.channel = CHANNEL;
        pattern.unit = 0;
        pattern.bit_width = SOC_ADC_DIGI_MAX_BITWIDTH;
        adc_digi_configuration_t config = {};
        config.conv_limit_en = true;
        config.conv_limit_num = 250;
        config.sample_freq_hz = 32000;
        config.conv_mode = ADC_CONV_SINGLE_UNIT_1;
        config.format = ADC_DIGI_OUTPUT_FORMAT_TYPE1;
        config.pattern_num = 1;
        config.adc_pattern = &pattern;
        error = adc_digi_controller_configure(&config);
        if (error == ESP_OK) error = adc_digi_start();
    }
    if (error != ESP_OK) {
        while (true) { packet(nullptr, 0, 4); delay(1000); }
    }
}

void loop() {
    alignas(4) uint8_t data[BLOCK_BYTES];
    uint32_t bytes = 0;
    const esp_err_t error = adc_digi_read_bytes(data, sizeof(data), &bytes, 1000);
    if (error == ESP_OK) packet(data, bytes, 0);
    else {
        packet(nullptr, 0, error == ESP_ERR_INVALID_STATE ? 1 : 2);
        delay(1);
    }
}
