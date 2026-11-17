#pragma once
#include <stddef.h>
#include <stdint.h>
#include <string.h>

// Wire format: magic[8], sequence:u32, sample_count:u16, kind:u16,
// PCM16 payload, CRC32(header + payload). All integers are little-endian.
class PcmParser {
public:
    static constexpr size_t MAX_SAMPLES = 800;
    static constexpr uint16_t AUDIO = 0, IDENTITY = 1;
    bool push(uint8_t byte) {
        if (ready_) { discard(frame_size_); ready_ = false; }
        if (used_ == sizeof(buffer_)) { discard(1); ++bad_frames; }
        buffer_[used_++] = byte;
        while (used_ >= 8) {
            if (memcmp(buffer_, "LPCM2!!!", 8)) { discard(1); continue; }
            if (used_ < 16) return false;
            const unsigned count = u16(buffer_ + 12), type = u16(buffer_ + 14);
            if (!((type == IDENTITY && count == 0) ||
                  (type == AUDIO && count && count <= MAX_SAMPLES && count % 160 == 0))) {
                ++bad_frames; discard(1); continue;
            }
            frame_size_ = 20 + count * 2;
            if (used_ < frame_size_) return false;
            if (crc32(buffer_, frame_size_ - 4) != u32(buffer_ + frame_size_ - 4)) {
                ++bad_frames; discard(1); continue;
            }
            ready_ = true;
            return true;
        }
        return false;
    }
    void reset() { used_ = 0; ready_ = false; }
    uint32_t sequence() const { return u32(buffer_ + 8); }
    uint16_t count() const { return u16(buffer_ + 12); }
    uint16_t kind() const { return u16(buffer_ + 14); }
    int16_t sample(size_t i) const { return static_cast<int16_t>(u16(buffer_ + 16 + i * 2)); }
    uint32_t bad_frames = 0;
private:
    static uint16_t u16(const uint8_t *p) { return uint16_t(p[0]) | (uint16_t(p[1]) << 8); }
    static uint32_t u32(const uint8_t *p) {
        return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
    }
    static uint32_t crc32(const uint8_t *p, size_t size) {
        uint32_t crc = 0xffffffffu;
        for (size_t i = 0; i < size; ++i) {
            crc ^= p[i];
            for (int bit = 0; bit < 8; ++bit) crc = (crc >> 1) ^ (0xedb88320u & (0u - (crc & 1u)));
        }
        return ~crc;
    }
    void discard(size_t count) {
        used_ -= count;
        memmove(buffer_, buffer_ + count, used_);
    }
    uint8_t buffer_[20 + MAX_SAMPLES * 2] = {};
    size_t used_ = 0, frame_size_ = 0;
    bool ready_ = false;
};
