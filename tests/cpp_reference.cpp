#include "rotohash.hpp"

#include <array>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

static uint8_t TestByte(uint64_t test_id, size_t index) {
    uint64_t value = static_cast<uint64_t>(index) +
        test_id * UINT64_C(0x9E3779B97F4A7C15);
    value ^= value >> 30;
    value *= UINT64_C(0xBF58476D1CE4E5B9);
    value ^= value >> 27;
    value *= UINT64_C(0x94D049BB133111EB);
    value ^= value >> 31;
    return static_cast<uint8_t>(value);
}

static void RunCase(size_t length, uint64_t seed, size_t offset, uint64_t test_id) {
    std::vector<uint8_t> storage(offset + length + 64, 0xA5);
    uint8_t* data = storage.data() + offset;
    for (size_t index = 0; index < length; ++index)
        data[index] = TestByte(test_id, index);

    auto hash = RotoHash::Hash(data, length, seed);
    std::array<uint8_t, 16> bytes;
    std::memcpy(bytes.data(), &hash, bytes.size());
    for (auto byte : bytes)
        std::printf("%02x", static_cast<unsigned>(byte));
    std::putchar('\n');
}

int main() {
    constexpr std::array<uint64_t, 4> seeds = {
        UINT64_C(0),
        UINT64_C(1),
        UINT64_C(0x0123456789ABCDEF),
        UINT64_MAX,
    };

    uint64_t test_id = 0;
    for (size_t length = 0; length <= 1024; ++length) {
        for (size_t seed_index = 0; seed_index < seeds.size(); ++seed_index) {
            size_t offset = (length * 13 + seed_index * 7) % 64;
            RunCase(length, seeds[seed_index], offset, test_id++);
        }
    }

    constexpr std::array<size_t, 7> large_lengths = {
        4095, 4096, 4097, 65535, 65536, 262144, 262145,
    };
    constexpr std::array<size_t, 5> offsets = {0, 1, 15, 31, 63};
    for (auto length : large_lengths)
        for (auto offset : offsets)
            for (auto seed : seeds)
                RunCase(length, seed, offset, test_id++);

    return RotoHash::VerifyImplementation() ? 0 : 2;
}

