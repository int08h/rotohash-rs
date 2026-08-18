#include "../tests/rotohash.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

constexpr std::array<size_t, 9> Sizes = {
    256,
    1024,
    4096,
    8192,
    16 * 1024,
    64 * 1024,
    256 * 1024,
    1024 * 1024,
    10 * 1024 * 1024,
};
constexpr auto CalibrationTime = std::chrono::milliseconds(100);
constexpr size_t Samples = 7;

volatile uint64_t Sink = 0;

struct BatchResult {
    std::chrono::steady_clock::duration elapsed;
    uint64_t accumulator;
};

uint8_t TestByte(size_t index) {
    uint64_t value = static_cast<uint64_t>(index) + UINT64_C(0x9E3779B97F4A7C15);
    value ^= value >> 30;
    value *= UINT64_C(0xBF58476D1CE4E5B9);
    value ^= value >> 27;
    value *= UINT64_C(0x94D049BB133111EB);
    value ^= value >> 31;
    return static_cast<uint8_t>(value);
}

BatchResult RunBatch(const uint8_t* data, size_t size, size_t iterations) {
    uint64_t accumulator = 0;
    auto start = std::chrono::steady_clock::now();
    for (size_t iteration = 0; iteration < iterations; ++iteration) {
        auto hash = RotoHash::Hash(data, size, static_cast<uint64_t>(iteration));
        std::array<uint64_t, 2> words;
        std::memcpy(words.data(), &hash, sizeof(hash));
        accumulator ^= words[0] ^ words[1];
    }
    auto elapsed = std::chrono::steady_clock::now() - start;
    Sink = accumulator;
    return {elapsed, accumulator};
}

size_t Calibrate(const uint8_t* data, size_t size) {
    size_t iterations = 1;
    while (true) {
        auto result = RunBatch(data, size, iterations);
        if (result.elapsed >= CalibrationTime)
            return iterations;
        if (iterations > SIZE_MAX / 2)
            return iterations;
        iterations *= 2;
    }
}

double Benchmark(const uint8_t* data, size_t size) {
    size_t iterations = Calibrate(data, size);
    std::array<double, Samples> samples;
    for (auto& sample : samples) {
        auto result = RunBatch(data, size, iterations);
        sample = std::chrono::duration<double, std::nano>(result.elapsed).count() /
            static_cast<double>(iterations);
    }
    std::sort(samples.begin(), samples.end());
    return samples[Samples / 2];
}

} // namespace

int main() {
    for (auto size : Sizes) {
        // All requested sizes are multiples of the alignment.
        auto* data = static_cast<uint8_t*>(std::aligned_alloc(64, size));
        if (data == nullptr)
            return 2;
        for (size_t index = 0; index < size; ++index)
            data[index] = TestByte(index);

        double nanoseconds = Benchmark(data, size);
        std::printf("%zu,%.9f\n", size, nanoseconds);
        std::free(data);
    }
    return Sink == UINT64_C(0xF00DFACECAFEBEEF) ? 3 : 0;
}
