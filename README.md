# rotohash-rs

`rotohash-rs` is a Rust port of the
[C++ reference implementation of RotoHash](https://github.com/jandrewrogers/RotoHash),
a high-throughput (>100 GiB/sec) non-cryptographic 128-bit hash for large inputs on x86-64.

## Design

RotoHash is a 128-bit hash algorithm optimized for extreme throughput and exceptional hash 
quality. It was developed to address the need for robust I/O and storage checksums at 
sustained rates exceeding 100 GB/s but the design is general in nature.

This crate provides both AVX-512 and AVX2 based implementations for stable Rust. Performance 
of both backends is at parity (or better) with the C++ reference implementation. ARM64 is
not supported.

```rust
use rotohash::{hash, hash_with_seed};

let unseeded = hash(b"some data");
let seeded = hash_with_seed(b"some data", 42);

assert_ne!(unseeded, seeded);
println!("{unseeded}");
```

`Hash128::to_bytes` returns bytes in the same order as the C++ `__m128i` result. 
`Hash128::to_u128` interprets those bytes as a little-endian integer.

## Run-time CPU detection

This crate compiles both `hash_avx2` and `hash_avx512` and chooses between them at 
**runtime**. 

`implementation()` uses `is_x86_feature_detected!` (a cached cpuid check) 
to take the 512-bit path when AVX-512F, AVX-512BW, and VAES are present, and the 
AVX2/AES-NI path otherwise. One binary runs as fast as the hardware allows on any 
x86-64 processor with at least AVX2 and AES-NI.

`rotohash::implementation()` reports which implementation the library selected on 
the current processor:

```rust
use rotohash::{implementation, Implementation};

match implementation() {
    Implementation::Avx512 => println!("using the 512-bit AVX-512/VAES path"),
    Implementation::Avx2 => println!("using the 128-bit AVX2/AES-NI path"),
    _ => println!("using {}", implementation()),
}
```

## Verification

This crate produces the same results as the reference C++ version over 
thousands of seeded, unseeded, aligned, unaligned, boundary-sized, and 
large inputs tests:

```console
cargo test
```

The integration test requires a C++ compiler. Set `CXX` to select one; it
defaults to `c++`.

## Benchmarks

```console
cargo bench --bench performance
```

This compiles C++ with `-O3 -march=native`. Set `CXX` to choose a GCC- or 
Clang-compatible C++ compiler.

### Results

Measured on an AMD 9950X (Zen 5) with rustc 1.97.1 and GCC 14.2.0.

#### AVX-512 + VAES (512-bit path)

| Size    | Rust ns/hash | C++ ns/hash | Rust GiB/s | C++ GiB/s | Rust/C++ |
|---------|-------------:|------------:|-----------:|----------:|---------:|
| 256 B   |         5.12 |        5.24 |      46.55 |     45.49 |   1.023x |
| 1024 B  |         6.81 |        6.96 |     139.99 |    137.05 |   1.021x |
| 4096 B  |        15.21 |       15.43 |     250.88 |    247.23 |   1.015x |
| 8192 B  |        26.86 |       27.23 |     284.02 |    280.14 |   1.014x |
| 16 KiB  |        50.59 |       50.76 |     301.62 |    300.62 |   1.003x |
| 64 KiB  |       264.87 |      258.12 |     230.44 |    236.46 |   0.975x |
| 256 KiB |      1029.25 |     1023.72 |     237.20 |    238.48 |   0.995x |
| 1 MiB   |      5128.07 |     4881.35 |     190.43 |    200.06 |   0.952x |
| 10 MiB  |     66140.08 |    66968.73 |     147.65 |    145.82 |   1.013x |

#### AVX2 + AES-NI (128-bit path)

| Size    | Rust ns/hash | C++ ns/hash | Rust GiB/s | C++ GiB/s | Rust/C++ |
|---------|-------------:|------------:|-----------:|----------:|---------:|
| 256 B   |        24.79 |       38.54 |       9.62 |      6.19 |   1.555x |
| 1024 B  |        29.69 |       44.50 |      32.12 |     21.43 |   1.499x |
| 4096 B  |        46.83 |       62.40 |      81.46 |     61.13 |   1.332x |
| 8192 B  |        70.45 |       86.78 |     108.30 |     87.92 |   1.232x |
| 16 KiB  |       116.61 |      133.22 |     130.85 |    114.54 |   1.142x |
| 64 KiB  |       402.17 |      429.84 |     151.77 |    141.99 |   1.069x |
| 256 KiB |      1549.67 |     1597.29 |     157.54 |    152.85 |   1.031x |
| 1 MiB   |      7075.68 |     7463.68 |     138.02 |    130.84 |   1.055x |
| 10 MiB  |     67048.66 |    68939.60 |     145.65 |    141.65 |   1.028x |


## License

RotoHash is Copyright (c) 2025 Andrew Rogers, see `LICENSE.rotohash`.

rotohash-rs is Copyright (c) 2026 Stuart Stock, distributed under the MIT license, 
see `LICENSE` for details.

