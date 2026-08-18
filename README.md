# rotohash-rs

`rotohash-rs` is a Rust port of the
[reference implementation of RotoHash](https://github.com/jandrewrogers/RotoHash),
a high-throughput (>100 GiB/sec) non-cryptographic 128-bit hash for large inputs.
It supports x86-64 (AVX2/AES-NI and AVX-512/VAES) and aarch64 (NEON/AES).

## Design

RotoHash is a 128-bit hash algorithm optimized for extreme throughput and exceptional hash 
quality. It was developed to address the need for robust I/O and storage checksums at 
sustained rates exceeding 100 GB/s but the design is general in nature.

This crate provides AVX-512, AVX2, and NEON based implementations for stable Rust. All three
produce identical hashes. Performance of the x86-64 backends is at parity (or better) with the
C++ reference implementation; the aarch64 backend has no C++ counterpart (the reference
implementation is x86-64 only) but is verified bit-for-bit against it.

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

On x86-64, this crate compiles both `hash_avx2` and `hash_avx512` and chooses between 
them at **runtime**. `implementation()` uses `is_x86_feature_detected!` (a cached cpuid 
check) to take the 512-bit path when AVX-512F, AVX-512BW, and VAES are present, and the 
AVX2/AES-NI path otherwise. One binary runs as fast as the hardware allows on any 
x86-64 processor with at least AVX2 and AES-NI.

On aarch64, the crate compiles `hash_neon`, which uses NEON and the ARMv8 AES 
extension (`AESE`/`AESMC`). `implementation()` uses `is_aarch64_feature_detected!` to
confirm the AES extension is present. Nearly every aarch64 processor made for general 
use has it (Apple M-series, AWS Graviton, Ampere, Snapdragon, Raspberry Pi 5); the 
Raspberry Pi 4 is a notable exception, and `hash` panics there.

`rotohash::implementation()` reports which implementation the library selected on 
the current processor:

```rust
use rotohash::{implementation, Implementation};

match implementation() {
    Implementation::Avx512 => println!("using the 512-bit AVX-512/VAES path"),
    Implementation::Avx2 => println!("using the 128-bit AVX2/AES-NI path"),
    Implementation::Neon => println!("using the 128-bit NEON/AES path"),
    _ => println!("using {}", implementation()),
}
```

### How the NEON port matches x86-64

x86 `aesenc(v, k)` computes `MixColumns(ShiftRows(SubBytes(v))) ^ k`, while ARMv8
`AESE(v, k)` computes `ShiftRows(SubBytes(v ^ k))` and `AESMC` applies `MixColumns`. 
The NEON port keeps each 128-bit lane in a *lagged* form, `state ^ pending_key`, and 
feeds `pending_key` into the next `AESE`, so each absorbed block costs one load, one 
`AESE`, and one `AESMC` per lane with no separate XOR. The per-32-bit-lane variable 
rotate uses NEON's signed variable shift (`vshlq_u32`) in both directions.

## Verification

This crate produces the same results as the reference C++ version over 
thousands of seeded, unseeded, aligned, unaligned, boundary-sized, and 
large inputs tests:

```console
cargo test
```

On x86-64, the `cross_language` integration test compiles and runs the C++ 
reference and compares every hash. It requires a C++ compiler; set `CXX` to 
select one (it defaults to `c++`).

On every architecture, the unit tests check the C++ author's authoritative 
verification vector (an aggregate over 513 inputs), and the `vectors` 
integration test re-hashes every case in every file under `tests/vectors/` 
at five different input alignments and compares against the recorded 
results. Those files record hashes computed by a specific architecture and 
implementation, so `cargo test` on an aarch64 machine checks NEON against 
recorded x86-64 results, and vice versa.

### Comparing aarch64 against x86-64 on two machines

The `vectors` example generates and verifies vector files. To check that the
aarch64 implementation agrees with the x86-64 implementation:

1. On the x86-64 machine, generate a vector file:

   ```console
   cargo run --release --example vectors -- generate tests/vectors/x86_64-avx512.txt
   ```

   The file records the machine's architecture and the implementation it used 
   (`AVX-512` or `AVX2`) in its header. Machines with and without AVX-512 
   produce identical hashes, so one file per architecture is enough; name it 
   for whichever path the machine took.

2. Copy the file to the aarch64 machine (`scp`, `git push`/`git pull`, or any
   other means) into `tests/vectors/`.

3. On the aarch64 machine, verify it:

   ```console
   cargo run --release --example vectors -- verify tests/vectors/x86_64-avx512.txt
   ```

   or simply run `cargo test`, which verifies every file in `tests/vectors/`.
   `verify` prints one line per file and any mismatching cases (length, seed,
   alignment, expected and actual hash), and exits with status 1 if any hash 
   disagrees.

The reverse direction works the same way: `tests/vectors/aarch64-neon.txt` 
was generated on an Apple M3 and is checked in, so `cargo test` on any x86-64 
machine already compares AVX2 or AVX-512 against NEON. Once you have generated
`x86_64-avx512.txt` (or `x86_64-avx2.txt`), commit it too so both directions 
are covered on every machine.

Each vector file has 2102 cases: every length from 0 to 1024 bytes with two 
seeds, and thirteen larger lengths around block and page boundaries (2 KiB, 
4 KiB, 64 KiB, 256 KiB, and 1 MiB, ±1) with four seeds. Input bytes are 
derived deterministically from the length and seed, so the file is 
self-contained.

## Benchmarks

```console
cargo bench --bench performance
```

On x86-64, this also compiles the C++ reference with `-O3 -march=native` and 
reports both. Set `CXX` to choose a GCC- or Clang-compatible C++ compiler. On 
aarch64, only the Rust results are reported.

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

#### NEON + AES (aarch64)

Measured on an Apple M3 with rustc 1.97.1.

| Size    | Rust ns/hash | Rust GiB/s |
|---------|-------------:|-----------:|
| 256 B   |        41.32 |       5.77 |
| 1024 B  |        44.37 |      21.49 |
| 4096 B  |        62.80 |      60.74 |
| 8192 B  |        87.34 |      87.36 |
| 16 KiB  |       136.47 |     111.81 |
| 64 KiB  |       434.87 |     140.35 |
| 256 KiB |      2500.51 |      97.64 |
| 1 MiB   |      9785.15 |      99.80 |
| 10 MiB  |    100253.52 |      97.41 |


## License

RotoHash is Copyright (c) 2025 Andrew Rogers, see `LICENSE.rotohash`.

rotohash-rs is Copyright (c) 2026 Stuart Stock, distributed under the MIT license, 
see `LICENSE` for details.

