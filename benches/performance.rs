use rotohash::hash_with_seed;
use std::hint::black_box;
#[cfg(target_arch = "x86_64")]
use std::path::{Path, PathBuf};
#[cfg(target_arch = "x86_64")]
use std::process::Command;
use std::time::{Duration, Instant};

const SIZES: [usize; 9] = [
    256,
    1024,
    4096,
    8192,
    16 * 1024,
    64 * 1024,
    256 * 1024,
    1024 * 1024,
    10 * 1024 * 1024,
];
const CALIBRATION_TIME: Duration = Duration::from_millis(100);
const SAMPLES: usize = 7;

#[repr(align(64))]
struct AlignedBlock([u8; 64]);

struct AlignedBuffer(Vec<AlignedBlock>);

impl AlignedBuffer {
    fn new(size: usize) -> Self {
        assert!(size.is_multiple_of(64));
        let mut blocks = Vec::with_capacity(size / 64);
        for block_index in 0..size / 64 {
            let mut block = AlignedBlock([0; 64]);
            for (byte_index, byte) in block.0.iter_mut().enumerate() {
                *byte = test_byte(block_index * 64 + byte_index);
            }
            blocks.push(block);
        }
        Self(blocks)
    }

    fn as_bytes(&self) -> &[u8] {
        // SAFETY: AlignedBlock has no padding because its field is exactly 64
        // bytes and its alignment is 64. The vector is contiguous and remains
        // borrowed for the lifetime of the returned slice.
        unsafe { std::slice::from_raw_parts(self.0.as_ptr().cast::<u8>(), self.0.len() * 64) }
    }
}

fn test_byte(index: usize) -> u8 {
    let mut value = (index as u64).wrapping_add(0x9E37_79B9_7F4A_7C15);
    value ^= value >> 30;
    value = value.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^= value >> 31;
    value as u8
}

fn run_batch(data: &[u8], iterations: usize) -> Duration {
    let mut accumulator = 0u128;
    let start = Instant::now();
    for iteration in 0..iterations {
        accumulator ^= hash_with_seed(data, iteration as u64).to_u128();
    }
    let elapsed = start.elapsed();
    black_box(accumulator);
    elapsed
}

fn calibrate(data: &[u8]) -> usize {
    let mut iterations = 1usize;
    loop {
        if run_batch(data, iterations) >= CALIBRATION_TIME {
            return iterations;
        }
        iterations = iterations.saturating_mul(2);
        if iterations == usize::MAX {
            return iterations;
        }
    }
}

fn benchmark_rust(data: &[u8]) -> f64 {
    let iterations = calibrate(data);
    let mut samples = [0.0; SAMPLES];
    for sample in &mut samples {
        *sample = run_batch(data, iterations).as_secs_f64() * 1e9 / iterations as f64;
    }
    samples.sort_by(f64::total_cmp);
    samples[SAMPLES / 2]
}

/// No C++ reference off x86-64.
#[cfg(not(target_arch = "x86_64"))]
fn cpp_results() -> Option<Vec<(usize, f64)>> {
    eprintln!("the C++ reference implementation is x86-64 only; reporting Rust results only");
    None
}

/// Runs the C++ reference benchmark.
#[cfg(target_arch = "x86_64")]
fn cpp_results() -> Option<Vec<(usize, f64)>> {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let source = manifest.join("benches/cpp_benchmark.cpp");
    let binary = temporary_binary();
    let compiler = std::env::var_os("CXX").unwrap_or_else(|| "c++".into());
    let status = Command::new(&compiler)
        .args(["-std=c++17", "-O3", "-DNDEBUG", "-march=native"])
        .arg(&source)
        .arg("-o")
        .arg(&binary)
        .status()
        .unwrap_or_else(|error| panic!("failed to run C++ compiler {compiler:?}: {error}"));
    assert!(status.success(), "C++ benchmark compilation failed");

    let output = Command::new(&binary)
        .output()
        .expect("failed to run compiled C++ benchmark");
    let _ = std::fs::remove_file(&binary);
    assert!(
        output.status.success(),
        "C++ benchmark failed with status {}: {}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );

    let results: Vec<(usize, f64)> = String::from_utf8(output.stdout)
        .expect("C++ output was not UTF-8")
        .lines()
        .map(|line| {
            let (size, nanoseconds) = line
                .split_once(',')
                .unwrap_or_else(|| panic!("invalid C++ result: {line}"));
            (
                size.parse().expect("invalid C++ size"),
                nanoseconds.parse().expect("invalid C++ timing"),
            )
        })
        .collect();
    assert_eq!(
        results.len(),
        SIZES.len(),
        "C++ returned the wrong row count"
    );
    for (&size, &(cpp_size, _)) in SIZES.iter().zip(&results) {
        assert_eq!(size, cpp_size, "C++ returned an unexpected size");
    }
    Some(results)
}

#[cfg(target_arch = "x86_64")]
fn temporary_binary() -> PathBuf {
    std::env::temp_dir().join(format!(
        "rotohash-cpp-benchmark-{}{}",
        std::process::id(),
        std::env::consts::EXE_SUFFIX
    ))
}

fn gib_per_second(size: usize, nanoseconds: f64) -> f64 {
    size as f64 / nanoseconds * 1e9 / 1024f64.powi(3)
}

fn size_label(size: usize) -> String {
    match size {
        16_384 => "16 KiB".into(),
        65_536 => "64 KiB".into(),
        262_144 => "256 KiB".into(),
        1_048_576 => "1 MiB".into(),
        10_485_760 => "10 MiB".into(),
        _ => format!("{size} B"),
    }
}

fn main() {
    if cfg!(debug_assertions) {
        eprintln!("performance benchmark requires `cargo bench --bench performance`");
        return;
    }

    let cpp = cpp_results();

    println!(
        "RotoHash performance ({} implementation, 64-byte aligned, hot input, median of {SAMPLES} samples)",
        rotohash::implementation()
    );
    match &cpp {
        Some(_) => println!(
            "{:<8} {:>14} {:>14} {:>14} {:>14} {:>12}",
            "Size", "Rust ns/hash", "C++ ns/hash", "Rust GiB/s", "C++ GiB/s", "Rust/C++"
        ),
        None => println!("{:<8} {:>14} {:>14}", "Size", "Rust ns/hash", "Rust GiB/s"),
    }

    for (index, &size) in SIZES.iter().enumerate() {
        let data = AlignedBuffer::new(size);
        let rust_ns = benchmark_rust(data.as_bytes());
        match &cpp {
            Some(cpp) => {
                let cpp_ns = cpp[index].1;
                println!(
                    "{:<8} {:>14.2} {:>14.2} {:>14.2} {:>14.2} {:>11.3}x",
                    size_label(size),
                    rust_ns,
                    cpp_ns,
                    gib_per_second(size, rust_ns),
                    gib_per_second(size, cpp_ns),
                    cpp_ns / rust_ns,
                );
            }
            None => println!(
                "{:<8} {:>14.2} {:>14.2}",
                size_label(size),
                rust_ns,
                gib_per_second(size, rust_ns),
            ),
        }
    }
}
