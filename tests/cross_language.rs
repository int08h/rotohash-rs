//! Compares against the C++ reference (x86-64 only; elsewhere, see the
//! `vectors` test).
#![cfg(target_arch = "x86_64")]

use rotohash::hash_with_seed;
use std::path::{Path, PathBuf};
use std::process::Command;

const SEEDS: [u64; 4] = [0, 1, 0x0123_4567_89AB_CDEF, u64::MAX];

fn test_byte(test_id: u64, index: usize) -> u8 {
    let mut value = (index as u64).wrapping_add(test_id.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    value ^= value >> 30;
    value = value.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^= value >> 31;
    value as u8
}

fn rust_results() -> Vec<String> {
    let mut results = Vec::new();
    let mut test_id = 0u64;

    for length in 0..=1024 {
        for seed in SEEDS {
            let data: Vec<_> = (0..length).map(|index| test_byte(test_id, index)).collect();
            results.push(format!("{:x}", hash_with_seed(&data, seed)));
            test_id += 1;
        }
    }

    for length in [4095, 4096, 4097, 65535, 65536, 262144, 262145] {
        for _offset in [0, 1, 15, 31, 63] {
            for seed in SEEDS {
                let data: Vec<_> = (0..length).map(|index| test_byte(test_id, index)).collect();
                results.push(format!("{:x}", hash_with_seed(&data, seed)));
                test_id += 1;
            }
        }
    }
    results
}

fn compile_and_run_cpp(name: &str, flags: &[&str]) -> Vec<String> {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let source = manifest.join("tests/cpp_reference.cpp");
    let binary = temporary_binary(name);
    let compiler = std::env::var_os("CXX").unwrap_or_else(|| "c++".into());

    let status = Command::new(&compiler)
        .arg("-std=c++17")
        .arg("-O2")
        .args(flags)
        .arg(&source)
        .arg("-o")
        .arg(&binary)
        .status()
        .unwrap_or_else(|error| panic!("failed to run C++ compiler {compiler:?}: {error}"));
    assert!(status.success(), "C++ reference compilation failed");

    let output = Command::new(&binary)
        .output()
        .expect("failed to run compiled C++ reference");
    let _ = std::fs::remove_file(&binary);
    assert!(
        output.status.success(),
        "C++ reference failed with status {}: {}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout)
        .expect("C++ output was not UTF-8")
        .lines()
        .map(str::to_owned)
        .collect()
}

fn temporary_binary(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "rotohash-{name}-{}{}",
        std::process::id(),
        std::env::consts::EXE_SUFFIX
    ))
}

#[test]
fn rust_matches_cpp_avx2() {
    let rust = rust_results();
    let cpp = compile_and_run_cpp("cpp-avx2", &["-mavx2", "-maes"]);
    assert_eq!(rust, cpp);
}

#[test]
fn rust_matches_cpp_avx512_vaes() {
    if !(std::arch::is_x86_feature_detected!("avx512f")
        && std::arch::is_x86_feature_detected!("avx512bw")
        && std::arch::is_x86_feature_detected!("avx512dq")
        && std::arch::is_x86_feature_detected!("avx512vl")
        && std::arch::is_x86_feature_detected!("vaes"))
    {
        eprintln!("skipping AVX-512 comparison: required CPU features are unavailable");
        return;
    }

    let rust = rust_results();
    let cpp = compile_and_run_cpp(
        "cpp-avx512",
        &[
            "-mavx2",
            "-maes",
            "-mavx512f",
            "-mavx512bw",
            "-mavx512dq",
            "-mavx512vl",
            "-mvaes",
        ],
    );
    assert_eq!(rust, cpp);
}
