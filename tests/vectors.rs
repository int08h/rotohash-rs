//! Verifies this machine's implementation against every vector file checked
//! in under `tests/vectors/`. Each file records hashes produced by a specific
//! architecture and implementation, so running `cargo test` on an aarch64
//! machine checks NEON against x86-64 results, and vice versa.
//!
//! Generate a new file on any supported machine with:
//!
//! ```console
//! cargo run --release --example vectors -- generate tests/vectors/<name>.txt
//! ```

#[path = "support/vectors.rs"]
mod support;

use std::path::{Path, PathBuf};

fn vector_files() -> Vec<PathBuf> {
    let directory = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/vectors");
    let mut files: Vec<_> = std::fs::read_dir(&directory)
        .unwrap_or_else(|error| panic!("cannot read {}: {error}", directory.display()))
        .map(|entry| entry.expect("cannot read directory entry").path())
        .filter(|path| path.extension().is_some_and(|extension| extension == "txt"))
        .collect();
    files.sort();
    files
}

#[test]
fn checked_in_vectors_match_this_machine() {
    let files = vector_files();
    assert!(!files.is_empty(), "no vector files found in tests/vectors");

    let mut failures = Vec::new();
    for path in &files {
        let text = std::fs::read_to_string(path)
            .unwrap_or_else(|error| panic!("cannot read {}: {error}", path.display()));
        let file = support::parse(&text)
            .unwrap_or_else(|error| panic!("cannot parse {}: {error}", path.display()));
        assert_eq!(
            file.vectors.len(),
            support::cases().len(),
            "{} does not contain the complete case list",
            path.display()
        );
        let mismatches = support::verify(&file);
        eprintln!(
            "{}: {} vectors from {} verified on {}: {} mismatches",
            path.display(),
            file.vectors.len(),
            file.origin(),
            support::local_origin(),
            mismatches.len()
        );
        for mismatch in mismatches.iter().take(10) {
            eprintln!("  {mismatch}");
        }
        if !mismatches.is_empty() {
            failures.push((path.clone(), mismatches.len()));
        }
    }
    assert!(failures.is_empty(), "vector mismatches: {failures:?}");
}

#[test]
fn generated_vectors_round_trip() {
    let mut text = Vec::new();
    support::write_vectors(&mut text).expect("writing to a Vec cannot fail");
    let text = String::from_utf8(text).expect("vector files are ASCII");

    let file = support::parse(&text).expect("generated file must parse");
    assert_eq!(file.metadata("arch"), Some(std::env::consts::ARCH));
    assert_eq!(
        file.metadata("implementation").map(str::to_owned),
        Some(rotohash::implementation().to_string())
    );
    assert_eq!(file.vectors.len(), support::cases().len());
    assert!(support::verify(&file).is_empty());
}

#[test]
fn parser_rejects_malformed_lines() {
    assert!(support::parse("1 0").is_err());
    assert!(support::parse("x 0 00000000000000000000000000000000").is_err());
    assert!(support::parse("1 zz 00000000000000000000000000000000").is_err());
    assert!(support::parse("1 0 0000").is_err());
    assert!(support::parse("# format=999\n").is_err());

    let file = support::parse(
        "# rotohash-rs test vectors\n# arch=x86_64\n\n1 ffffffffffffffff 000102030405060708090A0B0C0D0E0F\n",
    )
    .expect("well-formed file must parse");
    assert_eq!(file.metadata("arch"), Some("x86_64"));
    assert_eq!(file.vectors.len(), 1);
    assert_eq!(file.vectors[0].case.length, 1);
    assert_eq!(file.vectors[0].case.seed, u64::MAX);
    assert_eq!(file.vectors[0].hash, "000102030405060708090a0b0c0d0e0f");
}
