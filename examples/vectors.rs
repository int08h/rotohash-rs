//! Generates and verifies RotoHash test-vector files, for checking that the
//! implementations for different architectures agree with each other.
//!
//! On the first machine (say, x86-64):
//!
//! ```console
//! cargo run --release --example vectors -- generate tests/vectors/x86_64.txt
//! ```
//!
//! Copy the file to the second machine (say, aarch64) and run:
//!
//! ```console
//! cargo run --release --example vectors -- verify tests/vectors/x86_64.txt
//! ```
//!
//! `verify` re-hashes every case at several input alignments and exits with
//! status 1 if any hash disagrees. With no paths, it verifies every file in
//! `tests/vectors/`, which `cargo test` also does.

#[path = "../tests/support/vectors.rs"]
mod support;

use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

const USAGE: &str = "\
usage:
  cargo run --release --example vectors -- generate [OUTPUT]
  cargo run --release --example vectors -- verify [FILE...]

generate  hashes the built-in case list on this machine and writes a vector
          file to OUTPUT (or standard output)
verify    re-hashes every case in each FILE on this machine and reports
          mismatches; with no FILE, verifies every *.txt in tests/vectors/";

fn main() -> ExitCode {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    let result = match arguments.first().map(String::as_str) {
        Some("generate") if arguments.len() <= 2 => generate(arguments.get(1).map(Path::new)),
        Some("verify") => verify(&arguments[1..]),
        _ => {
            eprintln!("{USAGE}");
            return ExitCode::from(2);
        }
    };
    match result {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::from(1),
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::from(2)
        }
    }
}

fn generate(output: Option<&Path>) -> Result<bool, String> {
    let describe = |error: io::Error| match output {
        Some(path) => format!("cannot write {}: {error}", path.display()),
        None => format!("cannot write standard output: {error}"),
    };
    match output {
        Some(path) => {
            let file = std::fs::File::create(path).map_err(describe)?;
            let mut writer = BufWriter::new(file);
            support::write_vectors(&mut writer).map_err(describe)?;
            writer.flush().map_err(describe)?;
            eprintln!(
                "wrote {} vectors from {} to {}",
                support::cases().len(),
                support::local_origin(),
                path.display()
            );
        }
        None => {
            let stdout = io::stdout();
            let mut writer = BufWriter::new(stdout.lock());
            support::write_vectors(&mut writer).map_err(describe)?;
            writer.flush().map_err(describe)?;
        }
    }
    Ok(true)
}

fn verify(paths: &[String]) -> Result<bool, String> {
    let paths: Vec<PathBuf> = if paths.is_empty() {
        default_vector_files()?
    } else {
        paths.iter().map(PathBuf::from).collect()
    };
    if paths.is_empty() {
        return Err("no vector files to verify".into());
    }

    let mut all_ok = true;
    for path in &paths {
        let text = std::fs::read_to_string(path)
            .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
        let file = support::parse(&text)
            .map_err(|error| format!("cannot parse {}: {error}", path.display()))?;
        let mismatches = support::verify(&file);
        let verdict = if mismatches.is_empty() {
            "OK"
        } else {
            "FAILED"
        };
        println!(
            "{}: {} vectors generated on {}, verified on {} at {} alignments: {} ({} mismatches)",
            path.display(),
            file.vectors.len(),
            file.origin(),
            support::local_origin(),
            support::OFFSETS.len(),
            verdict,
            mismatches.len()
        );
        for mismatch in &mismatches {
            println!("  {mismatch}");
        }
        all_ok &= mismatches.is_empty();
    }
    Ok(all_ok)
}

fn default_vector_files() -> Result<Vec<PathBuf>, String> {
    let directory = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/vectors");
    let entries = std::fs::read_dir(&directory)
        .map_err(|error| format!("cannot read {}: {error}", directory.display()))?;
    let mut files: Vec<PathBuf> = entries
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|extension| extension == "txt"))
        .collect();
    files.sort();
    Ok(files)
}
