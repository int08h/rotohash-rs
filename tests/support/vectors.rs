//! Test-vector files, shared by the `vectors` test and example.
//!
//! `#` lines are metadata; other lines are `<length> <seed hex> <hash hex>`.
//! Inputs derive from length and seed, so files are self-contained.

#![allow(dead_code)]

use rotohash::hash_with_seed;
use std::fmt;
use std::io::{self, Write};

/// Bumped when cases or input derivation change.
pub const FORMAT_VERSION: u32 = 1;

/// Seeds used by the cases.
pub const SEEDS: [u64; 4] = [0, 1, 0x0123_4567_89AB_CDEF, u64::MAX];

/// Alignments at which every case is verified.
pub const OFFSETS: [usize; 5] = [0, 1, 15, 31, 63];

/// Lengths `0..=` this are hashed with two seeds.
pub const DENSE_LENGTH_LIMIT: usize = 1024;

/// Boundary lengths, hashed with each seed.
pub const SPARSE_LENGTHS: [usize; 13] = [
    2047,
    2048,
    2049,
    4095,
    4096,
    4097,
    65_535,
    65_536,
    65_537,
    262_144,
    262_145,
    1 << 20,
    (1 << 20) + 33,
];

/// One input, derived from `(length, seed)`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Case {
    pub length: usize,
    pub seed: u64,
}

impl Case {
    /// The input bytes (SplitMix64-style).
    pub fn data(&self) -> Vec<u8> {
        let id = (self.length as u64) ^ self.seed.wrapping_mul(0x2545_F491_4F6C_DD1D);
        (0..self.length)
            .map(|index| {
                let mut value = (index as u64).wrapping_add(id.wrapping_mul(0x9E37_79B9_7F4A_7C15));
                value ^= value >> 30;
                value = value.wrapping_mul(0xBF58_476D_1CE4_E5B9);
                value ^= value >> 27;
                value = value.wrapping_mul(0x94D0_49BB_1331_11EB);
                value ^= value >> 31;
                value as u8
            })
            .collect()
    }

    /// The hash on this machine.
    pub fn hash(&self) -> String {
        format!("{:x}", hash_with_seed(&self.data(), self.seed))
    }
}

/// The ordered case list.
pub fn cases() -> Vec<Case> {
    let mut cases = Vec::new();
    for length in 0..=DENSE_LENGTH_LIMIT {
        for seed in [SEEDS[0], SEEDS[2]] {
            cases.push(Case { length, seed });
        }
    }
    for length in SPARSE_LENGTHS {
        for seed in SEEDS {
            cases.push(Case { length, seed });
        }
    }
    cases
}

/// One vector-file line.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Vector {
    pub case: Case,
    pub hash: String,
}

/// A parsed vector file.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct VectorFile {
    /// `# key=value` lines, in order.
    pub metadata: Vec<(String, String)>,
    pub vectors: Vec<Vector>,
}

impl VectorFile {
    pub fn metadata(&self, key: &str) -> Option<&str> {
        self.metadata
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }

    /// E.g. `x86_64/AVX-512`.
    pub fn origin(&self) -> String {
        format!(
            "{}/{}",
            self.metadata("arch").unwrap_or("unknown"),
            self.metadata("implementation").unwrap_or("unknown")
        )
    }
}

/// [`VectorFile::origin`] for this machine.
pub fn local_origin() -> String {
    format!("{}/{}", std::env::consts::ARCH, rotohash::implementation())
}

/// Writes a vector file for this machine.
pub fn write_vectors(out: &mut impl Write) -> io::Result<()> {
    writeln!(out, "# rotohash-rs test vectors")?;
    writeln!(out, "# format={FORMAT_VERSION}")?;
    writeln!(out, "# arch={}", std::env::consts::ARCH)?;
    writeln!(out, "# os={}", std::env::consts::OS)?;
    writeln!(out, "# implementation={}", rotohash::implementation())?;
    writeln!(out, "# crate={}", env!("CARGO_PKG_VERSION"))?;
    writeln!(out, "# columns: length seed(hex) hash(hex)")?;
    for case in cases() {
        writeln!(out, "{} {:x} {}", case.length, case.seed, case.hash())?;
    }
    Ok(())
}

/// Parses a vector file.
pub fn parse(text: &str) -> Result<VectorFile, String> {
    let mut file = VectorFile::default();
    for (index, raw) in text.lines().enumerate() {
        let line_number = index + 1;
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }
        if let Some(comment) = line.strip_prefix('#') {
            let comment = comment.trim();
            if let Some((key, value)) = comment.split_once('=')
                && !key.contains(char::is_whitespace)
            {
                file.metadata
                    .push((key.trim().to_owned(), value.trim().to_owned()));
            }
            continue;
        }

        let mut fields = line.split_whitespace();
        let (Some(length), Some(seed), Some(hash), None) =
            (fields.next(), fields.next(), fields.next(), fields.next())
        else {
            return Err(format!("line {line_number}: expected `length seed hash`"));
        };
        let length = length
            .parse::<usize>()
            .map_err(|error| format!("line {line_number}: invalid length: {error}"))?;
        let seed = u64::from_str_radix(seed, 16)
            .map_err(|error| format!("line {line_number}: invalid seed: {error}"))?;
        if hash.len() != 32 || !hash.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Err(format!("line {line_number}: hash must be 32 hex digits"));
        }
        file.vectors.push(Vector {
            case: Case { length, seed },
            hash: hash.to_ascii_lowercase(),
        });
    }

    if let Some(format) = file.metadata("format")
        && format != FORMAT_VERSION.to_string()
    {
        return Err(format!(
            "unsupported vector format {format} (this build understands {FORMAT_VERSION})"
        ));
    }
    Ok(file)
}

/// A local hash that disagrees with the file.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Mismatch {
    pub case: Case,
    pub offset: usize,
    pub expected: String,
    pub actual: String,
}

impl fmt::Display for Mismatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "length {} seed {:x} offset {}: expected {}, got {}",
            self.case.length, self.case.seed, self.offset, self.expected, self.actual
        )
    }
}

/// Re-hashes every vector at every [`OFFSETS`] entry.
pub fn verify(file: &VectorFile) -> Vec<Mismatch> {
    #[derive(Clone, Copy)]
    #[repr(align(64))]
    struct Aligned([u8; 64]);

    let mut mismatches = Vec::new();
    for vector in &file.vectors {
        let data = vector.case.data();
        let padding = OFFSETS.iter().copied().max().unwrap_or(0);
        let mut buffer = vec![Aligned([0xA5; 64]); (padding + data.len()).div_ceil(64) + 1];
        // SAFETY: `Aligned` is 64 unpadded bytes.
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(buffer.as_mut_ptr().cast::<u8>(), buffer.len() * 64)
        };
        for offset in OFFSETS {
            bytes[offset..offset + data.len()].copy_from_slice(&data);
            let actual = format!(
                "{:x}",
                hash_with_seed(&bytes[offset..offset + data.len()], vector.case.seed)
            );
            if actual != vector.hash {
                mismatches.push(Mismatch {
                    case: vector.case,
                    offset,
                    expected: vector.hash.clone(),
                    actual,
                });
            }
        }
    }
    mismatches
}
