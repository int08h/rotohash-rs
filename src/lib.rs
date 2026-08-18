//! RotoHash is a high-throughput, non-cryptographic 128-bit hash. RotoHash is
//! specifically for hashing large inputs at >100 GiB/sec.
//!
//! A Rust port of the reference C++ implementation, for x86-64 (AVX2/AES-NI or
//! AVX-512/VAES) and aarch64 (NEON/AES). All implementations hash identically.

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
compile_error!("rotohash-rs requires x86-64 or aarch64");

use core::fmt;

/// The 128-bit output of RotoHash
#[derive(Clone, Copy, Default, Eq, Hash, PartialEq)]
#[repr(transparent)]
pub struct Hash128([u8; 16]);

impl Hash128 {
    /// Returns the hash as 16 bytes.
    #[inline]
    pub const fn to_bytes(self) -> [u8; 16] {
        self.0
    }

    /// Borrows the hash as 16 bytes.
    #[inline]
    pub const fn as_bytes(&self) -> &[u8; 16] {
        &self.0
    }

    /// Interprets the result bytes as a little-endian integer.
    #[inline]
    pub const fn to_u128(self) -> u128 {
        u128::from_le_bytes(self.0)
    }
}

impl From<Hash128> for [u8; 16] {
    #[inline]
    fn from(value: Hash128) -> Self {
        value.0
    }
}

impl From<Hash128> for u128 {
    #[inline]
    fn from(value: Hash128) -> Self {
        value.to_u128()
    }
}

impl fmt::LowerHex for Hash128 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for byte in self.0 {
            write!(f, "{byte:02x}")?;
        }
        Ok(())
    }
}

impl fmt::UpperHex for Hash128 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for byte in self.0 {
            write!(f, "{byte:02X}")?;
        }
        Ok(())
    }
}

impl fmt::Debug for Hash128 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Hash128({self:x})")
    }
}

impl fmt::Display for Hash128 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::LowerHex::fmt(self, f)
    }
}

/// Hashes `data` with the default seed of zero.
///
/// # Panics
///
/// Panics without AVX2+AES-NI (x86-64) or NEON+AES (aarch64).
#[inline]
pub fn hash(data: &[u8]) -> Hash128 {
    hash_with_seed(data, 0)
}

/// The hash implementation selected for the current processor.
///
/// All implementations produce identical hashes. They differ only in the
/// instruction set they use.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Implementation {
    /// The 128-bit x86-64 implementation, using AVX2 and AES-NI.
    Avx2,
    /// The 512-bit x86-64 implementation, using AVX-512F, AVX-512BW, and VAES.
    Avx512,
    /// The 128-bit aarch64 implementation, using NEON and AES.
    Neon,
}

impl fmt::Display for Implementation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Implementation::Avx2 => "AVX2",
            Implementation::Avx512 => "AVX-512",
            Implementation::Neon => "NEON",
        })
    }
}

/// Returns the implementation that [`hash`] and [`hash_with_seed`] use on the
/// current processor.
///
/// # Panics
///
/// Panics without AVX2+AES-NI (x86-64) or NEON+AES (aarch64).
#[inline]
pub fn implementation() -> Implementation {
    #[cfg(target_arch = "x86_64")]
    {
        assert!(
            std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("aes"),
            "rotohash-rs requires an x86-64 processor with AVX2 and AES-NI"
        );

        if std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512bw")
            && std::arch::is_x86_feature_detected!("vaes")
        {
            Implementation::Avx512
        } else {
            Implementation::Avx2
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        assert!(
            std::arch::is_aarch64_feature_detected!("neon")
                && std::arch::is_aarch64_feature_detected!("aes"),
            "rotohash-rs requires an aarch64 processor with NEON and the AES extension"
        );
        Implementation::Neon
    }
}

/// Hashes `data` with a 64-bit seed.
///
/// # Panics
///
/// Panics without AVX2+AES-NI (x86-64) or NEON+AES (aarch64).
#[inline]
pub fn hash_with_seed(data: &[u8], seed: u64) -> Hash128 {
    match implementation() {
        // SAFETY: `implementation` asserts AVX-512F, AVX-512BW, and VAES.
        #[cfg(target_arch = "x86_64")]
        Implementation::Avx512 => unsafe { avx512::hash_avx512(data, seed) },

        // SAFETY: `implementation` asserts AVX2 and AES-NI.
        #[cfg(target_arch = "x86_64")]
        Implementation::Avx2 => unsafe { avx2::hash_avx2(data, seed) },

        // SAFETY: `implementation` asserts NEON and AES.
        #[cfg(target_arch = "aarch64")]
        Implementation::Neon => unsafe { neon::hash_neon(data, seed) },
    }
}

#[repr(align(64))]
struct AlignedConstant([u8; 256]);

// This constant is part of the algorithm and is copied from the
// C++ reference implementation.
static CONSTANT: AlignedConstant = AlignedConstant([
    0x8F, 0x5B, 0x86, 0x39, 0x77, 0xFC, 0x2A, 0x2E, 0xB2, 0x70, 0x4C, 0x69, 0xC2, 0x65, 0xD1, 0x91,
    0x71, 0x18, 0x15, 0xAD, 0xF5, 0x62, 0x95, 0x3E, 0x6E, 0x99, 0x94, 0xE3, 0xB1, 0x6C, 0x30, 0x6D,
    0xF8, 0xCA, 0x4B, 0xEF, 0xA8, 0x98, 0x75, 0x40, 0xD8, 0x43, 0x6B, 0x0A, 0x63, 0x11, 0x38, 0x21,
    0x16, 0x4A, 0xA7, 0x5D, 0x42, 0xAA, 0x8B, 0x33, 0x47, 0x19, 0x59, 0xBC, 0xC5, 0xD4, 0xF3, 0xD0,
    0x7A, 0x74, 0x4E, 0xB0, 0x37, 0x52, 0x10, 0x73, 0x8E, 0x06, 0x17, 0x20, 0xAE, 0xF2, 0xD6, 0x48,
    0x3D, 0x8A, 0xE8, 0x8D, 0xC9, 0x84, 0x68, 0x41, 0xDD, 0x1A, 0x1E, 0x2D, 0xA1, 0xA3, 0x8C, 0x0D,
    0x7B, 0x45, 0x83, 0x04, 0xC0, 0xA2, 0xE5, 0x67, 0xD3, 0x9A, 0x9F, 0xBD, 0xC8, 0x34, 0x24, 0xDC,
    0xFF, 0x0C, 0x51, 0x7C, 0x89, 0xDF, 0xA6, 0xCE, 0x5A, 0x3A, 0x79, 0x35, 0x76, 0x0E, 0x22, 0x60,
    0xE4, 0x80, 0x9D, 0x14, 0xF9, 0xF4, 0x7E, 0x0B, 0xBB, 0x58, 0x6A, 0x3F, 0xC1, 0x4D, 0xAC, 0xF0,
    0xFA, 0x5E, 0xE0, 0xB8, 0x92, 0xB5, 0x4F, 0xB4, 0xB3, 0x08, 0xEC, 0x3B, 0x64, 0x9E, 0xEB, 0x07,
    0xBE, 0x13, 0x23, 0x7D, 0x9B, 0x09, 0x28, 0x88, 0x90, 0x72, 0x5C, 0xF7, 0x36, 0x29, 0x1F, 0x97,
    0xA4, 0x25, 0x56, 0xED, 0x66, 0x78, 0xD5, 0x44, 0x61, 0x57, 0x2B, 0xD2, 0x54, 0xFB, 0xEE, 0xD9,
    0x2C, 0x02, 0x05, 0x2F, 0x87, 0x85, 0x9C, 0xD7, 0xE6, 0x12, 0x7F, 0xC6, 0xFD, 0x49, 0xC3, 0xEA,
    0xB7, 0x03, 0x26, 0x1C, 0xA5, 0xE7, 0x1B, 0xFE, 0xCC, 0x81, 0xB6, 0x3C, 0xF1, 0x93, 0x01, 0xE9,
    0x27, 0xF6, 0x82, 0xCB, 0x1D, 0xBA, 0xDE, 0xDA, 0x32, 0x6F, 0xC4, 0xC7, 0xA0, 0xAB, 0x53, 0x46,
    0xE2, 0x55, 0xCF, 0xCD, 0x96, 0x00, 0x31, 0x5F, 0x0F, 0xA9, 0xB9, 0xE1, 0xBF, 0x50, 0xAF, 0xDB,
]);

#[cfg(target_arch = "x86_64")]
mod avx2;
#[cfg(target_arch = "x86_64")]
mod avx512;
#[cfg(target_arch = "aarch64")]
mod neon;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn authoritative_cpp_verification_vector() {
        let data: Vec<u8> = (0..512).map(|value| value as u8).collect();
        let mut aggregate = hash(&[]).to_bytes();
        for size in 1..=512 {
            let value = hash(&data[..size]).to_bytes();
            for (output, input) in aggregate.iter_mut().zip(value) {
                *output ^= input;
            }
        }
        assert_eq!(
            aggregate,
            [
                0x1B, 0x1E, 0xE0, 0x82, 0xCB, 0xB5, 0x89, 0xAD, 0x2F, 0x56, 0xC8, 0x2A, 0xFE, 0xE9,
                0xA3, 0x6F,
            ]
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_and_avx512_paths_match() {
        if !(std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512bw")
            && std::arch::is_x86_feature_detected!("vaes"))
        {
            return;
        }

        let data: Vec<u8> = (0..2048)
            .map(|index| (index as u8).wrapping_mul(157).wrapping_add(83))
            .collect();
        for size in 0..=2048 {
            for seed in [0, 1, 0x0123_4567_89ab_cdef, u64::MAX] {
                // SAFETY: this test explicitly checks every required feature.
                let avx2 = unsafe { avx2::hash_avx2(&data[..size], seed) };
                let avx512 = unsafe { avx512::hash_avx512(&data[..size], seed) };
                assert_eq!(avx2, avx512, "mismatch for size {size}, seed {seed}");
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn implementation_matches_selected_path() {
        let data: Vec<u8> = (0..300).map(|value| value as u8).collect();
        let expected = match implementation() {
            // SAFETY: `implementation` reports the features it checked.
            Implementation::Avx512 => unsafe { avx512::hash_avx512(&data, 7) },
            Implementation::Avx2 => unsafe { avx2::hash_avx2(&data, 7) },
            other => unreachable!("{other} is not an x86-64 implementation"),
        };
        assert_eq!(hash_with_seed(&data, 7), expected);
        assert_eq!(
            implementation() == Implementation::Avx512,
            std::arch::is_x86_feature_detected!("avx512f")
                && std::arch::is_x86_feature_detected!("avx512bw")
                && std::arch::is_x86_feature_detected!("vaes")
        );
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn implementation_matches_selected_path() {
        let data: Vec<u8> = (0..300).map(|value| value as u8).collect();
        assert_eq!(implementation(), Implementation::Neon);
        // SAFETY: `implementation` asserts that NEON and AES are present.
        let expected = unsafe { neon::hash_neon(&data, 7) };
        assert_eq!(hash_with_seed(&data, 7), expected);
    }

    #[test]
    fn seed_and_length_affect_the_hash() {
        let data: Vec<u8> = (0..1000).map(|value| value as u8).collect();
        assert_ne!(hash_with_seed(&data, 0), hash_with_seed(&data, 1));
        assert_ne!(hash(&data[..999]), hash(&data));
        // The zero-filled tail block must not collide with real zero bytes.
        assert_ne!(hash(&[0u8; 17]), hash(&[0u8; 32]));
        assert_ne!(hash(&[]), hash(&[0u8; 16]));
    }

    #[test]
    fn result_conversions_preserve_cpp_byte_order() {
        let value = Hash128([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
        assert_eq!(format!("{value}"), "000102030405060708090a0b0c0d0e0f");
        assert_eq!(value.to_u128(), 0x0f0e0d0c0b0a09080706050403020100);
    }
}
