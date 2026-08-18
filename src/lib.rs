//! RotoHash is a high-throughput, non-cryptographic 128-bit hash. RotoHash is
//! specifically for hashing large inputs at >100 GiB/sec.
//!
//! This crate is a Rust port of the reference C++ implementation. It supports
//! x86-64 processors with AVX2 and AES-NI (using AVX-512 and VAES when
//! available) and aarch64 processors with NEON and the ARMv8 AES extension.
//! Every implementation produces identical hashes.

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
/// Panics if the current processor lacks the required features: AVX2 and
/// AES-NI on x86-64, or NEON and the AES extension on aarch64.
#[inline]
pub fn hash(data: &[u8]) -> Hash128 {
    hash_with_seed(data, 0)
}

/// The hash implementation selected for the current processor.
///
/// All implementations produce identical hashes. They differ only in the
/// instruction set they use.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum Implementation {
    /// The 128-bit x86-64 implementation, using AVX2 and AES-NI.
    Avx2,
    /// The 512-bit x86-64 implementation, using AVX-512F, AVX-512BW, and VAES.
    Avx512,
    /// The 128-bit aarch64 implementation, using NEON and the ARMv8 AES
    /// extension.
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
/// Panics if the current processor lacks the required features: AVX2 and
/// AES-NI on x86-64, or NEON and the AES extension on aarch64.
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
/// Panics if the current processor lacks the required features: AVX2 and
/// AES-NI on x86-64, or NEON and the AES extension on aarch64.
#[inline]
pub fn hash_with_seed(data: &[u8], seed: u64) -> Hash128 {
    match implementation() {
        // SAFETY: `implementation` only returns `Avx512` after checking the
        // AVX-512F, AVX-512BW, and VAES features used by this implementation.
        #[cfg(target_arch = "x86_64")]
        Implementation::Avx512 => unsafe { x86_64::hash_avx512(data, seed) },

        // SAFETY: `implementation` asserts that AVX2 and AES-NI are present.
        // The implementation uses unaligned loads only where the full input is
        // in bounds, and copies partial tails to a local, zero-filled block.
        #[cfg(target_arch = "x86_64")]
        Implementation::Avx2 => unsafe { x86_64::hash_avx2(data, seed) },

        // SAFETY: `implementation` asserts that NEON and AES are present. The
        // implementation uses unaligned loads only where the full input is in
        // bounds, and copies partial tails to a local, zero-filled block.
        #[cfg(target_arch = "aarch64")]
        Implementation::Neon => unsafe { aarch64::hash_neon(data, seed) },

        // `implementation` never returns a variant from another architecture.
        other => unreachable!("{other} is not available on this architecture"),
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

#[cfg(target_arch = "aarch64")]
mod aarch64 {
    use super::{CONSTANT, Hash128};
    use core::arch::aarch64::*;

    /// One 128-bit lane of hash state.
    ///
    /// x86 `aesenc(v, k)` computes `MixColumns(ShiftRows(SubBytes(v))) ^ k`,
    /// while ARMv8 `AESE(v, k)` computes `ShiftRows(SubBytes(v ^ k))` and
    /// `AESMC` applies `MixColumns`. To avoid a separate XOR per round, the
    /// state is kept lagged: the true x86 lane value is `state ^ key`, where
    /// `key` is the last value that was XORed in. Feeding `key` to the next
    /// `AESE` performs that XOR for free.
    #[derive(Clone, Copy)]
    struct Lane {
        state: uint8x16_t,
        key: uint8x16_t,
    }

    #[inline(always)]
    unsafe fn load(ptr: *const u8) -> uint8x16_t {
        // SAFETY: callers guarantee that 16 bytes can be read from `ptr`.
        unsafe { vld1q_u8(ptr) }
    }

    #[inline(always)]
    fn tail_load(data: &[u8], offset: usize) -> uint8x16_t {
        let mut block = [0u8; 16];
        block[..data.len() - offset].copy_from_slice(&data[offset..]);
        // SAFETY: block contains exactly 16 initialized bytes.
        unsafe { load(block.as_ptr()) }
    }

    /// `MixColumns(ShiftRows(SubBytes(value ^ key)))`: an x86 `aesenc` round
    /// without the trailing XOR, applied to `value ^ key`.
    #[inline(always)]
    unsafe fn aes_round(value: uint8x16_t, key: uint8x16_t) -> uint8x16_t {
        // SAFETY: hash_neon verifies availability of the AES extension.
        unsafe { vaesmcq_u8(vaeseq_u8(value, key)) }
    }

    /// x86 `aesenc(a ^ b, key)`, folding the XOR into the AES round.
    #[inline(always)]
    unsafe fn enc_xor(a: uint8x16_t, b: uint8x16_t, key: uint8x16_t) -> uint8x16_t {
        // SAFETY: hash_neon verifies availability of the AES extension.
        unsafe { veorq_u8(aes_round(a, b), key) }
    }

    /// Absorbs one 16-byte input block into a lane, equivalent to
    /// `lane = aesenc(lane, input)` on the true lane value.
    #[inline(always)]
    unsafe fn absorb(lane: &mut Lane, input: uint8x16_t) {
        // SAFETY: hash_neon verifies availability of the AES extension.
        lane.state = unsafe { aes_round(lane.state, lane.key) };
        lane.key = input;
    }

    /// Per-32-bit-lane variable rotate left, matching the AVX2 emulation:
    /// the shift count is masked to five bits.
    #[inline]
    #[target_feature(enable = "neon")]
    fn rot(value: uint8x16_t, shift: uint8x16_t) -> uint8x16_t {
        let value = vreinterpretq_u32_u8(value);
        let left = vandq_u32(vreinterpretq_u32_u8(shift), vdupq_n_u32(0x1f));
        // NEON has no variable rotate, but its variable shift takes a signed
        // count: negative counts shift right, and a count of -32 yields zero,
        // matching `_mm_srlv_epi32` by 32.
        let right = vreinterpretq_s32_u32(vsubq_u32(left, vdupq_n_u32(32)));
        let left = vreinterpretq_s32_u32(left);
        vreinterpretq_u8_u32(vorrq_u32(vshlq_u32(value, left), vshlq_u32(value, right)))
    }

    #[inline]
    #[target_feature(enable = "neon")]
    fn xor(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
        veorq_u8(a, b)
    }

    #[target_feature(enable = "neon,aes")]
    pub(super) unsafe fn hash_neon(data: &[u8], seed: u64) -> Hash128 {
        let zero = vdupq_n_u8(0);
        let mut c = [zero; 16];
        for (i, constant) in c.iter_mut().enumerate() {
            // SAFETY: CONSTANT contains 256 bytes, in sixteen full lanes.
            *constant = unsafe { load(CONSTANT.0.as_ptr().add(i * 16)) };
        }

        let mut lanes = [Lane {
            state: zero,
            key: zero,
        }; 16];
        for (lane, constant) in lanes.iter_mut().zip(c) {
            lane.state = constant;
        }
        // Rather than XORing the seed into the state, record it as the pending
        // key; the first AES round folds it in.
        let seed_vector = vreinterpretq_u8_u64(vdupq_n_u64(seed));
        for lane in &mut lanes[..4] {
            lane.key = seed_vector;
        }

        let mut offset = 0usize;
        if data.len() >= 256 {
            // In the bulk loop, the pending key of every lane is simply that
            // lane's slice of the previous 256-byte block. Rather than holding
            // sixteen states plus sixteen keys in registers (which spills),
            // re-read the previous block from cache: the state stays in
            // registers and each lane costs one load, AESE, and AESMC.
            let mut state = [zero; 16];
            for (state, lane) in state.iter_mut().zip(&lanes) {
                // SAFETY: hash_neon verifies availability of the AES extension.
                *state = unsafe { aes_round(lane.state, lane.key) };
            }
            offset = 256;

            while data.len() - offset >= 256 {
                let previous = offset - 256;
                for (index, state) in state.iter_mut().enumerate() {
                    // SAFETY: the loop condition guarantees a complete 256-byte
                    // block at `offset`, so the previous block is in bounds too.
                    let key = unsafe { load(data.as_ptr().add(previous + index * 16)) };
                    *state = unsafe { aes_round(*state, key) };
                }
                offset += 256;
            }

            let previous = offset - 256;
            for (index, lane) in lanes.iter_mut().enumerate() {
                lane.state = state[index];
                // SAFETY: the block at `previous` was fully consumed above.
                lane.key = unsafe { load(data.as_ptr().add(previous + index * 16)) };
            }
        }

        let length_vector = vreinterpretq_u8_u64(vdupq_n_u64(data.len() as u64));
        for lane in &mut lanes[4..8] {
            lane.key = xor(lane.key, length_vector);
        }

        let remainder = data.len() - offset;
        for group in 0..3 {
            if remainder >= (group + 1) * 64 {
                for lane in &mut lanes[group * 4..group * 4 + 4] {
                    // SAFETY: each selected group is a complete 64-byte block.
                    let input = unsafe { load(data.as_ptr().add(offset)) };
                    unsafe { absorb(lane, input) };
                    offset += 16;
                }
            }
        }

        if offset < data.len() {
            let remaining = data.len() - offset;
            let full_lanes = remaining / 16;
            for (index, lane) in lanes[12..].iter_mut().enumerate() {
                let input = if index < full_lanes {
                    // SAFETY: full_lanes counts complete 16-byte blocks.
                    unsafe { load(data.as_ptr().add(offset + index * 16)) }
                } else if index == full_lanes && !remaining.is_multiple_of(16) {
                    tail_load(data, offset + index * 16)
                } else {
                    zero
                };
                unsafe { absorb(lane, input) };
            }
        }

        // Resolve the lagged representation into the true lane values.
        let mut v = [zero; 16];
        for (value, lane) in v.iter_mut().zip(lanes) {
            *value = xor(lane.state, lane.key);
        }

        let mut m = [zero; 16];

        m[0] = unsafe { enc_xor(rot(v[4], v[0]), v[0], rot(c[4], v[8])) };
        m[1] = unsafe { enc_xor(rot(v[5], v[1]), v[1], rot(c[5], v[9])) };
        m[2] = unsafe { enc_xor(rot(v[6], v[2]), v[2], rot(c[6], v[10])) };
        m[3] = unsafe { enc_xor(rot(v[7], v[3]), v[3], rot(c[7], v[11])) };
        m[4] = unsafe { enc_xor(rot(v[8], v[4]), v[4], rot(c[0], v[12])) };
        m[5] = unsafe { enc_xor(rot(v[9], v[5]), v[5], rot(c[1], v[13])) };
        m[6] = unsafe { enc_xor(rot(v[10], v[6]), v[6], rot(c[2], v[14])) };
        m[7] = unsafe { enc_xor(rot(v[11], v[7]), v[7], rot(c[3], v[15])) };
        m[8] = unsafe { enc_xor(rot(v[12], v[8]), v[8], rot(c[12], v[0])) };
        m[9] = unsafe { enc_xor(rot(v[13], v[9]), v[9], rot(c[13], v[1])) };
        m[10] = unsafe { enc_xor(rot(v[14], v[10]), v[10], rot(c[14], v[2])) };
        m[11] = unsafe { enc_xor(rot(v[15], v[11]), v[11], rot(c[15], v[3])) };
        m[12] = unsafe { enc_xor(rot(v[0], v[12]), v[12], rot(c[8], v[4])) };
        m[13] = unsafe { enc_xor(rot(v[1], v[13]), v[13], rot(c[9], v[5])) };
        m[14] = unsafe { enc_xor(rot(v[2], v[14]), v[14], rot(c[10], v[6])) };
        m[15] = unsafe { enc_xor(rot(v[3], v[15]), v[15], rot(c[11], v[7])) };

        m[0] = unsafe { enc_xor(rot(v[8], v[0]), m[0], rot(c[8], v[12])) };
        m[1] = unsafe { enc_xor(rot(v[9], v[1]), m[1], rot(c[9], v[13])) };
        m[2] = unsafe { enc_xor(rot(v[10], v[2]), m[2], rot(c[10], v[14])) };
        m[3] = unsafe { enc_xor(rot(v[11], v[3]), m[3], rot(c[11], v[15])) };
        m[4] = unsafe { enc_xor(rot(v[12], v[4]), m[4], rot(c[12], v[8])) };
        m[5] = unsafe { enc_xor(rot(v[13], v[5]), m[5], rot(c[13], v[9])) };
        m[6] = unsafe { enc_xor(rot(v[14], v[6]), m[6], rot(c[14], v[10])) };
        m[7] = unsafe { enc_xor(rot(v[15], v[7]), m[7], rot(c[15], v[11])) };
        m[8] = unsafe { enc_xor(rot(v[0], v[8]), m[8], rot(c[0], v[4])) };
        m[9] = unsafe { enc_xor(rot(v[1], v[9]), m[9], rot(c[1], v[5])) };
        m[10] = unsafe { enc_xor(rot(v[2], v[10]), m[10], rot(c[2], v[6])) };
        m[11] = unsafe { enc_xor(rot(v[3], v[11]), m[11], rot(c[3], v[7])) };
        m[12] = unsafe { enc_xor(rot(v[4], v[12]), m[12], rot(c[4], v[0])) };
        m[13] = unsafe { enc_xor(rot(v[5], v[13]), m[13], rot(c[5], v[1])) };
        m[14] = unsafe { enc_xor(rot(v[6], v[14]), m[14], rot(c[6], v[2])) };
        m[15] = unsafe { enc_xor(rot(v[7], v[15]), m[15], rot(c[7], v[3])) };

        m[0] = unsafe { enc_xor(rot(v[12], v[0]), m[0], rot(c[12], v[4])) };
        m[1] = unsafe { enc_xor(rot(v[13], v[1]), m[1], rot(c[13], v[5])) };
        m[2] = unsafe { enc_xor(rot(v[14], v[2]), m[2], rot(c[14], v[6])) };
        m[3] = unsafe { enc_xor(rot(v[15], v[3]), m[3], rot(c[15], v[7])) };
        m[4] = unsafe { enc_xor(rot(v[0], v[4]), m[4], rot(c[8], v[0])) };
        m[5] = unsafe { enc_xor(rot(v[1], v[5]), m[5], rot(c[9], v[1])) };
        m[6] = unsafe { enc_xor(rot(v[2], v[6]), m[6], rot(c[10], v[2])) };
        m[7] = unsafe { enc_xor(rot(v[3], v[7]), m[7], rot(c[11], v[3])) };
        m[8] = unsafe { enc_xor(rot(v[4], v[8]), m[8], rot(c[4], v[12])) };
        m[9] = unsafe { enc_xor(rot(v[5], v[9]), m[9], rot(c[5], v[13])) };
        m[10] = unsafe { enc_xor(rot(v[6], v[10]), m[10], rot(c[6], v[14])) };
        m[11] = unsafe { enc_xor(rot(v[7], v[11]), m[11], rot(c[7], v[15])) };
        m[12] = unsafe { enc_xor(rot(v[8], v[12]), m[12], rot(c[0], v[8])) };
        m[13] = unsafe { enc_xor(rot(v[9], v[13]), m[13], rot(c[1], v[9])) };
        m[14] = unsafe { enc_xor(rot(v[10], v[14]), m[14], rot(c[2], v[10])) };
        m[15] = unsafe { enc_xor(rot(v[11], v[15]), m[15], rot(c[3], v[11])) };

        for i in 0..4 {
            v[i] = xor(xor(m[i], m[i + 4]), xor(m[i + 8], m[i + 12]));
        }

        m[0] = unsafe { enc_xor(rot(v[1], v[0]), v[0], v[3]) };
        m[1] = unsafe { enc_xor(rot(v[2], v[1]), v[1], v[0]) };
        m[2] = unsafe { enc_xor(rot(v[3], v[2]), v[2], v[1]) };
        m[3] = unsafe { enc_xor(rot(v[0], v[3]), v[3], v[2]) };
        m[0] = unsafe { enc_xor(rot(v[2], v[0]), m[0], v[1]) };
        m[1] = unsafe { enc_xor(rot(v[3], v[1]), m[1], v[2]) };
        m[2] = unsafe { enc_xor(rot(v[0], v[2]), m[2], v[3]) };
        m[3] = unsafe { enc_xor(rot(v[1], v[3]), m[3], v[0]) };
        m[0] = unsafe { enc_xor(rot(v[3], v[0]), m[0], v[2]) };
        m[1] = unsafe { enc_xor(rot(v[0], v[1]), m[1], v[3]) };
        m[2] = unsafe { enc_xor(rot(v[1], v[2]), m[2], v[0]) };
        m[3] = unsafe { enc_xor(rot(v[2], v[3]), m[3], v[1]) };

        let result = xor(xor(m[0], m[1]), xor(m[2], m[3]));
        let mut bytes = [0u8; 16];
        // SAFETY: bytes has room for one complete 128-bit store.
        unsafe { vst1q_u8(bytes.as_mut_ptr(), result) };
        Hash128(bytes)
    }
}

#[cfg(target_arch = "x86_64")]
mod x86_64 {
    use super::{CONSTANT, Hash128};
    use core::arch::x86_64::*;

    #[inline(always)]
    unsafe fn load(ptr: *const u8) -> __m128i {
        // SAFETY: callers guarantee that 16 bytes can be read from `ptr`.
        unsafe { _mm_loadu_si128(ptr.cast()) }
    }

    #[inline(always)]
    unsafe fn enc(value: __m128i, key: __m128i) -> __m128i {
        // SAFETY: hash_avx2 verifies availability of AES-NI.
        unsafe { _mm_aesenc_si128(value, key) }
    }

    #[inline(always)]
    unsafe fn rot(value: __m128i, shift: __m128i) -> __m128i {
        // AVX2 has variable shifts but no native rotate.
        let left_shift = unsafe { _mm_and_si128(_mm_set1_epi32(0x1f), shift) };
        let right_shift = unsafe { _mm_sub_epi32(_mm_set1_epi32(0x20), left_shift) };
        let left = unsafe { _mm_sllv_epi32(value, left_shift) };
        let right = unsafe { _mm_srlv_epi32(value, right_shift) };
        unsafe { _mm_or_si128(left, right) }
    }

    #[inline(always)]
    unsafe fn tail_load(data: &[u8], offset: usize) -> __m128i {
        let mut block = [0u8; 16];
        block[..data.len() - offset].copy_from_slice(&data[offset..]);
        // SAFETY: block contains exactly 16 initialized bytes.
        unsafe { load(block.as_ptr()) }
    }

    #[inline(always)]
    unsafe fn load512(ptr: *const u8) -> __m512i {
        // SAFETY: callers guarantee that 64 bytes can be read from `ptr`.
        unsafe { _mm512_loadu_si512(ptr.cast()) }
    }

    #[inline(always)]
    unsafe fn tail_load512(ptr: *const u8, bytes: usize) -> __m512i {
        debug_assert!((1..64).contains(&bytes));
        let mask = u64::MAX >> (64 - bytes);
        // SAFETY: the mask limits the load to exactly `bytes` accessible bytes.
        unsafe { _mm512_maskz_loadu_epi8(mask, ptr.cast()) }
    }

    #[inline(always)]
    unsafe fn enc512(value: __m512i, key: __m512i) -> __m512i {
        // SAFETY: hash_avx512 verifies availability of VAES and AVX-512F.
        unsafe { _mm512_aesenc_epi128(value, key) }
    }

    #[inline(always)]
    unsafe fn rot512(value: __m512i, shift: __m512i) -> __m512i {
        // SAFETY: hash_avx512 verifies availability of AVX-512F.
        unsafe { _mm512_rolv_epi32(value, shift) }
    }

    #[target_feature(enable = "avx512f,avx512bw,vaes")]
    pub(super) unsafe fn hash_avx512(data: &[u8], seed: u64) -> Hash128 {
        // SAFETY: CONSTANT contains four complete 64-byte vectors.
        let c0 = unsafe { load512(CONSTANT.0.as_ptr()) };
        let c1 = unsafe { load512(CONSTANT.0.as_ptr().add(64)) };
        let c2 = unsafe { load512(CONSTANT.0.as_ptr().add(128)) };
        let c3 = unsafe { load512(CONSTANT.0.as_ptr().add(192)) };

        let mut v0 = _mm512_xor_si512(c0, _mm512_set1_epi64(seed as i64));
        let mut v1 = c1;
        let mut v2 = c2;
        let mut v3 = c3;
        let mut offset = 0usize;

        while data.len() - offset >= 256 {
            v0 = unsafe { enc512(v0, load512(data.as_ptr().add(offset))) };
            v1 = unsafe { enc512(v1, load512(data.as_ptr().add(offset + 64))) };
            v2 = unsafe { enc512(v2, load512(data.as_ptr().add(offset + 128))) };
            v3 = unsafe { enc512(v3, load512(data.as_ptr().add(offset + 192))) };
            offset += 256;
        }

        v1 = _mm512_xor_si512(v1, _mm512_set1_epi64(data.len() as i64));
        let remainder = data.len() - offset;
        if remainder >= 64 {
            v0 = unsafe { enc512(v0, load512(data.as_ptr().add(offset))) };
            offset += 64;
        }
        if remainder >= 128 {
            v1 = unsafe { enc512(v1, load512(data.as_ptr().add(offset))) };
            offset += 64;
        }
        if remainder >= 192 {
            v2 = unsafe { enc512(v2, load512(data.as_ptr().add(offset))) };
            offset += 64;
        }
        if !remainder.is_multiple_of(64) {
            v3 = unsafe { enc512(v3, tail_load512(data.as_ptr().add(offset), remainder % 64)) };
        }

        let mut m0 = unsafe { enc512(_mm512_xor_si512(rot512(v1, v0), v0), rot512(c1, v2)) };
        let mut m1 = unsafe { enc512(_mm512_xor_si512(rot512(v2, v1), v1), rot512(c0, v3)) };
        let mut m2 = unsafe { enc512(_mm512_xor_si512(rot512(v3, v2), v2), rot512(c3, v0)) };
        let mut m3 = unsafe { enc512(_mm512_xor_si512(rot512(v0, v3), v3), rot512(c2, v1)) };

        m0 = unsafe { enc512(_mm512_xor_si512(rot512(v2, v0), m0), rot512(c2, v3)) };
        m1 = unsafe { enc512(_mm512_xor_si512(rot512(v3, v1), m1), rot512(c3, v2)) };
        m2 = unsafe { enc512(_mm512_xor_si512(rot512(v0, v2), m2), rot512(c0, v1)) };
        m3 = unsafe { enc512(_mm512_xor_si512(rot512(v1, v3), m3), rot512(c1, v0)) };

        m0 = unsafe { enc512(_mm512_xor_si512(rot512(v3, v0), m0), rot512(c3, v1)) };
        m1 = unsafe { enc512(_mm512_xor_si512(rot512(v0, v1), m1), rot512(c2, v0)) };
        m2 = unsafe { enc512(_mm512_xor_si512(rot512(v1, v2), m2), rot512(c1, v3)) };
        m3 = unsafe { enc512(_mm512_xor_si512(rot512(v2, v3), m3), rot512(c0, v2)) };

        v0 = _mm512_xor_si512(_mm512_xor_si512(m0, m1), _mm512_xor_si512(m2, m3));
        m0 = v0;
        m1 = _mm512_shuffle_i32x4::<0x39>(v0, v0);
        m2 = _mm512_shuffle_i32x4::<0x4E>(v0, v0);
        m3 = _mm512_shuffle_i32x4::<0x93>(v0, v0);

        v0 = unsafe { enc512(_mm512_xor_si512(rot512(m1, m0), v0), m3) };
        v0 = unsafe { enc512(_mm512_xor_si512(rot512(m2, m0), v0), m1) };
        v0 = unsafe { enc512(_mm512_xor_si512(rot512(m3, m0), v0), m2) };

        // Fold the four 128-bit lanes together like the C++ Lane<N>() extracts do.
        let s0 = _mm512_extracti32x4_epi32::<0>(v0);
        let s1 = _mm512_extracti32x4_epi32::<1>(v0);
        let s2 = _mm512_extracti32x4_epi32::<2>(v0);
        let s3 = _mm512_extracti32x4_epi32::<3>(v0);
        let result = _mm_xor_si128(_mm_xor_si128(s0, s1), _mm_xor_si128(s2, s3));

        let mut bytes = [0u8; 16];
        // SAFETY: bytes has room for one complete 128-bit store.
        unsafe { _mm_storeu_si128(bytes.as_mut_ptr().cast(), result) };
        Hash128(bytes)
    }

    #[target_feature(enable = "avx2,aes")]
    pub(super) unsafe fn hash_avx2(data: &[u8], seed: u64) -> Hash128 {
        let mut c = [_mm_setzero_si128(); 16];
        let mut v = [_mm_setzero_si128(); 16];
        for i in 0..16 {
            // SAFETY: CONSTANT contains 256 bytes, in sixteen full lanes.
            c[i] = unsafe { load(CONSTANT.0.as_ptr().add(i * 16)) };
            v[i] = c[i];
        }

        let seed_vector = _mm_set1_epi64x(seed as i64);
        for value in &mut v[..4] {
            *value = _mm_xor_si128(*value, seed_vector);
        }

        let mut offset = 0usize;
        while data.len() - offset >= 256 {
            for value in &mut v {
                // SAFETY: the loop condition guarantees a complete 256-byte block.
                let input = unsafe { load(data.as_ptr().add(offset)) };
                *value = unsafe { enc(*value, input) };
                offset += 16;
            }
        }

        let length_vector = _mm_set1_epi64x(data.len() as i64);
        for value in &mut v[4..8] {
            *value = _mm_xor_si128(*value, length_vector);
        }

        let remainder = data.len() - offset;
        for group in 0..3 {
            if remainder >= (group + 1) * 64 {
                for lane in 0..4 {
                    // SAFETY: each selected group is a complete 64-byte block.
                    let input = unsafe { load(data.as_ptr().add(offset)) };
                    v[group * 4 + lane] = unsafe { enc(v[group * 4 + lane], input) };
                    offset += 16;
                }
            }
        }

        if offset < data.len() {
            let remaining = data.len() - offset;
            let full_lanes = remaining / 16;
            for lane in 0..4 {
                let input = if lane < full_lanes {
                    // SAFETY: full_lanes counts complete 16-byte blocks.
                    unsafe { load(data.as_ptr().add(offset + lane * 16)) }
                } else if lane == full_lanes && !remaining.is_multiple_of(16) {
                    unsafe { tail_load(data, offset + lane * 16) }
                } else {
                    _mm_setzero_si128()
                };
                v[12 + lane] = unsafe { enc(v[12 + lane], input) };
            }
        }

        let mut m = [_mm_setzero_si128(); 16];

        m[0] = unsafe { enc(_mm_xor_si128(rot(v[4], v[0]), v[0]), rot(c[4], v[8])) };
        m[1] = unsafe { enc(_mm_xor_si128(rot(v[5], v[1]), v[1]), rot(c[5], v[9])) };
        m[2] = unsafe { enc(_mm_xor_si128(rot(v[6], v[2]), v[2]), rot(c[6], v[10])) };
        m[3] = unsafe { enc(_mm_xor_si128(rot(v[7], v[3]), v[3]), rot(c[7], v[11])) };
        m[4] = unsafe { enc(_mm_xor_si128(rot(v[8], v[4]), v[4]), rot(c[0], v[12])) };
        m[5] = unsafe { enc(_mm_xor_si128(rot(v[9], v[5]), v[5]), rot(c[1], v[13])) };
        m[6] = unsafe { enc(_mm_xor_si128(rot(v[10], v[6]), v[6]), rot(c[2], v[14])) };
        m[7] = unsafe { enc(_mm_xor_si128(rot(v[11], v[7]), v[7]), rot(c[3], v[15])) };
        m[8] = unsafe { enc(_mm_xor_si128(rot(v[12], v[8]), v[8]), rot(c[12], v[0])) };
        m[9] = unsafe { enc(_mm_xor_si128(rot(v[13], v[9]), v[9]), rot(c[13], v[1])) };
        m[10] = unsafe { enc(_mm_xor_si128(rot(v[14], v[10]), v[10]), rot(c[14], v[2])) };
        m[11] = unsafe { enc(_mm_xor_si128(rot(v[15], v[11]), v[11]), rot(c[15], v[3])) };
        m[12] = unsafe { enc(_mm_xor_si128(rot(v[0], v[12]), v[12]), rot(c[8], v[4])) };
        m[13] = unsafe { enc(_mm_xor_si128(rot(v[1], v[13]), v[13]), rot(c[9], v[5])) };
        m[14] = unsafe { enc(_mm_xor_si128(rot(v[2], v[14]), v[14]), rot(c[10], v[6])) };
        m[15] = unsafe { enc(_mm_xor_si128(rot(v[3], v[15]), v[15]), rot(c[11], v[7])) };

        m[0] = unsafe { enc(_mm_xor_si128(rot(v[8], v[0]), m[0]), rot(c[8], v[12])) };
        m[1] = unsafe { enc(_mm_xor_si128(rot(v[9], v[1]), m[1]), rot(c[9], v[13])) };
        m[2] = unsafe { enc(_mm_xor_si128(rot(v[10], v[2]), m[2]), rot(c[10], v[14])) };
        m[3] = unsafe { enc(_mm_xor_si128(rot(v[11], v[3]), m[3]), rot(c[11], v[15])) };
        m[4] = unsafe { enc(_mm_xor_si128(rot(v[12], v[4]), m[4]), rot(c[12], v[8])) };
        m[5] = unsafe { enc(_mm_xor_si128(rot(v[13], v[5]), m[5]), rot(c[13], v[9])) };
        m[6] = unsafe { enc(_mm_xor_si128(rot(v[14], v[6]), m[6]), rot(c[14], v[10])) };
        m[7] = unsafe { enc(_mm_xor_si128(rot(v[15], v[7]), m[7]), rot(c[15], v[11])) };
        m[8] = unsafe { enc(_mm_xor_si128(rot(v[0], v[8]), m[8]), rot(c[0], v[4])) };
        m[9] = unsafe { enc(_mm_xor_si128(rot(v[1], v[9]), m[9]), rot(c[1], v[5])) };
        m[10] = unsafe { enc(_mm_xor_si128(rot(v[2], v[10]), m[10]), rot(c[2], v[6])) };
        m[11] = unsafe { enc(_mm_xor_si128(rot(v[3], v[11]), m[11]), rot(c[3], v[7])) };
        m[12] = unsafe { enc(_mm_xor_si128(rot(v[4], v[12]), m[12]), rot(c[4], v[0])) };
        m[13] = unsafe { enc(_mm_xor_si128(rot(v[5], v[13]), m[13]), rot(c[5], v[1])) };
        m[14] = unsafe { enc(_mm_xor_si128(rot(v[6], v[14]), m[14]), rot(c[6], v[2])) };
        m[15] = unsafe { enc(_mm_xor_si128(rot(v[7], v[15]), m[15]), rot(c[7], v[3])) };

        m[0] = unsafe { enc(_mm_xor_si128(rot(v[12], v[0]), m[0]), rot(c[12], v[4])) };
        m[1] = unsafe { enc(_mm_xor_si128(rot(v[13], v[1]), m[1]), rot(c[13], v[5])) };
        m[2] = unsafe { enc(_mm_xor_si128(rot(v[14], v[2]), m[2]), rot(c[14], v[6])) };
        m[3] = unsafe { enc(_mm_xor_si128(rot(v[15], v[3]), m[3]), rot(c[15], v[7])) };
        m[4] = unsafe { enc(_mm_xor_si128(rot(v[0], v[4]), m[4]), rot(c[8], v[0])) };
        m[5] = unsafe { enc(_mm_xor_si128(rot(v[1], v[5]), m[5]), rot(c[9], v[1])) };
        m[6] = unsafe { enc(_mm_xor_si128(rot(v[2], v[6]), m[6]), rot(c[10], v[2])) };
        m[7] = unsafe { enc(_mm_xor_si128(rot(v[3], v[7]), m[7]), rot(c[11], v[3])) };
        m[8] = unsafe { enc(_mm_xor_si128(rot(v[4], v[8]), m[8]), rot(c[4], v[12])) };
        m[9] = unsafe { enc(_mm_xor_si128(rot(v[5], v[9]), m[9]), rot(c[5], v[13])) };
        m[10] = unsafe { enc(_mm_xor_si128(rot(v[6], v[10]), m[10]), rot(c[6], v[14])) };
        m[11] = unsafe { enc(_mm_xor_si128(rot(v[7], v[11]), m[11]), rot(c[7], v[15])) };
        m[12] = unsafe { enc(_mm_xor_si128(rot(v[8], v[12]), m[12]), rot(c[0], v[8])) };
        m[13] = unsafe { enc(_mm_xor_si128(rot(v[9], v[13]), m[13]), rot(c[1], v[9])) };
        m[14] = unsafe { enc(_mm_xor_si128(rot(v[10], v[14]), m[14]), rot(c[2], v[10])) };
        m[15] = unsafe { enc(_mm_xor_si128(rot(v[11], v[15]), m[15]), rot(c[3], v[11])) };

        for i in 0..4 {
            v[i] = _mm_xor_si128(
                _mm_xor_si128(m[i], m[i + 4]),
                _mm_xor_si128(m[i + 8], m[i + 12]),
            );
        }

        m[0] = unsafe { enc(_mm_xor_si128(rot(v[1], v[0]), v[0]), v[3]) };
        m[1] = unsafe { enc(_mm_xor_si128(rot(v[2], v[1]), v[1]), v[0]) };
        m[2] = unsafe { enc(_mm_xor_si128(rot(v[3], v[2]), v[2]), v[1]) };
        m[3] = unsafe { enc(_mm_xor_si128(rot(v[0], v[3]), v[3]), v[2]) };
        m[0] = unsafe { enc(_mm_xor_si128(rot(v[2], v[0]), m[0]), v[1]) };
        m[1] = unsafe { enc(_mm_xor_si128(rot(v[3], v[1]), m[1]), v[2]) };
        m[2] = unsafe { enc(_mm_xor_si128(rot(v[0], v[2]), m[2]), v[3]) };
        m[3] = unsafe { enc(_mm_xor_si128(rot(v[1], v[3]), m[3]), v[0]) };
        m[0] = unsafe { enc(_mm_xor_si128(rot(v[3], v[0]), m[0]), v[2]) };
        m[1] = unsafe { enc(_mm_xor_si128(rot(v[0], v[1]), m[1]), v[3]) };
        m[2] = unsafe { enc(_mm_xor_si128(rot(v[1], v[2]), m[2]), v[0]) };
        m[3] = unsafe { enc(_mm_xor_si128(rot(v[2], v[3]), m[3]), v[1]) };

        let result = _mm_xor_si128(_mm_xor_si128(m[0], m[1]), _mm_xor_si128(m[2], m[3]));
        let mut bytes = [0u8; 16];
        // SAFETY: bytes has room for one complete 128-bit store.
        unsafe { _mm_storeu_si128(bytes.as_mut_ptr().cast(), result) };
        Hash128(bytes)
    }
}

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
                let avx2 = unsafe { x86_64::hash_avx2(&data[..size], seed) };
                let avx512 = unsafe { x86_64::hash_avx512(&data[..size], seed) };
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
            Implementation::Avx512 => unsafe { x86_64::hash_avx512(&data, 7) },
            Implementation::Avx2 => unsafe { x86_64::hash_avx2(&data, 7) },
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
        let expected = unsafe { aarch64::hash_neon(&data, 7) };
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
