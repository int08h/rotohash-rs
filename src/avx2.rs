//! The 128-bit x86-64 (AVX2 + AES-NI) implementation, which `avx512.rs` and
//! `neon.rs` mirror.

use crate::{CONSTANT, Hash128};
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

#[target_feature(enable = "avx2,aes")]
pub(crate) unsafe fn hash_avx2(data: &[u8], seed: u64) -> Hash128 {
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
