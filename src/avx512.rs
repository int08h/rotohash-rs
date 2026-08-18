//! The 512-bit x86-64 (AVX-512F/BW + VAES) implementation; each vector holds
//! four of the sixteen `avx2.rs` lanes.

use crate::{CONSTANT, Hash128};
use core::arch::x86_64::*;

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
pub(crate) unsafe fn hash_avx512(data: &[u8], seed: u64) -> Hash128 {
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
