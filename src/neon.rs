//! The aarch64 (NEON + AES) implementation. Mirrors `avx2.rs` lane for lane.

use crate::{CONSTANT, Hash128};
use core::arch::aarch64::*;

/// A lane of state, kept lagged: the true value is `state ^ key`.
///
/// x86 `aesenc(v, k)` = `MC(SR(SB(v))) ^ k`; ARMv8 `AESE(v, k)` =
/// `SR(SB(v ^ k))`, `AESMC` = `MC`. Feeding `key` to the next `AESE`
/// saves an XOR per round.
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

/// `aesenc(value ^ key, 0)`.
#[inline(always)]
unsafe fn aes_round(value: uint8x16_t, key: uint8x16_t) -> uint8x16_t {
    // SAFETY: hash_neon verifies availability of the AES extension.
    unsafe { vaesmcq_u8(vaeseq_u8(value, key)) }
}

/// `aesenc(a ^ b, key)`.
#[inline(always)]
unsafe fn enc_xor(a: uint8x16_t, b: uint8x16_t, key: uint8x16_t) -> uint8x16_t {
    // SAFETY: hash_neon verifies availability of the AES extension.
    unsafe { veorq_u8(aes_round(a, b), key) }
}

/// `lane = aesenc(lane, input)`.
#[inline(always)]
unsafe fn absorb(lane: &mut Lane, input: uint8x16_t) {
    // SAFETY: hash_neon verifies availability of the AES extension.
    lane.state = unsafe { aes_round(lane.state, lane.key) };
    lane.key = input;
}

/// Per-32-bit rotate left by `shift & 0x1f`, as in `avx2.rs`.
#[inline]
#[target_feature(enable = "neon")]
fn rot(value: uint8x16_t, shift: uint8x16_t) -> uint8x16_t {
    let value = vreinterpretq_u32_u8(value);
    let left = vandq_u32(vreinterpretq_u32_u8(shift), vdupq_n_u32(0x1f));
    // No NEON rotate; negative shifts go right, and -32 yields zero.
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
pub(crate) unsafe fn hash_neon(data: &[u8], seed: u64) -> Hash128 {
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
    // The first AES round folds the seed in.
    let seed_vector = vreinterpretq_u8_u64(vdupq_n_u64(seed));
    for lane in &mut lanes[..4] {
        lane.key = seed_vector;
    }

    let mut offset = 0usize;
    if data.len() >= 256 {
        // Each pending key is a slice of the previous block; re-reading it
        // from cache, rather than keeping 16 keys in registers, avoids spills.
        let mut state = [zero; 16];
        for (state, lane) in state.iter_mut().zip(&lanes) {
            // SAFETY: hash_neon verifies availability of the AES extension.
            *state = unsafe { aes_round(lane.state, lane.key) };
        }
        offset = 256;

        while data.len() - offset >= 256 {
            let previous = offset - 256;
            for (index, state) in state.iter_mut().enumerate() {
                // SAFETY: `previous` precedes a full block at `offset`.
                let key = unsafe { load(data.as_ptr().add(previous + index * 16)) };
                *state = unsafe { aes_round(*state, key) };
            }
            offset += 256;
        }

        let previous = offset - 256;
        for (index, lane) in lanes.iter_mut().enumerate() {
            lane.state = state[index];
            // SAFETY: read above.
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

    // Un-lag the lanes.
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
