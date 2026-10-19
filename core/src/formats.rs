//! Block layouts of the GSQ export and their reference (scalar, float) decoders.
//! Bit-for-bit the decoders of xolotl tools/v41/v41_formats.py and ggml's dequantize_row_*.

use crate::gguf::GType;
use half::{bf16, f16};

#[inline]
pub fn f16_at(b: &[u8], i: usize) -> f32 {
    f16::from_le_bytes([b[i], b[i + 1]]).to_f32()
}

/// One row of `n` weights in type `t` -> f32.
pub fn dequant_row(t: GType, raw: &[u8], out: &mut [f32]) {
    let n = out.len();
    assert_eq!(raw.len(), t.row_bytes(n), "{t:?} row bytes");
    match t {
        GType::F32 => {
            for (i, o) in out.iter_mut().enumerate() {
                *o = f32::from_le_bytes(raw[4 * i..4 * i + 4].try_into().unwrap());
            }
        }
        GType::F16 => {
            for (i, o) in out.iter_mut().enumerate() {
                *o = f16_at(raw, 2 * i);
            }
        }
        GType::BF16 => {
            for (i, o) in out.iter_mut().enumerate() {
                *o = bf16::from_le_bytes([raw[2 * i], raw[2 * i + 1]]).to_f32();
            }
        }
        GType::Q8_0 => {
            for (b, blk) in raw.chunks_exact(34).enumerate() {
                let d = f16_at(blk, 0);
                for j in 0..32 {
                    out[32 * b + j] = blk[2 + j] as i8 as f32 * d;
                }
            }
        }
        GType::Q2_0 => {
            // fp16 d, then qs[16]: byte m holds weights 4m..4m+3 (2 bits each, code - 1 in -1..2)
            for (b, blk) in raw.chunks_exact(18).enumerate() {
                let d = f16_at(blk, 0);
                for m in 0..16 {
                    let q = blk[2 + m];
                    for k in 0..4 {
                        out[64 * b + 4 * m + k] = (((q >> (2 * k)) & 3) as f32 - 1.0) * d;
                    }
                }
            }
        }
        GType::Ptq1_0 => {
            // qs[0..16]: weights m + 16n (n = 0..4); qs[16..24]: 80 + m + 8n; qh[0..2]: 120 + h + 2n (n = 0..3);
            // a byte is ceil(v * 256 / 243) of the base-3 number v; trit n = ((q * 3^n) & 255) * 3 >> 8
            for (b, blk) in raw.chunks_exact(28).enumerate() {
                let d = f16_at(blk, 26);
                let o = &mut out[128 * b..128 * b + 128];
                let trit = |q: u8, n: u32| -> f32 { (((((q as u32) * 3u32.pow(n)) & 0xFF) * 3) >> 8) as f32 - 1.0 };
                for m in 0..16 {
                    for n in 0..5 {
                        o[m + 16 * n as usize] = trit(blk[m], n) * d;
                    }
                }
                for m in 0..8 {
                    for n in 0..5 {
                        o[80 + m + 8 * n as usize] = trit(blk[16 + m], n) * d;
                    }
                }
                for h in 0..2 {
                    for n in 0..4 {
                        o[120 + h + 2 * n as usize] = trit(blk[24 + h], n) * d;
                    }
                }
            }
        }
        GType::Q3K => {
            for (b, blk) in raw.chunks_exact(110).enumerate() {
                let (hm, qs, raw12) = (&blk[0..32], &blk[32..96], &blk[96..108]);
                let d = f16_at(blk, 108);
                let mut s = [0i32; 16];
                for i in 0..4 {
                    let (a, c, tmp) = (raw12[i] as i32, raw12[4 + i] as i32, raw12[8 + i] as i32);
                    s[i] = (a & 15) | ((tmp & 3) << 4);
                    s[4 + i] = (c & 15) | (((tmp >> 2) & 3) << 4);
                    s[8 + i] = (a >> 4) | (((tmp >> 4) & 3) << 4);
                    s[12 + i] = (c >> 4) | (((tmp >> 6) & 3) << 4);
                }
                let o = &mut out[256 * b..256 * b + 256];
                for h in 0..2 {
                    for j in 0..4 {
                        for l in 0..32 {
                            let e = 128 * h + 32 * j + l;
                            let low = ((qs[32 * h + l] >> (2 * j)) & 3) as i32;
                            let hb = (hm[l] >> (4 * h + j)) & 1;
                            let q = low - if hb == 1 { 0 } else { 4 };
                            o[e] = q as f32 * d * (s[e / 16] - 32) as f32;
                        }
                    }
                }
            }
        }
        GType::Q6K => {
            // ggml block_q6_K: ql[128], qh[64], scales[16] (int8), d fp16
            for (b, blk) in raw.chunks_exact(210).enumerate() {
                let (ql, qh, sc) = (&blk[0..128], &blk[128..192], &blk[192..208]);
                let d = f16_at(blk, 208);
                let o = &mut out[256 * b..256 * b + 256];
                for half in 0..2 {
                    let (ql, qh, sc, o) = (&ql[64 * half..], &qh[32 * half..], &sc[8 * half..], &mut o[128 * half..]);
                    for l in 0..32 {
                        let is = l / 16;
                        let q1 = ((ql[l] & 0xF) | ((qh[l] & 3) << 4)) as i32 - 32;
                        let q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32;
                        let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32;
                        let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32;
                        o[l] = d * (sc[is] as i8) as f32 * q1 as f32;
                        o[l + 32] = d * (sc[is + 2] as i8) as f32 * q2 as f32;
                        o[l + 64] = d * (sc[is + 4] as i8) as f32 * q3 as f32;
                        o[l + 96] = d * (sc[is + 6] as i8) as f32 * q4 as f32;
                    }
                }
            }
        }
        GType::I8 | GType::I32 => panic!("{t:?} is not a weight format"),
        GType::Fp8B32 | GType::Fp4 => panic!("{t:?} needs its scales: dequant_row_scaled"),
    }
}

const E2M1_LUT: [f32; 16] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0];

/// HF checkpoint rows: `scale` = this row's E8M0 bytes, one per 32 columns.
pub fn dequant_row_scaled(t: GType, raw: &[u8], scale: &[u8], out: &mut [f32]) {
    let e8m0 = |b: u8| f32::from_bits((b as u32) << 23);
    match t {
        GType::Fp8B32 => {
            for (i, o) in out.iter_mut().enumerate() {
                *o = crate::model::fp8_e4m3_to_f32(raw[i]) * e8m0(scale[i / 32]);
            }
        }
        GType::Fp4 => {
            for (i, o) in out.iter_mut().enumerate() {
                let b = raw[i / 2];
                let code = if i % 2 == 0 { b & 15 } else { b >> 4 };
                *o = E2M1_LUT[code as usize] * e8m0(scale[i / 32]);
            }
        }
        _ => dequant_row(t, raw, out),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn h(x: f32) -> [u8; 2] {
        f16::from_f32(x).to_le_bytes()
    }

    #[test]
    fn q2_0_layout() {
        // codes 0..3 -> -1..2, weight 4m+k in byte m bits 2k
        let mut blk = vec![0u8; 18];
        blk[..2].copy_from_slice(&h(0.5));
        blk[2] = 0b11_10_01_00; // weights 0..3 = -1, 0, 1, 2
        let mut out = vec![0f32; 64];
        dequant_row(GType::Q2_0, &blk, &mut out);
        assert_eq!(&out[..4], &[-0.5, 0.0, 0.5, 1.0]);
        assert_eq!(out[4], -0.5);
    }

    #[test]
    fn ptq1_0_roundtrip() {
        // pack like v41_formats.pack_ptq1_0 and decode back
        let codes: Vec<i32> = (0..128).map(|i| ((i * 7 + 3) % 3) as i32 - 1).collect();
        let byte = |digits: &[u32]| -> u8 {
            let mut v = 0u32;
            for &d in digits {
                v = v * 3 + d;
            }
            if digits.len() == 4 {
                v *= 3;
            }
            ((v * 256 + 242) / 243) as u8
        };
        let g: Vec<u32> = codes.iter().map(|&c| (c + 1) as u32).collect();
        let mut blk = vec![0u8; 28];
        for m in 0..16 {
            blk[m] = byte(&(0..5).map(|n| g[m + 16 * n]).collect::<Vec<_>>());
        }
        for m in 0..8 {
            blk[16 + m] = byte(&(0..5).map(|n| g[80 + m + 8 * n]).collect::<Vec<_>>());
        }
        for hh in 0..2 {
            blk[24 + hh] = byte(&(0..4).map(|n| g[120 + hh + 2 * n]).collect::<Vec<_>>());
        }
        blk[26..28].copy_from_slice(&h(2.0));
        let mut out = vec![0f32; 128];
        dequant_row(GType::Ptq1_0, &blk, &mut out);
        for i in 0..128 {
            assert_eq!(out[i], codes[i] as f32 * 2.0, "weight {i}");
        }
    }
}
