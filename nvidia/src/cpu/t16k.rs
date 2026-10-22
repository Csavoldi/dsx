//! T16 routed experts on the CPU (core/src/t16.rs, docs/t16.md "v6"). A matrix is `trellis | suh | svh` in the stored
//! layout: tiles by column (tile (kb, nb) at nb * k/16 + kb), each tile's bytes in bit-stream order.
//!
//! y = had128(had128(x * suh) @ W_inner) * svh: `prep` makes the int8 activations (x * suh, Hadamard, per 256
//! inputs a scale and the sum of codes), `gemv` gives the raw dot products of tile columns nb0..nb1 (each column's
//! 64-bit view from byte-aligned loads and vpshufb, 16 windows by shifts, vpmullw by 0x3A2F, vpdpbusd of the product
//! bytes [lo hi] with [x x 0 0]: x * byte sum b, then 0.009567260742 b - 2.4375 per group), `post` the output
//! Hadamard and svh. AVX-VNNI / AVX-512 VNNI; other CPUs take the scalar decode.

use super::qdot::{Isa, isa};
use crate::numerics::fwht_blocks;
use crate::t16;
use half::f16;

pub const C_B: f32 = 0.009567260742;
pub const C_0: f32 = 2.4375;

/// int8 activations of one matrix input: x4[i] = bytes [q, q, 0, 0] of input i, d / s per 256 inputs
#[derive(Default, Clone)]
pub struct Act {
    pub x4: Vec<i32>,
    pub d: Vec<f32>,
    pub s: Vec<i32>,
}

/// suh / svh of a stored matrix as f32
pub fn scales(m: &[u8], k2: usize, k: usize, n: usize) -> (&[u8], &[u8]) {
    let tb = t16::trellis_bytes(k2, k, n);
    (&m[tb..tb + 2 * k], &m[tb + 2 * k..tb + 2 * k + 2 * n])
}

#[inline]
fn h(b: &[u8], i: usize) -> f32 {
    f16::from_bits(u16::from_le_bytes([b[2 * i], b[2 * i + 1]])).to_f32()
}

/// one 256-input group of `prep`: x * suh, Hadamard per 128, int8 (round half to even, d = amax / 127)
pub fn prep_group(x: &[f32], suh: &[u8], gr: usize, a: &mut Act) {
    let mut v = [0f32; 256];
    for (i, vi) in v.iter_mut().enumerate() {
        *vi = x[256 * gr + i] * h(suh, 256 * gr + i);
    }
    fwht_blocks(&mut v);
    let m = v.iter().fold(0f32, |m, x| m.max(x.abs()));
    let id = if m > 0.0 { 127.0 / m } else { 0.0 };
    let mut sum = 0i32;
    for (i, &vi) in v.iter().enumerate() {
        let q = (vi * id).round_ties_even() as i32;
        sum += q;
        let b = (q & 255) as u32;
        a.x4[256 * gr + i] = (b | b << 8) as i32;
    }
    a.d[gr] = m / 127.0;
    a.s[gr] = sum;
}

pub fn prep(x: &[f32], suh: &[u8], k: usize, a: &mut Act) {
    a.x4.resize(k, 0);
    a.d.resize(k / 256, 0.0);
    a.s.resize(k / 256, 0);
    for gr in 0..k / 256 {
        prep_group(x, suh, gr, a);
    }
}

/// y (n raw outputs, the slice of `nb0 * 16 ..`) -> had128(y) * svh, in place, for one 128-block at `o`
pub fn post_block(y: &mut [f32], svh: &[u8], o: usize) {
    fwht_blocks(&mut y[..128]);
    for (i, v) in y[..128].iter_mut().enumerate() {
        *v *= h(svh, o + i);
    }
}

fn hend(k2: usize, t: usize) -> usize {
    if k2 % 2 == 1 { (t + 1) * (k2 / 2) + (t + 1) / 2 } else { (t + 1) * (k2 / 2) }
}

/// raw outputs of tile columns nb0..nb1 into out[(nb - nb0) * 16 ..]: per 256-group d (C_B acc - C_0 s)
pub fn gemv(m: &[u8], k2: usize, k: usize, n: usize, a: &Act, nb0: usize, nb1: usize, out: &mut [f32]) {
    match isa() {
        #[cfg(target_arch = "x86_64")]
        Isa::Vnni => unsafe { vnni::gemv(m, k2, k, n, a, nb0, nb1, out) },
        #[cfg(target_arch = "x86_64")]
        Isa::Vnni512 => unsafe { vnni512::gemv(m, k2, k, n, a, nb0, nb1, out) },
        _ => gemv_scalar(m, k2, k, n, a, nb0, nb1, out),
    }
}

/// the same with the scalar decode (reference and fallback)
pub fn gemv_scalar(m: &[u8], k2: usize, k: usize, _n: usize, a: &Act, nb0: usize, nb1: usize, out: &mut [f32]) {
    let (kt, tb) = (k / 16, t16::tile_bytes(k2));
    for nb in nb0..nb1 {
        for c in 0..16 {
            let mut f = 0f32;
            for gr in 0..k / 256 {
                let mut acc = 0i64;
                for kb in 16 * gr..16 * gr + 16 {
                    let tile = &m[(nb * kt + kb) * tb..][..tb];
                    for r in 0..16 {
                        let p = t16::state(tile, k2, 16 * c + r).wrapping_mul(t16::MUL) & 0xFFFF;
                        let b = (p & 0xFF) + (p >> 8);
                        acc += b as i64 * (a.x4[16 * kb + r] as i8) as i64;
                    }
                }
                f += a.d[gr] * (C_B * acc as f32 - C_0 * a.s[gr] as f32);
            }
            out[(nb - nb0) * 16 + c] = f;
        }
    }
    let _ = hend;
}

/// per K2: for each of the 4 views (lo cols 0-7, lo 8-15, hi 0-7, hi 8-15) and 128-bit half (4 columns): one or two
/// 16-byte loads (stream byte offsets, may be negative = wrap to the tile end) and their vpshufb masks
#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy)]
struct Plan {
    src: [[[i32; 2]; 2]; 4],
    mask: [[[[i8; 16]; 2]; 2]; 4],
    nsrc: [[u8; 2]; 4],
    tb: i32,
}

#[cfg(target_arch = "x86_64")]
fn plan(k2: usize) -> Plan {
    let mut p = Plan { src: [[[0; 2]; 2]; 4], mask: [[[[-1; 16]; 2]; 2]; 4], nsrc: [[0; 2]; 4], tb: (16 * k2) as i32 };
    for v in 0..4 {
        for hh in 0..2 {
            let mut off = [0i32; 4];
            for (i, o) in off.iter_mut().enumerate() {
                let c = 8 * (v & 1) + 4 * hh + i;
                let e = (hend(k2, 16 * c + 15) / 8) as i32;
                *o = if v < 2 { e - 4 } else { e - 8 };
            }
            let one = off[3] + 4 - off[0] <= 16;
            p.nsrc[v][hh] = if one { 1 } else { 2 };
            for part in 0..p.nsrc[v][hh] as usize {
                let base = if one { off[0] } else { off[2 * part] };
                p.src[v][hh][part] = base;
                for i in 0..4 {
                    let mine = one || i / 2 == part;
                    for b in 0..4 {
                        p.mask[v][hh][part][4 * i + b] = if mine { (off[i] - base + 3 - b as i32) as i8 } else { -1 };
                    }
                }
            }
        }
    }
    p
}

#[cfg(target_arch = "x86_64")]
fn plans() -> &'static [Plan; 3] {
    static P: std::sync::OnceLock<[Plan; 3]> = std::sync::OnceLock::new();
    P.get_or_init(|| [plan(3), plan(4), plan(5)])
}

// window macros of the SIMD kernels (outside t16_kernels so their $-variables are their own); `dp` and `view` resolve
// at the expansion site
macro_rules! win {
    ($acc:ident, $l:ident, $h:ident, $xj:ident, $mul:ident, lo $o:literal) => {
        $acc = dp($acc, _mm256_mullo_epi16(_mm256_or_si256(_mm256_srli_epi32::<$o>($l), _mm256_slli_epi32::<{ 32 - $o }>($h)), $mul), $xj);
    };
    ($acc:ident, $l:ident, $h:ident, $xj:ident, $mul:ident, zero) => {
        $acc = dp($acc, _mm256_mullo_epi16($l, $mul), $xj);
    };
    ($acc:ident, $l:ident, $h:ident, $xj:ident, $mul:ident, hi $o:literal) => {
        $acc = dp($acc, _mm256_mullo_epi16(_mm256_srli_epi32::<$o>($h), $mul), $xj);
    };
}

macro_rules! tile {
    ($t:ident, $p:ident, $msk:ident, $x:ident, $kb:ident, $a0:ident, $a1:ident, $mul:ident; $($j:literal: $kind:ident $($o:literal)?),*) => {{
        let l0 = view($t, $p, &$msk, 0);
        let l1 = view($t, $p, &$msk, 1);
        let h0 = view($t, $p, &$msk, 2);
        let h1 = view($t, $p, &$msk, 3);
        $(
            let xj = _mm256_set1_epi32(*$x.get_unchecked(16 * $kb + $j));
            win!($a0, l0, h0, xj, $mul, $kind $($o)?);
            win!($a1, l1, h1, xj, $mul, $kind $($o)?);
        )*
    }};
}


macro_rules! t16_kernels {
    ($m:ident, $feat:literal, |$s:ident, $a:ident, $b:ident| $dp:expr) => {
        #[cfg(target_arch = "x86_64")]
        mod $m {
            use super::*;
            use std::arch::x86_64::*;

            #[inline]
            #[target_feature(enable = $feat)]
            unsafe fn dp($s: __m256i, $a: __m256i, $b: __m256i) -> __m256i {
                $dp
            }

            /// 16 bytes of the tile's stream from byte s (negative: wraps to the tile end)
            #[inline]
            #[target_feature(enable = $feat)]
            unsafe fn load16w(t: *const u8, s: i32, tb: i32) -> __m128i {
                unsafe {
                    if s >= 0 {
                        return _mm_loadu_si128(t.add(s as usize) as *const __m128i);
                    }
                    let head = _mm_loadu_si128(t as *const __m128i);
                    let tail = _mm_loadu_si128(t.add((tb - 16) as usize) as *const __m128i);
                    match -s {
                        1 => _mm_alignr_epi8::<15>(head, tail),
                        2 => _mm_alignr_epi8::<14>(head, tail),
                        3 => _mm_alignr_epi8::<13>(head, tail),
                        4 => _mm_alignr_epi8::<12>(head, tail),
                        5 => _mm_alignr_epi8::<11>(head, tail),
                        6 => _mm_alignr_epi8::<10>(head, tail),
                        7 => _mm_alignr_epi8::<9>(head, tail),
                        8 => _mm_alignr_epi8::<8>(head, tail),
                        9 => _mm_alignr_epi8::<7>(head, tail),
                        10 => _mm_alignr_epi8::<6>(head, tail),
                        11 => _mm_alignr_epi8::<5>(head, tail),
                        12 => _mm_alignr_epi8::<4>(head, tail),
                        _ => head,
                    }
                }
            }

            #[inline]
            #[target_feature(enable = $feat)]
            unsafe fn view(t: *const u8, p: &Plan, msk: &[[[__m128i; 2]; 2]; 4], v: usize) -> __m256i {
                unsafe {
                    let mut r = [_mm_setzero_si128(); 2];
                    for hh in 0..2 {
                        r[hh] = _mm_shuffle_epi8(load16w(t, p.src[v][hh][0], p.tb), msk[v][hh][0]);
                        if p.nsrc[v][hh] == 2 {
                            r[hh] = _mm_or_si128(r[hh], _mm_shuffle_epi8(load16w(t, p.src[v][hh][1], p.tb), msk[v][hh][1]));
                        }
                    }
                    _mm256_set_m128i(r[1], r[0])
                }
            }

            #[target_feature(enable = $feat)]
            pub(super) unsafe fn gemv(m: &[u8], k2: usize, k: usize, n: usize, a: &Act, nb0: usize, nb1: usize, out: &mut [f32]) {
                unsafe {
                    assert!(m.len() >= t16::trellis_bytes(k2, k, n) + 16 && nb1 <= n / 16 && a.x4.len() >= k);
                    let p = &plans()[k2 - 3];
                    let mut msk = [[[_mm_setzero_si128(); 2]; 2]; 4];
                    for v in 0..4 {
                        for hh in 0..2 {
                            for q in 0..2 {
                                msk[v][hh][q] = _mm_loadu_si128(p.mask[v][hh][q].as_ptr() as *const __m128i);
                            }
                        }
                    }
                    let (kt, tb) = (k / 16, 16 * k2);
                    let mul = _mm256_set1_epi16(t16::MUL as i16);
                    let x = &a.x4[..];
                    for nb in nb0..nb1 {
                        let (mut f0, mut f1) = (_mm256_setzero_ps(), _mm256_setzero_ps());
                        let col = m.as_ptr().add(nb * kt * tb);
                        for gr in 0..k / 256 {
                            let (mut a0, mut a1) = (_mm256_setzero_si256(), _mm256_setzero_si256());
                            for kb in 16 * gr..16 * gr + 16 {
                                let t = col.add(kb * tb);
                                match k2 {
                                    3 => tile!(t, p, msk, x, kb, a0, a1, mul; 0: lo 23, 1: lo 21, 2: lo 20, 3: lo 18, 4: lo 17, 5: lo 15, 6: lo 14, 7: lo 12, 8: lo 11, 9: lo 9, 10: lo 8, 11: lo 6, 12: lo 5, 13: lo 3, 14: lo 2, 15: zero),
                                    4 => tile!(t, p, msk, x, kb, a0, a1, mul; 0: lo 30, 1: lo 28, 2: lo 26, 3: lo 24, 4: lo 22, 5: lo 20, 6: lo 18, 7: lo 16, 8: lo 14, 9: lo 12, 10: lo 10, 11: lo 8, 12: lo 6, 13: lo 4, 14: lo 2, 15: zero),
                                    _ => tile!(t, p, msk, x, kb, a0, a1, mul; 0: hi 6, 1: hi 3, 2: hi 1, 3: lo 30, 4: lo 28, 5: lo 25, 6: lo 23, 7: lo 20, 8: lo 18, 9: lo 15, 10: lo 13, 11: lo 10, 12: lo 8, 13: lo 5, 14: lo 3, 15: zero),
                                }
                            }
                            let d = _mm256_set1_ps(a.d[gr]);
                            let c0 = _mm256_set1_ps(C_0 * a.s[gr] as f32);
                            let cb = _mm256_set1_ps(C_B);
                            f0 = _mm256_fmadd_ps(d, _mm256_fmsub_ps(cb, _mm256_cvtepi32_ps(a0), c0), f0);
                            f1 = _mm256_fmadd_ps(d, _mm256_fmsub_ps(cb, _mm256_cvtepi32_ps(a1), c0), f1);
                        }
                        let o = out.as_mut_ptr().add((nb - nb0) * 16);
                        _mm256_storeu_ps(o, f0);
                        _mm256_storeu_ps(o.add(8), f1);
                    }
                }
            }
        }
    };
}

t16_kernels!(vnni, "avx2,avxvnni,fma,f16c,ssse3", |s, a, b| _mm256_dpbusd_avx_epi32(s, a, b));
t16_kernels!(vnni512, "avx2,avx512vnni,avx512vl,fma,f16c,ssse3", |s, a, b| _mm256_dpbusd_epi32(s, a, b));

#[cfg(test)]
mod tests {
    use super::*;

    fn rng(s: &mut u64) -> u64 {
        *s ^= *s << 13;
        *s ^= *s >> 7;
        *s ^= *s << 17;
        *s
    }

    #[test]
    fn simd_matches_scalar_and_reference() {
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        for (k2, k, n) in [(3usize, 512usize, 256usize), (4, 768, 128), (5, 256, 384)] {
            let ty = t16::ty_for_k2(k2).unwrap();
            let mut m: Vec<u8> = (0..t16::matrix_bytes(ty, k, n)).map(|_| rng(&mut seed) as u8).collect();
            let tb = t16::trellis_bytes(k2, k, n);
            for c in m[tb..].chunks_exact_mut(2) {
                let v = 0.02 * (0.5 + (rng(&mut seed) % 1000) as f32 / 1000.0) * if rng(&mut seed) % 2 == 0 { 1.0 } else { -1.0 };
                c.copy_from_slice(&f16::from_f32(v).to_le_bytes());
            }
            let x: Vec<f32> = (0..k).map(|_| (rng(&mut seed) % 2001) as f32 / 1000.0 - 1.0).collect();
            let (suh, svh) = scales(&m, k2, k, n);
            let mut a = Act::default();
            prep(&x, suh, k, &mut a);
            let (mut ys, mut yv) = (vec![0f32; n], vec![0f32; n]);
            gemv_scalar(&m, k2, k, n, &a, 0, n / 16, &mut ys);
            gemv(&m, k2, k, n, &a, 0, n / 16, &mut yv);
            let rms = (ys.iter().map(|v| v * v).sum::<f32>() / n as f32).sqrt();
            let e = ys.iter().zip(&yv).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max) / rms;
            assert!(e < 1e-5, "k2 {k2}: SIMD vs scalar {e} ({:?})", isa());
            for o in (0..n).step_by(128) {
                post_block(&mut yv[o..], svh, o);
            }
            let want = t16::forward(&m, ty, k, n, &x);
            let wr = (want.iter().map(|v| v * v).sum::<f32>() / n as f32).sqrt();
            let er = (want.iter().zip(&yv).map(|(a, b)| (a - b).powi(2)).sum::<f32>() / n as f32).sqrt() / wr;
            println!("k2 {k2}: SIMD = scalar ({e:.1e}); vs f32 reference rms {er:.2e} (int8 activations), {:?}", isa());
            assert!(er < 0.02, "k2 {k2}: vs reference {er}");
        }
    }
}
