#![cfg(feature = "cuda")]
//! 6-bit dense planes: GPU GEMV vs the CPU decoder, and speed next to the repacked Q8 GEMV.
use dsx::gpu::*;
use std::time::Instant;

#[test]
fn q6_gemv() {
    let g = Gpu::new(0).unwrap();
    for (rows, k) in [(256usize, 1280usize), (32768, 1280), (5120, 8192)] {
        let x: Vec<f32> = (0..rows * k).map(|i| ((i * 2654435761usize) % 2001) as f32 / 1000.0 - 1.0).collect();
        let (mut lo, mut hi, mut sc, mut d) = (vec![0u8; rows * k / 2], vec![0u8; rows * k / 4], vec![0i8; rows * k / 16], vec![0u16; rows * k / 256]);
        for r in 0..rows {
            dsx::q6::quantize_row(&x[r * k..(r + 1) * k], &mut lo[r * k / 2..(r + 1) * k / 2], &mut hi[r * k / 4..(r + 1) * k / 4], &mut sc[r * k / 16..(r + 1) * k / 16], &mut d[r * k / 256..(r + 1) * k / 256]);
        }
        let v: Vec<f32> = (0..k).map(|i| (i % 13) as f32 * 0.1 - 0.6).collect();
        let (dl, dh, ds, dd, dv, dy) = (g.upload(&lo).unwrap(), g.upload(&hi).unwrap(), g.upload(&sc).unwrap(), g.upload(&d).unwrap(), g.upload(&v).unwrap(), g.alloc(rows * 4).unwrap());
        let kk = g.kernel("gemv_q6p").unwrap();
        let args = [Arg::P(dl.ptr), Arg::P(dv.ptr), Arg::P(dy.ptr), Arg::I(rows as i32), Arg::I(k as i32), Arg::I(1), Arg::I(k as i32), Arg::I(rows as i32), Arg::P(dh.ptr), Arg::P(ds.ptr), Arg::P(dd.ptr)];
        g.launch(kk, ((rows as u32).div_ceil(4), 1, 1), 128, 0, &args).unwrap();
        let got: Vec<f32> = g.download(dy.ptr, rows).unwrap();
        for r in (0..rows).step_by((rows / 8).max(1)) {
            let want: f64 = (0..k).map(|c| dsx::q6::dequant(&lo, &hi, &sc, &d, r * k + c) as f64 * v[c] as f64).sum();
            assert!((got[r] as f64 - want).abs() < 1e-3 * want.abs().max(1.0), "row {r}: {} vs {want}", got[r]);
        }
        let t = Instant::now();
        for _ in 0..50 { g.launch(kk, ((rows as u32).div_ceil(4), 1, 1), 128, 0, &args).unwrap(); }
        g.sync().unwrap();
        let us = t.elapsed().as_secs_f64() / 50.0 * 1e6;
        eprintln!("q6p {rows}x{k}: {us:.1} us, {:.0} GB/s ({:.0} G weights/s)", rows as f64 * k as f64 * 0.8203 / us / 1e3, (rows * k) as f64 / us / 1e3);
    }
}
