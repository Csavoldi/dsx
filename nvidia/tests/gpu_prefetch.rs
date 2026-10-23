#![cfg(feature = "cuda")]
//! DSX_PREFETCH's kernels (t16.cuh pf_*): an expert copied from a mapped host arena into a slot and handed to a store
//! for one layer runs bit for bit like the same expert resident in the store; the take mask names its pick; after
//! pf_reset the store runs as without it. Also: a slot whose copy has not landed, or whose expert the store already
//! holds, is not taken.
use dsx::experts::{ExpCfg, GpuExperts, RawExpert};
use dsx::gguf::GType;
use dsx::gpu::*;
use dsx::t16;

fn rng(seed: &mut u64) -> u64 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    *seed
}

fn random_matrix(ty: GType, k: usize, n: usize, seed: &mut u64) -> Vec<u8> {
    let k2 = t16::k2(ty).unwrap();
    let tb = t16::trellis_bytes(k2, k, n);
    let mut v: Vec<u8> = (0..t16::matrix_bytes(ty, k, n)).map(|_| rng(seed) as u8).collect();
    for c in v[tb..].chunks_exact_mut(2) {
        let s = 0.02 * (0.5 + (rng(seed) % 1000) as f32 / 1000.0) * if rng(seed) % 2 == 0 { 1.0 } else { -1.0 };
        c.copy_from_slice(&half::f16::from_f32(s).to_le_bytes());
    }
    v
}

#[test]
fn prefetched_expert_runs_like_a_resident_one() {
    let g = Gpu::new(0).unwrap();
    let mut seed = 0x5eed_0f_9a7e_1234u64;
    let c = ExpCfg { n_layers: 2, n_experts: 12, inter: 2304, dim: 5120, lim: 10.0, rot: 0 };
    let (gu, dn) = (GType::T16K15, GType::T16K2);
    let (kg, kd) = (t16::k2(gu).unwrap(), t16::k2(dn).unwrap());
    let mats: Vec<(Vec<u8>, Vec<u8>, Vec<u8>)> = (0..10)
        .map(|_| (random_matrix(gu, c.dim, c.inter, &mut seed), random_matrix(gu, c.dim, c.inter, &mut seed), random_matrix(dn, c.inter, c.dim, &mut seed)))
        .collect();
    let raw = |e: usize| RawExpert { gate: &mats[e].0, up: &mats[e].1, down: &mats[e].2, gu, dn };
    // layer 1: experts 0..8 resident; expert 9 is the prefetched one (resident in the reference store only)
    let st = GpuExperts::build(&g, c, &(0..9).map(|e| ((1, e), raw(e))).collect::<Vec<_>>(), true).unwrap();
    let rf = GpuExperts::build(&g, c, &(0..10).map(|e| ((1, e), raw(e))).collect::<Vec<_>>(), true).unwrap();
    let x_e = 9usize;
    let (gb, db) = (mats[x_e].0.len(), mats[x_e].2.len());
    assert!(gb % 64 == 0 && db % 64 == 0);

    // the arena: expert 9's raw matrices in mapped host memory
    let mut arena = HostBuf::new(2 * gb + db).unwrap();
    {
        let a = arena.as_mut_slice::<u8>();
        a[..gb].copy_from_slice(&mats[x_e].0);
        a[gb..2 * gb].copy_from_slice(&mats[x_e].1);
        a[2 * gb..2 * gb + db].copy_from_slice(&mats[x_e].2);
    }
    let src_dev = [arena.dev_ptr_at(&g, 0).unwrap(), arena.dev_ptr_at(&g, gb).unwrap(), arena.dev_ptr_at(&g, 2 * gb).unwrap()];
    let ne = c.n_experts;
    let mut src = vec![0u64; ne * 3];
    src[x_e * 3..x_e * 3 + 3].copy_from_slice(&src_dev);
    src[3 * 3..3 * 3 + 3].copy_from_slice(&src_dev); // expert 3 (resident) "in the arena" too: must never be taken
    let src = g.upload(&src).unwrap();
    let mut bits = vec![0u8; ne];
    bits[x_e] = 1;
    bits[3] = 1; // stale bit for a resident expert: pf_take still refuses it (its table entry is not empty)
    let bits = g.upload(&bits).unwrap();
    let m = 2usize;
    let pred = g.upload(&[11i32, 3, x_e as i32, 0, 1, 2, 4, 5]).unwrap(); // 11: not in the arena
    let desc = g.upload(&vec![-1i64; m * 4]).unwrap();
    let ready = g.alloc_zeroed(m * 4).unwrap();
    let taken = g.upload(&vec![-1i32; m]).unwrap();
    let stage = g.alloc(2 * gb + db).unwrap();
    let slot = g.alloc(m * (2 * gb + db)).unwrap();
    let slots: Vec<[u64; 3]> = (0..m).map(|i| {
        let b = slot.ptr + (i * (2 * gb + db)) as u64;
        [b, b + gb as u64, b + 2 * gb as u64]
    }).collect();
    let rel: Vec<u64> = slots.iter().flat_map(|s| s.map(|p| p.wrapping_sub(st.buf_ptr()))).collect();
    let rel = g.upload(&rel).unwrap();
    let mask = HostBuf::new(64).unwrap();
    let mask_dev = mask.dev_ptr(&g).unwrap();
    let k = |n: &str| g.kernel(n).unwrap();
    let id = |t: GType| dsx::experts::slot_id(t, true) as i32;

    // pick: 3 is resident-but-flagged, 9 the real one; 11 has no arena copy
    g.launch(k("pf_pick"), (1, 1, 1), 32, 0, &[Arg::P(pred.ptr), Arg::P(bits.ptr), Arg::P(src.ptr), Arg::I(ne as i32), Arg::I(m as i32), Arg::P(desc.ptr)]).unwrap();
    let d: Vec<i64> = g.download(desc.ptr, m * 4).unwrap();
    assert_eq!((d[0], d[4]), (3, x_e as i64), "pf_pick: the first m CPU-only experts with arena copies, in score order");
    for i in 0..m {
        let di = desc.ptr + (i * 32) as u64;
        g.launch(k("pf_copy"), (16, 3, 1), 256, 0, &[Arg::P(di), Arg::P(stage.ptr), Arg::P(stage.ptr + gb as u64), Arg::P(stage.ptr + 2 * gb as u64), Arg::I(gb as i32), Arg::I(gb as i32), Arg::I(db as i32)]).unwrap();
        let sl = slots[i];
        g.launch(k("repack_t16_if"), ((gb.max(db) as u32 / 4).div_ceil(256), 3, 1), 256, 0, &[Arg::P(di), Arg::I(kg as i32), Arg::I(kd as i32), Arg::P(stage.ptr), Arg::P(sl[0]), Arg::P(sl[1]), Arg::P(sl[2]), Arg::I(c.dim as i32), Arg::I(c.inter as i32), Arg::I(gb as i32)]).unwrap();
    }
    g.sync().unwrap();

    let kk = 6;
    let sel: Vec<i32> = vec![2, x_e as i32, 0, 3, 5, 11];
    let wts: Vec<f32> = (0..kk).map(|_| 0.05 + (rng(&mut seed) % 100) as f32 * 0.003).collect();
    let x: Vec<f32> = (0..c.dim).map(|_| (rng(&mut seed) % 2001) as f32 / 1000.0 - 1.0).collect();
    for s in [&st, &rf] {
        g.htod_sync(s.sel.ptr, &sel).unwrap();
        g.htod_sync(s.wts.ptr, &wts).unwrap();
        g.htod_sync(s.x.ptr, &x).unwrap();
    }
    let run = |s: &GpuExperts| -> Vec<f32> {
        s.run_device_fast(&g, 1, s.sel.ptr, s.wts.ptr, kk, 1, s.x.ptr, s.out.ptr).unwrap();
        g.download(s.out.ptr, c.dim).unwrap()
    };
    let take = |want: u32| {
        g.launch(k("pf_take"), (1, 1, 1), 32, 0, &[Arg::P(st.sel.ptr), Arg::I(kk as i32), Arg::P(desc.ptr), Arg::P(ready.ptr), Arg::I(want as i32), Arg::I(m as i32), Arg::P(st.tab_ptr() + (ne * 32) as u64), Arg::P(rel.ptr), Arg::I(id(gu)), Arg::I(id(dn)), Arg::P(mask_dev), Arg::P(taken.ptr)]).unwrap();
        g.sync().unwrap();
        mask.as_slice::<u32>()[0]
    };
    let without = run(&st);
    let want = run(&rf);

    // a copy that has not landed (ready < want) is not taken: the store runs as without expert 9
    assert_eq!(take(7), 0, "nothing ready: empty take mask");
    assert_eq!(run(&st), without);

    // landed: expert 9 (pick 1) runs from its slot like the resident one, bit for bit; expert 3 is not taken
    g.write_value(ready.ptr, 7).unwrap();
    g.write_value(ready.ptr + 4, 7).unwrap();
    assert_eq!(take(7), 1 << 1, "take mask: pick 1 (expert 9) only");
    let got = run(&st);
    assert!(got.iter().zip(&want).all(|(a, b)| a.to_bits() == b.to_bits()), "prefetched expert vs resident: not bit-identical");
    assert!(got != without, "the prefetched expert contributed nothing");
    let tk: Vec<i32> = g.download(taken.ptr, m).unwrap();
    assert_eq!(tk, vec![-1, x_e as i32]);

    // after pf_reset the store runs as without expert 9 again
    g.launch(k("pf_reset"), (1, 1, 1), 32, 0, &[Arg::P(taken.ptr), Arg::I(m as i32), Arg::P(st.tab_ptr() + (ne * 32) as u64)]).unwrap();
    assert_eq!(run(&st), without, "after pf_reset");
    println!("prefetch: slot expert bit-identical to resident; mask {:#b}", 1 << 1);
}

/// pf_copy's zero-copy read rate from mapped host memory by grid size, next to a DMA copy of the same bytes
/// (cargo test --release --features cuda --test gpu_prefetch -- --ignored --nocapture)
#[test]
#[ignore]
fn pf_copy_speed() {
    let g = Gpu::new(0).unwrap();
    let (gb, db) = (4_423_680usize, 2_957_312usize); // a K 2 / 2.5 expert's matrices, rounded to 64 bytes
    let total = 2 * gb + db;
    let mut arena = HostBuf::new(total).unwrap();
    arena.as_mut_slice::<u8>().iter_mut().enumerate().for_each(|(i, b)| *b = i as u8);
    let src = [arena.dev_ptr_at(&g, 0).unwrap(), arena.dev_ptr_at(&g, gb).unwrap(), arena.dev_ptr_at(&g, 2 * gb).unwrap()];
    let desc = g.upload(&[0i64, src[0] as i64, src[1] as i64, src[2] as i64]).unwrap();
    let stage = g.alloc(total).unwrap();
    let k = g.kernel("pf_copy").unwrap();
    let (e0, e1) = (g.timed_event().unwrap(), g.timed_event().unwrap());
    for blocks in [4u32, 8, 16, 32, 64] {
        let mut best = f32::MAX;
        for _ in 0..5 {
            g.record(e0, false).unwrap();
            g.launch(k, (blocks, 3, 1), 256, 0, &[Arg::P(desc.ptr), Arg::P(stage.ptr), Arg::P(stage.ptr + gb as u64), Arg::P(stage.ptr + 2 * gb as u64), Arg::I(gb as i32), Arg::I(gb as i32), Arg::I(db as i32)]).unwrap();
            g.record(e1, false).unwrap();
            g.sync().unwrap();
            best = best.min(g.elapsed_ms(e0, e1).unwrap());
        }
        println!("pf_copy {blocks:3} blocks x 3: {:.3} ms, {:.1} GB/s", best, total as f64 / best as f64 / 1e6);
    }
    let mut best = f32::MAX;
    for _ in 0..5 {
        g.record(e0, false).unwrap();
        g.htod(stage.ptr, &arena.as_slice::<u8>()[..total]).unwrap();
        g.record(e1, false).unwrap();
        g.sync().unwrap();
        best = best.min(g.elapsed_ms(e0, e1).unwrap());
    }
    println!("DMA copy: {:.3} ms, {:.1} GB/s", best, total as f64 / best as f64 / 1e6);
}
