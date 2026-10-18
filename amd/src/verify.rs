//! Speculative verify: the main model over a few tokens at once (the newest token + drafts), with decode-style
//! weight streaming (kernels/verify.h) and the prompt path's per-token kernels; state is committed for the
//! accepted prefix only.

use crate::decode::*;
use crate::hip::Arg;
use anyhow::Result;

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct VJob {
    pub w: u64,
    pub sc: u64,
    pub woff: i64,
    pub soff: i64,
    pub mul: f32,
    pub fmt: i32,
    pub n: i32,
    pub k: i32,
    pub split: i32,
    pub pad0: i32,
    pub xq: u64,
    pub xs: u64,
    pub xb: u64,
    pub ldx: i32,
    pub ntok: i32,
    pub rows: u64,
    pub y: u64,
    pub orows: u64,
    pub ldy: i32,
    pub atomic: i32,
    pub blocks: i32,
    pub epi: i32,
    pub ropec: i32,
    pub ndyn: i32,
    pub dyn_: u64,
}

const VMAX: usize = 12;

#[repr(C)]
#[derive(Clone, Copy)]
struct VJobs {
    j: [VJob; VMAX],
    n: i32,
    cb3lo: u32,
    cb3hi: u32,
    cb2: u32,
}

impl Decoder {
    /// one vgemv launch; split 0 = chosen so the launch has ~2048 waves
    pub(crate) fn vlaunch(&self, label: &str, mut jobs: Vec<VJob>, pos0: u64) -> Result<()> {
        for chunk in jobs.chunks_mut(VMAX) {
            let tiles: usize = chunk.iter().map(|j| (j.n as usize).div_ceil(16) * j.ndyn.max(1) as usize).sum();
            let mut blocks = 0u32;
            for j in chunk.iter_mut() {
                if j.split == 0 {
                    let ku = match j.fmt { 0 => 64, 4 => 32, _ => 128 };
                    let units = j.k as usize / ku;
                    let mut s = 1;
                    while tiles * s < 2048 && s * 2 <= units / 2 {
                        s *= 2;
                    }
                    j.split = s as i32;
                }
                if j.split > 1 {
                    j.atomic = 1;
                }
                j.blocks = ((j.n as usize).div_ceil(64) * j.split as usize * j.ndyn.max(1) as usize) as i32;
                blocks += j.blocks as u32;
            }
            let n = chunk.len();
            let mut arr = [chunk[0]; VMAX];
            arr[..n].copy_from_slice(chunk);
            let js = VJobs { j: arr, n: n as i32, cb3lo: self.cb3[0], cb3hi: self.cb3[1], cb2: self.cb2 };
            let b = unsafe { std::slice::from_raw_parts(&js as *const VJobs as *const u8, std::mem::size_of::<VJobs>()).to_vec() };
            self.g.set_label(Some(label));
            let r = self.g.launch("vgemv", (blocks, 1, 1), 256, 0, &[Arg::B(b), Arg::P(pos0), Arg::P(self.rope_f.ptr), Arg::P(self.rope_fc.ptr)]);
            self.g.set_label(None);
            r?;
        }
        Ok(())
    }
}
