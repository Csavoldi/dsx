//! The decoder: weights in HBM, per-layer state, and one decode step recorded as a HIP graph.

use crate::hip::{Arg, Buf, Gpu, Graph};
use crate::load::{self, DMat};
use anyhow::{Context, Result, bail};
use dsx::model::Config;
use dsx::safetensors::SafeTensors;
use std::path::Path;

pub const D: usize = 5120;
pub const HC: usize = 4;
pub const NH: usize = 64;
pub const HD: usize = 512;
pub const QL: usize = 1280;
pub const NE: usize = 384;
pub const INTER: usize = 2304;
pub const WIN: usize = 128;
pub const IH: usize = 32;
pub const ID: usize = 128;
pub const VOCAB: usize = 129280;
const ACH: usize = 32;
const NCHUNK: usize = (WIN + 512).div_ceil(ACH);
const CAND_CAP: usize = 65536;

// per-layer accumulator offsets (f32), all zeroed at the start of a step
const A_HMIX_A: usize = 0;
const A_HMIX_F: usize = 32;
const A_XSS_A: usize = 64;
const A_XSS_F: usize = 68;
const A_ENG_SUM: usize = 72;
const A_QR: usize = 128;
const A_KV: usize = A_QR + 1280;
const A_CKV: usize = A_KV + 512;
const A_CGATE: usize = A_CKV + 512;
const A_IW: usize = A_CGATE + 512;
const A_R: usize = 3072;
const A_A: usize = A_R + 8192;
const A_LOGITS: usize = A_A + 5120;
const A_SHG: usize = A_LOGITS + 384;
const A_SHU: usize = A_SHG + 2304;
const A_G: usize = A_SHU + 2304;
const A_U: usize = A_G + 6 * 2304;
const A_Y: usize = A_U + 6 * 2304;
const A_ENG_KV: usize = A_Y + 5120;
const ACC: usize = A_ENG_KV + 25600;

// ---- kernel argument structs (mirror kernels/dsxa.hip)

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct HcArgs {
    pub scale: [f32; 3],
    pub base: [f32; 24],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Job {
    w: u64,
    sc: u64,
    y: u64,
    x: u64,
    x2: u64,
    nw: u64,
    ss: u64,
    n: i32,
    k: i32,
    split: i32,
    wkind: i32,
    xkind: i32,
    blocks: i32,
    xgroup: i32,
    eslot: i32,
    wstride: i64,
    sstride: i64,
    lo: i32,
    epi: i32,
    ropec: i32,
    emat: i32,
}

const MAXJ: usize = 13;

#[repr(C)]
#[derive(Clone, Copy)]
struct Jobs {
    j: [Job; MAXJ],
    n: i32,
    eps: f32,
    lim: f32,
    cb3: [u32; 2],
    cb2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct ExpertMat {
    pub woff: i64,
    pub soff: i64,
    pub mul: f32,
    pub fmt: i32,
    pad: [i32; 2],
}

pub const W_FP8: i32 = 0;
pub const W_BF16: i32 = 1;
pub const W_EXPERT: i32 = 3;
pub const X_NORM: i32 = 0;
pub const X_RMS: i32 = 1;
pub const X_PLAIN: i32 = 2;
pub const X_BF16R: i32 = 3;
pub const X_SWIGLU: i32 = 4;
pub const X_ENGRAM: i32 = 5;
pub const X_Q: i32 = 6;
pub const E_FP4: i32 = 0;
pub const E_CB3: i32 = 1;
pub const E_CB2: i32 = 2;

fn bytes_of<T: Copy>(v: &T) -> Vec<u8> {
    unsafe { std::slice::from_raw_parts(v as *const T as *const u8, std::mem::size_of::<T>()).to_vec() }
}

impl Job {
    fn new(wkind: i32, w: u64, sc: u64, n: usize, k: usize, y: u64, xkind: i32, x: u64) -> Job {
        Job {
            w,
            sc,
            y,
            x,
            x2: 0,
            nw: 0,
            ss: 0,
            n: n as i32,
            k: k as i32,
            split: 1,
            wkind,
            xkind,
            blocks: 0,
            xgroup: 0,
            eslot: -1,
            wstride: 0,
            sstride: 0,
            lo: 0,
            epi: 0,
            ropec: 0,
            emat: 0,
        }
    }
    fn dense(m: &DMat, wkind: i32, y: u64, xkind: i32, x: u64) -> Job {
        Job::new(wkind, m.w.ptr, m.sc.as_ref().map(|b| b.ptr).unwrap_or(0), m.n, m.k, y, xkind, x)
    }
    fn norm(mut self, nw: u64, ss: u64) -> Job {
        self.nw = nw;
        self.ss = ss;
        self
    }
    fn split(mut self, s: usize) -> Job {
        self.split = s as i32;
        self
    }
    /// k units per weight-tile column step
    fn ku(&self) -> usize {
        match self.wkind {
            W_FP8 => 64,
            W_BF16 => 32,
            _ => 128,
        }
    }
    fn lds(&self) -> usize {
        let kn = (self.k as usize / self.ku()).div_ceil(self.split as usize) * self.ku();
        match self.wkind {
            W_BF16 => 2 * (kn + 8) * 2,
            W_FP8 => ((kn + 8 + 15) & !15) + kn / 32 * 4 + 4 * (kn / 64) * 2 * 4,
            _ => ((kn + 8 + 15) & !15) + kn / 32 * 4,
        }
    }
}

pub struct Layer {
    pub id: usize,
    pub ratio: usize,
    pub kv_src: bool,
    pub idx_src: bool,
    pub comp_src: usize,
    pub idxk_src: usize,
    hc_attn_fn: Buf,
    hc_attn: HcArgs,
    hc_ffn_fn: Buf,
    hc_ffn: HcArgs,
    attn_norm: Buf,
    ffn_norm: Buf,
    q_norm: Buf,
    kv_norm: Buf,
    sinks: Buf,
    wq_a: DMat,
    wkv: DMat,
    wq_b: DMat,
    wo_a: DMat,
    wo_b: DMat,
    comp_wkv: Option<DMat>,
    comp_wgate: Option<DMat>,
    comp_norm: Option<Buf>,
    idx_wq_b: Option<DMat>,
    idx_proj: Option<DMat>,
    idx_wk: Option<Buf>,
    idx_knorm: Option<Buf>,
    gate: DMat,
    gate_bias: Buf,
    sh_w1: DMat,
    sh_w3: DMat,
    sh_w2: DMat,
    pub experts: Buf, // ExpertMat [384][3]
    engram: Option<(DMat, Buf)>,
    // state
    ring: Buf,
    comp_cache: Option<Buf>,
    idx_cache: Option<Buf>,
    cstate: Option<Buf>,
}

pub struct Decoder {
    pub g: Gpu,
    pub cfg: Config,
    pub layers: Vec<Layer>,
    embed: Buf,
    head: DMat,
    out_norm: Buf,
    pub arena: Buf,
    rope_f: Buf,
    rope_fc: Buf,
    pub max_ctx: usize,
    // step buffers
    st: Buf,
    h: Buf,
    acc: Buf,
    x_a: Buf,
    x_f: Buf,
    q: Buf,
    iq: Buf,
    score: Buf,
    bscore: Buf,
    hist: Buf,
    ctl: Buf,
    sel: Buf,
    cand: Buf,
    ckey: Buf,
    comp_idx: Buf,
    ncomp: Buf,
    kb: Buf,
    nkb: Buf,
    nb: Buf,
    keep: Buf,
    nkeep: Buf,
    cscore: Buf,
    part: Buf,
    oq: Buf,
    os: Buf,
    route: Buf,
    xfin: Buf,
    pub logits: Buf,
    next: Buf,
    eng_rows: Buf,
    /// pinned host staging: [token i32, pad] then engram rows for the 2 engram layers
    pub host: *mut u8,
    graph: Option<Graph>,
    pub cb3: [u32; 2],
    pub cb2: u32,
    pub engram_st: Option<SafeTensors>,
    pub hist_tokens: Vec<i64>,
    pub pos: usize,
}

const ENG_ROW_BYTES: usize = 24 * 256 + 24 * 8;
const HOST_BYTES: usize = 256 + 2 * ENG_ROW_BYTES;

fn hc_args(st: &SafeTensors, l: usize, which: &str) -> Result<(Vec<f32>, HcArgs)> {
    let p = |s: &str| format!("layers.{l}.hc_{which}_{s}");
    let fnw = load::f32_of(st, &p("fn"))?;
    let scale = load::f32_of(st, &p("scale"))?;
    let base = load::f32_of(st, &p("base"))?;
    let mut a = HcArgs::default();
    a.scale.copy_from_slice(&scale);
    a.base.copy_from_slice(&base);
    Ok((fnw, a))
}

/// FNUZ fp8 byte for a value on the FNUZ grid (exact match required)
pub fn fnuz_byte(v: f32) -> u8 {
    for b in 0..=255u8 {
        if b == 0x80 {
            continue;
        }
        let s = b >> 7;
        let e = ((b >> 3) & 15) as i32;
        let m = (b & 7) as f32;
        let x = if e == 0 { m / 8.0 * 2f32.powi(-7) } else { (1.0 + m / 8.0) * 2f32.powi(e - 8) };
        let x = if s == 1 { -x } else { x };
        if x == v {
            return b;
        }
    }
    panic!("{v} is not an FNUZ fp8 value");
}

pub const CB3: [f32; 8] = [-8.0, -4.5, -2.5, -0.625, 1.25, 2.75, 4.5, 8.0];
pub const CB2: [f32; 4] = [-8.0, -1.875, 2.75, 8.0];

fn table(v: &[f32]) -> u32 {
    v.iter().enumerate().map(|(i, &x)| (fnuz_byte(x) as u32) << (8 * i)).sum()
}

/// Expert placement: format per (layer, expert)
pub struct Placement {
    pub fmt: Vec<i32>, // [layer * 384 + e]
}

const FP4_BYTES: usize = 3 * (INTER * D / 2 + INTER * D / 32);
const CB3_BYTES: usize = 3 * (INTER * D * 3 / 8 + INTER * D / 32);
const CB2_BYTES: usize = 3 * (INTER * D * 2 / 8 + INTER * D / 32);

pub fn expert_bytes(fmt: i32) -> usize {
    match fmt {
        E_FP4 => FP4_BYTES,
        E_CB3 => CB3_BYTES,
        _ => CB2_BYTES,
    }
}

impl Placement {
    /// Hottest experts (rank file order) stay MXFP4; then 3-bit; the coldest 2-bit if needed to fit `budget` bytes.
    /// `force_fp4`: (layer, expert) pairs that must stay exact (e.g. those a reference run routed to).
    pub fn plan(rank: &[(usize, usize)], budget: usize, force_fp4: &[(usize, usize)], n_layers: usize) -> Placement {
        let n = n_layers * NE;
        let mut order: Vec<usize> = rank.iter().filter(|&&(l, _)| l < n_layers).map(|&(l, e)| l * NE + e).collect();
        let mut seen = vec![false; n];
        for &i in &order {
            seen[i] = true;
        }
        for i in 0..n {
            if !seen[i] {
                order.push(i);
            }
        }
        let mut forced = vec![false; n];
        for &(l, e) in force_fp4 {
            if l < n_layers {
                forced[l * NE + e] = true;
            }
        }
        // start all FP4, demote from the cold end: FP4 -> CB3, then CB3 -> CB2, until it fits
        let mut fmt = vec![E_FP4; n];
        let total = |f: &Vec<i32>| f.iter().map(|&x| expert_bytes(x)).sum::<usize>();
        let mut tot = total(&fmt);
        for pass in [E_CB3, E_CB2] {
            for &i in order.iter().rev() {
                if tot <= budget {
                    break;
                }
                if forced[i] || fmt[i] >= pass {
                    continue;
                }
                tot -= expert_bytes(fmt[i]);
                fmt[i] = pass;
                tot += expert_bytes(pass);
            }
        }
        Placement { fmt }
    }
    pub fn counts(&self) -> [usize; 3] {
        let mut c = [0; 3];
        for &f in &self.fmt {
            c[f as usize] += 1;
        }
        c
    }
    pub fn bytes(&self) -> usize {
        self.fmt.iter().map(|&f| expert_bytes(f)).sum()
    }
}

impl Decoder {
    pub fn load(dir: &Path, max_ctx: usize, rank: &[(usize, usize)], force_fp4: &[(usize, usize)], n_layers_limit: Option<usize>) -> Result<Decoder> {
        let image: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/dsxa.hsaco"));
        if image.is_empty() {
            bail!("built without ROCm: no kernels");
        }
        let g = Gpu::new(image)?;
        let cfg = Config::from_hf(dir)?;
        let st = SafeTensors::open(dir, &[])?;
        let t0 = std::time::Instant::now();
        let nl = n_layers_limit.unwrap_or(cfg.n_layers);
        let mut layers = vec![];
        for l in 0..nl {
            let p = |s: &str| format!("layers.{l}.{s}");
            let has = |s: &str| st.tensors.contains_key(&p(s));
            let ratio = cfg.compress_ratios[l];
            let kv_src = cfg.kv_sources.contains(&l);
            let idx_src = cfg.index_sources.contains(&l);
            let comp_src = cfg.kv_sources.iter().filter(|&&s| s <= l).max().copied().unwrap_or(0);
            let idxk_src = comp_src;
            let (fa, ha) = hc_args(&st, l, "attn")?;
            let (ff, hf) = hc_args(&st, l, "ffn")?;
            let rows = |r: usize| if ratio > 0 { max_ctx / r + 1 } else { 0 };
            let engram = if has("engram.wkv.weight") {
                let q = load::f32_of(&st, &p("engram.q_weight"))?;
                let k = load::f32_of(&st, &p("engram.k_weight"))?;
                let qk: Vec<f32> = q.iter().zip(&k).map(|(a, b)| a * b).collect();
                Some((load::up_fp8(&g, &st, &p("engram.wkv.weight"))?, g.upload(&qk)?))
            } else {
                None
            };
            layers.push(Layer {
                id: l,
                ratio,
                kv_src,
                idx_src,
                comp_src,
                idxk_src,
                hc_attn_fn: g.upload(&fa)?,
                hc_attn: ha,
                hc_ffn_fn: g.upload(&ff)?,
                hc_ffn: hf,
                attn_norm: load::up_f32(&g, &st, &p("attn_norm.weight"))?,
                ffn_norm: load::up_f32(&g, &st, &p("ffn_norm.weight"))?,
                q_norm: load::up_f32(&g, &st, &p("attn.q_norm.weight"))?,
                kv_norm: load::up_f32(&g, &st, &p("attn.kv_norm.weight"))?,
                sinks: load::up_f32(&g, &st, &p("attn.attn_sink"))?,
                wq_a: load::up_fp8(&g, &st, &p("attn.wq_a.weight"))?,
                wkv: load::up_fp8(&g, &st, &p("attn.wkv.weight"))?,
                wq_b: load::up_fp8(&g, &st, &p("attn.wq_b.weight"))?,
                wo_a: load::up_fp8(&g, &st, &p("attn.wo_a.weight"))?,
                wo_b: load::up_fp8(&g, &st, &p("attn.wo_b.weight"))?,
                comp_wkv: if kv_src { Some(load::up_bf16(&g, &st, &p("attn.compressor.wkv.weight"))?) } else { None },
                comp_wgate: if kv_src && ratio > 1 { Some(load::up_bf16(&g, &st, &p("attn.compressor.wgate.weight"))?) } else { None },
                comp_norm: if kv_src { Some(load::up_f32(&g, &st, &p("attn.compressor.norm.weight"))?) } else { None },
                idx_wq_b: if idx_src { Some(load::up_fp8(&g, &st, &p("attn.indexer.wq_b.weight"))?) } else { None },
                idx_proj: if idx_src { Some(load::up_bf16(&g, &st, &p("attn.indexer.weights_proj.weight"))?) } else { None },
                idx_wk: if kv_src { Some(load::up_raw(&g, &st, &p("attn.indexer.wk.weight"))?) } else { None },
                idx_knorm: if kv_src { Some(load::up_f32(&g, &st, &p("attn.indexer.k_norm.weight"))?) } else { None },
                gate: load::up_bf16(&g, &st, &p("ffn.gate.weight"))?,
                gate_bias: load::up_f32(&g, &st, &p("ffn.gate.bias"))?,
                sh_w1: load::up_fp8(&g, &st, &p("ffn.shared_experts.w1.weight"))?,
                sh_w3: load::up_fp8(&g, &st, &p("ffn.shared_experts.w3.weight"))?,
                sh_w2: load::up_fp8(&g, &st, &p("ffn.shared_experts.w2.weight"))?,
                experts: g.alloc_zeroed(NE * 3 * std::mem::size_of::<ExpertMat>())?,
                engram,
                ring: g.alloc_zeroed(WIN * HD * 2)?,
                comp_cache: if kv_src { Some(g.alloc_zeroed(rows(ratio) * HD * 2)?) } else { None },
                idx_cache: if kv_src { Some(g.alloc_zeroed(rows(ratio) * ID * 2)?) } else { None },
                cstate: if kv_src { Some(g.alloc_zeroed(2 * 2 * HD * 4)?) } else { None },
            });
        }
        eprintln!("dense layers: {:.1} s, {:.1} GB on device", t0.elapsed().as_secs_f32(), g.allocated.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e9);
        let embed = load::up_raw(&g, &st, "embed.weight")?;
        let head = load::up_bf16(&g, &st, "head.weight")?;
        let out_norm = load::up_f32(&g, &st, "norm.weight")?;
        let ropes = |orig: usize, base: f32| -> Vec<f32> {
            dsx::numerics::Rope::new(cfg.rope_dim, orig, base, cfg.rope_factor, cfg.beta_fast, cfg.beta_slow).freqs
        };
        let rope_f = g.upload(&ropes(0, cfg.rope_theta))?;
        let rope_fc = g.upload(&ropes(cfg.original_seq_len, cfg.compress_rope_theta))?;
        let f = |n: usize| g.alloc_zeroed(n * 4);
        let host = g.host_alloc(HOST_BYTES)?;
        unsafe { std::ptr::write_bytes(host, 0, HOST_BYTES) };

        // ---- experts
        let (free, _) = g.mem_info()?;
        let reserve = 3usize << 30;
        let budget = free.saturating_sub(reserve + 64 << 20);
        let pl = Placement::plan(rank, budget, force_fp4, nl);
        let c = pl.counts();
        eprintln!("experts: {} MXFP4, {} 3-bit, {} 2-bit: {:.1} GB (free {:.1} GB)", c[0], c[1], c[2], pl.bytes() as f64 / 1e9, free as f64 / 1e9);
        let arena = g.alloc(pl.bytes())?;
        let mut dec = Decoder {
            embed,
            head,
            out_norm,
            arena,
            rope_f,
            rope_fc,
            max_ctx,
            st: g.alloc_zeroed(64)?,
            h: f(HC * D)?,
            acc: f(ACC * nl)?,
            x_a: f(D)?,
            x_f: f(D)?,
            q: f(NH * HD)?,
            iq: f(IH * ID)?,
            score: f(max_ctx + 64)?,
            bscore: f(max_ctx / 8 + 64)?,
            hist: f(4096)?,
            ctl: f(16)?,
            sel: f(2048)?,
            cand: f(CAND_CAP)?,
            ckey: f(CAND_CAP)?,
            comp_idx: f(512)?,
            ncomp: f(4)?,
            kb: f(2048)?,
            nkb: f(4)?,
            nb: f(4)?,
            keep: f(16384 + 64)?,
            nkeep: f(4)?,
            cscore: f(16384 + 64)?,
            part: f(NCHUNK * NH * (HD + 2))?,
            oq: f(NH * HD / 4)?,
            os: f(NH * HD / 32)?,
            route: f(16)?,
            xfin: f(D)?,
            logits: f(VOCAB)?,
            next: f(4)?,
            eng_rows: g.alloc_zeroed(2 * ENG_ROW_BYTES)?,
            host,
            graph: None,
            cb3: [table(&CB3[..4]), table(&CB3[4..])],
            cb2: table(&CB2),
            engram_st: None,
            hist_tokens: vec![],
            pos: 0,
            layers,
            cfg,
            g,
        };
        dec.load_experts(&st, &pl)?;
        let names: Vec<String> = dec.cfg.engram_layers.iter().flat_map(|l| [format!("layers.{l}.engram.embed.weight"), format!("layers.{l}.engram.embed.scale")]).collect();
        let want: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
        let est = SafeTensors::open(dir, &want)?;
        est.advise_random();
        dec.engram_st = Some(est);
        eprintln!("loaded in {:.1} s; {:.1} GB on device", t0.elapsed().as_secs_f32(), dec.g.allocated.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e9);
        Ok(dec)
    }

    fn load_experts(&mut self, st: &SafeTensors, pl: &Placement) -> Result<()> {
        let t0 = std::time::Instant::now();
        let g = &self.g;
        let n_layers = self.layers.len();
        // staging for re-quantization: raw fp4 + scales, codes [n][k] + scale bytes
        let mx = INTER * D;
        let st_w = g.alloc(mx / 2)?;
        let st_s = g.alloc(mx / 32)?;
        let st_c = g.alloc(mx)?;
        let st_b = g.alloc(mx / 32)?;
        let cb3 = g.upload(&CB3)?;
        let cb2 = g.upload(&CB2)?;
        let mut off = 0usize;
        // host-side FP4 tiling runs on a thread pool, one expert ahead of the uploads
        let mut table = vec![ExpertMat::default(); NE * 3];
        for l in 0..n_layers {
            for e in 0..NE {
                let fmt = pl.fmt[l * NE + e];
                for (mi, wn) in ["w1", "w3", "w2"].iter().enumerate() {
                    let name = format!("layers.{l}.ffn.experts.{e}.{wn}.weight");
                    let (n, kh) = load::shape2(st, &name)?;
                    let k = kh * 2;
                    let w = st.bytes(st.get(&name)?);
                    let s = st.bytes(st.get(&name.replace(".weight", ".scale"))?);
                    let em = &mut table[e * 3 + mi];
                    em.fmt = fmt;
                    em.woff = off as i64;
                    if fmt == E_FP4 {
                        let (codes, scales) = load::tile_fp4(w, s, n, k);
                        g.htod(self.arena.at(off), &codes)?;
                        off += codes.len();
                        em.soff = off as i64;
                        g.htod(self.arena.at(off), &scales)?;
                        off += scales.len();
                        em.mul = 1.0;
                    } else {
                        let bits = if fmt == E_CB3 { 3 } else { 2 };
                        // per-matrix multiplier: the largest group scale fits e4m3 (max 240) with headroom
                        let smax = s.iter().map(|&b| f32::from_bits((b as u32) << 23)).fold(0f32, f32::max);
                        let mul = (smax * 6.0 / 8.0 * 1.25) / 128.0;
                        g.htod(st_w.ptr, w)?;
                        g.htod(st_s.ptr, s)?;
                        let (cb, nlv) = if bits == 3 { (cb3.ptr, 8) } else { (cb2.ptr, 4) };
                        let groups = n * k / 32;
                        g.launch("cb_quant", ((groups as u32).div_ceil(256), 1, 1), 256, 0, &[Arg::P(st_w.ptr), Arg::P(st_s.ptr), Arg::I(n as i32), Arg::I(k as i32), Arg::P(cb), Arg::I(nlv), Arg::F(mul), Arg::P(st_c.ptr), Arg::P(st_b.ptr)])?;
                        let cbytes = n * k * bits / 8;
                        let threads = (n / 16) * (k / 128) * 64;
                        let wo = self.arena.at(off);
                        let so = self.arena.at(off + cbytes);
                        g.launch("cb_pack", ((threads as u32).div_ceil(256), 1, 1), 256, 0, &[Arg::P(st_c.ptr), Arg::P(st_b.ptr), Arg::I(n as i32), Arg::I(k as i32), Arg::I(bits as i32), Arg::P(wo), Arg::P(so)])?;
                        g.sync()?;
                        em.soff = (off + cbytes) as i64;
                        off += cbytes + n * k / 32;
                        em.mul = mul;
                    }
                }
            }
            g.htod(self.layers[l].experts.ptr, &table)?;
            if l % 4 == 3 || l + 1 == n_layers {
                eprintln!("experts: layer {l} done, {:.1} GB, {:.0} s", off as f64 / 1e9, t0.elapsed().as_secs_f32());
            }
        }
        if off != pl.bytes() {
            bail!("expert arena: wrote {off}, planned {}", pl.bytes());
        }
        Ok(())
    }

    fn accp(&self, l: usize, off: usize) -> u64 {
        self.acc.at((l * ACC + off) * 4)
    }

    fn jobs(&self, mut jobs: Vec<Job>, route_mode: i32, router_layer: Option<usize>) -> Result<()> {
        if jobs.len() > MAXJ {
            bail!("{} jobs", jobs.len());
        }
        let mut lds = 0;
        let mut blocks = 0u32;
        // jobs with split 0 share one k-split chosen so the whole launch has ~2048 waves
        let tiles: usize = jobs.iter().map(|j| (j.n as usize).div_ceil(16)).sum();
        let mut s = 1;
        while tiles * s < 2048 && s < 32 {
            s *= 2;
        }
        for j in jobs.iter_mut() {
            if j.split == 0 {
                let units = j.k as usize / j.ku();
                let mut sj = s;
                while sj > 1 && units / sj < 2 {
                    sj /= 2;
                }
                j.split = sj as i32;
            }
            j.blocks = ((j.n as usize).div_ceil(64) * j.split as usize) as i32;
            blocks += j.blocks as u32;
            lds = lds.max(j.lds());
        }
        let n = jobs.len();
        let pad = jobs[0];
        let mut arr = [pad; MAXJ];
        arr[..n].copy_from_slice(&jobs);
        let js = Jobs { j: arr, n: n as i32, eps: self.cfg.eps, lim: self.cfg.swiglu_limit, cb3: self.cb3, cb2: self.cb2 };
        let (logits, bias) = match router_layer {
            Some(l) => (self.accp(l, A_LOGITS), self.layers[l].gate_bias.ptr),
            None => (0, 0),
        };
        self.g.launch(
            "gemv_jobs",
            (blocks, 1, 1),
            256,
            lds as u32,
            &[
                Arg::B(bytes_of(&js)),
                Arg::P(self.st.ptr),
                Arg::P(self.rope_f.ptr),
                Arg::P(self.rope_fc.ptr),
                Arg::I(route_mode),
                Arg::P(logits),
                Arg::P(bias),
                Arg::F(self.cfg.route_scale),
                Arg::P(self.route.ptr),
                Arg::P(self.eng_rows.ptr),
            ],
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn hc_step(&self, y: u64, round_y: bool, pmix: u64, pa: &HcArgs, premix: u64, prea: &HcArgs, fnw: u64, nmix: u64, x_raw: u64, xss: u64, eng: Option<(u64, u64)>) -> Result<()> {
        let c = &self.cfg;
        let (ev, es) = eng.unwrap_or((0, 0));
        self.g.launch(
            "hc_step",
            ((D / 64) as u32, 1, 1),
            256,
            0,
            &[
                Arg::P(self.h.ptr),
                Arg::P(y),
                Arg::I(round_y as i32),
                Arg::P(pmix),
                Arg::B(bytes_of(pa)),
                Arg::P(premix),
                Arg::B(bytes_of(prea)),
                Arg::P(fnw),
                Arg::P(nmix),
                Arg::P(x_raw),
                Arg::P(xss),
                Arg::P(ev),
                Arg::P(es),
                Arg::P(0),
                Arg::F(c.hc_eps),
                Arg::F(c.eps),
                Arg::I(c.sinkhorn_iters as i32),
            ],
        )
    }

    /// radix top-k: n from `n_src` (device int) or (pos+1)/ratio; writes `out` (+offset, -1 padding to kmax), *nout
    #[allow(clippy::too_many_arguments)]
    fn topk(&self, scores: u64, ratio: usize, n_src: u64, k: usize, map: u64, offset: i32, out: u64, kmax: usize, nout: u64) -> Result<()> {
        let g = &self.g;
        g.launch("topk_hist", (128, 1, 1), 256, 0, &[Arg::P(self.st.ptr), Arg::I(ratio as i32), Arg::P(n_src), Arg::P(scores), Arg::P(self.hist.ptr), Arg::P(self.ctl.ptr)])?;
        g.launch("topk_scan", (1, 1, 1), 1024, 0, &[Arg::P(self.hist.ptr), Arg::I(k as i32), Arg::P(self.ctl.ptr)])?;
        g.launch("topk_collect", (128, 1, 1), 256, 0, &[Arg::P(scores), Arg::P(self.ctl.ptr), Arg::P(self.sel.ptr), Arg::P(self.cand.ptr), Arg::P(self.ckey.ptr), Arg::I(CAND_CAP as i32)])?;
        g.launch(
            "topk_final",
            (1, 1, 1),
            1024,
            0,
            &[Arg::P(self.ctl.ptr), Arg::P(self.sel.ptr), Arg::P(self.cand.ptr), Arg::P(self.ckey.ptr), Arg::P(map), Arg::I(offset), Arg::P(out), Arg::I(kmax as i32), Arg::P(nout), Arg::P(self.hist.ptr)],
        )
    }

    /// Enqueue one decode step (T = 1). Used for capture.
    fn record(&self) -> Result<()> {
        let g = &self.g;
        let c = &self.cfg;
        let nl = self.layers.len();
        g.htod_async(self.st.at(4), unsafe { self.host.add(0) }, 4)?; // token
        g.htod_async(self.eng_rows.ptr, unsafe { self.host.add(256) }, 2 * ENG_ROW_BYTES)?;
        g.memset(self.acc.ptr, 0, ACC * nl * 4)?;
        g.launch("embed", (20, 1, 1), 256, 0, &[Arg::P(self.embed.ptr), Arg::P(self.st.ptr), Arg::P(self.h.ptr)])?;
        let wscale = (c.idx_dim as f32).powf(-0.5) * (c.idx_heads as f32).powf(-0.5);
        for (li, l) in self.layers.iter().enumerate() {
            // ---- residual step into the attention half
            let none = HcArgs::default();
            if li == 0 {
                self.hc_step(0, false, 0, &none, 0, &none, l.hc_attn_fn.ptr, self.accp(0, A_HMIX_A), self.x_a.ptr, self.accp(0, A_XSS_A), None)?;
            } else {
                let p = &self.layers[li - 1];
                let (pm, py) = (self.accp(li - 1, A_HMIX_F), self.accp(li - 1, A_Y));
                if let Some((wkv, qk)) = &l.engram {
                    self.hc_step(py, true, pm, &p.hc_ffn, 0, &none, 0, 0, 0, 0, None)?;
                    let ei = c.engram_layers.iter().position(|&x| x == li).unwrap();
                    let mut j = Job::dense(wkv, W_FP8, self.accp(li, A_ENG_KV), X_ENGRAM, 0);
                    j.split = 1;
                    // rows buffer for this engram layer
                    self.jobs_with_rows(vec![j], self.eng_rows.at(ei * ENG_ROW_BYTES))?;
                    g.launch("engram_sums", (5, 4, 1), 256, 0, &[Arg::P(self.accp(li, A_ENG_KV)), Arg::P(self.h.ptr), Arg::P(qk.ptr), Arg::P(self.accp(li, A_ENG_SUM))])?;
                    self.hc_step(0, false, 0, &none, pm, &p.hc_ffn, l.hc_attn_fn.ptr, self.accp(li, A_HMIX_A), self.x_a.ptr, self.accp(li, A_XSS_A), Some((self.accp(li, A_ENG_KV + 4 * D), self.accp(li, A_ENG_SUM))))?;
                } else {
                    self.hc_step(py, true, pm, &p.hc_ffn, pm, &p.hc_ffn, l.hc_attn_fn.ptr, self.accp(li, A_HMIX_A), self.x_a.ptr, self.accp(li, A_XSS_A), None)?;
                }
            }
            // ---- attention projections
            let xs = self.accp(li, A_XSS_A);
            let mut jobs = vec![
                Job::dense(&l.wq_a, W_FP8, self.accp(li, A_QR), X_NORM, self.x_a.ptr).norm(l.attn_norm.ptr, xs).split(0),
                Job::dense(&l.wkv, W_FP8, self.accp(li, A_KV), X_NORM, self.x_a.ptr).norm(l.attn_norm.ptr, xs).split(0),
            ];
            if let Some(m) = &l.comp_wkv {
                jobs.push(Job::dense(m, W_BF16, self.accp(li, A_CKV), X_NORM, self.x_a.ptr).norm(l.attn_norm.ptr, xs).split(0));
            }
            if let Some(m) = &l.comp_wgate {
                jobs.push(Job::dense(m, W_BF16, self.accp(li, A_CGATE), X_NORM, self.x_a.ptr).norm(l.attn_norm.ptr, xs).split(0));
            }
            if let Some(m) = &l.idx_proj {
                jobs.push(Job::dense(m, W_BF16, self.accp(li, A_IW), X_NORM, self.x_a.ptr).norm(l.attn_norm.ptr, xs).split(0));
            }
            self.jobs(jobs, 0, None)?;
            let mut qj = Job::dense(&l.wq_b, W_FP8, self.q.ptr, X_RMS, self.accp(li, A_QR)).norm(l.q_norm.ptr, 0);
            qj.epi = 1;
            qj.ropec = (l.ratio > 0) as i32;
            let mut jobs = vec![qj];
            if let Some(m) = &l.idx_wq_b {
                let mut j = Job::dense(m, W_FP8, self.iq.ptr, X_RMS, self.accp(li, A_QR)).norm(l.q_norm.ptr, 0);
                j.epi = 2;
                j.ropec = 1;
                jobs.push(j);
            }
            self.jobs(jobs, 0, None)?;
            // window kv, compressor, indexer keys
            let opt = |b: &Option<Buf>| b.as_ref().map(|x| x.ptr).unwrap_or(0);
            g.launch(
                "attn_kv",
                (1, 1, 1),
                512,
                0,
                &[
                    Arg::P(self.st.ptr),
                    Arg::P(self.accp(li, A_KV)),
                    Arg::P(l.kv_norm.ptr),
                    Arg::P(l.ring.ptr),
                    Arg::P(self.rope_f.ptr),
                    Arg::I(l.ratio as i32),
                    Arg::P(if l.kv_src { self.accp(li, A_CKV) } else { 0 }),
                    Arg::P(self.accp(li, A_CGATE)),
                    Arg::P(opt(&l.cstate)),
                    Arg::P(opt(&l.comp_norm)),
                    Arg::P(opt(&l.idx_wk)),
                    Arg::P(opt(&l.idx_knorm)),
                    Arg::P(opt(&l.comp_cache)),
                    Arg::P(opt(&l.idx_cache)),
                    Arg::P(self.rope_fc.ptr),
                    Arg::F(c.eps),
                ],
            )?;
            // indexer + top-k
            if l.idx_src {
                let kc = self.layers[l.idxk_src].idx_cache.as_ref().context("idx cache")?.ptr;
                let after_cand = li > c.cand_layer;
                let (rows, nrows, sc) = if after_cand { (self.keep.ptr, self.nkeep.ptr, self.cscore.ptr) } else { (0, 0, self.score.ptr) };
                g.launch("idx_score", (256, 1, 1), 256, 0, &[Arg::P(self.st.ptr), Arg::I(l.ratio as i32), Arg::P(self.iq.ptr), Arg::P(self.accp(li, A_IW)), Arg::F(wscale), Arg::P(kc), Arg::P(rows), Arg::P(nrows), Arg::P(sc)])?;
                if li == c.cand_layer {
                    g.launch("block_max8", (64, 1, 1), 256, 0, &[Arg::P(self.st.ptr), Arg::I(l.ratio as i32), Arg::I(c.cand_block as i32), Arg::P(self.score.ptr), Arg::P(self.bscore.ptr), Arg::P(self.nb.ptr)])?;
                    self.topk(self.bscore.ptr, 0, self.nb.ptr, c.cand_blocks, 0, 0, self.kb.ptr, c.cand_blocks, self.nkb.ptr)?;
                    g.launch("expand_blocks", (64, 1, 1), 256, 0, &[Arg::P(self.st.ptr), Arg::I(l.ratio as i32), Arg::I(c.cand_block as i32), Arg::P(self.kb.ptr), Arg::P(self.nkb.ptr), Arg::P(self.keep.ptr), Arg::P(self.nkeep.ptr)])?;
                    g.launch("gather_f", (64, 1, 1), 256, 0, &[Arg::P(self.score.ptr), Arg::P(self.keep.ptr), Arg::P(self.nkeep.ptr), Arg::P(self.cscore.ptr)])?;
                    self.topk(self.cscore.ptr, 0, self.nkeep.ptr, c.idx_topk, self.keep.ptr, WIN as i32, self.comp_idx.ptr, c.idx_topk, self.ncomp.ptr)?;
                } else if after_cand {
                    self.topk(self.cscore.ptr, 0, self.nkeep.ptr, c.idx_topk, self.keep.ptr, WIN as i32, self.comp_idx.ptr, c.idx_topk, self.ncomp.ptr)?;
                } else {
                    self.topk(self.score.ptr, l.ratio, 0, c.idx_topk, 0, WIN as i32, self.comp_idx.ptr, c.idx_topk, self.ncomp.ptr)?;
                }
            }
            // sparse attention
            let comp = if l.ratio > 0 { self.layers[l.comp_src].comp_cache.as_ref().context("comp cache")?.ptr } else { 0 };
            let scale = 1.0 / (HD as f32).sqrt();
            g.launch("attn_part", (NCHUNK as u32, 8, 1), 256, 0, &[Arg::P(self.st.ptr), Arg::P(self.q.ptr), Arg::P(l.ring.ptr), Arg::P(comp), Arg::P(self.comp_idx.ptr), Arg::P(self.ncomp.ptr), Arg::I(WIN as i32), Arg::F(scale), Arg::P(self.part.ptr)])?;
            let rf = if l.ratio > 0 { self.rope_fc.ptr } else { self.rope_f.ptr };
            g.launch("attn_merge", (NH as u32, 1, 1), 512, 0, &[Arg::P(self.st.ptr), Arg::P(self.part.ptr), Arg::I(NCHUNK as i32), Arg::P(l.sinks.ptr), Arg::P(rf), Arg::P(self.oq.ptr), Arg::P(self.os.ptr)])?;
            let mut oa = Job::dense(&l.wo_a, W_FP8, self.accp(li, A_R), X_Q, self.oq.ptr).split(0);
            oa.x2 = self.os.ptr;
            oa.k = (NH * HD / 8) as i32;
            oa.xgroup = 1024;
            self.jobs(vec![oa], 0, None)?;
            self.jobs(vec![Job::dense(&l.wo_b, W_FP8, self.accp(li, A_A), X_BF16R, self.accp(li, A_R)).split(0)], 0, None)?;
            // ---- residual step into the ffn half
            let am = self.accp(li, A_HMIX_A);
            self.hc_step(self.accp(li, A_A), false, am, &l.hc_attn, am, &l.hc_attn, l.hc_ffn_fn.ptr, self.accp(li, A_HMIX_F), self.x_f.ptr, self.accp(li, A_XSS_F), None)?;
            let xs = self.accp(li, A_XSS_F);
            self.jobs(
                vec![
                    Job::dense(&l.gate, W_BF16, self.accp(li, A_LOGITS), X_NORM, self.x_f.ptr).norm(l.ffn_norm.ptr, xs).split(0),
                    Job::dense(&l.sh_w1, W_FP8, self.accp(li, A_SHG), X_NORM, self.x_f.ptr).norm(l.ffn_norm.ptr, xs).split(0),
                    Job::dense(&l.sh_w3, W_FP8, self.accp(li, A_SHU), X_NORM, self.x_f.ptr).norm(l.ffn_norm.ptr, xs).split(0),
                ],
                0,
                None,
            )?;
            // routed experts: gate/up for the 6 picks
            let mut jobs = vec![];
            for s in 0..6 {
                for (mi, base) in [(0, A_G), (1, A_U)] {
                    let mut j = Job::new(W_EXPERT, self.arena.ptr, l.experts.ptr, INTER, D, self.accp(li, base + s * INTER), X_NORM, self.x_f.ptr).norm(l.ffn_norm.ptr, xs);
                    j.eslot = s as i32;
                    j.emat = mi;
                    j.split = 0;
                    jobs.push(j);
                }
            }
            self.jobs(jobs, 1, Some(li))?;
            // down projections (+ shared expert) into y
            let mut jobs = vec![];
            for s in 0..6 {
                let mut j = Job::new(W_EXPERT, self.arena.ptr, l.experts.ptr, D, INTER, self.accp(li, A_Y), X_SWIGLU, self.accp(li, A_G + s * INTER));
                j.x2 = self.accp(li, A_U + s * INTER);
                j.eslot = s as i32;
                j.emat = 2;
                j.split = 0;
                jobs.push(j);
            }
            let mut sj = Job::dense(&l.sh_w2, W_FP8, self.accp(li, A_Y), X_SWIGLU, self.accp(li, A_SHG)).split(0);
            sj.x2 = self.accp(li, A_SHU);
            jobs.push(sj);
            // every job adds into y: atomics (split >= 2 for all)
            for j in jobs.iter_mut() {
                j.split = 2;
            }
            self.jobs(jobs, 2, Some(li))?;
        }
        // ---- final residual step, norm, head, argmax
        let none = HcArgs::default();
        let last = &self.layers[nl - 1];
        let (pm, py) = (self.accp(nl - 1, A_HMIX_F), self.accp(nl - 1, A_Y));
        self.hc_step(py, true, pm, &last.hc_ffn, 0, &none, 0, 0, 0, 0, None)?;
        g.launch("hc_final", (1, 1, 1), 256, 0, &[Arg::P(self.h.ptr), Arg::P(pm), Arg::B(bytes_of(&last.hc_ffn)), Arg::P(self.out_norm.ptr), Arg::P(self.xfin.ptr), Arg::F(c.hc_eps), Arg::F(c.eps), Arg::I(c.sinkhorn_iters as i32)])?;
        let mut hj = Job::dense(&self.head, W_BF16, self.logits.ptr, X_PLAIN, self.xfin.ptr);
        hj.lo = 1;
        hj.split = 1;
        self.jobs(vec![hj], 0, None)?;
        g.launch("argmax", (1, 1, 1), 1024, 0, &[Arg::P(self.logits.ptr), Arg::I(VOCAB as i32), Arg::P(self.st.ptr), Arg::P(self.next.ptr)])?;
        g.launch("advance", (1, 1, 1), 1, 0, &[Arg::P(self.st.ptr), Arg::P(self.next.ptr)])?;
        Ok(())
    }

    fn jobs_with_rows(&self, jobs: Vec<Job>, rows: u64) -> Result<()> {
        // same as jobs() but the engram rows pointer is this layer's slice
        let mut jobs = jobs;
        let mut lds = 0;
        let mut blocks = 0u32;
        for j in jobs.iter_mut() {
            j.blocks = ((j.n as usize).div_ceil(64) * j.split as usize) as i32;
            blocks += j.blocks as u32;
            lds = lds.max(j.lds());
        }
        let n = jobs.len();
        let mut arr = [jobs[0]; MAXJ];
        arr[..n].copy_from_slice(&jobs);
        let js = Jobs { j: arr, n: n as i32, eps: self.cfg.eps, lim: self.cfg.swiglu_limit, cb3: self.cb3, cb2: self.cb2 };
        self.g.launch(
            "gemv_jobs",
            (blocks, 1, 1),
            256,
            lds as u32,
            &[Arg::B(bytes_of(&js)), Arg::P(self.st.ptr), Arg::P(self.rope_f.ptr), Arg::P(self.rope_fc.ptr), Arg::I(0), Arg::P(0), Arg::P(0), Arg::F(0.0), Arg::P(self.route.ptr), Arg::P(rows)],
        )
    }

    pub fn capture(&mut self) -> Result<()> {
        self.g.sync()?;
        self.g.begin_capture()?;
        let r = self.record();
        let gr = self.g.end_capture();
        r?;
        self.graph = Some(gr?);
        Ok(())
    }

    /// Host side of the Engram lookup for the newest token: hash ids -> rows (fp8 + e8m0) into pinned memory.
    fn engram_rows(&mut self, token: u32) -> Result<()> {
        let c = &self.cfg;
        self.hist_tokens.push(c.engram_token_map[token as usize] as i64);
        let est = self.engram_st.as_ref().context("engram tables")?;
        let hd = c.engram_dim;
        for (ei, &l) in c.engram_layers.iter().enumerate() {
            let ids = engram_hashes(c, &self.hist_tokens, ei);
            let wt = est.get(&format!("layers.{l}.engram.embed.weight"))?;
            let sct = est.get(&format!("layers.{l}.engram.embed.scale"))?;
            let (wb, sb) = (est.bytes(wt), est.bytes(sct));
            let base = unsafe { self.host.add(256 + ei * ENG_ROW_BYTES) };
            for (r, &id) in ids.iter().enumerate() {
                let id = id as usize;
                unsafe {
                    std::ptr::copy_nonoverlapping(wb[id * hd..].as_ptr(), base.add(r * 256), 256);
                    std::ptr::copy_nonoverlapping(sb[id * (hd / 32)..].as_ptr(), base.add(24 * 256 + r * 8), 8);
                }
            }
        }
        Ok(())
    }

    /// Feed one token at the current position; returns the greedy next token. Logits stay on the device.
    pub fn step(&mut self, token: u32) -> Result<u32> {
        if self.pos >= self.max_ctx {
            bail!("context full ({})", self.max_ctx);
        }
        if self.graph.is_none() {
            self.capture()?;
        }
        unsafe { *(self.host as *mut i32) = token as i32 };
        self.engram_rows(token)?;
        self.g.graph_launch(self.graph.as_ref().unwrap())?;
        let next: Vec<i32> = self.g.dtoh(self.next.ptr, 1)?;
        self.pos += 1;
        Ok(next[0] as u32)
    }

    pub fn logits(&self) -> Result<Vec<f32>> {
        self.g.dtoh(self.logits.ptr, VOCAB)
    }

    pub fn route(&self) -> Result<Vec<i32>> {
        self.g.dtoh(self.route.ptr, 6)
    }

    /// Reset to an empty context.
    pub fn reset(&mut self) -> Result<()> {
        self.g.memset(self.st.ptr, 0, 64)?;
        for l in &self.layers {
            self.g.memset(l.ring.ptr, 0, l.ring.bytes)?;
            if let Some(b) = &l.cstate {
                self.g.memset(b.ptr, 0, b.bytes)?;
            }
        }
        self.g.sync()?;
        self.hist_tokens.clear();
        self.pos = 0;
        Ok(())
    }

    /// Benchmark helper: pretend `pos` tokens are already in context (caches hold whatever they hold).
    pub fn set_pos(&mut self, pos: usize) -> Result<()> {
        self.g.htod(self.st.ptr, &[pos as i32])?;
        self.pos = pos;
        Ok(())
    }

    /// Fill compressed caches and the window with random bf16 values (benchmarks at long context).
    pub fn fill_random(&self) -> Result<()> {
        for l in &self.layers {
            for b in [Some(&l.ring), l.comp_cache.as_ref(), l.idx_cache.as_ref()].into_iter().flatten() {
                self.g.launch("fill_rand", (1024, 1, 1), 256, 0, &[Arg::P(b.ptr), Arg::I((b.bytes / 2) as i32), Arg::I(l.id as i32)])?;
            }
        }
        self.g.sync()
    }

    pub fn time_step(&mut self, token: u32) -> Result<f32> {
        if self.graph.is_none() {
            self.capture()?;
        }
        unsafe { *(self.host as *mut i32) = token as i32 };
        self.engram_rows(token)?;
        let gr = self.graph.as_ref().unwrap();
        let us = self.g.time(|| self.g.graph_launch(gr))?;
        self.pos += 1;
        Ok(us)
    }
}

/// model.py NgramHashState for engram layer `li`: hash ids of the 2..max_ngram n-grams ending at the newest token
pub fn engram_hashes(c: &Config, hist: &[i64], li: usize) -> Vec<u64> {
    let n = c.engram_ngram;
    let t = hist.len() - 1;
    let mut toks = vec![0i64; n];
    let mut blocked = false;
    for s in 0..n {
        let src = if t >= s { hist[t - s] } else { -1 };
        blocked = blocked || t < s || src == -1;
        toks[s] = if blocked { c.engram_pad } else { src };
    }
    let mult = &c.engram_multipliers[li * n..(li + 1) * n];
    let per_layer = (n - 1) * c.engram_heads;
    let primes = &c.engram_primes[li * per_layer..(li + 1) * per_layer];
    let mut offsets = vec![0u64; per_layer];
    for i in 1..per_layer {
        offsets[i] = offsets[i - 1] + primes[i - 1];
    }
    let mut rolling = toks[0].wrapping_mul(mult[0] as i64);
    let mut out = vec![];
    for i in 1..n {
        rolling ^= toks[i].wrapping_mul(mult[i] as i64);
        for hd in 0..c.engram_heads {
            let col = (i - 1) * c.engram_heads + hd;
            out.push((rolling.rem_euclid(primes[col] as i64)) as u64 + offsets[col]);
        }
    }
    out
}
