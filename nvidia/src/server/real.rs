//! The GPU engine behind the server: GpuModel with prefix reuse.
//!
//! The KV state of the last conversation (prompt + generated ids) stays on the GPU. A request reuses the longest
//! common token prefix: if the new prompt extends everything cached, it just continues; if it diverges inside the
//! generated part (clients often re-send the assistant turn differently), it rewinds to the snapshot taken at the
//! end of the previous prompt; otherwise it starts over.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::Result;

use super::engine::{Engine, Finish, GenParams, GenStats};
use super::sample::Sampler;
use crate::gpumodel::{GpuModel, Placement, Snapshot, VisionWeights};
use crate::model::Model;

/// Mirrors the `gen` command's flags.
#[derive(Debug, Clone)]
pub struct RealEngineArgs {
    pub model: PathBuf,
    pub engram: Option<PathBuf>,
    pub ctx: usize,
    pub chunk: usize,
    pub gpu0_gb: f64,
    pub gpu1_gb: f64,
    pub arena_gb: f64,
    pub rank: Option<PathBuf>,
    /// vision weights: `vision.safetensors` or the official checkpoint directory (None = text only)
    pub vision: Option<PathBuf>,
}

struct Real {
    gm: GpuModel<'static>,
    cached: Vec<u32>,
    snap: Option<Snapshot>,
    stats: GenStats,
    vision: Option<VisionWeights>,
}

// GPU state is created and used only on the engine thread (EngineHandle::spawn builds us there).
unsafe impl Send for Real {}

/// Runs on the engine thread (`EngineHandle::spawn`), so GPU contexts are created there.
pub fn real_engine(args: &RealEngineArgs) -> Result<Box<dyn Engine>> {
    let m: &'static Model = Box::leak(Box::new(if args.model.is_dir() {
        Model::load_hf(&args.model)?
    } else {
        Model::load(&args.model, args.engram.as_deref())?
    }));
    let pl = Placement { gpu0_gb: args.gpu0_gb, aux_gb: args.gpu1_gb, arena_gb: args.arena_gb, prefill_chunk: args.chunk, rank: args.rank.clone() };
    let vision = args.vision.as_deref().map(VisionWeights::load).transpose()?;
    let gm = GpuModel::with_placement(m, -1, args.ctx, pl)?;
    Ok(Box::new(Real { gm, cached: vec![], snap: None, stats: GenStats::default(), vision }))
}

impl Engine for Real {
    fn generate(&mut self, prompt: &[u32], p: &GenParams, emit: &mut dyn FnMut(u32) -> bool) -> Result<Finish> {
        let token_audit_path = std::env::var("DSX_TOKEN_AUDIT").ok();
        let t0 = Instant::now();
        let profile = std::env::var("DSX_RUNTIME_STATS").is_ok_and(|v| v == "1");
        anyhow::ensure!(!prompt.is_empty(), "empty prompt");
        // the prefix-cache key: every image position carries the same token id, so key those by image content
        let mut key = prompt.to_vec();
        for im in p.images.iter() {
            for (i, k) in key[im.start..im.start + im.len()].iter_mut().enumerate() {
                *k = 0xf000_0000 | ((im.hash as u32 ^ (im.hash >> 32) as u32 ^ i as u32) & 0x0fff_ffff);
            }
        }
        // at least one prompt token must be fed to get the next-token logits
        let lcp = key.iter().zip(&self.cached).take_while(|(a, b)| a == b).count().min(prompt.len() - 1);
        let start = if lcp == self.cached.len() && self.gm.pos == self.cached.len() {
            lcp
        } else if let Some(s) = self.snap.as_ref().filter(|s| s.pos <= lcp) {
            self.gm.restore(s)?;
            s.pos
        } else {
            self.gm.reset();
            0
        };
        // images not already in the KV cache go through the vision tower; their span rows replace the embeddings
        self.gm.images.clear();
        for im in p.images.iter().filter(|im| im.start + im.len() > start) {
            let vw = self.vision.as_ref().ok_or_else(|| anyhow::anyhow!("image input needs the vision weights (--vision)"))?;
            let tv = Instant::now();
            let rows = self.gm.encode_image(vw, im)?;
            tracing::info!(patches = im.n_vit_h * im.n_vit_w, tokens = im.len(), ms = tv.elapsed().as_millis() as u64, "image encoded");
            self.gm.images.push((im.start, rows));
        }
        let pb = self.gm.pb.take().expect("prefill buffers");
        let prefill_clock = profile.then(Instant::now);
        let r = self.gm.prefill(&pb, &prompt[start..]);
        let model_prefill_s = prefill_clock.map(|t| t.elapsed().as_secs_f64()).unwrap_or(0.0);
        self.gm.pb = Some(pb);
        self.gm.images.clear();
        let mut logits = r?;
        let snapshot_clock = profile.then(Instant::now);
        self.snap = Some(self.gm.snapshot()?);
        let snapshot_s = snapshot_clock.map(|t| t.elapsed().as_secs_f64()).unwrap_or(0.0);
        self.cached = key;
        let initial_residency = token_audit_path.as_ref().map(|_| self.gm.validation_residency());
        let mut output_ids = token_audit_path.as_ref().map(|_| Vec::with_capacity(p.max_new.min(8192)));
        let prefill = t0.elapsed();
        let td = Instant::now();
        let mut sampler = Sampler::from_params(p);
        let gpu_sampling = std::env::var("DSX_GPU_SAMPLE").is_ok_and(|v| v == "1")
            && p.temperature == 1.0 && p.top_p >= 1.0
            && p.hesitation.as_ref().is_none_or(|h| h.lambda == 0.0);
        let mut gpu_next = None;
        let (mut gpu_samples, mut gpu_fallbacks) = (0usize,0usize);
        let mut fin = Finish::Length;
        let mut hes = p.hesitation.clone();
        let mut lookup_audit = crate::spec::LookupAudit::from_env();
        let (mut sample_s, mut emit_s, mut step_s, mut steps) = (0.0, 0.0, 0.0, 0usize);
        let adapt_before = self.gm.adapt_s;
        let picks_before = self.gm.cpu_picks();
        for _ in 0..p.max_new {
            if let Some(h) = &hes {
                for &i in h.ids.iter() {
                    if let Some(l) = logits.get_mut(i as usize) {
                        *l -= h.lambda;
                    }
                }
            }
            let clock = profile.then(Instant::now);
            let tok = gpu_next.take().unwrap_or_else(|| sampler.sample(&logits));
            if let Some(t) = clock { sample_s += t.elapsed().as_secs_f64(); }
            if let Some(ids) = &mut output_ids { ids.push(tok); }
            if hes.as_ref().is_some_and(|h| tok == h.close) {
                hes = None;
            }
            let clock = profile.then(Instant::now);
            let keep = emit(tok);
            if let Some(t) = clock { emit_s += t.elapsed().as_secs_f64(); }
            if !keep {
                fin = Finish::Cancelled;
                break;
            }
            if self.gm.pos + 1 >= self.gm.max_ctx() {
                break;
            }
            let clock = profile.then(Instant::now);
            if gpu_sampling {
                let saved = sampler.clone();
                let uniform = sampler.uniform_bits();
                match self.gm.step_sampled(tok, uniform)? {
                    crate::sampling::StepOutput::Token(next) => { gpu_next = Some(next); gpu_samples += 1; }
                    crate::sampling::StepOutput::Logits(next) => { sampler = saved; logits = next; gpu_fallbacks += 1; }
                }
            } else {
                logits = self.gm.step(tok)?;
            }
            if let Some(t) = clock { step_s += t.elapsed().as_secs_f64(); steps += 1; }
            self.cached.push(tok);
            if let Some(audit) = &mut lookup_audit {
                audit.observe(&self.cached);
            }
        }
        if profile {
            let picks = self.gm.cpu_picks();
            tracing::info!(stats = %serde_json::json!({"steps":steps,"gpu_sampling":gpu_sampling,"gpu_samples":gpu_samples,"gpu_fallbacks":gpu_fallbacks,"sample_s":sample_s,"emit_s":emit_s,"step_s":step_s,"adapt_host_s":self.gm.adapt_s-adapt_before,"model_prefill_s":model_prefill_s,"snapshot_s":snapshot_s,"prefix_lcp":lcp,"prefix_start":start,"prefix_replayed_tokens":lcp-start,"routed_picks":picks.0-picks_before.0,"cpu_picks":picks.1-picks_before.1}), "runtime stats");
        }
        if let Some(audit) = lookup_audit {
            tracing::info!(stats = %audit.report(), "lookup audit");
        }
        self.stats = GenStats { prefill: Some(prefill), decode: Some(td.elapsed()), cached_prompt_tokens: start };
        if let (Some(path),Some(output)) = (token_audit_path,output_ids) {
            use std::io::Write;
            let record = serde_json::json!({"event":"raw_generation","seed":p.seed,"temperature":p.temperature,"top_p":p.top_p,"prompt_token_ids":prompt,"output_token_ids":output,"processed_output_tokens":self.cached.len()-prompt.len(),"prefix_lcp":lcp,"prefix_start":start,"initial_residency":initial_residency,"final_residency":self.gm.validation_residency(),"gpu_sampling":gpu_sampling,"backend_finish":format!("{fin:?}"),"limit":"Raw sampled IDs include a stopping/cancelled token that may not have been processed; use processed_output_tokens when replaying KV/cache history."});
            let write = || -> std::io::Result<()> {
                let mut f=std::fs::OpenOptions::new().create(true).append(true).open(path)?;
                writeln!(f,"{record}")
            };
            if let Err(error)=write() {tracing::warn!(%error,"token audit write failed");}
        }

        let _ = Duration::ZERO;
        Ok(fin)
    }

    fn last_stats(&self) -> GenStats {
        GenStats { prefill: self.stats.prefill, decode: self.stats.decode, cached_prompt_tokens: self.stats.cached_prompt_tokens }
    }
}
