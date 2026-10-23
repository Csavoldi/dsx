//! The batched runner: many sequences on one MI325X with continuous batching (dsx-core's BatchEngine).
//!
//! Every sequence lives in a slot of a `Batch` (packed FP4 state, kernels/batch.h) or, while it is the only one
//! decoding or while its prompt is prefilled, in the single-sequence decoder (the "lane"). Each step:
//!   1. one prompt chunk (DSXR_CHUNK tokens, default 2048) of the oldest waiting prompt runs in the lane; a prompt
//!      that extends a finished slot's history (prefix reuse) starts from that slot's state
//!   2. every decoding sequence advances one token: batched (`batch_step`) when two or more decode or a prompt is
//!      waiting, else in the lane on the decode graph
//! A finished sequence keeps its slot's history until the slot is needed again (least recently used first).

use crate::batch::Batch;
use crate::decode::Decoder;
use anyhow::Result;
use dsx_core::server::engine::{BatchEngine, GenParams};
use dsx_core::server::sample::Sampler;
use std::collections::{HashMap, VecDeque};

enum Phase {
    /// waiting for / in the lane's prefill: prompt tokens not yet fed (from `done`)
    Prefill { prompt: Vec<u32>, done: usize },
    /// decoding: `next` is the newest token (emitted, not yet fed)
    Decode { next: u32 },
}

struct Seq {
    slot: usize,
    phase: Phase,
    /// tokens fed so far (the slot's history once parked)
    fed: Vec<u32>,
    sampler: Option<Sampler>,
    /// first stop token: emitted to end a sequence at the context limit
    stop: Option<u32>,
    cached: usize,
}

struct Slot {
    owner: Option<u64>,
    /// history of the state held in the slot (valid when `owner` is None and `parked`)
    tokens: Vec<u32>,
    parked: bool,
    used: u64,
}

pub struct Runner<'a> {
    dec: &'a mut Decoder,
    b: Batch,
    seqs: HashMap<u64, Seq>,
    slots: Vec<Slot>,
    /// sequences waiting for the lane's prefill, oldest first
    waiting: VecDeque<u64>,
    /// the sequence whose state is in the lane (prefilling, or decoding alone)
    lane: Option<u64>,
    chunk: usize,
    clock: u64,
}

// the decoder is created and used only on the engine thread
unsafe impl Send for Runner<'_> {}

impl<'a> Runner<'a> {
    pub fn new(dec: &'a mut Decoder, slots: usize, ctx: usize) -> Result<Runner<'a>> {
        anyhow::ensure!(ctx <= dec.max_ctx, "batch context {ctx} > lane context {}", dec.max_ctx);
        let b = dec.new_batch(slots, ctx)?;
        let chunk = std::env::var("DSXR_CHUNK").ok().and_then(|v| v.parse().ok()).unwrap_or(2048);
        let slots = (0..slots).map(|_| Slot { owner: None, tokens: vec![], parked: false, used: 0 }).collect();
        Ok(Runner { dec, b, seqs: HashMap::new(), slots, waiting: VecDeque::new(), lane: None, chunk, clock: 0 })
    }

    /// Move the lane's decoding sequence into its slot.
    fn park_lane(&mut self) -> Result<()> {
        if let Some(id) = self.lane.take() {
            let s = &self.seqs[&id];
            if matches!(s.phase, Phase::Decode { .. }) {
                self.dec.load_slot(&mut self.b, s.slot)?;
                self.slots[s.slot].parked = true;
            }
        }
        Ok(())
    }

    fn pick(&mut self, s: &mut Seq, logits: impl FnOnce(&Decoder) -> Result<Vec<f32>>, greedy: u32) -> Result<u32> {
        match s.sampler.as_mut() {
            Some(sm) => Ok(sm.sample(&logits(self.dec)?)),
            None => Ok(greedy),
        }
    }
}

impl BatchEngine for Runner<'_> {
    fn has_room(&self) -> bool {
        self.slots.iter().any(|s| s.owner.is_none())
    }

    fn start(&mut self, id: u64, prompt: &[u32], p: &GenParams) -> Result<()> {
        anyhow::ensure!(!prompt.is_empty(), "empty prompt");
        anyhow::ensure!(prompt.len() + 1 < self.b.ctx, "prompt of {} tokens does not fit the context of {}", prompt.len(), self.b.ctx);
        // a free slot whose history the prompt extends (longest), else the least recently used free slot
        let mut best: Option<(usize, usize)> = None;
        for (i, sl) in self.slots.iter().enumerate() {
            if sl.owner.is_some() {
                continue;
            }
            let reuse = if sl.parked && sl.tokens.len() < prompt.len() && prompt.starts_with(&sl.tokens) { sl.tokens.len() } else { 0 };
            let better = match best {
                None => true,
                Some((bi, br)) => reuse > br || (reuse == br && sl.used < self.slots[bi].used),
            };
            if better {
                best = Some((i, reuse));
            }
        }
        let (slot, reuse) = best.ok_or_else(|| anyhow::anyhow!("no free slot"))?;
        self.clock += 1;
        let sl = &mut self.slots[slot];
        sl.owner = Some(id);
        sl.used = self.clock;
        if reuse == 0 {
            sl.parked = false;
            sl.tokens.clear();
        }
        let sampler = if p.temperature > 0.0 { Some(Sampler::from_params(p)) } else { None };
        self.seqs.insert(id, Seq { slot, phase: Phase::Prefill { prompt: prompt.to_vec(), done: reuse }, fed: prompt[..reuse].to_vec(), sampler, stop: p.stop_token_ids.first().copied(), cached: reuse });
        self.waiting.push_back(id);
        Ok(())
    }

    fn step(&mut self) -> Result<Vec<(u64, u32)>> {
        let mut out = vec![];
        // ---- 1. prefill: one chunk of the oldest waiting prompt in the lane
        if let Some(&wid) = self.waiting.front() {
            if self.lane != Some(wid) {
                self.park_lane()?;
                let (slot, done) = match &self.seqs[&wid].phase {
                    Phase::Prefill { done, .. } => (self.seqs[&wid].slot, *done),
                    _ => unreachable!(),
                };
                if done > 0 {
                    self.dec.unload_slot(&self.b, slot)?;
                } else {
                    self.dec.reset()?;
                }
                self.lane = Some(wid);
            }
            let s = self.seqs.get_mut(&wid).unwrap();
            let Phase::Prefill { prompt, done } = &mut s.phase else { unreachable!() };
            let end = (*done + self.chunk).min(prompt.len());
            let next = self.dec.prefill(&prompt[*done..end])?;
            s.fed.extend_from_slice(&prompt[*done..end]);
            *done = end;
            if end == prompt.len() {
                self.waiting.pop_front();
                let mut s = self.seqs.remove(&wid).unwrap();
                let tok = self.pick(&mut s, |d| d.logits(), next)?;
                s.phase = Phase::Decode { next: tok };
                self.seqs.insert(wid, s);
                out.push((wid, tok));
                // stays in the lane; parked below if others decode or prompts wait
            }
        }
        // ---- 2. decode: every sequence with a token to feed
        let mut dec_ids: Vec<u64> = self.seqs.iter().filter(|(_, s)| matches!(s.phase, Phase::Decode { .. })).map(|(&id, _)| id).collect();
        // a sequence that just produced its first token is fed from the next step on
        dec_ids.retain(|id| !out.iter().any(|(o, _)| o == id));
        if dec_ids.is_empty() {
            return Ok(out);
        }
        // context limit: end with the stop token
        dec_ids.retain(|id| {
            let s = &self.seqs[id];
            if s.fed.len() + 2 >= self.b.ctx {
                if let Some(st) = s.stop {
                    out.push((*id, st));
                }
                false
            } else {
                true
            }
        });
        // a lone decoding sequence runs in the lane (the decode graph is ~2.5x faster than a batch of one)
        if dec_ids.len() == 1 && self.waiting.is_empty() && self.lane != Some(dec_ids[0]) {
            self.park_lane()?;
            let slot = self.seqs[&dec_ids[0]].slot;
            self.dec.unload_slot(&self.b, slot)?;
            self.lane = Some(dec_ids[0]);
        }
        let solo = dec_ids.len() == 1 && self.waiting.is_empty() && self.lane == Some(dec_ids[0]);
        if solo {
            let id = dec_ids[0];
            let mut s = self.seqs.remove(&id).unwrap();
            let Phase::Decode { next } = s.phase else { unreachable!() };
            let g = self.dec.step(next)?;
            s.fed.push(next);
            let tok = self.pick(&mut s, |d| d.logits(), g)?;
            s.phase = Phase::Decode { next: tok };
            self.seqs.insert(id, s);
            out.push((id, tok));
            return Ok(out);
        }
        // batched: the lane's decoding sequence (if any) joins the slots
        if let Some(l) = self.lane {
            if matches!(self.seqs[&l].phase, Phase::Decode { .. }) {
                self.park_lane()?;
            }
        }
        dec_ids.sort();
        let slots: Vec<usize> = dec_ids.iter().map(|id| self.seqs[id].slot).collect();
        let toks: Vec<u32> = dec_ids.iter().map(|id| match self.seqs[id].phase { Phase::Decode { next } => next, _ => unreachable!() }).collect();
        for chunk in (0..dec_ids.len()).collect::<Vec<_>>().chunks(crate::batch::BMAX) {
            let cs: Vec<usize> = chunk.iter().map(|&i| slots[i]).collect();
            let ct: Vec<u32> = chunk.iter().map(|&i| toks[i]).collect();
            let greedy = self.dec.batch_step(&mut self.b, &cs, &ct)?;
            for (j, &i) in chunk.iter().enumerate() {
                let id = dec_ids[i];
                let mut s = self.seqs.remove(&id).unwrap();
                s.fed.push(ct[j]);
                let b = &self.b;
                let tok = match s.sampler.as_mut() {
                    Some(sm) => sm.sample(&self.dec.batch_logits(b, j)?),
                    None => greedy[j],
                };
                s.phase = Phase::Decode { next: tok };
                self.seqs.insert(id, s);
                out.push((id, tok));
            }
        }
        Ok(out)
    }

    fn finish(&mut self, id: u64) {
        let Some(s) = self.seqs.remove(&id) else { return };
        self.waiting.retain(|&w| w != id);
        let decoding = matches!(s.phase, Phase::Decode { .. });
        // keep a decoded sequence's state for prefix reuse (its history = the tokens fed)
        let mut parked = self.slots[s.slot].parked && decoding;
        if self.lane == Some(id) {
            self.lane = None;
            parked = decoding && self.dec.load_slot(&mut self.b, s.slot).is_ok();
        }
        let sl = &mut self.slots[s.slot];
        sl.owner = None;
        sl.parked = parked;
        sl.tokens = if parked { s.fed } else { vec![] };
    }

    fn cached_prompt_tokens(&self, id: u64) -> usize {
        self.seqs.get(&id).map(|s| s.cached).unwrap_or(0)
    }
}
