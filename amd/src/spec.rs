//! Speculative decoding for batched sessions: per session, drafts come from its own context (prompt lookup: the
//! tokens that followed the last earlier occurrence of its newest 3 tokens) or, while they pay off, from DSpark;
//! every session's next token and drafts are verified in one batched step and the accepted prefix is committed.
//! A session with no drafts feeds one token, as plain decoding does.

use crate::batch::Batch;
use crate::decode::Decoder;
use anyhow::Result;
use std::collections::HashMap;

/// most drafts verified per session per step (slot rings hold 16 spare rows)
pub const MAX_DRAFTS: usize = 8;

pub struct SpecState {
    /// the session's tokens (prompt + generated, including the pending `next`)
    toks: Vec<u32>,
    /// newest earlier position following each 3-gram: (a, b, c) -> index of the token after it
    ngram: HashMap<(u32, u32, u32), usize>,
    /// running mean of DSpark drafts accepted per pass
    pub ema: f64,
    /// running mean of prompt-lookup drafts accepted per pass (sets the lookup draft length)
    pub lema: f64,
    pub passes: usize,
    pub accepted: usize,
    pub ngram_hits: usize,
    /// sampled (temperature > 0) sessions get no drafts and sample their token from the logits
    pub greedy: bool,
}

impl SpecState {
    pub fn new(history: &[u32]) -> SpecState {
        let mut s = SpecState { toks: vec![], ngram: HashMap::new(), ema: 2.0, lema: 4.0, passes: 0, accepted: 0, ngram_hits: 0, greedy: true };
        s.push(history);
        s
    }

    pub fn push(&mut self, new: &[u32]) {
        for &t in new {
            let n = self.toks.len();
            if n >= 3 {
                // the 3-gram ending at n-1 is followed by the token at n
                self.ngram.insert((self.toks[n - 3], self.toks[n - 2], self.toks[n - 1]), n);
            }
            self.toks.push(t);
        }
    }

    /// the newest `n` tokens
    pub fn tail(&self, n: usize) -> Vec<u32> {
        self.toks[self.toks.len().saturating_sub(n)..].to_vec()
    }

    /// up to `max` tokens that followed the newest earlier occurrence of the last 3 tokens
    fn lookup(&self, max: usize) -> Vec<u32> {
        let n = self.toks.len();
        if n < 4 {
            return vec![];
        }
        match self.ngram.get(&(self.toks[n - 3], self.toks[n - 2], self.toks[n - 1])) {
            Some(&i) if i < n => self.toks[i..n.min(i + max)].to_vec(),
            _ => vec![],
        }
    }
}

/// DSpark is used for a session while its acceptance average is at least this (DSXS_MIN, default 2.0: at lower
/// acceptance a draft pass plus the longer verify costs more than it returns once a few sessions share a step)
fn dspark_min() -> f64 {
    std::env::var("DSXS_MIN").ok().and_then(|v| v.parse().ok()).unwrap_or(2.0)
}

impl Decoder {
    /// One speculative step for slots[i] whose next (emitted, not yet fed) token is nexts[i]. Returns, per slot, the
    /// tokens it emits (accepted drafts, then its new next token); st[i].push gets the same tokens.
    pub fn spec_step(&mut self, b: &mut Batch, slots: &[usize], nexts: &[u32], st: &mut [&mut SpecState]) -> Result<Vec<Vec<u32>>> {
        self.spec_step_s(b, slots, nexts, st, &mut |_, _| 0)
    }

    /// spec_step with `sample(i, logits)` choosing the token of a sampled session i (no drafts)
    pub fn spec_step_s(&mut self, b: &mut Batch, slots: &[usize], nexts: &[u32], st: &mut [&mut SpecState], sample: &mut dyn FnMut(usize, &[f32]) -> u32) -> Result<Vec<Vec<u32>>> {
        let n = slots.len();
        let rooms: Vec<usize> = slots.iter().map(|&s| (b.ctx - b.pos[s] - 2).min(MAX_DRAFTS)).collect();
        // drafts: prompt lookup first, else DSpark while it pays off (a session below the bar still probes it now and then)
        // lookup draft length follows its acceptance (a wrong guess still lengthens the verify)
        let mut drafts: Vec<Vec<u32>> = (0..n).map(|i| if st[i].greedy { st[i].lookup(rooms[i].min((st[i].lema * 1.5 + 1.0).round().max(1.0) as usize)) } else { vec![] }).collect();
        let mut from_lookup: Vec<bool> = drafts.iter().map(|d| !d.is_empty()).collect();
        let want: Vec<usize> = if self.dspark.is_some() {
            (0..n).filter(|&i| st[i].greedy && drafts[i].is_empty() && rooms[i] > 0 && (st[i].ema >= dspark_min() || st[i].passes % 32 == 31)).collect()
        } else {
            vec![]
        };
        if !want.is_empty() {
            let ws: Vec<usize> = want.iter().map(|&i| slots[i]).collect();
            let wn: Vec<u32> = want.iter().map(|&i| nexts[i]).collect();
            let d = self.batch_draft(b, &ws, &wn)?;
            for (k, &i) in want.iter().enumerate() {
                // draft length follows the acceptance average
                let len = ((st[i].ema + 1.5).round() as usize).clamp(1, d[k].len()).min(rooms[i]);
                drafts[i] = d[k][..len].to_vec();
                from_lookup[i] = false;
            }
        }
        // verify every session's next token and drafts at once
        let feeds: Vec<Vec<u32>> = (0..n).map(|i| {
            let mut f = vec![nexts[i]];
            f.extend_from_slice(&drafts[i]);
            f
        }).collect();
        let pre: Vec<usize> = slots.iter().map(|&s| b.pos[s]).collect();
        let outs = self.batch_verify(b, slots, &feeds)?;
        let (mut rows, mut opos, mut oslot) = (vec![], vec![], vec![]);
        let mut emitted = vec![];
        let mut row = 0;
        for i in 0..n {
            let f = &feeds[i];
            let mut acc = 0;
            while acc + 1 < f.len() && outs[i][acc] == f[acc + 1] {
                acc += 1;
            }
            self.batch_commit(b, slots[i], row, f, acc + 1)?;
            for j in 0..=acc {
                rows.push((row + j) as i32);
                opos.push((pre[i] + j) as i32);
                oslot.push(slots[i] as i32);
            }
            let mut e = f[1..=acc].to_vec();
            e.push(if st[i].greedy { outs[i][acc] } else { sample(i, &self.batch_logits(b, row)?) });
            st[i].push(&e);
            if f.len() > 1 {
                st[i].passes += 1;
                st[i].accepted += acc;
                if from_lookup[i] {
                    st[i].ngram_hits += acc;
                    st[i].lema = 0.6 * st[i].lema + 0.4 * acc as f64;
                } else {
                    st[i].ema = 0.6 * st[i].ema + 0.4 * acc as f64;
                }
            } else {
                st[i].passes += 1;
            }
            emitted.push(e);
            row += f.len();
        }
        if self.dspark.is_some() {
            let t: usize = feeds.iter().map(|f| f.len()).sum();
            self.batch_observe(b, t, &rows, &opos, &oslot)?;
        }
        Ok(emitted)
    }
}
