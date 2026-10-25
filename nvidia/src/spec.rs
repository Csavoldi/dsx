//! Speculative decoding drafts that cost nothing to make: continuations looked up in the context itself.

/// Up to `k` tokens that followed the most recent earlier occurrence of the context's last 3 (else 2) tokens.
pub fn lookup(ctx: &[u32], k: usize) -> Vec<u32> {
    for n in [3usize, 2] {
        if ctx.len() <= n {
            continue;
        }
        let tail = &ctx[ctx.len() - n..];
        // newest match first; the match must leave at least one token after it
        for s in (0..ctx.len() - n).rev() {
            if &ctx[s..s + n] == tail {
                let from = s + n;
                let to = (from + k).min(ctx.len());
                if to > from {
                    return ctx[from..to].to_vec();
                }
            }
        }
    }
    vec![]
}

#[cfg(test)]
mod tests {
    #[test]
    fn finds_the_newest_continuation() {
        let c = [1, 2, 3, 9, 9, 1, 2, 3, 7, 8, 5, 1, 2, 3];
        assert_eq!(super::lookup(&c, 2), vec![7, 8]);
        assert_eq!(super::lookup(&[4, 5, 6], 3), Vec::<u32>::new());
    }
}

/// A bounded, causal copy proposal. Require a longer suffix than the CLI's permissive lookup;
/// all proposed IDs must already occur in the recorded context.
pub fn lookup_strict(ctx: &[u32], k: usize, suffix: usize, window: usize) -> Vec<u32> {
    if suffix == 0 || ctx.len() <= suffix || k == 0 {
        return vec![];
    }
    let tail = &ctx[ctx.len() - suffix..];
    let end = ctx.len() - suffix;
    for start in (end.saturating_sub(window)..end).rev() {
        let from = start + suffix;
        if from + k <= ctx.len() && &ctx[start..from] == tail {
            return ctx[from..from + k].to_vec();
        }
    }
    vec![]
}

/// Observe serial sampled output without changing it or consuming random numbers. A block's
/// known seed token proposes the next three tokens, corresponding to a width-four verification.
/// This measures copy opportunity; it does not run verification or prove speculative speedup.
pub struct LookupAudit {
    proposal: Vec<u32>,
    matched: usize,
    cooldown: bool,
    tokens: usize,
    attempts: usize,
    completed: usize,
    accepted_prefix: [usize; 4],
    elapsed: std::time::Duration,
}

impl LookupAudit {
    pub fn from_env() -> Option<Self> {
        (std::env::var("DSX_LOOKUP_AUDIT").is_ok_and(|v| v == "1")).then(|| Self {
            proposal: vec![], matched: 0, cooldown: false, tokens: 0, attempts: 0,
            completed: 0, accepted_prefix: [0; 4], elapsed: std::time::Duration::ZERO,
        })
    }

    /// Context includes the serial token that was just processed. Lookups use only this context.
    pub fn observe(&mut self, ctx: &[u32]) {
        let start = std::time::Instant::now();
        self.tokens += 1;
        if !self.proposal.is_empty() {
            if ctx.last() == self.proposal.get(self.matched) {
                self.matched += 1;
                if self.matched == self.proposal.len() {
                    self.accepted_prefix[self.matched] += 1;
                    self.completed += 1;
                    self.proposal.clear();
                    // A full block also emits a normally sampled bonus token; do not overlap it
                    // with the next proposed block when counting the conservative opportunity.
                    self.cooldown = true;
                }
            } else {
                self.accepted_prefix[self.matched] += 1;
                self.completed += 1;
                self.proposal.clear();
            }
        } else if self.cooldown {
            self.cooldown = false;
        } else {
            self.proposal = lookup_strict(ctx, 3, 8, 8192);
            if !self.proposal.is_empty() {
                self.attempts += 1;
                self.matched = 0;
            }
        }
        self.elapsed += start.elapsed();
    }

    pub fn report(&self) -> serde_json::Value {
        let accepted: usize = self.accepted_prefix.iter().enumerate().map(|(i, n)| i * n).sum();
        serde_json::json!({
            "tokens": self.tokens, "attempts": self.attempts, "completed_blocks": self.completed,
            "incomplete_blocks": self.attempts - self.completed, "accepted_prefix_histogram": self.accepted_prefix,
            "matched_future_tokens": accepted,
            "mean_matched_future_tokens": if self.completed > 0 { accepted as f64 / self.completed as f64 } else { 0.0 },
            "audit_seconds": self.elapsed.as_secs_f64(), "suffix_tokens": 8, "search_window": 8192,
            "verification_width": 4,
            "limit": "Serial sampled-path copy audit only; no verification, drafter or speculative speedup measured."
        })
    }
}

#[cfg(test)]
mod audit_tests {
    use super::*;
    fn audit() -> LookupAudit {
        LookupAudit { proposal: vec![], matched: 0, cooldown: false, tokens: 0, attempts: 0, completed: 0, accepted_prefix: [0;4], elapsed: std::time::Duration::ZERO }
    }
    fn context() -> Vec<u32> {
        [vec![1,2,3,4,5,6,7,8,90,91,92,99],vec![1,2,3,4,5,6,7,8]].concat()
    }
    #[test]
    fn strict_lookup_requires_complete_past_continuation() {
        assert_eq!(lookup_strict(&context(),3,8,8192),vec![90,91,92]);
        assert!(lookup_strict(&[1,2,3,4,5,6,7,8],3,8,8192).is_empty());
        assert!(lookup_strict(&context(),3,8,1).is_empty());
        assert!(lookup_strict(&context(),30,8,8192).is_empty());
    }
    #[test]
    fn audit_counts_full_and_rejected_prefixes() {
        for (future, expected) in [(vec![90,91,92],3),(vec![90,999],1),(vec![999],0)] {
            let mut a=audit();let mut ctx=context();a.observe(&ctx);
            for token in future {ctx.push(token);a.observe(&ctx);}
            assert_eq!(a.attempts,1);assert_eq!(a.completed,1);assert_eq!(a.accepted_prefix[expected],1);
            assert!(a.proposal.is_empty());
        }
    }
}
