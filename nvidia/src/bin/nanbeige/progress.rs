//! Live output rate from actual emitted model tokens, including reasoning/tools.
use serde_json::{Value, json};
use std::{collections::VecDeque, time::{Duration, Instant}};

#[derive(Default)]
pub struct LiveProgress {
    id: String,
    prompt: usize,
    generated: usize,
    first: Option<Instant>,
    last: Option<Instant>,
    samples: VecDeque<(Instant, usize)>,
    active: bool,
    phase: &'static str,
    finish: Option<String>,
}
impl LiveProgress {
    pub fn token(&mut self, id: &str, prompt: usize, generated: usize, phase: &'static str, now: Instant) {
        if self.id != id {
            *self = Self { id: id.into(), prompt, first: Some(now), ..Self::default() };
        }
        self.generated=generated;
        self.last=Some(now);
        self.active=true;
        self.phase=phase;
        self.samples.push_back((now,generated));
        while self.samples.len()>2 && now.duration_since(self.samples[0].0)>Duration::from_secs(2) {
            self.samples.pop_front();
        }
    }
    pub fn finish(&mut self, id: &str, reason: &str) {
        if self.id==id {self.active=false;self.finish=Some(reason.into());}
    }
    pub fn snapshot(&self, pending: usize, ctx: usize, now: Instant) -> Value {
        let average = self.first.zip(self.last).and_then(|(a,b)|rate(self.generated.saturating_sub(1),b.duration_since(a)));
        let rolling = self.samples.front().zip(self.samples.back()).and_then(|((a,n),(b,m))|rate(m-n,b.duration_since(*a)));
        let stale=self.last.is_some_and(|t|now.duration_since(t)>Duration::from_secs(2));
        let phase=if self.active {self.phase} else if pending>0 {"queued_or_prefill"} else {"idle"};
        json!({"request_id":self.id,"active":self.active,"phase":phase,"pending_requests":pending,
            "prompt_tokens":self.prompt,"generated_tokens":self.generated,
            "sequence_tokens":self.prompt+self.generated,"context_capacity":ctx,
            "output_tps":if self.active && stale {Some(0.)} else {rolling},"average_output_tps":average,
            "decode_seconds":self.first.zip(self.last).map(|(a,b)|b.duration_since(a).as_secs_f64()),
            "seconds_since_token":self.last.map(|t|now.duration_since(t).as_secs_f64()),"finish_reason":self.finish,
            "measurement":"actual emitted model tokens; rolling two-second and whole-response decode rates exclude the first token; sequence_tokens is prompt plus output, not a KV allocation measurement"})
    }
}
fn rate(tokens: usize, elapsed: Duration) -> Option<f64> {
    (elapsed.as_secs_f64()>0.).then(||tokens as f64/elapsed.as_secs_f64())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rates_stalls_and_request_boundaries() {
        let t=Instant::now();let mut p=LiveProgress::default();
        assert_eq!(p.snapshot(1,262144,t)["phase"],"queued_or_prefill");
        p.token("a",100,1,"reasoning",t);
        assert!(p.snapshot(1,262144,t)["output_tps"].is_null());
        p.token("a",100,101,"tool",t+Duration::from_secs(2));
        let live=p.snapshot(1,262144,t+Duration::from_secs(2));
        assert_eq!(live["output_tps"],50.);assert_eq!(live["average_output_tps"],50.);assert_eq!(live["sequence_tokens"],201);
        assert_eq!(p.snapshot(1,262144,t+Duration::from_secs(5))["output_tps"],0.);
        p.finish("wrong","cancelled");assert_eq!(p.snapshot(1,262144,t)["active"],true);
        p.finish("a","stop");let done=p.snapshot(0,262144,t+Duration::from_secs(9));
        assert_eq!(done["phase"],"idle");assert_eq!(done["average_output_tps"],50.);
        p.token("b",500,1,"content",t+Duration::from_secs(10));
        let next=p.snapshot(1,262144,t+Duration::from_secs(10));
        assert_eq!(next["generated_tokens"],1);assert!(next["average_output_tps"].is_null());assert!(next["finish_reason"].is_null());
    }
}
