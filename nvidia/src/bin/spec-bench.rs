//! Target-only speculative upper bounds, rollback checks, and real prompt-lookup generation.
//! No learned drafter is supplied; oracle results exclude drafting and are optimistic ceilings.
use anyhow::{Result, ensure};
use clap::Parser;
use dsx::{
    gpumodel::{GpuModel, Placement},
    model::Model,
};
use dsx_core::server::sample::{Sampler, argmax};
use serde_json::{Value, json};
use std::{path::PathBuf, time::Instant};
#[derive(Parser)]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    rank: PathBuf,
    #[arg(long)]
    ids: PathBuf,
    #[arg(long)]
    continuation: Option<PathBuf>,
    #[arg(long, default_value_t = 96)]
    steps: usize,
    #[arg(long, default_value_t = 262144)]
    ctx: usize,
    #[arg(long, default_value_t = 8192)]
    chunk: usize,
    #[arg(long, default_value_t = 110.)]
    arena_gb: f64,
    #[arg(long, default_value_t=-1.5, allow_negative_numbers=true)]
    main_gb: f64,
    #[arg(long, default_value_t=-1., allow_negative_numbers=true)]
    aux_gb: f64,
    #[arg(long, default_value_t = 3)]
    rounds: usize,
    #[arg(long, default_value = "1,2,4,8", value_delimiter = ',')]
    widths: Vec<usize>,
    #[arg(long, default_value_t = 8)]
    rollback_steps: usize,
    /// Generate real sampled output with prompt-lookup proposals after the oracle checks.
    #[arg(long)]
    lookup: bool,
    /// Diagnostic only: allow generation when the batched path differs from serial decoding.
    #[arg(long, requires = "lookup")]
    allow_path_differences: bool,
    #[arg(long, default_value_t = 1.)]
    temperature: f32,
    #[arg(long, default_value_t = 1.)]
    top_p: f32,
    #[arg(long)]
    seed: Option<u64>,
}
fn ids(p: &PathBuf) -> Result<Vec<u32>> {
    Ok(std::fs::read_to_string(p)?
        .split(|c: char| c == ',' || c.is_whitespace())
        .filter(|s| !s.is_empty())
        .map(str::parse)
        .collect::<std::result::Result<_, _>>()?)
}
fn emit(x: Value) {
    println!("{x}");
}
fn compare(a: &[Vec<f32>], b: &[Vec<f32>]) -> Value {
    let mut changed = 0usize;
    let mut max_abs = 0f32;
    let mut top_changed = 0usize;
    let mut nonfinite = 0usize;
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(b) {
        assert_eq!(x.len(), y.len());
        top_changed += usize::from(argmax(x) != argmax(y));
        for (&u, &v) in x.iter().zip(y) {
            changed += usize::from(u.to_bits() != v.to_bits());
            nonfinite += usize::from(!u.is_finite() || !v.is_finite());
            max_abs = max_abs.max((u - v).abs());
        }
    }
    json!({"rows":a.len(),"changed_values":changed,"max_abs":max_abs,"top1_changed_rows":top_changed,"nonfinite_values":nonfinite})
}
fn main() -> Result<()> {
    let a = Args::parse();
    ensure!(
        a.steps > 0 && a.rounds > 0 && a.chunk >= 8,
        "nonzero steps/rounds and chunk >=8 required"
    );
    ensure!(
        a.widths.iter().all(|&x| x > 0 && x <= 8),
        "widths must be 1..=8"
    );
    ensure!(
        std::env::var("DSX_ADAPT").is_ok_and(|v| v == "0"),
        "set DSX_ADAPT=0: matched fixed residency is required"
    );
    let prompt = ids(&a.ids)?;
    let forced = a.continuation.as_ref().map(ids).transpose()?;
    ensure!(
        !prompt.is_empty() && prompt.len() + a.steps + 8 < a.ctx,
        "prompt and continuation must fit context"
    );
    ensure!(
        forced.as_ref().is_none_or(|x| x.len() >= a.steps),
        "continuation is too short"
    );
    let m = Model::load(&a.model, None)?;
    ensure!(
        prompt
            .iter()
            .chain(forced.iter().flatten())
            .all(|&x| (x as usize) < m.cfg.vocab),
        "token outside vocabulary"
    );
    let mut gm = GpuModel::with_placement(
        &m,
        -1,
        a.ctx,
        Placement {
            gpu0_gb: a.main_gb,
            aux_gb: a.aux_gb,
            arena_gb: a.arena_gb,
            prefill_chunk: a.chunk,
            rank: Some(a.rank),
        },
    )?;
    let pb = gm.pb.take().expect("prefill buffers");
    let start = Instant::now();
    let initial = gm.prefill(&pb, &prompt)?;
    gm.g.sync()?;
    emit(
        json!({"event":"prefill","tokens":prompt.len(),"seconds":start.elapsed().as_secs_f64(),"defer":std::env::var("DSX_DEFER").ok(),"adapt":0,"ctx":a.ctx,"chunk":a.chunk}),
    );
    let root = gm.snapshot()?;
    let mut logits = initial.clone();
    let mut tokens = Vec::new();
    let mut reference = Vec::new();
    for i in 0..a.steps {
        let t = forced.as_ref().map_or_else(|| argmax(&logits), |x| x[i]);
        tokens.push(t);
        logits = gm.step(t)?;
        reference.push(logits.clone());
    }
    emit(
        json!({"event":"reference","tokens":tokens,"limit":"Forced acceptance is a target-only upper bound, not speculative generation. Deferral can change target semantics across batch sizes."}),
    );
    // Warm allocation and first-use kernels; no timed mode inherits another mode's sequence state.
    gm.restore(&root)?;
    gm.verify_spec(&pb, &tokens[..8.min(tokens.len())])?;
    gm.keep(8.min(tokens.len()))?;
    let mut serial_equivalent = true;
    let mut suffix_independent = true;
    for round in 0..a.rounds {
        let mut modes = vec![0];
        modes.extend(&a.widths);
        if round % 2 == 1 {
            modes.reverse();
        }
        for width in modes {
            gm.restore(&root)?;
            gm.g.sync()?;
            if let Some(p) = gm.prof.as_mut() {
                p.clear();
            }
            if let Some(p) = gm.evprof.as_mut() {
                p.clear();
            }
            let cpu = gm.cpu_busy_s();
            let picks = gm.cpu_picks();
            let split = gm.cpu_split_s();
            let start = Instant::now();
            let mut out = Vec::new();
            let mut calls = 0;
            let mut verify_s = 0.;
            let mut keep_s = 0.;
            for ts in tokens.chunks(width.max(1)) {
                let begin = Instant::now();
                if width == 0 {
                    out.push(gm.step(ts[0])?);
                } else {
                    out.extend(gm.verify_spec(&pb, ts)?);
                }
                verify_s += begin.elapsed().as_secs_f64();
                if width > 0 {
                    let begin = Instant::now();
                    gm.keep(ts.len())?;
                    keep_s += begin.elapsed().as_secs_f64();
                }
                calls += 1;
            }
            gm.g.sync()?;
            let elapsed = start.elapsed().as_secs_f64();
            let after = gm.cpu_picks();
            let after_split = gm.cpu_split_s();
            let difference = compare(&reference, &out);
            serial_equivalent &= difference["changed_values"].as_u64() == Some(0)
                && difference["nonfinite_values"].as_u64() == Some(0);
            emit(
                json!({"event":"oracle","round":round,"width":width,"tokens":tokens.len(),"calls":calls,"seconds":elapsed,"tps":tokens.len() as f64/elapsed,"verify_s":verify_s,"keep_s":keep_s,"cpu_s":gm.cpu_busy_s()-cpu,"cpu_picks":after.1-picks.1,"routed_picks":after.0-picks.0,"cpu_split_s":[after_split[0]-split[0],after_split[1]-split[1],after_split[2]-split[2]],"difference":difference,"stage_profile":gm.prof,"event_profile":gm.evprof}),
            );
        }
    }
    // Deliberately reject different suffixes, including every retained-prefix length.
    // A discarded suffix must not influence any later logits from the same accepted history.
    for width in a
        .widths
        .iter()
        .copied()
        .filter(|&w| w > 1 && a.rollback_steps > 0)
    {
        for keep in 1..width {
            let mut prior: Option<Vec<Vec<f32>>> = None;
            for dummy in [11u32, 4242] {
                gm.restore(&root)?;
                let mut out = Vec::new();
                let mut offset = 0;
                while offset < a.rollback_steps.min(tokens.len()) {
                    let n = keep.min(a.rollback_steps.min(tokens.len()) - offset);
                    let mut block = vec![dummy; width];
                    block[..n].copy_from_slice(&tokens[offset..offset + n]);
                    let rows = gm.verify_spec(&pb, &block)?;
                    gm.keep(n)?;
                    out.extend(rows.into_iter().take(n));
                    offset += n;
                }
                // A scalar step additionally catches corrupted ring/compressor carry after the last rejection.
                if offset < tokens.len() {
                    out.push(gm.step(tokens[offset])?);
                }
                gm.g.sync()?;
                if let Some(prev) = &prior {
                    let d = compare(prev, &out);
                    suffix_independent &= d["changed_values"].as_u64() == Some(0)
                        && d["nonfinite_values"].as_u64() == Some(0);
                }
                emit(
                    json!({"event":"rollback","width":width,"kept_prefix":keep,"dummy":dummy,"difference_from_serial":compare(&reference[..out.len()],&out),"difference_between_suffixes":prior.as_ref().map(|x|compare(x,&out))}),
                );
                prior = Some(out);
            }
        }
    }
    emit(
        json!({"event":"correctness_gate","serial_bit_identical":serial_equivalent,"rejected_suffix_independent":suffix_independent}),
    );
    if a.lookup {
        ensure!(a.rollback_steps > 0, "lookup needs rollback checks enabled");
        ensure!(
            suffix_independent,
            "rejected suffix changes accepted history: generation is unsafe"
        );
        ensure!(
            serial_equivalent || a.allow_path_differences,
            "batched verifier differs from serial target; use --allow-path-differences only for diagnostics"
        );
        let seed = a.seed.unwrap_or_else(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64
        });
        for round in 0..a.rounds {
            let mut modes = vec![0];
            modes.extend(a.widths.iter().copied().filter(|&w| w > 1));
            if round % 2 == 1 {
                modes.reverse();
            }
            for width in modes {
                gm.restore(&root)?;
                gm.g.sync()?;
                let mut sampler = Sampler::new(
                    a.temperature,
                    a.top_p,
                    Some(seed.wrapping_add(round as u64)),
                );
                let mut ctx = prompt.clone();
                let mut generated = Vec::new();
                let mut next = sampler.sample(&initial);
                let mut calls = 0;
                let mut proposed = 0;
                let mut accepted = 0;
                let mut draft_s = 0.;
                let start = Instant::now();
                while generated.len() < a.steps {
                    generated.push(next);
                    ctx.push(next);
                    let begin = Instant::now();
                    let draft = if width > 1 {
                        dsx::spec::lookup(&ctx, (width - 1).min(a.steps - generated.len()))
                    } else {
                        vec![]
                    };
                    draft_s += begin.elapsed().as_secs_f64();
                    if draft.is_empty() {
                        let rows = gm.step(next)?;
                        next = sampler.sample(&rows);
                        calls += 1;
                        continue;
                    }
                    let mut block = vec![next];
                    block.extend(&draft);
                    let rows = gm.verify_spec(&pb, &block)?;
                    proposed += draft.len();
                    calls += 1;
                    let mut n = 0;
                    while n < draft.len() {
                        let (ok, t) = sampler.accept_or_resample(&rows[n], draft[n]);
                        if !ok {
                            next = t;
                            break;
                        }
                        generated.push(t);
                        ctx.push(t);
                        n += 1;
                    }
                    if n == draft.len() {
                        next = sampler.sample(rows.last().unwrap());
                    }
                    gm.keep(n + 1)?;
                    accepted += n;
                }
                gm.g.sync()?;
                let seconds = start.elapsed().as_secs_f64();
                emit(
                    json!({"event":"lookup","round":round,"width":width,"seed":seed.wrapping_add(round as u64),"temperature":a.temperature,"top_p":a.top_p,"tokens":generated.len(),"seconds":seconds,"tps":generated.len() as f64/seconds,"calls":calls,"proposed":proposed,"accepted":accepted,"draft_s":draft_s,"output_ids":generated,"limit":"Uses the batched verifier distribution; do not claim serial-target equivalence if oracle/rollback checks differ. Includes final token processing; no EOS early stop."}),
                );
            }
        }
    }
    emit(json!({"event":"done"}));
    Ok(())
}
