//! Fixed-token verification upper bounds and rollback diagnostics. Does not run a draft model.
use anyhow::{Result, ensure};
use clap::Parser;
use dsx::{
    gpumodel::{GpuModel, Placement},
    model::Model,
};
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
    /// Consume these exact continuation IDs rather than generating a different path for each placement.
    #[arg(long)]
    continuation: Option<PathBuf>,
    #[arg(long, default_value_t = 128)]
    steps: usize,
    #[arg(long, default_value_t = 131072)]
    ctx: usize,
    #[arg(long, default_value_t = 4096)]
    chunk: usize,
    #[arg(long, default_value_t = 104.0)]
    arena_gb: f64,
    #[arg(long, default_value_t=-1.5, allow_negative_numbers=true)]
    main_gb: f64,
    #[arg(long, default_value_t=-1.0, allow_negative_numbers=true)]
    aux_gb: f64,
    #[arg(long, default_value_t = 3)]
    rounds: usize,
    #[arg(long, default_value = "1,2,3,4,6,8", value_delimiter = ',')]
    widths: Vec<usize>,
    #[arg(long, default_value_t = 16)]
    rollback_steps: usize,
    #[arg(long, default_value_t = 16)]
    warmup: usize,
    /// Optional Nsight range: "step"/"verifyN" with --rounds 1, or "warm-prefill" with --prefill-probe --rounds >= 2.
    #[arg(long)]
    trace_path: Option<String>,
    /// Save the serial reference logits for a cross-placement comparison.
    #[arg(long)]
    dump_reference: Option<PathBuf>,
    /// Compare the same forced continuation against another placement's serial logits.
    #[arg(long)]
    compare_reference: Option<PathBuf>,
    /// Run one fixed-token stream without restoring state, suitable for profiling the adaptive cache policy.
    #[arg(long)]
    serial_probe: bool,
    /// Diagnostic sampling during serial replay, with full logits and residency readback.
    #[arg(long, value_parser=["cpu","gpu"], requires="serial_probe")]
    replay_sampling: Option<String>,
    /// Reuse within a diagnostic comparison; independent gauntlets still use fresh seeds.
    #[arg(long, requires="replay_sampling")]
    replay_seed: Option<u64>,
    /// Warm with the exact continuation, then freeze placement for matched optimistic upper bounds.
    #[arg(long)]
    warm_cache_oracle: bool,
    /// Compare CPU sampling and device sampling on identical forced IDs and frozen placement.
    #[arg(long)]
    sampler_probe: bool,
    /// Time full and incremental prefill at fixed residency, then a forced decode continuation.
    #[arg(long)]
    prefill_probe: bool,
}

fn top(x: &[f32]) -> u32 {
    x.iter()
        .enumerate()
        .fold(
            (0, f32::NEG_INFINITY),
            |a, (i, &v)| if v > a.1 { (i, v) } else { a },
        )
        .0 as u32
}
fn hash_logits(rows: &[Vec<f32>]) -> String {
    // Stable diagnostic fingerprint; complete source/binary receipts use SHA-256 outside this process.
    let mut h = 0xcbf29ce484222325u64;
    for row in rows {
        for x in row {
            for b in x.to_bits().to_le_bytes() {
                h = (h ^ u64::from(b)).wrapping_mul(0x100000001b3);
            }
        }
    }
    format!("{h:016x}")
}
fn difference(a: &[Vec<f32>], b: &[Vec<f32>]) -> Value {
    let (mut changed, mut values, mut nonfinite, mut max_abs, mut top_changes) =
        (0usize, 0usize, 0usize, 0f64, 0usize);
    let (mut kl_sum, mut kl_max) = (0f64, 0f64);
    let mut changed_rows = Vec::new();
    for (row, (x, y)) in a.iter().zip(b).enumerate() {
        top_changes += usize::from(top(x) != top(y));
        let prior_changed = changed;
        for (&u, &v) in x.iter().zip(y) {
            values += 1;
            changed += usize::from(u.to_bits() != v.to_bits());
            if !u.is_finite() || !v.is_finite() {
                nonfinite += 1;
            } else {
                max_abs = max_abs.max((u as f64 - v as f64).abs());
            }
        }
        let mut row_kl = None;
        if changed != prior_changed && x.iter().chain(y).all(|v| v.is_finite()) {
            let mx = x.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
            let my = y.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
            let sx = x.iter().map(|&v| (v as f64 - mx).exp()).sum::<f64>();
            let sy = y.iter().map(|&v| (v as f64 - my).exp()).sum::<f64>();
            let lzx = mx + sx.ln();
            let lzy = my + sy.ln();
            let kl = x
                .iter()
                .zip(y)
                .map(|(&u, &v)| (u as f64 - lzx).exp() * (u as f64 - v as f64 + lzy - lzx))
                .sum::<f64>()
                .max(0.);
            kl_sum += kl;
            kl_max = kl_max.max(kl);
            row_kl = Some(kl);
        }
        if changed != prior_changed {changed_rows.push(json!({"row":row,"changed_values":changed-prior_changed,"top1_changed":top(x)!=top(y),"kl_reference_to_candidate":row_kl}));}
    }
    json!({"rows":a.len().min(b.len()),"values":values,"changed_bits_values":changed,"nonfinite_values":nonfinite,"max_abs_logit_difference":max_abs,"top1_changed_rows":top_changes,"mean_full_vocab_kl_reference_to_candidate":kl_sum/a.len().min(b.len()).max(1) as f64,"max_row_full_vocab_kl":kl_max,"changed_rows":changed_rows})
}
fn emit(x: Value) {
    println!("{}", x);
}
fn reference_io(a: &Args, rows: &[Vec<f32>]) -> Result<()> {
    if let Some(path) = &a.dump_reference {
        use std::io::Write;
        let mut out = std::io::BufWriter::new(std::fs::File::create(path)?);
        for row in rows { for value in row { out.write_all(&value.to_le_bytes())?; } }
        out.flush()?;
    }
    if let Some(path) = &a.compare_reference {
        let bytes = std::fs::read(path)?;
        let cols = rows.first().ok_or_else(|| anyhow::anyhow!("empty comparison"))?.len();
        ensure!(bytes.len() >= rows.len()*cols*4, "comparison reference too short");
        let values: Vec<f32> = bytes.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
        let prior: Vec<Vec<f32>> = values.chunks_exact(cols).take(rows.len()).map(|r|r.to_vec()).collect();
        emit(json!({"event":"placement_reference_comparison","reference":path,"difference":difference(&prior,rows)}));
    }
    Ok(())
}
fn read_ids(path: &std::path::Path) -> Result<Vec<u32>> {
    Ok(std::fs::read_to_string(path)?
        .split(|c: char| c == ',' || c.is_whitespace())
        .filter(|s| !s.is_empty())
        .map(str::parse)
        .collect::<std::result::Result<_, _>>()?)
}
fn main() -> Result<()> {
    let a = Args::parse();
    ensure!(
        a.steps > 0 && a.widths.iter().all(|&n| (1..=8).contains(&n)),
        "steps > 0 and widths 1..=8 required"
    );
    ensure!(
        match a.trace_path.as_deref() {None => true, Some("warm-prefill") => a.prefill_probe && a.rounds >= 2, Some(_) => !a.prefill_probe && a.rounds == 1},
        "decode trace capture requires one round; warm-prefill requires --prefill-probe and at least two rounds"
    );
    let ids = read_ids(&a.ids)?;
    let forced = a.continuation.as_deref().map(read_ids).transpose()?;
    ensure!(
        forced.as_ref().is_none_or(|x| x.len() >= a.steps),
        "continuation file must contain at least steps IDs"
    );
    ensure!(
        !ids.is_empty() && ids.len() + a.warmup + a.steps + 8 < a.ctx,
        "prompt and continuation must fit context"
    );
    let m = Model::load(&a.model, None)?;
    let mut gm = GpuModel::with_placement(
        &m,
        -1,
        a.ctx,
        Placement {
            gpu0_gb: a.main_gb,
            aux_gb: a.aux_gb,
            arena_gb: a.arena_gb,
            prefill_chunk: a.chunk,
            rank: Some(a.rank.clone()),
        },
    )?;
    let pb = gm.pb.take().expect("prefill buffers");
    if a.prefill_probe {
        emit(json!({"event":"prefill_residency","resident_experts":gm.validation_residency(),"adaptive":false,"limit":"Recorded before timing; compare full membership across arms."}));
        ensure!(std::env::var("DSX_ADAPT").is_ok_and(|v| v == "0"), "--prefill-probe requires DSX_ADAPT=0 for fixed residency");
        let continuation = forced.as_ref().ok_or_else(|| anyhow::anyhow!("--prefill-probe requires --continuation"))?;
        let mut sizes = vec![ids.len(), 256.min(ids.len()), 64.min(ids.len())];
        sizes.dedup();
        for round in 0..a.rounds {
            for &fresh in &sizes {
                gm.reset();
                let prefix = ids.len() - fresh;
                if prefix > 0 { let _ = gm.prefill(&pb, &ids[..prefix])?; }
                gm.g.sync()?;
                let tracing = round == 1 && fresh == ids.len() && a.trace_path.as_deref() == Some("warm-prefill");
                if tracing {unsafe {dsx::gpu::check(cudarc::driver::sys::cuProfilerStart(), "cuProfilerStart")?;}}
                let start = Instant::now();
                let logits = gm.prefill(&pb, &ids[prefix..])?;
                gm.g.sync()?;
                let elapsed = start.elapsed().as_secs_f64();
                if tracing {unsafe {dsx::gpu::check(cudarc::driver::sys::cuProfilerStop(), "cuProfilerStop")?;}}
                emit(json!({"event":"prefill_probe","profiled":tracing,"round":round,"prefix_tokens":prefix,"new_tokens":fresh,"elapsed_s":elapsed,"new_tok_s":fresh as f64/elapsed,"top1":top(&logits),"nonfinite_logits":logits.iter().filter(|v| !v.is_finite()).count(),"logit_fingerprint":hash_logits(&[logits])}));
                let picks = gm.cpu_picks();
                let start = Instant::now();
                let mut result = Vec::with_capacity(a.steps);
                for &token in &continuation[..a.steps] { result.push(gm.step(token)?); }
                gm.g.sync()?;
                let elapsed = start.elapsed().as_secs_f64();
                let after = gm.cpu_picks();
                emit(json!({"event":"prefill_decode_probe","round":round,"prefix_tokens":prefix,"new_tokens":fresh,"tokens":a.steps,"elapsed_s":elapsed,"tok_s":a.steps as f64/elapsed,"cpu_picks":after.1-picks.1,"routed_picks":after.0-picks.0,"logit_fingerprint":hash_logits(&result)}));
            }
        }
        emit(json!({"event":"done"}));
        return Ok(());
    }
    let start = Instant::now();
    let mut lg = gm.prefill(&pb, &ids)?;
    emit(json!({"event":"prefill","tokens":ids.len(),"elapsed_s":start.elapsed().as_secs_f64()}));
    for _ in 0..a.warmup {
        let next = top(&lg);
        lg = gm.step(next)?;
    }
    if a.serial_probe {
        let tokens = forced
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("--serial-probe requires --continuation"))?;
        if let Some(p) = gm.prof.as_mut() {
            p.clear();
        }
        if let Some(p) = gm.evprof.as_mut() {
            p.clear();
        }
        let cpu = gm.cpu_busy_s();
        let picks = gm.cpu_picks();
        let start = Instant::now();
        let mut result = Vec::with_capacity(a.steps);
        let seed = a.replay_seed.unwrap_or_else(|| std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos() as u64);
        let mut sampler = dsx_core::server::sample::Sampler::new(1.0,1.0,Some(seed));
        let mut sampled = Vec::new();
        if a.replay_sampling.is_some() {result.push(lg.clone());sampled.push(sampler.sample(&lg));}
        let mut fallbacks = 0usize;
        let mut residency = if a.replay_sampling.is_some() {gm.validation_residency()} else {vec![]};
        if a.replay_sampling.is_some() {emit(json!({"event":"replay_initial_residency","stores":residency,"limit":"Diagnostic readback and route logging alter timing; not production throughput."}));}
        for &tk in &tokens[..a.steps] {
            let logits = if a.replay_sampling.as_deref()==Some("gpu") {
                let saved = sampler.clone();
                match gm.step_sampled(tk,sampler.uniform_bits())? {
                    dsx::sampling::StepOutput::Token(next) => {sampled.push(next);gm.validation_logits()?},
                    dsx::sampling::StepOutput::Logits(logits) => {sampler=saved;fallbacks+=1;sampled.push(sampler.sample(&logits));logits},
                }
            } else {
                let logits=gm.step(tk)?;
                if a.replay_sampling.is_some() {sampled.push(sampler.sample(&logits));}
                logits
            };
            result.push(logits);
            if a.replay_sampling.is_some() {
                let after=gm.validation_residency();
                let changes:Vec<_>=residency.iter().zip(&after).enumerate().filter_map(|(store,(before,after))| {
                    let added:Vec<_>=after.iter().filter(|x|before.binary_search(x).is_err()).copied().collect();
                    let removed:Vec<_>=before.iter().filter(|x|after.binary_search(x).is_err()).copied().collect();
                    (!added.is_empty() || !removed.is_empty()).then(||json!({"store":store,"added":added,"removed":removed}))
                }).collect();
                emit(json!({"event":"replay_step","logit_row":result.len()-1,"forced_token":tk,"sampled_token":sampled.last(),"placement_changes":changes}));
                residency=after;
            }
        }
        gm.g.sync()?;
        let elapsed = start.elapsed().as_secs_f64();
        let after = gm.cpu_picks();
        emit(
            json!({"event":"serial_probe","tokens":a.steps,"elapsed_s":elapsed,"effective_tok_s":a.steps as f64/elapsed,"cpu_forward_s":gm.cpu_busy_s()-cpu,"routed_picks":after.0-picks.0,"cpu_picks":after.1-picks.1,"adapt_swaps":gm.adapt_swaps,"stage_profile":gm.prof,"event_profile":gm.evprof,"adapt":std::env::var("DSX_ADAPT").ok(),"defer":std::env::var("DSX_DEFER").ok(),"logit_fingerprint":hash_logits(&result)}),
        );
        if a.replay_sampling.is_some() {emit(json!({"event":"replay_sampling","mode":a.replay_sampling,"seed":seed,"sampled_tokens":sampled,"gpu_fallbacks":fallbacks}));}
        reference_io(&a,&result)?;
        return Ok(());
    }
    gm.g.sync()?;
    let root = gm.snapshot()?;
    let initial_logits = lg;
    let mut tokens = Vec::with_capacity(a.steps);
    let mut reference = Vec::with_capacity(a.steps);
    let mut lg = initial_logits.clone();
    // Produce a representative continuation once. Every timing arm subsequently consumes these exact IDs.
    for i in 0..a.steps {
        let next = forced.as_ref().map_or_else(|| top(&lg), |x| x[i]);
        tokens.push(next);
        lg = gm.step(next)?;
        reference.push(lg.clone());
    }
    if a.warm_cache_oracle {
        gm.freeze_residency_for_validation()?;
        gm.restore(&root)?;
        reference.clear();
        for &tk in &tokens {
            reference.push(gm.step(tk)?);
        }
        emit(
            json!({"event":"cache_oracle","warmup_tokens":tokens.len(),"timed_adaptation":false,"limit":"Placement was warmed using the exact future continuation; this is an optimistic cache upper bound."}),
        );
    }
    if a.sampler_probe {
        ensure!(a.warm_cache_oracle, "--sampler-probe requires --warm-cache-oracle");
        use dsx_core::server::sample::Sampler;
        use dsx::sampling::StepOutput;
        gm.restore(&root)?;
        let _ = gm.step_sampled(tokens[0], 0.5f64.to_bits())?;
        let seed_base = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_nanos() as u64;
        for round in 0..a.rounds {
            let mut cpu_tokens = None;
            let mut gpu_tokens = None;
            for gpu in if round % 2 == 0 { [false,true] } else { [true,false] } {
                gm.restore(&root)?;
                let picks = gm.cpu_picks();
                let mut sampler = Sampler::new(1.0,1.0,Some(seed_base.wrapping_add(round as u64)));
                let mut sampled = Vec::with_capacity(tokens.len());
                let mut fallbacks = 0usize;
                let start = Instant::now();
                for &token in &tokens {
                    let selected = if gpu {
                        let saved = sampler.clone();
                        let bits = sampler.uniform_bits();
                        match gm.step_sampled(token,bits)? {
                            StepOutput::Token(next) => next,
                            StepOutput::Logits(logits) => {sampler = saved;fallbacks += 1;sampler.sample(&logits)}
                        }
                    } else { sampler.sample(&gm.step(token)?) };
                    sampled.push(selected);
                }
                let elapsed = start.elapsed().as_secs_f64();
                let after = gm.cpu_picks();
                emit(json!({"event":"sampler_timing","round":round,"gpu_sampling":gpu,"tokens":tokens.len(),"elapsed_s":elapsed,"effective_tok_s":tokens.len() as f64/elapsed,"cpu_picks":after.1-picks.1,"routed_picks":after.0-picks.0,"fallbacks":fallbacks,"seed":seed_base.wrapping_add(round as u64),"sampled_tokens":sampled}));
                if gpu {gpu_tokens = Some(sampled);} else {cpu_tokens = Some(sampled);}
            }
            let cpu = cpu_tokens.unwrap();let gpu = gpu_tokens.unwrap();
            emit(json!({"event":"sampler_comparison","round":round,"changed_token_draws":cpu.iter().zip(&gpu).filter(|(a,b)|a!=b).count(),"draws":cpu.len()}));
        }
        emit(json!({"event":"done"}));
        return Ok(());
    }
    if let Some(path) = &a.dump_reference {
        use std::io::Write;
        let mut out = std::io::BufWriter::new(std::fs::File::create(path)?);
        for row in &reference {
            for value in row {
                out.write_all(&value.to_le_bytes())?;
            }
        }
        out.flush()?;
    }
    if let Some(path) = &a.compare_reference {
        let bytes = std::fs::read(path)?;
        let cols = reference[0].len();
        ensure!(
            bytes.len() >= a.steps * cols * 4,
            "comparison reference too short"
        );
        let values: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let prior: Vec<Vec<f32>> = values
            .chunks_exact(cols)
            .take(a.steps)
            .map(|r| r.to_vec())
            .collect();
        emit(
            json!({"event":"placement_reference_comparison","reference":path,"difference":difference(&prior,&reference)}),
        );
    }
    emit(
        json!({"event":"reference","tokens":tokens,"logit_fingerprint":hash_logits(&reference),"defer":std::env::var("DSX_DEFER").ok(),"adapt":std::env::var("DSX_ADAPT").ok(),"ctx":a.ctx,"warmup":a.warmup,"cpu_threads":std::env::var("DSX_CPU_THREADS").ok(),"oracle_semantics":"Fixed reference IDs, forced full acceptance: an upper bound, not a valid sampled speculative generation run"}),
    );
    for round in 0..a.rounds {
        let mut modes = vec![0usize];
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
            let trace_name = if width == 0 {
                "step".to_string()
            } else {
                format!("verify{width}")
            };
            let tracing = a.trace_path.as_deref() == Some(trace_name.as_str());
            if tracing {
                unsafe {
                    dsx::gpu::check(cudarc::driver::sys::cuProfilerStart(), "cuProfilerStart")?;
                }
            }
            let start = Instant::now();
            let mut result = Vec::with_capacity(tokens.len());
            let mut calls = 0usize;
            let mut call_s = Vec::new();
            let mut commit_s = 0.;
            if width == 0 {
                for &tk in &tokens {
                    let t = Instant::now();
                    result.push(gm.step(tk)?);
                    call_s.push(t.elapsed().as_secs_f64());
                    calls += 1;
                }
            } else {
                for chunk in tokens.chunks(width) {
                    let t = Instant::now();
                    let rows = gm.verify_spec(&pb, chunk)?;
                    call_s.push(t.elapsed().as_secs_f64());
                    let t = Instant::now();
                    gm.keep(chunk.len())?;
                    commit_s += t.elapsed().as_secs_f64();
                    result.extend(rows);
                    calls += 1;
                }
            }
            gm.g.sync()?;
            let elapsed = start.elapsed().as_secs_f64();
            let after = gm.cpu_picks();
            let after_split = gm.cpu_split_s();
            if tracing {
                unsafe {
                    dsx::gpu::check(cudarc::driver::sys::cuProfilerStop(), "cuProfilerStop")?;
                }
            }
            // Differences and fingerprints are computed outside the timed interval.
            emit(
                json!({"event":"timing","round":round,"path":if width==0{"step"}else{"forced_accept_verify"},"width":width,"tokens":tokens.len(),"calls":calls,"elapsed_s":elapsed,"effective_tok_s":tokens.len() as f64/elapsed,"verify_or_step_call_s":call_s,"keep_host_s":commit_s,"cpu_forward_s":gm.cpu_busy_s()-cpu,"routed_picks":after.0-picks.0,"cpu_picks":after.1-picks.1,"cpu_split_s":[after_split[0]-split[0],after_split[1]-split[1],after_split[2]-split[2]],"difference_from_serial_reference":difference(&reference,&result),"logit_fingerprint":hash_logits(&result),"stage_profile":gm.prof,"event_profile":gm.evprof}),
            );
        }
    }
    // Verify a known position followed by deliberately arbitrary suffixes, discard the suffix, and continue.
    // Each run must agree with the other dummy run; differences from step also expose cross-path numerics.
    if a.rollback_steps > 0 {
        let n = a.rollback_steps.min(tokens.len());
        for width in [2usize, 4, 8] {
            let mut previous: Option<Vec<Vec<f32>>> = None;
            for dummy in [11u32, 4242u32] {
                gm.restore(&root)?;
                gm.g.sync()?;
                let start = Instant::now();
                let mut result = Vec::new();
                let mut commits = Vec::new();
                for &tk in &tokens[..n] {
                    let mut block = vec![dummy; width];
                    block[0] = tk;
                    let mut out = gm.verify_spec(&pb, &block)?;
                    let t = Instant::now();
                    gm.keep(1)?;
                    gm.g.sync()?;
                    commits.push(t.elapsed().as_secs_f64());
                    result.push(out.remove(0));
                }
                let elapsed = start.elapsed().as_secs_f64();
                emit(
                    json!({"event":"rollback","width":width,"dummy":dummy,"tokens":n,"elapsed_s":elapsed,"keep_sync_s":commits,"difference_from_serial_reference":difference(&reference[..n],&result),"difference_between_dummy_runs":previous.as_ref().map(|p|difference(p,&result)),"logit_fingerprint":hash_logits(&result)}),
                );
                previous = Some(result);
            }
        }
    }
    gm.restore(&root)?;
    gm.g.sync()?;
    emit(json!({"event":"done"}));
    Ok(())
}
