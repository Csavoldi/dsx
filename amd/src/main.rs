//! dsx-amd: DeepSeek-V4.1-Flash-1017 on one AMD Instinct MI325X.

mod ck;
mod decode;
mod export;
mod hip;
mod load;
mod prefill;
mod selftest;
mod serve;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use decode::Decoder;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Parser)]
#[command(about = "DeepSeek-V4.1-Flash-1017 on one MI325X")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Write the Engram hashing constants (dsx-engram.json) next to the checkpoint, from a dsx GGUF export.
    EngramJson {
        #[arg(long)]
        gguf: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    /// CPU reference (FP8/FP4 checkpoint) vs the GPU decoder, token by token, on the same inputs.
    Check {
        #[arg(long)]
        model: PathBuf,
        #[arg(long, value_delimiter = ',')]
        ids: Vec<u32>,
        #[arg(short, default_value_t = 4)]
        n: usize,
        #[arg(long)]
        rank: Option<PathBuf>,
        /// reuse a previous reference run (written by an earlier check to this path)
        #[arg(long)]
        ref_cache: Option<PathBuf>,
    },
    /// Decode speed at a given context (caches filled with synthetic values).
    Bench {
        #[arg(long)]
        model: PathBuf,
        #[arg(long, value_delimiter = ',', default_value = "4096")]
        ctx: Vec<usize>,
        #[arg(long, default_value_t = 32)]
        steps: usize,
        #[arg(long)]
        rank: Option<PathBuf>,
    },
    /// Quality on Lucebox's 1017 eval (8 x 1024 tokens, FP8 teacher top-256 log-probs), teacher-forced through decode:
    /// KL(teacher || model) over the teacher's top-256, top-1 agreement, perplexity. Reads heldout.i32,
    /// teacher_idx.i32, teacher_lp.f32 (convert the .npy/.npz once with tools/eval_bins.py).
    Eval {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        eval: PathBuf,
        #[arg(long, default_value_t = 8)]
        seqs: usize,
        #[arg(long, default_value_t = 1024)]
        len: usize,
        #[arg(long)]
        rank: Option<PathBuf>,
    },
    /// Kernel self-tests on synthetic inputs against CPU implementations.
    Selftest,
    /// Load a build and write it as one GGUF (split it with `dsx split`); load it later with DSXA_WEIGHTS=<file>.
    Export {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        rank: Option<PathBuf>,
        #[arg(long)]
        out: PathBuf,
    },
    /// OpenAI / Anthropic compatible API server (dsx's server, llama.cpp web UI at /).
    Serve {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        rank: Option<PathBuf>,
        #[arg(long, default_value_t = 131072)]
        ctx: usize,
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        #[arg(long, default_value_t = 8090)]
        port: u16,
        #[arg(long, default_value_t = 32768)]
        max_tokens: usize,
    },
    /// Development loop: load once, then run commands from a file as it appears (one line per command):
    /// `reload PATH.hsaco`, `bench CTX STEPS`, `prof CTX`, `eval DIR SEQS`, `check REFCACHE`, `quit`.
    Dev {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        rank: Option<PathBuf>,
        #[arg(long, default_value_t = 262200)]
        ctx: usize,
        /// command file, consumed (renamed to .done) after each command
        #[arg(long, default_value = "/root/dev.cmd")]
        cmd: PathBuf,
    },
    /// Greedy generation from token ids.
    Gen {
        #[arg(long)]
        model: PathBuf,
        #[arg(long, value_delimiter = ',')]
        ids: Vec<u32>,
        #[arg(short, default_value_t = 32)]
        n: usize,
        #[arg(long)]
        rank: Option<PathBuf>,
        #[arg(long, default_value_t = 8192)]
        ctx: usize,
    },
}

fn read_rank(p: &Option<PathBuf>) -> Result<Vec<(usize, usize)>> {
    let Some(p) = p else { return Ok(vec![]) };
    let v: serde_json::Value = serde_json::from_slice(&std::fs::read(p)?)?;
    Ok(v.as_array().context("rank array")?.iter().map(|r| (r[0].as_u64().unwrap() as usize, r[1].as_u64().unwrap() as usize)).collect())
}

fn top(logits: &[f32], k: usize) -> Vec<(u32, f32)> {
    let mut v: Vec<(u32, f32)> = logits.iter().enumerate().map(|(i, &x)| (i as u32, x)).collect();
    v.select_nth_unstable_by(k - 1, |a, b| b.1.partial_cmp(&a.1).unwrap());
    v.truncate(k);
    v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    v
}

fn log_softmax(l: &[f32]) -> Vec<f64> {
    let mx = l.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
    let lse = mx + l.iter().map(|&v| (v as f64 - mx).exp()).sum::<f64>().ln();
    l.iter().map(|&v| v as f64 - lse).collect()
}

fn engram_json(gguf: &Path, out: &Path) -> Result<()> {
    let g = dsx::gguf::Gguf::open(gguf)?;
    let c = dsx::model::Config::from_gguf(&g)?;
    let per_layer = (c.engram_ngram - 1) * c.engram_heads;
    let primes: Vec<Vec<u64>> = c.engram_primes.chunks(per_layer).map(|x| x.to_vec()).collect();
    let mults: Vec<Vec<u64>> = c.engram_multipliers.chunks(c.engram_ngram).map(|x| x.to_vec()).collect();
    let j = serde_json::json!({ "pad_id": c.engram_pad, "token_map": c.engram_token_map, "primes": primes, "multipliers": mults });
    std::fs::write(out, serde_json::to_vec(&j)?)?;
    eprintln!("wrote {}", out.display());
    Ok(())
}

fn bench(dec: &mut Decoder, c: usize, steps: usize, prof: bool) -> Result<()> {
    dec.fill_random()?;
    dec.set_pos(c)?;
    let mut tok = 1000u32;
    let mut ts = vec![];
    for s in 0..steps {
        let us = dec.time_step(tok)?;
        tok = (tok * 7 + 13) % 100000;
        if s >= 2 {
            ts.push(us);
        }
    }
    ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = ts[ts.len() / 2];
    println!("ctx {c}: decode step {med:.0} us median ({:.1} tok/s), min {:.0} max {:.0}", 1e6 / med, ts[0], ts[ts.len() - 1]);
    if prof {
        let p = dec.step_prof(tok)?;
        let tot: f32 = p.iter().map(|x| x.1).sum();
        println!("  eager step: {tot:.0} us in {} launches", p.iter().map(|x| x.2).sum::<usize>());
        for (n, t, k) in p.iter().take(30) {
            println!("  {n:<28} {t:>8.1} us {k:>5} launches {:>6.2} us each", t / *k as f32);
        }
    }
    Ok(())
}

fn eval_run(dec: &mut Decoder, eval: &Path, seqs: usize, len: usize) -> Result<String> {
    let rd = |f: &str| std::fs::read(eval.join(f));
    let toks: Vec<u32> = rd("heldout.i32")?.chunks(4).map(|b| i32::from_le_bytes(b.try_into().unwrap()) as u32).collect();
    let tidx: Vec<i32> = rd("teacher_idx.i32")?.chunks(4).map(|b| i32::from_le_bytes(b.try_into().unwrap())).collect();
    let tlp: Vec<f32> = rd("teacher_lp.f32")?.chunks(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
    let topk = 256;
    let (mut nll, mut kl, mut agree, mut cnt) = (0f64, 0f64, 0usize, 0usize);
    let t0 = Instant::now();
    for sq in 0..seqs {
        dec.reset()?;
        let seq = &toks[sq * len..(sq + 1) * len];
        for p in 0..len - 1 {
            dec.step(seq[p])?;
            let l = dec.logits()?;
            let lq = log_softmax(&l);
            nll -= lq[seq[p + 1] as usize];
            let row = sq * (len - 1) + p;
            let (ti, tl) = (&tidx[row * topk..(row + 1) * topk], &tlp[row * topk..(row + 1) * topk]);
            kl += ti.iter().zip(tl).map(|(&i, &lt)| (lt as f64).exp() * (lt as f64 - lq[i as usize])).sum::<f64>();
            let am = l.iter().enumerate().fold((0usize, f32::NEG_INFINITY), |a, (i, &v)| if v > a.1 { (i, v) } else { a }).0;
            agree += (am as i32 == ti[0]) as usize;
            cnt += 1;
        }
        eprintln!("seq {sq}: running ppl {:.3} kl {:.4} top1 {:.4} ({:.0} s)", (nll / cnt as f64).exp(), kl / cnt as f64, agree as f64 / cnt as f64, t0.elapsed().as_secs_f64());
    }
    Ok(format!("{{\"tokens\": {cnt}, \"nll\": {:.6}, \"ppl\": {:.4}, \"kl_top{topk}\": {:.6}, \"top1_agree\": {:.6}}}", nll / cnt as f64, (nll / cnt as f64).exp(), kl / cnt as f64, agree as f64 / cnt as f64))
}

fn dev(model: &Path, rank: &[(usize, usize)], ctx: usize, cmd: &Path) -> Result<()> {
    let mut dec = Decoder::load(model, ctx, rank, &[], None)?;
    eprintln!("dev: ready, waiting for {}", cmd.display());
    loop {
        if !cmd.exists() {
            std::thread::sleep(std::time::Duration::from_millis(200));
            continue;
        }
        let text = std::fs::read_to_string(cmd)?;
        std::fs::rename(cmd, cmd.with_extension("done"))?;
        for line in text.lines() {
            let w: Vec<&str> = line.split_whitespace().collect();
            if w.is_empty() {
                continue;
            }
            println!("== {line}");
            let r: Result<()> = (|| {
                match w[0] {
                    "reload" => {
                        let img = std::fs::read(w[1])?;
                        std::fs::write("/root/cur.hsaco", &img)?;
                        dec.reload_kernels(&img)?;
                        println!("reloaded {}", w[1]);
                    }
                    "bench" => bench(&mut dec, w[1].parse()?, w.get(2).map(|s| s.parse()).transpose()?.unwrap_or(24), false)?,
                    "prof" => bench(&mut dec, w[1].parse()?, 8, true)?,
                    "pprof" => {
                        let n: usize = w[1].parse()?;
                        let toks: Vec<u32> = (0..n).map(|i| ((i * 7919 + 13) % 120000) as u32).collect();
                        dec.g.sync()?;
                        dec.g.prof_start();
                        let t0 = Instant::now();
                        let r = dec.prefill(&toks);
                        let p = dec.g.prof_end()?;
                        r?;
                        let tot: f32 = p.iter().map(|x| x.1).sum();
                        println!("prefill {n}: {:.2} s wall, {:.0} us in kernels ({} launches)", t0.elapsed().as_secs_f64(), tot, p.iter().map(|x| x.2).sum::<usize>());
                        for (nm, t, k) in p.iter().take(25) {
                            println!("  {nm:<28} {t:>10.0} us {k:>6} launches");
                        }
                    }
                    "pbench" => {
                        // prompt throughput on synthetic tokens from an empty context, then decode at that context
                        let n: usize = w[1].parse()?;
                        let toks: Vec<u32> = (0..n).map(|i| ((i * 7919 + 13) % 120000) as u32).collect();
                        dec.g.sync()?;
                        let t0 = Instant::now();
                        let next = dec.prefill(&toks)?;
                        dec.g.sync()?;
                        let s = t0.elapsed().as_secs_f64();
                        println!("prefill {n} tokens: {s:.2} s = {:.0} tok/s (next {next})", n as f64 / s);
                        let mut ts = vec![];
                        let mut tok = next;
                        for _ in 0..16 {
                            ts.push(dec.time_step(tok)?);
                            tok = (tok * 7 + 13) % 100000;
                        }
                        ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
                        println!("  decode after it: {:.0} us median ({:.1} tok/s)", ts[8], 1e6 / ts[8]);
                    }
                    "eval" => {
                        let r = eval_run(&mut dec, Path::new(w[1]), w.get(2).map(|s| s.parse()).transpose()?.unwrap_or(8), 1024)?;
                        println!("{r}");
                    }
                    "waves" => {
                        decode::TARGET_WAVES.store(w[1].parse()?, std::sync::atomic::Ordering::Relaxed);
                        if let Some(m) = w.get(2) {
                            decode::MIN_UNITS.store(m.parse()?, std::sync::atomic::Ordering::Relaxed);
                        }
                        dec.reload_kernels(&std::fs::read(w.get(3).copied().unwrap_or("/root/cur.hsaco"))?)?;
                        println!("waves {} min units {}", w[1], w.get(2).unwrap_or(&"-"));
                    }
                    "gtime" => {
                        // GEMV phase timing: needs a kernel built with -DGEMV_TIMING loaded; prints per-launch phases
                        dec.gemv_timing(w[1].parse()?)?;
                    }
                    "quit" => std::process::exit(0),
                    c => println!("unknown command {c}"),
                }
                Ok(())
            })();
            if let Err(e) = r {
                println!("error: {e:#}");
            }
            dec.reset()?;
        }
        println!("== idle");
        use std::io::Write;
        std::io::stdout().flush()?;
    }
}

fn main() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::EngramJson { gguf, out } => engram_json(&gguf, &out),
        Cmd::Selftest => selftest::run(),
        Cmd::Export { model, rank, out } => {
            let dec = Decoder::load(&model, 4096, &read_rank(&rank)?, &[], None)?;
            dec.export(&model, &out)
        }
        Cmd::Serve { model, rank, ctx, host, port, max_tokens } => serve::serve(model, read_rank(&rank)?, ctx, host, port, max_tokens),
        Cmd::Dev { model, rank, ctx, cmd } => dev(&model, &read_rank(&rank)?, ctx, &cmd),
        Cmd::Check { model, ids, n, rank, ref_cache } => {
            // ---- reference: tokens fed, logits per position, routed experts
            let (feed, ref_logits, routes, pos_routes): (Vec<u32>, Vec<Vec<f32>>, Vec<(usize, usize)>, Vec<Vec<Vec<u16>>>) = match ref_cache.as_ref().filter(|p| p.exists()) {
                Some(p) => {
                    let v: serde_json::Value = serde_json::from_slice(&std::fs::read(p)?)?;
                    let feed = v["feed"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect();
                    let routes = v["routes"].as_array().unwrap().iter().map(|r| (r[0].as_u64().unwrap() as usize, r[1].as_u64().unwrap() as usize)).collect();
                    let pr: Vec<Vec<Vec<u16>>> = serde_json::from_value(v["pos_routes"].clone()).unwrap_or_default();
                    let lb = std::fs::read(p.with_extension("logits"))?;
                    let fl: Vec<f32> = lb.chunks(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
                    (feed, fl.chunks(decode::VOCAB).map(|c| c.to_vec()).collect(), routes, pr)
                }
                None => {
                    let t0 = Instant::now();
                    let m = dsx::model::Model::load_hf(&model)?;
                    eprintln!("reference loaded in {:.1} s", t0.elapsed().as_secs_f32());
                    let mut st = dsx::model::RefState::new(&m);
                    let mut feed = ids.clone();
                    let mut out = vec![];
                    let mut i = 0;
                    while i < feed.len() {
                        let t = Instant::now();
                        let lg = m.ref_step(&mut st, feed[i])?;
                        eprintln!("ref pos {i} tok {}: top {:?} ({:.1} s)", feed[i], top(&lg, 3), t.elapsed().as_secs_f32());
                        if i + 1 == feed.len() && feed.len() < ids.len() + n {
                            feed.push(top(&lg, 1)[0].0);
                        }
                        out.push(lg);
                        i += 1;
                    }
                    let mut routes = vec![];
                    for tr in &st.routes {
                        for (l, ex) in tr.iter().enumerate() {
                            for &e in ex {
                                routes.push((l, e as usize));
                            }
                        }
                    }
                    routes.sort();
                    routes.dedup();
                    let pr = st.routes.clone();
                    if let Some(p) = &ref_cache {
                        std::fs::write(p, serde_json::to_vec(&serde_json::json!({"feed": feed, "routes": routes, "pos_routes": pr}))?)?;
                        let b: Vec<u8> = out.iter().flatten().flat_map(|v| v.to_le_bytes()).collect();
                        std::fs::write(p.with_extension("logits"), b)?;
                    }
                    (feed, out, routes, pr)
                }
            };
            eprintln!("reference: {} positions, {} distinct routed experts", feed.len(), routes.len());
            let mut dec = Decoder::load(&model, 4096, &read_rank(&rank)?, &routes, None)?;
            let dump_dir = std::env::var("DSX_DUMP").ok();
            let npf: usize = std::env::var("DSXA_PREFILL").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
            if std::env::var_os("DSXA_PFCMP").is_some() {
                // one token through decode (eager, dumps) and through prefill (dumps), same position 0
                dec.dump = Some(Default::default());
                let _ = dec.step_dump(feed[0])?;
                dec.reset()?;
                dec.prefill(&feed[..1])?;
                return Ok(());
            }
            if npf > 0 {
                let t0 = Instant::now();
                let next = dec.prefill(&feed[..npf])?;
                let lg = dec.logits()?;
                let r = &ref_logits[npf - 1];
                let (lr, lgp) = (log_softmax(r), log_softmax(&lg));
                let kl: f64 = lr.iter().zip(&lgp).map(|(a, b)| a.exp() * (a - b)).sum();
                println!("prefill {npf} tokens ({:.2} s): next {next} top {:?} | ref top {:?} | KL {kl:.2e}", t0.elapsed().as_secs_f32(), top(&lg, 3), top(r, 3));
            }
            for (i, &tok) in feed.iter().enumerate() {
                if i < npf {
                    continue;
                }
                let next = if let (Some(dir), true) = (&dump_dir, i < 4) {
                    let (nx, hs) = dec.step_dump(tok)?;
                    for (l, h) in hs.iter().enumerate() {
                        let p = format!("{dir}/ref_p{i}_l{l:02}.bin");
                        if let Ok(b) = std::fs::read(&p) {
                            let r: Vec<f32> = b.chunks(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
                            let md = r.iter().zip(h).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
                            let rn = r.iter().map(|a| a * a).sum::<f32>().sqrt();
                            let dn = r.iter().zip(h).map(|(a, b)| (a - b) * (a - b)).sum::<f32>().sqrt();
                            println!("  pos {i} layer {l}: max|dh| {md:.4}, rel {:.2e}", dn / rn);
                        }
                    }
                    nx
                } else {
                    dec.step(tok)?
                };
                let lg = dec.logits()?;
                let r = &ref_logits[i];
                let (lr, lgp) = (log_softmax(r), log_softmax(&lg));
                let kl: f64 = lr.iter().zip(&lgp).map(|(a, b)| a.exp() * (a - b)).sum();
                let md = r.iter().zip(&lg).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
                println!("pos {i} tok {tok}: gpu next {next} top {:?} | ref top {:?} | max|dlogit| {md:.4} KL {kl:.2e}", top(&lg, 3), top(r, 3));
                if let Some(pr) = pos_routes.get(i) {
                    let gr = dec.routes()?;
                    for (l, (a, b)) in gr.iter().zip(pr).enumerate() {
                        let mut a: Vec<i32> = a.clone();
                        let mut b: Vec<i32> = b.iter().map(|&x| x as i32).collect();
                        a.sort();
                        b.sort();
                        if a != b {
                            println!("  route differs at layer {l}: gpu {a:?} ref {b:?}");
                        }
                    }
                }
            }
            Ok(())
        }
        Cmd::Bench { model, ctx, steps, rank } => {
            let maxc = *ctx.iter().max().unwrap() + steps + 16;
            let mut dec = Decoder::load(&model, maxc, &read_rank(&rank)?, &[], None)?;
            dec.fill_random()?;
            for &c in &ctx {
                dec.set_pos(c)?;
                let mut tok = 1000u32;
                let mut ts = vec![];
                for s in 0..steps {
                    let us = dec.time_step(tok)?;
                    tok = (tok * 7 + 13) % 100000;
                    if s >= 2 {
                        ts.push(us);
                    }
                }
                ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let med = ts[ts.len() / 2];
                println!("ctx {c}: decode step {med:.0} us median ({:.1} tok/s), min {:.0} max {:.0}", 1e6 / med, ts[0], ts[ts.len() - 1]);
                if std::env::var_os("DSXA_PROF").is_some() {
                    let p = dec.step_prof(tok)?;
                    let tot: f32 = p.iter().map(|x| x.1).sum();
                    println!("  eager step: {tot:.0} us in {} launches", p.iter().map(|x| x.2).sum::<usize>());
                    for (n, t, k) in p.iter().take(30) {
                        println!("  {n:<28} {t:>8.1} us {k:>5} launches {:>6.2} us each", t / *k as f32);
                    }
                }
            }
            Ok(())
        }
        Cmd::Eval { model, eval, seqs, len, rank } => {
            let rd = |f: &str| std::fs::read(eval.join(f));
            let toks: Vec<u32> = rd("heldout.i32")?.chunks(4).map(|b| i32::from_le_bytes(b.try_into().unwrap()) as u32).collect();
            let tidx: Vec<i32> = rd("teacher_idx.i32")?.chunks(4).map(|b| i32::from_le_bytes(b.try_into().unwrap())).collect();
            let tlp: Vec<f32> = rd("teacher_lp.f32")?.chunks(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
            let topk = 256;
            let mut dec = Decoder::load(&model, len + 64, &read_rank(&rank)?, &[], None)?;
            let (mut nll, mut kl, mut agree, mut cnt) = (0f64, 0f64, 0usize, 0usize);
            let t0 = Instant::now();
            for sq in 0..seqs {
                dec.reset()?;
                let seq = &toks[sq * len..(sq + 1) * len];
                for p in 0..len - 1 {
                    dec.step(seq[p])?;
                    let l = dec.logits()?;
                    let lq = log_softmax(&l);
                    nll -= lq[seq[p + 1] as usize];
                    let row = sq * (len - 1) + p;
                    let (ti, tl) = (&tidx[row * topk..(row + 1) * topk], &tlp[row * topk..(row + 1) * topk]);
                    kl += ti.iter().zip(tl).map(|(&i, &lt)| (lt as f64).exp() * (lt as f64 - lq[i as usize])).sum::<f64>();
                    let am = l.iter().enumerate().fold((0usize, f32::NEG_INFINITY), |a, (i, &v)| if v > a.1 { (i, v) } else { a }).0;
                    agree += (am as i32 == ti[0]) as usize;
                    cnt += 1;
                }
                eprintln!("seq {sq}: running ppl {:.3} kl {:.4} top1 {:.4} ({:.0} s)", (nll / cnt as f64).exp(), kl / cnt as f64, agree as f64 / cnt as f64, t0.elapsed().as_secs_f64());
            }
            println!("{{\"tokens\": {cnt}, \"nll\": {:.6}, \"ppl\": {:.4}, \"kl_top{topk}\": {:.6}, \"top1_agree\": {:.6}}}", nll / cnt as f64, (nll / cnt as f64).exp(), kl / cnt as f64, agree as f64 / cnt as f64);
            Ok(())
        }
        Cmd::Gen { model, ids, n, rank, ctx } => {
            let mut dec = Decoder::load(&model, ctx, &read_rank(&rank)?, &[], None)?;
            let t0 = Instant::now();
            let mut next = 0;
            for &t in &ids {
                next = dec.step(t)?;
            }
            let tp = t0.elapsed().as_secs_f64();
            let mut out = vec![next];
            let t1 = Instant::now();
            for _ in 1..n {
                next = dec.step(next)?;
                out.push(next);
            }
            let td = t1.elapsed().as_secs_f64();
            println!("ids: {out:?}");
            eprintln!("prompt {} tokens {:.2} s; {} tokens in {:.2} s = {:.1} tok/s", ids.len(), tp, n - 1, td, (n - 1) as f64 / td);
            Ok(())
        }
    }
}
