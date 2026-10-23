//! dsx-amd: DeepSeek-V4.1-Flash-1017 on one AMD Instinct MI325X.

mod ck;
mod decode;
mod export;
mod hip;
mod load;
mod prefill;
mod selftest;
mod serve;
mod verify;
mod batch;
mod runner;
mod dspark;

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
        /// batched runner: this many concurrent sequences (0: one conversation at a time)
        #[arg(long, default_value_t = 0)]
        slots: usize,
        /// batched runner: context per slot (default: --ctx)
        #[arg(long)]
        batch_ctx: Option<usize>,
        /// require this key on every API request (`Authorization: Bearer <key>` or `x-api-key: <key>`);
        /// `openssl rand -hex 32` makes one. Health checks and the web UI's files stay open.
        #[arg(long, env = "DSX_API_KEY")]
        api_key: Option<String>,
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
    let g = dsx_core::gguf::Gguf::open(gguf)?;
    let c = dsx_core::model::Config::from_gguf(&g)?;
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
    eval_run_m(dec, eval, seqs, len, false)
}

/// eval_run through the prompt path in chunks of `chunk` tokens (logits of every position from chunk_logits)
fn ceval_run(dec: &mut Decoder, eval: &Path, seqs: usize, len: usize, chunk: usize) -> Result<String> {
    let rd = |f: &str| std::fs::read(eval.join(f));
    let toks: Vec<u32> = rd("heldout.i32")?.chunks(4).map(|b| i32::from_le_bytes(b.try_into().unwrap()) as u32).collect();
    let tidx: Vec<i32> = rd("teacher_idx.i32")?.chunks(4).map(|b| i32::from_le_bytes(b.try_into().unwrap())).collect();
    let tlp: Vec<f32> = rd("teacher_lp.f32")?.chunks(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
    let topk = 256;
    dec.build_bdense()?;
    let (mut nll, mut kl, mut agree, mut cnt) = (0f64, 0f64, 0usize, 0usize);
    let t0 = Instant::now();
    for sq in 0..seqs {
        dec.reset()?;
        let seq = &toks[sq * len..(sq + 1) * len];
        for c0 in (0..len - 1).step_by(chunk) {
            let c1 = (c0 + chunk).min(len - 1);
            dec.prefill(&seq[c0..c1])?;
            dec.chunk_logits(c1 - c0, &mut |i, l| {
                let p = c0 + i;
                let lq = log_softmax(l);
                nll -= lq[seq[p + 1] as usize];
                let row = sq * (len - 1) + p;
                let (ti, tl) = (&tidx[row * topk..(row + 1) * topk], &tlp[row * topk..(row + 1) * topk]);
                kl += ti.iter().zip(tl).map(|(&i, &lt)| (lt as f64).exp() * (lt as f64 - lq[i as usize])).sum::<f64>();
                let am = l.iter().enumerate().fold((0usize, f32::NEG_INFINITY), |a, (i, &v)| if v > a.1 { (i, v) } else { a }).0;
                agree += (am as i32 == ti[0]) as usize;
                cnt += 1;
                Ok(())
            })?;
        }
    }
    Ok(format!("{{\"chunk\": {chunk}, \"tokens\": {cnt}, \"ppl\": {:.4}, \"kl_top{topk}\": {:.6}, \"top1_agree\": {:.6}, \"s\": {:.1}}}", (nll / cnt as f64).exp(), kl / cnt as f64, agree as f64 / cnt as f64, t0.elapsed().as_secs_f64()))
}

/// `prompt`: every token through the prompt path (a one-token prefill) instead of the decode step
fn eval_run_m(dec: &mut Decoder, eval: &Path, seqs: usize, len: usize, prompt: bool) -> Result<String> {
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
            if prompt { dec.prefill(&seq[p..p + 1])?; } else { dec.step(seq[p])?; }
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

/// eval_run through batched decode: the SEQS sequences (each REP times, so B = SEQS * REP) decoded together from
/// empty slots, teacher-forced, one batched step per position
fn beval_run(dec: &mut Decoder, eval: &Path, seqs: usize, rep: usize, len: usize) -> Result<String> {
    let rd = |f: &str| std::fs::read(eval.join(f));
    let toks: Vec<u32> = rd("heldout.i32")?.chunks(4).map(|b| i32::from_le_bytes(b.try_into().unwrap()) as u32).collect();
    let tidx: Vec<i32> = rd("teacher_idx.i32")?.chunks(4).map(|b| i32::from_le_bytes(b.try_into().unwrap())).collect();
    let tlp: Vec<f32> = rd("teacher_lp.f32")?.chunks(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
    let topk = 256;
    let bsz = seqs * rep;
    let mut b = dec.new_batch(bsz, len + 16)?;
    let slots: Vec<usize> = (0..bsz).collect();
    let (mut nll, mut kl, mut agree, mut cnt) = (vec![0f64; rep], vec![0f64; rep], vec![0usize; rep], 0usize);
    let t0 = Instant::now();
    let mut tstep = 0f64;
    for p in 0..len - 1 {
        let feed: Vec<u32> = (0..bsz).map(|i| toks[(i % seqs) * len + p]).collect();
        let ts = Instant::now();
        dec.batch_step(&mut b, &slots, &feed)?;
        tstep += ts.elapsed().as_secs_f64();
        for i in 0..bsz {
            let (sq, r) = (i % seqs, i / seqs);
            let l = dec.batch_logits(&b, i)?;
            let lq = log_softmax(&l);
            nll[r] -= lq[toks[sq * len + p + 1] as usize];
            let row = sq * (len - 1) + p;
            let (ti, tl) = (&tidx[row * topk..(row + 1) * topk], &tlp[row * topk..(row + 1) * topk]);
            kl[r] += ti.iter().zip(tl).map(|(&i, &lt)| (lt as f64).exp() * (lt as f64 - lq[i as usize])).sum::<f64>();
            let am = l.iter().enumerate().fold((0usize, f32::NEG_INFINITY), |a, (i, &v)| if v > a.1 { (i, v) } else { a }).0;
            agree[r] += (am as i32 == ti[0]) as usize;
        }
        cnt += seqs;
        if (p + 1) % 256 == 0 {
            eprintln!("pos {}: kl {:.4} ({:.0} s, {:.1} ms/step)", p + 1, kl[0] / cnt as f64, t0.elapsed().as_secs_f64(), tstep * 1e3 / (p + 1) as f64);
        }
    }
    let per: Vec<String> = (0..rep).map(|r| format!("{{\"ppl\": {:.4}, \"kl_top{topk}\": {:.6}, \"top1_agree\": {:.6}}}", (nll[r] / cnt as f64).exp(), kl[r] / cnt as f64, agree[r] as f64 / cnt as f64)).collect();
    Ok(format!("{{\"batch\": {bsz}, \"tokens\": {cnt}, \"ms_per_step\": {:.2}, \"replicas\": [{}]}}", tstep * 1e3 / (len - 1) as f64, per.join(", ")))
}

fn dev(model: &Path, rank: &[(usize, usize)], ctx: usize, cmd: &Path) -> Result<()> {
    let nl = std::env::var("DSXA_LAYERS").ok().and_then(|v| v.parse().ok());
    let mut dec = Decoder::load(model, ctx, rank, &[], nl)?;
    dec.load_dspark(model)?;
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
            // a malformed command (missing argument) must not take the loaded model down
            let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<()> {
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
                    "spec" => {
                        // spec N [file]: greedy DSpark generation of N tokens from a chat prompt (file: user text)
                        let n: usize = w[1].parse()?;
                        let text = match w.get(2) { Some(f) => std::fs::read_to_string(f)?, None => "Explain how a hash map works, with a short Python example.".into() };
                        let tok = tokenizers::Tokenizer::from_file(model.join("tokenizer.json")).map_err(|e| anyhow::anyhow!("{e}"))?;
                        let chat = format!("<｜begin▁of▁sentence｜><｜User｜>{text}<｜Assistant｜></think>");
                        let ids: Vec<u32> = tok.encode(chat.as_str(), false).map_err(|e| anyhow::anyhow!("{e}"))?.get_ids().to_vec();
                        // reference: plain greedy decode
                        let t0 = Instant::now();
                        let mut next = dec.prefill(&ids)?;
                        let top2 = |lg: &[f32]| { let (mut a, mut b) = (f32::NEG_INFINITY, f32::NEG_INFINITY); for &v in lg { if v > a { b = a; a = v; } else if v > b { b = v; } } a - b };
                        let m0 = top2(&dec.logits()?);
                        let mut plain = vec![];
                        let mut margin = vec![];
                        for i in 0..n {
                            plain.push(next);
                            next = dec.step(next)?;
                            if i < 64 {
                                // top-2 gap of the logits that chose the next token (near-ties explain divergence)
                                let lg = dec.logits()?;
                                let (mut a, mut b) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
                                for &v in &lg {
                                    if v > a { b = a; a = v; } else if v > b { b = v; }
                                }
                                margin.push(a - b);
                            }
                        }
                        let tplain = t0.elapsed().as_secs_f64();
                        dec.reset()?;
                        let t1 = Instant::now();
                        let next = dec.prefill(&ids)?;
                        let m1 = top2(&dec.logits()?);
                        println!("first token: plain {} (top-2 gap {m0:.4}), spec run {next} (gap {m1:.4})", plain[0]);
                        let mut got = vec![];
                        let (sx, _) = dec.spec_loop(next, n, &mut |tk| { got.push(tk); true })?;
                        let tspec = t1.elapsed().as_secs_f64();
                        let same = got.iter().zip(&plain).take_while(|(a, b)| a == b).count();
                        if same < n && same >= 1 && same - 1 < margin.len() {
                            println!("first difference at {same}: plain {} vs spec {}; plain's top-2 logit gap there {:.4} (median gap {:.3})", plain[same], got[same], margin[same - 1], { let mut m = margin.clone(); m.sort_by(|a, b| a.partial_cmp(b).unwrap()); m[m.len() / 2] });
                        }
                        println!("prompt {} tokens; plain {n} tokens {:.2} s, spec {:.2} s ({:.2}x); {} passes, {:.2} drafts accepted per pass, {} tokens decoded plainly, draft {:.1} ms, verify {:.1} ms per pass; same as plain for {same}/{n}",
                            ids.len(), tplain, tspec, tplain / tspec, sx.passes, sx.accepted as f64 / sx.passes as f64, sx.plain, sx.draft_us / 1e3 / sx.passes as f64, sx.verify_us / 1e3 / sx.passes as f64);
                        println!("{}", tok.decode(&got, false).map_err(|e| anyhow::anyhow!("{e}"))?.chars().take(400).collect::<String>());
                    }
                    "pdet" => {
                        // prefill determinism: the same chat prompt N times, top-2 logits each time
                        let n: usize = w[1].parse()?;
                        let text = match w.get(2) { Some(f) => std::fs::read_to_string(f)?, None => "Explain how a hash map works, with a short Python example.".into() };
                        let tok = tokenizers::Tokenizer::from_file(model.join("tokenizer.json")).map_err(|e| anyhow::anyhow!("{e}"))?;
                        let chat = format!("<｜begin▁of▁sentence｜><｜User｜>{text}<｜Assistant｜></think>");
                        let ids: Vec<u32> = tok.encode(chat.as_str(), false).map_err(|e| anyhow::anyhow!("{e}"))?.get_ids().to_vec();
                        for _ in 0..n {
                            dec.reset()?;
                            let next = dec.prefill(&ids)?;
                            let lg = dec.logits()?;
                            let mut ix: Vec<usize> = (0..lg.len()).collect();
                            ix.sort_by(|&a, &b| lg[b].partial_cmp(&lg[a]).unwrap());
                            println!("next {next}: top {} {:.4}, {} {:.4}, {} {:.4}", ix[0], lg[ix[0]], ix[1], lg[ix[1]], ix[2], lg[ix[2]]);
                        }
                    }
                    "sprof" => {
                        // kernel time of DSpark generation (N tokens) after a short prompt
                        let n: usize = w[1].parse()?;
                        let prompt: Vec<u32> = vec![0, 671, 6102, 294, 8760, 344, 11111, 16, 455, 6102, 294, 10322, 344];
                        let next = dec.prefill(&prompt)?;
                        dec.g.sync()?;
                        dec.g.prof_start();
                        let t0 = Instant::now();
                        let r = dec.spec_loop(next, n, &mut |_| true);
                        let p = dec.g.prof_end()?;
                        let (sx, _) = r?;
                        let tot: f32 = p.iter().map(|x| x.1).sum();
                        println!("spec {n}: {} passes, {:.2} accepted per pass; {:.0} us wall per pass, {:.0} us in kernels per pass", sx.passes, sx.accepted as f64 / sx.passes as f64, t0.elapsed().as_secs_f64() * 1e6 / sx.passes as f64, tot / sx.passes as f32);
                        for (nm, t, k) in p.iter().take(30) {
                            println!("  {nm:<28} {:>10.0} us {:>6.1} launches per pass", t / sx.passes as f32, *k as f32 / sx.passes as f32);
                        }
                    }
                    "vprof" => {
                        // kernel time of one verify pass of T tokens after a short prompt (state left as before)
                        let t: usize = w[1].parse()?;
                        dec.reset()?;
                        let prompt: Vec<u32> = (0..64).map(|i| ((i * 7919 + 13) % 120000) as u32).collect();
                        let _ = dec.prefill(&prompt)?;
                        let feed: Vec<u32> = (0..t).map(|i| (100 + i * 37) as u32).collect();
                        let _ = dec.verify(&feed)?;
                        dec.commit(t, 1)?;
                        dec.g.sync()?;
                        dec.g.prof_start();
                        let t0 = Instant::now();
                        let r = dec.verify(&feed);
                        let p = dec.g.prof_end()?;
                        r?;
                        dec.commit(t, 1)?;
                        let tot: f32 = p.iter().map(|x| x.1).sum();
                        println!("verify {t}: {:.0} us wall, {:.0} us in kernels ({} launches)", t0.elapsed().as_secs_f64() * 1e6, tot, p.iter().map(|x| x.2).sum::<usize>());
                        for (nm, t, k) in p.iter().take(30) {
                            println!("  {nm:<28} {t:>10.0} us {k:>6} launches");
                        }
                    }
                    "bcheck" => {
                        // bcheck N [CTX]: two different prompts decoded greedily N tokens each, single-stream vs one
                        // batched pass of both (slots 1 and 3 of 4); prints where the token streams first differ
                        let n: usize = w[1].parse()?;
                        let ctx: usize = w.get(2).map(|s| s.parse()).transpose()?.unwrap_or(8192);
                        let prompts: Vec<Vec<u32>> = vec![(0..300).map(|i| ((i * 7919 + 13) % 120000) as u32).collect(), (0..1500).map(|i| ((i * 104729 + 7) % 120000) as u32).collect()];
                        let mut refs = vec![];
                        let mut b = dec.new_batch(4, ctx)?;
                        let mut first = vec![];
                        for (pi, pr) in prompts.iter().enumerate() {
                            dec.reset()?;
                            let mut tok = dec.prefill(pr)?;
                            dec.load_slot(&mut b, 1 + 2 * pi)?;
                            first.push(tok);
                            let mut out = vec![tok];
                            for _ in 1..n {
                                tok = dec.step(tok)?;
                                out.push(tok);
                            }
                            refs.push(out);
                        }
                        let mut toks = first.clone();
                        let mut got = vec![first.clone()];
                        let t0 = Instant::now();
                        for _ in 1..n {
                            toks = dec.batch_step(&mut b, &[1, 3], &toks)?;
                            got.push(toks.clone());
                        }
                        println!("batched {} steps of 2: {:.1} ms/step", n - 1, t0.elapsed().as_secs_f64() * 1e3 / (n - 1).max(1) as f64);
                        for pi in 0..2 {
                            let gs: Vec<u32> = got.iter().map(|v| v[pi]).collect();
                            let d = (0..n).find(|&i| gs[i] != refs[pi][i]);
                            println!("prompt {pi}: first difference at {:?} of {n}\n  single {:?}\n  batch  {:?}", d, &refs[pi][..n.min(24)], &gs[..n.min(24)]);
                        }
                    }
                    "blogit" => {
                        // blogit CTX [SLOT] [TEXTFILE]: logits of one token after a chat prompt: decode step vs batched step
                        // (slot SLOT of 4) vs the prompt path over prompt + token
                        let ctx: usize = w[1].parse()?;
                        let slot: usize = w.get(2).map(|s| s.parse()).transpose()?.unwrap_or(2);
                        let text = match w.get(3) { Some(f) => std::fs::read_to_string(f)?, None => "Write a Rust function that parses a CSV line with quoted fields, and explain the edge cases.".into() };
                        let tok = tokenizers::Tokenizer::from_file(model.join("tokenizer.json")).map_err(|e| anyhow::anyhow!("{e}"))?;
                        let chat = format!("<｜begin▁of▁sentence｜><｜User｜>{text}<｜Assistant｜></think>");
                        let ids: Vec<u32> = tok.encode(chat.as_str(), false).map_err(|e| anyhow::anyhow!("{e}"))?.get_ids().to_vec();
                        let mut b = dec.new_batch(4, ctx)?;
                        dec.reset()?;
                        let t1 = dec.prefill(&ids)?;
                        dec.load_slot(&mut b, slot)?;
                        let _ = dec.step(t1)?;
                        let l1 = dec.logits()?;
                        let nb = dec.batch_step(&mut b, &[slot], &[t1])?;
                        let l2 = dec.batch_logits(&b, 0)?;
                        dec.reset()?;
                        let mut ids2 = ids.clone();
                        ids2.push(t1);
                        let _ = dec.prefill(&ids2)?;
                        let l3 = dec.logits()?;
                        let rel = |a: &[f32], r: &[f32]| (a.iter().zip(r).map(|(x, y)| (x - y) * (x - y)).sum::<f32>() / r.iter().map(|y| y * y).sum::<f32>()).sqrt();
                        let top = |l: &[f32]| { let mut ix: Vec<usize> = (0..l.len()).collect(); ix.sort_by(|&a, &b| l[b].partial_cmp(&l[a]).unwrap()); ix[..5].to_vec() };
                        println!("prompt {} tokens, next {t1}; batch argmax {}", ids.len(), nb[0]);
                        println!("  decode top5 {:?}\n  batch  top5 {:?} rel {:.2e}\n  prompt top5 {:?} rel {:.2e}", top(&l1), top(&l2), rel(&l2, &l1), top(&l3), rel(&l3, &l1));
                    }
                    "bdense" => dec.build_bdense()?,
                    "plogit" => {
                        // plogit N: prefill N eval tokens from empty, print the top-5 and the rel. difference of the last
                        // logits from the previous plogit's (A/B of prompt-path settings)
                        let n: usize = w[1].parse()?;
                        let toks: Vec<u32> = std::fs::read("/data/models/lucebox/eval/heldout.i32")?.chunks(4).map(|b| i32::from_le_bytes(b.try_into().unwrap()) as u32).collect();
                        let toks: Vec<u32> = toks.iter().cycle().take(n).copied().collect();
                        dec.reset()?;
                        let t0 = Instant::now();
                        dec.prefill(&toks)?;
                        let l = dec.logits()?;
                        let dt = t0.elapsed().as_secs_f64();
                        let mut ix: Vec<usize> = (0..l.len()).collect();
                        ix.sort_by(|&a, &b| l[b].partial_cmp(&l[a]).unwrap());
                        let prev: Vec<f32> = std::fs::read("/root/plogit.bin").map(|b| b.chunks(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()).unwrap_or_default();
                        let rel = if prev.len() == l.len() { let lp = log_softmax(&prev); let lq = log_softmax(&l); Some(lp.iter().zip(&lq).map(|(a, b)| (a.exp() as f64) * (a - b) as f64).sum::<f64>()) } else { None };
                        std::fs::write("/root/plogit.bin", l.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>())?;
                        println!("plogit {n}: {:.2} s ({:.0} tok/s), top5 {:?}, KL vs previous {:?}", dt, n as f64 / dt, &ix[..5], rel);
                    }
                    "bsweep" => {
                        // bsweep CTX PLEN STEPS B1,B2,..: one PLEN-token prompt prefilled once, copied into max(B) slots, then
                        // STEPS timed batched steps at each B (tok/s per session and total) with a one-step kernel profile
                        let bctx: usize = w[1].parse()?;
                        let plen: usize = w[2].parse()?;
                        let steps: usize = w[3].parse()?;
                        let bs: Vec<usize> = w[4].split(',').map(|x| x.parse().unwrap()).collect();
                        let bmax = *bs.iter().max().unwrap();
                        let mut b = dec.new_batch(bmax, bctx)?;
                        // `distinct` (5th argument): every slot its own prompt (eval text at different offsets), so sessions route
                        // like independent ones; otherwise one prompt copied into every slot
                        let distinct = w.get(5) == Some(&"distinct");
                        let ev: Vec<u32> = std::fs::read("/data/models/lucebox/eval/heldout.i32").map(|b| b.chunks(4).map(|c| i32::from_le_bytes(c.try_into().unwrap()) as u32).collect()).unwrap_or_default();
                        let mut first = 0;
                        for s in 0..bmax {
                            if s == 0 || distinct {
                                dec.reset()?;
                                let pr: Vec<u32> = if distinct && !ev.is_empty() { (0..plen).map(|i| ev[(s * 997 + i) % ev.len()]).collect() } else { (0..plen).map(|i| ((i * 7919 + 13) % 120000) as u32).collect() };
                                first = dec.prefill(&pr)?;
                            }
                            dec.load_slot(&mut b, s)?;
                        }
                        for &bsz in &bs {
                            let slots: Vec<usize> = (0..bsz).collect();
                            let mut toks: Vec<u32> = (0..bsz).map(|i| (first + i as u32 * 31) % 120000).collect();
                            toks = dec.batch_step(&mut b, &slots, &toks)?;
                            dec.g.sync()?;
                            let t0 = Instant::now();
                            for _ in 0..steps {
                                toks = dec.batch_step(&mut b, &slots, &toks)?;
                            }
                            let st = t0.elapsed().as_secs_f64() / steps as f64;
                            dec.g.prof_start();
                            let r = dec.batch_step(&mut b, &slots, &toks);
                            let p = dec.g.prof_end()?;
                            toks = r?;
                            let _ = &toks;
                            let tot: f32 = p.iter().map(|x| x.1).sum();
                            let ntop = if std::env::var_os("DSXA_PROF_ALL").is_some() { p.len() } else { 6 };
                            let top: Vec<String> = p.iter().take(ntop).map(|(n, t, k)| format!("{n} {:.2}/{k}", t / 1e3)).collect();
                            println!("B={bsz:<3} ctx~{plen}: {:.1} ms/step, {:.1} tok/s per session, {:.0} total | kernels {:.1} ms in {} launches: {}", st * 1e3, 1.0 / st, bsz as f64 / st, tot / 1e3, p.iter().map(|x| x.2).sum::<usize>(), top.join(", "));
                            for s in 0..bsz {
                                b.pos[s] -= steps + 2;
                            }
                        }
                    }
                    "rbench" => {
                        // rbench SLOTS CTX N PROMPT GEN [TURNS]: N requests of PROMPT synthetic tokens (distinct per request),
                        // GEN greedy tokens each, through the batched runner; TURNS > 1 sends each conversation's
                        // follow-up (its history + 256 new tokens) after it finishes (prefix reuse)
                        use dsx_core::server::BatchEngine;
                        let slots: usize = w[1].parse()?;
                        let bctx: usize = w[2].parse()?;
                        let n: usize = w[3].parse()?;
                        let plen: usize = w[4].parse()?;
                        let ngen: usize = w[5].parse()?;
                        let turns: usize = w.get(6).map(|s| s.parse()).transpose()?.unwrap_or(1);
                        let p = dsx_core::server::GenParams { max_new: ngen, temperature: 0.0, top_p: 1.0, seed: None, stop_token_ids: vec![], hesitation: None, images: Default::default() };
                        let mut r = runner::Runner::new(&mut dec, slots, bctx)?;
                        let prompt = |i: usize, len: usize| -> Vec<u32> { (0..len).map(|j| ((j * 7919 + i * 104729 + 13) % 120000) as u32).collect() };
                        let t0 = Instant::now();
                        let mut hist: Vec<Vec<u32>> = (0..n).map(|i| prompt(i, plen)).collect();
                        let mut left: Vec<usize> = vec![turns; n];
                        let mut got = vec![0usize; n];
                        let mut first: Vec<Option<f64>> = vec![None; n];
                        let mut queue: std::collections::VecDeque<usize> = (0..n).collect();
                        let mut active = 0usize;
                        let (mut steps, mut toks) = (0usize, 0usize);
                        let mut tdec = 0f64;
                        while !queue.is_empty() || active > 0 {
                            while r.has_room() {
                                let Some(i) = queue.pop_front() else { break };
                                r.start(i as u64, &hist[i], &p)?;
                                got[i] = 0;
                                active += 1;
                            }
                            let ts = Instant::now();
                            let out = r.step()?;
                            let dt = ts.elapsed().as_secs_f64();
                            steps += 1;
                            if out.len() > 1 { tdec += dt; toks += out.len(); }
                            for (id, tk) in out {
                                let i = id as usize;
                                first[i].get_or_insert(t0.elapsed().as_secs_f64());
                                got[i] += 1;
                                hist[i].push(tk);
                                if got[i] >= ngen {
                                    // history = fed tokens: the last emitted token was not fed
                                    hist[i].pop();
                                    r.finish(id);
                                    active -= 1;
                                    left[i] -= 1;
                                    if left[i] > 0 {
                                        let more = prompt(1000 + i * 31 + left[i], 256);
                                        hist[i].extend(more);
                                        queue.push_back(i);
                                    }
                                }
                            }
                        }
                        let tot = t0.elapsed().as_secs_f64();
                        let mut ttft: Vec<f64> = first.iter().flatten().copied().collect();
                        ttft.sort_by(|a, b| a.partial_cmp(b).unwrap());
                        let gen_tot = n * ngen * turns;
                        println!("rbench {n} x ({plen} prompt + {ngen}) x {turns} turns on {slots} slots: {tot:.1} s, {:.0} tok/s generated overall; {steps} steps; batched steps {:.0} tok/s; first token median {:.1} s, max {:.1} s",
                            gen_tot as f64 / tot, toks as f64 / tdec.max(1e-9), ttft[ttft.len() / 2], ttft[ttft.len() - 1]);
                    }
                    "bbench" => {
                        // bbench B CTX [STEPS] [PROMPT]: B slots, each holding the same PROMPT-token context (default
                        // CTX/2), STEPS batched steps; tok/s and the kernel profile of one step
                        let bsz: usize = w[1].parse()?;
                        let ctx: usize = w[2].parse()?;
                        let steps: usize = w.get(3).map(|s| s.parse()).transpose()?.unwrap_or(16);
                        let plen: usize = w.get(4).map(|s| s.parse()).transpose()?.unwrap_or(ctx / 2);
                        let need = batch::Batch::state_bytes(&dec, bsz, ctx);
                        println!("batch state: {:.1} GB", need as f64 / 1e9);
                        let mut b = dec.new_batch(bsz, ctx)?;
                        dec.reset()?;
                        let pr: Vec<u32> = (0..plen).map(|i| ((i * 7919 + 13) % 120000) as u32).collect();
                        let tp = Instant::now();
                        let first = dec.prefill(&pr)?;
                        println!("prefill {plen} tokens: {:.1} s ({:.0} tok/s)", tp.elapsed().as_secs_f64(), plen as f64 / tp.elapsed().as_secs_f64());
                        for s in 0..bsz {
                            dec.load_slot(&mut b, s)?;
                        }
                        let slots: Vec<usize> = (0..bsz).collect();
                        let mut toks: Vec<u32> = (0..bsz).map(|i| (first + i as u32 * 31) % 120000).collect();
                        toks = dec.batch_step(&mut b, &slots, &toks)?;
                        dec.g.sync()?;
                        let t0 = Instant::now();
                        for _ in 0..steps {
                            toks = dec.batch_step(&mut b, &slots, &toks)?;
                        }
                        dec.g.sync()?;
                        let s = t0.elapsed().as_secs_f64() / steps as f64;
                        println!("B={bsz} ctx~{plen}: {:.1} ms/step = {:.0} tok/s", s * 1e3, bsz as f64 / s);
                        dec.g.prof_start();
                        let r = dec.batch_step(&mut b, &slots, &toks);
                        let p = dec.g.prof_end()?;
                        r?;
                        let tot: f32 = p.iter().map(|x| x.1).sum();
                        println!("  one step: {:.0} us in kernels ({} launches)", tot, p.iter().map(|x| x.2).sum::<usize>());
                        for (nm, t, k) in p.iter().take(30) {
                            println!("  {nm:<28} {t:>10.0} us {k:>6} launches");
                        }
                    }
                    "vwaves" => dsx_amd_vwaves(w[1].parse()?),
                    "env" => {
                        // env K [V]: set (or with no value, remove) a variable read at run time by debug switches
                        unsafe {
                            match w.get(2) {
                                Some(_) => std::env::set_var(w[1], w[2..].join(" ")),
                                None => std::env::remove_var(w[1]),
                            }
                        }
                    }
                    "vtest" => {
                        // decode a greedy continuation, then replay it through verify with the true continuation as
                        // drafts: every draft should be accepted; times the verify passes
                        let t: usize = w[1].parse()?;
                        let prompt: Vec<u32> = vec![0, 671, 6102, 294, 8760, 344, 11111, 16, 455, 6102, 294, 10322, 344];
                        let mut next = dec.prefill(&prompt)?;
                        let mut cont = vec![next];
                        for _ in 0..40 {
                            next = dec.step(next)?;
                            cont.push(next);
                        }
                        dec.reset()?;
                        let _ = dec.prefill(&prompt)?;
                        let (mut i, mut passes, mut acc_tot, mut us) = (0usize, 0usize, 0usize, 0f64);
                        while i + t < cont.len() {
                            let feed = &cont[i..i + t];
                            dec.g.sync()?;
                            let t0 = Instant::now();
                            let out = dec.verify(feed)?;
                            us += t0.elapsed().as_secs_f64() * 1e6;
                            if passes < 2 {
                                println!("  feed {:?}\n  out  {:?}\n  want {:?}", feed, out, &cont[i + 1..(i + t + 1).min(cont.len())]);
                            }
                            let mut acc = 0;
                            while acc + 1 < t && out[acc] == cont[i + acc + 1] {
                                acc += 1;
                            }
                            dec.commit(t, acc + 1)?;
                            passes += 1;
                            acc_tot += acc;
                            i += acc + 1;
                        }
                        println!("verify T={t}: {passes} passes, {:.2} drafts accepted per pass (of {}), {:.0} us per pass", acc_tot as f64 / passes as f64, t - 1, us / passes as f64);
                    }
                    "pbench" => {
                        // prompt throughput from an empty context (synthetic tokens, or @FILE of comma-separated ids),
                        // then decode at that context
                        let toks: Vec<u32> = if let Some(f) = w[1].strip_prefix('@') {
                            std::fs::read_to_string(f)?.split(|c: char| c == ',' || c.is_whitespace()).filter(|x| !x.is_empty()).map(|x| x.parse().unwrap()).collect()
                        } else {
                            let n: usize = w[1].parse()?;
                            (0..n).map(|i| ((i * 7919 + 13) % 120000) as u32).collect()
                        };
                        let n = toks.len();
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
                    "pcmp" => {
                        // pcmp EVALDIR N: residual stream after every layer, decode (eager) vs the prompt path one token at a
                        // time, positions 0..N of eval sequence 0; prints rel error per layer at each position
                        let n: usize = w[2].parse()?;
                        let toks: Vec<u32> = std::fs::read(Path::new(w[1]).join("heldout.i32"))?.chunks(4).map(|b| i32::from_le_bytes(b.try_into().unwrap()) as u32).collect();
                        let dir = "/root/pcmp";
                        std::fs::create_dir_all(dir)?;
                        dec.reset()?;
                        for p in 0..n {
                            let (_, d) = dec.step_dump(toks[p])?;
                            for (li, h) in d.iter().enumerate() {
                                std::fs::write(format!("{dir}/ref_p{p}_l{li:02}.bin"), h.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>())?;
                            }
                        }
                        dec.reset()?;
                        unsafe { std::env::set_var("DSX_DUMP", dir) };
                        for p in 0..n {
                            dec.prefill(&toks[p..p + 1])?;
                        }
                        unsafe { std::env::remove_var("DSX_DUMP") };
                    }
                    "ceval" => {
                        // ceval DIR SEQS CHUNK: Lucebox KL through the prompt path in chunks of CHUNK tokens
                        let r = ceval_run(&mut dec, Path::new(w[1]), w[2].parse()?, 1024, w[3].parse()?)?;
                        println!("{r}");
                    }
                    "peval" => {
                        let r = eval_run_m(&mut dec, Path::new(w[1]), w.get(2).map(|s| s.parse()).transpose()?.unwrap_or(8), 1024, true)?;
                        println!("{r}");
                    }
                    "beval" => {
                        let r = beval_run(&mut dec, Path::new(w[1]), w.get(2).map(|s| s.parse()).transpose()?.unwrap_or(8), w.get(3).map(|s| s.parse()).transpose()?.unwrap_or(1), 1024)?;
                        println!("{r}");
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
            }));
            match r {
                Ok(Err(e)) => println!("error: {e:#}"),
                Err(_) => println!("error: command panicked"),
                Ok(Ok(())) => {}
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
        Cmd::Serve { model, rank, ctx, host, port, max_tokens, api_key, slots, batch_ctx } => serve::serve(model, read_rank(&rank)?, ctx, host, port, max_tokens, api_key, slots, batch_ctx.unwrap_or(ctx)),
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
                    let m = dsx_core::model::Model::load_hf(&model)?;
                    eprintln!("reference loaded in {:.1} s", t0.elapsed().as_secs_f32());
                    let mut st = dsx_core::model::RefState::new(&m);
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

fn dsx_amd_vwaves(n: usize) {
    verify::VWAVES.store(n, std::sync::atomic::Ordering::Relaxed);
}
