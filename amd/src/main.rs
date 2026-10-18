//! dsx-amd: DeepSeek-V4.1-Flash-1017 on one AMD Instinct MI325X.

mod decode;
mod hip;
mod load;

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

fn main() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::EngramJson { gguf, out } => engram_json(&gguf, &out),
        Cmd::Check { model, ids, n, rank, ref_cache } => {
            // ---- reference: tokens fed, logits per position, routed experts
            let (feed, ref_logits, routes): (Vec<u32>, Vec<Vec<f32>>, Vec<(usize, usize)>) = match ref_cache.as_ref().filter(|p| p.exists()) {
                Some(p) => {
                    let v: serde_json::Value = serde_json::from_slice(&std::fs::read(p)?)?;
                    let feed = v["feed"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect();
                    let routes = v["routes"].as_array().unwrap().iter().map(|r| (r[0].as_u64().unwrap() as usize, r[1].as_u64().unwrap() as usize)).collect();
                    let lb = std::fs::read(p.with_extension("logits"))?;
                    let fl: Vec<f32> = lb.chunks(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
                    (feed, fl.chunks(decode::VOCAB).map(|c| c.to_vec()).collect(), routes)
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
                    if let Some(p) = &ref_cache {
                        std::fs::write(p, serde_json::to_vec(&serde_json::json!({"feed": feed, "routes": routes}))?)?;
                        let b: Vec<u8> = out.iter().flatten().flat_map(|v| v.to_le_bytes()).collect();
                        std::fs::write(p.with_extension("logits"), b)?;
                    }
                    (feed, out, routes)
                }
            };
            eprintln!("reference: {} positions, {} distinct routed experts", feed.len(), routes.len());
            let mut dec = Decoder::load(&model, 4096, &read_rank(&rank)?, &routes, None)?;
            for (i, &tok) in feed.iter().enumerate() {
                let next = dec.step(tok)?;
                let lg = dec.logits()?;
                let r = &ref_logits[i];
                let (lr, lgp) = (log_softmax(r), log_softmax(&lg));
                let kl: f64 = lr.iter().zip(&lgp).map(|(a, b)| a.exp() * (a - b)).sum();
                let md = r.iter().zip(&lg).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
                println!("pos {i} tok {tok}: gpu next {next} top {:?} | ref top {:?} | max|dlogit| {md:.4} KL {kl:.2e}", top(&lg, 3), top(r, 3));
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
            }
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
