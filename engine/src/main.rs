use anyhow::Result;
use clap::{Parser, Subcommand};
use dsx::model::{Model, RefState};
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser)]
#[command(about = "DeepSeek-V4.1-Flash on 2x RTX 3090")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// CPU reference forward, token by token: prints the top-5 after every prompt token, then decodes greedily.
    Ref {
        /// a GGUF export, or DeepSeek's HF checkpoint dir (FP8/FP4: the exact reference)
        #[arg(long)]
        model: PathBuf,
        /// HF checkpoint dir holding the Engram tables (GGUF models only)
        #[arg(long)]
        engram: Option<PathBuf>,
        /// prompt token ids, comma separated
        #[arg(long, value_delimiter = ',')]
        ids: Vec<u32>,
        #[arg(short, default_value_t = 8)]
        n: usize,
        /// write the final logits (f32 LE) here
        #[arg(long)]
        dump_logits: Option<PathBuf>,
    },
    /// DSpark acceptance on the CPU reference (HF checkpoint): greedy main decode, a 5-token draft after every
    /// token, scored against the tokens the main model actually produces next.
    Spec {
        #[arg(long)]
        model: PathBuf,
        #[arg(long, value_delimiter = ',')]
        ids: Vec<u32>,
        #[arg(short, default_value_t = 32)]
        n: usize,
    },
    /// Prefill a prompt in chunks, then decode greedily; --check also runs the prompt token by token and compares.
    #[cfg(feature = "cuda")]
    Gen {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        engram: Option<PathBuf>,
        /// prompt token ids, comma separated, or @file with comma/whitespace separated ids
        #[arg(long)]
        ids: String,
        #[arg(short, default_value_t = 32)]
        n: usize,
        #[arg(long, default_value_t = 32768)]
        ctx: usize,
        #[arg(long, default_value_t = 4096)]
        chunk: usize,
        #[arg(long, default_value_t = 0.0, allow_negative_numbers = true)]
        gpu0_gb: f64,
        #[arg(long, default_value_t = 0.0, allow_negative_numbers = true)]
        gpu1_gb: f64,
        #[arg(long, default_value_t = 0.0)]
        arena_gb: f64,
        #[arg(long)]
        check: bool,
        /// expert hotness ranking (rank.json from calibration)
        #[arg(long)]
        rank: Option<PathBuf>,
        /// after decoding, time verify passes of T tokens (speculative-decoding cost); state is not rolled back
        #[arg(long, default_value_t = 0)]
        bench_verify: usize,
    },
    /// Quality on a held-out set against teacher top-k log-probs (Lucebox's 1017 eval: 8 x 1024 tokens, top-256):
    /// KL(teacher || model) over the teacher's top-k, top-1 agreement, perplexity.
    #[cfg(feature = "cuda")]
    Eval {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        engram: Option<PathBuf>,
        /// directory with heldout.i32 [S][L], teacher_idx.i32 / teacher_lp.f32 [S*(L-1)][K]
        #[arg(long)]
        eval: PathBuf,
        #[arg(long, default_value_t = 8)]
        seqs: usize,
        #[arg(long, default_value_t = 1024)]
        len: usize,
        #[arg(long, default_value_t = 256)]
        topk: usize,
        #[arg(long, default_value_t = 0.0, allow_negative_numbers = true)]
        gpu0_gb: f64,
        #[arg(long, default_value_t = 0.0, allow_negative_numbers = true)]
        gpu1_gb: f64,
        #[arg(long, default_value_t = 0.0)]
        arena_gb: f64,
        #[arg(long)]
        rank: Option<PathBuf>,
        /// feed each sequence token by token through decode (step) instead of one prefill pass
        #[arg(long)]
        decode: bool,
    },
    /// GPU0 decode (experts on the CPU) next to the CPU reference, per-token logit comparison.
    #[cfg(feature = "cuda")]
    Cmp {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        engram: Option<PathBuf>,
        #[arg(long, value_delimiter = ',')]
        ids: Vec<u32>,
        #[arg(long, default_value_t = 4096)]
        ctx: usize,
        /// skip the CPU reference (GPU only, greedy)
        #[arg(long)]
        no_ref: bool,
        #[arg(short, default_value_t = 0)]
        n: usize,
        /// GB of routed experts to keep on GPU0 / on GPU1 (negative = all free memory minus that headroom)
        #[arg(long, default_value_t = 0.0, allow_negative_numbers = true)]
        gpu0_gb: f64,
        #[arg(long, default_value_t = 0.0, allow_negative_numbers = true)]
        gpu1_gb: f64,
        #[arg(long, default_value_t = 0.0)]
        arena_gb: f64,
    },
}

fn top(logits: &[f32], k: usize) -> Vec<(u32, f32)> {
    // partial selection, not a full sort of the 129K vocabulary
    let mut v: Vec<(u32, f32)> = logits.iter().enumerate().map(|(i, &x)| (i as u32, x)).collect();
    let k = k.min(v.len());
    v.select_nth_unstable_by(k - 1, |a, b| b.1.partial_cmp(&a.1).unwrap());
    v.truncate(k);
    v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    v
}

fn load(p: &std::path::Path, engram: Option<&std::path::Path>) -> Result<Model> {
    if p.is_dir() { Model::load_hf(p) } else { Model::load(p, engram) }
}

fn main() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::Ref { model, engram, ids, n, dump_logits } => {
            let t0 = Instant::now();
            let m = load(&model, engram.as_deref())?;
            eprintln!("loaded in {:.1} s", t0.elapsed().as_secs_f32());
            let mut st = RefState::new(&m);
            let mut logits = vec![];
            for (i, &tok) in ids.iter().enumerate() {
                let t = Instant::now();
                logits = m.ref_step(&mut st, tok)?;
                println!("prompt[{i}] tok {tok}: top5 {:?}  ({:.2} s)", top(&logits, 5), t.elapsed().as_secs_f32());
            }
            let mut out = vec![];
            for _ in 0..n {
                let next = top(&logits, 1)[0].0;
                out.push(next);
                let t = Instant::now();
                logits = m.ref_step(&mut st, next)?;
                println!("gen tok {next}: next top5 {:?}  ({:.2} s)", top(&logits, 5), t.elapsed().as_secs_f32());
            }
            println!("generated ids: {out:?}");
            // expert overlap: distinct experts per layer over windows of w consecutive tokens (verify-batch cost)
            let n_tok = st.routes.len();
            for w in [1usize, 2, 4, 5, 6, 8] {
                if n_tok < w { break; }
                let mut tot = 0f64; let mut cnt = 0f64;
                for s0 in 0..=n_tok - w {
                    for l in 0..m.cfg.n_layers {
                        let mut set: Vec<u16> = (s0..s0 + w).flat_map(|t| st.routes[t][l].iter().cloned()).collect();
                        set.sort(); set.dedup();
                        tot += set.len() as f64; cnt += 1.0;
                    }
                }
                println!("window {w}: {:.1} distinct experts per layer (of {} picks)", tot / cnt, w * m.cfg.top_k);
            }
            if let Some(p) = dump_logits {
                let b: Vec<u8> = logits.iter().flat_map(|v| v.to_le_bytes()).collect();
                std::fs::write(p, b)?;
            }
        }
        Cmd::Spec { model, ids, n } => {
            let m = Model::load_hf(&model)?;
            let st_hf = dsx::safetensors::SafeTensors::open(&model, &[])?;
            let ds = dsx::dspark::Dspark::load(&m, &st_hf, &model)?;
            let mut dst = ds.new_state(&m);
            let mut st = RefState::new(&m);
            let mut logits = vec![];
            for &tok in &ids {
                logits = m.ref_step(&mut st, tok)?;
                ds.observe(&m, &mut dst, &st.main_hidden, st.pos - 1);
            }
            let mut drafts: Vec<(usize, Vec<u32>, Vec<f32>)> = vec![];
            let mut seq: Vec<u32> = vec![];
            for _ in 0..n {
                let next = top(&logits, 1)[0].0;
                // draft from the newest position before feeding `next`
                let (d, cf) = ds.draft(&m, &dst, st.pos - 1, next);
                drafts.push((seq.len() + 1, d, cf));
                seq.push(next);
                logits = m.ref_step(&mut st, next)?;
                ds.observe(&m, &mut dst, &st.main_hidden, st.pos - 1);
            }
            let mut hist = vec![0usize; ds.block + 1];
            for (start, d, cf) in &drafts {
                let mut a = 0;
                while a < d.len() && start + a < seq.len() && d[a] == seq[start + a] {
                    a += 1;
                }
                if start + d.len() <= seq.len() {
                    hist[a] += 1;
                }
                println!("at {start}: drafts {d:?} conf {:?} accepted {a}", cf.iter().map(|x| (x * 100.0).round() / 100.0).collect::<Vec<_>>());
            }
            let tot: usize = hist.iter().sum();
            let mean = hist.iter().enumerate().map(|(a, n)| a * n).sum::<usize>() as f64 / tot.max(1) as f64;
            println!("accepted-draft histogram {hist:?}; mean {mean:.2} drafts accepted per step (+1 main token)");
            // verify-batch cost: distinct experts per layer over windows of consecutive tokens
            let n_tok = st.routes.len();
            for w in [1usize, 2, 4, 6] {
                if n_tok < w { break; }
                let (mut tot, mut cnt) = (0f64, 0f64);
                for s0 in 0..=n_tok - w {
                    for l in 0..m.cfg.n_layers {
                        let mut set: Vec<u16> = (s0..s0 + w).flat_map(|t| st.routes[t][l].iter().cloned()).collect();
                        set.sort();
                        set.dedup();
                        tot += set.len() as f64;
                        cnt += 1.0;
                    }
                }
                println!("window {w}: {:.1} distinct experts per layer (of {} picks)", tot / cnt, w * m.cfg.top_k);
            }
            println!("generated ids: {seq:?}");
        }
        #[cfg(feature = "cuda")]
        Cmd::Gen { model, engram, ids, n, ctx, chunk, gpu0_gb, gpu1_gb, arena_gb, check, rank, bench_verify } => {
            let ids: Vec<u32> = if let Some(f) = ids.strip_prefix('@') {
                std::fs::read_to_string(f)?.split(|ch: char| ch == ',' || ch.is_whitespace()).filter(|x| !x.is_empty()).map(|x| x.parse().unwrap()).collect()
            } else {
                ids.split(',').map(|x| x.trim().parse().unwrap()).collect()
            };
            let m = load(&model, engram.as_deref())?;
            let mut gm = dsx::gpumodel::GpuModel::with_placement(&m, 0, ctx, dsx::gpumodel::Placement { gpu0_gb, gpu1_gb, arena_gb, prefill_chunk: chunk, rank: rank.clone() })?;
            let pb = gm.pb.take().unwrap();
            // chunk plan: the final chunk must hold the decoder cone (19 x 128 tokens) or the whole prompt
            let cone = (m.cfg.n_layers - 1 - m.cfg.cand_layer) * m.cfg.window;
            let total = ids.len();
            let mut bounds = vec![];
            let mut s0 = 0;
            while s0 < total {
                let mut e = (s0 + chunk).min(total);
                if e < total && total - e < cone {
                    e = total.saturating_sub(cone).max(s0 + 1);
                }
                bounds.push((s0, e));
                s0 = e;
            }
            let t0 = Instant::now();
            let mut logits = vec![];
            for (i, &(a, b)) in bounds.iter().enumerate() {
                let tc = Instant::now();
                let r = gm.prefill_chunk(&pb, &ids[a..b], i + 1 == bounds.len())?;
                eprintln!("prefill chunk {a}..{b}: {:.2} s", tc.elapsed().as_secs_f64());
                if let Some(l) = r { logits = l; }
            }
            let tp = t0.elapsed().as_secs_f64();
            println!("prefill {total} tokens in {tp:.2} s ({:.0} tok/s); next top5 {:?}", total as f64 / tp, top(&logits, 5));
            if let Some(p) = &gm.prof {
                for (k, v) in p {
                    println!("  {k:<28} {:.2} s", v);
                }
            }
            let mut out = vec![];
            if let Some(p) = gm.prof.as_mut() {
                p.clear();
            }
            if let Some(p) = gm.evprof.as_mut() {
                p.clear();
            }
            let busy0 = gm.cpu_busy_s();
            let (p0, c0) = gm.cpu_picks();
            let ad0 = gm.adapt_s;
            let split0 = gm.cpu_split_s();
            let td = Instant::now();
            for _ in 0..n {
                let next = top(&logits, 1)[0].0;
                out.push(next);
                logits = gm.step(next)?;
            }
            let el = td.elapsed().as_secs_f64();
            let (p1, c1) = gm.cpu_picks();
            println!(
                "decode {n} tokens in {el:.2} s ({:.2} tok/s); CPU experts busy {:.1} ms/token, {:.1}% of picks on the CPU; GPU0 expert swaps {} ({:.2} ms/token host)",
                n as f64 / el,
                (gm.cpu_busy_s() - busy0) * 1e3 / n as f64,
                100.0 * (c1 - c0) as f64 / (p1 - p0).max(1) as f64,
                gm.adapt_swaps,
                (gm.adapt_s - ad0) * 1e3 / n as f64
            );
            if let Some(p) = &gm.prof {
                for (k, v) in p {
                    println!("  {k:<28} {:.2} ms/token", v * 1e3 / n as f64);
                }
            }
            let legs = gm.handoff_legs_us();
            if !legs.is_empty() && std::env::var_os("DSX_SPIN_WAIT").is_some() {
                let s: Vec<String> = legs.iter().enumerate().map(|(l, (a, b))| format!("{l}:{a:.0}/{b:.0}")).collect();
                println!("  handoff legs us (GPU->CPU / CPU->GPU): {}", s.join(" "));
            }
            let sp = gm.cpu_split_s();
            println!("  CPU worker idle / prep / forward: {:.2} / {:.2} / {:.2} ms/token", (sp[0] - split0[0]) * 1e3 / n as f64, (sp[1] - split0[1]) * 1e3 / n as f64, (sp[2] - split0[2]) * 1e3 / n as f64);
            if let Some(p) = &gm.evprof {
                let steps = p.get("steps").copied().unwrap_or(1.0);
                for (k, v) in p.iter().filter(|(k, _)| **k != "steps") {
                    println!("  {k:<40} {:.2} ms/token", v / steps);
                }
            }
            println!("generated: {out:?}");
            if bench_verify > 0 {
                // the same tokens again, T at a time: what one speculative verify pass costs next to a decode step
                let mut times = vec![];
                for i in 0..6 {
                    let toks: Vec<u32> = (0..bench_verify).map(|j| out[(i + j) % out.len()]).collect();
                    let t = Instant::now();
                    let lg = gm.verify(&pb, &toks)?;
                    times.push(t.elapsed().as_secs_f64());
                    let _ = lg;
                }
                times.sort_by(|a, b| a.partial_cmp(b).unwrap());
                println!("verify T={bench_verify}: median {:.1} ms per pass ({:.1} ms per token if all accepted) vs decode {:.1} ms per token",
                    times[3] * 1e3, times[3] * 1e3 / bench_verify as f64, el * 1e3 / n as f64);
            }
            if check {
                drop(gm);
                let mut g2 = dsx::gpumodel::GpuModel::with_placement(&m, 0, ctx, dsx::gpumodel::Placement { gpu0_gb, gpu1_gb, arena_gb, prefill_chunk: 0, rank: rank.clone() })?;
                let mut l2 = vec![];
                for &t in &ids { l2 = g2.step(t)?; }
                let maxd = l2.iter().zip(&logits).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
                let _ = maxd;
                println!("token-by-token next top5 {:?}", top(&l2, 5));
                let mut out2 = vec![];
                for _ in 0..n {
                    let next = top(&l2, 1)[0].0;
                    out2.push(next);
                    l2 = g2.step(next)?;
                }
                let same = out.iter().zip(&out2).take_while(|(a, b)| a == b).count();
                println!("token-by-token generated: {out2:?}\nidentical prefix: {same}/{n}");
            }
        }
        #[cfg(feature = "cuda")]
        Cmd::Eval { model, engram, eval, seqs, len, topk, gpu0_gb, gpu1_gb, arena_gb, rank, decode } => {
            let rd = |f: &str| std::fs::read(eval.join(f));
            let toks: Vec<u32> = rd("heldout.i32")?.chunks(4).map(|b| i32::from_le_bytes(b.try_into().unwrap()) as u32).collect();
            let tidx: Vec<i32> = rd("teacher_idx.i32")?.chunks(4).map(|b| i32::from_le_bytes(b.try_into().unwrap())).collect();
            let tlp: Vec<f32> = rd("teacher_lp.f32")?.chunks(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
            let m = load(&model, engram.as_deref())?;
            let mut gm = dsx::gpumodel::GpuModel::with_placement(&m, 0, len + 64, dsx::gpumodel::Placement { gpu0_gb, gpu1_gb, arena_gb, prefill_chunk: len, rank })?;
            let (mut nll, mut kl, mut agree, mut cnt) = (0f64, 0f64, 0usize, 0usize);
            let t0 = Instant::now();
            for sq in 0..seqs {
                gm.reset();
                let seq = &toks[sq * len..(sq + 1) * len];
                let lg = if decode { seq[..len - 1].iter().map(|&tk| gm.step(tk)).collect::<Result<Vec<_>>>()? } else { gm.eval_logits(seq)? };
                for p in 0..len - 1 {
                    let l = &lg[p];
                    let mx = l.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
                    let lse = mx + l.iter().map(|&v| (v as f64 - mx).exp()).sum::<f64>().ln();
                    let lq = |i: usize| l[i] as f64 - lse;
                    nll -= lq(seq[p + 1] as usize);
                    let row = sq * (len - 1) + p;
                    let (ti, tl) = (&tidx[row * topk..(row + 1) * topk], &tlp[row * topk..(row + 1) * topk]);
                    kl += ti.iter().zip(tl).map(|(&i, &lt)| (lt as f64).exp() * (lt as f64 - lq(i as usize))).sum::<f64>();
                    let am = l.iter().enumerate().fold((0usize, f32::NEG_INFINITY), |a, (i, &v)| if v > a.1 { (i, v) } else { a }).0;
                    agree += (am as i32 == ti[0]) as usize;
                    cnt += 1;
                }
                eprintln!("seq {sq}: running ppl {:.3} kl {:.4} top1 {:.4} ({:.0} s)", (nll / cnt as f64).exp(), kl / cnt as f64, agree as f64 / cnt as f64, t0.elapsed().as_secs_f64());
            }
            println!("{{\"tokens\": {cnt}, \"nll\": {:.6}, \"ppl\": {:.4}, \"kl_top{topk}\": {:.6}, \"top1_agree\": {:.6}}}", nll / cnt as f64, (nll / cnt as f64).exp(), kl / cnt as f64, agree as f64 / cnt as f64);
        }
        #[cfg(feature = "cuda")]
        Cmd::Cmp { model, engram, ids, ctx, no_ref, n, gpu0_gb, gpu1_gb, arena_gb } => {
            let m = load(&model, engram.as_deref())?;
            let mut gm = dsx::gpumodel::GpuModel::with_placement(&m, 0, ctx, dsx::gpumodel::Placement { gpu0_gb, gpu1_gb, arena_gb, prefill_chunk: 0, rank: None })?;
            let t_all = Instant::now();
            let mut st = RefState::new(&m);
            let mut toks = ids.clone();
            let mut i = 0;
            while i < toks.len() {
                let tok = toks[i];
                let t = Instant::now();
                let gl = gm.step(tok)?;
                let tg = t.elapsed().as_secs_f32();
                let gt = top(&gl, 5);
                if no_ref {
                    println!("[{i}] tok {tok}: gpu top5 {gt:?} ({tg:.3} s)");
                } else {
                    let t = Instant::now();
                    let rl = m.ref_step(&mut st, tok)?;
                    let tr = t.elapsed().as_secs_f32();
                    let rt = top(&rl, 5);
                    let maxd = gl.iter().zip(&rl).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
                    println!("[{i}] tok {tok}: gpu {:?} ref {:?} max|dlogit| {maxd:.4} (gpu {tg:.3} s, ref {tr:.2} s)", &gt[..3], &rt[..3]);
                }
                if i + 1 == toks.len() && toks.len() < ids.len() + n {
                    toks.push(gt[0].0);
                }
                i += 1;
            }
            println!("ids: {toks:?}");
            let el = t_all.elapsed().as_secs_f64();
            if let Some(p) = &gm.prof {
                for (k, v) in p {
                    println!("  {k:<10} {:.1} ms/token", v * 1e3 / toks.len() as f64);
                }
            }
            println!("{} tokens in {el:.1} s ({:.2} tok/s); CPU experts {:.1} s; routed evals GPU0/GPU1/CPU {:?}", toks.len(), toks.len() as f64 / el, gm.t_experts, gm.n_routed);
        }
    }
    Ok(())
}
