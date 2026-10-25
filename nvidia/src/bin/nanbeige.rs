use anyhow::{Result, anyhow};
use clap::Parser;
use dsx::nanbeige::Nanbeige;
use std::{io::Write, path::PathBuf, time::Instant};
#[derive(Parser)]
struct Args {
    #[arg(
        long,
        default_value = "/home/lxe/nanbeige-runner/models/Nanbeige4.3-33B-NVFP4"
    )]
    model: PathBuf,
    #[arg(long, default_value_t = 4096)]
    ctx: usize,
    #[arg(long, default_value_t = 16)]
    kv_bits: usize,
    #[arg(long)]
    host_embed: bool,
    #[arg(long)]
    head_q8: bool,
    #[arg(long, default_value_t = 0)]
    head_bits: usize,
    #[arg(long, default_value_t = 0)]
    lookup: usize,
    #[arg(long)]
    inspect: bool,
    #[arg(long)]
    bandwidth: bool,
    #[arg(long)]
    cache_check: bool,
    #[arg(long)]
    exl3_check: bool,
    #[arg(long)]
    attention_check: bool,
    #[arg(long)]
    attention_bench: bool,
    #[arg(long)]
    exl3_bench: bool,
    #[arg(long)]
    reference: bool,
    #[arg(long)]
    validate: bool,
    #[arg(long)]
    no_graphs: bool,
    #[arg(long)]
    simt_decode: bool,
    #[arg(long)]
    prefill_one: bool,
    #[arg(long)]
    prompt: Option<String>,
    #[arg(long)]
    tokens: Option<PathBuf>,
    #[arg(long, default_value_t = 64)]
    max_tokens: usize,
    #[arg(long, default_value_t = 0)]
    ppl_tail: usize,
    #[arg(long)]
    ppl_logits: Option<PathBuf>,
    #[arg(long)]
    receipt: Option<PathBuf>,
    #[arg(long)]
    logits: Option<PathBuf>,
}
fn main() -> Result<()> {
    let a = Args::parse();
    if a.exl3_bench { let r=dsx::nanbeige::benchmark_exl3()?; println!("{r}");if let Some(p)=a.receipt {std::fs::write(p,serde_json::to_vec_pretty(&r)?)?;}return Ok(()); }
    if a.attention_bench { let r=dsx::nanbeige::benchmark_attention()?; println!("{r}");if let Some(p)=a.receipt {std::fs::write(p,serde_json::to_vec_pretty(&r)?)?;}return Ok(()); }
    if a.attention_check { let r=dsx::nanbeige::validate_attention()?; println!("{r}");if let Some(p)=a.receipt {std::fs::write(p,serde_json::to_vec_pretty(&r)?)?;}return Ok(()); }
    if a.exl3_check { let r=dsx::nanbeige::validate_exl3()?; println!("{r}");if let Some(p)=a.receipt {std::fs::write(p,serde_json::to_vec_pretty(&r)?)?;}return Ok(()); }
    if a.bandwidth {
        let result = dsx::nanbeige::benchmark_memory()?;
        println!("{result}");
        if let Some(p) = a.receipt {
            std::fs::write(p, serde_json::to_vec_pretty(&result)?)?;
        }
        return Ok(());
    }
    if a.inspect {
        println!("{}", Nanbeige::inspect(&a.model)?);
        return Ok(());
    }
    let tokenizer = tokenizers::Tokenizer::from_file(a.model.join("tokenizer.json"))
        .map_err(|e| anyhow!("{e}"))?;
    let input: Vec<u32> = if let Some(p) = &a.tokens {
        serde_json::from_slice(&std::fs::read(p)?)?
    } else {
        let text=a.prompt.as_deref().unwrap_or("<|im_start|>user\nWhat is 2 + 2? Answer briefly.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
        tokenizer
            .encode(text, false)
            .map_err(|e| anyhow!("{e}"))?
            .get_ids()
            .to_vec()
    };
    let load = Instant::now();
    let mut m = Nanbeige::load_with_cache(
        &a.model,
        a.ctx,
        a.kv_bits,
        a.host_embed,
        if a.head_q8 { 8 } else { a.head_bits },
    )?;
    let weight_format=m.weight_format().to_owned();
    m.reference = a.reference;
    m.simt_decode = a.simt_decode;
    m.graphs_enabled = !a.no_graphs;
    if a.validate {
        let v = m.validate_linear()?;
        println!("{v}");
        if let Some(p) = &a.receipt {
            std::fs::write(p, serde_json::to_vec_pretty(&v)?)?;
        }
        return Ok(());
    }
    let load_s = load.elapsed().as_secs_f64();
    if a.cache_check { let mut e=dsx::nanbeige::Engine::new(m); let r=e.audit_prefix_cache(&input)?;println!("{r}");if let Some(p)=a.receipt {std::fs::write(p,serde_json::to_vec_pretty(&r)?)?;}return Ok(()); }

    if a.ppl_tail > 0 {
        anyhow::ensure!(
            a.ppl_tail < input.len(),
            "ppl-tail must be smaller than input"
        );
        let prefix = input.len() - a.ppl_tail;
        let start = Instant::now();
        m.prefill(&input[..prefix])?;
        let prefill_s = start.elapsed().as_secs_f64();
        let start = Instant::now();
        let mut losses = vec![];
        let mut hits = 0;
        let mut eval_logits=a.ppl_logits.as_ref().map(std::fs::File::create).transpose()?.map(std::io::BufWriter::new);
        for (i, token) in input[prefix..].iter().enumerate() {
            let logits = m.read_logits()?;
            anyhow::ensure!(logits.iter().all(|v| v.is_finite()), "nonfinite logits");
            if let Some(file)=eval_logits.as_mut() {
                let bytes:Vec<u8>=logits.iter().flat_map(|v|v.to_le_bytes()).collect();
                file.write_all(&bytes)?;
            }
            let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
            let sum: f64 = logits.iter().map(|v| (*v as f64 - max).exp()).sum();
            losses.push(max + sum.ln() - logits[*token as usize] as f64);
            if logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0
                == *token as usize
            {
                hits += 1;
            }
            m.forward(&[*token])?;
            if i % 32 == 31 {
                eprintln!("eval {}/{}", i + 1, a.ppl_tail);
            }
        }
        m.g.sync()?;
        if let Some(file)=eval_logits.as_mut() {file.flush()?;}
        if let Some(p) = &a.logits {
            let bytes: Vec<u8> = m
                .read_logits()?
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            std::fs::write(p, bytes)?;
        }
        let mean = losses.iter().sum::<f64>() / losses.len() as f64;
        let r = serde_json::json!({"input_tokens":input.len(),"evaluated_tokens":a.ppl_tail,"mean_nll":mean,"ppl":mean.exp(),"top1_hits":hits,"losses":losses,"prefill_seconds":prefill_s,"evaluation_seconds":start.elapsed().as_secs_f64(),"gpu_free_bytes":m.g.mem_info()?.0,"attention_partition_tokens":std::env::var("DSX_NB_PART_SIZE").unwrap_or_else(|_|"32768".into()),"prefill_batch":std::env::var("DSX_NB_BATCH").unwrap_or_else(|_|"512".into()),"prefill_attention":std::env::var("DSX_NB_ATTN8").unwrap_or_else(|_|"0".into()),"weights":weight_format,"ctx":a.ctx,"head_q8":a.head_q8,"head_bits":if a.head_q8{8}else{a.head_bits},"kv_bits":a.kv_bits,"host_embedding":a.host_embed});
        println!("{r}");
        if let Some(p) = a.receipt {
            std::fs::write(p, serde_json::to_vec_pretty(&r)?)?;
        }
        return Ok(());
    }
    if a.lookup > 0 {
        use dsx::server::{Engine as _, GenParams};
        use std::sync::Arc;
        let mut engine = dsx::nanbeige::Engine::new(m);
        engine.lookup = a.lookup;
        let params = GenParams {
            max_new: a.max_tokens,
            temperature: 0.,
            top_p: 1.,
            seed: Some(42),
            stop_token_ids: vec![166101],
            hesitation: None,
            thinking_close: None,
            images: Arc::new(vec![]),
        };
        let mut output = vec![];
        let start = Instant::now();
        engine.generate(&input, &params, &mut |t| {
            output.push(t);
            true
        })?;
        let stats = engine.last_stats();
        let text = tokenizer
            .decode(&output, false)
            .map_err(|e| anyhow!("{e}"))?;
        let r = serde_json::json!({"engine":"dsx-nanbeige","attention_partition_tokens":std::env::var("DSX_NB_PART_SIZE").unwrap_or_else(|_|"32768".into()),"prefill_batch":std::env::var("DSX_NB_BATCH").unwrap_or_else(|_|"512".into()),"prefill_attention":std::env::var("DSX_NB_ATTN8").unwrap_or_else(|_|"0".into()),"weights":weight_format,"ctx":a.ctx,"kv_bits":a.kv_bits,"head_q8":a.head_q8,"head_bits":if a.head_q8{8}else{a.head_bits},"input_tokens":input.len(),"output_tokens":output.len(),"output_ids":output,"text":text,"prefill_seconds":stats.prefill.unwrap().as_secs_f64(),"decode_seconds":stats.decode.unwrap().as_secs_f64(),"decode_tps":output.len() as f64/stats.decode.unwrap().as_secs_f64(),"wall_seconds":start.elapsed().as_secs_f64(),"lookup":engine.lookup_stats});
        println!("{r}");
        if let Some(p) = a.receipt {
            std::fs::write(p, serde_json::to_vec_pretty(&r)?)?;
        }
        return Ok(());
    }
    let t = Instant::now();
    if a.prefill_one {
        for token in &input {
            m.forward(&[*token])?;
        }
        m.g.sync()?;
    } else {
        m.prefill(&input)?;
    }
    let prefill_s = t.elapsed().as_secs_f64();
    let mut logits = m.read_logits()?;
    if let Some(p) = &a.logits {
        let bytes: Vec<u8> = logits.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(p, bytes)?;
    }
    let start = Instant::now();
    let mut output = vec![];
    for _ in 0..a.max_tokens {
        let next = dsx::server::sample::argmax(&logits);
        if next == 166101 {
            break;
        }
        output.push(next);
        m.forward(&[next])?;
        logits = m.read_logits()?;
    }
    let decode_s = start.elapsed().as_secs_f64();
    let text = tokenizer
        .decode(&output, false)
        .map_err(|e| anyhow!("{e}"))?;
    println!("{text}");
    std::io::stdout().flush()?;
    let r = serde_json::json!({"engine":"dsx-nanbeige","quant":weight_format,"simt_decode":a.simt_decode,"cuda_graphs":!a.no_graphs,"prefill_one":a.prefill_one,"gpu_free_bytes":m.g.mem_info()?.0,"attention_partition_tokens":std::env::var("DSX_NB_PART_SIZE").unwrap_or_else(|_|"32768".into()),"prefill_batch":std::env::var("DSX_NB_BATCH").unwrap_or_else(|_|"512".into()),"prefill_attention":std::env::var("DSX_NB_ATTN8").unwrap_or_else(|_|"0".into()),"weights":weight_format,"ctx":a.ctx,"head_q8":a.head_q8,"head_bits":if a.head_q8{8}else{a.head_bits},"kv_bits":a.kv_bits,"host_embedding":a.host_embed,"input_tokens":input.len(),"output_tokens":output.len(),"load_seconds":load_s,"prefill_seconds":prefill_s,"decode_seconds":decode_s,"prefill_tps":input.len() as f64/prefill_s,"decode_tps":output.len() as f64/decode_s,"output_ids":output,"text":text});
    eprintln!("{r}");
    if let Some(p) = a.receipt {
        std::fs::write(p, serde_json::to_vec_pretty(&r)?)?;
    }
    Ok(())
}
