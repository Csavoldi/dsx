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
    #[arg(long)]
    inspect: bool,
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
    #[arg(long)]
    receipt: Option<PathBuf>,
    #[arg(long)]
    logits: Option<PathBuf>,
}
fn main() -> Result<()> {
    let a = Args::parse();
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
    let mut m = Nanbeige::load(&a.model, a.ctx)?;
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
        let next = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0 as u32;
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
    let r = serde_json::json!({"engine":"dsx-nanbeige","quant":if a.reference{"NVFP4 W4A16 reference"}else{"NVFP4 W4A4 native SM120 cuBLASLt"},"simt_decode":a.simt_decode,"cuda_graphs":!a.no_graphs,"prefill_one":a.prefill_one,"ctx":a.ctx,"input_tokens":input.len(),"output_tokens":output.len(),"load_seconds":load_s,"prefill_seconds":prefill_s,"decode_seconds":decode_s,"prefill_tps":input.len() as f64/prefill_s,"decode_tps":output.len() as f64/decode_s,"output_ids":output,"text":text});
    eprintln!("{r}");
    if let Some(p) = a.receipt {
        std::fs::write(p, serde_json::to_vec_pretty(&r)?)?;
    }
    Ok(())
}
