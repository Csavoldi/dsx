//! Compare actual GPU categorical weights with the CPU distribution on saved logits.
use anyhow::{Result, ensure};
use clap::Parser;
use dsx::{gpu::Gpu, sampling::GpuSampler};
use dsx_core::server::sample::{Rng, Sampler};
use std::{io::Read, path::PathBuf};

#[derive(Parser)]
struct Args {
    #[arg(long)] logits: PathBuf,
    #[arg(long, default_value_t=129280)] vocab: usize,
    #[arg(long)] seed: u64,
}

fn main() -> Result<()> {
    let a=Args::parse();
    ensure!(a.vocab>0, "empty vocabulary");
    let file=std::fs::File::open(&a.logits)?;
    let bytes=file.metadata()?.len() as usize;
    ensure!(bytes>0 && bytes%(a.vocab*4)==0, "logit file must contain complete F32 vocabulary rows");
    let rows=bytes/(a.vocab*4);
    let mut input=std::io::BufReader::new(file);
    let g=Gpu::new(0)?;
    let device=g.alloc(a.vocab*4)?;
    let gpu=GpuSampler::new(&g,a.vocab)?;
    let mut cpu=Sampler::new(1.0,1.0,Some(a.seed));
    let mut rng=Rng::new(a.seed);
    let mut buffer=vec![0u8;a.vocab*4];
    let mut metrics=Vec::with_capacity(rows);
    for row in 0..rows {
        input.read_exact(&mut buffer)?;
        let logits:Vec<f32>=buffer.chunks_exact(4).map(|v|f32::from_le_bytes(v.try_into().unwrap())).collect();
        ensure!(logits.iter().all(|v|v.is_finite()), "nonfinite logit in row {row}");
        g.htod_sync(device.ptr,&logits)?;
        let expected=cpu.sample(&logits);
        let selected=gpu.sample(&g,device.ptr,rng.next_f64().to_bits())?.ok_or_else(||anyhow::anyhow!("unexpected GPU fallback in row {row}"))?;
        let (weights,total)=gpu.validation_distribution(&g)?;
        ensure!(total>0.0 && total.is_finite(), "invalid device mass in row {row}");
        let maximum=logits.iter().copied().fold(f32::NEG_INFINITY,f32::max);
        let cpu_weights:Vec<f64>=logits.iter().map(|&v|if v>=maximum-30.0 {((v-maximum) as f64).exp()} else {0.0}).collect();
        let mass: f64=cpu_weights.iter().sum();
        let (mut kl,mut tv,mut max_abs,mut qsum)=(0.0f64,0.0f64,0.0f64,0.0f64);
        let (mut pcdf,mut qcdf,mut overlap,mut max_cdf)=(0.0f64,0.0f64,0.0f64,0.0f64);
        for (&p,&q) in cpu_weights.iter().zip(&weights) {
            let (p,q)=(p/mass,q as f64/total);
            ensure!(q.is_finite() && q>=0.0 && (p==0.0 || q>0.0), "support mismatch in row {row}");
            qsum+=q;
            if p>0.0 {kl+=p*(p/q).ln();}
            tv+=(p-q).abs();max_abs=max_abs.max((p-q).abs());
            let (pnext,qnext)=(pcdf+p,qcdf+q);
            overlap+=(pnext.min(qnext)-pcdf.max(qcdf)).max(0.0);
            max_cdf=max_cdf.max((pnext-qnext).abs());pcdf=pnext;qcdf=qnext;
        }
        metrics.push(serde_json::json!({"row":row,"kl_cpu_to_gpu":kl.max(0.0),"total_variation":tv*0.5,"max_abs_probability_difference":max_abs,"gpu_probability_sum":qsum,"cdf_interval_disagreement_estimate":(1.0-overlap).max(0.0),"max_cdf_difference":max_cdf,"cpu_token":expected,"gpu_token":selected,"changed_draw":expected!=selected}));
    }
    let mean=|key:&str|metrics.iter().map(|x|x[key].as_f64().unwrap()).sum::<f64>()/rows as f64;
    let maximum=|key:&str|metrics.iter().map(|x|x[key].as_f64().unwrap()).fold(0.0f64,f64::max);
    println!("{}",serde_json::json!({"event":"sampler_distribution_audit","input":a.logits,"vocab":a.vocab,"rows":rows,"seed":a.seed,"mean_kl_cpu_to_gpu":mean("kl_cpu_to_gpu"),"max_kl_cpu_to_gpu":maximum("kl_cpu_to_gpu"),"max_total_variation":maximum("total_variation"),"max_abs_probability_difference":maximum("max_abs_probability_difference"),"max_cdf_interval_disagreement_estimate":maximum("cdf_interval_disagreement_estimate"),"mean_cdf_interval_disagreement_estimate":mean("cdf_interval_disagreement_estimate"),"max_cdf_difference":maximum("max_cdf_difference"),"changed_draws":metrics.iter().filter(|x|x["changed_draw"]==true).count(),"row_metrics":metrics,"limit":"Actual FP32 GPU weights versus FP64 CPU weights at temperature/top-p 1. CDF interval disagreement is a continuous-uniform estimate; finite RNG precision and double accumulation can add boundary effects; this is not end-task quality or production throughput."}));
    Ok(())
}
