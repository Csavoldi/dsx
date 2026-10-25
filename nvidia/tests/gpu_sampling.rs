#![cfg(feature = "cuda")]
use dsx::{gpu::Gpu, sampling::GpuSampler};
use dsx_core::server::sample::{Rng, Sampler};
use std::time::Instant;

#[test]
fn categorical_support_and_fallback() {
    let g = Gpu::new(0).unwrap();
    let n = 1025;
    let sampler = GpuSampler::new(&g,n).unwrap();
    let mut logits = vec![-1000.0f32;n];
    logits[0]=0.;logits[300]=2f32.ln();logits[700]=3f32.ln();
    let device = g.upload(&logits).unwrap();
    for (u,expected) in [(0.0f64,0u32),(0.1,0),(0.2,300),(0.4,300),(0.6,700),(1.0-2f64.powi(-53),700)] {
        assert_eq!(sampler.sample(&g,device.ptr,u.to_bits()).unwrap(),Some(expected));
    }
    for uniform in [-0.1f64,1.0,f64::NAN] {
        assert!(sampler.sample(&g,device.ptr,uniform.to_bits()).unwrap().is_none());
    }
    for bad in [f32::NAN,f32::INFINITY,f32::NEG_INFINITY] {
        logits[111]=bad;g.htod_sync(device.ptr,&logits).unwrap();
        assert!(sampler.sample(&g,device.ptr,0.4f64.to_bits()).unwrap().is_none());
    }
}

#[test]
fn full_vocabulary_seeded_comparison_and_timing() {
    let g = Gpu::new(0).unwrap();
    let n = 129_280;
    let logits:Vec<f32>=(0..n).map(|i|-((i*7919%40001) as f32)/1000.).collect();
    let device=g.upload(&logits).unwrap();let gpu=GpuSampler::new(&g,n).unwrap();
    let mut cpu=Sampler::new(1.,1.,Some(701));let mut rng=Rng::new(701);let mut changed=0;
    for _ in 0..512 {
        let a=cpu.sample(&logits);let b=gpu.sample(&g,device.ptr,rng.next_f64().to_bits()).unwrap().unwrap();
        changed+=usize::from(a!=b);
    }
    println!("full-vocab seeded draws: {changed}/512 changed (FP32 GPU exp vs FP64 CPU exp)");
    assert!(changed<=2,"GPU CDF disagrees with CPU draws too often: {changed}");
    let calls=400;
    for mode in ["cpu","gpu","gpu","cpu"] {
        let mut cpu=Sampler::new(1.,1.,Some(901));let mut rng=Rng::new(901);let mut checksum=0u64;
        let start=Instant::now();
        for _ in 0..calls {
            let token=if mode=="cpu" {cpu.sample(&logits)} else {gpu.sample(&g,device.ptr,rng.next_f64().to_bits()).unwrap().unwrap()};
            checksum=checksum.wrapping_add(token as u64);
        }
        println!("sampler {mode}: {:.3} ms/call; checksum {checksum}",start.elapsed().as_secs_f64()*1000./calls as f64);
    }
}

#[test]
fn categorical_frequencies() {
    let g=Gpu::new(0).unwrap();let logits=vec![0.,2f32.ln(),3f32.ln(),-100.];
    let device=g.upload(&logits).unwrap();let sampler=GpuSampler::new(&g,logits.len()).unwrap();let mut rng=Rng::new(509);let mut counts=[0usize;4];
    let n=12000;
    for _ in 0..n {counts[sampler.sample(&g,device.ptr,rng.next_f64().to_bits()).unwrap().unwrap() as usize]+=1;}
    for (i,want) in [1./6.,2./6.,3./6.,0.].iter().enumerate() {
        let observed=counts[i] as f64/n as f64;assert!((observed-want).abs()<0.02,"{counts:?}");
    }
    assert_eq!(counts[3],0,"masked tail must not be sampled");
    println!("categorical frequencies: {counts:?}");
}
