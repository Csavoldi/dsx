//! GPU categorical sampling for temperature=1, top_p>=1. The existing CPU path
//! handles other settings and nonfinite logits. Scratch is allocated lazily after placement.
use anyhow::{Result, ensure};
use crate::gpu::{Arg, Buf, Gpu, Kernel};

pub struct GpuSampler {
    n: usize,
    blocks: usize,
    maxima: Buf,
    bad: Buf,
    weights: Buf,
    sums: Buf,
    output: Buf,
    kernels: [Kernel; 3],
}

pub enum StepOutput {
    Logits(Vec<f32>),
    Token(u32),
}

impl GpuSampler {
    pub fn new(g: &Gpu, n: usize) -> Result<Self> {
        let blocks = n.div_ceil(256);
        ensure!(n > 0 && blocks <= 1024, "GPU sampler requires 1..=262144 logits");
        Ok(Self {
            n, blocks, maxima: g.alloc(blocks * 4)?, bad: g.alloc(blocks * 4)?,
            weights: g.alloc(n * 4)?, sums: g.alloc(blocks * 8)?, output: g.alloc(8)?,
            kernels: [g.kernel("dsx_sample_max")?, g.kernel("dsx_sample_weights")?, g.kernel("dsx_sample_pick")?],
        })
    }

    /// None requests CPU fallback. `uniform_bits` is f64::to_bits() for one variate in [0,1).
    pub fn sample(&self, g: &Gpu, logits: u64, uniform_bits: u64) -> Result<Option<u32>> {
        g.bind();
        let grid = (self.blocks as u32, 1, 1);
        g.launch(self.kernels[0], grid, 256, 0,
            &[Arg::P(logits), Arg::I(self.n as i32), Arg::P(self.maxima.ptr), Arg::P(self.bad.ptr)])?;
        g.launch(self.kernels[1], grid, 256, 0,
            &[Arg::P(logits), Arg::I(self.n as i32), Arg::P(self.maxima.ptr), Arg::I(self.blocks as i32), Arg::P(self.weights.ptr), Arg::P(self.sums.ptr)])?;
        g.launch(self.kernels[2], (1,1,1), 256, 0,
            &[Arg::P(self.weights.ptr), Arg::I(self.n as i32), Arg::P(self.sums.ptr), Arg::P(self.bad.ptr), Arg::I(self.blocks as i32), Arg::P(uniform_bits), Arg::P(self.output.ptr)])?;
        let output: Vec<u32> = g.download(self.output.ptr,2)?;
        Ok((output[1] == 0).then_some(output[0]))
    }

    /// Diagnostic weights and the same block-mass total used by the device CDF.
    /// Call only after a successful sample; this readback is not a fast path.
    pub fn validation_distribution(&self, g: &Gpu) -> Result<(Vec<f32>, f64)> {
        g.bind();
        let weights = g.download(self.weights.ptr, self.n)?;
        let sums: Vec<f64> = g.download(self.sums.ptr, self.blocks)?;
        Ok((weights, sums.iter().sum()))
    }
}
