//! dsx: DeepSeek-V4.1-Flash(-1017) inference tuned for one box (2x RTX 3090, i7-12700K, 128 GB DDR4-3200).

#[cfg(feature = "cuda")]
pub mod attn;
pub mod cpu;
#[cfg(feature = "cuda")]
pub mod gpu;
pub mod dspark;
#[cfg(feature = "cuda")]
pub mod experts;
pub mod formats;
pub mod gguf;
#[cfg(feature = "cuda")]
pub mod handoff;
#[cfg(feature = "cuda")]
pub mod gpumodel;
pub mod model;
pub mod numerics;
pub mod q6;
pub mod safetensors;
pub mod server;
#[cfg(feature = "cuda")]
pub mod topk;
