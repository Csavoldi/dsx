//! Weight sources: DeepSeek's HF checkpoint (safetensors) or a dsx-amd GGUF export of a build (the same tensor
//! names and dtypes, with the re-quantized experts as device tiles). Both look like `SafeTensors` to the loader.

use anyhow::{Context, Result};
use dsx::safetensors::{SafeTensors, StTensor};
use std::collections::HashMap;
use std::path::Path;

pub struct Ck {
    pub tensors: HashMap<String, StTensor>,
    /// codebook experts in an export: tensor name ("....w1.cb") -> (format, multiplier)
    pub cb: HashMap<String, (i32, f32)>,
    st: Option<SafeTensors>,
    gg: Option<dsx::gguf::Gguf>,
}

impl Ck {
    pub fn open(path: &Path) -> Result<Ck> {
        if path.is_dir() {
            let st = SafeTensors::open(path, &[])?;
            return Ok(Ck { tensors: st.tensors.clone(), cb: HashMap::new(), st: Some(st), gg: None });
        }
        let g = dsx::gguf::Gguf::open(path)?;
        let idx: serde_json::Value = serde_json::from_str(g.kv.get("dsxa.index").and_then(|v| v.as_str()).context("dsxa.index (not a dsx-amd export)")?)?;
        let mut tensors = HashMap::new();
        let mut cb = HashMap::new();
        for (name, t) in &g.tensors {
            let e = &idx[name];
            let dtype = e["dtype"].as_str().context("dtype")?.to_string();
            let shape: Vec<usize> = e["shape"].as_array().context("shape")?.iter().map(|x| x.as_u64().unwrap() as usize).collect();
            if let Some(f) = e["fmt"].as_i64() {
                cb.insert(name.clone(), (f as i32, e["mul"].as_f64().unwrap_or(1.0) as f32));
            }
            tensors.insert(name.clone(), StTensor { dtype, shape, file: t.shard, start: t.offset as usize, end: (t.offset + t.bytes) as usize });
        }
        Ok(Ck { tensors, cb, st: None, gg: Some(g) })
    }

    pub fn is_export(&self) -> bool {
        self.gg.is_some()
    }

    pub fn get(&self, name: &str) -> Result<&StTensor> {
        self.tensors.get(name).with_context(|| format!("no tensor {name}"))
    }

    pub fn bytes(&self, t: &StTensor) -> &[u8] {
        match (&self.st, &self.gg) {
            (Some(st), _) => st.bytes(t),
            (_, Some(g)) => &g.maps[t.file][t.start..t.end],
            _ => unreachable!(),
        }
    }
}
