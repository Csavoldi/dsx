//! GGUF v3 reader over mmap, with llama.cpp split files (`-00001-of-0000N.gguf`) read as one model.

use anyhow::{Context, Result, bail};
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    Str(String),
    Arr(Vec<Value>),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl Value {
    pub fn as_u64(&self) -> Option<u64> {
        Some(match *self {
            Value::U8(v) => v as u64,
            Value::U16(v) => v as u64,
            Value::U32(v) => v as u64,
            Value::U64(v) => v,
            Value::I8(v) if v >= 0 => v as u64,
            Value::I16(v) if v >= 0 => v as u64,
            Value::I32(v) if v >= 0 => v as u64,
            Value::I64(v) if v >= 0 => v as u64,
            _ => return None,
        })
    }
    pub fn as_f64(&self) -> Option<f64> {
        Some(match *self {
            Value::F32(v) => v as f64,
            Value::F64(v) => v,
            _ => return self.as_u64().map(|v| v as f64),
        })
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }
}

/// ggml type ids used by the V4.1 GSQ files (143 is llama-tq's PTQ1_0, not upstream ggml).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum GType {
    F32 = 0,
    F16 = 1,
    Q8_0 = 8,
    Q3K = 11,
    Q6K = 14,
    I8 = 24,
    I32 = 26,
    BF16 = 30,
    Q2_0 = 42,
    Ptq1_0 = 143,
    /// HF checkpoint FP8 E4M3, E8M0 scale per 32x32 block (not a GGUF type)
    Fp8B32 = 1000,
    /// HF checkpoint FP4 E2M1, two per byte (low nibble first), E8M0 scale per 32 (not a GGUF type)
    Fp4 = 1001,
}

impl GType {
    pub fn from_id(id: u32) -> Result<GType> {
        Ok(match id {
            0 => GType::F32,
            1 => GType::F16,
            8 => GType::Q8_0,
            11 => GType::Q3K,
            14 => GType::Q6K,
            24 => GType::I8,
            26 => GType::I32,
            30 => GType::BF16,
            42 => GType::Q2_0,
            143 => GType::Ptq1_0,
            _ => bail!("unsupported ggml type {id}"),
        })
    }
    /// (weights per block, bytes per block)
    pub fn block(self) -> (usize, usize) {
        match self {
            GType::F32 => (1, 4),
            GType::F16 | GType::BF16 => (1, 2),
            GType::I8 => (1, 1),
            GType::I32 => (1, 4),
            GType::Q8_0 => (32, 34),
            GType::Q3K => (256, 110),
            GType::Q6K => (256, 210),
            GType::Q2_0 => (64, 18),
            GType::Ptq1_0 => (128, 28),
            GType::Fp8B32 => (1, 1),
            GType::Fp4 => (2, 1),
        }
    }
    pub fn row_bytes(self, n: usize) -> usize {
        let (w, b) = self.block();
        assert!(n % w == 0, "{self:?}: row of {n} is not a whole number of blocks");
        n / w * b
    }
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    /// ggml order: dims[0] is the contiguous (row) dimension.
    pub dims: Vec<u64>,
    pub ty: GType,
    pub shard: usize,
    /// absolute byte offset in the shard file
    pub offset: u64,
    pub bytes: u64,
}

impl TensorInfo {
    pub fn n_elements(&self) -> u64 {
        self.dims.iter().product()
    }
    pub fn row_len(&self) -> usize {
        self.dims[0] as usize
    }
    pub fn n_rows(&self) -> usize {
        (self.n_elements() / self.dims[0]) as usize
    }
}

pub struct Gguf {
    pub kv: HashMap<String, Value>,
    pub tensors: HashMap<String, TensorInfo>,
    pub order: Vec<String>,
    pub paths: Vec<PathBuf>,
    pub maps: Vec<Mmap>,
}

struct Cursor<'a> {
    b: &'a [u8],
    p: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.p + n > self.b.len() {
            bail!("gguf header truncated at {}", self.p);
        }
        let s = &self.b[self.p..self.p + n];
        self.p += n;
        Ok(s)
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into()?))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into()?))
    }
    fn string(&mut self) -> Result<String> {
        let n = self.u64()? as usize;
        Ok(String::from_utf8_lossy(self.take(n)?).into_owned())
    }
    fn value(&mut self, ty: u32) -> Result<Value> {
        Ok(match ty {
            0 => Value::U8(self.take(1)?[0]),
            1 => Value::I8(self.take(1)?[0] as i8),
            2 => Value::U16(u16::from_le_bytes(self.take(2)?.try_into()?)),
            3 => Value::I16(i16::from_le_bytes(self.take(2)?.try_into()?)),
            4 => Value::U32(self.u32()?),
            5 => Value::I32(self.u32()? as i32),
            6 => Value::F32(f32::from_le_bytes(self.take(4)?.try_into()?)),
            7 => Value::Bool(self.take(1)?[0] != 0),
            8 => Value::Str(self.string()?),
            9 => {
                let et = self.u32()?;
                let n = self.u64()? as usize;
                let mut v = Vec::with_capacity(n.min(1 << 20));
                for _ in 0..n {
                    v.push(self.value(et)?);
                }
                Value::Arr(v)
            }
            10 => Value::U64(self.u64()?),
            11 => Value::I64(self.u64()? as i64),
            12 => Value::F64(f64::from_le_bytes(self.take(8)?.try_into()?)),
            _ => bail!("bad gguf value type {ty}"),
        })
    }
}

/// `a-00001-of-00003.gguf` -> all three paths; anything else -> itself.
pub fn split_paths(first: &Path) -> Vec<PathBuf> {
    let s = first.to_string_lossy();
    if let Some(i) = s.rfind("-00001-of-") {
        let rest = &s[i + 10..];
        if let Some(n) = rest.strip_suffix(".gguf").and_then(|n| n.parse::<usize>().ok()) {
            return (1..=n)
                .map(|k| PathBuf::from(format!("{}-{:05}-of-{}", &s[..i], k, rest)))
                .collect();
        }
    }
    vec![first.to_path_buf()]
}

impl Gguf {
    pub fn open(first: &Path) -> Result<Gguf> {
        let paths = split_paths(first);
        let mut g = Gguf { kv: HashMap::new(), tensors: HashMap::new(), order: vec![], paths: paths.clone(), maps: vec![] };
        for (shard, p) in paths.iter().enumerate() {
            let f = File::open(p).with_context(|| format!("open {}", p.display()))?;
            let map = unsafe { Mmap::map(&f)? };
            let mut c = Cursor { b: &map, p: 0 };
            if c.take(4)? != b"GGUF" {
                bail!("{} is not GGUF", p.display());
            }
            let ver = c.u32()?;
            if ver != 3 {
                bail!("gguf version {ver} (want 3)");
            }
            let nt = c.u64()? as usize;
            let nkv = c.u64()? as usize;
            for _ in 0..nkv {
                let k = c.string()?;
                let t = c.u32()?;
                let v = c.value(t)?;
                g.kv.entry(k).or_insert(v);
            }
            let align = g.kv.get("general.alignment").and_then(|v| v.as_u64()).unwrap_or(32);
            let mut infos = Vec::with_capacity(nt);
            for _ in 0..nt {
                let name = c.string()?;
                let nd = c.u32()? as usize;
                let dims = (0..nd).map(|_| c.u64()).collect::<Result<Vec<_>>>()?;
                let ty = GType::from_id(c.u32()?).with_context(|| format!("tensor {name}"))?;
                let off = c.u64()?;
                infos.push((name, dims, ty, off));
            }
            let data = (c.p as u64).div_ceil(align) * align;
            for (name, dims, ty, off) in infos {
                let n: u64 = dims.iter().product();
                let bytes = (ty.row_bytes(dims[0] as usize) as u64) * (n / dims[0]);
                let ti = TensorInfo { name: name.clone(), dims, ty, shard, offset: data + off, bytes };
                if ti.offset + ti.bytes > map.len() as u64 {
                    bail!("tensor {name} runs past the end of {}", p.display());
                }
                g.order.push(name.clone());
                g.tensors.insert(name, ti);
            }
            g.maps.push(map);
        }
        Ok(g)
    }

    pub fn get(&self, name: &str) -> Result<&TensorInfo> {
        self.tensors.get(name).with_context(|| format!("tensor {name} not in the model"))
    }

    pub fn bytes(&self, t: &TensorInfo) -> &[u8] {
        &self.maps[t.shard][t.offset as usize..(t.offset + t.bytes) as usize]
    }

    pub fn u64(&self, key: &str) -> Result<u64> {
        self.kv.get(key).and_then(|v| v.as_u64()).with_context(|| format!("metadata {key}"))
    }
    pub fn f64(&self, key: &str) -> Result<f64> {
        self.kv.get(key).and_then(|v| v.as_f64()).with_context(|| format!("metadata {key}"))
    }
    pub fn arr_u64(&self, key: &str) -> Result<Vec<u64>> {
        match self.kv.get(key) {
            Some(Value::Arr(v)) => v.iter().map(|x| x.as_u64().context("non-integer array item")).collect(),
            _ => bail!("metadata array {key}"),
        }
    }
}
