//! Minimal HIP runtime bindings: memory, module/kernels, streams, events, graphs. Everything this runner needs and
//! nothing else; errors become anyhow errors with the call name.
#![allow(non_camel_case_types, clippy::missing_safety_doc)]

use anyhow::{Result, bail};
use std::ffi::{CStr, CString, c_char, c_int, c_uint, c_void};

pub type hipError_t = c_int;
pub type hipStream_t = *mut c_void;
pub type hipEvent_t = *mut c_void;
pub type hipModule_t = *mut c_void;
pub type hipFunction_t = *mut c_void;
pub type hipGraph_t = *mut c_void;
pub type hipGraphExec_t = *mut c_void;

#[link(name = "amdhip64")]
unsafe extern "C" {
    fn hipInit(flags: c_uint) -> hipError_t;
    fn hipSetDevice(d: c_int) -> hipError_t;
    fn hipGetErrorString(e: hipError_t) -> *const c_char;
    fn hipMalloc(p: *mut *mut c_void, n: usize) -> hipError_t;
    fn hipFree(p: *mut c_void) -> hipError_t;
    fn hipHostMalloc(p: *mut *mut c_void, n: usize, flags: c_uint) -> hipError_t;
    fn hipMemcpy(dst: *mut c_void, src: *const c_void, n: usize, kind: c_int) -> hipError_t;
    fn hipMemcpyAsync(dst: *mut c_void, src: *const c_void, n: usize, kind: c_int, s: hipStream_t) -> hipError_t;
    fn hipMemsetAsync(dst: *mut c_void, v: c_int, n: usize, s: hipStream_t) -> hipError_t;
    fn hipMemGetInfo(free: *mut usize, total: *mut usize) -> hipError_t;
    fn hipDeviceSynchronize() -> hipError_t;
    fn hipStreamCreateWithFlags(s: *mut hipStream_t, flags: c_uint) -> hipError_t;
    fn hipStreamSynchronize(s: hipStream_t) -> hipError_t;
    fn hipModuleLoadData(m: *mut hipModule_t, image: *const c_void) -> hipError_t;
    fn hipModuleGetFunction(f: *mut hipFunction_t, m: hipModule_t, name: *const c_char) -> hipError_t;
    fn hipModuleLaunchKernel(
        f: hipFunction_t, gx: c_uint, gy: c_uint, gz: c_uint, bx: c_uint, by: c_uint, bz: c_uint, shmem: c_uint,
        s: hipStream_t, params: *mut *mut c_void, extra: *mut *mut c_void,
    ) -> hipError_t;
    fn hipFuncSetAttribute(f: *const c_void, attr: c_int, v: c_int) -> hipError_t;
    fn hipEventCreate(e: *mut hipEvent_t) -> hipError_t;
    fn hipEventRecord(e: hipEvent_t, s: hipStream_t) -> hipError_t;
    fn hipEventSynchronize(e: hipEvent_t) -> hipError_t;
    fn hipEventElapsedTime(ms: *mut f32, a: hipEvent_t, b: hipEvent_t) -> hipError_t;
    fn hipStreamBeginCapture(s: hipStream_t, mode: c_int) -> hipError_t;
    fn hipStreamEndCapture(s: hipStream_t, g: *mut hipGraph_t) -> hipError_t;
    fn hipGraphInstantiate(e: *mut hipGraphExec_t, g: hipGraph_t, node: *mut c_void, log: *mut c_char, n: usize) -> hipError_t;
    fn hipGraphLaunch(e: hipGraphExec_t, s: hipStream_t) -> hipError_t;
    fn hipGraphDestroy(g: hipGraph_t) -> hipError_t;
    fn hipGraphExecDestroy(e: hipGraphExec_t) -> hipError_t;
}

const H2D: c_int = 1;
const D2H: c_int = 2;
const D2D: c_int = 3;

fn ck(e: hipError_t, what: &str) -> Result<()> {
    if e != 0 {
        let s = unsafe { CStr::from_ptr(hipGetErrorString(e)) }.to_string_lossy();
        bail!("{what}: {s} ({e})");
    }
    Ok(())
}

/// One kernel argument, laid out as the kernel expects (8-byte pointers, 4-byte ints/floats).
#[derive(Clone, Debug)]
pub enum Arg {
    P(u64),
    I(i32),
    F(f32),
    /// a struct passed by value (its bytes)
    B(Vec<u8>),
}

pub struct Gpu {
    pub stream: hipStream_t,
    module: hipModule_t,
    funcs: std::sync::Mutex<std::collections::HashMap<String, hipFunction_t>>,
    pub allocated: std::sync::atomic::AtomicUsize,
}

unsafe impl Send for Gpu {}
unsafe impl Sync for Gpu {}

/// A device allocation (freed on drop).
pub struct Buf {
    pub ptr: u64,
    pub bytes: usize,
}

impl Drop for Buf {
    fn drop(&mut self) {
        if self.ptr != 0 {
            unsafe { hipFree(self.ptr as *mut c_void) };
        }
    }
}

impl Buf {
    pub fn at(&self, off: usize) -> u64 {
        self.ptr + off as u64
    }
}

pub struct Graph {
    exec: hipGraphExec_t,
    graph: hipGraph_t,
}
unsafe impl Send for Graph {}

impl Drop for Graph {
    fn drop(&mut self) {
        unsafe {
            hipGraphExecDestroy(self.exec);
            hipGraphDestroy(self.graph);
        }
    }
}

impl Gpu {
    pub fn new(image: &[u8]) -> Result<Gpu> {
        unsafe {
            ck(hipInit(0), "hipInit")?;
            ck(hipSetDevice(0), "hipSetDevice")?;
            let mut s = std::ptr::null_mut();
            ck(hipStreamCreateWithFlags(&mut s, 1), "hipStreamCreate")?;
            let mut m = std::ptr::null_mut();
            ck(hipModuleLoadData(&mut m, image.as_ptr() as *const c_void), "hipModuleLoadData")?;
            Ok(Gpu { stream: s, module: m, funcs: Default::default(), allocated: Default::default() })
        }
    }

    pub fn func(&self, name: &str) -> Result<hipFunction_t> {
        let mut map = self.funcs.lock().unwrap();
        if let Some(&f) = map.get(name) {
            return Ok(f);
        }
        let mut f = std::ptr::null_mut();
        let c = CString::new(name)?;
        ck(unsafe { hipModuleGetFunction(&mut f, self.module, c.as_ptr()) }, name)?;
        map.insert(name.to_string(), f);
        Ok(f)
    }

    pub fn launch(&self, name: &str, grid: (u32, u32, u32), block: u32, shmem: u32, args: &[Arg]) -> Result<()> {
        let f = self.func(name)?;
        // pack into one 8-byte-slot per argument; HIP takes pointers to each value
        let mut store: Vec<u64> = vec![0; args.len()];
        let mut blobs: Vec<Vec<u64>> = vec![];
        let mut ptrs: Vec<*mut c_void> = Vec::with_capacity(args.len());
        for (i, a) in args.iter().enumerate() {
            store[i] = match a {
                Arg::P(p) => *p,
                Arg::I(v) => *v as u32 as u64,
                Arg::F(v) => v.to_bits() as u64,
                Arg::B(b) => {
                    let mut v = vec![0u64; b.len().div_ceil(8)];
                    unsafe { std::ptr::copy_nonoverlapping(b.as_ptr(), v.as_mut_ptr() as *mut u8, b.len()) };
                    blobs.push(v);
                    0
                }
            };
        }
        let mut bi = 0;
        for (i, a) in args.iter().enumerate() {
            if let Arg::B(_) = a {
                ptrs.push(blobs[bi].as_mut_ptr() as *mut c_void);
                bi += 1;
            } else {
                ptrs.push(&mut store[i] as *mut u64 as *mut c_void);
            }
        }
        if shmem > 65536 {
            bail!("{name}: {shmem} bytes of LDS");
        }
        ck(
            unsafe {
                hipModuleLaunchKernel(f, grid.0, grid.1, grid.2, block, 1, 1, shmem, self.stream, ptrs.as_mut_ptr(), std::ptr::null_mut())
            },
            name,
        )
    }

    pub fn alloc(&self, bytes: usize) -> Result<Buf> {
        let mut p = std::ptr::null_mut();
        let n = bytes.max(256).div_ceil(256) * 256;
        ck(unsafe { hipMalloc(&mut p, n) }, &format!("hipMalloc {bytes}"))?;
        self.allocated.fetch_add(n, std::sync::atomic::Ordering::Relaxed);
        Ok(Buf { ptr: p as u64, bytes: n })
    }

    pub fn alloc_zeroed(&self, bytes: usize) -> Result<Buf> {
        let b = self.alloc(bytes)?;
        self.memset(b.ptr, 0, b.bytes)?;
        self.sync()?;
        Ok(b)
    }

    pub fn upload<T: Copy>(&self, v: &[T]) -> Result<Buf> {
        let bytes = std::mem::size_of_val(v);
        let b = self.alloc(bytes)?;
        self.htod(b.ptr, v)?;
        Ok(b)
    }

    pub fn htod<T: Copy>(&self, dst: u64, v: &[T]) -> Result<()> {
        ck(unsafe { hipMemcpy(dst as *mut c_void, v.as_ptr() as *const c_void, std::mem::size_of_val(v), H2D) }, "hipMemcpy htod")
    }

    pub fn htod_async(&self, dst: u64, src: *const u8, n: usize) -> Result<()> {
        ck(unsafe { hipMemcpyAsync(dst as *mut c_void, src as *const c_void, n, H2D, self.stream) }, "hipMemcpyAsync htod")
    }

    pub fn dtoh<T: Copy + Default>(&self, src: u64, n: usize) -> Result<Vec<T>> {
        self.sync()?;
        let mut v = vec![T::default(); n];
        ck(unsafe { hipMemcpy(v.as_mut_ptr() as *mut c_void, src as *const c_void, n * std::mem::size_of::<T>(), D2H) }, "hipMemcpy dtoh")?;
        Ok(v)
    }

    pub fn dtod(&self, dst: u64, src: u64, n: usize) -> Result<()> {
        ck(unsafe { hipMemcpyAsync(dst as *mut c_void, src as *const c_void, n, D2D, self.stream) }, "hipMemcpy dtod")
    }

    pub fn memset(&self, dst: u64, v: u8, n: usize) -> Result<()> {
        ck(unsafe { hipMemsetAsync(dst as *mut c_void, v as c_int, n, self.stream) }, "hipMemset")
    }

    pub fn sync(&self) -> Result<()> {
        ck(unsafe { hipStreamSynchronize(self.stream) }, "hipStreamSynchronize")
    }

    pub fn device_sync(&self) -> Result<()> {
        ck(unsafe { hipDeviceSynchronize() }, "hipDeviceSynchronize")
    }

    pub fn mem_info(&self) -> Result<(usize, usize)> {
        let (mut f, mut t) = (0, 0);
        ck(unsafe { hipMemGetInfo(&mut f, &mut t) }, "hipMemGetInfo")?;
        Ok((f, t))
    }

    /// Pinned host memory (freed only at exit).
    pub fn host_alloc(&self, bytes: usize) -> Result<*mut u8> {
        let mut p = std::ptr::null_mut();
        ck(unsafe { hipHostMalloc(&mut p, bytes, 0) }, "hipHostMalloc")?;
        Ok(p as *mut u8)
    }

    pub fn begin_capture(&self) -> Result<()> {
        ck(unsafe { hipStreamBeginCapture(self.stream, 0) }, "hipStreamBeginCapture")
    }

    pub fn end_capture(&self) -> Result<Graph> {
        let mut g = std::ptr::null_mut();
        ck(unsafe { hipStreamEndCapture(self.stream, &mut g) }, "hipStreamEndCapture")?;
        let mut e = std::ptr::null_mut();
        ck(unsafe { hipGraphInstantiate(&mut e, g, std::ptr::null_mut(), std::ptr::null_mut(), 0) }, "hipGraphInstantiate")?;
        Ok(Graph { exec: e, graph: g })
    }

    pub fn graph_launch(&self, g: &Graph) -> Result<()> {
        ck(unsafe { hipGraphLaunch(g.exec, self.stream) }, "hipGraphLaunch")
    }

    /// Wall time of `f`'s GPU work on the stream, in microseconds.
    pub fn time<F: FnMut() -> Result<()>>(&self, mut f: F) -> Result<f32> {
        unsafe {
            let (mut a, mut b) = (std::ptr::null_mut(), std::ptr::null_mut());
            ck(hipEventCreate(&mut a), "hipEventCreate")?;
            ck(hipEventCreate(&mut b), "hipEventCreate")?;
            ck(hipEventRecord(a, self.stream), "hipEventRecord")?;
            f()?;
            ck(hipEventRecord(b, self.stream), "hipEventRecord")?;
            ck(hipEventSynchronize(b), "hipEventSynchronize")?;
            let mut ms = 0f32;
            ck(hipEventElapsedTime(&mut ms, a, b), "hipEventElapsedTime")?;
            Ok(ms * 1000.0)
        }
    }
}

#[allow(dead_code)]
fn _unused() {
    let _ = hipFuncSetAttribute;
}
