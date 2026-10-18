use std::{env, path::PathBuf, process::Command};

fn main() {
    println!("cargo:rerun-if-changed=src/kernels");
    println!("cargo:rerun-if-env-changed=NVCC");
    println!("cargo:rerun-if-env-changed=CUDA_ARCHS");
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap()).join("ds41.fatbin");
    if env::var_os("CARGO_FEATURE_CUDA").is_none() {
        std::fs::write(&out, []).unwrap();
        return;
    }
    let nvcc = env::var("NVCC").unwrap_or_else(|_| {
        for p in ["/usr/local/cuda-13/bin/nvcc", "/opt/cuda/bin/nvcc", "/usr/local/cuda/bin/nvcc"] {
            if std::path::Path::new(p).exists() {
                return p.to_string();
            }
        }
        "nvcc".to_string()
    });
    // 86 = RTX 3090 (the target); 120 = the RTX 5090 dev box
    let archs = env::var("CUDA_ARCHS").unwrap_or_else(|_| "86,120".into());
    let mut cmd = Command::new(nvcc);
    cmd.args(["--fatbin", "-std=c++17", "-O3", "--prec-div=true", "--prec-sqrt=true", "-lineinfo"]);
    for a in archs.split(',') {
        cmd.arg(format!("-gencode=arch=compute_{a},code=sm_{a}"));
    }
    cmd.args(["src/kernels/ds41.cu", "-o"]).arg(&out);
    let st = cmd.status().expect("nvcc is required for the cuda feature");
    assert!(st.success(), "kernel compilation failed");
}
