use std::{env, path::PathBuf, process::Command};

fn main() {
    println!("cargo:rerun-if-changed=src/kernels");
    println!("cargo:rerun-if-env-changed=NVCC");
    println!("cargo:rerun-if-env-changed=CUDA_ARCHS");
    println!("cargo:rerun-if-env-changed=DSX_KDEFS");
    println!("cargo:rerun-if-env-changed=DSX_KS");
    println!("cargo:rerun-if-env-changed=DSX_ROWT");
    println!("cargo:rerun-if-env-changed=DSX_KERNEL_IMAGE");
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap()).join("dsx.fatbin");
    if env::var_os("CARGO_FEATURE_CUDA").is_none() {
        std::fs::write(&out, []).unwrap();
        std::fs::write(out.with_file_name("nanbeige-exl3.fatbin"), []).unwrap();
        return;
    }
    // Diagnostic builds can retain an archived deployed image while testing the
    // same host wrapper. Kernel-dependent experimental flags must remain off.
    if let Some(image) = env::var_os("DSX_KERNEL_IMAGE") {
        let image = PathBuf::from(image);
        println!("cargo:rerun-if-changed={}", image.display());
        let bytes = std::fs::read(&image).expect("read explicit DSX_KERNEL_IMAGE");
        assert!(!bytes.is_empty(), "DSX_KERNEL_IMAGE must not be empty");
        std::fs::write(&out, bytes).expect("copy explicit kernel image");
        println!("cargo:warning=using explicit diagnostic kernel image {}", image.display());
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
    let mut cmd = Command::new(&nvcc);
    cmd.args(["--fatbin", "-std=c++17", "-O3", "--prec-div=true", "--prec-sqrt=true", "-lineinfo"]);
    for a in archs.split(',') {
        cmd.arg(format!("-gencode=arch=compute_{a},code=sm_{a}"));
    }
    // DSX_KDEFS="-DABL=5": extra nvcc defines for experiments (expert_fast.cuh ABL ablations); never for real runs
    if let Ok(r) = env::var("DSX_ROWT") {
        cmd.arg(format!("-DROWT={r}"));
    }
    if let Ok(k) = env::var("DSX_KS") {
        cmd.arg(format!("-DKS={k}"));
    }
    if let Ok(d) = env::var("DSX_KDEFS") {
        cmd.args(d.split_whitespace());
    }
    cmd.args(["src/kernels/dsx.cu", "-o"]).arg(&out);
    let st = cmd.status().expect("nvcc is required for the cuda feature");
    assert!(st.success(), "kernel compilation failed");
    let mut exl3 = Command::new(&nvcc);
    exl3.args(["--fatbin", "-std=c++17", "-O3", "--prec-div=true", "--prec-sqrt=true", "-lineinfo"]);
    for a in archs.split(',') { exl3.arg(format!("-gencode=arch=compute_{a},code=sm_{a}")); }
    exl3.args(["src/kernels/nanbeige-exl3.cu", "-o"]).arg(out.with_file_name("nanbeige-exl3.fatbin"));
    assert!(exl3.status().expect("compile standalone EXL3 kernels").success(), "EXL3 kernel compilation failed");

}
