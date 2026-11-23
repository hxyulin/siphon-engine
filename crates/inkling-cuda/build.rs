//! Compiles every `kernels/*.cu` file to a cubin in `OUT_DIR`.
//!
//! Only runs when targeting Linux. Finds `nvcc` through `NVCC`, then `CUDA_HOME`,
//! then `PATH`. `INKLING_CUDA_ARCH` overrides the default `sm_120a` target.

use std::{env, fs, path::PathBuf, process::Command};

fn main() {
    println!("cargo::rerun-if-changed=kernels");
    for var in ["NVCC", "CUDA_HOME", "INKLING_CUDA_ARCH"] {
        println!("cargo::rerun-if-env-changed={var}");
    }
    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("linux") {
        return;
    }

    let nvcc = match (env::var_os("NVCC"), env::var_os("CUDA_HOME")) {
        (Some(nvcc), _) => PathBuf::from(nvcc),
        (None, Some(home)) => PathBuf::from(home).join("bin/nvcc"),
        (None, None) => PathBuf::from("nvcc"),
    };
    let arch = env::var("INKLING_CUDA_ARCH").unwrap_or_else(|_| "sm_120a".to_owned());
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("cargo sets OUT_DIR"));

    let mut sources: Vec<PathBuf> = fs::read_dir("kernels")
        .expect("kernels directory exists")
        .map(|entry| entry.expect("readable kernels entry").path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "cu"))
        .collect();
    sources.sort();

    for source in sources {
        let stem = source
            .file_stem()
            .expect("kernel file has a name")
            .to_owned();
        let cubin = out_dir.join(&stem).with_extension("cubin");
        let status = Command::new(&nvcc)
            .args(["-cubin", "-O3", "-std=c++20", "-lineinfo"])
            .arg(format!("-arch={arch}"))
            .args(["-I", "kernels"])
            .arg("-o")
            .arg(&cubin)
            .arg(&source)
            .status()
            .unwrap_or_else(|err| {
                panic!("running {}: {err}; set NVCC or CUDA_HOME", nvcc.display())
            });
        assert!(status.success(), "nvcc failed on {}", source.display());
    }
}
