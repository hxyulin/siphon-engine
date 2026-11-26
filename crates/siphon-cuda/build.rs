//! Compiles every `kernels/*.cu` file to a cubin in `OUT_DIR`, and hashes the cubins into
//! `SIPHON_KERNELS_HASH`, which the prefix cache's saved files record.
//!
//! Only runs when targeting Linux. Finds `nvcc` through `NVCC`, then `CUDA_HOME`,
//! then `PATH`. `SIPHON_CUDA_ARCH` overrides the default `sm_120a` target: `sm_89` builds
//! for the RTX 4090 and sets the `sm89` cfg, which turns off what only `sm_120a` can run.

use std::{env, fs, path::PathBuf, process::Command};

fn main() {
    println!("cargo::rerun-if-changed=kernels");
    for var in ["NVCC", "CUDA_HOME", "SIPHON_CUDA_ARCH", "SIPHON_FAST_MATH"] {
        println!("cargo::rerun-if-env-changed={var}");
    }
    let arch = env::var("SIPHON_CUDA_ARCH").unwrap_or_else(|_| "sm_120a".to_owned());
    let fast_math = env::var("SIPHON_FAST_MATH").as_deref() == Ok("1");
    println!("cargo::rustc-env=SIPHON_CUDA_ARCH={arch}");
    println!("cargo::rustc-check-cfg=cfg(sm89)");
    println!("cargo::rustc-check-cfg=cfg(fast_math)");
    if arch == "sm_89" {
        println!("cargo::rustc-cfg=sm89");
    }
    if fast_math {
        println!("cargo::rustc-cfg=fast_math");
    }
    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("linux") {
        println!("cargo::rustc-env=SIPHON_KERNELS_HASH=none");
        return;
    }

    let nvcc = match (env::var_os("NVCC"), env::var_os("CUDA_HOME")) {
        (Some(nvcc), _) => PathBuf::from(nvcc),
        (None, Some(home)) => PathBuf::from(home).join("bin/nvcc"),
        (None, None) => PathBuf::from("nvcc"),
    };
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("cargo sets OUT_DIR"));

    let mut sources: Vec<PathBuf> = fs::read_dir("kernels")
        .expect("kernels directory exists")
        .map(|entry| entry.expect("readable kernels entry").path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "cu"))
        .collect();
    sources.sort();

    // FNV-1a over every cubin's name and bytes.
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    let mut fnv = |bytes: &[u8]| {
        for &b in bytes {
            hash = (hash ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3);
        }
    };
    for source in sources {
        let stem = source
            .file_stem()
            .expect("kernel file has a name")
            .to_owned();
        let cubin = out_dir.join(&stem).with_extension("cubin");
        let mut cmd = Command::new(&nvcc);
        cmd.args(["-cubin", "-O3", "-std=c++20", "-lineinfo"])
            .arg(format!("-arch={arch}"))
            .args(["-I", "kernels"]);
        if fast_math {
            cmd.args(["--use_fast_math", "-DSIPHON_FAST_MATH=1"]);
        }
        let status = cmd
            .arg("-o")
            .arg(&cubin)
            .arg(&source)
            .status()
            .unwrap_or_else(|err| {
                panic!("running {}: {err}; set NVCC or CUDA_HOME", nvcc.display())
            });
        assert!(status.success(), "nvcc failed on {}", source.display());
        fnv(stem.as_encoded_bytes());
        fnv(&fs::read(&cubin).expect("nvcc wrote the cubin"));
    }
    println!("cargo::rustc-env=SIPHON_KERNELS_HASH={hash:016x}");
}
