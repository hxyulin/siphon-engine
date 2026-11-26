//! CUDA kernels for Inkling 2 Mini, precompiled for `sm_120a` (or `sm_89` with
//! `SIPHON_CUDA_ARCH=sm_89`) and loaded through cudarc.
//!
//! Each `ops` module owns one `kernels/<name>.cu` file. The crate is empty on targets
//! other than Linux.

#![cfg(target_os = "linux")]

pub mod draft;
mod gpu;
pub mod kv_options;
pub mod model;
pub mod ops;
pub mod spec;
pub mod vision;
pub mod vmm;
pub mod weights;

pub use gpu::{BLOCK_SCALED_MMA, DeviceIn, DeviceOut, Gpu, MemoryUse};

/// A hash of the kernels this build compiled, which changes whenever any kernel's code
/// does. The prefix cache's files on disk record it.
pub const KERNELS_HASH: &str = env!("SIPHON_KERNELS_HASH");
