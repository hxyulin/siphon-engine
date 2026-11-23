//! CUDA kernels for Inkling 2 Mini, precompiled for `sm_120a` and loaded through cudarc.
//!
//! Each `ops` module owns one `kernels/<name>.cu` file. The crate is empty on targets
//! other than Linux.

#![cfg(target_os = "linux")]

mod gpu;
pub mod ops;

pub use gpu::Gpu;
