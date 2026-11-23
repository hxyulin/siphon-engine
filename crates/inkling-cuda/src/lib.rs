//! CUDA kernels for Inkling 2 Mini, precompiled for `sm_120a` and loaded through cudarc.
//!
//! The crate is empty on targets other than Linux.

#![cfg(target_os = "linux")]

mod gpu;

pub use gpu::Gpu;
