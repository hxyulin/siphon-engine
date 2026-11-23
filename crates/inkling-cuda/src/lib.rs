//! CUDA kernels for Inkling 2 Mini, precompiled for `sm_120a` and loaded through cudarc.
//!
//! Each `ops` module owns one `kernels/<name>.cu` file. The crate is empty on targets
//! other than Linux.

#![cfg(target_os = "linux")]

pub mod draft;
mod gpu;
pub mod model;
pub mod ops;
pub mod weights;

pub use gpu::{DeviceIn, DeviceOut, Gpu};
