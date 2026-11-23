//! Model description and checkpoint access for Inkling 2 Mini.

mod checkpoint;
mod config;

pub use checkpoint::{Checkpoint, TensorEntry};
pub use config::{LoopConfig, TextConfig};
