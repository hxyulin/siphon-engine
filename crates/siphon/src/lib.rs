//! Model description and checkpoint access for Inkling 2 Mini.

mod checkpoint;
mod config;
mod format;
pub mod memory;
pub mod race;

pub use checkpoint::{Checkpoint, TensorEntry};
pub use config::{LoopConfig, TextConfig};
pub use format::{KvFormat, WeightFormat};
