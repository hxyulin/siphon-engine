//! Launchers for the kernels in `kernels/`, one module per `.cu` file.

pub mod attention;
pub mod conv;
pub mod draft;
pub mod elementwise;
pub mod experts;
pub mod experts_rows;
pub mod linear;
pub mod linear_fp8;
pub mod linear_rows;
pub mod misc;
pub mod norm;
pub mod routing;
pub mod spec;

use std::{fmt, str::FromStr};

/// How prefill multiplies its attention and dense products: in BF16 (F16 for attention's
/// weighted values), or on the block-scaled FP8 tensor cores with the activations quantized
/// to E4M3. FP8 lowers precision on purpose; decode never uses it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Products {
    #[default]
    Bf16,
    Fp8,
}

impl FromStr for Products {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> anyhow::Result<Self> {
        match s {
            "bf16" => Ok(Self::Bf16),
            "fp8" => Ok(Self::Fp8),
            _ => anyhow::bail!("unknown product format {s:?}; expected bf16 or fp8"),
        }
    }
}

impl fmt::Display for Products {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Bf16 => "bf16",
            Self::Fp8 => "fp8",
        })
    }
}
