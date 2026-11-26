use std::{fmt, str::FromStr};

use anyhow::bail;

/// The format of the large BF16 matrices decode reads: the attention input and output
/// projections and the LM head.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum WeightFormat {
    /// The checkpoint's BF16 weights.
    #[default]
    Bf16,
    /// E4M3 copies with one F32 scale per row, made at load time, of the attention input
    /// projections, the message gates and the LM head; the output projection follows
    /// [`OutputProjection`]. Decode reads about half the bytes; prefill keeps the BF16
    /// weights.
    Fp8,
}

impl FromStr for WeightFormat {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> anyhow::Result<Self> {
        match s {
            "bf16" => Ok(Self::Bf16),
            "fp8" => Ok(Self::Fp8),
            _ => bail!("unknown weight format {s:?}; expected bf16 or fp8"),
        }
    }
}

impl fmt::Display for WeightFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Bf16 => "bf16",
            Self::Fp8 => "fp8",
        })
    }
}

/// The attention output projection decode reads under [`WeightFormat::Fp8`]. Its FP8 copy
/// moved the engine furthest from the model of all the FP8 copies, so it stays BF16 unless
/// asked: on an RTX 5090 the FP8 copy saves 0.16 ms of a 2.93 ms step.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum OutputProjection {
    /// The checkpoint's BF16 weights, which prefill reads too.
    #[default]
    Bf16,
    /// An E4M3 copy with one scale per row, as the other FP8 copies.
    Fp8,
}

impl FromStr for OutputProjection {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> anyhow::Result<Self> {
        match s {
            "bf16" => Ok(Self::Bf16),
            "fp8" => Ok(Self::Fp8),
            _ => bail!("unknown output projection format {s:?}; expected bf16 or fp8"),
        }
    }
}

impl fmt::Display for OutputProjection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Bf16 => "bf16",
            Self::Fp8 => "fp8",
        })
    }
}

/// Element format of the global-layer KV caches. Sliding caches are always BF16.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum KvFormat {
    /// BF16, the projections' own values.
    #[default]
    Bf16,
    /// E4M3 without a scale: each BF16 value rounded to the nearest E4M3 value, which halves
    /// the bytes attention reads. Keys are L2-clipped to norm `sqrt(512)` and values are
    /// RMS-normed, so every element lies far inside E4M3's range of +-448.
    Fp8,
}

impl KvFormat {
    /// Bytes of one stored key or value element.
    #[must_use]
    pub fn element_bytes(self) -> usize {
        match self {
            Self::Bf16 => 2,
            Self::Fp8 => 1,
        }
    }
}

impl FromStr for KvFormat {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> anyhow::Result<Self> {
        match s {
            "bf16" => Ok(Self::Bf16),
            "fp8" => Ok(Self::Fp8),
            _ => bail!("unknown KV cache format {s:?}; expected bf16 or fp8"),
        }
    }
}

impl fmt::Display for KvFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Bf16 => "bf16",
            Self::Fp8 => "fp8",
        })
    }
}

/// The format of the vision tower's linear layers. Only BF16: FP8 weights with MXFP8
/// activations changed the model's answers (see docs/vision.md).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum VisionFormat {
    /// The checkpoint's BF16 weights.
    #[default]
    Bf16,
}

impl FromStr for VisionFormat {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> anyhow::Result<Self> {
        match s {
            "bf16" => Ok(Self::Bf16),
            _ => bail!("unknown vision format {s:?}; expected bf16"),
        }
    }
}

impl fmt::Display for VisionFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Bf16 => "bf16",
        })
    }
}
