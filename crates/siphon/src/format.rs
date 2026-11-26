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

/// Element format of the global-layer KV caches. Under quantized settings (`Fp8`, `Nvfp4`,
/// `Mixed`), sliding-window caches store E4M3 ([`KvFormat::sliding_format`]); under `Bf16`
/// they stay BF16.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum KvFormat {
    /// BF16, the projections' own values.
    #[default]
    Bf16,
    /// E4M3 without a scale: each BF16 value rounded to the nearest E4M3 value, which halves
    /// the bytes attention reads. Keys are L2-clipped to norm `sqrt(512)` and values are
    /// RMS-normed, so every element lies far inside E4M3's range of +-448.
    Fp8,
    /// NVFP4: E2M1 values with one E4M3 scale per 16 elements, the scale `amax / 6`
    /// rounded to nearest, 4.5 bits per element. Each slot of a cache holds a head's 64
    /// bytes of values per KV head, then their 8 scales per KV head.
    Nvfp4,
    /// FP8 for the first [`MIXED_FP8_VISITS`] global visits and NVFP4 for the rest. Layer 5
    /// and the first pass of layer 11 are the visits whose rounding to 4 bits costs the
    /// most (see engine.md, "KV cache reduction").
    Mixed,
}

/// Global visits, in visit order, that [`KvFormat::Mixed`] keeps in FP8.
pub const MIXED_FP8_VISITS: usize = 2;

impl KvFormat {
    /// The format of the cache of global visit `global` (its index among the global
    /// visits, in visit order): this format, or for [`KvFormat::Mixed`] FP8 or NVFP4.
    #[must_use]
    pub fn cache_format(self, global: usize) -> Self {
        match self {
            Self::Mixed if global < MIXED_FP8_VISITS => Self::Fp8,
            Self::Mixed => Self::Nvfp4,
            f => f,
        }
    }

    /// The format of the sliding-window caches under this setting: BF16 under
    /// [`KvFormat::Bf16`], E4M3 under every quantized KV setting (`Fp8`, `Nvfp4`, `Mixed`).
    #[must_use]
    pub const fn sliding_format(self) -> Self {
        match self {
            Self::Bf16 => Self::Bf16,
            Self::Fp8 | Self::Nvfp4 | Self::Mixed => Self::Fp8,
        }
    }

    /// Bytes one position takes in a key or a value buffer of `dims` elements of a cache of
    /// this format; for [`KvFormat::Mixed`], of an NVFP4 cache.
    #[must_use]
    pub fn row_bytes(self, dims: usize) -> usize {
        match self {
            Self::Bf16 => 2 * dims,
            Self::Fp8 => dims,
            Self::Nvfp4 | Self::Mixed => dims / 2 + dims / 16,
        }
    }
}

impl FromStr for KvFormat {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> anyhow::Result<Self> {
        match s {
            "bf16" => Ok(Self::Bf16),
            "fp8" => Ok(Self::Fp8),
            "nvfp4" => Ok(Self::Nvfp4),
            "mixed" => Ok(Self::Mixed),
            _ => bail!("unknown KV cache format {s:?}; expected bf16, fp8, nvfp4 or mixed"),
        }
    }
}

impl fmt::Display for KvFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Bf16 => "bf16",
            Self::Fp8 => "fp8",
            Self::Nvfp4 => "nvfp4",
            Self::Mixed => "mixed",
        })
    }
}

/// A 4-bit rounding the global caches' keys and values go through before they are stored,
/// to measure what a 4-bit cache would cost in quality. The storage stays in its
/// [`KvFormat`], so memory and speed do not change.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum KvQuant {
    /// The values as computed.
    #[default]
    None,
    /// NVFP4: E2M1 values with one E4M3 scale per 16 elements, the scale `amax / 6`
    /// rounded to nearest.
    Nvfp4,
    /// llama.cpp's `q4_0`: 4-bit integers with one F16 scale per 32 elements, the scale the
    /// signed element of largest magnitude over -8.
    Q4_0,
}

impl FromStr for KvQuant {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> anyhow::Result<Self> {
        match s {
            "none" => Ok(Self::None),
            "nvfp4" => Ok(Self::Nvfp4),
            "q4_0" => Ok(Self::Q4_0),
            _ => bail!("unknown KV rounding {s:?}; expected none, nvfp4 or q4_0"),
        }
    }
}

impl fmt::Display for KvQuant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::None => "none",
            Self::Nvfp4 => "nvfp4",
            Self::Q4_0 => "q4_0",
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
