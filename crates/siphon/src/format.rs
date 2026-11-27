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

/// The attention output projection decode reads under [`WeightFormat::Fp8`]. Its per-row
/// FP8 copy moved the engine furthest from the model of all the FP8 copies, so it stays
/// BF16 unless asked; [`OutputProjection::Mxfp8`] uses per-32 UE8M0 block scales to halve
/// the bytes read while preserving outlier head precision.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum OutputProjection {
    /// The checkpoint's BF16 weights, which prefill reads too.
    #[default]
    Bf16,
    /// An E4M3 copy with one scale per row, as the other FP8 copies.
    Fp8,
    /// An MXFP8 copy: E4M3 values with one UE8M0 power-of-two scale per 32 elements.
    Mxfp8,
}

impl FromStr for OutputProjection {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> anyhow::Result<Self> {
        match s {
            "bf16" => Ok(Self::Bf16),
            "fp8" => Ok(Self::Fp8),
            "mxfp8" => Ok(Self::Mxfp8),
            _ => bail!("unknown output projection format {s:?}; expected bf16, fp8 or mxfp8"),
        }
    }
}

impl fmt::Display for OutputProjection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Bf16 => "bf16",
            Self::Fp8 => "fp8",
            Self::Mxfp8 => "mxfp8",
        })
    }
}

/// Optional override for the target model's LM head weight format (`--lm-head`).
/// When omitted (`None`), the LM head follows [`WeightFormat`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LmHeadFormat {
    /// Keep the BF16 LM head on device.
    Bf16,
    /// Per-row E4M3 FP8 copy (~463 MB).
    Fp8,
    /// Per-16 Four Over Six NVFP4 copy (~261 MB, -44% bytes vs FP8).
    Nvfp4,
}

impl FromStr for LmHeadFormat {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> anyhow::Result<Self> {
        match s {
            "bf16" => Ok(Self::Bf16),
            "fp8" => Ok(Self::Fp8),
            "nvfp4" => Ok(Self::Nvfp4),
            _ => bail!("unknown LM head format {s:?}; expected bf16, fp8 or nvfp4"),
        }
    }
}

impl fmt::Display for LmHeadFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Bf16 => "bf16",
            Self::Fp8 => "fp8",
            Self::Nvfp4 => "nvfp4",
        })
    }
}

/// Element format of the KV caches. Under quantized settings (`Fp8`, `Nvfp4`, `Mixed`),
/// sliding-window caches store E4M3 ([`KvFormat::sliding_format`]); under `Nvfp4All`, both
/// global and sliding-window caches store NVFP4; under `Bf16` they stay BF16.
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
    /// bytes of values per KV head, then their 8 scales per KV head. Global visits use
    /// NVFP4 and sliding-window visits stay E4M3.
    Nvfp4,
    /// NVFP4 for both global visits and sliding-window visits.
    Nvfp4All,
    /// FP8 for the first [`MIXED_FP8_VISITS`] global visits and NVFP4 for the rest. Layer 5
    /// and the first pass of layer 11 are the visits whose rounding to 4 bits costs the
    /// most (see engine.md, "KV cache reduction").
    Mixed,
}

/// Global visits, in visit order, that [`KvFormat::Mixed`] keeps in FP8.
pub const MIXED_FP8_VISITS: usize = 2;

impl KvFormat {
    /// The format of the cache of global visit `global` (its index among the global
    /// visits, in visit order): this format, [`KvFormat::Nvfp4`] for [`KvFormat::Nvfp4All`],
    /// or for [`KvFormat::Mixed`] FP8 or NVFP4.
    #[must_use]
    pub fn cache_format(self, global: usize) -> Self {
        match self {
            Self::Mixed if global < MIXED_FP8_VISITS => Self::Fp8,
            Self::Mixed | Self::Nvfp4All => Self::Nvfp4,
            f => f,
        }
    }

    /// The format of the sliding-window caches under this setting: BF16 under
    /// [`KvFormat::Bf16`], NVFP4 under [`KvFormat::Nvfp4All`], and E4M3 under the other
    /// quantized KV settings (`Fp8`, `Nvfp4`, `Mixed`).
    #[must_use]
    pub const fn sliding_format(self) -> Self {
        match self {
            Self::Bf16 => Self::Bf16,
            Self::Fp8 | Self::Nvfp4 | Self::Mixed => Self::Fp8,
            Self::Nvfp4All => Self::Nvfp4,
        }
    }

    /// Bytes one position takes in a key or a value buffer of `dims` elements of a cache of
    /// this format; for [`KvFormat::Mixed`] and [`KvFormat::Nvfp4All`], of an NVFP4 cache.
    #[must_use]
    pub fn row_bytes(self, dims: usize) -> usize {
        match self {
            Self::Bf16 => 2 * dims,
            Self::Fp8 => dims,
            Self::Nvfp4 | Self::Nvfp4All | Self::Mixed => dims / 2 + dims / 16,
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
            "nvfp4-all" => Ok(Self::Nvfp4All),
            "mixed" => Ok(Self::Mixed),
            _ => bail!(
                "unknown KV cache format {s:?}; expected bf16, fp8, nvfp4, nvfp4-all or mixed"
            ),
        }
    }
}

impl fmt::Display for KvFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Bf16 => "bf16",
            Self::Fp8 => "fp8",
            Self::Nvfp4 => "nvfp4",
            Self::Nvfp4All => "nvfp4-all",
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

/// Storage format of NVFP4 expert block scales (`--nvfp4-scales`).
///
/// The checkpoint stores one 8-bit E4M3 scale per 16-element weight block (`E4m3`, 1.19 GiB
/// across layers 0–19). Within any weight row those E4M3 codes span a median of 13–14 codes,
/// so `Offset4` stores one 8-bit E4M3 base (`max`) per row plus a 4-bit offset (`0..=15`) per
/// block, re-rounding the ~0.3–0.4% of tiny blocks below `base - 15` at load time. That halves
/// scale VRAM (`-0.58 GiB`, ~`+68K` NVFP4 KV positions) and scale DRAM traffic.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Nvfp4ScaleFormat {
    /// Raw checkpoint 8-bit E4M3 scales per 16-element block.
    #[default]
    E4m3,
    /// Per-row 8-bit E4M3 base plus packed 4-bit offset (`base - off`) per 16-element block.
    Offset4,
}

impl FromStr for Nvfp4ScaleFormat {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> anyhow::Result<Self> {
        match s {
            "8bit" | "e4m3" => Ok(Self::E4m3),
            "4bit" | "offset4" => Ok(Self::Offset4),
            _ => bail!("unknown NVFP4 scale format {s:?}; expected 8bit or 4bit"),
        }
    }
}

impl fmt::Display for Nvfp4ScaleFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::E4m3 => "8bit",
            Self::Offset4 => "4bit",
        })
    }
}

/// Storage format of the checkpoint's MXFP8 expert layers (layers 20–23, `--mxfp8-experts`).
///
/// `Mxfp6` re-quantizes each 32-element MXFP8 block to OCP MXFP6 E2M3 at load time using
/// MSE-optimal UE8M0 block scales, saving 25% of weight bytes (`-0.95 GiB`, ~`+111K` NVFP4 KV
/// positions) while mapping bit-exactly back into E4M3 for single-pass FP8 MMA.
/// `Nvfp4` re-quantizes each 16-element half-block to Four Over Six NVFP4 at load time using
/// the same kernels and optional `--nvfp4-scales 4bit` compression as layers 0–19, saving
/// `1.78 GiB` (`1.90 GiB` with 4-bit scales) over MXFP8 and speeding up decode, verify, and prefill.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Mxfp8ExpertFormat {
    /// Raw checkpoint MXFP8 (E4M3 values + UE8M0 per-32 scales).
    #[default]
    Mxfp8,
    /// Load-time MXFP6 E2M3 (6-bit values in 2 planes `lo4` + `hi2` + UE8M0 per-32 scales).
    Mxfp6,
    /// Load-time Four Over Six NVFP4 (E2M1 values + E4M3 or 4-bit offset per-16 scales).
    Nvfp4,
}

impl FromStr for Mxfp8ExpertFormat {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> anyhow::Result<Self> {
        match s {
            "mxfp8" => Ok(Self::Mxfp8),
            "mxfp6" => Ok(Self::Mxfp6),
            "nvfp4" => Ok(Self::Nvfp4),
            _ => bail!("unknown MXFP8 expert format {s:?}; expected mxfp8, mxfp6 or nvfp4"),
        }
    }
}

impl fmt::Display for Mxfp8ExpertFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Mxfp8 => "mxfp8",
            Self::Mxfp6 => "mxfp6",
            Self::Nvfp4 => "nvfp4",
        })
    }
}
