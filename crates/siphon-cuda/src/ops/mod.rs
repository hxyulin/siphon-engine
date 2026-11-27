//! Launchers for the kernels in `kernels/`, one module per `.cu` file.

pub mod attention;
pub mod conv;
pub mod draft;
pub mod elementwise;
pub mod experts;
pub mod experts_rows;
pub mod linear;
pub mod linear_fp8;
pub mod linear_nvfp4;
pub mod linear_rows;
pub mod misc;
pub mod norm;
pub mod routing;
pub mod sample;
pub mod spec;
pub mod vision;

use std::{fmt, str::FromStr};

use anyhow::{Context as _, ensure};
use cudarc::driver::CudaViewMut;

/// Quantized activation layout written by fused norm-quant epilogues
/// (`kernels/quant_epilogue.cuh`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuantFormat {
    /// Natural MXFP8: E4M3 values `[rows, k]` and UE8M0 block scales `[rows, k / 32]`.
    Mxfp8 = 1,
    /// Paired MXFP8: E4M3 values `[rows, k]` in `expert_gemm_q8_nvfp4` word order and
    /// UE8M0 block scales `[rows, k / 32]`.
    Mxfp8Paired = 2,
    /// NVFP4: E2M1 pairs `[rows, k / 2]`, E4M3 block scales `[rows, k / 16]`, and per-row
    /// F32 outer scales `[rows]` (`amax / 1536`).
    Nvfp4 = 3,
}

impl QuantFormat {
    /// Mode constant passed to fused norm-quant kernels (`quant_epilogue::kQuant*`).
    #[must_use]
    pub const fn mode(self) -> i32 {
        self as i32
    }

    /// Quantized value bytes per row of `k` elements.
    #[must_use]
    pub const fn value_bytes(self, k: usize) -> usize {
        match self {
            Self::Mxfp8 | Self::Mxfp8Paired => k,
            Self::Nvfp4 => k / 2,
        }
    }

    /// Block-scale bytes per row of `k` elements.
    #[must_use]
    pub const fn scale_bytes(self, k: usize) -> usize {
        match self {
            Self::Mxfp8 | Self::Mxfp8Paired => k / 32,
            Self::Nvfp4 => k / 16,
        }
    }
}

/// Quantized activation destination for a fused norm-quant epilogue: `values` holds
/// `rows * format.value_bytes(2304)` bytes, `scales` holds `rows * format.scale_bytes(2304)`
/// bytes, and `outer` holds `rows` F32 outer scales (required for [`QuantFormat::Nvfp4`],
/// ignored for MXFP8 formats).
pub struct QuantDest<'a> {
    pub format: QuantFormat,
    pub values: CudaViewMut<'a, u8>,
    pub scales: CudaViewMut<'a, u8>,
    pub outer: Option<CudaViewMut<'a, f32>>,
}

impl QuantDest<'_> {
    pub(crate) fn check(&self, rows: usize, k: usize) -> anyhow::Result<()> {
        ensure!(
            crate::BLOCK_SCALED_MMA || self.format != QuantFormat::Nvfp4,
            "NVFP4 activation quantization needs sm_120a, which this sm_89 build lacks"
        );
        let need_vals = rows * self.format.value_bytes(k);
        let need_scales = rows * self.format.scale_bytes(k);
        ensure!(
            self.values.len() == need_vals,
            "quantized values has {} bytes, expected {need_vals}",
            self.values.len()
        );
        ensure!(
            self.scales.len() == need_scales,
            "quantized scales has {} bytes, expected {need_scales}",
            self.scales.len()
        );
        if self.format == QuantFormat::Nvfp4 {
            let outer = self
                .outer
                .as_ref()
                .context("NVFP4 quantization requires an outer scale buffer")?;
            ensure!(
                outer.len() == rows,
                "quantized outer scales has {} elements, expected {rows}",
                outer.len()
            );
        }
        Ok(())
    }

    pub(crate) fn check_strided(
        &self,
        rows: usize,
        k: usize,
        dst_stride: usize,
        col_offset: usize,
    ) -> anyhow::Result<()> {
        if dst_stride == k && col_offset == 0 {
            return self.check(rows, k);
        }
        ensure!(
            matches!(self.format, QuantFormat::Mxfp8 | QuantFormat::Mxfp8Paired),
            "strided quantized destinations ({dst_stride}, offset {col_offset}) are only supported for MXFP8"
        );
        ensure!(
            dst_stride >= k && dst_stride.is_multiple_of(32),
            "quantized dst_stride {dst_stride} must be >= {k} and a multiple of 32"
        );
        ensure!(
            col_offset + k <= dst_stride && col_offset.is_multiple_of(32),
            "quantized col_offset {col_offset} + {k} must be <= {dst_stride} and a multiple of 32"
        );
        let need_vals = (rows - 1) * dst_stride + col_offset + k;
        let need_scales = (rows - 1) * (dst_stride / 32) + (col_offset + k) / 32;
        ensure!(
            self.values.len() >= need_vals,
            "strided quantized values has {} bytes, need at least {need_vals}",
            self.values.len()
        );
        ensure!(
            self.scales.len() >= need_scales,
            "strided quantized scales has {} bytes, need at least {need_scales}",
            self.scales.len()
        );
        Ok(())
    }
}

/// How prefill multiplies its attention and dense products: in BF16 (F16 for attention's
/// weighted values), on the block-scaled FP8 tensor cores with activations quantized to E4M3,
/// or on Blackwell's native NVFP4 tensor cores for NVFP4 KV cache prefill attention
/// (`prefill_attention_fp4_mma`: NVFP4 `Q K^T` with FP8 `P V`). Lower-precision modes are
/// opt-in for prefill; decode never uses them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Products {
    #[default]
    Bf16,
    Fp8,
    Fp4,
}

impl FromStr for Products {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> anyhow::Result<Self> {
        match s {
            "bf16" => Ok(Self::Bf16),
            "fp8" => Ok(Self::Fp8),
            "fp4" => Ok(Self::Fp4),
            _ => anyhow::bail!("unknown product format {s:?}; expected bf16, fp8 or fp4"),
        }
    }
}

impl fmt::Display for Products {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Bf16 => "bf16",
            Self::Fp8 => "fp8",
            Self::Fp4 => "fp4",
        })
    }
}
