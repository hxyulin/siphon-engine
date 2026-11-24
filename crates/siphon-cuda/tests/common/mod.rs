//! Shared helpers for kernel tests: MLX fixtures and tolerance checks.

#![allow(dead_code, reason = "each test binary uses a different subset")]

use std::{fs, path::PathBuf};

use anyhow::{Context as _, bail};
use half::bf16;
use safetensors::{Dtype, SafeTensors};

/// Tensors that a `reference/fixtures/<name>.py` script saved from MLX.
pub struct Fixture {
    name: String,
    bytes: Vec<u8>,
}

impl Fixture {
    /// Loads `tests/fixtures/<name>.safetensors`.
    pub fn load(name: &str) -> anyhow::Result<Self> {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join(format!("tests/fixtures/{name}.safetensors"));
        let bytes = fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
        Ok(Self {
            name: name.to_owned(),
            bytes,
        })
    }

    fn raw(&self, key: &str, dtype: Dtype) -> anyhow::Result<(Vec<u8>, Vec<usize>)> {
        let tensors = SafeTensors::deserialize(&self.bytes)?;
        let view = tensors
            .tensor(key)
            .with_context(|| format!("{key} missing from fixture {}", self.name))?;
        if view.dtype() != dtype {
            bail!("{key} is {:?}, expected {dtype:?}", view.dtype());
        }
        Ok((view.data().to_vec(), view.shape().to_vec()))
    }

    pub fn bf16(&self, key: &str) -> anyhow::Result<(Vec<bf16>, Vec<usize>)> {
        let (data, shape) = self.raw(key, Dtype::BF16)?;
        let values = data
            .as_chunks()
            .0
            .iter()
            .map(|b| bf16::from_le_bytes(*b))
            .collect();
        Ok((values, shape))
    }

    pub fn f32(&self, key: &str) -> anyhow::Result<(Vec<f32>, Vec<usize>)> {
        let (data, shape) = self.raw(key, Dtype::F32)?;
        let values = data
            .as_chunks()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        Ok((values, shape))
    }

    pub fn i32(&self, key: &str) -> anyhow::Result<(Vec<i32>, Vec<usize>)> {
        let (data, shape) = self.raw(key, Dtype::I32)?;
        let values = data
            .as_chunks()
            .0
            .iter()
            .map(|b| i32::from_le_bytes(*b))
            .collect();
        Ok((values, shape))
    }
}

/// Distance between two BF16 values in units in the last place, ordering negatives correctly.
pub fn bf16_ulps(a: bf16, b: bf16) -> u32 {
    let key = |v: bf16| {
        let bits = i32::from(v.to_bits());
        if bits & 0x8000 == 0 {
            bits
        } else {
            0x8000 - bits
        }
    };
    key(a).abs_diff(key(b))
}

/// Fails if any element differs by more than `max_ulps`, and reports how many differ at all.
pub fn assert_bf16_close(label: &str, got: &[bf16], want: &[bf16], max_ulps: u32) {
    assert_eq!(got.len(), want.len(), "{label}: length");
    let mut differing = 0;
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let ulps = bf16_ulps(g, w);
        assert!(
            ulps <= max_ulps,
            "{label}[{i}]: got {g}, want {w} ({ulps} ulps)"
        );
        differing += usize::from(ulps > 0);
    }
    eprintln!(
        "{label}: {differing} of {} elements differ by at most {max_ulps} ulp",
        want.len()
    );
}

/// Fails if any element is outside `atol + rtol * |want|`.
pub fn assert_f32_close(label: &str, got: &[f32], want: &[f32], rtol: f32, atol: f32) {
    assert_eq!(got.len(), want.len(), "{label}: length");
    let mut worst = 0.0f32;
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let err = (g - w).abs();
        assert!(
            err <= atol + rtol * w.abs(),
            "{label}[{i}]: got {g}, want {w}"
        );
        worst = worst.max(err);
    }
    eprintln!("{label}: max abs error {worst:e}");
}

/// `len` deterministic values uniform in `[-scale, scale)`, different for each `seed`.
pub fn pseudo_random(len: usize, seed: u32, scale: f32) -> Vec<f32> {
    (0..len as u32)
        .map(|i| {
            let mut h = i.wrapping_mul(0x9E37_79B1) ^ seed.wrapping_mul(0x85EB_CA77);
            h ^= h >> 15;
            h = h.wrapping_mul(0x2C1B_3C6D);
            h ^= h >> 12;
            h = h.wrapping_mul(0x297A_2D39);
            h ^= h >> 15;
            let unit = (h >> 8) as f32 * 2f32.powi(-24);
            (2.0 * unit - 1.0) * scale
        })
        .collect()
}

/// [`pseudo_random`] rounded to BF16, offset by `offset`.
pub fn pseudo_random_bf16(len: usize, seed: u32, scale: f32, offset: f32) -> Vec<bf16> {
    pseudo_random(len, seed, scale)
        .into_iter()
        .map(|v| bf16::from_f32(v + offset))
        .collect()
}

/// Distance between two finite F32 values in units in the last place.
pub fn f32_ulp_distance(a: f32, b: f32) -> u64 {
    let key = |v: f32| {
        let bits = i64::from(v.to_bits());
        if bits & 0x8000_0000 == 0 {
            bits
        } else {
            0x8000_0000 - bits
        }
    };
    key(a).abs_diff(key(b))
}

/// Fails if any element differs by more than `max_ulps`, and reports how many differ at all.
pub fn assert_f32_within_ulps(label: &str, got: &[f32], want: &[f32], max_ulps: u64) {
    assert_eq!(got.len(), want.len(), "{label}: length");
    let mut differing = 0;
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let ulps = f32_ulp_distance(g, w);
        assert!(
            ulps <= max_ulps,
            "{label}[{i}]: got {g}, want {w} ({ulps} ulps)"
        );
        differing += usize::from(ulps > 0);
    }
    eprintln!(
        "{label}: {differing} of {} elements differ by at most {max_ulps} ulp",
        want.len()
    );
}
