//! The DSpark draft's small kernels against MLX outputs from `reference/fixtures/dspark.py`,
//! which runs `reference/dspark.py` in its BF16 mode.
//!
//! The kernels follow the reference's rounding points but not MLX's summation order or its
//! Metal exp, tanh, sin and cos, so a BF16 output may differ by one rounding step where the
//! F32 value sits near a rounding boundary. The tolerances:
//! - norms, RoPE, `silu * up`, the Markov gate and the Markov logits: 1 BF16 ulp;
//! - block attention: 2 BF16 ulps (a relative 2^-7), plus 2^-16 of the largest output for
//!   outputs near zero, since the softmax and the weighted sum of 1032 values sum in
//!   another order;
//! - the fold, summed in F64 before one rounding: bit-exact;
//! - copies (the block's mask rows, the residual sum) and the Markov argmax: exact.

#![cfg(target_os = "linux")]

mod common;

use std::sync::Arc;

use common::{Fixture, assert_bf16_close, assert_f32_close};
use cudarc::driver::{CudaSlice, CudaStream};
use half::bf16;
use siphon_cuda::{
    Gpu,
    draft::rope_inv_freq,
    ops::draft::{
        ArgmaxPartials, AttentionScratch, BLOCK, CONTROL, DraftKernels, KvCache, LOGIT_BLOCKS,
    },
};

const EPS: f32 = 1e-6;
const HIDDEN: usize = 2304;
const HEAD_DIM: usize = 128;
const KV_WIDTH: usize = 8 * HEAD_DIM;

struct Setup {
    fixture: Fixture,
    gpu: Gpu,
    kernels: DraftKernels,
}

impl Setup {
    fn new() -> anyhow::Result<Self> {
        let gpu = Gpu::new(0)?;
        let kernels = DraftKernels::load(gpu.context())?;
        Ok(Self {
            fixture: Fixture::load("dspark")?,
            gpu,
            kernels,
        })
    }

    fn stream(&self) -> &Arc<CudaStream> {
        self.gpu.stream()
    }

    fn bf16(&self, key: &str) -> anyhow::Result<CudaSlice<bf16>> {
        Ok(self.stream().clone_htod(&self.fixture.bf16(key)?.0)?)
    }

    fn f32(&self, key: &str) -> anyhow::Result<CudaSlice<f32>> {
        Ok(self.stream().clone_htod(&self.fixture.f32(key)?.0)?)
    }

    fn check(
        &self,
        label: &str,
        got: &CudaSlice<bf16>,
        key: &str,
        ulps: u32,
    ) -> anyhow::Result<()> {
        let got = self.stream().clone_dtoh(got)?;
        assert_bf16_close(label, &got, &self.fixture.bf16(key)?.0, ulps);
        Ok(())
    }
}

#[test]
fn mix_rms_norm_matches_reference() -> anyhow::Result<()> {
    let t = Setup::new()?;
    let taps = t.bf16("mix.taps")?;
    let alpha = t.f32("mix.alpha")?;
    let weight = t.bf16("mix.weight")?;
    let mut out = t.stream().alloc_zeros::<bf16>(taps.len() / 4)?;
    t.kernels
        .mix_rms_norm(t.stream(), &taps, &alpha.slice(..), &weight, &mut out, EPS)?;
    t.check("mix", &out, "mix.out", 1)
}

#[test]
fn add_rms_norm_matches_reference() -> anyhow::Result<()> {
    let t = Setup::new()?;
    let x = t.bf16("add_norm.x")?;
    let weight = t.bf16("add_norm.weight")?;
    let mut out = t.stream().alloc_zeros::<bf16>(x.len())?;

    let mut residual = t.bf16("add_norm.residual")?;
    t.kernels
        .add_rms_norm(t.stream(), None, &mut residual, &weight, &mut out, EPS)?;
    t.check("norm", &out, "norm.out", 1)?;
    t.check("norm residual", &residual, "add_norm.residual", 0)?;

    t.kernels
        .add_rms_norm(t.stream(), Some(&x), &mut residual, &weight, &mut out, EPS)?;
    t.check("add norm", &out, "add_norm.out", 1)?;
    t.check("add norm residual", &residual, "add_norm.residual_out", 0)
}

#[test]
fn block_input_matches_reference() -> anyhow::Result<()> {
    let t = Setup::new()?;
    let anchor = t
        .stream()
        .clone_htod(&t.fixture.i32("block_input.anchor")?.0)?;
    let embed = t.bf16("block_input.embed")?;
    let embed_norm = t.bf16("block_input.embed_norm")?;
    let mask = t.bf16("block_input.mask_embed")?;
    let mut out = t.stream().alloc_zeros::<bf16>(8 * HIDDEN)?;
    t.kernels.block_input(
        t.stream(),
        &anchor,
        &embed,
        &embed_norm,
        &mask,
        &mut out,
        EPS,
    )?;
    let got = t.stream().clone_dtoh(&out)?;
    let want = t.fixture.bf16("block_input.out")?.0;
    assert_bf16_close("anchor row", &got[..HIDDEN], &want[..HIDDEN], 1);
    assert_bf16_close("mask rows", &got[HIDDEN..], &want[HIDDEN..], 0);
    Ok(())
}

#[test]
fn head_norm_rope_matches_reference() -> anyhow::Result<()> {
    let t = Setup::new()?;
    let weight = t.bf16("rope.weight")?;
    let inv_freq = t.stream().clone_htod(&rope_inv_freq())?;
    for (name, heads) in [("q", 24), ("k", 8)] {
        let mut x = t.bf16(&format!("rope.{name}.x"))?;
        let start = t.fixture.i32(&format!("rope.{name}.start"))?.0[0];
        let rows = x.len() / (heads * HEAD_DIM);
        t.kernels.head_norm_rope(
            t.stream(),
            &mut x,
            rows,
            heads,
            heads * HEAD_DIM,
            &weight,
            &inv_freq,
            usize::try_from(start)?,
            EPS,
        )?;
        t.check(&format!("rope {name}"), &x, &format!("rope.{name}.out"), 1)?;
    }
    Ok(())
}

/// Block attention with the keys and values stored in a cache of `slots` slots, ring or not.
#[test]
fn attention_matches_reference() -> anyhow::Result<()> {
    let t = Setup::new()?;
    let mut scratch = AttentionScratch::new(t.stream())?;
    for start in [1030usize, 5] {
        let q = t.bf16(&format!("attention.{start}.q"))?;
        let keys = t.bf16(&format!("attention.{start}.keys"))?;
        let values = t.bf16(&format!("attention.{start}.values"))?;
        let positions = start + 8;
        for (label, local, slots) in [("local", true, 1032), ("global", false, 2048)] {
            let mut cache = KvCache::new(t.stream(), slots)?;
            // The local ring holds only the last 1032 positions, as a draft cache would.
            let first_stored = positions.saturating_sub(slots);
            let rows = positions - first_stored;
            let k_rows = keys.slice(first_stored * KV_WIDTH..);
            let v_rows = values.slice(first_stored * KV_WIDTH..);
            t.kernels.store_kv(
                t.stream(),
                &k_rows,
                &v_rows,
                rows,
                KV_WIDTH,
                &mut cache,
                first_stored,
            )?;
            let first = if local { start.saturating_sub(1024) } else { 0 };
            let mut out = t.stream().alloc_zeros::<bf16>(q.len())?;
            t.kernels.attention(
                t.stream(),
                &q,
                &cache,
                first,
                positions - first,
                &mut scratch,
                &mut out,
            )?;
            let got: Vec<f32> = t
                .stream()
                .clone_dtoh(&out)?
                .iter()
                .map(|v| v.to_f32())
                .collect();
            let want: Vec<f32> = t
                .fixture
                .bf16(&format!("attention.{start}.{label}"))?
                .0
                .iter()
                .map(|v| v.to_f32())
                .collect();
            // Outputs near zero come from cancellation: allow an absolute error at the
            // F32 summation level of the largest output as well.
            let largest = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            assert_f32_close(
                &format!("attention {start} {label}"),
                &got,
                &want,
                2.0 / 256.0,
                largest * 2f32.powi(-16),
            );
        }
    }
    Ok(())
}

/// The value of an E4M3 byte.
fn e4m3(byte: u8) -> f32 {
    let sign = if byte & 0x80 == 0 { 1.0 } else { -1.0 };
    let exponent = i32::from((byte >> 3) & 0xf);
    let mantissa = f32::from(byte & 7);
    sign * if exponent == 0 {
        mantissa / 8.0 * 2f32.powi(-6)
    } else {
        (1.0 + mantissa / 8.0) * 2f32.powi(exponent - 7)
    }
}

/// Widens an E4M3 cache's bytes to BF16 values on the host, checking that each is a nearest
/// E4M3 value to the matching BF16 value of `want`.
fn widen_checked(label: &str, bytes: &[u8], want: &[bf16]) -> Vec<bf16> {
    // Every finite E4M3 value (0x7f and 0xff are NaN).
    let finite: Vec<f32> = (0..=255u8).filter(|b| b & 0x7f != 0x7f).map(e4m3).collect();
    bytes
        .iter()
        .zip(want)
        .enumerate()
        .map(|(i, (&b, w))| {
            let got = e4m3(b);
            let x = w.to_f32();
            let best = finite
                .iter()
                .map(|v| (v - x).abs())
                .fold(f32::INFINITY, f32::min);
            assert!(
                (got - x).abs() <= best,
                "{label} element {i}: {got} is not a nearest E4M3 value to {x}"
            );
            bf16::from_f32(got)
        })
        .collect()
}

/// Prints the largest magnitude of `values` and the share below E4M3's smallest normal
/// value, 2^-6, where E4M3 keeps fewer than 4 significant bits.
fn print_range(label: &str, values: &[bf16]) {
    let largest = values.iter().fold(0.0f32, |m, v| m.max(v.to_f32().abs()));
    let small = values
        .iter()
        .filter(|v| v.to_f32() != 0.0 && v.to_f32().abs() < 2f32.powi(-6))
        .count();
    eprintln!(
        "{label}: largest |x| {largest}, {:.3}% below 2^-6",
        100.0 * small as f64 / values.len() as f64
    );
}

/// An E4M3 cache holds each stored BF16 value rounded to a nearest E4M3 value, and block
/// attention over it equals, bit for bit, attention over a BF16 cache of those values.
#[test]
fn fp8_cache_matches_a_bf16_cache_of_rounded_values() -> anyhow::Result<()> {
    let t = Setup::new()?;
    let mut scratch = AttentionScratch::new(t.stream())?;
    let start = 1030usize;
    let q = t.bf16(&format!("attention.{start}.q"))?;
    let keys = t.bf16(&format!("attention.{start}.keys"))?;
    let values = t.bf16(&format!("attention.{start}.values"))?;
    let positions = start + 8;
    let slots = 2048;
    let mut fp8 = KvCache::with_format(t.stream(), slots, true)?;
    t.kernels
        .store_kv(t.stream(), &keys, &values, positions, KV_WIDTH, &mut fp8, 0)?;
    let stored = positions * KV_WIDTH;
    let k_bytes = t.stream().clone_dtoh(&fp8.k.slice(..stored))?;
    let v_bytes = t.stream().clone_dtoh(&fp8.v.slice(..stored))?;
    let (k_host, v_host) = (
        t.stream().clone_dtoh(&keys)?,
        t.stream().clone_dtoh(&values)?,
    );
    print_range("keys", &k_host[..stored]);
    print_range("values", &v_host[..stored]);
    let k_rounded = widen_checked("keys", &k_bytes, &k_host[..stored]);
    let v_rounded = widen_checked("values", &v_bytes, &v_host[..stored]);
    let mut bf16_cache = KvCache::new(t.stream(), slots)?;
    t.kernels.store_kv(
        t.stream(),
        &t.stream().clone_htod(&k_rounded)?,
        &t.stream().clone_htod(&v_rounded)?,
        positions,
        KV_WIDTH,
        &mut bf16_cache,
        0,
    )?;
    let mut outs = Vec::new();
    for cache in [&fp8, &bf16_cache] {
        let mut out = t.stream().alloc_zeros::<bf16>(q.len())?;
        t.kernels
            .attention(t.stream(), &q, cache, 0, positions, &mut scratch, &mut out)?;
        outs.push(t.stream().clone_dtoh(&out)?);
    }
    assert!(outs[0] == outs[1], "attention over the E4M3 cache differs");

    // The block's stores round as the copies do.
    let rows = 8;
    let pattern = |n: usize, seed: usize| -> Vec<bf16> {
        (0..n)
            .map(|i| bf16::from_f32((((i * 7919 + seed) % 2003) as f32 - 1001.0) / 97.0))
            .collect()
    };
    let mut block_q = t.stream().clone_htod(&pattern(rows * 24 * HEAD_DIM, 1))?;
    let block_k = t.stream().clone_htod(&pattern(rows * KV_WIDTH, 2))?;
    let block_v = t.stream().clone_htod(&pattern(rows * KV_WIDTH, 3))?;
    let norm = t.stream().clone_htod(&pattern(HEAD_DIM, 4))?;
    let control = t.stream().clone_htod(&[100i32, 0, 0])?;
    let inv_freq = t.stream().clone_htod(&rope_inv_freq())?;
    let mut caches = [
        KvCache::with_format(t.stream(), slots, false)?,
        KvCache::with_format(t.stream(), slots, true)?,
    ];
    for cache in &mut caches {
        t.kernels.block_rope_store(
            t.stream(),
            &mut block_q,
            &block_k,
            &block_v,
            &control,
            &inv_freq,
            EPS,
            &norm,
            &norm,
            cache,
            false,
            siphon_cuda::ops::draft::Rope::F32,
        )?;
    }
    let range = 100 * KV_WIDTH..(100 + rows) * KV_WIDTH;
    for (label, bf, f8) in [
        ("block keys", &caches[0].k, &caches[1].k),
        ("block values", &caches[0].v, &caches[1].v),
    ] {
        let want: Vec<bf16> = t
            .stream()
            .clone_dtoh(&bf.slice(2 * range.start..2 * range.end))?
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| bf16::from_le_bytes(*b))
            .collect();
        widen_checked(
            label,
            &t.stream().clone_dtoh(&f8.slice(range.clone()))?,
            &want,
        );
    }
    Ok(())
}

#[test]
fn silu_mul_matches_reference() -> anyhow::Result<()> {
    let t = Setup::new()?;
    let gate = t.bf16("silu.gate")?;
    let up = t.bf16("silu.up")?;
    let mut out = t.stream().alloc_zeros::<bf16>(gate.len())?;
    t.kernels.silu_mul(t.stream(), &gate, &up, &mut out)?;
    t.check("silu", &out, "silu.out", 1)
}

#[test]
fn markov_gate_matches_reference() -> anyhow::Result<()> {
    let t = Setup::new()?;
    let context = t.f32("gate.context")?;
    let bias = t.f32("gate.bias")?;
    let raw = t.f32("gate.raw")?;
    let state: Vec<f32> = t
        .fixture
        .bf16("gate.state")?
        .0
        .iter()
        .map(|v| v.to_f32())
        .chain(std::iter::repeat_n(0.0, 256))
        .collect();
    let mut step_in = t.stream().clone_htod(&state)?;
    let mut latent = t.stream().alloc_zeros::<f32>(256)?;
    t.kernels.markov_gate(
        t.stream(),
        &raw,
        &context.slice(..),
        &bias,
        &mut step_in,
        &mut latent,
    )?;
    let to_bf16 = |v: Vec<f32>| v.into_iter().map(bf16::from_f32).collect::<Vec<_>>();
    let state_out = to_bf16(t.stream().clone_dtoh(&step_in)?);
    assert_bf16_close(
        "state",
        &state_out[..256],
        &t.fixture.bf16("gate.state_out")?.0,
        1,
    );
    let latent = to_bf16(t.stream().clone_dtoh(&latent)?);
    assert_bf16_close("latent", &latent, &t.fixture.bf16("gate.latent")?.0, 1);
    Ok(())
}

#[test]
fn markov_logits_and_pick_match_reference() -> anyhow::Result<()> {
    let t = Setup::new()?;
    let base = t.bf16("logits.base")?;
    let fold_bias = t.bf16("logits.fold_bias")?;
    let acc = t.f32("logits.acc")?;
    let vocab = base.len();
    let mut logits = t.stream().alloc_zeros::<bf16>(vocab)?;
    let mut partials = ArgmaxPartials::new(t.stream())?;
    t.kernels.markov_logits(
        t.stream(),
        &base.slice(..),
        &fold_bias,
        &acc,
        vocab,
        30.0,
        Some(logits.slice_mut(..)),
        &mut partials,
    )?;
    t.check("logits", &logits, "logits.out", 1)?;

    // A w1 table whose row r holds r, so step_in shows which row the pick copied.
    let w1: Vec<bf16> = (0..vocab * 256)
        .map(|i| bf16::from_f32((i / 256 % 256) as f32))
        .collect();
    let w1 = t.stream().clone_htod(&w1)?;
    let mut step_in = t.stream().alloc_zeros::<f32>(512)?;
    let mut id = t.stream().alloc_zeros::<i32>(1)?;
    t.kernels
        .markov_pick(t.stream(), &partials, &mut id, &w1, &mut step_in)?;
    let id = t.stream().clone_dtoh(&id)?[0];
    assert_eq!(
        id,
        t.fixture.i32("logits.argmax")?.0[0],
        "argmax over {LOGIT_BLOCKS} blocks"
    );
    let step_in = t.stream().clone_dtoh(&step_in)?;
    assert!(
        step_in[256..]
            .iter()
            .all(|&v| v.to_bits() == ((id % 256) as f32).to_bits()),
        "step_in holds the picked token's w1 row"
    );
    Ok(())
}

#[test]
fn fold_matches_reference() -> anyhow::Result<()> {
    let t = Setup::new()?;
    let lm_head = t.bf16("fold.lm_head")?;
    let hidden_out = t.bf16("fold.hidden_out")?;
    let hidden_out_bias = t.bf16("fold.hidden_out_bias")?;
    let rows = lm_head.len() / HIDDEN;
    let mut weight = t.stream().alloc_zeros::<bf16>(rows * 256)?;
    let mut bias = t.stream().alloc_zeros::<bf16>(rows)?;
    t.kernels.fold(
        t.stream(),
        &lm_head,
        &hidden_out,
        &hidden_out_bias,
        &mut weight,
        &mut bias,
    )?;
    t.check("fold weight", &weight, "fold.weight", 0)?;
    t.check("fold bias", &bias, "fold.bias", 0)
}

#[test]
fn silu_mul_rows_vectorized_and_tail_exact() -> anyhow::Result<()> {
    let gpu = Gpu::new(0)?;
    let kernels = DraftKernels::load(gpu.context())?;
    let stream = gpu.stream();

    // Exact reference in F32 rounded to BF16.
    let ref_silu_mul = |g: bf16, u: bf16| -> bf16 {
        let gf = g.to_f32();
        let uf = u.to_f32();
        let silu = gf / (1.0f32 + (-gf).exp());
        bf16::from_f32(silu * uf)
    };

    // 1. Large 16-byte-aligned multiple-of-8 length (matches B=32 DFlash 2 MLP: 256 * 6912).
    let n_vec = 256 * 6912;
    let gate_h: Vec<bf16> = (0..n_vec)
        .map(|i| bf16::from_f32(((i % 199) as f32 - 99.0) * 0.05))
        .collect();
    let up_h: Vec<bf16> = (0..n_vec)
        .map(|i| bf16::from_f32(((i % 151) as f32 - 75.0) * 0.03))
        .collect();
    let gate = stream.clone_htod(&gate_h)?;
    let up = stream.clone_htod(&up_h)?;
    let mut out = stream.alloc_zeros::<bf16>(n_vec)?;
    kernels.silu_mul_rows(stream, &gate, &up, &mut out)?;
    let got = stream.clone_dtoh(&out)?;
    let want: Vec<bf16> = gate_h
        .iter()
        .zip(&up_h)
        .map(|(&g, &u)| ref_silu_mul(g, u))
        .collect();
    assert_bf16_close("silu_mul_rows aligned", &got, &want, 1);

    // 2. Unaligned offset + non-multiple-of-8 length (exercises scalar fallback).
    let mut out_tail = stream.alloc_zeros::<bf16>(32)?;
    kernels.silu_mul_rows(
        stream,
        &gate.slice(1..14),
        &up.slice(1..14),
        &mut out_tail.slice_mut(1..14),
    )?;
    let got_tail = stream.clone_dtoh(&out_tail)?;
    assert_bf16_close("silu_mul_rows tail", &got_tail[1..14], &want[1..14], 1);
    Ok(())
}

#[test]
fn batch_helper_kernels_exact() -> anyhow::Result<()> {
    let gpu = Gpu::new(0)?;
    let kernels = DraftKernels::load(gpu.context())?;
    let stream = gpu.stream();

    // 1. set_control_kept_batch: verifies clamping of negative, normal, and overflow kept counts.
    let seqs = 3usize;
    let stride = 4usize;
    let starts = stream.clone_htod(&[10i32, 200, 1024])?;
    let results = stream.clone_htod(&[-1i32, 99, 99, 99, 3, 99, 99, 99, 12, 99, 99, 99])?;
    let anchors = stream.clone_htod(&[42i32, 7, 100])?;
    let mut controls = stream.alloc_zeros::<i32>(seqs * CONTROL)?;
    kernels.set_control_kept_batch(
        stream,
        &mut controls,
        &starts,
        &results,
        stride,
        8,
        &anchors,
        seqs,
    )?;
    let got_ctrl = stream.clone_dtoh(&controls)?;
    assert_eq!(got_ctrl, vec![10, 1, 42, 200, 4, 7, 1024, 8, 100]);

    // 2. block_input_batch: verifies vectorized uint4 copy of mask_embed into rows 1..7.
    let vocab = 128usize;
    let embed = stream.alloc_zeros::<bf16>(vocab * HIDDEN)?;
    let embed_norm = stream.clone_htod(&vec![bf16::from_f32(1.0); HIDDEN])?;
    let mask_h: Vec<bf16> = (0..HIDDEN)
        .map(|c| bf16::from_f32(((c % 97) as f32 - 48.0) * 0.125))
        .collect();
    let mask_embed = stream.clone_htod(&mask_h)?;
    let mut block_out = stream.alloc_zeros::<bf16>(seqs * BLOCK * HIDDEN)?;
    kernels.block_input_batch(
        stream,
        &controls,
        &embed,
        &embed_norm,
        &mask_embed,
        &mut block_out,
        seqs,
        EPS,
    )?;
    let got_block = stream.clone_dtoh(&block_out)?;
    for s in 0..seqs {
        for r in 1..BLOCK {
            let off = (s * BLOCK + r) * HIDDEN;
            assert_eq!(
                &got_block[off..off + HIDDEN],
                mask_h.as_slice(),
                "seq {s} row {r} mask_embed mismatch"
            );
        }
    }

    // 3. gather_mask_rows: verifies vectorized uint4 copy of rows 1..=steps for each sequence.
    let steps = 5usize;
    let y_h: Vec<bf16> = (0..seqs * BLOCK * HIDDEN)
        .map(|i| bf16::from_f32(((i / HIDDEN) as f32) + ((i % 17) as f32) * 0.0625))
        .collect();
    let y = stream.clone_htod(&y_h)?;
    let mut y_steps = stream.alloc_zeros::<bf16>(seqs * steps * HIDDEN)?;
    kernels.gather_mask_rows(stream, &y, &mut y_steps, seqs, steps)?;
    let got_steps = stream.clone_dtoh(&y_steps)?;
    for s in 0..seqs {
        for e in 0..steps {
            let src_off = (s * BLOCK + e + 1) * HIDDEN;
            let dst_off = (s * steps + e) * HIDDEN;
            assert_eq!(
                &got_steps[dst_off..dst_off + HIDDEN],
                &y_h[src_off..src_off + HIDDEN],
                "seq {s} step {e} gather mismatch"
            );
        }
    }
    Ok(())
}

/// Parity and tolerance check for DFlash 2 with `--draft-weights nvfp4-backbone` and
/// `--draft-weights nvfp4` against FP8:
/// - Full-vocab base logits cosine similarity and top-1 agreement (`fp8` vs `nvfp4-backbone`)
///   over real target hidden-state taps;
/// - Batched DFlash 2 (`seqs = 4`, `m = 32 > 16`, CUDA-graph captured) with both
///   `nvfp4-backbone` and `nvfp4` produces bit-identical draft tokens to single-stream.
#[test]
#[ignore = "needs SIPHON_MODEL, SIPHON_DFLASH (DFlash 2), and a GPU"]
#[expect(clippy::too_many_lines, reason = "end-to-end NVFP4 vs FP8 DFlash 2 parity test")]
fn dflash2_nvfp4_parity_with_fp8() -> anyhow::Result<()> {
    use std::{env, path::PathBuf};

    use anyhow::{Context as _, ensure};
    use siphon::{Checkpoint, TextConfig};
    use siphon_cuda::{
        draft::{DraftOptions, STEPS},
        model::{Model, PrefillLogits, Sampling},
        ops::{attention::KvFormat, linear::Linear},
        spec::{DraftSource, load_draft},
        weights::WeightFormat,
    };

    let model_dir = PathBuf::from(env::var_os("SIPHON_MODEL").context("set SIPHON_MODEL")?);
    let draft_dir = PathBuf::from(env::var_os("SIPHON_DFLASH").context("set SIPHON_DFLASH")?);

    let gpu = Gpu::new(0)?;
    let stream = Arc::clone(gpu.stream());
    let config = TextConfig::load(&model_dir)?;
    let vocab = config.unpadded_vocab_size;
    let padded_vocab = config.vocab_size;
    let passes = config.loop_config.num_passes;
    let checkpoint = Checkpoint::open(&model_dir)?;
    let weights = siphon_cuda::weights::Weights::load(&checkpoint, &config, &stream)?;
    let capacity = 2048;
    let mut model = Model::with_formats(
        &gpu,
        config,
        weights,
        passes,
        capacity,
        KvFormat::Fp8,
        WeightFormat::Fp8,
    )?;

    let mut prompt = vec![1u32];
    let mut tok = model.step(1, None)?;
    prompt.push(tok);
    for _ in 0..46 {
        tok = model.step(tok, None)?;
        prompt.push(tok);
    }
    model.reset()?;

    let source = DraftSource::Dflash(&draft_dir);
    let fp8_opts = DraftOptions::fp8();
    let mut nvfp4_bb_opts = DraftOptions::fp8();
    nvfp4_bb_opts.apply_draft_weights("nvfp4-backbone")?;
    ensure!(nvfp4_bb_opts.nvfp4_backbone && !nvfp4_bb_opts.fp8_backbone && nvfp4_bb_opts.fp8_head);

    let mut fp8_draft = load_draft(&mut model, &checkpoint, source, capacity, &fp8_opts)?;
    ensure!(fp8_draft.is_dflash2(), "SIPHON_DFLASH must be a DFlash 2 drafter");
    let layout = fp8_draft.tap_layout();
    let row_len = layout.len() * HIDDEN;
    let mut host_taps = vec![bf16::ZERO; prompt.len() * row_len];
    {
        let mut collect = |start: usize, taps: cudarc::driver::CudaView<'_, bf16>| {
            let vals = stream.clone_dtoh(&taps)?;
            host_taps[start * row_len..start * row_len + vals.len()].copy_from_slice(&vals);
            Ok(())
        };
        model.prefill_with_taps(&prompt, 2048, PrefillLogits::Last, &mut collect)?;
    }
    let dev_taps = stream.clone_htod(&host_taps)?;
    let linear = Linear::load(stream.context())?;
    let prompt_i32: Vec<i32> = prompt
        .iter()
        .map(|&t| Ok(i32::try_from(t)?))
        .collect::<anyhow::Result<_>>()?;
    let mut anchor = stream.alloc_zeros::<i32>(1)?;

    // 1. Full-vocab logit & draft comparison: FP8 backbone vs NVFP4 backbone (both with FP8 head).
    let first = 16usize;
    let anchors = 16usize;
    ensure!(prompt.len() >= first + anchors + 1);

    let mut fp8_ids = Vec::with_capacity(anchors * STEPS);
    let mut fp8_logits = Vec::new();
    {
        let shared = model.draft_shared();
        fp8_draft.commit(&stream, &linear, &dev_taps.slice(..first * row_len), 0)?;
        for a in 0..anchors {
            let p = first + a;
            stream.memcpy_htod(&prompt_i32[p..=p], &mut anchor)?;
            let commit_taps = if a == 0 {
                dev_taps.slice(..0)
            } else {
                dev_taps.slice((p - 1) * row_len..p * row_len)
            };
            let commit_start = if a == 0 { p } else { p - 1 };
            fp8_draft.set_round_inputs(&stream, &commit_taps, commit_start, &anchor)?;
            fp8_draft.replay_round(&stream, &linear, shared, STEPS)?;
            fp8_ids.extend(stream.clone_dtoh(fp8_draft.draft_ids())?);
            if a < 4 {
                fp8_logits.push(stream.clone_dtoh(fp8_draft.base_logits())?);
            }
        }
    }
    drop(fp8_draft);

    let mut nvfp4_draft = load_draft(&mut model, &checkpoint, source, capacity, &nvfp4_bb_opts)?;
    let mut nvfp4_ids = Vec::with_capacity(anchors * STEPS);
    let mut nvfp4_logits = Vec::new();
    {
        let shared = model.draft_shared();
        nvfp4_draft.commit(&stream, &linear, &dev_taps.slice(..first * row_len), 0)?;
        for a in 0..anchors {
            let p = first + a;
            stream.memcpy_htod(&prompt_i32[p..=p], &mut anchor)?;
            let commit_taps = if a == 0 {
                dev_taps.slice(..0)
            } else {
                dev_taps.slice((p - 1) * row_len..p * row_len)
            };
            let commit_start = if a == 0 { p } else { p - 1 };
            nvfp4_draft.set_round_inputs(&stream, &commit_taps, commit_start, &anchor)?;
            nvfp4_draft.replay_round(&stream, &linear, shared, STEPS)?;
            nvfp4_ids.extend(stream.clone_dtoh(nvfp4_draft.draft_ids())?);
            if a < 4 {
                nvfp4_logits.push(stream.clone_dtoh(nvfp4_draft.base_logits())?);
            }
        }
    }
    drop(nvfp4_draft);

    let mut dot = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;
    let mut base_top1 = 0usize;
    let mut base_total = 0usize;
    for (fl, nl) in fp8_logits.iter().zip(&nvfp4_logits) {
        for s in 0..STEPS {
            let fr = &fl[s * padded_vocab..s * padded_vocab + vocab];
            let nr = &nl[s * padded_vocab..s * padded_vocab + vocab];
            let (mut best_fp8_id, mut best_fp8_val) = (0usize, f32::NEG_INFINITY);
            let (mut best_nvfp4_id, mut best_nvfp4_val) = (0usize, f32::NEG_INFINITY);
            for (j, (fv, nv)) in fr.iter().zip(nr).enumerate() {
                let a = f64::from(fv.to_f32());
                let b = f64::from(nv.to_f32());
                dot += a * b;
                norm_a += a * a;
                norm_b += b * b;
                if fv.to_f32() > best_fp8_val {
                    best_fp8_val = fv.to_f32();
                    best_fp8_id = j;
                }
                if nv.to_f32() > best_nvfp4_val {
                    best_nvfp4_val = nv.to_f32();
                    best_nvfp4_id = j;
                }
            }
            base_top1 += usize::from(best_fp8_id == best_nvfp4_id);
            base_total += 1;
        }
    }
    let cosine = dot / (norm_a.sqrt() * norm_b.sqrt());
    let base_top1_rate = base_top1 as f64 / base_total as f64;
    let draft_agree = fp8_ids
        .iter()
        .zip(&nvfp4_ids)
        .filter(|(a, b)| a == b)
        .count();
    let draft_agree_rate = draft_agree as f64 / fp8_ids.len() as f64;
    eprintln!(
        "DFlash 2 NVFP4-backbone vs FP8 (full vocab): logit cosine {cosine:.5}, base top-1 \
         {base_top1}/{base_total} ({:.1}%), draft agreement {draft_agree}/{} ({:.1}%)",
        100.0 * base_top1_rate,
        fp8_ids.len(),
        100.0 * draft_agree_rate
    );
    ensure!(cosine >= 0.98, "NVFP4 draft logit cosine {cosine:.5} < 0.98");
    ensure!(
        base_top1_rate >= 0.70,
        "NVFP4 draft base top-1 agreement {base_top1_rate:.3} < 0.70"
    );
    ensure!(
        draft_agree_rate >= 0.65,
        "NVFP4 draft token agreement {draft_agree_rate:.3} < 0.65"
    );

    // 2. Fast (shortlist) single-stream vs batched (`seqs = 4`, `m = 32 > 16`, CUDA graph)
    // bit-exact parity for both `nvfp4-backbone` and `nvfp4` (backbone + head).
    for mode in ["nvfp4-backbone", "nvfp4"] {
        let mut fast_opts = DraftOptions::parse("fast", None)?;
        fast_opts.apply_draft_weights(mode)?;
        ensure!(fast_opts.nvfp4_backbone && fast_opts.shortlist.is_some());

        let mut d2 = load_draft(&mut model, &checkpoint, source, capacity, &fast_opts)?;
        let shared = model.draft_shared();
        d2.note_tokens(&stream, shared, &prompt_i32)?;
        d2.commit(&stream, &linear, &dev_taps.slice(..first * row_len), 0)?;
        stream.memcpy_htod(&prompt_i32[first..=first], &mut anchor)?;
        d2.set_round_inputs(&stream, &dev_taps.slice(..0), first, &anchor)?;
        d2.replay_round(&stream, &linear, shared, STEPS)?;
        let single_ids = stream.clone_dtoh(d2.draft_ids())?;

        let seqs = 4usize;
        for slot in 1..seqs {
            d2.copy_slot_rings(&stream, 0, slot)?;
        }
        let slots = [0, 1, 2, 3];
        let anchors_batch = [prompt[first]; 4];
        let samplings = [Sampling::GREEDY; 4];
        d2.batch_draft_initial(&stream, shared, &slots, &anchors_batch, &samplings, STEPS)?;
        let batch_tokens = stream.clone_dtoh(&d2.batch_verify_tokens(seqs, STEPS + 1)?)?;
        for s in 0..seqs {
            let got = &batch_tokens[s * (STEPS + 1) + 1..(s + 1) * (STEPS + 1)];
            assert_eq!(
                got,
                single_ids.as_slice(),
                "batched {mode} slot {s} (m = 32) differs from single-stream (m = 8)"
            );
        }
        eprintln!("DFlash 2 {mode} fast: batched (m = 32) matches single-stream (m = 8) bit-exact");
    }
    Ok(())
}
