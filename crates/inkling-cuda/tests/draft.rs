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
use inkling_cuda::{
    Gpu,
    draft::rope_inv_freq,
    ops::draft::{ArgmaxPartials, AttentionScratch, DraftKernels, KvCache, LOGIT_BLOCKS},
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
