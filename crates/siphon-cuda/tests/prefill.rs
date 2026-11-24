//! Prefill against stepping: `Model::prefill` must leave the caches and conv states that
//! stepping the same tokens one at a time leaves, and produce the same logits.
//!
//! Needs the checkpoint and about 20 GB of GPU memory, so it is ignored by default:
//!
//! ```sh
//! SIPHON_MODEL=<model-dir> cargo test --release -p siphon-cuda --test prefill -- --ignored
//! ```
//!
//! The prompt is the first 1300 tokens of `reference/out/anyhow-chunked.safetensors`
//! (or `SIPHON_CAPTURE`), so it crosses the sliding-window wrap at position 1024.
//!
//! Tolerances. With the decode kernels as the fallbacks of the three heavy operations,
//! every multi-row kernel gives the single-row results bit for bit, so the expected error
//! is zero and the test prints whether it is. Fast multi-row GEMM, expert and attention
//! kernels sum in other orders. The 48 layer visits amplify such rounding: with the
//! tensor-core GEMM, the relative state error grows smoothly from 1e-4 at the first visit
//! to about 0.2 in the third pass of the loop, while the logits stay within MLX's own
//! multi-row against single-row disagreement. So the asserted limits are:
//! - the first visit's caches and conv states may differ from stepping by a relative RMS
//!   error of 2^-7, one BF16 rounding step. This catches append, ring and conv bugs before
//!   the loop amplifies anything. Later visits are printed, not asserted;
//! - over the sampled prompt positions and the decode steps, the quality gate's KL limits:
//!   mean at most 4e-3, max at most 0.12;
//! - the greedy next token after the prompt must match, and top-1 tokens must agree at 97%
//!   of prompt positions, the quality gate's per-range limit.
//!
//! `fp8_prefill_matches_stepping` runs the same checks with the FP8 cache and weight copies,
//! and every FP8 prefill product on: attention (the default), expert activations and dense
//! products. Stepping keeps its own kernels, so the FP8 products' rounding adds to the
//! differences. The first visit's keys and values then come from projections of MXFP8
//! inputs (relative rounding up to 2^-4 per input), so its state limit is 2^-4; the KL and
//! top-1 limits stay.

#![cfg(target_os = "linux")]

use std::{env, fs, path::PathBuf};

use anyhow::{Context as _, ensure};
use safetensors::{Dtype, SafeTensors};
use siphon::{Checkpoint, TextConfig};
use siphon_cuda::{
    Gpu,
    model::{Model, PrefillLogits, VisitSnapshot},
    ops::{Products, attention::KvFormat, experts::ExpertActivations},
    weights::{WeightFormat, Weights},
};

const PROMPT: usize = 1300;
const DECODE: usize = 8;
/// Every this many prompt positions, and the last, the full logits are compared.
const SAMPLE_EVERY: usize = 97;
const MAX_STATE_ERROR: f64 = 1.0 / 128.0;
/// The first-visit state limit with FP8 prefill products: the E4M3 rounding of the
/// projections' inputs.
const MAX_FP8_STATE_ERROR: f64 = 1.0 / 16.0;
const MAX_MEAN_KL: f64 = 4e-3;
const MAX_KL: f64 = 0.12;
const MIN_TOP1: f64 = 0.97;

fn tokens() -> anyhow::Result<Vec<u32>> {
    let path = env::var_os("SIPHON_CAPTURE").map_or_else(
        || {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../reference/out/anyhow-chunked.safetensors")
        },
        PathBuf::from,
    );
    let bytes = fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
    let capture = SafeTensors::deserialize(&bytes)?;
    let view = capture.tensor("tokens")?;
    ensure!(view.dtype() == Dtype::I32, "tokens are {:?}", view.dtype());
    view.data()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| Ok(u32::try_from(i32::from_le_bytes(*b))?))
        .collect()
}

fn argmax(x: &[f32]) -> usize {
    let mut best = 0;
    for (i, &v) in x.iter().enumerate() {
        if v > x[best] || (v.is_nan() && !x[best].is_nan()) {
            best = i;
        }
    }
    best
}

/// KL(want || got) over softmax of the logits.
fn kl(want: &[f32], got: &[f32]) -> f64 {
    let log_softmax = |x: &[f32]| {
        let max = x.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let sum: f64 = x.iter().map(|&v| f64::from(v - max).exp()).sum();
        let log_sum = sum.ln();
        x.iter()
            .map(|&v| f64::from(v - max) - log_sum)
            .collect::<Vec<_>>()
    };
    let (p, q) = (log_softmax(want), log_softmax(got));
    p.iter().zip(&q).map(|(lp, lq)| lp.exp() * (lp - lq)).sum()
}

/// Relative RMS error of `got` against `want`, and whether the two are bit-identical.
fn relative_error(want: &[f32], got: &[f32]) -> (f64, bool) {
    let (mut diff, mut norm) = (0.0, 0.0);
    let mut identical = want.len() == got.len();
    for (&w, &g) in want.iter().zip(got) {
        diff += f64::from(w - g).powi(2);
        norm += f64::from(w).powi(2);
        identical &= w.to_bits() == g.to_bits();
    }
    ((diff / norm.max(f64::MIN_POSITIVE)).sqrt(), identical)
}

/// The largest relative error over every visit's buffers, and whether all are identical.
/// The worst relative error over each visit's caches and conv states, and whether every
/// state is bit-identical.
fn compare_states(want: &[VisitSnapshot], got: &[VisitSnapshot]) -> (Vec<f64>, bool) {
    let mut identical = want.len() == got.len();
    let errors = want
        .iter()
        .zip(got)
        .map(|(w, g)| {
            let pairs = [(&w.keys, &g.keys), (&w.values, &g.values)]
                .into_iter()
                .chain(w.conv.iter().zip(&g.conv));
            let mut worst = 0.0f64;
            for (a, b) in pairs {
                let (error, same) = relative_error(a, b);
                worst = worst.max(error);
                identical &= same;
            }
            worst
        })
        .collect();
    (errors, identical)
}

struct Stepped {
    /// Top-1 token at each prompt position.
    top1: Vec<usize>,
    /// (position, logits) at the sampled prompt positions.
    sampled: Vec<(usize, Vec<f32>)>,
    next: u32,
    states: Vec<VisitSnapshot>,
    /// Logits of each decode step after the prompt.
    decode: Vec<Vec<f32>>,
}

#[test]
#[ignore = "needs the checkpoint (SIPHON_MODEL) and about 20 GB of GPU memory"]
fn prefill_matches_stepping() -> anyhow::Result<()> {
    prefill_matches_stepping_with(KvFormat::Bf16, WeightFormat::Bf16, false)
}

#[test]
#[ignore = "needs the checkpoint (SIPHON_MODEL) and about 20 GB of GPU memory"]
fn fp8_prefill_matches_stepping() -> anyhow::Result<()> {
    prefill_matches_stepping_with(KvFormat::Fp8, WeightFormat::Fp8, true)
}

/// The checkpoint's model with the given cache and weight formats, and with
/// `fp8_products` every FP8 prefill product on.
fn load(
    gpu: &Gpu,
    kv: KvFormat,
    weight_format: WeightFormat,
    fp8_products: bool,
) -> anyhow::Result<Model> {
    let model_dir = PathBuf::from(env::var_os("SIPHON_MODEL").context("set SIPHON_MODEL")?);
    let config = TextConfig::load(&model_dir)?;
    let passes = config.loop_config.num_passes;
    let checkpoint = Checkpoint::open(&model_dir)?;
    let weights = Weights::load(&checkpoint, &config, gpu.stream())?;
    let mut model = Model::with_formats(gpu, config, weights, passes, 2048, kv, weight_format)?;
    if fp8_products {
        model.set_prefill_attention(Products::Fp8);
        model.set_expert_activations(ExpertActivations::Fp8)?;
        model.set_prefill_dense(Products::Fp8)?;
    }
    Ok(model)
}

fn prefill_matches_stepping_with(
    kv: KvFormat,
    weight_format: WeightFormat,
    fp8_products: bool,
) -> anyhow::Result<()> {
    let tokens = tokens()?;
    ensure!(tokens.len() >= PROMPT + DECODE, "the capture is too short");
    let (prompt, decode) = (&tokens[..PROMPT], &tokens[PROMPT..PROMPT + DECODE]);
    let gpu = Gpu::new(0)?;
    let mut model = load(&gpu, kv, weight_format, fp8_products)?;

    let sampled = |i: usize| i.is_multiple_of(SAMPLE_EVERY) || i == PROMPT - 1;
    let mut stepped = Stepped {
        top1: Vec::new(),
        sampled: Vec::new(),
        next: 0,
        states: Vec::new(),
        decode: Vec::new(),
    };
    for (i, &token) in prompt.iter().enumerate() {
        stepped.next = model.step(token, None)?;
        stepped.top1.push(stepped.next as usize);
        if sampled(i) {
            stepped.sampled.push((i, model.logits()?));
        }
    }
    stepped.states = model.cache_snapshot()?;
    for &token in decode {
        model.step(token, None)?;
        stepped.decode.push(model.logits()?);
    }

    // 512 is the default; 433 leaves a last chunk of one row; 2048 is the largest chunk and
    // runs the prompt as one chunk longer than the sliding window.
    for chunk in [512, 1024, 433, 2048] {
        model.reset()?;
        let mut top1_agree = 0;
        let mut kls = Vec::new();
        let mut identical = true;
        let mut sample = stepped.sampled.iter().peekable();
        let mut callback = |position: usize, logits: &[f32]| {
            top1_agree += usize::from(argmax(logits) == stepped.top1[position]);
            if let Some((_, want)) = sample.next_if(|(p, _)| *p == position) {
                kls.push(kl(want, logits));
                identical &= want
                    .iter()
                    .zip(logits)
                    .all(|(a, b)| a.to_bits() == b.to_bits());
            }
            Ok(())
        };
        let next = model.prefill(prompt, chunk, PrefillLogits::All(&mut callback))?;
        ensure!(
            model.position() == PROMPT,
            "prefill ended at {}",
            model.position()
        );
        let (state_errors, states_identical) =
            compare_states(&stepped.states, &model.cache_snapshot()?);
        let first_visit_error = state_errors[0];
        let worst_state_error = state_errors.iter().copied().fold(0.0, f64::max);
        for (&token, want) in decode.iter().zip(&stepped.decode) {
            model.step(token, None)?;
            let got = model.logits()?;
            kls.push(kl(want, &got));
            identical &= want
                .iter()
                .zip(&got)
                .all(|(a, b)| a.to_bits() == b.to_bits());
        }
        let top1 = top1_agree as f64 / PROMPT as f64;
        let mean_kl = kls.iter().sum::<f64>() / kls.len() as f64;
        let max_kl = kls.iter().copied().fold(0.0, f64::max);
        println!(
            "chunk {chunk}: state error first visit {first_visit_error:.2e}, worst visit \
             {worst_state_error:.2e} (identical: {states_identical}), KL mean {mean_kl:.2e} max \
             {max_kl:.2e}, top-1 {top1_agree}/{PROMPT}, logits identical: {identical}, next \
             {next} vs {}",
            stepped.next
        );
        ensure!(
            next == stepped.next,
            "chunk {chunk}: next token {next}, stepping gave {}",
            stepped.next
        );
        let max_state_error = if fp8_products {
            MAX_FP8_STATE_ERROR
        } else {
            MAX_STATE_ERROR
        };
        ensure!(
            first_visit_error <= max_state_error,
            "chunk {chunk}: first-visit state error {first_visit_error:.2e}"
        );
        ensure!(
            mean_kl <= MAX_MEAN_KL && max_kl <= MAX_KL,
            "chunk {chunk}: KL mean {mean_kl:.2e} max {max_kl:.2e}"
        );
        ensure!(top1 >= MIN_TOP1, "chunk {chunk}: top-1 agreement {top1:.4}");
    }
    Ok(())
}
