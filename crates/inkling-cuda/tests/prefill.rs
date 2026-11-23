//! Prefill against stepping: `Model::prefill` must leave the caches and conv states that
//! stepping the same tokens one at a time leaves, and produce the same logits.
//!
//! Needs the checkpoint and about 20 GB of GPU memory, so it is ignored by default:
//!
//! ```sh
//! INKLING_MODEL=<model-dir> cargo test --release -p inkling-cuda --test prefill -- --ignored
//! ```
//!
//! The prompt is the first 1300 tokens of `reference/out/anyhow-chunked.safetensors`
//! (or `INKLING_CAPTURE`), so it crosses the sliding-window wrap at position 1024.
//!
//! Tolerances. With the decode kernels as the fallbacks of the three heavy operations,
//! every multi-row kernel gives the single-row results bit for bit, so the expected error
//! is zero and the test prints whether it is. Fast multi-row GEMM, expert and attention
//! kernels sum in other orders, so the asserted limits allow rounding-level drift:
//! - each cache and conv state may differ from stepping by a relative RMS error of 2^-7,
//!   one BF16 rounding step;
//! - logits at every compared position may differ by a KL divergence of 4e-3, the quality
//!   gate's mean KL limit and twice MLX's own disagreement between its multi-row and
//!   single-row paths;
//! - the greedy next token after the prompt must match, and top-1 tokens must agree at 97%
//!   of prompt positions, the quality gate's per-range limit.

#![cfg(target_os = "linux")]

use std::{env, fs, path::PathBuf};

use anyhow::{Context as _, ensure};
use inkling::{Checkpoint, TextConfig};
use inkling_cuda::{
    Gpu,
    model::{Model, PrefillLogits, VisitSnapshot},
    weights::Weights,
};
use safetensors::{Dtype, SafeTensors};

const PROMPT: usize = 1300;
const DECODE: usize = 8;
/// Every this many prompt positions, and the last, the full logits are compared.
const SAMPLE_EVERY: usize = 97;
const MAX_STATE_ERROR: f64 = 1.0 / 128.0;
const MAX_KL: f64 = 4e-3;
const MIN_TOP1: f64 = 0.97;

fn tokens() -> anyhow::Result<Vec<u32>> {
    let path = env::var_os("INKLING_CAPTURE").map_or_else(
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
fn compare_states(want: &[VisitSnapshot], got: &[VisitSnapshot]) -> (f64, bool) {
    let mut worst = 0.0f64;
    let mut identical = want.len() == got.len();
    for (w, g) in want.iter().zip(got) {
        let pairs = [(&w.keys, &g.keys), (&w.values, &g.values)]
            .into_iter()
            .chain(w.conv.iter().zip(&g.conv));
        for (a, b) in pairs {
            let (error, same) = relative_error(a, b);
            worst = worst.max(error);
            identical &= same;
        }
    }
    (worst, identical)
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
#[ignore = "needs the checkpoint (INKLING_MODEL) and about 20 GB of GPU memory"]
fn prefill_matches_stepping() -> anyhow::Result<()> {
    let model_dir = PathBuf::from(env::var_os("INKLING_MODEL").context("set INKLING_MODEL")?);
    let tokens = tokens()?;
    ensure!(tokens.len() >= PROMPT + DECODE, "the capture is too short");
    let (prompt, decode) = (&tokens[..PROMPT], &tokens[PROMPT..PROMPT + DECODE]);

    let gpu = Gpu::new(0)?;
    let config = TextConfig::load(&model_dir)?;
    let passes = config.loop_config.num_passes;
    let checkpoint = Checkpoint::open(&model_dir)?;
    let weights = Weights::load(&checkpoint, &config, gpu.stream())?;
    let mut model = Model::new(&gpu, config, weights, passes, 2048)?;

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

    // 512 is the default; 1024 is the largest chunk; 433 leaves a last chunk of one row.
    for chunk in [512, 1024, 433] {
        model.reset()?;
        let mut top1_agree = 0;
        let mut worst_kl = 0.0f64;
        let mut identical = true;
        let mut sample = stepped.sampled.iter().peekable();
        let mut callback = |position: usize, logits: &[f32]| {
            top1_agree += usize::from(argmax(logits) == stepped.top1[position]);
            if let Some((_, want)) = sample.next_if(|(p, _)| *p == position) {
                worst_kl = worst_kl.max(kl(want, logits));
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
        let (state_error, states_identical) =
            compare_states(&stepped.states, &model.cache_snapshot()?);
        let mut decode_kl = 0.0f64;
        for (&token, want) in decode.iter().zip(&stepped.decode) {
            model.step(token, None)?;
            let got = model.logits()?;
            decode_kl = decode_kl.max(kl(want, &got));
            identical &= want
                .iter()
                .zip(&got)
                .all(|(a, b)| a.to_bits() == b.to_bits());
        }
        let top1 = top1_agree as f64 / PROMPT as f64;
        println!(
            "chunk {chunk}: state error {state_error:.2e} (identical: {states_identical}), \
             sampled-logit KL {worst_kl:.2e}, decode KL {decode_kl:.2e}, top-1 {top1_agree}/{PROMPT}, \
             logits identical: {identical}, next {next} vs {}",
            stepped.next
        );
        ensure!(
            next == stepped.next,
            "chunk {chunk}: next token {next}, stepping gave {}",
            stepped.next
        );
        ensure!(
            state_error <= MAX_STATE_ERROR,
            "chunk {chunk}: state error {state_error:.2e}"
        );
        ensure!(
            worst_kl <= MAX_KL && decode_kl <= MAX_KL,
            "chunk {chunk}: KL too large"
        );
        ensure!(top1 >= MIN_TOP1, "chunk {chunk}: top-1 agreement {top1:.4}");
    }
    Ok(())
}
