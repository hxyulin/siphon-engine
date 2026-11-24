//! The engine's DSpark draft against `reference/dspark.py`, fed the same MLX taps.
//!
//! Needs the checkpoint and a directory holding three files made on the Mac by
//! `reference/capture.py --text reference/prompts/regex-readme.txt --taps` and
//! `reference/dspark.py run` (BF16 and `--float32`):
//!
//! ```sh
//! SIPHON_MODEL=<model-dir> SIPHON_DSPARK_REF=<dir> \
//!     cargo test --release -p siphon-cuda --test draft_round -- --ignored --nocapture
//! ```
//!
//! The text has 2798 tokens, so the local draft layers' window of 1024 wraps. The test
//! commits every position's taps in order and runs a round anchored at every position,
//! with the true token as the anchor. At every anchor it compares the 7 draft ids with the
//! BF16 reference; at the reference's sampled anchors (every 128th and those around 1024)
//! it also compares the decoder output Y, the base logits and the Markov logits.
//!
//! Limits, from the reference's own BF16 against F32 difference (its noise floor):
//! - Y and the base logits: the relative L2 error against the BF16 reference must not
//!   exceed the BF16 reference's own error against the F32 one;
//! - draft ids: among steps whose earlier ids agree, so that both drafts fed the Markov
//!   head the same tokens, at least 99% must agree. MLX's BF16 and F32 drafts agree on
//!   99.1% of such steps on this text. Each disagreement must be a near-tie: the
//!   reference's top-2 Markov logit gap at the first differing step must be at most 0.5,
//!   four BF16 steps at the logits' magnitude.

#![cfg(target_os = "linux")]

use std::{
    env, fs,
    path::{Path, PathBuf},
    time::Instant,
};

use anyhow::{Context as _, ensure};
use half::bf16;
use safetensors::{Dtype, SafeTensors};
use siphon::{Checkpoint, TextConfig};
use siphon_cuda::{
    Gpu,
    draft::{Draft, STEPS},
    ops::linear::Linear,
    weights::draft::{DraftWeights, HIDDEN, SharedTensors, SharedWeights, TAPS},
};

const MIN_PREFIX_AGREEMENT: f64 = 0.99;
const MAX_TIE_MARGIN: f32 = 0.5;
const BLOCK: usize = STEPS + 1;

struct Tensors {
    bytes: Vec<u8>,
}

impl Tensors {
    fn read(path: &Path) -> anyhow::Result<Self> {
        let bytes = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        Ok(Self { bytes })
    }

    /// A tensor widened to F32, from F32 or BF16.
    fn f32(&self, key: &str) -> anyhow::Result<Vec<f32>> {
        let tensors = SafeTensors::deserialize(&self.bytes)?;
        let view = tensors.tensor(key)?;
        let data = view.data();
        Ok(match view.dtype() {
            Dtype::F32 => data
                .as_chunks()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect(),
            Dtype::BF16 => data
                .as_chunks()
                .0
                .iter()
                .map(|b| bf16::from_le_bytes(*b).to_f32())
                .collect(),
            other => anyhow::bail!("{key} is {other:?}"),
        })
    }

    fn bf16(&self, key: &str) -> anyhow::Result<Vec<bf16>> {
        let tensors = SafeTensors::deserialize(&self.bytes)?;
        let view = tensors.tensor(key)?;
        ensure!(view.dtype() == Dtype::BF16, "{key} is {:?}", view.dtype());
        Ok(view
            .data()
            .as_chunks()
            .0
            .iter()
            .map(|b| bf16::from_le_bytes(*b))
            .collect())
    }

    fn i32(&self, key: &str) -> anyhow::Result<Vec<i32>> {
        let tensors = SafeTensors::deserialize(&self.bytes)?;
        let view = tensors.tensor(key)?;
        ensure!(view.dtype() == Dtype::I32, "{key} is {:?}", view.dtype());
        Ok(view
            .data()
            .as_chunks()
            .0
            .iter()
            .map(|b| i32::from_le_bytes(*b))
            .collect())
    }
}

/// Sums for a relative L2 error and the largest absolute difference.
#[derive(Default)]
struct Error {
    diff: f64,
    norm: f64,
    max: f32,
}

impl Error {
    fn add(&mut self, want: &[f32], got: &[f32]) {
        for (&w, &g) in want.iter().zip(got) {
            self.diff += f64::from(w - g).powi(2);
            self.norm += f64::from(w).powi(2);
            self.max = self.max.max((w - g).abs());
        }
    }

    fn relative(&self) -> f64 {
        (self.diff / self.norm).sqrt()
    }
}

/// The engine's errors against the BF16 and F32 references, and the references' own.
#[derive(Default)]
struct Errors {
    engine_bf16: Error,
    engine_f32: Error,
    noise: Error,
}

impl Errors {
    fn add(&mut self, bf16_ref: &[f32], f32_ref: &[f32], got: &[f32]) {
        self.engine_bf16.add(bf16_ref, got);
        self.engine_f32.add(f32_ref, got);
        self.noise.add(f32_ref, bf16_ref);
    }

    fn report(&self, label: &str) {
        eprintln!(
            "{label}: relative L2 against BF16 reference {:.3e} (max |diff| {:.4}), against F32 \
             reference {:.3e} (max {:.4}); reference BF16 against F32 {:.3e} (max {:.4})",
            self.engine_bf16.relative(),
            self.engine_bf16.max,
            self.engine_f32.relative(),
            self.engine_f32.max,
            self.noise.relative(),
            self.noise.max,
        );
    }
}

#[test]
#[ignore = "needs the checkpoint (SIPHON_MODEL) and the reference outputs (SIPHON_DSPARK_REF)"]
#[expect(clippy::too_many_lines, reason = "one end-to-end comparison")]
fn draft_matches_reference() -> anyhow::Result<()> {
    let model_dir = PathBuf::from(env::var_os("SIPHON_MODEL").context("set SIPHON_MODEL")?);
    let ref_dir = PathBuf::from(env::var_os("SIPHON_DSPARK_REF").context("set SIPHON_DSPARK_REF")?);
    let capture = Tensors::read(&ref_dir.join("regex-readme-taps.safetensors"))?;
    let bf16_ref = Tensors::read(&ref_dir.join("regex-readme-dspark-bf16.safetensors"))?;
    let f32_ref = Tensors::read(&ref_dir.join("regex-readme-dspark-f32.safetensors"))?;
    let tokens = capture.i32("tokens")?;
    let taps = capture.bf16("taps")?;
    let n = tokens.len();
    let row_len = TAPS * HIDDEN;
    ensure!(taps.len() == n * row_len, "taps do not match the tokens");
    let ref_ids = bf16_ref.i32("draft_ids")?;
    let f32_ids = f32_ref.i32("draft_ids")?;
    let margins = bf16_ref.f32("markov_margin")?;
    let sampled = bf16_ref.i32("sampled_positions")?;
    let (y_bf16, y_f32) = (bf16_ref.f32("y")?, f32_ref.f32("y")?);
    let (base_bf16, base_f32) = (bf16_ref.f32("base_logits")?, f32_ref.f32("base_logits")?);
    let markov_bf16 = bf16_ref.f32("markov_logits")?;
    let markov_f32 = f32_ref.f32("markov_logits")?;
    ensure!(
        ref_ids.len() == n * STEPS,
        "draft ids do not match the tokens"
    );

    let gpu = Gpu::new(0)?;
    let stream = gpu.stream();
    let config = TextConfig::load(&model_dir)?;
    let vocab = config.unpadded_vocab_size;
    let padded = config.vocab_size;
    let checkpoint = Checkpoint::open(&model_dir)?;
    let linear = Linear::load(gpu.context())?;
    let shared_tensors = SharedTensors::load(&checkpoint, &config, stream)?;
    let shared = SharedWeights::from(&shared_tensors);
    let weights = DraftWeights::load(&checkpoint, &config, stream)?;
    let mut draft = Draft::new(stream, &config, weights, shared, n + BLOCK)?;
    draft.record_logits(stream)?;
    let taps_dev = stream.clone_htod(&taps)?;
    let mut anchor = stream.alloc_zeros::<i32>(1)?;

    let (mut pairs, mut pairs_equal) = (0usize, 0usize);
    let (mut prefix_steps, mut prefix_equal) = (0usize, 0usize);
    let (mut first_steps, mut whole_rounds) = (0usize, 0usize);
    let mut tie_margins = Vec::new();
    let (mut y_err, mut base_err, mut markov_err) =
        (Errors::default(), Errors::default(), Errors::default());
    let mut sample = 0;
    for p in 1..=n - BLOCK {
        let row = taps_dev.slice((p - 1) * row_len..p * row_len);
        draft.commit(stream, &linear, &row, p - 1)?;
        stream.memcpy_htod(&[tokens[p]], &mut anchor)?;
        draft.round(stream, &linear, shared, &anchor)?;
        let ids = stream.clone_dtoh(draft.draft_ids())?;
        let want = &ref_ids[p * STEPS..(p + 1) * STEPS];
        let first_diff = ids.iter().zip(want).position(|(a, b)| a != b);
        pairs += STEPS;
        pairs_equal += ids.iter().zip(want).filter(|(a, b)| a == b).count();
        first_steps += usize::from(ids[0] == want[0]);
        match first_diff {
            None => {
                whole_rounds += 1;
                prefix_steps += STEPS;
                prefix_equal += STEPS;
            }
            Some(step) => {
                prefix_steps += step + 1;
                prefix_equal += step;
                tie_margins.push((p, step, margins[p * STEPS + step]));
            }
        }

        if sampled
            .get(sample)
            .is_some_and(|&s| usize::try_from(s).ok() == Some(p))
        {
            let y = stream.clone_dtoh(draft.y())?;
            let y: Vec<f32> = y.iter().map(|v| v.to_f32()).collect();
            let at = sample * BLOCK * HIDDEN..(sample + 1) * BLOCK * HIDDEN;
            y_err.add(&y_bf16[at.clone()], &y_f32[at], &y);
            let base = stream.clone_dtoh(draft.base_logits())?;
            let logits = stream.clone_dtoh(draft.markov_logits().context("recorded")?)?;
            for step in 0..STEPS {
                let at = (sample * STEPS + step) * vocab..(sample * STEPS + step + 1) * vocab;
                let got: Vec<f32> = base[step * padded..step * padded + vocab]
                    .iter()
                    .map(|v| v.to_f32())
                    .collect();
                base_err.add(&base_bf16[at.clone()], &base_f32[at.clone()], &got);
                // Only steps whose earlier ids agree in all three drafts took the same
                // Markov inputs.
                let f32_want = &f32_ids[p * STEPS..(p + 1) * STEPS];
                let same_prefix = |a: &[i32]| a[..step] == want[..step];
                if same_prefix(&ids) && same_prefix(f32_want) {
                    let got: Vec<f32> = logits[step * vocab..(step + 1) * vocab]
                        .iter()
                        .map(|v| v.to_f32())
                        .collect();
                    markov_err.add(&markov_bf16[at.clone()], &markov_f32[at], &got);
                }
            }
            sample += 1;
        }
    }
    ensure!(
        sample == sampled.len(),
        "visited {sample} of {} sampled anchors",
        sampled.len()
    );

    let anchors = n - BLOCK;
    let ratio = |a: usize, b: usize| {
        #[expect(clippy::cast_precision_loss, reason = "counts are far below 2^52")]
        let r = a as f64 / b as f64;
        r
    };
    eprintln!(
        "draft ids over {anchors} anchors: {:.3}% of anchor-step pairs, first step {:.3}%, steps \
         with an equal prefix {:.3}% of {prefix_steps}, whole rounds {:.2}%",
        100.0 * ratio(pairs_equal, pairs),
        100.0 * ratio(first_steps, anchors),
        100.0 * ratio(prefix_equal, prefix_steps),
        100.0 * ratio(whole_rounds, anchors),
    );
    let mut sorted: Vec<f32> = tie_margins.iter().map(|&(_, _, m)| m).collect();
    sorted.sort_by(f32::total_cmp);
    if let Some(&max) = sorted.last() {
        let exact = sorted.iter().filter(|&&m| m == 0.0).count();
        eprintln!(
            "{} disagreements; reference top-2 margin at the first differing step: median \
             {:.4}, max {max:.4}, exact ties {exact}",
            sorted.len(),
            sorted[sorted.len() / 2],
        );
    }
    y_err.report("Y");
    base_err.report("base logits");
    markov_err.report("Markov logits, steps whose earlier ids agree in all three");

    // Time per round as it stands: commit one position, then a round, at the last anchor.
    let p = n - BLOCK;
    let mut best = f64::INFINITY;
    for _ in 0..5 {
        stream.synchronize()?;
        let start = Instant::now();
        for _ in 0..20 {
            let row = taps_dev.slice((p - 1) * row_len..p * row_len);
            draft.commit(stream, &linear, &row, p - 1)?;
            draft.round(stream, &linear, shared, &anchor)?;
        }
        stream.synchronize()?;
        best = best.min(start.elapsed().as_secs_f64() / 20.0);
    }
    eprintln!(
        "commit of one position and a round at position {p}: {:.3} ms",
        best * 1e3
    );

    ensure!(
        y_err.engine_bf16.relative() <= y_err.noise.relative(),
        "Y error exceeds the reference's noise"
    );
    ensure!(
        base_err.engine_bf16.relative() <= base_err.noise.relative(),
        "base logit error exceeds the reference's noise"
    );
    ensure!(
        ratio(prefix_equal, prefix_steps) >= MIN_PREFIX_AGREEMENT,
        "draft ids agree on too few steps"
    );
    for &(p, step, margin) in &tie_margins {
        ensure!(
            margin <= MAX_TIE_MARGIN,
            "anchor {p} step {step}: ids differ with a reference margin of {margin}"
        );
    }
    Ok(())
}
