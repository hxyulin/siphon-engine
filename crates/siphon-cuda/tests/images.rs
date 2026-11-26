//! Image rows in prefill (`Model::prefill_with_images`) and in the prefix cache's keys.
//!
//! An image's rows replace the normed embeddings of its placeholder tokens. These tests
//! stand text in for an image: the image's rows are the normed embeddings of some tokens
//! (`Model::embedding_rows`), so a prompt with placeholders and those rows must give what
//! the prompt with the tokens gives, bit for bit, wherever the chunks cut the image.
//!
//! Needs the checkpoint and about 20 GB of GPU memory, so every test is ignored:
//!
//! ```sh
//! SIPHON_MODEL=<model-dir> cargo test --release -p siphon-cuda --test images -- \
//!     --ignored --nocapture --test-threads 1
//! ```
//!
//! The prompts are slices of the tokens of `reference/out/anyhow-chunked.safetensors` (or
//! `SIPHON_CAPTURE`).

#![cfg(target_os = "linux")]

use std::{env, fs, ops::Range, path::PathBuf, sync::Arc, time::Duration};

use anyhow::{Context as _, ensure};
use safetensors::{Dtype, SafeTensors};
use siphon::{Checkpoint, TextConfig};
use siphon_cuda::{
    Gpu,
    draft::DraftOptions,
    model::{DEFAULT_PREFILL_CHUNK, ImageRows, Model, PrefillLogits},
    ops::attention::KvFormat,
    spec::{DraftSource, Drafts, Prompt, Speculator, Width, load_draft},
    weights::{WeightFormat, Weights},
};

const CAPACITY: usize = 8192;
const CHUNK: usize = DEFAULT_PREFILL_CHUNK;
const STEPS: usize = 48;
/// The image placeholder, `<|unused_200054|>`.
const PLACEHOLDER: u32 = 200_054;
/// The prompt's length: its last chunk boundary is 4,096.
const PROMPT: usize = 5000;
/// Image runs: one across the chunk boundary at 2,048, one of a single placeholder, and
/// one across the boundary at 4,096.
const RUNS: [Range<usize>; 3] = [1900..2300, 3000..3001, 4090..4100];

fn capture_tokens() -> anyhow::Result<Vec<u32>> {
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

/// FNV-1a over the bits of a row of logits.
fn hash(row: &[f32]) -> u64 {
    row.iter().fold(0xcbf2_9ce4_8422_2325, |h, v| {
        (h ^ u64::from(v.to_bits())).wrapping_mul(0x0100_0000_01b3)
    })
}

fn load() -> anyhow::Result<(Gpu, Model, Speculator)> {
    let model_dir = PathBuf::from(env::var_os("SIPHON_MODEL").context("set SIPHON_MODEL")?);
    let gpu = Gpu::new(0)?;
    let config = TextConfig::load(&model_dir)?;
    let passes = config.loop_config.num_passes;
    let checkpoint = Checkpoint::open(&model_dir)?;
    let weights = Weights::load(&checkpoint, &config, gpu.stream())?;
    let mut model = Model::with_formats(
        &gpu,
        config,
        weights,
        passes,
        CAPACITY,
        KvFormat::Fp8,
        WeightFormat::Fp8,
    )?;
    model.add_slots(1)?;
    let draft = load_draft(
        &mut model,
        &checkpoint,
        DraftSource::Bundled,
        CAPACITY,
        &DraftOptions::fast(),
    )?;
    let spec = Speculator::with_draft(&model, draft, Width::Fixed(7))?;
    Ok((gpu, model, spec))
}

/// A prompt with images: its placeholder tokens, the rows of each image, and the same
/// prompt with each image's stand-in tokens, whose prefill it must equal.
struct ImagePrompt {
    tokens: Vec<u32>,
    text: Vec<u32>,
    images: Vec<ImageRows>,
}

impl ImagePrompt {
    /// `base` with the runs of `RUNS` as images that stand in for `stand_in[run]`.
    fn new(model: &Model, base: &[u32], stand_in: &[u32]) -> anyhow::Result<Self> {
        let mut tokens = base.to_vec();
        let mut text = base.to_vec();
        let mut images = Vec::new();
        for run in RUNS {
            tokens[run.clone()].fill(PLACEHOLDER);
            text[run.clone()].copy_from_slice(&stand_in[run.clone()]);
            images.push(ImageRows {
                position: run.start,
                rows: Arc::new(model.embedding_rows(&stand_in[run.clone()])?),
            });
        }
        Ok(Self {
            tokens,
            text,
            images,
        })
    }

    /// The images overlapping `start..end`, and a note of the chunk asked for.
    fn source(&self, asked: &mut Vec<(usize, usize)>, start: usize, end: usize) -> Vec<ImageRows> {
        asked.push((start, end));
        self.images
            .iter()
            .zip(RUNS)
            .filter(|(_, run)| run.start < end && run.end > start)
            .map(|(image, _)| image.clone())
            .collect()
    }

    /// The prompt's cache key: the tokens with each image's placeholders replaced by ids
    /// above every token, from `seed`.
    fn key(&self, seed: u32) -> Vec<u32> {
        let mut key = self.tokens.clone();
        for run in RUNS {
            for (k, id) in key[run].iter_mut().enumerate() {
                *id = 0x8000_0000 | (seed.wrapping_mul(0x9e37_79b9) ^ k as u32);
            }
        }
        key
    }
}

#[derive(Debug, PartialEq, Eq)]
struct Run {
    logits: u64,
    tokens: Vec<u32>,
    rounds: usize,
    accepted_at: Vec<usize>,
}

fn generate(model: &mut Model, spec: &mut Speculator, first: u32) -> anyhow::Result<Run> {
    let logits = hash(&model.logits()?);
    let mut tokens = vec![first];
    tokens.extend(spec.generate(model, STEPS, Drafts::Dspark)?);
    tokens.truncate(STEPS + 1);
    let stats = spec.stats();
    Ok(Run {
        logits,
        tokens,
        rounds: stats.rounds,
        accepted_at: stats.accepted_at.clone(),
    })
}

/// Speculates after a prefill of `text` from position 0.
fn text_run(model: &mut Model, spec: &mut Speculator, text: &[u32]) -> anyhow::Result<Run> {
    model.reset()?;
    let first = spec.prefill(model, text, CHUNK)?;
    generate(model, spec, first)
}

/// Speculates after `prompt` with `key` through the prefix cache; returns the run and the
/// tokens that came from the cache.
fn cached_run(
    model: &mut Model,
    spec: &mut Speculator,
    prompt: &ImagePrompt,
    key: &[u32],
) -> anyhow::Result<(Run, usize)> {
    let mut asked = Vec::new();
    let mut source = |start: usize, end: usize| Ok(prompt.source(&mut asked, start, end));
    let (first, cached) = spec.prefill_cached(
        model,
        Prompt {
            tokens: &prompt.tokens,
            key,
            images: Some(&mut source),
        },
        CHUNK,
    )?;
    Ok((generate(model, spec, first)?, cached))
}

#[test]
#[ignore = "needs the checkpoint and a GPU"]
fn image_rows_stand_in_for_text() -> anyhow::Result<()> {
    let (_gpu, mut model, mut spec) = load()?;
    let capture = capture_tokens()?;
    ensure!(capture.len() >= 2 * PROMPT, "the capture is too short");
    let base = &capture[..PROMPT];
    let prompt = ImagePrompt::new(&model, base, &capture[PROMPT..2 * PROMPT])?;

    // The model alone: whole, and resumed at the chunk boundary inside the first image.
    model.reset()?;
    let want = model.prefill(&prompt.text, CHUNK, PrefillLogits::Last)?;
    let want_logits = hash(&model.logits()?);
    for split in [None, Some(CHUNK)] {
        model.reset()?;
        let mut asked = Vec::new();
        let mut source = |start: usize, end: usize| Ok(prompt.source(&mut asked, start, end));
        let pieces: Vec<&[u32]> = match split {
            None => vec![&prompt.tokens[..]],
            Some(at) => vec![&prompt.tokens[..at], &prompt.tokens[at..]],
        };
        let mut got = 0;
        for piece in pieces {
            got =
                model.prefill_with_images(piece, CHUNK, PrefillLogits::Last, None, &mut source)?;
        }
        let logits = hash(&model.logits()?);
        println!(
            "split at {split:?}: chunks asked {asked:?}, same logits: {}",
            logits == want_logits
        );
        ensure!(
            asked == [(0, 2048), (2048, 4096), (4096, PROMPT)],
            "prefill asked for {asked:?}"
        );
        ensure!(
            got == want && logits == want_logits,
            "split at {split:?}: logits differ"
        );
    }

    // Negative control: other rows for the first image change the logits.
    let other = ImagePrompt::new(&model, base, &capture[1000..1000 + PROMPT])?;
    model.reset()?;
    let mut asked = Vec::new();
    let mut source = |start: usize, end: usize| Ok(other.source(&mut asked, start, end));
    model.prefill_with_images(
        &prompt.tokens,
        CHUNK,
        PrefillLogits::Last,
        None,
        &mut source,
    )?;
    ensure!(
        hash(&model.logits()?) != want_logits,
        "other image rows left the logits unchanged"
    );

    // With the draft: the taps at image positions come from the image rows, so speculation
    // matches the text prompt's, tokens and acceptance.
    let want_run = text_run(&mut model, &mut spec, &prompt.text)?;
    model.reset()?;
    let (got_run, cached) = cached_run(&mut model, &mut spec, &prompt, &prompt.key(1))?;
    println!(
        "speculation: {} rounds, same as the text prompt: {}",
        got_run.rounds,
        got_run == want_run
    );
    ensure!(cached == 0, "{cached} tokens came from a disabled cache");
    ensure!(
        got_run == want_run,
        "speculation differs: {got_run:?} vs {want_run:?}"
    );
    Ok(())
}

#[test]
#[ignore = "needs the checkpoint and a GPU"]
fn prompts_that_differ_only_in_an_image_do_not_share_an_entry() -> anyhow::Result<()> {
    let (_gpu, mut model, mut spec) = load()?;
    let capture = capture_tokens()?;
    ensure!(capture.len() >= 2 * PROMPT, "the capture is too short");
    let base = &capture[..PROMPT];
    let x = ImagePrompt::new(&model, base, &capture[PROMPT..2 * PROMPT])?;
    let y = ImagePrompt::new(&model, base, &capture[3000..3000 + PROMPT])?;
    ensure!(
        x.tokens == y.tokens,
        "the prompts differ in more than their images"
    );
    let (key_x, key_y) = (x.key(1), x.key(2));
    let want_x = text_run(&mut model, &mut spec, &x.text)?;
    let want_y = text_run(&mut model, &mut spec, &y.text)?;
    ensure!(
        want_x.logits != want_y.logits,
        "the stand-in texts give the same logits"
    );

    spec.enable_prefix_cache(&model, 1, Duration::from_secs(600))?;
    let steps = [
        ("x, cold", &x, &key_x, 0, &want_x),
        ("x again", &x, &key_x, 4096, &want_x),
        ("y, same tokens, other image", &y, &key_y, 0, &want_y),
        ("y again", &y, &key_y, 4096, &want_y),
        ("x after y", &x, &key_x, 0, &want_x),
    ];
    for (name, prompt, key, want_cached, want) in steps {
        let (got, cached) = cached_run(&mut model, &mut spec, prompt, key)?;
        println!(
            "{name}: {cached} cached, same as a cold prefill: {}",
            got == *want
        );
        ensure!(
            cached == want_cached,
            "{name}: {cached} tokens came from the cache"
        );
        ensure!(got == *want, "{name}: the run differs from a cold prefill");
    }

    // Negative control: y under x's key takes x's entry, and with it x's image.
    let (got, cached) = cached_run(&mut model, &mut spec, &y, &key_x)?;
    println!(
        "y under x's key: {cached} cached, same as y: {}",
        got == want_y
    );
    ensure!(
        cached == 4096 && got.logits != want_y.logits,
        "y under x's key should hit x's entry and differ"
    );
    Ok(())
}
