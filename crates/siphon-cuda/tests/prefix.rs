//! The prefix cache (`Speculator::prefill_cached`): a prompt that resumes from a cached
//! prefix must give the first token, its logits bit for bit, the generated tokens and the
//! draft's acceptance of a prefill from position 0. A batch that takes slot 0 and a draft
//! rebuilt for another sequence must empty the cache.
//!
//! Needs the checkpoint and about 20 GB of GPU memory, so every test is ignored by
//! default. Each test loads its own model, so run them one at a time:
//!
//! ```sh
//! SIPHON_MODEL=<model-dir> cargo test --release -p siphon-cuda --test prefix -- \
//!     --ignored --nocapture --test-threads 1
//! ```
//!
//! The prompts are slices of the tokens of `reference/out/anyhow-chunked.safetensors` (or
//! `SIPHON_CAPTURE`). `prefix_cache_matches_prefill_dflash` runs the drafter in
//! `SIPHON_DFLASH`, and passes without it.

#![cfg(target_os = "linux")]

use std::{env, fs, path::PathBuf};

use anyhow::{Context as _, ensure};
use safetensors::{Dtype, SafeTensors};
use siphon::{Checkpoint, TextConfig};
use siphon_cuda::{
    Gpu,
    draft::DraftOptions,
    model::{DEFAULT_PREFILL_CHUNK, Model},
    ops::attention::KvFormat,
    spec::{DraftSource, Drafts, Speculator, Width, load_draft},
    weights::{WeightFormat, Weights},
};

const CAPACITY: usize = 12_288;
/// Tokens each run generates after its first.
const STEPS: usize = 96;
const CHUNK: usize = DEFAULT_PREFILL_CHUNK;

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

/// What one request produced.
#[derive(Debug, PartialEq, Eq)]
struct Run {
    /// The first token's logits, hashed.
    logits: u64,
    /// The first token and the generated ones.
    tokens: Vec<u32>,
    /// Verify rounds and the rounds that kept each draft, which follow from the draft's
    /// proposals at a fixed width.
    rounds: usize,
    accepted_at: Vec<usize>,
}

/// Prefills `prompt` into slot 0, through the prefix cache if `cached`, and speculates
/// `STEPS` tokens. Returns the run and the prompt tokens taken from the cache.
fn run(
    model: &mut Model,
    spec: &mut Speculator,
    prompt: &[u32],
    cached: bool,
) -> anyhow::Result<(Run, usize)> {
    let (first, from_cache) = if cached {
        spec.prefill_cached(model, prompt, CHUNK)?
    } else {
        model.reset()?;
        (spec.prefill(model, prompt, CHUNK)?, 0)
    };
    let logits = hash(&model.logits()?);
    let mut tokens = vec![first];
    tokens.extend(spec.generate(model, STEPS, Drafts::Dspark)?);
    tokens.truncate(STEPS + 1);
    let stats = spec.stats();
    Ok((
        Run {
            logits,
            tokens,
            rounds: stats.rounds,
            accepted_at: stats.accepted_at.clone(),
        },
        from_cache,
    ))
}

/// Runs `prompt` through the prefix cache and checks that `cached` of its tokens came
/// from the cache and that the run equals `want`, a run from position 0.
fn check(
    name: &str,
    model: &mut Model,
    spec: &mut Speculator,
    prompt: &[u32],
    want: &Run,
    cached: usize,
) -> anyhow::Result<()> {
    let (got, from_cache) = run(model, spec, prompt, true)?;
    println!(
        "{name}: {} prompt tokens, {from_cache} cached, {} rounds, same as a full prefill: {}",
        prompt.len(),
        got.rounds,
        got == *want
    );
    ensure!(
        from_cache == cached,
        "{name}: {from_cache} tokens came from the cache, expected {cached}"
    );
    ensure!(
        got.logits == want.logits,
        "{name}: the first token's logits differ"
    );
    let differ = got
        .tokens
        .iter()
        .zip(&want.tokens)
        .position(|(g, w)| g != w);
    ensure!(differ.is_none(), "{name}: token {differ:?} differs");
    ensure!(
        got == *want,
        "{name}: the draft's acceptance differs: {got:?} vs {want:?}"
    );
    Ok(())
}

fn prefix_cache_matches_prefill(drafter: Option<&std::path::Path>) -> anyhow::Result<()> {
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
    let source = drafter.map_or(DraftSource::Bundled, DraftSource::Dflash);
    let draft = load_draft(
        &mut model,
        &checkpoint,
        source,
        CAPACITY,
        &DraftOptions::fast(),
    )?;
    let mut spec = Speculator::with_draft(&model, draft, Width::Fixed(7))?;

    let capture = capture_tokens()?;
    ensure!(
        capture.len() >= 9000,
        "the capture has {} tokens",
        capture.len()
    );
    // `a` then `f`, a later turn of the same text; `g` shares `a`'s first 4,500 tokens.
    let a = &capture[..5000];
    let f = &capture[..9000];
    let g: Vec<u32> = capture[..4500]
        .iter()
        .chain(&capture[6000..8000])
        .copied()
        .collect();
    let mut want = Vec::new();
    for prompt in [a, f, &g] {
        want.push(run(&mut model, &mut spec, prompt, false)?.0);
    }
    let [want_a, want_f, want_g] = want.try_into().map_err(|_| anyhow::anyhow!("three runs"))?;

    let bytes = spec.enable_prefix_cache(&model)?;
    println!("prefix cache: {} MiB", bytes >> 20);
    check("a, empty cache", &mut model, &mut spec, a, &want_a, 0)?;
    check("f after a", &mut model, &mut spec, f, &want_f, 4096)?;
    // The cache now holds f below 8,192, which g does not start with.
    check("g after f", &mut model, &mut spec, &g, &want_g, 0)?;
    check("g again", &mut model, &mut spec, &g, &want_g, 6144)?;

    // A batch that continues slot 0's sequence beside another keeps the cache. The tokens
    // fed to slot 0 do not matter here, only that it writes from its position on.
    let mut last = want_g.tokens[0];
    model.reset_slot(1)?;
    let mut other = model.prefill_slot(1, &capture[..700], CHUNK)?;
    for _ in 0..8 {
        let next = model.step_batch(&[(1, other), (0, last)])?;
        (other, last) = (next[0], next[1]);
    }
    check(
        "g after a batch beside it",
        &mut model,
        &mut spec,
        &g,
        &want_g,
        6144,
    )?;

    // A batched request that takes slot 0 empties it.
    model.reset_slot(0)?;
    model.prefill_slot(0, &capture[..700], CHUNK)?;
    check(
        "g after slot 0 took another sequence",
        &mut model,
        &mut spec,
        &g,
        &want_g,
        0,
    )?;

    // So does a draft rebuilt for another sequence.
    check("g again", &mut model, &mut spec, &g, &want_g, 6144)?;
    spec.rebuild(&mut model, 1, &capture[..700])?;
    check("g after a rebuild", &mut model, &mut spec, &g, &want_g, 0)?;
    Ok(())
}

#[test]
#[ignore = "needs the checkpoint and a GPU"]
fn prefix_cache_matches_prefill_bundled() -> anyhow::Result<()> {
    prefix_cache_matches_prefill(None)
}

#[test]
#[ignore = "needs the checkpoint and a GPU"]
fn prefix_cache_matches_prefill_dflash() -> anyhow::Result<()> {
    let Some(dir) = env::var_os("SIPHON_DFLASH") else {
        println!("SIPHON_DFLASH is not set; skipped");
        return Ok(());
    };
    prefix_cache_matches_prefill(Some(std::path::Path::new(&dir)))
}
