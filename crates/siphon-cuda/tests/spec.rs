//! Greedy speculative decoding against greedy decoding through the verify pass.
//!
//! Needs the checkpoint, the three `reference/out/<text>-chunked.safetensors` captures (or
//! the directory in `SIPHON_CAPTURES`) and about 21 GB of GPU memory, so the tests are
//! ignored by default. Each loads its own model, so run them one at a time:
//!
//! ```sh
//! SIPHON_MODEL=<model-dir> cargo test --release -p siphon-cuda --test spec -- \
//!     --ignored --nocapture --test-threads 1
//! ```
//!
//! After each text, both generate 1024 tokens from the same state (`Model::save_state`).
//! - `speculation_is_exact` (test A): speculation with the DSpark draft at k = 1, 3 and 7
//!   and with an adaptive width over every width from 1 to 8 must emit the tokens of
//!   verifying one row at a time, every one. The test also prints the acceptance and how
//!   many leading tokens agree with plain `Model::step` decoding, which sums in other
//!   orders, without a pass criterion.
//! - `random_drafts_change_nothing` (test B): the same with uniformly random drafts at
//!   k = 3 and with widths cycling from 1 to 8; the target almost never accepts them.
//! - `dflash_speculation_is_exact` (test C): test A's widths with the DFlash draft in
//!   `SIPHON_DFLASH`, an export of the 2DSD project (see `DflashWeights`).

#![cfg(target_os = "linux")]

use std::{
    env, fs,
    path::{Path, PathBuf},
};

use anyhow::{Context as _, ensure};
use safetensors::{Dtype, SafeTensors};
use siphon::{Checkpoint, TextConfig};
use siphon_cuda::{
    Gpu,
    draft::DraftOptions,
    model::Model,
    spec::{DraftSource, Drafts, Speculator, Width, WidthPolicy, greedy_by_verify, load_draft},
    weights::Weights,
};

const TEXTS: [&str; 3] = ["anyhow", "half", "regex-readme"];
const GENERATED: usize = 1024;

fn tokens(text: &str) -> anyhow::Result<Vec<u32>> {
    let dir = env::var_os("SIPHON_CAPTURES").map_or_else(
        || PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../reference/out"),
        PathBuf::from,
    );
    let path = dir.join(format!("{text}-chunked.safetensors"));
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

struct Setup {
    _gpu: Gpu,
    checkpoint: Checkpoint,
    model: Model,
    capacity: usize,
}

fn load(prompts: &[Vec<u32>]) -> anyhow::Result<Setup> {
    let model_dir = PathBuf::from(env::var_os("SIPHON_MODEL").context("set SIPHON_MODEL")?);
    let gpu = Gpu::new(0)?;
    let config = TextConfig::load(&model_dir)?;
    let passes = config.loop_config.num_passes;
    let checkpoint = Checkpoint::open(&model_dir)?;
    let weights = Weights::load(&checkpoint, &config, gpu.stream())?;
    let longest = prompts.iter().map(Vec::len).max().unwrap_or(0);
    let capacity = longest + GENERATED + 16;
    let model = Model::new(&gpu, config, weights, passes, capacity)?;
    Ok(Setup {
        _gpu: gpu,
        checkpoint,
        model,
        capacity,
    })
}

/// An adaptive width over every width from 1 to 8, priced so that each can win: a round
/// of `w` rows costs `6 + 0.5 w` ms.
fn every_width() -> anyhow::Result<Width> {
    let round_ms = std::array::from_fn(|i| 6.0 + 0.5 * (i + 1) as f64);
    Ok(Width::Adaptive(WidthPolicy::new(
        &[1, 2, 3, 4, 5, 6, 7, 8],
        round_ms,
    )?))
}

/// Runs every text: the reference, then each `(width, drafts)` from the same state, and
/// checks that each emits the reference's tokens. `dflash` replaces the bundled draft.
fn check(runs: &[(Width, Drafts)], with_step: bool, dflash: Option<&Path>) -> anyhow::Result<()> {
    let prompts = TEXTS
        .iter()
        .map(|t| tokens(t))
        .collect::<anyhow::Result<Vec<_>>>()?;
    let Setup {
        _gpu,
        checkpoint,
        mut model,
        capacity,
    } = load(&prompts)?;
    let mut speculator = match dflash {
        None => Speculator::new(&model, &checkpoint, capacity, 1)?,
        Some(dir) => {
            let draft = load_draft(
                &mut model,
                &checkpoint,
                DraftSource::Dflash(dir),
                capacity,
                &DraftOptions::fast(),
            )?;
            Speculator::with_draft(&model, draft, Width::Fixed(1))?
        }
    };
    let mut failures = Vec::new();
    for (text, prompt) in TEXTS.iter().zip(&prompts) {
        model.reset()?;
        let first = speculator.prefill(&mut model, prompt, 2048)?;
        let saved = model.save_state()?;
        let want = greedy_by_verify(&mut model, first, GENERATED)?;
        if with_step {
            model.restore_state(&saved)?;
            let mut token = first;
            let mut stepped = Vec::with_capacity(GENERATED);
            for _ in 0..GENERATED {
                token = model.step(token, None)?;
                stepped.push(token);
            }
            let agree = want
                .iter()
                .zip(&stepped)
                .take_while(|(a, b)| a == b)
                .count();
            let same = want.iter().zip(&stepped).filter(|(a, b)| a == b).count();
            println!(
                "{text}: verify-path greedy against Model::step: first {agree} of {GENERATED} \
                 agree, {same} positions equal"
            );
        }
        for (width, drafts) in runs {
            let drafts = *drafts;
            model.restore_state(&saved)?;
            speculator.set_width(width.clone())?;
            speculator.restart(&mut model, first)?;
            let got = speculator.generate(&mut model, GENERATED, drafts)?;
            let stats = speculator.stats();
            let agree = want.iter().zip(&got).take_while(|(a, b)| a == b).count();
            let acceptance: Vec<String> = stats
                .acceptance()
                .iter()
                .map(|a| format!("{:.1}%", a * 100.0))
                .collect();
            let name = match width {
                Width::Fixed(k) => format!("k = {k}"),
                Width::Adaptive(_) => "adaptive".to_owned(),
            };
            println!(
                "{text}, {name}, {drafts:?}: {agree} of {GENERATED} tokens match, {} rounds, \
                 {:.3} tokens per round, rounds by width {:?}, acceptance by position [{}]",
                stats.rounds,
                stats.tokens_per_round(),
                stats.widths,
                acceptance.join(", ")
            );
            if agree < GENERATED {
                failures.push(format!(
                    "{text} {name} {drafts:?}: first mismatch at {agree}"
                ));
            }
        }
    }
    ensure!(
        failures.is_empty(),
        "speculation changed the output: {failures:?}"
    );
    Ok(())
}

#[test]
#[ignore = "needs the checkpoint (SIPHON_MODEL), the captures and about 21 GB of GPU memory"]
fn speculation_is_exact() -> anyhow::Result<()> {
    check(
        &[
            (Width::Fixed(1), Drafts::Dspark),
            (Width::Fixed(3), Drafts::Dspark),
            (Width::Fixed(7), Drafts::Dspark),
            (every_width()?, Drafts::Dspark),
        ],
        true,
        None,
    )
}

#[test]
#[ignore = "needs the checkpoint (SIPHON_MODEL), the captures and about 21 GB of GPU memory"]
fn random_drafts_change_nothing() -> anyhow::Result<()> {
    check(
        &[
            (Width::Fixed(3), Drafts::Random(0x5eed)),
            (every_width()?, Drafts::Random(0x5eed)),
        ],
        false,
        None,
    )
}

#[test]
#[ignore = "needs the checkpoint (SIPHON_MODEL), a DFlash draft (SIPHON_DFLASH), the captures \
            and about 22 GB of GPU memory"]
fn dflash_speculation_is_exact() -> anyhow::Result<()> {
    let dir = PathBuf::from(env::var_os("SIPHON_DFLASH").context("set SIPHON_DFLASH")?);
    check(
        &[
            (Width::Fixed(1), Drafts::Dspark),
            (Width::Fixed(3), Drafts::Dspark),
            (Width::Fixed(7), Drafts::Dspark),
            (every_width()?, Drafts::Dspark),
        ],
        false,
        Some(&dir),
    )
}
