//! Greedy speculative decoding against greedy decoding through the verify pass.
//!
//! Needs the checkpoint, the three `reference/out/<text>-chunked.safetensors` captures (or
//! the directory in `INKLING_CAPTURES`) and about 21 GB of GPU memory, so the tests are
//! ignored by default. Each loads its own model, so run them one at a time:
//!
//! ```sh
//! INKLING_MODEL=<model-dir> cargo test --release -p inkling-cuda --test spec -- \
//!     --ignored --nocapture --test-threads 1
//! ```
//!
//! After each text, both generate 1024 tokens from the same state (`Model::save_state`).
//! - `speculation_is_exact` (test A): speculation with the DSpark draft at k = 1, 3 and 7
//!   must emit the tokens of verifying one row at a time, every one. The test also prints
//!   the acceptance and how many leading tokens agree with plain `Model::step` decoding,
//!   which sums in other orders, without a pass criterion.
//! - `random_drafts_change_nothing` (test B): the same with uniformly random drafts at
//!   k = 3, which the target almost never accepts.

#![cfg(target_os = "linux")]

use std::{env, fs, path::PathBuf};

use anyhow::{Context as _, ensure};
use inkling::{Checkpoint, TextConfig};
use inkling_cuda::{
    Gpu,
    model::Model,
    spec::{Drafts, Speculator, greedy_by_verify},
    weights::Weights,
};
use safetensors::{Dtype, SafeTensors};

const TEXTS: [&str; 3] = ["anyhow", "half", "regex-readme"];
const GENERATED: usize = 1024;

fn tokens(text: &str) -> anyhow::Result<Vec<u32>> {
    let dir = env::var_os("INKLING_CAPTURES").map_or_else(
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
    let model_dir = PathBuf::from(env::var_os("INKLING_MODEL").context("set INKLING_MODEL")?);
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

/// Runs every text: the reference, then each `(k, drafts)` from the same state, and checks
/// that each emits the reference's tokens.
fn check(runs: &[(usize, Drafts)], with_step: bool) -> anyhow::Result<()> {
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
    let mut speculator = Speculator::new(&model, &checkpoint, capacity, runs[0].0)?;
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
        for &(k, drafts) in runs {
            model.restore_state(&saved)?;
            speculator.set_k(k)?;
            speculator.restart(&mut model, first)?;
            let got = speculator.generate(&mut model, GENERATED, drafts)?;
            let stats = speculator.stats();
            let agree = want.iter().zip(&got).take_while(|(a, b)| a == b).count();
            let acceptance: Vec<String> = stats
                .acceptance()
                .iter()
                .map(|a| format!("{:.1}%", a * 100.0))
                .collect();
            println!(
                "{text}, k = {k}, {drafts:?}: {agree} of {GENERATED} tokens match, {} rounds, \
                 {:.3} tokens per round, acceptance by position [{}]",
                stats.rounds,
                stats.tokens_per_round(),
                acceptance.join(", ")
            );
            if agree < GENERATED {
                failures.push(format!(
                    "{text} k = {k} {drafts:?}: first mismatch at {agree}"
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
#[ignore = "needs the checkpoint (INKLING_MODEL), the captures and about 21 GB of GPU memory"]
fn speculation_is_exact() -> anyhow::Result<()> {
    check(
        &[
            (1, Drafts::Dspark),
            (3, Drafts::Dspark),
            (7, Drafts::Dspark),
        ],
        true,
    )
}

#[test]
#[ignore = "needs the checkpoint (INKLING_MODEL), the captures and about 21 GB of GPU memory"]
fn random_drafts_change_nothing() -> anyhow::Result<()> {
    check(&[(3, Drafts::Random(0x5eed))], false)
}
