//! Run-to-run determinism: the same tokens, run twice, must give bit-identical logits and
//! caches, both within one process and across two processes.
//!
//! Needs the checkpoint and about 20 GB of GPU memory, so it is ignored by default:
//!
//! ```sh
//! INKLING_MODEL=<model-dir> cargo test --release -p inkling-cuda --test determinism -- --ignored
//! ```
//!
//! The tokens are the first 1300 of `reference/out/anyhow-chunked.safetensors` (or
//! `INKLING_CAPTURE`). Each run covers decode and prefill under two configurations, the
//! default one and FP8 weights, FP8 global caches and quantized expert activations:
//! - decode: from an empty cache, step the first 256 tokens one at a time, keeping every
//!   step's logits, then snapshot the caches and conv states;
//! - prefill: prefill all 1300 tokens, crossing the sliding window wrap at 1024, in chunks
//!   of 512 and again in one chunk, keeping every row's logits; snapshot after each, and
//!   after the second step 32 more tokens.
//!
//! The test first runs the child process (this test binary again, with
//! `INKLING_DETERMINISM_CHILD` set), which prints one digest per configuration, then runs
//! everything twice itself and compares. The child must finish first: two models do not
//! fit the GPU at once.

#![cfg(target_os = "linux")]

use std::{env, fs, path::PathBuf, process::Command};

use anyhow::{Context as _, ensure};
use inkling::{Checkpoint, TextConfig};
use inkling_cuda::{
    Gpu,
    model::{Model, PrefillLogits, VisitSnapshot},
    ops::{attention::KvFormat, experts::ExpertActivations},
    weights::{WeightFormat, Weights},
};
use safetensors::{Dtype, SafeTensors};

const PROMPT: usize = 1300;
const STEPS: usize = 256;
const DECODE_AFTER_PREFILL: usize = 32;
/// 512 is the default chunk; 2048 runs the prompt as one chunk longer than the window.
const CHUNKS: [usize; 2] = [512, 2048];
const CHILD_ENV: &str = "INKLING_DETERMINISM_CHILD";
const DIGEST_PREFIX: &str = "determinism digest";

#[derive(Clone, Copy, Debug)]
struct Config {
    name: &'static str,
    weights: WeightFormat,
    kv: KvFormat,
    experts: ExpertActivations,
}

const CONFIGS: [Config; 2] = [
    Config {
        name: "bf16",
        weights: WeightFormat::Bf16,
        kv: KvFormat::Bf16,
        experts: ExpertActivations::Bf16,
    },
    Config {
        name: "fp8",
        weights: WeightFormat::Fp8,
        kv: KvFormat::Fp8,
        experts: ExpertActivations::Quantized,
    },
];

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

/// FNV-1a over the bit patterns of `values`, continuing from `hash`.
fn fnv(hash: u64, values: &[f32]) -> u64 {
    values.iter().fold(hash, |h, v| {
        v.to_bits()
            .to_le_bytes()
            .iter()
            .fold(h, |h, &b| (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3))
    })
}

const FNV_START: u64 = 0xcbf2_9ce4_8422_2325;

fn hash(values: &[f32]) -> u64 {
    fnv(FNV_START, values)
}

/// One hash per visit over its caches and conv states.
fn snapshot_hashes(snapshot: &[VisitSnapshot]) -> Vec<u64> {
    snapshot
        .iter()
        .map(|v| {
            let h = fnv(fnv(FNV_START, &v.keys), &v.values);
            v.conv.iter().fold(h, |h, c| fnv(h, c))
        })
        .collect()
}

/// Hashes of everything one run produces.
#[derive(PartialEq, Eq)]
struct Run {
    /// Logits of each decode step from an empty cache.
    steps: Vec<u64>,
    /// Each visit's state after those steps.
    step_states: Vec<u64>,
    /// Logits of each prefill row, for each chunk size in turn.
    prefill: Vec<u64>,
    /// Each visit's state after each prefill.
    prefill_states: Vec<u64>,
    /// Logits of each decode step after the prefill.
    decode: Vec<u64>,
}

impl Run {
    fn digest(&self) -> u64 {
        [
            &self.steps,
            &self.step_states,
            &self.prefill,
            &self.prefill_states,
            &self.decode,
        ]
        .iter()
        .flat_map(|v| v.iter())
        .fold(FNV_START, |h, &x| {
            x.to_le_bytes()
                .iter()
                .fold(h, |h, &b| (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3))
        })
    }

    /// Where `other` first differs from this run, for the failure message.
    fn differences(&self, other: &Self) -> String {
        let first = |a: &[u64], b: &[u64]| a.iter().zip(b).position(|(x, y)| x != y);
        let count = |a: &[u64], b: &[u64]| a.iter().zip(b).filter(|(x, y)| x != y).count();
        let mut parts = Vec::new();
        for (name, a, b) in [
            ("decode step", &self.steps, &other.steps),
            ("visit after decode", &self.step_states, &other.step_states),
            ("prefill row", &self.prefill, &other.prefill),
            (
                "visit after prefill",
                &self.prefill_states,
                &other.prefill_states,
            ),
            ("decode after prefill", &self.decode, &other.decode),
        ] {
            if let Some(i) = first(a, b) {
                parts.push(format!(
                    "{name}s differ from {i} on ({} of {})",
                    count(a, b),
                    a.len()
                ));
            }
        }
        parts.join("; ")
    }
}

fn run(model: &mut Model, tokens: &[u32]) -> anyhow::Result<Run> {
    model.reset()?;
    let mut steps = Vec::new();
    for &token in &tokens[..STEPS] {
        model.step(token, None)?;
        steps.push(hash(&model.logits()?));
    }
    let step_states = snapshot_hashes(&model.cache_snapshot()?);

    let mut prefill = Vec::new();
    let mut prefill_states = Vec::new();
    for chunk in CHUNKS {
        model.reset()?;
        let start = prefill.len();
        let mut callback = |position: usize, logits: &[f32]| {
            ensure!(position == prefill.len() - start, "rows out of order");
            prefill.push(hash(logits));
            Ok(())
        };
        model.prefill(&tokens[..PROMPT], chunk, PrefillLogits::All(&mut callback))?;
        prefill_states.extend(snapshot_hashes(&model.cache_snapshot()?));
    }
    let mut decode = Vec::new();
    for &token in &tokens[PROMPT..PROMPT + DECODE_AFTER_PREFILL] {
        model.step(token, None)?;
        decode.push(hash(&model.logits()?));
    }
    Ok(Run {
        steps,
        step_states,
        prefill,
        prefill_states,
        decode,
    })
}

fn load(gpu: &Gpu, model_dir: &PathBuf, config: Config) -> anyhow::Result<Model> {
    let text = TextConfig::load(model_dir)?;
    let passes = text.loop_config.num_passes;
    let checkpoint = Checkpoint::open(model_dir)?;
    let weights = Weights::load(&checkpoint, &text, gpu.stream())?;
    let mut model =
        Model::with_formats(gpu, text, weights, passes, 2048, config.kv, config.weights)?;
    model.set_expert_activations(config.experts);
    Ok(model)
}

#[test]
#[ignore = "needs the checkpoint (INKLING_MODEL) and about 20 GB of GPU memory"]
fn reruns_are_bit_identical() -> anyhow::Result<()> {
    let model_dir = PathBuf::from(env::var_os("INKLING_MODEL").context("set INKLING_MODEL")?);
    let tokens = tokens()?;
    ensure!(
        tokens.len() >= PROMPT + DECODE_AFTER_PREFILL,
        "the capture is too short"
    );
    let child = env::var_os(CHILD_ENV).is_some();

    // The other process's digests, one per configuration.
    let mut theirs = Vec::new();
    if !child {
        let output = Command::new(env::current_exe()?)
            .args([
                "--exact",
                "reruns_are_bit_identical",
                "--ignored",
                "--nocapture",
            ])
            .env(CHILD_ENV, "1")
            .output()
            .context("running the child process")?;
        let stdout = String::from_utf8_lossy(&output.stdout);
        ensure!(
            output.status.success(),
            "child process failed: {stdout}{}",
            String::from_utf8_lossy(&output.stderr)
        );
        for line in stdout.lines() {
            if let Some(rest) = line.strip_prefix(DIGEST_PREFIX) {
                let (name, digest) = rest.trim().split_once(' ').context("digest line")?;
                theirs.push((name.to_owned(), u64::from_str_radix(digest, 16)?));
            }
        }
        ensure!(theirs.len() == CONFIGS.len(), "child printed {stdout}");
    }

    let gpu = Gpu::new(0)?;
    let mut failures = Vec::new();
    for (k, config) in CONFIGS.into_iter().enumerate() {
        let mut model = load(&gpu, &model_dir, config)?;
        let first = run(&mut model, &tokens)?;
        if child {
            println!("{DIGEST_PREFIX} {} {:016x}", config.name, first.digest());
            continue;
        }
        let second = run(&mut model, &tokens)?;
        let in_process = first == second;
        let across = theirs[k] == (config.name.to_owned(), first.digest());
        println!(
            "{}: in one process identical: {in_process}, across processes identical: {across}",
            config.name
        );
        if !in_process {
            failures.push(format!(
                "{}: second run: {}",
                config.name,
                first.differences(&second)
            ));
        }
        if !across {
            failures.push(format!(
                "{}: digest {:016x}, the other process's {:016x}",
                config.name,
                first.digest(),
                theirs[k].1
            ));
        }
    }
    ensure!(failures.is_empty(), "{}", failures.join("\n"));
    Ok(())
}
