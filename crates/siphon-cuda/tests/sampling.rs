//! Speculative sampling against plain sampling.
//!
//! Needs the checkpoint, the `regex-readme` capture and about 21 GB of GPU memory, so the
//! tests are ignored by default. Each loads its own model, so run them one at a time:
//!
//! ```sh
//! SIPHON_MODEL=<model-dir> cargo test --release -p siphon-cuda --test sampling -- \
//!     --ignored --nocapture --test-threads 1
//! ```
//!
//! - `speculation_keeps_the_distribution`: for each of `temperature 1` and `temperature
//!   0.8, top_p 0.95`, the first 4 tokens after a prompt over many seeds, with
//!   speculation (the default fast draft, 7 drafts per round) and with plain
//!   `Model::step` decoding, which sums in other orders. Plain decoding and each setting
//!   take seeds of their own, so the samples are independent. A two-sample chi-square
//!   test compares each position's token, the first two tokens together and the 4-token
//!   sequences; the test fails when any z-score passes 4.5 (a one-sided p-value of 3.4e-6
//!   each, 4e-5 for the 12 tests together). The prompt is the prefix of the
//!   `regex-readme` capture after which the next token has the most entropy. The test
//!   also checks that speculation and batched steps with the same seed give the same 48
//!   tokens: a draw depends only on the logits, the seed and the position, and a batched
//!   row's logits equal a verify row's.
//! - `draft_sampling_acceptance` measures what a sampling draft would gain over the greedy
//!   draft the speculator uses: at each verified draft position, the greedy draft is kept
//!   with probability p(x), while a draft sampled from its own filtered distribution q
//!   would be kept with probability sum(min(p, q)). Both come from the same rounds, so the
//!   positions after the first condition on the greedy drafts before them. It needs the
//!   exact draft, the only one that records its logits.
//!
//! `SIPHON_SAMPLING_SEEDS` sets the seeds per sample (default 1000).

#![cfg(target_os = "linux")]

use std::{
    collections::HashMap,
    env, fs,
    path::PathBuf,
    sync::{Arc, Mutex},
};

use anyhow::{Context as _, ensure};
use safetensors::{Dtype, SafeTensors};
use siphon::{Checkpoint, TextConfig};
use siphon_cuda::{
    Gpu,
    draft::DraftOptions,
    model::{DEFAULT_PREFILL_CHUNK, Model, PrefillLogits, Sampling},
    ops::attention::KvFormat,
    spec::{DraftSource, Drafts, RoundRecord, Speculator, Width, load_draft},
    weights::{WeightFormat, Weights},
};

const CAPACITY: usize = 4096;
/// Tokens per draw for the statistics.
const DRAWN: usize = 4;
/// Tokens per seed for the exact comparison with batched steps, and its seeds.
const EXACT_TOKENS: usize = 48;
const EXACT_SEEDS: u64 = 12;

fn capture(text: &str) -> anyhow::Result<Vec<u32>> {
    let dir = env::var_os("SIPHON_CAPTURES").map_or_else(
        || PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../reference/out"),
        PathBuf::from,
    );
    let path = dir.join(format!("{text}-chunked.safetensors"));
    let bytes = fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
    let tensors = SafeTensors::deserialize(&bytes)?;
    let view = tensors.tensor("tokens")?;
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
    model: Model,
    speculator: Speculator,
}

fn load(draft: &DraftOptions, slots: usize) -> anyhow::Result<Setup> {
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
    model.enable_sampling()?;
    if slots > 0 {
        model.add_slots(slots)?;
    }
    let source = DraftSource::Bundled;
    let d = load_draft(&mut model, &checkpoint, source, CAPACITY, draft)?;
    let speculator = Speculator::with_draft(&model, d, Width::Fixed(7))?;
    Ok(Setup {
        _gpu: gpu,
        model,
        speculator,
    })
}

fn seeds() -> u64 {
    env::var("SIPHON_SAMPLING_SEEDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1000)
}

/// The first `n` tokens after `prompt` with speculation.
fn speculated(
    s: &mut Setup,
    prompt: &[u32],
    sampling: Sampling,
    n: usize,
) -> anyhow::Result<Vec<u32>> {
    s.model.set_sampling(sampling);
    s.model.reset()?;
    let mut tokens = vec![
        s.speculator
            .prefill(&mut s.model, prompt, DEFAULT_PREFILL_CHUNK)?,
    ];
    if n > 1 {
        tokens.extend(s.speculator.generate(&mut s.model, n - 1, Drafts::Dspark)?);
    }
    tokens.truncate(n);
    Ok(tokens)
}

/// The first `n` tokens after `prompt` with plain steps.
fn stepped(
    model: &mut Model,
    prompt: &[u32],
    sampling: Sampling,
    n: usize,
) -> anyhow::Result<Vec<u32>> {
    model.set_sampling(sampling);
    model.reset()?;
    let mut token = model.prefill(prompt, DEFAULT_PREFILL_CHUNK, PrefillLogits::Last)?;
    let mut tokens = vec![token];
    while tokens.len() < n {
        token = model.step(token, None)?;
        tokens.push(token);
    }
    Ok(tokens)
}

/// The first `n` tokens after `prompt` with one-row batched steps in slot 1.
fn batched(
    model: &mut Model,
    prompt: &[u32],
    sampling: Sampling,
    n: usize,
) -> anyhow::Result<Vec<u32>> {
    model.set_sampling(sampling);
    model.reset_slot(1)?;
    let mut token = model.prefill_slot(1, prompt, DEFAULT_PREFILL_CHUNK)?;
    let mut tokens = vec![token];
    while tokens.len() < n {
        token = model.step_batch_sampled(&[(1, token)], &[sampling])?[0];
        tokens.push(token);
    }
    Ok(tokens)
}

/// The prefix length of `text`, among 120 to 1500 tokens in steps of 20, after which the
/// next token's distribution at temperature 1 has the most entropy, so that sampling has
/// many outcomes to compare.
fn open_cut(model: &mut Model, text: &[u32]) -> anyhow::Result<usize> {
    model.set_sampling(Sampling::GREEDY);
    let mut best = (0, f64::MIN);
    for cut in (120..=text.len().min(1500)).step_by(20) {
        model.reset()?;
        model.prefill(&text[..cut], DEFAULT_PREFILL_CHUNK, PrefillLogits::Last)?;
        let logits = model.logits()?;
        let max = logits.iter().copied().fold(f32::MIN, f32::max);
        let w: Vec<f64> = logits.iter().map(|&l| f64::from(l - max).exp()).collect();
        let total: f64 = w.iter().sum();
        let entropy: f64 = w
            .iter()
            .filter(|&&v| v > 0.0)
            .map(|&v| -(v / total) * (v / total).ln())
            .sum();
        if entropy > best.1 {
            best = (cut, entropy);
        }
    }
    println!(
        "prompt: the first {} tokens, next-token entropy {:.2} nats",
        best.0, best.1
    );
    Ok(best.0)
}

/// A z-score for a chi-square statistic with `dof` degrees of freedom (Wilson and
/// Hilferty).
fn chi_square_z(statistic: f64, dof: f64) -> f64 {
    let a = 2.0 / (9.0 * dof);
    ((statistic / dof).powf(1.0 / 3.0) - (1.0 - a)) / a.sqrt()
}

/// Two samples of the same size compared category by category: the chi-square z-score and
/// its degrees of freedom. Categories with fewer than 10 draws in both samples together
/// are pooled.
fn compare<K: std::hash::Hash + Eq + Clone>(a: &[K], b: &[K]) -> (f64, f64) {
    let mut counts: HashMap<K, (f64, f64)> = HashMap::new();
    for k in a {
        counts.entry(k.clone()).or_default().0 += 1.0;
    }
    for k in b {
        counts.entry(k.clone()).or_default().1 += 1.0;
    }
    let (mut statistic, mut dof) = (0.0, -1.0);
    let mut pooled = (0.0, 0.0);
    for &(x, y) in counts.values() {
        if x + y >= 10.0 {
            statistic += (x - y).powi(2) / (x + y);
            dof += 1.0;
        } else {
            pooled.0 += x;
            pooled.1 += y;
        }
    }
    if pooled.0 + pooled.1 >= 10.0 {
        statistic += (pooled.0 - pooled.1).powi(2) / (pooled.0 + pooled.1);
        dof += 1.0;
    }
    let z = if dof >= 1.0 {
        chi_square_z(statistic, dof)
    } else {
        0.0
    };
    (z, dof)
}

#[test]
#[ignore = "needs the checkpoint (SIPHON_MODEL), the captures and about 21 GB of GPU memory"]
fn speculation_keeps_the_distribution() -> anyhow::Result<()> {
    let mut s = load(&DraftOptions::fast(), 1)?;
    let readme = capture("regex-readme")?;
    let prompt = &readme[..open_cut(&mut s.model, &readme)?];
    let seeds = seeds();
    let mut worst: f64 = f64::MIN;
    let settings = [
        ("temperature 1", 1.0, 1.0),
        ("temperature 0.8, top_p 0.95", 0.8, 0.95),
    ];
    for (setting, (name, temperature, top_p)) in (0u64..).zip(settings) {
        // Each setting and each sample takes seeds of its own, so that no two draws share
        // their random variables.
        let sampling = |seed: u64| Sampling::new(temperature, 0, top_p, 0.0, setting << 32 | seed);
        for seed in 0..EXACT_SEEDS {
            let a = speculated(&mut s, prompt, sampling(seed)?, EXACT_TOKENS)?;
            let tokens_per_round = s.speculator.stats().tokens_per_round();
            let b = batched(&mut s.model, prompt, sampling(seed)?, EXACT_TOKENS)?;
            let differ = a.iter().zip(&b).position(|(x, y)| x != y);
            ensure!(
                differ.is_none(),
                "{name}, seed {seed}: speculation and batched steps differ at token {differ:?} \
                 ({tokens_per_round:.2} tokens per round)"
            );
        }
        println!(
            "{name}: speculation equals batched steps for {EXACT_SEEDS} seeds of {EXACT_TOKENS} tokens"
        );
        let mut spec = Vec::new();
        let mut plain = Vec::new();
        let (mut rounds, mut emitted) = (0, 0);
        for seed in 0..seeds {
            spec.push(speculated(&mut s, prompt, sampling(seed)?, DRAWN)?);
            rounds += s.speculator.stats().rounds;
            emitted += s.speculator.stats().emitted;
            plain.push(stepped(
                &mut s.model,
                prompt,
                sampling(1 << 31 | seed)?,
                DRAWN,
            )?);
        }
        let mut tests: Vec<(String, (f64, f64))> = Vec::new();
        for j in 0..DRAWN {
            let a: Vec<u32> = spec.iter().map(|t| t[j]).collect();
            let b: Vec<u32> = plain.iter().map(|t| t[j]).collect();
            tests.push((format!("token {}", j + 1), compare(&a, &b)));
        }
        let pairs =
            |v: &[Vec<u32>]| -> Vec<(u32, u32)> { v.iter().map(|t| (t[0], t[1])).collect() };
        tests.push((
            "tokens 1-2".to_owned(),
            compare(&pairs(&spec), &pairs(&plain)),
        ));
        tests.push(("tokens 1-4".to_owned(), compare(&spec, &plain)));
        println!(
            "{name}: {seeds} seeds each, speculation {:.2} tokens per round",
            emitted as f64 / rounds.max(1) as f64
        );
        for (what, (z, dof)) in &tests {
            println!("  {what}: chi-square z {z:.2} over {dof} degrees of freedom");
            worst = worst.max(*z);
        }
    }
    ensure!(
        worst < 4.5,
        "speculative and plain sampling differ (z {worst:.2})"
    );
    Ok(())
}

/// vLLM's filtered distribution over one row of logits, for measurements: tokens more than
/// 30 nats below the largest are left out, which changes no probability by more than 1e-7.
fn filtered(logits: &[f32], s: Sampling) -> Vec<(usize, f64)> {
    let t = f64::from(s.temperature);
    let max = logits.iter().copied().fold(f32::MIN, f32::max);
    let mut kept: Vec<(usize, f64)> = logits
        .iter()
        .enumerate()
        .map(|(i, &l)| (i, (f64::from(l) - f64::from(max)) / t))
        .filter(|&(_, z)| z > -30.0)
        .collect();
    kept.sort_by(|a, b| b.1.total_cmp(&a.1));
    let k = s.top_k as usize;
    if k > 0 && k < kept.len() {
        let kth = kept[k - 1].1;
        kept.retain(|&(_, z)| z >= kth);
    }
    if s.top_p < 1.0 {
        let total: f64 = kept.iter().map(|&(_, z)| z.exp()).sum();
        let mut above = 0.0;
        let mut end = 0;
        while end < kept.len() && above < f64::from(s.top_p) * total {
            let z = kept[end].1;
            while end < kept.len() && kept[end].1.to_bits() == z.to_bits() {
                above += z.exp();
                end += 1;
            }
        }
        kept.truncate(end);
    }
    if s.min_p > 0.0 {
        let floor = f64::from(s.min_p).ln();
        kept.retain(|&(_, z)| z >= floor);
    }
    let total: f64 = kept.iter().map(|&(_, z)| z.exp()).sum();
    kept.into_iter()
        .map(|(i, z)| (i, z.exp() / total))
        .collect()
}

#[derive(Default)]
struct Acceptance {
    rounds: usize,
    /// Per draft position: summed p(x) for the greedy draft, summed sum(min(p, q)).
    greedy: [f64; 7],
    sampled: [f64; 7],
    /// Expected tokens per 8-row round under each.
    greedy_tokens: f64,
    sampled_tokens: f64,
}

#[test]
#[ignore = "needs the checkpoint (SIPHON_MODEL), the captures and about 21 GB of GPU memory"]
fn draft_sampling_acceptance() -> anyhow::Result<()> {
    let mut s = load(&DraftOptions::default(), 0)?;
    s.speculator.record_draft_logits(&s.model)?;
    let vocab = s.model.config().unpadded_vocab_size;
    let prompts: Vec<Vec<u32>> = ["regex-readme", "half", "anyhow"]
        .iter()
        .map(|t| capture(t).map(|c| c[..c.len().min(1500)].to_vec()))
        .collect::<anyhow::Result<_>>()?;
    for (name, temperature, top_p) in [
        ("temperature 1", 1.0, 1.0),
        ("temperature 0.8, top_p 0.95", 0.8, 0.95),
    ] {
        let sampling = Sampling::new(temperature, 0, top_p, 0.0, 5)?;
        let totals = Arc::new(Mutex::new(Acceptance::default()));
        let sink = Arc::clone(&totals);
        s.speculator
            .set_round_hook(Some(Box::new(move |r: &RoundRecord<'_>| {
                let mut a = sink.lock().map_err(|_| anyhow::anyhow!("poisoned"))?;
                a.rounds += 1;
                let (mut g_chain, mut s_chain) = (1.0, 1.0);
                let (mut g_tokens, mut s_tokens) = (1.0, 1.0);
                for (j, &draft) in r.drafts.iter().enumerate() {
                    let p = filtered(&r.target_logits[j * vocab..(j + 1) * vocab], sampling);
                    let q = filtered(&r.draft_logits[j * vocab..(j + 1) * vocab], sampling);
                    let p_draft = p
                        .iter()
                        .find(|&&(i, _)| i == draft as usize)
                        .map_or(0.0, |&(_, v)| v);
                    let q_map: HashMap<usize, f64> = q.into_iter().collect();
                    let overlap: f64 = p
                        .iter()
                        .map(|&(i, v)| v.min(*q_map.get(&i).unwrap_or(&0.0)))
                        .sum();
                    a.greedy[j] += p_draft;
                    a.sampled[j] += overlap;
                    g_chain *= p_draft;
                    s_chain *= overlap;
                    g_tokens += g_chain;
                    s_tokens += s_chain;
                }
                a.greedy_tokens += g_tokens;
                a.sampled_tokens += s_tokens;
                Ok(())
            })));
        let mut kept = (0, 0);
        for prompt in &prompts {
            s.model.set_sampling(sampling);
            s.model.reset()?;
            s.speculator
                .prefill(&mut s.model, prompt, DEFAULT_PREFILL_CHUNK)?;
            s.speculator.generate(&mut s.model, 192, Drafts::Dspark)?;
            kept.0 += s.speculator.stats().emitted;
            kept.1 += s.speculator.stats().rounds;
        }
        s.speculator.set_round_hook(None);
        let a = totals.lock().map_err(|_| anyhow::anyhow!("poisoned"))?;
        let n = a.rounds as f64;
        let per = |v: &[f64; 7]| {
            v.iter()
                .map(|x| format!("{:.3}", x / n))
                .collect::<Vec<_>>()
                .join(", ")
        };
        println!(
            "{name}, {} rounds: greedy draft kept [{}], expected {:.2} tokens per round \
             (measured {:.2}); sampled draft would be kept [{}], expected {:.2} tokens per round",
            a.rounds,
            per(&a.greedy),
            a.greedy_tokens / n,
            kept.0 as f64 / kept.1.max(1) as f64,
            per(&a.sampled),
            a.sampled_tokens / n,
        );
    }
    Ok(())
}

/// Prints the verify pass's time per round at fixed widths, greedy and sampling, on the
/// prose prompt: the sampling kernels' cost on the model's own logits. Also times plain
/// steps, which sample after the step's graph.
#[test]
#[ignore = "a timing, not a check; needs the checkpoint and about 21 GB of GPU memory"]
fn sampling_cost_in_the_verify() -> anyhow::Result<()> {
    let mut s = load(&DraftOptions::fast(), 0)?;
    let readme = capture("regex-readme")?;
    let prompt = &readme[..open_cut(&mut s.model, &readme)?];
    let configs = [
        ("greedy", Sampling::GREEDY),
        ("temperature 1", Sampling::new(1.0, 0, 1.0, 0.0, 3)?),
        (
            "temperature 0.8, top_p 0.95",
            Sampling::new(0.8, 0, 0.95, 0.0, 3)?,
        ),
        (
            "temperature 0.7, top_k 40, top_p 0.95",
            Sampling::new(0.7, 40, 0.95, 0.0, 3)?,
        ),
    ];
    for width in [2, 4, 8] {
        for (name, sampling) in configs {
            let mut ms = Vec::new();
            for _ in 0..3 {
                s.speculator.set_width(Width::Fixed(width - 1))?;
                speculated(&mut s, prompt, sampling, 160)?;
                let stats = s.speculator.stats();
                ms.push(stats.verify.as_secs_f64() * 1e3 / stats.rounds as f64);
            }
            let best = ms.iter().copied().fold(f64::MAX, f64::min);
            println!("width {width}, {name}: verify {best:.3} ms per round");
        }
    }
    for (name, sampling) in configs {
        let started = std::time::Instant::now();
        stepped(&mut s.model, prompt, sampling, 129)?;
        println!(
            "plain steps, {name}: {:.3} ms per step, prefill included",
            started.elapsed().as_secs_f64() * 1e3 / 128.0
        );
    }
    Ok(())
}
