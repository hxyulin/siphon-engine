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
//! - `draft_sampling_acceptance` measures, from the rounds of the exact bundled draft
//!   (the only one that records its logits), what each kind of draft would keep at each
//!   verified position: the draft's argmax x is kept with probability p(x), a draft
//!   sampled from its own filtered distribution q with probability sum(min(p, q)), and the
//!   coupled draft, which speculation proposes, when its race over the draft's logits
//!   picks the target's draw (`SIPHON_COUPLED_SEEDS` seeds per position, default 32).
//!   The run's drafts are coupled picks, so every position after the first conditions on
//!   the coupled drafts before it as drawn with the run's seed; the other seeds' chains
//!   reuse those rounds, an approximation. The test also prints the run's measured share
//!   kept, the two confidences a coupled draft could report, and checks that the host's
//!   race (`siphon::race`) reproduces the draft's picks and the verify's draws.
//! - `coupled_draft_acceptance` runs speculation end to end with the draft's argmax and
//!   with coupled drafts, the same seeds each, and prints tokens per round, the share kept
//!   by position and the draft's time per round: the fast bundled draft, and DFlash 2
//!   with `SIPHON_DFLASH2` set to its directory (`SIPHON_COUPLED_RUNS` seeds per prompt,
//!   default 4).
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
use siphon::{Checkpoint, TextConfig, race};
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
/// The seed of `draft_sampling_acceptance`'s run.
const RUN_SEED: u64 = 5;

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

fn model_dir() -> anyhow::Result<PathBuf> {
    Ok(PathBuf::from(
        env::var_os("SIPHON_MODEL").context("set SIPHON_MODEL")?,
    ))
}

/// The model, sampling on, with `slots` batch slots.
fn load_model(slots: usize) -> anyhow::Result<(Gpu, Model)> {
    let model_dir = model_dir()?;
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
    Ok((gpu, model))
}

/// A speculator with the draft `source` names and its approximations `draft`, 7 drafts
/// per round.
fn speculator_for(
    model: &mut Model,
    source: DraftSource<'_>,
    draft: &DraftOptions,
) -> anyhow::Result<Speculator> {
    let checkpoint = Checkpoint::open(&model_dir()?)?;
    let d = load_draft(model, &checkpoint, source, CAPACITY, draft)?;
    Speculator::with_draft(model, d, Width::Fixed(7))
}

fn load(draft: &DraftOptions, slots: usize) -> anyhow::Result<Setup> {
    let (gpu, mut model) = load_model(slots)?;
    let speculator = speculator_for(&mut model, DraftSource::Bundled, draft)?;
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

/// The most likely tokens of `dist` (as [`filtered`] returns it, most likely first) that
/// hold all but 1e-6 of its mass. The race's winner over every token lies among them
/// except with probability 1e-6: a race picks each token with its probability.
fn likely(dist: &[(usize, f64)]) -> Vec<usize> {
    let mut mass = 0.0;
    let mut out = Vec::new();
    for &(i, p) in dist {
        out.push(i);
        mass += p;
        if mass >= 1.0 - 1e-6 {
            break;
        }
    }
    out
}

/// The first largest value's index, the draft kernels' argmax rule.
fn argmax(row: &[f32]) -> usize {
    let mut best = 0;
    for (i, &v) in row.iter().enumerate() {
        if v > row[best] {
            best = i;
        }
    }
    best
}

/// Seeds per draft position for the coupled draft's acceptance (`SIPHON_COUPLED_SEEDS`,
/// default 32).
fn coupled_seeds() -> u64 {
    env::var("SIPHON_COUPLED_SEEDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(32)
}

#[derive(Default)]
struct Acceptance {
    rounds: usize,
    /// Per draft position: summed p(x) for the greedy draft x, summed sum(min(p, q)) for a
    /// draft sampled from its filtered q, and the coupled draft's share kept over the
    /// seeds.
    greedy: [f64; 7],
    sampled: [f64; 7],
    coupled: [f64; 7],
    /// Per draft position, the confidences a coupled draft could report: the largest
    /// probability under softmax(l), which the kernels report, the largest under
    /// softmax(l / T), and the probability of the run's coupled pick.
    plain_confidence: [f64; 7],
    peak_confidence: [f64; 7],
    pick_confidence: [f64; 7],
    /// The run's drafts kept, by position.
    kept: [f64; 7],
    /// Expected tokens per 8-row round under each.
    greedy_tokens: f64,
    sampled_tokens: f64,
    coupled_tokens: f64,
    /// Positions checked, and those where the host's race disagreed with the draft's
    /// coupled pick or with the verify's draw.
    checked: usize,
    draft_mismatches: usize,
    draw_mismatches: usize,
}

impl Acceptance {
    /// Adds one round of the coupled draft at `sampling`, with `seeds` seeds per position
    /// for the coupled draft's share kept.
    fn add_round(
        &mut self,
        r: &RoundRecord<'_>,
        vocab: usize,
        sampling: Sampling,
        seeds: u64,
    ) -> anyhow::Result<()> {
        let temperature = sampling.temperature;
        // The coupled pick ignores the target's filters (see Draft::set_sampling).
        let unfiltered = Sampling::new(temperature, 0, 1.0, 0.0, sampling.seed)?;
        self.rounds += 1;
        let (mut g_chain, mut s_chain) = (1.0, 1.0);
        let (mut g_tokens, mut s_tokens) = (1.0, 1.0);
        let seeds = usize::try_from(seeds)?;
        let mut c_chain = vec![true; seeds];
        let mut c_tokens = 1.0;
        for (j, &draft) in r.drafts.iter().enumerate() {
            let target = &r.target_logits[j * vocab..(j + 1) * vocab];
            let logits = &r.draft_logits[j * vocab..(j + 1) * vocab];
            let p = filtered(target, sampling);
            let q = filtered(logits, sampling);
            let q_all = filtered(logits, unfiltered);
            let p_of = |x: usize| p.iter().find(|&&(i, _)| i == x).map_or(0.0, |&(_, v)| v);
            let q_all_of = |x: usize| {
                q_all
                    .iter()
                    .find(|&&(i, _)| i == x)
                    .map_or(0.0, |&(_, v)| v)
            };
            let p_greedy = p_of(argmax(logits));
            let q_map: HashMap<usize, f64> = q.into_iter().collect();
            let overlap: f64 = p
                .iter()
                .map(|&(i, v)| v.min(*q_map.get(&i).unwrap_or(&0.0)))
                .sum();
            self.greedy[j] += p_greedy;
            self.sampled[j] += overlap;
            g_chain *= p_greedy;
            s_chain *= overlap;
            g_tokens += g_chain;
            s_tokens += s_chain;

            // Verify row j draws the token at position start + j + 1.
            let position = u32::try_from(r.start + j + 1)?;
            let (p_set, q_set) = (likely(&p), likely(&q_all));
            let race_pair = |seed: u64| {
                let key = race::draw_key(seed, position);
                (
                    race::pick(target, p_set.iter().copied(), temperature, key),
                    race::pick(logits, q_set.iter().copied(), temperature, key),
                )
            };
            let (drawn, coupled) = race_pair(RUN_SEED);
            self.checked += 1;
            self.draft_mismatches += usize::from(coupled != Some(draft as usize));
            self.draw_mismatches += usize::from(drawn != Some(r.drawn[j] as usize));
            self.kept[j] += f64::from(u8::from(r.accepted > j));
            let plain = filtered(logits, Sampling::new(1.0, 0, 1.0, 0.0, sampling.seed)?);
            self.plain_confidence[j] += plain.first().map_or(0.0, |&(_, v)| v);
            self.peak_confidence[j] += q_all.first().map_or(0.0, |&(_, v)| v);
            self.pick_confidence[j] += q_all_of(draft as usize);
            let mut matched = 0;
            for (k, alive) in c_chain.iter_mut().enumerate() {
                // Seeds of their own, away from the run's.
                let (drawn, coupled) = race_pair(1 << 40 | k as u64);
                let hit = drawn == coupled;
                matched += usize::from(hit);
                *alive &= hit;
            }
            self.coupled[j] += matched as f64 / seeds as f64;
            c_tokens += c_chain.iter().filter(|&&alive| alive).count() as f64 / seeds as f64;
        }
        self.greedy_tokens += g_tokens;
        self.sampled_tokens += s_tokens;
        self.coupled_tokens += c_tokens;
        Ok(())
    }

    /// Prints the means per round, with `measured` tokens per round from the run.
    fn report(&self, name: &str, seeds: u64, measured: f64) {
        let n = self.rounds as f64;
        let per = |v: &[f64; 7]| {
            v.iter()
                .map(|x| format!("{:.3}", x / n))
                .collect::<Vec<_>>()
                .join(", ")
        };
        println!("{name}, {} rounds of the coupled draft:", self.rounds);
        println!(
            "  greedy draft kept [{}], expected {:.2} tokens per round",
            per(&self.greedy),
            self.greedy_tokens / n
        );
        println!(
            "  draft sampled from q kept [{}], expected {:.2} tokens per round",
            per(&self.sampled),
            self.sampled_tokens / n
        );
        println!(
            "  coupled draft kept [{}] over {seeds} seeds, {:.2} tokens per round",
            per(&self.coupled),
            self.coupled_tokens / n
        );
        println!(
            "  coupled draft kept in the run [{}], measured {:.2} tokens per round",
            per(&self.kept),
            measured
        );
        println!(
            "  confidence: largest softmax(l) [{}], largest softmax(l / T) [{}], the pick's [{}]",
            per(&self.plain_confidence),
            per(&self.peak_confidence),
            per(&self.pick_confidence)
        );
        println!(
            "  host race against the GPU over {} positions: {} draft picks and {} draws differ",
            self.checked, self.draft_mismatches, self.draw_mismatches
        );
    }
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
    let virtual_seeds = coupled_seeds();
    for (name, temperature, top_p) in [
        ("temperature 1", 1.0, 1.0),
        ("temperature 0.8, top_p 0.95", 0.8, 0.95),
    ] {
        let sampling = Sampling::new(temperature, 0, top_p, 0.0, RUN_SEED)?;
        let totals = Arc::new(Mutex::new(Acceptance::default()));
        let sink = Arc::clone(&totals);
        s.speculator
            .set_round_hook(Some(Box::new(move |r: &RoundRecord<'_>| {
                let mut a = sink.lock().map_err(|_| anyhow::anyhow!("poisoned"))?;
                a.add_round(r, vocab, sampling, virtual_seeds)
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
        a.report(name, virtual_seeds, kept.0 as f64 / kept.1.max(1) as f64);
        // A few may differ: the logarithms round differently on the GPU, and the host's
        // races leave out a tail of 1e-6 of the mass.
        ensure!(
            a.draft_mismatches * 200 <= a.checked && a.draw_mismatches * 200 <= a.checked,
            "the host's race does not reproduce the GPU's"
        );
    }
    Ok(())
}

/// Speculation with the draft's argmax and with the coupled pick, end to end, at 7 drafts
/// per round: tokens per round, each draft position's share kept and the draft's time per
/// round. The fast bundled draft, as speculation runs it, and DFlash 2 with
/// `SIPHON_DFLASH2` set to its directory. The output is the same in both modes; with the
/// same seeds, the tokens per round compare directly.
#[test]
#[ignore = "needs the checkpoint (SIPHON_MODEL), the captures and about 21 GB of GPU memory"]
fn coupled_draft_acceptance() -> anyhow::Result<()> {
    let (_gpu, mut model) = load_model(0)?;
    let prompts: Vec<Vec<u32>> = ["regex-readme", "half", "anyhow"]
        .iter()
        .map(|t| capture(t).map(|c| c[..c.len().min(1500)].to_vec()))
        .collect::<anyhow::Result<_>>()?;
    let runs = env::var("SIPHON_COUPLED_RUNS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4u64);
    let dflash2 = env::var_os("SIPHON_DFLASH2").map(PathBuf::from);
    let mut drafters = vec![("bundled", DraftSource::Bundled)];
    if let Some(dir) = &dflash2 {
        drafters.push(("DFlash 2", DraftSource::Dflash(dir)));
    }
    for (drafter, source) in drafters {
        let mut speculator = speculator_for(&mut model, source, &DraftOptions::fast())?;
        for (name, temperature, top_p) in [
            ("temperature 1", 1.0, 1.0),
            ("temperature 0.8, top_p 0.95", 0.8, 0.95),
        ] {
            for coupled in [false, true] {
                speculator.set_coupled_drafts(coupled);
                let (mut rounds, mut emitted, mut draft_ms) = (0, 0, 0.0);
                let mut kept = [0usize; 7];
                for prompt in &prompts {
                    for seed in 0..runs {
                        let sampling = Sampling::new(temperature, 0, top_p, 0.0, 100 + seed)?;
                        model.set_sampling(sampling);
                        model.reset()?;
                        speculator.prefill(&mut model, prompt, DEFAULT_PREFILL_CHUNK)?;
                        speculator.generate(&mut model, 192, Drafts::Dspark)?;
                        let stats = speculator.stats();
                        rounds += stats.rounds;
                        emitted += stats.emitted;
                        draft_ms += stats.draft.as_secs_f64() * 1e3;
                        for (k, &n) in kept.iter_mut().zip(&stats.accepted_at) {
                            *k += n;
                        }
                    }
                }
                let n = rounds.max(1) as f64;
                let per: Vec<String> = kept
                    .iter()
                    .map(|&k| format!("{:.3}", k as f64 / n))
                    .collect();
                println!(
                    "{drafter}, {name}, {} drafts: {:.2} tokens per round, kept [{}], draft {:.3} ms per round",
                    if coupled { "coupled" } else { "argmax" },
                    emitted as f64 / n,
                    per.join(", "),
                    draft_ms / n
                );
            }
        }
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
