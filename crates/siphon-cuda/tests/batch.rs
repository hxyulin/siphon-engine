//! Batched decode (`Model::step_batch`): each sequence of a batch must emit exactly the
//! tokens, and the logits bit for bit, of one-row verifies of that sequence alone.
//!
//! Needs the checkpoint and about 20 GB of GPU memory, so every test is ignored by
//! default. Each test loads its own model, so run them one at a time:
//!
//! ```sh
//! SIPHON_MODEL=<model-dir> cargo test --release -p siphon-cuda --test batch -- \
//!     --ignored --nocapture --test-threads 1
//! ```
//!
//! The prompts are slices of the tokens of `reference/out/anyhow-chunked.safetensors` (or
//! `SIPHON_CAPTURE`), of lengths on both sides of the sliding-window wrap at 1024. The
//! sequences join and leave the batch at different steps, so the batch size, the rows'
//! order and their positions all change during the run. `batch_matches_alone_*` runs 5
//! sequences, `_16_*` and `_32_*` that many, which fill a batch of 16 or 32 rows.
//!
//! `shared_prefill_matches_separate` checks the server's prefill sharing for requests with
//! `n` choices: one slot prefills a prompt, `Model::move_slot` copies it into the others,
//! and `Model::redraw_slot` draws each copy's first token with its own seed. Each copy must
//! decode, tokens and logits bit for bit, as the same seed does after its own prefill.
//!
//! `speculation_returns_after_batch_bundled` speculates, batches and speculates again with the
//! bundled draft; `speculation_returns_after_batch_dflash` does the same with the drafter
//! in `SIPHON_DFLASH`, and passes without it.

#![cfg(target_os = "linux")]

use std::{env, fs, path::PathBuf};

use anyhow::{Context as _, ensure};
use safetensors::{Dtype, SafeTensors};
use siphon::{Checkpoint, TextConfig};
use siphon_cuda::{
    Gpu,
    draft::DraftOptions,
    model::{
        Accept, BatchVerifyTokens, DEFAULT_PREFILL_CHUNK, Model, PackedPrefill, PrefillLogits,
        Sampling, VerifyTokens,
    },
    ops::attention::KvFormat,
    spec::{DraftSource, Drafts, Prompt, Speculator, Width, greedy_by_verify, load_draft},
    weights::{WeightFormat, Weights},
};

/// (first token, length) of each prompt in the capture.
const PROMPTS: [(usize, usize); 5] = [(0, 40), (100, 700), (0, 1100), (3000, 2500), (5000, 1023)];
/// Tokens each sequence generates after its first.
const STEPS: usize = 40;
/// The batched step at which each sequence starts decoding.
const JOINS: [usize; 5] = [0, 0, 3, 7, 12];
const CAPACITY: usize = 4096;

/// (first token, length, joining step) of `n` sequences: [`PROMPTS`] and [`JOINS`], then
/// more of the same lengths from other places in the capture, joining at steps 0 to 12.
fn sequences(n: usize) -> Vec<(usize, usize, usize)> {
    (0..n)
        .map(|i| {
            let (first, len) = PROMPTS[i % PROMPTS.len()];
            let round = i / PROMPTS.len();
            (
                first + 211 * round,
                len + 3 * round,
                (JOINS[i % JOINS.len()] + round) % 13,
            )
        })
        .collect()
}

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

fn load(kv: KvFormat, weights: WeightFormat) -> anyhow::Result<(Gpu, Model)> {
    let model_dir = PathBuf::from(env::var_os("SIPHON_MODEL").context("set SIPHON_MODEL")?);
    let gpu = Gpu::new(0)?;
    let config = TextConfig::load(&model_dir)?;
    let passes = config.loop_config.num_passes;
    let checkpoint = Checkpoint::open(&model_dir)?;
    let loaded = Weights::load(&checkpoint, &config, gpu.stream())?;
    let model = Model::with_formats(&gpu, config, loaded, passes, CAPACITY, kv, weights)?;
    Ok((gpu, model))
}

/// FNV-1a over the bits of a row of logits.
fn hash(row: &[f32]) -> u64 {
    row.iter().fold(0xcbf2_9ce4_8422_2325, |h, v| {
        (h ^ u64::from(v.to_bits())).wrapping_mul(0x0100_0000_01b3)
    })
}

/// The first token after `prompt` and the next `STEPS` tokens with their logits' hashes,
/// through one-row verifies in slot 0.
fn alone(model: &mut Model, prompt: &[u32]) -> anyhow::Result<(Vec<u32>, Vec<u64>)> {
    model.reset()?;
    let mut token = model.prefill(prompt, DEFAULT_PREFILL_CHUNK, PrefillLogits::Last)?;
    let mut tokens = vec![token];
    let mut hashes = Vec::new();
    for _ in 0..STEPS {
        let verdict = model.verify(VerifyTokens::Host(&[token]), Accept::Rows(1))?;
        hashes.push(hash(&model.verify_logits()?));
        token = verdict.next();
        tokens.push(token);
    }
    Ok((tokens, hashes))
}

fn batch_matches_alone(kv: KvFormat, weights: WeightFormat, n: usize) -> anyhow::Result<()> {
    let (_gpu, mut model) = load(kv, weights)?;
    let vocab = model.config().unpadded_vocab_size;
    let capture = capture_tokens()?;
    let sequences = sequences(n);
    let prompts: Vec<&[u32]> = sequences
        .iter()
        .map(|&(first, len, _)| &capture[first..first + len])
        .collect();
    let want = prompts
        .iter()
        .map(|p| alone(&mut model, p))
        .collect::<anyhow::Result<Vec<_>>>()?;

    model.add_slots(prompts.len() - 1)?;
    // Slot s holds prompt s, except that slots 0 and 1 swap, so slot 0 is not the first
    // row.
    let slot_of = |i: usize| match i {
        0 => 1,
        1 => 0,
        i => i,
    };
    let mut got: Vec<(Vec<u32>, Vec<u64>)> = Vec::new();
    for (i, prompt) in prompts.iter().enumerate() {
        let slot = slot_of(i);
        model.reset_slot(slot)?;
        let first = model.prefill_slot(slot, prompt, DEFAULT_PREFILL_CHUNK)?;
        ensure!(
            model.slot_position(slot)? == prompt.len(),
            "slot {slot} is at {}",
            model.slot_position(slot)?
        );
        got.push((vec![first], Vec::new()));
    }
    let mut step = 0;
    let mut widest = 0;
    while got.iter().any(|(tokens, _)| tokens.len() <= STEPS) {
        // Joined sequences that still need tokens, the later joiners first.
        let mut rows: Vec<usize> = (0..prompts.len())
            .filter(|&i| sequences[i].2 <= step && got[i].0.len() <= STEPS)
            .collect();
        widest = widest.max(rows.len());
        rows.reverse();
        let batch: Vec<(usize, u32)> = rows
            .iter()
            .map(|&i| (slot_of(i), *got[i].0.last().unwrap_or(&0)))
            .collect();
        let next = model.step_batch(&batch)?;
        let logits = model.batch_logits(rows.len())?;
        for (r, &i) in rows.iter().enumerate() {
            got[i].0.push(next[r]);
            got[i].1.push(hash(&logits[r * vocab..(r + 1) * vocab]));
        }
        step += 1;
    }
    for (i, (want, got)) in want.iter().zip(&got).enumerate() {
        let tokens_differ = want.0.iter().zip(&got.0).position(|(a, b)| a != b);
        let logits_differ = want.1.iter().zip(&got.1).position(|(a, b)| a != b);
        println!(
            "prompt {i} ({} tokens): first different token {tokens_differ:?}, first different logits {logits_differ:?}",
            prompts[i].len()
        );
        ensure!(
            tokens_differ.is_none() && logits_differ.is_none(),
            "prompt {i}: the batch differs from one-row verifies"
        );
    }
    println!("{n} sequences, at most {widest} rows per step");
    ensure!(widest == n, "the batch never held all {n} sequences");
    Ok(())
}

#[test]
#[ignore = "needs the checkpoint and a GPU"]
fn batch_matches_alone_fp8() -> anyhow::Result<()> {
    batch_matches_alone(KvFormat::Fp8, WeightFormat::Fp8, PROMPTS.len())
}

#[test]
#[ignore = "needs the checkpoint and a GPU"]
fn batch_matches_alone_nvfp4() -> anyhow::Result<()> {
    batch_matches_alone(KvFormat::Nvfp4, WeightFormat::Fp8, PROMPTS.len())
}

#[test]
#[ignore = "needs the checkpoint and a GPU"]
fn batch_matches_alone_16_nvfp4() -> anyhow::Result<()> {
    batch_matches_alone(KvFormat::Nvfp4, WeightFormat::Fp8, 16)
}

#[test]
#[ignore = "needs the checkpoint and a GPU"]
fn batch_matches_alone_mixed() -> anyhow::Result<()> {
    batch_matches_alone(KvFormat::Mixed, WeightFormat::Fp8, PROMPTS.len())
}

#[test]
#[ignore = "needs the checkpoint and a GPU"]
fn batch_matches_alone_bf16() -> anyhow::Result<()> {
    batch_matches_alone(KvFormat::Bf16, WeightFormat::Bf16, PROMPTS.len())
}

#[test]
#[ignore = "needs the checkpoint and a GPU"]
fn batch_matches_alone_16_fp8() -> anyhow::Result<()> {
    batch_matches_alone(KvFormat::Fp8, WeightFormat::Fp8, 16)
}

#[test]
#[ignore = "needs the checkpoint and a GPU"]
fn batch_matches_alone_16_bf16() -> anyhow::Result<()> {
    batch_matches_alone(KvFormat::Bf16, WeightFormat::Bf16, 16)
}

#[test]
#[ignore = "needs the checkpoint and a GPU"]
fn batch_matches_alone_32_fp8() -> anyhow::Result<()> {
    batch_matches_alone(KvFormat::Fp8, WeightFormat::Fp8, 32)
}

#[test]
#[ignore = "needs the checkpoint and a GPU"]
fn batch_matches_alone_32_bf16() -> anyhow::Result<()> {
    batch_matches_alone(KvFormat::Bf16, WeightFormat::Bf16, 32)
}

#[test]
#[ignore = "needs the checkpoint and a GPU"]
fn batch_matches_alone_64_fp8() -> anyhow::Result<()> {
    batch_matches_alone(KvFormat::Fp8, WeightFormat::Fp8, 64)
}

/// Each slot's tokens and logits' hashes for `STEPS` batched steps of slots `0..firsts.len()`
/// together, slot `s` from its first token `firsts[s]` with `samplings[s]`.
fn decode_slots(
    model: &mut Model,
    samplings: &[Sampling],
    firsts: &[u32],
) -> anyhow::Result<Vec<(Vec<u32>, Vec<u64>)>> {
    let vocab = model.config().unpadded_vocab_size;
    let mut out: Vec<(Vec<u32>, Vec<u64>)> =
        firsts.iter().map(|&t| (vec![t], Vec::new())).collect();
    for _ in 0..STEPS {
        let rows: Vec<(usize, u32)> = out
            .iter()
            .enumerate()
            .map(|(slot, (tokens, _))| (slot, *tokens.last().unwrap_or(&0)))
            .collect();
        let next = model.step_batch_sampled(&rows, samplings)?;
        let logits = model.batch_logits(rows.len())?;
        for (slot, (tokens, hashes)) in out.iter_mut().enumerate() {
            tokens.push(next[slot]);
            hashes.push(hash(&logits[slot * vocab..(slot + 1) * vocab]));
        }
    }
    Ok(out)
}

/// Choices of one prompt in `shared_prefill_matches_separate`.
const CHOICES: usize = 4;
/// The slot that prefills for the others, so that slot 0 is a copy.
const SOURCE: usize = 1;

#[test]
#[ignore = "needs the checkpoint and a GPU"]
fn shared_prefill_matches_separate() -> anyhow::Result<()> {
    let (_gpu, mut model) = load(KvFormat::Fp8, WeightFormat::Fp8)?;
    model.enable_sampling()?;
    model.add_slots(CHOICES - 1)?;
    let capture = capture_tokens()?;
    // Short, just under the sliding window, and past its wrap.
    for (first, len) in [(0, 40), (5000, 1023), (3000, 2500)] {
        let prompt = &capture[first..first + len];
        for (name, temperature, top_p) in [
            ("T=1", 1.0, 1.0),
            ("T=0.8", 0.8, 0.95),
            ("greedy", 0.0, 1.0),
        ] {
            // Choice i takes seed 7 + i, as the server derives them.
            let samplings = (0..CHOICES as u64)
                .map(|i| Sampling::new(temperature, 0, top_p, 0.0, 7 + i))
                .collect::<anyhow::Result<Vec<_>>>()?;

            // Each choice prefills the prompt in its own slot.
            let mut firsts = Vec::new();
            for (slot, &sampling) in samplings.iter().enumerate() {
                model.reset_slot(slot)?;
                model.set_sampling(sampling);
                firsts.push(model.prefill_slot(slot, prompt, DEFAULT_PREFILL_CHUNK)?);
            }
            let want = decode_slots(&mut model, &samplings, &firsts)?;

            // One prefill, copied into the other slots, which were reset as the server
            // resets a request's slot.
            for slot in 0..CHOICES {
                model.reset_slot(slot)?;
            }
            model.set_sampling(samplings[SOURCE]);
            let mut shared = vec![0; CHOICES];
            shared[SOURCE] = model.prefill_slot(SOURCE, prompt, DEFAULT_PREFILL_CHUNK)?;
            for slot in (0..CHOICES).filter(|&s| s != SOURCE) {
                model.move_slot(SOURCE, slot)?;
                shared[slot] = model.redraw_slot(slot, samplings[slot])?;
                ensure!(
                    model.slot_position(slot)? == len,
                    "slot {slot} is not at the prompt's end"
                );
            }
            ensure!(
                model.redraw_slot(SOURCE, samplings[SOURCE])? == shared[SOURCE],
                "{name}: a redraw with the prefill's own sampling differs from the prefill"
            );
            let got = decode_slots(&mut model, &samplings, &shared)?;

            for (slot, (want, got)) in want.iter().zip(&got).enumerate() {
                let tokens_differ = want.0.iter().zip(&got.0).position(|(a, b)| a != b);
                let logits_differ = want.1.iter().zip(&got.1).position(|(a, b)| a != b);
                println!(
                    "{len} tokens, {name}, choice {slot}: first different token {tokens_differ:?}, first different logits {logits_differ:?}"
                );
                ensure!(
                    tokens_differ.is_none() && logits_differ.is_none(),
                    "{len} tokens, {name}, choice {slot}: the copy differs from its own prefill"
                );
            }
            // Greedy choices agree; at T=1 the seeds must make them differ somewhere.
            let distinct = want.iter().any(|(tokens, _)| *tokens != want[0].0);
            ensure!(
                !(samplings[0].is_greedy() && distinct),
                "{name}: greedy choices differ"
            );
            ensure!(
                distinct || name != "T=1",
                "{name}: four seeds gave the same tokens"
            );
        }
    }
    Ok(())
}

/// Tokens of `prompt`'s greedy continuation through one-row verifies in slot 0, the first
/// token from the prefill included, `n` in all.
fn reference(model: &mut Model, prompt: &[u32], n: usize) -> anyhow::Result<Vec<u32>> {
    model.reset()?;
    let first = model.prefill(prompt, DEFAULT_PREFILL_CHUNK, PrefillLogits::Last)?;
    let mut tokens = vec![first];
    tokens.extend(greedy_by_verify(model, first, n - 1)?);
    Ok(tokens)
}

/// Sequence A speculates in slot 0, then runs in batched steps as row 1 beside sequence B
/// in slot 1 while the draft keeps A's taps, then speculates again. B then returns to
/// speculation too: the draft's state is rebuilt for it in slot 2 and B moves to slot 0.
/// Both must emit the tokens of one-row verifies.
fn speculation_returns_after_batch(drafter: Option<&std::path::Path>) -> anyhow::Result<()> {
    let (_gpu, mut model) = load(KvFormat::Fp8, WeightFormat::Fp8)?;
    let model_dir = PathBuf::from(env::var_os("SIPHON_MODEL").context("set SIPHON_MODEL")?);
    let checkpoint = Checkpoint::open(&model_dir)?;
    let capture = capture_tokens()?;
    let (a, b) = (&capture[100..800], &capture[3000..4100]);
    model.add_slots(2)?;
    model.set_batch_taps(true);
    let source = drafter.map_or(DraftSource::Bundled, DraftSource::Dflash);
    let draft = load_draft(
        &mut model,
        &checkpoint,
        source,
        CAPACITY,
        &DraftOptions::fast(),
    )?;
    let mut spec = Speculator::with_draft(&model, draft, Width::Fixed(7))?;

    model.reset()?;
    let mut got_a = vec![spec.prefill(&mut model, a, DEFAULT_PREFILL_CHUNK)?];
    got_a.extend(spec.generate(&mut model, 30, Drafts::Dspark)?);
    let before = spec.stats().tokens_per_round();
    spec.follow(&model);
    model.reset_slot(1)?;
    let mut got_b = vec![model.prefill_slot(1, b, DEFAULT_PREFILL_CHUNK)?];
    // More steps than the draft keeps before it commits, so that it commits mid-batch.
    for _ in 0..STEPS {
        let last = |t: &[u32]| t.last().copied().unwrap_or_default();
        let next = model.step_batch(&[(1, last(&got_b)), (0, last(&got_a))])?;
        spec.keep_batched(&model, 1, next[1])?;
        got_b.push(next[0]);
        got_a.push(next[1]);
    }
    spec.resume(&mut model, got_a.last().copied().unwrap_or_default())?;
    got_a.extend(spec.generate(&mut model, 30, Drafts::Dspark)?);
    let after = spec.stats().tokens_per_round();

    let fed: Vec<u32> = b.iter().chain(&got_b[..got_b.len() - 1]).copied().collect();
    spec.rebuild(&mut model, 2, 1, Prompt::text(&fed))?;
    model.move_slot(1, 0)?;
    spec.resume(&mut model, got_b.last().copied().unwrap_or_default())?;
    got_b.extend(spec.generate(&mut model, 30, Drafts::Dspark)?);
    let rebuilt = spec.stats().tokens_per_round();
    println!(
        "tokens per round: A {before:.2} before the batch, {after:.2} after; B {rebuilt:.2} after the rebuild"
    );

    for (name, prompt, got) in [("A", a, &got_a), ("B", b, &got_b)] {
        let want = reference(&mut model, prompt, got.len())?;
        let differ = want.iter().zip(got).position(|(w, g)| w != g);
        println!(
            "{name}: {} tokens, first different token {differ:?}",
            got.len()
        );
        ensure!(differ.is_none(), "{name} differs from one-row verifies");
    }
    Ok(())
}

#[test]
#[ignore = "needs the checkpoint and a GPU"]
fn speculation_returns_after_batch_bundled() -> anyhow::Result<()> {
    speculation_returns_after_batch(None)
}

#[test]
#[ignore = "needs the checkpoint and a GPU"]
fn speculation_returns_after_batch_dflash() -> anyhow::Result<()> {
    let Some(dir) = env::var_os("SIPHON_DFLASH") else {
        println!("SIPHON_DFLASH is not set; skipped");
        return Ok(());
    };
    speculation_returns_after_batch(Some(std::path::Path::new(&dir)))
}

struct RoundPlan {
    width: usize,
    corrupt: [Option<usize>; 5],
    accept: [Accept; 5],
}

struct RoundExpectation {
    accepted: usize,
    emitted: Vec<u32>,
    hashes: Vec<u64>,
}

/// Bit-exact parity for `Model::verify_batch`: across multiple sequences spanning the
/// sliding-window wrap, widths 1, 4 and 8, mixed greedy/sampled acceptance (0 accepts,
/// partial accepts, full accepts, and forced `Accept::Rows`), every sequence's verdicts,
/// logit hashes, post-rollback caches and subsequent tokens must equal single-sequence
/// `Model::verify`.
#[expect(clippy::too_many_lines, reason = "end-to-end multi-round parity check")]
fn batch_verify_matches_alone(kv: KvFormat, weights: WeightFormat) -> anyhow::Result<()> {
    let (_gpu, mut model) = load(kv, weights)?;
    model.enable_sampling()?;
    let vocab = model.config().unpadded_vocab_size;
    let capture = capture_tokens()?;
    let prompts: Vec<&[u32]> = PROMPTS
        .iter()
        .map(|&(first, len)| &capture[first..first + len])
        .collect();
    let n = prompts.len();
    let samplings: Vec<Sampling> = (0..n)
        .map(|i| {
            if i % 2 == 0 {
                Ok(Sampling::GREEDY)
            } else {
                Sampling::new(0.8, 0, 0.95, 0.0, 100 + i as u64)
            }
        })
        .collect::<anyhow::Result<_>>()?;

    // Reference trajectories via single-sequence Model::verify in slot 0:
    // First collect ground-truth continuation for each sequence so we can construct exact,
    // partial, and wrong draft blocks.
    let mut truth: Vec<Vec<u32>> = Vec::with_capacity(n);
    for (i, prompt) in prompts.iter().enumerate() {
        model.reset()?;
        model.set_sampling(samplings[i]);
        let mut tok = model.prefill(prompt, DEFAULT_PREFILL_CHUNK, PrefillLogits::Last)?;
        let mut seq = vec![tok];
        for _ in 0..32 {
            let v = model.verify(VerifyTokens::Host(&[tok]), Accept::Rows(1))?;
            tok = v.next();
            seq.push(tok);
        }
        truth.push(seq);
    }

    // Plan of (width, accept_fn, corrupt_index_per_seq):
    // Round 0: width 8, Greedy — seq 0 all match (7 accepted), seq 1 partial (3 accepted),
    //          seq 2 0 accepted (full conv+ring rollback), seq 3 partial (1 accepted),
    //          seq 4 all match (7 accepted, crosses 1024 sliding window!).
    // Round 1: width 4, mixed Greedy & Accept::Rows(2) to test explicit rollback.
    // Round 2: width 1, Greedy (verifies width=1 path & post-rollback cache state).
    // Round 3: width 8, Greedy with partial/full rollbacks again (replays captured graph).
    let plans = [
        RoundPlan {
            width: 8,
            corrupt: [None, Some(4), Some(1), Some(2), None],
            accept: [Accept::Greedy; 5],
        },
        RoundPlan {
            width: 4,
            corrupt: [Some(2), None, Some(3), None, Some(1)],
            accept: [
                Accept::Greedy,
                Accept::Rows(2),
                Accept::Greedy,
                Accept::Rows(3),
                Accept::Greedy,
            ],
        },
        RoundPlan {
            width: 1,
            corrupt: [None; 5],
            accept: [Accept::Greedy; 5],
        },
        RoundPlan {
            width: 8,
            corrupt: [Some(5), Some(1), None, Some(6), Some(3)],
            accept: [Accept::Greedy; 5],
        },
    ];

    // Run reference single-sequence verifies following the exact same round plans.
    let mut want_verdicts: Vec<Vec<RoundExpectation>> = (0..n).map(|_| Vec::new()).collect();
    for (i, prompt) in prompts.iter().enumerate() {
        model.reset()?;
        model.set_sampling(samplings[i]);
        let _first = model.prefill(prompt, DEFAULT_PREFILL_CHUNK, PrefillLogits::Last)?;
        let mut cursor = 0usize;
        for plan in &plans {
            let mut toks = truth[i][cursor..cursor + plan.width].to_vec();
            if let Some(bad) = plan.corrupt[i] {
                toks[bad] = (toks[bad] + 17) % (vocab as u32);
            }
            let v = model.verify(VerifyTokens::Host(&toks), plan.accept[i])?;
            let logits = model.verify_logits()?;
            let hashes: Vec<u64> = (0..plan.width)
                .map(|j| hash(&logits[j * vocab..(j + 1) * vocab]))
                .collect();
            cursor += v.accepted + 1;
            want_verdicts[i].push(RoundExpectation {
                accepted: v.accepted,
                emitted: v.emitted(),
                hashes,
            });
        }
    }

    // Now run all 5 sequences concurrently via Model::verify_batch in shuffled slot order.
    model.add_slots(n - 1)?;
    let slot_of = |i: usize| match i {
        0 => 2,
        2 => 0,
        other => other,
    };
    let mut cursors = vec![0usize; n];
    for (i, prompt) in prompts.iter().enumerate() {
        let slot = slot_of(i);
        model.reset_slot(slot)?;
        model.set_sampling(samplings[i]);
        let _first = model.prefill_slot(slot, prompt, DEFAULT_PREFILL_CHUNK)?;
    }

    let slots: Vec<usize> = (0..n).map(slot_of).collect();
    for (r, plan) in plans.iter().enumerate() {
        let mut buf: Vec<Vec<u32>> = Vec::with_capacity(n);
        for i in 0..n {
            let mut toks = truth[i][cursors[i]..cursors[i] + plan.width].to_vec();
            if let Some(bad) = plan.corrupt[i] {
                toks[bad] = (toks[bad] + 17) % (vocab as u32);
            }
            buf.push(toks);
        }
        let slices: Vec<&[u32]> = buf.iter().map(Vec::as_slice).collect();
        let verdicts = model.verify_batch(
            &slots,
            BatchVerifyTokens::Host(&slices),
            &plan.accept,
            &samplings,
        )?;
        let logits = model.batch_verify_logits()?;
        for (i, v) in verdicts.iter().enumerate() {
            let RoundExpectation {
                accepted: want_acc,
                emitted: want_emit,
                hashes: want_hashes,
            } = &want_verdicts[i][r];
            let got_hashes: Vec<u64> = (0..plan.width)
                .map(|j| {
                    let row = i * plan.width + j;
                    hash(&logits[row * vocab..(row + 1) * vocab])
                })
                .collect();
            println!(
                "round {r} (W={}) seq {i} (slot {}): accepted={} (want {}), emitted_match={}, logits_match={}",
                plan.width,
                slots[i],
                v.accepted,
                want_acc,
                v.emitted() == *want_emit,
                got_hashes == *want_hashes,
            );
            ensure!(
                v.accepted == *want_acc,
                "round {r} seq {i}: accepted {} != {want_acc}",
                v.accepted
            );
            ensure!(
                v.emitted() == *want_emit,
                "round {r} seq {i}: emitted {:?} != {want_emit:?}",
                v.emitted()
            );
            ensure!(
                got_hashes == *want_hashes,
                "round {r} seq {i}: logits hashes differ from single-sequence verify"
            );
            cursors[i] += v.accepted + 1;
            ensure!(
                model.slot_position(slots[i])? == prompts[i].len() + cursors[i],
                "round {r} seq {i}: slot position mismatch"
            );
        }
    }
    Ok(())
}

#[test]
#[ignore = "needs the checkpoint and a GPU"]
fn batch_verify_matches_alone_fp8() -> anyhow::Result<()> {
    batch_verify_matches_alone(KvFormat::Fp8, WeightFormat::Fp8)
}

#[test]
#[ignore = "needs the checkpoint and a GPU"]
fn batch_verify_matches_alone_nvfp4() -> anyhow::Result<()> {
    batch_verify_matches_alone(KvFormat::Nvfp4, WeightFormat::Fp8)
}

/// End-to-end bit-exact parity for Batched DFlash 2 + `Model::verify_batch`:
/// 5 sequences spanning the sliding-window wrap (mixed greedy + coupled sampling) decode
/// concurrently with DFlash 2 (`DraftOptions::fast()`, FP8 KV + FP8 backbone + shortlist)
/// and `Model::verify_batch`, and must emit 100% identical tokens to single-sequence
/// `Model::verify`.
#[test]
#[ignore = "needs the checkpoint, SIPHON_DFLASH (DFlash 2), and a GPU"]
#[expect(clippy::too_many_lines, reason = "end-to-end batched DFlash 2 parity test")]
fn batch_dflash2_matches_alone() -> anyhow::Result<()> {
    let Some(dflash_dir) = env::var_os("SIPHON_DFLASH") else {
        println!("SIPHON_DFLASH is not set; skipped");
        return Ok(());
    };
    let (_gpu, mut model) = load(KvFormat::Fp8, WeightFormat::Fp8)?;
    model.enable_sampling()?;
    let model_dir = PathBuf::from(env::var_os("SIPHON_MODEL").context("set SIPHON_MODEL")?);
    let checkpoint = Checkpoint::open(&model_dir)?;
    let capture = capture_tokens()?;
    let prompts: Vec<&[u32]> = PROMPTS
        .iter()
        .map(|&(first, len)| &capture[first..first + len])
        .collect();
    let n = prompts.len();
    let samplings: Vec<Sampling> = (0..n)
        .map(|i| {
            if i % 2 == 0 {
                Ok(Sampling::GREEDY)
            } else {
                Sampling::new(0.8, 0, 0.95, 0.0, 200 + i as u64)
            }
        })
        .collect::<anyhow::Result<_>>()?;
    let gen_tokens = 64usize;

    let mut draft = load_draft(
        &mut model,
        &checkpoint,
        DraftSource::Dflash(std::path::Path::new(&dflash_dir)),
        CAPACITY,
        &DraftOptions::fast(),
    )?;
    ensure!(draft.is_dflash2(), "SIPHON_DFLASH must be a DFlash 2 drafter");
    let linear = siphon_cuda::ops::linear::Linear::load(model.stream().context())?;

    // Ground truth via single-sequence one-row verifies in slot 0.
    let mut want: Vec<Vec<u32>> = Vec::with_capacity(n);
    for (i, prompt) in prompts.iter().enumerate() {
        model.reset()?;
        model.set_sampling(samplings[i]);
        let mut tok = model.prefill(prompt, DEFAULT_PREFILL_CHUNK, PrefillLogits::Last)?;
        let mut seq = vec![tok];
        while seq.len() <= gen_tokens {
            let v = model.verify(VerifyTokens::Host(&[tok]), Accept::Rows(1))?;
            tok = v.next();
            seq.push(tok);
        }
        want.push(seq);
    }

    // Prefill all 5 slots with taps committed into each slot's DFlash 2 rings.
    model.add_slots(n - 1)?;
    let slots: Vec<usize> = vec![2, 1, 0, 3, 4];
    let mut got: Vec<Vec<u32>> = Vec::with_capacity(n);
    let mut anchors = Vec::with_capacity(n);
    let stream = std::sync::Arc::clone(model.stream());
    for (i, &prompt) in prompts.iter().enumerate() {
        let slot = slots[i];
        model.reset_slot(slot)?;
        draft.reset_slot(slot);
        model.set_sampling(samplings[i]);
        let prompt_i32: Vec<i32> = prompt
            .iter()
            .map(|&t| Ok(i32::try_from(t)?))
            .collect::<anyhow::Result<_>>()?;
        draft.note_tokens(&stream, model.draft_shared(), &prompt_i32)?;
        let first = model.prefill_slot_with_taps(
            slot,
            prompt,
            DEFAULT_PREFILL_CHUNK,
            &mut |pos, taps| draft.commit_slot(&stream, &linear, slot, &taps, pos),
        )?;
        got.push(vec![first]);
        anchors.push(first);
    }

    let draft_steps = 7usize;
    let width = draft_steps + 1;
    let accepts = vec![Accept::Greedy; n];
    draft.batch_draft_initial(
        &stream,
        model.draft_shared(),
        &slots,
        &anchors,
        &samplings,
        draft_steps,
    )?;

    let mut rounds = 0usize;
    let mut total_emitted = 0usize;
    let started = std::time::Instant::now();
    while got.iter().any(|g| g.len() <= gen_tokens) {
        let starts: Vec<usize> = slots
            .iter()
            .map(|&s| model.slot_position(s))
            .collect::<anyhow::Result<_>>()?;
        let pending = model.start_verify_batch(
            &slots,
            BatchVerifyTokens::Device {
                tokens: draft.batch_verify_tokens(n, width)?,
                width,
            },
            &accepts,
            &samplings,
        )?;
        draft.batch_commit_and_draft(
            &stream,
            model.draft_shared(),
            &slots,
            &starts,
            &model.batch_verify_taps()?,
            &model.batch_verify_results()?,
            &model.batch_verify_anchors()?,
            &samplings,
            width,
            draft_steps,
        )?;
        let verdicts = model.finish_verify_batch(pending)?;
        rounds += 1;
        for (i, v) in verdicts.iter().enumerate() {
            let emitted = v.emitted();
            total_emitted += emitted.len();
            draft.set_slot_committed(slots[i], starts[i] + v.accepted + 1)?;
            got[i].extend_from_slice(&emitted);
        }
    }
    let elapsed = started.elapsed().as_secs_f64();
    println!(
        "batched DFlash 2 ({n} seqs, W={width}): {rounds} rounds, {total_emitted} tokens ({:.2} tok/round/seq, {:.1} agg tok/s)",
        total_emitted as f64 / (rounds * n) as f64,
        total_emitted as f64 / elapsed,
    );
    for i in 0..n {
        let len = want[i].len().min(got[i].len());
        let differ = want[i][..len]
            .iter()
            .zip(&got[i][..len])
            .position(|(a, b)| a != b);
        println!(
            "seq {i} (slot {}, {}): checked {len} tokens, first diff {differ:?}",
            slots[i],
            if samplings[i].is_greedy() {
                "greedy"
            } else {
                "sampled"
            }
        );
        ensure!(
            differ.is_none(),
            "seq {i}: batched DFlash 2 differs from single-sequence verify"
        );
    }
    Ok(())
}

/// Times `Model::verify_batch` vs `Draft::batch_commit_and_draft` across `(B, W)` shapes
/// on real book prompts so we can inspect verify/draft breakdown and kernel hotspots.
#[test]
#[ignore = "needs the checkpoint, SIPHON_DFLASH (DFlash 2), and a GPU"]
#[expect(clippy::too_many_lines, reason = "multi-shape timing table")]
fn batch_dflash2_profile() -> anyhow::Result<()> {
    let Some(dflash_dir) = env::var_os("SIPHON_DFLASH") else {
        println!("SIPHON_DFLASH is not set; skipped");
        return Ok(());
    };
    let (_gpu, mut model) = load(KvFormat::Fp8, WeightFormat::Fp8)?;
    let fp8_dense = env::var("SIPHON_PREFILL_DENSE").is_ok_and(|v| v == "fp8");
    if fp8_dense {
        model.set_prefill_dense(siphon_cuda::ops::Products::Fp8)?;
    }
    let model_dir = PathBuf::from(env::var_os("SIPHON_MODEL").context("set SIPHON_MODEL")?);
    let checkpoint = Checkpoint::open(&model_dir)?;
    let mut draft = load_draft(
        &mut model,
        &checkpoint,
        DraftSource::Dflash(std::path::Path::new(&dflash_dir)),
        2048,
        &DraftOptions::fast(),
    )?;
    if fp8_dense {
        draft.set_fp8_dense(true)?;
    }
    let stream = std::sync::Arc::clone(model.stream());
    let linear = siphon_cuda::ops::linear::Linear::load(stream.context())?;
    let max_b = 32usize;
    model.add_slots(max_b - 1)?;
    model.reserve_batch_verify(max_b)?;
    draft.ensure_batch_dflash2(&stream, max_b, max_b - 1)?;
    let prompt_base: Vec<u32> = (0..256u32).map(|j| 200 + (j * 7) % 500).collect();
    let mut anchors_all = Vec::with_capacity(max_b);
    for slot in 0..max_b {
        model.reset_slot(slot)?;
        draft.reset_slot(slot);
        let p: Vec<u32> = prompt_base
            .iter()
            .enumerate()
            .map(|(j, &t)| t + u32::try_from((slot * 37 + j) % 13).unwrap_or(0))
            .collect();
        let mut commit = |start: usize, taps: cudarc::driver::CudaView<'_, half::bf16>| {
            draft.commit_slot(&stream, &linear, slot, &taps, start)
        };
        let first = model.prefill_slot_with_taps(slot, &p, 256, &mut commit)?;
        anchors_all.push(first);
    }
    let shapes = [
        (1, 8),
        (8, 8),
        (8, 4),
        (16, 8),
        (16, 6),
        (16, 4),
        (32, 8),
        (32, 6),
        (32, 5),
        (32, 4),
        (32, 3),
        (32, 2),
    ];
    let only_32_8 = env::var_os("PROFILE_32_8_ONLY").is_some();
    for &(b, w) in &shapes {
        if only_32_8 && (b, w) != (32, 8) {
            continue;
        }
        let slots: Vec<usize> = (0..b).collect();
        let samplings = vec![Sampling::GREEDY; b];
        let accept = vec![Accept::Greedy; b];
        let mut anchors = anchors_all[..b].to_vec();
        // Reset committed & model pos via quick re-anchor isn't needed; just advance 15 rounds.
        draft.batch_draft_initial(
            &stream,
            model.draft_shared(),
            &slots,
            &anchors,
            &samplings,
            w - 1,
        )?;
        let mut verify_ms = Vec::new();
        let mut draft_ms = Vec::new();
        let mut tot_ms = Vec::new();
        let mut emitted_sum = 0usize;
        let iters = if only_32_8 { 12 } else { 10 };
        for iter in 0..iters {
            let starts: Vec<usize> = slots
                .iter()
                .map(|&s| model.slot_position(s))
                .collect::<anyhow::Result<_>>()?;
            let tokens = draft.batch_verify_tokens(b, w)?;
            let ev0 = stream.record_event(Some(
                cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT,
            ))?;
            let pending = model.start_verify_batch(
                &slots,
                BatchVerifyTokens::Device { tokens, width: w },
                &accept,
                &samplings,
            )?;
            let ev1 = stream.record_event(Some(
                cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT,
            ))?;
            draft.batch_commit_and_draft(
                &stream,
                model.draft_shared(),
                &slots,
                &starts,
                &model.batch_verify_taps()?,
                &model.batch_verify_results()?,
                &model.batch_verify_anchors()?,
                &samplings,
                w,
                w - 1,
            )?;
            let ev2 = stream.record_event(Some(
                cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT,
            ))?;
            let verdicts = model.finish_verify_batch(pending)?;
            for (i, v) in verdicts.iter().enumerate() {
                draft.set_slot_committed(slots[i], starts[i] + v.accepted + 1)?;
                anchors[i] = v.next();
                if iter >= 3 {
                    emitted_sum += v.emitted().len();
                }
            }
            if iter >= 3 {
                verify_ms.push(f64::from(ev0.elapsed_ms(&ev1)?));
                draft_ms.push(f64::from(ev1.elapsed_ms(&ev2)?));
                tot_ms.push(f64::from(ev0.elapsed_ms(&ev2)?));
            }
        }
        verify_ms.sort_by(f64::total_cmp);
        draft_ms.sort_by(f64::total_cmp);
        tot_ms.sort_by(f64::total_cmp);
        let mid = tot_ms.len() / 2;
        let tok_r = emitted_sum as f64 / ((iters - 3) * b) as f64;
        println!(
            "B={b:2} W={w} (M={:3}): verify={:6.2} ms  draft={:5.2} ms  total={:6.2} ms  ms/row={:.3}  (tok/r={tok_r:.2})",
            b * w,
            verify_ms[mid],
            draft_ms[mid],
            tot_ms[mid],
            tot_ms[mid] / (b * w) as f64,
        );
    }
    Ok(())
}

/// Parity check for `Model::prefill_packed`: packing 6 prompts of mixed lengths (spanning
/// both sides of the sliding-window boundary and mixed greedy/sampled first tokens) into one
/// 2048-row prefill pass must match separate `prefill_slot` calls on first tokens and
/// subsequent batched decode trajectories.
#[test]
#[ignore = "needs the checkpoint and a GPU"]
#[expect(clippy::too_many_lines, reason = "end-to-end packed prefill parity test")]
fn packed_prefill_matches_separate_fp8() -> anyhow::Result<()> {
    let (_gpu, mut model) = load(KvFormat::Fp8, WeightFormat::Fp8)?;
    model.enable_sampling()?;
    let capture = capture_tokens()?;
    let specs: [(usize, usize, Sampling); 6] = [
        (0, 64, Sampling::GREEDY),
        (200, 256, Sampling::GREEDY),
        (600, 310, Sampling::new(0.8, 0, 0.95, 0.0, 42)?),
        (1000, 1050, Sampling::GREEDY),
        (2200, 180, Sampling::new(0.7, 40, 0.9, 0.0, 99)?),
        (3000, 188, Sampling::GREEDY),
    ];
    let n = specs.len();
    model.add_slots(n - 1)?;

    // Reference: prefill each slot separately, then check first token + 1-step batched logits.
    let vocab = model.config().unpadded_vocab_size;
    let mut want_first = Vec::with_capacity(n);
    let mut want_logits = Vec::with_capacity(n);
    for (slot, &(first, len, sampling)) in specs.iter().enumerate() {
        model.reset_slot(slot)?;
        model.set_sampling(sampling);
        let tok = model.prefill_slot(slot, &capture[first..first + len], DEFAULT_PREFILL_CHUNK)?;
        let _ = model.step_batch_sampled(&[(slot, tok)], &[sampling])?;
        want_first.push(tok);
        want_logits.push(model.batch_logits(1)?);
    }

    // 1. Single-slice packed prefill (M=64) must match separate `prefill_slot` (M=64) bit-for-bit.
    model.reset_slot(0)?;
    let one = model.prefill_packed(
        &[PackedPrefill {
            slot: 0,
            tokens: &capture[specs[0].0..specs[0].0 + specs[0].1],
            images: Vec::new(),
            need_first_token: true,
            sampling: specs[0].2,
        }],
        DEFAULT_PREFILL_CHUNK,
    )?;
    let one_tok = one[0].context("missing first token for single packed slice")?;
    let _ = model.step_batch_sampled(&[(0, one_tok)], &[specs[0].2])?;
    let one_logits = model.batch_logits(1)?;
    let one_max_diff = one_logits
        .iter()
        .zip(&want_logits[0])
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!(
        "single-slice packed (len=64): first want={} got={} max_diff={one_max_diff:.6}",
        want_first[0], one_tok
    );
    ensure!(
        one_tok == want_first[0] && one_max_diff == 0.0,
        "single-slice packed prefill differed from prefill_slot: max_diff={one_max_diff}"
    );

    // 2. Packed: reset all slots and prefill all 6 prompts (sum = 2048 rows) in ONE pass.
    //    cuBLASLt picks M-dependent GEMM tiles between M=64..1050 and M=2048, so step-0 logits
    //    have small BF16 rounding differences while all first tokens stay identical.
    for slot in 0..n {
        model.reset_slot(slot)?;
    }
    let slices: Vec<PackedPrefill<'_>> = specs
        .iter()
        .enumerate()
        .map(|(slot, &(first, len, sampling))| PackedPrefill {
            slot,
            tokens: &capture[first..first + len],
            images: Vec::new(),
            need_first_token: true,
            sampling,
        })
        .collect();
    let firsts = model.prefill_packed(&slices, DEFAULT_PREFILL_CHUNK)?;
    let toks: Vec<u32> = firsts
        .into_iter()
        .enumerate()
        .map(|(i, t)| t.with_context(|| format!("missing first token for slice {i}")))
        .collect::<anyhow::Result<_>>()?;
    let samplings: Vec<Sampling> = specs.iter().map(|s| s.2).collect();
    let rows: Vec<(usize, u32)> = toks.iter().copied().enumerate().collect();
    let _ = model.step_batch_sampled(&rows, &samplings)?;
    let got_logits = model.batch_logits(n)?;
    for i in 0..n {
        let got_row = &got_logits[i * vocab..(i + 1) * vocab];
        let want_row = &want_logits[i];
        let mut max_diff = 0.0f32;
        let mut sum_diff = 0.0f64;
        for (&g, &w) in got_row.iter().zip(want_row) {
            let d = (g - w).abs();
            max_diff = max_diff.max(d);
            sum_diff += f64::from(d);
        }
        let mean_diff = sum_diff / vocab as f64;
        println!(
            "slice {i} (len={:4}): first want={} got={}  step0 logits max_diff={max_diff:.4} mean_diff={mean_diff:.5}",
            specs[i].1, want_first[i], toks[i]
        );
        ensure!(
            toks[i] == want_first[i],
            "packed prefill slice {i} first token diverged: got {}, want {}",
            toks[i],
            want_first[i]
        );
        ensure!(
            max_diff < 3.0 && mean_diff < 0.6,
            "packed prefill slice {i} logits drifted too far: max_diff={max_diff:.4}, mean_diff={mean_diff:.5}"
        );
    }
    Ok(())
}

/// Bit-exact GPU parity test for the 1-block PDL `pack_verify_tokens_batch` kernel against
/// host `[seqs, VERIFY_ROWS] -> [seqs, width]` repacking across all valid shapes.
#[test]
fn pack_verify_tokens_batch_matches_host() -> anyhow::Result<()> {
    use siphon_cuda::ops::spec::{Spec, VERIFY_ROWS};

    let gpu = Gpu::new(0)?;
    let stream = gpu.stream();
    let spec = Spec::load(gpu.context())?;

    for seqs in [1usize, 2, 3, 7, 16, 32] {
        let mut src_host = Vec::with_capacity(seqs * VERIFY_ROWS);
        for s in 0..seqs {
            for col in 0..VERIFY_ROWS {
                src_host.push(i32::try_from(s * 100 + col * 7 + 13)?);
            }
        }
        let src_dev = stream.clone_htod(&src_host)?;
        let mut dst_dev = stream.alloc_zeros::<i32>(seqs * VERIFY_ROWS)?;

        for width in [1usize, 2, 4, 6, 8] {
            if seqs * width > 256 {
                continue;
            }
            spec.pack_verify_tokens_batch(
                stream,
                &src_dev,
                &mut dst_dev.slice_mut(..seqs * width),
                seqs,
                width,
            )?;
            let got = stream.clone_dtoh(&dst_dev.slice(..seqs * width))?;
            let mut want = Vec::with_capacity(seqs * width);
            for s in 0..seqs {
                want.extend_from_slice(&src_host[s * VERIFY_ROWS..s * VERIFY_ROWS + width]);
            }
            ensure!(
                got == want,
                "pack_verify_tokens_batch mismatch for seqs={seqs}, width={width}: got={got:?}, want={want:?}"
            );
        }
    }
    Ok(())
}

/// Verifies that adding > 2048 unique context tokens via `Draft::note_tokens` (crossing
/// multiple old 1024-row bucket boundaries) and calling `Draft::reset` never invalidate
/// captured single-stream or batched DFlash 2 CUDA graphs, and that `warmup_batch_spec` +
/// zero-sync overlapped `speculate_batch_round` remain bit-exact against single-sequence
/// `Model::verify`.
#[test]
#[ignore = "needs the checkpoint, SIPHON_DFLASH (DFlash 2), and a GPU"]
fn shortlist_note_tokens_preserves_graphs() -> anyhow::Result<()> {
    let Some(dflash_dir) = env::var_os("SIPHON_DFLASH") else {
        println!("SIPHON_DFLASH is not set; skipped");
        return Ok(());
    };
    let (_gpu, mut model) = load(KvFormat::Fp8, WeightFormat::Fp8)?;
    let model_dir = PathBuf::from(env::var_os("SIPHON_MODEL").context("set SIPHON_MODEL")?);
    let checkpoint = Checkpoint::open(&model_dir)?;
    let capture = capture_tokens()?;
    let prompt = &capture[..256];
    let mut draft = load_draft(
        &mut model,
        &checkpoint,
        DraftSource::Dflash(std::path::Path::new(&dflash_dir)),
        CAPACITY,
        &DraftOptions::fast(),
    )?;
    ensure!(draft.is_dflash2(), "SIPHON_DFLASH must be a DFlash 2 drafter");
    let stream = std::sync::Arc::clone(model.stream());
    let linear = siphon_cuda::ops::linear::Linear::load(stream.context())?;

    // Capture a single-stream round graph and a batched draft graph.
    model.add_slots(1)?;
    let first = model.prefill_slot_with_taps(0, prompt, DEFAULT_PREFILL_CHUNK, &mut |pos, taps| {
        draft.commit_slot(&stream, &linear, 0, &taps, pos)
    })?;
    let first1 =
        model.prefill_slot_with_taps(1, prompt, DEFAULT_PREFILL_CHUNK, &mut |pos, taps| {
            draft.commit_slot(&stream, &linear, 1, &taps, pos)
        })?;
    let anchor = stream.clone_htod(&[i32::try_from(first)?])?;
    let empty_taps = stream.alloc_zeros::<half::bf16>(0)?;
    draft.set_round_inputs(&stream, &empty_taps, model.position(), &anchor)?;
    draft.replay_round(&stream, &linear, model.draft_shared(), 7)?;
    draft.batch_draft_initial(
        &stream,
        model.draft_shared(),
        &[0, 1],
        &[first, first1],
        &[Sampling::GREEDY; 2],
        7,
    )?;
    let before = draft.cached_graph_counts();
    ensure!(
        before.0 >= 1 && before.1 >= 1,
        "expected both single and batched draft graphs captured, got {before:?}"
    );

    // Note > 2048 unique token ids (well beyond the old 1024-row bucket boundary).
    let extra: Vec<i32> = (50_000..52_500).collect();
    draft.note_tokens(&stream, model.draft_shared(), &extra)?;
    let after_note = draft.cached_graph_counts();
    ensure!(
        after_note == before,
        "note_tokens invalidated draft CUDA graphs: before={before:?}, after={after_note:?}"
    );

    // Resetting the shortlist must also preserve captured graphs.
    draft.reset();
    let after_reset = draft.cached_graph_counts();
    ensure!(
        after_reset == before,
        "reset invalidated draft CUDA graphs: before={before:?}, after={after_reset:?}"
    );
    Ok(())
}

/// Parity check for wide FP8 dense tensor-core dispatch (`m >= 48`) + MXFP8/BF16-TC `W_o`
/// across `step_batch` (`B=32`) and `verify_batch` (`B=32 W=8` `M=256`):
/// verifies CUDA graph capture/replay and tight KL / top-1 agreement against default
/// `--prefill-dense bf16` (`LinearRows`).
#[test]
#[ignore = "needs the checkpoint (SIPHON_MODEL) and a GPU"]
#[expect(clippy::too_many_lines, reason = "end-to-end B=32 decode and B=32 W=8 verify parity")]
fn batch_wide_fp8_dense_and_wo_parity() -> anyhow::Result<()> {
    use siphon_cuda::ops::Products;

    fn kl_div(want: &[f32], got: &[f32]) -> f64 {
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

    if !siphon_cuda::BLOCK_SCALED_MMA {
        println!("skipped: FP8 dense GEMM needs block-scaled MMA, which sm_89 lacks");
        return Ok(());
    }

    let (_gpu, mut model) = load(KvFormat::Fp8, WeightFormat::Fp8)?;
    let vocab = model.config().unpadded_vocab_size;
    let capture = capture_tokens()?;
    let b = 32usize;
    let w = 8usize;
    model.add_slots(b - 1)?;
    let slots: Vec<usize> = (0..b).collect();
    let samplings = vec![Sampling::GREEDY; b];
    let accepts = vec![Accept::Rows(1); b];

    // Prefill 32 short prompts (64 tokens each) and record baseline step_batch + verify_batch.
    let reset_and_prefill = |model: &mut Model| -> anyhow::Result<Vec<u32>> {
        let mut firsts = Vec::with_capacity(b);
        for slot in 0..b {
            model.reset_slot(slot)?;
            let start = (slot * 29) % (capture.len() - 96);
            let tok = model.prefill_slot(slot, &capture[start..start + 64], 64)?;
            firsts.push(tok);
        }
        Ok(firsts)
    };

    // Baseline (`--prefill-dense bf16`).
    model.set_prefill_dense(Products::Bf16)?;
    let firsts = reset_and_prefill(&mut model)?;
    let step_rows: Vec<(usize, u32)> = firsts.iter().copied().enumerate().collect();
    let want_step = model.step_batch(&step_rows)?;
    let want_step_logits = model.batch_logits(b)?;

    let verify_toks: Vec<Vec<u32>> = (0..b)
        .map(|s| {
            let base = (s * 29) % (capture.len() - 96) + 65;
            let mut row = vec![want_step[s]];
            row.extend_from_slice(&capture[base..base + w - 1]);
            row
        })
        .collect();
    let verify_slices: Vec<&[u32]> = verify_toks.iter().map(Vec::as_slice).collect();
    let _ = model.verify_batch(
        &slots,
        BatchVerifyTokens::Host(&verify_slices),
        &accepts,
        &samplings,
    )?;
    let want_verify_logits = model.batch_verify_logits()?;

    // Compare both MXFP8 W_o and BF16-TC W_o under `--prefill-dense fp8`.
    let argmax = |row: &[f32]| -> usize {
        row.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map_or(0, |(i, _)| i)
    };
    let want_verify_top1: Vec<usize> = (0..b * w)
        .map(|r| argmax(&want_verify_logits[r * vocab..(r + 1) * vocab]))
        .collect();
    for (label, wo_mode) in [("mxfp8-wo", Products::Fp8), ("bf16-tc-wo", Products::Bf16)] {
        model.set_prefill_dense(Products::Fp8)?;
        model.set_prefill_dense_output(wo_mode)?;
        let _ = reset_and_prefill(&mut model)?;
        // First pass captures the CUDA graphs; second pass replays them and must match bit-for-bit.
        let got_step = model.step_batch(&step_rows)?;
        let got_step_logits = model.batch_logits(b)?;
        let _ = model.verify_batch(
            &slots,
            BatchVerifyTokens::Host(&verify_slices),
            &accepts,
            &samplings,
        )?;
        let got_verify_logits = model.batch_verify_logits()?;

        let _ = reset_and_prefill(&mut model)?;
        let replay_step = model.step_batch(&step_rows)?;
        let replay_step_logits = model.batch_logits(b)?;
        let _ = model.verify_batch(
            &slots,
            BatchVerifyTokens::Host(&verify_slices),
            &accepts,
            &samplings,
        )?;
        let replay_verify_logits = model.batch_verify_logits()?;
        ensure!(
            replay_step == got_step && replay_step_logits == got_step_logits,
            "{label}: B=32 step graph replay differed from capture"
        );
        ensure!(
            replay_verify_logits == got_verify_logits,
            "{label}: B=32 W=8 verify graph replay differed from capture"
        );

        let step_agree = got_step
            .iter()
            .zip(&want_step)
            .filter(|(a, b)| a == b)
            .count();
        let step_kls: Vec<f64> = (0..b)
            .map(|r| {
                kl_div(
                    &want_step_logits[r * vocab..(r + 1) * vocab],
                    &got_step_logits[r * vocab..(r + 1) * vocab],
                )
            })
            .collect();
        let step_mean_kl = step_kls.iter().sum::<f64>() / b as f64;
        let step_max_kl = step_kls.iter().copied().fold(0.0, f64::max);

        let verify_agree = (0..b * w)
            .filter(|&r| {
                argmax(&got_verify_logits[r * vocab..(r + 1) * vocab]) == want_verify_top1[r]
            })
            .count();
        let verify_kls: Vec<f64> = (0..b * w)
            .map(|r| {
                kl_div(
                    &want_verify_logits[r * vocab..(r + 1) * vocab],
                    &got_verify_logits[r * vocab..(r + 1) * vocab],
                )
            })
            .collect();
        let verify_mean_kl = verify_kls.iter().sum::<f64>() / (b * w) as f64;
        let verify_max_kl = verify_kls.iter().copied().fold(0.0, f64::max);
        println!(
            "{label}: B=32 step top1={step_agree}/{b} mean_kl={step_mean_kl:.3e} max_kl={step_max_kl:.3e} | \
             B=32 W=8 verify top1={verify_agree}/{} mean_kl={verify_mean_kl:.3e} max_kl={verify_max_kl:.3e} (graph replay bit-exact)",
            b * w
        );
        // Full 64-token FP8-dense prefill + FP8-dense step/verify vs BF16-dense baseline:
        // matches eval --prefill-dense fp8 parity (top-1 >= 90%, mean KL <= 2e-2).
        ensure!(
            step_agree >= 28 && step_mean_kl <= 1.5e-2 && step_max_kl <= 0.15,
            "{label}: B=32 step parity exceeded limit: top1={step_agree}/{b} mean={step_mean_kl:.3e} max={step_max_kl:.3e}"
        );
        ensure!(
            verify_agree >= 235 && verify_mean_kl <= 2.0e-2,
            "{label}: B=32 W=8 verify parity exceeded limit: top1={verify_agree}/{} mean={verify_mean_kl:.3e} max={verify_max_kl:.3e}",
            b * w
        );
    }
    Ok(())
}

/// GPU parity test for `Model::step_hybrid` (Track E8): piggybacking chunked prefill slices
/// (one short prompt completing mid-piggyback and one 1,500-token prompt spanning the 1,024
/// sliding-window wrap across 6 chunks) onto active batched decode slots (mixed greedy +
/// sampled, on both sides of the sliding window) must match separate `prefill_slot` +
/// `step_batch_sampled` calls on emitted tokens and step logits.
#[test]
#[ignore = "needs the checkpoint and a GPU"]
#[expect(clippy::too_many_lines, reason = "end-to-end hybrid piggyback parity test")]
fn hybrid_step_matches_separate_fp8() -> anyhow::Result<()> {
    let (_gpu, mut model) = load(KvFormat::Fp8, WeightFormat::Fp8)?;
    model.enable_sampling()?;
    let capture = capture_tokens()?;
    let vocab = model.config().unpadded_vocab_size;

    // 4 active decode slots (slots 0..4) + 2 piggybacked prefill slots (slots 4, 5).
    let decode_specs: [(usize, usize, Sampling); 4] = [
        (0, 64, Sampling::GREEDY),
        (200, 1100, Sampling::GREEDY),
        (600, 240, Sampling::new(0.8, 0, 0.95, 0.0, 42)?),
        (1800, 1023, Sampling::GREEDY),
    ];
    let prefill_specs: [(usize, usize, Sampling); 2] = [
        // Long prompt spanning sliding window boundary, chunked in 256-token piggyback slices.
        (3000, 1500, Sampling::GREEDY),
        // Short prompt completing on step 0 and joining decode on step 1.
        (4600, 90, Sampling::new(0.7, 40, 0.9, 0.0, 99)?),
    ];
    let decode_n = decode_specs.len();
    let total_slots = decode_n + prefill_specs.len();
    model.add_slots(total_slots - 1)?;

    // 1. Reference: separate `prefill_slot` + `step_batch_sampled` for 6 steps.
    let mut ref_decode_tokens: Vec<Vec<u32>> = Vec::with_capacity(decode_n);
    let mut ref_decode_logits: Vec<Vec<Vec<f32>>> = vec![Vec::new(); decode_n];
    for (slot, &(first, len, sampling)) in decode_specs.iter().enumerate() {
        model.reset_slot(slot)?;
        model.set_sampling(sampling);
        let first_tok = model.prefill_slot(slot, &capture[first..first + len], DEFAULT_PREFILL_CHUNK)?;
        ref_decode_tokens.push(vec![first_tok]);
    }
    let decode_samplings: Vec<Sampling> = decode_specs.iter().map(|s| s.2).collect();
    for _step in 0..6 {
        let rows: Vec<(usize, u32)> = (0..decode_n)
            .map(|slot| (slot, *ref_decode_tokens[slot].last().unwrap_or(&0)))
            .collect();
        let next = model.step_batch_sampled(&rows, &decode_samplings)?;
        let logits = model.batch_logits(decode_n)?;
        for slot in 0..decode_n {
            ref_decode_tokens[slot].push(next[slot]);
            ref_decode_logits[slot].push(logits[slot * vocab..(slot + 1) * vocab].to_vec());
        }
    }
    let mut ref_prefill_first = Vec::with_capacity(prefill_specs.len());
    let mut ref_prefill_step0_logits = Vec::with_capacity(prefill_specs.len());
    for (i, &(first, len, sampling)) in prefill_specs.iter().enumerate() {
        let slot = decode_n + i;
        model.reset_slot(slot)?;
        model.set_sampling(sampling);
        let first_tok =
            model.prefill_slot(slot, &capture[first..first + len], DEFAULT_PREFILL_CHUNK)?;
        let _ = model.step_batch_sampled(&[(slot, first_tok)], &[sampling])?;
        ref_prefill_first.push(first_tok);
        ref_prefill_step0_logits.push(model.batch_logits(1)?);
    }

    // 2. Hybrid piggybacked pass: prefill decode slots 0..4, reset slots 4 & 5, then run
    //    6 hybrid steps piggybacking 256-token chunks of slot 4 (plus 90-token slot 5 on step 0).
    let mut got_decode_tokens: Vec<Vec<u32>> = Vec::with_capacity(decode_n);
    for (slot, &(first, len, sampling)) in decode_specs.iter().enumerate() {
        model.reset_slot(slot)?;
        model.set_sampling(sampling);
        let first_tok = model.prefill_slot(slot, &capture[first..first + len], DEFAULT_PREFILL_CHUNK)?;
        got_decode_tokens.push(vec![first_tok]);
    }
    model.reset_slot(4)?;
    model.reset_slot(5)?;

    let long_prompt = &capture[prefill_specs[0].0..prefill_specs[0].0 + prefill_specs[0].1];
    let short_prompt = &capture[prefill_specs[1].0..prefill_specs[1].0 + prefill_specs[1].1];
    let mut long_done = 0usize;
    let mut got_short_first = None;
    let mut got_long_first = None;
    let mut got_short_step0_logits = None;
    let mut near_tie_flips = 0usize;

    for step in 0..6 {
        let mut decode_rows: Vec<(usize, u32)> = (0..decode_n)
            .map(|slot| (slot, *got_decode_tokens[slot].last().unwrap_or(&0)))
            .collect();
        let mut step_samplings = decode_samplings.clone();
        if let Some(short_tok) = got_short_first
            && step == 1
        {
            decode_rows.push((5, short_tok));
            step_samplings.push(prefill_specs[1].2);
        }
        let long_end = (long_done + 256).min(long_prompt.len());
        let mut slices = vec![PackedPrefill {
            slot: 4,
            tokens: &long_prompt[long_done..long_end],
            images: Vec::new(),
            need_first_token: long_end == long_prompt.len(),
            sampling: prefill_specs[0].2,
        }];
        if step == 0 {
            slices.push(PackedPrefill {
                slot: 5,
                tokens: short_prompt,
                images: Vec::new(),
                need_first_token: true,
                sampling: prefill_specs[1].2,
            });
        }
        let res = model.step_hybrid(&decode_rows, &step_samplings, &slices, DEFAULT_PREFILL_CHUNK)?;
        let logits = model.batch_logits(decode_rows.len())?;
        for slot in 0..decode_n {
            let want_tok = ref_decode_tokens[slot][step + 1];
            let got_tok = res.decode_tokens[slot];
            let got_row = &logits[slot * vocab..(slot + 1) * vocab];
            let want_row = &ref_decode_logits[slot][step];
            let mut max_diff = 0.0f32;
            let mut sum_diff = 0.0f64;
            for (&g, &w) in got_row.iter().zip(want_row) {
                let d = (g - w).abs();
                max_diff = max_diff.max(d);
                sum_diff += f64::from(d);
            }
            let mean_diff = sum_diff / vocab as f64;
            ensure!(
                max_diff < 3.0 && mean_diff < 0.6,
                "step {step} decode slot {slot} logits drifted: max_diff={max_diff:.4}, mean_diff={mean_diff:.5}"
            );
            if got_tok != want_tok {
                // `step_hybrid` runs BF16 dense GEMM (`m > 64`) while `step_batch_sampled`
                // uses FP8 decode weights; allow rare top-1 flips only on genuine near-ties
                // (small reference margin + small logit diff) and teacher-force the next input.
                let ref_margin =
                    (want_row[want_tok as usize] - want_row[got_tok as usize]).abs();
                println!(
                    "step {step} decode slot {slot} BF16-hybrid vs FP8-decode near-tie: \
                     got {got_tok}, want {want_tok}, ref_margin={ref_margin:.4}, \
                     max_diff={max_diff:.4}, mean_diff={mean_diff:.5}"
                );
                ensure!(
                    ref_margin < 1.0 && max_diff < 1.5,
                    "step {step} decode slot {slot} diverged beyond near-tie: \
                     got {got_tok}, want {want_tok} (ref_margin={ref_margin:.4}, max_diff={max_diff:.4})"
                );
                near_tie_flips += 1;
            }
            got_decode_tokens[slot].push(want_tok);
        }
        if step == 1 {
            got_short_step0_logits = Some(logits[decode_n * vocab..(decode_n + 1) * vocab].to_vec());
        }
        long_done = long_end;
        if step == 0 {
            ensure!(
                res.prefill_tokens[0].is_none(),
                "partial long prefill slice unexpectedly returned a token"
            );
            got_short_first = Some(
                res.prefill_tokens[1].context("short piggybacked prefill missing first token")?,
            );
        } else if long_end == long_prompt.len() {
            got_long_first = Some(
                res.prefill_tokens[0].context("finishing long piggybacked prefill missing token")?,
            );
        } else {
            ensure!(
                res.prefill_tokens[0].is_none(),
                "mid-chunk long prefill slice unexpectedly returned a token"
            );
        }
    }
    ensure!(
        near_tie_flips <= 2,
        "too many BF16-hybrid vs FP8-decode near-tie flips: {near_tie_flips} > 2"
    );
    ensure!(long_done == long_prompt.len(), "long prompt did not finish");
    let short_first = got_short_first.context("missing short first token")?;
    let long_first = got_long_first.context("missing long first token")?;
    ensure!(
        short_first == ref_prefill_first[1],
        "short piggybacked first token diverged: got {short_first}, want {}",
        ref_prefill_first[1]
    );
    ensure!(
        long_first == ref_prefill_first[0],
        "long piggybacked first token diverged: got {long_first}, want {}",
        ref_prefill_first[0]
    );

    // Check slot 4's KV cache & conv state after 6-chunk hybrid prefill via 1 decode step,
    // plus slot 5's step-0 logits decoded inside step_hybrid on step 1.
    let _ = model.step_batch_sampled(&[(4, long_first)], &[prefill_specs[0].2])?;
    let long_step0_logits = model.batch_logits(1)?;
    let short_got = got_short_step0_logits
        .as_ref()
        .context("missing short step0 logits")?;
    let argmax = |v: &[f32]| -> (usize, f32) {
        v.iter()
            .copied()
            .enumerate()
            .fold((0, f32::NEG_INFINITY), |(bi, bv), (i, x)| {
                if x > bv { (i, x) } else { (bi, bv) }
            })
    };
    let kl_and_cosine = |want_row: &[f32], got_row: &[f32]| -> (f64, f64) {
        let want_max = want_row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let got_max = got_row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let want_logz = want_row
            .iter()
            .map(|&v| f64::from(v - want_max).exp())
            .sum::<f64>()
            .ln()
            + f64::from(want_max);
        let got_logz = got_row
            .iter()
            .map(|&v| f64::from(v - got_max).exp())
            .sum::<f64>()
            .ln()
            + f64::from(got_max);
        let (mut kl, mut dot, mut want_sq, mut got_sq) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for (&want_elem, &got_elem) in want_row.iter().zip(got_row) {
            let want_f64 = f64::from(want_elem);
            let got_f64 = f64::from(got_elem);
            let want_lp = want_f64 - want_logz;
            let got_lp = got_f64 - got_logz;
            kl += want_lp.exp() * (want_lp - got_lp);
            dot += want_f64 * got_f64;
            want_sq += want_f64 * want_f64;
            got_sq += got_f64 * got_f64;
        }
        (kl, dot / (want_sq.sqrt() * got_sq.sqrt()))
    };
    for (name, got_row, want_row) in [
        ("long(1500)", &long_step0_logits, &ref_prefill_step0_logits[0]),
        ("short(90)", short_got, &ref_prefill_step0_logits[1]),
    ] {
        let mut max_diff = 0.0f32;
        let mut sum_diff = 0.0f64;
        for (&g, &w) in got_row.iter().zip(want_row) {
            let d = (g - w).abs();
            max_diff = max_diff.max(d);
            sum_diff += f64::from(d);
        }
        let mean_diff = sum_diff / vocab as f64;
        let (want_idx, want_val) = argmax(want_row);
        let (got_idx, got_val) = argmax(got_row);
        let (kl, cosine) = kl_and_cosine(want_row, got_row);
        println!(
            "hybrid piggyback {name}: top1 want=({want_idx}, {want_val:.3}) got=({got_idx}, \
             {got_val:.3}), KL={kl:.5}, cosine={cosine:.5}, max_diff={max_diff:.4}, \
             mean_diff={mean_diff:.5}"
        );
        ensure!(
            want_idx == got_idx,
            "hybrid piggyback {name} step0 top-1 diverged: got {got_idx}, want {want_idx}"
        );
        ensure!(
            kl < 0.005 && cosine > 0.95,
            "hybrid piggyback {name} distribution drifted: KL={kl:.5}, cosine={cosine:.5}"
        );
    }
    Ok(())
}

/// After `Speculator::warmup_batch_spec`, bucketed batched speculation across changing
/// `seqs` (including non-power-of-2 counts like 7, 5, 3) and adaptive verify widths must
/// trigger zero additional verify or draft CUDA graph captures and remain bit-exact with
/// single-sequence `greedy_by_verify`.
#[test]
#[ignore = "needs the checkpoint, DFlash 2 drafter, and a GPU"]
#[expect(clippy::too_many_lines, reason = "end-to-end bucketed graph + parity test")]
fn bucketed_batch_spec_avoids_live_graph_captures_and_matches_greedy() -> anyhow::Result<()> {
    let Some(drafter) = env::var_os("SIPHON_DFLASH").map(PathBuf::from) else {
        println!("SIPHON_DFLASH not set; skipping");
        return Ok(());
    };
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
    let max_seqs = 8;
    model.add_slots(max_seqs - 1)?;
    let draft = load_draft(
        &mut model,
        &checkpoint,
        DraftSource::Dflash(&drafter),
        CAPACITY,
        &DraftOptions::fast(),
    )?;
    ensure!(draft.is_dflash2(), "expected a DFlash 2 drafter");
    let mut spec = Speculator::with_draft(&model, draft, Width::Fixed(7))?;
    model.reserve_batch_verify(max_seqs)?;
    spec.reserve_batch_spec(&model, max_seqs, max_seqs - 1)?;
    let captured_pairs = spec.warmup_batch_spec(&mut model, max_seqs, 1)?;
    let expected_pairs = siphon::memory::batch_spec_graph_shapes(max_seqs).len();
    ensure!(
        captured_pairs == expected_pairs,
        "warmup captured {captured_pairs} pairs, expected {expected_pairs}"
    );
    let verify_graphs_after_warmup = model.batch_verify_graph_count();
    let (_, draft_graphs_after_warmup) = spec.cached_draft_graph_counts();

    let capture = capture_tokens()?;
    let active_n = 7; // non-power-of-2, pads to bucket 8 then shrinks 7 -> 5 -> 3 -> 1
    let prompts: Vec<&[u32]> = (0..active_n)
        .map(|i| &capture[i * 120..i * 120 + 64 + i * 17])
        .collect();
    let mut want = Vec::with_capacity(active_n);
    for prompt in &prompts {
        model.reset()?;
        let first = model.prefill(prompt, DEFAULT_PREFILL_CHUNK, PrefillLogits::Last)?;
        let mut seq = vec![first];
        seq.extend(greedy_by_verify(&mut model, first, 24)?);
        want.push(seq);
    }

    let mut anchors = Vec::with_capacity(active_n);
    let mut got: Vec<Vec<u32>> = Vec::with_capacity(active_n);
    for (slot, prompt) in prompts.iter().enumerate() {
        model.reset_slot(slot)?;
        let first = spec.prefill_slot_batch(
            &mut model,
            slot,
            prompt,
            None,
            DEFAULT_PREFILL_CHUNK,
            true,
        )?;
        anchors.push(first);
        got.push(vec![first]);
    }

    // Drive shrinking non-bucket concurrency levels (7 -> 5 -> 3 -> 1) and varied max widths.
    for (round, &count) in [7usize, 7, 5, 5, 3, 3, 1].iter().enumerate() {
        let slots: Vec<usize> = (0..count).collect();
        let round_anchors = anchors[..count].to_vec();
        let samplings = vec![Sampling::GREEDY; count];
        let max_w = [8usize, 4, 6, 2, 8, 4, 8][round];
        let verdicts =
            spec.speculate_batch_round(&mut model, &slots, &round_anchors, &samplings, max_w)?;
        for (i, v) in verdicts.into_iter().enumerate() {
            got[i].extend(v.emitted());
            anchors[i] = v.next();
        }
    }

    ensure!(
        model.batch_verify_graph_count() == verify_graphs_after_warmup,
        "live verify graph capture occurred: {} -> {}",
        verify_graphs_after_warmup,
        model.batch_verify_graph_count()
    );
    let (_, draft_graphs_after_live) = spec.cached_draft_graph_counts();
    ensure!(
        draft_graphs_after_live == draft_graphs_after_warmup,
        "live batched draft graph capture occurred: {draft_graphs_after_warmup} -> {draft_graphs_after_live}"
    );
    for i in 0..active_n {
        let check = got[i].len().min(want[i].len());
        ensure!(
            got[i][..check] == want[i][..check],
            "sequence {i} diverged under bucket-padded batched speculation"
        );
    }
    println!(
        "bucketed batch spec matched greedy_by_verify bit-for-bit across seqs 7/5/3/1 with 0 live graph captures ({verify_graphs_after_warmup} verify, {draft_graphs_after_warmup} draft)"
    );
    Ok(())
}

/// GPU parity test for `Speculator::speculate_hybrid_round` (batched DFlash 2 verify +
/// piggybacked chunked prefill): verifies 3 decoding slots with DFlash 2 while piggybacking
/// a 4th prompt in 256-token chunks, then hands off the 4th slot into `speculate_batch_round`
/// alongside the original 3 slots.
#[test]
#[ignore = "needs the checkpoint, SIPHON_DFLASH (DFlash 2), and a GPU"]
#[expect(clippy::too_many_lines, reason = "end-to-end hybrid DFlash 2 verify + prefill parity test")]
fn hybrid_verify_matches_separate_dflash2() -> anyhow::Result<()> {
    let Some(dflash_dir) = env::var_os("SIPHON_DFLASH") else {
        println!("SIPHON_DFLASH is not set; skipped");
        return Ok(());
    };
    let (_gpu, mut model) = load(KvFormat::Fp8, WeightFormat::Fp8)?;
    model.enable_sampling()?;
    let model_dir = PathBuf::from(env::var_os("SIPHON_MODEL").context("set SIPHON_MODEL")?);
    let checkpoint = Checkpoint::open(&model_dir)?;
    let draft = load_draft(
        &mut model,
        &checkpoint,
        DraftSource::Dflash(std::path::Path::new(&dflash_dir)),
        CAPACITY,
        &DraftOptions::fast(),
    )?;
    let mut spec = Speculator::with_draft(&model, draft, Width::Fixed(4))?;
    let capture = capture_tokens()?;
    let prompts: [&[u32]; 4] = [
        &capture[0..128],
        &capture[200..260],
        &capture[600..1100],
        &capture[2200..2712], // 512 tokens, piggybacked in two 256-token chunks
    ];
    let samplings = [Sampling::GREEDY; 4];
    model.add_slots(3)?;
    model.reserve_batch_verify(4)?;
    spec.reserve_batch_spec(&model, 4, 3)?;

    // Reference: prefill slot 3 separately and check its first token + 8 greedy verify tokens.
    model.reset_slot(0)?;
    let ref_first3 = model.prefill_slot(0, prompts[3], DEFAULT_PREFILL_CHUNK)?;
    let ref_gen3 = greedy_by_verify(&mut model, ref_first3, 8)?;

    // Hybrid run: prefill slots 0..3 with DFlash 2 taps, then piggyback slot 3 onto two
    // batched DFlash 2 verify rounds of slots 0..3.
    let mut anchors = Vec::with_capacity(4);
    for (slot, &prompt) in prompts.iter().enumerate().take(3) {
        model.reset_slot(slot)?;
        let first =
            spec.prefill_slot_batch(&mut model, slot, prompt, None, DEFAULT_PREFILL_CHUNK, true)?;
        anchors.push(first);
    }
    model.reset_slot(3)?;
    let verify_slots = [0usize, 1, 2];
    let mut done3 = 0usize;
    let mut got_first3 = None;
    for round in 0..2 {
        let end = (done3 + 256).min(prompts[3].len());
        let slices = [PackedPrefill {
            slot: 3,
            tokens: &prompts[3][done3..end],
            images: Vec::new(),
            need_first_token: end == prompts[3].len(),
            sampling: samplings[3],
        }];
        let res = spec.speculate_hybrid_round(
            &mut model,
            &verify_slots,
            &anchors[..3],
            &samplings[..3],
            4,
            &slices,
            &[true],
            &[round == 0],
            DEFAULT_PREFILL_CHUNK,
        )?;
        for (i, v) in res.verdicts.iter().enumerate() {
            anchors[i] = v.next();
        }
        done3 = end;
        if end == prompts[3].len() {
            got_first3 = res.prefill_tokens[0];
        } else {
            ensure!(res.prefill_tokens[0].is_none(), "unexpected early first token");
        }
    }
    let first3 = got_first3.context("missing first token for piggybacked DFlash 2 slot")?;
    ensure!(
        first3 == ref_first3,
        "hybrid verify piggybacked first token diverged: got {first3}, want {ref_first3}"
    );
    ensure!(
        spec.slot_draft_committed(3) == prompts[3].len(),
        "slot 3 DFlash 2 committed {} != {}",
        spec.slot_draft_committed(3),
        prompts[3].len()
    );

    // Now slot 3 joins slots 0..3 in `speculate_batch_round` seamlessly!
    anchors.push(first3);
    let all_slots = [0usize, 1, 2, 3];
    let mut got_gen3 = Vec::new();
    while got_gen3.len() < ref_gen3.len() {
        let verdicts = spec.speculate_batch_round(&mut model, &all_slots, &anchors, &samplings, 4)?;
        for (i, v) in verdicts.iter().enumerate() {
            anchors[i] = v.next();
        }
        got_gen3.extend(verdicts[3].emitted());
    }
    println!(
        "hybrid DFlash 2 handoff: first want={ref_first3} got={first3}, gen want={ref_gen3:?} got={:?}",
        &got_gen3[..ref_gen3.len()]
    );
    ensure!(
        got_gen3[..ref_gen3.len()] == ref_gen3[..],
        "slot 3 diverged after hybrid verify piggyback -> batched DFlash 2 handoff"
    );
    Ok(())
}
