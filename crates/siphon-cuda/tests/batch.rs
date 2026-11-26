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
        Accept, BatchVerifyTokens, DEFAULT_PREFILL_CHUNK, Model, PrefillLogits, Sampling,
        VerifyTokens,
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
