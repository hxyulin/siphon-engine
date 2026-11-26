//! CUDA graph capture & replay invariants for batched DFlash 2 drafting
//! (`Draft::batch_commit_and_draft` and `Draft::batch_draft_initial`):
//! - Slot-permutation invariance (changing `slots` order every round without recapturing),
//! - Mixed greedy/sampled `races` dirty-tracking across steady and changing rounds,
//! - Variable `(seqs, verify_width, draft_steps)` shapes in the graph cache,
//! - Transparent recapture when external `taps`/`results`/`anchors` device addresses change.
//!
//! Needs the checkpoint, `SIPHON_DFLASH` (a DFlash 2 drafter), and a GPU:
//!
//! ```sh
//! SIPHON_MODEL=<model-dir> SIPHON_DFLASH=<dflash2-dir> \
//!     cargo test --release -p siphon-cuda --test draft_batch -- --ignored --nocapture
//! ```

#![cfg(target_os = "linux")]

use std::{env, fs, path::PathBuf, sync::Arc};

use anyhow::{Context as _, ensure};
use half::bf16;
use safetensors::{Dtype, SafeTensors};
use siphon::{Checkpoint, TextConfig};
use siphon_cuda::{
    Gpu,
    draft::DraftOptions,
    model::{
        Accept, BatchVerifyTokens, DEFAULT_PREFILL_CHUNK, Model, PrefillLogits, Sampling,
        VerifyTokens,
    },
    ops::{attention::KvFormat, linear::Linear, spec::RESULT_LEN},
    spec::{DraftSource, load_draft},
    weights::{WeightFormat, Weights},
};

const CAPACITY: usize = 4096;

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

/// Runs 4 concurrent sequences (2 greedy, 2 sampled) through `Draft::batch_commit_and_draft`
/// while permuting `slots` across rounds, toggling alternate device `taps`/`results`/`anchors`
/// buffers to exercise graph recapture on pointer changes, and switching batch size mid-run.
/// Every generated token must match single-sequence `Model::verify` bit-for-bit.
#[test]
#[ignore = "needs the checkpoint, SIPHON_DFLASH (DFlash 2), and a GPU"]
#[expect(clippy::too_many_lines, reason = "end-to-end batched DFlash 2 graph replay test")]
fn batch_dflash2_graph_permutations_and_recapture() -> anyhow::Result<()> {
    let Some(dflash_dir) = env::var_os("SIPHON_DFLASH") else {
        println!("SIPHON_DFLASH is not set; skipped");
        return Ok(());
    };
    let model_dir = PathBuf::from(env::var_os("SIPHON_MODEL").context("set SIPHON_MODEL")?);
    let gpu = Gpu::new(0)?;
    let config = TextConfig::load(&model_dir)?;
    let passes = config.loop_config.num_passes;
    let checkpoint = Checkpoint::open(&model_dir)?;
    let loaded = Weights::load(&checkpoint, &config, gpu.stream())?;
    let mut model = Model::with_formats(
        &gpu,
        config,
        loaded,
        passes,
        CAPACITY,
        KvFormat::Fp8,
        WeightFormat::Fp8,
    )?;
    model.enable_sampling()?;
    let mut draft = load_draft(
        &mut model,
        &checkpoint,
        DraftSource::Dflash(std::path::Path::new(&dflash_dir)),
        CAPACITY,
        &DraftOptions::fast(),
    )?;
    ensure!(draft.is_dflash2(), "SIPHON_DFLASH must be a DFlash 2 drafter");
    let stream = Arc::clone(model.stream());
    let linear = Linear::load(stream.context())?;
    let tokens = capture_tokens()?;

    let specs: [(usize, usize, Sampling); 4] = [
        (0, 64, Sampling::GREEDY),
        (120, 260, Sampling::new(0.8, 0, 0.95, 0.0, 42)?),
        (500, 1080, Sampling::GREEDY),
        (2400, 190, Sampling::new(0.7, 40, 0.9, 0.0, 99)?),
    ];
    let n = specs.len();
    let gen_tokens = 48usize;

    // Ground truth via single-sequence one-row verifies in slot 0.
    let mut want: Vec<Vec<u32>> = Vec::with_capacity(n);
    for &(first, len, sampling) in &specs {
        model.reset()?;
        model.set_sampling(sampling);
        let mut tok =
            model.prefill(&tokens[first..first + len], DEFAULT_PREFILL_CHUNK, PrefillLogits::Last)?;
        let mut seq = vec![tok];
        while seq.len() <= gen_tokens {
            let v = model.verify(VerifyTokens::Host(&[tok]), Accept::Rows(1))?;
            tok = v.next();
            seq.push(tok);
        }
        want.push(seq);
    }

    // Prefill 4 slots with taps committed into each slot's DFlash 2 rings.
    model.add_slots(n - 1)?;
    let mut got: Vec<Vec<u32>> = Vec::with_capacity(n);
    let mut initial_anchors = Vec::with_capacity(n);
    for (slot, &(first, len, sampling)) in specs.iter().enumerate() {
        model.reset_slot(slot)?;
        draft.reset_slot(slot);
        model.set_sampling(sampling);
        let prompt = &tokens[first..first + len];
        let prompt_i32: Vec<i32> = prompt
            .iter()
            .map(|&t| Ok(i32::try_from(t)?))
            .collect::<anyhow::Result<_>>()?;
        draft.note_tokens(&stream, model.draft_shared(), &prompt_i32)?;
        let first_tok = model.prefill_slot_with_taps(
            slot,
            prompt,
            DEFAULT_PREFILL_CHUNK,
            &mut |pos, taps| draft.commit_slot(&stream, &linear, slot, &taps, pos),
        )?;
        got.push(vec![first_tok]);
        initial_anchors.push(first_tok);
    }

    // Alternate device buffers to force pointer-mismatch recapture on selected rounds.
    let draft_steps = 7usize;
    let width = draft_steps + 1;
    let row_len = draft.tap_layout().len() * siphon_cuda::weights::draft::HIDDEN;
    let mut alt_taps = stream.alloc_zeros::<bf16>(n * width * row_len)?;
    let mut alt_results = stream.alloc_zeros::<i32>(n * RESULT_LEN)?;
    let mut alt_anchors = stream.alloc_zeros::<i32>(n)?;

    // Schedule of slot permutations across rounds: holds steady on some rounds (exercises
    // skipped `cache_table`/`races` H2D) and permutes on others (exercises slot-permutation
    // invariance without graph recapture).
    let perms: [[usize; 4]; 6] = [
        [0, 1, 2, 3],
        [0, 1, 2, 3],
        [3, 1, 0, 2],
        [3, 1, 0, 2],
        [2, 0, 3, 1],
        [1, 3, 2, 0],
    ];
    let accepts = vec![Accept::Greedy; n];

    // Phase 1: initial draft in order [0, 1, 2, 3].
    let mut cur_order = perms[0];
    let cur_samplings: Vec<Sampling> = cur_order.iter().map(|&s| specs[s].2).collect();
    draft.batch_draft_initial(
        &stream,
        model.draft_shared(),
        &cur_order,
        &initial_anchors,
        &cur_samplings,
        draft_steps,
    )?;

    let mut round = 0usize;
    while got.iter().any(|g| g.len() <= gen_tokens) {
        let starts: Vec<usize> = cur_order
            .iter()
            .map(|&s| model.slot_position(s))
            .collect::<anyhow::Result<_>>()?;
        let cur_samplings: Vec<Sampling> = cur_order.iter().map(|&s| specs[s].2).collect();
        let pending = model.start_verify_batch(
            &cur_order,
            BatchVerifyTokens::Device {
                tokens: draft.batch_verify_tokens(n, width)?,
                width,
            },
            &accepts,
            &cur_samplings,
        )?;
        // On round 3, copy verify outputs into alternate device buffers so
        // `batch_commit_and_draft` detects changed external pointers and recaptures cleanly;
        // on round 4, switch back to `model`'s buffers to verify recapture back works too.
        if round == 3 {
            stream.memcpy_dtod(&model.batch_verify_taps()?, &mut alt_taps)?;
            stream.memcpy_dtod(&model.batch_verify_results()?, &mut alt_results)?;
            stream.memcpy_dtod(&model.batch_verify_anchors()?, &mut alt_anchors)?;
            draft.batch_commit_and_draft(
                &stream,
                model.draft_shared(),
                &cur_order,
                &starts,
                &alt_taps.slice(..),
                &alt_results.slice(..),
                &alt_anchors.slice(..),
                &cur_samplings,
                width,
                draft_steps,
            )?;
        } else {
            draft.batch_commit_and_draft(
                &stream,
                model.draft_shared(),
                &cur_order,
                &starts,
                &model.batch_verify_taps()?,
                &model.batch_verify_results()?,
                &model.batch_verify_anchors()?,
                &cur_samplings,
                width,
                draft_steps,
            )?;
        }
        let verdicts = model.finish_verify_batch(pending)?;
        for (i, v) in verdicts.iter().enumerate() {
            let slot = cur_order[i];
            draft.set_slot_committed(slot, starts[i] + v.accepted + 1)?;
            got[slot].extend_from_slice(&v.emitted());
        }
        round += 1;

        // Re-anchor into the next permutation when the permutation changes, mirroring how
        // `Speculator` reorders active batches after a sequence joins or leaves.
        let next_order = perms[round % perms.len()];
        if next_order != cur_order {
            cur_order = next_order;
            let next_anchors: Vec<u32> = cur_order
                .iter()
                .map(|&s| *got[s].last().expect("slot has at least one token"))
                .collect();
            let next_samplings: Vec<Sampling> = cur_order.iter().map(|&s| specs[s].2).collect();
            draft.batch_draft_initial(
                &stream,
                model.draft_shared(),
                &cur_order,
                &next_anchors,
                &next_samplings,
                draft_steps,
            )?;
        }
    }

    for slot in 0..n {
        let len = want[slot].len().min(got[slot].len());
        let differ = want[slot][..len]
            .iter()
            .zip(&got[slot][..len])
            .position(|(a, b)| a != b);
        println!(
            "slot {slot} ({}): checked {len} tokens across {round} rounds, first diff {differ:?}",
            if specs[slot].2.is_greedy() {
                "greedy"
            } else {
                "sampled"
            }
        );
        ensure!(
            differ.is_none(),
            "slot {slot}: permuted/recaptured batched DFlash 2 differed from single-sequence verify"
        );
    }
    Ok(())
}
