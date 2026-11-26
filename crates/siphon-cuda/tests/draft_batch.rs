//! Batched DFlash 2 GPU tests:
//! - CUDA graph capture & replay invariants (`Draft::batch_commit_and_draft` and
//!   `Draft::batch_draft_initial`),
//! - Adaptive split-K and fused single-split no-combine attention parity and microbenchmark
//!   (`draft_attention_g4_batch[_fp8]` and `draft_attention_g4_batch_nosplit[_fp8]`).
//!
//! ```sh
//! cargo test --release -p siphon-cuda --test draft_batch -- --nocapture
//! SIPHON_MODEL=<model-dir> SIPHON_DFLASH=<dflash2-dir> \
//!     cargo test --release -p siphon-cuda --test draft_batch -- --ignored --nocapture
//! ```

#![cfg(target_os = "linux")]

mod common;

use std::{env, fs, path::PathBuf, sync::Arc};

use anyhow::{Context as _, ensure};
use common::{bf16_ulps, pseudo_random_bf16};
use cudarc::driver::{
    CudaSlice, CudaStream, DevicePtr as _,
    sys::{CUevent_flags, CUgraphInstantiate_flags, CUstreamCaptureMode},
};
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
    ops::{
        attention::KvFormat,
        draft::{
            BLOCK, BatchAttentionScratch, CONTROL, DFLASH2_MAX_SPLITS, DraftBatchSplitMode,
            DraftKernels, KV_WIDTH, KvCache, MAX_BATCH_SEQS, adaptive_batch_scratch_slots,
            batch_attention_splits,
        },
        linear::Linear,
        spec::RESULT_LEN,
    },
    spec::{DraftSource, load_draft},
    weights::{
        WeightFormat, Weights,
        dflash2::{DFLASH2_HEADS, DFLASH2_WINDOW},
        draft::{HEAD_DIM, KV_HEADS},
    },
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

const SLOTS: usize = DFLASH2_WINDOW + BLOCK;
const WINDOW: usize = DFLASH2_WINDOW - 1;
const ROW_WINDOW: usize = DFLASH2_WINDOW;
const Q_WIDTH: usize = DFLASH2_HEADS * HEAD_DIM;
const DFLASH2_LAYERS: usize = 5;
const MIB: f64 = 1024.0 * 1024.0;

/// Diverse per-sequence positions spanning short context, 32-key chunk boundaries,
/// 128-key split boundaries, sliding-window wrap (`2047`/`2048`), and wrapped rings.
const POS_POOL: [usize; 18] = [
    0, 1, 7, 31, 32, 64, 127, 128, 129, 255, 512, 1024, 2040, 2047, 2048, 3000, 4095, 5000,
];

/// Populates one sequence's `SLOTS`-slot ring cache (`BF16` or `FP8`) with deterministic
/// pseudo-random keys and values.
fn build_ring_cache(
    stream: &Arc<CudaStream>,
    kernels: &DraftKernels,
    fp8: bool,
    seed: u32,
) -> anyhow::Result<KvCache> {
    let mut cache = KvCache::with_format(stream, SLOTS, fp8)?;
    let k_host = pseudo_random_bf16(SLOTS * KV_WIDTH, seed, 1.0, 0.0);
    let v_host = pseudo_random_bf16(SLOTS * KV_WIDTH, seed + 1, 1.0, 0.0);
    let k_dev = stream.clone_htod(&k_host)?;
    let v_dev = stream.clone_htod(&v_host)?;
    kernels.store_kv(stream, &k_dev, &v_dev, SLOTS, KV_WIDTH, &mut cache, 0)?;
    Ok(cache)
}

/// Builds a `[seqs, layers, 2]` device address table pointing sequence `s` at `caches[s]`
/// across all `layers` layers.
fn build_cache_table(
    stream: &Arc<CudaStream>,
    caches: &[KvCache],
    layers: usize,
) -> anyhow::Result<CudaSlice<u64>> {
    let mut table = Vec::with_capacity(caches.len() * layers * 2);
    for c in caches {
        let kp = c.k.device_ptr(stream).0;
        let vp = c.v.device_ptr(stream).0;
        for _ in 0..layers {
            table.push(kp);
            table.push(vp);
        }
    }
    Ok(stream.clone_htod(&table)?)
}

/// Encodes per-sequence `[start, commit_rows, anchor]` controls so each sequence attends
/// at `position = positions[s]`.
fn encode_controls(positions: &[usize]) -> anyhow::Result<Vec<i32>> {
    let mut controls = Vec::with_capacity(positions.len() * CONTROL);
    for (s, &pos) in positions.iter().enumerate() {
        let commit = (s % 4).min(pos);
        controls.push(i32::try_from(pos - commit)?);
        controls.push(i32::try_from(commit)?);
        controls.push(0);
    }
    Ok(controls)
}

/// Checks BF16 parity against the `splits = 17` reference: every element must be within
/// `max_ulps` BF16 ULPs (or within `largest * 2^-15` absolute error for near-zero
/// cancellation where ULP spacing shrinks toward subnormals).
fn assert_batch_bf16_parity(label: &str, got: &[bf16], want: &[bf16], max_ulps: u32) {
    assert_eq!(got.len(), want.len(), "{label}: length mismatch");
    let largest = want
        .iter()
        .fold(0.0f32, |m, v| m.max(v.to_f32().abs()));
    let near_zero_atol = largest * 2f32.powi(-15);
    let mut worst_ulp = 0u32;
    let mut worst_abs = 0.0f32;
    let mut over_1_ulp = 0usize;
    let mut differing = 0usize;
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let ulp = bf16_ulps(g, w);
        let abs = (g.to_f32() - w.to_f32()).abs();
        worst_abs = worst_abs.max(abs);
        if abs > near_zero_atol {
            worst_ulp = worst_ulp.max(ulp);
            assert!(
                ulp <= max_ulps,
                "{label}[{i}]: got {g} ({}), want {w} ({}) -> {ulp} ULPs, abs={abs:e} (max {max_ulps} ULPs)",
                g.to_f32(),
                w.to_f32(),
            );
        }
        if ulp > 0 {
            differing += 1;
        }
        if ulp > 1 && abs > near_zero_atol {
            over_1_ulp += 1;
        }
    }
    eprintln!(
        "{label}: {differing}/{} differ, >1ulp={over_1_ulp}, worst_ulp={worst_ulp}, worst_abs={worst_abs:.2e}",
        want.len()
    );
}

#[test]
fn batch_attention_splits_policy_and_vram_savings() {
    // 1 CTA/SM at 54 KiB SMEM on RTX 4090 (128), RTX 5090 (170), RTX PRO 6000 (188).
    for slots in [128usize, 170, 188] {
        // Large batches B = 16, 32 dispatch fused single-split no-combine (`splits == 1`).
        assert_eq!(
            batch_attention_splits(slots, 16, SLOTS, DFLASH2_MAX_SPLITS, DraftBatchSplitMode::Adaptive),
            1,
            "slots={slots} B=16 should use nosplit"
        );
        assert_eq!(
            batch_attention_splits(slots, 32, SLOTS, DFLASH2_MAX_SPLITS, DraftBatchSplitMode::Adaptive),
            1,
            "slots={slots} B=32 should use nosplit"
        );
        // Intermediate batches B = 2, 4, 8 use adaptive splits in 2..DFLASH2_MAX_SPLITS.
        for seqs in [2usize, 4, 8] {
            let s = batch_attention_splits(
                slots,
                seqs,
                SLOTS,
                DFLASH2_MAX_SPLITS,
                DraftBatchSplitMode::Adaptive,
            );
            assert!(
                (2..DFLASH2_MAX_SPLITS).contains(&s),
                "slots={slots} B={seqs}: expected adaptive splits in 2..17, got {s}"
            );
            assert!(
                seqs * KV_HEADS * s <= slots,
                "slots={slots} B={seqs} splits={s} overflows one occupancy wave"
            );
        }
        // Very short max_keys (<= 128) collapses straight to nosplit even at B = 1.
        assert_eq!(
            batch_attention_splits(slots, 1, 128, DFLASH2_MAX_SPLITS, DraftBatchSplitMode::Adaptive),
            1
        );
        assert_eq!(
            batch_attention_splits(slots, 1, 64, DFLASH2_MAX_SPLITS, DraftBatchSplitMode::Adaptive),
            1
        );
        // Reference mode always picks 17 splits for full DFlash 2 window.
        assert_eq!(
            batch_attention_splits(slots, 32, SLOTS, DFLASH2_MAX_SPLITS, DraftBatchSplitMode::Reference),
            DFLASH2_MAX_SPLITS
        );

        // Adaptive scratch covers every B in 1..=32 with splits >= 2 and saves > 60 MiB.
        let adaptive_slots =
            adaptive_batch_scratch_slots(slots, MAX_BATCH_SEQS, SLOTS, DFLASH2_MAX_SPLITS);
        for seqs in 1..=MAX_BATCH_SEQS {
            let s = batch_attention_splits(
                slots,
                seqs,
                SLOTS,
                DFLASH2_MAX_SPLITS,
                DraftBatchSplitMode::Adaptive,
            );
            if s > 1 {
                assert!(seqs * s <= adaptive_slots);
            }
        }
        let bytes_per_slot = KV_HEADS * BLOCK * 4 * (HEAD_DIM + 2) * size_of::<f32>();
        let old_bytes = MAX_BATCH_SEQS * DFLASH2_MAX_SPLITS * bytes_per_slot;
        let new_bytes = adaptive_slots * bytes_per_slot;
        let saved_mib = (old_bytes - new_bytes) as f64 / MIB;
        assert!(
            saved_mib > 60.0,
            "slots={slots}: expected >60 MiB VRAM saved, got {saved_mib:.2} MiB"
        );
    }
}

#[expect(clippy::too_many_arguments, reason = "test launch helper")]
fn run_attention_mode(
    stream: &Arc<CudaStream>,
    kernels: &DraftKernels,
    q: &CudaSlice<bf16>,
    cache_table: &CudaSlice<u64>,
    controls: &CudaSlice<i32>,
    layer: usize,
    seqs: usize,
    fp8: bool,
    scratch: &mut BatchAttentionScratch,
    mode: DraftBatchSplitMode,
) -> anyhow::Result<Vec<bf16>> {
    let mut out = stream.alloc_zeros::<bf16>(q.len())?;
    kernels.attention_g4_batch_with_mode(
        stream,
        q,
        cache_table,
        layer,
        DFLASH2_LAYERS,
        SLOTS,
        controls,
        WINDOW,
        ROW_WINDOW,
        SLOTS,
        seqs,
        fp8,
        scratch,
        &mut out,
        mode,
    )?;
    Ok(stream.clone_dtoh(&out)?)
}

fn check_parity_for_format(
    stream: &Arc<CudaStream>,
    kernels: &DraftKernels,
    fp8: bool,
    ref_scratch: &mut BatchAttentionScratch,
    adaptive_scratch: &mut BatchAttentionScratch,
) -> anyhow::Result<()> {
    let fmt = if fp8 { "fp8" } else { "bf16" };
    let active_slots = kernels.attention_batch_slots();
    let caches: Vec<KvCache> = (0..MAX_BATCH_SEQS)
        .map(|s| {
            build_ring_cache(
                stream,
                kernels,
                fp8,
                100 + u32::try_from(s)? * 17 + if fp8 { 10_000 } else { 0 },
            )
        })
        .collect::<anyhow::Result<_>>()?;

    for seqs in [1usize, 2, 4, 8, 16, 32] {
        let cache_table = build_cache_table(stream, &caches[..seqs], DFLASH2_LAYERS)?;
        let positions: Vec<usize> = (0..seqs)
            .map(|s| POS_POOL[s % POS_POOL.len()])
            .collect();
        let controls = stream.clone_htod(&encode_controls(&positions)?)?;
        let q_len = seqs * BLOCK * Q_WIDTH;
        let q = stream.clone_htod(&pseudo_random_bf16(
            q_len,
            500 + u32::try_from(seqs)? * 13,
            1.0,
            0.0,
        ))?;

        let want = run_attention_mode(
            stream,
            kernels,
            &q,
            &cache_table,
            &controls,
            2,
            seqs,
            fp8,
            ref_scratch,
            DraftBatchSplitMode::Reference,
        )?;
        let adaptive_splits = batch_attention_splits(
            active_slots,
            seqs,
            SLOTS,
            DFLASH2_MAX_SPLITS,
            DraftBatchSplitMode::Adaptive,
        );
        let got_adaptive = run_attention_mode(
            stream,
            kernels,
            &q,
            &cache_table,
            &controls,
            2,
            seqs,
            fp8,
            adaptive_scratch,
            DraftBatchSplitMode::Adaptive,
        )?;
        assert_batch_bf16_parity(
            &format!("{fmt} B={seqs:2} adaptive(splits={adaptive_splits:2})"),
            &got_adaptive,
            &want,
            2,
        );

        let got_nosplit = run_attention_mode(
            stream,
            kernels,
            &q,
            &cache_table,
            &controls,
            2,
            seqs,
            fp8,
            adaptive_scratch,
            DraftBatchSplitMode::Fixed(1),
        )?;
        assert_batch_bf16_parity(&format!("{fmt} B={seqs:2} nosplit"), &got_nosplit, &want, 2);
        if adaptive_splits == 1 {
            assert_eq!(
                got_adaptive, got_nosplit,
                "{fmt} B={seqs}: adaptive(splits=1) must be bit-identical to Fixed(1)"
            );
        }
    }
    Ok(())
}

#[test]
fn attention_g4_batch_adaptive_and_nosplit_parity() -> anyhow::Result<()> {
    let gpu = Gpu::new(0)?;
    let kernels = DraftKernels::load(gpu.context())?;
    let stream = gpu.stream();
    let active_slots = kernels.attention_batch_slots();
    eprintln!(
        "GPU: {} SMs, attention_batch_slots={active_slots}",
        kernels.sms()
    );

    let mut ref_scratch = BatchAttentionScratch::new(stream, MAX_BATCH_SEQS, DFLASH2_MAX_SPLITS)?;
    let mut adaptive_scratch = BatchAttentionScratch::for_adaptive(
        stream,
        active_slots,
        MAX_BATCH_SEQS,
        SLOTS,
        DFLASH2_MAX_SPLITS,
    )?;
    eprintln!(
        "BatchAttentionScratch: ref={:.2} MiB ({} slices) -> adaptive={:.2} MiB ({} slices), saved {:.2} MiB",
        ref_scratch.bytes() as f64 / MIB,
        ref_scratch.max_seq_splits(),
        adaptive_scratch.bytes() as f64 / MIB,
        adaptive_scratch.max_seq_splits(),
        (ref_scratch.bytes() - adaptive_scratch.bytes()) as f64 / MIB,
    );

    for fp8 in [false, true] {
        check_parity_for_format(stream, &kernels, fp8, &mut ref_scratch, &mut adaptive_scratch)?;
    }
    Ok(())
}

#[test]
fn attention_g4_batch_cuda_graph_replay() -> anyhow::Result<()> {
    let gpu = Gpu::new(0)?;
    let kernels = DraftKernels::load(gpu.context())?;
    let stream = gpu.stream();
    let mut ref_scratch = BatchAttentionScratch::new(stream, MAX_BATCH_SEQS, DFLASH2_MAX_SPLITS)?;
    let mut adaptive_scratch = BatchAttentionScratch::for_adaptive(
        stream,
        kernels.attention_batch_slots(),
        MAX_BATCH_SEQS,
        SLOTS,
        DFLASH2_MAX_SPLITS,
    )?;

    let caches: Vec<KvCache> = (0..MAX_BATCH_SEQS)
        .map(|s| build_ring_cache(stream, &kernels, true, 900 + u32::try_from(s)? * 11))
        .collect::<anyhow::Result<_>>()?;
    for seqs in [8usize, 16, 32] {
        let cache_table = build_cache_table(stream, &caches[..seqs], DFLASH2_LAYERS)?;
        let q_len = seqs * BLOCK * Q_WIDTH;
        let q = stream.clone_htod(&pseudo_random_bf16(
            q_len,
            777 + u32::try_from(seqs)?,
            1.0,
            0.0,
        ))?;
        let mut controls = stream.clone_htod(&encode_controls(&vec![64usize; seqs])?)?;
        let mut out_graph = stream.alloc_zeros::<bf16>(q_len)?;

        stream.begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
        let enqueued = kernels.attention_g4_batch(
            stream,
            &q,
            &cache_table,
            0,
            DFLASH2_LAYERS,
            SLOTS,
            &controls,
            WINDOW,
            ROW_WINDOW,
            SLOTS,
            seqs,
            true,
            &mut adaptive_scratch,
            &mut out_graph,
        );
        let graph = stream.end_capture(CUgraphInstantiate_flags(0));
        enqueued.context("capturing adaptive draft_attention_g4_batch")?;
        let graph = graph?.context("empty CUDA graph")?;
        graph.upload()?;

        for round in 0..2usize {
            let positions: Vec<usize> = (0..seqs)
                .map(|s| POS_POOL[(s + round * 7) % POS_POOL.len()])
                .collect();
            stream.memcpy_htod(&encode_controls(&positions)?, &mut controls)?;
            graph.launch()?;
            let got = stream.clone_dtoh(&out_graph)?;

            let mut out_ref = stream.alloc_zeros::<bf16>(q_len)?;
            kernels.attention_g4_batch_with_mode(
                stream,
                &q,
                &cache_table,
                0,
                DFLASH2_LAYERS,
                SLOTS,
                &controls,
                WINDOW,
                ROW_WINDOW,
                SLOTS,
                seqs,
                true,
                &mut ref_scratch,
                &mut out_ref,
                DraftBatchSplitMode::Reference,
            )?;
            let want = stream.clone_dtoh(&out_ref)?;
            assert_batch_bf16_parity(&format!("graph fp8 B={seqs:2} round={round}"), &got, &want, 2);
        }
    }
    Ok(())
}

#[expect(clippy::too_many_arguments, reason = "benchmark helper")]
fn time_attention_mode(
    stream: &Arc<CudaStream>,
    kernels: &DraftKernels,
    q: &CudaSlice<bf16>,
    cache_table: &CudaSlice<u64>,
    controls: &CudaSlice<i32>,
    seqs: usize,
    fp8: bool,
    mode: DraftBatchSplitMode,
    scratch: &mut BatchAttentionScratch,
    out: &mut CudaSlice<bf16>,
) -> anyhow::Result<f64> {
    const WARMUP: usize = 20;
    const ITERS: usize = 200;
    let timing = Some(CUevent_flags::CU_EVENT_DEFAULT);

    stream.begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
    let enq = kernels.attention_g4_batch_with_mode(
        stream,
        q,
        cache_table,
        0,
        DFLASH2_LAYERS,
        SLOTS,
        controls,
        WINDOW,
        ROW_WINDOW,
        SLOTS,
        seqs,
        fp8,
        scratch,
        out,
        mode,
    );
    let graph = stream.end_capture(CUgraphInstantiate_flags(0));
    enq?;
    let graph = graph?.context("empty benchmark graph")?;
    graph.upload()?;

    for _ in 0..WARMUP {
        graph.launch()?;
    }
    stream.synchronize()?;
    let start = stream.record_event(timing)?;
    for _ in 0..ITERS {
        graph.launch()?;
    }
    let end = stream.record_event(timing)?;
    stream.synchronize()?;
    Ok(f64::from(start.elapsed_ms(&end)?) * 1000.0 / ITERS as f64)
}

#[test]
#[ignore = "GPU microbenchmark"]
fn bench_attention_g4_batch_adaptive_vs_reference() -> anyhow::Result<()> {
    let gpu = Gpu::new(0)?;
    let kernels = DraftKernels::load(gpu.context())?;
    let stream = gpu.stream();
    let active_slots = kernels.attention_batch_slots();

    let mut ref_scratch = BatchAttentionScratch::new(stream, MAX_BATCH_SEQS, DFLASH2_MAX_SPLITS)?;
    let mut adaptive_scratch = BatchAttentionScratch::for_adaptive(
        stream,
        active_slots,
        MAX_BATCH_SEQS,
        SLOTS,
        DFLASH2_MAX_SPLITS,
    )?;

    eprintln!(
        "GPU: {} SMs, attention_batch_slots={active_slots}, scratch ref={:.2} MiB -> adaptive={:.2} MiB (saved {:.2} MiB)",
        kernels.sms(),
        ref_scratch.bytes() as f64 / MIB,
        adaptive_scratch.bytes() as f64 / MIB,
        (ref_scratch.bytes() - adaptive_scratch.bytes()) as f64 / MIB,
    );
    eprintln!(
        "{:<5} {:>3} {:>10} {:>10} {:>10} {:>8} {:>9}",
        "fmt", "B", "ref_us", "adapt_us", "nospl_us", "splits", "speedup"
    );

    for fp8 in [false, true] {
        let fmt = if fp8 { "fp8" } else { "bf16" };
        let caches: Vec<KvCache> = (0..MAX_BATCH_SEQS)
            .map(|s| build_ring_cache(stream, &kernels, fp8, 2000 + u32::try_from(s)? * 19))
            .collect::<anyhow::Result<_>>()?;

        for seqs in [8usize, 16, 32] {
            let cache_table = build_cache_table(stream, &caches[..seqs], DFLASH2_LAYERS)?;
            let positions = vec![3000usize; seqs];
            let controls = stream.clone_htod(&encode_controls(&positions)?)?;
            let q_len = seqs * BLOCK * Q_WIDTH;
            let q = stream.clone_htod(&pseudo_random_bf16(
                q_len,
                42 + u32::try_from(seqs)?,
                1.0,
                0.0,
            ))?;
            let mut out = stream.alloc_zeros::<bf16>(q_len)?;

            let ref_us = time_attention_mode(
                stream,
                &kernels,
                &q,
                &cache_table,
                &controls,
                seqs,
                fp8,
                DraftBatchSplitMode::Reference,
                &mut ref_scratch,
                &mut out,
            )?;
            let adapt_us = time_attention_mode(
                stream,
                &kernels,
                &q,
                &cache_table,
                &controls,
                seqs,
                fp8,
                DraftBatchSplitMode::Adaptive,
                &mut adaptive_scratch,
                &mut out,
            )?;
            let nospl_us = time_attention_mode(
                stream,
                &kernels,
                &q,
                &cache_table,
                &controls,
                seqs,
                fp8,
                DraftBatchSplitMode::Fixed(1),
                &mut adaptive_scratch,
                &mut out,
            )?;
            let adapt_splits = batch_attention_splits(
                active_slots,
                seqs,
                SLOTS,
                DFLASH2_MAX_SPLITS,
                DraftBatchSplitMode::Adaptive,
            );
            eprintln!(
                "{fmt:<5} {seqs:>3} {ref_us:>10.2} {adapt_us:>10.2} {nospl_us:>10.2} {adapt_splits:>8} {:>8.2}x",
                ref_us / adapt_us
            );
        }
    }
    Ok(())
}
