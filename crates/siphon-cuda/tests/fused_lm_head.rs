//! GPU parity tests for Wave 3 #2: Fused LM Head + Softcap + Argmax/Sampling + Native FP8 LM
//! Head (`m > 32`).
//!
//! Verifies:
//! - 0-diff argmax parity of fused `LinearRows::run_argmax -> Misc::argmax_partials` against
//!   unfused `LinearRows::run -> Elementwise::logit_softcap -> Misc::argmax_f32` for both BF16
//!   and FP8 weights across `m = 1, 4, 8, 32, 64, 256`, including tie-breaking, NaN handling,
//!   and `col >= unpadded_vocab_size` masking.
//! - 0-diff argmax parity of fused `LinearFp8::gemm_argmax -> Misc::argmax_partials` against
//!   unfused `LinearFp8::gemm -> Elementwise::logit_softcap -> Misc::argmax_f32` for wide
//!   batches (`m = 33, 64, 128, 129, 256`), including tie-breaking across 128-column tile
//!   boundaries and padding masking.
//! - 0-diff token parity of `Sample::run_bf16` (8-wide `uint4` loads + inline softcap) against
//!   unfused `Elementwise::logit_softcap -> Sample::run` across greedy, temperature, top-k,
//!   top-p, min-p, and mixed rows, with poisoned padding columns.
//! - Exactness of the vectorized PDL `Elementwise::logit_softcap` (even `float4` and odd
//!   `float2` rows) and `Misc::argmax_f32` (unaligned row strides, ties, NaN).

#![cfg(target_os = "linux")]

mod common;

use half::bf16;
use siphon_cuda::{
    Gpu,
    ops::{
        elementwise::Elementwise,
        linear::Start,
        linear_fp8::{Fp8Gemm, Fp8Rows, GEMM_TILE, LinearFp8},
        linear_nvfp4::{ARGMAX_MAX_BLOCKS as NVFP4_ARGMAX_MAX_BLOCKS, LinearNvfp4, Nvfp4Gemv},
        linear_rows::{LinearRows, RowsProduct, RowsWeight},
        misc::{ArgmaxScratch, Misc},
        sample::{Sample, SampleBuffers, Sampling},
    },
};

use common::pseudo_random_bf16;

const HIDDEN: usize = 2304;
const PADDED_VOCAB: usize = 201_024;
const VOCAB: usize = 200_058;
const SOFTCAP: f32 = 30.0;

struct Setup {
    gpu: Gpu,
    rows: LinearRows,
    fp8: LinearFp8,
    elementwise: Elementwise,
    misc: Misc,
    sample: Sample,
}

impl Setup {
    fn new() -> anyhow::Result<Self> {
        let gpu = Gpu::new(0)?;
        let ctx = gpu.context();
        Ok(Self {
            rows: LinearRows::load(ctx)?,
            fp8: LinearFp8::load(ctx)?,
            elementwise: Elementwise::load(ctx)?,
            misc: Misc::load(ctx)?,
            sample: Sample::load(ctx)?,
            gpu,
        })
    }
}

/// Builds a synthetic `[PADDED_VOCAB, HIDDEN]` LM head weight with realistic small values, a
/// few stronger rows, and huge poison rows in `VOCAB..PADDED_VOCAB` so any missing vocab mask
/// is immediately caught.
fn synthetic_lm_head(seed: u32) -> Vec<bf16> {
    let mut w = pseudo_random_bf16(PADDED_VOCAB * HIDDEN, seed, 0.04, 0.0);
    // Poison padding columns `col >= VOCAB` with large weights aligned to the first input dim.
    for row in [VOCAB, VOCAB + 1, VOCAB + 17, PADDED_VOCAB - 1] {
        for c in 0..HIDDEN {
            w[row * HIDDEN + c] = bf16::from_f32(2.0);
        }
    }
    w
}

/// Builds `[m, HIDDEN]` inputs with mild magnitudes so logits stay well within the strictly
/// monotonic range of `softcap(x, 30.0)`.
fn synthetic_hidden(m: usize, seed: u32) -> Vec<bf16> {
    let mut x = pseudo_random_bf16(m * HIDDEN, seed, 0.25, 0.0);
    // Positive bias on the first few dims so the poisoned padding rows would dominate if unmasked.
    for r in 0..m {
        for c in 0..16 {
            x[r * HIDDEN + c] = bf16::from_f32(0.5);
        }
    }
    x
}

#[test]
fn fused_linear_rows_argmax_matches_unfused_bf16_and_fp8() -> anyhow::Result<()> {
    let s = Setup::new()?;
    let stream = s.gpu.stream();
    let w_host = synthetic_lm_head(11);
    let w_dev = stream.clone_htod(&w_host)?;
    let w_fp8 = s.fp8.quantize(stream, &w_dev, PADDED_VOCAB, HIDDEN)?;

    let max_m = 256usize;
    let x_host = synthetic_hidden(max_m, 23);
    let x_dev = stream.clone_htod(&x_host)?;

    let max_partials = max_m * VOCAB.div_ceil(16);
    let mut partial_vals = stream.alloc_zeros::<f32>(max_partials)?;
    let mut partial_idxs = stream.alloc_zeros::<i32>(max_partials)?;
    let mut got_dev = stream.alloc_zeros::<i32>(max_m)?;
    let mut want_dev = stream.alloc_zeros::<i32>(max_m)?;

    // Chunked unfused reference buffers (32 rows at a time to keep test VRAM modest).
    let mut ref_logits = stream.alloc_zeros::<bf16>(32 * PADDED_VOCAB)?;
    let mut ref_capped = stream.alloc_zeros::<f32>(32 * VOCAB)?;

    for is_fp8 in [false, true] {
        let label = if is_fp8 { "fp8" } else { "bf16" };
        for &m in &[1usize, 4, 8, 32, 64, 256] {
            // Unfused reference: linear_rows -> logit_softcap_bf16_f32 -> argmax_f32
            let mut want = Vec::with_capacity(m);
            for first in (0..m).step_by(32) {
                let n = 32.min(m - first);
                let x_slice = x_dev.slice(first * HIDDEN..(first + n) * HIDDEN);
                let mut out_slice = ref_logits.slice_mut(..n * PADDED_VOCAB);
                let mut prod = [if is_fp8 {
                    RowsProduct::fp8(&w_fp8, &x_slice, &mut out_slice)
                } else {
                    RowsProduct::bf16(&w_dev, &x_slice, &mut out_slice)
                }];
                s.rows.run(stream, &mut prod, n, Start::Serial)?;
                s.elementwise.logit_softcap(
                    stream,
                    &ref_logits.slice(..n * PADDED_VOCAB),
                    &mut ref_capped.slice_mut(..n * VOCAB),
                    SOFTCAP,
                )?;
                s.misc.argmax_f32(
                    stream,
                    &ref_capped.slice(..n * VOCAB),
                    &mut want_dev.slice_mut(..n),
                    VOCAB,
                )?;
                want.extend(stream.clone_dtoh(&want_dev.slice(..n))?);
            }

            // Fused path: linear_rows::run_argmax -> misc::argmax_partials
            let weight = if is_fp8 {
                RowsWeight::Fp8(&w_fp8)
            } else {
                RowsWeight::Bf16(w_dev.as_view())
            };
            let n_partials = s.rows.run_argmax(
                stream,
                &weight,
                &x_dev.slice(..m * HIDDEN),
                VOCAB,
                &mut partial_vals,
                &mut partial_idxs,
                m,
                Start::Early,
            )?;
            s.misc.argmax_partials(
                stream,
                &partial_vals.slice(..m * n_partials),
                &partial_idxs.slice(..m * n_partials),
                &mut got_dev.slice_mut(..m),
                n_partials,
                n_partials,
            )?;
            let got = stream.clone_dtoh(&got_dev.slice(..m))?;
            let vocab_i32 = i32::try_from(VOCAB)?;
            assert_eq!(got, want, "{label} m = {m}: fused argmax differs from unfused");
            assert!(
                got.iter().all(|&id| (0..vocab_i32).contains(&id)),
                "{label} m = {m}: returned out-of-vocab token"
            );
            eprintln!("{label} m = {m}: 0-diff argmax parity ({n_partials} partials/row)");
        }
    }
    Ok(())
}

#[test]
fn fused_linear_rows_argmax_tie_breaking_nan_and_masking() -> anyhow::Result<()> {
    let s = Setup::new()?;
    let stream = s.gpu.stream();

    let run_both = |w_host: &[bf16], x_host: &[bf16], m: usize| -> anyhow::Result<(Vec<i32>, Vec<i32>)> {
        let w_dev = stream.clone_htod(w_host)?;
        let x_dev = stream.clone_htod(x_host)?;

        let mut logits = stream.alloc_zeros::<bf16>(m * PADDED_VOCAB)?;
        let mut capped = stream.alloc_zeros::<f32>(m * VOCAB)?;
        let mut want_dev = stream.alloc_zeros::<i32>(m)?;
        s.rows.run(
            stream,
            &mut [RowsProduct::bf16(&w_dev, &x_dev, &mut logits)],
            m,
            Start::Serial,
        )?;
        s.elementwise
            .logit_softcap(stream, &logits, &mut capped, SOFTCAP)?;
        s.misc.argmax_f32(stream, &capped, &mut want_dev, VOCAB)?;
        let want = stream.clone_dtoh(&want_dev)?;

        let max_partials = m * VOCAB.div_ceil(16);
        let mut partial_vals = stream.alloc_zeros::<f32>(max_partials)?;
        let mut partial_idxs = stream.alloc_zeros::<i32>(max_partials)?;
        let mut got_dev = stream.alloc_zeros::<i32>(m)?;
        let n_partials = s.rows.run_argmax(
            stream,
            &RowsWeight::Bf16(w_dev.as_view()),
            &x_dev,
            VOCAB,
            &mut partial_vals,
            &mut partial_idxs,
            m,
            Start::Early,
        )?;
        s.misc.argmax_partials(
            stream,
            &partial_vals.slice(..m * n_partials),
            &partial_idxs.slice(..m * n_partials),
            &mut got_dev,
            n_partials,
            n_partials,
        )?;
        Ok((stream.clone_dtoh(&got_dev)?, want))
    };

    // Construct weights where specific rows produce exact ties and masked-out padding spikes.
    let mut w_host = vec![bf16::ZERO; PADDED_VOCAB * HIDDEN];
    // Row 0 of x has x[0] = 1.0 -> tests intra-tile tie (5 vs 13) + padding spike (VOCAB).
    w_host[5 * HIDDEN] = bf16::from_f32(8.0);
    w_host[13 * HIDDEN] = bf16::from_f32(8.0);
    w_host[VOCAB * HIDDEN] = bf16::from_f32(32.0);
    w_host[(PADDED_VOCAB - 1) * HIDDEN] = bf16::from_f32(64.0);

    // Row 1 of x has x[1] = 1.0 -> tests cross-tile tie (100 vs 150_005 vs 200_057).
    w_host[100 * HIDDEN + 1] = bf16::from_f32(7.5);
    w_host[150_005 * HIDDEN + 1] = bf16::from_f32(7.5);
    w_host[(VOCAB - 1) * HIDDEN + 1] = bf16::from_f32(7.5);
    w_host[VOCAB * HIDDEN + 1] = bf16::from_f32(50.0);

    // Row 2 of x has x[2] = 1.0 -> tests last valid index VOCAB - 1 winning when VOCAB is larger.
    w_host[(VOCAB - 1) * HIDDEN + 2] = bf16::from_f32(6.0);
    w_host[VOCAB * HIDDEN + 2] = bf16::from_f32(60.0);

    let mut x_host = vec![bf16::ZERO; 3 * HIDDEN];
    x_host[0] = bf16::from_f32(1.0);
    x_host[HIDDEN + 1] = bf16::from_f32(1.0);
    x_host[2 * HIDDEN + 2] = bf16::from_f32(1.0);

    let (got, want) = run_both(&w_host, &x_host, 3)?;
    assert_eq!(got, want, "edge-case fused argmax differs from unfused");
    assert_eq!(got, vec![5, 100, i32::try_from(VOCAB - 1)?]);

    // Separate weight matrix for NaN handling (0.0 * NaN == NaN in IEEE-754 would otherwise
    // poison finite rows sharing the same weight matrix): first NaN at 77 beats finite (10)
    // and later NaN (90_000), while NaN in padding (VOCAB) is ignored.
    let mut w_nan = vec![bf16::ZERO; PADDED_VOCAB * HIDDEN];
    w_nan[10 * HIDDEN] = bf16::from_f32(12.0);
    w_nan[77 * HIDDEN] = bf16::NAN;
    w_nan[90_000 * HIDDEN] = bf16::NAN;
    w_nan[VOCAB * HIDDEN] = bf16::NAN;
    let mut x_nan = vec![bf16::ZERO; HIDDEN];
    x_nan[0] = bf16::from_f32(1.0);

    let (got_nan, want_nan) = run_both(&w_nan, &x_nan, 1)?;
    assert_eq!(got_nan, want_nan, "NaN fused argmax differs from unfused");
    assert_eq!(got_nan, vec![77]);
    Ok(())
}

#[test]
fn fused_linear_fp8_gemm_argmax_matches_unfused() -> anyhow::Result<()> {
    let s = Setup::new()?;
    let stream = s.gpu.stream();

    let mut w_host = synthetic_lm_head(41);
    // Add exact cross-128-tile tie on dim 0 (col 127 vs col 128) and dim 1 (col 3 vs col 199_999).
    for c in 0..HIDDEN {
        w_host[127 * HIDDEN + c] = w_host[128 * HIDDEN + c];
        w_host[3 * HIDDEN + c] = w_host[199_999 * HIDDEN + c];
    }
    let w_dev = stream.clone_htod(&w_host)?;
    let w_fp8 = s.fp8.quantize(stream, &w_dev, PADDED_VOCAB, HIDDEN)?;

    let max_m = 256usize;
    let x_host = synthetic_hidden(max_m, 53);
    let x_dev = stream.clone_htod(&x_host)?;

    let mut fp8_scratch = Fp8Rows::new(stream, max_m * HIDDEN)?;
    let tiles_n = VOCAB.div_ceil(GEMM_TILE);
    let mut partial_vals = stream.alloc_zeros::<f32>(max_m * tiles_n)?;
    let mut partial_idxs = stream.alloc_zeros::<i32>(max_m * tiles_n)?;
    let mut got_dev = stream.alloc_zeros::<i32>(max_m)?;
    let mut want_dev = stream.alloc_zeros::<i32>(max_m)?;

    for &m in &[33usize, 64, 128, 129, 256] {
        // Unfused reference: LinearFp8::gemm -> logit_softcap -> argmax_f32
        let mut ref_logits = stream.alloc_zeros::<bf16>(m * PADDED_VOCAB)?;
        let mut ref_capped = stream.alloc_zeros::<f32>(m * VOCAB)?;
        let x_view = x_dev.slice(..m * HIDDEN);
        s.fp8.gemm(
            stream,
            &mut [Fp8Gemm {
                weight: &w_fp8,
                x: x_view,
                out: ref_logits.as_view_mut(),
            }],
            m,
            &mut fp8_scratch,
        )?;
        s.elementwise
            .logit_softcap(stream, &ref_logits, &mut ref_capped, SOFTCAP)?;
        s.misc
            .argmax_f32(stream, &ref_capped, &mut want_dev.slice_mut(..m), VOCAB)?;
        let want = stream.clone_dtoh(&want_dev.slice(..m))?;

        // Fused path: LinearFp8::gemm_argmax -> Misc::argmax_partials
        let n_partials = s.fp8.gemm_argmax(
            stream,
            &w_fp8,
            &x_dev.slice(..m * HIDDEN),
            VOCAB,
            &mut fp8_scratch,
            &mut partial_vals,
            &mut partial_idxs,
            m,
        )?;
        assert_eq!(n_partials, tiles_n);
        s.misc.argmax_partials(
            stream,
            &partial_vals.slice(..m * n_partials),
            &partial_idxs.slice(..m * n_partials),
            &mut got_dev.slice_mut(..m),
            n_partials,
            n_partials,
        )?;
        let got = stream.clone_dtoh(&got_dev.slice(..m))?;

        let vocab_i32 = i32::try_from(VOCAB)?;
        assert_eq!(got, want, "gemm_fp8_128x128_argmax m = {m} differs from unfused");
        assert!(
            got.iter().all(|&id| (0..vocab_i32).contains(&id)),
            "gemm_fp8_128x128_argmax m = {m}: returned out-of-vocab token"
        );
        eprintln!("gemm_fp8_128x128_argmax m = {m}: 0-diff parity ({n_partials} partials/row)");
    }
    Ok(())
}

#[test]
fn sample_bf16_inline_softcap_matches_unfused() -> anyhow::Result<()> {
    let s = Setup::new()?;
    let stream = s.gpu.stream();

    let samplings = [
        Sampling::GREEDY,
        Sampling::new(0.8, 0, 1.0, 0.0, 101)?,
        Sampling::new(0.7, 40, 1.0, 0.0, 102)?,
        Sampling::new(0.9, 0, 0.9, 0.0, 103)?,
        Sampling::new(1.0, 0, 1.0, 0.05, 104)?,
        Sampling::new(1.1, 50, 0.95, 0.02, 105)?,
        Sampling::new(0.5, 1, 1.0, 0.0, 106)?,
        Sampling::GREEDY,
    ];

    for &rows in &[1usize, 8, 32, 64] {
        let mut host = pseudo_random_bf16(rows * PADDED_VOCAB, 700 + rows as u32, 6.0, 0.0);
        // Inject spikes inside vocab, ties on row 0 (greedy), NaN on row 7 (greedy), and huge
        // poison values in the padding columns `VOCAB..PADDED_VOCAB`.
        for r in 0..rows {
            let base = r * PADDED_VOCAB;
            host[base + (r * 997) % VOCAB] = bf16::from_f32(18.0);
            host[base + (r * 1301 + 17) % VOCAB] = bf16::from_f32(17.5);
            for pad in VOCAB..PADDED_VOCAB {
                host[base + pad] = if pad == VOCAB {
                    bf16::NAN
                } else {
                    bf16::from_f32(500.0)
                };
            }
        }
        // Tie on row 0 (greedy).
        host[42] = bf16::from_f32(25.0);
        host[100_003] = bf16::from_f32(25.0);
        // NaN on row 7 (greedy) when rows >= 8.
        if rows >= 8 {
            host[7 * PADDED_VOCAB + 321] = bf16::NAN;
        }

        let settings: Vec<(Sampling, usize)> = (0..rows)
            .map(|r| (samplings[r % samplings.len()], 1000 + r))
            .collect();

        let logits_bf16 = stream.clone_htod(&host)?;
        let mut capped_f32 = stream.alloc_zeros::<f32>(rows * VOCAB)?;
        let mut buffers = SampleBuffers::new(stream, rows)?;
        buffers.upload(stream, &settings)?;
        let mut want_dev = stream.alloc_zeros::<i32>(rows)?;
        let mut got_dev = stream.alloc_zeros::<i32>(rows)?;

        // Unfused: logit_softcap_bf16_f32 -> sample_prepare / sample_race (F32)
        s.elementwise
            .logit_softcap(stream, &logits_bf16, &mut capped_f32, SOFTCAP)?;
        s.sample
            .run(stream, &capped_f32, VOCAB, &mut buffers, &mut want_dev)?;
        let want = stream.clone_dtoh(&want_dev)?;

        // Fused: sample_prepare_bf16 / sample_race_bf16 directly on BF16 head_logits
        s.sample.run_bf16(
            stream,
            &logits_bf16,
            VOCAB,
            PADDED_VOCAB,
            SOFTCAP,
            &mut buffers,
            &mut got_dev,
        )?;
        let got = stream.clone_dtoh(&got_dev)?;

        assert_eq!(got, want, "sample_bf16 rows = {rows} differs from unfused");
        assert_eq!(got[0], 42, "tie-breaking on greedy row 0");
        if rows >= 8 {
            assert_eq!(got[7], 321, "NaN handling on greedy row 7");
        }
        eprintln!("sample_bf16 rows = {rows}: 0-diff parity across greedy + sampled rows");
    }
    Ok(())
}

#[test]
fn vectorized_logit_softcap_and_argmax_f32_alignment() -> anyhow::Result<()> {
    let s = Setup::new()?;
    let stream = s.gpu.stream();

    // Test odd and even row counts for logit_softcap (VOCAB = 200_058 -> odd rows are 8-byte aligned).
    let rows = 5usize;
    let host = pseudo_random_bf16(rows * PADDED_VOCAB, 909, 25.0, 0.0);
    let dev = stream.clone_htod(&host)?;
    let mut out = stream.alloc_zeros::<f32>(rows * VOCAB)?;
    s.elementwise
        .logit_softcap(stream, &dev, &mut out, SOFTCAP)?;
    let got = stream.clone_dtoh(&out)?;
    for r in 0..rows {
        for c in 0..VOCAB {
            let x = host[r * PADDED_VOCAB + c].to_f32();
            let want = (x / SOFTCAP).tanh() * SOFTCAP;
            let diff = (got[r * VOCAB + c] - want).abs();
            assert!(
                diff <= 2e-5,
                "row {r} col {c}: got {}, want {want} (diff {diff})",
                got[r * VOCAB + c]
            );
        }
    }

    // Test argmax_f32 across unaligned row strides (1, 3, 5, 17, 200_058) with ties and NaN.
    for &n in &[1usize, 3, 5, 17, 1025, VOCAB] {
        let rcount = 5usize;
        let mut x = vec![-10.0f32; rcount * n];
        let mut expected = vec![0i32; rcount];
        for (r, exp) in expected.iter_mut().enumerate() {
            let base = r * n;
            for c in 0..n {
                x[base + c] = -5.0 + ((c * 13 + r * 7) % 100) as f32 * 0.01;
            }
            let win = (r * 37) % n;
            x[base + win] = 10.0;
            if n > 1 && r == 1 {
                // Exact tie: smaller index must win.
                let other = (win + 1) % n;
                x[base + other] = 10.0;
                *exp = i32::try_from(win.min(other))?;
            } else if r == 2 {
                // NaN must win (first NaN).
                let nan_idx = (r * 19) % n;
                x[base + nan_idx] = f32::NAN;
                *exp = i32::try_from(nan_idx)?;
            } else {
                *exp = i32::try_from(win)?;
            }
        }
        let x_dev = stream.clone_htod(&x)?;
        let mut out_dev = stream.alloc_zeros::<i32>(rcount)?;
        s.misc.argmax_f32(stream, &x_dev, &mut out_dev, n)?;
        let got = stream.clone_dtoh(&out_dev)?;
        assert_eq!(got, expected, "argmax_f32 n = {n}");
    }
    Ok(())
}

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "verifies both B=1 gemv_argmax and multi-row rows_argmax plus tie/mask cases"
)]
fn fused_nvfp4_gemv_and_rows_argmax_match_unfused() -> anyhow::Result<()> {
    let s = Setup::new()?;
    let stream = s.gpu.stream();
    let nvfp4 = LinearNvfp4::load(s.gpu.context())?;

    let mut w_host = synthetic_lm_head(61);
    // Inject exact cross-tile ties (row 7 vs row 199_991) and intra-tile ties (row 2 vs row 10).
    for c in 0..HIDDEN {
        w_host[199_991 * HIDDEN + c] = w_host[7 * HIDDEN + c];
        w_host[10 * HIDDEN + c] = w_host[2 * HIDDEN + c];
    }
    let w_dev = stream.clone_htod(&w_host)?;
    let w_nvfp4 = nvfp4.quantize(stream, &w_dev, PADDED_VOCAB, HIDDEN)?;

    // 1. B = 1 fused gemv_nvfp4_k2304_argmax -> argmax_partials vs gemv -> softcap -> argmax_wide_f32.
    let x_host = synthetic_hidden(256, 73);
    let x_dev = stream.clone_htod(&x_host)?;
    let mut logits_b1 = stream.alloc_zeros::<bf16>(PADDED_VOCAB)?;
    let mut capped_b1 = stream.alloc_zeros::<f32>(VOCAB)?;
    let mut argmax_scratch = ArgmaxScratch::new(stream)?;
    let mut want_b1 = stream.alloc_zeros::<i32>(1)?;
    let mut got_b1 = stream.alloc_zeros::<i32>(1)?;
    let mut p_vals_b1 = stream.alloc_zeros::<f32>(NVFP4_ARGMAX_MAX_BLOCKS)?;
    let mut p_idxs_b1 = stream.alloc_zeros::<i32>(NVFP4_ARGMAX_MAX_BLOCKS)?;

    for row in [0usize, 1, 7, 15] {
        let x_row = x_dev.slice(row * HIDDEN..(row + 1) * HIDDEN);
        nvfp4.gemv(
            stream,
            &mut [Nvfp4Gemv::new(&w_nvfp4, &x_row, &mut logits_b1)],
            Start::Serial,
        )?;
        s.elementwise
            .logit_softcap(stream, &logits_b1, &mut capped_b1, SOFTCAP)?;
        s.misc
            .argmax_wide_f32(stream, &capped_b1, &mut want_b1, &mut argmax_scratch)?;
        let want = stream.clone_dtoh(&want_b1)?[0];

        let n_partials = nvfp4.gemv_argmax(
            stream,
            &w_nvfp4,
            &x_row,
            VOCAB,
            &mut p_vals_b1,
            &mut p_idxs_b1,
            Start::Early,
        )?;
        s.misc.argmax_partials(
            stream,
            &p_vals_b1.slice(..n_partials),
            &p_idxs_b1.slice(..n_partials),
            &mut got_b1,
            n_partials,
            n_partials,
        )?;
        let got = stream.clone_dtoh(&got_b1)?[0];
        let vocab_i32 = i32::try_from(VOCAB)?;
        assert_eq!(
            got, want,
            "gemv_nvfp4_k2304_argmax row {row} differs from unfused"
        );
        assert!(
            (0..vocab_i32).contains(&got),
            "gemv_nvfp4_k2304_argmax returned out-of-vocab token {got}"
        );
    }
    eprintln!("gemv_nvfp4_k2304_argmax: 0-diff parity across 4 rows");

    // 2. Multi-row rows_nvfp4_k2304_argmax -> argmax_partials vs rows_nvfp4 -> softcap -> argmax_f32.
    let max_m = 256usize;
    let max_partials = max_m * VOCAB.div_ceil(16);
    let mut partial_vals = stream.alloc_zeros::<f32>(max_partials)?;
    let mut partial_idxs = stream.alloc_zeros::<i32>(max_partials)?;
    let mut got_dev = stream.alloc_zeros::<i32>(max_m)?;
    let mut want_dev = stream.alloc_zeros::<i32>(max_m)?;
    let mut ref_logits = stream.alloc_zeros::<bf16>(32 * PADDED_VOCAB)?;
    let mut ref_capped = stream.alloc_zeros::<f32>(32 * VOCAB)?;

    for &m in &[1usize, 4, 8, 32, 64, 256] {
        let mut want = Vec::with_capacity(m);
        for first in (0..m).step_by(32) {
            let n = 32.min(m - first);
            let x_slice = x_dev.slice(first * HIDDEN..(first + n) * HIDDEN);
            let mut out_slice = ref_logits.slice_mut(..n * PADDED_VOCAB);
            s.rows.run(
                stream,
                &mut [RowsProduct::nvfp4(&w_nvfp4, &x_slice, &mut out_slice)],
                n,
                Start::Serial,
            )?;
            s.elementwise.logit_softcap(
                stream,
                &ref_logits.slice(..n * PADDED_VOCAB),
                &mut ref_capped.slice_mut(..n * VOCAB),
                SOFTCAP,
            )?;
            s.misc.argmax_f32(
                stream,
                &ref_capped.slice(..n * VOCAB),
                &mut want_dev.slice_mut(..n),
                VOCAB,
            )?;
            want.extend(stream.clone_dtoh(&want_dev.slice(..n))?);
        }

        let n_partials = s.rows.run_argmax(
            stream,
            &RowsWeight::Nvfp4(&w_nvfp4),
            &x_dev.slice(..m * HIDDEN),
            VOCAB,
            &mut partial_vals,
            &mut partial_idxs,
            m,
            Start::Early,
        )?;
        s.misc.argmax_partials(
            stream,
            &partial_vals.slice(..m * n_partials),
            &partial_idxs.slice(..m * n_partials),
            &mut got_dev.slice_mut(..m),
            n_partials,
            n_partials,
        )?;
        let got = stream.clone_dtoh(&got_dev.slice(..m))?;
        let vocab_i32 = i32::try_from(VOCAB)?;
        assert_eq!(got, want, "nvfp4 rows_argmax m = {m} differs from unfused");
        assert!(
            got.iter().all(|&id| (0..vocab_i32).contains(&id)),
            "nvfp4 rows_argmax m = {m}: returned out-of-vocab token"
        );
        eprintln!("nvfp4 rows_argmax m = {m}: 0-diff parity ({n_partials} partials/row)");
    }

    // 3. Explicit tie-breaking + padding masking for NVFP4 gemv_argmax and rows_argmax.
    let mut w_tie = vec![bf16::ZERO; PADDED_VOCAB * HIDDEN];
    w_tie[5 * HIDDEN] = bf16::from_f32(4.0);
    w_tie[13 * HIDDEN] = bf16::from_f32(4.0);
    w_tie[150_005 * HIDDEN] = bf16::from_f32(4.0);
    w_tie[VOCAB * HIDDEN] = bf16::from_f32(32.0);
    w_tie[(PADDED_VOCAB - 1) * HIDDEN] = bf16::from_f32(64.0);
    let w_tie_dev = stream.clone_htod(&w_tie)?;
    let w_tie_nvfp4 = nvfp4.quantize(stream, &w_tie_dev, PADDED_VOCAB, HIDDEN)?;
    let mut x_tie = vec![bf16::ZERO; HIDDEN];
    x_tie[0] = bf16::from_f32(1.0);
    let x_tie_dev = stream.clone_htod(&x_tie)?;

    let np = nvfp4.gemv_argmax(
        stream,
        &w_tie_nvfp4,
        &x_tie_dev,
        VOCAB,
        &mut p_vals_b1,
        &mut p_idxs_b1,
        Start::Early,
    )?;
    s.misc.argmax_partials(
        stream,
        &p_vals_b1.slice(..np),
        &p_idxs_b1.slice(..np),
        &mut got_b1,
        np,
        np,
    )?;
    assert_eq!(stream.clone_dtoh(&got_b1)?[0], 5, "NVFP4 gemv_argmax tie/mask");

    let np_r = s.rows.run_argmax(
        stream,
        &RowsWeight::Nvfp4(&w_tie_nvfp4),
        &x_tie_dev,
        VOCAB,
        &mut partial_vals,
        &mut partial_idxs,
        1,
        Start::Early,
    )?;
    s.misc.argmax_partials(
        stream,
        &partial_vals.slice(..np_r),
        &partial_idxs.slice(..np_r),
        &mut got_b1,
        np_r,
        np_r,
    )?;
    assert_eq!(stream.clone_dtoh(&got_b1)?[0], 5, "NVFP4 rows_argmax tie/mask");
    Ok(())
}
