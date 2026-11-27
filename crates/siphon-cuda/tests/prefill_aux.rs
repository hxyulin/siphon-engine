//! Parity tests for the Wave 5 prefill auxiliary surgical winners:
//! - `moe_conv_residual_rows_2304` (29-row x 128-col uint4 tiled conv, 0 ulp bit-exact)
//! - `message_gate_rows_fp8_2304` / `message_gate_rows_2304` persistent-cluster group reuse
//!
//! Run on a GPU host with:
//! ```sh
//! cargo test --release -p siphon-cuda --test prefill_aux
//! ```

#![cfg(target_os = "linux")]

mod common;

use common::{assert_bf16_close, assert_f32_within_ulps, pseudo_random, pseudo_random_bf16};
use cudarc::driver::CudaSlice;
use half::bf16;
use siphon_cuda::{
    Gpu,
    ops::{
        QuantDest, QuantFormat,
        conv::Conv,
        elementwise::Elementwise,
        linear_fp8::{
            Fp8Gemm, Fp8GemmQuantized, Fp8Matrix, Fp8Rows, LinearFp8, Mxfp8Gemm,
            Mxfp8GemmQuantized,
        },
        misc::{GATE_SLOTS, GateRowsSlot, Misc},
        routing::{
            DepthRows, DepthRowsQuant, EXPERTS, NormDest, QuantNormDest, RouterRowsScratch,
            Routing, RowsPartial,
        },
    },
};

const HIDDEN: usize = 2304;
const GATE_RANK: usize = 64;
const TOP_K: usize = 8;
const EPS: f32 = 1e-6;

/// `moe_conv_residual_rows` against single-row `moe_conv_residual` across tile boundaries
/// (1, 2, 7, 29, 30, 61, 130 rows) with carried conv state. Bit-for-bit exact.
#[test]
fn moe_conv_residual_rows_parity_wide() -> anyhow::Result<()> {
    let gpu = Gpu::new(0)?;
    let stream = gpu.stream();
    let conv = Conv::load(gpu.context())?;
    let weight = stream.clone_htod(&pseudo_random_bf16(HIDDEN * 4, 301, 0.5, 0.0))?;
    let start = pseudo_random(3 * HIDDEN, 302, 2.0);
    let mut state_rows = stream.clone_htod(&start)?;
    let mut state_single = stream.clone_htod(&start)?;

    for (chunk, &rows) in [1usize, 2, 7, 29, 30, 61, 130].iter().enumerate() {
        let seed = 400 + 3 * chunk as u32;
        let experts = pseudo_random_bf16(rows * TOP_K * HIDDEN, seed, 2.0, 0.0);
        let routing: Vec<f32> = pseudo_random(rows * TOP_K, seed + 1, 1.0)
            .into_iter()
            .map(f32::abs)
            .collect();
        let residual = pseudo_random(rows * HIDDEN, seed + 2, 4.0);

        let mut want = Vec::with_capacity(rows * HIDDEN);
        for r in 0..rows {
            let e = stream.clone_htod(&experts[r * TOP_K * HIDDEN..(r + 1) * TOP_K * HIDDEN])?;
            let w = stream.clone_htod(&routing[r * TOP_K..(r + 1) * TOP_K])?;
            let res = stream.clone_htod(&residual[r * HIDDEN..(r + 1) * HIDDEN])?;
            let mut out = stream.alloc_zeros::<bf16>(HIDDEN)?;
            conv.moe_conv_residual(stream, &e, &w, &weight, &mut state_single, &res, &mut out)?;
            want.extend(stream.clone_dtoh(&out)?);
        }

        let experts_dev = stream.clone_htod(&experts)?;
        let routing_dev = stream.clone_htod(&routing)?;
        let residual_dev = stream.clone_htod(&residual)?;
        let mut combined = stream.alloc_zeros::<bf16>(rows * HIDDEN)?;
        let mut got = stream.alloc_zeros::<bf16>(rows * HIDDEN)?;
        conv.moe_conv_residual_rows(
            stream,
            &experts_dev,
            &routing_dev,
            &weight,
            &mut state_rows,
            &residual_dev,
            &mut combined,
            &mut got,
            rows,
        )?;
        let label = format!("moe_conv_residual_rows chunk {chunk} ({rows} rows)");
        assert_bf16_close(&label, &stream.clone_dtoh(&got)?, &want, 0);
        assert_f32_within_ulps(
            &format!("{label} state"),
            &stream.clone_dtoh(&state_rows)?,
            &stream.clone_dtoh(&state_single)?,
            0,
        );
    }
    Ok(())
}

/// Persistent-cluster `message_gate_rows_fp8_2304` and `message_gate_rows_2304` across
/// multiple 8-row groups (65 and 256 rows, 1 and 3 gates): bit-for-bit identical to
/// single-group launches.
#[test]
#[expect(clippy::too_many_lines, reason = "tests BF16 and FP8 persistent-cluster row invariance")]
fn message_gate_rows_persistent_cluster_parity() -> anyhow::Result<()> {
    let gpu = Gpu::new(0)?;
    let stream = gpu.stream();
    let misc = Misc::load(gpu.context())?;
    let fp8 = LinearFp8::load(gpu.context())?;

    let downs: Vec<CudaSlice<bf16>> = (0..GATE_SLOTS)
        .map(|g| {
            Ok(stream.clone_htod(&pseudo_random_bf16(
                GATE_RANK * 2 * HIDDEN,
                700 + g as u32,
                0.05,
                0.0,
            ))?)
        })
        .collect::<anyhow::Result<_>>()?;
    let downs_fp8: Vec<Fp8Matrix> = downs
        .iter()
        .map(|d| fp8.quantize_any(stream, d, GATE_RANK, 2 * HIDDEN))
        .collect::<anyhow::Result<_>>()?;
    let forgets: Vec<CudaSlice<bf16>> = (0..GATE_SLOTS)
        .map(|g| {
            Ok(stream.clone_htod(&pseudo_random_bf16(
                HIDDEN * GATE_RANK,
                710 + g as u32,
                0.3,
                0.0,
            ))?)
        })
        .collect::<anyhow::Result<_>>()?;
    let gains: Vec<CudaSlice<bf16>> = (0..GATE_SLOTS)
        .map(|g| {
            Ok(stream.clone_htod(&pseudo_random_bf16(
                HIDDEN * GATE_RANK,
                720 + g as u32,
                0.3,
                0.0,
            ))?)
        })
        .collect::<anyhow::Result<_>>()?;
    let norm_weights: Vec<CudaSlice<bf16>> = (0..GATE_SLOTS)
        .map(|g| {
            Ok(stream.clone_htod(&pseudo_random_bf16(
                HIDDEN,
                730 + g as u32,
                0.5,
                1.0,
            ))?)
        })
        .collect::<anyhow::Result<_>>()?;

    for &(rows, n_gates, use_fp8) in &[
        (65usize, 3usize, true),
        (256, 3, true),
        (256, 1, true),
        (65, 3, false),
    ] {
        let nb = 3;
        let cand = nb + 1;
        let probs_stride = 4 * cand;
        let x_state: Vec<CudaSlice<bf16>> = (0..n_gates)
            .map(|g| {
                Ok(stream.clone_htod(&pseudo_random_bf16(
                    rows * HIDDEN,
                    740 + g as u32,
                    1.5,
                    0.0,
                ))?)
            })
            .collect::<anyhow::Result<_>>()?;
        let x_msg: Vec<CudaSlice<bf16>> = (0..n_gates)
            .map(|g| {
                Ok(stream.clone_htod(&pseudo_random_bf16(
                    rows * HIDDEN,
                    750 + g as u32,
                    1.5,
                    0.0,
                ))?)
            })
            .collect::<anyhow::Result<_>>()?;
        let h_dev = stream.clone_htod(&pseudo_random(rows * HIDDEN, 760, 2.0))?;
        let m_dev = stream.clone_htod(&pseudo_random(rows * HIDDEN, 761, 2.0))?;
        let blocks_host = pseudo_random(nb * rows * HIDDEN, 762, 2.0);
        let blocks_dev = stream.clone_htod(&blocks_host)?;
        let probs_host: Vec<f32> = pseudo_random(rows * probs_stride, 763, 0.5)
            .into_iter()
            .map(f32::abs)
            .collect();
        let probs_dev = stream.clone_htod(&probs_host)?;

        let mut outs: Vec<CudaSlice<f32>> = (0..n_gates)
            .map(|_| Ok(stream.alloc_zeros(rows * HIDDEN)?))
            .collect::<anyhow::Result<_>>()?;
        let mut normeds: Vec<CudaSlice<bf16>> = (0..n_gates)
            .map(|_| Ok(stream.alloc_zeros(rows * HIDDEN)?))
            .collect::<anyhow::Result<_>>()?;
        {
            let mut slots: Vec<GateRowsSlot<'_>> = (0..n_gates)
                .zip(outs.iter_mut())
                .zip(normeds.iter_mut())
                .map(|((g, out), normed)| GateRowsSlot {
                    down: &downs[g],
                    down_fp8: use_fp8.then_some(&downs_fp8[g]),
                    forget: &forgets[g],
                    gain: &gains[g],
                    x_state: x_state[g].as_view(),
                    x_message: x_msg[g].as_view(),
                    h: h_dev.as_view(),
                    probs: probs_dev.slice(g * cand..),
                    norm: (&norm_weights[g], normed.as_view_mut()),
                    out: out.as_view_mut(),
                })
                .collect();
            misc.message_gate_rows(
                stream,
                &mut slots,
                [HIDDEN, HIDDEN],
                HIDDEN,
                (&m_dev, HIDDEN),
                &blocks_dev.as_view(),
                probs_stride,
                EPS,
                rows,
            )?;
        }

        // Check representative 8-row slices (first, middle, last) run standalone (1 cluster iter)
        // match the persistent multi-group launch bit for bit.
        for &r0 in &[0usize, 24, rows - rows.min(8)] {
            let chunk = (rows - r0).min(8);
            let mut chunk_blocks = Vec::with_capacity(nb * chunk * HIDDEN);
            for b in 0..nb {
                let start = (b * rows + r0) * HIDDEN;
                chunk_blocks.extend_from_slice(&blocks_host[start..start + chunk * HIDDEN]);
            }
            let chunk_blocks_dev = stream.clone_htod(&chunk_blocks)?;
            let mut ref_outs: Vec<CudaSlice<f32>> = (0..n_gates)
                .map(|_| Ok(stream.alloc_zeros(chunk * HIDDEN)?))
                .collect::<anyhow::Result<_>>()?;
            let mut ref_normeds: Vec<CudaSlice<bf16>> = (0..n_gates)
                .map(|_| Ok(stream.alloc_zeros(chunk * HIDDEN)?))
                .collect::<anyhow::Result<_>>()?;
            {
                let mut slots: Vec<GateRowsSlot<'_>> = (0..n_gates)
                    .zip(ref_outs.iter_mut())
                    .zip(ref_normeds.iter_mut())
                    .map(|((g, out), normed)| GateRowsSlot {
                        down: &downs[g],
                        down_fp8: use_fp8.then_some(&downs_fp8[g]),
                        forget: &forgets[g],
                        gain: &gains[g],
                        x_state: x_state[g].slice(r0 * HIDDEN..(r0 + chunk) * HIDDEN),
                        x_message: x_msg[g].slice(r0 * HIDDEN..(r0 + chunk) * HIDDEN),
                        h: h_dev.slice(r0 * HIDDEN..(r0 + chunk) * HIDDEN),
                        probs: probs_dev.slice(r0 * probs_stride + g * cand..),
                        norm: (&norm_weights[g], normed.as_view_mut()),
                        out: out.as_view_mut(),
                    })
                    .collect();
                misc.message_gate_rows(
                    stream,
                    &mut slots,
                    [HIDDEN, HIDDEN],
                    HIDDEN,
                    (&m_dev.slice(r0 * HIDDEN..(r0 + chunk) * HIDDEN), HIDDEN),
                    &chunk_blocks_dev.as_view(),
                    probs_stride,
                    EPS,
                    chunk,
                )?;
            }
            for g in 0..n_gates {
                let full_out = stream.clone_dtoh(&outs[g])?;
                let full_norm = stream.clone_dtoh(&normeds[g])?;
                let span = r0 * HIDDEN..(r0 + chunk) * HIDDEN;
                let label =
                    format!("gate_rows rows={rows} n={n_gates} fp8={use_fp8} r0={r0} g={g}");
                assert_f32_within_ulps(
                    &format!("{label} out"),
                    &full_out[span.clone()],
                    &stream.clone_dtoh(&ref_outs[g])?,
                    0,
                );
                assert_bf16_close(
                    &format!("{label} normed"),
                    &full_norm[span],
                    &stream.clone_dtoh(&ref_normeds[g])?,
                    0,
                );
            }
        }
    }
    Ok(())
}

/// 2-term BF16 tensor-core `moe_router_logits_tc_rows_2304` + warp bitonic
/// `moe_router_route_rows_2304` match single-row `moe_router` across adaptive K-splits
/// (`rows = 64, 65, 512, 2048`), including exact tie-breaking on duplicated expert weights.
#[test]
fn moe_router_rows_tc_parity() -> anyhow::Result<()> {
    let gpu = Gpu::new(0)?;
    let stream = gpu.stream();
    let routing = Routing::load(gpu.context())?;

    let mut router_host = pseudo_random(EXPERTS * HIDDEN, 901, 0.03);
    // Duplicate pairs of expert weights so ties exercise exact lower-index tie-breaking.
    for e in [3usize, 10, 20, 40] {
        let partner = e ^ 64;
        router_host.copy_within(e * HIDDEN..(e + 1) * HIDDEN, partner * HIDDEN);
    }
    let mut bias_host = pseudo_random(EXPERTS, 902, 0.05);
    for e in [3usize, 10, 20, 40] {
        bias_host[e ^ 64] = bias_host[e];
    }
    let router = stream.clone_htod(&router_host)?;
    let bias = stream.clone_htod(&bias_host)?;

    for &rows in &[64usize, 65, 512, 2048] {
        let x_host = pseudo_random_bf16(rows * HIDDEN, 910 + rows as u32, 1.0, 0.0);
        let x_dev = stream.clone_htod(&x_host)?;
        let mut logits_dev = stream.alloc_zeros::<f32>(rows * EXPERTS)?;
        let mut indices_dev = stream.alloc_zeros::<i32>(rows * TOP_K)?;
        let mut weights_dev = stream.alloc_zeros::<f32>(rows * TOP_K)?;
        let mut scratch = RouterRowsScratch::new(stream, rows)?;
        routing.moe_router_rows(
            stream,
            &x_dev,
            &router,
            &bias,
            &mut logits_dev,
            &mut indices_dev,
            &mut weights_dev,
            &mut scratch,
            8.0,
            rows,
        )?;
        let got_logits = stream.clone_dtoh(&logits_dev)?;
        let got_indices = stream.clone_dtoh(&indices_dev)?;
        let got_weights = stream.clone_dtoh(&weights_dev)?;

        // Check representative rows (start, tile boundaries, end) against single-row F32 router.
        let sample_rows: Vec<usize> = [0, 1, 31, 63, rows / 2, rows - 1]
            .into_iter()
            .filter(|&r| r < rows)
            .collect();
        for r in sample_rows {
            // Reference logits via router_logits F32 + expert_routing directly on the same x_host.
            let x_row_bf16 = stream.clone_htod(&x_host[r * HIDDEN..(r + 1) * HIDDEN])?;
            let mut ref_logits_dev = stream.alloc_zeros::<f32>(EXPERTS)?;
            routing.router_logits(
                stream,
                &x_row_bf16,
                &router,
                &mut ref_logits_dev,
                siphon_cuda::ops::routing::RouterMatmul::F32,
            )?;
            let mut ref_idx_dev = stream.alloc_zeros::<i32>(TOP_K)?;
            let mut ref_w_dev = stream.alloc_zeros::<f32>(TOP_K)?;
            routing.expert_routing(
                stream,
                &ref_logits_dev,
                &bias,
                &mut ref_idx_dev,
                &mut ref_w_dev,
                8.0,
            )?;
            let ref_logits = stream.clone_dtoh(&ref_logits_dev)?;
            let ref_idx = stream.clone_dtoh(&ref_idx_dev)?;
            let ref_w = stream.clone_dtoh(&ref_w_dev)?;

            let gl = &got_logits[r * EXPERTS..(r + 1) * EXPERTS];
            for (e, (&g, &w)) in gl.iter().zip(&ref_logits).enumerate() {
                let err = (g - w).abs();
                assert!(
                    err <= 1e-5,
                    "router_tc rows={rows} row={r} expert={e}: got {g:e}, want {w:e} (err {err:e})"
                );
            }
            let gi = &got_indices[r * TOP_K..(r + 1) * TOP_K];
            let gw = &got_weights[r * TOP_K..(r + 1) * TOP_K];
            let mut gp: Vec<(i32, f32)> = gi.iter().copied().zip(gw.iter().copied()).collect();
            let mut wp: Vec<(i32, f32)> = ref_idx.iter().copied().zip(ref_w.iter().copied()).collect();
            gp.sort_by_key(|&(e, _)| e);
            wp.sort_by_key(|&(e, _)| e);
            for k in 0..TOP_K {
                assert_eq!(
                    gp[k].0, wp[k].0,
                    "router_tc rows={rows} row={r}: expert mismatch got={gi:?} want={ref_idx:?}"
                );
                let werr = (gp[k].1 - wp[k].1).abs();
                assert!(
                    werr <= 2e-6,
                    "router_tc rows={rows} row={r} slot={k}: weight got {} want {} (err {werr:e})",
                    gp[k].1,
                    wp[k].1
                );
            }
        }
    }
    Ok(())
}

/// Fused `attention_gate_quant_mxfp8` + `gemm_mxfp8_quantized` matches unfused
/// `attention_gate` + `gemm_mxfp8` bit for bit (0 ULP).
#[test]
fn attention_gate_quant_mxfp8_parity() -> anyhow::Result<()> {
    const Q_DIMS: usize = 24 * 128;
    if !siphon_cuda::BLOCK_SCALED_MMA {
        return Ok(());
    }
    let gpu = Gpu::new(0)?;
    let stream = gpu.stream();
    let elem = Elementwise::load(gpu.context())?;
    let fp8 = LinearFp8::load(gpu.context())?;

    let wo_bf16 = stream.clone_htod(&pseudo_random_bf16(HIDDEN * Q_DIMS, 950, 0.05, 0.0))?;
    let wo_mxfp8 = fp8.quantize_mxfp8(stream, &wo_bf16, HIDDEN, Q_DIMS)?;

    for &rows in &[2usize, 33, 130, 512] {
        let attn = stream.clone_htod(&pseudo_random_bf16(rows * Q_DIMS, 951 + rows as u32, 2.0, 0.0))?;
        let gate = stream.clone_htod(&pseudo_random_bf16(rows * Q_DIMS, 961 + rows as u32, 3.0, 0.0))?;

        let mut gated = stream.alloc_zeros::<bf16>(rows * Q_DIMS)?;
        let mut ref_out = stream.alloc_zeros::<bf16>(rows * HIDDEN)?;
        let mut ref_scratch = Fp8Rows::new(stream, rows * Q_DIMS)?;
        elem.attention_gate(stream, &attn, &gate, &mut gated)?;
        fp8.gemm_mxfp8(
            stream,
            &mut [Mxfp8Gemm {
                weight: &wo_mxfp8,
                x: gated.as_view(),
                out: ref_out.as_view_mut(),
            }],
            rows,
            &mut ref_scratch,
        )?;

        let mut got_out = stream.alloc_zeros::<bf16>(rows * HIDDEN)?;
        let mut got_scratch = Fp8Rows::new(stream, rows * Q_DIMS)?;
        {
            let mut slots = got_scratch.split_slots_mut(rows, Q_DIMS, 1)?;
            let (mut vals, mut scls) = slots.pop().expect("1 slot");
            elem.attention_gate_quant_mxfp8(stream, &attn, &gate, &mut vals, &mut scls)?;
        }
        fp8.gemm_mxfp8_quantized(
            stream,
            &mut [Mxfp8GemmQuantized {
                weight: &wo_mxfp8,
                x_offset: 0,
                out: got_out.as_view_mut(),
            }],
            rows,
            &got_scratch,
        )?;

        assert_eq!(
            stream.clone_dtoh(got_scratch.values())?,
            stream.clone_dtoh(ref_scratch.values())?,
            "attention_gate_quant_mxfp8 rows={rows}: MXFP8 values"
        );
        assert_eq!(
            stream.clone_dtoh(got_scratch.scales())?,
            stream.clone_dtoh(ref_scratch.scales())?,
            "attention_gate_quant_mxfp8 rows={rows}: MXFP8 scales"
        );
        assert_bf16_close(
            &format!("attention_gate_quant_mxfp8 rows={rows}: gemm_mxfp8 out"),
            &stream.clone_dtoh(&got_out)?,
            &stream.clone_dtoh(&ref_out)?,
            0,
        );
    }
    Ok(())
}

/// Fused strided `depth_attention_rows_quant` (`NQ = 4` message halves at col 2304 + `NQ = 3`
/// state halves at col 0, `dst_stride = 4608`) + `gemm_quantized` matches unfused
/// `depth_attention_rows` -> BF16 `[rows, 4608]` -> `gemm` bit for bit (0 ULP).
#[test]
#[expect(
    clippy::too_many_lines,
    reason = "end-to-end unfused vs fused strided NQ=4+NQ=3 message-gate parity"
)]
fn depth_attention_rows_quant_strided_nq4_parity() -> anyhow::Result<()> {
    const ROWS: usize = 130;
    const NB: usize = 2;
    if !siphon_cuda::BLOCK_SCALED_MMA {
        return Ok(());
    }
    let gpu = Gpu::new(0)?;
    let stream = gpu.stream();
    let routing = Routing::load(gpu.context())?;
    let fp8 = LinearFp8::load(gpu.context())?;

    let blocks = stream.clone_htod(&pseudo_random(NB * ROWS * HIDDEN, 970, 1.5))?;
    let bank_normed = stream.clone_htod(&pseudo_random(ROWS * HIDDEN, 971, 1.5))?;
    let partial_bf16 = stream.clone_htod(&pseudo_random_bf16(ROWS * HIDDEN, 972, 1.5, 0.0))?;
    let q4 = stream.clone_htod(&pseudo_random(4 * HIDDEN, 973, 0.5))?;
    let q3 = stream.clone_htod(&pseudo_random(3 * HIDDEN, 974, 0.5))?;
    let downs_fp8: Vec<Fp8Matrix> = (0..4u32)
        .map(|g| {
            let w = stream.clone_htod(&pseudo_random_bf16(
                GATE_RANK * 2 * HIDDEN,
                980 + g,
                0.05,
                0.0,
            ))?;
            fp8.quantize_any(stream, &w, GATE_RANK, 2 * HIDDEN)
        })
        .collect::<anyhow::Result<_>>()?;

    // Reference: unfused BF16 strided writes + LinearFp8::gemm.
    let mut ref_inputs: [CudaSlice<bf16>; 4] = [
        stream.alloc_zeros(ROWS * 2 * HIDDEN)?,
        stream.alloc_zeros(ROWS * 2 * HIDDEN)?,
        stream.alloc_zeros(ROWS * 2 * HIDDEN)?,
        stream.alloc_zeros(ROWS * 2 * HIDDEN)?,
    ];
    let mut ref_probs = stream.alloc_zeros::<f32>(ROWS * 4 * (NB + 1))?;
    {
        let dests = ref_inputs.each_mut().map(|x| {
            let end = (ROWS - 1) * (2 * HIDDEN) + HIDDEN + HIDDEN;
            Some(NormDest {
                out: x.slice_mut(HIDDEN..end),
                weight: None,
            })
        });
        routing.depth_attention_rows(
            stream,
            DepthRows {
                blocks: blocks.as_view(),
                partial: RowsPartial::Normalized(bank_normed.as_view()),
                normed_partial: None,
                queries: &q4,
                out: None,
                probs: Some(ref_probs.as_view_mut()),
                dests,
                dst_stride: 2 * HIDDEN,
                rows: ROWS,
            },
            EPS,
        )?;
    }
    {
        let [g0, g1, g2, _] = &mut ref_inputs;
        let end = (ROWS - 1) * (2 * HIDDEN) + HIDDEN;
        let [d0, d1, d2] = [g0, g1, g2].map(|g| {
            Some(NormDest {
                out: g.slice_mut(..end),
                weight: None,
            })
        });
        routing.depth_attention_rows(
            stream,
            DepthRows {
                blocks: blocks.as_view(),
                partial: RowsPartial::Bf16(partial_bf16.as_view()),
                normed_partial: None,
                queries: &q3,
                out: None,
                probs: None,
                dests: [d0, d1, d2, None],
                dst_stride: 2 * HIDDEN,
                rows: ROWS,
            },
            EPS,
        )?;
    }
    let mut ref_down = stream.alloc_zeros::<bf16>(3 * ROWS * GATE_RANK)?;
    let mut ref_dense = Fp8Rows::new(stream, 4 * ROWS * 2 * HIDDEN)?;
    {
        let (d0, mut rest) = ref_down.split_at_mut(ROWS * GATE_RANK);
        let (d1, d2) = rest.split_at_mut(ROWS * GATE_RANK);
        fp8.gemm(
            stream,
            &mut [
                Fp8Gemm {
                    weight: &downs_fp8[0],
                    x: ref_inputs[0].as_view(),
                    out: d0,
                },
                Fp8Gemm {
                    weight: &downs_fp8[1],
                    x: ref_inputs[1].as_view(),
                    out: d1,
                },
                Fp8Gemm {
                    weight: &downs_fp8[2],
                    x: ref_inputs[2].as_view(),
                    out: d2,
                },
            ],
            ROWS,
            &mut ref_dense,
        )?;
    }

    // Fused: strided depth_attention_rows_quant directly into Fp8Rows + gemm_quantized.
    let mut got_dense = Fp8Rows::new(stream, 4 * ROWS * 2 * HIDDEN)?;
    let mut got_probs = stream.alloc_zeros::<f32>(ROWS * 4 * (NB + 1))?;
    {
        let [s0, s1, s2, s3]: [_; 4] = got_dense
            .split_slots_range_mut(ROWS, 2 * HIDDEN, 0, 4)?
            .try_into()
            .map_err(|_| anyhow::anyhow!("expected 4 slots"))?;
        let dests = [s0, s1, s2, s3].map(|(values, scales)| {
            Some(QuantNormDest {
                out: None,
                weight: None,
                quant: QuantDest {
                    format: QuantFormat::Mxfp8,
                    values,
                    scales,
                    outer: None,
                },
            })
        });
        routing.depth_attention_rows_quant(
            stream,
            DepthRowsQuant {
                blocks: blocks.as_view(),
                partial: RowsPartial::Normalized(bank_normed.as_view()),
                normed_partial: None,
                queries: &q4,
                out: None,
                probs: Some(got_probs.as_view_mut()),
                dests,
                dst_stride: 2 * HIDDEN,
                quant_col_offset: HIDDEN,
                rows: ROWS,
            },
            EPS,
        )?;
    }
    {
        let [s0, s1, s2]: [_; 3] = got_dense
            .split_slots_range_mut(ROWS, 2 * HIDDEN, 0, 3)?
            .try_into()
            .map_err(|_| anyhow::anyhow!("expected 3 slots"))?;
        let [d0, d1, d2] = [s0, s1, s2].map(|(values, scales)| {
            Some(QuantNormDest {
                out: None,
                weight: None,
                quant: QuantDest {
                    format: QuantFormat::Mxfp8,
                    values,
                    scales,
                    outer: None,
                },
            })
        });
        routing.depth_attention_rows_quant(
            stream,
            DepthRowsQuant {
                blocks: blocks.as_view(),
                partial: RowsPartial::Bf16(partial_bf16.as_view()),
                normed_partial: None,
                queries: &q3,
                out: None,
                probs: None,
                dests: [d0, d1, d2, None],
                dst_stride: 2 * HIDDEN,
                quant_col_offset: 0,
                rows: ROWS,
            },
            EPS,
        )?;
    }
    let mut got_down = stream.alloc_zeros::<bf16>(3 * ROWS * GATE_RANK)?;
    {
        let (d0, mut rest) = got_down.split_at_mut(ROWS * GATE_RANK);
        let (d1, d2) = rest.split_at_mut(ROWS * GATE_RANK);
        let slot_vals = ROWS * 2 * HIDDEN;
        fp8.gemm_quantized(
            stream,
            &mut [
                Fp8GemmQuantized {
                    weight: &downs_fp8[0],
                    x_offset: 0,
                    out: d0,
                },
                Fp8GemmQuantized {
                    weight: &downs_fp8[1],
                    x_offset: slot_vals,
                    out: d1,
                },
                Fp8GemmQuantized {
                    weight: &downs_fp8[2],
                    x_offset: 2 * slot_vals,
                    out: d2,
                },
            ],
            ROWS,
            &got_dense,
        )?;
    }

    assert_f32_within_ulps(
        "depth_attention_rows_quant strided nq4 probs",
        &stream.clone_dtoh(&got_probs)?,
        &stream.clone_dtoh(&ref_probs)?,
        0,
    );
    assert_eq!(
        &stream.clone_dtoh(got_dense.values())?[..3 * ROWS * 2 * HIDDEN],
        &stream.clone_dtoh(ref_dense.values())?[..3 * ROWS * 2 * HIDDEN],
        "depth_attention_rows_quant strided nq4 values"
    );
    assert_eq!(
        &stream.clone_dtoh(got_dense.scales())?[..3 * ROWS * 2 * HIDDEN / 32],
        &stream.clone_dtoh(ref_dense.scales())?[..3 * ROWS * 2 * HIDDEN / 32],
        "depth_attention_rows_quant strided nq4 scales"
    );
    assert_bf16_close(
        "depth_attention_rows_quant strided nq4 down",
        &stream.clone_dtoh(&got_down)?,
        &stream.clone_dtoh(&ref_down)?,
        0,
    );
    Ok(())
}
