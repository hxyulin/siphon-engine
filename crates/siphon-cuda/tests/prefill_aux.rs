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
        conv::Conv,
        linear_fp8::{Fp8Matrix, LinearFp8},
        misc::{GATE_SLOTS, GateRowsSlot, Misc},
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
