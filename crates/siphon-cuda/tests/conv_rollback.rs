//! GPU parity tests for zero-backup in-place `conv_rollback` and `conv_rollback_batch`
//! (`kernels/spec.cu`) and the verify-mode state preservation in
//! `moe_conv_residual_step_rows_2304` and `moe_conv_residual_batch_verify_2304`
//! (`kernels/conv.cu`).
//!
//! Asserts 0-ULP bit-exact parity against a CPU reference across every `accepted` in `0..8`.

#![cfg(target_os = "linux")]

mod common;

use std::sync::Arc;

use common::{assert_bf16_close, pseudo_random, pseudo_random_bf16};
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr as _};
use half::bf16;
use siphon_cuda::{
    Gpu,
    ops::{
        conv::{Conv, VisitConvs},
        spec::{
            BATCH_STATE_WORDS, HIDDEN, KV_DIMS, RESULT_LEN, STATE_ROWS, STREAM_DIMS, STREAMS, Spec,
            VERIFY_ROWS, VISIT_CHANNELS,
        },
    },
};

const TOP_K: usize = 8;

/// Channel offset of conv stream `s` inside one `VISIT_CHANNELS`-wide row of `inputs`.
const fn stream_offset(s: usize) -> usize {
    if s < 2 {
        s * KV_DIMS
    } else {
        2 * KV_DIMS + (s - 2) * HIDDEN
    }
}

/// CPU reference for in-place conv rollback of one `(visit, stream)` state `[3, dims]`.
///
/// `rows_base` is the starting index of this `(visit[, seq])`'s `[VERIFY_ROWS, VISIT_CHANNELS]`
/// slice inside `inputs`.
fn cpu_conv_rollback(
    old_state: &[f32],
    inputs: &[bf16],
    rows_base: usize,
    stream_idx: usize,
    accepted: usize,
) -> Vec<f32> {
    let dims = STREAM_DIMS[stream_idx];
    let offset = stream_offset(stream_idx);
    let keep = accepted + 1;
    let mut out = vec![0.0f32; STATE_ROWS * dims];
    for r in 0..STATE_ROWS {
        let i = keep + r;
        let dst = &mut out[r * dims..(r + 1) * dims];
        if i < STATE_ROWS {
            dst.copy_from_slice(&old_state[i * dims..(i + 1) * dims]);
        } else {
            let src_row = rows_base + (i - STATE_ROWS) * VISIT_CHANNELS + offset;
            for (d, &v) in dst.iter_mut().zip(&inputs[src_row..src_row + dims]) {
                *d = v.to_f32();
            }
        }
    }
    out
}

/// Asserts exact bit equality (`to_bits()`) for every F32 element.
fn assert_f32_bits_exact(label: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{label}: length mismatch");
    for (idx, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert_eq!(
            g.to_bits(),
            w.to_bits(),
            "{label}[{idx}]: got {g} ({:#010x}), want {w} ({:#010x})",
            g.to_bits(),
            w.to_bits()
        );
    }
}

/// Deterministic inputs for a `total_rows`-row MoE + conv kernel test.
struct MoeInputs {
    experts: CudaSlice<bf16>,
    routing: CudaSlice<f32>,
    residual: CudaSlice<f32>,
    proj: CudaSlice<bf16>,
    k_raw: CudaSlice<bf16>,
    v_raw: CudaSlice<bf16>,
}

impl MoeInputs {
    fn new(stream: &Arc<CudaStream>, total_rows: usize, seed: u32) -> anyhow::Result<Self> {
        let routing: Vec<f32> = pseudo_random(total_rows * TOP_K, seed + 1, 1.0)
            .into_iter()
            .map(f32::abs)
            .collect();
        Ok(Self {
            experts: stream.clone_htod(&pseudo_random_bf16(
                total_rows * TOP_K * HIDDEN,
                seed,
                2.0,
                0.0,
            ))?,
            routing: stream.clone_htod(&routing)?,
            residual: stream.clone_htod(&pseudo_random(total_rows * HIDDEN, seed + 2, 4.0))?,
            proj: stream.clone_htod(&pseudo_random_bf16(
                total_rows * HIDDEN,
                seed + 3,
                2.0,
                0.0,
            ))?,
            k_raw: stream.clone_htod(&pseudo_random_bf16(
                total_rows * KV_DIMS,
                seed + 4,
                2.0,
                0.0,
            ))?,
            v_raw: stream.clone_htod(&pseudo_random_bf16(
                total_rows * KV_DIMS,
                seed + 5,
                2.0,
                0.0,
            ))?,
        })
    }
}

fn clone_four_states(
    stream: &Arc<CudaStream>,
    init: &[Vec<f32>; STREAMS],
) -> anyhow::Result<[CudaSlice<f32>; STREAMS]> {
    Ok([
        stream.clone_htod(&init[0])?,
        stream.clone_htod(&init[1])?,
        stream.clone_htod(&init[2])?,
        stream.clone_htod(&init[3])?,
    ])
}

#[test]
fn conv_rollback_inplace_matches_cpu_reference() -> anyhow::Result<()> {
    const VISITS: usize = 4;
    let gpu = Gpu::new(0)?;
    let stream = gpu.stream();
    let spec = Spec::load(gpu.context())?;

    let inputs_host = pseudo_random_bf16(VISITS * VERIFY_ROWS * VISIT_CHANNELS, 101, 4.0, 0.25);
    let inputs_dev = stream.clone_htod(&inputs_host)?;

    for accepted in 0..VERIFY_ROWS {
        let mut state_bufs: Vec<CudaSlice<f32>> = Vec::with_capacity(VISITS * STREAMS);
        let mut initial_host: Vec<Vec<f32>> = Vec::with_capacity(VISITS * STREAMS);
        let mut ptrs: Vec<u64> = Vec::with_capacity(VISITS * STREAMS);

        for v in 0..VISITS {
            for (s, &dims) in STREAM_DIMS.iter().enumerate() {
                let seed = 1000 + u32::try_from(accepted * 100 + v * STREAMS + s)?;
                let init = pseudo_random(STATE_ROWS * dims, seed, 3.5);
                let buf = stream.clone_htod(&init)?;
                ptrs.push(buf.device_ptr(stream).0);
                state_bufs.push(buf);
                initial_host.push(init);
            }
        }

        let ptrs_dev = stream.clone_htod(&ptrs)?;
        let mut result_host = vec![0i32; RESULT_LEN];
        result_host[0] = i32::try_from(accepted)?;
        let result_dev = stream.clone_htod(&result_host)?;

        spec.conv_rollback(stream, &ptrs_dev, &inputs_dev, &result_dev)?;

        for v in 0..VISITS {
            let rows_base = v * VERIFY_ROWS * VISIT_CHANNELS;
            for s in 0..STREAMS {
                let idx = v * STREAMS + s;
                let got = stream.clone_dtoh(&state_bufs[idx])?;
                let want =
                    cpu_conv_rollback(&initial_host[idx], &inputs_host, rows_base, s, accepted);
                assert_f32_bits_exact(
                    &format!("conv_rollback accepted={accepted} visit={v} stream={s}"),
                    &got,
                    &want,
                );
            }
        }
    }
    Ok(())
}

#[test]
fn conv_rollback_batch_inplace_matches_cpu_reference() -> anyhow::Result<()> {
    const VISITS: usize = 3;
    const MAX_SEQS: usize = 10;
    const ACTIVE_SEQS: usize = 8;

    let gpu = Gpu::new(0)?;
    let stream = gpu.stream();
    let spec = Spec::load(gpu.context())?;

    let inputs_host = pseudo_random_bf16(
        VISITS * MAX_SEQS * VERIFY_ROWS * VISIT_CHANNELS,
        202,
        4.0,
        -0.125,
    );
    let inputs_dev = stream.clone_htod(&inputs_host)?;

    // Run 8 rotations so every sequence slot exercises every accepted in 0..8,
    // while slots ACTIVE_SEQS..MAX_SEQS remain untouched.
    for shift in 0..VERIFY_ROWS {
        let mut state_bufs: Vec<CudaSlice<f32>> = Vec::with_capacity(VISITS * MAX_SEQS * STREAMS);
        let mut initial_host: Vec<Vec<f32>> = Vec::with_capacity(VISITS * MAX_SEQS * STREAMS);
        let mut table = vec![0u64; VISITS * MAX_SEQS * BATCH_STATE_WORDS];

        for v in 0..VISITS {
            for seq in 0..MAX_SEQS {
                let entry = v * MAX_SEQS + seq;
                for (s, &dims) in STREAM_DIMS.iter().enumerate() {
                    let seed = 5000 + u32::try_from(shift * 500 + entry * STREAMS + s)?;
                    let init = pseudo_random(STATE_ROWS * dims, seed, 3.0);
                    let buf = stream.clone_htod(&init)?;
                    table[entry * BATCH_STATE_WORDS + 2 + s] = buf.device_ptr(stream).0;
                    state_bufs.push(buf);
                    initial_host.push(init);
                }
            }
        }

        let tables_dev = stream.clone_htod(&table)?;
        let mut results_host = vec![0i32; ACTIVE_SEQS * RESULT_LEN];
        for seq in 0..ACTIVE_SEQS {
            results_host[seq * RESULT_LEN] = i32::try_from((seq + shift) % VERIFY_ROWS)?;
        }
        let results_dev = stream.clone_htod(&results_host)?;

        spec.conv_rollback_batch(
            stream,
            &tables_dev,
            VISITS,
            MAX_SEQS,
            ACTIVE_SEQS,
            &inputs_dev,
            &results_dev.as_view(),
        )?;

        for v in 0..VISITS {
            for seq in 0..MAX_SEQS {
                let entry = v * MAX_SEQS + seq;
                let rows_base = entry * VERIFY_ROWS * VISIT_CHANNELS;
                for s in 0..STREAMS {
                    let idx = entry * STREAMS + s;
                    let got = stream.clone_dtoh(&state_bufs[idx])?;
                    let want = if seq < ACTIVE_SEQS {
                        let accepted = (seq + shift) % VERIFY_ROWS;
                        cpu_conv_rollback(&initial_host[idx], &inputs_host, rows_base, s, accepted)
                    } else {
                        initial_host[idx].clone()
                    };
                    assert_f32_bits_exact(
                        &format!(
                            "conv_rollback_batch shift={shift} visit={v} seq={seq} stream={s}"
                        ),
                        &got,
                        &want,
                    );
                }
            }
        }
    }
    Ok(())
}

fn run_step_rows(
    stream: &Arc<CudaStream>,
    conv: &Conv,
    inp: &MoeInputs,
    weight: &CudaSlice<bf16>,
    states: &mut [CudaSlice<f32>; STREAMS],
    inputs: Option<&mut CudaSlice<bf16>>,
    rows: usize,
) -> anyhow::Result<CudaSlice<bf16>> {
    let mut out = stream.alloc_zeros::<bf16>(rows * HIDDEN)?;
    let [k_s, v_s, p_s, m_s] = states;
    conv.moe_conv_residual_step_rows(
        stream,
        &inp.experts,
        &inp.routing,
        weight,
        m_s,
        &inp.residual,
        &mut out,
        VisitConvs {
            proj: inp.proj.as_view(),
            proj_state: p_s,
            k_raw: inp.k_raw.as_view(),
            v_raw: inp.v_raw.as_view(),
            k_state: k_s,
            v_state: v_s,
            inputs: inputs.map(CudaSlice::as_view_mut),
        },
        rows,
    )?;
    Ok(out)
}

/// End-to-end check for single-stream `moe_conv_residual_step_rows` (verify mode) +
/// in-place `conv_rollback` against shifting mode (`inputs = None`), across widths 1, 2, 3,
/// 5 and 8.
#[test]
fn step_rows_verify_mode_preserves_states_and_rollback_is_exact() -> anyhow::Result<()> {
    let gpu = Gpu::new(0)?;
    let stream = gpu.stream();
    let conv = Conv::load(gpu.context())?;
    let spec = Spec::load(gpu.context())?;
    let weight = stream.clone_htod(&pseudo_random_bf16(HIDDEN * 4, 301, 0.5, 0.0))?;
    let init_states: [Vec<f32>; STREAMS] =
        std::array::from_fn(|s| pseudo_random(STATE_ROWS * STREAM_DIMS[s], 310 + s as u32, 2.0));

    for rows in [1usize, 2, 3, 5, 8] {
        let inp = MoeInputs::new(stream, rows, 400 + u32::try_from(rows * 10)?)?;
        let mut ref_states = clone_four_states(stream, &init_states)?;
        let ref_out = run_step_rows(stream, &conv, &inp, &weight, &mut ref_states, None, rows)?;

        let mut ver_states = clone_four_states(stream, &init_states)?;
        let mut inputs = stream.alloc_zeros::<bf16>(VERIFY_ROWS * VISIT_CHANNELS)?;
        let ver_out = run_step_rows(
            stream,
            &conv,
            &inp,
            &weight,
            &mut ver_states,
            Some(&mut inputs),
            rows,
        )?;

        assert_bf16_close(
            &format!("step_rows out rows={rows}"),
            &stream.clone_dtoh(&ver_out)?,
            &stream.clone_dtoh(&ref_out)?,
            0,
        );
        for (s, (got_buf, want)) in ver_states.iter().zip(&init_states).enumerate() {
            assert_f32_bits_exact(
                &format!("step_rows preserved state rows={rows} stream={s}"),
                &stream.clone_dtoh(got_buf)?,
                want,
            );
        }

        let ptrs: Vec<u64> = ver_states.iter().map(|b| b.device_ptr(stream).0).collect();
        let ptrs_dev = stream.clone_htod(&ptrs)?;
        let mut result_host = vec![0i32; RESULT_LEN];
        result_host[0] = i32::try_from(rows - 1)?;
        let result_dev = stream.clone_htod(&result_host)?;
        spec.conv_rollback(stream, &ptrs_dev, &inputs, &result_dev)?;

        for (s, (got_buf, want_buf)) in ver_states.iter().zip(&ref_states).enumerate() {
            assert_f32_bits_exact(
                &format!("step_rows rollback rows={rows} stream={s}"),
                &stream.clone_dtoh(got_buf)?,
                &stream.clone_dtoh(want_buf)?,
            );
        }
    }
    Ok(())
}

type BatchStateAlloc = (Vec<Vec<f32>>, Vec<CudaSlice<f32>>, CudaSlice<u64>);

fn alloc_batch_states(
    stream: &Arc<CudaStream>,
    seqs: usize,
    seed: u32,
) -> anyhow::Result<BatchStateAlloc> {
    let mut init_host = Vec::with_capacity(seqs * STREAMS);
    let mut bufs = Vec::with_capacity(seqs * STREAMS);
    let mut table = vec![0u64; seqs * BATCH_STATE_WORDS];
    for seq in 0..seqs {
        for (s, &dims) in STREAM_DIMS.iter().enumerate() {
            let init = pseudo_random(STATE_ROWS * dims, seed + u32::try_from(seq * STREAMS + s)?, 2.5);
            let buf = stream.clone_htod(&init)?;
            table[seq * BATCH_STATE_WORDS + 2 + s] = buf.device_ptr(stream).0;
            bufs.push(buf);
            init_host.push(init);
        }
    }
    Ok((init_host, bufs, stream.clone_htod(&table)?))
}

/// End-to-end check for batched `moe_conv_residual_batch_verify` (verify mode) + in-place
/// `conv_rollback_batch` against shifting mode (`inputs = None`), across widths 1, 2, 3 and 8.
#[test]
fn batch_verify_preserves_states_and_rollback_is_exact() -> anyhow::Result<()> {
    const SEQS: usize = 4;
    let gpu = Gpu::new(0)?;
    let stream = gpu.stream();
    let conv = Conv::load(gpu.context())?;
    let spec = Spec::load(gpu.context())?;
    let weight = stream.clone_htod(&pseudo_random_bf16(HIDDEN * 4, 601, 0.5, 0.0))?;

    for width in [1usize, 2, 3, 8] {
        let seed = 700 + u32::try_from(width * 10)?;
        let inp = MoeInputs::new(stream, SEQS * width, seed)?;
        let (_, ref_bufs, ref_table) = alloc_batch_states(stream, SEQS, seed + 20)?;
        let (init_host, ver_bufs, ver_table) = alloc_batch_states(stream, SEQS, seed + 20)?;

        let mut ref_out = stream.alloc_zeros::<bf16>(SEQS * width * HIDDEN)?;
        conv.moe_conv_residual_batch_verify(
            stream,
            &inp.experts,
            &inp.routing,
            &weight,
            &inp.residual,
            &mut ref_out,
            [inp.proj.as_view(), inp.k_raw.as_view(), inp.v_raw.as_view()],
            &ref_table.as_view(),
            None,
            SEQS,
            width,
        )?;

        let mut ver_out = stream.alloc_zeros::<bf16>(SEQS * width * HIDDEN)?;
        let mut inputs = stream.alloc_zeros::<bf16>(SEQS * VERIFY_ROWS * VISIT_CHANNELS)?;
        conv.moe_conv_residual_batch_verify(
            stream,
            &inp.experts,
            &inp.routing,
            &weight,
            &inp.residual,
            &mut ver_out,
            [inp.proj.as_view(), inp.k_raw.as_view(), inp.v_raw.as_view()],
            &ver_table.as_view(),
            Some(inputs.as_view_mut()),
            SEQS,
            width,
        )?;

        assert_bf16_close(
            &format!("batch_verify out width={width}"),
            &stream.clone_dtoh(&ver_out)?,
            &stream.clone_dtoh(&ref_out)?,
            0,
        );
        for (idx, (got_buf, want)) in ver_bufs.iter().zip(&init_host).enumerate() {
            assert_f32_bits_exact(
                &format!("batch_verify preserved state width={width} idx={idx}"),
                &stream.clone_dtoh(got_buf)?,
                want,
            );
        }

        let mut results_host = vec![0i32; SEQS * RESULT_LEN];
        for seq in 0..SEQS {
            results_host[seq * RESULT_LEN] = i32::try_from(width - 1)?;
        }
        let results_dev = stream.clone_htod(&results_host)?;
        spec.conv_rollback_batch(
            stream,
            &ver_table,
            1,
            SEQS,
            SEQS,
            &inputs,
            &results_dev.as_view(),
        )?;

        for (idx, (got_buf, want_buf)) in ver_bufs.iter().zip(&ref_bufs).enumerate() {
            assert_f32_bits_exact(
                &format!("batch_verify rollback width={width} idx={idx}"),
                &stream.clone_dtoh(got_buf)?,
                &stream.clone_dtoh(want_buf)?,
            );
        }
    }
    Ok(())
}
