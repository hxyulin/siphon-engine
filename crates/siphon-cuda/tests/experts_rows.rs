//! The multi-row expert MLP of `ops::experts_rows`.
//!
//! - Row invariance: each pair's activations and outputs are the same, bit for bit, for
//!   every row count from 1 to 8, whatever the other rows hold and route to, and when each
//!   row runs alone, as a one-row pass at its position would. Two calls in a row also
//!   agree, which checks that the kernel leaves its counters zero.
//! - Against the decode kernel `expert_mlp` at one row and the prefill path (grouping, then
//!   the grouped products) at m rows, which add the same exact products in other orders:
//!   the tests print how many values differ and require the outputs' relative RMS
//!   difference to stay below 2^-7, one BF16 rounding step.

#![cfg(target_os = "linux")]

mod common;

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream};
use half::bf16;
use siphon_cuda::{
    Gpu,
    ops::{
        experts::{ExpertFormat, ExpertScratch, ExpertWeights, Experts},
        experts_rows::{ExpertsRows, MAX_ROWS},
    },
};

const SLOTS: usize = 8;
const HIDDEN: usize = 2304;
const INTERMEDIATE: usize = 1152;
const GATE_SOFTCAP: f32 = 4.0;
const UP_SOFTCAP: f32 = 25.0;
/// Experts per synthetic layer: enough for rows to share some and not others.
const EXPERTS: usize = 20;

struct Setup {
    gpu: Gpu,
    rows: ExpertsRows,
    experts: Experts,
}

impl Setup {
    fn new() -> anyhow::Result<Self> {
        let gpu = Gpu::new(0)?;
        Ok(Self {
            rows: ExpertsRows::load(gpu.context())?,
            experts: Experts::load(gpu.context())?,
            gpu,
        })
    }

    fn stream(&self) -> &Arc<CudaStream> {
        self.gpu.stream()
    }
}

/// Deterministic pseudo-random numbers (SplitMix64).
struct SplitMix(u64);

impl SplitMix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn byte(&mut self) -> u8 {
        self.next().to_le_bytes()[7]
    }
}

/// Random weights with the checkpoint's scale ranges, as in `tests/experts.rs`.
struct Projection {
    weight: CudaSlice<u8>,
    scales: CudaSlice<u8>,
    global_scale: Option<CudaSlice<f32>>,
    n: usize,
    k: usize,
}

impl Projection {
    fn new(s: &Setup, fp4: bool, n: usize, k: usize, seed: u64) -> anyhow::Result<Self> {
        let mut random = SplitMix(seed);
        let stream = s.stream();
        let (weight, scales, global_scale): (Vec<u8>, Vec<u8>, _) = if fp4 {
            let weight = (0..EXPERTS * n * k / 2).map(|_| random.byte()).collect();
            // Positive E4M3 scales from 2^-3 to 2^2.
            let scales = (0..EXPERTS * n * k / 16)
                .map(|_| 32 + random.byte() % 48)
                .collect();
            let global: Vec<f32> = (0..EXPERTS)
                .map(|e| 448.0 * 6.0 * 0.004 * (1.0 + e as f32 / 16.0))
                .collect();
            (weight, scales, Some(stream.clone_htod(&global)?))
        } else {
            // Any E4M3 value but NaN.
            let weight = (0..EXPERTS * n * k)
                .map(|_| {
                    let v = random.byte();
                    if v & 0x7f == 0x7f { v ^ 1 } else { v }
                })
                .collect();
            // E8M0 scales from 2^-12 to 2^-5.
            let scales = (0..EXPERTS * n * k / 32)
                .map(|_| 115 + random.byte() % 8)
                .collect();
            (weight, scales, None)
        };
        Ok(Self {
            weight: stream.clone_htod(&weight)?,
            scales: stream.clone_htod(&scales)?,
            global_scale,
            n,
            k,
        })
    }

    fn weights(&self) -> ExpertWeights<'_> {
        ExpertWeights {
            format: match &self.global_scale {
                Some(global_scale) => ExpertFormat::Nvfp4 { global_scale },
                None => ExpertFormat::Mxfp8,
            },
            weight: &self.weight,
            scales: &self.scales,
            n: self.n,
            k: self.k,
        }
    }
}

/// 8 distinct experts per row; rows overlap, as consecutive tokens do.
fn routing(rows: usize, seed: u64) -> Vec<i32> {
    let mut random = SplitMix(seed);
    let mut ids = Vec::with_capacity(rows * SLOTS);
    for _ in 0..rows {
        let mut picked: Vec<i32> = Vec::new();
        while picked.len() < SLOTS {
            let e = i32::from(random.byte()) % i32::try_from(EXPERTS).expect("fits");
            if !picked.contains(&e) {
                picked.push(e);
            }
        }
        ids.extend(picked);
    }
    ids
}

struct Layer {
    w13: Projection,
    w2: Projection,
}

/// The multi-row MLP: activations `[m * 8, 1152]` and outputs `[m * 8, 2304]`.
fn run_rows(
    s: &Setup,
    l: &Layer,
    x: &[bf16],
    ids: &[i32],
    sync: &mut CudaSlice<u32>,
) -> anyhow::Result<(Vec<bf16>, Vec<bf16>)> {
    let stream = s.stream();
    let pairs = ids.len();
    let x = stream.clone_htod(x)?;
    let indices = stream.clone_htod(ids)?;
    let mut act = stream.alloc_zeros::<bf16>(pairs * INTERMEDIATE)?;
    let mut out = stream.alloc_zeros::<bf16>(pairs * HIDDEN)?;
    s.rows.expert_mlp_rows(
        stream,
        &x,
        &indices,
        &l.w13.weights(),
        &l.w2.weights(),
        &mut act,
        &mut out,
        sync,
        GATE_SOFTCAP,
        UP_SOFTCAP,
    )?;
    let sync_left = stream.clone_dtoh(sync)?;
    assert!(sync_left.iter().all(|&v| v == 0), "sync not reset");
    Ok((stream.clone_dtoh(&act)?, stream.clone_dtoh(&out)?))
}

fn same(a: &[bf16], b: &[bf16]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

fn layer(s: &Setup, fp4: bool) -> anyhow::Result<Layer> {
    Ok(Layer {
        w13: Projection::new(s, fp4, 2 * INTERMEDIATE, HIDDEN, 11)?,
        w2: Projection::new(s, fp4, HIDDEN, INTERMEDIATE, 13)?,
    })
}

#[test]
fn rows_are_invariant() -> anyhow::Result<()> {
    let s = Setup::new()?;
    let stream = s.stream();
    let mut sync = stream.alloc_zeros::<u32>(1 + MAX_ROWS * SLOTS)?;
    for fp4 in [true, false] {
        let name = if fp4 { "nvfp4" } else { "mxfp8" };
        let l = layer(&s, fp4)?;
        let x = common::pseudo_random_bf16(MAX_ROWS * HIDDEN, 5, 1.0, 0.0);
        let ids = routing(MAX_ROWS, 7);
        let (act8, out8) = run_rows(&s, &l, &x, &ids, &mut sync)?;
        let (again_act, again_out) = run_rows(&s, &l, &x, &ids, &mut sync)?;
        assert!(
            same(&act8, &again_act) && same(&out8, &again_out),
            "{name}: a second call differs"
        );
        let pair_act = |r: usize| r * SLOTS * INTERMEDIATE..(r + 1) * SLOTS * INTERMEDIATE;
        let pair_out = |r: usize| r * SLOTS * HIDDEN..(r + 1) * SLOTS * HIDDEN;
        for m in 1..MAX_ROWS {
            let (act, out) = run_rows(&s, &l, &x[..m * HIDDEN], &ids[..m * SLOTS], &mut sync)?;
            assert!(
                same(&act, &act8[..act.len()]) && same(&out, &out8[..out.len()]),
                "{name}: rows of m = {m} differ from m = 8"
            );
        }
        for r in 0..MAX_ROWS {
            let (act, out) = run_rows(
                &s,
                &l,
                &x[r * HIDDEN..(r + 1) * HIDDEN],
                &ids[r * SLOTS..(r + 1) * SLOTS],
                &mut sync,
            )?;
            assert!(
                same(&act, &act8[pair_act(r)]) && same(&out, &out8[pair_out(r)]),
                "{name}: row {r} alone differs from row {r} of 8"
            );
        }
        // Row 0 unchanged, the other rows with other inputs and other experts.
        let mut x_other = common::pseudo_random_bf16(MAX_ROWS * HIDDEN, 9, 30.0, 0.5);
        x_other[..HIDDEN].copy_from_slice(&x[..HIDDEN]);
        let mut ids_other = routing(MAX_ROWS, 21);
        ids_other[..SLOTS].copy_from_slice(&ids[..SLOTS]);
        let (act, out) = run_rows(&s, &l, &x_other, &ids_other, &mut sync)?;
        assert!(
            same(&act[pair_act(0)], &act8[pair_act(0)])
                && same(&out[pair_out(0)], &out8[pair_out(0)]),
            "{name}: row 0 depends on the other rows"
        );
        eprintln!("{name}: pairs bit-identical for m = 1 to 8, alone and with other rows");
    }
    Ok(())
}

/// Relative RMS difference, and the share of values that differ.
fn compare(label: &str, got: &[bf16], want: &[bf16]) -> f64 {
    let (mut diff, mut norm) = (0f64, 0f64);
    let mut differ = 0;
    let mut ulps = 0;
    for (a, b) in got.iter().zip(want) {
        let (a, b) = (f64::from(a.to_f32()), f64::from(b.to_f32()));
        diff += (a - b).powi(2);
        norm += b * b;
        if a.to_bits() != b.to_bits() {
            differ += 1;
        }
    }
    for (a, b) in got.iter().zip(want) {
        ulps = ulps.max(common::bf16_ulps(*a, *b));
    }
    let rel = (diff / norm.max(1e-30)).sqrt();
    eprintln!(
        "{label}: {differ} of {} values differ, at most {ulps} BF16 ulps, relative RMS {rel:.2e}",
        got.len()
    );
    rel
}

#[test]
fn rows_against_decode_and_prefill() -> anyhow::Result<()> {
    let setup = Setup::new()?;
    let stream = setup.stream();
    let mut sync = stream.alloc_zeros::<u32>(1 + MAX_ROWS * SLOTS)?;
    let mut decode_sync = stream.alloc_zeros::<u32>(1 + SLOTS)?;
    let mut scratch = ExpertScratch::new(stream, MAX_ROWS, EXPERTS)?;
    for fp4 in [true, false] {
        let name = if fp4 { "nvfp4" } else { "mxfp8" };
        let layer_w = layer(&setup, fp4)?;
        let input = common::pseudo_random_bf16(MAX_ROWS * HIDDEN, 15, 1.0, 0.0);
        let ids = routing(MAX_ROWS, 17);
        let (act, out) = run_rows(&setup, &layer_w, &input, &ids, &mut sync)?;

        // Each row through the decode kernel.
        let mut dec_act = Vec::new();
        let mut dec_out = Vec::new();
        for r in 0..MAX_ROWS {
            let xr = stream.clone_htod(&input[r * HIDDEN..(r + 1) * HIDDEN])?;
            let ir = stream.clone_htod(&ids[r * SLOTS..(r + 1) * SLOTS])?;
            let mut act_dec = stream.alloc_zeros::<bf16>(SLOTS * INTERMEDIATE)?;
            let mut out_dec = stream.alloc_zeros::<bf16>(SLOTS * HIDDEN)?;
            setup.experts.expert_mlp(
                stream,
                &xr,
                &ir,
                &layer_w.w13.weights(),
                &layer_w.w2.weights(),
                &mut act_dec,
                &mut out_dec,
                &mut decode_sync,
                GATE_SOFTCAP,
                UP_SOFTCAP,
            )?;
            dec_act.extend(stream.clone_dtoh(&act_dec)?);
            dec_out.extend(stream.clone_dtoh(&out_dec)?);
        }
        compare(&format!("{name} act against expert_mlp"), &act, &dec_act);
        let rel = compare(&format!("{name} out against expert_mlp"), &out, &dec_out);
        assert!(rel < 2f64.powi(-7), "{name}: too far from expert_mlp");

        // All rows through the prefill path.
        let xd = stream.clone_htod(&input)?;
        let id = stream.clone_htod(&ids)?;
        setup
            .experts
            .expert_group(stream, &id, EXPERTS, &mut scratch)?;
        let mut act_dec = stream.alloc_zeros::<bf16>(MAX_ROWS * SLOTS * INTERMEDIATE)?;
        let mut out_dec = stream.alloc_zeros::<bf16>(MAX_ROWS * SLOTS * HIDDEN)?;
        setup.experts.expert_matmul_swiglu_grouped(
            stream,
            &xd,
            &layer_w.w13.weights(),
            &mut act_dec,
            &scratch,
            GATE_SOFTCAP,
            UP_SOFTCAP,
        )?;
        setup.experts.expert_matmul_grouped(
            stream,
            &act_dec,
            &layer_w.w2.weights(),
            &mut out_dec,
            &scratch,
        )?;
        compare(
            &format!("{name} act against prefill"),
            &act,
            &stream.clone_dtoh(&act_dec)?,
        );
        let rel = compare(
            &format!("{name} out against prefill"),
            &out,
            &stream.clone_dtoh(&out_dec)?,
        );
        assert!(rel < 2f64.powi(-7), "{name}: too far from the prefill path");
    }
    Ok(())
}
