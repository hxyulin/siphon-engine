//! Routing and depth attention kernels against MLX outputs from
//! `reference/fixtures/routing.py`.
//!
//! The kernels repeat MLX's operations in MLX's order, so most results match bit for bit.
//! The exceptions, with their causes:
//! - Scores and softmax weights pass through Apple's hardware exp2, which is one ulp above
//!   the correctly rounded value for 36% of its table entries. The kernels use the correctly
//!   rounded value (see `apple_exp2` in `kernels/routing.cu`), so routing weights and depth
//!   attention outputs can be off by an ulp of the weight.
//! - MLX's GEMM for two or more router rows sums in an unknown order.

#![cfg(target_os = "linux")]

mod common;

use common::{
    Fixture, assert_bf16_close, assert_f32_within_ulps, pseudo_random, pseudo_random_bf16,
};
use cudarc::driver::CudaSlice;
use half::bf16;
use siphon_cuda::{
    Gpu,
    ops::{
        conv::Conv,
        misc::{MessageMix, Misc},
        norm::Norm,
        routing::{
            DepthRows, DepthStep, EXPERTS, FUSED_MAX_CANDIDATES, MoePruning, NormDest, Partial,
            RouterMatmul, RouterRowsScratch, Routing, RowsPartial, STEP_ROUTER_ROWS, StepPartial,
            TOP_K,
        },
    },
};

const HIDDEN: usize = 2304;
const ROUTE_SCALE: f32 = 8.0;

/// The router weight that `router_weight()` in `routing.py` generates, which is too
/// large to store in the fixture.
fn router_weight() -> Vec<f32> {
    let mut weight: Vec<f32> = (0..(EXPERTS * HIDDEN) as u32)
        .map(|i| {
            let mut h = i.wrapping_mul(0x9E37_79B1);
            h ^= h >> 15;
            h = h.wrapping_mul(0x2C1B_3C6D);
            h ^= h >> 12;
            let unit = (h >> 8) as f32 * 2f32.powi(-24);
            (unit - 0.5) * 0.06
        })
        .collect();
    for e in [3, 10, 20, 40] {
        let partner = e ^ 64;
        weight.copy_within(e * HIDDEN..(e + 1) * HIDDEN, partner * HIDDEN);
    }
    weight
}

/// Distance between two finite F32 values in units in the last place.
fn f32_ulps(a: f32, b: f32) -> u32 {
    let key = |v: f32| {
        let bits = v.to_bits().cast_signed();
        if bits < 0 { i32::MIN - bits } else { bits }
    };
    key(a).abs_diff(key(b))
}

/// Reports how many elements differ and by how much, and fails if any differs by more
/// than `max_abs`.
fn assert_f32_within(label: &str, got: &[f32], want: &[f32], max_abs: f32) {
    assert_eq!(got.len(), want.len(), "{label}: length");
    let mut differing = 0;
    let mut worst_ulps = 0;
    let mut worst_abs = 0.0f32;
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let abs = (g - w).abs();
        assert!(abs <= max_abs, "{label}[{i}]: got {g:e}, want {w:e}");
        differing += usize::from(g.to_bits() != w.to_bits());
        worst_ulps = worst_ulps.max(f32_ulps(g, w));
        worst_abs = worst_abs.max(abs);
    }
    eprintln!(
        "{label}: {differing} of {} elements differ, by at most {worst_ulps} ulp ({worst_abs:e})",
        want.len()
    );
}

/// Each row's (expert, weight) pairs sorted by expert.
fn pairs_by_expert(indices: &[i32], weights: &[f32]) -> Vec<Vec<(i32, f32)>> {
    indices
        .chunks(TOP_K)
        .zip(weights.chunks(TOP_K))
        .map(|(i, w)| {
            let mut row: Vec<_> = i.iter().copied().zip(w.iter().copied()).collect();
            row.sort_by_key(|&(e, _)| e);
            row
        })
        .collect()
}

struct Routed {
    logits: Vec<f32>,
    indices: Vec<i32>,
    weights: Vec<f32>,
}

/// Where the routing kernel's logits come from.
#[derive(Clone, Copy)]
enum Logits<'a> {
    /// Computed by the router logits kernel from BF16 hidden rows.
    Hidden(&'a [bf16], RouterMatmul),
    /// Given, as MLX computed them.
    Given(&'a [f32]),
}

fn route(gpu: &Gpu, routing: &Routing, logits: Logits, bias: &[f32]) -> anyhow::Result<Routed> {
    let stream = gpu.stream();
    let logits_dev = match logits {
        Logits::Given(logits) => stream.clone_htod(logits)?,
        Logits::Hidden(hidden, matmul) => {
            let hidden_dev = stream.clone_htod(hidden)?;
            let weight_dev = stream.clone_htod(&router_weight())?;
            let mut logits_dev = stream.alloc_zeros::<f32>(hidden.len() / HIDDEN * EXPERTS)?;
            routing.router_logits(stream, &hidden_dev, &weight_dev, &mut logits_dev, matmul)?;
            logits_dev
        }
    };
    let rows = logits_dev.len() / EXPERTS;
    let bias_dev = stream.clone_htod(bias)?;
    let mut indices_dev = stream.alloc_zeros::<i32>(rows * TOP_K)?;
    let mut weights_dev = stream.alloc_zeros::<f32>(rows * TOP_K)?;
    routing.expert_routing(
        stream,
        &logits_dev,
        &bias_dev,
        &mut indices_dev,
        &mut weights_dev,
        ROUTE_SCALE,
    )?;
    Ok(Routed {
        logits: stream.clone_dtoh(&logits_dev)?,
        indices: stream.clone_dtoh(&indices_dev)?,
        weights: stream.clone_dtoh(&weights_dev)?,
    })
}

#[test]
fn router_weight_generator_matches_fixture() -> anyhow::Result<()> {
    let fixture = Fixture::load("routing")?;
    let (want, _) = fixture.f32("router.weight_rows")?;
    let got: Vec<f32> = router_weight()
        .chunks(HIDDEN)
        .step_by(64)
        .flatten()
        .copied()
        .collect();
    assert_f32_within("router.weight_rows", &got, &want, 0.0);
    Ok(())
}

/// The routing kernel alone, fed MLX's logits. The experts must match MLX's in the same
/// order, including exact ties at the 8th place; the weights may be off by the exp2 ulp.
#[test]
fn expert_routing_matches_mlx() -> anyhow::Result<()> {
    let fixture = Fixture::load("routing")?;
    let (bias, _) = fixture.f32("router.bias")?;
    let gpu = Gpu::new(0)?;
    let routing = Routing::load(gpu.context())?;
    for case in ["ties", "router16", "router1"] {
        let (logits, _) = fixture.f32(&format!("{case}.logits"))?;
        let got = route(&gpu, &routing, Logits::Given(&logits), &bias)?;
        let (indices, _) = fixture.i32(&format!("{case}.indices"))?;
        let (weights, _) = fixture.f32(&format!("{case}.weights"))?;
        assert_eq!(got.indices, indices, "{case}: indices");
        assert_f32_within(&format!("{case}: weights"), &got.weights, &weights, 5e-7);
    }
    Ok(())
}

/// The full router from BF16 hidden states, with the matmul precision MLX used for that
/// row count. The experts must be the same set; the order may differ only among experts
/// whose biased scores differ by the logits' rounding.
#[test]
fn top_k_router_matches_mlx() -> anyhow::Result<()> {
    let fixture = Fixture::load("routing")?;
    let (bias, _) = fixture.f32("router.bias")?;
    let gpu = Gpu::new(0)?;
    let routing = Routing::load(gpu.context())?;
    for (case, matmul, logits_abs) in [
        ("router16", RouterMatmul::Tf32, 1e-5),
        ("router1", RouterMatmul::F32, 0.0),
    ] {
        let (hidden, _) = fixture.bf16(&format!("{case}.hidden"))?;
        let got = route(&gpu, &routing, Logits::Hidden(&hidden, matmul), &bias)?;
        let (logits, _) = fixture.f32(&format!("{case}.logits"))?;
        assert_f32_within(&format!("{case}: logits"), &got.logits, &logits, logits_abs);

        let (indices, _) = fixture.i32(&format!("{case}.indices"))?;
        let (weights, _) = fixture.f32(&format!("{case}.weights"))?;
        let got_pairs = pairs_by_expert(&got.indices, &got.weights);
        let want_pairs = pairs_by_expert(&indices, &weights);
        for (row, (g, w)) in got_pairs.iter().zip(&want_pairs).enumerate() {
            let g_experts: Vec<i32> = g.iter().map(|p| p.0).collect();
            let w_experts: Vec<i32> = w.iter().map(|p| p.0).collect();
            assert_eq!(g_experts, w_experts, "{case}: experts of row {row}");
        }
        let in_order = got
            .indices
            .chunks(TOP_K)
            .zip(indices.chunks(TOP_K))
            .filter(|(g, w)| g == w)
            .count();
        eprintln!(
            "{case}: {in_order} of {} rows pick the same experts in the same order",
            got_pairs.len()
        );
        let got_weights: Vec<f32> = got_pairs.iter().flatten().map(|p| p.1).collect();
        let want_weights: Vec<f32> = want_pairs.iter().flatten().map(|p| p.1).collect();
        assert_f32_within(
            &format!("{case}: weights"),
            &got_weights,
            &want_weights,
            2e-6,
        );
    }
    Ok(())
}

/// BlockAttentionResidual with one and three queries, MessageInput with four, and a
/// query scaled so one candidate takes almost all the weight.
#[test]
fn depth_attention_matches_mlx() -> anyhow::Result<()> {
    let fixture = Fixture::load("routing")?;
    let gpu = Gpu::new(0)?;
    let stream = gpu.stream();
    let routing = Routing::load(gpu.context())?;
    for (case, inputs) in [
        ("depth_a", "depth_a"),
        ("depth_b", "depth_b"),
        ("depth_c", "depth_c"),
        ("depth_d", "depth_c"),
    ] {
        let (blocks, _) = fixture.f32(&format!("{inputs}.blocks"))?;
        let (partial, _) = fixture.f32(&format!("{inputs}.partial"))?;
        let (queries, _) = fixture.f32(&format!("{case}.queries"))?;
        let (want, _) = fixture.f32(&format!("{case}.out"))?;
        let blocks_dev = stream.clone_htod(&blocks)?;
        let partial_dev = stream.clone_htod(&partial)?;
        let queries_dev = stream.clone_htod(&queries)?;
        let mut out_dev = stream.alloc_zeros::<f32>(want.len())?;
        routing.depth_attention(
            stream,
            &blocks_dev,
            &partial_dev,
            &queries_dev,
            &mut out_dev,
        )?;
        let got = stream.clone_dtoh(&out_dev)?;
        assert_f32_within(case, &got, &want, 1e-6);
    }
    Ok(())
}

const EPS: f32 = 1e-6;

fn norm_dest<'a>(
    out: &'a mut CudaSlice<bf16>,
    weight: Option<&'a CudaSlice<bf16>>,
) -> NormDest<'a> {
    NormDest {
        out: out.as_view_mut(),
        weight,
    }
}

/// The fused single-row depth attention against the separate kernels on the fixture cases,
/// row 0 of each, with at most the 5 block states the kernel holds:
/// - with the fixture's partial as given, its output against MLX's within the old kernel's
///   1e-6;
/// - with a partial that needs normalizing (F32, and BF16 for the first case), against
///   `unweighted_rms_norm` then `depth_attention`, then `rms_norm_f32` (weighted copies) or
///   `unweighted_rms_norm` and a cast (unweighted copies) of each output row.
///
/// The sums are the separate kernels'. The softmax uses CUDA's exp2f where the old kernel
/// emulates Apple's exp2; both are within an ulp of exp2, so outputs move by a few F32 ulps
/// of the largest candidate, under 1e-6 here. The norms compute 1 / sqrt in float where the
/// norm kernels round a double rsqrt once: at most one F32 ulp, so the normalized partial
/// is within one ulp and the BF16 copies within one BF16 ulp.
#[test]
#[expect(clippy::too_many_lines, reason = "three comparisons per fixture case")]
fn depth_attention_fused_matches_separate_kernels() -> anyhow::Result<()> {
    let fixture = Fixture::load("routing")?;
    let gpu = Gpu::new(0)?;
    let stream = gpu.stream();
    let routing = Routing::load(gpu.context())?;
    let norm = Norm::load(gpu.context())?;
    let misc = Misc::load(gpu.context())?;
    let weights: Vec<CudaSlice<bf16>> = (0..4)
        .map(|k| Ok(stream.clone_htod(&pseudo_random_bf16(HIDDEN, 200 + k, 0.5, 1.0))?))
        .collect::<anyhow::Result<_>>()?;
    for (case, inputs) in [
        ("depth_a", "depth_a"),
        ("depth_b", "depth_b"),
        ("depth_c", "depth_c"),
        ("depth_d", "depth_c"),
    ] {
        let (blocks, _) = fixture.f32(&format!("{inputs}.blocks"))?;
        let (partial, _) = fixture.f32(&format!("{inputs}.partial"))?;
        let (queries, _) = fixture.f32(&format!("{case}.queries"))?;
        let (mlx, _) = fixture.f32(&format!("{case}.out"))?;
        let rows = partial.len() / HIDDEN;
        let all_blocks = blocks.len() / partial.len();
        // Cases c and d have more block states than the model's 4; they use the first ones
        // against the separate kernels only.
        let nb = all_blocks.min(FUSED_MAX_CANDIDATES - 1);
        let nq = queries.len() / HIDDEN;
        let row_blocks: Vec<f32> = (0..nb)
            .flat_map(|i| {
                blocks[i * rows * HIDDEN..(i * rows + 1) * HIDDEN]
                    .iter()
                    .copied()
            })
            .collect();
        let blocks_dev = stream.clone_htod(&row_blocks)?;
        let queries_dev = stream.clone_htod(&queries)?;

        // The fixture's partial as given, against MLX.
        if nb == all_blocks {
            let partial_dev = stream.clone_htod(&partial[..HIDDEN])?;
            let mut out = stream.alloc_zeros::<f32>(nq * HIDDEN)?;
            routing.depth_attention_fused(
                stream,
                &blocks_dev,
                Partial::Normalized(partial_dev.as_view()),
                None,
                &queries_dev,
                Some(out.as_view_mut()),
                [None, None, None, None],
                EPS,
            )?;
            // MLX's out is [rows, nq, hidden]; row 0 is its first nq * hidden values.
            assert_f32_within(case, &stream.clone_dtoh(&out)?, &mlx[..nq * HIDDEN], 1e-6);
        }

        // A partial that needs normalizing.
        let raw: Vec<f32> = partial[..HIDDEN]
            .iter()
            .zip(pseudo_random(HIDDEN, 300, 0.5))
            .map(|(p, r)| 3.0 * p + r)
            .collect();
        let raw_bf16: Vec<bf16> = raw.iter().map(|&v| bf16::from_f32(v)).collect();
        let as_bf16 = case == "depth_a";
        let wide: Vec<f32> = if as_bf16 {
            raw_bf16.iter().map(|v| v.to_f32()).collect()
        } else {
            raw.clone()
        };
        let wide_dev = stream.clone_htod(&wide)?;
        let mut normed_want = stream.alloc_zeros::<f32>(HIDDEN)?;
        norm.unweighted_rms_norm(stream, &wide_dev, &mut normed_want, EPS)?;
        let mut out_want = stream.alloc_zeros::<f32>(nq * HIDDEN)?;
        routing.depth_attention(
            stream,
            &blocks_dev,
            &normed_want,
            &queries_dev,
            &mut out_want,
        )?;
        // Destination k takes row min(k, nq - 1); the last one is unweighted.
        let weighted = |k: usize| k < 3;
        let mut dest_want = Vec::new();
        for (k, weight) in weights.iter().enumerate() {
            let row = out_want.slice(k.min(nq - 1) * HIDDEN..(k.min(nq - 1) + 1) * HIDDEN);
            let mut bf = stream.alloc_zeros::<bf16>(HIDDEN)?;
            if weighted(k) {
                norm.rms_norm_f32(stream, &row, weight, &mut bf, EPS)?;
            } else {
                let mut f = stream.alloc_zeros::<f32>(HIDDEN)?;
                norm.unweighted_rms_norm(stream, &row, &mut f, EPS)?;
                misc.cast_f32_to_bf16(stream, &f, &mut bf)?;
            }
            dest_want.push(stream.clone_dtoh(&bf)?);
        }

        let raw_dev = stream.clone_htod(&raw)?;
        let raw_bf16_dev = stream.clone_htod(&raw_bf16)?;
        let mut normed = stream.alloc_zeros::<f32>(HIDDEN)?;
        let mut out = stream.alloc_zeros::<f32>(nq * HIDDEN)?;
        let mut dests: Vec<CudaSlice<bf16>> = (0..4)
            .map(|_| Ok(stream.alloc_zeros::<bf16>(HIDDEN)?))
            .collect::<anyhow::Result<_>>()?;
        let [d0, d1, d2, d3] = &mut dests[..] else {
            unreachable!()
        };
        let weight = |k: usize| weighted(k).then(|| &weights[k]);
        routing.depth_attention_fused(
            stream,
            &blocks_dev,
            if as_bf16 {
                Partial::Bf16(raw_bf16_dev.as_view())
            } else {
                Partial::F32(raw_dev.as_view())
            },
            Some(normed.as_view_mut()),
            &queries_dev,
            Some(out.as_view_mut()),
            [
                Some(norm_dest(d0, weight(0))),
                Some(norm_dest(d1, weight(1))),
                Some(norm_dest(d2, weight(2))),
                Some(norm_dest(d3, weight(3))),
            ],
            EPS,
        )?;
        assert_f32_within_ulps(
            &format!("{case}: normed partial"),
            &stream.clone_dtoh(&normed)?,
            &stream.clone_dtoh(&normed_want)?,
            1,
        );
        assert_f32_within(
            &format!("{case}: out"),
            &stream.clone_dtoh(&out)?,
            &stream.clone_dtoh(&out_want)?,
            1e-6,
        );
        for (k, want) in dest_want.iter().enumerate() {
            assert_bf16_close(
                &format!("{case}: destination {k}"),
                &stream.clone_dtoh(&dests[k])?,
                want,
                1,
            );
        }
    }
    Ok(())
}

/// The fused MoE norm and router against `rms_norm_f32`, `router_logits` (F32) and
/// `expert_routing` for several rows, run one at a time.
///
/// The norm and logits sum in the separate kernels' order. The norm computes 1 / sqrt in
/// float rather than a rounded double rsqrt, at most one F32 ulp, so the BF16 input may
/// differ by one ulp and the logits slightly. The scores use CUDA's expf where
/// `expert_routing` emulates Apple's exp, so weights move by a few ulps, within the 2e-6
/// the router test above allows. The experts must be the same set.
#[test]
fn moe_router_matches_separate_kernels() -> anyhow::Result<()> {
    let fixture = Fixture::load("routing")?;
    let (bias, _) = fixture.f32("router.bias")?;
    let gpu = Gpu::new(0)?;
    let stream = gpu.stream();
    let routing = Routing::load(gpu.context())?;
    let norm = Norm::load(gpu.context())?;
    let router = stream.clone_htod(&router_weight())?;
    let bias_dev = stream.clone_htod(&bias)?;
    let weight = stream.clone_htod(&pseudo_random_bf16(HIDDEN, 400, 0.5, 1.0))?;
    let mut counter = stream.alloc_zeros::<u32>(1)?;
    for row in 0..6u32 {
        let x = stream.clone_htod(&pseudo_random(HIDDEN, 500 + row, 4.0))?;
        let mut x_want = stream.alloc_zeros::<bf16>(HIDDEN)?;
        norm.rms_norm_f32(stream, &x, &weight, &mut x_want, EPS)?;
        let mut logits_want = stream.alloc_zeros::<f32>(EXPERTS)?;
        routing.router_logits(
            stream,
            &x_want,
            &router,
            &mut logits_want,
            RouterMatmul::F32,
        )?;
        let mut indices_want = stream.alloc_zeros::<i32>(TOP_K)?;
        let mut weights_want = stream.alloc_zeros::<f32>(TOP_K)?;
        routing.expert_routing(
            stream,
            &logits_want,
            &bias_dev,
            &mut indices_want,
            &mut weights_want,
            ROUTE_SCALE,
        )?;

        let mut x_out = stream.alloc_zeros::<bf16>(HIDDEN)?;
        let mut logits = stream.alloc_zeros::<f32>(EXPERTS)?;
        let mut indices = stream.alloc_zeros::<i32>(TOP_K)?;
        let mut weights = stream.alloc_zeros::<f32>(TOP_K)?;
        routing.moe_router(
            stream,
            &x,
            &weight,
            EPS,
            &router,
            &bias_dev,
            &mut x_out,
            &mut logits,
            &mut indices,
            &mut weights,
            &mut counter,
            ROUTE_SCALE,
        )?;
        let label = format!("row {row}");
        assert_bf16_close(
            &format!("{label}: input"),
            &stream.clone_dtoh(&x_out)?,
            &stream.clone_dtoh(&x_want)?,
            1,
        );
        assert_f32_within(
            &format!("{label}: logits"),
            &stream.clone_dtoh(&logits)?,
            &stream.clone_dtoh(&logits_want)?,
            1e-5,
        );
        let got = pairs_by_expert(&stream.clone_dtoh(&indices)?, &stream.clone_dtoh(&weights)?);
        let want = pairs_by_expert(
            &stream.clone_dtoh(&indices_want)?,
            &stream.clone_dtoh(&weights_want)?,
        );
        let experts = |p: &[Vec<(i32, f32)>]| -> Vec<i32> { p[0].iter().map(|e| e.0).collect() };
        assert_eq!(experts(&got), experts(&want), "{label}: experts");
        let weights_of = |p: &[Vec<(i32, f32)>]| -> Vec<f32> { p[0].iter().map(|e| e.1).collect() };
        assert_f32_within(
            &format!("{label}: weights"),
            &weights_of(&got),
            &weights_of(&want),
            2e-6,
        );
        assert_eq!(
            stream.clone_dtoh(&counter)?,
            [0],
            "the kernel resets its counter"
        );
    }
    Ok(())
}

/// The chunked router against single-row calls, bit for bit: the norm, the logits (a tiled
/// product that keeps the single-row sums), the experts and their weights. 130 rows cover
/// two full row tiles of 64 and a partial one.
/// The chunked router against `moe_router` run one row at a time, from the normalized rows
/// that `moe_router` writes: logits within 1e-5 (the tiled product sums in another order),
/// the same experts, and weights within 2e-6, as for the decode router above. 130 rows
/// cover two full row tiles of 64 and a partial one.
#[test]
fn moe_router_rows_matches_single_row() -> anyhow::Result<()> {
    const ROWS: usize = 130;
    let fixture = Fixture::load("routing")?;
    let (bias, _) = fixture.f32("router.bias")?;
    let gpu = Gpu::new(0)?;
    let stream = gpu.stream();
    let routing = Routing::load(gpu.context())?;
    let router = stream.clone_htod(&router_weight())?;
    let bias_dev = stream.clone_htod(&bias)?;
    let weight = stream.clone_htod(&pseudo_random_bf16(HIDDEN, 400, 0.5, 1.0))?;
    let x_host = pseudo_random(ROWS * HIDDEN, 700, 4.0);

    let mut counter = stream.alloc_zeros::<u32>(1)?;
    let mut x_norm = Vec::new();
    let mut want = Vec::new();
    for row in 0..ROWS {
        let x_row = stream.clone_htod(&x_host[row * HIDDEN..(row + 1) * HIDDEN])?;
        let mut x_want = stream.alloc_zeros::<bf16>(HIDDEN)?;
        let mut logits_want = stream.alloc_zeros::<f32>(EXPERTS)?;
        let mut indices_want = stream.alloc_zeros::<i32>(TOP_K)?;
        let mut weights_want = stream.alloc_zeros::<f32>(TOP_K)?;
        routing.moe_router(
            stream,
            &x_row,
            &weight,
            EPS,
            &router,
            &bias_dev,
            &mut x_want,
            &mut logits_want,
            &mut indices_want,
            &mut weights_want,
            &mut counter,
            ROUTE_SCALE,
        )?;
        x_norm.extend(stream.clone_dtoh(&x_want)?);
        want.push((
            stream.clone_dtoh(&logits_want)?,
            stream.clone_dtoh(&indices_want)?,
            stream.clone_dtoh(&weights_want)?,
        ));
    }

    let x = stream.clone_htod(&x_norm)?;
    let mut logits = stream.alloc_zeros::<f32>(ROWS * EXPERTS)?;
    let mut indices = stream.alloc_zeros::<i32>(ROWS * TOP_K)?;
    let mut weights = stream.alloc_zeros::<f32>(ROWS * TOP_K)?;
    // More rows than the call uses, as in a short last chunk.
    let mut scratch = RouterRowsScratch::new(stream, ROWS + 7)?;
    routing.moe_router_rows(
        stream,
        &x,
        &router,
        &bias_dev,
        &mut logits,
        &mut indices,
        &mut weights,
        &mut scratch,
        ROUTE_SCALE,
        ROWS,
    )?;
    let logits = stream.clone_dtoh(&logits)?;
    let indices = stream.clone_dtoh(&indices)?;
    let weights = stream.clone_dtoh(&weights)?;
    for (row, (logits_want, indices_want, weights_want)) in want.iter().enumerate() {
        let label = format!("row {row}");
        assert_f32_within(
            &format!("{label}: logits"),
            &logits[row * EXPERTS..(row + 1) * EXPERTS],
            logits_want,
            1e-5,
        );
        let range = row * TOP_K..(row + 1) * TOP_K;
        let got = pairs_by_expert(&indices[range.clone()], &weights[range]);
        let want = pairs_by_expert(indices_want, weights_want);
        let experts = |p: &[Vec<(i32, f32)>]| -> Vec<i32> { p[0].iter().map(|e| e.0).collect() };
        assert_eq!(experts(&got), experts(&want), "{label}: experts");
        let weights_of = |p: &[Vec<(i32, f32)>]| -> Vec<f32> { p[0].iter().map(|e| e.1).collect() };
        assert_f32_within(
            &format!("{label}: weights"),
            &weights_of(&got),
            &weights_of(&want),
            2e-6,
        );
    }
    Ok(())
}

/// The fused kernel's results for one query set: out, normed partial and four destinations.
struct DepthResults {
    out: Vec<f32>,
    normed: Vec<f32>,
    dests: Vec<Vec<bf16>>,
}

/// Runs `depth_attention_fused` with four destinations (the last unweighted).
fn fused_depth(
    gpu: &Gpu,
    routing: &Routing,
    blocks: &CudaSlice<f32>,
    partial: Partial<'_>,
    queries: &CudaSlice<f32>,
    weights: &[CudaSlice<bf16>],
) -> anyhow::Result<DepthResults> {
    let stream = gpu.stream();
    let nq = queries.len() / HIDDEN;
    let mut normed = stream.alloc_zeros::<f32>(HIDDEN)?;
    let mut out = stream.alloc_zeros::<f32>(nq * HIDDEN)?;
    let mut dests: Vec<CudaSlice<bf16>> = (0..4)
        .map(|_| Ok(stream.alloc_zeros::<bf16>(HIDDEN)?))
        .collect::<anyhow::Result<_>>()?;
    let [d0, d1, d2, d3] = &mut dests[..] else {
        unreachable!()
    };
    routing.depth_attention_fused(
        stream,
        blocks,
        partial,
        Some(normed.as_view_mut()),
        queries,
        Some(out.as_view_mut()),
        [
            Some(norm_dest(d0, Some(&weights[0]))),
            Some(norm_dest(d1, Some(&weights[1]))),
            Some(norm_dest(d2, Some(&weights[2]))),
            Some(norm_dest(d3, None)),
        ],
        EPS,
    )?;
    Ok(DepthResults {
        out: stream.clone_dtoh(&out)?,
        normed: stream.clone_dtoh(&normed)?,
        dests: dests
            .iter()
            .map(|d| Ok(stream.clone_dtoh(d)?))
            .collect::<anyhow::Result<_>>()?,
    })
}

/// Depth attention in F64 on the host: the `nb` rows of `blocks` and the partial
/// (normalized first if `normalize`) attend for each of the `nq` queries.
fn depth_reference(blocks: &[f32], partial: &[f32], normalize: bool, queries: &[f32]) -> Vec<f64> {
    let wide = |v: &[f32]| -> Vec<f64> { v.iter().map(|&x| f64::from(x)).collect() };
    let mut candidates: Vec<Vec<f64>> = blocks.chunks(HIDDEN).map(wide).collect();
    let mut p = wide(partial);
    if normalize {
        let mean = p.iter().map(|x| x * x).sum::<f64>() / HIDDEN as f64;
        let inv = 1.0 / (mean + f64::from(EPS)).sqrt();
        for x in &mut p {
            *x *= inv;
        }
    }
    candidates.push(p);
    let scale = 1.0 / (HIDDEN as f64).sqrt();
    let mut out = Vec::with_capacity(queries.len());
    for q in queries.chunks(HIDDEN) {
        let logits: Vec<f64> = candidates
            .iter()
            .map(|c| c.iter().zip(q).map(|(a, &b)| a * f64::from(b)).sum::<f64>() * scale)
            .collect();
        let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let e: Vec<f64> = logits.iter().map(|l| (l - max).exp()).collect();
        let z: f64 = e.iter().sum();
        out.extend((0..HIDDEN).map(|d| {
            candidates
                .iter()
                .zip(&e)
                .map(|(c, w)| c[d] * w / z)
                .sum::<f64>()
        }));
    }
    out
}

fn max_error(got: &[f32], exact: &[f64]) -> f64 {
    got.iter()
        .zip(exact)
        .map(|(&g, &x)| (f64::from(g) - x).abs())
        .fold(0.0, f64::max)
}

/// Checks the decode kernel's results (`got`) against the fused kernel's (`want`) and
/// the F64 reference `exact`. The output's largest error from `exact` may be at most
/// twice the fused kernel's, or `floor` times the largest output, whichever is larger. The
/// normalized partial may differ by `normed_ulps`. A destination element may differ from the fused kernel's by one BF16 ulp, or by the
/// output difference carried through the norm (twice the output bound over the row's
/// RMS, times 1.5, the largest norm weight): elements near zero can move by more ulps.
fn assert_depth_close(
    label: &str,
    got: &DepthResults,
    want: &DepthResults,
    exact: &[f64],
    (floor, normed_ulps): (f64, u64),
) {
    let magnitude = exact.iter().fold(0.0f64, |m, x| m.max(x.abs()));
    let fused_error = max_error(&want.out, exact);
    let step_error = max_error(&got.out, exact);
    let bound = (2.0 * fused_error).max(magnitude * floor);
    eprintln!(
        "{label}: out error {step_error:.2e}, fused kernel {fused_error:.2e}, largest output \
         {magnitude:.2e}"
    );
    assert!(
        step_error <= bound,
        "{label}: out error {step_error:e} exceeds {bound:e}"
    );
    assert_f32_within_ulps(
        &format!("{label}: normed partial"),
        &got.normed,
        &want.normed,
        normed_ulps,
    );
    let nq = want.out.len() / HIDDEN;
    for (k, (g, w)) in got.dests.iter().zip(&want.dests).enumerate() {
        let row = &want.out[k.min(nq - 1) * HIDDEN..(k.min(nq - 1) + 1) * HIDDEN];
        let rms = (row.iter().map(|v| v * v).sum::<f32>() / HIDDEN as f32).sqrt();
        let atol = 2.0 * (bound + fused_error) as f32 / rms * 1.5;
        let mut differing = 0;
        for (i, (g, w)) in g.iter().zip(w).enumerate() {
            let (g, w) = (g.to_f32(), w.to_f32());
            let ulp = if w == 0.0 {
                0.0
            } else {
                2f32.powi(w.abs().log2().floor() as i32 - 7)
            };
            assert!(
                (g - w).abs() <= ulp.max(atol),
                "{label}: destination {k}[{i}]: got {g:e}, want {w:e}"
            );
            differing += usize::from(g.to_bits() != w.to_bits());
        }
        eprintln!("{label}: destination {k}: {differing} of {HIDDEN} elements differ");
    }
}

/// The decode depth attention against the fused kernel on the fixture cases with up to 5
/// block states, for every kind of partial, with a batch of three query sets (the
/// fixture's and two others) in one launch, each against its own fused call. Outputs agree
/// within `STEP_REL_TOLERANCE` of the largest candidate element, the normalized partial
/// within 2 ulps, and the BF16 destinations within one ulp.
#[test]
#[expect(clippy::too_many_lines, reason = "three partials, three query sets")]
fn depth_attention_step_matches_fused() -> anyhow::Result<()> {
    const BATCH: usize = 3;
    let fixture = Fixture::load("routing")?;
    let gpu = Gpu::new(0)?;
    let stream = gpu.stream();
    let routing = Routing::load(gpu.context())?;
    let weights: Vec<CudaSlice<bf16>> = (0..3)
        .map(|k| Ok(stream.clone_htod(&pseudo_random_bf16(HIDDEN, 200 + k, 0.5, 1.0))?))
        .collect::<anyhow::Result<_>>()?;
    for (case, inputs) in [
        ("depth_a", "depth_a"),
        ("depth_b", "depth_b"),
        ("depth_c", "depth_c"),
        ("depth_d", "depth_c"),
    ] {
        let (blocks, _) = fixture.f32(&format!("{inputs}.blocks"))?;
        let (partial, _) = fixture.f32(&format!("{inputs}.partial"))?;
        let (queries, _) = fixture.f32(&format!("{case}.queries"))?;
        let rows = partial.len() / HIDDEN;
        let nb = (blocks.len() / partial.len()).min(FUSED_MAX_CANDIDATES - 1);
        let nq = queries.len() / HIDDEN;
        let row_blocks: Vec<f32> = (0..nb)
            .flat_map(|i| {
                blocks[i * rows * HIDDEN..(i * rows + 1) * HIDDEN]
                    .iter()
                    .copied()
            })
            .collect();
        let blocks_dev = stream.clone_htod(&row_blocks)?;
        // Query set 0 is the fixture's; the others perturb it.
        let query_sets: Vec<Vec<f32>> = (0..BATCH)
            .map(|b| {
                queries
                    .iter()
                    .zip(pseudo_random(queries.len(), 600 + b as u32, 0.2))
                    .map(|(q, r)| if b == 0 { *q } else { q + r })
                    .collect()
            })
            .collect();
        let all_queries = stream.clone_htod(&query_sets.concat())?;

        let raw: Vec<f32> = partial[..HIDDEN]
            .iter()
            .zip(pseudo_random(HIDDEN, 300, 0.5))
            .map(|(p, r)| 3.0 * p + r)
            .collect();
        let raw_bf16: Vec<bf16> = raw.iter().map(|&v| bf16::from_f32(v)).collect();
        let normalized = stream.clone_htod(&partial[..HIDDEN])?;
        let raw_dev = stream.clone_htod(&raw)?;
        let raw_bf16_dev = stream.clone_htod(&raw_bf16)?;
        let raw_wide: Vec<f32> = raw_bf16.iter().map(|v| v.to_f32()).collect();
        let partial_host = partial[..HIDDEN].to_vec();

        for kind in ["normalized", "f32", "bf16"] {
            let partial = || match kind {
                "normalized" => Partial::Normalized(normalized.as_view()),
                "f32" => Partial::F32(raw_dev.as_view()),
                _ => Partial::Bf16(raw_bf16_dev.as_view()),
            };
            let step_partial = || match kind {
                "normalized" => StepPartial::Normalized(normalized.as_view()),
                "f32" => StepPartial::F32(raw_dev.as_view()),
                _ => StepPartial::Bf16(raw_bf16_dev.as_view()),
            };
            let mut normed = stream.alloc_zeros::<f32>(HIDDEN)?;
            let mut out = stream.alloc_zeros::<f32>(BATCH * nq * HIDDEN)?;
            // Each destination holds the batch's rows HIDDEN apart.
            let mut dests: Vec<CudaSlice<bf16>> = (0..4)
                .map(|_| Ok(stream.alloc_zeros::<bf16>(BATCH * HIDDEN)?))
                .collect::<anyhow::Result<_>>()?;
            let [d0, d1, d2, d3] = &mut dests[..] else {
                unreachable!()
            };
            routing.depth_attention_step(
                stream,
                DepthStep {
                    blocks: blocks_dev.as_view(),
                    partial: step_partial(),
                    normed_partial: Some(normed.as_view_mut()),
                    queries: all_queries.as_view(),
                    batch: BATCH,
                    out: Some(out.as_view_mut()),
                    dests: [
                        Some(norm_dest(d0, Some(&weights[0]))),
                        Some(norm_dest(d1, Some(&weights[1]))),
                        Some(norm_dest(d2, Some(&weights[2]))),
                        Some(norm_dest(d3, None)),
                    ],
                    dst_stride: HIDDEN,
                },
                EPS,
            )?;
            let out = stream.clone_dtoh(&out)?;
            let normed = stream.clone_dtoh(&normed)?;
            let dests: Vec<Vec<bf16>> = dests
                .iter()
                .map(|d| Ok(stream.clone_dtoh(d)?))
                .collect::<anyhow::Result<_>>()?;
            for (b, set) in query_sets.iter().enumerate() {
                let want = fused_depth(
                    &gpu,
                    &routing,
                    &blocks_dev,
                    partial(),
                    &stream.clone_htod(set)?,
                    &weights,
                )?;
                let got = DepthResults {
                    out: out[b * nq * HIDDEN..(b + 1) * nq * HIDDEN].to_vec(),
                    normed: normed.clone(),
                    dests: dests
                        .iter()
                        .map(|d| d[b * HIDDEN..(b + 1) * HIDDEN].to_vec())
                        .collect(),
                };
                let exact = match kind {
                    "normalized" => depth_reference(&row_blocks, &partial_host, false, set),
                    "f32" => depth_reference(&row_blocks, &raw, true, set),
                    _ => depth_reference(&row_blocks, &raw_wide, true, set),
                };
                assert_depth_close(
                    &format!("{case} {kind} set {b}"),
                    &got,
                    &want,
                    &exact,
                    (2f64.powi(-22), 2),
                );
            }
        }
    }
    Ok(())
}

/// The decode depth attention with the post-attention conv and residual add computed in
/// the kernel, against `short_conv_residual` and then the fused kernel with that residual,
/// over three decode steps, with and without the hidden state. The residual and conv
/// state match bit for bit; the attention results as in the test above.
#[test]
#[expect(clippy::too_many_lines, reason = "two references per step")]
fn depth_attention_step_conv_matches_separate_kernels() -> anyhow::Result<()> {
    let fixture = Fixture::load("routing")?;
    let gpu = Gpu::new(0)?;
    let stream = gpu.stream();
    let routing = Routing::load(gpu.context())?;
    let conv = Conv::load(gpu.context())?;
    let weights: Vec<CudaSlice<bf16>> = (0..3)
        .map(|k| Ok(stream.clone_htod(&pseudo_random_bf16(HIDDEN, 200 + k, 0.5, 1.0))?))
        .collect::<anyhow::Result<_>>()?;
    let (blocks, _) = fixture.f32("depth_b.blocks")?;
    let (partial, _) = fixture.f32("depth_b.partial")?;
    let rows = partial.len() / HIDDEN;
    let nb = (blocks.len() / partial.len()).min(FUSED_MAX_CANDIDATES - 1);
    let row_blocks: Vec<f32> = (0..nb)
        .flat_map(|i| {
            blocks[i * rows * HIDDEN..(i * rows + 1) * HIDDEN]
                .iter()
                .copied()
        })
        .collect();
    let blocks_dev = stream.clone_htod(&row_blocks)?;
    let query = stream.clone_htod(&pseudo_random(HIDDEN, 700, 1.0))?;
    let conv_weight = stream.clone_htod(&pseudo_random_bf16(HIDDEN * 4, 1, 0.5, 0.0))?;
    let start = pseudo_random(3 * HIDDEN, 2, 2.0);
    let mut state_step = stream.clone_htod(&start)?;
    let mut state_want = stream.clone_htod(&start)?;
    for (step, with_hidden) in [(0u32, true), (1, false), (2, true)] {
        let x = stream.clone_htod(&pseudo_random_bf16(HIDDEN, 10 + step, 3.0, 0.0))?;
        let hidden = stream.clone_htod(&pseudo_random_bf16(HIDDEN, 20 + step, 3.0, 0.0))?;
        let mut residual_want = stream.alloc_zeros::<f32>(HIDDEN)?;
        conv.short_conv_residual(
            stream,
            &x,
            &conv_weight,
            &mut state_want,
            with_hidden.then_some(&hidden),
            &mut residual_want,
        )?;
        let want = fused_depth(
            &gpu,
            &routing,
            &blocks_dev,
            Partial::F32(residual_want.as_view()),
            &query,
            &weights,
        )?;

        let mut residual = stream.alloc_zeros::<f32>(HIDDEN)?;
        let mut normed = stream.alloc_zeros::<f32>(HIDDEN)?;
        let mut out = stream.alloc_zeros::<f32>(HIDDEN)?;
        let mut dests: Vec<CudaSlice<bf16>> = (0..4)
            .map(|_| Ok(stream.alloc_zeros::<bf16>(HIDDEN)?))
            .collect::<anyhow::Result<_>>()?;
        let [d0, d1, d2, d3] = &mut dests[..] else {
            unreachable!()
        };
        routing.depth_attention_step(
            stream,
            DepthStep {
                blocks: blocks_dev.as_view(),
                partial: StepPartial::ConvResidual {
                    x: x.as_view(),
                    weight: &conv_weight,
                    state: state_step.as_view_mut(),
                    hidden: with_hidden.then(|| hidden.as_view()),
                    residual: residual.as_view_mut(),
                },
                normed_partial: Some(normed.as_view_mut()),
                queries: query.as_view(),
                batch: 1,
                out: Some(out.as_view_mut()),
                dests: [
                    Some(norm_dest(d0, Some(&weights[0]))),
                    Some(norm_dest(d1, Some(&weights[1]))),
                    Some(norm_dest(d2, Some(&weights[2]))),
                    Some(norm_dest(d3, None)),
                ],
                dst_stride: 0,
            },
            EPS,
        )?;
        let label = format!("step {step}");
        let residual_want = stream.clone_dtoh(&residual_want)?;
        assert_f32_within_ulps(
            &format!("{label}: residual"),
            &stream.clone_dtoh(&residual)?,
            &residual_want,
            0,
        );
        assert_f32_within_ulps(
            &format!("{label}: state"),
            &stream.clone_dtoh(&state_step)?,
            &stream.clone_dtoh(&state_want)?,
            0,
        );
        let got = DepthResults {
            out: stream.clone_dtoh(&out)?,
            normed: stream.clone_dtoh(&normed)?,
            dests: dests
                .iter()
                .map(|d| Ok(stream.clone_dtoh(d)?))
                .collect::<anyhow::Result<_>>()?,
        };
        let exact = depth_reference(
            &row_blocks,
            &residual_want,
            true,
            &stream.clone_dtoh(&query)?,
        );
        assert_depth_close(&label, &got, &want, &exact, (2f64.powi(-22), 2));
    }
    Ok(())
}

/// The decode router, from the normalized input `moe_router` writes, against
/// `moe_router`: logits within 1e-5 (another summation order), the same experts, and
/// weights within 2e-6.
#[test]
fn moe_router_step_matches_moe_router() -> anyhow::Result<()> {
    let fixture = Fixture::load("routing")?;
    let (bias, _) = fixture.f32("router.bias")?;
    let gpu = Gpu::new(0)?;
    let stream = gpu.stream();
    let routing = Routing::load(gpu.context())?;
    let router = stream.clone_htod(&router_weight())?;
    let bias_dev = stream.clone_htod(&bias)?;
    let weight = stream.clone_htod(&pseudo_random_bf16(HIDDEN, 400, 0.5, 1.0))?;
    let mut counter = stream.alloc_zeros::<u32>(1)?;
    for row in 0..6u32 {
        let x = stream.clone_htod(&pseudo_random(HIDDEN, 500 + row, 4.0))?;
        let mut x_norm = stream.alloc_zeros::<bf16>(HIDDEN)?;
        let mut logits_want = stream.alloc_zeros::<f32>(EXPERTS)?;
        let mut indices_want = stream.alloc_zeros::<i32>(TOP_K)?;
        let mut weights_want = stream.alloc_zeros::<f32>(TOP_K)?;
        routing.moe_router(
            stream,
            &x,
            &weight,
            EPS,
            &router,
            &bias_dev,
            &mut x_norm,
            &mut logits_want,
            &mut indices_want,
            &mut weights_want,
            &mut counter,
            ROUTE_SCALE,
        )?;
        let mut logits = stream.alloc_zeros::<f32>(EXPERTS)?;
        let mut indices = stream.alloc_zeros::<i32>(TOP_K)?;
        let mut weights = stream.alloc_zeros::<f32>(TOP_K)?;
        routing.moe_router_step(
            stream,
            &x_norm,
            &router,
            &bias_dev,
            &mut logits,
            &mut indices,
            &mut weights,
            ROUTE_SCALE,
        )?;
        let label = format!("row {row}");
        assert_f32_within(
            &format!("{label}: logits"),
            &stream.clone_dtoh(&logits)?,
            &stream.clone_dtoh(&logits_want)?,
            1e-5,
        );
        let got = pairs_by_expert(&stream.clone_dtoh(&indices)?, &stream.clone_dtoh(&weights)?);
        let want = pairs_by_expert(
            &stream.clone_dtoh(&indices_want)?,
            &stream.clone_dtoh(&weights_want)?,
        );
        let experts = |p: &[Vec<(i32, f32)>]| -> Vec<i32> { p[0].iter().map(|e| e.0).collect() };
        assert_eq!(experts(&got), experts(&want), "{label}: experts");
        let weights_of = |p: &[Vec<(i32, f32)>]| -> Vec<f32> { p[0].iter().map(|e| e.1).collect() };
        assert_f32_within(
            &format!("{label}: weights"),
            &weights_of(&got),
            &weights_of(&want),
            2e-6,
        );
        assert_eq!(
            stream.clone_dtoh(&counter)?,
            [0],
            "the kernel resets its counter"
        );
    }
    Ok(())
}

/// The multi-row step router against the one-row step router: every row's logits,
/// indices and weights bit for bit, for 1 to 8 rows.
#[test]
fn moe_router_step_rows_matches_step() -> anyhow::Result<()> {
    let fixture = Fixture::load("routing")?;
    let (bias, _) = fixture.f32("router.bias")?;
    let gpu = Gpu::new(0)?;
    let stream = gpu.stream();
    let routing = Routing::load(gpu.context())?;
    let router = stream.clone_htod(&router_weight())?;
    let bias_dev = stream.clone_htod(&bias)?;
    let rows_max = STEP_ROUTER_ROWS;
    let x_host = pseudo_random_bf16(rows_max * HIDDEN, 600, 2.0, 0.0);
    // Each row alone through the one-row kernel.
    let mut want = (Vec::new(), Vec::new(), Vec::new());
    for r in 0..rows_max {
        let x = stream.clone_htod(&x_host[r * HIDDEN..(r + 1) * HIDDEN])?;
        let mut logits = stream.alloc_zeros::<f32>(EXPERTS)?;
        let mut indices = stream.alloc_zeros::<i32>(TOP_K)?;
        let mut weights = stream.alloc_zeros::<f32>(TOP_K)?;
        routing.moe_router_step(
            stream,
            &x,
            &router,
            &bias_dev,
            &mut logits,
            &mut indices,
            &mut weights,
            ROUTE_SCALE,
        )?;
        want.0.extend(stream.clone_dtoh(&logits)?);
        want.1.extend(stream.clone_dtoh(&indices)?);
        want.2.extend(stream.clone_dtoh(&weights)?);
    }
    let to_bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    for rows in 1..=rows_max {
        let x_rows = stream.clone_htod(&x_host[..rows * HIDDEN])?;
        let mut logits = stream.alloc_zeros::<f32>(rows * EXPERTS)?;
        let mut indices = stream.alloc_zeros::<i32>(rows * TOP_K)?;
        let mut weights = stream.alloc_zeros::<f32>(rows * TOP_K)?;
        routing.moe_router_step_rows(
            stream,
            &x_rows,
            &router,
            &bias_dev,
            &mut logits,
            &mut indices,
            &mut weights,
            ROUTE_SCALE,
            rows,
        )?;
        assert_eq!(
            to_bits(&stream.clone_dtoh(&logits)?),
            to_bits(&want.0[..rows * EXPERTS]),
            "{rows} rows: logits"
        );
        assert_eq!(
            stream.clone_dtoh(&indices)?,
            want.1[..rows * TOP_K],
            "{rows} rows: indices"
        );
        assert_eq!(
            to_bits(&stream.clone_dtoh(&weights)?),
            to_bits(&want.2[..rows * TOP_K]),
            "{rows} rows: weights"
        );
    }
    Ok(())
}

/// The chunked depth attention against the fused single-row kernel, for 1 to 5 block
/// states, 1, 3 and 4 queries, and every kind of partial, over 600 rows, more than the
/// grid's blocks, so blocks loop over rows. Every 37th row and the last are checked as in
/// `depth_attention_step_matches_fused`, with looser floors for the 1080 sampled rows: the
/// output within twice the fused kernel's error from an F64 reference or 2^-20 of the
/// largest output (the worst row was 3 F32 ulps off), the normalized partial within 4 ulps
/// (its sum of squares adds 2304 terms in another order; the worst row was 3 ulps off), and the destinations within one BF16 ulp or the output difference carried
/// through the norm. Destinations sit 2 x 2304 apart, as halves of the gate inputs. With
/// the conv partial, the residual matches `short_conv_residual_rows` bit for bit.
#[test]
#[expect(clippy::too_many_lines, reason = "four partials, one reference each")]
fn depth_attention_rows_matches_fused() -> anyhow::Result<()> {
    const ROWS: usize = 600;
    const STRIDE: usize = 2 * HIDDEN;
    let gpu = Gpu::new(0)?;
    let stream = gpu.stream();
    let routing = Routing::load(gpu.context())?;
    let conv = Conv::load(gpu.context())?;
    let weights: Vec<CudaSlice<bf16>> = (0..3)
        .map(|k| Ok(stream.clone_htod(&pseudo_random_bf16(HIDDEN, 200 + k, 0.5, 1.0))?))
        .collect::<anyhow::Result<_>>()?;
    let conv_weight = stream.clone_htod(&pseudo_random_bf16(HIDDEN * 4, 1, 0.5, 0.0))?;
    let conv_state_host = pseudo_random(3 * HIDDEN, 2, 2.0);
    let conv_state = stream.clone_htod(&conv_state_host)?;
    let raw_host = pseudo_random(ROWS * HIDDEN, 300, 2.0);
    let raw = stream.clone_htod(&raw_host)?;
    let raw_bf16_host: Vec<bf16> = raw_host.iter().map(|&v| bf16::from_f32(v)).collect();
    let raw_bf16 = stream.clone_htod(&raw_bf16_host)?;
    let hidden = stream.clone_htod(&pseudo_random_bf16(ROWS * HIDDEN, 301, 3.0, 0.0))?;
    // The reference residual; the call also shifts its own copy of the state.
    let mut residual_want = stream.alloc_zeros::<f32>(ROWS * HIDDEN)?;
    conv.short_conv_residual_rows(
        stream,
        &raw_bf16,
        &conv_weight,
        &mut stream.clone_htod(&conv_state_host)?,
        Some(&hidden as &dyn siphon_cuda::DeviceIn<bf16>),
        &mut residual_want,
        ROWS,
    )?;
    let residual_want = stream.clone_dtoh(&residual_want)?;
    let sampled: Vec<usize> = (0..ROWS).step_by(37).chain([ROWS - 1]).collect();

    for nb in 1..FUSED_MAX_CANDIDATES {
        let blocks_host = pseudo_random(nb * ROWS * HIDDEN, 400 + nb as u32, 1.0);
        let blocks = stream.clone_htod(&blocks_host)?;
        for nq in [1, 3, 4] {
            let queries_host = pseudo_random(nq * HIDDEN, 500 + nq as u32, 1.0);
            let queries = stream.clone_htod(&queries_host)?;
            for kind in ["normalized", "f32", "bf16", "conv"] {
                let mut residual = stream.alloc_zeros::<f32>(ROWS * HIDDEN)?;
                let partial = match kind {
                    "normalized" => RowsPartial::Normalized(raw.as_view()),
                    "f32" => RowsPartial::F32(raw.as_view()),
                    "bf16" => RowsPartial::Bf16(raw_bf16.as_view()),
                    _ => RowsPartial::ConvResidual {
                        x: raw_bf16.as_view(),
                        weight: &conv_weight,
                        state: conv_state.as_view(),
                        hidden: Some(hidden.as_view()),
                        residual: residual.as_view_mut(),
                    },
                };
                let mut normed = stream.alloc_zeros::<f32>(ROWS * HIDDEN)?;
                let mut out = stream.alloc_zeros::<f32>(ROWS * nq * HIDDEN)?;
                let mut dests: Vec<CudaSlice<bf16>> = (0..4)
                    .map(|_| Ok(stream.alloc_zeros::<bf16>(ROWS * STRIDE)?))
                    .collect::<anyhow::Result<_>>()?;
                let [d0, d1, d2, d3] = &mut dests[..] else {
                    unreachable!()
                };
                routing.depth_attention_rows(
                    stream,
                    DepthRows {
                        blocks: blocks.slice(..nb * ROWS * HIDDEN),
                        partial,
                        normed_partial: Some(normed.as_view_mut()),
                        queries: &queries,
                        out: Some(out.as_view_mut()),
                        probs: None,
                        dests: [
                            Some(norm_dest(d0, Some(&weights[0]))),
                            Some(norm_dest(d1, Some(&weights[1]))),
                            Some(norm_dest(d2, Some(&weights[2]))),
                            Some(norm_dest(d3, None)),
                        ],
                        dst_stride: STRIDE,
                        rows: ROWS,
                    },
                    EPS,
                )?;
                let out = stream.clone_dtoh(&out)?;
                let normed = stream.clone_dtoh(&normed)?;
                let dests: Vec<Vec<bf16>> = dests
                    .iter()
                    .map(|d| Ok(stream.clone_dtoh(d)?))
                    .collect::<anyhow::Result<_>>()?;
                if kind == "conv" {
                    assert_f32_within_ulps(
                        &format!("{nb} blocks, {nq} queries: residual"),
                        &stream.clone_dtoh(&residual)?,
                        &residual_want,
                        0,
                    );
                }
                for &row in &sampled {
                    let rh = row * HIDDEN..(row + 1) * HIDDEN;
                    let row_blocks: Vec<f32> = (0..nb)
                        .flat_map(|i| {
                            blocks_host[(i * ROWS + row) * HIDDEN..(i * ROWS + row + 1) * HIDDEN]
                                .iter()
                                .copied()
                        })
                        .collect();
                    let row_blocks_dev = stream.clone_htod(&row_blocks)?;
                    let (partial_host, normalize): (Vec<f32>, bool) = match kind {
                        "normalized" => (raw_host[rh.clone()].to_vec(), false),
                        "f32" => (raw_host[rh.clone()].to_vec(), true),
                        "bf16" => (
                            raw_bf16_host[rh.clone()]
                                .iter()
                                .map(|v| v.to_f32())
                                .collect(),
                            true,
                        ),
                        _ => (residual_want[rh.clone()].to_vec(), true),
                    };
                    let partial_dev = stream.clone_htod(&partial_host)?;
                    let bf16_dev = stream.clone_htod(&raw_bf16_host[rh.clone()])?;
                    let reference = match kind {
                        "normalized" => Partial::Normalized(partial_dev.as_view()),
                        "bf16" => Partial::Bf16(bf16_dev.as_view()),
                        _ => Partial::F32(partial_dev.as_view()),
                    };
                    let want = fused_depth(
                        &gpu,
                        &routing,
                        &row_blocks_dev,
                        reference,
                        &queries,
                        &weights,
                    )?;
                    let got = DepthResults {
                        out: out[row * nq * HIDDEN..(row + 1) * nq * HIDDEN].to_vec(),
                        normed: normed[rh].to_vec(),
                        dests: dests
                            .iter()
                            .map(|d| d[row * STRIDE..row * STRIDE + HIDDEN].to_vec())
                            .collect(),
                    };
                    let exact =
                        depth_reference(&row_blocks, &partial_host, normalize, &queries_host);
                    assert_depth_close(
                        &format!("{nb} blocks, {nq} queries, {kind}, row {row}"),
                        &got,
                        &want,
                        &exact,
                        (2f64.powi(-20), 4),
                    );
                }
            }
        }
    }
    Ok(())
}

/// The chunked message gate fed with a message rebuilt from the depth attention's weights
/// against the same gate fed with the depth attention's output rows, bit for bit, for 1
/// and 2 bank states and each of the 4 message slots, over 150 rows.
#[test]
fn message_gate_rebuilds_messages_bit_for_bit() -> anyhow::Result<()> {
    const ROWS: usize = 150;
    const SLOTS: usize = 4;
    let gpu = Gpu::new(0)?;
    let stream = gpu.stream();
    let routing = Routing::load(gpu.context())?;
    let misc = Misc::load(gpu.context())?;
    let upload_bf16 = |len: usize, seed: u32, scale: f32| -> anyhow::Result<CudaSlice<bf16>> {
        Ok(stream.clone_htod(&pseudo_random_bf16(len, seed, scale, 0.0))?)
    };
    let normed = stream.clone_htod(&pseudo_random(ROWS * HIDDEN, 900, 1.0))?;
    let queries = stream.clone_htod(&pseudo_random(SLOTS * HIDDEN, 901, 1.0))?;
    let h = stream.clone_htod(&pseudo_random(ROWS * HIDDEN, 902, 3.0))?;
    let down = upload_bf16(ROWS * 64, 903, 2.0)?;
    let forget = upload_bf16(HIDDEN * 64, 904, 0.3)?;
    let gain = upload_bf16(HIDDEN * 64, 905, 0.3)?;
    let norm_weight = stream.clone_htod(&pseudo_random_bf16(HIDDEN, 906, 0.5, 1.0))?;
    for nb in [1, 2] {
        let bank = stream.clone_htod(&pseudo_random(nb * ROWS * HIDDEN, 910 + nb as u32, 1.0))?;
        let mut messages = stream.alloc_zeros::<f32>(ROWS * SLOTS * HIDDEN)?;
        let mut probs = stream.alloc_zeros::<f32>(ROWS * SLOTS * (nb + 1))?;
        routing.depth_attention_rows(
            stream,
            DepthRows {
                blocks: bank.as_view(),
                partial: RowsPartial::Normalized(normed.as_view()),
                normed_partial: None,
                queries: &queries,
                out: Some(messages.as_view_mut()),
                probs: Some(probs.as_view_mut()),
                dests: [None, None, None, None],
                dst_stride: HIDDEN,
                rows: ROWS,
            },
            EPS,
        )?;
        for slot in 0..SLOTS {
            let mut results = Vec::new();
            for rebuild in [false, true] {
                let mut out = stream.alloc_zeros::<f32>(ROWS * HIDDEN)?;
                let mut out_normed = stream.alloc_zeros::<bf16>(ROWS * HIDDEN)?;
                let message = messages.slice(slot * HIDDEN..);
                let (m, m_stride) = if rebuild {
                    (normed.slice(..), HIDDEN)
                } else {
                    (
                        message.slice(..(ROWS - 1) * SLOTS * HIDDEN + HIDDEN),
                        SLOTS * HIDDEN,
                    )
                };
                let mix = MessageMix {
                    blocks: bank.as_view(),
                    probs: probs.slice(slot * (nb + 1)..),
                    probs_stride: SLOTS * (nb + 1),
                };
                misc.message_gate_out_rows(
                    stream,
                    &down,
                    &forget,
                    &gain,
                    (&h, HIDDEN),
                    (&m, m_stride),
                    rebuild.then_some(&mix),
                    &mut out,
                    Some((&norm_weight, &mut out_normed)),
                    EPS,
                    ROWS,
                )?;
                results.push((stream.clone_dtoh(&out)?, stream.clone_dtoh(&out_normed)?));
            }
            let label = format!("{nb} bank states, slot {slot}");
            assert_f32_within_ulps(&format!("{label}: out"), &results[1].0, &results[0].0, 0);
            assert_bf16_close(&format!("{label}: normed"), &results[1].1, &results[0].1, 0);
        }
    }
    Ok(())
}

/// Fused `RowsPartial::ConvResidualBatch` inside `depth_attention_rows` must match
/// `Conv::conv_residual_batch` followed by `RowsPartial::F32` bit for bit across single-row
/// batched steps (`rows_per_seq == 1`) and batched verify widths (`2..=8`), with and without
/// `hidden`.
#[test]
#[expect(
    clippy::too_many_lines,
    reason = "checks fused conv_residual_batch against unfused across shapes"
)]
fn depth_attention_rows_conv_residual_batch_matches_unfused() -> anyhow::Result<()> {
    use cudarc::driver::DevicePtr as _;
    use siphon_cuda::ops::spec::BATCH_STATE_WORDS;

    let gpu = Gpu::new(0)?;
    let stream = gpu.stream();
    let routing = Routing::load(gpu.context())?;
    let conv = Conv::load(gpu.context())?;
    let conv_weight = stream.clone_htod(&pseudo_random_bf16(HIDDEN * 4, 701, 0.5, 0.0))?;
    let norm_weight = stream.clone_htod(&pseudo_random_bf16(HIDDEN, 702, 0.5, 1.0))?;

    for &(seqs, rows_per_seq, has_hidden) in &[
        (7usize, 1usize, true),
        (4, 2, false),
        (5, 3, true),
        (3, 5, true),
        (4, 8, true),
    ] {
        let rows = seqs * rows_per_seq;
        let proj = stream.clone_htod(&pseudo_random_bf16(rows * HIDDEN, 710, 2.0, 0.0))?;
        let hidden = stream.clone_htod(&pseudo_random_bf16(rows * HIDDEN, 711, 3.0, 0.0))?;
        let conv_states: Vec<CudaSlice<f32>> = (0..seqs)
            .map(|s| Ok(stream.clone_htod(&pseudo_random(3 * HIDDEN, 720 + s as u32, 2.0))?))
            .collect::<anyhow::Result<_>>()?;
        let mut table = vec![0u64; seqs * BATCH_STATE_WORDS];
        for (s, st) in conv_states.iter().enumerate() {
            table[s * BATCH_STATE_WORDS + 4] = st.device_ptr(stream).0;
        }
        let states_dev = stream.clone_htod(&table)?;

        for nb in [1usize, 3, 5] {
            let blocks = stream.clone_htod(&pseudo_random(nb * rows * HIDDEN, 730 + nb as u32, 1.0))?;
            let queries = stream.clone_htod(&pseudo_random(HIDDEN, 740, 1.0))?;

            let mut residual_want = stream.alloc_zeros::<f32>(rows * HIDDEN)?;
            conv.conv_residual_batch(
                stream,
                &proj,
                &conv_weight,
                &states_dev.slice(..),
                has_hidden.then(|| hidden.as_view()).as_ref(),
                &mut residual_want,
                rows,
                rows_per_seq,
            )?;
            let mut out_want = stream.alloc_zeros::<f32>(rows * HIDDEN)?;
            let mut dst_want = stream.alloc_zeros::<bf16>(rows * HIDDEN)?;
            routing.depth_attention_rows(
                stream,
                DepthRows {
                    blocks: blocks.as_view(),
                    partial: RowsPartial::F32(residual_want.as_view()),
                    normed_partial: None,
                    queries: &queries,
                    out: Some(out_want.as_view_mut()),
                    probs: None,
                    dests: [Some(norm_dest(&mut dst_want, Some(&norm_weight))), None, None, None],
                    dst_stride: HIDDEN,
                    rows,
                },
                EPS,
            )?;

            let mut residual_got = stream.alloc_zeros::<f32>(rows * HIDDEN)?;
            let mut out_got = stream.alloc_zeros::<f32>(rows * HIDDEN)?;
            let mut dst_got = stream.alloc_zeros::<bf16>(rows * HIDDEN)?;
            routing.depth_attention_rows(
                stream,
                DepthRows {
                    blocks: blocks.as_view(),
                    partial: RowsPartial::ConvResidualBatch {
                        x: proj.as_view(),
                        weight: &conv_weight,
                        states: states_dev.slice(..),
                        rows_per_seq,
                        hidden: has_hidden.then(|| hidden.as_view()),
                        residual: residual_got.as_view_mut(),
                    },
                    normed_partial: None,
                    queries: &queries,
                    out: Some(out_got.as_view_mut()),
                    probs: None,
                    dests: [Some(norm_dest(&mut dst_got, Some(&norm_weight))), None, None, None],
                    dst_stride: HIDDEN,
                    rows,
                },
                EPS,
            )?;

            let label = format!("seqs={seqs} w={rows_per_seq} hidden={has_hidden} nb={nb}");
            assert_f32_within_ulps(
                &format!("{label}: residual"),
                &stream.clone_dtoh(&residual_got)?,
                &stream.clone_dtoh(&residual_want)?,
                0,
            );
            assert_f32_within_ulps(
                &format!("{label}: out"),
                &stream.clone_dtoh(&out_got)?,
                &stream.clone_dtoh(&out_want)?,
                0,
            );
            assert_bf16_close(
                &format!("{label}: dst"),
                &stream.clone_dtoh(&dst_got)?,
                &stream.clone_dtoh(&dst_want)?,
                0,
            );
        }
    }
    Ok(())
}

/// Validates [`MoePruning`] parameter bounds and checks dynamic top-k / probability-threshold
/// expert pruning across every top-8 router kernel:
/// - default pruning (`top_k = 8, threshold = 0.0`) is bit-exact against stock routing;
/// - hard `top_k < 8` emits `-1` / `0.0` in pruned slots and renormalizes kept weights;
/// - threshold pruning respects `min_k` and `top_k` and matches bit-for-bit between
///   `moe_router_step_pruned` and `moe_router_step_rows_pruned`.
#[test]
#[expect(clippy::too_many_lines, reason = "covers all five router entry points")]
fn moe_pruning_matches_reference_across_routers() -> anyhow::Result<()> {
    assert!(MoePruning::default().validate().is_ok());
    assert!(!MoePruning::default().is_active());
    for bad in [
        MoePruning {
            top_k: 0,
            threshold: 0.0,
            min_k: 0,
        },
        MoePruning {
            top_k: 9,
            threshold: 0.0,
            min_k: 4,
        },
        MoePruning {
            top_k: 4,
            threshold: 0.0,
            min_k: 0,
        },
        MoePruning {
            top_k: 4,
            threshold: 0.0,
            min_k: 5,
        },
        MoePruning {
            top_k: 8,
            threshold: -0.01,
            min_k: 4,
        },
        MoePruning {
            top_k: 8,
            threshold: 1.01,
            min_k: 4,
        },
        MoePruning {
            top_k: 8,
            threshold: f32::NAN,
            min_k: 4,
        },
    ] {
        assert!(bad.validate().is_err(), "expected {bad:?} to be rejected");
    }

    let fixture = Fixture::load("routing")?;
    let (bias, _) = fixture.f32("router.bias")?;
    let (logits16, _) = fixture.f32("router16.logits")?;
    let (hidden16, _) = fixture.bf16("router16.hidden")?;
    let rows = logits16.len() / EXPERTS;

    let gpu = Gpu::new(0)?;
    let stream = gpu.stream();
    let routing = Routing::load(gpu.context())?;

    let logits_dev = stream.clone_htod(&logits16)?;
    let bias_dev = stream.clone_htod(&bias)?;
    let mut stock_idx = stream.alloc_zeros::<i32>(rows * TOP_K)?;
    let mut stock_w = stream.alloc_zeros::<f32>(rows * TOP_K)?;
    routing.expert_routing(
        stream,
        &logits_dev,
        &bias_dev,
        &mut stock_idx,
        &mut stock_w,
        ROUTE_SCALE,
    )?;
    let stock_idx_h = stream.clone_dtoh(&stock_idx)?;
    let stock_w_h = stream.clone_dtoh(&stock_w)?;

    // Default MoePruning must match stock bit-for-bit.
    let mut def_idx = stream.alloc_zeros::<i32>(rows * TOP_K)?;
    let mut def_w = stream.alloc_zeros::<f32>(rows * TOP_K)?;
    routing.expert_routing_pruned(
        stream,
        &logits_dev,
        &bias_dev,
        &mut def_idx,
        &mut def_w,
        ROUTE_SCALE,
        MoePruning::default(),
    )?;
    assert_eq!(stream.clone_dtoh(&def_idx)?, stock_idx_h);
    assert_f32_within_ulps(
        "default pruning weights",
        &stream.clone_dtoh(&def_w)?,
        &stock_w_h,
        0,
    );

    // Hard top_k = 4 on fixture logits.
    let hard4 = MoePruning {
        top_k: 4,
        threshold: 0.0,
        min_k: 4,
    };
    assert!(hard4.is_active());
    let mut p_idx = stream.alloc_zeros::<i32>(rows * TOP_K)?;
    let mut p_w = stream.alloc_zeros::<f32>(rows * TOP_K)?;
    routing.expert_routing_pruned(
        stream,
        &logits_dev,
        &bias_dev,
        &mut p_idx,
        &mut p_w,
        ROUTE_SCALE,
        hard4,
    )?;
    let p_idx_h = stream.clone_dtoh(&p_idx)?;
    let p_w_h = stream.clone_dtoh(&p_w)?;
    for r in 0..rows {
        let base = r * TOP_K;
        assert_eq!(&p_idx_h[base..base + 4], &stock_idx_h[base..base + 4]);
        assert_eq!(&p_idx_h[base + 4..base + TOP_K], &[-1, -1, -1, -1]);
        for k in 4..TOP_K {
            assert_eq!(p_w_h[base + k].to_bits(), 0.0f32.to_bits());
        }
        let kept_sum: f32 = p_w_h[base..base + 4].iter().sum();
        assert!(
            (kept_sum - ROUTE_SCALE).abs() < 1e-5,
            "row {r}: kept weights sum to {kept_sum}, expected {ROUTE_SCALE}"
        );
    }

    // Synthetic logits with wide score gaps for unambiguous threshold pruning:
    // top 3 scores ~0.953, next 5 scores ~0.047 -> shares ~0.308 vs ~0.015.
    let mut synth_logits = vec![-10.0f32; EXPERTS];
    synth_logits[..3].fill(3.0);
    synth_logits[3..8].fill(-3.0);
    let zero_bias = vec![0.0f32; EXPERTS];
    let synth_dev = stream.clone_htod(&synth_logits)?;
    let zero_bias_dev = stream.clone_htod(&zero_bias)?;
    let mut s_idx = stream.alloc_zeros::<i32>(TOP_K)?;
    let mut s_w = stream.alloc_zeros::<f32>(TOP_K)?;
    routing.expert_routing_pruned(
        stream,
        &synth_dev,
        &zero_bias_dev,
        &mut s_idx,
        &mut s_w,
        ROUTE_SCALE,
        MoePruning {
            top_k: 6,
            threshold: 0.10,
            min_k: 2,
        },
    )?;
    let s_idx_h = stream.clone_dtoh(&s_idx)?;
    let s_w_h = stream.clone_dtoh(&s_w)?;
    assert_eq!(s_idx_h, vec![0, 1, 2, -1, -1, -1, -1, -1]);
    for &w in &s_w_h[..3] {
        assert!((w - ROUTE_SCALE / 3.0).abs() < 1e-5);
    }
    for &w in &s_w_h[3..TOP_K] {
        assert_eq!(w.to_bits(), 0.0f32.to_bits());
    }

    // Threshold + top_k across moe_router_step_pruned, moe_router_step_rows_pruned,
    // moe_router_rows_pruned, and moe_router_pruned.
    let dyn_prune = MoePruning {
        top_k: 6,
        threshold: 0.12,
        min_k: 2,
    };
    let weight_dev = stream.clone_htod(&router_weight())?;
    let hidden_dev = stream.clone_htod(&hidden16)?;
    let single_row_dev = stream.clone_htod(&hidden16[..HIDDEN])?;

    let mut step_logits = stream.alloc_zeros::<f32>(EXPERTS)?;
    let mut step_idx = stream.alloc_zeros::<i32>(TOP_K)?;
    let mut step_w = stream.alloc_zeros::<f32>(TOP_K)?;
    routing.moe_router_step_pruned(
        stream,
        &single_row_dev,
        &weight_dev,
        &bias_dev,
        &mut step_logits,
        &mut step_idx,
        &mut step_w,
        ROUTE_SCALE,
        dyn_prune,
    )?;

    let mut step_rows_logits = stream.alloc_zeros::<f32>(rows * EXPERTS)?;
    let mut step_rows_idx = stream.alloc_zeros::<i32>(rows * TOP_K)?;
    let mut step_rows_w = stream.alloc_zeros::<f32>(rows * TOP_K)?;
    routing.moe_router_step_rows_pruned(
        stream,
        &hidden_dev,
        &weight_dev,
        &bias_dev,
        &mut step_rows_logits,
        &mut step_rows_idx,
        &mut step_rows_w,
        ROUTE_SCALE,
        rows,
        dyn_prune,
    )?;

    let mut rows_logits = stream.alloc_zeros::<f32>(rows * EXPERTS)?;
    let mut rows_idx = stream.alloc_zeros::<i32>(rows * TOP_K)?;
    let mut rows_w = stream.alloc_zeros::<f32>(rows * TOP_K)?;
    let mut scratch = RouterRowsScratch::new(stream, rows)?;
    routing.moe_router_rows_pruned(
        stream,
        &hidden_dev,
        &weight_dev,
        &bias_dev,
        &mut rows_logits,
        &mut rows_idx,
        &mut rows_w,
        &mut scratch,
        ROUTE_SCALE,
        rows,
        dyn_prune,
    )?;

    let step_idx_h = stream.clone_dtoh(&step_idx)?;
    let step_w_h = stream.clone_dtoh(&step_w)?;
    let step_rows_idx_h = stream.clone_dtoh(&step_rows_idx)?;
    let step_rows_w_h = stream.clone_dtoh(&step_rows_w)?;
    let rows_idx_h = stream.clone_dtoh(&rows_idx)?;
    let rows_w_h = stream.clone_dtoh(&rows_w)?;

    assert_eq!(&step_rows_idx_h[..TOP_K], &step_idx_h[..]);
    assert_f32_within_ulps(
        "step_rows[0] vs step",
        &step_rows_w_h[..TOP_K],
        &step_w_h,
        0,
    );

    for (label, idx_h, w_h) in [
        ("moe_router_step_rows", &step_rows_idx_h, &step_rows_w_h),
        ("moe_router_rows", &rows_idx_h, &rows_w_h),
    ] {
        for r in 0..rows {
            let base = r * TOP_K;
            let kept = (0..TOP_K).filter(|&k| idx_h[base + k] >= 0).count();
            assert!(
                (dyn_prune.min_k..=dyn_prune.top_k).contains(&kept),
                "{label} row {r}: kept {kept} outside {}..={}",
                dyn_prune.min_k,
                dyn_prune.top_k
            );
            for k in 0..dyn_prune.min_k {
                assert!(idx_h[base + k] >= 0 && w_h[base + k] > 0.0);
            }
            for k in dyn_prune.top_k..TOP_K {
                assert_eq!(idx_h[base + k], -1);
                assert_eq!(w_h[base + k].to_bits(), 0.0f32.to_bits());
            }
            let mut sum = 0.0f32;
            for k in 0..TOP_K {
                if idx_h[base + k] < 0 {
                    assert_eq!(w_h[base + k].to_bits(), 0.0f32.to_bits());
                } else {
                    assert!(w_h[base + k] > 0.0);
                    sum += w_h[base + k];
                }
            }
            assert!(
                (sum - ROUTE_SCALE).abs() < 1e-5,
                "{label} row {r}: sum {sum} != {ROUTE_SCALE}"
            );
        }
    }

    // Single-row fused norm + router (`moe_router_pruned`).
    let x_f32 = stream.clone_htod(&pseudo_random(HIDDEN, 777, 1.0))?;
    let norm_w = stream.clone_htod(&pseudo_random_bf16(HIDDEN, 778, 0.5, 1.0))?;
    let mut x_out = stream.alloc_zeros::<bf16>(HIDDEN)?;
    let mut r_logits = stream.alloc_zeros::<f32>(EXPERTS)?;
    let mut r_idx = stream.alloc_zeros::<i32>(TOP_K)?;
    let mut r_w = stream.alloc_zeros::<f32>(TOP_K)?;
    let mut counter = stream.alloc_zeros::<u32>(1)?;
    routing.moe_router_pruned(
        stream,
        &x_f32,
        &norm_w,
        EPS,
        &weight_dev,
        &bias_dev,
        &mut x_out,
        &mut r_logits,
        &mut r_idx,
        &mut r_w,
        &mut counter,
        ROUTE_SCALE,
        dyn_prune,
    )?;
    let r_idx_h = stream.clone_dtoh(&r_idx)?;
    let r_w_h = stream.clone_dtoh(&r_w)?;
    for k in 0..dyn_prune.min_k {
        assert!(r_idx_h[k] >= 0 && r_w_h[k] > 0.0);
    }
    for k in dyn_prune.top_k..TOP_K {
        assert_eq!(r_idx_h[k], -1);
        assert_eq!(r_w_h[k].to_bits(), 0.0f32.to_bits());
    }
    let sum: f32 = r_w_h.iter().sum();
    assert!((sum - ROUTE_SCALE).abs() < 1e-5);
    Ok(())
}
