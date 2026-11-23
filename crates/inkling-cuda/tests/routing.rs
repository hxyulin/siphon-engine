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
use inkling_cuda::{
    Gpu,
    ops::{
        misc::Misc,
        norm::Norm,
        routing::{EXPERTS, FUSED_MAX_CANDIDATES, NormDest, Partial, RouterMatmul, Routing, TOP_K},
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
