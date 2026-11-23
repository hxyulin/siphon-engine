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

use common::Fixture;
use half::bf16;
use inkling_cuda::{
    Gpu,
    ops::routing::{EXPERTS, RouterMatmul, Routing, TOP_K},
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
