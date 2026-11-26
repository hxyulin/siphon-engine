//! Expert products against MLX outputs from `reference/fixtures/experts.py` and an F64
//! reference.
//!
//! The kernels add the exact products in another order than MLX's `fp_gather_qmv`, so
//! outputs may differ from MLX in the last BF16 bit. Every output must stay within the F64
//! bound described above `check_reference`, and the tests print how many differ from MLX.
//! `expert_mlp` must equal `expert_matvec`, the softcapped SwiGLU and `expert_matvec`
//! again bit for bit, and with the end of the MoE block, also `moe_conv_residual`.

#![cfg(target_os = "linux")]

mod common;

use std::{fs, path::PathBuf, sync::Arc};

use anyhow::{Context as _, ensure};
use common::Fixture;
use cudarc::driver::{CudaSlice, CudaStream};
use half::bf16;
use safetensors::{Dtype, SafeTensors};
use siphon::Checkpoint;
use siphon_cuda::{
    Gpu,
    ops::{
        conv::Conv,
        elementwise::Elementwise,
        experts::{ExpertFormat, ExpertScratch, ExpertWeights, Experts, MoeEnd, QuantizedRows},
    },
};

const SLOTS: usize = 8;
const HIDDEN: usize = 2304;
const INTERMEDIATE: usize = 1152;
const GATE_SOFTCAP: f32 = 4.0;
const UP_SOFTCAP: f32 = 25.0;

/// Raw U8 tensor `key` of `tests/fixtures/<name>.safetensors`, which `Fixture` does not read.
fn fixture_bytes(name: &str, key: &str) -> anyhow::Result<Vec<u8>> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join(format!("tests/fixtures/{name}.safetensors"));
    let bytes = fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
    let tensors = SafeTensors::deserialize(&bytes)?;
    let view = tensors
        .tensor(key)
        .with_context(|| format!("{key} missing from fixture {name}"))?;
    ensure!(
        view.dtype() == Dtype::U8,
        "{key} is {:?}, expected U8",
        view.dtype()
    );
    Ok(view.data().to_vec())
}

struct Setup {
    gpu: Gpu,
    experts: Experts,
}

impl Setup {
    fn new() -> anyhow::Result<Self> {
        let gpu = Gpu::new(0)?;
        let experts = Experts::load(gpu.context())?;
        Ok(Self { gpu, experts })
    }

    fn stream(&self) -> &Arc<CudaStream> {
        self.gpu.stream()
    }
}

/// Projection `name` (`gate_up` or `down`) of fixture `file`, on the GPU.
struct Projection {
    weight: CudaSlice<u8>,
    scales: CudaSlice<u8>,
    global_scale: Option<CudaSlice<f32>>,
    n: usize,
    k: usize,
}

impl Projection {
    fn load(s: &Setup, fixture: &Fixture, file: &str, name: &str) -> anyhow::Result<Self> {
        let (_, x_shape) = fixture.bf16(&format!("{name}.x"))?;
        let (_, out_shape) = fixture.bf16(&format!("{name}.out"))?;
        let global_scale = if file.ends_with("nvfp4") {
            let (values, _) = fixture.f32(&format!("{name}.global_scale"))?;
            Some(s.stream().clone_htod(&values)?)
        } else {
            None
        };
        Ok(Self {
            weight: s
                .stream()
                .clone_htod(&fixture_bytes(file, &format!("{name}.weight"))?)?,
            scales: s
                .stream()
                .clone_htod(&fixture_bytes(file, &format!("{name}.scales"))?)?,
            global_scale,
            n: out_shape[2],
            k: x_shape[x_shape.len() - 1],
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

/// Runs both projections of `experts_<format>`, checks them against the F64 reference and
/// prints how far they are from MLX.
fn check_fixture(format: &str) -> anyhow::Result<()> {
    let s = Setup::new()?;
    let file = format!("experts_{format}");
    let fixture = Fixture::load(&file)?;
    let (indices, _) = fixture.i32("indices")?;
    let indices_gpu = s.stream().clone_htod(&indices)?;
    for name in ["gate_up", "down"] {
        let projection = Projection::load(&s, &fixture, &file, name)?;
        let (x, x_shape) = fixture.bf16(&format!("{name}.x"))?;
        let (want, _) = fixture.bf16(&format!("{name}.out"))?;
        let x_gpu = s.stream().clone_htod(&x)?;
        let mut out = s.stream().alloc_zeros::<bf16>(want.len())?;
        s.experts.expert_matvec(
            s.stream(),
            &x_gpu,
            &indices_gpu,
            &projection.weights(),
            &mut out,
        )?;
        let got = s.stream().clone_dtoh(&out)?;
        let weight = fixture_bytes(&file, &format!("{name}.weight"))?;
        let scales = fixture_bytes(&file, &format!("{name}.scales"))?;
        let host = HostProjection {
            weight: &weight,
            scales: &scales,
            global_scale: match format {
                "nvfp4" => Some(fixture.f32(&format!("{name}.global_scale"))?.0),
                _ => None,
            },
            n: projection.n,
            k: projection.k,
        };
        let inputs_per_row = if x_shape.len() == 3 { SLOTS } else { 1 };
        let label = format!("{format} {name}");
        let worst = check_reference(&label, &host, &to_f64(&x), &indices, inputs_per_row, &got);
        report_ulps(
            &format!("{label}: worst error {worst:.3} of bound; against MLX"),
            &got,
            &want,
        );
    }
    Ok(())
}

/// Prints how many values of `got` differ from `want` by 0, 1, 2 or more BF16 ulps.
fn report_ulps(label: &str, got: &[bf16], want: &[bf16]) {
    let mut ulps = [0usize; 4];
    let mut max_ulps = 0;
    for (a, b) in got.iter().zip(want) {
        let d = common::bf16_ulps(*a, *b);
        ulps[(d as usize).min(3)] += 1;
        max_ulps = max_ulps.max(d);
    }
    eprintln!(
        "{label}: {} equal, {} at 1 ulp, {} at 2, {} beyond (max {max_ulps}) of {}",
        ulps[0],
        ulps[1],
        ulps[2],
        ulps[3],
        got.len()
    );
}

#[test]
fn nvfp4_matches_mlx() -> anyhow::Result<()> {
    check_fixture("nvfp4")
}

#[test]
fn mxfp8_matches_mlx() -> anyhow::Result<()> {
    check_fixture("mxfp8")
}

#[test]
fn unknown_expert_gives_nan() -> anyhow::Result<()> {
    let s = Setup::new()?;
    let file = "experts_nvfp4";
    let fixture = Fixture::load(file)?;
    let projection = Projection::load(&s, &fixture, file, "gate_up")?;
    // The fixture keeps 10 experts, so ids -1 and 10 are outside it.
    let indices = s.stream().clone_htod(&[0, 1, 2, -1, 4, 5, 10, 7])?;
    let (x, _) = fixture.bf16("gate_up.x")?;
    let x = s.stream().clone_htod(&x[..projection.k])?;
    let mut out = s.stream().alloc_zeros::<bf16>(SLOTS * projection.n)?;
    s.experts
        .expert_matvec(s.stream(), &x, &indices, &projection.weights(), &mut out)?;
    let got = s.stream().clone_dtoh(&out)?;
    for (slot, row) in got.chunks(projection.n).enumerate() {
        let bad = slot == 3 || slot == 6;
        assert!(
            row.iter().all(|v| v.is_nan() == bad),
            "slot {slot}: expected {} values",
            if bad { "NaN" } else { "finite" }
        );
    }
    Ok(())
}

/// Directory of the original checkpoint: `SIPHON_MODEL`, or the GPU host's usual location.
fn model_dir() -> anyhow::Result<PathBuf> {
    if let Some(dir) = std::env::var_os("SIPHON_MODEL") {
        return Ok(dir.into());
    }
    let home = std::env::var_os("HOME").context("HOME is not set")?;
    Ok(PathBuf::from(home).join(".local/share/inkling-mini/model"))
}

fn f32s(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}

/// One full projection of `layer` from the checkpoint, uploaded as raw bytes.
fn checkpoint_projection(
    stream: &Arc<CudaStream>,
    checkpoint: &Checkpoint,
    layer: usize,
    which: &str,
    k: usize,
) -> anyhow::Result<Projection> {
    let prefix = format!("model.llm.layers.{layer}.mlp.experts.{which}_weight");
    let bytes = |suffix: &str| {
        let name = format!("{prefix}{suffix}");
        let entry = checkpoint
            .get(&name)
            .with_context(|| format!("{name} missing from checkpoint"))?;
        anyhow::Ok(checkpoint.bytes(entry))
    };
    let global_scale = match checkpoint.get(&format!("{prefix}_amax")) {
        Some(_) => {
            // As MLX-LM's `sanitize` computes it, in F32.
            let denominator = f32s(bytes("_outer_scale_denominator")?)[0];
            let scale = 6.0 * 448.0 / denominator;
            let values: Vec<f32> = f32s(bytes("_amax")?).iter().map(|a| a * scale).collect();
            Some(stream.clone_htod(&values)?)
        }
        None => None,
    };
    Ok(Projection {
        weight: stream.clone_htod(bytes("")?)?,
        scales: stream.clone_htod(bytes("_scale")?)?,
        global_scale,
        n: HIDDEN,
        k,
    })
}

/// Layers 0 (NVFP4) and 20 (MXFP8) of the checkpoint on MLX's inputs: prints how far the
/// separate kernels are from MLX, and requires `expert_mlp` to equal them.
#[test]
#[ignore = "reads the original checkpoint from SIPHON_MODEL or ~/.local/share/inkling-mini/model"]
fn expert_mlp_matches_mlx() -> anyhow::Result<()> {
    let s = Setup::new()?;
    let elementwise = Elementwise::load(s.gpu.context())?;
    let checkpoint = Checkpoint::open(model_dir()?)?;
    let fixture = Fixture::load("experts_mlp")?;
    for layer in [0, 20] {
        let stream = s.stream();
        let w13 = checkpoint_projection(stream, &checkpoint, layer, "w13", HIDDEN)?;
        let w2 = checkpoint_projection(stream, &checkpoint, layer, "w2", INTERMEDIATE)?;
        let (x, _) = fixture.bf16(&format!("layer{layer}.x"))?;
        let (indices, _) = fixture.i32(&format!("layer{layer}.indices"))?;
        let (want_gate_up, _) = fixture.bf16(&format!("layer{layer}.gate_up"))?;
        let (want, _) = fixture.bf16(&format!("layer{layer}.out"))?;

        let x = stream.clone_htod(&x)?;
        let indices = stream.clone_htod(&indices)?;
        let (gate_up, act, out) = separate_mlp(
            &s,
            &elementwise,
            &x,
            &indices,
            &w13.weights(),
            &w2.weights(),
        )?;
        report_ulps(&format!("layer {layer} gate_up"), &gate_up, &want_gate_up);
        report_ulps(&format!("layer {layer} out"), &out, &want);
        check_mlp(&s, &x, &indices, &w13.weights(), &w2.weights(), &act, &out)?;
    }
    Ok(())
}

/// `expert_matvec` on `w13`, the softcapped SwiGLU and `expert_matvec` on `w2`: the gate/up
/// projection, the activations and the output.
fn separate_mlp(
    s: &Setup,
    elementwise: &Elementwise,
    x: &CudaSlice<bf16>,
    indices: &CudaSlice<i32>,
    w13: &ExpertWeights<'_>,
    w2: &ExpertWeights<'_>,
) -> anyhow::Result<(Vec<bf16>, Vec<bf16>, Vec<bf16>)> {
    let stream = s.stream();
    let pairs = indices.len();
    let mut gate_up = stream.alloc_zeros::<bf16>(pairs * w13.n)?;
    s.experts
        .expert_matvec(stream, x, indices, w13, &mut gate_up)?;
    let mut act = stream.alloc_zeros::<bf16>(pairs * w2.k)?;
    elementwise.softcapped_swiglu(stream, &gate_up, &mut act, GATE_SOFTCAP, UP_SOFTCAP)?;
    let mut out = stream.alloc_zeros::<bf16>(pairs * w2.n)?;
    s.experts
        .expert_matvec(stream, &act, indices, w2, &mut out)?;
    Ok((
        stream.clone_dtoh(&gate_up)?,
        stream.clone_dtoh(&act)?,
        stream.clone_dtoh(&out)?,
    ))
}

/// Runs `expert_mlp` twice, so the second run starts from the counters the first left, and
/// requires both to give `want_act` and `want_out` bit for bit.
fn check_mlp(
    s: &Setup,
    x: &CudaSlice<bf16>,
    indices: &CudaSlice<i32>,
    w13: &ExpertWeights<'_>,
    w2: &ExpertWeights<'_>,
    want_act: &[bf16],
    want_out: &[bf16],
) -> anyhow::Result<()> {
    let stream = s.stream();
    let pairs = indices.len();
    let mut sync = stream.alloc_zeros::<u32>(1 + pairs)?;
    for run in 0..2 {
        let mut act = stream.alloc_zeros::<bf16>(pairs * w2.k)?;
        let mut out = stream.alloc_zeros::<bf16>(pairs * w2.n)?;
        s.experts.expert_mlp(
            stream,
            x,
            indices,
            w13,
            w2,
            &mut act,
            &mut out,
            &mut sync,
            GATE_SOFTCAP,
            UP_SOFTCAP,
            None,
        )?;
        let same = |got: &[bf16], want: &[bf16]| {
            got.iter()
                .zip(want)
                .all(|(a, b)| a.to_bits() == b.to_bits())
        };
        assert!(
            same(&stream.clone_dtoh(&act)?, want_act),
            "run {run}: act differs"
        );
        assert!(
            same(&stream.clone_dtoh(&out)?, want_out),
            "run {run}: out differs"
        );
        assert!(
            stream.clone_dtoh(&sync)?.iter().all(|&v| v == 0),
            "run {run}: sync not reset"
        );
    }
    Ok(())
}

/// `expert_mlp` equals the separate kernels on random weights, with unknown ids.
#[test]
fn expert_mlp_matches_separate_kernels() -> anyhow::Result<()> {
    let s = Setup::new()?;
    let elementwise = Elementwise::load(s.gpu.context())?;
    let stream = s.stream();
    let experts = 10;
    for fp4 in [true, false] {
        // 256 output rows: a multiple of both formats' w2 block heights.
        let w13 = Synthetic::with_experts(&s, experts, fp4, 2 * INTERMEDIATE, HIDDEN, 41)?;
        let w2 = Synthetic::with_experts(&s, experts, fp4, 256, INTERMEDIATE, 43)?;
        for rows in [1, 3] {
            let mut ids: Vec<i32> = (0..rows * SLOTS)
                .map(|i| i32::try_from(i * 7 % experts).expect("ids fit in i32"))
                .collect();
            if rows == 3 {
                ids[4] = -1;
                ids[13] = 10;
            }
            let indices = stream.clone_htod(&ids)?;
            let x = stream.clone_htod(&common::pseudo_random_bf16(
                rows * HIDDEN,
                rows as u32,
                1.0,
                0.0,
            ))?;
            let (_, act, out) = separate_mlp(
                &s,
                &elementwise,
                &x,
                &indices,
                &w13.gpu.weights(),
                &w2.gpu.weights(),
            )?;
            for (pair, &id) in ids.iter().enumerate() {
                let bad = !usize::try_from(id).is_ok_and(|id| id < experts);
                assert!(
                    act[pair * INTERMEDIATE..(pair + 1) * INTERMEDIATE]
                        .iter()
                        .all(|v| v.is_nan() == bad)
                );
            }
            check_mlp(
                &s,
                &x,
                &indices,
                &w13.gpu.weights(),
                &w2.gpu.weights(),
                &act,
                &out,
            )?;
        }
    }
    Ok(())
}

/// The inputs of one decode row's MoE block in `expert_mlp_end_matches_moe_conv_residual`.
struct MoeRow {
    x: CudaSlice<bf16>,
    indices: CudaSlice<i32>,
    routing: CudaSlice<f32>,
    residual: CudaSlice<f32>,
}

impl MoeRow {
    fn new(stream: &Arc<CudaStream>, ids: &[i32], seed: u32) -> anyhow::Result<Self> {
        let routing: Vec<f32> = common::pseudo_random(SLOTS, seed + 1, 1.0)
            .iter()
            .map(|v| v.abs())
            .collect();
        Ok(Self {
            x: stream.clone_htod(&common::pseudo_random_bf16(HIDDEN, seed, 1.0, 0.0))?,
            indices: stream.clone_htod(ids)?,
            routing: stream.clone_htod(&routing)?,
            residual: stream.clone_htod(&common::pseudo_random(HIDDEN, seed + 2, 4.0))?,
        })
    }
}

/// The activations, expert outputs and MoE block end of one row, through `expert_mlp` with
/// the end when `fused`, or without it and then `moe_conv_residual`. `state` is the conv
/// state, updated.
fn moe_end(
    s: &Setup,
    conv: &Conv,
    (w13, w2, conv_weight): (&ExpertWeights<'_>, &ExpertWeights<'_>, &CudaSlice<bf16>),
    row: &MoeRow,
    (state, sync): (&mut CudaSlice<f32>, &mut CudaSlice<u32>),
    fused: bool,
) -> anyhow::Result<[Vec<u16>; 3]> {
    let stream = s.stream();
    let mut act = stream.alloc_zeros::<bf16>(SLOTS * INTERMEDIATE)?;
    let mut out = stream.alloc_zeros::<bf16>(SLOTS * HIDDEN)?;
    let mut hidden = stream.alloc_zeros::<bf16>(HIDDEN)?;
    let end = fused.then_some(MoeEnd {
        weights: &row.routing,
        conv_weight,
        conv_state: &mut *state,
        residual: &row.residual,
        hidden: &mut hidden,
    });
    s.experts.expert_mlp(
        stream,
        &row.x,
        &row.indices,
        w13,
        w2,
        &mut act,
        &mut out,
        sync,
        GATE_SOFTCAP,
        UP_SOFTCAP,
        end,
    )?;
    if !fused {
        conv.moe_conv_residual(
            stream,
            &out,
            &row.routing,
            conv_weight,
            state,
            &row.residual,
            &mut hidden,
        )?;
    }
    let bits = |v: &CudaSlice<bf16>| -> anyhow::Result<Vec<u16>> {
        Ok(stream
            .clone_dtoh(v)?
            .into_iter()
            .map(bf16::to_bits)
            .collect())
    };
    Ok([bits(&act)?, bits(&out)?, bits(&hidden)?])
}

/// `expert_mlp` with the end of the MoE block equals `expert_mlp` without it followed by
/// `moe_conv_residual`, bit for bit, in its outputs, products and conv state, with a
/// repeated and an unknown expert; two steps in a row check that the counters reset.
#[test]
fn expert_mlp_end_matches_moe_conv_residual() -> anyhow::Result<()> {
    let s = Setup::new()?;
    let conv = Conv::load(s.gpu.context())?;
    let stream = s.stream();
    let experts = 10;
    for fp4 in [true, false] {
        let w13 = Synthetic::with_experts(&s, experts, fp4, 2 * INTERMEDIATE, HIDDEN, 51)?;
        let w2 = Synthetic::with_experts(&s, experts, fp4, HIDDEN, INTERMEDIATE, 53)?;
        let conv_weight =
            stream.clone_htod(&common::pseudo_random_bf16(4 * HIDDEN, 7, 0.5, 0.0))?;
        let weights = (&w13.gpu.weights(), &w2.gpu.weights(), &conv_weight);
        let start = common::pseudo_random(3 * HIDDEN, 9, 2.0);
        let mut state_want = stream.clone_htod(&start)?;
        let mut state_got = stream.clone_htod(&start)?;
        let mut sync_want = stream.alloc_zeros::<u32>(1 + SLOTS)?;
        let mut sync_got = stream.alloc_zeros::<u32>(1 + SLOTS + HIDDEN / 16)?;
        let steps: [&[i32]; 2] = [&[3, 1, 4, 1, 5, 9, 2, 6], &[0, 7, -1, 8, 2, 3, 3, 6]];
        for (step, ids) in (0u32..).zip(steps) {
            let row = MoeRow::new(stream, ids, 60 + 10 * step)?;
            let label = format!("fp4 {fp4}, step {step}");
            let want = moe_end(
                &s,
                &conv,
                weights,
                &row,
                (&mut state_want, &mut sync_want),
                false,
            )?;
            let got = moe_end(
                &s,
                &conv,
                weights,
                &row,
                (&mut state_got, &mut sync_got),
                true,
            )?;
            for (name, got, want) in ["act", "out", "hidden"]
                .into_iter()
                .zip(&got)
                .zip(&want)
                .map(|((n, g), w)| (n, g, w))
            {
                assert_eq!(got, want, "{label}: {name}");
            }
            let state = |v: &CudaSlice<f32>| -> anyhow::Result<Vec<u32>> {
                Ok(stream
                    .clone_dtoh(v)?
                    .into_iter()
                    .map(f32::to_bits)
                    .collect())
            };
            assert_eq!(
                state(&state_got)?,
                state(&state_want)?,
                "{label}: conv state"
            );
            assert!(
                stream.clone_dtoh(&sync_got)?.iter().all(|&v| v == 0),
                "{label}: sync not reset"
            );
        }
    }
    Ok(())
}

// `expert_matmul`, the expert-grouped product for prefill.
//
// It computes the same exact products as `expert_matvec` but adds them in another order, so
// results may differ in the last BF16 bit. The checks compare it with an F64 reference on the
// decoded weights and allow, per output, `2^-8 * |ref| + 2^-14 * sum`, where `sum` is the sum
// of the absolute products times the output scale:
//
// - Each F32 addition, rounded or truncated by the tensor cores, costs at most 2^-23 of its
//   partial sum. A product passes through at most k / 16 + 4 additions, 148 for k = 2304,
//   which bounds the F32 error by 148 * 2^-23 * sum < 2^-15.8 * sum.
// - Rounding the F32 result to BF16 costs at most half an ulp: 2^-8 of the F32 value, which is
//   at most `|ref|` plus the F32 error.
// - The total is at most 2^-8 * |ref| + (1 + 2^-8) * 2^-15.8 * sum, inside the bound with a
//   factor of 3.5 to spare on the second term.
//
// `expert_matvec` must meet the same bound, which checks the reference itself.

const EXPERTS: usize = 128;
/// `expert_group` writes NaN for unknown ids and the GEMM writes every other output, so a
/// value left over from before the call shows up as this.
const SENTINEL: f32 = 1234.0;

fn e2m1(nibble: u8) -> f64 {
    const VALUES: [f64; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let v = VALUES[usize::from(nibble & 7)];
    if nibble & 8 == 0 { v } else { -v }
}

fn e4m3(bits: u8) -> f64 {
    let sign = if bits & 0x80 == 0 { 1.0 } else { -1.0 };
    let exponent = i32::from((bits >> 3) & 15);
    let mantissa = f64::from(bits & 7) / 8.0;
    if exponent == 0 {
        sign * mantissa * 2f64.powi(-6)
    } else {
        sign * (1.0 + mantissa) * 2f64.powi(exponent - 7)
    }
}

fn e8m0(bits: u8) -> f64 {
    if bits == 0 {
        0.0
    } else {
        2f64.powi(i32::from(bits) - 127)
    }
}

/// Host copy of a projection's bytes, for the F64 reference.
struct HostProjection<'a> {
    weight: &'a [u8],
    scales: &'a [u8],
    /// NVFP4 only.
    global_scale: Option<Vec<f32>>,
    n: usize,
    k: usize,
}

impl HostProjection<'_> {
    /// The decoded weights of `expert`, `[n, k]`, before the global scale.
    fn decode(&self, expert: usize) -> Vec<f64> {
        let k = self.k;
        let mut decoded = vec![0.0; self.n * k];
        for row in 0..self.n {
            let flat_row = expert * self.n + row;
            for col in 0..k {
                decoded[row * k + col] = if self.global_scale.is_some() {
                    let byte = self.weight[flat_row * k / 2 + col / 2];
                    let nibble = if col % 2 == 0 { byte & 15 } else { byte >> 4 };
                    e2m1(nibble) * e4m3(self.scales[flat_row * k / 16 + col / 16])
                } else {
                    e4m3(self.weight[flat_row * k + col])
                        * e8m0(self.scales[flat_row * k / 32 + col / 32])
                };
            }
        }
        decoded
    }

    fn output_scale(&self, expert: usize) -> f64 {
        self.global_scale
            .as_ref()
            .map_or(1.0, |g| f64::from(g[expert]) / (448.0 * 6.0))
    }
}

/// Checks every output of `got` (`[pairs, n]`) against the F64 reference, and
/// returns the largest error as a fraction of its bound.
fn check_reference(
    label: &str,
    host: &HostProjection<'_>,
    x: &[f64],
    indices: &[i32],
    inputs_per_row: usize,
    got: &[bf16],
) -> f64 {
    let (n, k) = (host.n, host.k);
    let expert_of = |pair: usize| usize::try_from(indices[pair]).ok();
    let mut experts: Vec<usize> = (0..indices.len()).filter_map(expert_of).collect();
    experts.sort_unstable();
    experts.dedup();
    std::thread::scope(|scope| {
        let threads = std::thread::available_parallelism().map_or(1, usize::from);
        let handles: Vec<_> = (0..threads)
            .map(|thread| {
                let experts = &experts;
                scope.spawn(move || {
                    let mut worst = 0.0f64;
                    for &expert in experts.iter().skip(thread).step_by(threads) {
                        let decoded = host.decode(expert);
                        let scale = host.output_scale(expert);
                        for pair in (0..indices.len()).filter(|&p| expert_of(p) == Some(expert)) {
                            let input = if inputs_per_row == 1 {
                                pair / SLOTS
                            } else {
                                pair
                            };
                            let xin = &x[input * k..(input + 1) * k];
                            for col in 0..n {
                                let (mut dot, mut sum) = (0.0f64, 0.0f64);
                                let row = &decoded[col * k..(col + 1) * k];
                                for (xi, wi) in xin.iter().zip(row) {
                                    let product = xi * wi;
                                    dot += product;
                                    sum += product.abs();
                                }
                                let want = dot * scale;
                                let bound =
                                    2f64.powi(-8) * want.abs() + 2f64.powi(-14) * sum * scale;
                                let value = f64::from(got[pair * n + col].to_f32());
                                let err = (value - want).abs();
                                assert!(
                                    err <= bound,
                                    "{label}: pair {pair} (expert {expert}) column {col}: got \
                                     {value}, want {want}, bound {bound}"
                                );
                                if bound > 0.0 {
                                    worst = worst.max(err / bound);
                                }
                            }
                        }
                    }
                    worst
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("reference thread panicked"))
            .fold(0.0, f64::max)
    })
}

fn to_f64(x: &[bf16]) -> Vec<f64> {
    x.iter().map(|v| f64::from(v.to_f32())).collect()
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

    fn bytes(&mut self, len: usize) -> Vec<u8> {
        (0..len).map(|_| self.byte()).collect()
    }

    /// Uniform in `[0, bound)`, for bound below 2^20.
    fn below(&mut self, bound: usize) -> usize {
        (usize::try_from(self.next() >> 44).expect("20 bits fit") * bound) >> 20
    }
}

/// Random bytes and scales in the ranges real checkpoints use, on the host and the GPU.
struct Synthetic {
    weight: Vec<u8>,
    scales: Vec<u8>,
    global_scale: Option<Vec<f32>>,
    gpu: Projection,
}

impl Synthetic {
    fn new(s: &Setup, fp4: bool, n: usize, k: usize, seed: u32) -> anyhow::Result<Self> {
        Self::with_experts(s, EXPERTS, fp4, n, k, seed)
    }

    fn with_experts(
        s: &Setup,
        experts: usize,
        fp4: bool,
        n: usize,
        k: usize,
        seed: u32,
    ) -> anyhow::Result<Self> {
        let mut random = SplitMix(u64::from(seed));
        let (weight, scales, global_scale) = if fp4 {
            let weight = random.bytes(experts * n * k / 2);
            // Positive E4M3 scales from 2^-3 to 2^2: exponent fields 4 to 9.
            let scales = (0..experts * n * k / 16)
                .map(|_| 32 + random.byte() % 48)
                .collect();
            let global_scale = common::pseudo_random(experts, seed + 2, 1.0)
                .into_iter()
                .map(|v| 448.0 * 6.0 * 0.004 * (1.5 + v))
                .collect();
            (weight, scales, Some(global_scale))
        } else {
            // Any E4M3 value but NaN (0x7f, 0xff).
            let weight = (0..experts * n * k)
                .map(|_| {
                    let v = random.byte();
                    if v & 0x7f == 0x7f { v ^ 1 } else { v }
                })
                .collect();
            // E8M0 scales from 2^-12 to 2^-5.
            let scales = (0..experts * n * k / 32)
                .map(|_| 115 + random.byte() % 8)
                .collect();
            (weight, scales, None)
        };
        let gpu = Projection {
            weight: s.stream().clone_htod(&weight)?,
            scales: s.stream().clone_htod(&scales)?,
            global_scale: global_scale
                .as_ref()
                .map(|g| s.stream().clone_htod(g))
                .transpose()?,
            n,
            k,
        };
        Ok(Self {
            weight,
            scales,
            global_scale,
            gpu,
        })
    }

    fn host(&self) -> HostProjection<'_> {
        HostProjection {
            weight: &self.weight,
            scales: &self.scales,
            global_scale: self.global_scale.clone(),
            n: self.gpu.n,
            k: self.gpu.k,
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum Routing {
    /// 8 distinct experts per row, uniform over all 128.
    Uniform,
    /// 8 distinct experts per row, drawn as `128 * u^3`: expert 0 takes about a fifth of the
    /// pairs and most rows pick it.
    Skewed,
    /// Every row picks experts 0 to 7.
    Same,
    /// 8 distinct experts per row from the first 24 only; 104 experts get no pairs.
    Subset,
}

fn routing(rows: usize, routing: Routing, seed: u32) -> Vec<i32> {
    let mut random = SplitMix(u64::from(seed));
    let mut indices = Vec::with_capacity(rows * SLOTS);
    for _ in 0..rows {
        let mut picked: Vec<usize> = Vec::with_capacity(SLOTS);
        while picked.len() < SLOTS {
            let expert = match routing {
                Routing::Uniform => random.below(EXPERTS),
                Routing::Skewed => {
                    // 128 * u^3 for u uniform in [0, 1), from 16 random bits.
                    let u = random.below(1 << 16);
                    (((u * u) >> 16) * u * EXPERTS) >> 32
                }
                Routing::Same => picked.len(),
                Routing::Subset => random.below(24),
            };
            if !picked.contains(&expert) {
                picked.push(expert);
            }
        }
        indices.extend(
            picked
                .into_iter()
                .map(|e| i32::try_from(e).expect("expert ids fit in i32")),
        );
    }
    indices
}

/// Runs `expert_matvec` and `expert_matmul` on the same inputs, checks both against the F64
/// reference, checks that `expert_matmul` writes every output and repeats bit for bit, and
/// reports how far the two kernels are apart.
#[expect(
    clippy::too_many_arguments,
    reason = "the kernels' inputs are separate buffers"
)]
fn compare_kernels(
    s: &Setup,
    scratch: &mut ExpertScratch,
    label: &str,
    projection: &Projection,
    host: &HostProjection<'_>,
    x: &[bf16],
    indices: &[i32],
    inputs_per_row: usize,
) -> anyhow::Result<()> {
    let stream = s.stream();
    let pairs = indices.len();
    let n = projection.n;
    let x_gpu = stream.clone_htod(x)?;
    let indices_gpu = stream.clone_htod(indices)?;
    let weights = projection.weights();

    let mut matvec = stream.alloc_zeros::<bf16>(pairs * n)?;
    s.experts
        .expert_matvec(stream, &x_gpu, &indices_gpu, &weights, &mut matvec)?;
    let mut runs = Vec::new();
    for _ in 0..2 {
        let mut out = stream.clone_htod(&vec![bf16::from_f32(SENTINEL); pairs * n])?;
        s.experts
            .expert_matmul(stream, &x_gpu, &indices_gpu, &weights, &mut out, scratch)?;
        runs.push(stream.clone_dtoh(&out)?);
    }
    let matvec = stream.clone_dtoh(&matvec)?;
    let got = &runs[0];
    assert!(
        runs[0]
            .iter()
            .zip(&runs[1])
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "{label}: two runs differ"
    );
    assert!(
        !got.iter().any(|v| v.to_f32() == SENTINEL),
        "{label}: some outputs were not written"
    );

    let x64 = to_f64(x);
    let worst_matvec = check_reference(
        &format!("{label} matvec"),
        host,
        &x64,
        indices,
        inputs_per_row,
        &matvec,
    );
    let worst = check_reference(label, host, &x64, indices, inputs_per_row, got);
    let mut ulps = [0usize; 4];
    let mut max_ulps = 0;
    for (a, b) in got.iter().zip(&matvec) {
        let d = common::bf16_ulps(*a, *b);
        ulps[(d as usize).min(3)] += 1;
        max_ulps = max_ulps.max(d);
    }
    eprintln!(
        "{label}: worst error {worst:.3} of bound (matvec {worst_matvec:.3}); against matvec, \
         {} equal, {} at 1 ulp, {} at 2, {} beyond (max {max_ulps}) of {}",
        ulps[0],
        ulps[1],
        ulps[2],
        ulps[3],
        got.len()
    );
    Ok(())
}

fn check_synthetic(fp4: bool) -> anyhow::Result<()> {
    let s = Setup::new()?;
    let mut scratch = ExpertScratch::new(s.stream(), 2048, EXPERTS)?;
    // Two and a half column tiles: 128 + 128 + 64.
    let n = 320;
    let format = if fp4 { "nvfp4" } else { "mxfp8" };
    for (name, k, inputs_per_row) in [("w13", HIDDEN, 1), ("w2", INTERMEDIATE, SLOTS)] {
        let projection = Synthetic::new(&s, fp4, n, k, 7 + k as u32)?;
        let host = projection.host();
        let cases = [
            (1, Routing::Uniform),
            (7, Routing::Uniform),
            (64, Routing::Subset),
            (512, Routing::Uniform),
            (512, Routing::Skewed),
            (1024, Routing::Uniform),
            (1024, Routing::Same),
            (2048, Routing::Skewed),
        ];
        for (rows, kind) in cases {
            let indices = routing(rows, kind, rows as u32 * 31 + 5);
            let x = common::pseudo_random_bf16(rows * inputs_per_row * k, rows as u32, 1.0, 0.0);
            compare_kernels(
                &s,
                &mut scratch,
                &format!("{format} {name} rows {rows} {kind:?}"),
                &projection.gpu,
                &host,
                &x,
                &indices,
                inputs_per_row,
            )?;
        }
    }
    Ok(())
}

#[test]
fn matmul_nvfp4_matches_reference() -> anyhow::Result<()> {
    check_synthetic(true)
}

#[test]
fn matmul_mxfp8_matches_reference() -> anyhow::Result<()> {
    check_synthetic(false)
}

/// The MLX fixtures hold real checkpoint slices; checks `expert_matmul` on them.
#[test]
fn matmul_matches_reference_on_fixtures() -> anyhow::Result<()> {
    let s = Setup::new()?;
    for format in ["nvfp4", "mxfp8"] {
        let file = format!("experts_{format}");
        let fixture = Fixture::load(&file)?;
        let (indices, _) = fixture.i32("indices")?;
        let mut scratch = ExpertScratch::new(s.stream(), indices.len() / SLOTS, 10)?;
        for name in ["gate_up", "down"] {
            let projection = Projection::load(&s, &fixture, &file, name)?;
            let weight = fixture_bytes(&file, &format!("{name}.weight"))?;
            let scales = fixture_bytes(&file, &format!("{name}.scales"))?;
            let host = HostProjection {
                weight: &weight,
                scales: &scales,
                global_scale: match format {
                    "nvfp4" => Some(fixture.f32(&format!("{name}.global_scale"))?.0),
                    _ => None,
                },
                n: projection.n,
                k: projection.k,
            };
            let (x, x_shape) = fixture.bf16(&format!("{name}.x"))?;
            let inputs_per_row = if x_shape.len() == 3 { SLOTS } else { 1 };
            compare_kernels(
                &s,
                &mut scratch,
                &format!("fixture {format} {name}"),
                &projection,
                &host,
                &x,
                &indices,
                inputs_per_row,
            )?;
        }
    }
    Ok(())
}

#[test]
fn matmul_unknown_expert_gives_nan() -> anyhow::Result<()> {
    let s = Setup::new()?;
    let file = "experts_nvfp4";
    let fixture = Fixture::load(file)?;
    let projection = Projection::load(&s, &fixture, file, "gate_up")?;
    let mut scratch = ExpertScratch::new(s.stream(), 2, 10)?;
    // The fixture keeps 10 experts, so ids -1, 10 and 99 are outside it.
    let ids = [0, 1, 2, -1, 4, 5, 10, 7, 3, 99, 3, 3, 9, 8, 6, 5];
    let indices = s.stream().clone_htod(&ids)?;
    let (x, _) = fixture.bf16("gate_up.x")?;
    let x = s.stream().clone_htod(&x[..2 * projection.k])?;
    let mut out = s.stream().alloc_zeros::<bf16>(ids.len() * projection.n)?;
    s.experts.expert_matmul(
        s.stream(),
        &x,
        &indices,
        &projection.weights(),
        &mut out,
        &mut scratch,
    )?;
    let got = s.stream().clone_dtoh(&out)?;
    for (pair, row) in got.chunks(projection.n).enumerate() {
        let bad = !(0..10).contains(&ids[pair]);
        assert!(
            row.iter().all(|v| v.is_nan() == bad),
            "pair {pair}: expected {} values",
            if bad { "NaN" } else { "finite" }
        );
    }
    Ok(())
}

/// One grouping serves both projections: `expert_matmul_grouped` after one `expert_group`
/// matches `expert_matmul` bit for bit on each.
#[test]
fn matmul_grouped_reuses_grouping() -> anyhow::Result<()> {
    let s = Setup::new()?;
    let stream = s.stream();
    let rows = 300;
    let n = 192;
    let mut scratch = ExpertScratch::new(stream, rows, EXPERTS)?;
    let mut ids = routing(rows, Routing::Skewed, 17);
    // Unknown ids still give NaN rows in each projection.
    ids[5] = -3;
    ids[700] = 128;
    let indices = stream.clone_htod(&ids)?;
    for fp4 in [true, false] {
        let w13 = Synthetic::new(&s, fp4, n, HIDDEN, 23)?;
        let w2 = Synthetic::new(&s, fp4, n, INTERMEDIATE, 29)?;
        let x13 = stream.clone_htod(&common::pseudo_random_bf16(rows * HIDDEN, 3, 1.0, 0.0))?;
        let x2 = stream.clone_htod(&common::pseudo_random_bf16(
            rows * SLOTS * INTERMEDIATE,
            4,
            1.0,
            0.0,
        ))?;
        let mut separate = Vec::new();
        for (projection, x) in [(&w13, &x13), (&w2, &x2)] {
            let mut out = stream.alloc_zeros::<bf16>(rows * SLOTS * n)?;
            s.experts.expert_matmul(
                stream,
                x,
                &indices,
                &projection.gpu.weights(),
                &mut out,
                &mut scratch,
            )?;
            separate.push(stream.clone_dtoh(&out)?);
        }
        s.experts
            .expert_group(stream, &indices, EXPERTS, &mut scratch)?;
        for ((projection, x), expected) in [(&w13, &x13), (&w2, &x2)].into_iter().zip(&separate) {
            let mut out = stream.alloc_zeros::<bf16>(rows * SLOTS * n)?;
            s.experts.expert_matmul_grouped(
                stream,
                x,
                &projection.gpu.weights(),
                &mut out,
                &scratch,
            )?;
            let got = stream.clone_dtoh(&out)?;
            assert!(
                got.iter()
                    .zip(expected)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "grouped and separate results differ"
            );
            assert!(got[5 * n..6 * n].iter().all(|v| v.is_nan()));
            assert!(got[700 * n..701 * n].iter().all(|v| v.is_nan()));
            assert!(!got[6 * n..700 * n].iter().any(|v| v.is_nan()));
        }
    }
    Ok(())
}

/// `expert_matmul_swiglu_grouped` equals `expert_matmul_grouped` on `w13` followed by
/// `softcapped_swiglu`, bit for bit, unknown ids included.
#[test]
fn matmul_swiglu_matches_separate_kernels() -> anyhow::Result<()> {
    let s = Setup::new()?;
    let stream = s.stream();
    let elementwise = Elementwise::load(s.gpu.context())?;
    let mut scratch = ExpertScratch::new(stream, 2048, EXPERTS)?;
    for fp4 in [true, false] {
        let w13 = Synthetic::new(&s, fp4, 2 * INTERMEDIATE, HIDDEN, 41)?;
        let weights = w13.gpu.weights();
        for (rows, kind, x_scale) in [
            (7, Routing::Uniform, 1.0),
            (300, Routing::Skewed, 8.0),
            (2048, Routing::Uniform, 4.0),
        ] {
            let mut ids = routing(rows, kind, rows as u32 + 3);
            ids[5] = -3;
            ids[rows * SLOTS - 1] = 128;
            let indices = stream.clone_htod(&ids)?;
            let x = stream.clone_htod(&common::pseudo_random_bf16(
                rows * HIDDEN,
                rows as u32,
                x_scale,
                0.0,
            ))?;
            let pairs = rows * SLOTS;
            s.experts
                .expert_group(stream, &indices, EXPERTS, &mut scratch)?;
            let mut gate_up = stream.alloc_zeros::<bf16>(pairs * 2 * INTERMEDIATE)?;
            s.experts
                .expert_matmul_grouped(stream, &x, &weights, &mut gate_up, &scratch)?;
            let mut want = stream.alloc_zeros::<bf16>(pairs * INTERMEDIATE)?;
            elementwise.softcapped_swiglu(stream, &gate_up, &mut want, GATE_SOFTCAP, UP_SOFTCAP)?;
            let mut got =
                stream.clone_htod(&vec![bf16::from_f32(SENTINEL); pairs * INTERMEDIATE])?;
            s.experts.expert_matmul_swiglu_grouped(
                stream,
                &x,
                &weights,
                &mut got,
                &scratch,
                GATE_SOFTCAP,
                UP_SOFTCAP,
            )?;
            let (got, want) = (stream.clone_dtoh(&got)?, stream.clone_dtoh(&want)?);
            let differing = got
                .iter()
                .zip(&want)
                .filter(|(a, b)| a.to_bits() != b.to_bits() && !(a.is_nan() && b.is_nan()))
                .count();
            eprintln!("fp4 {fp4} rows {rows}: {differing} of {} differ", got.len());
            assert_eq!(differing, 0, "fused and separate SwiGLU differ");
            assert!(
                got[5 * INTERMEDIATE..6 * INTERMEDIATE]
                    .iter()
                    .all(|v| v.is_nan())
            );
            assert!(got[(pairs - 1) * INTERMEDIATE..].iter().all(|v| v.is_nan()));
            assert!(
                !got[6 * INTERMEDIATE..(pairs - 1) * INTERMEDIATE]
                    .iter()
                    .any(|v| v.is_nan())
            );
        }
    }
    Ok(())
}

/// `expert_matmul_grouped` refuses a scratch without a grouping and weights whose expert
/// count differs from the grouping's.
#[test]
fn matmul_grouped_checks_grouping() -> anyhow::Result<()> {
    let s = Setup::new()?;
    let stream = s.stream();
    let file = "experts_nvfp4";
    let fixture = Fixture::load(file)?;
    let projection = Projection::load(&s, &fixture, file, "gate_up")?;
    let mut scratch = ExpertScratch::new(stream, 2, EXPERTS)?;
    let (x, _) = fixture.bf16("gate_up.x")?;
    let x = stream.clone_htod(&x[..2 * projection.k])?;
    let mut out = stream.alloc_zeros::<bf16>(2 * SLOTS * projection.n)?;
    let weights = projection.weights();
    assert!(
        s.experts
            .expert_matmul_grouped(stream, &x, &weights, &mut out, &scratch)
            .is_err()
    );
    // The fixture keeps 10 experts; group for 128.
    let indices = stream.clone_htod(&[0i32; 2 * SLOTS])?;
    s.experts
        .expert_group(stream, &indices, EXPERTS, &mut scratch)?;
    assert!(
        s.experts
            .expert_matmul_grouped(stream, &x, &weights, &mut out, &scratch)
            .is_err()
    );
    s.experts.expert_group(stream, &indices, 10, &mut scratch)?;
    s.experts
        .expert_matmul_grouped(stream, &x, &weights, &mut out, &scratch)?;
    Ok(())
}

/// Full projections of layers 0 (NVFP4) and 20 (MXFP8) from the checkpoint, 512 rows.
#[test]
#[ignore = "reads the original checkpoint from SIPHON_MODEL or ~/.local/share/inkling-mini/model"]
fn matmul_matches_reference_on_checkpoint() -> anyhow::Result<()> {
    let s = Setup::new()?;
    let checkpoint = Checkpoint::open(model_dir()?)?;
    let rows = 512;
    let mut scratch = ExpertScratch::new(s.stream(), rows, EXPERTS)?;
    for layer in [0, 20] {
        for (which, k, inputs_per_row) in [("w13", HIDDEN, 1), ("w2", INTERMEDIATE, SLOTS)] {
            let projection = checkpoint_projection(s.stream(), &checkpoint, layer, which, k)?;
            let prefix = format!("model.llm.layers.{layer}.mlp.experts.{which}_weight");
            let bytes = |suffix: &str| {
                let name = format!("{prefix}{suffix}");
                let entry = checkpoint
                    .get(&name)
                    .with_context(|| format!("{name} missing from checkpoint"))?;
                anyhow::Ok(checkpoint.bytes(entry))
            };
            let host = HostProjection {
                weight: bytes("")?,
                scales: bytes("_scale")?,
                global_scale: projection
                    .global_scale
                    .as_ref()
                    .map(|g| s.stream().clone_dtoh(g))
                    .transpose()?,
                n: HIDDEN,
                k,
            };
            let indices = routing(rows, Routing::Uniform, layer as u32 + 1);
            // Hidden states after RMS norm are about unit size; expert activations smaller.
            let x = common::pseudo_random_bf16(rows * inputs_per_row * k, layer as u32, 2.0, 0.0);
            compare_kernels(
                &s,
                &mut scratch,
                &format!("layer {layer} {which}"),
                &projection,
                &host,
                &x,
                &indices,
                inputs_per_row,
            )?;
        }
    }
    Ok(())
}

/// Times `expert_matvec` and `expert_matmul` at the model's shapes with random weights.
#[test]
#[ignore = "benchmark; run alone on an idle GPU with --nocapture"]
fn bench_expert_matmul() -> anyhow::Result<()> {
    let s = Setup::new()?;
    let stream = s.stream();
    let mut scratch = ExpertScratch::new(stream, 2048, EXPERTS)?;
    let timing = Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT);
    // Runs only the cases whose label (such as "NVFP4 w13 Uniform 512") contains this.
    let filter = std::env::var("EXPERT_BENCH_FILTER").unwrap_or_default();
    println!("| format | projection | routing | rows | matvec (us) | matmul (us) | speedup |");
    println!("| --- | --- | --- | ---: | ---: | ---: | ---: |");
    for fp4 in [true, false] {
        for (name, k, inputs_per_row) in [("w13", HIDDEN, 1), ("w2", INTERMEDIATE, SLOTS)] {
            let projection = Synthetic::new(&s, fp4, HIDDEN, k, 3)?;
            let weights = projection.gpu.weights();
            for kind in [Routing::Uniform, Routing::Skewed] {
                for rows in [1, 16, 64, 512, 1024, 2048] {
                    let format = if fp4 { "NVFP4" } else { "MXFP8" };
                    let label = format!("{format} {name} {kind:?} {rows}");
                    if !label.contains(&filter) {
                        continue;
                    }
                    let indices = stream.clone_htod(&routing(rows, kind, 11))?;
                    let x = stream.clone_htod(&common::pseudo_random_bf16(
                        rows * inputs_per_row * k,
                        1,
                        1.0,
                        0.0,
                    ))?;
                    let mut out = stream.alloc_zeros::<bf16>(rows * SLOTS * HIDDEN)?;
                    let mut time = |matmul: bool| -> anyhow::Result<f64> {
                        let mut call = |out: &mut CudaSlice<bf16>| {
                            if matmul {
                                s.experts.expert_matmul(
                                    stream,
                                    &x,
                                    &indices,
                                    &weights,
                                    out,
                                    &mut scratch,
                                )
                            } else {
                                s.experts.expert_matvec(stream, &x, &indices, &weights, out)
                            }
                        };
                        // Run for about 100 ms first so the clocks settle, then report the
                        // best of 5 repetitions.
                        let mut iterations = 1;
                        loop {
                            let start = stream.record_event(timing)?;
                            for _ in 0..iterations {
                                call(&mut out)?;
                            }
                            let end = stream.record_event(timing)?;
                            end.synchronize()?;
                            if start.elapsed_ms(&end)? > 100.0 {
                                break;
                            }
                            iterations *= 2;
                        }
                        let iterations = (iterations / 4).max(1);
                        let mut best = f64::INFINITY;
                        for _ in 0..5 {
                            let start = stream.record_event(timing)?;
                            for _ in 0..iterations {
                                call(&mut out)?;
                            }
                            let end = stream.record_event(timing)?;
                            end.synchronize()?;
                            let us =
                                f64::from(start.elapsed_ms(&end)?) * 1000.0 / f64::from(iterations);
                            best = best.min(us);
                        }
                        Ok(best)
                    };
                    let matvec = time(false)?;
                    let matmul = time(true)?;
                    println!(
                        "| {format} | {name} | {kind:?} | {rows} | {matvec:.1} | {matmul:.1} | {:.2} |",
                        matvec / matmul
                    );
                }
            }
        }
    }
    Ok(())
}

// Quantized activations: `quantize_rows` against a CPU model of vLLM's Inkling activation
// quantizers (`vllm/models/inkling2/nvidia/ops/nvfp4.py` and `mxfp8.py`), bit for bit, and
// the block-scaled products against the F64 reference on the dequantized inputs.

/// Positive finite E4M3 values by code, 0 to 0x7e.
fn e4m3_table() -> Vec<f64> {
    (0..=0x7eu8).map(e4m3).collect()
}

/// The value of E4M3 bits, exact in F32.
fn e4m3_f32(bits: u8) -> f32 {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "E4M3 values are exact in F32"
    )]
    let value = e4m3(bits) as f32;
    value
}

/// E4M3 bits of `x` rounded to nearest even, saturating at 448 (`cvt.rn.satfinite`).
fn to_e4m3(table: &[f64], x: f32) -> u8 {
    if x.is_nan() {
        return 0x7f;
    }
    let sign = if x.is_sign_negative() { 0x80 } else { 0 };
    let a = f64::from(x.abs());
    let above = table.partition_point(|&v| v < a).min(table.len() - 1);
    let below = above.saturating_sub(1);
    let (d_above, d_below) = (table[above] - a, a - table[below]);
    let code = match d_below.total_cmp(&d_above) {
        std::cmp::Ordering::Less => below,
        std::cmp::Ordering::Equal if below % 2 == 0 => below,
        _ => above,
    };
    sign | u8::try_from(code).expect("E4M3 codes fit in u8")
}

/// The E2M1 code (sign in bit 3) of `x` rounded to nearest even with saturation, and its
/// value: vLLM's `_e2m1`.
fn to_e2m1(x: f32) -> (u8, f32) {
    let a = x.abs();
    let code: u8 = if a <= 0.25 {
        0
    } else if a < 0.75 {
        1
    } else if a <= 1.25 {
        2
    } else if a < 1.75 {
        3
    } else if a <= 2.5 {
        4
    } else if a < 3.5 {
        5
    } else if a <= 5.0 {
        6
    } else {
        7
    };
    let value = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0][usize::from(code)];
    if x.is_sign_negative() {
        (code | 8, -value)
    } else {
        (code, value)
    }
}

/// One row quantized as the GPU stores it: values, block scales and outer scale.
struct QuantizedRow {
    values: Vec<u8>,
    scales: Vec<u8>,
    outer: f32,
}

/// One 16-value NVFP4 block candidate: its 8 bytes and its error, summed pairwise.
fn fp4_candidate(x: &[f32], scale: f32, inverse: f32, outer: f32) -> ([u8; 8], f32) {
    let mut bytes = [0u8; 8];
    let mut error = [0f32; 16];
    for i in 0..16 {
        let (code, value) = to_e2m1(x[i] * inverse);
        bytes[i / 2] |= code << (4 * (i % 2));
        error[i] = ((value * scale) * outer - x[i]).abs();
    }
    let mut width = 8;
    while width > 0 {
        for i in 0..width {
            error[i] = error[2 * i] + error[2 * i + 1];
        }
        width /= 2;
    }
    (bytes, error[0])
}

/// vLLM's `_quantize_row`: per-row amax, 4-over-6 block scales chosen by absolute error.
fn quantize_nvfp4(table: &[f64], x: &[f32]) -> QuantizedRow {
    let k = x.len();
    let amax = x.iter().fold(0f32, |m, v| m.max(v.abs()));
    let encoder = (1536.0 / amax).min(f32::MAX);
    let decoder = 1.0 / encoder;
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the F32 constants vLLM uses"
    )]
    let (sixth, inv_1536) = ((1.0f64 / 6.0) as f32, (1.0f64 / 1536.0) as f32);
    let outer = amax * inv_1536;
    let (encoder6, encoder4) = (encoder * sixth, encoder * 0.25);
    let mut values = vec![0u8; k / 2];
    let mut block_scales = vec![0u8; k / 16];
    for (b, block) in x.chunks(16).enumerate() {
        let block_max = block.iter().fold(0f32, |m, v| m.max(v.abs()));
        let round = |v: f32| e4m3_f32(to_e4m3(table, v));
        let scale6 = round((block_max * encoder6).min(448.0));
        let scale4 = round((block_max * encoder4).min(448.0));
        let inverse6 = (1.0 / (scale6 * decoder)).min(f32::MAX);
        let inverse4 = (1.0 / (scale4 * decoder)).min(f32::MAX);
        let (bytes6, error6) = fp4_candidate(block, scale6, inverse6, outer);
        let (bytes4, error4) = fp4_candidate(block, scale4, inverse4, outer);
        let (bytes, scale) = if error4 < error6 {
            (bytes4, scale4)
        } else {
            (bytes6, scale6)
        };
        block_scales[b] = to_e4m3(table, scale);
        values[8 * b..8 * b + 8].copy_from_slice(&bytes);
    }
    QuantizedRow {
        values,
        scales: block_scales,
        outer,
    }
}

/// vLLM's `_mxfp8_block` on each 32-value block.
fn quantize_mxfp8(table: &[f64], x: &[f32]) -> QuantizedRow {
    let k = x.len();
    let mut values = vec![0u8; k];
    let mut scales = vec![0u8; k / 32];
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the F32 constant vLLM uses"
    )]
    let inv_448 = (1.0f64 / 448.0) as f32;
    let min_scale = 2f32.powi(-127);
    for (b, block) in x.chunks(32).enumerate() {
        let amax = block.iter().fold(0f32, |m, v| m.max(v.abs()));
        let raw = amax * inv_448;
        let mut exponent = (raw.to_bits().wrapping_add(0x007f_ffff) & 0x7f80_0000) >> 23;
        exponent = if raw <= min_scale {
            0
        } else {
            exponent.min(254)
        };
        let scale = f32::from_bits(exponent << 23).max(min_scale);
        let inverse = if exponent == 0 {
            2f32.powi(127)
        } else {
            1.0 / scale
        };
        scales[b] = u8::try_from(exponent).expect("exponents fit in u8");
        for (i, x) in block.iter().enumerate() {
            let v = (x * inverse).clamp(-448.0, 448.0);
            values[b * 32 + i] = to_e4m3(table, v);
        }
    }
    QuantizedRow {
        values,
        scales,
        outer: 0.0,
    }
}

/// Rows that exercise the quantizers: uniform values at several magnitudes, outliers, a
/// zero row, zero blocks inside a row, tiny and huge values.
fn quantizer_inputs(rows: usize, k: usize, seed: u32) -> Vec<bf16> {
    let mut x = common::pseudo_random(rows * k, seed, 1.0);
    let mut random = SplitMix(u64::from(seed));
    for (r, row) in x.chunks_mut(k).enumerate() {
        let scale = match r % 8 {
            0 => 1.0,
            1 => 1e-3,
            2 => 30.0,
            3 => 0.0,
            4 => 1e-30,
            5 => 1e30,
            _ => 2f32.powi(i32::try_from(random.below(20)).expect("small") - 10),
        };
        for v in row.iter_mut() {
            *v *= scale;
        }
        if r % 8 >= 6 {
            // A few outliers and a zero block.
            for _ in 0..4 {
                row[random.below(k)] *= 64.0;
            }
            let block = random.below(k / 32) * 32;
            row[block..block + 32].fill(0.0);
        }
    }
    x.into_iter().map(bf16::from_f32).collect()
}

/// Runs `quantize_rows` on `x` and returns its values, scales and outer scales.
fn gpu_quantize(
    s: &Setup,
    x: &[bf16],
    k: usize,
    format: ExpertFormat<'_>,
    out: &mut QuantizedRows,
) -> anyhow::Result<(Vec<u8>, Vec<u8>, Vec<f32>)> {
    let stream = s.stream();
    let rows = x.len() / k;
    let x_gpu = stream.clone_htod(x)?;
    s.experts.quantize_rows(stream, &x_gpu, k, format, out)?;
    let fp4 = matches!(format, ExpertFormat::Nvfp4 { .. });
    let (value_bytes, scale_bytes) = if fp4 {
        (rows * k / 2, rows * k / 16)
    } else {
        (rows * k, rows * k / 32)
    };
    Ok((
        stream.clone_dtoh(&out.values().slice(..value_bytes))?,
        stream.clone_dtoh(&out.scales().slice(..scale_bytes))?,
        stream.clone_dtoh(&out.outer().slice(..rows))?,
    ))
}

#[test]
#[cfg_attr(sm89, ignore = "needs the block-scaled MMA, which sm_89 lacks")]
fn quantize_rows_matches_cpu_model() -> anyhow::Result<()> {
    let s = Setup::new()?;
    let table = e4m3_table();
    let dummy = s.stream().clone_htod(&[1f32])?;
    let mut out = QuantizedRows::new(s.stream(), 400 * HIDDEN, 400)?;
    for fp4 in [true, false] {
        let format = if fp4 {
            ExpertFormat::Nvfp4 {
                global_scale: &dummy,
            }
        } else {
            ExpertFormat::Mxfp8
        };
        for (k, rows) in [(HIDDEN, 400), (INTERMEDIATE, 333)] {
            let x = quantizer_inputs(rows, k, u32::try_from(k + rows).expect("small"));
            let (values, scales, outer) = gpu_quantize(&s, &x, k, format, &mut out)?;
            let (row_values, row_scales) = (values.len() / rows, scales.len() / rows);
            let mut chose4 = 0;
            for (r, row) in x.chunks(k).enumerate() {
                let row: Vec<f32> = row.iter().map(|v| v.to_f32()).collect();
                let want = if fp4 {
                    quantize_nvfp4(&table, &row)
                } else {
                    quantize_mxfp8(&table, &row)
                };
                let got_values = &values[r * row_values..(r + 1) * row_values];
                let got_scales = &scales[r * row_scales..(r + 1) * row_scales];
                if let Some(i) = (0..row_values).find(|&i| got_values[i] != want.values[i]) {
                    panic!(
                        "fp4 {fp4} k {k} row {r}: value byte {i} is {:#04x}, want {:#04x}",
                        got_values[i], want.values[i]
                    );
                }
                if let Some(i) = (0..row_scales).find(|&i| got_scales[i] != want.scales[i]) {
                    panic!(
                        "fp4 {fp4} k {k} row {r}: scale {i} is {:#04x}, want {:#04x}",
                        got_scales[i], want.scales[i]
                    );
                }
                if fp4 {
                    assert_eq!(
                        outer[r].to_bits(),
                        want.outer.to_bits(),
                        "k {k} row {r}: outer scale"
                    );
                    // Count blocks where the 4-based scale won, to show the rule is live.
                    let amax = row.iter().fold(0f32, |m, v| m.max(v.abs()));
                    let encoder = (1536.0 / amax).min(f32::MAX);
                    chose4 += row
                        .chunks(16)
                        .zip(&want.scales)
                        .filter(|&(block, &scale)| {
                            let block_max = block.iter().fold(0f32, |m, v| m.max(v.abs()));
                            let round =
                                |v: f32| to_e4m3(&table, (block_max * (encoder * v)).min(448.0));
                            block_max > 0.0 && scale == round(0.25) && scale != round(1.0 / 6.0)
                        })
                        .count();
                }
            }
            let detail = if fp4 {
                format!(
                    "; {chose4} of {} blocks took the 4-based scale",
                    rows * k / 16
                )
            } else {
                String::new()
            };
            let name = if fp4 { "nvfp4" } else { "mxfp8" };
            eprintln!("quantize {name} k {k}: {rows} rows equal the CPU model{detail}");
        }
    }
    Ok(())
}

/// The F64 values of quantized rows (`QuantizedRow` layout), outer scale included.
fn dequantize(row: &QuantizedRow, k: usize, fp4: bool) -> Vec<f64> {
    (0..k)
        .map(|i| {
            if fp4 {
                let byte = row.values[i / 2];
                let nibble = if i % 2 == 0 { byte & 15 } else { byte >> 4 };
                e2m1(nibble) * e4m3(row.scales[i / 16]) * f64::from(row.outer)
            } else {
                e4m3(row.values[i]) * 2f64.powi(i32::from(row.scales[i / 32]) - 127)
            }
        })
        .collect()
}

/// `quantize_rows_fp8` for NVFP4 weights writes the MXFP8 rows of `quantize_rows`, each block
/// of 32 values in the paired order: word w holds values 8t + 4h.. for (h, t) = (w / 4, w % 4).
#[test]
fn quantize_rows_fp8_pairs_mxfp8() -> anyhow::Result<()> {
    let s = Setup::new()?;
    let dummy = s.stream().clone_htod(&[1f32])?;
    let nvfp4 = ExpertFormat::Nvfp4 {
        global_scale: &dummy,
    };
    let mut out = QuantizedRows::new(s.stream(), 300 * HIDDEN, 300)?;
    for (k, rows) in [(HIDDEN, 300), (INTERMEDIATE, 77)] {
        let x = quantizer_inputs(rows, k, 3 + u32::try_from(k).expect("small"));
        let (plain, plain_scales, _) = gpu_quantize(&s, &x, k, ExpertFormat::Mxfp8, &mut out)?;
        let x_gpu = s.stream().clone_htod(&x)?;
        s.experts
            .quantize_rows_fp8(s.stream(), &x_gpu, k, nvfp4, &mut out)?;
        let paired = s.stream().clone_dtoh(&out.values().slice(..rows * k))?;
        let scales = s
            .stream()
            .clone_dtoh(&out.scales().slice(..rows * k / 32))?;
        assert_eq!(scales, plain_scales, "k {k}: scales");
        for (b, (got, want)) in paired.chunks(32).zip(plain.chunks(32)).enumerate() {
            for w in 0..8 {
                let (h, t) = (w / 4, w % 4);
                let from = 8 * t + 4 * h;
                assert_eq!(
                    got[4 * w..4 * w + 4],
                    want[from..from + 4],
                    "k {k} block {b} word {w}"
                );
            }
        }
    }
    Ok(())
}

/// `expert_matmul_quantized_grouped` against the F64 reference on the dequantized inputs,
/// with every output written and two runs equal. With `fp8`, the inputs are MXFP8 for both
/// weight formats (`quantize_rows_fp8`).
fn check_quantized(fp4: bool, fp8: bool) -> anyhow::Result<()> {
    let s = Setup::new()?;
    let stream = s.stream();
    let table = e4m3_table();
    let mut scratch = ExpertScratch::new(stream, 2048, EXPERTS)?;
    let mut xq = QuantizedRows::new(stream, 2048 * SLOTS * INTERMEDIATE, 2048 * SLOTS)?;
    let n = 320;
    let label = match (fp4, fp8) {
        (true, false) => "nvfp4",
        (true, true) => "nvfp4 with fp8 inputs",
        (false, _) => "mxfp8",
    };
    // FP4 inputs only for NVFP4 weights without `fp8`.
    let fp4_inputs = fp4 && !fp8;
    for (name, k, inputs_per_row) in [("w13", HIDDEN, 1), ("w2", INTERMEDIATE, SLOTS)] {
        let projection = Synthetic::new(&s, fp4, n, k, 11 + k as u32)?;
        let host = projection.host();
        let weights = projection.gpu.weights();
        for (rows, kind) in [
            (1, Routing::Uniform),
            (64, Routing::Subset),
            (512, Routing::Skewed),
            (2048, Routing::Uniform),
        ] {
            let indices = routing(rows, kind, rows as u32 * 7 + 1);
            let x = common::pseudo_random_bf16(rows * inputs_per_row * k, rows as u32, 2.0, 0.0);
            let x_gpu = stream.clone_htod(&x)?;
            if fp8 {
                s.experts
                    .quantize_rows_fp8(stream, &x_gpu, k, weights.format, &mut xq)?;
            } else {
                s.experts
                    .quantize_rows(stream, &x_gpu, k, weights.format, &mut xq)?;
            }
            let x64: Vec<f64> = x
                .chunks(k)
                .flat_map(|row| {
                    let row: Vec<f32> = row.iter().map(|v| v.to_f32()).collect();
                    let q = if fp4_inputs {
                        quantize_nvfp4(&table, &row)
                    } else {
                        quantize_mxfp8(&table, &row)
                    };
                    dequantize(&q, k, fp4_inputs)
                })
                .collect();
            let indices_gpu = stream.clone_htod(&indices)?;
            s.experts
                .expert_group_quantized(stream, &indices_gpu, EXPERTS, &mut scratch)?;
            let mut runs = Vec::new();
            for _ in 0..2 {
                let mut out =
                    stream.clone_htod(&vec![bf16::from_f32(SENTINEL); rows * SLOTS * n])?;
                s.experts
                    .expert_matmul_quantized_grouped(stream, &xq, &weights, &mut out, &scratch)?;
                runs.push(stream.clone_dtoh(&out)?);
            }
            let got = &runs[0];
            assert!(
                runs[0]
                    .iter()
                    .zip(&runs[1])
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "two runs differ"
            );
            assert!(!got.iter().any(|v| v.to_f32() == SENTINEL));
            let case = format!("quantized {label} {name} rows {rows} {kind:?}");
            let worst = check_reference(&case, &host, &x64, &indices, inputs_per_row, got);
            eprintln!("{case}: worst error {worst:.3} of bound");
        }
    }
    Ok(())
}

#[test]
#[cfg_attr(sm89, ignore = "needs the block-scaled MMA, which sm_89 lacks")]
fn matmul_quantized_nvfp4_matches_reference() -> anyhow::Result<()> {
    check_quantized(true, false)
}

#[test]
#[cfg_attr(sm89, ignore = "needs the block-scaled MMA, which sm_89 lacks")]
fn matmul_quantized_mxfp8_matches_reference() -> anyhow::Result<()> {
    check_quantized(false, false)
}

#[test]
#[cfg_attr(sm89, ignore = "needs the block-scaled MMA, which sm_89 lacks")]
fn matmul_fp8_nvfp4_matches_reference() -> anyhow::Result<()> {
    check_quantized(true, true)
}

/// The host model of `nvfp4_scale_spread`'s rule: whether `expert_gemm_q8_*` multiplies a
/// block with scale bytes `a` and `b` exactly.
fn fp8_exact_block(a: u8, b: u8) -> bool {
    let field = |s: u8| u32::from((s & 0x7f) >> 3).max(1);
    let top = field(a).max(field(b));
    [a, b].iter().all(|&s| {
        let s = s & 0x7f;
        let m = u32::from(s & 7) | if s >= 8 { 8 } else { 0 };
        let gap = top - field(s);
        m == 0 || (gap <= 10 && (m << 7) >= (1 << gap))
    })
}

/// The FP8 NVFP4 products stay exact at the edge of their range: block scales up to 2^10
/// apart, subnormal and zero scales; and `fp8_inexact_scales` counts the scales beyond it.
#[test]
#[cfg_attr(sm89, ignore = "needs the block-scaled MMA, which sm_89 lacks")]
fn matmul_fp8_nvfp4_exact_at_wide_scale_gaps() -> anyhow::Result<()> {
    let s = Setup::new()?;
    let stream = s.stream();
    let table = e4m3_table();
    let (n, k, rows) = (320, HIDDEN, 512);
    let mut projection = Synthetic::new(&s, true, n, k, 77)?;
    let mut random = SplitMix(5);
    for block in projection.scales.chunks_mut(2) {
        let pair = loop {
            let top = 11 + u8::try_from(random.below(5)).expect("small");
            let gap = u8::try_from(random.below(11)).expect("small");
            let larger = (top << 3) | (random.byte() % 8);
            let smaller = match random.below(8) {
                0 => 0,
                1 => random.byte() % 8,
                _ => ((top - gap).max(1) << 3) | (random.byte() % 8),
            };
            if larger != 0x7f && fp8_exact_block(larger, smaller) {
                break if random.below(2) == 0 {
                    [larger, smaller]
                } else {
                    [smaller, larger]
                };
            }
        };
        block.copy_from_slice(&pair);
    }
    projection.gpu.scales = stream.clone_htod(&projection.scales)?;
    assert_eq!(
        s.experts
            .fp8_inexact_scales(stream, &projection.gpu.scales)?,
        0
    );
    let host = projection.host();
    let weights = projection.gpu.weights();
    let mut scratch = ExpertScratch::new(stream, rows, EXPERTS)?;
    let mut xq = QuantizedRows::new(stream, rows * k, rows)?;
    let indices = routing(rows, Routing::Uniform, 3);
    let x = common::pseudo_random_bf16(rows * k, 8, 2.0, 0.0);
    s.experts
        .quantize_rows_fp8(stream, &stream.clone_htod(&x)?, k, weights.format, &mut xq)?;
    let x64: Vec<f64> = x
        .chunks(k)
        .flat_map(|row| {
            let row: Vec<f32> = row.iter().map(|v| v.to_f32()).collect();
            dequantize(&quantize_mxfp8(&table, &row), k, false)
        })
        .collect();
    s.experts.expert_group_quantized(
        stream,
        &stream.clone_htod(&indices)?,
        EXPERTS,
        &mut scratch,
    )?;
    let mut out = stream.alloc_zeros::<bf16>(rows * SLOTS * n)?;
    s.experts
        .expert_matmul_quantized_grouped(stream, &xq, &weights, &mut out, &scratch)?;
    let got = stream.clone_dtoh(&out)?;
    let worst = check_reference("fp8 wide scale gaps", &host, &x64, &indices, 1, &got);
    eprintln!("fp8 wide scale gaps: worst error {worst:.3} of bound");
    // A gap of 11 is one too many.
    projection.scales[..2].copy_from_slice(&[12 << 3, 1 << 3]);
    projection.gpu.scales = stream.clone_htod(&projection.scales)?;
    assert_eq!(
        s.experts
            .fp8_inexact_scales(stream, &projection.gpu.scales)?,
        1
    );
    Ok(())
}

/// `expert_matmul_quantized_swiglu_grouped` equals `expert_matmul_quantized_grouped`
/// followed by `softcapped_swiglu`, bit for bit, unknown ids included.
#[test]
#[cfg_attr(sm89, ignore = "needs the block-scaled MMA, which sm_89 lacks")]
fn matmul_quantized_swiglu_matches_separate_kernels() -> anyhow::Result<()> {
    let s = Setup::new()?;
    let stream = s.stream();
    let elementwise = Elementwise::load(s.gpu.context())?;
    let rows = 700;
    let pairs = rows * SLOTS;
    let mut scratch = ExpertScratch::new(stream, rows, EXPERTS)?;
    let mut xq = QuantizedRows::new(stream, rows * HIDDEN, rows)?;
    let mut ids = routing(rows, Routing::Skewed, 5);
    ids[9] = -1;
    let indices = stream.clone_htod(&ids)?;
    let x = stream.clone_htod(&common::pseudo_random_bf16(rows * HIDDEN, 9, 4.0, 0.0))?;
    s.experts
        .expert_group_quantized(stream, &indices, EXPERTS, &mut scratch)?;
    for (fp4, fp8) in [(true, false), (false, false), (true, true)] {
        let w13 = Synthetic::new(&s, fp4, 2 * INTERMEDIATE, HIDDEN, 43)?;
        let weights = w13.gpu.weights();
        if fp8 {
            s.experts
                .quantize_rows_fp8(stream, &x, HIDDEN, weights.format, &mut xq)?;
        } else {
            s.experts
                .quantize_rows(stream, &x, HIDDEN, weights.format, &mut xq)?;
        }
        let mut gate_up = stream.alloc_zeros::<bf16>(pairs * 2 * INTERMEDIATE)?;
        s.experts
            .expert_matmul_quantized_grouped(stream, &xq, &weights, &mut gate_up, &scratch)?;
        let mut want = stream.alloc_zeros::<bf16>(pairs * INTERMEDIATE)?;
        elementwise.softcapped_swiglu(stream, &gate_up, &mut want, GATE_SOFTCAP, UP_SOFTCAP)?;
        let mut got = stream.alloc_zeros::<bf16>(pairs * INTERMEDIATE)?;
        s.experts.expert_matmul_quantized_swiglu_grouped(
            stream,
            &xq,
            &weights,
            &mut got,
            &scratch,
            GATE_SOFTCAP,
            UP_SOFTCAP,
        )?;
        let (got, want) = (stream.clone_dtoh(&got)?, stream.clone_dtoh(&want)?);
        let differing = got
            .iter()
            .zip(&want)
            .filter(|(a, b)| a.to_bits() != b.to_bits() && !(a.is_nan() && b.is_nan()))
            .count();
        assert_eq!(
            differing, 0,
            "fp4 {fp4} fp8 {fp8}: fused and separate SwiGLU differ"
        );
        assert!(
            got[9 * INTERMEDIATE..10 * INTERMEDIATE]
                .iter()
                .all(|v| v.is_nan())
        );
    }
    Ok(())
}

/// Times one 2048-row chunk's expert products with each activation format: BF16, quantized
/// (NVFP4 or MXFP8 inputs) and FP8 (MXFP8 inputs), each with its quantization.
#[test]
#[ignore = "benchmark; run alone on an idle GPU with --nocapture"]
fn bench_expert_activations() -> anyhow::Result<()> {
    if !siphon_cuda::BLOCK_SCALED_MMA {
        eprintln!("skipped: the quantized products need the block-scaled MMA, which sm_89 lacks");
        return Ok(());
    }
    let s = Setup::new()?;
    let stream = s.stream();
    let rows = 2048;
    let mut scratch = ExpertScratch::new(stream, rows, EXPERTS)?;
    let mut xq = QuantizedRows::new(stream, rows * SLOTS * INTERMEDIATE, rows * SLOTS)?;
    let timing = Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT);
    let indices = stream.clone_htod(&routing(rows, Routing::Uniform, 11))?;
    println!("| format | projection | bf16 (us) | quantized (us) | fp8 (us) |");
    println!("| --- | --- | ---: | ---: | ---: |");
    for fp4 in [true, false] {
        for (name, k, inputs_per_row) in [("w13", HIDDEN, 1), ("w2", INTERMEDIATE, SLOTS)] {
            let projection = Synthetic::new(&s, fp4, HIDDEN, k, 3)?;
            let weights = projection.gpu.weights();
            let x = stream.clone_htod(&common::pseudo_random_bf16(
                rows * inputs_per_row * k,
                1,
                1.0,
                0.0,
            ))?;
            let mut out = stream.alloc_zeros::<bf16>(rows * SLOTS * HIDDEN)?;
            let mut time = |mode: usize| -> anyhow::Result<f64> {
                if mode == 0 {
                    s.experts
                        .expert_group(stream, &indices, EXPERTS, &mut scratch)?;
                } else {
                    s.experts
                        .expert_group_quantized(stream, &indices, EXPERTS, &mut scratch)?;
                }
                let mut call = |out: &mut CudaSlice<bf16>| -> anyhow::Result<()> {
                    if mode == 0 {
                        return s
                            .experts
                            .expert_matmul_grouped(stream, &x, &weights, out, &scratch);
                    }
                    if mode == 1 {
                        s.experts
                            .quantize_rows(stream, &x, k, weights.format, &mut xq)?;
                    } else {
                        s.experts
                            .quantize_rows_fp8(stream, &x, k, weights.format, &mut xq)?;
                    }
                    s.experts
                        .expert_matmul_quantized_grouped(stream, &xq, &weights, out, &scratch)
                };
                let mut best = f64::INFINITY;
                for repeat in 0..8 {
                    let start = stream.record_event(timing)?;
                    for _ in 0..20 {
                        call(&mut out)?;
                    }
                    let end = stream.record_event(timing)?;
                    end.synchronize()?;
                    if repeat > 1 {
                        best = best.min(f64::from(start.elapsed_ms(&end)?) * 1000.0 / 20.0);
                    }
                }
                Ok(best)
            };
            let format = if fp4 { "NVFP4" } else { "MXFP8" };
            let (bf16_us, quantized_us, fp8_us) = (time(0)?, time(1)?, time(2)?);
            println!("| {format} | {name} | {bf16_us:.1} | {quantized_us:.1} | {fp8_us:.1} |");
        }
    }
    Ok(())
}
