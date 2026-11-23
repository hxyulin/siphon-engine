//! Expert matrix-vector products against MLX outputs from `reference/fixtures/experts.py`.
//!
//! The kernels copy the reduction order of MLX's `fp_gather_qmv`, so every projection must
//! match bit for bit. The whole-MLP test also runs the softcapped SwiGLU, whose tanh and
//! exp can put a BF16 result 1 ulp from Metal's. A differing activation would pass through
//! the down projection, but none differs in this fixture, so the test also requires
//! bit-exact output.

#![cfg(target_os = "linux")]

mod common;

use std::{fs, path::PathBuf, sync::Arc};

use anyhow::{Context as _, ensure};
use common::{Fixture, assert_bf16_close};
use cudarc::driver::{CudaSlice, CudaStream};
use half::bf16;
use inkling::Checkpoint;
use inkling_cuda::{
    Gpu,
    ops::{
        elementwise::Elementwise,
        experts::{ExpertFormat, ExpertWeights, Experts},
    },
};
use safetensors::{Dtype, SafeTensors};

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

/// Runs both projections of `experts_<format>` and requires bit-exact output.
fn check_fixture(format: &str) -> anyhow::Result<()> {
    let s = Setup::new()?;
    let file = format!("experts_{format}");
    let fixture = Fixture::load(&file)?;
    let (indices, _) = fixture.i32("indices")?;
    let indices = s.stream().clone_htod(&indices)?;
    for name in ["gate_up", "down"] {
        let projection = Projection::load(&s, &fixture, &file, name)?;
        let (x, _) = fixture.bf16(&format!("{name}.x"))?;
        let (want, _) = fixture.bf16(&format!("{name}.out"))?;
        let x = s.stream().clone_htod(&x)?;
        let mut out = s.stream().alloc_zeros::<bf16>(want.len())?;
        s.experts
            .expert_matvec(s.stream(), &x, &indices, &projection.weights(), &mut out)?;
        let got = s.stream().clone_dtoh(&out)?;
        assert_bf16_close(&format!("{format} {name}"), &got, &want, 0);
    }
    Ok(())
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

/// Directory of the original checkpoint: `INKLING_MODEL`, or the GPU host's usual location.
fn model_dir() -> anyhow::Result<PathBuf> {
    if let Some(dir) = std::env::var_os("INKLING_MODEL") {
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

#[test]
#[ignore = "reads the original checkpoint from INKLING_MODEL or ~/.local/share/inkling-mini/model"]
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
        let mut gate_up = stream.alloc_zeros::<bf16>(SLOTS * 2 * INTERMEDIATE)?;
        s.experts
            .expert_matvec(stream, &x, &indices, &w13.weights(), &mut gate_up)?;
        let mut act = stream.alloc_zeros::<bf16>(SLOTS * INTERMEDIATE)?;
        elementwise.softcapped_swiglu(stream, &gate_up, &mut act, GATE_SOFTCAP, UP_SOFTCAP)?;
        let mut out = stream.alloc_zeros::<bf16>(SLOTS * HIDDEN)?;
        s.experts
            .expert_matvec(stream, &act, &indices, &w2.weights(), &mut out)?;

        let got_gate_up = stream.clone_dtoh(&gate_up)?;
        assert_bf16_close(
            &format!("layer {layer} gate_up"),
            &got_gate_up,
            &want_gate_up,
            0,
        );
        let got = stream.clone_dtoh(&out)?;
        assert_bf16_close(&format!("layer {layer} out"), &got, &want, 0);
    }
    Ok(())
}
