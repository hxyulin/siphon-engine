//! Message-gate combine, casts and argmax against MLX outputs from `reference/fixtures/misc.py`.
//!
//! Casts, argmax and every message-gate step before the combine are bit-exact. The combine
//! uses CUDA's tanh, which rounds differently from Metal's `precise::tanh`, so its output is
//! checked against a bound on the error that a few-ulp tanh difference can cause.

#![cfg(target_os = "linux")]

mod common;

use common::{Fixture, assert_bf16_close, assert_f32_close, pseudo_random, pseudo_random_bf16};
use cudarc::driver::{CudaSlice, CudaStream};
use half::bf16;
use siphon_cuda::{
    Gpu,
    ops::{
        elementwise::Elementwise,
        linear::{Gemv, Linear, Start},
        linear_fp8::{Fp8Matrix, LinearFp8},
        misc::{ArgmaxScratch, DepthGateSlot, GATE_SLOTS, GateSlot, GateStep, Misc},
        norm::Norm,
        routing::{DepthStep, NormDest, Routing, StepPartial},
    },
};

const HIDDEN: usize = 2304;
const GATE_RANK: usize = 64;
const EPS: f32 = 1e-6;
/// Allowed combine error, as a multiple of `|h| + |m|`. Metal's tanh is up to 3 F32 ulps from
/// the true value, which is at most 3 * 2^-24 for a result below 1.
const COMBINE_TOLERANCE: f32 = 1.0 / (1 << 21) as f32;
/// Allowed difference between the chunked and single-row gate outputs, as a multiple of
/// `|h| + |m|`: the dots of 64 products, summed in two orders, differ by a few F32 ulps of
/// their magnitude, and tanh' <= 1 carries that into each factor.
const GATE_ROWS_TOLERANCE: f32 = 1.0 / (1 << 18) as f32;

struct Setup {
    fixture: Fixture,
    gpu: Gpu,
    misc: Misc,
}

impl Setup {
    fn new() -> anyhow::Result<Self> {
        let gpu = Gpu::new(0)?;
        let misc = Misc::load(gpu.context())?;
        Ok(Self {
            fixture: Fixture::load("misc")?,
            gpu,
            misc,
        })
    }

    fn stream(&self) -> &std::sync::Arc<CudaStream> {
        self.gpu.stream()
    }

    fn f32(&self, key: &str) -> anyhow::Result<CudaSlice<f32>> {
        let (host, _) = self.fixture.f32(key)?;
        Ok(self.stream().clone_htod(&host)?)
    }
}

/// Fails unless `got` equals `want` bit for bit, NaNs included.
fn assert_f32_exact(label: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{label}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            g.to_bits() == w.to_bits(),
            "{label}[{i}]: got {g} ({:#010x}), want {w} ({:#010x})",
            g.to_bits(),
            w.to_bits()
        );
    }
    eprintln!("{label}: all {} elements bit-exact", want.len());
}

/// Fails if any combine output is off by more than `COMBINE_TOLERANCE * (|h| + |m|)`.
fn assert_combine_close(label: &str, got: &[f32], want: &[f32], h: &[f32], m: &[f32]) {
    assert_eq!(got.len(), want.len(), "{label}: length");
    let mut worst = 0.0f32;
    let mut differing = 0;
    for i in 0..want.len() {
        let scale = h[i].abs() + m[i].abs();
        let err = (got[i] - want[i]).abs();
        assert!(
            err <= COMBINE_TOLERANCE * scale,
            "{label}[{i}]: got {}, want {}, |h| + |m| = {scale}",
            got[i],
            want[i]
        );
        worst = worst.max(err / scale);
        differing += usize::from(got[i].to_bits() != want[i].to_bits());
    }
    eprintln!(
        "{label}: {differing} of {} elements differ; worst error {worst:e} x (|h| + |m|)",
        want.len()
    );
}

#[test]
fn message_gate_combine_matches_mlx() -> anyhow::Result<()> {
    let s = Setup::new()?;
    let (h, _) = s.fixture.f32("combine.h")?;
    let (m, _) = s.fixture.f32("combine.m")?;
    let (want, _) = s.fixture.f32("combine.out")?;
    let mut out = s.stream().alloc_zeros::<f32>(want.len())?;
    s.misc.message_gate_combine(
        s.stream(),
        &s.stream().clone_htod(&h)?,
        &s.stream().clone_htod(&m)?,
        &s.f32("combine.f_pre")?,
        &s.f32("combine.g_pre")?,
        &mut out,
    )?;
    assert_combine_close("combine", &s.stream().clone_dtoh(&out)?, &want, &h, &m);
    Ok(())
}

#[test]
fn cast_f32_to_bf16_matches_mlx() -> anyhow::Result<()> {
    let s = Setup::new()?;
    let (want, _) = s.fixture.bf16("cast_f32_to_bf16.out")?;
    let mut out = s.stream().alloc_zeros::<bf16>(want.len())?;
    s.misc
        .cast_f32_to_bf16(s.stream(), &s.f32("cast_f32_to_bf16.x")?, &mut out)?;
    let got = s.stream().clone_dtoh(&out)?;
    // Compare bits: the inputs include NaNs, and MLX writes one NaN pattern for all of them.
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert!(
            g.to_bits() == w.to_bits(),
            "cast_f32_to_bf16[{i}]: got {:#06x}, want {:#06x}",
            g.to_bits(),
            w.to_bits()
        );
    }
    eprintln!("cast_f32_to_bf16: all {} elements bit-exact", want.len());
    Ok(())
}

#[test]
fn cast_bf16_to_f32_matches_mlx() -> anyhow::Result<()> {
    let s = Setup::new()?;
    let (x, _) = s.fixture.bf16("cast_bf16_to_f32.x")?;
    let (want, _) = s.fixture.f32("cast_bf16_to_f32.out")?;
    let mut out = s.stream().alloc_zeros::<f32>(want.len())?;
    s.misc
        .cast_bf16_to_f32(s.stream(), &s.stream().clone_htod(&x)?, &mut out)?;
    assert_f32_exact("cast_bf16_to_f32", &s.stream().clone_dtoh(&out)?, &want);
    Ok(())
}

#[test]
fn argmax_matches_mlx() -> anyhow::Result<()> {
    let s = Setup::new()?;
    for name in ["argmax.vocab", "argmax.special"] {
        let (x, shape) = s.fixture.f32(&format!("{name}.x"))?;
        let (want, _) = s.fixture.i32(&format!("{name}.out"))?;
        let mut out = s.stream().alloc_zeros::<i32>(want.len())?;
        s.misc
            .argmax_f32(s.stream(), &s.stream().clone_htod(&x)?, &mut out, shape[1])?;
        assert_eq!(s.stream().clone_dtoh(&out)?, want, "{name}");
    }
    Ok(())
}

/// Runs `MessageGate.__call__` for one row with the norm, linear and misc kernels, checking
/// each intermediate against MLX.
#[test]
fn message_gate_chain_matches_mlx() -> anyhow::Result<()> {
    let s = Setup::new()?;
    let stream = s.stream();
    let norm = Norm::load(s.gpu.context())?;
    let linear = Linear::load(s.gpu.context())?;
    let fixture = &s.fixture;

    // 1. Unweighted RMS norm of h and m, side by side in one 4608-wide row.
    let (h, _) = fixture.f32("gate.h")?;
    let (m, _) = fixture.f32("gate.m")?;
    let h_dev = stream.clone_htod(&h)?;
    let m_dev = stream.clone_htod(&m)?;
    let mut h_norm = stream.alloc_zeros::<f32>(HIDDEN)?;
    let mut m_norm = stream.alloc_zeros::<f32>(HIDDEN)?;
    norm.unweighted_rms_norm(stream, &h_dev, &mut h_norm, EPS)?;
    norm.unweighted_rms_norm(stream, &m_dev, &mut m_norm, EPS)?;
    let mut states = stream.alloc_zeros::<f32>(2 * HIDDEN)?;
    stream.memcpy_dtod(&h_norm, &mut states.slice_mut(..HIDDEN))?;
    stream.memcpy_dtod(&m_norm, &mut states.slice_mut(HIDDEN..))?;
    let (want, _) = fixture.f32("gate.states")?;
    assert_f32_exact("gate.states", &stream.clone_dtoh(&states)?, &want);

    // 2-3. Cast to BF16, down projection, cast back to F32.
    let mut states_bf16 = stream.alloc_zeros::<bf16>(2 * HIDDEN)?;
    s.misc.cast_f32_to_bf16(stream, &states, &mut states_bf16)?;
    let (down_weight, _) = fixture.bf16("gate.down_weight")?;
    let mut down = stream.alloc_zeros::<bf16>(GATE_RANK)?;
    let down_weight = stream.clone_htod(&down_weight)?;
    linear.gemv_bf16(
        stream,
        &mut [Gemv::new(&down_weight, &states_bf16, &mut down)],
        Start::Serial,
    )?;
    // The linear kernels sum in another order than MLX; tests/linear.rs bounds the error.
    let (want, _) = fixture.bf16("gate.down")?;
    assert_bf16_close("gate.down", &stream.clone_dtoh(&down)?, &want, 1);
    let mut down_f32 = stream.alloc_zeros::<f32>(GATE_RANK)?;
    s.misc.cast_bf16_to_f32(stream, &down, &mut down_f32)?;

    // 4. Forget and gain projections in one launch, with the BF16 weights MLX widens to F32.
    let (forget, _) = fixture.bf16("gate.forget_weight")?;
    let (gain, _) = fixture.bf16("gate.gain_weight")?;
    let (forget, gain) = (stream.clone_htod(&forget)?, stream.clone_htod(&gain)?);
    let mut f_pre = stream.alloc_zeros::<f32>(HIDDEN)?;
    let mut g_pre = stream.alloc_zeros::<f32>(HIDDEN)?;
    linear.gemv_f32(
        stream,
        &mut [
            Gemv::new(&forget, &down_f32, &mut f_pre),
            Gemv::new(&gain, &down_f32, &mut g_pre),
        ],
        Start::Serial,
    )?;
    let (want, _) = fixture.f32("gate.f_pre")?;
    assert_f32_close("gate.f_pre", &stream.clone_dtoh(&f_pre)?, &want, 1e-5, 1e-6);
    let (want, _) = fixture.f32("gate.g_pre")?;
    assert_f32_close("gate.g_pre", &stream.clone_dtoh(&g_pre)?, &want, 1e-5, 1e-6);

    // 5. Combine.
    let mut out = stream.alloc_zeros::<f32>(HIDDEN)?;
    s.misc
        .message_gate_combine(stream, &h_dev, &m_dev, &f_pre, &g_pre, &mut out)?;
    let (want, _) = fixture.f32("gate.out")?;
    assert_combine_close("gate.out", &stream.clone_dtoh(&out)?, &want, &h, &m);
    Ok(())
}

/// The fused end of `MessageGate`, from MLX's own BF16 down projection, against MLX's gate
/// output. The forget and gain projections sum in MLX's GEMV order, so only tanh differs,
/// within the combine bound above. With the norm, its output is checked against
/// `rms_norm_f32` of the gate output: the same sums, with 1 / sqrt in float rather than a
/// rounded double rsqrt, so BF16 values differ by at most one ulp.
#[test]
fn message_gate_out_matches_mlx() -> anyhow::Result<()> {
    let s = Setup::new()?;
    let stream = s.stream();
    let norm = Norm::load(s.gpu.context())?;
    let fixture = &s.fixture;
    let (h, _) = fixture.f32("gate.h")?;
    let (m, _) = fixture.f32("gate.m")?;
    let (down, _) = fixture.bf16("gate.down")?;
    let (forget, _) = fixture.bf16("gate.forget_weight")?;
    let (gain, _) = fixture.bf16("gate.gain_weight")?;
    let (want, _) = fixture.f32("gate.out")?;
    let h_dev = stream.clone_htod(&h)?;
    let m_dev = stream.clone_htod(&m)?;
    let down = stream.clone_htod(&down)?;
    let (forget, gain) = (stream.clone_htod(&forget)?, stream.clone_htod(&gain)?);

    let mut counter = stream.alloc_zeros::<u32>(GATE_SLOTS)?;
    let mut out = stream.alloc_zeros::<f32>(HIDDEN)?;
    s.misc.message_gate_out(
        stream,
        &down,
        &mut [GateSlot {
            forget: &forget,
            gain: &gain,
            norm: None,
        }],
        &h_dev,
        0,
        &m_dev,
        &mut out,
        &mut counter,
        EPS,
    )?;
    assert_combine_close("gate.out", &stream.clone_dtoh(&out)?, &want, &h, &m);

    let weight: Vec<bf16> = (0..HIDDEN)
        .map(|i| bf16::from_f32(0.5 + (i % 97) as f32 / 97.0))
        .collect();
    let weight = stream.clone_htod(&weight)?;
    // Twice, so the second run relies on the first resetting the counter.
    for run in 0..2 {
        let mut out = stream.alloc_zeros::<f32>(HIDDEN)?;
        let mut normed = stream.alloc_zeros::<bf16>(HIDDEN)?;
        s.misc.message_gate_out(
            stream,
            &down,
            &mut [GateSlot {
                forget: &forget,
                gain: &gain,
                norm: Some((&weight, normed.as_view_mut())),
            }],
            &h_dev,
            0,
            &m_dev,
            &mut out,
            &mut counter,
            EPS,
        )?;
        let mut normed_want = stream.alloc_zeros::<bf16>(HIDDEN)?;
        norm.rms_norm_f32(stream, &out, &weight, &mut normed_want, EPS)?;
        assert_bf16_close(
            &format!("gate.normed, run {run}"),
            &stream.clone_dtoh(&normed)?,
            &stream.clone_dtoh(&normed_want)?,
            1,
        );
        assert_eq!(
            stream.clone_dtoh(&counter)?,
            [0; GATE_SLOTS],
            "the kernel resets its counter"
        );
    }
    Ok(())
}

/// Three gates in one launch against one launch each, bit for bit: different weights,
/// inputs and norms per gate, the last without a norm, and h rows given once and shared.
#[test]
fn message_gate_out_slots_match_single_slots() -> anyhow::Result<()> {
    let s = Setup::new()?;
    let stream = s.stream();
    let upload = |seed: u32, scale: f32| -> anyhow::Result<CudaSlice<bf16>> {
        Ok(stream.clone_htod(&pseudo_random_bf16(HIDDEN * GATE_RANK, seed, scale, 0.0))?)
    };
    let forgets = [upload(900, 0.3)?, upload(901, 0.3)?, upload(902, 0.3)?];
    let gains = [upload(903, 0.3)?, upload(904, 0.3)?, upload(905, 0.3)?];
    let norm_weights: Vec<CudaSlice<bf16>> = (0..2)
        .map(|k| Ok(stream.clone_htod(&pseudo_random_bf16(HIDDEN, 906 + k, 0.5, 1.0))?))
        .collect::<anyhow::Result<_>>()?;
    let h_host = pseudo_random(GATE_SLOTS * HIDDEN, 910, 3.0);
    let m_host = pseudo_random(GATE_SLOTS * HIDDEN, 911, 3.0);
    let down_host = pseudo_random_bf16(GATE_SLOTS * GATE_RANK, 912, 2.0, 0.0);
    let (h, m) = (stream.clone_htod(&h_host)?, stream.clone_htod(&m_host)?);
    let down = stream.clone_htod(&down_host)?;
    let mut counter = stream.alloc_zeros::<u32>(GATE_SLOTS)?;
    let bits = |v: &[f32]| -> Vec<u32> { v.iter().map(|x| x.to_bits()).collect() };
    let bf_bits = |v: &[bf16]| -> Vec<u16> { v.iter().map(|x| x.to_bits()).collect() };

    for h_stride in [HIDDEN, 0] {
        let mut out = stream.alloc_zeros::<f32>(GATE_SLOTS * HIDDEN)?;
        let mut normed: Vec<CudaSlice<bf16>> = (0..2)
            .map(|_| Ok(stream.alloc_zeros::<bf16>(HIDDEN)?))
            .collect::<anyhow::Result<_>>()?;
        let [n0, n1] = &mut normed[..] else {
            unreachable!()
        };
        s.misc.message_gate_out(
            stream,
            &down,
            &mut [
                GateSlot {
                    forget: &forgets[0],
                    gain: &gains[0],
                    norm: Some((&norm_weights[0], n0.as_view_mut())),
                },
                GateSlot {
                    forget: &forgets[1],
                    gain: &gains[1],
                    norm: Some((&norm_weights[1], n1.as_view_mut())),
                },
                GateSlot {
                    forget: &forgets[2],
                    gain: &gains[2],
                    norm: None,
                },
            ],
            &h,
            h_stride,
            &m,
            &mut out,
            &mut counter,
            EPS,
        )?;
        let out = stream.clone_dtoh(&out)?;
        for k in 0..GATE_SLOTS {
            let label = format!("h stride {h_stride}, gate {k}");
            let h_row = stream.clone_htod(&h_host[k * h_stride..k * h_stride + HIDDEN])?;
            let m_row = stream.clone_htod(&m_host[k * HIDDEN..(k + 1) * HIDDEN])?;
            let down_row = stream.clone_htod(&down_host[k * GATE_RANK..(k + 1) * GATE_RANK])?;
            let mut out_want = stream.alloc_zeros::<f32>(HIDDEN)?;
            let mut normed_want = stream.alloc_zeros::<bf16>(HIDDEN)?;
            s.misc.message_gate_out(
                stream,
                &down_row,
                &mut [GateSlot {
                    forget: &forgets[k],
                    gain: &gains[k],
                    norm: norm_weights.get(k).map(|w| (w, normed_want.as_view_mut())),
                }],
                &h_row,
                0,
                &m_row,
                &mut out_want,
                &mut counter,
                EPS,
            )?;
            assert_eq!(
                bits(&out[k * HIDDEN..(k + 1) * HIDDEN]),
                bits(&stream.clone_dtoh(&out_want)?),
                "{label}: out"
            );
            if k < 2 {
                assert_eq!(
                    bf_bits(&stream.clone_dtoh(&normed[k])?),
                    bf_bits(&stream.clone_dtoh(&normed_want)?),
                    "{label}: normed"
                );
            }
        }
        assert_eq!(
            stream.clone_dtoh(&counter)?,
            [0; GATE_SLOTS],
            "the kernel resets its counters"
        );
    }
    Ok(())
}

/// The many-block argmax against the one-block kernel on long rows: the fixture's rows,
/// a row whose maximum repeats (the first wins), one with NaNs (the first NaN wins), and
/// lengths that leave a scalar tail.
#[test]
fn argmax_wide_matches_argmax() -> anyhow::Result<()> {
    let s = Setup::new()?;
    let stream = s.stream();
    let mut scratch = ArgmaxScratch::new(stream)?;
    let mut rows: Vec<(String, Vec<f32>)> = Vec::new();
    for name in ["argmax.vocab", "argmax.special"] {
        let (x, shape) = s.fixture.f32(&format!("{name}.x"))?;
        for (r, row) in x.chunks(shape[1]).enumerate() {
            rows.push((format!("{name} row {r}"), row.to_vec()));
        }
    }
    for n in [201_024, 201_027, 5, 1] {
        let mut x = pseudo_random(n, 920 + n as u32, 10.0);
        rows.push((format!("random {n}"), x.clone()));
        if n > 1000 {
            x[777] = 50.0;
            x[n - 2] = 50.0;
            x[150_000] = 50.0;
            rows.push((format!("repeated maximum {n}"), x.clone()));
            x[160_000] = f32::NAN;
            x[90_000] = f32::NAN;
            rows.push((format!("NaN {n}"), x));
        }
    }
    for (label, x) in rows {
        let x_dev = stream.clone_htod(&x)?;
        let mut want = stream.alloc_zeros::<i32>(1)?;
        s.misc.argmax_f32(stream, &x_dev, &mut want, x.len())?;
        // Twice, so the second run relies on the first resetting the counter.
        for run in 0..2 {
            let mut got = stream.alloc_zeros::<i32>(1)?;
            s.misc
                .argmax_wide_f32(stream, &x_dev, &mut got, &mut scratch)?;
            assert_eq!(
                stream.clone_dtoh(&got)?,
                stream.clone_dtoh(&want)?,
                "{label}, run {run}"
            );
        }
    }
    Ok(())
}

/// The chunked message gate against single-row calls, with and without the norm. 77 rows
/// cover a full row tile of 64 and a partial one; h and m rows sit 2 x 2304 floats apart,
/// as in the model.
///
/// The chunked kernel sums the two products with the down projection on tensor cores, the
/// single-row kernel with FMA chains, so the dots differ by F32 rounding: 64 products of
/// BF16 values, each exact in F32, whose sums round at most 64 times. The gate output is
/// checked within `GATE_ROWS_TOLERANCE` of `|h| + |m|`, and the BF16 normalized output
/// within one ulp.
#[test]
fn message_gate_out_rows_matches_single_row() -> anyhow::Result<()> {
    const ROWS: usize = 77;
    const STRIDE: usize = 2 * HIDDEN;
    let s = Setup::new()?;
    let stream = s.stream();
    let fixture = &s.fixture;
    let (forget, _) = fixture.bf16("gate.forget_weight")?;
    let (gain, _) = fixture.bf16("gate.gain_weight")?;
    let (forget, gain) = (stream.clone_htod(&forget)?, stream.clone_htod(&gain)?);
    let h_host = pseudo_random(ROWS * STRIDE, 800, 3.0);
    let m_host = pseudo_random(ROWS * STRIDE, 801, 3.0);
    let down_host = pseudo_random_bf16(ROWS * GATE_RANK, 802, 2.0, 0.0);
    let (h, m) = (stream.clone_htod(&h_host)?, stream.clone_htod(&m_host)?);
    let down = stream.clone_htod(&down_host)?;
    let weight: Vec<bf16> = (0..HIDDEN)
        .map(|i| bf16::from_f32(0.5 + (i % 97) as f32 / 97.0))
        .collect();
    let weight = stream.clone_htod(&weight)?;

    let mut out = stream.alloc_zeros::<f32>(ROWS * HIDDEN)?;
    let mut normed = stream.alloc_zeros::<bf16>(ROWS * HIDDEN)?;
    s.misc.message_gate_out_rows(
        stream,
        &down,
        &forget,
        &gain,
        (&h, STRIDE),
        (&m, STRIDE),
        None,
        &mut out,
        Some((&weight, &mut normed)),
        EPS,
        ROWS,
    )?;
    let mut out_plain = stream.alloc_zeros::<f32>(ROWS * HIDDEN)?;
    s.misc.message_gate_out_rows(
        stream,
        &down,
        &forget,
        &gain,
        (&h, STRIDE),
        (&m, STRIDE),
        None,
        &mut out_plain,
        None,
        EPS,
        ROWS,
    )?;
    let (out, normed) = (stream.clone_dtoh(&out)?, stream.clone_dtoh(&normed)?);
    let out_plain = stream.clone_dtoh(&out_plain)?;

    let mut counter = stream.alloc_zeros::<u32>(GATE_SLOTS)?;
    let mut worst = 0.0f32;
    for row in 0..ROWS {
        let h_row = stream.clone_htod(&h_host[row * STRIDE..row * STRIDE + HIDDEN])?;
        let m_row = stream.clone_htod(&m_host[row * STRIDE..row * STRIDE + HIDDEN])?;
        let down_row = stream.clone_htod(&down_host[row * GATE_RANK..(row + 1) * GATE_RANK])?;
        let mut out_want = stream.alloc_zeros::<f32>(HIDDEN)?;
        let mut normed_want = stream.alloc_zeros::<bf16>(HIDDEN)?;
        s.misc.message_gate_out(
            stream,
            &down_row,
            &mut [GateSlot {
                forget: &forget,
                gain: &gain,
                norm: Some((&weight, normed_want.as_view_mut())),
            }],
            &h_row,
            0,
            &m_row,
            &mut out_want,
            &mut counter,
            EPS,
        )?;
        let out_want = stream.clone_dtoh(&out_want)?;
        let range = row * HIDDEN..(row + 1) * HIDDEN;
        assert_eq!(
            out[range.clone()],
            out_plain[range.clone()],
            "row {row}: out with and without the norm"
        );
        for (d, (&got, &want)) in out[range.clone()].iter().zip(&out_want).enumerate() {
            let scale = h_host[row * STRIDE + d].abs() + m_host[row * STRIDE + d].abs();
            let error = (got - want).abs() / scale;
            worst = worst.max(error);
            assert!(
                error <= GATE_ROWS_TOLERANCE,
                "row {row}, column {d}: got {got:e}, want {want:e}"
            );
        }
        assert_bf16_close(
            &format!("row {row}: normed"),
            &normed[range],
            &stream.clone_dtoh(&normed_want)?,
            1,
        );
    }
    eprintln!("largest gate output error: {worst:.2e} of |h| + |m|");
    Ok(())
}

/// The one-launch gate against the down-projection GEMV and `message_gate_out`, for three
/// gates (the last without a norm) with h rows given apart and shared, three runs each so
/// the barrier counters cycle. The down projection sums in another order than the GEMV, so
/// each BF16 value may differ by one ulp. From the kernel's own down projection,
/// `message_gate_out` must give the same output bit for bit and the same normalized output
/// within one BF16 ulp (the sum of squares adds in another order).
#[test]
#[expect(clippy::too_many_lines, reason = "three gates, two references")]
fn message_gate_step_matches_gemv_and_gate_out() -> anyhow::Result<()> {
    let s = Setup::new()?;
    let stream = s.stream();
    let linear = Linear::load(s.gpu.context())?;
    let upload = |len: usize, seed: u32, scale: f32| -> anyhow::Result<CudaSlice<bf16>> {
        Ok(stream.clone_htod(&pseudo_random_bf16(len, seed, scale, 0.0))?)
    };
    let downs = [
        upload(GATE_RANK * 2 * HIDDEN, 930, 0.05)?,
        upload(GATE_RANK * 2 * HIDDEN, 931, 0.05)?,
        upload(GATE_RANK * 2 * HIDDEN, 932, 0.05)?,
    ];
    let inputs = [
        upload(2 * HIDDEN, 933, 2.0)?,
        upload(2 * HIDDEN, 934, 2.0)?,
        upload(2 * HIDDEN, 935, 2.0)?,
    ];
    let forgets = [
        upload(HIDDEN * GATE_RANK, 936, 0.3)?,
        upload(HIDDEN * GATE_RANK, 937, 0.3)?,
        upload(HIDDEN * GATE_RANK, 938, 0.3)?,
    ];
    let gains = [
        upload(HIDDEN * GATE_RANK, 939, 0.3)?,
        upload(HIDDEN * GATE_RANK, 940, 0.3)?,
        upload(HIDDEN * GATE_RANK, 941, 0.3)?,
    ];
    let norm_weights: Vec<CudaSlice<bf16>> = (0..2)
        .map(|k| Ok(stream.clone_htod(&pseudo_random_bf16(HIDDEN, 942 + k, 0.5, 1.0))?))
        .collect::<anyhow::Result<_>>()?;
    let h_host = pseudo_random(GATE_SLOTS * HIDDEN, 944, 3.0);
    let m_host = pseudo_random(GATE_SLOTS * HIDDEN, 945, 3.0);
    let (h, m) = (stream.clone_htod(&h_host)?, stream.clone_htod(&m_host)?);
    let mut counter = stream.alloc_zeros::<u32>(GATE_SLOTS)?;
    let bits = |v: &[f32]| -> Vec<u32> { v.iter().map(|x| x.to_bits()).collect() };

    for (run, h_stride) in [HIDDEN, 0, HIDDEN].into_iter().enumerate() {
        let mut out = stream.alloc_zeros::<f32>(GATE_SLOTS * HIDDEN)?;
        let mut down = stream.alloc_zeros::<bf16>(GATE_SLOTS * GATE_RANK)?;
        let mut normed: Vec<CudaSlice<bf16>> = (0..2)
            .map(|_| Ok(stream.alloc_zeros::<bf16>(HIDDEN)?))
            .collect::<anyhow::Result<_>>()?;
        let [n0, n1] = &mut normed[..] else {
            unreachable!()
        };
        let mut gates = [
            GateStep {
                down_weight: Some(&downs[0]),
                input: inputs[0].as_view(),
                forget: &forgets[0],
                gain: &gains[0],
                fp8: None,
                norm: Some((&norm_weights[0], n0.as_view_mut())),
            },
            GateStep {
                down_weight: Some(&downs[1]),
                input: inputs[1].as_view(),
                forget: &forgets[1],
                gain: &gains[1],
                fp8: None,
                norm: Some((&norm_weights[1], n1.as_view_mut())),
            },
            GateStep {
                down_weight: Some(&downs[2]),
                input: inputs[2].as_view(),
                forget: &forgets[2],
                gain: &gains[2],
                fp8: None,
                norm: None,
            },
        ];
        s.misc.message_gate_step(
            stream,
            &mut gates,
            &h,
            h_stride,
            &m,
            &mut out,
            Some(&mut down),
            EPS,
        )?;
        let out = stream.clone_dtoh(&out)?;
        let down = stream.clone_dtoh(&down)?;
        for k in 0..GATE_SLOTS {
            let label = format!("run {run}, gate {k}");
            let mut down_want = stream.alloc_zeros::<bf16>(GATE_RANK)?;
            linear.gemv_bf16(
                stream,
                &mut [Gemv::new(&downs[k], &inputs[k], &mut down_want)],
                Start::Serial,
            )?;
            let own_down = &down[k * GATE_RANK..(k + 1) * GATE_RANK];
            assert_bf16_close(
                &format!("{label}: down"),
                own_down,
                &stream.clone_dtoh(&down_want)?,
                1,
            );
            let h_row = stream.clone_htod(&h_host[k * h_stride..k * h_stride + HIDDEN])?;
            let m_row = stream.clone_htod(&m_host[k * HIDDEN..(k + 1) * HIDDEN])?;
            let mut out_want = stream.alloc_zeros::<f32>(HIDDEN)?;
            let mut normed_want = stream.alloc_zeros::<bf16>(HIDDEN)?;
            s.misc.message_gate_out(
                stream,
                &stream.clone_htod(own_down)?,
                &mut [GateSlot {
                    forget: &forgets[k],
                    gain: &gains[k],
                    norm: norm_weights.get(k).map(|w| (w, normed_want.as_view_mut())),
                }],
                &h_row,
                0,
                &m_row,
                &mut out_want,
                &mut counter,
                EPS,
            )?;
            assert_eq!(
                bits(&out[k * HIDDEN..(k + 1) * HIDDEN]),
                bits(&stream.clone_dtoh(&out_want)?),
                "{label}: out"
            );
            if k < 2 {
                assert_bf16_close(
                    &format!("{label}: normed"),
                    &stream.clone_dtoh(&normed[k])?,
                    &stream.clone_dtoh(&normed_want)?,
                    1,
                );
            }
        }
    }
    Ok(())
}

/// The value of an E4M3 code.
fn e4m3_value(bits: u8) -> f64 {
    let sign = if bits & 0x80 == 0 { 1.0 } else { -1.0 };
    let exponent = i32::from((bits >> 3) & 0xf);
    let mantissa = f64::from(bits & 7);
    let magnitude = if exponent == 0 {
        mantissa / 8.0 * 2f64.powi(-6)
    } else {
        (1.0 + mantissa / 8.0) * 2f64.powi(exponent - 7)
    };
    sign * magnitude
}

/// An FP8 matrix's rows dequantized to F64: each E4M3 value times its row's scale.
fn dequantize(stream: &std::sync::Arc<CudaStream>, m: &Fp8Matrix) -> anyhow::Result<Vec<f64>> {
    let bytes = stream.clone_dtoh(&m.values())?;
    let scales: Vec<f32> = stream
        .clone_dtoh(&m.scales())?
        .as_chunks()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    Ok(bytes
        .iter()
        .enumerate()
        .map(|(i, &q)| e4m3_value(q) * f64::from(scales[i / m.k()]))
        .collect())
}

/// Checks one FP8 gate's results against an F64 model over its dequantized weights: the
/// down projection within one BF16 ulp, the output from the kernel's own down projection
/// within 1e-5 relative to |h| + |m|, and with a norm weight the normalized output within
/// one BF16 ulp of the model's norm of the kernel's output.
#[expect(clippy::too_many_arguments, reason = "one gate's inputs and outputs")]
fn check_fp8_gate(
    stream: &std::sync::Arc<CudaStream>,
    label: &str,
    weights: &[Fp8Matrix; 3],
    input: &[bf16],
    (h_row, m_row): (&[f32], &[f32]),
    down: &[bf16],
    out: &[f32],
    normed: Option<(&[bf16], &[bf16])>,
) -> anyhow::Result<()> {
    let down_w = dequantize(stream, &weights[0])?;
    let forget_w = dequantize(stream, &weights[1])?;
    let gain_w = dequantize(stream, &weights[2])?;
    let input: Vec<f64> = input.iter().map(|v| v.to_f64()).collect();
    let down_want: Vec<bf16> = (0..GATE_RANK)
        .map(|r| {
            let row = &down_w[r * 2 * HIDDEN..(r + 1) * 2 * HIDDEN];
            bf16::from_f64(row.iter().zip(&input).map(|(w, x)| w * x).sum())
        })
        .collect();
    assert_bf16_close(&format!("{label}: down"), down, &down_want, 1);

    let own_down: Vec<f64> = down.iter().map(|v| v.to_f64()).collect();
    let dot = |w: &[f64], row: usize| -> f64 {
        w[row * GATE_RANK..(row + 1) * GATE_RANK]
            .iter()
            .zip(&own_down)
            .map(|(w, d)| w * d)
            .sum()
    };
    for row in 0..HIDDEN {
        let (hv, mv) = (f64::from(h_row[row]), f64::from(m_row[row]));
        let want = (1.0 - dot(&forget_w, row).tanh()) * hv + dot(&gain_w, row).tanh() * mv;
        let err = (f64::from(out[row]) - want).abs();
        assert!(
            err <= 1e-5 * (hv.abs() + mv.abs()),
            "{label}: out[{row}] = {}, want {want}",
            out[row]
        );
    }
    if let Some((normed, norm_weight)) = normed {
        let mean = out.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>() / HIDDEN as f64;
        let inv = 1.0 / (mean + f64::from(EPS)).sqrt();
        let normed_want: Vec<bf16> = out
            .iter()
            .zip(norm_weight)
            .map(|(&o, w)| bf16::from_f64(f64::from(o) * inv * w.to_f64()))
            .collect();
        assert_bf16_close(&format!("{label}: normed"), normed, &normed_want, 1);
    }
    Ok(())
}

/// The gate with FP8 weights against [`check_fp8_gate`]'s model, for two gates in one
/// launch (the second without a norm), twice so the barrier counters cycle.
#[test]
fn message_gate_step_fp8_matches_reference() -> anyhow::Result<()> {
    const GATES: usize = 2;
    let setup = Setup::new()?;
    let stream = setup.stream();
    let fp8 = LinearFp8::load(setup.gpu.context())?;
    let mut gates_fp8 = Vec::new();
    let mut bf16_weights = Vec::new();
    let mut inputs = Vec::new();
    for k in 0..GATES {
        let seed = 960 + 10 * u32::try_from(k)?;
        let down =
            stream.clone_htod(&pseudo_random_bf16(GATE_RANK * 2 * HIDDEN, seed, 0.05, 0.0))?;
        let forget =
            stream.clone_htod(&pseudo_random_bf16(HIDDEN * GATE_RANK, seed + 1, 0.3, 0.0))?;
        let gain =
            stream.clone_htod(&pseudo_random_bf16(HIDDEN * GATE_RANK, seed + 2, 0.3, 0.0))?;
        gates_fp8.push([
            fp8.quantize_any(stream, &down, GATE_RANK, 2 * HIDDEN)?,
            fp8.quantize_any(stream, &forget, HIDDEN, GATE_RANK)?,
            fp8.quantize_any(stream, &gain, HIDDEN, GATE_RANK)?,
        ]);
        bf16_weights.push([down, forget, gain]);
        inputs.push(pseudo_random_bf16(2 * HIDDEN, seed + 3, 2.0, 0.0));
    }
    let input_dev: Vec<CudaSlice<bf16>> = inputs
        .iter()
        .map(|x| Ok(stream.clone_htod(x)?))
        .collect::<anyhow::Result<_>>()?;
    let norm_host = pseudo_random_bf16(HIDDEN, 980, 0.5, 1.0);
    let norm_weight = stream.clone_htod(&norm_host)?;
    let h_host = pseudo_random(GATES * HIDDEN, 981, 3.0);
    let m_host = pseudo_random(GATES * HIDDEN, 982, 3.0);
    let (h_dev, m_dev) = (stream.clone_htod(&h_host)?, stream.clone_htod(&m_host)?);
    for run in 0..2 {
        let mut out = stream.alloc_zeros::<f32>(GATES * HIDDEN)?;
        let mut down = stream.alloc_zeros::<bf16>(GATES * GATE_RANK)?;
        let mut normed = stream.alloc_zeros::<bf16>(HIDDEN)?;
        let mut gates: Vec<GateStep<'_>> = (0..GATES)
            .map(|k| GateStep {
                down_weight: Some(&bf16_weights[k][0]),
                input: input_dev[k].as_view(),
                forget: &bf16_weights[k][1],
                gain: &bf16_weights[k][2],
                fp8: Some(&gates_fp8[k]),
                norm: None,
            })
            .collect();
        gates[0].norm = Some((&norm_weight, normed.as_view_mut()));
        setup.misc.message_gate_step(
            stream,
            &mut gates,
            &h_dev,
            HIDDEN,
            &m_dev,
            &mut out,
            Some(&mut down),
            EPS,
        )?;
        drop(gates);
        let (out, down, normed) = (
            stream.clone_dtoh(&out)?,
            stream.clone_dtoh(&down)?,
            stream.clone_dtoh(&normed)?,
        );
        for k in 0..GATES {
            let rows = k * HIDDEN..(k + 1) * HIDDEN;
            check_fp8_gate(
                stream,
                &format!("run {run}, gate {k}"),
                &gates_fp8[k],
                &inputs[k],
                (&h_host[rows.clone()], &m_host[rows.clone()]),
                &down[k * GATE_RANK..(k + 1) * GATE_RANK],
                &out[rows],
                (k == 0).then_some((normed.as_slice(), norm_host.as_slice())),
            )?;
        }
    }
    Ok(())
}

/// The decode step's first kernel equals the embedding and the three norms it replaces, bit
/// for bit, for tokens inside the vocabulary and one outside it.
#[test]
fn decode_prologue_matches_separate_kernels() -> anyhow::Result<()> {
    let gpu = Gpu::new(0)?;
    let stream = gpu.stream();
    let misc = Misc::load(gpu.context())?;
    let norm = Norm::load(gpu.context())?;
    let elementwise = Elementwise::load(gpu.context())?;
    let vocab = 50;
    let table = stream.clone_htod(&pseudo_random_bf16(vocab * HIDDEN, 1, 0.05, 0.0))?;
    let embed_norm = stream.clone_htod(&pseudo_random_bf16(HIDDEN, 2, 0.5, 1.0))?;
    let input_norm = stream.clone_htod(&pseudo_random_bf16(HIDDEN, 3, 0.5, 1.0))?;
    let bits16 = |v: Vec<bf16>| v.into_iter().map(bf16::to_bits).collect::<Vec<_>>();
    let bits32 = |v: Vec<f32>| v.into_iter().map(f32::to_bits).collect::<Vec<_>>();
    for token in [0, 17, 49, 50] {
        let ids = stream.clone_htod(&[token])?;
        let mut embedded = stream.alloc_zeros::<bf16>(HIDDEN)?;
        let mut hidden_want = stream.alloc_zeros::<bf16>(HIDDEN)?;
        let mut normed_want = stream.alloc_zeros::<f32>(HIDDEN)?;
        let mut input_want = stream.alloc_zeros::<bf16>(HIDDEN)?;
        elementwise.embedding(stream, &ids, &table, &mut embedded)?;
        norm.rms_norm(stream, &embedded, &embed_norm, &mut hidden_want, EPS)?;
        norm.unweighted_rms_norm_bf16(stream, &hidden_want, &mut normed_want, EPS)?;
        norm.rms_norm(stream, &hidden_want, &input_norm, &mut input_want, EPS)?;
        let mut hidden = stream.alloc_zeros::<bf16>(HIDDEN)?;
        let mut normed = stream.alloc_zeros::<f32>(HIDDEN)?;
        let mut input = stream.alloc_zeros::<bf16>(HIDDEN)?;
        misc.decode_prologue(
            stream,
            &ids,
            &table,
            &embed_norm,
            &input_norm,
            &mut hidden,
            &mut normed,
            &mut input,
            EPS,
        )?;
        assert_eq!(
            bits16(stream.clone_dtoh(&hidden)?),
            bits16(stream.clone_dtoh(&hidden_want)?),
            "token {token}: hidden"
        );
        assert_eq!(
            bits32(stream.clone_dtoh(&normed)?),
            bits32(stream.clone_dtoh(&normed_want)?),
            "token {token}: normed"
        );
        assert_eq!(
            bits16(stream.clone_dtoh(&input)?),
            bits16(stream.clone_dtoh(&input_want)?),
            "token {token}: input"
        );
    }
    Ok(())
}

/// Fused pre-attention `depth_gate_step` matches separate `depth_attention_step` +
/// `message_gate_step` for both BF16 and FP8 weights, both shared (1 query) and separate
/// (3 queries) attention inputs, across block counts.
#[test]
#[expect(clippy::too_many_lines, reason = "tests BF16 and FP8, shared and separate queries")]
fn depth_gate_step_matches_separate_kernels() -> anyhow::Result<()> {
    let gpu = Gpu::new(0)?;
    let stream = gpu.stream();
    let misc = Misc::load(gpu.context())?;
    let routing = Routing::load(gpu.context())?;
    let linear_fp8 = LinearFp8::load(gpu.context())?;

    let down_w: Vec<CudaSlice<bf16>> = (0..3)
        .map(|k| Ok(stream.clone_htod(&pseudo_random_bf16(GATE_RANK * 2 * HIDDEN, 800 + k, 0.1, 0.0))?))
        .collect::<anyhow::Result<_>>()?;
    let forget_w: Vec<CudaSlice<bf16>> = (0..3)
        .map(|k| Ok(stream.clone_htod(&pseudo_random_bf16(HIDDEN * GATE_RANK, 810 + k, 0.1, 0.0))?))
        .collect::<anyhow::Result<_>>()?;
    let gain_w: Vec<CudaSlice<bf16>> = (0..3)
        .map(|k| Ok(stream.clone_htod(&pseudo_random_bf16(HIDDEN * GATE_RANK, 820 + k, 0.1, 0.0))?))
        .collect::<anyhow::Result<_>>()?;
    let fp8_w: Vec<[Fp8Matrix; 3]> = (0..3)
        .map(|k| {
            Ok([
                linear_fp8.quantize_any(stream, &down_w[k], GATE_RANK, 2 * HIDDEN)?,
                linear_fp8.quantize_any(stream, &forget_w[k], HIDDEN, GATE_RANK)?,
                linear_fp8.quantize_any(stream, &gain_w[k], HIDDEN, GATE_RANK)?,
            ])
        })
        .collect::<anyhow::Result<_>>()?;
    let norm_w: Vec<CudaSlice<bf16>> = (0..3)
        .map(|k| Ok(stream.clone_htod(&pseudo_random_bf16(HIDDEN, 830 + k, 0.5, 1.0))?))
        .collect::<anyhow::Result<_>>()?;
    let msg_half: Vec<CudaSlice<bf16>> = (0..3)
        .map(|k| Ok(stream.clone_htod(&pseudo_random_bf16(HIDDEN, 840 + k, 0.6, 0.0))?))
        .collect::<anyhow::Result<_>>()?;
    let m_buf = stream.clone_htod(&pseudo_random(3 * HIDDEN, 850, 0.7))?;
    let partial_bf16 = stream.clone_htod(&pseudo_random_bf16(HIDDEN, 851, 0.8, 0.0))?;

    for &(nb, shared, use_fp8) in &[
        (2usize, true, false),
        (3, true, true),
        (4, false, false),
        (5, false, true),
    ] {
        let nq = if shared { 1 } else { 3 };
        let blocks = stream.clone_htod(&pseudo_random(nb * HIDDEN, 860 + nb as u32, 0.5))?;
        let queries = stream.clone_htod(&pseudo_random(nq * HIDDEN, 870 + nq as u32, 0.4))?;

        // Reference: depth_attention_step -> message_gate_step.
        let mut ref_normed_partial = stream.alloc_zeros::<f32>(HIDDEN)?;
        let mut ref_attn_in = stream.alloc_zeros::<f32>(nq * HIDDEN)?;
        let mut ref_gate_in: Vec<CudaSlice<bf16>> = (0..3)
            .map(|k| {
                let mut buf = stream.alloc_zeros::<bf16>(2 * HIDDEN)?;
                stream.memcpy_dtod(&msg_half[k], &mut buf.slice_mut(HIDDEN..))?;
                Ok(buf)
            })
            .collect::<anyhow::Result<_>>()?;
        let [rg0, rg1, rg2] = &mut ref_gate_in[..] else {
            unreachable!()
        };
        routing.depth_attention_step(
            stream,
            DepthStep {
                blocks: blocks.as_view(),
                partial: StepPartial::Bf16(partial_bf16.as_view()),
                normed_partial: Some(ref_normed_partial.as_view_mut()),
                queries: queries.as_view(),
                batch: 1,
                out: Some(ref_attn_in.as_view_mut()),
                dests: [
                    Some(NormDest {
                        out: rg0.slice_mut(..HIDDEN),
                        weight: None,
                    }),
                    Some(NormDest {
                        out: rg1.slice_mut(..HIDDEN),
                        weight: None,
                    }),
                    Some(NormDest {
                        out: rg2.slice_mut(..HIDDEN),
                        weight: None,
                    }),
                    None,
                ],
                dst_stride: 0,
            },
            EPS,
        )?;
        let mut ref_out = stream.alloc_zeros::<f32>(3 * HIDDEN)?;
        let mut ref_qkv: [CudaSlice<bf16>; 3] = [
            stream.alloc_zeros(HIDDEN)?,
            stream.alloc_zeros(HIDDEN)?,
            stream.alloc_zeros(HIDDEN)?,
        ];
        let [rq0, rq1, rq2] = &mut ref_qkv;
        let mut ref_steps = [
            GateStep {
                down_weight: Some(&down_w[0]),
                input: rg0.as_view(),
                forget: &forget_w[0],
                gain: &gain_w[0],
                fp8: use_fp8.then_some(&fp8_w[0]),
                norm: Some((&norm_w[0], rq0.as_view_mut())),
            },
            GateStep {
                down_weight: Some(&down_w[1]),
                input: rg1.as_view(),
                forget: &forget_w[1],
                gain: &gain_w[1],
                fp8: use_fp8.then_some(&fp8_w[1]),
                norm: Some((&norm_w[1], rq1.as_view_mut())),
            },
            GateStep {
                down_weight: Some(&down_w[2]),
                input: rg2.as_view(),
                forget: &forget_w[2],
                gain: &gain_w[2],
                fp8: use_fp8.then_some(&fp8_w[2]),
                norm: Some((&norm_w[2], rq2.as_view_mut())),
            },
        ];
        misc.message_gate_step(
            stream,
            &mut ref_steps,
            &ref_attn_in,
            if shared { 0 } else { HIDDEN },
            &m_buf,
            &mut ref_out,
            None,
            EPS,
        )?;

        // Fused kernel.
        let mut got_normed_partial = stream.alloc_zeros::<f32>(HIDDEN)?;
        let mut got_out = stream.alloc_zeros::<f32>(3 * HIDDEN)?;
        let mut got_qkv: [CudaSlice<bf16>; 3] = [
            stream.alloc_zeros(HIDDEN)?,
            stream.alloc_zeros(HIDDEN)?,
            stream.alloc_zeros(HIDDEN)?,
        ];
        let [gq0, gq1, gq2] = &mut got_qkv;
        let mut slots = [
            DepthGateSlot {
                down_weight: Some(&down_w[0]),
                input_msg: msg_half[0].as_view(),
                forget: &forget_w[0],
                gain: &gain_w[0],
                fp8: use_fp8.then_some(&fp8_w[0]),
                norm_weight: &norm_w[0],
                normed: gq0.as_view_mut(),
            },
            DepthGateSlot {
                down_weight: Some(&down_w[1]),
                input_msg: msg_half[1].as_view(),
                forget: &forget_w[1],
                gain: &gain_w[1],
                fp8: use_fp8.then_some(&fp8_w[1]),
                norm_weight: &norm_w[1],
                normed: gq1.as_view_mut(),
            },
            DepthGateSlot {
                down_weight: Some(&down_w[2]),
                input_msg: msg_half[2].as_view(),
                forget: &forget_w[2],
                gain: &gain_w[2],
                fp8: use_fp8.then_some(&fp8_w[2]),
                norm_weight: &norm_w[2],
                normed: gq2.as_view_mut(),
            },
        ];
        let blocks_view = blocks.as_view();
        let pb_view = partial_bf16.as_view();
        let q_view = queries.as_view();
        misc.depth_gate_step(
            stream,
            &blocks_view,
            &pb_view,
            Some(got_normed_partial.as_view_mut()),
            &q_view,
            if shared { 0 } else { HIDDEN },
            &mut slots,
            &m_buf,
            Some(got_out.as_view_mut()),
            EPS,
        )?;

        let label = format!("depth_gate_step(nb={nb}, shared={shared}, fp8={use_fp8})");
        assert_f32_exact(
            &format!("{label}: normed_partial"),
            &stream.clone_dtoh(&got_normed_partial)?,
            &stream.clone_dtoh(&ref_normed_partial)?,
        );
        assert_f32_exact(
            &format!("{label}: out"),
            &stream.clone_dtoh(&got_out)?,
            &stream.clone_dtoh(&ref_out)?,
        );
        for k in 0..3 {
            assert_bf16_close(
                &format!("{label}: qkv[{k}]"),
                &stream.clone_dtoh(&got_qkv[k])?,
                &stream.clone_dtoh(&ref_qkv[k])?,
                0,
            );
        }
    }
    Ok(())
}

/// Fused pre-MLP `depth_gate_mlp_step` matches separate `depth_attention_step` +
/// `message_gate_step` bit-for-bit (0-ULP) for both BF16 and FP8 weights, with and without
/// `conv_hidden`, across block counts.
#[test]
#[expect(clippy::too_many_lines, reason = "tests BF16 and FP8 pre-MLP fused chain")]
fn depth_gate_mlp_step_matches_separate_kernels() -> anyhow::Result<()> {
    let gpu = Gpu::new(0)?;
    let stream = gpu.stream();
    let misc = Misc::load(gpu.context())?;
    let routing = Routing::load(gpu.context())?;
    let linear_fp8 = LinearFp8::load(gpu.context())?;

    let down_w = stream.clone_htod(&pseudo_random_bf16(GATE_RANK * 2 * HIDDEN, 900, 0.1, 0.0))?;
    let forget_w = stream.clone_htod(&pseudo_random_bf16(HIDDEN * GATE_RANK, 901, 0.1, 0.0))?;
    let gain_w = stream.clone_htod(&pseudo_random_bf16(HIDDEN * GATE_RANK, 902, 0.1, 0.0))?;
    let fp8_w = [
        linear_fp8.quantize_any(stream, &down_w, GATE_RANK, 2 * HIDDEN)?,
        linear_fp8.quantize_any(stream, &forget_w, HIDDEN, GATE_RANK)?,
        linear_fp8.quantize_any(stream, &gain_w, HIDDEN, GATE_RANK)?,
    ];
    let norm_w = stream.clone_htod(&pseudo_random_bf16(HIDDEN, 903, 0.5, 1.0))?;
    let msg_half = stream.clone_htod(&pseudo_random_bf16(HIDDEN, 904, 0.6, 0.0))?;
    let m_buf = stream.clone_htod(&pseudo_random(HIDDEN, 905, 0.7))?;
    let conv_w = stream.clone_htod(&pseudo_random_bf16(4 * HIDDEN, 906, 0.3, 0.0))?;
    let query = stream.clone_htod(&pseudo_random(HIDDEN, 907, 0.4))?;
    let partial_x = stream.clone_htod(&pseudo_random_bf16(HIDDEN, 908, 0.8, 0.0))?;
    let conv_hidden_buf = stream.clone_htod(&pseudo_random_bf16(HIDDEN, 909, 0.8, 0.0))?;
    let conv_state_init = pseudo_random(3 * HIDDEN, 910, 0.6);

    for (idx, (nb, with_hidden, use_fp8)) in [
        (2usize, false, false),
        (3, true, true),
        (4, false, true),
        (5, true, false),
    ]
    .into_iter()
    .enumerate()
    {
        let blocks = stream.clone_htod(&pseudo_random(nb * HIDDEN, 920 + idx as u32, 0.5))?;

        // Reference: depth_attention_step -> message_gate_step.
        let mut ref_conv_state = stream.clone_htod(&conv_state_init)?;
        let mut ref_residual = stream.alloc_zeros::<f32>(HIDDEN)?;
        let mut ref_mlp_in = stream.alloc_zeros::<f32>(HIDDEN)?;
        let mut ref_gate_in = stream.alloc_zeros::<bf16>(2 * HIDDEN)?;
        stream.memcpy_dtod(&msg_half, &mut ref_gate_in.slice_mut(HIDDEN..))?;
        routing.depth_attention_step(
            stream,
            DepthStep {
                blocks: blocks.as_view(),
                partial: StepPartial::ConvResidual {
                    x: partial_x.as_view(),
                    weight: &conv_w,
                    state: ref_conv_state.as_view_mut(),
                    hidden: with_hidden.then(|| conv_hidden_buf.as_view()),
                    residual: ref_residual.as_view_mut(),
                },
                normed_partial: None,
                queries: query.as_view(),
                batch: 1,
                out: Some(ref_mlp_in.as_view_mut()),
                dests: [
                    None,
                    None,
                    None,
                    Some(NormDest {
                        out: ref_gate_in.slice_mut(..HIDDEN),
                        weight: None,
                    }),
                ],
                dst_stride: 0,
            },
            EPS,
        )?;
        let mut ref_gated = stream.alloc_zeros::<f32>(HIDDEN)?;
        let mut ref_mlp_x = stream.alloc_zeros::<bf16>(HIDDEN)?;
        let mut ref_steps = [GateStep {
            down_weight: Some(&down_w),
            input: ref_gate_in.as_view(),
            forget: &forget_w,
            gain: &gain_w,
            fp8: use_fp8.then_some(&fp8_w),
            norm: Some((&norm_w, ref_mlp_x.as_view_mut())),
        }];
        misc.message_gate_step(
            stream,
            &mut ref_steps,
            &ref_mlp_in,
            0,
            &m_buf,
            &mut ref_gated,
            None,
            EPS,
        )?;

        // Fused kernel.
        let mut got_conv_state = stream.clone_htod(&conv_state_init)?;
        let mut got_residual = stream.alloc_zeros::<f32>(HIDDEN)?;
        let mut got_gated = stream.alloc_zeros::<f32>(HIDDEN)?;
        let mut got_mlp_x = stream.alloc_zeros::<bf16>(HIDDEN)?;
        let mut slot = DepthGateSlot {
            down_weight: Some(&down_w),
            input_msg: msg_half.as_view(),
            forget: &forget_w,
            gain: &gain_w,
            fp8: use_fp8.then_some(&fp8_w),
            norm_weight: &norm_w,
            normed: got_mlp_x.as_view_mut(),
        };
        let blocks_view = blocks.as_view();
        let px_view = partial_x.as_view();
        let hid_view = with_hidden.then(|| conv_hidden_buf.as_view());
        let q_view = query.as_view();
        misc.depth_gate_mlp_step(
            stream,
            &blocks_view,
            &px_view,
            &conv_w,
            got_conv_state.as_view_mut(),
            hid_view.as_ref(),
            got_residual.as_view_mut(),
            &q_view,
            &mut slot,
            &m_buf,
            Some(got_gated.as_view_mut()),
            EPS,
        )?;

        let label = format!("depth_gate_mlp_step(nb={nb}, hidden={with_hidden}, fp8={use_fp8})");
        assert_f32_exact(
            &format!("{label}: residual"),
            &stream.clone_dtoh(&got_residual)?,
            &stream.clone_dtoh(&ref_residual)?,
        );
        assert_f32_exact(
            &format!("{label}: conv_state"),
            &stream.clone_dtoh(&got_conv_state)?,
            &stream.clone_dtoh(&ref_conv_state)?,
        );
        assert_f32_exact(
            &format!("{label}: gated"),
            &stream.clone_dtoh(&got_gated)?,
            &stream.clone_dtoh(&ref_gated)?,
        );
        assert_bf16_close(
            &format!("{label}: mlp_x"),
            &stream.clone_dtoh(&got_mlp_x)?,
            &stream.clone_dtoh(&ref_mlp_x)?,
            0,
        );
    }
    Ok(())
}
