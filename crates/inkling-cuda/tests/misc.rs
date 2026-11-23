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
use inkling_cuda::{
    Gpu,
    ops::{
        linear::{Gemv, Linear, Start},
        misc::{ArgmaxScratch, GATE_SLOTS, GateSlot, GateStep, GateStepScratch, Misc},
        norm::Norm,
    },
};

const HIDDEN: usize = 2304;
const GATE_RANK: usize = 64;
const EPS: f32 = 1e-6;
/// Allowed combine error, as a multiple of `|h| + |m|`. Metal's tanh is up to 3 F32 ulps from
/// the true value, which is at most 3 * 2^-24 for a result below 1.
const COMBINE_TOLERANCE: f32 = 1.0 / (1 << 21) as f32;

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

/// The chunked message gate against single-row calls, bit for bit, with and without the
/// norm. 45 rows cover a full row tile of 32 and a partial one; h and m rows sit 2 x 2304
/// floats apart, as in the model.
#[test]
fn message_gate_out_rows_matches_single_row() -> anyhow::Result<()> {
    const ROWS: usize = 45;
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
        &mut out_plain,
        None,
        EPS,
        ROWS,
    )?;
    let (out, normed) = (stream.clone_dtoh(&out)?, stream.clone_dtoh(&normed)?);
    let out_plain = stream.clone_dtoh(&out_plain)?;

    let bits = |v: &[f32]| -> Vec<u32> { v.iter().map(|x| x.to_bits()).collect() };
    let mut counter = stream.alloc_zeros::<u32>(GATE_SLOTS)?;
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
        let out_want = bits(&stream.clone_dtoh(&out_want)?);
        let range = row * HIDDEN..(row + 1) * HIDDEN;
        assert_eq!(bits(&out[range.clone()]), out_want, "row {row}: out");
        assert_eq!(
            bits(&out_plain[range.clone()]),
            out_want,
            "row {row}: out without the norm"
        );
        let normed_bits: Vec<u16> = normed[range].iter().map(|x| x.to_bits()).collect();
        let want_bits: Vec<u16> = stream
            .clone_dtoh(&normed_want)?
            .iter()
            .map(|x| x.to_bits())
            .collect();
        assert_eq!(normed_bits, want_bits, "row {row}: normed");
    }
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
    let mut scratch = GateStepScratch::new(stream)?;
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
                down_weight: &downs[0],
                input: inputs[0].as_view(),
                forget: &forgets[0],
                gain: &gains[0],
                norm: Some((&norm_weights[0], n0.as_view_mut())),
            },
            GateStep {
                down_weight: &downs[1],
                input: inputs[1].as_view(),
                forget: &forgets[1],
                gain: &gains[1],
                norm: Some((&norm_weights[1], n1.as_view_mut())),
            },
            GateStep {
                down_weight: &downs[2],
                input: inputs[2].as_view(),
                forget: &forgets[2],
                gain: &gains[2],
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
            &mut scratch,
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
