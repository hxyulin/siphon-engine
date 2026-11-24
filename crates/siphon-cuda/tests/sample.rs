//! The sampling kernels (`ops::sample`) against a host version of vLLM's filters.
//!
//! - Greedy rows give `Misc::argmax_f32`'s tokens.
//! - Sampled rows: many rows of the same logits, each with its own position, give tokens
//!   whose counts fit the filtered distribution by a chi-square test, and never a token the
//!   filters drop.
//! - A row's token depends only on its logits, seed and position.

#![cfg(target_os = "linux")]

use std::sync::Arc;

use cudarc::driver::CudaStream;
use siphon_cuda::{
    Gpu,
    ops::{
        misc::Misc,
        sample::{Sample, SampleBuffers, Sampling},
    },
};

/// The model's vocabulary without padding.
const VOCAB: usize = 200_058;

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// Logits shaped like a language model's: a bulk of roughly normal values with spread
/// `scale`, `spikes` larger ones, and repeated values so that ties occur.
fn logits(n: usize, scale: f32, spikes: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    let mut uniform = || (splitmix64(&mut state) >> 40) as f32 / (1u64 << 24) as f32;
    let mut x: Vec<f32> = (0..n)
        .map(|_| {
            let normal: f32 = (0..4).map(|_| uniform()).sum::<f32>() - 2.0;
            // Round to 1/64 so that equal logits occur.
            (normal * scale * 64.0).round() / 64.0
        })
        .collect();
    for i in 0..n.min(spikes) {
        let j = (splitmix64(&mut state) % n as u64) as usize;
        x[j] = scale * 2.5 + i as f32 * 0.3 * scale;
    }
    x
}

/// vLLM's filtered distribution over `logits`, in F64: temperature, then top-k (ties at
/// the k-th value kept), then top-p (a token stays when the strictly more likely kept
/// tokens hold less than `top_p`), then min-p.
fn filtered(logits: &[f32], sampling: Sampling) -> Vec<f64> {
    let len = logits.len();
    let scaled: Vec<f64> = logits
        .iter()
        .map(|&l| f64::from(l) / f64::from(sampling.temperature))
        .collect();
    let mut order: Vec<usize> = (0..len).collect();
    order.sort_by(|&a, &b| scaled[b].total_cmp(&scaled[a]));
    let max = scaled[order[0]];
    let mut keep = vec![true; len];
    let top_k = sampling.top_k as usize;
    if top_k > 0 && top_k < len {
        let kth = scaled[order[top_k - 1]];
        for (kept, &value) in keep.iter_mut().zip(&scaled) {
            *kept = value >= kth;
        }
    }
    if sampling.top_p < 1.0 {
        let total: f64 = (0..len)
            .filter(|&t| keep[t])
            .map(|t| (scaled[t] - max).exp())
            .sum();
        let mut above = 0.0;
        let mut start = 0;
        while start < len {
            // One run of equal values.
            let value = scaled[order[start]];
            let mut end = start;
            let mut run = 0.0;
            while end < len && scaled[order[end]].to_bits() == value.to_bits() {
                if keep[order[end]] {
                    run += (value - max).exp();
                }
                end += 1;
            }
            let stays = above < f64::from(sampling.top_p) * total;
            for &t in &order[start..end] {
                keep[t] &= stays;
            }
            above += run;
            start = end;
        }
    }
    if sampling.min_p > 0.0 {
        for (kept, &value) in keep.iter_mut().zip(&scaled) {
            *kept &= (value - max).exp() >= f64::from(sampling.min_p);
        }
    }
    let mut probs: Vec<f64> = keep
        .iter()
        .zip(&scaled)
        .map(|(&kept, &value)| if kept { (value - max).exp() } else { 0.0 })
        .collect();
    let sum: f64 = probs.iter().sum();
    for prob in &mut probs {
        *prob /= sum;
    }
    probs
}

/// A z-score for a chi-square statistic with `dof` degrees of freedom (Wilson and
/// Hilferty). z = 4.75 is a one-sided p-value of about 1e-6.
fn chi_square_z(statistic: f64, dof: f64) -> f64 {
    let a = 2.0 / (9.0 * dof);
    ((statistic / dof).powf(1.0 / 3.0) - (1.0 - a)) / a.sqrt()
}

struct Setup {
    gpu: Gpu,
    sample: Sample,
}

impl Setup {
    fn new() -> anyhow::Result<Self> {
        let gpu = Gpu::new(0)?;
        let sample = Sample::load(gpu.context())?;
        Ok(Self { gpu, sample })
    }

    fn stream(&self) -> &Arc<CudaStream> {
        self.gpu.stream()
    }

    /// Each row's token, for rows of `n` logits each with its own settings.
    fn run(
        &self,
        logits: &[f32],
        n: usize,
        settings: &[(Sampling, usize)],
    ) -> anyhow::Result<Vec<i32>> {
        let rows = settings.len();
        let stream = self.stream();
        let device = stream.clone_htod(logits)?;
        let mut buffers = SampleBuffers::new(stream, rows)?;
        buffers.upload(stream, settings)?;
        let mut out = stream.alloc_zeros::<i32>(rows)?;
        self.sample
            .run(stream, &device, n, &mut buffers, &mut out)?;
        Ok(stream.clone_dtoh(&out)?)
    }
}

#[test]
fn greedy_rows_take_the_argmax() -> anyhow::Result<()> {
    let setup = Setup::new()?;
    let misc = Misc::load(setup.gpu.context())?;
    let rows = 8;
    let mut host = Vec::with_capacity(rows * VOCAB);
    for r in 0..rows {
        host.extend(logits(VOCAB, 3.0, 12, r as u64));
    }
    // Ties at the top in row 1, and a NaN in row 2, which argmax_f32 picks.
    let top = host[VOCAB..2 * VOCAB]
        .iter()
        .copied()
        .fold(f32::MIN, f32::max);
    host[VOCAB + 5] = top;
    host[VOCAB + 70_000] = top;
    host[2 * VOCAB + 1234] = f32::NAN;
    let stream = setup.stream();
    let device = stream.clone_htod(&host)?;
    let mut want = stream.alloc_zeros::<i32>(rows)?;
    misc.argmax_f32(stream, &device, &mut want, VOCAB)?;
    let want = stream.clone_dtoh(&want)?;
    let settings: Vec<(Sampling, usize)> = (0..rows).map(|r| (Sampling::GREEDY, r)).collect();
    let got = setup.run(&host, VOCAB, &settings)?;
    assert_eq!(got, want);
    assert_eq!(got[2], 1234);
    Ok(())
}

/// Draws `draws` tokens from rows of `logits` at successive positions and checks them
/// against [`filtered`].
fn check_distribution(
    setup: &Setup,
    logits: &[f32],
    sampling: Sampling,
    draws: usize,
) -> anyhow::Result<()> {
    let n = logits.len();
    let p = filtered(logits, sampling);
    let rows = (4 << 20) / n;
    let rows = rows.clamp(1, 4096);
    let batch: Vec<f32> = logits.iter().copied().cycle().take(rows * n).collect();
    let mut counts = vec![0usize; n];
    let mut done = 0;
    while done < draws {
        let settings: Vec<(Sampling, usize)> =
            (0..rows).map(|r| (sampling, 1000 + done + r)).collect();
        for t in setup.run(&batch, n, &settings)? {
            counts[usize::try_from(t)?] += 1;
        }
        done += rows;
    }
    let dropped: usize = (0..n).filter(|&i| p[i] == 0.0).map(|i| counts[i]).sum();
    // Categories with at least 5 expected draws, the rest pooled.
    let (mut statistic, mut dof) = (0.0, -1.0);
    let (mut pooled_expected, mut pooled_count) = (0.0, 0.0);
    let kept = p.iter().filter(|&&v| v > 0.0).count();
    for i in 0..n {
        if p[i] == 0.0 {
            continue;
        }
        let expected = p[i] * done as f64;
        if expected >= 5.0 {
            statistic += (counts[i] as f64 - expected).powi(2) / expected;
            dof += 1.0;
        } else {
            pooled_expected += expected;
            pooled_count += counts[i] as f64;
        }
    }
    if pooled_expected >= 5.0 {
        statistic += (pooled_count - pooled_expected).powi(2) / pooled_expected;
        dof += 1.0;
    }
    let z = if dof >= 1.0 {
        chi_square_z(statistic, dof)
    } else {
        0.0
    };
    eprintln!(
        "{sampling:?}: {done} draws, {kept} tokens kept, chi-square {statistic:.1} over {dof} \
         degrees of freedom (z {z:.2}), {dropped} dropped tokens drawn"
    );
    assert_eq!(dropped, 0, "drew tokens the filters drop");
    assert!(z < 4.75, "the draws do not fit the filtered distribution");
    Ok(())
}

#[test]
fn draws_fit_the_filtered_distribution() -> anyhow::Result<()> {
    let setup = Setup::new()?;
    let small = logits(1000, 1.5, 12, 7);
    let flat = logits(1000, 1.5, 0, 8);
    let s = |t, k, p, m| Sampling::new(t, k, p, m, 42);
    for sampling in [
        s(1.0, 0, 1.0, 0.0)?,
        s(0.8, 0, 0.95, 0.0)?,
        s(0.7, 20, 1.0, 0.0)?,
        s(1.0, 0, 1.0, 0.05)?,
        s(1.3, 50, 0.9, 0.02)?,
        s(0.5, 1, 1.0, 0.0)?,
        s(1.0, 0, 0.3, 0.0)?,
    ] {
        check_distribution(&setup, &small, sampling, 200_000)?;
    }
    for sampling in [
        s(1.0, 0, 1.0, 0.0)?,
        s(1.0, 0, 0.9, 0.0)?,
        s(0.8, 100, 0.95, 0.1)?,
    ] {
        check_distribution(&setup, &flat, sampling, 200_000)?;
    }
    // The model's vocabulary, with a spread that leaves thousands of tokens likely.
    let wide = logits(VOCAB, 2.0, 12, 11);
    for sampling in [
        s(1.0, 0, 1.0, 0.0)?,
        s(0.8, 0, 0.95, 0.0)?,
        s(0.6, 40, 0.9, 0.0)?,
    ] {
        check_distribution(&setup, &wide, sampling, 20_000)?;
    }
    Ok(())
}

#[test]
fn draws_depend_on_seed_and_position_only() -> anyhow::Result<()> {
    let setup = Setup::new()?;
    let rows = 8;
    let mut host = Vec::with_capacity(rows * VOCAB);
    for r in 0..rows {
        host.extend(logits(VOCAB, 2.0, 12, 100 + r as u64));
    }
    let sampling = Sampling::new(0.9, 0, 0.95, 0.0, 7)?;
    let settings: Vec<(Sampling, usize)> = (0..rows).map(|r| (sampling, 500 + r)).collect();
    let together = setup.run(&host, VOCAB, &settings)?;
    for r in 0..rows {
        let alone = setup.run(&host[r * VOCAB..(r + 1) * VOCAB], VOCAB, &settings[r..=r])?;
        assert_eq!(alone[0], together[r], "row {r}");
    }
    assert_eq!(setup.run(&host, VOCAB, &settings)?, together);
    // Another seed changes some tokens.
    let other: Vec<(Sampling, usize)> = settings
        .iter()
        .map(|&(s, p)| (Sampling { seed: 8, ..s }, p))
        .collect();
    assert_ne!(setup.run(&host, VOCAB, &other)?, together);
    Ok(())
}

/// Prints the time of one launch for 1 and 8 rows of the model's vocabulary, the mean over
/// 200 launches after a warmup, launch overhead included.
#[test]
#[ignore = "a timing, not a check"]
fn sampling_cost() -> anyhow::Result<()> {
    let setup = Setup::new()?;
    let misc = Misc::load(setup.gpu.context())?;
    let stream = setup.stream();
    let mut host = Vec::with_capacity(8 * VOCAB);
    for r in 0..8 {
        host.extend(logits(VOCAB, 3.0, 12, 200 + r));
    }
    let device = stream.clone_htod(&host)?;
    let mut out = stream.alloc_zeros::<i32>(8)?;
    let mut buffers = SampleBuffers::new(stream, 8)?;
    let time = |f: &mut dyn FnMut() -> anyhow::Result<()>| -> anyhow::Result<f64> {
        for _ in 0..20 {
            f()?;
        }
        stream.synchronize()?;
        let started = std::time::Instant::now();
        for _ in 0..200 {
            f()?;
        }
        stream.synchronize()?;
        Ok(started.elapsed().as_secs_f64() * 1e6 / 200.0)
    };
    for rows in [1, 8] {
        let view = device.slice(..rows * VOCAB);
        let us = time(&mut || misc.argmax_f32(stream, &view, &mut out.slice_mut(..rows), VOCAB))?;
        eprintln!("{rows} rows, argmax_f32: {us:.1} us");
        let s = |t, k, p, m| Sampling::new(t, k, p, m, 1);
        for sampling in [
            Sampling::GREEDY,
            s(1.0, 0, 1.0, 0.0)?,
            s(0.8, 0, 0.95, 0.0)?,
            s(1.0, 0, 1.0, 0.05)?,
            s(0.7, 40, 0.95, 0.0)?,
        ] {
            let settings: Vec<(Sampling, usize)> = (0..rows).map(|r| (sampling, r)).collect();
            buffers.upload(stream, &settings)?;
            let us = time(&mut || {
                setup.sample.run(
                    stream,
                    &view,
                    VOCAB,
                    &mut buffers,
                    &mut out.slice_mut(..rows),
                )
            })?;
            eprintln!("{rows} rows, {sampling:?}: {us:.1} us");
        }
    }
    Ok(())
}
