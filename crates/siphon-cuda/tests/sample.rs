//! The sampling kernels (`ops::sample`) against a host version of vLLM's filters.
//!
//! - Greedy rows give `Misc::argmax_f32`'s tokens.
//! - Sampled rows: many rows of the same logits, each with its own position, give tokens
//!   whose counts fit the filtered distribution by a chi-square test, and never a token the
//!   filters drop.
//! - A row's token depends only on its logits, seed and position.
//! - The race's noise and draws match the host port, `siphon::race`, which the draft's
//!   measurements use.

#![cfg(target_os = "linux")]

use std::sync::Arc;

use cudarc::driver::CudaStream;
use siphon::race;
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

/// The kernel's noise against `siphon::race::race_boost`, over the whole vocabulary for a
/// few seeds and positions: equal up to the last bits of the GPU's `logf` and `log1pf`,
/// which are not correctly rounded. Then unfiltered draws at temperatures 1 and 0.8
/// against the host's race over the same logits: the same token unless the host's best
/// two scores are within rounding of each other.
#[test]
fn race_matches_the_host_port() -> anyhow::Result<()> {
    let setup = Setup::new()?;
    let stream = setup.stream();
    let mut noise = stream.alloc_zeros::<f32>(VOCAB)?;
    let (mut equal, mut total, mut worst) = (0usize, 0usize, 0f32);
    for (seed, position) in [
        (0u64, 0u32),
        (5, 1),
        (42, 2790),
        (1 << 32 | 7, 131_071),
        (u64::MAX, u32::MAX),
    ] {
        setup.sample.noise(stream, seed, position, &mut noise)?;
        let gpu = stream.clone_dtoh(&noise)?;
        let key = race::draw_key(seed, position);
        for (i, &g) in (0u32..).zip(&gpu) {
            let h = race::race_boost(key, i);
            equal += usize::from(g.to_bits() == h.to_bits());
            total += 1;
            worst = worst.max((g - h).abs() / h.abs().max(1.0));
        }
    }
    eprintln!(
        "noise: {equal} of {total} values bit-equal, largest difference {worst:.2e} \
         (relative above 1, absolute below)"
    );
    assert!(
        worst <= 8.0 * f32::EPSILON,
        "the host's noise is {worst:.2e} off"
    );

    let rows = 64;
    let seed = 99;
    for temperature in [1.0f32, 0.8] {
        let mut host = Vec::with_capacity(rows * VOCAB);
        for r in 0..rows {
            host.extend(logits(VOCAB, 2.0, 12, 300 + r as u64));
        }
        let sampling = Sampling::new(temperature, 0, 1.0, 0.0, seed)?;
        let settings: Vec<(Sampling, usize)> = (0..rows).map(|r| (sampling, 7000 + r)).collect();
        let got = setup.run(&host, VOCAB, &settings)?;
        let mut near_ties = 0;
        for (r, &token) in got.iter().enumerate() {
            let row = &host[r * VOCAB..(r + 1) * VOCAB];
            let key = race::draw_key(seed, u32::try_from(7000 + r)?);
            let inv_t = 1.0 / temperature;
            let mut scores: Vec<(f32, usize)> = (0u32..)
                .zip(row)
                .map(|(i, &l)| (race::score(l, inv_t, key, i), i as usize))
                .collect();
            scores.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
            let want = race::pick(row, 0..VOCAB, temperature, key);
            assert_eq!(want, Some(scores[0].1));
            if usize::try_from(token)? != scores[0].1 {
                let gap = scores[0].0 - scores[1].0;
                assert!(
                    gap <= 1e-5 * scores[0].0.abs().max(1.0),
                    "temperature {temperature}, row {r}: the kernel drew {token}, the host {} \
                     by {gap}",
                    scores[0].1
                );
                near_ties += 1;
            }
        }
        eprintln!("temperature {temperature}: {rows} draws, {near_ties} differ at near-ties");
    }
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

/// Verifies P0 (`draft_select` & `draft_select_batch` in-register `top_k`/`top_p`/`min_p`
/// filtering + `temp_scale` sharpening + exact `q_probs` output) against host F64 vLLM
/// filtering and `siphon::race`.
#[test]
#[expect(clippy::too_many_lines, reason = "exercises both single and batched draft_select")]
fn draft_select_filter_and_q_probs_exact() -> anyhow::Result<()> {
    use half::bf16;
    use siphon_cuda::{
        ops::draft::{BLOCK, DraftKernels, RACE_WORDS, TOP_K, race_words_sampling},
        weights::draft::RANK,
    };

    let setup = Setup::new()?;
    let stream = setup.stream();
    let draft = DraftKernels::load(setup.gpu.context())?;
    let vocab = 64usize;
    let steps = 4usize;
    let softcap = 30.0f32;

    // Zero predecessor/successor/hp so selector scores equal softcap * tanh(unary / softcap).
    let pred = stream.alloc_zeros::<bf16>(vocab * RANK)?;
    let succ = stream.alloc_zeros::<bf16>(vocab * RANK)?;
    let hp = stream.alloc_zeros::<bf16>((BLOCK - 1) * RANK)?;

    let mut cand_host = vec![0i32; (BLOCK - 1) * TOP_K];
    let mut unary_host = vec![0.0f32; (BLOCK - 1) * TOP_K];
    for e in 0..steps {
        for c in 0..TOP_K {
            cand_host[e * TOP_K + c] = i32::try_from((e * 11 + c) % vocab)?;
            // Include a deliberate tie at c=1,2 on step 1.
            unary_host[e * TOP_K + c] = if e == 1 && (c == 1 || c == 2) {
                3.5
            } else {
                5.0 - c as f32 * 0.45 + (e as f32) * 0.1
            };
        }
    }
    let cand = stream.clone_htod(&cand_host)?;
    let unary = stream.clone_htod(&unary_host)?;
    let control = stream.clone_htod(&[10i32, 1, 0])?;

    let mut ids = stream.alloc_zeros::<i32>(steps)?;
    let mut conf = stream.alloc_zeros::<f32>(steps)?;
    let mut q_probs = stream.alloc_zeros::<f32>((BLOCK - 1) * TOP_K)?;

    let cases = [
        (Sampling::new(0.8, 0, 0.90, 0.05, 42)?, 0.85f32),
        (Sampling::new(1.0, 4, 0.95, 0.0, 77)?, 0.85f32),
        (Sampling::new(0.6, 0, 1.0, 0.10, 99)?, 1.0f32),
        (Sampling::GREEDY, 0.85f32),
    ];

    for (sampling, alpha) in cases {
        let words = race_words_sampling(sampling, alpha);
        let race_dev = stream.clone_htod(&words)?;
        draft.select(
            stream,
            &cand,
            &unary,
            &hp,
            &pred,
            &succ,
            &control,
            steps,
            softcap,
            &mut ids,
            &mut conf,
            Some(&mut q_probs),
            &race_dev,
        )?;
        let gpu_ids = stream.clone_dtoh(&ids)?;
        let gpu_q = stream.clone_dtoh(&q_probs)?;

        // Also check batched select_batch (seqs=2, identical inputs) matches bit-for-bit.
        let mut cand_b = Vec::with_capacity(2 * steps * TOP_K);
        let mut unary_b = Vec::with_capacity(2 * steps * TOP_K);
        for _ in 0..2 {
            cand_b.extend_from_slice(&cand_host[..steps * TOP_K]);
            unary_b.extend_from_slice(&unary_host[..steps * TOP_K]);
        }
        let cand_bd = stream.clone_htod(&cand_b)?;
        let unary_bd = stream.clone_htod(&unary_b)?;
        let hp_bd = stream.alloc_zeros::<bf16>(2 * steps * RANK)?;
        let ctrl_bd = stream.clone_htod(&[10i32, 1, 0, 10, 1, 0])?;
        let mut races_b = Vec::with_capacity(2 * RACE_WORDS);
        races_b.extend_from_slice(&words);
        races_b.extend_from_slice(&words);
        let races_bd = stream.clone_htod(&races_b)?;
        let mut ids_b = stream.alloc_zeros::<i32>(2 * steps)?;
        let mut conf_b = stream.alloc_zeros::<f32>(2 * steps)?;
        let mut q_b = stream.alloc_zeros::<f32>(2 * steps * TOP_K)?;
        draft.select_batch(
            stream,
            &cand_bd,
            &unary_bd,
            &hp_bd,
            &pred,
            &succ,
            &ctrl_bd,
            2,
            steps,
            softcap,
            &mut ids_b,
            &mut conf_b,
            Some(q_b.slice_mut(..)),
            None,
            &races_bd,
        )?;
        let gpu_ids_b = stream.clone_dtoh(&ids_b)?;
        let gpu_q_b = stream.clone_dtoh(&q_b)?;
        assert_eq!(&gpu_ids_b[..steps], &gpu_ids[..]);
        assert_eq!(&gpu_ids_b[steps..], &gpu_ids[..]);
        for i in 0..steps * TOP_K {
            assert!((gpu_q_b[i] - gpu_q[i]).abs() <= 1e-6);
            assert!((gpu_q_b[steps * TOP_K + i] - gpu_q[i]).abs() <= 1e-6);
        }

        for e in 0..steps {
            let scores: Vec<f32> = (0..TOP_K)
                .map(|c| softcap * (unary_host[e * TOP_K + c] / softcap).tanh())
                .collect();
            let q_slice = &gpu_q[e * TOP_K..(e + 1) * TOP_K];
            if sampling.is_greedy() {
                assert_eq!(gpu_ids[e], cand_host[e * TOP_K]);
                assert!((q_slice[0] - 1.0).abs() <= f32::EPSILON);
                assert!(q_slice[1..].iter().all(|&v| v == 0.0));
                continue;
            }
            let eff = Sampling::new(
                sampling.temperature * alpha,
                sampling.top_k,
                sampling.top_p,
                sampling.min_p,
                sampling.seed,
            )?;
            let want_q = filtered(&scores, eff);
            let sum_gpu: f32 = q_slice.iter().sum();
            assert!((sum_gpu - 1.0).abs() < 1e-5, "step {e}: sum={sum_gpu}");
            for c in 0..TOP_K {
                assert!(
                    (f64::from(q_slice[c]) - want_q[c]).abs() < 2e-5,
                    "step {e} c={c} ({sampling:?}, alpha={alpha}): gpu={} want={}",
                    q_slice[c],
                    want_q[c]
                );
            }
            let pos = u32::try_from(10 + 1 + e + 1)?;
            let key = race::draw_key(sampling.seed, pos);
            let inv_t = 1.0 / (sampling.temperature * alpha);
            let mut best = (f32::NEG_INFINITY, usize::MAX);
            for c in 0..TOP_K {
                if want_q[c] > 0.0 {
                    let cid = cand_host[e * TOP_K + c].cast_unsigned();
                    let sc = race::score(scores[c], inv_t, key, cid);
                    if sc > best.0 || (sc.to_bits() == best.0.to_bits() && c < best.1) {
                        best = (sc, c);
                    }
                }
            }
            assert_eq!(
                gpu_ids[e],
                cand_host[e * TOP_K + best.1],
                "step {e} ({sampling:?}, alpha={alpha}) race pick mismatch"
            );
        }
    }
    Ok(())
}

/// Verifies `accept_margin` and `accept_margin_batch` enforce two-sided `|Δlogit| <= margin`:
/// greedy (where target is argmax) behaves identically to before, while sampling rejects a
/// draft whose logit exceeds the sampled non-argmax target by more than `margin`.
#[test]
fn accept_margin_two_sided_prevents_argmax_collapse() -> anyhow::Result<()> {
    use siphon_cuda::ops::spec::{RESULT_LEN, Spec};

    let setup = Setup::new()?;
    let stream = setup.stream();
    let spec = Spec::load(setup.gpu.context())?;
    let vocab = 8usize;
    let m = 4usize;

    let mut logits = vec![0.0f32; m * vocab];
    logits[1] = 7.0;
    logits[2] = 5.0;
    logits[vocab + 3] = 5.0;
    logits[vocab + 4] = 4.95;
    logits[2 * vocab] = 5.0;
    logits[2 * vocab + 5] = 0.0;

    let tokens = stream.clone_htod(&[0i32, 1, 4, 5])?;
    let argmax = stream.clone_htod(&[2i32, 3, 0, 0])?;
    let logits_d = stream.clone_htod(&logits)?;
    let keep = stream.alloc_zeros::<u32>(1)?;
    let mut result = stream.alloc_zeros::<i32>(RESULT_LEN)?;
    let mut anchor = stream.alloc_zeros::<i32>(1)?;

    spec.accept_margin(
        stream, &tokens, &argmax, &logits_d, vocab, &keep, m, 0.1, &mut result, &mut anchor,
    )?;
    let r = stream.clone_dtoh(&result)?;
    assert_eq!(r[0], 0, "two-sided margin must reject +2.0 logit overshoot");

    logits[1] = 5.08;
    logits[vocab + 4] = 4.92;
    let logits_d2 = stream.clone_htod(&logits)?;
    spec.accept_margin(
        stream, &tokens, &argmax, &logits_d2, vocab, &keep, m, 0.1, &mut result, &mut anchor,
    )?;
    let r2 = stream.clone_dtoh(&result)?;
    assert_eq!(
        r2[0], 2,
        "two-sided margin should accept |Δ| <= 0.1 on both sides"
    );

    let mut res_b = stream.alloc_zeros::<i32>(RESULT_LEN)?;
    let mut anc_b = stream.alloc_zeros::<i32>(1)?;
    spec.accept_margin_batch(
        stream,
        &tokens,
        &argmax,
        &logits_d,
        vocab,
        &keep.slice(..),
        1,
        m,
        0.1,
        &mut res_b.slice_mut(..),
        &mut anc_b.slice_mut(..),
    )?;
    assert_eq!(stream.clone_dtoh(&res_b)?[0], 0);
    Ok(())
}

/// Statistical proof that `Sample::run_beta_bf16` + `Spec::accept_beta` preserves the exact
/// filtered target distribution at `beta = 1.0` (0 out-of-support tokens, theoretical
/// acceptance rate, chi-square z-score < 4.0) and matches theoretical relaxed acceptance at
/// `beta = 0.80` with 0 out-of-support tokens and greedy/batch parity.
#[test]
#[expect(clippy::too_many_lines, reason = "8,000-seed statistical verification + greedy check")]
fn beta_rejection_exact_and_relaxed_distribution() -> anyhow::Result<()> {
    use half::bf16;
    use siphon_cuda::ops::{
        sample::BETA_CAND_K,
        spec::{RESULT_LEN, Spec},
    };

    let setup = Setup::new()?;
    let stream = setup.stream();
    let spec = Spec::load(setup.gpu.context())?;

    let vocab = 1024usize;
    let synth_logits = logits(vocab, 1.5, 16, 12345);
    let row_bf16: Vec<bf16> = synth_logits.iter().map(|&v| bf16::from_f32(v)).collect();
    let quantized_f32: Vec<f32> = row_bf16.iter().map(|&b| b.to_f32()).collect();

    let sampling = Sampling::new(0.9, 0, 0.95, 0.05, 1)?;
    let target_p = filtered(&quantized_f32, sampling);

    let mut order: Vec<usize> = (0..vocab).collect();
    order.sort_by(|&a, &b| target_p[b].total_cmp(&target_p[a]));
    let mut cand16 = [0i32; BETA_CAND_K];
    let mut q16 = [0.0f32; BETA_CAND_K];
    for k in 0..14 {
        cand16[k] = i32::try_from(order[k])?;
        let factor = if k % 2 == 0 { 1.6 } else { 0.55 };
        q16[k] = ((target_p[order[k]] * factor) as f32).max(0.01);
    }
    let oos: Vec<usize> = order
        .iter()
        .rev()
        .copied()
        .filter(|&i| target_p[i] == 0.0)
        .take(2)
        .collect();
    assert_eq!(oos.len(), 2);
    cand16[14] = i32::try_from(oos[0])?;
    cand16[15] = i32::try_from(oos[1])?;
    q16[14] = 0.08;
    q16[15] = 0.07;
    let q_sum: f32 = q16.iter().sum();
    for v in &mut q16 {
        *v /= q_sum;
    }

    let batch_seqs = 64usize;
    let width = 2usize;
    let batches = 125usize;
    let total_trials = batch_seqs * batches;

    let mut logits_batch = Vec::with_capacity(batch_seqs * width * vocab);
    for _ in 0..batch_seqs * width {
        logits_batch.extend_from_slice(&row_bf16);
    }
    let logits_d = stream.clone_htod(&logits_batch)?;
    let mut cand_batch = Vec::with_capacity(batch_seqs * BETA_CAND_K);
    let mut q_batch = Vec::with_capacity(batch_seqs * BETA_CAND_K);
    for _ in 0..batch_seqs {
        cand_batch.extend_from_slice(&cand16);
        q_batch.extend_from_slice(&q16);
    }
    let cand_d = stream.clone_htod(&cand_batch)?;
    let q_d = stream.clone_htod(&q_batch)?;
    let keep_d = stream.alloc_zeros::<u32>(batch_seqs)?;
    let mut buffers = SampleBuffers::new(stream, batch_seqs * width)?;
    let mut out_d = stream.alloc_zeros::<i32>(batch_seqs * width)?;
    let mut results_d = stream.alloc_zeros::<i32>(batch_seqs * RESULT_LEN)?;
    let mut anchors_d = stream.alloc_zeros::<i32>(batch_seqs)?;

    for beta in [1.0f32, 0.80f32] {
        let mut counts = vec![0usize; vocab];
        let mut accepted_total = 0usize;
        let mut oos_emitted = 0usize;
        let mut rng_state = 0x1234_5678_9abc_def0u64 ^ u64::from(beta.to_bits());

        for b in 0..batches {
            let mut tokens_h = vec![0i32; batch_seqs * width];
            let mut settings = Vec::with_capacity(batch_seqs * width);
            for s in 0..batch_seqs {
                let u = (splitmix64(&mut rng_state) >> 11) as f64 / ((1u64 << 53) as f64);
                let mut cdf = 0.0f64;
                let mut picked = cand16[0];
                for k in 0..BETA_CAND_K {
                    cdf += f64::from(q16[k]);
                    if u < cdf {
                        picked = cand16[k];
                        break;
                    }
                }
                tokens_h[s * width] = 0;
                tokens_h[s * width + 1] = picked;
                let trial_seed = (b * batch_seqs + s + 1) as u64;
                let row_s = Sampling {
                    seed: trial_seed,
                    ..sampling
                };
                settings.push((row_s, 100));
                settings.push((row_s, 101));
            }
            let tokens_d = stream.clone_htod(&tokens_h)?;
            buffers.upload(stream, &settings)?;
            setup.sample.run_beta_bf16(
                stream,
                &logits_d,
                vocab,
                vocab,
                0.0,
                &tokens_d,
                &keep_d,
                width,
                &cand_d,
                &q_d,
                1,
                beta,
                &mut buffers,
                &mut out_d,
            )?;
            spec.accept_beta_batch(
                stream,
                &tokens_d,
                &out_d,
                &buffers.accept_flags(batch_seqs * width),
                &keep_d.slice(..),
                batch_seqs,
                width,
                &mut results_d.slice_mut(..),
                &mut anchors_d.slice_mut(..),
            )?;
            let res_h = stream.clone_dtoh(&results_d)?;
            for s in 0..batch_seqs {
                let r = &res_h[s * RESULT_LEN..(s + 1) * RESULT_LEN];
                let acc = usize::try_from(r[0])?;
                let emitted_tok = if acc == 1 {
                    accepted_total += 1;
                    usize::try_from(r[2])?
                } else {
                    usize::try_from(r[1 + 8])?
                };
                if target_p[emitted_tok] == 0.0 {
                    oos_emitted += 1;
                }
                counts[emitted_tok] += 1;
            }
        }

        assert_eq!(oos_emitted, 0, "beta={beta}: emitted out-of-support tokens");

        let theory_acc: f64 = (0..BETA_CAND_K)
            .map(|k| {
                let idx = usize::try_from(cand16[k]).unwrap_or(0);
                f64::from(q16[k]).min(target_p[idx] / f64::from(beta))
            })
            .sum();
        let emp_acc = accepted_total as f64 / total_trials as f64;
        let se = (theory_acc * (1.0 - theory_acc) / total_trials as f64).sqrt();
        assert!(
            (emp_acc - theory_acc).abs() <= 4.0 * se,
            "beta={beta}: emp_acc={emp_acc:.4} vs theory={theory_acc:.4} (se={se:.4})"
        );

        if (beta - 1.0).abs() < 1e-6 {
            let mut chi2 = 0.0f64;
            let mut df = 0usize;
            let (mut tail_obs, mut tail_exp) = (0.0f64, 0.0f64);
            for i in 0..vocab {
                let exp = target_p[i] * total_trials as f64;
                if exp >= 5.0 {
                    let d = counts[i] as f64 - exp;
                    chi2 += d * d / exp;
                    df += 1;
                } else if exp > 0.0 {
                    tail_obs += counts[i] as f64;
                    tail_exp += exp;
                }
            }
            if tail_exp >= 1.0 {
                let d = tail_obs - tail_exp;
                chi2 += d * d / tail_exp;
                df += 1;
            }
            let df = (df.saturating_sub(1)).max(1) as f64;
            let z = ((chi2 / df).cbrt() - (1.0 - 2.0 / (9.0 * df))) / (2.0 / (9.0 * df)).sqrt();
            eprintln!(
                "beta=1.0: emp_acc={emp_acc:.4} (theory {theory_acc:.4}), chi2={chi2:.2}, df={df:.0}, z={z:.2}"
            );
            assert!(
                z < 4.0,
                "beta=1.0 distribution mismatch: chi2={chi2:.2}, df={df:.0}, z={z:.2}"
            );
        } else {
            eprintln!("beta={beta:.2}: emp_acc={emp_acc:.4} (theory {theory_acc:.4}), oos=0");
        }
    }

    // Verify greedy row inside run_beta_bf16 + accept_beta works cleanly.
    let greedy_argmax = i32::try_from(order[0])?;
    let mut g_tokens = vec![0i32; width];
    g_tokens[1] = greedy_argmax;
    let g_tok_d = stream.clone_htod(&g_tokens)?;
    let g_logits_d = stream.clone_htod(&[&row_bf16[..], &row_bf16[..]].concat())?;
    let mut g_buf = SampleBuffers::new(stream, width)?;
    g_buf.upload(stream, &[(Sampling::GREEDY, 0), (Sampling::GREEDY, 1)])?;
    let mut g_out = stream.alloc_zeros::<i32>(width)?;
    let mut g_res = stream.alloc_zeros::<i32>(RESULT_LEN)?;
    let mut g_anc = stream.alloc_zeros::<i32>(1)?;
    let g_keep = stream.alloc_zeros::<u32>(1)?;
    setup.sample.run_beta_bf16(
        stream,
        &g_logits_d,
        vocab,
        vocab,
        0.0,
        &g_tok_d,
        &g_keep,
        width,
        &cand_d.slice(..BETA_CAND_K),
        &q_d.slice(..BETA_CAND_K),
        1,
        0.8,
        &mut g_buf,
        &mut g_out,
    )?;
    spec.accept_beta(
        stream,
        &g_tok_d,
        &g_out,
        &g_buf.accept_flags(width),
        &g_keep,
        width,
        &mut g_res,
        &mut g_anc,
    )?;
    assert_eq!(stream.clone_dtoh(&g_res)?[0], 1, "greedy match should accept");
    Ok(())
}

/// Verifies the fast `Z_kept` paths in `sample_prepare_beta_bf16` (`top_p < 1.0, min_p = 0`
/// via `threshold()`'s `kept_mass`, and `top_p = 1.0, min_p = 0` via online softmax) plus
/// multi-row (`width = 4`) `first_reject` early-exit in `sample_race_beta_bf16` (including
/// `keep != 0` forced rows).
#[test]
#[expect(clippy::too_many_lines, reason = "tests both Z_kept fast paths and width=4 first_reject")]
fn beta_fast_paths_z_kept_and_first_reject() -> anyhow::Result<()> {
    use half::bf16;
    use siphon_cuda::ops::{
        sample::BETA_CAND_K,
        spec::{RESULT_LEN, Spec},
    };

    let setup = Setup::new()?;
    let stream = setup.stream();
    let spec = Spec::load(setup.gpu.context())?;

    let vocab = 2048usize;
    let synth_logits = logits(vocab, 1.4, 24, 54321);
    let row_bf16: Vec<bf16> = synth_logits.iter().map(|&v| bf16::from_f32(v)).collect();
    let quantized_f32: Vec<f32> = row_bf16.iter().map(|&b| b.to_f32()).collect();

    let batch_seqs = 32usize;
    let width = 4usize;
    let steps_stride = width - 1;
    let batches = 125usize;
    let total_trials = batch_seqs * batches;

    let mut logits_batch = Vec::with_capacity(batch_seqs * width * vocab);
    for _ in 0..batch_seqs * width {
        logits_batch.extend_from_slice(&row_bf16);
    }
    let logits_d = stream.clone_htod(&logits_batch)?;
    let keep_d = stream.alloc_zeros::<u32>(batch_seqs)?;
    let mut buffers = SampleBuffers::new(stream, batch_seqs * width)?;
    let mut out_d = stream.alloc_zeros::<i32>(batch_seqs * width)?;
    let mut results_d = stream.alloc_zeros::<i32>(batch_seqs * RESULT_LEN)?;
    let mut anchors_d = stream.alloc_zeros::<i32>(batch_seqs)?;

    for (label, sampling) in [
        ("top_p=0.95 (threshold kept_mass)", Sampling::new(0.85, 0, 0.95, 0.0, 1)?),
        ("pure T=0.85 (online softmax)", Sampling::new(0.85, 0, 1.0, 0.0, 1)?),
    ] {
        let target_p = filtered(&quantized_f32, sampling);
        let mut order: Vec<usize> = (0..vocab).collect();
        order.sort_by(|&a, &b| target_p[b].total_cmp(&target_p[a]));

        let mut cand16 = [0i32; BETA_CAND_K];
        let mut q16 = [0.0f32; BETA_CAND_K];
        for k in 0..BETA_CAND_K {
            cand16[k] = i32::try_from(order[k])?;
            let factor = if k % 2 == 0 { 1.4 } else { 0.65 };
            q16[k] = ((target_p[order[k]] * factor) as f32).max(0.005);
        }
        let q_sum: f32 = q16.iter().sum();
        for v in &mut q16 {
            *v /= q_sum;
        }

        let mut cand_batch = Vec::with_capacity(batch_seqs * steps_stride * BETA_CAND_K);
        let mut q_batch = Vec::with_capacity(batch_seqs * steps_stride * BETA_CAND_K);
        for _ in 0..batch_seqs * steps_stride {
            cand_batch.extend_from_slice(&cand16);
            q_batch.extend_from_slice(&q16);
        }
        let cand_d = stream.clone_htod(&cand_batch)?;
        let q_d = stream.clone_htod(&q_batch)?;

        let mut counts_pos0 = vec![0usize; vocab];
        let mut accepted_sum = 0usize;
        let mut bonus_hits = 0usize;
        let mut oos_emitted = 0usize;
        let mut rng_state = 0xfeed_face_cafe_beef_u64 ^ u64::from(sampling.top_p.to_bits());

        for b in 0..batches {
            let mut tokens_h = vec![0i32; batch_seqs * width];
            let mut settings = Vec::with_capacity(batch_seqs * width);
            for s in 0..batch_seqs {
                tokens_h[s * width] = 0;
                for j in 0..steps_stride {
                    let u = (splitmix64(&mut rng_state) >> 11) as f64 / ((1u64 << 53) as f64);
                    let mut cdf = 0.0f64;
                    let mut picked = cand16[0];
                    for k in 0..BETA_CAND_K {
                        cdf += f64::from(q16[k]);
                        if u < cdf {
                            picked = cand16[k];
                            break;
                        }
                    }
                    tokens_h[s * width + 1 + j] = picked;
                }
                let trial_seed = (b * batch_seqs + s + 1) as u64;
                let row_s = Sampling {
                    seed: trial_seed,
                    ..sampling
                };
                for j in 0..width {
                    settings.push((row_s, 200 + j));
                }
            }
            let tokens_d = stream.clone_htod(&tokens_h)?;
            buffers.upload(stream, &settings)?;
            setup.sample.run_beta_bf16(
                stream,
                &logits_d,
                vocab,
                vocab,
                0.0,
                &tokens_d,
                &keep_d,
                width,
                &cand_d,
                &q_d,
                steps_stride,
                1.0,
                &mut buffers,
                &mut out_d,
            )?;
            spec.accept_beta_batch(
                stream,
                &tokens_d,
                &out_d,
                &buffers.accept_flags(batch_seqs * width),
                &keep_d.slice(..),
                batch_seqs,
                width,
                &mut results_d.slice_mut(..),
                &mut anchors_d.slice_mut(..),
            )?;
            let res_h = stream.clone_dtoh(&results_d)?;
            let anc_h = stream.clone_dtoh(&anchors_d)?;
            for s in 0..batch_seqs {
                let r = &res_h[s * RESULT_LEN..(s + 1) * RESULT_LEN];
                let acc = usize::try_from(r[0])?;
                assert!(acc < width);
                accepted_sum += acc;
                if acc == width - 1 {
                    bonus_hits += 1;
                }
                let next_tok = usize::try_from(anc_h[s])?;
                assert_eq!(next_tok, usize::try_from(r[1 + 8 + acc])?);
                if target_p[next_tok] == 0.0 {
                    oos_emitted += 1;
                }
                let pos0_tok = if acc >= 1 {
                    usize::try_from(r[2])?
                } else {
                    usize::try_from(r[1 + 8])?
                };
                if target_p[pos0_tok] == 0.0 {
                    oos_emitted += 1;
                }
                counts_pos0[pos0_tok] += 1;
            }
        }

        assert_eq!(oos_emitted, 0, "{label}: emitted out-of-support token");
        assert!(bonus_hits > 0, "{label}: expected some all-accepted bonus draws");

        let theory_p1: f64 = (0..BETA_CAND_K)
            .map(|k| {
                let idx = usize::try_from(cand16[k]).unwrap_or(0);
                f64::from(q16[k]).min(target_p[idx])
            })
            .sum();
        let theory_mean_acc = theory_p1 + theory_p1.powi(2) + theory_p1.powi(3);
        let emp_mean_acc = accepted_sum as f64 / total_trials as f64;
        assert!(
            (emp_mean_acc - theory_mean_acc).abs() < 0.08,
            "{label}: emp_mean_acc={emp_mean_acc:.4} vs theory={theory_mean_acc:.4}"
        );

        let mut chi2 = 0.0f64;
        let mut df = 0usize;
        let (mut tail_obs, mut tail_exp) = (0.0f64, 0.0f64);
        for i in 0..vocab {
            let exp = target_p[i] * total_trials as f64;
            if exp >= 5.0 {
                let d = counts_pos0[i] as f64 - exp;
                chi2 += d * d / exp;
                df += 1;
            } else if exp > 0.0 {
                tail_obs += counts_pos0[i] as f64;
                tail_exp += exp;
            }
        }
        if tail_exp >= 1.0 {
            let d = tail_obs - tail_exp;
            chi2 += d * d / tail_exp;
            df += 1;
        }
        let df = (df.saturating_sub(1)).max(1) as f64;
        let z = ((chi2 / df).cbrt() - (1.0 - 2.0 / (9.0 * df))) / (2.0 / (9.0 * df)).sqrt();
        eprintln!(
            "{label}: width=4 mean_acc={emp_mean_acc:.4} (theory {theory_mean_acc:.4}), bonus={bonus_hits}, z={z:.2}"
        );
        assert!(z < 4.0, "{label}: distribution mismatch z={z:.2}");
    }

    // Verify forced `keep = 2` (Accept::Rows(2)) draws row 1 even when row 0 is rejected.
    let keep_forced = stream.clone_htod(&[2u32])?;
    let tok_forced = stream.clone_htod(&[0i32, 0, 0, 0])?;
    let logits_1 = stream.clone_htod(&[&row_bf16[..], &row_bf16[..], &row_bf16[..], &row_bf16[..]].concat())?;
    let mut cand_1 = vec![0i32; steps_stride * BETA_CAND_K];
    let mut q_1 = vec![0.0f32; steps_stride * BETA_CAND_K];
    let sampling_p = Sampling::new(0.85, 0, 0.95, 0.0, 77)?;
    let target_p = filtered(&quantized_f32, sampling_p);
    let oos_tok = target_p
        .iter()
        .position(|&p| p == 0.0)
        .ok_or_else(|| anyhow::anyhow!("expected out-of-support token"))?;
    for step in 0..steps_stride {
        cand_1[step * BETA_CAND_K] = i32::try_from(oos_tok)?;
        q_1[step * BETA_CAND_K] = 1.0;
    }
    let cand_1_d = stream.clone_htod(&cand_1)?;
    let q_1_d = stream.clone_htod(&q_1)?;
    let mut b1 = SampleBuffers::new(stream, width)?;
    b1.upload(
        stream,
        &(0..width).map(|j| (sampling_p, 50 + j)).collect::<Vec<_>>(),
    )?;
    let mut out_1 = stream.alloc_zeros::<i32>(width)?;
    let mut res_1 = stream.alloc_zeros::<i32>(RESULT_LEN)?;
    let mut anc_1 = stream.alloc_zeros::<i32>(1)?;
    setup.sample.run_beta_bf16(
        stream,
        &logits_1,
        vocab,
        vocab,
        0.0,
        &tok_forced,
        &keep_forced,
        width,
        &cand_1_d,
        &q_1_d,
        steps_stride,
        1.0,
        &mut b1,
        &mut out_1,
    )?;
    spec.accept_beta(
        stream,
        &tok_forced,
        &out_1,
        &b1.accept_flags(width),
        &keep_forced,
        width,
        &mut res_1,
        &mut anc_1,
    )?;
    let anc_val = usize::try_from(stream.clone_dtoh(&anc_1)?[0])?;
    assert!(
        target_p[anc_val] > 0.0,
        "forced keep=2 must draw valid in-support token for row 1, got {anc_val}"
    );
    Ok(())
}

/// Statistical proof that 2-branch without-replacement beta-residual tree verification
/// (`Sample::run_beta_tree_bf16` + `Spec::accept_tree_beta`) preserves the exact filtered
/// target distribution at `beta = 1.0` (0 out-of-support tokens, primary & secondary
/// acceptance matching analytical theory, chi-square z-score well within `±3`) and matches
/// analytical theory at `beta = 0.80` with 0 out-of-support tokens. Also checks
/// `draft_select_top2` / `draft_select_branch` `q_probs` output and greedy fallback.
#[test]
#[expect(clippy::too_many_lines, reason = "8,000-seed 2-branch tree chi-square + selector check")]
fn beta_tree_two_branch_without_replacement_exact_and_relaxed() -> anyhow::Result<()> {
    use half::bf16;
    use siphon::tree::{SpecTree, sample_without_replacement, secondary_draw_uniform};
    use siphon_cuda::{
        ops::{
            draft::{BLOCK, DraftKernels, TOP_K, race_words_sampling},
            sample::BETA_CAND_K,
            spec::{RESULT_LEN, Spec, VERIFY_ROWS},
        },
        weights::draft::RANK,
    };

    let setup = Setup::new()?;
    let stream = setup.stream();
    let spec = Spec::load(setup.gpu.context())?;
    let draft_k = DraftKernels::load(setup.gpu.context())?;

    // 1. Verify `draft_select_top2` writes identical `q_probs` to `draft_select` and
    //    `draft_select_branch` writes normalized `branch_q_probs` for continuation steps.
    {
        let svocab = 64usize;
        let steps = 4usize;
        let softcap = 30.0f32;
        let pred = stream.alloc_zeros::<bf16>(svocab * RANK)?;
        let succ = stream.alloc_zeros::<bf16>(svocab * RANK)?;
        let hp = stream.alloc_zeros::<bf16>((BLOCK - 1) * RANK)?;
        let mut cand_h = vec![0i32; (BLOCK - 1) * TOP_K];
        let mut unary_h = vec![0.0f32; (BLOCK - 1) * TOP_K];
        for e in 0..steps {
            for c in 0..TOP_K {
                cand_h[e * TOP_K + c] = i32::try_from((e * 13 + c) % svocab)?;
                unary_h[e * TOP_K + c] = 4.0 - c as f32 * 0.35 + (e as f32) * 0.05;
            }
        }
        let cand_d = stream.clone_htod(&cand_h)?;
        let unary_d = stream.clone_htod(&unary_h)?;
        let control = stream.clone_htod(&[10i32, 1, 0])?;
        let sampling = Sampling::new(0.8, 0, 0.95, 0.05, 42)?;
        let race_dev = stream.clone_htod(&race_words_sampling(sampling, 0.85))?;

        let mut ids1 = stream.alloc_zeros::<i32>(steps)?;
        let mut conf1 = stream.alloc_zeros::<f32>(steps)?;
        let mut q1 = stream.alloc_zeros::<f32>((BLOCK - 1) * TOP_K)?;
        draft_k.select(
            stream,
            &cand_d,
            &unary_d,
            &hp,
            &pred,
            &succ,
            &control,
            steps,
            softcap,
            &mut ids1,
            &mut conf1,
            Some(&mut q1),
            &race_dev,
        )?;

        let mut ids2 = stream.alloc_zeros::<i32>(steps)?;
        let mut conf2 = stream.alloc_zeros::<f32>(steps)?;
        let mut alt_ids = stream.alloc_zeros::<i32>(steps)?;
        let mut alt_conf = stream.alloc_zeros::<f32>(steps)?;
        let mut q2 = stream.alloc_zeros::<f32>((BLOCK - 1) * TOP_K)?;
        draft_k.select_top2(
            stream,
            &cand_d,
            &unary_d,
            &hp,
            &pred,
            &succ,
            &control,
            steps,
            softcap,
            &mut ids2,
            &mut conf2,
            &mut alt_ids,
            &mut alt_conf,
            Some(&mut q2),
            &race_dev,
        )?;
        assert_eq!(stream.clone_dtoh(&ids1)?, stream.clone_dtoh(&ids2)?);
        let q1_h = stream.clone_dtoh(&q1)?;
        let q2_h = stream.clone_dtoh(&q2)?;
        for i in 0..steps * TOP_K {
            assert!(
                (q1_h[i] - q2_h[i]).abs() <= 1e-6,
                "select_top2 q_probs[{i}] {} != select {}",
                q2_h[i],
                q1_h[i]
            );
        }

        // Check `select_branch` writes normalized `branch_q_probs` for t = 1..k2.
        let mut br_ids = stream.alloc_zeros::<i32>(steps)?;
        let mut br_q = stream.alloc_zeros::<f32>(steps * TOP_K)?;
        draft_k.select_branch(
            stream,
            &cand_d,
            &unary_d,
            &hp,
            &pred,
            &succ,
            &control,
            0,
            cand_h[1],
            3,
            steps,
            softcap,
            &mut br_ids,
            Some(&mut br_q),
            &race_dev,
        )?;
        let br_ids_h = stream.clone_dtoh(&br_ids)?;
        let br_q_h = stream.clone_dtoh(&br_q)?;
        assert_eq!(br_ids_h[0], cand_h[1]);
        for t in 1..3 {
            let slice = &br_q_h[t * TOP_K..(t + 1) * TOP_K];
            let sum: f32 = slice.iter().sum();
            assert!((sum - 1.0).abs() < 1e-5, "branch step {t}: sum={sum}");
            // With zero pred/succ/hp, branch scores equal primary scores at step `t`.
            for c in 0..TOP_K {
                assert!(
                    (slice[c] - q1_h[t * TOP_K + c]).abs() <= 1e-6,
                    "branch step {t} c={c}: {} != {}",
                    slice[c],
                    q1_h[t * TOP_K + c]
                );
            }
        }
    }

    // 2. 8,000-seed statistical proof for the 2-branch without-replacement branch node.
    let vocab = 1024usize;
    let synth_logits = logits(vocab, 1.4, 16, 54321);
    let row_bf16: Vec<bf16> = synth_logits.iter().map(|&v| bf16::from_f32(v)).collect();
    let quantized_f32: Vec<f32> = row_bf16.iter().map(|&b| b.to_f32()).collect();

    let sampling = Sampling::new(0.9, 0, 0.95, 0.05, 1)?;
    let target_p = filtered(&quantized_f32, sampling);

    let mut order: Vec<usize> = (0..vocab).collect();
    order.sort_by(|&a, &b| target_p[b].total_cmp(&target_p[a]));
    let mut cand16 = [0i32; BETA_CAND_K];
    let mut q16 = [0.0f32; BETA_CAND_K];
    for k in 0..14 {
        cand16[k] = i32::try_from(order[k])?;
        // Deliberately skew q vs p so both primary rejection and secondary acceptance happen often.
        let factor = if k % 2 == 0 { 1.8 } else { 0.45 };
        q16[k] = ((target_p[order[k]] * factor) as f32).max(0.015);
    }
    let oos: Vec<usize> = order
        .iter()
        .rev()
        .copied()
        .filter(|&i| target_p[i] == 0.0)
        .take(2)
        .collect();
    assert_eq!(oos.len(), 2);
    cand16[14] = i32::try_from(oos[0])?;
    cand16[15] = i32::try_from(oos[1])?;
    q16[14] = 0.08;
    q16[15] = 0.07;
    let q_sum: f32 = q16.iter().sum();
    for v in &mut q16 {
        *v /= q_sum;
    }

    // 3-node 2-branch tree: root 0 -> primary 1, secondary 2 (both leaves).
    let tree = SpecTree::two_branch(1, 0, 1)?;
    let m = tree.num_nodes; // 3
    let trials = 8_000usize;

    let mut logits_3 = Vec::with_capacity(m * vocab);
    for _ in 0..m {
        logits_3.extend_from_slice(&row_bf16);
    }
    let logits_d = stream.clone_htod(&logits_3)?;

    let mut tree_cand_h = vec![-1i32; m * BETA_CAND_K];
    let mut tree_q_h = vec![0.0f32; m * BETA_CAND_K];
    tree_cand_h[..BETA_CAND_K].copy_from_slice(&cand16);
    tree_q_h[..BETA_CAND_K].copy_from_slice(&q16);
    let tree_cand_d = stream.clone_htod(&tree_cand_h)?;
    let tree_q_d = stream.clone_htod(&tree_q_h)?;
    let primary_d = stream.clone_htod(&tree.primary_child)?;
    let secondary_d = stream.clone_htod(&tree.secondary_child)?;
    let forced_path_d = stream.alloc_zeros::<i32>(VERIFY_ROWS)?;
    let forced_len_d = stream.alloc_zeros::<u32>(1)?;

    // Batch 64 independent root-node trials into one `run_beta_tree_bf16` launch by giving
    // each trial 3 rows? No: `run_beta_tree_bf16` indexes `tokens[c1]` and `tokens[c2]`
    // directly! We can pack 64 independent 3-node mini-trees into 192 rows with shifted
    // `primary_child` / `secondary_child` indices (`3*s + 1`, `3*s + 2`) for fast 8,000-seed
    // execution!
    let batch_trees = 64usize;
    let total_rows = batch_trees * m;
    let batches = trials / batch_trees;
    assert_eq!(batches * batch_trees, trials);

    let mut b_logits = Vec::with_capacity(total_rows * vocab);
    let mut b_primary = vec![-1i32; total_rows];
    let mut b_secondary = vec![-1i32; total_rows];
    let mut b_cand = vec![-1i32; total_rows * BETA_CAND_K];
    let mut b_q = vec![0.0f32; total_rows * BETA_CAND_K];
    for s in 0..batch_trees {
        let base = s * m;
        for _ in 0..m {
            b_logits.extend_from_slice(&row_bf16);
        }
        b_primary[base] = i32::try_from(base + 1)?;
        b_secondary[base] = i32::try_from(base + 2)?;
        b_cand[base * BETA_CAND_K..(base + 1) * BETA_CAND_K].copy_from_slice(&cand16);
        b_q[base * BETA_CAND_K..(base + 1) * BETA_CAND_K].copy_from_slice(&q16);
    }
    let b_logits_d = stream.clone_htod(&b_logits)?;
    let b_primary_d = stream.clone_htod(&b_primary)?;
    let b_secondary_d = stream.clone_htod(&b_secondary)?;
    let b_cand_d = stream.clone_htod(&b_cand)?;
    let b_q_d = stream.clone_htod(&b_q)?;
    let mut buffers = SampleBuffers::new(stream, total_rows)?;
    let mut out_d = stream.alloc_zeros::<i32>(total_rows)?;

    for beta in [1.0f32, 0.80f32] {
        let mut counts = vec![0usize; vocab];
        let mut acc1_total = 0usize;
        let mut acc2_total = 0usize;
        let mut oos_emitted = 0usize;
        let mut rng_state = 0xfeed_face_cafe_beefu64 ^ u64::from(beta.to_bits());

        for b in 0..batches {
            let mut tokens_h = vec![0i32; total_rows];
            let mut settings = Vec::with_capacity(total_rows);
            for s in 0..batch_trees {
                let trial_seed = (b * batch_trees + s + 1) as u64;
                // Draw x1 ~ q16.
                let u1_draw = (splitmix64(&mut rng_state) >> 11) as f64 / ((1u64 << 53) as f64);
                let mut cdf = 0.0f64;
                let mut x1 = cand16[0];
                for k in 0..BETA_CAND_K {
                    cdf += f64::from(q16[k]);
                    if u1_draw < cdf {
                        x1 = cand16[k];
                        break;
                    }
                }
                // Draw x2 ~ q2(. | x1) without replacement.
                let u_sec = secondary_draw_uniform(trial_seed, 100);
                let x2 = sample_without_replacement(&cand16, &q16, x1, u_sec)
                    .expect("non-degenerate q16 has secondary support");
                assert_ne!(x1, x2);

                let base = s * m;
                tokens_h[base] = 0;
                tokens_h[base + 1] = x1;
                tokens_h[base + 2] = x2;
                let row_s = Sampling {
                    seed: trial_seed,
                    ..sampling
                };
                settings.push((row_s, 100));
                settings.push((row_s, 101));
                settings.push((row_s, 101));
            }
            let tokens_d = stream.clone_htod(&tokens_h)?;
            buffers.upload(stream, &settings)?;
            setup.sample.run_beta_tree_bf16(
                stream,
                &b_logits_d,
                vocab,
                vocab,
                0.0,
                &tokens_d,
                &b_primary_d,
                &b_secondary_d,
                &b_cand_d,
                &b_q_d,
                beta,
                &mut buffers,
                &mut out_d,
            )?;
            let flags_h = stream.clone_dtoh(&buffers.accept_flags(total_rows))?;
            let out_h = stream.clone_dtoh(&out_d)?;
            for s in 0..batch_trees {
                let base = s * m;
                let flag = flags_h[base];
                let emitted = match flag {
                    1 => {
                        acc1_total += 1;
                        assert_eq!(out_h[base], tokens_h[base + 1]);
                        usize::try_from(tokens_h[base + 1])?
                    }
                    2 => {
                        acc2_total += 1;
                        assert_eq!(out_h[base], tokens_h[base + 2]);
                        usize::try_from(tokens_h[base + 2])?
                    }
                    0 => usize::try_from(out_h[base])?,
                    other => anyhow::bail!("unexpected accept_flag {other}"),
                };
                if target_p[emitted] == 0.0 {
                    oos_emitted += 1;
                }
                counts[emitted] += 1;
            }
        }

        assert_eq!(oos_emitted, 0, "beta={beta}: emitted out-of-support tokens");

        // Analytical theory for primary acceptance and without-replacement secondary acceptance.
        let mut r1 = vec![0.0f64; vocab];
        let mut z1 = 0.0f64;
        for y in 0..vocab {
            let mut qy = 0.0f64;
            for k in 0..BETA_CAND_K {
                if usize::try_from(cand16[k]).ok() == Some(y) {
                    qy = f64::from(q16[k]);
                    break;
                }
            }
            r1[y] = (target_p[y] - f64::from(beta) * qy).max(0.0);
            z1 += r1[y];
        }
        for v in &mut r1 {
            *v /= z1;
        }
        let mut theory_acc1 = 0.0f64;
        let mut theory_acc2 = 0.0f64;
        for k1 in 0..BETA_CAND_K {
            let x1 = usize::try_from(cand16[k1])?;
            let qx1 = f64::from(q16[k1]);
            let a1 = (target_p[x1] / (f64::from(beta) * qx1)).min(1.0);
            theory_acc1 += qx1 * a1;
            let p_rej1 = qx1 * (1.0 - a1);
            let rem = 1.0 - qx1;
            if p_rej1 > 0.0 && rem > 0.0 {
                let mut acc2_given_x1 = 0.0f64;
                for k2 in 0..BETA_CAND_K {
                    if k2 == k1 {
                        continue;
                    }
                    let x2 = usize::try_from(cand16[k2])?;
                    let q2_x2 = f64::from(q16[k2]) / rem;
                    let a2 = (r1[x2] / (f64::from(beta) * q2_x2)).min(1.0);
                    acc2_given_x1 += q2_x2 * a2;
                }
                theory_acc2 += p_rej1 * acc2_given_x1;
            }
        }
        let emp_acc1 = acc1_total as f64 / trials as f64;
        let emp_acc2 = acc2_total as f64 / trials as f64;
        let se1 = (theory_acc1 * (1.0 - theory_acc1) / trials as f64).sqrt();
        let se2 = (theory_acc2 * (1.0 - theory_acc2) / trials as f64).sqrt();
        assert!(
            (emp_acc1 - theory_acc1).abs() <= 4.0 * se1,
            "beta={beta}: emp_acc1={emp_acc1:.4} vs theory={theory_acc1:.4} (se={se1:.4})"
        );
        assert!(
            acc2_total > 0 && (emp_acc2 - theory_acc2).abs() <= 4.0 * se2,
            "beta={beta}: emp_acc2={emp_acc2:.4} vs theory={theory_acc2:.4} (se={se2:.4})"
        );

        if (beta - 1.0).abs() < 1e-6 {
            let mut chi2 = 0.0f64;
            let mut df = 0usize;
            let (mut tail_obs, mut tail_exp) = (0.0f64, 0.0f64);
            for i in 0..vocab {
                let exp = target_p[i] * trials as f64;
                if exp >= 5.0 {
                    let d = counts[i] as f64 - exp;
                    chi2 += d * d / exp;
                    df += 1;
                } else if exp > 0.0 {
                    tail_obs += counts[i] as f64;
                    tail_exp += exp;
                }
            }
            if tail_exp >= 1.0 {
                let d = tail_obs - tail_exp;
                chi2 += d * d / tail_exp;
                df += 1;
            }
            let df = (df.saturating_sub(1)).max(1) as f64;
            let z = ((chi2 / df).cbrt() - (1.0 - 2.0 / (9.0 * df))) / (2.0 / (9.0 * df)).sqrt();
            eprintln!(
                "beta_tree beta=1.0: acc1={emp_acc1:.4} (theory {theory_acc1:.4}), \
                 acc2={emp_acc2:.4} (theory {theory_acc2:.4}), chi2={chi2:.2}, df={df:.0}, z={z:.2}, oos=0"
            );
            assert!(
                z.abs() < 3.5,
                "beta_tree beta=1.0 distribution mismatch: chi2={chi2:.2}, df={df:.0}, z={z:.2}"
            );
        } else {
            eprintln!(
                "beta_tree beta={beta:.2}: acc1={emp_acc1:.4} (theory {theory_acc1:.4}), \
                 acc2={emp_acc2:.4} (theory {theory_acc2:.4}), oos=0"
            );
        }
    }

    // 3. Verify `accept_tree_beta` walks primary, secondary, rejection, and greedy correctly.
    let mut single_buf = SampleBuffers::new(stream, m)?;
    let mut single_out = stream.alloc_zeros::<i32>(m)?;
    let mut res_d = stream.alloc_zeros::<i32>(RESULT_LEN)?;
    let mut path_d = stream.alloc_zeros::<i32>(VERIFY_ROWS)?;
    let mut anc_d = stream.alloc_zeros::<i32>(1)?;

    // Greedy secondary match: primary != argmax, secondary == argmax -> path [0, 2]!
    let greedy_top = i32::try_from(order[0])?;
    let greedy_alt = i32::try_from(order[1])?;
    let g_tokens_d = stream.clone_htod(&[0i32, greedy_alt, greedy_top])?;
    single_buf.upload(
        stream,
        &[
            (Sampling::GREEDY, 0),
            (Sampling::GREEDY, 1),
            (Sampling::GREEDY, 1),
        ],
    )?;
    setup.sample.run_beta_tree_bf16(
        stream,
        &logits_d,
        vocab,
        vocab,
        0.0,
        &g_tokens_d,
        &primary_d,
        &secondary_d,
        &tree_cand_d,
        &tree_q_d,
        1.0,
        &mut single_buf,
        &mut single_out,
    )?;
    spec.accept_tree_beta(
        stream,
        &g_tokens_d,
        &single_out,
        &single_buf.accept_flags(m),
        &primary_d,
        &secondary_d,
        &forced_path_d,
        &forced_len_d,
        m,
        &mut res_d,
        &mut path_d,
        &mut anc_d,
    )?;
    assert_eq!(stream.clone_dtoh(&res_d)?[0], 1, "greedy secondary should accept");
    assert_eq!(&stream.clone_dtoh(&path_d)?[..2], &[0, 2]);
    Ok(())
}
