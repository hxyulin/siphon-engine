//! The verify pass of speculative decoding (`Model::verify`): its feature taps, its
//! agreement with stepping, its rollback, the sliding ring size, and row invariance.
//!
//! Needs the checkpoint and about 20 GB of GPU memory, so every test is ignored by
//! default. Each test loads its own model, so run them one at a time:
//!
//! ```sh
//! SIPHON_MODEL=<model-dir> cargo test --release -p siphon-cuda --test verify -- \
//!     --ignored --nocapture --test-threads 1
//! ```
//!
//! The prompts are the tokens of `reference/out/anyhow-chunked.safetensors` (or
//! `SIPHON_CAPTURE`). Each check runs at a position before the sliding-window wrap at
//! 1024, at one whose rows cross it, and at 2500.
//!
//! - `taps_match_mlx` (test A) compares the taps of prefill, and of plain steps
//!   (`Model::step_with_taps`) for the last positions against the prefill's there, with
//!   the MLX capture of `reference/capture.py --taps` in `SIPHON_TAPS`.
//! - `verify_matches_stepping` (test B): a verify of m rows against m steps, within the
//!   tolerances of `tests/prefill.rs`.
//! - `rollback_matches_shorter_verify` (test C): a verify of 8 rows whose rows after `a`
//!   hold wrong tokens, rolled back to `a`, then 20 steps, against a verify of rows 0..=a
//!   and the same steps. Every logit and the caches must match bit for bit.
//! - `ring_size` (test D): rings of 1032 slots against 1024, decode and prefill.
//! - `verify_rows_are_invariant` (test E): every width from 1 to 8 gives the same logits
//!   for each shared row, bit for bit, and so do rows 0..=j when the rows after j change,
//!   and 8 verifies of one row each.

#![cfg(target_os = "linux")]

use std::{env, fs, path::PathBuf};

use anyhow::{Context as _, ensure};
use half::bf16;
use safetensors::{Dtype, SafeTensors};
use siphon::{Checkpoint, TextConfig};
use siphon_cuda::{
    Gpu,
    model::{
        Accept, DEFAULT_PREFILL_CHUNK, Model, PrefillLogits, SavedState, VerifyTokens,
        VisitSnapshot,
    },
    weights::Weights,
};

/// Prefix lengths: rows of a verify there stay below the window wrap, cross it, or run
/// well past it.
const PREFIXES: [usize; 3] = [600, 1020, 2500];
const ROWS: usize = 8;
const MAX_STATE_ERROR: f64 = 1.0 / 128.0;
const MAX_MEAN_KL: f64 = 4e-3;
const MAX_KL: f64 = 0.12;
/// Steps after the rollback in test C.
const AFTER_ROLLBACK: usize = 20;

fn capture_tokens() -> anyhow::Result<Vec<u32>> {
    let path = env::var_os("SIPHON_CAPTURE").map_or_else(
        || {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../reference/out/anyhow-chunked.safetensors")
        },
        PathBuf::from,
    );
    read_tokens(&path)
}

fn read_tokens(path: &std::path::Path) -> anyhow::Result<Vec<u32>> {
    let bytes = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let capture = SafeTensors::deserialize(&bytes)?;
    let view = capture.tensor("tokens")?;
    ensure!(view.dtype() == Dtype::I32, "tokens are {:?}", view.dtype());
    view.data()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| Ok(u32::try_from(i32::from_le_bytes(*b))?))
        .collect()
}

fn load(capacity: usize) -> anyhow::Result<(Gpu, Model)> {
    let model_dir = PathBuf::from(env::var_os("SIPHON_MODEL").context("set SIPHON_MODEL")?);
    let gpu = Gpu::new(0)?;
    let config = TextConfig::load(&model_dir)?;
    let passes = config.loop_config.num_passes;
    let checkpoint = Checkpoint::open(&model_dir)?;
    let weights = Weights::load(&checkpoint, &config, gpu.stream())?;
    let model = Model::new(&gpu, config, weights, passes, capacity)?;
    Ok((gpu, model))
}

/// Resets the model, prefills `prompt` and saves the state. Each comparison restores this
/// state, so that it starts from the same caches without prefilling again.
fn prefill_state(model: &mut Model, prompt: &[u32]) -> anyhow::Result<SavedState> {
    model.reset()?;
    model.prefill(prompt, DEFAULT_PREFILL_CHUNK, PrefillLogits::Last)?;
    model.save_state()
}

/// KL(want || got) over softmax of the logits.
fn kl(want: &[f32], got: &[f32]) -> f64 {
    let log_softmax = |x: &[f32]| {
        let max = x.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let sum: f64 = x.iter().map(|&v| f64::from(v - max).exp()).sum();
        let log_sum = sum.ln();
        x.iter()
            .map(|&v| f64::from(v - max) - log_sum)
            .collect::<Vec<_>>()
    };
    let (p, q) = (log_softmax(want), log_softmax(got));
    p.iter().zip(&q).map(|(lp, lq)| lp.exp() * (lp - lq)).sum()
}

fn identical(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

/// Relative RMS error of `got` against `want`.
fn relative_error(want: &[f32], got: &[f32]) -> f64 {
    let (mut diff, mut norm) = (0.0, 0.0);
    for (&w, &g) in want.iter().zip(got) {
        diff += f64::from(w - g).powi(2);
        norm += f64::from(w).powi(2);
    }
    (diff / norm.max(f64::MIN_POSITIVE)).sqrt()
}

/// Each visit's worst relative error over its caches and conv states, and whether all
/// are bit-identical.
fn compare_states(want: &[VisitSnapshot], got: &[VisitSnapshot]) -> (Vec<f64>, bool) {
    let mut same = want.len() == got.len();
    let errors = want
        .iter()
        .zip(got)
        .map(|(w, g)| {
            let pairs = [(&w.keys, &g.keys), (&w.values, &g.values)]
                .into_iter()
                .chain(w.conv.iter().zip(&g.conv));
            let mut worst = 0.0f64;
            for (a, b) in pairs {
                worst = worst.max(relative_error(a, b));
                same &= identical(a, b);
            }
            worst
        })
        .collect();
    (errors, same)
}

/// `count` tokens that differ from `avoid` at each row, spread over the vocabulary.
fn wrong_tokens(avoid: &[u32], salt: u32, vocab: u32) -> Vec<u32> {
    avoid
        .iter()
        .enumerate()
        .map(|(i, &t)| {
            let i = u32::try_from(i).expect("few rows");
            let mut w = (t ^ 0x5bd1)
                .wrapping_mul(2_654_435_761)
                .wrapping_add(salt * 7919 + i)
                % vocab;
            if w == t {
                w = (w + 1) % vocab;
            }
            w
        })
        .collect()
}

#[test]
#[ignore = "needs the checkpoint (SIPHON_MODEL), SIPHON_TAPS and about 20 GB of GPU memory"]
fn taps_match_mlx() -> anyhow::Result<()> {
    let path = PathBuf::from(env::var_os("SIPHON_TAPS").context("set SIPHON_TAPS")?);
    let bytes = fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
    let capture = SafeTensors::deserialize(&bytes)?;
    let tokens = read_tokens(&path)?;
    let view = capture.tensor("taps")?;
    ensure!(view.dtype() == Dtype::BF16, "taps are {:?}", view.dtype());
    let want: Vec<f32> = view
        .data()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| bf16::from_le_bytes(*b).to_f32())
        .collect();
    let hidden = 2304;
    ensure!(want.len() == tokens.len() * 4 * hidden, "taps shape");

    let (gpu, mut model) = load(tokens.len() + 16)?;
    let stream = gpu.stream().clone();
    let copy_into = |got: &mut [f32], start: usize, taps: cudarc::driver::CudaView<'_, bf16>| {
        let host = stream.clone_dtoh(&taps)?;
        for (dst, v) in got[start * 4 * hidden..].iter_mut().zip(host) {
            *dst = v.to_f32();
        }
        anyhow::Ok(())
    };
    // MLX ran the text in chunks of 512.
    let mut prefilled = vec![0f32; want.len()];
    model.prefill_with_taps(&tokens, 512, PrefillLogits::Last, &mut |start, taps| {
        copy_into(&mut prefilled, start, taps)
    })?;
    // Again with the last STEPPED positions as plain steps.
    let split = tokens.len() - STEPPED;
    let mut stepped = vec![0f32; want.len()];
    model.reset()?;
    model.prefill_with_taps(
        &tokens[..split],
        512,
        PrefillLogits::Last,
        &mut |start, taps| copy_into(&mut stepped, start, taps),
    )?;
    for (p, &token) in tokens.iter().enumerate().skip(split) {
        model.step_with_taps(token)?;
        copy_into(&mut stepped, p, model.step_taps()?)?;
    }

    // Per tap: the prefill's relative RMS error and worst cosine over every position, and
    // over the stepped positions the steps' error against the prefill's there.
    let mut failures = Vec::new();
    for tap in 0..4 {
        let (error, worst) = tap_error(&want, &prefilled, tap, 0..tokens.len());
        println!(
            "tap {tap}, prefill: relative RMS error {error:.3e}, worst position {} with cosine \
             {:.6}",
            worst.1, worst.0
        );
        if error > TAP_ERROR {
            failures.push(format!("prefill tap {tap}: {error:.3e}"));
        }
        let (step_error, _) = tap_error(&want, &stepped, tap, split..tokens.len());
        let (prefill_error, _) = tap_error(&want, &prefilled, tap, split..tokens.len());
        println!(
            "tap {tap}, last {STEPPED} positions: plain steps {step_error:.3e}, prefill \
             {prefill_error:.3e}"
        );
        if step_error > STEP_TAP_RATIO * prefill_error.max(TAP_FLOOR) {
            failures.push(format!(
                "stepped tap {tap}: {step_error:.3e} against prefill {prefill_error:.3e}"
            ));
        }
    }
    ensure!(failures.is_empty(), "taps differ from MLX: {failures:?}");
    Ok(())
}

/// The relative RMS error of tap `tap` of `got` against `want` over `positions`, and the
/// worst position's cosine similarity and position.
fn tap_error(
    want: &[f32],
    got: &[f32],
    tap: usize,
    positions: std::ops::Range<usize>,
) -> (f64, (f64, usize)) {
    let hidden = 2304;
    let (mut want_tap, mut got_tap) = (Vec::new(), Vec::new());
    let mut worst = (1.0f64, 0);
    for p in positions {
        let range = (p * 4 + tap) * hidden..(p * 4 + tap + 1) * hidden;
        let (w, g) = (&want[range.clone()], &got[range]);
        want_tap.extend_from_slice(w);
        got_tap.extend_from_slice(g);
        let dot: f64 = w
            .iter()
            .zip(g)
            .map(|(a, b)| f64::from(*a) * f64::from(*b))
            .sum();
        let nw: f64 = w.iter().map(|a| f64::from(*a).powi(2)).sum();
        let ng: f64 = g.iter().map(|a| f64::from(*a).powi(2)).sum();
        let cos = dot / (nw.sqrt() * ng.sqrt()).max(f64::MIN_POSITIVE);
        if cos < worst.0 {
            worst = (cos, p);
        }
    }
    (relative_error(&want_tap, &got_tap), worst)
}

/// Largest relative RMS error of a tap against MLX's chunked capture. Engine prefill and
/// MLX sum in different orders; the loop amplifies the difference up to the third pass
/// (tap 2), where `tests/prefill.rs` sees relative state errors of about 0.2 against
/// stepping.
const TAP_ERROR: f64 = 0.25;
/// Positions at the end of the text that test A runs again as plain steps.
const STEPPED: usize = 32;
/// How far the plain steps' taps may differ from MLX, as a multiple of the prefill's
/// error at the same positions (at least [`TAP_FLOOR`]). Decode sums in other orders than
/// prefill, and the loop amplifies the difference as it does prefill's: on `anyhow` the
/// steps' error was 0.49 to 1.09 times the prefill's, tap 2 the highest at 0.257.
const STEP_TAP_RATIO: f64 = 1.5;
const TAP_FLOOR: f64 = 0.01;

#[test]
#[ignore = "needs the checkpoint (SIPHON_MODEL) and about 20 GB of GPU memory"]
fn verify_matches_stepping() -> anyhow::Result<()> {
    let tokens = capture_tokens()?;
    let (_gpu, mut model) = load(4096)?;
    let mut kls = Vec::new();
    for prefix in PREFIXES {
        let saved = prefill_state(&mut model, &tokens[..prefix])?;
        for m in 2..=ROWS {
            let rows = &tokens[prefix..prefix + m];
            model.restore_state(&saved)?;
            let mut stepped = Vec::new();
            for &t in rows {
                model.step(t, None)?;
                stepped.extend(model.logits()?);
            }
            let want_states = model.cache_snapshot()?;

            model.restore_state(&saved)?;
            let verdict = model.verify(VerifyTokens::Host(rows), Accept::Rows(m))?;
            ensure!(
                verdict.accepted == m - 1,
                "kept {} rows",
                verdict.accepted + 1
            );
            ensure!(
                model.position() == prefix + m,
                "position {}",
                model.position()
            );
            let got = model.verify_logits()?;
            let (errors, same) = compare_states(&want_states, &model.cache_snapshot()?);
            let vocab = got.len() / m;
            let row_kls: Vec<f64> = (0..m)
                .map(|i| {
                    let r = i * vocab..(i + 1) * vocab;
                    kl(&stepped[r.clone()], &got[r])
                })
                .collect();
            let worst = row_kls.iter().copied().fold(0.0, f64::max);
            let worst_state = errors.iter().copied().fold(0.0, f64::max);
            println!(
                "prefix {prefix}, {m} rows: KL max {worst:.2e}, state error first visit {:.2e} \
                 worst {worst_state:.2e} (identical: {same})",
                errors[0]
            );
            ensure!(
                errors[0] <= MAX_STATE_ERROR,
                "prefix {prefix}, {m} rows: first-visit state error {:.2e}",
                errors[0]
            );
            kls.extend(row_kls);
        }
    }
    let mean = kls.iter().sum::<f64>() / kls.len() as f64;
    let max = kls.iter().copied().fold(0.0, f64::max);
    println!("KL over every row: mean {mean:.2e}, max {max:.2e}");
    ensure!(
        mean <= MAX_MEAN_KL && max <= MAX_KL,
        "KL mean {mean:.2e} max {max:.2e}"
    );
    Ok(())
}

#[test]
#[ignore = "needs the checkpoint (SIPHON_MODEL) and about 20 GB of GPU memory"]
fn rollback_matches_shorter_verify() -> anyhow::Result<()> {
    let tokens = capture_tokens()?;
    let (_gpu, mut model) = load(4096)?;
    let vocab = u32::try_from(model.config().unpadded_vocab_size)?;
    let mut failures = Vec::new();
    for prefix in PREFIXES {
        let saved = prefill_state(&mut model, &tokens[..prefix])?;
        let real = &tokens[prefix..prefix + ROWS];
        for a in 0..ROWS {
            let after = &tokens[prefix + a + 1..prefix + a + 1 + AFTER_ROLLBACK];
            // Reference: verify only rows 0..=a, then step.
            model.restore_state(&saved)?;
            model.verify(VerifyTokens::Host(&real[..=a]), Accept::Rows(a + 1))?;
            let mut want = model.verify_logits()?;
            for &t in after {
                model.step(t, None)?;
                want.extend(model.logits()?);
            }
            let want_states = model.cache_snapshot()?;

            // Rows after a hold wrong tokens; keep rows 0..=a.
            let mut rows = real[..=a].to_vec();
            rows.extend(wrong_tokens(&real[a + 1..], u32::try_from(a)?, vocab));
            model.restore_state(&saved)?;
            let verdict = model.verify(VerifyTokens::Host(&rows), Accept::Rows(a + 1))?;
            ensure!(verdict.accepted == a, "kept {} rows", verdict.accepted + 1);
            let vocab_len = model.config().unpadded_vocab_size;
            let mut got = model.verify_logits()?[..(a + 1) * vocab_len].to_vec();
            for &t in after {
                model.step(t, None)?;
                got.extend(model.logits()?);
            }
            let (errors, same_states) = compare_states(&want_states, &model.cache_snapshot()?);
            let same_logits = identical(&want, &got);
            println!(
                "prefix {prefix}, a = {a}: logits identical {same_logits} (max diff {:.3e}), \
                 states identical {same_states} (worst {:.2e})",
                max_diff(&want, &got),
                errors.iter().copied().fold(0.0, f64::max)
            );
            if !(same_logits && same_states) {
                failures.push((prefix, a));
            }
        }
    }
    ensure!(failures.is_empty(), "rollbacks that differ: {failures:?}");
    Ok(())
}

#[test]
#[ignore = "needs the checkpoint (SIPHON_MODEL) and about 20 GB of GPU memory"]
fn ring_size() -> anyhow::Result<()> {
    const TOKENS: usize = 1300;
    let tokens = capture_tokens()?;
    let prompt = &tokens[..TOKENS];
    let (_gpu, mut model) = load(4096)?;
    let window = model.config().sliding_window_size;
    let mut runs = Vec::new();
    // Each ring size twice, to tell the ring size from run-to-run differences.
    for slots in [window, window, window + ROWS, window + ROWS] {
        model.set_ring_slots(slots)?;
        let mut decode = Vec::new();
        for &t in prompt {
            model.step(t, None)?;
            decode.push(model.logits()?);
        }
        model.reset()?;
        let mut prefill = Vec::new();
        let mut keep = |_: usize, logits: &[f32]| {
            prefill.push(logits.to_vec());
            Ok(())
        };
        model.prefill(prompt, 512, PrefillLogits::All(&mut keep))?;
        // Verify passes of 4 rows across the wrap, after 1000 decode steps.
        model.reset()?;
        for &t in &prompt[..1000] {
            model.step(t, None)?;
        }
        let mut verify = Vec::new();
        for rows in prompt[1000..1100].chunks(4) {
            model.verify(VerifyTokens::Host(rows), Accept::Rows(rows.len()))?;
            verify.push(model.verify_logits()?);
        }
        runs.push([decode, prefill, verify]);
    }
    let differ =
        |a: &[Vec<f32>], b: &[Vec<f32>]| a.iter().zip(b).filter(|(x, y)| !identical(x, y)).count();
    let names = ["decode positions", "prefill positions", "verify passes"];
    let mut verify_differs = false;
    for (i, name) in names.iter().enumerate() {
        let total = runs[0][i].len();
        let repeat_small = differ(&runs[0][i], &runs[1][i]);
        let repeat_large = differ(&runs[2][i], &runs[3][i]);
        let across = differ(&runs[0][i], &runs[2][i]);
        println!(
            "{name}: {window} slots twice differ at {repeat_small} of {total}, {} slots twice \
             at {repeat_large}, {window} against {} at {across}",
            window + ROWS,
            window + ROWS
        );
        verify_differs |= i == 2 && repeat_small + repeat_large > 0;
    }
    // The engine keeps rings of `window` slots and backs up the slots a verify overwrites:
    // decode attention splits its cache by slot, so its sum order changes with the ring
    // size. The verify passes start from decode steps here, so they differ across ring
    // sizes only through those; each must repeat exactly.
    ensure!(!verify_differs, "the verify differs from run to run");
    Ok(())
}

#[test]
#[ignore = "needs the checkpoint (SIPHON_MODEL) and about 20 GB of GPU memory"]
fn verify_rows_are_invariant() -> anyhow::Result<()> {
    let tokens = capture_tokens()?;
    let (_gpu, mut model) = load(4096)?;
    let vocab = model.config().unpadded_vocab_size;
    let vocab_u32 = u32::try_from(vocab)?;
    let mut width_failures = Vec::new();
    let mut repeat_failures = Vec::new();
    for prefix in PREFIXES {
        let saved = prefill_state(&mut model, &tokens[..prefix])?;
        let rows = &tokens[prefix..prefix + ROWS];
        let run = |model: &mut Model, rows: &[u32]| -> anyhow::Result<Vec<f32>> {
            model.restore_state(&saved)?;
            model.verify(VerifyTokens::Host(rows), Accept::Rows(rows.len()))?;
            model.verify_logits()
        };
        let full = run(&mut model, rows)?;
        let again = run(&mut model, rows)?;
        let repeat_same = identical(&full, &again);
        println!(
            "prefix {prefix}: width 8 twice identical {repeat_same} (max diff {:.3e})",
            max_diff(&full, &again)
        );
        if !repeat_same {
            repeat_failures.push((prefix, ROWS));
        }
        for w in 1..ROWS {
            let got = run(&mut model, &rows[..w])?;
            let same = identical(&full[..w * vocab], &got);
            let diff_rows: Vec<usize> = (0..w)
                .filter(|&i| {
                    let r = i * vocab..(i + 1) * vocab;
                    !identical(&full[r.clone()], &got[r])
                })
                .collect();
            println!(
                "prefix {prefix}: width {w} against 8: identical {same}, differing rows \
                 {diff_rows:?}, max diff {:.3e}",
                max_diff(&full[..w * vocab], &got)
            );
            if !same {
                // Width dependence or run-to-run noise: repeat the same width.
                let again = run(&mut model, &rows[..w])?;
                let repeat = identical(&got, &again);
                println!("  width {w} twice identical {repeat}");
                width_failures.push((prefix, w, repeat));
            }
        }
        // One row at a time: row i of the 8-row verify is row 0 of a verify at its position.
        model.restore_state(&saved)?;
        let mut singles = Vec::with_capacity(ROWS * vocab);
        for &t in rows {
            model.verify(VerifyTokens::Host(&[t]), Accept::Rows(1))?;
            singles.extend(model.verify_logits()?);
        }
        let same = identical(&full, &singles);
        println!(
            "prefix {prefix}: 8 single-row verifies against one of 8 rows: identical {same} (max \
             diff {:.3e})",
            max_diff(&full, &singles)
        );
        if !same {
            width_failures.push((prefix, 0, false));
        }
        for j in 0..ROWS - 1 {
            let mut changed = rows[..=j].to_vec();
            changed.extend(wrong_tokens(
                &rows[j + 1..],
                u32::try_from(j)? + 100,
                vocab_u32,
            ));
            let got = run(&mut model, &changed)?;
            let n = (j + 1) * vocab;
            let same = identical(&full[..n], &got[..n]);
            println!(
                "prefix {prefix}: rows after {j} changed: rows 0..={j} identical {same} (max \
                 diff {:.3e})",
                max_diff(&full[..n], &got[..n])
            );
            if !same {
                width_failures.push((prefix, 100 + j, false));
            }
        }
    }
    ensure!(
        width_failures.is_empty() && repeat_failures.is_empty(),
        "rows differ: by width or later rows {width_failures:?} (prefix, width, 100 + j, or 0 \
         for single rows, and whether a repeat of the same run was identical); run to run \
         {repeat_failures:?}"
    );
    Ok(())
}
