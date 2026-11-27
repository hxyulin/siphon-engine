//! GPU parity tests for Track E7b: 2-branch speculative tree verification
//! (`Model::verify_tree` and `Speculator::set_spec_tree`).
//!
//! Needs the checkpoint and about 21 GB of GPU memory, so every test is ignored by
//! default. Run them one at a time on a CUDA host:
//!
//! ```sh
//! SIPHON_MODEL=<model-dir> cargo test --release -p siphon-cuda --test spec_tree -- \
//!     --ignored --nocapture --test-threads 1
//! ```
//!
//! Optionally set `SIPHON_DFLASH=<drafter-dir>` to test end-to-end 2-branch tree
//! speculation with DFlash 2 (otherwise uses the bundled DSpark draft).
//!
//! - `tree_single_branch_matches_linear`: `SpecTree::linear(m)` (`m = 1..=8`) matches
//!   `Model::verify` bit-for-bit on every row's logits, argmax, KV caches, and conv states
//!   across prefixes before, crossing, and past the sliding-window wrap.
//! - `tree_two_branch_matches_independent_branches`: a single 2-branch `Model::verify_tree`
//!   produces bit-identical logits on every primary node AND every secondary node compared
//!   to running the two branches independently through linear `Model::verify`.
//! - `tree_commit_matches_winning_branch`: committing either a primary or secondary winning
//!   path followed by 20 plain decode steps matches running only the winning branch linearly
//!   followed by the same 20 decode steps, bit-for-bit on logits, KV caches, and conv states.
//! - `tree_speculation_greedy_and_sampled`: end-to-end `Speculator` with `SpecTreeConfig`
//!   enabled emits bit-identical tokens to single-row verify at both `T = 0` (greedy) and
//!   `T = 0.8` (stochastic coupled race).

#![cfg(target_os = "linux")]

use std::{env, fs, path::PathBuf};

use anyhow::{Context as _, ensure};
use safetensors::{Dtype, SafeTensors};
use siphon::{Checkpoint, TextConfig};
use siphon_cuda::{
    Gpu,
    draft::DraftOptions,
    model::{
        Accept, AcceptTree, DEFAULT_PREFILL_CHUNK, Model, PrefillLogits, Sampling, SavedState,
        VerifyTokens, VerifyTreeTokens, VisitSnapshot,
    },
    spec::{
        DraftSource, Drafts, SpecTree, SpecTreeConfig, Speculator, Width, greedy_by_verify,
        load_draft,
    },
    weights::Weights,
};

/// Prefixes below, crossing, and past the 1024-token sliding-window wrap.
const PREFIXES: [usize; 3] = [600, 1020, 2500];
const ROWS: usize = 8;
const AFTER_COMMIT: usize = 20;
const GENERATED: usize = 256;

fn capture_tokens() -> anyhow::Result<Vec<u32>> {
    let path = env::var_os("SIPHON_CAPTURE").map_or_else(
        || {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../reference/out/anyhow-chunked.safetensors")
        },
        PathBuf::from,
    );
    let bytes = fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
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

fn load(capacity: usize) -> anyhow::Result<(Gpu, Checkpoint, Model)> {
    let model_dir = PathBuf::from(env::var_os("SIPHON_MODEL").context("set SIPHON_MODEL")?);
    let gpu = Gpu::new(0)?;
    let config = TextConfig::load(&model_dir)?;
    let passes = config.loop_config.num_passes;
    let checkpoint = Checkpoint::open(&model_dir)?;
    let weights = Weights::load(&checkpoint, &config, gpu.stream())?;
    let model = Model::new(&gpu, config, weights, passes, capacity)?;
    Ok((gpu, checkpoint, model))
}

fn prefill_state(model: &mut Model, prompt: &[u32]) -> anyhow::Result<SavedState> {
    model.reset()?;
    model.prefill(prompt, DEFAULT_PREFILL_CHUNK, PrefillLogits::Last)?;
    model.save_state()
}

fn identical(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

fn states_identical(want: &[VisitSnapshot], got: &[VisitSnapshot]) -> bool {
    want.len() == got.len()
        && want.iter().zip(got).all(|(w, g)| {
            identical(&w.keys, &g.keys)
                && identical(&w.values, &g.values)
                && w.conv.len() == g.conv.len()
                && w.conv.iter().zip(&g.conv).all(|(a, b)| identical(a, b))
        })
}

fn alt_tokens(avoid: &[u32], salt: u32, vocab: u32) -> Vec<u32> {
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
#[ignore = "needs the checkpoint (SIPHON_MODEL) and about 21 GB of GPU memory"]
fn tree_single_branch_matches_linear() -> anyhow::Result<()> {
    let tokens = capture_tokens()?;
    let capacity = PREFIXES[2] + ROWS + 64;
    let (_gpu, _ckpt, mut model) = load(capacity)?;
    let vocab = model.config().unpadded_vocab_size;

    for &prefix in &PREFIXES {
        ensure!(tokens.len() > prefix + ROWS, "capture too short for {prefix}");
        let saved = prefill_state(&mut model, &tokens[..prefix])?;
        let chunk = &tokens[prefix..prefix + ROWS];
        for m in 1..=ROWS {
            model.restore_state(&saved)?;
            let linear_verdict = model.verify(VerifyTokens::Host(&chunk[..m]), Accept::Rows(m))?;
            let linear_logits = model.verify_logits()?;
            let linear_state = model.cache_snapshot()?;

            model.restore_state(&saved)?;
            let tree = SpecTree::linear(m)?;
            let path: Vec<usize> = (0..m).collect();
            let tree_verdict = model.verify_tree(
                VerifyTreeTokens::Host(&chunk[..m]),
                &tree,
                AcceptTree::Path(path),
            )?;
            let tree_logits = model.verify_logits()?;
            let tree_state = model.cache_snapshot()?;

            ensure!(
                linear_verdict.accepted == tree_verdict.accepted
                    && linear_verdict.argmax == tree_verdict.argmax,
                "prefix {prefix}, m={m}: argmax mismatch: linear={:?} tree={:?}",
                linear_verdict.argmax,
                tree_verdict.argmax
            );
            for r in 0..m {
                let l = &linear_logits[r * vocab..(r + 1) * vocab];
                let t = &tree_logits[r * vocab..(r + 1) * vocab];
                ensure!(
                    identical(l, t),
                    "prefix {prefix}, m={m}, row {r}: tree single-branch logits differ from linear"
                );
            }
            ensure!(
                states_identical(&linear_state, &tree_state),
                "prefix {prefix}, m={m}: KV/conv states differ after single-branch tree commit"
            );
        }
        println!("prefix {prefix}: single-branch tree m=1..=8 matches linear verify bit-for-bit");
    }
    Ok(())
}

#[test]
#[ignore = "needs the checkpoint (SIPHON_MODEL) and about 21 GB of GPU memory"]
fn tree_two_branch_matches_independent_branches() -> anyhow::Result<()> {
    let tokens = capture_tokens()?;
    let capacity = PREFIXES[2] + ROWS + 64;
    let (_gpu, _ckpt, mut model) = load(capacity)?;
    let vocab = model.config().unpadded_vocab_size;
    let vocab_u32 = u32::try_from(vocab)?;

    // (k1, branch_parent, k2) configurations covering depth 0..=3 branches.
    let configs = [(5, 0, 2), (5, 1, 2), (5, 2, 2), (4, 3, 3)];

    for &prefix in &PREFIXES {
        let saved = prefill_state(&mut model, &tokens[..prefix])?;
        let anchor = tokens[prefix];
        let primary_all = &tokens[prefix + 1..prefix + ROWS];
        let secondary_all = alt_tokens(primary_all, 17, vocab_u32);

        for &(k1, b, k2) in &configs {
            let tree = SpecTree::two_branch(k1, b, k2)?;
            let primary_seq: Vec<u32> = std::iter::once(anchor)
                .chain(primary_all[..k1].iter().copied())
                .collect();
            let secondary_seq: Vec<u32> = std::iter::once(anchor)
                .chain(primary_all[..b].iter().copied())
                .chain(secondary_all[..k2].iter().copied())
                .collect();
            let mut tree_tokens = primary_seq.clone();
            tree_tokens.extend_from_slice(&secondary_all[..k2]);

            model.restore_state(&saved)?;
            model.verify(VerifyTokens::Host(&primary_seq), Accept::Rows(1))?;
            let want_primary = model.verify_logits()?;

            model.restore_state(&saved)?;
            model.verify(VerifyTokens::Host(&secondary_seq), Accept::Rows(1))?;
            let want_secondary = model.verify_logits()?;

            model.restore_state(&saved)?;
            model.verify_tree(
                VerifyTreeTokens::Host(&tree_tokens),
                &tree,
                AcceptTree::Path(vec![0]),
            )?;
            let got = model.verify_logits()?;

            // Every primary node (0..=k1) must match the independent primary verify bit-for-bit.
            for r in 0..=k1 {
                let want = &want_primary[r * vocab..(r + 1) * vocab];
                let actual = &got[r * vocab..(r + 1) * vocab];
                ensure!(
                    identical(want, actual),
                    "prefix {prefix}, (k1={k1}, b={b}, k2={k2}), primary node {r}: logits differ"
                );
            }
            // Every secondary node (k1+1..k1+1+k2) must match the independent secondary verify bit-for-bit.
            for t in 0..k2 {
                let tree_node = 1 + k1 + t;
                let sec_row = b + 1 + t;
                let want = &want_secondary[sec_row * vocab..(sec_row + 1) * vocab];
                let actual = &got[tree_node * vocab..(tree_node + 1) * vocab];
                ensure!(
                    identical(want, actual),
                    "prefix {prefix}, (k1={k1}, b={b}, k2={k2}), secondary node {tree_node}: logits differ"
                );
            }
        }
        println!(
            "prefix {prefix}: all 2-branch tree configs match independent linear branches bit-for-bit"
        );
    }
    Ok(())
}

#[test]
#[ignore = "needs the checkpoint (SIPHON_MODEL) and about 21 GB of GPU memory"]
fn tree_commit_matches_winning_branch() -> anyhow::Result<()> {
    let tokens = capture_tokens()?;
    let capacity = PREFIXES[2] + ROWS + AFTER_COMMIT + 64;
    let (_gpu, _ckpt, mut model) = load(capacity)?;
    let vocab_u32 = u32::try_from(model.config().unpadded_vocab_size)?;

    let (k1, b, k2) = (5, 1, 2);
    let tree = SpecTree::two_branch(k1, b, k2)?;

    for &prefix in &PREFIXES {
        let saved = prefill_state(&mut model, &tokens[..prefix])?;
        let anchor = tokens[prefix];
        let primary = &tokens[prefix + 1..prefix + 1 + k1];
        let secondary = alt_tokens(&primary[b..b + k2], 29, vocab_u32);
        let mut tree_tokens = Vec::with_capacity(1 + k1 + k2);
        tree_tokens.push(anchor);
        tree_tokens.extend_from_slice(primary);
        tree_tokens.extend_from_slice(&secondary);

        let cases: [(&str, Vec<usize>, Vec<u32>, bool); 2] = [
            (
                "primary",
                vec![0, 1, 2, 3],
                vec![anchor, primary[0], primary[1], primary[2]],
                false,
            ),
            (
                "secondary",
                vec![0, 1, 6, 7],
                vec![anchor, primary[0], secondary[0], secondary[1]],
                true,
            ),
        ];

        for (name, path, winning_tokens, expect_secondary) in cases {
            model.restore_state(&saved)?;
            let lin = model.verify(
                VerifyTokens::Host(&winning_tokens),
                Accept::Rows(winning_tokens.len()),
            )?;
            let mut lin_steps = Vec::with_capacity(AFTER_COMMIT);
            let mut tok = lin.next();
            for _ in 0..AFTER_COMMIT {
                tok = model.step(tok, None)?;
                lin_steps.push(tok);
            }
            let lin_logits = model.logits()?;
            let lin_state = model.cache_snapshot()?;

            model.restore_state(&saved)?;
            let tv = model.verify_tree(
                VerifyTreeTokens::Host(&tree_tokens),
                &tree,
                AcceptTree::Path(path.clone()),
            )?;
            ensure!(
                tv.took_secondary == expect_secondary && tv.path == path && tv.next() == lin.next(),
                "prefix {prefix}, {name}: verdict mismatch: {tv:?}"
            );
            let mut tree_steps = Vec::with_capacity(AFTER_COMMIT);
            let mut tok = tv.next();
            for _ in 0..AFTER_COMMIT {
                tok = model.step(tok, None)?;
                tree_steps.push(tok);
            }
            let tree_logits = model.logits()?;
            let tree_state = model.cache_snapshot()?;

            ensure!(
                lin_steps == tree_steps,
                "prefix {prefix}, {name}: decode tokens diverged after tree commit"
            );
            ensure!(
                identical(&lin_logits, &tree_logits),
                "prefix {prefix}, {name}: final step logits differ after tree commit"
            );
            ensure!(
                states_identical(&lin_state, &tree_state),
                "prefix {prefix}, {name}: KV/conv caches differ after tree commit"
            );
        }
        println!(
            "prefix {prefix}: primary & secondary tree commits + {AFTER_COMMIT} steps match bit-for-bit"
        );
    }
    Ok(())
}

#[test]
#[ignore = "needs the checkpoint (SIPHON_MODEL) and about 21 GB of GPU memory"]
fn tree_speculation_greedy_and_sampled() -> anyhow::Result<()> {
    let tokens = capture_tokens()?;
    let prompt = &tokens[..512.min(tokens.len() - 16)];
    let capacity = (prompt.len() + GENERATED + 64).max(1040);
    let (_gpu, checkpoint, mut model) = load(capacity)?;
    model.enable_sampling()?;

    let dflash_dir = env::var_os("SIPHON_DFLASH").map(PathBuf::from);
    let source = dflash_dir
        .as_deref()
        .map_or(DraftSource::Bundled, DraftSource::Dflash);
    let draft = load_draft(
        &mut model,
        &checkpoint,
        source,
        capacity,
        &DraftOptions::fast(),
    )?;
    let mut speculator = Speculator::with_draft(&model, draft, Width::Fixed(7))?;

    for (label, sampling) in [
        ("greedy", Sampling::GREEDY),
        ("T=0.8", Sampling::new(0.8, 0, 0.95, 0.0, 42)?),
    ] {
        model.set_sampling(sampling);
        model.reset()?;
        let first = speculator.prefill(&mut model, prompt, DEFAULT_PREFILL_CHUNK)?;
        let saved = model.save_state()?;

        // Reference: single-row verify (exact greedy / exact sampled draw).
        let want = greedy_by_verify(&mut model, first, GENERATED)?;

        // Linear baseline.
        model.restore_state(&saved)?;
        speculator.set_spec_tree(SpecTreeConfig::default())?;
        speculator.restart(&mut model, first)?;
        let linear_out = speculator.generate(&mut model, GENERATED, Drafts::Dspark)?;
        let linear_stats = speculator.stats().clone();
        ensure!(
            linear_out[..GENERATED] == want[..GENERATED],
            "{label}: linear speculation mismatched reference"
        );

        // 2-branch speculative tree verification.
        model.restore_state(&saved)?;
        speculator.set_spec_tree(SpecTreeConfig::two_branch())?;
        speculator.restart(&mut model, first)?;
        let tree_out = speculator.generate(&mut model, GENERATED, Drafts::Dspark)?;
        let tree_stats = speculator.stats().clone();
        ensure!(
            tree_out[..GENERATED] == want[..GENERATED],
            "{label}: 2-branch tree speculation mismatched reference"
        );
        ensure!(
            tree_stats.tree_rounds > 0,
            "{label}: expected at least one tree verify round"
        );
        println!(
            "{label}: linear {:.3} tok/round ({} rounds) vs 2-branch tree {:.3} tok/round \
             ({} rounds, {} tree rounds, {} secondary wins) — 100% bit-exact!",
            linear_stats.tokens_per_round(),
            linear_stats.rounds,
            tree_stats.tokens_per_round(),
            tree_stats.rounds,
            tree_stats.tree_rounds,
            tree_stats.secondary_accepted,
        );
    }
    Ok(())
}

/// Full-model statistical proof that entropy-gated without-replacement 2-branch residual
/// tree verification (`--spec-tree --spec-beta 1.0`) preserves the exact target
/// distribution on `regex-readme` (1,000 seeds per setting, two-sample chi-square against
/// plain `Model::step`, z < 4.5, with active tree rounds and secondary branch wins).
#[test]
#[ignore = "needs the checkpoint (SIPHON_MODEL), DFlash 2 (SIPHON_DFLASH), and ~21 GB GPU memory"]
#[expect(clippy::too_many_lines, reason = "1,000-seed chi-square verification across two sampling settings")]
fn tree_speculation_beta_lossless_distribution() -> anyhow::Result<()> {
    use std::collections::HashMap;
    use siphon_cuda::{ops::attention::KvFormat, weights::WeightFormat};
    const DRAWN: usize = 4;

    let dflash_dir = env::var_os("SIPHON_DFLASH")
        .or_else(|| env::var_os("SIPHON_DFLASH2"))
        .map(PathBuf::from)
        .context("set SIPHON_DFLASH (or SIPHON_DFLASH2) to the DFlash 2 drafter directory")?;
    let cap_dir = env::var_os("SIPHON_CAPTURES").map_or_else(
        || PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../reference/out"),
        PathBuf::from,
    );
    let readme_path = cap_dir.join("regex-readme-chunked.safetensors");
    let bytes = fs::read(&readme_path)
        .with_context(|| format!("reading {}", readme_path.display()))?;
    let tensors = SafeTensors::deserialize(&bytes)?;
    let view = tensors.tensor("tokens")?;
    ensure!(view.dtype() == Dtype::I32, "tokens are {:?}", view.dtype());
    let readme: Vec<u32> = view
        .data()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| Ok(u32::try_from(i32::from_le_bytes(*b))?))
        .collect::<anyhow::Result<_>>()?;

    let capacity = 4096usize;
    let model_dir = PathBuf::from(env::var_os("SIPHON_MODEL").context("set SIPHON_MODEL")?);
    let gpu = Gpu::new(0)?;
    let config = TextConfig::load(&model_dir)?;
    let passes = config.loop_config.num_passes;
    let checkpoint = Checkpoint::open(&model_dir)?;
    let weights = Weights::load(&checkpoint, &config, gpu.stream())?;
    let mut model = Model::with_formats(
        &gpu,
        config,
        weights,
        passes,
        capacity,
        KvFormat::Fp8,
        WeightFormat::Fp8,
    )?;
    model.enable_sampling()?;

    let draft = load_draft(
        &mut model,
        &checkpoint,
        DraftSource::Dflash(&dflash_dir),
        capacity,
        &DraftOptions::fast(),
    )?;
    let mut speculator = Speculator::with_draft(&model, draft, Width::Fixed(7))?;

    // Find highest-entropy prefix in regex-readme (same procedure as tests/sampling.rs).
    model.set_sampling(Sampling::GREEDY);
    let mut best = (120usize, f64::MIN);
    for cut in (120..=readme.len().min(1500)).step_by(20) {
        model.reset()?;
        model.prefill(&readme[..cut], DEFAULT_PREFILL_CHUNK, PrefillLogits::Last)?;
        let logits = model.logits()?;
        let max = logits.iter().copied().fold(f32::MIN, f32::max);
        let w: Vec<f64> = logits.iter().map(|&l| f64::from(l - max).exp()).collect();
        let total: f64 = w.iter().sum();
        let h: f64 = w
            .iter()
            .filter(|&&v| v > 0.0)
            .map(|&v| -(v / total) * (v / total).ln())
            .sum();
        if h > best.1 {
            best = (cut, h);
        }
    }
    let prompt = &readme[..best.0];
    println!(
        "regex-readme open_cut: {} tokens, next-token entropy {:.2} nats",
        best.0, best.1
    );

    let seeds: u64 = env::var("SIPHON_SAMPLING_SEEDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1000);

    let compare = |a: &[Vec<u32>], b: &[Vec<u32>]| -> (f64, f64) {
        let mut counts: HashMap<&[u32], (f64, f64)> = HashMap::new();
        for k in a {
            counts.entry(k.as_slice()).or_default().0 += 1.0;
        }
        for k in b {
            counts.entry(k.as_slice()).or_default().1 += 1.0;
        }
        let (mut stat, mut dof) = (0.0f64, -1.0f64);
        let mut pooled = (0.0, 0.0);
        for &(x, y) in counts.values() {
            if x + y < 10.0 {
                pooled.0 += x;
                pooled.1 += y;
            } else {
                stat += (x - y).powi(2) / (x + y);
                dof += 1.0;
            }
        }
        if pooled.0 + pooled.1 > 0.0 {
            stat += (pooled.0 - pooled.1).powi(2) / (pooled.0 + pooled.1);
            dof += 1.0;
        }
        let dof = dof.max(1.0);
        let a_wh = 2.0 / (9.0 * dof);
        let z = ((stat / dof).cbrt() - (1.0 - a_wh)) / a_wh.sqrt();
        (z, dof)
    };

    model.set_sampling(Sampling::GREEDY);
    model.reset()?;
    let first_greedy = speculator.prefill(&mut model, prompt, DEFAULT_PREFILL_CHUNK)?;
    let saved = model.save_state()?;

    let settings = [
        ("T=0.8, p=0.95, m=0.05", 0.8f32, 0.95f32, 0.05f32),
        ("T=1.0, p=0.95, m=0.00", 1.0f32, 0.95f32, 0.0f32),
    ];
    let mut worst = f64::MIN;

    for (setting_idx, (name, temperature, top_p, min_p)) in (0u64..).zip(settings) {
        let make_s = |tag: u64, seed: u64| {
            Sampling::new(
                temperature,
                0,
                top_p,
                min_p,
                (setting_idx << 40) | (tag << 32) | seed,
            )
        };

        // 1. Plain `Model::step` reference.
        model.set_spec_beta(None)?;
        let mut plain = Vec::with_capacity(usize::try_from(seeds)?);
        for seed in 0..seeds {
            let sampling = make_s(1, seed)?;
            model.set_sampling(sampling);
            model.restore_state(&saved)?;
            let mut tok = first_greedy;
            let mut seq = Vec::with_capacity(DRAWN);
            for _ in 0..DRAWN {
                tok = model.step(tok, None)?;
                seq.push(tok);
            }
            plain.push(seq);
        }

        // 2. Entropy-gated without-replacement 2-branch residual tree (`--spec-tree --spec-beta 1.0`).
        model.set_spec_beta(Some(1.0))?;
        speculator.set_spec_tree(SpecTreeConfig::two_branch())?;
        let mut tree_seqs = Vec::with_capacity(usize::try_from(seeds)?);
        let (mut rounds, mut emitted, mut tree_rounds, mut secondary_wins) = (0usize, 0usize, 0usize, 0usize);
        for seed in 0..seeds {
            let sampling = make_s(4, seed)?;
            model.set_sampling(sampling);
            model.restore_state(&saved)?;
            speculator.restart(&mut model, first_greedy)?;
            let mut seq = speculator.generate(&mut model, DRAWN, Drafts::Dspark)?;
            let st = speculator.stats();
            rounds += st.rounds;
            emitted += st.emitted;
            tree_rounds += st.tree_rounds;
            secondary_wins += st.secondary_accepted;
            seq.truncate(DRAWN);
            tree_seqs.push(seq);
        }

        println!(
            "P2 beta=1.0 tree ({name}): {seeds} seeds, {:.3} tok/rd, tree_rounds={tree_rounds}/{rounds}, \
             secondary_accepted={secondary_wins}",
            emitted as f64 / rounds.max(1) as f64
        );
        ensure!(
            tree_rounds > 0 && secondary_wins > 0,
            "P2 ({name}): expected active tree rounds ({tree_rounds}) and secondary wins ({secondary_wins})"
        );
        for j in 0..DRAWN {
            let a: Vec<Vec<u32>> = tree_seqs.iter().map(|t| vec![t[j]]).collect();
            let b: Vec<Vec<u32>> = plain.iter().map(|t| vec![t[j]]).collect();
            let (z, dof) = compare(&a, &b);
            println!("  token {}: chi-square z {z:.2} over {dof:.0} dof", j + 1);
            worst = worst.max(z);
        }
        let pairs_a: Vec<Vec<u32>> = tree_seqs.iter().map(|t| t[..2].to_vec()).collect();
        let pairs_b: Vec<Vec<u32>> = plain.iter().map(|t| t[..2].to_vec()).collect();
        let (z12, dof12) = compare(&pairs_a, &pairs_b);
        println!("  tokens 1-2: chi-square z {z12:.2} over {dof12:.0} dof");
        worst = worst.max(z12);
        let (z14, dof14) = compare(&tree_seqs, &plain);
        println!("  tokens 1-4: chi-square z {z14:.2} over {dof14:.0} dof");
        worst = worst.max(z14);
    }

    ensure!(
        worst < 4.5,
        "P2 beta=1.0 tree distribution differs from plain step (worst z {worst:.2})"
    );
    Ok(())
}
