//! Times the DSpark draft round and measures its acceptance.
//!
//! usage: dspark time <model-dir> <taps.safetensors> [--positions 2790,32768]
//!        [--steps 3,7] [--repeats N] [--variant SPEC] [--shortlist FILE]
//!        dspark accept <model-dir> <taps-dir> [--texts regex-readme,anyhow,half]
//!        [--variants SPEC,SPEC] [--shortlist FILE] [--positions 2790,32768]
//!        [--steps 3,5,7] [--repeats N]
//!
//! `time` commits a text's taps (from `reference/capture.py --taps`), repeated to fill the
//! context, up to each position. It then times a commit of one position through
//! `Draft::commit` followed by `Draft::round`, and the CUDA graph of a round from device
//! inputs that commits that position and drafts each `--steps` count. It reports the
//! minimum over repeats of 20 rounds each.
//!
//! `accept` feeds the draft each text's taps, `<taps-dir>/<text>-taps.safetensors`, and
//! runs a round anchored at every position with the text's token, as `tests/draft_round.rs`
//! does. Draft i of the round anchored at P is accepted if it and every earlier draft
//! equal MLX's top-1 token at the position before it (the taps file's `argmax`), as
//! `reference/dspark.py accept` counts. It prints the acceptance per draft position, the
//! tokens per round at 3, 5 and 7 drafts, the round time on the first text, and accepted
//! tokens per millisecond against a verify-time model.
//!
//! A variant SPEC is `exact`, `fp8`, `fast` or a `+`-joined set of `fp8-backbone`, `fp8-head`,
//! `fp8-fold` and `shortlist` (see `DraftOptions`). `shortlist` ranks tokens by id, or
//! by `--shortlist`, a token ranking of little-endian u32 ids, as Ingine's
//! `draft_shortlist.bin`.

#[cfg(target_os = "linux")]
fn main() -> anyhow::Result<()> {
    linux::main()
}

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("dspark needs the CUDA backend, which builds only on Linux");
    std::process::exit(1);
}

#[cfg(target_os = "linux")]
mod linux {
    use std::{
        env, fs,
        path::{Path, PathBuf},
        time::Instant,
    };

    use anyhow::{Context as _, bail, ensure};
    use cudarc::driver::CudaSlice;
    use half::bf16;
    use safetensors::{Dtype, SafeTensors};
    use siphon::{Checkpoint, TextConfig};
    use siphon_cuda::{
        Gpu,
        draft::{Draft, DraftOptions, STEPS, read_ranking},
        ops::linear::Linear,
        weights::draft::{DraftWeights, HIDDEN, SharedTensors, SharedWeights, TAPS},
    };

    const USAGE: &str = "usage: dspark time <model-dir> <taps.safetensors> \
                         [--positions 2790,32768] [--steps 3,7] [--repeats N] \
                         [--variant SPEC] [--shortlist FILE]\n       \
                         dspark accept <model-dir> <taps-dir> \
                         [--texts regex-readme,anyhow,half] [--variants SPEC,SPEC] \
                         [--shortlist FILE] [--positions 2790,32768] [--steps 3,5,7] \
                         [--repeats N]";
    const ROW: usize = TAPS * HIDDEN;
    /// Rows one fill commit takes.
    const FILL_ROWS: usize = 512;
    /// Verify time of k drafts in ms, from the design's cost model: about 5.7 ms at k = 3
    /// with BF16 weights and 5.0 ms with FP8, and the step times of its section 4 less the
    /// draft for k = 5 and 7.
    const VERIFY_MS: [(usize, f64, f64); 3] = [(3, 5.7, 5.0), (5, 6.4, 5.7), (7, 7.2, 6.6)];

    struct Text {
        name: String,
        tokens: Vec<i32>,
        argmax: Vec<i32>,
        taps: Vec<bf16>,
    }

    fn read_text(path: &Path) -> anyhow::Result<Text> {
        let bytes = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        let st = SafeTensors::deserialize(&bytes)?;
        let i32s = |key: &str| -> anyhow::Result<Vec<i32>> {
            let view = st.tensor(key)?;
            ensure!(view.dtype() == Dtype::I32, "{key} is not I32");
            Ok(view
                .data()
                .as_chunks()
                .0
                .iter()
                .map(|b| i32::from_le_bytes(*b))
                .collect())
        };
        let tokens = i32s("tokens")?;
        let argmax = i32s("argmax").unwrap_or_default();
        let view = st.tensor("taps")?;
        ensure!(view.dtype() == Dtype::BF16, "taps are not BF16");
        let taps: Vec<bf16> = view
            .data()
            .as_chunks()
            .0
            .iter()
            .map(|b| bf16::from_le_bytes(*b))
            .collect();
        ensure!(
            taps.len() == tokens.len() * ROW,
            "taps do not match the tokens"
        );
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("text")
            .trim_end_matches("-taps.safetensors")
            .to_owned();
        Ok(Text {
            name,
            tokens,
            argmax,
            taps,
        })
    }

    fn list(value: &str) -> anyhow::Result<Vec<usize>> {
        value
            .split(',')
            .map(|v| v.parse().with_context(|| format!("bad number {v:?}")))
            .collect()
    }

    struct Env {
        gpu: Gpu,
        config: TextConfig,
        checkpoint: Checkpoint,
        linear: Linear,
        shared: SharedTensors,
    }

    impl Env {
        fn new(model: &Path) -> anyhow::Result<Self> {
            let gpu = Gpu::new(0)?;
            let config = TextConfig::load(model)?;
            let checkpoint = Checkpoint::open(model)?;
            let linear = Linear::load(gpu.context())?;
            let shared = SharedTensors::load(&checkpoint, &config, gpu.stream())?;
            Ok(Self {
                gpu,
                config,
                checkpoint,
                linear,
                shared,
            })
        }

        fn draft(&self, options: &DraftOptions, capacity: usize) -> anyhow::Result<Draft> {
            let stream = self.gpu.stream();
            let weights = DraftWeights::load(&self.checkpoint, &self.config, stream)?;
            Draft::with_options(
                stream,
                &self.config,
                weights,
                SharedWeights::from(&self.shared),
                capacity,
                options,
            )
        }
    }

    /// The minimum time of `f` in ms over `repeats` runs of 20 calls.
    fn time(
        env: &Env,
        repeats: usize,
        f: &mut dyn FnMut() -> anyhow::Result<()>,
    ) -> anyhow::Result<f64> {
        let stream = env.gpu.stream();
        f()?;
        let mut best = f64::INFINITY;
        for _ in 0..repeats {
            stream.synchronize()?;
            let start = Instant::now();
            for _ in 0..20 {
                f()?;
            }
            stream.synchronize()?;
            best = best.min(start.elapsed().as_secs_f64() / 20.0);
        }
        Ok(best * 1e3)
    }

    /// Fills positions `0..p` with the text's taps, repeated, and sets the next round to
    /// commit position `p - 1` again and draft at `p`. Returns the round's taps row.
    fn fill(
        env: &Env,
        draft: &mut Draft,
        text: &Text,
        taps: &CudaSlice<bf16>,
        anchor: &mut CudaSlice<i32>,
        p: usize,
    ) -> anyhow::Result<()> {
        let stream = env.gpu.stream();
        let shared = SharedWeights::from(&env.shared);
        let n = text.tokens.len();
        ensure!(p >= 1, "positions start at 1");
        draft.reset();
        let mut q = 0;
        while q < p {
            let start = q % n;
            let rows = FILL_ROWS.min(p - q).min(n - start);
            let chunk = taps.slice(start * ROW..(start + rows) * ROW);
            draft.commit(stream, &env.linear, &chunk, q)?;
            draft.note_tokens(stream, shared, &text.tokens[start..start + rows])?;
            q += rows;
        }
        let last = (p - 1) % n;
        stream.memcpy_htod(&[text.tokens[p % n]], anchor)?;
        let row = taps.slice(last * ROW..(last + 1) * ROW);
        draft.set_round_inputs(stream, &row, p - 1, anchor)
    }

    /// Graph round times in ms for each position and draft count.
    fn round_times(
        env: &Env,
        draft: &mut Draft,
        text: &Text,
        positions: &[usize],
        steps: &[usize],
        repeats: usize,
    ) -> anyhow::Result<Vec<Vec<f64>>> {
        let stream = env.gpu.stream();
        let shared = SharedWeights::from(&env.shared);
        let taps = stream.clone_htod(&text.taps)?;
        let mut anchor = stream.alloc_zeros::<i32>(1)?;
        let mut out = Vec::new();
        for &p in positions {
            fill(env, draft, text, &taps, &mut anchor, p)?;
            let mut row = Vec::new();
            for &k in steps {
                row.push(time(env, repeats, &mut || {
                    draft.replay_round(stream, &env.linear, shared, k)
                })?);
            }
            out.push(row);
        }
        Ok(out)
    }

    /// Teacher-forced acceptance per draft position, and the anchor count.
    fn acceptance(
        env: &Env,
        draft: &mut Draft,
        text: &Text,
    ) -> anyhow::Result<([f64; STEPS], usize)> {
        let stream = env.gpu.stream();
        let shared = SharedWeights::from(&env.shared);
        let n = text.tokens.len();
        ensure!(text.argmax.len() == n, "{} has no argmax", text.name);
        let taps = stream.clone_htod(&text.taps)?;
        let mut anchor = stream.alloc_zeros::<i32>(1)?;
        draft.reset();
        draft.note_tokens(stream, shared, &text.tokens[..1])?;
        let mut accepted = [0usize; STEPS];
        let anchors = n - STEPS - 1;
        for p in 1..n - STEPS {
            draft.note_tokens(stream, shared, &text.tokens[p..=p])?;
            stream.memcpy_htod(&text.tokens[p..=p], &mut anchor)?;
            let row = taps.slice((p - 1) * ROW..p * ROW);
            draft.set_round_inputs(stream, &row, p - 1, &anchor)?;
            draft.replay_round(stream, &env.linear, shared, STEPS)?;
            let ids = stream.clone_dtoh(draft.draft_ids())?;
            for (i, count) in accepted.iter_mut().enumerate() {
                if (0..=i).all(|j| ids[j] == text.argmax[p + j]) {
                    *count += 1;
                } else {
                    break;
                }
            }
        }
        #[expect(clippy::cast_precision_loss, reason = "counts are far below 2^52")]
        let rates = accepted.map(|a| a as f64 / anchors as f64);
        Ok((rates, anchors))
    }

    fn tokens_per_round(rates: &[f64; STEPS], k: usize) -> f64 {
        1.0 + rates[..k].iter().sum::<f64>()
    }

    pub fn main() -> anyhow::Result<()> {
        let mut args = env::args().skip(1);
        let mode = args.next().context(USAGE)?;
        let model = PathBuf::from(args.next().context(USAGE)?);
        let path = PathBuf::from(args.next().context(USAGE)?);
        let mut positions = vec![2790, 32768];
        let mut repeats = 10;
        let mut steps = None;
        let mut texts = vec![
            "regex-readme".to_owned(),
            "anyhow".to_owned(),
            "half".to_owned(),
        ];
        let mut variants = vec!["exact".to_owned()];
        let mut ranking = None;
        while let Some(flag) = args.next() {
            let value = args
                .next()
                .with_context(|| format!("{flag} needs a value"))?;
            match flag.as_str() {
                "--positions" => positions = list(&value)?,
                "--repeats" => repeats = value.parse()?,
                "--steps" => steps = Some(list(&value)?),
                "--texts" => texts = value.split(',').map(str::to_owned).collect(),
                "--variants" | "--variant" => {
                    variants = value.split(',').map(str::to_owned).collect();
                }
                "--shortlist" => ranking = Some(read_ranking(Path::new(&value))?),
                _ => bail!("unknown flag {flag}\n{USAGE}"),
            }
        }
        let env = Env::new(&model)?;
        match mode.as_str() {
            "time" => {
                let steps = steps.unwrap_or_else(|| vec![3, 7]);
                let text = read_text(&path)?;
                let capacity = positions.iter().max().context("no positions")? + STEPS + 2;
                for spec in &variants {
                    let options = DraftOptions::parse(spec, ranking.as_deref())?;
                    let mut draft = env.draft(&options, capacity)?;
                    host_times(&env, &mut draft, &text, &positions, repeats)?;
                    let times = round_times(&env, &mut draft, &text, &positions, &steps, repeats)?;
                    for (p, row) in positions.iter().zip(&times) {
                        for (k, t) in steps.iter().zip(row) {
                            println!("{spec}: position {p}, round graph of {k} drafts: {t:.3} ms");
                        }
                    }
                }
            }
            "accept" => {
                let steps = steps.unwrap_or_else(|| vec![3, 5, 7]);
                let texts: Vec<Text> = texts
                    .iter()
                    .map(|t| read_text(&path.join(format!("{t}-taps.safetensors"))))
                    .collect::<anyhow::Result<_>>()?;
                accept(
                    &env,
                    &texts,
                    &variants,
                    ranking.as_ref(),
                    &positions,
                    &steps,
                    repeats,
                )?;
            }
            _ => bail!("unknown mode {mode:?}\n{USAGE}"),
        }
        Ok(())
    }

    /// Times `Draft::commit` of one position followed by `Draft::round` at each position.
    fn host_times(
        env: &Env,
        draft: &mut Draft,
        text: &Text,
        positions: &[usize],
        repeats: usize,
    ) -> anyhow::Result<()> {
        let stream = env.gpu.stream();
        let shared = SharedWeights::from(&env.shared);
        let taps = stream.clone_htod(&text.taps)?;
        let mut anchor = stream.alloc_zeros::<i32>(1)?;
        let n = text.tokens.len();
        for &p in positions {
            fill(env, draft, text, &taps, &mut anchor, p)?;
            let last = (p - 1) % n;
            let row = taps.slice(last * ROW..(last + 1) * ROW);
            let t = time(env, repeats, &mut || {
                draft.commit(stream, &env.linear, &row, p - 1)?;
                draft.round(stream, &env.linear, shared, &anchor)
            })?;
            println!("position {p}: commit and round, {STEPS} drafts: {t:.3} ms");
        }
        Ok(())
    }

    fn accept(
        env: &Env,
        texts: &[Text],
        variants: &[String],
        ranking: Option<&Vec<u32>>,
        positions: &[usize],
        steps: &[usize],
        repeats: usize,
    ) -> anyhow::Result<()> {
        let capacity = texts
            .iter()
            .map(|t| t.tokens.len())
            .chain(positions.iter().copied())
            .max()
            .context("no texts")?
            + STEPS
            + 2;
        let mut summary = Vec::new();
        for spec in variants {
            let options = DraftOptions::parse(spec, ranking.map(Vec::as_slice))?;
            let mut draft = env.draft(&options, capacity)?;
            println!("\n### {spec}\n");
            println!(
                "| Text | Anchors | Acceptance per draft position (%) | Tokens per round at k = 3, 5, 7 |"
            );
            println!("| --- | ---: | --- | --- |");
            let mut pooled = [0.0; STEPS];
            let mut pooled_anchors = 0;
            for text in texts {
                let (rates, anchors) = acceptance(env, &mut draft, text)?;
                #[expect(clippy::cast_precision_loss, reason = "counts are far below 2^52")]
                for (p, r) in pooled.iter_mut().zip(rates) {
                    *p += r * anchors as f64;
                }
                pooled_anchors += anchors;
                println!(
                    "| {} | {anchors} | {} | {} |",
                    text.name,
                    rates
                        .iter()
                        .map(|r| format!("{:.1}", 100.0 * r))
                        .collect::<Vec<_>>()
                        .join(", "),
                    [3, 5, 7]
                        .iter()
                        .map(|&k| format!("{:.2}", tokens_per_round(&rates, k)))
                        .collect::<Vec<_>>()
                        .join(", ")
                );
            }
            #[expect(clippy::cast_precision_loss, reason = "counts are far below 2^52")]
            let pooled = pooled.map(|p| p / pooled_anchors as f64);
            println!(
                "| all | {pooled_anchors} | {} | {} |",
                pooled
                    .iter()
                    .map(|r| format!("{:.1}", 100.0 * r))
                    .collect::<Vec<_>>()
                    .join(", "),
                [3, 5, 7]
                    .iter()
                    .map(|&k| format!("{:.2}", tokens_per_round(&pooled, k)))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
            let times = round_times(env, &mut draft, &texts[0], positions, steps, repeats)?;
            println!();
            for (p, row) in positions.iter().zip(&times) {
                println!(
                    "round time at position {p}: {}",
                    steps
                        .iter()
                        .zip(row)
                        .map(|(k, t)| format!("k = {k}: {t:.3} ms"))
                        .collect::<Vec<_>>()
                        .join(", ")
                );
            }
            summary.push((spec.clone(), pooled, times));
        }

        println!("\n### Accepted tokens per ms, all texts, round time at the first position\n");
        println!(
            "| Variant | k | Tokens per round | Draft ms | Tokens/ms, BF16 verify | Tokens/ms, FP8 verify |"
        );
        println!("| --- | ---: | ---: | ---: | ---: | ---: |");
        for (spec, rates, times) in &summary {
            for (i, &k) in steps.iter().enumerate() {
                let Some(&(_, bf16_ms, fp8_ms)) = VERIFY_MS.iter().find(|v| v.0 == k) else {
                    continue;
                };
                let tokens = tokens_per_round(rates, k);
                let draft_ms = times[0][i];
                println!(
                    "| {spec} | {k} | {tokens:.3} | {draft_ms:.3} | {:.4} | {:.4} |",
                    tokens / (bf16_ms + draft_ms),
                    tokens / (fp8_ms + draft_ms)
                );
            }
        }
        Ok(())
    }
}
