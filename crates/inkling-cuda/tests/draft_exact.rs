//! Bit-identity of the engine draft against a golden file that an earlier engine draft
//! wrote over the same inputs.
//!
//! Needs the checkpoint and the taps of `reference/prompts/regex-readme.txt` (see
//! `tests/draft_round.rs`):
//!
//! ```sh
//! INKLING_MODEL=<model-dir> INKLING_DSPARK_REF=<dir> \
//!     cargo test --release -p inkling-cuda --test draft_exact -- --ignored --nocapture
//! ```
//!
//! At every anchor `p` from 2 on, the test commits the taps of the 2 to 8 positions before
//! `p` (rewriting positions an earlier round committed) and runs a round anchored at `p`
//! with the text's token. It compares the 7 draft ids, and hashes of the decoder output,
//! of each base-logit row and of each step's Markov logits, with
//! `<dir>/regex-readme-engine-draft.safetensors`. With `INKLING_DSPARK_WRITE_GOLDEN=1` it
//! writes that file instead.

#![cfg(target_os = "linux")]

use std::{
    collections::HashMap,
    env, fs,
    path::{Path, PathBuf},
};

use anyhow::{Context as _, ensure};
use half::bf16;
use inkling::{Checkpoint, TextConfig};
use inkling_cuda::{
    Gpu,
    draft::{Draft, STEPS},
    ops::linear::Linear,
    weights::draft::{DraftWeights, HIDDEN, SharedTensors, SharedWeights, TAPS},
};
use safetensors::{Dtype, SafeTensors, tensor::TensorView};

const BLOCK: usize = STEPS + 1;
const GOLDEN: &str = "regex-readme-engine-draft.safetensors";

/// Positions committed before the round anchored at `p`: 2 to 8, ending at `p - 1`.
fn commit_rows(p: usize) -> usize {
    (2 + p % 7).min(p)
}

/// FNV-style hash of a buffer's bytes, 8 at a time.
fn hash<T: Copy>(values: &[T]) -> u64 {
    // SAFETY: the slice's bytes are initialized plain data (BF16 or I32 values).
    let bytes = unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    };
    let (words, tail) = bytes.as_chunks::<8>();
    let mut h = 0xcbf2_9ce4_8422_2325_u64;
    for w in words {
        h = (h ^ u64::from_le_bytes(*w)).wrapping_mul(0x0100_0000_01b3);
        h ^= h >> 29;
    }
    for &b in tail {
        h = (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3);
    }
    h
}

/// What one round produced, as stored in the golden file.
#[derive(Default)]
struct Record {
    positions: Vec<i32>,
    ids: Vec<i32>,
    y: Vec<u64>,
    base: Vec<u64>,
    markov: Vec<u64>,
}

impl Record {
    fn read(path: &Path) -> anyhow::Result<Self> {
        let bytes = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        let tensors = SafeTensors::deserialize(&bytes)?;
        let get =
            |key: &str| -> anyhow::Result<Vec<u8>> { Ok(tensors.tensor(key)?.data().to_vec()) };
        let i32s = |b: Vec<u8>| -> Vec<i32> {
            b.as_chunks()
                .0
                .iter()
                .map(|c| i32::from_le_bytes(*c))
                .collect()
        };
        let u64s = |b: Vec<u8>| -> Vec<u64> {
            b.as_chunks()
                .0
                .iter()
                .map(|c| u64::from_le_bytes(*c))
                .collect()
        };
        Ok(Self {
            positions: i32s(get("positions")?),
            ids: i32s(get("draft_ids")?),
            y: u64s(get("y_hash")?),
            base: u64s(get("base_hash")?),
            markov: u64s(get("markov_hash")?),
        })
    }

    fn write(&self, path: &Path) -> anyhow::Result<()> {
        let anchors = self.positions.len();
        let i32_bytes = |v: &[i32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let u64_bytes = |v: &[u64]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let data = [
            (
                "positions",
                Dtype::I32,
                vec![anchors],
                i32_bytes(&self.positions),
            ),
            (
                "draft_ids",
                Dtype::I32,
                vec![anchors, STEPS],
                i32_bytes(&self.ids),
            ),
            ("y_hash", Dtype::U64, vec![anchors], u64_bytes(&self.y)),
            (
                "base_hash",
                Dtype::U64,
                vec![anchors, STEPS],
                u64_bytes(&self.base),
            ),
            (
                "markov_hash",
                Dtype::U64,
                vec![anchors, STEPS],
                u64_bytes(&self.markov),
            ),
        ];
        let views: HashMap<String, TensorView<'_>> = data
            .iter()
            .map(|(name, dtype, shape, bytes)| {
                Ok((
                    (*name).to_owned(),
                    TensorView::new(*dtype, shape.clone(), bytes)?,
                ))
            })
            .collect::<anyhow::Result<_>>()?;
        safetensors::serialize_to_file(views, None, path)?;
        Ok(())
    }
}

struct Setup {
    gpu: Gpu,
    config: TextConfig,
    checkpoint: Checkpoint,
    linear: Linear,
    shared: SharedTensors,
    tokens: Vec<i32>,
    taps: Vec<bf16>,
    golden: PathBuf,
}

fn setup() -> anyhow::Result<Setup> {
    let model_dir = PathBuf::from(env::var_os("INKLING_MODEL").context("set INKLING_MODEL")?);
    let ref_dir =
        PathBuf::from(env::var_os("INKLING_DSPARK_REF").context("set INKLING_DSPARK_REF")?);
    let path = ref_dir.join("regex-readme-taps.safetensors");
    let bytes = fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
    let capture = SafeTensors::deserialize(&bytes)?;
    let tokens: Vec<i32> = capture
        .tensor("tokens")?
        .data()
        .as_chunks()
        .0
        .iter()
        .map(|b| i32::from_le_bytes(*b))
        .collect();
    let taps_view = capture.tensor("taps")?;
    ensure!(taps_view.dtype() == Dtype::BF16, "taps are not BF16");
    let taps: Vec<bf16> = taps_view
        .data()
        .as_chunks()
        .0
        .iter()
        .map(|b| bf16::from_le_bytes(*b))
        .collect();
    ensure!(
        taps.len() == tokens.len() * TAPS * HIDDEN,
        "taps do not match the tokens"
    );
    let gpu = Gpu::new(0)?;
    let config = TextConfig::load(&model_dir)?;
    let checkpoint = Checkpoint::open(&model_dir)?;
    let linear = Linear::load(gpu.context())?;
    let shared = SharedTensors::load(&checkpoint, &config, gpu.stream())?;
    Ok(Setup {
        gpu,
        config,
        checkpoint,
        linear,
        shared,
        tokens,
        taps,
        golden: ref_dir.join(GOLDEN),
    })
}

/// How the rounds run.
#[derive(Clone, Copy)]
enum Run {
    /// [`Draft::commit`], then [`Draft::round`].
    Host,
    /// [`Draft::set_round_inputs`], then [`Draft::run_round`] with this many drafts.
    Device(usize),
    /// [`Draft::set_round_inputs`], then [`Draft::replay_round`] with this many drafts.
    Graph(usize),
    /// As `Graph`, without recording the Markov logits, so the round prunes the Markov
    /// argmax.
    Pruned(usize),
    /// As `Pruned`, with [`Draft::set_round_inputs_kept`] given 8 rows of taps and the kept
    /// count on the device.
    Kept(usize),
}

/// Runs a round anchored at every position and records it.
fn run(s: &Setup, path: Run) -> anyhow::Result<Record> {
    let stream = s.gpu.stream();
    let n = s.tokens.len();
    let shared = SharedWeights::from(&s.shared);
    let weights = DraftWeights::load(&s.checkpoint, &s.config, stream)?;
    let mut draft = Draft::new(stream, &s.config, weights, shared, n + BLOCK)?;
    let record_logits = !matches!(path, Run::Pruned(_) | Run::Kept(_));
    let mut kept = stream.alloc_zeros::<i32>(1)?;
    if record_logits {
        draft.record_logits(stream)?;
    }
    let taps = stream.clone_htod(&s.taps)?;
    let mut anchor = stream.alloc_zeros::<i32>(1)?;
    let row_len = TAPS * HIDDEN;
    let (vocab, padded) = (s.config.unpadded_vocab_size, s.config.vocab_size);
    let mut record = Record::default();
    for p in 2..=n - BLOCK {
        let rows = commit_rows(p);
        let start = p - rows;
        let chunk = taps.slice(start * row_len..p * row_len);
        stream.memcpy_htod(&[s.tokens[p]], &mut anchor)?;
        match path {
            Run::Host => {
                draft.commit(stream, &s.linear, &chunk, start)?;
                draft.round(stream, &s.linear, shared, &anchor)?;
            }
            Run::Device(steps) => {
                draft.set_round_inputs(stream, &chunk, start, &anchor)?;
                draft.run_round(stream, &s.linear, shared, steps)?;
            }
            Run::Graph(steps) | Run::Pruned(steps) => {
                draft.set_round_inputs(stream, &chunk, start, &anchor)?;
                draft.replay_round(stream, &s.linear, shared, steps)?;
            }
            Run::Kept(steps) => {
                let verify = taps.slice(start * row_len..(start + BLOCK) * row_len);
                stream.memcpy_htod(&[i32::try_from(rows - 1)?], &mut kept)?;
                draft.set_round_inputs_kept(stream, &verify, start, &kept.slice(..), &anchor)?;
                draft.replay_round(stream, &s.linear, shared, steps)?;
            }
        }
        record.positions.push(i32::try_from(p)?);
        record.ids.extend(stream.clone_dtoh(draft.draft_ids())?);
        record.y.push(hash(&stream.clone_dtoh(draft.y())?));
        let base = stream.clone_dtoh(draft.base_logits())?;
        for step in 0..STEPS {
            record
                .base
                .push(hash(&base[step * padded..(step + 1) * padded]));
        }
        if record_logits {
            let logits = stream.clone_dtoh(draft.markov_logits().context("recorded")?)?;
            for step in 0..STEPS {
                record
                    .markov
                    .push(hash(&logits[step * vocab..(step + 1) * vocab]));
            }
        }
    }
    Ok(record)
}

/// Compares the first `steps` draft steps of every round with the golden record, and the
/// Markov logits if `got` recorded them.
fn compare(label: &str, golden: &Record, got: &Record, steps: usize) -> anyhow::Result<()> {
    ensure!(
        golden.positions == got.positions,
        "{label}: different anchors"
    );
    let mut bad = [0usize; 4];
    for (a, &p) in golden.positions.iter().enumerate() {
        let r = a * STEPS..a * STEPS + steps;
        let diffs = [
            golden.ids[r.clone()] != got.ids[r.clone()],
            golden.y[a] != got.y[a],
            golden.base[r.clone()] != got.base[r.clone()],
            !got.markov.is_empty() && golden.markov[r.clone()] != got.markov[r],
        ];
        for (count, diff) in bad.iter_mut().zip(diffs) {
            *count += usize::from(diff);
        }
        if diffs.iter().any(|&d| d) && bad.iter().sum::<usize>() <= 4 {
            eprintln!("{label}: anchor {p} differs (ids, Y, base, Markov): {diffs:?}");
        }
    }
    eprintln!(
        "{label}: {} anchors, {steps} steps; rounds that differ in ids {}, Y {}, base logits {}, \
         Markov logits {}",
        golden.positions.len(),
        bad[0],
        bad[1],
        bad[2],
        bad[3]
    );
    ensure!(
        bad == [0; 4],
        "{label}: the draft differs from the golden file"
    );
    Ok(())
}

#[test]
#[ignore = "needs the checkpoint (INKLING_MODEL) and the taps (INKLING_DSPARK_REF)"]
fn draft_matches_golden() -> anyhow::Result<()> {
    let s = setup()?;
    let host = run(&s, Run::Host)?;
    if env::var_os("INKLING_DSPARK_WRITE_GOLDEN").is_some() {
        host.write(&s.golden)?;
        eprintln!("wrote {}", s.golden.display());
        return Ok(());
    }
    let golden = Record::read(&s.golden)?;
    let mut failed = Vec::new();
    let mut check = |label: &str, got: anyhow::Result<Record>, steps: usize| {
        if let Err(err) = got.and_then(|got| compare(label, &golden, &got, steps)) {
            eprintln!("{label}: {err:#}");
            failed.push(label.to_owned());
        }
    };
    check("commit and round", Ok(host), STEPS);
    check(
        "device inputs, 7 drafts",
        run(&s, Run::Device(STEPS)),
        STEPS,
    );
    for steps in [7, 3, 1] {
        check(
            &format!("graph, {steps} drafts"),
            run(&s, Run::Graph(steps)),
            steps,
        );
        check(
            &format!("graph, {steps} drafts, pruned argmax"),
            run(&s, Run::Pruned(steps)),
            steps,
        );
        check(
            &format!("graph, {steps} drafts, kept count on the device"),
            run(&s, Run::Kept(steps)),
            steps,
        );
    }
    ensure!(
        failed.is_empty(),
        "differs from the golden file: {failed:?}"
    );
    Ok(())
}
