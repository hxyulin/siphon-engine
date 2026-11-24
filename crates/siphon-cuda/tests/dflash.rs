//! A drafter given with `--drafter` against a PyTorch reference, given the same target
//! features: 2DSD's DFlash draft (`Draft::dflash`) against the 2DSD project's runner, or
//! Inco AI's DFlash 2 drafter (`Draft::dflash2`) against `reference/dflash2.py`.
//!
//! Needs the checkpoint the drafter was trained for (`SIPHON_MODEL`: the research
//! checkpoint for 2DSD's draft, the official one for DFlash 2), the drafter's directory
//! (`SIPHON_DFLASH`, see `DflashWeights` and `Dflash2Weights`) and a directory
//! `SIPHON_DFLASH_REF` that holds `tokens.safetensors`, an I32 `tokens` tensor: a prompt
//! and the target's greedy continuation. With `SIPHON_DFLASH_CONTINUE=N` the test appends
//! N greedy tokens of plain decode to `tokens` first. About 20 GB of GPU memory:
//!
//! ```sh
//! SIPHON_MODEL=<model-dir> SIPHON_DFLASH=<draft-dir> SIPHON_DFLASH_REF=<dir> \
//!     cargo test --release -p siphon-cuda --test dflash -- --ignored --nocapture
//! ```
//!
//! The test prefills the tokens with the draft's taps and writes them to
//! `<dir>/taps.safetensors` (BF16 `[tokens, taps, 2304]`). At every anchor `p` from 16 to
//! the end less 8 it commits the taps of the positions before `p` and runs a round of 7
//! drafts anchored at `tokens[p]`, with the exact draft and the fast one
//! (`DraftOptions::fast`, or the variant in `SIPHON_DFLASH_VARIANT`; a shortlist is given
//! every token of the text). It writes the drafts and confidences to
//! `<dir>/siphon.safetensors`.
//!
//! A reference script runs the runner over `taps.safetensors` and writes
//! `<dir>/reference.safetensors`: `ids` (I32 `[anchors, 7]`), `confidence` (F32) and
//! `gap` (F32), the difference between the largest and the second-largest corrected logit
//! (DFlash 2: capped selector score) at each draft. For DFlash 2:
//!
//! ```sh
//! uv run --script reference/dflash2.py <dir>/taps.safetensors <drafter-dir> <model-dir> \
//!     <dir>/reference.safetensors
//! ```
//!
//! DFlash 2's reference also writes `kgap` (F32 `[anchors, 7]`), how far the 16th logit of
//! each draft row lies above the 17th: the selector picks among the top 16, and a tie at
//! that boundary lets either token in.
//!
//! When that file exists, the test compares the exact draft with it. In each round the
//! drafts before the first difference must match; that difference must fall on a
//! near-tie, a gap (or `kgap`) of at most `NEAR_TIE`. Later drafts follow another token
//! and are not compared.

#![cfg(target_os = "linux")]

use std::{collections::HashMap, env, fs, path::PathBuf, sync::Arc};

use anyhow::{Context as _, ensure};
use half::bf16;
use safetensors::{Dtype, SafeTensors, tensor::TensorView};
use siphon::{Checkpoint, TextConfig};
use siphon_cuda::{
    Gpu,
    draft::{Draft, DraftOptions, STEPS},
    model::{Model, PrefillLogits},
    ops::linear::Linear,
    spec::{DraftSource, load_draft},
    weights::{
        WeightFormat, Weights,
        draft::{HIDDEN, SharedWeights},
    },
};

/// The first anchor: the round needs some context.
const FIRST: usize = 16;
/// Largest top-2 logit gap of the reference at which a different draft counts as a
/// near-tie: two BF16 steps of a logit between 16 and 32.
const NEAR_TIE: f32 = 0.25;

fn read_i32(path: &std::path::Path, key: &str) -> anyhow::Result<Vec<i32>> {
    let bytes = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let st = SafeTensors::deserialize(&bytes)?;
    let view = st.tensor(key)?;
    ensure!(view.dtype() == Dtype::I32, "{key} is not I32");
    Ok(view
        .data()
        .as_chunks()
        .0
        .iter()
        .map(|b| i32::from_le_bytes(*b))
        .collect())
}

fn read_f32(path: &std::path::Path, key: &str) -> anyhow::Result<Vec<f32>> {
    let bytes = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let st = SafeTensors::deserialize(&bytes)?;
    let view = st.tensor(key)?;
    ensure!(view.dtype() == Dtype::F32, "{key} is not F32");
    Ok(view
        .data()
        .as_chunks()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect())
}

fn write(
    path: &std::path::Path,
    data: &[(&str, Dtype, Vec<usize>, Vec<u8>)],
) -> anyhow::Result<()> {
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

fn le_bytes<T: Copy, const N: usize>(v: &[T], f: impl Fn(T) -> [u8; N]) -> Vec<u8> {
    v.iter().flat_map(|&x| f(x)).collect()
}

/// Every anchor's 7 draft ids and confidences, and the decoder output of the 7 mask rows.
struct Rounds {
    ids: Vec<i32>,
    confidence: Vec<f32>,
    y: Vec<bf16>,
}

fn rounds(
    draft: &mut Draft,
    stream: &Arc<cudarc::driver::CudaStream>,
    linear: &Linear,
    shared: SharedWeights<'_>,
    taps: &cudarc::driver::CudaSlice<bf16>,
    tokens: &[i32],
    row: usize,
) -> anyhow::Result<Rounds> {
    let mut out = Rounds {
        ids: Vec::new(),
        confidence: Vec::new(),
        y: Vec::new(),
    };
    let mut committed = 0;
    let mut anchor = stream.alloc_zeros::<i32>(1)?;
    for p in FIRST..tokens.len() - STEPS - 1 {
        draft.commit(
            stream,
            linear,
            &taps.slice(committed * row..p * row),
            committed,
        )?;
        committed = p;
        stream.memcpy_htod(&tokens[p..=p], &mut anchor)?;
        draft.round(stream, linear, shared, &anchor)?;
        out.ids.extend(stream.clone_dtoh(draft.draft_ids())?);
        out.confidence
            .extend(stream.clone_dtoh(draft.confidences())?);
        out.y.extend(stream.clone_dtoh(&draft.y().slice(HIDDEN..))?);
    }
    Ok(out)
}

#[test]
#[ignore = "needs the drafter's checkpoint (SIPHON_MODEL), SIPHON_DFLASH, SIPHON_DFLASH_REF and \
            about 20 GB of GPU memory"]
fn dflash_matches_runner() -> anyhow::Result<()> {
    let model_dir = PathBuf::from(env::var_os("SIPHON_MODEL").context("set SIPHON_MODEL")?);
    let draft_dir = PathBuf::from(env::var_os("SIPHON_DFLASH").context("set SIPHON_DFLASH")?);
    let dir = PathBuf::from(env::var_os("SIPHON_DFLASH_REF").context("set SIPHON_DFLASH_REF")?);
    let mut tokens = read_i32(&dir.join("tokens.safetensors"), "tokens")?;
    let extra: usize = env::var("SIPHON_DFLASH_CONTINUE").map_or(Ok(0), |v| v.parse())?;
    let variant = env::var("SIPHON_DFLASH_VARIANT").unwrap_or_else(|_| "fast".to_owned());
    ensure!(
        tokens.len() > FIRST + STEPS + 2,
        "{} tokens are too few",
        tokens.len()
    );

    let gpu = Gpu::new(0)?;
    let stream = Arc::clone(gpu.stream());
    let config = TextConfig::load(&model_dir)?;
    let passes = config.loop_config.num_passes;
    let checkpoint = Checkpoint::open(&model_dir)?;
    let weights = Weights::load(&checkpoint, &config, &stream)?;
    let capacity = tokens.len() + extra + 64;
    let mut model = Model::with_formats(
        &gpu,
        config.clone(),
        weights,
        passes,
        capacity,
        siphon_cuda::ops::attention::KvFormat::Bf16,
        WeightFormat::Bf16,
    )?;
    let source = DraftSource::Dflash(&draft_dir);
    let mut exact = load_draft(
        &mut model,
        &checkpoint,
        source,
        capacity,
        &DraftOptions::default(),
    )?;
    let mut fast = load_draft(
        &mut model,
        &checkpoint,
        source,
        capacity,
        &DraftOptions::parse(&variant, None)?,
    )?;
    let layout = exact.tap_layout();
    let row = layout.len() * HIDDEN;
    if extra > 0 {
        extend_greedy(
            &mut model,
            &mut tokens,
            extra,
            &dir.join("tokens.safetensors"),
        )?;
    }
    println!("taps {layout:?}, {} tokens", tokens.len());

    let prompt: Vec<u32> = tokens
        .iter()
        .map(|&t| u32::try_from(t))
        .collect::<Result<_, _>>()?;
    let mut host_taps = vec![bf16::ZERO; tokens.len() * row];
    let mut collect = |start: usize, taps: cudarc::driver::CudaView<'_, bf16>| {
        let values = stream.clone_dtoh(&taps)?;
        host_taps[start * row..start * row + values.len()].copy_from_slice(&values);
        Ok(())
    };
    model.prefill_with_taps(&prompt, 2048, PrefillLogits::Last, &mut collect)?;
    write_taps(
        &dir.join("taps.safetensors"),
        &tokens,
        &host_taps,
        layout.len(),
    )?;
    let taps = stream.clone_htod(&host_taps)?;
    let linear = Linear::load(stream.context())?;
    let shared = model.draft_shared();
    let e = rounds(&mut exact, &stream, &linear, shared, &taps, &tokens, row)?;
    // The speculator notes the context's tokens; the whole text stands in for them here.
    fast.note_tokens(&stream, shared, &tokens)?;
    let f = rounds(&mut fast, &stream, &linear, shared, &taps, &tokens, row)?;
    let anchors = e.ids.len() / STEPS;
    write_rounds(&dir.join("siphon.safetensors"), &e, &f)?;
    println!("wrote the taps and {anchors} rounds to {}", dir.display());

    let reference = dir.join("reference.safetensors");
    if !reference.exists() {
        println!("no {}: nothing to compare", reference.display());
        return Ok(());
    }
    let reference = Reference {
        ids: read_i32(&reference, "ids")?,
        confidence: read_f32(&reference, "confidence")?,
        gap: read_f32(&reference, "gap")?,
        kgap: read_f32(&reference, "kgap").ok(),
    };
    ensure!(
        reference.ids.len() == e.ids.len()
            && reference.gap.len() == e.ids.len()
            && reference.confidence.len() == e.ids.len(),
        "the reference has {} drafts, the engine {}",
        reference.ids.len(),
        e.ids.len()
    );
    reference.compare(&e, "exact", true)?;
    reference.compare(&f, &variant, false)
}

/// Appends `extra` greedy tokens of plain decode to `tokens`, writes them to `path` and
/// resets the model.
fn extend_greedy(
    model: &mut Model,
    tokens: &mut Vec<i32>,
    extra: usize,
    path: &std::path::Path,
) -> anyhow::Result<()> {
    let prompt: Vec<u32> = tokens
        .iter()
        .map(|&t| u32::try_from(t))
        .collect::<Result<_, _>>()?;
    let mut next = model.prefill(&prompt, 2048, PrefillLogits::Last)?;
    for _ in 0..extra {
        tokens.push(i32::try_from(next)?);
        next = model.step(next, None)?;
    }
    model.reset()?;
    write(
        path,
        &[(
            "tokens",
            Dtype::I32,
            vec![tokens.len()],
            le_bytes(tokens, i32::to_le_bytes),
        )],
    )
}

/// The runner's drafts, their confidences and the top-2 gap of each, and for DFlash 2 the
/// gap at the top-16 boundary of each row.
struct Reference {
    ids: Vec<i32>,
    confidence: Vec<f32>,
    gap: Vec<f32>,
    kgap: Option<Vec<f32>>,
}

impl Reference {
    /// Prints how `got` agrees with the reference, and with `strict` fails if a round's
    /// first different draft is not a near-tie.
    fn compare(&self, got: &Rounds, name: &str, strict: bool) -> anyhow::Result<()> {
        let anchors = self.ids.len() / STEPS;
        let (mut compared, mut differ, mut far, mut boundary) = (0, 0, Vec::new(), 0);
        let mut whole = 0;
        let mut conf_diff = 0f32;
        for a in 0..anchors {
            let r = a * STEPS..(a + 1) * STEPS;
            for j in r.clone() {
                compared += 1;
                conf_diff = conf_diff.max((got.confidence[j] - self.confidence[j]).abs());
                if got.ids[j] != self.ids[j] {
                    differ += 1;
                    let kgap = self.kgap.as_ref().map_or(f32::INFINITY, |k| k[j]);
                    if self.gap[j] > NEAR_TIE && kgap <= NEAR_TIE {
                        boundary += 1;
                    } else if self.gap[j] > NEAR_TIE {
                        far.push((FIRST + a, j - r.start, self.gap[j]));
                    }
                    break;
                }
            }
            whole += usize::from(got.ids[r.clone()] == self.ids[r]);
        }
        println!(
            "{name}: {whole} of {anchors} rounds identical; {compared} drafts compared, \
             {differ} rounds differ, {boundary} of them on a tie at the top-16 boundary and \
             {} at a gap above {NEAR_TIE}; largest confidence difference {conf_diff:.4} over \
             the compared drafts",
            far.len()
        );
        for (p, j, g) in far.iter().take(20) {
            println!("  anchor {p}, draft {j}: gap {g:.4}");
        }
        ensure!(
            !strict || far.is_empty(),
            "{} {name} drafts differ from the runner away from a near-tie",
            far.len()
        );
        Ok(())
    }
}

/// Writes every anchor's drafts, confidences and the exact draft's decoder output.
fn write_rounds(path: &std::path::Path, e: &Rounds, f: &Rounds) -> anyhow::Result<()> {
    let anchors = e.ids.len() / STEPS;
    let first: Vec<i32> = (FIRST..FIRST + anchors)
        .map(i32::try_from)
        .collect::<Result<_, _>>()?;
    write(
        path,
        &[
            (
                "anchors",
                Dtype::I32,
                vec![anchors],
                le_bytes(&first, i32::to_le_bytes),
            ),
            (
                "exact_ids",
                Dtype::I32,
                vec![anchors, STEPS],
                le_bytes(&e.ids, i32::to_le_bytes),
            ),
            (
                "exact_confidence",
                Dtype::F32,
                vec![anchors, STEPS],
                le_bytes(&e.confidence, f32::to_le_bytes),
            ),
            (
                "exact_y",
                Dtype::BF16,
                vec![anchors, STEPS, HIDDEN],
                le_bytes(&e.y, bf16::to_le_bytes),
            ),
            (
                "fast_ids",
                Dtype::I32,
                vec![anchors, STEPS],
                le_bytes(&f.ids, i32::to_le_bytes),
            ),
            (
                "fast_confidence",
                Dtype::F32,
                vec![anchors, STEPS],
                le_bytes(&f.confidence, f32::to_le_bytes),
            ),
        ],
    )
}

/// Writes the tokens and their taps, BF16 `[tokens, count, 2304]`.
fn write_taps(
    path: &std::path::Path,
    tokens: &[i32],
    taps: &[bf16],
    count: usize,
) -> anyhow::Result<()> {
    write(
        path,
        &[
            (
                "tokens",
                Dtype::I32,
                vec![tokens.len()],
                le_bytes(tokens, i32::to_le_bytes),
            ),
            (
                "taps",
                Dtype::BF16,
                vec![tokens.len(), count, HIDDEN],
                le_bytes(taps, bf16::to_le_bytes),
            ),
        ],
    )
}
