//! The prefix cache (`Speculator::prefill_cached`, `Speculator::claim_slot`): a prompt that
//! resumes from a cached prefix must give the first token, its logits bit for bit, the
//! generated tokens and the draft's acceptance of a prefill from position 0. A slot that
//! takes another sequence must drop its entry, and a draft rebuilt for another sequence
//! must drop the draft's state of the entries of a draft with global layers.
//!
//! Needs the checkpoint and about 20 GB of GPU memory, so every test is ignored by
//! default. Each test loads its own model, so run them one at a time:
//!
//! ```sh
//! SIPHON_MODEL=<model-dir> cargo test --release -p siphon-cuda --test prefix -- \
//!     --ignored --nocapture --test-threads 1
//! ```
//!
//! The prompts are slices of the tokens of `reference/out/anyhow-chunked.safetensors` (or
//! `SIPHON_CAPTURE`). The `dflash` tests run the drafter in `SIPHON_DFLASH`, the `dflash2`
//! tests the one in `SIPHON_DFLASH2`, and pass without them.

#![cfg(target_os = "linux")]

use std::{env, fs, path::PathBuf, thread, time::Duration};

use anyhow::{Context as _, ensure};
use safetensors::{Dtype, SafeTensors};
use siphon::{Checkpoint, TextConfig};
use siphon_cuda::{
    Gpu,
    draft::DraftOptions,
    model::{DEFAULT_PREFILL_CHUNK, Model},
    ops::attention::KvFormat,
    spec::{DraftSource, Drafts, Speculator, Width, load_draft},
    weights::{WeightFormat, Weights},
};

const CAPACITY: usize = 12_288;
/// Tokens each run generates after its first.
const STEPS: usize = 96;
const CHUNK: usize = DEFAULT_PREFILL_CHUNK;
/// Batched steps each batched run takes after its first token.
const BATCH_STEPS: usize = 16;
/// Entries live this long unless a test shortens it.
const TTL: Duration = Duration::from_secs(600);

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

/// FNV-1a over the bits of a row of logits.
fn hash(row: &[f32]) -> u64 {
    row.iter().fold(0xcbf2_9ce4_8422_2325, |h, v| {
        (h ^ u64::from(v.to_bits())).wrapping_mul(0x0100_0000_01b3)
    })
}

/// What one request produced.
#[derive(Debug, PartialEq, Eq)]
struct Run {
    /// The first token's logits, hashed.
    logits: u64,
    /// The first token and the generated ones.
    tokens: Vec<u32>,
    /// Verify rounds and the rounds that kept each draft, which follow from the draft's
    /// proposals at a fixed width.
    rounds: usize,
    accepted_at: Vec<usize>,
}

/// Prefills `prompt` into slot 0, through the prefix cache if `cached`, and speculates
/// `STEPS` tokens. Returns the run and the prompt tokens taken from the cache.
fn run(
    model: &mut Model,
    spec: &mut Speculator,
    prompt: &[u32],
    cached: bool,
) -> anyhow::Result<(Run, usize)> {
    let (first, from_cache) = if cached {
        spec.prefill_cached(model, prompt, CHUNK)?
    } else {
        model.reset()?;
        (spec.prefill(model, prompt, CHUNK)?, 0)
    };
    let logits = hash(&model.logits()?);
    let mut tokens = vec![first];
    tokens.extend(spec.generate(model, STEPS, Drafts::Dspark)?);
    tokens.truncate(STEPS + 1);
    let stats = spec.stats();
    Ok((
        Run {
            logits,
            tokens,
            rounds: stats.rounds,
            accepted_at: stats.accepted_at.clone(),
        },
        from_cache,
    ))
}

/// Runs `prompt` through the prefix cache and checks that `cached` of its tokens came
/// from the cache and that the run equals `want`, a run from position 0.
fn check(
    name: &str,
    model: &mut Model,
    spec: &mut Speculator,
    prompt: &[u32],
    want: &Run,
    cached: usize,
) -> anyhow::Result<()> {
    let (got, from_cache) = run(model, spec, prompt, true)?;
    println!(
        "{name}: {} prompt tokens, {from_cache} cached, {} rounds, same as a full prefill: {}",
        prompt.len(),
        got.rounds,
        got == *want
    );
    ensure!(
        from_cache == cached,
        "{name}: {from_cache} tokens came from the cache, expected {cached}"
    );
    ensure!(
        got.logits == want.logits,
        "{name}: the first token's logits differ"
    );
    let differ = got
        .tokens
        .iter()
        .zip(&want.tokens)
        .position(|(g, w)| g != w);
    ensure!(differ.is_none(), "{name}: token {differ:?} differs");
    ensure!(
        got == *want,
        "{name}: the draft's acceptance differs: {got:?} vs {want:?}"
    );
    Ok(())
}

/// Loads the model with `slots` sequence slots and a speculator around `drafter`, or the
/// bundled draft.
fn load(
    drafter: Option<&std::path::Path>,
    slots: usize,
) -> anyhow::Result<(Gpu, Model, Speculator)> {
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
        CAPACITY,
        KvFormat::Fp8,
        WeightFormat::Fp8,
    )?;
    model.add_slots(slots - 1)?;
    let source = drafter.map_or(DraftSource::Bundled, DraftSource::Dflash);
    let draft = load_draft(
        &mut model,
        &checkpoint,
        source,
        CAPACITY,
        &DraftOptions::fast(),
    )?;
    let spec = Speculator::with_draft(&model, draft, Width::Fixed(7))?;
    Ok((gpu, model, spec))
}

fn prefix_cache_matches_prefill(drafter: Option<&std::path::Path>) -> anyhow::Result<()> {
    let (_gpu, mut model, mut spec) = load(drafter, 2)?;

    let capture = capture_tokens()?;
    ensure!(
        capture.len() >= 9000,
        "the capture has {} tokens",
        capture.len()
    );
    // `a` then `f`, a later turn of the same text; `g` shares `a`'s first 4,500 tokens.
    let a = &capture[..5000];
    let f = &capture[..9000];
    let g: Vec<u32> = capture[..4500]
        .iter()
        .chain(&capture[6000..8000])
        .copied()
        .collect();
    let mut want = Vec::new();
    for prompt in [a, f, &g] {
        want.push(run(&mut model, &mut spec, prompt, false)?.0);
    }
    let [want_a, want_f, want_g] = want.try_into().map_err(|_| anyhow::anyhow!("three runs"))?;

    let bytes = spec.enable_prefix_cache(&model, 2, TTL)?;
    println!("prefix cache: {} MiB", bytes >> 20);
    check("a, empty cache", &mut model, &mut spec, a, &want_a, 0)?;
    check("f after a", &mut model, &mut spec, f, &want_f, 4096)?;
    // The cache now holds f below 8,192, which g does not start with.
    check("g after f", &mut model, &mut spec, &g, &want_g, 0)?;
    check("g again", &mut model, &mut spec, &g, &want_g, 6144)?;

    // A batch that continues slot 0's sequence beside another keeps the cache. The tokens
    // fed to slot 0 do not matter here, only that it writes from its position on.
    let mut last = want_g.tokens[0];
    model.reset_slot(1)?;
    let mut other = model.prefill_slot(1, &capture[..700], CHUNK)?;
    for _ in 0..8 {
        let next = model.step_batch(&[(1, other), (0, last)])?;
        (other, last) = (next[0], next[1]);
    }
    check(
        "g after a batch beside it",
        &mut model,
        &mut spec,
        &g,
        &want_g,
        6144,
    )?;

    // A batched request that takes slot 0 empties it.
    model.reset_slot(0)?;
    model.prefill_slot(0, &capture[..700], CHUNK)?;
    check(
        "g after slot 0 took another sequence",
        &mut model,
        &mut spec,
        &g,
        &want_g,
        0,
    )?;

    // A draft rebuilt for another sequence keeps only the rings of a draft without global
    // layers, such as DFlash 2.
    check("g again", &mut model, &mut spec, &g, &want_g, 6144)?;
    spec.rebuild(&mut model, 1, 0, &capture[..700])?;
    let kept = if global_draft(drafter) { 0 } else { 6144 };
    check(
        "g after a rebuild",
        &mut model,
        &mut spec,
        &g,
        &want_g,
        kept,
    )?;
    Ok(())
}

/// Whether the draft keeps global layers: all but DFlash 2.
fn global_draft(drafter: Option<&std::path::Path>) -> bool {
    drafter.is_none_or(|dir| {
        fs::read_to_string(dir.join("config.json")).is_ok_and(|c| !c.contains("DFlash2DraftModel"))
    })
}

/// What one batched request produced: the first token's logits, hashed, and its tokens.
#[derive(Debug, PartialEq, Eq)]
struct Batched {
    logits: u64,
    tokens: Vec<u32>,
}

/// A batched request from position 0 in slot 1, as reference: one prefill, then
/// `BATCH_STEPS` one-row batched steps.
fn batched_from_zero(model: &mut Model, prompt: &[u32]) -> anyhow::Result<Batched> {
    model.reset_slot(1)?;
    let mut tokens = vec![model.prefill_slot(1, prompt, CHUNK)?];
    let logits = hash(&model.logits()?);
    for _ in 0..BATCH_STEPS {
        let last = tokens.last().copied().unwrap_or_default();
        tokens.push(model.step_batch(&[(1, last)])?[0]);
    }
    Ok(Batched { logits, tokens })
}

/// Prefills `prompt` into slot `slot` from `start` in chunks, as the server does between
/// batched steps, and keeps its state in the cache at its boundary. Returns the first
/// token.
fn prefill_batched(
    model: &mut Model,
    spec: &mut Speculator,
    slot: usize,
    prompt: &[u32],
    start: usize,
    busy: &[usize],
) -> anyhow::Result<u32> {
    let boundary = (prompt.len() - 1) / CHUNK * CHUNK;
    let mut at = start;
    loop {
        let end = (at + CHUNK).min(prompt.len());
        let first = model.prefill_slot(slot, &prompt[at..end], CHUNK)?;
        at = end;
        if end == boundary && end > start {
            spec.save_prefix(model, slot, prompt, CHUNK, busy)?;
        }
        if end == prompt.len() {
            return Ok(first);
        }
    }
}

/// Runs `prompts` as one batch, as the server does: each takes a slot from the cache and
/// prefills, then all decode `BATCH_STEPS` batched steps together. Returns each one's run,
/// slot and cached tokens.
fn serve_batch(
    model: &mut Model,
    spec: &mut Speculator,
    prompts: &[&[u32]],
) -> anyhow::Result<Vec<(Batched, usize, usize)>> {
    let mut busy = Vec::new();
    let mut runs = Vec::new();
    for prompt in prompts {
        let (slot, cached) = spec.claim_slot(model, prompt, CHUNK, &busy)?;
        busy.push(slot);
        let first = prefill_batched(model, spec, slot, prompt, cached, &busy)?;
        let logits = hash(&model.logits()?);
        runs.push((
            Batched {
                logits,
                tokens: vec![first],
            },
            slot,
            cached,
        ));
    }
    for _ in 0..BATCH_STEPS {
        let rows: Vec<(usize, u32)> = runs
            .iter()
            .map(|(r, slot, _)| (*slot, r.tokens.last().copied().unwrap_or_default()))
            .collect();
        let next = model.step_batch(&rows)?;
        for ((r, ..), token) in runs.iter_mut().zip(next) {
            r.tokens.push(token);
        }
    }
    Ok(runs)
}

/// Runs `prompt` as the server runs a request that arrives at an idle server. From an
/// entry with the draft's state, or none, it speculates in slot 0 through
/// `prefill_cached`. From an entry without it, it prefills as a batch of one in the entry's
/// slot, then moves to slot 0, rebuilds the draft and speculates (the server's
/// `speculate_again`). Returns the run, the cached tokens and the path it took.
fn serve_alone(
    model: &mut Model,
    spec: &mut Speculator,
    prompt: &[u32],
) -> anyhow::Result<(Run, usize, &'static str)> {
    let hit = spec.prefix_lookup(model, prompt, CHUNK, &[]);
    let (first, logits, cached, path) = if hit.is_some_and(|h| !h.draft) {
        let (slot, cached) = spec.claim_slot(model, prompt, CHUNK, &[])?;
        ensure!(
            Some(slot) == hit.map(|h| h.slot),
            "the batch took another slot"
        );
        let first = prefill_batched(model, spec, slot, prompt, cached, &[slot])?;
        let logits = hash(&model.logits()?);
        spec.move_to_slot0(model, slot)?;
        let free = spec.idle_slot(model, &[0]).context("no idle slot")?;
        spec.rebuild(model, free, 0, prompt)?;
        spec.resume(model, first)?;
        (first, logits, cached, "batch of one, rebuild")
    } else {
        let path = match hit {
            Some(h) if h.slot != 0 => "moved to slot 0 with the draft's rings",
            Some(_) => "in slot 0 with the draft's rings",
            None => "from position 0",
        };
        let (first, cached) = spec.prefill_cached(model, prompt, CHUNK)?;
        (first, hash(&model.logits()?), cached, path)
    };
    let mut tokens = vec![first];
    tokens.extend(spec.generate(model, STEPS, Drafts::Dspark)?);
    tokens.truncate(STEPS + 1);
    let stats = spec.stats();
    Ok((
        Run {
            logits,
            tokens,
            rounds: stats.rounds,
            accepted_at: stats.accepted_at.clone(),
        },
        cached,
        path,
    ))
}

/// Checks a run against `want` and its cached tokens against `cached`.
fn same<T: PartialEq + std::fmt::Debug>(
    name: &str,
    got: &T,
    want: &T,
    from_cache: usize,
    cached: usize,
) -> anyhow::Result<()> {
    println!(
        "{name}: {from_cache} cached (expected {cached}), same as from position 0: {}",
        got == want
    );
    ensure!(
        from_cache == cached,
        "{name}: {from_cache} tokens came from the cache, expected {cached}"
    );
    ensure!(
        got == want,
        "{name}: differs from position 0: {got:?} vs {want:?}"
    );
    Ok(())
}

/// Several conversations through a cache of one entry per slot: lone turns that alternate
/// between three conversations and each hit their own entry, batched hits, eviction, the
/// time to live, and a slot that another conversation takes.
#[expect(clippy::too_many_lines, reason = "one scenario, step by step")]
fn prefix_cache_entries(drafter: Option<&std::path::Path>) -> anyhow::Result<()> {
    let (_gpu, mut model, mut spec) = load(drafter, 4)?;
    let capture = capture_tokens()?;
    ensure!(
        capture.len() >= 9000,
        "the capture has {} tokens",
        capture.len()
    );
    // Three turns each of conversations A, B and C, then D, E and F, which start
    // elsewhere in the text.
    let turn = |from: usize, len: usize| capture[from..from + len].to_vec();
    let lengths = [2300, 4500, 6600];
    let conversations: Vec<Vec<Vec<u32>>> = [0, 1000, 2000]
        .iter()
        .map(|&from| lengths.iter().map(|&len| turn(from, len)).collect())
        .collect();
    let (d, e, f) = (turn(3000, 4500), turn(4000, 2300), turn(5000, 2300));

    let mut alone = Vec::new();
    for turns in &conversations {
        let mut runs = Vec::new();
        for prompt in turns {
            runs.push(run(&mut model, &mut spec, prompt, false)?.0);
        }
        alone.push(runs);
    }
    let mut batch = Vec::new();
    for prompt in [
        &conversations[0][2],
        &conversations[1][2],
        &conversations[2][2],
        &d,
        &e,
        &f,
    ] {
        batch.push(batched_from_zero(&mut model, prompt)?);
    }
    let [batch_a, batch_b, batch_c, batch_d, batch_e, batch_f] =
        batch.try_into().map_err(|_| anyhow::anyhow!("six runs"))?;

    let bytes = spec.enable_prefix_cache(&model, 4, TTL)?;
    println!("prefix cache: 4 entries, {} MiB", bytes >> 20);
    let names = ["A", "B", "C"];

    // Lone turns alternate between the conversations; from the second turn on each starts
    // with its own entry, which another slot holds by then.
    for t in 0..3 {
        for (c, turns) in conversations.iter().enumerate() {
            let (got, cached, path) = serve_alone(&mut model, &mut spec, &turns[t])?;
            let name = format!("{}{} alone ({path})", names[c], t + 1);
            same(&name, &got, &alone[c][t], cached, [0, 2048, 4096][t])?;
        }
    }

    // The same prompts batched: each prefills only its last 456 tokens, fewer than a
    // window, which a stale ring would show.
    let prompts = [
        &conversations[0][2][..],
        &conversations[1][2],
        &conversations[2][2],
    ];
    let runs = serve_batch(&mut model, &mut spec, &prompts)?;
    for ((got, _, cached), (name, want)) in
        runs.iter()
            .zip([("A3", &batch_a), ("B3", &batch_b), ("C3", &batch_c)])
    {
        same(&format!("{name} batched"), got, want, *cached, 6144)?;
    }
    // D takes the slot without an entry beside A.
    let runs = serve_batch(&mut model, &mut spec, &[&d, &conversations[0][2]])?;
    same("D batched beside A3", &runs[0].0, &batch_d, runs[0].2, 0)?;
    same("A3 batched beside D", &runs[1].0, &batch_a, runs[1].2, 6144)?;

    // Every slot holds an entry, so E takes the least recently used one, B's.
    let runs = serve_batch(&mut model, &mut spec, &[&e])?;
    same("E batched", &runs[0].0, &batch_e, runs[0].2, 0)?;
    let (got, cached, path) = serve_alone(&mut model, &mut spec, &conversations[1][2])?;
    same(
        &format!("B3 alone after E took its slot ({path})"),
        &got,
        &alone[1][2],
        cached,
        0,
    )?;
    let (got, cached, path) = serve_alone(&mut model, &mut spec, &conversations[0][2])?;
    same(
        &format!("A3 alone ({path})"),
        &got,
        &alone[0][2],
        cached,
        6144,
    )?;

    // An entry unused for longer than its time to live is gone.
    spec.set_prefix_ttl(Duration::from_secs(2));
    thread::sleep(Duration::from_millis(2500));
    let (got, cached, path) = serve_alone(&mut model, &mut spec, &conversations[0][2])?;
    same(
        &format!("A3 alone after its entry expired ({path})"),
        &got,
        &alone[0][2],
        cached,
        0,
    )?;
    spec.set_prefix_ttl(TTL);

    // F runs while every other slot is busy, so it takes A's slot; A then misses.
    let a_slot = spec
        .prefix_lookup(&model, &conversations[0][2], CHUNK, &[])
        .context("A3 has an entry")?
        .slot;
    let busy: Vec<usize> = (0..4).filter(|&s| s != a_slot).collect();
    let (slot, cached) = spec.claim_slot(&mut model, &f, CHUNK, &busy)?;
    ensure!(slot == a_slot, "F took slot {slot}, not A's slot {a_slot}");
    let first = prefill_batched(&mut model, &mut spec, slot, &f, cached, &[slot])?;
    let logits = hash(&model.logits()?);
    ensure!(
        first == batch_f.tokens[0] && logits == batch_f.logits,
        "F differs from position 0"
    );
    let (got, cached, path) = serve_alone(&mut model, &mut spec, &conversations[0][2])?;
    same(
        &format!("A3 alone after F took its slot ({path})"),
        &got,
        &alone[0][2],
        cached,
        0,
    )?;

    // A reset of the slot drops its entry too.
    let hit = spec
        .prefix_lookup(&model, &conversations[0][2], CHUNK, &[])
        .context("A3 has an entry")?;
    model.reset_slot(hit.slot)?;
    let (got, cached, path) = serve_alone(&mut model, &mut spec, &conversations[0][2])?;
    same(
        &format!("A3 alone after its slot was reset ({path})"),
        &got,
        &alone[0][2],
        cached,
        0,
    )?;
    Ok(())
}

#[test]
#[ignore = "needs the checkpoint and a GPU"]
fn prefix_cache_entries_bundled() -> anyhow::Result<()> {
    prefix_cache_entries(None)
}

#[test]
#[ignore = "needs the checkpoint and a GPU"]
fn prefix_cache_entries_dflash2() -> anyhow::Result<()> {
    let Some(dir) = env::var_os("SIPHON_DFLASH2") else {
        println!("SIPHON_DFLASH2 is not set; skipped");
        return Ok(());
    };
    prefix_cache_entries(Some(std::path::Path::new(&dir)))
}

#[test]
#[ignore = "needs the checkpoint and a GPU"]
fn prefix_cache_matches_prefill_dflash2() -> anyhow::Result<()> {
    let Some(dir) = env::var_os("SIPHON_DFLASH2") else {
        println!("SIPHON_DFLASH2 is not set; skipped");
        return Ok(());
    };
    prefix_cache_matches_prefill(Some(std::path::Path::new(&dir)))
}

#[test]
#[ignore = "needs the checkpoint and a GPU"]
fn prefix_cache_matches_prefill_bundled() -> anyhow::Result<()> {
    prefix_cache_matches_prefill(None)
}

#[test]
#[ignore = "needs the checkpoint and a GPU"]
fn prefix_cache_matches_prefill_dflash() -> anyhow::Result<()> {
    let Some(dir) = env::var_os("SIPHON_DFLASH") else {
        println!("SIPHON_DFLASH is not set; skipped");
        return Ok(());
    };
    prefix_cache_matches_prefill(Some(std::path::Path::new(&dir)))
}
