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
    spec::{DraftSource, Drafts, Prompt, Speculator, Width, load_draft},
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
        spec.prefill_cached(model, Prompt::text(prompt), CHUNK)?
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
    kv: KvFormat,
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
        kv,
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

/// The served cache.
const SERVED: KvFormat = KvFormat::Fp8;

fn prefix_cache_matches_prefill(
    drafter: Option<&std::path::Path>,
    kv: KvFormat,
) -> anyhow::Result<()> {
    let (_gpu, mut model, mut spec) = load(drafter, 2, kv)?;

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
    spec.rebuild(&mut model, 1, 0, Prompt::text(&capture[..700]))?;
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
            hit.is_some_and(|h| h.slot.is_none_or(|s| s == slot)),
            "the batch took another slot"
        );
        let first = prefill_batched(model, spec, slot, prompt, cached, &[slot])?;
        let logits = hash(&model.logits()?);
        spec.move_to_slot0(model, slot)?;
        let free = spec.idle_slot(model, &[0]).context("no idle slot")?;
        spec.rebuild(model, free, 0, Prompt::text(prompt))?;
        spec.resume(model, first)?;
        let path = if hit.is_some_and(|h| h.slot.is_none()) {
            "from host memory as a batch of one, rebuild"
        } else {
            "batch of one, rebuild"
        };
        (first, logits, cached, path)
    } else {
        let path = match hit.map(|h| h.slot) {
            Some(None) => "from host memory with the draft's state",
            Some(Some(0)) => "in slot 0 with the draft's rings",
            Some(Some(_)) => "moved to slot 0 with the draft's rings",
            None => "from position 0",
        };
        let (first, cached) = spec.prefill_cached(model, Prompt::text(prompt), CHUNK)?;
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
fn prefix_cache_entries(drafter: Option<&std::path::Path>, kv: KvFormat) -> anyhow::Result<()> {
    let (_gpu, mut model, mut spec) = load(drafter, 4, kv)?;
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
        .and_then(|h| h.slot)
        .context("A3 has an entry")?;
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
    let slot = spec
        .prefix_lookup(&model, &conversations[0][2], CHUNK, &[])
        .and_then(|h| h.slot)
        .context("A3 has an entry")?;
    model.reset_slot(slot)?;
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

/// Bytes of host memory the host cache tests page-lock.
const HOST_BYTES: usize = 2 << 30;
/// The paths of [`serve_alone`] from no entry and from a host entry with the draft's state.
const MISS: &str = "from position 0";
const HOST: &str = "from host memory with the draft's state";

/// The prefix cache's host tier (`Speculator::enable_host_cache`): with one device entry
/// for two slots, every conversation loses its device entry to the next one, and must come
/// back from host memory equal to a prefill from position 0: alone with the draft's state,
/// batched without it, extended past its boundary, sharing chunks with a conversation of
/// the same start, and through a batch of one whose rebuild then gives the host entry the
/// draft's state. Most remainders after the restored boundary are shorter than the
/// 1,024-slot window, where a stale ring would show.
#[expect(
    clippy::similar_names,
    clippy::too_many_lines,
    reason = "one scenario, step by step, over the turns of named conversations"
)]
fn host_cache_matches_prefill(
    drafter: Option<&std::path::Path>,
    kv: KvFormat,
) -> anyhow::Result<()> {
    let (_gpu, mut model, mut spec) = load(drafter, 2, kv)?;
    let capture = capture_tokens()?;
    ensure!(
        capture.len() >= 9000,
        "the capture has {} tokens",
        capture.len()
    );
    let a = &capture[..4500];
    let a1 = &capture[..4900];
    let a2 = &capture[..7000];
    let a3 = &capture[..8500];
    let a4 = &capture[..8700];
    let b = &capture[1000..5600];
    let b1 = &capture[1000..6000];
    let c = &capture[3000..7600];
    let c1 = &capture[3000..8000];
    // `g` shares `a`'s first 4,300 tokens, so two chunks of its global caches.
    let g: Vec<u32> = capture[..4300]
        .iter()
        .chain(&capture[6000..8000])
        .copied()
        .collect();
    let g1: Vec<u32> = g.iter().chain(&capture[8000..8300]).copied().collect();

    let mut alone = Vec::new();
    for prompt in [a1, a2, a4, b1, &g1] {
        alone.push(run(&mut model, &mut spec, prompt, false)?.0);
    }
    let [want_a1, want_a2, want_a4, want_b1, want_g1] =
        alone.try_into().map_err(|_| anyhow::anyhow!("five runs"))?;
    let mut batch = Vec::new();
    for prompt in [a3, b1, c1] {
        batch.push(batched_from_zero(&mut model, prompt)?);
    }
    let [batch_a3, batch_b1, batch_c1] = batch
        .try_into()
        .map_err(|_| anyhow::anyhow!("three runs"))?;

    spec.enable_prefix_cache(&model, 1, TTL)?;
    let (chunks, chunk_bytes) = spec.enable_host_cache(&model, HOST_BYTES)?;
    println!(
        "host cache: {chunks} chunks of {} MiB, one device entry",
        chunk_bytes >> 20
    );
    let notes = |spec: &mut Speculator| {
        for line in spec.host_notes() {
            println!("  {line}");
        }
    };
    let alone_turn = |name: &str,
                      model: &mut Model,
                      spec: &mut Speculator,
                      prompt: &[u32],
                      want: Option<&Run>,
                      cached: usize,
                      path: &str|
     -> anyhow::Result<()> {
        let (got, from_cache, took) = serve_alone(model, spec, prompt)?;
        notes(spec);
        let name = format!("{name} alone ({took})");
        ensure!(took == path, "{name}: expected the path {path}");
        if let Some(want) = want {
            same(&name, &got, want, from_cache, cached)?;
        } else {
            println!("{name}: {from_cache} cached");
            ensure!(
                from_cache == cached,
                "{name}: {from_cache} cached, expected {cached}"
            );
        }
        Ok(())
    };
    alone_turn("A", &mut model, &mut spec, a, None, 0, MISS)?;
    // B takes the one device entry, so A lives only in host memory.
    alone_turn("B", &mut model, &mut spec, b, None, 0, MISS)?;
    // 804 tokens past the restored boundary.
    alone_turn("A1", &mut model, &mut spec, a1, Some(&want_a1), 4096, HOST)?;
    alone_turn("B1", &mut model, &mut spec, b1, Some(&want_b1), 4096, HOST)?;
    // A2 prefills a chunk past A's boundary and saves a new entry, which shares A's two
    // chunks and replaces A's entry.
    alone_turn("A2", &mut model, &mut spec, a2, Some(&want_a2), 4096, HOST)?;
    // G shares A2's first two chunks.
    alone_turn("G", &mut model, &mut spec, &g, None, 0, MISS)?;
    alone_turn("C", &mut model, &mut spec, c, None, 0, MISS)?;
    alone_turn("G1", &mut model, &mut spec, &g1, Some(&want_g1), 6144, HOST)?;
    alone_turn(
        "A2 again",
        &mut model,
        &mut spec,
        a2,
        Some(&want_a2),
        6144,
        HOST,
    )?;

    // Batched requests restore without the draft's state; B1 and C1 each prefill fewer
    // tokens than a window past their boundary.
    let runs = serve_batch(&mut model, &mut spec, &[b1, c1])?;
    notes(&mut spec);
    same("B1 batched", &runs[0].0, &batch_b1, runs[0].2, 4096)?;
    same("C1 batched", &runs[1].0, &batch_c1, runs[1].2, 4096)?;
    // A3 restores A2 and saves its own entry from the batch, without the draft's state.
    let runs = serve_batch(&mut model, &mut spec, &[a3])?;
    notes(&mut spec);
    same("A3 batched", &runs[0].0, &batch_a3, runs[0].2, 6144)?;
    // B1 takes the device entry; A4 then finds A3's host entry without the draft's state,
    // runs as a batch of one, and its rebuild gives the host entry the draft's state.
    alone_turn("B1", &mut model, &mut spec, b1, Some(&want_b1), 4096, HOST)?;
    alone_turn(
        "A4",
        &mut model,
        &mut spec,
        a4,
        Some(&want_a4),
        8192,
        "from host memory as a batch of one, rebuild",
    )?;
    alone_turn("B1", &mut model, &mut spec, b1, Some(&want_b1), 4096, HOST)?;
    alone_turn(
        "A4 again",
        &mut model,
        &mut spec,
        a4,
        Some(&want_a4),
        8192,
        HOST,
    )?;
    Ok(())
}

/// The host tier through disk (`Speculator::save_host_cache`, `load_host_cache`): entries
/// saved to files and read back into a new host tier, with every slot reset, restore as a
/// prefill from position 0 would, with 804 and 904 tokens past their boundaries. A
/// manifest of another identity is discarded with its files, and a directory without a
/// manifest, as a save that stopped midway leaves it, loads nothing.
fn host_cache_round_trips(drafter: Option<&std::path::Path>, kv: KvFormat) -> anyhow::Result<()> {
    use siphon_cuda::spec::DiskLoaded;

    let (_gpu, mut model, mut spec) = load(drafter, 2, kv)?;
    let capture = capture_tokens()?;
    let (a, a1) = (&capture[..4500], &capture[..4900]);
    let (b, b1) = (&capture[1000..5600], &capture[1000..6000]);
    let want_a = run(&mut model, &mut spec, a1, false)?.0;
    let want_b = run(&mut model, &mut spec, b1, false)?.0;
    let dir = env::temp_dir().join(format!("siphon-host-cache-test-{}", std::process::id()));
    let identity = serde_json::json!({"test": "host cache round trip"});

    spec.enable_prefix_cache(&model, 1, TTL)?;
    spec.enable_host_cache(&model, HOST_BYTES)?;
    serve_alone(&mut model, &mut spec, a)?;
    serve_alone(&mut model, &mut spec, b)?;
    let saved = spec.save_host_cache(&model, &dir, &identity)?;
    println!("saved {saved:?}");
    ensure!(saved.entries == 2, "saved {} entries, not 2", saved.entries);

    // A new process in all but name: a new cache, and no slot holds anything.
    spec.enable_prefix_cache(&model, 1, TTL)?;
    spec.enable_host_cache(&model, HOST_BYTES)?;
    for slot in 0..model.slots() {
        model.reset_slot(slot)?;
    }
    let loaded = spec.load_host_cache(&dir, &identity)?;
    println!("loaded {loaded:?}");
    ensure!(
        matches!(
            loaded,
            DiskLoaded::Loaded {
                entries: 2,
                skipped: 0,
                ..
            }
        ),
        "loaded {loaded:?}"
    );
    for (name, prompt, want) in [("A1", a1, &want_a), ("B1", b1, &want_b)] {
        let (got, cached, path) = serve_alone(&mut model, &mut spec, prompt)?;
        ensure!(path == HOST, "{name} took the path {path}");
        same(&format!("{name} after a restart"), &got, want, cached, 4096)?;
    }

    // Another identity: the directory goes.
    spec.save_host_cache(&model, &dir, &identity)?;
    let other = serde_json::json!({"test": "another checkpoint"});
    let loaded = spec.load_host_cache(&dir, &other)?;
    println!("with another identity: {loaded:?}");
    ensure!(matches!(loaded, DiskLoaded::Discarded(_)), "{loaded:?}");
    let left = fs::read_dir(&dir)?.count();
    ensure!(left == 0, "{left} files left after the discard");
    let loaded = spec.load_host_cache(&dir, &identity)?;
    ensure!(matches!(loaded, DiskLoaded::Missing), "{loaded:?}");

    // A save that stopped before its manifest loads nothing.
    spec.save_host_cache(&model, &dir, &identity)?;
    fs::remove_file(dir.join("manifest.json"))?;
    let loaded = spec.load_host_cache(&dir, &identity)?;
    ensure!(matches!(loaded, DiskLoaded::Missing), "{loaded:?}");
    fs::remove_dir_all(&dir)?;
    Ok(())
}

#[test]
#[ignore = "needs the checkpoint and a GPU"]
fn host_cache_round_trips_bundled() -> anyhow::Result<()> {
    host_cache_round_trips(None, SERVED)
}

#[test]
#[ignore = "needs the checkpoint and a GPU"]
fn host_cache_round_trips_dflash2() -> anyhow::Result<()> {
    let Some(dir) = env::var_os("SIPHON_DFLASH2") else {
        println!("SIPHON_DFLASH2 is not set; skipped");
        return Ok(());
    };
    host_cache_round_trips(Some(std::path::Path::new(&dir)), SERVED)
}

#[test]
#[ignore = "needs the checkpoint and a GPU"]
fn host_cache_matches_prefill_bundled() -> anyhow::Result<()> {
    host_cache_matches_prefill(None, SERVED)
}

#[test]
#[ignore = "needs the checkpoint and a GPU"]
fn host_cache_matches_prefill_dflash2() -> anyhow::Result<()> {
    let Some(dir) = env::var_os("SIPHON_DFLASH2") else {
        println!("SIPHON_DFLASH2 is not set; skipped");
        return Ok(());
    };
    host_cache_matches_prefill(Some(std::path::Path::new(&dir)), SERVED)
}

#[test]
#[ignore = "needs the checkpoint and a GPU"]
fn prefix_cache_entries_bundled() -> anyhow::Result<()> {
    prefix_cache_entries(None, SERVED)
}

#[test]
#[ignore = "needs the checkpoint and a GPU"]
fn prefix_cache_entries_dflash2() -> anyhow::Result<()> {
    let Some(dir) = env::var_os("SIPHON_DFLASH2") else {
        println!("SIPHON_DFLASH2 is not set; skipped");
        return Ok(());
    };
    prefix_cache_entries(Some(std::path::Path::new(&dir)), SERVED)
}

#[test]
#[ignore = "needs the checkpoint and a GPU"]
fn prefix_cache_matches_prefill_dflash2() -> anyhow::Result<()> {
    let Some(dir) = env::var_os("SIPHON_DFLASH2") else {
        println!("SIPHON_DFLASH2 is not set; skipped");
        return Ok(());
    };
    prefix_cache_matches_prefill(Some(std::path::Path::new(&dir)), SERVED)
}

#[test]
#[ignore = "needs the checkpoint and a GPU"]
fn prefix_cache_matches_prefill_bundled() -> anyhow::Result<()> {
    prefix_cache_matches_prefill(None, SERVED)
}

#[test]
#[ignore = "needs the checkpoint and a GPU"]
fn prefix_cache_matches_prefill_dflash() -> anyhow::Result<()> {
    let Some(dir) = env::var_os("SIPHON_DFLASH") else {
        println!("SIPHON_DFLASH is not set; skipped");
        return Ok(());
    };
    prefix_cache_matches_prefill(Some(std::path::Path::new(&dir)), SERVED)
}

#[test]
#[ignore = "needs the checkpoint and a GPU"]
fn prefix_cache_entries_nvfp4() -> anyhow::Result<()> {
    prefix_cache_entries(None, KvFormat::Nvfp4)
}

#[test]
#[ignore = "needs the checkpoint and a GPU"]
fn prefix_cache_entries_mixed() -> anyhow::Result<()> {
    prefix_cache_entries(None, KvFormat::Mixed)
}

#[test]
#[ignore = "needs the checkpoint and a GPU"]
fn host_cache_matches_prefill_nvfp4() -> anyhow::Result<()> {
    host_cache_matches_prefill(None, KvFormat::Nvfp4)
}

#[test]
#[ignore = "needs the checkpoint and a GPU"]
fn host_cache_matches_prefill_mixed() -> anyhow::Result<()> {
    host_cache_matches_prefill(None, KvFormat::Mixed)
}

#[test]
#[ignore = "needs the checkpoint and a GPU"]
fn host_cache_round_trips_nvfp4() -> anyhow::Result<()> {
    host_cache_round_trips(None, KvFormat::Nvfp4)
}

#[test]
#[ignore = "needs the checkpoint and a GPU"]
fn host_cache_round_trips_mixed() -> anyhow::Result<()> {
    host_cache_round_trips(None, KvFormat::Mixed)
}
