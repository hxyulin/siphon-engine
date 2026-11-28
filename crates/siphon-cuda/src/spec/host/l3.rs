//! Active L3 NVMe spillover tier for the prefix cache's host tier ([`super::HostCache`]).
//!
//! Entries live in page-locked host RAM first. Only when RAM is full and an entry is
//! evicted LRU does it spill here, using content-addressed deduplicated KV chunk files
//! (`chunk-<key>.bin`) plus one local-state file per entry (`local-<id>.bin`) and a
//! `manifest.json` index.
//!
//! Wear-protection (anti-churn) properties:
//! - **Write-back (eviction-only), never write-through:** active conversations that grow
//!   turn-by-turn replace their own shorter prefixes in RAM without ever touching disk.
//! - **Content-addressed KV chunk deduplication:** shared system prompts, shared codebase
//!   prefixes, and earlier turns of a conversation are written to NVMe once and referenced
//!   by count across entries.
//! - **Clean-entry skip:** an entry promoted from disk to RAM and later evicted unchanged
//!   writes zero bytes; only its LRU timestamp updates in memory.
//! - **Prefix-superseded skip:** an evicted entry already covered by a longer on-disk
//!   entry with equal or better draft completeness is not written.
//! - **Minimum spill size (`min_spill_tokens`):** tiny entries whose fixed local state
//!   dominates their KV chunks stay in RAM only.
//! - **Sustained write-rate token bucket (`WriteBudget`):** absorbs legitimate bursts
//!   while capping long-term SSD write rate under pathological thrashing.
//! - **Async background I/O without per-spill `fsync`:** spills copy only missing bytes
//!   into bounded heap buffers and write atomically via `.tmp` + `rename` off the GPU
//!   thread. `manifest.json` checkpoints at most every 60 s and calls `sync_all()` only
//!   on graceful shutdown.

use std::{
    collections::{HashMap, HashSet},
    fs::{self, File},
    io::{BufReader, BufWriter, Read, Write},
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
        mpsc,
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use anyhow::{Context as _, bail, ensure};
use serde_json::{Value, json};
use siphon::prefix_hash::{PrefixHashes, chunk_hashes};

use super::{
    HOST_CHUNK_POSITIONS, HostCache, HostEntry,
    disk::{Loaded, MANIFEST, Saved, discard, pad8, write_atomic},
};

/// Disk format version for the active L3 deduplicated tier.
pub const DISK_FORMAT_L3: u32 = 2;

const CHUNK_MAGIC: &[u8; 8] = b"SIPHONCK";
const LOCAL_MAGIC: &[u8; 8] = b"SIPHONLC";
const FILE_HEADER_BYTES: usize = 40;

/// Minimum interval between background `manifest.json` checkpoints during serving.
const MANIFEST_CHECKPOINT_INTERVAL: Duration = Duration::from_secs(60);
/// Minimum interval between repeated throttle/busy log lines.
const LOG_THROTTLE_INTERVAL: Duration = Duration::from_secs(60);

/// Default minimum prompt length (in tokens) eligible to spill to NVMe (2 chunks = 4,096).
pub const DEFAULT_MIN_SPILL_TOKENS: usize = 2 * HOST_CHUNK_POSITIONS;
/// Default sustained NVMe spill rate cap: 32 MiB/s (~2.7 TB/day worst-case ceiling).
pub const DEFAULT_MAX_WRITE_BYTES_PER_SEC: u64 = 32 << 20;
/// Default burst allowance for the spill write-rate token bucket: 2 GiB.
pub const DEFAULT_WRITE_BURST_BYTES: u64 = 2 << 30;
/// Default maximum bytes of in-flight spill buffers queued for the background writer.
pub const DEFAULT_MAX_INFLIGHT_BYTES: usize = 1 << 30;

/// Configuration for the active L3 NVMe disk cache tier.
#[derive(Clone, Debug)]
pub struct DiskTierConfig {
    /// Directory storing `manifest.json`, `chunk-*.bin`, and `local-*.bin`.
    pub dir: PathBuf,
    /// Maximum total bytes stored in `dir` across live chunk and local files.
    pub max_bytes: u64,
    /// Entries shorter than this many tokens stay in RAM and do not spill on eviction.
    pub min_spill_tokens: usize,
    /// Sustained write-rate cap in bytes/sec (`0` disables rate limiting).
    pub max_write_bytes_per_sec: u64,
    /// Burst capacity in bytes for the write-rate token bucket.
    pub write_burst_bytes: u64,
    /// Maximum heap bytes queued for the background writer before shedding spills.
    pub max_inflight_bytes: usize,
}

impl DiskTierConfig {
    /// Production defaults for an L3 NVMe cache of `max_bytes` in `dir`.
    #[must_use]
    pub fn new(dir: PathBuf, max_bytes: u64) -> Self {
        Self {
            dir,
            max_bytes,
            min_spill_tokens: DEFAULT_MIN_SPILL_TOKENS,
            max_write_bytes_per_sec: DEFAULT_MAX_WRITE_BYTES_PER_SEC,
            write_burst_bytes: DEFAULT_WRITE_BURST_BYTES,
            max_inflight_bytes: DEFAULT_MAX_INFLIGHT_BYTES,
        }
    }
}

/// Summary of an L3 disk cache tier's state and lifetime counters.
#[derive(Clone, Copy, Debug, Default)]
pub struct DiskTierStats {
    /// Entries currently indexed on disk.
    pub entries: usize,
    /// Deduplicated KV chunks currently on disk.
    pub unique_chunks: usize,
    /// Live bytes stored on disk (chunks + locals).
    pub disk_bytes: u64,
    /// Configured capacity ceiling in bytes.
    pub max_bytes: u64,
    /// Entries spilled from RAM to disk.
    pub spills: u64,
    /// Evicted entries skipped because they were already clean on disk.
    pub skips_clean: u64,
    /// Evicted entries skipped because they were shorter than `min_spill_tokens`.
    pub skips_small: u64,
    /// Evicted entries skipped because a longer entry already covers them on disk.
    pub skips_superseded: u64,
    /// Evicted entries shed by the write-rate budget or writer backpressure.
    pub skips_throttled: u64,
    /// L3 hits promoted back into RAM.
    pub hits: u64,
    /// Entries evicted from disk to stay within `max_bytes`.
    pub evictions: u64,
    /// Cumulative payload bytes scheduled for NVMe writes.
    pub bytes_written: u64,
    /// Cumulative payload bytes read back from NVMe.
    pub bytes_read: u64,
}

/// Computes a 128-bit cumulative content key after each full `chunk` of `tokens`.
///
/// Uses two independent four-lane mixers so each chunk key uniquely identifies the exact
/// token prefix `tokens[..(j + 1) * chunk]`.
#[must_use]
pub(super) fn chunk_keys_128(tokens: &[u32], chunk: usize) -> Vec<u128> {
    const MUL_A: u64 = 0x9e37_79b1_85eb_ca87;
    const MUL_B: u64 = 0xc2b2_ae3d_27d4_eb4f;
    if chunk == 0 {
        return Vec::new();
    }
    let count = tokens.len() / chunk;
    let mut out = Vec::with_capacity(count);
    let mut a: [u64; 4] = [
        0xcbf2_9ce4_8422_2325,
        0x9e37_79b9_7f4a_7c15,
        0xbf58_476d_1ce4_e5b9,
        0x94d0_49bb_1331_11eb,
    ];
    let mut b: [u64; 4] = [
        0x243f_6a88_85a3_08d3,
        0x1319_8a2e_0370_7344,
        0xa409_3822_299f_31d0,
        0x082e_fa98_ec4e_6c89,
    ];
    let mut pos = 0usize;
    for block in tokens.chunks_exact(chunk) {
        let (octets, rem) = block.as_chunks::<8>();
        for c in octets {
            let w0 = u64::from(c[0]) | (u64::from(c[1]) << 32);
            let w1 = u64::from(c[2]) | (u64::from(c[3]) << 32);
            let w2 = u64::from(c[4]) | (u64::from(c[5]) << 32);
            let w3 = u64::from(c[6]) | (u64::from(c[7]) << 32);
            a[0] = (a[0] ^ w0).wrapping_mul(MUL_A);
            a[1] = (a[1] ^ w1).wrapping_mul(MUL_A);
            a[2] = (a[2] ^ w2).wrapping_mul(MUL_A);
            a[3] = (a[3] ^ w3).wrapping_mul(MUL_A);
            b[0] = (b[0] ^ w0.rotate_left(17)).wrapping_mul(MUL_B);
            b[1] = (b[1] ^ w1.rotate_left(23)).wrapping_mul(MUL_B);
            b[2] = (b[2] ^ w2.rotate_left(31)).wrapping_mul(MUL_B);
            b[3] = (b[3] ^ w3.rotate_left(41)).wrapping_mul(MUL_B);
        }
        pos += octets.len() * 8;
        for &t in rem {
            let lane = (pos >> 1) & 3;
            let shift = if pos.is_multiple_of(2) { 0 } else { 32 };
            a[lane] = (a[lane] ^ (u64::from(t) << shift)).wrapping_mul(MUL_A);
            b[lane] =
                (b[lane] ^ (u64::from(t).wrapping_mul(MUL_B) << (shift / 2))).wrapping_mul(MUL_B);
            pos += 1;
        }
        let mut lo = a[0]
            ^ a[1].rotate_left(13)
            ^ a[2].rotate_left(29)
            ^ a[3].rotate_left(47)
            ^ (pos as u64);
        lo = (lo ^ (lo >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        lo = (lo ^ (lo >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        lo ^= lo >> 31;

        let mut hi = b[0]
            ^ b[1].rotate_left(19)
            ^ b[2].rotate_left(37)
            ^ b[3].rotate_left(53)
            ^ ((pos as u64).wrapping_mul(MUL_A));
        hi = (hi ^ (hi >> 33)).wrapping_mul(0xff51_afd7_ed55_8ccd);
        hi = (hi ^ (hi >> 33)).wrapping_mul(0xc4ce_b9fe_1a85_ec53);
        hi ^= hi >> 33;

        out.push(u128::from(lo) | (u128::from(hi) << 64));
    }
    out
}

fn chunk_path(dir: &Path, key: u128) -> PathBuf {
    dir.join(format!("chunk-{key:032x}.bin"))
}

fn local_path(dir: &Path, id: u64) -> PathBuf {
    dir.join(format!("local-{id:016x}.bin"))
}

/// Token bucket limiting sustained NVMe writes while allowing bursts.
struct WriteBudget {
    rate_bps: u64,
    burst: f64,
    tokens: f64,
    last: Instant,
}

impl WriteBudget {
    fn new(rate_bps: u64, burst_bytes: u64) -> Self {
        let burst = burst_bytes.max(rate_bps) as f64;
        Self {
            rate_bps,
            burst,
            tokens: burst,
            last: Instant::now(),
        }
    }

    fn try_consume(&mut self, bytes: u64) -> bool {
        if self.rate_bps == 0 || bytes == 0 {
            return true;
        }
        let now = Instant::now();
        let dt = now.saturating_duration_since(self.last).as_secs_f64();
        self.last = now;
        self.tokens = (self.tokens + dt * self.rate_bps as f64).min(self.burst);
        let cost = bytes as f64;
        if self.tokens + 1.0 >= cost {
            self.tokens = (self.tokens - cost).max(0.0);
            true
        } else {
            false
        }
    }
}

/// One KV chunk stored on disk, shared across entries by reference count.
#[derive(Clone, Debug)]
struct DiskChunk {
    refs: u32,
    bytes: u64,
    has_draft: bool,
}

/// One prompt entry indexed in the L3 disk tier.
#[derive(Clone, Debug)]
struct DiskEntry {
    id: u64,
    tokens: Vec<u32>,
    hashes: Vec<u64>,
    chunk_keys: Vec<u128>,
    rings: bool,
    local_file_bytes: u64,
    used: u64,
    in_flight: bool,
}

enum WriterCmd {
    WriteFile {
        path: PathBuf,
        data: Vec<u8>,
        sync: bool,
    },
    RemoveFile(PathBuf),
    Flush(mpsc::Sender<()>),
    Shutdown,
}

struct DiskWriter {
    tx: Option<mpsc::Sender<WriterCmd>>,
    handle: Option<JoinHandle<()>>,
    inflight_bytes: Arc<AtomicUsize>,
    max_inflight_bytes: usize,
}

impl DiskWriter {
    fn spawn(max_inflight_bytes: usize) -> Self {
        let (tx, rx) = mpsc::channel::<WriterCmd>();
        let inflight_bytes = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&inflight_bytes);
        let handle = thread::Builder::new()
            .name("siphon-l3-io".to_owned())
            .spawn(move || {
                while let Ok(cmd) = rx.recv() {
                    match cmd {
                        WriterCmd::WriteFile { path, data, sync } => {
                            let len = data.len();
                            let _ = write_file_atomic(&path, &data, sync);
                            counter.fetch_sub(len, Ordering::Relaxed);
                        }
                        WriterCmd::RemoveFile(path) => {
                            let _ = fs::remove_file(&path);
                        }
                        WriterCmd::Flush(ack) => {
                            let _ = ack.send(());
                        }
                        WriterCmd::Shutdown => break,
                    }
                }
            })
            .expect("spawning L3 disk cache writer thread");
        Self {
            tx: Some(tx),
            handle: Some(handle),
            inflight_bytes,
            max_inflight_bytes,
        }
    }

    fn has_room(&self, bytes: usize) -> bool {
        if self.max_inflight_bytes == 0 {
            return true;
        }
        self.inflight_bytes
            .load(Ordering::Relaxed)
            .saturating_add(bytes)
            <= self.max_inflight_bytes
    }

    fn write_file(&self, path: PathBuf, data: Vec<u8>, sync: bool) {
        if let Some(tx) = &self.tx {
            let len = data.len();
            self.inflight_bytes.fetch_add(len, Ordering::Relaxed);
            if tx.send(WriterCmd::WriteFile { path, data, sync }).is_err() {
                self.inflight_bytes.fetch_sub(len, Ordering::Relaxed);
            }
        }
    }

    fn remove_file(&self, path: PathBuf) {
        if let Some(tx) = &self.tx {
            let _ = tx.send(WriterCmd::RemoveFile(path));
        }
    }

    fn flush(&self) {
        if let Some(tx) = &self.tx {
            let (ack_tx, ack_rx) = mpsc::channel();
            if tx.send(WriterCmd::Flush(ack_tx)).is_ok() {
                let _ = ack_rx.recv();
            }
        }
    }
}

impl Drop for DiskWriter {
    fn drop(&mut self) {
        if let Some(tx) = self.tx.take() {
            let _ = tx.send(WriterCmd::Shutdown);
        }
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

fn write_file_atomic(path: &Path, data: &[u8], sync: bool) -> anyhow::Result<()> {
    let tmp = path.with_extension("tmp");
    let file = File::create(&tmp).with_context(|| format!("creating {}", tmp.display()))?;
    let mut w = BufWriter::with_capacity(1 << 20, file);
    w.write_all(data)?;
    let file = w.into_inner()?;
    if sync {
        file.sync_all()?;
    }
    fs::rename(&tmp, path)?;
    Ok(())
}

/// Active L3 NVMe tier owned by [`HostCache`].
pub(super) struct DiskTier {
    config: DiskTierConfig,
    header: Value,
    manifest_key: u64,
    next_id: u64,
    entries: Vec<DiskEntry>,
    chunks: HashMap<u128, DiskChunk>,
    disk_bytes: u64,
    budget: WriteBudget,
    writer: DiskWriter,
    stats: DiskTierStats,
    manifest_dirty: bool,
    last_manifest_save: Instant,
    last_throttle_log: Option<Instant>,
}

impl DiskTier {
    /// Opens or initializes an L3 disk cache in `config.dir`.
    ///
    /// Returns the initialized tier, a [`Loaded`] status describing what was found on disk,
    /// and the list of disk entry IDs ordered newest-last (oldest-first) for RAM pre-warming.
    #[expect(
        clippy::too_many_lines,
        reason = "manifest validation, chunk/local verification, and cleanup"
    )]
    pub(super) fn open(
        config: DiskTierConfig,
        header: Value,
        manifest_key: u64,
        has_draft_kv: bool,
    ) -> anyhow::Result<(Self, Loaded, Vec<u64>)> {
        let started = Instant::now();
        fs::create_dir_all(&config.dir)
            .with_context(|| format!("creating {}", config.dir.display()))?;
        let manifest_path = config.dir.join(MANIFEST);
        let budget = WriteBudget::new(config.max_write_bytes_per_sec, config.write_burst_bytes);
        let writer = DiskWriter::spawn(config.max_inflight_bytes);

        let mut tier = Self {
            stats: DiskTierStats {
                max_bytes: config.max_bytes,
                ..DiskTierStats::default()
            },
            config,
            header,
            manifest_key,
            next_id: 1,
            entries: Vec::new(),
            chunks: HashMap::new(),
            disk_bytes: 0,
            budget,
            writer,
            manifest_dirty: false,
            last_manifest_save: Instant::now(),
            last_throttle_log: None,
        };

        let Ok(text) = fs::read_to_string(&manifest_path) else {
            // Clean up any orphaned partial files when no manifest exists.
            let _ = remove_orphan_files(&tier.config.dir, &HashSet::new());
            return Ok((tier, Loaded::Missing, Vec::new()));
        };

        let saved: Value = serde_json::from_str(&text).unwrap_or(Value::Null);
        let differs = ["format", "tier", "identity", "layout"]
            .into_iter()
            .find(|k| saved.get(k) != tier.header.get(k));
        if let Some(what) = differs {
            discard(&tier.config.dir)?;
            return Ok((
                tier,
                Loaded::Discarded(format!("the saved {what} differs from this server's")),
                Vec::new(),
            ));
        }

        tier.next_id = saved["next_id"].as_u64().unwrap_or(1).max(1);
        let mut chunk_meta: HashMap<u128, (u64, bool)> = HashMap::new();
        if let Some(arr) = saved["chunks"].as_array() {
            for item in arr {
                let Some(key_str) = item["key"].as_str() else {
                    continue;
                };
                let Ok(key) = u128::from_str_radix(key_str, 16) else {
                    continue;
                };
                let bytes = item["bytes"].as_u64().unwrap_or(0);
                let draft = item["draft"].as_bool().unwrap_or(false);
                let path = chunk_path(&tier.config.dir, key);
                if fs::metadata(&path).is_ok_and(|m| m.len() == bytes && bytes > 0) {
                    chunk_meta.insert(key, (bytes, draft));
                }
            }
        }

        let mut skipped = 0usize;
        let mut indexed_bytes = 0u64;
        if let Some(arr) = saved["entries"].as_array() {
            for item in arr {
                let Some(id) = item["id"].as_u64() else {
                    skipped += 1;
                    continue;
                };
                let used = item["used"].as_u64().unwrap_or(0);
                let rings = item["rings"].as_bool().unwrap_or(false);
                let local_file_bytes = item["local_bytes"].as_u64().unwrap_or(0);
                let Some(keys_arr) = item["chunks"].as_array() else {
                    skipped += 1;
                    continue;
                };
                let mut keys = Vec::with_capacity(keys_arr.len());
                let mut ok = !keys_arr.is_empty();
                for k in keys_arr {
                    let Some(ks) = k.as_str() else {
                        ok = false;
                        break;
                    };
                    let Ok(key) = u128::from_str_radix(ks, 16) else {
                        ok = false;
                        break;
                    };
                    if !chunk_meta.contains_key(&key) {
                        ok = false;
                        break;
                    }
                    keys.push(key);
                }
                if !ok {
                    skipped += 1;
                    continue;
                }
                let lpath = local_path(&tier.config.dir, id);
                let Ok((file_rings, tokens, actual_size)) =
                    read_local_header(&lpath, tier.manifest_key, id)
                else {
                    skipped += 1;
                    continue;
                };
                if actual_size != local_file_bytes
                    || file_rings != rings
                    || tokens.len() != keys.len() * HOST_CHUNK_POSITIONS
                {
                    skipped += 1;
                    continue;
                }
                let hashes = chunk_hashes(&tokens, HOST_CHUNK_POSITIONS);
                // Register chunk refcounts.
                for &key in &keys {
                    let (bytes, draft) = chunk_meta[&key];
                    let entry = tier.chunks.entry(key).or_insert_with(|| {
                        tier.disk_bytes += bytes;
                        indexed_bytes += bytes;
                        DiskChunk {
                            refs: 0,
                            bytes,
                            has_draft: draft && has_draft_kv,
                        }
                    });
                    entry.refs += 1;
                }
                tier.disk_bytes += local_file_bytes;
                indexed_bytes += local_file_bytes;
                tier.entries.push(DiskEntry {
                    id,
                    tokens,
                    hashes,
                    chunk_keys: keys,
                    rings,
                    local_file_bytes,
                    used,
                    in_flight: false,
                });
                tier.next_id = tier.next_id.max(id.saturating_add(1));
            }
        }

        // Enforce max_bytes if the configured disk cap shrank since the last run.
        tier.evict_to_fit(0, None);

        // Remove any orphaned chunk/local/tmp files not referenced by live entries.
        let mut live_files: HashSet<String> = HashSet::new();
        live_files.insert(MANIFEST.to_owned());
        for &key in tier.chunks.keys() {
            live_files.insert(format!("chunk-{key:032x}.bin"));
        }
        for e in &tier.entries {
            live_files.insert(format!("local-{:016x}.bin", e.id));
        }
        let _ = remove_orphan_files(&tier.config.dir, &live_files);

        tier.entries.sort_by_key(|e| e.used);
        let order: Vec<u64> = tier.entries.iter().map(|e| e.id).collect();
        tier.sync_stats();

        let loaded = Loaded::Loaded {
            entries: tier.entries.len(),
            skipped,
            bytes: indexed_bytes,
            seconds: started.elapsed().as_secs_f64(),
        };
        Ok((tier, loaded, order))
    }

    fn sync_stats(&mut self) {
        self.stats.entries = self.entries.len();
        self.stats.unique_chunks = self.chunks.len();
        self.stats.disk_bytes = self.disk_bytes;
    }

    pub(super) fn stats(&self) -> DiskTierStats {
        self.stats
    }

    pub(super) fn max_used_tick(&self) -> u64 {
        self.entries.iter().map(|e| e.used).max().unwrap_or(0)
    }

    pub(super) fn max_len_below(&self, limit: usize) -> usize {
        self.entries
            .iter()
            .map(|e| e.tokens.len())
            .filter(|&len| len < limit)
            .max()
            .unwrap_or(0)
    }

    fn complete(&self, e: &DiskEntry, has_draft_kv: bool) -> bool {
        e.rings
            && (!has_draft_kv
                || e.chunk_keys
                    .iter()
                    .all(|k| self.chunks.get(k).is_some_and(|c| c.has_draft)))
    }

    pub(super) fn has_draft(&self, id: u64, has_draft_kv: bool) -> bool {
        self.entries
            .iter()
            .find(|e| e.id == id)
            .is_some_and(|e| self.complete(e, has_draft_kv))
    }

    pub(super) fn lookup(
        &self,
        prompt: &[u32],
        draft: bool,
        has_draft_kv: bool,
        hashes: &PrefixHashes,
    ) -> Option<(u64, usize)> {
        let mut best: Option<(u64, usize)> = None;
        for e in &self.entries {
            let len = e.tokens.len();
            if len == 0
                || best.is_some_and(|(_, best_len)| len <= best_len)
                || prompt.len() <= len
                || (draft && !self.complete(e, has_draft_kv))
            {
                continue;
            }
            if e.hashes.last().copied() == hashes.hash_at_len(len) && prompt.starts_with(&e.tokens)
            {
                best = Some((e.id, len));
            }
        }
        best
    }

    fn evict_entry_index(&mut self, idx: usize) {
        let e = self.entries.swap_remove(idx);
        self.disk_bytes = self.disk_bytes.saturating_sub(e.local_file_bytes);
        self.writer.remove_file(local_path(&self.config.dir, e.id));
        for key in e.chunk_keys {
            if let Some(c) = self.chunks.get_mut(&key) {
                c.refs = c.refs.saturating_sub(1);
                if c.refs == 0 {
                    let bytes = c.bytes;
                    self.chunks.remove(&key);
                    self.disk_bytes = self.disk_bytes.saturating_sub(bytes);
                    self.writer.remove_file(chunk_path(&self.config.dir, key));
                }
            }
        }
        self.manifest_dirty = true;
    }

    fn evict_to_fit(&mut self, incoming: u64, protect_id: Option<u64>) -> bool {
        if incoming > self.config.max_bytes {
            return false;
        }
        while self.disk_bytes.saturating_add(incoming) > self.config.max_bytes {
            let Some(lru) = self
                .entries
                .iter()
                .enumerate()
                .filter(|(_, e)| Some(e.id) != protect_id)
                .min_by_key(|(_, e)| e.used)
                .map(|(i, _)| i)
            else {
                return false;
            };
            self.evict_entry_index(lru);
            self.stats.evictions += 1;
        }
        true
    }

    fn build_manifest_bytes(&self) -> Vec<u8> {
        let mut root = self.header.clone();
        root["next_id"] = json!(self.next_id);
        let mut chunks: Vec<Value> = self
            .chunks
            .iter()
            .map(|(k, c)| {
                json!({
                    "key": format!("{k:032x}"),
                    "bytes": c.bytes,
                    "draft": c.has_draft,
                })
            })
            .collect();
        chunks.sort_by(|a, b| a["key"].as_str().cmp(&b["key"].as_str()));
        let mut entries: Vec<&DiskEntry> = self.entries.iter().collect();
        entries.sort_by_key(|e| e.used);
        let entries_json: Vec<Value> = entries
            .into_iter()
            .map(|e| {
                json!({
                    "id": e.id,
                    "tokens": e.tokens.len(),
                    "used": e.used,
                    "rings": e.rings,
                    "local_bytes": e.local_file_bytes,
                    "chunks": e.chunk_keys.iter().map(|k| format!("{k:032x}")).collect::<Vec<_>>(),
                })
            })
            .collect();
        root["chunks"] = Value::Array(chunks);
        root["entries"] = Value::Array(entries_json);
        root.to_string().into_bytes()
    }

    fn maybe_checkpoint_manifest(&mut self) {
        if !self.manifest_dirty {
            return;
        }
        if self.last_manifest_save.elapsed() < MANIFEST_CHECKPOINT_INTERVAL {
            return;
        }
        let data = self.build_manifest_bytes();
        self.writer
            .write_file(self.config.dir.join(MANIFEST), data, false);
        self.manifest_dirty = false;
        self.last_manifest_save = Instant::now();
    }
}

fn read_local_header(
    path: &Path,
    manifest_key: u64,
    expected_id: u64,
) -> anyhow::Result<(bool, Vec<u32>, u64)> {
    let file = File::open(path)?;
    let size = file.metadata()?.len();
    let mut r = BufReader::new(file);
    let mut fixed = [0u8; FILE_HEADER_BYTES];
    r.read_exact(&mut fixed)?;
    ensure!(&fixed[..8] == LOCAL_MAGIC, "not an L3 local file");
    let ver = u32::from_le_bytes(fixed[8..12].try_into().expect("4 bytes"));
    let flags = u32::from_le_bytes(fixed[12..16].try_into().expect("4 bytes"));
    let key = u64::from_le_bytes(fixed[16..24].try_into().expect("8 bytes"));
    let id = u64::from_le_bytes(fixed[24..32].try_into().expect("8 bytes"));
    let tokens_len = usize::try_from(u64::from_le_bytes(
        fixed[32..40].try_into().expect("8 bytes"),
    ))?;
    ensure!(
        ver == DISK_FORMAT_L3 && key == manifest_key && id == expected_id,
        "local file metadata mismatch"
    );
    let mut raw = vec![0u8; tokens_len * 4 + pad8(tokens_len * 4)];
    r.read_exact(&mut raw)?;
    let tokens: Vec<u32> = raw[..tokens_len * 4]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| u32::from_le_bytes(*b))
        .collect();
    Ok((flags & 1 != 0, tokens, size))
}

fn remove_orphan_files(dir: &Path, live: &HashSet<String>) -> anyhow::Result<()> {
    for item in fs::read_dir(dir)? {
        let path = item?.path();
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or_default();
        let managed = (name.starts_with("chunk-")
            || name.starts_with("local-")
            || name.starts_with("entry-")
            || name == "manifest.tmp")
            && Path::new(name)
                .extension()
                .is_some_and(|x| x == "bin" || x == "tmp");
        if managed && !live.contains(name) {
            let _ = fs::remove_file(&path);
        }
    }
    Ok(())
}

/// Removes all L3 files (`chunk-*.bin`, `local-*.bin`) alongside legacy files.
pub(super) fn discard_l3_files(dir: &Path) -> anyhow::Result<()> {
    for item in fs::read_dir(dir)? {
        let path = item?.path();
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or_default();
        if (name.starts_with("chunk-") || name.starts_with("local-"))
            && Path::new(name)
                .extension()
                .is_some_and(|x| x == "bin" || x == "tmp")
        {
            match fs::remove_file(&path) {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => bail!("removing {}: {e}", path.display()),
            }
        }
    }
    Ok(())
}

impl HostCache {
    /// Header JSON and key for the L3 disk format.
    fn l3_header(&self, identity: &Value) -> (Value, u64) {
        let header = json!({
            "format": DISK_FORMAT_L3,
            "tier": "l3",
            "identity": identity,
            "layout": self.layout(),
        });
        let key = super::disk::fnv(header.to_string().as_bytes());
        (header, key)
    }

    /// Enables an active L3 NVMe disk cache tier, loads any matching manifest/index from
    /// `config.dir`, and pre-warms pinned host RAM with the most recently used entries that
    /// fit without evicting.
    pub fn enable_disk_cache(
        &mut self,
        config: DiskTierConfig,
        identity: &Value,
    ) -> anyhow::Result<Loaded> {
        ensure!(config.max_bytes > 0, "--disk-cache must be positive");
        let (header, key) = self.l3_header(identity);
        let has_draft_kv = !self.draft.caches.is_empty();
        let (tier, loaded, order) = DiskTier::open(config, header, key, has_draft_kv)?;
        self.tick = self.tick.max(tier.max_used_tick());
        self.disk = Some(tier);

        // Pre-warm pinned RAM with the newest entries from disk that fit without eviction.
        if !order.is_empty() {
            let mut warmed = 0usize;
            for &id in order.iter().rev() {
                let needed_upper = self
                    .disk
                    .as_ref()
                    .and_then(|d| d.entries.iter().find(|e| e.id == id))
                    .map_or(usize::MAX, |e| e.chunk_keys.len() + self.local_chunks());
                if self.pool.free.len() < needed_upper {
                    break;
                }
                if self.promote_disk_inner(id, false)?.is_some() {
                    warmed += 1;
                }
            }
            if warmed > 0 {
                self.note(format!(
                    "disk cache: pre-warmed {warmed} most recent entries into pinned host RAM"
                ));
            }
        }
        Ok(loaded)
    }

    /// Returns statistics for the active L3 disk tier, if enabled.
    #[must_use]
    pub fn disk_stats(&self) -> Option<DiskTierStats> {
        self.disk.as_ref().map(DiskTier::stats)
    }

    /// Longest disk entry strictly shorter than `limit`, or 0 when none.
    pub fn disk_max_len_below(&self, limit: usize) -> usize {
        self.disk.as_ref().map_or(0, |d| d.max_len_below(limit))
    }

    /// Longest L3 disk entry that `prompt` starts with and goes past, with draft state if
    /// `draft`.
    pub fn lookup_disk(
        &self,
        prompt: &[u32],
        draft: bool,
        hashes: &PrefixHashes,
    ) -> Option<(u64, usize)> {
        let has_draft_kv = !self.draft.caches.is_empty();
        self.disk
            .as_ref()?
            .lookup(prompt, draft, has_draft_kv, hashes)
    }

    /// Whether L3 disk entry `id` holds complete draft state.
    pub fn has_disk_draft(&self, id: u64) -> bool {
        let has_draft_kv = !self.draft.caches.is_empty();
        self.disk
            .as_ref()
            .is_some_and(|d| d.has_draft(id, has_draft_kv))
    }

    /// Spills an evicted RAM entry `e` to the L3 disk tier if enabled and admitted by
    /// anti-churn guards. Called right before releasing `e`'s pinned chunks.
    #[expect(
        clippy::too_many_lines,
        reason = "admission guards, deduplication, budgeting, and async enqueue"
    )]
    pub(super) fn spill_evicted_to_disk(
        &mut self,
        e: &HostEntry,
        protect_disk_id: Option<u64>,
        force_shutdown: bool,
    ) {
        let (target_bytes, draft_bytes) = self.parts();
        let has_draft_kv = !self.draft.caches.is_empty();
        let entry_complete =
            e.rings && (!has_draft_kv || e.kv.iter().all(|&c| self.drafted[c as usize]));
        let Some(disk) = self.disk.as_mut() else {
            return;
        };

        // 1. Minimum spill size guard (skipped on graceful shutdown so even 1-chunk RAM
        //    entries survive restarts).
        if !force_shutdown && e.tokens.len() < disk.config.min_spill_tokens {
            disk.stats.skips_small += 1;
            return;
        }

        // 2. Clean-entry fast path: already on disk with matching content & draft state.
        if let Some(id) = e.disk_id
            && !e.disk_dirty
            && let Some(de) = disk.entries.iter_mut().find(|de| de.id == id)
            && de.tokens.len() == e.tokens.len()
            && de.hashes.last() == e.hashes.last()
        {
            de.used = de.used.max(e.used);
            disk.stats.skips_clean += 1;
            disk.manifest_dirty = true;
            disk.maybe_checkpoint_manifest();
            return;
        }

        // 3. Check if an exact match already exists on disk with equal or better draft state:
        //    if so, just touch its LRU tick (zero bytes written). Strictly longer entries do
        //    not replace shorter prefixes on disk, because a shorter prompt cannot hit a
        //    longer entry and KV chunks are already 100% deduplicated.
        let full_hash = e.hashes.last().copied();
        let exact_idx = disk.entries.iter().position(|de| {
            de.tokens.len() == e.tokens.len()
                && de.hashes.last().copied() == full_hash
                && de.tokens == e.tokens
        });
        if let Some(idx) = exact_idx {
            let disk_complete = disk.complete(&disk.entries[idx], has_draft_kv);
            if !entry_complete || disk_complete {
                disk.entries[idx].used = disk.entries[idx].used.max(e.used);
                disk.stats.skips_clean += 1;
                disk.manifest_dirty = true;
                disk.maybe_checkpoint_manifest();
                return;
            }
        }

        let keys = chunk_keys_128(&e.tokens, HOST_CHUNK_POSITIONS);
        if keys.len() != e.kv.len() {
            return;
        }

        // Compute exact incremental bytes that actually need writing after deduplication.
        let mut chunks_to_write: Vec<(usize, u128, bool, u64)> = Vec::new();
        let mut incremental_bytes = 0u64;
        let mut net_new_disk_bytes = 0u64;
        for (j, (&c, &key)) in e.kv.iter().zip(&keys).enumerate() {
            let c_drafted = self.drafted[c as usize] && draft_bytes > 0;
            let payload_len = (target_bytes + if c_drafted { draft_bytes } else { 0 }) as u64;
            let file_bytes = FILE_HEADER_BYTES as u64 + payload_len;
            match disk.chunks.get(&key) {
                Some(existing) if existing.has_draft || !c_drafted => {}
                Some(existing) => {
                    // Upgrade existing chunk file to include draft KV part.
                    chunks_to_write.push((j, key, c_drafted, file_bytes));
                    incremental_bytes += file_bytes;
                    net_new_disk_bytes += file_bytes.saturating_sub(existing.bytes);
                }
                None => {
                    chunks_to_write.push((j, key, c_drafted, file_bytes));
                    incremental_bytes += file_bytes;
                    net_new_disk_bytes += file_bytes;
                }
            }
        }
        let local_payload = (self.local_bytes + if e.rings { self.rings_bytes } else { 0 }) as u64;
        let tokens_padded = (e.tokens.len() * 4 + pad8(e.tokens.len() * 4)) as u64;
        let local_file_bytes = FILE_HEADER_BYTES as u64 + tokens_padded + local_payload;
        incremental_bytes += local_file_bytes;
        net_new_disk_bytes += local_file_bytes;

        if !force_shutdown {
            // Backpressure guard: never grow heap spill buffers unboundedly.
            let inc_usize = usize::try_from(incremental_bytes).unwrap_or(usize::MAX);
            if !disk.writer.has_room(inc_usize) {
                disk.stats.skips_throttled += 1;
                Self::note_throttled(disk, &mut self.notes, "writer queue full");
                return;
            }
            // Wear-protection write-rate token bucket.
            if !disk.budget.try_consume(incremental_bytes) {
                disk.stats.skips_throttled += 1;
                Self::note_throttled(disk, &mut self.notes, "write-rate cap reached");
                return;
            }
        }

        // Wait for any in-flight GPU->host D2H copies before reading pinned memory on CPU.
        if let Some(event) = self.pending.as_ref()
            && event.synchronize().is_err()
        {
            return;
        }
        // Re-borrow disk after sync_event.
        let Some(disk) = self.disk.as_mut() else {
            return;
        };

        // Drop any exact entry on disk being upgraded with draft state.
        let mut idx = 0;
        while idx < disk.entries.len() {
            let de = &disk.entries[idx];
            let is_same = de.tokens.len() == e.tokens.len()
                && de.hashes.last().copied() == full_hash
                && de.tokens == e.tokens;
            if is_same && Some(de.id) != protect_disk_id {
                disk.evict_entry_index(idx);
            } else {
                idx += 1;
            }
        }

        // Ensure room within `--disk-cache` budget.
        if !disk.evict_to_fit(net_new_disk_bytes, protect_disk_id) {
            self.notes.push(format!(
                "disk cache: {} tokens ({} MiB) do not fit --disk-cache ({:.2} GiB)",
                e.tokens.len(),
                net_new_disk_bytes >> 20,
                disk.config.max_bytes as f64 / f64::from(1u32 << 30)
            ));
            return;
        }

        let manifest_key = disk.manifest_key;
        let dir = disk.config.dir.clone();
        let entry_id = disk.next_id;
        disk.next_id += 1;

        // Materialize only missing/upgraded chunk buffers from pinned RAM and queue writes.
        let mut written_chunks = 0usize;
        for &(j, key, c_drafted, file_bytes) in &chunks_to_write {
            let c = e.kv[j];
            let payload_len = target_bytes + if c_drafted { draft_bytes } else { 0 };
            let mut buf = Vec::with_capacity(FILE_HEADER_BYTES + payload_len);
            buf.extend_from_slice(CHUNK_MAGIC);
            buf.extend_from_slice(&DISK_FORMAT_L3.to_le_bytes());
            buf.extend_from_slice(&u32::from(c_drafted).to_le_bytes());
            buf.extend_from_slice(&manifest_key.to_le_bytes());
            buf.extend_from_slice(&(key as u64).to_le_bytes());
            buf.extend_from_slice(&((key >> 64) as u64).to_le_bytes());
            // SAFETY: `self.pending` was synchronized above, and chunk `c` is live in `e`.
            buf.extend_from_slice(unsafe { Self::host_bytes(self.pool.address(c), payload_len) });
            debug_assert_eq!(buf.len() as u64, file_bytes);
            disk.writer.write_file(chunk_path(&dir, key), buf, false);
            written_chunks += 1;
        }

        // Materialize local state buffer (header + tokens + rings/conv + optional draft rings).
        let mut lbuf = Vec::with_capacity(usize::try_from(local_file_bytes).unwrap_or(0));
        lbuf.extend_from_slice(LOCAL_MAGIC);
        lbuf.extend_from_slice(&DISK_FORMAT_L3.to_le_bytes());
        lbuf.extend_from_slice(&u32::from(e.rings).to_le_bytes());
        lbuf.extend_from_slice(&manifest_key.to_le_bytes());
        lbuf.extend_from_slice(&entry_id.to_le_bytes());
        lbuf.extend_from_slice(&(e.tokens.len() as u64).to_le_bytes());
        for &t in &e.tokens {
            lbuf.extend_from_slice(&t.to_le_bytes());
        }
        lbuf.resize(lbuf.len() + pad8(e.tokens.len() * 4), 0);
        let mut left = self.local_bytes + if e.rings { self.rings_bytes } else { 0 };
        for &c in &e.local {
            let len = left.min(self.pool.chunk_bytes);
            // SAFETY: `self.pending` was synchronized above, and chunk `c` is live in `e`.
            lbuf.extend_from_slice(unsafe { Self::host_bytes(self.pool.address(c), len) });
            left -= len;
        }
        debug_assert_eq!(lbuf.len() as u64, local_file_bytes);
        disk.writer
            .write_file(local_path(&dir, entry_id), lbuf, false);

        // Update chunk refcounts and byte accounting.
        for (&c, &key) in e.kv.iter().zip(&keys) {
            let c_drafted = self.drafted[c as usize] && draft_bytes > 0;
            let file_bytes = FILE_HEADER_BYTES as u64
                + (target_bytes + if c_drafted { draft_bytes } else { 0 }) as u64;
            if let Some(existing) = disk.chunks.get_mut(&key) {
                existing.refs += 1;
                if c_drafted && !existing.has_draft {
                    disk.disk_bytes += file_bytes.saturating_sub(existing.bytes);
                    existing.bytes = file_bytes;
                    existing.has_draft = true;
                }
            } else {
                disk.disk_bytes += file_bytes;
                disk.chunks.insert(
                    key,
                    DiskChunk {
                        refs: 1,
                        bytes: file_bytes,
                        has_draft: c_drafted,
                    },
                );
            }
        }
        disk.disk_bytes += local_file_bytes;
        disk.entries.push(DiskEntry {
            id: entry_id,
            tokens: e.tokens.clone(),
            hashes: e.hashes.clone(),
            chunk_keys: keys,
            rings: e.rings,
            local_file_bytes,
            used: e.used,
            in_flight: true,
        });
        disk.stats.spills += 1;
        disk.stats.bytes_written += incremental_bytes;
        disk.manifest_dirty = true;
        disk.sync_stats();
        disk.maybe_checkpoint_manifest();

        if !force_shutdown {
            let dedup_chunks = e.kv.len().saturating_sub(written_chunks);
            self.notes.push(format!(
                "disk cache: spilled {} tokens ({} MiB written, {written_chunks} new + \
                 {dedup_chunks} dedup chunks); {:.2}/{:.2} GiB used by {} entries ({} unique chunks)",
                e.tokens.len(),
                incremental_bytes >> 20,
                disk.disk_bytes as f64 / f64::from(1u32 << 30),
                disk.config.max_bytes as f64 / f64::from(1u32 << 30),
                disk.entries.len(),
                disk.chunks.len(),
            ));
        }
    }

    fn note_throttled(disk: &mut DiskTier, notes: &mut Vec<String>, reason: &str) {
        let should_log = disk
            .last_throttle_log
            .is_none_or(|t| t.elapsed() >= LOG_THROTTLE_INTERVAL);
        if should_log {
            disk.last_throttle_log = Some(Instant::now());
            notes.push(format!(
                "disk cache: skipped spilling an evicted entry ({reason}; {} skipped so far)",
                disk.stats.skips_throttled
            ));
        }
    }

    /// Promotes L3 disk entry `id` into pinned host RAM, sharing any KV chunks already in
    /// RAM and reading only the missing KV chunks + local state from NVMe.
    ///
    /// Returns `Ok(Some((host_entry_index, bytes_read, seconds)))`, or `Ok(None)` if the
    /// entry could not be promoted (e.g. missing/corrupt file or larger than RAM pool).
    pub fn promote_disk(&mut self, id: u64) -> anyhow::Result<Option<(usize, u64, f64)>> {
        self.promote_disk_inner(id, true)
    }

    #[expect(
        clippy::too_many_lines,
        reason = "RAM allocation, shared chunk reuse, disk read, and rollback"
    )]
    fn promote_disk_inner(
        &mut self,
        id: u64,
        count_hit: bool,
    ) -> anyhow::Result<Option<(usize, u64, f64)>> {
        let started = Instant::now();
        let Some(disk) = self.disk.as_mut() else {
            return Ok(None);
        };
        let Some(pos) = disk.entries.iter().position(|e| e.id == id) else {
            return Ok(None);
        };
        if disk.entries[pos].in_flight {
            disk.writer.flush();
            for e in &mut disk.entries {
                e.in_flight = false;
            }
        }
        let de = disk.entries[pos].clone();
        let dir = disk.config.dir.clone();
        let manifest_key = disk.manifest_key;

        // If an exact match already lives in RAM (e.g., from concurrent promotion), return it.
        let full_hash = de.hashes.last().copied();
        if let Some(h) = self.entries.iter().position(|e| {
            e.tokens.len() == de.tokens.len()
                && e.hashes.last().copied() == full_hash
                && e.tokens == de.tokens
        }) {
            return Ok(Some((h, 0, started.elapsed().as_secs_f64())));
        }

        // Allocate pinned RAM chunks (sharing any prefix KV chunks already in RAM) while
        // protecting `id` from being evicted if RAM eviction spills to disk.
        let Some((kv, local, shared)) = self.allocate_with_protect(&de.tokens, Some(id))? else {
            return Ok(None);
        };

        let (target_bytes, draft_bytes) = self.parts();
        let mut bytes_read = 0u64;
        let read_result = (|| -> anyhow::Result<()> {
            for (j, (&c, &key)) in kv.iter().zip(&de.chunk_keys).enumerate() {
                let path = chunk_path(&dir, key);
                let at = self.pool.address(c);
                if j < shared {
                    // Chunk already in RAM; only read draft part if RAM lacks it and disk has it.
                    let disk_has_draft = self
                        .disk
                        .as_ref()
                        .and_then(|d| d.chunks.get(&key))
                        .is_some_and(|dc| dc.has_draft);
                    if disk_has_draft && !self.drafted[c as usize] && draft_bytes > 0 {
                        let n = read_chunk_into(
                            &path,
                            manifest_key,
                            key,
                            target_bytes,
                            draft_bytes,
                            at,
                            true,
                        )?;
                        self.drafted[c as usize] = true;
                        bytes_read += n;
                    }
                    continue;
                }
                let (has_draft, n) =
                    read_chunk_full_into(&path, manifest_key, key, target_bytes, draft_bytes, at)?;
                self.drafted[c as usize] = has_draft;
                bytes_read += n;
            }
            let lpath = local_path(&dir, de.id);
            let n = read_local_into(
                &lpath,
                manifest_key,
                de.id,
                de.tokens.len(),
                self.local_bytes,
                if de.rings { self.rings_bytes } else { 0 },
                &self.pool,
                &local,
            )?;
            bytes_read += n;
            Ok(())
        })();

        if let Err(err) = read_result {
            for &c in kv.iter().chain(&local) {
                self.pool.release(c);
            }
            // Drop the corrupt/missing disk entry so future lookups won't hit it again.
            if let Some(disk) = self.disk.as_mut()
                && let Some(idx) = disk.entries.iter().position(|e| e.id == id)
            {
                disk.evict_entry_index(idx);
                disk.sync_stats();
            }
            self.note(format!(
                "disk cache: failed to promote {} tokens (falling back to prefill): {err:#}",
                de.tokens.len()
            ));
            return Ok(None);
        }

        self.tick += 1;
        let used = if count_hit { self.tick } else { de.used };
        self.entries.push(HostEntry {
            tokens: de.tokens,
            hashes: de.hashes,
            kv,
            local,
            rings: de.rings,
            used,
            used_at: Instant::now(),
            disk_id: Some(id),
            disk_dirty: false,
        });
        let h = self.entries.len() - 1;
        let seconds = started.elapsed().as_secs_f64();
        if let Some(disk) = self.disk.as_mut() {
            if let Some(entry) = disk.entries.iter_mut().find(|e| e.id == id) {
                entry.used = used;
            }
            if count_hit {
                disk.stats.hits += 1;
                disk.stats.bytes_read += bytes_read;
                disk.manifest_dirty = true;
                disk.maybe_checkpoint_manifest();
            }
        }
        Ok(Some((h, bytes_read, seconds)))
    }

    /// Saves all RAM entries into the active L3 disk store on graceful shutdown (reusing
    /// already-persisted chunks and clean entries) and writes `manifest.json` with `sync_all`.
    pub(super) fn save_l3_dir(&mut self, dir: &Path, identity: &Value) -> anyhow::Result<Saved> {
        let started = Instant::now();
        self.copy.synchronize()?;
        self.main.synchronize()?;

        // Snapshot entries sorted oldest -> newest so newest entries survive budget capping.
        let mut order: Vec<usize> = (0..self.entries.len()).collect();
        order.sort_by_key(|&i| self.entries[i].used);

        let before_written = self.disk.as_ref().map_or(0, |d| d.stats.bytes_written);
        // Ensure manifest identity/dir matches if caller passed the active dir.
        let layout = self.layout();
        if let Some(disk) = self.disk.as_mut() {
            if disk.config.dir != dir {
                disk.config.dir = dir.to_path_buf();
            }
            let h = json!({
                "format": DISK_FORMAT_L3,
                "tier": "l3",
                "identity": identity,
                "layout": layout,
            });
            let k = super::disk::fnv(h.to_string().as_bytes());
            disk.header = h;
            disk.manifest_key = k;
        }

        // Persist every dirty RAM entry into the L3 store without rate limiting.
        for idx in order {
            let snapshot = HostEntry {
                tokens: self.entries[idx].tokens.clone(),
                hashes: self.entries[idx].hashes.clone(),
                kv: self.entries[idx].kv.clone(),
                local: self.entries[idx].local.clone(),
                rings: self.entries[idx].rings,
                used: self.entries[idx].used,
                used_at: self.entries[idx].used_at,
                disk_id: self.entries[idx].disk_id,
                disk_dirty: self.entries[idx].disk_dirty,
            };
            self.spill_evicted_to_disk(&snapshot, None, true);
        }

        let Some(disk) = self.disk.as_mut() else {
            bail!("L3 disk cache is not enabled");
        };
        disk.writer.flush();
        for e in &mut disk.entries {
            e.in_flight = false;
        }
        let manifest_bytes = disk.build_manifest_bytes();
        write_atomic(&dir.join(MANIFEST), &manifest_bytes)?;
        disk.manifest_dirty = false;
        disk.last_manifest_save = Instant::now();
        disk.sync_stats();

        let written_now = disk.stats.bytes_written.saturating_sub(before_written);
        Ok(Saved {
            entries: disk.entries.len(),
            bytes: written_now,
            seconds: started.elapsed().as_secs_f64(),
        })
    }
}

fn read_chunk_full_into(
    path: &Path,
    manifest_key: u64,
    expected_key: u128,
    target_bytes: usize,
    draft_bytes: usize,
    dst_addr: usize,
) -> anyhow::Result<(bool, u64)> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let size = file.metadata()?.len();
    let mut r = BufReader::with_capacity(1 << 20, file);
    let mut fixed = [0u8; FILE_HEADER_BYTES];
    r.read_exact(&mut fixed)?;
    ensure!(&fixed[..8] == CHUNK_MAGIC, "not an L3 chunk file");
    let ver = u32::from_le_bytes(fixed[8..12].try_into().expect("4 bytes"));
    let flags = u32::from_le_bytes(fixed[12..16].try_into().expect("4 bytes"));
    let key = u64::from_le_bytes(fixed[16..24].try_into().expect("8 bytes"));
    let lo = u64::from_le_bytes(fixed[24..32].try_into().expect("8 bytes"));
    let hi = u64::from_le_bytes(fixed[32..40].try_into().expect("8 bytes"));
    let ckey = u128::from(lo) | (u128::from(hi) << 64);
    ensure!(
        ver == DISK_FORMAT_L3 && key == manifest_key && ckey == expected_key,
        "chunk file header mismatch"
    );
    let has_draft = (flags & 1 != 0) && draft_bytes > 0;
    let payload_len = target_bytes + if has_draft { draft_bytes } else { 0 };
    ensure!(
        size == FILE_HEADER_BYTES as u64 + payload_len as u64,
        "chunk file size {size} != expected {}",
        FILE_HEADER_BYTES + payload_len
    );
    // SAFETY: `dst_addr` is a freshly allocated pinned chunk not read by any other entry.
    r.read_exact(unsafe { HostCache::host_bytes_mut(dst_addr, payload_len) })?;
    Ok((has_draft, payload_len as u64))
}

fn read_chunk_into(
    path: &Path,
    manifest_key: u64,
    expected_key: u128,
    target_bytes: usize,
    draft_bytes: usize,
    dst_addr: usize,
    draft_only: bool,
) -> anyhow::Result<u64> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut r = BufReader::with_capacity(1 << 20, file);
    let mut fixed = [0u8; FILE_HEADER_BYTES];
    r.read_exact(&mut fixed)?;
    ensure!(&fixed[..8] == CHUNK_MAGIC, "not an L3 chunk file");
    let ver = u32::from_le_bytes(fixed[8..12].try_into().expect("4 bytes"));
    let flags = u32::from_le_bytes(fixed[12..16].try_into().expect("4 bytes"));
    let key = u64::from_le_bytes(fixed[16..24].try_into().expect("8 bytes"));
    let lo = u64::from_le_bytes(fixed[24..32].try_into().expect("8 bytes"));
    let hi = u64::from_le_bytes(fixed[32..40].try_into().expect("8 bytes"));
    let ckey = u128::from(lo) | (u128::from(hi) << 64);
    ensure!(
        ver == DISK_FORMAT_L3 && key == manifest_key && ckey == expected_key,
        "chunk file header mismatch"
    );
    ensure!(
        (flags & 1 != 0) && draft_bytes > 0,
        "chunk file lacks draft part"
    );
    if draft_only {
        r.seek_relative(i64::try_from(target_bytes)?)?;
        // SAFETY: the draft part of a shared chunk that no entry reads before `drafted` flips.
        r.read_exact(unsafe { HostCache::host_bytes_mut(dst_addr + target_bytes, draft_bytes) })?;
        Ok(draft_bytes as u64)
    } else {
        let len = target_bytes + draft_bytes;
        // SAFETY: freshly allocated pinned chunk.
        r.read_exact(unsafe { HostCache::host_bytes_mut(dst_addr, len) })?;
        Ok(len as u64)
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "local file validation + pool scatter read"
)]
fn read_local_into(
    path: &Path,
    manifest_key: u64,
    expected_id: u64,
    expected_tokens: usize,
    local_bytes: usize,
    rings_bytes: usize,
    pool: &super::ChunkPool,
    local_chunks: &[u32],
) -> anyhow::Result<u64> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let size = file.metadata()?.len();
    let mut r = BufReader::with_capacity(1 << 20, file);
    let mut fixed = [0u8; FILE_HEADER_BYTES];
    r.read_exact(&mut fixed)?;
    ensure!(&fixed[..8] == LOCAL_MAGIC, "not an L3 local file");
    let ver = u32::from_le_bytes(fixed[8..12].try_into().expect("4 bytes"));
    let flags = u32::from_le_bytes(fixed[12..16].try_into().expect("4 bytes"));
    let key = u64::from_le_bytes(fixed[16..24].try_into().expect("8 bytes"));
    let id = u64::from_le_bytes(fixed[24..32].try_into().expect("8 bytes"));
    let tokens_len = usize::try_from(u64::from_le_bytes(
        fixed[32..40].try_into().expect("8 bytes"),
    ))?;
    ensure!(
        ver == DISK_FORMAT_L3
            && key == manifest_key
            && id == expected_id
            && tokens_len == expected_tokens
            && (flags & 1 != 0) == (rings_bytes > 0),
        "local file header mismatch"
    );
    let skip_tokens = tokens_len * 4 + pad8(tokens_len * 4);
    let total_payload = local_bytes + rings_bytes;
    ensure!(
        size == (FILE_HEADER_BYTES + skip_tokens + total_payload) as u64,
        "local file size mismatch"
    );
    r.seek_relative(i64::try_from(skip_tokens)?)?;
    let mut left = total_payload;
    for &c in local_chunks {
        let len = left.min(pool.chunk_bytes);
        // SAFETY: freshly allocated local chunks of this entry.
        r.read_exact(unsafe { HostCache::host_bytes_mut(pool.address(c), len) })?;
        left -= len;
    }
    Ok(total_payload as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_keys_are_prefix_invariant_and_distinct() {
        let a: Vec<u32> = (0..6144u32).map(|i| i.wrapping_mul(37) + 1).collect();
        let mut b = a.clone();
        b[4096] ^= 1; // diverges only in chunk 2
        let ka = chunk_keys_128(&a, 2048);
        let kb = chunk_keys_128(&b, 2048);
        assert_eq!(ka.len(), 3);
        assert_eq!(ka[0], kb[0]);
        assert_eq!(ka[1], kb[1]);
        assert_ne!(ka[2], kb[2]);
        assert_eq!(chunk_keys_128(&a[..4096], 2048), ka[..2]);
    }

    #[test]
    fn write_budget_allows_burst_and_caps_sustained() {
        let mut b = WriteBudget::new(10 << 20, 20 << 20);
        assert!(b.try_consume(15 << 20));
        assert!(b.try_consume(5 << 20));
        assert!(!b.try_consume(1 << 20));
        // Unlimited when rate == 0.
        let mut u = WriteBudget::new(0, 0);
        assert!(u.try_consume(100 << 30));
    }
}
