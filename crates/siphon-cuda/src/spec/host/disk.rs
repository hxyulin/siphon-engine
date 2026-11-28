//! The host tier's entries on disk: one file per entry and a manifest, so that a restarted
//! server starts with the entries of the one before.
//!
//! The manifest (`manifest.json`) holds the format version, an identity from the caller
//! (the checkpoint, the formats, the draft, the kernels: whatever changes the saved bytes
//! or their meaning) and the chunk layout, and lists the entry files with their sizes. A
//! directory whose manifest differs in any of these from the running server's is discarded
//! whole. Every file is written to a temporary name and renamed, and the manifest goes
//! last, after the old one is removed, so that a save that stops midway leaves no manifest
//! and nothing that loads.
//!
//! An entry file holds, little-endian:
//!
//! - the magic `SIPHONHC`, the format version (u32) and 4 zero bytes;
//! - the manifest's key (u64), a hash of the format, identity and layout, so that a file
//!   loads only under the manifest it was written with;
//! - the entry's use order (u64), its tokens (u64), its global cache chunks (u64), whether
//!   it holds the draft's rings (u64), and per chunk whether it holds the draft's global
//!   caches (one byte each, then zeros to a multiple of 8);
//! - the tokens (u32 each, then zeros to a multiple of 8): the prompt's key, whose image
//!   placeholders are ids of the images' hashes, above every token id;
//! - each chunk's target part, followed by its draft part if it holds one;
//! - the rings and conv states, then the draft's rings if it holds them.
//!
//! A file says everything needed to read it back into any pool of the same layout, so a
//! disk tier that keeps entries past their time in memory could use the same files.

use std::{
    fs::{self, File},
    io::{BufReader, BufWriter, Read, Seek, Write},
    path::{Path, PathBuf},
    time::Instant,
};

use anyhow::{Context as _, bail, ensure};
use serde_json::{Value, json};
use siphon::prefix_hash::chunk_hashes;

use super::{HOST_CHUNK_POSITIONS, HostCache, HostEntry};

/// Changes whenever the files' layout changes, or the engine computes the saved state
/// differently in a way the identity does not show.
pub const DISK_FORMAT: u32 = 1;

const MAGIC: &[u8; 8] = b"SIPHONHC";
pub(super) const MANIFEST: &str = "manifest.json";

/// Files written by [`super::super::Speculator::save_host_cache`].
#[derive(Debug)]
pub struct Saved {
    /// Host cache entries written.
    pub entries: usize,
    /// Total bytes written to entry files, including their headers.
    pub bytes: u64,
    /// Elapsed time for the save, in seconds.
    pub seconds: f64,
}

/// Result of [`super::super::Speculator::load_host_cache`].
#[derive(Debug)]
pub enum Loaded {
    /// No manifest: nothing was saved, or a save stopped midway.
    Missing,
    /// The manifest differed from the server's; the directory's files were removed.
    Discarded(String),
    /// The manifest matched and loading was attempted.
    Loaded {
        /// Host cache entries restored.
        entries: usize,
        /// Entries that did not fit the pool or could not be read.
        skipped: usize,
        /// Total bytes read from the restored entry files.
        bytes: u64,
        /// Elapsed time for the load, in seconds.
        seconds: f64,
    },
}

/// FNV-1a.
pub(super) fn fnv(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |h, &b| {
        (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
    })
}

pub(super) fn pad8(n: usize) -> usize {
    n.next_multiple_of(8) - n
}

impl HostCache {
    /// The layout part of the manifest: what the chunks hold and where.
    pub(super) fn layout(&self) -> Value {
        json!({
            "chunk_positions": HOST_CHUNK_POSITIONS,
            "chunk_bytes": self.pool.chunk_bytes,
            "target_caches": self.target.caches.len(),
            "target_position_bytes": self.target.position_bytes,
            "draft_caches": self.draft.caches.len(),
            "draft_position_bytes": self.draft.position_bytes,
            "local_bytes": self.local_bytes,
            "rings_bytes": self.rings_bytes,
        })
    }

    /// The manifest without its entries, and its key.
    fn header(&self, identity: &Value) -> (Value, u64) {
        let header = json!({
            "format": DISK_FORMAT,
            "identity": identity,
            "layout": self.layout(),
        });
        let key = fnv(header.to_string().as_bytes());
        (header, key)
    }

    /// Bytes of a chunk's target part and draft part.
    pub(super) fn parts(&self) -> (usize, usize) {
        (
            HOST_CHUNK_POSITIONS * self.target.bytes_per_position(),
            HOST_CHUNK_POSITIONS * self.draft.bytes_per_position(),
        )
    }

    /// The bytes at host address `at`.
    ///
    /// # Safety
    ///
    /// `at .. at + len` must lie in one chunk, and no copy may write it meanwhile.
    pub(super) unsafe fn host_bytes<'a>(at: usize, len: usize) -> &'a [u8] {
        // SAFETY: the caller's.
        unsafe { std::slice::from_raw_parts(at as *const u8, len) }
    }

    /// The bytes at host address `at`, to fill.
    ///
    /// # Safety
    ///
    /// `at .. at + len` must lie in one chunk that no entry reads and no copy touches
    /// meanwhile.
    pub(super) unsafe fn host_bytes_mut<'a>(at: usize, len: usize) -> &'a mut [u8] {
        // SAFETY: the caller's.
        unsafe { std::slice::from_raw_parts_mut(at as *mut u8, len) }
    }

    /// Writes every entry to `dir`, one file each, and a manifest with `identity` last.
    /// Waits for the copies to the host first. When an active L3 disk tier is enabled,
    /// flushes and persists into the deduplicated L3 store instead.
    pub fn save_dir(&mut self, dir: &Path, identity: &Value) -> anyhow::Result<Saved> {
        if self.disk.is_some() {
            return self.save_l3_dir(dir, identity);
        }
        let started = Instant::now();
        self.copy.synchronize()?;
        self.main.synchronize()?;
        fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
        let manifest = dir.join(MANIFEST);
        if manifest.exists() {
            fs::remove_file(&manifest)?;
        }
        let (mut header, key) = self.header(identity);
        let mut order: Vec<&HostEntry> = self.entries.iter().collect();
        order.sort_by_key(|e| e.used);
        let mut files = Vec::new();
        let mut total = 0;
        for (n, e) in order.iter().enumerate() {
            let name = format!("entry-{n:05}.bin");
            let bytes = self.write_entry(&dir.join(&name), e, key)?;
            total += bytes;
            files.push(json!({"file": name, "bytes": bytes, "tokens": e.tokens.len()}));
        }
        header["entries"] = Value::Array(files);
        write_atomic(&manifest, header.to_string().as_bytes())?;
        // Files of an earlier save that this one did not rewrite.
        let kept: Vec<String> = (0..order.len())
            .map(|n| format!("entry-{n:05}.bin"))
            .collect();
        for path in entry_files(dir)? {
            let name = path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or_default();
            if !kept.iter().any(|k| k == name) {
                fs::remove_file(&path)?;
            }
        }
        Ok(Saved {
            entries: order.len(),
            bytes: total,
            seconds: started.elapsed().as_secs_f64(),
        })
    }

    fn write_entry(&self, path: &Path, e: &HostEntry, key: u64) -> anyhow::Result<u64> {
        let tmp = path.with_extension("tmp");
        let file = File::create(&tmp).with_context(|| format!("creating {}", tmp.display()))?;
        let mut w = BufWriter::with_capacity(1 << 20, file);
        let mut head = Vec::new();
        head.extend_from_slice(MAGIC);
        head.extend_from_slice(&DISK_FORMAT.to_le_bytes());
        head.extend_from_slice(&[0; 4]);
        for v in [
            key,
            e.used,
            e.tokens.len() as u64,
            e.kv.len() as u64,
            u64::from(e.rings),
        ] {
            head.extend_from_slice(&v.to_le_bytes());
        }
        head.extend(e.kv.iter().map(|&c| u8::from(self.drafted[c as usize])));
        head.resize(head.len() + pad8(e.kv.len()), 0);
        for t in &e.tokens {
            head.extend_from_slice(&t.to_le_bytes());
        }
        head.resize(head.len() + pad8(e.tokens.len() * 4), 0);
        w.write_all(&head)?;
        let (target, draft) = self.parts();
        let mut bytes = head.len() as u64;
        for &c in &e.kv {
            let len = if self.drafted[c as usize] {
                target + draft
            } else {
                target
            };
            // SAFETY: a chunk holds `target + draft` bytes, and the copies are done.
            w.write_all(unsafe { Self::host_bytes(self.pool.address(c), len) })?;
            bytes += len as u64;
        }
        let mut left = self.local_bytes + if e.rings { self.rings_bytes } else { 0 };
        for &c in &e.local {
            let len = left.min(self.pool.chunk_bytes);
            // SAFETY: as above.
            w.write_all(unsafe { Self::host_bytes(self.pool.address(c), len) })?;
            bytes += len as u64;
            left -= len;
        }
        w.into_inner()?.sync_all()?;
        fs::rename(&tmp, path)?;
        Ok(bytes)
    }

    /// Loads the entries that `dir`'s manifest lists, oldest first, if its format,
    /// `identity` and layout match this cache's. Removes the directory's files otherwise.
    /// Entries that do not fit the pool push out older ones, as saves do.
    pub fn load_dir(&mut self, dir: &Path, identity: &Value) -> anyhow::Result<Loaded> {
        let started = Instant::now();
        let manifest = dir.join(MANIFEST);
        let Ok(text) = fs::read_to_string(&manifest) else {
            return Ok(Loaded::Missing);
        };
        let (header, key) = self.header(identity);
        let saved: Value = serde_json::from_str(&text).unwrap_or(Value::Null);
        let differs = ["format", "identity", "layout"]
            .into_iter()
            .find(|k| saved.get(k) != header.get(k));
        if let Some(what) = differs {
            discard(dir)?;
            return Ok(Loaded::Discarded(format!(
                "the saved {what} differs from this server's"
            )));
        }
        let files = saved["entries"].as_array().cloned().unwrap_or_default();
        let (mut entries, mut skipped, mut bytes) = (0, 0, 0);
        for f in &files {
            let name = f["file"].as_str().unwrap_or_default();
            match self.read_entry(&dir.join(name), key, f["bytes"].as_u64()) {
                Ok(Some(n)) => {
                    entries += 1;
                    bytes += n;
                }
                Ok(None) => skipped += 1,
                Err(e) => {
                    skipped += 1;
                    self.note(format!("host cache: skipping {name}: {e:#}"));
                }
            }
        }
        Ok(Loaded::Loaded {
            entries,
            skipped,
            bytes,
            seconds: started.elapsed().as_secs_f64(),
        })
    }

    /// Reads the header, draft flags, and token ids of an entry file.
    fn read_entry_header(
        r: &mut BufReader<File>,
        key: u64,
    ) -> anyhow::Result<(u64, bool, Vec<u8>, Vec<u32>)> {
        let mut fixed = [0u8; 56];
        r.read_exact(&mut fixed)?;
        let word = |i: usize| {
            u64::from_le_bytes(fixed[16 + i * 8..24 + i * 8].try_into().expect("8 bytes"))
        };
        ensure!(&fixed[..8] == MAGIC, "not an entry file");
        ensure!(
            fixed[8..12] == DISK_FORMAT.to_le_bytes() && word(0) == key,
            "written under another manifest"
        );
        let (used, tokens, chunks, rings) = (
            word(1),
            usize::try_from(word(2))?,
            usize::try_from(word(3))?,
            word(4) != 0,
        );
        ensure!(
            tokens == chunks * HOST_CHUNK_POSITIONS && chunks > 0,
            "{tokens} tokens in {chunks} chunks"
        );
        let mut drafted = vec![0u8; chunks + pad8(chunks)];
        r.read_exact(&mut drafted)?;
        drafted.truncate(chunks);
        let mut raw = vec![0u8; tokens * 4 + pad8(tokens * 4)];
        r.read_exact(&mut raw)?;
        let ids: Vec<u32> = raw[..tokens * 4]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| u32::from_le_bytes(*b))
            .collect();
        Ok((used, rings, drafted, ids))
    }

    /// Reads one entry file into the pool. Returns the bytes read, or `None` when the
    /// entry does not fit.
    fn read_entry(
        &mut self,
        path: &Path,
        key: u64,
        expected: Option<u64>,
    ) -> anyhow::Result<Option<u64>> {
        let file = File::open(path)?;
        let size = file.metadata()?.len();
        ensure!(
            expected == Some(size),
            "{size} bytes, the manifest says {expected:?}"
        );
        let mut r = BufReader::with_capacity(1 << 20, file);
        let (used, rings, drafted, ids) = Self::read_entry_header(&mut r, key)?;
        let entry_hashes = chunk_hashes(&ids, HOST_CHUNK_POSITIONS);
        let full_hash = entry_hashes.last().copied();
        if self.entries.iter().any(|e| {
            e.tokens.len() == ids.len() && e.hashes.last().copied() == full_hash && e.tokens == ids
        }) {
            return Ok(Some(0));
        }
        let Some((kv, local, shared)) = self.allocate(&ids)? else {
            return Ok(None);
        };
        let (target, draft) = self.parts();
        let mut read = 0u64;
        let result = (|| -> anyhow::Result<()> {
            for (j, &c) in kv.iter().enumerate() {
                let has_draft = drafted[j] != 0;
                let at = self.pool.address(c);
                if j < shared {
                    // Another entry's chunk: only a draft part it lacks is new.
                    r.seek_relative(i64::try_from(target)?)?;
                    if has_draft && !self.drafted[c as usize] && draft > 0 {
                        // SAFETY: the draft part of a chunk no copy touches, which no
                        // entry reads before `drafted` says it is there.
                        r.read_exact(unsafe { Self::host_bytes_mut(at + target, draft) })?;
                        self.drafted[c as usize] = true;
                        read += draft as u64;
                    } else if has_draft {
                        r.seek_relative(i64::try_from(draft)?)?;
                    }
                    continue;
                }
                // SAFETY: a new chunk of this entry, which nothing else reads yet.
                r.read_exact(unsafe { Self::host_bytes_mut(at, target) })?;
                read += target as u64;
                if has_draft {
                    // SAFETY: as above.
                    r.read_exact(unsafe { Self::host_bytes_mut(at + target, draft) })?;
                    read += draft as u64;
                }
                self.drafted[c as usize] = has_draft;
            }
            let mut left = self.local_bytes + if rings { self.rings_bytes } else { 0 };
            for &c in &local {
                let len = left.min(self.pool.chunk_bytes);
                // SAFETY: a new chunk of this entry.
                r.read_exact(unsafe { Self::host_bytes_mut(self.pool.address(c), len) })?;
                read += len as u64;
                left -= len;
            }
            ensure!(
                r.stream_position()? == size,
                "the file has bytes past the entry"
            );
            Ok(())
        })();
        if let Err(e) = result {
            for &c in kv.iter().chain(&local) {
                self.pool.release(c);
            }
            return Err(e);
        }
        self.tick = self.tick.max(used);
        self.entries.push(HostEntry {
            tokens: ids,
            hashes: entry_hashes,
            kv,
            local,
            rings,
            used,
            used_at: std::time::Instant::now(),
            disk_id: None,
            disk_dirty: true,
        });
        Ok(Some(read))
    }
}

/// Writes `bytes` to `path` through a temporary file and a rename.
pub(super) fn write_atomic(path: &Path, bytes: &[u8]) -> anyhow::Result<()> {
    let tmp = path.with_extension("tmp");
    let mut file = File::create(&tmp)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    fs::rename(&tmp, path)?;
    Ok(())
}

/// The entry files and temporary files in `dir`.
fn entry_files(dir: &Path) -> anyhow::Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    for item in fs::read_dir(dir)? {
        let path = item?.path();
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or_default();
        if name.starts_with("entry-")
            && Path::new(name)
                .extension()
                .is_some_and(|x| x == "bin" || x == "tmp")
        {
            files.push(path);
        }
    }
    Ok(files)
}

/// Removes the manifest and every entry/L3 file of `dir`, and leaves other files alone.
pub(super) fn discard(dir: &Path) -> anyhow::Result<()> {
    super::l3::discard_l3_files(dir)?;
    for path in entry_files(dir)?
        .into_iter()
        .chain([dir.join(MANIFEST), dir.join("manifest.tmp")])
    {
        match fs::remove_file(&path) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => bail!("removing {}: {e}", path.display()),
        }
    }
    Ok(())
}
