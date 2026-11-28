//! Cumulative chunk and exact-length hashing for fast prefix-cache lookup and chunk sharing.
//!
//! Prefix cache entries store prompt token keys at exact prompt boundaries (`prompt_len - 1`
//! once above [`MIN_PREFIX_TOKENS`]). Computing a single pass of cumulative 64-bit hashes and
//! streaming hasher states at each full `chunk` boundary lets device, host, and disk caches
//! reject mismatched entries in O(1) on full chunks and hash only the short trailing tail
//! (`< chunk` tokens) for arbitrary-length entries, while still verifying exact slice
//! equality on hash hits.

const MUL: u64 = 0x9e37_79b1_85eb_ca87;

/// Minimum prefix length (in tokens) worth caching. Below this threshold, copying the fixed
/// local sliding-ring and conv state (~80–105 MiB) costs more than prefilling from scratch.
pub const MIN_PREFIX_TOKENS: usize = 256;

/// Returns the exact prefix-cache save boundary for a prompt of `prompt_len` tokens:
/// `prompt_len - 1` when at least [`MIN_PREFIX_TOKENS`] tokens would be saved (leaving 1
/// token to compute the first token's logits, and allowing exact duplicate prompts to hit
/// `prompt_len - 1` tokens), or `0` for tiny prompts below the threshold.
#[must_use]
pub const fn prefix_boundary(prompt_len: usize) -> usize {
    let boundary = prompt_len.saturating_sub(1);
    if boundary >= MIN_PREFIX_TOKENS {
        boundary
    } else {
        0
    }
}

/// Four-lane streaming 64-bit hasher over `u32` tokens: packs pairs into `u64` across
/// four independent lanes for high instruction-level parallelism while guaranteeing
/// split-invariant hashes (`prefix_hash(&t[..k * chunk]) == chunk_hashes(t, chunk)[k - 1]`).
#[derive(Clone, Copy, Debug)]
struct PrefixHasher {
    lanes: [u64; 4],
    pos: usize,
}

impl PrefixHasher {
    #[inline]
    const fn new() -> Self {
        Self {
            lanes: [
                0xcbf2_9ce4_8422_2325,
                0x9e37_79b9_7f4a_7c15,
                0xbf58_476d_1ce4_e5b9,
                0x94d0_49bb_1331_11eb,
            ],
            pos: 0,
        }
    }

    #[inline]
    fn push_one(&mut self, token: u32) {
        let lane = (self.pos >> 1) & 3;
        if self.pos.is_multiple_of(2) {
            self.lanes[lane] ^= u64::from(token);
        } else {
            self.lanes[lane] = (self.lanes[lane] ^ (u64::from(token) << 32)).wrapping_mul(MUL);
        }
        self.pos += 1;
    }

    #[inline]
    fn update(&mut self, mut tokens: &[u32]) {
        while !self.pos.is_multiple_of(8) && !tokens.is_empty() {
            self.push_one(tokens[0]);
            tokens = &tokens[1..];
        }
        let (chunks, rem) = tokens.as_chunks::<8>();
        for c in chunks {
            let w0 = u64::from(c[0]) | (u64::from(c[1]) << 32);
            let w1 = u64::from(c[2]) | (u64::from(c[3]) << 32);
            let w2 = u64::from(c[4]) | (u64::from(c[5]) << 32);
            let w3 = u64::from(c[6]) | (u64::from(c[7]) << 32);
            self.lanes[0] = (self.lanes[0] ^ w0).wrapping_mul(MUL);
            self.lanes[1] = (self.lanes[1] ^ w1).wrapping_mul(MUL);
            self.lanes[2] = (self.lanes[2] ^ w2).wrapping_mul(MUL);
            self.lanes[3] = (self.lanes[3] ^ w3).wrapping_mul(MUL);
        }
        self.pos += chunks.len() * 8;
        for &t in rem {
            self.push_one(t);
        }
    }

    #[inline]
    const fn finish(&self) -> u64 {
        let mut h = self.lanes[0]
            ^ self.lanes[1].rotate_left(13)
            ^ self.lanes[2].rotate_left(29)
            ^ self.lanes[3].rotate_left(47)
            ^ (self.pos as u64);
        h = (h ^ (h >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        h = (h ^ (h >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        h ^ (h >> 31)
    }
}

/// Cumulative 64-bit hash of every token in `tokens`.
///
/// Equals the last entry of [`chunk_hashes`] whenever `tokens.len()` is a positive
/// multiple of `chunk`.
#[must_use]
pub fn prefix_hash(tokens: &[u32]) -> u64 {
    let mut hasher = PrefixHasher::new();
    hasher.update(tokens);
    hasher.finish()
}

/// Cumulative 64-bit hash after each full `chunk` of `tokens`: entry `k` is the hash of
/// `&tokens[..(k + 1) * chunk]`.
#[must_use]
pub fn chunk_hashes(tokens: &[u32], chunk: usize) -> Vec<u64> {
    if chunk == 0 {
        return Vec::new();
    }
    let count = tokens.len() / chunk;
    let mut out = Vec::with_capacity(count);
    let mut state = PrefixHasher::new();
    for block in tokens.chunks_exact(chunk) {
        state.update(block);
        out.push(state.finish());
    }
    out
}

/// Precomputed cumulative chunk hashes and streaming states for an incoming prompt,
/// bounded to `max_chunks`.
#[derive(Clone, Debug)]
pub struct PrefixHashes {
    chunk: usize,
    hashes: Vec<u64>,
    states: Vec<PrefixHasher>,
}

impl PrefixHashes {
    /// Computes cumulative chunk hashes and streaming states for the first `max_chunks` full
    /// chunks of `tokens`.
    #[must_use]
    pub fn new(tokens: &[u32], chunk: usize, max_chunks: usize) -> Self {
        if chunk == 0 || max_chunks == 0 {
            return Self {
                chunk,
                hashes: Vec::new(),
                states: Vec::new(),
            };
        }
        let count = (tokens.len() / chunk).min(max_chunks);
        let mut out = Vec::with_capacity(count);
        let mut states = Vec::with_capacity(count);
        let mut state = PrefixHasher::new();
        for block in tokens.chunks_exact(chunk).take(count) {
            state.update(block);
            out.push(state.finish());
            states.push(state);
        }
        Self {
            chunk,
            hashes: out,
            states,
        }
    }

    /// Cumulative hash of the first `chunks` full chunks (`chunks >= 1`).
    #[inline]
    #[must_use]
    pub fn hash_at_chunks(&self, chunks: usize) -> Option<u64> {
        chunks
            .checked_sub(1)
            .and_then(|i| self.hashes.get(i).copied())
    }

    /// Cumulative hash of `&tokens[..len]`, when `len` is a positive multiple of `chunk`
    /// covered by this table.
    #[inline]
    #[must_use]
    pub fn hash_at_len(&self, len: usize) -> Option<u64> {
        if self.chunk == 0 || len == 0 || !len.is_multiple_of(self.chunk) {
            return None;
        }
        self.hash_at_chunks(len / self.chunk)
    }

    /// Cumulative hash of `&tokens[..len]` for any positive `len <= tokens.len()`.
    ///
    /// Chunk-aligned lengths within the precomputed table return in O(1); arbitrary lengths
    /// resume from the nearest cached chunk boundary state and hash only the remaining tail
    /// (`< chunk` tokens when covered by `max_chunks`).
    #[inline]
    #[must_use]
    pub fn hash_at_len_in(&self, tokens: &[u32], len: usize) -> Option<u64> {
        if len == 0 || len > tokens.len() {
            return None;
        }
        if self.chunk > 0 {
            let full = len / self.chunk;
            let rem = len % self.chunk;
            if rem == 0
                && let Some(h) = self.hash_at_chunks(full)
            {
                return Some(h);
            }
            if full > 0 && full <= self.states.len() {
                let mut state = self.states[full - 1];
                state.update(&tokens[full * self.chunk..len]);
                return Some(state.finish());
            }
            if full == 0 {
                return Some(prefix_hash(&tokens[..len]));
            }
        }
        Some(prefix_hash(&tokens[..len]))
    }

    /// Number of leading chunks whose cumulative hashes match `entry_hashes`.
    #[inline]
    #[must_use]
    pub fn shared_chunks(&self, entry_hashes: &[u64]) -> usize {
        self.hashes
            .iter()
            .zip(entry_hashes)
            .take_while(|(a, b)| a == b)
            .count()
    }
}

/// Longest entry in `entries` strictly shorter than `prompt` that `prompt` starts with,
/// using `hashes` for fast mismatch rejection and exact slice verification on hash hits.
#[must_use]
pub fn longest_prefix_match<T>(
    prompt: &[u32],
    hashes: &PrefixHashes,
    entries: &[T],
    mut tokens_of: impl FnMut(&T) -> &[u32],
    mut hash_of: impl FnMut(&T) -> u64,
) -> Option<usize> {
    let mut best: Option<(usize, usize)> = None;
    for (i, entry) in entries.iter().enumerate() {
        let tokens = tokens_of(entry);
        let len = tokens.len();
        if len == 0 || best.is_some_and(|(_, best_len)| len <= best_len) || prompt.len() <= len {
            continue;
        }
        if prompt[len - 1] != tokens[len - 1] || prompt[0] != tokens[0] {
            continue;
        }
        if hashes.hash_at_len_in(prompt, len) == Some(hash_of(entry)) && prompt.starts_with(tokens)
        {
            best = Some((i, len));
        }
    }
    best.map(|(i, _)| i)
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use super::*;

    #[test]
    fn chunk_hashes_match_prefix_hash_at_every_boundary() {
        let tokens: Vec<u32> = (0..10_000)
            .map(|i: u32| i.wrapping_mul(2_654_435_761) ^ 0x5a5a)
            .collect();
        let chunk = 2048;
        let ch = chunk_hashes(&tokens, chunk);
        assert_eq!(ch.len(), 4);
        for (k, &h) in ch.iter().enumerate() {
            assert_eq!(h, prefix_hash(&tokens[..(k + 1) * chunk]));
        }
        let ph = PrefixHashes::new(&tokens, chunk, 3);
        assert_eq!(ph.hash_at_len(2048), Some(ch[0]));
        assert_eq!(ph.hash_at_len(4096), Some(ch[1]));
        assert_eq!(ph.hash_at_len(6144), Some(ch[2]));
        assert_eq!(ph.hash_at_len(8192), None);
        assert_eq!(ph.hash_at_len(1000), None);
        assert_eq!(ph.hash_at_len(0), None);
        // Arbitrary lengths (below chunk, mid-chunk, exact chunk, beyond max_chunks).
        for len in [
            1, 7, 8, 9, 255, 256, 1000, 2047, 2048, 2049, 3499, 6144, 7000, 9999,
        ] {
            assert_eq!(
                ph.hash_at_len_in(&tokens, len),
                Some(prefix_hash(&tokens[..len])),
                "mismatch at len={len}"
            );
        }
        assert_eq!(ph.hash_at_len_in(&tokens, 0), None);
        assert_eq!(ph.hash_at_len_in(&tokens, tokens.len() + 1), None);
    }

    #[test]
    fn prefix_boundary_respects_min_threshold() {
        assert_eq!(prefix_boundary(0), 0);
        assert_eq!(prefix_boundary(1), 0);
        assert_eq!(prefix_boundary(MIN_PREFIX_TOKENS), 0);
        assert_eq!(prefix_boundary(MIN_PREFIX_TOKENS + 1), MIN_PREFIX_TOKENS);
        assert_eq!(prefix_boundary(1500), 1499);
        assert_eq!(prefix_boundary(2048), 2047);
        assert_eq!(prefix_boundary(4097), 4096);
    }

    #[test]
    fn shared_chunks_and_longest_match_agree_with_linear_scan() {
        let chunk = 64;
        let base: Vec<u32> = (0..512).collect();
        let mut entries: Vec<(Vec<u32>, Vec<u64>, u64)> = Vec::new();
        for n_chunks in 1..=7 {
            let mut e = base[..n_chunks * chunk].to_vec();
            if n_chunks % 2 == 0 {
                // Diverge in the last chunk.
                *e.last_mut().unwrap() ^= 0x03e7;
            }
            let ch = chunk_hashes(&e, chunk);
            let h = *ch.last().unwrap();
            entries.push((e, ch, h));
        }
        let mut prompt = base.clone();
        prompt.push(42);
        let hashes = PrefixHashes::new(&prompt, chunk, 8);

        let got = longest_prefix_match(&prompt, &hashes, &entries, |e| &e.0, |e| e.2);
        let expected = entries
            .iter()
            .enumerate()
            .filter(|(_, e)| prompt.len() > e.0.len() && prompt.starts_with(&e.0))
            .max_by_key(|(_, e)| e.0.len())
            .map(|(i, _)| i);
        assert_eq!(got, expected);
        // Longest exact match is n_chunks = 7 (index 6).
        assert_eq!(got, Some(6));

        for (e, ch, _) in &entries {
            let exact = e.iter().zip(&prompt).take_while(|(a, b)| a == b).count() / chunk;
            assert_eq!(hashes.shared_chunks(ch), exact);
        }
    }

    #[test]
    fn benchmark_hashed_vs_linear_prefix_lookup() {
        // 64 entries sharing a 32K-token system prompt that diverge near the end of chunk 16.
        let chunk = 2048;
        let len = 16 * chunk; // 32,768 tokens
        let system: Vec<u32> = (0..len as u32).map(|x| x.wrapping_mul(31) + 7).collect();
        let entries: Vec<(Vec<u32>, u64)> = (0..64u32)
            .map(|id| {
                let mut e = system.clone();
                e[len - 1] = id;
                let h = prefix_hash(&e);
                (e, h)
            })
            .collect();
        let mut query = system;
        query[len - 1] = 63; // matches the last entry
        query.push(12345);

        let iters: u32 = 200;
        let t0 = Instant::now();
        let mut lin_acc = 0usize;
        for _ in 0..iters {
            let hit = entries
                .iter()
                .enumerate()
                .filter(|(_, (e, _))| query.len() > e.len() && query.starts_with(e))
                .max_by_key(|(_, (e, _))| e.len())
                .map(|(i, _)| i);
            lin_acc += hit.unwrap();
        }
        let lin_us = t0.elapsed().as_secs_f64() * 1e6 / f64::from(iters);

        let t1 = Instant::now();
        let mut hash_acc = 0usize;
        for _ in 0..iters {
            let hashes = PrefixHashes::new(&query, chunk, len / chunk);
            let hit = longest_prefix_match(&query, &hashes, &entries, |e| &e.0, |e| e.1);
            hash_acc += hit.unwrap();
        }
        let hash_us = t1.elapsed().as_secs_f64() * 1e6 / f64::from(iters);

        assert_eq!(lin_acc, hash_acc);
        assert_eq!(lin_acc, 63 * iters as usize);
        eprintln!(
            "prefix_lookup (64 x 32K shared-prefix): linear={lin_us:.1} us, hashed={hash_us:.1} us ({:.1}x)",
            lin_us / hash_us.max(1e-9)
        );
    }
}
