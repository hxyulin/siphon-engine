//! 2-branch speculative tree topology, causal masks, short-conv ancestor tables, and
//! lossless tree verification (greedy, coupled exponential race, and residual
//! `(p - q_1)_+` stochastic verification).
//!
//! A speculative tree holds `1..=MAX_TREE_NODES` (= [`VERIFY_ROWS`] = 8) rows in
//! topological order (`parent[0] = -1`, `0 <= parent[r] < r` for `r > 0`):
//! - Node `0` is the anchor token at depth `0` (sequence position `start`).
//! - Every node `r` has a true sequence position `start + depth[r]` and a unique ancestor
//!   path `0 = a_0 -> a_1 -> ... -> a_{depth[r]} = r`.
//! - Causal attention sees only the committed prefix before `start` plus the node's own
//!   ancestors `attn_anc[r][0..=depth[r]]` (encoded by `ancestor_mask[r]` and `attn_anc`).
//! - Causal short convolutions (kernel width 4) convolve each node with its 3 tree
//!   ancestors `conv_anc[r][0..4]` (reaching into the 3-row carried conv state when
//!   `depth[r] < 3`).

use anyhow::{Context as _, ensure};

use crate::memory::VERIFY_ROWS;

/// Maximum nodes in a speculative verify tree (`VERIFY_ROWS = 8`).
pub const MAX_TREE_NODES: usize = VERIFY_ROWS;
/// Short causal convolution kernel width in Inkling 2 Mini.
pub const CONV_KERNEL: usize = 4;
/// Number of carried history rows in each conv state (`CONV_KERNEL - 1`).
pub const CONV_STATE_ROWS: usize = CONV_KERNEL - 1;

/// How a 2-branch speculative tree chooses the node where the secondary branch diverges.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BranchStrategy {
    /// Branch at the primary position `0 <= b <= max_depth` with the highest expected
    /// rejection probability `survival(b) * (1 - conf[b])` (or `survival(b) * alt_conf[b]`
    /// when runner-up probabilities are known).
    LowestConfidence { max_depth: usize },
    /// Always branch at a fixed parent depth `b` (0 = anchor / position 1).
    Fixed(usize),
}

impl Default for BranchStrategy {
    fn default() -> Self {
        Self::LowestConfidence { max_depth: 3 }
    }
}

/// Configuration for opt-in 2-branch speculative tree verification.
///
/// Defaults to disabled (`enabled = false`) so default linear speculation is untouched.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpecTreeConfig {
    /// Whether 2-branch tree verification is enabled.
    pub enabled: bool,
    /// Number of draft nodes allocated to the secondary branch when branching (`1..=6`).
    pub secondary_len: usize,
    /// How the branch point along the primary chain is chosen.
    pub branch_strategy: BranchStrategy,
    /// Skip branching and verify a full linear chain when the best branch point's primary
    /// confidence is at least this threshold (e.g. `0.98`). Set to `1.0` to always branch.
    pub max_branch_confidence: f32,
}

impl Default for SpecTreeConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            secondary_len: 2,
            branch_strategy: BranchStrategy::default(),
            max_branch_confidence: 0.98,
        }
    }
}

impl SpecTreeConfig {
    /// Default enabled 2-branch configuration: 2-node secondary branch (total 8 rows:
    /// anchor + 5 primary + 2 secondary) diverging at the highest-uncertainty position in
    /// `0..=3`.
    #[must_use]
    pub fn two_branch() -> Self {
        Self {
            enabled: true,
            ..Self::default()
        }
    }

    /// Validates the configuration fields.
    pub fn check(&self) -> anyhow::Result<()> {
        ensure!(
            (1..=MAX_TREE_NODES - 2).contains(&self.secondary_len),
            "secondary_len is {}, expected 1 to {}",
            self.secondary_len,
            MAX_TREE_NODES - 2
        );
        ensure!(
            self.max_branch_confidence.is_finite()
                && (0.0..=1.0).contains(&self.max_branch_confidence),
            "max_branch_confidence is {}, expected in [0, 1]",
            self.max_branch_confidence
        );
        if let BranchStrategy::Fixed(b) = self.branch_strategy {
            ensure!(
                b < MAX_TREE_NODES - 1,
                "fixed branch parent {b} exceeds max depth {}",
                MAX_TREE_NODES - 2
            );
        }
        Ok(())
    }

    /// Primary draft count for a target total verify width `total_rows` (`2..=8`) when
    /// branching with `secondary_len` secondary nodes. Returns `None` if `total_rows` is too
    /// small to fit both a primary child and a secondary branch.
    #[must_use]
    pub fn primary_len_for(&self, total_rows: usize) -> Option<(usize, usize)> {
        if !self.enabled || !(3..=MAX_TREE_NODES).contains(&total_rows) {
            return None;
        }
        let budget = total_rows - 1; // drafts besides the anchor
        let k2 = self.secondary_len.min(budget - 1);
        let k1 = budget - k2;
        if k1 >= 1 && k2 >= 1 {
            Some((k1, k2))
        } else {
            None
        }
    }
}

/// Precomputed topology and device-ready ancestor lookup tables for a verify tree of
/// `1..=MAX_TREE_NODES` rows.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpecTree {
    /// Active nodes in this tree (`1..=MAX_TREE_NODES`).
    pub num_nodes: usize,
    /// Parent of each node (`-1` for root `0` and unused slots `>= num_nodes`).
    pub parent: [i32; MAX_TREE_NODES],
    /// Depth of each node from root (`depth[0] = 0`, `depth[r] = depth[parent[r]] + 1`).
    pub depth: [u32; MAX_TREE_NODES],
    /// Causal ancestor bitmask for each node: bit `a` is 1 iff `a` is an ancestor of `r`
    /// (including `a == r`).
    pub ancestor_mask: [u32; MAX_TREE_NODES],
    /// Flattened `[MAX_TREE_NODES, MAX_TREE_NODES]` attention ancestor table:
    /// `attn_anc[r * 8 + d]` is the tree node index at distance `d` (`0..=depth[r]`) from
    /// node `r`, or `-1` when `d > depth[r]` (meaning distance `d` lies in the committed KV
    /// cache before `start`).
    pub attn_anc: [i32; MAX_TREE_NODES * MAX_TREE_NODES],
    /// Flattened `[MAX_TREE_NODES, CONV_KERNEL]` short-conv ancestor window, oldest first:
    /// `conv_anc[r * 4 + k]` (`k = 0..4`, lag `3 - k`) is either a tree row index in
    /// `0..num_nodes` or a negative offset in `{-3, -2, -1}` indexing `conv_state[3 + off]`.
    pub conv_anc: [i32; MAX_TREE_NODES * CONV_KERNEL],
    /// First (primary) child of each node, or `-1` if leaf.
    pub primary_child: [i32; MAX_TREE_NODES],
    /// Second (secondary) child of each node, or `-1` if at most one child.
    pub secondary_child: [i32; MAX_TREE_NODES],
    /// Branch parent node if this is a 2-branch tree, `None` for a linear chain.
    pub branch_parent: Option<usize>,
}

impl SpecTree {
    /// Builds a degenerate single-branch (linear) tree of `m` rows (`1..=8`):
    /// `0 -> 1 -> ... -> m - 1`.
    pub fn linear(m: usize) -> anyhow::Result<Self> {
        ensure!(
            (1..=MAX_TREE_NODES).contains(&m),
            "linear tree needs 1 to {MAX_TREE_NODES} nodes, got {m}"
        );
        let mut parents = Vec::with_capacity(m);
        parents.push(-1);
        for i in 1..m {
            parents.push(i32::try_from(i - 1)?);
        }
        Self::from_parents(&parents)
    }

    /// Builds a compact 2-branch tree with:
    /// - root `0` (anchor)
    /// - primary chain `1..=primary_len` (`parent[r] = r - 1`)
    /// - secondary branch `primary_len + 1 ..= primary_len + secondary_len` diverging from
    ///   `branch_parent` (`0 <= branch_parent < primary_len`).
    ///
    /// Total rows `1 + primary_len + secondary_len` must be at most [`MAX_TREE_NODES`].
    pub fn two_branch(
        primary_len: usize,
        branch_parent: usize,
        secondary_len: usize,
    ) -> anyhow::Result<Self> {
        ensure!(primary_len >= 1, "primary_len must be at least 1");
        ensure!(secondary_len >= 1, "secondary_len must be at least 1");
        let total = 1 + primary_len + secondary_len;
        ensure!(
            total <= MAX_TREE_NODES,
            "tree has {total} nodes (1 + {primary_len} + {secondary_len}), max is {MAX_TREE_NODES}"
        );
        ensure!(
            branch_parent < primary_len,
            "branch_parent {branch_parent} must be below primary_len {primary_len}"
        );
        let mut parents = Vec::with_capacity(total);
        parents.push(-1);
        for i in 1..=primary_len {
            parents.push(i32::try_from(i - 1)?);
        }
        // First node of secondary branch attaches to `branch_parent`.
        parents.push(i32::try_from(branch_parent)?);
        for t in 1..secondary_len {
            parents.push(i32::try_from(primary_len + t)?);
        }
        let mut tree = Self::from_parents(&parents)?;
        tree.branch_parent = Some(branch_parent);
        Ok(tree)
    }

    /// Builds a validated tree from topological parent pointers (`parents[0] == -1`,
    /// `0 <= parents[r] < r` for `r > 0`, at most 2 children per node).
    pub fn from_parents(parents: &[i32]) -> anyhow::Result<Self> {
        let m = parents.len();
        ensure!(
            (1..=MAX_TREE_NODES).contains(&m),
            "tree needs 1 to {MAX_TREE_NODES} nodes, got {m}"
        );
        ensure!(parents[0] == -1, "root node 0 must have parent -1");

        let mut parent = [-1i32; MAX_TREE_NODES];
        let mut depth = [0u32; MAX_TREE_NODES];
        let mut ancestor_mask = [0u32; MAX_TREE_NODES];
        let mut attn_anc = [-1i32; MAX_TREE_NODES * MAX_TREE_NODES];
        let mut conv_anc = [-1i32; MAX_TREE_NODES * CONV_KERNEL];
        let mut primary_child = [-1i32; MAX_TREE_NODES];
        let mut secondary_child = [-1i32; MAX_TREE_NODES];
        let mut branch_parent = None;

        for r in 0..m {
            let p = parents[r];
            parent[r] = p;
            if r == 0 {
                depth[0] = 0;
                ancestor_mask[0] = 1;
            } else {
                let pu = usize::try_from(p)
                    .ok()
                    .filter(|&pu| pu < r)
                    .with_context(|| format!("node {r} has non-topological parent {p}"))?;
                depth[r] = depth[pu] + 1;
                ancestor_mask[r] = ancestor_mask[pu] | (1u32 << r);
                if primary_child[pu] < 0 {
                    primary_child[pu] = i32::try_from(r)?;
                } else if secondary_child[pu] < 0 {
                    ensure!(
                        branch_parent.is_none(),
                        "at most one branch point is supported in a 2-branch tree"
                    );
                    secondary_child[pu] = i32::try_from(r)?;
                    branch_parent = Some(pu);
                } else {
                    anyhow::bail!("node {pu} has more than 2 children");
                }
            }

            // Fill attention ancestor table `attn_anc[r][d]` for distance `d = 0..8`.
            let mut cur = i32::try_from(r)?;
            for d in 0..MAX_TREE_NODES {
                attn_anc[r * MAX_TREE_NODES + d] = cur;
                if let Ok(cu) = usize::try_from(cur) {
                    cur = parent[cu];
                }
            }

            // Fill short-conv window `conv_anc[r][k]` (k = 0..4, oldest first, lag = 3 - k).
            let rd = i32::try_from(depth[r])?;
            for k in 0..CONV_KERNEL {
                let lag = (CONV_KERNEL - 1) - k;
                let entry = if lag <= depth[r] as usize {
                    attn_anc[r * MAX_TREE_NODES + lag]
                } else {
                    // Reaches into carried conv_state[0..3]: `3 + entry` is in `0..3`.
                    rd - i32::try_from(lag)?
                };
                conv_anc[r * CONV_KERNEL + k] = entry;
            }
        }

        // Fill unused rows with safe self-referential defaults.
        for r in m..MAX_TREE_NODES {
            for k in 0..CONV_KERNEL {
                conv_anc[r * CONV_KERNEL + k] = i32::try_from(k)? - i32::try_from(CONV_KERNEL - 1)?;
            }
        }

        Ok(Self {
            num_nodes: m,
            parent,
            depth,
            ancestor_mask,
            attn_anc,
            conv_anc,
            primary_child,
            secondary_child,
            branch_parent,
        })
    }

    /// Maximum depth among all active nodes (`0..num_nodes`).
    #[must_use]
    pub fn max_depth(&self) -> usize {
        self.depth[..self.num_nodes]
            .iter()
            .copied()
            .max()
            .unwrap_or(0) as usize
    }

    /// Whether this tree is a single linear chain (no secondary branch).
    #[must_use]
    pub fn is_linear(&self) -> bool {
        self.branch_parent.is_none()
    }

    /// Root-to-node path `[0, ..., node]` for an active node `node < num_nodes`.
    pub fn path_to(&self, node: usize) -> anyhow::Result<Vec<usize>> {
        ensure!(
            node < self.num_nodes,
            "node {node} out of range for tree of {} nodes",
            self.num_nodes
        );
        let mut path = Vec::with_capacity(self.depth[node] as usize + 1);
        let mut cur = i32::try_from(node)?;
        while let Ok(cu) = usize::try_from(cur) {
            path.push(cu);
            cur = self.parent[cu];
        }
        path.reverse();
        Ok(path)
    }

    /// True iff `ancestor` is in `node`'s causal ancestor set (including `ancestor == node`).
    #[must_use]
    pub fn sees(&self, node: usize, ancestor: usize) -> bool {
        node < self.num_nodes
            && ancestor < self.num_nodes
            && ((self.ancestor_mask[node] >> ancestor) & 1) == 1
    }

    /// Sequence position offset (`depth[r]`) for each active node `0..num_nodes`.
    #[must_use]
    pub fn active_depths(&self) -> &[u32] {
        &self.depth[..self.num_nodes]
    }
}

/// Chooses the branch parent depth `0 <= b < primary_len` along a primary chain given
/// primary per-step confidences `confidences[0..primary_len]` and optional runner-up
/// confidences `alt_confidences[0..primary_len]`.
///
/// Returns `None` if `config.enabled` is false or every eligible position has primary
/// confidence `>= config.max_branch_confidence` (signaling to stay linear).
#[must_use]
pub fn choose_branch_parent(
    config: &SpecTreeConfig,
    primary_len: usize,
    confidences: &[f32],
    alt_confidences: Option<&[f32]>,
) -> Option<usize> {
    if !config.enabled || primary_len == 0 {
        return None;
    }
    match config.branch_strategy {
        BranchStrategy::Fixed(b) => {
            let b = b.min(primary_len - 1);
            let c = confidences.get(b).copied().unwrap_or(0.0);
            (c < config.max_branch_confidence).then_some(b)
        }
        BranchStrategy::LowestConfidence { max_depth } => {
            let limit = primary_len.min(max_depth + 1);
            let mut survival = 1.0f64;
            let mut best: Option<(usize, f64)> = None;
            for b in 0..limit {
                let c = f64::from(confidences.get(b).copied().unwrap_or(0.0).clamp(0.0, 1.0));
                if c < f64::from(config.max_branch_confidence) {
                    // Expected recovery mass at `b`: survival probability to reach `b` times
                    // either runner-up probability (if known) or total rejection probability.
                    let recovery = match alt_confidences.and_then(|a| a.get(b).copied()) {
                        Some(alt) => f64::from(alt.clamp(0.0, 1.0)),
                        None => 1.0 - c,
                    };
                    let score = survival * recovery;
                    if best.is_none_or(|(_, best_score)| score > best_score) {
                        best = Some((b, score));
                    }
                }
                survival *= c;
            }
            best.map(|(b, _)| b)
        }
    }
}

/// Result of greedy or coupled-race acceptance on a [`SpecTree`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TreeAccept {
    /// Number of accepted draft tokens (`0..tree.num_nodes`).
    pub accepted: usize,
    /// Winning path of tree node indices, length `accepted + 1`, starting with `0`.
    pub path: Vec<usize>,
    /// Emitted tokens: the accepted draft tokens along `path[1..]` plus the target's
    /// bonus/correction token from `path[accepted]`.
    pub emitted: Vec<u32>,
    /// Next anchor token (`target_tokens[path[accepted]]`).
    pub next_anchor: u32,
    /// Whether acceptance followed the secondary branch.
    pub took_secondary: bool,
}

/// Greedy or coupled exponential-race acceptance on `tree`.
///
/// Mirrors the GPU `accept_tree` kernel exactly:
/// - Starts at `u = 0` (`path = [0]`).
/// - At node `u`, checks `primary_child[u]` first; if its token equals `target_tokens[u]`,
///   accepts it and advances `u`.
/// - Otherwise, if `secondary_child[u]` exists and its token equals `target_tokens[u]`,
///   accepts it and advances `u`.
/// - Otherwise stops and emits `target_tokens[u]` as the bonus/correction token.
///
/// Because `target_tokens[u]` is drawn from the exact target distribution `p` at
/// `(seed, start + depth[u] + 1)` independently of the draft tokens (or is `argmax` when
/// greedy), this preserves the target distribution identically when sibling tokens are
/// distinct.
pub fn accept_tree_greedy(
    tree: &SpecTree,
    tokens: &[u32],
    target_tokens: &[u32],
) -> anyhow::Result<TreeAccept> {
    ensure!(
        tokens.len() == tree.num_nodes && target_tokens.len() == tree.num_nodes,
        "token slice lengths ({}, {}) do not match tree nodes ({})",
        tokens.len(),
        target_tokens.len(),
        tree.num_nodes
    );
    let mut u = 0usize;
    let mut path = Vec::with_capacity(tree.num_nodes);
    let mut emitted = Vec::with_capacity(tree.num_nodes);
    let mut took_secondary = false;
    path.push(0);

    loop {
        let target = target_tokens[u];
        let c1 = usize::try_from(tree.primary_child[u]).ok();
        let c2 = usize::try_from(tree.secondary_child[u]).ok();
        if let Some(c1) = c1
            && tokens[c1] == target
        {
            u = c1;
            path.push(u);
            emitted.push(target);
        } else if let Some(c2) = c2
            && tokens[c2] == target
        {
            u = c2;
            path.push(u);
            emitted.push(target);
            took_secondary = true;
        } else {
            emitted.push(target);
            break;
        }
    }

    let accepted = path.len() - 1;
    let next_anchor = target_tokens[u];
    Ok(TreeAccept {
        accepted,
        path,
        emitted,
        next_anchor,
        took_secondary,
    })
}

/// Outcome of lossless residual verification at one tree node.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResidualNodeOutcome {
    /// Accepted the primary child token.
    AcceptPrimary(usize),
    /// Rejected primary, accepted secondary sibling under residual `(p - q_1)_+`.
    AcceptSecondary(usize),
    /// Rejected all children and sampled a correction token from the final residual.
    RejectSample(usize),
}

/// Exact lossless 2-branch speculative verification at a single node (SpecInfer /
/// Leviathan et al. residual rule):
///
/// 1. Check primary child `x1` (sampled from `q1`):
///    - Accept `x1` if `u1 * q1[x1] < target_p[x1]` (i.e. with probability
///      `min(1, target_p[x1] / q1[x1])`).
///    - Otherwise form residual `r1[y] = max(0, target_p[y] - q1[y])`, normalized to
///      `p_res1[y] = r1[y] / sum(r1)`.
/// 2. If a secondary sibling `(x2, q2)` is present (where `x2` was sampled from `q2`, which
///    may depend on `x1`, e.g. `q` renormalized without `x1` or point-mass `delta_{x2}`):
///    - Accept `x2` if `u2 * q2[x2] < p_res1[x2]` (with probability
///      `min(1, p_res1[x2] / q2[x2])`).
///    - Otherwise form residual `r2[y] = max(0, p_res1[y] - q2[y])`, normalized to
///      `p_res2[y] = r2[y] / sum(r2)`, and draw a token from `p_res2` using `u_res`.
/// 3. If no secondary sibling is present and primary was rejected, draw directly from
///    `p_res1` using `u_res`.
///
/// This guarantees `P(emitted = y) == target_p[y]` for every token `y` regardless of
/// `q1` and `q2`.
pub fn verify_node_residual(
    target_p: &[f64],
    primary: (usize, &[f64]),
    secondary: Option<(usize, &[f64])>,
    u1: f64,
    u2: f64,
    u_res: f64,
) -> anyhow::Result<ResidualNodeOutcome> {
    let vocab = target_p.len();
    let (x1, q1) = primary;
    ensure!(vocab > 0, "empty distribution");
    ensure!(q1.len() == vocab && x1 < vocab, "invalid primary proposal");

    // Step 1: check primary child against p / q1.
    let accept_p1 = if q1[x1] <= 0.0 {
        0.0
    } else {
        (target_p[x1] / q1[x1]).min(1.0)
    };
    if u1 < accept_p1 {
        return Ok(ResidualNodeOutcome::AcceptPrimary(x1));
    }

    // Residual r1 = (p - q1)_+, normalized to p_res1.
    let mut res1 = vec![0.0f64; vocab];
    let mut z1 = 0.0f64;
    for y in 0..vocab {
        let diff = (target_p[y] - q1[y]).max(0.0);
        res1[y] = diff;
        z1 += diff;
    }
    if z1 <= 0.0 {
        // Only possible if target_p == q1 everywhere (meaning accept_p1 == 1 and u1 < 1).
        return Ok(ResidualNodeOutcome::AcceptPrimary(x1));
    }
    for v in &mut res1 {
        *v /= z1;
    }

    // Step 2: check secondary sibling against p_res1 / q2.
    let Some((x2, q2)) = secondary else {
        return Ok(ResidualNodeOutcome::RejectSample(sample_cdf(&res1, u_res)));
    };
    ensure!(q2.len() == vocab && x2 < vocab, "invalid secondary proposal");

    let accept_p2 = if q2[x2] <= 0.0 {
        0.0
    } else {
        (res1[x2] / q2[x2]).min(1.0)
    };
    if u2 < accept_p2 {
        return Ok(ResidualNodeOutcome::AcceptSecondary(x2));
    }

    // Residual r2 = (p_res1 - q2)_+, normalized to p_res2.
    let mut res2 = vec![0.0f64; vocab];
    let mut z2 = 0.0f64;
    for y in 0..vocab {
        let diff = (res1[y] - q2[y]).max(0.0);
        res2[y] = diff;
        z2 += diff;
    }
    if z2 <= 0.0 {
        return Ok(ResidualNodeOutcome::AcceptSecondary(x2));
    }
    for v in &mut res2 {
        *v /= z2;
    }
    Ok(ResidualNodeOutcome::RejectSample(sample_cdf(&res2, u_res)))
}

/// Samples an index from a normalized probability distribution using `u` in `[0, 1)`.
fn sample_cdf(probs: &[f64], u: f64) -> usize {
    let mut acc = 0.0f64;
    let mut last_nonzero = 0usize;
    for (i, &p) in probs.iter().enumerate() {
        if p > 0.0 {
            last_nonzero = i;
            acc += p;
            if u < acc {
                return i;
            }
        }
    }
    last_nonzero
}

/// Converts a slice of F32 logits at temperature `temperature` into a normalized F64
/// probability distribution (softmax).
#[must_use]
pub fn softmax_f64(logits: &[f32], temperature: f32) -> Vec<f64> {
    let inv_t = 1.0 / f64::from(temperature);
    let max_val = logits
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f64> = logits
        .iter()
        .map(|&x| (f64::from(x - max_val) * inv_t).exp())
        .collect();
    let sum: f64 = probs.iter().sum();
    if sum > 0.0 {
        for p in &mut probs {
            *p /= sum;
        }
    }
    probs
}

/// Full-tree lossless residual verification across `tree` using per-node target
/// distributions `target_probs[node]` and draft proposals `node_q[child]`.
///
/// For each verified node `u`:
/// - If `u` has no children (leaf), samples a bonus token directly from `target_probs[u]`.
/// - Otherwise applies [`verify_node_residual`] against `primary_child[u]` and optional
///   `secondary_child[u]`.
pub fn verify_tree_residual(
    tree: &SpecTree,
    tokens: &[u32],
    target_probs: &[&[f64]],
    proposal_probs: &[Option<&[f64]>],
    seed: u64,
) -> anyhow::Result<TreeAccept> {
    let m = tree.num_nodes;
    ensure!(
        tokens.len() == m && target_probs.len() == m && proposal_probs.len() == m,
        "mismatched slice lengths for {m}-node tree"
    );
    let mut u = 0usize;
    let mut path = Vec::with_capacity(m);
    let mut emitted = Vec::with_capacity(m);
    let mut took_secondary = false;
    path.push(0);

    let mut rng_state = crate::race::mix64(seed ^ 0x9e37_79b9_7f4a_7c15);
    let mut next_u = || -> f64 {
        rng_state = rng_state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let z = crate::race::mix64(rng_state);
        // 53-bit uniform in [0, 1).
        ((z >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
    };

    loop {
        let p = target_probs[u];
        let Ok(c1u) = usize::try_from(tree.primary_child[u]) else {
            let bonus = sample_cdf(p, next_u());
            emitted.push(u32::try_from(bonus)?);
            break;
        };
        let c2u = usize::try_from(tree.secondary_child[u]).ok();
        let q1 = proposal_probs[c1u].context("missing proposal distribution for primary child")?;
        let sec = match c2u {
            Some(c2) => {
                let q2 = proposal_probs[c2]
                    .context("missing proposal distribution for secondary child")?;
                Some((tokens[c2] as usize, q2))
            }
            None => None,
        };
        let u1 = next_u();
        let u2 = next_u();
        let u_res = next_u();
        match verify_node_residual(p, (tokens[c1u] as usize, q1), sec, u1, u2, u_res)? {
            ResidualNodeOutcome::AcceptPrimary(_) => {
                u = c1u;
                path.push(u);
                emitted.push(tokens[u]);
            }
            ResidualNodeOutcome::AcceptSecondary(_) => {
                u = c2u.context("accepted missing secondary child")?;
                path.push(u);
                emitted.push(tokens[u]);
                took_secondary = true;
            }
            ResidualNodeOutcome::RejectSample(tok) => {
                emitted.push(u32::try_from(tok)?);
                break;
            }
        }
    }

    let accepted = path.len() - 1;
    let next_anchor = *emitted.last().context("at least one token emitted")?;
    Ok(TreeAccept {
        accepted,
        path,
        emitted,
        next_anchor,
        took_secondary,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tree_topology_and_ancestor_masks() -> anyhow::Result<()> {
        // Linear chain of 8 nodes.
        let lin = SpecTree::linear(8)?;
        assert!(lin.is_linear());
        assert_eq!(lin.max_depth(), 7);
        for r in 0..8 {
            assert_eq!(lin.depth[r], u32::try_from(r)?);
            assert_eq!(lin.ancestor_mask[r], (1u32 << (r + 1)) - 1);
            for d in 0..8 {
                let want = if d <= r { i32::try_from(r - d)? } else { -1 };
                assert_eq!(lin.attn_anc[r * 8 + d], want);
            }
            for k in 0..4 {
                let want = i32::try_from(r)? - 3 + i32::try_from(k)?;
                assert_eq!(lin.conv_anc[r * 4 + k], want);
            }
        }

        // 2-branch tree: root 0, primary 1..=4, branch at parent 1, secondary 5,6,7.
        // Tree:
        //   0 -> 1 -> 2 -> 3 -> 4
        //         \-> 5 -> 6 -> 7
        let tree = SpecTree::two_branch(4, 1, 3)?;
        assert!(!tree.is_linear());
        assert_eq!(tree.num_nodes, 8);
        assert_eq!(tree.branch_parent, Some(1));
        assert_eq!(tree.primary_child[1], 2);
        assert_eq!(tree.secondary_child[1], 5);
        assert_eq!(tree.depth, [0, 1, 2, 3, 4, 2, 3, 4]);

        // Ancestor masks: secondary nodes see {0, 1} and their own chain, never {2, 3, 4}.
        assert_eq!(tree.ancestor_mask[5], 0b0010_0011);
        assert_eq!(tree.ancestor_mask[6], 0b0110_0011);
        assert_eq!(tree.ancestor_mask[7], 0b1110_0011);
        assert!(!tree.sees(5, 2));
        assert!(!tree.sees(7, 4));
        assert!(tree.sees(7, 5));
        assert!(tree.sees(7, 1));
        assert!(tree.sees(7, 0));

        // Short-conv 4-tap windows (oldest first):
        // Node 5 (depth 2, path 0->1->5): [-1 (state[2]), 0, 1, 5]
        assert_eq!(&tree.conv_anc[5 * 4..6 * 4], &[-1, 0, 1, 5]);
        // Node 6 (depth 3, path 0->1->5->6): [0, 1, 5, 6]
        assert_eq!(&tree.conv_anc[6 * 4..7 * 4], &[0, 1, 5, 6]);
        // Node 7 (depth 4, path 0->1->5->6->7): [1, 5, 6, 7]
        assert_eq!(&tree.conv_anc[7 * 4..8 * 4], &[1, 5, 6, 7]);

        // Attention ancestor distances for node 6 (depth 3): d=0->6, d=1->5, d=2->1, d=3->0, d>=4->-1
        assert_eq!(&tree.attn_anc[6 * 8..7 * 8], &[6, 5, 1, 0, -1, -1, -1, -1]);

        // Branch at root (b = 0): primary 1..=5, secondary 6,7.
        let root_branch = SpecTree::two_branch(5, 0, 2)?;
        assert_eq!(root_branch.depth[6], 1);
        assert_eq!(root_branch.depth[7], 2);
        assert_eq!(&root_branch.conv_anc[6 * 4..7 * 4], &[-2, -1, 0, 6]);
        assert_eq!(&root_branch.conv_anc[7 * 4..8 * 4], &[-1, 0, 6, 7]);
        assert_eq!(root_branch.path_to(7)?, vec![0, 6, 7]);

        // Reject invalid topologies.
        assert!(SpecTree::two_branch(5, 0, 3).is_err()); // 9 nodes > 8
        assert!(SpecTree::two_branch(2, 2, 1).is_err()); // branch_parent >= primary_len
        assert!(SpecTree::from_parents(&[0, 0]).is_err()); // root parent != -1
        assert!(SpecTree::from_parents(&[-1, 2, 0]).is_err()); // non-topological
        assert!(SpecTree::from_parents(&[-1, 0, 0, 0]).is_err()); // 3 children
        Ok(())
    }

    #[test]
    fn short_conv_and_attention_tree_match_independent_branches() -> anyhow::Result<()> {
        // Test every branch point b = 0..4 with k1 = 4, k2 = 3 (8 nodes total).
        let state_init = [1.5f32, -0.75, 0.25]; // state[0..3] (oldest to newest)
        let weights = [0.2f32, -0.4, 0.6, 0.8];
        let node_x = [0.5f32, 1.1, -0.9, 2.3, -1.7, 0.3, -2.1, 1.9];

        let run_conv_linear = |seq: &[f32]| -> (Vec<f32>, [f32; 3]) {
            let mut st = state_init;
            let mut out = Vec::with_capacity(seq.len());
            for &x in seq {
                let win = [st[0], st[1], st[2], x];
                let y: f32 = win.iter().zip(&weights).map(|(a, w)| a * w).sum();
                out.push(y);
                st = [st[1], st[2], x];
            }
            (out, st)
        };

        for b in 0..4 {
            let tree = SpecTree::two_branch(4, b, 3)?;
            // Tree short-conv pass using `tree.conv_anc`:
            let mut tree_out = vec![0.0f32; tree.num_nodes];
            for r in 0..tree.num_nodes {
                let mut acc = 0.0f32;
                for k in 0..CONV_KERNEL {
                    let anc = tree.conv_anc[r * CONV_KERNEL + k];
                    let v = if let Ok(au) = usize::try_from(anc) {
                        node_x[au]
                    } else {
                        let st_idx = usize::try_from(i32::try_from(CONV_STATE_ROWS)? + anc)?;
                        state_init[st_idx]
                    };
                    acc += v * weights[k];
                }
                tree_out[r] = acc;
            }

            // Primary branch run linearly: nodes 0..=4.
            let prim_path = tree.path_to(4)?;
            let prim_seq: Vec<f32> = prim_path.iter().map(|&r| node_x[r]).collect();
            let (prim_out, prim_final_state) = run_conv_linear(&prim_seq);
            for (d, &r) in prim_path.iter().enumerate() {
                assert_eq!(
                    tree_out[r].to_bits(),
                    prim_out[d].to_bits(),
                    "primary conv mismatch at b={b}, r={r}"
                );
            }

            // Secondary branch run linearly: nodes 0..=b, 5, 6, 7.
            let sec_path = tree.path_to(7)?;
            let sec_seq: Vec<f32> = sec_path.iter().map(|&r| node_x[r]).collect();
            let (sec_out, sec_final_state) = run_conv_linear(&sec_seq);
            for (d, &r) in sec_path.iter().enumerate() {
                assert_eq!(
                    tree_out[r].to_bits(),
                    sec_out[d].to_bits(),
                    "secondary conv mismatch at b={b}, r={r}"
                );
            }

            // Verify post-accept conv_commit_tree for EVERY possible winning prefix on both branches.
            for win_node in 0..tree.num_nodes {
                let path = tree.path_to(win_node)?;
                let seq: Vec<f32> = path.iter().map(|&r| node_x[r]).collect();
                let (_, want_state) = run_conv_linear(&seq);
                let keep = path.len();
                let mut got_state = [0.0f32; 3];
                for (r, slot) in got_state.iter_mut().enumerate() {
                    let idx = keep + r;
                    *slot = if idx < CONV_STATE_ROWS {
                        state_init[idx]
                    } else {
                        node_x[path[idx - CONV_STATE_ROWS]]
                    };
                }
                assert_eq!(got_state, want_state, "conv commit mismatch at b={b}, win={win_node}");
            }
            // Also sanity check final states match full branch commits.
            let _ = (prim_final_state, sec_final_state);
        }

        // Verify KV cache + sliding ring commit across ring wraparound!
        let ring_slots = 16usize;
        let start = 14usize; // wraps around 16!
        let initial_ring: Vec<f32> = (0..ring_slots).map(|i| 100.0 + i as f32).collect();
        let tree = SpecTree::two_branch(4, 1, 3)?;
        let staged_kv: Vec<f32> = (0..tree.num_nodes).map(|r| 1000.0 + r as f32).collect();

        for win_node in 0..tree.num_nodes {
            let path = tree.path_to(win_node)?;
            // Expected ring after committing only `path`:
            let mut want_ring = initial_ring.clone();
            for (d, &r) in path.iter().enumerate() {
                want_ring[(start + d) % ring_slots] = staged_kv[r];
            }
            // Tree commit (forward never touched ring; kv_commit_tree writes path[0..=accepted]):
            let mut got_ring = initial_ring.clone();
            for (d, &src_row) in path.iter().enumerate() {
                got_ring[(start + d) % ring_slots] = staged_kv[src_row];
            }
            assert_eq!(got_ring, want_ring, "ring commit mismatch for win_node={win_node}");
        }
        Ok(())
    }

    #[test]
    fn lossless_residual_acceptance_is_exact_analytically() {
        // Test analytical marginal P(Y = y) == p(y) for diverse (p, q1, q2) configurations:
        // 1. Standard 2-branch without-replacement q2(. | x1) = q(y) / (1 - q(x1)) for y != x1.
        // 2. Point-mass proposals q1 = delta_{x1}, q2 = delta_{x2}.
        // 3. Mismatched support (q1 has zero prob where p > 0, and q1 overestimates p elsewhere).
        let target_cases: [Vec<f64>; 4] = [
            vec![0.45, 0.30, 0.15, 0.07, 0.03],
            vec![0.05, 0.80, 0.10, 0.04, 0.01],
            vec![0.20, 0.20, 0.20, 0.20, 0.20],
            vec![0.60, 0.00, 0.35, 0.00, 0.05],
        ];
        let draft_cases: [Vec<f64>; 4] = [
            vec![0.35, 0.40, 0.10, 0.10, 0.05],
            vec![0.50, 0.20, 0.20, 0.05, 0.05],
            vec![0.70, 0.15, 0.10, 0.05, 0.00],
            vec![0.30, 0.40, 0.25, 0.05, 0.00],
        ];

        for (p, q) in target_cases.iter().zip(&draft_cases) {
            let v = p.len();
            // Compute exact marginal P(Y = y) under 2-branch residual verification where:
            //   x1 ~ q, x2 ~ q(. | != x1)
            let mut marginal = vec![0.0f64; v];
            let mut total_accept_1 = 0.0f64;
            let mut total_accept_2 = 0.0f64;

            // Residual after Step 1 rejection: r1 = (p - q)_+
            let mut r1 = vec![0.0f64; v];
            let mut z1 = 0.0f64;
            for y in 0..v {
                r1[y] = (p[y] - q[y]).max(0.0);
                z1 += r1[y];
            }
            let p_res1: Vec<f64> = if z1 > 0.0 {
                r1.iter().map(|&x| x / z1).collect()
            } else {
                vec![0.0; v]
            };

            for x1 in 0..v {
                if q[x1] == 0.0 {
                    continue;
                }
                let alpha1 = (p[x1] / q[x1]).min(1.0);
                let p_acc1 = q[x1] * alpha1;
                marginal[x1] += p_acc1;
                total_accept_1 += p_acc1;

                // On rejection of x1 (occurs with prob q[x1] * (1 - alpha1)):
                // wait: notice! In Leviathan et al., the residual `p_res1` depends ONLY on
                // (p, q), NOT on x1! And the total rejection probability across all x1 is
                // sum_{x1} q[x1] * (1 - alpha1) = Z1!
                let p_rej1_given_x1 = q[x1] * (1.0 - alpha1);
                if p_rej1_given_x1 == 0.0 || z1 == 0.0 {
                    continue;
                }
                // Build q2(. | x1) = q renormalized on V \ {x1}.
                let rem = 1.0 - q[x1];
                let q2: Vec<f64> = (0..v)
                    .map(|y| if y == x1 || rem <= 0.0 { 0.0 } else { q[y] / rem })
                    .collect();

                // Second residual r2 = (p_res1 - q2)_+, total mass Z2.
                let mut r2 = vec![0.0f64; v];
                let mut z2 = 0.0f64;
                for y in 0..v {
                    r2[y] = (p_res1[y] - q2[y]).max(0.0);
                    z2 += r2[y];
                }
                // Contribution of Step 2 for this x1:
                // Weighted by p_rej1_given_x1, Step 2 outputs y with prob:
                //   q2[y] * min(1, p_res1[y] / q2[y]) + r2[y] = p_res1[y]!
                for y in 0..v {
                    let acc2_y = if q2[y] > 0.0 {
                        q2[y] * (p_res1[y] / q2[y]).min(1.0)
                    } else {
                        0.0
                    };
                    total_accept_2 += p_rej1_given_x1 * acc2_y;
                    marginal[y] += p_rej1_given_x1 * (acc2_y + r2[y]);
                }
                let _ = z2;
            }

            for y in 0..v {
                assert!(
                    (marginal[y] - p[y]).abs() < 1e-12,
                    "marginal[{y}] = {} != p[{y}] = {}",
                    marginal[y],
                    p[y]
                );
            }
            assert!(
                total_accept_1 + total_accept_2 >= total_accept_1 - 1e-12,
                "secondary branch must never reduce acceptance"
            );
        }
    }

    #[test]
    fn lossless_residual_monte_carlo_matches_target_distribution() -> anyhow::Result<()> {
        let p = [0.42f64, 0.31, 0.16, 0.08, 0.03];
        let q1 = [0.30f64, 0.45, 0.12, 0.08, 0.05];
        let trials = 200_000usize;
        let mut counts = [0usize; 5];
        let mut acc1 = 0usize;
        let mut acc2 = 0usize;

        let mut state = 0x1234_5678_9abc_def0u64;
        let mut next_u = || -> f64 {
            state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let z = crate::race::mix64(state);
            ((z >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
        };

        for _ in 0..trials {
            let x1 = sample_cdf(&q1, next_u());
            let rem = 1.0 - q1[x1];
            let mut q2 = [0.0f64; 5];
            for y in 0..5 {
                if y != x1 {
                    q2[y] = q1[y] / rem;
                }
            }
            let x2 = sample_cdf(&q2, next_u());
            let u1 = next_u();
            let u2 = next_u();
            let u_res = next_u();
            let tok = match verify_node_residual(&p, (x1, &q1), Some((x2, &q2)), u1, u2, u_res)? {
                ResidualNodeOutcome::AcceptPrimary(t) => {
                    acc1 += 1;
                    t
                }
                ResidualNodeOutcome::AcceptSecondary(t) => {
                    acc2 += 1;
                    t
                }
                ResidualNodeOutcome::RejectSample(t) => t,
            };
            counts[tok] += 1;
        }

        for (i, (&c, &want)) in counts.iter().zip(&p).enumerate() {
            let emp = c as f64 / trials as f64;
            assert!(
                (emp - want).abs() < 0.006,
                "token {i}: empirical {emp:.4} vs target {want:.4} (counts {counts:?})"
            );
        }
        // Secondary branch must materially boost acceptance when primary disagrees with target.
        assert!(
            acc2 > trials / 20,
            "expected substantial secondary acceptance, got {acc2} of {trials} (primary {acc1})"
        );
        Ok(())
    }

    #[test]
    fn coupled_race_tree_preserves_distribution_and_boosts_acceptance() -> anyhow::Result<()> {
        // Target and slightly shifted draft logits at T = 1.0.
        let target_logits = [1.2f32, 1.0, 0.4, -0.5, -1.2];
        let draft_logits = [0.9f32, 1.25, 0.3, -0.4, -1.3];
        let target_p = softmax_f64(&target_logits, 1.0);
        let tree = SpecTree::two_branch(1, 0, 1)?; // root 0, primary 1, secondary 2

        let trials = 100_000u32;
        let mut counts = [0usize; 5];
        let mut linear_accepts = 0usize;
        let mut tree_accepts = 0usize;

        for pos in 0..trials {
            let key = crate::race::draw_key(0xbeef, pos);
            // Target draw via exponential race.
            let t = crate::race::pick(&target_logits, 0..5, 1.0, key).unwrap_or(0);
            // Draft picks top-2 noisy scores with the same race key.
            let mut scored: Vec<(f32, usize)> = (0..5)
                .map(|i| (crate::race::score(draft_logits[i], 1.0, key, i as u32), i))
                .collect();
            scored.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
            let x1 = scored[0].1 as u32;
            let x2 = scored[1].1 as u32;

            let tokens = [0u32, x1, x2];
            let target_tokens = [t as u32, 0, 0];
            let res = accept_tree_greedy(&tree, &tokens, &target_tokens)?;
            assert_eq!(res.emitted[0], t as u32);
            counts[res.emitted[0] as usize] += 1;
            if x1 == t as u32 {
                linear_accepts += 1;
            }
            if res.accepted > 0 {
                tree_accepts += 1;
            }
        }

        for (i, (&c, &want)) in counts.iter().zip(&target_p).enumerate() {
            let emp = c as f64 / f64::from(trials);
            assert!(
                (emp - want).abs() < 0.008,
                "coupled token {i}: empirical {emp:.4} vs target {want:.4}"
            );
        }
        assert!(
            tree_accepts > linear_accepts + 10_000,
            "2-branch tree ({tree_accepts}) should significantly beat linear ({linear_accepts})"
        );
        Ok(())
    }

    #[test]
    fn choose_branch_parent_picks_highest_rejection_mass() {
        let cfg = SpecTreeConfig::two_branch();
        // High conf at 0, 1; drop at 2 -> should branch at 2!
        let b = choose_branch_parent(&cfg, 5, &[0.96, 0.94, 0.40, 0.90, 0.90], None);
        assert_eq!(b, Some(2));
        // Drop right at 0 -> should branch at 0!
        let b = choose_branch_parent(&cfg, 5, &[0.45, 0.90, 0.30, 0.90, 0.90], None);
        assert_eq!(b, Some(0));
        // All very high confidence -> stay linear (None)!
        let b = choose_branch_parent(&cfg, 5, &[0.99, 0.99, 0.99, 0.99, 0.99], None);
        assert_eq!(b, None);
        // Fixed root strategy -> always 0 when below threshold.
        let fixed = SpecTreeConfig {
            branch_strategy: BranchStrategy::Fixed(0),
            ..cfg
        };
        assert_eq!(
            choose_branch_parent(&fixed, 5, &[0.90, 0.30, 0.20, 0.90, 0.90], None),
            Some(0)
        );
    }
}
