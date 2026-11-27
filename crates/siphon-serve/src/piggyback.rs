//! Hybrid chunked-prefill + decode/verify piggybacking planner.
//!
//! When `--piggyback-budget` is positive and at least one request is already decoding or
//! verifying, arriving prompts carve bounded chunks (up to `--piggyback-budget` tokens)
//! that ride along inside the active decode/verify forward pass instead of running an
//! exclusive prefill step that stalls active streams' inter-token latency (ITL).
//!
//! This module holds the platform-independent scheduling policy and row-packing layout
//! validation so both the Linux CUDA engine and macOS host unit tests share one source
//! of truth.

/// Snapshot of one active request's scheduling metadata for piggyback planning.
#[cfg_attr(
    all(not(target_os = "linux"), not(test)),
    expect(dead_code, reason = "used by the Linux engine and host unit tests")
)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrefillCandidate {
    /// Index in the engine's `active` vector.
    pub index: usize,
    /// Sequence slot owning this request's KV cache.
    pub slot: usize,
    /// Submission group id (shared by `n > 1` sibling choices).
    pub group: u64,
    /// Prompt tokens already prefilled.
    pub prefilled: usize,
    /// Total prompt tokens.
    pub prompt_len: usize,
    /// Whether this request has waited at least `AGE_LIMIT`.
    pub aged: bool,
    /// Monotonic arrival tie-breaker (smaller = arrived earlier).
    pub arrival_rank: usize,
    /// Whether this request is currently prefilling (`next.is_none()`).
    pub is_prefilling: bool,
    /// Whether this request is still alive (not done, failed, cancelled, or closed).
    pub alive: bool,
}

impl PrefillCandidate {
    /// Remaining prompt tokens for a live prefilling candidate.
    #[cfg_attr(
        all(not(target_os = "linux"), not(test)),
        expect(dead_code, reason = "used by the Linux engine and host unit tests")
    )]
    #[must_use]
    pub fn remaining(self) -> usize {
        self.prompt_len.saturating_sub(self.prefilled)
    }
}

/// One carved prefill slice chosen to piggyback onto an active decode/verify pass.
#[cfg_attr(
    all(not(target_os = "linux"), not(test)),
    expect(dead_code, reason = "used by the Linux engine and host unit tests")
)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PiggybackSlice {
    /// Index in the engine's `active` vector.
    pub index: usize,
    /// Sequence slot.
    pub slot: usize,
    /// Starting prompt token offset (`prefilled`).
    pub start: usize,
    /// Ending prompt token offset (`start < end <= prompt_len`).
    pub end: usize,
    /// Whether `end == prompt_len` (slice finishes prompt and needs its first token).
    pub finishes_prompt: bool,
    /// Whether `end` lands on the prefix-cache save boundary
    /// (`(prompt_len - 1) / prefill_chunk * prefill_chunk`, non-zero).
    pub hits_prefix_boundary: bool,
}

impl PiggybackSlice {
    /// Prompt tokens carried by this slice.
    #[cfg_attr(not(test), expect(dead_code, reason = "used by host unit tests"))]
    #[must_use]
    pub const fn len(self) -> usize {
        self.end - self.start
    }
}

/// Scheduling decision for one turn of the batched engine loop.
#[cfg_attr(
    all(not(target_os = "linux"), not(test)),
    expect(dead_code, reason = "used by the Linux engine and host unit tests")
)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TurnPlan {
    /// Run the normal exclusive `prefill_chunk` path (piggybacking off, no active
    /// decoders, no room in `prefill_chunk` beyond decode rows, aged/only sibling group
    /// needing `share_prefill`, or KV pool room needs victim parking).
    ExclusivePrefill,
    /// Skip exclusive prefill and piggyback `slices` onto the active decode/verify step.
    Piggyback {
        slices: Vec<PiggybackSlice>,
        total_prefill_tokens: usize,
    },
    /// No live prefilling requests exist; proceed directly to decode/verify.
    DecodeOnly,
}

/// Pure-Rust row layout for a hybrid decode/verify + piggybacked prefill forward pass,
/// used to verify contiguous row packing (`0..batch_rows`, then each prefill slice) and
/// output unpacking invariants.
#[cfg_attr(not(test), expect(dead_code, reason = "used by host unit tests"))]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HybridPackLayout {
    pub batch_rows: usize,
    pub total_rows: usize,
    pub slice_offsets: Vec<(usize, usize, usize)>,
    pub finishing_indices: Vec<usize>,
}

/// Validates and computes the packed row layout for `batch_slots` (each contributing
/// `rows_per_seq` contiguous rows in `0..batch_rows`) followed by `slices`.
#[cfg_attr(not(test), expect(dead_code, reason = "used by host unit tests"))]
pub fn build_pack_layout(
    batch_slots: &[usize],
    rows_per_seq: usize,
    slices: &[PiggybackSlice],
    prefill_chunk: usize,
) -> anyhow::Result<HybridPackLayout> {
    anyhow::ensure!(!batch_slots.is_empty(), "hybrid pass requires active decoders");
    anyhow::ensure!(rows_per_seq > 0, "rows_per_seq must be positive");
    anyhow::ensure!(!slices.is_empty(), "hybrid pass requires piggybacked prefill slices");
    for (i, &slot) in batch_slots.iter().enumerate() {
        anyhow::ensure!(
            batch_slots[..i].iter().all(|&s| s != slot),
            "duplicate decode/verify slot {slot}"
        );
    }
    let batch_rows = batch_slots.len() * rows_per_seq;
    let mut offset = batch_rows;
    let mut slice_offsets = Vec::with_capacity(slices.len());
    let mut finishing_indices = Vec::new();
    for (i, slice) in slices.iter().enumerate() {
        anyhow::ensure!(slice.end > slice.start, "empty piggyback slice {i}");
        anyhow::ensure!(
            !batch_slots.contains(&slice.slot),
            "slot {} appears in both decode/verify and prefill",
            slice.slot
        );
        anyhow::ensure!(
            slices[..i].iter().all(|prev| prev.slot != slice.slot),
            "duplicate prefill slot {}",
            slice.slot
        );
        let len = slice.len();
        slice_offsets.push((slice.slot, offset, len));
        if slice.finishes_prompt {
            finishing_indices.push(i);
        }
        offset += len;
    }
    anyhow::ensure!(
        offset <= prefill_chunk,
        "packed hybrid rows {offset} exceed prefill_chunk {prefill_chunk}"
    );
    Ok(HybridPackLayout {
        batch_rows,
        total_rows: offset,
        slice_offsets,
        finishing_indices,
    })
}

/// Unpacks compact `finishing_tokens` (one per finishing slice in order) into a
/// per-slice `Vec<Option<u32>>` aligned with `layout.slice_offsets`.
#[cfg_attr(not(test), expect(dead_code, reason = "used by host unit tests"))]
pub fn unpack_prefill_tokens(
    layout: &HybridPackLayout,
    finishing_tokens: &[u32],
) -> anyhow::Result<Vec<Option<u32>>> {
    anyhow::ensure!(
        finishing_tokens.len() == layout.finishing_indices.len(),
        "got {} finishing tokens for {} finishing slices",
        finishing_tokens.len(),
        layout.finishing_indices.len()
    );
    let mut out = vec![None; layout.slice_offsets.len()];
    for (&slice_idx, &tok) in layout.finishing_indices.iter().zip(finishing_tokens) {
        out[slice_idx] = Some(tok);
    }
    Ok(out)
}

/// Plans whether this scheduler turn should piggyback prefill chunk(s) onto the active
/// decode/verify forward pass or fall back to exclusive prefill.
///
/// `can_fit_pages(candidate, end)` checks whether the KV pool can accommodate extending
/// `candidate` to position `end` alongside already-selected slices (without parking active
/// requests). If no eligible slice fits without parking, returns
/// [`TurnPlan::ExclusivePrefill`] so `prefill_chunk` can invoke `room_for` to park victims.
#[cfg_attr(
    all(not(target_os = "linux"), not(test)),
    expect(dead_code, reason = "used by the Linux engine and host unit tests")
)]
pub fn plan_piggyback(
    candidates: &[PrefillCandidate],
    active_decode_rows: usize,
    prefill_chunk: usize,
    piggyback_budget: Option<usize>,
    max_slices: usize,
    mut can_fit_pages: impl FnMut(&PrefillCandidate, usize) -> bool,
) -> TurnPlan {
    let mut prefilling: Vec<PrefillCandidate> = candidates
        .iter()
        .copied()
        .filter(|c| c.alive && c.is_prefilling && c.prefilled < c.prompt_len)
        .collect();
    if prefilling.is_empty() {
        return TurnPlan::DecodeOnly;
    }
    let Some(budget) = piggyback_budget else {
        return TurnPlan::ExclusivePrefill;
    };
    if budget == 0 || active_decode_rows == 0 || max_slices == 0 || prefill_chunk == 0 {
        return TurnPlan::ExclusivePrefill;
    }
    let avail_tokens = budget.min(prefill_chunk.saturating_sub(active_decode_rows));
    if avail_tokens == 0 {
        return TurnPlan::ExclusivePrefill;
    }

    // Priority order matches `prefill_chunk`: aged requests first (FIFO), then shortest
    // remaining prompt (SRPT), with arrival order breaking ties.
    prefilling.sort_by_key(|c| {
        if c.aged {
            (0u8, c.arrival_rank, 0usize)
        } else {
            (1u8, c.remaining(), c.arrival_rank)
        }
    });

    // Exclude requests with unprefilled siblings in the same submission group so
    // `share_prefill` + `redraw_slot` remains exact. If the top-priority request is aged
    // and has unprefilled siblings, fall back to exclusive prefill so it never starves.
    let has_unprefilled_sibling = |c: &PrefillCandidate| {
        prefilling
            .iter()
            .any(|o| o.group == c.group && o.slot != c.slot && o.prefilled == 0)
    };
    if prefilling[0].aged && has_unprefilled_sibling(&prefilling[0]) {
        return TurnPlan::ExclusivePrefill;
    }

    let mut remaining = avail_tokens;
    let mut slices = Vec::new();
    for c in &prefilling {
        if remaining == 0 || slices.len() >= max_slices {
            break;
        }
        if has_unprefilled_sibling(c) {
            continue;
        }
        let left = c.remaining();
        let boundary = c.prompt_len.saturating_sub(1) / prefill_chunk * prefill_chunk;
        let mut take = left.min(remaining);
        // Never step across the prefix-cache save boundary without stopping on it.
        if boundary > 0 && c.prefilled < boundary && c.prefilled + take > boundary {
            take = boundary - c.prefilled;
        }
        let end = c.prefilled + take;
        if !can_fit_pages(c, end) {
            continue;
        }
        slices.push(PiggybackSlice {
            index: c.index,
            slot: c.slot,
            start: c.prefilled,
            end,
            finishes_prompt: end == c.prompt_len,
            hits_prefix_boundary: boundary > 0 && end == boundary,
        });
        remaining -= take;
    }
    if slices.is_empty() {
        return TurnPlan::ExclusivePrefill;
    }
    TurnPlan::Piggyback {
        total_prefill_tokens: avail_tokens - remaining,
        slices,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cand(
        index: usize,
        slot: usize,
        group: u64,
        prefilled: usize,
        prompt_len: usize,
    ) -> PrefillCandidate {
        PrefillCandidate {
            index,
            slot,
            group,
            prefilled,
            prompt_len,
            aged: false,
            arrival_rank: index,
            is_prefilling: true,
            alive: true,
        }
    }

    #[test]
    fn disabled_flag_always_uses_exclusive_prefill() {
        let cands = [cand(0, 1, 1, 0, 512)];
        let plan = plan_piggyback(&cands, 4, 2048, None, 64, |_, _| true);
        assert_eq!(plan, TurnPlan::ExclusivePrefill);
    }

    #[test]
    fn decode_only_when_no_prefillers() {
        let mut decoding = cand(0, 0, 1, 512, 512);
        decoding.is_prefilling = false;
        let plan = plan_piggyback(&[decoding], 1, 2048, Some(256), 64, |_, _| true);
        assert_eq!(plan, TurnPlan::DecodeOnly);
    }

    #[test]
    fn no_active_decoders_uses_exclusive_prefill() {
        let cands = [cand(0, 0, 1, 0, 1024)];
        let plan = plan_piggyback(&cands, 0, 2048, Some(256), 64, |_, _| true);
        assert_eq!(plan, TurnPlan::ExclusivePrefill);
    }

    #[test]
    fn carves_single_long_prompt_within_budget() {
        let cands = [cand(1, 2, 10, 0, 1000)];
        let plan = plan_piggyback(&cands, 8, 2048, Some(256), 64, |_, _| true);
        assert_eq!(
            plan,
            TurnPlan::Piggyback {
                slices: vec![PiggybackSlice {
                    index: 1,
                    slot: 2,
                    start: 0,
                    end: 256,
                    finishes_prompt: false,
                    hits_prefix_boundary: false,
                }],
                total_prefill_tokens: 256,
            }
        );
    }

    #[test]
    fn packs_multiple_short_prompts_and_partial_long_prompt() {
        let cands = [
            cand(0, 1, 1, 0, 1000),
            cand(1, 2, 2, 0, 80),
            cand(2, 3, 3, 20, 140), // remaining = 120
        ];
        // SRPT order: idx 1 (80) -> idx 2 (120) -> idx 0 (56 of 1000)
        let plan = plan_piggyback(&cands, 16, 2048, Some(256), 64, |_, _| true);
        assert_eq!(
            plan,
            TurnPlan::Piggyback {
                slices: vec![
                    PiggybackSlice {
                        index: 1,
                        slot: 2,
                        start: 0,
                        end: 80,
                        finishes_prompt: true,
                        hits_prefix_boundary: false,
                    },
                    PiggybackSlice {
                        index: 2,
                        slot: 3,
                        start: 20,
                        end: 140,
                        finishes_prompt: true,
                        hits_prefix_boundary: false,
                    },
                    PiggybackSlice {
                        index: 0,
                        slot: 1,
                        start: 0,
                        end: 56,
                        finishes_prompt: false,
                        hits_prefix_boundary: false,
                    },
                ],
                total_prefill_tokens: 256,
            }
        );
    }

    #[test]
    fn aligns_chunk_end_to_prefix_cache_boundary_when_budget_misaligned() {
        // prompt_len = 5000, chunk = 2048 -> boundary = 4096.
        // prefilled = 4000, budget = 300 -> must cap at 4096 (96 tokens) and flag boundary.
        let cands = [cand(0, 1, 1, 4000, 5000), cand(1, 2, 2, 0, 500)];
        let plan = plan_piggyback(&cands, 4, 2048, Some(300), 64, |_, _| true);
        // SRPT picks idx 1 (500 remaining) first for 300 tokens; test idx 0 alone first:
        let plan_single = plan_piggyback(&cands[..1], 4, 2048, Some(300), 64, |_, _| true);
        assert_eq!(
            plan_single,
            TurnPlan::Piggyback {
                slices: vec![PiggybackSlice {
                    index: 0,
                    slot: 1,
                    start: 4000,
                    end: 4096,
                    finishes_prompt: false,
                    hits_prefix_boundary: true,
                }],
                total_prefill_tokens: 96,
            }
        );
        // When idx 0 is aged, it goes first (96 tokens to boundary), and idx 1 fills remaining 204.
        let mut aged_cands = cands;
        aged_cands[0].aged = true;
        let plan_aged = plan_piggyback(&aged_cands, 4, 2048, Some(300), 64, |_, _| true);
        assert_eq!(
            plan_aged,
            TurnPlan::Piggyback {
                slices: vec![
                    PiggybackSlice {
                        index: 0,
                        slot: 1,
                        start: 4000,
                        end: 4096,
                        finishes_prompt: false,
                        hits_prefix_boundary: true,
                    },
                    PiggybackSlice {
                        index: 1,
                        slot: 2,
                        start: 0,
                        end: 204,
                        finishes_prompt: false,
                        hits_prefix_boundary: false,
                    },
                ],
                total_prefill_tokens: 300,
            }
        );
        // Sanity check non-aged plan used SRPT (idx 1 took 300).
        assert!(matches!(
            plan,
            TurnPlan::Piggyback {
                total_prefill_tokens: 300,
                ..
            }
        ));
    }

    #[test]
    fn excludes_unprefilled_siblings_and_prevents_aged_sibling_starvation() {
        let sib_a = cand(0, 1, 77, 0, 100);
        let sib_b = cand(1, 2, 77, 0, 100);
        // Only siblings -> must fall back to ExclusivePrefill so `share_prefill` runs.
        assert_eq!(
            plan_piggyback(&[sib_a, sib_b], 2, 2048, Some(256), 64, |_, _| true),
            TurnPlan::ExclusivePrefill
        );
        // Siblings + a younger non-sibling -> non-sibling piggybacks cleanly.
        let solo = cand(2, 3, 88, 0, 200);
        assert_eq!(
            plan_piggyback(&[sib_a, sib_b, solo], 2, 2048, Some(256), 64, |_, _| true),
            TurnPlan::Piggyback {
                slices: vec![PiggybackSlice {
                    index: 2,
                    slot: 3,
                    start: 0,
                    end: 200,
                    finishes_prompt: true,
                    hits_prefix_boundary: false,
                }],
                total_prefill_tokens: 200,
            }
        );
        // Aged sibling group -> ExclusivePrefill so younger solo requests cannot starve it.
        let mut aged_a = sib_a;
        aged_a.aged = true;
        assert_eq!(
            plan_piggyback(&[aged_a, sib_b, solo], 2, 2048, Some(256), 64, |_, _| true),
            TurnPlan::ExclusivePrefill
        );
    }

    #[test]
    fn respects_aged_priority_over_srpt() {
        let mut old_long = cand(0, 1, 1, 0, 4000);
        old_long.aged = true;
        old_long.arrival_rank = 1;
        let mut old_second = cand(1, 2, 2, 0, 100);
        old_second.aged = true;
        old_second.arrival_rank = 2;
        let young_short = cand(2, 3, 3, 0, 20);
        let plan = plan_piggyback(
            &[young_short, old_second, old_long],
            4,
            2048,
            Some(128),
            64,
            |_, _| true,
        );
        // Aged FIFO: old_long (arrival_rank 1) takes full 128 budget first.
        assert_eq!(
            plan,
            TurnPlan::Piggyback {
                slices: vec![PiggybackSlice {
                    index: 0,
                    slot: 1,
                    start: 0,
                    end: 128,
                    finishes_prompt: false,
                    hits_prefix_boundary: false,
                }],
                total_prefill_tokens: 128,
            }
        );
    }

    #[test]
    fn skips_cancelled_and_dead_candidates() {
        let mut dead = cand(0, 1, 1, 0, 50);
        dead.alive = false;
        let live = cand(1, 2, 2, 0, 100);
        let plan = plan_piggyback(&[dead, live], 2, 2048, Some(256), 64, |_, _| true);
        assert_eq!(
            plan,
            TurnPlan::Piggyback {
                slices: vec![PiggybackSlice {
                    index: 1,
                    slot: 2,
                    start: 0,
                    end: 100,
                    finishes_prompt: true,
                    hits_prefix_boundary: false,
                }],
                total_prefill_tokens: 100,
            }
        );
        // Only dead candidates left -> DecodeOnly.
        assert_eq!(
            plan_piggyback(&[dead], 2, 2048, Some(256), 64, |_, _| true),
            TurnPlan::DecodeOnly
        );
    }

    #[test]
    fn respects_kv_room_callback_and_falls_back_when_none_fit() {
        let cands = [cand(0, 1, 1, 0, 100), cand(1, 2, 2, 0, 200)];
        // First candidate cannot fit KV pages, second fits -> picks second.
        let plan = plan_piggyback(&cands, 4, 2048, Some(256), 64, |c, _| c.slot == 2);
        assert_eq!(
            plan,
            TurnPlan::Piggyback {
                slices: vec![PiggybackSlice {
                    index: 1,
                    slot: 2,
                    start: 0,
                    end: 200,
                    finishes_prompt: true,
                    hits_prefix_boundary: false,
                }],
                total_prefill_tokens: 200,
            }
        );
        // None fit -> falls back to ExclusivePrefill so `room_for` can park victims.
        assert_eq!(
            plan_piggyback(&cands, 4, 2048, Some(256), 64, |_, _| false),
            TurnPlan::ExclusivePrefill
        );
    }

    #[test]
    fn caps_by_prefill_chunk_minus_active_decode_rows_and_max_slices() {
        let cands = [
            cand(0, 1, 1, 0, 40),
            cand(1, 2, 2, 0, 40),
            cand(2, 3, 3, 0, 40),
        ];
        // prefill_chunk = 256, active_decode_rows = 200 -> avail = 56 even though budget = 256.
        let plan = plan_piggyback(&cands, 200, 256, Some(256), 64, |_, _| true);
        assert_eq!(
            plan,
            TurnPlan::Piggyback {
                slices: vec![
                    PiggybackSlice {
                        index: 0,
                        slot: 1,
                        start: 0,
                        end: 40,
                        finishes_prompt: true,
                        hits_prefix_boundary: false,
                    },
                    PiggybackSlice {
                        index: 1,
                        slot: 2,
                        start: 0,
                        end: 16,
                        finishes_prompt: false,
                        hits_prefix_boundary: false,
                    },
                ],
                total_prefill_tokens: 56,
            }
        );
        // Active decode rows fill the entire prefill_chunk -> ExclusivePrefill.
        assert_eq!(
            plan_piggyback(&cands, 256, 256, Some(256), 64, |_, _| true),
            TurnPlan::ExclusivePrefill
        );
        // max_slices = 1 caps slice count even with leftover token budget.
        let capped = plan_piggyback(&cands, 16, 2048, Some(256), 1, |_, _| true);
        assert_eq!(
            capped,
            TurnPlan::Piggyback {
                slices: vec![PiggybackSlice {
                    index: 0,
                    slot: 1,
                    start: 0,
                    end: 40,
                    finishes_prompt: true,
                    hits_prefix_boundary: false,
                }],
                total_prefill_tokens: 40,
            }
        );
    }

    #[test]
    fn multi_turn_prefill_handoff_and_preemption_simulation() {
        // Simulates 2 active decoders (slots 0, 1) + 1 long prompt (slot 2, 600 tokens) +
        // 1 short prompt (slot 3, 100 tokens) + 1 cancelled prompt (slot 4, 50 tokens).
        let mut active = vec![
            PrefillCandidate {
                index: 0,
                slot: 0,
                group: 1,
                prefilled: 200,
                prompt_len: 200,
                aged: false,
                arrival_rank: 0,
                is_prefilling: false,
                alive: true,
            },
            PrefillCandidate {
                index: 1,
                slot: 1,
                group: 2,
                prefilled: 300,
                prompt_len: 300,
                aged: false,
                arrival_rank: 1,
                is_prefilling: false,
                alive: true,
            },
            cand(2, 2, 3, 0, 600),
            cand(3, 3, 4, 0, 100),
            cand(4, 4, 5, 0, 50),
        ];
        // Preempt / cancel slot 4 before turn 1.
        active[4].alive = false;

        let mut decode_steps_without_stall = 0usize;
        let mut handoff_turn_slot3 = None;
        let mut handoff_turn_slot2 = None;

        for turn in 0..10 {
            let decode_slots: Vec<usize> = active
                .iter()
                .filter(|c| c.alive && !c.is_prefilling)
                .map(|c| c.slot)
                .collect();
            let plan = plan_piggyback(&active, decode_slots.len(), 2048, Some(256), 64, |_, _| {
                true
            });
            match plan {
                TurnPlan::Piggyback { slices, .. } => {
                    let layout = build_pack_layout(&decode_slots, 1, &slices, 2048).unwrap();
                    let fake_firsts: Vec<u32> = (0..layout.finishing_indices.len())
                        .map(|k| 1000 + u32::try_from(k).unwrap())
                        .collect();
                    let unpacked = unpack_prefill_tokens(&layout, &fake_firsts).unwrap();
                    decode_steps_without_stall += 1;
                    for (slice, tok) in slices.iter().zip(unpacked) {
                        let req = &mut active[slice.index];
                        req.prefilled = slice.end;
                        if slice.finishes_prompt {
                            assert!(tok.is_some());
                            req.is_prefilling = false;
                            if req.slot == 3 {
                                handoff_turn_slot3 = Some(turn);
                            } else if req.slot == 2 {
                                handoff_turn_slot2 = Some(turn);
                            }
                        } else {
                            assert!(tok.is_none());
                        }
                    }
                }
                TurnPlan::DecodeOnly => {
                    break;
                }
                TurnPlan::ExclusivePrefill => {
                    panic!("unexpected exclusive prefill while decoders active on turn {turn}");
                }
            }
        }
        // Turn 0: slot 3 finishes (100) + slot 2 gets 156 (156/600).
        // Turn 1: slot 3 now decodes alongside slots 0, 1! slot 2 gets 256 (412/600).
        // Turn 2: slot 2 gets 188 (600/600) and finishes!
        // Turn 3: DecodeOnly!
        assert_eq!(handoff_turn_slot3, Some(0));
        assert_eq!(handoff_turn_slot2, Some(2));
        assert_eq!(decode_steps_without_stall, 3);
        assert_eq!(active[2].prefilled, 600);
        assert!(!active[2].is_prefilling);
        assert_eq!(active[3].prefilled, 100);
        assert!(!active[3].is_prefilling);
        // Cancelled slot 4 never ran.
        assert_eq!(active[4].prefilled, 0);
    }

    #[test]
    fn hybrid_pack_layout_validates_offsets_and_rejects_duplicate_slots() {
        let slices = vec![
            PiggybackSlice {
                index: 2,
                slot: 4,
                start: 0,
                end: 128,
                finishes_prompt: false,
                hits_prefix_boundary: false,
            },
            PiggybackSlice {
                index: 3,
                slot: 5,
                start: 0,
                end: 64,
                finishes_prompt: true,
                hits_prefix_boundary: false,
            },
        ];
        // 3 verify sequences of width 4 => batch_rows = 12.
        let layout = build_pack_layout(&[0, 1, 2], 4, &slices, 2048).unwrap();
        assert_eq!(layout.batch_rows, 12);
        assert_eq!(layout.total_rows, 12 + 128 + 64);
        assert_eq!(layout.slice_offsets, vec![(4, 12, 128), (5, 140, 64)]);
        assert_eq!(layout.finishing_indices, vec![1]);
        assert_eq!(unpack_prefill_tokens(&layout, &[42]).unwrap(), vec![None, Some(42)]);

        // Overlap between decode slot and prefill slot must fail.
        assert!(build_pack_layout(&[0, 4], 1, &slices, 2048).is_err());
        // Duplicate decode slot must fail.
        assert!(build_pack_layout(&[0, 0], 1, &slices, 2048).is_err());
        // Exceeding prefill_chunk must fail.
        assert!(build_pack_layout(&[0, 1, 2], 4, &slices, 200).is_err());
    }
}
