//! Speculative decoding helpers (`kernels/spec.cu`): the verify pass's feature taps, its
//! greedy acceptance, and the rollback of the sliding rings and conv states that rejected
//! rows changed.
//!
//! The rollback kernels address every visit's state through a table of device pointers,
//! so one launch covers all visits. The buffers they point to must stay allocated while
//! the table is in use.

use std::sync::Arc;

use anyhow::{Context as _, ensure};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg as _,
};
use half::bf16;

use crate::gpu::{DeviceIn, DeviceOut, cubin, load_module};

/// Hidden width.
pub const HIDDEN: usize = 2304;
/// Width of the key and value projections.
pub const KV_DIMS: usize = 1024;
/// Feature taps per row of the bundled draft: the partial block entering visits 5, 23 and
/// 41, and the final normed hidden state.
pub const TAPS: usize = 4;
/// Most feature taps per row that a [`TapLayout`] may ask for.
pub const MAX_TAPS: usize = 8;

/// The hidden states a draft reads for each row: the partial block entering each visit of
/// `visits`, in order, then the final normed hidden state if `final_hidden` is set.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TapLayout {
    pub visits: Vec<usize>,
    pub final_hidden: bool,
}

impl TapLayout {
    /// The bundled draft's taps (`target_layer_ids [5, 23, 41, 48]` of `mtp_config`).
    #[must_use]
    pub fn stock() -> Self {
        Self {
            visits: vec![5, 23, 41],
            final_hidden: true,
        }
    }

    /// Taps per row.
    #[must_use]
    pub fn len(&self) -> usize {
        self.visits.len() + usize::from(self.final_hidden)
    }

    /// Whether the layout records nothing.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Default for TapLayout {
    fn default() -> Self {
        Self::stock()
    }
}
/// Most rows one verify takes: the anchor and up to 7 drafts.
pub const VERIFY_ROWS: usize = 8;
const _: () = assert!(VERIFY_ROWS == siphon::memory::VERIFY_ROWS);
/// Conv state rows per stream.
pub const STATE_ROWS: usize = 3;
/// Conv streams per visit: keys, values, post-attention and post-MLP.
pub const STREAMS: usize = 4;
/// One row of a visit's conv inputs: keys, values, post-attention and post-MLP.
pub const VISIT_CHANNELS: usize = 2 * KV_DIMS + 2 * HIDDEN;
/// Channels of each conv stream.
pub const STREAM_DIMS: [usize; STREAMS] = [KV_DIMS, KV_DIMS, HIDDEN, HIDDEN];
/// I32 values of an acceptance result: the accepted count, the verify tokens and the
/// argmax of each row.
pub const RESULT_LEN: usize = 1 + 2 * VERIFY_ROWS;
/// Blocks per (stream, visit) of the conv backup and rollback.
const STATE_BLOCKS: u32 = 9;
const THREADS: u32 = 256;

fn check_len(name: &str, len: usize, want: usize) -> anyhow::Result<()> {
    ensure!(len == want, "{name} has {len} elements, expected {want}");
    Ok(())
}

#[derive(Debug)]
pub struct Spec {
    copy_tap: CudaFunction,
    conv_backup: CudaFunction,
    conv_rollback: CudaFunction,
    ring_backup: CudaFunction,
    ring_restore: CudaFunction,
    accept_greedy: CudaFunction,
}

impl Spec {
    pub fn load(ctx: &Arc<CudaContext>) -> anyhow::Result<Self> {
        let module = load_module(ctx, "spec", cubin!("spec"))?;
        let f = |name: &str| module.load_function(name);
        Ok(Self {
            copy_tap: f("copy_tap")?,
            conv_backup: f("conv_backup")?,
            conv_rollback: f("conv_rollback")?,
            ring_backup: f("ring_backup")?,
            ring_restore: f("ring_restore")?,
            accept_greedy: f("accept_greedy")?,
        })
    }

    /// Copies the rows of `src` `[rows, 2304]` into tap `tap` of `taps`
    /// `[rows, count, 2304]`.
    pub fn copy_tap(
        &self,
        stream: &CudaStream,
        src: &impl DeviceIn<bf16>,
        taps: &mut impl DeviceOut<bf16>,
        tap: usize,
        count: usize,
    ) -> anyhow::Result<()> {
        let (src, mut taps) = (src.view(), taps.view_mut());
        ensure!(
            tap < count && count <= MAX_TAPS,
            "tap {tap} of {count}, at most {MAX_TAPS}"
        );
        ensure!(
            src.len().is_multiple_of(HIDDEN) && !src.is_empty(),
            "src has {} elements, not rows of {HIDDEN}",
            src.len()
        );
        let rows = src.len() / HIDDEN;
        check_len("taps", taps.len(), rows * count * HIDDEN)?;
        let tap = i32::try_from(tap)?;
        let count = i32::try_from(count)?;
        let config = LaunchConfig {
            grid_dim: (u32::try_from(rows).context("too many rows")?, 1, 1),
            block_dim: (u32::try_from(HIDDEN / 8)?, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launch = stream.launch_builder(&self.copy_tap);
        launch.arg(&src).arg(&mut taps).arg(&tap).arg(&count);
        // SAFETY: the arguments match `copy_tap(const bf16*, bf16*, int, int)`; block `r`
        // reads row `r` of `src` and writes row `r`, tap `tap` of `taps`, both checked to
        // exist.
        unsafe { launch.launch(config) }?;
        Ok(())
    }

    /// Copies every visit's four conv states into `backup` `[visits, 3 * 6656]`. `states`
    /// holds the device addresses of the states, F32 `[3, dims]` each, 4 per visit in the
    /// order of [`STREAM_DIMS`].
    pub fn conv_backup(
        &self,
        stream: &CudaStream,
        states: &CudaSlice<u64>,
        backup: &mut CudaSlice<f32>,
    ) -> anyhow::Result<()> {
        let visits = visits(states)?;
        check_len("backup", backup.len(), visits * STATE_ROWS * VISIT_CHANNELS)?;
        let mut launch = stream.launch_builder(&self.conv_backup);
        launch.arg(states).arg(backup);
        // SAFETY: the arguments match `conv_backup(const unsigned long long*, float*)`. The
        // caller vouches that each address in `states` is a live F32 [3, dims] buffer of the
        // stream's dims; block (x, s, v) copies stream `s` of visit `v` into its place in
        // `backup`, which holds every visit.
        unsafe { launch.launch(state_config(visits)?) }?;
        Ok(())
    }

    /// Sets every conv state to rows `keep .. keep + 3` of its backup followed by its
    /// recorded inputs, where `keep` is `result[0] + 1`. `inputs` is `[visits, 8, 6656]`.
    pub fn conv_rollback(
        &self,
        stream: &CudaStream,
        states: &CudaSlice<u64>,
        backup: &CudaSlice<f32>,
        inputs: &CudaSlice<bf16>,
        result: &CudaSlice<i32>,
    ) -> anyhow::Result<()> {
        let visits = visits(states)?;
        check_len("backup", backup.len(), visits * STATE_ROWS * VISIT_CHANNELS)?;
        check_len(
            "inputs",
            inputs.len(),
            visits * VERIFY_ROWS * VISIT_CHANNELS,
        )?;
        check_len("result", result.len(), RESULT_LEN)?;
        let mut launch = stream.launch_builder(&self.conv_rollback);
        launch.arg(states).arg(backup).arg(inputs).arg(result);
        // SAFETY: the arguments match `conv_rollback(const unsigned long long*, const float*,
        // const bf16*, const int*)`. The caller vouches for the addresses in `states`, as in
        // `conv_backup`, and that `result[0]` is below the recorded rows, which the
        // acceptance kernel guarantees; the kernel then reads backup rows below 3 and input
        // rows below 8.
        unsafe { launch.launch(state_config(visits)?) }?;
        Ok(())
    }

    /// Saves the slots that rows `0..rows` of a verify at `start` will overwrite in each
    /// sliding ring: slot `(start + j) % slots` of the BF16 keys and values whose addresses
    /// `rings` holds, a keys and a values address per visit. `backup` is
    /// `[visits, 8, 2, 1024]`.
    pub fn ring_backup(
        &self,
        stream: &CudaStream,
        rings: &CudaSlice<u64>,
        slots: usize,
        start: &CudaSlice<u32>,
        rows: usize,
        backup: &mut CudaSlice<bf16>,
    ) -> anyhow::Result<()> {
        let (config, slots) = ring_config(rings, slots, start, rows, backup.len())?;
        let mut launch = stream.launch_builder(&self.ring_backup);
        launch.arg(rings).arg(&slots).arg(start).arg(backup);
        // SAFETY: the arguments match `ring_backup(const unsigned long long*, unsigned, const
        // unsigned*, bf16*)`. The caller vouches that `rings` holds the addresses of BF16
        // rings of `slots` rows of 1024; block (j, v) reads one slot below `slots` of visit
        // `v` and writes row `j` of visit `v` in `backup`, which holds 8 rows per visit.
        unsafe { launch.launch(config) }?;
        Ok(())
    }

    /// Puts back the slots of the rows after `result[0]` from [`Self::ring_backup`]'s copy.
    #[expect(clippy::too_many_arguments, reason = "mirrors the kernel signature")]
    pub fn ring_restore(
        &self,
        stream: &CudaStream,
        rings: &CudaSlice<u64>,
        slots: usize,
        start: &CudaSlice<u32>,
        rows: usize,
        backup: &CudaSlice<bf16>,
        result: &CudaSlice<i32>,
    ) -> anyhow::Result<()> {
        let (config, slots) = ring_config(rings, slots, start, rows, backup.len())?;
        check_len("result", result.len(), RESULT_LEN)?;
        let mut launch = stream.launch_builder(&self.ring_restore);
        launch
            .arg(rings)
            .arg(&slots)
            .arg(start)
            .arg(backup)
            .arg(result);
        // SAFETY: the arguments match `ring_restore(const unsigned long long*, unsigned, const
        // unsigned*, const bf16*, const int*)`, with the same bounds as `ring_backup`.
        unsafe { launch.launch(config) }?;
        Ok(())
    }

    /// Greedy acceptance of a verify of `rows` rows: `tokens` holds the anchor and the
    /// drafts, `argmax` each row's greedy token. Keeps the longest run of drafts that match
    /// the previous row's argmax, or exactly `keep - 1` drafts when `keep` holds a nonzero
    /// value. Writes `[accepted, tokens, argmax]` to `result` and the next anchor,
    /// `argmax[accepted]`, to `anchor`.
    #[expect(clippy::too_many_arguments, reason = "mirrors the kernel signature")]
    pub fn accept_greedy(
        &self,
        stream: &CudaStream,
        tokens: &impl DeviceIn<i32>,
        argmax: &impl DeviceIn<i32>,
        keep: &CudaSlice<u32>,
        rows: usize,
        result: &mut CudaSlice<i32>,
        anchor: &mut CudaSlice<i32>,
    ) -> anyhow::Result<()> {
        let (tokens, argmax) = (tokens.view(), argmax.view());
        ensure!(
            (1..=VERIFY_ROWS).contains(&rows),
            "{rows} rows, expected 1 to {VERIFY_ROWS}"
        );
        check_len("tokens", tokens.len(), rows)?;
        check_len("argmax", argmax.len(), rows)?;
        check_len("keep", keep.len(), 1)?;
        check_len("result", result.len(), RESULT_LEN)?;
        check_len("anchor", anchor.len(), 1)?;
        let rows = i32::try_from(rows)?;
        let mut launch = stream.launch_builder(&self.accept_greedy);
        launch
            .arg(&tokens)
            .arg(&argmax)
            .arg(keep)
            .arg(&rows)
            .arg(result)
            .arg(anchor);
        // SAFETY: the arguments match `accept_greedy(const int*, const int*, const unsigned*,
        // int, int*, int*)`; the one thread reads `rows` tokens and argmax values and writes
        // at most `RESULT_LEN` results and one anchor.
        unsafe { launch.launch(one_block(1)) }?;
        Ok(())
    }
}

/// One block of `threads` threads.
fn one_block(threads: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// Visits in a table of conv state addresses, 4 per visit.
fn visits(states: &CudaSlice<u64>) -> anyhow::Result<usize> {
    ensure!(
        states.len().is_multiple_of(STREAMS) && !states.is_empty(),
        "{} state addresses, not a positive multiple of {STREAMS}",
        states.len()
    );
    Ok(states.len() / STREAMS)
}

fn state_config(visits: usize) -> anyhow::Result<LaunchConfig> {
    Ok(LaunchConfig {
        grid_dim: (
            STATE_BLOCKS,
            u32::try_from(STREAMS)?,
            u32::try_from(visits).context("too many visits")?,
        ),
        block_dim: (THREADS, 1, 1),
        shared_mem_bytes: 0,
    })
}

fn ring_config(
    rings: &CudaSlice<u64>,
    slots: usize,
    start: &CudaSlice<u32>,
    rows: usize,
    backup_len: usize,
) -> anyhow::Result<(LaunchConfig, u32)> {
    ensure!(
        rings.len().is_multiple_of(2) && !rings.is_empty(),
        "{} ring addresses, not pairs",
        rings.len()
    );
    let visits = rings.len() / 2;
    ensure!(
        (1..=VERIFY_ROWS).contains(&rows),
        "{rows} rows, expected 1 to {VERIFY_ROWS}"
    );
    check_len("start", start.len(), 1)?;
    check_len("backup", backup_len, visits * VERIFY_ROWS * 2 * KV_DIMS)?;
    let config = LaunchConfig {
        grid_dim: (u32::try_from(rows)?, u32::try_from(visits)?, 1),
        block_dim: (u32::try_from(KV_DIMS / 8)?, 1, 1),
        shared_mem_bytes: 0,
    };
    Ok((config, u32::try_from(slots).context("too many slots")?))
}
