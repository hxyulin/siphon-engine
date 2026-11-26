//! Picking each row's next token from its logits (`kernels/sample.cu`): greedy, or one
//! draw from vLLM's filtered distribution, seeded by the row's seed and position.

use std::sync::Arc;

use anyhow::{Context as _, ensure};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg as _,
};
use half::bf16;

use crate::gpu::{DeviceIn, DeviceOut, KernelArgs, cubin, load_module};

/// Threads of `sample_prepare`'s blocks, and its blocks per row, one cluster:
/// `kPrepareThreads` and `kCluster` in the kernel. Ada has no clusters, so the `sm_89`
/// build runs one block per row.
const PREPARE_THREADS: u32 = 512;
const PREPARE_CLUSTER: u32 = if cfg!(sm89) { 1 } else { 8 };
/// Threads and blocks per row of `sample_race`, `kRaceBlocks` in the kernel.
const RACE_THREADS: u32 = 256;
const RACE_BLOCKS: usize = 32;
/// Most tokens per row, `kMaxTokens` in the kernel.
const MAX_TOKENS: usize = 1 << 18;
/// Words of a row's `Params` and `State` in the kernel.
const PARAM_WORDS: usize = 8;
const STATE_WORDS: usize = 4;

/// How a sequence picks its tokens, with vLLM's meaning for each field. The filters apply
/// in the order temperature, top-k, top-p, min-p.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sampling {
    /// 0 for greedy decoding; otherwise the logits are divided by it.
    pub temperature: f32,
    /// Keep the `top_k` most likely tokens; 0 keeps all.
    pub top_k: u32,
    /// Keep the most likely tokens up to this share of the probability; 1 keeps all.
    pub top_p: f32,
    /// Keep tokens at least this many times as likely as the most likely one; 0 keeps all.
    pub min_p: f32,
    /// With the position of each token, chooses the draw.
    pub seed: u64,
}

impl Sampling {
    /// Greedy decoding.
    pub const GREEDY: Self = Self {
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        min_p: 0.0,
        seed: 0,
    };

    /// Checks the fields and normalizes them as vLLM does: a temperature below 1e-5 is
    /// greedy, which ignores the filters.
    pub fn new(
        temperature: f32,
        top_k: u32,
        top_p: f32,
        min_p: f32,
        seed: u64,
    ) -> anyhow::Result<Self> {
        ensure!(
            temperature.is_finite() && temperature >= 0.0,
            "temperature must be at least 0, not {temperature}"
        );
        ensure!(
            top_p > 0.0 && top_p <= 1.0,
            "top_p must be in (0, 1], not {top_p}"
        );
        ensure!(
            (0.0..=1.0).contains(&min_p),
            "min_p must be in [0, 1], not {min_p}"
        );
        if temperature < 1e-5 {
            return Ok(Self::GREEDY);
        }
        Ok(Self {
            temperature,
            top_k,
            top_p,
            min_p,
            seed,
        })
    }

    #[must_use]
    pub fn is_greedy(&self) -> bool {
        self.temperature == 0.0
    }

    /// The kernel's `Params` of a row whose token takes position `position`.
    fn words(&self, position: usize) -> anyhow::Result<[u32; PARAM_WORDS]> {
        Ok([
            self.temperature.to_bits(),
            self.top_k.min(i32::MAX as u32),
            self.top_p.to_bits(),
            self.min_p.to_bits(),
            self.seed as u32,
            (self.seed >> 32) as u32,
            u32::try_from(position).context("position does not fit the kernels")?,
            0,
        ])
    }
}

impl Default for Sampling {
    fn default() -> Self {
        Self::GREEDY
    }
}

/// The settings and intermediates of up to `rows` rows.
pub struct SampleBuffers {
    rows: usize,
    params: CudaSlice<u32>,
    state: CudaSlice<u32>,
    scores: CudaSlice<f32>,
    idxs: CudaSlice<i32>,
    /// `sample_race`'s block counters, 0 between launches.
    counters: CudaSlice<u32>,
}

impl SampleBuffers {
    pub fn new(stream: &Arc<CudaStream>, rows: usize) -> anyhow::Result<Self> {
        Ok(Self {
            rows,
            params: stream.alloc_zeros(rows * PARAM_WORDS)?,
            state: stream.alloc_zeros(rows * STATE_WORDS)?,
            scores: stream.alloc_zeros(rows * RACE_BLOCKS)?,
            idxs: stream.alloc_zeros(rows * RACE_BLOCKS)?,
            counters: stream.alloc_zeros(rows)?,
        })
    }

    /// Uploads the settings of the first rows: each row's sampling and the position its
    /// token takes.
    pub fn upload(
        &mut self,
        stream: &Arc<CudaStream>,
        rows: &[(Sampling, usize)],
    ) -> anyhow::Result<()> {
        ensure!(
            rows.len() <= self.rows,
            "{} rows, the buffers hold {}",
            rows.len(),
            self.rows
        );
        let mut words = Vec::with_capacity(rows.len() * PARAM_WORDS);
        for (sampling, position) in rows {
            words.extend(sampling.words(*position)?);
        }
        if !words.is_empty() {
            stream.memcpy_htod(&words, &mut self.params.slice_mut(..words.len()))?;
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct Sample {
    prepare: CudaFunction,
    prepare_bf16: CudaFunction,
    race: CudaFunction,
    race_bf16: CudaFunction,
    noise: CudaFunction,
}

impl Sample {
    pub fn load(ctx: &Arc<CudaContext>) -> anyhow::Result<Self> {
        let module = load_module(ctx, "sample", cubin!("sample"))?;
        Ok(Self {
            prepare: module.load_function("sample_prepare")?,
            prepare_bf16: module.load_function("sample_prepare_bf16")?,
            race: module.load_function("sample_race")?,
            race_bf16: module.load_function("sample_race_bf16")?,
            noise: module.load_function("sample_noise")?,
        })
    }

    /// Writes `-ln E_i`, the race's noise, of tokens `0..out.len()` of the draw with `seed`
    /// for the token at `position`: what `sample_race` adds to each scaled logit. For tests
    /// of the host port, `siphon::race`.
    pub fn noise(
        &self,
        stream: &CudaStream,
        seed: u64,
        position: u32,
        out: &mut impl DeviceOut<f32>,
    ) -> anyhow::Result<()> {
        let mut out = out.view_mut();
        let n = i32::try_from(out.len()).context("too many tokens")?;
        let (seed_lo, seed_hi) = (seed as u32, (seed >> 32) as u32);
        let config = LaunchConfig::for_num_elems(n.cast_unsigned());
        let mut launch = stream.launch_builder(&self.noise);
        launch
            .arg(&seed_lo)
            .arg(&seed_hi)
            .arg(&position)
            .arg(&n)
            .arg(&mut out);
        // SAFETY: the arguments match `sample_noise(unsigned, unsigned, unsigned, int,
        // float*)`, and the kernel writes the `n` elements `out` holds.
        unsafe { launch.launch(config) }?;
        Ok(())
    }

    /// Writes each row's token of `logits` (`[rows, n]`, F32) to `out` (`[rows]`) with the
    /// settings that [`SampleBuffers::upload`] gave the first `rows` rows of `buffers`. A
    /// greedy row takes the argmax with [`crate::ops::misc::Misc::argmax_f32`]'s rule.
    pub fn run(
        &self,
        stream: &CudaStream,
        logits: &impl DeviceIn<f32>,
        n: usize,
        buffers: &mut SampleBuffers,
        out: &mut impl DeviceOut<i32>,
    ) -> anyhow::Result<()> {
        let (logits, mut out) = (logits.view(), out.view_mut());
        ensure!(n > 0, "n must be positive");
        ensure!(
            logits.len().is_multiple_of(n),
            "logits have {} elements, not a multiple of {n}",
            logits.len()
        );
        ensure!(
            n <= MAX_TOKENS,
            "rows of {n} tokens, the kernels take at most {MAX_TOKENS}"
        );
        let rows = logits.len() / n;
        ensure!(
            (1..=buffers.rows).contains(&rows),
            "{rows} rows, the buffers hold {}",
            buffers.rows
        );
        ensure!(
            out.len() == rows,
            "out has {} elements, not {rows}",
            out.len()
        );
        let n = i32::try_from(n).context("n too large")?;
        let grid_rows = u32::try_from(rows)?;
        let config = LaunchConfig {
            grid_dim: (grid_rows * PREPARE_CLUSTER, 1, 1),
            block_dim: (PREPARE_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut args = KernelArgs::default();
        args.ptr(&logits, stream)
            .i32(n)
            .ptr(&buffers.params, stream)
            .ptr_mut(&mut buffers.state, stream)
            .ptr_mut(&mut out, stream);
        // SAFETY: the arguments match `sample_prepare(const float*, int, const Params*,
        // State*, int*)`: `logits` holds `rows` rows of `n`, the buffers hold `rows` rows of
        // 8 and 4 words, `out` holds `rows` elements, and the kernel waits at `pdl_wait`
        // before any global load.
        unsafe { args.launch_pdl(stream, &self.prepare, config) }?;
        let config = LaunchConfig {
            grid_dim: (u32::try_from(RACE_BLOCKS)?, grid_rows, 1),
            block_dim: (RACE_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut args = KernelArgs::default();
        args.ptr(&logits, stream)
            .i32(n)
            .ptr(&buffers.params, stream)
            .ptr(&buffers.state, stream)
            .ptr_mut(&mut buffers.scores, stream)
            .ptr_mut(&mut buffers.idxs, stream)
            .ptr_mut(&mut buffers.counters, stream)
            .ptr_mut(&mut out, stream);
        // SAFETY: the arguments match `sample_race(const float*, int, const Params*, const
        // State*, float*, int*, unsigned*, int*)`; the partials hold `RACE_BLOCKS` per row,
        // the counters one per row (all zero between launches), and the kernel reads nothing
        // before `pdl_wait`.
        unsafe { args.launch_pdl(stream, &self.race, config) }
    }

    /// Writes each row's token of BF16 `logits` (`[rows, stride]`, reading the first `n`
    /// columns with 8-wide `uint4` loads and applying `softcap` inline on non-greedy rows)
    /// to `out` (`[rows]`) with the settings in `buffers`. Eliminates the separate
    /// `logit_softcap` pass and F32 `head_capped` buffer.
    #[expect(clippy::too_many_arguments, reason = "logits, shape, softcap, buffers and output")]
    pub fn run_bf16(
        &self,
        stream: &CudaStream,
        logits: &impl DeviceIn<bf16>,
        n: usize,
        stride: usize,
        softcap: f32,
        buffers: &mut SampleBuffers,
        out: &mut impl DeviceOut<i32>,
    ) -> anyhow::Result<()> {
        let (logits, mut out) = (logits.view(), out.view_mut());
        ensure!(n > 0, "n must be positive");
        ensure!(stride >= n, "stride {stride} < n {n}");
        ensure!(
            logits.len().is_multiple_of(stride),
            "logits have {} elements, not a multiple of stride {stride}",
            logits.len()
        );
        ensure!(
            n <= MAX_TOKENS,
            "rows of {n} tokens, the kernels take at most {MAX_TOKENS}"
        );
        ensure!(
            softcap.is_finite() && softcap >= 0.0,
            "softcap must be finite and non-negative, not {softcap}"
        );
        let rows = logits.len() / stride;
        ensure!(
            (1..=buffers.rows).contains(&rows),
            "{rows} rows, the buffers hold {}",
            buffers.rows
        );
        ensure!(
            out.len() == rows,
            "out has {} elements, not {rows}",
            out.len()
        );
        let n_arg = i32::try_from(n).context("n too large")?;
        let stride_arg = i32::try_from(stride).context("stride too large")?;
        let grid_rows = u32::try_from(rows)?;
        let config = LaunchConfig {
            grid_dim: (grid_rows * PREPARE_CLUSTER, 1, 1),
            block_dim: (PREPARE_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut args = KernelArgs::default();
        args.ptr(&logits, stream)
            .i32(n_arg)
            .i32(stride_arg)
            .f32(softcap)
            .ptr(&buffers.params, stream)
            .ptr_mut(&mut buffers.state, stream)
            .ptr_mut(&mut out, stream);
        // SAFETY: the arguments match `sample_prepare_bf16(const bf16*, int, int, float,
        // const Params*, State*, int*)`; `logits` holds `rows` rows of `stride >= n`, the
        // buffers hold `rows` entries, and the kernel waits at `pdl_wait` before any load.
        unsafe { args.launch_pdl(stream, &self.prepare_bf16, config) }?;
        let config = LaunchConfig {
            grid_dim: (u32::try_from(RACE_BLOCKS)?, grid_rows, 1),
            block_dim: (RACE_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut args = KernelArgs::default();
        args.ptr(&logits, stream)
            .i32(n_arg)
            .i32(stride_arg)
            .f32(softcap)
            .ptr(&buffers.params, stream)
            .ptr(&buffers.state, stream)
            .ptr_mut(&mut buffers.scores, stream)
            .ptr_mut(&mut buffers.idxs, stream)
            .ptr_mut(&mut buffers.counters, stream)
            .ptr_mut(&mut out, stream);
        // SAFETY: the arguments match `sample_race_bf16(const bf16*, int, int, float, const
        // Params*, const State*, float*, int*, unsigned*, int*)`; the partials hold
        // `RACE_BLOCKS` per row, counters are zero between launches, and the kernel reads
        // nothing before `pdl_wait`.
        unsafe { args.launch_pdl(stream, &self.race_bf16, config) }
    }
}
