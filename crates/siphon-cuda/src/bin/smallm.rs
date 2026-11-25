//! Measurements of the multi-row kernels at the row counts of a speculative verify.
//!
//! ```sh
//! smallm prefill <model-dir> <capture> --chunk M [--tokens N] [--repeat R] [--skip S]
//! smallm dense <model-dir>
//! smallm experts <model-dir> <routing-capture>
//! smallm attention
//! smallm verify <model-dir> <capture> --width M [--passes N] [--weights bf16|fp8] [--kv bf16|fp8] [--hash 1]
//!     [--prefill-attention auto|bf16|fp8]
//! ```
//!
//! `prefill` runs real text through `Model::prefill` in chunks of a few rows, for kernel
//! traces. The capture is any `reference/capture_eval.py` or `capture.py --text` file; its
//! `tokens` are the text. Each repeat resets the model, prefills the first `--skip` tokens
//! in one chunk, then the next `--tokens` tokens in chunks of `--chunk` rows, and prints the
//! time per chunk. Run it under `nsys` to get each kernel's time at that row count.
//!
//! `dense` times one step's dense products at 1 to 8 rows with the checkpoint's weights:
//! every visit's attention input and output projections, the message gates' down
//! projections and the LM head, with the multi-row kernels (BF16 and FP8 weights), the
//! tensor-core GEMM the verify pass runs, and the decode GEMVs at one row.
//!
//! `experts` times one step's expert MLPs at 1 to 8 rows with the checkpoint's experts and
//! real routing: `reference/capture.py --text ... --routing` saves each visit's top-8 ids
//! of every position, and windows of m consecutive positions give the rows. It runs the
//! multi-row MLP, the prefill path (grouping and grouped products) and the decode MLP at
//! one row, and prints the floor that reading each distinct expert once sets.
//!
//! `attention` times one global layer's attention at 1 to 8 rows on random caches at
//! several context lengths, BF16 and FP8: the multi-row attention against one append and
//! decode attention per row, with the bytes of keys and values each reads.
//!
//! `verify` prefills the capture's text but its last 512 tokens, then runs `--passes`
//! verify passes of `--width` rows over them, keeping every row, for kernel traces of the
//! verify pass. `--hash 1` prints a hash of every pass's logits, to compare two builds bit
//! for bit. `--prefill-attention` picks the prompt prefill's attention products (FP8 by
//! default), which shape the cache the passes read.

#[cfg(target_os = "linux")]
fn main() -> anyhow::Result<()> {
    linux::main()
}

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("smallm needs the CUDA backend, which builds only on Linux");
    std::process::exit(1);
}

#[cfg(target_os = "linux")]
mod linux {
    use std::{env, fs, path::PathBuf, time::Instant};

    use anyhow::{Context as _, ensure};
    use safetensors::{Dtype, SafeTensors};
    use siphon::{Checkpoint, TextConfig};
    use siphon_cuda::{
        Gpu,
        model::{Accept, DEFAULT_PREFILL_ATTENTION, Model, PrefillLogits, VerifyTokens},
        ops::attention::KvFormat,
        weights::{WeightFormat, Weights},
    };

    struct Args {
        model: PathBuf,
        capture: PathBuf,
        chunk: usize,
        tokens: usize,
        repeat: usize,
        skip: usize,
    }

    fn args() -> anyhow::Result<Args> {
        let mut it = env::args().skip(2);
        let usage = "usage: smallm prefill <model-dir> <capture> --chunk M [--tokens N] \
                     [--repeat R] [--skip S]";
        let model = PathBuf::from(it.next().context(usage)?);
        let capture = PathBuf::from(it.next().context(usage)?);
        let mut a = Args {
            model,
            capture,
            chunk: 1,
            tokens: 64,
            repeat: 3,
            skip: 0,
        };
        while let Some(flag) = it.next() {
            let value: usize = it
                .next()
                .with_context(|| format!("{flag} needs a value"))?
                .parse()?;
            match flag.as_str() {
                "--chunk" => a.chunk = value,
                "--tokens" => a.tokens = value,
                "--repeat" => a.repeat = value,
                "--skip" => a.skip = value,
                _ => anyhow::bail!("unknown flag {flag}; {usage}"),
            }
        }
        ensure!(a.chunk > 0 && a.tokens > 0, "{usage}");
        Ok(a)
    }

    fn tokens(path: &PathBuf) -> anyhow::Result<Vec<u32>> {
        let bytes = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
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

    pub fn main() -> anyhow::Result<()> {
        match env::args().nth(1).as_deref() {
            Some("prefill") => prefill(),
            Some("dense") => dense::run(&PathBuf::from(
                env::args()
                    .nth(2)
                    .context("usage: smallm dense <model-dir>")?,
            )),
            Some("experts") => {
                let usage = "usage: smallm experts <model-dir> <routing-capture>";
                experts::run(
                    &PathBuf::from(env::args().nth(2).context(usage)?),
                    &PathBuf::from(env::args().nth(3).context(usage)?),
                )
            }
            Some("attention") => attention::run(),
            Some("verify") => verify(),
            _ => anyhow::bail!("usage: smallm prefill|dense|experts|attention ..."),
        }
    }

    fn verify() -> anyhow::Result<()> {
        let mut it = env::args().skip(2);
        let usage = "usage: smallm verify <model-dir> <capture> --width M [--passes N] \
                     [--weights bf16|fp8] [--kv bf16|fp8] [--hash 1] \
                     [--prefill-attention auto|bf16|fp8]";
        let model_dir = PathBuf::from(it.next().context(usage)?);
        let capture = PathBuf::from(it.next().context(usage)?);
        let (mut width, mut passes, mut weights, mut kv, mut hash) =
            (4usize, 20usize, WeightFormat::Bf16, KvFormat::Bf16, false);
        let mut prefill_attention = DEFAULT_PREFILL_ATTENTION;
        while let Some(flag) = it.next() {
            let value = it.next().with_context(|| format!("{flag} needs a value"))?;
            match flag.as_str() {
                "--width" => width = value.parse()?,
                "--passes" => passes = value.parse()?,
                "--weights" => weights = value.parse()?,
                "--kv" => kv = value.parse()?,
                "--hash" => hash = value == "1",
                "--prefill-attention" => prefill_attention = value.parse()?,
                _ => anyhow::bail!("unknown flag {flag}; {usage}"),
            }
        }
        let text = tokens(&capture)?;
        ensure!(
            text.len() > 512 + passes * width,
            "the capture is too short"
        );
        let (prompt, rest) = text.split_at(text.len() - 512);
        let config = TextConfig::load(&model_dir)?;
        let passes_cfg = config.loop_config.num_passes;
        let gpu = Gpu::new(0)?;
        let checkpoint = Checkpoint::open(&model_dir)?;
        let w = Weights::load(&checkpoint, &config, gpu.stream())?;
        let mut model =
            Model::with_formats(&gpu, config, w, passes_cfg, text.len() + 16, kv, weights)?;
        model.set_prefill_attention(prefill_attention);
        model.prefill(prompt, 2048, PrefillLogits::Last)?;
        // FNV-1a over the bits of every pass's logits.
        let mut fnv = 0xcbf2_9ce4_8422_2325_u64;
        for (i, rows) in rest.chunks_exact(width).take(passes).enumerate() {
            let t = Instant::now();
            model.verify(VerifyTokens::Host(rows), Accept::Rows(width))?;
            if i + 1 == passes {
                println!("last pass: {:.3} ms", t.elapsed().as_secs_f64() * 1e3);
            }
            if hash {
                for x in model.verify_logits()? {
                    fnv = (fnv ^ u64::from(x.to_bits())).wrapping_mul(0x0100_0000_01b3);
                }
            }
        }
        if hash {
            println!("logits hash: {fnv:016x}");
        }
        Ok(())
    }

    fn prefill() -> anyhow::Result<()> {
        let a = args()?;
        let text = tokens(&a.capture)?;
        ensure!(
            text.len() >= a.skip + a.tokens,
            "the capture has {} tokens",
            text.len()
        );
        let config = TextConfig::load(&a.model)?;
        let passes = config.loop_config.num_passes;
        let gpu = Gpu::new(0)?;
        let checkpoint = Checkpoint::open(&a.model)?;
        let weights = Weights::load(&checkpoint, &config, gpu.stream())?;
        let capacity = (a.skip + a.tokens).max(4096);
        let mut model =
            Model::with_kv_format(&gpu, config, weights, passes, capacity, KvFormat::Bf16)?;
        let body = &text[a.skip..a.skip + a.tokens];
        for r in 0..a.repeat {
            model.reset()?;
            if a.skip > 0 {
                model.prefill(&text[..a.skip], a.skip, PrefillLogits::Last)?;
            }
            gpu.stream().synchronize()?;
            let start = Instant::now();
            model.prefill(body, a.chunk, PrefillLogits::Last)?;
            let seconds = start.elapsed().as_secs_f64();
            let chunks = a.tokens.div_ceil(a.chunk);
            println!(
                "repeat {r}: {} tokens in chunks of {}: {:.3} ms per chunk",
                a.tokens,
                a.chunk,
                seconds * 1e3 / chunks as f64
            );
        }
        Ok(())
    }

    mod dense {
        use std::path::Path;

        use anyhow::Context as _;
        use cudarc::driver::{CudaSlice, CudaStream, sys::CUevent_flags};
        use half::bf16;
        use siphon::{Checkpoint, TextConfig};
        use siphon_cuda::{
            Gpu,
            ops::{
                linear::{Gemm, Gemv, Linear, Start},
                linear_fp8::{Fp8Gemv, LinearFp8},
                linear_rows::{LinearRows, RowsProduct},
            },
            weights::{Table, Weights},
        };

        const BANDWIDTH: f64 = 1.70e12;

        #[derive(Clone, Copy, PartialEq, Eq, Debug)]
        enum Kind {
            Inputs,
            Output,
            GateDown,
            Head,
        }

        #[derive(Clone, Copy, PartialEq, Eq, Debug)]
        enum Path_ {
            RowsBf16,
            RowsFp8,
            TiledGemm,
            GemvBf16,
            GemvFp8,
        }

        struct Kernels {
            rows: LinearRows,
            linear: Linear,
            fp8: LinearFp8,
        }

        /// The layer of each visit and whether its message gates run.
        pub(super) fn visits(config: &TextConfig) -> Vec<(usize, bool)> {
            let lc = &config.loop_config;
            let mut v: Vec<(usize, bool)> = (0..lc.core_start).map(|l| (l, false)).collect();
            for pass in 0..lc.num_passes {
                v.extend((lc.core_start..lc.core_end).map(|l| (l, pass > 0)));
            }
            v.extend((lc.core_end..config.num_hidden_layers).map(|l| (l, false)));
            v
        }

        /// The products of one launch: BF16 weight, FP8 copy and k.
        type Launch<'a> = Vec<(
            &'a CudaSlice<bf16>,
            Option<&'a siphon_cuda::ops::linear_fp8::Fp8Matrix>,
            usize,
        )>;

        /// Enqueues one step's products of `kind` over `rows` rows; returns the weight bytes.
        #[expect(clippy::too_many_arguments, reason = "the benchmark's buffers")]
        #[expect(
            clippy::too_many_lines,
            reason = "one match arm per product kind and path"
        )]
        fn enqueue(
            kern: &Kernels,
            stream: &CudaStream,
            weights: &Weights,
            visits: &[(usize, bool)],
            kind: Kind,
            path: Path_,
            rows: usize,
            input: &CudaSlice<bf16>,
            outs: &mut [CudaSlice<bf16>],
        ) -> anyhow::Result<usize> {
            let hidden = weights.embed_norm.len();
            let mut bytes = 0;
            // (BF16 weight, FP8 weight, k) of each product of one launch.
            let launches: Vec<Launch<'_>> = match kind {
                Kind::Inputs => visits
                    .iter()
                    .map(|&(layer, _)| {
                        let attn = &weights.layers[layer].attention;
                        let fp8w = weights.layers[layer].fp8.as_ref();
                        attn.input_projections()
                            .into_iter()
                            .enumerate()
                            .map(|(i, bf)| (bf, fp8w.map(|fp8w| &fp8w.inputs[i]), hidden))
                            .collect()
                    })
                    .collect(),
                Kind::Output => visits
                    .iter()
                    .map(|&(layer, _)| {
                        let attn = &weights.layers[layer].attention;
                        let fp8w = weights.layers[layer].fp8.as_ref();
                        vec![(&attn.o, fp8w.map(|fp8w| &fp8w.o), attn.o.len() / hidden)]
                    })
                    .collect(),
                Kind::GateDown => visits
                    .iter()
                    .filter(|v| v.1)
                    .flat_map(|&(layer, _)| {
                        let gates = weights.layers[layer]
                            .message_gates
                            .as_ref()
                            .expect("looped layer");
                        let one = |i: usize| {
                            (
                                &gates[i].down,
                                gates[i].fp8.as_ref().map(|fp8w| &fp8w[0]),
                                2 * hidden,
                            )
                        };
                        [vec![one(0), one(1), one(2)], vec![one(3)]]
                    })
                    .collect(),
                Kind::Head => vec![vec![(
                    weights.lm_head.slice(),
                    weights.lm_head_fp8.as_ref(),
                    hidden,
                )]],
            };
            for launch in &launches {
                let kk = launch[0].2;
                let xs = input.slice(..rows * kk);
                // Each product's output in its own buffer.
                let mut outs: Vec<_> = launch
                    .iter()
                    .zip(outs.iter_mut())
                    .map(|(&(bf, _, kk), out)| out.slice_mut(..rows * (bf.len() / kk)))
                    .collect();
                match path {
                    Path_::RowsBf16 | Path_::RowsFp8 => {
                        let mut p: Vec<RowsProduct<'_>> = launch
                            .iter()
                            .zip(outs.iter_mut())
                            .map(|(&(bf, fp8w, _), out)| {
                                if path == Path_::RowsFp8 {
                                    bytes += fp8w.expect("FP8 weights").bytes();
                                    RowsProduct::fp8(fp8w.expect("FP8 weights"), &xs, out)
                                } else {
                                    bytes += 2 * bf.len();
                                    RowsProduct::bf16(bf, &xs, out)
                                }
                            })
                            .collect();
                        kern.rows.run(stream, &mut p, rows, Start::Early)?;
                    }
                    Path_::TiledGemm => {
                        let mut p: Vec<Gemm<'_, bf16>> = launch
                            .iter()
                            .zip(outs.iter_mut())
                            .map(|(&(bf, _, _), out)| {
                                bytes += 2 * bf.len();
                                Gemm::new(bf, &xs, out)
                            })
                            .collect();
                        kern.linear.gemm_bf16_tiled(stream, &mut p, rows)?;
                    }
                    Path_::GemvBf16 => {
                        let mut p: Vec<Gemv<'_, bf16>> = launch
                            .iter()
                            .zip(outs.iter_mut())
                            .map(|(&(bf, _, _), out)| {
                                bytes += 2 * bf.len();
                                Gemv::new(bf, &xs, out)
                            })
                            .collect();
                        kern.linear.gemv_bf16(stream, &mut p, Start::Serial)?;
                    }
                    Path_::GemvFp8 => {
                        if !kern.fp8.supports(kk) {
                            // The gates' down projections run inside the decode gate kernel.
                            continue;
                        }
                        let mut p: Vec<Fp8Gemv<'_>> = launch
                            .iter()
                            .zip(outs.iter_mut())
                            .map(|(&(_, fp8w, _), out)| {
                                let fp8w = fp8w.expect("FP8 weights");
                                bytes += fp8w.bytes();
                                Fp8Gemv::new(fp8w, &xs, out)
                            })
                            .collect();
                        kern.fp8.gemv(stream, &mut p, Start::Early)?;
                    }
                }
            }
            Ok(bytes)
        }

        pub fn run(model: &Path) -> anyhow::Result<()> {
            let config = TextConfig::load(model)?;
            let gpu = Gpu::new(0)?;
            let stream = gpu.stream();
            let ctx = gpu.context();
            let checkpoint = Checkpoint::open(model)?;
            let mut weights = Weights::load(&checkpoint, &config, stream)?;
            let k = Kernels {
                rows: LinearRows::load(ctx)?,
                linear: Linear::load(ctx)?,
                fp8: LinearFp8::load(ctx)?,
            };
            weights.quantize_fp8(&k.fp8, stream)?;
            // The BF16 products time device weights: copy back the LM head, which
            // quantizing moved to host memory.
            let mut head = stream.alloc_zeros::<bf16>(weights.lm_head.slice().len())?;
            stream.memcpy_dtod(weights.lm_head.slice(), &mut head)?;
            weights.lm_head = Table::Device(head);
            let visits = visits(&config);
            let x_host: Vec<bf16> = (0..8 * 4608)
                .map(|i| bf16::from_f32(((i * 37 % 101) as f32 - 50.0) / 50.0))
                .collect();
            let x = stream.clone_htod(&x_host)?;
            let mut outs = vec![stream.alloc_zeros::<bf16>(8 * config.vocab_size)?];
            for _ in 1..8 {
                outs.push(stream.alloc_zeros::<bf16>(8 * 4096)?);
            }
            println!("| products | path | m | us per step | GB/s | share of 1.70 TB/s |");
            println!("| --- | --- | ---: | ---: | ---: | ---: |");
            for kind in [Kind::Inputs, Kind::Output, Kind::GateDown, Kind::Head] {
                for path in [
                    Path_::RowsBf16,
                    Path_::RowsFp8,
                    Path_::TiledGemm,
                    Path_::GemvBf16,
                    Path_::GemvFp8,
                ] {
                    let widths: &[usize] = match path {
                        Path_::GemvBf16 | Path_::GemvFp8 => &[1],
                        _ => &[1, 2, 3, 4, 5, 6, 7, 8],
                    };
                    for &m in widths {
                        let mut best = f64::MAX;
                        let mut bytes = 0;
                        for rep in 0..6 {
                            let start =
                                stream.record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
                            bytes = enqueue(
                                &k, stream, &weights, &visits, kind, path, m, &x, &mut outs,
                            )?;
                            let end = stream.record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
                            let ms = f64::from(start.elapsed_ms(&end)?);
                            if rep > 0 {
                                best = best.min(ms);
                            }
                        }
                        if bytes == 0 {
                            continue;
                        }
                        let us = best * 1e3;
                        let gbs = bytes as f64 / (us * 1e-6) / 1e9;
                        println!(
                            "| {kind:?} | {path:?} | {m} | {us:.1} | {gbs:.0} | {:.0}% |",
                            100.0 * gbs * 1e9 / BANDWIDTH
                        );
                    }
                }
            }
            stream.synchronize().context("finishing")?;
            Ok(())
        }
    }

    mod experts {
        use std::{fs, path::Path};

        use anyhow::{Context as _, ensure};
        use cudarc::driver::sys::CUevent_flags;
        use half::bf16;
        use safetensors::{Dtype, SafeTensors};
        use siphon::{Checkpoint, TextConfig};
        use siphon_cuda::{
            Gpu,
            ops::{
                experts::{self, ExpertScratch, ExpertWeights, Experts},
                experts_rows::ExpertsRows,
            },
            weights::{ExpertFormat, LayerWeights, Weights},
        };

        const BANDWIDTH: f64 = 1.70e12;
        /// Windows of positions averaged per row count.
        const WINDOWS: usize = 16;

        fn projections<'a>(
            config: &TextConfig,
            w: &'a LayerWeights,
        ) -> (ExpertWeights<'a>, ExpertWeights<'a>) {
            let (hidden, width) = (config.hidden_size, config.intermediate_size);
            let e = &w.experts;
            let (f13, f2) = match &e.format {
                ExpertFormat::Nvfp4 {
                    w13_global_scale,
                    w2_global_scale,
                } => (
                    experts::ExpertFormat::Nvfp4 {
                        global_scale: w13_global_scale,
                    },
                    experts::ExpertFormat::Nvfp4 {
                        global_scale: w2_global_scale,
                    },
                ),
                ExpertFormat::Mxfp8 => (experts::ExpertFormat::Mxfp8, experts::ExpertFormat::Mxfp8),
            };
            (
                ExpertWeights {
                    format: f13,
                    weight: &e.w13,
                    scales: &e.w13_scale,
                    n: 2 * width,
                    k: hidden,
                },
                ExpertWeights {
                    format: f2,
                    weight: &e.w2,
                    scales: &e.w2_scale,
                    n: hidden,
                    k: width,
                },
            )
        }

        #[expect(clippy::too_many_lines, reason = "one timed loop per kernel path")]
        pub fn run(model: &Path, capture: &Path) -> anyhow::Result<()> {
            let bytes =
                fs::read(capture).with_context(|| format!("reading {}", capture.display()))?;
            let st = SafeTensors::deserialize(&bytes)?;
            let view = st.tensor("routing")?;
            ensure!(view.dtype() == Dtype::I32, "routing is {:?}", view.dtype());
            let shape = view.shape().to_vec();
            ensure!(shape.len() == 3 && shape[2] == 8, "routing is {shape:?}");
            let (visits_n, positions) = (shape[0], shape[1]);
            let routing: Vec<i32> = view
                .data()
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| i32::from_le_bytes(*b))
                .collect();

            let config = TextConfig::load(model)?;
            let gpu = Gpu::new(0)?;
            let stream = gpu.stream();
            let ctx = gpu.context();
            let checkpoint = Checkpoint::open(model)?;
            let weights = Weights::load(&checkpoint, &config, stream)?;
            let rows_k = ExpertsRows::load(ctx)?;
            let experts_k = Experts::load(ctx)?;
            let visits = super::dense::visits(&config);
            ensure!(visits.len() == visits_n, "{visits_n} visits in the capture");
            let caps = (config.mlp_gate_softcap, config.mlp_up_softcap);
            let x = stream.clone_htod(
                &(0..8 * config.hidden_size)
                    .map(|i| bf16::from_f32(((i * 37 % 101) as f32 - 50.0) / 50.0))
                    .collect::<Vec<_>>(),
            )?;
            let mut act = stream.alloc_zeros::<bf16>(64 * config.intermediate_size)?;
            let mut out = stream.alloc_zeros::<bf16>(64 * config.hidden_size)?;
            let mut sync = stream.alloc_zeros::<u32>(65)?;
            let mut decode_sync = stream.alloc_zeros::<u32>(9)?;
            let mut scratch = ExpertScratch::new(stream, 8, config.n_routed_experts)?;
            let expert_bytes = |l: usize| -> usize {
                let e = &weights.layers[l].experts;
                (e.w13.len() + e.w13_scale.len() + e.w2.len() + e.w2_scale.len())
                    / config.n_routed_experts
            };

            println!(
                "| m | distinct experts per visit | floor us | rows kernel us | share | prefill path us | decode us |"
            );
            println!("| ---: | ---: | ---: | ---: | ---: | ---: | ---: |");
            for m in 1..=8 {
                let (mut t_rows, mut t_pre, mut t_dec, mut floor, mut distinct) =
                    (0f64, 0f64, 0f64, 0f64, 0f64);
                for wnd in 0..WINDOWS {
                    let start = wnd * (positions - 8) / WINDOWS;
                    // Every visit's ids for rows start .. start + m.
                    let ids: Vec<Vec<i32>> = (0..visits_n)
                        .map(|v| {
                            (0..m)
                                .flat_map(|r| {
                                    let p = (v * positions + start + r) * 8;
                                    routing[p..p + 8].to_vec()
                                })
                                .collect()
                        })
                        .collect();
                    let dev: Vec<_> = ids
                        .iter()
                        .map(|i| stream.clone_htod(i))
                        .collect::<Result<_, _>>()?;
                    for (v, i) in ids.iter().enumerate() {
                        let mut d = i.clone();
                        d.sort_unstable();
                        d.dedup();
                        distinct += d.len() as f64 / visits_n as f64 / WINDOWS as f64;
                        floor += (d.len() * expert_bytes(visits[v].0)) as f64
                            / BANDWIDTH
                            / WINDOWS as f64;
                    }
                    let time = |f: &mut dyn FnMut() -> anyhow::Result<()>| -> anyhow::Result<f64> {
                        let mut best = f64::MAX;
                        for rep in 0..4 {
                            let a = stream.record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
                            f()?;
                            let b = stream.record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
                            let ms = f64::from(a.elapsed_ms(&b)?);
                            if rep > 0 {
                                best = best.min(ms);
                            }
                        }
                        Ok(best * 1e3)
                    };
                    let xs = x.slice(..m * config.hidden_size);
                    t_rows += time(&mut || {
                        for (v, d) in dev.iter().enumerate() {
                            let (w13, w2) = projections(&config, &weights.layers[visits[v].0]);
                            rows_k.expert_mlp_rows(
                                stream,
                                &xs,
                                d,
                                &w13,
                                &w2,
                                &mut act.slice_mut(..m * 8 * config.intermediate_size),
                                &mut out.slice_mut(..m * 8 * config.hidden_size),
                                &mut sync,
                                caps.0,
                                caps.1,
                            )?;
                        }
                        Ok(())
                    })? / WINDOWS as f64;
                    t_pre += time(&mut || {
                        for (v, d) in dev.iter().enumerate() {
                            let (w13, w2) = projections(&config, &weights.layers[visits[v].0]);
                            experts_k.expert_group(
                                stream,
                                d,
                                config.n_routed_experts,
                                &mut scratch,
                            )?;
                            let mut a = act.slice_mut(..m * 8 * config.intermediate_size);
                            experts_k.expert_matmul_swiglu_grouped(
                                stream, &xs, &w13, &mut a, &scratch, caps.0, caps.1,
                            )?;
                            let a = act.slice(..m * 8 * config.intermediate_size);
                            experts_k.expert_matmul_grouped(
                                stream,
                                &a,
                                &w2,
                                &mut out.slice_mut(..m * 8 * config.hidden_size),
                                &scratch,
                            )?;
                        }
                        Ok(())
                    })? / WINDOWS as f64;
                    if m == 1 {
                        t_dec += time(&mut || {
                            for (v, d) in dev.iter().enumerate() {
                                let (w13, w2) = projections(&config, &weights.layers[visits[v].0]);
                                experts_k.expert_mlp(
                                    stream,
                                    &xs,
                                    d,
                                    &w13,
                                    &w2,
                                    &mut act.slice_mut(..8 * config.intermediate_size),
                                    &mut out.slice_mut(..8 * config.hidden_size),
                                    &mut decode_sync,
                                    caps.0,
                                    caps.1,
                                    None,
                                )?;
                            }
                            Ok(())
                        })? / WINDOWS as f64;
                    }
                }
                let dec = if m == 1 {
                    format!("{t_dec:.0}")
                } else {
                    String::new()
                };
                println!(
                    "| {m} | {distinct:.1} | {:.0} | {t_rows:.0} | {:.0}% | {t_pre:.0} | {dec} |",
                    floor * 1e6,
                    100.0 * floor * 1e6 / t_rows
                );
            }
            Ok(())
        }
    }

    mod attention {
        use cudarc::driver::sys::CUevent_flags;
        use half::bf16;
        use siphon_cuda::{
            Gpu,
            ops::attention::{
                Attention, AttentionScratch, HEAD_DIM, HEADS, KV_HEADS, KvCache, KvFormat,
                RowsAttentionScratch,
            },
        };

        const EXTENT: usize = 1024;

        pub fn run() -> anyhow::Result<()> {
            let gpu = Gpu::new(0)?;
            let stream = gpu.stream();
            let attention = Attention::load(gpu.context())?;
            let mut rows_scratch = RowsAttentionScratch::new(stream)?;
            let mut decode = AttentionScratch::new(stream)?;
            let kv_row = KV_HEADS * HEAD_DIM;
            let q = stream.clone_htod(&vec![bf16::from_f32(0.1); 8 * HEADS * HEAD_DIM])?;
            let k = stream.clone_htod(&vec![bf16::from_f32(0.1); 8 * kv_row])?;
            let bias = stream.clone_htod(&vec![bf16::from_f32(0.0); 8 * HEADS * EXTENT])?;
            let sinks = stream.clone_htod(&vec![bf16::from_f32(0.0); 8 * HEADS])?;
            let mut out = stream.alloc_zeros::<bf16>(8 * HEADS * HEAD_DIM)?;
            println!("| cache | context | m | rows kernel us | GB/s | per-row decode us |");
            println!("| --- | ---: | ---: | ---: | ---: | ---: |");
            for format in [KvFormat::Bf16, KvFormat::Fp8] {
                for context in [2048usize, 32768, 131_072] {
                    let mut cache = KvCache::global_with_format(stream, context + 16, format)?;
                    let elem = if format == KvFormat::Bf16 { 2 } else { 1 };
                    for m in [1usize, 2, 4, 8] {
                        let start = stream.clone_htod(&[u32::try_from(context)?])?;
                        let positions = (0..m)
                            .map(|j| Ok(stream.clone_htod(&[u32::try_from(context + j)?])?))
                            .collect::<anyhow::Result<Vec<_>>>()?;
                        let time =
                            |f: &mut dyn FnMut() -> anyhow::Result<()>| -> anyhow::Result<f64> {
                                let mut best = f64::MAX;
                                for _ in 0..5 {
                                    let a = stream
                                        .record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
                                    for _ in 0..10 {
                                        f()?;
                                    }
                                    let b = stream
                                        .record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
                                    best = best.min(f64::from(a.elapsed_ms(&b)?) * 100.0);
                                }
                                Ok(best)
                            };
                        let t_rows = time(&mut || {
                            attention.attention_rows(
                                stream,
                                &q.slice(..m * HEADS * HEAD_DIM),
                                &k.slice(..m * kv_row),
                                &k.slice(..m * kv_row),
                                &mut cache,
                                &start,
                                m,
                                &bias.slice(..m * HEADS * EXTENT),
                                &sinks.slice(..m * HEADS),
                                None,
                                &mut rows_scratch,
                                &mut out.slice_mut(..m * HEADS * HEAD_DIM),
                            )
                        })?;
                        let t_dec = time(&mut || {
                            for (j, p) in positions.iter().enumerate() {
                                attention.append(
                                    stream,
                                    &mut cache,
                                    p,
                                    &k.slice(j * kv_row..(j + 1) * kv_row),
                                    &k.slice(j * kv_row..(j + 1) * kv_row),
                                )?;
                                attention.decode_attention(
                                    stream,
                                    &q.slice(j * HEADS * HEAD_DIM..(j + 1) * HEADS * HEAD_DIM),
                                    &cache,
                                    p,
                                    &bias.slice(j * HEADS * EXTENT..(j + 1) * HEADS * EXTENT),
                                    &sinks.slice(j * HEADS..(j + 1) * HEADS),
                                    &mut decode,
                                    &mut out.slice_mut(
                                        j * HEADS * HEAD_DIM..(j + 1) * HEADS * HEAD_DIM,
                                    ),
                                )?;
                            }
                            Ok(())
                        })?;
                        let bytes = ((context + m) * kv_row * 2 * elem) as f64;
                        println!(
                            "| {format} | {context} | {m} | {t_rows:.1} | {:.0} | {t_dec:.1} |",
                            bytes / (t_rows * 1e-6) / 1e9
                        );
                    }
                }
            }
            Ok(())
        }
    }
}
