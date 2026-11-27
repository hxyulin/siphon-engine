use std::{net::SocketAddr, path::PathBuf, time::Duration};

use anyhow::{Context as _, ensure};
use clap::{
    ArgMatches, Command, FromArgMatches, Parser, ValueEnum, ValueHint,
    builder::RangedU64ValueParser, parser::ValueSource,
};
use siphon::{Mxfp8ExpertFormat, Nvfp4ScaleFormat, OutputProjection, memory};

use crate::serve::{engine, vision};

/// Granularity of --draft-shortlist and --draft-context-rows.
const DRAFT_HEAD_BUCKET: usize = 1024;
#[cfg(target_os = "linux")]
const _: () = assert!(DRAFT_HEAD_BUCKET == siphon_cuda::draft::HEAD_BUCKET);

/// Tokens per second each decoding request gets at least while another prefills.
const DECODE_FLOOR: f64 = 20.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub(super) enum Backend {
    /// CUDA on NVIDIA GPUs under Linux.
    Cuda,
    /// Baseline MLX-LM on Apple Silicon.
    Mlx,
}

impl Backend {
    fn native() -> Option<Self> {
        if cfg!(target_os = "linux") {
            Some(Self::Cuda)
        } else if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
            Some(Self::Mlx)
        } else {
            None
        }
    }

    pub(super) fn name(self) -> &'static str {
        match self {
            Self::Cuda => "CUDA",
            Self::Mlx => "MLX",
        }
    }
}

/// Serve Inkling 2 Mini over HTTP on one GPU.
#[derive(Parser)]
#[command(
    name = "serve",
    args_override_self = true,
    after_help = "Set SIPHON_API_KEY to require authentication outside /health, /version, and /metrics."
)]
pub(super) struct Cli {
    /// Directory containing the model configuration and weights.
    #[arg(value_name = "MODEL_DIR", value_hint = ValueHint::DirPath)]
    model_dir: PathBuf,

    /// Select the GPU backend. Defaults to CUDA on Linux and MLX on Apple Silicon.
    #[arg(long, value_enum, help_heading = "Server")]
    backend: Option<Backend>,

    #[command(flatten, next_help_heading = "Server")]
    server: ServerOptions,
    #[command(flatten, next_help_heading = "GPU memory and batching")]
    gpu: GpuOptions,
    #[command(flatten, next_help_heading = "Weights and attention")]
    formats: FormatOptions,
    #[command(flatten, next_help_heading = "Speculative decoding")]
    spec: SpecOptions,

    /// Disable the GPU prefix cache.
    #[arg(long, help_heading = "Prefix cache")]
    no_prefix_cache: bool,
    /// Keep this many GPU prefix entries (1 to --streams). Defaults to --streams.
    #[arg(long, value_name = "COUNT", help_heading = "Prefix cache")]
    prefix_cache_entries: Option<usize>,
    /// Expire unused prefix cache entries after this many seconds.
    #[arg(
        long,
        value_name = "SECONDS",
        default_value_t = 600,
        help_heading = "Prefix cache"
    )]
    prefix_cache_ttl: u64,
    /// Reserve page-locked host memory for cached prefixes. Zero disables it.
    #[arg(long, value_name = "GIB", default_value = "0", value_parser = gib_bytes::<true>,
          allow_negative_numbers = true, help_heading = "Prefix cache")]
    host_cache: usize,
    /// Load and save the host cache in this directory. Requires --host-cache.
    #[arg(long, value_name = "DIR", value_hint = ValueHint::DirPath, help_heading = "Prefix cache")]
    cache_dir: Option<PathBuf>,

    /// Enable image inputs.
    #[arg(long, help_heading = "Vision")]
    vision: bool,
    /// Shrink larger images to this many pixels, keeping their aspect ratio.
    #[arg(long, value_name = "PIXELS", default_value_t = siphon::vision::DEFAULT_MAX_PIXELS,
          value_parser = RangedU64ValueParser::<usize>::new().range(siphon::vision::MIN_IMAGE_PIXELS as u64..),
          help_heading = "Vision")]
    max_image_pixels: usize,
    /// Set the maximum number of images per request.
    #[arg(long, value_name = "COUNT", default_value_t = vision::DEFAULT_MAX_IMAGES,
          value_parser = RangedU64ValueParser::<usize>::new().range(1..), help_heading = "Vision")]
    max_images: usize,

    /// Round KV values through a 4-bit format with --kv bf16 for quality comparisons.
    #[arg(long, value_name = "FORMAT", value_parser = ["none", "nvfp4", "q4_0"],
          help_heading = "Experimental options")]
    kv_quant: Option<String>,
    /// Apply a Hadamard rotation before KV rounding.
    #[arg(long, help_heading = "Experimental options")]
    kv_rotate: bool,
    /// Limit KV rounding to these comma-separated visit indices.
    #[arg(long, value_name = "VISITS", help_heading = "Experimental options")]
    kv_quant_visits: Option<String>,
    /// Cap the number of active experts per token.
    #[arg(long, value_name = "COUNT", default_value_t = 8,
          value_parser = RangedU64ValueParser::<usize>::new().range(1..=8),
          help_heading = "Experimental options")]
    moe_top_k: usize,
    /// Keep at least this many experts per token. Defaults to min(4, --moe-top-k).
    #[arg(long, value_name = "COUNT", value_parser = RangedU64ValueParser::<usize>::new().range(1..=8),
          help_heading = "Experimental options")]
    moe_min_k: Option<usize>,
    /// Drop experts whose share of the top-8 score sum falls below this threshold (0 to 1).
    #[arg(long, value_name = "FRACTION", default_value_t = 0.0, value_parser = moe_prune,
          allow_negative_numbers = true, help_heading = "Experimental options")]
    moe_prune: f32,
}

#[derive(clap::Args)]
struct ServerOptions {
    /// IP address to listen on (use brackets for IPv6).
    #[arg(long, value_name = "IP", default_value = "127.0.0.1")]
    host: String,
    /// TCP port to listen on.
    #[arg(long, value_name = "PORT", default_value_t = 8000)]
    port: u16,
    /// Model ID reported by the API.
    #[arg(long, default_value = "inkling-2-mini")]
    name: String,
    /// Allow this many requests to wait once all streams are busy.
    #[arg(long, value_name = "COUNT", default_value_t = 8)]
    queue: usize,
    /// Give running and queued requests this long to finish after shutdown starts.
    #[arg(long, value_name = "SECONDS", default_value_t = 600)]
    drain_timeout: u64,
    /// Send keep-alive bytes for silent non-streaming replies after this long. Zero disables it.
    #[arg(long, value_name = "SECONDS", default_value_t = 20)]
    keepalive_after: u64,
}

#[derive(clap::Args)]
struct GpuOptions {
    /// Set the CUDA KV pool size or the MLX per-request context limit.
    /// Defaults to the largest CUDA pool that fits or 8192 positions in MLX.
    #[arg(long, value_name = "POSITIONS", value_parser = RangedU64ValueParser::<usize>::new().range(1..))]
    context: Option<usize>,
    /// Cap GPU memory use, including the CUDA context.
    #[arg(long, value_name = "GIB", value_parser = gib_bytes::<false>, allow_negative_numbers = true)]
    memory: Option<usize>,
    /// Leave this much GPU memory free. Defaults to 0.25 GiB.
    #[arg(long, value_name = "GIB", value_parser = gib_bytes::<true>, allow_negative_numbers = true)]
    margin: Option<usize>,
    /// Set prefill chunk size in rows. Defaults to 2048.
    #[arg(long, value_name = "ROWS",
          value_parser = RangedU64ValueParser::<usize>::new().range(1..=memory::MAX_PREFILL_CHUNK as u64))]
    prefill_chunk: Option<usize>,
    /// Set the maximum number of running requests sharing the KV cache pool.
    #[arg(long, value_name = "COUNT", default_value_t = 1,
          value_parser = RangedU64ValueParser::<usize>::new().range(1..=memory::BATCH_ROWS as u64))]
    streams: usize,
    /// Target this decode rate per request during prefill. Zero runs one step per chunk.
    #[arg(long, value_name = "TOKENS_PER_SEC", default_value_t = DECODE_FLOOR,
          value_parser = decode_floor, allow_negative_numbers = true)]
    decode_floor: f64,
    /// Enable piggybacking with this many prefill tokens per step. Defaults to off; zero disables it.
    #[arg(long, value_name = "TOKENS",
          value_parser = RangedU64ValueParser::<usize>::new().range(0..=memory::MAX_PREFILL_CHUNK as u64))]
    piggyback_budget: Option<usize>,
}

#[derive(clap::Args)]
struct FormatOptions {
    /// Set the dense weight format.
    #[arg(long, value_name = "FORMAT", default_value = "fp8", value_parser = ["fp8", "bf16"])]
    weights: String,
    /// Set the attention output projection format under FP8 weights (bf16, fp8, mxfp8).
    #[arg(long, value_name = "FORMAT", default_value_t)]
    output_projection: OutputProjection,
    /// Set the LM head format. Defaults to the weight format.
    #[arg(long, value_name = "FORMAT", value_parser = ["bf16", "fp8", "nvfp4"])]
    lm_head: Option<String>,
    /// Set the KV cache format. nvfp4-all compresses both global and sliding caches.
    #[arg(long, value_name = "FORMAT", default_value = "fp8",
          value_parser = ["fp8", "bf16", "nvfp4", "nvfp4-all", "mixed"])]
    kv: String,
    /// Set global prefill attention precision. auto uses FP8 past 2048 keys.
    #[arg(long, value_name = "FORMAT", default_value = "auto",
          value_parser = ["auto", "bf16", "fp8", "fp4"])]
    prefill_attention: String,
    /// Set the dense prefill format under FP8 weights.
    #[arg(long, value_name = "FORMAT", default_value = "bf16", value_parser = ["bf16", "fp8"])]
    prefill_dense: String,
    /// Set the expert activation format in prefill and wide batches.
    #[arg(long, value_name = "FORMAT", default_value = "bf16",
          value_parser = ["bf16", "quantized", "fp8"])]
    expert_activations: String,
    /// Set the NVFP4 expert block scale format (8bit or 4bit).
    #[arg(long, value_name = "FORMAT", default_value_t)]
    nvfp4_scales: Nvfp4ScaleFormat,
    /// Set the MXFP8 expert layer storage format (mxfp8 or mxfp6).
    #[arg(long, value_name = "FORMAT", default_value_t)]
    mxfp8_experts: Mxfp8ExpertFormat,
}

#[derive(clap::Args)]
#[expect(clippy::struct_excessive_bools, reason = "independent CLI switches")]
struct SpecOptions {
    /// Force greedy decoding for every request.
    #[arg(long)]
    greedy: bool,
    /// Run this DFlash drafter in place of the bundled draft.
    #[arg(long, value_name = "DIR", value_hint = ValueHint::DirPath)]
    drafter: Option<PathBuf>,
    /// Select fast, fp8, exact, or a + joined list of draft approximations.
    ///
    /// The list accepts fp8-backbone, fp8-head, fp8-fold, fp8-kv, shortlist,
    /// shortlist=N, and context=N.
    #[arg(long, value_name = "PROFILE", default_value = "fast")]
    draft: String,
    /// Set the draft weight format.
    #[arg(long, value_name = "FORMAT", default_value = "auto",
          value_parser = ["auto", "bf16", "fp8", "nvfp4", "nvfp4-backbone"])]
    draft_weights: String,
    /// Set the ranked-token budget for the draft shortlist. Defaults to 32768.
    #[arg(long, value_name = "ROWS", value_parser = draft_rows::<false>)]
    draft_shortlist: Option<usize>,
    /// Set the context-token budget for the draft shortlist. Defaults to 16384.
    #[arg(long, value_name = "ROWS", value_parser = draft_rows::<true>)]
    draft_context_rows: Option<usize>,
    /// Set the verify widths to consider, as an ascending comma-separated list.
    #[arg(long, value_name = "WIDTHS", default_value = "1,2,3,4,5,6,7,8")]
    widths: Widths,
    /// Use plain decode when speculation is slower. Results can vary across runs.
    #[arg(long)]
    fallback: bool,
    /// Enable entropy-gated two-branch speculation with DFlash 2.
    #[arg(long)]
    spec_tree: bool,
    /// Accept draft tokens within this logit margin. A positive value conflicts with --spec-beta.
    #[arg(long, value_name = "MARGIN", default_value_t = 0.0, value_parser = accept_margin,
          allow_negative_numbers = true)]
    accept_margin: f32,
    /// Use rejection sampling with this beta (0 < beta <= 1). A beta of 1 is lossless.
    #[arg(long, value_name = "BETA", value_parser = spec_beta, allow_negative_numbers = true)]
    spec_beta: Option<f32>,
    /// Scale the DFlash 2 draft temperature by this factor.
    #[arg(long, value_name = "FACTOR", default_value_t = 0.85, value_parser = draft_temp_scale,
          allow_negative_numbers = true)]
    draft_temp_scale: f32,
    /// Split batched DFlash 2 verification by draft confidence.
    #[arg(long)]
    group_by_confidence: bool,
}

pub(super) struct Args {
    pub(super) model_dir: PathBuf,
    pub(super) backend: Backend,
    pub(super) address: SocketAddr,
    pub(super) name: String,
    pub(super) engine: engine::Options,
    pub(super) drain_timeout: Duration,
    pub(super) keepalive_after: Duration,
    pub(super) max_images: usize,
}

impl Args {
    pub(super) fn from_matches(
        command: &mut Command,
        matches: &ArgMatches,
    ) -> Result<Self, clap::Error> {
        Self::parse(command, matches, Backend::native())
    }

    fn parse(
        command: &mut Command,
        matches: &ArgMatches,
        native: Option<Backend>,
    ) -> Result<Self, clap::Error> {
        let cli = Cli::from_arg_matches(matches)?;
        let backend = cli.backend.or(native);
        let Some(backend) = backend else {
            return Err(command.error(
                clap::error::ErrorKind::InvalidValue,
                "serving requires Linux with CUDA or Apple Silicon macOS with MLX",
            ));
        };
        if Some(backend) != native {
            let platform = match backend {
                Backend::Cuda => "Linux with an NVIDIA GPU",
                Backend::Mlx => "Apple Silicon macOS",
            };
            return Err(command.error(
                clap::error::ErrorKind::InvalidValue,
                format!("the {} backend requires {platform}", backend.name()),
            ));
        }
        if backend == Backend::Mlx {
            let cuda_only = command.get_arguments().find_map(|arg| {
                let long = arg.get_long()?;
                let supported = matches!(
                    long,
                    "backend"
                        | "host"
                        | "port"
                        | "name"
                        | "queue"
                        | "drain-timeout"
                        | "keepalive-after"
                        | "context"
                        | "prefill-chunk"
                        | "greedy"
                );
                (!supported
                    && matches.value_source(arg.get_id().as_str())
                        == Some(ValueSource::CommandLine))
                .then(|| long.to_owned())
            });
            if let Some(flag) = cuda_only {
                return Err(command.error(
                    clap::error::ErrorKind::InvalidValue,
                    format!("--{flag} is only supported by the CUDA backend"),
                ));
            }
        }
        cli.into_args(backend)
            .map_err(|err| command.error(clap::error::ErrorKind::InvalidValue, format!("{err:#}")))
    }
}

impl Cli {
    fn into_args(self, backend: Backend) -> anyhow::Result<Args> {
        let Self {
            model_dir,
            server,
            gpu,
            formats,
            spec,
            ..
        } = self;
        let host = &server.host;
        let port = server.port;
        let address = format!("{host}:{port}").parse().with_context(|| {
            format!("invalid --host {host:?} with --port {port}: expected an IP address")
        })?;

        let prefix_entries = if self.no_prefix_cache {
            0
        } else {
            let entries = self.prefix_cache_entries.unwrap_or(gpu.streams);
            ensure!(
                (1..=gpu.streams).contains(&entries),
                "--prefix-cache-entries must be between 1 and --streams ({})",
                gpu.streams
            );
            entries
        };
        let moe_min_k = self.moe_min_k.unwrap_or_else(|| 4.min(self.moe_top_k));
        ensure!(
            moe_min_k <= self.moe_top_k,
            "--moe-min-k must be at most --moe-top-k ({})",
            self.moe_top_k
        );
        ensure!(
            spec.accept_margin == 0.0 || spec.spec_beta.is_none(),
            "--accept-margin and --spec-beta cannot both be enabled"
        );

        let mut kv_args = Vec::new();
        if let Some(quant) = self.kv_quant {
            kv_args.extend(["--kv-quant".into(), quant]);
        }
        if self.kv_rotate {
            kv_args.push("--kv-rotate".into());
        }
        if let Some(visits) = self.kv_quant_visits {
            kv_args.extend(["--kv-quant-visits".into(), visits]);
        }

        Ok(Args {
            backend,
            engine: engine::Options {
                model_dir: model_dir.clone(),
                capacity: gpu.context,
                memory: gpu.memory,
                margin: gpu.margin,
                prefill_chunk: gpu.prefill_chunk,
                fp8_weights: formats.weights == "fp8",
                output_projection: formats.output_projection,
                lm_head: formats.lm_head,
                kv: formats.kv,
                widths: spec.widths.0,
                queue: server.queue,
                streams: gpu.streams,
                drafter: spec.drafter,
                draft: spec.draft,
                draft_weights: spec.draft_weights,
                draft_shortlist: spec.draft_shortlist,
                draft_context_rows: spec.draft_context_rows,
                fallback: spec.fallback,
                spec_tree: spec.spec_tree,
                prefix_entries,
                prefix_ttl: Duration::from_secs(self.prefix_cache_ttl),
                host_cache: self.host_cache,
                cache_dir: self.cache_dir,
                prefill_attention: formats.prefill_attention,
                prefill_dense: formats.prefill_dense,
                expert_activations: formats.expert_activations,
                nvfp4_scales: formats.nvfp4_scales,
                mxfp8_experts: formats.mxfp8_experts,
                greedy: spec.greedy,
                decode_floor: gpu.decode_floor,
                piggyback_budget: gpu.piggyback_budget.filter(|&budget| budget > 0),
                vision: self.vision.then_some(engine::VisionOptions {
                    max_pixels: self.max_image_pixels,
                }),
                accept_margin: spec.accept_margin,
                spec_beta: spec.spec_beta,
                draft_temp_scale: spec.draft_temp_scale,
                group_by_confidence: spec.group_by_confidence,
                kv_args,
                moe_top_k: self.moe_top_k,
                moe_min_k,
                moe_prune: self.moe_prune,
            },
            max_images: self.max_images,
            model_dir,
            address,
            name: server.name,
            drain_timeout: Duration::from_secs(server.drain_timeout),
            keepalive_after: Duration::from_secs(server.keepalive_after),
        })
    }
}

#[derive(Clone)]
struct Widths(Vec<usize>);

impl std::str::FromStr for Widths {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> anyhow::Result<Self> {
        let widths = value
            .split(',')
            .map(str::parse)
            .collect::<Result<Vec<_>, _>>()
            .context("expected comma-separated widths")?;
        memory::validate_verify_widths(&widths)?;
        Ok(Self(widths))
    }
}

/// Parses a GiB argument into bytes. A zero limit is only valid for margin and host cache.
fn gib_bytes<const ALLOW_ZERO: bool>(value: &str) -> anyhow::Result<usize> {
    let bytes = memory::parse_gib(value)?;
    ensure!(ALLOW_ZERO || bytes > 0, "expected at least one byte");
    Ok(bytes)
}

fn draft_rows<const CONTEXT: bool>(value: &str) -> anyhow::Result<usize> {
    let rows: usize = value.parse().context("expected a row count")?;
    let range = if CONTEXT {
        0..=memory::SHORTLIST_CONTEXT_ROWS
    } else {
        DRAFT_HEAD_BUCKET..=memory::SHORTLIST_ROWS
    };
    ensure!(
        range.contains(&rows) && rows.is_multiple_of(DRAFT_HEAD_BUCKET),
        "expected a multiple of {DRAFT_HEAD_BUCKET} in {range:?}"
    );
    Ok(rows)
}

fn decode_floor(value: &str) -> anyhow::Result<f64> {
    let rate = value.parse()?;
    ensure!(rate >= 0.0, "expected 0 or more tokens per second");
    Ok(rate)
}

fn accept_margin(value: &str) -> anyhow::Result<f32> {
    let margin: f32 = value.parse()?;
    ensure!(
        margin.is_finite() && margin >= 0.0,
        "expected a finite non-negative logit margin"
    );
    Ok(margin)
}

fn spec_beta(value: &str) -> anyhow::Result<f32> {
    let beta: f32 = value.parse()?;
    ensure!(
        beta > 0.0 && beta <= 1.0,
        "expected a number greater than 0 and at most 1"
    );
    Ok(beta)
}

fn draft_temp_scale(value: &str) -> anyhow::Result<f32> {
    let scale: f32 = value.parse()?;
    ensure!(
        scale.is_finite() && scale > 0.0,
        "expected a finite positive scale"
    );
    Ok(scale)
}

fn moe_prune(value: &str) -> anyhow::Result<f32> {
    let fraction = value.parse()?;
    ensure!(
        (0.0..=1.0).contains(&fraction),
        "expected a fraction between 0 and 1"
    );
    Ok(fraction)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn try_parse_from<'a>(args: impl IntoIterator<Item = &'a str>) -> Result<Args, clap::Error> {
        parse_on(Backend::Cuda, args)
    }

    fn parse_on<'a>(
        backend: Backend,
        args: impl IntoIterator<Item = &'a str>,
    ) -> Result<Args, clap::Error> {
        let mut root = Command::new("siphon").subcommand(crate::serve::command());
        let mut matches =
            root.try_get_matches_from_mut(["siphon", "serve"].into_iter().chain(args))?;
        let (_, matches) = matches
            .remove_subcommand()
            .expect("server subcommand was provided");
        let command = root
            .find_subcommand_mut("serve")
            .expect("server subcommand was defined");
        Args::parse(command, &matches, Some(backend))
    }

    #[test]
    fn mlx_defaults_on_apple_silicon_and_accepts_shared_flags() {
        let args = parse_on(
            Backend::Mlx,
            [
                "model",
                "--context",
                "4096",
                "--prefill-chunk",
                "512",
                "--greedy",
            ],
        )
        .unwrap_or_else(|error| panic!("MLX flags were rejected: {error}"));
        assert_eq!(args.backend, Backend::Mlx);
        assert_eq!(args.engine.capacity, Some(4096));
        assert_eq!(args.engine.prefill_chunk, Some(512));
        assert!(args.engine.greedy);
    }

    #[test]
    fn mlx_rejects_cuda_flags_and_cross_platform_selection() {
        for (native, flags, message) in [
            (
                Backend::Mlx,
                ["--streams", "2"],
                "--streams is only supported by the CUDA backend",
            ),
            (
                Backend::Mlx,
                ["--backend", "cuda"],
                "CUDA backend requires Linux",
            ),
            (
                Backend::Cuda,
                ["--backend", "mlx"],
                "MLX backend requires Apple Silicon",
            ),
        ] {
            let Err(error) = parse_on(native, ["model"].into_iter().chain(flags)) else {
                panic!("{flags:?} accepted for {native:?}");
            };
            assert!(error.to_string().contains(message), "{error}");
        }
    }

    #[test]
    fn piggyback_budget_controls_whether_piggybacking_is_enabled() {
        let cases: &[(&[&str], Option<usize>)] = &[
            (&[], None),
            (&["--piggyback-budget", "512"], Some(512)),
            (&["--piggyback-budget", "0"], None),
            (
                &["--piggyback-budget", "512", "--piggyback-budget", "0"],
                None,
            ),
        ];
        for &(flags, expected) in cases {
            let args = try_parse_from(["model"].into_iter().chain(flags.iter().copied()))
                .unwrap_or_else(|err| panic!("could not parse {flags:?}: {err}"));
            assert_eq!(args.engine.piggyback_budget, expected, "flags {flags:?}");
        }
    }

    #[test]
    fn cache_entries_follow_the_last_stream_count() {
        let args = try_parse_from(["--streams", "2", "model", "--streams", "8"])
            .unwrap_or_else(|err| panic!("could not parse repeated stream counts: {err}"));
        assert_eq!(args.engine.streams, 8);
        assert_eq!(args.engine.prefix_entries, 8);
    }

    #[test]
    fn invalid_verify_widths_are_rejected_during_parsing() {
        for widths in ["99", "1", "4,2", "2,2", "0,2"] {
            let Err(error) = try_parse_from(["model", "--widths", widths]) else {
                panic!("invalid verify widths {widths:?} parsed successfully");
            };
            assert_eq!(error.kind(), clap::error::ErrorKind::ValueValidation);
            assert!(error.to_string().contains("--widths"));
        }
        let args = try_parse_from(["model", "--widths", "2,4,8"])
            .unwrap_or_else(|error| panic!("valid verify widths were rejected: {error}"));
        assert_eq!(args.engine.widths, [2, 4, 8]);
    }

    #[test]
    fn incompatible_sampling_flags_get_a_usage_error() {
        let result = try_parse_from(["model", "--accept-margin", "0.1", "--spec-beta", "1"]);
        let Err(error) = result else {
            panic!("positive --accept-margin parsed with --spec-beta");
        };
        assert_eq!(error.kind(), clap::error::ErrorKind::InvalidValue);
        let message = error.to_string();
        assert!(
            message.contains("--accept-margin and --spec-beta"),
            "{message}"
        );
        assert!(message.contains("Usage: siphon serve"), "{message}");
    }
}
