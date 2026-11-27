//! `siphon plan`: how Siphon will use GPU memory, worked out from the model config and the
//! safetensors headers before anything is loaded.
//!
mod app;
mod device;
mod ui;

use std::{
    borrow::Cow,
    fmt::Write as _,
    io::{self, IsTerminal as _, Write as _},
    path::{Path, PathBuf},
};

use anyhow::{Context as _, ensure};
use clap::{
    ArgMatches, Command, CommandFactory, FromArgMatches, Parser, ValueHint,
    builder::RangedU64ValueParser,
};
use ratatui::{
    backend::IntoCrossterm as _,
    buffer::Buffer,
    crossterm::{
        event::{self, Event, KeyCode, KeyEventKind, KeyModifiers},
        queue,
        style::{Attribute, Print, SetAttribute, SetBackgroundColor, SetForegroundColor},
        terminal,
    },
    layout::Rect,
    style::Modifier,
};

use siphon::{
    KvFormat, VisionFormat, WeightFormat,
    memory::{self, VisionInputs},
};

use crate::plan::{
    app::{App, Settings},
    device::Device,
};

/// Columns of a printed frame.
const PRINT_WIDTH: u16 = ui::SIDE_BY_SIDE_WIDTH;

/// Preview GPU memory use from the model configuration and checkpoint headers.
///
/// Prints one plan by default. Use --interactive to explore and export a server command.
#[derive(Parser)]
#[command(name = "plan", args_override_self = true)]
struct Args {
    /// Directory containing the model configuration and weights.
    #[arg(value_name = "MODEL_DIR", value_hint = ValueHint::DirPath)]
    model: PathBuf,
    /// Plan for an external DFlash drafter alongside the bundled draft.
    #[arg(long, value_name = "DIR", value_hint = ValueHint::DirPath)]
    drafter: Option<PathBuf>,
    /// Set the dense weight format (fp8 or bf16).
    #[arg(long, value_name = "FORMAT", default_value_t = WeightFormat::Fp8)]
    weights: WeightFormat,
    /// Set the KV cache format (fp8, bf16, nvfp4, nvfp4-all, mixed).
    #[arg(long, value_name = "FORMAT", default_value_t = KvFormat::Fp8)]
    kv: KvFormat,
    /// Cap GPU memory use, including the CUDA context.
    #[arg(long, value_name = "GIB", value_parser = gib_bytes::<false>, allow_negative_numbers = true)]
    memory: Option<usize>,
    /// Leave this much GPU memory free. Defaults to 0.25 GiB.
    #[arg(long, value_name = "GIB", value_parser = gib_bytes::<true>, allow_negative_numbers = true)]
    margin: Option<usize>,
    /// Set the number of rows in each prefill chunk.
    #[arg(long, value_name = "ROWS", default_value_t = memory::DEFAULT_PREFILL_CHUNK,
          value_parser = RangedU64ValueParser::<usize>::new().range(1..=memory::MAX_PREFILL_CHUNK as u64))]
    prefill_chunk: usize,
    /// Set the attention output projection format under FP8 weights (bf16, fp8, mxfp8).
    #[arg(long, value_name = "FORMAT", default_value_t)]
    output_projection: siphon::OutputProjection,
    /// Set the LM head format (bf16, fp8, nvfp4). Defaults to the weight format.
    #[arg(long, value_name = "FORMAT")]
    lm_head: Option<siphon::LmHeadFormat>,
    /// Set the dense prefill format under FP8 weights.
    #[arg(long, value_name = "FORMAT", default_value = "bf16", value_parser = ["bf16", "fp8"])]
    prefill_dense: String,
    /// Set NVFP4 expert block scale format (8bit or 4bit).
    #[arg(long, value_name = "FORMAT", default_value_t)]
    nvfp4_scales: siphon::Nvfp4ScaleFormat,
    /// Set MXFP8 expert layer storage format (mxfp8 or mxfp6).
    #[arg(long, value_name = "FORMAT", default_value_t)]
    mxfp8_experts: siphon::Mxfp8ExpertFormat,
    /// Set the draft weight format.
    #[arg(long, value_name = "FORMAT", default_value = "auto",
          value_parser = ["auto", "bf16", "fp8", "nvfp4", "nvfp4-backbone"])]
    draft_weights: String,
    /// Override GPU memory free after CUDA context creation.
    ///
    /// Without this value, the planner probes with nvidia-smi or assumes an idle 32 GB RTX 5090.
    #[arg(long, value_name = "GIB", value_parser = gib_bytes::<false>, allow_negative_numbers = true)]
    free: Option<usize>,
    /// Start at this many KV cache positions. Defaults to the most that fit.
    #[arg(long, value_name = "POSITIONS")]
    context: Option<usize>,
    /// Plan for this many concurrent streams sharing the KV cache pool.
    #[arg(long, value_name = "COUNT", default_value_t = 1,
          value_parser = RangedU64ValueParser::<usize>::new().range(1..=memory::BATCH_ROWS as u64))]
    streams: usize,
    /// Include the vision tower and its buffers.
    #[arg(long)]
    vision: bool,
    /// Set the largest image size to plan for with --vision.
    #[arg(long, value_name = "PIXELS", default_value_t = siphon::vision::DEFAULT_MAX_PIXELS,
          value_parser = RangedU64ValueParser::<usize>::new().range(siphon::vision::MIN_IMAGE_PIXELS as u64..))]
    max_image_pixels: usize,
    /// Open the planner and print a server command for the selection on exit.
    #[arg(long, short)]
    interactive: bool,
}

impl Args {
    fn load_plan(&self) -> anyhow::Result<App> {
        let device = Device::detect(self.free)?;
        let vision = if self.vision {
            let config = siphon::vision::VisionConfig::load(&self.model)?;
            Some(VisionInputs {
                format: VisionFormat::Bf16,
                max_image_tokens: memory::max_image_tokens(
                    config.token_size(),
                    self.max_image_pixels,
                ),
                patches_per_token: config.merge_size * config.merge_size,
            })
        } else {
            None
        };
        App::new(
            &self.model,
            self.drafter.as_deref(),
            device,
            &Settings {
                weights: self.weights,
                kv: self.kv,
                limit: self.memory,
                margin: self.margin.unwrap_or(memory::DEFAULT_MARGIN),
                prefill_chunk: self.prefill_chunk,
                context: self.context,
                streams: self.streams,
                vision,
                output: self.output_projection,
                lm_head: self.lm_head,
                prefill_dense_fp8: self.prefill_dense == "fp8",
                nvfp4_scales: self.nvfp4_scales,
                mxfp8_experts: self.mxfp8_experts,
                draft_weights: self.draft_weights.clone(),
            },
        )
    }
}

/// Parses a GiB argument into bytes. Only the margin can be zero.
fn gib_bytes<const ALLOW_ZERO: bool>(value: &str) -> anyhow::Result<usize> {
    let bytes = memory::parse_gib(value)?;
    ensure!(ALLOW_ZERO || bytes > 0, "expected at least one byte");
    Ok(bytes)
}

/// Defines the GPU memory planner subcommand.
#[must_use]
pub fn command() -> Command {
    Args::command()
}

/// Displays the memory plan for arguments parsed by [`command`].
///
/// # Errors
///
/// Returns an error if loading the checkpoint, planning its memory, or displaying the plan fails.
pub fn run(mut matches: ArgMatches) -> anyhow::Result<()> {
    let args = Args::from_arg_matches_mut(&mut matches)?;
    ensure!(
        !args.interactive || (io::stdin().is_terminal() && io::stdout().is_terminal()),
        "--interactive requires a terminal for both input and output"
    );
    let mut app = args.load_plan()?;
    if !args.interactive {
        return print_frame(&app);
    }
    let mut terminal = ratatui::try_init()?;
    let result = (|| -> anyhow::Result<bool> {
        loop {
            terminal.draw(|frame| {
                ui::draw(
                    &app,
                    frame.area(),
                    frame.buffer_mut(),
                    ui::Mode::Interactive,
                );
            })?;
            if let Event::Key(key) = event::read()?
                && key.kind == KeyEventKind::Press
            {
                if key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL) {
                    return Ok(false);
                }
                if !app.key(key.code) {
                    return Ok(true);
                }
            }
        }
    })();
    let restored = ratatui::try_restore();
    let export = result?;
    restored?;
    if export {
        writeln!(io::stdout().lock(), "{}", serve_command(&args, &app)?)?;
    }
    Ok(())
}

fn serve_command(args: &Args, app: &App) -> anyhow::Result<String> {
    let (context, _) = app.context();
    ensure!(context > 0, "the selected plan has no context that fits");
    let planned = app.plan().total();
    ensure!(
        planned <= app.budget,
        "cannot export: {context} positions need {} GiB, above the {} GiB budget",
        memory::gib(planned),
        memory::gib(app.budget)
    );
    let config = app.config();
    let mut command = format!(
        "siphon serve --weights {} --kv {} --context {context} --streams {} --prefill-chunk {}",
        config.weights, config.kv, args.streams, args.prefill_chunk,
    );
    if let app::Draft::Dflash(dir) = &app.drafts[app.draft] {
        write!(command, " --drafter={}", shell_path(dir)?)?;
    }
    if let Some(bytes) = args.memory {
        write!(
            command,
            " --memory {}",
            bytes as f64 / f64::from(1u32 << 30)
        )?;
    }
    if let Some(bytes) = args.margin {
        write!(
            command,
            " --margin {}",
            bytes as f64 / f64::from(1u32 << 30)
        )?;
    }
    write!(
        command,
        " --output-projection {} --prefill-dense {} --draft-weights {}",
        args.output_projection, args.prefill_dense, args.draft_weights,
    )?;
    if let Some(format) = args.lm_head {
        write!(command, " --lm-head {format}")?;
    }
    if args.nvfp4_scales != siphon::Nvfp4ScaleFormat::default() {
        write!(command, " --nvfp4-scales {}", args.nvfp4_scales)?;
    }
    if args.mxfp8_experts != siphon::Mxfp8ExpertFormat::default() {
        write!(command, " --mxfp8-experts {}", args.mxfp8_experts)?;
    }
    if args.vision {
        write!(
            command,
            " --vision --max-image-pixels {}",
            args.max_image_pixels
        )?;
    }
    write!(command, " -- {}", shell_path(&args.model)?)?;
    Ok(command)
}

fn shell_path(path: &Path) -> anyhow::Result<Cow<'_, str>> {
    let path = path
        .to_str()
        .context("cannot print a shell command for a non-UTF-8 path")?;
    ensure!(
        !path.chars().any(char::is_control),
        "cannot print a shell command for a path containing control characters"
    );
    Ok(shlex::try_quote(path)?)
}

/// Prints a styled frame in a color terminal and ASCII everywhere else.
fn print_frame(app: &App) -> anyhow::Result<()> {
    let stdout = io::stdout();
    let width = if stdout.is_terminal() {
        terminal::size().map_or(PRINT_WIDTH, |(cols, _)| cols.clamp(1, PRINT_WIDTH))
    } else {
        PRINT_WIDTH
    };
    let area = Rect::new(0, 0, width, ui::height(app, width));
    let mut buffer = Buffer::empty(area);
    let mut out = anstream::AutoStream::auto(stdout.lock());
    let mode = if out.current_choice() == anstream::ColorChoice::Never {
        ui::Mode::Ascii
    } else {
        ui::Mode::Color
    };
    ui::draw(app, area, &mut buffer, mode);
    for row in buffer.content().chunks_exact(usize::from(area.width)) {
        let end = row
            .iter()
            .rposition(|cell| cell.symbol() != " ")
            .map_or(0, |index| index + 1);
        let row = &row[..end];
        if mode == ui::Mode::Ascii {
            let line: String = row
                .iter()
                .flat_map(|cell| cell.symbol().chars())
                .map(ascii_char)
                .collect();
            writeln!(out, "{line}")?;
            continue;
        }
        for run in row.chunk_by(|a, b| a.fg == b.fg && a.bg == b.bg && a.modifier == b.modifier) {
            let cell = &run[0];
            let text: String = run.iter().map(ratatui::buffer::Cell::symbol).collect();
            queue!(
                out,
                SetAttribute(Attribute::Reset),
                SetForegroundColor(cell.fg.into_crossterm()),
                SetBackgroundColor(cell.bg.into_crossterm()),
            )?;
            if cell.modifier.contains(Modifier::BOLD) {
                queue!(out, SetAttribute(Attribute::Bold))?;
            }
            queue!(out, Print(text))?;
        }
        queue!(out, SetAttribute(Attribute::Reset))?;
        writeln!(out)?;
    }
    Ok(())
}

fn ascii_char(ch: char) -> char {
    match ch {
        '╭' | '╮' | '╰' | '╯' => '+',
        '─' => '-',
        '│' => '|',
        '·' => '.',
        '×' => 'x',
        ' '..='~' => ch,
        _ => '?',
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exported_command_parses_as_server_arguments() -> anyhow::Result<()> {
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../reference/fixtures/vision");
        let temp = tempfile::tempdir()?;
        let model = temp.path().join("Inkling's model");
        std::fs::create_dir(&model)?;
        for name in [
            "config.json",
            "processor_config.json",
            "tokenizer_config.json",
        ] {
            std::fs::copy(fixture.join("config").join(name), model.join(name))?;
        }
        std::fs::copy(
            fixture.join("tiny.safetensors"),
            model.join("tiny.safetensors"),
        )?;

        let args = Args::try_parse_from([
            "plan",
            model
                .to_str()
                .context("temporary model path must be UTF-8")?,
            "--free",
            "32",
            "--weights",
            "bf16",
            "--kv",
            "nvfp4",
            "--context",
            "8192",
            "--streams",
            "2",
            "--vision",
            "--max-image-pixels",
            "1600",
            "--draft-weights",
            "bf16",
        ])?;
        let mut app = args.load_plan()?;
        assert_eq!(app.config().weights, WeightFormat::Bf16);
        assert_eq!(app.config().kv, KvFormat::Nvfp4);
        app.key(KeyCode::Char('k'));
        app.key(KeyCode::Char('w'));
        app.key(KeyCode::Right);
        let command = serve_command(&args, &app)?;
        let words = shlex::split(&command).context("export must be valid shell syntax")?;
        let matches = Command::new("siphon")
            .subcommand(crate::serve::command())
            .try_get_matches_from(&words)?;
        let serve = matches
            .subcommand_matches("serve")
            .context("missing serve subcommand")?;
        assert_eq!(serve.get_one::<PathBuf>("model_dir"), Some(&model));
        assert_eq!(serve.get_one::<usize>("context"), Some(&16_384));
        assert_eq!(serve.get_one::<usize>("max_image_pixels"), Some(&1600));
        assert!(serve.get_flag("vision"));
        assert_eq!(
            serve.get_one::<String>("weights").map(String::as_str),
            Some("fp8")
        );
        assert_eq!(
            serve.get_one::<String>("kv").map(String::as_str),
            Some("nvfp4-all")
        );

        app.budget = 0;
        let error = serve_command(&args, &app).expect_err("a plan over budget cannot be exported");
        assert!(error.to_string().contains("above the"));
        Ok(())
    }
}
