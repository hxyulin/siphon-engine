//! `siphon-plan`: how Siphon will use GPU memory, worked out from the model config and the
//! safetensors headers before anything is loaded.
//!
//! ```text
//! siphon-plan <model-dir> [--drafter DIR] [--memory GiB] [--margin GiB] [--prefill-chunk N] [--free GiB] [--context N] [--streams N] [--print]
//! ```
//!
//! It shows the plan of one configuration as a bar over the whole GPU and a table of its
//! parts, next to the longest context of every configuration. The keys switch the weight
//! format, the KV cache format and the draft, and move the context. `--print`, or output
//! that is not a terminal, prints one frame as plain text instead.

mod app;
mod device;
mod ui;

use std::{
    io::{self, IsTerminal as _, Write as _},
    path::PathBuf,
};

use anyhow::{Context as _, bail, ensure};
use ratatui::{
    buffer::Buffer,
    crossterm::event::{self, Event, KeyEventKind},
    layout::Rect,
};

use siphon::memory;

use crate::{
    app::{App, Settings},
    device::Device,
};

const USAGE: &str = "usage: siphon-plan <model-dir> [--drafter DIR] [--memory GiB] [--margin GiB] \
                     [--prefill-chunk N] [--free GiB] [--context N] [--streams N] [--print]
--drafter adds a DFlash drafter's plan beside the bundled draft's.
--memory caps the GPU memory the plan may use, the CUDA context included, as siphon-serve's
--memory does.
--margin sets the free memory the plan leaves unclaimed, as siphon-serve's --margin does
(0.5 GiB by default).
--prefill-chunk plans prefill chunks of N rows, 1 to 2048, as siphon-serve's --prefill-chunk
does (2048 by default).
--free sets the free memory right after creating the CUDA context instead of asking
nvidia-smi. Without either, the plan assumes an idle 32 GB RTX 5090.
--context starts at N positions instead of the most that fit.
--streams plans N streams, 1 to 32, as siphon-serve's --streams does: N sets of caches of
the context each, the batched step's buffers and a prefix cache entry per stream.
--print prints one frame as text and exits.";

/// Columns of a printed frame.
const PRINT_WIDTH: u16 = 104;

struct Args {
    model: PathBuf,
    drafter: Option<PathBuf>,
    memory: Option<usize>,
    margin: usize,
    prefill_chunk: usize,
    free: Option<usize>,
    context: Option<usize>,
    streams: usize,
    print: bool,
}

fn gib_arg(value: &str, flag: &str) -> anyhow::Result<usize> {
    let gib: f64 = value.parse().with_context(|| format!("{flag} takes GiB"))?;
    ensure!(gib > 0.0, "{flag} takes a positive number of GiB");
    Ok(gib_bytes(gib))
}

/// `gib` GiB, which is not negative, in bytes.
fn gib_bytes(gib: f64) -> usize {
    #[expect(clippy::cast_sign_loss, reason = "gib is not negative")]
    let bytes = (gib * f64::from(1u32 << 30)) as usize;
    bytes
}

fn args() -> anyhow::Result<Args> {
    let mut args = std::env::args().skip(1);
    let mut model = None;
    let mut drafter = None;
    let mut memory = None;
    let mut margin = memory::DEFAULT_MARGIN;
    let mut prefill_chunk = memory::DEFAULT_PREFILL_CHUNK;
    let mut free = None;
    let mut context = None;
    let mut streams = 1;
    let mut print = false;
    while let Some(arg) = args.next() {
        let mut value = || args.next().with_context(|| format!("{arg} needs a value"));
        match arg.as_str() {
            "--drafter" => drafter = Some(PathBuf::from(value()?)),
            "--memory" => memory = Some(gib_arg(&value()?, "--memory")?),
            "--margin" => {
                let gib: f64 = value()?.parse().context("--margin takes GiB")?;
                ensure!(gib >= 0.0, "--margin takes GiB, 0 or more");
                margin = gib_bytes(gib);
            }
            "--prefill-chunk" => {
                prefill_chunk = value()?.parse().context("--prefill-chunk takes rows")?;
                ensure!(
                    (1..=memory::MAX_PREFILL_CHUNK).contains(&prefill_chunk),
                    "--prefill-chunk takes 1 to {}",
                    memory::MAX_PREFILL_CHUNK
                );
            }
            "--free" => free = Some(gib_arg(&value()?, "--free")?),
            "--context" => context = Some(value()?.parse().context("--context takes positions")?),
            "--streams" => {
                streams = value()?.parse().context("--streams takes a count")?;
                ensure!(
                    (1..=memory::BATCH_ROWS).contains(&streams),
                    "--streams takes 1 to {}",
                    memory::BATCH_ROWS
                );
            }
            "--print" => print = true,
            "-h" | "--help" => {
                println!("{USAGE}");
                std::process::exit(0);
            }
            _ if arg.starts_with("--") => bail!("unknown flag {arg}\n{USAGE}"),
            _ if model.is_none() => model = Some(PathBuf::from(arg)),
            _ => bail!("unexpected argument {arg}\n{USAGE}"),
        }
    }
    Ok(Args {
        model: model.context(USAGE)?,
        drafter,
        memory,
        margin,
        prefill_chunk,
        free,
        context,
        streams,
        print,
    })
}

fn main() -> anyhow::Result<()> {
    let args = args()?;
    let device = Device::detect(args.free);
    let mut app = App::new(
        &args.model,
        args.drafter.as_deref(),
        device,
        &Settings {
            limit: args.memory,
            margin: args.margin,
            prefill_chunk: args.prefill_chunk,
            context: args.context,
            streams: args.streams,
        },
    )?;
    if args.print || !io::stdout().is_terminal() {
        return print_frame(&app);
    }
    let mut terminal = ratatui::init();
    let result = (|| -> anyhow::Result<()> {
        loop {
            terminal.draw(|frame| ui::draw(&app, frame.area(), frame.buffer_mut()))?;
            if let Event::Key(key) = event::read()?
                && key.kind == KeyEventKind::Press
                && !app.key(key.code)
            {
                return Ok(());
            }
        }
    })();
    ratatui::restore();
    result
}

/// Renders one frame into a buffer and prints it as plain text.
fn print_frame(app: &App) -> anyhow::Result<()> {
    let area = Rect::new(0, 0, PRINT_WIDTH, ui::HEIGHT);
    let mut buffer = Buffer::empty(area);
    ui::draw(app, area, &mut buffer);
    let mut out = io::stdout().lock();
    for y in 0..area.height {
        let line: String = (0..area.width).map(|x| buffer[(x, y)].symbol()).collect();
        writeln!(out, "{}", line.trim_end())?;
    }
    Ok(())
}
