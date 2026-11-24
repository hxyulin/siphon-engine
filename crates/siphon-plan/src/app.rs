//! The configurations the plan compares and the one on screen.

use std::path::{Path, PathBuf};

use anyhow::Context as _;
use ratatui::crossterm::event::KeyCode;
use siphon::{
    Checkpoint, KvFormat, TextConfig, WeightFormat,
    memory::{self, DraftApprox, DraftChoice, Plan, PlanInputs, Planner},
};

use crate::device::Device;

/// Rows of the prefill chunk siphon-serve runs (`siphon_cuda::model::DEFAULT_PREFILL_CHUNK`).
const PREFILL_CHUNK: usize = 2048;
/// Positions the arrow keys move the context by: left and right, up and down.
const SMALL_STEP: usize = 8 * 1024;
const LARGE_STEP: usize = 64 * 1024;

/// The weight and KV formats, in the order the keys cycle through them.
pub const FORMATS: [(WeightFormat, KvFormat); 4] = [
    (WeightFormat::Fp8, KvFormat::Fp8),
    (WeightFormat::Fp8, KvFormat::Bf16),
    (WeightFormat::Bf16, KvFormat::Fp8),
    (WeightFormat::Bf16, KvFormat::Bf16),
];

/// A draft the server can run, as siphon-serve's `--drafter` picks it.
#[derive(Clone, Debug)]
pub enum Draft {
    Bundled,
    Dflash(PathBuf),
    None,
}

impl Draft {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Bundled => "DSpark",
            Self::Dflash(_) => "DFlash",
            Self::None => "No draft",
        }
    }
}

/// One configuration's planner.
pub struct Config {
    pub weights: WeightFormat,
    pub kv: KvFormat,
    pub planner: Planner,
}

pub struct App {
    pub model: String,
    pub device: Device,
    /// `--memory`.
    pub limit: Option<usize>,
    /// Bytes a plan may take.
    pub budget: usize,
    pub drafts: Vec<Draft>,
    /// `configs[draft][format]`, in the order of `drafts` and [`FORMATS`].
    pub configs: Vec<Vec<Config>>,
    pub draft: usize,
    pub format: usize,
    /// Positions on screen; `None` follows the most that fit.
    context: Option<usize>,
}

impl App {
    pub fn new(
        model_dir: &Path,
        drafter: Option<&Path>,
        device: Device,
        limit: Option<usize>,
        context: Option<usize>,
    ) -> anyhow::Result<Self> {
        let config = TextConfig::load(model_dir)?;
        let checkpoint = Checkpoint::open(model_dir)?;
        let mut drafts = vec![Draft::Bundled];
        drafts.extend(drafter.map(|d| Draft::Dflash(d.to_owned())));
        drafts.push(Draft::None);
        // siphon-serve's default draft options: FP8 backbone, head and fold, and a shortlist.
        let approx = DraftApprox {
            fp8_backbone: true,
            fp8_head: true,
            fp8_fold: true,
            shortlist: true,
        };
        let configs = drafts
            .iter()
            .map(|draft| {
                FORMATS
                    .iter()
                    .map(|&(weights, kv)| {
                        let planner = Planner::new(&PlanInputs {
                            config: &config,
                            checkpoint: &checkpoint,
                            weights,
                            kv,
                            draft: match draft {
                                Draft::Bundled => DraftChoice::Bundled(approx),
                                Draft::Dflash(dir) => DraftChoice::Dflash(dir, approx),
                                Draft::None => DraftChoice::None,
                            },
                            prefill_chunk: PREFILL_CHUNK,
                            streams: 1,
                            prefix_cache: true,
                        })
                        .with_context(|| format!("planning with the {} draft", draft.name()))?;
                        Ok(Config {
                            weights,
                            kv,
                            planner,
                        })
                    })
                    .collect::<anyhow::Result<Vec<_>>>()
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        let model = model_dir
            .file_name()
            .map_or_else(|| "model".to_owned(), |n| n.to_string_lossy().into_owned());
        let budget = memory::budget(device.free, limit);
        Ok(Self {
            model,
            device,
            limit,
            budget,
            drafts,
            configs,
            draft: 0,
            format: 0,
            context,
        })
    }

    pub fn config(&self) -> &Config {
        &self.configs[self.draft][self.format]
    }

    /// The longest context that fits the budget in `config`.
    pub fn max_context(&self, config: &Config) -> Option<usize> {
        config.planner.max_capacity(self.budget)
    }

    /// The positions on screen and whether they follow the most that fit.
    pub fn context(&self) -> (usize, bool) {
        match self.context {
            Some(c) => (c, false),
            None => (self.max_context(self.config()).unwrap_or(0), true),
        }
    }

    pub fn plan(&self) -> Plan {
        self.config().planner.plan(self.context().0)
    }

    /// Handles a key; returns false to quit.
    pub fn key(&mut self, code: KeyCode) -> bool {
        let (context, _) = self.context();
        let limit = self.config().planner.max_positions();
        let step = |by: isize| {
            let moved = context.saturating_add_signed(by);
            Some(moved.clamp(memory::CAPACITY_STEP, limit))
        };
        match code {
            KeyCode::Char('q') | KeyCode::Esc => return false,
            KeyCode::Char('w') => self.format ^= 2,
            KeyCode::Char('k') => self.format ^= 1,
            KeyCode::Char('d') => self.draft = (self.draft + 1) % self.drafts.len(),
            KeyCode::Char('m') => self.context = None,
            KeyCode::Right => self.context = step(SMALL_STEP.cast_signed()),
            KeyCode::Left => self.context = step(-SMALL_STEP.cast_signed()),
            KeyCode::Up => self.context = step(LARGE_STEP.cast_signed()),
            KeyCode::Down => self.context = step(-LARGE_STEP.cast_signed()),
            _ => {}
        }
        true
    }
}
