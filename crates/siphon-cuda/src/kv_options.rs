//! The KV cache option that siphon-serve and the bench, eval, parity and kvdiff tools
//! share: the 4-bit rounding that measures a 4-bit cache's quality on a BF16 cache.

use anyhow::Context as _;
use siphon::KvQuant;

use crate::{model::Model, ops::attention::KvRounding};

/// Usage text of the flags [`KvOptions::parse`] takes.
pub const USAGE: &str = "[--kv-quant none|nvfp4|q4_0] [--kv-rotate] [--kv-quant-visits V,V,...] \
                         [--kv-recent N] [--kv-keys-per-channel]";

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct KvOptions {
    pub rounding: KvRounding,
}

impl KvOptions {
    /// Takes `arg`, and its value from `rest`, if it is one of these flags; returns whether
    /// it was.
    pub fn parse(
        &mut self,
        arg: &str,
        rest: &mut impl Iterator<Item = String>,
    ) -> anyhow::Result<bool> {
        let mut value = || rest.next().with_context(|| format!("{arg} needs a value"));
        match arg {
            "--kv-quant" => self.rounding.quant = value()?.parse::<KvQuant>()?,
            "--kv-rotate" => self.rounding.rotate = true,
            "--kv-recent" => self.rounding.recent = value()?.parse()?,
            "--kv-keys-per-channel" => self.rounding.keys_per_channel = true,
            "--kv-quant-visits" => {
                let mut visits = 0u64;
                for v in value()?.split(',') {
                    let v: u32 = v.parse().context("--kv-quant-visits takes visit indices")?;
                    anyhow::ensure!(v < 64, "visit {v} is past the 64 a mask holds");
                    visits |= 1 << v;
                }
                self.rounding.visits = visits;
            }
            _ => return Ok(false),
        }
        Ok(true)
    }

    /// Sets the rounding on `model`.
    pub fn apply(&self, model: &mut Model) -> anyhow::Result<()> {
        anyhow::ensure!(
            !self.rounding.keys_per_channel || self.rounding.recent > 0,
            "--kv-keys-per-channel needs --kv-recent"
        );
        if self.rounding.quant != KvQuant::None {
            model.set_kv_rounding(self.rounding)?;
        }
        Ok(())
    }
}

impl std::fmt::Display for KvOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.rounding.quant == KvQuant::None {
            return f.write_str("no 4-bit rounding");
        }
        {
            write!(f, "rounded to {}", self.rounding.quant)?;
            if self.rounding.rotate {
                f.write_str(" after a Hadamard rotation")?;
            }
            if self.rounding.recent > 0 {
                write!(
                    f,
                    " past the newest {} positions (E4M3 before)",
                    self.rounding.recent
                )?;
            }
            if self.rounding.keys_per_channel {
                f.write_str(", keys per channel")?;
            }
            if self.rounding.visits != u64::MAX {
                write!(f, " in visits {:#x}", self.rounding.visits)?;
            }
        }
        Ok(())
    }
}
