//! The GPU the plan is for: its memory, and how much of it is free once Siphon has created
//! its CUDA context.

use std::process::Command;

use siphon::memory::CONTEXT_BYTES;

/// Memory of an RTX 5090 as nvidia-smi reports it, and what an idle one with a desktop
/// holds: 459 MiB reserved by the driver and 454 MiB in use.
const RTX_5090_TOTAL: usize = 32_607 << 20;
const RTX_5090_IDLE_HELD: usize = (459 + 454) << 20;

/// Where [`Device`]'s numbers come from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    NvidiaSmi,
    /// `--free`.
    Given,
    /// No nvidia-smi: an idle RTX 5090.
    Assumed,
}

#[derive(Clone, Debug)]
pub struct Device {
    pub name: String,
    /// Bytes on the device.
    pub total: usize,
    /// Bytes the driver reserves and other processes hold.
    pub others: usize,
    /// Bytes free once the CUDA context exists: what a plan must fit, as
    /// `siphon_cuda`'s `cuMemGetInfo` after creating the context reads it.
    pub free: usize,
    pub source: Source,
}

impl Device {
    /// The first GPU nvidia-smi lists, or with `free` given, that much free memory on a
    /// device of the same size.
    pub fn detect(free: Option<usize>) -> Self {
        let device = query().unwrap_or_else(|| Self {
            name: "RTX 5090".to_owned(),
            total: RTX_5090_TOTAL,
            others: RTX_5090_IDLE_HELD,
            free: RTX_5090_TOTAL - RTX_5090_IDLE_HELD - CONTEXT_BYTES,
            source: Source::Assumed,
        });
        match free {
            Some(free) => Self {
                others: device.total.saturating_sub(free + CONTEXT_BYTES),
                free,
                source: Source::Given,
                ..device
            },
            None => device,
        }
    }
}

/// Asks nvidia-smi for the first GPU's name and memory, in MiB.
fn query() -> Option<Device> {
    let out = Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,memory.total,memory.reserved,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8(out.stdout).ok()?;
    let mut fields = text.lines().next()?.split(", ");
    let name = fields.next()?.trim().trim_start_matches("NVIDIA GeForce ");
    let mib = |s: Option<&str>| s?.trim().parse::<usize>().ok().map(|m| m << 20);
    let total = mib(fields.next())?;
    let reserved = mib(fields.next())?;
    let used = mib(fields.next())?;
    let free = mib(fields.next())?;
    Some(Device {
        name: name.to_owned(),
        total,
        others: reserved + used,
        // CONTEXT_BYTES was measured as nvidia-smi's free memory less the free memory the
        // process reads right after creating its context.
        free: free.saturating_sub(CONTEXT_BYTES),
        source: Source::NvidiaSmi,
    })
}
