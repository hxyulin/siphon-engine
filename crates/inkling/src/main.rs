use std::{collections::BTreeMap, env, process::ExitCode};

use anyhow::bail;
use inkling::{Checkpoint, TextConfig};

const USAGE: &str = "usage: inkling inspect <model-dir> [--all]";

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("error: {err:#}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> anyhow::Result<()> {
    let args: Vec<String> = env::args().skip(1).collect();
    match args
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>()
        .as_slice()
    {
        ["inspect", dir] => inspect(dir, Listing::Grouped),
        ["inspect", dir, "--all"] => inspect(dir, Listing::All),
        _ => bail!(USAGE),
    }
}

#[derive(Clone, Copy)]
enum Listing {
    /// One row per name pattern, with layer and expert indices replaced by `*`.
    Grouped,
    All,
}

fn inspect(dir: &str, listing: Listing) -> anyhow::Result<()> {
    let config = TextConfig::load(dir)?;
    println!(
        "hidden {} | layers {} ({} visits per token) | experts {} top-{} width {} | vocab {}",
        config.hidden_size,
        config.num_hidden_layers,
        config.layer_visits(),
        config.n_routed_experts,
        config.num_experts_per_tok,
        config.intermediate_size,
        config.vocab_size,
    );

    let checkpoint = Checkpoint::open(dir)?;
    let mut total = 0;
    let mut groups: BTreeMap<(String, String), (usize, usize, Vec<usize>)> = BTreeMap::new();
    for (name, entry) in checkpoint.tensors() {
        total += entry.byte_len();
        match listing {
            Listing::All => println!(
                "{name:<72} {:<9} {:?}",
                format!("{:?}", entry.dtype),
                entry.shape
            ),
            Listing::Grouped => {
                let key = (pattern(name), format!("{:?}", entry.dtype));
                let group = groups.entry(key).or_insert((0, 0, entry.shape.clone()));
                group.0 += 1;
                group.1 += entry.byte_len();
            }
        }
    }
    if let Listing::Grouped = listing {
        println!(
            "{:<72} {:<9} {:>6} {:>10}  shape",
            "pattern", "dtype", "count", "MiB"
        );
        for ((name, dtype), (count, bytes, shape)) in &groups {
            println!(
                "{name:<72} {dtype:<9} {count:>6} {:>10.1}  {shape:?}",
                mib(*bytes)
            );
        }
    }
    println!(
        "{} tensors, {:.2} GiB",
        checkpoint.tensors().count(),
        mib(total) / 1024.0
    );
    Ok(())
}

/// Replaces each all-digit path segment with `*`.
fn pattern(name: &str) -> String {
    name.split('.')
        .map(|part| {
            if part.bytes().all(|b| b.is_ascii_digit()) {
                "*"
            } else {
                part
            }
        })
        .collect::<Vec<_>>()
        .join(".")
}

fn mib(bytes: usize) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}
