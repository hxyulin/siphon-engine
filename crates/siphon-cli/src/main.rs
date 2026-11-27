#[cfg(feature = "archive")]
mod archive;

use std::{
    collections::BTreeMap,
    env, io,
    path::{Path, PathBuf},
    process::ExitCode,
};

use clap::{Arg, ArgAction, Command, ValueHint, value_parser};
use clap_complete::{Generator as _, Shell};
use siphon::{Checkpoint, TextConfig};

fn command() -> Command {
    Command::new("siphon")
        .bin_name("siphon")
        .version(env!("CARGO_PKG_VERSION"))
        .about("Serve, plan, and inspect Inkling 2 Mini model checkpoints")
        .subcommand_required(true)
        .arg_required_else_help(true)
        .subcommands([
            Command::new("inspect")
                .about("Show the model configuration and list checkpoint tensors")
                .args_override_self(true)
                .arg(
                    Arg::new("model_dir")
                        .value_name("MODEL_DIR")
                        .value_hint(ValueHint::DirPath)
                        .value_parser(value_parser!(PathBuf))
                        .help("Directory containing the model configuration and weights")
                        .required(true),
                )
                .arg(
                    Arg::new("all")
                        .long("all")
                        .action(ArgAction::SetTrue)
                        .help("List each tensor individually"),
                ),
            siphon_plan::command(),
            siphon_serve::command(),
            #[cfg(feature = "archive")]
            archive::command(),
            Command::new("completions")
                .about("Print a shell completion script")
                .arg(
                    Arg::new("shell")
                        .value_name("SHELL")
                        .value_parser(clap::builder::EnumValueParser::<Shell>::new())
                        .required(true),
                ),
        ])
}

fn main() -> ExitCode {
    let mut command = command();
    let (name, mut matches) = command
        .get_matches_mut()
        .remove_subcommand()
        .expect("a subcommand is required");

    let result = match name.as_str() {
        "inspect" => {
            let dir = matches
                .remove_one::<PathBuf>("model_dir")
                .expect("a model directory is required");
            let listing = if matches.get_flag("all") {
                Listing::All
            } else {
                Listing::Grouped
            };
            inspect(&dir, listing)
        }
        "plan" => siphon_plan::run(matches),
        "serve" => {
            // Include the loaded kernel code in the startup memory check.
            if env::var_os("CUDA_MODULE_LOADING").is_none() {
                // SAFETY: main has not started any threads.
                unsafe { env::set_var("CUDA_MODULE_LOADING", "EAGER") };
            }
            let subcommand = command
                .find_subcommand_mut("serve")
                .expect("serve is defined");
            return siphon_serve::run(subcommand, matches).unwrap_or_else(|error| error.exit());
        }
        #[cfg(feature = "archive")]
        "archive" => archive::run(&matches),
        "completions" => {
            let shell = matches
                .remove_one::<Shell>("shell")
                .expect("a shell is required");
            command.build();
            shell
                .try_generate(&command, &mut io::stdout().lock())
                .map_err(Into::into)
        }
        _ => unreachable!("clap accepted unknown subcommand {name}"),
    };

    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error:#}");
            ExitCode::FAILURE
        }
    }
}

#[derive(Clone, Copy)]
enum Listing {
    /// One row per name pattern, with layer and expert indices replaced by `*`.
    Grouped,
    All,
}

fn inspect(dir: &Path, listing: Listing) -> anyhow::Result<()> {
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
