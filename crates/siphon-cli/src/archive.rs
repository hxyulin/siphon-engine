//! An offline conversation archive. Each zstd frame is a JSONL stream with its own
//! dictionary. An item's ID is its zero-based position among the frame's item records.

use std::{
    borrow::Cow,
    collections::{BTreeMap, HashMap},
    fs::File,
    io::{self, BufRead, BufReader, BufWriter, Write},
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail, ensure};
use clap::{Arg, ArgMatches, Command, ValueHint, value_parser};
use serde::{Deserialize, Serialize};
use serde_json::value::{RawValue, to_raw_value};

const VERSION: u32 = 1;
const DICTIONARY_BYTES: usize = 64 * 1024 * 1024;
const DICTIONARY_ITEMS: usize = 65_536;
const INPUT_FIELDS: &[&str] = &[
    "instructions",
    "system",
    "messages",
    "input",
    "tools",
    "prompt",
];
const OUTPUT_FIELDS: &[&str] = &["output", "content", "choices"];

// Keep content as raw JSON so large numbers, escaped text, and nested structures survive
// the round trip without conversion. Only the maps whose fields we rewrite are parsed.
type Object = BTreeMap<String, Box<RawValue>>;

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Record<'a> {
    Header { version: u32 },
    Item(Cow<'a, RawValue>),
    Turn(Object),
    End {},
}

pub fn command() -> Command {
    Command::new("archive")
        .about("Pack and expand conversation logs")
        .subcommand_required(true)
        .arg_required_else_help(true)
        .subcommand(files(
            Command::new("pack").about("Deduplicate and compress a JSONL log"),
        ))
        .subcommand(files(
            Command::new("expand").about("Restore a conversation archive to JSONL"),
        ))
}

fn files(command: Command) -> Command {
    command
        .arg(
            Arg::new("input")
                .value_name("INPUT")
                .default_value("-")
                .value_hint(ValueHint::FilePath)
                .value_parser(value_parser!(PathBuf))
                .help("Input file, or - for stdin"),
        )
        .arg(
            Arg::new("output")
                .short('o')
                .long("output")
                .value_name("OUTPUT")
                .default_value("-")
                .value_hint(ValueHint::FilePath)
                .value_parser(value_parser!(PathBuf))
                .help("New output file, or - for stdout"),
        )
}

pub fn run(matches: &ArgMatches) -> Result<()> {
    let (name, args) = matches
        .subcommand()
        .expect("an archive subcommand is required");
    let input = args
        .get_one::<PathBuf>("input")
        .expect("input has a default");
    let output = args
        .get_one::<PathBuf>("output")
        .expect("output has a default");
    let stdin = input == Path::new("-");
    let reader: Box<dyn BufRead> = if stdin {
        Box::new(io::stdin().lock())
    } else {
        Box::new(BufReader::new(
            File::open(input).with_context(|| format!("opening {}", input.display()))?,
        ))
    };
    let transform = |writer: &mut dyn Write| {
        let result = match name {
            "pack" => pack(reader, writer, DICTIONARY_BYTES),
            "expand" => expand(reader, writer),
            _ => unreachable!("clap accepted unknown archive subcommand {name}"),
        };
        if stdin {
            result.with_context(|| format!("archive {name} from stdin"))
        } else {
            result.with_context(|| format!("archive {name} from {}", input.display()))
        }
    };
    if output == Path::new("-") {
        let mut writer = BufWriter::new(io::stdout().lock());
        transform(&mut writer)?;
        writer.flush().context("writing stdout")?;
    } else {
        // Publish the destination only after the full input was read and the output flushed.
        let dir = output
            .parent()
            .filter(|dir| !dir.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        let mut temp = tempfile::NamedTempFile::new_in(dir)
            .with_context(|| format!("creating temporary output in {}", dir.display()))?;
        {
            let mut writer = BufWriter::new(temp.as_file_mut());
            transform(&mut writer)?;
            writer
                .flush()
                .with_context(|| format!("writing {}", output.display()))?;
        }
        temp.persist_noclobber(output)
            .map_err(|error| error.error)
            .with_context(|| format!("saving {}", output.display()))?;
    }
    Ok(())
}

fn write_jsonl(output: &mut impl Write, value: &impl Serialize) -> Result<()> {
    serde_json::to_writer(&mut *output, value)?;
    output.write_all(b"\n")?;
    Ok(())
}

fn pack(input: impl BufRead, mut output: impl Write, dictionary_bytes: usize) -> Result<()> {
    let mut turns = input
        .lines()
        .enumerate()
        .filter(|(_, line)| match line {
            Ok(text) => !text.trim().is_empty(),
            Err(_) => true,
        })
        .peekable();
    loop {
        let mut frame = zstd::Encoder::new(&mut output, 3)?;
        frame.include_checksum(true)?;
        write_jsonl(&mut frame, &Record::Header { version: VERSION })?;
        let mut items = Items::default();
        for (index, line) in turns.by_ref() {
            let line = line.with_context(|| format!("reading line {}", index + 1))?;
            pack_turn(&line, &mut items, &mut frame)
                .with_context(|| format!("JSONL line {}", index + 1))?;
            if items.bytes >= dictionary_bytes || items.ids.len() >= DICTIONARY_ITEMS {
                break;
            }
        }
        write_jsonl(&mut frame, &Record::End {})?;
        frame.finish()?;
        if turns.peek().is_none() {
            return Ok(());
        }
    }
}

#[derive(Default)]
struct Items {
    ids: HashMap<Box<str>, usize>,
    bytes: usize,
}

impl Items {
    fn intern(&mut self, value: &RawValue, output: &mut impl Write) -> Result<usize> {
        if let Some(id) = self.ids.get(value.get()) {
            return Ok(*id);
        }
        let id = self.ids.len();
        write_jsonl(output, &Record::Item(Cow::Borrowed(value)))?;
        self.ids.insert(value.get().into(), id);
        self.bytes += value.get().len();
        Ok(id)
    }
}

fn pack_turn(line: &str, items: &mut Items, output: &mut impl Write) -> Result<()> {
    let mut turn: Object = serde_json::from_str(line).context("expected a JSON object")?;
    rewrite_turn(&mut turn, |value| {
        if value.get().starts_with('[') {
            let values: Vec<&RawValue> = serde_json::from_str(value.get())?;
            let ids = values
                .into_iter()
                .map(|item| items.intern(item, output))
                .collect::<Result<Vec<_>>>()?;
            Ok(to_raw_value(&ids)?)
        } else {
            Ok(to_raw_value(&items.intern(value, output)?)?)
        }
    })?;
    write_jsonl(output, &Record::Turn(turn))
}

fn expand(mut input: impl BufRead, mut output: impl Write) -> Result<()> {
    for frame in 1.. {
        let decoder = zstd::Decoder::with_buffer(&mut input)?.single_frame();
        expand_frame(BufReader::new(decoder), &mut output)
            .with_context(|| format!("archive segment {frame}"))?;
        if input.fill_buf()?.is_empty() {
            break;
        }
    }
    Ok(())
}

fn expand_frame(input: impl BufRead, output: &mut impl Write) -> Result<()> {
    let mut records = input.lines().enumerate().map(|(index, line)| {
        let line = line.with_context(|| format!("reading archive line {}", index + 1))?;
        serde_json::from_str::<Record<'static>>(&line)
            .with_context(|| format!("invalid archive record on line {}", index + 1))
    });
    match records.next().transpose()? {
        Some(Record::Header { version }) => {
            ensure!(version == VERSION, "unsupported archive version {version}");
        }
        _ => bail!("missing archive header"),
    }
    let mut items = Vec::new();
    for (index, record) in records.by_ref().enumerate() {
        match record? {
            Record::Item(value) => items.push(value.into_owned()),
            Record::Turn(mut turn) => {
                expand_turn(&mut turn, &items)
                    .with_context(|| format!("restoring archive line {}", index + 2))?;
                write_jsonl(output, &turn)?;
            }
            Record::End {} => {
                // Read to the end of the zstd frame to check its checksum.
                ensure!(
                    records.next().transpose()?.is_none(),
                    "data after archive end"
                );
                return Ok(());
            }
            Record::Header { .. } => bail!("unexpected archive header on line {}", index + 2),
        }
    }
    bail!("missing archive end")
}

fn expand_turn(turn: &mut Object, items: &[Box<RawValue>]) -> Result<()> {
    let lookup = |id: usize| {
        items
            .get(id)
            .with_context(|| format!("unknown item ID {id}"))
    };
    rewrite_turn(turn, |value| {
        if value.get().starts_with('[') {
            let ids: Vec<usize> = serde_json::from_str(value.get())?;
            let values = ids.into_iter().map(lookup).collect::<Result<Vec<_>>>()?;
            Ok(to_raw_value(&values)?)
        } else {
            Ok(lookup(serde_json::from_str(value.get())?)?.clone())
        }
    })
}

fn rewrite_turn(
    turn: &mut Object,
    mut rewrite: impl FnMut(&RawValue) -> Result<Box<RawValue>>,
) -> Result<()> {
    for &key in INPUT_FIELDS.iter().chain(OUTPUT_FIELDS) {
        // A capture can use top-level "prompt" for its prompt token count.
        if key == "prompt" {
            continue;
        }
        if let Some(value) = turn.get_mut(key) {
            *value = rewrite(value).with_context(|| format!("field {key}"))?;
        }
    }
    for (key, fields) in [("request", INPUT_FIELDS), ("response", OUTPUT_FIELDS)] {
        if let Some(value) = turn
            .get_mut(key)
            .filter(|value| value.get().starts_with('{'))
        {
            let mut object: Object = serde_json::from_str(value.get())?;
            for &field in fields {
                if let Some(value) = object.get_mut(field) {
                    *value = rewrite(value).with_context(|| format!("field {key}.{field}"))?;
                }
            }
            *value = to_raw_value(&object)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::io::{Cursor, Read};

    use super::{expand, pack};

    #[test]
    fn dictionary_rollover_starts_a_self_contained_frame() -> anyhow::Result<()> {
        let turn = b"{\"input\":[\"same message\"]}\n";
        let mut archive = Vec::new();
        pack(Cursor::new(turn.repeat(2)), &mut archive, 1)?;
        let mut input = Cursor::new(archive.as_slice());
        zstd::Decoder::with_buffer(&mut input)?
            .single_frame()
            .read_to_end(&mut Vec::new())?;
        let split = usize::try_from(input.position())?;
        for frame in [&archive[..split], &archive[split..]] {
            let mut restored = Vec::new();
            expand(Cursor::new(frame), &mut restored)?;
            assert_eq!(restored, turn);
        }
        Ok(())
    }
}
