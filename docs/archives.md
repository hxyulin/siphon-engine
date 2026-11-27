# Conversation archives

`siphon archive` packs exported JSONL logs on any platform, without loading a model.
Enable it with the `archive` Cargo feature. Use one JSON object per line, with unique
object keys. A turn can include the request and response bodies alongside any IDs,
timings, token counts, or errors the export provides. On Linux, first activate the
CUDA build environment described in [Installation](install.md#linux-with-cuda).

```sh
cargo build --release -p siphon-cli --features archive
target/release/siphon archive pack turns.jsonl -o turns.jsonl.zst
target/release/siphon archive expand turns.jsonl.zst -o restored.jsonl
```

The default build has no zstd dependency. The separate [HTTP zstd feature](server.md#http-compression)
can be enabled with `--features archive,http-zstd` when you want both.

Both commands default to stdin and stdout. Use `-o` for a new file that appears only
after the command succeeds. Expansion preserves the JSON values and their record
and array order. Blank lines are skipped, and object key order and whitespace can change.

Packing stores identical JSON items once per zstd segment. It looks in the
top-level `instructions`, `system`, `messages`, `input`, `tools`, `output`,
`content`, and `choices` fields. In a `request` object it packs
`instructions`, `system`, `messages`, `input`, `tools`, and `prompt`.
In a `response` object it packs `output`, `content`, and `choices`.
All other fields pass through as JSON. Matching uses each item's serialized bytes,
so consistent key order and whitespace within items improve deduplication.
Archive references show repeated items. To see which requests actually reused the
server's KV cache, include its usage fields or the `cached` and `cache` fields from
the [request log](server.md#logs) in the export.

Use `zstd -dc turns.jsonl.zst` to inspect the packed JSONL. Every zstd frame starts
with `{"header":{"version":1}}` and ends with `{"end":{}}`. Between them,
`{"item":...}` defines an item at the next zero-based ID. A `{"turn":{...}}`
replaces each packed field with an ID or, for an array, an ordered list of IDs.
Each segment has its own dictionary. The packer rotates after a turn brings the
dictionary to 64 MiB of item JSON or 65,536 items. You can also concatenate
packed files and expand them together.
