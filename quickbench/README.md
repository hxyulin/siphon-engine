# quickbench

One command runs three tracks against Siphon or any OpenAI-compatible server and writes one report. On one RTX 5090 a full run takes about 20 minutes. Use it to check an engine change (sampling, KV format, drafter, kernels) and to compare engines.

- **Quality**: hand-picked items that the model gets right only some of the time, each sampled 4 times with thinking on: GPQA Diamond, AIME 2025, HumanEval+ code tasks, synthetic retrieval at about 30K and 125K tokens, and tool-call checks. The report gives the pass rate per source with Wilson intervals, and counts replies cut off at the token cap and replies that loop.
- **Speed**: 4 fixed prompts (chat, code, the lighthouse story, an agent request with tools) at temperature 0.8 with top-p 0.95 and at temperature 1.0, 6 seeds each, at 1 stream and at 8 streams. The report gives median decode tok/s, time to first token and, for siphon-serve, tokens per speculation round.
- **Fidelity** (Siphon only): Siphon's `eval` binary over MLX captures of 4K, 32K and 128K-token texts, with the KV cache in BF16 and in FP8. It reports KL divergence, top-1 agreement and perplexity against MLX.

## Setup

The commands run from this directory with [uv](https://docs.astral.sh/uv/). Fetch the datasets once. GPQA is gated: accept its terms on Hugging Face, then log in with `hf auth login`.

```sh
uv run python -m quickbench fetch
```

The files go to `~/.cache/quickbench/data`, outside the repository. The repository holds only item ids and measured pass rates (`items.json`), never question text, answers or replies, because GPQA's terms forbid publishing its examples.

The fidelity track needs MLX captures, made once on a Mac with the `reference` environment, then copied to `~/.cache/quickbench/refs` on the GPU host:

```sh
M=<model-dir> R=~/.cache/quickbench/refs
uv run --project reference python -I quickbench/reference_texts.py $M $R
for n in 4k 32k 128k; do
  uv run --project reference python -I reference/capture_eval.py capture $M $R/fidelity-$n.txt $R/fidelity-$n.safetensors
done
```

The texts are public-domain books from Project Gutenberg. On an M5 Max the three captures take about 4 minutes. `eval`'s pass and fail thresholds come from MLX's own noise on the code texts in `reference/prompts`. On these prose texts the BF16 cache already sits at the mean |dlogprob| limit of 0.045 (0.040 to 0.045 on 2026-11-25), so `eval` can print FAIL by a few percent. Compare configurations by their KL and |dlogprob| side by side, not by the verdict.

## Running

Build the binaries with `cargo build --release -p siphon-serve -p siphon-cuda --bin siphon-serve --bin eval`, then:

```sh
uv run python -m quickbench run --serve ../target/release/siphon-serve \
    --eval ../target/release/eval --model <model-dir> --name fp8-default
```

The bench starts siphon-serve twice: with 8 streams of 40,960 positions for the speed track and most quality items, then with 1 stream of 135,168 positions for the 125K-token retrieval items. Then it runs the fidelity track. `--serve-args '--kv bf16'` passes options to both servers. Leave out `--context` and `--streams`, which the bench sets.

Against another server, the HTTP tracks run alone:

```sh
uv run python -m quickbench run --url http://localhost:8000/v1 --name vllm
```

Set `QUICKBENCH_API_KEY` if the server needs a key. All items go to that one endpoint, so the 125K-token items fail on a server with a shorter context.

Before it starts and after each track, the bench lists the GPU's compute processes. It waits while a process other than the desktop compositor uses the GPU (`--no-wait` skips the wait), and the report warns when one appeared during a track. Run it under the GPU lock that your other GPU jobs take.

Each run writes `report.md`, `results.json`, the replies (`replies.jsonl`) and the server logs to `~/.cache/quickbench/runs/<name>`, or to `--out`. Keep the replies out of the repository.

## Comparing runs

```sh
uv run python -m quickbench compare ~/.cache/quickbench/runs/fp8-default ~/.cache/quickbench/runs/kv-bf16
```

Quality attempts are paired by item, seed and option order. The table gives the difference in pass rate with a 95% interval from a paired bootstrap over items, the attempts only one side passed, and an exact McNemar test on those. A difference counts as real when the interval excludes zero. Speed is paired by prompt, sampling and seed. Fidelity metrics are listed side by side.

## Items

`items.json` lists each item's id, its measured pass rate and its median reply length. Ids are:

| Id | Source |
| --- | --- |
| `gpqa/<row>` | Row of `gpqa_diamond.csv` in `Idavidrein/gpqa` at revision `83022cef`. The option order is drawn per attempt from the item and attempt number. |
| `aime/<id>` | `id` in `math-ai/aime25` at revision `563bb840`. |
| `code/<n>` | `HumanEval/<n>` in `evalplus/humanevalplus` at revision `d32357cf`. The reply's last Python block runs against the HumanEval+ tests in a subprocess with a 60-second timeout, in a network namespace (`unshare -rn`) where available. |
| `long/<tokens>/<seed>` | Locker records generated from the seed. The model must return the codes of 4 lockers, each reached through a chain of 3 records that point to another locker, with near-duplicate labels next to the last one. |
| `tool/<name>` | Tool-call checks defined in `quality.py`. |

Output caps are 16,384 tokens for GPQA, AIME and code, 8,192 for retrieval and 4,096 for tools (`quality.CAPS`, `--cap gpqa=12288` to change one). Items were chosen for pass rates between about 20% and 90% and for short replies, so that 4 attempts of every item fit in about 10 minutes at 8 streams.

To find new items, run a pool of candidate ids with `--tracks quality --attempts 1`, then more attempts of the borderline ones with `--first-attempt 1 --attempts 3`, and copy the pass rates into `items.json`.
