# quickbench

Two benchmarks for Siphon and other OpenAI-compatible servers, run with [uv](https://docs.astral.sh/uv/) from this directory:

- `quickbench run` is the build check. It answers "did this engine change alter speed or numerics?" in about 12 minutes on one RTX 5090.
- `quickbench suite` is the quality suite. It answers "how should we run the model?" (sampling, reasoning effort, output caps) over 313 unfiltered items in about 35 minutes per configuration.

## Setup

Fetch the datasets once. GPQA is gated: accept its terms on Hugging Face, then log in with `hf auth login`.

```sh
uv run python -m quickbench fetch
```

The files go to `~/.cache/quickbench/data`, outside the repository. The repository holds only item ids (`items.json`, `suite.json`), never question text, answers or replies, because GPQA's terms forbid publishing its examples.

The fidelity track needs MLX captures, made once on a Mac with the `reference` environment, then copied to `~/.cache/quickbench/refs` on the GPU host:

```sh
M=<model-dir> R=~/.cache/quickbench/refs
uv run --project reference python -I quickbench/reference_texts.py $M $R
for n in 4k 32k 128k; do
  uv run --project reference python -I reference/capture_eval.py capture $M $R/fidelity-$n.txt $R/fidelity-$n.safetensors
done
```

The texts are public-domain books from Project Gutenberg. On an M5 Max the three captures take about 4 minutes. `eval`'s pass and fail thresholds come from MLX's own noise on the code texts in `reference/prompts`. On these prose texts the BF16 cache already sits at the mean |dlogprob| limit of 0.045 (0.040 to 0.045 on 2026-11-25), so `eval` can print FAIL by a few percent. Compare configurations by their KL and |dlogprob| side by side, not by the verdict.

Build the binaries with `cargo build --release -p siphon-serve -p siphon-cuda --bin siphon-serve --bin eval`.

## Build check

```sh
uv run python -m quickbench run --serve ../target/release/siphon-serve \
    --eval ../target/release/eval --model <model-dir> --name main-abc1234
```

It runs three tracks:

- **Speed**: 4 fixed prompts (chat, code, the lighthouse story, an agent request with tools) at temperature 1, the model's default, with 6 seeds each, at 1 stream and at 8 streams. A fifth prompt of about 32,000 tokens of generated locker records decodes 512 tokens at 1 stream. Each seed's long prompt differs from its first line, so its first request misses siphon-serve's one-entry prefix cache and gives the cold time to first token. The same request again gives the warm one. The report gives median decode tok/s, time to first token and, for siphon-serve, tokens per speculation round. `--speed-sampling t0.8-p0.95` adds temperature 0.8 with top-p 0.95.
- **Quality**: 1 attempt of each of the 23 items in `items.json`, with thinking on: GPQA Diamond, AIME 2025, HumanEval+, generated retrieval at about 30K and 125K tokens, and tool calls. The report gives pass rates, cut-offs at the output cap, looping replies, tokens and seconds per reply.
- **Fidelity** (Siphon only): Siphon's `eval` binary over MLX captures of 4K, 32K and 128K-token texts, through the served prefill (chunks of 2,048 rows) and the served verify pass (8 rows, FP8 weights and KV cache). It reports KL divergence, top-1 agreement and perplexity against MLX. `eval --prefill` keeps BF16 weights whatever `--weights` says, so only the verify config measures the FP8 weight copies decode reads.

Wall time, estimated from the runs of 2026-11-25: server starts 0.2 minutes, speed 1.8, quality 3 to 4.5, fidelity 6. `--tracks quality,speed` skips fidelity and takes about 6 minutes.

The bench starts siphon-serve twice: with 8 streams of 40,960 positions for the speed track and most quality items, then with 1 stream of 135,168 positions for the 125K-token retrieval items. `--serve-args '--kv bf16'` passes options to both servers. Leave out `--context` and `--streams`, which the bench sets.

### What the build check detects

Seeded runs on one build repeat exactly: every reply and token count. So compare a build against a baseline run of main:

```sh
uv run python -m quickbench compare ~/.cache/quickbench/runs/main-abc1234 ~/.cache/quickbench/runs/my-change
```

The compare counts the replies whose reasoning, content and tool calls are byte-identical with equal token counts, and lists the items whose replies differ. A build that changes numerics changes replies, and shows in the fidelity KL, long before pass rates move. Speed is paired by prompt, sampling and seed.

The build check's pass rates carry no verdict. The 23 items were picked because sampled replies passed them 20% to 90% of the time, and for short replies. That biases any comparison between decoding settings, and it leaves out the long problems where greedy decoding loops. At 1 attempt each they resolve only tens of points. Use the suite to choose how to run the model.

## Quality suite

```sh
uv run python -m quickbench suite --serve ../target/release/siphon-serve --model <model-dir> --name s-t1
uv run python -m quickbench suite --serve ../target/release/siphon-serve --model <model-dir> \
    --serve-args=--greedy --name s-greedy
uv run python -m quickbench compare ~/.cache/quickbench/runs/s-t1 ~/.cache/quickbench/runs/s-greedy
```

Against another server, pass `--url http://host:8000/v1` instead of `--serve`. Set `QUICKBENCH_API_KEY` if the server needs a key. All items go to that one endpoint, so its context must hold a 127K-token prompt and a 32K-token reply.

`suite.json` lists 313 items, drawn without looking at pass rates or reply lengths (`python -m quickbench.suite` draws them again from the fetched datasets):

| Source | Items |
| --- | --- |
| GPQA Diamond | 100 of the 198 rows, sampled with a fixed seed |
| HumanEval+ | all 164 |
| AIME 2025 | all 30 |
| Retrieval | 8 at about 30K and 8 at about 125K tokens, seeds the build check does not use |
| Tool calls | the 3 checks |

Every source gets a 32,768-token output cap (`--cap gpqa=16384` changes one). Other options: `--effort` (sent as `reasoning_effort`: a number in [0, 0.99] or a name such as `high`; siphon-serve defaults to 0.9), `--temperature` (default 1), `--top-p`, `--attempts` (default 1) and `--first-attempt` (default 0). One attempt per item over more items estimates the population without bias and pairs at item level.

With `--serve`, the suite starts siphon-serve with 8 streams of 40,960 positions for GPQA, AIME, code and tools, then with 2 streams of 163,840 positions for the retrieval items.

The report gives, per source and overall: the pass rate with a 95% Wilson interval, the cut-off and loop rates, mean and median tokens per reply, mean seconds per reply, and seconds and tokens per correct answer (totals over passes). The Wilson interval assumes independent replies, which holds at 1 attempt per item.

`compare` pairs two suite runs by item: the difference in pass rate with a 95% paired bootstrap interval over items, an exact sign test over the items each side passed more often, and cut-offs, loops, tokens and seconds side by side. Both tests count items, not attempts: greedy decoding repeats one reply on every attempt of an item whose prompt does not change.

`--dry-run` prints the items and the estimated tokens and wall time without a server. At temperature 1 it estimates 1.55 million tokens: 30 minutes at 8 streams (850 tok/s in total), then 3 minutes for the retrieval items. The tokens per reply come from earlier runs and live in `suite.json`. One reply that runs to the cap takes 5 minutes at 110 tok/s, so a late cut-off adds up to that much.

## GPU use

Before it starts and after each track, the bench lists the GPU's compute processes. It waits while a process other than the desktop compositor uses the GPU (`--no-wait` skips the wait), and the report warns when one appeared during a track. Run it under the GPU lock that your other GPU jobs take.

Each run writes `report.md`, `results.json`, the replies (`replies.jsonl`) and the server logs to `~/.cache/quickbench/runs/<name>`, or to `--out`. `results.json` holds a hash of each reply; `compare` hashes `replies.jsonl` for older runs. Keep the replies out of the repository.

## Items

Ids are:

| Id | Source |
| --- | --- |
| `gpqa/<row>` | Row of `gpqa_diamond.csv` in `Idavidrein/gpqa` at revision `83022cef`. The option order is drawn per attempt from the item and attempt number. |
| `aime/<id>` | `id` in `math-ai/aime25` at revision `563bb840`. |
| `code/<n>` | `HumanEval/<n>` in `evalplus/humanevalplus` at revision `d32357cf`. The reply's last Python block runs against the HumanEval+ tests in a subprocess with a 60-second timeout, in a network namespace (`unshare -rn`) where available. |
| `long/<tokens>/<seed>` | Locker records generated from the seed. The model must return the codes of 4 lockers, each reached through a chain of 3 records that point to another locker, with near-duplicate labels next to the last one. |
| `tool/<name>` | Tool-call checks defined in `quality.py`. |

Each attempt's seed and GPQA option order come from the item and the attempt number, so a run repeats exactly.

`items.json` lists the build check's items with their measured pass rates and median reply lengths. Its output caps are 16,384 tokens for GPQA, AIME and code, 8,192 for retrieval and 4,096 for tools (`quality.CAPS`). It scores from attempt 4 (`--first-attempt 4`, the default): attempts 0 to 3 picked the items, and an item made the list because some of those sampled replies failed, so scoring them again would bias a sampled run against any other configuration. To find new items, run a pool of candidate ids with `--tracks quality --first-attempt 0`, then more attempts of the borderline ones with `--first-attempt 1 --attempts 3`, and copy the pass rates into `items.json`.
