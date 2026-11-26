# Siphon engine internals

How the engine is built, tested and measured. The [README](../README.md) has the overview.

## Kernel tests

Each `crates/siphon-cuda/src/ops/<group>.rs` launches the kernels in `kernels/<group>.cu` and is tested by `tests/<group>.rs`. The tests compare GPU output with MLX output saved in `tests/fixtures/<group>.safetensors`, which `reference/fixtures/<group>.py` regenerates:

```sh
cd reference/fixtures && uv run --project .. python <group>.py
```

## Prefill

`Model::step` runs one token. `Model::prefill(tokens, chunk, logits)` runs a whole prompt from the current position in chunks of up to `chunk` rows: every layer visit for a chunk, then the next chunk. It fills every KV cache and conv state as stepping the tokens would, advances the position by `tokens.len()`, and returns the greedy next token, so `step` continues from there.

- `chunk` is 1 to 2048 (`MAX_PREFILL_CHUNK`). The default, `DEFAULT_PREFILL_CHUNK`, is 2048, the fastest measured: bigger chunks read each expert weight once for more rows. MLX's chunked captures use 512. The first call allocates buffers for its chunk size: about 0.27 GB at 512 rows, 0.47 GB at 1024 and 0.9 GB at 2048.
- `PrefillLogits::Last` computes only the last row's logits, which `Model::logits` then returns. `PrefillLogits::All(callback)` also passes every row's F32 logits after the softcap to the callback, with the row's position. The LM head runs over 32 rows at a time for it.
- `Model::reset` zeroes the caches and returns to position 0 for a new sequence.

`Model::prefill_with_images` prefills a prompt with images. Before each chunk it asks its caller for the images whose placeholders overlap the chunk's positions, as `ImageRows`: the position of an image's first placeholder and its rows, BF16 `[placeholders, 2304]`. After the embedding gather and `embed_norm`, the chunk copies the rows that fall inside it over its normed embeddings, which is where Transformers' Inkling 2 scatters the vision tower's output ([vision.md](vision.md#text-model-input)). Positions stay sequential. An image may start before the chunk, as when a prompt resumes from a prefix cache boundary inside an image, or end after it; each chunk copies only its own rows. Verify passes and batched steps take no images. `tests/images.rs` stands text in for an image, the image's rows being the normed embeddings of other tokens (`Model::embedding_rows`), and checks that the logits equal the text prompt's bit for bit with images across the chunk boundaries at 2,048 and 4,096 and with a prefill resumed at 2,048 inside an image, and that speculation then gives the text prompt's tokens and acceptance.

With an FP8 cache, global attention runs on FP8 tensor cores by default once a chunk reads more than 16,384 keys, and the expert and dense products can too; [Prefill on FP8 tensor cores](#prefill-on-fp8-tensor-cores) has the options.

The small kernels have multi-row forms. Most give the single-row results bit for bit; depth attention, the router logits and the message gates' products sum in their own orders and differ by F32 rounding. The GEMMs, expert products and attention go through `Ctx::gemm_rows`, `Ctx::moe_rows` and `Ctx::attention_rows` in `src/model.rs`. `gemm_rows` runs a tensor-core GEMM. `moe_rows` runs an expert-grouped GEMM, with the softcapped SwiGLU in the `w13` product's epilogue, and `attention_rows` a flash attention over the cache and the chunk. `tests/prefill.rs` checks prefill against stepping on 1300 tokens at chunks of 512, 1024, 433 and 2048. At 2048 the whole prompt is one chunk wider than the sliding window. `fp8_prefill_matches_stepping` repeats it with the FP8 cache and weights and every FP8 prefill product on. The tests need the checkpoint and about 20 GB of GPU memory, so they are ignored by default:

```sh
SIPHON_MODEL=<model-dir> cargo test --release -p siphon-cuda --test prefill -- --ignored --nocapture
```

## DSpark draft

`draft::Draft` runs the speculative draft bundled in the checkpoint (`mtp.safetensors`): a 3-layer dense transformer over one block of 8 rows, then a recurrent Markov head that turns the 7 mask rows into 7 greedy draft tokens. It reads the target's embedding, `embed_norm` and LM head, and folds `hidden_out` into the LM head at load in F64. Its kernels are in `kernels/draft.cu`; the BF16 products run through `ops::linear`.

- `Draft::commit(taps, start)` writes the draft cache entries of target positions `start..` from their four feature taps, BF16 `[rows, 4, 2304]`: the partial block entering visits 5, 23 and 41, and the final normed hidden state.
- `Draft::round(anchor)` runs the block at the next uncommitted position, with the anchor token read from device memory, and leaves the 7 ids in `Draft::draft_ids` and their confidences in `Draft::confidences`.
- Each draft's confidence is its softmax probability under the Markov logits, `exp(logit - 30) / sum exp(logit_c - 30)`: the softcap bounds every logit by 30, so the sum needs no running maximum. The kernel that scans every row (`draft_markov_bound`, `draft_markov_fold` or `draft_markov_fold_any`) writes a partial sum per block, and the next step's pick divides. The pruned argmax sums the logits of the E4M3 estimates and a shortlist head sums only its rows, so the confidence is an estimate; the round takes no measurable extra time. A sampling request makes the head pick by the target's race instead of the argmax, and its confidence stays the softmax at temperature 1 (see [Speculative sampling](#speculative-sampling)).
- `Draft::round_inputs()` holds a round's inputs in device memory: the taps of up to 8 positions to commit, BF16 `[8, 4, 2304]`, and `control`, I32 `[start, rows, anchor]`. `Draft::run_round(steps)` commits those positions, runs the block right after them, and runs the base logits and Markov head for `steps` drafts (1 to 7). `Draft::replay_round(steps)` replays the same work as a CUDA graph, captured once per `steps`. A verify pass can write the inputs on the device and launch the draft graph right after its own; no position goes through the host. `Draft::set_round_inputs` fills the inputs from the host.
- Layers 0 and 2 keep rings of 1032 slots, the window of 1024 and the block. Layer 1 keeps every position up to the capacity, 4 KiB each, or 2 KiB as E4M3 with `fp8_kv` (below).

The block attention takes its split count from the key count on the device: one warp per block row, one lane per key of each chunk of 32, chunks staged with `cp.async`. The Markov head picks each step's argmax without summing every fold row. An E4M3 copy of the fold bounds each row's logit with the L1 norm of its quantization residual, and only the rows whose upper bound reaches the best lower bound get the BF16 product. The draft ids stay the same as with the full sum. Rounds that record the Markov logits (`Draft::record_logits`) still sum every row. `tests/draft_exact.rs` checks the draft ids and hashes of Y, the base logits and the Markov logits at every anchor of `regex-readme` against a golden file that the draft wrote before these changes, for `commit` and `round`, for device inputs, and for the graphs of 7, 3 and 1 drafts.

Round times at position 2790 (one committed position plus the round, graph replay, RTX 5090):

| Change | 3 drafts | 7 drafts |
| --- | ---: | ---: |
| Before: `commit` of one position, then `round` | 1.726 ms | 1.726 ms |
| Device inputs as a graph, rewritten attention, 2 kernels per Markov step, only the verified drafts | 1.272 ms | 1.512 ms |
| Fewer shared-memory loads in the attention | 1.249 ms | 1.490 ms |
| Pruned argmax over an E4M3 copy of the fold | 1.186 ms | 1.348 ms |
| Weight loads without branches, coalesced bound pass | 1.175 ms | 1.316 ms |

At position 32768, where layer 1 attends over every position, the same steps take 2.169, 1.420, 1.354, 1.288 and 1.277 ms for 3 drafts, and 2.169, 1.660, 1.593, 1.450 and 1.419 ms for 7.

`DraftOptions` switches on approximations. A draft token only changes how often the target accepts, never the target's output, so each one is kept only if it raises accepted tokens per millisecond or frees memory without lowering them. `DraftOptions::fast` turns on all five and is the draft speculation uses (`Speculator::new`, `bench --spec`, `siphon-serve`); `DraftOptions::fp8` leaves out the shortlist, and `DraftOptions::default` stays exact:

- `fp8_backbone`: E4M3 copies of the block's and the round commit's projections, through a multi-row FP8 product (`draft_gemm_fp8`). The draft then frees the BF16 query, output, gate, up and down projections, which only the block reads; the prefill's commit (`Draft::commit`, up to 512 rows) still reads the BF16 keys, values and context projections.
- `fp8_head`: base logits from an E4M3 copy of the LM head.
- `fp8_fold`: the Markov logits from the E4M3 fold, without the exact correction.
- `fp8_kv`: the global layers' caches as E4M3 without a scale, as the target's FP8 cache (see [KV cache format](#kv-cache-format)): 2 KiB per position and layer instead of 4. Keys are RMS-normed per head; the block attention fixture's keys and values reach 5.2 in magnitude, far inside E4M3's 448. The stores round each BF16 value to the nearest E4M3 value. The block attention loads each chunk's E4M3 keys and values into registers before the previous chunk's arithmetic and widens them exactly to BF16 in shared memory after it, so it computes what the BF16 kernel computes over the rounded values (`tests/draft.rs` checks this bit for bit).
- `shortlist`: Ingine's shortlist head (see below). The base and Markov logits cover only the 32,768 highest-ranked tokens, the special tokens and up to 16,384 context tokens passed to `Draft::note_tokens`. The speculator notes the tokens of the prompt's last 8,192 positions and each round's emitted tokens, so a long prompt cannot fill every context row and shut out the tokens that generation adds. The ranking is `draft::counted_ranking`, the tokens the target generates most often (`crates/siphon-cuda/data/shortlist.bin`). A ranking file of little-endian u32 token ids, as Ingine's `draft_shortlist.bin`, can replace it (`--shortlist`).

The shortlist halves the round (0.51 against 0.96 ms). A shortlist in token id order (the tokenizer numbers tokens in BPE merge order) cost acceptance wherever the text used tokens above id 32,768 before they appeared in the context: reasoning text opens with ids near 150,000, and 46% of generated tokens lie above that id. Noting each round's anchor on the device, so the next round can draft it, changed neither acceptance nor speed on any prompt below, so the loss is the ranking's coverage, not the one-round lag. `bench --spec auto`, 512 tokens, FP8 weights and KV, tok/s and tokens per round (chat, coding and 32K-document prompts from the Ingine comparison, then the three texts):

With the shortlist in id order:

| Prompt | Bundled, `fp8` | Bundled, id order | DFlash 2, `fp8` | DFlash 2, id order |
| --- | --- | --- | --- | --- |
| Chat | 466, 3.27 | 405, 2.70 | 544, 4.21 | 407, 3.09 |
| Coding task | 471, 3.36 | 424, 2.89 | 566, 4.20 | 465, 3.44 |
| 32K document | 534, 4.89 | 470, 4.01 | 364, 3.13 | 345, 2.90 |
| `regex-readme`, 2,286 | 494, 3.66 | 525, 3.71 | 516, 4.06 | 520, 3.97 |
| `half`, 2,273 | 683, 5.67 | 689, 5.43 | 585, 4.90 | 597, 4.90 |
| `anyhow`, 10,674 | 761, 6.76 | 793, 6.68 | 759, 6.83 | 779, 6.83 |

The counted ranking keeps both. `scripts/shortlist/generate.py` sent 1,783 open-perfectblend prompts (`scripts/shortlist/download.py` and `prompts.py`, efforts none, low and medium in turn, at most 1,024 tokens) to `siphon-serve` with `return_token_ids` and stored the token ids: 1.0 million generated tokens in 40 minutes. `scripts/shortlist/rank.py` ranks output tokens by count, then prompt tokens by count, then the rest by id. Share of generated tokens in the first 32,768 ranked ids or the special tokens:

| Ranking | Held-out prompts (107) | Chat | Coding | Instruction | Math | 2DSD core prompts (20,515 tokens) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Token id order | 44.5% | 38.2% | 46.2% | 41.0% | 48.8% | 54.0% |
| Ingine's `draft_shortlist.bin` | 96.7% | 94.1% | 98.0% | 94.4% | 97.2% | 96.5% |
| Counted | 98.9% | 98.2% | 99.3% | 96.5% | 99.4% | 99.2% |

Same runs as above with the counted shortlist (`fast`), tok/s, tokens per round and draft ms per round:

| Prompt | Bundled, `fp8` | Bundled, `fast` | DFlash 2, `fp8` | DFlash 2, `fast` |
| --- | --- | --- | --- | --- |
| Chat | 465, 3.27, 0.95 | 495, 3.23, 0.52 | 540, 4.17, 1.03 | 542, 3.98, 0.79 |
| Coding task | 470, 3.36, 0.95 | 502, 3.34, 0.51 | 565, 4.20, 1.03 | 594, 4.23, 0.79 |
| 32K document | 535, 4.89, 1.08 | 557, 4.79, 0.64 | 364, 3.13, 1.05 | 375, 3.13, 0.81 |
| `regex-readme`, 2,286 | 494, 3.66, 0.97 | 527, 3.63, 0.53 | 518, 4.06, 1.05 | 534, 4.03, 0.80 |
| `half`, 2,273 | 683, 5.67, 0.97 | 722, 5.67, 0.53 | 585, 4.90, 1.04 | 596, 4.85, 0.80 |
| `anyhow`, 10,674 | 762, 6.76, 1.01 | 800, 6.68, 0.57 | 759, 6.83, 1.05 | 783, 6.83, 0.81 |

DFlash 2 attends to only the last 2,048 positions (`sliding_window` in its config, every layer), so on the 32K document, whose answer restates code from far back, it keeps 3.13 tokens per round against the bundled draft's 4.79. Widening its window at inference made it worse: 326 tok/s at 8,192 and 264 at 32,768, against 366 at 2,048.

`dspark accept` measures each variant's teacher-forced acceptance on the stage 0 taps against MLX's top-1 tokens, as `reference/dspark.py accept` counts it, and times its round of 7 drafts at position 2,790 and 32,768. Over all three texts (16,745 anchors):

| Variant | Acceptance, drafts 1 to 3 | Tokens per round, k = 7 | Round, 2,790 | Round, 32,768 |
| --- | --- | ---: | ---: | ---: |
| Exact | 79.2, 58.7, 45.2% | 3.793 | 1.316 ms | 1.420 ms |
| FP8 backbone | 79.2, 58.7, 45.2% | 3.794 | 1.220 ms | 1.331 ms |
| FP8 head | 79.3, 58.7, 45.2% | 3.792 | 1.062 ms | 1.166 ms |
| FP8 fold | 79.2, 58.7, 45.1% | 3.792 | 1.321 ms | 1.425 ms |
| Shortlist, id order | 79.2, 58.7, 45.2% | 3.794 | 0.708 ms | 0.812 ms |
| Shortlist, ranked by counts over a code sample | 76.7, 56.3, 43.4% | 3.691 | 0.708 ms | 0.813 ms |
| Fast: shortlist in id order, FP8 backbone, head and fold | 79.2, 58.6, 45.0% | 3.787 | 0.491 ms | 0.628 ms |

`fp8_kv` keeps acceptance: over the same anchors with the counted shortlist, `fast` without it keeps 79.0, 58.4 and 44.8% for drafts 1 to 3 and 3.777 tokens per round at k = 7, and with it 79.1, 58.4, 44.9% and 3.778; the exact draft keeps 3.793, and with only `fp8_kv` 3.789. The E4M3 cache makes the round of 7 drafts faster at long context, 0.621 against 0.633 ms at 32,768, and leaves it at 0.491 ms at 2,790 (`dspark time`, graph replay). On the 12 prompts of the server check in [Tensors in host memory](#tensors-in-host-memory), whose greedy outputs did not change, the bundled draft kept 3.086 tokens per round against 3.097 on main.

FP8 fold alone saves nothing, since the pruned argmax already reads the E4M3 fold; it matters with a shortlist, which skips the pruning. That earlier counted ranking took 72 million tokens of open-source Rust crates and the CPython standard library, the evaluation texts' crates left out. Code-heavy counts drop common prose tokens that the id order keeps.

In a kernel trace of the round of 7 drafts at position 2,790 (each kernel from the previous one's end), the exact draft spends 918 of its 1,315 us in BF16 GEMMs, most of it the 7-row LM-head product over the full vocabulary, and 212 us in the pruned argmax's bound pass. The fast draft takes 494 us: 339 us in the FP8 GEMMs (16 products), 42 us in attention (138 us at 32,768), 34 us in the shortlist Markov fold and 20 us in the Markov steps.

```sh
cargo run --release -p siphon-cuda --bin dspark -- accept <model-dir> <taps-dir> \
    --variants exact,shortlist,fast --steps 7
cargo run --release -p siphon-cuda --bin dspark -- time <model-dir> <taps-dir>/regex-readme-taps.safetensors
SIPHON_MODEL=<model-dir> SIPHON_DSPARK_REF=<dir> cargo test --release -p siphon-cuda --test draft_exact -- --ignored --nocapture
```

The draft follows the rounding points of vLLM's draft rather than MLX's kernels. `reference/dspark.py` implements the same draft in MLX, in BF16 or F32, and reports teacher-forced acceptance on a text. `tests/draft.rs` checks each kernel against `reference/fixtures/dspark.py`. `tests/draft_round.rs` feeds the engine draft the MLX taps of `reference/prompts/regex-readme.txt` and compares it at every anchor with the MLX draft, within the MLX draft's own BF16 against F32 difference:

```sh
uv run --project reference python -I reference/capture.py <model-dir> reference/out/regex-readme-taps.safetensors --text reference/prompts/regex-readme.txt --taps
uv run --project reference python -I reference/dspark.py run <model-dir> reference/out/regex-readme-taps.safetensors reference/out/regex-readme-dspark-bf16.safetensors
uv run --project reference python -I reference/dspark.py run <model-dir> reference/out/regex-readme-taps.safetensors reference/out/regex-readme-dspark-f32.safetensors --float32
SIPHON_MODEL=<model-dir> SIPHON_DSPARK_REF=reference/out cargo test --release -p siphon-cuda --test draft_round -- --ignored --nocapture
```

`reference/dspark.py routing` reads captures made with `capture.py --routing` and reports the distinct experts that m consecutive positions route to at each visit, which sets the expert bytes of a verify of m rows.

## DFlash draft

`Draft::dflash` runs a second kind of draft in place of the bundled one: the 2DSD project's DFlash drafter with DSpark heads, as its exporter writes it (`config.json` with `architectures: DFlashDraftModel` and a `loopedtf_drafter.dspark` section, and `model.safetensors`). `DflashWeights` checks the config against what the kernels run. The released drafter for the research checkpoint (`minseo25/2dsd-inkling-2-mini-varloop-dspark-drafter`) has this shape:

- 5 taps, the partial block entering visits 2, 31, 36, 41 and 46 (`dflash_config.target_layer_ids`). `Model::set_tap_layout` makes prefill and verify record them in place of the bundled draft's 4.
- `fc` (`[2304, 11520]`) over the concatenated taps, then `hidden_norm`: one context row per position, which every layer projects with the same `k_proj` and `v_proj` as the block's rows.
- 5 layers, every one attending to every position. The cache takes 20 KiB per position, against 4 KiB for the bundled draft's one global layer (10 and 2 KiB with `fp8_kv`).
- A vanilla Markov head: draft j adds `w2 · w1[t]`, with t the token before it, to its base logits, and the step takes the argmax of the uncapped BF16 sum, as the 2DSD runner does. The softcap is monotone, but in BF16 it maps neighbouring logits to one value and changes the tie-break.
- A confidence head, `sigmoid(proj · [y_j; w1[t]] + bias)`, which `WidthPolicy` reads as each draft's acceptance chance.
- A learned mask vector in `mask_embedding.pt`, a pickle. Convert it once in a trusted Python environment with `torch.load(path, weights_only=True)` and save its `embedding` as BF16 `mask_embedding` in `mask_embedding.safetensors` beside `model.safetensors`.

RoPE reads its cosine and sine rounded to BF16, as the runner's table and Transformers' rotary embedding in training hold them. Everything else reuses the bundled draft's kernels, approximations (`DraftOptions`) and CUDA graphs. `bench`, `siphon-serve` and `tests/spec.rs` take the export directory (`--drafter DIR`, `SIPHON_DFLASH`).

`tests/dflash.rs` compares the drafts with the runner's (`vllm/v1/spec_decode/looped_dflash_context.py` of the 2DSD vLLM fork) given the same taps, at each of 1,891 anchors of one core prompt and its greedy continuation on the research checkpoint. The exact draft gives the runner's 7 drafts in 1,827 rounds. In the other 64 the first different draft falls on a near-tie, a top-2 gap of at most 0.25, and the confidences differ by at most 0.011. The runner's own BF16 and F32 paths agree in 94.5% of rounds, the engine and the F32 path in 95.5%, and the engine's decoder output is as far from the F32 path (median relative error 0.0080) as the runner's BF16 output is (0.0087).

On this checkpoint the shortlist in token id order costs both drafts acceptance: 46% of the tokens the target generates on the core prompts lie outside the first 32,768 ids and the special tokens. The runs below without a shortlist keep the FP8 backbone, head and fold.

The 32 core prompts of 2DSD's 1d-last run (math, code and science), and 4 prose and chat requests, BF16 weights and KV, verify width 8. Every run emitted the same 19,552 tokens on the core prompts:

| Draft | Variant | Core tokens per round | Core tok/s | Chat tokens per round | Chat tok/s |
| --- | --- | ---: | ---: | ---: | ---: |
| Bundled | `fast` | 3.365 | 320.1 | 1.662 | 156.9 |
| Bundled | no shortlist | 3.629 | 331.6 | 1.769 | 160.9 |
| Bundled | `exact` | 3.626 | 319.6 | 1.769 | 155.3 |
| DFlash | `fast` | 3.907 | 360.3 | 1.738 | 159.3 |
| DFlash | no shortlist | 4.474 | 395.3 | 1.929 | 170.3 |
| DFlash | no shortlist, BF16 backbone | 4.474 | 390.2 | 1.934 | 167.6 |
| DFlash | `exact` | 4.467 | 375.0 | 1.932 | 161.5 |

The 2DSD runner keeps 4.25 tokens per round with this drafter on the same prompts.

The same with FP8 weights and KV and adaptive widths from 1 to 8. Every run emitted the same 19,089 tokens on the core prompts. A DFlash round priced 0.26 to 0.31 ms more than a bundled one at every width:

| Draft | Variant | Core tokens per round | Core tok/s | Chat tokens per round | Chat tok/s |
| --- | --- | ---: | ---: | ---: | ---: |
| Bundled | `fast` | 3.275 | 368.4 | 1.678 | 215.9 |
| Bundled | no shortlist | 3.482 | 379.4 | 1.804 | 229.7 |
| DFlash | `fast` | 3.690 | 419.6 | 1.687 | 230.0 |
| DFlash | no shortlist | 4.144 | 420.1 | 1.836 | 236.9 |

A draft round of 7 drafts with BF16 weights (`bench --spec 7`, the round's commit included), after the core prompt above (1,403 tokens) and with arbitrary ids before it up to 10,240 and 32,768 tokens:

| Draft | Variant | 1,403 | 10,240 | 32,768 |
| --- | --- | ---: | ---: | ---: |
| Bundled | `fast` | 0.522 ms | 0.569 ms | 0.646 ms |
| Bundled | no shortlist | 0.958 ms | 0.997 ms | 1.074 ms |
| DFlash | `fast` | 0.773 ms | 0.963 ms | 1.339 ms |
| DFlash | no shortlist | 1.218 ms | 1.392 ms | 1.767 ms |

The two extra layers and `fc` cost about 0.25 ms. The rest of the gap grows with the context: 5 layers read every position's keys and values, 0.64 GB at 32,768.

## DFlash 2 draft

`Draft::dflash2` runs Inco AI's DFlash 2 drafter for the official checkpoint ([`incoai/Inkling-2-Mini-DFlash2`](https://huggingface.co/incoai/Inkling-2-Mini-DFlash2) at revision `4f9088ea`) in place of the bundled draft. `--drafter DIR` picks it when `config.json` names `DFlash2DraftModel`, and `Dflash2Weights` checks the config against what the kernels run. The model card's numbers come from SGLang with the repository's compatibility patch (`sglang_inkling_dflash2.py`), so the engine follows SGLang's DFlash worker where it differs from z-lab's generator ([z-lab/dflash](https://github.com/z-lab/dflash)):

- Taps. SGLang's Inkling model captures the partial block entering each visit in `target_layer_ids` and, for the id equal to the visit count, the final normed hidden state. So `[5, 23, 41, 48]` are the bundled draft's taps, and the target records them as it always does. `fc` (`[2304, 9216]`) and `hidden_norm` turn each position's taps into one context row, which every layer projects with the block's own `k_proj` and `v_proj`.
- Block. The anchor and 7 mask rows sit at positions P to P + 7. The rows are the target's input embeddings, `embed_norm(embed(id))`, and the mask rows embed token 200,056. z-lab's generator reads the raw embedding table instead; with raw rows the drafter accepts nothing.
- Layers. 5 Qwen3-style layers with 32 query heads, 8 KV heads and an MLP of 6912. RoPE uses theta 10,000 from a BF16 table, with each product and the sum rounded to BF16 (`Rope::Bf16Ops`), as SGLang's `table_qk_norm_rope` does. Every row attends to the whole block and to the 2048 positions that end at itself: row j at P + j sees P + j - 2047 onward. Each layer's cache is a ring of 2056 slots, so the round costs the same at any context. A commit skips positions that no later block's window reaches.
- Convolutions. A two-tap grouped dynamic convolution wraps each attention and each MLP, over the block's rows only. Tap j of channel c in row r weighs `base[s, j, c] + delta[r, s, j, c / 16]`, where `delta` (576 values per row) projects the sublayer's normed input. Side 0 convolves the sublayer's input; side 1 convolves its output and runs fused with the residual add and the next norm. Row 0 has no previous row, so no convolution state carries over between rounds. Sums run in F32 and round once, as SGLang's Triton kernel does.
- Selector. The drafter has no LM head. Rows 1 to 7 go through the target's LM head, and the top 16 logits over the 200,058 real tokens become each row's candidates. Walking from the anchor, candidate c after token t scores `30 tanh((logit_c + bf16(<bf16(A[t] h), B[c]>)) / 30)`, with `h = hidden_projection · y` (rank 256). The softcap applies after the correction (`selector_softcap_mode: after_correction`, which the patch adds; z-lab caps the logits before adding). The greedy walk takes the best score, ties to the larger logit. A draft's confidence, which `WidthPolicy` reads, is its softmax probability over its row's 16 scores: the distribution a walk at temperature 1 samples from. A sampling request makes the walk pick by the target's race (see [Speculative sampling](#speculative-sampling)).

The block forward shares the other drafts' code. The block attention takes 3 or 4 query heads per KV head and an optional window per row; the bundled draft's results stay the same bit for bit (`tests/draft_exact.rs`). New kernels: `draft_conv` and `draft_conv_add_rms_norm` for the convolutions, `draft_topk_partial` and `draft_topk_merge` for the candidates (two passes, ties to the smaller token), and `draft_select` for the walk (one block, one warp per candidate). Of the `DraftOptions`, `fp8_backbone` and `fp8_head` apply, `shortlist` limits the candidates to the shortlist's tokens, and `fp8_fold` does nothing.

`reference/dflash2.py` runs z-lab's `DFlash2DraftModel` at every anchor of a taps file, with SGLang's rules above around it, and `tests/dflash.rs` compares the engine with it (`SIPHON_DFLASH` set to the drafter, `SIPHON_MODEL` to the official checkpoint). On a math chat prompt and 1,400 tokens of its greedy continuation (1,480 anchors), the exact draft gives the reference's 7 drafts in 1,403 rounds. In the other 77 the first different draft falls on a near-tie: a selector gap of at most 0.25 in 67, and a tie at the top-16 boundary of its row in 10, where the 16th and 17th logits differ by at most two BF16 steps. Against the reference run in F32, the engine's whole rounds agree in 95.4% of anchors and the BF16 reference's in 95.7%. The engine's decoder output is closer to the F32 reference (median relative error 0.0108) than the BF16 reference's (0.0151). Teacher-forced on MLX's top-1 tokens of the three texts, the reference keeps 3.91, 3.27 and 3.43 tokens per round at k = 7 (`anyhow`, `half`, `regex-readme`), against 3.79 for the bundled draft over all three. With the softcap before the correction, as z-lab applies it, `regex-readme` drops from 3.43 to 3.30.

`tests/spec.rs` (`dflash_speculation_is_exact`) checks exactness with the drafter: speculation at k = 1, 3, 7 and adaptive widths returns all 1024 greedy tokens after each of the three texts. The `anyhow` prompt of 10,674 tokens runs every round past the window.

Chat prompts through `siphon-serve` with the bundled chat template, FP8 weights and KV, adaptive widths: two math problems, two coding tasks, a travel-planning conversation and a logic puzzle at effort `xhigh`, 10,338 tokens in all. Every run emitted the same tokens. Without a shortlist, both drafts keep their FP8 backbone and head:

| Draft | Variant | tok/s | Tokens per round | Acceptance, drafts 1 to 7 | Draft per round |
| --- | --- | ---: | ---: | --- | ---: |
| Bundled | no shortlist | 484.9 | 3.78 | 79.8, 59.9, 43.3, 34.7, 26.1, 19.2, 14.9% | 0.956 ms |
| DFlash 2 | no shortlist | 554.9 | 4.61 | 85.6, 71.1, 54.8, 46.9, 39.8, 33.1, 29.2% | 1.046 ms |
| Bundled | `fast` | 468.8 | 3.46 | 72.4, 52.6, 37.9, 29.9, 22.8, 16.7, 13.1% | 0.516 ms |
| DFlash 2 | `fast` | 495.8 | 3.97 | 75.5, 58.5, 45.3, 38.0, 31.2, 26.1, 22.6% | 0.798 ms |

Per prompt, without a shortlist, tok/s and tokens per round:

| Prompt | Tokens | Bundled | DFlash 2 |
| --- | ---: | --- | --- |
| Math: cheapest packs of pencils | 1,057 | 473.0, 3.64 | 557.3, 5.04 |
| Math: nested square roots | 900 | 469.1, 3.58 | 648.7, 5.33 |
| Code: merge intervals in Python | 2,048 | 526.1, 4.30 | 602.3, 5.19 |
| Code: LRU cache in Rust | 2,048 | 525.0, 4.19 | 544.7, 4.57 |
| Chat: a trip to Kyoto | 2,048 | 529.8, 4.32 | 603.2, 5.14 |
| Logic puzzle, effort `xhigh` | 2,237 | 406.2, 2.97 | 468.4, 3.58 |

On math DFlash 2 keeps 5.0 to 5.3 tokens per round against 3.6 for the bundled draft, close to the model card's 4.8 to 5.0 against 3.5 with sampling. The shortlist in token id order costs DFlash 2 more than the bundled draft: reasoning text uses many tokens above id 32,768.

`bench --spec auto`, 256 tokens after each text, FP8 weights and KV, tok/s and tokens per round. The prompt-length column is the second, warm run:

| Text, context | Plain decode | Bundled, no shortlist | DFlash 2, no shortlist | Bundled, `fast` | DFlash 2, `fast` |
| --- | ---: | --- | --- | --- | --- |
| `anyhow`, 10,674 | 328 | 650, 6.12 | 629, 6.12 | 674, 6.12 | 652, 6.24 |
| `anyhow`, 32,768 | 302 | 568, 6.09 | 545, 5.91 | 570, 5.82 | 552, 5.91 |
| `half`, 2,273 | 340 | 609, 5.25 | 490, 4.34 | 635, 5.35 | 481, 4.13 |
| `half`, 32,768 | 301 | 430, 4.34 | 431, 4.43 | 452, 4.41 | 444, 4.51 |
| `regex-readme`, 2,286 | 340 | 383, 2.78 | 395, 3.08 | 404, 2.84 | 395, 3.01 |
| `regex-readme`, 32,768 | 301 | 365, 3.49 | 439, 4.37 | 366, 3.40 | 415, 4.10 |

These greedy continuations repeat their text, which suits the bundled draft; DFlash 2 wins on `regex-readme` and ties or loses on the others. Its round costs 1.047 ms at every context (0.804 ms with the shortlist). The bundled draft's grows from 0.96 to 1.08 ms at 32,768 without the shortlist and from 0.52 to 0.65 ms with it, since its middle layer reads every position.

In a kernel trace of a round of 7 drafts (each kernel from the previous one's end, 1,031 us in all), FP8 products take 766 us: 297 us for the LM head (461 MB at 1.55 TB/s) and 418 us for the five layers' projections. The output (K = 4096) and down (K = 6912) projections run at 0.6 to 0.7 TB/s, since they stage their input in tiles of 3,072 columns per group of rows. Each layer's attention over 2,055 keys takes 18 us with its combine, and its convolutions 23 us (two coefficient products, two input convolutions and two fused output convolutions). The commit takes 58 us; the top 16, the hidden projection and the walk 34 us. Running the five layers' context keys and values as one launch saved 17 us. Groups of 8 rows instead of 16 for the wide products did not help: the down projection got slower.

```sh
SIPHON_MODEL=<official-model-dir> SIPHON_DFLASH=<dflash2-dir> SIPHON_DFLASH_REF=<dir> SIPHON_DFLASH_CONTINUE=1400 \
    cargo test --release -p siphon-cuda --test dflash -- --ignored --nocapture
uv run --script reference/dflash2.py <dir>/taps.safetensors <dflash2-dir> <official-model-dir> <dir>/reference.safetensors
SIPHON_MODEL=<official-model-dir> SIPHON_DFLASH=<dflash2-dir> \
    cargo test --release -p siphon-cuda --test spec -- --ignored --nocapture --test-threads 1 dflash
```

## Speculative decoding

`spec::Speculator` runs speculative decoding with the draft, greedy or sampled ([Sampling](#sampling)). Each round verifies the anchor and its leading drafts, a width of 1 to 8 rows that `spec::Width` sets: a fixed k drafts (`Width::Fixed`), or a width per round from the drafts' confidences (`Width::Adaptive`).

1. `Model::start_verify` launches the anchor and the round's drafts as one pass through the chunked-prefill code. On the device it keeps the anchor and the longest run of drafts that each equal the previous row's token, greedy or drawn, and rolls back the rest.
2. The draft's next round goes on the stream right behind it: it commits the feature taps of the kept rows, whose count the verify left on the device, runs its block from the next anchor, and drafts with a confidence each.
3. `Model::finish_verify` synchronizes once, to read the verdict, and the host reads the new confidences and picks the next round's width. Host time stays near 0.02 ms per round.

`spec::WidthPolicy` takes each confidence c_j as the chance that the target accepts draft j after the drafts before it, so draft j survives with s_j = c_1 ... c_j and a width w emits 1 + s_1 + ... + s_(w-1) tokens on average. It picks the width with the most such tokens per millisecond of round time, from a time per width that the caller measures: `bench --spec auto` times verify passes of each width at the context, plus the draft and host time per round. On the greedy continuations of the three texts, the share of accepted drafts in each confidence bin is within a few points of the confidence at every draft position, so the policy uses no other calibration.

The policy was chosen by simulation over sweeps (`bench --spec K --sweep FILE`), which decode 1,024 tokens one verify row at a time and run a draft round of 7 at every anchor. A round sees the same drafts at an anchor whatever earlier rounds verified, so each sweep prices any policy exactly, given the measured verify time per width. Over the three texts at their own length and at 32,768, BF16 and FP8 weights (12 sweeps), geometric mean tok/s at the first 256 tokens:

| Policy | tok/s, measured verify | tok/s, verify of 0.6 x today's one-row pass + 0.3 ms per extra row |
| --- | ---: | ---: |
| Fixed k = 3 | 362 | 576 |
| Fixed k = 7 | 416 | 662 |
| Ingine: leading drafts above 0.8, widths 2, 4, 8 | 425 | 662 |
| Leading drafts above 0.5, widths 2 to 8 | 444 | 700 |
| Most expected tokens per ms, widths 1 to 8 (`WidthPolicy`) | 441 | 698 |
| Oracle: the width of the accepted run | 489 | 765 |

Thresholds from 0.5 to 0.7 and the cost rule land within 2% of each other; the cost rule keeps that place when the verify cost per row changes, without retuning. Widths 2, 4 and 8 alone lose about 1.5%. The policy can pick width 1, a verify of one row, which a plain step and a draft round beat (below).

A round emits the kept drafts and the target's token after them. `Model::prefill_with_taps` records the draft's taps of every prompt row, so the draft's cache covers the prompt.

The verify pass has one CUDA graph per width, captured on first use, with the tokens, the start position and the acceptance mode in device buffers. `kernels/spec.cu` holds its own kernels:

- Before the pass, it copies every conv state (3.7 MB) and the sliding-ring slots the pass will overwrite (40 visits, 4 KiB per row each). During the pass, each visit records its conv inputs: the raw keys and values, the attention output projection, and the MoE combine.
- After the pass, every row's greedy token, then the acceptance. Each conv state becomes the last 3 rows of its old state followed by the kept rows' inputs. The rejected rows' ring slots get their old keys and values back. The global caches need nothing: attention reads only slots below each row's position, and later rows overwrite the stale ones.

The rings keep 1024 slots. A ring of 1032 slots would make the backup unnecessary, but decode attention splits its cache by slot, so its results change with the ring size.

Each verify row gives the same logits, bit for bit, as a verify of that row alone at its position, whatever the other rows hold. So speculation emits exactly the tokens of greedy decoding through the verify pass one row at a time, whatever the draft proposes. The pass runs kernels built for 1 to 8 rows, whose rows each sum in an order that depends only on their own position and data:

- Dense products (`ops::linear_rows`): each 16-row weight tile streams once for all rows, through per-warp `cp.async` rings onto `mma.sync m16n8k16`, with the weights as the 16-row operand and the rows as the 8 columns. With `--weights fp8` the verify reads the FP8 copies, widened exactly to BF16.
- Router (`Routing::moe_router_step_rows`): the decode router's sums for every row, so each row's experts and weights equal a step's bit for bit. Each block holds 8 experts' router rows and runs 9 warps, one per warp of the decode router's sum order. A warp loads its share of every row once for all 8 experts, and its partial sums reduce 32 at a time by a transposed butterfly that adds the same pairs as the decode router's warp sum, in 31 shuffles per 32 sums instead of 5 per sum. With one warp per expert, each block loaded every row 8 times: 4.2 against 2.9 us per call at 8 rows, 1.9 against 1.7 at 1 row.
- Experts (`ops::experts_rows`): one kernel groups the row-slot pairs by expert on the device and streams each distinct expert's `w13` and `w2` once, on tensor cores with the expert's pairs as the columns. An E2M1 or E4M3 weight times its block scale is an exact BF16 value.
- Attention (`Attention::attention_rows`): each row's keys split by distance. Two blocks per KV head take every row's 8 nearest keys and its sink, reading the rows' own keys from the step's inputs and the rest from the cache. The keys at distance 8 and more come from the cache as it is, since the rows' new keys are within 7 of every row that sees them. 32 blocks per KV head split those slots by absolute slot, and each warp takes 16 keys of each 64-slot split: `mma.sync m16n8k16` computes the scores with the keys as the 16-row operand and the 3 heads × rows as the columns, then the weighted values with the head dims as the 16-row operand. The slot a later row's key goes to still shows its old entry to earlier rows, at distance 1017 or more. The rows' keys append after attention, so 1024-slot rings work. The last blocks of each KV head to finish combine its partials in block order, gate the output and append; there is no separate combine kernel.

Around these kernels the pass follows a decode step's schedule, in fewer launches than chunked prefill uses:

- With FP8 dense weights it forks the decode step's L2 prefetches: the first visit's projections before the pass, and after each visit's experts the next visit's output projection, MLP gate and router, and the projections and sliding ring of the visit after it. With BF16 weights the prefetch made every width slower, so it stays off.
- Up to 8 rows, a visit ends in one kernel, `Conv::moe_conv_residual_step_rows`: the MoE combine, the post-MLP conv and the residual add, the shifts of all four conv states, and the record of the verify's conv inputs. Each warp takes one row of 32 channels and the block shares the conv window in shared memory, so the rows' loads issue together: 2.0 us at 8 rows, against 8.5 us when one thread walked a channel's rows. `attention_post_rows_bf16` also computes the relative bias.
- Each attention warp copies its first key and value chunk before `pdl_wait`: no kernel of the pass writes a visit's cache before that visit's attention. Copying three E4M3 chunks early instead slowed the projections before the attention by as much as it saved. The combine runs in the last blocks of each KV head to finish, one per query row: one block reading every partial was bound by one SM's L2 bandwidth, and 8 blocks of 3 rows took 0.7 us longer at 8 rows. They wait on a counter for the head's other blocks, which have all started, since the next kernel launches only once every block has. Blocks without keys are skipped there; their partials add exactly nothing.
- The nearest-key blocks issue every load before any arithmetic, from both places a key may come from: with the loads behind per-key branches they waited for memory once per key, and a sliding visit's attention took 6.6 instead of 4.6 us at one row.
- The router runs as the decode router's cluster: each block sends row r's logits to block r with `st.async`, and block r routes row r. The expert kernel copies each stage's block scales into the stage's ring slot with its weights, in the same copy group, and verifies of 1 to 3 rows run it with a 4-slot copy ring instead of 3. Loading the scales into registers a few stages ahead cost 0.04-0.08 ms per pass at 1 to 5 rows.
- Each group of message gates, the three of the attention inputs or the MLP's, is one launch of `message_gate_rows_2304`, one 16-block cluster per gate, instead of a down product and two kernels per gate. Block r runs two of the down product's eight warp splits and sends each partial row to the block that owns it with `st.async`; the owner adds the splits in order and sends its 4 rows of the down projection to every block. Each block then computes 144 output columns with the products and combine of `message_gate_out_rows_2304`, and after a cluster barrier block j normalizes row j. A pass of one row runs 500 kernels.
- The looped layers' message depth attentions attend over the bank, which a pass does not change, so one launch per pass computes them all (`Routing::depth_attention_rows_batch`) instead of one per visit.
- The depth attention with the post-attention conv loads its conv taps before `pdl_wait` and pulls the conv state into L2.
- The L2 prefetch kernels ask for the largest shared-memory carveout. With the small carveout the driver gave them, a gate cluster, which needs 16 SMs of one GPC at once, waited up to 13 us for SMs running a prefetch to drain.

Apart from the tensor-core attention, none of these changes a result: the verify logits stay the same bit for bit, which `smallm verify --width M --hash 1` checks across builds by hashing every pass's logits. The attention rounds its weights to F16 for the value product, as prefill attention does, and passes the quality gate with `eval --verify 8` in both weight formats.

`tests/linear_rows.rs`, `tests/experts_rows.rs`, `tests/attention_rows.rs` and `moe_router_step_rows_matches_step` in `tests/routing.rs` check each kernel's row invariance bit for bit, and its distance from the decode and prefill kernels.

The verify pass computes the same model as `Model::step` but sums in other orders, so the two greedy outputs split at the first near-tie: after 247 to 817 of 1024 tokens on the three texts.

`tests/verify.rs` checks the taps of prefill and of plain steps against MLX, a verify against stepping, the rollback against a shorter verify (bit for bit), the ring size, and row invariance. `tests/spec.rs` checks exactness with the draft at k = 1, 3 and 7 and at an adaptive width priced so that every width from 1 to 8 gets picked, and with random drafts at k = 3 and at widths cycling from 1 to 8, 1024 tokens after each text. The same adaptive width with plain prices, with the draft and with random drafts, must equal `greedy_by_path` along its own path. Both need the checkpoint:

```sh
SIPHON_MODEL=<model-dir> SIPHON_TAPS=reference/out/regex-readme-taps.safetensors \
    cargo test --release -p siphon-cuda --test verify --test spec -- --ignored --nocapture --test-threads 1
```

`eval --verify WIDTH` teacher-forces a capture through verify passes of WIDTH rows, and passes the quality gate at widths 1 and 4. `bench --prompt <capture> --spec K|auto` compares speculation with plain decode from one prefilled state, `--draft` picks the draft variant, `--sweep` records drafts and confidences at every anchor, and `--verify-widths` times one pass of each width.

One verify pass, after the `anyhow` prompt of 10,674 tokens or with the caches marked full to 32,768 or 131,072, the fastest of 40, ms. Before is the pass without the attention, router and visit-end changes above:

| Rows | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10,674, FP8 weights and cache | 3.75 | 4.32 | 4.83 | 5.26 | 5.70 | 6.24 | 6.65 | 7.02 |
| Before | 4.11 | 4.83 | 5.42 | 5.93 | 6.49 | 7.16 | 7.71 | 8.17 |
| 32,768, FP8 weights and cache | 4.06 | 4.70 | 5.33 | 5.75 | 6.24 | 6.99 | 7.39 | 7.81 |
| Before | 4.76 | 5.57 | 6.22 | 6.73 | 7.34 | 8.23 | 8.72 | 9.24 |
| 10,674, BF16 weights and cache | 5.09 | 5.62 | 6.10 | 6.55 | 6.95 | 7.46 | 7.80 | 8.16 |
| Before | 5.38 | 6.02 | 6.66 | 7.19 | 7.75 | 8.38 | 8.89 | 9.29 |
| 32,768, BF16 weights and cache | 5.55 | 6.10 | 6.61 | 7.03 | 7.47 | 8.10 | 8.48 | 8.91 |
| Before | 5.90 | 6.63 | 7.42 | 7.97 | 8.58 | 9.41 | 9.89 | 10.47 |
| 131,072, BF16 weights, FP8 cache | 6.20 | 7.12 | 8.33 | 8.83 | 9.46 | 10.56 | 11.09 | 11.69 |
| Before | 8.55 | 9.75 | 10.47 | 11.15 | 11.92 | 13.53 | 14.19 | 14.78 |
| Bandwidth model | 3.82 | 4.49 | 5.03 | 5.49 | 5.92 | 6.31 | 6.68 | 7.02 |

A plain decode step takes 2.78 ms at context 0 and 3.31 ms at 32,768 with FP8 weights and cache, and 3.69 and 4.75 ms with BF16. The bandwidth model reads the BF16 dense weights once and 8 m^0.6 distinct experts per visit at 84% of 1.70 TB/s, without KV.

Kernel traces of `bench --verify-widths` with FP8 weights and cache, after a 30-token chat prompt and with the caches marked full to 32,768, us per pass with each kernel counted from the previous one's end, before and after the changes above:

| Operation | 30, 2 rows | 30, 8 rows | 32,768, 2 rows | 32,768, 8 rows |
| --- | ---: | ---: | ---: | ---: |
| Attention: post-processing, attention, combine | 648 / 411 | 1,029 / 708 | 1,664 / 880 | 2,494 / 1,681 |
| Router | 169 / 118 | 554 / 241 | 178 / 122 | 608 / 286 |
| The visit's last kernel | 137 / 100 | 361 / 95 | 147 / 105 | 400 / 100 |
| Experts | 2,129 / 2,139 | 4,539 / 4,544 | 2,159 / 2,177 | 4,614 / 4,625 |
| Dense projections and LM head | 757 / 739 | 866 / 854 | 809 / 817 | 976 / 938 |
| Message gates | 433 / 437 | 458 / 455 | 459 / 460 | 482 / 482 |
| Depth attention | 327 / 328 | 331 / 325 | 330 / 328 | 336 / 330 |
| Pass span | 4,656 / 4,304 | 8,198 / 7,271 | 5,772 / 4,906 | 9,923 / 8,474 |

At 32,768 an FP8 global attention call takes 58 us at 2 rows and 124 us at 8, against 130 and 190 before and 57 us for decode's kernel at one row. At 8 rows its FMA chains and the broadcast query loads from shared memory bound it. What remained at 2 rows was mostly the message gates, then three kernels per gate (norm, down projection, gate), and the visit's last kernel beside the expert kernel, which decode folds into its expert kernel.

`bench --spec`, 256 tokens after each text, FP8 global KV cache, tok/s. Before is the commit before the adaptive width and the fast draft, with the exact draft at a fixed k; after is `--spec auto` with the fast draft, the faster of two runs each:

| Text, context | Weights | Plain decode | Before, k = 3 / 5 / 7 | After, auto | Tokens and verify rows per round |
| --- | --- | ---: | --- | ---: | --- |
| `anyhow`, 10,674 | BF16 | 250 | 390 / 487 / 515 | 558 | 6.14, 7.36 |
| `anyhow`, 32,768 | BF16 | 229 | 360 / 435 / 475 | 521 | 6.29, 7.39 |
| `anyhow`, 10,674 | FP8 | 313 | 437 / 520 / 575 | 624 | 6.29, 7.37 |
| `anyhow`, 32,768 | FP8 | 289 | 391 / 471 / 506 | 523 | 5.82, 7.33 |
| `half`, 2,273 | BF16 | 260 | 362 / 418 / 442 | 506 | 5.35, 6.98 |
| `half`, 32,768 | BF16 | 229 | 326 / 372 / 376 | 421 | 5.16, 7.26 |
| `half`, 2,273 | FP8 | 325 | 400 / 468 / 473 | 568 | 5.47, 6.94 |
| `half`, 32,768 | FP8 | 289 | 337 / 370 / 366 | 406 | 4.41, 6.62 |
| `regex-readme`, 2,286 | BF16 | 259 | 290 / 271 / 258 | 326 | 3.08, 5.17 |
| `regex-readme`, 32,768 | BF16 | 229 | 278 / 286 / 270 | 307 | 3.49, 6.01 |
| `regex-readme`, 2,286 | FP8 | 325 | 299 / 283 / 264 | 343 | 2.86, 4.88 |
| `regex-readme`, 32,768 | FP8 | 289 | 301 / 293 / 285 | 334 | 3.44, 5.77 |

The same runs before and after the attention, router and visit-end changes above, `--spec auto`, one run each:

| Text, context | Weights | Plain decode | Before | After | Tokens and verify rows per round, after |
| --- | --- | ---: | ---: | ---: | --- |
| `anyhow`, 10,674 | FP8 | 327 | 679 | 757 (2.31x) | 6.12, 7.21 |
| `anyhow`, 32,768 | FP8 | 302 | 574 | 661 (2.19x) | 5.82, 7.24 |
| `half`, 2,273 | FP8 | 340 | 640 | 704 (2.07x) | 5.35, 6.65 |
| `half`, 32,768 | FP8 | 302 | 455 | 524 (1.74x) | 4.49, 6.47 |
| `regex-readme`, 2,286 | FP8 | 340 | 405 | 438 (1.29x) | 2.84, 4.77 |
| `regex-readme`, 32,768 | FP8 | 302 | 372 | 429 (1.42x) | 3.44, 5.60 |
| `anyhow`, 10,674 | BF16 | 240 | 601 | 668 (2.78x) | 6.34, 7.61 |
| `anyhow`, 32,768 | BF16 | 210 | 551 | 633 (3.01x) | 6.40, 7.35 |
| `half`, 2,273 | BF16 | 257 | 564 | 614 (2.39x) | 5.57, 7.07 |
| `half`, 32,768 | BF16 | 210 | 442 | 497 (2.37x) | 4.92, 6.77 |
| `regex-readme`, 2,286 | BF16 | 256 | 379 | 407 (1.59x) | 3.37, 5.72 |
| `regex-readme`, 32,768 | BF16 | 210 | 288 | 323 (1.54x) | 2.87, 5.24 |

The context of 32,768 puts arbitrary token ids before the text. The greedy continuations of these texts repeat themselves, so the draft is right far more often than on chat: vLLM measured 58%, 33% and 20% there. At low acceptance the policy verifies 2 to 4 rows. With `--fallback` it falls back to plain steps on free prose (see below).

### Falling back to plain steps

On free prose the draft is wrong too often for any width to pay: a round of one row takes 4.6 ms for one token, against 2.8 ms for a plain step. `price` also times a plain step and the draft round that follows a run of them, and the policy then falls back to plain steps (`WidthPolicy::with_plain`). It is off by default; `bench --spec auto --fallback` and `siphon-serve --fallback` turn it on (see below for why):

1. `Model::step_with_taps` runs a step that also copies the draft's taps of its row, through its own CUDA graph. The copies are four `copy_tap` launches of one row each. A step takes 2.941 ms with them against 2.935 ms without at context 2,273, and 3.318 against 3.317 ms at 32,768.
2. The speculator keeps each step's taps. After a run of steps the draft commits all of them at once (`Draft::commit`), notes the tokens for its shortlist, and runs its round from the new anchor. That probe costs a draft round: 0.55 ms with the bundled draft and 0.86 ms with DFlash 2 at short context.
3. The policy picks a run of `r` plain steps when `r` steps and the probe after them, `r * step + probe` ms for `r` tokens, beat the best width's expected tokens per ms. A run starts at one step, since a step and a probe (3.3 ms) already beat a verify of one row (4.6 ms). It doubles while the probes keep choosing plain steps, up to 32, and starts again at one once a verify wins. Longer runs need a larger loss before the speculator leaves the verifies, which gives the switch its hysteresis.

The fallback decides by calibrated confidences. On prose the bundled draft's confidences run high: with the raw confidences, the essay's verifies kept 2.18 tokens per round at about 6.5 ms, slower than plain steps, and the essay ran at 329 tok/s against 355 for plain decode (before the counted shortlist, whose base ran it at 307). The speculator counts, for each tenth of the confidence range, the drafts the target kept against those it verified in this generation. The plain-or-verify test uses each bin's share kept, starting from the confidence itself with a weight of 4 drafts. The width still comes from the raw confidences, so a round verifies what it would without the fallback. Variants that lost or tied, at 512 tokens:

- A first run of 4 steps, and probes left out of the price, before the counted shortlist: the essay at 339 tok/s against 306 for its base, and chat with the bundled draft at 446 against 465.
- One share of the expected drafts kept over the last 20 or so rounds: it stayed low after a stretch of new tokens, into the repetitive stretch after it, and runs of plain steps never brought it back up. Code with the bundled draft fell from 501 to 451 tok/s, also with the share decaying back during plain steps.
- Choosing the width by the calibrated confidences too: within noise of the kept version (geometric mean +2.6% against +2.2% over the 16 runs below), with the essay at 378 tok/s instead of 356 but DFlash 2 slower on chat and the story.

Plain steps sum in other orders than the verify pass, so a generation that falls back no longer equals `greedy_by_verify`. It equals greedy decoding along its own path, a plain step for each token it stepped and a verify of one row for each other token (`spec::greedy_by_path` and `SpecStats::stepped`). Where the fallback steps depends on the measured prices, so the same request can come out differently on two servers at a near-tie. With fixed widths, or without plain prices, the output equals `greedy_by_verify` as before.

`bench --spec auto`, FP8 weights and cache, 512 tokens after each prompt, tok/s and tokens per verify round. The base is main before the fallback, which a run without `--fallback` matches; a generation that falls back drifts from the base's text at a near-tie after its first plain step, so each comparison after that point is on a different continuation of the same prompt:

| Prompt, tokens | Plain decode | Bundled, base | Bundled, fallback | Plain steps, probes | DFlash 2, base | DFlash 2, fallback | Plain steps, probes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Essay, 34 | 355 | 331 (1.92) | 356 (2.39) | 292, 41 | 399 (2.55) | 414 (2.88) | 52, 21 |
| Story, 42 | 356 | 317 (1.74) | 354 (2.29) | 363, 66 | 392 (2.37) | 418 (2.74) | 50, 32 |
| Chat question, 30 | 357 | 491 (3.21) | 536 (3.84) | 28, 15 | 543 (3.98) | 588 (4.53) | 15, 9 |
| Coding request, 40 | 356 | 501 (3.33) | 534 (3.71) | 27, 15 | 594 (4.23) | 546 (4.14) | 23, 15 |
| Document question, 32,765 | 301 | 554 (4.79) | 560 (4.83) | 1, 1 | 376 (3.15) | 377 (3.14) | 19, 13 |
| `anyhow`, 10,674 | 324 | 799 (6.67) | 785 (6.65) | 1, 1 | 783 (6.83) | 771 (6.82) | 1, 1 |
| `half`, 2,273 | 340 | 726 (5.67) | 726 (5.78) | 2, 2 | 596 (4.85) | 605 (5.05) | 4, 3 |
| `regex-readme`, 2,286 | 340 | 528 (3.63) | 486 (3.41) | 17, 15 | 534 (4.03) | 538 (4.08) | 3, 3 |

The geometric mean over the 16 runs is 2.2% above the base. Where a run took few plain steps, its swing against the base comes mostly from the different continuation: `regex-readme` with the bundled draft took 17 plain steps and 15 probes, at most about 2.5% of its time, yet its verify rounds kept 3.41 tokens instead of 3.63. At 2,048 tokens the same cases come out:

| Prompt, draft | Plain decode | Base | Fallback | Plain steps, probes |
| --- | ---: | ---: | ---: | ---: |
| `regex-readme`, bundled | 323 | 529 (5.17) | 659 (5.38) | 17, 15 |
| Coding request, bundled | 347 | 635 (4.59) | 613 (5.42) | 132, 58 |
| Coding request, DFlash 2 | 347 | 700 (5.40) | 687 (5.50) | 35, 26 |
| Chat question, bundled | 348 | 439 (3.02) | 402 (2.94) | 298, 172 |
| Essay, bundled | 347 | 357 (2.07) | 397 (3.22) | 1,063, 189 |

Through `siphon-serve` (FP8 weights and cache, `--context 16384`), a request for a 1,200-word essay at reasoning effort low, job log lines:

| Draft | Base | Fallback |
| --- | --- | --- |
| Bundled | 1,623 tokens in 5.27 s (307.8 tok/s, 1.78 tokens per round) | 1,430 tokens in 4.16 s (343.9 tok/s, 2.07 tokens per round, 846 plain steps, 167 probes) |
| DFlash 2 | 1,623 tokens in 4.39 s (369.3 tok/s, 2.24 tokens per round) | 1,436 tokens in 3.72 s (385.6 tok/s, 2.62 tokens per round, 315 plain steps, 119 probes) |

The fallback is opt-in for two reasons, measured on one build (FP8 weights and cache, 512 tokens). First, where it steps depends on the step and round times `price` measures at startup, so two runs of the same build can take different paths, and a different path can resolve a near-tie differently: the chat question took 54 plain steps in one run and 28 in the next. Second, it costs the coding request about 4% with both drafters, repeatably:

| Prompt, draft | Fallback, two runs | Without |
| --- | ---: | ---: |
| Chat question, bundled | 514, 542 | 488 |
| Chat question, DFlash 2 | 553, 573 | 550 |
| Coding request, bundled | 477, 482 | 498 |
| Coding request, DFlash 2 | 571, 572 | 594 |
| `regex-readme`, bundled | 486, 535 | 528 |

Fixed prices instead of measured ones would make the paths reproducible; the coding request's loss needs a better plain-or-verify test.

## Sampling

`ops::sample` picks each row's next token from its F32 logits wherever the engine picked the argmax: after a prefill or a plain step, in the verify pass and in batched steps. `Model::set_sampling` sets slot 0's `Sampling`, which the prefill, the steps and the verify use; `Model::step_batch_sampled` takes one per row. The verify and batch heads sample only after `Model::enable_sampling`, which siphon-serve calls unless `--greedy` is set. Temperature 0 takes the argmax with the argmax kernel's rule (the first maximum, NaN above every number), so greedy rows give the same tokens.

The filters follow vLLM, in its order: temperature, then top-k (every token at least as large as the k-th largest stays), then top-p (a token stays while the strictly larger tokens hold less than p of the mass; equal logits stay or go together), then min-p (a token stays at min_p times the largest probability or more). Then one draw by the exponential race: token i scores x_i / T - ln E_i, with E_i an Exp(1) variable hashed from the request's seed, the token's position and i, and the kept token with the highest score is a draw from the filtered distribution. A token therefore depends only on the logits, the seed and the position: not on the other rows of a pass, the width or the reduction order.

### Speculative sampling

The draft proposes one token per position, so its proposal distribution q is a point mass on its draft x. The standard rule keeps x with probability min(1, p(x) / q(x)) = p(x), and after a rejection draws from normalize(max(0, p - q)), which is p without x. The verify draws each row's token y from p with the row's seed and position, and keeps the draft when x = y. That keeps x with probability p(x), and after a rejection y is a draw from p conditioned on y != x: the same rule, coupled to the plain draw at that position. So:

- The output follows the target's filtered distribution exactly, and the acceptance kernel is the greedy one (`accept_greedy`), comparing drafts with drawn tokens.
- How the draft picks x, its shortlist and its approximations change only how often drafts are kept, never the tokens.
- With a seed, speculation gives the tokens of batched steps, whose rows have the verify's logits bit for bit. A seeded request gets the same tokens alone or in a batch, on any width schedule. Plain steps (`--fallback`) sum in other orders, so a run that falls back follows the distribution but not these exact tokens.

The draft picks x by the target's own race. Draft k of a round, k from 1, goes to verify row k and is kept when it equals the draw of row k - 1, which takes position P + k for the anchor's position P. The draft takes the best of l_i / T - ln E_i over its own logits l_i, with the E_i that the verify's draw uses for that seed and position (`kernels/race.cuh`, which `sample.cu` and `draft.cu` both include). When the draft's distribution q equals p, its pick is the target's draw. In general it is kept with probability at least (1 - TV) / (1 + TV) for the total variation distance TV between p and q, where 1 - TV = sum(min(p, q)) is what a draft sampled independently from q keeps; the draft's argmax is kept with probability p(x). Its pick depends only on the seed, the position and the draft's logits, so the output is still the target's draw and a seed still reproduces it on any width schedule or batch.

- Positions and settings. `Draft::set_sampling` writes the temperature and seed to `RoundInputs::race` on the device, so the captured round graphs replay with them; temperature 0 keeps the argmax. Each step's kernel computes P = control[0] + control[1] from the round's control, the same P that `Model::start_verify` starts the pass at. The speculator gives the draft the model's sampling at the start of each generation when the verify samples (`Model::enable_sampling`).
- Bundled draft. The Markov head's argmax pass compares noisy scores, l * (1 / T) + boost, rounded after each operation as `sample_race` computes them, and skips the hash for rows that cannot win even with the largest noise. The fast head (shortlist and E4M3 fold, `draft_markov_fold_any`) and the recorded-logits fold scan every row, so the noise adds a hash per row. The exact pruned argmax keeps its pruning: `draft_markov_bound` widens each row's bounds by a BF16 step, then scales them and adds the row's noise, so they bound the row's noisy score, and `draft_markov_exact` computes the noisy score of each row left in the running. The pick is the one a full scan finds.
- DFlash 2. The selector's walk picks the best s / T plus the noise over its row's 16 candidates, the top 16 logits of the LM head, so a target draw outside them cannot be proposed. 2DSD's DFlash drafts stay greedy.
- Filters. The draft ignores top-k, top-p and min-p: it can propose a token that the verify filters out, which is then never kept. With top-p 0.95 the filtered tokens hold 5% of p's mass, so when q is close to p about 5% of proposals land there.
- Confidence. A coupled draft reports the largest probability under softmax(l), at temperature 1 whatever the request's, as the greedy draft does. The largest probability under softmax(l / T) overstates how often the target keeps the draft below temperature 1. At 0.8 with top-p 0.95 it read 0.80 to 0.70 by position, against 0.71 to 0.42 kept, and the width policy verified too many rows: on the 12 registry prompts the rounds kept 2.60 tokens but ran at 431 tok/s, against 438 for the argmax draft. With softmax(l), 0.74 to 0.62, the rounds keep 2.55 tokens and run at 445 tok/s. At temperature 1 the two are the same. The coupled pick's own probability, 0.63 to 0.55 at temperature 1, would understate it. `draft_sampling_acceptance` prints all three.

`tests/sampling.rs` (`draft_sampling_acceptance`) measures the three proposals from the same rounds of the exact bundled draft, whose logits it records, over three prompts and 192 tokens each. The argmax is kept with probability p(x), a draft sampled from its own filtered distribution with sum(min(p, q)), and the coupled draft when the host's race (`siphon::race`) over the draft's logits finds the target's draw, over 32 seeds per position. Before the coupled draft, the rounds' drafts were the greedy ones, and positions after the first conditioned on them:

| | Greedy draft kept, by position | Tokens per 8-row round | Sampled draft kept, by position | Tokens per 8-row round |
| --- | --- | ---: | --- | ---: |
| Temperature 1 | 0.57, 0.49, 0.49, 0.45, 0.41, 0.37, 0.36 | 2.50 (measured 2.54) | 0.68, 0.58, 0.55, 0.51, 0.47, 0.43, 0.41 | 2.79 |
| Temperature 0.8, top-p 0.95 | 0.68, 0.57, 0.51, 0.46, 0.46, 0.41, 0.46 | 2.89 (measured 2.90) | 0.72, 0.62, 0.56, 0.49, 0.49, 0.46, 0.52 | 3.05 |

With the coupled draft in the rounds, all three columns condition on the coupled drafts drawn with the run's seed, and the other seeds reuse those rounds (198 rounds at temperature 1, 192 at 0.8). The coupled draft recovers most of the gap between the argmax and an independent sample from q:

| | Argmax kept, by position | Tokens per round | Sampled from q kept | Tokens per round | Coupled kept, 32 seeds | Tokens per round |
| --- | --- | ---: | --- | ---: | --- | ---: |
| Temperature 1 | 0.59, 0.52, 0.47, 0.40, 0.40, 0.36, 0.35 | 2.60 | 0.68, 0.59, 0.53, 0.46, 0.46, 0.42, 0.43 | 2.83 | 0.67, 0.56, 0.51, 0.44, 0.44, 0.40, 0.41 | 2.75 |
| Temperature 0.8, top-p 0.95 | 0.69, 0.58, 0.50, 0.43, 0.43, 0.40, 0.44 | 2.89 | 0.73, 0.62, 0.54, 0.46, 0.47, 0.44, 0.49 | 3.04 | 0.71, 0.61, 0.51, 0.44, 0.44, 0.42, 0.46 | 2.96 |

The host's race and the GPU's agreed on every draft pick and draw, over 1,386 and 1,344 positions. The pick's own probability averaged 0.55 to 0.63 at temperature 1, against 0.66 to 0.72 for the argmax's.

`coupled_draft_acceptance` runs speculation end to end with the argmax and with the coupled pick, FP8, on the same prompts and seeds:

| Draft | Sampling | Argmax: tokens per round | Coupled: tokens per round | Draft time per round |
| --- | --- | ---: | ---: | ---: |
| Bundled, fast head | Temperature 1 | 2.42 | 2.78 | 0.525 → 0.532 ms |
| Bundled, fast head | Temperature 0.8, top-p 0.95 | 3.07 | 3.18 | 0.526 → 0.532 ms |
| DFlash 2 | Temperature 1 | 2.51 | 2.82 | 0.802 → 0.804 ms |
| DFlash 2 | Temperature 0.8, top-p 0.95 | 3.14 | 3.27 | 0.802 → 0.805 ms |

The first draft is kept 63% of the time instead of 53% at temperature 1. The gain is smaller at temperature 0.8, where p is sharper and its argmax already holds more of the mass. In `bench` at temperature 1, the median of 6 seeds rose by 0.1 to 12% over the argmax draft across the 16 prompt, context and drafter cases, with the same tokens for each seed. Through siphon-serve, on the 12 registry prompts (3,000 tokens at most, two runs of each build in one hold, identical replies), decode went from 397 to 416 tok/s at temperature 1 and from 438 to 445 tok/s at 0.8 with top-p 0.95.

A draft that samples from q independently of the target's noise would need q over the vocabulary for each draft, which the pruned head and DFlash 2's 16 candidates do not have, and its draws would tie a request's tokens to the width schedule and to batching.

### Kernels

`sample_prepare` runs a cluster of 8 blocks per row, each taking an eighth of the row, and merges their histograms through distributed shared memory. A greedy row takes the argmax there. A row with top-k or top-p finds the row's maximum, then each filter's threshold: a histogram of the tokens by distance below the maximum in 2,048 bins over 28.5 nats, then a radix over the ordered bits of the logits in the bin where the weight reaches the target, 11 bits at a time, over a shared-memory copy of that bin. Top-k weighs each token 1; top-p weighs it by its probability in fixed point (2^40 for the largest token), summed in 32-bit integer atomics as three 14-bit limbs, so the threshold does not depend on the order of the adds. `sample_race` runs the race over the kept tokens in 32 blocks per row, skipping tokens that cannot beat the best score so far, and the last block to finish reduces the partials.

Kernel time from `tests/sample.rs` (`sampling_cost`), 8 rows of 200,058 synthetic logits, launch included: argmax kernel 10.0 us; greedy rows 4.8 us; temperature 1, 13.0 us; temperature 0.8 with top-p 0.95, 42 us; temperature 0.7 with top-k 40 and top-p 0.95, 64 us. A single block per row took 77 us for top-p: it was bound by issuing about 50 branchy instructions per token on one SM. 64-bit shared-memory atomics compile to compare-and-swap loops on sm_120, and 64-bit float conversions run at the FP64 rate, so the kernels avoid both. In the verify pass on the model's logits (`sampling_cost_in_the_verify`, FP8, best of 3 runs of 160 tokens), the costs at widths 2 and 4 are within the run-to-run noise of about 0.03 ms. At width 8 they are larger, against 6.58 ms greedy:

| Width | Greedy | Temperature 1 | Temperature 0.8, top-p 0.95 | Temperature 0.7, top-k 40, top-p 0.95 |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 3.876 ms | 3.856 | 3.889 | 3.937 |
| 4 | 4.917 | 4.891 | 4.919 | 4.919 |
| 8 | 6.584 | 6.655 | 6.681 | 6.769 |

A plain step samples after its graph: 3.008 ms greedy, 3.009 at temperature 1, 3.042 with top-p and 3.069 with top-k and top-p.

siphon-serve on the 12 prompts of the registry runs (3,000 tokens at most, FP8, one request at a time), against the greedy build before sampling (097e99f, 537.5 tok/s and 3.50 tokens per round):

| Sampling | Decode tok/s | Tokens per round | Story prompt |
| --- | ---: | ---: | --- |
| Temperature 0 | 537.5 | 3.48 | 549 tok/s, 3,000 tokens |
| `--greedy` | 536.4 | 3.48 | 548 tok/s |
| Temperature 1 | 396.3 | 2.21 | 318 tok/s, 1,539 tokens, 1.53 tokens per round |
| Temperature 0.8, top-p 0.95 | 438.4 | 2.51 | 340 tok/s, 1,536 tokens, 1.72 tokens per round |

With `--streams 8` and 8 requests at a time, the 12 prompts ran at 780 tok/s greedy (779 before), 734 at temperature 1 and 768 at 0.8 with top-p 0.95, wall clock. Temperature 0, `--greedy` and the batched greedy run gave the earlier greedy outputs for all 12 prompts. Each sampled run with seeds gave the same 12 outputs again, alone and with `--streams 8`.

`tests/sample.rs` checks greedy rows against the argmax kernel, draws against a host version of vLLM's filters by a chi-square test (200,000 draws of a 1,000-token row per setting, 20,000 of a 200,058-token row), and that a row's token depends only on its logits, seed and position. `tests/sampling.rs` (`speculation_keeps_the_distribution`) compares speculative and plain sampling of the first 4 tokens after a prompt over many seeds, at temperature 1 and at 0.8 with top-p 0.95, by two-sample chi-square tests, and checks that speculation and batched steps give the same 48 tokens for 12 seeds:

```sh
cargo test --release -p siphon-cuda --test sample
SIPHON_MODEL=<model-dir> cargo test --release -p siphon-cuda --test sampling -- --ignored --nocapture --test-threads 1
```

With 1,000 seeds per sample, on the 260-token prefix of `regex-readme` (next-token entropy 3.98 nats), FP8:

| Test | Temperature 1, z (degrees of freedom) | Temperature 0.8, top-p 0.95, z (degrees of freedom) |
| --- | ---: | ---: |
| Token 1 | 0.52 (27) | -0.12 (24) |
| Token 2 | -0.62 (22) | 0.42 (16) |
| Token 3 | 0.86 (31) | -0.79 (24) |
| Token 4 | -1.65 (35) | 1.29 (34) |
| Tokens 1-2 | -0.80 (27) | 0.63 (25) |
| Tokens 1-4 | -0.50 (14) | 1.01 (37) |

No z-score comes near the 4.5 limit. Speculation kept 1.59 and 1.73 tokens per round over those 4-token draws, and gave the batched steps' 48 tokens for all 12 seeds at both settings.

## Batched decode

`Model::add_slots` adds sequence slots, each with its own global caches, sliding rings and conv states of slot 0's sizes. `Model::prefill_slot` prefills a slot, and `Model::step_batch` runs the next token of 1 to 32 slots as one pass through the verify pass's code, with each row reading and writing its own slot through a table of device addresses per visit. Each row's logits equal, bit for bit, a one-row verify of its sequence, so a sequence decoded in batches emits the tokens of speculation, greedy or sampled with the same seed. [batching.md](batching.md) has the design, the kernel changes and the measurements.

siphon-serve returns a request left alone in a batch to speculation. The draft follows the request that left speculation: batched steps record their rows' taps, and the speculator commits the followed row's (`Speculator::follow`, `keep_batched`, `resume`). A request without draft state gets it rebuilt by a prefill with taps into a free slot (`Speculator::rebuild`), and `Model::move_slot` moves a request to slot 0. [batching.md](batching.md#returning-to-speculation) has the costs.

`tests/batch.rs` decodes five prompts of 40 to 2,500 tokens that join the batch at different steps, and checks every token and every row's logits against one-row verifies, with BF16 and FP8 weights and caches. Two more tests speculate, batch and speculate again, with the bundled draft and with the drafter in `SIPHON_DFLASH`, and check every token:

```sh
SIPHON_MODEL=<model-dir> cargo test --release -p siphon-cuda --test batch -- --ignored --nocapture --test-threads 1
```

`bench --batch 1,2,4,8 --prompt <capture>` times batched steps at each context against a plain step and a one-row verify. Row r feeds the capture's tokens from offset r / 8 of its length on, so the rows route their experts as different texts do: 8 rows from 8 texts touch about 50 distinct experts per visit, against 29 for 8 consecutive rows of one text.

## Prefix cache

Chat clients and agents resend the whole conversation each turn. Without a cache, a follow-up turn on 90,607 tokens of history prefills all of them again in 7.6 s. HTTP, the chat template and the tokenizer take about 37 ms of a 97K-token request.

The speculator's prefix cache (`crates/siphon-cuda/src/spec/prefix.rs`) keeps the state of recent prompts, one entry per sequence slot, and a prompt that starts with an entry's tokens prefills only the rest. The state of a sequence at position `p` in its slot is:

| State | After later tokens |
| --- | --- |
| Global KV caches of the slot, slots below `p` | Unchanged: later tokens write slots from their own position on |
| Sliding rings and conv states of the slot | Overwritten |
| The draft's global caches, positions below `p` | Unchanged: commits and rounds write from the committed position on |
| The draft's rings | Overwritten |
| The draft's Markov context, the shortlist's noted tokens, the speculator's prompt tail | Rebuilt by every prefill |

So an entry copies only what later tokens overwrite: the slot's rings and conv states (`Model::save_local_state`, 164 MiB) and the draft's rings (`Draft::save_rings`, 8 MiB for the bundled draft, 40 MiB for DFlash 2, none for a 2DSD DFlash draft). The global caches stay in the slot. A hit restores the copies into the slot, and the draft starts committing from the cached position. A miss resets a slot and prefills from position 0.

Prefill sums differ with the chunk boundaries, so the copies are taken at a chunk boundary: the last multiple of the prefill chunk (2,048) below the prompt's end, so at least one row remains for the logits. The speculator prefills up to that boundary, saves the copies and prefills the rest. Prefill splits chunks only at the end of its tokens, so both halves run the chunks of one prefill from position 0, and the only extra work is the first half's last-row logits. A later prompt that starts with the cached tokens restores them and prefills from the boundary in the same chunks. Its caches, logits, tokens and drafts equal those of a prefill from position 0 bit for bit. The cost is that a follow-up prefills up to 2,047 tokens of the previous prompt again, plus the previous reply and its own new tokens.

Entries match prompts by key, not by tokens (`spec::Prompt`). A prompt of text is its own key. Two prompts that differ only in an image have the same placeholder tokens, so siphon-serve replaces each image's placeholders in the key with ids above the vocabulary, 31 bits of the image's SHA-256 each (placeholder `k` takes word `k % 8`), mixed with the pixel limit. The cache saves and compares only keys, on the device, in the host tier and in `--cache-dir`'s files, so an entry serves a prompt only when every image below its boundary is the same. `prompts_that_differ_only_in_an_image_do_not_share_an_entry` in `tests/images.rs` runs two prompts with the same tokens and different images through one entry: each misses the other's entry and gives a cold prefill's logits, tokens and acceptance, and hits its own. As a negative control, the second prompt under the first's key hits the first's entry and its first token's logits differ from its own cold prefill.

An entry belongs to the slot that holds its global caches. It stays live while the slot holds the same sequence: every slot's sequence has a number no other sequence has had (`Model::slot_sequence`), and `Model::reset`, `restore_state` and a `move_slot` into a slot give it a new one. Anything that starts another sequence in a slot so drops its entry, without the cache's help. An entry also dies when unused for its time to live.

The draft keeps one sequence, so not every entry can restore it. An entry holds the draft's rings only when its prompt was prefilled with the draft: alone, in slot 0, or by a rebuild (below). The bundled draft and 2DSD DFlash drafts also have global layers, which hold the positions of the last sequence the draft started from position 0. Their rings are of use only while that sequence is the entry's, which the speculator tracks with a number of its own. DFlash 2 has rings alone, so any entry's rings restore it.

In siphon-serve:

- A batched request takes the slot of the longest live entry its prompt starts with, restores the entry's copies there and prefills from the entry's boundary (`Speculator::claim_slot`). Otherwise it takes a slot without a live entry, slot 0 last, and else the slot of the least recently used entry. Its prefill stops at its own boundary to save its slot's state into the slot's entry (`Speculator::save_prefix`), without the draft's state.
- A request that runs alone, with an entry that holds the draft's state, speculates in slot 0: the entry moves to slot 0 if it is elsewhere, and slot 0's own entry first moves to a slot without one (`Speculator::move_to_slot0`, a copy of 16 KiB per position with an FP8 cache). Without any entry, slot 0's entry moves aside the same way and slot 0 prefills from position 0. Either way slot 0's entry then takes the prompt's state, with the draft's rings (`Speculator::prefill_cached`).
- A request that runs alone with an entry that lacks the draft's state prefills from that entry as a batch of one in the entry's slot. Once it has its first token it returns to speculation as any request left alone by a batch does (see [batching.md](batching.md)): it moves to slot 0 and, when enough tokens remain, rebuilds the draft's state by prefilling its tokens again with taps in an idle slot. The rebuild stops at the entry's boundary to give the entry the draft's rings, so the next turn of the same conversation speculates from the cache. Rebuilding is a whole prefill with taps, so the cache saves this request only its time to first token; when too few tokens remain to rebuild, it decodes in one-row batched steps without the prefill.

`--prefix-cache-entries` sets the number of entries, 1 to `--streams`, and `--streams` by default. Fewer entries than slots work the same way: a slot without an entry is free, and a new prompt's state goes to an entry that is not live, else the least recently used one. Each entry takes 172 MiB with the bundled draft and 204 MiB with DFlash 2, whatever the context. The largest FP8/FP8 context per stream on an idle 32 GB RTX 5090, from the memory planner:

| Streams | Bundled, 1 entry | Bundled, an entry per stream | DFlash 2, 1 entry | DFlash 2, an entry per stream |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 633,856 | 633,856 | 701,440 | 701,440 |
| 4 | 163,840 | 155,648 | 165,888 | 156,672 |
| 8 | 77,824 | 68,608 | 77,824 | 66,560 |

`tests/prefix.rs` checks cached prefills against prefills from position 0 with the bundled draft, the 2DSD drafter in `SIPHON_DFLASH` and DFlash 2 in `SIPHON_DFLASH2`: the first token's logits, 97 speculated tokens and the draft's acceptance per round, or 17 batched tokens, must all match bit for bit. `prefix_cache_matches_prefill` covers a miss, a hit, a prompt that diverges after the cached boundary, an exact repeat whose remainder is shorter than the 1,024-slot window, a batch beside slot 0, a batch that takes slot 0 and a rebuild. `prefix_cache_entries` runs three conversations of three turns each with four slots: lone turns that alternate between them and each hit their own entry in another slot (the bundled draft through a batch of one and a rebuild, DFlash 2 by moving the entry and its draft rings to slot 0), the same prompts batched with 456-token remainders, a new conversation batched beside a hit, eviction of the least recently used entry, expiry, a slot taken by another conversation and a reset slot. Three negative controls fail: skipping the restore of the slot's rings and conv states changes the first token's logits of the first batched hit; skipping the draft's rings but keeping its committed position changes the bundled draft's rounds from 16 to 43 with the same tokens, and DFlash 2's tokens on the 456-token remainder; skipping the draft's restore whole stops at the draft's check that commits leave no gap.

```sh
SIPHON_MODEL=<model-dir> cargo test --release -p siphon-cuda --test prefix -- --ignored --nocapture --test-threads 1
```

Time to the first token of a follow-up chat turn through the HTTP API, FP8 weights and cache, `--context 131072`, the bundled draft, under the GPU lock with no other compute process. The history is the repo's source files as user messages, each with a short assistant reply; the follow-up adds the reply to a question and a new question. "Without" is the same request right after a short unrelated request, which empties the cache. The median of three repetitions, which spread by at most 35 ms; the cached and uncached replies were identical. In the same holds, a server with `--no-prefix-cache` took 0.53 s and 7.66 to 7.70 s for the 8,174 and 90,607 rows, so a miss costs nothing extra:

| Follow-up prompt | Cached tokens | Without | With | Server prefill with |
| ---: | ---: | ---: | ---: | ---: |
| 8,174 | 6,144 | 0.53 s | 0.14 s | 2,030 tokens |
| 12,453 | 12,288 | 0.82 s | 0.044 s | 165 tokens |
| 90,607 | 90,112 | 7.63 s | 0.11 s | 495 tokens |
| 101,718 | 100,352 | 8.81 s | 0.20 s | 1,366 tokens |

Agent swarms run several conversations at once, and each turn of each one resends its history. A client ran 4 conversations against `--streams 4`, then 8 against `--streams 8`, all at once, with `--context 65536`, FP8 weights and cache, the bundled draft, under the GPU lock with no other compute process. Each conversation starts with about 15K tokens of the repo's files as a user message. Each turn asks for a reply of at most 384 tokens at temperature 1 with a seed fixed by conversation and turn, at `reasoning_effort` low, and the next turn adds the reply, with its reasoning, and about 4,700 tokens of other files, after 0.5 s. Nine turns take each history to about 53K tokens. Replies averaged 161 and 169 tokens. Every reply was identical with the cache on and off, 36 of 36 and 72 of 72. Median time to first token over the conversations at each turn:

| Turn | Prompt | Cached | 4 streams, without | 4 streams, with | 8 streams, without | 8 streams, with |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 15,450 | 0 | 2.43 s | 2.42 s | 4.64 s | 4.50 s |
| 1 | 20,327 | 14,336 | 1.99 s | 0.52 s | 2.47 s | 0.65 s |
| 2 | 25,281 | 18,432 | 2.45 s | 0.45 s | 3.99 s | 0.91 s |
| 3 | 29,944 | 24,576 | 3.95 s | 0.77 s | 6.39 s | 0.79 s |
| 4 | 34,742 | 28,672 | 6.15 s | 0.64 s | 7.87 s | 0.93 s |
| 5 | 39,256 | 33,792 | 6.63 s | 0.94 s | 5.19 s | 0.62 s |
| 6 | 44,144 | 38,912 | 4.61 s | 0.56 s | 6.70 s | 1.24 s |
| 7 | 48,429 | 43,008 | 8.33 s | 0.63 s | 7.15 s | 1.09 s |
| 8 | 53,246 | 47,104 | 5.10 s | 0.94 s | 9.80 s | 1.00 s |

The prompt and cached columns are the 8-stream run's medians; the 4-stream run's are within 1,000 tokens. The mean time to first token fell from 4.81 to 0.91 s at 4 streams and from 6.25 to 1.37 s at 8, and 80% of the prompt tokens came from the cache. All turns took 29.8 s instead of 96.4 s at 4 streams (195 against 60 tok/s of replies) and 52.8 s instead of 186.9 s at 8 (231 against 65 tok/s). With short replies the prefill dominates, so the reply rate moves with it. A new prompt prefills up to 2,047 tokens of the previous one again, plus the previous reply and its new tokens. The server's memory check stayed under the plan: a peak of 30,224 MiB against 30,261 MiB planned at 8 streams. A lone request in these runs hit an entry saved by a batch, and with 384 tokens left it decoded batched rather than rebuild the draft.

### Host tier

A device entry leaves the global KV caches in its slot, so it dies as soon as the slot starts another sequence. Conversations that outnumber the slots, such as several coding agents and their subagents on one GPU, then keep evicting each other, and each turn back prefills its whole history again: 2.1 s at 32K tokens and 39 s at 330K on an RTX PRO 6000 Blackwell Workstation Edition. `siphon-serve --host-cache GiB` adds a tier in page-locked host memory (`crates/siphon-cuda/src/spec/host.rs`) that copies all of an entry's state, global caches included, and returns it over PCIe.

The memory is one pool, page-locked at startup in slabs of up to 1 GiB (12 GiB took 9.7 s), and split into chunks of one size so that entries of any length use it without fragmenting it. A chunk holds 2,048 positions of every global cache of the target, then the same positions of the draft's global caches: 32 MiB plus 4 MiB for the bundled draft with FP8 caches, 32 MiB for DFlash 2, and 64 MiB plus 4 MiB with BF16 caches. 2,048 positions is the prefill chunk at whose multiples the prefix cache saves, so an entry's global caches fill whole chunks, and a chunk depends on the prompt's tokens up to its own end and on nothing after. An entry's rings and conv states (164 MiB) and the draft's rings (8 MiB bundled, 40 MiB DFlash 2) take a run of chunks of their own, 5 to 7 of them. An entry is its tokens, its global cache chunks and its local chunks.

Every time the prefix cache saves a device entry, the host tier copies it too. Two entries whose tokens agree through the end of a chunk hold the same bytes in it, since prefill runs the same chunks from position 0 whatever follows, so they share the chunk, counted by references. A new entry shares every chunk it can with the entry that agrees with it longest and copies only the rest, and an entry whose tokens start the new entry's is dropped. A conversation that grows thus copies only its new chunks and its latest rings and conv states each turn, and agents that start with the same system prompt share its chunks. When the pool is full, whole entries go, least recently used first; there is no time to live in host memory. An entry saved by a batched prefill has no draft state. When a lone request's rebuild gives the device entry the draft's rings, the host entry gets them too, and the draft's global caches in the chunks that lack them.

The copies to the host run on a stream of their own, beside serving. Each save waits for the model's stream first, so it reads what the prefill wrote, and records an event. The model's stream waits for that event, a wait on the GPU only, before anything that could overwrite what the copies read: a slot's reset or a move into it, a save into a device entry's copies, and the draft starting a new sequence. Copies from a slot read only positions below the entry's boundary, which later tokens of the same sequence do not write. At 330K tokens a save of 5.8 GiB overlaps the rest of the prefill and the first tokens of the reply.

A prompt looks for the longest prefix among the live device entries, as before, and the host entries; a host entry wins only when it is longer. A request that runs alone needs an entry with the draft's state: slot 0's entry moves aside, slot 0 is reset, and the host entry's global caches, rings and conv states go into slot 0 and its draft state into the draft, global caches included, so speculation resumes at once. A batched request takes a slot as before and gets the host entry without the draft's state, but for a draft of rings alone such as DFlash 2, whose rings come along. Either way the restored state also goes into a device entry, so the next turn of the conversation finds it on the device. Restores run on the model's stream, and the request then prefills only the rest, in the same chunks as a prefill from position 0, so its caches, logits, tokens and drafts equal a cold prefill's bit for bit.

`host_cache_matches_prefill` in `tests/prefix.rs` runs, with two slots and one device entry, conversations that keep taking the entry from each other, and checks every host restore against a prefill from position 0: the first token's logits, 97 speculated tokens and the draft's acceptance per round alone, 17 tokens batched. It covers remainders of 300 to 904 tokens past the restored boundary (shorter than the 1,024-slot window, where a stale ring shows), an exact repeat, an entry that extends another and shares its chunks, a conversation that shares two chunks with another, batched restores, and a batch of one whose rebuild then gives the host entry the draft's state, with the bundled draft and DFlash 2. Three negative controls fail at the first restore, with an 804-token remainder: skipping the rings and conv states changes the first token's logits, skipping the target's last global cache chunk changes them too, and skipping the draft's global caches keeps the logits but changes the draft's acceptance.

Time to the first token of a conversation that another one pushed off the GPU, through the HTTP API with `max_tokens` 1, on an RTX PRO 6000 Blackwell Workstation Edition (PCIe 4.0 x16). One server with `--host-cache 12` and one without, both `--streams 1 --context 400000 --greedy` with FP8 weights and caches and the bundled draft, ran side by side under the GPU lock, asked in turn one request at a time. Each gets conversation A, then B, which takes A's entry, then A with a reply and a new question, then B again, three times (twice at 330K). The prompts are the repo's files. Medians:

| Prompt | Cached | Without | With | Server prefill without | Server prefill with | Restore |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 31,878 | 30,720 | 2.13 s | 0.135 s | 2,100 ms | 102 ms | 711 MiB in 28 ms |
| 109,765 | 108,544 | 8.79 s | 0.317 s | 8,684 ms | 181 ms | 2,079 MiB in 80 ms |
| 331,197 | 329,728 | 38.9 s | 0.81 s | 38,694 ms | 456 ms | 5,967 MiB in 229 ms |

Restores ran at 26.5 to 27.4 GB/s. The rest of the time with the cache is the prefill of the 1,158 to 1,469 tokens past the boundary, and at 330K about 0.35 s of HTTP and tokenizing outside the engine. The tier does not slow the rest of the server: a device prefix cache hit prefilled in 71 to 74 ms against 77 ms at 31,916 tokens, 97 against 98 ms at 109,803 and 228 ms on both at 331,235, and 512 speculated tokens ran at 307.6 against 307.0 tok/s at 32K, 257.1 against 257.5 at 110K and 169.6 against 168.7 at 331K.

Pinned host pages land on the NUMA node of the thread that first touches them. On a two-socket host whose GPU hangs off one socket, restores from memory on the other socket ran at 15 GB/s against 26 GB/s. These runs pinned both servers to the GPU's node with `taskset`; a one-socket host needs nothing.

`siphon-serve --cache-dir` writes the host entries to files when the server stops and reads them back at startup (`crates/siphon-cuda/src/spec/host/disk.rs`, [server.md](server.md#restarts)). Each file holds one entry whole: its tokens, each global cache chunk's target part and, when the chunk holds it, its draft part, then the rings and conv states and the draft's rings, with a header that says which chunks hold the draft's caches. Chunks shared in memory are written into each entry's file; loading shares them again with the same rule as saving, and skips the bytes of chunks already in memory. A file says everything needed to read it into any pool of the same layout, so a disk tier that keeps entries past their time in memory could use the same files.

The server's round robin ([batching.md](batching.md#memory-one-pool-of-pages)) swaps running sequences out to the same pool (`HostCache::park`, `unpark`). A swapped-out sequence holds chunks of its global caches up to its position, the last one partly filled, and a run of chunks for its rings and conv states at that position. Only the whole chunks of its prompt can come from a prefill, so only they are shared with entries; the chunks with generated tokens hold the batched steps' values and are its own. LRU eviction drops entries, never swapped-out sequences, and their shared chunks stay counted. A sequence that does not fit stays on the GPU.

With FP8 caches and the bundled draft, a 12 GiB pool, the size that suits a 30 GB serving box with an RTX 5090, holds 341 chunks: 698K positions of global caches, less 5 chunks for each entry's rings and conv states. That is two conversations of 300K or ten of 50K, and more where they share a start.

## Benchmarks and profiling

The `bench` binary times greedy decode at several context lengths. For each length it marks every cache as full to that length without computing it, runs warmup steps, then times `--steps` steps. It prints one markdown table row per context length, or JSON with `--json`, along with the time, the GPU clocks and the processes on the GPU. Numbers are comparable only on an otherwise idle GPU.

```sh
pixi run cargo build --release -p siphon-cuda --bin bench
target/release/bench <model-dir> [--contexts 0,4096,32768] [--passes N] [--steps 100] [--warmup 10] [--kv bf16|fp8] [--weights bf16|fp8] [--json]
```

Without `--contexts` it runs 0, 4096 and 32768 tokens, and 131072 if the memory plan for that capacity fits in free memory (see [Memory planning](#memory-planning)). Bandwidth columns divide 5.5 GB of weights per token (3.75 GB with `--weights fp8`), plus the KV bytes read, by the step time, as a share of the RTX 5090's 1.79 TB/s.

With `--prefill`, `bench` times `Model::prefill` instead, once per prompt length from an empty cache, after one warmup chunk. It reports the time to the first token. The prompt is arbitrary token ids, so expert routing differs from real text:

```sh
target/release/bench <model-dir> --prefill 512,4096,32768 [--chunk 512] [--kv bf16|fp8] [--json]
```

## Memory planning

`siphon::memory` works out the device memory a configuration needs before any weight is loaded, from the config, the safetensors headers (`Checkpoint::open` maps the files and reads only the headers), the weight and KV formats, the draft, the prefill chunk, the number of streams and the capacity. With `PlanInputs::pooled`, as `siphon-serve` and `siphon-plan` plan, the global caches of every stream and of the draft, the rings and the prefix cache entries' copies of the rings share one pool of 2 MiB pages, and the capacity is the positions the pool holds over every stream ([batching.md](batching.md#memory-one-pool-of-pages)): the pool holds the capacity's chunks of 2,048 positions of the target's caches, the draft's caches for one sequence of that length, and one sequence's rings and its entry's copies of them. Otherwise each stream gets its own global caches of the capacity, rings and conv states, as `bench` and the tests allocate. More than one stream adds the batched step's buffers. It counts:

- the target's `model.llm.*` tensors as uploaded, with the router, the depth-attention queries and the static conv weights widened to F32, less the tensors in host memory (see [Tensors in host memory](#tensors-in-host-memory)): the embedding table, and with `--weights fp8` the BF16 LM head. The vision tower's `model.visual.*` tensors count only with `--vision`, below. The bundled draft's `mtp.safetensors` is loaded only with the bundled draft;
- the FP8 copies of `--weights fp8`, 0.97 GB, or 1.14 GB with `--output-projection fp8`;
- the global KV caches: 32 KiB per position, 16 KiB with `--kv fp8`, 9 KiB with `--kv nvfp4` or 10.75 KiB with `--kv mixed`, over the 8 global visits (`TextConfig::global_cache_bytes`);
- the 40 sliding rings of 1,024 slots (160 MiB), the conv states and the decode step's vocabulary-sized buffers;
- the prefill buffers, 0.38 MB per chunk row plus 64 MB, and the verify pass's;
- the draft's tensors less the BF16 projections that `fp8_backbone` frees, the bundled draft's fold, the copies its approximations make (the shortlist rows of the LM head, the FP8 backbone), its rings, its buffers and the speculator's tap tails. A draft without an FP8 or shortlist head under `--weights fp8` keeps its own device copy of the BF16 LM head;
- the draft's global caches, 4 KiB per position and layer, or 2 KiB with `fp8_kv` (the default): one layer of the bundled draft, every layer of a 2DSD DFlash draft, none of DFlash 2, which keeps rings of 2,056 slots;
- the prefix cache's copies of one slot's rings and conv states and of the draft's rings (see [Prefix cache](#prefix-cache)), 171 MiB with the bundled draft and 203 MiB with DFlash 2;
- with `--vision`, the vision tower (`vision_bytes`: its weights and its buffers for the largest image, that image's rows included) and the other image rows prefill holds: those of one chunk and of at most one more image that runs past it, 2,304 BF16 values each. The largest image has `(ceil(sqrt(max_pixels) / 40) + 1)^2` tokens, 1,369 at the default 1920 x 1080;
- `RUNTIME_BYTES`, 128 MiB for the module images, the CUDA graphs, the small buffers and the allocator's rounding. It was measured with a thread that sampled `cuMemGetInfo` every millisecond while `siphon-serve` loaded, prefilled more than a chunk and generated speculatively; the uncounted part came to 69 to 87 MiB. Each stream past the first adds `STREAM_RUNTIME_BYTES`, 4 MiB, for one more batch size's CUDA graph and the allocator's rounding of its caches: on an RTX PRO 6000 Blackwell Max-Q at 16,384 positions, the modules and graphs took 0.03 GiB at 1 stream and 0.11 GiB at 32, and without it the peak came 11 to 43 MiB past the plan at 32 streams.

The CUDA context itself takes 507 MiB, or 392 MiB on an RTX 4090. `siphon-serve` reads the free memory right after creating it, so the plan must fit that free memory less `--margin`, 0.25 GiB by default (0.5 GiB before). `siphon-serve` sets `CUDA_MODULE_LOADING=EAGER` unless the environment sets it, so every kernel's code loads with its module at startup, before the memory check, rather than at its first launch. On an RTX 4090 the modules took 16 MiB eagerly against 10 MiB lazily, and a filled context peaked at the same bytes either way.

`--prefill-chunk N` (`siphon-serve` and `siphon-plan`) sets the rows of each prefill chunk, 2048 by default. The prefill buffers take 0.85 GiB at 2048 rows and 0.45 GiB at 1024, and the planner sizes them from the chunk. Prefix cache entries fall on its multiples. Chunks of 1024 cost 6 to 7% of prefill speed: on an RTX 4090, 8,939 against 9,570 tok/s at 8,192 tokens and 8,291 against 8,841 at 32,768; on an RTX 5090, 32,768 tokens took 2.55 s against 2.39 s.

Planned against measured, FP8 weights and cache, the bundled draft, the most that fit the free memory (23.13 GiB on an RTX 4090, 30.46 GiB on an RTX 5090), each filled by one prompt and 256 generated tokens:

| GPU | Margin, chunk | Streams | Capacity | Planned MiB | Peak MiB | Over, MiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| RTX 4090 | 0.5 GiB, 2048 | 1 | 207,872 | 23,171 | 23,168 | 3 |
| | 0.5 GiB, 1024 | 1 | 230,400 | 23,162 | 23,136 | 26 |
| | 0.25 GiB, 2048 | 1 | 222,208 | 23,423 | 23,424 | -1 |
| | 0.25 GiB, 1024 | 1 | 244,736 | 23,414 | 23,392 | 22 |
| | 0.25 GiB, 1024 | 2 | 117,760 | 23,419 | 23,404 | 15 |
| | 0.5 GiB, 2048, short requests | 2 | 97,280 | 23,144 | 23,148 | -4 |
| RTX 5090 | 0.5 GiB, 2048 | 1 | 633,856 | 30,670 | 30,622 | 48 |
| | 0.25 GiB, 2048 | 1 | 648,192 | 30,922 | 30,878 | 44 |
| | 0.25 GiB, 1024 | 1 | 671,744 | 30,931 | 30,878 | 53 |
| | 0.25 GiB, 1024 | 2 | 343,040 | 30,915 | 30,850 | 65 |

The planned and peak bytes here include the context. The peak came at most 4 MiB past the plan on the 4090, where the `sm_89` build's 10 MB of module globals take part of `RUNTIME_BYTES`, and stayed 44 to 65 MiB under it on the 5090. Either way at least 252 MiB of the 0.25 GiB margin stayed free. `Model::save_state` copies only the filled slots of the global caches, so pricing the verify widths at startup does not double the cache.

`siphon-serve` also checks the plan against what it uses. It allocates the full prefill chunk's buffers at startup (`Model::reserve_prefill`), so every buffer the plan counts exists once the warmup has run. Every buffer comes from the device's memory pool, because cudarc allocates with `cuMemAllocAsync`. `Gpu::memory` reads the pool's reserved bytes and their peak (`cuMemPoolGetAttribute`) along with the device's free memory. The bytes outside the pool, the module images and the CUDA graphs, are the drop in free memory since the context was created less the pool's bytes, measured once after warmup. The check prints the pool's peak plus those bytes against the plan, prints the peak again after a job whenever it rises, and warns once if a job takes it past the plan.

`siphon-plan` draws the plan in the terminal. It plans every combination of weight format, KV format and draft (the bundled draft, a `--drafter`, none) with `siphon::memory`. It takes the free memory after the context from nvidia-smi's free memory less the context, 507 MiB or the RTX 4090's 392 MiB, which is how the context was measured. `--margin` and `--prefill-chunk` plan as `siphon-serve`'s do. `--streams N` plans N streams as `siphon-serve --streams N` does.

Planned against measured, as the memory check of `siphon-serve` reports it after a request: the pool's peak plus the bytes outside the pool. At 65,536 positions the request had a 3,820-token prompt and 256 speculative tokens. At the largest capacity, the most that fit 30.46 GiB of free memory, its prompt and 512 speculative tokens left 399 to 436 positions of the cache unused:

| Weights, KV cache | Draft | Capacity | Planned GiB | Measured GiB | Over, MiB |
| --- | --- | ---: | ---: | ---: | ---: |
| FP8, FP8 | DSpark | 65,536 | 20.194 | 20.154 | 41 |
| FP8, FP8 | DSpark | 633,856 (largest) | 29.950 | 29.904 | 47 |
| FP8, FP8 | DSpark | 77,824 x 8 streams (largest) | 29.901 | 29.859 | 43 |
| FP8, FP8 | DFlash 2 | 65,536 | 20.232 | 20.186 | 48 |
| FP8, FP8 | DFlash 2 | 702,464 (largest) | 29.951 | 29.904 | 48 |
| BF16, BF16 | DSpark | 65,536 | 20.995 | 20.928 | 69 |
| BF16, BF16 | DSpark | 342,016 (largest) | 29.960 | 29.896 | 65 |
| BF16, BF16 | DFlash 2 | 65,536 | 21.032 | 20.990 | 43 |
| BF16, BF16 | DFlash 2 | 357,376 (largest) | 29.938 | 29.896 | 43 |

The prompts at the largest capacity:

| Weights, KV cache | Draft | Prompt tokens | Prefill |
| --- | --- | ---: | ---: |
| FP8, FP8 | DSpark | 632,917 | 90.0 s |
| FP8, FP8 | DFlash 2 | 701,516 | 106.7 s |
| BF16, BF16 | DSpark | 341,105 | 51.8 s |
| BF16, BF16 | DFlash 2 | 356,465 | 55.9 s |

The E4M3 draft cache, the tensors in host memory and the BF16 draft projections that `fp8_backbone` frees took the largest FP8 context with the bundled draft from 456,704 positions to 633,856 at 30.45 GiB free, and with 8 streams from 59,392 positions each to 77,824. At 65,536 positions with FP8 weights they free 2.28 GiB: 0.86 GiB each for the embedding table and the BF16 LM head, 0.43 GiB of draft projections and 0.13 GiB of draft cache.

## KV cache format

The global layers (5, 11, 17 and 23, 8 of the 48 layer visits) keep every earlier key and value, so at long context their attention reads most of the bytes of a step. `bench`, `eval`, `parity` and `kvdiff` take `--kv bf16|fp8|nvfp4`, and `Model::with_kv_format` takes the same choice:

- `bf16`, the default, stores the keys and values as computed.
- `fp8` stores the global caches as E4M3, which halves the bytes global attention reads. The sliding-window caches stay BF16.

- `nvfp4` stores them as NVFP4, 4.5 bits per element. See [KV cache reduction](#kv-cache-reduction).

The FP8 cache has no scale. Keys are L2-clipped to norm √512 and values are RMS-normed per head, so no element exceeds 22.7, far inside E4M3's ±448. The append kernel rounds each BF16 value to the nearest E4M3 value, ties to even. Attention decodes E4M3 exactly with the hardware converter, so it differs from BF16 only by that rounding. Ingine stores its FP8 global caches the same way.

## KV cache reduction

The global caches are 16 KiB per position with the FP8 cache, and at long context they hold most of the memory and are most of the bytes a step reads. Two ways to shrink them were measured: 4-bit elements, and one cache per looped global layer instead of one per pass. The second lost and was removed (see [Shared loop caches](#shared-loop-caches-removed)).

| Configuration | Flags | Per position |
| --- | --- | ---: |
| BF16 | `--kv bf16` | 32 KiB |
| FP8 (default) | `--kv fp8` | 16 KiB |
| NVFP4 | `--kv nvfp4` | 9 KiB |
| Mixed: FP8 for visits 5 and 11, NVFP4 for the rest | `--kv mixed` | 10.75 KiB |

### NVFP4 cache

`--kv nvfp4` (`KvFormat::Nvfp4`) stores the global keys and values as NVFP4: E2M1 values with one E4M3 scale per 16 elements, the scale `amax / 6` rounded to nearest and each value `x / scale` rounded to nearest even and saturated at 6. A slot holds 576 bytes per buffer: the 8 heads' values, 64 bytes each, then their 8 scales each. The scales need no global factor: keys are L2-clipped and values RMS-normed, so a group's `amax / 6` stays far inside E4M3's range.

Rounding a group a second time gives the same bytes, since its largest element is then exactly 6 times the scale. So the verify pass, batched steps and prefill round their rows to NVFP4 values in BF16 first (`kv_round_bf16`), and the appends (`kv_append_rows_fp4`, the combine of `attention_rows_fp4`) store them without further change: every reader, including a row reading its own key, sees the same value. The fused decode step rounds its new row across the 4 lanes of each group. The attention kernels read the 4-bit bytes and scales straight into shared memory with `cp.async` and decode them with the hardware converter (`cvt.rn.f16x2.e2m1x2`) times the scale, which is exact in F16 and BF16; no full-size copy exists:

- decode (`decode_step_fp4`): tiles of 64 keys, each warp decoding its 32 dims of a key in the score phase and each lane its 4 dims in the value phase;
- verify and batched steps (`attention_rows_fp4_*`, `attention_batch_fp4`): the E4M3 kernels' fragment layout, with each lane decoding 32 dims of a key for S and one group of 16 dims of 4 keys for P V;
- prefill (`prefill_attention_fp4`): staged tiles decoded in shared memory to BF16 keys and F16 values, then the BF16 path. The FP8 tensor-core attention that `--prefill-attention auto` uses past 16,384 keys has no NVFP4 variant yet.

Prefill over the NVFP4 cache gives the same logits as the BF16 cache with the same rounding (`--kv bf16 --kv-quant nvfp4`, below) to every printed digit on the 4K fidelity text. Verify and decode differ slightly, through their F16 products and the decode step's rounding of its own row. `batch_matches_alone_nvfp4` and `batch_matches_alone_16_nvfp4` (batched rows equal one-row verifies bit for bit), `prefix_cache_entries_nvfp4` and `nvfp4_prefill_matches_stepping` pass. The prefix cache's host tier copies each global cache with its own bytes per position (`GlobalKv::layout`), values and scales alike, so a mixed slot's FP8 and NVFP4 caches round-trip too: `host_cache_matches_prefill_nvfp4`, `host_cache_round_trips_nvfp4`, their mixed variants and siphon-serve's `drain_then_restart_from_disk` with `SIPHON_KV=nvfp4` and `mixed` pass.

### Mixed cache

`--kv mixed` (`KvFormat::Mixed`) keeps the first two global visits (layer 5 and layer 11's first pass, `MIXED_FP8_VISITS`) in FP8 and the other six in NVFP4, after the measurements below found those two the most sensitive to 4 bits. Each cache picks its kernels by its own format, so nothing else changes. `TextConfig::cache_formats` gives each visit's format, and the planner sums the caches (`TextConfig::global_cache_bytes`). `batch_matches_alone_mixed` and `prefix_cache_entries_mixed` pass.

### Method

Quality was measured before any 4-bit kernel existed: `KvOptions` (`--kv-quant nvfp4|q4_0`, with `--kv bf16`) rounds the keys and values each global visit writes through a 4-bit format and back, keeping the BF16 storage, which holds NVFP4 values exactly. `q4_0` is llama.cpp's format: 4-bit integers with one F16 scale per 32 elements, as the third-party llama.cpp build for this model stores its global caches. A decode step rounds its row after its own attention; every other path rounds before. Variants: `--kv-rotate` (a 128-point Hadamard rotation per head before rounding), `--kv-quant-visits` (only some visits rounded), and, in prefill only, `--kv-recent N` (the newest N positions rounded to E4M3, older ones to 4 bits) with `--kv-keys-per-channel` (aged keys with one scale per channel and group of positions, as KIVI does).

- **Fidelity against MLX**: `eval` over quickbench's fidelity captures of 4K and 32K tokens (Gutenberg prose), through prefill in chunks of 2,048 and through verify passes of 8 rows with FP8 weights, as served.
- **Long-context divergence**: no MLX references exist past 32K, so `kvdiff` teacher-forces a long text through prefill in chunks of 2,048 and keeps, at every 64th position, the 256 largest log-probabilities. Against the run of the BF16 cache it reports KL (over those 256 tokens and the rest as one outcome), top-1 agreement and the next token's log-probability, by range of positions. The texts are 1,048,000 tokens of 11 Gutenberg books (`prose`) and of CPython 3.13.1's `Lib/` sources without tests (`code`). 256K-token runs used the RTX 5090, 1M-token runs the RTX PRO 6000.
- **Tasks**: Lucebox's 130 questions (MATH-500 0 to 39, HumanEval 0 to 39, MMLU-Pro 0 to 49) at temperature 0 and 1, and 100 GPQA Diamond questions at temperature 1, through siphon-serve with 32,768-token caps, 8 streams of 40,960 positions (6 for the BF16 cache of the rounding runs). Two runs of 100 questions differ by about 9 points by chance; the intervals are Wilson 95%.
- **Recall**: through siphon-serve, greedy, `reasoning_effort` low: six vault codes hidden at depths 5% to 95% of Gutenberg text, asked for at once; a generated 10-line Python function hidden at 15% or 40% of CPython sources, to be written out again exactly; and the local-ai-registry lab's context gate (a code at the end of archive lines).

### Results

Long-context divergence from the BF16 cache, positions 128K to 256K of the 256K-token texts:

| Configuration | Code KL | Code top-1 | Prose KL | Prose top-1 |
| --- | ---: | ---: | ---: | ---: |
| FP8 | 4.0e-3 | 98.9% | 2.8e-3 | 97.7% |
| NVFP4 | 8.5e-3 | 98.1% | 5.4e-3 | 96.7% |
| q4_0 | 6.7e-3 | 98.3% | 5.4e-3 | 96.2% |
| NVFP4, rotated | 7.0e-3 | 98.4% | 6.0e-3 | 96.2% |
| q4_0, rotated | 5.8e-3 | 98.4% | 4.9e-3 | 96.5% |

The same over 1,048,000 tokens (the code text's last quarter is repetitive source, with a mean next-token loss of 0.12 nats against 0.68 before it):

| Configuration | Code KL, 512K to 768K | Code top-1 | Code KL, all | Prose KL, 512K to 768K | Prose top-1 | Prose KL, all |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FP8 | 3.8e-3 | 98.4% | 3.0e-3 | 2.9e-3 | 97.5% | 2.8e-3 |
| NVFP4 | 6.8e-3 | 98.0% | 5.1e-3 | 6.7e-3 | 96.8% | 6.0e-3 |
| q4_0 | 6.2e-3 | 98.0% | 5.1e-3 | 6.9e-3 | 96.5% | 6.2e-3 |

The 4-bit formats double FP8's divergence and stay flat from 64K to 1M; NVFP4 raises the mean next-token loss on prose by 0.2% (1.9935 against 1.9900 nats at 128K to 256K). Rotation helped q4_0 by 10% to 15% and NVFP4 on code only, and NVFP4 and q4_0 are within 20% of each other either way.

Which global visits matter, positions 64K to 128K of the 128K-token texts, with only one visit's writes rounded to NVFP4 (visits 11, 23, 35 are layer 11's three passes; 17, 29, 41 layer 17's):

| Visits rounded | Prose KL | Code KL |
| --- | ---: | ---: |
| all 8 | 4.8e-3 | 6.5e-3 |
| 5 | 3.0e-3 | 3.8e-3 |
| 11 | 1.9e-3 | 5.2e-3 |
| 17 | 1.3e-3 | 1.5e-3 |
| 23 | 9.1e-4 | 9.5e-4 |
| 29 | 6.1e-4 | 5.0e-4 |
| 35 | 5.4e-4 | 4.0e-4 |
| 41 | 5.0e-4 | 3.5e-4 |
| 47 | 6.7e-4 | 2.8e-4 |
| FP8 cache, for scale | 2.6e-3 | 3.2e-3 |

The first pass's caches and layer 5's carry most of the error; the later passes' caches are 5 to 10 times less sensitive, and the errors do not add up. Keeping visits 5 and 11 unrounded (BF16) brought all-NVFP4's 4.8e-3 and 6.5e-3 down to 2.9e-3 and 2.5e-3, FP8's level; keeping only visit 5, to 3.7e-3 and 5.6e-3. The real mixed cache, with those two visits in FP8, gave 3.9e-3 and 3.8e-3.

The FP8 recent window helped little: 4.3e-3 on prose and 4.4e-3 on code with the newest 4,096 positions in E4M3, 4.5e-3 and 5.1e-3 with 1,024, against 4.8e-3 and 6.5e-3, and both raised the code's mean next-token loss by 1.3%. KIVI's per-channel keys on top of a 1,024-position window gave 5.1e-3 and 4.5e-3. Neither earns a second storage format.

Fidelity against MLX, mean KL at the captures' sampled positions and top-1 agreement over all positions:

| Configuration | 32K prefill | top-1 | 32K verify | top-1 | 4K verify | top-1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 | 2.4e-3 | 97.6% | 4.9e-3 | 96.6% | 5.6e-3 | 96.5% |
| FP8 | 2.5e-3 | 97.4% | 5.4e-3 | 96.5% | 5.8e-3 | 96.5% |
| Mixed | 3.8e-3 | 96.9% | 6.0e-3 | 96.1% | 5.1e-3 | 96.1% |
| NVFP4 | 5.1e-3 | 96.6% | 6.7e-3 | 95.9% | 8.4e-3 | 96.0% |

The verify pass runs on FP8 weights, as served, which adds its own 3e-3.

Tasks, with Wilson 95% intervals (NVFP4 through the rounding on a BF16 cache, 6 streams; prefill over it equals the real cache's bit for bit):

| Configuration | 130 questions, greedy | 130 questions, T=1 | GPQA, T=1 | GPQA cut off |
| --- | ---: | ---: | ---: | ---: |
| FP8 | 113, 86.9% (80.1 to 91.7) | 122, 93.8% (88.3 to 96.8) | 77, 77.0% (67.8 to 84.2) | 14 |
| NVFP4 | 116, 89.2% (82.7 to 93.5) | 121, 93.1% (87.4 to 96.3) | 76, 76.0% (66.8 to 83.3) | 13 |
The mixed cache was not run on the tasks: its divergence lies between FP8's and NVFP4's, which these sets cannot tell apart. Every difference lies inside the intervals. The FP8 runs of the 130 questions gave the same scores as the main build's earlier runs, as seeded runs do.

Recall, passes per trial (greedy, `reasoning_effort` low, 4,096-token cap; 32K and 128K on 3 trials, 512K on 2, 1M on 1, all on the RTX 5090):

| Configuration | 32K | 128K | 512K | 1M |
| --- | --- | --- | --- | --- |
| FP8 | 7/7 | 7/7 | 5/5 | does not fit |
| NVFP4 | 7/7 | 7/7 | 4/5 | 3/3 |
| Mixed | not run | 7/7 | 5/5 | not run |

Each cell counts the context gate, the needle trials and the code-copy trials. Every failure was a needle trial whose reasoning looped until the 4,096-token cap, without an answer; every reply that answered named all 6 codes, and every code copy was exact. The gate prompts hold 26K, 106K, 426K and 832K tokens.

Speed on the RTX 5090, FP8 weights, the same hold for each row's configurations. Decode is `bench`'s median step over 200 after 20, with every cache filled to the context; speculation is `bench --spec auto` over 512 tokens of the anyhow capture, so its tokens per round depend on the text each cache produces; prefill is the time to the first token of random ids:

| Configuration | Decode 32K | 128K | 512K | Speculative 32K | 128K | 512K | Prefill 32K | 128K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FP8 | 309.6 tok/s | 236.0 | 124.5 | 806.7 | 717.4 | | 2.14 s | 10.05 s |
| NVFP4 | 322.3 | 267.6 | 166.8 | 846.6 | 721.3 | 487.0 | 2.27 s | 12.28 s |
| Mixed | 320.1 | 260.0 | 153.8 | 824.8 | 729.5 | | 2.24 s | 11.73 s |

A second run of the decode rows in the same hold agreed within 0.1%. These runs predate the BF16 output projection that `--weights fp8` now keeps (`--output-projection`), so their absolute times include the FP8 copy; the comparisons between caches hold, since all read the same weights. NVFP4 reads 44% fewer cache bytes: decode is 4% faster at 32K, 13% at 128K and 34% at 512K. Its prefill is 6% slower at 32K and 22% at 128K, because past 16,384 keys the FP8 cache's prefill runs attention on FP8 tensor cores and NVFP4 has no such variant; at 508K tokens through siphon-serve the prefill took 98.9 s against 67.0 s.

Memory: the planner's longest context (`siphon-plan --print`, default 0.25 GiB margin) with the bundled draft and FP8 weights, the prefix cache's entry per stream included (the RTX PRO 6000 Blackwell Workstation Edition with 95.1 GiB free, the RTX 5090 with 30.46 GiB):

| Configuration | 5090, 1 stream | 2 | 4 | 8 | PRO 6000, 8 streams |
| --- | ---: | ---: | ---: | ---: | ---: |
| FP8 | 657,408 | 335,872 | 161,792 | 71,680 | 592,896 |
| Mixed | 928,768 | 486,400 | 238,592 | 105,472 | 875,520 |
| NVFP4 | 1,048,576 | 571,392 | 281,600 | 125,952 | 1,041,408 |

On the RTX PRO 6000 every configuration reaches the model's 1,048,576 positions at 1, 2 and 4 streams. The servers' own check stayed under the plan in every recall run: after a job at 1M on the RTX 5090, 30,302 MiB used of 30,352 MiB planned for NVFP4; at 512K, 28,574 of 28,636 for FP8 and 25,054 of 25,094 for NVFP4. No scratch copy grows with the context: the NVFP4 kernels decode tiles in shared memory.

### What to run

- **Default: FP8**, as before. It is the closest to the BF16 cache.
- **Opt-in: `--kv nvfp4`** where memory or long-context decode speed matters more than the last bit of fidelity: 9 KiB per position, 1.6 times FP8's context, decode 34% faster at 512K, at about twice FP8's divergence and no measurable task or recall loss. `--kv mixed` takes 10.75 KiB for divergence close to FP8's.

What lost: shared loop caches (below); the Hadamard rotation (10% to 15% better for q4_0, worse for NVFP4 on prose); an FP8 window of recent positions (at most a third less divergence, and a second format); KIVI's per-channel keys (no gain); q4_0 over NVFP4 (within 20% either way, and no hardware decode).

Open: NVFP4 prefill attention on tensor cores (the FP8 MMA cannot take NVFP4's values exactly; the experts' hi/lo split could); the BF16 rounding leaves a decode step's own row unrounded for that step's attention, so its token-by-token runs differ slightly from the real NVFP4 cache; the FP8 cache's tests that compare single elements (`tests/attention.rs`) have no NVFP4 counterpart, which `eval` and the end-to-end tests cover instead.

### Shared loop caches (removed)

Layers 11 and 17 run in each of the loop's three passes, so they make 6 of the 8 global visits. Giving each of them one cache that all its passes write, as Transformers' `recurrent_kv_cache_mode="last_loop"` does ([PR 51161](https://github.com/huggingface/transformers/pull/51161)), takes 8 KiB per position with FP8: the earlier passes then read the last pass's keys and values at earlier positions. It lost, and the code lives on the branch `experiment/kv-loop-shared`. Against the BF16 cache at 128K to 256K its KL was 2.9e-2 on code and 4.4e-2 on prose, 7 and 16 times FP8's, and top-1 agreement fell from 98.9% to 96.5% on code and from 97.7% to 90.5% on prose (89.7% at 512K to 768K of the 1M-token text). Against MLX at 32K its KL was 3.5e-2 through prefill and 7.0e-2 through verify. On GPQA it scored 69/100 (59.4% to 77.2%) against FP8's 77, with 27 replies cut off at 32,768 tokens against 14. NVFP4 saves 7 of its 8 KiB at a third to an eighth of the divergence. The PR's identical greedy tokens came from a 2,048-token prompt run as one forward pass, where only the 128 decode steps see the approximation; in Siphon every prefill chunk after the first and every decode step do.

## Tensors in host memory

Two BF16 tensors of 0.86 GiB each are read only a few rows at a time: the embedding table, which a step, a verify pass, a batched step, a prefill chunk and a draft round gather their tokens' rows from, and, with `--weights fp8`, the BF16 LM head once every product reads its FP8 copy. Both live in page-locked host memory mapped into the device's address space (`gpu::HostMapped`, `cuMemHostAlloc` with `CU_MEMHOSTALLOC_DEVICEMAP`), and kernels read their rows over PCIe through the mapped address, which CUDA graphs keep. The embedding gather reads the same values as before, so every result stays the same bit for bit.

`Weights::quantize_fp8` moves the BF16 LM head to host memory after making its FP8 copy. Decode and verify passes read the FP8 copy already. The prefill's last row, which gives the first token of a reply, and `PrefillLogits::All` now run the verify pass's multi-row FP8 product too, so the first token is the one a one-row verify at that position would pick. The bundled draft builds its fold from the host copy at load and gathers its shortlist rows from it. A draft without an FP8 or shortlist head keeps its own device copy for its base logits, as before.

Over 212 prompts (200 open-perfectblend chat prompts as `siphon-serve` renders them and the 12 prompts of the server check), the first token's logits from the FP8 head against the BF16 head, both after the same FP8 prefill: mean KL 6.8e-9, largest 5.8e-7, the same top-1 token in 212 of 212, largest logit difference 0.75. With `--weights bf16` nothing changes. `siphon-serve --greedy` gave the same tokens as before on 12 chat prompts (stories, code, explanations, a sonnet, a word problem) of 529 to 3,000 generated tokens.

Times against the build before, FP8 weights and caches, `anyhow` prompt, minimum over 5 to 10 runs alternating builds in one hold of the GPU (prefill with BF16 weights and cache, whose head did not change):

| Work | Context | Before, ms | After, ms | Change |
| --- | ---: | ---: | ---: | ---: |
| Plain step | 0 | 2.713 | 2.710 | -0.1% |
| Plain step | 32,768 | 3.227 | 3.214 | -0.4% |
| Verify, 1 row | 10,674 | 3.388 | 3.368 | -0.6% |
| Verify, 1 row | 32,768 | 3.696 | 3.683 | -0.4% |
| Verify, 8 rows | 10,674 | 6.108 | 6.109 | 0.0% |
| Verify, 8 rows | 32,768 | 6.485 | 6.462 | -0.4% |
| Batched step, 8 streams | 0 | 8.229 | 8.222 | -0.1% |
| Batched step, 8 streams | 32,768 | 11.812 | 11.818 | +0.1% |
| Prefill of 2,048 tokens | 0 | 123.6 | 123.6 | 0.0% |
| Prefill of 16,384 tokens | 0 | 1,055.8 | 1,057.2 | +0.1% |

The medians moved by at most +0.3%. The draft round in `siphon-serve` went from 0.515 to 0.520 ms. `dspark time`, which keeps the embedding table on the device, shows no change in the round, so the 5 us are most likely the PCIe read of the anchor's row.

## Dense weight format

Decode reads 3.62 GB of BF16 dense weights per token: every visit's attention input and output projections, the looped layers' message gates (113 MB), and the LM head. `bench`, `eval` and `parity` take `--weights bf16|fp8`, and `Model::with_formats` takes the same choice:

- `bf16`, the default, reads the checkpoint's weights.
- `fp8` quantizes the attention input projections, the message gates and the LM head at load time on the GPU to E4M3, with one F32 scale per row, `amax / 448`. The output projection stays BF16 (see [The output projection](#the-output-projection)) unless `--output-projection fp8` (`Weights::output`) quantizes it too. Decode then reads 2.16 GB for these matrices, or 1.82 GB with an FP8 output projection. Prefill keeps the BF16 weights unless `--prefill-dense fp8` (see [Prefill on FP8 tensor cores](#dense-products)), except the LM head's: its BF16 copy moves to host memory and the prefill's head reads the FP8 copy (see [Tensors in host memory](#tensors-in-host-memory)). The copies take 0.97 GB of device memory, or 1.14 GB with the output projection's: one per layer, since the looped layers' three visits share theirs.

The FP8 path lowers precision on purpose. Each byte is the BF16 weight over its row's scale, rounded to the nearest E4M3 value. The products widen each value exactly, sum in F32, and multiply the row's scale into the sum once before rounding to BF16. The router and the small vectors stay as they are. `tests/linear_fp8.rs` checks the quantizer against a CPU model and the products against an F64 reference over the quantized weights.

Engine stepping with the BF16 cache, against the F32 captures of `anyhow`, `half` and `regex-readme`, over all positions:

| Dense weights | Mean KL | \|log-prob difference\| | Top-1 | Decode at context 0 |
| --- | --- | --- | --- | --- |
| `bf16` | 1.5e-3, 9.4e-4, 1.1e-3 | 0.023, 0.028, 0.028 | 99.0%, 98.6%, 98.5% | 4.09 ms |
| `fp8`, FP8 output projection | 2.3e-3, 1.5e-3, 2.2e-3 | 0.029, 0.038, 0.040 | 98.7%, 98.2%, 98.0% | 3.05 ms |

The FP8 path's mean KL is 1.5 to 2.0 times the BF16 path's, and its \|log-prob difference\| 1.3 to 1.4 times. It still passes the gate against MLX's BF16 captures.

### The output projection

With the FP8 cache, stepping on an FP8 copy of every dense matrix failed the gate on `regex-readme`: the range after the window wraps came to a \|log-prob difference\| of 0.0451 against the 0.045 limit. Rounding single matrices to their E4M3 values in place, with BF16 weights otherwise, put most of the excess on the output projection, then the value projection; the query and key projections, the other input projections, the message gates and the LM head added little. The excess was spread over the text, mostly on tokens MLX finds unlikely, as small logit noise is. So `--weights fp8` now leaves the output projection BF16.

Two fixes were tried. One keeps the output projection's BF16 weights in decode. The other gave its FP8 copy a scale per 32 weights of a row: a UE8M0 power of two, the block's `amax / 448` rounded up, as MXFP8 scales its blocks. A power-of-two scale multiplies exactly, so the GEMV applied it to each sum of 16 products and the multi-row kernel folded it into the 2^112 of its E4M3 widening, and neither added a rounding. Stepping and `eval --verify 8`, FP8 weights and cache, on an RTX 5090. The \|log-prob difference\| is the mean over all positions and, after the slash, that of the worst range; the KL is the mean at MLX's sampled positions:

| Text, tokens | Output projection | Stepping against F32 | Stepping against MLX | Verify against F32 | Verify KL against MLX |
| --- | --- | --- | --- | --- | --- |
| `regex-readme`, 2,798 | FP8, row scales | 0.0413 / 0.0456 | 0.0408 / **0.0451** | 0.0411 | 3.0e-3 |
| | FP8, block scales | 0.0397 / 0.0434 | 0.0394 / 0.0432 | 0.0399 | 2.0e-3 |
| | BF16 | 0.0372 / 0.0411 | 0.0358 / 0.0390 | 0.0360 | 2.7e-3 |
| | BF16 weights throughout | 0.0309 / 0.0331 | 0.0319 / 0.0339 | 0.0308 | 9.5e-4 |
| `half`, 2,785 | FP8, row scales | 0.0392 / 0.0426 | 0.0387 / 0.0406 | 0.0375 | 1.1e-3 |
| | FP8, block scales | 0.0376 / 0.0395 | 0.0379 / 0.0400 | 0.0370 | 8.7e-4 |
| | BF16 | 0.0352 / 0.0369 | 0.0366 / 0.0385 | 0.0347 | 1.4e-3 |
| | BF16 weights throughout | 0.0291 / 0.0312 | 0.0320 / 0.0341 | 0.0289 | 1.1e-3 |
| `anyhow`, 11,186 | FP8, row scales | 0.0299 / 0.0354 | 0.0304 / 0.0340 | 0.0305 | 2.6e-3 |
| | FP8, block scales | 0.0325 / 0.0365 | 0.0326 / 0.0378 | 0.0313 | 2.5e-3 |
| | BF16 | 0.0288 / 0.0335 | 0.0298 / 0.0341 | 0.0289 | 2.9e-3 |
| | BF16 weights throughout | 0.0246 / 0.0290 | 0.0254 / 0.0298 | 0.0254 | 2.3e-3 |
| 4K Gutenberg text, 3,993 | FP8, row scales | 0.0677 / 0.0712 | 0.0682 / 0.0719 | 0.0688 | 5.8e-3 |
| | FP8, block scales | 0.0669 / 0.0672 | 0.0685 / 0.0702 | 0.0680 | 6.7e-3 |
| | BF16 | 0.0624 / 0.0648 | 0.0613 / 0.0635 | 0.0595 | 4.7e-3 |
| | BF16 weights throughout | 0.0480 / 0.0492 | 0.0498 / 0.0518 | 0.0486 | 2.2e-3 |
| 32K Gutenberg text, 31,993 | FP8, row scales | 0.0658 / 0.0676 | 0.0675 / 0.0691 | 0.0659 | 5.4e-3 |
| | FP8, block scales | 0.0652 / 0.0691 | 0.0664 / 0.0689 | 0.0651 | 6.0e-3 |
| | BF16 | 0.0598 / 0.0610 | 0.0613 / 0.0625 | 0.0593 | 4.8e-3 |
| | BF16 weights throughout | 0.0450 / 0.0466 | 0.0479 / 0.0495 | 0.0450 | 2.8e-3 |
| 128K Gutenberg text, 128,000 | FP8, row scales | 0.0615 / 0.0621 | 0.0624 / 0.0630 | 0.0617 | 4.2e-3 |
| | FP8, block scales | 0.0608 / 0.0614 | 0.0619 / 0.0625 | 0.0609 | 4.0e-3 |
| | BF16 | 0.0560 / 0.0565 | 0.0570 / 0.0575 | 0.0559 | 3.7e-3 |
| | BF16 weights throughout | 0.0427 / 0.0432 | 0.0441 / 0.0446 | 0.0428 | 2.6e-3 |

The Gutenberg texts are quickbench's fidelity texts, with F32 captures made as for the code texts (`capture --float32`, 18 s, 262 s and 31 minutes on an M5 Max). They fail the gate's \|log-prob difference\| limit with any weights, since the limit comes from MLX's noise on code.

The BF16 output projection lowers the \|log-prob difference\| against F32 on every text and mode, by 0.001 to 0.002 on `anyhow` and by 0.003 to 0.009 on the others, and every code text passes the gate in both modes. The block scales help `regex-readme` and `half` but make `anyhow` worse, and leave the prose texts within 0.001 to 0.002 of row scales. E4M3 keeps 3 mantissa bits in every binade, so a finer scale changes how a weight rounds but not by how much; it gains only on weights that a row's scale pushes into the subnormals. Block scales on the value projection as well gave nothing either way. The block scales were dropped. The verify KL at the 80 to 100 sampled positions moves both ways by more than these differences on the code texts, so the table also gives the verify pass's \|log-prob difference\| over all positions.

The cost, on an RTX 5090 and on an RTX 4090 with the `sm_89` port applied, from `bench` (plain steps, 200 per context) and `bench --spec auto` (the verify pass alone, 8 rows and 1 row), the second of two alternating runs, which agreed within 2%:

| GPU | Output projection | Plain step, context 0 | Plain step, 32K | Verify of 8 rows at 10,674 · 32K | Verify of 1 row at 10,674 · 32K |
| --- | --- | ---: | ---: | ---: | ---: |
| RTX 5090 | FP8 | 2.776 ms · 360 tok/s | 3.288 ms · 304 tok/s | 7.74 · 8.14 ms | 4.12 · 4.55 ms |
| | BF16 | 2.933 ms · 340 tok/s | 3.413 ms · 293 tok/s | 7.86 · 8.27 ms | 4.20 · 4.65 ms |
| RTX 4090 | FP8 | 4.835 ms · 207 tok/s | 6.026 ms · 166 tok/s | 13.68 · 14.34 ms | 6.84 · 7.67 ms |
| | BF16 | 5.092 ms · 196 tok/s | 6.371 ms · 157 tok/s | 14.00 · 14.39 ms | 7.09 · 7.91 ms |

Decode reads 0.34 GB more per token. A plain step takes 4 to 6% longer and a verify pass 0.3 to 3.6% longer, since the verify streams the matrix once for all its rows. Speculative tok/s cannot be compared directly here: each build's greedy continuation is its own text. The plan drops the copy's 170 MB, which gives the longest FP8 context on an idle 32 GB card 9,216 more positions (657,408 with the bundled draft). The block scales cost 0.3% of a plain step.

`--output-projection fp8` reproduces the earlier FP8 logits bit for bit. The verify pass's logits, `smallm verify --hash 1` over `anyhow-chunked`, on an RTX 5090 (`sm_120a`):

| Weights and cache | Output projection | Width 1 | Width 3 | Width 8 |
| --- | --- | --- | --- | --- |
| FP8 | BF16 (default) | `ecb8ef5d7c18444c` | `f4ed3f1f56f4aa8b` | `e7e9433e3d18e5b6` |
| FP8 | FP8 | `93ecee4b34af1ded` | `4b6b4009599bb7e5` | `4f41eb437bf879d3` |
| BF16 | | `9ad367f3963f7b15` | `1850454fe63cef4f` | `aa9bae88dba01800` |

The FP8 hashes change because the output projection's products now multiply BF16 weights, through the BF16 GEMV and the BF16 multi-row kernel. The BF16 hashes do not change.

The pixi environment includes Nsight Systems (`nsys`, CLI only) and Nsight Compute (`ncu`) from conda-forge. `ncu` 2026.4 is the release for the pinned CUDA 13.5 toolkit. A kernel trace of 20 steps:

```sh
pixi run nsys profile -t cuda --cuda-memory-usage=false -o decode \
    target/release/bench <model-dir> --contexts 32768 --warmup 5 --steps 20
pixi run nsys export --type sqlite -o decode.sqlite decode.nsys-rep
```

The trace marks the end of each step with a 4-byte device-to-host copy, the sampled token. `ncu` needs the GPU performance counters, which the driver limits to administrators unless the `NVreg_RestrictProfilingToAdminUsers=0` module option is set. Without it, `ncu` fails with `ERR_NVGPUCTRPERM`.

## Expert activation format

Prefill's expert products take `--expert-activations bf16|quantized|fp8` in `bench` and `eval`, and `Model::set_expert_activations` takes the same choice. Decode always takes BF16 activations. This section covers `quantized`; [Prefill on FP8 tensor cores](#expert-products) covers `fp8`.

- `bf16`, the default, decodes the NVFP4 and MXFP8 weights exactly to BF16 and multiplies them with the BF16 activations, as MLX does.
- `quantized` quantizes each expert input row as the checkpoint's `hf_quant_config.json` declares (`required_activation_quantization`) and multiplies the raw weights on the block-scaled tensor cores (`mma.sync ... kind::mxf4nvf4.block_scale` and `kind::mxf8f6f4.block_scale`).

The quantized path lowers precision on purpose. `Experts::quantize_rows` quantizes each row of the `w13` input (one per token) and of the `w2` input (one per token and expert) once:

- Layers 0 to 19, NVFP4: E2M1 values in blocks of 16 with E4M3 block scales, under a per-row outer scale `amax / 1536`. Each block tries the scale that maps its largest value to 6 and the one that maps it to 4, and keeps the one with the smaller sum of absolute errors ("Four Over Six", 6 on ties).
- Layers 20 to 23, MXFP8: E4M3 values in blocks of 32 with a UE8M0 scale, the block's `amax / 448` rounded up to a power of two.

Both round to nearest even. The rules follow vLLM's Inkling kernels (`vllm/models/inkling2/nvidia/ops/nvfp4.py` and `mxfp8.py`) bit for bit; `tests/experts.rs` checks the quantizer against a CPU model of them, and the products against an F64 reference over the dequantized inputs. vLLM multiplies the routing weight into the `w2` input before quantizing it. The engine applies the routing weight after `w2`, as the BF16 path does. The products sum in F32 and round to BF16 where the BF16 path rounds.

mma.sync with F32 accumulators on the RTX 5090: BF16 254 TFLOP/s, FP8 1,015 and FP4 2,030. The quantized products are then bound by memory, mostly L2: each 64-pair tile re-reads its expert's weights. Per call at 2048-row chunks, with the floor that reading the weights and scales once and writing the outputs sets at 1.70 TB/s:

| Projection | `bf16` | `quantized` | Memory floor |
| --- | ---: | ---: | ---: |
| NVFP4 `w13` with SwiGLU | 955 us | 374 us | 250 us |
| NVFP4 `w2` | 484 us | 189 us | 162 us |
| MXFP8 `w13` with SwiGLU | 995 us | 613 us | 437 us |
| MXFP8 `w2` | 507 us | 326 us | 262 us |

Quantizing one projection's inputs takes 14 us. Time to first token falls from 262 to 183 ms at 4,096 tokens, from 2.34 to 1.72 s at 32,768 and from 12.78 to 10.28 s at 131,072.

Prefill in chunks of 2048 with the BF16 cache, against the F32 captures of `anyhow`, `half` and `regex-readme`: KL at the sampled positions, the rest over all positions:

| Expert activations | Mean KL | \|log-prob difference\| | Top-1 |
| --- | --- | --- | --- |
| `bf16` | 2.0e-3, 6.4e-4, 9.7e-4 | 0.024, 0.027, 0.027 | 99.0%, 98.9%, 98.6% |
| `quantized` | 3.3e-3, 2.1e-3, 3.3e-3 | 0.038, 0.048, 0.051 | 98.3%, 97.5%, 97.6% |

The NVFP4 activations cause the difference: quantizing only the MXFP8 layers stays close to the `bf16` numbers (0.024, 0.027, 0.028), and quantizing only the NVFP4 layers gives all of it, about half from each projection. Always taking the scale that maps each block's maximum to 6 is worse on all three texts (0.039, 0.051, 0.053). Perplexity stays within 0.3% of the F32 captures'. The greedy continuation still matches all 32 tokens after `anyhow` and `half`, and 12 after `regex-readme` against 26 with `bf16`. The mean |log-prob difference| exceeds the gate's 0.045 on some position ranges, so `eval` fails with `quantized` on 6 of the 7 captures tried.

## Prefill on FP8 tensor cores

Three prefill options run products on the block-scaled FP8 MMA (`mma.sync ... kind::mxf8f6f4.block_scale`, m16n8k32). With F32 accumulators it runs at 1,015 TFLOP/s on the RTX 5090, against 254 for BF16; without block scales the same MMA runs at 509. Each option rounds activations to E4M3, so each went through the quality gate. Decode never uses them. `bench` and `eval` take the flags (`siphon-serve` takes `--prefill-attention`), and the `Model` setters take the same choices:

| Option | Flag | Setter | Default |
| --- | --- | --- | --- |
| Global attention | `--prefill-attention auto\|bf16\|fp8` | `Model::set_prefill_attention` | `auto`: BF16 up to 16,384 keys, FP8 past them, with an FP8 cache |
| Expert activations | `--expert-activations fp8` | `Model::set_expert_activations` | off |
| Dense products | `--prefill-dense bf16\|fp8` | `Model::set_prefill_dense` | off |

Time to first token with the FP8 cache and weights, chunks of 2048, from `bench --prefill`. Each column ran alternating with the build before these options, twice, in one GPU lock hold per length:

| Prompt | Before | FP8 attention (default) | FP8 expert activations | FP8 dense products | All three |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8,192 | 506 ms | 497 ms | 457 ms | 468 ms | 410 ms |
| 32,768 | 2.23 s | 2.09 s | 2.04 s | 2.09 s | 1.75 s |
| 92,160 | 7.69 s | 6.55 s | 7.15 s | 7.28 s | 5.61 s |
| 131,072 | 12.32 s | 10.03 s | 11.53 s | 11.71 s | 8.70 s |

The expert and dense columns have the other options off. The expert column ran in holds of its own, where the build before took 506 ms, 2.23 s, 7.67 s and 12.25 s. At 92,160 tokens with all three, nsys gives FP8 attention 1.00 s (2.14 s before), the NVFP4 expert products 2.43 s (2.88 s), and the dense products 0.45 s on FP8 plus 0.39 s for the output projection, which stays BF16 (1.29 s).

The quality gate with the FP8 cache and weights, prefill in chunks of 2048, against MLX's chunked captures of `anyhow`, `regex-readme` and `half`: mean KL at the sampled positions, mean |log-prob difference| over all positions, and that of the worst position range (limits 4e-3 and 0.045):

| Prefill products | Mean KL | \|log-prob difference\| | Worst range | Verdict |
| --- | --- | --- | --- | --- |
| BF16 (before) | 1.9e-3, 1.4e-3, 8.7e-4 | 0.026, 0.032, 0.030 | 0.029, 0.035, 0.032 | pass |
| FP8 attention (default) | 2.0e-3, 1.5e-3, 7.3e-4 | 0.027, 0.032, 0.031 | 0.030, 0.035, 0.033 | pass |
| FP8 expert activations | 2.0e-3, 1.7e-3, 1.0e-3 | 0.028, 0.038, 0.036 | 0.033, 0.042, 0.037 | pass |
| FP8 dense products | 3.5e-3, 1.4e-3, 1.4e-3 | 0.032, 0.037, 0.040 | 0.038, 0.039, 0.042 | pass |
| FP8 attention and expert activations | 3.2e-3, 1.8e-3, 9.3e-4 | 0.028, 0.038, 0.034 | 0.034, 0.041, 0.036 | fails `half`: perplexity ratio 5.2e-3, limit 5e-3 |
| All three | 2.8e-3, 2.3e-3, 1.6e-3 | 0.034, 0.042, 0.041 | 0.044, 0.046, 0.043 | fails `regex-readme`: worst range 0.046 |

Each option passes alone. Attention is the default: it saves the most time at long prompts and adds the least error. The expert activations fail together with it, so they stay opt-in, as do the dense products.

FP8 attention throughout still changed short prompts' greedy output, and with it speculation. The registry lab's speed prompt, a 31-token request for a 600-word story, went from 6,304 tokens at 4.19 tokens per verify round (668 tok/s) to 1,264 tokens at 1.72 (343 tok/s): the changed cache tipped decode into the model's terse reasoning style, which the draft predicts poorly. Over 12 varied prompts the mean went from 3.04 to 2.79 tokens per round, with large swings both ways. Short prompts gain almost nothing from FP8, 2% of time to first token at 8,192 tokens, and the eval's top-1 agreement dropped most over the first 1,024 positions (99.12% to 98.34% on `anyhow`), where few keys share the rounded weights. So `auto`, the default, runs a chunk on FP8 only when its last row reads more than `FP8_ATTENTION_KEYS` (16,384) keys. Shorter prompts prefill bit for bit as with `bf16`: the 12 prompts give the same outputs as the build before FP8 attention. Longer ones keep most of the saving, since attention's work grows with the square of the length. Time to first token, alternating the three, twice, in one hold per length:

| Prompt | `bf16` | `auto` | `fp8` |
| ---: | ---: | ---: | ---: |
| 8,192 | 505 ms | 505 ms | 496 ms |
| 32,768 | 2.23 s | 2.12 s | 2.08 s |
| 92,160 | 7.67 s | 6.58 s | 6.53 s |
| 131,072 | 12.29 s | 10.06 s | 10.01 s |

The tables above, FP8 attention rows included, ran with `fp8`.

### Global attention

`prefill_attention_fp8_mma` runs both products of a global layer's FP8 cache on the FP8 MMA, with the scheduling of `prefill_attention_fp8`:

- S = Q K^T: a segment quantizes its queries to MXFP8 when it loads them: per row and 32 dims, a UE8M0 scale (the block's amax / 448 rounded up to a power of two) and E4M3 values. The scales go into the MMA. The keys are the cache's bytes, read with ldmatrix from the staged tile.
- O += P V: the softmax weights carry a factor of 2^8 instead of 2^15, so the largest is 256, and round to E4M3. A pass over each staged value tile transposes it to [dim][key] for the B operand. Within each block of 32 keys the MMA's k positions hold the keys in the order a lane holds them in the S fragments, so the weights pack into A registers without shuffles, and the transpose writes the values in the same order.

The online softmax and its sums stay F32. Against an F64 reference with the same quantized queries, `tests/attention.rs` allows 1/14 of each output's magnitude for the weights' rounding; the largest error seen is 5.4%. The kernel uses 255 registers and one block per SM, as before. Sliding layers keep BF16 caches and the BF16 kernel.

### Expert products

`fp8` quantizes every expert input to MXFP8. The MXFP8 experts (layers 20 to 23) then run `expert_gemm_q_mxfp8`, the kernel of `quantized`. The NVFP4 experts (layers 0 to 19) need more: their weights carry an E4M3 scale per 16 values, while the FP8 MMA takes one UE8M0 scale per 32. An E2M1 value times an E4M3 scale has up to 6 significant bits, more than E4M3's 4, so no single E4M3 operand holds a scaled weight.

`expert_gemm_q8_nvfp4` splits each weight w into two E4M3 values under one power of two per row and 32 values: w = 2^e (hi + lo / 16), with 2^e the larger of the block's two scale exponents over 32, hi the top 4 significant bits and lo the rest. Each k step of 32 runs one MMA on hi with A scale 2^e and one on lo with A scale 2^(e - 4), both accumulating in F32. Every product is exact. The split depends only on the E2M1 magnitude, the scale's significand and the exponent gap to the block's larger scale, so each CTA builds a 2.8 KB table of hi and lo bytes and decodes with `prmt` lookups. It is exact while the two scales of a block are within 2^10; the checkpoint's widest gap is 2^5, and `Model::set_expert_activations` checks every scale (`nvfp4_scale_spread`) before it enables `fp8`. Each of 8 warps takes 16 output columns and all 128 pairs of a tile, so each weight is decoded once.

Per call at 2048-row chunks (`bench_expert_activations` in `tests/experts.rs`, uniform routing, quantization included):

| Projection | `bf16` | `quantized` | `fp8` |
| --- | ---: | ---: | ---: |
| NVFP4 `w13` with SwiGLU | 906 us | 350 us | 737 us |
| NVFP4 `w2` | 467 us | 209 us | 447 us |
| MXFP8 `w13` with SwiGLU | 933 us | 570 us | 570 us |
| MXFP8 `w2` | 479 us | 326 us | 341 us |

The NVFP4 `fp8` kernel issues twice the MMAs of an exact single-pass one and keeps the tensor pipe about 60% busy; ncu puts most stalls on barriers, MMA throttling and shared-memory loads. Two other designs ran slower: one MMA per 16-value group with a fresh accumulator and four F32 FMAs to apply the scale (807 us for NVFP4 `w13`), and the hi/lo split decoded with F16 arithmetic in each warp (818 to 929 us).

### Dense products

`--prefill-dense fp8` runs the attention input projections and the message gates' down projections with `gemm_fp8_128x128` on the FP8 weight copies of `--weights fp8` (E4M3 with one F32 scale per row). Each distinct input is quantized once per call to MXFP8 into a scratch buffer of the prefill chunk (29 MB at 2048 rows, which the memory planner counts). The inputs' UE8M0 scales go into the MMA; the weight row's scale multiplies each F32 sum before it rounds to BF16. `tests/linear_fp8.rs` checks the products against an F64 reference over the quantized inputs and weights.

The output projection stays BF16. Alone on `regex-readme`, its FP8 product raised the worst range's |log-prob difference| from 0.035 to 0.043; all the input projections together raised it to 0.038, and the message gates not at all. With it, the dense products failed the gate there (0.050).

## RTX 4090 (sm_89)

`SIPHON_CUDA_ARCH=sm_89` builds the kernels for Ada, the RTX 4090, and sets the `sm89` cfg for the Rust code:

```sh
SIPHON_CUDA_ARCH=sm_89 cargo build --release
```

`Gpu::new` accepts only a device of the build's compute capability, so the `sm_89` build refuses an RTX 5090 and the default build refuses an RTX 4090. The `sm_89` build was tested with nvcc 13.2 and driver 595 (`CUDA_HOME=/usr/local/cuda`). The default `sm_120a` build compiles to the same SASS as before, kernel for kernel.

Ada lacks five things the `sm_120a` kernels use. `kernels/arch.cuh` defines `SIPHON_SM89`, which selects the replacements:

- **Programmatic dependent launch.** `pdl_wait` and `pdl_trigger` do nothing, and the launchers leave out the launch attribute, so each kernel starts after the previous one ends.
- **Thread-block clusters.** The router (`moe_router_step_2304`, `moe_router_step_rows_2304`), the message gates (`message_gate_step*`, `message_gate_rows*`) and the split-k GEMM run each cluster as plain blocks that meet in global memory (`kernels/cluster_sm89.cuh`). Each block has a mailbox that mirrors its shared memory by address. A send writes the values at the target's address in the receiver's mailbox and adds their bytes to the word at the address of the receiver's mbarrier. The receiver waits for the bytes it expects, copies them to shared memory and clears the word for the next launch. Cluster barriers become a sense-reversing barrier on two counters per cluster. Every sum keeps its values and order. The blocks of a cluster wait for each other. That cannot deadlock while a cluster has at most 16 blocks and the GPU starts blocks in index order: the earliest unfinished cluster then always has every block running. `sample_prepare` runs each row in one block instead of a cluster of 8. Its histograms count in integers and its maxima compare exactly, so its thresholds stay the same.
- **Bulk prefetches into L2.** `cp.async.bulk.prefetch` becomes one `prefetch.global.L2` per 128-byte line. The paced decode prefetch runs at 0.8 TB/s instead of 1.5.
- **The block-scaled MMA.** The kernels of FP8 prefill attention, quantized and FP8 expert activations and FP8 dense products trap there, and the `Model` setters refuse them. `--prefill-attention auto` stays on BF16 at every length. `siphon_cuda::BLOCK_SCALED_MMA` is false, and the 9 kernel tests of those paths are skipped.
- **An E2M1 converter.** CUDA's `__nv_cvt_fp4x2_to_halfraw2` falls back to an E2M3 conversion in software on Ada, which took the decode step's NVFP4 expert MLP to 131 us per call, 27% of the DRAM bandwidth, while the MXFP8 one ran at 87%. The F16 forms of the eight E2M1 magnitudes differ only in their high byte, so one `prmt` looks up both values of a byte (`e2m1x2_to_f16x2_sm89`). It matches CUDA's conversion on all 256 bytes.

The memory plan has the same parts. The cluster stand-ins add 10.3 MB of module globals, inside `RUNTIME_BYTES`. The CUDA context takes 392 MiB with driver 595, against the 507 MiB of `CONTEXT_BYTES`, so `siphon-plan` without `--free` underestimates the free memory by 115 MiB there.

### Results on the RTX 4090

The kernel tests pass, but for the 9 skipped ones. Logits differ from the `sm_120a` build's: `tanhf`, `expf`, `sqrtf`, `rsqrtf`, `logf` and `exp2f` give the same bits on both GPUs, but BF16 and F16 `mma.sync` with F32 accumulators round their sums differently on Ada. So `smallm verify --hash 1` over `anyhow-chunked` has its own values there:

| Weights and cache | Width 1 | Width 3 | Width 8 |
| --- | --- | --- | --- |
| FP8 | `85c88c08dbe6b415` | `044d7bcabd13d3f3` | `118d8e3e8c5637e0` |
| FP8, `--output-projection fp8` | `2a0d5034f3af2068` | `37d27a7a9cabc0aa` | `94f10cdd3dceb7c5` |
| BF16 | `655ad0def7fe068c` | `95960bc6a6b86a70` | `97f25d91b288b3d5` |

The runs below read the FP8 copy of the output projection, now `--output-projection fp8` ([The output projection](#the-output-projection)). The quality gate with FP8 weights and cache, against MLX's chunked captures of `anyhow`, `half` and `regex-readme`, on the RTX 4090 and on an RTX 5090 with the same checkpoint:

| Mode | GPU | Mean KL | Worst range's \|log-prob difference\| | Verdict |
| --- | --- | --- | --- | --- |
| Prefill, chunks of 2048 | RTX 4090 | 2.0e-3, 1.2e-3, 1.2e-3 | 0.031, 0.033, 0.034 | pass |
| | RTX 5090 | 2.1e-3, 1.0e-3, 1.5e-3 | 0.030, 0.032, 0.036 | pass |
| Verify, 8 rows | RTX 4090 | 2.8e-3, 1.2e-3, 2.1e-3 | 0.035, 0.041, 0.044 | pass |
| | RTX 5090 | 2.6e-3, 1.1e-3, 3.0e-3 | 0.034, 0.038, 0.044 | pass |
| Stepping | both | 2.7e-3, 1.1e-3, 1.9e-3 | 0.034, 0.041, 0.0451 | fails `regex-readme`: limit 0.045 |

Stepping runs the GEMVs, not `mma.sync`, and gives the same numbers on both GPUs, so its failure on `regex-readme` comes from the FP8 path itself, not from Ada.

## References

Targets the official checkpoint at revision [`b85c3cfa`](https://huggingface.co/thinkingmachines/Inkling-2-Mini/tree/b85c3cfa615fc413cc9af3379f491b963f3b586a), with its NVFP4/MXFP8 experts, BF16 dense weights and bundled DSpark draft.

| Reference | Use |
| --- | --- |
| [Model card](https://thinkingmachines.ai/model-card/inkling-2-mini/) | Architecture, weight formats, tokenizer, chat template |
| [Ingine at `306eaa8a`](https://github.com/satellitedown/inkling-2-mini-abliterated-ingine/tree/306eaa8a90c9840f042a59d565f6dc4536bbb5ea/native/src) | Native CUDA kernels: low-bit expert tiles, expert-grouped prefill, relative-position attention, CUDA graphs. Apache-2.0: keep its license and notices with any copied code |
| [vLLM Inkling attention](https://github.com/vllm-project/vllm/blob/main/vllm/models/inkling2/nvidia/attention.py) and [DSpark speculator](https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/spec_decode/dspark/speculator.py) | Independent reference for numerics, paged KV and speculation |
| [MLX-LM #2086](https://github.com/ml-explore/mlx-lm/pull/2086) | Compact Python model definition, Metal path |
| [DSpark paper](https://arxiv.org/abs/2607.05147) | Draft scheduling and verifier semantics |

The measured control is Ingine with FP8 outer attention: 547, 439 and 286 tok/s median decode at 32K, 128K and 512K context on one RTX 5090.

## Milestones

1. Text-only, single-stream decode with the original per-visit KV layout, and token and logit parity against a pinned reference. Done.
2. Chunked prefill and prefix restore. Chunked prefill is done; prefix restore is not started.
3. Stock DSpark speculation. Greedy acceptance is done; probabilistic acceptance is not started.

## Reference captures

`reference/capture.py` runs MLX-LM on a Mac and records every layer visit's inputs and outputs, plus logits, for one prefill and a few greedy decode steps. The engine's parity tests compare against these files.

```sh
uv run --project reference python -I reference/capture.py <model-dir> reference/out/<name>.safetensors
```

With `--token-by-token`, the prompt goes through one token per step, so MLX runs the same single-row decode path as the engine. The `parity` binary replays such a capture on the GPU and reports, per step, the logit difference, the KL divergence from MLX, whether the greedy token matches, and the layer visit furthest from MLX:

```sh
cargo run --release -p siphon-cuda --bin parity -- <model-dir> reference/out/<name>.safetensors [--detail step016] [--kv bf16|fp8]
```

`--detail` prints every visit of one step with the number of differing BF16 values.

## Quality gate

Optimized kernels change summation order, so the engine will not stay bit-exact with MLX. The gate checks instead that the engine computes the same model: it runs the engine teacher-forced over long texts and compares it with MLX within a multiple of MLX's own noise. The texts in `reference/prompts` run past the sliding-window wrap at position 1024. The longest also runs past 8192, where the global layers start scaling queries.

`reference/capture_eval.py capture` runs MLX over a text on the Mac. It saves the token ids, MLX's log-probability of each actual next token, its top-1 token at every position, full logits at about 80 sampled positions, and 32 greedy tokens after the text. MLX runs the text in chunks of 512 by default, or one token per step with `--token-by-token`. `compare` reports how far two captures of the same text are apart:

```sh
uv run --project reference python -I reference/capture_eval.py capture <model-dir> reference/prompts/anyhow.txt reference/out/anyhow-chunked.safetensors
uv run --project reference python -I reference/capture_eval.py compare reference/out/anyhow-chunked.safetensors reference/out/anyhow-tbt.safetensors
```

The `eval` binary runs the engine over a capture on the GPU. It reports top-1 agreement, the mean absolute difference of the next-token log-probability, and perplexity, per range of positions. It also reports KL(MLX || engine) at the sampled positions, with the worst positions, and the greedy continuation match. It prints a verdict and exits with status 1 on failure:

```sh
cargo run --release -p siphon-cuda --bin eval -- <model-dir> reference/out/anyhow-chunked.safetensors [--kv bf16|fp8] [--prefill CHUNK]
```

With `--prefill CHUNK`, the engine runs the text through `Model::prefill` in chunks of CHUNK rows and scores every row's logits, then generates the continuation with `step`.

The thresholds are about 1.5 to 2 times MLX's disagreement between its chunked and token-by-token paths on the three texts:

| Check | MLX chunked vs token-by-token | Threshold |
| --- | --- | --- |
| Mean KL at sampled positions | 6.2e-4 to 2.0e-3 | 4e-3 |
| Max KL at sampled positions | 4.1e-3 to 5.9e-2 | 0.12 |
| Top-1 agreement, per range | 98.05% to 99.08% | 97% |
| Mean \|log-prob difference\|, per range | 0.021 to 0.031 | 0.045 |
| \|Perplexity ratio - 1\| | 6.8e-4 to 2.5e-3 | 5e-3 |

The greedy continuation is reported but not checked: one near-tie splits it, and MLX's two paths agree on only 12 of 32 tokens after `regex-readme.txt`.

The gate catches a sliding window that is not enforced. Run that way, MLX alone gives a mean KL of 9.5e-3 and a max KL of 0.21 on `anyhow.txt`. It cannot see query log-scaling: at 11,186 tokens the scale is at most 1.031, and MLX with log-scaling removed differs from MLX by less than its own noise (mean |log-prob difference| 0.006). A sliding window off by one position is also below the noise. A unit test covers log-scaling (`log_scale_matches_mlx`).

`capture --float32` runs MLX with F32 activations and dense weights, the packed expert weights unchanged, and every product at full F32 precision. MLX's F32 matmul on an M5 Max keeps only about 11 bits of inputs that are not BF16 values, so `reference/exact_products.py` splits those operands into three BF16 parts; run as a script, it checks each operation against F64. Captures made before that ran their chunked products at about TF32 precision, the NVFP4 expert products at one BF16 rounding, and differ from the exact ones by a mean |log-prob difference| of 0.020 to 0.022. A token-by-token F32 capture now matches the chunked one within 2e-5.

The F32 capture is a higher-precision reference: it shows whether a path is closer to or further from the model than MLX's own BF16 paths, which a BF16 reference cannot. Against the F32 captures, over all positions of the three texts:

| Path | Mean KL | \|log-prob difference\| | Top-1 |
| --- | --- | --- | --- |
| MLX chunked and token-by-token | 5.3e-4 to 1.8e-3 | 0.023 to 0.027 | 98.5% to 99.0% |
| Engine stepping, BF16 cache | 9.4e-4 to 1.5e-3 | 0.023 to 0.028 | 98.5% to 99.0% |
| Engine prefill, chunks of 2048, BF16 cache | 7.7e-4 to 1.4e-3 | 0.024 to 0.029 | 98.5% to 99.0% |
| Engine stepping, FP8 cache | 9.4e-4 to 1.5e-3 | 0.025 to 0.031 | 98.3% to 98.9% |
| Engine prefill, chunks of 2048, FP8 cache | 6.7e-4 to 1.3e-3 | 0.026 to 0.030 | 98.4% to 99.0% |

So the engine is as close to the model as MLX is, in both modes: its mean KL lies inside MLX's range, and its |log-prob difference| is at most 0.002 above the larger of MLX's two on the same text. The FP8 cache adds 0.001 to 0.003 to the mean |log-prob difference| when stepping and at most 0.002 in prefill. After `regex-readme.txt` the continuation splits at near-ties: against F32, MLX's chunked path keeps 12 of 32 tokens and its token-by-token path 26, the engine's BF16 stepping 0 and its prefill 32.
