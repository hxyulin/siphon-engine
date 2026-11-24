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

With an FP8 cache, global attention runs on FP8 tensor cores by default, and the expert and dense products can too; [Prefill on FP8 tensor cores](#prefill-on-fp8-tensor-cores) has the options.

The small kernels have multi-row forms. Most give the single-row results bit for bit; depth attention, the router logits and the message gates' products sum in their own orders and differ by F32 rounding. The GEMMs, expert products and attention go through `Ctx::gemm_rows`, `Ctx::moe_rows` and `Ctx::attention_rows` in `src/model.rs`. `gemm_rows` runs a tensor-core GEMM. `moe_rows` runs an expert-grouped GEMM, with the softcapped SwiGLU in the `w13` product's epilogue, and `attention_rows` a flash attention over the cache and the chunk. `tests/prefill.rs` checks prefill against stepping on 1300 tokens at chunks of 512, 1024, 433 and 2048. At 2048 the whole prompt is one chunk wider than the sliding window. `fp8_prefill_matches_stepping` repeats it with the FP8 cache and weights and every FP8 prefill product on. The tests need the checkpoint and about 20 GB of GPU memory, so they are ignored by default:

```sh
SIPHON_MODEL=<model-dir> cargo test --release -p siphon-cuda --test prefill -- --ignored --nocapture
```

## DSpark draft

`draft::Draft` runs the speculative draft bundled in the checkpoint (`mtp.safetensors`): a 3-layer dense transformer over one block of 8 rows, then a recurrent Markov head that turns the 7 mask rows into 7 greedy draft tokens. It reads the target's embedding, `embed_norm` and LM head, and folds `hidden_out` into the LM head at load in F64. Its kernels are in `kernels/draft.cu`; the BF16 products run through `ops::linear`.

- `Draft::commit(taps, start)` writes the draft cache entries of target positions `start..` from their four feature taps, BF16 `[rows, 4, 2304]`: the partial block entering visits 5, 23 and 41, and the final normed hidden state.
- `Draft::round(anchor)` runs the block at the next uncommitted position, with the anchor token read from device memory, and leaves the 7 ids in `Draft::draft_ids` and their confidences in `Draft::confidences`.
- Each draft's confidence is its softmax probability under the Markov logits, `exp(logit - 30) / sum exp(logit_c - 30)`: the softcap bounds every logit by 30, so the sum needs no running maximum. The kernel that scans every row (`draft_markov_bound`, `draft_markov_fold` or `draft_markov_fold_any`) writes a partial sum per block, and the next step's pick divides. The pruned argmax sums the logits of the E4M3 estimates and a shortlist head sums only its rows, so the confidence is an estimate; the round takes no measurable extra time.
- `Draft::round_inputs()` holds a round's inputs in device memory: the taps of up to 8 positions to commit, BF16 `[8, 4, 2304]`, and `control`, I32 `[start, rows, anchor]`. `Draft::run_round(steps)` commits those positions, runs the block right after them, and runs the base logits and Markov head for `steps` drafts (1 to 7). `Draft::replay_round(steps)` replays the same work as a CUDA graph, captured once per `steps`. A verify pass can write the inputs on the device and launch the draft graph right after its own; no position goes through the host. `Draft::set_round_inputs` fills the inputs from the host.
- Layers 0 and 2 keep rings of 1032 slots, the window of 1024 and the block. Layer 1 keeps every position up to the capacity, 4 KiB each.

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

`DraftOptions` switches on approximations. A draft token only changes how often the target accepts, never the target's output, so each one is kept only if it raises accepted tokens per millisecond. `DraftOptions::fast` turns on all four and is the draft speculation uses (`Speculator::new`, `bench --spec`, `siphon-serve`); `DraftOptions::fp8` leaves out the shortlist, and `DraftOptions::default` stays exact:

- `fp8_backbone`: E4M3 copies of the block's and the commit's projections, through a multi-row FP8 product (`draft_gemm_fp8`).
- `fp8_head`: base logits from an E4M3 copy of the LM head.
- `fp8_fold`: the Markov logits from the E4M3 fold, without the exact correction.
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
- 5 layers, every one attending to every position. The cache takes 20 KiB per position, against 4 KiB for the bundled draft's one global layer.
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
- Selector. The drafter has no LM head. Rows 1 to 7 go through the target's LM head, and the top 16 logits over the 200,058 real tokens become each row's candidates. Walking from the anchor, candidate c after token t scores `30 tanh((logit_c + bf16(<bf16(A[t] h), B[c]>)) / 30)`, with `h = hidden_projection · y` (rank 256). The softcap applies after the correction (`selector_softcap_mode: after_correction`, which the patch adds; z-lab caps the logits before adding). The greedy walk takes the best score, ties to the larger logit. A draft's confidence, which `WidthPolicy` reads, is its softmax probability over its row's 16 scores: the distribution a walk at temperature 1 samples from.

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

`spec::Speculator` runs greedy speculative decoding with the draft. Each round verifies the anchor and its leading drafts, a width of 1 to 8 rows that `spec::Width` sets: a fixed k drafts (`Width::Fixed`), or a width per round from the drafts' confidences (`Width::Adaptive`).

1. `Model::start_verify` launches the anchor and the round's drafts as one pass through the chunked-prefill code. On the device it keeps the anchor and the longest run of drafts that each equal the previous row's greedy token, and rolls back the rest.
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

A round emits the kept drafts and the target's greedy token after them. `Model::prefill_with_taps` records the draft's taps of every prompt row, so the draft's cache covers the prompt.

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

## Batched decode

`Model::add_slots` adds sequence slots, each with its own global caches, sliding rings and conv states of slot 0's sizes. `Model::prefill_slot` prefills a slot, and `Model::step_batch` runs the next token of 1 to 8 slots as one pass through the verify pass's code, with each row reading and writing its own slot through a table of device addresses per visit. Each row's logits equal, bit for bit, a one-row verify of its sequence, so a sequence decoded in batches emits the tokens of greedy speculation. [batching.md](batching.md) has the design, the kernel changes and the measurements.

siphon-serve returns a request left alone in a batch to speculation. The draft follows the request that left speculation: batched steps record their rows' taps, and the speculator commits the followed row's (`Speculator::follow`, `keep_batched`, `resume`). A request without draft state gets it rebuilt by a prefill with taps into a free slot (`Speculator::rebuild`), and `Model::move_slot` moves a request to slot 0. [batching.md](batching.md#returning-to-speculation) has the costs.

`tests/batch.rs` decodes five prompts of 40 to 2,500 tokens that join the batch at different steps, and checks every token and every row's logits against one-row verifies, with BF16 and FP8 weights and caches. Two more tests speculate, batch and speculate again, with the bundled draft and with the drafter in `SIPHON_DFLASH`, and check every token:

```sh
SIPHON_MODEL=<model-dir> cargo test --release -p siphon-cuda --test batch -- --ignored --nocapture --test-threads 1
```

`bench --batch 1,2,4,8 --prompt <capture>` times batched steps at each context against a plain step and a one-row verify. Row r feeds the capture's tokens from offset r / 8 of its length on, so the rows route their experts as different texts do: 8 rows from 8 texts touch about 50 distinct experts per visit, against 29 for 8 consecutive rows of one text.

## Prefix cache

Chat clients and agents resend the whole conversation each turn. Without a cache, a follow-up turn on 90,607 tokens of history prefills all of them again in 7.6 s. HTTP, the chat template and the tokenizer take about 37 ms of a 97K-token request.

`Speculator::prefill_cached` keeps the state of the last prompt it prefilled, and a prompt that starts with the same tokens prefills only the rest. The state of a sequence at position `p` is:

| State | After later tokens |
| --- | --- |
| Global KV caches of slot 0, slots below `p` | Unchanged: later tokens write slots from their own position on |
| Sliding rings and conv states of slot 0 | Overwritten |
| The draft's global caches, positions below `p` | Unchanged: commits and rounds write from the committed position on |
| The draft's rings | Overwritten |
| The draft's Markov context, the shortlist's noted tokens, the speculator's prompt tail | Rebuilt by every prefill |

So the cache copies only what later tokens overwrite: slot 0's rings and conv states (`Model::save_local_state`, 164 MiB) and the draft's rings (`Draft::save_rings`, 8 MiB for the bundled draft, 40 MiB for DFlash 2, none for a 2DSD DFlash draft). The global caches stay where they are. A hit restores the copies, and the draft starts committing from the cached position. A miss resets slot 0 and prefills from position 0.

Prefill sums differ with the chunk boundaries, so the copies are taken at a chunk boundary: the last multiple of the prefill chunk (2,048) below the prompt's end, so at least one row remains for the logits. The speculator prefills up to that boundary, saves the copies and prefills the rest. Prefill splits chunks only at the end of its tokens, so both halves run the chunks of one prefill from position 0, and the only extra work is the first half's last-row logits. A later prompt that starts with the cached tokens restores them and prefills from the boundary in the same chunks. Its caches, logits, tokens and drafts equal those of a prefill from position 0 bit for bit. The cost is that a follow-up prefills up to 2,047 tokens of the previous prompt again, plus the previous reply and its own new tokens.

Only the last prompt is kept, and only slot 0's. The copies remember which sequence slot 0 held: `Model::reset`, `restore_state` and a `move_slot` into slot 0 start a new one, and a restore needs slot 0 to hold the same sequence (`Model::can_restore_local_state`). A plain `Speculator::prefill`, and so a `rebuild`, empties the cache, since the draft then holds another sequence. In siphon-serve, a request that runs alone prefills through the cache. A request that joins a batch takes slot 0 only when the other slots are busy, so a batch of up to `--streams` - 1 new requests leaves the cache in place. A request that left speculation for a batch keeps slot 0, and its prompt stays cached.

`tests/prefix.rs` checks a cached prefill against a prefill from position 0 with the bundled draft and with the drafter in `SIPHON_DFLASH`: the first token's logits, 97 tokens and the draft's acceptance per round must all match. It covers a miss, a hit, a prompt that diverges after the cached boundary, an exact repeat whose remainder is shorter than the 1,024-slot window, a batch beside slot 0, a batch that takes slot 0 and a rebuild. Skipping the restore of slot 0's rings makes the first token's logits differ, and skipping the draft's rings changes the rounds from 16 to 44 with the same tokens:

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

More entries would help several conversations that alternate. Each entry needs its own global caches, which slot 0 provides for the one entry. With `--streams` above 1 the idle slots could hold more entries, moved to slot 0 on a hit (`Model::move_slot`, 16 KiB per position with an FP8 cache).

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

`siphon::memory` works out the device memory a configuration needs before any weight is loaded, from the config, the safetensors headers (`Checkpoint::open` maps the files and reads only the headers), the weight and KV formats, the draft, the prefill chunk, the number of streams and the capacity. Each stream gets its own global caches of the capacity, rings and conv states, and more than one stream adds the batched step's buffers. It counts:

- the target's `model.llm.*` tensors as uploaded, with the router, the depth-attention queries and the static conv weights widened to F32. The vision tower is not loaded. The bundled draft's `mtp.safetensors` is loaded only with the bundled draft;
- the FP8 copies of `--weights fp8`, 1.14 GB;
- the global KV caches: 32 KiB per position, or 16 KiB with `--kv fp8`, over the 8 global visits;
- the 40 sliding rings of 1,024 slots (160 MiB), the conv states and the decode step's vocabulary-sized buffers;
- the prefill buffers, 0.38 MB per chunk row plus 64 MB, and the verify pass's;
- the draft's tensors, the bundled draft's fold, the copies its approximations make (the shortlist rows of the LM head, the FP8 backbone), its rings, its buffers and the speculator's tap tails;
- the draft's global caches, 4 KiB per position and layer: one layer of the bundled draft, every layer of a 2DSD DFlash draft, none of DFlash 2, which keeps rings of 2,056 slots;
- the prefix cache's copies of one slot's rings and conv states and of the draft's rings (see [Prefix cache](#prefix-cache)), 171 MiB with the bundled draft and 203 MiB with DFlash 2;
- `RUNTIME_BYTES`, 128 MiB for the module images, the CUDA graphs, the small buffers and the allocator's rounding. It was measured with a thread that sampled `cuMemGetInfo` every millisecond while `siphon-serve` loaded, prefilled more than a chunk and generated speculatively; the uncounted part came to 69 to 87 MiB.

The CUDA context itself takes 507 MiB. `siphon-serve` reads the free memory right after creating it, so the plan must fit that free memory less a 0.5 GiB margin. `Model::save_state` copies only the filled slots of the global caches, so pricing the verify widths at startup does not double the cache.

`siphon-serve` also checks the plan against what it uses. It allocates the full prefill chunk's buffers at startup (`Model::reserve_prefill`), so every buffer the plan counts exists once the warmup has run. Every buffer comes from the device's memory pool, because cudarc allocates with `cuMemAllocAsync`. `Gpu::memory` reads the pool's reserved bytes and their peak (`cuMemPoolGetAttribute`) along with the device's free memory. The bytes outside the pool, the module images and the CUDA graphs, are the drop in free memory since the context was created less the pool's bytes, measured once after warmup. The check prints the pool's peak plus those bytes against the plan, and warns once if a later job takes the pool's peak past the plan.

`siphon-plan` draws the plan in the terminal. It plans every combination of weight format, KV format and draft (the bundled draft, a `--drafter`, none) with `siphon::memory`. It takes the free memory after the context from nvidia-smi's free memory less the 507 MiB context, which is how the context was measured.

Planned against measured, as the free memory after creating the context less the lowest sample, after a 3,820-token prompt and 256 speculative tokens:

| Weights, KV cache | Draft | Capacity | Planned GiB | Measured GiB | Over, MiB |
| --- | --- | ---: | ---: | ---: | ---: |
| FP8, FP8 | DSpark | 65,536 | 22.284 | 22.244 | 41 |
| FP8, FP8 | DSpark | 262,144 | 26.034 | 25.994 | 41 |
| FP8, FP8 | DSpark | 466,944 (largest) | 29.940 | 29.900 | 41 |
| FP8, FP8 | DFlash 2 | 65,536 | 22.350 | 22.307 | 45 |
| FP8, FP8 | DFlash 2 | 262,144 | 25.350 | 25.307 | 45 |
| FP8, FP8 | DFlash 2 | 563,200 (largest) | 29.944 | 29.900 | 45 |
| BF16, BF16 | DSpark | 65,536 | 22.222 | 22.180 | 43 |
| BF16, BF16 | DSpark | 131,072 | 24.472 | 24.430 | 43 |
| BF16, BF16 | DSpark | 290,816 (largest) | 29.956 | 29.898 | 59 |
| BF16, BF16 | DFlash 2 | 65,536 | 22.288 | 22.242 | 47 |
| BF16, BF16 | DFlash 2 | 131,072 | 24.288 | 24.242 | 47 |
| BF16, BF16 | DFlash 2 | 316,416 (largest) | 29.944 | 29.898 | 47 |

At the largest capacity, `siphon-serve` also prefilled real prompts near the end of the cache and generated 512 tokens speculatively from there, with the same peak memory:

| Weights, KV cache | Draft | Prompt tokens | Prefill |
| --- | --- | ---: | ---: |
| FP8, FP8 | DSpark | 461,517 | 81.1 s |
| FP8, FP8 | DFlash 2 | 556,964 | 111.1 s |
| BF16, BF16 | DSpark | 287,138 | 38.5 s |

## KV cache format

The global layers (5, 11, 17 and 23, 8 of the 48 layer visits) keep every earlier key and value, so at long context their attention reads most of the bytes of a step. `bench`, `eval` and `parity` take `--kv bf16|fp8`, and `Model::with_kv_format` takes the same choice:

- `bf16`, the default, stores the keys and values as computed.
- `fp8` stores the global caches as E4M3, which halves the bytes global attention reads. The sliding-window caches stay BF16.

The FP8 cache has no scale. Keys are L2-clipped to norm √512 and values are RMS-normed per head, so no element exceeds 22.7, far inside E4M3's ±448. The append kernel rounds each BF16 value to the nearest E4M3 value, ties to even. Attention decodes E4M3 exactly with the hardware converter, so it differs from BF16 only by that rounding. Ingine stores its FP8 global caches the same way.

## Dense weight format

Decode reads 3.62 GB of BF16 dense weights per token: every visit's attention input and output projections, the looped layers' message gates (113 MB), and the LM head. `bench`, `eval` and `parity` take `--weights bf16|fp8`, and `Model::with_formats` takes the same choice:

- `bf16`, the default, reads the checkpoint's weights.
- `fp8` quantizes those matrices at load time on the GPU to E4M3, with one F32 scale per row, `amax / 448`. Decode then reads 1.82 GB for them. Prefill keeps the BF16 weights unless `--prefill-dense fp8` (see [Prefill on FP8 tensor cores](#dense-products)). The copies take 1.14 GB of device memory: one per layer, since the looped layers' three visits share theirs.

The FP8 path lowers precision on purpose. Each byte is the BF16 weight over its row's scale, rounded to the nearest E4M3 value. The products widen each value exactly, sum in F32, and multiply the row's scale into the sum once before rounding to BF16. The router and the small vectors stay as they are. `tests/linear_fp8.rs` checks the quantizer against a CPU model and the products against an F64 reference over the quantized weights.

Engine stepping with the BF16 cache, against the F32 captures of `anyhow`, `half` and `regex-readme`, over all positions:

| Dense weights | Mean KL | \|log-prob difference\| | Top-1 | Decode at context 0 |
| --- | --- | --- | --- | --- |
| `bf16` | 1.5e-3, 9.4e-4, 1.1e-3 | 0.023, 0.028, 0.028 | 99.0%, 98.6%, 98.5% | 4.09 ms |
| `fp8` | 2.3e-3, 1.5e-3, 2.2e-3 | 0.029, 0.038, 0.040 | 98.7%, 98.2%, 98.0% | 3.05 ms |

The FP8 path's mean KL is 1.5 to 2.0 times the BF16 path's, and its \|log-prob difference\| 1.3 to 1.4 times. It still passes the gate against MLX's BF16 captures.

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

Three prefill options run products on the block-scaled FP8 MMA (`mma.sync ... kind::mxf8f6f4.block_scale`, m16n8k32). With F32 accumulators it runs at 1,015 TFLOP/s on the RTX 5090, against 254 for BF16; without block scales the same MMA runs at 509. Each option rounds activations to E4M3, so each went through the quality gate. Decode never uses them. `bench` and `eval` take the flags, and the `Model` setters take the same choices:

| Option | Flag | Setter | Default |
| --- | --- | --- | --- |
| Global attention | `--prefill-attention bf16\|fp8` | `Model::set_prefill_attention` | `fp8`, with an FP8 cache |
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
