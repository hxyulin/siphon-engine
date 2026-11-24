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

The small kernels have multi-row forms. Most give the single-row results bit for bit; depth attention, the router logits and the message gates' products sum in their own orders and differ by F32 rounding. The GEMMs, expert products and attention go through `Ctx::gemm_rows`, `Ctx::moe_rows` and `Ctx::attention_rows` in `src/model.rs`. `gemm_rows` runs a tensor-core GEMM. `moe_rows` runs an expert-grouped GEMM, with the softcapped SwiGLU in the `w13` product's epilogue, and `attention_rows` a flash attention over the cache and the chunk. `tests/prefill.rs` checks prefill against stepping on 1300 tokens at chunks of 512, 1024, 433 and 2048. At 2048 the whole prompt is one chunk wider than the sliding window. It needs the checkpoint and about 20 GB of GPU memory, so it is ignored by default:

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

`DraftOptions` switches on approximations. A draft token only changes how often the target accepts, never the target's output, so each one is kept only if it raises accepted tokens per millisecond. `DraftOptions::fast` turns on all four, with the shortlist in token id order; speculation (`Speculator::new`, `bench --spec`) uses it, and `DraftOptions::default` stays exact:

- `fp8_backbone`: E4M3 copies of the block's and the commit's projections, through a multi-row FP8 product (`draft_gemm_fp8`).
- `fp8_head`: base logits from an E4M3 copy of the LM head.
- `fp8_fold`: the Markov logits from the E4M3 fold, without the exact correction.
- `shortlist`: Ingine's shortlist head. The base and Markov logits cover only the 32,768 highest-ranked tokens, the special tokens and up to 16,384 context tokens passed to `Draft::note_tokens`. The speculator notes the tokens of the prompt's last 8,192 positions and each round's emitted tokens, so a long prompt cannot fill every context row and shut out the tokens that generation adds. The ranking defaults to token id order (`draft::id_ranking`): the tokenizer numbers its tokens in BPE merge order, so the first 32,768 ids are the earliest merges. A ranking file of little-endian u32 token ids, as Ingine's `draft_shortlist.bin`, can replace it.

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

FP8 fold alone saves nothing, since the pruned argmax already reads the E4M3 fold; it matters with a shortlist, which skips the pruning. The counted ranking took 72 million tokens of open-source Rust crates and the CPython standard library, the evaluation texts' crates left out. Code-heavy counts drop common prose tokens that the id order keeps.

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

Thresholds from 0.5 to 0.7 and the cost rule land within 2% of each other; the cost rule keeps that place when the verify cost per row changes, without retuning. Widths 2, 4 and 8 alone lose about 1.5%. The policy can pick width 1, a verify of one row, but plain `Model::step` decoding cannot stand in for it: decode sums in other orders than the verify path, so the output would depend on the policy.

A round emits the kept drafts and the target's greedy token after them. `Model::prefill_with_taps` records the draft's taps of every prompt row, so the draft's cache covers the prompt.

The verify pass has one CUDA graph per width, captured on first use, with the tokens, the start position and the acceptance mode in device buffers. `kernels/spec.cu` holds its own kernels:

- Before the pass, it copies every conv state (3.7 MB) and the sliding-ring slots the pass will overwrite (40 visits, 4 KiB per row each). During the pass, each visit records its conv inputs: the raw keys and values, the attention output projection, and the MoE combine.
- After the pass, every row's greedy token, then the acceptance. Each conv state becomes the last 3 rows of its old state followed by the kept rows' inputs. The rejected rows' ring slots get their old keys and values back. The global caches need nothing: attention reads only slots below each row's position, and later rows overwrite the stale ones.

The rings keep 1024 slots. A ring of 1032 slots would make the backup unnecessary, but decode attention splits its cache by slot, so its results change with the ring size.

Each verify row gives the same logits, bit for bit, as a verify of that row alone at its position, whatever the other rows hold. So speculation emits exactly the tokens of greedy decoding through the verify pass one row at a time, whatever the draft proposes. The pass runs kernels built for 1 to 8 rows, whose rows each sum in an order that depends only on their own position and data:

- Dense products (`ops::linear_rows`): each 16-row weight tile streams once for all rows, through per-warp `cp.async` rings onto `mma.sync m16n8k16`, with the weights as the 16-row operand and the rows as the 8 columns. With `--weights fp8` the verify reads the FP8 copies, widened exactly to BF16.
- Router (`Routing::moe_router_step_rows`): the decode router's sums for every row, so each row's experts and weights equal a step's bit for bit.
- Experts (`ops::experts_rows`): one kernel groups the row-slot pairs by expert on the device and streams each distinct expert's `w13` and `w2` once, on tensor cores with the expert's pairs as the columns. An E2M1 or E4M3 weight times its block scale is an exact BF16 value.
- Attention (`Attention::attention_rows`): each key and value tile streams once for the 3 query heads and all rows. The cache splits by absolute slot, a row's masked keys weigh exactly zero, and the slot a row's key goes to shows the new key to that row and later rows and the old entry to earlier ones. The rows' keys append after attention, so 1024-slot rings work.

Around these kernels the pass follows a decode step's schedule, in fewer launches than chunked prefill uses:

- With FP8 dense weights it forks the decode step's L2 prefetches: the first visit's projections before the pass, and after each visit's experts the next visit's output projection, MLP gate and router, and the projections of the visit after it. With BF16 weights the prefetch made every width slower, so it stays off.
- Up to 8 rows, a visit ends in one kernel, `Conv::moe_conv_residual_step_rows`: the MoE combine, the post-MLP conv and the residual add, the shifts of all four conv states, and the record of the verify's conv inputs. `attention_post_rows_bf16` also computes the relative bias, and the attention combine also gates the output and appends the rows' keys and values. A pass of one row runs 762 kernels instead of 1,146.
- The attention copies its first key and value tile before `pdl_wait`: no kernel of the pass writes a visit's cache before that visit's attention.
- The router runs as the decode router's cluster, every row's sums side by side. The expert kernel loads each stage's block scales with its weights, and verifies of 1 to 3 rows run it with a 4-slot copy ring instead of 3.

None of these changes a result: the verify logits stay the same bit for bit, which `smallm verify --width M --hash 1` checks across builds by hashing every pass's logits.

`tests/linear_rows.rs`, `tests/experts_rows.rs`, `tests/attention_rows.rs` and `moe_router_step_rows_matches_step` in `tests/routing.rs` check each kernel's row invariance bit for bit, and its distance from the decode and prefill kernels.

The verify pass computes the same model as `Model::step` but sums in other orders, so the two greedy outputs split at the first near-tie: after 247 to 817 of 1024 tokens on the three texts.

`tests/verify.rs` checks the taps against MLX, a verify against stepping, the rollback against a shorter verify (bit for bit), the ring size, and row invariance. `tests/spec.rs` checks exactness with the draft at k = 1, 3 and 7 and at an adaptive width priced so that every width from 1 to 8 gets picked, and with random drafts at k = 3 and at widths cycling from 1 to 8, 1024 tokens after each text. Both need the checkpoint:

```sh
SIPHON_MODEL=<model-dir> SIPHON_TAPS=reference/out/regex-readme-taps.safetensors \
    cargo test --release -p siphon-cuda --test verify --test spec -- --ignored --nocapture --test-threads 1
```

`eval --verify WIDTH` teacher-forces a capture through verify passes of WIDTH rows, and passes the quality gate at widths 1 and 4. `bench --prompt <capture> --spec K|auto` compares speculation with plain decode from one prefilled state, `--draft` picks the draft variant, `--sweep` records drafts and confidences at every anchor, and `--verify-widths` times one pass of each width.

One verify pass, after the `anyhow` prompt of 10,674 tokens or with the caches marked full to 32,768 or 131,072, the fastest of 40, ms. Before is the pass without the schedule above:

| Rows | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10,674, FP8 weights and cache | 4.12 | 4.83 | 5.43 | 5.95 | 6.50 | 7.19 | 7.69 | 8.17 |
| Before | 5.33 | 5.90 | 6.62 | 7.13 | 7.63 | 8.27 | 8.66 | 9.04 |
| 32,768, FP8 weights and cache | 4.78 | 5.60 | 6.22 | 6.75 | 7.37 | 8.23 | 8.76 | 9.28 |
| Before | 5.80 | 6.46 | 7.36 | 7.90 | 8.49 | 9.28 | 9.70 | 10.16 |
| 10,674, BF16 weights and cache | 5.38 | 6.04 | 6.65 | 7.20 | 7.75 | 8.48 | 8.88 | 9.29 |
| Before | 6.29 | 6.87 | 7.54 | 8.06 | 8.57 | 9.20 | 9.60 | 9.95 |
| 32,768, BF16 weights and cache | 5.91 | 6.62 | 7.43 | 8.08 | 8.58 | 9.45 | 9.93 | 10.50 |
| Before | 6.82 | 7.49 | 8.46 | 8.84 | 9.47 | 10.29 | 10.73 | 11.15 |
| 131,072, BF16 weights, FP8 cache | 8.54 | 9.73 | 10.48 | 11.12 | 11.86 | 13.50 | 14.15 | 14.76 |
| Before | 8.76 | 9.64 | 11.40 | 12.10 | 12.80 | 14.39 | 14.99 | 15.48 |
| Bandwidth model | 3.82 | 4.49 | 5.03 | 5.49 | 5.92 | 6.31 | 6.68 | 7.02 |

A plain decode step takes 2.78 ms at context 0 and 3.31 ms at 32,768 with FP8 weights and cache, and 3.69 and 4.75 ms with BF16. The bandwidth model reads the BF16 dense weights once and 8 m^0.6 distinct experts per visit at 84% of 1.70 TB/s, without KV.

In a kernel trace of one row at 10,674 tokens with FP8 weights and cache (each kernel from the previous one's end), a pass spans 4.14 ms against 3.06 ms for a decode step. The difference: the attention chain takes 0.91 ms against 0.51 (post-processing, attention and combine against one fused decode kernel), the NVFP4 and MXFP8 experts 1.48 ms against 1.32, the message gates 0.42 ms in three kernels per gate against 0.23 in one, the input projections 5.5 to 9.7 us each against 3.3 us for the decode GEMV, and the visit's last kernel 0.10 ms, which decode folds into its expert kernel. At 3 rows the pass spans 5.85 ms, 2.69 ms of it in the experts.

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

The same runs with the verify schedule above (after) and without it (before), `--spec auto`, one run each:

| Text, context | Weights | Plain decode | Before | After | Tokens and verify rows per round, after |
| --- | --- | ---: | ---: | ---: | --- |
| `anyhow`, 10,674 | FP8 | 327 | 625 | 675 (2.06x) | 6.12, 7.14 |
| `anyhow`, 32,768 | FP8 | 302 | 525 | 572 (1.90x) | 5.82, 7.20 |
| `half`, 2,273 | FP8 | 340 | 570 | 637 (1.87x) | 5.35, 6.60 |
| `half`, 32,768 | FP8 | 301 | 408 | 454 (1.51x) | 4.41, 6.29 |
| `regex-readme`, 2,286 | FP8 | 340 | 343 | 404 (1.19x) | 2.84, 4.59 |
| `regex-readme`, 32,768 | FP8 | 301 | 335 | 371 (1.23x) | 3.44, 5.65 |
| `anyhow`, 10,674 | BF16 | 241 | 566 | 601 (2.50x) | 6.32, 7.49 |
| `anyhow`, 32,768 | BF16 | 210 | 520 | 552 (2.63x) | 6.40, 7.33 |
| `half`, 2,273 | BF16 | 257 | 519 | 565 (2.20x) | 5.57, 6.87 |
| `half`, 32,768 | BF16 | 210 | 406 | 440 (2.10x) | 4.92, 6.58 |
| `regex-readme`, 2,286 | BF16 | 256 | 341 | 380 (1.48x) | 3.37, 5.50 |
| `regex-readme`, 32,768 | BF16 | 210 | 263 | 288 (1.37x) | 2.83, 4.93 |

The context of 32,768 puts arbitrary token ids before the text. The greedy continuations of these texts repeat themselves, so the draft is right far more often than on chat: vLLM measured 58%, 33% and 20% there. At low acceptance the policy verifies 2 to 4 rows; it cannot fall back to plain decode, whose output differs from the verify path's.

## Benchmarks and profiling

The `bench` binary times greedy decode at several context lengths. For each length it marks every cache as full to that length without computing it, runs warmup steps, then times `--steps` steps. It prints one markdown table row per context length, or JSON with `--json`, along with the time, the GPU clocks and the processes on the GPU. Numbers are comparable only on an otherwise idle GPU.

```sh
pixi run cargo build --release -p siphon-cuda --bin bench
target/release/bench <model-dir> [--contexts 0,4096,32768] [--passes N] [--steps 100] [--warmup 10] [--kv bf16|fp8] [--weights bf16|fp8] [--json]
```

Without `--contexts` it runs 0, 4096 and 32768 tokens, and 131072 if the global KV cache (32 KiB per token, 16 KiB with `--kv fp8`) fits in free memory. Bandwidth columns divide 5.5 GB of weights per token (3.75 GB with `--weights fp8`), plus the KV bytes read, by the step time, as a share of the RTX 5090's 1.79 TB/s.

With `--prefill`, `bench` times `Model::prefill` instead, once per prompt length from an empty cache, after one warmup chunk. It reports the time to the first token. The prompt is arbitrary token ids, so expert routing differs from real text:

```sh
target/release/bench <model-dir> --prefill 512,4096,32768 [--chunk 512] [--kv bf16|fp8] [--json]
```

## KV cache format

The global layers (5, 11, 17 and 23, 8 of the 48 layer visits) keep every earlier key and value, so at long context their attention reads most of the bytes of a step. `bench`, `eval` and `parity` take `--kv bf16|fp8`, and `Model::with_kv_format` takes the same choice:

- `bf16`, the default, stores the keys and values as computed.
- `fp8` stores the global caches as E4M3, which halves the bytes global attention reads. The sliding-window caches stay BF16.

The FP8 cache has no scale. Keys are L2-clipped to norm √512 and values are RMS-normed per head, so no element exceeds 22.7, far inside E4M3's ±448. The append kernel rounds each BF16 value to the nearest E4M3 value, ties to even. Attention decodes E4M3 exactly with the hardware converter, so it differs from BF16 only by that rounding. Ingine stores its FP8 global caches the same way.

## Dense weight format

Decode reads 3.62 GB of BF16 dense weights per token: every visit's attention input and output projections, the looped layers' message gates (113 MB), and the LM head. `bench`, `eval` and `parity` take `--weights bf16|fp8`, and `Model::with_formats` takes the same choice:

- `bf16`, the default, reads the checkpoint's weights.
- `fp8` quantizes those matrices at load time on the GPU to E4M3, with one F32 scale per row, `amax / 448`. Decode then reads 1.82 GB for them. Prefill keeps the BF16 weights, since its GEMMs are compute-bound. The copies take about 1.17 GB of device memory: one per layer, since the looped layers' three visits share theirs.

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

Prefill's expert products take `--expert-activations bf16|quantized` in `bench` and `eval`, and `Model::set_expert_activations` takes the same choice. Decode always takes BF16 activations.

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
