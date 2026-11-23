# Inkling 2 Mini engine

A single-model inference engine for Inkling 2 Mini. The host code is Rust. GPU kernels are CUDA C++ in `crates/inkling-cuda/kernels`, compiled by `build.rs` to `sm_120a` cubins and loaded through cudarc.

| Crate | Contents |
| --- | --- |
| `inkling` | Config parsing, memory-mapped safetensors checkpoint, `inkling inspect` CLI |
| `inkling-cuda` | Kernels and their Rust launchers. Builds only on Linux |

## Build

On macOS only `inkling` builds; `inkling-cuda` compiles to an empty crate.

On Linux, pixi supplies `nvcc` and the driver comes from the host:

```sh
pixi run cargo test --workspace
pixi run cargo run --release -p inkling -- inspect <model-dir>
```

Set `NVCC` or `CUDA_HOME` to use another toolkit, and `INKLING_CUDA_ARCH` to target another GPU.

From a Mac, `scripts/remote.sh <ssh-host> <remote-dir> <command>` syncs the workspace to a Linux host and runs the command in its pixi environment.

## Kernel tests

Each `crates/inkling-cuda/src/ops/<group>.rs` launches the kernels in `kernels/<group>.cu` and is tested by `tests/<group>.rs`. The tests compare GPU output with MLX output saved in `tests/fixtures/<group>.safetensors`, which `reference/fixtures/<group>.py` regenerates:

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
INKLING_MODEL=<model-dir> cargo test --release -p inkling-cuda --test prefill -- --ignored --nocapture
```

## DSpark draft

`draft::Draft` runs the speculative draft bundled in the checkpoint (`mtp.safetensors`): a 3-layer dense transformer over one block of 8 rows, then a recurrent Markov head that turns the 7 mask rows into 7 greedy draft tokens. It reads the target's embedding, `embed_norm` and LM head, and folds `hidden_out` into the LM head at load in F64. Its kernels are in `kernels/draft.cu`; the BF16 products run through `ops::linear`.

- `Draft::commit(taps, start)` writes the draft cache entries of target positions `start..` from their four feature taps, BF16 `[rows, 4, 2304]`: the partial block entering visits 5, 23 and 41, and the final normed hidden state.
- `Draft::round(anchor)` runs the block at the next uncommitted position, with the anchor token read from device memory, and leaves the 7 ids in `Draft::draft_ids`.
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

`DraftOptions` switches on approximations. A draft token only changes how often the target accepts, never the target's output, so each one is kept only if it raises accepted tokens per millisecond:

- `fp8_backbone`: E4M3 copies of the block's and the commit's projections, through a multi-row FP8 product (`draft_gemm_fp8`).
- `fp8_head`: base logits from an E4M3 copy of the LM head.
- `fp8_fold`: the Markov logits from the E4M3 fold, without the exact correction.
- `shortlist`: Ingine's shortlist head. The base and Markov logits cover only the 32,768 highest-ranked tokens of a ranking file, the special tokens and up to 16,384 context tokens passed to `Draft::note_tokens`. The ranking is 65,536 little-endian u32 token ids, as Ingine's `draft_shortlist.bin`.

`dspark accept` measures each variant's teacher-forced acceptance on the stage 0 taps against MLX's top-1 tokens, as `reference/dspark.py accept` counts it, and times its round. Over all three texts (16,745 anchors), with the verify-time model of the design (5.7 ms with BF16 weights and 5.0 ms with FP8 at 3 drafts):

| Variant | Acceptance, drafts 1 to 3 | Tokens per round, k = 3 | Round, k = 3 | Tokens per ms, BF16 verify | Tokens per ms, FP8 verify |
| --- | --- | ---: | ---: | ---: | ---: |
| Exact | 79.2, 58.7, 45.2% | 2.832 | 1.175 ms | 0.412 | 0.459 |
| FP8 backbone | 79.2, 58.7, 45.2% | 2.832 | 1.077 ms | 0.418 | 0.466 |
| FP8 head | 79.3, 58.7, 45.2% | 2.832 | 0.916 ms | 0.428 | 0.479 |
| Shortlist | 79.2, 58.7, 45.2% | 2.830 | 0.650 ms | 0.446 | 0.501 |
| Shortlist, FP8 backbone, head and fold | 79.2, 58.5, 45.0% | 2.827 | 0.460 ms | 0.459 | 0.518 |

```sh
cargo run --release -p inkling-cuda --bin dspark -- accept <model-dir> <taps-dir> --shortlist <ranking> \
    --variants exact,shortlist+fp8-backbone+fp8-head+fp8-fold
cargo run --release -p inkling-cuda --bin dspark -- time <model-dir> <taps-dir>/regex-readme-taps.safetensors
INKLING_MODEL=<model-dir> INKLING_DSPARK_REF=<dir> cargo test --release -p inkling-cuda --test draft_exact -- --ignored --nocapture
```

The draft follows the rounding points of vLLM's draft rather than MLX's kernels. `reference/dspark.py` implements the same draft in MLX, in BF16 or F32, and reports teacher-forced acceptance on a text. `tests/draft.rs` checks each kernel against `reference/fixtures/dspark.py`. `tests/draft_round.rs` feeds the engine draft the MLX taps of `reference/prompts/regex-readme.txt` and compares it at every anchor with the MLX draft, within the MLX draft's own BF16 against F32 difference:

```sh
uv run --project reference python -I reference/capture.py <model-dir> reference/out/regex-readme-taps.safetensors --text reference/prompts/regex-readme.txt --taps
uv run --project reference python -I reference/dspark.py run <model-dir> reference/out/regex-readme-taps.safetensors reference/out/regex-readme-dspark-bf16.safetensors
uv run --project reference python -I reference/dspark.py run <model-dir> reference/out/regex-readme-taps.safetensors reference/out/regex-readme-dspark-f32.safetensors --float32
INKLING_MODEL=<model-dir> INKLING_DSPARK_REF=reference/out cargo test --release -p inkling-cuda --test draft_round -- --ignored --nocapture
```

`reference/dspark.py routing` reads captures made with `capture.py --routing` and reports the distinct experts that m consecutive positions route to at each visit, which sets the expert bytes of a verify of m rows.

## Speculative decoding

`spec::Speculator` runs greedy speculative decoding with the draft and a fixed number k of drafts per round (1 to 7):

1. `Model::verify` runs the anchor and the first k drafts as one pass of k + 1 rows through the chunked-prefill code, keeps the anchor and the longest run of drafts that each equal the previous row's greedy token, and rolls back the rest. It synchronizes once, to read the verdict.
2. The draft commits the feature taps of the kept rows and runs its block from the next anchor, which the verify left on the device.

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

`tests/linear_rows.rs`, `tests/experts_rows.rs`, `tests/attention_rows.rs` and `moe_router_step_rows_matches_step` in `tests/routing.rs` check each kernel's row invariance bit for bit, and its distance from the decode and prefill kernels.

The verify pass computes the same model as `Model::step` but sums in other orders, so the two greedy outputs split at the first near-tie: after 247 to 817 of 1024 tokens on the three texts.

`tests/verify.rs` checks the taps against MLX, a verify against stepping, the rollback against a shorter verify (bit for bit), the ring size, and row invariance. `tests/spec.rs` checks exactness with the draft at k = 1, 3 and 7 and with random drafts, 1024 tokens after each text. Both need the checkpoint:

```sh
INKLING_MODEL=<model-dir> INKLING_TAPS=reference/out/regex-readme-taps.safetensors \
    cargo test --release -p inkling-cuda --test verify --test spec -- --ignored --nocapture --test-threads 1
```

`eval --verify WIDTH` teacher-forces a capture through verify passes of WIDTH rows, and passes the quality gate at widths 1 and 4. `bench --prompt <capture> --spec K` compares speculation with plain decode from one prefilled state, and `--verify-widths` times one pass of each width.

One verify pass, after the `anyhow` prompt of 10,674 tokens or with the caches marked full to 32,768 or 131,072:

| Rows | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10,674, BF16 weights, ms | 6.41 | 6.93 | 7.62 | 8.15 | 8.67 | 9.30 | 9.64 | 10.03 |
| 10,674, FP8 weights, ms | 5.56 | 6.09 | 6.74 | 7.29 | 7.79 | 8.44 | 8.82 | 9.20 |
| 32,768, BF16 weights, ms | 6.95 | 7.59 | 8.44 | 8.91 | 9.56 | 10.42 | 10.79 | 11.22 |
| 32,768, FP8 weights, ms | 6.10 | 6.74 | 7.59 | 8.09 | 8.72 | 9.57 | 9.97 | 10.36 |
| 131,072, BF16 weights, ms | 9.29 | 10.21 | 11.88 | 12.48 | 13.32 | 15.10 | 15.64 | 16.19 |
| Bandwidth model, ms | 3.82 | 4.49 | 5.03 | 5.49 | 5.92 | 6.31 | 6.68 | 7.02 |

The bandwidth model reads the BF16 dense weights once and 8 m^0.6 distinct experts per visit at 84% of 1.70 TB/s, without KV. At 4 rows and 10,674 tokens with BF16 weights, a pass spends 3.37 ms in the experts, 2.1 ms in the dense products (0.55 ms of it the LM head), 1.0 ms in attention and about 2 ms in small kernels: depth attention, message gates, the router, the relative bias, conv states and norms.

`bench --spec 3`, 512 tokens after each text:

| Text, context | Plain decode | Speculation, BF16 weights | Speculation, FP8 weights | Tokens per round | Round with BF16 weights: verify + draft |
| --- | ---: | ---: | ---: | ---: | --- |
| `regex-readme`, 2,286 | 255 tok/s | 334 tok/s | 328 tok/s | 3.20 | 8.37 + 1.18 ms |
| `anyhow`, 10,674 | 239 tok/s | 395 tok/s | 426 tok/s | 3.92 | 8.68 + 1.21 ms |
| `anyhow`, 32,768 | 209 tok/s | 365 tok/s | 393 tok/s | 3.93 | 9.46 + 1.28 ms |

The context of 32,768 puts arbitrary token ids before the text. The greedy continuations of these texts repeat themselves, so the draft is right far more often than on chat: vLLM measured 58%, 33% and 20% there. At that acceptance, 2.1 tokens per round of about 9.6 ms would give 220 tok/s with BF16 weights. With `--weights fp8`, plain decode runs at 265 tok/s at context 0.

## Benchmarks and profiling

The `bench` binary times greedy decode at several context lengths. For each length it marks every cache as full to that length without computing it, runs warmup steps, then times `--steps` steps. It prints one markdown table row per context length, or JSON with `--json`, along with the time, the GPU clocks and the processes on the GPU. Numbers are comparable only on an otherwise idle GPU.

```sh
pixi run cargo build --release -p inkling-cuda --bin bench
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
cargo run --release -p inkling-cuda --bin parity -- <model-dir> reference/out/<name>.safetensors [--detail step016] [--kv bf16|fp8]
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
cargo run --release -p inkling-cuda --bin eval -- <model-dir> reference/out/anyhow-chunked.safetensors [--kv bf16|fp8] [--prefill CHUNK]
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
