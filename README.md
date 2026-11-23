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

- `chunk` is 1 to 2048 (`MAX_PREFILL_CHUNK`). The default, `DEFAULT_PREFILL_CHUNK`, is 2048, the fastest measured: bigger chunks read each expert weight once for more rows. MLX's chunked captures use 512. The first call allocates buffers for its chunk size: about 0.28 GB at 512 rows, 0.5 GB at 1024 and 0.95 GB at 2048.
- `PrefillLogits::Last` computes only the last row's logits, which `Model::logits` then returns. `PrefillLogits::All(callback)` also passes every row's F32 logits after the softcap to the callback, with the row's position. The LM head runs over 32 rows at a time for it.
- `Model::reset` zeroes the caches and returns to position 0 for a new sequence.

The small kernels have multi-row forms that give the single-row results bit for bit. The GEMMs, expert products and attention go through `Ctx::gemm_rows`, `Ctx::moe_rows` and `Ctx::attention_rows` in `src/model.rs`. `gemm_rows` runs a tensor-core GEMM. `moe_rows` runs an expert-grouped GEMM, and `attention_rows` a flash attention over the cache and the chunk. `tests/prefill.rs` checks prefill against stepping on 1300 tokens at chunks of 512, 1024, 433 and 2048. At 2048 the whole prompt is one chunk wider than the sliding window. It needs the checkpoint and about 20 GB of GPU memory, so it is ignored by default:

```sh
INKLING_MODEL=<model-dir> cargo test --release -p inkling-cuda --test prefill -- --ignored --nocapture
```

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

Decode reads 3.51 GB of BF16 dense weights per token: every visit's attention input and output projections, and the LM head. `bench`, `eval` and `parity` take `--weights bf16|fp8`, and `Model::with_formats` takes the same choice:

- `bf16`, the default, reads the checkpoint's weights.
- `fp8` quantizes those matrices at load time on the GPU to E4M3, with one F32 scale per row, `amax / 448`. Decode then reads 1.76 GB for them. Prefill keeps the BF16 weights, since its GEMMs are compute-bound. The copies take 1.11 GB of device memory: one per layer, since the looped layers' three visits share theirs.

The FP8 path lowers precision on purpose. Each byte is the BF16 weight over its row's scale, rounded to the nearest E4M3 value. The products widen each value exactly, sum in F32, and multiply the row's scale into the sum once before rounding to BF16. The message gates, the router and the small vectors stay as they are. `tests/linear_fp8.rs` checks the quantizer against a CPU model and the products against an F64 reference over the quantized weights.

Engine stepping with the BF16 cache, against the F32 captures of `anyhow`, `half` and `regex-readme`, over all positions:

| Dense weights | Mean KL | \|log-prob difference\| | Top-1 | Decode at context 0 |
| --- | --- | --- | --- | --- |
| `bf16` | 2.0e-3, 7.1e-4, 1.6e-3 | 0.023, 0.028, 0.027 | 98.9%, 98.5%, 98.4% | 4.09 ms |
| `fp8` | 2.8e-3, 1.1e-3, 2.5e-3 | 0.030, 0.036, 0.039 | 98.7%, 98.1%, 98.0% | 3.05 ms |

The FP8 path still passes the gate against MLX's BF16 captures.

The pixi environment includes Nsight Systems (`nsys`, CLI only) and Nsight Compute (`ncu`) from conda-forge. `ncu` is pinned to 2025.3, the last release built for CUDA 13.0, which the pinned `nvcc` requires. A kernel trace of 20 steps:

```sh
pixi run nsys profile -t cuda --cuda-memory-usage=false -o decode \
    target/release/bench <model-dir> --contexts 32768 --warmup 5 --steps 20
pixi run nsys export --type sqlite -o decode.sqlite decode.nsys-rep
```

The trace marks the end of each step with a 4-byte device-to-host copy, the sampled token. `ncu` needs the GPU performance counters, which the driver limits to administrators unless the `NVreg_RestrictProfilingToAdminUsers=0` module option is set. Without it, `ncu` fails with `ERR_NVGPUCTRPERM`.

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

1. Text-only, single-stream decode with the original per-visit KV layout, and token and logit parity against a pinned reference.
2. Chunked prefill and prefix restore.
3. Stock DSpark speculation.

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

`capture --float32` runs MLX with F32 activations and dense weights, the packed expert weights unchanged. It is a higher-precision reference: it shows whether a path is closer to or further from the model than MLX's own BF16 paths, which a BF16 reference cannot. Against the F32 captures, over all positions of the three texts:

| Path | Mean KL | \|log-prob difference\| | Top-1 |
| --- | --- | --- | --- |
| MLX chunked and token-by-token | 6.0e-4 to 2.3e-3 | 0.023 to 0.027 | 98.5% to 99.0% |
| Engine stepping, BF16 cache | 7.6e-4 to 2.3e-3 | 0.023 to 0.027 | 98.6% to 99.0% |
| Engine prefill, chunks of 512, BF16 cache | 4.6e-4 to 2.1e-3 | 0.023 to 0.027 | 98.4% to 98.8% |
| Engine stepping, FP8 cache | 6.4e-4 to 1.7e-3 | 0.026 to 0.030 | 98.3% to 98.8% |
| Engine prefill, chunks of 512, FP8 cache | 9.0e-4 to 2.1e-3 | 0.025 to 0.031 | 98.2% to 98.9% |

So the engine is as close to the model as MLX is, in both modes. The FP8 cache adds about 0.003 to the mean |log-prob difference|. MLX's chunked path also keeps only 12 of 32 continuation tokens after `regex-readme.txt` against F32, as does the engine's BF16 stepping.
