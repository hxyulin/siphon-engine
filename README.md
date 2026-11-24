# Siphon

Siphon is an inference engine for one model, [Inkling 2 Mini](https://thinkingmachines.ai/model-card/inkling-2-mini/), on one NVIDIA RTX 5090. The host code is Rust; the kernels are CUDA C++ written for this model and this GPU (`sm_120a`). It aims to keep decode close to the time it takes to read the weights from memory. One sequence decodes with speculation; up to 8 decode together in batches.

It runs the official checkpoint as published: NVFP4 and MXFP8 routed experts, BF16 dense weights and the bundled DSpark draft. Its output is checked against MLX-LM, layer by layer and over long texts.

## Speed on an RTX 5090

Decode, ms per token:

| Dense weights, KV cache | Context 0 | 32K | 128K |
| --- | ---: | ---: | ---: |
| BF16, BF16 | 3.70 | 4.74 | 6.63 |
| FP8, FP8 | 2.73 (366 tok/s) | 3.24 | 4.23 |

The DRAM floor at context 0 is 3.25 ms with BF16 weights and 2.21 ms with FP8.

Prefill: 1.66 s for a 32K-token prompt (19,800 tok/s).

Greedy speculative decoding, FP8 weights and cache, 512 tokens, tok/s. The last column runs Inco AI's DFlash 2 drafter (`--drafter`) in place of the bundled DSpark draft:

| Prompt, context | Plain decode | DSpark draft | DFlash 2 |
| --- | ---: | ---: | ---: |
| Rust source (`anyhow`), 10.7K | 337 | 799 (2.37x) | 783 (2.32x) |
| Rust source (`anyhow`), 32K | 309 | 720 (2.33x) | 702 (2.27x) |
| Rust source (`half`), 2.3K | 353 | 722 (2.05x) | 600 (1.70x) |
| Rust source (`half`), 32K | 309 | 577 (1.87x) | 538 (1.74x) |
| A README (`regex`), 2.3K | 352 | 528 (1.50x) | 536 (1.52x) |
| A README (`regex`), 32K | 308 | 494 (1.60x) | 577 (1.87x) |
| A chat question, 30 tokens | 364 | 489 (1.34x) | 549 (1.51x) |
| A coding request, 40 tokens | 364 | 498 (1.37x) | 592 (1.63x) |

The speedup depends on how predictable the text is. The DSpark draft's first token is accepted 92% to 99% of the time on the Rust sources, 75% to 80% on the README and 66% to 70% on the chat answers; DFlash 2's, 77% to 82% on the chat answers.

## Features

- Decode as one CUDA graph per token, with programmatic dependent launch and an L2 prefetch stream for the next layer's weights.
- Chunked prefill of up to 2,048 rows on tensor-core GEMMs, an expert-grouped MoE GEMM and flash attention.
- Opt-in FP8 copies of the dense weights (`--weights fp8`), an FP8 global KV cache (`--kv fp8`) and quantized expert activations for prefill (`--expert-activations quantized`).
- Greedy speculative decoding with the DSpark draft. The verify pass gives each row the same result at every width, so speculation returns exactly the tokens that verifying one row at a time would. Each round picks its width from the drafts' confidences. With `--fallback` the speculator runs plain steps where the draft cannot predict the text, such as free prose; the output is then greedy along its own path and can differ between runs. `--drafter DIR` runs Inco AI's DFlash 2 drafter or a DFlash draft exported by the 2DSD project instead.
- Batched greedy decode of up to 8 sequences, each in its own KV cache, through the verify pass's multi-row kernels. Each sequence gets the same tokens as with speculation alone ([docs/batching.md](docs/batching.md)).
- An OpenAI-compatible server (`siphon-serve`): chat completions with streaming, reasoning and tool calls, and raw completions, with continuous batching.
- A quality gate that compares the engine with MLX over long texts, and parity tests per layer visit.

Not yet: sampling other than greedy, probabilistic acceptance, speculation for more than one sequence at a time, prefix caching.

## Build

Linux with an RTX 5090 and a driver for CUDA 13.5 or later. [pixi](https://pixi.sh) supplies `nvcc`; the driver comes from the host.

```sh
pixi run cargo build --release
pixi run cargo test --workspace
target/release/bench <model-dir> --contexts 0,32768 --weights fp8 --kv fp8
target/release/bench <model-dir> --prompt <capture> --spec auto --weights fp8 --kv fp8
```

`<model-dir>` holds the checkpoint from Hugging Face (`thinkingmachines/Inkling-2-Mini`).

The `Dockerfile` builds an image with `siphon-serve` as its entrypoint. The driver comes from the host through the NVIDIA container toolkit:

```sh
docker build -t siphon .
docker run --gpus all -p 8000:8000 -v <model-dir>:/models/inkling-2-mini:ro siphon \
  /models/inkling-2-mini --host 0.0.0.0 --port 8000
```

## Server

```sh
SIPHON_API_KEY=<key> target/release/siphon-serve <model-dir> --port 8000
```

Before loading anything, it plans the GPU memory from the config and the checkpoint headers. Without `--context`, it takes the longest context that fits the free memory less a 0.5 GiB margin, up to the model's 1,048,576 positions; `--memory <GiB>` caps the total it plans for. With `--context`, it stops and prints the plan if that context does not fit. It prints the plan in two lines at startup. On a 32 GB RTX 5090 with 0.45 GiB taken by the desktop, the longest contexts are:

| Dense weights, KV cache | DSpark draft | DFlash 2 |
| --- | ---: | ---: |
| FP8, FP8 | 466,944 | 563,200 |
| BF16, BF16 | 290,816 | 316,416 |

After warmup, it compares what the process uses with the plan and prints both. It reads its own buffers from the CUDA memory pool that every allocation comes from, and the rest, the module images and the CUDA graphs, from the change in free memory. It checks the pool's peak again after each job and warns if it goes past the plan.

`siphon-plan` shows the same plan in the terminal before you start the server, and runs on macOS too. It draws the GPU's memory as a bar and a table of what holds it, with the longest context of every combination of formats and drafts. Keys switch the formats and the draft and move the context. It reads the free memory from nvidia-smi, or takes `--free <GiB>`, or assumes an idle 32 GB RTX 5090. `--print` prints one frame as text:

```sh
target/release/siphon-plan <model-dir> [--drafter DIR] [--memory GiB] [--context N]
```

It loads the model with FP8 dense weights and KV cache (`--weights bf16` and `--kv bf16` switch back), prices the verify widths on a short prompt, and then serves:

| Route | Notes |
| --- | --- |
| `POST /v1/chat/completions` | Messages rendered as the checkpoint's chat template renders them. Streams with `stream`, and `stream_options.include_usage` adds a usage chunk. The model's thinking comes back as `reasoning_content`. |
| `POST /v1/completions` | A text prompt or token ids, streaming or not. The text includes any special tokens generated. |
| `POST /tokenize`, `POST /detokenize` | vLLM's shape, also at `/v1/tokenize` and `/v1/detokenize`: `{"prompt"}` or `{"messages", "tools"}` to `{"tokens", "count", "max_model_len"}`, and `{"tokens"}` to `{"prompt"}`. |
| `POST /v1/token/encode`, `POST /v1/token/decode` | TabbyAPI's shape: `{"text"}`, a string or messages, to `{"tokens", "length"}`, and `{"tokens"}` to `{"text"}`. |
| `GET /v1/models`, `GET /v1/models/{id}` | The served model, with `max_model_len`. |
| `GET /health`, `GET /version` | No API key needed. |

With `--streams N` (1 to 8, default 1), up to N requests run at once and up to `--queue` (8) more wait. Each stream has its own KV cache of `--context` positions, and without `--context` the server takes the longest context at which N caches fit. A request that arrives at an idle server speculates. When another arrives, the first stops speculating after its round, and every running request decodes in a batched step of one row each. When one request is left and nothing waits, it speculates again. A new request's prompt prefills in chunks of 2,048 tokens between the batch's steps. Either way each request gets the same tokens it would get alone. FP8 weights and cache, tok/s in all and per stream:

| Context of each stream | 1 stream | 2 | 4 | 8 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 300 | 457, 229 | 666, 166 | 971, 121 |
| 10,674 | 275 | 408, 204 | 576, 144 | 797, 100 |
| 32,768 | 254 | 373, 186 | 513, 128 | 683, 85 |

A batch of one runs about 23% slower than plain decode, and a request that speculates alone runs at 490 to 800 tok/s, so batching pays from about three streams on.

Chat requests take these fields:

- `reasoning_effort`: a name such as `low`, or 0 to 0.99; the default is 0.9. Without it, `chat_template_kwargs.reasoning_effort` counts, and `chat_template_kwargs.enable_thinking: false` means effort 0. The template has no switch that turns thinking off, so at effort 0 the model still writes a short thinking block.
- `tools` and `tool_choice`: `auto`, `none`, `required`, or `{"type": "function", "function": {"name": ...}}`. A named function, or `required` with one tool, ends the prompt with that tool's call opened. `required` with several tools opens a call without a name, and the model names the tool in the call. A forced call skips the thinking block. The reply ends after the named call, or at the first text after the calls `required` forced. `parallel_tool_calls: false` ends the reply after its first call.
- `stop`: a string or an array of strings. Stop strings match the reply's content, not its thinking or tool calls. Text that could still begin a stop string is held back while streaming. The reply is cut before the match, finishes with `stop`, and the job ends.
- vLLM's extensions: `return_token_ids` adds `prompt_token_ids` and each choice's `token_ids`, streaming or not; the ids end with `<|content_model_end_sampling|>` when the model ended its turn. Each choice has `stop_reason`, the stop string that matched or null. `continue_final_message` with `add_generation_prompt: false` continues the last assistant message.

Raw completions take `stop` and `return_token_ids` too. A text prompt, like the text sent to the tokenizer routes, is encoded with special-token spellings as plain text and nothing added; send token ids to use special tokens. Decoding keeps special tokens.

Decoding is greedy, so `temperature`, `top_p`, `top_k`, `min_p`, the penalties, `seed` and other sampling fields are accepted and ignored, as are `user`, `skip_special_tokens` and unknown fields. Fields the server cannot honour get a 400: `n` or `best_of` above 1, `logprobs`, `top_logprobs`, `prompt_logprobs`, `echo`, `suffix`, `logit_bias`, `ignore_eos`, `min_tokens`, `stop_token_ids`, `response_format` other than `text`, vLLM's guided decoding fields, and the deprecated `functions` and `function_call`. With `SIPHON_API_KEY` set, every route but `/health` and `/version` needs `Authorization: Bearer <key>`. Set `NVCC` or `CUDA_HOME` to use another toolkit, and `SIPHON_CUDA_ARCH` to target another GPU.

On macOS, `siphon-cuda` compiles to an empty crate and `siphon-serve` builds without its engine, so its tests run but it cannot serve. From a Mac, `scripts/remote.sh <ssh-host> <remote-dir> <command>` syncs the workspace to a Linux host and runs the command in its pixi environment.

## Layout

| Path | Contents |
| --- | --- |
| `crates/siphon` | Config parsing, the memory-mapped safetensors checkpoint, the GPU memory planner, and the `siphon inspect` CLI |
| `crates/siphon-cuda` | Kernels (`kernels/*.cu`), their Rust launchers (`src/ops`), the model, the draft and speculation, and the `bench`, `eval`, `parity` and `dspark` binaries |
| `crates/siphon-serve` | The HTTP server: chat rendering, output parsing and the engine thread |
| `crates/siphon-plan` | `siphon-plan`, a terminal view of the memory plan |
| `reference` | MLX-LM captures and kernel fixtures, run on a Mac with uv |
| `scripts` | Remote sync, the PDL read-order audit, and profiling scripts |
| `docs/engine.md` | How the engine works, how it is tested and measured, and the formats it uses |

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT), at your option.

Unless you explicitly state otherwise, any contribution you submit for inclusion in this work, as defined in the Apache-2.0 license, is dual licensed as above, without any additional terms or conditions.

## Acknowledgements

- [MLX-LM](https://github.com/ml-explore/mlx-lm)'s Inkling model is the numerical reference.
- [vLLM](https://github.com/vllm-project/vllm)'s Inkling kernels define the activation quantization that the quantized prefill path follows.
- [Ingine](https://github.com/satellitedown/inkling-2-mini-abliterated-ingine), another engine for this model, informed the speculation design: confidence-chosen verify widths and a shortlist draft head.
- [CUTLASS](https://github.com/NVIDIA/cutlass)'s sm_120 grouped GEMM informed the tile shape of the quantized expert GEMM.
