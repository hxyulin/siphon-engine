<div align="center">

# Siphon

**An inference engine for one model on one GPU: [Inkling 2 Mini](https://thinkingmachines.ai/model-card/inkling-2-mini/) on an NVIDIA RTX 5090.**

Rust host code, CUDA kernels written for this model and `sm_120a`, and an OpenAI-compatible server.

[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue)](#license)
[![Rust 1.89+](https://img.shields.io/badge/rust-1.89%2B-orange?logo=rust)](https://www.rust-lang.org)
[![CUDA 13.5](https://img.shields.io/badge/CUDA-13.5-76B900?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![GPU: RTX 5090](https://img.shields.io/badge/GPU-RTX%205090%20%28sm__120a%29-76B900?logo=nvidia)](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/)
[![Model: Inkling 2 Mini](https://img.shields.io/badge/model-Inkling%202%20Mini-FFD21E?logo=huggingface)](https://huggingface.co/thinkingmachines/Inkling-2-Mini)
[![Image](https://github.com/hxyulin/siphon-engine/actions/workflows/image.yml/badge.svg)](https://github.com/hxyulin/siphon-engine/actions/workflows/image.yml)

| 889 tok/s | 999 tok/s | 81–88% | 456,704 | 0.10 s |
| :---: | :---: | :---: | :---: | :---: |
| one stream, speculative,<br>on Rust source | 8 streams together | of the DRAM floor<br>in plain decode | tokens of context<br>on a 32 GB card | to first token of a<br>103K-token follow-up |

</div>

## Why Siphon

- **Close to the hardware limit.** One plain decode step takes 2.73 ms with FP8 weights and cache, 81% of the time it takes to read the weights once from memory. Decode runs as one CUDA graph per token, with programmatic dependent launch and an L2 prefetch stream for the next layer's weights.
- **The official checkpoint, unchanged.** Siphon runs the checkpoint as published: NVFP4 and MXFP8 routed experts, BF16 dense weights and the bundled DSpark draft. It builds optional FP8 copies of the dense weights at load time.
- **Speculation without changing the output.** The verify pass gives each row the same result at every width, so speculative decoding returns exactly the tokens that verifying one row at a time would. Batched decode keeps that property: a request gets the same tokens alone or next to seven others.
- **Checked against a reference.** A quality gate compares the engine with MLX-LM over long texts, and parity tests compare it layer by layer.

## Quick start

Siphon needs Linux, an RTX 5090 and a driver for CUDA 13.5 or later. Download the checkpoint from Hugging Face ([`thinkingmachines/Inkling-2-Mini`](https://huggingface.co/thinkingmachines/Inkling-2-Mini)), then build and run the server image. The NVIDIA container toolkit passes the host's driver in:

```sh
docker build -t siphon .
docker run --gpus all -p 8000:8000 -v <model-dir>:/models/inkling-2-mini:ro siphon \
  /models/inkling-2-mini --host 0.0.0.0 --port 8000
```

Prebuilt images are published as `ghcr.io/hxyulin/siphon-engine`, tagged `sha-<commit>`, with build provenance attestations.

At startup the server plans the GPU memory, takes the longest context that fits, loads the model and prices its verify widths. Then it serves the OpenAI API:

```sh
curl http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "inkling-2-mini",
  "messages": [{"role": "user", "content": "Explain how a hash map handles collisions."}],
  "reasoning_effort": "low"
}'
```

[docs/server.md](docs/server.md) lists the options, routes and request fields.

## Performance

All numbers come from one RTX 5090 (32 GB, 600 W) running main at d60a6c9, with no other process on the GPU. Speculation and batching use FP8 dense weights and an FP8 KV cache, the server's default.

### Decode

One token per step, 400 steps. The DRAM floor is the time to read the weights and the KV cache once at 1.70 TB/s.

| Dense weights, KV cache | Context 0 | 32K | 128K |
| --- | ---: | ---: | ---: |
| FP8, FP8 | 2.73 ms · 367 tok/s · 81% | 3.23 ms · 310 tok/s · 81% | 4.23 ms · 236 tok/s · 85% |
| BF16, BF16 | 3.77 ms · 265 tok/s · 86% | 4.75 ms · 211 tok/s · 84% | 6.67 ms · 150 tok/s · 88% |

### Speculative decoding

Greedy, 512 tokens, tok/s and speedup over plain decode. The draft proposes up to 7 tokens per round, and each round picks its verify width from the drafts' confidences. The last column runs Inco AI's DFlash 2 drafter (`--drafter`) instead of the bundled DSpark draft.

| Prompt | Context | Plain decode | DSpark draft | DFlash 2 | Tokens per round |
| --- | ---: | ---: | ---: | ---: | ---: |
| Rust source (`anyhow`) | 10.7K | 336 | **889** · 2.64x | 849 · 2.53x | 6.66 · 6.65 |
| Rust source (`anyhow`) | 32K | 310 | **817** · 2.64x | 807 · 2.61x | 6.56 · 6.65 |
| Rust source (`half`) | 2.3K | 353 | **796** · 2.26x | 662 · 1.88x | 5.67 · 4.94 |
| Rust source (`half`) | 32K | 310 | **683** · 2.21x | 637 · 2.06x | 5.37 · 5.15 |
| A README (`regex`) | 2.3K | 352 | 587 · 1.67x | **632** · 1.79x | 3.66 · 4.30 |
| A README (`regex`) | 32K | 310 | 566 · 1.83x | **624** · 2.01x | 4.19 · 4.87 |
| A chat question | 30 | 364 | 556 · 1.53x | **599** · 1.65x | 3.37 · 4.13 |
| A coding request | 40 | 363 | 508 · 1.40x | **583** · 1.61x | 3.01 · 3.82 |

The speedup depends on how predictable the text is. The DSpark draft's first token is accepted 89% to 96% of the time on the Rust sources, 78% to 86% on the README and 66% to 70% on the chat answers. DFlash 2 drafts prose and chat better, and the bundled draft drafts code better.

### Concurrent requests

`siphon-serve --streams N` runs up to 8 requests at once, each in its own KV cache. A request alone speculates. When others arrive, all of them decode in batched steps of one row each, and a request left alone speculates again. Tok/s in all, with tok/s per stream in brackets:

| Context of each stream | 1 stream | 2 streams | 4 streams | 8 streams |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 318 | 476 (238) | 689 (172) | **999** (125) |
| 10,674 | 286 | 412 (206) | 579 (145) | **804** (100) |
| 32,768 | 264 | 376 (188) | 519 (130) | **696** (87) |

A batch of one takes 3.14 ms per step against 2.71 ms for plain decode, because the batch keeps the verify pass's numerics. A request that speculates alone runs at 510 to 890 tok/s, so batching serves more in total from about three streams on.

### Prefill and the prefix cache

Prefill runs in chunks of 2,048 rows on tensor-core GEMMs, an expert-grouped MoE GEMM and flash attention. Past 16,384 keys, global attention over the FP8 cache runs on FP8 tensor cores.

| Prompt | Time to first token | Prefill tok/s |
| ---: | ---: | ---: |
| 8,192 tokens | 0.51 s | 16,150 |
| 32,768 tokens | 2.11 s | 15,510 |
| 131,072 tokens | 9.91 s | 13,220 |

Chat clients resend the whole conversation each turn. The server keeps the state of the last prompt at its last multiple of 2,048 tokens. A follow-up turn prefills only its new tokens and gets the same tokens and logits as without the cache. Time to the first token of a follow-up turn over HTTP:

| Conversation | Without the cache | With it | Speedup |
| ---: | ---: | ---: | ---: |
| 8,174 tokens | 0.53 s | 0.14 s | 3.8x |
| 102,848 tokens | 7.70 s | 0.10 s | 79x |

### Context length

The server plans its GPU memory from the config and the checkpoint headers before it loads anything. Without `--context` it takes the longest context that fits, up to the model's 1,048,576 positions. With 30.45 GiB free after the CUDA context, the longest contexts are:

| Dense weights, KV cache | DSpark draft | DFlash 2 |
| --- | ---: | ---: |
| FP8, FP8 | 456,704 | 547,840 |
| BF16, BF16 | 284,672 | 308,224 |

`siphon-plan <model-dir>` shows the plan in a terminal before you start the server, and runs on macOS too.

## Features

- **Server.** OpenAI chat completions and completions, with streaming, reasoning (`reasoning_content`), tool calls, `tool_choice`, stop strings and `reasoning_effort`. vLLM and TabbyAPI tokenizer routes, and vLLM's `return_token_ids`, `ignore_eos` and `continue_final_message`.
- **Scheduling.** Continuous batching of up to 8 requests. Speculation for a request that runs alone, and a one-entry prefix cache.
- **Speculation.** The bundled DSpark draft by default, or Inco AI's DFlash 2 drafter or a 2DSD DFlash export with `--drafter DIR`. `--fallback` runs plain steps where the draft cannot predict the text. It speeds up free prose, but the output can then differ between runs.
- **Formats.** FP8 dense weights (`--weights fp8`) and an FP8 global KV cache (`--kv fp8`) by default, BF16 for both on request. Prefill can opt in to quantized expert activations and FP8 dense products, which are faster and fail the quality gate on some texts.
- **Memory planning.** The server checks the planned GPU memory against what it uses after warmup and after each job.

Not yet: sampling other than greedy (sampling fields are accepted and ignored), speculation for more than one request at a time, a prefix cache with more than one entry, and image input.

## Build from source

[pixi](https://pixi.sh) supplies `nvcc`. The driver comes from the host.

```sh
pixi run cargo build --release
pixi run cargo test --workspace
target/release/siphon-serve <model-dir> --port 8000
```

The `bench` binary measures decode, prefill, speculation and batched decode:

```sh
target/release/bench <model-dir> --contexts 0,32768 --weights fp8 --kv fp8
target/release/bench <model-dir> --prefill 8192,32768 --weights fp8 --kv fp8
target/release/bench <model-dir> --prompt <capture> --spec auto --weights fp8 --kv fp8
target/release/bench <model-dir> --batch 1,2,4,8 --prompt <capture> --contexts 0 --weights fp8 --kv fp8
```

On macOS, `siphon-cuda` compiles to an empty crate, so the tests run but the server cannot serve. `scripts/remote.sh <ssh-host> <remote-dir> <command>` syncs the workspace to a Linux host and runs the command there.

## Project layout

| Path | Contents |
| --- | --- |
| `crates/siphon` | Config parsing, the memory-mapped safetensors checkpoint, the GPU memory planner and the `siphon inspect` CLI |
| `crates/siphon-cuda` | Kernels (`kernels/*.cu`), their Rust launchers (`src/ops`), the model, the drafts and speculation, and the `bench`, `eval`, `parity` and `dspark` binaries |
| `crates/siphon-serve` | The HTTP server: chat rendering, output parsing and the engine thread |
| `crates/siphon-plan` | `siphon-plan`, a terminal view of the memory plan |
| `reference` | MLX-LM captures and kernel fixtures, run on a Mac with uv |
| `scripts` | Remote sync, the PDL read-order audit and profiling scripts |
| [`docs/server.md`](docs/server.md) | Server options, routes and request fields |
| [`docs/engine.md`](docs/engine.md) | How the engine works, how it is tested and measured, and its formats |
| [`docs/batching.md`](docs/batching.md) | The design of batched decode |

## Acknowledgements

- [MLX-LM](https://github.com/ml-explore/mlx-lm)'s Inkling model is the numerical reference.
- [vLLM](https://github.com/vllm-project/vllm)'s Inkling kernels define the activation quantization that the quantized prefill path follows.
- [Ingine](https://github.com/satellitedown/inkling-2-mini-abliterated-ingine), another engine for this model, informed the speculation design: confidence-chosen verify widths and a shortlist draft head.
- [CUTLASS](https://github.com/NVIDIA/cutlass)'s sm_120 grouped GEMM informed the tile shape of the quantized expert GEMM.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT), at your option. Unless you explicitly state otherwise, any contribution you submit for inclusion in this work, as defined in the Apache-2.0 license, is dual licensed as above, without any additional terms or conditions.
