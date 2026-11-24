# Siphon

Siphon is an inference engine for one model, [Inkling 2 Mini](https://thinkingmachines.ai/model-card/inkling-2-mini/), on one NVIDIA RTX 5090. The host code is Rust; the kernels are CUDA C++ written for this model and this GPU (`sm_120a`). It serves one sequence at a time and aims to keep decode close to the time it takes to read the weights from memory.

It runs the official checkpoint as published: NVFP4 and MXFP8 routed experts, BF16 dense weights and the bundled DSpark draft. Its output is checked against MLX-LM, layer by layer and over long texts.

## Speed on an RTX 5090

Decode, ms per token:

| Dense weights, KV cache | Context 0 | 32K | 128K |
| --- | ---: | ---: | ---: |
| BF16, BF16 | 3.68 | 4.75 | 6.65 |
| FP8, FP8 | 2.77 (361 tok/s) | 3.32 | 4.30 |

The DRAM floor at context 0 is 3.25 ms with BF16 weights and 2.21 ms with FP8.

Prefill: 1.66 s for a 32K-token prompt (19,800 tok/s).

Greedy speculative decoding, FP8 weights and cache, 512 tokens, tok/s. The last column runs Inco AI's DFlash 2 drafter (`--drafter`) in place of the bundled DSpark draft:

| Prompt, context | Plain decode | DSpark draft | DFlash 2 |
| --- | ---: | ---: | ---: |
| Rust source (`anyhow`), 10.7K | 327 | 799 (2.44x) | 779 (2.38x) |
| Rust source (`anyhow`), 32K | 301 | 712 (2.36x) | 699 (2.32x) |
| Rust source (`half`), 2.3K | 340 | 721 (2.12x) | 590 (1.74x) |
| Rust source (`half`), 32K | 301 | 575 (1.91x) | 536 (1.78x) |
| A README (`regex`), 2.3K | 340 | 526 (1.55x) | 532 (1.56x) |
| A README (`regex`), 32K | 301 | 495 (1.65x) | 575 (1.91x) |
| A chat question, 30 tokens | 356 | 491 (1.38x) | 541 (1.53x) |
| A coding request, 40 tokens | 356 | 501 (1.41x) | 594 (1.67x) |

The speedup depends on how predictable the text is. The DSpark draft's first token is accepted 92% to 99% of the time on the Rust sources, 75% to 80% on the README and 66% to 70% on the chat answers; DFlash 2's, 76% to 82% on the chat answers.

## Features

- Decode as one CUDA graph per token, with programmatic dependent launch and an L2 prefetch stream for the next layer's weights.
- Chunked prefill of up to 2,048 rows on tensor-core GEMMs, an expert-grouped MoE GEMM and flash attention.
- Opt-in FP8 copies of the dense weights (`--weights fp8`), an FP8 global KV cache (`--kv fp8`) and quantized expert activations for prefill (`--expert-activations quantized`).
- Greedy speculative decoding with the DSpark draft. The verify pass gives each row the same result at every width, so speculation returns exactly the tokens that verifying one row at a time would. Each round picks its width from the drafts' confidences. `--drafter DIR` runs Inco AI's DFlash 2 drafter or a DFlash draft exported by the 2DSD project instead.
- An OpenAI-compatible server (`siphon-serve`): chat completions with streaming, reasoning and tool calls, and raw completions.
- A quality gate that compares the engine with MLX over long texts, and parity tests per layer visit.

Not yet: sampling other than greedy, probabilistic acceptance, batching several sequences, prefix caching, stop sequences.

## Build

Linux with an RTX 5090 and a driver for CUDA 13.5 or later. [pixi](https://pixi.sh) supplies `nvcc`; the driver comes from the host.

```sh
pixi run cargo build --release
pixi run cargo test --workspace
target/release/bench <model-dir> --contexts 0,32768 --weights fp8 --kv fp8
target/release/bench <model-dir> --prompt <capture> --spec auto --weights fp8 --kv fp8
```

`<model-dir>` holds the checkpoint from Hugging Face (`thinkingmachines/Inkling-2-Mini`).

## Server

```sh
SIPHON_API_KEY=<key> target/release/siphon-serve <model-dir> --port 8000 --context 65536
```

It loads the model with FP8 dense weights and KV cache (`--weights bf16` and `--kv bf16` switch back), prices the verify widths on a short prompt, and then serves:

| Route | Notes |
| --- | --- |
| `POST /v1/chat/completions` | Messages rendered as the checkpoint's chat template renders them, with `stream`, `tools` and `reasoning_effort` (a name such as `low`, or 0 to 0.99; default 0.9). The model's thinking comes back as `reasoning_content`. Without streaming, `return_token_ids` also returns `prompt_token_ids` and each choice's `token_ids`, as vLLM does. |
| `POST /v1/completions` | A text prompt or token ids, without streaming. |
| `GET /v1/models`, `GET /health` | |

Requests run one at a time, in arrival order, and up to `--queue` (8) wait. Decoding is greedy, so `temperature`, `top_p` and other sampling fields are ignored. With `SIPHON_API_KEY` set, `/v1` routes need `Authorization: Bearer <key>`. Set `NVCC` or `CUDA_HOME` to use another toolkit, and `SIPHON_CUDA_ARCH` to target another GPU.

On macOS, `siphon-cuda` compiles to an empty crate and `siphon-serve` builds without its engine, so its tests run but it cannot serve. From a Mac, `scripts/remote.sh <ssh-host> <remote-dir> <command>` syncs the workspace to a Linux host and runs the command in its pixi environment.

## Layout

| Path | Contents |
| --- | --- |
| `crates/siphon` | Config parsing, the memory-mapped safetensors checkpoint, and the `siphon inspect` CLI |
| `crates/siphon-cuda` | Kernels (`kernels/*.cu`), their Rust launchers (`src/ops`), the model, the draft and speculation, and the `bench`, `eval`, `parity` and `dspark` binaries |
| `crates/siphon-serve` | The HTTP server: chat rendering, output parsing and the engine thread |
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
