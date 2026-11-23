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
