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

## Benchmarks and profiling

The `bench` binary times greedy decode at several context lengths. For each length it marks every cache as full to that length without computing it, runs warmup steps, then times `--steps` steps. It prints one markdown table row per context length, or JSON with `--json`, along with the time, the GPU clocks and the processes on the GPU. Numbers are comparable only on an otherwise idle GPU.

```sh
pixi run cargo build --release -p inkling-cuda --bin bench
target/release/bench <model-dir> [--contexts 0,4096,32768] [--passes N] [--steps 100] [--warmup 10] [--json]
```

Without `--contexts` it runs 0, 4096 and 32768 tokens, and 131072 if the global KV cache (32 KiB per token) fits in free memory. Bandwidth columns divide 5.5 GB of weights per token, plus the KV bytes read, by the step time, as a share of the RTX 5090's 1.79 TB/s.

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
cargo run --release -p inkling-cuda --bin parity -- <model-dir> reference/out/<name>.safetensors [--detail step016]
```

`--detail` prints every visit of one step with the number of differing BF16 values.
