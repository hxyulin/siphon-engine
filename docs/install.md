# Installation and CLI

Use a local copy of the [official Inkling 2 Mini checkpoint](https://huggingface.co/thinkingmachines/Inkling-2-Mini). The commands below use `$MODEL_DIR` for its directory.

## Linux with CUDA

Use an NVIDIA driver compatible with CUDA 13.5. Siphon targets the RTX 5090 and RTX PRO 6000 Blackwell by default. For an RTX 4090, use the source build below with `SIPHON_CUDA_ARCH=sm_89`.

### Container

Install the NVIDIA Container Toolkit, then build and run the image.

```sh
docker build -t siphon .
docker run --gpus all -p 127.0.0.1:8000:8000 -v "$MODEL_DIR:/model:ro" siphon \
  /model --host 0.0.0.0
```

Prebuilt images are published as `ghcr.io/hxyulin/siphon-engine` and tagged `sha-<commit>`.

### Build from source

[pixi](https://pixi.sh) supplies `nvcc`. The driver comes from the host.

```sh
pixi run cargo build --release -p siphon-cli
target/release/siphon serve "$MODEL_DIR"
```

For an RTX 4090, build with `SIPHON_CUDA_ARCH=sm_89 pixi run cargo build --release -p siphon-cli`. See the [hardware recipes](recipes.md#rtx-4090) for launch options and [the engine notes](engine.md#rtx-4090-sm_89) for kernel differences.

## Apple Silicon

Install the pinned MLX-LM runtime with [uv](https://docs.astral.sh/uv/). The server keeps the model loaded in a Python worker and serves one text sequence at a time. The default context limit is 8,192 tokens per request.

```sh
uv sync --project reference --locked
export SIPHON_MLX_PYTHON="$PWD/reference/.venv/bin/python"
cargo run --release -p siphon-cli -- serve "$MODEL_DIR"
```

## CLI tools

```sh
siphon inspect "$MODEL_DIR"
siphon plan "$MODEL_DIR" --free 32
siphon plan "$MODEL_DIR" --interactive
```

`inspect` lists model and weight metadata. `plan` estimates CUDA memory on Linux or macOS. Interactive planning prints the `siphon serve` command for the selected settings.

## Shell completions

Run `siphon completions --help` for the supported shells. The commands below load completions into your current shell when `siphon` is on your PATH.

- In Bash, run `source <(siphon completions bash)`.
- In Zsh, run `autoload -Uz compinit && compinit` if your shell has not initialized completions, then `source <(siphon completions zsh)`.
- In Fish, run `siphon completions fish | source`.

## CUDA benchmarks and tests

Set `CAPTURE` to an MLX-LM reference capture for commands that take a prompt.

```sh
pixi run cargo test --workspace
pixi run cargo build --release -p siphon-cuda --bin bench
target/release/bench "$MODEL_DIR" --contexts 0,32768 --weights fp8 --kv fp8
target/release/bench "$MODEL_DIR" --prefill 8192,32768 --weights fp8 --kv fp8
target/release/bench "$MODEL_DIR" --prompt "$CAPTURE" --spec auto --weights fp8 --kv fp8
target/release/bench "$MODEL_DIR" --batch 1,2,4,8 --prompt "$CAPTURE" --contexts 0 --weights fp8 --kv fp8
```

## Project layout

| Path | Contents |
| --- | --- |
| [`crates/siphon`](../crates/siphon) (`siphon-engine`) | Config parsing, the memory-mapped safetensors checkpoint and GPU memory planning |
| [`crates/siphon-cli`](../crates/siphon-cli) | The `siphon` command, HTTP server, terminal memory planner, and shell completions |
| [`crates/siphon-cuda`](../crates/siphon-cuda) | Kernels (`kernels/*.cu`), their Rust launchers (`src/ops`), the model, the drafts and speculation, and the `bench`, `eval`, `parity` and `dspark` binaries |
| [`crates/siphon-mlx`](../crates/siphon-mlx) | Baseline token generation through an MLX-LM worker on Apple Silicon |
| [`reference`](../reference) | MLX-LM captures and kernel fixtures, run on a Mac with uv |
| [`scripts`](../scripts) | Remote sync, the PDL read-order audit and profiling scripts |

## Acknowledgements

- [MLX-LM](https://github.com/ml-explore/mlx-lm)'s Inkling model is the numerical reference.
- [vLLM](https://github.com/vllm-project/vllm)'s Inkling kernels define the activation quantization that the quantized prefill path follows.
- [Ingine](https://github.com/satellitedown/inkling-2-mini-abliterated-ingine), another engine for this model, informed the speculation design: confidence-chosen verify widths and a shortlist draft head.
- [CUTLASS](https://github.com/NVIDIA/cutlass)'s sm_120 grouped GEMM informed the tile shape of the quantized expert GEMM.
