# Siphon

Siphon serves [Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) through an OpenAI-compatible API on Linux with CUDA and on Apple Silicon with MLX.

[![License MIT or Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue)](#license)
[![Rust 1.89+](https://img.shields.io/badge/rust-1.89%2B-orange?logo=rust)](https://www.rust-lang.org)
[![CUDA 13.5](https://img.shields.io/badge/CUDA-13.5-76B900?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![GPU RTX 5090 · RTX PRO 6000 · RTX 4090](https://img.shields.io/badge/GPU-RTX%205090%20%C2%B7%20RTX%20PRO%206000%20%C2%B7%20RTX%204090-76B900?logo=nvidia)](#cuda-performance)
[![Model Inkling 2 Mini](https://img.shields.io/badge/model-Inkling%202%20Mini-FFD21E?logo=huggingface)](https://huggingface.co/thinkingmachines/Inkling-2-Mini)
[![Image build](https://github.com/hxyulin/siphon-engine/actions/workflows/image.yml/badge.svg)](https://github.com/hxyulin/siphon-engine/actions/workflows/image.yml)

| Platform | Backend |
| --- | --- |
| Linux with RTX 5090, RTX PRO 6000 Blackwell, or RTX 4090 | CUDA with speculative decoding, batching, prefix caching, and image inputs |
| Apple Silicon macOS | MLX-LM baseline for one text request at a time |

## CUDA performance

| GPU | Speculative code<br>Greedy / sampled | Prefill<br>4K prompt | Batched decode<br>32 streams | Single-stream context<br>NVFP4 KV |
| --- | ---: | ---: | ---: | ---: |
| RTX 5090 | 885 / 715 tok/s | 34,316 tok/s | 2,145 tok/s | 1,048,576 |
| RTX PRO 6000 Blackwell | 883 / 706 tok/s | 36,053 tok/s | 2,176 tok/s | 1,048,576 |
| RTX 4090 | 479 / 375 tok/s | 14,457 tok/s | 1,208 tok/s | 606,208 |

Sampled decode uses temperature 0.8 and lossless speculation. Batched decode reports aggregate throughput. Configurations and longer-context results are in the [CUDA benchmarks](docs/performance.md#supported-gpus).

## Quick start

With Rust installed, run the commands below from the repository root. Download the [model checkpoint](https://huggingface.co/thinkingmachines/Inkling-2-Mini) and set `MODEL_DIR` to its local directory.

**Linux with CUDA.** Install a CUDA 13.5-compatible driver and [pixi](https://pixi.sh).

```sh
pixi run cargo build --release -p siphon-cli
target/release/siphon serve "$MODEL_DIR"
```

On an RTX 4090, prefix the build command with `SIPHON_CUDA_ARCH=sm_89`. See [installation](docs/install.md#linux-with-cuda) for Docker and source builds.

### Apple Silicon

Install [uv](https://docs.astral.sh/uv/) to set up the MLX runtime.

```sh
uv sync --project reference --locked
export SIPHON_MLX_PYTHON="$PWD/reference/.venv/bin/python"
cargo run --release -p siphon-cli -- serve "$MODEL_DIR"
```

### Send a request

```sh
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"inkling-2-mini","messages":[{"role":"user","content":"What is 2 + 2?"}],"reasoning_effort":"none","max_tokens":32}'
```

## Documentation

- [Installation and CLI](docs/install.md) covers builds, the memory planner, and shell completions.
- [Server reference](docs/server.md) and [client setup](docs/clients.md) cover API routes and configuration.
- [GPU recipes](docs/recipes.md) and [performance results](docs/performance.md) cover CUDA tuning and measured throughput.
- [Engine internals](docs/engine.md) covers architecture, [batching](docs/batching.md), and [vision](docs/vision.md).
- [Conversation archives](docs/archives.md) covers optional JSONL deduplication and compression.

## Development

Install [prek](https://prek.j178.dev/) and run `prek install` to enable the pre-commit and Conventional Commits hooks. If a hook fixes whitespace or line endings, stage the fixes before retrying the commit.

The hooks check formatting on changed Rust files and type-check all workspace targets and features for the host when Rust sources or Cargo inputs change. Run `prek run --all-files` to audit all files or `prek run --stage manual cargo-fmt` to audit workspace formatting alone. On Linux, use `pixi run git commit` or `pixi run prek run --all-files` to put the CUDA toolkit on the hook's path.

## License

Licensed under [Apache-2.0](LICENSE-APACHE) or [MIT](LICENSE-MIT), at your option. Contributions are dual licensed under the same terms unless you state otherwise.
