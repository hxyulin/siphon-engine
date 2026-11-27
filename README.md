# Siphon

Siphon serves [Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) through an OpenAI-compatible API on Linux with CUDA and on Apple Silicon with MLX.

| Platform | Backend |
| --- | --- |
| Linux with RTX 5090, RTX PRO 6000 Blackwell, or RTX 4090 | CUDA with speculative decoding, batching, prefix caching, and image inputs |
| Apple Silicon macOS | MLX-LM baseline for one text request at a time |

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

## License

Licensed under [Apache-2.0](LICENSE-APACHE) or [MIT](LICENSE-MIT), at your option. Contributions are dual licensed under the same terms unless you state otherwise.
