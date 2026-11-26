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

| 875 · 507 tok/s | 999 tok/s | 81–88% | 633,856 | 0.10 s |
| :---: | :---: | :---: | :---: | :---: |
| one stream, speculative, on Rust<br>source: greedy · temperature 1 | 8 streams together | of the DRAM floor<br>in plain decode | tokens of context<br>on a 32 GB card | to first token of a<br>103K-token follow-up |

</div>

## Why Siphon

- **Close to the hardware limit.** One plain decode step takes 2.73 ms with FP8 weights and cache, 81% of the time it takes to read the weights once from memory. Decode runs as one CUDA graph per token, with programmatic dependent launch and an L2 prefetch stream for the next layer's weights.
- **The official checkpoint, unchanged.** Siphon runs the checkpoint as published: NVFP4 and MXFP8 routed experts, BF16 dense weights and the bundled DSpark draft. It builds optional FP8 copies of the dense weights at load time.
- **Speculation without changing the output.** The verify pass gives each row the same result at every width, so greedy speculation returns exactly the tokens that verifying one row at a time would. Sampled speculation keeps the model's filtered distribution exactly. Batched decode keeps both properties: a greedy or seeded request gets the same tokens alone or next to seven others.
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

Siphon also runs [`hxyulin/Inkling-2-Mini-Fast-NVFP4`](https://huggingface.co/hxyulin/Inkling-2-Mini-Fast-NVFP4), the official checkpoint with [Lucebox's tuned attention output projections](https://huggingface.co/Lucebox/Inkling-2-Mini-Fast-GGUF), with no options changed. On 100 GPQA questions at temperature 1 it answered 81 correctly against the official checkpoint's 77, with 3,055 output tokens per reply on average against 11,871. These numbers come from a build with Q8_0 copies of the tuned matrices, about 0.8% from the BF16 ones in that repo.

## Performance

All numbers come from one RTX 5090 (32 GB, 600 W) running main at d60a6c9, unless a section names another commit, with no other process on the GPU. Speculation and batching use FP8 dense weights and an FP8 KV cache, the server's default.

### Decode

One token per step, 400 steps. The DRAM floor is the time to read the weights and the KV cache once at 1.70 TB/s.

| Dense weights, KV cache | Context 0 | 32K | 128K |
| --- | ---: | ---: | ---: |
| FP8, FP8 | 2.73 ms · 367 tok/s · 81% | 3.23 ms · 310 tok/s · 81% | 4.23 ms · 236 tok/s · 85% |
| BF16, BF16 | 3.77 ms · 265 tok/s · 86% | 4.75 ms · 211 tok/s · 84% | 6.67 ms · 150 tok/s · 88% |

### Speculative decoding

512 tokens after each prompt, tok/s and speedup over plain decode. Greedy is one run at a3fc890. Temperature 1, the checkpoint's default, is the median of 6 seeds at 1ab0b44, with plain decode from the same runs. The draft proposes up to 7 tokens per round, and each round picks its verify width from the drafts' confidences. The DFlash 2 columns run Inco AI's drafter (`--drafter`) instead of the bundled DSpark draft.

| Prompt | Context | Plain decode | DSpark, greedy | DSpark, temp. 1 | DFlash 2, greedy | DFlash 2, temp. 1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Rust source (`anyhow`) | 10.7K | 336 | **875** · 2.61x | **507** · 1.52x | 844 · 2.52x | 457 · 1.37x |
| Rust source (`anyhow`) | 32K | 310 | **812** · 2.62x | **497** · 1.61x | 802 · 2.59x | 439 · 1.42x |
| Rust source (`half`) | 2.3K | 352 | **787** · 2.24x | **539** · 1.54x | 662 · 1.88x | 490 · 1.40x |
| Rust source (`half`) | 32K | 310 | **684** · 2.21x | **447** · 1.45x | 635 · 2.05x | 397 · 1.29x |
| A README (`regex`) | 2.3K | 352 | 594 · 1.69x | **395** · 1.13x | **629** · 1.79x | 382 · 1.09x |
| A README (`regex`) | 32K | 310 | 567 · 1.83x | **345** · 1.12x | **621** · 2.01x | **346** · 1.12x |
| A chat question | 30 | 364 | 546 · 1.50x | 494 · 1.36x | **601** · 1.65x | **559** · 1.54x |
| A coding request | 40 | 364 | 508 · 1.40x | 615 · 1.70x | **583** · 1.60x | **711** · 1.96x |

The speedup depends on how predictable the text is. Greedy decoding continues a source file almost as written, so the drafts keep 4.9 to 6.7 tokens per round there. Sampled text is less predictable: at temperature 1 the drafts keep 2.8 to 3.4 tokens per round on the Rust sources and 2.0 to 2.3 on the README, where speculation beats plain decode by 9 to 13%. When sampling, the draft proposes the token the verify's own draw would pick for that seed and position, instead of its argmax. Against the argmax draft at a3fc890, that raised every temperature 1 case by 0.1 to 12%, with the same tokens for each seed. Seeds differ widely: the six DSpark runs on `anyhow` at 10.7K ran from 472 to 746 tok/s. The coding request runs faster sampled than greedy: the DSpark draft keeps 4.0 tokens per round on its sampled answers and 3.0 on its greedy one. DFlash 2 drafts prose, chat and the coding request better, and the bundled draft drafts the Rust sources better.

### Sampling

Sampled text is harder to predict than greedy text, so the draft keeps fewer tokens per round. The 12 prompts of the registry runs, up to 3,000 tokens each, one request at a time and then 8 at a time with `--streams 8`:

| Sampling | Decode tok/s, alone | Tokens per round | Tok/s, 8 streams |
| --- | ---: | ---: | ---: |
| Greedy (`temperature: 0`) | 536 | 3.47 | 780 |
| Temperature 1, the checkpoint's default | 416 | 2.32 | 737 |
| Temperature 0.8, top-p 0.95 | 445 | 2.55 | 768 |

Requests with seeds gave the same 12 outputs on a second run and with `--streams 8`. Sampling adds 0.08 to 0.19 ms to an 8-row verify of 6.57 ms.

Neither mode is clearly more accurate. On 23 quickbench items at a3fc890, 4 attempts each (GPQA, AIME, HumanEval+, retrieval at 30K and 124K tokens, tool calls), temperature 1 passed 68.5% and greedy 78.3%. The paired difference has a 95% interval of -1.1 to +21.7 points. Temperature 1 hit the token cap 11 times against 5 and averaged 5,682 tokens per reply against 4,741. Four of its cut-offs came at 124K tokens of context, where the thinking started copying the haystack. MLX-LM does the same at temperature 1: 6 of 17 replies on those prompts ran to the cap that way, so this is the model's behaviour, not the engine's. Greedy repeats one reply on every attempt unless the options are shuffled, so its code, retrieval and AIME rates rest on one reply per item. The items were picked for sampled pass rates between 20% and 90% and for short replies, which leaves out the long problems where greedy loops. An earlier gpqa_diamond run on an older commit, 298 replies each with 32K tokens of output, went the other way: temperature 1 passed 87.6% against 77.2%, and repetition loops cut off 18.9% of greedy replies against 5.3% sampled.

### Concurrent requests

`siphon-serve --streams N` runs up to 32 requests at once, each in its own KV cache. A request alone speculates. When others arrive, all of them decode in batched steps of one row each, and a request left alone speculates again. Tok/s in all, with tok/s per stream in brackets:

| Context of each stream | 1 stream | 2 streams | 4 streams | 8 streams |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 318 | 476 (238) | 689 (172) | **999** (125) |
| 10,674 | 286 | 412 (206) | 579 (145) | **804** (100) |
| 32,768 | 264 | 376 (188) | 519 (130) | **696** (87) |

A batch of one takes 3.14 ms per step against 2.71 ms for plain decode, because the batch keeps the verify pass's numerics. A request that speculates alone runs at 508 to 875 tok/s greedy and 345 to 711 tok/s at temperature 1, so batching serves more in total from about three streams on. The batched steps here decode greedily. Over HTTP with `--streams 8`, temperature 1 served 734 tok/s in all against 780 greedy (see [Sampling](#sampling)).

A 32 GB card fits 8 streams of 69,632 positions. An RTX PRO 6000 Blackwell Max-Q (95 GiB, 1,645 GB/s, 275 W) fits 16 streams of 284,672 positions or 32 of 132,096, and serves more with each step up to 32. The same batched-step benchmark there:

| Context of each stream | 1 stream | 8 streams | 16 streams | 32 streams |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 262 | 782 (98) | 1,082 (68) | **1,512** (47) |
| 10,674 | 239 | 661 (83) | 871 (54) | **1,138** (36) |
| 32,768 | 225 | 580 (73) | 740 (46) | **918** (29) |

Every request still gets the tokens it gets alone: each row of a batched step of up to 32 rows equals a one-row verify bit for bit ([batching.md](docs/batching.md#more-than-8-rows)).

### Prefill and the prefix cache

Prefill runs in chunks of 2,048 rows on tensor-core GEMMs, an expert-grouped MoE GEMM and flash attention. Past 16,384 keys, global attention over the FP8 cache runs on FP8 tensor cores.

| Prompt | Time to first token | Prefill tok/s |
| ---: | ---: | ---: |
| 8,192 tokens | 0.51 s | 16,150 |
| 32,768 tokens | 2.11 s | 15,510 |
| 131,072 tokens | 9.91 s | 13,220 |

Chat clients resend the whole conversation each turn. The server keeps the state of recent prompts, one per stream, at their last multiple of 2,048 tokens. A follow-up turn prefills only its new tokens and gets the same tokens and logits as without the cache. Time to the first token of a follow-up turn over HTTP:

| Conversation | Without the cache | With it | Speedup |
| ---: | ---: | ---: | ---: |
| 8,174 tokens | 0.53 s | 0.14 s | 3.8x |
| 102,848 tokens | 7.70 s | 0.10 s | 79x |

With `--streams 8`, eight agent-like conversations at once, each growing from 15K to 53K tokens over nine turns, got their first token after 1.37 s on average instead of 6.25 s, with the same replies (see [engine.md](docs/engine.md#prefix-cache)).

### Context length

The server plans its GPU memory from the config and the checkpoint headers before it loads anything. Without `--context` it takes the longest context that fits the free memory less a 0.25 GiB margin (`--margin`), up to the model's 1,048,576 positions. With 30.45 GiB free after the CUDA context, the longest contexts are:

| Dense weights, KV cache | DSpark draft | DFlash 2 |
| --- | ---: | ---: |
| FP8, FP8 | 648,192 | 717,824 |
| BF16, BF16 | 349,184 | 365,568 |

`--prefill-chunk 1024` halves the prefill buffers, which raises the FP8 contexts to 670,720 and 744,448 for 6 to 7% of prefill speed.

`siphon-plan <model-dir>` shows the plan in a terminal before you start the server, and runs on macOS too.

## Features

- **Server.** OpenAI chat completions and completions, with streaming, reasoning (`reasoning_content`), tool calls, `tool_choice`, stop strings, `reasoning_effort`, sampling and several choices per request (`n`) from one prefill. vLLM and TabbyAPI tokenizer routes, and vLLM's `return_token_ids`, `ignore_eos` and `continue_final_message`.
- **Scheduling.** Continuous batching of up to 32 requests. Speculation for a request that runs alone, and a prefix cache with an entry per stream, for batched and lone requests.
- **Sampling.** Temperature, top-k, top-p and min-p on the GPU, applied in vLLM's order, with the checkpoint's `generation_config.json` for the fields a request leaves out (temperature 1). Each draw depends only on the logits, the request's seed and the token's position. `temperature: 0` or `--greedy` decodes greedily.
- **Speculation.** The bundled DSpark draft by default, or Inco AI's DFlash 2 drafter or a 2DSD DFlash export with `--drafter DIR`. `--fallback` runs plain steps where the draft cannot predict the text. It speeds up free prose, but the output can then differ between runs.
- **Formats.** FP8 dense weights (`--weights fp8`) and an FP8 global KV cache (`--kv fp8`) by default, BF16 for both on request. Prefill can opt in to quantized expert activations and FP8 dense products, which are faster and fail the quality gate on some texts.
- **Memory planning.** The server checks the planned GPU memory against what it uses after warmup and after each job.

Not yet: the presence, frequency and repetition penalties (accepted and ignored), speculation for more than one request at a time, prefix cache entries shared by prompts that agree only in their first tokens, and image input.

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
