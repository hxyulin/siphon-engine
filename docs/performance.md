# CUDA performance

## Supported GPUs

The same kernels serve the RTX 5090 and the RTX PRO 6000 Blackwell (`sm_120a`). The RTX 4090 runs an `sm_89` build with native Ada FP8 Tensor Core prefill attention, dense GEMMs, and MoE grouped GEMMs. Decode reads the weights once per token, so it runs about as fast on the two Blackwell cards, which have the same memory bandwidth. The PRO 6000's 96 GB hold the model's full context and up to 32 speculative streams, and it prefills fastest. Headline numbers across the three cards (tok/s, fast FP8/MXFP8/NVFP4 profile with DFlash 2 where noted; default BF16 `W_o` numbers are in the sections below):

| GPU | Memory & Arch | Speculative code<br>greedy · T=0.8 | Plain decode<br>ctx 0 · 32K | Prefill<br>4K · 16K | Batched steps<br>16 · 32 · 64 streams | Longest context<br>1 stream (FP8 · NVFP4) | Recipes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| **RTX 5090** | 32 GB, 1.79 TB/s, `sm_120a` | 885 · 715 | 384 · 310 | 34,316 · 32,669 | 1,452 · 2,145 · 3,237 | 729,088 · 1,048,576 | [RTX 5090](recipes.md#rtx-5090) |
| **RTX PRO 6000 Blackwell** | 96 GB, 1.79 TB/s, `sm_120a` | 883 · 706 | 352 · 326 | 36,053 · 33,975 | 1,449 · 2,176 · 3,462 | 1,048,576 · 1,048,576 | [RTX PRO 6000](recipes.md#rtx-pro-6000-blackwell) |
| **RTX 4090** | 24 GB, 1.01 TB/s, `sm_89` | 479 · 375 | 226 · 194 | 14,457 · 14,324 | 814 · 1,208 · — | 247,808 · **606,208** | [RTX 4090](recipes.md#rtx-4090) |

See [docs/recipes.md](recipes.md) for copy-pasteable `siphon serve` commands per card (single-user coding, multi-user serving, relaxed speculation, and max context). [RTX PRO 6000 Blackwell](#rtx-pro-6000-blackwell) and [RTX 4090](#rtx-4090) below have full breakdowns.

## Checkpoints

Siphon also runs [`hxyulin/Inkling-2-Mini-Fast-NVFP4`](https://huggingface.co/hxyulin/Inkling-2-Mini-Fast-NVFP4), the official checkpoint with [Lucebox's tuned attention output projections](https://huggingface.co/Lucebox/Inkling-2-Mini-Fast-GGUF), with no options changed. On 100 GPQA questions at temperature 1 it answered 81 correctly against the official checkpoint's 77, with 3,055 output tokens per reply on average against 11,871. These numbers come from a build with Q8_0 copies of the tuned matrices, about 0.8% from the BF16 ones in that repo.

## Measurements

The numbers in this section come from one RTX 5090 (32 GB, 600 W) running main at d60a6c9, unless a section names another commit or GPU, with no other process on the GPU. Speculation and batching use FP8 dense weights and an FP8 KV cache, the server's default. These runs read an FP8 copy of the attention output projection, now `--output-projection fp8`. The default reads its BF16 weights, which takes 0.16 ms more per plain step at context 0 and 0.13 ms at 32K, and 1 to 2% more per verify pass ([engine.md](engine.md#the-output-projection)).

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

`siphon serve --streams N` runs up to 32 requests at once, each in its own KV cache. A request alone speculates. When others arrive, all of them decode in batched steps of one row each, and a request left alone speculates again. Tok/s in all, with tok/s per stream in brackets:

| Context of each stream | 1 stream | 2 streams | 4 streams | 8 streams |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 318 | 476 (238) | 689 (172) | **999** (125) |
| 10,674 | 286 | 412 (206) | 579 (145) | **804** (100) |
| 32,768 | 264 | 376 (188) | 519 (130) | **696** (87) |

A batch of one takes 3.14 ms per step against 2.71 ms for plain decode, because the batch keeps the verify pass's numerics. A request that speculates alone runs at 508 to 875 tok/s greedy and 345 to 711 tok/s at temperature 1, so batching serves more in total from about three streams on. The batched steps here decode greedily. Over HTTP with `--streams 8`, temperature 1 served 734 tok/s in all against 780 greedy (see [Sampling](#sampling)).

A 32 GB card fits 8 streams of 71,680 positions. An RTX PRO 6000 Blackwell Max-Q (95 GiB, 1,645 GB/s, 275 W) fits 16 streams of 284,672 positions or 32 of 132,096, and serves more with each step up to 32. The same batched-step benchmark there:

| Context of each stream | 1 stream | 8 streams | 16 streams | 32 streams |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 262 | 782 (98) | 1,082 (68) | **1,512** (47) |
| 10,674 | 239 | 661 (83) | 871 (54) | **1,138** (36) |
| 32,768 | 225 | 580 (73) | 740 (46) | **918** (29) |

Every request still gets the tokens it gets alone: each row of a batched step of up to 32 rows equals a one-row verify bit for bit ([batching.md](batching.md#more-than-8-rows)).

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

With `--streams 8`, eight agent-like conversations at once, each growing from 15K to 53K tokens over nine turns, got their first token after 1.37 s on average instead of 6.25 s, with the same replies (see [engine.md](engine.md#prefix-cache)).

### Context length

The server plans its GPU memory from the config and the checkpoint headers before it loads anything. Without `--context` it takes the longest context that fits the free memory less a 0.25 GiB margin (`--margin`), up to `--streams` times the model's 1,048,576 per-request limit (each individual request stays capped at 1,048,576). With 30.45 GiB free after the CUDA context on a 32 GB card, the longest single-stream contexts are:

| Dense weights, KV cache | DSpark draft | DFlash 2 |
| --- | ---: | ---: |
| FP8, FP8 | 657,408 | 729,088 |
| BF16, BF16 | 349,184 | 365,568 |

On a 96 GB RTX PRO 6000 Blackwell (`94.2 GiB` budget) with `--streams 32`, the shared KV pool scales to **4,665,344** positions with FP8 KV (`4,575,232` with prefix cache) and **8,292,352** positions with `--kv nvfp4` (`8,132,608` with prefix cache) while capping each individual request at `1,048,576`.

`--prefill-chunk 1024` halves the prefill buffers, which raises the FP8 contexts on a 32 GB card to 679,936 and 754,688 for 6 to 7% of prefill speed.

`siphon plan <model-dir>` shows the plan in a terminal before you start the server, and runs on macOS too.

### RTX PRO 6000 Blackwell

Measured on an RTX PRO 6000 Blackwell Workstation Edition (96 GB, 600 W, driver 595.84, empirical `1,645 GB/s` streaming read bandwidth), with the Fast NVFP4 checkpoint (`--weights fp8 --kv fp8`, 3 passes):

- **Plain decode** (`B=1`, `bench`, 100 steps): default BF16 `W_o` / FP8 head runs `2.851 ms` (`350.7 tok/s`) at ctx 0, `3.004 ms` (`332.9 tok/s`) at 4K, and `3.328 ms` (`300.4 tok/s`) at 32K (`93ad984`). With the fast profile (`--output-projection mxfp8 --lm-head nvfp4`), `B=1` decode reaches `352.1 tok/s` at 10.7K and `326.2 tok/s` at 32K (and on RTX 5090, `384.1 / 368.2 / 347.5 tok/s` at 0 / 4K / 16K).
- **Prefill** (`--prefill-dense fp8 --expert-activations quantized --prefill-attention fp8`, adaptive `2560`-row chunking at >=16K): `36,053 tok/s` (`113.6 ms`) at 4K, `33,975 tok/s` (`482.2 ms`) at 16K, and `31,929 tok/s` (`1,026 ms`) at 32K (RTX 5090: `34,316 tok/s` at 4K, `32,669 tok/s` at 16K).
- **Single-stream DFlash 2 speculation** (`anyhow` 10.7K / 32K, 256 tokens, fast profile + `--draft-weights nvfp4`):
  - Greedy `--spec auto`: **`883.4 tok/s`** (`2.51x`) at 10.7K, **`797.5 tok/s`** (`2.44x`) at 32K (RTX 5090: **`884.6` / `816.6 tok/s`**).
  - Sampled lossless (`--spec-beta 1.0`, 6-seed medians): `T=0.6, p=0.95` **`829.4`** / `732.5 tok/s`; `T=0.8, p=0.95` **`705.6`** / `743.6 tok/s`; `T=1.0, p=0.95, m=0.05` linear `585.9` / `591.0` → entropy-gated 2-branch tree (`--spec-tree`) **`647.7`** (`+10.5%`) / **`639.8 tok/s`** (`+8.3%`). On RTX 5090 (4-seed medians): `T=0.8` lossless tree **`714.5` / `602.3 tok/s`** (`--spec-beta 0.80`: **`805.1` / `717.5 tok/s`**), `T=1.0, m=0.05` lossless tree **`616.4` / `532.6 tok/s`**.
- **Live `siphon serve` batched DFlash 2 speculation** (`--streams 32 --no-prefix-cache`, 1.77K prompt → 200 tokens, `T=0.8, p=0.95`):
  - Default FP8 + `--spec-beta 1.0`: `C=1` `375.2`, `C=8` `773.0`, `C=16` `900.6`, `C=32` `961.5 tok/s` (`p50 TTFT 2.13 s`).
  - Fast profile + `--spec-beta 1.0` (lossless): `C=1` `389.9`, `C=8` `884.4`, `C=16` `1,133.6`, `C=32` **`1,232.3 tok/s`** (`p50 TTFT 1.35 s`).
  - Fast profile + `--spec-beta 0.80`: `C=1` `437.0` (`p50 user 508.7 tok/s`), `C=8` `943.9`, `C=16` `1,168.7`, `C=32` **`1,314.4 tok/s`** (`p50 TTFT 1.31 s`).
- **Batched plain steps** (`bench`, `tok/s` aggregate with per-stream in brackets):

| Context of each stream | 1 stream | 16 streams | 32 streams | 64 streams |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 306 | 1,449 (91) | 2,176 (68) | **3,462** (54) |
| 4,096 | 285 | 1,230 (77) | 1,787 (56) | **2,622** (41) |

At `B=32, ctx=16,384` (fast profile), `--kv nvfp4` raises batched decode from `1,478` (`--kv fp8`) to **`1,632 tok/s`** (`+10.4%`), and `--kv nvfp4-all` reaches **`1,663 tok/s`** (`+12.5%`). With 94.2 GiB available, the shared KV pool holds **`4.67M`** FP8 or **`8.29M`** NVFP4 positions across `--streams 32` (tested live serving 8 concurrent 162K-token requests / `1.30M` simultaneous prompt tokens without swapping). With `--host-cache`, a 330K-token conversation that left the GPU came back in 0.81 s instead of 38.9 s of prefill ([server.md](server.md#prefix-cache)). [batching.md](batching.md#against-vllm-and-sglang-on-an-rtx-pro-6000-workstation-edition) compares batched decode on this card with vLLM and SGLang.

### RTX 4090

Measured on an RTX 4090 (24 GB, 450 W, driver 620.70.10) with the `sm_89` build ([Build from source](install.md#linux-with-cuda)) and the Fast NVFP4 checkpoint (`--weights fp8 --output-projection mxfp8 --lm-head nvfp4`, [RTX 4090 recipes](recipes.md#rtx-4090)):

- **Plain decode** (`B=1`, `bench`, 100 steps):

| Context | FP8 KV | NVFP4 KV |
| ---: | ---: | ---: |
| 0 | 225.7 tok/s | 224.8 tok/s |
| 32,768 | 193.6 tok/s | 200.9 tok/s |
| 131,072 | 144.8 tok/s *(84.7% DRAM floor)* | **159.2 tok/s** |
| 524,288 | — | **85.8 tok/s** |

- **Prefill** (`--prefill-dense fp8 --expert-activations fp8 --prefill-attention fp8`, native Ada FP8 Tensor Cores, `--chunk 2560`): **`14,457 tok/s`** at 4K, **`14,356 tok/s`** at 8K, **`14,324 tok/s`** at 16K, and **`13,832 tok/s`** at 32K (`+31%` to `+39%` over BF16 prefill, within `<= 0.41%` PPL on `eval`).
- **Single-stream speculation** (`bench --spec auto`, 256 tokens, DFlash 2 with `--draft-weights nvfp4`, 4-seed medians for sampled runs with `--spec-tree`):
  - **Code (`anyhow`, 10.7K / 32K):** greedy **`479.3`** (`2.30x`) / **`458.1 tok/s`** (`2.37x`, bundled DSpark `460.2` / `413.4`; **`486.2 tok/s`** with `--kv nvfp4`); sampled lossless (`--spec-beta 1.0`) `T=0.8, p=0.95` **`375.3 tok/s`** (`449.0 tok/s` at `--spec-beta 0.80`), `T=1.0, p=0.95, m=0.05` **`346.7 tok/s`**.
  - **Prose (`regex-readme`, 2.3K / 32K):** greedy **`394.8`** (`1.79x`) / **`385.4 tok/s`** (`1.99x`, bundled DSpark `339.4` / `362.3`); sampled lossless (`--spec-beta 1.0 --fallback`) `T=0.8, p=0.95` **`254.0 tok/s`** (`280.5 tok/s` at `--spec-beta 0.80`), `T=1.0, p=0.95, m=0.05` **`249.1 tok/s`**.
- **Batched plain decode** (`bench`, FP8 KV, `ctx 0 / 4,096`): `B=8` `563 / 502`, `B=16` **`814 / 709`**, `B=24` **`1,009 / 869`**, `B=32` **`1,208 / 1,016 tok/s`**.
- **Context budget (23.17 GiB free after CUDA context):** with `--weights fp8 --prefill-dense fp8` (which streams layer quantization and frees ~1.0–1.3 GiB of unused BF16 weights) and DFlash 2, one stream at full `--prefill-chunk 2048 --margin 0.25` reaches **`563,200`** positions with `--kv nvfp4 --output-projection mxfp8`, **`606,208`** with `--lm-head nvfp4 --draft-weights nvfp4`, and **`614,400`** with `--kv nvfp4-all` (verified live in `siphon serve --context 524288` with `peak <= planned` and full `--host-cache` compatibility; up to **`632,832`** with `--prefill-chunk 1024 --margin 0.1`). See [docs/recipes.md](recipes.md#rtx-4090) and [server.md](server.md#serving-on-an-rtx-4090) for full configurations and [engine.md](engine.md#rtx-4090-sm_89) for architectural details.
