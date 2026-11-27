# Hardware Recipes

Copy-pasteable `siphon-serve` configurations for each supported GPU. All speeds below are measured on `inkling-2-mini` / [`hxyulin/Inkling-2-Mini-Fast-NVFP4`](https://huggingface.co/hxyulin/Inkling-2-Mini-Fast-NVFP4) with Inco AI's DFlash 2 drafter ([`incoai/Inkling-2-Mini-DFlash2`](https://huggingface.co/incoai/Inkling-2-Mini-DFlash2)).

Before launching, you can inspect the exact VRAM breakdown and context ceiling for any machine with:

```sh
target/release/siphon-plan <model-dir> --drafter <dflash2-dir> --output-projection mxfp8
```

> **Checkpoints & Drafters:**
> - `<model-dir>` is either the official [`thinkingmachines/Inkling-2-Mini`](https://huggingface.co/thinkingmachines/Inkling-2-Mini) checkpoint or [`hxyulin/Inkling-2-Mini-Fast-NVFP4`](https://huggingface.co/hxyulin/Inkling-2-Mini-Fast-NVFP4) (same weight sizes and formats, ~3.9× shorter reasoning traces).
> - `<dflash2-dir>` is [`incoai/Inkling-2-Mini-DFlash2`](https://huggingface.co/incoai/Inkling-2-Mini-DFlash2). Omit `--drafter` to use the checkpoint's bundled DSpark draft instead.
> - **Lossless vs. relaxed speculation:** `--spec-beta 1.0 --spec-tree` keeps the target model's exact filtered sampling distribution. Changing `--spec-beta 1.0` to `--spec-beta 0.80` trades a slight distribution relaxation for +10–20% higher sampled decode speed.

---

## RTX 5090

*32 GB GDDR7 · 1.79 TB/s · `sm_120a` (default build)*

### 1. Fastest single-user / coding agent (lossless)

Best all-around recipe for Claude Code, Codex, or interactive coding on a 32 GB card. Uses MXFP8 output projections, NVFP4 LM head and draft weights, FP8/quantized prefill, lossless tree speculation, 4 streams (so background agent calls don't evict the main conversation), and a 12 GiB host prefix cache.

```sh
target/release/siphon-serve <model-dir> \
  --drafter <dflash2-dir> --draft-weights nvfp4 \
  --weights fp8 --output-projection mxfp8 --lm-head nvfp4 \
  --kv fp8 --prefill-dense fp8 --expert-activations quantized --prefill-attention fp8 \
  --spec-beta 1.0 --spec-tree \
  --streams 4 --host-cache 12 --cache-dir /var/cache/siphon \
  --host 0.0.0.0 --port 8000
```

- **Measured speeds:** **885 tok/s** greedy code (`anyhow` 10.7K) · **715 tok/s** sampled `T=0.8` lossless · **616 tok/s** sampled `T=1.0, min_p=0.05` lossless · **384 tok/s** plain decode (ctx 0) · **34,316 tok/s** prefill (4K) / **32,669 tok/s** (16K).
- **Context:** ~655K–729K shared KV pool positions (FP8 KV).

### 2. Maximum sampled speed (relaxed speculation + prose fallback)

Same fast weights/prefill profile, switching to `--spec-beta 0.80` for higher sampled acceptance and `--fallback` so unpredictable prose drops to plain decode automatically.

```sh
target/release/siphon-serve <model-dir> \
  --drafter <dflash2-dir> --draft-weights nvfp4 \
  --weights fp8 --output-projection mxfp8 --lm-head nvfp4 \
  --kv fp8 --prefill-dense fp8 --expert-activations quantized --prefill-attention fp8 \
  --spec-beta 0.80 --spec-tree --fallback \
  --streams 4 --host-cache 12 \
  --host 0.0.0.0 --port 8000
```

- **Measured speeds:** **805 tok/s** sampled code at `T=0.8` (10.7K) / **718 tok/s** at 32K.

### 3. Full 1M context on 32 GB

Switches global KV caches to NVFP4 (9 KiB/pos instead of 16 KiB/pos) with native Blackwell NVFP4 prefill attention, hitting the model's full `1,048,576` token context limit on a single 32 GB card.

```sh
target/release/siphon-serve <model-dir> \
  --drafter <dflash2-dir> --draft-weights nvfp4 \
  --weights fp8 --output-projection mxfp8 --lm-head nvfp4 \
  --kv nvfp4 --prefill-dense fp8 --expert-activations quantized --prefill-attention fp4 \
  --spec-beta 1.0 --spec-tree \
  --streams 2 --host-cache 12 \
  --host 0.0.0.0 --port 8000
```

- **Context:** Full **1,048,576** positions per request.

### 4. Exact-reference baseline (default flags)

Zero extra approximations beyond the default FP8 dense weights and FP8 KV cache, with BF16 output projections and the bundled DSpark draft:

```sh
target/release/siphon-serve <model-dir> --host 0.0.0.0 --port 8000
```

- **Measured speeds:** **356 / 303 tok/s** plain decode (ctx 0 / 32K) · **875 tok/s** greedy code (with `--output-projection fp8`) · **16,203 tok/s** BF16 prefill (8K) · **661,504** context.

---

## RTX PRO 6000 Blackwell

*96 GB GDDR7 · 1.79 TB/s · `sm_120a` (default build)*

### 1. Multi-user / agent swarm serving (32 streams, lossless batched DFlash 2)

Full 32-stream continuous batching with batched DFlash 2 speculation, confidence-grouped micro-batches, and a **4.67M-position** shared FP8 KV pool (capped at `1,048,576` per individual request).

```sh
target/release/siphon-serve <model-dir> \
  --drafter <dflash2-dir> --draft-weights nvfp4 \
  --weights fp8 --output-projection mxfp8 --lm-head nvfp4 \
  --kv fp8 --prefill-dense fp8 --expert-activations quantized --prefill-attention fp8 \
  --spec-beta 1.0 --spec-tree --group-by-confidence \
  --streams 32 --host-cache 32 --cache-dir /var/cache/siphon \
  --host 0.0.0.0 --port 8000
```

- **Measured live `siphon-serve` throughput (`T=0.8, p=0.95`):** `C=1` **390** · `C=8` **884** · `C=16` **1,134** · `C=32` **1,232 tok/s** (p50 TTFT 1.35 s).
- Swap `--spec-beta 1.0` → `--spec-beta 0.80` for `C=1` **437** (p50 user 509 tok/s) · `C=32` **1,314 tok/s**.
- **Prefill:** **36,053 tok/s** (4K) · **33,975 tok/s** (16K) · **31,929 tok/s** (32K).

### 2. Massive shared KV pool (`8.29M` positions, long-context multi-agent)

Stores global KV caches in NVFP4 with native Blackwell FP4 prefill attention. Tested live serving 8 concurrent 162K-token requests (`1.30M` simultaneous prompt tokens) without swapping.

```sh
target/release/siphon-serve <model-dir> \
  --drafter <dflash2-dir> --draft-weights nvfp4 \
  --weights fp8 --output-projection mxfp8 --lm-head nvfp4 \
  --kv nvfp4 --prefill-dense fp8 --expert-activations quantized --prefill-attention fp4 \
  --spec-beta 1.0 --spec-tree --group-by-confidence \
  --streams 32 --host-cache 32 --cache-dir /var/cache/siphon \
  --host 0.0.0.0 --port 8000
```

- **Context:** **8,292,352** pooled positions (`8,132,608` with prefix cache), up to **1,048,576** per request.
- **Measured batched decode (`B=32, 16K`):** **1,632 tok/s** (`+10.4%` over FP8 KV; **1,663 tok/s** with `--kv nvfp4-all`).

### 3. Fastest single-stream interactive / coding

Minimizes multi-stream overhead while keeping 4 slots for parallel background agent calls and 1M context per request:

```sh
target/release/siphon-serve <model-dir> \
  --drafter <dflash2-dir> --draft-weights nvfp4 \
  --weights fp8 --output-projection mxfp8 --lm-head nvfp4 \
  --kv fp8 --prefill-dense fp8 --expert-activations quantized --prefill-attention fp8 \
  --spec-beta 1.0 --spec-tree \
  --streams 4 --host-cache 16 --cache-dir /var/cache/siphon \
  --host 0.0.0.0 --port 8000
```

- **Measured speeds:** **883 tok/s** greedy code (`anyhow` 10.7K) · **829 tok/s** `T=0.6` · **706 tok/s** `T=0.8` · **648 tok/s** `T=1.0, min_p=0.05` (lossless tree) · **36,053 tok/s** prefill (4K).

---

## RTX 4090

*24 GB GDDR6X · 1.01 TB/s · `sm_89` (build with `SIPHON_CUDA_ARCH=sm_89 pixi run cargo build --release`)*

The `sm_89` build uses native Ada FP8 Tensor Cores for prefill attention, dense GEMMs (`--prefill-dense fp8`), and MoE grouped GEMMs (`--expert-activations fp8`, using overflow-proof 660 TFLOP/s `f16`-acc FP8 MMA for NVFP4 layers 0–19 and 330 TFLOP/s `f32`-acc FP8 MMA for MXFP8 layers 20–23). Note that `sm_89` uses `--expert-activations fp8` (not `quantized`).

### 1. Fastest single-user / coding agent (FP8 KV, ~248K context)

Maximum speed on code, prose, and prefill with full FP8 KV fidelity:

```sh
target/release/siphon-serve <model-dir> \
  --drafter <dflash2-dir> --draft-weights nvfp4 \
  --weights fp8 --output-projection mxfp8 --lm-head nvfp4 \
  --kv fp8 --prefill-dense fp8 --expert-activations fp8 --prefill-attention fp8 \
  --spec-beta 1.0 --spec-tree --fallback \
  --host 0.0.0.0 --port 8000
```

- **Measured speeds:**
  - **Code (`anyhow` 10.7K):** **479 tok/s** greedy · **375 tok/s** `T=0.8` lossless · **347 tok/s** `T=1.0, min_p=0.05` lossless *(449 tok/s at `T=0.8` with `--spec-beta 0.80`)*.
  - **Prose (`regex-readme` 2.3K):** **395 tok/s** greedy · **254 tok/s** `T=0.8` lossless · **249 tok/s** `T=1.0, min_p=0.05` lossless *(281 tok/s at `T=0.8` with `--spec-beta 0.80`)*.
  - **Plain decode:** **226 tok/s** (ctx 0) · **194 tok/s** (32K) · **145 tok/s** (128K).
  - **Prefill:** **14,457 tok/s** (4K) · **14,324 tok/s** (16K) · **13,832 tok/s** (32K).
- **Context:** **247,808** positions (1 stream). For coding agents that issue parallel background requests (like Claude Code), add `--streams 2 --kv mixed --host-cache 12 --cache-dir /var/cache/siphon` (~371K shared KV pool at near-FP8 quality).

### 2. Long context (440K–506K with DFlash 2)

Switches global KV caches to NVFP4 (9 KiB/pos), boosting both capacity and long-context decode bandwidth (`159 tok/s` plain at 128K vs. `145 tok/s` FP8):

```sh
# 440K context, keeps 2048-row prefill chunks and --host-cache support:
target/release/siphon-serve <model-dir> \
  --drafter <dflash2-dir> --draft-weights nvfp4 \
  --weights fp8 --output-projection mxfp8 --lm-head nvfp4 \
  --kv nvfp4 --prefill-dense fp8 --expert-activations fp8 \
  --spec-beta 1.0 --spec-tree --fallback \
  --host-cache 12 --cache-dir /var/cache/siphon \
  --host 0.0.0.0 --port 8000

# 506K context (without --host-cache, which requires 2048-row chunks):
target/release/siphon-serve <model-dir> \
  --drafter <dflash2-dir> --draft-weights nvfp4 \
  --weights fp8 --output-projection mxfp8 --lm-head nvfp4 \
  --kv nvfp4 --prefill-dense fp8 --expert-activations fp8 \
  --prefill-chunk 1024 --margin 0.1 \
  --spec-beta 1.0 --spec-tree --fallback \
  --host 0.0.0.0 --port 8000
```

- **Context:** **440,320** (chunk 2048, margin 0.25) · **505,856** (chunk 1024, margin 0.10).

### 3. Multi-user / small team (2–4 streams with host swapping)

Shares a ~370K mixed-precision KV pool across 4 streams and swaps idle conversations to pinned host RAM at 16–25 GB/s when oversubscribed:

```sh
target/release/siphon-serve <model-dir> \
  --drafter <dflash2-dir> --draft-weights nvfp4 \
  --weights fp8 --output-projection mxfp8 --lm-head nvfp4 \
  --kv mixed --prefill-dense fp8 --expert-activations fp8 \
  --spec-beta 1.0 --spec-tree \
  --streams 4 --host-cache 12 --cache-dir /var/cache/siphon \
  --host 0.0.0.0 --port 8000
```

- **Batched plain decode ceiling (RTX 4090, ctx 0 / 4K):** `B=8` 563 / 502 · `B=16` **814 / 709** · `B=32` **1,208 / 1,016 tok/s**.
