# Batching several sequences

Siphon serves one sequence at a time. This note describes how it runs up to 8 sequences together with plain decode, reusing the verify pass's multi-row kernels, and how the server schedules them. [engine.md](engine.md) describes the verify pass this builds on.

## Goal and exactness

A batched step runs one row for each of up to 8 sequences. Every sequence must get exactly the tokens it would get if it were served alone.

The verify pass already gives each row the same logits, bit for bit, as a verify of that row alone at its position, whatever the other rows hold. A batched step reuses those kernels and gives each row its own caches and position, so a batched row equals a one-row verify of its sequence. A sequence decoded in a batch therefore emits `greedy_by_verify`'s tokens: the tokens of greedy decoding through one-row verifies. Speculation without `--fallback` emits the same tokens. So a request gets the same output whether it runs alone with speculation, in a batch, or switches between the two mid-generation.

Sampling keeps this. A sampled token depends only on the row's logits, the request's seed and the token's position, and speculation keeps a draft only when it equals the token the target draws at that position ([engine.md](engine.md#sampling)). A request with a seed therefore gets the same tokens alone, in a batch, or switching between the two. Each batched row samples with its own request's settings (`Model::step_batch_sampled`).

Plain decode (`Model::step`) sums in other orders, so its output differs from both after a near-tie. Batching does not use it. With `--fallback` the speculator takes plain steps, and its output was already allowed to differ between runs.

The first token after a prompt comes from the prefill's logits in every case. Prefill must use the same chunk boundaries as when served alone (chunks of `DEFAULT_PREFILL_CHUNK` from position 0), since its kernels are not row-invariant.

## State per sequence

A sequence keeps, for each of the 48 layer visits:

| State | Size |
| --- | --- |
| Global KV cache (8 visits) | 16 KiB per position with `--kv fp8`, 32 KiB with BF16 |
| Sliding ring (40 visits) | 1,024 slots, 160 MiB in all, BF16 |
| Conv states: keys, values, post-attention, post-MLP | 3 rows each, 3.7 MB in all, F32 |

plus its position and the next token to feed. The draft's state (its caches, rings, tap tail and shortlist notes) belongs only to the sequence that speculates.

The model keeps sequence slots. Slot 0 is the model's existing state, which `Model::step`, `prefill` and the verify pass use as before. Slots 1 to N-1 each own the same set of caches in separate allocations. Kernels reach a slot's caches through device tables of pointers, so nothing assumes the slots are contiguous.

## Memory: fixed slots

The planner gives every slot the same capacity: `--streams N` slots of `capacity` positions each. It counts N sets of global caches, rings and conv states, one batch scratch (the verify pass's multi-row buffers for 8 rows and their taps), and the draft and verify buffers once. Without `--context`, the server takes the largest capacity at which N slots fit.

On a 32 GB card with FP8 weights and caches and the bundled draft, about 11 GiB remain after the fixed parts. Eight slots get 77,824 positions each, four 163,840.

A shared paged pool would let one long request and several short ones share the memory, but every attention kernel would have to read pages through a block table, including decode's and prefill's. That is a large change to kernels that three other efforts are tuning. Fixed slots need no change to the cache layout. A request whose prompt and limit exceed the slot capacity gets a 400 error, as it does today.

## A batched step

A batched step of B rows runs the verify pass's code path (`Ctx::prefill_enqueue` in invariant mode) with row r taken from slot `slot[r]` at position `pos[r]`. It keeps the verify pass's schedule, its FP8 L2 prefetch and its kernels. It skips the verify's backups, acceptance and rollback, and ends with every row's logits and token, greedy or drawn with the row's sampling.

Most kernels are row-independent and run unchanged: the embedding and norms, depth attention, the multi-row dense products, the message gates, the router, the experts, the LM head and the argmax. Four places read a sequence's caches or its position:

| Step | Single-sequence chunk | Batched |
| --- | --- | --- |
| `attention_post_rows` (key and value convs, log scale) | Row r's conv window is rows r - 3 .. r of the chunk, then the state; position `start + r` | Window is the row's own state and the row; position `pos[r]`. Two new arguments, null in the verify. |
| `attention_rows` (attention and KV append) | One cache shared by every row | A new entry point: grid z is the row, each z an instance of the one-row kernel on its own cache and position, with its own partials and counters. |
| Post-attention conv and residual (inside `depth_attention_rows`) | Window from the chunk's rows | A new small kernel computes it from the row's own state into the F32 residual; depth attention then reads that residual. |
| Visit end (`moe_conv_residual_step_rows`) | Shifts the chunk's rows into one set of states | A new kernel with the same arithmetic per row, shifting each row into its own slot's four states. |

Each batched row then computes exactly the one-row verify's arithmetic:

- `attention_post_rows` treats every batched row as row 0 of its own chunk.
- The attention entry point calls the same device function with one row, so each instance has the one-row verify's 34 blocks per KV head and its 3 combining blocks. A combining block waits on a counter for the other blocks of its instance. At most 2 blocks per instance wait, 128 in a batch of 8 rows, while the GPU holds at least 340 of these blocks at once, so the blocks they wait for always get an SM.
- The post-attention conv kernel uses the same operations as the fused one, and depth attention's F32 mode reads back the same values the fused mode keeps in registers.

The kernel changes stay small for merging with the attention work:

- `attention_rows` gains a new `extern "C"` entry point and no other change.
- `attention_post_rows_bf16` gains two pointer arguments.
- Everything else is new kernels. `depth_attention_rows` is not touched.

Pointer tables: for each visit, `table[visit][row]` holds the row's key cache, value cache and four conv states. The host rewrites it only when the batch's membership changes. Positions and tokens go to the device before each step, 64 bytes.

## CUDA graphs

Each batch size B from 1 to 8 gets its own graph, captured on first use, as the verify widths do. The graph reads the tokens, the positions and the pointer tables from device buffers, so it replays for any assignment of slots to rows. Membership changes rewrite the tables and pick the graph of the new size.

## New requests join

A new request takes a free slot and is prefilled into it in chunks of 2,048 rows, one chunk per engine iteration, between the batch's decode steps. Each chunk pauses the others for about 0.1 s. Smaller chunks would pause them less but change the prompt's KV state, since the prefill kernels are not row-invariant, and so break exactness. A request whose prefill finishes joins the next batched step with its first token.

Prefill into slot k runs the existing prefill code on that slot's caches. Prefill launches its kernels directly, not through a graph, so it can take any slot's caches.

## Speculation

Two ways to combine speculation with batching:

1. One sequence speculates while it runs alone; once a second request arrives, every active sequence runs batched plain decode.
2. Batched verify: every sequence drafts, and one pass verifies each sequence's anchor and drafts.

Option 2 gives more per stream. Its attention would extend the batched entry point to several rows per instance. It also needs a draft state per slot (draft caches, tap tails and shortlists), batched draft rounds, and a width policy across sequences. That roughly doubles the work.

Recommendation: option 1. A request that arrives at an idle server takes slot 0 and speculates. When another request arrives, the speculating one stops after its current round and joins the batch from slot 0, where its caches already are. When the batch drains to one request and nothing waits, that request speculates again. Since speculation and batched steps emit the same tokens, neither switch changes its output.

## Returning to speculation

A lone request in a batch runs a one-row batched step, 3.6 ms at 10,674 tokens, against about 1.3 to 2 ms per token for speculation. To speculate again it needs two things: its caches in slot 0, whose addresses the verify graphs hold, and a draft whose state covers every position before its next token.

The draft keeps one sequence's state. Speculation stops after a round or a run of plain steps, when the draft has committed every position before the anchor. From there the draft follows the request through its batched steps (`Speculator::follow`):

- Batched steps record every row's taps, as the verify pass does (`Model::set_batch_taps`). A batched row equals a one-row verify, so these are the taps speculation would have committed.
- After each step, the speculator copies the followed row's taps into its run buffer, 18 KiB per row with the bundled draft, and commits them to the draft every 32 rows (`Speculator::keep_batched`). The run buffer is the one the fallback's plain steps use.
- When the request is left alone, the speculator commits the remaining rows and makes the request's next token the anchor (`Speculator::resume`), and the round loop continues as after a prefill. This takes 0.1 ms.

A request that never speculated, or whose draft state was dropped because another request took the draft, has no draft state. The speculator rebuilds it: it prefills the request's prompt and generated tokens with taps into a free slot and commits every chunk, as a prompt's prefill does (`Speculator::rebuild`). The request's own caches stay as the batched steps left them, so its tokens do not change. Only the draft's state comes from prefill, whose sums differ slightly from the verify pass's, which can change the drafts but never the accepted tokens. A rebuild prefills about 20,000 tokens per second, and each remaining token saves about 1.7 ms, so it pays off once the remaining tokens exceed about 1/34 of the position. The server rebuilds when they exceed 1/16, since a request often stops before its limit; otherwise the request stays batched.

The request may sit in any slot. If it is not in slot 0, `Model::move_slot` copies its global caches up to its position, its rings and conv states into slot 0: 16 KiB per position with an FP8 cache plus 164 MiB, 331 MiB at 10,674 positions, as device-to-device copies on the stream.

Cost to the batch: recording the taps adds one copy kernel per tap visit, and following a row adds a 18 KiB copy per step and a 32-row commit every 32 steps. `bench --batch 1,2,8 --follow` against the same build without it, alternating in one locked hold, 96 steps, mean step:

| Context | Streams | Without | Following row 0 |
| ---: | ---: | ---: | ---: |
| 0 | 1 | 3.349, 3.345 | 3.367, 3.362 |
| 0 | 2 | 4.382, 4.384 | 4.407, 4.407 |
| 0 | 8 | 8.443, 8.450 | 8.440, 8.447 |
| 10,674 | 1 | 3.635, 3.640 | 3.652, 3.655 |
| 10,674 | 2 | 4.898, 4.898 | 4.929, 4.928 |
| 10,674 | 8 | 10.303, 10.308 | 10.286, 10.292 |

Following costs 0.015 to 0.03 ms per step at 1 and 2 streams, under 0.7%, and nothing measurable at 8.

`speculation_returns_after_batch_bundled` and `_dflash` in `tests/batch.rs` speculate, batch, and speculate again, and check every token against one-row verifies. The first sequence returns from slot 0 with its kept taps. The second returns from slot 1 after a rebuild in slot 2.

End to end, siphon-serve with `--streams 4` served three cases with the bundled draft and with DFlash 2 (`--drafter`), and every request returned the token ids it gets alone. In the first case a 600-token request speculates, a 60-token request joins for 59 batched steps, and the first speculates again. In the second, the first request reaches its 300-token limit in the batch and the second returns from slot 1 after a 33 ms rebuild at position 273. In the third, the first request leaves speculation twice. With the bundled draft, the first request's resumed speculation ran its last 265 tokens at 680 tok/s, and the same tokens of the request alone came at 693 tok/s, measured from the client. With DFlash 2 the figures were 699 and 715 tok/s.

## Throughput

The estimate before building took a batched step of B rows to cost a verify of B rows, plus the extra sequences' own KV reads (about 0.19 ms each at 10,674 tokens), plus about 0.1 ms for the separate post-attention conv kernels. That gave 434, 675 and 947 tok/s at 2, 4 and 8 streams at 10,674 tokens.

The measurements came out lower: 408, 576 and 797 tok/s. The experts explain most of the gap. A verify's rows are consecutive tokens of one text, which share experts: 8 rows touch 29 distinct experts per visit. Rows from 8 different texts touch about 50. A kernel trace at 10,674 tokens and 8 streams, also locked and alone on the GPU, showed the experts at 5.9 ms and attention at 2.1 ms of a 10.0 ms step, both close to DRAM bandwidth. That trace predates the tensor-core attention, which took 0.15 ms off a one-row verify at 10,674 tokens.

`bench --batch 1,2,3,4,6,8 --prompt <capture>`, FP8 weights and cache, on an RTX 5090, in one run under the GPU lock with no other compute process on the GPU but a 35 MiB desktop one. Each sequence's caches skip to the context, and row r feeds the `anyhow` capture's tokens from offset r / 8 of its length on, so each row routes as a different text does. The median step of 40, ms, with tok/s in all and per stream:

| Context | Plain step | One-row verify | 1 stream | 2 | 3 | 4 | 6 | 8 |
| ---: | ---: | ---: | --- | --- | --- | --- | --- | --- |
| 0 | 2.71 | 3.43 | 3.33: 300 | 4.37: 457, 229 | 5.22: 575, 192 | 6.01: 666, 166 | 7.30: 823, 137 | 8.24: 971, 121 |
| 4,096 | 2.86 | 3.55 | 3.56: 281 | 4.75: 421, 210 | 5.75: 522, 174 | 6.73: 594, 149 | 8.34: 719, 120 | 9.63: 831, 104 |
| 10,674 | 2.96 | 3.65 | 3.63: 275 | 4.90: 408, 204 | 5.92: 507, 169 | 6.94: 576, 144 | 8.60: 698, 116 | 10.04: 797, 100 |
| 32,768 | 3.23 | 3.96 | 3.93: 254 | 5.37: 373, 186 | 6.64: 452, 151 | 7.79: 513, 128 | 9.80: 612, 102 | 11.72: 683, 85 |

One stream in a batch matches a one-row verify. Plain decode runs 366 tok/s and speculation 490 to 800 tok/s for one sequence. With option 1, a speculating request drops to about 200 tok/s when a second arrives, so two streams together get about what one speculating stream gets alone. From three or four streams on, batching serves more in total. A lone sequence in a batch runs about 23% slower than plain decode, the price of keeping the verify's numerics, which is why it returns to speculation.

The two separate conv kernels cost about 0.15 ms per step. Folding the post-attention conv into depth attention would save most of that, once the depth-attention work settles.
