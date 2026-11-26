# siphon-serve

`siphon-serve` is an OpenAI-compatible HTTP server for Inkling 2 Mini. This page lists its options, routes and request fields. [engine.md](engine.md) covers how the engine works.

```sh
SIPHON_API_KEY=<key> target/release/siphon-serve <model-dir> --port 8000
```

## Startup

Before loading anything, the server plans the GPU memory from the config and the checkpoint headers. Without `--context`, it takes the longest context that fits the free memory less a 0.5 GiB margin, up to the model's 1,048,576 positions. With `--context`, it stops and prints the plan if that context does not fit. It prints the plan in two lines.

It then loads the model, prices each verify width on a short prompt, and serves. After warmup it compares the memory the process uses with the plan and prints both. It reads its own buffers from the CUDA memory pool that every allocation comes from, and the module images and CUDA graphs from the change in free memory. After each job it checks the pool's peak again and warns if the peak went past the plan.

`siphon-plan <model-dir> [--drafter DIR] [--memory GiB] [--free GiB] [--context N] [--print]` shows the same plan in a terminal before you start the server, and runs on macOS too. It draws the GPU memory as a bar and a table of what holds it, with the longest context for every combination of formats and drafts. Keys switch the formats and the draft and move the context. It reads the free memory from nvidia-smi, or takes `--free`, or assumes an idle 32 GB RTX 5090.

## Options

| Option | Default | Effect |
| --- | --- | --- |
| `--host`, `--port` | `127.0.0.1`, `8000` | Address to listen on. |
| `--name` | `inkling-2-mini` | Model id in responses and `/v1/models`. |
| `--context N` | most that fits | KV cache positions per stream: prompt plus generated tokens. |
| `--memory GiB` | free memory | Caps the GPU memory the plan may use, the CUDA context included. |
| `--weights fp8\|bf16` | `fp8` | Dense weights: FP8 copies or the checkpoint's BF16. |
| `--kv fp8\|bf16` | `fp8` | Global KV cache format. |
| `--streams N` | `1` | Requests that run at once, 1 to 32, each with its own KV cache. |
| `--queue N` | `8` | Requests that wait beyond the running ones. |
| `--widths` | `1,2,...,8` | Verify widths speculation picks from. |
| `--drafter DIR` | bundled DSpark | Runs Inco AI's DFlash 2 drafter or a 2DSD DFlash export instead. |
| `--draft` | `fast` | The draft's approximations: `fast`, `fp8`, `exact`, or a `+`-joined set of `fp8-backbone`, `fp8-head`, `fp8-fold` and `shortlist`. |
| `--fallback` | off | Runs plain steps where no verify width is expected to be faster. It speeds up free prose, but the output can then differ between runs. |
| `--no-prefix-cache` | cache on | Turns off the prefix cache and gives its memory, about 0.17 GiB per entry, to the KV cache. |
| `--prefix-cache-entries N` | `--streams` | Prompts the prefix cache keeps, 1 to `--streams`. |
| `--prefix-cache-ttl SECONDS` | `600` | Drops a prefix cache entry unused for that long. |
| `--prefill-attention auto\|bf16\|fp8` | `auto` | Prefill's global attention over an FP8 cache. `auto` runs BF16 up to 16,384 keys and FP8 tensor cores past them. |
| `--greedy` | off | Decodes every request greedily, whatever its sampling fields ask. |

With `SIPHON_API_KEY` set, every route but `/health`, `/version` and `/metrics` needs `Authorization: Bearer <key>` or `x-api-key: <key>`.

## Routes

| Route | Notes |
| --- | --- |
| `POST /v1/chat/completions` | Messages rendered as the checkpoint's chat template renders them. Streams with `stream`, and `stream_options.include_usage` adds a usage chunk. The model's thinking comes back as `reasoning_content`. |
| `POST /v1/completions` | A text prompt or token ids, streaming or not. The text includes any special tokens generated. |
| `POST /v1/messages` | Anthropic's Messages API, streaming or not, for Claude Code and Anthropic's SDKs. See [Anthropic Messages API](#anthropic-messages-api). |
| `POST /v1/messages/count_tokens` | `{"input_tokens": n}`: the tokens of the prompt that `/v1/messages` would run for the same body. |
| `POST /tokenize`, `POST /detokenize` | vLLM's shape, also at `/v1/tokenize` and `/v1/detokenize`: `{"prompt"}` or `{"messages", "tools"}` to `{"tokens", "count", "max_model_len"}`, and `{"tokens"}` to `{"prompt"}`. |
| `POST /v1/token/encode`, `POST /v1/token/decode` | TabbyAPI's shape: `{"text"}`, a string or messages, to `{"tokens", "length"}`, and `{"tokens"}` to `{"text"}`. |
| `GET /v1/models`, `GET /v1/models/{id}` | The served model, with `max_model_len`. |
| `GET /health`, `GET /version` | No API key needed. |
| `GET /metrics` | Prometheus metrics under vLLM's names, so tools written for vLLM read them. No API key needed. See [Metrics](#metrics). |

## Chat request fields

- `reasoning_effort`: a name such as `low`, or 0 to 0.99. The default is 0.9. Without it, `chat_template_kwargs.reasoning_effort` counts, and `chat_template_kwargs.enable_thinking: false` means effort 0. The template has no switch that turns thinking off, so at effort 0 the model still writes a short thinking block.
- `tools` and `tool_choice`: `auto`, `none`, `required`, or `{"type": "function", "function": {"name": ...}}`. A named function, or `required` with one tool, ends the prompt with that tool's call opened. `required` with several tools opens a call without a name, and the model names the tool. A forced call skips the thinking block. The reply ends after the named call, or at the first text after the calls `required` forced. `parallel_tool_calls: false` ends the reply after its first call.
- `stop`: a string or an array of strings. Stop strings match the reply's content, not its thinking or tool calls. While streaming, text that could still begin a stop string is held back. The reply is cut before the match and finishes with `stop`.
- vLLM's extensions:
  - `return_token_ids` adds `prompt_token_ids` and each choice's `token_ids`, streaming or not. The ids end with `<|content_model_end_sampling|>` when the model ended its turn.
  - `stop_token_ids`: token ids that end the reply. As in vLLM, the reply keeps the stop token's id in `token_ids` and leaves its text out, and the choice finishes with `stop`. The check runs on the server's output, so the engine generates one more round before it stops.
  - Each choice has `stop_reason`: the stop string that matched, the stop token's id, or null.
  - `ignore_eos` generates `max_tokens` tokens and passes the model's end tokens on as output, for benchmarks.
  - `continue_final_message` with `add_generation_prompt: false` continues the last assistant message.

Raw completions take `stop`, `stop_token_ids`, `return_token_ids` and `ignore_eos` too. A text prompt, like the text sent to the tokenizer routes, is encoded with special-token spellings as plain text and nothing added; send token ids to use special tokens. Decoding keeps special tokens.

## Anthropic Messages API

`POST /v1/messages` takes Anthropic's request and answers in Anthropic's shape. The server renders the request with the same chat template as `/v1/chat/completions`, and runs it on the same engine. Query parameters, such as Claude Code's `?beta=true`, are ignored.

The request maps onto the chat route as follows:

- `system` is a string or text blocks. The blocks are joined with blank lines into one system message. Claude Code's `x-anthropic-billing-header` block is dropped: it changes between requests and would defeat the prefix cache.
- `messages` take the roles `user`, `assistant` and `system`. The template renders a `system` message mid-conversation where it stands.
- A user's `tool_result` block becomes a tool message, in its place among the user's text blocks. Its `is_error` is ignored.
- `image` and `document` blocks, also inside a `tool_result`, become the text `[image omitted: the model cannot see images]` or `[document omitted: the model cannot read documents]`. The model sees no images, and a 400 would end Claude Code's session.
- An assistant's `thinking` blocks become its past reasoning, which the template renders before its text and tool calls. `redacted_thinking` is dropped.
- A last `assistant` message is a prefill, and the reply continues it.
- `tools` entries with a `type` other than `custom` are Anthropic's server tools, such as web search, and are left out.
- `tool_choice`: `auto`, `any`, `{"type": "tool", "name"}` and `none` work as `auto`, `required`, a named function and `none` do on the chat route. `disable_parallel_tool_use` ends the reply after its first call.
- `thinking: {"type": "disabled"}` is effort 0. Otherwise `output_config.effort` (`low`, `medium`, `high`, `xhigh` or `max`) sets the effort as `reasoning_effort` does, and the default is 0.9. `budget_tokens` and unknown effort names are ignored.
- `max_tokens` is required. A prompt and `max_tokens` that do not fit the context get a 400 with Anthropic's message, which Claude Code reads to shrink `max_tokens` or to compact.
- `stop_sequences` are stop strings. `temperature`, `top_p` and `top_k` sample as on the chat route.

The server ignores `metadata`, `cache_control` anywhere, `context_management`, `service_tier`, `output_config.format`, `container`, `mcp_servers`, the `anthropic-beta` and `anthropic-version` headers, and unknown fields.

The reply holds `thinking` blocks with an empty `signature`, then `text` and `tool_use` blocks in the order the model wrote them. `stop_reason` is `end_turn`, `max_tokens`, `stop_sequence` (with `stop_sequence` set) or `tool_use`. In `usage`, `input_tokens` leaves out the prompt tokens read from the prefix cache, which `cache_read_input_tokens` counts, and `cache_creation_input_tokens` is 0. A stream sends Anthropic's events, from `message_start` to `message_stop`, and a `ping` event after 5 seconds without bytes, such as during a long prefill. A tool call arrives whole, in one `input_json_delta`.

Errors on these routes take Anthropic's shape, `{"type": "error", "error": {"type", "message"}}`. A full queue answers 529 with `overloaded_error`, which Claude Code retries. A failure during a stream sends an `error` event.

### Claude Code

Point Claude Code at the server with these variables:

```sh
export ANTHROPIC_BASE_URL=http://<host>:8000   # no /v1
export ANTHROPIC_AUTH_TOKEN=<key>              # or ANTHROPIC_API_KEY; any text without SIPHON_API_KEY
export ANTHROPIC_MODEL=inkling-2-mini          # the server's --name
export ANTHROPIC_DEFAULT_HAIKU_MODEL=inkling-2-mini
export CLAUDE_CODE_MAX_CONTEXT_TOKENS=<context>
```

Claude Code sends background requests, such as titles and summaries, to its Haiku model, so that name must point at this server too. It assumes a context of 1,048,576 tokens for a model it does not know. Set `CLAUDE_CODE_MAX_CONTEXT_TOKENS` to the server's `--context`, or to the `max_model_len` that `/v1/models` lists, so that it compacts the conversation before the prompt overflows.

With `--streams 1`, the [prefix cache](#prefix-cache) holds one prompt. A background request that does not start with the conversation's tokens then replaces it with its own prompt, and the next turn of the main conversation prefills its whole prompt again. With `--streams 2` or more, the background request's prompt takes another entry, and the conversation keeps its own.

## Sampling

Both routes sample with `temperature`, `top_k`, `top_p`, `min_p` and `seed`, in vLLM's order: temperature, then top-k, then top-p, then min-p. A field the request leaves out comes from the checkpoint's `generation_config.json`, as in vLLM. For Inkling 2 Mini that is temperature 1 with no filters. `top_k` of 0 or -1 keeps every token.

`temperature: 0` decodes greedily and gives the same tokens as earlier greedy builds. A request with a `seed` gets the same tokens again on the same server configuration, alone or in a batch. A request without one gets a random seed. `--greedy` decodes every request greedily.

## Several choices

Both routes take `n`, the number of choices to generate from one prompt, 1 by default. Choice `i` of a request with seed `s` samples with seed `s + i`, as vLLM's parallel sampling seeds its child requests. Choice 0 so gets the tokens the same request with `n: 1` gets. Without a seed, each choice gets its own random seed. Greedy choices would all be the same, so `n` above 1 gets a 400 with temperature 0, as in vLLM, and on a server started with `--greedy`.

Each choice runs as its own request in the engine, with its own slot, and has its own `index`, `finish_reason`, `stop_reason`, text, reasoning, tool calls and `token_ids`. Stop strings, `stop_token_ids`, `ignore_eos`, `tool_choice` and the reasoning parser apply to each choice alone. A choice that a stop ends stops generating; the others go on.

A stream sends one choice per chunk, as vLLM and OpenAI do. A chat stream starts with one role chunk per choice, in index order. The choices' chunks then interleave as the engine generates them, and each choice ends with its own chunk holding its `finish_reason`. In `usage`, the prompt counts once and `completion_tokens` sums the choices. `prompt_tokens_details.cached_tokens` is choice 0's.

A request with `n` choices takes `n` places among the running and queued requests: up to `--queue` may wait beyond the `--streams` that run. If they do not all fit, the request gets the queue-full 503, and none of its choices runs. An `n` above `--streams` plus `--queue` gets a 400. A client that leaves cancels every choice.

With `--streams` of 2 or more, the choices run batched, not speculating. The first choice to prefill copies its state into the slots of the others, and each draws its first token from the same logits with its own seed, so the prompt is prefilled once and each choice gets the tokens of its own prefill. Choices that start later, because the slots were full, prefill the prompt themselves. With `--streams 1`, the choices run one after another, and each one after the first reads the prompt up to its last multiple of 2,048 tokens from the [prefix cache](#prefix-cache).

With `--streams 8` and the prefix cache off, one request with `n: 4` on a 12,916-token prompt, 256 tokens per choice, took 2.67 s. Four concurrent `n: 1` requests with the same seeds took 5.31 s and returned the same tokens. Streamed, the `n: 4` request's first token came after 0.87 s.

## Accepted, ignored and refused fields

The penalties (`presence_penalty`, `frequency_penalty`, `repetition_penalty`) and other sampling fields are accepted and ignored. So are `user`, `skip_special_tokens` and unknown fields.

Fields the server cannot honour get a 400: `best_of` other than `n` (ranking choices needs log probabilities), `logprobs`, `top_logprobs`, `prompt_logprobs`, `echo`, `suffix`, `logit_bias`, `min_tokens`, `response_format` other than `text`, vLLM's guided decoding fields, and the deprecated `functions` and `function_call`.

## Metrics

`GET /metrics` serves Prometheus text with vLLM's metric names and a `model_name` label. The server records them as each request's events arrive at the HTTP layer:

| Metric | Type | Counts |
| --- | --- | --- |
| `vllm:prompt_tokens_total` | counter | Prompt tokens of requests that reached their first token. |
| `vllm:generation_tokens_total` | counter | Tokens the engine generated, including any a stop string cut off. |
| `vllm:prefix_cache_queries_total`, `vllm:prefix_cache_hits_total` | counter | Prompt tokens looked up in the prefix cache, and those found there. |
| `vllm:request_success_total` | counter | Finished requests, labelled `finished_reason`: `stop`, `length`, `abort` (the client left) or `error`. |

Each choice of a request with `n` choices counts as a request in the request counts, the gauges and the latency histograms, since it runs as its own sequence. Its prompt counts once, with choice 0, in `vllm:prompt_tokens_total` and the prefix cache counters.
| `vllm:num_requests_running`, `vllm:num_requests_waiting` | gauge | Requests the engine has started, and requests still in its queue. |
| `vllm:request_queue_time_seconds` | histogram | Time from queueing to the engine starting the request. |
| `vllm:time_to_first_token_seconds` | histogram | Time from queueing to the first token. |
| `vllm:inter_token_latency_seconds` | histogram | Time per token after the first. Speculation delivers several tokens at once, so a gap that delivered n tokens counts n times as gap / n. The buckets start at 0.5 ms, below vLLM's 10 ms. |
| `vllm:e2e_request_latency_seconds` | histogram | Time from queueing to the end of the request. |

## Concurrent requests

With `--streams N`, up to N requests run at once and up to `--queue` more wait. Each stream has its own KV cache of `--context` positions. Without `--context`, the server takes the longest context at which N caches fit. `siphon-plan --streams N` shows that context: with FP8 weights and caches, 77,824 positions for 8 streams on a 32 GB card, and 283,648 for 16 or 131,072 for 32 on an RTX PRO 6000 Blackwell Max-Q.

More streams serve more in total and less per stream. On an RTX PRO 6000 Blackwell Max-Q, 32 streams at 10,674 tokens of context ran 1,138 tok/s in all against 661 for 8 ([batching.md](batching.md#16-and-32-streams-on-an-rtx-pro-6000-blackwell-max-q)). With `--streams 32 --context 16384`, 32 concurrent requests of 512 tokens came back at 1,198 tok/s in all.

A request that arrives at an idle server speculates. When another arrives, the first stops speculating after its round, and every running request decodes in a batched step of one row each. When one request is left and nothing waits, it speculates again. A new request's prompt prefills in chunks of 2,048 tokens between the batch's steps. Either way each request gets the same tokens it would get alone, greedy or sampled with a seed. [batching.md](batching.md) has the design.

## Prefix cache

Chat clients and agents resend the whole conversation each turn. The server keeps the state of recent prompts, one per stream unless `--prefix-cache-entries` says fewer, each at its last multiple of 2,048 tokens. A request whose prompt starts with one of them prefills only the rest, batched or alone, and gets the same tokens and logits as without the cache. The usage object reports the reused tokens as `prompt_tokens_details.cached_tokens`.

Each entry belongs to a stream's KV cache, so a request that hits one runs in that stream. A new prompt takes a stream without an entry, else the one whose entry was used least recently, and an entry unused for `--prefix-cache-ttl` seconds is dropped. So up to `--streams` conversations, such as an agent and its parallel subagents, each keep their own history cached between turns. Prompts that share only a beginning, such as two conversations with the same system prompt, do not share an entry.

A request that runs alone speculates. When its entry was saved by a batched request, it lacks the draft's state: the request then prefills from the entry and decodes batched, and rebuilds the draft's state, a prefill of its whole prompt, only when its `max_tokens` leaves more than 1/16 of the position to generate (see [engine.md](engine.md#prefix-cache)). Each entry takes 172 MiB with the bundled draft and 204 MiB with DFlash 2, whatever the context.

## Building on macOS

On macOS, `siphon-cuda` compiles to an empty crate and `siphon-serve` builds without its engine, so its tests run but it cannot serve. From a Mac, `scripts/remote.sh <ssh-host> <remote-dir> <command>` syncs the workspace to a Linux host and runs the command in its pixi environment. Set `NVCC` or `CUDA_HOME` to use another toolkit, and `SIPHON_CUDA_ARCH` to target another GPU.
