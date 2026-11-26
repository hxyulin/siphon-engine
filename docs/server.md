# siphon-serve

`siphon-serve` is an OpenAI-compatible HTTP server for Inkling 2 Mini. This page lists its options, routes and request fields. [engine.md](engine.md) covers how the engine works.

```sh
SIPHON_API_KEY=<key> target/release/siphon-serve <model-dir> --port 8000
```

## Startup

Before loading anything, the server plans the GPU memory from the config and the checkpoint headers. The global KV caches of every stream share one pool of pages ([Concurrent requests](#concurrent-requests)). Without `--context`, the pool takes the free memory less `--margin` (0.25 GiB by default) and what the weights and buffers need, up to the model's 1,048,576 positions. With `--context`, it stops and prints the plan if that context does not fit. It prints the plan in two lines.

It then loads the model, prices each verify width on a short prompt, and serves. After warmup it compares the memory the process uses with the plan and prints both. It reads its own buffers from the CUDA memory pool that every allocation comes from, and the module images and CUDA graphs from the change in free memory. After each job it checks the pool's peak again and warns if the peak went past the plan.

`siphon-plan <model-dir> [--drafter DIR] [--memory GiB] [--margin GiB] [--prefill-chunk N] [--free GiB] [--context N] [--streams N] [--vision] [--print]` shows the same plan in a terminal before you start the server, and runs on macOS too. It draws the GPU memory as a bar and a table of what holds it, with the longest context for every combination of formats and drafts. Keys switch the formats and the draft and move the context. It reads the free memory from nvidia-smi, or takes `--free`, or assumes an idle 32 GB RTX 5090.

## Options

| Option | Default | Effect |
| --- | --- | --- |
| `--host`, `--port` | `127.0.0.1`, `8000` | Address to listen on. |
| `--name` | `inkling-2-mini` | Model id in responses and `/v1/models`. |
| `--context N` | most that fits | KV cache positions of the pool that every running request shares: prompts plus generated tokens. One request may take all of them. |
| `--memory GiB` | free memory | Caps the GPU memory the plan may use, the CUDA context included. |
| `--margin GiB` | `0.25` | Free memory the plan leaves unclaimed. |
| `--prefill-chunk N` | `2048` | Rows of each prefill chunk, 1 to 2048. 1024 frees 0.4 GiB for the context and costs 6 to 7% of prefill speed. |
| `--weights fp8\|bf16` | `fp8` | Dense weights: FP8 copies or the checkpoint's BF16. |
| `--output-projection bf16\|fp8` | `bf16` | The attention output projection decode reads with FP8 weights. `fp8` saves 4 to 6% of a plain step and moves the logits further from the model. |
| `--kv fp8\|bf16\|nvfp4\|mixed` | `fp8` | Global KV cache format: 16, 32, 9 or 10.75 KiB per position. See [engine.md](engine.md#kv-cache-reduction). |
| `--streams N` | `1` | Requests that run at once at most, 1 to 32, as many as the pool holds. |
| `--queue N` | `8` | Requests that wait beyond the running ones. |
| `--decode-floor R` | `20` | Tokens per second each decoding request gets at least while another prefills. 0 runs one batched step per prefill chunk. |
| `--widths` | `1,2,...,8` | Verify widths speculation picks from. |
| `--drafter DIR` | bundled DSpark | Runs Inco AI's DFlash 2 drafter or a 2DSD DFlash export instead. |
| `--draft` | `fast` | The draft's approximations: `fast`, `fp8`, `exact`, or a `+`-joined set of `fp8-backbone`, `fp8-head`, `fp8-fold` and `shortlist`. |
| `--fallback` | off | Runs plain steps where no verify width is expected to be faster. It speeds up free prose, but the output can then differ between runs. |
| `--no-prefix-cache` | cache on | Turns off the prefix cache and gives its memory, about 0.17 GiB per entry, to the KV cache. |
| `--prefix-cache-entries N` | `--streams` | Prompts the prefix cache keeps, 1 to `--streams`. |
| `--prefix-cache-ttl SECONDS` | `600` | Drops a prefix cache entry unused for that long. |
| `--host-cache GiB` | `0`, off | Keeps prefix cache entries in that much page-locked host memory too, global KV cache included, so a conversation whose stream another request took restores over PCIe instead of prefilling again. Needs the default `--prefill-chunk` of 2048. |
| `--cache-dir DIR` | none | Saves the host cache to DIR when the server stops and loads it at startup. Needs `--host-cache`. See [Restarts](#restarts). |
| `--drain-timeout SECONDS` | `600` | How long a drain on SIGTERM waits for the running and queued requests before it ends them. |
| `--prefill-attention auto\|bf16\|fp8` | `auto` | Prefill's global attention over an FP8 cache. `auto` runs BF16 up to 16,384 keys and FP8 tensor cores past them. |
| `--greedy` | off | Decodes every request greedily, whatever its sampling fields ask. |
| `--vision` | off | Loads the vision tower and takes images in requests. See [Images](#images). |
| `--max-image-pixels N` | `2073600` | Shrinks a larger image to N pixels, keeping its aspect ratio. The default is 1920 x 1080. |
| `--max-images N` | `16` | Images one request may hold. |

With `SIPHON_API_KEY` set, every route but `/health`, `/version` and `/metrics` needs `Authorization: Bearer <key>` or `x-api-key: <key>`.

Each request gets an ID. The server takes the client's `x-request-id`, or else `x-client-request-id`, when it holds 1 to 128 printable ASCII characters without spaces, and otherwise makes one such as `req-5e0c1f7a9b3d2e64`. Every response, errors included, returns the ID in `x-request-id`. The `job:` and `job failed:` log lines start with it and the response's own id, so a proxy's request log matches them:

```text
job: trace-7f3a chatcmpl-6929a1c4000003: 31204 prompt tokens (30720 cached), prefill 48 ms, ...
```

A request with `n` choices logs one line per choice, labelled `choice i`.

## Routes

| Route | Notes |
| --- | --- |
| `POST /v1/chat/completions` | Messages rendered as the checkpoint's chat template renders them. Streams with `stream`, and `stream_options.include_usage` adds a usage chunk. The model's thinking comes back as `reasoning_content`. |
| `POST /v1/completions` | A text prompt or token ids, streaming or not. The text includes any special tokens generated. |
| `POST /v1/messages` | Anthropic's Messages API, streaming or not, for Claude Code and Anthropic's SDKs. See [Anthropic Messages API](#anthropic-messages-api). |
| `POST /v1/messages/count_tokens` | `{"input_tokens": n}`: the tokens of the prompt that `/v1/messages` would run for the same body. |
| `POST /tokenize`, `POST /detokenize` | vLLM's shape, also at `/v1/tokenize` and `/v1/detokenize`: `{"prompt"}` or `{"messages", "tools"}` to `{"tokens", "count", "max_model_len"}`, and `{"tokens"}` to `{"prompt"}`. With `--vision`, the messages' images count as the chat route counts them. |
| `POST /v1/token/encode`, `POST /v1/token/decode` | TabbyAPI's shape: `{"text"}`, a string or messages, to `{"tokens", "length"}`, and `{"tokens"}` to `{"text"}`. |
| `GET /v1/models`, `GET /v1/models/{id}` | The served model, with `max_model_len`. |
| `GET /health`, `GET /version` | No API key needed. |
| `GET /metrics` | Prometheus metrics under vLLM's names, so tools written for vLLM read them. No API key needed. See [Metrics](#metrics). |

## Chat request fields

- `reasoning_effort`: a name such as `low`, or 0 to 0.99. The default is 0.9. Without it, `chat_template_kwargs.reasoning_effort` counts. `none` and `chat_template_kwargs.enable_thinking: false` turn thinking off, and a request without tools then gets no thinking block (see [Thinking off](#thinking-off)). A number is passed to the template as it is, and at effort 0 the model still writes a short thinking block.
- `tools` and `tool_choice`: `auto`, `none`, `required`, or `{"type": "function", "function": {"name": ...}}`. A named function, or `required` with one tool, ends the prompt with that tool's call opened. `required` with several tools opens a call without a name, and the model names the tool. A forced call skips the thinking block. The reply ends after the named call, or at the first text after the calls `required` forced. `parallel_tool_calls: false` ends the reply after its first call.
- `stop`: a string or an array of strings. Stop strings match the reply's content, not its thinking or tool calls. While streaming, text that could still begin a stop string is held back. The reply is cut before the match and finishes with `stop`.
- `max_tokens` or `max_completion_tokens` past the context is cut to the positions left after the prompt, and the reply then finishes with `length`. A client's token count can differ from the server's, so a cap that a client sized to the context can overshoot it. vLLM refuses such a request. A prompt that fills the context alone still gets a 400.
- A stream sends each choice's role chunk with the first token, as vLLM does, so a proxy that times the first chunk measures the prefill. During a long prefill, a keep-alive comment every 15 seconds holds the connection open.
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
- With `--vision`, `image` blocks with a `base64` source, also inside a `tool_result`, reach the prompt as images (see [Images](#images)). An image block with a `url` source gets a 400. Without `--vision`, `image` blocks become the text `[image omitted: the model cannot see images]`, since a 400 would end Claude Code's session. `document` blocks always become `[document omitted: the model cannot read documents]`.
- An assistant's `thinking` blocks become its past reasoning, which the template renders before its text and tool calls. `redacted_thinking` is dropped.
- A last `assistant` message is a prefill, and the reply continues it.
- `tools` entries with a `type` other than `custom` are Anthropic's server tools, such as web search, and are left out.
- `tool_choice`: `auto`, `any`, `{"type": "tool", "name"}` and `none` work as `auto`, `required`, a named function and `none` do on the chat route. `disable_parallel_tool_use` ends the reply after its first call.
- `thinking: {"type": "disabled"}` turns thinking off (see [Thinking off](#thinking-off)). Without tools, the reply then has no `thinking` block. With tools, the model still thinks briefly before it decides on a tool call. Otherwise `output_config.effort` (`low`, `medium`, `high`, `xhigh` or `max`) sets the effort as `reasoning_effort` does, and the default is 0.9. `budget_tokens` and unknown effort names are ignored.
- `max_tokens` is required. A prompt and `max_tokens` that do not fit the context get a 400 with Anthropic's message, which Claude Code reads to shrink `max_tokens` or to compact.
- `stop_sequences` are stop strings. `temperature`, `top_p` and `top_k` sample as on the chat route.

The server ignores `metadata`, `cache_control` anywhere, `context_management`, `service_tier`, `output_config.format`, `container`, `mcp_servers`, the `anthropic-beta` and `anthropic-version` headers, and unknown fields.

The reply holds `thinking` blocks with an empty `signature`, then `text` and `tool_use` blocks in the order the model wrote them. `stop_reason` is `end_turn`, `max_tokens`, `stop_sequence` (with `stop_sequence` set) or `tool_use`. In `usage`, `input_tokens` leaves out the prompt tokens read from the prefix cache, which `cache_read_input_tokens` counts, and `cache_creation_input_tokens` is 0. A stream sends Anthropic's events, from `message_start` to `message_stop`. `message_start` waits for the reply's first content block, as the chat route's role chunk waits for the first token, so a proxy that times the first event measures the prefill. Before it, a stream sends an SSE comment after 5 seconds without bytes, such as during a long prefill, because Anthropic's protocol puts `message_start` first. After it, the stream sends a `ping` event instead. A tool call arrives whole, in one `input_json_delta`.

Errors on these routes take Anthropic's shape, `{"type": "error", "error": {"type", "message"}}`. A full queue answers 529 with `overloaded_error`, which Claude Code retries. A failure during a stream sends an `error` event, also before `message_start`.

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

With `--streams 1`, a background request waits for the conversation's reply, and the [prefix cache](#prefix-cache) holds one prompt: the background request then replaces it with its own, and the next turn of the main conversation prefills its whole prompt again. With `--streams 4`, the background request starts beside the conversation's reply, in the same pool of KV cache pages, and takes another entry, so the conversation keeps its own.

## Images

With `--vision`, the server loads the vision tower beside the model and takes images in the chat and Messages routes:

- `/v1/chat/completions`: content parts `{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}`, also with `image_url` as a plain string, and `input_image` parts. `detail` is ignored.
- `/v1/messages`: `image` blocks with `{"type": "base64", "media_type", "data"}` sources, in user messages and inside `tool_result` blocks.

Images must be inline: PNG, JPEG, WebP or GIF (its first frame) in base64. The server does not fetch URLs, so an `http` or `https` image URL gets a 400 that says so. Images are allowed in user and tool messages only, as the chat template allows them.

Each image goes through the reference preprocessing ([vision.md](vision.md)). An image of more than `--max-image-pixels` shrinks to fit with PIL's bicubic filter, keeping its aspect ratio. Its sides are then padded up to multiples of 40 pixels, and each 40 x 40 square becomes one image token: 676 tokens for a 1024 x 1024 image and 1,296 for 1920 x 1080. The prompt holds `<|content_image|>`, one `<|unused_200054|>` placeholder per image token and `<|end_message|>`, after the message's header, where the chat template puts an image. `usage.prompt_tokens` and the tokenizer routes count the placeholders.

A request gets a 400 when it holds more than `--max-images` images, when an image is not one of the four formats or does not decode, or when an image has more than 50 million pixels before shrinking. An image that still needs more image tokens after shrinking than a square of `--max-image-pixels` with one more row and column of tokens, 1,369 by default, gets a 400 too; only very narrow or wide images do. The request body may grow by 20 MiB per image allowed.

The tower runs on the engine thread right before the prefill chunk that first reaches an image, and its time counts as prefill for `--decode-floor`. Prefill writes the image's rows over the placeholders' embeddings after `embed_norm`, as Transformers does, and positions stay sequential. An image that crosses a prefill chunk boundary is encoded once and its rows serve both chunks.

Two prompts with different images have the same tokens, so the [prefix cache](#prefix-cache) does not match prompts by their tokens alone. Its device entries, its host tier and the files in `--cache-dir` hold each prompt's key: its tokens with each image's placeholders replaced by ids above the vocabulary, taken from the SHA-256 of the image file and from `--max-image-pixels`. A prompt whose images differ from an entry's below the entry's boundary does not match it. A follow-up turn of a conversation with images reads every image below its boundary from the cache, and the tower encodes only the images past it.

The speculative draft reads the target's hidden states at every prompt position, image positions included, so it sees what the target sees. Verify decides every token, so images do not change speculation's output either way. Acceptance on replies about images is not measured yet.

`siphon-plan --vision [--max-image-pixels N]` adds the tower's weights, its buffers for the largest image and the image rows of one prefill chunk to the plan. On an idle 32 GB RTX 5090, `--vision` at the defaults takes 0.69 GiB and cuts the longest context with FP8 weights and caches from 657,408 to 617,472 positions, with the default 0.25 GiB margin. Without `--vision` the plan and the server are as before.

## Sampling

Both routes sample with `temperature`, `top_k`, `top_p`, `min_p` and `seed`, in vLLM's order: temperature, then top-k, then top-p, then min-p. A field the request leaves out comes from the checkpoint's `generation_config.json`, as in vLLM. For Inkling 2 Mini that is temperature 1 with no filters. `top_k` of 0 or -1 keeps every token.

`temperature: 0` decodes greedily and gives the same tokens as earlier greedy builds. A request with a `seed` gets the same tokens again on the same server configuration, alone or in a batch. A request without one gets a random seed. `--greedy` decodes every request greedily.

## Several choices

Both routes take `n`, the number of choices to generate from one prompt, 1 by default. Choice `i` of a request with seed `s` samples with seed `s + i`, as vLLM's parallel sampling seeds its child requests. Choice 0 so gets the tokens the same request with `n: 1` gets. Without a seed, each choice gets its own random seed. Greedy choices would all be the same, so `n` above 1 gets a 400 with temperature 0, as in vLLM, and on a server started with `--greedy`.

Each choice runs as its own request in the engine, with its own slot, and has its own `index`, `finish_reason`, `stop_reason`, text, reasoning, tool calls and `token_ids`. Stop strings, `stop_token_ids`, `ignore_eos`, `tool_choice` and the reasoning parser apply to each choice alone. A choice that a stop ends stops generating; the others go on.

A stream sends one choice per chunk, as vLLM and OpenAI do. A chat stream starts with one role chunk per choice, in index order. The choices' chunks then interleave as the engine generates them, and each choice ends with its own chunk holding its `finish_reason`. In `usage`, the prompt counts once and `completion_tokens` sums the choices. `prompt_tokens_details.cached_tokens` is choice 0's.

A request with `n` choices takes `n` places among the running and queued requests: up to `--queue` may wait beyond the `--streams` that run. If they do not all fit, the request gets the queue-full 503, and none of its choices runs. An `n` above `--streams` plus `--queue` gets a 400. A client that leaves cancels every choice.

With `--streams` of 2 or more, the choices run batched, not speculating. The first choice to prefill copies its state into the slots of the others, and each draws its first token from the same logits with its own seed, so the prompt is prefilled once and each choice gets the tokens of its own prefill. Choices that start later, because the slots were full, prefill the prompt themselves. With `--streams 1`, the choices run one after another, and each one after the first reads the prompt up to its last multiple of `--prefill-chunk` tokens from the [prefix cache](#prefix-cache).

With `--streams 8` and the prefix cache off, one request with `n: 4` on a 12,916-token prompt, 256 tokens per choice, took 2.67 s. Four concurrent `n: 1` requests with the same seeds took 5.31 s and returned the same tokens. Streamed, the `n: 4` request's first token came after 0.87 s.

## Thinking off

The chat template has no switch that turns thinking off. At effort 0 the model still writes a thinking block before it answers, and a small `max_tokens` can go entirely to that block. When a request turns thinking off and declares no tools, the server renders effort 0 and ends the prompt with a thinking block of its own, `<|message_model|><|content_thinking|>Respond directly.<|end_message|>`, then the `<|message_model|>` that starts the reply. The model ends its own effort-0 thinking with those words. It then writes its reply at once. An empty thinking block does not work: the model opens a second one and thinks there.

On 37 prompts at temperature 0 (20 short facts, 4 math word problems, 3 small coding tasks with tests, 6 open requests including a multi-turn one), effort 0 thought on every prompt and its text began after a median of 70 tokens, at most 653. With thinking off, no reply had a thinking block, the text began at the first generated token, and the replies took a median of 22 tokens against 93. Both answered all 20 facts, 4 math problems and 3 coding tasks correctly, and finished all 6 open requests. At temperature 1 the results matched, with a median of 22 tokens against 101. Without thinking, the model sometimes works a problem out in its reply text.

A request with tools keeps effort 0's own thinking block, because the model decides on a tool call there. With the server's thought, or with the reply forced to start as text, the model answered 4 tool requests in text and called no tool. At effort 0 it called the right tool with the right arguments in 4 of 4. `tool_choice` `none` removes the tools, so such a request gets the thought. A forced tool call or an assistant prefill already skips the thinking block and gets no thought.

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
| `vllm:num_requests_running`, `vllm:num_requests_waiting` | gauge | Requests the engine has started, and requests still in its queue. |
| `vllm:request_queue_time_seconds` | histogram | Time from queueing to the engine starting the request. |
| `vllm:time_to_first_token_seconds` | histogram | Time from queueing to the first token. |
| `vllm:inter_token_latency_seconds` | histogram | Time per token after the first. Speculation delivers several tokens at once, so a gap that delivered n tokens counts n times as gap / n. The buckets start at 0.5 ms, below vLLM's 10 ms. |
| `vllm:e2e_request_latency_seconds` | histogram | Time from queueing to the end of the request. |

Each choice of a request with `n` choices counts as a request in the request counts, the gauges and the latency histograms, since it runs as its own sequence. Its prompt counts once, with choice 0, in `vllm:prompt_tokens_total` and the prefix cache counters.

### Request timings

Each completion, chat and Messages response carries a `timings` object with the field names of llama.cpp's server, so tools written for llama.cpp read them. Siphon adds its queue time, time to first token and speculation:

| Field | Value |
| --- | --- |
| `cache_n` | Prompt tokens read from the prefix cache. |
| `prompt_n` | Prompt tokens prefilled: the prompt less `cache_n`. |
| `prompt_ms`, `prompt_per_token_ms`, `prompt_per_second` | Time from the engine starting the request to its first token, per `prompt_n` token, and `prompt_n` per second. |
| `predicted_n` | Generated tokens in the reply, as `usage` counts them. |
| `predicted_ms`, `predicted_per_token_ms`, `predicted_per_second` | Time from the first token to the end, per `predicted_n` token, and `predicted_n` per second. |
| `queue_ms` | Time from queueing to the engine starting the request. |
| `ttft_ms` | Time from queueing to the first token: `queue_ms` plus `prompt_ms`. |
| `draft_rounds`, `tokens_per_round` | Speculation rounds and the tokens each delivered on average, when the request speculated. A plain step under `--fallback` counts as a round of one token. |

The times are taken when the engine's events reach the HTTP layer. The queue starts once the server has parsed and tokenized the request, so parsing does not count as prefill. The prompt time includes any prefix cache restore and image encoding. A request with `n` choices counts from its earliest start and first token to its last end, and sums the choices' tokens and rounds.

A response that is not streamed has `timings` at the top level and a `Server-Timing` header, `queue;dur=…, prefill;dur=…, decode;dur=…` in milliseconds. A stream has sent its headers before the times are known, so its last chunk holds `timings`: in a chat or completion stream, the chunk with the last `finish_reason`, or the usage chunk when `stream_options.include_usage` asks for one, and in a Messages stream, the `message_delta` event. OpenAI's and Anthropic's SDKs ignore fields they do not know.

The log repeats the numbers in the request's last `job:` line:

```text
job: trace-7f3a chatcmpl-6929a1c4000003: timings {"cache_n":30720,"predicted_ms":1210.4,"predicted_n":612,...}
```

## Concurrent requests

With `--streams N`, up to N requests run at once and up to `--queue` more wait. Their global KV caches and sliding rings share one pool of 2 MiB pages, which each takes as its sequence grows and keeps after it ends until the pool needs them, so one long conversation and several short requests run together, and one request alone may fill the pool. `--streams` costs little past the first: on a 32 GB card with FP8 weights and caches, with 30.45 GiB free and the default margin, the largest single request takes 653,312 positions with `--streams 1`, 645,120 with 4 and 641,024 with 8. Before the pool, `--streams 1` took 657,408 and 4 streams 161,792 each. [batching.md](batching.md#memory-one-pool-of-pages) has the design.

A request starts when the pool has room for its prompt and 2,048 positions more. The shortest prompts start first, unless a request has waited 20 s. When the running requests outgrow the pool, prompts that the prefix cache keeps for idle streams go first. With `--host-cache`, requests that still do not fit then take turns: one is swapped out to host memory, the others run, and it comes back, each running at least nine times as long as its swap takes. A request with images that is swapped out mid-prefill encodes the image its next chunk needs again when it returns. Without `--host-cache`, a request starts only when the pool holds every position it may reach.

More streams serve more in total and less per stream. On an RTX PRO 6000 Blackwell Max-Q, 32 streams at 10,674 tokens of context ran 1,138 tok/s in all against 661 for 8 ([batching.md](batching.md#16-and-32-streams-on-an-rtx-pro-6000-blackwell-max-q)). With `--streams 32 --context 16384`, 32 concurrent requests of 512 tokens came back at 1,198 tok/s in all.

A request that arrives at an idle server speculates. When another arrives, the first stops speculating after its round, and every running request decodes in a batched step of one row each. When one request is left and nothing waits, it speculates again. A new request's prompt prefills in chunks of `--prefill-chunk` tokens (2,048 by default) between the batch's steps. After each chunk the others run as many steps as give each at least `--decode-floor` tokens per second: with the default of 20, a 100,000-token prefill on an RTX 5090 took 8% longer and the decoding request got 22.4 tok/s instead of 6.1 ([batching.md](batching.md#decode-floor)). Either way each request gets the same tokens it would get alone, greedy or sampled with a seed. [batching.md](batching.md) has the design.

## Prefix cache

Chat clients and agents resend the whole conversation each turn. The server keeps the state of recent prompts, one per stream unless `--prefix-cache-entries` says fewer, each at its last multiple of `--prefill-chunk` tokens (2,048 by default). A request whose prompt starts with one of them prefills only the rest, batched or alone, and gets the same tokens and logits as without the cache. The usage object reports the reused tokens as `prompt_tokens_details.cached_tokens`.

Each entry belongs to a stream's KV cache, so a request that hits one runs in that stream. A new prompt takes a stream without an entry, else the one whose entry was used least recently, and an entry unused for `--prefix-cache-ttl` seconds is dropped. An entry keeps its stream's pages until the pool needs them for a running request. So up to `--streams` conversations, such as an agent and its parallel subagents, each keep their own history cached between turns. Prompts that share only a beginning, such as two conversations with the same system prompt, do not share an entry.

A request that runs alone speculates. When its entry was saved by a batched request, it lacks the draft's state: the request then prefills from the entry and decodes batched, and rebuilds the draft's state, a prefill of its whole prompt, only when its `max_tokens` leaves more than 1/16 of the position to generate (see [engine.md](engine.md#prefix-cache)). Each entry takes 172 MiB with the bundled draft and 204 MiB with DFlash 2, whatever the context, of which the 160 MiB of copies of the rings come from the pool while the entry lives.

With `--host-cache GiB`, every entry is also copied to page-locked host memory with its whole KV cache, and a request whose entry left the GPU restores it from there into the stream it takes, then prefills only the rest, with the same tokens and logits. A conversation that a subagent or another session pushed off the GPU then comes back in 0.135 s instead of 2.1 s at 32K tokens, and in 0.81 s instead of 39 s at 330K, on an RTX PRO 6000 Blackwell Workstation Edition. The memory is allocated at startup (9.7 s for 12 GiB). An entry takes 18 KiB per token with FP8 caches and the bundled draft (16 KiB with DFlash 2, 34 KiB with BF16 caches) plus about 0.2 GiB, conversations that share a start share its memory, and the least recently used entries go when it is full. The log reports each save and restore. On a two-socket host, run the server on the GPU's socket (`taskset` or `numactl`), or restores run at about half speed. [engine.md](engine.md#host-tier) has the design and the measurements.

## Restarts

On SIGTERM or Ctrl-C the server drains instead of stopping:

1. It logs the running and queued requests, refuses new ones with 503 and `Retry-After: 1`, and `/health` answers 503 `draining`, so the proxy in front sends new requests and their retries to the process that replaces this one. The Anthropic route answers 503 too, not its usual 529 for a full queue.
2. It finishes the running and queued requests, for up to `--drain-timeout` seconds.
3. It ends the requests still left with an error at their next step, and logs how many it finished and how many it cut.
4. With `--cache-dir`, it copies the conversations the streams hold into the host cache, writes every host cache entry to DIR, and exits.

At startup the server loads DIR's entries into the host cache, so the first turn of each saved conversation restores from host memory in place of a full prefill, with the same tokens and logits as a cold prefill. DIR holds one file per entry and a `manifest.json`. The manifest records what changes the saved bytes or their meaning: a hash of the checkpoint's `config.json` and of its weight files' names and sizes, the same for `--drafter`, the KV cache and weight formats, `--draft`, `--prefill-attention`, a hash of the kernels this build compiled, the chunk layout and the file format version. A directory whose manifest differs in any of them is discarded with a log line, so a build with other kernels starts empty. Each file is written under a temporary name and renamed, and the manifest is removed before a save and written after it, so a crash midway leaves nothing that loads.

On an RTX PRO 6000 Blackwell Workstation Edition with `--host-cache 12`, two conversations of 330K tokens filled 11.66 GiB of the host cache. SIGTERM to exit took 14.1 s, 8.8 s of it saving the files (1.42 GB/s, with `fsync`). The next server loaded them in 4.4 s, read from disk after the files were dropped from the page cache, after the 9.7 s that page-locking the pool takes. Its first turn of each conversation read 329,728 tokens from the cache and took 1.13 s and 0.65 s instead of 37 s, and the 64-token reply equaled the one before the restart. With `--drain-timeout 2`, a request in its reply was cut: its client got a 500 with the reason, and the server saved and exited.

`tests/drain.rs` checks a deploy end to end with the real binary: a running request finishes all of its 1,500 tokens after SIGTERM, while `/health` and a new request get 503 with `Retry-After`, the server saves and exits with status 0, and the next server loads the entry and gives the same request the same reply from 2,048 cached tokens. `host_cache_round_trips` in `crates/siphon-cuda/tests/prefix.rs` checks that entries read back from disk restore with the first token's logits, tokens and draft acceptance of a cold prefill, with the bundled draft and DFlash 2, that a manifest of another identity is discarded with its files, and that a directory without a manifest loads nothing.

## Known gaps

Found serving a coding agent from one RTX 5090: 144 requests, a median prompt of 41,281 tokens (largest 153,426) and a median reply of 151 tokens. The prefix cache served 90% of the tokens of prompts over 4,096.

- **Another request in a slot loses its conversation without `--host-cache`.** A device entry keeps only the slot's rings and conv states. The global KV cache stays in the slot, so any other sequence there overwrites it. With `--streams 1`, short side requests caused 6 full prefills of 77,421 to 150,445 tokens, at 5.7 to 12.5 s each. The side requests included the agent's own 539-token request and 21-token tests. The host tier, `--host-cache`, now restores such prompts in well under a second.
- **A prefill stalled every other stream.** In a test on an RTX PRO 6000 Blackwell with two streams of 1,048,576 positions, the running request delivered 2.2 tok/s while the second request's 325 s prefill ran, one batched step per prefill chunk. `--decode-floor` now runs more steps between the chunks (see [Concurrent requests](#concurrent-requests)).
- **Decode slows past 100,000 tokens.** Agent turns decoded at a median 534 tok/s with prompts of 32,000 to 100,000 tokens, and 417 tok/s above 100,000. Speculation kept a median of 3.09 tokens per round.

## Building on macOS

On macOS, `siphon-cuda` compiles to an empty crate and `siphon-serve` builds without its engine, so its tests run but it cannot serve. From a Mac, `scripts/remote.sh <ssh-host> <remote-dir> <command>` syncs the workspace to a Linux host and runs the command in its pixi environment. Set `NVCC` or `CUDA_HOME` to use another toolkit, and `SIPHON_CUDA_ARCH=sm_89` to build for an RTX 4090 (see [engine.md](engine.md#rtx-4090-sm_89)).
