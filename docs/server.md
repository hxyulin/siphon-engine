# siphon serve

`siphon serve` is an OpenAI-compatible HTTP server for Inkling 2 Mini. This page lists its options, routes and request fields. [engine.md](engine.md) covers how the engine works, and [clients.md](clients.md) sets up coding agents against the server.

```sh
SIPHON_API_KEY=<key> target/release/siphon serve <model-dir> --port 8000
```

## Startup

The server selects CUDA on Linux and MLX on Apple Silicon. Use `--backend cuda` or `--backend mlx` to make the selection explicit.

### CUDA

Before loading anything, the server plans the GPU memory from the config and the checkpoint headers. The global KV caches of every stream share one pool of pages ([Concurrent requests](#concurrent-requests)). Without `--context`, the pool takes the free memory less `--margin` (0.25 GiB by default) and what the weights and buffers need, up to `--streams` times the model's 1,048,576 per-request limit (each individual request stays capped at 1,048,576). With `--context`, it stops and prints the plan if that context does not fit. It prints the plan in two lines.

It then loads the model, prices each verify width on a short prompt, and serves. After warmup it compares the memory the process uses with the plan and prints both. It reads its own buffers from the CUDA memory pool that every allocation comes from, and the module images and CUDA graphs from the change in free memory. After each job it checks the pool's peak again and warns if the peak went past the plan.

`siphon plan <model-dir>` prints a GPU memory breakdown and compares the longest contexts across formats and drafts. It runs on macOS too. It reads the free memory from nvidia-smi, takes a `--free` override, or assumes an idle 32 GB RTX 5090.

Set `--weights` and `--kv` to plan for the formats you intend to serve. Both default to `fp8`. In `--interactive` mode, press `w` to toggle weight precision, `k` to cycle through the KV formats, and `d` to switch between the bundled draft and the directory supplied with `--drafter`. The arrow keys move the context in steps of 8K or 64K positions. Press `m` to return to the largest context that fits. Press `q` or Esc to print a `siphon serve` command and exit. Press Ctrl-C to cancel.

### MLX

Install the pinned MLX-LM environment as described in the [Apple Silicon quick start](../README.md#apple-silicon). `SIPHON_MLX_PYTHON` sets the Python executable and defaults to `python3`. The worker loads the checkpoint once and decodes one text sequence at a time, with no shared prefix cache or speculative decoding.

MLX supports `--host`, `--port`, `--name`, `--queue`, `--drain-timeout`, `--keepalive-after`, `--context`, `--prefill-chunk` and `--greedy`. The MLX context limit defaults to 8192 positions per request, including prompt and output. Choose a larger limit with `--context` if the Mac has enough memory. `siphon plan` estimates CUDA memory.

## Options

The table shows CUDA defaults and behavior. See [MLX startup](#mlx) for supported options and context limits on Apple Silicon.

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
| `--disk-cache GiB` | `0`, off | Enables an active L3 NVMe spillover tier in `--cache-dir` capped at that many GiB. Evicted RAM entries spill asynchronously using content-addressed deduplicated KV chunks, skipping clean re-evictions, short entries (`< 4,096` tokens), stale entries older than `--disk-cache-ttl`, per-spill `fsync`, and writes above a sustained wear-protection cap (`32 MiB/s`, `2 GiB` burst). Needs `--host-cache` and `--cache-dir`. |
| `--disk-cache-ttl SECONDS` | `3600` | Drops evicted RAM entries idle longer than this many seconds instead of writing them to `--disk-cache` (`0` disables the spill TTL). |
| `--drain-timeout SECONDS` | `600` | How long a drain on SIGTERM waits for the running and queued requests before it ends them. |
| `--keepalive-after SECONDS` | `20` | How long a reply that is not streamed may stay silent before the server commits to status 200 and sends keep-alive spaces; 0 turns this off. See [Keep-alive](#keep-alive). |
| `--prefill-attention auto\|bf16\|fp8` | `auto` | Prefill's global attention over an FP8 cache. `auto` runs BF16 up to 16,384 keys and FP8 tensor cores past them. |
| `--greedy` | off | Decodes every request greedily, whatever its sampling fields ask. |
| `--vision` | off | Loads the vision tower and takes images in requests. See [Images](#images). |
| `--max-image-pixels N` | `2073600` | Shrinks a larger image to N pixels, keeping its aspect ratio. The default is 1920 x 1080. |
| `--max-images N` | `16` | Images one request may hold. |

With `SIPHON_API_KEY` set, every route but `/health`, `/version` and `/metrics` needs `Authorization: Bearer <key>` or `x-api-key: <key>`.

Each request gets an ID. The server takes the client's `x-request-id`, or else `x-client-request-id`, when it holds 1 to 128 printable ASCII characters without spaces, and otherwise makes one such as `req-5e0c1f7a9b3d2e64`. Every response, errors included, returns the ID in `x-request-id`. The request's [log](#logs) line carries it as `id=`, and the `job:` lines start with it and the response's own id, so a proxy's request log matches them. The `job:` lines of a request with `n` choices are labelled `choice i`.

## HTTP compression

A request body may be compressed with `Content-Encoding: gzip`. Build with `pixi run cargo build --release -p siphon-cli --features http-zstd` to also support zstd for HTTP. Unsupported request encodings get a 415. Whole responses use the supported encoding negotiated via `Accept-Encoding`. Streams are sent uncompressed so each event reaches the client as the server sends it.

A body may hold 32 bytes per position of the context, at least 2 MiB, plus 20 MiB per image with `--vision`. The limit applies to the decompressed bytes, so a small compressed body that inflates past it gets a 413, as an uncompressed one does.

## Routes

| Route | Notes |
| --- | --- |
| `POST /v1/chat/completions` | Messages rendered as the checkpoint's chat template renders them. Streams with `stream`, and `stream_options.include_usage` adds a usage chunk. The model's thinking comes back as `reasoning_content`. |
| `POST /v1/completions` | A text prompt or token ids, streaming or not. The text includes any special tokens generated. |
| `POST /v1/messages` | Anthropic's Messages API, streaming or not, for Claude Code and Anthropic's SDKs. See [Anthropic Messages API](#anthropic-messages-api). |
| `POST /v1/messages/count_tokens` | `{"input_tokens": n}`: the tokens of the prompt that `/v1/messages` would run for the same body. |
| `POST /v1/responses` | OpenAI's Responses API, streaming or not, for the Codex CLI and OpenAI's SDKs. It stores nothing. See [Responses API](#responses-api). |
| `POST /tokenize`, `POST /detokenize` | vLLM's shape, also at `/v1/tokenize` and `/v1/detokenize`: `{"prompt"}` or `{"messages", "tools"}` to `{"tokens", "count", "max_model_len"}`, and `{"tokens"}` to `{"prompt"}`. With `--vision`, the messages' images count as the chat route counts them. |
| `POST /v1/token/encode`, `POST /v1/token/decode` | TabbyAPI's shape: `{"text"}`, a string or messages, to `{"tokens", "length"}`, and `{"tokens"}` to `{"text"}`. |
| `GET /v1/models`, `GET /v1/models/{id}` | The served model, with `max_model_len`. |
| `GET /health`, `GET /version` | No API key needed. |
| `GET /metrics` | Prometheus metrics under vLLM's names, so tools written for vLLM read them. No API key needed. See [Metrics](#metrics). |

## Chat request fields

- `reasoning_effort`: a name such as `low`, or 0 to 0.99. The default is 0.9. Without it, `chat_template_kwargs.reasoning_effort` counts. `none` and `chat_template_kwargs.enable_thinking: false` turn thinking off, and a request without tools then gets no thinking block (see [Thinking off](#thinking-off)). A number is passed to the template as it is, and at effort 0 the model still writes a short thinking block.
- `tools` and `tool_choice`: `auto`, `none`, `required`, or `{"type": "function", "function": {"name": ...}}`. A named function, or `required` with one tool, ends the prompt with that tool's call opened. `required` with several tools opens a call without a name, and the model names the tool. A forced call skips the thinking block. The reply ends after the named call, or at the first text after the calls `required` forced. `parallel_tool_calls: false` ends the reply after its first call.
- `stop`: a string or an array of strings. Stop strings match the reply's content, not its thinking or tool calls. While streaming, text that could still begin a stop string is held back. The reply is cut before the match and finishes with `stop`.
- `max_tokens` or `max_completion_tokens` past the context is cut to the positions left after the prompt, and the reply then finishes with `length`. A client's token count can differ from the server's, so a cap that a client sized to the context can overshoot it. vLLM refuses such a request. A prompt that fills the context alone still gets a 400, with the error code `context_length_exceeded`.
- A stream sends each choice's role chunk with the first token, as vLLM does, so a proxy that times the first chunk measures the prefill. During a long prefill, a keep-alive comment every 15 seconds holds the connection open.
- A tool call's `function.arguments` holds its argument object as the model wrote it. A stream sends each call as OpenAI does: first a chunk with the call's `index`, `id`, `type` and `function.name`, as soon as the model has written the name and opened the arguments, then chunks with pieces of `function.arguments` as the model writes them. The pieces join to the arguments the same reply has unstreamed. A call payload of another shape, such as one with `args` before `name`, arrives in two chunks when its block ends.
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

The reply holds `thinking` blocks with an empty `signature`, then `text` and `tool_use` blocks in the order the model wrote them. `stop_reason` is `end_turn`, `max_tokens`, `stop_sequence` (with `stop_sequence` set) or `tool_use`. In `usage`, `input_tokens` leaves out the prompt tokens read from the prefix cache, which `cache_read_input_tokens` counts, and `cache_creation_input_tokens` is 0. A stream sends Anthropic's events, from `message_start` to `message_stop`. `message_start` waits for the reply's first content block, as the chat route's role chunk waits for the first token, so a proxy that times the first event measures the prefill. Before it, a stream sends an SSE comment after 5 seconds without bytes, such as during a long prefill, because Anthropic's protocol puts `message_start` first. After it, the stream sends a `ping` event instead. A `tool_use` block starts as soon as the model has written the call's name and opened its arguments, and its input follows in `input_json_delta` pieces as the model writes them.

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

## Responses API

`POST /v1/responses` takes OpenAI's Responses request and answers in its shape. The server renders the request with the same chat template as `/v1/chat/completions`, and runs it on the same engine.

The server stores no responses. A request carries the whole conversation in `input`, as Codex sends it with `store: false`. `store` is ignored. `previous_response_id`, `conversation`, `item_reference` input items and `background: true` get a 400.

The request maps onto the chat route as follows:

- `instructions` becomes the first system message.
- `input` is a string, which becomes a user message, or a list of items:
  - `message` items, also without a `type`, with the roles `user`, `system`, `developer` and `assistant`. Their content is a string or a list of parts: `input_text` and `output_text` parts are text. With `--vision`, `input_image` parts with a `data:` URL reach the prompt as images (see [Images](#images)). Without it, an image becomes `[image omitted: the model cannot see images]`, since a 400 would end a Codex session. An `input_file` part always becomes `[document omitted: the model cannot read documents]`.
  - `reasoning`, `function_call` and assistant `message` items that follow each other become one assistant message, as the model wrote them in one turn: the reasoning as its past thinking, then its text, then its tool calls. A reasoning item's thinking comes from its `reasoning_text` content, else from `encrypted_content`, else from its summary.
  - A `function_call_output` becomes a tool message. Its `output` is a string or a list of parts, as above.
  - Other items, such as hosted tool calls and their outputs, are left out.
- `tools`: `function` tools in the flat Responses shape. Tools of other types, such as `web_search`, `local_shell`, custom tools and Codex's `namespace` tools, are left out.
- `tool_choice` (`auto`, `none`, `required` or `{"type": "function", "name": ...}`) and `parallel_tool_calls` work as on the chat route.
- `reasoning.effort` sets the effort as `reasoning_effort` does, and `none` turns thinking off (see [Thinking off](#thinking-off)).
- `max_output_tokens`, `temperature`, `top_p` and `stop` work as `max_tokens`, `temperature`, `top_p` and `stop` do on the chat route. `top_k`, `min_p` and `seed` work too.
- `text.format` other than `text` gets a 400, as `response_format` does.

The server ignores `store`, `prompt_cache_key`, `client_metadata`, `metadata`, `truncation`, `service_tier`, `user`, `text.verbosity`, `include` values other than `reasoning.encrypted_content`, and unknown fields.

The response has `output` items in the order the model wrote them: `reasoning`, `message` with one `output_text` part, and `function_call` with `call_id`, `name` and `arguments`. `status` is `completed`, or `incomplete` with `incomplete_details.reason` `max_output_tokens` when the reply ran out of tokens. In `usage`, `input_tokens` counts the whole prompt and `input_tokens_details.cached_tokens` the tokens read from the prefix cache. `output_tokens_details.reasoning_tokens` counts the thinking.

The thinking is public. Each reasoning item holds the whole text in `content`, as one `reasoning_text` part, the way open-weight servers return it. When the request sets `reasoning.summary`, as Codex does, the item's `summary` holds the same text, since Codex shows summaries by default. With `include: ["reasoning.encrypted_content"]`, `encrypted_content` holds the text in plain base64: it is not encrypted, and any client can decode it. A reasoning item sent back on the next turn renders as the thinking the model wrote, so the prompt matches what the chat route renders for the same conversation.

A stream sends OpenAI's typed events, each with `sequence_number`: `response.created` and `response.in_progress`, then for each item `response.output_item.added`, its content's events and `response.output_item.done`, then `response.completed`, `response.incomplete` or `response.failed`. A reasoning item streams `response.reasoning_text.delta` and `.done`, and with a summary also `response.reasoning_summary_part.added`, `response.reasoning_summary_text.delta`, `.done` and `response.reasoning_summary_part.done`. A message streams `response.content_part.added`, `response.output_text.delta`, `.done` and `response.content_part.done`. A function call item is added as soon as the model has written the call's name and opened its arguments, and streams them in `response.function_call_arguments.delta` pieces as the model writes them, then `.done`. `response.created` waits for the reply's first item, as the chat route's role chunk waits for the first token, so a proxy that times the first event measures the prefill. Before it and after it, a stream quiet for 15 seconds sends an SSE comment.

Errors take OpenAI's shape. A full queue answers 503, as on the chat route. A failure during a stream sends `response.failed` with `error.code` `server_error`, after `response.created` and `response.in_progress` if the stream had not started.

### Codex

[clients.md](clients.md#codex) has the Codex setup, with Pi, OpenCode and DeepSeek Harness.

## Images

With `--vision`, the server loads the vision tower beside the model and takes images in the chat, Messages and Responses routes:

- `/v1/chat/completions`: content parts `{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}`, also with `image_url` as a plain string, and `input_image` parts. `detail` is ignored.
- `/v1/messages`: `image` blocks with `{"type": "base64", "media_type", "data"}` sources, in user messages and inside `tool_result` blocks.
- `/v1/responses`: `input_image` parts with a `data:` URL in `image_url`, in user messages and in `function_call_output` parts.

Images must be inline: PNG, JPEG, WebP or GIF (its first frame) in base64. The server does not fetch URLs, so an `http` or `https` image URL gets a 400 that says so. Images are allowed in user and tool messages only, as the chat template allows them.

Each image goes through the reference preprocessing ([vision.md](vision.md)). An image of more than `--max-image-pixels` shrinks to fit with PIL's bicubic filter, keeping its aspect ratio. Its sides are then padded up to multiples of 40 pixels, and each 40 x 40 square becomes one image token: 676 tokens for a 1024 x 1024 image and 1,296 for 1920 x 1080. The prompt holds `<|content_image|>`, one `<|unused_200054|>` placeholder per image token and `<|end_message|>`, after the message's header, where the chat template puts an image. `usage.prompt_tokens` and the tokenizer routes count the placeholders.

A request gets a 400 when it holds more than `--max-images` images, when an image is not one of the four formats or does not decode, or when an image has more than 50 million pixels before shrinking. An image that still needs more image tokens after shrinking than a square of `--max-image-pixels` with one more row and column of tokens, 1,369 by default, gets a 400 too; only very narrow or wide images do. The request body may grow by 20 MiB per image allowed.

The tower runs on the engine thread right before the prefill chunk that first reaches an image, and its time counts as prefill for `--decode-floor`. Prefill writes the image's rows over the placeholders' embeddings after `embed_norm`, as Transformers does, and positions stay sequential. An image that crosses a prefill chunk boundary is encoded once and its rows serve both chunks. The request's [log](#logs) line counts the images in its `engine` field, as in `2 images: 1 encoded in 38 ms, 1 from the cache`, and its `images` and `image_ms` fields, and its `prefill_ms` leaves the encodes out, as [`timings`](#request-timings) does.

Two prompts with different images have the same tokens, so the [prefix cache](#prefix-cache) does not match prompts by their tokens alone. Its device entries, its host tier and the files in `--cache-dir` hold each prompt's key: its tokens with each image's placeholders replaced by ids above the vocabulary, taken from the SHA-256 of the image file and from `--max-image-pixels`. A prompt whose images differ from an entry's below the entry's boundary does not match it. A follow-up turn of a conversation with images reads every image below its boundary from the cache, and the tower encodes only the images past it.

The speculative draft reads the target's hidden states at every prompt position, image positions included, so it sees what the target sees. Verify decides every token, so images do not change speculation's output either way. Acceptance on replies about images is not measured yet.

`siphon plan --vision [--max-image-pixels N]` adds the tower's weights, its buffers for the largest image and the image rows of one prefill chunk to the plan. On an idle 32 GB RTX 5090, `--vision` at the defaults takes 0.69 GiB and cuts the longest context with FP8 weights and caches from 657,408 to 617,472 positions, with the default 0.25 GiB margin. Without `--vision` the plan and the server are as before.

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

A 400's message starts with the field's path, quotes the value with its JSON type, and says what the server accepts. OpenAI-shaped errors also carry the path as `param`; Anthropic's error shape has no such field, so its message alone holds it. A number or boolean sent as a string gets advice to send it unquoted:

```json
{"error": {"message": "reasoning_effort: got \"0.9\" (a string); send 0.9 as a JSON number, not a string. Accepted: a number from 0 to 0.99, or one of \"none\", \"minimal\", \"low\", \"medium\", \"high\", \"xhigh\" or \"max\"", "type": "invalid_request_error", "param": "reasoning_effort", "code": null}}
```

An error about a Messages or Responses request names the field of that request, such as `messages[2].content[1]` or `input[4]`, not of the chat messages the server translates it to. An image sent without `--vision` gets an error that names the flag on the chat route; the Messages and Responses routes replace it with a note to the model, since a 400 would end a Claude Code or Codex session.

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

Each completion, chat, Messages and Responses response carries a `timings` object with the field names of llama.cpp's server, so tools written for llama.cpp read them. Siphon adds its queue time, time to first token and speculation:

| Field | Value |
| --- | --- |
| `cache_n` | Prompt tokens read from the prefix cache. |
| `prompt_n` | Prompt tokens prefilled: the prompt less `cache_n`. |
| `prompt_ms`, `prompt_per_token_ms`, `prompt_per_second` | Time from the engine starting the request to its first token, per `prompt_n` token, and `prompt_n` per second. |
| `predicted_n` | Generated tokens in the reply, as `usage` counts them. |
| `predicted_ms`, `predicted_per_token_ms`, `predicted_per_second` | Time from the first token to the end, per `predicted_n` token, and `predicted_n` per second. |
| `queue_ms` | Time from queueing to the engine starting the request. |
| `ttft_ms` | Time from queueing to the first token: `queue_ms` plus `image_ms` plus `prompt_ms`. |
| `draft_rounds`, `tokens_per_round` | Speculation rounds and the tokens each delivered on average, when the request speculated. A plain step under `--fallback` counts as a round of one token. |
| `image_n`, `image_cache_n`, `image_ms` | For a prompt with images: the images the vision tower encoded, the images inside the prefix cache hit, which it did not encode, and the GPU time of the encodes. |

The times are taken when the engine's events reach the HTTP layer. The queue starts once the server has parsed and tokenized the request, so parsing does not count as prefill. The prompt time includes any prefix cache restore. It leaves out image encodes, which the engine times on the GPU with CUDA events. A request with `n` choices counts from its earliest start and first token to its last end, and sums the choices' tokens, rounds and encodes.

A response that is not streamed has `timings` at the top level and a `Server-Timing` header, `queue;dur=…, prefill;dur=…, decode;dur=…` in milliseconds, with `image;dur=…` after the queue for a prompt with images. A stream has sent its headers before the times are known, so its last chunk holds `timings`: in a chat or completion stream, the chunk with the last `finish_reason`, or the usage chunk when `stream_options.include_usage` asks for one, in a Messages stream, the `message_delta` event, and in a Responses stream, the response of its last event. OpenAI's and Anthropic's SDKs ignore fields they do not know.

The request's [log](#logs) line repeats the main numbers.

## Logs

The server logs to stderr, one line per event, each after a UTC timestamp to the millisecond. At startup it logs its version and the commit it was built from (`SIPHON_COMMIT` at build time, else `git rev-parse`; `scripts/remote.sh` and the Docker build set it), the memory plan and what it loaded:

```text
2026-11-27T10:14:58.204Z siphon-serve 0.1.0 (commit 060d897a1b2c) starting with /home/u/models/inkling-2-mini
```

Each HTTP request ends with one `request` line of `key=value` fields, quoted when they hold a space. It comes after the response's last byte and after the request's jobs have ended, so rejected requests, unknown routes and clients that leave get one too. Successful `GET /health`, `GET /metrics` and `GET /v1/models` requests are left out, since proxies, scrapers and gateways poll them.

```text
2026-11-27T10:15:02.412Z request id=req-5fc02f989d827497 route="POST /v1/chat/completions" status=200 ms=412 reply=chatcmpl-6b095471000000 stream=true effort=0.9 finish=stop prompt=31204 cached=30720 cache=slot queue_ms=0 prefill_ms=48 output=215 tok_s=634.3 per_round=3.79 engine="216 speculated, 0 batched steps, last speculated 214 tokens in 0.34 s (631.4 tok/s, 3.79 tokens per round, ...)"
2026-11-27T10:15:03.020Z request id=req-0a5e4c1d9b7f2e31 route="POST /v1/chat/completions" status=400 ms=1 error="reasoning_effort: got \"0.9\" (a string); send 0.9 as a JSON number, not a string. Accepted: a number from 0 to 0.99, or one of \"none\", \"minimal\", \"low\", \"medium\", \"high\", \"xhigh\" or \"max\""
2026-11-27T10:15:09.733Z request id=req-77c3a0e2f1d4b569 route="POST /v1/messages" status=200 ms=6120 reply=msg_6b0954710003 stream=true effort=0.9 finish=abort prompt=1840 cached=0 queue_ms=0 prefill_ms=95 output=57 tok_s=402.1 per_round=3.10 client=disconnected
```

| Field | Value |
| --- | --- |
| `id`, `route`, `status`, `ms` | The request ID, method and path, status, and milliseconds from arrival to the line. `status=499` means the client left before the response started. |
| `reply`, `stream`, `effort` | The response's id, whether it streamed, and the thinking effort, `off` when thinking is off. |
| `finish` | How each choice's job ended: `stop` (the model's end or a stop string or token), `length`, `abort` (the client left) or `error`. |
| `prompt`, `cached`, `cache` | Prompt tokens, those read from the prefix cache, and where they came from: `slot` (an entry on the GPU), `host` (the host cache) or `copied` (another choice's prefill). |
| `queue_ms`, `prefill_ms`, `output`, `tok_s`, `per_round` | As [`timings`](#request-timings): queue and prefill time, generated tokens, decode tokens per second, and tokens per speculation round. |
| `images`, `image_ms` | Images the vision tower encoded, and their GPU time. |
| `engine` | The engine's summary: tokens from speculation and from batched steps, swaps to the host cache, and the last speculation episode. |
| `keepalive`, `client`, `error` | `keepalive=committed` when [keep-alive](#keep-alive) sent status 200 before the reply was ready, while `status` is the reply's own; `client=disconnected` when the client left before the end, with `output` counting the tokens it had been sent; the error message of a response that is not a success, or of a job that failed mid-stream. |

Other lines report engine events, one line each:

```text
2026-11-27T10:16:40.118Z job: req-1b2c3d4e5f607182 msg_6b0954710007: prefix cache hit, 438272 of 440017 prompt tokens from the host tier
2026-11-27T10:16:40.119Z job: req-9a8b7c6d5e4f3021 chatcmpl-6b095471000009: prefix cache miss, prefilling all 97311 prompt tokens
2026-11-27T10:16:52.640Z prefix cache: dropped the entry of 96256 tokens in slot 2, unused for 412 s: every free slot holds a cached prompt, and it was the least recently used
2026-11-27T10:16:52.641Z prefix cache: dropped the entry of 36864 tokens in slot 1, unused for 75 s: the KV pool needs 5784 free pages, and it held 1032
2026-11-27T10:17:30.002Z prefix cache: dropped the entry of 12288 tokens in slot 3, unused for 640 s: expired, unused past the 600 s time to live
2026-11-27T10:18:05.871Z host cache: evicted an entry of 438272 tokens, unused for 1832 s, to save a new entry: the host cache is full (1021 of 1024 chunks in use, 3 free, 214 needed)
2026-11-27T10:18:06.120Z host cache: saving 98304 tokens with the draft's state: 167 MiB, 2 chunks copied, 46 shared; 181 of 1024 chunks in use by 3 entries
2026-11-27T10:18:09.477Z job: req-77c3a0e2f1d4b569 msg_6b0954710003: client left after 57 of 4096 tokens in 0.14 s (1840 prompt tokens, 0 cached): 60 speculated, 0 batched steps, ...
```

The prefix cache lookup gets a line for prompts of 16,384 tokens or more, whose prefill takes seconds; every request's line has `cached` and `cache`. A prefix cache entry that its own conversation's next turn replaces is not logged as dropped, nor is a host entry that a longer one of the same conversation replaces. The `job: ... client left` line comes from the engine when it ends a job whose request has already logged its line.

## Keep-alive

A proxy may drop a request that sends no bytes for a while: Cloudflare answers 524 after 100 s. A 512K-token cold prefill takes about 65 s on an RTX 5090, a request may first wait behind others, and a reply of 4,096 tokens at 40 tok/s takes about 100 s, so no response stays silent for long:

- Streams send bytes at least every 15 s, also before the first token: chat and completion streams an SSE comment after 15 s without an event, Messages streams a comment before `message_start` and a `ping` after it every 5 s, and Responses streams a comment every 15 s.
- A reply that is not streamed, on `/v1/chat/completions`, `/v1/completions`, `/v1/messages` and `/v1/responses`, goes out as before when it is ready within `--keepalive-after` seconds (20 by default). Past that, the server sends status 200, `Content-Type: application/json` and a chunked body: a space at once and every 15 s, then the reply's JSON. JSON parsers skip leading whitespace, as `serde_json`, Python's `json` and JavaScript's `JSON.parse` do. Such a response is never compressed, since a compressor would hold the spaces back, and it has no `Server-Timing` header, whose times are not known when the headers go out.

The trade-off: a reply that fails after the 200 has gone out keeps status 200, with the route's error JSON (`{"error": ...}`, or Anthropic's `{"type": "error", ...}`) as its body, so a client that trusts the status alone sees a success. The request's log line records the real status and error, with `keepalive=committed`. A client that leaves during the wait cancels the request's jobs, as with a stream. Set `--keepalive-after 0` for clients that need the real status and do not sit behind such a proxy.

## Concurrent requests

With `--streams N`, up to N requests run at once and up to `--queue` more wait. Their global KV caches and sliding rings share one pool of 2 MiB pages, which each takes as its sequence grows and keeps after it ends until the pool needs them, so one long conversation and several short requests run together, and one request alone may fill the pool. `--streams` costs little past the first: on a 32 GB card with FP8 weights and caches, with 30.45 GiB free and the default margin, the largest single request takes 661,504 positions with `--streams 1`, 655,360 with 4 and 651,264 with 8. Before the pool, `--streams 1` took 657,408 and 4 streams 161,792 each. [batching.md](batching.md#memory-one-pool-of-pages) has the design.

A request starts when the pool has room for its prompt and 2,048 positions more. A request that runs alone then takes pages as its reply grows, 2,048 positions at a time, from the prompts that the prefix cache keeps for idle streams if it must; its token limit only sets where it stops. The shortest prompts start first, unless a request has waited 20 s. When the running requests outgrow the pool, prompts that the prefix cache keeps for idle streams go first. With `--host-cache`, requests that still do not fit then take turns: one is swapped out to host memory, the others run, and it comes back, each running at least nine times as long as its swap takes. A request with images that is swapped out mid-prefill encodes the image its next chunk needs again when it returns. Without `--host-cache`, a request that starts beside others starts only when the pool holds every position it may reach.

More streams serve more in total and less per stream. On an RTX PRO 6000 Blackwell Max-Q, 32 streams at 10,674 tokens of context ran 1,138 tok/s in all against 661 for 8 ([batching.md](batching.md#16-and-32-streams-on-an-rtx-pro-6000-blackwell-max-q)). With `--streams 32 --context 16384`, 32 concurrent requests of 512 tokens came back at 1,198 tok/s in all.

A request that arrives at an idle server speculates. When another arrives, the first stops speculating after its round, and every running request decodes in a batched step of one row each. When one request is left and nothing waits, it speculates again. A new request's prompt prefills in chunks of `--prefill-chunk` tokens (2,048 by default) between the batch's steps. After each chunk the others run as many steps as give each at least `--decode-floor` tokens per second: with the default of 20, a 100,000-token prefill on an RTX 5090 took 8% longer and the decoding request got 22.4 tok/s instead of 6.1 ([batching.md](batching.md#decode-floor)). Either way each request gets the same tokens it would get alone, greedy or sampled with a seed. [batching.md](batching.md) has the design.

## Prefix cache

Chat clients and agents resend the whole conversation each turn. The server keeps the state of recent prompts, one per stream unless `--prefix-cache-entries` says fewer, each at its last multiple of `--prefill-chunk` tokens (2,048 by default). A request whose prompt starts with one of them prefills only the rest, batched or alone, and gets the same tokens and logits as without the cache. The usage object reports the reused tokens as `prompt_tokens_details.cached_tokens`.

Each entry belongs to a stream's KV cache, so a request that hits one runs in that stream. A new prompt takes a stream without an entry, else the one whose entry was used least recently, and an entry unused for `--prefix-cache-ttl` seconds is dropped. An entry keeps its stream's pages until the pool needs them for a running request. So up to `--streams` conversations, such as an agent and its parallel subagents, each keep their own history cached between turns. Prompts that share only a beginning, such as two conversations with the same system prompt, do not share an entry.

A request that runs alone speculates. When its entry was saved by a batched request, or another request that speculated took the draft since, it lacks the draft's state: the request then prefills from the entry and decodes batched. A rebuild gives the draft its state back by prefilling the request's tokens again in a free stream, one chunk of `--prefill-chunk` positions at a time between batched steps, so a request that arrives meanwhile waits for one chunk at most. The rebuild starts once the request's batched steps have taken about as long as the rebuild will, whatever its `max_tokens`: agents send caps of 16,384 tokens and more and reply with a few hundred. With two or more streams, once the server has been idle for half a second, the same rebuild runs for the longest cached prompt that lacks the draft's state, so that its conversation's next turn speculates from its first token (see [engine.md](engine.md#prefix-cache)). Each entry takes 172 MiB with the bundled draft and 204 MiB with DFlash 2, whatever the context, of which the copies of the rings come from the pool while the entry lives: 80 MiB with a quantized cache, 160 MiB with `--kv bf16`.

With `--host-cache GiB`, every entry is also copied to page-locked host memory with its whole KV cache, and a request whose entry left the GPU restores it from there into the stream it takes, then prefills only the rest, with the same tokens and logits. A conversation that a subagent or another session pushed off the GPU then comes back in 0.135 s instead of 2.1 s at 32K tokens, and in 0.81 s instead of 39 s at 330K, on an RTX PRO 6000 Blackwell Workstation Edition. The memory is allocated at startup (9.7 s for 12 GiB). An entry takes 18 KiB per token with FP8 caches and the bundled draft (16 KiB with DFlash 2, 34 KiB with BF16 caches) plus about 0.2 GiB, conversations that share a start share its memory, and the least recently used entries go when it is full. The log reports each save and restore. On a two-socket host, run the server on the GPU's socket (`taskset` or `numactl`), or restores run at about half speed. [engine.md](engine.md#host-tier) has the design and the measurements.

Adding `--disk-cache GiB` alongside `--host-cache` and `--cache-dir` turns `--cache-dir` into an active L3 NVMe tier. Entries remain write-back (never write-through): active conversations replace their own shorter prefixes in RAM without touching disk, and only LRU-evicted RAM entries of at least 4,096 tokens spill asynchronously to disk. KV chunks are stored once under a 128-bit content hash (`chunk-*.bin`) and shared across entries by reference count alongside per-entry local state (`local-*.bin`), clean entries promoted from disk and later re-evicted write zero bytes, and a sustained token-bucket cap (`32 MiB/s` with `2 GiB` burst) plus zero per-spill `fsync` prevents SSD thrashing. On a hit, missing chunks promote into pinned RAM and restore to the GPU (`tier=disk` in the log); on startup, the newest entries pre-warm RAM while the rest stay indexed on disk.

## Restarts

On SIGTERM or Ctrl-C the server drains instead of stopping:

1. It logs the running and queued requests, refuses new ones with 503 and `Retry-After: 1`, and `/health` answers 503 `draining`, so the proxy in front sends new requests and their retries to the process that replaces this one. The Anthropic route answers 503 too, not its usual 529 for a full queue.
2. It finishes the running and queued requests, for up to `--drain-timeout` seconds.
3. It ends the requests still left with an error at their next step, and logs how many it finished and how many it cut.
4. With `--cache-dir`, it copies the conversations the streams hold into the host cache, writes every host cache entry to DIR, and exits.

At startup the server loads DIR's entries into the host cache, so the first turn of each saved conversation restores from host memory in place of a full prefill, with the same tokens and logits as a cold prefill. DIR holds one file per entry and a `manifest.json`. The manifest records what changes the saved bytes or their meaning: a hash of the checkpoint's `config.json` and of its weight files' names and sizes, the same for `--drafter`, the KV cache and weight formats, `--draft`, `--prefill-attention`, a hash of the kernels this build compiled, the chunk layout and the file format version. A directory whose manifest differs in any of them is discarded with a log line, so a build with other kernels starts empty. Each file is written under a temporary name and renamed, and the manifest is removed before a save and written after it, so a crash midway leaves nothing that loads.

On an RTX PRO 6000 Blackwell Workstation Edition with `--host-cache 12`, two conversations of 330K tokens filled 11.66 GiB of the host cache. SIGTERM to exit took 14.1 s, 8.8 s of it saving the files (1.42 GB/s, with `fsync`). The next server loaded them in 4.4 s, read from disk after the files were dropped from the page cache, after the 9.7 s that page-locking the pool takes. Its first turn of each conversation read 329,728 tokens from the cache and took 1.13 s and 0.65 s instead of 37 s, and the 64-token reply equaled the one before the restart. With `--drain-timeout 2`, a request in its reply was cut: its client got a 500 with the reason, and the server saved and exited.

`tests/drain.rs` checks a deploy end to end with the real binary: a running request finishes all of its 1,500 tokens after SIGTERM, while `/health` and a new request get 503 with `Retry-After`, the server saves and exits with status 0, and the next server loads the entry and gives the same request the same reply from 2,048 cached tokens. `host_cache_round_trips` in `crates/siphon-cuda/tests/prefix.rs` checks that entries read back from disk restore with the first token's logits, tokens and draft acceptance of a cold prefill, with the bundled draft and DFlash 2, that a manifest of another identity is discarded with its files, and that a directory without a manifest loads nothing.

## Serving on an RTX 4090

Measured with the `sm_89` build ([engine.md](engine.md#rtx-4090-sm_89)), the Inkling 2 Mini Fast NVFP4 checkpoint with its tuned bundled draft, FP8 weights, greedy decoding at effort 0, and 23.13 GiB free after the CUDA context. Context is the limit on this card: the weights take 16.8 GiB, so the KV pool gets about 4 GiB.

For one user, such as a coding agent:

```sh
siphon serve <model-dir> --streams 2 --kv mixed --host-cache 12 --cache-dir <dir>
```

Two streams let the agent's background requests run beside its conversation instead of taking its only prefix cache entry, for 6,144 positions less than one stream. The mixed cache plans 325,632 positions at about FP8's quality. The host cache restores a conversation that another request pushed off the GPU, and `--cache-dir` keeps it across restarts.

For the most context in one stream, at NVFP4's lower quality and without the host cache, which needs chunks of 2048:

```sh
siphon serve <model-dir> --kv nvfp4 --drafter <dflash2-dir> --prefill-chunk 1024 --margin 0.1
```

For 2 to 4 users:

```sh
siphon serve <model-dir> --streams 4 --kv mixed --host-cache 12
```

The four streams share a pool of 323,584 positions.

The longest context of one stream, planned and filled with one prompt:

| Configuration | Positions planned | Prompt | First token | Decode | Peak / planned |
| --- | ---: | ---: | ---: | ---: | --- |
| FP8 cache | 235,520 | 225,901 | 41.2 s | 184 tok/s | 23,330 / 23,419 MiB |
| Mixed cache | 331,776 | 316,494 | 69.3 s | 158 tok/s | 23,318 / 23,407 MiB |
| NVFP4 cache | 385,024 | 367,616 | 87.9 s | 156 tok/s | 23,322 / 23,411 MiB |
| NVFP4, DFlash 2, chunks of 1024, margin 0.1 | 516,096 | 478,195 | 138.5 s | 207 tok/s | 23,490 / 23,564 MiB |

The prompts, times and peaks were measured before the FP8 sliding rings of a quantized cache shared pages, on plans 8,192 to 18,432 positions shorter. The positions planned now have not been filled.

Each option's effect on the planned positions with the NVFP4 cache, against 385,024 for the bundled draft:

| Option | Bundled draft | DFlash 2 | Cost |
| --- | ---: | ---: | --- |
| none | 385,024 | 450,560 | |
| `--prefill-chunk 1024` | 421,888 | 497,664 | First token 8 to 11% later; no `--host-cache` |
| `--margin 0.1` | 399,360 | 468,992 | 0.15 GiB less headroom; the peak stayed 74 MiB under the plan |
| both | 436,224 | 516,096 | |
| `--output-projection fp8` | 370,688 | 432,128 | Adds an FP8 copy of the output projections |
| `--weights bf16` | 389,120 | 458,752 | Plain decode about 27% slower |

Time to first token on one stream: 0.94 s at 8,221 tokens, 3.77 s at 32,483 and 14.5 s at 103,597 with the FP8 cache, within 2% of that with the NVFP4 cache; 1.03, 4.17 and 15.8 s with chunks of 1024. A prefix cache hit at 155,648 tokens answered in 0.49 s, a restore from the host cache in 0.72 s (2,907 MiB in 157 ms) and a restore after a restart in 1.09 s, against 25.5 s cold.

Decode on one stream, tok/s, with the same replies for both drafts:

| Prompt | Bundled draft | DFlash 2 |
| --- | ---: | ---: |
| Code | 349 | 390 |
| JSON | 411 | 442 |
| Prose | 235 to 249 | 252 to 263 |
| 36,184-token prompt | 308 | 283 |
| 103,620-token prompt | 240 | 222 |

DFlash 2 is 5 to 12% faster on short prompts and 8% slower on long ones, and it leaves room for 65,536 more NVFP4 positions. On the 103,620-token prompt the mixed and NVFP4 caches decoded 246 and 234 tok/s against FP8's 240. Their replies differ from FP8's, so speculation accepts other tokens and the rates are not a like-for-like comparison.

Concurrent requests of 512 tokens each: 2 on `--streams 2` ran 262 tok/s in all, 134 each; 4 on `--streams 4` ran 341 tok/s in all, 87 each, with the FP8 or NVFP4 cache. While a 100,000-token prompt prefilled, the default `--decode-floor 20` gave a decoding request 20.8 tok/s and made the prefill 7% longer; a floor of 40 gave 40 tok/s for 23%.

When requests outgrow the pool, they take turns through the host cache: two prompts of 112K tokens on 223,232 positions each returned 1,500 tokens, and four of 67K to 75K each answered correctly, with swaps of 16 to 25 GB/s. `--host-cache 16` left 11.5 GB of 31 GB of RAM free; size it to the RAM other processes need.

## Known gaps

Found serving a coding agent from one RTX 5090: 144 requests, a median prompt of 41,281 tokens (largest 153,426) and a median reply of 151 tokens. The prefix cache served 90% of the tokens of prompts over 4,096.

- **Another request in a slot loses its conversation without `--host-cache`.** A device entry keeps only the slot's rings and conv states. The global KV cache stays in the slot, so any other sequence there overwrites it. With `--streams 1`, short side requests caused 6 full prefills of 77,421 to 150,445 tokens, at 5.7 to 12.5 s each. The side requests included the agent's own 539-token request and 21-token tests. The host tier, `--host-cache`, now restores such prompts in well under a second.
- **A prefill stalled every other stream.** In a test on an RTX PRO 6000 Blackwell with two streams of 1,048,576 positions, the running request delivered 2.2 tok/s while the second request's 325 s prefill ran, one batched step per prefill chunk. `--decode-floor` now runs more steps between the chunks (see [Concurrent requests](#concurrent-requests)).
- **A request without a token cap missed the prefix cache with `--streams 1`.** Each turn of a 5-turn Codex session, which sends no `max_output_tokens`, prefilled its whole prompt of 8,495 to 11,002 tokens, in 0.56 to 0.71 s. A lone request reserved pages for every position it might reach before it started, the whole pool without a cap. That reservation counted the cached prompt's ring copies twice, so it could never be met and dropped the draft's pages, and with them the cached prompt's draft state, which a lone request needs from its entry. A lone request now reserves its prompt and 2,048 positions and takes the rest as its reply grows. In a Codex-like session of 5 turns from 5,668 to 10,173 tokens on the Responses route without a cap, turns 2 to 5 now read 4,096 to 8,192 tokens from the cache and prefill in 80 to 177 ms, against 416 to 650 ms.
- **Decode slows past 100,000 tokens.** Agent turns decoded at a median 534 tok/s with prompts of 32,000 to 100,000 tokens, and 417 tok/s above 100,000. Speculation kept a median of 3.09 tokens per round.

## Building on macOS

On macOS, `siphon-cuda` compiles to an empty crate and `siphon serve` builds without its engine, so its tests run but it cannot serve. From a Mac, `scripts/remote.sh <ssh-host> <remote-dir> <command>` syncs the workspace to a Linux host and runs the command in its pixi environment. Set `NVCC` or `CUDA_HOME` to use another toolkit, and `SIPHON_CUDA_ARCH=sm_89` to build for an RTX 4090 (see [engine.md](engine.md#rtx-4090-sm_89)).
