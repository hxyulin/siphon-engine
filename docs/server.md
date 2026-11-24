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
| `--streams N` | `1` | Requests that run at once, 1 to 8, each with its own KV cache. |
| `--queue N` | `8` | Requests that wait beyond the running ones. |
| `--widths` | `1,2,...,8` | Verify widths speculation picks from. |
| `--drafter DIR` | bundled DSpark | Runs Inco AI's DFlash 2 drafter or a 2DSD DFlash export instead. |
| `--draft` | `fast` | The draft's approximations: `fast`, `fp8`, `exact`, or a `+`-joined set of `fp8-backbone`, `fp8-head`, `fp8-fold` and `shortlist`. |
| `--fallback` | off | Runs plain steps where no verify width is expected to be faster. It speeds up free prose, but the output can then differ between runs. |
| `--no-prefix-cache` | cache on | Turns off the prefix cache and gives its 0.2 GiB to the KV cache. |
| `--prefill-attention auto\|bf16\|fp8` | `auto` | Prefill's global attention over an FP8 cache. `auto` runs BF16 up to 16,384 keys and FP8 tensor cores past them. |
| `--greedy` | off | Decodes every request greedily, whatever its sampling fields ask. |

With `SIPHON_API_KEY` set, every route but `/health` and `/version` needs `Authorization: Bearer <key>`.

## Routes

| Route | Notes |
| --- | --- |
| `POST /v1/chat/completions` | Messages rendered as the checkpoint's chat template renders them. Streams with `stream`, and `stream_options.include_usage` adds a usage chunk. The model's thinking comes back as `reasoning_content`. |
| `POST /v1/completions` | A text prompt or token ids, streaming or not. The text includes any special tokens generated. |
| `POST /tokenize`, `POST /detokenize` | vLLM's shape, also at `/v1/tokenize` and `/v1/detokenize`: `{"prompt"}` or `{"messages", "tools"}` to `{"tokens", "count", "max_model_len"}`, and `{"tokens"}` to `{"prompt"}`. |
| `POST /v1/token/encode`, `POST /v1/token/decode` | TabbyAPI's shape: `{"text"}`, a string or messages, to `{"tokens", "length"}`, and `{"tokens"}` to `{"text"}`. |
| `GET /v1/models`, `GET /v1/models/{id}` | The served model, with `max_model_len`. |
| `GET /health`, `GET /version` | No API key needed. |

## Chat request fields

- `reasoning_effort`: a name such as `low`, or 0 to 0.99. The default is 0.9. Without it, `chat_template_kwargs.reasoning_effort` counts, and `chat_template_kwargs.enable_thinking: false` means effort 0. The template has no switch that turns thinking off, so at effort 0 the model still writes a short thinking block.
- `tools` and `tool_choice`: `auto`, `none`, `required`, or `{"type": "function", "function": {"name": ...}}`. A named function, or `required` with one tool, ends the prompt with that tool's call opened. `required` with several tools opens a call without a name, and the model names the tool. A forced call skips the thinking block. The reply ends after the named call, or at the first text after the calls `required` forced. `parallel_tool_calls: false` ends the reply after its first call.
- `stop`: a string or an array of strings. Stop strings match the reply's content, not its thinking or tool calls. While streaming, text that could still begin a stop string is held back. The reply is cut before the match and finishes with `stop`.
- vLLM's extensions:
  - `return_token_ids` adds `prompt_token_ids` and each choice's `token_ids`, streaming or not. The ids end with `<|content_model_end_sampling|>` when the model ended its turn.
  - Each choice has `stop_reason`: the stop string that matched, or null.
  - `ignore_eos` generates `max_tokens` tokens and passes the model's end tokens on as output, for benchmarks.
  - `continue_final_message` with `add_generation_prompt: false` continues the last assistant message.

Raw completions take `stop`, `return_token_ids` and `ignore_eos` too. A text prompt, like the text sent to the tokenizer routes, is encoded with special-token spellings as plain text and nothing added; send token ids to use special tokens. Decoding keeps special tokens.

## Sampling

Both routes sample with `temperature`, `top_k`, `top_p`, `min_p` and `seed`, in vLLM's order: temperature, then top-k, then top-p, then min-p. A field the request leaves out comes from the checkpoint's `generation_config.json`, as in vLLM. For Inkling 2 Mini that is temperature 1 with no filters. `top_k` of 0 or -1 keeps every token.

`temperature: 0` decodes greedily and gives the same tokens as earlier greedy builds. A request with a `seed` gets the same tokens again on the same server configuration, alone or in a batch. A request without one gets a random seed. `--greedy` decodes every request greedily.

## Accepted, ignored and refused fields

The penalties (`presence_penalty`, `frequency_penalty`, `repetition_penalty`) and other sampling fields are accepted and ignored. So are `user`, `skip_special_tokens` and unknown fields.

Fields the server cannot honour get a 400: `n` or `best_of` above 1, `logprobs`, `top_logprobs`, `prompt_logprobs`, `echo`, `suffix`, `logit_bias`, `min_tokens`, `stop_token_ids`, `response_format` other than `text`, vLLM's guided decoding fields, and the deprecated `functions` and `function_call`.

## Concurrent requests

With `--streams N`, up to N requests run at once and up to `--queue` more wait. Each stream has its own KV cache of `--context` positions. Without `--context`, the server takes the longest context at which N caches fit.

A request that arrives at an idle server speculates. When another arrives, the first stops speculating after its round, and every running request decodes in a batched step of one row each. When one request is left and nothing waits, it speculates again. A new request's prompt prefills in chunks of 2,048 tokens between the batch's steps. Either way each request gets the same tokens it would get alone, greedy or sampled with a seed. [batching.md](batching.md) has the design.

## Prefix cache

Chat clients and agents resend the whole conversation each turn. The server keeps the state of the last prompt that ran alone, at that prompt's last multiple of 2,048 tokens. A request whose prompt starts with those tokens prefills only the rest, and gets the same tokens and logits as without the cache. The usage object reports the reused tokens as `prompt_tokens_details.cached_tokens`.

A request that runs batched prefills its whole prompt and leaves the cache alone, unless it takes slot 0 (see [engine.md](engine.md#prefix-cache)). The copies take 171 MiB with the bundled draft and 203 MiB with DFlash 2, whatever the context.

## Building on macOS

On macOS, `siphon-cuda` compiles to an empty crate and `siphon-serve` builds without its engine, so its tests run but it cannot serve. From a Mac, `scripts/remote.sh <ssh-host> <remote-dir> <command>` syncs the workspace to a Linux host and runs the command in its pixi environment. Set `NVCC` or `CUDA_HOME` to use another toolkit, and `SIPHON_CUDA_ARCH` to target another GPU.
