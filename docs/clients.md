# Coding agents

Setup recipes that point coding-agent clients at `siphon-serve`. [server.md](server.md) covers the routes and their fields. The examples assume a server at `http://localhost:8000` with `--name inkling-2-mini`, the defaults. They follow the source of Codex 0.171.0, Pi 1.6.1, OpenCode 2.4.1 and DeepSeek Harness 0.2.3-alpha.2, and are not yet run against a server.

## Common settings

### Context

Each client needs the context window set to what the server planned. The server prints its planned context at startup, and `/v1/models` lists it as `max_model_len`. A client value above it lets a prompt overflow: the server then answers 400 with the code `context_length_exceeded` instead of the client compacting first.

| GPU | `siphon-serve` flags | Planned positions |
| --- | --- | ---: |
| RTX 4090, one user | `--streams 2 --kv mixed --host-cache 12` | 314,368 |
| RTX 4090, most context | `--kv nvfp4 --drafter <dflash2-dir> --prefill-chunk 1024 --margin 0.1` | 497,664 |
| RTX 4090, 2 to 4 users | `--streams 4 --kv mixed --host-cache 12` | 311,296, shared |
| RTX 5090, one user | `--streams 2 --host-cache 12` | 648,192 |

The RTX 4090 rows are measured ([Serving on an RTX 4090](server.md#serving-on-an-rtx-4090)). The RTX 5090 row is `siphon-plan`'s plan for an idle 32 GB card with FP8 caches. A display or another process on the card lowers it, so use the number your server prints. With several streams, one request may still take the whole pool.

At 478K tokens on the RTX 4090, the first token took 138.5 s. A client that gives up on a quiet connection sooner cuts long prompts off, so each section names its timeout.

### Output cap

Agents send caps of 16K to 32K tokens and reply with 150 to 2,000. Where a client lets you set the cap, use about 8,192. A request without a cap, or with a large one, misses the [prefix cache](server.md#prefix-cache) for now ([Known gaps](server.md#known-gaps)); a fix is in progress.

### Reasoning

The server takes these effort names on every route. A number from 0 to 0.99 also works on the chat route.

| Name | Template effort |
| --- | ---: |
| `none` | thinking off, see below |
| `minimal` | 0.1 |
| `low` | 0.2 |
| `medium` | 0.7 |
| `high` | 0.9, the default |
| `xhigh`, `max` | 0.99 |

Any other name gets a 400. `none` turns thinking off only for requests without tools. Coding agents send tools on every turn, and the model decides on a tool call in its thinking, so with tools `none` runs at effort 0 and the model still writes a short thinking block ([Thinking off](server.md#thinking-off)).

The thinking is public: the chat route returns it as `reasoning_content`, the Messages route as `thinking` blocks, and the Responses route as reasoning items with the whole text. Each section says how to show it.

### API key

The server needs no key unless it runs with `SIPHON_API_KEY`. Some clients refuse to start without one, so each section names the dummy value it needs.

### Sampling and images

A request that sets no sampling fields gets the checkpoint's defaults: temperature 1 with no filters. Images need `siphon-serve --vision`. Without it, the server replaces each image with the text `[image omitted: the model cannot see images]`.

## Codex

Codex uses `/v1/responses`, its only wire API for custom providers. Put the provider in a profile file, `~/.codex/siphon.config.toml`, and start Codex with `codex -p siphon`. The profile layers on top of `~/.codex/config.toml`, so the default OpenAI setup stays as it is.

```toml
model = "inkling-2-mini"
model_provider = "siphon"
model_reasoning_effort = "medium"
model_context_window = 272000            # Codex's limit for a model it does not know

[model_providers.siphon]
name = "Siphon"
base_url = "http://localhost:8000/v1"
wire_api = "responses"
```

```sh
codex -p siphon
```

- **Context.** Codex caps `model_context_window` at 272,000 tokens for a model missing from its catalog, so a larger value has no effect. To use more of the context, give Codex a catalog entry for the model, as below.
- **Compaction.** Codex compacts when the conversation passes `model_auto_compact_token_limit`, which defaults to 90% of the context window and cannot exceed it. At 272,000 that is 244,800, which leaves 27,200 tokens for the summary request, so the default works. Codex compacts locally for a custom provider: it sends the history and a summary prompt to `/v1/responses`, not to OpenAI's compaction endpoint.
- **Effort.** `model_reasoning_effort` reaches the server as `reasoning.effort`: `none`, `minimal`, `low`, `medium`, `high`, `xhigh` and `max` pass as they are. Codex always sends tools, so `none` runs at effort 0. Do not pick `ultra` or `persistent`: the server does not know them. Without `model_reasoning_effort`, Codex sends no effort and the server uses 0.9.
- **Showing the thinking.** Codex sends `reasoning.summary`, and the server fills each summary with the whole thinking, which Codex shows by default. Leave `show_raw_agent_reasoning` off, or Codex shows the text twice. `hide_agent_reasoning = true` hides it.
- **Output cap.** Codex sends no `max_output_tokens` and has no setting for one, so every turn misses the prefix cache for now.
- **Tools.** Codex warns that it has no metadata for the model and uses defaults: function tools such as `exec_command` and `write_stdin`, and no `apply_patch` tool, so the model edits files through the shell.
- **Sampling.** Codex sends no temperature.
- **Images.** Codex sends pasted images as `input_image` parts with `data:` URLs.
- **API key.** Codex needs none. Without `env_key`, it sends no `Authorization` header. With `SIPHON_API_KEY` on the server, add `env_key = "SIPHON_API_KEY"` to the provider and export the key.
- **Timeout.** Codex drops a stream after `stream_idle_timeout_ms`, 300,000 by default, without an event. The server's keep-alives are SSE comments, which Codex may not count as events. The 138.5 s prefill at 478K tokens fits; for the RTX 4090 setup with the most context, `stream_idle_timeout_ms = 600000` on the provider leaves more room.

### More context than 272,000

`model_catalog_json` replaces Codex's model catalog with a file. An entry for `inkling-2-mini` lifts the cap. The entry needs Codex's base instructions, which the fallback entry takes from the Codex binary. This script takes them from the Codex source at the installed version and writes the file:

```sh
v=$(codex --version | awk '{print $2}')
curl -fsSL "https://raw.githubusercontent.com/openai/codex/rust-v$v/codex-rs/models-manager/prompt.md" |
jq -n --rawfile p /dev/stdin --argjson ctx 314368 '{models: [{
  slug: "inkling-2-mini", display_name: "Inkling 2 Mini", description: null,
  base_instructions: $p,
  supported_reasoning_levels: [
    {effort: "none", description: "Effort 0 with tools"},
    {effort: "minimal", description: "Effort 0.1"},
    {effort: "low", description: "Effort 0.2"},
    {effort: "medium", description: "Effort 0.7"},
    {effort: "high", description: "Effort 0.9"},
    {effort: "xhigh", description: "Effort 0.99"}],
  default_reasoning_level: "medium",
  shell_type: "unified_exec", visibility: "list", supported_in_api: true, priority: 0,
  availability_nux: null, upgrade: null, support_verbosity: false, default_verbosity: null,
  apply_patch_tool_type: null, truncation_policy: {mode: "bytes", limit: 10000},
  context_window: $ctx, max_context_window: $ctx, experimental_supported_tools: []
}]}' > ~/.codex/siphon-models.json
```

Set `--argjson ctx` to the server's planned context. Then, in `~/.codex/siphon.config.toml`, replace the `model_context_window` line with:

```toml
model_catalog_json = "/Users/<you>/.codex/siphon-models.json"   # an absolute path
```

The fields match Codex's fallback entry for an unknown model, apart from the context, the effort list and the entry's place in `/model`. The catalog only applies under `-p siphon`. Codex 0.171.0 loads this file; a later Codex may add required fields, and then it refuses to start and names the missing one.

### Live checks

- The 272,000 cap: with `model_context_window = 314368` and no catalog, Codex's status line shows 272K.
- The catalog: `codex -p siphon` starts, the status line shows the planned context, `/model` offers the six efforts, and the first request carries the base instructions.
- Auto-compaction: the summary request reaches `/v1/responses`, finishes inside the context, and the next turn continues from the summary.
- `stream_idle_timeout_ms`: whether a prefill longer than the timeout, with only SSE comments, ends the stream.
- `model_reasoning_effort = "none"`: the request carries `reasoning.effort: "none"` and the model still calls tools.
- Images: a pasted image reaches the server as a `data:` URL.

## Pi

Pi, the coding agent `pi` (npm `@earendil-works/pi-coding-agent`, formerly `@mariozechner/pi-coding-agent`), reads custom providers from `~/.pi/agent/models.json`. Use the chat route, `api: "openai-completions"`. Pi reads `reasoning_content` from its stream and sends it back on the next turn, and it needs no compatibility flags for a local URL. The other two routes need workarounds: on `anthropic-messages`, Pi sends `budget_tokens`, which the server ignores, and turns the server's empty signatures into text unless `compat.allowEmptySignature` is set; on `openai-responses`, it adds the whole thinking and its identical summary to one block while streaming.

```json
{
  "providers": {
    "siphon": {
      "baseUrl": "http://localhost:8000/v1",
      "api": "openai-completions",
      "apiKey": "none",
      "models": [
        {
          "id": "inkling-2-mini",
          "name": "Inkling 2 Mini",
          "reasoning": true,
          "thinkingLevelMap": { "off": "none", "xhigh": "xhigh" },
          "input": ["text"],
          "contextWindow": 314368,
          "maxTokens": 8192
        }
      ]
    }
  }
}
```

In `~/.pi/agent/settings.json`:

```json
{
  "defaultProvider": "siphon",
  "defaultModel": "inkling-2-mini",
  "defaultThinkingLevel": "medium"
}
```

- **Context.** Set `contextWindow` to the server's planned context: 314,368 on the RTX 4090 setup for one user, 648,192 on the RTX 5090. Pi's default is 128,000.
- **Output cap.** Pi sends `maxTokens` as `max_completion_tokens` on every request, cut to fit the context left after its own estimate of the prompt. Its default is 16,384.
- **Compaction.** Pi compacts when the context passes `contextWindow` less `compaction.reserveTokens` (16,384 by default) in `settings.json`. The default leaves room for a reply at the 8,192 cap. The summary request keeps out the last `compaction.keepRecentTokens` (20,000) and asks for at most `maxTokens`, so it fits too.
- **Effort.** Pi's levels `minimal`, `low`, `medium` and `high` send `reasoning_effort` with the same name. Without `thinkingLevelMap`, `off` sends no `reasoning_effort`, so the server's 0.9 applies, and `xhigh` falls back to `high`. The map above sends `none` for `off` and offers `xhigh`. With tools, `off` runs at effort 0. Pi's compaction summary carries no tools, so at `off` it gets no thinking. Set the level with `/thinking`, Shift+Tab, `--thinking <level>` or `defaultThinkingLevel`.
- **Showing the thinking.** Pi shows it by default (`hideThinkingBlock: false`). Ctrl+T collapses and expands it.
- **Sampling.** Pi sends no temperature or `top_p`. `samplingParams` on the model adds fields to every request.
- **Tools.** Pi sends its tools (`read`, `bash`, `edit`, `write`) as function tools.
- **Images.** Set `"input": ["text", "image"]` when the server runs with `--vision`. Pi then sends images as `data:` URLs. Without `image`, Pi replaces images with text itself.
- **API key.** A provider without a key is left out of `/model`, so set any literal, such as `"none"`. `"$SIPHON_API_KEY"` reads the variable instead.
- **Timeout.** Pi's `httpIdleTimeoutMs` in `settings.json` is 300,000, and it also times the provider request through `retry.provider.timeoutMs`. During a prefill the chat route sends a keep-alive comment every 15 s.

### Live checks

- The prompt count Pi's compaction uses: Pi reads the server's streamed `usage`. Check that a turn past 297,984 tokens (314,368 less 16,384) starts a compaction.
- The summary request: it carries no tools, finishes inside the context, and the next turn continues from it.
- The output cap near the threshold: the `max_completion_tokens` Pi sends there, and that the server does not answer `context_length_exceeded`.
- `off`: the request carries `reasoning_effort: "none"` and tool calls still work.
- Past thinking: the second turn's request carries `reasoning_content` on the assistant message.
- The timeout: whether a prefill longer than 300 s, with only keep-alive comments, keeps the request alive, or whether `retry.provider.timeoutMs` ends it first.

## OpenCode

OpenCode 2 (github.com/anomalyco/opencode, formerly sst/opencode) reads providers from `~/.config/opencode/opencode.json`, or `opencode.json` in the project. Use the chat route through `@opencode/ai/providers/openai-compatible`. It reads `reasoning_content` from the stream and sends it back on assistant messages. The `/responses` and `anthropic-compatible` packages also exist but are not checked against the server.

```json
{
  "$schema": "https://opencode.ai/config.json",
  "model": "siphon/inkling-2-mini",
  "compaction": { "buffer": 32768 },
  "providers": {
    "siphon": {
      "name": "Siphon",
      "package": "@opencode/ai/providers/openai-compatible",
      "settings": { "baseURL": "http://localhost:8000/v1" },
      "models": {
        "inkling-2-mini": {
          "name": "Inkling 2 Mini",
          "settings": { "reasoningEffort": "medium" },
          "capabilities": {
            "tools": true,
            "input": ["text"],
            "output": ["text"],
            "reasoning": {
              "supported": true,
              "effort": ["none", "minimal", "low", "medium", "high", "xhigh"]
            }
          },
          "limit": { "context": 314368, "output": 8192 }
        }
      }
    }
  }
}
```

To show the thinking, add this to `~/.config/opencode/cli.json`:

```json
{ "session": { "thinking": "show" } }
```

- **Config shape.** OpenCode 2 still loads the 1.x shape (`provider`, `npm: "@ai-sdk/openai-compatible"`, `options.baseURL`) and translates it, but ignores 1.x model fields such as `reasoning` and `temperature`.
- **Context.** Set `limit.context` to the server's planned context: 314,368 on the RTX 4090 setup for one user, 648,192 on the RTX 5090. OpenCode's default for a custom model is 200,000.
- **Output cap.** OpenCode sends `max_completion_tokens` on agent and compaction requests: `limit.output`, cut to fit the context left after the prompt, and never below 1,024. Its default is 32,000. Title requests send no cap.
- **Compaction.** OpenCode compacts when the prompt passes the window less `compaction.buffer`. Without a buffer it keeps the larger of 15% and 16,000 tokens free, which is 47,155 at 314,368. A buffer of 32,768 leaves room for a reply at the 8,192 cap and a 16,000-token summary. `compaction.background` (on by default) writes the summary as a separate request while the session nears the threshold. With `--streams 1`, that request waits for the turn and takes the conversation's prefix cache entry, so run `--streams 2` or set `"background": false`.
- **Effort.** `settings.reasoningEffort` on the model is sent as `reasoning_effort` on every request. Each name in `capabilities.reasoning.effort` becomes a variant that overrides it. Cycle variants with Ctrl+T, pick one with `/variants` (also `/thinking` or `/effort`), or name it as `siphon/inkling-2-mini#high`. With tools, `none` runs at effort 0. Without `settings.reasoningEffort` and with no variant picked, OpenCode sends no effort and the server uses 0.9.
- **Showing the thinking.** OpenCode collapses thinking to one line by default. `session.thinking: "show"` in `cli.json` expands it, and the command palette's thinking toggle flips it. `/thinking` picks the effort, not the display.
- **Sampling.** OpenCode sends no temperature or `top_p`. A model's `body`, such as `"body": { "temperature": 0.6 }`, adds fields to every request.
- **Tools.** OpenCode sends its tools as function tools.
- **Images.** Set `"input": ["text", "image"]` when the server runs with `--vision`. OpenCode then sends `image_url` parts. With `["text"]`, OpenCode replaces images itself. The default for a custom model is `["text", "image"]`, so set it.
- **API key.** OpenCode needs none: without a key it sends no `Authorization` header. With `SIPHON_API_KEY` on the server, add `"apiKey": "{env:SIPHON_API_KEY}"` to `settings`.
- **Timeout.** OpenCode waits 300 s for response headers (`settings.headerTimeout`) and 900 s between stream chunks (`settings.chunkTimeout`). The server sends a stream's headers at once and a keep-alive comment every 15 s during a prefill.

### Live checks

- The 2.x config loads as written, and `siphon/inkling-2-mini` shows the six variants.
- `settings.reasoningEffort` reaches the server as `reasoning_effort`, and a picked variant replaces it.
- Background compaction: when the summary request starts, how it shares `--streams 2` with the turn, and that the next turn continues from the summary.
- The threshold: compaction starts near 281,600 tokens (314,368 less 32,768).
- The output cap sent near the threshold, and that the server does not answer `context_length_exceeded`.
- Past thinking: the second turn carries `reasoning_content` on the assistant message.
- `session.thinking: "show"` shows the whole thinking while it streams.

## DeepSeek Harness

DeepSeek Harness (`dsh`, github.com/deepseek-ai/deepseek-harness, a developer preview) has no `settings.yaml`. Each profile reads a list of plugin rows from `~/.dsh/profiles/<profile>/cordis.patch.yml`; the web UI's profile is `web`. Its `llm-pi-ai` plugin wraps the Pi library, so the recipe matches Pi's: the chat route, `api: openai-completions`. The library reads `reasoning_content` from the stream and sends it back for the same model.

In `~/.dsh/profiles/web/cordis.patch.yml`:

```yaml
- id: llm-pi-ai
  config:
    providers:
      siphon:
        displayName: Siphon
        api: openai-completions
        baseURL: http://localhost:8000/v1
        apiKeyEnv: SIPHON_API_KEY
        maxRequestImages: 16
        models:
          - id: inkling-2-mini
            name: Inkling 2 Mini
            contextWindow: 314368
            maxTokens: 8192
            input: [text]
            reasoningEfforts:
              off: none
              minimal: minimal
              low: low
              medium: medium
              high: high
              xhigh: xhigh

- id: agent-default-model
  config:
    provider: siphon
    model: inkling-2-mini
    reasoningEffort: medium
```

In `~/.dsh/.env`:

```sh
SIPHON_API_KEY=none
```

```sh
dsh web --dump-config   # prints the composed config
dsh web
```

- **Patch rows.** A row in the file replaces that plugin's whole `config`. Providers added on the Models page are saved to the same row, so keep them under the same `providers:`. `~/.dsh/cordis.patch.yml` applies to every profile, but it outranks the profile file, and the Models page and model picker then cannot save. For the headless profile, put the same rows in `~/.dsh/profiles/headless/cordis.patch.yml`.
- **Context.** Set `contextWindow` to the server's planned context: 314,368 on the RTX 4090 setup for one user, 648,192 on the RTX 5090. A model without one gets the plugin's default of 1,000,000.
- **Output cap.** Every request sends `maxTokens` as `max_completion_tokens`, cut to the context less the estimated prompt and 4,096. Without `maxTokens`, the cap is 128,000.
- **Compaction.** The `compaction-basic` plugin compacts at the lower of 80% of the window and the window less the request's cap less 65,536 headroom tokens: 240,640 at 314,368 with the 8,192 cap. That leaves 73,728 tokens for the summary request, which asks for at most the headroom, so the defaults work. The settings are `thresholdRatio`, `headroomTokens` and `maxTokens` on a `- id: compaction-basic` row, which works in the headless profile. The `web` profile mounts the plugin inside its agent preset, so changing it there means restating the whole `preset-standard` row.
- **Effort.** A model without `reasoningEfforts` has no effort menu. Levels you leave out of the map are hidden. Each level sends `reasoning_effort` with its mapped name, and `off` sends `none`, which with tools runs at effort 0. Set a default in `agent-default-model.reasoningEffort`: with `off: none` and no default, new sessions start at `off`.
- **Showing the thinking.** The web UI shows the thinking as a collapsed row that expands. Settings, General, Work details sets its first-line preview, which the Compact mode hides. The headless profile prints the thinking to stderr.
- **Sampling.** `dsh` sends no temperature or `top_p`, and has no setting for them.
- **Tools.** `dsh` sends function tools and no `tool_choice`.
- **Images.** Set `input: [text, image]` when the server runs with `--vision`. `dsh` sends `data:` URLs. `maxRequestImages: 16` matches the server's default `--max-images`; the plugin's default is 50.
- **API key.** The library refuses a request without a key, so the route needs `apiKeyEnv` and any non-empty value. `dsh` reads it from `~/.dsh/.credentials.yaml` (`version: 1`, then `refs: {SIPHON_API_KEY: none}`), the environment, `./.env` or `~/.dsh/.env`. The headless profile skips the credentials file, so `~/.dsh/.env` covers every profile.
- **Timeout.** `streamIdleTimeoutMs` on the provider, 300,000 by default, ends a stream that stays idle. `timeoutMs` sets the HTTP timeout.

### Live checks

- `dsh web --dump-config` shows the `siphon` route and the `agent-default-model` row.
- The Models page lists Inkling 2 Mini with the six efforts, and a picked effort reaches the server as `reasoning_effort`.
- Compaction starts near 240,640 tokens, the summary request finishes inside the context, and the next turn continues from it.
- Whether the server's keep-alive comments count as activity for `streamIdleTimeoutMs` during a prefill longer than 300 s.
- Past thinking: the second turn carries `reasoning_content` on the assistant message.
