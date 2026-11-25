"""Streaming client for an OpenAI-compatible chat endpoint, standard library only."""

import json
import os
import time
import urllib.error
import urllib.request


def get(url: str, timeout: float = 10) -> dict:
    request = urllib.request.Request(url, headers=headers())
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def headers() -> dict:
    h = {"Content-Type": "application/json"}
    if key := os.environ.get("QUICKBENCH_API_KEY"):
        h["Authorization"] = f"Bearer {key}"
    return h


def chat(base: str, body: dict, timeout: float = 1800) -> dict:
    """Sends one streamed chat request and returns the reply with its timings.

    `ttft` runs to the first streamed text (thinking or content) or tool call,
    `decode_s` from there to the end. Errors come back in `error`, never raised.
    """
    body = {**body, "stream": True, "stream_options": {"include_usage": True}}
    request = urllib.request.Request(
        f"{base}/chat/completions", data=json.dumps(body).encode(), headers=headers()
    )
    out = {"content": "", "reasoning": "", "tool_calls": {}, "finish_reason": None,
           "completion_tokens": None, "prompt_tokens": None, "error": None}
    started = time.perf_counter()
    first = None
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw in response:
                line = raw.decode().strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                if usage := chunk.get("usage"):
                    out["completion_tokens"] = usage.get("completion_tokens")
                    out["prompt_tokens"] = usage.get("prompt_tokens")
                for choice in chunk.get("choices") or []:
                    delta = choice.get("delta") or {}
                    text = delta.get("reasoning_content") or delta.get("reasoning") or ""
                    out["reasoning"] += text
                    out["content"] += delta.get("content") or ""
                    for call in delta.get("tool_calls") or []:
                        slot = out["tool_calls"].setdefault(
                            call.get("index", 0), {"name": "", "arguments": ""})
                        fn = call.get("function") or {}
                        slot["name"] += fn.get("name") or ""
                        slot["arguments"] += fn.get("arguments") or ""
                    if first is None and (text or delta.get("content") or delta.get("tool_calls")):
                        first = time.perf_counter()
                    if choice.get("finish_reason"):
                        out["finish_reason"] = choice["finish_reason"]
    except urllib.error.HTTPError as e:
        out["error"] = f"HTTP {e.code}: {e.read().decode(errors='replace')[:300]}"
    except Exception as e:  # noqa: BLE001 - a failed request is a result, not a crash
        out["error"] = f"{type(e).__name__}: {e}"
    end = time.perf_counter()
    out["tool_calls"] = [out["tool_calls"][k] for k in sorted(out["tool_calls"])]
    out["ttft"] = (first or end) - started
    out["decode_s"] = end - (first or end)
    out["seconds"] = end - started
    return out
