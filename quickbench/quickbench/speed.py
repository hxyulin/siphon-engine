"""Speed track: fixed prompts, sampled with several seeds, at 1 and at 8 streams, and a
32K-token prompt at 1 stream with its time to first token on a prefix-cache miss and hit."""

import random
import time
from concurrent.futures import ThreadPoolExecutor

from . import client, quality

SAMPLINGS = {"t1.0": {"temperature": 1.0}, "t0.8-p0.95": {"temperature": 0.8, "top_p": 0.95}}
# The model's default: temperature 1 and no filters.
DEFAULT_SAMPLINGS = ("t1.0",)
# The long row: generated locker records, decoded for this many tokens.
LONG_PROMPT, LONG_TOKENS, LONG_REPLY = "long32k", 32000, 512

AGENT_SYSTEM = (
    "You are a coding agent working in a Python repository. Use the tools to read files, "
    "run commands and edit code. Keep changes small, run the tests after every edit, and "
    "explain what you changed and why when you finish."
)
AGENT_TOOLS = [
    {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": {
            p: {"type": "string", "description": d} for p, d in params.items()},
            "required": list(params)}}}
    for name, description, params in [
        ("read_file", "Read a file of the repository and return its text with line numbers.",
         {"path": "Path relative to the repository root."}),
        ("run", "Run a shell command in the repository root and return its output.",
         {"command": "The command line."}),
        ("edit_file", "Replace one exact occurrence of old text with new text in a file.",
         {"path": "File to edit.", "old": "Text to replace.", "new": "Replacement text."}),
    ]
]


def test_log(seed: int = 7) -> str:
    """A pytest log with one failure, generated the same way every run."""
    rng = random.Random(seed)
    words = "parse load merge split render cache index flush token config retry".split()
    lines = ["============================= test session starts =============================",
             "platform linux -- Python 3.12.3, pytest-8.3.2", "collected 180 items", ""]
    for module in ("config", "loader", "render", "cache", "cli", "index"):
        for i in range(30):
            name = f"test_{rng.choice(words)}_{rng.choice(words)}_{i}"
            status = "FAILED" if (module, i) == ("loader", 17) else "PASSED"
            lines.append(f"tests/test_{module}.py::{name} {status}")
    lines += [
        "", "=================================== FAILURES ===================================",
        "_____________________ test_merge_split_17 _____________________",
        "    def test_merge_split_17():",
        "        cfg = load_config(FIXTURES / 'nested.ini')",
        ">       assert cfg['server']['ports'] == [8000, 8001]",
        "E       AssertionError: assert '8000, 8001' == [8000, 8001]",
        "tests/test_loader.py:212: AssertionError",
        "=========================== 1 failed, 179 passed in 4.21s ===========================",
    ]
    return "\n".join(lines)


LOADER = '''def load_config(path):
    """Reads an INI file into nested dictionaries."""
    sections, current = {}, None
    for raw in Path(path).read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            current = sections.setdefault(line[1:-1].strip(), {})
            continue
        key, _, value = line.partition("=")
        value = value.strip()
        if value.isdigit():
            value = int(value)
        elif value.startswith('"') and value.endswith('"'):
            value = value[1:-1]
        current[key.strip()] = value
    return sections
'''

PROMPTS = {
    "chat": [{"role": "user", "content": "What are the main differences between a heat pump "
              "and a gas furnace for heating a house? Give a balanced answer."}],
    "code": [{"role": "user", "content": "Write a Python function that merges overlapping "
              "time intervals given as (start, end) pairs, with type hints, a docstring and "
              "a few pytest tests."}],
    "story": [{"role": "user",
               "content": "Write a detailed 600-word story about a lighthouse keeper."}],
    "agent": [
        {"role": "system", "content": AGENT_SYSTEM},
        {"role": "user", "content": "The test suite fails since my last change to the "
         "config loader. Find out why and fix it."},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function",
         "function": {"name": "run", "arguments": '{"command": "pytest -q"}'}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": test_log()},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "call_2", "type": "function",
         "function": {"name": "read_file", "arguments": '{"path": "src/app/loader.py"}'}}]},
        {"role": "tool", "tool_call_id": "call_2", "content": LOADER},
        {"role": "user", "content": "Before you edit anything, explain the cause of the failure "
         "and write out the fixed function in full."},
    ],
}


def long_messages(seed: int) -> list[dict]:
    """A 32K-token prompt of locker records. Each seed's prompt differs from its first line,
    so its first request misses siphon-serve's one-entry prefix cache."""
    messages, _ = quality.long_messages(LONG_TOKENS, 1000 + seed)
    return [{"role": "user", "content": f"Record set {seed}.\n\n{messages[0]['content']}"}]


def body(model: str, prompt: str, sampling: str, seed: int, max_tokens: int) -> dict:
    if prompt == LONG_PROMPT:
        return {"model": model, "messages": long_messages(seed), "max_tokens": LONG_REPLY,
                "seed": seed, **SAMPLINGS[sampling]}
    out = {"model": model, "messages": PROMPTS[prompt], "max_tokens": max_tokens,
           "seed": seed, **SAMPLINGS[sampling]}
    if prompt == "agent":
        # The tools stay in the prompt, but the reply is text: a tool call arrives in one
        # chunk at the end and has no decode time to measure.
        out |= {"tools": AGENT_TOOLS, "tool_choice": "none"}
    return out


def request(base: str, model: str, prompt: str, sampling: str, seed: int, max_tokens: int):
    reply = client.chat(base, body(model, prompt, sampling, seed, max_tokens))
    n = reply["completion_tokens"] or 0
    return {"prompt": prompt, "sampling": sampling, "seed": seed, "error": reply["error"],
            "completion_tokens": n, "prompt_tokens": reply["prompt_tokens"],
            "ttft": round(reply["ttft"], 4), "seconds": round(reply["seconds"], 3),
            # Only when the reply streamed: a reply sent in one chunk has no decode time.
            "decode_tok_s": round((n - 1) / reply["decode_s"], 1)
            if n > 1 and reply["decode_s"] > 0.5 * (reply["seconds"] - reply["ttft"]) > 0
            and reply["decode_s"] > 0.05 else None}


def run(base: str, model: str, seeds: int, max_tokens: int, server=None,
        samplings=DEFAULT_SAMPLINGS) -> list[dict]:
    jobs = [(p, s, seed) for p in PROMPTS for s in samplings for seed in range(1, seeds + 1)]
    long = [(LONG_PROMPT, s, seed) for s in samplings for seed in range(1, seeds + 1)]
    out = []
    for p, s, seed in jobs + long:
        before = len(server.jobs()) if server else 0
        record = request(base, model, p, s, seed, max_tokens) | {"streams": 1}
        if server:
            new = []
            for _ in range(40):  # the job line follows the last streamed chunk
                if new := server.jobs()[before:]:
                    break
                time.sleep(0.05)
            record["tokens_per_round"] = new[-1]["tokens_per_round"] if new else None
        if p == LONG_PROMPT:
            # The same request again hits the prefix cache.
            record["ttft_warm"] = round(
                client.chat(base, body(model, p, s, seed, max_tokens))["ttft"], 4)
        out.append(record)
    started = time.perf_counter()
    with ThreadPoolExecutor(8) as pool:
        batch = list(pool.map(lambda j: request(base, model, *j, max_tokens), jobs))
    wall = time.perf_counter() - started
    for record in batch:
        record |= {"streams": 8, "wall": round(wall, 2)}
    return out + batch
