"""Quality track: items, prompts, scoring and the sampled run.

Item ids name the source and a stable key: `gpqa/<row>` (row of gpqa_diamond.csv),
`aime/<id>`, `code/<HumanEval number>`, `long/<prompt tokens>/<seed>` (generated
here) and `tool/<name>` (defined here). `items.json` lists the chosen ids with
the pass rate measured for each.
"""

import csv
import hashlib
import json
import random
import re
import shutil
import subprocess
import sys
import tempfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import client, fetch

# Output caps per source, in tokens, for the build check. The suite caps every source
# at SUITE_CAP.
SUITE_CAP = 32768
CAPS = {"gpqa": 16384, "aime": 16384, "code": 16384, "long": 8192, "tool": 4096}

GPQA_TEMPLATE = (
    "Answer the following multiple choice question. The last line of your response should "
    "be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of "
    "ABCD. Think step by step before answering.\n\n{Question}\n\nA) {A}\nB) {B}\nC) {C}\nD) {D}"
)
AIME_SUFFIX = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
CODE_TEMPLATE = (
    "Complete the following Python function. Reply with the complete function, and any "
    "imports it needs, in one ```python code block.\n\n```python\n{prompt}```"
)


def seed_of(*parts) -> int:
    return int(hashlib.sha256("/".join(map(str, parts)).encode()).hexdigest()[:8], 16) >> 1


class Datasets:
    """The fetched datasets, loaded on first use."""

    def __init__(self, cache: Path):
        self.cache = cache
        self._rows = {}

    def rows(self, source: str):
        if source not in self._rows:
            path = fetch.path(self.cache, source)
            if not path.exists():
                raise SystemExit(f"{path} is missing: run `quickbench fetch` first")
            if source == "gpqa":
                with open(path, newline="") as f:
                    self._rows[source] = list(csv.DictReader(f))
            else:
                rows = [json.loads(line) for line in path.open()]
                key = "id" if source == "aime" else "task_id"
                self._rows[source] = {str(r[key]).removeprefix("HumanEval/"): r for r in rows}
        return self._rows[source]


# Long-context retrieval: locker records, some of which point to another locker. Each
# asked locker reaches its code through a chain of pointers spread over the text, and
# the chain's last locker has near-duplicate labels nearby.
ADJECTIVES = ("amber azure brisk calm coral crisp dusky eager faint gilded hazy ivory jade "
              "keen lunar mossy noble olive pale quiet rapid rustic sable silver tawny umber "
              "vivid wary woven young zesty").split()
NOUNS = ("falcon harbor meadow lantern quarry ember thistle canyon willow beacon cobalt "
         "garnet hollow juniper kestrel lagoon marble nettle orchid pebble quill raven "
         "saffron tundra violet walnut yarrow zephyr").split()
# Characters per token of the record lines, measured with the model's tokenizer.
CHARS_PER_TOKEN = 2.6
ASKED, HOPS, TWINS = 4, 3, 4


def long_messages(tokens: int, seed: int):
    rng = random.Random(seed)
    used = set()

    def label():
        while True:
            key = f"{rng.choice(ADJECTIVES)}-{rng.choice(NOUNS)}-{rng.randrange(1000, 10000)}"
            if key not in used:
                used.add(key)
                return key

    def code():
        return f"pass code {rng.randrange(100000, 1000000)}"

    lines, chars = [], 0
    while chars < tokens * CHARS_PER_TOKEN:
        lines.append([label(), code()])
        chars += len(lines[-1][0]) + 26
    asked, want = [], []
    for _ in range(ASKED):
        i = rng.randrange(len(lines))
        root, value = lines[i]
        want.append(value.split()[-1])
        stem, digits = root.rsplit("-", 1)
        for _ in range(TWINS):
            twin = f"{stem}-{''.join(rng.sample(digits, len(digits)))}"
            if twin not in used:
                used.add(twin)
                lines.insert(max(0, i + rng.randrange(-100, 100)), [twin, code()])
        target = root
        for _ in range(HOPS):
            key = label()
            lines.insert(rng.randrange(len(lines) + 1), [key, f"same pass code as locker {target}"])
            target = key
        asked.append(target)
    body = "".join(f"Locker {k}: {v}.\n" for k, v in lines)
    names = ", ".join(asked)
    prompt = (
        "Below is a list of locker records. Each line gives a locker label and either its pass "
        "code or another locker whose pass code it shares.\n\n"
        f"{body}\nWhat are the pass codes of lockers {names}? Labels must match exactly. End "
        "your reply with a line of the form 'Answer: <code>, <code>, <code>, <code>', with the "
        "codes in the order the lockers are asked."
    )
    return [{"role": "user", "content": prompt}], want


WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "Current weather for a city.",
    "parameters": {"type": "object", "properties": {
        "city": {"type": "string"},
        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
        "required": ["city", "unit"]}}}
FLIGHTS = {"type": "function", "function": {
    "name": "search_flights", "description": "Search one-way flights.",
    "parameters": {"type": "object", "properties": {
        "origin": {"type": "string", "description": "IATA airport code"},
        "destination": {"type": "string", "description": "IATA airport code"},
        "date": {"type": "string", "description": "YYYY-MM-DD"},
        "passengers": {"type": "integer"}},
        "required": ["origin", "destination", "date", "passengers"]}}}
TOOLS = {
    "weather": ("What's the weather like in Oslo and in Lima right now? Use Celsius.",
                [{"name": "get_weather", "args": {"city": "oslo", "unit": "celsius"}},
                 {"name": "get_weather", "args": {"city": "lima", "unit": "celsius"}}]),
    "flights": ("Find me flights from Zurich to Lisbon on 14 March 2027 for 3 people.",
                [{"name": "search_flights", "args": {"origin": "zrh", "destination": "lis",
                                                     "date": "2027-03-14", "passengers": 3}}]),
    "none": ("What is 17 times 23? Answer with the number only.", []),
}


def build(item_id: str, attempt: int, data: Datasets):
    """Returns (request body without sampling fields, scorer of a reply)."""
    source, key = item_id.split("/", 1)
    if source == "gpqa":
        row = data.rows("gpqa")[int(key)]
        choices = [row["Correct Answer"], row["Incorrect Answer 1"],
                   row["Incorrect Answer 2"], row["Incorrect Answer 3"]]
        # A fresh option order per attempt, as simple-evals draws one per repeat.
        order = random.Random(seed_of(item_id, attempt)).sample(range(4), 4)
        shuffled = [choices[i] for i in order]
        gold = "ABCD"[order.index(0)]
        prompt = GPQA_TEMPLATE.format(Question=row["Question"], **dict(zip("ABCD", shuffled)))
        messages = [{"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}]

        def score(reply):
            m = re.search(r"(?i)Answer[ \t]*:[ \t]*\$?([A-D])\$?", reply["content"])
            return bool(m) and m.group(1) == gold
        return {"messages": messages}, score
    if source == "aime":
        row = data.rows("aime")[key]
        messages = [{"role": "user", "content": row["problem"] + AIME_SUFFIX}]

        def score(reply):
            boxed = re.findall(r"\\boxed\{([^{}]*)\}", reply["content"])
            digits = re.sub(r"[^0-9-]", "", boxed[-1]) if boxed else ""
            return digits.lstrip("-").isdigit() and int(digits) == int(row["answer"])
        return {"messages": messages}, score
    if source == "code":
        row = data.rows("code")[key]
        messages = [{"role": "user", "content": CODE_TEMPLATE.format(prompt=row["prompt"])}]
        return {"messages": messages}, lambda reply: run_tests(row, reply["content"])
    if source == "long":
        tokens, seed = (int(x) for x in key.split("/"))
        messages, want = long_messages(tokens, seed)

        def score(reply):
            lines = [x for x in reply["content"].splitlines() if "answer" in x.lower()]
            return re.findall(r"\b\d{6}\b", lines[-1] if lines else "") == want
        return {"messages": messages}, score
    if source == "tool":
        prompt, want = TOOLS[key]
        body = {"messages": [{"role": "user", "content": prompt}],
                "tools": [WEATHER, FLIGHTS], "tool_choice": "auto"}
        return body, lambda reply: tool_calls_match(reply, want)
    raise ValueError(f"unknown item {item_id}")


def tool_calls_match(reply, want) -> bool:
    if not want:
        return not reply["tool_calls"] and "391" in reply["content"]
    got = []
    for call in reply["tool_calls"]:
        try:
            args = json.loads(call["arguments"])
        except json.JSONDecodeError:
            return False
        if not isinstance(args, dict):
            return False
        got.append({"name": call["name"],
                    "args": {k: v.lower() if isinstance(v, str) else v for k, v in args.items()}})

    def same(g, w):
        if g["name"] != w["name"]:
            return False
        for k, v in w["args"].items():
            have = g["args"].get(k)
            if k == "city" and isinstance(have, str):
                have = have.split(",")[0].strip()
            if have != v or type(have) is not type(v):
                return False
        return True
    return len(got) == len(want) and all(any(same(g, w) for g in got) for w in want)


def unshare_works() -> bool:
    if not shutil.which("unshare"):
        return False
    return subprocess.run(["unshare", "-rn", "true"], capture_output=True).returncode == 0


NO_NETWORK = None


def run_tests(row, content: str) -> bool:
    """Runs HumanEval+ tests on the reply's last Python block, offline, in a subprocess."""
    global NO_NETWORK
    blocks = re.findall(r"```(?:python|py)?\n(.*?)```", content, re.S)
    if not blocks:
        return False
    code = blocks[-1]
    imports = "".join(line + "\n" for line in row["prompt"].splitlines()
                      if line.startswith(("import ", "from ")))
    if f"def {row['entry_point']}" not in code:
        code = row["prompt"] + code
    program = f"{imports}\n{code}\n\n{row['test']}\n\ncheck({row['entry_point']})\n"
    if NO_NETWORK is None:
        NO_NETWORK = unshare_works()
    # Without a network namespace, sockets are switched off inside the interpreter.
    guard = "" if NO_NETWORK else "import socket\nsocket.socket = None\n"
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp, "candidate.py")
        path.write_text(guard + program)
        command = [sys.executable, "-I", str(path)]
        if NO_NETWORK:
            command = ["unshare", "-rn", *command]
        try:
            done = subprocess.run(command, cwd=tmp, capture_output=True, timeout=60,
                                  stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            return False
    return done.returncode == 0


def reply_hash(reply) -> str:
    """A hash of the reply's text and tool calls, to find replies that changed between runs."""
    text = json.dumps([reply["reasoning"], reply["content"], reply["tool_calls"]])
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def looping(reply) -> bool:
    """A reply whose last 4,000 characters compress below 15% repeats itself."""
    tail = (reply["reasoning"] + reply["content"])[-4000:].encode()
    return len(tail) >= 4000 and len(zlib.compress(tail)) < 0.15 * len(tail)


def run(base: str, model: str, item_ids, attempts: int, sampling: dict, concurrency: int,
        data: Datasets, expected_tokens: dict, log=None, first_attempt: int = 0):
    """Runs every attempt of every item and returns one record per attempt."""
    jobs = [(i, a) for i in item_ids for a in range(first_attempt, first_attempt + attempts)]
    # Longest first, so the slowest replies do not start last.
    jobs.sort(key=lambda j: -expected_tokens.get(j[0], CAPS[j[0].split("/")[0]] // 4))

    def one(job):
        item_id, attempt = job
        body, score = build(item_id, attempt, data)
        seed = seed_of(item_id, attempt)
        body |= {"model": model, "max_tokens": CAPS[item_id.split("/")[0]], "seed": seed,
                 **sampling}
        reply = client.chat(base, body)
        passed = reply["error"] is None and score(reply)
        record = {"item": item_id, "attempt": attempt, "seed": seed, "passed": passed,
                  "finish_reason": reply["finish_reason"], "error": reply["error"],
                  "completion_tokens": reply["completion_tokens"],
                  "prompt_tokens": reply["prompt_tokens"], "looping": looping(reply),
                  "seconds": round(reply["seconds"], 2), "reply_hash": reply_hash(reply)}
        if log:
            log({**record, "reasoning": reply["reasoning"], "content": reply["content"],
                 "tool_calls": reply["tool_calls"]})
        return record

    with ThreadPoolExecutor(concurrency) as pool:
        return list(pool.map(one, jobs))
