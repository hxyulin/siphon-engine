#!/usr/bin/env python3
"""Select a deterministic, deduplicated sample of open-perfectblend prompts: 4,800 coding,
3,600 math, 2,400 chat and 1,200 instruction, shuffled.

usage: prompts.py <shard-dir> <tokenizer.json> <out.jsonl> [--previous FILE...]

Needs pyarrow and tokenizers, for example `uv run --with pyarrow --with tokenizers`.
`--previous` skips prompts whose last user message appears in those files.
"""

import argparse
from collections import Counter
from hashlib import sha256
from itertools import zip_longest
import json
from pathlib import Path
import random
import re

import pyarrow.parquet as parquet
from tokenizers import Tokenizer


QUOTAS = {"coding": 4800, "math": 3600, "chat": 2400, "instruction": 1200}
CAPS = {"coding": 768, "math": 640, "chat": 384, "instruction": 512}
ROLES = {"human": "user", "user": "user", "system": "system", "gpt": "assistant", "assistant": "assistant"}
CODE = re.compile(r"\b(python|javascript|typescript|rust|c\+\+|java|golang|kotlin|swift|ruby|algorithm|regex|code|"
                  r"function|programming|pseudocode|sql)\b", re.I)
SKIP = re.compile(r"\b(exploit|malware|ransomware|webshell|phishing|CVE-\d+|credential theft|shellcode|"
                  r"password cracking|auth bypass)\b", re.I)


def key(content):
    return " ".join(content.replace("<|", "< |").split())


def prior_messages(files):
    for path in files:
        with path.open() as stream:
            for row in map(json.loads, stream):
                for m in row.get("messages", ()):
                    if m.get("role") == "user" and isinstance(m.get("content"), str):
                        yield key(m["content"])


def category(source, prompt):
    label = source.lower()
    if "autoif" in label:
        return "instruction"
    if "math" in label:
        return "math"
    if "codealpaca" in label:
        return "coding"
    if "ultrainteract" in label:
        return "coding" if CODE.search(prompt) else "math"
    return "chat"


def prompt_from(row):
    items = row["conversations"]
    if not items or any(m["from"] not in ROLES or not m["value"] for m in items):
        return None
    last_user = max((i for i, m in enumerate(items) if ROLES[m["from"]] == "user"), default=-1)
    if last_user < 0:
        return None
    result = [dict(role=ROLES[m["from"]], content=m["value"].replace("<|", "< |").strip())
              for m in items[:last_user + 1]]
    conversation = result[1:] if result[0]["role"] == "system" else result
    if not all(m["role"] == ("assistant" if i % 2 else "user") and m["content"]
               for i, m in enumerate(conversation)):
        return None
    return result


def rows(path):
    index = 0
    source = parquet.ParquetFile(path)
    for batch in source.iter_batches(batch_size=256, columns=["source", "conversations"]):
        batch_rows = batch.to_pylist()
        yield [(i, row, path.name) for i, row in enumerate(batch_rows, index)]
        index += len(batch_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shards", type=Path)
    parser.add_argument("tokenizer", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--previous", type=Path, nargs="*", default=[])
    args = parser.parse_args()
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    used = set(prior_messages(args.previous))
    selected, counts = [], Counter()
    shards = sorted(args.shards.glob("*.parquet"))
    assert len(shards) == 3
    for batches in zip_longest(*(rows(path) for path in shards)):
        if counts.total() == sum(QUOTAS.values()):
            break
        for index, row, shard in (item for batch in batches if batch for item in batch):
            messages = prompt_from(row)
            if not messages:
                continue
            last = messages[-1]["content"]
            identity = key(last)
            if len(identity) < 16 or identity in used or any(SKIP.search(m["content"]) for m in messages):
                continue
            group = category(row["source"], last)
            if counts[group] >= QUOTAS[group] or sum(len(m["content"]) for m in messages) > 20_000:
                continue
            encoded = tokenizer.encode_batch([m["content"] for m in messages], add_special_tokens=False)
            length = 12 + sum(8 + len(x.ids) for x in encoded)
            if not 32 <= length <= 4000:
                continue
            digest = sha256(identity.encode()).hexdigest()
            selected.append(dict(
                prompt_id=digest, source="mlabonne/open-perfectblend", source_subset=row["source"],
                source_row=f"{shard}:{index}", category=group, panel="short" if length <= 1800 else "4k",
                partition="train", input_estimate=length, messages=messages, effort="none", limit=CAPS[group],
            ))
            counts[group] += 1
            used.add(identity)
    if counts != QUOTAS:
        raise ValueError(f"Insufficient eligible prompts: {counts}")
    random.Random(98431).shuffle(selected)
    with args.output.open("x") as stream:
        for row in selected:
            print(json.dumps(row, ensure_ascii=False), file=stream)
    result = dict(prompts=len(selected), categories=dict(counts),
                  panels=dict(Counter(row["panel"] for row in selected)),
                  planned_prompt_tokens=sum(row["input_estimate"] for row in selected))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
