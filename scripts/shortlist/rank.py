"""Count generated tokens into a shortlist ranking and report its coverage.

usage: rank.py <tokens.jsonl> <out.bin> [--rows 32768]

Reads the records `generate.py` writes. Records whose `prompt_id` ends in 0 (about 1 in
16) are held out. The ranking lists output tokens by count, then prompt tokens by count,
then every other token by id, special tokens (id 199,990 and up) left out since the
shortlist keeps them all. It writes the first `--rows` ids as little-endian u32 and
prints the share of held-out generated tokens they cover, by prompt category.
"""

import argparse
import collections
import json
import struct

SPECIAL = 199_990

ap = argparse.ArgumentParser()
ap.add_argument("tokens")
ap.add_argument("out")
ap.add_argument("--rows", type=int, default=32768)
a = ap.parse_args()

records = [json.loads(line) for line in open(a.tokens) if line.strip()]
held = [r for r in records if r["prompt_id"].endswith("0")]
train = [r for r in records if not r["prompt_id"].endswith("0")]
outputs = collections.Counter(t for r in train for t in r["output_ids"] if t < SPECIAL)
prompts = collections.Counter(t for r in train for t in r["prompt_ids"] if t < SPECIAL)
ranking = [t for t, _ in outputs.most_common()]
ranking += [t for t, _ in prompts.most_common() if t not in outputs]
seen = set(ranking)
ranking += [t for t in range(SPECIAL) if t not in seen]
with open(a.out, "wb") as f:
    f.write(struct.pack(f"<{a.rows}I", *ranking[: a.rows]))

kept = set(ranking[: a.rows])
by_category = collections.defaultdict(list)
for r in held:
    by_category[r["category"]].extend(r["output_ids"])
print(f"{len(train)} records counted, {sum(outputs.values())} output tokens, "
      f"{len(outputs)} distinct; {len(held)} held out")
for name, tokens in [("all held out", [t for v in by_category.values() for t in v])] + sorted(by_category.items()):
    covered = sum(1 for t in tokens if t in kept or t >= SPECIAL)
    print(f"{name:14s} {covered / len(tokens) * 100:6.2f}% of {len(tokens)} tokens")
