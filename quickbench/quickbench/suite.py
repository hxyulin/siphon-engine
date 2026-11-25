"""The quality suite's items and its wall-time estimate.

`python -m quickbench.suite [--cache DIR]` draws the items from the fetched datasets and
writes suite.json. The draw takes no pass rates or reply lengths into account.
"""

import argparse
import json
import random
from pathlib import Path

from . import quality

SEED = 20261125
GPQA_ITEMS = 100
# Generated retrieval items, seeds the build check does not use.
LONG = {30000: range(101, 109), 124000: range(201, 209)}
# Tokens per reply at 32K caps, sampled at temperature 1: GPQA from a 660-reply gpqa_diamond
# run, the others from the calibration runs at 8K to 16K caps, raised for the replies those
# caps cut off. The estimate and the longest-first order use them.
TOKENS = {"gpqa": 9200, "aime": 11500, "code": 1500, "long": 2700, "tool": 120}

# Rates for the estimate on one RTX 5090: 8 streams decode about 850 tok/s in total, the
# long-context server prefills about 13,000 tok/s and its 2 streams decode about 600 tok/s.
# One of the 8 streams decodes about 110 tok/s.
SHORT_TOK_S, STREAM_TOK_S, PREFILL_TOK_S, LONG_TOK_S = 850, 110, 13000, 600
# Prompt tokens of a generated retrieval item per token of its id, measured.
LONG_PROMPT_RATIO = 1.03


def draw(data: quality.Datasets) -> dict:
    gpqa = sorted(random.Random(SEED).sample(range(len(data.rows("gpqa"))), GPQA_ITEMS))
    number = lambda key: int(key) if key.isdigit() else key
    ids = ([f"gpqa/{r}" for r in gpqa]
           + [f"code/{k}" for k in sorted(data.rows("code"), key=number)]
           + [f"aime/{k}" for k in sorted(data.rows("aime"), key=number)]
           + [f"long/{t}/{s}" for t, seeds in LONG.items() for s in seeds]
           + [f"tool/{k}" for k in quality.TOOLS])
    return {"drawn": f"{GPQA_ITEMS} gpqa_diamond rows sampled with random.Random({SEED}); "
                     "every HumanEval+ task and AIME 2025 problem; generated retrieval "
                     "items; every tool check. Dataset revisions are in fetch.DATASETS.",
            "tokens": TOKENS, "items": ids}


def estimate(ids, tokens: dict, is_long, long_streams: int) -> list[str]:
    """A markdown table of the expected tokens and minutes per source, and the total."""
    rows, short_tokens, long_tokens, prefill = {}, 0, 0, 0
    for item in ids:
        source = item.split("/")[0]
        n = tokens.get(source, quality.CAPS[source] // 4)
        count, total = rows.get(source, (0, 0))
        rows[source] = (count + 1, total + n)
        if is_long(item):
            long_tokens += n
            prefill += int(item.split("/")[1]) * LONG_PROMPT_RATIO
        else:
            short_tokens += n
    short_s = short_tokens / SHORT_TOK_S
    long_s = prefill / PREFILL_TOK_S + long_tokens / (LONG_TOK_S * long_streams / 2)
    lines = ["| Source | Items | Tokens per reply | Tokens |", "| --- | --- | --- | --- |"]
    for source, (count, total) in rows.items():
        lines.append(f"| {source} | {count} | {total // count:,} | {total:,} |")
    lines.append(f"| all | {sum(c for c, _ in rows.values())} | | "
                 f"{short_tokens + long_tokens:,} |")
    lines += ["", (f"Estimated wall time: {short_s / 60:.0f} min at 8 streams "
              f"({short_tokens:,} tokens at {SHORT_TOK_S} tok/s), then {long_s / 60:.0f} min "
              f"for the long-context items, {(short_s + long_s) / 60:.0f} min in all, plus "
              f"server starts. One reply that runs to the {quality.CAPS['gpqa']:,}-token cap "
              f"takes {quality.CAPS['gpqa'] / STREAM_TOK_S / 60:.0f} min at {STREAM_TOK_S} "
              "tok/s, so a late cut-off can add that much.")]
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", default=str(Path.home() / ".cache" / "quickbench"))
    parser.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "suite.json"))
    args = parser.parse_args()
    suite = draw(quality.Datasets(Path(args.cache)))
    Path(args.out).write_text(json.dumps(suite, indent=1) + "\n")
    print(f"wrote {len(suite['items'])} items to {args.out}")


if __name__ == "__main__":
    main()
