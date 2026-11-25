"""Statistics, the markdown report of one run, and the paired comparison of two runs."""

import math
import random
import statistics
from collections import defaultdict


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def median(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def fmt(x, spec="{:.1f}"):
    return "-" if x is None else spec.format(x)


def source(record) -> str:
    return record["item"].split("/")[0]


def costs(rs) -> dict:
    """Cut-offs, loops and the token and time cost of a group of replies."""
    n, k = len(rs), sum(r["passed"] for r in rs)
    tokens = [r["completion_tokens"] or 0 for r in rs]
    seconds = [r["seconds"] for r in rs]
    return {"n": n, "passed": k, "cut": sum(r["finish_reason"] == "length" for r in rs),
            "looping": sum(r["looping"] for r in rs),
            "errors": sum(r["error"] is not None for r in rs),
            "tokens": statistics.fmean(tokens), "median_tokens": statistics.median(tokens),
            "seconds": statistics.fmean(seconds),
            "seconds_per_pass": sum(seconds) / k if k else None,
            "tokens_per_pass": sum(tokens) / k if k else None}


def by_source(records) -> list[tuple[str, list]]:
    groups = defaultdict(list)
    for r in records:
        groups[source(r)].append(r)
    return [*sorted(groups.items()), ("all", records)]


def quality_table(records) -> list[str]:
    lines = ["| Source | Items | Passed | 95% interval | Cut off | Looping | Errors | Mean tokens | Median tokens | Seconds per reply | Seconds per pass | Tokens per pass |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, rs in by_source(records):
        c = costs(rs)
        lo, hi = wilson(c["passed"], c["n"])
        lines.append(
            f"| {name} | {len({r['item'] for r in rs})} | {c['passed']}/{c['n']}, "
            f"{c['passed'] / c['n']:.1%} | {lo:.1%} to {hi:.1%} | "
            f"{c['cut']}, {c['cut'] / c['n']:.1%} | {c['looping']}, {c['looping'] / c['n']:.1%} | "
            f"{c['errors']} | {c['tokens']:,.0f} | {c['median_tokens']:,.0f} | "
            f"{c['seconds']:.1f} | {fmt(c['seconds_per_pass'])} | "
            f"{fmt(c['tokens_per_pass'], '{:,.0f}')} |")
    return lines


def speed_tables(records) -> list[str]:
    lines = []
    for streams in (1, 8):
        rs = [r for r in records if r["streams"] == streams and r["error"] is None]
        if not rs:
            continue
        groups = defaultdict(list)
        for r in rs:
            groups[(r["prompt"], r["sampling"])].append(r)
        lines += [f"{streams} stream{'s' if streams > 1 else ''}, medians over seeds:", "",
                  "| Prompt | Sampling | Replies | Decode tok/s | TTFT ms | Tokens per round | Tokens |",
                  "| --- | --- | --- | --- | --- | --- | --- |"]
        for (prompt, sampling), g in groups.items():
            lines.append(
                f"| {prompt} | {sampling} | {len(g)} | {fmt(median(r['decode_tok_s'] for r in g))} | "
                f"{fmt(median(r['ttft'] * 1e3 for r in g), '{:.0f}')} | "
                f"{fmt(median(r.get('tokens_per_round') for r in g), '{:.2f}')} | "
                f"{fmt(median(r['completion_tokens'] for r in g), '{:.0f}')} |")
        long = defaultdict(list)
        for r in rs:
            if r.get("ttft_warm") is not None:
                long[(r["prompt"], r["sampling"])].append(r)
        for (prompt, sampling), g in long.items():
            lines.append(
                f"\n{prompt} at {sampling} ({fmt(median(r['prompt_tokens'] for r in g), '{:,.0f}')} "
                f"prompt tokens): median TTFT {median(r['ttft'] for r in g) * 1e3:.0f} ms on a "
                f"prefix-cache miss, {median(r['ttft_warm'] for r in g) * 1e3:.0f} ms when the "
                "same request repeats.")
        if streams == 8:
            total = sum(r["completion_tokens"] for r in rs)
            lines.append(f"\nAll {len(rs)} requests, 8 at a time: {total} tokens in "
                         f"{rs[0]['wall']:.1f} s, {total / rs[0]['wall']:.0f} tok/s.")
        lines.append("")
    errors = sum(r["error"] is not None for r in records)
    if errors:
        lines.append(f"{errors} speed requests failed.")
    return lines


def fidelity_table(records) -> list[str]:
    lines = ["| Text | Config | Tokens | Mean KL | Max KL | Top-1 | Mean \\|dlogprob\\| | Perplexity / MLX | Continuation | Verdict | Seconds |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in records:
        if "verdict" not in r:
            lines.append(f"| {r['text']} | {r.get('config', '-')} | error: {r.get('error')} |"
                         + " |" * 8)
            continue
        lines.append(
            f"| {r['text']} | `{r['config']}` | {r['tokens']} | {r['kl_mean']:.2e} | "
            f"{r['kl_max']:.2e} | {r['top1']:.2%} | {r['logprob_diff']:.4f} | "
            f"{r['ppl']:.4f} / {r['ppl_mlx']:.4f} | {r['continuation']} | {r['verdict']} | "
            f"{r['seconds']:.0f} |")
    return lines


def render(run: dict) -> str:
    meta = run["meta"]
    title = "Quickbench suite" if meta.get("mode") == "suite" else "Quickbench"
    lines = [f"# {title} {meta['name']}", "",
             f"- Target: {meta['target']}",
             f"- Commit: {meta.get('commit') or 'unknown'}",
             f"- Started: {meta['started']}",
             f"- Quality sampling: {meta['sampling']}, {meta['attempts']} "
             f"attempt{'s' if meta['attempts'] != 1 else ''} per item, "
             f"{meta['concurrency']} at a time, output caps {meta.get('caps')}",
             "- Wall time: " + ", ".join(f"{k} {v / 60:.1f} min" for k, v in run["timings"].items()),
             ""]
    for track, apps in run.get("gpu_others", {}).items():
        if apps:
            names = ", ".join(f"{a['name']} (pid {a['pid']}, {a['mib']} MiB)" for a in apps)
            lines += [f"**Warning:** other GPU processes during {track}: {names}. "
                      "Discard its numbers.", ""]
    if run.get("quality"):
        lines += ["## Quality", "", *quality_table(run["quality"]), ""]
    if run.get("speed"):
        lines += ["## Speed", "", *speed_tables(run["speed"])]
    if run.get("fidelity"):
        lines += ["## Fidelity", "", *fidelity_table(run["fidelity"]), ""]
    return "\n".join(lines) + "\n"


def bootstrap(values, statistic=statistics.fmean, rounds: int = 10000, seed: int = 0):
    """The statistic of the values, with a 95% interval from resampling them."""
    rng = random.Random(seed)
    stats = sorted(statistic(rng.choices(values, k=len(values))) for _ in range(rounds))
    return statistic(values), stats[int(0.025 * rounds)], stats[int(0.975 * rounds) - 1]


def sign_test(a_better: int, b_better: int) -> float:
    """Two-sided exact sign test over items, ties left out.

    It counts each item once, so a side that repeats one reply on every attempt (greedy
    decoding of an item whose prompt does not change) is not counted as several samples.
    """
    n = a_better + b_better
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(a_better, b_better) + 1)) / 2**n
    return min(1.0, 2 * tail)


# Items listed by name when replies differ.
LISTED = 30


def identical(qa: dict, qb: dict, common: list) -> list[str]:
    """Replies that repeat byte for byte with equal token counts, and the items that do not.

    A seeded run on an unchanged build repeats every reply, so a changed reply shows a
    change in numerics long before pass rates move.
    """
    keys = [k for k in common if qa[k].get("reply_hash") and qb[k].get("reply_hash")]
    if not keys:
        return ["Identical replies: not known, a run has neither reply hashes nor replies.jsonl."]
    differ = defaultdict(list)
    for k in keys:
        if (qa[k]["reply_hash"], qa[k]["completion_tokens"]) != (
                qb[k]["reply_hash"], qb[k]["completion_tokens"]):
            differ[k[0]].append(k[1])
    same = len(keys) - sum(map(len, differ.values()))
    lines = [f"Identical replies (reasoning, content and tool calls byte for byte, equal token "
             f"counts): {same} of {len(keys)}."]
    if differ:
        items = sorted(differ)
        more = f", and {len(items) - LISTED} more" if len(items) > LISTED else ""
        lines.append(f"Items with a reply that differs ({len(items)}): "
                     + ", ".join(items[:LISTED]) + more + ".")
    return lines


def compare(a: dict, b: dict) -> str:
    lines = [f"# Quickbench compare: A = {a['meta']['name']}, B = {b['meta']['name']}", "",
             *(f"- {side}: {run['meta']['target']}, commit {run['meta'].get('commit')}, "
               f"sampling {run['meta'].get('sampling')}" for side, run in (("A", a), ("B", b))),
             ""]
    qa = {(r["item"], r["attempt"]): r for r in a.get("quality", [])}
    qb = {(r["item"], r["attempt"]): r for r in b.get("quality", [])}
    common = sorted(set(qa) & set(qb))
    if common:
        lines += ["## Quality, paired by item", "",
                  ("B − A is the mean over items of the difference in pass rate. The interval "
                   "comes from a paired bootstrap over items (10,000 resamples). Attempts are "
                   "paired by seed and option order. The sign test counts the items each side "
                   "passed more often."), "",
                  "| Source | Items | A | B | B − A | 95% interval | Items A better / B better | Sign test p |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        groups = by_source([{"item": k[0], "key": k} for k in common])
        for name, rows in groups:
            keys = [r["key"] for r in rows]
            items = defaultdict(lambda: [[], []])
            for key in keys:
                items[key[0]][0].append(qa[key]["passed"])
                items[key[0]][1].append(qb[key]["passed"])
            diffs = [sum(y) / len(y) - sum(x) / len(x) for x, y in items.values()]
            mean, lo, hi = bootstrap(diffs)
            a_better = sum(d < 0 for d in diffs)
            b_better = sum(d > 0 for d in diffs)
            pa = sum(qa[k]["passed"] for k in keys) / len(keys)
            pb = sum(qb[k]["passed"] for k in keys) / len(keys)
            lines.append(f"| {name} | {len(items)} | {pa:.1%} | {pb:.1%} | {mean:+.1%} | "
                         f"{lo:+.1%} to {hi:+.1%} | {a_better} / {b_better} | "
                         f"{sign_test(a_better, b_better):.3f} |")
        lines += ["", "Over the same replies, A / B:", "",
                  "| Source | Cut off | Looping | Mean tokens | Seconds per reply | Seconds per pass | Tokens per pass |",
                  "| --- | --- | --- | --- | --- | --- | --- |"]
        for name, rows in groups:
            x = costs([qa[r["key"]] for r in rows])
            y = costs([qb[r["key"]] for r in rows])
            lines.append(
                f"| {name} | {x['cut']} / {y['cut']} | {x['looping']} / {y['looping']} | "
                f"{x['tokens']:,.0f} / {y['tokens']:,.0f} | {x['seconds']:.1f} / {y['seconds']:.1f} | "
                f"{fmt(x['seconds_per_pass'])} / {fmt(y['seconds_per_pass'])} | "
                f"{fmt(x['tokens_per_pass'], '{:,.0f}')} / {fmt(y['tokens_per_pass'], '{:,.0f}')} |")
        lines += ["", *identical(qa, qb, common), ""]
    sa = {(r["prompt"], r["sampling"], r["seed"], r["streams"]): r for r in a.get("speed", [])}
    sb = {(r["prompt"], r["sampling"], r["seed"], r["streams"]): r for r in b.get("speed", [])}
    keys = [k for k in sorted(set(sa) & set(sb))
            if sa[k]["decode_tok_s"] and sb[k]["decode_tok_s"]]
    if keys:
        lines += ["## Speed", "", "Medians over seeds. B / A is the median of per-seed ratios.", "",
                  "| Prompt | Sampling | Streams | A tok/s | B tok/s | B / A |",
                  "| --- | --- | --- | --- | --- | --- |"]
        groups = defaultdict(list)
        for k in keys:
            groups[(k[0], k[1], k[3])].append(k)
        for (prompt, sampling, streams), ks in sorted(groups.items(), key=lambda g: g[0][2]):
            lines.append(
                f"| {prompt} | {sampling} | {streams} | {median(sa[k]['decode_tok_s'] for k in ks):.1f} | "
                f"{median(sb[k]['decode_tok_s'] for k in ks):.1f} | "
                f"{median(sb[k]['decode_tok_s'] / sa[k]['decode_tok_s'] for k in ks):.3f} |")
        long = defaultdict(list)
        for k in keys:
            if sa[k].get("ttft_warm") is not None and sb[k].get("ttft_warm") is not None:
                long[k[:2]].append(k)
        for (prompt, sampling), warm in long.items():
            lines.append(
                f"\n{prompt} at {sampling}, TTFT medians over seeds: on a prefix-cache miss "
                f"A {median(sa[k]['ttft'] for k in warm) * 1e3:.0f} ms, "
                f"B {median(sb[k]['ttft'] for k in warm) * 1e3:.0f} ms; on a hit "
                f"A {median(sa[k]['ttft_warm'] for k in warm) * 1e3:.0f} ms, "
                f"B {median(sb[k]['ttft_warm'] for k in warm) * 1e3:.0f} ms.")
        for streams in (1, 8):
            ks = [k for k in keys if k[3] == streams]
            if ks:
                ratios = [sb[k]["decode_tok_s"] / sa[k]["decode_tok_s"] for k in ks]
                mid, lo, hi = bootstrap(ratios, statistics.median)
                lines.append(f"\n{streams} stream{'s' if streams > 1 else ''}, all prompts: "
                             f"median B / A {mid:.3f}, 95% interval {lo:.3f} to {hi:.3f} "
                             f"(bootstrap over {len(ks)} requests).")
        lines.append("")
    fa = {(r["text"], r.get("config")): r for r in a.get("fidelity", []) if "verdict" in r}
    fb = {(r["text"], r.get("config")): r for r in b.get("fidelity", []) if "verdict" in r}
    if common_f := [k for k in fa if k in fb]:
        lines += ["## Fidelity", "", "| Text | Config | Mean KL A | Mean KL B | Top-1 A | Top-1 B | Perplexity A | Perplexity B |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for k in common_f:
            x, y = fa[k], fb[k]
            lines.append(f"| {k[0]} | `{k[1]}` | {x['kl_mean']:.2e} | {y['kl_mean']:.2e} | "
                         f"{x['top1']:.2%} | {y['top1']:.2%} | {x['ppl']:.4f} | {y['ppl']:.4f} |")
        lines.append("")
    return "\n".join(lines) + "\n"
