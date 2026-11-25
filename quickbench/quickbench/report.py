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


def quality_table(records) -> list[str]:
    groups = defaultdict(list)
    for r in records:
        groups[source(r)].append(r)
    lines = ["| Source | Items | Replies | Passed | Pass rate | 95% interval | Cut off | Looping | Errors | Median tokens |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, rs in [*sorted(groups.items()), ("all", records)]:
        k, n = sum(r["passed"] for r in rs), len(rs)
        lo, hi = wilson(k, n)
        lines.append(
            f"| {name} | {len({r['item'] for r in rs})} | {n} | {k} | {k / n:.1%} | "
            f"{lo:.1%} to {hi:.1%} | {sum(r['finish_reason'] == 'length' for r in rs)} | "
            f"{sum(r['looping'] for r in rs)} | {sum(r['error'] is not None for r in rs)} | "
            f"{fmt(median(r['completion_tokens'] for r in rs), '{:.0f}')} |")
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
    lines = [f"# Quickbench {meta['name']}", "",
             f"- Target: {meta['target']}",
             f"- Commit: {meta.get('commit') or 'unknown'}",
             f"- Started: {meta['started']}",
             f"- Quality sampling: {meta['sampling']}, {meta['attempts']} attempts per item, "
             f"{meta['concurrency']} at a time",
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


# Fewer items than this get no verdict: their bootstrap interval is too coarse.
MIN_ITEMS = 5


def bootstrap(values, statistic=statistics.fmean, rounds: int = 10000, seed: int = 0):
    """The statistic of the values, with a 95% interval from resampling them."""
    rng = random.Random(seed)
    stats = sorted(statistic(rng.choices(values, k=len(values))) for _ in range(rounds))
    return statistic(values), stats[int(0.025 * rounds)], stats[int(0.975 * rounds) - 1]


def mcnemar(a_only: int, b_only: int) -> float:
    n = a_only + b_only
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(a_only, b_only) + 1)) / 2**n
    return min(1.0, 2 * tail)


def compare(a: dict, b: dict) -> str:
    lines = [f"# Quickbench compare: A = {a['meta']['name']}, B = {b['meta']['name']}", "",
             f"- A: {a['meta']['target']}", f"- B: {b['meta']['target']}", ""]
    qa = {(r["item"], r["attempt"]): r for r in a.get("quality", [])}
    qb = {(r["item"], r["attempt"]): r for r in b.get("quality", [])}
    common = sorted(set(qa) & set(qb))
    if common:
        lines += ["## Quality, paired by item", "",
                  "B − A is the mean over items of the difference in pass rate. The interval "
                  "comes from a paired bootstrap over items (10,000 resamples). Attempts are "
                  "paired by seed and option order.", "",
                  "| Source | Items | A | B | B − A | 95% interval | Attempts A only / B only | McNemar p | Verdict |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        by_source = defaultdict(list)
        for key in common:
            by_source[key[0].split("/")[0]].append(key)
        for name, keys in [*sorted(by_source.items()), ("all", common)]:
            items = defaultdict(lambda: [[], []])
            for key in keys:
                items[key[0]][0].append(qa[key]["passed"])
                items[key[0]][1].append(qb[key]["passed"])
            diffs = [sum(y) / len(y) - sum(x) / len(x) for x, y in items.values()]
            mean, lo, hi = bootstrap(diffs)
            a_only = sum(qa[k]["passed"] and not qb[k]["passed"] for k in keys)
            b_only = sum(qb[k]["passed"] and not qa[k]["passed"] for k in keys)
            pa = sum(qa[k]["passed"] for k in keys) / len(keys)
            pb = sum(qb[k]["passed"] for k in keys) / len(keys)
            verdict = ("too few items" if len(items) < MIN_ITEMS
                       else "real" if lo > 0 or hi < 0 else "within noise")
            lines.append(f"| {name} | {len(items)} | {pa:.1%} | {pb:.1%} | {mean:+.1%} | "
                         f"{lo:+.1%} to {hi:+.1%} | {a_only} / {b_only} | "
                         f"{mcnemar(a_only, b_only):.3f} | {verdict} |")
        for label, field in (("cut off", "length"), ("looping", None)):
            ca = sum((qa[k]["finish_reason"] == field) if field else qa[k]["looping"] for k in common)
            cb = sum((qb[k]["finish_reason"] == field) if field else qb[k]["looping"] for k in common)
            lines.append(f"\nReplies {label}: A {ca}, B {cb} of {len(common)}.")
        lines.append("")
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
