"""quickbench: fetch | run | suite | compare. See quickbench/README.md."""

import argparse
import datetime
import json
import shlex
import subprocess
import threading
import time
from pathlib import Path

from . import client, fetch, fidelity, quality, report, serve, speed, suite

HERE = Path(__file__).resolve().parent.parent
CACHE = Path.home() / ".cache" / "quickbench"
# siphon serve contexts: 8 streams for everything up to 32K-token prompts, then a second
# server for the 128K-token prompts: 1 stream in the build check, and 2 streams in the
# suite, whose 32K caps also move the 30K-token prompts there. 2 x 163,840 positions are
# as many as 8 x 40,960.
SHORT_CONTEXT = 40960
LONG_SERVER = {"check": (1, 135168), "suite": (2, 163840)}


def needs_long_context(item_id: str) -> bool:
    source, key = item_id.split("/", 1)
    return source == "long" and int(key.split("/")[0]) > SHORT_CONTEXT - quality.CAPS["long"] - 2048


def wait_for_gpu(own: set[int], wait: bool) -> list[dict]:
    while True:
        others = serve.others(serve.gpu_apps(), own)
        if not others or not wait:
            return others
        names = ", ".join(f"{a['name']} (pid {a['pid']})" for a in others)
        print(f"waiting for other GPU processes to finish: {names}", flush=True)
        time.sleep(30)


def commit() -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(HERE), "describe", "--always", "--dirty"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except OSError:
        return None


def effort(value: str):
    """`--effort` as siphon serve takes it: a number, or a name such as high."""
    try:
        return float(value)
    except ValueError:
        return value


def load_items(path: str) -> tuple[list[str], dict]:
    """Item ids and expected reply tokens per id, from items.json or suite.json."""
    spec = json.loads(Path(path).read_text())
    per_source = spec.get("tokens", {})
    ids, expected = [], {}
    for item in spec["items"]:
        item_id = item if isinstance(item, str) else item["id"]
        ids.append(item_id)
        expected[item_id] = (item.get("tokens") if isinstance(item, dict) else None) \
            or per_source.get(item_id.split("/")[0], 0)
    return ids, expected


def set_caps(args) -> None:
    if args.mode == "suite":
        for source in quality.CAPS:
            quality.CAPS[source] = quality.SUITE_CAP
    for cap in args.cap:
        source, tokens = cap.split("=")
        quality.CAPS[source] = int(tokens)


def run(args) -> None:
    set_caps(args)
    name = args.name or datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = Path(args.out or CACHE / "runs" / name)
    out.mkdir(parents=True, exist_ok=True)
    tracks = set(args.tracks.split(","))
    items, expected = load_items(args.items)
    sampling = ({"temperature": args.temperature} | ({"top_p": args.top_p} if args.top_p else {})
                | ({"reasoning_effort": effort(args.effort)} if args.effort else {}))
    extra = shlex.split(args.serve_args)
    target = (f"siphon serve {args.serve_args}".strip() if args.serve else args.url)
    result = {"meta": {"name": name, "mode": args.mode, "target": target,
                       "commit": args.commit or commit(),
                       "started": datetime.datetime.now().isoformat(timespec="seconds"),
                       "sampling": sampling, "attempts": args.attempts,
                       "first_attempt": args.first_attempt,
                       "concurrency": args.concurrency, "items": args.items,
                       "caps": quality.CAPS, "speed_samplings": args.speed_sampling},
              "timings": {}, "gpu_others": {}, "quality": [], "speed": [], "fidelity": []}
    data = quality.Datasets(Path(args.cache))
    lock = threading.Lock()
    replies = (out / "replies.jsonl").open("w")

    def log(record):
        with lock:
            replies.write(json.dumps(record) + "\n")
            replies.flush()

    def timed(track, fn, own=frozenset()):
        before = wait_for_gpu(set(own), False)
        started = time.perf_counter()
        value = fn()
        result["timings"][track] = result["timings"].get(track, 0) + time.perf_counter() - started
        after = serve.others(serve.gpu_apps(), set(own))
        seen = {a["pid"]: a for a in result["gpu_others"].get(track, []) + before + after}
        result["gpu_others"][track] = list(seen.values())
        print(f"{track}: {time.perf_counter() - started:.0f} s", flush=True)
        return value

    short = [i for i in items if not needs_long_context(i)]
    long = [i for i in items if needs_long_context(i)]
    if args.serve:
        wait_for_gpu(set(), not args.no_wait)
        phases = [(short, ["--streams", "8", "--context", str(SHORT_CONTEXT)], args.concurrency)]
        if long and "quality" in tracks:
            streams, context = LONG_SERVER[args.mode]
            phases.append((long, ["--streams", str(streams), "--context", str(context)], streams))
        for n, (ids, options, concurrency) in enumerate(phases):
            started = time.perf_counter()
            server = serve.Siphon(Path(args.serve), Path(args.model), options + extra,
                                  out / f"serve-{n + 1}.log")
            with server:
                result["timings"]["server start"] = (result["timings"].get("server start", 0)
                                                     + time.perf_counter() - started)
                own = {server.process.pid}
                model = client.get(f"{server.base}/models")["data"][0]["id"]
                if n == 0 and "speed" in tracks:
                    result["speed"] = timed("speed", lambda: speed.run(
                        server.base, model, args.seeds, args.speed_tokens, server,
                        args.speed_sampling), own)
                if "quality" in tracks and ids:
                    result["quality"] += timed("quality", lambda: quality.run(
                        server.base, model, ids, args.attempts, sampling, concurrency, data,
                        expected, log, args.first_attempt), own)
    elif args.url:
        model = client.get(f"{args.url}/models")["data"][0]["id"]
        if "speed" in tracks:
            result["speed"] = timed("speed", lambda: speed.run(
                args.url, model, args.seeds, args.speed_tokens, None, args.speed_sampling))
        if "quality" in tracks:
            result["quality"] = timed("quality", lambda: quality.run(
                args.url, model, items, args.attempts, sampling, args.concurrency, data,
                expected, log, args.first_attempt))
    if args.eval and "fidelity" in tracks:
        wait_for_gpu(set(), not args.no_wait)
        result["fidelity"] = timed("fidelity", lambda: fidelity.run(
            Path(args.eval), Path(args.model), Path(args.refs),
            args.fidelity_config or fidelity.CONFIGS, out / "fidelity"))
    result["timings"]["total"] = sum(result["timings"].values())
    (out / "results.json").write_text(json.dumps(result, indent=1))
    text = report.render(result)
    (out / "report.md").write_text(text)
    print(text)
    print(f"wrote {out}/report.md and results.json")


def load(path: str) -> dict:
    p = Path(path)
    run = json.loads((p / "results.json" if p.is_dir() else p).read_text())
    # Runs from before reply hashes were recorded: hash their replies.jsonl.
    replies = (p if p.is_dir() else p.parent) / "replies.jsonl"
    missing = [r for r in run.get("quality", []) if "reply_hash" not in r]
    if missing and replies.exists():
        hashes = {}
        for line in replies.open():
            reply = json.loads(line)
            hashes[(reply["item"], reply["attempt"])] = quality.reply_hash(reply)
        for r in missing:
            if (key := (r["item"], r["attempt"])) in hashes:
                r["reply_hash"] = hashes[key]
    return run


def quality_options(p, items: Path, attempts: int, first_attempt: int, first_help: str) -> None:
    p.add_argument("--model", help="checkpoint directory, for --serve and --eval")
    p.add_argument("--serve-args", default="", help="extra siphon serve options, such as '--kv bf16'")
    p.add_argument("--items", default=str(items))
    p.add_argument("--attempts", type=int, default=attempts)
    p.add_argument("--first-attempt", type=int, default=first_attempt, help=first_help)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float)
    p.add_argument("--effort", help="sent as reasoning_effort: a number, or a name such as high")
    p.add_argument("--cap", action="append", default=[], metavar="SOURCE=TOKENS",
                   help="override an output cap, such as gpqa=12288")
    p.add_argument("--cache", default=str(CACHE))
    p.add_argument("--name")
    p.add_argument("--out")
    p.add_argument("--commit", help="the engine commit, when git cannot tell")
    p.add_argument("--no-wait", action="store_true",
                   help="start even when other processes use the GPU")


def main() -> None:
    parser = argparse.ArgumentParser(prog="quickbench", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    f = sub.add_parser("fetch", help="download the datasets into the cache")
    f.add_argument("--cache", default=str(CACHE))
    r = sub.add_parser("run", help="the build check: speed, quality and fidelity of one build")
    target = r.add_mutually_exclusive_group(required=True)
    target.add_argument("--url", help="an OpenAI-compatible base URL, such as http://host:8000/v1")
    target.add_argument("--serve", help="a siphon binary to run with serve (needs --model)")
    target.add_argument("--eval-only", action="store_true", help="run only the fidelity track")
    # Attempts 0 to 3 picked the items, so their sampled replies are biased towards
    # failing. Scoring starts after them.
    quality_options(r, HERE / "items.json", 1, 4,
                    "number the attempts from here (0 to 3 calibrated items.json)")
    r.add_argument("--eval", help="Siphon's eval binary, for the fidelity track")
    r.add_argument("--fidelity-config", action="append",
                   help="eval options, repeatable (default: fidelity.CONFIGS, the served prefill and verify)")
    r.add_argument("--refs", default=str(CACHE / "refs"), help="directory of the MLX captures")
    r.add_argument("--tracks", default="quality,speed,fidelity")
    r.add_argument("--seeds", type=int, default=6)
    r.add_argument("--speed-tokens", type=int, default=1024)
    r.add_argument("--speed-sampling", action="append", choices=list(speed.SAMPLINGS),
                   help="a speed sampling, repeatable (default: t1.0 only)")
    s = sub.add_parser("suite", help="the quality suite, for choosing how to run the model")
    target = s.add_mutually_exclusive_group()
    target.add_argument("--url", help="an OpenAI-compatible base URL, such as http://host:8000/v1")
    target.add_argument("--serve", help="a siphon binary to run with serve (needs --model)")
    quality_options(s, HERE / "suite.json", 1, 0, "number the attempts from here")
    s.add_argument("--dry-run", action="store_true",
                   help="print the items and the estimated tokens and wall time, then stop")
    c = sub.add_parser("compare", help="pair two runs item by item")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--out")
    args = parser.parse_args()
    if args.command == "fetch":
        fetch.fetch(Path(args.cache))
    elif args.command in ("run", "suite"):
        args.mode = "check" if args.command == "run" else "suite"
        if args.mode == "suite":
            args.tracks, args.eval, args.speed_sampling = "quality", None, []
            if args.dry_run:
                set_caps(args)
                ids, _ = load_items(args.items)
                tokens = json.loads(Path(args.items).read_text()).get("tokens", {})
                print("\n".join(suite.estimate(ids, tokens, needs_long_context,
                                               LONG_SERVER["suite"][0])))
                return
            if not (args.url or args.serve):
                parser.error("suite needs --url or --serve")
        else:
            args.speed_sampling = args.speed_sampling or list(speed.DEFAULT_SAMPLINGS)
        if (args.serve or args.eval) and not args.model:
            parser.error("--serve and --eval need --model")
        run(args)
    else:
        text = report.compare(load(args.a), load(args.b))
        if args.out:
            Path(args.out).write_text(text)
        print(text)


if __name__ == "__main__":
    main()
