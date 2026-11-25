"""quickbench: fetch | run | compare. See quickbench/README.md."""

import argparse
import datetime
import json
import shlex
import subprocess
import threading
import time
from pathlib import Path

from . import client, fetch, fidelity, quality, report, serve, speed

HERE = Path(__file__).resolve().parent.parent
CACHE = Path.home() / ".cache" / "quickbench"
# siphon-serve contexts: 8 streams for everything up to 32K-token prompts, then one
# stream for the 128K-token prompts.
SHORT_CONTEXT, LONG_CONTEXT = 40960, 135168


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


def run(args) -> None:
    for cap in args.cap:
        source, tokens = cap.split("=")
        quality.CAPS[source] = int(tokens)
    name = args.name or datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = Path(args.out or CACHE / "runs" / name)
    out.mkdir(parents=True, exist_ok=True)
    tracks = set(args.tracks.split(","))
    spec = json.loads(Path(args.items).read_text())
    items = [i["id"] for i in spec["items"]]
    expected = {i["id"]: i.get("tokens", 0) for i in spec["items"]}
    sampling = {"temperature": args.temperature} | ({"top_p": args.top_p} if args.top_p else {})
    extra = shlex.split(args.serve_args)
    target = (f"siphon-serve {args.serve_args}".strip() if args.serve else args.url)
    result = {"meta": {"name": name, "target": target, "commit": args.commit or commit(),
                       "started": datetime.datetime.now().isoformat(timespec="seconds"),
                       "sampling": sampling, "attempts": args.attempts,
                       "concurrency": args.concurrency, "items": args.items,
                       "caps": quality.CAPS},
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
            phases.append((long, ["--streams", "1", "--context", str(LONG_CONTEXT)], 1))
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
                        server.base, model, args.seeds, args.speed_tokens, server), own)
                if "quality" in tracks and ids:
                    result["quality"] += timed("quality", lambda: quality.run(
                        server.base, model, ids, args.attempts, sampling, concurrency, data,
                        expected, log, args.first_attempt), own)
    elif args.url:
        model = client.get(f"{args.url}/models")["data"][0]["id"]
        if "speed" in tracks:
            result["speed"] = timed("speed", lambda: speed.run(
                args.url, model, args.seeds, args.speed_tokens))
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
    return json.loads((p / "results.json" if p.is_dir() else p).read_text())


def main() -> None:
    parser = argparse.ArgumentParser(prog="quickbench", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    f = sub.add_parser("fetch", help="download the datasets into the cache")
    f.add_argument("--cache", default=str(CACHE))
    r = sub.add_parser("run", help="run the tracks and write a report")
    target = r.add_mutually_exclusive_group(required=True)
    target.add_argument("--url", help="an OpenAI-compatible base URL, such as http://host:8000/v1")
    target.add_argument("--serve", help="a siphon-serve binary to start (needs --model)")
    target.add_argument("--eval-only", action="store_true", help="run only the fidelity track")
    r.add_argument("--model", help="checkpoint directory, for --serve and --eval")
    r.add_argument("--serve-args", default="", help="extra siphon-serve options, such as '--kv bf16'")
    r.add_argument("--eval", help="Siphon's eval binary, for the fidelity track")
    r.add_argument("--fidelity-config", action="append",
                   help="eval options, repeatable (default: fidelity.CONFIGS, the served prefill and verify)")
    r.add_argument("--refs", default=str(CACHE / "refs"), help="directory of the MLX captures")
    r.add_argument("--tracks", default="quality,speed,fidelity")
    r.add_argument("--items", default=str(HERE / "items.json"))
    r.add_argument("--attempts", type=int, default=4)
    # Attempts 0 to 3 picked the items, so their sampled replies are biased towards
    # failing. Scoring starts after them.
    r.add_argument("--first-attempt", type=int, default=4,
                   help="number the attempts from here (0 to 3 calibrated items.json)")
    r.add_argument("--concurrency", type=int, default=8)
    r.add_argument("--temperature", type=float, default=1.0)
    r.add_argument("--top-p", type=float)
    r.add_argument("--cap", action="append", default=[], metavar="SOURCE=TOKENS",
                   help="override an output cap, such as gpqa=12288")
    r.add_argument("--seeds", type=int, default=6)
    r.add_argument("--speed-tokens", type=int, default=1024)
    r.add_argument("--cache", default=str(CACHE))
    r.add_argument("--name")
    r.add_argument("--out")
    r.add_argument("--commit", help="the engine commit, when git cannot tell")
    r.add_argument("--no-wait", action="store_true",
                   help="start even when other processes use the GPU")
    c = sub.add_parser("compare", help="pair two runs item by item")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--out")
    args = parser.parse_args()
    if args.command == "fetch":
        fetch.fetch(Path(args.cache))
    elif args.command == "run":
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
