"""Fidelity track: drives Siphon's `eval` binary over stored MLX captures."""

import re
import shlex
import subprocess
import time
from pathlib import Path

TEXTS = ("fidelity-4k", "fidelity-32k", "fidelity-128k")
# What siphon-serve runs by default: prefill in chunks of 2,048 rows, and the verify pass,
# which every decoded token goes through, on FP8 weights. `eval --prefill` keeps BF16 weights
# whatever `--weights` says, so only the verify config measures the FP8 copies.
CONFIGS = ("--prefill 2048 --kv fp8", "--verify 8 --kv fp8 --weights fp8")

ALL = re.compile(r"^all\s+\S+\s+(\d+)\s+([\d.]+)%\s+([\d.]+)\s+([\d.]+) / ([\d.]+)", re.M)
KL = re.compile(r"KL\(MLX \|\| engine\) at (\d+) sampled positions: mean (\S+) median (\S+) max (\S+)")
CONTINUATION = re.compile(r"greedy continuation matches MLX for (\d+)/(\d+)")


def parse(text: str) -> dict:
    out = {}
    if m := ALL.search(text):
        out |= {"tokens": int(m.group(1)), "top1": float(m.group(2)) / 100,
                "logprob_diff": float(m.group(3)), "ppl": float(m.group(4)),
                "ppl_mlx": float(m.group(5))}
    if m := KL.search(text):
        out |= {"kl_mean": float(m.group(2)), "kl_median": float(m.group(3)),
                "kl_max": float(m.group(4))}
    if m := CONTINUATION.search(text):
        out["continuation"] = f"{m.group(1)}/{m.group(2)}"
    if m := re.search(r"verdict: (\w+)", text):
        out["verdict"] = m.group(1)
    return out


def run(binary: Path, model: Path, refs: Path, configs, logs: Path, on_start=None) -> list[dict]:
    logs.mkdir(parents=True, exist_ok=True)
    out = []
    for text in TEXTS:
        capture = refs / f"{text}.safetensors"
        if not capture.exists():
            out.append({"text": text, "error": f"{capture} is missing"})
            continue
        for config in configs:
            command = [str(binary), str(model), str(capture), *shlex.split(config)]
            started = time.perf_counter()
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True)
            if on_start:
                on_start(process.pid)
            stdout, _ = process.communicate()
            name = f"{text}{config.replace(' ', '_')}.log"
            (logs / name).write_text(stdout)
            record = {"text": text, "config": config,
                      "seconds": round(time.perf_counter() - started, 1), **parse(stdout)}
            if "verdict" not in record:
                record["error"] = stdout.strip().splitlines()[-1] if stdout.strip() else "no output"
            out.append(record)
    return out
