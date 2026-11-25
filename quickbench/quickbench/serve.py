"""Starts and stops siphon-serve, reads its job log, and lists GPU processes."""

import os
import re
import subprocess
import time
from pathlib import Path

from . import client

JOB = re.compile(
    r"job: (\d+) prompt tokens \((\d+) cached\), prefill (\d+) ms, (\d+) tokens in "
    r"([\d.]+) s \(([\d.]+) tok/s, (\d+) speculated, (\d+) batched steps(.*)\), (\S+)"
)
PER_ROUND = re.compile(r"([\d.]+) tokens per round")


class Siphon:
    def __init__(self, binary: Path, model: Path, args: list[str], log: Path, port: int = 8471):
        self.command = [str(binary), str(model), "--port", str(port), *args]
        self.base = f"http://127.0.0.1:{port}/v1"
        self.log = log
        self.process = None

    def __enter__(self):
        self.log.parent.mkdir(parents=True, exist_ok=True)
        out = self.log.open("w")
        out.write(" ".join(self.command) + "\n")
        out.flush()
        self.process = subprocess.Popen(self.command, stdout=out, stderr=subprocess.STDOUT,
                                        stdin=subprocess.DEVNULL)
        deadline = time.time() + 900
        while time.time() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"siphon-serve exited, see {self.log}:\n{self.tail()}")
            try:
                client.get(f"{self.base}/models", timeout=2)
                return self
            except OSError:
                time.sleep(1)
        raise RuntimeError(f"siphon-serve did not start in 15 minutes, see {self.log}")

    def __exit__(self, *_):
        self.process.terminate()
        try:
            self.process.wait(60)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()

    def tail(self, lines: int = 20) -> str:
        return "".join(self.log.read_text(errors="replace").splitlines(True)[-lines:])

    def jobs(self) -> list[dict]:
        """Every finished job in the log so far, in finishing order."""
        out = []
        for line in self.log.read_text(errors="replace").splitlines():
            if m := JOB.search(line):
                per_round = PER_ROUND.search(m.group(9))
                out.append({"prompt_tokens": int(m.group(1)), "cached": int(m.group(2)),
                            "tokens": int(m.group(4)), "speculated": int(m.group(7)),
                            "batched": int(m.group(8)),
                            "tokens_per_round": float(per_round.group(1)) if per_round else None})
        return out


# Compositors show up as compute apps but do not compete for the GPU.
DESKTOP = ("kwin", "Xorg", "Xwayland", "gnome-shell", "plasmashell")


def gpu_apps() -> list[dict] | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    apps = []
    for line in out.stdout.strip().splitlines():
        pid, name, mib = (x.strip() for x in line.split(","))
        apps.append({"pid": int(pid), "name": name, "mib": int(mib) if mib.isdigit() else None})
    return apps


def others(apps: list[dict] | None, own: set[int]) -> list[dict]:
    """GPU processes that are neither ours nor the desktop compositor."""
    own = own | {os.getpid()}
    return [a for a in apps or [] if a["pid"] not in own
            and not any(d in Path(a["name"]).name for d in DESKTOP)]
