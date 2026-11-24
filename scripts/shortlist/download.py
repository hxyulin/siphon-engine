#!/usr/bin/env python3
"""Fetch three pinned shards of mlabonne/open-perfectblend, the prompts for `prompts.py`.

usage: download.py <directory>
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil
import time
from urllib.request import urlopen


REVISION = "af60f3c18201652a83a93f46fcfee1b646ba3df7"
ROOT = f"https://huggingface.co/datasets/mlabonne/open-perfectblend/resolve/{REVISION}"
SHARDS = {
    "train-00000-of-00006.parquet": 248696008,
    "train-00002-of-00006.parquet": 246915862,
    "train-00004-of-00006.parquet": 246966726,
}


def fetch(root, item):
    name, expected = item
    path = root / name
    if path.is_file() and path.stat().st_size == expected:
        return
    pending = root / f"{name}.part"
    for attempt in range(4):
        try:
            with urlopen(f"{ROOT}/data/{name}", timeout=45) as data, pending.open("wb") as out:
                shutil.copyfileobj(data, out, length=1024 * 1024)
            if pending.stat().st_size != expected:
                raise OSError(f"Unexpected size for {name}")
            pending.replace(path)
            print(name, expected, flush=True)
            return
        except OSError:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    root = parser.parse_args().directory
    root.mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(lambda item: fetch(root, item), SHARDS.items()))


if __name__ == "__main__":
    main()
