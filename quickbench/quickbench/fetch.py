"""Downloads the quality track's datasets from Hugging Face into the cache.

The files stay in the cache directory, outside the repository. GPQA is gated:
huggingface_hub reads the token of an existing `hf auth login`.
"""

from pathlib import Path

from huggingface_hub import hf_hub_download

# source -> (dataset repository, revision, file)
DATASETS = {
    "gpqa": ("Idavidrein/gpqa", "83022cefff930aea54f654c0b282e74b9eeda5c6", "gpqa_diamond.csv"),
    "aime": ("math-ai/aime25", "563bb8404243c5f09de6ec262f2db674fe5bce9b", "test.jsonl"),
    "code": ("evalplus/humanevalplus", "d32357cf319e50e9c8d8dab5ea876c72b0fd321b", "test.jsonl"),
}


def path(cache: Path, source: str) -> Path:
    repo, revision, file = DATASETS[source]
    return cache / "data" / repo.replace("/", "--") / revision / file


def fetch(cache: Path) -> None:
    for source, (repo, revision, file) in DATASETS.items():
        out = path(cache, source)
        if out.exists():
            print(f"{source}: {out} (cached)")
            continue
        hf_hub_download(repo, file, repo_type="dataset", revision=revision, local_dir=out.parent)
        print(f"{source}: {out}")
