"""Shared helper for the fixture scripts in this directory."""

from pathlib import Path

import mlx.core as mx

FIXTURES = Path(__file__).resolve().parents[2] / "crates/inkling-cuda/tests/fixtures"


def save(name: str, tensors: dict[str, mx.array]) -> None:
    """Writes `tensors` to `crates/inkling-cuda/tests/fixtures/<name>.safetensors`."""
    FIXTURES.mkdir(parents=True, exist_ok=True)
    path = FIXTURES / f"{name}.safetensors"
    mx.save_safetensors(str(path), tensors)
    size = path.stat().st_size
    print(f"{path}: {len(tensors)} tensors, {size / 1024:.0f} KiB")
