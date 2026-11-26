# /// script
# requires-python = "==3.12.*"
# dependencies = [
#     "mlx==0.32.4",
#     "mlx-vlm @ git+https://github.com/Blaizzy/mlx-vlm@111ac668078e017c9db1eb3936f4d7a6e0a88f89",
#     "numpy",
#     "pillow==12.0.0",
#     "safetensors",
#     "torch==2.9.1",
#     "torchvision==0.24.1",
#     "transformers==5.24.0",
# ]
# ///
"""Reference fixtures for image input: preprocessing, encoder and text model.

`images` draws the test images into `reference/fixtures/vision/images/`. They
are generated here, so they carry no third-party content.

`capture <model-dir> <out-dir>` runs every image through the references:

- Transformers' `Inkling2ImageProcessor` gives the patches. The image goes in
  as a PIL image with its alpha, so the processor composites transparency onto
  its checkerboard. An image above `--max-pixels` is first composited the same
  way and shrunk with PIL's bicubic filter, the size limit `vision.rs` adds.
- Transformers' `Inkling2VisionModel` on the CPU in F32, from the checkpoint's
  BF16 weights, gives the encoder states, the merged features and the
  projected embeddings: the high-precision reference.
- mlx-vlm's vision tower in BF16 gives the same states the way a BF16
  implementation computes them, as a yardstick for tolerances.
- mlx-vlm's full model gives the text model's logits for a short chat prompt
  around each image, and 24 greedy tokens.

Small results go to `reference/fixtures/vision/` (checked in): per image a
`<name>.json` with sizes, token counts, hashes and comparisons, and the RGB
pixels the processor normalized as `<name>.rgb.png` for the images in
`RGB_IN_GIT`. The tiny image's tensors
go to `tiny.safetensors`. Everything else goes to `<out-dir>/<name>.safetensors`.

    uv run --script reference/vision_capture.py images
    uv run --script reference/vision_capture.py capture <model-dir> <out-dir>
"""

import argparse
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parent
FIXTURES = ROOT / "fixtures" / "vision"
IMAGES = FIXTURES / "images"
REPO = ROOT.parent

DEFAULT_MAX_PIXELS = 1920 * 1080
TOKEN = 40
PATCH = 20
# Intermediate encoder layers recorded besides the last one.
LAYERS = (0, 5, 11, 17)
PROMPT_TEXT = "Describe this image in one sentence."
# Images whose normalized RGB is checked in. The JPEG needs it, since its decode differs
# from PIL's; the others keep the patch-order test independent of decoding.
RGB_IN_GIT = ("tiny", "alpha", "tall", "screenshot")
GREEDY_TOKENS = 24

# Token ids from the checkpoint's tokenizer.
MESSAGE_SYSTEM = 200002
MESSAGE_USER = 200000
MESSAGE_MODEL = 200001
CONTENT_TEXT = 200004
CONTENT_IMAGE = 200005
END_MESSAGE = 200010
IMAGE_TOKEN = 200054


# ----------------------------------------------------------------------------
# Test images


def font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.load_default(size=size)


def text_lines(path: Path, count: int) -> list[str]:
    return path.read_text().splitlines()[:count]


def draw_images() -> None:
    IMAGES.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)

    # 7 x 5: smaller than one token.
    tiny = rng.integers(0, 256, (5, 7, 3), dtype=np.uint8)
    Image.fromarray(tiny).save(IMAGES / "tiny.png")

    # 1024 x 1024 lossy WebP: gradients, shapes and a label.
    y, x = np.mgrid[0:1024, 0:1024].astype(np.float32) / 1023
    square = np.stack([x * 255, y * 255, (1 - x) * y * 255], -1).astype(np.uint8)
    img = Image.fromarray(square)
    d = ImageDraw.Draw(img)
    for i in range(6):
        c = tuple(int(v) for v in rng.integers(0, 256, 3))
        x0, y0 = 80 + i * 140, 120 + (i % 3) * 230
        d.ellipse([x0, y0, x0 + 180, y0 + 180], fill=c, outline=(0, 0, 0), width=6)
    d.rectangle([60, 820, 964, 960], fill=(250, 250, 245))
    d.text((90, 840), "siphon vision fixture 1024", fill=(20, 20, 20), font=font(72))
    img.save(IMAGES / "square.webp", quality=90, method=6)

    # 1920 x 1080 JPEG: this repo's source rendered like an editor screenshot.
    img = Image.new("RGB", (1920, 1080), (30, 32, 38))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 1920, 40], fill=(50, 52, 60))
    d.text((16, 8), "crates/siphon/src/config.rs", fill=(220, 220, 220), font=font(22))
    lines = text_lines(REPO / "crates" / "siphon" / "src" / "config.rs", 38)
    palette = [(220, 220, 220), (152, 195, 121), (97, 175, 239), (229, 192, 123)]
    for i, line in enumerate(lines):
        d.text((16, 52 + i * 27), f"{i + 1:3}", fill=(110, 110, 120), font=font(20))
        d.text((80, 52 + i * 27), line, fill=palette[i % 4], font=font(20))
    img.save(IMAGES / "screenshot.jpg", quality=90)

    # 360 x 1500 GIF: a tall strip of colored bands with labels.
    img = Image.new("RGB", (360, 1500), (255, 255, 255))
    d = ImageDraw.Draw(img)
    for i in range(15):
        c = tuple(int(v) for v in rng.integers(40, 256, 3))
        d.rectangle([0, i * 100, 360, i * 100 + 90], fill=c)
        d.text((20, i * 100 + 25), f"band {i + 1}", fill=(0, 0, 0), font=font(36))
    img.convert("P", palette=Image.Palette.ADAPTIVE, colors=64).save(IMAGES / "tall.gif")

    # 333 x 217 RGBA PNG: an alpha ramp and a fully transparent hole.
    y, x = np.mgrid[0:217, 0:333].astype(np.float32)
    rgba = np.zeros((217, 333, 4), np.uint8)
    rgba[..., 0] = (x / 332 * 255).astype(np.uint8)
    rgba[..., 1] = (y / 216 * 255).astype(np.uint8)
    rgba[..., 2] = 180
    rgba[..., 3] = (x / 332 * 255).astype(np.uint8)
    rgba[((x - 240) ** 2 + (y - 110) ** 2) < 40**2, 3] = 0
    rgba[:, :20, 3] = 255
    Image.fromarray(rgba, "RGBA").save(IMAGES / "alpha.png")

    # 3000 x 2000 PNG: above the default size limit.
    img = Image.new("RGB", (3000, 2000), (245, 245, 240))
    d = ImageDraw.Draw(img)
    for i in range(12):
        c = tuple(int(v) for v in rng.integers(0, 256, 3))
        x0, y0 = int(rng.integers(0, 2600)), int(rng.integers(0, 1700))
        d.rectangle([x0, y0, x0 + 400, y0 + 300], fill=c)
    for i, line in enumerate(text_lines(REPO / "README.md", 40)):
        d.text((60, 60 + i * 46), line[:110], fill=(10, 10, 10), font=font(36))
    img.save(IMAGES / "huge.png", optimize=True)


# ----------------------------------------------------------------------------
# Preprocessing


def checkerboard_composite(rgba: np.ndarray) -> np.ndarray:
    """The processor's alpha composite, on an `[h, w, 4]` uint8 array."""
    h, w = rgba.shape[:2]
    rows = np.arange(h)[:, None] // 8
    cols = np.arange(w)[None, :] // 8
    bg = np.where((rows + cols) % 2 == 0, 204, 153)[..., None]
    v = rgba.astype(np.int32)
    a = v[..., 3:4]
    return ((v[..., :3] * a + bg * (255 - a) + 127) // 255).astype(np.uint8)


def limited_size(w: int, h: int, max_pixels: int) -> tuple[int, int] | None:
    """The size `vision.rs` shrinks an image to, or None if it fits."""
    if w * h <= max_pixels:
        return None
    s = math.sqrt(max_pixels / (w * h))
    nw, nh = max(1, math.floor(w * s)), max(1, math.floor(h * s))
    while nw * nh > max_pixels:
        if nw >= nh:
            nw -= 1
        else:
            nh -= 1
    return nw, nh


def open_image(path: Path, max_pixels: int) -> tuple[Image.Image, tuple[int, int] | None]:
    """Opens an image the way the reference loaders do, keeping alpha."""
    img = Image.open(path)
    img.load()
    img = ImageOps.exif_transpose(img)
    resized = limited_size(img.width, img.height, max_pixels)
    if resized is None:
        return img, None
    if img.has_transparency_data:
        rgb = Image.fromarray(checkerboard_composite(np.asarray(img.convert("RGBA"))))
    else:
        rgb = img.convert("RGB")
    return rgb.resize(resized, Image.Resampling.BICUBIC), resized


def encoder_order(pixel_values: np.ndarray) -> np.ndarray:
    """`[tokens, 40, 40, 3]` to the encoder's `[4 * tokens, 1200]` patch rows."""
    t = pixel_values.shape[0]
    return (
        pixel_values.reshape(t, 2, PATCH, 2, PATCH, 3)
        .transpose(0, 1, 3, 2, 4, 5)
        .reshape(4 * t, PATCH * PATCH * 3)
    )


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ----------------------------------------------------------------------------
# Transformers vision tower, F32


def torch_vision(model_dir: Path):
    import torch
    from safetensors import safe_open
    from transformers import AutoConfig
    from transformers.models.inkling2.modeling_inkling2 import Inkling2VisionModel

    config = AutoConfig.from_pretrained(model_dir).vision_config
    config._attn_implementation = "eager"
    model = Inkling2VisionModel(config)

    def rename(key: str) -> str:
        key = key.removeprefix("model.visual.").removeprefix("backbone.")
        if key == "embeddings.position_embedding":
            return "embeddings.position_embedding.weight"
        for a, b in (
            ("ln_pre.", "pre_layernorm."),
            (".ln_1.", ".layernorm_before."),
            (".ln_2.", ".layernorm_after."),
            (".ls_1.gamma", ".lambda_1"),
            (".ls_2.gamma", ".lambda_2"),
            (".mlp.c_fc.", ".mlp.fc1."),
            (".mlp.c_proj.", ".mlp.fc2."),
            ("proj_in.", "merger.linear_1."),
            ("proj_out.", "merger.linear_2."),
        ):
            key = key.replace(a, b)
        return key

    state = {}
    for shard in sorted(model_dir.glob("model-*.safetensors")):
        with safe_open(shard, "pt") as f:
            for key in f.keys():
                if key.startswith("model.visual."):
                    state[rename(key)] = f.get_tensor(key).float()
    model.load_state_dict(state, strict=True)
    model.float().eval()
    torch.set_num_threads(max(1, torch.get_num_threads()))
    return model


def run_torch(model, pixel_values: np.ndarray, grid: list[int]) -> dict[str, np.ndarray]:
    import torch

    record: dict[str, np.ndarray] = {}
    hooks = [
        model.embeddings.register_forward_hook(
            lambda m, i, o: record.__setitem__("embeddings", o.numpy().copy())
        ),
        model.pre_layernorm.register_forward_hook(
            lambda m, i, o: record.__setitem__("ln_pre", o.numpy().copy())
        ),
    ]
    last = len(model.layers) - 1
    for n, layer in enumerate(model.layers):
        if n in LAYERS or n == last:
            hooks.append(
                layer.register_forward_hook(
                    lambda m, i, o, n=n: record.__setitem__(f"layer{n:02}", o.numpy().copy())
                )
            )
    with torch.no_grad():
        out = model(
            pixel_values=torch.from_numpy(pixel_values),
            image_grid_thw=torch.tensor([grid]),
        )
    for h in hooks:
        h.remove()
    hidden = out.last_hidden_state.numpy()
    record["merged"] = hidden[1:].reshape(-1, 4 * hidden.shape[-1])
    record["projected"] = out.pooler_output.numpy()
    return record


# ----------------------------------------------------------------------------
# mlx-vlm: BF16 vision tower and the full model


class Recorder:
    def __init__(self, layer, key: str, record: dict):
        self.layer, self.key, self.record = layer, key, record

    def __call__(self, *args, **kwargs):
        out = self.layer(*args, **kwargs)
        self.record[self.key] = out
        return out


def run_mlx_vision(model, pixel_values: np.ndarray, grid: list[int]) -> dict[str, np.ndarray]:
    import mlx.core as mx

    tower = model.vision_tower
    record: dict = {}
    originals = list(tower.layers)
    last = len(originals) - 1
    for n, layer in enumerate(originals):
        if n in LAYERS or n == last:
            tower.layers[n] = Recorder(layer, f"layer{n:02}", record)
    try:
        projected = tower(mx.array(pixel_values), image_grid_thw=mx.array([grid]))
        mx.eval(projected, *record.values())
    finally:
        for n, layer in enumerate(originals):
            tower.layers[n] = layer
    out = {k: np.asarray(v.astype(mx.float32)) for k, v in record.items()}
    out["projected"] = np.asarray(projected.astype(mx.float32))
    return out


def prompt_ids(tokenizer, tokens: int) -> list[int]:
    """The ids siphon-serve builds for one user message: image, then text."""
    ids = [MESSAGE_SYSTEM, CONTENT_TEXT]
    ids += tokenizer.encode("Thinking effort level: 0", add_special_tokens=False)
    ids += [END_MESSAGE, MESSAGE_USER, CONTENT_IMAGE] + [IMAGE_TOKEN] * tokens
    ids += [END_MESSAGE, MESSAGE_USER, CONTENT_TEXT]
    ids += tokenizer.encode(PROMPT_TEXT, add_special_tokens=False)
    ids += [END_MESSAGE, MESSAGE_MODEL]
    return ids


def run_mlx_text(model, tokenizer, ids: list[int], pixel_values: np.ndarray, grid: list[int]):
    import mlx.core as mx

    cache = model.make_cache()
    out = model(
        mx.array([ids]),
        pixel_values=mx.array(pixel_values),
        image_grid_thw=mx.array([grid]),
        cache=cache,
    )
    logits = out.logits[0, -1].astype(mx.float32)
    mx.eval(logits)
    first = np.asarray(logits)
    tokens = [int(first.argmax())]
    for _ in range(GREEDY_TOKENS - 1):
        step = model.language_model(inputs=mx.array([[tokens[-1]]]), cache=cache)
        tokens.append(int(mx.argmax(step.logits[0, -1]).item()))
    return first, tokens, tokenizer.decode(tokens)


# ----------------------------------------------------------------------------


def compare(ref: np.ndarray, other: np.ndarray) -> dict[str, float]:
    diff = np.abs(ref.astype(np.float64) - other.astype(np.float64))
    scale = np.abs(ref).astype(np.float64)
    return {
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "ref_mean_abs": float(scale.mean()),
        "rel_rms": float(np.sqrt((diff**2).mean() / max((scale**2).mean(), 1e-30))),
        "cosine_min": float(
            np.min(
                (ref * other).sum(-1)
                / np.maximum(np.linalg.norm(ref, axis=-1) * np.linalg.norm(other, axis=-1), 1e-30)
            )
        ),
    }


def capture(args: argparse.Namespace) -> None:
    from safetensors.numpy import save_file
    from transformers.models.inkling2.image_processing_inkling2 import Inkling2ImageProcessor

    processor_config = json.loads((args.model / "processor_config.json").read_text())
    processor = Inkling2ImageProcessor(**processor_config["image_processor"])
    args.out.mkdir(parents=True, exist_ok=True)

    tvision = torch_vision(args.model) if not args.no_torch else None
    mlx_model = tokenizer = None
    if not args.no_mlx:
        from mlx_vlm import load

        mlx_model, mlx_processor = load(str(args.model))
        tokenizer = mlx_processor.tokenizer

    names = args.only or sorted(p.stem for p in IMAGES.iterdir())
    for name in names:
        path = next(IMAGES.glob(f"{name}.*"))
        print(f"{name}: {path.name}", flush=True)
        raw = path.read_bytes()
        img, resized = open_image(path, args.max_pixels)
        source = Image.open(io.BytesIO(raw))
        batch = processor(images=[img], return_tensors="np")
        pixel_values = batch["pixel_values"].astype(np.float32)
        token_grid = [int(v) for v in batch["image_grid_thw"][0]]
        tokens = token_grid[1] * token_grid[2]
        patches = np.ascontiguousarray(encoder_order(pixel_values))

        # The RGB values the processor normalized, cropped to the image.
        w, h = img.size
        rows, cols = token_grid[1], token_grid[2]
        full = pixel_values.reshape(rows, cols, TOKEN, TOKEN, 3).transpose(0, 2, 1, 3, 4)
        full = full.reshape(rows * TOKEN, cols * TOKEN, 3)
        rgb = np.rint(full[:h, :w] * np.float32(127.5) + np.float32(127.5)).astype(np.uint8)
        normalized = (rgb.astype(np.float32) - np.float32(127.5)) / np.float32(127.5)
        assert np.array_equal(full[:h, :w], normalized), "values are not (v - 127.5) / 127.5"
        pad = np.float32(-128.5) / np.float32(127.5)
        assert np.all(full[h:] == pad) and np.all(full[:, w:] == pad), "padding is not -1"
        rgb_dir = FIXTURES if name in RGB_IN_GIT else args.out
        Image.fromarray(rgb).save(rgb_dir / f"{name}.rgb.png", optimize=True)

        meta = {
            "file": f"images/{path.name}",
            "file_sha256": sha256(raw),
            "source_size": [source.width, source.height],
            "resized": list(resized) if resized else None,
            "max_pixels": args.max_pixels,
            "token_grid": token_grid,
            "grid": [1, 2 * token_grid[1], 2 * token_grid[2]],
            "tokens": tokens,
            "patches": int(patches.shape[0]),
            "patches_sha256": sha256(patches.astype("<f4").tobytes()),
        }
        tensors = {"patches": patches}

        if tvision is not None:
            ref = run_torch(tvision, pixel_values, token_grid)
            tensors.update({f"f32.{k}": v for k, v in ref.items()})
        if mlx_model is not None:
            bf16 = run_mlx_vision(mlx_model, pixel_values, token_grid)
            tensors.update({f"bf16.{k}": v for k, v in bf16.items()})
            if tvision is not None:
                meta["bf16_vs_f32"] = {
                    k: compare(ref[k], bf16[k]) for k in bf16 if k in ref
                }
            ids = prompt_ids(tokenizer, tokens)
            logits, greedy, text = run_mlx_text(mlx_model, tokenizer, ids, pixel_values, token_grid)
            top = np.argsort(-logits)[:10]
            tensors["prompt_ids"] = np.array(ids, np.int32)
            tensors["logits"] = logits
            tensors["greedy"] = np.array(greedy, np.int32)
            meta["text"] = {
                "prompt": PROMPT_TEXT,
                "prompt_tokens": len(ids),
                "top10": [[int(t), float(logits[t])] for t in top],
                "greedy": greedy,
                "greedy_text": text,
            }
            print(f"  {text!r}", flush=True)

        save_file({k: np.ascontiguousarray(v) for k, v in tensors.items()}, str(args.out / f"{name}.safetensors"))
        if name == "tiny":
            keep = {k: v for k, v in tensors.items() if k in ("patches", "f32.layer22", "f32.merged", "f32.projected", "bf16.projected", "prompt_ids", "greedy")}
            save_file({k: np.ascontiguousarray(v) for k, v in keep.items()}, str(FIXTURES / "tiny.safetensors"))
        (FIXTURES / f"{name}.json").write_text(json.dumps(meta, indent=1) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("images", help="draw the test images")
    cap = sub.add_parser("capture", help="run the references over the test images")
    cap.add_argument("model", type=Path)
    cap.add_argument("out", type=Path)
    cap.add_argument("--max-pixels", type=int, default=DEFAULT_MAX_PIXELS)
    cap.add_argument("--only", nargs="*", help="image names, without extension")
    cap.add_argument("--no-torch", action="store_true", help="skip the F32 Transformers encoder")
    cap.add_argument("--no-mlx", action="store_true", help="skip mlx-vlm")
    args = parser.parse_args()
    if args.command == "images":
        draw_images()
    else:
        capture(args)


if __name__ == "__main__":
    main()
