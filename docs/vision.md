# Image input

Inkling 2 Mini reads images through a 23-layer ViT whose output replaces placeholder tokens in the text model's input. This page records what the reference implementations do, step by step, so that the engine can match them. `crates/siphon/src/vision.rs` implements the CPU part (decode, pad, normalize, patch order, prompt tokens).

## Sources

| Reference | Version | Files |
| --- | --- | --- |
| Transformers | 5.24.0 | `models/inkling2/image_processing_inkling2.py`, `processing_inkling2.py`, `modeling_inkling2.py` (`Inkling2Vision*`, `Inkling2Model.forward`), `vision_utils.py` |
| mlx-vlm | [`111ac66`](https://github.com/Blaizzy/mlx-vlm/commit/111ac668078e017c9db1eb3936f4d7a6e0a88f89) (#2778) | `mlx_vlm/models/inkling2/{processing_inkling2,vision,inkling2}.py` |
| Checkpoint | the official one | `config.json` (`vision_config`), `processor_config.json`, `chat_template.jinja` |

The checkpoint's `model_type` is `inkling2_loop`. Transformers' `inkling2_loop` module reuses the `inkling2` vision tower, processor and image merge unchanged. The reference project pins Transformers 5.8.0 for the tokenizer, and 5.8.0 has no Inkling 2 vision code, so the vision captures run Transformers 5.24.0 in their own environment.

The two references agree on every step below except alpha handling, noted where it comes up.

## Preprocessing

There is no resize. The processor keeps the image at its own size and pads it up to a multiple of 40 pixels.

1. **Decode and orient.** Transformers' `load_image` and mlx-vlm's `load_image` both apply the EXIF orientation (`ImageOps.exif_transpose`) when they open a file or URL. A GIF gives its first frame.
2. **Convert to RGB.**
   - Grayscale becomes RGB by copying the channel.
   - An image with transparency (PIL's `has_transparency_data`: RGBA, LA, PA, or a `tRNS` chunk) is composited onto a gray checkerboard of 8-pixel squares, 204 where `(y / 8 + x / 8)` is even and 153 where it is odd, in integer arithmetic: `out = (c * a + bg * (255 - a) + 127) / 255`, truncating division. Fully opaque pixels keep their values exactly.
   - This happens only when the processor receives a PIL image with its alpha. Both references' `load_image` call `convert("RGB")` first, which drops alpha without compositing. Transformers' processor documents the checkerboard as intended, so the engine composites. See [open questions](#open-questions).
3. **Pad.** With `rows = ceil(height / 40)` and `cols = ceil(width / 40)`, the image is padded at the bottom and right to `40 * rows` by `40 * cols` with the value -1 on the 0 to 255 scale.
4. **Normalize.** `rescale_factor` 1/255, mean 0.5 and std 0.5. Transformers fuses them into `(v - 127.5) / 127.5` in F32. So a pixel value `v` in 0..255 maps to `(v - 127.5) / 127.5`, and the padding maps to `-128.5 / 127.5 = -1.0078431`. mlx-vlm computes `(v / 255 - 0.5) / 0.5`, which can differ in the last F32 bit. Both cast to BF16 before the patch embedding.
5. **Cut into tokens.** Each 40 x 40 block is one image token: `pixel_values` has shape `[rows * cols, 40, 40, 3]`, blocks in raster order (row of blocks, then column), each block in (y, x, channel) order. `image_grid_thw` is `[1, rows, cols]`, counted in tokens.

The encoder then splits each 40 x 40 block into a 2 x 2 group of 20 x 20 patches (`reshape(-1, 2, 20, 2, 20, 3).transpose(2, 3)`). The 1200 values of one patch are in (y, x, channel) order, 20 x 20 x 3. There is no temporal axis: `temporal_patch_size: 2` in `config.json` is unused by both references, `patch_embedding.weight` is `[1024, 1200]`, the grid's first entry is always 1, and mlx-vlm rejects any other value.

So the encoder's patch sequence for one image, in order, is:

```
for token_row in 0..rows:
  for token_col in 0..cols:
    for sub_row in 0..2:
      for sub_col in 0..2:
        patch at pixel rows 40*token_row + 20*sub_row .. +20,
                 pixel cols 40*token_col + 20*sub_col .. +20
        values: for y in 0..20, for x in 0..20, for c in R, G, B
```

`vision::prepare` emits the patches in exactly this order, `[4 * rows * cols, 1200]`, so the patch embedding is one plain matrix product.

An image of `H x W` pixels costs `ceil(H / 40) * ceil(W / 40)` tokens: 1 for anything up to 40 x 40, 676 for 1024 x 1024, 1,296 for 1920 x 1080 (27 rows, the last one half padding). The encoder sees four times as many patches plus a class token.

### Size limit

The references take any size. A 4K screenshot would cost 5,184 tokens and a 20,737-patch encoder pass. The engine adds a limit that the reference does not have: an image with more than `max_pixels` pixels is scaled down, keeping its aspect ratio, to `floor(W * s)` by `floor(H * s)` with `s = sqrt(max_pixels / (W * H))`, then processed as above. The resize uses PIL's bicubic filter (a = -0.5, support widened by the scale factor when shrinking, 22-bit fixed-point coefficients), ported so the result equals `Image.resize((w, h), Image.BICUBIC)` exactly. The default `max_pixels` is 1920 x 1080 = 2,073,600, which keeps a 1080p screenshot unchanged.

## Encoder

Hidden size 1024, 23 layers, 16 heads of 64, MLP 4096, LayerNorm eps 1e-5. All weights are BF16. Tensor names below are the checkpoint's, under `model.visual.`.

1. **Patch embedding.** `backbone.embeddings.patch_embedding.weight` `[1024, 1200]`, no bias, applied to each patch row.
2. **Position embedding.** `backbone.embeddings.position_embedding` `[1024, 1024]` is a 32 x 32 grid of vectors, index `row * 32 + col`. It is resampled to the image's patch grid of `2 * rows` by `2 * cols` with bilinear interpolation, `align_corners=False`, clamping at the borders (exactly `F.interpolate(mode="bilinear", align_corners=False)`):
   - along an axis of `n` patches, patch `i` samples the grid at `src = (i + 0.5) * 32 / n - 0.5`;
   - taps `floor(src)` and `floor(src) + 1`, each clamped to 0..31, with weights `1 - |src - tap|` (unclamped tap distance, so a clamped tap keeps its weight);
   - the 2-D weight is the product of the row and column weights.

   The gather and weighted sum run in F32 over the BF16 table. The result is cast to BF16 and added to the patch embedding (BF16 + BF16).
3. **Class token.** Each image's sequence starts with one class token, `class_embedding + cls_position_embedding`, added in BF16. The sequence is `[class, patch 0, patch 1, ...]`, `4 * rows * cols + 1` long.
4. **`ln_pre`** LayerNorm over the whole sequence, class token included.
5. **23 layers**, pre-norm with layer scale:
   ```
   x = x + ls_1.gamma * attn(ln_1(x))
   x = x + ls_2.gamma * mlp(ln_2(x))
   ```
   - `attn`: `self_attn.{q,k,v,out}_proj`, all with bias. Full bidirectional attention over the image's sequence, no mask, no window, scale 1/8. Different images in one batch do not attend to each other (`cu_seqlens` per image, class token included).
   - Rotary position on q and k, 2-D axial, theta 10000 (the Transformers default, since `config.json` sets no `rope_parameters`). Positions are 1-based `(row, col)` in patch units; the class token sits at `(0, 0)`, so it is not rotated. With `inv_freq[j] = 10000^(-2j / 32)` for `j` in 0..16, the 32 channel pairs `(2p, 2p + 1)` rotate by `col * inv_freq[p]` for `p` < 16 and by `row * inv_freq[p - 16]` for `p` >= 16: width first, then height. The rotation is `(a, b) -> (a cos - b sin, a sin + b cos)`, computed in F32 and cast back.
   - `mlp`: `mlp.c_fc` `[4096, 1024]` with bias, exact (erf) GELU, `mlp.c_proj` `[1024, 4096]` with bias.
6. **No final norm.** The last layer's output goes straight to the merge.

## Merge and projector

1. Drop the class token.
2. The patches are already grouped four per token, so `[4 * T, 1024]` reshapes to `[T, 4096]`: the 2 x 2 group of one 40 x 40 block concatenated in the order top-left, top-right, bottom-left, bottom-right.
3. `proj_in` `[4096, 4096]`, no bias, exact GELU, `proj_out` `[2304, 4096]`, no bias. No norm before, between or after.

The result is one 2304-wide vector per image token, cast to the text model's dtype.

## Text model input

The chat template renders an image content part as its own block, with the role header in front like any other part:

```
<|message_user|><|content_image|><|unused_200054|><|end_message|>
```

Images are allowed in user and tool messages only. The processor then repeats `<|unused_200054|>` `rows * cols` times. The ids are 200005 (`<|content_image|>`), 200054 (the placeholder) and 200010 (`<|end_message|>`). A user message with an image and then text renders as:

```
<|message_user|><|content_image|>{200054 x N}<|end_message|><|message_user|><|content_text|>What is this?<|end_message|>
```

`vision::prompt_tokens` returns the part from `<|content_image|>` through `<|end_message|>`; the caller writes the role header first, as it does for a text block.

The text model embeds all ids as usual, `embed_norm` included, and then overwrites the rows at placeholder positions with the image vectors, in order (`masked_scatter` in Transformers, the same in mlx-vlm). So the image vectors bypass `embed_norm`. Positions stay sequential: the placeholders take ordinary consecutive positions, with no multi-axis rotary positions and no change to the attention mask. Inkling's text layers use relative position bias, and an image token is one position like any other.

## Fixtures

`reference/vision_capture.py` makes the reference fixtures. It is a uv script with its own pinned environment (Transformers 5.24.0, torch 2.9.1 on the CPU, mlx-vlm `111ac66`, Pillow 12.0.0), separate from the reference project:

```sh
uv run --script reference/vision_capture.py images
uv run --script reference/vision_capture.py capture <model-dir> <out-dir>
```

`images` draws six test images into `reference/fixtures/vision/images/`. The script generates them itself: gradients, shapes and this repo's own source rendered as text. `capture` runs each one through Transformers' processor, Transformers' vision tower in F32 on the CPU (the BF16 weights upcast), mlx-vlm's vision tower in BF16, and mlx-vlm's full model. The chat prompt is siphon-serve's rendering of one user message with the image and then "Describe this image in one sentence.", at effort 0.

| Image | Format | Size | Tokens | Encoder sequence | Prompt ids |
| --- | --- | --- | ---: | ---: | ---: |
| `tiny` | PNG | 7 x 5 | 1 | 5 | 26 |
| `alpha` | RGBA PNG | 333 x 217 | 54 | 217 | 79 |
| `tall` | GIF | 360 x 1500 | 342 | 1,369 | 367 |
| `square` | lossy WebP | 1024 x 1024 | 676 | 2,705 | 701 |
| `screenshot` | JPEG | 1920 x 1080 | 1,296 | 5,185 | 1,321 |
| `huge` | PNG | 3000 x 2000, shrunk to 1763 x 1175 | 1,350 | 5,401 | 1,375 |

In git, under `reference/fixtures/vision/`:

- `<name>.json`: sizes, grid, token count, SHA-256 of the file and of the reference patches (F32, little-endian, encoder order), the BF16-vs-F32 comparison per recorded layer, and the text model's top 10 logits and 24 greedy tokens.
- `<name>.rgb.png` for `tiny`, `alpha`, `tall` and `screenshot`: the RGB pixels the processor normalized.
- `tiny.safetensors`: the tiny image's patches, last encoder layer, merged and projected features in F32, the BF16 projection, prompt ids and greedy tokens.
- `config/`: copies of the checkpoint's `config.json`, `processor_config.json` and `tokenizer_config.json` for the unit tests.

The full tensors (0.9 GiB) live on the GPU host under `~/vision-fixtures/`, with a README listing the keys: patches, the embeddings, `ln_pre`, layers 0, 5, 11, 17 and 22, the merged and projected features in F32, layers and projection in BF16, prompt ids, logits after the prompt and the greedy tokens.

On every image, mlx-vlm's greedy reply starts by describing the picture correctly ("a vertical stack of 15 colored horizontal" bands, "a code editor screenshot"), so the reference pipeline works end to end.

### What the Rust tests check

`cargo test -p siphon vision` runs against these fixtures:

- **Patch order and normalization, bit for bit.** Patching the reference's own RGB pixels gives the reference's patch hash for all four checked-in RGB images, the JPEG included.
- **The whole of `prepare`, bit for bit,** for PNG, GIF and lossy WebP: the `image` crate's WebP decoder matches libwebp exactly on this image.
- **JPEG within 4 levels.** zune-jpeg and PIL's libjpeg-turbo differ by at most 4 levels on the 0 to 255 scale, mean 0.041, on the screenshot. The test bounds them at 6 and 0.1. A level is 0.0078 after normalization.
- **The resize, bit for bit.** The 3000 x 2000 image shrinks to 1763 x 1175 and its patches equal the reference's, which ran PIL's `Image.resize(..., BICUBIC)`.
- Token counts and prompt ids for sizes from 1 x 1 to 1920 x 1080, the alpha composite, and rejection of other formats.

### Tolerances for the encoder

BF16 drifts a long way from the F32 reference on long sequences. Relative RMS error of mlx-vlm in BF16 against Transformers in F32, and the lowest cosine similarity of any row:

| Image | Layer 0 | Layer 11 | Layer 22 | Projected | Projected, lowest row cosine |
| --- | ---: | ---: | ---: | ---: | ---: |
| `tiny` | 0.6% | 0.9% | 0.6% | 1.9% | 0.9998 |
| `alpha` | 0.7% | 1.5% | 3.5% | 3.9% | 0.992 |
| `tall` | 0.7% | 2.2% | 5.1% | 4.5% | 0.970 |
| `square` | 0.7% | 2.5% | 5.6% | 4.5% | 0.986 |
| `screenshot` | 0.8% | 4.0% | 7.0% | 6.3% | 0.633 |
| `huge` | 0.7% | 2.4% | 4.4% | 3.9% | 0.932 |

This is precision, not a different model. mlx-vlm with its vision tower cast to F32 matches Transformers in F32 within 0.9% on the screenshot's projection (lowest row cosine 0.995) and 0.2% on the tiny image. Even then the error is not near zero: MLX's F32 matrix products on the M5 Max round inputs that are not BF16 values. A CUDA encoder in BF16 should land in the BF16 rows' range. One that lands near the F32 rows is closer to the model than mlx-vlm is.

The rows that drift are not small ones: on the screenshot, 6.5% of the last layer's rows fall below cosine 0.99, with norms of 2.7 to 4.4 against a median of 3.9. The class token, with a norm near 59, stays at cosine 1.0. A few rows are simply sensitive to rounding.

What the drift does to the text model: mlx-vlm's text model run twice over the same prompt, once with the F32 projection in the placeholder rows and once with the BF16 one, compared over the rows after the image (KL from the F32 run):

| Image | Mean KL | Max KL | Greedy tokens that agree |
| --- | ---: | ---: | ---: |
| `tiny` | 9.1e-2 | 0.56 | 24 of 24 |
| `alpha` | 4.5e-3 | 3.4e-2 | 24 of 24 |
| `tall` | 9.5e-3 | 6.1e-2 | 24 of 24 |
| `square` | 7.9e-3 | 3.5e-2 | 22 of 24 |
| `screenshot` | 3.1e-2 | 0.18 | 24 of 24 |
| `huge` | 4.7e-2 | 0.34 | 19 of 24 |

The next token after the prompt is the same in every case. An engine's image features compared the same way should give KLs of this size or smaller.

## Open questions

- **Alpha.** The image processor composites transparency onto a checkerboard, but both references' file loaders drop alpha first. Which one the model saw in training is not stated. The engine follows the processor.
- **16-bit images.** PIL keeps the high byte of 16-bit RGB PNGs and clips 16-bit grayscale. The `image` crate rounds. The engine does not match PIL there.
- **JPEG decoding.** zune-jpeg differs from libjpeg-turbo by up to 4 levels on the test screenshot. Matching it exactly would need libjpeg-turbo's IDCT and chroma upsampling.
- **Size limit.** The reference has none, so the downscale is the engine's choice, not the model's.
