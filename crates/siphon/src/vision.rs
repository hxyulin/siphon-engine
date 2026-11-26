//! CPU image preprocessing for the vision encoder.
//!
//! `prepare` turns encoded image bytes into normalized 20 x 20 patches in the order the
//! encoder consumes them, following Transformers' `Inkling2ImageProcessor`: decode, apply
//! the EXIF orientation, composite transparency onto a gray checkerboard, pad to a multiple
//! of 40 pixels and normalize. The reference never resizes. This module adds one step it
//! lacks: an image above a pixel limit is shrunk with PIL's bicubic filter first.
//! `docs/vision.md` describes every step.

use std::{fs, io::Cursor, path::Path};

use anyhow::{Context as _, bail, ensure};
use image::{DynamicImage, ImageDecoder, ImageFormat, ImageReader, RgbImage, RgbaImage};
use serde::Deserialize;
use sha2::{Digest as _, Sha256};

/// Default for [`VisionConfig::max_pixels`]: 1920 x 1080, so a 1080p screenshot keeps its
/// size and costs 1,296 tokens.
pub const DEFAULT_MAX_PIXELS: usize = 1920 * 1080;

/// Values in one patch: 20 x 20 pixels x 3 channels.
const CHANNELS: usize = 3;

/// The vision encoder's hyperparameters and the image processor's settings, read from the
/// checkpoint's `config.json`, `processor_config.json` and `tokenizer_config.json`.
#[derive(Debug, Clone, PartialEq)]
pub struct VisionConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    /// MLP width: `hidden_size * mlp_ratio`.
    pub intermediate_size: usize,
    /// Side of one encoder patch in pixels.
    pub patch_size: usize,
    /// Side, in patches, of the square merged into one text token.
    pub merge_size: usize,
    /// Side of the learned position embedding grid.
    pub posemb_grid_size: usize,
    pub projection_hidden_dim: usize,
    pub projector_bias: bool,
    pub layer_norm_eps: f32,
    pub text_hidden_size: usize,
    /// Axial rotary base. `config.json` sets none, so this is Transformers' default.
    pub rope_theta: f32,
    /// Per-channel mean and std on the 0 to 255 scale, after folding in the rescale factor.
    pub mean: [f32; 3],
    pub std: [f32; 3],
    /// `<|unused_200054|>`, repeated once per image token.
    pub image_token_id: u32,
    /// `<|content_image|>`, which opens an image block.
    pub content_image_id: u32,
    /// `<|end_message|>`, which closes it.
    pub end_message_id: u32,
    /// Images with more pixels are shrunk to fit before patching. Not in the reference.
    pub max_pixels: usize,
}

#[derive(Deserialize)]
struct ConfigFile {
    vision_config: VisionConfigFile,
}

#[derive(Deserialize)]
struct VisionConfigFile {
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    mlp_ratio: f64,
    native_patch_size: usize,
    adapter_merge_size: usize,
    posemb_grid_size: usize,
    projection_hidden_dim: usize,
    projector_bias: bool,
    layer_norm_eps: f32,
    text_hidden_size: usize,
}

#[derive(Deserialize)]
struct ProcessorFile {
    image_processor: ImageProcessorFile,
    image_token: String,
}

#[derive(Deserialize)]
#[allow(clippy::struct_excessive_bools)]
struct ImageProcessorFile {
    do_convert_rgb: bool,
    do_normalize: bool,
    do_rescale: bool,
    image_mean: [f64; 3],
    image_std: [f64; 3],
    rescale_factor: f64,
    size: Size,
}

#[derive(Deserialize)]
struct Size {
    height: usize,
    width: usize,
}

#[derive(Deserialize)]
struct TokenizerConfigFile {
    added_tokens_decoder: std::collections::HashMap<String, AddedToken>,
}

#[derive(Deserialize)]
struct AddedToken {
    content: String,
}

impl VisionConfig {
    /// Reads the vision settings from the checkpoint folder `model_dir`.
    pub fn load(model_dir: &Path) -> anyhow::Result<Self> {
        let read = |name: &str| {
            let path = model_dir.join(name);
            fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))
        };
        Self::from_json(
            &read("config.json")?,
            &read("processor_config.json")?,
            &read("tokenizer_config.json")?,
        )
    }

    /// Parses the contents of `config.json`, `processor_config.json` and
    /// `tokenizer_config.json`.
    #[allow(clippy::cast_sign_loss)]
    pub fn from_json(config: &str, processor: &str, tokenizer: &str) -> anyhow::Result<Self> {
        let v = serde_json::from_str::<ConfigFile>(config)
            .context("parsing vision_config in config.json")?
            .vision_config;
        let p: ProcessorFile =
            serde_json::from_str(processor).context("parsing processor_config.json")?;
        let t: TokenizerConfigFile =
            serde_json::from_str(tokenizer).context("parsing tokenizer_config.json")?;
        let ip = p.image_processor;
        ensure!(
            ip.do_convert_rgb && ip.do_normalize && ip.do_rescale,
            "the image processor must convert to RGB, rescale and normalize"
        );
        let token = v.native_patch_size * v.adapter_merge_size;
        ensure!(
            ip.size.height == token && ip.size.width == token,
            "processor size {}x{} is not patch size x merge size ({token})",
            ip.size.height,
            ip.size.width
        );
        let id = |content: &str| {
            t.added_tokens_decoder
                .iter()
                .find(|(_, tok)| tok.content == content)
                .map(|(id, _)| id.parse::<u32>())
                .with_context(|| format!("{content} is not an added token"))?
                .with_context(|| format!("bad id for {content}"))
        };
        // Transformers fuses rescale and normalize: mean and std are divided by the rescale
        // factor, as F32 tensors times a Python float, and the pixels are not rescaled.
        let fold = |x: f64| (x as f32) * ((1.0 / ip.rescale_factor) as f32);
        Ok(Self {
            hidden_size: v.hidden_size,
            num_hidden_layers: v.num_hidden_layers,
            num_attention_heads: v.num_attention_heads,
            intermediate_size: (v.hidden_size as f64 * v.mlp_ratio) as usize,
            patch_size: v.native_patch_size,
            merge_size: v.adapter_merge_size,
            posemb_grid_size: v.posemb_grid_size,
            projection_hidden_dim: v.projection_hidden_dim,
            projector_bias: v.projector_bias,
            layer_norm_eps: v.layer_norm_eps,
            text_hidden_size: v.text_hidden_size,
            rope_theta: 10_000.0,
            mean: ip.image_mean.map(fold),
            std: ip.image_std.map(fold),
            image_token_id: id(&p.image_token)?,
            content_image_id: id("<|content_image|>")?,
            end_message_id: id("<|end_message|>")?,
            max_pixels: DEFAULT_MAX_PIXELS,
        })
    }

    /// Side of the pixel square that becomes one text token: 40.
    #[must_use]
    pub fn token_size(&self) -> usize {
        self.patch_size * self.merge_size
    }

    /// Values in one patch row: 1200.
    #[must_use]
    pub fn patch_len(&self) -> usize {
        self.patch_size * self.patch_size * CHANNELS
    }
}

/// One image, decoded and cut into normalized patches.
#[derive(Debug, Clone, PartialEq)]
pub struct PreparedImage {
    /// Temporal, height and width in 20-pixel patches after any resize and the padding.
    /// The first entry is always 1; height and width are even.
    pub grid: [usize; 3],
    /// `[patches, 1200]`: token rows, then token columns, then the 2 x 2 patches of a
    /// token (top-left, top-right, bottom-left, bottom-right), each in (y, x, channel)
    /// order. The encoder's patch embedding is one product over these rows.
    pub patches: Vec<f32>,
    /// SHA-256 of the encoded image bytes.
    pub hash: [u8; 32],
}

impl PreparedImage {
    /// Placeholder tokens the image expands to: one per 2 x 2 patches.
    #[must_use]
    pub fn tokens(&self) -> usize {
        self.grid[0] * self.grid[1] * self.grid[2] / 4
    }

    /// Rows of `patches`.
    #[must_use]
    pub fn num_patches(&self) -> usize {
        self.grid[0] * self.grid[1] * self.grid[2]
    }
}

/// Decodes `bytes` (PNG, JPEG, WebP, or the first frame of a GIF) and prepares it, shrinking
/// it to at most `cfg.max_pixels` pixels.
pub fn prepare(cfg: &VisionConfig, bytes: &[u8]) -> anyhow::Result<PreparedImage> {
    prepare_with_max_pixels(cfg, bytes, cfg.max_pixels)
}

/// [`prepare`] with an explicit pixel limit instead of `cfg.max_pixels`.
pub fn prepare_with_max_pixels(
    cfg: &VisionConfig,
    bytes: &[u8],
    max_pixels: usize,
) -> anyhow::Result<PreparedImage> {
    ensure!(max_pixels > 0, "max_pixels must be positive");
    let mut rgb = decode_rgb(bytes)?;
    if let Some((w, h)) = limited_size(rgb.width() as usize, rgb.height() as usize, max_pixels) {
        rgb = resize_bicubic(&rgb, w, h);
    }
    let (grid, patches) = patchify(cfg, &rgb);
    Ok(PreparedImage {
        grid,
        patches,
        hash: Sha256::digest(bytes).into(),
    })
}

/// The ids that stand for one image: `<|content_image|>`, one `<|unused_200054|>` per token
/// and `<|end_message|>`. The caller writes the role header in front, as for a text block.
#[must_use]
pub fn prompt_tokens(cfg: &VisionConfig, image: &PreparedImage) -> Vec<u32> {
    let mut ids = Vec::with_capacity(image.tokens() + 2);
    ids.push(cfg.content_image_id);
    ids.extend(std::iter::repeat_n(cfg.image_token_id, image.tokens()));
    ids.push(cfg.end_message_id);
    ids
}

/// Decodes to 8-bit RGB the way the reference processor sees the image: upright, first
/// frame, transparency composited onto the checkerboard.
fn decode_rgb(bytes: &[u8]) -> anyhow::Result<RgbImage> {
    let reader = ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .context("reading the image header")?;
    match reader.format() {
        Some(ImageFormat::Png | ImageFormat::Jpeg | ImageFormat::WebP | ImageFormat::Gif) => {}
        Some(other) => bail!("unsupported image format {other:?}"),
        None => bail!("unrecognized image format"),
    }
    let mut decoder = reader.into_decoder().context("decoding the image")?;
    let orientation = decoder
        .orientation()
        .context("reading the image orientation")?;
    let mut image = DynamicImage::from_decoder(decoder).context("decoding the image")?;
    image.apply_orientation(orientation);
    ensure!(
        image.width() > 0 && image.height() > 0,
        "the image has no pixels"
    );
    Ok(if image.color().has_alpha() {
        composite(&image.to_rgba8())
    } else {
        image.to_rgb8()
    })
}

/// Composites onto the processor's checkerboard of 8-pixel squares, 204 and 153, in its
/// integer arithmetic. Opaque pixels keep their values.
fn composite(rgba: &RgbaImage) -> RgbImage {
    RgbImage::from_fn(rgba.width(), rgba.height(), |x, y| {
        let p = rgba.get_pixel(x, y).0;
        let bg: u32 = if (x / 8 + y / 8) % 2 == 0 { 204 } else { 153 };
        let a = u32::from(p[3]);
        image::Rgb([0, 1, 2].map(|c| ((u32::from(p[c]) * a + bg * (255 - a) + 127) / 255) as u8))
    })
}

/// The size an image of `w` x `h` shrinks to under `max_pixels`, or `None` if it fits.
/// Keeps the aspect ratio and rounds both sides down.
#[allow(clippy::cast_sign_loss)]
fn limited_size(w: usize, h: usize, max_pixels: usize) -> Option<(usize, usize)> {
    if w * h <= max_pixels {
        return None;
    }
    let s = (max_pixels as f64 / (w * h) as f64).sqrt();
    let mut nw = ((w as f64 * s).floor() as usize).max(1);
    let mut nh = ((h as f64 * s).floor() as usize).max(1);
    while nw * nh > max_pixels {
        if nw >= nh {
            nw -= 1;
        } else {
            nh -= 1;
        }
    }
    Some((nw, nh))
}

/// Pads to whole tokens, normalizes and emits the patch rows in encoder order.
#[allow(clippy::many_single_char_names)]
fn patchify(cfg: &VisionConfig, rgb: &RgbImage) -> ([usize; 3], Vec<f32>) {
    let (w, h) = (rgb.width() as usize, rgb.height() as usize);
    let (p, m) = (cfg.patch_size, cfg.merge_size);
    let token = cfg.token_size();
    let rows = h.div_ceil(token);
    let cols = w.div_ceil(token);
    // Every 8-bit value per channel, and the padding value -1, normalized as Transformers
    // does: (v - mean) / std in F32.
    let mut lut = [[0f32; 256]; 3];
    for (c, table) in lut.iter_mut().enumerate() {
        for (v, out) in table.iter_mut().enumerate() {
            *out = (v as f32 - cfg.mean[c]) / cfg.std[c];
        }
    }
    let pad = [0, 1, 2].map(|c| (-1.0 - cfg.mean[c]) / cfg.std[c]);
    let raw = rgb.as_raw();
    let mut patches = Vec::with_capacity(rows * cols * m * m * cfg.patch_len());
    for tr in 0..rows {
        for tc in 0..cols {
            for sr in 0..m {
                for sc in 0..m {
                    let y0 = tr * token + sr * p;
                    let x0 = tc * token + sc * p;
                    for y in y0..y0 + p {
                        for x in x0..x0 + p {
                            if y < h && x < w {
                                let i = (y * w + x) * CHANNELS;
                                patches.extend((0..CHANNELS).map(|c| lut[c][raw[i + c] as usize]));
                            } else {
                                patches.extend_from_slice(&pad);
                            }
                        }
                    }
                }
            }
        }
    }
    ([1, rows * m, cols * m], patches)
}

/// PIL's `Image.resize(size, Image.BICUBIC)` for 8-bit RGB, ported from Pillow's
/// `Resample.c` so the output is identical: separable passes, horizontal first, with
/// coefficients computed in F64 and rounded to 22-bit fixed point.
fn resize_bicubic(src: &RgbImage, out_w: usize, out_h: usize) -> RgbImage {
    let (in_w, in_h) = (src.width() as usize, src.height() as usize);
    let horizontal = Coeffs::new(in_w, out_w);
    let vertical = Coeffs::new(in_h, out_h);
    // Horizontal pass over every row, then vertical. Pillow restricts the first pass to the
    // rows the second one reads, which gives the same values.
    let raw = src.as_raw();
    let mut tmp = vec![0u8; out_w * in_h * CHANNELS];
    for y in 0..in_h {
        let row = &raw[y * in_w * CHANNELS..(y + 1) * in_w * CHANNELS];
        for x in 0..out_w {
            let (start, k) = horizontal.taps(x);
            for c in 0..CHANNELS {
                let mut ss = 1i64 << (PRECISION_BITS - 1);
                for (j, &kj) in k.iter().enumerate() {
                    ss += i64::from(row[(start + j) * CHANNELS + c]) * i64::from(kj);
                }
                tmp[(y * out_w + x) * CHANNELS + c] = clip8(ss);
            }
        }
    }
    let mut out = vec![0u8; out_w * out_h * CHANNELS];
    let stride = out_w * CHANNELS;
    for y in 0..out_h {
        let (start, k) = vertical.taps(y);
        for i in 0..stride {
            let mut ss = 1i64 << (PRECISION_BITS - 1);
            for (j, &kj) in k.iter().enumerate() {
                ss += i64::from(tmp[(start + j) * stride + i]) * i64::from(kj);
            }
            out[y * stride + i] = clip8(ss);
        }
    }
    RgbImage::from_raw(out_w as u32, out_h as u32, out).expect("buffer matches the size")
}

/// Pillow's fixed-point fraction bits for 8-bit images: `32 - 8 - 2`.
const PRECISION_BITS: u32 = 22;

#[allow(clippy::cast_sign_loss)]
fn clip8(ss: i64) -> u8 {
    (ss >> PRECISION_BITS).clamp(0, 255) as u8
}

/// Pillow's bicubic kernel, a = -0.5, support 2.
fn bicubic(x: f64) -> f64 {
    const A: f64 = -0.5;
    let x = x.abs();
    if x < 1.0 {
        ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        (((x - 5.0) * x + 8.0) * x - 4.0) * A
    } else {
        0.0
    }
}

/// One axis's taps, as Pillow's `precompute_coeffs` and `normalize_coeffs_8bpc` make them.
struct Coeffs {
    ksize: usize,
    /// First input index and tap count per output index.
    bounds: Vec<(usize, usize)>,
    /// `ksize` fixed-point weights per output index.
    k: Vec<i32>,
}

impl Coeffs {
    #[allow(clippy::cast_sign_loss, clippy::cast_possible_wrap)]
    fn new(in_size: usize, out_size: usize) -> Self {
        let scale = in_size as f64 / out_size as f64;
        let filterscale = scale.max(1.0);
        let support = 2.0 * filterscale;
        let ksize = support.ceil() as usize * 2 + 1;
        let mut bounds = Vec::with_capacity(out_size);
        let mut k = vec![0i32; out_size * ksize];
        let mut weights = vec![0f64; ksize];
        for xx in 0..out_size {
            let center = (xx as f64 + 0.5) * scale;
            let ss = 1.0 / filterscale;
            // C casts truncate toward zero.
            let xmin = ((center - support + 0.5) as i64).max(0) as usize;
            let xmax = ((center + support + 0.5) as i64).min(in_size as i64) as usize - xmin;
            let mut ww = 0.0;
            for (x, w) in weights.iter_mut().enumerate().take(xmax) {
                *w = bicubic(((x + xmin) as f64 - center + 0.5) * ss);
                ww += *w;
            }
            for (x, w) in weights.iter().enumerate().take(xmax) {
                let w = if ww == 0.0 { *w } else { w / ww };
                let fixed = w * f64::from(1u32 << PRECISION_BITS);
                k[xx * ksize + x] = if w < 0.0 {
                    (-0.5 + fixed) as i32
                } else {
                    (0.5 + fixed) as i32
                };
            }
            bounds.push((xmin, xmax));
        }
        Self { ksize, bounds, k }
    }

    fn taps(&self, i: usize) -> (usize, &[i32]) {
        let (start, n) = self.bounds[i];
        (start, &self.k[i * self.ksize..i * self.ksize + n])
    }
}

#[cfg(test)]
mod tests {
    use std::{fmt::Write as _, path::PathBuf};

    use super::*;

    fn fixtures() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../reference/fixtures/vision")
    }

    /// The checkpoint's settings, from the copies of its config files in the fixtures.
    fn config() -> VisionConfig {
        let dir = fixtures().join("config");
        VisionConfig::load(&dir).unwrap()
    }

    #[derive(Deserialize)]
    struct Meta {
        file: String,
        file_sha256: String,
        resized: Option<[usize; 2]>,
        max_pixels: usize,
        grid: [usize; 3],
        tokens: usize,
        patches: usize,
        patches_sha256: String,
    }

    fn meta(name: &str) -> Meta {
        let text = fs::read_to_string(fixtures().join(format!("{name}.json"))).unwrap();
        serde_json::from_str(&text).unwrap()
    }

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().fold(String::new(), |mut s, b| {
            write!(s, "{b:02x}").unwrap();
            s
        })
    }

    fn patches_sha256(patches: &[f32]) -> String {
        let mut h = Sha256::new();
        for v in patches {
            h.update(v.to_le_bytes());
        }
        hex(&h.finalize())
    }

    /// The RGB pixels the reference processor normalized, after its decode, composite and
    /// resize.
    fn reference_rgb(name: &str) -> RgbImage {
        image::open(fixtures().join(format!("{name}.rgb.png")))
            .unwrap()
            .to_rgb8()
    }

    const NAMES: [&str; 6] = ["tiny", "square", "screenshot", "tall", "alpha", "huge"];
    /// Decoded exactly as PIL decodes them. The `image` crate's WebP decoder matches
    /// libwebp even for lossy WebP; only JPEG differs.
    const EXACT: [&str; 5] = ["tiny", "square", "tall", "alpha", "huge"];

    #[test]
    fn config_matches_checkpoint() {
        let cfg = config();
        assert_eq!(cfg.hidden_size, 1024);
        assert_eq!(cfg.num_hidden_layers, 23);
        assert_eq!(cfg.intermediate_size, 4096);
        assert_eq!(cfg.patch_len(), 1200);
        assert_eq!(cfg.token_size(), 40);
        assert_eq!(cfg.text_hidden_size, 2304);
        assert_eq!(cfg.mean, [127.5; 3]);
        assert_eq!(cfg.std, [127.5; 3]);
        assert_eq!(
            (cfg.content_image_id, cfg.image_token_id, cfg.end_message_id),
            (200_005, 200_054, 200_010)
        );
    }

    /// Patch order and normalization: patching the reference's own RGB pixels gives the
    /// reference's patches bit for bit.
    #[test]
    fn patches_match_reference_bit_for_bit() {
        let cfg = config();
        for name in ["tiny", "alpha", "tall", "screenshot"] {
            let m = meta(name);
            let (grid, patches) = patchify(&cfg, &reference_rgb(name));
            assert_eq!(grid, m.grid, "{name}");
            assert_eq!(patches.len(), m.patches * cfg.patch_len(), "{name}");
            assert_eq!(patches_sha256(&patches), m.patches_sha256, "{name}");
        }
    }

    /// The tiny image's patch values, checked one by one.
    #[test]
    fn tiny_patch_values() {
        let cfg = config();
        let bytes = fs::read(fixtures().join("tiny.safetensors")).unwrap();
        let st = safetensors::SafeTensors::deserialize(&bytes).unwrap();
        let t = st.tensor("patches").unwrap();
        assert_eq!(t.shape(), [4, 1200]);
        let want: Vec<f32> = t
            .data()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        let m = meta("tiny");
        let image = prepare(&cfg, &fs::read(fixtures().join(&m.file)).unwrap()).unwrap();
        assert_eq!(image.patches, want);
        // A pixel at (y, x) = (1, 2), channel G: first patch, offset (1 * 20 + 2) * 3 + 1.
        let v = reference_rgb("tiny").get_pixel(2, 1).0[1];
        assert_eq!(
            image.patches[(20 + 2) * 3 + 1],
            (f32::from(v) - 127.5) / 127.5
        );
        // Padding fills the rest: -1 on the 0 to 255 scale.
        assert_eq!(*image.patches.last().unwrap(), -128.5 / 127.5);
    }

    /// The whole pipeline from the encoded file: exact for PNG, GIF and WebP, and within a
    /// few levels for JPEG, where zune-jpeg and libjpeg-turbo differ.
    #[test]
    fn prepare_matches_reference() {
        let cfg = config();
        for name in NAMES {
            let m = meta(name);
            let bytes = fs::read(fixtures().join(&m.file)).unwrap();
            let image = prepare_with_max_pixels(&cfg, &bytes, m.max_pixels).unwrap();
            assert_eq!(hex(&image.hash), m.file_sha256, "{name}");
            assert_eq!(image.grid, m.grid, "{name}");
            assert_eq!(image.tokens(), m.tokens, "{name}");
            assert_eq!(image.num_patches(), m.patches, "{name}");
            if EXACT.contains(&name) {
                assert_eq!(patches_sha256(&image.patches), m.patches_sha256, "{name}");
                continue;
            }
            // JPEG: compare decoded levels with the reference's.
            let want = reference_rgb(name);
            let got = decode_rgb(&bytes).unwrap();
            assert!(m.resized.is_none());
            assert_eq!(got.dimensions(), want.dimensions(), "{name}");
            let diffs: Vec<u8> = got
                .as_raw()
                .iter()
                .zip(want.as_raw())
                .map(|(a, b)| a.abs_diff(*b))
                .collect();
            let max = *diffs.iter().max().unwrap();
            let mean = diffs.iter().map(|&d| f64::from(d)).sum::<f64>() / diffs.len() as f64;
            eprintln!("{name}: decoded levels differ from PIL by at most {max}, mean {mean:.4}");
            assert!(
                max <= LOSSY_MAX_LEVELS && mean <= LOSSY_MEAN_LEVELS,
                "{name}: {max} {mean}"
            );
        }
    }

    /// Bounds for JPEG against PIL, in 0 to 255 levels. The screenshot measures 4 and 0.041.
    const LOSSY_MAX_LEVELS: u8 = 6;
    const LOSSY_MEAN_LEVELS: f64 = 0.1;

    #[test]
    fn resize_matches_pil() {
        let m = meta("huge");
        let [w, h] = m.resized.unwrap();
        assert_eq!(limited_size(3000, 2000, m.max_pixels), Some((w, h)));
        let src = image::open(fixtures().join(&m.file)).unwrap().to_rgb8();
        let (grid, patches) = patchify(&config(), &resize_bicubic(&src, w, h));
        assert_eq!(grid, m.grid);
        assert_eq!(patches_sha256(&patches), m.patches_sha256);
    }

    #[test]
    fn size_limit() {
        assert_eq!(limited_size(1920, 1080, DEFAULT_MAX_PIXELS), None);
        assert_eq!(
            limited_size(1921, 1080, DEFAULT_MAX_PIXELS),
            Some((1920, 1079))
        );
        assert_eq!(limited_size(100_000, 1, 1000), Some((1000, 1)));
        for (w, h, max) in [(3000, 2000, 2_073_600), (4000, 3, 999), (7, 7, 48)] {
            let (nw, nh) = limited_size(w, h, max).unwrap();
            assert!(nw * nh <= max && nw >= 1 && nh >= 1, "{w}x{h} -> {nw}x{nh}");
        }
    }

    #[test]
    fn token_counts() {
        let cfg = config();
        let cases = [
            ((1, 1), 1),
            ((40, 40), 1),
            ((41, 40), 2),
            ((7, 5), 1),
            ((1024, 1024), 676),
            ((1920, 1080), 1296),
            ((360, 1500), 342),
            ((333, 217), 54),
        ];
        for ((w, h), tokens) in cases {
            let (grid, patches) = patchify(&cfg, &RgbImage::new(w, h));
            let image = PreparedImage {
                grid,
                patches,
                hash: [0; 32],
            };
            assert_eq!(image.tokens(), tokens, "{w}x{h}");
            assert_eq!(image.patches.len(), image.num_patches() * 1200);
            let ids = prompt_tokens(&cfg, &image);
            assert_eq!(ids.len(), tokens + 2);
            assert_eq!(ids[0], 200_005);
            assert_eq!(*ids.last().unwrap(), 200_010);
            assert!(ids[1..=tokens].iter().all(|&id| id == 200_054));
        }
    }

    #[test]
    fn composite_matches_processor() {
        let mut rgba = RgbaImage::new(16, 9);
        rgba.put_pixel(0, 0, image::Rgba([10, 20, 30, 255]));
        rgba.put_pixel(8, 0, image::Rgba([10, 20, 30, 0]));
        rgba.put_pixel(9, 8, image::Rgba([255, 0, 100, 128]));
        let rgb = composite(&rgba);
        assert_eq!(rgb.get_pixel(0, 0).0, [10, 20, 30]);
        assert_eq!(rgb.get_pixel(8, 0).0, [153; 3]);
        assert_eq!(rgb.get_pixel(1, 1).0, [204; 3]);
        // (255 * 128 + 204 * 127 + 127) / 255 = 230, (204 * 127 + 127) / 255 = 102,
        // (100 * 128 + 204 * 127 + 127) / 255 = 152.
        assert_eq!(rgb.get_pixel(9, 8).0, [230, 102, 152]);
    }

    #[test]
    fn rejects_other_formats() {
        let cfg = config();
        assert!(prepare(&cfg, b"not an image").is_err());
        let mut bmp = b"BM".to_vec();
        bmp.resize(64, 0);
        let err = prepare(&cfg, &bmp).unwrap_err();
        assert!(err.to_string().contains("unsupported"), "{err}");
    }
}
