//! Image input with `--vision`: content parts to image bytes, the limits on images, and the
//! prefix cache key of a prompt with images.
//!
//! An image arrives inline, base64 in a `data:` URL (OpenAI) or a `base64` source
//! (Anthropic). The server fetches no URLs. Before decoding, it reads the image's format
//! and size from its header and refuses anything but PNG, JPEG, WebP and GIF, and images of
//! more than [`MAX_DECODE_PIXELS`], so a small file that decodes to a huge image costs
//! nothing. Preprocessing ([`siphon::vision::prepare`]) shrinks an image of more than
//! `--max-image-pixels` to fit, keeping its aspect ratio, and pads its sides to whole image
//! tokens of 40 pixels. An image so narrow or wide that it still takes more tokens than
//! [`siphon::memory::max_image_tokens`] allows is refused.
//!
//! Each image becomes `<|content_image|>`, one `<|unused_200054|>` placeholder per image
//! token and `<|end_message|>`, after its message's header, where the chat template puts
//! it. Prefill writes the vision tower's rows over the placeholders' embeddings.

use std::{io::Cursor, sync::Arc};

use base64::{
    Engine as _, alphabet,
    engine::{DecodePaddingMode, GeneralPurpose, GeneralPurposeConfig},
};
use image::{ImageFormat, ImageReader};
use serde_json::Value;
use siphon::{
    memory,
    vision::{self, PreparedImage, VisionConfig},
};

use crate::serve::invalid::InvalidRequest;

/// Images a request may send unless `--max-images` says otherwise.
pub const DEFAULT_MAX_IMAGES: usize = 16;

/// Pixels of the largest image the server decodes, before shrinking it: 50 megapixels,
/// 150 MB as RGB.
pub const MAX_DECODE_PIXELS: usize = 50_000_000;

/// Request body bytes each image may add: a file of 15 MB in base64.
const IMAGE_BODY_BYTES: usize = 20 << 20;

/// Changes when the server's image preprocessing changes what an image's rows hold, so that
/// prefix cache entries saved to `--cache-dir` before then never match.
const KEY_VERSION: u32 = 1;

/// Key ids start here, above every token id, so a key run never equals a token.
const KEY_BASE: u32 = 0x8000_0000;

/// Base64 with or without padding, as clients send it.
const BASE64: GeneralPurpose = GeneralPurpose::new(
    &alphabet::STANDARD,
    GeneralPurposeConfig::new().with_decode_padding_mode(DecodePaddingMode::Indifferent),
);

/// What the server needs from image preprocessing: the checkpoint's, or a stand-in in
/// tests.
pub trait Preprocess: Send + Sync {
    fn prepare(&self, bytes: &[u8]) -> anyhow::Result<PreparedImage>;
    /// The ids one image takes in a prompt, markers included.
    fn prompt_tokens(&self, image: &PreparedImage) -> Vec<u32>;
}

impl Preprocess for VisionConfig {
    fn prepare(&self, bytes: &[u8]) -> anyhow::Result<PreparedImage> {
        vision::prepare(self, bytes)
    }

    fn prompt_tokens(&self, image: &PreparedImage) -> Vec<u32> {
        vision::prompt_tokens(self, image)
    }
}

/// An image of a prompt.
#[derive(Debug)]
pub struct PromptImage {
    /// The prompt position of its first placeholder.
    pub offset: usize,
    pub image: PreparedImage,
}

impl PromptImage {
    /// The prompt positions of its placeholders.
    pub fn positions(&self) -> std::ops::Range<usize> {
        self.offset..self.offset + self.image.tokens()
    }
}

/// `--vision`'s image handling.
pub struct Vision {
    preprocess: Box<dyn Preprocess>,
    /// Pixels on each side of the square one image token covers: 40.
    token_pixels: usize,
    /// `<|unused_200054|>`.
    placeholder: u32,
    /// `--max-image-pixels`: preprocessing shrinks larger images to this.
    max_pixels: usize,
    max_images: usize,
    /// Mixed into the key ids, so that entries saved under another pixel limit or
    /// preprocessing never serve a prompt.
    salt: u32,
}

impl Vision {
    /// `max_pixels` must be what `preprocess` shrinks images to.
    pub fn new(
        preprocess: Box<dyn Preprocess>,
        token_pixels: usize,
        placeholder: u32,
        max_pixels: usize,
        max_images: usize,
    ) -> Self {
        let salt = [KEY_VERSION, max_pixels as u32]
            .iter()
            .fold(0x811c_9dc5_u32, |h, &v| (h ^ v).wrapping_mul(0x0100_0193));
        Self {
            preprocess,
            token_pixels,
            placeholder,
            max_pixels,
            max_images,
            salt,
        }
    }

    /// Image tokens of the largest image a request may send.
    pub fn max_image_tokens(&self) -> usize {
        memory::max_image_tokens(self.token_pixels, self.max_pixels)
    }

    /// Request body bytes the images of one request may add.
    pub fn body_bytes(&self) -> usize {
        self.max_images * IMAGE_BODY_BYTES
    }

    /// The image of one content part, read and prepared, and the ids it takes in the
    /// prompt with the offset of its first placeholder among them.
    pub fn image(&self, part: &Value) -> Result<(PreparedImage, Vec<u32>, usize), InvalidRequest> {
        let bytes = image_bytes(part)?;
        let size = image_size(&bytes)?;
        let image = self
            .preprocess
            .prepare(&bytes)
            .map_err(|e| InvalidRequest::new(format!("the image could not be read: {e:#}")))?;
        let tokens = image.tokens();
        if tokens == 0 || tokens > self.max_image_tokens() {
            return Err(InvalidRequest::new(format!(
                "the {}x{} image takes {tokens} image tokens once shrunk to {} pixels; this \
                 server takes at most {}, so an image this narrow or wide does not fit",
                size.0,
                size.1,
                self.max_pixels,
                self.max_image_tokens()
            )));
        }
        let ids = self.preprocess.prompt_tokens(&image);
        let first = ids.iter().position(|&id| id == self.placeholder);
        let run = first.map_or(0, |f| {
            ids[f..]
                .iter()
                .take_while(|&&id| id == self.placeholder)
                .count()
        });
        let count = ids.iter().filter(|&&id| id == self.placeholder).count();
        match first {
            Some(first) if run == tokens && count == tokens => Ok((image, ids, first)),
            _ => Err(InvalidRequest::new(format!(
                "the image's prompt ids hold {count} placeholders, not one run of {tokens}"
            ))),
        }
    }

    /// The prefix cache key of `prompt` with `images`: the tokens, with each image's
    /// placeholders replaced by ids above every token id, from the image's hash. Placeholder
    /// `k` of an image takes 31 bits of the hash's word `k % 8`, so an image of 8 or more
    /// tokens is keyed by all 256 bits. `None` for a prompt without images, whose key is its
    /// tokens.
    pub fn cache_key(&self, prompt: &[u32], images: &[PromptImage]) -> Option<Vec<u32>> {
        if images.is_empty() {
            return None;
        }
        let mut key = prompt.to_vec();
        for image in images {
            let words: Vec<u32> = image
                .image
                .hash
                .as_chunks::<4>()
                .0
                .iter()
                .map(|w| u32::from_le_bytes(*w))
                .collect();
            for (k, id) in key[image.positions()].iter_mut().enumerate() {
                *id = KEY_BASE | ((words[k % words.len()] ^ self.salt) & !KEY_BASE);
            }
        }
        Some(key)
    }
}

/// The image's width and height from its header, if it is PNG, JPEG, WebP or GIF of at
/// most [`MAX_DECODE_PIXELS`].
fn image_size(bytes: &[u8]) -> Result<(u32, u32), InvalidRequest> {
    let reader = ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|e| InvalidRequest::new(format!("the image could not be read: {e}")))?;
    if !matches!(
        reader.format(),
        Some(ImageFormat::Png | ImageFormat::Jpeg | ImageFormat::WebP | ImageFormat::Gif)
    ) {
        return Err(InvalidRequest::new(
            "the image is not PNG, JPEG, WebP or GIF",
        ));
    }
    let (width, height) = reader
        .into_dimensions()
        .map_err(|e| InvalidRequest::new(format!("the image's header could not be read: {e}")))?;
    let pixels = width as usize * height as usize;
    if pixels == 0 || pixels > MAX_DECODE_PIXELS {
        return Err(InvalidRequest::new(format!(
            "the image is {width}x{height} pixels; this server decodes at most \
             {MAX_DECODE_PIXELS}"
        )));
    }
    Ok((width, height))
}

/// Collects the images of one prompt as the chat template renders it.
pub struct Collector<'a> {
    vision: Option<&'a Vision>,
    pub images: Vec<PromptImage>,
}

impl<'a> Collector<'a> {
    pub fn new(vision: Option<&'a Vision>) -> Self {
        Self {
            vision,
            images: Vec::new(),
        }
    }

    /// The ids of an image part whose ids start at prompt position `at`.
    pub fn image(&mut self, part: &Value, at: usize) -> Result<Vec<u32>, InvalidRequest> {
        let Some(vision) = self.vision else {
            return Err(InvalidRequest::new(
                "image input is off on this server; start it with --vision to accept images",
            ));
        };
        if self.images.len() == vision.max_images {
            return Err(InvalidRequest::new(format!(
                "the request has more than {} images, the most this server takes; \
                 --max-images sets the limit",
                vision.max_images
            )));
        }
        let (image, ids, first) = vision.image(part)?;
        self.images.push(PromptImage {
            offset: at + first,
            image,
        });
        Ok(ids)
    }
}

/// A prompt's images, shared by the jobs of its choices.
pub type Images = Arc<[PromptImage]>;

/// What the vision tower did for one job's prompt.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ImageStats {
    /// Images the tower encoded, and the milliseconds of GPU time the encodes took.
    pub encoded: usize,
    pub ms: f64,
    /// Images inside the prompt's prefix cache hit, which the tower did not encode.
    pub cached: usize,
}

/// The bytes of an image content part: OpenAI's `image_url` and `input_image` with a
/// `data:` URL, or Anthropic's `image` with a `base64` source.
pub fn image_bytes(part: &Value) -> Result<Vec<u8>, InvalidRequest> {
    if let Some(source) = part.get("source") {
        return anthropic_source(source);
    }
    let url = ["image_url", "image", "url"]
        .iter()
        .find_map(|key| part.get(*key))
        .map(|v| match v {
            Value::Object(o) => o.get("url").unwrap_or(&Value::Null),
            other => other,
        })
        .and_then(Value::as_str)
        .ok_or_else(|| {
            InvalidRequest::missing("image_url.url", "a data:image/...;base64,... URL string")
        })?;
    data_url(url)
}

/// The bytes of a `data:image/...;base64,` URL.
fn data_url(url: &str) -> Result<Vec<u8>, InvalidRequest> {
    let Some(rest) = url.strip_prefix("data:") else {
        let what = if url.starts_with("http://") || url.starts_with("https://") {
            "this server does not fetch image URLs"
        } else {
            "an image URL must be a data: URL"
        };
        return Err(InvalidRequest::new(format!(
            "{what}, got {}; send the image inline as data:image/png;base64,...",
            crate::serve::invalid::quote(url)
        )));
    };
    let (header, data) = rest
        .split_once(',')
        .ok_or_else(|| InvalidRequest::new("the image's data: URL has no comma"))?;
    let Some(media_type) = header.strip_suffix(";base64") else {
        return Err(InvalidRequest::new("the image's data: URL must be base64"));
    };
    check_media_type(media_type)?;
    decode(data)
}

/// The bytes of an Anthropic image source.
fn anthropic_source(source: &Value) -> Result<Vec<u8>, InvalidRequest> {
    match source.get("type").and_then(Value::as_str) {
        Some("base64") => {
            let media_type = source
                .get("media_type")
                .and_then(Value::as_str)
                .unwrap_or("");
            check_media_type(media_type)?;
            let data = source.get("data").and_then(Value::as_str).ok_or_else(|| {
                InvalidRequest::missing("source.data", "the image's base64 string")
            })?;
            decode(data)
        }
        Some("url") => Err(InvalidRequest::field(
            "source.type",
            "\"url\" is not supported: this server does not fetch image URLs; send a \
             \"base64\" source",
        )),
        _ => Err(InvalidRequest::expected(
            "source.type",
            "\"base64\"",
            source.get("type").unwrap_or(&Value::Null),
        )),
    }
}

fn check_media_type(media_type: &str) -> Result<(), InvalidRequest> {
    match media_type {
        "image/png" | "image/jpeg" | "image/jpg" | "image/webp" | "image/gif" => Ok(()),
        _ => Err(InvalidRequest::new(format!(
            "image type {media_type:?} is not supported; send PNG, JPEG, WebP or GIF"
        ))),
    }
}

fn decode(data: &str) -> Result<Vec<u8>, InvalidRequest> {
    let data: String = data.chars().filter(|c| !c.is_ascii_whitespace()).collect();
    BASE64
        .decode(data)
        .map_err(|e| InvalidRequest::new(format!("the image's base64 does not decode: {e}")))
}

#[cfg(test)]
pub(crate) mod tests {
    use serde_json::json;

    use super::*;

    /// A 1x1 PNG.
    pub(crate) const PNG_1X1: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk\
                                      +M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==";

    /// A PNG of `width` x `height` with no pixel data: enough for the size check.
    pub(crate) fn png_header(width: u32, height: u32) -> String {
        let chunk = |kind: &[u8], data: &[u8]| {
            let body: Vec<u8> = kind.iter().chain(data).copied().collect();
            let crc = !body.iter().fold(!0u32, |mut c, &b| {
                c ^= u32::from(b);
                for _ in 0..8 {
                    c = (c >> 1) ^ (0xedb8_8320 & (c & 1).wrapping_neg());
                }
                c
            });
            let mut out = (data.len() as u32).to_be_bytes().to_vec();
            out.extend(body);
            out.extend(crc.to_be_bytes());
            out
        };
        let mut header = width.to_be_bytes().to_vec();
        header.extend(height.to_be_bytes());
        header.extend([8, 6, 0, 0, 0]);
        let mut bytes = b"\x89PNG\r\n\x1a\n".to_vec();
        bytes.extend(chunk(b"IHDR", &header));
        bytes.extend(chunk(b"IDAT", &[]));
        bytes.extend(chunk(b"IEND", &[]));
        BASE64.encode(bytes)
    }

    /// Preprocessing that reads only the header: a grid of 20-pixel patches, padded to whole
    /// 40-pixel tokens, and a hash of the bytes. It shrinks nothing.
    pub(crate) struct Fake;

    /// Stand-in marker ids around the placeholders.
    pub(crate) const START: u32 = 200_005;
    pub(crate) const PLACEHOLDER: u32 = 200_054;
    pub(crate) const END: u32 = 200_010;

    impl Preprocess for Fake {
        fn prepare(&self, bytes: &[u8]) -> anyhow::Result<PreparedImage> {
            let (width, height) = ImageReader::new(Cursor::new(bytes))
                .with_guessed_format()?
                .into_dimensions()?;
            let mut hash = [0u8; 32];
            for (i, b) in bytes.iter().enumerate() {
                hash[i % 32] = hash[i % 32].wrapping_mul(31).wrapping_add(*b);
            }
            let patches = |side: u32| 2 * side.div_ceil(40) as usize;
            Ok(PreparedImage {
                grid: [1, patches(height), patches(width)],
                patches: Vec::new(),
                hash,
            })
        }

        fn prompt_tokens(&self, image: &PreparedImage) -> Vec<u32> {
            let mut ids = vec![START];
            ids.extend(std::iter::repeat_n(PLACEHOLDER, image.tokens()));
            ids.push(END);
            ids
        }
    }

    /// Images shrink to 1,600 x 1,600 pixels at most, so up to 41 x 41 tokens.
    pub(crate) fn vision(max_images: usize) -> Vision {
        Vision::new(Box::new(Fake), 40, PLACEHOLDER, 1600 * 1600, max_images)
    }

    pub(crate) fn image_url(width: u32, height: u32) -> Value {
        json!({"type": "image_url", "image_url": {"url": format!(
            "data:image/png;base64,{}", png_header(width, height))}})
    }

    #[test]
    fn image_parts_decode_data_urls_and_base64_sources() {
        let png = BASE64.decode(PNG_1X1).unwrap();
        let url = format!("data:image/png;base64,{PNG_1X1}");
        for part in [
            json!({"type": "image_url", "image_url": {"url": url, "detail": "high"}}),
            json!({"type": "image_url", "image_url": url}),
            json!({"type": "input_image", "image_url": url}),
            json!({"type": "image", "image": url}),
            json!({"type": "image", "source": {"type": "base64", "media_type": "image/png",
                   "data": PNG_1X1}}),
        ] {
            assert_eq!(image_bytes(&part).unwrap(), png, "{part}");
        }
        // Unpadded and wrapped base64 decodes too.
        let wrapped = format!("data:image/png;base64,{}\n", PNG_1X1.trim_end_matches('='));
        assert_eq!(
            image_bytes(&json!({"type": "image_url", "image_url": wrapped})).unwrap(),
            png
        );
    }

    #[test]
    fn image_parts_refuse_urls_and_other_types() {
        let refused = |part: Value| image_bytes(&part).unwrap_err().to_string();
        assert!(
            refused(json!({"type": "image_url", "image_url": {"url": "https://x/y.png"}}))
                .starts_with("this server does not fetch image URLs")
        );
        assert_eq!(
            refused(json!({"type": "image", "source": {"type": "url", "url": "https://x"}})),
            "source.type: \"url\" is not supported: this server does not fetch image URLs; \
             send a \"base64\" source"
        );
        assert!(
            refused(json!({"type": "image_url", "image_url": "data:image/tiff;base64,AAAA"}))
                .contains("not supported")
        );
        assert!(
            refused(json!({"type": "image_url", "image_url": "data:image/png,rawbytes"}))
                .contains("must be base64")
        );
        assert!(
            refused(json!({"type": "image_url", "image_url": "data:image/png;base64,***"}))
                .contains("does not decode")
        );
        assert_eq!(
            refused(json!({"type": "image_url"})),
            "image_url.url: missing; expected a data:image/...;base64,... URL string"
        );
    }

    #[test]
    fn limits_count_tokens_decoded_pixels_and_images() {
        let v = vision(2);
        assert_eq!(v.max_image_tokens(), 41 * 41);
        // 400 x 400 pixels is 100 tokens, and 401 pads to 11 x 10.
        let (image, ids, first) = v.image(&image_url(400, 400)).unwrap();
        assert_eq!((image.tokens(), ids.len(), first), (100, 102, 1));
        assert_eq!(v.image(&image_url(401, 400)).unwrap().0.tokens(), 110);
        // Fake shrinks nothing: 1,641 x 1,641 pixels pads to 42 x 42 tokens, past the
        // limit, as an image too narrow to shrink into it would.
        let refused = v.image(&image_url(1641, 1641)).unwrap_err().to_string();
        assert!(refused.contains("1764 image tokens"), "{refused}");
        let refused = v.image(&image_url(10_000, 5001)).unwrap_err().to_string();
        assert!(refused.contains("decodes at most"), "{refused}");
        let not_an_image = json!({"type": "image_url", "image_url": "data:image/png;base64,AAAA"});
        assert!(
            v.image(&not_an_image)
                .unwrap_err()
                .to_string()
                .contains("not PNG")
        );

        let mut collector = Collector::new(Some(&v));
        assert_eq!(collector.image(&image_url(40, 80), 10).unwrap().len(), 4);
        collector.image(&image_url(40, 40), 20).unwrap();
        let third = collector
            .image(&image_url(40, 40), 30)
            .unwrap_err()
            .to_string();
        assert!(third.contains("--max-images"), "{third}");
        let offsets: Vec<usize> = collector.images.iter().map(|i| i.offset).collect();
        assert_eq!(offsets, [11, 21]);
        // Without --vision, today's refusal.
        let off = Collector::new(None)
            .image(&image_url(40, 40), 0)
            .unwrap_err()
            .to_string();
        assert_eq!(
            off,
            "image input is off on this server; start it with --vision to accept images"
        );
    }

    /// The checkpoint's preprocessing config, at its default pixel limit.
    fn checkpoint_config() -> VisionConfig {
        let dir = std::env::var_os("SIPHON_MODEL").expect("set SIPHON_MODEL");
        VisionConfig::load(std::path::Path::new(&dir)).unwrap()
    }

    /// siphon-serve's prompt for each reference fixture's image and question equals the one
    /// `reference/vision_capture.py` built: the same ids, so the same token count.
    #[test]
    #[ignore = "needs the tokenizer (SIPHON_MODEL)"]
    fn renders_the_reference_fixture_prompts() {
        let mut vocab = crate::serve::chat::tests::vocab();
        let config = checkpoint_config();
        let (start, placeholder, end) = (
            config.content_image_id,
            config.image_token_id,
            config.end_message_id,
        );
        let side = config.token_size();
        let max_pixels = vision::DEFAULT_MAX_PIXELS;
        vocab.set_vision(Some(Vision::new(
            Box::new(config),
            side,
            placeholder,
            max_pixels,
            4,
        )));
        let s = vocab.special;
        assert_eq!(end, s.end_message);
        let fixtures = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../reference/fixtures/vision");
        for name in ["tiny", "alpha", "tall", "square", "screenshot", "huge"] {
            let meta: Value = serde_json::from_str(
                &std::fs::read_to_string(fixtures.join(format!("{name}.json"))).unwrap(),
            )
            .unwrap();
            let file = meta["file"].as_str().unwrap();
            let media_type = match file.rsplit('.').next() {
                Some("png") => "image/png",
                Some("gif") => "image/gif",
                Some("webp") => "image/webp",
                _ => "image/jpeg",
            };
            let data = BASE64.encode(std::fs::read(fixtures.join(file)).unwrap());
            let question = meta["text"]["prompt"].as_str().unwrap();
            let messages = json!([{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": format!(
                    "data:{media_type};base64,{data}")}},
                {"type": "text", "text": question},
            ]}]);
            let rendered = vocab
                .render_prompt(
                    messages.as_array().unwrap(),
                    &[],
                    0.0,
                    &crate::serve::chat::Ending::Generate,
                )
                .unwrap();
            let tokens = usize::try_from(meta["tokens"].as_u64().unwrap()).unwrap();
            let mut want = vec![s.message_system, s.content_text];
            want.extend(vocab.encode("Thinking effort level: 0"));
            want.extend([s.end_message, s.message_user, start]);
            want.extend(std::iter::repeat_n(placeholder, tokens));
            want.extend([s.end_message, s.message_user, s.content_text]);
            want.extend(vocab.encode(question));
            want.extend([s.end_message, s.message_model]);
            assert_eq!(rendered.ids, want, "{name}");
            assert_eq!(
                rendered.ids.len() as u64,
                meta["text"]["prompt_tokens"].as_u64().unwrap(),
                "{name}"
            );
            let image = &rendered.images[0];
            assert_eq!(image.image.tokens(), tokens, "{name}");
            assert_eq!(rendered.ids[image.positions()].len(), tokens);
            assert!(
                rendered.ids[image.positions()]
                    .iter()
                    .all(|&id| id == placeholder)
            );
        }
    }

    #[test]
    #[ignore = "needs the tokenizer (SIPHON_MODEL)"]
    fn chat_images_take_their_place_in_the_template() {
        let mut vocab = crate::serve::chat::tests::vocab();
        let messages = json!([
            {"role": "user", "content": [
                {"type": "text", "text": "Compare"},
                image_url(40, 40),
                {"type": "text", "text": "with"},
                image_url(80, 80),
            ]},
        ]);
        let messages = messages.as_array().unwrap();
        let generate = &crate::serve::chat::Ending::Generate;
        // Without --vision, today's refusal.
        let off = vocab
            .render_prompt(messages, &[], 0.0, generate)
            .unwrap_err()
            .to_string();
        assert_eq!(
            off,
            "messages[0].content[1]: image input is off on this server; start it with --vision \
             to accept images"
        );

        vocab.set_vision(Some(vision(4)));
        let rendered = vocab.render_prompt(messages, &[], 0.0, generate).unwrap();
        let text = vocab.decode(&rendered.ids);
        assert!(
            text.ends_with(
                "<|message_user|><|content_text|>Compare<|end_message|>\
                 <|message_user|><|content_image|><|unused_200054|><|end_message|>\
                 <|message_user|><|content_text|>with<|end_message|>\
                 <|message_user|><|content_image|><|unused_200054|><|unused_200054|>\
                 <|unused_200054|><|unused_200054|><|end_message|><|message_model|>"
            ),
            "{text}"
        );
        let runs: Vec<(usize, usize)> = rendered
            .images
            .iter()
            .map(|i| (i.offset, i.image.tokens()))
            .collect();
        let first = rendered
            .ids
            .iter()
            .position(|&id| id == PLACEHOLDER)
            .unwrap();
        // The second run starts after the first's end marker, the text block of "with" and
        // its own header and start marker.
        assert_eq!(runs, [(first, 1), (first + 8, 4)]);
        // The tokenizer route counts the same prompt.
        assert_eq!(
            vocab.render_to(messages, &[], 0.0, generate).unwrap(),
            rendered.ids
        );
        // Images belong to user and tool messages only.
        let assistant = json!([{"role": "user", "content": "Hi"},
                               {"role": "assistant", "content": [image_url(40, 40)]}]);
        let refused = vocab
            .render_prompt(assistant.as_array().unwrap(), &[], 0.0, generate)
            .unwrap_err()
            .to_string();
        assert!(refused.contains("user and tool messages"), "{refused}");
    }

    #[test]
    fn keys_replace_placeholders_by_the_image_hash() {
        let v = vision(4);
        let image = |offset, seed: u8| PromptImage {
            offset,
            image: PreparedImage {
                grid: [1, 4, 10],
                patches: Vec::new(),
                hash: std::array::from_fn(|i| seed.wrapping_add(i as u8)),
            },
        };
        let prompt: Vec<u32> = (0..30).collect();
        assert_eq!(v.cache_key(&prompt, &[]), None);
        let a = v.cache_key(&prompt, &[image(5, 1)]).unwrap();
        let b = v.cache_key(&prompt, &[image(5, 2)]).unwrap();
        assert_eq!(a.len(), prompt.len());
        assert_eq!(a[..5], prompt[..5]);
        assert_eq!(a[15..], prompt[15..]);
        assert!(a[5..15].iter().all(|&id| id >= KEY_BASE));
        assert!(a[5..15].iter().zip(&b[5..15]).all(|(x, y)| x != y));
        assert_eq!(a, v.cache_key(&prompt, &[image(5, 1)]).unwrap());
        // The same image under another pixel limit keys differently.
        let smaller = Vision::new(Box::new(Fake), 40, PLACEHOLDER, 800 * 800, 4);
        let c = smaller.cache_key(&prompt, &[image(5, 1)]).unwrap();
        assert!(a[5..15].iter().zip(&c[5..15]).all(|(x, y)| x != y));
    }
}
