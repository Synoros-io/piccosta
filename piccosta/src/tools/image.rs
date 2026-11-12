// Images a tool shows the model: which files count as images, and shrinking big ones to what
// providers accept inline.
//
// Based on pi 1.4.0 (MIT, see LICENSE.md): packages/coding-agent/src/utils/mime.ts,
// image-process.ts and image-resize-core.ts. pi does the pixel work with Photon, a WebAssembly
// build of the `image` crate; here the `image` crate is called directly.

use base64::Engine;
use image::{DynamicImage, ImageDecoder, ImageEncoder, ImageReader};
use std::io::Cursor;

// Bytes read from the start of a file to decide whether it's an image (mime.ts:3).
pub const SNIFF_BYTES: usize = 4100;

// What an image is fitted to before it goes inline: a box in pixels, a size in base64 bytes, and
// the JPEG quality tried first, after which JPEG steps through JPEG_QUALITIES.
#[derive(Debug, Clone, Copy)]
pub struct Limits {
    pub max_width: u32,
    pub max_height: u32,
    pub max_bytes: usize,
    pub jpeg_quality: u8,
}

// Anthropic's: 2000x2000 pixels and 4.5 MB of base64, under its 5 MB per image (pi's
// image-resize-core.ts:31-38). The limits for a model whose catalog entry gives none.
const ANTHROPIC_LIMITS: Limits = Limits { max_width: 2000, max_height: 2000, max_bytes: 4_718_592, jpeg_quality: 80 };
const JPEG_QUALITIES: [u8; 4] = [85, 70, 55, 40];

impl Limits {
    // A model's own limits, from its catalog `inputLimits` (`images.resize`: maxWidth, maxHeight,
    // maxBytes, jpegQuality); each one it doesn't give is Anthropic's.
    pub fn of(input_limits: Option<&serde_json::Value>) -> Limits {
        let resize = input_limits.and_then(|limits| limits.pointer("/images/resize"));
        let field = |key: &str| resize.and_then(|resize| resize.get(key)).and_then(serde_json::Value::as_u64);
        let fallback = ANTHROPIC_LIMITS;
        Limits {
            max_width: field("maxWidth").and_then(|value| value.try_into().ok()).unwrap_or(fallback.max_width),
            max_height: field("maxHeight").and_then(|value| value.try_into().ok()).unwrap_or(fallback.max_height),
            max_bytes: field("maxBytes").and_then(|value| value.try_into().ok()).unwrap_or(fallback.max_bytes),
            jpeg_quality: field("jpegQuality").and_then(|value| value.try_into().ok()).unwrap_or(fallback.jpeg_quality),
        }
    }
}
// Each pass that still doesn't fit shrinks both sides to three quarters.
const SHRINK_FACTOR: f64 = 0.75;

const PNG_SIGNATURE: [u8; 8] = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];

fn u32_be(bytes: &[u8], at: usize) -> Option<u32> {
    bytes.get(at..at + 4).map(|word| u32::from_be_bytes([word[0], word[1], word[2], word[3]]))
}
fn u32_le(bytes: &[u8], at: usize) -> Option<u32> {
    bytes.get(at..at + 4).map(|word| u32::from_le_bytes([word[0], word[1], word[2], word[3]]))
}
fn u16_le(bytes: &[u8], at: usize) -> Option<u16> {
    bytes.get(at..at + 2).map(|word| u16::from_le_bytes([word[0], word[1]]))
}

// An animated PNG has an acTL chunk before its first IDAT. Providers take only still images.
fn is_animated_png(bytes: &[u8]) -> bool {
    let mut offset = PNG_SIGNATURE.len();
    while offset + 8 <= bytes.len() {
        let Some(length) = u32_be(bytes, offset) else { return false };
        match &bytes[offset + 4..offset + 8] {
            b"acTL" => return true,
            b"IDAT" => return false,
            _ => {}
        }
        let next = offset + 12 + length as usize;
        if next <= offset || next > bytes.len() {
            return false;
        }
        offset = next;
    }
    false
}

// A BMP header that's self-consistent: sane sizes and offsets, one color plane, a real bit depth.
fn is_bmp(bytes: &[u8]) -> bool {
    let (Some(file_size), Some(pixel_offset), Some(header_size)) = (u32_le(bytes, 2), u32_le(bytes, 10), u32_le(bytes, 14)) else {
        return false;
    };
    if bytes.len() < 26 || (file_size != 0 && file_size < 26) || pixel_offset < 14 + header_size {
        return false;
    }
    if file_size != 0 && pixel_offset >= file_size {
        return false;
    }
    let (planes, bits) = match header_size {
        12 => (u16_le(bytes, 22), u16_le(bytes, 24)),
        40..=124 => (u16_le(bytes, 26), u16_le(bytes, 28)),
        _ => return false,
    };
    planes == Some(1) && bits.is_some_and(|bits| [1, 4, 8, 16, 24, 32].contains(&bits))
}

// Input: the first bytes of a file. Output: its image type when it's one we send, by content,
// never by file name (mime.ts:6-23). JPEG-LS (FF D8 FF F7) and animated PNG don't count.
pub fn sniff_mime_type(bytes: &[u8]) -> Option<&'static str> {
    if bytes.starts_with(&[0xff, 0xd8, 0xff]) {
        return (bytes.get(3) != Some(&0xf7)).then_some("image/jpeg");
    }
    if bytes.starts_with(&PNG_SIGNATURE) {
        let is_png = bytes.len() >= 16 && u32_be(bytes, 8) == Some(13) && &bytes[12..16] == b"IHDR";
        return (is_png && !is_animated_png(bytes)).then_some("image/png");
    }
    if bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a") {
        return Some("image/gif");
    }
    if bytes.starts_with(b"RIFF") && bytes.get(8..12) == Some(b"WEBP") {
        return Some("image/webp");
    }
    if bytes.starts_with(b"BM") && is_bmp(bytes) {
        return Some("image/bmp");
    }
    None
}

// An image ready to send, with the notes the model gets beside it.
pub struct Prepared {
    pub data: String,
    pub mime_type: String,
    pub hints: Vec<String>,
    // The size it goes out at.
    pub width: u32,
    pub height: u32,
}

fn base64_len(bytes: usize) -> usize {
    bytes.div_ceil(3) * 4
}

pub(crate) fn encode_png(image: &DynamicImage) -> Option<Vec<u8>> {
    let mut out = Vec::new();
    let rgba = image.to_rgba8();
    image::codecs::png::PngEncoder::new(&mut out)
        .write_image(rgba.as_raw(), rgba.width(), rgba.height(), image::ExtendedColorType::Rgba8)
        .ok()?;
    Some(out)
}

fn encode_jpeg(image: &DynamicImage, quality: u8) -> Option<Vec<u8>> {
    let mut out = Vec::new();
    let rgb = image.to_rgb8();
    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, quality)
        .write_image(rgb.as_raw(), rgb.width(), rgb.height(), image::ExtendedColorType::Rgb8)
        .ok()?;
    Some(out)
}

// Decode with the file's EXIF orientation applied, so a phone photo isn't sideways.
pub(super) fn decode(bytes: &[u8]) -> Option<DynamicImage> {
    let reader = ImageReader::new(Cursor::new(bytes)).with_guessed_format().ok()?;
    let mut decoder = reader.into_decoder().ok()?;
    let orientation = decoder.orientation().ok();
    let mut image = DynamicImage::from_decoder(decoder).ok()?;
    if let Some(orientation) = orientation {
        image.apply_orientation(orientation);
    }
    Some(image)
}

// A plugin may supply compressed images, not just screenshots. Four default-sized 2000x2000
// frames is the source-pixel ceiling; 64 MB bounds its RGBA8 decode before any normalization.
pub(crate) const MAX_PLUGIN_RESPONSE_PIXELS: u64 = 4 * 2000 * 2000;
const MAX_PLUGIN_DECODE_BYTES: u64 = MAX_PLUGIN_RESPONSE_PIXELS * 4;

// Validate dimensions and decoder allocation before decoding. Always normalize plugin bytes,
// then fit that PNG to the provider: a small JPEG can become a much larger PNG.
pub(super) fn prepare_plugin(bytes: &[u8], limits: Limits, pixels_left: &mut u64) -> Result<Prepared, String> {
    let image = decode_guarded(bytes, pixels_left)?;
    // No original bytes to forward: fit the decoded frame first, then encode a clean PNG (or
    // JPEG when PNG won't fit). Encoding the full source first needlessly expands large JPEGs.
    prepare_decoded(&[], "image/png", limits, image)
}

// Input: compressed bytes and the remaining decoded-pixel allowance. Check dimensions and
// allocation before decoding; both plugins and the scratchpad need this guard on small zip bombs.
pub(crate) fn decode_guarded(bytes: &[u8], pixels_left: &mut u64) -> Result<DynamicImage, String> {
    let mut reader = ImageReader::new(Cursor::new(bytes)).with_guessed_format().map_err(|e| format!("bad image: {e}"))?;
    let mut budget = image::Limits::default();
    budget.max_alloc = Some(MAX_PLUGIN_DECODE_BYTES);
    reader.limits(budget);
    let mut decoder = reader.into_decoder().map_err(|e| format!("bad image: {e}"))?;
    let (width, height) = decoder.dimensions();
    *pixels_left = pixels_left.checked_sub(u64::from(width) * u64::from(height))
        .ok_or_else(|| format!("bad image: response decoded-pixel budget exceeded ({width}x{height}, {pixels_left} pixels left)"))?;
    let orientation = decoder.orientation().ok();
    let mut image = DynamicImage::from_decoder(decoder).map_err(|e| format!("bad image: could not decode: {e}"))?;
    if let Some(orientation) = orientation { image.apply_orientation(orientation); }
    Ok(image)
}

// Input: an image file's bytes, its sniffed type and the limits to fit. Output: the image as it goes to the model,
// or the note that replaces it when it can't be made small enough (image-process.ts:75-110).
// BMP isn't an inline type anywhere, so it's always re-encoded. An image already within both
// limits goes out as it was.
pub fn prepare(bytes: &[u8], mime_type: &str, limits: Limits) -> Result<Prepared, String> {
    let image = decode(bytes).ok_or_else(|| {
        if mime_type == "image/bmp" {
            "[Image omitted: could not be converted to a supported inline image format.]".to_string()
        } else {
            "[Image omitted: could not be resized below the inline image size limit.]".to_string()
        }
    })?;
    prepare_decoded(bytes, mime_type, limits, image)
}

fn prepare_decoded(bytes: &[u8], mime_type: &str, limits: Limits, image: DynamicImage) -> Result<Prepared, String> {
    let Limits { max_width, max_height, max_bytes, jpeg_quality } = limits;
    let mut hints = Vec::new();
    let converted_from = (mime_type == "image/bmp").then_some(mime_type);
    let (width, height) = (image.width(), image.height());

    if !bytes.is_empty() && converted_from.is_none() && width <= max_width && height <= max_height && base64_len(bytes.len()) < max_bytes {
        let data = base64::engine::general_purpose::STANDARD.encode(bytes);
        return Ok(Prepared { data, mime_type: mime_type.to_string(), hints, width, height });
    }

    // Fit inside the box first, keeping the aspect ratio, then try PNG and every JPEG quality at
    // that size, and shrink by a quarter until something fits (image-resize-core.ts:113-161).
    let (mut target_width, mut target_height) = (width, height);
    if target_width > max_width {
        target_height = ((target_height as f64 * max_width as f64) / target_width as f64).round() as u32;
        target_width = max_width;
    }
    if target_height > max_height {
        target_width = ((target_width as f64 * max_height as f64) / target_height as f64).round() as u32;
        target_height = max_height;
    }
    let (mut current_width, mut current_height) = (target_width.max(1), target_height.max(1));
    loop {
        let resized = if (current_width, current_height) == (width, height) {
            image.clone()
        } else {
            image.resize_exact(current_width, current_height, image::imageops::FilterType::Lanczos3)
        };
        let candidates = std::iter::once(encode_png(&resized).map(|png| (png, "image/png")))
            .chain(std::iter::once(jpeg_quality).chain(JPEG_QUALITIES.into_iter().filter(|quality| *quality != jpeg_quality)).map(|quality| encode_jpeg(&resized, quality).map(|jpeg| (jpeg, "image/jpeg"))));
        for (encoded, encoded_type) in candidates.flatten() {
            if base64_len(encoded.len()) < max_bytes {
                if let Some(from) = converted_from
                    && from != encoded_type
                {
                    hints.push(format!("[Image converted from {from} to {encoded_type}.]"));
                }
                if (current_width, current_height) != (width, height) {
                    let scale = crate::js::to_fixed(width as f64 / current_width as f64, 2);
                    hints.push(format!(
                        "[Image: original {width}x{height}, displayed at {current_width}x{current_height}. Multiply coordinates by {scale} to map to original image.]"
                    ));
                }
                let data = base64::engine::general_purpose::STANDARD.encode(&encoded);
                return Ok(Prepared { data, mime_type: encoded_type.to_string(), hints, width: current_width, height: current_height });
            }
        }
        if current_width == 1 && current_height == 1 {
            break;
        }
        let next_width = if current_width == 1 { 1 } else { ((current_width as f64 * SHRINK_FACTOR).floor() as u32).max(1) };
        let next_height = if current_height == 1 { 1 } else { ((current_height as f64 * SHRINK_FACTOR).floor() as u32).max(1) };
        if (next_width, next_height) == (current_width, current_height) {
            break;
        }
        (current_width, current_height) = (next_width, next_height);
    }
    Err("[Image omitted: could not be resized below the inline image size limit.]".to_string())
}
