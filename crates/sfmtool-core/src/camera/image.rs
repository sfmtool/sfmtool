// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Image containers shared across the crate: [`ImageU8`], a packed `u8` image
//! with 1, 3 or 4 interleaved channels, read from and written to image files;
//! [`ImageU8Pyramid`], a stack of
//! successive 2x box-filtered downsamples of one; and [`ImageF32WithGrad`], a
//! float image carrying its per-channel `x` and `y` gradients. The resampling
//! functions in [`remap`](super::remap) read and write them, and the patch,
//! spherical-tile, bench and photograph-cache code hold them.

/// The JPEG quality a caller of [`ImageU8::write`] passes when it has no
/// reason to choose another. It is 95, the default of OpenCV's
/// `cv2.imwrite`, so a JPEG keeps the quality OpenCV gives it.
pub const DEFAULT_JPEG_QUALITY: u8 = 95;

/// A multi-channel image stored as packed u8 values.
///
/// Supports 1 (gray), 3 (RGB), or 4 (RGBA) channels.
/// Data is stored row-major with channels interleaved:
/// `data[row * width * channels + col * channels + channel]`.
pub struct ImageU8 {
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) channels: u32,
    pub(super) data: Vec<u8>,
}

impl ImageU8 {
    /// Create a new image from existing data.
    ///
    /// # Panics
    ///
    /// Panics if `data.len() != width * height * channels`.
    pub fn new(width: u32, height: u32, channels: u32, data: Vec<u8>) -> Self {
        let expected = (width as usize) * (height as usize) * (channels as usize);
        assert_eq!(
            data.len(),
            expected,
            "ImageU8::new: data length {} does not match {}x{}x{} = {}",
            data.len(),
            width,
            height,
            channels,
            expected,
        );
        Self {
            width,
            height,
            channels,
            data,
        }
    }

    /// Decode the image file at `path` to 3-channel RGB.
    ///
    /// The EXIF orientation is ignored, as the feature extractors ignore it, so
    /// the pixels line up with the keypoints and with the camera's width and
    /// height. A grey image has its value repeated in the three channels, an
    /// alpha channel is dropped, and a 16-bit image is scaled to 8 bits. The
    /// file's contents, not its extension, choose the decoder.
    pub fn read_rgb(path: &std::path::Path) -> Result<Self, ::image::ImageError> {
        let rgb = Self::decode(path)?.to_rgb8();
        Ok(Self::new(rgb.width(), rgb.height(), 3, rgb.into_raw()))
    }

    /// Decode the image file at `path` to 4-channel RGBA.
    ///
    /// The colour channels equal [`read_rgb`](Self::read_rgb)'s, and the EXIF
    /// orientation is ignored in the same way. The alpha is the file's own
    /// where it has one and 255 (opaque) where it has none, so `alpha > 0`
    /// marks pixels with data, as in a reconstruction's stored patch bitmaps.
    pub fn read_rgba(path: &std::path::Path) -> Result<Self, ::image::ImageError> {
        let rgba = Self::decode(path)?.to_rgba8();
        Ok(Self::new(rgba.width(), rgba.height(), 4, rgba.into_raw()))
    }

    /// Decode the file at `path` with the decoder its contents name, not its
    /// extension, so a PNG saved under a `.jpg` name still reads.
    fn decode(path: &std::path::Path) -> Result<::image::DynamicImage, ::image::ImageError> {
        ::image::ImageReader::open(path)?
            .with_guessed_format()?
            .decode()
    }

    /// Encode the image to the file at `path`, in the format the path's
    /// extension names (case does not matter).
    ///
    /// `.jpg` / `.jpeg` write a baseline JPEG with 4:2:0 chroma subsampling
    /// at `jpeg_quality`; callers pass [`DEFAULT_JPEG_QUALITY`] unless they
    /// choose otherwise. `.png` writes a lossless PNG at the `png` crate's
    /// fast compression with adaptive filtering. Any other extension whose
    /// format the `image` crate encodes 8-bit pixels to (`.tif`, `.bmp`,
    /// `.webp`, ...) is written with that crate's defaults, which leave a
    /// TIFF uncompressed. `jpeg_quality` must be 1 to 100 whatever the
    /// format, though only a JPEG uses it. The channels are written as they
    /// are held: 1 is grey, 3 is RGB and 4 is RGBA. Every format is written
    /// at 8 bits per channel.
    ///
    /// The file is encoded in memory, written to a temporary file beside
    /// `path` and renamed into place, so a failed encode or write leaves
    /// neither a partial file nor a changed old one. Where the rename is
    /// refused because another process holds `path` open (Windows reports
    /// this as permission denied), the bytes are written to `path` directly
    /// instead, as a plain `std::fs::write` would, and a failure part way
    /// through that write can leave a partial file.
    ///
    /// The errors are
    /// [`ImageError::Unsupported`](::image::ImageError::Unsupported) for an
    /// extension that names no format the crate encodes, or a channel count
    /// the format cannot hold, among them 4 channels to JPEG, whose alpha is
    /// refused rather than dropped;
    /// [`ImageError::Parameter`](::image::ImageError::Parameter) for a
    /// `jpeg_quality` outside 1 to 100, a channel count other than 1, 3 or 4,
    /// a side of 0, or a side over 65535 in a JPEG;
    /// [`ImageError::Encoding`](::image::ImageError::Encoding) when the
    /// format's encoder refuses the image, such as a WebP side over 16384;
    /// and [`ImageError::IoError`](::image::ImageError::IoError) when the
    /// file cannot be written.
    pub fn write(
        &self,
        path: &std::path::Path,
        jpeg_quality: u8,
    ) -> Result<(), ::image::ImageError> {
        use ::image::error::{
            EncodingError, ImageFormatHint, ParameterError, ParameterErrorKind, UnsupportedError,
            UnsupportedErrorKind,
        };
        use ::image::{ExtendedColorType, ImageEncoder, ImageError, ImageFormat};

        let parameter_error = |message: String| {
            ImageError::Parameter(ParameterError::from_kind(ParameterErrorKind::Generic(
                message,
            )))
        };
        let format = ImageFormat::from_path(path)?;
        if !(1..=100).contains(&jpeg_quality) {
            return Err(parameter_error(format!(
                "JPEG quality {jpeg_quality} is outside 1 to 100"
            )));
        }
        if self.width == 0 || self.height == 0 {
            return Err(parameter_error(format!(
                "a {} x {} image has no pixels to write",
                self.width, self.height
            )));
        }
        let color = match self.channels {
            1 => ExtendedColorType::L8,
            3 => ExtendedColorType::Rgb8,
            4 => ExtendedColorType::Rgba8,
            n => {
                return Err(parameter_error(format!(
                    "an image to write has 1, 3 or 4 channels, not {n}"
                )))
            }
        };
        let mut bytes = Vec::new();
        match format {
            ImageFormat::Jpeg => {
                if self.channels == 4 {
                    return Err(ImageError::Unsupported(
                        UnsupportedError::from_format_and_kind(
                            ImageFormatHint::Exact(ImageFormat::Jpeg),
                            UnsupportedErrorKind::Color(color),
                        ),
                    ));
                }
                let (Ok(width), Ok(height)) =
                    (u16::try_from(self.width), u16::try_from(self.height))
                else {
                    return Err(parameter_error(format!(
                        "a {} x {} image is too large for a JPEG, whose sides are \
                         at most 65535",
                        self.width, self.height
                    )));
                };
                let mut encoder = jpeg_encoder::Encoder::new(&mut bytes, jpeg_quality);
                // 4:2:0 with each chroma sample the average of its 2x2 block,
                // as libjpeg and so OpenCV write it, whatever the quality.
                encoder.set_sampling_factor(jpeg_encoder::SamplingFactor::R_4_2_0);
                encoder
                    .set_chroma_subsampling_method(jpeg_encoder::ChromaSubsamplingMethod::Average);
                // Optimized Huffman tables stay off, though off is the
                // default, to record the choice: with them the encoder writes
                // each component in a scan of its own. Those files are valid
                // and OpenCV decodes them, but our reader (image 0.25.10 with
                // zune-jpeg 0.5.15) returns wrong pixels for them without an
                // error; see the ignored test
                // `read_decodes_a_jpeg_with_one_scan_per_component`.
                encoder.set_optimized_huffman_tables(false);
                let color_type = if self.channels == 1 {
                    jpeg_encoder::ColorType::Luma
                } else {
                    jpeg_encoder::ColorType::Rgb
                };
                encoder
                    .encode(&self.data, width, height, color_type)
                    .map_err(|e| {
                        ImageError::Encoding(EncodingError::new(
                            ImageFormatHint::Exact(ImageFormat::Jpeg),
                            e,
                        ))
                    })?;
            }
            ImageFormat::Png => {
                ::image::codecs::png::PngEncoder::new_with_quality(
                    &mut bytes,
                    ::image::codecs::png::CompressionType::Fast,
                    ::image::codecs::png::FilterType::Adaptive,
                )
                .write_image(&self.data, self.width, self.height, color)?;
            }
            _ => {
                ::image::write_buffer_with_format(
                    &mut std::io::Cursor::new(&mut bytes),
                    &self.data,
                    self.width,
                    self.height,
                    color,
                    format,
                )?;
            }
        }
        write_via_temporary(path, &bytes)?;
        Ok(())
    }

    /// Create a zeroed image with the given dimensions and channel count.
    pub fn from_channels(width: u32, height: u32, channels: u32) -> Self {
        let len = (width as usize) * (height as usize) * (channels as usize);
        Self {
            width,
            height,
            channels,
            data: vec![0u8; len],
        }
    }

    /// Image width in pixels.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Image height in pixels.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Number of channels (1, 3, or 4).
    pub fn channels(&self) -> u32 {
        self.channels
    }

    /// Immutable access to the raw pixel data.
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Mutable access to the raw pixel data.
    pub fn data_mut(&mut self) -> &mut [u8] {
        &mut self.data
    }

    /// Consume the image and return its raw pixel data, in the layout
    /// [`data`](Self::data) describes.
    pub fn into_data(self) -> Vec<u8> {
        self.data
    }

    /// Get a single pixel value.
    ///
    /// # Panics
    ///
    /// Panics if `col >= width`, `row >= height`, or `channel >= channels`.
    pub fn get_pixel(&self, col: u32, row: u32, channel: u32) -> u8 {
        let idx = (row as usize) * (self.width as usize) * (self.channels as usize)
            + (col as usize) * (self.channels as usize)
            + (channel as usize);
        self.data[idx]
    }

    /// Downsample by 2x using a box filter (2x2 average).
    ///
    /// Each output pixel is the average of the four corresponding input pixels.
    /// Operates independently per channel.
    pub fn downsample_2x(&self) -> Self {
        let out_w = self.width / 2;
        let out_h = self.height / 2;
        let c = self.channels as usize;
        let in_stride = self.width as usize * c;
        let out_stride = out_w as usize * c;
        let mut out_data = vec![0u8; (out_w as usize) * (out_h as usize) * c];

        for oy in 0..out_h as usize {
            let iy = oy * 2;
            let row0 = &self.data[iy * in_stride..][..in_stride];
            let row1 = &self.data[(iy + 1) * in_stride..][..in_stride];
            let out_row = &mut out_data[oy * out_stride..][..out_stride];

            for ox in 0..out_w as usize {
                let ix = ox * 2;
                for ch in 0..c {
                    let v00 = row0[ix * c + ch] as u16;
                    let v10 = row0[(ix + 1) * c + ch] as u16;
                    let v01 = row1[ix * c + ch] as u16;
                    let v11 = row1[(ix + 1) * c + ch] as u16;
                    out_row[ox * c + ch] = ((v00 + v10 + v01 + v11 + 2) / 4) as u8;
                }
            }
        }

        Self {
            width: out_w,
            height: out_h,
            channels: self.channels,
            data: out_data,
        }
    }
}

/// Write `bytes` to `path` through a temporary file beside it that is renamed
/// into place, so a failed write leaves neither a partial file at `path` nor a
/// changed old one. The temporary file is removed whether or not the write
/// succeeds.
///
/// On Windows a file another process holds open cannot be replaced by a
/// rename, which fails with permission denied, though it can still be
/// written. In that case the bytes are written to `path` directly, as
/// `std::fs::write` alone would.
fn write_via_temporary(path: &std::path::Path, bytes: &[u8]) -> std::io::Result<()> {
    // The process id and a per-process count keep two writers apart, whether
    // they are threads of one process or separate processes.
    static COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let count = COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let name = path.file_name().unwrap_or_default().to_string_lossy();
    let temporary = path.with_file_name(format!(".{name}.{}-{count}.tmp", std::process::id()));
    let written = std::fs::write(&temporary, bytes)
        .and_then(|()| std::fs::rename(&temporary, path))
        .or_else(|e| match e.kind() {
            std::io::ErrorKind::PermissionDenied if !path.is_dir() => std::fs::write(path, bytes),
            _ => Err(e),
        });
    // Gone already after a rename; still there after a failure or the direct
    // write.
    let _ = std::fs::remove_file(&temporary);
    written
}

/// Whether the image file at `path` stores an alpha channel, read from its
/// header alone, never the pixel data.
///
/// The contents, not the extension, choose the decoder, as in
/// [`ImageU8::read_rgb`]. A caller that writes an image back out reads it with
/// [`ImageU8::read_rgba`] when this is true, so the alpha survives, and with
/// [`ImageU8::read_rgb`] otherwise.
pub fn image_has_alpha(path: &std::path::Path) -> Result<bool, ::image::ImageError> {
    use ::image::ImageDecoder;
    let decoder = ::image::ImageReader::open(path)?
        .with_guessed_format()?
        .into_decoder()?;
    Ok(decoder.color_type().has_alpha())
}

/// Gaussian pyramid of [`ImageU8`] images for anisotropic resampling.
pub struct ImageU8Pyramid {
    levels: Vec<ImageU8>,
}

impl ImageU8Pyramid {
    /// The full pyramid depth for an image of `width x height`: levels down to
    /// about one pixel on the short side, and never fewer than two.
    ///
    /// What a kernel that picks its own level per sample is given, so a
    /// footprint of any size finds a level to read. Every caller that builds a
    /// pyramid for the patch kernels from a list of photographs uses this, so
    /// two of them handed the same image build the same levels.
    pub fn full_levels(width: u32, height: u32) -> usize {
        let min_dim = width.min(height).max(1);
        ((min_dim as f32).log2().floor() as usize).max(1) + 1
    }

    /// Build a Gaussian pyramid from a full-resolution image.
    ///
    /// Level 0 is a copy of the input. Each subsequent level is 2x downsampled
    /// using a box filter. The pyramid has `num_levels` entries total.
    pub fn build(image: &ImageU8, num_levels: usize) -> Self {
        Self::from_image(
            ImageU8::new(
                image.width,
                image.height,
                image.channels,
                image.data.clone(),
            ),
            num_levels,
        )
    }

    /// Build a Gaussian pyramid from an image it takes over, so that level 0 is
    /// that image itself.
    ///
    /// The same pyramid [`Self::build`] produces, without the full-resolution
    /// copy: a caller that already owns the pixels and has no other use for
    /// them hands them over, and the megabytes of a photograph are moved rather
    /// than duplicated. `build` is this call after its own copy, so the two
    /// share one downsample loop and cannot disagree about a level.
    pub fn from_image(image: ImageU8, num_levels: usize) -> Self {
        assert!(num_levels >= 1, "Pyramid must have at least 1 level");
        let mut levels = Vec::with_capacity(num_levels);
        levels.push(image);

        for _ in 1..num_levels {
            let prev = levels.last().unwrap();
            // Stop if either dimension would become 0.
            if prev.width < 2 || prev.height < 2 {
                break;
            }
            levels.push(prev.downsample_2x());
        }

        Self { levels }
    }

    /// Access a specific pyramid level. Level 0 is full resolution.
    pub fn level(&self, i: usize) -> &ImageU8 {
        &self.levels[i]
    }

    /// Number of levels actually built (may be less than requested if the
    /// image became too small).
    pub fn num_levels(&self) -> usize {
        self.levels.len()
    }

    /// The pixel bytes held across every level: what the pyramid costs in
    /// memory, not counting the small per-level headers.
    pub fn byte_len(&self) -> usize {
        self.levels.iter().map(|level| level.data.len()).sum()
    }
}

/// Float image plus per-channel image gradient `(∂I/∂x, ∂I/∂y)` in source-pixel
/// coords. Produced by
/// [`remap_aniso_with_grad`](super::remap::remap_aniso_with_grad) /
/// [`remap_bilinear_with_grad`](super::remap::remap_bilinear_with_grad) (or the
/// `_into` variants) and consumed by the photometric subpixel refiner;
/// the values are *not* rounded to `u8` (the refiner needs the unquantized
/// gradient and an in-range float value for the GN normal equations).
///
/// Storage layout per output pixel `(col, row)` and channel `ch`:
/// `idx = (row * width + col) * channels + ch` for each of the three buffers.
///
/// Designed for scratch reuse: callers that render many tiles back-to-back
/// (e.g. the per-GN-step gradient build in
/// [`keypoint_subpixel`](crate::patch::keypoint_subpixel)) hold one of these as
/// a scratch field, [`resize`](Self::resize) it for the new tile's shape (cheap
/// when shape is unchanged), and pass it as the `out` of an `_into` variant.
pub struct ImageF32WithGrad {
    width: u32,
    height: u32,
    channels: u32,
    pub(super) value: Vec<f32>,
    pub(super) grad_x: Vec<f32>,
    pub(super) grad_y: Vec<f32>,
}

impl ImageF32WithGrad {
    /// An empty image, sized 0×0×0. Reuse via [`resize`](Self::resize).
    pub fn empty() -> Self {
        Self {
            width: 0,
            height: 0,
            channels: 0,
            value: Vec::new(),
            grad_x: Vec::new(),
            grad_y: Vec::new(),
        }
    }

    /// Resize the buffers to fit a `width × height × channels` image, zeroing
    /// every pixel. Reuses the existing allocation when the new total fits.
    pub fn resize(&mut self, width: u32, height: u32, channels: u32) {
        let total = width as usize * height as usize * channels as usize;
        self.width = width;
        self.height = height;
        self.channels = channels;
        self.value.clear();
        self.value.resize(total, 0.0);
        self.grad_x.clear();
        self.grad_x.resize(total, 0.0);
        self.grad_y.clear();
        self.grad_y.resize(total, 0.0);
    }

    pub fn width(&self) -> u32 {
        self.width
    }
    pub fn height(&self) -> u32 {
        self.height
    }
    pub fn channels(&self) -> u32 {
        self.channels
    }

    /// `(value, ∂I/∂x, ∂I/∂y)` at output pixel `(col, row)`, channel `ch`. Mirrors
    /// [`ImageU8::get_pixel`]'s pattern; in inner-loop callers prefer
    /// [`value`](Self::value) / [`grad_x`](Self::grad_x) / [`grad_y`](Self::grad_y)
    /// and walk by raw index to avoid the per-access bounds check.
    pub fn get_pixel_with_grad(&self, col: u32, row: u32, ch: u32) -> (f32, f32, f32) {
        let idx = (row as usize * self.width as usize + col as usize) * self.channels as usize
            + ch as usize;
        (self.value[idx], self.grad_x[idx], self.grad_y[idx])
    }

    /// Raw value slice (`channels`-interleaved, row-major). Use with the public
    /// `width()`/`height()`/`channels()` to compute the per-pixel index. Public
    /// so hot inner-loop callers (the photometric refiner) can index without
    /// per-access bounds checks beyond the slice's own range check.
    pub fn value(&self) -> &[f32] {
        &self.value
    }
    pub fn grad_x(&self) -> &[f32] {
        &self.grad_x
    }
    pub fn grad_y(&self) -> &[f32] {
        &self.grad_y
    }
}

#[cfg(test)]
mod tests;
