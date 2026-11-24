// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Synthetic photographs for the mipmap tests and the photograph cache's
//! tests of reading a mipmap: a few hundred pixels each, so no test reads or
//! makes a large photograph.

use jpeg_encoder::{ColorType, Encoder, SamplingFactor};

use crate::camera::image::ImageU8;

/// A `width` x `height` RGB image with smooth gradients and some texture, the
/// same for the same size.
pub(crate) fn synthetic_rgb(width: u32, height: u32) -> ImageU8 {
    let mut data = Vec::with_capacity((width * height * 3) as usize);
    for y in 0..height {
        for x in 0..width {
            let texture = ((x * 7 + y * 13) % 23) as u8;
            data.push((x * 255 / width.max(1)) as u8 ^ texture);
            data.push((y * 255 / height.max(1)) as u8);
            data.push(((x + y) % 256) as u8 / 2 + texture);
        }
    }
    ImageU8::new(width, height, 3, data)
}

/// How a test JPEG is written.
#[derive(Clone, Copy)]
pub(crate) struct JpegOptions {
    pub(crate) sampling: SamplingFactor,
    /// 0 writes no `DRI`.
    pub(crate) restart_interval: u16,
    pub(crate) progressive: bool,
    pub(crate) grey: bool,
}

impl Default for JpegOptions {
    fn default() -> Self {
        Self {
            sampling: SamplingFactor::F_2_2,
            restart_interval: 0,
            progressive: false,
            grey: false,
        }
    }
}

/// `image` encoded as a JPEG by `jpeg-encoder` at quality 90.
pub(crate) fn encode_jpeg(image: &ImageU8, options: JpegOptions) -> Vec<u8> {
    let mut bytes = Vec::new();
    let mut encoder = Encoder::new(&mut bytes, 90);
    encoder.set_sampling_factor(options.sampling);
    encoder.set_progressive(options.progressive);
    if options.restart_interval > 0 {
        encoder.set_restart_interval(options.restart_interval);
    }
    let (w, h) = (image.width() as u16, image.height() as u16);
    if options.grey {
        let grey: Vec<u8> = image
            .data()
            .as_chunks::<3>()
            .0
            .iter()
            .map(|p| p[1])
            .collect();
        encoder.encode(&grey, w, h, ColorType::Luma).unwrap();
    } else {
        encoder.encode(image.data(), w, h, ColorType::Rgb).unwrap();
    }
    bytes
}

/// `image` encoded as a PNG.
pub(crate) fn encode_png(image: &ImageU8) -> Vec<u8> {
    use ::image::ImageEncoder;
    let mut bytes = Vec::new();
    ::image::codecs::png::PngEncoder::new(&mut bytes)
        .write_image(
            image.data(),
            image.width(),
            image.height(),
            ::image::ExtendedColorType::Rgb8,
        )
        .unwrap();
    bytes
}
