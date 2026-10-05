// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The photographs as the patch reads sample them: grey, as 32-bit floats,
//! blurred by a Gaussian of one pixel, built the first time an image is read
//! and kept.
//!
//! Every step reproduces what the harness's `cv2` calls do, so the reads score
//! the same: the grey conversion is OpenCV's fixed-point one, the blur is a
//! nine-tap separable Gaussian with the image reflected at its edges (without
//! repeating the edge pixel), and a sample is bilinear, with no value where any
//! of its four pixels is outside the photograph. What is left differs from
//! OpenCV's arithmetic in the last bits of a 32-bit float, from the order the
//! blur and the interpolation add their terms in.

use std::sync::OnceLock;

use crate::camera::image::ImageU8;
use crate::patch::normal_refine::ProjectedImage;

/// The standard deviation of the blur, in px.
pub const GREY_BLUR_SIGMA: f64 = 1.0;

/// One photograph in grey, on the 0 to 255 scale of its 8-bit pixels, as
/// 32-bit floats in row-major order.
#[derive(Debug, Clone, PartialEq)]
pub struct GreyImage {
    width: u32,
    height: u32,
    data: Vec<f32>,
}

impl GreyImage {
    /// An image from its row-major values.
    ///
    /// # Panics
    ///
    /// Panics if `data` does not hold `width * height` values.
    pub fn new(width: u32, height: u32, data: Vec<f32>) -> Self {
        assert_eq!(data.len(), width as usize * height as usize);
        Self {
            width,
            height,
            data,
        }
    }

    /// The width, in px.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// The height, in px.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// The values, row-major.
    pub fn data(&self) -> &[f32] {
        &self.data
    }
}

/// Every view's photograph in grey and blurred, each built the first time it
/// is read.
///
/// Built once per set of views and shared by every read of them: the grey
/// images hold nothing of the reconstruction, so a query needs no version of
/// them. The cache is filled through a shared reference, so one set can be
/// read from several threads.
#[derive(Debug, Default)]
pub struct GreyImages {
    images: Vec<OnceLock<GreyImage>>,
}

impl GreyImages {
    /// An empty cache for `count` images.
    pub fn new(count: usize) -> Self {
        Self {
            images: (0..count).map(|_| OnceLock::new()).collect(),
        }
    }

    /// How many images the cache is for.
    pub fn len(&self) -> usize {
        self.images.len()
    }

    /// Whether the cache is for no images.
    pub fn is_empty(&self) -> bool {
        self.images.is_empty()
    }

    /// Image `image` in grey and blurred, built from `views[image]`'s
    /// full-resolution photograph the first time it is asked for.
    ///
    /// # Panics
    ///
    /// Panics if `image` is not an index into both the cache and `views`.
    pub fn get(&self, views: &[ProjectedImage<'_>], image: usize) -> &GreyImage {
        self.images[image].get_or_init(|| blurred_grey(views[image].pyramid.level(0)))
    }
}

/// `image` in grey as OpenCV's `COLOR_RGB2GRAY` makes it, then blurred by
/// [`GREY_BLUR_SIGMA`] as `cv2.GaussianBlur` does a 32-bit float image.
pub fn blurred_grey(image: &ImageU8) -> GreyImage {
    let (w, h) = (image.width() as usize, image.height() as usize);
    let c = image.channels() as usize;
    let data = image.data();
    let grey: Vec<f32> = (0..w * h)
        .map(|i| {
            let p = &data[i * c..i * c + c];
            if c >= 3 {
                // OpenCV's 15-bit fixed-point weights for R, G and B.
                let v = u32::from(p[0]) * 9798 + u32::from(p[1]) * 19235 + u32::from(p[2]) * 3735;
                ((v + (1 << 14)) >> 15) as f32
            } else {
                f32::from(p[0])
            }
        })
        .collect();
    GreyImage::new(
        w as u32,
        h as u32,
        gaussian_blur(&grey, w, h, GREY_BLUR_SIGMA),
    )
}

/// OpenCV's Gaussian kernel for a 32-bit float image of blur `sigma`: `ksize =
/// round(8 sigma + 1) | 1` taps, each rounded to `f32` and then scaled so the
/// taps sum to one.
fn gaussian_kernel(sigma: f64) -> Vec<f32> {
    let n = ((sigma * 8.0 + 1.0).round() as usize) | 1;
    let scale = -0.5 / (sigma * sigma);
    let taps: Vec<f32> = (0..n)
        .map(|i| {
            let x = i as f64 - (n - 1) as f64 * 0.5;
            (scale * x * x).exp() as f32
        })
        .collect();
    let sum: f64 = taps.iter().map(|&t| f64::from(t)).sum();
    taps.iter().map(|&t| (f64::from(t) / sum) as f32).collect()
}

/// Where tap `i` of a line of `n` samples reads, reflected at the ends without
/// repeating the end sample (OpenCV's `BORDER_REFLECT_101`).
fn reflect_101(i: isize, n: usize) -> usize {
    let n = n as isize;
    if n == 1 {
        return 0;
    }
    let mut i = i;
    while i < 0 || i >= n {
        i = if i < 0 { -i } else { 2 * n - 2 - i };
    }
    i as usize
}

/// A separable Gaussian blur of the `w x h` image `src`, rows then columns.
fn gaussian_blur(src: &[f32], w: usize, h: usize, sigma: f64) -> Vec<f32> {
    let kernel = gaussian_kernel(sigma);
    let r = (kernel.len() / 2) as isize;
    let mut rows = vec![0.0f32; w * h];
    for y in 0..h {
        let line = &src[y * w..y * w + w];
        for x in 0..w {
            let mut acc = 0.0f32;
            for (k, &kw) in kernel.iter().enumerate() {
                acc += kw * line[reflect_101(x as isize + k as isize - r, w)];
            }
            rows[y * w + x] = acc;
        }
    }
    let mut out = vec![0.0f32; w * h];
    for y in 0..h {
        for (k, &kw) in kernel.iter().enumerate() {
            let from = reflect_101(y as isize + k as isize - r, h) * w;
            for x in 0..w {
                out[y * w + x] += kw * rows[from + x];
            }
        }
    }
    out
}

/// A bilinear sample of `image` at `(x, y)`, in the convention that the
/// centre of pixel `(i, j)` is `(i + 0.5, j + 0.5)`; `None` where any of the
/// four pixels it reads is outside the image.
///
/// The coordinate is rounded to `f32` first, as `cv2.remap` reads its map, and
/// the four pixels are weighted by the offset from the first of them.
pub fn sample_grey(image: &GreyImage, x: f64, y: f64) -> Option<f32> {
    let gx = x as f32 - 0.5;
    let gy = y as f32 - 0.5;
    if !gx.is_finite() || !gy.is_finite() {
        return None;
    }
    let (x0, y0) = (gx.floor(), gy.floor());
    let (w, h) = (image.width() as f32, image.height() as f32);
    if x0 < 0.0 || y0 < 0.0 || x0 + 1.0 >= w || y0 + 1.0 >= h {
        return None;
    }
    let (fx, fy) = (gx - x0, gy - y0);
    let stride = image.width() as usize;
    let at = y0 as usize * stride + x0 as usize;
    let data = image.data();
    let (p00, p01) = (data[at], data[at + 1]);
    let (p10, p11) = (data[at + stride], data[at + stride + 1]);
    let top = p00 + fx * (p01 - p00);
    let bottom = p10 + fx * (p11 - p10);
    Some(top + fy * (bottom - top))
}
