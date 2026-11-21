// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The photographs as the patch reads sample them: grey, as 32-bit floats,
//! blurred by a Gaussian of one pixel. Each read makes grey only the
//! rectangle of level 0 its samples need ([`read_grey`]), read through the
//! caller's region pool.
//!
//! The grey is the matchers' 8-bit BT.601 grey
//! ([`matcher_gray_from_rgb`]) on the 0 to 255 scale, the conversion the
//! optical flow reads too. The blur and the sample reproduce what the
//! harness's `cv2` calls do: the blur is a nine-tap separable Gaussian with the
//! image reflected at its edges (without repeating the edge pixel), and a
//! sample is bilinear, with no value where any of its four pixels is outside
//! the photograph. What is left differs from OpenCV's arithmetic in the last
//! bits of a 32-bit float, from the order the blur and the interpolation add
//! their terms in.

use crate::camera::image::ImageU8;
use crate::camera::sampler::{check_source, read_source_rect, LevelRect, RegionPool};
use crate::camera::{PhotographReadError, TiledPhotograph};
use crate::features::optical_flow::matcher_gray_from_rgb;

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

/// Why a grey image could not be made.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GreyImageError {
    /// The values do not fill the image: `data` does not hold
    /// `width * height` of them.
    DataSize {
        /// The image's width, in px.
        width: u32,
        /// The image's height, in px.
        height: u32,
        /// How many values `data` holds.
        got: usize,
    },
}

impl std::fmt::Display for GreyImageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DataSize { width, height, got } => write!(
                f,
                "a {width}x{height} grey image needs {} values, but {got} were given",
                *width as usize * *height as usize
            ),
        }
    }
}

impl std::error::Error for GreyImageError {}

impl GreyImage {
    /// An image from its row-major values.
    ///
    /// # Errors
    ///
    /// [`GreyImageError::DataSize`] when `data` does not hold
    /// `width * height` values.
    pub fn new(width: u32, height: u32, data: Vec<f32>) -> Result<Self, GreyImageError> {
        if data.len() != width as usize * height as usize {
            return Err(GreyImageError::DataSize {
                width,
                height,
                got: data.len(),
            });
        }
        Ok(Self {
            width,
            height,
            data,
        })
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

/// `image` in grey on the 0 to 255 scale, then blurred by [`GREY_BLUR_SIGMA`]
/// as `cv2.GaussianBlur` does a 32-bit float image.
///
/// A colour image is made grey by [`matcher_gray_from_rgb`], BT.601 luma
/// rounded to 8 bits, each level `k` held as `k`; an alpha channel is dropped.
/// A grey image (one or two channels) keeps its first channel. The patch
/// reads make grey only the rectangle they sample, with the values this gives.
pub fn blurred_grey(image: &ImageU8) -> GreyImage {
    let (width, height) = (image.width(), image.height());
    let whole = LevelRect {
        x0: 0,
        y0: 0,
        x1: width,
        y1: height,
    };
    // The blur keeps the `width x height` size, so the values fill the image.
    GreyImage {
        width,
        height,
        data: blur_box(&grey_values(image), whole, whole, [width, height]),
    }
}

/// `image`'s pixels in grey on the 0 to 255 scale, row-major, as
/// [`blurred_grey`] makes them before the blur.
fn grey_values(image: &ImageU8) -> Vec<f32> {
    let (w, h) = (image.width(), image.height());
    let c = image.channels() as usize;
    let data = image.data();
    if c >= 3 {
        let rgb: std::borrow::Cow<'_, [u8]> = if c == 3 {
            data.into()
        } else {
            data.chunks_exact(c)
                .flat_map(|p| [p[0], p[1], p[2]])
                .collect::<Vec<u8>>()
                .into()
        };
        // Each sample is the f32 nearest k / 255, so times 255 it rounds to k.
        matcher_gray_from_rgb(w, h, &rgb)
            .data()
            .iter()
            .map(|v| (v * 255.0).round())
            .collect()
    } else {
        data.chunks_exact(c).map(|p| f32::from(p[0])).collect()
    }
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

/// How far the blur of [`GREY_BLUR_SIGMA`] reaches from its pixel, in px: half
/// its taps, 4.
fn blur_reach() -> u32 {
    (gaussian_kernel(GREY_BLUR_SIGMA).len() / 2) as u32
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

/// The separable Gaussian blur, rows then columns, of the pixels `inner` of an
/// image of size `full`, from `src`, the grey of its rectangle `rect`,
/// row-major; the result is `inner`'s, row-major.
///
/// Each tap reads the image reflected at its own edges, never at an edge of
/// `rect`, so a value is the one a blur of the whole image gives, bit for bit,
/// adding the same terms in the same order. `rect` must hold every pixel a tap
/// of `inner`'s pixels reads: `inner` grown by [`blur_reach`] and clipped to
/// the image does, since a tap reflected at an edge lands no further from its
/// pixel than the tap itself.
fn blur_box(src: &[f32], rect: LevelRect, inner: LevelRect, full: [u32; 2]) -> Vec<f32> {
    let kernel = gaussian_kernel(GREY_BLUR_SIGMA);
    let r = (kernel.len() / 2) as isize;
    let (rw, rh) = (rect.width() as usize, rect.height() as usize);
    let (iw, ih) = (inner.width() as usize, inner.height() as usize);
    let (fw, fh) = (full[0] as usize, full[1] as usize);
    let (rx0, ry0) = (rect.x0 as usize, rect.y0 as usize);
    // The rows pass, over every row of `rect` and the columns of `inner`.
    let mut rows = vec![0.0f32; iw * rh];
    for ry in 0..rh {
        let line = &src[ry * rw..ry * rw + rw];
        for ix in 0..iw {
            let x = (inner.x0 as usize + ix) as isize;
            let mut acc = 0.0f32;
            for (k, &kw) in kernel.iter().enumerate() {
                acc += kw * line[reflect_101(x + k as isize - r, fw) - rx0];
            }
            rows[ry * iw + ix] = acc;
        }
    }
    let mut out = vec![0.0f32; iw * ih];
    for iy in 0..ih {
        let y = (inner.y0 as usize + iy) as isize;
        let line = &mut out[iy * iw..iy * iw + iw];
        for (k, &kw) in kernel.iter().enumerate() {
            let from = (reflect_101(y + k as isize - r, fh) - ry0) * iw;
            for (o, &v) in line.iter_mut().zip(&rows[from..from + iw]) {
                *o += kw * v;
            }
        }
    }
    out
}

/// Where a bilinear sample at `(x, y)` reads in an image of size `full`: the
/// first of its four pixels and its offsets from that pixel, or `None` where
/// any of the four is outside the image.
///
/// The coordinate is rounded to `f32` first, as `cv2.remap` reads its map.
fn bilinear_cell(full: [u32; 2], x: f64, y: f64) -> Option<([u32; 2], [f32; 2])> {
    let gx = x as f32 - 0.5;
    let gy = y as f32 - 0.5;
    if !gx.is_finite() || !gy.is_finite() {
        return None;
    }
    let (x0, y0) = (gx.floor(), gy.floor());
    let (w, h) = (full[0] as f32, full[1] as f32);
    if x0 < 0.0 || y0 < 0.0 || x0 + 1.0 >= w || y0 + 1.0 >= h {
        return None;
    }
    Some(([x0 as u32, y0 as u32], [gx - x0, gy - y0]))
}

/// The bilinear sample of the row-major `data` of `stride` columns whose
/// first pixel is `at`, at offsets `f` from it.
fn bilinear(data: &[f32], stride: usize, at: usize, f: [f32; 2]) -> f32 {
    let (p00, p01) = (data[at], data[at + 1]);
    let (p10, p11) = (data[at + stride], data[at + stride + 1]);
    let top = p00 + f[0] * (p01 - p00);
    let bottom = p10 + f[0] * (p11 - p10);
    top + f[1] * (bottom - top)
}

/// A bilinear sample of `image` at `(x, y)`, in the convention that the
/// centre of pixel `(i, j)` is `(i + 0.5, j + 0.5)`; `None` where any of the
/// four pixels it reads is outside the image.
///
/// The coordinate is rounded to `f32` first, as `cv2.remap` reads its map, and
/// the four pixels are weighted by the offset from the first of them.
pub fn sample_grey(image: &GreyImage, x: f64, y: f64) -> Option<f32> {
    let ([x0, y0], f) = bilinear_cell([image.width, image.height], x, y)?;
    let stride = image.width as usize;
    Some(bilinear(
        &image.data,
        stride,
        y0 as usize * stride + x0 as usize,
        f,
    ))
}

/// The samples [`sample_grey`] takes at `positions` of the [`blurred_grey`]
/// of `source`'s level 0, read through `pool`, or `None`, with nothing read,
/// when any of them has no value.
///
/// Only the bounding box of the pixels the samples take is made grey and
/// blurred, from the rectangle of level 0 that box needs, grown by the blur's
/// reach and clipped to the photograph, read with [`read_source_rect`]; its
/// reservation is given back before this returns.
pub(super) fn read_grey(
    source: &TiledPhotograph,
    pool: &RegionPool,
    positions: &[[f64; 2]],
) -> Result<Option<Vec<f32>>, PhotographReadError> {
    check_source(source)?;
    let (width, height) = source.layout().level_size(0);
    let full = [width, height];
    let Some(cells) = positions
        .iter()
        .map(|&[x, y]| bilinear_cell(full, x, y))
        .collect::<Option<Vec<_>>>()
    else {
        return Ok(None);
    };
    let Some(inner) = cells
        .iter()
        .map(|&([x0, y0], _)| LevelRect {
            x0,
            y0,
            x1: x0 + 2,
            y1: y0 + 2,
        })
        .reduce(|a, b| LevelRect {
            x0: a.x0.min(b.x0),
            y0: a.y0.min(b.y0),
            x1: a.x1.max(b.x1),
            y1: a.y1.max(b.y1),
        })
    else {
        return Ok(Some(Vec::new()));
    };
    let reach = blur_reach();
    let rect = LevelRect {
        x0: inner.x0.saturating_sub(reach),
        y0: inner.y0.saturating_sub(reach),
        x1: inner.x1.saturating_add(reach).min(width),
        y1: inner.y1.saturating_add(reach).min(height),
    };
    let grey = grey_values(read_source_rect(source, pool, 0, rect)?.image());
    let blurred = blur_box(&grey, rect, inner, full);
    let stride = inner.width() as usize;
    Ok(Some(
        cells
            .iter()
            .map(|&([x0, y0], f)| {
                let at = (y0 - inner.y0) as usize * stride + (x0 - inner.x0) as usize;
                bilinear(&blurred, stride, at, f)
            })
            .collect(),
    ))
}
