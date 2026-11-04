// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The two per-image readings of an image's sharpness: the variance of the
//! Laplacian and the gradient structure tensor of its reading image.
//!
//! The **reading image** is the image made grey by an [`ImageToGray`]
//! conversion ([`ImageToGray::gray_row`]) and reduced by an area average
//! ([`area_resize`]) to a long side of [`READING_IMAGE_LONG_SIDE`] pixels (an
//! image no larger is used at its own size). On its interior pixels, those
//! whose 3 x 3 neighbourhood lies inside it:
//!
//! - the **sharpness** is the population variance of the Laplacian
//!   `[[0, 1, 0], [1, −4, 1], [0, 1, 0]]`;
//! - the **gradient structure tensor** `[Exx, Eyy, Exy]` is the means of
//!   `Gx²`, `Gy²` and `Gx·Gy`, with `Gx` and `Gy` the 3 x 3 Sobel kernels
//!   divided by 8, `x` to the right and `y` down.
//!
//! Both are defined in `specs/formats/sift-file-format.md` § "The reading
//! image", "Sharpness" and "Gradient structure tensor", which also says what
//! a reader derives from the tensor (coherence, dominant direction, axis
//! ratio). [`image_readings_from_rgb`] is the one implementation of both, and
//! [`ImageReadings::axis_ratio`] and [`ImageReadings::coherence`] the one
//! derivation of those two values.
//!
//! **Precision.** Everything is single precision, `f32`. The readings are
//! compared within a capture, as ranks, ratios to the median and z-scores,
//! which need about three significant digits. The grey samples and the
//! resize are `f32` for the reasons [`ImageToGray::gray_row`] and
//! [`area_resize`] give. The Laplacian and the Sobel derivatives combine
//! neighbouring samples in `f32`, each rounding about `2^-24` of a grey unit,
//! against differences between neighbours of `10^-3` grey units and more
//! where an image has detail. The means are sums of about 130,000 terms, the
//! interior of a 480 x 270 reading image. Each row's terms are summed in `f32`
//! from left to right and the row sums in `f32` from top to bottom, which
//! bounds the rounding error of a mean by about `(W' + H') · 2^-24`, under
//! `5 · 10^-5` of the sum of the terms' magnitudes, where one running sum over
//! every term would be bounded only by `n · 2^-24`, about `8 · 10^-3`. On
//! photographs the readings come within `2 · 10^-6` of a double-precision
//! evaluation. Summing by rows costs no more than one running sum, splits the
//! work by rows over the pool and keeps the order fixed, and an `f64`
//! accumulator would be no faster and add nothing the readings' use needs.

use rayon::prelude::*;

use super::gray::{GrayRowScratch, ImageToGray};
use super::GrayImage;
use crate::camera::image::area_resize::area_resize;

/// The long side of the reading image, in pixels.
pub const READING_IMAGE_LONG_SIDE: usize = 480;

/// An image's sharpness and gradient structure tensor, from
/// [`image_readings_from_rgb`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ImageReadings {
    /// The population variance of the Laplacian over the reading image's
    /// interior, in grey units squared. Finite and not negative.
    pub sharpness: f32,
    /// `[Exx, Eyy, Exy]`: the means of `Gx²`, `Gy²` and `Gx·Gy` over the
    /// reading image's interior, in grey units squared per pixel squared, not
    /// normalised. Finite; `Exx` and `Eyy` not negative.
    pub gradient_structure_tensor_xx_yy_xy: [f32; 3],
}

impl ImageReadings {
    /// The readings of an image whose reading image has no interior pixel:
    /// all zero.
    pub const ZERO: Self = Self {
        sharpness: 0.0,
        gradient_structure_tensor_xx_yy_xy: [0.0; 3],
    };

    /// The readings as stored, in a `.sift` file's metadata or a `.sfmr`'s
    /// image columns.
    ///
    /// # Errors
    ///
    /// [`ImageReadingsError::Invalid`] when a value is not finite, or the
    /// sharpness, `Exx` or `Eyy` is negative, which no image reads.
    ///
    /// # Example
    ///
    /// ```
    /// use sfmtool_core::features::sift::ImageReadings;
    ///
    /// let readings = ImageReadings::new(0.0125, [0.004, 0.001, 0.001])?;
    /// // |log(0.004 / 0.001)| = log 4.
    /// assert!((readings.axis_ratio().unwrap() - 4f32.ln()).abs() < 1e-6);
    /// assert!(ImageReadings::new(0.0125, [-0.004, 0.001, 0.001]).is_err());
    /// # Ok::<(), sfmtool_core::features::sift::ImageReadingsError>(())
    /// ```
    pub fn new(
        sharpness: f32,
        gradient_structure_tensor_xx_yy_xy: [f32; 3],
    ) -> Result<Self, ImageReadingsError> {
        let readings = Self {
            sharpness,
            gradient_structure_tensor_xx_yy_xy,
        };
        let [xx, yy, xy] = gradient_structure_tensor_xx_yy_xy;
        let finite = [sharpness, xx, yy, xy].iter().all(|v| v.is_finite());
        if finite && sharpness >= 0.0 && xx >= 0.0 && yy >= 0.0 {
            Ok(readings)
        } else {
            Err(ImageReadingsError::Invalid { readings })
        }
    }

    /// The axis ratio `|log(Exx / Eyy)|`, with the natural logarithm: how far
    /// the gradient energy along `x` differs from that along `y`.
    /// `f32::INFINITY` when exactly one of `Exx` and `Eyy` is 0, and `None`
    /// when both are, a flat image.
    ///
    /// It is taken as `|ln Exx − ln Eyy|`, so that a ratio beyond the range of
    /// an `f32` still gives a finite value. Each logarithm is within about
    /// `|ln E| · 2^-24` of exact, under `10^-5` for any `f32`, against the
    /// three decimals the value is compared to.
    pub fn axis_ratio(&self) -> Option<f32> {
        let [xx, yy, _] = self.gradient_structure_tensor_xx_yy_xy;
        if xx > 0.0 && yy > 0.0 {
            Some((xx.ln() - yy.ln()).abs())
        } else if xx == yy {
            None
        } else {
            Some(f32::INFINITY)
        }
    }

    /// The coherence `sqrt((Exx − Eyy)² + 4·Exy²) / (Exx + Eyy)`, from 0
    /// (gradients equally strong in every direction) to 1 (all gradients
    /// along one direction), and 0 for a flat image, where `Exx + Eyy = 0`.
    ///
    /// The square root is taken by `hypot`, which squares nothing, and a
    /// tensor whose `Exx + Eyy` is beyond the range of an `f32` is halved
    /// first, which leaves the ratio unchanged, so a finite tensor gives a
    /// finite coherence.
    pub fn coherence(&self) -> f32 {
        let [mut xx, mut yy, mut xy] = self.gradient_structure_tensor_xx_yy_xy;
        if !(xx + yy).is_finite() {
            (xx, yy, xy) = (0.5 * xx, 0.5 * yy, 0.5 * xy);
        }
        let energy = xx + yy;
        if energy > 0.0 {
            (xx - yy).hypot(2.0 * xy) / energy
        } else {
            0.0
        }
    }
}

/// Why [`image_readings_from_rgb`] could not take an image's readings.
#[derive(Debug, Clone, PartialEq)]
pub enum ImageReadingsError {
    /// `rgb` does not hold `3 * width * height` bytes.
    PixelCount {
        /// The width given.
        width: u32,
        /// The height given.
        height: u32,
        /// The number of bytes given.
        len: usize,
    },
    /// The conversion made a reading that is not finite, as an unclamped
    /// formula can (`R**-1` at `R = 0`, say), or one too large for an `f32`.
    NotFinite {
        /// The conversion's `image_to_gray` object, as JSON text.
        image_to_gray: String,
        /// The readings as computed.
        readings: ImageReadings,
    },
    /// Stored readings given to [`ImageReadings::new`] that no image reads: a
    /// value that is not finite, or a negative sharpness, `Exx` or `Eyy`.
    Invalid {
        /// The readings as given.
        readings: ImageReadings,
    },
}

impl std::fmt::Display for ImageReadingsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PixelCount { width, height, len } => write!(
                f,
                "a {width} x {height} RGB image needs 3 * {width} * {height} bytes, got {len}"
            ),
            Self::NotFinite {
                image_to_gray,
                readings,
            } => write!(
                f,
                "the image's readings are not finite under image_to_gray {image_to_gray} \
                 (sharpness {}, gradient structure tensor {:?})",
                readings.sharpness, readings.gradient_structure_tensor_xx_yy_xy
            ),
            Self::Invalid { readings } => write!(
                f,
                "image readings must be finite, with the sharpness, Exx and Eyy not negative \
                 (sharpness {}, gradient structure tensor {:?})",
                readings.sharpness, readings.gradient_structure_tensor_xx_yy_xy
            ),
        }
    }
}

impl std::error::Error for ImageReadingsError {}

/// The size `(W', H')` of the reading image of a `width` x `height` image.
///
/// With `L = max(width, height)` and `S = min(width, height)`: when
/// `L ≤ 480` the size is unchanged (the image is never enlarged); otherwise
/// the long side becomes 480 and the short side
/// `max(1, floor((2 · S · 480 + L) / (2 · L)))`, which is `S · 480 / L`
/// rounded to the nearest integer with an exact half rounded up. Which side
/// is long is kept.
pub fn reading_image_size(width: usize, height: usize) -> (usize, usize) {
    let long = width.max(height);
    if long <= READING_IMAGE_LONG_SIDE {
        return (width, height);
    }
    let short = width.min(height) as u128;
    let (l, target) = (long as u128, READING_IMAGE_LONG_SIDE as u128);
    let short_out = ((2 * short * target + l) / (2 * l)).max(1) as usize;
    if width >= height {
        (READING_IMAGE_LONG_SIDE, short_out)
    } else {
        (short_out, READING_IMAGE_LONG_SIDE)
    }
}

/// The sharpness and gradient structure tensor of an 8-bit RGB image.
///
/// `rgb` is `height` rows of `width` pixels, `[R, G, B]` per pixel, in stored
/// order (no EXIF orientation applied). Each row is made grey by
/// [`ImageToGray::gray_row`], and the grey image is area averaged by
/// [`area_resize`] to [`reading_image_size`]: treating it as constant over
/// each unit pixel square, output pixel `(i, j)` is its mean over
/// `[i · W / W', (i + 1) · W / W') × [j · H / H', (j + 1) · H / H')`. The
/// readings are then taken over the interior, `1 ≤ x ≤ W' − 2` and
/// `1 ≤ y ≤ H' − 2`, with nothing padded; when `W' < 3` or `H' < 3` there is
/// no interior and every reading is `0` ([`ImageReadings::ZERO`]), without the
/// conversion being evaluated. All of it is in `f32`, in the order of
/// operations `specs/formats/sift-file-format.md` § "The reading image" states.
///
/// The work is a pass over the pixels and two passes over the 480-pixel
/// reading image, split by rows over the rayon pool; every sum is taken in a
/// fixed order, so the result does not depend on the number of threads. A
/// 3840 x 2160 frame takes a few milliseconds, so the function takes no
/// `Progress`.
///
/// # Errors
///
/// [`ImageReadingsError::PixelCount`] when `rgb` is not `3 * width * height`
/// bytes, and [`ImageReadingsError::NotFinite`] when a reading is not finite,
/// as a formula that gives a NaN or an infinity on some pixel makes it.
///
/// # Example
///
/// ```
/// use sfmtool_core::features::sift::{image_readings_from_rgb, ImageToGray};
///
/// // A 4 x 3 image, black on the left half and white on the right.
/// let rgb: Vec<u8> = (0..3)
///     .flat_map(|_| [0u8, 0, 255, 255])
///     .flat_map(|v| [v, v, v])
///     .collect();
/// let readings = image_readings_from_rgb(4, 3, &rgb, &ImageToGray::default())?;
/// // The interior is pixels (1, 1) and (2, 1); the step's Sobel derivative is
/// // 1/2 at both, so Exx = 1/4, and the Laplacians are +1 and −1.
/// assert!((readings.gradient_structure_tensor_xx_yy_xy[0] - 0.25).abs() < 1e-6);
/// assert!((readings.sharpness - 1.0).abs() < 1e-6);
/// # Ok::<(), sfmtool_core::features::sift::ImageReadingsError>(())
/// ```
pub fn image_readings_from_rgb(
    width: u32,
    height: u32,
    rgb: &[u8],
    image_to_gray: &ImageToGray,
) -> Result<ImageReadings, ImageReadingsError> {
    let (w, h) = (width as usize, height as usize);
    if w.checked_mul(h).and_then(|n| n.checked_mul(3)) != Some(rgb.len()) {
        return Err(ImageReadingsError::PixelCount {
            width,
            height,
            len: rgb.len(),
        });
    }
    let (out_w, out_h) = reading_image_size(w, h);
    if out_w < 3 || out_h < 3 {
        return Ok(ImageReadings::ZERO);
    }
    let image = reading_image(w, h, rgb, image_to_gray, out_w, out_h);
    finite_readings(readings_of(&image, out_w, out_h), image_to_gray)
}

/// [`image_readings_from_rgb`] of an image already made grey by
/// [`gray_from_rgb`](super::gray_from_rgb) through `image_to_gray`, with the
/// same result: the reading image is resized from `gray`'s rows, which are
/// the samples [`ImageToGray::gray_row`] gives. `.sift` writing uses it so
/// that the image is made grey once, for the detector and the readings both.
pub(crate) fn image_readings_from_gray(
    gray: &GrayImage,
    image_to_gray: &ImageToGray,
) -> Result<ImageReadings, ImageReadingsError> {
    let (w, h) = (gray.width() as usize, gray.height() as usize);
    let (out_w, out_h) = reading_image_size(w, h);
    if out_w < 3 || out_h < 3 {
        return Ok(ImageReadings::ZERO);
    }
    let image = area_resize(
        (w, h),
        1,
        (out_w, out_h),
        || (),
        |y, row, _| row.copy_from_slice(&gray.data()[y * w..(y + 1) * w]),
    );
    finite_readings(readings_of(&image, out_w, out_h), image_to_gray)
}

/// `readings`, or [`ImageReadingsError::NotFinite`] when one is not finite.
fn finite_readings(
    readings: ImageReadings,
    image_to_gray: &ImageToGray,
) -> Result<ImageReadings, ImageReadingsError> {
    let finite = readings.sharpness.is_finite()
        && readings
            .gradient_structure_tensor_xx_yy_xy
            .iter()
            .all(|v| v.is_finite());
    if finite {
        Ok(readings)
    } else {
        Err(ImageReadingsError::NotFinite {
            image_to_gray: image_to_gray.to_json().to_string(),
            readings,
        })
    }
}

/// The reading image, `out_h` rows of `out_w` samples, of a `w` x `h` RGB
/// image whose size is at least `out_w` x `out_h`: each source row made grey
/// by [`ImageToGray::gray_row`] and the grey image resized by
/// [`area_resize`].
pub(crate) fn reading_image(
    w: usize,
    h: usize,
    rgb: &[u8],
    image_to_gray: &ImageToGray,
    out_w: usize,
    out_h: usize,
) -> Vec<f32> {
    area_resize(
        (w, h),
        1,
        (out_w, out_h),
        GrayRowScratch::default,
        |y, row, scratch| image_to_gray.gray_row(&rgb[y * w * 3..(y + 1) * w * 3], row, scratch),
    )
}

/// The readings over the interior of a `w` x `h` reading image, `w, h ≥ 3`,
/// in `f32`: each row's terms summed from left to right, the row sums from
/// top to bottom, and the variance in a second pass about the mean.
pub(crate) fn readings_of(image: &[f32], w: usize, h: usize) -> ImageReadings {
    let at = |x: usize, y: usize| image[y * w + x];
    let laplacian = |x: usize, y: usize| {
        at(x - 1, y) + at(x + 1, y) + at(x, y - 1) + at(x, y + 1) - 4.0 * at(x, y)
    };
    // Each interior row's sums of the Laplacian, Gx², Gy² and Gx·Gy.
    let rows: Vec<[f32; 4]> = (1..h - 1)
        .into_par_iter()
        .map(|y| {
            let mut sums = [0f32; 4];
            for x in 1..w - 1 {
                let gx = (at(x + 1, y - 1) + 2.0 * at(x + 1, y) + at(x + 1, y + 1)
                    - at(x - 1, y - 1)
                    - 2.0 * at(x - 1, y)
                    - at(x - 1, y + 1))
                    / 8.0;
                let gy = (at(x - 1, y + 1) + 2.0 * at(x, y + 1) + at(x + 1, y + 1)
                    - at(x - 1, y - 1)
                    - 2.0 * at(x, y - 1)
                    - at(x + 1, y - 1))
                    / 8.0;
                sums[0] += laplacian(x, y);
                sums[1] += gx * gx;
                sums[2] += gy * gy;
                sums[3] += gx * gy;
            }
            sums
        })
        .collect();
    let total = |k: usize| rows.iter().fold(0f32, |acc, row| acc + row[k]);
    // At most 478 x 478 interior pixels, so n is exact in an f32.
    let n = ((w - 2) * (h - 2)) as f32;
    let mean = total(0) / n;
    let squares: Vec<f32> = (1..h - 1)
        .into_par_iter()
        .map(|y| {
            (1..w - 1).fold(0f32, |acc, x| {
                let d = laplacian(x, y) - mean;
                acc + d * d
            })
        })
        .collect();
    ImageReadings {
        sharpness: squares.iter().fold(0f32, |acc, v| acc + v) / n,
        gradient_structure_tensor_xx_yy_xy: [total(1) / n, total(2) / n, total(3) / n],
    }
}

#[cfg(test)]
mod tests;
