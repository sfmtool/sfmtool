// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The two per-image readings of an image's sharpness: the variance of the
//! Laplacian and the gradient structure tensor of its reading image.
//!
//! The **reading image** is the image made grey by an [`ImageToGray`]
//! conversion, in double precision, and reduced by an exact area average to a
//! long side of [`READING_IMAGE_LONG_SIDE`] pixels (an image no larger is used
//! at its own size). On its interior pixels, those whose 3 x 3 neighbourhood
//! lies inside it:
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
//! ratio). [`image_readings_from_rgb`] is the one implementation of both.

use rayon::prelude::*;

use super::gray::ImageToGray;

/// The long side of the reading image, in pixels.
pub const READING_IMAGE_LONG_SIDE: usize = 480;

/// An image's sharpness and gradient structure tensor, from
/// [`image_readings_from_rgb`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ImageReadings {
    /// The population variance of the Laplacian over the reading image's
    /// interior, in grey units squared. Finite and not negative.
    pub sharpness: f64,
    /// `[Exx, Eyy, Exy]`: the means of `Gx²`, `Gy²` and `Gx·Gy` over the
    /// reading image's interior, in grey units squared per pixel squared, not
    /// normalised. Finite; `Exx` and `Eyy` not negative.
    pub gradient_structure_tensor_xx_yy_xy: [f64; 3],
}

impl ImageReadings {
    /// The readings of an image whose reading image has no interior pixel:
    /// all zero.
    pub const ZERO: Self = Self {
        sharpness: 0.0,
        gradient_structure_tensor_xx_yy_xy: [0.0; 3],
    };
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
    /// formula can (`R**-1` at `R = 0`, say).
    NotFinite {
        /// The conversion's `image_to_gray` object, as JSON text.
        image_to_gray: String,
        /// The readings as computed.
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
/// order (no EXIF orientation applied). Each channel is divided by 255 and the
/// pixel made grey by `image_to_gray` in double precision; the grey image is
/// area averaged to [`reading_image_size`]: treating it as constant over each
/// unit pixel square, output pixel `(i, j)` is its mean over
/// `[i · W / W', (i + 1) · W / W') × [j · H / H', (j + 1) · H / H')`. Each
/// coverage weight is the source pixel's exact overlap with that footprint
/// (a ratio of integers) divided by the footprint's width, rounded once.
/// The readings are then taken over the interior, `1 ≤ x ≤ W' − 2` and
/// `1 ≤ y ≤ H' − 2`, with nothing padded; when `W' < 3` or `H' < 3` there is
/// no interior and every reading is `0` ([`ImageReadings::ZERO`]), without the
/// conversion being evaluated.
///
/// The work is a pass over the pixels and a pass over the 480-pixel reading
/// image, split by rows over the rayon pool; every sum is taken in a fixed
/// order, so the result does not depend on the number of threads. A 3840 x
/// 2160 frame takes a few milliseconds, so the function takes no `Progress`.
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
/// assert!((readings.gradient_structure_tensor_xx_yy_xy[0] - 0.25).abs() < 1e-12);
/// assert!((readings.sharpness - 1.0).abs() < 1e-12);
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
    let readings = readings_of(&image, out_w, out_h);
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

/// One output sample's source taps along an axis of `src_len` samples
/// reduced to `dst_len` (`1 ≤ dst_len ≤ src_len`): `(source index, weight)`
/// in source order.
///
/// In units of `1 / dst_len` source pixels, output `d` covers
/// `[d · src_len, (d + 1) · src_len)` and source `s` covers
/// `[s · dst_len, (s + 1) · dst_len)`, so each overlap is an exact integer and
/// its weight `overlap / src_len` is rounded once.
pub(crate) fn area_taps(src_len: usize, dst_len: usize) -> Vec<Vec<(usize, f64)>> {
    let (n, m) = (src_len as u128, dst_len as u128);
    (0..m)
        .map(|d| {
            let (lo, hi) = (d * n, (d + 1) * n);
            let first = lo / m;
            let last = hi.div_ceil(m); // exclusive
            (first..last)
                .filter_map(|s| {
                    let overlap = hi.min((s + 1) * m) - lo.max(s * m);
                    (overlap > 0).then(|| (s as usize, overlap as f64 / n as f64))
                })
                .collect()
        })
        .collect()
}

/// The reading image, `out_h` rows of `out_w` samples, of a `w` x `h` RGB
/// image whose size is at least `out_w` x `out_h`.
///
/// Each source row is made grey and reduced across into `out_w` sums, then
/// each output row is the weighted sum of the reduced rows under it, in
/// source order.
pub(crate) fn reading_image(
    w: usize,
    h: usize,
    rgb: &[u8],
    image_to_gray: &ImageToGray,
    out_w: usize,
    out_h: usize,
) -> Vec<f64> {
    let x_taps = area_taps(w, out_w);
    let y_taps = area_taps(h, out_h);

    // Each 8-bit sample divided by 255, looked up rather than divided per pixel.
    let unit: [f64; 256] = std::array::from_fn(|v| v as f64 / 255.0);

    // Source rows go to the pool in bands, each band with its own row buffers,
    // so the buffers are allocated once per band rather than once per row.
    const BAND_ROWS: usize = 16;
    let mut across = vec![0f64; h * out_w];
    across
        .par_chunks_mut(out_w * BAND_ROWS)
        .enumerate()
        .for_each(|(band, out_rows)| {
            let [mut r, mut g, mut b, mut gray] = std::array::from_fn(|_| vec![0f64; w]);
            let mut scratch = Vec::new();
            for (i, out_row) in out_rows.chunks_exact_mut(out_w).enumerate() {
                let y = band * BAND_ROWS + i;
                let src = &rgb[y * w * 3..(y + 1) * w * 3];
                for (x, p) in src.as_chunks::<3>().0.iter().enumerate() {
                    r[x] = unit[usize::from(p[0])];
                    g[x] = unit[usize::from(p[1])];
                    b[x] = unit[usize::from(p[2])];
                }
                image_to_gray.gray_slices((&r, &g, &b), &mut gray, &mut scratch);
                for (out, taps) in out_row.iter_mut().zip(&x_taps) {
                    *out = taps.iter().map(|&(s, wt)| wt * gray[s]).sum();
                }
            }
        });

    let mut image = vec![0f64; out_h * out_w];
    image
        .par_chunks_mut(out_w)
        .zip(y_taps.par_iter())
        .for_each(|(out_row, taps)| {
            for (x, out) in out_row.iter_mut().enumerate() {
                *out = taps.iter().map(|&(s, wt)| wt * across[s * out_w + x]).sum();
            }
        });
    image
}

/// A compensated (Neumaier) running sum, so the means over a few hundred
/// thousand interior pixels lose no more than a rounding or two.
#[derive(Default)]
struct Sum {
    total: f64,
    compensation: f64,
}

impl Sum {
    fn add(&mut self, v: f64) {
        let t = self.total + v;
        if self.total.abs() >= v.abs() {
            self.compensation += (self.total - t) + v;
        } else {
            self.compensation += (v - t) + self.total;
        }
        self.total = t;
    }

    fn value(&self) -> f64 {
        self.total + self.compensation
    }
}

/// The readings over the interior of a `w` x `h` reading image, `w, h ≥ 3`.
pub(crate) fn readings_of(image: &[f64], w: usize, h: usize) -> ImageReadings {
    let at = |x: usize, y: usize| image[y * w + x];
    let n = ((w - 2) * (h - 2)) as f64;
    let mut laplacians = Vec::with_capacity((w - 2) * (h - 2));
    let (mut sum_l, mut sum_xx, mut sum_yy, mut sum_xy) = (
        Sum::default(),
        Sum::default(),
        Sum::default(),
        Sum::default(),
    );
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let laplacian =
                at(x - 1, y) + at(x + 1, y) + at(x, y - 1) + at(x, y + 1) - 4.0 * at(x, y);
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
            laplacians.push(laplacian);
            sum_l.add(laplacian);
            sum_xx.add(gx * gx);
            sum_yy.add(gy * gy);
            sum_xy.add(gx * gy);
        }
    }
    let mean = sum_l.value() / n;
    let mut sum_sq = Sum::default();
    for l in &laplacians {
        let d = l - mean;
        sum_sq.add(d * d);
    }
    ImageReadings {
        sharpness: sum_sq.value() / n,
        gradient_structure_tensor_xx_yy_xy: [
            sum_xx.value() / n,
            sum_yy.value() / n,
            sum_xy.value() / n,
        ],
    }
}

#[cfg(test)]
mod tests;
