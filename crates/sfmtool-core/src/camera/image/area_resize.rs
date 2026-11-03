// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! sfmtool's one area-average resampler: each output pixel is the mean of the
//! source image over the output pixel's footprint, the source taken as
//! constant over each unit pixel square.
//!
//! The reading image of
//! [`image_readings_from_rgb`](crate::features::sift::image_readings_from_rgb)
//! and [`resize_area`](crate::reconstruction::thumbnail::resize_area) both
//! resize through [`area_resize`]. A thumbnail of a photograph at least 128
//! pixels on each side is made instead by
//! [`thumbnail_from_rgb`](crate::reconstruction::thumbnail::thumbnail_from_rgb),
//! which follows OpenCV's `INTER_AREA` arithmetic so that its bytes equal the
//! ones the Python extractors store.
//!
//! **Why single precision.** The samples are 8-bit values or grey samples made
//! from them, so they carry under three significant digits. An output sums at
//! most `ceil(src / dst) + 1` taps per axis (10 for a 3840-pixel side reduced
//! to 480), so the two passes' roundings bound its relative error by about
//! 20 · 2^-24, 1.2e-6, a hundred-thousandth of an 8-bit level. An `f32` row is
//! twice the lanes of an `f64` row per vector instruction and half the memory.

use rayon::prelude::*;

/// One output sample's source taps along an axis of `src_len` samples resized
/// to `dst_len` (both at least 1): `(source index, weight)` in source order.
///
/// In units of `1 / dst_len` source pixels, output `d` covers
/// `[d · src_len, (d + 1) · src_len)` and source `s` covers
/// `[s · dst_len, (s + 1) · dst_len)`, so each overlap is an exact integer.
/// Its weight is `overlap / src_len` rounded once to the nearest `f32`: the
/// quotient is formed in `f64`, which for lengths under 2^28 never lands on a
/// tie between two `f32`s, so rounding it again to `f32` gives the nearest
/// `f32` to the exact ratio. An output's weights sum to one up to that
/// rounding. Enlarging an axis (`dst_len > src_len`) gives each output the one
/// or two source samples under it.
pub fn area_taps(src_len: usize, dst_len: usize) -> Vec<Vec<(usize, f32)>> {
    let (n, m) = (src_len as u128, dst_len as u128);
    (0..m)
        .map(|d| {
            let (lo, hi) = (d * n, (d + 1) * n);
            let first = lo / m;
            let last = hi.div_ceil(m); // exclusive
            (first..last)
                .filter_map(|s| {
                    let overlap = hi.min((s + 1) * m) - lo.max(s * m);
                    (overlap > 0).then(|| (s as usize, (overlap as f64 / n as f64) as f32))
                })
                .collect()
        })
        .collect()
}

/// The source rows a rayon task fills and reduces with one set of buffers.
const BAND_ROWS: usize = 16;

/// Resize a `width` x `height` image of `channels` interleaved samples per
/// pixel to `dst_width` x `dst_height` by area averaging, in `f32`.
///
/// `fill_row(y, row, state)` writes source row `y`, `width * channels`
/// samples, into `row`; `state` is a buffer it may keep between rows, made by
/// `init` once per band of rows. The source is never held whole: each source
/// row is reduced across into `dst_width` pixels as soon as it is filled,
/// each output sample `Σ w · row[s]` over [`area_taps`]`(width, dst_width)`
/// added in source order from `0`; then each output row is `Σ w · across[s]`
/// over `area_taps(height, dst_height)`, in source order. Every product and
/// every sum is rounded to `f32`. Bands of rows run on the rayon pool, and
/// each output is the same whatever the thread count.
///
/// Returns `dst_height` rows of `dst_width * channels` samples, row major.
/// A zero size on either side gives an empty result without calling
/// `fill_row`.
///
/// # Example
///
/// ```
/// use sfmtool_core::camera::image::area_resize::area_resize;
///
/// // A 3 x 1 grey row reduced to 2: the middle sample is split half and half.
/// let src = [0.0f32, 0.3, 0.6];
/// let out = area_resize((3, 1), 1, (2, 1), || (), |_, row, _| row.copy_from_slice(&src));
/// assert!((out[0] - 0.1).abs() < 1e-7 && (out[1] - 0.5).abs() < 1e-7);
/// ```
pub fn area_resize<S>(
    (width, height): (usize, usize),
    channels: usize,
    (dst_width, dst_height): (usize, usize),
    init: impl Fn() -> S + Sync,
    fill_row: impl Fn(usize, &mut [f32], &mut S) + Sync,
) -> Vec<f32> {
    if width == 0 || height == 0 || channels == 0 || dst_width == 0 || dst_height == 0 {
        return Vec::new();
    }
    let x_taps = area_taps(width, dst_width);
    let y_taps = area_taps(height, dst_height);
    let out_row_len = dst_width * channels;

    let mut across = vec![0f32; height * out_row_len];
    across
        .par_chunks_mut(out_row_len * BAND_ROWS)
        .enumerate()
        .for_each(|(band, out_rows)| {
            let mut state = init();
            let mut row = vec![0f32; width * channels];
            for (i, out_row) in out_rows.chunks_exact_mut(out_row_len).enumerate() {
                fill_row(band * BAND_ROWS + i, &mut row, &mut state);
                for (out_px, taps) in out_row.chunks_exact_mut(channels).zip(&x_taps) {
                    for (c, out) in out_px.iter_mut().enumerate() {
                        let mut acc = 0f32;
                        for &(s, w) in taps {
                            acc += w * row[s * channels + c];
                        }
                        *out = acc;
                    }
                }
            }
        });

    let mut image = vec![0f32; dst_height * out_row_len];
    image
        .par_chunks_mut(out_row_len)
        .zip(y_taps.par_iter())
        .for_each(|(out_row, taps)| {
            for (i, out) in out_row.iter_mut().enumerate() {
                let mut acc = 0f32;
                for &(s, w) in taps {
                    acc += w * across[s * out_row_len + i];
                }
                *out = acc;
            }
        });
    image
}

#[cfg(test)]
mod tests;
