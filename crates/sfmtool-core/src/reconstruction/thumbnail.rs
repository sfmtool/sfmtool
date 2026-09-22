// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The thumbnail a reconstruction carries for an image, made from its
//! photograph.
//!
//! A `.sfmr` thumbnail row is the source photograph resized to
//! [`THUMBNAIL_SIZE`] x [`THUMBNAIL_SIZE`] by area averaging, stretched to the
//! square (see `specs/formats/sfmr-file-format.md`). [`resize_area`] is that
//! resize for any Rust caller. It matches OpenCV's `INTER_AREA` in method, each
//! output pixel the coverage-weighted mean of the source pixels under it, but
//! not bit for bit, so a caller that must reproduce stored rows exactly (such
//! as `sfm xform --add-thumbnails`) uses the extractors' own OpenCV path
//! instead. The viewer uses this one for display thumbnails.

use crate::THUMBNAIL_SIZE;

/// One output sample's source taps along one axis: `(source index, weight)`,
/// the weights summing to one.
fn axis_taps(src_len: usize, dst_len: usize) -> Vec<Vec<(usize, f32)>> {
    let scale = src_len as f64 / dst_len as f64;
    (0..dst_len)
        .map(|d| {
            let start = d as f64 * scale;
            let end = (d + 1) as f64 * scale;
            let mut taps = Vec::new();
            let mut s = start.floor() as usize;
            while (s as f64) < end && s < src_len {
                let lo = start.max(s as f64);
                let hi = end.min((s + 1) as f64);
                if hi > lo {
                    taps.push((s, ((hi - lo) / scale) as f32));
                }
                s += 1;
            }
            taps
        })
        .collect()
}

/// Resize an interleaved 8-bit image by area averaging.
///
/// `pixels` is `height` rows of `width` pixels of `channels` bytes each, row
/// major. Returns `dst_height` rows of `dst_width` pixels in the same layout.
/// Each output pixel is the mean of the source area it covers, each source
/// pixel weighted by the fraction of it inside that area, rounded to the
/// nearest integer. Upscaling an axis degenerates to sampling the covering
/// source pixel.
///
/// # Panics
///
/// Panics if `pixels.len() != width * height * channels`, or any dimension is
/// zero.
pub fn resize_area(
    pixels: &[u8],
    width: usize,
    height: usize,
    channels: usize,
    dst_width: usize,
    dst_height: usize,
) -> Vec<u8> {
    assert!(
        width > 0 && height > 0 && channels > 0 && dst_width > 0 && dst_height > 0,
        "resize_area needs non-empty images"
    );
    assert_eq!(
        pixels.len(),
        width * height * channels,
        "pixels must hold width * height * channels bytes"
    );
    let x_taps = axis_taps(width, dst_width);
    let y_taps = axis_taps(height, dst_height);

    // Horizontal pass into f32 rows, then the vertical pass per output row.
    let mut horizontal = vec![0f32; height * dst_width * channels];
    for y in 0..height {
        let src_row = &pixels[y * width * channels..(y + 1) * width * channels];
        let dst_row = &mut horizontal[y * dst_width * channels..(y + 1) * dst_width * channels];
        for (dx, taps) in x_taps.iter().enumerate() {
            for c in 0..channels {
                let mut acc = 0f32;
                for &(sx, w) in taps {
                    acc += w * f32::from(src_row[sx * channels + c]);
                }
                dst_row[dx * channels + c] = acc;
            }
        }
    }
    let mut out = vec![0u8; dst_height * dst_width * channels];
    for (dy, taps) in y_taps.iter().enumerate() {
        let dst_row = &mut out[dy * dst_width * channels..(dy + 1) * dst_width * channels];
        for (i, value) in dst_row.iter_mut().enumerate() {
            let mut acc = 0f32;
            for &(sy, w) in taps {
                acc += w * horizontal[sy * dst_width * channels + i];
            }
            *value = (acc + 0.5).clamp(0.0, 255.0) as u8;
        }
    }
    out
}

/// The `THUMBNAIL_SIZE` x `THUMBNAIL_SIZE` RGB thumbnail row of an RGB
/// photograph `width` x `height`, by [`resize_area`].
///
/// # Panics
///
/// As [`resize_area`], with three channels.
pub fn thumbnail_from_rgb(pixels: &[u8], width: usize, height: usize) -> Vec<u8> {
    resize_area(pixels, width, height, 3, THUMBNAIL_SIZE, THUMBNAIL_SIZE)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_integer_downscale_is_the_block_mean() {
        // 4x2 single-channel image to 2x1: each output is the mean of a 2x2 block.
        let pixels = [0u8, 10, 100, 200, 20, 30, 50, 50];
        let out = resize_area(&pixels, 4, 2, 1, 2, 1);
        assert_eq!(out, vec![15, 100]);
    }

    #[test]
    fn a_fractional_downscale_weights_by_coverage() {
        // 3 pixels to 2: the middle one is split half and half.
        let pixels = [0u8, 90, 180];
        let out = resize_area(&pixels, 3, 1, 1, 2, 1);
        // (0 * 1 + 90 * 0.5) / 1.5 = 30, (90 * 0.5 + 180 * 1) / 1.5 = 150.
        assert_eq!(out, vec![30, 150]);
    }

    #[test]
    fn a_flat_image_stays_flat_and_channels_stay_apart() {
        let pixels: Vec<u8> = (0..270 * 480).flat_map(|_| [7u8, 128, 250]).collect();
        let out = thumbnail_from_rgb(&pixels, 270, 480);
        assert_eq!(out.len(), THUMBNAIL_SIZE * THUMBNAIL_SIZE * 3);
        assert!(out.chunks(3).all(|p| p == [7, 128, 250]));
    }

    #[test]
    fn the_same_size_is_the_identity() {
        let pixels: Vec<u8> = (0..5 * 4 * 3).map(|i| (i * 37 % 256) as u8).collect();
        assert_eq!(resize_area(&pixels, 5, 4, 3, 5, 4), pixels);
    }
}
