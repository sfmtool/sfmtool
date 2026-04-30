// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Image resampling for warp maps.

use rayon::prelude::*;

use crate::warp_map::WarpMap;

/// A multi-channel image stored as packed u8 values.
///
/// Supports 1 (gray), 3 (RGB), or 4 (RGBA) channels.
/// Data is stored row-major with channels interleaved:
/// `data[row * width * channels + col * channels + channel]`.
pub struct ImageU8 {
    width: u32,
    height: u32,
    channels: u32,
    data: Vec<u8>,
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

/// Gaussian pyramid of [`ImageU8`] images for anisotropic resampling.
pub struct ImageU8Pyramid {
    levels: Vec<ImageU8>,
}

impl ImageU8Pyramid {
    /// Build a Gaussian pyramid from a full-resolution image.
    ///
    /// Level 0 is a copy of the input. Each subsequent level is 2x downsampled
    /// using a box filter. The pyramid has `num_levels` entries total.
    pub fn build(image: &ImageU8, num_levels: usize) -> Self {
        assert!(num_levels >= 1, "Pyramid must have at least 1 level");
        let mut levels = Vec::with_capacity(num_levels);
        // Level 0 is a copy of the original.
        levels.push(ImageU8::new(
            image.width,
            image.height,
            image.channels,
            image.data.clone(),
        ));

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
}

/// Bilinear sample from a multi-channel u8 image, returning an f32 value for
/// the specified channel.
///
/// Coordinates use the pixel-center-at-0.5 convention: sampling at (0.5, 0.5)
/// returns the exact value of the top-left pixel. Out-of-bounds coordinates
/// are clamped to the nearest edge pixel.
fn sample_bilinear_u8(img: &ImageU8, x: f32, y: f32, channel: u32) -> f32 {
    let gx = x - 0.5;
    let gy = y - 0.5;

    let w = img.width as i32;
    let h = img.height as i32;
    let c = img.channels as usize;

    let x0 = gx.floor() as i32;
    let y0 = gy.floor() as i32;
    let x1 = x0.saturating_add(1);
    let y1 = y0.saturating_add(1);

    let fx = gx - x0 as f32;
    let fy = gy - y0 as f32;

    let cx0 = x0.clamp(0, w - 1) as usize;
    let cx1 = x1.clamp(0, w - 1) as usize;
    let cy0 = y0.clamp(0, h - 1) as usize;
    let cy1 = y1.clamp(0, h - 1) as usize;

    let stride = img.width as usize * c;
    let ch = channel as usize;

    let v00 = img.data[cy0 * stride + cx0 * c + ch] as f32;
    let v10 = img.data[cy0 * stride + cx1 * c + ch] as f32;
    let v01 = img.data[cy1 * stride + cx0 * c + ch] as f32;
    let v11 = img.data[cy1 * stride + cx1 * c + ch] as f32;

    (1.0 - fx) * (1.0 - fy) * v00 + fx * (1.0 - fy) * v10 + (1.0 - fx) * fy * v01 + fx * fy * v11
}

/// Apply a warp map to an image using bilinear interpolation.
///
/// For each pixel `(col, row)` in the output:
///   1. Look up source coordinates `(sx, sy)` from the warp map.
///   2. If `(sx, sy)` is valid (not NaN), bilinearly interpolate from the
///      source image.
///   3. If invalid, write zero (black).
///
/// The output image has the same dimensions as the warp map and the same
/// number of channels as the input image. Rows are processed in parallel
/// with rayon.
pub fn remap_bilinear(src: &ImageU8, map: &WarpMap) -> ImageU8 {
    let out_w = map.width();
    let out_h = map.height();
    let c = src.channels;
    let out_stride = out_w as usize * c as usize;

    let rows: Vec<Vec<u8>> = (0..out_h)
        .into_par_iter()
        .map(|row| {
            let mut row_data = vec![0u8; out_stride];
            for col in 0..out_w {
                let (sx, sy) = map.get(col, row);
                if sx.is_nan() || sy.is_nan() {
                    // Leave as zero (black).
                    continue;
                }
                let base = col as usize * c as usize;
                for ch in 0..c {
                    let val = sample_bilinear_u8(src, sx, sy, ch);
                    row_data[base + ch as usize] = (val + 0.5).clamp(0.0, 255.0) as u8;
                }
            }
            row_data
        })
        .collect();

    let mut data = Vec::with_capacity((out_w as usize) * (out_h as usize) * c as usize);
    for row in rows {
        data.extend_from_slice(&row);
    }

    ImageU8 {
        width: out_w,
        height: out_h,
        channels: c,
        data,
    }
}

/// Apply a warp map with anisotropic filtering.
///
/// Requires [`WarpMap::compute_svd`] to have been called first (panics if not).
///
/// Builds a Gaussian pyramid of the source image. For each output pixel,
/// reads the precomputed SVD of the local Jacobian, selects the pyramid level
/// from the minor singular value, and takes multiple trilinearly-blended
/// samples along the major axis direction. Falls back to a single bilinear
/// sample when the mapping is non-compressive (`sigma_major <= 1`).
///
/// `max_anisotropy` caps the number of samples along the major axis.
pub fn remap_aniso(src: &ImageU8, map: &WarpMap, max_anisotropy: u32) -> ImageU8 {
    assert!(
        map.has_svd(),
        "remap_aniso requires WarpMap SVD data; call compute_svd() first"
    );

    let out_w = map.width();
    let out_h = map.height();
    let c = src.channels;
    let out_stride = out_w as usize * c as usize;

    // Build the Gaussian pyramid. Number of levels = floor(log2(min(w, h))) + 1,
    // but at least 1.
    let min_dim = src.width.min(src.height).max(1);
    let max_levels = ((min_dim as f32).log2().floor() as usize).max(1) + 1;
    let pyramid = ImageU8Pyramid::build(src, max_levels);
    let num_levels = pyramid.num_levels();

    let rows: Vec<Vec<u8>> = (0..out_h)
        .into_par_iter()
        .map(|row| {
            let mut row_data = vec![0u8; out_stride];
            for col in 0..out_w {
                let (sx, sy) = map.get(col, row);
                if sx.is_nan() || sy.is_nan() {
                    continue;
                }

                let (sigma_major, sigma_minor, major_dx, major_dy) = map.get_svd(col, row);

                let base = col as usize * c as usize;

                // Non-compressive case: single bilinear sample from level 0.
                if sigma_major <= 1.0 {
                    for ch in 0..c {
                        let val = sample_bilinear_u8(pyramid.level(0), sx, sy, ch);
                        row_data[base + ch as usize] = (val + 0.5).clamp(0.0, 255.0) as u8;
                    }
                    continue;
                }

                // Select pyramid level from sigma_minor.
                let level_f = sigma_minor.max(1.0_f32).log2();
                let level_lo = (level_f.floor() as usize).min(num_levels - 1);
                let level_hi = (level_lo + 1).min(num_levels - 1);
                let frac = if level_lo == level_hi {
                    0.0
                } else {
                    level_f - level_lo as f32
                };

                // Number of samples along the major axis.
                let ratio = sigma_major / sigma_minor.max(1.0);
                let n = (ratio.ceil() as u32).clamp(1, max_anisotropy);

                let scale_lo = (1u32 << level_lo) as f32;
                let scale_hi = (1u32 << level_hi) as f32;

                for ch in 0..c {
                    let mut sum_lo = 0.0f32;
                    let mut sum_hi = 0.0f32;

                    for i in 0..n {
                        let t = (i as f32 + 0.5) / n as f32 - 0.5;
                        let sample_x = sx + t * sigma_major * major_dx;
                        let sample_y = sy + t * sigma_major * major_dy;

                        sum_lo += sample_bilinear_u8(
                            pyramid.level(level_lo),
                            sample_x / scale_lo,
                            sample_y / scale_lo,
                            ch,
                        );
                        sum_hi += sample_bilinear_u8(
                            pyramid.level(level_hi),
                            sample_x / scale_hi,
                            sample_y / scale_hi,
                            ch,
                        );
                    }

                    let avg_lo = sum_lo / n as f32;
                    let avg_hi = sum_hi / n as f32;
                    let val = avg_lo * (1.0 - frac) + avg_hi * frac;
                    row_data[base + ch as usize] = (val + 0.5).clamp(0.0, 255.0) as u8;
                }
            }
            row_data
        })
        .collect();

    let mut data = Vec::with_capacity((out_w as usize) * (out_h as usize) * c as usize);
    for row in rows {
        data.extend_from_slice(&row);
    }

    ImageU8 {
        width: out_w,
        height: out_h,
        channels: c,
        data,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create a simple identity warp map (each pixel maps to itself).
    fn identity_warp_map(width: u32, height: u32) -> WarpMap {
        let mut data = vec![0.0f32; 2 * (width as usize) * (height as usize)];
        for row in 0..height {
            for col in 0..width {
                let idx = (row as usize * width as usize + col as usize) * 2;
                data[idx] = col as f32 + 0.5;
                data[idx + 1] = row as f32 + 0.5;
            }
        }
        WarpMap::new(width, height, data)
    }

    /// Helper: create a warp map that applies a translation (dx, dy).
    fn translation_warp_map(width: u32, height: u32, dx: f32, dy: f32) -> WarpMap {
        let mut data = vec![0.0f32; 2 * (width as usize) * (height as usize)];
        for row in 0..height {
            for col in 0..width {
                let idx = (row as usize * width as usize + col as usize) * 2;
                data[idx] = col as f32 + 0.5 + dx;
                data[idx + 1] = row as f32 + 0.5 + dy;
            }
        }
        WarpMap::new(width, height, data)
    }

    // -----------------------------------------------------------------------
    // ImageU8 basic tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_image_u8_construction_and_pixel_access() {
        let data = vec![10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120];
        let img = ImageU8::new(2, 2, 3, data);

        assert_eq!(img.width(), 2);
        assert_eq!(img.height(), 2);
        assert_eq!(img.channels(), 3);

        // Top-left pixel: (10, 20, 30)
        assert_eq!(img.get_pixel(0, 0, 0), 10);
        assert_eq!(img.get_pixel(0, 0, 1), 20);
        assert_eq!(img.get_pixel(0, 0, 2), 30);

        // Top-right pixel: (40, 50, 60)
        assert_eq!(img.get_pixel(1, 0, 0), 40);
        assert_eq!(img.get_pixel(1, 0, 1), 50);
        assert_eq!(img.get_pixel(1, 0, 2), 60);

        // Bottom-left pixel: (70, 80, 90)
        assert_eq!(img.get_pixel(0, 1, 0), 70);
        assert_eq!(img.get_pixel(0, 1, 1), 80);
        assert_eq!(img.get_pixel(0, 1, 2), 90);

        // Bottom-right pixel: (100, 110, 120)
        assert_eq!(img.get_pixel(1, 1, 0), 100);
        assert_eq!(img.get_pixel(1, 1, 1), 110);
        assert_eq!(img.get_pixel(1, 1, 2), 120);
    }

    #[test]
    fn test_image_u8_single_channel() {
        let data = vec![0, 64, 128, 255];
        let img = ImageU8::new(2, 2, 1, data);
        assert_eq!(img.get_pixel(0, 0, 0), 0);
        assert_eq!(img.get_pixel(1, 0, 0), 64);
        assert_eq!(img.get_pixel(0, 1, 0), 128);
        assert_eq!(img.get_pixel(1, 1, 0), 255);
    }

    #[test]
    fn test_image_u8_from_channels() {
        let img = ImageU8::from_channels(4, 3, 3);
        assert_eq!(img.width(), 4);
        assert_eq!(img.height(), 3);
        assert_eq!(img.channels(), 3);
        assert_eq!(img.data().len(), 4 * 3 * 3);
        assert!(img.data().iter().all(|&v| v == 0));
    }

    #[test]
    #[should_panic(expected = "data length")]
    fn test_image_u8_wrong_data_length_panics() {
        ImageU8::new(2, 2, 3, vec![0; 10]);
    }

    #[test]
    fn test_image_u8_data_mut() {
        let mut img = ImageU8::from_channels(2, 2, 1);
        img.data_mut()[0] = 42;
        assert_eq!(img.get_pixel(0, 0, 0), 42);
    }

    // -----------------------------------------------------------------------
    // Downsample tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_downsample_2x_single_channel() {
        // 4x4 image with known values.
        #[rustfmt::skip]
        let data = vec![
            10, 20, 30, 40,
            50, 60, 70, 80,
            90, 100, 110, 120,
            130, 140, 150, 160,
        ];
        let img = ImageU8::new(4, 4, 1, data);
        let down = img.downsample_2x();
        assert_eq!(down.width(), 2);
        assert_eq!(down.height(), 2);

        // Top-left block average: (10+20+50+60)/4 = 35
        assert_eq!(down.get_pixel(0, 0, 0), 35);
        // Top-right block: (30+40+70+80)/4 = 55
        assert_eq!(down.get_pixel(1, 0, 0), 55);
        // Bottom-left block: (90+100+130+140)/4 = 115
        assert_eq!(down.get_pixel(0, 1, 0), 115);
        // Bottom-right block: (110+120+150+160)/4 = 135
        assert_eq!(down.get_pixel(1, 1, 0), 135);
    }

    #[test]
    fn test_downsample_2x_rgb() {
        // 2x2 RGB image.
        #[rustfmt::skip]
        let data = vec![
            10, 20, 30,   40, 50, 60,
            70, 80, 90,  100, 110, 120,
        ];
        let img = ImageU8::new(2, 2, 3, data);
        let down = img.downsample_2x();
        assert_eq!(down.width(), 1);
        assert_eq!(down.height(), 1);

        // R: (10+40+70+100)/4 = 55
        assert_eq!(down.get_pixel(0, 0, 0), 55);
        // G: (20+50+80+110)/4 = 65
        assert_eq!(down.get_pixel(0, 0, 1), 65);
        // B: (30+60+90+120)/4 = 75
        assert_eq!(down.get_pixel(0, 0, 2), 75);
    }

    // -----------------------------------------------------------------------
    // Pyramid tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_pyramid_builds_correct_levels() {
        let img = ImageU8::from_channels(64, 32, 1);
        let pyr = ImageU8Pyramid::build(&img, 4);
        assert_eq!(pyr.num_levels(), 4);
        assert_eq!(pyr.level(0).width(), 64);
        assert_eq!(pyr.level(0).height(), 32);
        assert_eq!(pyr.level(1).width(), 32);
        assert_eq!(pyr.level(1).height(), 16);
        assert_eq!(pyr.level(2).width(), 16);
        assert_eq!(pyr.level(2).height(), 8);
        assert_eq!(pyr.level(3).width(), 8);
        assert_eq!(pyr.level(3).height(), 4);
    }

    #[test]
    fn test_pyramid_halves_dimensions() {
        let img = ImageU8::from_channels(128, 128, 3);
        let pyr = ImageU8Pyramid::build(&img, 6);
        for i in 1..pyr.num_levels() {
            assert_eq!(pyr.level(i).width(), pyr.level(i - 1).width() / 2);
            assert_eq!(pyr.level(i).height(), pyr.level(i - 1).height() / 2);
            assert_eq!(pyr.level(i).channels(), 3);
        }
    }

    #[test]
    fn test_pyramid_single_level() {
        let img = ImageU8::from_channels(8, 8, 1);
        let pyr = ImageU8Pyramid::build(&img, 1);
        assert_eq!(pyr.num_levels(), 1);
        assert_eq!(pyr.level(0).width(), 8);
    }

    #[test]
    fn test_pyramid_stops_at_small_dimension() {
        // 4x4 → 2x2 → 1x1. Requesting 10 levels should stop after 3
        // because 1x1 has width < 2 so no further downsample.
        let img = ImageU8::from_channels(4, 4, 1);
        let pyr = ImageU8Pyramid::build(&img, 10);
        assert_eq!(pyr.num_levels(), 3);
        assert_eq!(pyr.level(0).width(), 4);
        assert_eq!(pyr.level(1).width(), 2);
        assert_eq!(pyr.level(2).width(), 1);
    }

    // -----------------------------------------------------------------------
    // sample_bilinear_u8 tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_sample_bilinear_at_pixel_center() {
        let data = vec![100, 200, 50, 150];
        let img = ImageU8::new(2, 2, 1, data);

        // Pixel center of (0,0) is at (0.5, 0.5).
        let val = sample_bilinear_u8(&img, 0.5, 0.5, 0);
        assert!((val - 100.0).abs() < 1e-3, "got {}", val);

        // Pixel center of (1,0) is at (1.5, 0.5).
        let val = sample_bilinear_u8(&img, 1.5, 0.5, 0);
        assert!((val - 200.0).abs() < 1e-3, "got {}", val);

        // Pixel center of (1,1) is at (1.5, 1.5).
        let val = sample_bilinear_u8(&img, 1.5, 1.5, 0);
        assert!((val - 150.0).abs() < 1e-3, "got {}", val);
    }

    #[test]
    fn test_sample_bilinear_interpolated() {
        let data = vec![0, 100, 0, 100];
        let img = ImageU8::new(2, 2, 1, data);

        // Midpoint between pixel (0,0) and (1,0): x=1.0, y=0.5
        let val = sample_bilinear_u8(&img, 1.0, 0.5, 0);
        assert!((val - 50.0).abs() < 1e-3, "got {}", val);
    }

    #[test]
    fn test_sample_bilinear_four_pixel_average() {
        let data = vec![0, 100, 200, 60];
        let img = ImageU8::new(2, 2, 1, data);

        // Center of four pixels: (1.0, 1.0) -> average = (0+100+200+60)/4 = 90
        let val = sample_bilinear_u8(&img, 1.0, 1.0, 0);
        assert!((val - 90.0).abs() < 1e-3, "got {}", val);
    }

    // -----------------------------------------------------------------------
    // remap_bilinear tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_remap_bilinear_identity_preserves_image() {
        // 4x4 single-channel image with gradient values.
        let data: Vec<u8> = (0..16).map(|i| (i * 16) as u8).collect();
        let src = ImageU8::new(4, 4, 1, data.clone());
        let map = identity_warp_map(4, 4);

        let result = remap_bilinear(&src, &map);

        assert_eq!(result.width(), 4);
        assert_eq!(result.height(), 4);
        for (i, (got, want)) in result.data().iter().zip(data.iter()).enumerate() {
            assert_eq!(got, want, "pixel {i} differs: {got} vs {want}");
        }
    }

    #[test]
    fn test_remap_bilinear_identity_rgb() {
        // 3x3 RGB image.
        let mut data = Vec::new();
        for i in 0..9 {
            data.push((i * 10) as u8); // R
            data.push((i * 20) as u8); // G
            data.push((i * 30) as u8); // B
        }
        let src = ImageU8::new(3, 3, 3, data.clone());
        let map = identity_warp_map(3, 3);

        let result = remap_bilinear(&src, &map);

        assert_eq!(result.width(), 3);
        assert_eq!(result.height(), 3);
        assert_eq!(result.channels(), 3);
        for (i, (got, want)) in result.data().iter().zip(data.iter()).enumerate() {
            assert_eq!(got, want, "byte {i} differs: {got} vs {want}");
        }
    }

    #[test]
    fn test_remap_bilinear_with_translation() {
        // 4x4 single-channel image: value = col * 50.
        #[rustfmt::skip]
        let data = vec![
            0, 50, 100, 150,
            0, 50, 100, 150,
            0, 50, 100, 150,
            0, 50, 100, 150,
        ];
        let src = ImageU8::new(4, 4, 1, data);

        // Shift by +1 pixel in x: each output pixel samples from (col+1.5)
        // in source, i.e. one pixel to the right.
        let map = translation_warp_map(4, 4, 1.0, 0.0);
        let result = remap_bilinear(&src, &map);

        // Output col 0 should get source col 1 value (50).
        assert_eq!(result.get_pixel(0, 0, 0), 50);
        // Output col 1 should get source col 2 value (100).
        assert_eq!(result.get_pixel(1, 0, 0), 100);
        // Output col 2 should get source col 3 value (150).
        assert_eq!(result.get_pixel(2, 0, 0), 150);
        // Output col 3 is clamped to source col 3 (150).
        assert_eq!(result.get_pixel(3, 0, 0), 150);
    }

    #[test]
    fn test_remap_bilinear_nan_gives_black() {
        let src = ImageU8::new(2, 2, 1, vec![100, 100, 100, 100]);

        // Warp map with NaN entries.
        let data = vec![f32::NAN, f32::NAN, 1.5, 0.5, 0.5, 1.5, f32::NAN, f32::NAN];
        let map = WarpMap::new(2, 2, data);

        let result = remap_bilinear(&src, &map);

        // NaN pixels should be black (0).
        assert_eq!(result.get_pixel(0, 0, 0), 0);
        assert_eq!(result.get_pixel(1, 1, 0), 0);
        // Valid pixels should be 100.
        assert_eq!(result.get_pixel(1, 0, 0), 100);
        assert_eq!(result.get_pixel(0, 1, 0), 100);
    }

    #[test]
    fn test_remap_bilinear_multichannel_rgb() {
        // 2x2 RGB image.
        #[rustfmt::skip]
        let data = vec![
            255, 0, 0,      0, 255, 0,
            0, 0, 255,    128, 128, 128,
        ];
        let src = ImageU8::new(2, 2, 3, data.clone());
        let map = identity_warp_map(2, 2);

        let result = remap_bilinear(&src, &map);

        for (i, (got, want)) in result.data().iter().zip(data.iter()).enumerate() {
            assert_eq!(got, want, "byte {i} differs");
        }
    }

    // -----------------------------------------------------------------------
    // remap_aniso tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_remap_aniso_isotropic_matches_bilinear() {
        // With an identity warp map and SVD indicating no compression
        // (sigma_major=1, sigma_minor=1), remap_aniso should produce the
        // same result as remap_bilinear.
        let data: Vec<u8> = (0..16).map(|i| (i * 16) as u8).collect();
        let src = ImageU8::new(4, 4, 1, data);
        let mut map = identity_warp_map(4, 4);
        map.compute_svd();

        let bilinear_result = remap_bilinear(&src, &map);
        let aniso_result = remap_aniso(&src, &map, 16);

        assert_eq!(bilinear_result.width(), aniso_result.width());
        assert_eq!(bilinear_result.height(), aniso_result.height());
        for i in 0..bilinear_result.data().len() {
            let diff = (bilinear_result.data()[i] as i32 - aniso_result.data()[i] as i32).abs();
            assert!(
                diff <= 1,
                "pixel {} differs too much: {} vs {} (diff {})",
                i,
                bilinear_result.data()[i],
                aniso_result.data()[i],
                diff
            );
        }
    }

    #[test]
    #[should_panic(expected = "requires WarpMap SVD")]
    fn test_remap_aniso_panics_without_svd() {
        let src = ImageU8::from_channels(4, 4, 1);
        let map = identity_warp_map(4, 4);
        remap_aniso(&src, &map, 16);
    }
}
