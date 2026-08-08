// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! [`GrayImage`], the single-channel f32 image the flow pipeline reads.
//!
//! Pixel centers are at `(col + 0.5, row + 0.5)`, matching the
//! `.sfmr` / `.sift` (COLMAP) convention. Used well beyond optical flow —
//! `sift::scale_space` documents its conventions as following this type.

use super::interp;

/// Simple wrapper for grayscale image data.
///
/// Coordinate convention: pixel centers are at (col + 0.5, row + 0.5), matching the
/// .sfmr/.sift format convention (COLMAP convention). Sampling at (0.5, 0.5)
/// returns the exact value of the top-left pixel.
pub struct GrayImage {
    width: u32,
    height: u32,
    /// Pixel data normalized to [0, 1], row-major.
    data: Vec<f32>,
}

impl GrayImage {
    /// Create a new image from raw f32 data (must be width * height elements).
    pub fn new(width: u32, height: u32, data: Vec<f32>) -> Self {
        assert_eq!(data.len(), (width as usize) * (height as usize));
        Self {
            width,
            height,
            data,
        }
    }

    /// Create a constant-valued image.
    pub fn new_constant(width: u32, height: u32, value: f32) -> Self {
        Self {
            width,
            height,
            data: vec![value; (width as usize) * (height as usize)],
        }
    }

    /// Create from u8 data, normalizing to [0, 1].
    pub fn from_u8(width: u32, height: u32, data: &[u8]) -> Self {
        assert_eq!(data.len(), (width as usize) * (height as usize));
        Self {
            width,
            height,
            data: data.iter().map(|&v| v as f32 / 255.0).collect(),
        }
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    /// Get pixel value at grid position (col, row).
    pub fn get_pixel(&self, col: u32, row: u32) -> f32 {
        self.data[(row as usize) * (self.width as usize) + (col as usize)]
    }

    /// Set pixel value at grid position (col, row).
    pub fn set_pixel(&mut self, col: u32, row: u32, value: f32) {
        self.data[(row as usize) * (self.width as usize) + (col as usize)] = value;
    }

    /// Access raw data as a slice.
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    /// Access raw data as a mutable slice.
    pub fn data_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }

    /// Create a synthetic checkerboard pattern for testing.
    pub fn checkerboard(width: u32, height: u32) -> Self {
        let mut data = vec![0.0f32; (width as usize) * (height as usize)];
        for row in 0..height {
            for col in 0..width {
                let checker = ((col / 8) + (row / 8)) % 2;
                data[(row as usize) * (width as usize) + (col as usize)] =
                    if checker == 0 { 0.2 } else { 0.8 };
            }
        }
        Self {
            width,
            height,
            data,
        }
    }

    /// Create an image shifted by integer pixels from a source image.
    /// Pixels that fall outside the source are filled with 0.
    pub fn shifted(src: &GrayImage, shift_x: f32, shift_y: f32) -> Self {
        let w = src.width;
        let h = src.height;
        let mut data = vec![0.0f32; (w as usize) * (h as usize)];
        for row in 0..h {
            for col in 0..w {
                let sx = col as f32 + 0.5 - shift_x;
                let sy = row as f32 + 0.5 - shift_y;
                data[(row as usize) * (w as usize) + (col as usize)] =
                    interp::sample_bilinear(src, sx, sy);
            }
        }
        Self {
            width: w,
            height: h,
            data,
        }
    }
}
