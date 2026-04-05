// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Gaussian image pyramid for coarse-to-fine optical flow.

use super::GrayImage;

/// Gaussian image pyramid with 2x downsampling per level.
pub struct ImagePyramid {
    levels: Vec<GrayImage>,
    /// The full-pyramid index of `levels[0]`. Usually 0 for pyramids built from
    /// full resolution, but nonzero for pyramids built from a mid-level image
    /// via [`build_from_level`].
    start_level: u32,
}

impl ImagePyramid {
    /// Build pyramid by fused Gaussian blur + 2x downsample at each level.
    ///
    /// Uses a 6-tap binomial kernel `[1, 5, 10, 10, 5, 1] / 32` applied separably.
    /// The even width places the filter center between two input pixels, producing
    /// properly centered samples for 2x downsampling. The horizontal pass fuses the
    /// blur and horizontal downsample into a single step.
    ///
    /// Level 0 is the original image. Level i is 2^i times downsampled.
    pub fn build(image: &GrayImage, num_levels: u32) -> Self {
        let mut levels = Vec::with_capacity(num_levels as usize);
        levels.push(GrayImage::new(
            image.width(),
            image.height(),
            image.data().to_vec(),
        ));

        for _ in 1..num_levels {
            let prev = levels.last().unwrap();
            levels.push(blur_downsample_2x(prev));
        }

        Self {
            levels,
            start_level: 0,
        }
    }

    /// Build pyramid starting from a pre-built level image.
    ///
    /// `start_level` is the index of the provided image in the full pyramid.
    /// The returned pyramid has `num_additional_levels + 1` entries: entry 0
    /// is the provided image (at `start_level` resolution), and subsequent
    /// entries are further downsampled.
    ///
    /// Use [`level_in_full`] to access by full-pyramid index.
    pub fn build_from_level(
        start_image: &GrayImage,
        start_level: u32,
        num_additional_levels: u32,
    ) -> Self {
        let total = num_additional_levels + 1;
        let mut levels = Vec::with_capacity(total as usize);
        levels.push(GrayImage::new(
            start_image.width(),
            start_image.height(),
            start_image.data().to_vec(),
        ));

        for _ in 0..num_additional_levels {
            let prev = levels.last().unwrap();
            levels.push(blur_downsample_2x(prev));
        }

        Self {
            levels,
            start_level,
        }
    }

    /// Get a specific pyramid level. Level 0 is original resolution.
    pub fn level(&self, i: usize) -> &GrayImage {
        &self.levels[i - self.start_level as usize]
    }

    #[cfg(test)]
    pub fn num_levels(&self) -> usize {
        self.levels.len()
    }
}

/// 6-tap Gaussian kernel (sigma=1.0) for fused blur + 2x downsample.
/// Taps at -2.5, -1.5, -0.5, +0.5, +1.5, +2.5 relative to center.
const GAUSS6_1D: [f32; 6] = [0.017560, 0.129748, 0.352692, 0.352692, 0.129748, 0.017560];

/// Fused Gaussian blur + 2x downsample using a 6-tap separable kernel.
///
/// Pass 1 (horizontal): blur + downsample columns by 2x, producing a half-width intermediate.
/// Pass 2 (vertical): blur + downsample rows by 2x on the intermediate, producing the final result.
fn blur_downsample_2x(img: &GrayImage) -> GrayImage {
    let in_w = img.width() as usize;
    let in_h = img.height() as usize;
    let out_w = in_w / 2;
    let out_h = in_h / 2;

    // Horizontal blur + downsample: in_h rows × out_w columns
    let mut horiz = vec![0.0f32; in_h * out_w];
    for row in 0..in_h {
        let row_data = &img.data()[row * in_w..][..in_w];
        for oc in 0..out_w {
            // Center of the 6-tap filter is between input pixels 2*oc and 2*oc+1.
            // Taps at: 2*oc - 2, 2*oc - 1, 2*oc, 2*oc + 1, 2*oc + 2, 2*oc + 3
            let base = 2 * oc;
            let mut sum = 0.0;
            for (k, &weight) in GAUSS6_1D.iter().enumerate() {
                let ic = (base as i32 + k as i32 - 2).clamp(0, in_w as i32 - 1) as usize;
                sum += weight * row_data[ic];
            }
            horiz[row * out_w + oc] = sum;
        }
    }

    // Vertical blur + downsample: out_h rows × out_w columns
    let mut result = vec![0.0f32; out_h * out_w];
    for or in 0..out_h {
        for col in 0..out_w {
            let base = 2 * or;
            let mut sum = 0.0;
            for (k, &weight) in GAUSS6_1D.iter().enumerate() {
                let ir = (base as i32 + k as i32 - 2).clamp(0, in_h as i32 - 1) as usize;
                sum += weight * horiz[ir * out_w + col];
            }
            result[or * out_w + col] = sum;
        }
    }

    GrayImage::new(out_w as u32, out_h as u32, result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pyramid_levels() {
        let img = GrayImage::checkerboard(64, 64);
        let pyr = ImagePyramid::build(&img, 4);
        assert_eq!(pyr.num_levels(), 4);
        assert_eq!(pyr.level(0).width(), 64);
        assert_eq!(pyr.level(0).height(), 64);
        assert_eq!(pyr.level(1).width(), 32);
        assert_eq!(pyr.level(1).height(), 32);
        assert_eq!(pyr.level(2).width(), 16);
        assert_eq!(pyr.level(2).height(), 16);
        assert_eq!(pyr.level(3).width(), 8);
        assert_eq!(pyr.level(3).height(), 8);
    }

    #[test]
    fn test_pyramid_preserves_original() {
        let img = GrayImage::checkerboard(32, 32);
        let pyr = ImagePyramid::build(&img, 3);
        // Level 0 should be identical to the original
        assert_eq!(pyr.level(0).data(), img.data());
    }

    #[test]
    fn test_blur_downsample_constant_image() {
        let img = GrayImage::new_constant(16, 16, 0.5);
        let down = blur_downsample_2x(&img);
        assert_eq!(down.width(), 8);
        assert_eq!(down.height(), 8);
        // Blurring+downsampling a constant image should preserve the constant
        for &val in down.data() {
            assert!((val - 0.5).abs() < 1e-5, "Expected 0.5, got {}", val);
        }
    }

    #[test]
    fn test_pyramid_single_level() {
        let img = GrayImage::new_constant(8, 8, 0.3);
        let pyr = ImagePyramid::build(&img, 1);
        assert_eq!(pyr.num_levels(), 1);
        assert_eq!(pyr.level(0).width(), 8);
    }

    #[test]
    fn test_blur_downsample_centering() {
        // Create image with a linear gradient: pixel value = column index
        let w = 16u32;
        let h = 4u32;
        let data: Vec<f32> = (0..h).flat_map(|_| (0..w).map(|c| c as f32)).collect();
        let img = GrayImage::new(w, h, data);
        let down = blur_downsample_2x(&img);

        assert_eq!(down.width(), 8);
        assert_eq!(down.height(), 2);

        // For a linear gradient, each output pixel should be centered between
        // input pairs: output[oc] ≈ 2*oc + 0.5, since the 6-tap kernel is
        // centered at 2*oc + 0.5 and a linear function evaluates exactly at
        // the center regardless of kernel shape.
        // Interior pixels (away from edges) should be exact.
        for oc in 2..6 {
            let expected = 2.0 * oc as f32 + 0.5;
            let actual = down.data()[oc]; // row 0
            assert!(
                (actual - expected).abs() < 1e-4,
                "col {}: expected {}, got {}",
                oc,
                expected,
                actual
            );
        }
    }
}
