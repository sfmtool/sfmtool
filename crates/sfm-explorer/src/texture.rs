// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! RGB → RGBA expansion for egui texture uploads.
//!
//! egui wants `ColorImage` (RGBA); every image source the viewer holds is
//! 3-channel RGB — the embedded thumbnails in
//! [`SfmrReconstruction::thumbnails_y_x_rgb`] and the decoded full-resolution
//! images in [`crate::state::AppState`]'s full-res cache. Three panels used to expand it
//! themselves, and the copies disagreed: two derived the output extent from the
//! data while the third hard-coded 128, which is the thumbnail edge the `.sfmr`
//! format happens to pin today.
//!
//! That disagreement was a latent abort rather than a cosmetic one:
//! [`egui::ColorImage::from_rgba_unmultiplied`] asserts
//! `size[0] * size[1] * 4 == rgba.len()`, so the moment a reconstruction
//! carried thumbnails of any other size, the hard-coded panel panicked while
//! its neighbour rendered fine. Both extents now come from the same place the
//! bytes do.
//!
//! [`SfmrReconstruction::thumbnails_y_x_rgb`]: sfmtool_core::SfmrReconstruction::thumbnails_y_x_rgb

use ndarray::ArrayView3;

/// Expand tightly-packed 3-channel RGB bytes into an opaque [`egui::ColorImage`].
///
/// `rgb` must hold `width * height` pixels in row-major order; the alpha
/// channel is filled with 255. Panics if the length disagrees with the extent,
/// via the assertion inside [`egui::ColorImage::from_rgba_unmultiplied`] —
/// callers derive both from one source rather than naming a size twice.
pub(crate) fn rgb_to_color_image(rgb: &[u8], [width, height]: [usize; 2]) -> egui::ColorImage {
    let mut rgba = Vec::with_capacity(width * height * 4);
    for pixel in rgb.as_chunks::<3>().0.iter() {
        rgba.extend_from_slice(&[pixel[0], pixel[1], pixel[2], 255]);
    }
    egui::ColorImage::from_rgba_unmultiplied([width, height], &rgba)
}

/// Build a [`egui::ColorImage`] from one image's `(y, x, rgb)` thumbnail view.
///
/// The extent is read off `view`, so this tracks whatever thumbnail size the
/// loaded reconstruction actually carries instead of the format's current
/// 128×128. A non-contiguous view (any slicing upstream of the caller) is
/// copied element-wise rather than rejected.
pub(crate) fn thumbnail_color_image(view: ArrayView3<u8>) -> egui::ColorImage {
    let (height, width) = (view.shape()[0], view.shape()[1]);
    match view.as_slice() {
        Some(contiguous) => rgb_to_color_image(contiguous, [width, height]),
        None => {
            let owned: Vec<u8> = view.iter().copied().collect();
            rgb_to_color_image(&owned, [width, height])
        }
    }
}

#[cfg(test)]
mod tests;
