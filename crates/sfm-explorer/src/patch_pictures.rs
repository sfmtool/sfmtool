// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The pictures the viewer draws of a point's patch, for the panels that show
//! them: Track View's header slot and recent items strip, and the Track
//! Table's *Patch* column.
//!
//! [`stored_patch_image`] turns a patch bitmap into an opaque picture, and
//! [`STORED_PATCH_HOVER`] is what both panels say over a point's stored patch,
//! so the same bitmap is drawn and described the same way in each.

/// The hover text over a point's stored patch: Track View's header slot for a
/// viewed point, and a Track Table *Patch* cell.
pub(crate) const STORED_PATCH_HOVER: &str =
    "The point's stored patch: the render of its reference \
     observation, or, where an older file stores none, a mean of its views";

/// The hover text where a point stores no patch: an all-zero bitmap row.
pub(crate) const NO_STORED_PATCH_HOVER: &str = "The point has no stored patch";

/// A patch bitmap, `(rows, columns, channels)`, as an opaque RGBA image, or
/// `None` when it is all zero, which is how a point with no stored patch is
/// written.
///
/// One channel is repeated across RGB, three are RGB, and a fourth, the
/// alpha (on the photograph for a view's tile, the cross-view confidence for
/// a fused mean an older file stores), is dropped for an opaque alpha. What
/// Track View's header slot draws at the track stage, the patch bitmap, which
/// for a track read from a point is that point's stored patch and which a
/// commit writes as it; and what a Track Table *Patch* cell draws.
pub(crate) fn stored_patch_image(bitmap: ndarray::ArrayView3<'_, u8>) -> Option<egui::ColorImage> {
    let [h, w, channels] = [bitmap.shape()[0], bitmap.shape()[1], bitmap.shape()[2]];
    if channels == 0 || bitmap.iter().all(|&b| b == 0) {
        return None;
    }
    let mut rgba = Vec::with_capacity(h * w * 4);
    for row in 0..h {
        for col in 0..w {
            let level = |c: usize| bitmap[[row, col, c.min(channels - 1)]];
            if channels >= 3 {
                rgba.extend_from_slice(&[level(0), level(1), level(2), 255]);
            } else {
                rgba.extend_from_slice(&[level(0), level(0), level(0), 255]);
            }
        }
    }
    Some(egui::ColorImage::from_rgba_unmultiplied([w, h], &rgba))
}

#[cfg(test)]
mod tests;
