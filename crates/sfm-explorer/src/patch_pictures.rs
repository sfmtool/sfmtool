// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The pictures the viewer draws of a point's patch, for the panels that show
//! them: Track View's header slot and recent items strip, and the Track
//! Table's *Patch* and *Crop* columns.
//!
//! [`stored_patch_image`] turns a patch bitmap into an opaque picture, and
//! [`STORED_PATCH_HOVER`] is what both panels say over a point's stored patch,
//! so the same bitmap is drawn and described the same way in each. [`crop`]
//! cuts a photograph around a patch's outline: Track View's crop beside each
//! row's tile, and the Track Table's *Crop* cell. [`marks`] is what their
//! hover views draw over the picture. Each panel reads its photographs
//! through a [`PanelRegionPool`] of its own.

use sfmtool_core::camera::sampler::{RegionPool, REGION_POOL_BYTES_PER_THREAD};

use crate::state::BUILDING_MIPMAP_LABEL;

pub(crate) mod crop;
pub(crate) mod marks;

/// The region pool one panel reads its photographs through: Track View's
/// tiles, crops and their hover views share one, and the Track Table's
/// *Crop* cells and their hover views another.
///
/// A panel's reads on the GUI thread are many small kernel calls a frame
/// rather than one, so the panel owns the pool they all reserve their regions
/// from: [`REGION_POOL_BYTES_PER_THREAD`], 3.1 MB, one rayon thread's share
/// of a kernel call's pool. Each read gives its regions back once its picture
/// is cut, since the panel keeps the picture, not the regions; a read the
/// pool cannot hold reads through the photograph's tiles, with the same
/// pixels.
pub(crate) struct PanelRegionPool(RegionPool);

impl PanelRegionPool {
    /// The pool the panel's reads reserve from.
    pub(crate) fn pool(&self) -> &RegionPool {
        &self.0
    }

    /// A pool of `bytes`, for the tests that send every read through tiles.
    #[cfg(test)]
    pub(crate) fn with_capacity(bytes: u64) -> Self {
        Self(RegionPool::with_capacity(bytes))
    }
}

impl Default for PanelRegionPool {
    fn default() -> Self {
        Self(RegionPool::with_capacity(REGION_POOL_BYTES_PER_THREAD))
    }
}

impl std::fmt::Debug for PanelRegionPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PanelRegionPool")
            .field("capacity", &self.0.capacity())
            .field("held_bytes", &self.0.held_bytes())
            .finish()
    }
}

/// The hover text over a point's stored patch: Track View's header slot for a
/// viewed point, and a Track Table *Patch* cell.
pub(crate) const STORED_PATCH_HOVER: &str =
    "The point's stored patch: the render of its reference \
     observation, or, where an older file stores none, a mean of its views";

/// The hover text over a Track Table *Patch* cell whose point stores no
/// reference observation: its stored bitmap need not be the render of the
/// photograph the *Crop* cell shows, which is the reference-view rule's pick.
pub(crate) const UNREFERENCED_PATCH_HOVER: &str =
    "The point's stored patch. The point stores no reference observation, so this \
     bitmap is not the render of the photograph in the Crop cell, which is the \
     reference-view rule's pick";

/// What the hover of a picture cut from a photograph says while the
/// photograph tiles of its hover view are loading: Track View's tiles and
/// crops, and the Track Table's crops.
pub(crate) const HOVER_LOADING: &str = "Loading the photograph…";

/// The hover text where a point stores no patch: an all-zero bitmap row.
pub(crate) const NO_STORED_PATCH_HOVER: &str = "The point has no stored patch";

/// Paint, in the picture cell `rect`, the state of a picture whose
/// photograph's mipmap *Build mipmaps* is making
/// ([`crate::state::PeekedRead::Building`]): a spinner, with
/// [`BUILDING_MIPMAP_LABEL`] under it, wrapped to the cell's width, when the
/// two fit in the cell, and the spinner alone when they do not, which the
/// caller's hover then says. What Track View's tiles and crops, the Track
/// Table's crops and Measurement View's endpoint crops draw in place of
/// their pictures meanwhile.
///
/// Painted shapes, which take no pointer, so the cell's own hover and
/// clicks are the caller's. The spinner asks for the frames it turns in,
/// so the panel repaints only while it draws one, and the run of *Build
/// mipmaps* that makes the mipmap ends with a repaint that draws the
/// picture.
pub(crate) fn paint_building_mipmap(ui: &egui::Ui, rect: egui::Rect) {
    let visuals = ui.visuals();
    let label = ui.painter().layout(
        BUILDING_MIPMAP_LABEL.to_string(),
        egui::TextStyle::Small.resolve(ui.style()),
        visuals.weak_text_color(),
        rect.width(),
    );
    let spinner = ui
        .spacing()
        .interact_size
        .y
        .min(rect.width())
        .min(rect.height());
    let gap = 2.0;
    let content = spinner + gap + label.size().y;
    let fits = content <= rect.height() && label.size().x <= rect.width();
    let top = if fits {
        rect.center().y - content / 2.0
    } else {
        rect.center().y - spinner / 2.0
    };
    egui::Spinner::new().paint_at(
        ui,
        egui::Rect::from_center_size(
            egui::pos2(rect.center().x, top + spinner / 2.0),
            egui::vec2(spinner, spinner),
        ),
    );
    if fits {
        let at = egui::pos2(rect.center().x - label.size().x / 2.0, top + spinner + gap);
        ui.painter().galley(at, label, visuals.weak_text_color());
    }
}

/// A hover view's place while its photograph's mipmap is being made: a
/// spinner beside [`BUILDING_MIPMAP_LABEL`], for the hovers of the cells
/// [`paint_building_mipmap`] paints. The spinner asks for the frames it
/// turns in while the hover is shown.
pub(crate) fn show_building_mipmap(ui: &mut egui::Ui) {
    ui.horizontal(|ui| {
        ui.spinner();
        ui.label(BUILDING_MIPMAP_LABEL);
    });
}

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
    if channels == 0 || !sfmtool_core::reconstruction::patch_bitmap_is_stored(bitmap) {
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
pub(crate) mod tests;
