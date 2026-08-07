// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Point Track Detail panel — shows all observations of a selected 3D point.
//!
//! When a 3D point is selected (via click in the 3D viewer or feature click in
//! the Image Detail panel), this panel displays a header with point summary
//! statistics and a scrollable table of per-image observations.
//!
//! This module owns the panel state and the [`PointTrackDetail::show`] entry
//! point that orchestrates one frame; the work lives in five children:
//!
//! - [`prepare`] — builds [`TrackObservationData`] when the selection changes,
//!   so the per-frame drawing code has nothing left to compute.
//! - [`header`] — the point summary bar and the stored-patch tile.
//! - [`table`] — the scrollable observation table, its rows and thumbnails.
//! - [`patch`] — oriented-patch frames and the textures built from them.
//! - [`metrics`] — reprojection error, ray angles, triangulation diagnostics
//!   and the error→color ramp.

use std::collections::HashMap;

use sfmtool_core::camera::remap::ImageU8;
use sfmtool_core::patch::cloud::OrientedPatch;
use sfmtool_core::SfmrReconstruction;

use crate::platform::{self, GestureEvent};
use crate::scene::{ImageRef, PointRef, ReconId};
use crate::state::CachedSiftFeatures;

mod header;
mod metrics;
mod patch;
mod prepare;
mod table;

#[cfg(test)]
mod tests;

// Re-exported at the old path: `image_detail` imports both as
// `crate::point_track_detail::<name>`.
pub(crate) use metrics::{compute_max_pairwise_angle, compute_point_diagnostics};

/// Precomputed data for one observation in the track.
struct TrackObservationData {
    /// Index into `recon.images`.
    image_index: usize,
    /// Feature index within the image's SIFT file.
    feature_index: usize,
    /// Feature position in image pixel coordinates.
    feature_xy: [f32; 2],
    /// Per-observation reprojection error in pixels.
    reproj_error: f32,
    /// Angular discrepancy between observation ray and point direction, in degrees.
    ray_angle_deg: f32,
    /// The observation's two feature extents in pixels — the *full* widths of
    /// the affine shape along its two axes (twice the half-vector column
    /// norms), ordered larger first. This is the same diameter convention the
    /// rendered patch quad spans and that `embed-patches --patch-size` uses.
    /// `[0.0, 0.0]` when no shape is available.
    feature_extents: [f32; 2],
    /// Truncated display name (e.g. "…/fisheye_left/image_0345.jpg").
    image_name: String,
    /// Full image path from the reconstruction.
    image_full_name: String,
}

/// Point Track Detail panel state.
pub struct PointTrackDetail {
    /// The point we've prepared data for, or None. A ref, so re-preparing is
    /// forced when a new reconstruction reuses the same point index — which is
    /// also what makes the texture caches below safe to rebuild wholesale.
    prepared_point: Option<PointRef>,
    /// Precomputed observation data for the current point.
    observations: Vec<TrackObservationData>,
    /// Maximum angle (degrees) between any pair of observation rays in the track.
    max_angle_deg: f32,
    /// Inverse-depth z-score (`depth / σ_depth`) of the triangulation; NaN when
    /// undefined (point at infinity or fewer than two usable rays).
    inverse_depth_z: f32,
    /// Condition number of the triangulation's normal matrix; NaN when undefined.
    condition_number: f32,
    /// Cached thumbnail textures keyed by image.
    thumbnail_textures: HashMap<ImageRef, egui::TextureHandle>,
    /// The selected point's oriented patch frame (from the stored patch
    /// half-vectors), or None when the reconstruction carries no frame or the
    /// point has no patch. Gates the per-observation "Patch" column.
    patch_frame: Option<OrientedPatch>,
    /// Stored patch bitmap texture for the selected point (header tile), if any.
    stored_patch_texture: Option<egui::TextureHandle>,
    /// Per-observation patch tiles rendered from full-res images, keyed by
    /// image. Rebuilt on point-selection change. Tiles where the patch is
    /// not visible in the view warp to all-black and are drawn as such (a future
    /// N/A flag may distinguish "not visible" from a genuinely dark surface).
    rendered_patch_textures: HashMap<ImageRef, egui::TextureHandle>,
    /// The content_xxh128 hash prefix (first 8 hex chars) for Point IDs.
    hash_prefix: String,
    /// Tracked vertical scroll offset for DM gesture scrolling.
    scroll_offset_y: Option<f32>,
}

/// Response from the Point Track Detail panel.
///
/// Image indices are local to the reconstruction the panel was shown with;
/// `dock.rs` pairs them back into [`ImageRef`]s. A track never spans
/// reconstructions, so every row belongs to the selected point's own recon.
pub struct PointTrackDetailResponse {
    /// If Some, the user clicked a row — select this image.
    pub select_image: Option<usize>,
    /// If Some, the user double-clicked a row — enter camera view for this image.
    pub request_camera_view: Option<usize>,
    /// Image index currently under the pointer (for cross-panel hover).
    pub hovered_image: Option<usize>,
    /// Whether the pointer is currently inside the panel.
    pub has_pointer: bool,
}

/// Height of each thumbnail in the observation table.
const THUMB_SIZE: f32 = 48.0;
/// Display size of the per-observation rendered patch tile (matches the
/// thumbnail so the tile sits flush beside it).
const PATCH_TILE: f32 = THUMB_SIZE;
/// Display size of the stored-patch header tile.
const STORED_PATCH_SIZE: f32 = 64.0;

impl PointTrackDetail {
    pub fn new() -> Self {
        Self {
            prepared_point: None,
            observations: Vec::new(),
            max_angle_deg: 0.0,
            inverse_depth_z: f32::NAN,
            condition_number: f32::NAN,
            thumbnail_textures: HashMap::new(),
            patch_frame: None,
            stored_patch_texture: None,
            rendered_patch_textures: HashMap::new(),
            hash_prefix: String::new(),
            scroll_offset_y: None,
        }
    }

    /// Show the point track detail panel.
    #[allow(clippy::too_many_arguments)]
    pub fn show(
        &mut self,
        ui: &mut egui::Ui,
        recon: &SfmrReconstruction,
        recon_id: ReconId,
        selected_point: Option<usize>,
        hovered_image: Option<usize>,
        sift_cache: &HashMap<ImageRef, CachedSiftFeatures>,
        full_res_cache: &HashMap<ImageRef, Option<ImageU8>>,
        gesture_events: &[GestureEvent],
        scroll_input: &platform::ScrollInput,
    ) -> PointTrackDetailResponse {
        let mut response = PointTrackDetailResponse {
            select_image: None,
            request_camera_view: None,
            hovered_image: None,
            has_pointer: false,
        };

        // Check if pointer is in this panel
        let panel_rect = ui.available_rect_before_wrap();
        if let Some(pos) = ui.input(|i| i.pointer.hover_pos()) {
            if panel_rect.contains(pos) {
                response.has_pointer = true;
            }
        }

        // No point selected — show placeholder
        let Some(point_idx) = selected_point else {
            ui.centered_and_justified(|ui| {
                ui.label("No point selected");
            });
            self.prepared_point = None;
            self.observations.clear();
            return response;
        };

        if point_idx >= recon.points.len() {
            ui.centered_and_justified(|ui| {
                ui.label("No point selected");
            });
            self.prepared_point = None;
            self.observations.clear();
            return response;
        }

        // Prepare observation data if selected point changed
        let point_ref = PointRef::new(recon_id, point_idx);
        if self.prepared_point != Some(point_ref) {
            // Set first: everything below derives the reconstruction its
            // caches belong to from `prepared_point`.
            self.prepared_point = Some(point_ref);
            self.prepare_observations(ui.ctx(), recon, point_ref, sift_cache);
            self.scroll_offset_y = None;
            // Update hash prefix from reconstruction
            let hash = &recon.content_hash.content_xxh128;
            self.hash_prefix = if hash.len() >= 8 {
                hash[..8].to_string()
            } else {
                "00000000".to_string()
            };
        }

        let point = &recon.points[point_idx];

        // --- Header: Point Summary ---
        self.show_header(ui, recon, point_idx, point);

        // --- Stored-patch header tile (embedded-patches reconstructions) ---
        self.show_stored_patch_tile(ui);

        ui.separator();

        // --- Observation Table ---
        self.show_observation_table(
            ui,
            recon,
            recon_id,
            hovered_image,
            full_res_cache,
            gesture_events,
            scroll_input,
            &mut response,
        );

        response
    }

    /// Drop everything cached for a reconstruction that has left the scene.
    pub fn forget_recon(&mut self, id: ReconId) {
        self.thumbnail_textures.retain(|image, _| image.recon != id);
        self.rendered_patch_textures
            .retain(|image, _| image.recon != id);
        if self.prepared_point.is_some_and(|point| point.recon == id) {
            self.clear();
        }
    }

    /// Clear all cached state (e.g. when reconstruction changes).
    pub fn clear(&mut self) {
        self.prepared_point = None;
        self.observations.clear();
        self.max_angle_deg = 0.0;
        self.inverse_depth_z = f32::NAN;
        self.condition_number = f32::NAN;
        self.thumbnail_textures.clear();
        self.patch_frame = None;
        self.stored_patch_texture = None;
        self.rendered_patch_textures.clear();
        self.hash_prefix.clear();
        self.scroll_offset_y = None;
    }
}
