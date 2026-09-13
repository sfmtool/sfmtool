// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Builds the panel's data model for a newly selected point.
//!
//! Everything the table draws is precomputed once here — on selection change,
//! not per frame — into [`super::TrackObservationData`]. This is the seam
//! between the reconstruction and the display code: it reads observations,
//! keypoints and SIFT features, delegates the numbers to [`crate::metrics`] and
//! the patch state to [`super::patch`], and leaves the rendering modules with
//! nothing to compute.

use std::collections::HashMap;
use std::path::Path;

use sfmtool_core::EditedReconstruction;

use super::patch::{build_patch_frame, build_stored_patch_texture};
use super::{PointTrackDetail, TrackObservationData};
use crate::metrics::{
    compute_max_pairwise_angle, compute_observation_metrics, compute_point_diagnostics,
};
use crate::scene::{ImageRef, PointRef};
use crate::state::CachedSiftFeatures;

impl PointTrackDetail {
    /// Prepare observation data for a newly selected point.
    pub(super) fn prepare_observations(
        &mut self,
        ctx: &egui::Context,
        edited: &EditedReconstruction,
        point: PointRef,
        sift_cache: &HashMap<ImageRef, CachedSiftFeatures>,
    ) {
        let point_idx = point.index();
        self.observations.clear();
        self.thumbnail_textures.clear();
        // The images and cameras are the base's; the point is the overlay's.
        let recon = &*edited.base;
        let Some(view) = edited.point(point_idx as u32) else {
            self.patch_frame = None;
            self.stored_patch_texture = None;
            self.rendered_patch_textures.clear();
            return;
        };

        // Per-point patch state (embedded-patches reconstructions): the
        // oriented patch frame gates the per-observation "Patch" column, the
        // stored bitmap feeds the header tile. Rendered tiles rebuild lazily.
        self.patch_frame = build_patch_frame(&view);
        self.stored_patch_texture = build_stored_patch_texture(ctx, &view, point_idx);
        self.rendered_patch_textures.clear();

        let point3d = view.point();
        // Keypoints come from one of two sources: SIFT feature positions read
        // into the cache (`sift_files`, via `feature_indexes`) or keypoints
        // stored inline on the reconstruction (`embedded_patches`, via
        // `keypoints_xy`, indexed per observation). For embedded keypoints the
        // affine shape (and hence size) is derived by projecting the point's
        // patch frame into the view (`observation_affine_shape`).
        let feature_indexes = view.feature_indexes();
        let observations = view.observations();

        // Collect world-space rays from each camera center to the point
        // for max-angle computation.
        let mut world_rays: Vec<[f64; 3]> = Vec::with_capacity(observations.len());

        for (k, obs) in observations.iter().enumerate() {
            let img_idx = obs.image_index as usize;
            let image = &recon.image_table.images[img_idx];
            let camera = &recon.image_table.cameras[image.camera_index as usize];

            // Feature index (SIFT), position, and extents for this observation.
            let (feature_index, feature_xy, feature_extents) = if let Some(fis) = feature_indexes {
                let feat_idx = fis[k] as usize;
                let cached_sift = sift_cache.get(&ImageRef::new(point.recon, img_idx));
                let xy = cached_sift
                    .and_then(|sift| sift.positions_xy.get(feat_idx))
                    .copied()
                    .unwrap_or([0.0, 0.0]);
                let extents = cached_sift
                    .and_then(|sift| sift.affine_shapes.get(feat_idx))
                    .map(affine_full_extents)
                    .unwrap_or([0.0, 0.0]);
                (feat_idx, xy, extents)
            } else if let Some(xy) = view.keypoint_xy(k) {
                // Embedded keypoint: no SIFT feature index, so report the
                // observation's place in the track. The affine shape (and hence
                // the extents) is derived by projecting the point's patch frame
                // into this image, through the overlay so that an edited frame
                // or position is the one projected.
                let extents = edited
                    .observation_affine_shape(point_idx as u32, img_idx, xy)
                    .map(|a| affine_full_extents(&a))
                    .unwrap_or([0.0, 0.0]);
                (k, xy, extents)
            } else {
                (0, [0.0, 0.0], [0.0, 0.0])
            };

            // --- Compute per-observation reprojection error and ray angle ---
            let (reproj_error, ray_angle_deg) =
                compute_observation_metrics(point3d, image, camera, feature_xy);

            // Collect world-space ray for max-angle computation. Every camera
            // sees a point at infinity along the same stored unit direction,
            // so its max pairwise angle is honestly zero.
            if point3d.is_at_infinity() {
                let d = point3d.position.coords;
                world_rays.push([d.x, d.y, d.z]);
            } else {
                let dir = point3d.position - image.camera_center();
                let len = (dir.x * dir.x + dir.y * dir.y + dir.z * dir.z).sqrt();
                if len > 1e-12 {
                    world_rays.push([dir.x / len, dir.y / len, dir.z / len]);
                }
            }

            let image_full_name = image.name.clone();
            let image_name = truncated_path_suffix(&image_full_name);

            self.observations.push(TrackObservationData {
                image_index: img_idx,
                feature_index,
                feature_xy,
                reproj_error,
                ray_angle_deg,
                feature_extents,
                image_name,
                image_full_name,
            });
        }

        // Sort by image index (should already be sorted, but ensure it)
        self.observations.sort_by_key(|o| o.image_index);

        // Compute max angle between any pair of observation rays.
        self.max_angle_deg = compute_max_pairwise_angle(&world_rays);

        // Triangulation observability diagnostics for this point.
        let (condition_number, inverse_depth_z) =
            compute_point_diagnostics(&recon.image_table, &view);
        self.condition_number = condition_number;
        self.inverse_depth_z = inverse_depth_z;
    }
}

/// The two full extents in pixels of an affine shape matrix, ordered larger
/// first.
///
/// The matrix columns are the projected patch *half*-vectors, so each column
/// norm is a semi-axis; doubling turns them into the full widths the rendered
/// quad actually spans (`±u ±v`), which is the diameter convention the rest of
/// the toolkit uses for patch sizes. Both keypoint sources are measured this
/// way, so the two branches above stay comparable.
fn affine_full_extents(a: &[[f32; 2]; 2]) -> [f32; 2] {
    let col0 = (a[0][0] * a[0][0] + a[1][0] * a[1][0]).sqrt();
    let col1 = (a[0][1] * a[0][1] + a[1][1] * a[1][1]).sqrt();
    [2.0 * col0.max(col1), 2.0 * col0.min(col1)]
}

/// Return a short display name from an image path, keeping the filename plus
/// its parent directory so that rig images sharing the same filename are
/// distinguishable. For example `images/fisheye_left/image_0345.jpg` becomes
/// `…/fisheye_left/image_0345.jpg`, and `images/image_0345.jpg` stays
/// `images/image_0345.jpg`, since nothing was left out of it. Plain filenames
/// without a parent are returned as-is.
fn truncated_path_suffix(path_str: &str) -> String {
    let p = Path::new(path_str);
    let file_name = match p.file_name() {
        Some(f) => f.to_string_lossy(),
        None => return path_str.to_string(),
    };
    let Some(parent) = p.parent() else {
        return file_name.into_owned();
    };
    let Some(parent_dir) = parent.file_name() else {
        return file_name.into_owned();
    };
    // The mark stands for an ancestor that was left out, so it is only earned
    // when there is one. `images/a.jpg` is the whole of what the file stores,
    // and writing it `\u{2026}/images/a.jpg` claims a directory above `images` that
    // nothing knows about, in a column that has no room to spare.
    let elided = parent
        .parent()
        .is_some_and(|above| above.file_name().is_some());
    let mark = if elided { "\u{2026}/" } else { "" };
    format!("{mark}{}/{}", parent_dir.to_string_lossy(), file_name)
}
