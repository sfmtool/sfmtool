// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Builds the panel's data model for a newly selected point.
//!
//! Everything the table draws is precomputed once here — on selection change,
//! not per frame — into [`super::TrackObservationData`]. This is the seam
//! between the reconstruction and the display code: it reads observations,
//! keypoints and SIFT features, delegates the numbers to [`super::metrics`] and
//! the patch state to [`super::patch`], and leaves the rendering modules with
//! nothing to compute.

use std::collections::HashMap;
use std::path::Path;

use sfmtool_core::SfmrReconstruction;

use super::metrics::{
    compute_max_pairwise_angle, compute_observation_metrics, compute_point_diagnostics,
};
use super::patch::{build_patch_frame, build_stored_patch_texture};
use super::{PointTrackDetail, TrackObservationData};
use crate::scene::{ImageRef, PointRef};
use crate::state::CachedSiftFeatures;

impl PointTrackDetail {
    /// Prepare observation data for a newly selected point.
    pub(super) fn prepare_observations(
        &mut self,
        ctx: &egui::Context,
        recon: &SfmrReconstruction,
        point: PointRef,
        sift_cache: &HashMap<ImageRef, CachedSiftFeatures>,
    ) {
        let point_idx = point.index();
        self.observations.clear();
        self.thumbnail_textures.clear();

        // Per-point patch state (embedded-patches reconstructions): the
        // oriented patch frame gates the per-observation "Patch" column, the
        // stored bitmap feeds the header tile. Rendered tiles rebuild lazily.
        self.patch_frame = build_patch_frame(recon, point_idx);
        self.stored_patch_texture = build_stored_patch_texture(ctx, recon, point_idx);
        self.rendered_patch_textures.clear();

        let point_pos = recon.points[point_idx].position;
        // Keypoints come from one of two sources: SIFT feature positions read
        // into the cache (`sift_files`, via `feature_indexes`) or keypoints
        // stored inline on the reconstruction (`embedded_patches`, via
        // `keypoints_xy`, indexed per observation). For embedded keypoints the
        // affine shape (and hence size) is derived by projecting the point's
        // patch frame into the view (`observation_affine_shape`).
        let feature_indexes = recon.feature_indexes();
        let keypoints_xy = recon.keypoints_xy();
        let obs_start = recon.observation_offsets[point_idx];
        let observations = recon.observations_for_point(point_idx);

        // Collect world-space rays from each camera center to the point
        // for max-angle computation.
        let mut world_rays: Vec<[f64; 3]> = Vec::with_capacity(observations.len());

        for (k, obs) in observations.iter().enumerate() {
            let img_idx = obs.image_index as usize;
            let obs_global = obs_start + k;
            let image = &recon.images[img_idx];
            let camera = &recon.cameras[image.camera_index as usize];

            // Feature index (SIFT), position, and extents for this observation.
            let (feature_index, feature_xy, feature_extents) = if let Some(fis) = feature_indexes {
                let feat_idx = fis[obs_global] as usize;
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
            } else if let Some(kxy) = keypoints_xy {
                // Embedded keypoint: no SIFT feature index, so report the
                // observation index. The affine shape (and hence the extents) is
                // derived by projecting the point's patch frame into this image.
                let xy = [kxy[[obs_global, 0]], kxy[[obs_global, 1]]];
                let extents = recon
                    .observation_affine_shape(point_idx, img_idx, xy)
                    .map(|a| affine_full_extents(&a))
                    .unwrap_or([0.0, 0.0]);
                (obs_global, xy, extents)
            } else {
                (0, [0.0, 0.0], [0.0, 0.0])
            };

            // --- Compute per-observation reprojection error and ray angle ---
            let (reproj_error, ray_angle_deg) =
                compute_observation_metrics(&point_pos, image, camera, feature_xy);

            // Collect world-space ray for max-angle computation
            let cam_center = image.camera_center();
            let dir = point_pos - cam_center;
            let len = (dir.x * dir.x + dir.y * dir.y + dir.z * dir.z).sqrt();
            if len > 1e-12 {
                world_rays.push([dir.x / len, dir.y / len, dir.z / len]);
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
        let (condition_number, inverse_depth_z) = compute_point_diagnostics(recon, point_idx);
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
/// `…/fisheye_left/image_0345.jpg`. Plain filenames without a parent are
/// returned as-is.
fn truncated_path_suffix(path_str: &str) -> String {
    let p = Path::new(path_str);
    let file_name = match p.file_name() {
        Some(f) => f.to_string_lossy(),
        None => return path_str.to_string(),
    };
    match p.parent().and_then(|par| par.file_name()) {
        Some(parent_dir) => format!("\u{2026}/{}/{}", parent_dir.to_string_lossy(), file_name),
        None => file_name.into_owned(),
    }
}
