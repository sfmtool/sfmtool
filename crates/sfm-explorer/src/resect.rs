// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The image menu's `Resect Image` action: what the viewer adds on top of the
//! shared estimate.
//!
//! The estimate itself — the hold-out, the pose fit, the re-triangulation at the
//! new pose and the report that describes them — lives in
//! [`mod@sfmtool_core::geometry::resect_images`], which holds a whole set of
//! images out together; the viewer's action is that primitive on a one-element
//! set, so the GUI and any offline caller resect an image exactly the same way.
//! See `specs/gui/edits/resect-image.md`.
//!
//! The viewer always resects from the tracks and the node's cluster-patches
//! file together, so the action needs that file to be current. What stays here
//! is the viewer's own share: the sentence that says why the file will not do,
//! and the status-line text the outcome is reported in. Landing the answer
//! (reading the file, materialising the current value, pushing the version,
//! following the selection) is [`crate::state::AppState::resect_image`]'s job.

pub use sfmtool_core::geometry::{
    resect_image_in_place, ResectImageOptions, ResectImageReport, ResectSource,
};

use crate::index_files::IndexFileState;
use crate::scene::ReconId;
use crate::state::AppState;

/// What the estimate did, with nothing about where the answer went: `214 pts
/// (150 tracks, 64 clusters), inliers 198/214 (0.93; 140 tracks, 58 clusters),
/// rotation 12.40°, translation 0.081 (scene-scale), 190 re-triangulated;
/// clusters 80 considered, 12 skipped, 4 failed to triangulate`.
///
/// The translation is reported in scene-scale units when the reconstruction has
/// a camera-to-structure distance to divide by, and in its own units when it
/// does not (a rotation-only reconstruction has no such distance).
pub fn outcome_summary(report: &ResectImageReport) -> String {
    let translation = match report.translation_scene {
        Some(scaled) => format!("{scaled:.3} (scene-scale)"),
        None => format!("{:.3}", report.translation),
    };
    format!(
        "{} pts ({} tracks, {} clusters), inliers {}/{} ({:.2}; {} tracks, {} clusters), \
         rotation {:.2}°, translation {translation}, {} re-triangulated; clusters {} \
         considered, {} skipped, {} failed to triangulate",
        report.correspondences,
        report.track_correspondences,
        report.cluster_correspondences,
        report.inliers,
        report.correspondences,
        report.inlier_fraction,
        report.track_inliers,
        report.cluster_inliers,
        report.rotation_deg,
        report.retriangulated,
        report.clusters_considered,
        report.clusters_skipped,
        report.clusters_failed,
    )
}

/// `Resect IMG_0007.jpg in run_a refused: <reason>`.
///
/// Covers every refusal: an estimate that missed its acceptance gate, one that
/// could not be attempted at all, and a node whose cluster-patches file is
/// missing or out of date. None pushes a version.
pub fn failure_message(image: &str, node: &str, reason: &str) -> String {
    format!("Resect {image} in {node} refused: {reason}")
}

/// The last component of a workspace-relative image name — what the version's
/// label is written with and what the status line says.
pub fn basename(name: &str) -> &str {
    name.rsplit(['/', '\\']).next().unwrap_or(name)
}

impl AppState {
    /// Why `Resect Image` cannot read `id`'s cluster-patches file, or `None`
    /// when the file is current.
    ///
    /// Reads the states as they stand; the caller refreshes them first where a
    /// version may have moved them. The sentence names the file's state and
    /// ends with what makes a current one: the build entry under its present
    /// name, or why the node cannot have index files yet.
    pub(crate) fn resect_cluster_patches_refusal(&self, id: ReconId) -> Option<String> {
        let state = match self.cluster_patches_state(id) {
            IndexFileState::Current => return None,
            IndexFileState::None => "none is open.".to_string(),
            IndexFileState::Stale => {
                let why = self
                    .cluster_patches(id)
                    .and_then(|file| file.stale_reason())
                    .unwrap_or_default();
                format!("the one open is out of date. {why}")
            }
        };
        let remedy = match self.index_files_home_refusal(id) {
            Some(why) => why,
            None => format!(
                "{} (the Index Files row in the Scene tree) makes it.",
                self.index_files_build_label(id)
            ),
        };
        Some(format!(
            "Resect Image reads this reconstruction's cluster patches file, and {state} {remedy}"
        ))
    }
}

#[cfg(test)]
pub(crate) mod tests;
