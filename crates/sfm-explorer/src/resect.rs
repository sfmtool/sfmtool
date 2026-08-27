// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The Scene Graph panel's `Resect Image` action: what the viewer adds on top
//! of the shared estimate.
//!
//! The estimate itself — the hold-out, the pose fit, the re-triangulation at the
//! new pose and the report that describes them — lives in
//! [`mod@sfmtool_core::geometry::resect_images`], which holds a whole set of
//! images out together; the panel's action is that primitive on a one-element
//! set, so the GUI and any offline caller resect an image exactly the same way.
//! See `specs/gui/gui-resect-image.md`.
//!
//! What stays here is the viewer's own share: which correspondence source the
//! menu asked for, and the status-line text the outcome is reported in. Making
//! the derived node — name, inherited frame, selection, the marker on the
//! resected row — is [`crate::state::AppState::resect_image`]'s job.

pub use sfmtool_core::geometry::{
    resect_images, ResectImageOptions, ResectImageReport, ResectSource,
};

/// Which of the two menu entries was chosen.
///
/// The panel reports the *choice* rather than the parsed `.matches` file: the
/// file is picked, read and cached one layer up, where the file dialog and the
/// per-node memory of the last path live.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResectFrom {
    /// `Resect Image` — the target's own observations.
    Observations,
    /// `Resect Image from Matches…` — the match graph of a chosen `.matches`
    /// file.
    Matches,
}

/// `Resected IMG_0007.jpg in run_a: 214 pts, inliers 198/214 (0.93), rotation
/// 12.40°, translation 0.081 (scene-scale), 190 re-triangulated`.
///
/// The translation is reported in scene-scale units when the reconstruction has
/// a camera-to-structure distance to divide by, and in its own units when it
/// does not (a rotation-only reconstruction has no such distance).
pub fn success_message(image: &str, node: &str, report: &ResectImageReport) -> String {
    let translation = match report.translation_scene {
        Some(scaled) => format!("{scaled:.3} (scene-scale)"),
        None => format!("{:.3}", report.translation),
    };
    format!(
        "Resected {image} in {node}: {} pts, inliers {}/{} ({:.2}), rotation {:.2}°, \
         translation {translation}, {} re-triangulated",
        report.correspondences,
        report.inliers,
        report.correspondences,
        report.inlier_fraction,
        report.rotation_deg,
        report.retriangulated,
    )
}

/// `Resect IMG_0007.jpg in run_a refused: <reason>`.
///
/// Covers both refusals: an estimate that missed its acceptance gate (which
/// still produces a node, showing the held-out re-triangulation on its own) and
/// one that could not be attempted at all (which does not).
pub fn failure_message(image: &str, node: &str, reason: &str) -> String {
    format!("Resect {image} in {node} refused: {reason}")
}

/// The last component of a workspace-relative image name — what the derived
/// node is named after and what the status line says.
pub fn basename(name: &str) -> &str {
    name.rsplit(['/', '\\']).next().unwrap_or(name)
}
