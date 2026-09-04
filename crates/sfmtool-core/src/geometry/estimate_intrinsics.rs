// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! One typed answer from the structure-free focal vote: the camera model, the
//! focal, and whether the verdict is corroborated
//! ([`estimate_intrinsics`]). See `specs/core/geometry/estimate-intrinsics.md`.
//!
//! [`focal_vote_with_options`] is a diagnostic table -- two camera-model
//! columns, each with its own consensus, spreads, certificate counts and
//! per-pair scans. A caller that wants a camera rather than a table has to read
//! the verdict off that table, and the reading is not obvious: whether an
//! equidistant verdict is corroborated lives in the equidistant column's
//! certified rotation mass, and the result's top-level `epipolar_votes` /
//! `rotation_votes` always describe the pinhole closed-form kernel no matter
//! which column won. Every caller that re-derived those by hand was one
//! copy of the same three rules.
//!
//! This module owns those rules and nothing else. It composes the vote, reads
//! its output, and adds no threshold of its own beyond
//! [`IntrinsicsOptions::min_rotation_mass`]; the full [`FocalVoteResult`] comes
//! back untouched for callers that still want the table.

use crate::geometry::focal_vote::{
    focal_vote_with_options, CameraModel, ColumnDiagnostics, FocalVoteOptions, FocalVoteResult,
    ScanVote,
};

/// Certified rotation-cell votes an equidistant verdict needs before it counts
/// as confirmed. See [`IntrinsicsOptions::min_rotation_mass`].
const DEFAULT_MIN_ROTATION_MASS: usize = 1;

/// Tuning for [`estimate_intrinsics`].
#[derive(Clone, Debug)]
pub struct IntrinsicsOptions {
    /// Passed through to the vote (seed, epipolar displacement floor, the
    /// camera-model column set).
    pub vote: FocalVoteOptions,
    /// Certified rotation-cell votes an equidistant verdict needs in the
    /// equidistant column before it counts as confirmed.
    ///
    /// The default `1` is the structural rule rather than a tuned threshold: a
    /// wrong ray map cannot fake a pure rotation of rays, so any certified
    /// rotation mass at all separates a real fisheye from an arbitration
    /// artifact. Measured over the fleet, false fisheye verdicts carry exactly
    /// zero certified rotation mass while true fisheyes carry 4 to 44, with no
    /// band between.
    ///
    /// It is an option because a caller running a reduced cell set changes that
    /// geometry -- with the epipolar cell absent, a single rotation vote has
    /// been observed to confirm a false verdict -- and such a caller must raise
    /// the floor to what its own population supports.
    pub min_rotation_mass: usize,
}

impl Default for IntrinsicsOptions {
    fn default() -> Self {
        Self {
            // Both columns: the verdict is the point of this API, and a
            // single-column run has nothing to arbitrate.
            vote: FocalVoteOptions {
                columns: vec![CameraModel::Pinhole, CameraModel::EquidistantFisheye],
                ..Default::default()
            },
            min_rotation_mass: DEFAULT_MIN_ROTATION_MASS,
        }
    }
}

/// The camera the vote's evidence supports, with the vote itself attached.
#[derive(Clone, Debug)]
pub struct IntrinsicsEstimate {
    /// The verdict model, `None` when no column produced one.
    pub camera_model: Option<CameraModel>,
    /// Whether a `EquidistantFisheye` verdict is corroborated by certified
    /// rotation mass. `None` when the question does not arise: a `Pinhole`
    /// verdict, no verdict at all, or a single-column run, which arbitrates
    /// nothing.
    pub confirmed: Option<bool>,
    /// The winning column's consensus focal, in pixels.
    pub focal_px: Option<f64>,
    /// The winning column's certified scan votes, in that column's stored vote
    /// order -- the per-pair evidence behind THIS verdict.
    ///
    /// The vote result's own `epipolar_votes` / `rotation_votes` always belong
    /// to the pinhole closed-form kernel regardless of the verdict, so pairing
    /// those with the verdict's scalar fields pairs numbers from two different
    /// columns. Empty when no column ran (the pinhole-only vote runs no scan).
    pub verdict_votes: Vec<ScanVote>,
    /// The full vote result, untouched, for diagnostics.
    pub vote: FocalVoteResult,
}

/// Estimate a camera from cluster-track observations: the model verdict, its
/// corroboration, the consensus focal, and the votes behind them.
/// See `specs/core/geometry/estimate-intrinsics.md`.
///
/// The arguments before `options` are the vote's own
/// ([`focal_vote_with_options`]): `cluster_indexes` nondecreasing, one
/// `image_indexes` entry and one full-pixel `positions_xy` entry per
/// observation, and the shared image size whose centre is the principal point.
/// The function does no I/O and estimates no structure.
///
/// ```no_run
/// use sfmtool_core::geometry::estimate_intrinsics::{estimate_intrinsics, IntrinsicsOptions};
///
/// # let (cluster_indexes, image_indexes, positions_xy) = (vec![], vec![], vec![]);
/// let estimate = estimate_intrinsics(
///     &cluster_indexes,
///     &image_indexes,
///     &positions_xy,
///     640,
///     480,
///     &IntrinsicsOptions::default(),
/// );
/// if estimate.confirmed != Some(false) {
///     println!("{:?} at {:?} px", estimate.camera_model, estimate.focal_px);
/// }
/// ```
pub fn estimate_intrinsics(
    cluster_indexes: &[u32],
    image_indexes: &[u32],
    positions_xy: &[[f64; 2]],
    width: u32,
    height: u32,
    options: &IntrinsicsOptions,
) -> IntrinsicsEstimate {
    let vote = focal_vote_with_options(
        cluster_indexes,
        image_indexes,
        positions_xy,
        width,
        height,
        &options.vote,
    );
    let camera_model = vote.camera_model;

    // A single column is the verdict by construction, so nothing was arbitrated
    // and there is nothing to corroborate. The pinhole-only vote runs no scan at
    // all and reports an empty column list, which is the same case.
    let arbitrated = vote.columns.len() > 1;
    let confirmed =
        (arbitrated && camera_model == Some(CameraModel::EquidistantFisheye)).then(|| {
            column(&vote, CameraModel::EquidistantFisheye)
                .is_some_and(|c| c.n_certified_rotation >= options.min_rotation_mass)
        });

    // The verdict's own evidence: the winning column's certified scans, in the
    // order that column stored them (epipolar cell first, then rotation).
    let verdict_votes = camera_model
        .and_then(|m| column(&vote, m))
        .map(|c| {
            c.scan_votes
                .iter()
                .filter(|v| v.certified)
                .copied()
                .collect()
        })
        .unwrap_or_default();

    IntrinsicsEstimate {
        camera_model,
        confirmed,
        focal_px: vote.focal_px,
        verdict_votes,
        vote,
    }
}

/// One column's diagnostics, if the vote ran that column.
fn column(vote: &FocalVoteResult, model: CameraModel) -> Option<&ColumnDiagnostics> {
    vote.columns.iter().find(|c| c.model == model)
}

#[cfg(test)]
mod tests;
