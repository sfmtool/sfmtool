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
//! [`IntrinsicsOptions::min_rotation_mass`] and the weak-vote escalation
//! below; the full [`FocalVoteResult`] comes back untouched for callers that
//! still want the table.
//!
//! It also owns WHEN to pay for the second column. The camera-model columns
//! cost a pair of self-consistency scans per candidate pair, which the
//! closed-form pinhole vote does not run at all, and a capture whose pinhole
//! vote is strong has nothing for the arbitration to overturn. So
//! [`ColumnPolicy::Auto`] screens on the pinhole-only vote and re-runs with
//! both columns exactly when that vote comes back weak
//! ([`escalation_reasons`]) -- adaptive strategy inside the estimator, not a
//! decision each caller re-derives from the vote's diagnostics.

use crate::geometry::focal_vote::{
    focal_vote_with_options, CameraModel, ColumnDiagnostics, FocalVoteOptions, FocalVoteResult,
    ScanVote, VoteFamily, FAMILY_DISAGREEMENT_BAND, ORTHO_GRID_HI, ORTHO_GRID_LO, ORTHO_GRID_N,
};

/// Certified rotation-cell votes an equidistant verdict needs before it counts
/// as confirmed. See [`IntrinsicsOptions::min_rotation_mass`].
const DEFAULT_MIN_ROTATION_MASS: usize = 1;

/// Pooled pinhole votes at or below which the vote is too thin to answer the
/// camera-model question alone. See [`escalation_reasons`].
const THIN_POOL: usize = 9;

/// How [`estimate_intrinsics`] arrives at its camera-model columns.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ColumnPolicy {
    /// Run [`IntrinsicsOptions::vote`]'s own `columns`, always. The caller has
    /// already decided what to ask, so nothing is screened and nothing is
    /// re-run.
    #[default]
    Fixed,
    /// Decide from the evidence: run the pinhole-only vote first and re-run
    /// with both columns only when that vote comes back weak
    /// ([`escalation_reasons`]).
    ///
    /// Auto chooses both column sets itself, so
    /// [`FocalVoteOptions::columns`] is not read under this policy; the vote's
    /// other knobs (seed, epipolar displacement floor) are passed to both runs
    /// unchanged.
    Auto,
}

/// Why a pinhole-only vote is too weak to be taken as the camera-model answer.
///
/// [`EscalationReason::as_str`] gives each one a stable name, and that name is
/// how the reason travels -- into an estimate's `escalation`, out through the
/// Python binding, and into whatever a caller records the trigger as.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EscalationReason {
    /// The vote reached no consensus focal at all.
    NoConsensus,
    /// A rotation-family consensus landed on the bottom rung of the rotation
    /// self-calibration's focal grid -- an answer pinned by the grid's floor
    /// rather than by the evidence.
    RotationRailed,
    /// The two vote families' medians disagree beyond the vote's own
    /// family-disagreement band, so the pinhole answer is contested.
    FamilyDisagreement,
    /// Too few pooled votes to read a camera model off.
    ThinPool,
}

impl EscalationReason {
    /// Stable string name, for the Python binding and for callers that record
    /// the trigger.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::NoConsensus => "no_consensus",
            Self::RotationRailed => "rotation_railed",
            Self::FamilyDisagreement => "family_disagreement",
            Self::ThinPool => "thin_pool",
        }
    }
}

/// Every way the pinhole-only `vote` on a `width` x `height` capture reads as
/// too weak to answer the camera-model question on its own, in check order.
/// Empty means the vote stands and [`ColumnPolicy::Auto`] runs no second pass.
///
/// The disjunction and its four cut points are a fleet measurement, not tuning
/// room: over 40 captures it fires on all 4 fisheye ones and on 9 of the 36
/// rectilinear ones, and every one of those 9 is a genuinely weak vote. What
/// screening avoids is the arbitration error of running the columns
/// unconditionally, which claims 3 of those 36 rectilinear captures as
/// fisheyes.
///
/// - **No consensus.** Fewer than two pooled votes leaves no focal to judge.
/// - **Rotation railed.** A rotation-family consensus within one grid step of
///   `ORTHO_GRID_LO` x the longer image dimension sits on the grid's floor:
///   the scan wanted a shorter focal than a pinhole grid can express, which is
///   what a fisheye capture looks like through a perspective chart.
/// - **Family disagreement.** A gap over the vote's own
///   `FAMILY_DISAGREEMENT_BAND` means the two families are answering different
///   questions on this capture.
/// - **Thin pool.** At most `THIN_POOL` (9) pooled votes is too small a
///   population for its median to settle a model.
pub fn escalation_reasons(
    vote: &FocalVoteResult,
    width: u32,
    height: u32,
) -> Vec<EscalationReason> {
    let max_wh = f64::from(width.max(height));
    // The grid's multiplicative step: `ORTHO_GRID_N` focals spread
    // log-uniformly from `ORTHO_GRID_LO` to `ORTHO_GRID_HI`.
    let step = (ORTHO_GRID_HI / ORTHO_GRID_LO).powf(1.0 / (ORTHO_GRID_N - 1) as f64);
    let mut reasons = Vec::new();
    if vote.focal_px.is_none() {
        reasons.push(EscalationReason::NoConsensus);
    }
    if vote.family == Some(VoteFamily::Rotation)
        && vote
            .rotation_focal_px
            .is_some_and(|f| f <= step * ORTHO_GRID_LO * max_wh)
    {
        reasons.push(EscalationReason::RotationRailed);
    }
    if vote
        .family_disagreement
        .is_some_and(|d| d > FAMILY_DISAGREEMENT_BAND)
    {
        reasons.push(EscalationReason::FamilyDisagreement);
    }
    if vote.n_pool <= THIN_POOL {
        reasons.push(EscalationReason::ThinPool);
    }
    reasons
}

/// Tuning for [`estimate_intrinsics`].
#[derive(Clone, Debug)]
pub struct IntrinsicsOptions {
    /// Passed through to the vote (seed, epipolar displacement floor, and --
    /// under [`ColumnPolicy::Fixed`] -- the camera-model column set).
    pub vote: FocalVoteOptions,
    /// Whether the column set is the one `vote` names, or one the estimator
    /// escalates its way to.
    pub columns: ColumnPolicy,
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
            columns: ColumnPolicy::Fixed,
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
    /// What [`ColumnPolicy::Auto`] decided: the weak-vote reasons that fired,
    /// in check order. Empty means the pinhole-only vote stood on its own, no
    /// second run happened, and `vote` is that pinhole-only result -- no
    /// columns, and a `Pinhole` verdict by construction. `None` under
    /// [`ColumnPolicy::Fixed`], which asks a fixed question and never
    /// escalates.
    pub escalation: Option<Vec<EscalationReason>>,
    /// The pinhole-only vote the escalation decision was read off, kept only
    /// when the estimate then re-ran with both columns. A caller that wants
    /// the capture's PINHOLE numbers -- its consensus, spread and pool -- needs
    /// them from here, because a two-column result reports the winning column
    /// at the top level and that is the fisheye answer whenever the escalation
    /// paid off. `None` when `vote` already is the pinhole-only vote, or under
    /// [`ColumnPolicy::Fixed`].
    pub screening_vote: Option<FocalVoteResult>,
    /// The full vote result behind the verdict, untouched, for diagnostics.
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
/// [`IntrinsicsOptions::columns`] decides how many times the vote runs: once
/// under [`ColumnPolicy::Fixed`], and under [`ColumnPolicy::Auto`] once when
/// the pinhole-only vote stands, twice when it does not.
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
    let run = |columns: Vec<CameraModel>| {
        focal_vote_with_options(
            cluster_indexes,
            image_indexes,
            positions_xy,
            width,
            height,
            &FocalVoteOptions {
                columns,
                ..options.vote.clone()
            },
        )
    };
    let (escalation, screening_vote, vote) = match options.columns {
        ColumnPolicy::Fixed => (None, None, run(options.vote.columns.clone())),
        ColumnPolicy::Auto => {
            let screening = run(vec![CameraModel::Pinhole]);
            let reasons = escalation_reasons(&screening, width, height);
            if reasons.is_empty() {
                // The pinhole vote answered; the scans would have nothing to
                // overturn and are not run at all.
                (Some(reasons), None, screening)
            } else {
                (
                    Some(reasons),
                    Some(screening),
                    run(vec![CameraModel::Pinhole, CameraModel::EquidistantFisheye]),
                )
            }
        }
    };
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
        escalation,
        screening_vote,
        vote,
    }
}

/// One column's diagnostics, if the vote ran that column.
fn column(vote: &FocalVoteResult, model: CameraModel) -> Option<&ColumnDiagnostics> {
    vote.columns.iter().find(|c| c.model == model)
}

#[cfg(test)]
mod tests;
