// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The pairwise-flow normal, [`pairwise_flow_normal`]: the bench normal step
//! that renders every observation's photograph onto one plane through the
//! point, computes dense optical flow between the renders of pairs of
//! observations, fits the plane through the point to each pair's flow over
//! discs of several radii, and turns the patch to the combined normal at the
//! largest radius where the pairs agree within their own precision.
//! `specs/core/bench/pairwise-flow-normal.md` is the design.
//!
//! The step runs in rounds, each rendering on the plane the last found. Its
//! pieces each do one part of a round, for one round's plane and one pair:
//!
//! - `crops`: the grid every view of a round shares, the views left out as
//!   too oblique, each remaining view's crop rendered on that grid through
//!   its camera model, the Gaussian pyramid of each crop, and the level a pair
//!   is read at;
//! - `model`: the flow a candidate plane through the point predicts from one
//!   crop to another, by exact ray-plane geometry, and its derivative;
//! - `flow`: the DIS flow between two crops at a pair's level;
//! - `texture`: how much each sample's flow is worth, from the two crops'
//!   structure tensors significant against noise, the photometric residual
//!   and the floor on the flow's precision;
//! - `fit`: the robust weighted fit of one pair's normal and offset over a
//!   disc, and its two σ;
//! - `support`: the pairs combined at each radius, the support rule and a
//!   round's answer;
//! - `rounds`: one round composed from the pieces, and its report;
//! - `report`: the report's public types.

use std::collections::HashSet;

use nalgebra::Vector3;

use crate::features::optical_flow::DisFlowParams;
use crate::patch::normal_refine::ProjectedImage;
use crate::progress::Progress;
use crate::reconstruction::edited::EditedReconstruction;

use super::super::evaluate::{check_observation_views, check_views};
use super::super::fit::FitOptions;
use super::super::track::{EditableTrack, Verdict};
use super::{
    normal_preconditions, patch_frame_of, turn_and_read, NormalError, NormalEstimate, NormalReport,
};

mod crops;
mod fit;
mod flow;
mod model;
mod report;
mod rounds;
mod support;
mod texture;

#[cfg(test)]
mod scene;
#[cfg(test)]
mod tests;

pub use report::{
    PairwiseFlowCrop, PairwiseFlowFit, PairwiseFlowOutcome, PairwiseFlowPair, PairwiseFlowRadius,
    PairwiseFlowRound, PairwiseFlowRounds,
};

/// The most samples along one side of a rendered crop
/// [`PairwiseFlowNormalOptions::max_crop_samples`] takes: a crop of 4096²
/// samples is 16 million per view per round, and a pair read on it at level 0
/// a dense flow of that size, many seconds each.
const MAX_CROP_SAMPLES: u32 = 4096;

/// The most normals the coarse grid each pair's fit starts from may hold,
/// about 42 times the default grid's 97: each is scored on up to 4,096 of a
/// disc's samples, for every pair at every radius.
const MAX_GRID_NORMALS: f64 = 4096.0;

/// How the pairwise-flow normal renders its crops, computes its flows, fits
/// its planes and chooses its support.
///
/// The defaults and where each comes from are in
/// `specs/core/bench/pairwise-flow-normal.md` § "Parameters"; [`Self::check`]
/// says which values the step takes.
#[derive(Debug, Clone)]
pub struct PairwiseFlowNormalOptions {
    /// How many rounds the step runs, each rendering on the plane the last
    /// found, the first on the given normal. A point with exactly two
    /// observations, one pair, runs one round. 3.
    pub rounds: u32,
    /// A view more than this from the round's plane normal, at the point, is
    /// left out of that round, degrees. 80.
    pub max_view_angle_deg: f64,
    /// A normal more than this from the mean viewing direction of the point's
    /// views is no answer, degrees: the views cannot see such a plane. 80.
    pub max_normal_from_mean_view_deg: f64,
    /// The radius whose combined normal, over the pairs with a finite σ, is
    /// the next round's plane when a round has no answer, in patch
    /// half-extents; one of [`Self::radii`]. 1.
    pub fallback_radius: f64,
    /// The grid is one sample per photograph pixel of the finest view, but at
    /// least this many samples along one side of the patch. 24.
    pub min_samples_per_patch: u32,
    /// The most samples along one side of the patch the grid takes. 128.
    pub max_samples_per_patch: u32,
    /// The crop's half-width in the world, in patch half-extents. 3.
    pub crop_half_extents: f64,
    /// The least crop half-width, in photograph px of the point's finest
    /// view. 48.
    pub min_crop_half_width_px: f64,
    /// The most samples along one side of a rendered crop; past it the grid
    /// is coarsened. 512.
    pub max_crop_samples: u32,
    /// The disc radii the planes are fitted over, in patch half-extents,
    /// ascending. `[0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8, 12]`.
    pub radii: Vec<f64>,
    /// The most pairs a round reads. A round with more pairs of rendered
    /// crops reads this many, chosen by a shuffle with a fixed seed, the same
    /// choice for the same number of crops. 10.
    pub max_pairs: usize,
    /// The flow between two crops. `DisFlowParams::high_quality` with
    /// `finest_scale` `Some(0)` and `patch_stride` 1.
    pub flow: DisFlowParams,
    /// The largest tilt of the coarse grid of normals each pair's fit starts
    /// from, degrees. 85.
    pub max_grid_tilt_deg: f64,
    /// The step between the coarse grid's tilts, degrees. 10.
    pub grid_tilt_step_deg: f64,
    /// The step between the coarse grid's azimuths, degrees. 30.
    pub grid_azimuth_step_deg: f64,
    /// Tukey's constant for the pair fit's robust weights, in units of the
    /// whitened residuals' robust scale. 4.685.
    pub tukey_c: f64,
    /// How far apart two flow errors are before they are independent, in
    /// pixels of the pair's coarser view as its crop reads them (one sample
    /// per pixel for a view read through its pyramid); also the side of the
    /// square the crops' structure tensors are summed over. 8.
    pub correlation_length_px: f64,
    /// The share of pure-noise windows whose larger tensor eigenvalue may
    /// pass the significance threshold. 0.001.
    pub noise_false_alarm: f64,
    /// A floor on the precision of one correlated square's flow, in pixels of
    /// the pair's coarser view as its crop reads them: its covariance is the
    /// texture's plus this squared. 0.03.
    pub flow_floor_px: f64,
    /// The largest σ a normal may have and still be a determined answer,
    /// degrees: a pair's fit, a radius's combined normal, or a single pair's.
    /// 10.
    pub max_sigma_deg: f64,
    /// A pair joins the combined normal at a radius when its normal is within
    /// this many times its σ, plus [`Self::agreement_margin_deg`], of the
    /// pairs' medoid. 3.
    pub agreement_sigmas: f64,
    /// The margin added to [`Self::agreement_sigmas`] times a pair's σ,
    /// degrees. 5.
    pub agreement_margin_deg: f64,
    /// The pairs agree at a radius when their mean χ² is at most this. 2.
    pub max_mean_chi2: f64,
    /// The radius a point with exactly two observations is read at, where
    /// there is no agreement to test, in patch half-extents; one of
    /// [`Self::radii`]. 1.
    pub single_pair_radius: f64,
    /// The reading the step ends with, as for the other normal steps.
    pub fit: FitOptions,
}

impl Default for PairwiseFlowNormalOptions {
    fn default() -> Self {
        Self {
            rounds: 3,
            max_view_angle_deg: 80.0,
            max_normal_from_mean_view_deg: 80.0,
            fallback_radius: 1.0,
            min_samples_per_patch: 24,
            max_samples_per_patch: 128,
            crop_half_extents: 3.0,
            min_crop_half_width_px: 48.0,
            max_crop_samples: 512,
            radii: vec![0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0],
            max_pairs: 10,
            flow: flow::default_flow(),
            max_grid_tilt_deg: 85.0,
            grid_tilt_step_deg: 10.0,
            grid_azimuth_step_deg: 30.0,
            tukey_c: 4.685,
            correlation_length_px: 8.0,
            noise_false_alarm: 0.001,
            flow_floor_px: 0.03,
            max_sigma_deg: 10.0,
            agreement_sigmas: 3.0,
            agreement_margin_deg: 5.0,
            max_mean_chi2: 2.0,
            single_pair_radius: 1.0,
            fit: FitOptions::default(),
        }
    }
}

impl PairwiseFlowNormalOptions {
    /// Whether the settings are ones the step takes, naming the first that
    /// fails as [`NormalError::BadPairwiseFlowOption`]: at least one round;
    /// view and normal angles above 0° and at most 90°; finite radii above
    /// zero (`radii` non-empty and strictly ascending, and `fallback_radius`
    /// and `single_pair_radius` each one of its entries);
    /// `min_samples_per_patch` at least 2 and at most `max_samples_per_patch`;
    /// `crop_half_extents` of at least 1; a finite `min_crop_half_width_px` of
    /// at least 0; `max_crop_samples` of at least twice
    /// `min_samples_per_patch` and at most 4,096; `max_pairs` of at least 2,
    /// the fewest whose agreement can be tested; a grid of tilts from 0° to
    /// under 90° in finite steps above 0, at azimuth steps above 0° and at
    /// most 360°, holding at most 4,096 normals (a grid whose tilts alone
    /// pass that is refused as `grid_tilt_step_deg`, any other as
    /// `grid_azimuth_step_deg`); a finite positive `tukey_c`; a
    /// `correlation_length_px` above 0 and at most 16, so the correlated
    /// square is at most 32 samples a side at a pair's level;
    /// `noise_false_alarm` strictly between 0 and 1;
    /// a finite `flow_floor_px` of at least 0; `max_sigma_deg` and
    /// `max_mean_chi2` above 0 (infinite allowed); finite agreement settings
    /// of at least 0; and a read-back fit whose sub-pixel kernel runs at a
    /// patch resolution of at least 2.
    ///
    /// The flow's settings are checked as the flow checks them, except
    /// against an image's size, and refused as [`NormalError::Flow`]. A pair
    /// whose crops at its level are narrower than the flow's `patch_size` is
    /// not refused here: it has no flow and no fits.
    pub fn check(&self) -> Result<(), NormalError> {
        let bad = |name| Err(NormalError::BadPairwiseFlowOption(name));
        let positive = |v: f64| v.is_finite() && v > 0.0;
        let non_negative = |v: f64| v.is_finite() && v >= 0.0;
        let angle = |v: f64| v > 0.0 && v <= 90.0;
        if self.rounds == 0 {
            return bad("rounds");
        }
        if !angle(self.max_view_angle_deg) {
            return bad("max_view_angle_deg");
        }
        if !angle(self.max_normal_from_mean_view_deg) {
            return bad("max_normal_from_mean_view_deg");
        }
        if self.min_samples_per_patch < 2 || self.min_samples_per_patch > self.max_samples_per_patch
        {
            return bad("min_samples_per_patch");
        }
        if !(self.crop_half_extents.is_finite() && self.crop_half_extents >= 1.0) {
            return bad("crop_half_extents");
        }
        if !non_negative(self.min_crop_half_width_px) {
            return bad("min_crop_half_width_px");
        }
        if self.max_crop_samples < 2 * self.min_samples_per_patch
            || self.max_crop_samples > MAX_CROP_SAMPLES
        {
            return bad("max_crop_samples");
        }
        if self.radii.is_empty()
            || !self.radii.iter().all(|&r| positive(r))
            || self.radii.windows(2).any(|w| w[0] >= w[1])
        {
            return bad("radii");
        }
        let listed = |r: f64| self.radii.contains(&r);
        if !(positive(self.fallback_radius) && listed(self.fallback_radius)) {
            return bad("fallback_radius");
        }
        if !(positive(self.single_pair_radius) && listed(self.single_pair_radius)) {
            return bad("single_pair_radius");
        }
        if self.max_pairs < 2 {
            return bad("max_pairs");
        }
        self.flow.check_fields().map_err(NormalError::Flow)?;
        if !(non_negative(self.max_grid_tilt_deg) && self.max_grid_tilt_deg < 90.0) {
            return bad("max_grid_tilt_deg");
        }
        if !positive(self.grid_tilt_step_deg) {
            return bad("grid_tilt_step_deg");
        }
        if !(self.grid_azimuth_step_deg > 0.0 && self.grid_azimuth_step_deg <= 360.0) {
            return bad("grid_azimuth_step_deg");
        }
        // As `fit::grid_normals` lays the grid out.
        let tilts = ((self.max_grid_tilt_deg + 1e-9) / self.grid_tilt_step_deg).floor();
        let azimuths = (360.0 / self.grid_azimuth_step_deg).ceil().max(1.0);
        if 1.0 + tilts > MAX_GRID_NORMALS {
            return bad("grid_tilt_step_deg");
        }
        if 1.0 + tilts * azimuths > MAX_GRID_NORMALS {
            return bad("grid_azimuth_step_deg");
        }
        if !positive(self.tukey_c) {
            return bad("tukey_c");
        }
        if !(positive(self.correlation_length_px) && self.correlation_length_px <= 16.0) {
            return bad("correlation_length_px");
        }
        if !(self.noise_false_alarm > 0.0 && self.noise_false_alarm < 1.0) {
            return bad("noise_false_alarm");
        }
        if !non_negative(self.flow_floor_px) {
            return bad("flow_floor_px");
        }
        if self.max_sigma_deg.is_nan() || self.max_sigma_deg <= 0.0 {
            return bad("max_sigma_deg");
        }
        if !non_negative(self.agreement_sigmas) {
            return bad("agreement_sigmas");
        }
        if !non_negative(self.agreement_margin_deg) {
            return bad("agreement_margin_deg");
        }
        if self.max_mean_chi2.is_nan() || self.max_mean_chi2 <= 0.0 {
            return bad("max_mean_chi2");
        }
        if self.fit.refine.resolution < 2 {
            return bad("resolution");
        }
        Ok(())
    }
}

/// Turn the track's patch to the plane on which dense optical flow between
/// pairs of its observations agrees, at the largest support where it agrees.
///
/// In each round every `in` observation's photograph is rendered onto one
/// plane through the patch's centre, on one grid all the views share; a view
/// more than [`PairwiseFlowNormalOptions::max_view_angle_deg`] from the
/// plane's normal is left out. For every pair of rendered crops, or
/// [`PairwiseFlowNormalOptions::max_pairs`] of them, the DIS flow from one to
/// the other is computed at the pyramid level of the pair's coarser view, and
/// the plane through the point with a free 2D offset is fitted to that flow
/// over a disc at each of [`PairwiseFlowNormalOptions::radii`], with a σ from
/// the crops' texture and one from the fit's residuals. At each radius the
/// determined pairs (σ at most [`PairwiseFlowNormalOptions::max_sigma_deg`])
/// are combined, and the round's answer is the combined normal at the
/// largest radius where at least two are determined and their mean χ² is at
/// most [`PairwiseFlowNormalOptions::max_mean_chi2`].
///
/// The first round renders on the given normal, each later round on the
/// answer of the round before, or where it had none on its fallback, the
/// combination at [`PairwiseFlowNormalOptions::fallback_radius`] over every
/// pair with a finite σ. A point with exactly two observations runs one round
/// and reads its one pair at
/// [`PairwiseFlowNormalOptions::single_pair_radius`]. An answer more than
/// [`PairwiseFlowNormalOptions::max_normal_from_mean_view_deg`] from the mean
/// viewing direction ends the step as [`PairwiseFlowOutcome::Unstable`].
///
/// Only [`PairwiseFlowOutcome::Turned`] and
/// [`PairwiseFlowOutcome::SinglePair`] move the normal; under
/// [`PairwiseFlowOutcome::Undetermined`] and
/// [`PairwiseFlowOutcome::Unstable`] the track is read back at the normal it
/// was given. The step reads no keypoint and moves no keypoint and not the
/// centre, and ends as the other normal steps do: the track is turned with
/// [`tilt_patch`](crate::bench::tilt_patch) and read back, its bitmap
/// rendered.
///
/// Refused, as every normal step refuses them, for a track at the cluster
/// stage, one with no patch, one at infinity, one with fewer than two `in`
/// observations, too few views, and settings
/// [`PairwiseFlowNormalOptions::check`] refuses; an observation naming an
/// image `images` holds no view for is refused as
/// [`EvaluateError::NoView`](crate::bench::EvaluateError::NoView); a flow
/// refusal the check could not foresee is [`NormalError::Flow`]. Cancelled
/// through `progress` between pairs and between radii, as
/// [`NormalError::Cancelled`].
///
/// # Example
///
/// ```no_run
/// # use sfmtool_core::bench::{pairwise_flow_normal, PairwiseFlowNormalOptions};
/// # use sfmtool_core::progress::Progress;
/// # fn run(
/// #     track: &sfmtool_core::bench::EditableTrack,
/// #     edited: &sfmtool_core::EditedReconstruction,
/// #     images: &[sfmtool_core::patch::normal_refine::ProjectedImage<'_>],
/// # ) -> Result<(), Box<dyn std::error::Error>> {
/// let (turned, report) = pairwise_flow_normal(
///     track,
///     edited,
///     images,
///     &PairwiseFlowNormalOptions::default(),
///     &Progress::none(),
/// )?;
/// println!("{report}"); // "3 rounds; 10 of 10 pairs agree at 1.5 half-extents (χ² 1.4), σ 1.83°, …"
/// # let _ = turned;
/// # Ok(())
/// # }
/// ```
pub fn pairwise_flow_normal(
    track: &EditableTrack,
    edited: &EditedReconstruction,
    images: &[ProjectedImage<'_>],
    options: &PairwiseFlowNormalOptions,
    progress: &Progress<'_>,
) -> Result<(EditableTrack, NormalReport), NormalError> {
    check_views(edited, images)?;
    normal_preconditions(track)?;
    check_observation_views(track, images)?;
    options.check()?;
    let patch = patch_frame_of(track);
    let given = patch.normal();
    let mut images_seen = HashSet::new();
    let attempted: Vec<(usize, u32)> = track
        .observations
        .iter()
        .enumerate()
        .filter(|(_, o)| o.verdict == Verdict::In && images_seen.insert(o.image))
        .map(|(k, o)| (k, o.image))
        .collect();
    let views: Vec<ProjectedImage<'_>> = attempted
        .iter()
        .map(|&(_, image)| images[image as usize])
        .collect();
    let point = patch.center;
    let mean_view = {
        let sum = views
            .iter()
            .map(|view| view.cam_from_world.inverse_translation_origin().coords - point.coords)
            .filter(|d| d.norm().is_finite() && d.norm() > 0.0)
            .fold(Vector3::zeros(), |sum, d| sum + d.normalize());
        // Views on opposite sides cancel; the given normal then stands in.
        if sum.norm() > 1e-9 {
            sum.normalize()
        } else {
            given
        }
    };
    let state = rounds::Point {
        point,
        axes: [patch.u_axis, patch.v_axis],
        half_extent: patch.half_extent[0],
        observations: track.observations.len(),
        attempted,
        views,
        mean_view,
        options,
    };
    let count = if state.single_pair() {
        1
    } else {
        options.rounds
    };
    // The plane has no sign; it is rendered facing the views.
    let mut plane = if given.dot(&mean_view) < 0.0 {
        -given
    } else {
        given
    };
    let mut rounds = Vec::new();
    let mut last = None;
    {
        let phase = progress.phase("pairwise flow normal");
        for k in 0..count {
            let round_phase = phase.phase("round");
            let (round, decision) = state.round(plane, k + 1 < count, &round_phase)?;
            rounds.push(round);
            let next = decision.next_plane;
            last = Some(decision);
            match next {
                Some(n) => plane = n,
                None => break,
            }
        }
    }
    let decision = last.expect("at least one round runs");
    // Keep the face the patch showed.
    let answer = decision
        .normal
        .map(|n| if n.dot(&given) < 0.0 { -n } else { n });
    let report = PairwiseFlowRounds {
        rounds,
        outcome: decision.outcome,
        chosen_radius: answer.and(decision.chosen.map(|k| options.radii[k])),
        normal: answer.map(Into::into),
        sigma_deg: answer.and(decision.sigma_deg),
        turn_deg: answer.map(|n| support::angle_deg(given, n)),
    };
    turn_and_read(
        track,
        edited,
        images,
        answer.unwrap_or(given),
        NormalEstimate::PairwiseFlow(Box::new(report)),
        &options.fit,
        progress,
    )
}
