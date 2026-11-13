// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The building blocks of the pairwise-flow normal, the bench normal step
//! proposed in `specs/drafts/pairwise-flow-normal.md`, and its settings,
//! [`PairwiseFlowNormalOptions`].
//!
//! The step renders every observation's photograph onto one plane through the
//! point, computes dense optical flow between the renders of every pair of
//! observations, and fits the plane through the point to each pair's flow
//! over discs of several radii. The pieces here each do one part of that, for
//! one round's plane and one pair:
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
//!   disc, and its two σ.
//!
//! The step itself, `pairwise_flow_normal`, composes them over rounds and
//! radii and chooses the support; it is a later change. Until it lands only
//! these modules' tests call the pieces, so dead code is allowed outside the
//! tests.

#![cfg_attr(not(test), allow(dead_code))]

use crate::features::optical_flow::DisFlowParams;

use super::super::fit::FitOptions;
use super::NormalError;

mod crops;
mod fit;
mod flow;
mod model;
mod texture;

#[cfg(test)]
mod scene;
#[cfg(test)]
mod tests;

/// How the pairwise-flow normal renders its crops, computes its flows, fits
/// its planes and chooses its support.
///
/// The defaults and where each comes from are in
/// `specs/drafts/pairwise-flow-normal.md` § "Parameters"; [`Self::check`]
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
    /// half-extents. 1.
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
    /// there is no agreement to test, in patch half-extents. 1.
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
    /// zero (`fallback_radius`, `single_pair_radius`, and `radii`, which is
    /// also non-empty and strictly ascending); `min_samples_per_patch` at
    /// least 2 and at most `max_samples_per_patch`; `crop_half_extents` of at
    /// least 1; a finite `min_crop_half_width_px` of at least 0;
    /// `max_crop_samples` of at least twice `min_samples_per_patch`; a grid
    /// of tilts from 0° to under 90° in finite steps above 0, at azimuth
    /// steps above 0° and at most 360°; finite positive `tukey_c` and
    /// `correlation_length_px`; `noise_false_alarm` strictly between 0 and 1;
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
        if !positive(self.fallback_radius) {
            return bad("fallback_radius");
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
        if self.max_crop_samples < 2 * self.min_samples_per_patch {
            return bad("max_crop_samples");
        }
        if self.radii.is_empty()
            || !self.radii.iter().all(|&r| positive(r))
            || self.radii.windows(2).any(|w| w[0] >= w[1])
        {
            return bad("radii");
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
        if !positive(self.tukey_c) {
            return bad("tukey_c");
        }
        if !positive(self.correlation_length_px) {
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
        if !positive(self.single_pair_radius) {
            return bad("single_pair_radius");
        }
        if self.fit.refine.resolution < 2 {
            return bad("resolution");
        }
        Ok(())
    }
}
