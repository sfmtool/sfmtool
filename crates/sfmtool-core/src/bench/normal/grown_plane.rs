// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! [`grown_plane_normal`]: the patch's normal from one plane shared by every
//! view, fitted directly to the photographs, near to far within each view,
//! and the views' planes combined by their information.
//!
//! The reference view's working grid is the template. For every other `in`
//! view the plane, carried onto the patch that view sees (the patch slid onto
//! its keypoint, `OrientedPatch::anchored_at_keypoint`), predicts where each
//! grid sample lands in that view's photograph (`render`); the step fits the
//! plane's tilt and the view's image shift so that the view's samples agree
//! with the reference's by ZNCC. The step trusts the keypoints it is given:
//! within a view it starts a small window at the grid's centre at the view's
//! keypoint, where its slid patch's centre projects, and
//! never searches for another shift, so a view whose keypoint is off by more
//! than the growth recovers is left out. Fitting the keypoints is a separate
//! step the caller runs. The growth widens the window outward, fitting tilt
//! and shift by Levenberg–Marquardt (`solve`) at each size, and ends with a
//! fit of the whole grid on the robust cell score. With the turned support
//! (the default), every fitted stage lays the working grid and the template
//! again on the patch square turned to the plane the growth has reached
//! ([`Grid::laid_at`](render::Grid::laid_at)), may turn the plane only a cap
//! from there, and the whole-grid fit repeats as rounds whose cap halves when
//! a round reverses the one before. A growth stops early,
//! refused, at a window stage whose score shows it will not reach the floor.
//! Each growth gives the view's own plane and its information about the tilt.
//! A view with no keypoint, or whose anchor does not project into it, has no
//! shift to start from and is left out without a growth.
//!
//! A selection of views, picked near to far by their angle from the
//! reference, is grown against the reference, the first from the starting
//! plane and each later one from the plane of the views kept before it,
//! and again from the starting plane where a second-rung pick is refused
//! there; their planes are combined by a robust information-weighted mean,
//! with no fit of the views together
//! (`combine`); the widest pairs of kept views are then matched against each
//! other and their planes added to the mean (`selected`). The patch is turned
//! to the combined plane whenever a view joined, unless the turn is large and
//! the evidence for it weak, when the starting normal is kept; on a turn,
//! each view not picked has its shift alone fitted at that plane, over a
//! mid-size window and then briefly over the whole grid. The shifts are the
//! tilt fit's own unknowns and are reported, never written: the turn is made
//! about the patch centre, so every keypoint stays as it was given.
//!
//! `specs/core/bench/grown-plane-normal.md` specifies the step.

use std::cell::RefCell;
use std::collections::BTreeSet;
use std::rc::Rc;

use nalgebra::{Matrix2, Vector3};

use crate::patch::keypoint_localize::resolve_reference;
use crate::patch::normal_refine::ProjectedImage;
use crate::patch::reference_view::ReferenceViewOptions;
use crate::progress::Progress;
use crate::progress_note;
use crate::reconstruction::edited::EditedReconstruction;

use super::super::evaluate::{
    check_observation_views, check_views, seed_of, stored_patch_resolution, track_reference,
};
use super::super::fit::FitOptions;
use super::super::steps::tilt_degrees;
use super::super::track::EditableTrack;
use super::grid_plane::{working_side, MIN_WORKING_RESOLUTION};
use super::{
    normal_preconditions, patch_frame_of, turn_and_read, NormalError, NormalEstimate, NormalReport,
};

mod combine;
mod render;
mod selected;
mod solve;

use combine::{sigmas_deg_at, Estimate};
use render::{anchored, Grid, Samples, ViewImage, ViewRender};
use solve::{
    angle_deg, carried_information, disc_of, live_cells, maximise, normal_matrix, schur_tilt,
    score_and_normal_matrix, shift_response, Iteration, Reading, Rounds, ViewFit,
};

/// A growth stops, refused ([`GrownPlaneRefusal::StoppedEarly`]), at a window
/// stage from [`EARLY_REFUSAL_FROM_STAGE`] on whose ZNCC is under
/// [`GrownPlaneNormalOptions::min_held_zncc`] less this. On the ground-truth
/// sweep, 24 of 5545 growths that passed the floor had such a window stage,
/// and 496 of 1562 refused growths stop at one.
const EARLY_REFUSAL_MARGIN: f64 = 0.25;

/// The first stage, counted in [`GrownPlaneAttempt::stage_zncc`] from the
/// first window's unfitted score at 0, that [`EARLY_REFUSAL_MARGIN`] applies
/// to: the second window fitted. The first fitted window's score is too often
/// low on a growth that passes.
const EARLY_REFUSAL_FROM_STAGE: usize = 2;

/// The window a shift-only fit fits the shift over first, as an index into
/// the stages' half-widths: the second fitted window's, `1/5` of the grid's
/// side under the default fractions, or the widest on a grid with fewer
/// windows.
const SHIFT_ONLY_WINDOW: usize = 2;

/// The Levenberg–Marquardt iterations of the whole-grid fit that follows a
/// shift-only fit's window fit.
const SHIFT_ONLY_GRID_ITERATIONS: u32 = 2;

/// How [`grown_plane_normal`] grows its windows, fits its planes and decides
/// which views it keeps.
#[derive(Debug, Clone)]
pub struct GrownPlaneNormalOptions {
    /// Side of the working grid the views are sampled on, in samples. `None`
    /// follows the reference view's footprint of the patch at the starting
    /// patch frame, about one sample per photograph pixel, clamped to `R ..= 2R`,
    /// the rule [`grid_plane_normal`](super::grid_plane_normal) uses. At least
    /// [`Self::spot_window`] either way.
    pub working_resolution: Option<u32>,
    /// Side of the first window, grid px, centred on the grid's centre
    /// sample. 7.
    pub spot_window: u32,
    /// Half-widths of the later windows about the same sample, as fractions
    /// of the working grid's side, each rounded to grid px; one no wider than
    /// the first window's half-width is dropped, and `1/2` covers the whole
    /// grid. `[1/8, 1/5, 1/3, 1/2]`, which is 6, 10, 16, 24 on a 48 grid.
    pub window_fractions: Vec<f64>,
    /// The ZNCC difference noise can produce, `τ`: an estimate's information
    /// about the tilt becomes the covariance `2τ·I⁻¹`. 0.01.
    pub zncc_tolerance: f64,
    /// An estimate agrees when the Mahalanobis distance between its own plane
    /// and the combined plane, the two covariances added, is at most this. 3.
    pub max_agreement_distance: f64,
    /// The floor every whole-grid score is held to: a view's own score, its
    /// score at the combined plane, a pair's score between its two views and
    /// a shift-only view's score at the final plane. A growth stops early at
    /// a window stage more than 0.25 under it. 0.5.
    pub min_held_zncc: f64,
    /// A weakest-direction `σ` above this, degrees, is reported as low
    /// confidence. It does not by itself stop the turn. 3.67.
    pub low_confidence_sigma_deg: f64,
    /// The smallest turn, degrees, that must be backed by evidence: a turn
    /// from the starting normal to the combined plane of at least this, with
    /// its ratio to the weakest-direction `σ` under
    /// [`Self::min_turn_sigma_ratio`], is not made, and the step keeps the
    /// starting normal ([`GrownPlaneOutcome::WeakTurn`]). Infinite never
    /// keeps it. 70.
    pub weak_turn_deg: f64,
    /// The ratio of the turn to the combined plane's weakest-direction `σ`
    /// ([`GrownPlanePrecision::sigma_weak_deg`], every kept estimate's
    /// information summed, no pair halved) that a turn of at least
    /// [`Self::weak_turn_deg`] needs to be made. A `σ` that is not finite
    /// counts as no evidence, a ratio of 0. 0 never keeps the start. 3.
    pub min_turn_sigma_ratio: f64,
    /// The most pairs of kept views matched against each other. 3.
    pub max_pairs: usize,
    /// The selection stops matching views once the combined plane's weakest
    /// `σ` is at most this, degrees. 2.75.
    pub target_sigma_deg: f64,
    /// The selection stops matching views once this many have passed the
    /// floor. 6.
    pub max_selected_views: usize,
    /// The selection's first rung, degrees from the reference: its first
    /// pick is the view nearest this angle within [`Self::first_rung_min_deg`]
    /// to [`Self::first_rung_max_deg`]. 15.
    pub first_rung_deg: f64,
    /// The narrowest angle of the selection's first rung, degrees. 10.
    pub first_rung_min_deg: f64,
    /// The widest angle of the selection's first rung, degrees. 20.
    pub first_rung_max_deg: f64,
    /// Each later rung of the selection aims at this many times the widest
    /// angle kept so far, and takes views within a factor of its square root
    /// of that aim. 1.6.
    pub rung_ratio: f64,
    /// A ZNCC is read only when at least this share of its samples lands on
    /// the photograph. 0.6.
    pub min_coverage: f64,
    /// The most Levenberg–Marquardt iterations one stage takes. 20.
    pub max_iterations: u32,
    /// The damping a stage's fit starts from. 1e-3.
    pub initial_damping: f64,
    /// A stage stops once an iteration raises its score by less than this.
    /// 1e-6.
    pub min_improvement: f64,
    /// Lay the working grid and the template again, at the start of every
    /// fitted window stage and every whole-grid round, on the starting square
    /// turned to the plane the growth has reached, cap each stage's turn,
    /// repeat the whole-grid stage as rounds, and read every whole grid at a
    /// plane on the grid laid there. `false` is the step with one grid laid on
    /// the starting square, no cap and one whole-grid stage, and then the
    /// three settings below are not read. `true`.
    pub turned_support: bool,
    /// The most a window stage, and the first whole-grid round, may turn the
    /// plane from where the stage began, degrees: above 0 and under 90, or
    /// infinite for no cap. 12.
    pub stage_turn_cap_deg: f64,
    /// The whole-grid rounds stop once their cap would fall under this, and
    /// after a round that turned less than half of it, degrees: finite, above
    /// 0 and at most [`Self::stage_turn_cap_deg`]. 0.5.
    pub min_stage_turn_cap_deg: f64,
    /// The most whole-grid rounds a growth or a pair runs; at least 1. 16.
    pub max_grid_rounds: u32,
    /// The reading the step ends with, and the reference-view rule's grazing
    /// gate and sampler (`localize`).
    pub fit: FitOptions,
}

impl Default for GrownPlaneNormalOptions {
    fn default() -> Self {
        Self {
            working_resolution: None,
            spot_window: 7,
            window_fractions: vec![1.0 / 8.0, 1.0 / 5.0, 1.0 / 3.0, 1.0 / 2.0],
            zncc_tolerance: 0.01,
            max_agreement_distance: 3.0,
            min_held_zncc: 0.5,
            low_confidence_sigma_deg: 3.67,
            weak_turn_deg: 70.0,
            min_turn_sigma_ratio: 3.0,
            max_pairs: 3,
            target_sigma_deg: 2.75,
            max_selected_views: 6,
            first_rung_deg: 15.0,
            first_rung_min_deg: 10.0,
            first_rung_max_deg: 20.0,
            rung_ratio: 1.6,
            min_coverage: 0.6,
            max_iterations: 20,
            initial_damping: 1e-3,
            min_improvement: 1e-6,
            turned_support: true,
            stage_turn_cap_deg: 12.0,
            min_stage_turn_cap_deg: 0.5,
            max_grid_rounds: 16,
            fit: FitOptions::default(),
        }
    }
}

impl GrownPlaneNormalOptions {
    /// Whether the settings are ones the step takes: a first window of at
    /// least one sample, finite fractions above zero, a positive tolerance
    /// and target `σ`, a weak-turn bound of at least zero (infinite allowed)
    /// and a finite turn-to-`σ` ratio of at least zero, at least one selected
    /// view, a first rung of finite angles of at least zero with its aim
    /// within its bounds, a finite rung ratio above 1, a coverage in `0 ..= 1`,
    /// a finite damping of at least zero, a stage turn cap above 0 and under
    /// 90 or infinite, a finite floor on it above 0 and at most the cap, at
    /// least one whole-grid round, and a fit whose sub-pixel kernel runs at a
    /// patch resolution of at least 2. The three settings of the turned
    /// support are checked with it off too. The error names the first setting
    /// that fails.
    pub fn check(&self) -> Result<(), NormalError> {
        let bad = |name| Err(NormalError::BadGrownPlaneOption(name));
        let positive = |v: f64| v.is_finite() && v > 0.0;
        if self.spot_window == 0 {
            return bad("spot_window");
        }
        if !self.window_fractions.iter().all(|&f| positive(f)) {
            return bad("window_fractions");
        }
        if !positive(self.zncc_tolerance) {
            return bad("zncc_tolerance");
        }
        if !positive(self.target_sigma_deg) {
            return bad("target_sigma_deg");
        }
        if self.weak_turn_deg.is_nan() || self.weak_turn_deg < 0.0 {
            return bad("weak_turn_deg");
        }
        if !(self.min_turn_sigma_ratio.is_finite() && self.min_turn_sigma_ratio >= 0.0) {
            return bad("min_turn_sigma_ratio");
        }
        if self.max_selected_views == 0 {
            return bad("max_selected_views");
        }
        let angle = |v: f64| v.is_finite() && v >= 0.0;
        if !(angle(self.first_rung_min_deg) && angle(self.first_rung_max_deg))
            || self.first_rung_min_deg > self.first_rung_max_deg
        {
            return bad("first_rung_min_deg");
        }
        if !(self.first_rung_min_deg..=self.first_rung_max_deg).contains(&self.first_rung_deg) {
            return bad("first_rung_deg");
        }
        if !(self.rung_ratio.is_finite() && self.rung_ratio > 1.0) {
            return bad("rung_ratio");
        }
        if !(0.0..=1.0).contains(&self.min_coverage) {
            return bad("min_coverage");
        }
        if !(self.initial_damping.is_finite() && self.initial_damping >= 0.0) {
            return bad("initial_damping");
        }
        let cap = self.stage_turn_cap_deg;
        if !(cap > 0.0 && (cap < 90.0 || cap == f64::INFINITY)) {
            return bad("stage_turn_cap_deg");
        }
        if !positive(self.min_stage_turn_cap_deg) || self.min_stage_turn_cap_deg > cap {
            return bad("min_stage_turn_cap_deg");
        }
        if self.max_grid_rounds == 0 {
            return bad("max_grid_rounds");
        }
        if self.fit.refine.resolution < 2 {
            return bad("resolution");
        }
        Ok(())
    }

    fn iteration(&self) -> Iteration {
        Iteration {
            max_iterations: self.max_iterations,
            damping: self.initial_damping,
            min_improvement: self.min_improvement,
            disc: None,
        }
    }
}

/// How [`grown_plane_normal`] ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GrownPlaneOutcome {
    /// A view joined and the patch was turned to the combined plane.
    Turned,
    /// No view joined, or no reference was found: there is no plane, and the
    /// normal and the keypoints are left as given.
    NoneJoined,
    /// A view joined, but the turn to the combined plane is at least
    /// [`GrownPlaneNormalOptions::weak_turn_deg`] and under
    /// [`GrownPlaneNormalOptions::min_turn_sigma_ratio`] times its weakest
    /// `σ`: a large turn on weak evidence. The step keeps the starting normal
    /// and the keypoints as given, and reports the plane it found.
    WeakTurn,
}

/// Which plane a growth's windows started from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GrownPlaneTiltStart {
    /// The starting plane, `(a, b) = (0, 0)`.
    Starting,
    /// The combined plane of the views the selection kept before it.
    Combined,
}

/// Why a view's growth, a view or a pair was not kept.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GrownPlaneRefusal {
    /// Its own robust grid score (a pair's: the score between its two views)
    /// is under [`GrownPlaneNormalOptions::min_held_zncc`].
    UnderFloor,
    /// Its own plane is further than
    /// [`GrownPlaneNormalOptions::max_agreement_distance`] from the combined
    /// plane.
    TooFar,
    /// At the combined plane, at its shift moved there, its robust grid score
    /// is under [`GrownPlaneNormalOptions::min_held_zncc`].
    HeldUnderFloor,
    /// A window stage of a growth, from the second window fitted on, scored
    /// under [`GrownPlaneNormalOptions::min_held_zncc`] less 0.25, and the
    /// growth stopped there with no whole-grid stage.
    StoppedEarly,
    /// A pair one of whose views was left out.
    ViewLeftOut,
    /// The view has no keypoint, or the anchor does not project into it, so
    /// there is no keypoint-implied shift to grow it from. It is not grown,
    /// picked or fitted for its shift alone.
    NoKeypoint,
}

/// One growth of one view against the reference, and how it was judged.
#[derive(Debug, Clone, PartialEq)]
pub struct GrownPlaneAttempt {
    /// Which plane the windows started from.
    pub tilt_start: GrownPlaneTiltStart,
    /// Each stage's score: the first window's ZNCC at the keypoint-implied
    /// shift, each later window's ZNCC after its fit, and the whole grid's
    /// robust score after the last fit, the last round's.
    /// A growth that stopped early ([`GrownPlaneRefusal::StoppedEarly`])
    /// ends at the window stage it stopped at, with no whole-grid score.
    pub stage_zncc: Vec<f64>,
    /// The shift the growth ended at, image px.
    pub shift: [f64; 2],
    /// The outward normal of the plane the growth ended at, world frame.
    pub normal: [f64; 3],
    /// The robust grid score there; `NaN` for a growth that stopped early.
    /// With [`GrownPlaneNormalOptions::turned_support`] it is read on the
    /// grid laid at that plane, and so can differ slightly from the last
    /// entry of `stage_zncc`; with it off the two are equal.
    pub grid_zncc: f64,
    /// `σ` along its own plane's weakest direction, degrees, from its own
    /// information at its own plane and shift, in that plane's own angular
    /// frame; `NaN` for a growth that stopped early.
    pub sigma_deg: f64,
    /// Why it was refused, [`GrownPlaneRefusal::UnderFloor`] or
    /// [`GrownPlaneRefusal::StoppedEarly`]; `None` when it was taken. Whether a
    /// taken one is kept is the combination's verdict, in
    /// [`GrownPlaneSequence::views`].
    pub refusal: Option<GrownPlaneRefusal>,
    /// The angle each fitted window stage turned the plane, degrees: one per
    /// window stage after the first, in order, as far as the growth ran.
    pub stage_turn_deg: Vec<f64>,
    /// Each whole-grid round, in order: empty for a growth that stopped
    /// early, one round with [`GrownPlaneNormalOptions::turned_support`] off.
    pub grid_rounds: Vec<GrownPlaneGridRound>,
}

/// One whole-grid round of a growth or a pair.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GrownPlaneGridRound {
    /// The cap it was fitted under, degrees; infinite with no cap.
    pub cap_deg: f64,
    /// The angle from the plane it began at to the plane it ended at,
    /// degrees.
    pub turn_deg: f64,
    /// Its robust score there, read on its own grid.
    pub zncc: f64,
}

/// What became of one try of a view.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GrownPlaneTryOutcome {
    /// A growth passed the floor on its own, and its plane goes to the
    /// combination, which decides whether it is kept.
    Matched,
    /// Every growth was refused, and the view was left out.
    LeftOut,
}

/// One try of one view, and the combined plane after it.
#[derive(Debug, Clone, PartialEq)]
pub struct GrownPlaneTry {
    /// The observation tried.
    pub observation: usize,
    /// Its image.
    pub image: u32,
    /// Its angle from the reference as seen from the patch centre, degrees.
    pub angle_deg: f64,
    /// Its growths, in the order made, each from the keypoint-implied shift:
    /// from the combined plane first where a view had been kept before it;
    /// then, where none had been kept or a pick of the second rung was
    /// refused there, from the starting plane. One or two growths.
    pub attempts: Vec<GrownPlaneAttempt>,
    pub outcome: GrownPlaneTryOutcome,
    /// The combined plane's outward normal after the try, world frame: the
    /// plane of the views picked so far, this one among them. `None` for a
    /// try that left the view out.
    pub combined_normal: Option<[f64; 3]>,
    /// `σ` along that plane's weakest direction after the try, degrees, in
    /// that plane's own angular frame;
    /// `NaN` where `combined_normal` is `None`.
    pub combined_sigma_deg: f64,
}

/// One pair of kept views matched against each other.
#[derive(Debug, Clone, PartialEq)]
pub struct GrownPlanePair {
    /// The two observations, the first rendered at its held shift and the
    /// second at its held shift plus `relative_shift`.
    pub observations: [usize; 2],
    pub images: [u32; 2],
    /// The angle between the two views as seen from the patch centre,
    /// degrees.
    pub angle_deg: f64,
    /// The outward normal of the pair's own plane, world frame.
    pub normal: [f64; 3],
    /// The second view's shift against the first's, beyond the two views'
    /// held shifts, image px.
    pub relative_shift: [f64; 2],
    /// The robust score between the two views' renders at the pair's plane,
    /// on the grid laid there with
    /// [`GrownPlaneNormalOptions::turned_support`].
    pub zncc: f64,
    /// `σ` along the pair's plane's weakest direction, degrees, in that
    /// plane's own angular frame.
    pub sigma_deg: f64,
    /// The Mahalanobis distance of its plane from the combined plane; `None`
    /// for a pair refused under the floor, which was never combined.
    pub distance: Option<f64>,
    /// Why it was not kept; `None` when its plane is in the combined plane.
    pub refusal: Option<GrownPlaneRefusal>,
    /// Each whole-grid round of its fit, in order; one with
    /// [`GrownPlaneNormalOptions::turned_support`] off.
    pub grid_rounds: Vec<GrownPlaneGridRound>,
}

/// One view the selection picked, and why.
#[derive(Debug, Clone, PartialEq)]
pub struct GrownPlanePick {
    pub observation: usize,
    pub image: u32,
    /// Its angle from the reference, degrees.
    pub angle_deg: f64,
    /// The unit direction of its offset from the reference in the starting
    /// patch frame's `(u, v)`, along which its predicted information lies.
    pub offset_direction: [f64; 2],
    /// Its photograph px per world unit at the anchor, the square root of
    /// the projection's area scale there.
    pub scale_px: f64,
    /// The smallest eigenvalue of the information predicted with it added to
    /// that measured so far, in the measured information's units; `NaN` for
    /// a pick of the first rung, made by angle alone.
    pub predicted_weak: f64,
    /// Whether its growth passed the floor.
    pub matched: bool,
    /// The selection's rung it was picked for, the first `0`, counting every
    /// rung the selection aimed at, those with no view in range among them.
    pub rung: u32,
    /// The combined plane's weakest `σ` after it, degrees; `NaN` when it was
    /// not matched.
    pub sigma_after_deg: f64,
}

/// What the step made of one observation of the track.
#[derive(Debug, Clone, PartialEq)]
pub enum GrownPlaneView {
    /// The reference.
    Reference,
    /// Its own plane is in the final plane. `distance` and `held_zncc` are
    /// its kept growth's plane's distance from the final plane and its score
    /// there. `shift` is its growth's image shift, px, from where its slid
    /// patch's centre projects (its keypoint where the slide followed it),
    /// moved to the final plane by its own normal matrix, and `grid_zncc` its
    /// robust grid score there, the same value as `held_zncc`: the
    /// combination's last held check read it at the final plane and that
    /// shift. Where the turn was not made, `shift` is its growth's own
    /// shift, not moved. The shift is one of the tilt fit's own unknowns and
    /// is not written to the keypoint.
    Joined {
        shift: [f64; 2],
        grid_zncc: f64,
        distance: Option<f64>,
        held_zncc: Option<f64>,
    },
    /// Not picked by the selection: its shift alone was fitted at the final
    /// plane from its keypoint-implied shift, over a mid-size window and then
    /// for two iterations over the whole grid, and it is not in the plane.
    /// `grid_zncc` is its score there, which
    /// [`GrownPlaneNormalOptions::min_held_zncc`] is the floor for. Its
    /// keypoint is not written.
    ShiftOnly { shift: [f64; 2], grid_zncc: f64 },
    /// Left out, with the refusal, distance and held score that left it out:
    /// [`GrownPlaneRefusal::NoKeypoint`] for a view with no keypoint-implied
    /// shift, under every outcome.
    LeftOut {
        reason: GrownPlaneRefusal,
        distance: Option<f64>,
        held_zncc: Option<f64>,
    },
    /// Not tried: not `in`, a second row in an image already tried, no
    /// reference was found, or not picked by a selection that joined no view
    /// or whose turn was not made ([`GrownPlaneOutcome::WeakTurn`]).
    NotAttempted,
}

/// How precisely the kept estimates determine the plane.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GrownPlanePrecision {
    /// `σ` along the plane's weakest and strongest directions, degrees, from
    /// every kept estimate's information summed, each read at its own plane,
    /// expressed in the final plane's own angular frame: turns of its normal,
    /// in radians, about two orthonormal axes in the plane.
    pub sigma_weak_deg: f64,
    pub sigma_strong_deg: f64,
    /// The same with each kept pair's information halved, so that a view
    /// matched against the reference and in a pair counts about once.
    /// Equal to the two above with no pair kept.
    pub discounted_sigma_weak_deg: f64,
    pub discounted_sigma_strong_deg: f64,
    /// `discounted_sigma_weak_deg` is over
    /// [`GrownPlaneNormalOptions::low_confidence_sigma_deg`]. It does not by
    /// itself decide the turn: that is the rule of
    /// [`GrownPlaneNormalOptions::weak_turn_deg`], on `sigma_weak_deg`.
    pub low_confidence: bool,
}

/// What [`grown_plane_normal`] did, carried by [`NormalEstimate::GrownPlane`].
#[derive(Debug, Clone, PartialEq)]
pub struct GrownPlaneSequence {
    /// The reference observation; `None` when the reference-view rule found
    /// none, and then nothing was tried.
    pub reference: Option<usize>,
    /// The side of the working grid, samples; `0` with no reference.
    pub working_resolution: u32,
    /// The half-widths of the stages' windows, grid px, the first window's
    /// first.
    pub window_half_widths: Vec<u32>,
    /// One entry per observation of the track.
    pub views: Vec<GrownPlaneView>,
    /// Every try, in the order made.
    pub trace: Vec<GrownPlaneTry>,
    /// The selection's picks, in the order made.
    pub picks: Vec<GrownPlanePick>,
    /// The pairs matched against each other, widest first.
    pub pairs: Vec<GrownPlanePair>,
    /// How the step ended: turned, no view joined, or a large turn on weak
    /// evidence not made.
    pub outcome: GrownPlaneOutcome,
    /// The final plane's outward normal, world frame: the plane the kept
    /// estimates were combined on, which the patch was turned to when
    /// `outcome` is [`GrownPlaneOutcome::Turned`] and not when it is
    /// [`GrownPlaneOutcome::WeakTurn`]; `None` when no view joined.
    pub normal: Option<[f64; 3]>,
    /// The turn from the starting normal to `normal`, degrees: the
    /// [`TiltReport::degrees`](crate::bench::TiltReport::degrees) the turn
    /// gives, or would have given where it was not made. `None` when no view
    /// joined.
    pub turn_deg: Option<f64>,
    /// `None` when no view joined.
    pub precision: Option<GrownPlanePrecision>,
}

impl GrownPlaneSequence {
    /// How many views joined.
    pub fn joined(&self) -> usize {
        self.count(|v| matches!(v, GrownPlaneView::Joined { .. }))
    }

    /// How many views were left out.
    pub fn left_out(&self) -> usize {
        self.count(|v| matches!(v, GrownPlaneView::LeftOut { .. }))
    }

    /// How many views had their shift alone fitted.
    pub fn shift_only(&self) -> usize {
        self.count(|v| matches!(v, GrownPlaneView::ShiftOnly { .. }))
    }

    /// How many views were grown against the reference: every view with a
    /// try.
    pub fn matched(&self) -> usize {
        self.trace
            .iter()
            .map(|t| t.observation)
            .collect::<BTreeSet<_>>()
            .len()
    }

    /// How many pairs were kept.
    pub fn pairs_kept(&self) -> usize {
        self.pairs.iter().filter(|p| p.refusal.is_none()).count()
    }

    /// How well the views agree at the final plane: the mean `grid_zncc` of
    /// the views that joined it ([`GrownPlaneView::Joined`]) or had their
    /// shift fitted at it ([`GrownPlaneView::ShiftOnly`]), each a robust grid
    /// score against the reference, with a `NaN` score left out. The
    /// reference has no score of its own and does not count. `NaN` when no
    /// view has a score, as when no view joined.
    ///
    /// Under [`GrownPlaneOutcome::WeakTurn`] it is the joined views' mean at
    /// the plane found and not turned to, and no view has its shift fitted.
    pub fn agreement(&self) -> f64 {
        let (sum, n) = self
            .views
            .iter()
            .filter_map(|v| match v {
                GrownPlaneView::Joined { grid_zncc, .. }
                | GrownPlaneView::ShiftOnly { grid_zncc, .. } => Some(*grid_zncc),
                _ => None,
            })
            .filter(|z| !z.is_nan())
            .fold((0.0, 0usize), |(sum, n), z| (sum + z, n + 1));
        if n == 0 {
            f64::NAN
        } else {
            sum / n as f64
        }
    }

    /// Whether the step turned the patch: some view joined, and the turn
    /// was not a large one on weak evidence.
    pub fn turned(&self) -> bool {
        self.outcome == GrownPlaneOutcome::Turned
    }

    /// The ratio of the turn to the combined plane's weakest-direction `σ`,
    /// the quantity [`GrownPlaneNormalOptions::min_turn_sigma_ratio`] bounds:
    /// 0 where that `σ` is not finite. `None` when no view joined.
    pub fn turn_sigma_ratio(&self) -> Option<f64> {
        let turn = self.turn_deg?;
        let sigma = self.precision?.sigma_weak_deg;
        Some(turn_sigma_ratio(turn, sigma))
    }

    fn count(&self, f: impl Fn(&GrownPlaneView) -> bool) -> usize {
        self.views.iter().filter(|v| f(v)).count()
    }
}

impl std::fmt::Display for GrownPlaneSequence {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (joined, left) = (self.joined(), self.left_out());
        write!(f, "{joined} of {} views joined", joined + left)?;
        if left > 0 {
            write!(f, ", {left} left out")?;
        }
        let shift_only = self.shift_only();
        if shift_only > 0 {
            write!(f, ", {shift_only} shift only")?;
        }
        if !self.pairs.is_empty() {
            write!(
                f,
                ", {} of {} pairs kept",
                self.pairs_kept(),
                self.pairs.len()
            )?;
        }
        match &self.precision {
            Some(p) => {
                write!(
                    f,
                    "; \u{3c3} {:.2}\u{b0} weakest, {:.2}\u{b0} strongest",
                    p.sigma_weak_deg, p.sigma_strong_deg
                )?;
                if p.discounted_sigma_weak_deg != p.sigma_weak_deg {
                    write!(
                        f,
                        " ({:.2}\u{b0} weakest with each view once)",
                        p.discounted_sigma_weak_deg
                    )?;
                }
                if p.low_confidence {
                    write!(f, ", low confidence")?;
                }
                if self.outcome == GrownPlaneOutcome::WeakTurn {
                    write!(
                        f,
                        "; normal left unchanged: large turn on weak evidence, \
                         {:.1}\u{b0} at {:.1}\u{3c3}",
                        self.turn_deg.unwrap_or(f64::NAN),
                        self.turn_sigma_ratio().unwrap_or(f64::NAN)
                    )
                } else {
                    Ok(())
                }
            }
            None => write!(f, ", normal left unchanged"),
        }
    }
}

/// Turn the track's patch to the one plane its views agree on, fitted
/// directly to the photographs.
///
/// The reference is the fit's: the reference the track holds where its row is
/// `in` with a keypoint, else the reference-view rule's pick among the `in`
/// rows, one row per image. Its working grid is the starting patch frame re-anchored
/// on its keypoint, [`GrownPlaneNormalOptions::working_resolution`] samples a
/// side. The plane is the starting normal tilted by `(a, b)` along the patch frame's
/// axes through that anchor. Each other view reads the same tilted patch slid
/// onto its own keypoint, centred where its keypoint's ray meets the plane
/// through the patch centre facing that camera, which no tilt moves, and adds
/// an image shift, zero where the growth starts.
///
/// A view is grown from a plane, the starting plane or the combined plane
/// of the views kept before it: a 7×7 window at the grid's centre starts from
/// the shift the view's keypoint implies, the window grows by
/// [`GrownPlaneNormalOptions::window_fractions`] with tilt and shift fitted
/// at each size, and a last fit takes the whole grid on the robust cell
/// score. A growth stops, refused, at a window stage from the second fitted
/// on that scores under [`GrownPlaneNormalOptions::min_held_zncc`] less 0.25.
/// Each growth gives the view's own plane and its information about the
/// tilt, the Schur complement of its normal matrix.
///
/// With [`GrownPlaneNormalOptions::turned_support`], every fitted window
/// stage and every whole-grid round starts on the working grid laid again on
/// the patch square turned to the plane the growth has reached, with the
/// reference's template rendered there, and fits the tilt in that grid's own
/// frame within [`GrownPlaneNormalOptions::stage_turn_cap_deg`] of where it
/// began; the whole-grid stage repeats as rounds whose cap halves when a
/// round turns back against the one before, until the cap would fall under
/// [`GrownPlaneNormalOptions::min_stage_turn_cap_deg`], a round turns less
/// than half of that, or [`GrownPlaneNormalOptions::max_grid_rounds`] have
/// run. Every whole grid read at a plane (a growth's own score and
/// information, the held scores, the pairs and the shift-only fits) is read
/// on the grid laid at that plane, and each estimate is carried into the
/// starting frame's tilt before it is combined. Without it the grid is laid
/// once on the starting square.
///
/// The step trusts the keypoints it is given and searches for no other
/// shift: a view whose keypoint is off by more than the growing windows
/// recover is refused and left out. A caller whose keypoints may be off fits
/// them first with a separate step, such as [`fit()`](crate::bench::fit()). A
/// view with no keypoint, or whose anchor does not project into it, has no
/// shift to start from; it is left out as
/// [`GrownPlaneRefusal::NoKeypoint`] and never grown.
///
/// The views grown are a selection, picked one at a time near to far in
/// rungs of angle from the reference: the view nearest
/// [`GrownPlaneNormalOptions::first_rung_deg`] within
/// [`GrownPlaneNormalOptions::first_rung_min_deg`] to
/// [`GrownPlaneNormalOptions::first_rung_max_deg`], grown from the starting
/// plane, then rungs each aiming at
/// [`GrownPlaneNormalOptions::rung_ratio`] times the widest angle kept, out
/// to the widest views, within a rung the view whose predicted information
/// most raises the weakest information measured so far. Each later pick is
/// grown from the combined plane of the views kept so far, from its
/// keypoint-implied shift; a pick of the second rung refused there is grown
/// again from the starting plane, at the same shift, and a pick of a later
/// rung is not. A rung
/// waits until one of its views is kept. The selection stops when the combined plane's weakest `σ`
/// reaches
/// [`GrownPlaneNormalOptions::target_sigma_deg`] or
/// [`GrownPlaneNormalOptions::max_selected_views`] views have passed the
/// floor. Their planes are combined by a robust information-weighted mean,
/// never by fitting the views together: an estimate more than
/// [`GrownPlaneNormalOptions::max_agreement_distance`] from the mean by the
/// Mahalanobis distance is left out, worst first, the mean read again, and
/// every kept view must still score the floor at the final plane at its shift
/// moved there. Then up to
/// [`GrownPlaneNormalOptions::max_pairs`] of the widest pairs of kept views
/// are matched against each other and their planes added to the mean. Every
/// view not picked has its shift alone fitted at the final plane from its
/// keypoint-implied shift: over the second fitted window's samples, then for
/// two iterations over the whole grid.
///
/// When at least one view joined, the turn to the final plane is judged: a
/// turn of at least [`GrownPlaneNormalOptions::weak_turn_deg`] under
/// [`GrownPlaneNormalOptions::min_turn_sigma_ratio`] times the combined
/// plane's weakest `σ` is a large turn on weak evidence and is not made
/// ([`GrownPlaneOutcome::WeakTurn`]): the normal and the keypoints are left
/// as given, the views not picked have no shift fitted, the track is read
/// back, and the report gives the plane found, its precision, the views,
/// picks and pairs. Otherwise the patch is turned to the final plane with
/// [`tilt_patch`](crate::bench::tilt_patch), and the track is read back with
/// its bitmap rendered; the report gives the plane's precision and says when
/// it is low. When none joined, the normal is left as given and the track is
/// read back.
///
/// **The step writes no keypoint.** The turn is made about the patch centre,
/// which does not move, and keeps each keypoint where it was given; the
/// shifts the fit solved for stay in the report. A caller who wants the
/// keypoints relocalized against the turned patch runs a fit afterwards.
///
/// A track at infinity is refused, as by every normal step: its patch faces
/// along its own bearing and has no plane to fit. A track with an observation
/// naming an image `images` holds no view for is refused as
/// [`EvaluateError::NoView`](crate::bench::EvaluateError::NoView) before any
/// view is read.
///
/// The step opens the phase `grown plane normal` on `progress`, and under it
/// `grown plane view` for each try. A detailed `progress` adds `setup`,
/// `selection` for each pick, `mean` for each combination of planes, `pairs`
/// with a `pair fit` per pair, `shift-only fit` for each view not picked,
/// and `turn` under the step; `growth` under each try; and
/// `window stage` per later window, `whole-grid stage` per round and `judge`
/// under each growth, each
/// noted with what it reached. The
/// result is the same with detail on or off.
///
/// # Example
///
/// ```no_run
/// # use sfmtool_core::bench::{grown_plane_normal, GrownPlaneNormalOptions};
/// # use sfmtool_core::progress::Progress;
/// # fn run(
/// #     track: &sfmtool_core::bench::EditableTrack,
/// #     edited: &sfmtool_core::EditedReconstruction,
/// #     images: &[sfmtool_core::patch::normal_refine::ProjectedImage<'_>],
/// # ) -> Result<(), Box<dyn std::error::Error>> {
/// let (turned, report) = grown_plane_normal(
///     track,
///     edited,
///     images,
///     &GrownPlaneNormalOptions::default(),
///     &Progress::none(),
/// )?;
/// println!("{report}"); // "6 of 7 views joined, 1 left out, 10 shift only, 3 of 3 pairs kept; σ …"
/// # let _ = turned;
/// # Ok(())
/// # }
/// ```
pub fn grown_plane_normal(
    track: &EditableTrack,
    edited: &EditedReconstruction,
    images: &[ProjectedImage<'_>],
    options: &GrownPlaneNormalOptions,
    progress: &Progress<'_>,
) -> Result<(EditableTrack, NormalReport), NormalError> {
    check_views(edited, images)?;
    normal_preconditions(track)?;
    check_observation_views(track, images)?;
    options.check()?;
    let step = progress.phase("grown plane normal");
    let progress: &Progress<'_> = &step;
    let mut setup = progress.detail_phase("setup");
    let patch_frame = patch_frame_of(track);
    let base = stored_patch_resolution(&edited.base)
        .unwrap_or(options.fit.refine.resolution)
        .max(MIN_WORKING_RESOLUTION);

    // One row per image, as the other normal steps read them.
    let mut seen = BTreeSet::new();
    let rows: Vec<usize> = track
        .in_observations()
        .into_iter()
        .filter(|&i| seen.insert(track.observations[i].image))
        .collect();
    let patch_views: Vec<u32> = rows.iter().map(|&i| track.observations[i].image).collect();
    let seeds: Vec<Option<[f64; 2]>> = rows
        .iter()
        .map(|&i| seed_of(&track.observations[i]))
        .collect();
    let reference = resolve_reference(
        &patch_frame,
        images,
        &patch_views,
        &seeds,
        track_reference(track, &rows),
        options.fit.localize.min_grazing_cos,
        base,
        options.fit.localize.sampler,
        &ReferenceViewOptions::default(),
        &setup,
    )
    .map(|k| rows[k]);

    let mut views = vec![GrownPlaneView::NotAttempted; track.observations.len()];
    let Some(reference) = reference else {
        progress_note!(setup, "no reference");
        drop(setup);
        let turn = progress.detail_phase("turn");
        let sequence = GrownPlaneSequence {
            reference: None,
            working_resolution: 0,
            window_half_widths: Vec::new(),
            views,
            trace: Vec::new(),
            picks: Vec::new(),
            pairs: Vec::new(),
            outcome: GrownPlaneOutcome::NoneJoined,
            normal: None,
            turn_deg: None,
            precision: None,
        };
        return turn_and_read(
            track,
            edited,
            images,
            patch_frame.normal(),
            NormalEstimate::GrownPlane(Box::new(sequence)),
            &options.fit,
            &turn,
        );
    };

    let ref_observation = &track.observations[reference];
    let ref_view = &images[ref_observation.image as usize];
    let ref_keypoint = seed_of(ref_observation);
    let floor = MIN_WORKING_RESOLUTION.max(options.spot_window);
    let side = match options.working_resolution {
        Some(side) => side.max(floor),
        None => {
            let anchored_patch_frame = anchored(&patch_frame, ref_view, ref_keypoint);
            let jacobian = crate::camera::warp_map::patch_grid_jacobian(
                &anchored_patch_frame,
                ref_view.camera,
                ref_view.cam_from_world,
                base as usize,
            );
            working_side(jacobian, base).max(floor)
        }
    } as usize;
    let grid = Grid::new(&patch_frame, ref_view, ref_keypoint, side);
    let ref_image = ViewImage::new(ref_view, &grid.anchored_patch_frame, side, Vector3::zeros());
    let start = Rc::new(Laid::new(grid, &ref_image));
    let grid = &start.grid;
    let radii = window_radii(side, options);

    // The candidates: every other row with a keypoint-implied shift, nearest
    // the reference in angle first, ties by row. A row with none is left out
    // here, never grown.
    let centre = patch_frame.center.coords;
    let ref_direction =
        (ref_view.cam_from_world.inverse_translation_origin().coords - centre).try_normalize(1e-12);
    let mut candidates: Vec<Candidate<'_>> = rows
        .iter()
        .filter(|&&row| row != reference)
        .filter_map(|&row| {
            let observation = &track.observations[row];
            let view = &images[observation.image as usize];
            let keypoint = seed_of(observation);
            // The patch this view sees: the starting patch slid onto its
            // keypoint, read through the same tilts about its own centre.
            let seen = anchored(&patch_frame, view, keypoint);
            let image = ViewImage::new(view, &seen, side, seen.center.coords - grid.anchor);
            let Some(kp_shift) = keypoint.and_then(|kp| {
                grid.anchor_pixel(&image)
                    .map(|at| [kp[0] - at[0], kp[1] - at[1]])
            }) else {
                views[row] = GrownPlaneView::LeftOut {
                    reason: GrownPlaneRefusal::NoKeypoint,
                    distance: None,
                    held_zncc: None,
                };
                return None;
            };
            let direction = (view.cam_from_world.inverse_translation_origin().coords - centre)
                .try_normalize(1e-12);
            let angle_deg = match (ref_direction, direction) {
                (Some(a), Some(b)) => a.dot(&b).clamp(-1.0, 1.0).acos().to_degrees(),
                _ => f64::INFINITY,
            };
            let offset_direction = match (ref_direction, direction) {
                (Some(a), Some(b)) => {
                    let d = b - a;
                    let (x, y) = (d.dot(&patch_frame.u_axis), d.dot(&patch_frame.v_axis));
                    let n = x.hypot(y);
                    if n > 1e-12 {
                        [x / n, y / n]
                    } else {
                        [0.0; 2]
                    }
                }
                _ => [0.0; 2],
            };
            let scale_px = grid.scale_px(&image);
            Some(Candidate {
                row,
                image_index: observation.image,
                image,
                kp_shift,
                angle_deg,
                direction,
                offset_direction,
                scale_px,
            })
        })
        .collect();
    candidates.sort_by(|a, b| a.angle_deg.total_cmp(&b.angle_deg));
    progress_note!(
        setup,
        "grid {side}, {} other views, {} with no keypoint, {} windows",
        candidates.len(),
        rows.len() - 1 - candidates.len(),
        radii.len()
    );
    drop(setup);

    let state = Sequence {
        options,
        iteration: options.iteration(),
        start: Rc::clone(&start),
        reference: &ref_image,
        radii: &radii,
        candidates: &candidates,
        last_laid: RefCell::new(None),
    };
    let mut outcome = state.run(progress)?;

    let mut sequence = GrownPlaneSequence {
        reference: Some(reference),
        working_resolution: side as u32,
        window_half_widths: radii.iter().map(|&r| r as u32).collect(),
        views,
        trace: outcome.trace,
        picks: outcome.picks,
        pairs: outcome.pairs,
        outcome: GrownPlaneOutcome::NoneJoined,
        normal: None,
        turn_deg: None,
        precision: None,
    };
    let Some(tilt) = outcome.tilt else {
        record_views(&mut sequence.views, reference, &candidates, &outcome.fates);
        let mut turn = progress.detail_phase("turn");
        progress_note!(turn, "no view joined, normal left unchanged");
        return turn_and_read(
            track,
            edited,
            images,
            patch_frame.normal(),
            NormalEstimate::GrownPlane(Box::new(sequence)),
            &options.fit,
            &turn,
        );
    };

    // σ at the final plane, in its own angular frame.
    let tau = options.zncc_tolerance;
    let [weak, strong] = sigmas_deg_at(&outcome.information, &grid.start, tilt, tau);
    let [d_weak, d_strong] = sigmas_deg_at(&outcome.discounted, &grid.start, tilt, tau);
    sequence.precision = Some(GrownPlanePrecision {
        sigma_weak_deg: weak,
        sigma_strong_deg: strong,
        discounted_sigma_weak_deg: d_weak,
        discounted_sigma_strong_deg: d_strong,
        low_confidence: d_weak.is_nan() || d_weak > options.low_confidence_sigma_deg,
    });
    let normal = grid.plane_normal(tilt);
    sequence.normal = Some(normal.into());
    let turn_deg = tilt_degrees(track, edited, normal)?;
    sequence.turn_deg = Some(turn_deg);
    if weak_turn(turn_deg, weak, options) {
        sequence.outcome = GrownPlaneOutcome::WeakTurn;
        record_views(&mut sequence.views, reference, &candidates, &outcome.fates);
        let mut turn = progress.detail_phase("turn");
        progress_note!(
            turn,
            "large turn on weak evidence, {turn_deg:.1}\u{b0} at {:.1}\u{3c3}: normal left unchanged",
            turn_sigma_ratio(turn_deg, weak)
        );
        return turn_and_read(
            track,
            edited,
            images,
            patch_frame.normal(),
            NormalEstimate::GrownPlane(Box::new(sequence)),
            &options.fit,
            &turn,
        );
    }
    sequence.outcome = GrownPlaneOutcome::Turned;
    state.fit_shift_only(tilt, &mut outcome.fates, progress)?;

    let mut turn = progress.detail_phase("turn");
    let finals = state.final_views(tilt, &outcome.fates);
    record_views(&mut sequence.views, reference, &candidates, &finals);
    progress_note!(turn, "turned {turn_deg:.2}\u{b0}, keypoints kept");
    turn_and_read(
        track,
        edited,
        images,
        normal,
        NormalEstimate::GrownPlane(Box::new(sequence)),
        &options.fit,
        &turn,
    )
}

/// Fill `views` from what became of each candidate: the reference, each
/// joined view's growth that was kept and its shift and score at the final
/// plane, each shift-only view's fit, and each view left out with its
/// refusal.
fn record_views(
    views: &mut [GrownPlaneView],
    reference: usize,
    candidates: &[Candidate<'_>],
    fates: &[Fate],
) {
    views[reference] = GrownPlaneView::Reference;
    for (candidate, fate) in candidates.iter().zip(fates) {
        views[candidate.row] = match fate {
            Fate::Joined {
                estimate,
                distance,
                held,
                final_shift,
            } => GrownPlaneView::Joined {
                shift: final_shift.unwrap_or(estimate.shift),
                grid_zncc: held.unwrap_or(f64::NAN),
                distance: *distance,
                held_zncc: *held,
            },
            Fate::ShiftOnly { shift, grid_zncc } => GrownPlaneView::ShiftOnly {
                shift: *shift,
                grid_zncc: *grid_zncc,
            },
            Fate::LeftOut {
                reason,
                distance,
                held,
            } => GrownPlaneView::LeftOut {
                reason: *reason,
                distance: *distance,
                held_zncc: *held,
            },
            Fate::NotTried => GrownPlaneView::NotAttempted,
        };
    }
}

/// The ratio of a turn to the weakest-direction `σ`: 0 where `sigma_deg` is
/// not finite, which is no evidence.
fn turn_sigma_ratio(turn_deg: f64, sigma_deg: f64) -> f64 {
    if sigma_deg.is_finite() {
        turn_deg / sigma_deg
    } else {
        0.0
    }
}

/// Whether a turn of `turn_deg` to a plane of weakest `σ` `sigma_deg` is a
/// large turn on weak evidence, which the step does not make.
fn weak_turn(turn_deg: f64, sigma_deg: f64, options: &GrownPlaneNormalOptions) -> bool {
    turn_deg >= options.weak_turn_deg
        && turn_sigma_ratio(turn_deg, sigma_deg) < options.min_turn_sigma_ratio
}

/// Which plane a growth started from, in a detail phase's note.
fn tilt_start_word(from: GrownPlaneTiltStart) -> &'static str {
    match from {
        GrownPlaneTiltStart::Starting => "from the starting plane",
        GrownPlaneTiltStart::Combined => "from the combined plane",
    }
}

/// Whether a growth was taken, and why not, in a detail phase's note.
fn verdict_words(refusal: Option<GrownPlaneRefusal>) -> &'static str {
    match refusal {
        None => "taken",
        Some(GrownPlaneRefusal::UnderFloor) => "refused under the floor",
        Some(GrownPlaneRefusal::TooFar) => "refused too far",
        Some(GrownPlaneRefusal::HeldUnderFloor) => "refused held under the floor",
        Some(GrownPlaneRefusal::ViewLeftOut) => "refused, a view left out",
        Some(GrownPlaneRefusal::StoppedEarly) => "refused, stopped early",
        Some(GrownPlaneRefusal::NoKeypoint) => "refused, no keypoint",
    }
}

/// The stages' window half-widths on a grid of `side` samples: the first
/// window's, then each of `window_fractions` of the side rounded, those no
/// wider than the first dropped, in increasing order without repeats.
fn window_radii(side: usize, options: &GrownPlaneNormalOptions) -> Vec<usize> {
    let first = (options.spot_window / 2) as usize;
    let mut later: Vec<usize> = options
        .window_fractions
        .iter()
        .map(|f| (side as f64 * f).round() as usize)
        .filter(|&r| r > first)
        .collect();
    later.sort_unstable();
    later.dedup();
    std::iter::once(first).chain(later).collect()
}

/// One view the step may try.
struct Candidate<'a> {
    row: usize,
    image_index: u32,
    image: ViewImage<'a>,
    /// The keypoint less the projection of the view's slid patch's centre:
    /// zero, up to rounding, where the slide followed the keypoint, and the
    /// keypoint's gap from the unslid centre's projection where the slide
    /// refused it and the view reads the patch as it stands. The same at every
    /// tilt, since no tilt moves that centre. A row with no keypoint is not a
    /// candidate.
    kp_shift: [f64; 2],
    angle_deg: f64,
    /// The unit direction from the patch centre to its camera.
    direction: Option<Vector3<f64>>,
    /// The unit direction of its offset from the reference in the starting
    /// patch frame's `(u, v)`; zero where it has none.
    offset_direction: [f64; 2],
    /// Photograph px per world unit at the anchor; `NaN` where the anchor
    /// does not project.
    scale_px: f64,
}

/// A working grid laid at one plane, with the reference's template on it.
struct Laid {
    grid: Grid,
    /// The reference rendered on the grid at no tilt and no shift: the
    /// samples every view read on this grid is scored against.
    template: Samples,
    /// Which of the grid's cells the template is textured in.
    live: [bool; 9],
}

impl Laid {
    /// `grid` with the template rendered on it from `reference`, one render.
    fn new(grid: Grid, reference: &ViewImage<'_>) -> Laid {
        let template = ViewRender::new(&grid, reference).render([0.0; 2], [0.0; 2], &grid.all());
        let live = live_cells(&grid.cells, &template);
        Laid {
            grid,
            template,
            live,
        }
    }
}

/// What every growth of one step shares.
struct Sequence<'s, 'a> {
    options: &'s GrownPlaneNormalOptions,
    iteration: Iteration,
    /// The grid laid on the starting square, with its template.
    start: Rc<Laid>,
    /// The reference's photograph, which every template is rendered from;
    /// its render count is the step's template renders.
    reference: &'s ViewImage<'a>,
    radii: &'s [usize],
    candidates: &'s [Candidate<'a>],
    /// The last grid laid at a plane other than the starting one, keyed by
    /// the bits of the normal it was laid at.
    last_laid: RefCell<Option<([u64; 3], Rc<Laid>)>>,
}

/// One growth of one view.
#[derive(Clone)]
struct Grown {
    /// Its plane, as a tilt of the starting frame.
    tilt: [f64; 2],
    /// Its plane's outward normal, world frame.
    normal: Vector3<f64>,
    attempt: GrownPlaneAttempt,
}

/// The plane a fit on `grid` reached, moving the tilt from `from` to `to`,
/// and the angle it turned, degrees: `normal`, the plane at `from`, is kept
/// bit for bit when the fit did not move the tilt.
fn reached(grid: &Grid, normal: Vector3<f64>, from: [f64; 2], to: [f64; 2]) -> (Vector3<f64>, f64) {
    if to == from {
        return (normal, 0.0);
    }
    let next = grid.plane_normal(to);
    (next, angle_deg(normal, next))
}

/// A judged growth: the growth, its verdict in its attempt, and the estimate
/// it gives.
struct Judged {
    grown: Grown,
    estimate: Estimate,
}

/// What became of one candidate.
enum Fate {
    /// Its plane is in the final plane; `held` is its score there, from the
    /// combination's last held check, and `final_shift` its shift there,
    /// filled in once the final plane is known.
    Joined {
        estimate: Box<Estimate>,
        distance: Option<f64>,
        held: Option<f64>,
        final_shift: Option<[f64; 2]>,
    },
    ShiftOnly {
        shift: [f64; 2],
        grid_zncc: f64,
    },
    LeftOut {
        reason: GrownPlaneRefusal,
        distance: Option<f64>,
        held: Option<f64>,
    },
    NotTried,
}

/// How the selection, the combination and the pairs ended.
struct Outcome {
    /// The final plane; `None` when no view joined.
    tilt: Option<[f64; 2]>,
    /// Every kept estimate's information summed.
    information: Matrix2<f64>,
    /// The same with each kept pair's halved.
    discounted: Matrix2<f64>,
    /// One per candidate.
    fates: Vec<Fate>,
    trace: Vec<GrownPlaneTry>,
    picks: Vec<GrownPlanePick>,
    pairs: Vec<GrownPlanePair>,
}

impl<'a> Sequence<'_, 'a> {
    /// Whether the working grid is laid again at the plane each stage starts
    /// at ([`GrownPlaneNormalOptions::turned_support`]).
    fn turned(&self) -> bool {
        self.options.turned_support
    }

    /// The grid laid at the plane of unit normal `normal`, with its
    /// template: the starting grid where `normal` is the starting normal,
    /// the last grid laid where it was laid at the same normal, bit for bit,
    /// and otherwise a grid laid now, with one template render.
    fn lay(&self, normal: Vector3<f64>) -> Rc<Laid> {
        if normal == self.start.grid.start.normal {
            return Rc::clone(&self.start);
        }
        let key = [normal.x.to_bits(), normal.y.to_bits(), normal.z.to_bits()];
        if let Some((k, laid)) = self.last_laid.borrow().as_ref() {
            if *k == key {
                return Rc::clone(laid);
            }
        }
        let laid = Rc::new(Laid::new(self.start.grid.laid_at(normal), self.reference));
        *self.last_laid.borrow_mut() = Some((key, Rc::clone(&laid)));
        laid
    }

    /// The grid a whole grid is read on at the plane `tilt` of the starting
    /// frame, and that plane's tilt in the grid's frame: with the turned
    /// support, the grid laid there, at no tilt, the starting grid at the
    /// starting plane; without it, the starting grid at `tilt`.
    fn laid_for(&self, tilt: [f64; 2]) -> (Rc<Laid>, [f64; 2]) {
        if !self.turned() || tilt == [0.0; 2] {
            return (Rc::clone(&self.start), tilt);
        }
        (self.lay(self.start.grid.plane_normal(tilt)), [0.0; 2])
    }

    /// The starting frame's tilt of the plane of normal `normal`. Every
    /// plane a fit keeps is under 90° from the starting normal
    /// ([`Grid::faces_start`]), so the chart states it; `(0, 0)` stands in
    /// where it would not.
    fn start_tilt(&self, normal: Vector3<f64>) -> [f64; 2] {
        self.start.grid.start.tilt_of(normal).unwrap_or([0.0; 2])
    }

    /// The whole-grid rounds' schedule: [`Rounds`] from
    /// [`GrownPlaneNormalOptions::stage_turn_cap_deg`] with the turned
    /// support, one round with no cap without it.
    fn rounds(&self) -> Rounds {
        let options = self.options;
        if self.turned() {
            Rounds::new(
                options.stage_turn_cap_deg,
                options.min_stage_turn_cap_deg,
                options.max_grid_rounds,
            )
        } else {
            Rounds::new(f64::INFINITY, 0.0, 1)
        }
    }

    /// How a stage fitted under the cap `cap_deg` iterates: with the turned
    /// support, within the disc of that cap about the stage's own `(0, 0)`.
    fn capped_iteration(&self, cap_deg: f64) -> Iteration {
        Iteration {
            disc: if self.turned() {
                disc_of(cap_deg)
            } else {
                None
            },
            ..self.iteration
        }
    }

    /// The fit of the view `render` on `laid`'s grid, read as `reading` says
    /// against `laid`'s template, with the tilt held at `held` where given.
    fn fit_on<'r, 'g>(
        &self,
        laid: &'r Laid,
        render: &'r ViewRender<'g, 'a>,
        reading: Reading,
        held: Option<[f64; 2]>,
    ) -> ViewFit<'r, 'g, 'a> {
        ViewFit {
            render,
            reference: &laid.template,
            live: &laid.live,
            reading,
            held,
            min_coverage: self.options.min_coverage,
        }
    }

    /// How many renders every view has made so far, on every grid.
    fn renders_done(&self) -> u64 {
        self.candidates.iter().map(|c| c.image.renders()).sum()
    }

    /// The estimate candidate `i`'s growth gives: its own plane and shift,
    /// and its information about the tilt and normal matrix read there,
    /// with the score read from the same render where the turned support
    /// lays the grid at its plane for them.
    ///
    /// Without the turned support the normal matrix is read on the starting
    /// grid at the growth's tilt, in the starting frame. With it, the grid is
    /// laid at the growth's plane, the normal matrix and the score are read
    /// there at no tilt, the normal matrix stays in that grid's frame, and
    /// the information is carried into the starting frame's tilt by the
    /// chart's Jacobian.
    fn estimate_of(&self, i: usize, grown: &Grown) -> (Estimate, Option<f64>) {
        let image = &self.candidates[i].image;
        let shift = grown.attempt.shift;
        if !self.turned() {
            let render = ViewRender::new(&self.start.grid, image);
            let p = [grown.tilt[0], grown.tilt[1], shift[0], shift[1]];
            let fit = self.fit_on(&self.start, &render, Reading::Grid, None);
            let normal = normal_matrix(&fit, &p);
            let information = normal.as_ref().map_or_else(Matrix2::zeros, schur_tilt);
            let estimate = Estimate {
                index: i,
                tilt: grown.tilt,
                shift,
                information,
                normal,
                frame: None,
            };
            return (estimate, None);
        }
        let laid = self.lay(grown.normal);
        let render = ViewRender::new(&laid.grid, image);
        let fit = self.fit_on(&laid, &render, Reading::Grid, None);
        let (score, normal) = score_and_normal_matrix(&fit, &[0.0, 0.0, shift[0], shift[1]]);
        let own = normal.as_ref().map_or_else(Matrix2::zeros, schur_tilt);
        let estimate = Estimate {
            index: i,
            tilt: grown.tilt,
            shift,
            information: carried_information(&own, &laid.grid.start_chart_jacobian()),
            normal,
            frame: Some(laid.grid.frame),
        };
        (estimate, Some(score))
    }

    /// `estimate`'s shift moved to the plane `tilt` of the starting frame by
    /// its own normal matrix: the shift the fit would reach there to first
    /// order, with no render. The tilt change is read in the frame the
    /// normal matrix is in: `tilt` less the estimate's own in the starting
    /// frame, or, for an estimate read on a laid grid, the chart of that grid
    /// at the target plane less at the estimate's own. A target at or past
    /// 90° from that grid's normal leaves the shift unmoved.
    fn moved_shift(&self, estimate: &Estimate, tilt: [f64; 2]) -> [f64; 2] {
        let Some(n) = &estimate.normal else {
            return estimate.shift;
        };
        let dt = match &estimate.frame {
            None => [tilt[0] - estimate.tilt[0], tilt[1] - estimate.tilt[1]],
            Some(frame) => {
                let target = self.start.grid.plane_normal(tilt);
                match frame.tilt_step(frame.normal, target) {
                    Some(step) => step,
                    None => return estimate.shift,
                }
            }
        };
        let ds = shift_response(n, dt);
        [estimate.shift[0] + ds[0], estimate.shift[1] + ds[1]]
    }

    /// Candidate `i`'s robust grid score at the plane `tilt` of the starting
    /// frame and `shift`, on the grid laid there with the turned support: one
    /// render, and one template render where that grid is laid anew.
    fn score_at(&self, i: usize, tilt: [f64; 2], shift: [f64; 2]) -> f64 {
        let (laid, own) = self.laid_for(tilt);
        let render = ViewRender::new(&laid.grid, &self.candidates[i].image);
        self.fit_on(&laid, &render, Reading::Grid, None)
            .score_at(own, shift)
    }

    /// Each joined view's shift at the final plane `tilt`, moved there with
    /// no render. Its score there is not read again: the combination's last
    /// held check scored every kept view at that same plane and shift and
    /// left none out, so `held` already is that score.
    fn final_views(&self, tilt: [f64; 2], fates: &[Fate]) -> Vec<Fate> {
        fates
            .iter()
            .map(|fate| match fate {
                Fate::Joined {
                    estimate,
                    distance,
                    held,
                    ..
                } => Fate::Joined {
                    estimate: estimate.clone(),
                    distance: *distance,
                    held: *held,
                    final_shift: Some(self.moved_shift(estimate, tilt)),
                },
                Fate::ShiftOnly { shift, grid_zncc } => Fate::ShiftOnly {
                    shift: *shift,
                    grid_zncc: *grid_zncc,
                },
                Fate::LeftOut {
                    reason,
                    distance,
                    held,
                } => Fate::LeftOut {
                    reason: *reason,
                    distance: *distance,
                    held: *held,
                },
                Fate::NotTried => Fate::NotTried,
            })
            .collect()
    }

    /// The growth of candidate `i` from the plane `from`, from its
    /// keypoint-implied shift, judged on its own, in a `growth` phase on
    /// `phase`.
    fn growth(
        &self,
        i: usize,
        from: (GrownPlaneTiltStart, [f64; 2]),
        phase: &Progress<'_>,
    ) -> Result<Judged, NormalError> {
        let image = &self.candidates[i].image;
        let shift = self.candidates[i].kp_shift;
        let mut growth = phase.detail_phase("growth");
        let templates_before = self.reference.renders();
        // Stage 1 reads the grid the growth starts on: the starting grid at
        // the plane's tilt, or with the turned support the grid laid at the
        // plane, at no tilt.
        let (laid, t0) = self.laid_for(from.1);
        let rows = laid.grid.window(self.radii[0]);
        let z = self.spot_zncc(i, &laid, t0, &rows, shift);
        let renders_before = image.renders();
        let grown = self.grow(i, from.0, (laid, t0), (z, shift), &growth)?;
        let j = self.judge(i, grown, &growth);
        let attempt = &j.grown.attempt;
        progress_note!(
            growth,
            "{}: first window ZNCC {:.3}, grid ZNCC {:.3}, {}, {} renders, {} template renders",
            tilt_start_word(attempt.tilt_start),
            attempt.stage_zncc[0],
            attempt.grid_zncc,
            verdict_words(attempt.refusal),
            image.renders() - renders_before,
            self.reference.renders() - templates_before
        );
        Ok(j)
    }

    /// Judge a growth of candidate `i` on its own, in a `judge` phase: taken
    /// when its own robust grid score reaches the floor, its information read
    /// at its own plane. With the turned support that score is read on the
    /// grid laid at the growth's plane, with the information. A growth that
    /// stopped early is refused already, and its information is not read.
    fn judge(&self, i: usize, mut grown: Grown, progress: &Progress<'_>) -> Judged {
        let options = self.options;
        let image = &self.candidates[i].image;
        let mut phase = progress.detail_phase("judge");
        let renders_before = image.renders();
        if grown.attempt.refusal == Some(GrownPlaneRefusal::StoppedEarly) {
            // No refused growth is combined, so its information is not read.
            progress_note!(phase, "{}", verdict_words(grown.attempt.refusal));
            drop(phase);
            let estimate = Estimate {
                index: i,
                tilt: grown.tilt,
                shift: grown.attempt.shift,
                information: Matrix2::zeros(),
                normal: None,
                frame: None,
            };
            return Judged { grown, estimate };
        }
        let (estimate, score) = self.estimate_of(i, &grown);
        // σ at the growth's own plane, in that plane's own angular frame.
        let sigma = sigmas_deg_at(
            &estimate.information,
            &self.start.grid.start,
            estimate.tilt,
            options.zncc_tolerance,
        )[0];
        let attempt = &mut grown.attempt;
        if let Some(score) = score {
            attempt.grid_zncc = score;
        }
        attempt.sigma_deg = sigma;
        attempt.refusal = (attempt.grid_zncc.is_nan() || attempt.grid_zncc < options.min_held_zncc)
            .then_some(GrownPlaneRefusal::UnderFloor);
        progress_note!(
            phase,
            "grid ZNCC {:.3}, \u{3c3} {sigma:.2}\u{b0}: {}, {} renders",
            attempt.grid_zncc,
            verdict_words(attempt.refusal),
            image.renders() - renders_before
        );
        drop(phase);
        Judged { grown, estimate }
    }

    /// Grow candidate `i` from `start`, the grid stage 1 was read on and the
    /// starting plane's tilt there, from the first window's ZNCC and shift in
    /// `first`: the growing windows, and the whole grid's robust fit, as
    /// rounds ([`Self::rounds`]). A window stage from
    /// [`EARLY_REFUSAL_FROM_STAGE`] on that scores under the floor less
    /// [`EARLY_REFUSAL_MARGIN`] ends the growth there, refused
    /// ([`GrownPlaneRefusal::StoppedEarly`]).
    ///
    /// Without the turned support every stage fits the tilt on the starting
    /// grid, from where the stage before ended, and the whole-grid stage is
    /// one round. With it, every fitted stage lays the grid at the plane the
    /// growth has reached and fits from that grid's `(0, 0)` within its cap,
    /// [`GrownPlaneNormalOptions::stage_turn_cap_deg`] for a window stage and
    /// the round's for a whole-grid round. The shift carries over from stage
    /// to stage either way.
    fn grow(
        &self,
        i: usize,
        tilt_start: GrownPlaneTiltStart,
        start: (Rc<Laid>, [f64; 2]),
        first: (f64, [f64; 2]),
        progress: &Progress<'_>,
    ) -> Result<Grown, NormalError> {
        let image = &self.candidates[i].image;
        let turned = self.turned();
        let (mut laid, t0) = start;
        let (z1, shift) = first;
        let mut p = vec![t0[0], t0[1], shift[0], shift[1]];
        let mut normal = if turned {
            laid.grid.frame.normal
        } else {
            laid.grid.plane_normal(t0)
        };
        let mut stage_zncc = vec![z1];
        let mut stage_turn_deg = Vec::new();
        let window_iteration = self.capped_iteration(self.options.stage_turn_cap_deg);
        for &half in &self.radii[1..] {
            progress.check_cancel()?;
            let mut stage = progress.detail_phase("window stage");
            let renders_before = image.renders();
            if turned {
                laid = self.lay(normal);
                (p[0], p[1]) = (0.0, 0.0);
            }
            let render = ViewRender::new(&laid.grid, image);
            let fit = self.fit_on(
                &laid,
                &render,
                Reading::Window(laid.grid.window(half)),
                None,
            );
            let (q, s, taken) = maximise(&fit, &p, &window_iteration);
            let (next, turn) = reached(&laid.grid, normal, [p[0], p[1]], [q[0], q[1]]);
            (normal, p) = (next, q);
            stage_zncc.push(s);
            stage_turn_deg.push(turn);
            progress_note!(
                stage,
                "half-width {half}: ZNCC {s:.3}, turned {turn:.2}\u{b0}, {taken} iterations, {} renders",
                image.renders() - renders_before
            );
            let early_floor = self.options.min_held_zncc - EARLY_REFUSAL_MARGIN;
            if stage_zncc.len() > EARLY_REFUSAL_FROM_STAGE && (s.is_nan() || s < early_floor) {
                let tilt = if turned {
                    self.start_tilt(normal)
                } else {
                    [p[0], p[1]]
                };
                return Ok(Grown {
                    tilt,
                    normal,
                    attempt: GrownPlaneAttempt {
                        tilt_start,
                        stage_zncc,
                        shift: [p[2], p[3]],
                        normal: normal.into(),
                        grid_zncc: f64::NAN,
                        sigma_deg: f64::NAN,
                        refusal: Some(GrownPlaneRefusal::StoppedEarly),
                        stage_turn_deg,
                        grid_rounds: Vec::new(),
                    },
                });
            }
        }
        progress.check_cancel()?;
        let mut rounds = self.rounds();
        let mut grid_rounds = Vec::new();
        let mut score;
        loop {
            let mut stage = progress.detail_phase("whole-grid stage");
            let renders_before = image.renders();
            if turned {
                laid = self.lay(normal);
                (p[0], p[1]) = (0.0, 0.0);
            }
            let cap_deg = rounds.cap_deg();
            let render = ViewRender::new(&laid.grid, image);
            let fit = self.fit_on(&laid, &render, Reading::Grid, None);
            let (q, s, taken) = maximise(&fit, &p, &self.capped_iteration(cap_deg));
            let (next, turn) = reached(&laid.grid, normal, [p[0], p[1]], [q[0], q[1]]);
            let more = rounds.after(turn, next - normal);
            (normal, p, score) = (next, q, s);
            grid_rounds.push(GrownPlaneGridRound {
                cap_deg,
                turn_deg: turn,
                zncc: s,
            });
            progress_note!(
                stage,
                "round {}, cap {cap_deg:.2}\u{b0}: turned {turn:.2}\u{b0}, robust ZNCC {s:.3}, {taken} iterations, {} renders",
                grid_rounds.len(),
                image.renders() - renders_before
            );
            drop(stage);
            if !more {
                break;
            }
            progress.check_cancel()?;
        }
        stage_zncc.push(score);
        let tilt = if turned {
            self.start_tilt(normal)
        } else {
            [p[0], p[1]]
        };
        Ok(Grown {
            tilt,
            normal,
            attempt: GrownPlaneAttempt {
                tilt_start,
                stage_zncc,
                shift: [p[2], p[3]],
                normal: normal.into(),
                grid_zncc: score,
                sigma_deg: f64::NAN,
                refusal: None,
                stage_turn_deg,
                grid_rounds,
            },
        })
    }

    /// The first window's ZNCC on `laid`'s grid at the plane `tilt` of that
    /// grid and image shift `shift`, against `laid`'s template.
    fn spot_zncc(
        &self,
        i: usize,
        laid: &Laid,
        tilt: [f64; 2],
        rows: &[usize],
        shift: [f64; 2],
    ) -> f64 {
        let image = &self.candidates[i].image;
        let pixels = laid.grid.project(image, tilt, rows);
        let mut samples = Samples::new(laid.grid.side * laid.grid.side, image.channels);
        image.sample(&pixels, rows, shift, &mut samples);
        render::zncc(&laid.template, &samples, rows, self.options.min_coverage)
    }
}

#[cfg(test)]
mod jacobian_tests;
#[cfg(test)]
mod laid_tests;
#[cfg(test)]
mod tests;
