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
//! fit of the whole grid on the robust cell score. A growth stops early,
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
//! each joined view's keypoint is written where that plane and its shift put
//! the patch, and each view not picked has its shift alone fitted at that
//! plane, over a mid-size window and then briefly over the whole grid.
//!
//! `specs/drafts/grown-plane-normal.md` is the design.

use std::collections::BTreeSet;

use nalgebra::{Matrix2, Vector3};

use crate::patch::keypoint_localize::resolve_reference;
use crate::patch::normal_refine::ProjectedImage;
use crate::patch::reference_view::ReferenceViewOptions;
use crate::progress::Progress;
use crate::progress_note;
use crate::reconstruction::edited::EditedReconstruction;

use super::super::evaluate::{check_views, seed_of, stored_patch_resolution, track_reference};
use super::super::fit::FitOptions;
use super::super::steps::tilt_degrees;
use super::super::track::{EditableTrack, TrackMeasurement};
use super::grid_plane::{working_side, MIN_WORKING_RESOLUTION};
use super::{
    normal_preconditions, patch_frame_of, read_back, NormalError, NormalEstimate, NormalReport,
};

mod combine;
mod render;
mod selected;
mod solve;

use combine::{sigmas_deg, Estimate};
use render::{anchored, Grid, Samples, ViewImage, ViewRender};
use solve::{
    live_cells, maximise, normal_matrix, schur_tilt, shift_response, Iteration, Reading, ViewFit,
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
    /// within its bounds, a finite rung ratio above 1, a coverage in `0 ..= 1`
    /// and a finite damping of at least zero. The error names the first
    /// setting that fails.
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
        Ok(())
    }

    fn iteration(&self) -> Iteration {
        Iteration {
            max_iterations: self.max_iterations,
            damping: self.initial_damping,
            min_improvement: self.min_improvement,
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
    /// robust score after the last fit.
    /// A growth that stopped early ([`GrownPlaneRefusal::StoppedEarly`])
    /// ends at the window stage it stopped at, with no whole-grid score.
    pub stage_zncc: Vec<f64>,
    /// The shift the growth ended at, image px.
    pub shift: [f64; 2],
    /// The outward normal of the plane the growth ended at, world frame.
    pub normal: [f64; 3],
    /// The robust grid score there; `NaN` for a growth that stopped early.
    pub grid_zncc: f64,
    /// `σ` along its own plane's weakest direction, degrees, from its own
    /// information at its own plane and shift; `NaN` for a growth that
    /// stopped early.
    pub sigma_deg: f64,
    /// Why it was refused, [`GrownPlaneRefusal::UnderFloor`] or
    /// [`GrownPlaneRefusal::StoppedEarly`]; `None` when it was taken. Whether a
    /// taken one is kept is the combination's verdict, in
    /// [`GrownPlaneSequence::views`].
    pub refusal: Option<GrownPlaneRefusal>,
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
    /// try that left the view out and for a try made after the selection.
    pub combined_normal: Option<[f64; 3]>,
    /// `σ` along that plane's weakest direction after the try, degrees;
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
    /// The robust score between the two views' renders at the pair's plane.
    pub zncc: f64,
    /// `σ` along the pair's plane's weakest direction, degrees.
    pub sigma_deg: f64,
    /// The Mahalanobis distance of its plane from the combined plane; `None`
    /// for a pair refused under the floor, which was never combined.
    pub distance: Option<f64>,
    /// Why it was not kept; `None` when its plane is in the combined plane.
    pub refusal: Option<GrownPlaneRefusal>,
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
    /// The reference, with the keypoint the step wrote for it after the
    /// turn: the one it had before the step, or the anchor's projection
    /// where it had none. `None` when the step did not turn the patch and
    /// nothing was written.
    Reference { keypoint: Option<[f64; 2]> },
    /// Its own plane is in the final plane. `distance` and `held_zncc` are
    /// its kept growth's plane's distance from the final plane and its score
    /// there. `shift` is its growth's shift
    /// moved to the final plane by its own normal matrix, and `grid_zncc` its
    /// robust grid score there, the same value as `held_zncc`: the
    /// combination's last held check read it at the final plane and that
    /// shift. `was` is its keypoint before the step and `keypoint` the one
    /// the step wrote, both image px; `keypoint` is `None` where that pixel
    /// is off the photograph, or the turn was not made, and nothing was
    /// written. Where the turn was not made, `shift` is its growth's own
    /// shift, not moved.
    Joined {
        shift: [f64; 2],
        grid_zncc: f64,
        distance: Option<f64>,
        held_zncc: Option<f64>,
        was: Option<[f64; 2]>,
        keypoint: Option<[f64; 2]>,
    },
    /// Not picked by the selection: its shift alone was fitted at the final
    /// plane from its keypoint-implied shift, over a mid-size window and then
    /// for two iterations over the whole grid, and it is not in the plane.
    /// `keypoint` is written when `grid_zncc` reaches
    /// [`GrownPlaneNormalOptions::min_held_zncc`] and the pixel is on the
    /// photograph, and is `None` otherwise.
    ShiftOnly {
        shift: [f64; 2],
        grid_zncc: f64,
        was: Option<[f64; 2]>,
        keypoint: Option<[f64; 2]>,
    },
    /// Left out, with the refusal, distance and held score that left it out:
    /// [`GrownPlaneRefusal::NoKeypoint`] for a view with no keypoint-implied
    /// shift, under every outcome. The step writes no keypoint for it.
    LeftOut {
        reason: GrownPlaneRefusal,
        distance: Option<f64>,
        held_zncc: Option<f64>,
    },
    /// Not tried: not `in`, a second row in an image already tried, no
    /// reference was found, or not picked by a selection that joined no view
    /// or whose turn was not made ([`GrownPlaneOutcome::WeakTurn`]). The step
    /// writes no keypoint for it.
    NotAttempted,
}

/// How precisely the kept estimates determine the plane.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GrownPlanePrecision {
    /// `σ` along the plane's weakest and strongest directions, degrees, from
    /// every kept estimate's information summed, each read at its own plane.
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
    /// How many keypoints the step wrote, the reference's among them.
    pub written: usize,
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
                    write!(f, "; {} keypoints written", self.written)
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
/// [`tilt_patch`](crate::bench::tilt_patch), the reference's keypoint is
/// written back as it was before the step, each joined view's at the anchor's
/// projection plus its shift moved to the final plane, and each shift-only
/// view's that passed the floor at its fitted shift, and the track is read
/// back with its bitmap rendered; the report gives the plane's precision and
/// says when it is low. When none joined, the normal and the keypoints are
/// left as given and the track is read back. The centre does not move.
///
/// A track at infinity is refused, as by every normal step: its patch faces
/// along its own bearing and has no plane to fit.
///
/// The step opens the phase `grown plane normal` on `progress`, and under it
/// `grown plane view` for each try. A detailed `progress` adds `setup`,
/// `selection` for each pick, `mean` for each combination of planes, `pairs`
/// with a `pair fit` per pair, `shift-only fit` for each view not picked,
/// and `turn and write` under the step; `growth` under each try; and
/// `window stage`, `whole-grid stage` and `judge` under each growth, each
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
        let turn = progress.detail_phase("turn and write");
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
            written: 0,
        };
        return read_back(
            track,
            edited,
            images,
            patch_frame.normal(),
            NormalEstimate::GrownPlane(Box::new(sequence)),
            &options.fit,
            &turn,
            |_| {},
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
    let reference_samples =
        ViewRender::new(&grid, &ref_image).render([0.0; 2], [0.0; 2], &grid.all());
    let live = live_cells(&grid.cells, &reference_samples);
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
                keypoint,
                kp_shift,
                angle_deg,
                direction,
                offset_direction,
                scale_px,
            })
        })
        .collect();
    candidates.sort_by(|a, b| a.angle_deg.total_cmp(&b.angle_deg));
    let renders: Vec<ViewRender<'_, '_>> = candidates
        .iter()
        .map(|c| ViewRender::new(&grid, &c.image))
        .collect();
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
        grid: &grid,
        reference: &reference_samples,
        live: &live,
        radii: &radii,
        candidates: &candidates,
        renders: &renders,
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
        written: 0,
    };
    let Some(tilt) = outcome.tilt else {
        record_views(&mut sequence.views, reference, &candidates, &outcome.fates);
        let mut turn = progress.detail_phase("turn and write");
        progress_note!(turn, "no view joined, normal left unchanged");
        return read_back(
            track,
            edited,
            images,
            patch_frame.normal(),
            NormalEstimate::GrownPlane(Box::new(sequence)),
            &options.fit,
            &turn,
            |_| {},
        );
    };

    let [weak, strong] = sigmas_deg(&outcome.information, options.zncc_tolerance);
    let [d_weak, d_strong] = sigmas_deg(&outcome.discounted, options.zncc_tolerance);
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
        let mut turn = progress.detail_phase("turn and write");
        progress_note!(
            turn,
            "large turn on weak evidence, {turn_deg:.1}\u{b0} at {:.1}\u{3c3}: normal left unchanged",
            turn_sigma_ratio(turn_deg, weak)
        );
        return read_back(
            track,
            edited,
            images,
            patch_frame.normal(),
            NormalEstimate::GrownPlane(Box::new(sequence)),
            &options.fit,
            &turn,
            |_| {},
        );
    }
    sequence.outcome = GrownPlaneOutcome::Turned;
    state.fit_shift_only(tilt, &mut outcome.fates, progress)?;

    let mut turn = progress.detail_phase("turn and write");
    let finals = state.final_views(tilt, &outcome.fates);
    record_views(&mut sequence.views, reference, &candidates, &finals);
    let writes = keypoint_writes(
        &grid,
        &ref_image,
        ref_keypoint,
        reference,
        &candidates,
        options.min_held_zncc,
        &mut sequence.views,
    );
    sequence.written = writes.len();
    progress_note!(turn, "{} keypoints written", writes.len());
    read_back(
        track,
        edited,
        images,
        normal,
        NormalEstimate::GrownPlane(Box::new(sequence)),
        &options.fit,
        &turn,
        move |turned: &mut EditableTrack| {
            for (row, p) in writes {
                turned.observations[row].track = Some(TrackMeasurement {
                    keypoint: Some([p[0] as f32, p[1] as f32]),
                    ..TrackMeasurement::default()
                });
            }
        },
    )
}

/// Fill `views` from what became of each candidate: the reference, each
/// joined view's growth that was kept and its shift and score at the final
/// plane, each shift-only view's fit, and each view left out with its
/// refusal. The written keypoints are filled in later.
fn record_views(
    views: &mut [GrownPlaneView],
    reference: usize,
    candidates: &[Candidate<'_>],
    fates: &[Fate],
) {
    views[reference] = GrownPlaneView::Reference { keypoint: None };
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
                was: candidate.keypoint,
                keypoint: None,
            },
            Fate::ShiftOnly { shift, grid_zncc } => GrownPlaneView::ShiftOnly {
                shift: *shift,
                grid_zncc: *grid_zncc,
                was: candidate.keypoint,
                keypoint: None,
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

/// The keypoints the step writes after the turn, as `(row, pixel)`: the
/// reference's as it was before the step (the anchor's projection where it
/// had none), each joined view's at the projection of its slid patch's centre
/// plus its shift at the final plane, the pixel its score there was read
/// around, and each
/// shift-only view's the same way where its score reaches `floor`. A pixel
/// off its photograph is not written. Each is recorded in `views` too.
fn keypoint_writes(
    grid: &Grid,
    ref_image: &ViewImage<'_>,
    ref_keypoint: Option<[f64; 2]>,
    reference: usize,
    candidates: &[Candidate<'_>],
    floor: f64,
    views: &mut [GrownPlaneView],
) -> Vec<(usize, [f64; 2])> {
    let mut writes = Vec::new();
    let ref_pixel = ref_keypoint
        .or_else(|| grid.anchor_pixel(ref_image))
        .filter(|&p| on_photograph(ref_image, p));
    if let Some(p) = ref_pixel {
        writes.push((reference, p));
    }
    views[reference] = GrownPlaneView::Reference {
        keypoint: ref_pixel,
    };
    for candidate in candidates {
        let at = |shift: [f64; 2]| {
            grid.anchor_pixel(&candidate.image)
                .map(|at| [at[0] + shift[0], at[1] + shift[1]])
                .filter(|&p| on_photograph(&candidate.image, p))
        };
        match &mut views[candidate.row] {
            GrownPlaneView::Joined {
                shift, keypoint, ..
            } => {
                *keypoint = at(*shift);
            }
            GrownPlaneView::ShiftOnly {
                shift,
                grid_zncc,
                keypoint,
                ..
            } => {
                *keypoint = (*grid_zncc >= floor).then(|| at(*shift)).flatten();
            }
            _ => continue,
        }
        if let GrownPlaneView::Joined {
            keypoint: Some(p), ..
        }
        | GrownPlaneView::ShiftOnly {
            keypoint: Some(p), ..
        } = &views[candidate.row]
        {
            writes.push((candidate.row, *p));
        }
    }
    writes
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

/// Whether `pixel` lies on `view`'s photograph, `[0, width) × [0, height)`.
fn on_photograph(view: &ViewImage<'_>, pixel: [f64; 2]) -> bool {
    pixel[0] >= 0.0
        && pixel[1] >= 0.0
        && pixel[0] < f64::from(view.camera.width)
        && pixel[1] < f64::from(view.camera.height)
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
    /// Its keypoint before the step.
    keypoint: Option<[f64; 2]>,
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

/// What every growth of one step shares.
struct Sequence<'s, 'g, 'a> {
    options: &'s GrownPlaneNormalOptions,
    iteration: Iteration,
    grid: &'g Grid,
    reference: &'s Samples,
    live: &'s [bool; 9],
    radii: &'s [usize],
    candidates: &'s [Candidate<'a>],
    renders: &'s [ViewRender<'g, 'a>],
}

/// One growth of one view.
#[derive(Clone)]
struct Grown {
    tilt: [f64; 2],
    attempt: GrownPlaneAttempt,
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

impl Sequence<'_, '_, '_> {
    /// The fit of candidate `i` over the whole grid on the robust cell score,
    /// with the tilt held at `held` where given.
    fn grid_fit(&self, i: usize, held: Option<[f64; 2]>) -> ViewFit<'_, '_, '_> {
        ViewFit {
            render: &self.renders[i],
            reference: self.reference,
            live: self.live,
            reading: Reading::Grid,
            held,
            min_coverage: self.options.min_coverage,
        }
    }

    /// How many renders every view has made so far.
    fn renders_done(&self) -> u64 {
        self.renders.iter().map(ViewRender::renders).sum()
    }

    /// The estimate candidate `i`'s growth gives: its own plane and shift,
    /// and its information about the tilt and normal matrix read there.
    fn estimate_of(&self, i: usize, grown: &Grown) -> Estimate {
        let p = [
            grown.tilt[0],
            grown.tilt[1],
            grown.attempt.shift[0],
            grown.attempt.shift[1],
        ];
        let normal = normal_matrix(&self.grid_fit(i, None), &p);
        let information = normal.as_ref().map_or_else(Matrix2::zeros, schur_tilt);
        Estimate {
            index: i,
            tilt: grown.tilt,
            shift: grown.attempt.shift,
            information,
            normal,
        }
    }

    /// `estimate`'s shift moved to the plane `tilt` by its own normal matrix:
    /// the shift the fit would reach there to first order, with no render.
    fn moved_shift(&self, estimate: &Estimate, tilt: [f64; 2]) -> [f64; 2] {
        let Some(n) = &estimate.normal else {
            return estimate.shift;
        };
        let dt = [tilt[0] - estimate.tilt[0], tilt[1] - estimate.tilt[1]];
        let ds = shift_response(n, dt);
        [estimate.shift[0] + ds[0], estimate.shift[1] + ds[1]]
    }

    /// Candidate `i`'s robust grid score at `tilt` and `shift`: one render.
    fn score_at(&self, i: usize, tilt: [f64; 2], shift: [f64; 2]) -> f64 {
        self.grid_fit(i, None).score_at(tilt, shift)
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
        let shift = self.candidates[i].kp_shift;
        let rows = self.grid.window(self.radii[0]);
        let mut growth = phase.detail_phase("growth");
        let z = self.spot_zncc(i, from.1, &rows, shift);
        let renders_before = self.renders[i].renders();
        let grown = self.grow(i, from, (z, shift), &growth)?;
        let j = self.judge(i, grown, &growth);
        let attempt = &j.grown.attempt;
        progress_note!(
            growth,
            "{}: first window ZNCC {:.3}, grid ZNCC {:.3}, {}, {} renders",
            tilt_start_word(attempt.tilt_start),
            attempt.stage_zncc[0],
            attempt.grid_zncc,
            verdict_words(attempt.refusal),
            self.renders[i].renders() - renders_before
        );
        Ok(j)
    }

    /// Judge a growth of candidate `i` on its own, in a `judge` phase: taken
    /// when its own robust grid score reaches the floor, its information read
    /// at its own plane. A growth that stopped early is refused already, and
    /// its information is not read.
    fn judge(&self, i: usize, mut grown: Grown, progress: &Progress<'_>) -> Judged {
        let options = self.options;
        let mut phase = progress.detail_phase("judge");
        let renders_before = self.renders[i].renders();
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
            };
            return Judged { grown, estimate };
        }
        let estimate = self.estimate_of(i, &grown);
        let sigma = sigmas_deg(&estimate.information, options.zncc_tolerance)[0];
        let attempt = &mut grown.attempt;
        attempt.sigma_deg = sigma;
        attempt.refusal = (attempt.grid_zncc.is_nan() || attempt.grid_zncc < options.min_held_zncc)
            .then_some(GrownPlaneRefusal::UnderFloor);
        progress_note!(
            phase,
            "grid ZNCC {:.3}, \u{3c3} {sigma:.2}\u{b0}: {}, {} renders",
            attempt.grid_zncc,
            verdict_words(attempt.refusal),
            self.renders[i].renders() - renders_before
        );
        drop(phase);
        Judged { grown, estimate }
    }

    /// Grow candidate `i` from the plane `from`, from the first window's
    /// ZNCC and shift in `first`: the growing windows, and the whole
    /// grid's robust fit. A window stage from [`EARLY_REFUSAL_FROM_STAGE`] on
    /// that scores under the floor less [`EARLY_REFUSAL_MARGIN`] ends the
    /// growth there, refused ([`GrownPlaneRefusal::StoppedEarly`]).
    fn grow(
        &self,
        i: usize,
        from: (GrownPlaneTiltStart, [f64; 2]),
        first: (f64, [f64; 2]),
        progress: &Progress<'_>,
    ) -> Result<Grown, NormalError> {
        let render = &self.renders[i];
        let (z1, shift) = first;
        let (tilt_start, t0) = from;
        let mut p = vec![t0[0], t0[1], shift[0], shift[1]];
        let mut stage_zncc = vec![z1];
        for &half in &self.radii[1..] {
            progress.check_cancel()?;
            let mut stage = progress.detail_phase("window stage");
            let renders_before = render.renders();
            let fit = ViewFit {
                render,
                reference: self.reference,
                live: self.live,
                reading: Reading::Window(self.grid.window(half)),
                held: None,
                min_coverage: self.options.min_coverage,
            };
            let (q, s, taken) = maximise(&fit, &p, &self.iteration);
            p = q;
            stage_zncc.push(s);
            progress_note!(
                stage,
                "half-width {half}: ZNCC {s:.3}, {taken} iterations, {} renders",
                render.renders() - renders_before
            );
            let early_floor = self.options.min_held_zncc - EARLY_REFUSAL_MARGIN;
            if stage_zncc.len() > EARLY_REFUSAL_FROM_STAGE && (s.is_nan() || s < early_floor) {
                let tilt = [p[0], p[1]];
                return Ok(Grown {
                    tilt,
                    attempt: GrownPlaneAttempt {
                        tilt_start,
                        stage_zncc,
                        shift: [p[2], p[3]],
                        normal: self.grid.plane_normal(tilt).into(),
                        grid_zncc: f64::NAN,
                        sigma_deg: f64::NAN,
                        refusal: Some(GrownPlaneRefusal::StoppedEarly),
                    },
                });
            }
        }
        progress.check_cancel()?;
        let mut stage = progress.detail_phase("whole-grid stage");
        let renders_before = render.renders();
        let (q, score, taken) = maximise(&self.grid_fit(i, None), &p, &self.iteration);
        p = q;
        stage_zncc.push(score);
        progress_note!(
            stage,
            "robust ZNCC {score:.3}, {taken} iterations, {} renders",
            render.renders() - renders_before
        );
        drop(stage);
        let tilt = [p[0], p[1]];
        Ok(Grown {
            tilt,
            attempt: GrownPlaneAttempt {
                tilt_start,
                stage_zncc,
                shift: [p[2], p[3]],
                normal: self.grid.plane_normal(tilt).into(),
                grid_zncc: score,
                sigma_deg: f64::NAN,
                refusal: None,
            },
        })
    }

    /// The first window's ZNCC at the plane `tilt` and image shift `shift`.
    fn spot_zncc(&self, i: usize, tilt: [f64; 2], rows: &[usize], shift: [f64; 2]) -> f64 {
        let render = &self.renders[i];
        let pixels = self.grid.project(render.view, tilt, rows);
        let mut samples = Samples::new(self.grid.side * self.grid.side, render.view.channels);
        render.view.sample(&pixels, rows, shift, &mut samples);
        render::zncc(self.reference, &samples, rows, self.options.min_coverage)
    }
}

#[cfg(test)]
mod jacobian_tests;
#[cfg(test)]
mod tests;
