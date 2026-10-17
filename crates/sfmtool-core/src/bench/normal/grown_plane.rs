// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! [`grown_plane_normal`]: the patch's normal from one plane shared by every
//! view, fitted directly to the photographs, near to far within each view and
//! view by view across the track.
//!
//! The reference view's working grid is the template. For every other `in`
//! view the plane predicts where each grid sample lands in that view's
//! photograph (`render`); the step fits the plane's tilt and the view's image
//! shift so that the view's samples agree with the reference's by ZNCC. Within
//! a view it registers a small window at the grid's centre first, by an
//! exhaustive shift search about the shift the view's keypoint implies, and
//! grows the window outward, fitting tilt and shift by Levenberg–Marquardt
//! (`solve`) at each size, and ends with a fit of the whole grid on the robust
//! cell score. Across views it takes the views nearest the reference in angle
//! first; each starts from the plane the views before it share, and joins when
//! its own plane agrees with the shared one within the two planes' precision.
//! The patch is turned to the shared plane whenever a view joined, and each
//! joined view's keypoint is written where the plane and its shift put the
//! patch.
//!
//! `specs/drafts/grown-plane-normal.md` is the design.

use std::collections::BTreeSet;

use nalgebra::{Matrix2, Vector2};

use crate::patch::keypoint_localize::resolve_reference;
use crate::patch::normal_refine::ProjectedImage;
use crate::patch::reference_view::ReferenceViewOptions;
use crate::progress::Progress;
use crate::progress_note;
use crate::reconstruction::edited::EditedReconstruction;

use super::super::evaluate::{check_views, seed_of, stored_patch_resolution, track_reference};
use super::super::fit::FitOptions;
use super::super::track::{EditableTrack, TrackMeasurement};
use super::grid_plane::{working_side, MIN_WORKING_RESOLUTION};
use super::{frame_of, normal_preconditions, read_back, NormalError, NormalEstimate, NormalReport};

mod render;
mod solve;

use render::{anchored, Grid, Samples, ViewImage, ViewRender};
use solve::{live_cells, maximise, tilt_information, Iteration, JointFit, Reading, ViewFit};

/// How [`grown_plane_normal`] grows its windows, fits its planes and decides
/// when a view joins.
#[derive(Debug, Clone)]
pub struct GrownPlaneNormalOptions {
    /// Side of the working grid the views are sampled on, in samples. `None`
    /// follows the reference view's footprint of the patch at the starting
    /// frame, about one sample per photograph pixel, clamped to `R ..= 2R`,
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
    /// The first window's search for the view's shift about the shift the
    /// view's keypoint implies: its bound, image px. 3.
    pub keypoint_search_px: f64,
    /// The search widens to [`Self::spot_search_px`] when no shift within
    /// [`Self::keypoint_search_px`] reaches this ZNCC, or the best is on its
    /// edge. 0.7.
    pub min_keypoint_zncc: f64,
    /// The widened search's bound about zero shift, image px: also the whole
    /// search of a view with no keypoint. 12.
    pub spot_search_px: f64,
    /// The step of both searches, image px. 0.5.
    pub spot_search_step_px: f64,
    /// The ZNCC difference noise can produce, `τ`: a view's information about
    /// the tilt becomes the covariance `2τ·I⁻¹`. 0.01.
    pub zncc_tolerance: f64,
    /// A view agrees when the Mahalanobis distance between its own plane and
    /// the shared plane, the two covariances added, is at most this. 3.
    pub max_agreement_distance: f64,
    /// A view agrees only when, held to the shared plane with its shift
    /// fitted again, its robust grid score is at least this; the first view
    /// joins on its own score reaching this alone. 0.5.
    pub min_held_zncc: f64,
    /// A weakest-direction `σ` above this, degrees, is reported as low
    /// confidence. It does not stop the turn. 2.
    pub low_confidence_sigma_deg: f64,
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
    /// The forward-difference step of the Jacobian along each tilt
    /// parameter, tilt units. 1e-3.
    pub jacobian_tilt_step: f64,
    /// The forward-difference step of the Jacobian along each shift
    /// parameter, image px. 0.05.
    pub jacobian_shift_step_px: f64,
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
            keypoint_search_px: 3.0,
            min_keypoint_zncc: 0.7,
            spot_search_px: 12.0,
            spot_search_step_px: 0.5,
            zncc_tolerance: 0.01,
            max_agreement_distance: 3.0,
            min_held_zncc: 0.5,
            low_confidence_sigma_deg: 2.0,
            min_coverage: 0.6,
            max_iterations: 20,
            initial_damping: 1e-3,
            min_improvement: 1e-6,
            jacobian_tilt_step: 1e-3,
            jacobian_shift_step_px: 0.05,
            fit: FitOptions::default(),
        }
    }
}

impl GrownPlaneNormalOptions {
    /// Whether the settings are ones the step takes: a first window of at
    /// least one sample, finite fractions above zero, finite search bounds of
    /// at least zero, a positive search step and Jacobian steps, a positive
    /// tolerance, a coverage in `0 ..= 1` and a finite damping of at least
    /// zero. The error names the first setting that fails.
    pub fn check(&self) -> Result<(), NormalError> {
        let bad = |name| Err(NormalError::BadGrownPlaneOption(name));
        let positive = |v: f64| v.is_finite() && v > 0.0;
        if self.spot_window == 0 {
            return bad("spot_window");
        }
        if !self.window_fractions.iter().all(|&f| positive(f)) {
            return bad("window_fractions");
        }
        if !(self.keypoint_search_px.is_finite() && self.keypoint_search_px >= 0.0) {
            return bad("keypoint_search_px");
        }
        if !(self.spot_search_px.is_finite() && self.spot_search_px >= 0.0) {
            return bad("spot_search_px");
        }
        if !positive(self.spot_search_step_px) {
            return bad("spot_search_step_px");
        }
        if !positive(self.zncc_tolerance) {
            return bad("zncc_tolerance");
        }
        if !(0.0..=1.0).contains(&self.min_coverage) {
            return bad("min_coverage");
        }
        if !positive(self.jacobian_tilt_step) {
            return bad("jacobian_tilt_step");
        }
        if !positive(self.jacobian_shift_step_px) {
            return bad("jacobian_shift_step_px");
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
            tilt_step: self.jacobian_tilt_step,
            shift_step: self.jacobian_shift_step_px,
        }
    }
}

/// How a view's first window found its shift.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GrownPlaneStart {
    /// Within [`GrownPlaneNormalOptions::keypoint_search_px`] of the shift its
    /// keypoint implies.
    Keypoint,
    /// The search about the keypoint found no shift reaching
    /// [`GrownPlaneNormalOptions::min_keypoint_zncc`], or its best was on the
    /// edge, so it widened to [`GrownPlaneNormalOptions::spot_search_px`].
    Widened,
    /// The view has no keypoint, so the whole search ran about zero shift.
    NoKeypoint,
}

/// Why a view's try did not join the shared plane.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GrownPlaneRefusal {
    /// The first view's own robust grid score is under
    /// [`GrownPlaneNormalOptions::min_held_zncc`].
    UnderFloor,
    /// Its own plane is further than
    /// [`GrownPlaneNormalOptions::max_agreement_distance`] from the shared
    /// plane.
    TooFar,
    /// Held to the shared plane, its robust grid score is under
    /// [`GrownPlaneNormalOptions::min_held_zncc`].
    HeldUnderFloor,
}

/// One growth of one view from one starting plane, and how it was judged.
#[derive(Debug, Clone, PartialEq)]
pub struct GrownPlaneAttempt {
    /// Whether it grew from the starting plane rather than the shared one.
    /// The first view, and a retry, grow from the starting plane.
    pub from_start: bool,
    /// How the first window found its shift.
    pub start: GrownPlaneStart,
    /// Each stage's score: the first window's ZNCC at its best shift, each
    /// later window's ZNCC after its fit, and the whole grid's robust score
    /// after the last fit.
    pub stage_zncc: Vec<f64>,
    /// The shift the growth ended at, image px.
    pub shift: [f64; 2],
    /// The outward normal of the plane the growth ended at, world frame.
    pub normal: [f64; 3],
    /// The robust grid score there.
    pub grid_zncc: f64,
    /// The Mahalanobis distance of its plane from the shared plane; `None`
    /// for the first view, which has none to agree with.
    pub distance: Option<f64>,
    /// Its robust grid score held to the shared plane with its shift fitted
    /// again; `None` for the first view.
    pub held_zncc: Option<f64>,
    /// `σ` along its own plane's weakest direction, degrees, from its own
    /// information; `NaN` for the first view, whose information is read at
    /// the shared plane once it joins.
    pub sigma_deg: f64,
    /// Why it did not join; `None` when it did.
    pub refusal: Option<GrownPlaneRefusal>,
}

/// What became of one try of a view.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GrownPlaneTryOutcome {
    /// It joined the shared plane.
    Joined,
    /// Both starts failed, and the view went to the back of the queue once.
    Deferred,
    /// Both starts failed on its second try, and the view was left out.
    LeftOut,
}

/// One try of one view in the sequence, and the shared plane after it.
#[derive(Debug, Clone, PartialEq)]
pub struct GrownPlaneTry {
    /// The observation tried.
    pub observation: usize,
    /// Its image.
    pub image: u32,
    /// Its angle from the reference as seen from the patch centre, degrees.
    pub angle_deg: f64,
    /// Its growths, from the shared plane first and then, where that failed
    /// and a shared plane existed, from the starting plane.
    pub attempts: Vec<GrownPlaneAttempt>,
    pub outcome: GrownPlaneTryOutcome,
    /// The shared plane's outward normal after the try, world frame; `None`
    /// while no view has joined.
    pub shared_normal: Option<[f64; 3]>,
    /// `σ` along the shared plane's weakest direction after the try,
    /// degrees, from the information the joined views gave when each
    /// joined; `NaN` while no view has joined.
    pub shared_sigma_deg: f64,
}

/// What the step made of one observation of the track.
#[derive(Debug, Clone, PartialEq)]
pub enum GrownPlaneView {
    /// The reference, with the keypoint the step wrote for it after the
    /// turn: the one it had before the step, or the anchor's projection
    /// where it had none. `None` when no view joined and nothing was written.
    Reference { keypoint: Option<[f64; 2]> },
    /// Joined the shared plane. `from_start`, `start`, `distance` and
    /// `held_zncc` are those of the growth that joined; `shift` and
    /// `grid_zncc` are its shift and robust grid score at the final shared
    /// plane. `was` is its keypoint before the step and `keypoint` the one
    /// the step wrote, both image px; `keypoint` is `None` where that pixel
    /// is off the photograph and nothing was written.
    Joined {
        from_start: bool,
        start: GrownPlaneStart,
        shift: [f64; 2],
        grid_zncc: f64,
        distance: Option<f64>,
        held_zncc: Option<f64>,
        was: Option<[f64; 2]>,
        keypoint: Option<[f64; 2]>,
    },
    /// Left out after its second try, with that try's last growth's refusal,
    /// distance and held score. The step writes no keypoint for it.
    LeftOut {
        reason: GrownPlaneRefusal,
        distance: Option<f64>,
        held_zncc: Option<f64>,
    },
    /// Not tried: not `in`, a second row in an image already tried, or no
    /// reference was found. The step writes no keypoint for it.
    NotAttempted,
}

/// How precisely the joined views determine the shared plane.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GrownPlanePrecision {
    /// `σ` along the plane's weakest and strongest directions, degrees, from
    /// the joined views' information read at the final shared plane.
    pub sigma_weak_deg: f64,
    pub sigma_strong_deg: f64,
    /// `sigma_weak_deg` is over
    /// [`GrownPlaneNormalOptions::low_confidence_sigma_deg`]. The patch is
    /// turned either way.
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
    /// The shared plane's outward normal, world frame, which the patch was
    /// turned to; `None` when no view joined and the normal was left as it
    /// was.
    pub normal: Option<[f64; 3]>,
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

    /// Whether the step turned the patch: some view joined.
    pub fn turned(&self) -> bool {
        self.normal.is_some()
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
        match &self.precision {
            Some(p) => {
                write!(
                    f,
                    "; \u{3c3} {:.2}\u{b0} weakest, {:.2}\u{b0} strongest",
                    p.sigma_weak_deg, p.sigma_strong_deg
                )?;
                if p.low_confidence {
                    write!(f, ", low confidence")?;
                }
                write!(f, "; {} keypoints written", self.written)
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
/// rows, one row per image. Its working grid is the starting frame re-anchored
/// on its keypoint, [`GrownPlaneNormalOptions::working_resolution`] samples a
/// side. The plane is the starting normal tilted by `(a, b)` along the frame's
/// axes through that anchor; each view adds an image shift.
///
/// The other views are taken in order of their angle from the reference.
/// Each grows from the shared plane: a 7×7 window at the grid's centre finds
/// the view's shift by an exhaustive search about the shift its keypoint
/// implies, the window grows by [`GrownPlaneNormalOptions::window_fractions`]
/// with tilt and shift fitted at each size, and a last fit takes the whole
/// grid on the robust cell score. The first view joins when that score reaches
/// [`GrownPlaneNormalOptions::min_held_zncc`]; each later one when its own
/// plane lies within [`GrownPlaneNormalOptions::max_agreement_distance`] of
/// the shared plane by the Mahalanobis distance and, held to the shared
/// plane, it still scores [`GrownPlaneNormalOptions::min_held_zncc`]. After
/// each join the shared tilt and every joined view's shift are fitted
/// together. A view that fails is grown again from the starting plane, and
/// one that fails both is tried once more at the back of the queue before it
/// is left out.
///
/// When at least one view joined, the patch is turned to the shared plane with
/// [`tilt_patch`](crate::bench::tilt_patch), the reference's keypoint is
/// written back as it was before the step and each joined view's at the
/// anchor's projection plus its shift, and the track is read back with its
/// bitmap rendered; the report gives the plane's precision and says when it is
/// low. When none joined, the normal and the keypoints are left as given and
/// the track is read back. The centre does not move.
///
/// A track at infinity is refused, as by every normal step: its patch faces
/// along its own bearing and has no plane to fit.
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
/// println!("{report}"); // "16 of 17 views joined, 1 left out; σ 0.38° weakest, …"
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
    let frame = frame_of(track);
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
    let view_set: Vec<u32> = rows.iter().map(|&i| track.observations[i].image).collect();
    let seeds: Vec<Option<[f64; 2]>> = rows
        .iter()
        .map(|&i| seed_of(&track.observations[i]))
        .collect();
    let reference = resolve_reference(
        &frame,
        images,
        &view_set,
        &seeds,
        track_reference(track, &rows),
        options.fit.localize.min_grazing_cos,
        base,
        options.fit.localize.sampler,
        &ReferenceViewOptions::default(),
        progress,
    )
    .map(|k| rows[k]);

    let mut views = vec![GrownPlaneView::NotAttempted; track.observations.len()];
    let Some(reference) = reference else {
        let sequence = GrownPlaneSequence {
            reference: None,
            working_resolution: 0,
            window_half_widths: Vec::new(),
            views,
            trace: Vec::new(),
            normal: None,
            precision: None,
            written: 0,
        };
        return read_back(
            track,
            edited,
            images,
            frame.normal(),
            NormalEstimate::GrownPlane(Box::new(sequence)),
            &options.fit,
            progress,
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
            let placement = anchored(&frame, ref_view, ref_keypoint);
            let jacobian = crate::camera::warp_map::patch_grid_jacobian(
                &placement,
                ref_view.camera,
                ref_view.cam_from_world,
                base as usize,
            );
            working_side(jacobian, base).max(floor)
        }
    } as usize;
    let grid = Grid::new(&frame, ref_view, ref_keypoint, side);
    let ref_image = ViewImage::new(ref_view, &grid.placement, side);
    let reference_samples =
        ViewRender::new(&grid, &ref_image).render([0.0; 2], [0.0; 2], &grid.all());
    let live = live_cells(&grid.cells, &reference_samples);
    let radii = window_radii(side, options);

    // The candidates: every other row, nearest the reference in angle first,
    // ties by row.
    let centre = frame.center.coords;
    let ref_direction =
        (ref_view.cam_from_world.inverse_translation_origin().coords - centre).try_normalize(1e-12);
    let mut candidates: Vec<Candidate<'_>> = rows
        .iter()
        .filter(|&&row| row != reference)
        .map(|&row| {
            let observation = &track.observations[row];
            let view = &images[observation.image as usize];
            let image = ViewImage::new(view, &grid.placement, side);
            let keypoint = seed_of(observation);
            let direction = (view.cam_from_world.inverse_translation_origin().coords - centre)
                .try_normalize(1e-12);
            let angle_deg = match (ref_direction, direction) {
                (Some(a), Some(b)) => a.dot(&b).clamp(-1.0, 1.0).acos().to_degrees(),
                _ => f64::INFINITY,
            };
            let kp_shift = keypoint.and_then(|kp| {
                grid.anchor_pixel(&image)
                    .map(|at| [kp[0] - at[0], kp[1] - at[1]])
            });
            Candidate {
                row,
                image_index: observation.image,
                image,
                keypoint,
                kp_shift,
                angle_deg,
            }
        })
        .collect();
    candidates.sort_by(|a, b| a.angle_deg.total_cmp(&b.angle_deg));
    let renders: Vec<ViewRender<'_, '_>> = candidates
        .iter()
        .map(|c| ViewRender::new(&grid, &c.image))
        .collect();

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
    let run = {
        let phase = progress.phase("grown plane normal");
        state.run(&phase)?
    };

    record_views(&mut views, reference, &candidates, &run);
    let mut sequence = GrownPlaneSequence {
        reference: Some(reference),
        working_resolution: side as u32,
        window_half_widths: radii.iter().map(|&r| r as u32).collect(),
        views,
        trace: run.trace,
        normal: None,
        precision: None,
        written: 0,
    };
    if run.members.is_empty() {
        return read_back(
            track,
            edited,
            images,
            frame.normal(),
            NormalEstimate::GrownPlane(Box::new(sequence)),
            &options.fit,
            progress,
            |_| {},
        );
    }

    // The precision at the final plane, and each joined view's score there.
    let mut information = Matrix2::zeros();
    for member in &run.members {
        let fit = state.grid_fit(member.index, None);
        let p = [run.tilt[0], run.tilt[1], member.shift[0], member.shift[1]];
        information += tilt_information(&fit, &p, &state.iteration);
        if let GrownPlaneView::Joined { grid_zncc, .. } =
            &mut sequence.views[candidates[member.index].row]
        {
            *grid_zncc = fit.score_at(run.tilt, member.shift);
        }
    }
    let [weak, strong] = sigmas_deg(&information, options.zncc_tolerance);
    sequence.precision = Some(GrownPlanePrecision {
        sigma_weak_deg: weak,
        sigma_strong_deg: strong,
        low_confidence: weak.is_nan() || weak > options.low_confidence_sigma_deg,
    });
    let normal = grid.plane_normal(run.tilt);
    sequence.normal = Some(normal.into());

    let writes = keypoint_writes(
        &grid,
        &ref_image,
        ref_keypoint,
        reference,
        &candidates,
        &run.members,
        &mut sequence.views,
    );
    sequence.written = writes.len();
    read_back(
        track,
        edited,
        images,
        normal,
        NormalEstimate::GrownPlane(Box::new(sequence)),
        &options.fit,
        progress,
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

/// Fill `views` from how the sequence ended: the reference, each joined
/// view's growth that joined and its final shift, and each view left out with
/// its last growth's refusal. The scores at the final plane and the written
/// keypoints are filled in later.
fn record_views(
    views: &mut [GrownPlaneView],
    reference: usize,
    candidates: &[Candidate<'_>],
    run: &Run,
) {
    views[reference] = GrownPlaneView::Reference { keypoint: None };
    for (i, candidate) in candidates.iter().enumerate() {
        views[candidate.row] = match run.members.iter().find(|m| m.index == i) {
            Some(member) => GrownPlaneView::Joined {
                from_start: member.attempt.from_start,
                start: member.attempt.start,
                shift: member.shift,
                grid_zncc: f64::NAN,
                distance: member.attempt.distance,
                held_zncc: member.attempt.held_zncc,
                was: candidate.keypoint,
                keypoint: None,
            },
            None => match run.left_out.iter().find(|(k, _)| *k == i) {
                Some((_, attempt)) => GrownPlaneView::LeftOut {
                    reason: attempt.refusal.unwrap_or(GrownPlaneRefusal::UnderFloor),
                    distance: attempt.distance,
                    held_zncc: attempt.held_zncc,
                },
                None => GrownPlaneView::NotAttempted,
            },
        };
    }
}

/// The keypoints the step writes after the turn, as `(row, pixel)`: the
/// reference's as it was before the step (the anchor's projection where it
/// had none), and each joined view's at the anchor's projection plus its
/// final shift, the pixel its whole-grid score was read around. A pixel off
/// its photograph is not written. Each is recorded in `views` too.
fn keypoint_writes(
    grid: &Grid,
    ref_image: &ViewImage<'_>,
    ref_keypoint: Option<[f64; 2]>,
    reference: usize,
    candidates: &[Candidate<'_>],
    members: &[Member],
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
    for member in members {
        let candidate = &candidates[member.index];
        let pixel = grid
            .anchor_pixel(&candidate.image)
            .map(|at| [at[0] + member.shift[0], at[1] + member.shift[1]])
            .filter(|&p| on_photograph(&candidate.image, p));
        if let Some(p) = pixel {
            writes.push((candidate.row, p));
        }
        if let GrownPlaneView::Joined { keypoint, .. } = &mut views[candidate.row] {
            *keypoint = pixel;
        }
    }
    writes
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

/// `σ` along the weakest and strongest directions of `information`, degrees:
/// `√(2τ/λ)` for each eigenvalue `λ`, floored at `1e-12`, a small tilt
/// `(a, b)` being that many radians.
fn sigmas_deg(information: &Matrix2<f64>, tau: f64) -> [f64; 2] {
    let eigen = information.symmetric_eigen().eigenvalues;
    let (lo, hi) = if eigen[0] <= eigen[1] {
        (eigen[0], eigen[1])
    } else {
        (eigen[1], eigen[0])
    };
    let sigma = |lambda: f64| (2.0 * tau / lambda.max(1e-12)).sqrt().to_degrees();
    [sigma(lo), sigma(hi)]
}

/// The covariance of the tilt that `information` gives: `2τ·(I + 10⁻⁶)⁻¹`,
/// regularised so a direction with no information has a large variance
/// rather than none.
fn covariance(information: &Matrix2<f64>, tau: f64) -> Matrix2<f64> {
    let reg = information + Matrix2::identity() * 1e-6;
    reg.try_inverse()
        .unwrap_or_else(|| Matrix2::identity() * 1e6)
        * (2.0 * tau)
}

/// One view the step may try.
struct Candidate<'a> {
    row: usize,
    image_index: u32,
    image: ViewImage<'a>,
    /// Its keypoint before the step.
    keypoint: Option<[f64; 2]>,
    /// The keypoint less the anchor's projection: the shift the keypoint
    /// implies, the same at every tilt since every plane passes through the
    /// anchor.
    kp_shift: Option<[f64; 2]>,
    angle_deg: f64,
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

/// A view in the shared plane.
struct Member {
    /// Its index among the candidates.
    index: usize,
    shift: [f64; 2],
    /// The growth that joined.
    attempt: GrownPlaneAttempt,
    /// Its information about the tilt, read when it joined.
    information: Matrix2<f64>,
}

/// How the sequence ended.
struct Run {
    tilt: [f64; 2],
    members: Vec<Member>,
    left_out: Vec<(usize, GrownPlaneAttempt)>,
    trace: Vec<GrownPlaneTry>,
}

/// A judgement of one growth against the shared plane.
struct Judged {
    refusal: Option<GrownPlaneRefusal>,
    distance: Option<f64>,
    held_zncc: Option<f64>,
    sigma_deg: f64,
    information: Option<Matrix2<f64>>,
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

    /// Take the views in order until every one has joined or been left out.
    fn run(&self, progress: &Progress<'_>) -> Result<Run, NormalError> {
        let mut queue: std::collections::VecDeque<usize> = (0..self.candidates.len()).collect();
        let mut deferred = BTreeSet::new();
        let mut tilt = [0.0; 2];
        let mut members: Vec<Member> = Vec::new();
        let mut left_out = Vec::new();
        let mut trace = Vec::new();
        while let Some(i) = queue.pop_front() {
            progress.check_cancel()?;
            let candidate = &self.candidates[i];
            let mut phase = progress.phase("grown plane view");
            progress_note!(phase, "image {}", candidate.image_index);

            let first = self.grow(i, tilt, members.is_empty(), &phase)?;
            let mut attempts = vec![self.judge(i, first, tilt, &members)];
            if attempts[0].0.attempt.refusal.is_some() && !members.is_empty() {
                let again = self.grow(i, [0.0; 2], true, &phase)?;
                attempts.push(self.judge(i, again, tilt, &members));
            }
            let joining = attempts
                .iter()
                .position(|(g, _)| g.attempt.refusal.is_none());
            let outcome = match joining {
                Some(k) => {
                    let grown = attempts[k].0.clone();
                    let information = attempts[k].1.information;
                    if members.is_empty() {
                        tilt = grown.tilt;
                    }
                    members.push(Member {
                        index: i,
                        shift: grown.attempt.shift,
                        attempt: grown.attempt.clone(),
                        information: information.unwrap_or_else(Matrix2::zeros),
                    });
                    tilt = self.joint(tilt, &mut members);
                    if members.len() == 1 {
                        // The first view's information is read at the shared
                        // plane, which its own growth found.
                        let m = &mut members[0];
                        let p = [tilt[0], tilt[1], m.shift[0], m.shift[1]];
                        m.information =
                            tilt_information(&self.grid_fit(i, None), &p, &self.iteration);
                    }
                    GrownPlaneTryOutcome::Joined
                }
                None if deferred.insert(i) => {
                    queue.push_back(i);
                    GrownPlaneTryOutcome::Deferred
                }
                None => {
                    let last = attempts.last().expect("one growth at least");
                    left_out.push((i, last.0.attempt.clone()));
                    GrownPlaneTryOutcome::LeftOut
                }
            };
            let shared = members.iter().map(|m| m.information).sum::<Matrix2<f64>>();
            trace.push(GrownPlaneTry {
                observation: candidate.row,
                image: candidate.image_index,
                angle_deg: candidate.angle_deg,
                attempts: attempts.into_iter().map(|(g, _)| g.attempt).collect(),
                outcome,
                shared_normal: (!members.is_empty()).then(|| self.grid.plane_normal(tilt).into()),
                shared_sigma_deg: if members.is_empty() {
                    f64::NAN
                } else {
                    sigmas_deg(&shared, self.options.zncc_tolerance)[0]
                },
            });
        }
        Ok(Run {
            tilt,
            members,
            left_out,
            trace,
        })
    }

    /// Grow candidate `i` from the plane at `tilt0`: the first window's shift
    /// search, the growing windows, and the whole grid's robust fit.
    fn grow(
        &self,
        i: usize,
        tilt0: [f64; 2],
        from_start: bool,
        progress: &Progress<'_>,
    ) -> Result<Grown, NormalError> {
        let render = &self.renders[i];
        let rows = self.grid.window(self.radii[0]);
        let (z1, shift, start) = self.first_window(i, tilt0, &rows);
        let mut p = vec![tilt0[0], tilt0[1], shift[0], shift[1]];
        let mut stage_zncc = vec![z1];
        for &half in &self.radii[1..] {
            progress.check_cancel()?;
            let fit = ViewFit {
                render,
                reference: self.reference,
                live: self.live,
                reading: Reading::Window(self.grid.window(half)),
                held: None,
                min_coverage: self.options.min_coverage,
            };
            let (q, s) = maximise(&fit, &p, &self.iteration);
            p = q;
            stage_zncc.push(s);
        }
        progress.check_cancel()?;
        let (q, score) = maximise(&self.grid_fit(i, None), &p, &self.iteration);
        p = q;
        stage_zncc.push(score);
        let tilt = [p[0], p[1]];
        Ok(Grown {
            tilt,
            attempt: GrownPlaneAttempt {
                from_start,
                start,
                stage_zncc,
                shift: [p[2], p[3]],
                normal: self.grid.plane_normal(tilt).into(),
                grid_zncc: score,
                distance: None,
                held_zncc: None,
                sigma_deg: f64::NAN,
                refusal: None,
            },
        })
    }

    /// The first window's shift at the plane `tilt`: the exhaustive search
    /// about the keypoint-implied shift, widened about zero when it finds
    /// nothing; its ZNCC, the shift and how it was found.
    fn first_window(
        &self,
        i: usize,
        tilt: [f64; 2],
        rows: &[usize],
    ) -> (f64, [f64; 2], GrownPlaneStart) {
        let render = &self.renders[i];
        let pixels = self.grid.project(render.view, tilt, rows);
        let mut samples = Samples::new(self.grid.side * self.grid.side, render.view.channels);
        let step = self.options.spot_search_step_px;
        let mut search = |centre: [f64; 2], reach: f64| {
            let n = (2.0 * reach / step + 1e-9).floor() as i64;
            let mut best = (f64::NEG_INFINITY, centre);
            for iy in 0..=n {
                for ix in 0..=n {
                    let shift = [
                        centre[0] - reach + ix as f64 * step,
                        centre[1] - reach + iy as f64 * step,
                    ];
                    render.view.sample(&pixels, rows, shift, &mut samples);
                    let z = render::zncc(self.reference, &samples, rows, self.options.min_coverage);
                    if z > best.0 {
                        best = (z, shift);
                    }
                }
            }
            best
        };
        let wide = self.options.spot_search_px;
        match self.candidates[i].kp_shift {
            Some(base) => {
                let reach = self.options.keypoint_search_px;
                let (z, t) = search(base, reach);
                let off = (t[0] - base[0]).abs().max((t[1] - base[1]).abs());
                if z >= self.options.min_keypoint_zncc && off < reach - 1e-9 {
                    return (z, t, GrownPlaneStart::Keypoint);
                }
                let (z, t) = search([0.0; 2], wide);
                (z, t, GrownPlaneStart::Widened)
            }
            None => {
                let (z, t) = search([0.0; 2], wide);
                (z, t, GrownPlaneStart::NoKeypoint)
            }
        }
    }

    /// Judge a growth of candidate `i` against the shared plane at `tilt`
    /// over `members`.
    fn judge(
        &self,
        i: usize,
        mut grown: Grown,
        tilt: [f64; 2],
        members: &[Member],
    ) -> (Grown, Judged) {
        let options = self.options;
        let judged = if members.is_empty() {
            Judged {
                refusal: (grown.attempt.grid_zncc < options.min_held_zncc
                    || grown.attempt.grid_zncc.is_nan())
                .then_some(GrownPlaneRefusal::UnderFloor),
                distance: None,
                held_zncc: None,
                sigma_deg: f64::NAN,
                information: None,
            }
        } else {
            let own = &grown.attempt;
            let p = [grown.tilt[0], grown.tilt[1], own.shift[0], own.shift[1]];
            let information = tilt_information(&self.grid_fit(i, None), &p, &self.iteration);
            let shared = members.iter().map(|m| m.information).sum::<Matrix2<f64>>();
            let d = Vector2::new(grown.tilt[0] - tilt[0], grown.tilt[1] - tilt[1]);
            let c = covariance(&information, options.zncc_tolerance)
                + covariance(&shared, options.zncc_tolerance);
            let d2 = c
                .try_inverse()
                .map_or(f64::INFINITY, |inv| (d.transpose() * inv * d)[(0, 0)]);
            let distance = d2.max(0.0).sqrt();
            let (_, held) = maximise(&self.grid_fit(i, Some(tilt)), &own.shift, &self.iteration);
            let bar = options.max_agreement_distance;
            let refusal = if distance.is_nan() || distance > bar {
                Some(GrownPlaneRefusal::TooFar)
            } else if held.is_nan() || held < options.min_held_zncc {
                Some(GrownPlaneRefusal::HeldUnderFloor)
            } else {
                None
            };
            Judged {
                refusal,
                distance: Some(distance),
                held_zncc: Some(held),
                sigma_deg: sigmas_deg(&information, options.zncc_tolerance)[0],
                information: Some(information),
            }
        };
        grown.attempt.refusal = judged.refusal;
        grown.attempt.distance = judged.distance;
        grown.attempt.held_zncc = judged.held_zncc;
        grown.attempt.sigma_deg = judged.sigma_deg;
        (grown, judged)
    }

    /// Fit the shared tilt and every member's shift together, from `tilt`
    /// and the members' current shifts; the tilt reached. The members'
    /// shifts are updated.
    fn joint(&self, tilt: [f64; 2], members: &mut [Member]) -> [f64; 2] {
        let fit = JointFit {
            views: members
                .iter()
                .map(|m| self.grid_fit(m.index, None))
                .collect(),
        };
        let mut p = vec![tilt[0], tilt[1]];
        for m in members.iter() {
            p.extend_from_slice(&m.shift);
        }
        let (q, _) = maximise(&fit, &p, &self.iteration);
        for (k, m) in members.iter_mut().enumerate() {
            m.shift = [q[2 + 2 * k], q[3 + 2 * k]];
        }
        [q[0], q[1]]
    }
}

#[cfg(test)]
mod tests;
