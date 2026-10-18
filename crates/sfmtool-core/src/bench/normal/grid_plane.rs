// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! [`grid_plane_normal`]: the patch's normal from the plane through the nine
//! cells of the ZNCC grid, each registered by a shift search on one render
//! per view, gated in each view, and triangulated from the views it survived
//! in.
//!
//! Each `in` view is rendered once at the track's current patch frame as a context
//! tile: the patch re-anchored in its plane on the view's keypoint and drawn
//! through that plane ([`WarpMap::from_patch`]) on the working grid plus a
//! margin of the cell search bound, the render the keypoint localizer makes
//! for its context tiles. The reference view's render, without the margin,
//! gives the nine templates. The cell registration kernel
//! (`crate::patch::cell_registration`) registers and gates every cell of
//! every other view and fits each view's robust affine map; the cell plane
//! kernel (`crate::patch::cell_plane_fit`) triangulates the cells that
//! survive and fits the plane with its determinacy verdict. The patch is
//! turned toward that plane with [`tilt_patch`] and measured again, under the
//! acceptance and stopping rules of the piecewise refinement's loop.
//!
//! `specs/core/bench/editable-track.md` § "Estimating the normal" describes
//! the step; `specs/drafts/piece-gated-grid-normal.md` is the design it is
//! built from.

use std::cell::RefCell;

use nalgebra::Vector3;

use crate::camera::sampler::{render_phase, render_tile};
use crate::camera::warp_map::{patch_grid_jacobian, singular_values_2x2};
use crate::camera::WarpMap;
use crate::patch::cell_plane_fit::{
    fit_cell_plane_recorded, normal_precision_deg, CellPlaneRecord, CellRay, PosedView,
};
use crate::patch::cell_plane_normals::{
    CellPlaneNormal, CellPlaneParams, CellPlaneStatus, NormalDeterminacy,
};
use crate::patch::cell_registration::{
    fit_pass, read_cells, CellGateOptions, CellLayout, CellReading, CellTemplate, CellTemplates,
    Pass, UpdateModel, WorkingPatch,
};
use crate::patch::cloud::OrientedPatch;
use crate::patch::cluster_refine::{
    CellStatus, ClusterRefineParams, LoopStop, PiecewiseParams, ACCEPT_ZNCC_TOLERANCE,
};
use crate::patch::keypoint_localize::{resolve_reference, shifted_center};
use crate::patch::normal_refine::{window_weights, ProjectedImage};
use crate::patch::reference_view::ReferenceViewOptions;
use crate::progress::Progress;
use crate::progress_note;
use crate::reconstruction::edited::EditedReconstruction;

use super::super::evaluate::{check_views, seed_of, stored_patch_resolution, track_reference};
use super::super::fit::FitOptions;
use super::super::steps::tilt_patch;
use super::super::track::EditableTrack;
use super::{
    normal_preconditions, patch_frame_of, perpendicular_part, turn_and_read, NormalError,
    NormalEstimate, NormalReport,
};

mod detail;

pub use detail::{
    GridPlaneCell, GridPlaneDetail, GridPlaneFitModel, GridPlaneLayout, GridPlanePatchFrame,
    GridPlanePlane, GridPlaneRay, GridPlaneRender, GridPlaneTile, GridPlaneView, GridPlaneViewCell,
    GridPlaneViewFit,
};

/// The smallest working grid the step renders: three samples a side are the
/// fewest that cut into nine cells.
pub(super) const MIN_WORKING_RESOLUTION: u32 = 3;

/// A sample variance under this is flat for the acceptance score.
const FLAT_VARIANCE: f64 = 1e-9;

/// The bench's default [`GridPlaneNormalOptions::min_cell_zncc`].
///
/// `0.6`, below the piecewise refinement's own
/// [`DEFAULT_MIN_CELL_ZNCC`](crate::patch::cluster_refine::DEFAULT_MIN_CELL_ZNCC)
/// (`0.8`), which stays the cluster pass's bar. Chosen by the maintainer from
/// a run over every point of seoul_bull, kerry_park and a candidate ground
/// truth of dino_dog_toy: many cells refused under `0.8` looked like usable
/// data, and at `0.6` the step refused far fewer cells, found a plane at more
/// points and turned more of them, with the first plane no further from the
/// stored normals at wide baselines and closer on dino_dog_toy at narrow ones.
/// Below `0.6` the first plane moved further from the stored normals.
/// `specs/drafts/piece-gated-grid-normal.md` gives the numbers.
pub const BENCH_MIN_CELL_ZNCC: f32 = 0.6;

/// How [`grid_plane_normal`] measures the cells and when its loop stops.
#[derive(Debug, Clone)]
pub struct GridPlaneNormalOptions {
    /// Search bound for a cell's shift, grid px of the working grid; also the
    /// margin, rounded up, each view's context tile is rendered with. The
    /// piecewise refinement's
    /// [`cell_shift_bound_px`](PiecewiseParams::cell_shift_bound_px), 2.0.
    pub cell_shift_bound_px: f32,
    /// A cell whose ZNCC at its optimum is below this is refused in that view.
    /// [`BENCH_MIN_CELL_ZNCC`], 0.6.
    pub min_cell_zncc: f32,
    /// A cell whose ZNCC peak is flatter than this along its flattest
    /// direction is refused in that view, in ZNCC per grid px².
    /// [`DEFAULT_MIN_CELL_CURVATURE`](crate::patch::cluster_refine::DEFAULT_MIN_CELL_CURVATURE),
    /// 0.02.
    pub min_cell_curvature: f32,
    /// Side of the working grid the views are rendered on, in samples, at
    /// least 3. `None` renders at about one sample per photograph pixel of
    /// the reference view: the patch's longer side as the reference view
    /// sees it at the starting patch frame, rounded up and clamped to `R ..= 2R`
    /// (see `working_side`), with `R` the reconstruction's own patch
    /// resolution where it stores patch bitmaps and otherwise the fit's
    /// sub-pixel resolution, the grid the track's bitmap is rendered on.
    pub working_resolution: Option<u32>,
    /// Triangulation, floors, Tukey loop and determinacy, as the cell plane
    /// normal kernel reads them.
    pub plane: CellPlaneParams,
    /// The loop stops when a turn moves no cell centre by more than this, in
    /// grid px.
    pub update_tolerance_px: f32,
    /// The most renders the loop makes, at least 1. A turn is kept only once
    /// a render at the turned patch frame has scored it, so the last render's own
    /// turn is not taken unless it is under
    /// [`Self::update_tolerance_px`].
    pub max_iterations: u8,
    /// Which rule decides whether a rendered turn is kept.
    /// [`GridPlaneAcceptance::WindowedZncc`] by default.
    pub acceptance: GridPlaneAcceptance,
    /// A plane whose predicted precision
    /// ([`GridPlaneIteration::normal_precision_deg`]) is worse than this, in
    /// degrees, or not finite, proposes no turn, so a loop that meets one ends
    /// [`LoopStop::Measured`] and the step [`GridPlaneOutcome::Undetermined`].
    /// `None`, the default, gates nothing.
    pub max_normal_precision_deg: Option<f64>,
    /// The reading the step ends with, the sampler the context tiles are
    /// rendered with (`refine.sampler`), and the reference-view rule's
    /// grazing gate and sampler (`localize`).
    pub fit: FitOptions,
    /// Keep a [`GridPlaneDetail`] of every render in
    /// [`GridPlaneCells::detail`]: the tiles, every cell's reading in every
    /// view, each view's affine fit, each cell's rays and triangulation, and
    /// the plane. Off by default; the step's result does not depend on it.
    pub record_detail: bool,
}

impl Default for GridPlaneNormalOptions {
    fn default() -> Self {
        let piecewise = PiecewiseParams::default();
        Self {
            cell_shift_bound_px: piecewise.cell_shift_bound_px,
            min_cell_zncc: BENCH_MIN_CELL_ZNCC,
            min_cell_curvature: piecewise.min_cell_curvature,
            working_resolution: None,
            plane: CellPlaneParams::default(),
            update_tolerance_px: piecewise.update_tolerance_px,
            max_iterations: piecewise.max_iterations,
            acceptance: GridPlaneAcceptance::WindowedZncc,
            max_normal_precision_deg: None,
            fit: FitOptions::default(),
            record_detail: false,
        }
    }
}

/// Which rule [`grid_plane_normal`]'s loop keeps a rendered turn by.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GridPlaneAcceptance {
    /// The piecewise loop's rule: a turn is kept only when the mean windowed
    /// ZNCC of the views against the reference does not fall by more than
    /// [`ACCEPT_ZNCC_TOLERANCE`], nor below its starting value
    /// ([`GridPlaneCells::score_before`]).
    WindowedZncc,
    /// Every rendered turn is kept; the loop stops only on the tolerance, the
    /// render cap, an oscillation (keeping the later patch frame), or a plane that
    /// proposes no turn. For measuring what the planes propose without the
    /// score's veto.
    None,
}

/// One render of [`grid_plane_normal`]'s loop and the turn its cells asked
/// for: the loop's trace, in render order.
#[derive(Debug, Clone, PartialEq)]
pub struct GridPlaneIteration {
    /// The acceptance score at this render; `NaN` when no view scored.
    pub score: f64,
    /// Which axes of the normal this render's cells fixed.
    pub determinacy: NormalDeterminacy,
    /// How many cells the plane fit gave weight.
    pub in_plane: usize,
    /// The precision the plane fit's weights predict for the normal, degrees:
    /// `σ = 1 / √(Σⱼ wⱼ (dⱼ · e)²)` along the plane's weaker in-plane axis `e`
    /// under [`NormalDeterminacy::BothAxes`], and along the line the cells
    /// lie on, the one axis they fix, under [`NormalDeterminacy::OneAxis`];
    /// `NaN` under [`NormalDeterminacy::None`].
    pub normal_precision_deg: f64,
    /// Whether [`GridPlaneNormalOptions::max_normal_precision_deg`] stopped
    /// this render's turn from being proposed.
    pub gated: bool,
    /// The outward normal the proposed turn reaches, world frame; `None`
    /// when this render proposed no turn (no axis fixed, gated, or the loop
    /// stopped at this render before proposing).
    pub proposed_normal: Option<[f64; 3]>,
    /// The farthest the proposed turn moves a cell centre, grid px; `NaN`
    /// with no turn.
    pub largest_move_px: f64,
    /// Whether the loop took the proposed turn: under the tolerance, or
    /// rendered and kept. A turn taken in a loop that does not end
    /// [`LoopStop::Converged`] is still not applied to the track (see
    /// [`GridPlaneOutcome`]).
    pub taken: bool,
}

/// What the cells said, carried by [`NormalEstimate::GridPlaneCells`]. Every
/// reading is of the last render the loop kept: when the step turned the
/// patch, the render one turn of under
/// [`GridPlaneNormalOptions::update_tolerance_px`] before the patch frame it turned
/// to; otherwise the render at the patch frame the loop had reached, which the step
/// does not keep (see [`GridPlaneOutcome`]).
#[derive(Debug, Clone, PartialEq)]
pub struct GridPlaneCells {
    /// Which axes of the normal the cells fixed.
    pub determinacy: NormalDeterminacy,
    /// What became of each cell in the triangulation and the plane fit,
    /// `[row * 3 + col]` from the top-left cell.
    pub cell_status: [CellPlaneStatus; 9],
    /// The rays each cell was triangulated from, the reference's among them.
    pub cell_rays: [u32; 9],
    /// Each cell's weight in the plane fit; `0` where it has none.
    pub cell_weight: [f64; 9],
    /// Per observation of the track, each cell's registration status in that
    /// view. The reference's row reads [`CellStatus::Fitted`] for a cell its
    /// template can be registered with, [`CellStatus::RefusedCurvature`] for
    /// a flat one and [`CellStatus::NotAttempted`] for one partly off its
    /// photograph; a row that is not `in`, repeats an image, or could not be
    /// rendered reads [`CellStatus::NotAttempted`] throughout.
    pub view_cells: Vec<[CellStatus; 9]>,
    /// The weighted rms distance of the cells from their plane, world units;
    /// `NaN` when no plane was fitted.
    pub plane_rms: f64,
    /// Signed distance of the fitted plane from the patch centre along the
    /// fitted normal, world units; `NaN` when the cells fixed no axis. The
    /// step does not move the centre.
    pub plane_offset: f64,
    /// The side of the working grid every render was made on, in samples.
    pub working_resolution: u32,
    /// The renders the loop made.
    pub iterations: u8,
    /// Why the loop stopped. [`LoopStop::NotRun`] when no reference could be
    /// rendered or no view could be scored against it, and
    /// [`LoopStop::Measured`] when the last render's cells fixed no axis, so
    /// no turn was proposed.
    pub stop: LoopStop,
    /// The acceptance score at the patch frame the step started from: the mean over
    /// the non-reference views of the windowed ZNCC of each view's tile against
    /// the reference's render, with the cluster refinement's default window.
    /// `NaN` when no view scored.
    pub score_before: f64,
    /// The acceptance score at the last render the loop kept, the render the
    /// other readings are of.
    pub score_after: f64,
    /// Whether the step turned the patch, and if not, why not.
    pub outcome: GridPlaneOutcome,
    /// The predicted precision of the last kept render's plane, degrees, as
    /// [`GridPlaneIteration::normal_precision_deg`].
    pub normal_precision_deg: f64,
    /// The views' baseline: the largest angle between two of the rendered
    /// `in` sightings' directions from the patch centre at the starting
    /// patch frame, degrees; `NaN` with fewer than two.
    pub view_baseline_deg: f64,
    /// Every render of the loop, in order.
    pub trace: Vec<GridPlaneIteration>,
    /// What the step read and decided at every render, with
    /// [`GridPlaneNormalOptions::record_detail`]; `None` without it.
    pub detail: Option<Box<GridPlaneDetail>>,
}

/// What [`grid_plane_normal`] made of the cells: whether it turned the patch
/// and, where it did not, why.
///
/// Only [`Self::Turned`] moves the normal. The other two leave the track at
/// the normal it was given, not at a patch frame part way along the loop, and the
/// step still reads the track back and returns it: neither is a refusal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GridPlaneOutcome {
    /// The loop converged ([`LoopStop::Converged`]) with at least one axis
    /// fixed, and the patch was turned: to the plane's normal under both
    /// axes, about the fixed axis only under one.
    Turned,
    /// The cells cannot constrain the normal: the last render's cells fixed
    /// no axis (too few live cells, or no spread; [`LoopStop::Measured`]), or
    /// no view could be scored against the reference ([`LoopStop::NotRun`]).
    /// Lack of evidence.
    Undetermined,
    /// The cells constrain the normal but the loop did not settle: a turn
    /// the acceptance rule rejected ([`LoopStop::Rejected`]), an oscillation
    /// ([`LoopStop::Oscillation`]), or the render cap without convergence
    /// ([`LoopStop::Cap`]), even with every turn before it accepted. It
    /// suggests a patch over two surfaces, a wrong correspondence on
    /// repeating texture, or a surface that is not planar.
    Unstable,
}

impl GridPlaneOutcome {
    /// The outcome of a loop that stopped for `stop`.
    pub fn of(stop: LoopStop) -> GridPlaneOutcome {
        match stop {
            LoopStop::Converged => GridPlaneOutcome::Turned,
            LoopStop::NotRun | LoopStop::Measured => GridPlaneOutcome::Undetermined,
            LoopStop::Rejected | LoopStop::Oscillation | LoopStop::Cap => {
                GridPlaneOutcome::Unstable
            }
        }
    }
}

impl GridPlaneCells {
    /// How many cells the plane fit gave weight.
    pub fn in_plane(&self) -> usize {
        self.cell_status
            .iter()
            .filter(|&&s| s == CellPlaneStatus::InPlane)
            .count()
    }
}

impl std::fmt::Display for GridPlaneCells {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} of 9 cells in the plane", self.in_plane())?;
        match self.determinacy {
            NormalDeterminacy::BothAxes => write!(
                f,
                ", both axes fixed, rms {:.4} off the plane",
                self.plane_rms
            )?,
            NormalDeterminacy::OneAxis { .. } => write!(f, ", one axis fixed")?,
            NormalDeterminacy::None => write!(f, ", no axis fixed")?,
        }
        let stop = match self.stop {
            LoopStop::NotRun => "not run",
            LoopStop::Converged => "converged",
            LoopStop::Cap => "at the render cap",
            LoopStop::Rejected => "last turn rejected",
            LoopStop::Oscillation => "oscillating",
            LoopStop::Measured => "nothing to turn to",
        };
        let renders = if self.iterations == 1 {
            "render"
        } else {
            "renders"
        };
        write!(f, ", {} {renders}, {stop}", self.iterations)?;
        match self.outcome {
            GridPlaneOutcome::Turned => Ok(()),
            GridPlaneOutcome::Undetermined => write!(f, ", normal left unchanged: undetermined"),
            GridPlaneOutcome::Unstable => write!(f, ", normal left unchanged: unstable"),
        }
    }
}

/// Turn the track's patch to the plane through its cells.
///
/// The patch's square is cut into the nine cells of the ZNCC grid. Each `in`
/// view is rendered once at the current patch frame, centred on its keypoint and
/// drawn through the patch's plane, and each cell of it is registered against
/// the same cell of the reference view's render by a shift search within
/// [`GridPlaneNormalOptions::cell_shift_bound_px`]. A cell is refused in a view
/// for curvature, for ZNCC, for the bound, or as an outlier of the view's
/// robust affine fit, by the piecewise refinement's rules. Every surviving
/// cell is a ray through the pixel its registered position maps to, the
/// reference contributes a ray through each cell's centre, and the cells are
/// triangulated and fitted with a weighted, robust plane that says whether it
/// fixes both axes of the normal, one, or none.
///
/// Under both axes the patch is turned to the plane's normal, signed to keep
/// the face the patch shows; under one, only about the axis the cells fix,
/// keeping its rotation about the line they lie along; under none, not at
/// all. Each turn is rendered and measured again, and kept only when the mean
/// windowed ZNCC of the views against the reference does not fall by more
/// than [`ACCEPT_ZNCC_TOLERANCE`], nor below its starting value. The loop
/// stops when a turn moves no cell centre by more than
/// [`GridPlaneNormalOptions::update_tolerance_px`] (that turn is taken), at
/// [`GridPlaneNormalOptions::max_iterations`] renders, when a turn moves the
/// cells no less than the one before (the better-scoring of the two patch frames
/// is kept), or when a turn is rejected.
///
/// Only a loop that converges turns the patch ([`GridPlaneOutcome::Turned`]).
/// Every other ending leaves the normal as it was before the step, even when
/// earlier turns were accepted: [`GridPlaneOutcome::Undetermined`] when the
/// cells fix no axis or no view can be scored, [`GridPlaneOutcome::Unstable`]
/// when the loop stops on a rejected turn, an oscillation or the render cap.
///
/// The step ends as the other normal steps do: the original track is turned
/// to the converged patch frame's normal, or to its own, with [`tilt_patch`] and
/// read back, its bitmap rendered. The centre does not move.
///
/// # Example
///
/// ```no_run
/// # use sfmtool_core::bench::{grid_plane_normal, GridPlaneNormalOptions};
/// # use sfmtool_core::progress::Progress;
/// # fn run(
/// #     track: &sfmtool_core::bench::EditableTrack,
/// #     edited: &sfmtool_core::EditedReconstruction,
/// #     images: &[sfmtool_core::patch::normal_refine::ProjectedImage<'_>],
/// # ) -> Result<(), Box<dyn std::error::Error>> {
/// let (turned, report) = grid_plane_normal(
///     track,
///     edited,
///     images,
///     &GridPlaneNormalOptions::default(),
///     &Progress::none(),
/// )?;
/// println!("{report}"); // "9 of 9 cells in the plane, both axes fixed, …"
/// # let _ = turned;
/// # Ok(())
/// # }
/// ```
pub fn grid_plane_normal(
    track: &EditableTrack,
    edited: &EditedReconstruction,
    images: &[ProjectedImage<'_>],
    options: &GridPlaneNormalOptions,
    progress: &Progress<'_>,
) -> Result<(EditableTrack, NormalReport), NormalError> {
    check_views(edited, images)?;
    normal_preconditions(track)?;
    let base = stored_patch_resolution(&edited.base)
        .unwrap_or(options.fit.refine.resolution)
        .max(MIN_WORKING_RESOLUTION);
    let cells = Cells::new(track, images, base, options, progress);
    // One entry per render: `measure` opens it, `propose` fills in the turn,
    // and the loop's decisions mark the turns it took.
    let trace: RefCell<Vec<GridPlaneIteration>> = RefCell::new(Vec::new());
    // One entry per render, when the detail is recorded.
    let renders: RefCell<Vec<GridPlaneRender>> = RefCell::new(Vec::new());
    let end = {
        let mut phase = progress.phase("grid plane normal");
        let tolerance = f64::from(options.update_tolerance_px);
        let end = turn_loop(
            track.clone(),
            options.max_iterations,
            tolerance,
            options.acceptance,
            |t: &EditableTrack| {
                let mut m = cells.measure(t)?;
                if let Some(render) = m.record.take() {
                    renders.borrow_mut().push(*render);
                }
                trace.borrow_mut().push(GridPlaneIteration {
                    score: m.score,
                    determinacy: m.plane.determinacy,
                    in_plane: in_plane_count(&m.plane),
                    normal_precision_deg: m.precision_deg,
                    gated: false,
                    proposed_normal: None,
                    largest_move_px: f64::NAN,
                    taken: false,
                });
                Ok::<_, NormalError>(m)
            },
            |m: &Measurement| m.score,
            |t: &EditableTrack, m: &Measurement| {
                let proposal = cells.propose(t, edited, m)?;
                let mut trace = trace.borrow_mut();
                let entry = trace.last_mut().expect("a render precedes every proposal");
                match &proposal {
                    Proposal::Turn { normal, moved, .. } => {
                        entry.proposed_normal = Some(*normal);
                        entry.largest_move_px = *moved;
                    }
                    Proposal::Gated => entry.gated = true,
                    Proposal::Nothing => {}
                }
                Ok(proposal.into_turn())
            },
        )?;
        progress_note!(phase, "{} renders", end.iterations);
        end
    };
    // The loop's decisions, one per proposal, in the order it met them; a
    // proposal is always made from the latest render.
    let mut trace = trace.into_inner();
    let mut decisions = end.taken.iter();
    for entry in trace.iter_mut().filter(|e| e.proposed_normal.is_some()) {
        entry.taken = decisions.next().copied().unwrap_or(false);
    }
    let detail = options.record_detail.then(|| {
        let mut renders = renders.into_inner();
        for (render, entry) in renders.iter_mut().zip(&trace) {
            render.iteration = entry.clone();
        }
        Box::new(GridPlaneDetail {
            layout: cells.layout_record(),
            renders,
        })
    });

    let kept = patch_frame_of(&end.state);
    // Only a converged loop turns the patch; any other ending leaves the
    // normal the step was given.
    let outcome = GridPlaneOutcome::of(end.stop);
    let normal = match outcome {
        GridPlaneOutcome::Turned => kept.normal(),
        GridPlaneOutcome::Undetermined | GridPlaneOutcome::Unstable => {
            patch_frame_of(track).normal()
        }
    };
    let measured = &end.measured;
    let report = GridPlaneCells {
        determinacy: measured.plane.determinacy,
        cell_status: measured.plane.cell_status,
        cell_rays: measured.plane.cell_rays,
        cell_weight: measured.plane.cell_weight,
        view_cells: measured.view_cells.clone(),
        plane_rms: measured.plane.plane_rms,
        plane_offset: plane_offset(&measured.plane, &kept),
        working_resolution: cells.layout.resolution as u32,
        iterations: end.iterations,
        stop: end.stop,
        score_before: end.start_score,
        score_after: measured.score,
        outcome,
        normal_precision_deg: measured.precision_deg,
        view_baseline_deg: cells.baseline_deg,
        trace,
        detail,
    };
    turn_and_read(
        track,
        edited,
        images,
        normal,
        NormalEstimate::GridPlaneCells(Box::new(report)),
        &options.fit,
        progress,
    )
}

/// What every render of one [`grid_plane_normal`] shares.
struct Cells<'a, 'v> {
    images: &'a [ProjectedImage<'v>],
    options: &'a GridPlaneNormalOptions,
    progress: &'a Progress<'a>,
    layout: CellLayout,
    gates: CellGateOptions,
    /// The `in` rows the step renders, one per image, in row order.
    rows: Vec<usize>,
    /// The reference row, from `rows`; `None` when the rule found none.
    reference: Option<usize>,
    /// The farthest a cell centre lies from the grid's middle, grid px.
    reach: f64,
    /// The acceptance score's window over the working grid's `R×R` middle:
    /// the cluster refinement's default window, which the piecewise loop's
    /// whole-member ZNCC reads with.
    window: Vec<f64>,
    /// The largest angle between two rows' sighting directions from the
    /// starting patch frame's centre, degrees.
    baseline_deg: f64,
}

/// What one render's cells ask of the loop.
enum Proposal {
    /// Turn to `state`, which reaches `normal` and moves the farthest cell
    /// centre `moved` grid px.
    Turn {
        state: Box<EditableTrack>,
        normal: [f64; 3],
        moved: f64,
    },
    /// The cells fix an axis, but the plane's predicted precision fails
    /// [`GridPlaneNormalOptions::max_normal_precision_deg`].
    Gated,
    /// The cells fix no axis.
    Nothing,
}

impl Proposal {
    /// The turn as [`turn_loop`] takes it.
    fn into_turn(self) -> Option<(EditableTrack, f64)> {
        match self {
            Proposal::Turn { state, moved, .. } => Some((*state, moved)),
            Proposal::Gated | Proposal::Nothing => None,
        }
    }
}

/// How many cells `plane` gave weight.
fn in_plane_count(plane: &CellPlaneNormal) -> usize {
    plane
        .cell_status
        .iter()
        .filter(|&&s| s == CellPlaneStatus::InPlane)
        .count()
}

/// One render of every view at one patch frame, and what the cells said.
struct Measurement {
    plane: CellPlaneNormal,
    view_cells: Vec<[CellStatus; 9]>,
    /// The acceptance score, `NaN` when no view scored.
    score: f64,
    /// The plane's predicted precision, degrees, along the axis its verdict
    /// turns (see [`GridPlaneIteration::normal_precision_deg`]).
    precision_deg: f64,
    /// The render's detail, with [`GridPlaneNormalOptions::record_detail`];
    /// the loop moves it into the report as the render is made.
    record: Option<Box<GridPlaneRender>>,
}

impl Measurement {
    /// The measurement of a render whose cells `plane` fitted, finishing
    /// `record`'s plane where it is kept.
    fn new(
        plane: CellPlaneNormal,
        view_cells: Vec<[CellStatus; 9]>,
        score: f64,
        mut record: Option<Box<GridPlaneRender>>,
    ) -> Self {
        let [weak, strong] = normal_precision_deg(&plane);
        let precision_deg = match plane.determinacy {
            NormalDeterminacy::BothAxes => weak,
            NormalDeterminacy::OneAxis { .. } => strong,
            NormalDeterminacy::None => f64::NAN,
        };
        if let Some(record) = record.as_deref_mut() {
            record.plane.precision_deg = [weak, strong];
        }
        Measurement {
            plane,
            view_cells,
            score,
            precision_deg,
            record,
        }
    }
}

/// One view's context tile.
struct Render {
    /// The patch re-anchored on the view's keypoint, at the working grid's
    /// size: what the cell positions are read on.
    patch_frame: OrientedPatch,
    /// `side × side × channels`, interleaved, `NaN` off the photograph.
    samples: Vec<f32>,
    side: usize,
    channels: usize,
}

impl<'a, 'v> Cells<'a, 'v> {
    /// The rows, the reference and the working grid of one step, chosen
    /// once from the starting patch frame. `base` is the resolution the reference
    /// rule renders at and the working grid's floor, `R`.
    fn new(
        track: &EditableTrack,
        images: &'a [ProjectedImage<'v>],
        base: u32,
        options: &'a GridPlaneNormalOptions,
        progress: &'a Progress<'a>,
    ) -> Self {
        let gates = CellGateOptions {
            min_cell_zncc: options.min_cell_zncc,
            min_cell_curvature: options.min_cell_curvature,
        };
        // One row per image, as the photometric step reads them: a second
        // sighting in one image adds no view.
        let mut seen = std::collections::BTreeSet::new();
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
        // The fit's reference: the one the track holds where its row is `in`
        // with a keypoint, and otherwise the reference-view rule's pick.
        let patch_frame = patch_frame_of(track);
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
            progress,
        )
        .map(|k| rows[k]);
        // The working grid, chosen once so that every render of the loop is
        // on one grid: the caller's, or the reference view's footprint of the
        // patch at the starting patch frame, at its keypoint.
        let resolution = match options.working_resolution {
            Some(side) => side.max(MIN_WORKING_RESOLUTION),
            None => {
                let jacobian = reference.and_then(|r| {
                    let observation = &track.observations[r];
                    let view = &images[observation.image as usize];
                    let anchored_patch_frame = seed_of(observation)
                        .and_then(|kp| {
                            patch_frame.anchored_at_keypoint(view.camera, view.cam_from_world, kp)
                        })
                        .unwrap_or_else(|| patch_frame.clone());
                    patch_grid_jacobian(
                        &anchored_patch_frame,
                        view.camera,
                        view.cam_from_world,
                        base as usize,
                    )
                });
                working_side(jacobian, base)
            }
        };
        let layout = CellLayout::new(resolution, options.cell_shift_bound_px);
        let reach = layout
            .centres
            .iter()
            .flatten()
            .map(|c| c[0].hypot(c[1]))
            .fold(0.0, f64::max);
        let baseline_deg = view_baseline_deg(
            patch_frame.center.coords,
            rows.iter().map(|&i| {
                images[track.observations[i].image as usize]
                    .cam_from_world
                    .inverse_translation_origin()
                    .coords
            }),
        );
        Self {
            images,
            options,
            progress,
            layout,
            gates,
            rows,
            reference,
            reach,
            window: window_weights(ClusterRefineParams::default().window, resolution),
            baseline_deg,
        }
    }

    /// The working grid and its cells as the detail record states them.
    fn layout_record(&self) -> GridPlaneLayout {
        let layout = &self.layout;
        GridPlaneLayout {
            working_resolution: layout.resolution as u32,
            margin: layout.margin as u32,
            reach: layout.reach as u32,
            cell_bounds: layout.bounds.map(|b| b as u32),
            cell_centres: std::array::from_fn(|k| layout.centres[k / 3][k % 3]),
        }
    }

    /// Render every view at `track`'s patch frame, register and gate its cells,
    /// triangulate them and fit the plane, and score the views against the
    /// reference.
    fn measure(&self, track: &EditableTrack) -> Result<Measurement, NormalError> {
        let patch_frame = patch_frame_of(track);
        let r = self.layout.resolution;
        let wpp = 2.0 * patch_frame.half_extent[0] / r as f64;
        let params = &self.options.plane;
        let mut view_cells = vec![[CellStatus::NotAttempted; 9]; track.observations.len()];
        let mut rays: [Vec<CellRay>; 9] = Default::default();
        // The render's detail, only when it is asked for.
        let mut record = self
            .options
            .record_detail
            .then(|| Box::new(empty_render(&patch_frame)));
        let mut plane_record = record.as_ref().map(|_| CellPlaneRecord::empty());
        let nothing = |view_cells, mut record: Option<Box<GridPlaneRender>>| {
            let plane = fit_cell_plane_recorded(&Default::default(), params, None);
            if let Some(record) = record.as_deref_mut() {
                record_plane(record, &plane, &CellPlaneRecord::empty());
            }
            Measurement::new(plane, view_cells, f64::NAN, record)
        };

        let Some(reference) = self.reference else {
            return Ok(nothing(view_cells, record));
        };
        self.progress.check_cancel()?;
        let ref_observation = &track.observations[reference];
        let ref_view = &self.images[ref_observation.image as usize];
        let Some(ref_render) = self.render(&patch_frame, ref_view, seed_of(ref_observation), 0)
        else {
            return Ok(nothing(view_cells, record));
        };
        if let Some(record) = record.as_deref_mut() {
            record.reference = Some(tile_record(reference, ref_observation.image, &ref_render));
        }
        let planes = planar(&ref_render.samples, r * r, ref_render.channels);
        let templates = CellTemplates::from_planes(&planes, ref_render.channels, &self.layout);
        let src_channels: Vec<usize> = (0..ref_render.channels).collect();
        let ref_posed = PosedView::new(ref_view.camera, ref_view.cam_from_world);
        for (k, centre) in self.layout.centres.iter().flatten().enumerate() {
            let status = template_status(&templates.cells[k / 3][k % 3]);
            view_cells[reference][k] = status;
            if status == CellStatus::Fitted {
                if let Some((ray, pixel)) = cell_ray(
                    &ref_render.patch_frame,
                    ref_view,
                    &ref_posed,
                    *centre,
                    wpp,
                    params,
                ) {
                    if let Some(record) = record.as_deref_mut() {
                        record.cells[k].rays.push(ray_record(
                            reference,
                            ref_observation.image,
                            &ray,
                            pixel,
                        ));
                    }
                    rays[k].push(ray);
                }
            }
        }

        let mut scores: Vec<f64> = Vec::with_capacity(self.rows.len());
        for &row in &self.rows {
            if row == reference {
                continue;
            }
            // The cancel flag is polled per render: the render is the one read
            // of the photograph.
            self.progress.check_cancel()?;
            let observation = &track.observations[row];
            let view = &self.images[observation.image as usize];
            let Some(render) =
                self.render(&patch_frame, view, seed_of(observation), self.layout.margin)
            else {
                continue;
            };
            let view_score = tile_zncc(
                &planes,
                ref_render.channels,
                &render,
                self.layout.margin,
                &self.window,
            );
            if let Some(z) = view_score {
                scores.push(z);
            }
            let patch = WorkingPatch::new(render.side, render.channels, render.samples.clone());
            let readings = read_cells(&templates, &src_channels, &patch, &self.layout, &self.gates);
            let (statuses, pass) = gated(readings, &self.layout);
            if let Some(record) = record.as_deref_mut() {
                record.views.push(view_record(
                    tile_record(row, observation.image, &render),
                    view_score,
                    &readings,
                    &statuses,
                    pass.as_ref(),
                ));
            }
            let posed = PosedView::new(view.camera, view.cam_from_world);
            for (k, (status, reading)) in statuses
                .iter()
                .flatten()
                .zip(readings.iter().flatten())
                .enumerate()
            {
                view_cells[row][k] = *status;
                let usable = *status == CellStatus::Fitted
                    || (params.include_refused_outlier && *status == CellStatus::RefusedOutlier);
                if !usable {
                    continue;
                }
                let c = self.layout.centres[k / 3][k % 3];
                let at = [c[0] + reading.shift[0], c[1] + reading.shift[1]];
                if let Some((ray, pixel)) =
                    cell_ray(&render.patch_frame, view, &posed, at, wpp, params)
                {
                    if let Some(record) = record.as_deref_mut() {
                        record.cells[k]
                            .rays
                            .push(ray_record(row, observation.image, &ray, pixel));
                    }
                    rays[k].push(ray);
                }
            }
        }
        let score = if scores.is_empty() {
            f64::NAN
        } else {
            scores.iter().sum::<f64>() / scores.len() as f64
        };
        let plane = fit_cell_plane_recorded(&rays, params, plane_record.as_mut());
        if let (Some(record), Some(plane_record)) = (record.as_deref_mut(), &plane_record) {
            record_plane(record, &plane, plane_record);
        }
        Ok(Measurement::new(plane, view_cells, score, record))
    }

    /// The turn the cells of `measured` ask of `track`, with the normal it
    /// reaches and the farthest it moves a cell centre, in grid px; none when
    /// they fix no axis or the plane fails the precision gate.
    fn propose(
        &self,
        track: &EditableTrack,
        edited: &EditedReconstruction,
        measured: &Measurement,
    ) -> Result<Proposal, NormalError> {
        let was = patch_frame_of(track).normal();
        if measured.plane.determinacy == NormalDeterminacy::None {
            return Ok(Proposal::Nothing);
        }
        if let Some(bar) = self.options.max_normal_precision_deg {
            // A precision that is not finite fails every bar.
            if measured.precision_deg.is_nan() || measured.precision_deg > bar {
                return Ok(Proposal::Gated);
            }
        }
        let target = match measured.plane.determinacy {
            NormalDeterminacy::None => return Ok(Proposal::Nothing),
            NormalDeterminacy::BothAxes => {
                let n = Vector3::from(measured.plane.normal);
                // A plane has no sign: keep the face the patch shows.
                if n.dot(&was) < 0.0 {
                    -n
                } else {
                    n
                }
            }
            NormalDeterminacy::OneAxis { free_axis } => {
                let line = Vector3::from(free_axis);
                let length = line.norm();
                if length.is_nan() || length <= 0.0 {
                    return Ok(Proposal::Nothing);
                }
                perpendicular_part(was, line / length).unwrap_or(was)
            }
        };
        if !target.iter().all(|c| c.is_finite()) {
            return Ok(Proposal::Nothing);
        }
        let (turned, tilt) = tilt_patch(track, edited, target)?;
        // A turn by `θ` about the centre carries a cell centre `ρ` grid px
        // from it through a chord of `2ρ sin(θ/2)`.
        let largest_move = 2.0 * self.reach * (tilt.degrees.to_radians() / 2.0).sin();
        Ok(Proposal::Turn {
            state: Box::new(turned),
            normal: tilt.normal.into(),
            moved: largest_move,
        })
    }

    /// One view's context tile at `patch_frame`: the patch re-anchored on
    /// `keypoint` (on the patch frame itself where there is none or its ray misses
    /// the plane), widened by `margin` grid px on every side, and drawn
    /// through its plane with the sampler the fit's bitmap render picks from
    /// the view's Jacobian. `None` when no sample lands on the photograph.
    fn render(
        &self,
        patch_frame: &OrientedPatch,
        view: &ProjectedImage<'_>,
        keypoint: Option<[f64; 2]>,
        margin: usize,
    ) -> Option<Render> {
        let r = self.layout.resolution;
        let anchored_patch_frame = keypoint
            .and_then(|kp| patch_frame.anchored_at_keypoint(view.camera, view.cam_from_world, kp))
            .unwrap_or_else(|| patch_frame.clone());
        let jacobian =
            patch_grid_jacobian(&anchored_patch_frame, view.camera, view.cam_from_world, r);
        let sampler = self.options.fit.refine.sampler.for_jacobian(jacobian);
        let side = r + 2 * margin;
        let scale = side as f64 / r as f64;
        let mut context = anchored_patch_frame.clone();
        context.half_extent = [
            anchored_patch_frame.half_extent[0] * scale,
            anchored_patch_frame.half_extent[1] * scale,
        ];
        let mut map = WarpMap::from_patch(&context, view.camera, view.cam_from_world, side as u32);
        let tile = {
            let _phase = render_phase(self.progress, sampler, 1);
            render_tile(view.pyramid, &mut map, sampler)
        };
        // A fourth channel is the photograph's alpha, which carries no
        // texture.
        let channels = (tile.channels() as usize).clamp(1, 3);
        let mut samples = vec![f32::NAN; side * side * channels];
        let mut any = false;
        for row in 0..side as u32 {
            for col in 0..side as u32 {
                if !map.is_valid(col, row) {
                    continue;
                }
                any = true;
                let at = (row as usize * side + col as usize) * channels;
                for c in 0..channels {
                    samples[at + c] = f32::from(tile.get_pixel(col, row, c as u32));
                }
            }
        }
        any.then_some(Render {
            patch_frame: anchored_patch_frame,
            samples,
            side,
            channels,
        })
    }
}

/// The largest angle between two of the directions from `centre` to the
/// camera centres, degrees; `NaN` with fewer than two usable directions.
fn view_baseline_deg(centre: Vector3<f64>, cameras: impl Iterator<Item = Vector3<f64>>) -> f64 {
    let directions: Vec<Vector3<f64>> = cameras
        .filter_map(|c| (c - centre).try_normalize(1e-12))
        .collect();
    let mut widest = f64::NAN;
    for (i, a) in directions.iter().enumerate() {
        for b in &directions[i + 1..] {
            let angle = a.dot(b).clamp(-1.0, 1.0).acos().to_degrees();
            widest = if widest.is_nan() {
                angle
            } else {
                widest.max(angle)
            };
        }
    }
    widest
}

/// How far over a whole number of photograph pixels a footprint may read and
/// still round up to that number, in px.
///
/// An absolute tolerance, because the footprint is compared with whole
/// pixels whatever its size. The Jacobian is a central difference of the
/// projection half a grid px either side of the centre, so a footprint that
/// is a whole number of pixels reads a few ten-millionths of a pixel over it
/// (32.0000005 on the tests' scene); a thousandth of a pixel is well above
/// that and far below any footprint difference that matters to the grid.
const FOOTPRINT_TOLERANCE_PX: f64 = 1e-3;

/// The working grid's side for `None`: the patch's longer side in the
/// reference view's photograph pixels, rounded up and clamped to
/// `base ..= 2 · base`.
///
/// `jacobian` is the reference render map's Jacobian at the patch centre on a
/// grid of `base` samples a side ([`patch_grid_jacobian`]), photograph px per
/// grid px; its two columns are the patch's `u` and `v` axes as the view sees
/// them, so the patch's side along each is `base` times a column's length.
/// One sample per photograph pixel has no bound against the patch, which can
/// span hundreds of pixels close up, and the cost grows with the square of
/// the side, so the side is at most `2 · base`; at least `base`, it is never
/// coarser than the patch's own grid. Without a Jacobian it is `base`.
///
/// The side is rounded up after [`FOOTPRINT_TOLERANCE_PX`] is taken off, so a
/// footprint that is a whole number of pixels up to the Jacobian's own error
/// keeps that number.
pub(super) fn working_side(jacobian: Option<[[f64; 2]; 2]>, base: u32) -> u32 {
    let Some(j) = jacobian else {
        return base;
    };
    let along_u = j[0][0].hypot(j[1][0]);
    let along_v = j[0][1].hypot(j[1][1]);
    let side_px = f64::from(base) * along_u.max(along_v);
    if !side_px.is_finite() {
        return base;
    }
    let cap = base.saturating_mul(2);
    let side = (side_px - FOOTPRINT_TOLERANCE_PX).ceil();
    (side.min(f64::from(cap)) as u32).clamp(base, cap)
}

/// A reference cell's status as a template: refused for curvature where it
/// is flat in every channel, not attempted where some sample of it is off the
/// photograph, and fitted otherwise.
fn template_status(template: &CellTemplate) -> CellStatus {
    if template.missing {
        CellStatus::NotAttempted
    } else if template.channels.iter().all(Option::is_none) {
        CellStatus::RefusedCurvature
    } else {
        CellStatus::Fitted
    }
}

/// The statuses of one view's readings after the view's robust affine fit: a
/// fitted cell the fit gives no weight is an outlier, and where the fit fails
/// (its map reflects or is not finite) no cell of the view is kept, as the
/// piecewise refinement stores a failed fit. Returns the fit too, `None` where
/// it failed or no cell passed its own gates.
fn gated(
    readings: [[CellReading; 3]; 3],
    layout: &CellLayout,
) -> ([[CellStatus; 3]; 3], Option<Pass>) {
    let mut statuses = readings.map(|line| line.map(|reading| reading.status));
    let pass = fit_pass(readings, layout);
    for (row, line) in statuses.iter_mut().enumerate() {
        for (col, status) in line.iter_mut().enumerate() {
            if *status != CellStatus::Fitted {
                continue;
            }
            *status = match &pass {
                Some(pass) if pass.weights[row][col] > 0.0 => CellStatus::Fitted,
                Some(_) => CellStatus::RefusedOutlier,
                None => CellStatus::NotAttempted,
            };
        }
    }
    (statuses, pass)
}

/// The ray through the photograph's pixel under grid position `at` of a
/// render through `patch_frame`, in grid px centred on the grid. Its noise floor
/// is a measured shift's precision carried into the view's pixels by the
/// render map's local Jacobian, floored at the kernel's absolute floor.
/// Returns the ray with the pixel it passes through; `None` where the
/// position does not project.
fn cell_ray(
    patch_frame: &OrientedPatch,
    view: &ProjectedImage<'_>,
    posed: &PosedView<'_>,
    at: [f64; 2],
    wpp: f64,
    params: &CellPlaneParams,
) -> Option<(CellRay, [f64; 2])> {
    let pixel = |gx: f64, gy: f64| {
        let world = shifted_center(patch_frame, gx, gy, wpp, wpp);
        view.camera
            .project_homogeneous(view.cam_from_world, world.coords, patch_frame.w)
    };
    let centre = pixel(at[0], at[1])?;
    let (right, left) = (pixel(at[0] + 0.5, at[1])?, pixel(at[0] - 0.5, at[1])?);
    let (down, up) = (pixel(at[0], at[1] + 0.5)?, pixel(at[0], at[1] - 0.5)?);
    let jacobian = [
        [right[0] - left[0], down[0] - up[0]],
        [right[1] - left[1], down[1] - up[1]],
    ];
    let largest = singular_values_2x2(jacobian)[0];
    let floor = (params.cell_shift_precision_grid_px * largest).max(params.ray_noise_floor_px);
    posed.ray_through(centre, floor).map(|ray| (ray, centre))
}

/// A render's detail before anything is read: the patch frame, no tiles, no rays,
/// and a plane with nothing fitted. The trace entry is filled in when the loop
/// ends.
fn empty_render(patch_frame: &OrientedPatch) -> GridPlaneRender {
    let nan3 = [f64::NAN; 3];
    let cell = GridPlaneCell {
        rays: Vec::new(),
        status: CellPlaneStatus::TooFewRays,
        position: nan3,
        residual_px: f64::NAN,
        sigma_px: f64::NAN,
        normal_variance: f64::NAN,
        precision_weight: 0.0,
        tukey_weight: 0.0,
        weight: 0.0,
    };
    GridPlaneRender {
        patch_frame: GridPlanePatchFrame {
            center: patch_frame.center.coords.into(),
            u_axis: patch_frame.u_axis.into(),
            v_axis: patch_frame.v_axis.into(),
            normal: patch_frame.normal().into(),
            half_extent: patch_frame.half_extent,
        },
        reference: None,
        views: Vec::new(),
        cells: std::array::from_fn(|_| cell.clone()),
        plane: GridPlanePlane {
            centroid: nan3,
            fitted_normal: nan3,
            normal: nan3,
            determinacy: NormalDeterminacy::None,
            scatter_eigenvalues: nan3,
            anisotropy: f64::NAN,
            n_eff: f64::NAN,
            plane_rms: f64::NAN,
            precision_deg: [f64::NAN; 2],
        },
        iteration: GridPlaneIteration {
            score: f64::NAN,
            determinacy: NormalDeterminacy::None,
            in_plane: 0,
            normal_precision_deg: f64::NAN,
            gated: false,
            proposed_normal: None,
            largest_move_px: f64::NAN,
            taken: false,
        },
    }
}

/// The detail record of `render`, the tile of `observation` in `image`.
fn tile_record(observation: usize, image: u32, render: &Render) -> GridPlaneTile {
    GridPlaneTile {
        observation,
        image,
        side: render.side as u32,
        channels: render.channels as u32,
        samples: render.samples.clone(),
    }
}

/// The detail record of a ray of `observation` in `image` through `pixel`.
fn ray_record(observation: usize, image: u32, ray: &CellRay, pixel: [f64; 2]) -> GridPlaneRay {
    GridPlaneRay {
        observation,
        image,
        pixel,
        origin: ray.origin(),
        direction: ray.direction(),
        floor_px: ray.floor_px(),
    }
}

/// The detail record of one view: its tile, its score, each cell's reading
/// and gated status, and its robust fit.
fn view_record(
    tile: GridPlaneTile,
    score: Option<f64>,
    readings: &[[CellReading; 3]; 3],
    statuses: &[[CellStatus; 3]; 3],
    pass: Option<&Pass>,
) -> GridPlaneView {
    let cells = std::array::from_fn(|k| {
        let (row, col) = (k / 3, k % 3);
        let reading = readings[row][col];
        GridPlaneViewCell {
            whole_shift: reading.whole_shift,
            shift: reading.shift,
            zncc: reading.zncc,
            curvature: reading.curvature,
            reading: reading.status,
            status: statuses[row][col],
            fit_weight: pass.map_or(0.0, |p| p.weights[row][col]),
        }
    });
    let fit = pass.map(|p| GridPlaneViewFit {
        model: match p.model {
            UpdateModel::Affine => GridPlaneFitModel::Affine,
            UpdateModel::Similarity => GridPlaneFitModel::Similarity,
            UpdateModel::Shift => GridPlaneFitModel::Shift,
        },
        a: p.update.a,
        b: p.update.b,
    });
    GridPlaneView {
        tile,
        score,
        cells,
        fit,
    }
}

/// Fill `record`'s cells and plane from the plane fit: `plane` and the
/// numbers `fitted` kept of it. The predicted precision is filled in by
/// [`Measurement::new`].
fn record_plane(record: &mut GridPlaneRender, plane: &CellPlaneNormal, fitted: &CellPlaneRecord) {
    for (k, cell) in record.cells.iter_mut().enumerate() {
        cell.status = plane.cell_status[k];
        cell.position = plane.cell_positions[k];
        cell.residual_px = plane.cell_residual_px[k];
        cell.sigma_px = fitted.sigma_px[k];
        cell.normal_variance = fitted.normal_variance[k];
        cell.precision_weight = fitted.precision_weight[k];
        cell.tukey_weight = fitted.tukey_weight[k];
        cell.weight = plane.cell_weight[k];
    }
    record.plane = GridPlanePlane {
        centroid: fitted.centroid,
        fitted_normal: fitted.fitted_normal,
        normal: plane.normal,
        determinacy: plane.determinacy,
        scatter_eigenvalues: fitted.scatter_eigenvalues,
        anisotropy: plane.anisotropy,
        n_eff: plane.n_eff,
        plane_rms: plane.plane_rms,
        precision_deg: record.plane.precision_deg,
    };
}

/// Interleaved `n × channels` samples as `channels` planes of `n`.
fn planar(samples: &[f32], n: usize, channels: usize) -> Vec<f32> {
    let mut planes = vec![f32::NAN; n * channels];
    for (k, px) in samples.chunks_exact(channels).enumerate() {
        for (c, &v) in px.iter().enumerate() {
            planes[c * n + k] = v;
        }
    }
    planes
}

/// The windowed ZNCC of a view's tile at zero shift, its working grid's
/// middle `R×R`, against the reference's render `planes`, each sample
/// weighted by `window` (row-major over the `R×R` grid) and read where both
/// carry it, averaged over the channels the reference is not flat in; a
/// channel the view is flat in scores `0`. `None` when no channel scores.
fn tile_zncc(
    planes: &[f32],
    ref_channels: usize,
    render: &Render,
    margin: usize,
    window: &[f64],
) -> Option<f64> {
    let side = render.side;
    let r = side - 2 * margin;
    let n = r * r;
    let (mut sum, mut scored) = (0.0, 0usize);
    for c in 0..ref_channels.min(render.channels) {
        let plane = &planes[c * n..][..n];
        let mut pairs: Vec<(f64, f64, f64)> = Vec::with_capacity(n);
        for y in 0..r {
            for x in 0..r {
                let w = window[y * r + x];
                let a = plane[y * r + x];
                let b = render.samples[((y + margin) * side + x + margin) * render.channels + c];
                if w > 0.0 && a.is_finite() && b.is_finite() {
                    pairs.push((w, f64::from(a), f64::from(b)));
                }
            }
        }
        if pairs.len() < 2 {
            continue;
        }
        let total: f64 = pairs.iter().map(|p| p.0).sum();
        let (ma, mb) = pairs
            .iter()
            .fold((0.0, 0.0), |(sa, sb), (w, a, b)| (sa + w * a, sb + w * b));
        let (ma, mb) = (ma / total, mb / total);
        let (mut saa, mut sbb, mut sab) = (0.0, 0.0, 0.0);
        for (w, a, b) in &pairs {
            let (da, db) = (a - ma, b - mb);
            saa += w * da * da;
            sbb += w * db * db;
            sab += w * da * db;
        }
        if saa / total < FLAT_VARIANCE {
            continue;
        }
        scored += 1;
        if sbb / total >= FLAT_VARIANCE {
            sum += sab / (saa * sbb).sqrt();
        }
    }
    (scored > 0).then(|| sum / scored as f64)
}

/// The signed distance of `plane`'s weighted cell centroid from `patch_frame`'s
/// centre along its normal; `NaN` when the cells fixed no axis.
fn plane_offset(plane: &CellPlaneNormal, patch_frame: &OrientedPatch) -> f64 {
    if plane.determinacy == NormalDeterminacy::None {
        return f64::NAN;
    }
    let normal = Vector3::from(plane.normal);
    let (mut centroid, mut total) = (Vector3::zeros(), 0.0);
    for (position, &w) in plane.cell_positions.iter().zip(&plane.cell_weight) {
        if w > 0.0 {
            centroid += Vector3::from(*position) * w;
            total += w;
        }
    }
    if total.is_nan() || total <= 0.0 {
        return f64::NAN;
    }
    (centroid / total - patch_frame.center.coords).dot(&normal)
}

/// How [`turn_loop`] ended.
struct LoopEnd<T, M> {
    /// The state kept.
    state: T,
    /// The last measurement of the state kept.
    measured: M,
    /// Measurements made.
    iterations: u8,
    stop: LoopStop,
    /// The score of the starting state.
    start_score: f64,
    /// For each proposal, in order, whether the loop took it.
    taken: Vec<bool>,
}

/// The piecewise refinement's two-level loop with the measurement and the
/// turn behind closures: `measure` renders and reads a state, `score` is its
/// acceptance score, and `propose` gives the turned state the measurement
/// asks for with the farthest the turn moves a cell centre, or `None` when it
/// asks for none.
///
/// The loop measures the start, then until it stops proposes a turn and
/// measures the turned state. A turn that moves no cell centre by more than
/// `tolerance` is taken without a render ([`LoopStop::Converged`]); at
/// `max_iterations` measurements the turn the last one asks is not taken,
/// having no render to score it ([`LoopStop::Cap`]); a turn whose score falls
/// below the current one by more than [`ACCEPT_ZNCC_TOLERANCE`], or below the
/// start's, is not taken ([`LoopStop::Rejected`]); a turn that moves the
/// cells no less than the one before ends the loop on whichever of the two
/// states scores higher ([`LoopStop::Oscillation`]). A start that does not
/// score stops at once ([`LoopStop::NotRun`]), and a measurement that asks for
/// no turn stops where it is ([`LoopStop::Measured`]).
///
/// Under [`GridPlaneAcceptance::None`] the score decides nothing after the
/// start: no turn is rejected, and an oscillation ends on the later state.
fn turn_loop<T, M, E>(
    start: T,
    max_iterations: u8,
    tolerance: f64,
    acceptance: GridPlaneAcceptance,
    mut measure: impl FnMut(&T) -> Result<M, E>,
    score: impl Fn(&M) -> f64,
    mut propose: impl FnMut(&T, &M) -> Result<Option<(T, f64)>, E>,
) -> Result<LoopEnd<T, M>, E> {
    let max_iterations = max_iterations.max(1);
    let mut state = start;
    let mut measured = measure(&state)?;
    let mut iterations = 1u8;
    let start_score = score(&measured);
    let mut taken: Vec<bool> = Vec::new();
    let end = |state, measured, iterations, stop, taken| {
        Ok(LoopEnd {
            state,
            measured,
            iterations,
            stop,
            start_score,
            taken,
        })
    };
    if !start_score.is_finite() {
        return end(state, measured, iterations, LoopStop::NotRun, taken);
    }
    let scored = acceptance == GridPlaneAcceptance::WindowedZncc;
    let mut current = start_score;
    let mut previous_move = f64::INFINITY;
    loop {
        let Some((next, largest_move)) = propose(&state, &measured)? else {
            return end(state, measured, iterations, LoopStop::Measured, taken);
        };
        if largest_move <= tolerance {
            taken.push(true);
            return end(next, measured, iterations, LoopStop::Converged, taken);
        }
        if iterations >= max_iterations {
            taken.push(false);
            return end(state, measured, iterations, LoopStop::Cap, taken);
        }
        let next_measured = measure(&next)?;
        iterations += 1;
        let next_score = score(&next_measured);
        // The floor: the current score less the tolerance, but never below
        // the start's, so the tolerance cannot accumulate over several turns.
        let floor = (current - ACCEPT_ZNCC_TOLERANCE).max(start_score);
        if scored && (next_score.is_nan() || next_score < floor) {
            taken.push(false);
            return end(state, measured, iterations, LoopStop::Rejected, taken);
        }
        if largest_move >= previous_move {
            let keep_next = !scored || next_score >= current;
            taken.push(keep_next);
            return if keep_next {
                end(
                    next,
                    next_measured,
                    iterations,
                    LoopStop::Oscillation,
                    taken,
                )
            } else {
                end(state, measured, iterations, LoopStop::Oscillation, taken)
            };
        }
        taken.push(true);
        (state, measured, current) = (next, next_measured, next_score);
        previous_move = largest_move;
    }
}

#[cfg(test)]
mod tests;
