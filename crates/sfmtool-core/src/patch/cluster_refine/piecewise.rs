// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Piecewise refinement of a kept member: the nine cells of the template are
//! registered separately against the member's photograph at the member's
//! cascade shape, and a robust affine map is fitted to their shifts.
//!
//! See `specs/core/patch/cluster-patch-refinement.md`. The photograph
//! is rendered through the member's affine shape into a working patch: the
//! template's `R×R` grid plus a margin of
//! [`PiecewiseParams::cell_shift_bound_px`] on every side. Each of the nine
//! cells of the reference's square is registered against that working patch by
//! ZNCC over whole-pixel shifts within the bound, with a sub-pixel peak and its
//! curvature read at the best one; flat and low-scoring cells are refused, and
//! an affine map is fitted to the cells that remain by iteratively reweighted
//! least squares with a Tukey biweight, so a cell that disagrees with the
//! others is refused as an outlier.
//!
//! The render is this module's; everything after it, from the working patch's
//! summed-area tables through the cell search, the cell gates and the robust
//! affine fit, is the shared cell registration kernel,
//! [`crate::patch::cell_registration`], which this stage calls with
//! [`PiecewiseParams`] supplying the bound and the two bars.
//!
//! By default ([`PiecewiseParams::move_shape`] is `true`) the fitted map is an
//! update of the shape and the stage is a loop with two levels. An update is
//! applied only when the cascade's own objective, the whole-member windowed
//! ZNCC, does not fall at the updated shape (see [`ACCEPT_ZNCC_TOLERANCE`]), and the loop renders again with the
//! updated shape until the update moves no cell centre by more than
//! [`PiecewiseParams::update_tolerance_px`], until an update is rejected, until
//! the update's largest movement stops shrinking, or until
//! [`PiecewiseParams::max_iterations`]. With `move_shape` off, the stage is
//! one render and a measurement, and the member's shape is left as the
//! cascade found it.
//!
//! The cells are the `[0, R/3, R - R/3, R]` split the ZNCC grid uses. Shifts
//! and cell centres are in template grid px. A grid position `g` (column,
//! row) is carried into the member's image by `p + S·(off + step·g)`, which in
//! grid coordinates centred on the middle of the grid, `c = g − (R − 1)/2`, is
//! `p + S·(step·c)`. An update `c ↦ A·c + b` in those coordinates therefore
//! composes into the shape as `S' = S·A`, `p' = p + S·(step·b)`.

use sfmtool_matches_format::{ClusterCellStatus, MemberCellData};

use crate::camera::image::ImageU8Pyramid;
use crate::patch::cell_registration::{
    fit_pass, read_cells, CellGateOptions, CellLayout, CellStatus, CellTemplates, Pass, Update,
    UpdateModel, WorkingPatch,
};

use super::kernels::TemplateKernel;
use super::{inv2, mul2, sample_grid_window, GridEdge, Mat2};

#[cfg(test)]
mod tests;

/// How far the whole-member windowed ZNCC may fall at an updated shape before
/// the update is rejected, never below the ZNCC at the starting shape. The
/// cascade maximised that ZNCC, so an update that lowers it moves the shape
/// away from the cascade's optimum; the tolerance only lets the loop cross a
/// flat stretch above the start, and cannot accumulate past it.
pub const ACCEPT_ZNCC_TOLERANCE: f64 = 1e-4;

/// Provisional default for [`PiecewiseParams::min_cell_zncc`].
///
/// Provisional: chosen on the synthetic tests, not measured on captures. It
/// sits below the whole-member bar (`0.85`) because a cell holds about a ninth
/// of the samples the whole-member score reads, so its ZNCC is noisier.
pub const DEFAULT_MIN_CELL_ZNCC: f32 = 0.8;

/// Provisional default for [`PiecewiseParams::min_cell_curvature`], in ZNCC
/// per grid px².
///
/// Provisional: chosen on the synthetic tests, not measured on captures. A
/// texture whose ZNCC falls by `0.01` one grid px from its peak along its
/// flattest direction reads `0.02`. A flat cell reads `0`, and a straight edge
/// reads near `0` along the edge.
pub const DEFAULT_MIN_CELL_CURVATURE: f32 = 0.02;

/// Tunables of the piecewise refinement that follows the affine cascade for
/// every kept member.
#[derive(Clone, Debug, PartialEq)]
pub struct PiecewiseParams {
    /// Whether the fitted affine map may move the member's shape. `true`, the
    /// default, runs the loop that applies the fitted map as an update of the
    /// shape while the whole-member ZNCC does not fall. It is on by default
    /// because a blind human review of the shapes it moves preferred them to
    /// the cascade's, increasingly so the further they moved. `false`
    /// measures the cells once at the cascade's shape and leaves the member's
    /// shape, position and readings exactly as the cascade produced them.
    pub move_shape: bool,
    /// Search bound for a cell's shift from its affine placement, in template
    /// grid px. The working patch is rendered with a margin of this many grid
    /// px on each side (rounded up), and the shift search covers the whole
    /// shifts within it (rounded down), so no shift reads past the render.
    /// The sub-pixel peak needs a whole-pixel neighbour on each side, so a
    /// cell whose best whole shift lies on the bound is refused as
    /// [`CellStatus::RefusedBound`], and a bound under `1` refuses every cell
    /// that way.
    pub cell_shift_bound_px: f32,
    /// A cell whose ZNCC at its optimum is below this is refused as
    /// [`CellStatus::RefusedZncc`]. See [`DEFAULT_MIN_CELL_ZNCC`].
    pub min_cell_zncc: f32,
    /// A cell whose ZNCC peak is flatter than this along its flattest
    /// direction is refused as [`CellStatus::RefusedCurvature`]: the smaller
    /// eigenvalue of the negated Hessian of the ZNCC over the shift, in ZNCC
    /// per grid px², read from the three-by-three neighbourhood of the best
    /// whole shift. See [`DEFAULT_MIN_CELL_CURVATURE`].
    pub min_cell_curvature: f32,
    /// The loop stops when the affine update moves every cell centre by less
    /// than this, in grid px. Read only with [`Self::move_shape`].
    pub update_tolerance_px: f32,
    /// The most renders the loop makes for one member. Read only with
    /// [`Self::move_shape`]; without it the stage renders once.
    pub max_iterations: u8,
}

impl Default for PiecewiseParams {
    fn default() -> Self {
        Self {
            move_shape: true,
            cell_shift_bound_px: 2.0,
            min_cell_zncc: DEFAULT_MIN_CELL_ZNCC,
            min_cell_curvature: DEFAULT_MIN_CELL_CURVATURE,
            update_tolerance_px: 0.05,
            max_iterations: 5,
        }
    }
}

impl PiecewiseParams {
    /// The two bars the cell registration gates each cell's reading by.
    pub(super) fn cell_gates(&self) -> CellGateOptions {
        CellGateOptions {
            min_cell_zncc: self.min_cell_zncc,
            min_cell_curvature: self.min_cell_curvature,
        }
    }
}

/// Why the loop of one member stopped. Without
/// [`PiecewiseParams::move_shape`] there is no loop, and a member with
/// readings reads [`LoopStop::Measured`].
#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LoopStop {
    /// The stage produced no reading: every cell is
    /// [`CellStatus::NotAttempted`] and the member keeps its cascade shape.
    NotRun = 0,
    /// The last update moved no cell centre by more than
    /// [`PiecewiseParams::update_tolerance_px`], and was applied.
    Converged = 1,
    /// The loop reached [`PiecewiseParams::max_iterations`] with every update
    /// accepted; the last one was applied.
    Cap = 2,
    /// The last update would have lowered the whole-member ZNCC by more than
    /// [`ACCEPT_ZNCC_TOLERANCE`], so it was not applied and the member keeps
    /// the shape the last render was made at.
    Rejected = 3,
    /// The last update's largest cell-centre movement did not shrink from the
    /// one before, so the loop was alternating rather than converging. The
    /// member keeps whichever of the last two shapes has the higher
    /// whole-member ZNCC.
    Oscillation = 4,
    /// [`PiecewiseParams::move_shape`] is off: the cells were measured once
    /// at the cascade's shape, and no update was applied.
    Measured = 5,
}

/// Per-cell registration of one member against the template, relative to the
/// affine shape the stage returns for the member. `[row][col]` from the
/// top-left cell, rows and columns cut at `R/3` and `R - R/3`.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct CellRefinement {
    /// Displacement of each cell's centre from where the returned affine
    /// shape places it, in template grid px, `[x, y]` along the grid's columns
    /// and rows. No fitted affine map is removed from it, so it carries
    /// whatever first-order disagreement the cells have with that shape.
    /// Without [`PiecewiseParams::move_shape`] the returned shape is the
    /// cascade's, and the displacement is the shift `d` as measured. With it,
    /// a shift `d` measured at cell centre `c` at a render whose update
    /// `c ↦ A·c + b` was applied is carried into the returned shape's grid as
    /// `A⁻¹·(d − (A·c + b − c))`: the point `c + d` of the last render's grid
    /// expressed in the returned shape's grid, less `c`. `NaN` where no
    /// sub-pixel shift was measured: a cell refused by curvature or by the
    /// bound, or not attempted.
    pub shift_px: [[[f32; 2]; 3]; 3],
    /// ZNCC of each cell at its optimum at the last render: the sub-pixel
    /// peak's value where one was read, the best whole shift's otherwise.
    /// `NaN` where nothing was read. Without
    /// [`PiecewiseParams::move_shape`] the last render is at the returned
    /// shape. With it, when the last update was applied
    /// ([`Self::final_update_accepted`]), the last render is at the shape
    /// before that update, so this ZNCC and [`Self::status`] were read there,
    /// not at the returned shape.
    pub zncc: [[f32; 3]; 3],
    /// Each cell's status at the last render.
    pub status: [[CellStatus; 3]; 3],
    /// Renders the stage made (see [`Self::stop`]): `1` without
    /// [`PiecewiseParams::move_shape`].
    pub iterations: u8,
    /// Why the loop stopped: [`LoopStop::Measured`] without
    /// [`PiecewiseParams::move_shape`].
    pub stop: LoopStop,
    /// Whether the last fitted update was applied to the returned shape.
    /// `false` when it was rejected, when an oscillation kept the shape before
    /// it, when the loop did not run, and always without
    /// [`PiecewiseParams::move_shape`].
    pub final_update_accepted: bool,
}

impl From<CellStatus> for ClusterCellStatus {
    /// The `.matches` cell status a [`CellStatus`] is stored as. The two
    /// enums share their discriminants, which a test checks, so the stored
    /// code is the discriminant either way.
    fn from(status: CellStatus) -> ClusterCellStatus {
        match status {
            CellStatus::Fitted => ClusterCellStatus::Fitted,
            CellStatus::RefusedCurvature => ClusterCellStatus::RefusedCurvature,
            CellStatus::RefusedZncc => ClusterCellStatus::RefusedZncc,
            CellStatus::NotAttempted => ClusterCellStatus::NotAttempted,
            CellStatus::RefusedBound => ClusterCellStatus::RefusedBound,
            CellStatus::RefusedOutlier => ClusterCellStatus::RefusedOutlier,
        }
    }
}

/// The per-cell columns of a cluster-patches file for one refinement's
/// [`ClusterRefineResult::cells`](super::ClusterRefineResult::cells), one row
/// per member in member order.
///
/// A member with cells gets its shifts, ZNCCs, statuses and iteration count;
/// a member without (`None`, every member that is not kept) gets `NaN`
/// shifts and ZNCCs, every cell not attempted and `0` iterations, which is
/// what the format stores for a member the stage did not run on.
pub fn member_cell_data(cells: &[Option<CellRefinement>]) -> MemberCellData {
    let mut out = MemberCellData::not_attempted(cells.len());
    for (m, cell) in cells.iter().enumerate() {
        let Some(cell) = cell else { continue };
        for row in 0..3 {
            for col in 0..3 {
                out.shift_px[[m, row, col, 0]] = cell.shift_px[row][col][0];
                out.shift_px[[m, row, col, 1]] = cell.shift_px[row][col][1];
                out.zncc[[m, row, col]] = cell.zncc[row][col];
                out.status[[m, row, col]] = ClusterCellStatus::from(cell.status[row][col]) as u8;
            }
        }
        out.iterations[m] = cell.iterations;
    }
    out
}

impl CellRefinement {
    /// Every cell not attempted, after `iterations` renders.
    pub fn not_attempted(iterations: u8) -> CellRefinement {
        CellRefinement {
            shift_px: [[[f32::NAN; 2]; 3]; 3],
            zncc: [[f32::NAN; 3]; 3],
            status: [[CellStatus::NotAttempted; 3]; 3],
            iterations,
            stop: LoopStop::NotRun,
            final_update_accepted: false,
        }
    }
}

impl CellTemplates {
    /// Cut the nine cells out of the template's samples over the whole square.
    pub(super) fn new(tmpl: &TemplateKernel, layout: &CellLayout) -> CellTemplates {
        CellTemplates::from_planes(&tmpl.square_samples, tmpl.channels, layout)
    }
}

impl WorkingPatch {
    /// Render the member at position `p` and absolute shape `s`. `None` when
    /// the map is degenerate or the selected pyramid level is too small to
    /// sample.
    pub(super) fn render(
        pyramid: &ImageU8Pyramid,
        p: [f64; 2],
        s: &Mat2,
        layout: &CellLayout,
        step: f64,
        off: f64,
    ) -> Option<WorkingPatch> {
        let side = layout.patch_side();
        let samples = sample_grid_window(
            pyramid,
            p,
            s,
            step,
            off,
            -(layout.margin as i64),
            side,
            GridEdge::Missing,
        )?;
        let channels = samples.len() / (side * side);
        Some(WorkingPatch::new(side, channels, samples))
    }
}

/// The outcome of the piecewise refinement of one member.
pub(super) struct MemberCells {
    /// The refined absolute shape and position; the cascade's without
    /// [`PiecewiseParams::move_shape`], when nothing survived, or when every
    /// update was rejected.
    pub(super) shape: Mat2,
    pub(super) position: [f64; 2],
    pub(super) cells: CellRefinement,
    /// Whether the shape or position differs from the cascade's.
    pub(super) updated: bool,
    /// The model the last affine map was fitted with.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(super) model: Option<UpdateModel>,
}

/// The shape and position the update `c ↦ A·c + b` carries `(s, p)` to:
/// `S' = S·A`, `p' = p + S·(step·b)`.
fn compose(s: &Mat2, p: [f64; 2], update: &Update, step: f64) -> (Mat2, [f64; 2]) {
    let sb = [update.b[0] * step, update.b[1] * step];
    let p = [
        p[0] + s[0][0] * sb[0] + s[0][1] * sb[1],
        p[1] + s[1][0] * sb[0] + s[1][1] * sb[1],
    ];
    (mul2(s, &update.a), p)
}

/// Run the piecewise refinement for one kept member, starting from its
/// cascade shape `s` and position `p`.
///
/// Without [`PiecewiseParams::move_shape`] the stage renders the working patch
/// once at `(s, p)`, reads the nine cells and fits the robust affine map to the
/// cells that survive, only to find the cells it refuses as outliers. The
/// member keeps `(s, p)`, `score` is not read, the stored shifts are the
/// shifts as measured, and the cells read one iteration and
/// [`LoopStop::Measured`]. A failed render or fit leaves every cell
/// [`CellStatus::NotAttempted`] after one iteration.
///
/// With `move_shape`, it runs the two-level loop. Each iteration renders the
/// working patch at the current shape, reads the nine cells and fits an
/// update to the cells that survive (see
/// [`fit_update_robust`](crate::patch::cell_registration::fit_update_robust)).
/// `score` reads the whole-member windowed ZNCC the cascade maximised at a
/// shape and position, `None` when the support leaves the frame. The loop
/// stops at the first of:
///
/// - the update lowers that ZNCC by more than [`ACCEPT_ZNCC_TOLERANCE`], or
///   below its value at the starting shape ([`LoopStop::Rejected`]): the
///   update is not applied;
/// - the update moves no cell centre by more than
///   [`PiecewiseParams::update_tolerance_px`] ([`LoopStop::Converged`]): it is
///   applied;
/// - the update's largest cell-centre movement is not smaller than the one
///   before ([`LoopStop::Oscillation`]): of the shape before the update and
///   the shape after it, the one with the higher ZNCC is kept;
/// - [`PiecewiseParams::max_iterations`] renders ([`LoopStop::Cap`]): the last
///   update is applied.
///
/// The returned cells are the readings of the last render. Their shifts are
/// each cell's displacement from where the returned shape places it: a shift
/// measured at a render whose update was applied is carried into the updated
/// shape's grid (see [`CellRefinement::shift_px`]); one measured at a render
/// whose update was not applied is already relative to the returned shape and
/// is stored as measured. A fitted cell the robust fit gave weight `0` is stored as
/// [`CellStatus::RefusedOutlier`].
///
/// A failed iteration leaves the member at its cascade shape and position
/// with every cell [`CellStatus::NotAttempted`], discarding what earlier
/// iterations fitted. An iteration fails when the render fails, when no cell
/// survives, when the fitted update reflects or is not finite (see
/// `fit_update`), or when the updated shape is not finite. The loop does not
/// run when `score` cannot read the starting shape.
#[allow(clippy::too_many_arguments)]
pub(super) fn refine_member_cells(
    pyramid: &ImageU8Pyramid,
    src_channels: &[usize],
    templates: &CellTemplates,
    layout: &CellLayout,
    s: Mat2,
    p: [f64; 2],
    step: f64,
    off: f64,
    params: &PiecewiseParams,
    score: impl FnMut(&Mat2, [f64; 2]) -> Option<f64>,
) -> MemberCells {
    let gates = params.cell_gates();
    let pass = |shape: &Mat2, position: [f64; 2]| {
        let patch = WorkingPatch::render(pyramid, position, shape, layout, step, off)?;
        let readings = read_cells(templates, src_channels, &patch, layout, &gates);
        fit_pass(readings, layout)
    };
    run_loop(s, p, step, layout, params, pass, score)
}

/// The loop of [`refine_member_cells`], with the render, the readings and the
/// fit behind `pass`, which returns `None` when the iteration fails.
fn run_loop(
    s: Mat2,
    p: [f64; 2],
    step: f64,
    layout: &CellLayout,
    params: &PiecewiseParams,
    mut pass: impl FnMut(&Mat2, [f64; 2]) -> Option<Pass>,
    mut score: impl FnMut(&Mat2, [f64; 2]) -> Option<f64>,
) -> MemberCells {
    let unchanged = |iterations: u8| MemberCells {
        shape: s,
        position: p,
        cells: CellRefinement::not_attempted(iterations),
        updated: false,
        model: None,
    };
    if !params.move_shape {
        let Some(only) = pass(&s, p) else {
            return unchanged(1);
        };
        return MemberCells {
            shape: s,
            position: p,
            cells: stored_cells(&only, layout, 1, LoopStop::Measured, false),
            updated: false,
            model: Some(only.model),
        };
    }
    if params.max_iterations == 0 {
        return unchanged(0);
    }
    let Some(start_zncc) = score(&s, p) else {
        return unchanged(0);
    };
    let (mut shape, mut position, mut zncc) = (s, p, start_zncc);
    let mut previous_move = f64::INFINITY;
    let mut iterations = 0u8;
    let (last, stop, apply) = loop {
        iterations += 1;
        let Some(last) = pass(&shape, position) else {
            return unchanged(iterations);
        };
        let (next_shape, next_position) = compose(&shape, position, &last.update, step);
        let finite = next_shape
            .iter()
            .flatten()
            .chain(next_position.iter())
            .all(|v| v.is_finite());
        if !finite {
            return unchanged(iterations);
        }
        let largest_move = layout
            .centres
            .iter()
            .flatten()
            .map(|&c| {
                let m = last.update.moves(c);
                m[0].hypot(m[1])
            })
            .fold(0.0f64, f64::max);
        let next_zncc = score(&next_shape, next_position).unwrap_or(f64::NEG_INFINITY);
        // The floor: the current ZNCC less the tolerance, but never below
        // the starting shape's, so the tolerance cannot accumulate over
        // several applied updates.
        let floor = (zncc - ACCEPT_ZNCC_TOLERANCE).max(start_zncc);
        let (stop, apply) = if next_zncc < floor {
            (LoopStop::Rejected, false)
        } else if largest_move <= f64::from(params.update_tolerance_px) {
            (LoopStop::Converged, true)
        } else if largest_move >= previous_move {
            (LoopStop::Oscillation, next_zncc >= zncc)
        } else if iterations >= params.max_iterations {
            (LoopStop::Cap, true)
        } else {
            (shape, position, zncc) = (next_shape, next_position, next_zncc);
            previous_move = largest_move;
            continue;
        };
        if apply {
            (shape, position) = (next_shape, next_position);
        }
        break (last, stop, apply);
    };
    MemberCells {
        shape,
        position,
        cells: stored_cells(&last, layout, iterations, stop, apply),
        updated: shape != s || position != p,
        model: Some(last.model),
    }
}

/// The cells of the last pass as stored, relative to the shape the stage
/// returns. A fitted cell the robust fit gave weight `0` is stored as
/// [`CellStatus::RefusedOutlier`].
fn stored_cells(
    last: &Pass,
    layout: &CellLayout,
    iterations: u8,
    stop: LoopStop,
    apply: bool,
) -> CellRefinement {
    // The displacement from the shape the stage returns. The last render read
    // cell `c` at `c + d` of its grid. When its update was applied, the
    // returned shape's grid is the last render's under `c' ↦ A·c' + b`, so
    // that point is `A⁻¹·(c + d − b)` there, and its offset from `c` is
    // `A⁻¹·(d − moves(c))`. When it was not, the returned shape is the one
    // the last render was made at, and `d` is already the displacement.
    let a_inv = inv2(&last.update.a);
    let mut cells = CellRefinement::not_attempted(iterations);
    cells.stop = stop;
    cells.final_update_accepted = apply;
    for (row, line) in last.readings.iter().enumerate() {
        for (col, rd) in line.iter().enumerate() {
            cells.status[row][col] = match rd.status {
                CellStatus::Fitted if last.weights[row][col] <= 0.0 => CellStatus::RefusedOutlier,
                status => status,
            };
            cells.zncc[row][col] = rd.zncc as f32;
            if rd.shift.iter().all(|v| v.is_finite()) {
                let r = if apply {
                    let m = last.update.moves(layout.centres[row][col]);
                    mul2v(&a_inv, [rd.shift[0] - m[0], rd.shift[1] - m[1]])
                } else {
                    rd.shift
                };
                cells.shift_px[row][col] = [r[0] as f32, r[1] as f32];
            }
        }
    }
    cells
}

/// `a·v`.
fn mul2v(a: &Mat2, v: [f64; 2]) -> [f64; 2] {
    [
        a[0][0] * v[0] + a[0][1] * v[1],
        a[1][0] * v[0] + a[1][1] * v[1],
    ]
}
