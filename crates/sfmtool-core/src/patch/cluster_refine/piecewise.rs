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
//! By default ([`PiecewiseParams::move_shape`] is `false`) that is the whole
//! stage: one render, and the member's shape is left as the cascade found it.
//! With `move_shape`, the fitted map is an update of the shape and the stage is
//! a loop with two levels. An update is applied only when the cascade's own
//! objective, the whole-member windowed ZNCC, does not fall at the updated
//! shape (see [`ACCEPT_ZNCC_TOLERANCE`]), and the loop renders again with the
//! updated shape until the update moves no cell centre by more than
//! [`PiecewiseParams::update_tolerance_px`], until an update is rejected, until
//! the update's largest movement stops shrinking, or until
//! [`PiecewiseParams::max_iterations`].
//!
//! The cells are the `[0, R/3, R - R/3, R]` split the ZNCC grid uses. Shifts
//! and cell centres are in template grid px. A grid position `g` (column,
//! row) is carried into the member's image by `p + S·(off + step·g)`, which in
//! grid coordinates centred on the middle of the grid, `c = g − (R − 1)/2`, is
//! `p + S·(step·c)`. An update `c ↦ A·c + b` in those coordinates therefore
//! composes into the shape as `S' = S·A`, `p' = p + S·(step·b)`.

use sfmtool_matches_format::{ClusterCellStatus, MemberCellData};

use crate::camera::image::ImageU8Pyramid;
use crate::numeric::median_in_place;
use crate::patch::normal_refine::{grid_bounds, grid_cell_centres, FLAT_NORM_SQ_EPS};

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

/// Rounds of reweighting after the first, curvature-weighted fit of the
/// update.
const IRLS_ROUNDS: usize = 3;

/// The Tukey biweight's cut-off, in units of the residual scale: a cell whose
/// residual to the fitted update is this many scales or more gets weight `0`.
/// `4.685` is the conventional cut-off for one-dimensional residuals; here it
/// is applied to the length of a two-dimensional residual, in units of the
/// per-axis scale [`MEDIAN_LENGTH_TO_SIGMA`] estimates.
const TUKEY_CUTOFF: f64 = 4.685;

/// Converts the median length of the cells' two-dimensional residuals to the
/// per-axis standard deviation `σ` of isotropic Gaussian noise: the length of
/// such a residual is Rayleigh-distributed, with median `σ·√(2 ln 2)`, so
/// `σ = median / √(2 ln 2)`. The factor `1.4826` used for one-dimensional
/// residuals would overstate `σ` here by a factor of 1.75.
const MEDIAN_LENGTH_TO_SIGMA: f64 = 0.849_321_800_288_019_1;

/// The smallest residual scale the reweighting uses, grid px. Without it a
/// fit through cells that agree to a few hundredths of a pixel would refuse a
/// cell a tenth of a pixel off.
const MIN_RESIDUAL_SCALE_PX: f64 = 0.1;

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
    /// Whether the fitted affine map may move the member's shape. `false`, the
    /// default, measures the cells once at the cascade's shape and leaves the
    /// member's shape, position and readings exactly as the cascade produced
    /// them. `true` runs the loop that applies the fitted map as an update of
    /// the shape while the whole-member ZNCC does not fall. It is off by
    /// default because, on captures, the few members the loop moves, each to a
    /// shape of higher whole-member ZNCC, changed which seed candidate passes.
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
            move_shape: false,
            cell_shift_bound_px: 2.0,
            min_cell_zncc: DEFAULT_MIN_CELL_ZNCC,
            min_cell_curvature: DEFAULT_MIN_CELL_CURVATURE,
            update_tolerance_px: 0.05,
            max_iterations: 5,
        }
    }
}

/// What the piecewise refinement concluded about one cell of a member.
///
/// The discriminants are the canonical codes of the `.matches` cell-status
/// column, [`ClusterCellStatus`], which [`member_cell_data`] stores them as.
#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CellStatus {
    /// The cell's shift was measured and the robust affine fit over the
    /// member's cells gave it weight.
    Fitted = 0,
    /// The cell's own reading refused it: the reference is flat over the cell,
    /// or the ZNCC peak is flatter than
    /// [`PiecewiseParams::min_cell_curvature`].
    RefusedCurvature = 1,
    /// The cell's ZNCC at its optimum is below
    /// [`PiecewiseParams::min_cell_zncc`]: it is over a different surface.
    RefusedZncc = 2,
    /// The cell was not registered: a sample the search needs could not be
    /// read, or no cell of the member survived. Every cell is stored this way
    /// when a render failed or the fitted map was not finite or reflected,
    /// and, with [`PiecewiseParams::move_shape`], when the refined shape could
    /// not be accepted (the readings at it failed the member's gates, or its
    /// support left the frame). The member then keeps its cascade shape, and
    /// the iteration count includes the pass that failed.
    NotAttempted = 3,
    /// The cell's best whole shift lies on the search bound
    /// [`PiecewiseParams::cell_shift_bound_px`], so no whole-pixel neighbour
    /// on that side was read and no sub-pixel peak can be fitted: the cell's
    /// optimum is at or past the bound.
    RefusedBound = 4,
    /// The cell passed its own gates, but its shift disagrees with the
    /// affine map the other cells agree on by so much that the robust fit
    /// gave it weight `0`. Its shift is measured and stored.
    RefusedOutlier = 5,
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

/// The fixed geometry of the cells and the search for one grid size and one
/// bound.
pub(super) struct CellLayout {
    /// `R`, the template's samples per axis.
    resolution: usize,
    /// Working-patch margin on each side, grid px: the bound rounded up.
    margin: usize,
    /// Half-width of the whole-shift search: the bound rounded down, at most
    /// `margin`.
    reach: usize,
    /// `[0, R/3, R - R/3, R]`.
    bounds: [usize; 4],
    /// Cell centres in grid coordinates centred on the grid, `[x, y]`.
    centres: [[[f64; 2]; 3]; 3],
    /// The side of a corner cell, grid px.
    cell_side: f64,
}

impl CellLayout {
    pub(super) fn new(resolution: u32, bound: f32) -> CellLayout {
        let resolution = resolution as usize;
        let bound = if bound.is_finite() {
            f64::from(bound.max(0.0))
        } else {
            0.0
        };
        let margin = bound.ceil() as usize;
        let reach = (bound.floor() as usize).min(margin);
        let bounds = grid_bounds(resolution as u32);
        let centres = grid_cell_centres(resolution as u32);
        CellLayout {
            resolution,
            margin,
            reach,
            bounds,
            centres,
            cell_side: (resolution / 3).max(1) as f64,
        }
    }

    /// Side of the working patch, grid px.
    fn patch_side(&self) -> usize {
        self.resolution + 2 * self.margin
    }
}

/// The reference's samples over one cell, per template channel.
struct CellTemplate {
    /// Per template channel: the mean-removed samples, row-major over the
    /// cell, and their sum of squares; `None` where the channel is flat over
    /// the cell.
    channels: Vec<Option<(Vec<f64>, f64)>>,
    /// Some pixel of the cell could not be sampled in the reference.
    missing: bool,
}

/// The reference's samples over each of the nine cells.
pub(super) struct CellTemplates {
    cells: [[CellTemplate; 3]; 3],
}

impl CellTemplates {
    /// Cut the nine cells out of the template's samples over the whole square.
    pub(super) fn new(tmpl: &TemplateKernel, layout: &CellLayout) -> CellTemplates {
        let r = layout.resolution;
        let square = r * r;
        let b = layout.bounds;
        let cells = std::array::from_fn(|row| {
            std::array::from_fn(|col| {
                let mut missing = false;
                let channels = (0..tmpl.channels)
                    .map(|c| {
                        let plane = &tmpl.square_samples[c * square..][..square];
                        let mut vals = Vec::new();
                        for y in b[row]..b[row + 1] {
                            for x in b[col]..b[col + 1] {
                                let v = plane[y * r + x];
                                if !v.is_finite() {
                                    missing = true;
                                }
                                vals.push(f64::from(v));
                            }
                        }
                        centred(vals)
                    })
                    .collect();
                CellTemplate { channels, missing }
            })
        });
        CellTemplates { cells }
    }
}

/// Mean-remove `vals`; `None` when they are flat (sum of squares under
/// [`FLAT_NORM_SQ_EPS`]) or hold a non-finite value.
fn centred(mut vals: Vec<f64>) -> Option<(Vec<f64>, f64)> {
    if vals.is_empty() || vals.iter().any(|v| !v.is_finite()) {
        return None;
    }
    let mean = vals.iter().sum::<f64>() / vals.len() as f64;
    let mut norm_sq = 0.0;
    for v in vals.iter_mut() {
        *v -= mean;
        norm_sq += *v * *v;
    }
    (norm_sq >= FLAT_NORM_SQ_EPS).then_some((vals, norm_sq))
}

/// The member's photograph rendered on the template's grid plus the margin,
/// interleaved `side × side × channels`, `NaN` where a sample left the image,
/// with the summed-area tables the cell search reads its window sums from.
pub(super) struct WorkingPatch {
    pub(super) side: usize,
    pub(super) channels: usize,
    pub(super) samples: Vec<f32>,
    sums: PatchSums,
}

/// Summed-area tables of one working patch, `(side + 1)²` entries each,
/// row-major, entry `(y, x)` the sum over the samples above and left of it.
///
/// A window's sum and sum of squares come from four entries each instead of a
/// pass over the window, which is what makes the whole-shift search cheap:
/// the template side of the ZNCC is mean-removed, so the cross term
/// `Σ (u − ū)·t = Σ u·t` needs no window mean, and the window's own
/// `Σ (u − ū)² = Σ u² − (Σ u)² / n` needs only the two sums.
struct PatchSums {
    /// Per channel, the mean of the channel's finite samples, subtracted from
    /// every sample before it is summed so the tables hold small numbers and
    /// `Σ u² − (Σ u)² / n` does not cancel. The ZNCC does not depend on it.
    offset: Vec<f64>,
    /// Per channel, the table of `u − offset`, a missing sample counted as
    /// `0`.
    sum: Vec<Vec<f64>>,
    /// Per channel, the table of `(u − offset)²`.
    sum_sq: Vec<Vec<f64>>,
    /// The table of missing samples: `1` where any channel of the sample is
    /// non-finite.
    missing: Vec<u32>,
}

impl PatchSums {
    fn new(side: usize, channels: usize, samples: &[f32]) -> PatchSums {
        let n = side + 1;
        let offset: Vec<f64> = (0..channels)
            .map(|c| {
                let (mut s, mut k) = (0.0f64, 0usize);
                for v in samples.iter().skip(c).step_by(channels) {
                    if v.is_finite() {
                        s += f64::from(*v);
                        k += 1;
                    }
                }
                if k > 0 {
                    s / k as f64
                } else {
                    0.0
                }
            })
            .collect();
        let mut sum = vec![vec![0.0f64; n * n]; channels];
        let mut sum_sq = vec![vec![0.0f64; n * n]; channels];
        let mut missing = vec![0u32; n * n];
        for y in 0..side {
            for x in 0..side {
                let px = &samples[(y * side + x) * channels..][..channels];
                let i = (y + 1) * n + x + 1;
                let (up, left, diag) = (y * n + x + 1, (y + 1) * n + x, y * n + x);
                let gone = px.iter().any(|v| !v.is_finite());
                missing[i] = missing[up] + missing[left] - missing[diag] + u32::from(gone);
                for c in 0..channels {
                    let v = if px[c].is_finite() {
                        f64::from(px[c]) - offset[c]
                    } else {
                        0.0
                    };
                    let (s, q) = (&mut sum[c], &mut sum_sq[c]);
                    s[i] = s[up] + s[left] - s[diag] + v;
                    q[i] = q[up] + q[left] - q[diag] + v * v;
                }
            }
        }
        PatchSums {
            offset,
            sum,
            sum_sq,
            missing,
        }
    }
}

/// A window of the working patch, columns `x0..x1` and rows `y0..y1`, read
/// from a summed-area table whose rows are `stride` entries long.
#[derive(Clone, Copy)]
struct Window {
    x0: usize,
    x1: usize,
    y0: usize,
    y1: usize,
    stride: usize,
}

impl Window {
    /// The window's total in `table`.
    fn rect<T>(&self, table: &[T]) -> T
    where
        T: Copy + std::ops::Add<Output = T> + std::ops::Sub<Output = T>,
    {
        let at = |y: usize, x: usize| table[y * self.stride + x];
        // Added before subtracted, so an unsigned table never underflows.
        at(self.y1, self.x1) + at(self.y0, self.x0) - at(self.y0, self.x1) - at(self.y1, self.x0)
    }
}

impl WorkingPatch {
    /// A working patch of `side × side × channels` interleaved samples, with
    /// its summed-area tables.
    pub(super) fn new(side: usize, channels: usize, samples: Vec<f32>) -> WorkingPatch {
        let sums = PatchSums::new(side, channels, &samples);
        WorkingPatch {
            side,
            channels,
            samples,
            sums,
        }
    }

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

/// One cell's reading at one render.
#[derive(Clone, Copy, Debug)]
struct CellReading {
    status: CellStatus,
    /// Sub-pixel shift, grid px; `NaN` where none was read.
    shift: [f64; 2],
    zncc: f64,
    /// The peak's curvature, the least-squares weight of a fitted cell.
    curvature: f64,
}

impl CellReading {
    fn refused(status: CellStatus, zncc: f64) -> CellReading {
        CellReading {
            status,
            shift: [f64::NAN; 2],
            zncc,
            curvature: 0.0,
        }
    }
}

/// The ZNCC of a cell's template against the working patch moved by
/// `(dx, dy)` whole grid px, averaged over the template's textured channels.
/// A channel the moved window is flat in, or the member image lacks, scores
/// `0`. `None` when a sample the window reads is missing.
///
/// The window's sum and sum of squares come from the patch's summed-area
/// tables; only the cross term with the mean-removed template is a pass over
/// the window, and it reads the patch in place.
#[allow(clippy::too_many_arguments)]
fn cell_zncc_at(
    tc: &CellTemplate,
    src_channels: &[usize],
    patch: &WorkingPatch,
    layout: &CellLayout,
    row: usize,
    col: usize,
    dx: i64,
    dy: i64,
) -> Option<f64> {
    let b = layout.bounds;
    let m = layout.margin as i64;
    let side = patch.side;
    let (x0, y0) = (b[col] as i64 + dx + m, b[row] as i64 + dy + m);
    let (w, h) = (b[col + 1] - b[col], b[row + 1] - b[row]);
    // The search never reads past the render: the margin covers the
    // whole-shift reach on every side.
    debug_assert!(
        x0 >= 0 && y0 >= 0 && x0 as usize + w <= side && y0 as usize + h <= side,
        "cell ({row}, {col}) at shift ({dx}, {dy}) reads from ({x0}, {y0}) \
         outside the {side}x{side} working patch"
    );
    let (x0, y0) = (x0 as usize, y0 as usize);
    let win = Window {
        x0,
        x1: x0 + w,
        y0,
        y1: y0 + h,
        stride: side + 1,
    };
    let count = (w * h) as f64;
    let sums = &patch.sums;
    let (mut sum, mut scored) = (0.0, 0usize);
    let mut checked = false;
    for (tch, t) in tc.channels.iter().enumerate() {
        let Some((t, t_norm)) = t else {
            continue;
        };
        scored += 1;
        let src = src_channels[tch];
        if src >= patch.channels {
            continue;
        }
        if !checked {
            if win.rect(&sums.missing) > 0 {
                return None;
            }
            checked = true;
        }
        let su = win.rect(&sums.sum[src]);
        let norm_sq = win.rect(&sums.sum_sq[src]) - su * su / count;
        if norm_sq < FLAT_NORM_SQ_EPS {
            continue;
        }
        let k = sums.offset[src];
        let mut cross = 0.0;
        let mut tv = t.iter();
        for y in y0..y0 + h {
            let line = &patch.samples[(y * side + x0) * patch.channels..];
            for x in 0..w {
                let u = f64::from(line[x * patch.channels + src]) - k;
                cross += u * tv.next().expect("one template sample per window sample");
            }
        }
        sum += cross / (norm_sq * t_norm).sqrt();
    }
    (scored > 0).then(|| sum / scored as f64)
}

/// Register one cell against the working patch: the ZNCC at every whole
/// shift within the reach, the best of them (first in row-major order on a
/// tie), and a quadratic fitted to its three-by-three neighbourhood for the
/// sub-pixel peak and its curvature.
fn read_cell(
    tc: &CellTemplate,
    src_channels: &[usize],
    patch: &WorkingPatch,
    layout: &CellLayout,
    params: &PiecewiseParams,
    row: usize,
    col: usize,
) -> CellReading {
    if tc.missing {
        return CellReading::refused(CellStatus::NotAttempted, f64::NAN);
    }
    if tc.channels.iter().all(Option::is_none) {
        return CellReading::refused(CellStatus::RefusedCurvature, f64::NAN);
    }
    let n = layout.reach as i64;
    let w = (2 * n + 1) as usize;
    let mut surface = vec![0.0f64; w * w];
    let mut best = (0usize, f64::NEG_INFINITY);
    for dy in -n..=n {
        for dx in -n..=n {
            let Some(z) = cell_zncc_at(tc, src_channels, patch, layout, row, col, dx, dy) else {
                return CellReading::refused(CellStatus::NotAttempted, f64::NAN);
            };
            let k = ((dy + n) as usize) * w + (dx + n) as usize;
            surface[k] = z;
            if z > best.1 {
                best = (k, z);
            }
        }
    }
    let (bx, by) = ((best.0 % w) as i64 - n, (best.0 / w) as i64 - n);
    if bx.abs() >= n || by.abs() >= n {
        // No whole-pixel neighbour on one side: the optimum is at or past the
        // bound.
        return CellReading::refused(CellStatus::RefusedBound, best.1);
    }
    let at = |dx: i64, dy: i64| surface[((by + dy + n) as usize) * w + (bx + dx + n) as usize];
    let z0 = at(0, 0);
    let gx = (at(1, 0) - at(-1, 0)) / 2.0;
    let gy = (at(0, 1) - at(0, -1)) / 2.0;
    // Negated Hessian of the ZNCC: positive definite at a peak.
    let kxx = 2.0 * z0 - at(1, 0) - at(-1, 0);
    let kyy = 2.0 * z0 - at(0, 1) - at(0, -1);
    let kxy = -(at(1, 1) - at(1, -1) - at(-1, 1) + at(-1, -1)) / 4.0;
    let half_trace = (kxx + kyy) / 2.0;
    let gap = (((kxx - kyy) / 2.0).powi(2) + kxy * kxy).sqrt();
    let curvature = half_trace - gap;
    let peaked = curvature > 0.0 && curvature >= f64::from(params.min_cell_curvature);
    if !peaked {
        return CellReading::refused(CellStatus::RefusedCurvature, best.1);
    }
    // The quadratic's peak, `K⁻¹·g`, kept within the whole pixel it was found
    // in.
    let det = kxx * kyy - kxy * kxy;
    let ox = ((kyy * gx - kxy * gy) / det).clamp(-0.5, 0.5);
    let oy = ((kxx * gy - kxy * gx) / det).clamp(-0.5, 0.5);
    let peak = z0 + gx * ox + gy * oy - 0.5 * (kxx * ox * ox + 2.0 * kxy * ox * oy + kyy * oy * oy);
    let zncc = peak.min(1.0);
    let status = if zncc < f64::from(params.min_cell_zncc) {
        CellStatus::RefusedZncc
    } else {
        CellStatus::Fitted
    };
    CellReading {
        status,
        shift: [bx as f64 + ox, by as f64 + oy],
        zncc,
        curvature,
    }
}

/// The affine update `c ↦ A·c + b` in centred grid coordinates.
#[derive(Clone, Copy, Debug)]
struct Update {
    a: Mat2,
    b: [f64; 2],
}

impl Update {
    /// How far the update moves the point `c`.
    fn moves(&self, c: [f64; 2]) -> [f64; 2] {
        [
            (self.a[0][0] - 1.0) * c[0] + self.a[0][1] * c[1] + self.b[0],
            self.a[1][0] * c[0] + (self.a[1][1] - 1.0) * c[1] + self.b[1],
        ]
    }
}

/// Which model the update was fitted with, by the survivors' count and
/// spread.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum UpdateModel {
    Affine,
    Similarity,
    Shift,
}

/// One fitted cell as the update's fit reads it: its centre, its shift and
/// its weight.
type Correspondence = ([f64; 2], [f64; 2], f64);

/// The weighted least-squares update through the surviving cells: each is a
/// correspondence `centre → centre + shift` with weight `weight`.
///
/// The model is chosen by the survivors' count and spread, falling back one
/// step at a time:
///
/// - a full affine from five or more cells whose spread is that of at least
///   two rows or two columns of cells;
/// - a similarity from three or more cells otherwise. That covers three or
///   four cells, five or more that are not spread over two rows and two
///   columns (a single row or column pins only one axis of an affine), and
///   five or more well-spread cells whose affine normal equations are
///   singular;
/// - a shift from one or two cells, or from three or more whose spread is
///   zero.
///
/// `None` without a survivor, or when the fitted update reflects or is not
/// finite.
fn fit_update(points: &[Correspondence], cell_side: f64) -> Option<(Update, UpdateModel)> {
    let total: f64 = points.iter().map(|p| p.2).sum();
    let weighted = total > 0.0;
    if points.is_empty() || !weighted {
        return None;
    }
    let mean = |f: &dyn Fn(&Correspondence) -> [f64; 2]| {
        let mut m = [0.0; 2];
        for p in points {
            let v = f(p);
            m[0] += p.2 * v[0];
            m[1] += p.2 * v[1];
        }
        [m[0] / total, m[1] / total]
    };
    let src_mean = mean(&|p| p.0);
    let dst_mean = mean(&|p| [p.0[0] + p.1[0], p.0[1] + p.1[1]]);
    // Centred second moments: `sxx = Σ w·x·xᵀ`, `syx = Σ w·y·xᵀ`.
    let mut sxx = [[0.0; 2]; 2];
    let mut syx = [[0.0; 2]; 2];
    for (c, d, w) in points {
        let x = [c[0] - src_mean[0], c[1] - src_mean[1]];
        let y = [c[0] + d[0] - dst_mean[0], c[1] + d[1] - dst_mean[1]];
        for i in 0..2 {
            for j in 0..2 {
                sxx[i][j] += w * x[i] * x[j];
                syx[i][j] += w * y[i] * x[j];
            }
        }
    }
    let spread_sq = sxx[0][0] + sxx[1][1];
    let half_trace = spread_sq / 2.0;
    let gap = (((sxx[0][0] - sxx[1][1]) / 2.0).powi(2) + sxx[0][1] * sxx[0][1]).sqrt();
    let least_spread = (half_trace - gap) / total;
    // Two full rows of cells a cell apart have a variance of `side²/4` across
    // them; half of that still pins the affine's second axis.
    let well_spread = least_spread >= cell_side * cell_side / 8.0;
    let n = points.len();
    let shift_only = || {
        (
            Update {
                a: [[1.0, 0.0], [0.0, 1.0]],
                b: [dst_mean[0] - src_mean[0], dst_mean[1] - src_mean[1]],
            },
            UpdateModel::Shift,
        )
    };
    let affine = (n >= 5 && well_spread)
        .then(|| {
            let det = sxx[0][0] * sxx[1][1] - sxx[0][1] * sxx[1][0];
            let invertible = det.abs() > 1e-12 * spread_sq * spread_sq;
            invertible.then(|| {
                let inv = [
                    [sxx[1][1] / det, -sxx[0][1] / det],
                    [-sxx[1][0] / det, sxx[0][0] / det],
                ];
                (mul2(&syx, &inv), UpdateModel::Affine)
            })
        })
        .flatten();
    let similarity = || {
        (n >= 3 && spread_sq > 1e-12).then(|| {
            let a = (syx[0][0] + syx[1][1]) / spread_sq;
            let b = (syx[1][0] - syx[0][1]) / spread_sq;
            ([[a, -b], [b, a]], UpdateModel::Similarity)
        })
    };
    let Some((a, model)) = affine.or_else(similarity) else {
        return Some(shift_only());
    };
    let b = [
        dst_mean[0] - (a[0][0] * src_mean[0] + a[0][1] * src_mean[1]),
        dst_mean[1] - (a[1][0] * src_mean[0] + a[1][1] * src_mean[1]),
    ];
    let det = a[0][0] * a[1][1] - a[0][1] * a[1][0];
    let orientable = det > 0.0;
    if !orientable || !b.iter().all(|v| v.is_finite()) {
        return None;
    }
    Some((Update { a, b }, model))
}

/// The robust update through the surviving cells: [`fit_update`] reweighted
/// by a Tukey biweight on each cell's residual to the update fitted in the
/// round before.
///
/// The first fit takes each cell's own weight (its peak's curvature). Each of
/// the [`IRLS_ROUNDS`] that follow measures every cell's residual to the last
/// fit, the distance from `c + d` to `A·c + b`, takes the residual scale as
/// [`MEDIAN_LENGTH_TO_SIGMA`] times the median residual length, floored at
/// [`MIN_RESIDUAL_SCALE_PX`], and multiplies the cell's own weight by the
/// biweight `(1 − (r / (TUKEY_CUTOFF·scale))²)²`, which is `0` at and past the
/// cut-off. The update is refitted to the cells with a positive weight, so the
/// model [`fit_update`] chooses follows the count and spread of those.
///
/// Returns the last fit and each cell's last weight, parallel to `points`;
/// `None` when a fit fails (see [`fit_update`]). The median residual is under
/// the cut-off, so at least half of the cells keep a positive weight.
fn fit_update_robust(
    points: &[Correspondence],
    cell_side: f64,
) -> Option<(Update, UpdateModel, Vec<f64>)> {
    let fit_with = |weights: &[f64]| {
        let kept: Vec<Correspondence> = points
            .iter()
            .zip(weights)
            .filter(|(_, &w)| w > 0.0)
            .map(|(&(c, d, _), &w)| (c, d, w))
            .collect();
        fit_update(&kept, cell_side)
    };
    let prior: Vec<f64> = points.iter().map(|p| p.2).collect();
    let mut weights = prior.clone();
    let (mut update, mut model) = fit_with(&weights)?;
    for _ in 0..IRLS_ROUNDS {
        let residuals: Vec<f64> = points
            .iter()
            .map(|(c, d, _)| {
                let m = update.moves(*c);
                (d[0] - m[0]).hypot(d[1] - m[1])
            })
            .collect();
        let median = median_in_place(&mut residuals.clone());
        let scale = (MEDIAN_LENGTH_TO_SIGMA * median).max(MIN_RESIDUAL_SCALE_PX);
        let cut = TUKEY_CUTOFF * scale;
        weights = prior
            .iter()
            .zip(&residuals)
            .map(|(&w, &r)| {
                if r < cut {
                    let u = r / cut;
                    w * (1.0 - u * u).powi(2)
                } else {
                    0.0
                }
            })
            .collect();
        (update, model) = fit_with(&weights)?;
    }
    Some((update, model, weights))
}

/// One render's readings with the update fitted to them.
#[derive(Clone, Copy, Debug)]
struct Pass {
    readings: [[CellReading; 3]; 3],
    update: Update,
    model: UpdateModel,
    /// Each cell's weight in the update's last fit: `0` for a cell that was
    /// not fitted, or that the robust fit refused.
    weights: [[f64; 3]; 3],
}

/// Fit the update to one render's readings: the cells whose status is
/// [`CellStatus::Fitted`] are the correspondences, each weighted by its
/// peak's curvature, and [`fit_update_robust`] fits them. `None` when no cell
/// survived or the fit failed.
fn fit_pass(readings: [[CellReading; 3]; 3], layout: &CellLayout) -> Option<Pass> {
    let fitted: Vec<(usize, usize)> = (0..9)
        .map(|i| (i / 3, i % 3))
        .filter(|&(r, c)| readings[r][c].status == CellStatus::Fitted)
        .collect();
    let points: Vec<Correspondence> = fitted
        .iter()
        .map(|&(r, c)| {
            let rd = readings[r][c];
            (layout.centres[r][c], rd.shift, rd.curvature)
        })
        .collect();
    let (update, model, w) = fit_update_robust(&points, layout.cell_side)?;
    let mut weights = [[0.0; 3]; 3];
    for (&(r, c), w) in fitted.iter().zip(w) {
        weights[r][c] = w;
    }
    Some(Pass {
        readings,
        update,
        model,
        weights,
    })
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
/// With `move_shape`, it runs the two-level loop. Each iteration renders the working patch at the current shape, reads the
/// nine cells and fits an update to the cells that survive (see
/// [`fit_update_robust`]). `score` reads the whole-member windowed ZNCC the
/// cascade maximised at a shape and position, `None` when the support leaves
/// the frame. The loop stops at the first of:
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
    let pass = |shape: &Mat2, position: [f64; 2]| {
        let patch = WorkingPatch::render(pyramid, position, shape, layout, step, off)?;
        let readings: [[CellReading; 3]; 3] = std::array::from_fn(|row| {
            std::array::from_fn(|col| {
                read_cell(
                    &templates.cells[row][col],
                    src_channels,
                    &patch,
                    layout,
                    params,
                    row,
                    col,
                )
            })
        });
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
