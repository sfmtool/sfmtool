// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Piecewise refinement of a kept member: the nine cells of the template are
//! registered separately against the member's photograph, and their shifts
//! are fitted back to an update of the member's affine shape.
//!
//! See `specs/drafts/cluster-patches-piecewise-refinement.md`. The fit is a
//! loop with two levels. The outer level renders the photograph through the
//! member's current affine shape into a working patch: the template's `R×R`
//! grid plus a margin of [`PiecewiseParams::cell_shift_bound_px`] on every
//! side. The inner level registers each of the nine cells of the reference's
//! square against that working patch by ZNCC over whole-pixel shifts within the
//! bound, reads a sub-pixel peak and its curvature, refuses flat and
//! low-scoring cells, and fits a weighted least-squares update of the affine
//! shape to the cells that remain. The loop renders again with the updated
//! shape until the update moves no cell centre by more than
//! [`PiecewiseParams::update_tolerance_px`], or until
//! [`PiecewiseParams::max_iterations`].
//!
//! The cells are the `[0, R/3, R - R/3, R]` split the ZNCC grid uses. Shifts
//! and cell centres are in template grid px. A grid position `g` (column,
//! row) is carried into the member's image by `p + S·(off + step·g)`, which in
//! grid coordinates centred on the middle of the grid, `c = g − (R − 1)/2`, is
//! `p + S·(step·c)`. An update `c ↦ A·c + b` in those coordinates therefore
//! composes into the shape as `S' = S·A`, `p' = p + S·(step·b)`.

use crate::camera::image::ImageU8Pyramid;
use crate::patch::normal_refine::{grid_bounds, FLAT_NORM_SQ_EPS};

use super::kernels::TemplateKernel;
use super::{mul2, sample_grid_window, GridEdge, Mat2};

#[cfg(test)]
mod tests;

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
    /// Search bound for a cell's shift from its affine placement, in template
    /// grid px. The working patch is rendered with a margin of this many grid
    /// px on each side (rounded up), and the shift search covers the whole
    /// shifts within it (rounded down), so no shift reads past the render.
    /// The sub-pixel peak needs a whole-pixel neighbour on each side, so a
    /// cell whose best whole shift lies on the bound is refused, and a bound
    /// under `1` refuses every cell.
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
    /// than this, in grid px.
    pub update_tolerance_px: f32,
    /// The most renders the loop makes for one member.
    pub max_iterations: u8,
}

impl Default for PiecewiseParams {
    fn default() -> Self {
        Self {
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
/// The discriminants are the values a `u8` column of cell statuses would
/// carry.
#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CellStatus {
    /// The cell's shift was measured and the affine update was fitted to it.
    Fitted = 0,
    /// The cell's own reading refused it: the reference is flat over the cell,
    /// the ZNCC peak is flatter than
    /// [`PiecewiseParams::min_cell_curvature`], or the best whole shift lies
    /// on the search bound.
    RefusedCurvature = 1,
    /// The cell's ZNCC at its optimum is below
    /// [`PiecewiseParams::min_cell_zncc`]: it is over a different surface.
    RefusedZncc = 2,
    /// The cell was not registered: a sample the search needs could not be
    /// read, or no cell of the member survived, in which case the member
    /// keeps its cascade shape and every cell is stored this way.
    NotAttempted = 3,
}

/// Per-cell registration of one member against the template, as the residual
/// to the member's converged affine shape. `[row][col]` from the top-left
/// cell, rows and columns cut at `R/3` and `R - R/3`.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct CellRefinement {
    /// Displacement of each cell's centre from where the converged affine
    /// shape places it, in template grid px, `[x, y]` along the grid's columns
    /// and rows. Measured at the loop's last render, less the last update.
    /// `NaN` where no sub-pixel shift was measured: a cell refused by
    /// curvature or not attempted.
    pub shift_px: [[[f32; 2]; 3]; 3],
    /// ZNCC of each cell at its optimum at the last render: the sub-pixel
    /// peak's value where one was read, the best whole shift's otherwise.
    /// `NaN` where nothing was read.
    pub zncc: [[f32; 3]; 3],
    /// Each cell's status at the last render.
    pub status: [[CellStatus; 3]; 3],
    /// Renders the loop made before its update fell below
    /// [`PiecewiseParams::update_tolerance_px`], or the cap
    /// [`PiecewiseParams::max_iterations`].
    pub iterations: u8,
}

impl CellRefinement {
    /// Every cell not attempted, after `iterations` renders.
    pub fn not_attempted(iterations: u8) -> CellRefinement {
        CellRefinement {
            shift_px: [[[f32::NAN; 2]; 3]; 3],
            zncc: [[f32::NAN; 3]; 3],
            status: [[CellStatus::NotAttempted; 3]; 3],
            iterations,
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
        let mid = (resolution as f64 - 1.0) / 2.0;
        let span_centre = |t: usize| (bounds[t] + bounds[t + 1] - 1) as f64 / 2.0 - mid;
        let centres = std::array::from_fn(|row| {
            std::array::from_fn(|col| [span_centre(col), span_centre(row)])
        });
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
/// interleaved `side × side × channels`, `NaN` where a sample left the image.
pub(super) struct WorkingPatch {
    pub(super) side: usize,
    pub(super) channels: usize,
    pub(super) samples: Vec<f32>,
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
        Some(WorkingPatch {
            side,
            channels,
            samples,
        })
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
    window: &mut Vec<f64>,
) -> Option<f64> {
    let b = layout.bounds;
    let m = layout.margin as i64;
    let side = patch.side as i64;
    let (mut sum, mut scored) = (0.0, 0usize);
    for (tch, t) in tc.channels.iter().enumerate() {
        let Some((t, t_norm)) = t else {
            continue;
        };
        scored += 1;
        let src = src_channels[tch];
        if src >= patch.channels {
            continue;
        }
        window.clear();
        for y in b[row]..b[row + 1] {
            let py = y as i64 + dy + m;
            for x in b[col]..b[col + 1] {
                let px = x as i64 + dx + m;
                // The search never reads past the render: the margin covers
                // the whole-shift reach on every side.
                debug_assert!(
                    (0..side).contains(&px) && (0..side).contains(&py),
                    "cell ({row}, {col}) at shift ({dx}, {dy}) reads ({px}, {py}) \
                     outside the {side}x{side} working patch"
                );
                let v = patch.samples[(py * side + px) as usize * patch.channels + src];
                if !v.is_finite() {
                    return None;
                }
                window.push(f64::from(v));
            }
        }
        let mean = window.iter().sum::<f64>() / window.len() as f64;
        let (mut cross, mut norm_sq) = (0.0, 0.0);
        for (u, tv) in window.iter().zip(t) {
            let du = u - mean;
            cross += du * tv;
            norm_sq += du * du;
        }
        if norm_sq < FLAT_NORM_SQ_EPS {
            continue;
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
    let mut window = Vec::new();
    let mut best = (0usize, f64::NEG_INFINITY);
    for dy in -n..=n {
        for dx in -n..=n {
            let Some(z) = cell_zncc_at(
                tc,
                src_channels,
                patch,
                layout,
                row,
                col,
                dx,
                dy,
                &mut window,
            ) else {
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
        return CellReading::refused(CellStatus::RefusedCurvature, best.1);
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
/// correspondence `centre → centre + shift` with weight `weight`. A full
/// affine from five or more cells whose spread is that of at least two rows or
/// two columns of cells, a similarity from three or more otherwise, a shift
/// from one or two. `None` without a survivor or with a degenerate fit.
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
    let (a, model) = if n >= 5 && well_spread {
        let det = sxx[0][0] * sxx[1][1] - sxx[0][1] * sxx[1][0];
        let invertible = det.abs() > 1e-12 * spread_sq * spread_sq;
        if !invertible {
            return Some(shift_only());
        }
        let inv = [
            [sxx[1][1] / det, -sxx[0][1] / det],
            [-sxx[1][0] / det, sxx[0][0] / det],
        ];
        (mul2(&syx, &inv), UpdateModel::Affine)
    } else if n >= 3 && spread_sq > 1e-12 {
        let a = (syx[0][0] + syx[1][1]) / spread_sq;
        let b = (syx[1][0] - syx[0][1]) / spread_sq;
        ([[a, -b], [b, a]], UpdateModel::Similarity)
    } else {
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

/// The outcome of the piecewise refinement of one member.
pub(super) struct MemberCells {
    /// The refined absolute shape and position; the cascade's when nothing
    /// survived.
    pub(super) shape: Mat2,
    pub(super) position: [f64; 2],
    pub(super) cells: CellRefinement,
    /// Whether the shape or position differs from the cascade's.
    pub(super) updated: bool,
    /// The model the last update was fitted with.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(super) model: Option<UpdateModel>,
}

/// Run the two-level loop for one kept member, starting from its cascade
/// shape `s` and position `p`.
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
) -> MemberCells {
    let unchanged = |iterations: u8| MemberCells {
        shape: s,
        position: p,
        cells: CellRefinement::not_attempted(iterations),
        updated: false,
        model: None,
    };
    let (mut shape, mut position) = (s, p);
    let mut iterations = 0u8;
    let mut last = None;
    while iterations < params.max_iterations {
        iterations += 1;
        let Some(patch) = WorkingPatch::render(pyramid, position, &shape, layout, step, off) else {
            return unchanged(iterations);
        };
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
        let points: Vec<Correspondence> = (0..9)
            .map(|i| (i / 3, i % 3))
            .filter(|&(r, c)| readings[r][c].status == CellStatus::Fitted)
            .map(|(r, c)| {
                let rd = readings[r][c];
                (layout.centres[r][c], rd.shift, rd.curvature)
            })
            .collect();
        let Some((update, model)) = fit_update(&points, layout.cell_side) else {
            // No cell survived, or the fit is degenerate: the member keeps
            // its cascade shape.
            return unchanged(iterations);
        };
        // Compose `c ↦ A·c + b` into the shape: `S' = S·A`,
        // `p' = p + S·(step·b)`.
        let sb = [update.b[0] * step, update.b[1] * step];
        position = [
            position[0] + shape[0][0] * sb[0] + shape[0][1] * sb[1],
            position[1] + shape[1][0] * sb[0] + shape[1][1] * sb[1],
        ];
        shape = mul2(&shape, &update.a);
        let largest_move = layout
            .centres
            .iter()
            .flatten()
            .map(|&c| {
                let m = update.moves(c);
                m[0].hypot(m[1])
            })
            .fold(0.0f64, f64::max);
        last = Some((readings, update, model));
        if largest_move <= f64::from(params.update_tolerance_px) {
            break;
        }
    }
    let Some((readings, update, model)) = last else {
        return unchanged(iterations);
    };
    if !shape
        .iter()
        .flatten()
        .chain(position.iter())
        .all(|v| v.is_finite())
    {
        return unchanged(iterations);
    }
    // The residual to the shape the loop returns: the shift measured at the
    // last render, less what the last update moved the cell centre by.
    let mut cells = CellRefinement::not_attempted(iterations);
    for (row, line) in readings.iter().enumerate() {
        for (col, rd) in line.iter().enumerate() {
            cells.status[row][col] = rd.status;
            cells.zncc[row][col] = rd.zncc as f32;
            if rd.shift.iter().all(|v| v.is_finite()) {
                let m = update.moves(layout.centres[row][col]);
                cells.shift_px[row][col] =
                    [(rd.shift[0] - m[0]) as f32, (rd.shift[1] - m[1]) as f32];
            }
        }
    }
    MemberCells {
        shape,
        position,
        cells,
        updated: true,
        model: Some(model),
    }
}
