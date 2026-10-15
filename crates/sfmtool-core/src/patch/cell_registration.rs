// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Cell registration: the nine cells of a reference's `R×R` grid registered
//! separately against a working patch, each cell gated on its own reading,
//! and a robust affine map fitted to the shifts of the cells that remain.
//!
//! The working patch is a view's photograph rendered on the reference's grid
//! plus a margin on every side, interleaved samples with `NaN` where a sample
//! left the image. How it is rendered is the caller's: the piecewise
//! refinement of cluster patches
//! ([`refine_cluster_patches`](crate::patch::cluster_refine::refine_cluster_patches))
//! renders it through a member's affine shape. Everything after the render is
//! here, starting at [`WorkingPatch::new`].
//!
//! Each cell of the reference's square is compared with the working patch by
//! ZNCC at every whole shift within the search bound, read from the patch's
//! summed-area tables. At the best whole shift a quadratic fitted to the
//! three-by-three neighbourhood gives the sub-pixel shift and the peak's
//! curvature. A cell is refused for curvature (the reference is flat over it,
//! or its peak is flatter than the curvature bar), for ZNCC (its score at the
//! optimum is below the ZNCC bar), or for the bound (its best whole shift lies
//! on the search bound). An affine map `c ↦ A·c + b` in grid coordinates
//! centred on the grid is fitted to the cells that remain by iteratively
//! reweighted least squares with a Tukey biweight; a cell the fit gives
//! weight `0` disagrees with the others.
//!
//! The cells are the `[0, R/3, R - R/3, R]` split the ZNCC grid uses, with the
//! centres `grid_cell_centres` gives. Shifts and cell centres are in grid px.
//! See `specs/core/patch/cluster-patch-refinement.md` § "Piecewise
//! refinement".

use crate::numeric::median_in_place;
use crate::patch::normal_refine::{grid_bounds, grid_cell_centres, FLAT_NORM_SQ_EPS};

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
pub(in crate::patch) const MEDIAN_LENGTH_TO_SIGMA: f64 = 0.849_321_800_288_019_1;

/// The smallest residual scale the reweighting uses, grid px. Without it a
/// fit through cells that agree to a few hundredths of a pixel would refuse a
/// cell a tenth of a pixel off.
const MIN_RESIDUAL_SCALE_PX: f64 = 0.1;

/// What the registration concluded about one cell in one view.
///
/// The discriminants are the canonical codes of the `.matches` cell-status
/// column,
/// [`ClusterCellStatus`](sfmtool_matches_format::ClusterCellStatus), which
/// [`member_cell_data`](crate::patch::cluster_refine::member_cell_data)
/// stores them as.
#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CellStatus {
    /// The cell's shift was measured and the robust affine fit over the
    /// member's cells gave it weight.
    Fitted = 0,
    /// The cell's own reading refused it: the reference is flat over the cell,
    /// or the ZNCC peak is flatter than
    /// [`PiecewiseParams::min_cell_curvature`](crate::patch::cluster_refine::PiecewiseParams::min_cell_curvature).
    RefusedCurvature = 1,
    /// The cell's ZNCC at its optimum is below
    /// [`PiecewiseParams::min_cell_zncc`](crate::patch::cluster_refine::PiecewiseParams::min_cell_zncc):
    /// it is over a different surface.
    RefusedZncc = 2,
    /// The cell was not registered: a sample the search needs could not be
    /// read, or no cell of the member survived. Every cell is stored this way
    /// when a render failed or the fitted map was not finite or reflected,
    /// and, with
    /// [`PiecewiseParams::move_shape`](crate::patch::cluster_refine::PiecewiseParams::move_shape),
    /// when the refined shape could not be accepted (the readings at it
    /// failed the member's gates, or its support left the frame). The member
    /// then keeps its cascade shape, and the iteration count includes the
    /// pass that failed.
    NotAttempted = 3,
    /// The cell's best whole shift lies on the search bound
    /// [`PiecewiseParams::cell_shift_bound_px`](crate::patch::cluster_refine::PiecewiseParams::cell_shift_bound_px),
    /// so no whole-pixel neighbour on that side was read and no sub-pixel
    /// peak can be fitted: the cell's optimum is at or past the bound.
    RefusedBound = 4,
    /// The cell passed its own gates, but its shift disagrees with the
    /// affine map the other cells agree on by so much that the robust fit
    /// gave it weight `0`. Its shift is measured and stored.
    RefusedOutlier = 5,
}

/// The two bars a cell's reading is gated by. The third gate, the search
/// bound, is the [`CellLayout`]'s.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct CellGateOptions {
    /// A cell whose ZNCC at its optimum is below this is refused as
    /// [`CellStatus::RefusedZncc`].
    pub(crate) min_cell_zncc: f32,
    /// A cell whose ZNCC peak is flatter than this along its flattest
    /// direction is refused as [`CellStatus::RefusedCurvature`]: the smaller
    /// eigenvalue of the negated Hessian of the ZNCC over the shift, in ZNCC
    /// per grid px².
    pub(crate) min_cell_curvature: f32,
}

/// The fixed geometry of the cells and the search for one grid size and one
/// bound.
pub(crate) struct CellLayout {
    /// `R`, the template's samples per axis.
    pub(crate) resolution: usize,
    /// Working-patch margin on each side, grid px: the bound rounded up.
    pub(crate) margin: usize,
    /// Half-width of the whole-shift search: the bound rounded down, at most
    /// `margin`.
    pub(crate) reach: usize,
    /// `[0, R/3, R - R/3, R]`.
    pub(crate) bounds: [usize; 4],
    /// Cell centres in grid coordinates centred on the grid, `[x, y]`.
    pub(crate) centres: [[[f64; 2]; 3]; 3],
    /// The side of a corner cell, grid px.
    pub(in crate::patch) cell_side: f64,
}

impl CellLayout {
    /// The layout of an `R = resolution` grid searched within `bound` grid px
    /// of each cell's placement. The working patch is rendered with a margin
    /// of the bound rounded up, and the search covers the whole shifts within
    /// the bound rounded down, so no shift reads past the render. A
    /// non-finite or negative bound is `0`.
    pub(crate) fn new(resolution: u32, bound: f32) -> CellLayout {
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
    pub(in crate::patch) fn patch_side(&self) -> usize {
        self.resolution + 2 * self.margin
    }
}

/// The reference's samples over one cell, per template channel.
pub(crate) struct CellTemplate {
    /// Per template channel: the mean-removed samples, row-major over the
    /// cell, and their sum of squares; `None` where the channel is flat over
    /// the cell.
    pub(crate) channels: Vec<Option<(Vec<f64>, f64)>>,
    /// Some pixel of the cell could not be sampled in the reference.
    pub(crate) missing: bool,
}

/// The reference's samples over each of the nine cells.
pub(crate) struct CellTemplates {
    pub(crate) cells: [[CellTemplate; 3]; 3],
}

impl CellTemplates {
    /// Cut the nine cells out of the reference's samples over the whole
    /// square: `channels` planes of `R×R` samples one after the other, each
    /// row-major, non-finite where a sample could not be read.
    pub(crate) fn from_planes(
        square_samples: &[f32],
        channels: usize,
        layout: &CellLayout,
    ) -> CellTemplates {
        let r = layout.resolution;
        let square = r * r;
        let b = layout.bounds;
        let cells = std::array::from_fn(|row| {
            std::array::from_fn(|col| {
                let mut missing = false;
                let channels = (0..channels)
                    .map(|c| {
                        let plane = &square_samples[c * square..][..square];
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

/// A view's photograph rendered on the template's grid plus the margin,
/// interleaved `side × side × channels`, `NaN` where a sample left the image,
/// with the summed-area tables the cell search reads its window sums from.
pub(crate) struct WorkingPatch {
    pub(in crate::patch) side: usize,
    pub(in crate::patch) channels: usize,
    pub(in crate::patch) samples: Vec<f32>,
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
    pub(crate) fn new(side: usize, channels: usize, samples: Vec<f32>) -> WorkingPatch {
        let sums = PatchSums::new(side, channels, &samples);
        WorkingPatch {
            side,
            channels,
            samples,
            sums,
        }
    }
}

/// One cell's reading at one render.
#[derive(Clone, Copy, Debug)]
pub(crate) struct CellReading {
    pub(crate) status: CellStatus,
    /// Sub-pixel shift, grid px; `NaN` where none was read.
    pub(crate) shift: [f64; 2],
    /// The best whole shift of the search, grid px; `None` where the search
    /// did not run to its end.
    pub(crate) whole_shift: Option<[i64; 2]>,
    /// The ZNCC at the optimum: the sub-pixel peak's where one was fitted, the
    /// best whole shift's where the cell was refused before that, `NaN` where
    /// the search did not run to its end.
    pub(crate) zncc: f64,
    /// The peak's curvature, the least-squares weight of a fitted cell; `0`
    /// where no peak was fitted.
    pub(crate) curvature: f64,
}

impl CellReading {
    fn refused(status: CellStatus, zncc: f64, whole_shift: Option<[i64; 2]>) -> CellReading {
        CellReading {
            status,
            shift: [f64::NAN; 2],
            whole_shift,
            zncc,
            curvature: 0.0,
        }
    }
}

/// The ZNCC of a cell's template against the working patch moved by
/// `(dx, dy)` whole grid px, averaged over the template's textured channels.
/// A channel the moved window is flat in, or the view's image lacks, scores
/// `0`. `None` when a sample the window reads is missing.
///
/// The window's sum and sum of squares come from the patch's summed-area
/// tables; only the cross term with the mean-removed template is a pass over
/// the window, and it reads the patch in place.
#[allow(clippy::too_many_arguments)]
pub(in crate::patch) fn cell_zncc_at(
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
/// sub-pixel peak and its curvature, gated by `gates` and by the bound.
///
/// `src_channels[c]` is the working patch's channel that template channel `c`
/// is compared with.
pub(in crate::patch) fn read_cell(
    tc: &CellTemplate,
    src_channels: &[usize],
    patch: &WorkingPatch,
    layout: &CellLayout,
    gates: &CellGateOptions,
    row: usize,
    col: usize,
) -> CellReading {
    if tc.missing {
        return CellReading::refused(CellStatus::NotAttempted, f64::NAN, None);
    }
    if tc.channels.iter().all(Option::is_none) {
        return CellReading::refused(CellStatus::RefusedCurvature, f64::NAN, None);
    }
    let n = layout.reach as i64;
    let w = (2 * n + 1) as usize;
    let mut surface = vec![0.0f64; w * w];
    let mut best = (0usize, f64::NEG_INFINITY);
    for dy in -n..=n {
        for dx in -n..=n {
            let Some(z) = cell_zncc_at(tc, src_channels, patch, layout, row, col, dx, dy) else {
                return CellReading::refused(CellStatus::NotAttempted, f64::NAN, None);
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
        return CellReading::refused(CellStatus::RefusedBound, best.1, Some([bx, by]));
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
    let peaked = curvature > 0.0 && curvature >= f64::from(gates.min_cell_curvature);
    if !peaked {
        return CellReading::refused(CellStatus::RefusedCurvature, best.1, Some([bx, by]));
    }
    // The quadratic's peak, `K⁻¹·g`, kept within the whole pixel it was found
    // in.
    let det = kxx * kyy - kxy * kxy;
    let ox = ((kyy * gx - kxy * gy) / det).clamp(-0.5, 0.5);
    let oy = ((kxx * gy - kxy * gx) / det).clamp(-0.5, 0.5);
    let peak = z0 + gx * ox + gy * oy - 0.5 * (kxx * ox * ox + 2.0 * kxy * ox * oy + kyy * oy * oy);
    let zncc = peak.min(1.0);
    let status = if zncc < f64::from(gates.min_cell_zncc) {
        CellStatus::RefusedZncc
    } else {
        CellStatus::Fitted
    };
    CellReading {
        status,
        shift: [bx as f64 + ox, by as f64 + oy],
        whole_shift: Some([bx, by]),
        zncc,
        curvature,
    }
}

/// [`read_cell`] for each of the nine cells, `[row][col]` from the top-left.
pub(crate) fn read_cells(
    templates: &CellTemplates,
    src_channels: &[usize],
    patch: &WorkingPatch,
    layout: &CellLayout,
    gates: &CellGateOptions,
) -> [[CellReading; 3]; 3] {
    std::array::from_fn(|row| {
        std::array::from_fn(|col| {
            read_cell(
                &templates.cells[row][col],
                src_channels,
                patch,
                layout,
                gates,
                row,
                col,
            )
        })
    })
}

/// The affine update `c ↦ A·c + b` in centred grid coordinates.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Update {
    pub(crate) a: [[f64; 2]; 2],
    pub(crate) b: [f64; 2],
}

impl Update {
    /// How far the update moves the point `c`.
    pub(in crate::patch) fn moves(&self, c: [f64; 2]) -> [f64; 2] {
        [
            (self.a[0][0] - 1.0) * c[0] + self.a[0][1] * c[1] + self.b[0],
            self.a[1][0] * c[0] + (self.a[1][1] - 1.0) * c[1] + self.b[1],
        ]
    }
}

/// Which model the update was fitted with, by the survivors' count and
/// spread.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum UpdateModel {
    Affine,
    Similarity,
    Shift,
}

/// One fitted cell as the update's fit reads it: its centre, its shift and
/// its weight.
pub(in crate::patch) type Correspondence = ([f64; 2], [f64; 2], f64);

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
pub(in crate::patch) fn fit_update(
    points: &[Correspondence],
    cell_side: f64,
) -> Option<(Update, UpdateModel)> {
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
                // `A = syx · sxx⁻¹`.
                let a = [
                    [
                        syx[0][0] * inv[0][0] + syx[0][1] * inv[1][0],
                        syx[0][0] * inv[0][1] + syx[0][1] * inv[1][1],
                    ],
                    [
                        syx[1][0] * inv[0][0] + syx[1][1] * inv[1][0],
                        syx[1][0] * inv[0][1] + syx[1][1] * inv[1][1],
                    ],
                ];
                (a, UpdateModel::Affine)
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
pub(in crate::patch) fn fit_update_robust(
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
pub(crate) struct Pass {
    pub(in crate::patch) readings: [[CellReading; 3]; 3],
    pub(crate) update: Update,
    pub(crate) model: UpdateModel,
    /// Each cell's weight in the update's last fit: `0` for a cell that was
    /// not fitted, or that the robust fit refused.
    pub(crate) weights: [[f64; 3]; 3],
}

/// Fit the update to one render's readings: the cells whose status is
/// [`CellStatus::Fitted`] are the correspondences, each weighted by its
/// peak's curvature, and [`fit_update_robust`] fits them. `None` when no cell
/// survived or the fit failed.
pub(crate) fn fit_pass(readings: [[CellReading; 3]; 3], layout: &CellLayout) -> Option<Pass> {
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

#[cfg(test)]
mod tests;
