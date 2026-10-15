// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The detail record of [`grid_plane_normal`](super::grid_plane_normal): what
//! the step read and decided at every render, kept when
//! [`GridPlaneNormalOptions::record_detail`](super::GridPlaneNormalOptions::record_detail)
//! is set.
//!
//! Every value is recorded by the function that computes it for the step, as
//! it computes it; nothing here is computed a second time for the record.

use crate::patch::cell_plane_normals::{CellPlaneStatus, NormalDeterminacy};
use crate::patch::cluster_refine::CellStatus;

use super::GridPlaneIteration;

/// Every render of one [`grid_plane_normal`](super::grid_plane_normal), in
/// render order, carried by [`GridPlaneCells::detail`](super::GridPlaneCells::detail).
#[derive(Debug, Clone, PartialEq)]
pub struct GridPlaneDetail {
    /// The working grid and its cells, the same at every render.
    pub layout: GridPlaneLayout,
    /// One entry per render, parallel to
    /// [`GridPlaneCells::trace`](super::GridPlaneCells::trace).
    pub renders: Vec<GridPlaneRender>,
}

/// The working grid every render is made on and the nine cells cut from it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GridPlaneLayout {
    /// `R`, the side of the working grid, in samples.
    pub working_resolution: u32,
    /// The margin each non-reference view's tile is widened by on every side,
    /// grid px: the search bound rounded up. A view's tile is
    /// `R + 2 · margin` samples a side.
    pub margin: u32,
    /// The half-width of the whole-shift search, grid px: the bound rounded
    /// down.
    pub reach: u32,
    /// The cells' edges along each axis of the grid, `[0, R/3, R - R/3, R]`:
    /// cell `(row, col)` covers columns `bounds[col]..bounds[col + 1]` and
    /// rows `bounds[row]..bounds[row + 1]` of the reference's tile.
    pub cell_bounds: [u32; 4],
    /// Each cell's centre, `[row * 3 + col]`, `[x, y]` in grid px centred on
    /// the grid.
    pub cell_centres: [[f64; 2]; 9],
}

/// One render of the loop.
#[derive(Debug, Clone, PartialEq)]
pub struct GridPlaneRender {
    /// The frame the render was made at.
    pub frame: GridPlaneFrame,
    /// The reference's render, the tile the nine templates are cut from;
    /// `None` when the step has no reference or its render left the
    /// photograph.
    pub reference: Option<GridPlaneTile>,
    /// Each non-reference `in` view the render reached, in row order.
    pub views: Vec<GridPlaneView>,
    /// The triangulation of each cell, `[row * 3 + col]`.
    pub cells: [GridPlaneCell; 9],
    /// The plane through the cells and the turn it proposed.
    pub plane: GridPlanePlane,
    /// The render's entry of the loop's trace: its score, verdict, predicted
    /// precision, whether it was gated, the normal its turn reaches, the
    /// farthest that turn moves a cell centre, and whether the loop took it.
    pub iteration: GridPlaneIteration,
}

/// A patch frame, world frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GridPlaneFrame {
    /// The patch centre, world frame.
    pub center: [f64; 3],
    /// Unit, in-plane.
    pub u_axis: [f64; 3],
    /// Unit, in-plane, perpendicular to `u_axis`.
    pub v_axis: [f64; 3],
    /// `u_axis × v_axis`.
    pub normal: [f64; 3],
    /// Half-size along `(u, v)`, world units.
    pub half_extent: [f64; 2],
}

/// One view's rendered tile.
#[derive(Debug, Clone, PartialEq)]
pub struct GridPlaneTile {
    /// The track's observation the tile is of.
    pub observation: usize,
    /// The image of that observation.
    pub image: u32,
    /// Samples a side: `R` for the reference, `R + 2 · margin` for a view.
    pub side: u32,
    /// Channels per sample, 1 to 3.
    pub channels: u32,
    /// `side × side × channels`, interleaved, row-major; `NaN` where a sample
    /// left the photograph.
    pub samples: Vec<f32>,
}

/// One non-reference view at one render: its tile, its cells' readings and
/// its robust affine fit.
#[derive(Debug, Clone, PartialEq)]
pub struct GridPlaneView {
    /// The view's tile, with the margin.
    pub tile: GridPlaneTile,
    /// The windowed ZNCC of the tile's middle against the reference's tile,
    /// the view's term of the acceptance score; `None` where it did not score.
    pub score: Option<f64>,
    /// Each cell's reading in this view, `[row * 3 + col]`.
    pub cells: [GridPlaneViewCell; 9],
    /// The view's robust affine fit; `None` when no cell passed its own
    /// gates or the fitted map reflected or was not finite.
    pub fit: Option<GridPlaneViewFit>,
}

/// One cell's registration in one view.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GridPlaneViewCell {
    /// The best whole shift of the search, grid px; `None` where the search
    /// did not run to its end (the template is flat or partly off its
    /// photograph, or a sample the search needs is missing).
    pub whole_shift: Option<[i64; 2]>,
    /// The sub-pixel shift, grid px; `NaN` where no peak was fitted.
    pub shift: [f64; 2],
    /// The ZNCC at the optimum: the sub-pixel peak's, or the best whole
    /// shift's where the cell was refused before a peak was fitted; `NaN`
    /// where the search did not run to its end.
    pub zncc: f64,
    /// The peak's curvature along its flattest direction, ZNCC per grid px²;
    /// `0` where no peak was fitted.
    pub curvature: f64,
    /// The status the cell's own reading gave it, before the view's fit.
    pub reading: CellStatus,
    /// The status after the view's fit: a cell the fit gave no weight is
    /// [`CellStatus::RefusedOutlier`], and every passing cell of a view whose
    /// fit failed is [`CellStatus::NotAttempted`].
    pub status: CellStatus,
    /// The cell's IRLS weight in the view's fit; `0` where it has none.
    pub fit_weight: f64,
}

/// A view's robust affine fit, `c ↦ A·c + b` in grid coordinates centred on
/// the grid.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GridPlaneViewFit {
    /// The model the survivors' count and spread chose.
    pub model: GridPlaneFitModel,
    /// `A`, row-major.
    pub a: [[f64; 2]; 2],
    /// `b`, grid px.
    pub b: [f64; 2],
}

/// The model of a view's robust affine fit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GridPlaneFitModel {
    /// A full affine map, from five or more well-spread cells.
    Affine,
    /// A rotation, scale and shift, from three or more cells.
    Similarity,
    /// A shift alone, from one or two cells or no spread.
    Shift,
}

/// One cell's triangulation and its place in the plane fit.
#[derive(Debug, Clone, PartialEq)]
pub struct GridPlaneCell {
    /// The rays the cell was triangulated from: the reference's through the
    /// cell's centre, then one per view the cell survived in, in row order.
    pub rays: Vec<GridPlaneRay>,
    /// What became of the cell.
    pub status: CellPlaneStatus,
    /// The triangulated position, world frame; `NaN` where it was not
    /// triangulated.
    pub position: [f64; 3],
    /// The ray-intersection residual, pixels, before any floor; `NaN` where
    /// not triangulated.
    pub residual_px: f64,
    /// `σ`, pixels: the residual floored at the root mean square of the rays'
    /// floors and at the kernel's absolute floor; the position's covariance
    /// is scaled by `σ²`. `NaN` where not triangulated.
    pub sigma_px: f64,
    /// The position's variance along the normal the final weights were
    /// computed with, world units squared; `NaN` where no plane was fitted.
    pub normal_variance: f64,
    /// The inverse of that variance; `0` where no plane was fitted.
    pub precision_weight: f64,
    /// The cell's Tukey weight after the last IRLS round; `0` where no plane
    /// was fitted.
    pub tukey_weight: f64,
    /// The cell's weight in the plane fit, precision times Tukey; `0` where it
    /// has none.
    pub weight: f64,
}

/// One cell's sighting in one view.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GridPlaneRay {
    /// The track's observation the ray is from.
    pub observation: usize,
    /// The image of that observation.
    pub image: u32,
    /// The photograph pixel the ray passes through.
    pub pixel: [f64; 2],
    /// The camera centre, world frame.
    pub origin: [f64; 3],
    /// The unit direction, world frame.
    pub direction: [f64; 3],
    /// The ray's pixel-noise floor, pixels.
    pub floor_px: f64,
}

/// The plane through one render's cells.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GridPlanePlane {
    /// The weighted centroid of the cells, world frame; `NaN` when no plane
    /// was fitted.
    pub centroid: [f64; 3],
    /// The least-variance direction of the weighted cells, facing the
    /// cameras, whether or not the verdict fixes it; `NaN` when no plane was
    /// fitted.
    pub fitted_normal: [f64; 3],
    /// The normal the verdict gives: the fitted normal under both axes, the
    /// one perpendicular to the cells' line and closest to the mean viewing
    /// direction under one; `NaN` under none.
    pub normal: [f64; 3],
    /// Which axes of the normal the cells fixed, with the free axis under
    /// one.
    pub determinacy: NormalDeterminacy,
    /// The eigenvalues, ascending, of the live cells' equal-weight scatter
    /// projected onto the fitted plane, world units squared; the verdict
    /// reads their last two.
    pub scatter_eigenvalues: [f64; 3],
    /// `λ_mid / λ_max` of that scatter; `NaN` when no plane was fitted.
    pub anisotropy: f64,
    /// `(Σw)² / Σw²` over the weighted cells; `NaN` when no plane was fitted.
    pub n_eff: f64,
    /// The weighted rms distance of the cells from the plane, world units.
    pub plane_rms: f64,
    /// The predicted precision of the normal, degrees, along the weaker and
    /// the stronger in-plane axis of the weighted scatter; `NaN` under no
    /// axis fixed.
    pub precision_deg: [f64; 2],
}
