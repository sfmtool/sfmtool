// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Patch window kernel, the frozen `R×R` scoring [`Support`], and the
//! patch-placement helpers ([`repose_patch`], [`view_render_patch`]) shared by
//! the refinement and the per-point patch operations.

use nalgebra::Vector3;

use crate::patch::cloud::OrientedPatch;
use crate::patch::keypoint_localize;

use super::params::{PatchWindow, ProjectedImage, FLAT_NORM_SQ_EPS};

/// Per-pixel window weight over the `R×R` patch grid, in row-major order.
pub(in crate::patch) fn window_weights(window: PatchWindow, resolution: u32) -> Vec<f64> {
    let r = resolution as usize;
    let step = 2.0 / r as f64;
    let mut w = Vec::with_capacity(r * r);
    for row in 0..r {
        let t = (row as f64 + 0.5) * step - 1.0;
        for col in 0..r {
            let s = (col as f64 + 0.5) * step - 1.0;
            let r2 = s * s + t * t;
            let weight = match window {
                PatchWindow::Uniform => 1.0,
                PatchWindow::Gaussian { sigma } => (-r2 / (2.0 * sigma * sigma)).exp(),
                PatchWindow::GaussianDisk { sigma } => {
                    if r2 > 1.0 {
                        0.0
                    } else {
                        (-r2 / (2.0 * sigma * sigma)).exp()
                    }
                }
            };
            w.push(weight);
        }
    }
    w
}

/// The frozen window support over the `R×R` core: the linear `row * R + col`
/// indices of the positive-weight pixels, their window weights, `√weight` per
/// pixel (folded into the z-normalized space so a plain dot product realizes the
/// windowed inner product), and the total weight (the windowed mean's
/// denominator). Shared by the per-point patch operations (keypoint localize,
/// keypoint subpixel refine) that score on a fixed `R×R` core.
pub(in crate::patch) struct Support {
    pub pixels: Vec<usize>,
    pub weights: Vec<f64>,
    pub sqrt_weights: Vec<f32>,
    pub total_weight: f64,
}

/// Build the frozen [`Support`] for the `R×R` core from the window kernel — keep
/// only pixels whose window weight is positive, in row-major order.
pub(in crate::patch) fn build_support(window: PatchWindow, resolution: u32) -> Support {
    let w_full = window_weights(window, resolution);
    let mut pixels = Vec::new();
    let mut weights = Vec::new();
    for (p, &w) in w_full.iter().enumerate() {
        if w > 0.0 {
            pixels.push(p);
            weights.push(w);
        }
    }
    let total_weight: f64 = weights.iter().sum();
    let sqrt_weights: Vec<f32> = weights.iter().map(|&w| w.sqrt() as f32).collect();
    Support {
        pixels,
        weights,
        sqrt_weights,
        total_weight,
    }
}

/// The middle of an `R×R` patch grid: the rows, and the columns, `R/4 .. R -
/// R/4`. That is the centred square half the grid's width, the middle `12×12`
/// of a `24×24` grid. It stays symmetric about the grid's centre, so an odd `R`
/// gets the odd side `R - 2·⌊R/4⌋`.
pub(in crate::patch) fn middle_span(resolution: u32) -> std::ops::Range<usize> {
    let r = resolution as usize;
    r / 4..r - r / 4
}

impl Support {
    /// The positions in `pixels` that fall inside the grid's [`middle_span`]
    /// square, in order: the subset a middle ZNCC is read over.
    pub(in crate::patch) fn middle(&self, resolution: u32) -> Vec<usize> {
        let r = resolution as usize;
        let span = middle_span(resolution);
        self.pixels
            .iter()
            .enumerate()
            .filter(|&(_, &p)| span.contains(&(p / r)) && span.contains(&(p % r)))
            .map(|(k, _)| k)
            .collect()
    }

    /// The positions in `pixels` that fall in each cell of the grid's
    /// [`grid_bounds`] three-by-three split, row-major from the top-left cell.
    /// Over a support that does not cover the square, a corner cell holds only
    /// the part of its square the support covers; [`square_cells`] is the
    /// split of the whole square.
    pub(in crate::patch) fn grid_cells(&self, resolution: u32) -> [[Vec<usize>; 3]; 3] {
        let r = resolution as usize;
        let bounds = grid_bounds(resolution);
        let third = |i: usize| (0..3).find(|&t| i < bounds[t + 1]).unwrap_or(2);
        let mut cells: [[Vec<usize>; 3]; 3] = Default::default();
        for (k, &p) in self.pixels.iter().enumerate() {
            cells[third(p / r)][third(p % r)].push(k);
        }
        cells
    }
}

/// Where an `R×R` patch grid's rows, and its columns, split into the thirds a
/// ZNCC grid is read over: `[0, R/3, R - R/3, R]`. Symmetric about the grid's
/// centre, so when `R` is not a multiple of three the middle third takes the
/// remainder, and a `24×24` grid splits into `8×8` cells.
pub(in crate::patch) fn grid_bounds(resolution: u32) -> [usize; 4] {
    let r = resolution as usize;
    [0, r / 3, r - r / 3, r]
}

/// The cells of the [`grid_bounds`] split over the **whole** `R×R` square, as
/// grid positions `row · R + col`, `cells[row][col]` from the top-left cell.
/// Every cell is a full rectangle: the window's disk plays no part.
pub(in crate::patch) fn square_cells(resolution: u32) -> [[Vec<usize>; 3]; 3] {
    build_support(PatchWindow::Uniform, resolution).grid_cells(resolution)
}

/// The **ZNCC grid** of two sample sets: one ZNCC per cell of `cells`,
/// `grid[row][col]` from the top-left cell.
///
/// Each cell is read the way [`windowed_zncc_at`] reads a subset, with the
/// same channel and flat conventions, but with every pixel weighted equally.
/// `NaN` in a cell where no channel of `reference` carries texture, and in a
/// cell holding a non-finite value in either set, which is how a caller marks
/// a pixel it could not sample.
pub(in crate::patch) fn zncc_grid(
    sample: &[f32],
    reference: &[f32],
    channels: usize,
    n: usize,
    cells: &[[Vec<usize>; 3]; 3],
) -> [[f64; 3]; 3] {
    let sampled = |at: &[usize]| {
        (0..channels).all(|c| {
            at.iter()
                .all(|&k| sample[c * n + k].is_finite() && reference[c * n + k].is_finite())
        })
    };
    cells.each_ref().map(|row| {
        row.each_ref().map(|at| {
            if sampled(at) {
                zncc_over(sample, reference, channels, n, |_| 1.0, at)
            } else {
                f64::NAN
            }
        })
    })
}

/// The parts of a patch that the readings beside a whole-patch ZNCC are taken
/// over: the middle square, as positions of a [`Support`], and the nine cells
/// of the ZNCC grid, as positions of the whole `R×R` square.
pub(in crate::patch) struct Parts {
    /// [`Support::middle`].
    pub middle: Vec<usize>,
    /// [`square_cells`].
    pub cells: [[Vec<usize>; 3]; 3],
    /// `R²`, the length of one channel of a sample set over the square.
    pub square: usize,
}

/// The readings of one sample set against a reference over the [`Parts`] of
/// the grid: the middle ZNCC and the ZNCC grid.
#[derive(Debug, Clone, Copy)]
pub(in crate::patch) struct PartZncc {
    /// The windowed ZNCC over the middle square ([`windowed_zncc_at`]).
    pub middle: f64,
    /// The unweighted ZNCC of each cell of the whole square ([`zncc_grid`]).
    pub grid: [[f64; 3]; 3],
}

impl PartZncc {
    /// No reading: every value `NaN`.
    pub const NAN: PartZncc = PartZncc {
        middle: f64::NAN,
        grid: [[f64::NAN; 3]; 3],
    };
}

impl Parts {
    pub(in crate::patch) fn new(support: &Support, resolution: u32) -> Parts {
        Parts {
            middle: support.middle(resolution),
            cells: square_cells(resolution),
            square: (resolution as usize).pow(2),
        }
    }

    /// Read a sample set against a reference over each part. `on_support` is
    /// the pair over the support's pixels, planar `[c · n + k]` with `n` the
    /// length of `weights`, the support's window weights; `on_square` is the
    /// pair over the whole square, planar `[c · R² + p]`, with a non-finite
    /// value wherever a pixel could not be sampled.
    pub(in crate::patch) fn read(
        &self,
        on_support: (&[f32], &[f32]),
        on_square: (&[f32], &[f32]),
        channels: usize,
        weights: &[f64],
    ) -> PartZncc {
        let n = weights.len();
        PartZncc {
            middle: windowed_zncc_at(
                on_support.0,
                on_support.1,
                channels,
                n,
                weights,
                &self.middle,
            ),
            grid: zncc_grid(on_square.0, on_square.1, channels, self.square, &self.cells),
        }
    }
}

/// The channel-averaged windowed ZNCC of two sample sets over a subset `at` of
/// their support positions.
///
/// `sample` and `reference` are planar `[c · n + k]` over the same `n` support
/// pixels and `channels` channels, and `weights` are the support's window
/// weights. Each channel is mean-removed and normalized over `at` alone, so the
/// answer is the ZNCC the two would give had the patch been only that subset.
/// A per-channel affine rescaling of either input leaves it unchanged, so a
/// caller may pass a z-normalized template divided back by `√w`.
///
/// The conventions are the whole-patch ones. A channel flat in `sample`
/// (windowed norm² below [`FLAT_NORM_SQ_EPS`]) contributes `0`, and a channel
/// flat in `reference` is left out of the average. `NaN` when no channel of
/// `reference` carries texture over `at`, or `at` is empty.
pub(in crate::patch) fn windowed_zncc_at(
    sample: &[f32],
    reference: &[f32],
    channels: usize,
    n: usize,
    weights: &[f64],
    at: &[usize],
) -> f64 {
    zncc_over(sample, reference, channels, n, |k| weights[k], at)
}

/// [`windowed_zncc_at`] with the weight of support position `k` given by
/// `weight(k)`.
fn zncc_over(
    sample: &[f32],
    reference: &[f32],
    channels: usize,
    n: usize,
    weight: impl Fn(usize) -> f64,
    at: &[usize],
) -> f64 {
    let total: f64 = at.iter().map(|&k| weight(k)).sum();
    if at.is_empty() || total <= 0.0 {
        return f64::NAN;
    }
    let moments = |col: &[f32]| {
        let mean = at
            .iter()
            .map(|&k| weight(k) * f64::from(col[k]))
            .sum::<f64>()
            / total;
        let norm_sq = at
            .iter()
            .map(|&k| weight(k) * (f64::from(col[k]) - mean).powi(2))
            .sum::<f64>();
        (mean, norm_sq)
    };
    let (mut sum, mut scored) = (0.0, 0usize);
    for c in 0..channels {
        let a = &sample[c * n..][..n];
        let b = &reference[c * n..][..n];
        let (mean_b, norm_b) = moments(b);
        if norm_b < FLAT_NORM_SQ_EPS {
            continue;
        }
        scored += 1;
        let (mean_a, norm_a) = moments(a);
        if norm_a < FLAT_NORM_SQ_EPS {
            continue;
        }
        let cross: f64 = at
            .iter()
            .map(|&k| weight(k) * (f64::from(a[k]) - mean_a) * (f64::from(b[k]) - mean_b))
            .sum();
        sum += cross / (norm_a * norm_b).sqrt();
    }
    if scored == 0 {
        f64::NAN
    } else {
        sum / scored as f64
    }
}

/// Rebuild the patch on a new plane: same `center` / `half_extent`, the input
/// `v_axis` ("up") reprojected onto the plane of `n` (`u = v × n`), which
/// preserves the in-plane orientation across the normal change.
pub(super) fn repose_patch(base: &OrientedPatch, n: &Vector3<f64>) -> OrientedPatch {
    let mut p = OrientedPatch::from_center_normal(base.center, *n, base.v_axis, base.half_extent);
    p.w = base.w;
    p
}

/// The patch to render into `view`: when `keypoint` is given, recenter `patch`
/// in-plane so it projects at that keypoint (falling back to `patch` if the ray
/// is degenerate); otherwise `patch` unchanged.
///
/// Borrows `patch` in the no-keypoint hot path (and on a degenerate ray), so a
/// refinement without keypoints allocates nothing extra. With a keypoint, the
/// `wpp` factors cancel in the `seed_offset → shifted_center` round trip, so they
/// are passed as `1.0`.
///
// TODO(perf): the keypoint's world ray in `seed_offset` is invariant across
// candidate normals (only the ray∩plane intersection depends on the plane), but
// it is recomputed every candidate/level here. If keypoint-anchored refine ever
// becomes hot, precompute the per-view world ray once per refine call.
pub(in crate::patch) fn view_render_patch<'a>(
    patch: &'a OrientedPatch,
    view: &ProjectedImage<'_>,
    keypoint: Option<[f64; 2]>,
) -> std::borrow::Cow<'a, OrientedPatch> {
    use std::borrow::Cow;
    let Some(kp) = keypoint else {
        return Cow::Borrowed(patch);
    };
    let Some([au, av]) = keypoint_localize::seed_offset(patch, view, kp, 1.0, 1.0) else {
        return Cow::Borrowed(patch);
    };
    let center = keypoint_localize::shifted_center(patch, au, av, 1.0, 1.0);
    let mut shifted =
        OrientedPatch::from_center_normal(center, patch.normal(), patch.v_axis, patch.half_extent);
    shifted.w = patch.w;
    Cow::Owned(shifted)
}
