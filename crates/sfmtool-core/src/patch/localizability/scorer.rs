// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The pure structure-tensor scorer: luminance gradients over the `R×R` window,
//! the window-weighted second-moment matrix, and its 2×2 symmetric eigensystem.
//!
//! See `specs/core/patch-localizability.md`. Given a point's cross-view consensus
//! patch, [`patch_localizability`] measures the curvature of the ZNCC
//! self-similarity surface (the Harris/Shi–Tomasi structure tensor) and reports
//! the noise-normalized weak-axis positional uncertainty `σ_pos` in **grid** px.

use rayon::prelude::*;

use crate::patch::normal_refine::{build_support, PatchWindow, Support};

/// Per-point localizability: the structure-tensor eigenvalues (summed form, the
/// contrast-carrying raw Shi–Tomasi scale — `λ_sum = W·λ`), the weak-axis
/// direction, and the noise-normalized weak-axis positional uncertainty in
/// **patch-grid** px. `lam2` is the smaller eigenvalue (the Shi–Tomasi "does this
/// pin a 2D position" number); `theta` is the angle (radians, in the grid frame
/// where `x`=column-right and `y`=row-down) of the `lam2` eigenvector — the slide
/// direction, the axis of greatest positional ambiguity. Every field is `NaN`
/// when the patch is empty (all pixels zero — a culled / uncovered consensus).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Localizability {
    /// Larger eigenvalue of `M_sum = Σ_k w_k ∇I∇Iᵀ` (steep axis).
    pub lam1: f64,
    /// Smaller eigenvalue of `M_sum` (the Shi–Tomasi weak-axis score).
    pub lam2: f64,
    /// Angle (radians) of the `lam2` eigenvector in the grid frame — the slide
    /// direction.
    pub theta: f64,
    /// `σ_noise / √λ₂_sum` — weak-axis positional uncertainty in patch-grid px.
    pub sigma_pos_grid: f64,
}

impl Localizability {
    /// The unscorable sentinel (empty / uncovered consensus): every field `NaN`.
    fn nan() -> Self {
        Self {
            lam1: f64::NAN,
            lam2: f64::NAN,
            theta: f64::NAN,
            sigma_pos_grid: f64::NAN,
        }
    }
}

/// Rec.601 luminance weights, matching the prototype and the pipeline's
/// per-channel ZNCC (luminance is the v1 simplification; see the spec).
const LUMA: [f64; 3] = [0.299, 0.587, 0.114];

/// Window-weighted variance floor on `λ₂`, so a perfectly flat (but non-empty)
/// patch yields a finite (very large) `σ_pos` rather than a division by zero —
/// mirrors the prototype's `np.maximum(lam2_sum, 1e-12)`.
const LAM2_FLOOR: f64 = 1e-12;

/// The `R×R` luminance grid (row-major) of a flat `R×R×C` consensus patch, plus
/// whether the patch carries any signal at all. `valid` is `true` iff **any** of
/// the `R·R·C` raw values is non-zero (matching the prototype's
/// `bitmaps.reshape(P, -1).any(axis=1)`), so a hard-zero-filled culled/uncovered
/// consensus reads as empty regardless of the luminance projection. Channels
/// beyond the first three (the alpha of an RGBA consensus) contribute to the
/// emptiness test but not the luminance — the scorer ignores alpha per the spec.
fn luminance_grid(patch: &[f32], resolution: usize, channels: usize) -> (Vec<f64>, bool) {
    let n = resolution * resolution;
    let mut gray = vec![0.0f64; n];
    let mut any_nonzero = false;
    let luma_channels = channels.min(3);
    for (px, g) in gray.iter_mut().enumerate() {
        let base = px * channels;
        let mut acc = 0.0;
        for (c, value) in patch[base..base + channels].iter().enumerate() {
            let v = *value as f64;
            if v != 0.0 {
                any_nonzero = true;
            }
            if c < luma_channels {
                acc += LUMA[c] * v;
            }
        }
        *g = acc;
    }
    (gray, any_nonzero)
}

/// Discrete gradient along both axes of an `R×R` row-major grid (unit spacing):
/// central differences in the interior, one-sided differences at the borders.
/// Returns `(grad_row, grad_col)` (i.e. `∂/∂y`, `∂/∂x`), parallel to `grid`.
fn central_diff_gradient(grid: &[f64], r: usize) -> (Vec<f64>, Vec<f64>) {
    let mut grad_row = vec![0.0f64; r * r];
    let mut grad_col = vec![0.0f64; r * r];
    let at = |row: usize, col: usize| grid[row * r + col];
    for row in 0..r {
        for col in 0..r {
            // ∂/∂row (down the columns).
            grad_row[row * r + col] = if r == 1 {
                0.0
            } else if row == 0 {
                at(1, col) - at(0, col)
            } else if row == r - 1 {
                at(r - 1, col) - at(r - 2, col)
            } else {
                0.5 * (at(row + 1, col) - at(row - 1, col))
            };
            // ∂/∂col (along the rows).
            grad_col[row * r + col] = if r == 1 {
                0.0
            } else if col == 0 {
                at(row, 1) - at(row, 0)
            } else if col == r - 1 {
                at(row, r - 1) - at(row, r - 2)
            } else {
                0.5 * (at(row, col + 1) - at(row, col - 1))
            };
        }
    }
    (grad_row, grad_col)
}

/// Eigen-decompose the symmetric `[[a, b], [b, c]]` second-moment matrix into
/// `(lam1 ≥ lam2, theta)`, where `theta` is the angle (radians) of the `lam2`
/// eigenvector — the weak / slide axis.
fn eig_2x2_sym(a: f64, b: f64, c: f64) -> (f64, f64, f64) {
    let tr = a + c;
    let disc = (tr * tr - 4.0 * (a * c - b * b)).max(0.0).sqrt();
    let lam1 = 0.5 * (tr + disc);
    let lam2 = 0.5 * (tr - disc);
    // Weak-axis eigenvector. Off-diagonal present: (b, λ₂ − a) solves
    // (M − λ₂ I) v = 0. Diagonal matrix: the weak axis is the smaller-diagonal
    // coordinate axis (x when a ≤ c, else y).
    let theta = if b.abs() > 1e-12 {
        (lam2 - a).atan2(b)
    } else if a <= c {
        0.0
    } else {
        std::f64::consts::FRAC_PI_2
    };
    (lam1, lam2, theta)
}

/// Score one `R×R×C` consensus patch (row-major, `C` channels; RGBA alpha, if
/// present, is used only for the emptiness test). `window` is the frozen scoring
/// support over the same `R×R` grid; `sigma_noise` is the global consensus noise
/// constant (intensity units) that sets the absolute px scale of `sigma_pos_grid`.
///
/// The structure tensor `M_sum = Σ_k w_k ∇I∇Iᵀ` is accumulated over the window's
/// positive-weight pixels (gradients themselves are the full-grid central
/// differences, so a support pixel's stencil may reach just outside the disk —
/// exactly the prototype). Returns the [`Localizability`] (all-`NaN` for an empty patch).
pub(in crate::patch) fn patch_localizability(
    patch: &[f32],
    resolution: usize,
    channels: usize,
    window: &Support,
    sigma_noise: f64,
) -> Localizability {
    debug_assert_eq!(patch.len(), resolution * resolution * channels);
    let (gray, valid) = luminance_grid(patch, resolution, channels);
    if !valid {
        return Localizability::nan();
    }
    let (grad_row, grad_col) = central_diff_gradient(&gray, resolution);
    let (mut sxx, mut syy, mut sxy) = (0.0f64, 0.0f64, 0.0f64);
    for (&p, &w) in window.pixels.iter().zip(&window.weights) {
        let gx = grad_col[p];
        let gy = grad_row[p];
        sxx += w * gx * gx;
        syy += w * gy * gy;
        sxy += w * gx * gy;
    }
    let (lam1, lam2, theta) = eig_2x2_sym(sxx, sxy, syy);
    let sigma_pos_grid = sigma_noise / lam2.max(LAM2_FLOOR).sqrt();
    Localizability {
        lam1,
        lam2,
        theta,
        sigma_pos_grid,
    }
}

/// Batch-score a `(P, R, R, C)` consensus stack (flat row-major, `P` patches of
/// `R·R·C` values each), rayon-parallel across patches. Builds the frozen scoring
/// [`Support`] once for `window` at resolution `R` and reuses it for every patch.
///
/// Returns one [`Localizability`] per patch, in input order (all-`NaN` for an
/// empty patch). The grid→source-px mapping and `σ_pos` in source px are the
/// caller's (they need the recon geometry, not the consensus); this entry stops
/// at `sigma_pos_grid`.
///
/// # Panics
///
/// Panics if `patches.len() != num_patches * resolution * resolution * channels`.
pub fn score_localizability_stack(
    patches: &[f32],
    num_patches: usize,
    resolution: usize,
    channels: usize,
    window: PatchWindow,
    sigma_noise: f64,
) -> Vec<Localizability> {
    let stride = resolution * resolution * channels;
    assert_eq!(
        patches.len(),
        num_patches * stride,
        "patch stack must be P*R*R*C values"
    );
    if stride == 0 {
        return vec![Localizability::nan(); num_patches];
    }
    let support = build_support(window, resolution as u32);
    patches
        .par_chunks(stride)
        .map(|p| patch_localizability(p, resolution, channels, &support, sigma_noise))
        .collect()
}
