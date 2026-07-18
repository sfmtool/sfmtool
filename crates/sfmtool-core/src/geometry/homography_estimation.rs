// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Homography estimation from 2D-2D correspondences: a 4-point DLT minimal
//! solver ([`homography_4pt`]), a normalized N-point refit ([`homography_dlt`]),
//! and a deterministic LO-RANSAC estimator ([`estimate_homography`]).
//!
//! See `specs/core/focal-vote.md` (Homography estimation). Given `N` pixel
//! correspondences `(x₁, x₂)` between two views, estimate the `3×3` homography
//! `H` with `x̃₂ ≃ H x̃₁` (equality up to scale), robust to a contaminated
//! correspondence set. The estimator mirrors
//! [`crate::geometry::epipolar_estimation::estimate_fundamental`]: seeded
//! minimal sampling (four correspondences), symmetric-transfer-error gating,
//! local refit on the consensus set, and adaptive termination. Same inputs and
//! seed give bit-identical output.
//!
//! The direct-linear-transform conditioning (Hartley normalization) and the
//! `9`-vector-to-matrix reshape are shared with the epipolar module.

use nalgebra::{Matrix3, SMatrix, SVector, Vector3};

use crate::geometry::epipolar_estimation::{hartley_normalize, vec_to_mat3};

/// A least-squares design whose largest eigenvalue is below this (relative to
/// the machine-scale zero used elsewhere) carries no constraint — reject.
const DESIGN_ZERO_EPS: f64 = 0.0;

// ── Minimal and refit solvers ────────────────────────────────────────────────

/// Homography from four correspondences (minimal DLT).
///
/// Returns the `3×3` `H` (unit Frobenius norm) with `x̃₂ ≃ H x̃₁`, or `None` for
/// a degenerate sample (three collinear points, coincident points, non-finite
/// input). Pure and deterministic.
pub fn homography_4pt(x1: &[[f64; 2]; 4], x2: &[[f64; 2]; 4]) -> Option<Matrix3<f64>> {
    homography_dlt(x1, x2)
}

/// Least-squares homography from `N ≥ 4` correspondences (normalized DLT).
///
/// Each correspondence contributes two rows of the `2N×9` design matrix
/// `A·vec(H) = 0`; `H` is the smallest right singular vector, reshaped
/// row-major and denormalized to `H = T₂⁻¹ Ĥ T₁` at unit Frobenius norm.
/// `None` for `N < 4`, non-finite input, a coincident point set, or a
/// rank-deficient design.
pub fn homography_dlt(x1: &[[f64; 2]], x2: &[[f64; 2]]) -> Option<Matrix3<f64>> {
    let n = x1.len();
    if n < 4 || x2.len() != n {
        return None;
    }
    for p in x1.iter().chain(x2.iter()) {
        if !p[0].is_finite() || !p[1].is_finite() {
            return None;
        }
    }
    let (n1, t1) = hartley_normalize(x1)?;
    let (n2, t2) = hartley_normalize(x2)?;

    // Accumulate AᵀA over the two DLT rows per correspondence:
    //   [ -x, -y, -1,  0,  0,  0,  x'x, x'y, x' ]
    //   [  0,  0,  0, -x, -y, -1,  y'x, y'y, y' ]
    let mut ata = SMatrix::<f64, 9, 9>::zeros();
    for (a, b) in n1.iter().zip(n2.iter()) {
        let (x, y) = (a[0], a[1]);
        let (xp, yp) = (b[0], b[1]);
        let r1 = SVector::<f64, 9>::from_column_slice(&[
            -x,
            -y,
            -1.0,
            0.0,
            0.0,
            0.0,
            xp * x,
            xp * y,
            xp,
        ]);
        let r2 = SVector::<f64, 9>::from_column_slice(&[
            0.0,
            0.0,
            0.0,
            -x,
            -y,
            -1.0,
            yp * x,
            yp * y,
            yp,
        ]);
        ata += r1 * r1.transpose();
        ata += r2 * r2.transpose();
    }

    let eig = ata.symmetric_eigen();
    let mut best = 0usize;
    for j in 1..9 {
        if eig.eigenvalues[j] < eig.eigenvalues[best] {
            best = j;
        }
    }
    if eig.eigenvalues.iter().cloned().fold(0.0, f64::max) <= DESIGN_ZERO_EPS {
        return None;
    }
    let h_hat = vec_to_mat3(&eig.eigenvectors.column(best).into_owned());
    let t2_inv = t2.try_inverse()?;
    let h = t2_inv * h_hat * t1;
    let norm = h.norm();
    if norm < 1e-300 || !h.iter().all(|v| v.is_finite()) {
        return None;
    }
    Some(h / norm)
}

/// Symmetric transfer error (pixels²) of a correspondence against `H` and its
/// inverse: `‖x₂ − H x₁‖² + ‖x₁ − H⁻¹ x₂‖²`. Returns `+∞` when either transfer
/// sends a point to the plane at infinity.
fn symmetric_transfer_sq(
    h: &Matrix3<f64>,
    h_inv: &Matrix3<f64>,
    x1: [f64; 2],
    x2: [f64; 2],
) -> f64 {
    let p1 = Vector3::new(x1[0], x1[1], 1.0);
    let p2 = Vector3::new(x2[0], x2[1], 1.0);
    let hp1 = h * p1;
    let hip2 = h_inv * p2;
    if hp1[2].abs() < 1e-12 || hip2[2].abs() < 1e-12 {
        return f64::INFINITY;
    }
    let fx = hp1[0] / hp1[2];
    let fy = hp1[1] / hp1[2];
    let bx = hip2[0] / hip2[2];
    let by = hip2[1] / hip2[2];
    let d_fwd = (fx - x2[0]).powi(2) + (fy - x2[1]).powi(2);
    let d_bwd = (bx - x1[0]).powi(2) + (by - x1[1]).powi(2);
    d_fwd + d_bwd
}

// ── Robust estimator ─────────────────────────────────────────────────────────

/// Tuning for [`estimate_homography`].
#[derive(Clone, Debug)]
pub struct HomographyOptions {
    /// Inlier bound on the symmetric transfer error, pixels.
    pub max_error_px: f64,
    /// Adaptive-termination target: stop once the probability that an
    /// all-inlier sample was drawn exceeds this (given the best inlier count).
    pub confidence: f64,
    /// Hard trial cap.
    pub max_iterations: u32,
    /// Reject an estimate supported by fewer inliers than this.
    pub min_inliers: usize,
    /// SplitMix64 seed for the sampler.
    pub seed: u64,
    /// Refit each new best consensus on its inliers (normalized DLT), repeating
    /// while the inlier set grows.
    pub local_optimization: bool,
}

impl Default for HomographyOptions {
    fn default() -> Self {
        Self {
            max_error_px: 3.0,
            confidence: 0.999,
            max_iterations: 10_000,
            min_inliers: 4,
            seed: 0,
            local_optimization: true,
        }
    }
}

/// Result of [`estimate_homography`].
#[derive(Clone, Debug)]
pub struct HomographyEstimate {
    /// Unit Frobenius norm.
    pub h_matrix: Matrix3<f64>,
    /// Per-input-correspondence inlier mask.
    pub inliers: Vec<bool>,
    /// Trials actually run.
    pub iterations: u32,
}

/// SplitMix64 — the deterministic index sampler (no `rand` dependency,
/// identical across platforms). Mirrors the generator in
/// [`crate::geometry::epipolar_estimation`].
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

/// Draw four distinct indices in `0..n` with the seeded sampler.
fn draw4(state: &mut u64, n: usize) -> [usize; 4] {
    let mut idx = [0usize; 4];
    let mut k = 0;
    while k < 4 {
        let cand = (splitmix64(state) % n as u64) as usize;
        if !idx[..k].contains(&cand) {
            idx[k] = cand;
            k += 1;
        }
    }
    idx
}

/// Count and mark inliers for `H`, scoring in input order (deterministic).
/// A non-invertible `H` scores zero inliers.
fn score_h(
    h: &Matrix3<f64>,
    x1: &[[f64; 2]],
    x2: &[[f64; 2]],
    thresh2: f64,
    mask: &mut [bool],
) -> usize {
    let Some(h_inv) = h.try_inverse() else {
        mask.iter_mut().for_each(|m| *m = false);
        return 0;
    };
    let mut count = 0;
    for i in 0..x1.len() {
        let inlier = symmetric_transfer_sq(h, &h_inv, x1[i], x2[i]) <= thresh2;
        mask[i] = inlier;
        count += inlier as usize;
    }
    count
}

/// Refit `H` on its inliers with the DLT solver, rescore, and repeat while the
/// inlier count strictly grows (bounded rounds).
fn local_optimize_h(
    mut h: Matrix3<f64>,
    x1: &[[f64; 2]],
    x2: &[[f64; 2]],
    thresh2: f64,
    mut count: usize,
    scratch: &mut [bool],
) -> (Matrix3<f64>, usize) {
    const MAX_ROUNDS: usize = 10;
    let mut cur_mask: Vec<bool> = scratch.to_vec();
    for _ in 0..MAX_ROUNDS {
        let idx: Vec<usize> = (0..x1.len()).filter(|&i| cur_mask[i]).collect();
        if idx.len() < 4 {
            break;
        }
        let sub1: Vec<[f64; 2]> = idx.iter().map(|&i| x1[i]).collect();
        let sub2: Vec<[f64; 2]> = idx.iter().map(|&i| x2[i]).collect();
        let Some(h_ref) = homography_dlt(&sub1, &sub2) else {
            break;
        };
        let new_count = score_h(&h_ref, x1, x2, thresh2, scratch);
        if new_count > count {
            h = h_ref;
            count = new_count;
            cur_mask.copy_from_slice(scratch);
        } else {
            scratch.copy_from_slice(&cur_mask);
            break;
        }
    }
    (h, count)
}

/// Robustly estimate the homography from pixel correspondences (4-point
/// LO-RANSAC). Returns `None` when no consensus reaches `min_inliers`.
/// See `specs/core/focal-vote.md`.
pub fn estimate_homography(
    x1: &[[f64; 2]],
    x2: &[[f64; 2]],
    options: &HomographyOptions,
) -> Option<HomographyEstimate> {
    let n = x1.len();
    if n != x2.len() || n < 4 {
        return None;
    }
    let thresh2 = options.max_error_px * options.max_error_px;

    let mut state = options.seed;
    let mut best_count = 0usize;
    let mut best: Option<Matrix3<f64>> = None;
    let mut best_mask = vec![false; n];
    let mut scratch = vec![false; n];

    let mut iterations = 0u32;
    let mut required = options.max_iterations as u64;
    while (iterations as u64) < required && iterations < options.max_iterations {
        iterations += 1;

        let idx = draw4(&mut state, n);
        let s1: [[f64; 2]; 4] = core::array::from_fn(|k| x1[idx[k]]);
        let s2: [[f64; 2]; 4] = core::array::from_fn(|k| x2[idx[k]]);

        let Some(h) = homography_4pt(&s1, &s2) else {
            continue;
        };
        let mut count = score_h(&h, x1, x2, thresh2, &mut scratch);
        if count > best_count {
            let mut h_best = h;
            if options.local_optimization {
                (h_best, count) = local_optimize_h(h_best, x1, x2, thresh2, count, &mut scratch);
            }
            best_count = count;
            best = Some(h_best);
            best_mask.copy_from_slice(&scratch);

            // Adaptive termination bound from the current best inlier rate.
            let w = best_count as f64 / n as f64;
            let w4 = w.powi(4);
            required = if w4 >= 1.0 {
                iterations as u64
            } else if w4 <= 0.0 {
                options.max_iterations as u64
            } else {
                let num = (1.0 - options.confidence).ln();
                let den = (1.0 - w4).ln();
                (num / den).ceil().max(0.0) as u64
            };
        }
    }

    let h = best?;
    // Refresh the mask so it is consistent with the returned `H`.
    best_count = score_h(&h, x1, x2, thresh2, &mut best_mask);
    if best_count < options.min_inliers {
        return None;
    }
    Some(HomographyEstimate {
        h_matrix: h,
        inliers: best_mask,
        iterations,
    })
}

#[cfg(test)]
mod tests;
