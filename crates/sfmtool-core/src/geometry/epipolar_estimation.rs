// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Epipolar geometry from 2D-2D correspondences: a 7-point minimal solver
//! ([`fundamental_7pt`]), a normalized 8-point refit ([`fundamental_8pt`]), a
//! deterministic LO-RANSAC estimator ([`estimate_fundamental`]), and the
//! Bougnoux focal-length recovery ([`focal_from_fundamental`]).
//!
//! See `specs/core/epipolar-estimation.md`. Given `N` pixel correspondences
//! `(x₁, x₂)` between two views, estimate the rank-2 fundamental matrix `F`
//! with `x̃₂ᵀ F x̃₁ = 0`, robust to a contaminated correspondence set, and
//! extract a focal-length estimate from it.
//!
//! This module is *estimation from data*. The complementary direction — the
//! fundamental matrix of two **known** cameras — lives in
//! [`crate::camera::epipolar`] (`compute_fundamental_matrix`), which the tests
//! use to generate ground truth.
//!
//! The minimal solver returns one to three candidates from seven
//! correspondences: the design matrix has a two-dimensional null space
//! `span(F_a, F_b)`, and the rank-2 constraint `det(α F_a + (1−α) F_b) = 0` is
//! a cubic in `α` solved in closed form. It is pure and bit-stable. The
//! estimator draws minimal samples with a SplitMix64 sampler (seeded, so same
//! inputs + same seed give bit-identical output), scores every candidate by the
//! Sampson inlier test in input order, keeps the best consensus, optionally
//! refits it with the 8-point solver on its inliers, and terminates adaptively.

use nalgebra::{Matrix3, SMatrix, SVector, Vector3};

use crate::geometry::polynomial::{polish_cubic_root, solve_cubic};
use crate::geometry::rotation::skew_symmetric;

/// Null space of the 7-point design is required to be exactly 2-D: the
/// third-smallest eigenvalue of `AᵀA` must exceed this fraction of the largest,
/// else the constraints are rank-deficient (repeated points, degenerate
/// configuration) and no candidate is returned.
const NULLSPACE_RANK_EPS: f64 = 1e-9;

/// Below this Frobenius norm the fundamental matrix is treated as numerically
/// zero — it carries no epipolar geometry (rotation-only / zero-baseline
/// motion), so no focal is recoverable. Physically-scaled `F` (from real
/// two-view geometry) and the estimator's unit-norm `F` sit far above this.
const ZERO_F_EPS: f64 = 1e-12;

/// The Bougnoux denominator is an inherent near-cancellation even for a healthy
/// pair (order `1e-8` with `F` at unit Frobenius norm and pixel-scale principal
/// points), so this vanishing threshold sits well below that; the classical
/// forward-motion and fixating degeneracies collapse it far past this floor
/// (and are also caught by the `f₁² ≤ 0` sign test).
const BOUGNOUX_DEN_EPS: f64 = 1e-12;

// ── Hartley normalization ────────────────────────────────────────────────────

/// Translate points to zero mean and scale so the mean distance from the origin
/// is `√2`. Returns the normalized points and the `3×3` transform `T` with
/// `x̂ = T·x̃`. `None` when the points are coincident (zero spread).
fn hartley_normalize(pts: &[[f64; 2]]) -> Option<(Vec<[f64; 2]>, Matrix3<f64>)> {
    let n = pts.len();
    if n == 0 {
        return None;
    }
    let (mut mx, mut my) = (0.0, 0.0);
    for p in pts {
        mx += p[0];
        my += p[1];
    }
    mx /= n as f64;
    my /= n as f64;
    let mut mean_dist = 0.0;
    for p in pts {
        let dx = p[0] - mx;
        let dy = p[1] - my;
        mean_dist += (dx * dx + dy * dy).sqrt();
    }
    mean_dist /= n as f64;
    if mean_dist < 1e-12 {
        return None;
    }
    let s = std::f64::consts::SQRT_2 / mean_dist;
    let t = Matrix3::new(s, 0.0, -s * mx, 0.0, s, -s * my, 0.0, 0.0, 1.0);
    let out: Vec<[f64; 2]> = pts
        .iter()
        .map(|p| [s * (p[0] - mx), s * (p[1] - my)])
        .collect();
    Some((out, t))
}

/// Reshape a 9-vector (row-major `F`) into a `3×3` matrix.
fn vec_to_mat3(v: &SVector<f64, 9>) -> Matrix3<f64> {
    Matrix3::new(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8])
}

/// Determinant of the matrix whose columns are `a, b, c`.
fn det_cols(a: Vector3<f64>, b: Vector3<f64>, c: Vector3<f64>) -> f64 {
    Matrix3::from_columns(&[a, b, c]).determinant()
}

/// Replace `F` by the closest rank-2 matrix in Frobenius norm (zero the smallest
/// singular value). `None` when the SVD does not yield `U`/`Vᵀ`.
fn enforce_rank2(f: &Matrix3<f64>) -> Option<Matrix3<f64>> {
    let svd = f.svd(true, true);
    let u = svd.u?;
    let v_t = svd.v_t?;
    let mut s = svd.singular_values;
    s[2] = 0.0;
    Some(u * Matrix3::from_diagonal(&s) * v_t)
}

/// Accumulate the `9×9` normal matrix `AᵀA` over the epipolar constraint rows
/// `(u₂u₁, u₂v₁, u₂, v₂u₁, v₂v₁, v₂, u₁, v₁, 1)` in normalized coordinates.
fn design_normal_matrix(n1: &[[f64; 2]], n2: &[[f64; 2]]) -> SMatrix<f64, 9, 9> {
    let mut ata = SMatrix::<f64, 9, 9>::zeros();
    for (a, b) in n1.iter().zip(n2.iter()) {
        let (u1, v1) = (a[0], a[1]);
        let (u2, v2) = (b[0], b[1]);
        let r = SVector::<f64, 9>::from_column_slice(&[
            u2 * u1,
            u2 * v1,
            u2,
            v2 * u1,
            v2 * v1,
            v2,
            u1,
            v1,
            1.0,
        ]);
        ata += r * r.transpose();
    }
    ata
}

// ── Minimal and refit solvers ────────────────────────────────────────────────

/// One to three fundamental matrices from seven correspondences.
///
/// Each returned `F` has unit Frobenius norm and rank 2 (satisfies the seven
/// epipolar constraints to floating-point accuracy and `det F = 0`). Degenerate
/// inputs return an empty vector: a design matrix whose null space exceeds two
/// dimensions (repeated points or otherwise fewer than seven independent
/// constraints) and non-finite values. Pure and deterministic.
pub fn fundamental_7pt(x1: &[[f64; 2]; 7], x2: &[[f64; 2]; 7]) -> Vec<Matrix3<f64>> {
    let mut out = Vec::new();
    for p in x1.iter().chain(x2.iter()) {
        if !p[0].is_finite() || !p[1].is_finite() {
            return out;
        }
    }
    let Some((n1, t1)) = hartley_normalize(x1) else {
        return out;
    };
    let Some((n2, t2)) = hartley_normalize(x2) else {
        return out;
    };

    // Two-dimensional null space of the 7×9 design, from the two smallest
    // eigenvectors of AᵀA (thin SVD would not expose the full null space).
    let eig = design_normal_matrix(&n1, &n2).symmetric_eigen();
    let mut idx = [0usize, 1, 2, 3, 4, 5, 6, 7, 8];
    idx.sort_by(|&i, &j| eig.eigenvalues[i].total_cmp(&eig.eigenvalues[j]));
    let lmax = eig.eigenvalues[idx[8]];
    if lmax <= 0.0 || eig.eigenvalues[idx[2]] <= NULLSPACE_RANK_EPS * lmax {
        return out; // null space exceeds two dimensions — degenerate
    }
    let fa = vec_to_mat3(&eig.eigenvectors.column(idx[0]).into_owned());
    let fb = vec_to_mat3(&eig.eigenvectors.column(idx[1]).into_owned());

    // Cubic det(F_b + α·(F_a − F_b)) = 0 in the pencil basis.
    let d = fa - fb;
    let bc = |j: usize| fb.column(j).into_owned();
    let dc = |j: usize| d.column(j).into_owned();
    let c0 = fb.determinant();
    let c1 = det_cols(dc(0), bc(1), bc(2))
        + det_cols(bc(0), dc(1), bc(2))
        + det_cols(bc(0), bc(1), dc(2));
    let c2 = det_cols(bc(0), dc(1), dc(2))
        + det_cols(dc(0), bc(1), dc(2))
        + det_cols(dc(0), dc(1), bc(2));
    let c3 = d.determinant();

    for &alpha_raw in &solve_cubic(c3, c2, c1, c0) {
        let alpha = polish_cubic_root(c3, c2, c1, c0, alpha_raw);
        let f_hat = fb + alpha * d; // = α·F_a + (1−α)·F_b
        let f = t2.transpose() * f_hat * t1;
        let norm = f.norm();
        if norm < 1e-300 || !f.iter().all(|v| v.is_finite()) {
            continue;
        }
        out.push(f / norm);
    }
    out
}

/// Least-squares fundamental matrix from `N ≥ 8` correspondences (normalized
/// 8-point) with rank-2 enforcement.
///
/// The smallest right singular vector of the `N×9` design in normalized
/// coordinates, reshaped to `F̂`, projected to the closest rank-2 matrix, then
/// denormalized to `F = T₂ᵀ F̂ T₁` at unit Frobenius norm. `None` for `N < 8`,
/// non-finite input, or a rank-deficient design.
pub fn fundamental_8pt(x1: &[[f64; 2]], x2: &[[f64; 2]]) -> Option<Matrix3<f64>> {
    let n = x1.len();
    if n < 8 || x2.len() != n {
        return None;
    }
    for p in x1.iter().chain(x2.iter()) {
        if !p[0].is_finite() || !p[1].is_finite() {
            return None;
        }
    }
    let (n1, t1) = hartley_normalize(x1)?;
    let (n2, t2) = hartley_normalize(x2)?;

    let eig = design_normal_matrix(&n1, &n2).symmetric_eigen();
    let mut best = 0usize;
    for j in 1..9 {
        if eig.eigenvalues[j] < eig.eigenvalues[best] {
            best = j;
        }
    }
    // A rank-deficient design (fewer than 8 independent constraints) leaves the
    // null direction ambiguous — reject when the largest eigenvalue is ~0.
    if eig.eigenvalues.iter().cloned().fold(0.0, f64::max) <= 0.0 {
        return None;
    }
    let f_hat = vec_to_mat3(&eig.eigenvectors.column(best).into_owned());
    let f_rank2 = enforce_rank2(&f_hat)?;
    let f = t2.transpose() * f_rank2 * t1;
    let norm = f.norm();
    if norm < 1e-300 || !f.iter().all(|v| v.is_finite()) {
        return None;
    }
    Some(f / norm)
}

/// Squared Sampson distance (pixels²) of a correspondence against `F` — the
/// first-order reprojection error. Returns `+∞` when the denominator vanishes.
fn sampson_sq(f: &Matrix3<f64>, x1: [f64; 2], x2: [f64; 2]) -> f64 {
    let p1 = Vector3::new(x1[0], x1[1], 1.0);
    let p2 = Vector3::new(x2[0], x2[1], 1.0);
    let fx1 = f * p1;
    let ftx2 = f.transpose() * p2;
    let num = p2.dot(&fx1); // x̃₂ᵀ F x̃₁
    let den = fx1[0] * fx1[0] + fx1[1] * fx1[1] + ftx2[0] * ftx2[0] + ftx2[1] * ftx2[1];
    if den <= 1e-300 {
        return f64::INFINITY;
    }
    num * num / den
}

// ── Robust estimator ─────────────────────────────────────────────────────────

/// Tuning for [`estimate_fundamental`].
#[derive(Clone, Debug)]
pub struct FundamentalOptions {
    /// Inlier bound on the Sampson distance, pixels.
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
    /// Refit each new best consensus on its inliers (normalized 8-point),
    /// repeating while the inlier set grows.
    pub local_optimization: bool,
}

impl Default for FundamentalOptions {
    fn default() -> Self {
        Self {
            max_error_px: 3.0,
            confidence: 0.999,
            max_iterations: 10_000,
            min_inliers: 12,
            seed: 0,
            local_optimization: true,
        }
    }
}

/// Result of [`estimate_fundamental`].
#[derive(Clone, Debug)]
pub struct FundamentalEstimate {
    /// Unit Frobenius norm, rank 2.
    pub f_matrix: Matrix3<f64>,
    /// Per-input-correspondence inlier mask.
    pub inliers: Vec<bool>,
    /// Trials actually run.
    pub iterations: u32,
}

/// SplitMix64 — the deterministic index sampler (no `rand` dependency,
/// identical across platforms). Mirrors the generator in
/// [`crate::geometry::absolute_pose`].
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

/// Draw seven distinct indices in `0..n` with the seeded sampler.
fn draw7(state: &mut u64, n: usize) -> [usize; 7] {
    let mut idx = [0usize; 7];
    let mut k = 0;
    while k < 7 {
        let cand = (splitmix64(state) % n as u64) as usize;
        if !idx[..k].contains(&cand) {
            idx[k] = cand;
            k += 1;
        }
    }
    idx
}

/// Count and mark inliers for `F`, scoring in input order (deterministic).
fn score_f(
    f: &Matrix3<f64>,
    x1: &[[f64; 2]],
    x2: &[[f64; 2]],
    thresh2: f64,
    mask: &mut [bool],
) -> usize {
    let mut count = 0;
    for i in 0..x1.len() {
        let inlier = sampson_sq(f, x1[i], x2[i]) <= thresh2;
        mask[i] = inlier;
        count += inlier as usize;
    }
    count
}

/// Refit `F` on its inliers with the 8-point solver, rescore, and repeat while
/// the inlier count strictly grows (bounded rounds). Returns the best refit `F`,
/// its inlier mask (in `scratch`), and inlier count.
fn local_optimize_f(
    mut f: Matrix3<f64>,
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
        if idx.len() < 8 {
            break;
        }
        let sub1: Vec<[f64; 2]> = idx.iter().map(|&i| x1[i]).collect();
        let sub2: Vec<[f64; 2]> = idx.iter().map(|&i| x2[i]).collect();
        let Some(f_ref) = fundamental_8pt(&sub1, &sub2) else {
            break;
        };
        let new_count = score_f(&f_ref, x1, x2, thresh2, scratch);
        if new_count > count {
            f = f_ref;
            count = new_count;
            cur_mask.copy_from_slice(scratch);
        } else {
            // Did not grow the consensus: restore the last accepted mask.
            scratch.copy_from_slice(&cur_mask);
            break;
        }
    }
    (f, count)
}

/// Robustly estimate the fundamental matrix from pixel correspondences (7-point
/// LO-RANSAC). Returns `None` when no consensus reaches `min_inliers`.
/// See `specs/core/epipolar-estimation.md`.
pub fn estimate_fundamental(
    x1: &[[f64; 2]],
    x2: &[[f64; 2]],
    options: &FundamentalOptions,
) -> Option<FundamentalEstimate> {
    let n = x1.len();
    if n != x2.len() || n < 7 {
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

        let idx = draw7(&mut state, n);
        let s1: [[f64; 2]; 7] = core::array::from_fn(|k| x1[idx[k]]);
        let s2: [[f64; 2]; 7] = core::array::from_fn(|k| x2[idx[k]]);

        for f in fundamental_7pt(&s1, &s2) {
            let mut count = score_f(&f, x1, x2, thresh2, &mut scratch);
            if count > best_count {
                let mut f_best = f;
                if options.local_optimization {
                    (f_best, count) =
                        local_optimize_f(f_best, x1, x2, thresh2, count, &mut scratch);
                }
                best_count = count;
                best = Some(f_best);
                best_mask.copy_from_slice(&scratch);

                // Adaptive termination bound from the current best inlier rate.
                let w = best_count as f64 / n as f64;
                let w7 = w.powi(7);
                required = if w7 >= 1.0 {
                    iterations as u64
                } else if w7 <= 0.0 {
                    options.max_iterations as u64
                } else {
                    let num = (1.0 - options.confidence).ln();
                    let den = (1.0 - w7).ln();
                    (num / den).ceil().max(0.0) as u64
                };
            }
        }
    }

    let f = best?;
    // Return a strictly rank-2, unit-Frobenius matrix, and refresh the mask so
    // it is consistent with the returned `F`.
    let f = enforce_rank2(&f).unwrap_or(f);
    let norm = f.norm();
    let f = if norm > 1e-300 { f / norm } else { f };
    best_count = score_f(&f, x1, x2, thresh2, &mut best_mask);
    if best_count < options.min_inliers {
        return None;
    }
    Some(FundamentalEstimate {
        f_matrix: f,
        inliers: best_mask,
        iterations,
    })
}

// ── Focal length from the fundamental matrix (Bougnoux) ──────────────────────

/// Focal length of camera 1 in pixels from the fundamental matrix and the two
/// principal points (Bougnoux). `None` when the pair is degenerate for focal
/// recovery — vanishing denominator or `f₁² ≤ 0` (intersecting optical axes,
/// pure forward translation, rotation-dominant motion). Assumes square pixels,
/// zero skew, and known principal points. See `specs/core/epipolar-estimation.md`.
pub fn focal_from_fundamental(
    f_matrix: &Matrix3<f64>,
    pp1: [f64; 2],
    pp2: [f64; 2],
) -> Option<f64> {
    if !f_matrix.iter().all(|v| v.is_finite()) {
        return None;
    }
    let norm = f_matrix.norm();
    if norm < ZERO_F_EPS {
        return None; // no epipolar geometry (e.g. rotation-only motion)
    }
    // Scale-invariant, but normalize so the denominator threshold is meaningful.
    let f = f_matrix / norm;

    let p1 = Vector3::new(pp1[0], pp1[1], 1.0);
    let p2 = Vector3::new(pp2[0], pp2[1], 1.0);

    // Epipole in image 2: Fᵀ ẽ₂ = 0 — the smallest left singular vector of F.
    let svd = f.svd(true, true);
    let u = svd.u?;
    let e2 = u.column(2).into_owned();
    let e2x = skew_symmetric(&e2);
    let i_tilde = Matrix3::new(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0);

    // f₁² = −(p̃₂ᵀ [ẽ₂]ₓ Ĩ F p̃₁)(p̃₂ᵀ F p̃₁) / (p̃₂ᵀ [ẽ₂]ₓ Ĩ F Ĩ Fᵀ p̃₂)
    let num =
        -(p2.transpose() * e2x * i_tilde * f * p1)[(0, 0)] * (p2.transpose() * f * p1)[(0, 0)];
    let den = (p2.transpose() * e2x * i_tilde * f * i_tilde * f.transpose() * p2)[(0, 0)];
    if den.abs() < BOUGNOUX_DEN_EPS {
        return None;
    }
    let f1_sq = num / den;
    // `f₁² ≤ 0` (or non-finite) is the classical degeneracy sign test.
    if !f1_sq.is_finite() || f1_sq <= 0.0 {
        return None;
    }
    Some(f1_sq.sqrt())
}

#[cfg(test)]
mod tests;
