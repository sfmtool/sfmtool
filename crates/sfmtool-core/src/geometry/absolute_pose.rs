// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Absolute pose from 2D-3D correspondences: a Lambda Twist minimal solver
//! ([`p3p_solve`]) and a deterministic RANSAC estimator
//! ([`estimate_absolute_pose`]).
//!
//! See `specs/core/absolute-pose.md`. Given `N` correspondences between
//! observed image **bearings** (unit ray directions in the canonical camera
//! frame — a camera looks along `−Z`, a point in front has `z < 0`) and known
//! world points, estimate the camera's world-to-camera pose `x_cam = R·X + t`,
//! robust to a heavily contaminated correspondence set.
//!
//! The minimal solver returns up to four poses from three correspondences by
//! solving the two-quadric depth system with the Lambda Twist parameterization
//! (Persson & Nordberg, ECCV 2018): the depths follow from a real root of the
//! pencil-degeneracy cubic and the eigendecomposition of a 3×3 symmetric
//! matrix, and each depth triple is upgraded to a rigid motion by a three-point
//! Kabsch alignment. It is pure: no allocation beyond the fixed-capacity
//! result, no randomness, bit-stable across runs.
//!
//! The estimator draws minimal samples with a SplitMix64 sampler (seeded, so
//! same inputs + same seed give bit-identical output), scores every candidate
//! pose against all correspondences by the angular inlier test in input order,
//! keeps the best consensus, optionally locally optimizes it (Gauss-Newton on
//! angular residuals over the inliers), and terminates adaptively.
//!
//! **Deviation from the spec** (2026-07-14): the spec's [`p3p_solve`] signature
//! returns `ArrayVec<_, 4>`. The workspace has no `arrayvec` dependency and
//! does not use fixed-capacity vectors elsewhere, so the solver returns a
//! `Vec<(UnitQuaternion<f64>, Vector3<f64>)>` (allocated once with capacity 4).
//! Recorded in the spec.

use nalgebra::{Matrix3, Matrix6, Point3, UnitQuaternion, Vector3, Vector6};

/// A world-to-camera pose: rotation and translation with `x_cam = R·X + t`.
type Pose = (UnitQuaternion<f64>, Vector3<f64>);

/// Below this the pencil's leading cubic coefficient is treated as zero and
/// the cubic collapses to a quadratic (the `det(D2)` term vanishes when `D2`
/// is singular).
const CUBIC_LEADING_EPS: f64 = 1e-13;

/// A depth-system direction is accepted only if the quadric it is scaled
/// against has magnitude above this (relative to the point spread), guarding
/// the `k² = a / dᵀMd` scale recovery against division by ~0.
const SCALE_DENOM_EPS: f64 = 1e-12;

/// Kabsch rejects a point triple whose cross-covariance smallest singular
/// value is below this fraction of the largest — collinear or coincident world
/// points, where the rigid alignment is not determined.
const KABSCH_RANK_EPS: f64 = 1e-9;

/// Up to four world-to-camera poses from three bearing/point correspondences.
///
/// `bearings[i]` is the observed unit ray of point `points[i]` in the canonical
/// camera frame. Returns every real, positive-depth solution `(R, t)` with
/// `x_cam = R·X + t`; degenerate inputs (collinear points, coincident or
/// antipodal bearings, non-finite values) return an empty vector rather than a
/// NaN pose. Pure and deterministic.
pub fn p3p_solve(bearings: &[Vector3<f64>; 3], points: &[Point3<f64>; 3]) -> Vec<Pose> {
    let mut out: Vec<Pose> = Vec::with_capacity(4);

    // Reject non-finite input up front.
    for b in bearings {
        if !b.iter().all(|v| v.is_finite()) {
            return out;
        }
    }
    for p in points {
        if !p.coords.iter().all(|v| v.is_finite()) {
            return out;
        }
    }

    // Normalize bearings defensively; a zero-length bearing is degenerate.
    let mut y = [Vector3::zeros(); 3];
    for i in 0..3 {
        let n = bearings[i].norm();
        if n < 1e-12 {
            return out;
        }
        y[i] = bearings[i] / n;
    }

    // Pairwise bearing cosines and squared world distances.
    let b12 = y[0].dot(&y[1]);
    let b13 = y[0].dot(&y[2]);
    let b23 = y[1].dot(&y[2]);
    let a12 = (points[0] - points[1]).norm_squared();
    let a13 = (points[0] - points[2]).norm_squared();
    let a23 = (points[1] - points[2]).norm_squared();

    // Coincident world points or coincident/antipodal bearings are degenerate.
    if a12 < 1e-18 || a13 < 1e-18 || a23 < 1e-18 {
        return out;
    }
    if (b12.abs() - 1.0).abs() < 1e-12
        || (b13.abs() - 1.0).abs() < 1e-12
        || (b23.abs() - 1.0).abs() < 1e-12
    {
        return out;
    }

    // Quadric matrices: Λᵀ M·Λ = a for each pair, with Λ = (λ1, λ2, λ3).
    let m12 = Matrix3::new(1.0, -b12, 0.0, -b12, 1.0, 0.0, 0.0, 0.0, 0.0);
    let m13 = Matrix3::new(1.0, 0.0, -b13, 0.0, 0.0, 0.0, -b13, 0.0, 1.0);
    let m23 = Matrix3::new(0.0, 0.0, 0.0, 0.0, 1.0, -b23, 0.0, -b23, 1.0);

    // Two homogeneous quadrics through the origin: Λᵀ D·Λ = 0.
    let d1 = a23 * m12 - a12 * m23;
    let d2 = a23 * m13 - a13 * m23;

    // Cubic det(D1 + γ·D2) = 0: the pencil member that becomes rank-deficient
    // (its quadric splits into two real planes).
    let (c1, c2, c3) = (d1.column(0), d1.column(1), d1.column(2));
    let (e1, e2, e3) = (d2.column(0), d2.column(1), d2.column(2));
    let det3 = |a: nalgebra::Vector3<f64>, b: nalgebra::Vector3<f64>, c: nalgebra::Vector3<f64>| {
        Matrix3::from_columns(&[a, b, c]).determinant()
    };
    let coeff0 = d1.determinant();
    let coeff1 = det3(e1.into(), c2.into(), c3.into())
        + det3(c1.into(), e2.into(), c3.into())
        + det3(c1.into(), c2.into(), e3.into());
    let coeff2 = det3(c1.into(), e2.into(), e3.into())
        + det3(e1.into(), c2.into(), e3.into())
        + det3(e1.into(), e2.into(), c3.into());
    let coeff3 = d2.determinant();

    for &gamma_raw in &solve_cubic(coeff3, coeff2, coeff1, coeff0) {
        // Newton polish of the root against the exact cubic.
        let gamma = polish_cubic_root(coeff3, coeff2, coeff1, coeff0, gamma_raw);
        let d0 = d1 + gamma * d2;

        // Eigendecompose the (rank-deficient) quadric; the two planes exist
        // only when the two dominant eigenvalues have opposite sign.
        let eig = d0.symmetric_eigen();
        let mut idx = [0usize, 1, 2];
        idx.sort_by(|&i, &j| {
            eig.eigenvalues[j]
                .abs()
                .total_cmp(&eig.eigenvalues[i].abs())
        });
        let (ip, iq, iz) = (idx[0], idx[1], idx[2]);
        let (sp, sq) = (eig.eigenvalues[ip], eig.eigenvalues[iq]);
        if sp * sq >= 0.0 || sp.abs() < 1e-15 {
            continue;
        }
        let ep = eig.eigenvectors.column(ip).into_owned();
        let eq = eig.eigenvectors.column(iq).into_owned();
        let ez = eig.eigenvectors.column(iz).into_owned();
        // σp·(epᵀΛ)² + σq·(eqᵀΛ)² = 0 ⇒ epᵀΛ = ±w·eqᵀΛ, w = √(−σq/σp).
        let w = (-sq / sp).sqrt();
        if !w.is_finite() {
            continue;
        }

        for &tau in &[1.0f64, -1.0] {
            // The plane's spanning direction (paired with the null vector ez).
            let u = tau * w * ep + eq;
            // Intersect the plane with D1 = 0: quadratic in (s = coord along ez).
            let a_q = (u.transpose() * d1 * u)[(0, 0)];
            let b_q = (u.transpose() * d1 * ez)[(0, 0)];
            let c_q = (ez.transpose() * d1 * ez)[(0, 0)];
            for dir in plane_quadric_dirs(a_q, b_q, c_q, &u, &ez) {
                if let Some(pose) =
                    pose_from_direction(&dir, &y, points, a12, a13, a23, m12, m13, m23)
                {
                    push_unique(&mut out, pose);
                }
            }
        }

        // One valid degenerate pencil member contains the whole intersection;
        // no need to try the remaining cubic roots.
        if !out.is_empty() {
            break;
        }
    }

    out
}

/// The ≤ 2 directions on a plane where it meets the quadric `D1 = 0`, from the
/// homogeneous quadratic `a·s₀² + 2b·s₀s₁ + c·s₁² = 0` in the plane's
/// coordinates `(s₀ along u, s₁ along ez)`.
fn plane_quadric_dirs(
    a: f64,
    b: f64,
    c: f64,
    u: &Vector3<f64>,
    ez: &Vector3<f64>,
) -> Vec<Vector3<f64>> {
    let mut dirs = Vec::with_capacity(2);
    // Solve with the better-conditioned variable in the denominator.
    if a.abs() >= c.abs() {
        // s = s₁/s₀ root of c·s² + 2b·s + a = 0; dir = u + s·ez.
        for s in solve_quadratic(c, 2.0 * b, a) {
            dirs.push(u + s * ez);
        }
    } else {
        // s = s₀/s₁ root of a·s² + 2b·s + c = 0; dir = s·u + ez.
        for s in solve_quadratic(a, 2.0 * b, c) {
            dirs.push(s * u + ez);
        }
    }
    dirs
}

/// Recover a world-to-camera pose from a depth-ratio direction: fix the scale
/// against the largest quadric constraint, enforce positive depths, and align
/// the camera-frame points to the world points by Kabsch. `None` when the
/// scale is imaginary, a depth is non-positive, or the alignment is
/// rank-deficient (collinear points).
#[allow(clippy::too_many_arguments)]
fn pose_from_direction(
    dir: &Vector3<f64>,
    y: &[Vector3<f64>; 3],
    points: &[Point3<f64>; 3],
    a12: f64,
    a13: f64,
    a23: f64,
    m12: Matrix3<f64>,
    m13: Matrix3<f64>,
    m23: Matrix3<f64>,
) -> Option<Pose> {
    if !dir.iter().all(|v| v.is_finite()) || dir.norm() < 1e-15 {
        return None;
    }
    // k² = a / (dirᵀ·M·dir); all three pairs are consistent, so pick the
    // best-conditioned denominator.
    let denoms = [
        (a12, (dir.transpose() * m12 * dir)[(0, 0)]),
        (a13, (dir.transpose() * m13 * dir)[(0, 0)]),
        (a23, (dir.transpose() * m23 * dir)[(0, 0)]),
    ];
    let (a_best, d_best) = *denoms
        .iter()
        .max_by(|x, y| x.1.abs().total_cmp(&y.1.abs()))?;
    if d_best.abs() < SCALE_DENOM_EPS {
        return None;
    }
    let k2 = a_best / d_best;
    // Reject non-positive or non-finite scale (NaN fails the `> 0` test).
    if k2 <= 0.0 || !k2.is_finite() {
        return None;
    }
    let k = k2.sqrt();

    // Depths must all be strictly positive. `dir` fixes the line; ±k choose
    // the ray — pick the sign that makes every depth positive.
    let lambda = if dir[0] > 0.0 && dir[1] > 0.0 && dir[2] > 0.0 {
        [k * dir[0], k * dir[1], k * dir[2]]
    } else if dir[0] < 0.0 && dir[1] < 0.0 && dir[2] < 0.0 {
        [-k * dir[0], -k * dir[1], -k * dir[2]]
    } else {
        return None;
    };
    if lambda.iter().any(|&l| l <= 0.0 || !l.is_finite()) {
        return None;
    }

    // Camera-frame points λ·b, then a rigid alignment to the world points.
    let cam = [
        Point3::from(lambda[0] * y[0]),
        Point3::from(lambda[1] * y[1]),
        Point3::from(lambda[2] * y[2]),
    ];
    kabsch(points, &cam)
}

/// Exact three-point rigid alignment: the world-to-camera `(R, t)` with
/// `cam ≈ R·world + t`. `None` when the point cloud is collinear/coincident
/// (rank-deficient cross-covariance).
fn kabsch(world: &[Point3<f64>; 3], cam: &[Point3<f64>; 3]) -> Option<Pose> {
    let world_c = (world[0].coords + world[1].coords + world[2].coords) / 3.0;
    let cam_c = (cam[0].coords + cam[1].coords + cam[2].coords) / 3.0;
    let mut h = Matrix3::zeros();
    for i in 0..3 {
        let p = world[i].coords - world_c;
        let q = cam[i].coords - cam_c;
        h += p * q.transpose();
    }
    let svd = h.svd(true, true);
    let (u, v_t) = (svd.u?, svd.v_t?);
    // Three points are always coplanar, so the cross-covariance is rank 2 by
    // construction (its third singular value is ~0 — the free plane-normal
    // direction, fixed by the determinant correction below). Collinear or
    // coincident points collapse the *second* singular value: that is the
    // rank-deficiency to reject.
    let smax = svd.singular_values[0];
    let s1 = svd.singular_values[1];
    if smax <= 0.0 || s1 < KABSCH_RANK_EPS * smax {
        return None; // collinear / coincident points
    }
    // R = V · diag(1, 1, det(V·Uᵀ)) · Uᵀ, so R is a proper rotation.
    let v = v_t.transpose();
    let ut = u.transpose();
    let mut d = Matrix3::identity();
    d[(2, 2)] = (v * ut).determinant().signum();
    let rot = v * d * ut;
    let rotation =
        UnitQuaternion::from_rotation_matrix(&nalgebra::Rotation3::from_matrix_unchecked(rot));
    let translation = cam_c - rot * world_c;
    if !rotation.coords.iter().all(|v| v.is_finite()) || !translation.iter().all(|v| v.is_finite())
    {
        return None;
    }
    Some((rotation, translation))
}

/// Append a pose unless a near-identical one is already present (guards against
/// coincident roots producing the same solution twice).
fn push_unique(out: &mut Vec<Pose>, pose: Pose) {
    for (r, t) in out.iter() {
        if (r.angle_to(&pose.0) < 1e-9) && ((t - pose.1).norm() < 1e-9) {
            return;
        }
    }
    out.push(pose);
}

/// Real roots of `c3·x³ + c2·x² + c1·x + c0`. Collapses to lower degree when
/// leading coefficients vanish.
fn solve_cubic(c3: f64, c2: f64, c1: f64, c0: f64) -> Vec<f64> {
    if c3.abs() < CUBIC_LEADING_EPS {
        return solve_quadratic(c2, c1, c0);
    }
    // Depressed cubic t³ + p·t + q via x = t − b/3.
    let (a, b, c) = (c2 / c3, c1 / c3, c0 / c3);
    let shift = a / 3.0;
    let p = b - a * a / 3.0;
    let q = 2.0 * a * a * a / 27.0 - a * b / 3.0 + c;
    let disc = q * q / 4.0 + p * p * p / 27.0;
    let mut roots = Vec::with_capacity(3);
    if disc > 0.0 {
        // One real root.
        let sqrt_disc = disc.sqrt();
        let u = (-q / 2.0 + sqrt_disc).cbrt();
        let v = (-q / 2.0 - sqrt_disc).cbrt();
        roots.push(u + v - shift);
    } else if disc.abs() <= 1e-300 || (p.abs() < 1e-300 && q.abs() < 1e-300) {
        // Triple/degenerate real root.
        let t = if p.abs() < 1e-300 { 0.0 } else { 3.0 * q / p };
        roots.push(t - shift);
    } else {
        // Three real roots (trigonometric form).
        let m = 2.0 * (-p / 3.0).sqrt();
        let theta = (3.0 * q / (p * m)).clamp(-1.0, 1.0).acos();
        for k in 0..3 {
            let t = m * ((theta + 2.0 * std::f64::consts::PI * k as f64) / 3.0).cos();
            roots.push(t - shift);
        }
    }
    roots
}

/// Real roots of `a·x² + b·x + c`.
fn solve_quadratic(a: f64, b: f64, c: f64) -> Vec<f64> {
    if a.abs() < 1e-300 {
        if b.abs() < 1e-300 {
            return Vec::new();
        }
        return vec![-c / b];
    }
    let disc = b * b - 4.0 * a * c;
    if disc < 0.0 {
        return Vec::new();
    }
    let sqrt_disc = disc.sqrt();
    // Numerically stable roots (avoid cancellation).
    let q = -0.5 * (b + b.signum() * sqrt_disc);
    let mut roots = Vec::with_capacity(2);
    if q.abs() > 1e-300 {
        roots.push(q / a);
        roots.push(c / q);
    } else {
        roots.push((-b + sqrt_disc) / (2.0 * a));
        roots.push((-b - sqrt_disc) / (2.0 * a));
    }
    roots
}

/// A few Newton steps polishing a cubic root against the exact polynomial.
fn polish_cubic_root(c3: f64, c2: f64, c1: f64, c0: f64, mut x: f64) -> f64 {
    for _ in 0..3 {
        let f = ((c3 * x + c2) * x + c1) * x + c0;
        let df = (3.0 * c3 * x + 2.0 * c2) * x + c1;
        if df.abs() < 1e-300 {
            break;
        }
        let step = f / df;
        x -= step;
        if step.abs() < 1e-15 * (1.0 + x.abs()) {
            break;
        }
    }
    x
}

// ── Robust estimator ───────────────────────────────────────────────────────

/// Tuning for [`estimate_absolute_pose`].
#[derive(Clone, Debug)]
pub struct AbsolutePoseOptions {
    /// Inlier bound on the bearing/prediction angle, radians. Any value below
    /// `π/2` subsumes the cheirality check.
    pub max_angular_error: f64,
    /// Adaptive-termination target: stop once the probability that an
    /// all-inlier sample was drawn exceeds this (given the best inlier count).
    pub confidence: f64,
    /// Hard trial cap.
    pub max_iterations: u32,
    /// Reject an estimate supported by fewer inliers than this.
    pub min_inliers: usize,
    /// SplitMix64 seed for the sampler.
    pub seed: u64,
    /// Refit each new best consensus on its inliers (Gauss-Newton on angular
    /// residuals), repeating while the inlier set grows.
    pub local_optimization: bool,
}

impl Default for AbsolutePoseOptions {
    fn default() -> Self {
        Self {
            max_angular_error: 0.01,
            confidence: 0.999,
            max_iterations: 50_000,
            min_inliers: 6,
            seed: 0,
            local_optimization: true,
        }
    }
}

/// Result of [`estimate_absolute_pose`].
#[derive(Clone, Debug)]
pub struct AbsolutePoseEstimate {
    /// World-to-camera rotation (canonical convention).
    pub rotation: UnitQuaternion<f64>,
    /// World-to-camera translation.
    pub translation: Vector3<f64>,
    /// Per-input-correspondence inlier mask.
    pub inliers: Vec<bool>,
    /// Trials actually run.
    pub iterations: u32,
}

/// SplitMix64 — the deterministic index sampler (no `rand` dependency,
/// identical across platforms). Mirrors the generator in
/// `patch/cluster_refine/consistency.rs`.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

/// Predicted unit direction `normalize(R·X + t)` in the canonical frame.
fn predict_dir(r: &UnitQuaternion<f64>, t: &Vector3<f64>, x: &Point3<f64>) -> Option<Vector3<f64>> {
    let g = r * x.coords + t;
    let n = g.norm();
    if n < 1e-15 || !n.is_finite() {
        return None;
    }
    Some(g / n)
}

/// Count and mark inliers for a pose, scoring in input order (deterministic).
fn score(
    r: &UnitQuaternion<f64>,
    t: &Vector3<f64>,
    bearings: &[Vector3<f64>],
    points: &[Point3<f64>],
    cos_thresh: f64,
    mask: &mut [bool],
) -> usize {
    let mut count = 0;
    for i in 0..bearings.len() {
        let inlier = match predict_dir(r, t, &points[i]) {
            Some(d) => bearings[i].dot(&d) >= cos_thresh,
            None => false,
        };
        mask[i] = inlier;
        count += inlier as usize;
    }
    count
}

/// Robustly estimate the world-to-camera pose from bearing/point
/// correspondences. Returns `None` when no consensus reaches `min_inliers`.
/// See `specs/core/absolute-pose.md`.
pub fn estimate_absolute_pose(
    bearings: &[Vector3<f64>],
    points: &[Point3<f64>],
    options: &AbsolutePoseOptions,
) -> Option<AbsolutePoseEstimate> {
    let n = bearings.len();
    if n != points.len() || n < 3 {
        return None;
    }
    let cos_thresh = options.max_angular_error.cos();

    let mut state = options.seed;
    let mut best_count = 0usize;
    let mut best: Option<Pose> = None;
    let mut best_mask = vec![false; n];
    let mut scratch = vec![false; n];

    let mut iterations = 0u32;
    let mut required = options.max_iterations as u64;
    while (iterations as u64) < required && iterations < options.max_iterations {
        iterations += 1;

        // Draw three distinct indices with the seeded sampler.
        let i0 = (splitmix64(&mut state) % n as u64) as usize;
        let mut i1 = (splitmix64(&mut state) % n as u64) as usize;
        while i1 == i0 {
            i1 = (splitmix64(&mut state) % n as u64) as usize;
        }
        let mut i2 = (splitmix64(&mut state) % n as u64) as usize;
        while i2 == i0 || i2 == i1 {
            i2 = (splitmix64(&mut state) % n as u64) as usize;
        }

        let sample_b = [bearings[i0], bearings[i1], bearings[i2]];
        let sample_x = [points[i0], points[i1], points[i2]];
        for (mut r, mut t) in p3p_solve(&sample_b, &sample_x) {
            let mut count = score(&r, &t, bearings, points, cos_thresh, &mut scratch);
            if count > best_count {
                // Local optimization: refit on inliers while the set grows.
                if options.local_optimization {
                    (r, t, count) =
                        local_optimize(r, t, bearings, points, cos_thresh, count, &mut scratch);
                }
                best_count = count;
                best = Some((r, t));
                best_mask.copy_from_slice(&scratch);

                // Adaptive termination bound from the current best inlier rate.
                let w = best_count as f64 / n as f64;
                let w3 = w * w * w;
                required = if w3 >= 1.0 {
                    iterations as u64
                } else if w3 <= 0.0 {
                    options.max_iterations as u64
                } else {
                    let num = (1.0 - options.confidence).ln();
                    let den = (1.0 - w3).ln();
                    (num / den).ceil().max(0.0) as u64
                };
            }
        }
    }

    let (rotation, translation) = best?;
    if best_count < options.min_inliers {
        return None;
    }
    Some(AbsolutePoseEstimate {
        rotation,
        translation,
        inliers: best_mask,
        iterations,
    })
}

/// Refit a pose on its inliers, rescore, and repeat while the inlier count
/// strictly grows (bounded rounds). Returns the best refit pose, its inlier
/// mask (in `scratch`), and inlier count.
fn local_optimize(
    mut r: UnitQuaternion<f64>,
    mut t: Vector3<f64>,
    bearings: &[Vector3<f64>],
    points: &[Point3<f64>],
    cos_thresh: f64,
    mut count: usize,
    scratch: &mut [bool],
) -> (UnitQuaternion<f64>, Vector3<f64>, usize) {
    const MAX_ROUNDS: usize = 10;
    let mut cur_mask: Vec<bool> = scratch.to_vec();
    for _ in 0..MAX_ROUNDS {
        let inliers: Vec<usize> = (0..bearings.len()).filter(|&i| cur_mask[i]).collect();
        if inliers.len() < 3 {
            break;
        }
        let (rr, tt) = refine_pose(r, t, bearings, points, &inliers);
        let new_count = score(&rr, &tt, bearings, points, cos_thresh, scratch);
        if new_count > count {
            r = rr;
            t = tt;
            count = new_count;
            cur_mask.copy_from_slice(scratch);
        } else {
            // Keep the refined pose only if it did not shrink the consensus;
            // otherwise restore the last accepted mask into `scratch`.
            scratch.copy_from_slice(&cur_mask);
            break;
        }
    }
    (r, t, count)
}

/// Gauss-Newton refinement of a pose minimizing the sum of squared angular
/// residuals over `inliers`, with a local `SO(3) × R³` parameterization
/// (left-composed rotation-vector increments).
fn refine_pose(
    mut r: UnitQuaternion<f64>,
    mut t: Vector3<f64>,
    bearings: &[Vector3<f64>],
    points: &[Point3<f64>],
    inliers: &[usize],
) -> (UnitQuaternion<f64>, Vector3<f64>) {
    const MAX_ITERS: usize = 10;
    for _ in 0..MAX_ITERS {
        let rot = r.to_rotation_matrix();
        let mut h = Matrix6::<f64>::zeros();
        let mut g = Vector6::<f64>::zeros();
        let mut any = false;
        for &i in inliers {
            let rot_x = rot * points[i].coords;
            let gv = rot_x + t;
            let ng = gv.norm();
            if ng < 1e-15 {
                continue;
            }
            let d = gv / ng;
            let b = bearings[i];
            // r = (I − b·bᵀ)·d ; ∂r/∂d = (I − b·bᵀ) ; ∂d/∂g = (I − d·dᵀ)/‖g‖.
            let pb = Matrix3::identity() - b * b.transpose();
            let pd = Matrix3::identity() - d * d.transpose();
            let jr_g = pb * pd / ng;
            // ∂g/∂δ (rotation) = −[rot·X]_× ; ∂g/∂t = I.
            let jrot = jr_g * (-skew(&rot_x)); // 3×3, columns 0..3
            let jt = jr_g; // 3×3, columns 3..6 (since ∂g/∂t = I)
            let resid = pb * d; // 3-vector
                                // Stacked 3×6 Jacobian J = [jrot | jt]; accumulate H += Jᵀ·J and
                                // g += Jᵀ·resid over all six parameters uniformly.
            let col = |p: usize| -> Vector3<f64> {
                if p < 3 {
                    jrot.column(p).into_owned()
                } else {
                    jt.column(p - 3).into_owned()
                }
            };
            for a in 0..6 {
                let ja = col(a);
                g[a] += ja.dot(&resid);
                for c in a..6 {
                    let v = ja.dot(&col(c));
                    h[(a, c)] += v;
                    if a != c {
                        h[(c, a)] += v;
                    }
                }
            }
            any = true;
        }
        if !any {
            break;
        }
        // Small ridge for conditioning, then solve H·δ = −g.
        for k in 0..6 {
            h[(k, k)] += 1e-12;
        }
        let Some(hi) = h.try_inverse() else { break };
        let delta = -(hi * g);
        let drot = Vector3::new(delta[0], delta[1], delta[2]);
        let dt = Vector3::new(delta[3], delta[4], delta[5]);
        r = UnitQuaternion::from_scaled_axis(drot) * r;
        t += dt;
        if drot.norm() < 1e-12 && dt.norm() < 1e-12 {
            break;
        }
    }
    (r, t)
}

/// Skew-symmetric matrix `[v]_×` with `[v]_× w = v × w`.
fn skew(v: &Vector3<f64>) -> Matrix3<f64> {
    Matrix3::new(0.0, -v[2], v[1], v[2], 0.0, -v[0], -v[1], v[0], 0.0)
}

#[cfg(test)]
mod tests;
