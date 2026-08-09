// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Relative pose between two images of the same scene, taken with the same
//! **known** camera intrinsics and differing only in pose. Known intrinsics
//! turn each observed pixel into the unit ray the camera saw it along, so the
//! only unknown is the motion between the two poses: a rotation, plus a
//! translation direction (two views fix no scale). This is the 2D–2D sibling
//! of [`super::absolute_pose`], with one robust estimator per motion model of
//! such a pair: general motion (the epipolar matrix) and rotation-only motion.
//!
//! The focal-vote's camera-model columns
//! ([`super::focal_vote::column_scan`]) estimate exactly these two models —
//! a ray-space epipolar matrix and a ray-space rotation — but they estimate
//! them *per candidate focal*, as a self-consistency scan whose output is a
//! focal. A caller that already has a camera (a confirmed model verdict and
//! its focal) wants the same estimators evaluated for **fixed camera
//! intrinsics**, and wants the geometry back rather than a focal. That is what this module exposes; the
//! sampling, consensus, local-optimization and residual machinery is the
//! column scan's, shared rather than reimplemented.
//!
//! Everything is on the sphere. A field of view past 180° puts a substantial
//! share of the correspondences at `θ ≥ 90°`, where there is no `z = 1` plane
//! to normalize onto: rays with a non-positive `z` are ordinary members of the
//! population here, residuals are angles, and the cheirality a caller tests
//! after decomposing the essential matrix must be depth **along the ray**, not
//! `z > 0`.
//!
//! Both estimators are seeded and deterministic: minimal samples are drawn
//! from the caller's seed, so identical input plus identical seed gives a
//! bit-identical answer.

use nalgebra::{Matrix3, Vector3};

use super::focal_vote::column_scan::{
    draw_samples, epipolar_residuals, epipolar_rows, kabsch, null_from_rows, rotation_residuals,
    singular_values_desc,
};

/// Minimal-sample size of the linear 8-point ray-space epipolar constraint.
const EPI_SAMPLE: usize = 8;
/// Minimal-sample size of the rotation fit (three rays fix a rotation).
const ROT_SAMPLE: usize = 3;
/// Local-optimization rounds on the consensus set.
const LO_ROUNDS: usize = 3;

/// How an epipolar consensus scores a correspondence.
///
/// The one-sided forms are the column scan's, where the two directions are
/// deliberately separate measurements. A caller estimating *geometry* rather
/// than certifying a focal wants both constraints at once, which is
/// [`EpipolarSide::Both`] — the default.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum EpipolarSide {
    /// The larger of the two one-sided angles.
    #[default]
    Both,
    /// Image-2 rays against the epipolar plane `E·x₁`.
    Two,
    /// Image-1 rays against the epipolar plane `Eᵀ·x₂`.
    One,
}

/// Settings of [`estimate_essential_rays`].
#[derive(Clone, Copy, Debug)]
pub struct RayEssentialOptions {
    /// Consensus bound on the ray-to-epipolar-plane angle, radians. Derive it
    /// from a keypoint localization tolerance through the camera map's local
    /// `dr/dθ` (for an equidistant map, `tol_px / f`).
    pub max_angle_rad: f64,
    /// Reject a consensus below this many correspondences.
    pub min_inliers: usize,
    /// Minimal samples drawn (each is `EPI_SAMPLE` = 8 distinct indices).
    pub samples: usize,
    /// SplitMix64 sampler seed.
    pub seed: u64,
    /// Which side(s) the residual measures.
    pub side: EpipolarSide,
}

impl Default for RayEssentialOptions {
    fn default() -> Self {
        Self {
            max_angle_rad: 0.01,
            min_inliers: 12,
            samples: 512,
            seed: 0,
            side: EpipolarSide::Both,
        }
    }
}

/// A robust ray-space epipolar estimate.
#[derive(Clone, Debug)]
pub struct RayEssential {
    /// The epipolar matrix of the consensus refit, unit Frobenius norm. Under
    /// a correct camera model it is essential, i.e. its two leading singular
    /// values are equal.
    pub e_matrix: Matrix3<f64>,
    /// Consensus mask over the input correspondences.
    pub inliers: Vec<bool>,
    /// Essentialness residual `(σ₁ − σ₂)/(σ₁ + σ₂)` — zero for a perfectly
    /// essential matrix, and the column scan's cost.
    pub essentialness: f64,
    /// Per-correspondence angular residual, radians.
    pub residuals_rad: Vec<f64>,
    /// Root-mean-square angular residual over the consensus, radians.
    pub rms_rad: f64,
}

/// Angular residual of every correspondence against `e`, per `side`.
fn residuals_for(
    e: &Matrix3<f64>,
    r1: &[Vector3<f64>],
    r2: &[Vector3<f64>],
    side: EpipolarSide,
    scratch: &mut [f64],
    out: &mut [f64],
) {
    match side {
        EpipolarSide::Two => epipolar_residuals(e, r1, r2, true, out),
        EpipolarSide::One => epipolar_residuals(e, r1, r2, false, out),
        EpipolarSide::Both => {
            epipolar_residuals(e, r1, r2, true, out);
            epipolar_residuals(e, r1, r2, false, scratch);
            for (o, s) in out.iter_mut().zip(scratch.iter()) {
                *o = o.max(*s);
            }
        }
    }
}

/// Robust ray-space epipolar matrix from unit-ray correspondences.
///
/// Scores seeded minimal samples of the linear 8-point constraint by the
/// angular residual, keeps the largest consensus, then locally optimizes on
/// it. Returns `None` when no consensus reaches `min_inliers`.
///
/// The returned matrix is **not** projected onto the essential manifold: how
/// far it is from essential is the measurement the column scan reads
/// ([`RayEssential::essentialness`]), and a caller decomposing it to a
/// relative pose gets the projection for free from the SVD it already needs.
pub fn estimate_essential_rays(
    r1: &[Vector3<f64>],
    r2: &[Vector3<f64>],
    options: &RayEssentialOptions,
) -> Option<RayEssential> {
    let n = r1.len().min(r2.len());
    if n < EPI_SAMPLE.max(options.min_inliers)
        || options.max_angle_rad <= 0.0
        || !options.max_angle_rad.is_finite()
    {
        return None;
    }
    let sin_tol = options.max_angle_rad.min(std::f64::consts::FRAC_PI_2).sin();
    let rows = epipolar_rows(&r1[..n], &r2[..n]);
    let mut state = options.seed;
    let samples = draw_samples(&mut state, n, EPI_SAMPLE, options.samples.max(1));

    let mut resid = vec![0.0f64; n];
    let mut scratch = vec![0.0f64; n];
    let mut best_count = 0usize;
    let mut best_e: Option<Matrix3<f64>> = None;
    let mut inliers = vec![false; n];
    for s in samples.chunks_exact(EPI_SAMPLE) {
        let Some(e) = null_from_rows(&rows, s.iter().copied()) else {
            continue;
        };
        residuals_for(&e, r1, r2, options.side, &mut scratch, &mut resid);
        let count = (0..n).filter(|&i| resid[i] < sin_tol).count();
        if count > best_count {
            best_count = count;
            best_e = Some(e);
            for i in 0..n {
                inliers[i] = resid[i] < sin_tol;
            }
        }
    }
    if best_count < options.min_inliers {
        return None;
    }
    let mut e = best_e?;
    for _ in 0..LO_ROUNDS {
        let Some(refit) = null_from_rows(&rows, (0..n).filter(|&i| inliers[i])) else {
            break;
        };
        residuals_for(&refit, r1, r2, options.side, &mut scratch, &mut resid);
        let new: Vec<bool> = (0..n).map(|i| resid[i] < sin_tol).collect();
        if new.iter().filter(|&&b| b).count() < options.min_inliers {
            break;
        }
        e = refit;
        let done = new == inliers;
        inliers = new;
        if done {
            break;
        }
    }
    residuals_for(&e, r1, r2, options.side, &mut scratch, &mut resid);
    for i in 0..n {
        inliers[i] = resid[i] < sin_tol;
    }
    let n_inl = inliers.iter().filter(|&&b| b).count();
    if n_inl < options.min_inliers {
        return None;
    }
    let s = singular_values_desc(&e);
    let denom = s[0] + s[1];
    if denom <= 0.0 || !denom.is_finite() {
        return None;
    }
    let residuals_rad: Vec<f64> = resid.iter().map(|&x| x.min(1.0).asin()).collect();
    let sq: f64 = (0..n)
        .filter(|&i| inliers[i])
        .map(|i| residuals_rad[i] * residuals_rad[i])
        .sum();
    Some(RayEssential {
        e_matrix: e / e.norm(),
        inliers,
        essentialness: (s[0] - s[1]) / denom,
        residuals_rad,
        rms_rad: (sq / n_inl as f64).sqrt(),
    })
}

/// Settings of [`fit_ray_rotation`].
#[derive(Clone, Copy, Debug)]
pub struct RayRotationOptions {
    /// Consensus bound on the angle between a rotated ray and its partner,
    /// radians.
    pub max_angle_rad: f64,
    /// Reject a consensus below this many correspondences.
    pub min_inliers: usize,
    /// Minimal samples drawn (each is `ROT_SAMPLE` = 3 distinct indices).
    pub samples: usize,
    /// SplitMix64 sampler seed.
    pub seed: u64,
}

impl Default for RayRotationOptions {
    fn default() -> Self {
        Self {
            max_angle_rad: 0.01,
            min_inliers: 20,
            samples: 512,
            seed: 0,
        }
    }
}

/// A robust ray-space rotation estimate.
#[derive(Clone, Debug)]
pub struct RayRotation {
    /// The rotation taking image-1 rays onto image-2 rays: `R x₁ ≈ x₂`.
    pub rotation: Matrix3<f64>,
    /// Consensus mask over the input correspondences.
    pub inliers: Vec<bool>,
    /// Per-correspondence angular residual, radians.
    pub residuals_rad: Vec<f64>,
    /// Root-mean-square angular residual over the consensus, radians.
    pub rms_rad: f64,
}

/// Robust ray-space rotation from unit-ray correspondences — the far-field
/// model of a pair, valid over the whole sphere (`θ ≥ 90°` rays participate
/// like any others).
///
/// Scores seeded three-ray minimal samples by the angular residual, keeps the
/// largest consensus, then refits on it. Returns `None` when no consensus
/// reaches `min_inliers`.
pub fn fit_ray_rotation(
    r1: &[Vector3<f64>],
    r2: &[Vector3<f64>],
    options: &RayRotationOptions,
) -> Option<RayRotation> {
    let n = r1.len().min(r2.len());
    if n < ROT_SAMPLE.max(options.min_inliers)
        || options.max_angle_rad <= 0.0
        || !options.max_angle_rad.is_finite()
    {
        return None;
    }
    let tol = options.max_angle_rad;
    let all: Vec<usize> = (0..n).collect();
    let mut state = options.seed;
    let samples = draw_samples(&mut state, n, ROT_SAMPLE, options.samples.max(1));

    let mut resid = vec![0.0f64; n];
    let mut best_count = 0usize;
    let mut best: Option<Matrix3<f64>> = None;
    let mut inliers = vec![false; n];
    for s in samples.chunks_exact(ROT_SAMPLE) {
        let Some(rot) = kabsch(r1, r2, s) else {
            continue;
        };
        rotation_residuals(&rot, r1, r2, &all, &mut resid);
        let count = (0..n).filter(|&i| resid[i] < tol).count();
        if count > best_count {
            best_count = count;
            best = Some(rot);
            for i in 0..n {
                inliers[i] = resid[i] < tol;
            }
        }
    }
    if best_count < options.min_inliers {
        return None;
    }
    let mut rot = best?;
    for _ in 0..LO_ROUNDS {
        let idx: Vec<usize> = (0..n).filter(|&i| inliers[i]).collect();
        let Some(refit) = kabsch(r1, r2, &idx) else {
            break;
        };
        rotation_residuals(&refit, r1, r2, &all, &mut resid);
        let new: Vec<bool> = (0..n).map(|i| resid[i] < tol).collect();
        if new.iter().filter(|&&b| b).count() < options.min_inliers {
            break;
        }
        rot = refit;
        let done = new == inliers;
        inliers = new;
        if done {
            break;
        }
    }
    rotation_residuals(&rot, r1, r2, &all, &mut resid);
    for i in 0..n {
        inliers[i] = resid[i] < tol;
    }
    let n_inl = inliers.iter().filter(|&&b| b).count();
    if n_inl < options.min_inliers {
        return None;
    }
    let sq: f64 = (0..n)
        .filter(|&i| inliers[i])
        .map(|i| resid[i] * resid[i])
        .sum();
    Some(RayRotation {
        rotation: rot,
        inliers,
        residuals_rad: resid,
        rms_rad: (sq / n_inl as f64).sqrt(),
    })
}

#[cfg(test)]
mod tests;
