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
use rayon::prelude::*;

use super::baseline_direction::in_front_of_both;
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
/// The fewest minimal samples scored on the rayon pool; fewer are scored on
/// the calling thread. Fitting and scoring one essential sample takes 2 to
/// 5 µs for 20 to 1,500 correspondences, and on the pool 16 samples took
/// about as long as on one thread, so from 32 the pool is faster at every
/// input size measured.
const PARALLEL_MIN_SAMPLES: usize = 32;

/// The most minimal samples either estimator draws: 65,536, 128 times the
/// default. The samples are held in one list of indices, eight per sample for
/// the essential matrix, so the bound keeps a caller's count from asking for
/// more memory than there is. For a larger count the estimators return `None`.
pub const MAX_RAY_SAMPLES: usize = 1 << 16;

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
    /// Minimal samples drawn (each is `EPI_SAMPLE` = 8 distinct indices), at
    /// most [`MAX_RAY_SAMPLES`].
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

/// The model of the first minimal sample in `samples` with the most inliers,
/// and that count, or `None` when no sample's model has an inlier: the sample
/// a serial loop keeping only a strictly larger count would end on.
///
/// `model` fits a sample, and `count` counts the inliers of a model with the
/// scratch space `scratch` makes. From [`PARALLEL_MIN_SAMPLES`] samples on,
/// they are scored on the rayon pool, each worker with scratch space of its
/// own; the winner is the same either way, and its model is fitted again
/// rather than carried through the reduction.
fn best_sample<const K: usize, M, S>(
    samples: &[[usize; K]],
    model: impl Fn(&[usize; K]) -> Option<M> + Sync,
    scratch: impl Fn() -> S + Sync + Send,
    count: impl Fn(&M, &mut S) -> usize + Sync,
) -> Option<(usize, M)> {
    // Of two (count, sample index) scores, the larger count, then the earlier
    // sample.
    let better = |a: (usize, usize), b: (usize, usize)| {
        if b.0 > a.0 || (b.0 == a.0 && b.1 < a.1) {
            b
        } else {
            a
        }
    };
    let score = |s: &mut S, k: usize| {
        let m = model(&samples[k])?;
        Some((count(&m, s), k))
    };
    let best = if samples.len() >= PARALLEL_MIN_SAMPLES {
        (0..samples.len())
            .into_par_iter()
            .map_init(&scratch, score)
            .flatten()
            .reduce_with(better)
    } else {
        let mut s = scratch();
        (0..samples.len())
            .filter_map(|k| score(&mut s, k))
            .reduce(better)
    };
    let (best_count, k) = best.filter(|&(c, _)| c > 0)?;
    Some((best_count, model(&samples[k])?))
}

/// Robust ray-space epipolar matrix from unit-ray correspondences.
///
/// Scores seeded minimal samples of the linear 8-point constraint by the
/// angular residual, keeps the largest consensus, then locally optimizes on
/// it. Returns `None` when no consensus reaches `min_inliers`, and for
/// `samples` above [`MAX_RAY_SAMPLES`].
///
/// `r1[i]` and `r2[i]` are one correspondence, and both must be unit length;
/// they are not normalized or checked. Slices of different lengths are read
/// to the shorter one, and the mask and residuals have that length.
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
    let (r1, r2) = (&r1[..n], &r2[..n]);
    if n < EPI_SAMPLE.max(options.min_inliers)
        || options.samples > MAX_RAY_SAMPLES
        || options.max_angle_rad <= 0.0
        || !options.max_angle_rad.is_finite()
    {
        return None;
    }
    let sin_tol = options.max_angle_rad.min(std::f64::consts::FRAC_PI_2).sin();
    let rows = epipolar_rows(r1, r2);
    let mut state = options.seed;
    let samples = draw_samples(&mut state, n, EPI_SAMPLE, options.samples.max(1));

    let best = best_sample(
        samples.as_chunks::<EPI_SAMPLE>().0,
        |s| null_from_rows(&rows, s.iter().copied()),
        || (vec![0.0f64; n], vec![0.0f64; n]),
        |e, (resid, scratch)| {
            residuals_for(e, r1, r2, options.side, scratch, resid);
            resid.iter().filter(|&&r| r < sin_tol).count()
        },
    );
    let (best_count, mut e) = best?;
    if best_count < options.min_inliers {
        return None;
    }
    let mut resid = vec![0.0f64; n];
    let mut scratch = vec![0.0f64; n];
    residuals_for(&e, r1, r2, options.side, &mut scratch, &mut resid);
    let mut inliers: Vec<bool> = resid.iter().map(|&r| r < sin_tol).collect();
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

/// The relative pose a ray-space essential matrix decomposes into, chosen by
/// cheirality.
#[derive(Clone, Debug)]
pub struct RayRelativePose {
    /// The rotation `R` of the motion `X₂ = R·X₁ + t`, where `X₁` and `X₂` are
    /// one point in the first and second camera's frames: `R·x₁` is the
    /// direction of `x₁` in the second camera's frame.
    pub rotation: Matrix3<f64>,
    /// The unit translation `t` of the same motion, its sign fixed by
    /// cheirality. Two views fix no scale.
    pub translation: Vector3<f64>,
    /// Per correspondence, whether it was considered and its point lies in
    /// front of both cameras at this pose, depth measured along each ray.
    pub in_front: Vec<bool>,
}

/// The relative pose an epipolar matrix `e` (with `x₂ᵀ·E·x₁ = 0`, as
/// [`estimate_essential_rays`] returns it) decomposes into.
///
/// The SVD of `e` gives two rotations and a translation direction up to sign,
/// so four poses; the one that puts the most of the `considered`
/// correspondences in front of both cameras is kept, a tie going to the
/// first in the fixed order `(R₁, +t)`, `(R₁, −t)`, `(R₂, +t)`, `(R₂, −t)`.
/// Depth is measured along each ray, so rays past 90° off the axis take part
/// like any other. Taking `diag(1, 1, 0)` for the singular values projects `e`
/// onto the essential manifold.
///
/// `r1[i]` and `r2[i]` are one correspondence, unit rays; the slices and
/// `considered` are read to the shortest of the three. Returns `None` when the
/// SVD fails or no considered correspondence is in front at any of the four
/// poses.
pub fn decompose_essential_rays(
    e: &Matrix3<f64>,
    r1: &[Vector3<f64>],
    r2: &[Vector3<f64>],
    considered: &[bool],
) -> Option<RayRelativePose> {
    let n = r1.len().min(r2.len()).min(considered.len());
    if !e.iter().all(|v| v.is_finite()) {
        return None;
    }
    let svd = e.svd(true, true);
    let (mut u, mut v_t) = (svd.u?, svd.v_t?);
    // The singular values come in no promised order: put the smallest's
    // vectors last, so that its left vector is the translation.
    let sv = svd.singular_values;
    let k = (0..3).fold(0, |k, j| if sv[j] < sv[k] { j } else { k });
    if k != 2 {
        u.swap_columns(k, 2);
        v_t.swap_rows(k, 2);
    }
    if u.determinant() < 0.0 {
        u = -u;
    }
    if v_t.determinant() < 0.0 {
        v_t = -v_t;
    }
    let w = Matrix3::new(0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
    let t = u.column(2).into_owned();
    let candidates = [
        (u * w * v_t, t),
        (u * w * v_t, -t),
        (u * w.transpose() * v_t, t),
        (u * w.transpose() * v_t, -t),
    ];
    let mut best: Option<(usize, RayRelativePose)> = None;
    for (rotation, translation) in candidates {
        // In camera 1's frame: camera 2's centre is −Rᵀt, its rays Rᵀ·x₂.
        let r_t = rotation.transpose();
        let centre = -(r_t * translation);
        let in_front: Vec<bool> = (0..n)
            .map(|i| considered[i] && in_front_of_both(&r1[i], &(r_t * r2[i]), &centre))
            .collect();
        let count = in_front.iter().filter(|&&b| b).count();
        if count > best.as_ref().map_or(0, |(c, _)| *c) {
            best = Some((
                count,
                RayRelativePose {
                    rotation,
                    translation,
                    in_front,
                },
            ));
        }
    }
    best.map(|(_, pose)| pose)
}

/// Settings of [`fit_ray_rotation`].
#[derive(Clone, Copy, Debug)]
pub struct RayRotationOptions {
    /// Consensus bound on the angle between a rotated ray and its partner,
    /// radians.
    pub max_angle_rad: f64,
    /// Reject a consensus below this many correspondences.
    pub min_inliers: usize,
    /// Minimal samples drawn (each is `ROT_SAMPLE` = 3 distinct indices), at
    /// most [`MAX_RAY_SAMPLES`].
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
/// reaches `min_inliers`, and for `samples` above [`MAX_RAY_SAMPLES`].
///
/// `r1[i]` and `r2[i]` are one correspondence, and both must be unit length;
/// they are not normalized or checked. Slices of different lengths are read
/// to the shorter one, and the mask and residuals have that length.
pub fn fit_ray_rotation(
    r1: &[Vector3<f64>],
    r2: &[Vector3<f64>],
    options: &RayRotationOptions,
) -> Option<RayRotation> {
    let n = r1.len().min(r2.len());
    let (r1, r2) = (&r1[..n], &r2[..n]);
    if n < ROT_SAMPLE.max(options.min_inliers)
        || options.samples > MAX_RAY_SAMPLES
        || options.max_angle_rad <= 0.0
        || !options.max_angle_rad.is_finite()
    {
        return None;
    }
    let tol = options.max_angle_rad;
    let all: Vec<usize> = (0..n).collect();
    let mut state = options.seed;
    let samples = draw_samples(&mut state, n, ROT_SAMPLE, options.samples.max(1));

    let best = best_sample(
        samples.as_chunks::<ROT_SAMPLE>().0,
        |s| kabsch(r1, r2, s),
        || vec![0.0f64; n],
        |rot, resid| {
            rotation_residuals(rot, r1, r2, &all, resid);
            resid.iter().filter(|&&r| r < tol).count()
        },
    );
    let (best_count, mut rot) = best?;
    if best_count < options.min_inliers {
        return None;
    }
    let mut resid = vec![0.0f64; n];
    rotation_residuals(&rot, r1, r2, &all, &mut resid);
    let mut inliers: Vec<bool> = resid.iter().map(|&r| r < tol).collect();
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
