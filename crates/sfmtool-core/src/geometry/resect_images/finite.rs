// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The finite path of [`super::resect_images`]: one target's pose from its
//! track pairs, its bearings and its cluster pairs.
//!
//! Every pair is scored in pixels. A pair with a finite position is scored by
//! its reprojection distance. A bearing (a track whose point is at infinity) is
//! scored by the angle between its held-out world direction, rotated into the
//! camera, and the ray the target observed it along, times the camera's own
//! focal length: the currency the rotation-only path uses. A bearing therefore
//! constrains the rotation and says nothing about the translation.
//!
//! RANSAC draws its minimal P3P samples from the **tracks** when the target has
//! at least three finite track pairs, and from every finite pair (tracks and
//! clusters) when it has fewer. Every pair counts toward each hypothesis's
//! inlier score and takes part in the refinement. Drawing from the tracks means
//! each hypothesis fits three of them exactly, so a pose that makes the tracks
//! outliers is never proposed, however many clusters agree with it.
//!
//! The winning hypothesis is refined by trimmed least squares on its inliers
//! (the rounds and the kept fraction of the batch-registration primitive's
//! refinement), then refit once on every pair within the inlier bound.

use nalgebra::{Matrix6, Point3, UnitQuaternion, Vector3, Vector6};

use crate::camera::report::angle_between;
use crate::camera::CameraIntrinsics;
use crate::geometry::absolute_pose::p3p_solve;
use crate::geometry::pose_refine::{
    compose_pose_jacobian, project_with_jac, quantile, INVALID_RESIDUAL,
};
use crate::numeric::splitmix64;

use super::{Pose, INLIER_PX};

/// Fewest inliers a hypothesis needs before it is refined, the
/// batch-registration primitive's own P3P consensus floor.
pub(super) const MIN_CONSENSUS: usize = 8;
/// Largest number of minimal samples drawn. A pool small enough to have at most
/// this many triples is enumerated whole instead.
const MAX_TRIALS: usize = 2000;
/// The random sampler stops once an all-inlier triple has been drawn with this
/// probability, given the best inlier rate among the sampled pairs so far.
const CONFIDENCE: f64 = 0.999;
/// Trimmed refinement: rounds, and the fraction of the inliers kept per round.
const TRIM_ROUNDS: usize = 5;
const KEEP_FRACTION: f64 = 0.6;
/// Levenberg-Marquardt iterations per refit.
const REFIT_ITERATIONS: usize = 30;
/// A final refit on the inliers runs when at least this many qualify.
const MIN_FINAL_REFIT: usize = 6;

/// Where a pair came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Source {
    /// A track whose point has a finite held-out position.
    Track,
    /// A track whose point is at infinity, with a held-out bearing.
    Bearing,
    /// A cluster of the cluster-patches file.
    Cluster,
}

/// What a pair's pixel is scored against.
#[derive(Clone, Copy, Debug)]
pub(super) enum World {
    /// A finite world position.
    Point([f64; 3]),
    /// A unit world direction: a point at infinity.
    Direction(Vector3<f64>),
}

/// One correspondence of the target: its source, the pixel the target observed
/// it at, the unit camera ray through that pixel, and what it is scored
/// against.
#[derive(Clone, Copy, Debug)]
pub(super) struct Pair {
    pub(super) source: Source,
    pub(super) uv: [f64; 2],
    pub(super) ray: Vector3<f64>,
    pub(super) world: World,
}

impl Pair {
    /// A pair observed at `uv` by `camera`.
    pub(super) fn new(
        source: Source,
        uv: [f64; 2],
        world: World,
        camera: &CameraIntrinsics,
    ) -> Option<Self> {
        let ray = camera.pixel_to_ray(uv[0], uv[1]);
        let ray = Vector3::new(ray[0], ray[1], ray[2]);
        let norm = ray.norm();
        (norm > 0.0 && norm.is_finite()).then(|| Pair {
            source,
            uv,
            ray: ray / norm,
            world,
        })
    }

    fn is_finite(&self) -> bool {
        matches!(self.world, World::Point(_))
    }
}

/// Pixels one radian of angle is worth on `camera`: its larger focal length,
/// at least one.
pub(super) fn pixels_per_radian(camera: &CameraIntrinsics) -> f64 {
    let (fx, fy) = camera.focal_lengths();
    fx.max(fy).max(1.0)
}

/// A fitted pose and which pairs it puts within [`INLIER_PX`].
pub(super) struct Fit {
    pub(super) pose: Pose,
    pub(super) inliers: Vec<bool>,
}

/// Estimate one target's pose from `pairs`. `Err` carries the refusal.
///
/// The minimal samples come from the finite track pairs when there are at least
/// three, and from every finite pair otherwise. Deterministic for a given
/// `seed`.
pub(super) fn estimate(
    camera: &CameraIntrinsics,
    pairs: &[Pair],
    seed: u64,
) -> Result<Fit, String> {
    let ppr = pixels_per_radian(camera);
    let tracks: Vec<usize> = (0..pairs.len())
        .filter(|&k| pairs[k].source == Source::Track)
        .collect();
    let pool: Vec<usize> = if tracks.len() >= 3 {
        tracks
    } else {
        (0..pairs.len()).filter(|&k| pairs[k].is_finite()).collect()
    };
    if pool.len() < 3 {
        return Err(format!(
            "{} finite correspondence{}, and P3P needs 3",
            pool.len(),
            if pool.len() == 1 { "" } else { "s" }
        ));
    }

    let mut best: Option<Scored> = None;
    let try_triple = |sample: [usize; 3], best: &mut Option<Scored>| {
        let bearings = sample.map(|k| pairs[pool[k]].ray);
        let points = sample.map(|k| match pairs[pool[k]].world {
            World::Point(x) => Point3::new(x[0], x[1], x[2]),
            World::Direction(_) => unreachable!("the pool is finite pairs"),
        });
        for pose in p3p_solve(&bearings, &points) {
            let mut scored = score(camera, ppr, pairs, pose);
            if best.as_ref().is_some_and(|b| !scored.beats(b)) {
                continue;
            }
            // Refit once on the hypothesis's own inliers: a pose solved exactly
            // through three noisy pixels undercounts the pairs it explains.
            let idx: Vec<usize> = (0..pairs.len()).filter(|&k| scored.inliers[k]).collect();
            let refit = score(camera, ppr, pairs, refit(camera, ppr, pairs, &idx, pose));
            if refit.beats(&scored) {
                scored = refit;
            }
            *best = Some(scored);
        }
    };

    let n = pool.len();
    if n * (n - 1) * (n - 2) / 6 <= MAX_TRIALS {
        for a in 0..n {
            for b in a + 1..n {
                for c in b + 1..n {
                    try_triple([a, b, c], &mut best);
                }
            }
        }
    } else {
        let mut state = seed;
        let mut required = MAX_TRIALS;
        let mut trials = 0;
        while trials < required {
            trials += 1;
            let a = (splitmix64(&mut state) % n as u64) as usize;
            let mut b = (splitmix64(&mut state) % n as u64) as usize;
            while b == a {
                b = (splitmix64(&mut state) % n as u64) as usize;
            }
            let mut c = (splitmix64(&mut state) % n as u64) as usize;
            while c == a || c == b {
                c = (splitmix64(&mut state) % n as u64) as usize;
            }
            try_triple([a, b, c], &mut best);
            if let Some(leader) = &best {
                let w = pool.iter().filter(|&&k| leader.inliers[k]).count() as f64 / n as f64;
                let w3 = w * w * w;
                if w3 >= 1.0 {
                    break;
                }
                if w3 > 0.0 {
                    let needed = ((1.0 - CONFIDENCE).ln() / (1.0 - w3).ln()).ceil();
                    required = required.min(needed.max(1.0) as usize);
                }
            }
        }
    }

    let best = best.ok_or_else(|| "no minimal sample gave a pose".to_string())?;
    if best.count < MIN_CONSENSUS {
        return Err(format!(
            "the best pose hypothesis has {} inlier{} ({MIN_CONSENSUS} needed)",
            best.count,
            if best.count == 1 { "" } else { "s" }
        ));
    }

    // Trimmed refinement on the hypothesis's inliers.
    let consensus: Vec<usize> = (0..pairs.len()).filter(|&k| best.inliers[k]).collect();
    let mut pose = best.pose;
    for _ in 0..TRIM_ROUNDS {
        let residuals: Vec<f64> = consensus
            .iter()
            .map(|&k| residual_px(camera, ppr, &pairs[k], &pose))
            .collect();
        let bound = quantile(&residuals, KEEP_FRACTION);
        let keep: Vec<usize> = consensus
            .iter()
            .zip(&residuals)
            .filter(|&(_, &r)| r <= bound)
            .map(|(&k, _)| k)
            .collect();
        pose = refit(camera, ppr, pairs, &keep, pose);
    }
    let within: Vec<usize> = (0..pairs.len())
        .filter(|&k| residual_px(camera, ppr, &pairs[k], &pose) < INLIER_PX)
        .collect();
    if within.len() >= MIN_FINAL_REFIT {
        pose = refit(camera, ppr, pairs, &within, pose);
    }
    let inliers = pairs
        .iter()
        .map(|pair| residual_px(camera, ppr, pair, &pose) < INLIER_PX)
        .collect();
    Ok(Fit { pose, inliers })
}

/// A pose with its inlier mask, count and truncated residual sum.
struct Scored {
    pose: Pose,
    inliers: Vec<bool>,
    count: usize,
    /// Sum over the pairs of each residual, capped at [`INLIER_PX`]: breaks a
    /// tie in `count` toward the pose that fits its pairs more closely.
    cost: f64,
}

impl Scored {
    fn beats(&self, other: &Scored) -> bool {
        self.count > other.count || (self.count == other.count && self.cost < other.cost)
    }
}

fn score(camera: &CameraIntrinsics, ppr: f64, pairs: &[Pair], pose: Pose) -> Scored {
    let mut inliers = Vec::with_capacity(pairs.len());
    let mut cost = 0.0;
    for pair in pairs {
        let r = residual_px(camera, ppr, pair, &pose);
        inliers.push(r < INLIER_PX);
        cost += r.min(INLIER_PX);
    }
    let count = inliers.iter().filter(|&&i| i).count();
    Scored {
        pose,
        inliers,
        count,
        cost,
    }
}

/// One pair's residual in pixels under `pose`: the reprojection distance of a
/// finite position (infinite outside the camera model's domain), or the angle
/// of a bearing times `ppr`.
pub(super) fn residual_px(camera: &CameraIntrinsics, ppr: f64, pair: &Pair, pose: &Pose) -> f64 {
    let (rotation, translation) = pose;
    match pair.world {
        World::Point(x) => {
            let local = rotation * Vector3::new(x[0], x[1], x[2]) + translation;
            match camera.ray_to_pixel([local.x, local.y, local.z]) {
                Some((u, v)) => (u - pair.uv[0]).hypot(v - pair.uv[1]),
                None => f64::INFINITY,
            }
        }
        World::Direction(d) => {
            let local = rotation * d;
            ppr * angle_between(local.into(), pair.ray.into())
        }
    }
}

/// One Jacobian row of the pose fit: a residual component and its derivative
/// with respect to `[δθ, δt]`, the rotation perturbed on the left.
type Row = (f64, [f64; 6]);

/// The residual components of one pair, with their Jacobian rows.
///
/// A finite position gives the two pixel components of its reprojection error.
/// A bearing gives the three components of `ppr · (ray × R·d)`, whose length is
/// `ppr · sin(angle)`: the scored residual to first order, with a derivative
/// that stays defined at zero angle. It has no translation derivative.
fn rows(
    camera: &CameraIntrinsics,
    analytic: bool,
    ppr: f64,
    pair: &Pair,
    pose: &Pose,
    out: &mut Vec<Row>,
) {
    let (rotation, translation) = pose;
    match pair.world {
        World::Point(x) => {
            let rotated = rotation * Vector3::new(x[0], x[1], x[2]);
            match project_with_jac(camera, rotated + translation, analytic) {
                Some(((u, v), jp)) => {
                    let j = compose_pose_jacobian(&jp, &rotated);
                    out.push((u - pair.uv[0], j[0]));
                    out.push((v - pair.uv[1], j[1]));
                }
                // Outside the domain: a large residual with no derivative, so
                // it is penalized without steering the step.
                None => {
                    out.push((INVALID_RESIDUAL, [0.0; 6]));
                    out.push((0.0, [0.0; 6]));
                }
            }
        }
        World::Direction(d) => {
            let rotated = rotation * d;
            let residual = pair.ray.cross(&rotated) * ppr;
            // δ(R·d) = δθ × R·d, so ∂(ray × R·d)/∂δθ = −[ray]ₓ [R·d]ₓ.
            let j = -(pair.ray.cross_matrix() * rotated.cross_matrix()) * ppr;
            for a in 0..3 {
                out.push((
                    residual[a],
                    [j[(a, 0)], j[(a, 1)], j[(a, 2)], 0.0, 0.0, 0.0],
                ));
            }
        }
    }
}

fn cost(camera: &CameraIntrinsics, ppr: f64, pairs: &[Pair], idx: &[usize], pose: &Pose) -> f64 {
    let analytic = camera.model.supports_pixel_jacobian();
    let mut buf = Vec::new();
    for &k in idx {
        rows(camera, analytic, ppr, &pairs[k], pose, &mut buf);
    }
    buf.iter().map(|(r, _)| r * r).sum()
}

/// Levenberg-Marquardt over the six pose parameters on the pairs `idx`,
/// starting from `pose`. Returns `pose` unchanged when fewer than three pairs
/// are given.
fn refit(camera: &CameraIntrinsics, ppr: f64, pairs: &[Pair], idx: &[usize], pose: Pose) -> Pose {
    if idx.len() < 3 {
        return pose;
    }
    let analytic = camera.model.supports_pixel_jacobian();
    let (mut rotation, mut translation) = pose;
    let mut lambda = 1e-3;
    let mut previous = cost(camera, ppr, pairs, idx, &(rotation, translation));
    let mut buf: Vec<Row> = Vec::new();
    for _ in 0..REFIT_ITERATIONS {
        buf.clear();
        for &k in idx {
            rows(
                camera,
                analytic,
                ppr,
                &pairs[k],
                &(rotation, translation),
                &mut buf,
            );
        }
        let mut jtj = Matrix6::<f64>::zeros();
        let mut jtr = Vector6::<f64>::zeros();
        for (r, j) in &buf {
            let j = Vector6::from_row_slice(j);
            jtr += j * *r;
            jtj += j * j.transpose();
        }
        let mut improved = false;
        for _ in 0..12 {
            let mut damped = jtj;
            for d in 0..6 {
                damped[(d, d)] += lambda * jtj[(d, d)].max(1e-12);
            }
            let Some(delta) = damped.lu().solve(&(-jtr)) else {
                lambda *= 4.0;
                continue;
            };
            let r_candidate =
                UnitQuaternion::from_scaled_axis(Vector3::new(delta[0], delta[1], delta[2]))
                    * rotation;
            let t_candidate = translation + Vector3::new(delta[3], delta[4], delta[5]);
            let next = cost(camera, ppr, pairs, idx, &(r_candidate, t_candidate));
            if next < previous {
                rotation = r_candidate;
                translation = t_candidate;
                previous = next;
                lambda = (lambda * 0.5).max(1e-12);
                improved = true;
                break;
            }
            lambda *= 4.0;
            if lambda > 1e12 {
                break;
            }
        }
        if !improved {
            break;
        }
    }
    (rotation, translation)
}
