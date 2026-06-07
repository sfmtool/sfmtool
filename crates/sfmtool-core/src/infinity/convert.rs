// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Finite ↔ infinity point conversions for [`SfmrReconstruction`].
//!
//! A point at infinity (`w = 0`) is a feature track whose observation rays are
//! parallel to within measurement noise — distant content whose depth the SfM
//! solve cannot pin down. The two conversions here move points across that
//! boundary; see `specs/drafts/sfmr-v2-points-at-infinity.md` for the theory.
//!
//! The conversions are not inverses: finite → infinity drops a depth the data
//! never determined, while infinity → finite must *supply* a depth the data
//! cannot give.

use nalgebra::{Point3, Vector3};

use crate::reconstruction::{count_points_at_infinity, SfmrReconstruction};
use crate::triangulation::{depth_uncertainty_batch, triangulate_batch};
use crate::viewing_angle::viewing_rays;

/// SIFT keypoint localisation noise floor (pixels).
///
/// The classifier estimates a track's measurement noise from its reprojection
/// error but never lets it fall below this floor: a short track is
/// triangulated to fit its few observations almost exactly regardless of depth
/// conditioning, so its reprojection error under-states the true noise.
pub const DEFAULT_NOISE_FLOOR_PX: f64 = 1.0;

/// Provisional inverse-depth z-score cutoff: a track whose `depth / σ_depth`
/// falls below this is statistically indistinguishable from infinity and is
/// classified as a `w = 0` point. (KerryPark360 populations: genuine z ≈ 62 vs
/// discovered z ≈ 3.) The scale-free z-score is the decision variable; final
/// calibration on larger captures is deferred — see the spec's open questions.
pub const DEFAULT_INVERSE_DEPTH_Z_CUTOFF: f64 = 4.0;

/// Cheap geometric pre-filter on the condition number of the normal matrix `A`.
/// A track this well-conditioned has an observable depth and is finite without
/// computing the noise-calibrated z-score; the z-score is only consulted in the
/// ill-conditioned regime above this. (KerryPark360 medians: genuine 82 vs
/// degenerate 89,599.) Note the condition number scales with track length, so
/// it is a pre-filter, not the decision variable.
pub const CONDITION_NUMBER_PREFILTER: f64 = 1e4;

/// What a track's rays resolve to.
#[derive(Debug, Clone, Copy)]
pub enum Classification {
    /// Triangulated finite point.
    Finite(Point3<f64>),
    /// Point at infinity — a unit bearing direction.
    Infinity(Point3<f64>),
    /// The baseline could not place a point even at `finite_horizon`, so neither
    /// finite nor infinity is earned (see [`classify_rays_at_infinity`]).
    Indeterminate,
}

/// A track's classification plus the diagnostics behind it, kept for debug
/// review of the points that get dropped.
#[derive(Debug, Clone, Copy)]
pub struct RayClassification {
    pub class: Classification,
    pub condition_number: f64,
    pub resolvable_distance: f64,
    pub inverse_depth_z: f64,
    pub bearing: Point3<f64>,
    pub num_views: usize,
}

/// Spatial extent (bounding-box diagonal) of a set of camera centers — the
/// scale of the region a capture explored, and the default `finite_horizon`.
pub(crate) fn camera_extents(centers: &[Point3<f64>]) -> f64 {
    let Some(first) = centers.first() else {
        return 0.0;
    };
    let mut lo = first.coords;
    let mut hi = first.coords;
    for c in centers {
        lo = lo.inf(&c.coords);
        hi = hi.sup(&c.coords);
    }
    (hi - lo).norm()
}

/// Classify one track from its observation rays into finite / at-infinity /
/// indeterminate.
///
/// `dirs` are the unit world-space rays (at least one), `centers` the matching
/// camera centers, and `sigma_rad` the per-ray angular noise (`noise_px / fᵢ`).
/// The decision:
///
/// - A clearly well-conditioned, in-front solve (condition number below
///   [`CONDITION_NUMBER_PREFILTER`]) is **finite** — no noise model needed.
/// - Otherwise, if the geometry cannot resolve a point even at `finite_horizon`
///   (`resolvable_distance < finite_horizon`), the call is **indeterminate**:
///   the baseline is too small to tell a scene-scale finite point from infinity.
/// - With adequate baseline, a degenerate/behind solve or an inverse-depth
///   z-score below `z_cutoff` is **at infinity** (a `w = 0` bearing direction —
///   the mean of the rays, or the first ray if they cancel exactly); else
///   **finite**.
pub(crate) fn classify_rays_at_infinity(
    dirs: &[Vector3<f64>],
    centers: &[Point3<f64>],
    sigma_rad: &[f64],
    z_cutoff: f64,
    finite_horizon: f64,
) -> RayClassification {
    let offsets = [0usize, dirs.len()];
    let tri = triangulate_batch(dirs, centers, &offsets)
        .pop()
        .expect("one track");

    // The direction to store for a w = 0 point: the bearing mean of the rays,
    // or the first ray if they cancel exactly (degenerate; unreachable for
    // genuine near-parallel infinity tracks, whose rays sum to ≈ K·d ≠ 0).
    let bearing = {
        let mut sum = Vector3::zeros();
        for d in dirs {
            sum += d;
        }
        let norm = sum.norm();
        if norm > 0.0 {
            Point3::from(sum / norm)
        } else {
            Point3::from(dirs[0])
        }
    };
    let num_views = dirs.len();

    // A rank-deficient solve (parallel rays → infinite condition number), a
    // point behind a camera, or a non-finite point cannot be a physical finite
    // point.
    let geometrically_finite = tri.in_front_of_all_cameras
        && tri.condition_number.is_finite()
        && tri.point.coords.iter().all(|c| c.is_finite());

    // Cheap pre-filter: a well-conditioned, in-front depth is finite without the
    // noise-calibrated test (and has ample baseline, so never indeterminate).
    if geometrically_finite && tri.condition_number < CONDITION_NUMBER_PREFILTER {
        return RayClassification {
            class: Classification::Finite(tri.point),
            condition_number: tri.condition_number,
            resolvable_distance: f64::NAN,
            inverse_depth_z: f64::NAN,
            bearing,
            num_views,
        };
    }

    // Degenerate / near-parallel / behind regime: needs the noise-calibrated
    // diagnostics. `resolvable_distance` is depth-independent, so it stays valid
    // even when the solved point (and hence `inverse_depth_z`) is noise.
    let du = depth_uncertainty_batch(&[tri], dirs, centers, &offsets, sigma_rad)
        .pop()
        .expect("one track");
    let class = if du.resolvable_distance < finite_horizon {
        // Baseline can't reach the required distance — can't adjudicate.
        Classification::Indeterminate
    } else if !geometrically_finite || du.inverse_depth_z < z_cutoff {
        Classification::Infinity(bearing)
    } else {
        Classification::Finite(tri.point)
    };
    RayClassification {
        class,
        condition_number: tri.condition_number,
        resolvable_distance: du.resolvable_distance,
        inverse_depth_z: du.inverse_depth_z,
        bearing,
        num_views,
    }
}

impl SfmrReconstruction {
    /// Largest focal length (pixels) for each image's camera.
    fn per_image_focal_max(&self) -> Vec<f64> {
        self.images
            .iter()
            .map(|im| {
                let (fx, fy) = self.cameras[im.camera_index as usize].focal_lengths();
                fx.max(fy)
            })
            .collect()
    }

    /// Reclassify finite points whose depth is unconstrained as points at
    /// infinity, returning a new reconstruction.
    ///
    /// A finite point becomes `w = 0` only on a **confident** infinity call from
    /// the triangulation of its observation rays — adequate baseline (resolvable
    /// distance ≥ the camera extents) and a degenerate/behind solve or an
    /// inverse-depth z-score below [`DEFAULT_INVERSE_DEPTH_Z_CUTOFF`]. Its
    /// coordinate is replaced with the bearing-mean direction of its rays. The
    /// per-ray angular noise is `max(reprojection_error, noise_floor_px) / fᵢ`.
    ///
    /// This is non-destructive and relabel-only: a point we lack the baseline to
    /// adjudicate (indeterminate) is left as the finite point the solve
    /// produced, never removed. Points already at infinity, and points with
    /// fewer than two observations, are left unchanged.
    pub fn classify_points_at_infinity(&self, noise_floor_px: f64) -> Self {
        let centers: Vec<Point3<f64>> = self.images.iter().map(|im| im.camera_center()).collect();
        let focal_max = self.per_image_focal_max();
        let finite_horizon = camera_extents(&centers);

        let mut recon = self.clone();
        for (pidx, pt) in recon.points.iter_mut().enumerate() {
            if pt.is_at_infinity() {
                continue;
            }
            let obs = self.observations_for_point(pidx);
            if obs.len() < 2 {
                continue;
            }

            // World-frame rays from each observing camera toward the point,
            // paired with their image index. A camera coincident with the point
            // contributes no ray.
            let rays = viewing_rays(
                pt.position,
                &centers,
                obs.iter().map(|o| o.image_index as usize),
            );
            if rays.len() < 2 {
                continue;
            }

            // Per-ray angular noise from the track's measurement noise (its
            // reprojection error, floored) and each observing camera's focal
            // length.
            let noise = (pt.error as f64).max(noise_floor_px);
            let dirs: Vec<Vector3<f64>> = rays.iter().map(|(_, r)| *r).collect();
            let ray_centers: Vec<Point3<f64>> = rays.iter().map(|(img, _)| centers[*img]).collect();
            let sigma_rad: Vec<f64> = rays
                .iter()
                .map(|(img, _)| noise / focal_max[*img])
                .collect();

            let rc = classify_rays_at_infinity(
                &dirs,
                &ray_centers,
                &sigma_rad,
                DEFAULT_INVERSE_DEPTH_Z_CUTOFF,
                finite_horizon,
            );
            // Relabel-only: demote on a confident infinity call; leave finite and
            // indeterminate points exactly as the solve produced them.
            if let Classification::Infinity(direction) = rc.class {
                pt.position = direction;
                pt.w = 0.0;
                pt.estimated_normal = Vector3::zeros();
            }
        }

        recon.infinity_point_count = count_points_at_infinity(&recon.points);
        recon
    }

    /// Materialise every point at infinity as a finite point, returning a new
    /// reconstruction.
    ///
    /// A `w = 0` point has no depth to recover, so this does not triangulate.
    /// It places the point along its stored direction, at the camera-cloud
    /// centroid plus a distance `t · d`. `t` is the largest per-camera
    /// distance beyond which the materialised point's parallax falls below one
    /// pixel (`fᵢ · r⊥ᵢ`, the focal length times the camera-to-origin offset
    /// perpendicular to the direction) — far enough to be faithful in every
    /// camera, no farther. Finite points are left unchanged, as is any
    /// malformed `w = 0` point whose stored direction has zero length.
    ///
    /// The result exists for consumers that cannot represent `w = 0` (COLMAP
    /// export, a finite-only solver). It is not the inverse of
    /// [`Self::classify_points_at_infinity`].
    pub fn materialize_points_at_infinity(&self) -> Self {
        if self.images.is_empty() {
            return self.clone();
        }
        let centers: Vec<Point3<f64>> = self.images.iter().map(|im| im.camera_center()).collect();
        let focal_max = self.per_image_focal_max();

        // Reference origin: the camera-cloud centroid.
        let mut sum = Vector3::zeros();
        for c in &centers {
            sum += c.coords;
        }
        let origin = Point3::from(sum / centers.len() as f64);

        // Fallback placement distance when the pixel-differential geometry is
        // degenerate (every camera lies on the line origin + t·d).
        let cloud_radius = centers
            .iter()
            .map(|c| (c.coords - origin.coords).norm())
            .fold(0.0_f64, f64::max)
            .max(1.0);

        let mut recon = self.clone();
        for (pidx, pt) in recon.points.iter_mut().enumerate() {
            if !pt.is_at_infinity() {
                continue;
            }
            // The stored coordinate is meant to be a unit direction;
            // renormalise defensively so the placement geometry below holds
            // even if the input drifted off the unit sphere. A zero-norm
            // direction is malformed — leave that point untouched.
            let d = pt.position.coords;
            let d_norm = d.norm();
            if d_norm == 0.0 {
                continue;
            }
            let d = d / d_norm;

            let mut t = 0.0_f64;
            for o in self.observations_for_point(pidx) {
                let img = o.image_index as usize;
                let r = origin.coords - centers[img].coords;
                let r_perp = (r - r.dot(&d) * d).norm();
                t = t.max(focal_max[img] * r_perp);
            }
            if !t.is_finite() || t <= 0.0 {
                t = cloud_radius;
            }

            pt.position = Point3::from(origin.coords + t * d);
            pt.w = 1.0;
        }

        recon.infinity_point_count = count_points_at_infinity(&recon.points);
        recon
    }
}

#[cfg(test)]
mod tests;
