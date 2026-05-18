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

use crate::reconstruction::SfmrReconstruction;
use crate::viewing_angle::{max_viewing_angle, viewing_rays};

/// SIFT keypoint localisation noise floor (pixels).
///
/// The classifier estimates a track's measurement noise from its reprojection
/// error but never lets it fall below this floor: a short track is
/// triangulated to fit its few observations almost exactly regardless of depth
/// conditioning, so its reprojection error under-states the true noise.
pub const DEFAULT_NOISE_FLOOR_PX: f64 = 1.0;

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
    /// A finite point becomes `w = 0` when its parallax signal in pixels
    /// (`α_max · f_max`, the largest viewing angle times the largest observing
    /// focal length) falls below the track's measurement noise,
    /// `max(reprojection_error, noise_floor_px)`. Its coordinate is replaced
    /// with the bearing-mean direction `normalise(Σ rᵢ)` of its observation
    /// rays. Points already at infinity, and points with fewer than two
    /// observations, are left unchanged.
    pub fn classify_points_at_infinity(&self, noise_floor_px: f64) -> Self {
        let centers: Vec<Point3<f64>> = self.images.iter().map(|im| im.camera_center()).collect();
        let focal_max = self.per_image_focal_max();

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
            // paired with their image index.
            let rays = viewing_rays(
                pt.position,
                &centers,
                obs.iter().map(|o| o.image_index as usize),
            );
            if rays.len() < 2 {
                continue;
            }

            // Parallax signal = max viewing angle x largest focal length, both
            // taken over the cameras whose rays survived: a camera coincident
            // with the point contributes neither a ray nor a focal length.
            let alpha_max = max_viewing_angle(&rays);
            let f_max = rays
                .iter()
                .map(|(img, _)| focal_max[*img])
                .fold(0.0_f64, f64::max);

            let parallax_px = alpha_max * f_max;
            let noise = (pt.error as f64).max(noise_floor_px);
            if parallax_px >= noise {
                continue;
            }

            // Depth is lost in noise: store the bearing-mean direction.
            let mut sum = Vector3::zeros();
            for (_, r) in &rays {
                sum += r;
            }
            let norm = sum.norm();
            if norm > 0.0 {
                pt.position = Point3::from(sum / norm);
                pt.w = 0.0;
                pt.estimated_normal = Vector3::zeros();
            }
            // norm == 0 (rays exactly cancel) is degenerate — leave finite.
        }

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

        recon
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_leaves_well_conditioned_points_finite() {
        // demo() points sit ~1 unit from cameras ~5 units away: wide parallax.
        let recon = SfmrReconstruction::demo(100);
        let classified = recon.classify_points_at_infinity(DEFAULT_NOISE_FLOOR_PX);
        let n_inf = classified.points.iter().filter(|p| p.w == 0.0).count();
        assert_eq!(n_inf, 0, "well-conditioned demo points must stay finite");
    }

    #[test]
    fn classify_detects_far_point_as_infinity() {
        // demo(1): point 0 observed by images 0 and 1. Pushing it far away
        // collapses the parallax below the noise floor.
        let mut recon = SfmrReconstruction::demo(1);
        recon.points[0].position = Point3::new(0.0, 0.0, 10_000.0);
        recon.points[0].error = 0.5;

        let classified = recon.classify_points_at_infinity(DEFAULT_NOISE_FLOOR_PX);
        assert!(classified.points[0].is_at_infinity());
        // The stored direction is a unit vector.
        assert!((classified.points[0].position.coords.norm() - 1.0).abs() < 1e-9);
        // estimated_normal is zeroed for a point at infinity.
        assert_eq!(classified.points[0].estimated_normal, Vector3::zeros());
    }

    #[test]
    fn classify_uses_per_point_reprojection_error() {
        // A point whose parallax signal sits between the noise floor and its
        // own reprojection error: the fixed floor would keep it finite, the
        // per-point error reclassifies it.
        let mut recon = SfmrReconstruction::demo(1);
        // Distance tuned so parallax (α_max · f_max) lands near ~2 px.
        recon.points[0].position = Point3::new(0.0, 0.0, 1_900.0);

        recon.points[0].error = 0.5;
        let with_floor = recon.classify_points_at_infinity(DEFAULT_NOISE_FLOOR_PX);
        assert!(
            !with_floor.points[0].is_at_infinity(),
            "parallax exceeds the 1 px floor — point stays finite"
        );

        recon.points[0].error = 5.0;
        let with_error = recon.classify_points_at_infinity(DEFAULT_NOISE_FLOOR_PX);
        assert!(
            with_error.points[0].is_at_infinity(),
            "parallax is below the point's own 5 px reprojection error"
        );
    }

    #[test]
    fn classify_idempotent_on_infinity_points() {
        let mut recon = SfmrReconstruction::demo(20);
        for pt in &mut recon.points {
            pt.position = Point3::from(pt.position.coords.normalize());
            pt.w = 0.0;
        }
        let classified = recon.classify_points_at_infinity(DEFAULT_NOISE_FLOOR_PX);
        assert!(classified.points.iter().all(|p| p.is_at_infinity()));
        for pt in &classified.points {
            assert!((pt.position.coords.norm() - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn materialize_makes_every_point_finite() {
        let mut recon = SfmrReconstruction::demo(50);
        for pt in &mut recon.points {
            pt.position = Point3::from(pt.position.coords.normalize());
            pt.w = 0.0;
            pt.estimated_normal = Vector3::zeros();
        }
        let materialised = recon.materialize_points_at_infinity();
        assert!(materialised.points.iter().all(|p| p.w == 1.0));
    }

    #[test]
    fn materialize_places_point_along_its_direction() {
        let mut recon = SfmrReconstruction::demo(1);
        let dir = Vector3::new(0.0, 0.0, 1.0);
        recon.points[0].position = Point3::from(dir);
        recon.points[0].w = 0.0;

        let materialised = recon.materialize_points_at_infinity();
        let pt = &materialised.points[0];
        assert_eq!(pt.w, 1.0);
        // The point lies on the ray origin + t·d, so (pt - origin) ∥ d.
        let centers: Vec<Point3<f64>> = recon.images.iter().map(|im| im.camera_center()).collect();
        let mut origin = Vector3::zeros();
        for c in &centers {
            origin += c.coords;
        }
        origin /= centers.len() as f64;
        let offset = pt.position.coords - origin;
        let perp = offset - offset.dot(&dir) * dir;
        assert!(perp.norm() < 1e-6, "materialised point must lie along d");
        assert!(offset.dot(&dir) > 0.0, "placed in the +d half-line");
    }

    #[test]
    fn materialize_renormalises_non_unit_directions() {
        // A w = 0 point whose stored direction drifted off the unit sphere
        // must materialise to the same finite point as the unit direction:
        // the placement geometry depends only on the normalised direction.
        let mut unit = SfmrReconstruction::demo(1);
        unit.points[0].position = Point3::from(Vector3::new(0.0, 0.0, 1.0));
        unit.points[0].w = 0.0;

        let mut scaled = SfmrReconstruction::demo(1);
        scaled.points[0].position = Point3::from(Vector3::new(0.0, 0.0, 7.5));
        scaled.points[0].w = 0.0;

        let from_unit = unit.materialize_points_at_infinity();
        let from_scaled = scaled.materialize_points_at_infinity();
        let delta =
            (from_unit.points[0].position.coords - from_scaled.points[0].position.coords).norm();
        assert!(
            delta < 1e-9,
            "non-unit direction must materialise identically"
        );
    }

    #[test]
    fn materialize_leaves_zero_direction_point_untouched() {
        // A malformed w = 0 point with a zero-length direction cannot be
        // placed — it is left at w = 0 rather than producing a bogus finite
        // coordinate.
        let mut recon = SfmrReconstruction::demo(1);
        recon.points[0].position = Point3::from(Vector3::zeros());
        recon.points[0].w = 0.0;

        let materialised = recon.materialize_points_at_infinity();
        assert!(materialised.points[0].is_at_infinity());
    }

    #[test]
    fn materialize_is_inverse_free_but_round_trips_classification() {
        // A genuine infinity point, materialised then reclassified, returns to
        // infinity: its parallax is still nil.
        let mut recon = SfmrReconstruction::demo(1);
        recon.points[0].position = Point3::from(Vector3::new(0.0, 0.0, 1.0));
        recon.points[0].w = 0.0;
        recon.points[0].error = 0.5;

        let materialised = recon.materialize_points_at_infinity();
        assert!(!materialised.points[0].is_at_infinity());

        let reclassified = materialised.classify_points_at_infinity(DEFAULT_NOISE_FLOOR_PX);
        assert!(reclassified.points[0].is_at_infinity());
    }
}
