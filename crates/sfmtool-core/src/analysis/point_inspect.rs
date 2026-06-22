// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Per-point triangulation inspection.
//!
//! Re-derives a 3D point's observation rays from the workspace `.sift` files
//! (un-projecting each member keypoint), runs the same triangulation and
//! classification the discovery/reclassify paths use, and reports the full set
//! of diagnostics behind the finite / at-infinity / indeterminate call. Backs
//! `sfm inspect pt3d_*`.

use nalgebra::{Point3, Vector3};

use crate::analysis::infinity::{
    camera_extents, classify_rays_at_infinity, Classification, DEFAULT_INVERSE_DEPTH_Z_CUTOFF,
};
use crate::reconstruction::triangulation::{depth_uncertainty_batch, triangulate_batch};
use crate::reconstruction::{ReconstructionError, SfmrReconstruction};

/// One observation in a point's track, with where it lands in its image.
pub struct ObservationInspection {
    pub image_index: usize,
    pub image_name: String,
    pub feature_index: u32,
    /// Angle of the un-projected ray off the camera optical axis (degrees).
    /// Large values (→ 90°) sit near the fisheye edge, where the un-projection
    /// is least reliable.
    pub incidence_deg: f64,
}

/// Full triangulation analysis of one 3D point.
pub struct PointInspection {
    /// Stored homogeneous kind: `1.0` finite, `0.0` at infinity.
    pub w: f64,
    /// Stored Euclidean position (finite) or unit direction (at infinity).
    pub position: Point3<f64>,
    pub error: f32,
    pub color: [u8; 3],
    pub observations: Vec<ObservationInspection>,
    /// Classification re-derived from the rays (matches the production gate).
    pub classification: Classification,
    /// Least-squares point from the re-derived rays.
    pub triangulated_point: Point3<f64>,
    pub eigenvalues: [f64; 3],
    pub condition_number: f64,
    pub in_front: bool,
    pub depth: f64,
    pub sigma: f64,
    pub inverse_depth_z: f64,
    pub resolvable_distance: f64,
    /// Camera extents — the default `finite_horizon` the gate compares against.
    pub finite_horizon: f64,
    /// Bounding-box diagonal of the *observing* camera centers.
    pub baseline_span: f64,
    /// Largest angle (degrees) of any observation ray to the mean direction.
    pub max_ray_angle_deg: f64,
}

impl SfmrReconstruction {
    /// Inspect a single 3D point's triangulation. Reads the `.sift` files of the
    /// observing images, so the workspace feature files must be present.
    ///
    /// `point_idx` must be in range; callers should validate it first.
    pub fn inspect_point(
        &self,
        point_idx: usize,
        noise_floor_px: f64,
    ) -> Result<PointInspection, ReconstructionError> {
        let pt = &self.points[point_idx];
        // Inspection reads `.sift` features, so it requires a sift_files recon.
        let feature_indexes =
            self.feature_indexes()
                .ok_or_else(|| ReconstructionError::SiftRead {
                    path: self.workspace_dir.clone(),
                    source: "inspect_point requires a sift_files reconstruction".into(),
                })?;
        let start = self.observation_offsets[point_idx];
        let observations = self.observations_for_point(point_idx);
        let noise = (pt.error as f64).max(noise_floor_px);

        let mut dirs: Vec<Vector3<f64>> = Vec::with_capacity(observations.len());
        let mut centers: Vec<Point3<f64>> = Vec::with_capacity(observations.len());
        let mut sigma_rad: Vec<f64> = Vec::with_capacity(observations.len());
        let mut obs_out: Vec<ObservationInspection> = Vec::with_capacity(observations.len());

        for (k, obs) in observations.iter().enumerate() {
            let feature_index = feature_indexes[start + k];
            let img_idx = obs.image_index as usize;
            let image = &self.images[img_idx];
            let camera = &self.cameras[image.camera_index as usize];
            let (fx, fy) = camera.focal_lengths();

            let sift_path = self.sift_path_for_image(img_idx);
            let sift = sift_format::read_sift_partial(&sift_path, feature_index as usize + 1)
                .map_err(|e| ReconstructionError::SiftRead {
                    path: sift_path.clone(),
                    source: e.to_string(),
                })?;
            let f = feature_index as usize;
            let u = sift.positions_xy[[f, 0]] as f64;
            let v = sift.positions_xy[[f, 1]] as f64;

            let ray_cam = camera.pixel_to_ray(u, v);
            let ray_cam = Vector3::new(ray_cam[0], ray_cam[1], ray_cam[2]);
            let rc_norm = ray_cam.norm();
            let incidence_deg = if rc_norm > 0.0 {
                (ray_cam.z / rc_norm).clamp(-1.0, 1.0).acos().to_degrees()
            } else {
                0.0
            };

            let world = image.quaternion_wxyz.inverse() * ray_cam;
            let wn = world.norm();
            dirs.push(if wn > 0.0 { world / wn } else { world });
            centers.push(image.camera_center());
            sigma_rad.push(noise / fx.max(fy));
            obs_out.push(ObservationInspection {
                image_index: img_idx,
                image_name: image.name.clone(),
                feature_index,
                incidence_deg,
            });
        }

        let centers_all: Vec<Point3<f64>> =
            self.images.iter().map(|im| im.camera_center()).collect();
        let finite_horizon = camera_extents(&centers_all);

        let offsets = [0usize, dirs.len()];
        let tri = triangulate_batch(&dirs, &centers, &offsets)[0];
        let du = depth_uncertainty_batch(&[tri], &dirs, &centers, &offsets, &sigma_rad)[0];
        let classification = classify_rays_at_infinity(
            &dirs,
            &centers,
            &sigma_rad,
            DEFAULT_INVERSE_DEPTH_Z_CUTOFF,
            finite_horizon,
        )
        .class;

        // Largest angle of any ray to the mean viewing direction.
        let mut mean = Vector3::zeros();
        for d in &dirs {
            mean += d;
        }
        let max_ray_angle_deg = if mean.norm() > 0.0 {
            let m = mean.normalize();
            dirs.iter()
                .map(|d| d.dot(&m).clamp(-1.0, 1.0).acos().to_degrees())
                .fold(0.0_f64, f64::max)
        } else {
            0.0
        };

        Ok(PointInspection {
            w: pt.w,
            position: pt.position,
            error: pt.error,
            color: pt.color,
            observations: obs_out,
            classification,
            triangulated_point: tri.point,
            eigenvalues: tri.eigenvalues,
            condition_number: tri.condition_number,
            in_front: tri.in_front_of_all_cameras,
            depth: du.depth,
            sigma: du.sigma,
            inverse_depth_z: du.inverse_depth_z,
            resolvable_distance: du.resolvable_distance,
            finite_horizon,
            baseline_span: camera_extents(&centers),
            max_ray_angle_deg,
        })
    }
}
