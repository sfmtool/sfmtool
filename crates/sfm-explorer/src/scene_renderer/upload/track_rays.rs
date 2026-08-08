// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Observation ray geometry for the selected point track.

use super::super::gpu_types::EdgeInstance;
use super::super::SceneRenderer;
use crate::scene::{ImageRef, PointRef};
use sfmtool_core::{Se3Transform, SfmrReconstruction};
use wgpu::util::DeviceExt;

/// Length of a point-at-infinity track ray, as a multiple of the camera-cloud
/// extent — long enough to clearly head out past the scene toward infinity.
const INFINITY_RAY_SCENE_MULTIPLE: f64 = 2.0;

/// Bounding-box diagonal of the reconstruction's camera centers — a
/// characteristic scene scale, used to size rays toward points at infinity.
fn camera_cloud_extent(recon: &SfmrReconstruction) -> f64 {
    let mut iter = recon.images.iter().map(|im| im.camera_center().coords);
    let Some(first) = iter.next() else {
        return 0.0;
    };
    let (mut lo, mut hi) = (first, first);
    for c in iter {
        lo = lo.inf(&c);
        hi = hi.sup(&c);
    }
    (hi - lo).norm()
}

/// Build the observation-ray geometry for `point_idx`.
///
/// Each ray goes from the camera center along the true observation direction
/// (unprojected from the SIFT feature position through camera intrinsics) to
/// the nearest point on the ray to the 3D point. The gap between the ray
/// endpoint and the 3D point visualizes reprojection error in 3D space.
///
/// `sift_cache` is the shared SIFT position cache from `AppState`. Feature
/// positions are looked up from this cache (the caller must ensure that
/// relevant images have been cached via `ensure_sift_cached` before calling).
///
/// Split out from the upload so the geometry is assertable without a GPU
/// buffer readback — the edges land in a VERTEX-usage buffer, which is
/// write-only from the CPU side.
///
/// The rays are built in the reconstruction's own coordinates and then put
/// through `transform`, the owning node's similarity: the track-ray pass is one
/// of the CPU world-space paths, drawn from a shared singleton buffer with no
/// per-recon `model` matrix of its own. Because a similarity is affine, mapping
/// the two endpoints is the same as mapping the whole segment.
pub(super) fn track_ray_edges(
    recon: &SfmrReconstruction,
    point_ref: PointRef,
    sift_cache: &std::collections::HashMap<ImageRef, crate::state::CachedSiftFeatures>,
    transform: &Se3Transform,
) -> Vec<EdgeInstance> {
    let point_idx = point_ref.index();
    let point = &recon.points[point_idx];
    let point_pos = point.position;
    let at_infinity = point.is_at_infinity();

    // A point at infinity has no finite location — its stored position is a
    // unit direction at the origin, which would project onto every forward
    // ray at t < 0 and collapse to a zero-length (invisible) ray. Instead,
    // shoot each ray outward along its own bearing to a fixed, scene-scaled
    // length (a multiple of the camera-cloud extent) so the bundle is
    // visible heading off toward infinity.
    let infinity_ray_length = if at_infinity {
        INFINITY_RAY_SCENE_MULTIPLE * camera_cloud_extent(recon)
    } else {
        0.0
    };

    // The observation keypoint lives in one of two places: SIFT feature
    // positions read from `.sift` companions (`sift_files`, indexed through
    // `feature_indexes` into the shared cache) or keypoints stored inline in
    // the reconstruction (`embedded_patches`). Both are photometrically
    // placed and need not point exactly at the 3D point, so the ray is
    // unprojected from whichever the reconstruction carries.
    let feature_indexes = recon.feature_indexes();
    let keypoints_xy = recon.keypoints_xy();
    let obs_start = recon.observation_offsets[point_idx];
    let observations = recon.observations_for_point(point_idx);
    let edges: Vec<EdgeInstance> = observations
        .iter()
        .enumerate()
        .filter_map(|(k, obs)| {
            let image = &recon.images[obs.image_index as usize];
            let camera = &recon.cameras[image.camera_index as usize];
            let center = image.camera_center();
            let endpoint_a = [center.x as f32, center.y as f32, center.z as f32];

            // The observed keypoint pixel for this observation, from the
            // SIFT feature or the inline embedded keypoint. Skip the
            // observation when neither source yields one (e.g. a missing or
            // truncated `.sift` file) rather than drawing a misleading ray.
            let obs_pixel: [f64; 2] = if let Some(fis) = feature_indexes {
                let fi = fis[obs_start + k] as usize;
                let cached =
                    sift_cache.get(&ImageRef::new(point_ref.recon, obs.image_index as usize))?;
                let xy = cached.positions_xy.get(fi)?;
                [xy[0] as f64, xy[1] as f64]
            } else {
                let kxy = keypoints_xy?;
                let row = obs_start + k;
                [kxy[[row, 0]] as f64, kxy[[row, 1]] as f64]
            };

            // Unproject the keypoint to a camera-local unit ray, then rotate
            // to world space: d_world = R^T * d_cam.
            let d_cam = camera.pixel_to_ray(obs_pixel[0], obs_pixel[1]);
            let r_flat = image.camera_to_world_rotation_flat();
            let d_world = [
                r_flat[0] * d_cam[0] + r_flat[1] * d_cam[1] + r_flat[2] * d_cam[2],
                r_flat[3] * d_cam[0] + r_flat[4] * d_cam[1] + r_flat[5] * d_cam[2],
                r_flat[6] * d_cam[0] + r_flat[7] * d_cam[1] + r_flat[8] * d_cam[2],
            ];

            let endpoint_b = if at_infinity {
                // Point at infinity: shoot the ray outward along the
                // observed bearing (a point at infinity has no parallax).
                [
                    (center.x + infinity_ray_length * d_world[0]) as f32,
                    (center.y + infinity_ray_length * d_world[1]) as f32,
                    (center.z + infinity_ray_length * d_world[2]) as f32,
                ]
            } else {
                // Finite point: terminate at the nearest point on the
                // observed ray (so reprojection error shows),
                // t = dot(P - C, d_world) clamped to the forward direction.
                let cp = [
                    point_pos.x - center.x,
                    point_pos.y - center.y,
                    point_pos.z - center.z,
                ];
                let t = (cp[0] * d_world[0] + cp[1] * d_world[1] + cp[2] * d_world[2]).max(0.0);
                [
                    (center.x + t * d_world[0]) as f32,
                    (center.y + t * d_world[1]) as f32,
                    (center.z + t * d_world[2]) as f32,
                ]
            };

            Some(EdgeInstance {
                endpoint_a: to_world(transform, endpoint_a),
                endpoint_b: to_world(transform, endpoint_b),
            })
        })
        .collect();
    edges
}

/// One ray endpoint through the node transform.
fn to_world(transform: &Se3Transform, p: [f32; 3]) -> [f32; 3] {
    let world = transform.apply_to_point(&nalgebra::Point3::new(
        p[0] as f64,
        p[1] as f64,
        p[2] as f64,
    ));
    [world.x as f32, world.y as f32, world.z as f32]
}

impl SceneRenderer {
    /// Upload track ray edge geometry for the selected point's observations.
    ///
    /// See [`track_ray_edges`] for how each ray is built.
    pub fn upload_track_rays(
        &mut self,
        device: &wgpu::Device,
        recon: &SfmrReconstruction,
        point: PointRef,
        sift_cache: &std::collections::HashMap<ImageRef, crate::state::CachedSiftFeatures>,
        transform: &Se3Transform,
    ) {
        let edges = track_ray_edges(recon, point, sift_cache, transform);

        if edges.is_empty() {
            self.track_ray_edge_buffer = None;
            self.track_ray_count = 0;
            return;
        }

        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("track ray edges"),
            contents: bytemuck::cast_slice(&edges),
            usage: wgpu::BufferUsages::VERTEX,
        });
        self.track_ray_edge_buffer = Some(buffer);
        self.track_ray_count = edges.len() as u32;
    }

    /// Clear track ray geometry (no point selected).
    pub fn clear_track_rays(&mut self) {
        self.track_ray_edge_buffer = None;
        self.track_ray_count = 0;
    }
}
