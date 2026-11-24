// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The track rays: the producer of the overlay ribbon store that draws the
//! observation rays of the selected point.
//!
//! `specs/gui/viewer-3d-bench-layer.md` § "The overlay ribbon store" and
//! `specs/gui/architecture.md` § "Track Ray Visualization" describe them.

use super::super::gpu_types::OverlayInstance;
use super::super::overlay_ribbons::{OverlayTarget, ProducerId};
use super::super::SceneRenderer;
use crate::scene::{ImageRef, PointRef};
use sfmtool_core::{EditedReconstruction, Se3Transform};

/// The rays' orange, in linear light, the colour space of the scene target
/// they are drawn into.
///
/// The orange of the observing frustums and of the Image Browser's track
/// highlights. Its alpha is below 1 so that where the rays of a track meet,
/// near their point, they build up, and a ray over a bright point cloud does
/// not hide the points under it.
pub(super) const TRACK_RAY_COLOR: [f32; 4] = [1.0, 0.647, 0.0, 0.6];

/// Half a track ray's width, in panel points: a stroke 1.5 points wide.
///
/// With the point of anti-aliasing the shader adds, a ray's coverage is full
/// over its middle half point and falls to nothing 1.25 points from its
/// middle. At 100% scale that is about the profile of the 1.5-pixel glow the
/// rays were drawn with before they joined the store, and at
/// [`TRACK_RAY_COLOR`]'s alpha it puts about as much orange on screen across
/// a ray.
pub(super) const TRACK_RAY_HALF_WIDTH: f32 = 0.75;

/// The rays of `point_ref` as overlay segments, one per observation whose
/// keypoint can be read.
///
/// Each ray leaves the camera centre (`a`, `w = 1`) along the observation
/// direction: the keypoint unprojected through the image's own intrinsics and
/// rotated into the world. Where it ends (`b`) depends on the point:
/// - **A finite point:** the nearest point on the ray to the 3D point, so the
///   gap between the ray's end and the point shows the reprojection error.
/// - **A point at infinity:** the observation direction itself, with `w = 0`.
///   The shader projects it to its vanishing point, where the point's own
///   splat is drawn, and pins it at the depth points at infinity are drawn
///   at, so the ray reaches the horizon and is behind every finite thing.
///
/// Each is a segment in [`TRACK_RAY_COLOR`], [`TRACK_RAY_HALF_WIDTH`] points
/// to either side, held at `BEHIND_OPACITY` where the scene hides it and not
/// pulled toward the eye.
///
/// `sift_cache` is the shared SIFT position cache from `AppState`; the caller
/// has cached the track's images with `ensure_sift_cached`. An observation
/// whose keypoint is not there (a missing or truncated `.sift` file) draws no
/// ray rather than a misleading one.
///
/// The rays are built in the reconstruction's own coordinates and put through
/// `transform`, the owning node's similarity, since the store has no per-node
/// model matrix: a place through the whole similarity, a direction through
/// its rotation only, as a point at infinity is drawn.
pub(super) fn track_ray_instances(
    edited: &EditedReconstruction,
    point_ref: PointRef,
    sift_cache: &std::collections::HashMap<ImageRef, crate::state::CachedSiftFeatures>,
    transform: &Se3Transform,
) -> Vec<OverlayInstance> {
    // The point and its track are the overlay's, so a modified or added point
    // draws the rays it holds now; the images and cameras are the base's.
    let recon = &*edited.base;
    let point_idx = point_ref.index();
    let Some(view) = edited.point(point_idx as u32) else {
        return Vec::new();
    };
    let point = view.point();
    let point_pos = point.position;
    let at_infinity = point.is_at_infinity();

    // The observation keypoint lives in one of two places: SIFT feature
    // positions read from `.sift` companions (`sift_files`, indexed through
    // `feature_indexes` into the shared cache) or keypoints stored inline in
    // the reconstruction (`embedded_patches`). Both are photometrically
    // placed and need not point exactly at the 3D point, so the ray is
    // unprojected from whichever the reconstruction carries.
    let feature_indexes = view.feature_indexes();
    let observations = view.observations();
    observations
        .iter()
        .enumerate()
        .filter_map(|(k, obs)| {
            let image = &recon.image_set.images[obs.image_index as usize];
            // The ray leaves through the image's own intrinsics. Built when the
            // selected point changes, for the one point's observations, so an
            // adjusted image's intrinsics are built once per observation here
            // rather than once per frame.
            let camera = recon.image_set.image_intrinsics(obs.image_index as usize);
            let center = image.camera_center();

            let obs_pixel: [f64; 2] = if let Some(fis) = feature_indexes {
                let fi = fis[k] as usize;
                let cached =
                    sift_cache.get(&ImageRef::new(point_ref.recon, obs.image_index as usize))?;
                let xy = cached.positions_xy.get(fi)?;
                [xy[0] as f64, xy[1] as f64]
            } else {
                let xy = view.keypoint_xy(k)?;
                [xy[0] as f64, xy[1] as f64]
            };

            // Unproject the keypoint to a camera-local unit ray, then rotate
            // to world space: d_world = R^T * d_cam.
            let d_cam = camera.pixel_to_ray(obs_pixel[0], obs_pixel[1]);
            let r_flat = image.camera_to_world_rotation_flat();
            let d_world = nalgebra::Vector3::new(
                r_flat[0] * d_cam[0] + r_flat[1] * d_cam[1] + r_flat[2] * d_cam[2],
                r_flat[3] * d_cam[0] + r_flat[4] * d_cam[1] + r_flat[5] * d_cam[2],
                r_flat[6] * d_cam[0] + r_flat[7] * d_cam[1] + r_flat[8] * d_cam[2],
            );

            let b = if at_infinity {
                // A point at infinity has no parallax: the ray is the
                // direction, drawn to the horizon.
                direction_to_world(transform, d_world)
            } else {
                // Finite point: terminate at the nearest point on the
                // observed ray (so reprojection error shows),
                // t = dot(P - C, d_world) clamped to the forward direction.
                let t = (point_pos - center).dot(&d_world).max(0.0);
                place_to_world(transform, center + t * d_world)
            };
            Some(OverlayInstance::segment(
                place_to_world(transform, center),
                b,
                TRACK_RAY_COLOR,
                TRACK_RAY_HALF_WIDTH,
                OverlayInstance::HOLD,
            ))
        })
        .collect()
}

/// A place in the node's coordinates as a homogeneous world point, `w = 1`.
fn place_to_world(transform: &Se3Transform, p: nalgebra::Point3<f64>) -> [f32; 4] {
    let world = transform.apply_to_point(&p);
    [world.x as f32, world.y as f32, world.z as f32, 1.0]
}

/// A direction in the node's coordinates as a homogeneous world direction,
/// `w = 0`: turned by the similarity's rotation, which is all of it that
/// moves a direction.
fn direction_to_world(transform: &Se3Transform, d: nalgebra::Vector3<f64>) -> [f32; 4] {
    let world = transform.rotation.rotate_vector(&d);
    [world.x as f32, world.y as f32, world.z as f32, 0.0]
}

impl SceneRenderer {
    /// Write the selected point's track rays into their region of the overlay
    /// ribbon store, drawn on the scene target. The caller writes only when
    /// what they are built from has changed.
    ///
    /// See [`track_ray_instances`] for how each ray is built.
    pub(crate) fn upload_track_rays(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        edited: &EditedReconstruction,
        point: PointRef,
        sift_cache: &std::collections::HashMap<ImageRef, crate::state::CachedSiftFeatures>,
        transform: &Se3Transform,
    ) {
        let rays = track_ray_instances(edited, point, sift_cache, transform);
        self.overlay_ribbons.write(
            device,
            queue,
            ProducerId::TrackRays,
            OverlayTarget::Scene,
            &rays,
        );
    }

    /// Empty the track rays' region: no point selected, or none to draw. The
    /// region keeps its capacity.
    pub(crate) fn clear_track_rays(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        self.overlay_ribbons.write(
            device,
            queue,
            ProducerId::TrackRays,
            OverlayTarget::Scene,
            &[],
        );
    }
}

#[cfg(test)]
impl SceneRenderer {
    /// How many rays the track rays' region draws.
    pub(crate) fn track_ray_count(&self) -> u32 {
        self.overlay_ribbons
            .region(ProducerId::TrackRays)
            .map_or(0, |(_, count, _)| count)
    }

    /// The track rays' last write into the store, as instances: the rays it
    /// draws.
    pub(crate) fn track_rays(&self) -> Vec<OverlayInstance> {
        let Some((offset, count, _)) = self.overlay_ribbons.region(ProducerId::TrackRays) else {
            return Vec::new();
        };
        let stride = std::mem::size_of::<OverlayInstance>();
        let start = offset as usize * stride;
        bytemuck::pod_collect_to_vec(
            &self.overlay_ribbons.contents()[start..start + count as usize * stride],
        )
    }

    /// How many buffer writes the store has issued for the track rays.
    pub(crate) fn track_ray_writes(&self) -> usize {
        self.overlay_ribbons
            .issued
            .iter()
            .filter(|write| write.producer == ProducerId::TrackRays)
            .count()
    }
}
