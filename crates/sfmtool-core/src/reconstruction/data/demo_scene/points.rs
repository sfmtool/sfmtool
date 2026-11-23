// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The demo scene's points and who sees them: the camera poses as the
//! reconstruction stores them, the grid of points on the ground and on each
//! cube face, and the exact visibility test that decides which cameras
//! observe each point.
//!
//! The geometry is `f64`, for the reasons the renderer's is
//! (`super::render`): the visibility test is the renderer's slab test, and a
//! point on a face has to stay on its side of the face's plane through it.

use nalgebra::{Matrix3, Rotation3, UnitQuaternion, Vector3};

use crate::camera::CameraIntrinsics;
use crate::progress::{Cancelled, Progress};

use super::render::{camera_rotation, cube_hit, dot, CAMERA_CENTRES};
use super::texture::{lattice_hash, Surface};
use super::DemoSceneOptions;

/// The half side of the square of the ground the points are sampled on,
/// `[-2.5, 2.5] × [-2.5, 2.5]`, in metres.
pub(super) const GROUND_SAMPLING_HALF_SIDE: f64 = 2.5;

/// A camera's stored pose: its centre, and the canonical world-to-camera
/// rotation and translation, `p_cam = R · p_world + t`.
pub(super) struct DemoPose {
    pub(super) centre: [f64; 3],
    pub(super) rotation: UnitQuaternion<f64>,
    pub(super) translation: Vector3<f64>,
}

/// The five cameras' poses, in image order. The rotation is
/// [`camera_rotation`]'s, whose rows are the camera's axes, and the
/// translation `−R · C`.
pub(super) fn demo_poses() -> Vec<DemoPose> {
    CAMERA_CENTRES
        .iter()
        .map(|&centre| {
            let rows = camera_rotation(centre);
            let matrix = Matrix3::from_fn(|r, c| rows[r][c]);
            let translation = [0, 1, 2].map(|r| -dot(rows[r], centre));
            DemoPose {
                centre,
                rotation: UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(
                    matrix,
                )),
                translation: Vector3::from(translation),
            }
        })
        .collect()
}

/// The pixel where `camera` at `pose` sees the world point `x`, through the
/// stored pose, or `None` where `x` is not in front of the camera
/// (camera-space `z ≥ 0`). The pixel may lie outside the image.
pub(super) fn project(camera: &CameraIntrinsics, pose: &DemoPose, x: [f64; 3]) -> Option<[f64; 2]> {
    let p = pose.rotation * Vector3::from(x) + pose.translation;
    if p.z >= 0.0 || p.z.is_nan() {
        return None;
    }
    let (u, v) = camera.project(p.x / -p.z, p.y / -p.z);
    Some([u, v])
}

/// One sampled point that two or more cameras observe.
#[derive(Debug, Clone)]
pub(super) struct SampledPoint {
    /// The surface it lies on.
    pub(super) surface: Surface,
    /// Its position, in metres.
    pub(super) position: [f64; 3],
    /// The images that observe it, ascending.
    pub(super) images: Vec<u32>,
}

/// Whether the camera at `pose` observes the point `x` on a surface with
/// outward normal `normal`: `x` projects inside the image in front of the
/// camera; the camera is on the side the normal faces, no more than the
/// obliquity cut off it, `n · (C − x) / |C − x| ≥ cos_cut`; and the segment
/// from the camera centre to `x` does not enter the cube before it reaches
/// `x`, less `10⁻⁹` of its length, so a point on the face the segment enters
/// is not hidden by its own face.
pub(super) fn observes(
    camera: &CameraIntrinsics,
    pose: &DemoPose,
    normal: [f64; 3],
    x: [f64; 3],
    cos_cut: f64,
) -> bool {
    let Some([u, v]) = project(camera, pose, x) else {
        return false;
    };
    let inside =
        (0.0..f64::from(camera.width)).contains(&u) && (0.0..f64::from(camera.height)).contains(&v);
    if !inside {
        return false;
    }
    let to_camera = [0, 1, 2].map(|k| pose.centre[k] - x[k]);
    if dot(normal, to_camera) < cos_cut * dot(to_camera, to_camera).sqrt() {
        return false;
    }
    let segment = to_camera.map(|c| -c);
    !matches!(cube_hit(pose.centre, segment), Some((_, entry)) if entry < 1.0 - 1e-9)
}

/// The offset of grid point `(i, j)` from its grid position, as fractions of
/// a quarter of the spacing along each axis, each in `[-1, 1)`: two 32-bit
/// halves of [`lattice_hash`] of the node and `seed`.
fn grid_offset(i: i64, j: i64, seed: u64) -> [f64; 2] {
    let h = lattice_hash(i, j, seed);
    let unit = |bits: u64| (bits & 0xFFFF_FFFF) as f64 / 2_147_483_648.0 - 1.0;
    [unit(h), unit(h >> 32)]
}

/// The seed of a surface's grid offsets under the scene's `seed`: its noise
/// seed with a constant mixed in, so the offsets do not follow the texture.
fn offset_seed(surface: Surface, seed: u64) -> u64 {
    surface.noise_seed(seed) ^ 0x5A3C_96E1_0F87_D24B
}

/// The number of rows and columns of points on each cube face,
/// `round(1 / face_point_spacing)`, and at least 1.
pub(super) fn face_grid_side(options: &DemoSceneOptions) -> u32 {
    grid_side(1.0, options.face_point_spacing)
}

/// The number of rows and columns of points on the ground's sampling
/// square, `round(5 / ground_point_spacing)`, and at least 1.
pub(super) fn ground_grid_side(options: &DemoSceneOptions) -> u32 {
    grid_side(
        2.0 * GROUND_SAMPLING_HALF_SIDE,
        options.ground_point_spacing,
    )
}

/// `round(side / spacing)`, at least 1 and at most `u32::MAX`.
fn grid_side(side: f64, spacing: f64) -> u32 {
    // A spacing is finite and positive, so the ratio is at least 0; a spacing
    // under about 2⁻³² of the side would give more points than a
    // reconstruction holds anyway.
    (side / spacing).round().clamp(1.0, f64::from(u32::MAX)) as u32
}

/// The candidate points on `surface`, each with its grid position moved by a
/// seeded offset of up to a quarter of the spacing along each axis.
///
/// Each grid is a whole number of rows and columns, each point at the centre
/// of its cell, so the count scales with the inverse square of the spacing.
/// On the ground: `m` rows and columns over the sampling square
/// ([`ground_grid_side`]), at `−2.5 + (i + 0.5) · 5 / m` along each axis,
/// leaving out the points that land inside the cube's footprint. On a cube
/// face: `k` rows and columns at `(i + 0.5) / k` along each in-face axis from
/// the face's corner ([`face_grid_side`]), with the
/// four points nearest the face's corners not moved, so each cube corner lies
/// at a known place in those points' patches.
fn candidates(surface: Surface, options: &DemoSceneOptions) -> Vec<[f64; 3]> {
    let seed = offset_seed(surface, options.seed);
    let (origin, u, v) = surface.frame();
    let at = |s: f64, t: f64| [0, 1, 2].map(|k| origin[k] + s * u[k] + t * v[k]);
    let mut points = Vec::new();
    if surface == Surface::Ground {
        let m = ground_grid_side(options);
        let cell = 2.0 * GROUND_SAMPLING_HALF_SIDE / f64::from(m);
        for j in 0..m {
            for i in 0..m {
                let [di, dj] = grid_offset(i64::from(i), i64::from(j), seed);
                let x = at(
                    -GROUND_SAMPLING_HALF_SIDE + (f64::from(i) + 0.5 + 0.25 * di) * cell,
                    -GROUND_SAMPLING_HALF_SIDE + (f64::from(j) + 0.5 + 0.25 * dj) * cell,
                );
                let in_footprint = dot(x, super::texture::CUBE_A).abs() < 0.5
                    && dot(x, super::texture::CUBE_B).abs() < 0.5;
                if !in_footprint {
                    points.push(x);
                }
            }
        }
    } else {
        let k = face_grid_side(options);
        let side = f64::from(k);
        for j in 0..k {
            for i in 0..k {
                let at_corner = (i == 0 || i == k - 1) && (j == 0 || j == k - 1);
                let [di, dj] = if at_corner {
                    [0.0, 0.0]
                } else {
                    grid_offset(i64::from(i), i64::from(j), seed)
                };
                points.push(at(
                    (f64::from(i) + 0.5 + 0.25 * di) / side,
                    (f64::from(j) + 0.5 + 0.25 * dj) / side,
                ));
            }
        }
    }
    points
}

/// Sample the points on the ground and on every cube face but the base, in
/// the order of [`Surface::ALL`], and keep the ones two or more cameras
/// observe ([`observes`]), with those cameras. Counts the candidate points in
/// `progress`, one surface at a time.
pub(super) fn sample_points(
    options: &DemoSceneOptions,
    camera: &CameraIntrinsics,
    poses: &[DemoPose],
    progress: &Progress<'_>,
) -> Result<Vec<SampledPoint>, Cancelled> {
    let cos_cut = options.max_obliquity_deg.to_radians().cos();
    let surfaces: Vec<Surface> = Surface::ALL
        .into_iter()
        .filter(|&s| s != Surface::Base)
        .collect();
    let candidates: Vec<(Surface, Vec<[f64; 3]>)> = surfaces
        .iter()
        .map(|&s| (s, candidates(s, options)))
        .collect();
    let total: u64 = candidates.iter().map(|(_, c)| c.len() as u64).sum();
    let mut done = 0u64;
    let mut kept = Vec::new();
    for (surface, points) in candidates {
        progress.check_cancel()?;
        let normal = surface.normal();
        for position in points {
            let images: Vec<u32> = poses
                .iter()
                .enumerate()
                .filter(|(_, pose)| observes(camera, pose, normal, position, cos_cut))
                .map(|(i, _)| i as u32)
                .collect();
            if images.len() >= 2 {
                kept.push(SampledPoint {
                    surface,
                    position,
                    images,
                });
            }
            done += 1;
        }
        progress.count(done, Some(total), "points");
    }
    Ok(kept)
}
