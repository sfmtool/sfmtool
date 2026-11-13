// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! A synthetic capture for the pairwise-flow pieces' tests: pinhole cameras
//! looking at a plane through the origin that carries a smooth random
//! texture, each photograph ray-cast pixel by pixel, with white noise added.
//! The crops are then rendered from these photographs by the real code path,
//! through each camera's model.

use nalgebra::{Matrix3, Point3, Vector3};

use crate::camera::image::{ImageU8, ImageU8Pyramid};
use crate::camera::{CameraIntrinsics, CameraModel};
use crate::geometry::{RigidTransform, RotQuaternion};
use crate::patch::normal_refine::ProjectedImage;

use super::model::tangent_basis;
use super::texture::Gaussian;

/// The photographs' side, pixels.
pub(super) const IMAGE_SIDE: u32 = 160;
/// The cameras' focal length, pixels.
pub(super) const FOCAL: f64 = 200.0;
/// How far each camera stands from the point.
pub(super) const DISTANCE: f64 = 4.0;
/// The patch half-extent the tests read the scene with: 12.5 px of a camera
/// facing the plane, so the grid is close to one sample per pixel.
pub(super) const HALF_EXTENT: f64 = 0.25;

/// A pinhole camera of [`IMAGE_SIDE`] square pixels and [`FOCAL`].
pub(super) fn pinhole() -> CameraIntrinsics {
    CameraIntrinsics {
        model: CameraModel::Pinhole {
            focal_length_x: FOCAL,
            focal_length_y: FOCAL,
            principal_point_x: f64::from(IMAGE_SIDE) / 2.0,
            principal_point_y: f64::from(IMAGE_SIDE) / 2.0,
        },
        width: IMAGE_SIDE,
        height: IMAGE_SIDE,
        rolling_shutter: None,
    }
}

/// The world-to-camera pose of a camera at `centre` looking at `target`, in
/// the camera frame the models use (`-z` forward, `+y` up): its `-z` along
/// the line of sight, its `y` as close to world `-y` as that allows.
pub(super) fn look_at(centre: Vector3<f64>, target: Vector3<f64>) -> RigidTransform {
    let forward = (target - centre).normalize();
    let hint = if forward.y.abs() > 0.99 {
        Vector3::x()
    } else {
        -Vector3::y()
    };
    let up = (hint - forward * hint.dot(&forward)).normalize();
    let back = -forward;
    let right = up.cross(&back);
    let r = Matrix3::from_rows(&[right.transpose(), up.transpose(), back.transpose()]);
    RigidTransform {
        rotation: RotQuaternion::from_rotation_matrix(r),
        translation: -(r * centre),
    }
}

/// `count` camera centres `DISTANCE` from the origin, on a cone of half-angle
/// `cone_deg` about `axis`, evenly spaced in azimuth from `phase_deg`.
pub(super) fn ring(
    axis: Vector3<f64>,
    cone_deg: f64,
    count: usize,
    phase_deg: f64,
) -> Vec<Vector3<f64>> {
    let axis = axis.normalize();
    let (p, q) = tangent_basis(axis, Vector3::x());
    let cone = cone_deg.to_radians();
    (0..count)
        .map(|k| {
            let azimuth = (phase_deg + 360.0 * k as f64 / count as f64).to_radians();
            let side = p * azimuth.cos() + q * azimuth.sin();
            (axis * cone.cos() + side * cone.sin()) * DISTANCE
        })
        .collect()
}

/// A smooth random texture on a plane: a sum of sinusoids with wavelengths
/// of 3 to 12 pixels of a camera facing the plane from [`DISTANCE`], whose
/// standard deviation is `contrast` grey levels.
struct Texture {
    waves: Vec<([f64; 2], f64, f64)>,
}

impl Texture {
    fn new(contrast: f64, seed: u64) -> Self {
        const WAVES: usize = 32;
        let mut normal = Gaussian::new(seed);
        let pixel = DISTANCE / FOCAL;
        let amplitude = contrast * (2.0 / WAVES as f64).sqrt();
        let waves = (0..WAVES)
            .map(|k| {
                let angle = normal.next() * 10.0;
                let fraction = k as f64 / (WAVES - 1) as f64;
                let wavelength = pixel * (3.0 + 9.0 * fraction);
                let frequency = std::f64::consts::TAU / wavelength;
                let phase = normal.next() * 10.0;
                (
                    [frequency * angle.cos(), frequency * angle.sin()],
                    phase,
                    amplitude,
                )
            })
            .collect();
        Self { waves }
    }

    fn at(&self, a: f64, b: f64) -> f64 {
        self.waves
            .iter()
            .map(|&(k, phase, amplitude)| amplitude * (k[0] * a + k[1] * b + phase).sin())
            .sum()
    }
}

/// Cameras photographing one textured plane.
pub(super) struct Scene {
    cameras: Vec<CameraIntrinsics>,
    poses: Vec<RigidTransform>,
    pyramids: Vec<ImageU8Pyramid>,
}

impl Scene {
    /// The plane through the origin with unit normal `normal`, textured at
    /// `contrast` grey levels about mid-grey, photographed by a camera at
    /// each of `centres` looking at the origin, with white noise of
    /// `noise` grey levels.
    pub(super) fn plane(
        normal: Vector3<f64>,
        centres: &[Vector3<f64>],
        contrast: f64,
        noise: f64,
        seed: u64,
    ) -> Self {
        let normal = normal.normalize();
        let (e1, e2) = tangent_basis(normal, Vector3::x());
        let texture = Texture::new(contrast, seed);
        let camera = pinhole();
        let mut grain = Gaussian::new(seed ^ 0xA5A5);
        let mut poses = Vec::new();
        let mut pyramids = Vec::new();
        for &centre in centres {
            let pose = look_at(centre, Vector3::zeros());
            let r = pose.to_rotation_matrix();
            let mut data = Vec::with_capacity((IMAGE_SIDE * IMAGE_SIDE) as usize);
            for row in 0..IMAGE_SIDE {
                for col in 0..IMAGE_SIDE {
                    let ray_cam = Vector3::from(
                        camera.pixel_to_ray(f64::from(col) + 0.5, f64::from(row) + 0.5),
                    );
                    let ray = r.transpose() * ray_cam;
                    let along = normal.dot(&ray);
                    let t = -normal.dot(&centre) / along;
                    let value = if along.abs() > 1e-12 && t > 0.0 {
                        let p = centre + ray * t;
                        128.0 + texture.at(p.dot(&e1), p.dot(&e2))
                    } else {
                        128.0
                    };
                    let value = value + noise * grain.next();
                    data.push(value.round().clamp(0.0, 255.0) as u8);
                }
            }
            let image = ImageU8::new(IMAGE_SIDE, IMAGE_SIDE, 1, data);
            pyramids.push(ImageU8Pyramid::build(&image, 4));
            poses.push(pose);
        }
        Self {
            cameras: vec![camera; centres.len()],
            poses,
            pyramids,
        }
    }

    /// Every camera as a view the pieces read.
    pub(super) fn views(&self) -> Vec<ProjectedImage<'_>> {
        (0..self.poses.len())
            .map(|k| ProjectedImage {
                camera: &self.cameras[k],
                cam_from_world: &self.poses[k],
                pyramid: &self.pyramids[k],
            })
            .collect()
    }
}

/// The point the scenes' planes pass through.
pub(super) fn origin() -> Point3<f64> {
    Point3::origin()
}
