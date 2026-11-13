// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The crops' grid, sights, renders, pyramids and pair levels.

use nalgebra::{Point3, Vector3};

use crate::camera::image::{ImageU8, ImageU8Pyramid};
use crate::camera::{CameraIntrinsics, CameraModel};
use crate::features::optical_flow::GrayImage;
use crate::geometry::RigidTransform;
use crate::patch::normal_refine::ProjectedImage;

use super::super::scene::{look_at, origin, ring, Scene, HALF_EXTENT};
use super::super::PairwiseFlowNormalOptions;
use super::{render_crops, Crop, CropGrid, CropPyramid, PairLevel, ViewSight};

fn pinhole(focal: f64) -> CameraIntrinsics {
    CameraIntrinsics {
        model: CameraModel::Pinhole {
            focal_length_x: focal,
            focal_length_y: focal,
            principal_point_x: 64.0,
            principal_point_y: 64.0,
        },
        width: 128,
        height: 128,
        rolling_shutter: None,
    }
}

/// Cameras of the given focal lengths, each 4 units from the origin straight
/// along `+z`, facing the plane `z = 0`, with no photograph worth reading.
struct Facing {
    cameras: Vec<CameraIntrinsics>,
    poses: Vec<RigidTransform>,
    pyramid: ImageU8Pyramid,
}

impl Facing {
    fn new(focals: &[f64]) -> Self {
        let pose = look_at(Vector3::new(0.0, 0.0, 4.0), Vector3::zeros());
        Self {
            cameras: focals.iter().map(|&f| pinhole(f)).collect(),
            poses: vec![pose; focals.len()],
            pyramid: ImageU8Pyramid::build(&ImageU8::new(8, 8, 1, vec![128; 64]), 1),
        }
    }

    fn views(&self) -> Vec<ProjectedImage<'_>> {
        (0..self.cameras.len())
            .map(|k| ProjectedImage {
                camera: &self.cameras[k],
                cam_from_world: &self.poses[k],
                pyramid: &self.pyramid,
            })
            .collect()
    }
}

/// The grid for a patch of half-extent 0.5 on `z = 0`, axes along world `x`
/// and `y`, seen by cameras of these focal lengths.
fn grid_for(focals: &[f64]) -> (CropGrid, Vec<ViewSight>) {
    let facing = Facing::new(focals);
    let (grid, sights) = CropGrid::new(
        Point3::origin(),
        Vector3::z(),
        [Vector3::x(), Vector3::y()],
        0.5,
        &facing.views(),
        &PairwiseFlowNormalOptions::default(),
    );
    (grid.unwrap(), sights)
}

/// A patch 1 unit across seen from 4 units by a focal length of `f` covers
/// `f / 4` pixels: 10, 43 and 1000 pixels give 24, 43 and 128 samples a
/// side, and the crop is the larger of 3 half-extents and 48 of the finest
/// view's pixels. A coarser second view does not change the grid.
#[test]
fn the_grid_follows_the_finest_view() {
    let (grid, sights) = grid_for(&[40.0, 20.0]);
    assert!((grid.samples_per_patch - 24.0).abs() < 1e-6, "{grid:?}");
    // One pixel is 2.4 samples, so 48 pixels are 115.2 samples.
    assert_eq!(grid.half_width, 116);
    let ViewSight::Seen { pixel_size, .. } = sights[0] else {
        panic!("{sights:?}");
    };
    assert!((pixel_size - 0.1).abs() < 1e-6);

    let (grid, _) = grid_for(&[172.0, 100.0]);
    assert!((grid.samples_per_patch - 43.0).abs() < 1e-6, "{grid:?}");
    // 3 half-extents are 64.5 samples, more than the 48-pixel floor.
    assert_eq!(grid.half_width, 65);

    let (grid, _) = grid_for(&[4000.0]);
    assert!((grid.samples_per_patch - 128.0).abs() < 1e-6, "{grid:?}");
    assert_eq!(grid.side(), 384);
}

/// A patch 4 pixels across in its finest view: 48 of its pixels would be
/// 288 samples of the 24-sample grid, past the 512-sample cap, so the grid
/// is coarsened to keep the world extent at 256 samples.
#[test]
fn the_crop_cap_coarsens_the_grid_and_keeps_the_extent() {
    let (grid, _) = grid_for(&[16.0]);
    assert_eq!(grid.half_width, 256);
    let world = f64::from(grid.half_width) * grid.plane.spacing;
    assert!((world - 48.0 * 0.25).abs() < 1e-9, "{world}");
    assert!(grid.samples_per_patch < 24.0, "{grid:?}");
}

#[test]
fn views_too_oblique_or_at_the_point_are_left_out() {
    let camera = pinhole(100.0);
    let pyramid = ImageU8Pyramid::build(&ImageU8::new(8, 8, 1, vec![128; 64]), 1);
    let at = |deg: f64| {
        let r = deg.to_radians();
        Vector3::new(4.0 * r.sin(), 0.0, 4.0 * r.cos())
    };
    let poses = [
        look_at(at(30.0), Vector3::zeros()),
        look_at(at(85.0), Vector3::zeros()),
        // A camera centred on the point itself.
        RigidTransform::identity(),
    ];
    let views: Vec<ProjectedImage<'_>> = poses
        .iter()
        .map(|pose| ProjectedImage {
            camera: &camera,
            cam_from_world: pose,
            pyramid: &pyramid,
        })
        .collect();
    let (grid, sights) = CropGrid::new(
        Point3::origin(),
        Vector3::z(),
        [Vector3::x(), Vector3::y()],
        0.5,
        &views,
        &PairwiseFlowNormalOptions::default(),
    );
    assert!(grid.is_some());
    assert!(
        matches!(sights[0], ViewSight::Seen { angle_deg, .. } if (angle_deg - 30.0).abs() < 1e-9)
    );
    assert!(
        matches!(sights[1], ViewSight::TooOblique { angle_deg } if (angle_deg - 85.0).abs() < 1e-9)
    );
    assert_eq!(sights[2], ViewSight::NoSight);
    // With only the oblique view there is no grid.
    let (grid, _) = CropGrid::new(
        Point3::origin(),
        Vector3::z(),
        [Vector3::x(), Vector3::y()],
        0.5,
        &views[1..],
        &PairwiseFlowNormalOptions::default(),
    );
    assert!(grid.is_none());
}

/// The crops' first axis is the patch's `u` projected onto the plane; where
/// `u` is nearly the plane's normal, `v` is projected instead. The crop's
/// `u × v` is the plane's normal either way.
#[test]
fn the_crops_axes_come_from_the_patchs_axes() {
    let facing = Facing::new(&[100.0]);
    let normal = Vector3::new(0.2, 0.0, 1.0).normalize();
    let axes_for = |u: Vector3<f64>, v: Vector3<f64>| {
        let (grid, _) = CropGrid::new(
            Point3::origin(),
            normal,
            [u, v],
            0.5,
            &facing.views(),
            &PairwiseFlowNormalOptions::default(),
        );
        let plane = grid.unwrap().plane;
        (plane.x_axis, -plane.y_axis)
    };
    let (u, v) = axes_for(Vector3::x(), Vector3::y());
    let projected = (Vector3::x() - normal * normal.x).normalize();
    assert!((u - projected).norm() < 1e-12);
    assert!((u.cross(&v) - normal).norm() < 1e-12);

    let (u, v) = axes_for(normal, Vector3::y());
    assert!((v - Vector3::y()).norm() < 1e-12, "{v:?}");
    assert!((u.cross(&v) - normal).norm() < 1e-12);
}

/// Two views 90° apart, rendered on the plane they photograph, show the same
/// texture at the same samples: the crops differ far less than one crop
/// differs from itself shifted by two samples.
#[test]
fn crops_on_the_photographed_plane_line_up() {
    let normal = Vector3::new(0.3, -0.2, 1.0).normalize();
    let centres = ring(normal, 45.0, 2, 10.0);
    let scene = Scene::plane(normal, &centres, 30.0, 0.0, 11);
    let views = scene.views();
    let options = PairwiseFlowNormalOptions::default();
    let (grid, sights) = CropGrid::new(
        origin(),
        normal,
        [Vector3::x(), Vector3::y()],
        HALF_EXTENT,
        &views,
        &options,
    );
    let grid = grid.unwrap();
    let crops: Vec<Crop> = render_crops(&grid, &views, &sights)
        .into_iter()
        .map(Option::unwrap)
        .collect();
    let side = grid.side() as usize;
    for crop in &crops {
        assert!(crop.coverage > 0.9, "{}", crop.coverage);
        assert!((crop.angle_deg - 45.0).abs() < 1e-6);
        // At 45° a pixel spans 1/cos 45° = 1.41 times its frontal size on
        // the plane along its coarser axis.
        assert!(crop.samples_per_pixel > 1.0, "{}", crop.samples_per_pixel);
    }
    let difference = |shift: usize| {
        let (a, b) = (crops[0].image.data(), crops[1].image.data());
        let (mut sum, mut count) = (0.0f64, 0usize);
        for r in 8..side - 8 {
            for c in 8..side - 8 - shift {
                let (i, j) = (r * side + c, r * side + c + shift);
                if crops[0].valid[i] && crops[1].valid[j] {
                    sum += f64::from((a[i] - b[j]).abs());
                    count += 1;
                }
            }
        }
        sum / count as f64 * 255.0
    };
    let aligned = difference(0);
    let shifted = difference(2);
    assert!(
        aligned < 0.25 * shifted,
        "aligned {aligned}, shifted {shifted}"
    );
}

/// Off the photograph the crop holds the mean of what is on it, and a
/// pyramid level is valid only where the whole kernel under it is.
#[test]
fn the_pyramids_mask_shrinks_with_its_kernel() {
    let side = 32u32;
    let valid: Vec<bool> = (0..side * side).map(|k| k % side >= 10).collect();
    let crop = Crop {
        image: GrayImage::new_constant(side, side, 0.5),
        valid,
        centre: Vector3::zeros(),
        angle_deg: 0.0,
        samples_per_pixel: 1.0,
        coverage: 22.0 / 32.0,
    };
    let pyramid = CropPyramid::new(&crop, 2);
    let (image, mask) = pyramid.level(1);
    assert_eq!(image.width(), 16);
    // Level-1 column `c` reads level-0 columns `2c − 2 ..= 2c + 3`.
    for (c, &valid) in mask.iter().take(16).enumerate() {
        assert_eq!(valid, 2 * c >= 12, "column {c}");
    }
    let (image, mask) = pyramid.level(2);
    assert_eq!(image.width(), 8);
    for (c, &valid) in mask.iter().take(8).enumerate() {
        assert_eq!(valid, 2 * c >= 8, "column {c}");
    }
}

#[test]
fn a_pair_is_read_at_the_level_of_its_coarser_view() {
    assert_eq!(
        PairLevel::new(0.5, 0.9),
        PairLevel {
            level: 0,
            upsampling: 1.0
        }
    );
    assert_eq!(
        PairLevel::new(1.5, 1.0),
        PairLevel {
            level: 0,
            upsampling: 1.5
        }
    );
    assert_eq!(
        PairLevel::new(1.0, 3.0),
        PairLevel {
            level: 1,
            upsampling: 1.5
        }
    );
    assert_eq!(
        PairLevel::new(4.0, 0.2),
        PairLevel {
            level: 2,
            upsampling: 1.0
        }
    );
    assert_eq!(PairLevel::new(4.0, 0.2).scale(), 4.0);
    assert_eq!(
        PairLevel::new(f64::NAN, 1.0),
        PairLevel {
            level: 0,
            upsampling: 1.0
        }
    );
}
