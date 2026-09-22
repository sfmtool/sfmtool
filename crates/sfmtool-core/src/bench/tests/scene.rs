// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The synthetic capture the bench tests are decided against.
//!
//! The scene is the one the localization kernel's own tests use -- pinhole
//! cameras looking down world `+z` at a textured plane -- wrapped in an
//! `embedded_patches` reconstruction whose stored keypoints are the exact
//! projections, so what a step should have written is known to the pixel.

use std::sync::Arc;

use nalgebra::{Point3, Quaternion, UnitQuaternion, Vector3};
use ndarray::{Array2, Array4};

use crate::camera::remap::{ImageU8, ImageU8Pyramid};
use crate::camera::{CameraIntrinsics, CameraModel};
use crate::geometry::RigidTransform;
use crate::patch::cloud::OrientedPatch;
use crate::patch::normal_refine::ProjectedImage;
use crate::reconstruction::data::{
    ObservationSource, Point3D, SfmrImage, SfmrReconstruction, TrackObservation,
};
use crate::reconstruction::edited::EditedReconstruction;

pub(super) const IMG_W: u32 = 128;
pub(super) const IMG_H: u32 = 128;
const FOCAL: f64 = 160.0;
const PLANE_Z: f64 = 4.0;
const HALF_EXTENT: f64 = 0.12;
/// The camera centres, in world space. The first two observe the point; the
/// third is the one an observation is added in.
const CENTERS: [[f64; 3]; 3] = [[-0.5, -0.3, 0.0], [0.45, 0.25, 0.0], [0.1, -0.55, 0.0]];

fn pinhole() -> CameraIntrinsics {
    CameraIntrinsics {
        model: CameraModel::Pinhole {
            focal_length_x: FOCAL,
            focal_length_y: FOCAL,
            principal_point_x: IMG_W as f64 / 2.0,
            principal_point_y: IMG_H as f64 / 2.0,
        },
        width: IMG_W,
        height: IMG_H,
    }
}

/// The plane's texture, in world units, at a frequency scaled for a plane
/// `depth` away.
///
/// The frequencies are divided by the depth so that one period always spans the
/// same number of *pixels*, whatever distance the plane is put at: a scene built
/// two hundred units out shows a photograph of the same detail as one built four
/// units out, rather than an aliased mess.
fn texture(x: f64, y: f64, depth: f64) -> f64 {
    let s = PLANE_Z / depth;
    127.5
        + 55.0 * (x * 17.0 * s).sin()
        + 45.0 * (y * 23.0 * s).cos()
        + 25.0 * ((x + y) * 31.0 * s).sin()
}

/// What a pinhole at `center` looking down world `+z` sees of the textured
/// plane `z = depth`.
fn render_plane_view(center: [f64; 3], depth: f64) -> ImageU8 {
    let (cx, cy) = (IMG_W as f64 / 2.0, IMG_H as f64 / 2.0);
    let mut data = Vec::with_capacity((IMG_W * IMG_H) as usize);
    for row in 0..IMG_H {
        for col in 0..IMG_W {
            let dx = (col as f64 + 0.5 - cx) / FOCAL;
            let dy = (row as f64 + 0.5 - cy) / FOCAL;
            let lambda = depth - center[2];
            let x = center[0] + lambda * dx;
            let y = center[1] + lambda * dy;
            data.push(texture(x, y, depth).clamp(0.0, 255.0).round() as u8);
        }
    }
    ImageU8::new(IMG_W, IMG_H, 1, data)
}

/// The pose of a camera at `center`: a half-turn about `x`, which puts the
/// canonical `-z` camera axis along world `+z`.
fn pose(center: [f64; 3]) -> RigidTransform {
    RigidTransform::from_wxyz_translation([0.0, 1.0, 0.0, 0.0], [-center[0], center[1], center[2]])
}

/// The decoded views, held so the borrows in [`views`] have something to point
/// at.
pub(super) struct Scene {
    centers: Vec<[f64; 3]>,
    cameras: Vec<CameraIntrinsics>,
    poses: Vec<RigidTransform>,
    pyramids: Vec<ImageU8Pyramid>,
}

impl Scene {
    pub(super) fn new() -> Self {
        Self::from_centers(&CENTERS, PLANE_Z)
    }

    /// A capture of the textured plane `z = depth` from a pinhole at each of
    /// `centers`, all looking down world `+z`.
    ///
    /// What the baseline and the depth are together is what decides whether a
    /// track of this scene has an observable depth at all, so the two are the
    /// knobs a test turns: the near scene [`Scene::new`] builds resolves a
    /// depth, and one whose cameras step by centimetres at a plane hundreds of
    /// units out resolves only a bearing.
    pub(super) fn from_centers(centers: &[[f64; 3]], depth: f64) -> Self {
        Self {
            centers: centers.to_vec(),
            cameras: centers.iter().map(|_| pinhole()).collect(),
            poses: centers.iter().map(|&c| pose(c)).collect(),
            pyramids: centers
                .iter()
                .map(|&c| ImageU8Pyramid::build(&render_plane_view(c, depth), 4))
                .collect(),
        }
    }

    pub(super) fn views(&self) -> Vec<ProjectedImage<'_>> {
        self.cameras
            .iter()
            .zip(&self.poses)
            .zip(&self.pyramids)
            .map(|((camera, cam_from_world), pyramid)| ProjectedImage {
                camera,
                cam_from_world,
                pyramid,
            })
            .collect()
    }

    /// Where `world` lands in image `i`, in source-image px.
    pub(super) fn project(&self, i: usize, world: Point3<f64>) -> [f64; 2] {
        self.project_homogeneous(i, world, 1.0)
    }

    /// Where the homogeneous world point `(coords, w)` lands in image `i`, in
    /// source-image px: a place at `w == 1`, a direction at `w == 0`.
    pub(super) fn project_homogeneous(&self, i: usize, coords: Point3<f64>, w: f64) -> [f64; 2] {
        let cam = self.poses[i].transform_point_homogeneous(coords.coords, w);
        let (u, v) = self.cameras[i]
            .ray_to_pixel([cam.x, cam.y, cam.z])
            .expect("the point is in front of every camera of this scene");
        [u, v]
    }

    /// How many images the scene holds.
    pub(super) fn len(&self) -> usize {
        self.centers.len()
    }
}

/// The patch the point carries: on the plane, facing the cameras.
fn plane_patch(center: Point3<f64>) -> OrientedPatch {
    OrientedPatch::from_center_normal(
        center,
        Vector3::new(0.0, 0.0, -1.0),
        Vector3::new(0.0, 1.0, 0.0),
        [HALF_EXTENT, HALF_EXTENT],
    )
}

/// An `embedded_patches` reconstruction over [`Scene`] holding one point on the
/// plane, observed by images 0 and 1 at their exact projections. Image 2 does
/// not observe it, and is where an observation is added.
fn fixture(scene: &Scene, world: Point3<f64>) -> SfmrReconstruction {
    fixture_of(scene, world, 1.0, &[0, 1], &plane_patch(world))
}

/// An `embedded_patches` reconstruction over `scene` holding one point at the
/// homogeneous coordinate `(coordinate, w)`, standing on `patch`, observed by
/// `observing` at its exact projections in each.
///
/// `w` is `1.0` for a place and `0.0` for a bearing, in which case `coordinate`
/// is the unit direction and `patch` the tangent frame the format states for
/// such a row. Everything else is the same fixture either way, which is the
/// point: a test of the finite/infinity boundary needs the two sides built the
/// same way apart from that one number.
pub(super) fn fixture_of(
    scene: &Scene,
    coordinate: Point3<f64>,
    w: f64,
    observing: &[u32],
    patch: &OrientedPatch,
) -> SfmrReconstruction {
    let mut recon = SfmrReconstruction::demo(1);
    let n = scene.len();

    recon.image_table.cameras = vec![pinhole()];
    recon.image_table.images = (0..n)
        .map(|i| SfmrImage {
            name: format!("image_{i}.jpg"),
            camera_index: 0,
            quaternion_wxyz: UnitQuaternion::from_quaternion(Quaternion::new(0.0, 1.0, 0.0, 0.0)),
            translation_xyz: Vector3::new(
                -scene.centers[i][0],
                scene.centers[i][1],
                scene.centers[i][2],
            ),
        })
        .collect();
    // One row per image either way: `resize` covers a scene with fewer cameras
    // than the demo value it is built on and one with more.
    let stats_row = recon.image_table.depth_statistics.images[0].clone();
    recon
        .image_table
        .depth_statistics
        .images
        .resize(n, stats_row);
    let histogram_row = recon.image_table.depth_histogram_counts[0].clone();
    recon
        .image_table
        .depth_histogram_counts
        .resize(n, histogram_row);

    let set = &mut recon.point_set;
    set.points = vec![Point3D {
        position: coordinate,
        w,
        color: [120, 130, 140],
        error: 0.5,
        // A `w = 0` row carries a zero normal, which is what the format states
        // and what the demotion pass leaves.
        normal: if w == 0.0 {
            Vector3::zeros()
        } else {
            Vector3::new(0.0, 0.0, -1.0)
        },
    }];
    set.tracks = observing
        .iter()
        .map(|&image_index| TrackObservation {
            image_index,
            point_index: 0,
        })
        .collect();
    set.observation_counts = vec![observing.len() as u32];
    let mut keypoints = Array2::<f32>::zeros((observing.len(), 2));
    for (k, &image) in observing.iter().enumerate() {
        let p = scene.project_homogeneous(image as usize, coordinate, w);
        keypoints[[k, 0]] = p[0] as f32;
        keypoints[[k, 1]] = p[1] as f32;
    }
    set.observations = ObservationSource::EmbeddedPatches {
        keypoints_xy: keypoints,
        image_file_hashes: vec![[0u8; 16]; n],
    };
    let halfvec = |axis: Vector3<f64>, half: f64| {
        let v = axis * half;
        [v.x as f32, v.y as f32, v.z as f32]
    };
    let u = halfvec(patch.u_axis, patch.half_extent[0]);
    let v = halfvec(patch.v_axis, patch.half_extent[1]);
    set.patch_u_halfvec_xyz = Some(Array2::from_shape_vec((1, 3), u.to_vec()).unwrap());
    set.patch_v_halfvec_xyz = Some(Array2::from_shape_vec((1, 3), v.to_vec()).unwrap());
    recon.metadata.feature_source =
        sfmtool_sfmr_format::FEATURE_SOURCE_EMBEDDED_PATCHES.to_string();
    recon.rebuild_derived_fields();
    recon
}

/// A reconstruction carrying the optional per-observation and per-point columns
/// a created point has to fill in: an `(P, r, r, 4)` bitmap column, an
/// observation confidence and a normal confidence.
pub(super) fn with_columns(mut recon: SfmrReconstruction, r: usize) -> SfmrReconstruction {
    let set = &mut recon.point_set;
    set.patch_bitmaps_y_x_rgba = Some(Arc::new(Array4::zeros((set.points.len(), r, r, 4))));
    set.observation_confidence = Some(vec![200; set.tracks.len()]);
    set.normal_confidence = Some(vec![180; set.points.len()]);
    recon.rebuild_derived_fields();
    recon
}

/// [`fixture`] carrying the optional per-observation and per-point columns a
/// created point has to fill in: an `(P, r, r, 4)` bitmap column, an
/// observation confidence and a normal confidence.
pub(super) fn fixture_with_columns(
    scene: &Scene,
    world: Point3<f64>,
    r: usize,
) -> SfmrReconstruction {
    with_columns(fixture(scene, world), r)
}

/// The fixture wrapped as a version with no edits.
pub(super) fn edited(scene: &Scene, world: Point3<f64>) -> EditedReconstruction {
    EditedReconstruction::new(Arc::new(fixture(scene, world)))
}

pub(super) const WORLD: Point3<f64> = Point3::new(0.0, 0.0, PLANE_Z);
