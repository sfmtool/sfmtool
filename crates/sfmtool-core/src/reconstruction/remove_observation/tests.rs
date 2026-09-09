// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Removing an observation: the refusals, the three outcomes -- a shorter
//! finite track, a bearing, a deleted point -- and the round trip back to a
//! finite position through `add_observation`.
//!
//! The scene here needs no pixels, because the edit needs none: four pinhole
//! cameras looking down world `+z`, and one point whose stored keypoints are
//! the exact projections, so the re-triangulation's truth is known.

use std::sync::Arc;

use nalgebra::{Point3, Quaternion, UnitQuaternion, Vector3};
use ndarray::{Array2, Array4};

use crate::camera::{CameraIntrinsics, CameraModel};
use crate::reconstruction::data::{
    ObservationSource, Point3D, SfmrImage, SfmrReconstruction, TrackObservation,
};

use super::*;

const IMG_W: u32 = 128;
const IMG_H: u32 = 128;
const FOCAL: f64 = 160.0;
const PLANE_Z: f64 = 4.0;
const HALF_EXTENT: f64 = 0.12;
/// The camera centres, in world space.
const CENTERS: [[f64; 3]; 4] = [
    [-0.5, -0.3, 0.0],
    [0.45, 0.25, 0.0],
    [0.1, -0.55, 0.0],
    [-0.2, 0.5, 0.0],
];
/// The point every fixture holds.
const WORLD: Point3<f64> = Point3::new(0.0, 0.0, PLANE_Z);
/// The bitmap column's tile resolution.
const TILE: usize = 4;

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

/// Where `world` lands in image `i` of `table`, in source-image px.
fn project(table: &ImageTable, i: usize, world: Point3<f64>) -> [f32; 2] {
    let image = &table.images[i];
    let cam = image.quaternion_wxyz.to_rotation_matrix() * world.coords + image.translation_xyz;
    let (u, v) = table.cameras[image.camera_index as usize]
        .ray_to_pixel([cam.x, cam.y, cam.z])
        .expect("the point is in front of every camera of this scene");
    [u as f32, v as f32]
}

/// An `embedded_patches` reconstruction holding one point on the plane,
/// observed by the first `observed` images at their exact projections, with a
/// patch frame, a patch bitmap and both confidence columns.
fn fixture(observed: usize) -> SfmrReconstruction {
    let n = CENTERS.len();
    let mut recon = SfmrReconstruction::demo(1);
    recon.image_table.cameras = vec![pinhole()];
    recon.image_table.images = (0..n)
        .map(|i| SfmrImage {
            name: format!("image_{i}.jpg"),
            camera_index: 0,
            quaternion_wxyz: UnitQuaternion::from_quaternion(Quaternion::new(0.0, 1.0, 0.0, 0.0)),
            translation_xyz: Vector3::new(-CENTERS[i][0], CENTERS[i][1], CENTERS[i][2]),
        })
        .collect();
    recon.image_table.thumbnails_y_x_rgb = Arc::new(Array4::zeros((
        n,
        sfmr_format::THUMBNAIL_SIZE,
        sfmr_format::THUMBNAIL_SIZE,
        3,
    )));
    recon.image_table.depth_statistics.images.truncate(n);
    recon.image_table.depth_histogram_counts.truncate(n);

    let set = &mut recon.point_set;
    set.points = vec![Point3D {
        position: WORLD,
        w: 1.0,
        color: [120, 130, 140],
        error: 0.5,
        normal: Vector3::new(0.0, 0.0, -1.0),
    }];
    set.tracks = (0..observed)
        .map(|i| TrackObservation {
            image_index: i as u32,
            point_index: 0,
        })
        .collect();
    set.observation_counts = vec![observed as u32];
    let mut keypoints = Array2::<f32>::zeros((observed, 2));
    for i in 0..observed {
        let p = project(&recon.image_table, i, WORLD);
        keypoints[[i, 0]] = p[0];
        keypoints[[i, 1]] = p[1];
    }
    let set = &mut recon.point_set;
    set.observations = ObservationSource::EmbeddedPatches {
        keypoints_xy: keypoints,
        image_file_hashes: vec![[0u8; 16]; n],
    };
    // A fronto-parallel frame on the plane, in world units at the depth the
    // point stands at.
    set.patch_u_halfvec_xyz = Some(
        Array2::from_shape_vec((1, 3), vec![HALF_EXTENT as f32, 0.0, 0.0])
            .expect("one row of three"),
    );
    set.patch_v_halfvec_xyz = Some(
        Array2::from_shape_vec((1, 3), vec![0.0, HALF_EXTENT as f32, 0.0])
            .expect("one row of three"),
    );
    let mut bitmap = Array4::<u8>::zeros((1, TILE, TILE, 4));
    bitmap[[0, 1, 2, 0]] = 77;
    set.patch_bitmaps_y_x_rgba = Some(Arc::new(bitmap));
    set.observation_confidence = Some(vec![200; observed]);
    set.normal_confidence = Some(vec![180]);
    recon.metadata.feature_source = sfmr_format::FEATURE_SOURCE_EMBEDDED_PATCHES.to_string();
    recon.rebuild_derived_fields();
    recon
}

/// The fixture wrapped as a version with no edits.
fn edited(observed: usize) -> EditedReconstruction {
    EditedReconstruction::new(Arc::new(fixture(observed)))
}

// ── The refusals ─────────────────────────────────────────────────────

#[test]
fn a_point_that_is_not_live_is_refused() {
    let mut value = edited(4);
    value.delete_point(0).expect("a live point");
    let err = remove_observation(&value, 0, 1).expect_err("point 0 was deleted");
    assert_eq!(err, RemoveObservationError::NoSuchPoint(0));
}

#[test]
fn an_image_past_the_table_is_refused() {
    let value = edited(4);
    let err = remove_observation(&value, 0, 9).expect_err("there is no image 9");
    assert_eq!(
        err,
        RemoveObservationError::ImageOutOfRange {
            image: 9,
            image_count: 4
        }
    );
}

#[test]
fn an_image_that_does_not_observe_the_point_is_refused() {
    let value = edited(2);
    let err = remove_observation(&value, 0, 3).expect_err("image 3 is not in the track");
    assert_eq!(err, RemoveObservationError::ImageNotInTrack(3));
    assert_eq!(
        value.point(0).expect("live").observations().len(),
        2,
        "a refusal changed the value"
    );
}

// ── Three or more left: a re-triangulated finite point ───────────────

#[test]
fn removing_one_of_four_re_triangulates_and_keeps_the_frame() {
    let value = edited(4);
    let before = value.point(0).expect("live");
    let (u_before, v_before) = (
        before.patch_u_halfvec().expect("a frame"),
        before.patch_v_halfvec().expect("a frame"),
    );

    let (next, report) = remove_observation(&value, 0, 1).expect("image 1 is in the track");

    assert_eq!(report.replaced, 0);
    assert_eq!(report.image, 1);
    assert_eq!(report.observation_count, 3);
    assert!(!report.deleted && !report.to_infinity);
    assert!(report.retriangulated);
    assert!(report.condition_number.is_finite());

    let after = next
        .point(report.point.expect("a live point"))
        .expect("live");
    assert!(
        (after.point().position - WORLD).norm() < 1e-6,
        "the three remaining rays state the truth, and got {:?}",
        after.point().position
    );
    assert_eq!(after.point().w, 1.0);
    // The images that remain, in order, with their own keypoints.
    let images: Vec<u32> = after.observations().iter().map(|o| o.image_index).collect();
    assert_eq!(images, vec![0, 2, 3]);
    for (k, &image) in images.iter().enumerate() {
        assert_eq!(
            after.keypoint_xy(k).expect("a keypoint"),
            project(&next.base.image_table, image as usize, WORLD)
        );
    }
    // Everything the edit was not asked to change.
    assert_eq!(after.patch_u_halfvec(), Some(u_before));
    assert_eq!(after.patch_v_halfvec(), Some(v_before));
    assert_eq!(after.point().color, [120, 130, 140]);
    assert_eq!(after.point().normal, Vector3::new(0.0, 0.0, -1.0));
    assert_eq!(after.normal_confidence(), Some(180));
}

#[test]
fn the_edit_leaves_the_base_and_the_input_value_alone() {
    let value = edited(4);
    let (next, _) = remove_observation(&value, 0, 1).expect("image 1 is in the track");
    assert!(
        Arc::ptr_eq(&value.base, &next.base),
        "the edit produced a new base"
    );
    assert!(value.added.points.is_empty() && value.deleted_points.is_empty());
    assert_eq!(value.point(0).expect("live").observations().len(), 4);
}

#[test]
fn the_shortened_track_materialises_into_the_place_it_came_from() {
    let value = edited(4);
    let (next, report) = remove_observation(&value, 0, 1).expect("image 1 is in the track");
    let index = report.point.expect("a live point");

    let (plain, map) = next.materialize();
    assert_eq!(map.forward(index), Some(0));
    assert_eq!(plain.point_count(), 1);
    let rows = plain.observations_for_point(0);
    assert_eq!(rows.len(), 3);
    assert_eq!(
        rows.iter().map(|o| o.image_index).collect::<Vec<_>>(),
        vec![0, 2, 3]
    );
    assert_eq!(
        plain.point_set.points[0].position,
        next.point(index).expect("live").point().position,
        "the materialised point is the edited one"
    );
    assert_eq!(
        plain.point_set.observation_confidence.as_deref(),
        Some(&[200u8, 200, 200][..]),
        "the rows that remain keep their per-observation columns"
    );
}

// ── One left: a bearing ──────────────────────────────────────────────

#[test]
fn removing_down_to_one_view_gives_a_bearing_of_the_same_angular_size() {
    let value = edited(2);
    let before = value.point(0).expect("live");
    let u_before = Vector3::from(before.patch_u_halfvec().expect("a frame").map(f64::from));
    let depth = placement_scale(&before.point().position, &value.base.image_table);

    let (next, report) = remove_observation(&value, 0, 1).expect("image 1 is in the track");
    assert!(report.to_infinity);
    assert!(!report.deleted && !report.retriangulated);
    assert_eq!(report.observation_count, 1);

    let after = next
        .point(report.point.expect("a live point"))
        .expect("live");
    assert_eq!(after.point().w, 0.0);
    // The bearing is image 0's own ray, and it is a unit vector.
    let direction = after.point().position.coords;
    assert!((direction.norm() - 1.0).abs() < 1e-12);
    let truth = (WORLD.coords - next.base.image_table.images[0].camera_center().coords).normalize();
    assert!(
        (direction - truth).norm() < 1e-6,
        "the bearing is not image 0's ray to the point: {direction:?} vs {truth:?}"
    );
    assert_eq!(report.position, [direction.x, direction.y, direction.z]);

    // The frame's angular size: the world half-vector divided by the depth the
    // point stood at, which is what makes the patch subtend the angle it did.
    let u_after = Vector3::from(after.patch_u_halfvec().expect("a frame").map(f64::from));
    assert!(
        (u_after.norm() * depth - u_before.norm()).abs() < 1e-6 * u_before.norm(),
        "the angular extent {} does not match {} at depth {depth}",
        u_after.norm(),
        u_before.norm() / depth
    );
    // A bearing states no surface, and its bitmap is still the appearance the
    // point is known by.
    assert_eq!(after.point().normal, Vector3::zeros());
    assert_eq!(after.normal_confidence(), Some(0));
    assert_eq!(
        after.patch_bitmap().expect("a bitmap")[[1, 2, 0]],
        77,
        "the tile changed"
    );
}

// ── None left: the point goes ────────────────────────────────────────

#[test]
fn removing_the_last_view_deletes_the_point() {
    let value = edited(1);
    let (next, report) = remove_observation(&value, 0, 0).expect("image 0 is in the track");
    assert!(report.deleted);
    assert_eq!(report.point, None);
    assert_eq!(report.observation_count, 0);
    assert!(next.point(0).is_none());
    assert_eq!(next.point_count(), 0);
    let (plain, map) = next.materialize();
    assert_eq!(plain.point_count(), 0);
    assert_eq!(map.forward(0), None);
    assert_eq!(value.point_count(), 1, "the input value lost its point");
}

// ── The round trip ───────────────────────────────────────────────────

#[test]
fn adding_an_observation_back_brings_the_point_home() {
    use crate::reconstruction::add_observation::tests::{fixture as add_fixture, Scene, WORLD};
    use crate::reconstruction::add_observation::{add_observation, AddObservationOptions};

    let scene = Scene::new();
    let value = EditedReconstruction::new(Arc::new(add_fixture(&scene, WORLD)));
    let start = value.point(0).expect("live").point().position;

    // Down to one view: a bearing along image 0's ray.
    let (bearing, removed) = remove_observation(&value, 0, 1).expect("image 1 is in the track");
    assert!(removed.to_infinity);
    let point = removed.point.expect("a live point");

    // And back: the second sighting is what fixes a distance again.
    let truth = scene.project(1, WORLD);
    let (finite, added) = add_observation(
        &bearing,
        point,
        1,
        [truth[0] as f32, truth[1] as f32],
        &scene.views(),
        &AddObservationOptions::default(),
    )
    .expect("the patch registers in image 1");

    assert!(added.from_infinity);
    let home = finite.point(added.point).expect("live");
    assert_eq!(home.point().w, 1.0);
    assert!(
        (home.point().position - start).norm() < 0.05,
        "the round trip landed at {:?}, from {start:?}",
        home.point().position
    );
    assert_eq!(home.observations().len(), 2);
}
