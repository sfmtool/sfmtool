// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Moving one camera: the refusals, the four ways a track it observes is
//! settled, and the report.
//!
//! The scene is four pinhole cameras looking down world `+z` at points on a
//! plane, with every stored keypoint the exact projection at a **truth** pose.
//! One camera is then displaced in the fixture, so moving it back to the truth
//! is a move whose right answer is known: every re-triangulated point returns
//! to where the truth solve put it.

use std::sync::Arc;

use nalgebra::{Point3, Quaternion, UnitQuaternion, Vector3};
use ndarray::{Array2, Array4};

use crate::camera::{CameraIntrinsics, CameraModel};
use crate::reconstruction::data::{
    ImageTable, ObservationSource, Point3D, SfmrImage, SfmrReconstruction, TrackObservation,
};
use crate::{RotQuaternion, Se3Transform};

use super::*;

const IMG_W: u32 = 128;
const IMG_H: u32 = 128;
const FOCAL: f64 = 160.0;
/// The camera centres of the truth solve, in world space.
const CENTERS: [[f64; 3]; 4] = [
    [-0.5, -0.3, 0.0],
    [0.45, 0.25, 0.0],
    [0.1, -0.55, 0.0],
    [-0.2, 0.5, 0.0],
];
/// The points every fixture holds, on and around a plane in front of the
/// cameras.
const WORLD: [[f64; 3]; 5] = [
    [0.0, 0.0, 4.0],
    [0.3, -0.2, 3.6],
    [-0.25, 0.15, 4.4],
    [0.1, 0.3, 3.9],
    [-0.15, -0.35, 4.2],
];
/// The bitmap column's tile resolution.
const TILE: usize = 4;
/// The patch frame's half-extent, in world units at the depth its point stands.
const HALF_EXTENT: f32 = 0.12;

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

/// The canonical looking-down-`+z` rotation every truth camera carries: a half
/// turn about `x`, which puts world `+z` on the camera's `-z` forward axis.
fn looking_down_z() -> UnitQuaternion<f64> {
    UnitQuaternion::from_quaternion(Quaternion::new(0.0, 1.0, 0.0, 0.0))
}

/// One image row at `centre` with world-to-camera rotation `rotation`.
fn image_row(name: &str, rotation: UnitQuaternion<f64>, centre: Point3<f64>) -> SfmrImage {
    SfmrImage {
        name: name.to_string(),
        camera_index: 0,
        quaternion_wxyz: rotation,
        translation_xyz: -(rotation * centre.coords),
    }
}

/// Where `world` lands in image `i` of `table`, in source-image px.
fn project(table: &ImageTable, i: usize, world: Point3<f64>) -> [f32; 2] {
    let image = &table.images[i];
    let cam = image.quaternion_wxyz * world.coords + image.translation_xyz;
    let (u, v) = table
        .camera_for_image(i)
        .ray_to_pixel([cam.x, cam.y, cam.z])
        .expect("the point is in front of every camera of this scene");
    [u as f32, v as f32]
}

/// The world-from-camera pose image `i` of `recon` stands at.
fn pose(recon: &SfmrReconstruction, i: usize) -> Se3Transform {
    super::pose_of(recon, i)
}

/// A world-from-camera pose from a rotation and a centre.
fn world_from_camera(rotation: UnitQuaternion<f64>, centre: Point3<f64>) -> Se3Transform {
    Se3Transform::new(
        RotQuaternion::from_nalgebra(rotation.inverse()),
        centre.coords,
        1.0,
    )
}

/// The truth solve: four cameras, [`WORLD`]'s points, every keypoint the exact
/// projection, a patch frame and bitmap per point.
///
/// `at_infinity` names the points stored as bearings; each of those carries the
/// unit direction of its first observation's ray, which is what a bearing is.
/// `observed_by` says which images see each point.
fn truth(observed_by: &[Vec<usize>], at_infinity: &[usize]) -> SfmrReconstruction {
    let n = CENTERS.len();
    let mut recon = SfmrReconstruction::demo(1);
    recon.image_table.cameras = vec![pinhole()];
    recon.image_table.images = (0..n)
        .map(|i| {
            image_row(
                &format!("image_{i}.jpg"),
                looking_down_z(),
                Point3::from(Vector3::from_row_slice(&CENTERS[i])),
            )
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

    let count = observed_by.len();
    let mut tracks = Vec::new();
    let mut keypoints = Vec::new();
    for (p, images) in observed_by.iter().enumerate() {
        for &i in images {
            tracks.push(TrackObservation {
                image_index: i as u32,
                point_index: p as u32,
            });
            let xy = project(
                &recon.image_table,
                i,
                Point3::from(Vector3::from_row_slice(&WORLD[p])),
            );
            keypoints.push(xy[0]);
            keypoints.push(xy[1]);
        }
    }
    let observation_count = tracks.len();

    let set = &mut recon.point_set;
    set.points = (0..count)
        .map(|p| Point3D {
            position: Point3::from(Vector3::from_row_slice(&WORLD[p])),
            w: 1.0,
            color: [120, 130, 140],
            error: 0.5,
            normal: Vector3::new(0.0, 0.0, -1.0),
        })
        .collect();
    set.observation_counts = observed_by.iter().map(|v| v.len() as u32).collect();
    set.tracks = tracks;
    set.observations = ObservationSource::EmbeddedPatches {
        keypoints_xy: Array2::from_shape_vec((observation_count, 2), keypoints)
            .expect("two columns per observation"),
        image_file_hashes: vec![[0u8; 16]; n],
    };
    let mut u = Array2::<f32>::zeros((count, 3));
    let mut v = Array2::<f32>::zeros((count, 3));
    for p in 0..count {
        u[[p, 0]] = HALF_EXTENT;
        v[[p, 1]] = HALF_EXTENT;
    }
    set.patch_u_halfvec_xyz = Some(u);
    set.patch_v_halfvec_xyz = Some(v);
    let mut bitmap = Array4::<u8>::zeros((count, TILE, TILE, 4));
    bitmap[[0, 1, 2, 0]] = 77;
    set.patch_bitmaps_y_x_rgba = Some(Arc::new(bitmap));
    set.observation_confidence = Some(vec![200; observation_count]);
    set.normal_confidence = Some(vec![180; count]);
    recon.metadata.feature_source = sfmr_format::FEATURE_SOURCE_EMBEDDED_PATCHES.to_string();
    recon.rebuild_derived_fields();

    // A bearing carries a unit direction rather than a location: the ray of its
    // first observation, which is exactly what the value says such a point is.
    for &p in at_infinity {
        let start = recon.point_set.observation_offsets[p];
        let row = recon.point_set.tracks[start];
        let xy = recon.point_set.keypoints_xy().expect("inline keypoints");
        let ray = recon
            .image_table
            .world_ray(
                row.image_index as usize,
                [f64::from(xy[[start, 0]]), f64::from(xy[[start, 1]])],
            )
            .expect("a ray through a stored keypoint");
        recon.point_set.points[p].position = Point3::from(ray);
        recon.point_set.points[p].w = 0.0;
        recon.point_set.points[p].normal = Vector3::zeros();
    }
    recon.rebuild_derived_fields();
    recon
}

/// Every point seen by every camera.
fn all_seen(points: usize) -> Vec<Vec<usize>> {
    vec![(0..CENTERS.len()).collect(); points]
}

/// The truth solve with image 0 displaced, which is the value a move puts back.
fn displaced(recon: &SfmrReconstruction) -> SfmrReconstruction {
    let mut out = recon.clone();
    let rotation = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 0.05) * looking_down_z();
    out.image_table.images[0] = image_row(
        &out.image_table.images[0].name.clone(),
        rotation,
        Point3::new(-0.5 + 0.08, -0.3 - 0.04, 0.06),
    );
    out
}

// ── The refusals ─────────────────────────────────────────────────────

#[test]
fn an_image_past_the_table_is_refused() {
    let recon = truth(&all_seen(WORLD.len()), &[]);
    let err = move_camera(&recon, 9, &pose(&recon, 0))
        .map(|_| ())
        .expect_err("there is no image 9");
    assert_eq!(
        err,
        MoveCameraError::ImageOutOfRange {
            image: 9,
            image_count: 4
        }
    );
}

#[test]
fn an_image_with_no_pose_is_refused() {
    let mut recon = truth(&all_seen(WORLD.len()), &[]);
    let target = pose(&recon, 1);
    recon.image_table.images[1].translation_xyz = Vector3::new(f64::NAN, 0.0, 0.0);
    let err = move_camera(&recon, 1, &target)
        .map(|_| ())
        .expect_err("image 1 carries no pose");
    assert_eq!(err, MoveCameraError::NoPose(1));
}

#[test]
fn a_non_finite_pose_is_refused() {
    let recon = truth(&all_seen(WORLD.len()), &[]);
    let broken = Se3Transform::new(
        RotQuaternion::identity(),
        Vector3::new(0.0, f64::INFINITY, 0.0),
        1.0,
    );
    let err = move_camera(&recon, 0, &broken)
        .map(|_| ())
        .expect_err("the centre is not finite");
    assert_eq!(err, MoveCameraError::InvalidPose);
}

#[test]
fn a_rotation_that_is_not_unit_is_refused() {
    let recon = truth(&all_seen(WORLD.len()), &[]);
    // `RotQuaternion::new` normalises, and normalising a zero quaternion leaves
    // a non-finite rotation -- which is the same refusal by the same test.
    let broken = Se3Transform::new(
        RotQuaternion::new(0.0, 0.0, 0.0, 0.0),
        Vector3::zeros(),
        1.0,
    );
    let err = move_camera(&recon, 0, &broken)
        .map(|_| ())
        .expect_err("that rotation is not a rotation");
    assert_eq!(err, MoveCameraError::InvalidPose);
}

// ── The move itself ──────────────────────────────────────────────────

#[test]
fn moving_a_camera_back_to_the_truth_re_triangulates_every_point_onto_it() {
    let truth_recon = truth(&all_seen(WORLD.len()), &[]);
    let wrong = displaced(&truth_recon);
    let target = pose(&truth_recon, 0);

    let (moved, report) = move_camera(&wrong, 0, &target).expect("a finite pose on image 0");

    assert_eq!(report.observed, WORLD.len());
    assert_eq!(report.retriangulated, WORLD.len());
    assert_eq!(report.kept, 0);
    assert_eq!(report.rotated_bearings, 0);
    for (p, expected) in WORLD.iter().enumerate() {
        let got = moved.point_set.points[p].position;
        assert!(
            (got - Point3::from(Vector3::from_row_slice(expected))).norm() < 1e-3,
            "point {p} landed at {got:?}, not at {expected:?}"
        );
    }
    // The pose that came back is the one asked for, in the value's own storage
    // convention.
    let landed = pose(&moved, 0);
    assert!((landed.translation - target.translation).norm() < 1e-12);
    assert!(
        landed
            .rotation
            .as_nalgebra()
            .angle_to(target.rotation.as_nalgebra())
            < 1e-12
    );
}

#[test]
fn the_input_is_untouched() {
    let truth_recon = truth(&all_seen(WORLD.len()), &[]);
    let wrong = displaced(&truth_recon);
    let before = wrong.clone();

    let (_moved, _report) =
        move_camera(&wrong, 0, &pose(&truth_recon, 0)).expect("a finite pose on image 0");

    assert_eq!(
        wrong.image_table.images[0].translation_xyz,
        before.image_table.images[0].translation_xyz
    );
    for p in 0..WORLD.len() {
        assert_eq!(
            wrong.point_set.points[p].position,
            before.point_set.points[p].position
        );
    }
}

#[test]
fn the_counts_account_for_every_observed_track() {
    let mut observed_by = all_seen(WORLD.len());
    observed_by[1] = vec![0]; // a bearing image 0 alone sees
    observed_by[2] = vec![0, 2]; // a bearing two images see
    let recon = truth(&observed_by, &[1, 2]);
    let wrong = displaced(&recon);

    let (_moved, report) = move_camera(&wrong, 0, &pose(&recon, 0)).expect("a finite pose");

    assert_eq!(report.observed, WORLD.len());
    assert_eq!(
        report.retriangulated + report.kept + report.rotated_bearings,
        report.observed
    );
    assert_eq!(report.rotated_bearings, 1);
}

#[test]
fn a_single_view_bearing_becomes_the_moved_cameras_ray() {
    let mut observed_by = all_seen(WORLD.len());
    observed_by[1] = vec![0];
    let recon = truth(&observed_by, &[1]);
    let wrong = displaced(&recon);
    let target = pose(&recon, 0);

    let (moved, report) = move_camera(&wrong, 0, &target).expect("a finite pose");

    assert_eq!(report.rotated_bearings, 1);
    let start = moved.point_set.observation_offsets[1];
    let xy = moved.point_set.keypoints_xy().expect("inline keypoints");
    let expected = moved
        .image_table
        .world_ray(0, [f64::from(xy[[start, 0]]), f64::from(xy[[start, 1]])])
        .expect("a ray through the stored keypoint");
    let got = moved.point_set.points[1].position;
    assert!((got.coords - expected).norm() < 1e-12, "got {got:?}");
    assert_eq!(moved.point_set.points[1].w, 0.0);
}

#[test]
fn a_bearing_two_images_see_keeps_its_direction() {
    let mut observed_by = all_seen(WORLD.len());
    observed_by[2] = vec![0, 2];
    let recon = truth(&observed_by, &[2]);
    let wrong = displaced(&recon);
    let before = wrong.point_set.points[2].position;

    let (moved, report) = move_camera(&wrong, 0, &pose(&recon, 0)).expect("a finite pose");

    assert_eq!(moved.point_set.points[2].position, before);
    assert_eq!(moved.point_set.points[2].w, 0.0);
    assert_eq!(report.rotated_bearings, 0);
    assert!(report.kept >= 1);
}

#[test]
fn a_value_with_no_keypoints_moves_its_camera_and_keeps_every_point() {
    let mut recon = truth(&all_seen(WORLD.len()), &[]);
    let observation_count = recon.point_set.tracks.len();
    let image_count = recon.image_table.images.len();
    recon.point_set.observations = ObservationSource::SiftFiles {
        feature_indexes: (0..observation_count as u32).collect(),
        keypoints_xy: None,
        feature_tool_hashes: vec![[0u8; 16]; image_count],
        sift_content_hashes: vec![[0u8; 16]; image_count],
    };
    recon.metadata.feature_source = sfmr_format::FEATURE_SOURCE_SIFT_FILES.to_string();
    recon.rebuild_derived_fields();
    let target = world_from_camera(looking_down_z(), Point3::new(0.9, 0.9, 0.2));

    let (moved, report) = move_camera(&recon, 0, &target).expect("a finite pose");

    assert_eq!(report.kept, report.observed);
    assert_eq!(report.retriangulated, 0);
    assert_eq!(report.residual_before_px, None);
    assert_eq!(report.residual_after_px, None);
    for p in 0..WORLD.len() {
        assert_eq!(
            moved.point_set.points[p].position,
            recon.point_set.points[p].position
        );
    }
    assert!((pose(&moved, 0).translation - target.translation).norm() < 1e-12);
}

#[test]
fn a_track_that_will_not_re_triangulate_is_kept_and_counted() {
    let recon = truth(&all_seen(WORLD.len()), &[]);
    // Turned to face away, every ray from image 0 puts its point behind that
    // camera, which is one of the three signals a triangulation is refused on.
    let backwards = world_from_camera(
        UnitQuaternion::from_axis_angle(&Vector3::y_axis(), std::f64::consts::PI)
            * looking_down_z(),
        Point3::new(-0.5, -0.3, 0.0),
    );

    let (moved, report) = move_camera(&recon, 0, &backwards).expect("a finite pose");

    assert_eq!(report.kept, report.observed);
    assert_eq!(report.retriangulated, 0);
    for p in 0..WORLD.len() {
        assert_eq!(
            moved.point_set.points[p].position,
            recon.point_set.points[p].position
        );
    }
}

#[test]
fn a_re_triangulated_points_frame_is_rescaled_by_its_depth_ratio() {
    let truth_recon = truth(&all_seen(WORLD.len()), &[]);
    let wrong = displaced(&truth_recon);
    let before_position = wrong.point_set.points[0].position;
    let before_scale = wrong.image_table.placement_scale(&before_position);

    let (moved, _report) =
        move_camera(&wrong, 0, &pose(&truth_recon, 0)).expect("a finite pose on image 0");

    let after_scale = moved
        .image_table
        .placement_scale(&moved.point_set.points[0].position);
    let expected = HALF_EXTENT * (after_scale / before_scale) as f32;
    let u = moved
        .point_set
        .patch_u_halfvec_xyz
        .as_ref()
        .expect("frames");
    assert!(
        (u[[0, 0]] - expected).abs() < 1e-6,
        "the frame is {}, and the depth ratio asks for {expected}",
        u[[0, 0]]
    );
}

#[test]
fn the_error_column_is_rewritten_at_the_new_geometry() {
    let truth_recon = truth(&all_seen(WORLD.len()), &[]);
    let wrong = displaced(&truth_recon);
    assert_eq!(wrong.point_set.points[0].error, 0.5);

    let (moved, _report) =
        move_camera(&wrong, 0, &pose(&truth_recon, 0)).expect("a finite pose on image 0");

    // Every keypoint is the truth projection, so the truth pose reconciles them
    // all and the RMS falls to the quantisation of the stored f32 pixels.
    assert!(
        moved.point_set.points[0].error < 1e-2,
        "the error column is {}",
        moved.point_set.points[0].error
    );
}

#[test]
fn the_report_measures_the_residual_before_and_after() {
    let truth_recon = truth(&all_seen(WORLD.len()), &[]);
    let wrong = displaced(&truth_recon);

    let (moved, report) =
        move_camera(&wrong, 0, &pose(&truth_recon, 0)).expect("a finite pose on image 0");

    let before = report.residual_before_px.expect("inline keypoints");
    let after = report.residual_after_px.expect("inline keypoints");
    assert!(before[0] > 1.0, "the displaced pose disagrees: {before:?}");
    assert!(after[0] < 1e-2, "the truth pose reconciles: {after:?}");
    assert!(
        before[1] >= before[0] && after[1] >= after[0],
        "p90 below the median"
    );
    // The same numbers a caller can compute for itself from the value that came
    // back, which is what the viewer's own readout does every frame.
    let direct = residual_quantiles_px(
        moved.image_table.camera_for_image(0),
        &pose(&moved, 0),
        &image_reprojection_samples(&moved, 0),
    )
    .expect("inline keypoints");
    assert_eq!(direct, after);
}

#[test]
fn the_report_measures_the_move_itself() {
    let truth_recon = truth(&all_seen(WORLD.len()), &[]);
    let wrong = displaced(&truth_recon);
    let target = pose(&truth_recon, 0);

    let (_moved, report) = move_camera(&wrong, 0, &target).expect("a finite pose on image 0");

    let expected_translation =
        (Point3::from(target.translation) - wrong.image_table.images[0].camera_center()).norm();
    assert!((report.translation - expected_translation).abs() < 1e-12);
    assert!((report.rotation_deg - 0.05_f64.to_degrees()).abs() < 1e-9);
    let scene = report
        .translation_scene
        .expect("finite structure has a scale");
    assert!(scene > 0.0 && scene.is_finite());
}

#[test]
fn a_rotation_only_value_reports_no_scene_units() {
    let mut recon = truth(&all_seen(WORLD.len()), &[]);
    for point in &mut recon.point_set.points {
        point.w = 0.0;
        point.position = Point3::from(point.position.coords.normalize());
    }
    recon.rebuild_derived_fields();

    let (_moved, report) = move_camera(&recon, 0, &pose(&recon, 0)).expect("a finite pose");

    assert_eq!(report.translation_scene, None);
}

#[test]
fn the_edited_gatherer_agrees_with_the_plain_one() {
    let recon = truth(&all_seen(WORLD.len()), &[]);
    let edited = crate::EditedReconstruction::new(Arc::new(recon.clone()));

    let plain = image_reprojection_samples(&recon, 2);
    let over_overlay = edited_image_reprojection_samples(&edited, 2);

    assert_eq!(plain.len(), over_overlay.len());
    assert!(!plain.is_empty());
    for sample in &plain {
        assert!(over_overlay.contains(sample), "missing {sample:?}");
    }
}
