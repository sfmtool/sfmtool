// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Tests for [`super::resect_images`].
//!
//! The synthetic fixtures are `embedded_patches` reconstructions so the 2D
//! observations are inline: the geometry under test is the hold-out and the
//! estimate, not the `.sift` reader that feeds them on the other source.
//!
//! Two `#[ignore]`d tests at the bottom run the whole thing against real
//! candidate solves on this machine. They are the end-to-end evidence that the
//! finite and rotation-only paths do what the spec says on files nobody
//! constructed for them, and they are ignored by default because the data lives
//! outside the repository.

use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;

use nalgebra::{Matrix3, Point3, Rotation3, UnitQuaternion, Vector3};
use ndarray::{Array2, Array4};

use sfmr_format::{
    ContentHash, DepthStatistics, SfmrMetadata, FEATURE_SOURCE_EMBEDDED_PATCHES,
    FEATURE_SOURCE_SIFT_FILES,
};

use crate::camera::{CameraIntrinsics, CameraModel};
use crate::reconstruction::{
    ObservationSource, Point3D, SfmrImage, SfmrReconstruction, TrackObservation,
};

use super::{
    resect_images, ResectImageError, ResectImageOptions, ResectImageReport, ResectSource,
    MIN_OTHER_POSED_IMAGES,
};

/// The outcome for a single target: a one-element [`resect_images`] call
/// unpacked, which is what the single-image tests below read.
struct SingleResection {
    reconstruction: SfmrReconstruction,
    report: ResectImageReport,
}

/// The report alone: an [`SfmrReconstruction`] is not `Debug`, and a failing
/// assertion wants to see what the estimate did.
impl std::fmt::Debug for SingleResection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SingleResection")
            .field("report", &self.report)
            .finish_non_exhaustive()
    }
}

/// [`resect_images`] on a one-element set. Most of what there is to test about
/// the mechanism is visible with one image held out; the tests that need a set
/// call [`resect_images`] directly.
fn resect_image(
    recon: &SfmrReconstruction,
    image_index: usize,
    source: ResectSource<'_>,
    options: &ResectImageOptions,
) -> Result<SingleResection, ResectImageError> {
    let out = resect_images(recon, &[image_index], source, options)?;
    assert_eq!(out.reports.len(), 1);
    assert_eq!(out.totals.targets, 1);
    Ok(SingleResection {
        reconstruction: out.reconstruction,
        report: out.reports.into_iter().next().expect("one report"),
    })
}

// ── Fixtures ────────────────────────────────────────────────────────────────

/// A pinhole of the given focal, 640x480. The dome fixture wants a wide one:
/// bearings are only shared between cameras whose fields of view overlap, and a
/// narrow lens leaves every direction seen once.
fn camera(focal: f64) -> CameraIntrinsics {
    CameraIntrinsics {
        model: CameraModel::Pinhole {
            focal_length_x: focal,
            focal_length_y: focal,
            principal_point_x: 320.0,
            principal_point_y: 240.0,
        },
        width: 640,
        height: 480,
    }
}

/// World-to-camera rotation of a camera at `eye` looking at `at`, in the
/// canonical convention (−Z forward, +Y up) the rest of the crate uses.
fn look_at(eye: Point3<f64>, at: Point3<f64>) -> UnitQuaternion<f64> {
    let forward = (at - eye).normalize();
    let world_up = Vector3::z();
    let right = forward.cross(&world_up).normalize();
    let up = right.cross(&forward);
    let m = Matrix3::new(
        right.x, right.y, right.z, up.x, up.y, up.z, -forward.x, -forward.y, -forward.z,
    );
    UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(m))
}

fn image_at(name: &str, eye: Point3<f64>, at: Point3<f64>) -> SfmrImage {
    let rotation = look_at(eye, at);
    SfmrImage {
        name: name.to_string(),
        camera_index: 0,
        quaternion_wxyz: rotation,
        translation_xyz: rotation * (-eye.coords),
    }
}

/// A ring of `count` cameras on a circle of radius `radius` about the origin,
/// each looking at it.
fn ring(count: usize, radius: f64) -> Vec<SfmrImage> {
    (0..count)
        .map(|i| {
            let angle = i as f64 * std::f64::consts::TAU / count as f64;
            let eye = Point3::new(radius * angle.cos(), radius * angle.sin(), 0.4 * i as f64);
            image_at(&format!("frames/{i:03}.jpg"), eye, Point3::origin())
        })
        .collect()
}

/// A deterministic scatter of `count` points inside a unit ball, as a
/// low-discrepancy lattice rather than a random draw (a test that only passes
/// for one seed is a test of the seed).
fn cloud(count: usize) -> Vec<Point3<f64>> {
    (0..count)
        .map(|i| {
            let t = (i as f64 + 0.5) / count as f64;
            let phi = t * std::f64::consts::PI;
            let theta = i as f64 * 2.399_963_229_728_653;
            let r = 0.4 + 0.6 * ((i % 7) as f64 / 7.0);
            Point3::new(
                r * phi.sin() * theta.cos(),
                r * phi.sin() * theta.sin(),
                r * phi.cos(),
            )
        })
        .collect()
}

/// Project `world` into `image`, or `None` when it lands behind the camera or
/// outside the frame.
fn project(
    camera: &CameraIntrinsics,
    image: &SfmrImage,
    world: &Point3<f64>,
    at_infinity: bool,
) -> Option<[f32; 2]> {
    let local = if at_infinity {
        image.quaternion_wxyz * world.coords
    } else {
        image.quaternion_wxyz * world.coords + image.translation_xyz
    };
    let (u, v) = camera.ray_to_pixel([local.x, local.y, local.z])?;
    (u >= 0.0 && v >= 0.0 && u < camera.width as f64 && v < camera.height as f64)
        .then_some([u as f32, v as f32])
}

/// An `embedded_patches` reconstruction of `images` observing `points`, with
/// every observation the camera model actually admits.
///
/// `at_infinity` builds the rotation-only twin: the points are unit bearings
/// with `w = 0`, projected translation-free.
fn build(
    images: Vec<SfmrImage>,
    positions: Vec<Point3<f64>>,
    at_infinity: bool,
    focal: f64,
) -> SfmrReconstruction {
    let camera = camera(focal);
    let mut points = Vec::new();
    let mut tracks = Vec::new();
    let mut observation_counts = Vec::new();
    let mut keypoints: Vec<[f32; 2]> = Vec::new();
    for position in &positions {
        let position = if at_infinity {
            Point3::from(position.coords.normalize())
        } else {
            *position
        };
        let mut rows = Vec::new();
        for (i, image) in images.iter().enumerate() {
            if let Some(uv) = project(&camera, image, &position, at_infinity) {
                rows.push((i as u32, uv));
            }
        }
        if rows.len() < 2 {
            continue;
        }
        let point_index = points.len() as u32;
        for (image_index, uv) in rows {
            tracks.push(TrackObservation {
                image_index,
                point_index,
            });
            keypoints.push(uv);
        }
        observation_counts.push(
            tracks
                .iter()
                .filter(|t| t.point_index == point_index)
                .count() as u32,
        );
        points.push(Point3D {
            position,
            w: if at_infinity { 0.0 } else { 1.0 },
            color: [128, 128, 128],
            error: 0.0,
            normal: Vector3::zeros(),
        });
    }

    let n_images = images.len();
    let mut keypoints_xy = Array2::<f32>::zeros((keypoints.len(), 2));
    for (row, uv) in keypoints.iter().enumerate() {
        keypoints_xy[[row, 0]] = uv[0];
        keypoints_xy[[row, 1]] = uv[1];
    }
    let metadata = SfmrMetadata {
        version: 6,
        operation: "test".into(),
        tool: "sfmtool".into(),
        tool_version: "0".into(),
        tool_options: BTreeMap::new(),
        workspace: sfmr_format::WorkspaceMetadata {
            absolute_path: String::new(),
            relative_path: ".".into(),
            contents: sfmr_format::WorkspaceContents {
                feature_tool: "none".into(),
                feature_type: "sift".into(),
                feature_options: serde_json::json!({}),
                feature_prefix_dir: String::new(),
            },
        },
        timestamp: String::new(),
        image_count: n_images as u32,
        point_count: points.len() as u32,
        infinity_point_count: if at_infinity { points.len() as u32 } else { 0 },
        observation_count: tracks.len() as u32,
        camera_count: 1,
        rig_count: None,
        sensor_count: None,
        frame_count: None,
        world_space_unit: None,
        feature_source: FEATURE_SOURCE_EMBEDDED_PATCHES.to_string(),
    };
    let mut recon = SfmrReconstruction {
        workspace_dir: PathBuf::new(),
        metadata,
        content_hash: ContentHash {
            metadata_xxh128: String::new(),
            cameras_xxh128: String::new(),
            rigs_xxh128: None,
            frames_xxh128: None,
            images_xxh128: String::new(),
            points3d_xxh128: String::new(),
            tracks_xxh128: String::new(),
            content_xxh128: String::new(),
        },
        cameras: vec![camera],
        images,
        points,
        tracks,
        observation_counts,
        thumbnails_y_x_rgb: Array4::zeros((n_images, 1, 1, 3)),
        depth_statistics: DepthStatistics {
            num_histogram_buckets: 0,
            images: Vec::new(),
        },
        depth_histogram_counts: Vec::new(),
        rig_frame_data: None,
        patch_u_halfvec_xyz: None,
        patch_v_halfvec_xyz: None,
        patch_bitmaps_y_x_rgba: None,
        has_normals: false,
        normal_confidence: None,
        observation_confidence: None,
        observations: ObservationSource::EmbeddedPatches {
            keypoints_xy,
            image_file_hashes: vec![[0u8; 16]; n_images],
        },
        observation_offsets: Vec::new(),
        image_feature_to_point: Vec::new(),
        max_track_feature_index: Vec::new(),
        infinity_point_count: 0,
    };
    recon.rebuild_derived_fields();
    recon
}

/// A well-conditioned finite fixture: eight ring cameras over a 200-point ball.
fn orbit() -> SfmrReconstruction {
    build(ring(8, 4.0), cloud(200), false, 800.0)
}

/// Angle between two world-to-camera rotations, degrees.
fn angle_deg(a: &UnitQuaternion<f64>, b: &UnitQuaternion<f64>) -> f64 {
    a.rotation_to(b).angle().to_degrees()
}

/// Rewrite one image's inline keypoints, so a test can corrupt exactly the
/// observations the hold-out is supposed to be blind to.
fn corrupt_observations(recon: &mut SfmrReconstruction, image_index: usize) {
    let rows: Vec<usize> = recon
        .tracks
        .iter()
        .enumerate()
        .filter(|(_, t)| t.image_index as usize == image_index)
        .map(|(row, _)| row)
        .collect();
    let ObservationSource::EmbeddedPatches { keypoints_xy, .. } = &mut recon.observations else {
        unreachable!("the fixtures are embedded_patches");
    };
    for (n, row) in rows.into_iter().enumerate() {
        keypoints_xy[[row, 0]] = (n % 640) as f32;
        keypoints_xy[[row, 1]] = ((n * 7) % 480) as f32;
    }
}

// ── The finite path ─────────────────────────────────────────────────────────

#[test]
fn a_perturbed_pose_is_recovered_from_held_out_structure() {
    let truth = orbit();
    let target = 0;
    let true_pose = truth.images[target].clone();

    let mut source = truth.clone();
    // A pose that is wrong by tens of degrees and a substantial fraction of the
    // scene: the disagreement this feature exists to show.
    let spin = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 0.35);
    source.images[target].quaternion_wxyz = spin * true_pose.quaternion_wxyz;
    source.images[target].translation_xyz =
        true_pose.translation_xyz + Vector3::new(0.6, -0.3, 0.2);

    let out = resect_image(
        &source,
        target,
        ResectSource::StoredObservations,
        &ResectImageOptions::default(),
    )
    .expect("the orbit has support");
    assert!(out.report.accepted, "refused: {:?}", out.report.refusal);
    assert!(out.report.correspondences >= 100);

    let fitted = &out.reconstruction.images[target];
    assert!(
        angle_deg(&fitted.quaternion_wxyz, &true_pose.quaternion_wxyz) < 0.1,
        "rotation off by {}",
        angle_deg(&fitted.quaternion_wxyz, &true_pose.quaternion_wxyz)
    );
    assert!(
        (fitted.camera_center() - true_pose.camera_center()).norm() < 0.01,
        "centre off by {}",
        (fitted.camera_center() - true_pose.camera_center()).norm()
    );
    // The report describes the move away from the *stored* pose, which is the
    // corrupted one.
    assert!(out.report.rotation_deg > 15.0);
    assert!(out.report.translation > 0.5);
    assert!(out.report.translation_scene.is_some());
    assert!(out.report.retriangulated > 0);
    assert_eq!(out.report.source, "observations");
    assert_eq!(out.reconstruction.metadata.operation, "explorer_resect");

    // The source is untouched under every outcome.
    assert_eq!(
        source.images[target].quaternion_wxyz,
        spin * true_pose.quaternion_wxyz
    );
}

#[test]
fn the_hold_out_never_reads_the_targets_own_observations() {
    let truth = orbit();
    let target = 0;
    let mut source = truth.clone();
    // Junk observations *and* a junk pose. Nothing the target says is usable,
    // so the estimate must be refused — and the held-out positions, which come
    // from the other seven cameras alone, must still be the truth.
    corrupt_observations(&mut source, target);
    source.images[target].translation_xyz += Vector3::new(3.0, 3.0, 3.0);

    let out = resect_image(
        &source,
        target,
        ResectSource::StoredObservations,
        &ResectImageOptions::default(),
    )
    .expect("the other cameras still supply support");
    assert!(!out.report.accepted, "junk observations should not resect");
    assert!(out.report.refusal.is_some());
    assert_eq!(out.report.retriangulated, 0);
    assert!(out.report.held_out_points > 100);

    let worst = out
        .reconstruction
        .points
        .iter()
        .zip(&truth.points)
        .map(|(a, b)| (a.position - b.position).norm())
        .fold(0.0, f64::max);
    // Not zero: the fixture stores its keypoints as `f32`, so a position
    // re-triangulated from them lands within the quantization of a pixel of the
    // truth. What matters is that corrupting the target moved nothing.
    assert!(worst < 1e-5, "held-out positions drifted by {worst}");
}

#[test]
fn the_same_input_gives_a_bit_identical_answer() {
    let source = orbit();
    let options = ResectImageOptions::default();
    let one = resect_image(&source, 2, ResectSource::StoredObservations, &options).unwrap();
    let two = resect_image(&source, 2, ResectSource::StoredObservations, &options).unwrap();
    assert_eq!(
        one.reconstruction.images[2].quaternion_wxyz,
        two.reconstruction.images[2].quaternion_wxyz
    );
    assert_eq!(
        one.reconstruction.images[2].translation_xyz,
        two.reconstruction.images[2].translation_xyz
    );
    for (a, b) in one
        .reconstruction
        .points
        .iter()
        .zip(&two.reconstruction.points)
    {
        assert_eq!(a.position, b.position);
    }
    assert_eq!(one.report.inlier_fraction, two.report.inlier_fraction);
}

// ── The set ─────────────────────────────────────────────────────────────────

/// Perturb one image of `recon` by a rotation about `axis` and a fixed shift.
fn perturbed(mut recon: SfmrReconstruction, target: usize, angle: f64) -> SfmrReconstruction {
    let spin = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), angle);
    recon.images[target].quaternion_wxyz = spin * recon.images[target].quaternion_wxyz;
    recon.images[target].translation_xyz += Vector3::new(0.6, -0.3, 0.2);
    recon
}

#[test]
fn two_targets_held_out_together_both_recover() {
    let truth = orbit();
    let targets = [0usize, 1usize];
    let true_poses: Vec<SfmrImage> = targets.iter().map(|&t| truth.images[t].clone()).collect();

    let mut source = truth.clone();
    source = perturbed(source, targets[0], 0.35);
    source = perturbed(source, targets[1], -0.28);

    let out = resect_images(
        &source,
        &targets,
        ResectSource::StoredObservations,
        &ResectImageOptions::default(),
    )
    .expect("six non-target cameras still carry the scene");

    assert_eq!(out.reports.len(), 2);
    assert_eq!(out.totals.targets, 2);
    assert_eq!(out.totals.accepted, 2, "{:?}", out.reports);
    assert_eq!(out.totals.refused, 0);
    assert_eq!(
        out.totals.correspondences,
        out.reports.iter().map(|r| r.correspondences).sum::<usize>()
    );
    for (report, truth_pose) in out.reports.iter().zip(&true_poses) {
        assert_eq!(report.image_index, truth_pose_index(&truth, truth_pose));
        let fitted = &out.reconstruction.images[report.image_index];
        assert!(
            angle_deg(&fitted.quaternion_wxyz, &truth_pose.quaternion_wxyz) < 0.1,
            "{} rotation off by {}",
            report.image_name,
            angle_deg(&fitted.quaternion_wxyz, &truth_pose.quaternion_wxyz)
        );
        assert!(
            (fitted.camera_center() - truth_pose.camera_center()).norm() < 0.01,
            "{} centre off by {}",
            report.image_name,
            (fitted.camera_center() - truth_pose.camera_center()).norm()
        );
        assert!(
            report.rotation_deg > 10.0,
            "the report is off the stored pose"
        );
    }

    // The source is untouched under every outcome.
    for &t in &targets {
        assert_ne!(
            source.images[t].quaternion_wxyz,
            truth.images[t].quaternion_wxyz
        );
    }
}

#[test]
fn the_hold_out_ignores_every_target_not_just_one() {
    // Both targets' observations are junk *and* both poses are junk. A hold-out
    // that dropped only the image being estimated would read the other target's
    // corrupted rows and place the shared points wrong; one that drops the whole
    // set reads the six honest cameras and lands on the truth.
    let truth = orbit();
    let targets = [0usize, 1usize];
    let mut source = truth.clone();
    for &t in &targets {
        corrupt_observations(&mut source, t);
        source.images[t].translation_xyz += Vector3::new(3.0, 3.0, 3.0);
    }

    let out = resect_images(
        &source,
        &targets,
        ResectSource::StoredObservations,
        &ResectImageOptions::default(),
    )
    .expect("the other six cameras still supply support");

    assert_eq!(
        out.totals.accepted, 0,
        "junk observations should not resect"
    );
    assert_eq!(out.totals.retriangulated, 0);
    assert!(out.totals.held_out_points > 100);
    for report in &out.reports {
        assert!(!report.accepted);
        assert!(report.refusal.is_some());
    }
    // The fixture's points are all seen by every camera, so the hold-out places
    // every one of them and nothing is dropped.
    assert_eq!(out.totals.removed_points, 0);

    let worst = out
        .reconstruction
        .points
        .iter()
        .zip(&truth.points)
        .map(|(a, b)| (a.position - b.position).norm())
        .fold(0.0, f64::max);
    // Not zero: the fixture stores its keypoints as `f32`, so a position
    // re-triangulated from them lands within the quantization of a pixel of the
    // truth. What matters is that corrupting the targets moved nothing.
    assert!(worst < 1e-5, "held-out positions drifted by {worst}");
}

#[test]
fn an_empty_target_set_is_refused() {
    let source = orbit();
    let err = resect_images(
        &source,
        &[],
        ResectSource::StoredObservations,
        &ResectImageOptions::default(),
    )
    .unwrap_err();
    assert!(matches!(err, ResectImageError::NoTargets), "{err}");
}

#[test]
fn a_target_named_twice_is_refused() {
    let source = orbit();
    let err = resect_images(
        &source,
        &[2, 2],
        ResectSource::StoredObservations,
        &ResectImageOptions::default(),
    )
    .unwrap_err();
    assert!(matches!(err, ResectImageError::DuplicateTarget(2)), "{err}");
}

#[test]
fn a_set_that_leaves_too_few_posed_images_is_refused() {
    // Five images, three of them held out: two are left, below the floor.
    let source = build(ring(5, 4.0), cloud(120), false, 800.0);
    let err = resect_images(
        &source,
        &[0, 1, 2],
        ResectSource::StoredObservations,
        &ResectImageOptions::default(),
    )
    .unwrap_err();
    assert!(
        matches!(err, ResectImageError::TooFewPosedImages(2)),
        "{err}"
    );
}

/// The index of the image `pose` came from, by name.
fn truth_pose_index(recon: &SfmrReconstruction, pose: &SfmrImage) -> usize {
    recon
        .images
        .iter()
        .position(|i| i.name == pose.name)
        .expect("the pose is one of the reconstruction's images")
}

// ── The rotation-only path ──────────────────────────────────────────────────

/// A rotation-only fixture: cameras that share a centre and differ only in
/// where they point, over a sky of bearings.
fn dome() -> SfmrReconstruction {
    let images: Vec<SfmrImage> = (0..8)
        .map(|i| {
            let yaw = i as f64 * std::f64::consts::TAU / 8.0;
            let at = Point3::new(yaw.cos(), yaw.sin(), 0.2 * (i as f64 - 3.5));
            image_at(&format!("frames/{i:03}.jpg"), Point3::origin(), at)
        })
        .collect();
    build(images, cloud(400), true, 250.0)
}

#[test]
fn a_rotation_only_reconstruction_recovers_a_perturbed_rotation() {
    let truth = dome();
    let target = 0;
    let true_rotation = truth.images[target].quaternion_wxyz;
    let stored_translation = truth.images[target].translation_xyz;

    let mut source = truth.clone();
    source.images[target].quaternion_wxyz =
        UnitQuaternion::from_axis_angle(&Vector3::x_axis(), 0.2) * true_rotation;

    let out = resect_image(
        &source,
        target,
        ResectSource::StoredObservations,
        &ResectImageOptions::default(),
    )
    .expect("the dome has bearings");
    assert!(out.report.rotation_only);
    assert!(out.report.accepted, "refused: {:?}", out.report.refusal);
    assert!(out.report.scene_scale.is_none());
    assert_eq!(out.report.held_out_points, 0);

    let fitted = &out.reconstruction.images[target];
    assert!(
        angle_deg(&fitted.quaternion_wxyz, &true_rotation) < 0.05,
        "rotation off by {}",
        angle_deg(&fitted.quaternion_wxyz, &true_rotation)
    );
    // The translation is left exactly as it was found.
    assert_eq!(fitted.translation_xyz, stored_translation);
}

// ── Refusals ────────────────────────────────────────────────────────────────

#[test]
fn an_out_of_range_image_is_refused() {
    let source = orbit();
    let count = source.images.len();
    let err = resect_image(
        &source,
        count,
        ResectSource::StoredObservations,
        &ResectImageOptions::default(),
    )
    .unwrap_err();
    assert!(matches!(
        err,
        ResectImageError::ImageOutOfRange { index, count: c } if index == count && c == count
    ));
}

#[test]
fn a_reconstruction_with_too_few_other_images_is_refused() {
    let source = build(ring(MIN_OTHER_POSED_IMAGES, 4.0), cloud(120), false, 800.0);
    let err = resect_image(
        &source,
        0,
        ResectSource::StoredObservations,
        &ResectImageOptions::default(),
    )
    .unwrap_err();
    assert!(
        matches!(err, ResectImageError::TooFewPosedImages(n) if n == MIN_OTHER_POSED_IMAGES - 1),
        "{err}"
    );
    assert!(err.to_string().contains("non-target posed image"));
}

#[test]
fn too_few_held_out_points_and_no_bearings_is_refused() {
    let mut source = orbit();
    // Strip every observation of the target's points that belongs to another
    // image: nothing is left to hold out against.
    let target = 0;
    let mine: std::collections::HashSet<u32> = source
        .tracks
        .iter()
        .filter(|t| t.image_index as usize == target)
        .map(|t| t.point_index)
        .collect();
    let keep: Vec<bool> = (0..source.points.len())
        .map(|p| mine.contains(&(p as u32)))
        .collect();
    source = source.filter_points_by_mask(&keep);
    let rows: Vec<usize> = source
        .tracks
        .iter()
        .enumerate()
        .filter(|(_, t)| t.image_index as usize == target)
        .map(|(row, _)| row)
        .collect();
    source = drop_all_but(source, &rows);

    let out = resect_image(
        &source,
        target,
        ResectSource::StoredObservations,
        &ResectImageOptions::default(),
    )
    .expect("the call itself is well formed");
    // Neither path has support, which is this image's refusal rather than the
    // call's failure: the derived reconstruction still stands.
    assert!(!out.report.accepted);
    let refusal = out.report.refusal.expect("a reason");
    assert!(
        refusal.contains("no support") && refusal.contains("0 bearings"),
        "{refusal}"
    );
}

#[test]
fn bearings_that_span_no_angle_are_refused() {
    // A dome whose points are all one direction: three bearings, no spread, no
    // rotation determined by them.
    let images: Vec<SfmrImage> = (0..5)
        .map(|i| {
            image_at(
                &format!("frames/{i:03}.jpg"),
                Point3::origin(),
                Point3::new(1.0, 0.02 * i as f64, 0.0),
            )
        })
        .collect();
    let one = Vector3::new(1.0, 0.0, 0.0);
    let positions: Vec<Point3<f64>> = (0..8)
        .map(|i| Point3::from(one + Vector3::new(0.0, 1e-9 * i as f64, 0.0)))
        .collect();
    let source = build(images, positions, true, 250.0);
    assert!(source.points.len() >= super::MIN_BEARINGS);

    let out = resect_image(
        &source,
        0,
        ResectSource::StoredObservations,
        &ResectImageOptions::default(),
    )
    .expect("the call itself is well formed");
    assert!(!out.report.accepted);
    let refusal = out.report.refusal.expect("a reason");
    assert!(refusal.contains("span no measurable angle"), "{refusal}");
}

#[test]
fn an_unposed_target_is_refused() {
    let mut source = orbit();
    source.images[1].translation_xyz = Vector3::new(f64::NAN, 0.0, 0.0);
    let err = resect_image(
        &source,
        1,
        ResectSource::StoredObservations,
        &ResectImageOptions::default(),
    )
    .unwrap_err();
    assert!(matches!(err, ResectImageError::NotPosed(1)), "{err}");
}

#[test]
fn a_matches_join_needs_feature_indexes() {
    // The synthetic fixtures are embedded_patches, which is exactly the case the
    // matches source cannot serve — and it says so rather than joining nothing.
    let source = orbit();
    let empty = matches_fixture();
    let err = resect_image(
        &source,
        0,
        ResectSource::Matches(&empty),
        &ResectImageOptions::default(),
    )
    .unwrap_err();
    assert!(matches!(err, ResectImageError::Matches(_)), "{err}");
    assert!(err.to_string().contains("sift_files"));
}

/// Keep only `rows` of the reconstruction's observations, rebuilding the
/// derived indexes. Used to starve a target of held-out support.
fn drop_all_but(mut recon: SfmrReconstruction, rows: &[usize]) -> SfmrReconstruction {
    let set: std::collections::HashSet<usize> = rows.iter().copied().collect();
    let kept: Vec<usize> = (0..recon.tracks.len())
        .filter(|r| set.contains(r))
        .collect();
    let tracks: Vec<TrackObservation> = kept.iter().map(|&r| recon.tracks[r]).collect();
    let mut counts = vec![0u32; recon.points.len()];
    for track in &tracks {
        counts[track.point_index as usize] += 1;
    }
    let ObservationSource::EmbeddedPatches {
        keypoints_xy,
        image_file_hashes,
    } = &recon.observations
    else {
        unreachable!("the fixtures are embedded_patches");
    };
    recon.observations = ObservationSource::EmbeddedPatches {
        keypoints_xy: keypoints_xy.select(ndarray::Axis(0), &kept),
        image_file_hashes: image_file_hashes.clone(),
    };
    recon.tracks = tracks;
    recon.observation_counts = counts;
    recon.rebuild_derived_fields();
    recon
}

/// An empty `.matches` value, enough to reach the join's own guards.
fn matches_fixture() -> matches_format::MatchesData {
    matches_format::MatchesData {
        metadata: matches_format::MatchesMetadata {
            version: 3,
            matching_method: "test".into(),
            matching_tool: "test".into(),
            matching_tool_version: "0".into(),
            matching_options: BTreeMap::new(),
            workspace: matches_format::WorkspaceMetadata {
                absolute_path: String::new(),
                relative_path: ".".into(),
                contents: matches_format::WorkspaceContents {
                    feature_tool: "none".into(),
                    feature_type: "sift".into(),
                    feature_options: serde_json::json!({}),
                    feature_prefix_dir: String::new(),
                },
            },
            timestamp: String::new(),
            image_count: 0,
            image_pair_count: None,
            match_count: None,
            cluster_count: None,
            cluster_member_count: None,
            has_two_view_geometries: false,
            has_clusters: false,
            has_cluster_patches: false,
        },
        content_hash: matches_format::MatchesContentHash {
            metadata_xxh128: String::new(),
            images_xxh128: String::new(),
            image_pairs_xxh128: None,
            clusters_xxh128: None,
            cluster_patches_xxh128: None,
            two_view_geometries_xxh128: None,
            content_xxh128: String::new(),
        },
        image_names: Vec::new(),
        feature_tool_hashes: Vec::new(),
        sift_content_hashes: Vec::new(),
        feature_counts: ndarray::Array1::zeros(0),
        image_dims: None,
        image_pairs: None,
        clusters: None,
        cluster_patches: None,
        two_view_geometries: None,
    }
}

// ── Real files ──────────────────────────────────────────────────────────────

/// A candidate solve whose far frames a human adjudicated as wrong. Resecting
/// one of them against the rest should move it a long way.
const FAR_FRAME_SOLVE: &str = r"C:\DataSets\workspace-prep\evo-survey-20260823\results\20250425_135433677\candidate_solves\h04.sfmr";
const FAR_FRAME: &str = "frames/20250425_135433677_2651.jpg";

/// A candidate carrying points at infinity.
const INFINITY_SOLVE: &str = r"C:\DataSets\workspace-prep\evo-survey-20260823\results\20240702_224718414\candidate_solves\h08.sfmr";

fn report_of(path: &str, image: &str) -> super::ResectImageReport {
    let recon = SfmrReconstruction::load(std::path::Path::new(path)).expect("load");
    let index = recon
        .images
        .iter()
        .position(|i| i.name == image)
        .unwrap_or_else(|| panic!("{image} is not in {path}"));
    let out = resect_image(
        &recon,
        index,
        ResectSource::StoredObservations,
        &ResectImageOptions::default(),
    )
    .expect("support");
    let r = &out.report;
    println!(
        "{}: {} pts, inliers {}/{} ({:.3}), rotation {:.2}deg, translation {:.4} \
         ({:?} scene), {} re-triangulated, held-out {}, removed {}, rotation_only {}, \
         accepted {} {:?}",
        r.image_name,
        r.correspondences,
        r.inliers,
        r.correspondences,
        r.inlier_fraction,
        r.rotation_deg,
        r.translation,
        r.translation_scene,
        r.retriangulated,
        r.held_out_points,
        r.removed_points,
        r.rotation_only,
        r.accepted,
        r.refusal,
    );
    out.report
}

#[test]
#[ignore = "reads a candidate solve from outside the repository"]
fn resects_the_adjudicated_far_frame() {
    let report = report_of(FAR_FRAME_SOLVE, FAR_FRAME);
    assert!(!report.rotation_only, "the candidate has finite structure");
    assert!(report.accepted, "refused: {:?}", report.refusal);
    // The frame a human called wrong: held out from the structure it helped
    // build, it lands tens of degrees and better than a scene-scale away.
    assert!(report.rotation_deg > 10.0, "{}", report.rotation_deg);
    assert!(report.translation_scene.unwrap() > 0.5);
}

#[test]
#[ignore = "reads a candidate solve from outside the repository"]
fn resects_a_member_of_the_infinity_candidate() {
    let recon = SfmrReconstruction::load(std::path::Path::new(INFINITY_SOLVE)).expect("load");
    let infinity = recon.points.iter().filter(|p| p.is_at_infinity()).count();
    println!(
        "{} images, {} points ({infinity} at infinity), {} observations",
        recon.images.len(),
        recon.points.len(),
        recon.tracks.len()
    );
    // Whichever image carries the most observations — the best-supported member.
    let mut counts: HashMap<usize, usize> = HashMap::new();
    for track in &recon.tracks {
        *counts.entry(track.image_index as usize).or_default() += 1;
    }
    let mut ranked: Vec<(usize, usize)> = counts.into_iter().collect();
    ranked.sort_by_key(|&(image, n)| (std::cmp::Reverse(n), image));
    for &(image, n) in ranked.iter().take(3) {
        println!("  candidate image {image} ({n} observations)");
        let name = recon.images[image].name.clone();
        let report = report_of(INFINITY_SOLVE, &name);
        // Every point of this candidate is a bearing, so there is no finite
        // support at all and the rotation-only path is the only one available.
        assert!(report.rotation_only);
        assert_eq!(report.held_out_points, 0);
        assert!(report.scene_scale.is_none());
    }
    assert_eq!(recon.metadata.feature_source, FEATURE_SOURCE_SIFT_FILES);
    assert_eq!(infinity, recon.points.len());
}
