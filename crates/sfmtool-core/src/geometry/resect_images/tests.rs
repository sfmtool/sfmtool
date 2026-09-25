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

use approx::assert_relative_eq;
use nalgebra::{Matrix3, Point3, Rotation3, UnitQuaternion, Vector3};
use ndarray::Array2;

use sfmtool_sfmr_format::{
    ContentHash, DepthStatistics, SfmrMetadata, FEATURE_SOURCE_EMBEDDED_PATCHES,
    FEATURE_SOURCE_SIFT_FILES,
};

use crate::camera::{CameraIntrinsics, CameraModel};
use crate::progress::Progress;
use crate::reconstruction::{
    ImageTable, ObservationSource, Point3D, PointSet, SfmrImage, SfmrReconstruction,
    TrackObservation,
};

use sfmtool_matches_format::{ClusterMemberStatus, ClusterPatchData, ClustersData, MatchesData};

use super::clusters::member_residual;
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
    let points = positions.into_iter().map(|p| (p, at_infinity)).collect();
    build_points(images, points, focal)
}

/// [`build`] with each point's own `at_infinity`, so one reconstruction can
/// hold finite points and points at infinity.
fn build_points(
    images: Vec<SfmrImage>,
    positions: Vec<(Point3<f64>, bool)>,
    focal: f64,
) -> SfmrReconstruction {
    let camera = camera(focal);
    let mut points = Vec::new();
    let mut tracks = Vec::new();
    let mut observation_counts = Vec::new();
    let mut keypoints: Vec<[f32; 2]> = Vec::new();
    for &(position, at_infinity) in &positions {
        let position = if at_infinity {
            Point3::from(position.coords.normalize())
        } else {
            position
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
        workspace: sfmtool_sfmr_format::WorkspaceMetadata {
            absolute_path: String::new(),
            relative_path: ".".into(),
            contents: sfmtool_sfmr_format::WorkspaceContents {
                feature_tool: "none".into(),
                feature_type: "sift".into(),
                feature_options: serde_json::json!({}),
                feature_prefix_dir: String::new(),
            },
        },
        timestamp: String::new(),
        image_count: n_images as u32,
        point_count: points.len() as u32,
        infinity_point_count: points.iter().filter(|p: &&Point3D| p.w == 0.0).count() as u32,
        observation_count: tracks.len() as u32,
        camera_count: 1,
        rig_count: None,
        sensor_count: None,
        frame_count: None,
        world_space_unit: None,
        feature_source: FEATURE_SOURCE_EMBEDDED_PATCHES.to_string(),
        lineage: Vec::new(),
    };
    let mut recon = SfmrReconstruction {
        workspace_dir: PathBuf::new(),
        metadata,
        content_hash: ContentHash {
            derived_xxh128: None,
            metadata_xxh128: String::new(),
            cameras_xxh128: String::new(),
            rigs_xxh128: None,
            frames_xxh128: None,
            images_xxh128: String::new(),
            points3d_xxh128: String::new(),
            tracks_xxh128: String::new(),
            content_xxh128: String::new(),
        },
        image_table: ImageTable {
            cameras: vec![camera],
            images,
            thumbnails_y_x_rgb: None,
            depth_statistics: DepthStatistics {
                num_histogram_buckets: 0,
                images: Vec::new(),
            },
            depth_histogram_counts: Vec::new(),
            rig_frame_data: None,
        },
        point_set: PointSet {
            points,
            tracks,
            observation_counts,
            patch_u_halfvec_xyz: None,
            patch_v_halfvec_xyz: None,
            patch_bitmaps_y_x_rgba: None,
            patch_bitmaps_for_display: false,
            has_normals: false,
            normal_confidence: None,
            point_constraints: None,
            observation_confidence: None,
            observations: ObservationSource::EmbeddedPatches {
                keypoints_xy,
                image_file_hashes: vec![[0u8; 16]; n_images],
            },
            observation_offsets: Vec::new(),
            image_feature_to_point: Vec::new(),
            max_track_feature_index: Vec::new(),
            infinity_point_count: 0,
        },
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
        .point_set
        .tracks
        .iter()
        .enumerate()
        .filter(|(_, t)| t.image_index as usize == image_index)
        .map(|(row, _)| row)
        .collect();
    let ObservationSource::EmbeddedPatches { keypoints_xy, .. } = &mut recon.point_set.observations
    else {
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
    let true_pose = truth.image_table.images[target].clone();

    let mut source = truth.clone();
    // A pose that is wrong by tens of degrees and a substantial fraction of the
    // scene: the disagreement this feature exists to show.
    let spin = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 0.35);
    source.image_table.images[target].quaternion_wxyz = spin * true_pose.quaternion_wxyz;
    source.image_table.images[target].translation_xyz =
        true_pose.translation_xyz + Vector3::new(0.6, -0.3, 0.2);

    let out = resect_image(
        &source,
        target,
        ResectSource::Tracks,
        &ResectImageOptions::default(),
    )
    .expect("the orbit has support");
    assert!(out.report.accepted, "refused: {:?}", out.report.refusal);
    assert!(out.report.correspondences >= 100);

    let fitted = &out.reconstruction.image_table.images[target];
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
    assert_eq!(out.report.source, "tracks");
    assert_eq!(out.reconstruction.metadata.operation, "explorer_resect");

    // The source is untouched under every outcome.
    assert_eq!(
        source.image_table.images[target].quaternion_wxyz,
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
    source.image_table.images[target].translation_xyz += Vector3::new(3.0, 3.0, 3.0);

    let out = resect_image(
        &source,
        target,
        ResectSource::Tracks,
        &ResectImageOptions::default(),
    )
    .expect("the other cameras still supply support");
    assert!(!out.report.accepted, "junk observations should not resect");
    assert!(out.report.refusal.is_some());
    assert_eq!(out.report.retriangulated, 0);
    assert!(out.report.held_out_points > 100);

    let worst = out
        .reconstruction
        .point_set
        .points
        .iter()
        .zip(&truth.point_set.points)
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
    let one = resect_image(&source, 2, ResectSource::Tracks, &options).unwrap();
    let two = resect_image(&source, 2, ResectSource::Tracks, &options).unwrap();
    assert_eq!(
        one.reconstruction.image_table.images[2].quaternion_wxyz,
        two.reconstruction.image_table.images[2].quaternion_wxyz
    );
    assert_eq!(
        one.reconstruction.image_table.images[2].translation_xyz,
        two.reconstruction.image_table.images[2].translation_xyz
    );
    for (a, b) in one
        .reconstruction
        .point_set
        .points
        .iter()
        .zip(&two.reconstruction.point_set.points)
    {
        assert_eq!(a.position, b.position);
    }
    assert_eq!(one.report.inlier_fraction, two.report.inlier_fraction);
}

// ── The set ─────────────────────────────────────────────────────────────────

/// Perturb one image of `recon` by a rotation about `axis` and a fixed shift.
fn perturbed(mut recon: SfmrReconstruction, target: usize, angle: f64) -> SfmrReconstruction {
    let spin = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), angle);
    recon.image_table.images[target].quaternion_wxyz =
        spin * recon.image_table.images[target].quaternion_wxyz;
    recon.image_table.images[target].translation_xyz += Vector3::new(0.6, -0.3, 0.2);
    recon
}

#[test]
fn two_targets_held_out_together_both_recover() {
    let truth = orbit();
    let targets = [0usize, 1usize];
    let true_poses: Vec<SfmrImage> = targets
        .iter()
        .map(|&t| truth.image_table.images[t].clone())
        .collect();

    let mut source = truth.clone();
    source = perturbed(source, targets[0], 0.35);
    source = perturbed(source, targets[1], -0.28);

    let out = resect_images(
        &source,
        &targets,
        ResectSource::Tracks,
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
        let fitted = &out.reconstruction.image_table.images[report.image_index];
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
            source.image_table.images[t].quaternion_wxyz,
            truth.image_table.images[t].quaternion_wxyz
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
        source.image_table.images[t].translation_xyz += Vector3::new(3.0, 3.0, 3.0);
    }

    let out = resect_images(
        &source,
        &targets,
        ResectSource::Tracks,
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
        .point_set
        .points
        .iter()
        .zip(&truth.point_set.points)
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
        ResectSource::Tracks,
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
        ResectSource::Tracks,
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
        ResectSource::Tracks,
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
        .image_table
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
    let true_rotation = truth.image_table.images[target].quaternion_wxyz;
    let stored_translation = truth.image_table.images[target].translation_xyz;

    let mut source = truth.clone();
    source.image_table.images[target].quaternion_wxyz =
        UnitQuaternion::from_axis_angle(&Vector3::x_axis(), 0.2) * true_rotation;

    let out = resect_image(
        &source,
        target,
        ResectSource::Tracks,
        &ResectImageOptions::default(),
    )
    .expect("the dome has bearings");
    assert!(out.report.rotation_only);
    assert!(out.report.accepted, "refused: {:?}", out.report.refusal);
    assert!(out.report.scene_scale.is_none());
    assert_eq!(out.report.held_out_points, 0);

    let fitted = &out.reconstruction.image_table.images[target];
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
    let count = source.image_table.images.len();
    let err = resect_image(
        &source,
        count,
        ResectSource::Tracks,
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
        ResectSource::Tracks,
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
        .point_set
        .tracks
        .iter()
        .filter(|t| t.image_index as usize == target)
        .map(|t| t.point_index)
        .collect();
    let keep: Vec<bool> = (0..source.point_set.points.len())
        .map(|p| mine.contains(&(p as u32)))
        .collect();
    source = source.filter_points_by_mask(&keep);
    let rows: Vec<usize> = source
        .point_set
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
        ResectSource::Tracks,
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
    assert!(source.point_set.points.len() >= super::MIN_BEARINGS);

    let out = resect_image(
        &source,
        0,
        ResectSource::Tracks,
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
    source.image_table.images[1].translation_xyz = Vector3::new(f64::NAN, 0.0, 0.0);
    let err = resect_image(
        &source,
        1,
        ResectSource::Tracks,
        &ResectImageOptions::default(),
    )
    .unwrap_err();
    assert!(matches!(err, ResectImageError::NotPosed(1)), "{err}");
}

/// Keep only `rows` of the reconstruction's observations, rebuilding the
/// derived indexes. Used to starve a target of held-out support.
fn drop_all_but(mut recon: SfmrReconstruction, rows: &[usize]) -> SfmrReconstruction {
    let set: std::collections::HashSet<usize> = rows.iter().copied().collect();
    let kept: Vec<usize> = (0..recon.point_set.tracks.len())
        .filter(|r| set.contains(r))
        .collect();
    let tracks: Vec<TrackObservation> = kept.iter().map(|&r| recon.point_set.tracks[r]).collect();
    let mut counts = vec![0u32; recon.point_set.points.len()];
    for track in &tracks {
        counts[track.point_index as usize] += 1;
    }
    let ObservationSource::EmbeddedPatches {
        keypoints_xy,
        image_file_hashes,
    } = &recon.point_set.observations
    else {
        unreachable!("the fixtures are embedded_patches");
    };
    recon.point_set.observations = ObservationSource::EmbeddedPatches {
        keypoints_xy: keypoints_xy.select(ndarray::Axis(0), &kept),
        image_file_hashes: image_file_hashes.clone(),
    };
    recon.point_set.tracks = tracks;
    recon.point_set.observation_counts = counts;
    recon.rebuild_derived_fields();
    recon
}

/// An empty `.matches` value: no clusters section, no cluster-patches section.
fn matches_fixture() -> sfmtool_matches_format::MatchesData {
    sfmtool_matches_format::MatchesData {
        metadata: sfmtool_matches_format::MatchesMetadata {
            version: 3,
            matching_method: "test".into(),
            matching_tool: "test".into(),
            matching_tool_version: "0".into(),
            matching_options: BTreeMap::new(),
            workspace: sfmtool_matches_format::WorkspaceMetadata {
                absolute_path: String::new(),
                relative_path: ".".into(),
                contents: sfmtool_matches_format::WorkspaceContents {
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
        content_hash: sfmtool_matches_format::MatchesContentHash {
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

// ── Clusters ────────────────────────────────────────────────────────────────

/// One cluster member: its image, its refined position, and its status.
type Member = (usize, [f32; 2], ClusterMemberStatus);

/// A cluster-patches file over `recon`'s images, holding `clusters`.
fn cluster_file(recon: &SfmrReconstruction, clusters: &[Vec<Member>]) -> MatchesData {
    let mut data = matches_fixture();
    let n = recon.image_table.images.len();
    data.image_names = recon
        .image_table
        .images
        .iter()
        .map(|image| image.name.clone())
        .collect();
    data.feature_tool_hashes = vec![[0u8; 16]; n];
    data.sift_content_hashes = vec![[0u8; 16]; n];
    data.feature_counts = ndarray::Array1::zeros(n);
    let members: Vec<Member> = clusters.iter().flatten().copied().collect();
    let m = members.len();
    let mut starts = vec![0u32];
    for cluster in clusters {
        starts.push(starts.last().unwrap() + cluster.len() as u32);
    }
    let mut positions = Array2::<f32>::zeros((m, 2));
    for (k, member) in members.iter().enumerate() {
        positions[[k, 0]] = member.1[0];
        positions[[k, 1]] = member.1[1];
    }
    let mut shapes = ndarray::Array3::<f32>::zeros((m, 2, 2));
    for k in 0..m {
        shapes[[k, 0, 0]] = 1.0;
        shapes[[k, 1, 1]] = 1.0;
    }
    data.clusters = Some(ClustersData {
        cluster_starts: ndarray::Array1::from(starts.clone()),
        member_images: members.iter().map(|member| member.0 as u32).collect(),
        member_features: (0..m as u32).collect(),
        member_positions: Some(positions),
        member_affine_shapes: Some(shapes),
        matcher_options: serde_json::json!({}),
    });
    data.cluster_patches = Some(ClusterPatchData {
        reference_members: starts[..clusters.len()].iter().copied().collect(),
        member_status: members.iter().map(|member| member.2 as u8).collect(),
        member_zncc: ndarray::Array1::from_elem(m, f32::NAN),
        member_shift_px: ndarray::Array1::zeros(m),
        member_consistency_residual: ndarray::Array1::from_elem(m, f32::NAN),
        refine_options: serde_json::json!({}),
    });
    data
}

/// One cluster per point of `recon`, its members at the point's observations
/// (the fixtures' exact projections), the first the reference and the rest
/// kept: clusters that say what the tracks say.
fn clusters_like_tracks(recon: &SfmrReconstruction) -> Vec<Vec<Member>> {
    let keypoints = recon.keypoints_xy().expect("the fixtures are embedded");
    (0..recon.point_set.points.len())
        .map(|p| {
            let rows =
                recon.point_set.observation_offsets[p]..recon.point_set.observation_offsets[p + 1];
            rows.enumerate()
                .map(|(k, row)| {
                    let image = recon.point_set.tracks[row].image_index as usize;
                    let status = if k == 0 {
                        ClusterMemberStatus::Reference
                    } else {
                        ClusterMemberStatus::Kept
                    };
                    (image, [keypoints[[row, 0]], keypoints[[row, 1]]], status)
                })
                .collect()
        })
        .collect()
}

/// `recon` with every observation of `image` removed: an image with no tracks.
fn without_observations_of(recon: SfmrReconstruction, image: usize) -> SfmrReconstruction {
    let rows: Vec<usize> = recon
        .point_set
        .tracks
        .iter()
        .enumerate()
        .filter(|(_, t)| t.image_index as usize != image)
        .map(|(row, _)| row)
        .collect();
    drop_all_but(recon, &rows)
}

/// The pixel `world` projects to in image `image` of `recon`, which must be in
/// the frame.
fn pixel_of(recon: &SfmrReconstruction, image: usize, world: &Point3<f64>) -> [f32; 2] {
    project(
        &recon.image_table.cameras[0],
        &recon.image_table.images[image],
        world,
        false,
    )
    .expect("in the frame")
}

#[test]
fn a_file_without_both_cluster_sections_is_refused() {
    let source = orbit();
    let empty = matches_fixture();
    let err = resect_image(
        &source,
        0,
        ResectSource::TracksAndClusters(&empty),
        &ResectImageOptions::default(),
    )
    .unwrap_err();
    assert!(matches!(err, ResectImageError::Clusters(_)), "{err}");
    assert!(err.to_string().contains("no clusters section"), "{err}");

    let mut unrefined = cluster_file(&source, &clusters_like_tracks(&source));
    unrefined.cluster_patches = None;
    let err = resect_image(
        &source,
        0,
        ResectSource::TracksAndClusters(&unrefined),
        &ResectImageOptions::default(),
    )
    .unwrap_err();
    assert!(err.to_string().contains("cluster patches section"), "{err}");
}

#[test]
fn clusters_alone_resect_an_image_with_no_tracks() {
    let truth = orbit();
    let target = 0;
    let file = cluster_file(&truth, &clusters_like_tracks(&truth));
    let source = perturbed(without_observations_of(truth.clone(), target), target, 0.35);

    // The tracks give this image nothing to fit.
    let tracks = resect_image(
        &source,
        target,
        ResectSource::Tracks,
        &ResectImageOptions::default(),
    )
    .unwrap();
    assert!(!tracks.report.accepted);
    assert!(tracks.report.refusal.unwrap().contains("no support"));

    let out = resect_image(
        &source,
        target,
        ResectSource::TracksAndClusters(&file),
        &ResectImageOptions::default(),
    )
    .unwrap();
    let r = &out.report;
    assert!(r.accepted, "refused: {:?}", r.refusal);
    assert_eq!(r.source, "tracks_and_clusters");
    assert_eq!(r.track_correspondences, 0);
    assert!(r.cluster_correspondences >= 100, "{r:?}");
    assert_eq!(r.correspondences, r.cluster_correspondences);
    assert_eq!(r.inliers, r.cluster_inliers);
    assert_eq!(r.track_inliers, 0);
    assert_eq!(r.clusters_considered, r.cluster_correspondences);
    assert_eq!(
        (
            r.clusters_skipped,
            r.clusters_failed,
            r.clusters_inconsistent
        ),
        (0, 0, 0)
    );

    let fitted = &out.reconstruction.image_table.images[target];
    let true_pose = &truth.image_table.images[target];
    assert!(angle_deg(&fitted.quaternion_wxyz, &true_pose.quaternion_wxyz) < 0.1);
    assert!((fitted.camera_center() - true_pose.camera_center()).norm() < 0.01);

    // Clusters create no points and move none: the image observes no point, so
    // the structure is the source's exactly.
    assert_eq!(r.retriangulated, 0);
    assert_eq!(
        out.reconstruction.point_set.points.len(),
        source.point_set.points.len()
    );
    for (a, b) in out
        .reconstruction
        .point_set
        .points
        .iter()
        .zip(&source.point_set.points)
    {
        assert_eq!(a.position, b.position);
    }
}

#[test]
fn the_tracks_and_the_clusters_reach_the_estimate_side_by_side() {
    let truth = orbit();
    let target = 0;
    let file = cluster_file(&truth, &clusters_like_tracks(&truth));
    let source = perturbed(truth.clone(), target, 0.35);

    let options = ResectImageOptions::default();
    let tracks = resect_image(&source, target, ResectSource::Tracks, &options).unwrap();
    let both = resect_images(
        &source,
        &[target],
        ResectSource::TracksAndClusters(&file),
        &options,
    )
    .unwrap();
    let r = &both.reports[0];
    assert!(r.accepted, "refused: {:?}", r.refusal);
    // Every track pair is still there, and every cluster that mirrors one is a
    // pair of its own beside it: nothing is merged or dropped.
    assert_eq!(r.track_correspondences, tracks.report.correspondences);
    assert_eq!(r.cluster_correspondences, tracks.report.correspondences);
    assert_eq!(
        r.correspondences,
        r.track_correspondences + r.cluster_correspondences
    );
    assert_eq!(r.inliers, r.track_inliers + r.cluster_inliers);
    assert_eq!(tracks.report.cluster_correspondences, 0);
    assert_eq!(tracks.report.clusters_considered, 0);
    assert_eq!(both.totals.track_correspondences, r.track_correspondences);
    assert_eq!(
        both.totals.cluster_correspondences,
        r.cluster_correspondences
    );
    assert_eq!(both.totals.cluster_inliers, r.cluster_inliers);

    let fitted = &both.reconstruction.image_table.images[target];
    let true_pose = &truth.image_table.images[target];
    assert!(angle_deg(&fitted.quaternion_wxyz, &true_pose.quaternion_wxyz) < 0.1);
    // The structure is the tracks' own: clusters add no point.
    assert_eq!(
        both.reconstruction.point_set.points.len(),
        tracks.reconstruction.point_set.points.len()
    );
}

#[test]
fn a_cluster_with_two_kept_members_in_the_target_contributes_nothing() {
    let truth = orbit();
    let target = 0;
    let mut clusters = clusters_like_tracks(&truth);
    let baseline = {
        let file = cluster_file(&truth, &clusters);
        resect_image(
            &truth,
            target,
            ResectSource::TracksAndClusters(&file),
            &ResectImageOptions::default(),
        )
        .unwrap()
        .report
    };
    // A second kept member in the target, on a cluster that already has one.
    let doubled = clusters
        .iter()
        .position(|c| c.iter().any(|m| m.0 == target))
        .expect("the target is in some cluster");
    let first = clusters[doubled]
        .iter()
        .find(|m| m.0 == target)
        .copied()
        .unwrap();
    clusters[doubled].push((
        target,
        [first.1[0] + 5.0, first.1[1]],
        ClusterMemberStatus::Kept,
    ));
    let file = cluster_file(&truth, &clusters);
    let r = resect_image(
        &truth,
        target,
        ResectSource::TracksAndClusters(&file),
        &ResectImageOptions::default(),
    )
    .unwrap()
    .report;
    assert_eq!(r.clusters_considered, baseline.clusters_considered);
    assert_eq!(r.clusters_skipped, baseline.clusters_skipped + 1);
    assert_eq!(
        r.cluster_correspondences,
        baseline.cluster_correspondences - 1
    );
    assert_eq!(r.track_correspondences, baseline.track_correspondences);
}

/// The first point of the cloud that every one of `images` sees, by its world
/// position.
fn a_point_seen_by(recon: &SfmrReconstruction, images: &[usize]) -> Point3<f64> {
    (0..recon.point_set.points.len())
        .find(|&p| {
            let seen: Vec<usize> = (recon.point_set.observation_offsets[p]
                ..recon.point_set.observation_offsets[p + 1])
                .map(|row| recon.point_set.tracks[row].image_index as usize)
                .collect();
            images.iter().all(|i| seen.contains(i))
        })
        .map(|p| recon.point_set.points[p].position)
        .expect("the orbit's cameras overlap")
}

#[test]
fn a_cluster_is_held_out_from_the_whole_target_set() {
    let truth = orbit();
    let world = a_point_seen_by(&truth, &[0, 1, 2]);
    // Kept members in images 0, 1 and 2, and a rejected one in image 3.
    let cluster: Vec<Member> = vec![
        (
            0,
            pixel_of(&truth, 0, &world),
            ClusterMemberStatus::Reference,
        ),
        (1, pixel_of(&truth, 1, &world), ClusterMemberStatus::Kept),
        (2, pixel_of(&truth, 2, &world), ClusterMemberStatus::Kept),
        (3, [10.0, 10.0], ClusterMemberStatus::RejectedLowZncc),
    ];
    let file = cluster_file(&truth, &[cluster]);

    // Image 0 alone: images 1 and 2 are non-target, so the cluster is placed.
    let alone = resect_image(
        &truth,
        0,
        ResectSource::TracksAndClusters(&file),
        &ResectImageOptions::default(),
    )
    .unwrap()
    .report;
    assert_eq!(alone.clusters_considered, 1);
    assert_eq!(alone.cluster_correspondences, 1);

    // Images 0 and 1 together: image 1's member is a target's and places
    // nothing, the rejected member is no evidence, and one non-target image is
    // left, so the cluster is set aside for both targets.
    let both = resect_images(
        &truth,
        &[0, 1],
        ResectSource::TracksAndClusters(&file),
        &ResectImageOptions::default(),
    )
    .unwrap();
    for r in &both.reports {
        assert_eq!(r.clusters_considered, 1, "{r:?}");
        assert_eq!(r.clusters_skipped, 1, "{r:?}");
        assert_eq!(r.cluster_correspondences, 0, "{r:?}");
    }
}

/// The orbit with two more cameras side by side: the same rotation, centres a
/// unit apart along the cameras' own x axis. Their optical axes are exactly
/// parallel.
fn orbit_with_a_side_pair() -> SfmrReconstruction {
    let mut images = ring(8, 4.0);
    let side_a = image_at(
        "frames/side_a.jpg",
        Point3::new(0.0, -7.0, 0.3),
        Point3::origin(),
    );
    let rotation = side_a.quaternion_wxyz;
    let right = rotation.inverse() * Vector3::x();
    let eye_b = side_a.camera_center() + right;
    let side_b = SfmrImage {
        name: "frames/side_b.jpg".to_string(),
        camera_index: 0,
        quaternion_wxyz: rotation,
        translation_xyz: rotation * (-eye_b.coords),
    };
    images.push(side_a);
    images.push(side_b);
    build(images, cloud(200), false, 800.0)
}

#[test]
fn a_cluster_that_does_not_triangulate_contributes_nothing() {
    let truth = orbit_with_a_side_pair();
    let (a, b) = (8, 9);
    let world = a_point_seen_by(&truth, &[0, 1, 2]);
    let target_pixel = pixel_of(&truth, 0, &world);
    let member = |image, uv| (image, uv, ClusterMemberStatus::Kept);
    let clusters = vec![
        // Both principal points: two exactly parallel rays, whose depth is not
        // observable.
        vec![
            member(0, target_pixel),
            member(a, [320.0, 240.0]),
            member(b, [320.0, 240.0]),
        ],
        // Rays that diverge: the left camera looks left and the right camera
        // right, so the lines meet behind both.
        vec![
            member(0, target_pixel),
            member(a, [220.0, 240.0]),
            member(b, [420.0, 240.0]),
        ],
    ];
    let file = cluster_file(&truth, &clusters);
    let r = resect_image(
        &truth,
        0,
        ResectSource::TracksAndClusters(&file),
        &ResectImageOptions::default(),
    )
    .unwrap()
    .report;
    assert_eq!(r.clusters_considered, 2);
    assert_eq!(r.clusters_skipped, 0);
    assert_eq!(r.clusters_failed, 2);
    assert_eq!(r.cluster_correspondences, 0);
    assert!(r.accepted, "the tracks still carry it: {:?}", r.refusal);
}

/// A cluster at `world` with a kept member at its exact pixel in each of
/// `images`, the first the reference.
fn cluster_at(recon: &SfmrReconstruction, world: &Point3<f64>, images: &[usize]) -> Vec<Member> {
    images
        .iter()
        .enumerate()
        .map(|(k, &image)| {
            let status = if k == 0 {
                ClusterMemberStatus::Reference
            } else {
                ClusterMemberStatus::Kept
            };
            (image, pixel_of(recon, image, world), status)
        })
        .collect()
}

#[test]
fn a_cluster_whose_members_agree_with_its_position_is_kept() {
    let truth = orbit();
    let world = a_point_seen_by(&truth, &[0, 1, 2, 3]);
    let file = cluster_file(&truth, &[cluster_at(&truth, &world, &[0, 1, 2, 3])]);
    let r = resect_image(
        &truth,
        0,
        ResectSource::TracksAndClusters(&file),
        &ResectImageOptions::default(),
    )
    .unwrap()
    .report;
    assert_eq!(r.clusters_considered, 1);
    assert_eq!(r.clusters_inconsistent, 0);
    assert_eq!(r.cluster_correspondences, 1);
}

#[test]
fn a_cluster_with_a_member_off_its_position_contributes_nothing() {
    let truth = orbit();
    let world = a_point_seen_by(&truth, &[0, 1, 2, 3]);
    let mut cluster = cluster_at(&truth, &world, &[0, 1, 2, 3]);
    // Image 3's member 10 px away from where the point is: the three
    // non-target rays no longer meet, and no position fits all of them.
    cluster[3].1[0] += 10.0;
    let file = cluster_file(&truth, &[cluster]);

    let out = resect_images(
        &truth,
        &[0],
        ResectSource::TracksAndClusters(&file),
        &ResectImageOptions::default(),
    )
    .unwrap();
    let r = &out.reports[0];
    assert_eq!(r.clusters_considered, 1);
    assert_eq!(
        (
            r.clusters_skipped,
            r.clusters_failed,
            r.clusters_inconsistent
        ),
        (0, 0, 1)
    );
    assert_eq!(r.cluster_correspondences, 0);
    assert_eq!(out.totals.clusters_inconsistent, 1);
    assert!(r.accepted, "the tracks still carry it: {:?}", r.refusal);

    // A threshold wider than the offset keeps it.
    let wide = ResectImageOptions {
        max_cluster_residual_px: 20.0,
        ..ResectImageOptions::default()
    };
    let r = resect_image(&truth, 0, ResectSource::TracksAndClusters(&file), &wide)
        .unwrap()
        .report;
    assert_eq!(r.clusters_inconsistent, 0);
    assert_eq!(r.cluster_correspondences, 1);
}

#[test]
fn a_member_behind_its_camera_or_outside_its_frame_does_not_agree() {
    let truth = orbit();
    let world = a_point_seen_by(&truth, &[0]);
    let uv = pixel_of(&truth, 0, &world);
    let uv = [f64::from(uv[0]), f64::from(uv[1])];
    let residual = |p: Point3<f64>, uv: [f64; 2]| member_residual(&truth, 0, uv, [p.x, p.y, p.z]);
    assert!(residual(world, uv) < 1e-3);

    // The point mirrored through the camera centre: on the line of the
    // member's ray, but behind the camera.
    let image = &truth.image_table.images[0];
    let centre = image.camera_center();
    assert_eq!(residual(centre - (world - centre), uv), f64::INFINITY);

    // In front of the camera, but where the frame does not reach: past its left
    // edge.
    let ray = truth.image_table.cameras[0].pixel_to_ray(-50.0, 240.0);
    let outside =
        centre + image.quaternion_wxyz.inverse() * Vector3::new(ray[0], ray[1], ray[2]) * 4.0;
    assert_eq!(residual(outside, [10.0, 240.0]), f64::INFINITY);

    // A member whose position is not a pixel.
    assert_eq!(residual(world, [f64::NAN, uv[1]]), f64::INFINITY);
}

#[test]
fn the_rotation_only_path_reads_the_tracks_bearings_only() {
    let truth = dome();
    let file = cluster_file(&truth, &clusters_like_tracks(&truth));
    let mut source = truth.clone();
    source.image_table.images[0].quaternion_wxyz =
        UnitQuaternion::from_axis_angle(&Vector3::x_axis(), 0.2)
            * truth.image_table.images[0].quaternion_wxyz;
    let out = resect_image(
        &source,
        0,
        ResectSource::TracksAndClusters(&file),
        &ResectImageOptions::default(),
    )
    .unwrap();
    let r = &out.report;
    assert!(r.rotation_only, "{r:?}");
    assert!(r.accepted, "refused: {:?}", r.refusal);
    assert_eq!(r.cluster_correspondences, 0);
    assert_eq!(r.cluster_inliers, 0);
    assert_eq!(r.correspondences, r.track_correspondences);
    // Cameras that share a centre give a cluster no depth to place it at.
    assert!(r.clusters_considered > 0);
    assert_eq!(
        r.clusters_failed + r.clusters_skipped,
        r.clusters_considered
    );
}

// ── Images without tracks ───────────────────────────────────────────────────

#[test]
fn a_member_in_an_image_with_no_tracks_does_not_count() {
    let truth = orbit();
    let untracked = 3;
    let world = a_point_seen_by(&truth, &[0, 1, 2, untracked]);
    let mut cluster = cluster_at(&truth, &world, &[0, 1, 2, untracked]);
    // Image 3's member 10 px off: counted, it makes the cluster inconsistent.
    cluster[3].1[0] += 10.0;
    let file = cluster_file(&truth, &[cluster]);
    let options = ResectImageOptions::default();

    let tracked = resect_image(&truth, 0, ResectSource::TracksAndClusters(&file), &options)
        .unwrap()
        .report;
    assert_eq!(tracked.clusters_inconsistent, 1, "{tracked:?}");
    assert_eq!(tracked.cluster_correspondences, 0);

    // With image 3 holding no track observation, its member is not counted:
    // the cluster is placed from images 1 and 2 alone and agrees with them.
    let source = without_observations_of(truth, untracked);
    let r = resect_image(&source, 0, ResectSource::TracksAndClusters(&file), &options)
        .unwrap()
        .report;
    assert_eq!(
        (
            r.clusters_considered,
            r.clusters_untracked,
            r.clusters_inconsistent
        ),
        (1, 0, 0),
        "{r:?}"
    );
    assert_eq!(r.cluster_correspondences, 1);
    assert_eq!(r.cluster_inliers, 1);
}

#[test]
fn a_cluster_left_with_one_tracked_image_gives_no_pair() {
    let truth = orbit();
    let untracked = 3;
    let world = a_point_seen_by(&truth, &[0, 1, untracked]);
    let file = cluster_file(&truth, &[cluster_at(&truth, &world, &[0, 1, untracked])]);
    let options = ResectImageOptions::default();

    let tracked = resect_image(&truth, 0, ResectSource::TracksAndClusters(&file), &options)
        .unwrap()
        .report;
    assert_eq!(tracked.cluster_correspondences, 1, "{tracked:?}");
    assert_eq!(tracked.clusters_untracked, 0);

    let source = without_observations_of(truth, untracked);
    let out = resect_images(
        &source,
        &[0],
        ResectSource::TracksAndClusters(&file),
        &options,
    )
    .unwrap();
    let r = &out.reports[0];
    assert_eq!(
        (
            r.clusters_considered,
            r.clusters_skipped,
            r.clusters_untracked
        ),
        (1, 0, 1),
        "{r:?}"
    );
    assert_eq!(r.cluster_correspondences, 0);
    assert_eq!(out.totals.clusters_untracked, 1);
}

#[test]
fn the_cluster_counts_add_up() {
    let truth = orbit_with_a_side_pair();
    let (a, b) = (8, 9);
    let untracked = 5;
    let world = a_point_seen_by(&truth, &[0, 1, 2, 3, untracked]);
    let member = |image, uv| (image, uv, ClusterMemberStatus::Kept);
    let target_pixel = pixel_of(&truth, 0, &world);
    let mut inconsistent = cluster_at(&truth, &world, &[0, 1, 2, 3]);
    inconsistent[3].1[0] += 10.0;
    let clusters = vec![
        // Two kept members in the target: skipped.
        vec![
            member(0, target_pixel),
            member(0, [target_pixel[0] + 5.0, target_pixel[1]]),
            member(1, pixel_of(&truth, 1, &world)),
            member(2, pixel_of(&truth, 2, &world)),
        ],
        // One tracked non-target image and one untracked: untracked.
        cluster_at(&truth, &world, &[0, 1, untracked]),
        // Parallel rays: fails to triangulate.
        vec![
            member(0, target_pixel),
            member(a, [320.0, 240.0]),
            member(b, [320.0, 240.0]),
        ],
        inconsistent,
        // Agrees with its position: one pair.
        cluster_at(&truth, &world, &[0, 1, 2, 3]),
    ];
    let file = cluster_file(&truth, &clusters);
    let source = without_observations_of(truth, untracked);
    let out = resect_images(
        &source,
        &[0],
        ResectSource::TracksAndClusters(&file),
        &ResectImageOptions::default(),
    )
    .unwrap();
    let r = &out.reports[0];
    assert!(r.accepted, "refused: {:?}", r.refusal);
    assert_eq!(r.clusters_considered, 5);
    assert_eq!(
        (
            r.clusters_skipped,
            r.clusters_untracked,
            r.clusters_failed,
            r.clusters_inconsistent,
            r.cluster_correspondences
        ),
        (1, 1, 1, 1, 1),
        "{r:?}"
    );
    assert_eq!(
        r.correspondences,
        r.track_correspondences + r.cluster_correspondences
    );
    assert_eq!(r.inliers, r.track_inliers + r.cluster_inliers);
    assert!(r.bearing_correspondences <= r.track_correspondences);
    assert!(r.bearing_inliers <= r.track_inliers);
    let t = &out.totals;
    assert_eq!(
        (
            t.correspondences,
            t.track_correspondences,
            t.bearing_correspondences,
            t.cluster_correspondences
        ),
        (
            r.correspondences,
            r.track_correspondences,
            r.bearing_correspondences,
            r.cluster_correspondences
        )
    );
    assert_eq!(
        (
            t.inliers,
            t.track_inliers,
            t.bearing_inliers,
            t.cluster_inliers
        ),
        (
            r.inliers,
            r.track_inliers,
            r.bearing_inliers,
            r.cluster_inliers
        )
    );
    assert_eq!((t.clusters_untracked, t.clusters_inconsistent), (1, 1));
}

// ── Tracks lead, clusters support ───────────────────────────────────────────

/// `recon` with image `image` keeping only its first `keep` observations.
fn with_first_observations_of(
    recon: SfmrReconstruction,
    image: usize,
    keep: usize,
) -> SfmrReconstruction {
    let mut seen = 0;
    let rows: Vec<usize> = (0..recon.point_set.tracks.len())
        .filter(|&row| {
            if recon.point_set.tracks[row].image_index as usize != image {
                return true;
            }
            seen += 1;
            seen <= keep
        })
        .collect();
    drop_all_but(recon, &rows)
}

/// Clusters over the orbit's points past the first `tracked_points`: the first
/// `agree` put image 0's member where the true pose puts it, and the next
/// `disagree` where `other` (a different pose of image 0) puts it. Every
/// cluster's non-target members sit at their exact pixels, so each one
/// triangulates to its point and agrees with its own members.
fn clusters_for_image_0(
    truth: &SfmrReconstruction,
    tracked_points: usize,
    agree: usize,
    disagree: usize,
    other: &SfmrImage,
) -> Vec<Vec<Member>> {
    let camera = &truth.image_table.cameras[0];
    let mut out = Vec::new();
    for p in tracked_points..truth.point_set.points.len() {
        if out.len() == agree + disagree {
            break;
        }
        let world = truth.point_set.points[p].position;
        let rows =
            truth.point_set.observation_offsets[p]..truth.point_set.observation_offsets[p + 1];
        let images: Vec<usize> = rows
            .map(|row| truth.point_set.tracks[row].image_index as usize)
            .collect();
        if images.first() != Some(&0) || images.len() < 3 {
            continue;
        }
        let target_pixel = if out.len() < agree {
            pixel_of(truth, 0, &world)
        } else {
            match project(camera, other, &world, false) {
                Some(uv) => uv,
                None => continue,
            }
        };
        let mut cluster = vec![(0, target_pixel, ClusterMemberStatus::Reference)];
        for &image in &images[1..] {
            cluster.push((
                image,
                pixel_of(truth, image, &world),
                ClusterMemberStatus::Kept,
            ));
        }
        out.push(cluster);
    }
    assert_eq!(out.len(), agree + disagree, "the orbit has too few points");
    out
}

/// Image 0 of `truth` at a different pose: turned 4° and moved.
fn another_pose_of_image_0(truth: &SfmrReconstruction) -> SfmrImage {
    let mut other = truth.image_table.images[0].clone();
    other.quaternion_wxyz = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 4f64.to_radians())
        * other.quaternion_wxyz;
    other.translation_xyz += Vector3::new(0.3, 0.1, 0.0);
    other
}

#[test]
fn tracks_lead_over_a_larger_set_of_clusters_that_agree_with_another_pose() {
    let truth = orbit();
    let other = another_pose_of_image_0(&truth);
    // Ten tracks and fifteen clusters say where image 0 is; thirty clusters
    // agree with each other on a pose four degrees away.
    let tracks = 10;
    let file = cluster_file(
        &truth,
        &clusters_for_image_0(&truth, tracks, 15, 30, &other),
    );
    let source = perturbed(
        with_first_observations_of(truth.clone(), 0, tracks),
        0,
        0.35,
    );

    let r = resect_image(
        &source,
        0,
        ResectSource::TracksAndClusters(&file),
        &ResectImageOptions::default(),
    )
    .unwrap();
    let report = &r.report;
    assert!(report.accepted, "refused: {:?}", report.refusal);
    assert_eq!(report.track_correspondences, tracks);
    assert_eq!(report.track_inliers, tracks);
    assert_eq!(report.cluster_correspondences, 45);
    assert_eq!(report.cluster_inliers, 15, "{report:?}");
    let fitted = &r.reconstruction.image_table.images[0];
    let true_pose = &truth.image_table.images[0];
    assert!(angle_deg(&fitted.quaternion_wxyz, &true_pose.quaternion_wxyz) < 0.05);
    assert!((fitted.camera_center() - true_pose.camera_center()).norm() < 0.005);
}

#[test]
fn below_three_finite_tracks_the_tracks_and_the_clusters_are_sampled_together() {
    let truth = orbit();
    let other = another_pose_of_image_0(&truth);
    let options = ResectImageOptions::default();

    // Two tracks and clusters that agree with them: the pairs are sampled
    // together, and the two tracks are inliers of the pose they give.
    let tracks = 2;
    let source = perturbed(
        with_first_observations_of(truth.clone(), 0, tracks),
        0,
        0.35,
    );
    let file = cluster_file(&truth, &clusters_for_image_0(&truth, tracks, 40, 0, &other));
    let r = resect_image(&source, 0, ResectSource::TracksAndClusters(&file), &options).unwrap();
    assert!(r.report.accepted, "refused: {:?}", r.report.refusal);
    assert_eq!(r.report.track_correspondences, tracks);
    assert_eq!(r.report.track_inliers, tracks);
    assert_eq!(r.report.cluster_inliers, 40);
    let fitted = &r.reconstruction.image_table.images[0];
    let true_pose = &truth.image_table.images[0];
    assert!(angle_deg(&fitted.quaternion_wxyz, &true_pose.quaternion_wxyz) < 0.05);

    // With the clusters of the test above, two tracks do not lead: the thirty
    // clusters that agree on the other pose outnumber the rest, and the tracks
    // are outliers of the pose they give.
    let file = cluster_file(
        &truth,
        &clusters_for_image_0(&truth, tracks, 15, 30, &other),
    );
    let r = resect_image(&source, 0, ResectSource::TracksAndClusters(&file), &options).unwrap();
    assert_eq!(r.report.track_correspondences, tracks);
    assert_eq!(r.report.track_inliers, 0, "{:?}", r.report);
    let fitted = &r.reconstruction.image_table.images[0];
    assert!(angle_deg(&fitted.quaternion_wxyz, &other.quaternion_wxyz) < 0.05);
}

// ── Bearings in the finite path ─────────────────────────────────────────────

/// The orbit under a sky: its cameras at a wide focal, so neighbours share
/// directions, observing the ball and sixty points at infinity.
fn orbit_under_a_sky() -> SfmrReconstruction {
    let mut points: Vec<(Point3<f64>, bool)> = cloud(200).into_iter().map(|p| (p, false)).collect();
    points.extend(cloud(60).into_iter().map(|p| (p, true)));
    build_points(ring(8, 4.0), points, 300.0)
}

#[test]
fn a_bearing_is_a_correspondence_of_the_finite_path() {
    let truth = orbit_under_a_sky();
    let source = perturbed(truth.clone(), 0, 0.35);
    let options = ResectImageOptions::default();
    let r = resect_image(&source, 0, ResectSource::Tracks, &options).unwrap();
    let report = &r.report;
    assert!(report.accepted, "refused: {:?}", report.refusal);
    assert!(!report.rotation_only);
    assert!(report.bearing_correspondences >= 3, "{report:?}");
    assert_eq!(report.bearing_inliers, report.bearing_correspondences);
    assert_eq!(
        report.track_correspondences,
        report.held_out_points + report.bearing_correspondences
    );
    assert_eq!(report.correspondences, report.track_correspondences);
    let fitted = &r.reconstruction.image_table.images[0];
    let true_pose = &truth.image_table.images[0];
    assert!(angle_deg(&fitted.quaternion_wxyz, &true_pose.quaternion_wxyz) < 0.05);

    // One bearing observed 30 px from where it is: an outlier.
    let row = (0..source.point_set.tracks.len())
        .find(|&row| {
            let t = source.point_set.tracks[row];
            t.image_index == 0 && source.point_set.points[t.point_index as usize].is_at_infinity()
        })
        .expect("image 0 observes a point at infinity");
    let mut moved = source.clone();
    let ObservationSource::EmbeddedPatches { keypoints_xy, .. } = &mut moved.point_set.observations
    else {
        unreachable!("the fixtures are embedded_patches");
    };
    keypoints_xy[[row, 0]] += 30.0;
    let r = resect_image(&moved, 0, ResectSource::Tracks, &options)
        .unwrap()
        .report;
    assert_eq!(r.bearing_correspondences, report.bearing_correspondences);
    assert_eq!(r.bearing_inliers, report.bearing_correspondences - 1);
}

#[test]
fn a_bearing_constrains_the_rotation_and_not_the_translation() {
    use super::finite::{pixels_per_radian, residual_px, Pair, Source, World};
    let recon = orbit_under_a_sky();
    let camera = &recon.image_table.cameras[0];
    let image = &recon.image_table.images[0];
    let ppr = pixels_per_radian(camera);
    let pose = (image.quaternion_wxyz, image.translation_xyz);
    let direction = (image.quaternion_wxyz.inverse() * Vector3::new(0.1, 0.05, -1.0)).normalize();
    let uv = project(camera, image, &Point3::from(direction), true).expect("in the frame");
    let pair = Pair::new(
        Source::Bearing,
        [f64::from(uv[0]), f64::from(uv[1])],
        World::Direction(direction),
        camera,
    )
    .expect("a ray");
    assert!(residual_px(camera, ppr, &pair, &pose) < 1e-3);

    // Moving the camera moves nothing at infinity.
    let moved = (pose.0, pose.1 + Vector3::new(5.0, -3.0, 2.0));
    assert!(residual_px(camera, ppr, &pair, &moved) < 1e-3);

    // Turning it by an angle across the ray costs that angle in focal-length
    // pixels.
    let angle = 0.004;
    let axis = nalgebra::Unit::new_normalize(pair.ray.cross(&Vector3::x()));
    let turned = (
        UnitQuaternion::from_axis_angle(&axis, angle) * pose.0,
        pose.1,
    );
    assert_relative_eq!(
        residual_px(camera, ppr, &pair, &turned),
        angle * 300.0,
        max_relative = 1e-3
    );
}

// ── In place ────────────────────────────────────────────────────────────────

/// The refusal of an in-place resection of `image`, panicking if there was
/// none. Spelled out rather than `expect_err` because the success side holds an
/// [`SfmrReconstruction`], which is not `Debug`.
fn in_place_error(recon: &SfmrReconstruction, image: usize) -> super::ResectInPlaceError {
    match super::resect_image_in_place(
        recon,
        image,
        ResectSource::Tracks,
        &ResectImageOptions::default(),
    ) {
        Ok((_, report)) => panic!("the estimate was accepted: {report:?}"),
        Err(error) => error,
    }
}

#[test]
fn in_place_hands_back_the_accepted_value_and_its_report() {
    let truth = orbit();
    let target = 0;
    let source = perturbed(truth.clone(), target, 0.35);

    let (next, report) = super::resect_image_in_place(
        &source,
        target,
        ResectSource::Tracks,
        &ResectImageOptions::default(),
    )
    .expect("the orbit corroborates the target");

    assert!(report.refusal.is_none());
    assert!(report.accepted);
    assert_eq!(report.image_index, target);
    // The value is the set call's, so the pose it carries is the estimate's
    // and the image table is the source's.
    assert_eq!(next.image_count(), source.image_count());
    let fitted = &next.image_table.images[target];
    let true_pose = &truth.image_table.images[target];
    assert!(angle_deg(&fitted.quaternion_wxyz, &true_pose.quaternion_wxyz) < 0.1);
    // The input is left exactly as it was.
    assert_eq!(
        source.image_table.images[target].quaternion_wxyz,
        perturbed(truth, target, 0.35).image_table.images[target].quaternion_wxyz
    );
}

#[test]
fn in_place_refuses_a_refused_estimate_rather_than_installing_it() {
    let mut source = perturbed(orbit(), 0, 0.35);
    // Nothing the target says is usable, so the estimate is refused: the
    // outcome the set call keeps and this one must not.
    corrupt_observations(&mut source, 0);

    let error = in_place_error(&source, 0);
    assert!(
        matches!(error, super::ResectInPlaceError::Refused(_)),
        "{error:?}"
    );
}

#[test]
fn in_place_passes_a_whole_call_refusal_through() {
    let source = orbit();
    let error = in_place_error(&source, source.image_count());
    assert!(
        matches!(
            error,
            super::ResectInPlaceError::Resect(ResectImageError::ImageOutOfRange { .. })
        ),
        "{error:?}"
    );
}

// ── Real files ──────────────────────────────────────────────────────────────

/// A candidate solve whose far frames a human adjudicated as wrong. Resecting
/// one of them against the rest should move it a long way.
const FAR_FRAME_SOLVE: &str = r"C:\DataSets\workspace-prep\evo-survey-20260823\results\20250425_135433677\candidate_solves\h04.sfmr";
const FAR_FRAME: &str = "frames/20250425_135433677_2651.jpg";

/// A candidate carrying points at infinity.
const INFINITY_SOLVE: &str = r"C:\DataSets\workspace-prep\evo-survey-20260823\results\20240702_224718414\candidate_solves\h08.sfmr";

fn report_of(path: &str, image: &str) -> super::ResectImageReport {
    let recon =
        SfmrReconstruction::load(std::path::Path::new(path), &Progress::none()).expect("load");
    let index = recon
        .image_table
        .images
        .iter()
        .position(|i| i.name == image)
        .unwrap_or_else(|| panic!("{image} is not in {path}"));
    let out = resect_image(
        &recon,
        index,
        ResectSource::Tracks,
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
    let recon = SfmrReconstruction::load(std::path::Path::new(INFINITY_SOLVE), &Progress::none())
        .expect("load");
    let infinity = recon
        .point_set
        .points
        .iter()
        .filter(|p| p.is_at_infinity())
        .count();
    println!(
        "{} images, {} points ({infinity} at infinity), {} observations",
        recon.image_table.images.len(),
        recon.point_set.points.len(),
        recon.point_set.tracks.len()
    );
    // Whichever image carries the most observations — the best-supported member.
    let mut counts: HashMap<usize, usize> = HashMap::new();
    for track in &recon.point_set.tracks {
        *counts.entry(track.image_index as usize).or_default() += 1;
    }
    let mut ranked: Vec<(usize, usize)> = counts.into_iter().collect();
    ranked.sort_by_key(|&(image, n)| (std::cmp::Reverse(n), image));
    for &(image, n) in ranked.iter().take(3) {
        println!("  candidate image {image} ({n} observations)");
        let name = recon.image_table.images[image].name.clone();
        let report = report_of(INFINITY_SOLVE, &name);
        // Every point of this candidate is a bearing, so there is no finite
        // support at all and the rotation-only path is the only one available.
        assert!(report.rotation_only);
        assert_eq!(report.held_out_points, 0);
        assert!(report.scene_scale.is_none());
    }
    assert_eq!(recon.metadata.feature_source, FEATURE_SOURCE_SIFT_FILES);
    assert_eq!(infinity, recon.point_set.points.len());
}

/// [`super::scene_scale`] takes the crate's median, which **averages the two
/// middle values** on an even population.
///
/// Worth pinning rather than assuming, because this module carried its own
/// `median` until 2026-08-29 and that one returned the *lower* of the two
/// middles. The two rules agree on every odd population, so nothing in this
/// file caught the difference; the fixture below is built with an even count
/// at both levels of the reduction precisely so they cannot agree.
///
/// Four points strung along one optical axis, seen by two cameras a unit
/// apart, so every distance is exact in binary and the expected answer can be
/// read off the construction:
///
/// | | distances | averaging median | lower-middle median |
/// |---|---|---|---|
/// | camera A at the origin | 1, 2, 3, 4 | **2.5** | 2 |
/// | camera B one unit back | 2, 3, 4, 5 | **3.5** | 3 |
/// | over images | | **3.0** | 2 |
#[test]
fn scene_scale_averages_the_two_middles_of_an_even_population() {
    let images = vec![
        image_at("frames/a.jpg", Point3::origin(), Point3::new(1.0, 0.0, 0.0)),
        image_at(
            "frames/b.jpg",
            Point3::new(-1.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
        ),
    ];
    let positions: Vec<Point3<f64>> = (1..=4).map(|d| Point3::new(d as f64, 0.0, 0.0)).collect();
    let recon = build(images, positions, false, 800.0);
    // Every point has to have survived, or the population is not the one the
    // table above describes.
    assert_eq!(recon.point_set.points.len(), 4);

    let scale = super::scene_scale(&recon).expect("finite structure has a scale");
    assert_relative_eq!(scale, 3.0, epsilon = 1e-12);
    assert!(
        (scale - 2.0).abs() > 0.5,
        "scene_scale fell back to the lower-middle median: {scale}"
    );
}
