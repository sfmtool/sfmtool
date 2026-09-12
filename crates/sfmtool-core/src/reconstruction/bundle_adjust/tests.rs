// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Bundle-adjusting a whole reconstruction: what the value that comes back
//! holds, what it no longer holds, and every refusal.
//!
//! The fixture is a synthetic scene whose truth is known -- cameras on a
//! shallow arc looking at a cloud of points, every observation the exact
//! projection -- so an adjustment run from a perturbed copy of it has somewhere
//! to converge to. It needs no pixels, because the adjustment reads none.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use nalgebra::{Point3, UnitQuaternion, Vector3};
use ndarray::{Array2, Array4};

use crate::camera::{CameraIntrinsics, CameraModel};
use crate::progress::{Event, Level};
use crate::reconstruction::data::{
    ObservationSource, Point3D, PointConstraintColumns, SfmrImage, SfmrReconstruction,
    TrackObservation,
};

use super::*;

const IMG_W: u32 = 640;
const IMG_H: u32 = 480;
const FOCAL: f64 = 500.0;
const IMAGES: usize = 6;
const POINTS: usize = 40;

/// Deterministic pseudo-random in `[-1, 1]` from an index.
fn jitter(i: usize, salt: u64) -> f64 {
    let mut z = (i as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ salt;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z ^= z >> 27;
    ((z % 20001) as f64 / 10000.0) - 1.0
}

fn pinhole() -> CameraIntrinsics {
    CameraIntrinsics {
        model: CameraModel::SimplePinhole {
            focal_length: FOCAL,
            principal_point_x: IMG_W as f64 / 2.0,
            principal_point_y: IMG_H as f64 / 2.0,
        },
        width: IMG_W,
        height: IMG_H,
    }
}

/// The truth: cameras on an arc at radius 8 looking at the origin, points in a
/// box around it, and every observation at the exact projection.
///
/// Patch frames are a fronto-parallel square per point, in world units at the
/// depth the point stands at, so a frame's rescale is measurable.
fn truth() -> SfmrReconstruction {
    let mut recon = SfmrReconstruction::demo(1);
    recon.image_table.cameras = vec![pinhole()];
    recon.image_table.images = (0..IMAGES)
        .map(|i| {
            let angle = 0.25 * (i as f64 - (IMAGES as f64 - 1.0) / 2.0);
            let centre = Vector3::new(8.0 * angle.sin(), 0.5 * jitter(i, 11), 8.0 * angle.cos());
            // Canonical look-at: the camera looks along -Z, so its local +Z
            // points away from the origin, along the centre.
            let rotation = UnitQuaternion::face_towards(&centre, &Vector3::y()).inverse();
            SfmrImage {
                name: format!("image_{i:03}.jpg"),
                camera_index: 0,
                quaternion_wxyz: rotation,
                translation_xyz: -(rotation * centre),
            }
        })
        .collect();
    recon.image_table.thumbnails_y_x_rgb = Arc::new(Array4::zeros((
        IMAGES,
        sfmr_format::THUMBNAIL_SIZE,
        sfmr_format::THUMBNAIL_SIZE,
        3,
    )));
    let stats = recon.image_table.depth_statistics.images[0].clone();
    recon.image_table.depth_statistics.images = vec![stats; IMAGES];
    recon.image_table.depth_histogram_counts =
        vec![recon.image_table.depth_histogram_counts[0].clone(); IMAGES];

    recon.point_set.points = (0..POINTS)
        .map(|p| Point3D {
            position: Point3::new(2.0 * jitter(p, 1), 2.0 * jitter(p, 2), 1.5 * jitter(p, 3)),
            w: 1.0,
            color: [10, 20, 30],
            error: 0.25,
            normal: Vector3::new(0.0, 0.0, -1.0),
        })
        .collect();

    let mut tracks = Vec::new();
    let mut counts = Vec::new();
    let mut keypoints: Vec<[f32; 2]> = Vec::new();
    for p in 0..POINTS {
        let mut count = 0u32;
        for i in 0..IMAGES {
            let Some(pixel) = project(&recon, i, recon.point_set.points[p].position) else {
                continue;
            };
            tracks.push(TrackObservation {
                image_index: i as u32,
                point_index: p as u32,
            });
            keypoints.push(pixel);
            count += 1;
        }
        counts.push(count);
    }
    assert!(
        counts.iter().all(|&c| c >= 3),
        "the fixture is degenerate: {counts:?}"
    );
    let mut keypoints_xy = Array2::<f32>::zeros((keypoints.len(), 2));
    for (row, uv) in keypoints.iter().enumerate() {
        keypoints_xy[[row, 0]] = uv[0];
        keypoints_xy[[row, 1]] = uv[1];
    }
    let set = &mut recon.point_set;
    set.tracks = tracks;
    set.observation_counts = counts;
    set.observations = ObservationSource::EmbeddedPatches {
        keypoints_xy,
        image_file_hashes: vec![[0u8; 16]; IMAGES],
    };
    let mut u = Array2::<f32>::zeros((POINTS, 3));
    let mut v = Array2::<f32>::zeros((POINTS, 3));
    for p in 0..POINTS {
        u[[p, 0]] = 0.05;
        v[[p, 1]] = 0.05;
    }
    set.patch_u_halfvec_xyz = Some(u);
    set.patch_v_halfvec_xyz = Some(v);
    recon.metadata.feature_source = sfmr_format::FEATURE_SOURCE_EMBEDDED_PATCHES.to_string();
    recon.rebuild_derived_fields();
    recon
}

/// Where `world` lands in image `i`, or `None` when it is outside the frame.
fn project(recon: &SfmrReconstruction, i: usize, world: Point3<f64>) -> Option<[f32; 2]> {
    let image = &recon.image_table.images[i];
    let camera = &recon.image_table.cameras[image.camera_index as usize];
    let local = image.quaternion_wxyz * world.coords + image.translation_xyz;
    let (u, v) = camera.ray_to_pixel([local.x, local.y, local.z])?;
    if u < 0.0 || v < 0.0 || u >= camera.width as f64 || v >= camera.height as f64 {
        return None;
    }
    Some([u as f32, v as f32])
}

/// The truth with every pose and point nudged off it: what an adjustment has to
/// find its way back from.
fn perturbed() -> SfmrReconstruction {
    let mut recon = truth();
    for (i, image) in recon.image_table.images.iter_mut().enumerate().skip(1) {
        let spin = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 0.01 * jitter(i, 5));
        image.quaternion_wxyz = spin * image.quaternion_wxyz;
        image.translation_xyz += Vector3::new(
            0.03 * jitter(i, 6),
            0.03 * jitter(i, 7),
            0.03 * jitter(i, 8),
        );
    }
    for (p, point) in recon.point_set.points.iter_mut().enumerate() {
        point.position += Vector3::new(
            0.02 * jitter(p, 21),
            0.02 * jitter(p, 22),
            0.02 * jitter(p, 23),
        );
    }
    recon
}

/// `recon` with every sighting of `point` but its first taken out of the track.
fn with_one_sighting(mut recon: SfmrReconstruction, point: u32) -> SfmrReconstruction {
    let mut seen = false;
    let keep: Vec<bool> = recon
        .point_set
        .tracks
        .iter()
        .map(|o| {
            if o.point_index != point {
                return true;
            }
            let first = !seen;
            seen = true;
            first
        })
        .collect();
    let rows: Vec<usize> = keep
        .iter()
        .enumerate()
        .filter(|(_, &k)| k)
        .map(|(row, _)| row)
        .collect();
    let ObservationSource::EmbeddedPatches { keypoints_xy, .. } = &mut recon.point_set.observations
    else {
        panic!("the fixture is embedded_patches");
    };
    *keypoints_xy = keypoints_xy.select(ndarray::Axis(0), &rows);
    recon.point_set.tracks = rows
        .iter()
        .map(|&row| recon.point_set.tracks[row])
        .collect();
    recon.point_set.observation_counts[point as usize] = 1;
    recon.rebuild_derived_fields();
    recon
}

/// How far image `i`'s camera centre stands from the truth's.
fn centre_error(recon: &SfmrReconstruction, truth: &SfmrReconstruction, i: usize) -> f64 {
    (recon.image_table.images[i].camera_center() - truth.image_table.images[i].camera_center())
        .norm()
}

/// The largest distance any point stands from the truth's.
fn worst_point_error(recon: &SfmrReconstruction, truth: &SfmrReconstruction) -> f64 {
    recon
        .point_set
        .points
        .iter()
        .zip(&truth.point_set.points)
        .map(|(a, b)| (a.position - b.position).norm())
        .fold(0.0, f64::max)
}

// ── The solve ───────────────────────────────────────────────────────────────

#[test]
fn the_poses_and_the_points_converge_on_the_truth() {
    let truth = truth();
    // The adjustment's gauge is free, so a solution that fits the pixels
    // perfectly may still sit anywhere in the seven-parameter family of
    // similarities. Three points held at the truth pin it, which is what lets
    // the assertions below name the truth at all.
    let mut source = perturbed();
    let mut columns = PointConstraintColumns::all_free(POINTS);
    for p in 0..3 {
        source.point_set.points[p].position = truth.point_set.points[p].position;
        columns.point_constraints[p] = sfmr_format::POINT_CONSTRAINT_HELD;
    }
    source.point_set.point_constraints = Some(columns);
    let source = source;
    let before_worst = worst_point_error(&source, &truth);

    let (out, report) = bundle_adjust(&source, &BundleAdjustOptions::default(), &Progress::none())
        .expect("the fixture is well posed");

    assert_eq!(report.images, IMAGES);
    assert_eq!(report.points, POINTS);
    assert_eq!(report.observations, source.point_set.tracks.len());
    assert_eq!(report.points_deleted, 0);
    assert!(
        report.median_residual_after < report.median_residual_before,
        "{report:?}"
    );
    assert!(report.median_residual_after < 0.05, "{report:?}");
    assert!(!report.focal_released);
    assert_eq!(report.focal_before, report.focal_after);

    for i in 0..IMAGES {
        assert!(
            centre_error(&out, &truth, i) < 0.01,
            "camera {i} is {} from the truth, having started {}",
            centre_error(&out, &truth, i),
            centre_error(&source, &truth, i)
        );
    }
    assert!(
        worst_point_error(&out, &truth) < 0.2 * before_worst,
        "{} against {before_worst}",
        worst_point_error(&out, &truth)
    );
}

#[test]
fn the_input_is_untouched() {
    let source = perturbed();
    let poses: Vec<Vector3<f64>> = source
        .image_table
        .images
        .iter()
        .map(|i| i.translation_xyz)
        .collect();
    let positions: Vec<Point3<f64>> = source.point_set.points.iter().map(|p| p.position).collect();

    bundle_adjust(&source, &BundleAdjustOptions::default(), &Progress::none()).expect("well posed");

    for (image, translation) in source.image_table.images.iter().zip(&poses) {
        assert_eq!(image.translation_xyz, *translation);
    }
    for (point, position) in source.point_set.points.iter().zip(&positions) {
        assert_eq!(point.position, *position);
    }
}

#[test]
fn a_released_focal_is_found_and_reported() {
    let truth = truth();
    let mut source = perturbed();
    // A lens 6 % off. Everything else is the perturbed fixture, so the focal is
    // the parameter that has to move for the residuals to come down.
    source.image_table.cameras[0] = source.image_table.cameras[0].with_focal(FOCAL * 1.06);

    let options = BundleAdjustOptions {
        opt_f: true,
        ..BundleAdjustOptions::default()
    };
    let (out, report) = bundle_adjust(&source, &options, &Progress::none()).expect("well posed");

    assert!(report.focal_released);
    assert_eq!(report.focal_before, FOCAL * 1.06);
    assert!(
        (report.focal_after - FOCAL).abs() < 0.02 * FOCAL,
        "{report:?}"
    );
    assert_eq!(
        out.image_table.cameras[0].focal_lengths().0,
        report.focal_after
    );
    assert_eq!(
        truth.image_table.cameras[0].width,
        out.image_table.cameras[0].width
    );
}

#[test]
fn a_point_at_infinity_comes_back_a_direction() {
    let mut source = perturbed();
    // A bearing among the finite points: the adjustment honours the
    // representation it was handed, so this one must not be given a depth.
    source.point_set.points[0].w = 0.0;
    source.point_set.points[0].position =
        Point3::from(source.point_set.points[0].position.coords.normalize());
    source.point_set.points[0].normal = Vector3::zeros();
    source.rebuild_derived_fields();

    let (out, report) = bundle_adjust(&source, &BundleAdjustOptions::default(), &Progress::none())
        .expect("well posed");

    assert_eq!(report.points_deleted, 0);
    let point = &out.point_set.points[0];
    assert_eq!(point.w, 0.0, "the bearing was given a depth");
    assert!(
        (point.position.coords.norm() - 1.0).abs() < 1e-9,
        "the direction came back unnormalised: {}",
        point.position.coords.norm()
    );
    assert_eq!(out.point_set.infinity_point_count, 1);
}

#[test]
fn a_held_point_comes_back_exactly() {
    let mut source = perturbed();
    let mut columns = PointConstraintColumns::all_free(source.point_set.points.len());
    columns.point_constraints[3] = sfmr_format::POINT_CONSTRAINT_HELD;
    source.point_set.point_constraints = Some(columns);
    let held = source.point_set.points[3].position;

    let (out, _) = bundle_adjust(&source, &BundleAdjustOptions::default(), &Progress::none())
        .expect("well posed");

    assert_eq!(out.point_set.points[3].position, held);
    // And the constraint travels with it.
    let columns = out
        .point_set
        .point_constraints
        .as_ref()
        .expect("the columns survived");
    assert_eq!(
        columns.point_constraints[3],
        sfmr_format::POINT_CONSTRAINT_HELD
    );
}

#[test]
fn a_point_the_trim_wears_out_is_deleted_and_the_scan_says_so() {
    // Point 5 keeps one sighting, which is below `min_track`: the round's trim
    // drops it, the re-estimation has fewer than two rays to rebuild it from,
    // and it comes back non-finite.
    let source = with_one_sighting(perturbed(), 5);

    let (out, report) = bundle_adjust(&source, &BundleAdjustOptions::default(), &Progress::none())
        .expect("well posed");

    assert_eq!(report.points_deleted, 1);
    assert_eq!(
        out.point_set.points.len(),
        source.point_set.points.len() - 1
    );

    let map = crate::RowMap::by_scan(&source, &out, None).expect("the tables match");
    assert_eq!(map.forward(5), None, "the deleted point still maps forward");
    assert_eq!(map.forward(6), Some(5), "the survivors did not close up");
}

#[test]
fn a_patch_frame_follows_its_point_in_depth() {
    let source = perturbed();
    // A point pushed twice as far from the camera cloud as it stands, with its
    // observations left where they were: the adjustment pulls it back, and the
    // frame has to come back with it.
    let mut moved = source.clone();
    let centroid_distance = source
        .image_table
        .placement_scale(&source.point_set.points[7].position);
    let direction = source.point_set.points[7].position.coords.normalize();
    moved.point_set.points[7].position += direction * centroid_distance;
    let stretched = moved
        .image_table
        .placement_scale(&moved.point_set.points[7].position);

    let (out, _) = bundle_adjust(&moved, &BundleAdjustOptions::default(), &Progress::none())
        .expect("well posed");

    let settled = out
        .image_table
        .placement_scale(&out.point_set.points[7].position);
    let before = moved
        .point_set
        .patch_u_halfvec_xyz
        .as_ref()
        .expect("the fixture carries frames")[[7, 0]] as f64;
    let after = out
        .point_set
        .patch_u_halfvec_xyz
        .as_ref()
        .expect("the fixture carries frames")[[7, 0]] as f64;
    assert!(
        (after / before - settled / stretched).abs() < 1e-3,
        "the frame scaled by {} and the depth by {}",
        after / before,
        settled / stretched
    );
    // And the point really did move, or the assertion above is vacuous.
    assert!((settled / stretched - 1.0).abs() > 0.1);
}

// ── The refusals ────────────────────────────────────────────────────────────

#[test]
fn a_value_with_no_keypoints_is_refused() {
    let mut source = perturbed();
    let observations = source.point_set.tracks.len();
    source.point_set.observations = ObservationSource::SiftFiles {
        feature_indexes: (0..observations as u32).collect(),
        keypoints_xy: None,
        feature_tool_hashes: vec![[0u8; 16]; IMAGES],
        sift_content_hashes: vec![[0u8; 16]; IMAGES],
    };
    source.metadata.feature_source = sfmr_format::FEATURE_SOURCE_SIFT_FILES.to_string();

    assert_eq!(
        bundle_adjust(&source, &BundleAdjustOptions::default(), &Progress::none()).err(),
        Some(BundleAdjustError::NoKeypoints)
    );
}

#[test]
fn two_cameras_over_the_posed_images_are_refused() {
    let mut source = perturbed();
    let second = source.image_table.cameras[0].with_focal(FOCAL * 1.2);
    source.image_table.cameras.push(second);
    source.image_table.images[2].camera_index = 1;

    assert_eq!(
        bundle_adjust(&source, &BundleAdjustOptions::default(), &Progress::none()).err(),
        Some(BundleAdjustError::MixedCameras { cameras: 2 })
    );
}

#[test]
fn a_focal_release_on_a_model_that_cannot_take_one_is_refused() {
    let mut source = perturbed();
    source.image_table.cameras[0] = CameraIntrinsics {
        model: CameraModel::Pinhole {
            focal_length_x: FOCAL,
            focal_length_y: FOCAL,
            principal_point_x: IMG_W as f64 / 2.0,
            principal_point_y: IMG_H as f64 / 2.0,
        },
        width: IMG_W,
        height: IMG_H,
    };
    let options = BundleAdjustOptions {
        opt_f: true,
        ..BundleAdjustOptions::default()
    };

    assert_eq!(
        bundle_adjust(&source, &options, &Progress::none()).err(),
        Some(BundleAdjustError::FocalNotReleasable("PINHOLE"))
    );
    // Without the release it runs: the model is only a problem for the focal.
    assert!(bundle_adjust(&source, &BundleAdjustOptions::default(), &Progress::none()).is_ok());
}

#[test]
fn an_unposed_table_and_an_empty_schedule_are_refused() {
    let mut source = perturbed();
    for image in source.image_table.images.iter_mut() {
        image.translation_xyz = Vector3::new(f64::NAN, 0.0, 0.0);
    }
    assert_eq!(
        bundle_adjust(&source, &BundleAdjustOptions::default(), &Progress::none()).err(),
        Some(BundleAdjustError::NoPosedImages)
    );

    let options = BundleAdjustOptions {
        schedule: Vec::new(),
        ..BundleAdjustOptions::default()
    };
    assert_eq!(
        bundle_adjust(&perturbed(), &options, &Progress::none()).err(),
        Some(BundleAdjustError::EmptySchedule)
    );
}

#[test]
fn a_solve_that_exits_degenerate_is_refused_rather_than_emptying_the_value() {
    let mut source = perturbed();
    // Nothing agrees with anything: every observation is trimmed, the round
    // exits under `min_obs`, and the state passes through with every residual
    // infinite.
    let ObservationSource::EmbeddedPatches { keypoints_xy, .. } =
        &mut source.point_set.observations
    else {
        panic!("the fixture is embedded_patches");
    };
    for row in 0..keypoints_xy.nrows() {
        keypoints_xy[[row, 0]] = (row % 600) as f32;
        keypoints_xy[[row, 1]] = ((row * 7) % 440) as f32;
    }

    let error = bundle_adjust(&source, &BundleAdjustOptions::default(), &Progress::none())
        .err()
        .expect("nothing survives the trim");
    assert!(
        matches!(error, BundleAdjustError::Degenerate { .. }),
        "{error:?}"
    );
}

#[test]
fn a_constraint_the_adjustment_cannot_honour_is_refused() {
    let mut source = perturbed();
    let mut columns = PointConstraintColumns::all_free(source.point_set.points.len());
    // Ranged at a finite distance from nowhere: the gauge is free, so a distance
    // from the world frame constrains nothing.
    columns.point_constraints[2] = sfmr_format::POINT_CONSTRAINT_RANGED;
    columns.constraint_distances[2] = 4.0;
    source.point_set.point_constraints = Some(columns);

    let error = bundle_adjust(&source, &BundleAdjustOptions::default(), &Progress::none())
        .err()
        .expect("that constraint names no reference");
    assert!(
        matches!(error, BundleAdjustError::Constraints(_)),
        "{error:?}"
    );
}

// -- What the call reports ---------------------------------------------------

/// An owned mirror of [`Event`], because an `Event<'a>` borrows its text only
/// for the duration of the sink call.
#[derive(Debug, Clone, PartialEq)]
enum Owned {
    Enter(&'static str, u8),
    Leave(&'static str, u8, Option<String>),
    Message(Level, u8, String),
    Count(u64, Option<u64>, &'static str),
    Fraction(f32),
}

/// Somewhere for the events to land, with the cancel flag beside them so that
/// the sink can set it as it reads. That is how an operation is really
/// cancelled (something watching what it says decides it has seen enough), and
/// it stops a test mid-run with no timer in it.
#[derive(Default)]
struct Collector {
    events: Mutex<Vec<Owned>>,
    cancel: AtomicBool,
    /// Set the flag the first time a round is counted.
    stop_after_a_round: bool,
}

impl Collector {
    fn recording() -> Self {
        Self::default()
    }

    fn cancelling() -> Self {
        Self {
            stop_after_a_round: true,
            ..Self::default()
        }
    }

    fn push(&self, event: Event<'_>) {
        let owned = match event {
            Event::Enter { phase, depth } => Owned::Enter(phase, depth),
            Event::Leave {
                phase, depth, note, ..
            } => Owned::Leave(phase, depth, note.map(str::to_string)),
            Event::Message { level, depth, text } => Owned::Message(level, depth, text.to_string()),
            Event::Count { done, total, unit } => {
                if self.stop_after_a_round && unit == "round" {
                    self.cancel.store(true, Ordering::Relaxed);
                }
                Owned::Count(done, total, unit)
            }
            Event::Fraction { of_whole } => Owned::Fraction(of_whole),
            // The adjustment sets no status, and a status is live state a
            // finished entry would not keep anyway.
            Event::Status { .. } => return,
        };
        self.events.lock().unwrap().push(owned);
    }

    fn events(&self) -> Vec<Owned> {
        self.events.lock().unwrap().clone()
    }

    /// `(name, depth)` of every phase opened.
    fn entered(&self) -> Vec<(&'static str, u8)> {
        self.events()
            .into_iter()
            .filter_map(|event| match event {
                Owned::Enter(name, depth) => Some((name, depth)),
                _ => None,
            })
            .collect()
    }

    /// `(name, depth, note)` of every phase closed.
    fn left(&self) -> Vec<(&'static str, u8, Option<String>)> {
        self.events()
            .into_iter()
            .filter_map(|event| match event {
                Owned::Leave(name, depth, note) => Some((name, depth, note)),
                _ => None,
            })
            .collect()
    }

    /// Every phase boundary at `depth`, `Enter` and `Leave` told apart by the
    /// leading character.
    fn boundaries_at(&self, depth: u8) -> Vec<String> {
        self.events()
            .into_iter()
            .filter_map(|event| match event {
                Owned::Enter(name, d) if d == depth => Some(format!(">{name}")),
                Owned::Leave(name, d, _) if d == depth => Some(format!("<{name}")),
                _ => None,
            })
            .collect()
    }

    fn messages(&self) -> Vec<(Level, u8, String)> {
        self.events()
            .into_iter()
            .filter_map(|event| match event {
                Owned::Message(level, depth, text) => Some((level, depth, text)),
                _ => None,
            })
            .collect()
    }

    fn fractions(&self) -> Vec<f32> {
        self.events()
            .into_iter()
            .filter_map(|event| match event {
                Owned::Fraction(f) => Some(f),
                _ => None,
            })
            .collect()
    }

    fn counts(&self, unit: &str) -> Vec<(u64, Option<u64>)> {
        self.events()
            .into_iter()
            .filter_map(|event| match event {
                Owned::Count(done, total, u) if u == unit => Some((done, total)),
                _ => None,
            })
            .collect()
    }
}

/// Every number the adjustment writes into the value, as bit patterns: the
/// poses, the camera, the points and the patch frames. Bits rather than a
/// tolerance, because the question is whether reporting moved anything at all.
fn value_bits(recon: &SfmrReconstruction) -> Vec<u64> {
    let mut bits = Vec::new();
    for image in &recon.image_table.images {
        bits.extend(image.quaternion_wxyz.coords.iter().map(|c| c.to_bits()));
        bits.extend(image.translation_xyz.iter().map(|c| c.to_bits()));
    }
    for camera in &recon.image_table.cameras {
        let (fx, fy) = camera.focal_lengths();
        bits.push(fx.to_bits());
        bits.push(fy.to_bits());
    }
    for point in &recon.point_set.points {
        bits.extend(point.position.coords.iter().map(|c| c.to_bits()));
        bits.push(point.w.to_bits());
        bits.push(u64::from(point.error.to_bits()));
    }
    for column in [
        &recon.point_set.patch_u_halfvec_xyz,
        &recon.point_set.patch_v_halfvec_xyz,
    ] {
        if let Some(array) = column.as_ref() {
            bits.extend(array.iter().map(|v| u64::from(v.to_bits())));
        }
    }
    bits
}

/// The same for the report.
fn report_bits(report: &BundleAdjustReport) -> Vec<u64> {
    vec![
        report.images as u64,
        report.points as u64,
        report.observations as u64,
        report.points_deleted as u64,
        report.median_residual_before.to_bits(),
        report.median_residual_after.to_bits(),
        report.focal_before.to_bits(),
        report.focal_after.to_bits(),
        u64::from(report.focal_released),
    ]
}

fn same_bits(a: &[u64], b: &[u64], what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: different lengths");
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert_eq!(x, y, "{what} {i}: {x:016x} against {y:016x}");
    }
}

/// The defaults with the focal released, so the shared camera is part of the
/// answer the parity test compares.
fn released() -> BundleAdjustOptions {
    BundleAdjustOptions {
        opt_f: true,
        ..BundleAdjustOptions::default()
    }
}

#[test]
fn reporting_moves_not_one_bit_of_the_answer() {
    let source = perturbed();
    let (quiet, quiet_report) =
        bundle_adjust(&source, &released(), &Progress::none()).expect("well posed");

    let collector = Collector::recording();
    let sink = |event: Event<'_>| collector.push(event);
    // Detail on as well, so every phase this call can open is open while the
    // arithmetic runs.
    let progress = Progress::to(&sink).detailed(true);
    let (loud, loud_report) = bundle_adjust(&source, &released(), &progress).expect("well posed");

    assert!(
        !collector.events().is_empty(),
        "the recorded run reported nothing, so it proves nothing"
    );
    same_bits(&value_bits(&quiet), &value_bits(&loud), "value");
    same_bits(
        &report_bits(&quiet_report),
        &report_bits(&loud_report),
        "report",
    );
}

#[test]
fn a_recorded_run_names_its_stages() {
    let source = perturbed();
    let collector = Collector::recording();
    let sink = |event: Event<'_>| collector.push(event);

    bundle_adjust(
        &source,
        &BundleAdjustOptions::default(),
        &Progress::to(&sink),
    )
    .expect("well posed");

    // The four stages of the call, opened and closed in order at the top.
    assert_eq!(
        collector.boundaries_at(0),
        [
            ">gather arrays",
            "<gather arrays",
            ">residuals before",
            "<residuals before",
            ">solve",
            "<solve",
            ">write back",
            "<write back",
        ]
    );

    // The kernel's rounds sit under the solve, one per schedule round.
    let rounds: Vec<_> = collector
        .left()
        .into_iter()
        .filter(|(name, ..)| *name == "round")
        .collect();
    assert_eq!(rounds.len(), DEFAULT_SCHEDULE.len());
    assert!(rounds.iter().all(|(_, depth, _)| *depth == 1), "{rounds:?}");

    // The schedule is the solve's, not any one round's: the rounds fold into a
    // single row for a reader, and a trim true only of whichever ran last would
    // be worse than no trim at all.
    let trims = format!(
        "{} rounds, trim {} px",
        DEFAULT_SCHEDULE.len(),
        DEFAULT_SCHEDULE
            .iter()
            .map(|stage| format!("{}", stage.trim_px))
            .collect::<Vec<_>>()
            .join("/")
    );

    // Detail is off, so the stages inside an LM iteration are not recorded.
    assert!(
        collector
            .entered()
            .iter()
            .all(|(name, _)| *name != "linearise"),
        "detail was off"
    );

    // The size of the problem is stated once, beside the stages rather than
    // inside one of them; the schedule under the solve that runs it and not
    // under the empty-schedule call that runs no round; and a median under
    // each of the two stages that measured one, so the pair reads as a before
    // and an after over the same population.
    let obs = source.point_set.tracks.len();
    let said = collector.messages();
    assert_eq!(
        said.iter().map(|(_, depth, _)| *depth).collect::<Vec<_>>(),
        [0, 1, 1, 1],
        "{said:?}"
    );
    assert_eq!(
        said[0].2,
        format!("{IMAGES} images, {POINTS} points, {obs} observations")
    );
    assert_eq!(said[2].2, trims);
    let median = |said: &str| {
        said.strip_prefix("median ")
            .and_then(|rest| rest.strip_suffix(&format!(" px over {obs} observations")))
            .unwrap_or_else(|| panic!("not a median line: {said:?}"))
            .parse::<f64>()
            .expect("a median in px")
    };
    let (before, after) = (median(&said[1].2), median(&said[3].2));
    assert!(
        before > after,
        "the adjustment reported no improvement: {before} then {after}"
    );
}

#[test]
fn detail_adds_the_stages_inside_an_iteration() {
    let source = perturbed();
    let collector = Collector::recording();
    let sink = |event: Event<'_>| collector.push(event);

    bundle_adjust(
        &source,
        &BundleAdjustOptions::default(),
        &Progress::to(&sink).detailed(true),
    )
    .expect("well posed");

    // Under a round, which is under the solve.
    for name in ["linearise", "normal equations", "damping ladder"] {
        assert!(
            collector.entered().contains(&(name, 2)),
            "no {name} at depth 2"
        );
    }
    // Every iteration is counted against the budget it was given.
    let iterations = collector.counts("iteration");
    assert!(!iterations.is_empty());
    assert!(iterations
        .iter()
        .all(|&(done, total)| total == Some(DEFAULT_MAX_ITERS as u64)
            && done < DEFAULT_MAX_ITERS as u64));
}

#[test]
fn the_fraction_only_ever_moves_forward() {
    let source = perturbed();
    let collector = Collector::recording();
    let sink = |event: Event<'_>| collector.push(event);

    bundle_adjust(
        &source,
        &BundleAdjustOptions::default(),
        &Progress::to(&sink),
    )
    .expect("well posed");

    let fractions = collector.fractions();
    assert!(!fractions.is_empty(), "nothing moved the bar");
    for pair in fractions.windows(2) {
        assert!(pair[1] >= pair[0], "{} then {}", pair[0], pair[1]);
    }
    assert!(
        fractions.iter().all(|f| (0.0..=1.0).contains(f)),
        "{fractions:?}"
    );
    // The solve owns the middle of the range, so the last thing reported is
    // its last round ending: inside the range rather than at the end of it.
    let last = *fractions.last().expect("not empty");
    assert!((0.9..=1.0).contains(&last), "{last}");
}

#[test]
fn a_cancelled_adjustment_refuses_and_writes_nothing() {
    let source = perturbed();
    let poses: Vec<Vector3<f64>> = source
        .image_table
        .images
        .iter()
        .map(|i| i.translation_xyz)
        .collect();
    let positions: Vec<Point3<f64>> = source.point_set.points.iter().map(|p| p.position).collect();

    let collector = Collector::cancelling();
    let sink = |event: Event<'_>| collector.push(event);
    let progress = Progress::to(&sink).cancelled_by(&collector.cancel);

    let error = bundle_adjust(&source, &BundleAdjustOptions::default(), &progress)
        .err()
        .expect("the sink stopped it");
    assert_eq!(error, BundleAdjustError::Cancelled);

    // It stopped where it was told to: the first round finished, the second
    // never opened, and nothing was written back.
    let rounds = collector
        .entered()
        .iter()
        .filter(|(name, _)| *name == "round")
        .count();
    assert_eq!(rounds, 1);
    assert!(
        collector
            .entered()
            .iter()
            .all(|(name, _)| *name != "write back"),
        "a cancelled adjustment wrote something back"
    );

    for (image, translation) in source.image_table.images.iter().zip(&poses) {
        assert_eq!(image.translation_xyz, *translation);
    }
    for (point, position) in source.point_set.points.iter().zip(&positions) {
        assert_eq!(point.position, *position);
    }
}

#[test]
fn a_silent_run_reaches_no_sink_it_was_not_handed() {
    let source = perturbed();
    let collector = Collector::recording();
    let sink = |event: Event<'_>| collector.push(event);
    // Built, and deliberately not passed: `Progress::none()` carries no sink,
    // and nothing in the call reaches one by any other route.
    let _elsewhere = Progress::to(&sink);

    bundle_adjust(&source, &BundleAdjustOptions::default(), &Progress::none()).expect("well posed");

    assert!(collector.events().is_empty(), "{:?}", collector.events());
}
