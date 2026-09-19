// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Re-solving the points of a whole reconstruction: what the value that comes
//! back holds, what it leaves alone, and every refusal.
//!
//! The fixture is a synthetic scene whose truth is known -- three cameras on a
//! short arc looking at a handful of points, every observation the exact
//! projection -- so a retriangulation run from a copy whose points have been
//! nudged has somewhere to converge to.

use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

use nalgebra::{Point3, UnitQuaternion, Vector3};
use ndarray::{Array2, Array4};

use crate::camera::{CameraIntrinsics, CameraModel};
use crate::progress::{Event, Progress};
use crate::reconstruction::data::{
    ObservationSource, Point3D, PointConstraintColumns, SfmrImage, SfmrReconstruction,
    TrackObservation,
};
use crate::reconstruction::edited::{EditedReconstruction, PointMap};
use crate::reconstruction::triangulation::{
    retriangulate_points, PointVerdict, RetriangulateError, RetriangulateOptions,
    RetriangulateWhich,
};

use sfmtool_sfmr_format::{POINT_CONSTRAINT_FREE, POINT_CONSTRAINT_HELD, POINT_CONSTRAINT_RANGED};

const IMG_W: u32 = 640;
const IMG_H: u32 = 480;
const IMAGES: usize = 3;
/// How far a re-solved point may stand from the truth. The pixels are an
/// `f32` column, so the rays they state are the truth's to about a part in
/// ten million of the frame, and a place read off them cannot be nearer
/// than that.
const TRUTH_TOLERANCE: f64 = 1e-3;
const POINTS: usize = 5;

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
            focal_length: 500.0,
            principal_point_x: IMG_W as f64 / 2.0,
            principal_point_y: IMG_H as f64 / 2.0,
        },
        width: IMG_W,
        height: IMG_H,
    }
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

/// The truth: cameras on an arc at radius 8 around the origin, points in a
/// box inside it, every observation the exact projection, and one square
/// patch frame per point so a rescale is measurable.
fn truth() -> SfmrReconstruction {
    let mut recon = SfmrReconstruction::demo(1);
    recon.image_table.cameras = vec![pinhole()];
    recon.image_table.images = (0..IMAGES)
        .map(|i| {
            let angle = 0.3 * (i as f64 - (IMAGES as f64 - 1.0) / 2.0);
            let centre = Vector3::new(8.0 * angle.sin(), 0.0, 8.0 * angle.cos());
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
        sfmtool_sfmr_format::THUMBNAIL_SIZE,
        sfmtool_sfmr_format::THUMBNAIL_SIZE,
        3,
    )));
    let stats = recon.image_table.depth_statistics.images[0].clone();
    recon.image_table.depth_statistics.images = vec![stats; IMAGES];
    recon.image_table.depth_histogram_counts =
        vec![recon.image_table.depth_histogram_counts[0].clone(); IMAGES];

    recon.point_set.points = (0..POINTS)
        .map(|p| Point3D {
            position: Point3::new(1.5 * jitter(p, 1), 1.5 * jitter(p, 2), 1.0 * jitter(p, 3)),
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
        counts.iter().all(|&c| c == IMAGES as u32),
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
    set.point_constraints = Some(PointConstraintColumns::all_free(POINTS));
    recon.metadata.feature_source =
        sfmtool_sfmr_format::FEATURE_SOURCE_EMBEDDED_PATCHES.to_string();
    recon.rebuild_derived_fields();
    recon
}

/// The truth with every point nudged off it, its pixels left where the truth
/// put them: what a retriangulation has to find its way back from.
fn nudged() -> SfmrReconstruction {
    let mut recon = truth();
    for (p, point) in recon.point_set.points.iter_mut().enumerate() {
        point.position += Vector3::new(
            0.2 * jitter(p, 21),
            0.2 * jitter(p, 22),
            0.2 * jitter(p, 23),
        );
    }
    recon
}

fn edited(recon: SfmrReconstruction) -> EditedReconstruction {
    EditedReconstruction::new(Arc::new(recon))
}

/// How far point `p` of `recon` stands from the truth's.
fn error(recon: &SfmrReconstruction, truth: &SfmrReconstruction, p: usize) -> f64 {
    (recon.point_set.points[p].position - truth.point_set.points[p].position).norm()
}

/// Set point `p`'s constraint triple.
fn constrain(recon: &mut SfmrReconstruction, p: usize, kind: u8, distance: f64, reference: u32) {
    let columns = recon
        .point_set
        .point_constraints
        .as_mut()
        .expect("the fixture carries the columns");
    columns.point_constraints[p] = kind;
    columns.constraint_distances[p] = distance;
    columns.constraint_reference_images[p] = reference;
}

#[test]
fn every_point_lands_back_on_the_truth_its_pixels_state() {
    let truth = truth();
    let start = edited(nudged());
    let before: Vec<f64> = (0..POINTS).map(|p| error(&start.base, &truth, p)).collect();
    assert!(before.iter().all(|&e| e > 0.01), "{before:?}");

    let (next, map, report) = retriangulate_points(
        &start,
        RetriangulateWhich::All,
        &RetriangulateOptions::default(),
        &Progress::none(),
    )
    .expect("the fixture retriangulates");

    assert_eq!(report.read, POINTS);
    assert_eq!(report.observations, POINTS * IMAGES);
    assert_eq!(report.moved, POINTS);
    assert_eq!(report.crossed, 0);
    assert_eq!(report.kept, 0);
    assert_eq!(report.held, 0);
    assert_eq!(report.census.finite, POINTS);
    for p in 0..POINTS {
        assert!(
            error(&next.base, &truth, p) < TRUTH_TOLERANCE,
            "point {p} landed at {:?}",
            next.base.point_set.points[p].position
        );
    }
    // Nothing was deleted and nothing created, so every index still means
    // what it meant.
    for p in 0..POINTS as u32 {
        assert_eq!(map.forward(p), Some(p));
    }
    assert!(report.median_shift > 0.0, "{}", report.median_shift);
}

#[test]
fn the_input_is_left_exactly_as_it_was() {
    let start = edited(nudged());
    let before = start.base.point_set.points.clone();
    let _ = retriangulate_points(
        &start,
        RetriangulateWhich::All,
        &RetriangulateOptions::default(),
        &Progress::none(),
    )
    .expect("the fixture retriangulates");
    assert_eq!(start.base.point_set.points, before);
}

#[test]
fn a_held_point_is_never_read_and_never_moves() {
    let mut recon = nudged();
    constrain(&mut recon, 2, POINT_CONSTRAINT_HELD, f64::NAN, u32::MAX);
    let held_at = recon.point_set.points[2].position;
    let start = edited(recon);

    let (next, _, report) = retriangulate_points(
        &start,
        RetriangulateWhich::All,
        &RetriangulateOptions::default(),
        &Progress::none(),
    )
    .expect("the fixture retriangulates");

    assert_eq!(report.held, 1);
    assert_eq!(report.read, POINTS - 1);
    assert_eq!(report.moved, POINTS - 1);
    assert_eq!(next.base.point_set.points[2].position, held_at);
}

#[test]
fn a_held_point_on_its_own_leaves_nothing_to_solve() {
    let mut recon = nudged();
    constrain(&mut recon, 1, POINT_CONSTRAINT_HELD, f64::NAN, u32::MAX);
    let start = edited(recon);
    assert_eq!(
        retriangulate_points(
            &start,
            RetriangulateWhich::These(&[1]),
            &RetriangulateOptions::default(),
            &Progress::none(),
        )
        .expect_err("a held point is not solved"),
        RetriangulateError::NothingToSolve
    );
}

#[test]
fn a_ranged_point_keeps_the_distance_the_value_states() {
    let truth = truth();
    let centre = truth.image_table.images[0].camera_center();
    let range = (truth.point_set.points[3].position - centre).norm() + 1.0;
    let mut recon = nudged();
    constrain(&mut recon, 3, POINT_CONSTRAINT_RANGED, range, 0);
    let start = edited(recon);

    let (next, _, report) = retriangulate_points(
        &start,
        RetriangulateWhich::All,
        &RetriangulateOptions::default(),
        &Progress::none(),
    )
    .expect("the fixture retriangulates");

    assert_eq!(report.census.ranged, 1);
    let got = (next.base.point_set.points[3].position - centre).norm();
    assert!((got - range).abs() < 1e-9 * range, "{got} is not {range}");
    // And the free points beside it are unaffected.
    assert!(error(&next.base, &truth, 0) < TRUTH_TOLERANCE);
}

#[test]
fn a_ranged_point_naming_no_reference_is_solved_free() {
    let truth = truth();
    let mut recon = nudged();
    constrain(
        &mut recon,
        3,
        POINT_CONSTRAINT_RANGED,
        2.0,
        sfmtool_sfmr_format::NO_REFERENCE_IMAGE,
    );
    let start = edited(recon);
    let (next, _, report) = retriangulate_points(
        &start,
        RetriangulateWhich::All,
        &RetriangulateOptions::default(),
        &Progress::none(),
    )
    .expect("the fixture retriangulates");
    assert_eq!(report.census.ranged, 0);
    assert!(error(&next.base, &truth, 3) < TRUTH_TOLERANCE);
}

#[test]
fn a_cancelled_retriangulation_writes_nothing() {
    let start = edited(nudged());
    let flag = AtomicBool::new(true);
    let progress = Progress::none().cancelled_by(&flag);
    assert_eq!(
        retriangulate_points(
            &start,
            RetriangulateWhich::All,
            &RetriangulateOptions::default(),
            &progress,
        )
        .expect_err("a cancelled call has no answer"),
        RetriangulateError::Cancelled
    );
}

#[test]
fn a_cancel_that_lands_while_the_arrays_are_gathered_still_stops_it() {
    // The flag is set by the first thing the gather reports, so it is
    // already set when the stage boundary after it reads it.
    let start = edited(nudged());
    let flag = AtomicBool::new(false);
    let sink = |_: Event<'_>| flag.store(true, std::sync::atomic::Ordering::Relaxed);
    let progress = Progress::to(&sink).cancelled_by(&flag);
    assert_eq!(
        retriangulate_points(
            &start,
            RetriangulateWhich::All,
            &RetriangulateOptions::default(),
            &progress,
        )
        .expect_err("a cancelled call has no answer"),
        RetriangulateError::Cancelled
    );
}

#[test]
fn one_point_is_an_overlay_edit_over_the_very_same_base() {
    let truth = truth();
    let start = edited(nudged());
    let (next, map, report) = retriangulate_points(
        &start,
        RetriangulateWhich::These(&[1]),
        &RetriangulateOptions::default(),
        &Progress::none(),
    )
    .expect("the fixture retriangulates");

    assert_eq!(report.read, 1);
    assert_eq!(report.moved, 1);
    assert_eq!(report.observations, IMAGES);
    assert!(
        Arc::ptr_eq(&next.base, &start.base),
        "the base was rewritten"
    );
    assert_eq!(next.point_count(), POINTS);

    // Delete-and-re-add: the point took a new index, and the map says so.
    let to = map.forward(1).expect("the point survived");
    assert_ne!(to, 1);
    assert!(matches!(map, PointMap::Replaced(_)));
    let moved = next.point(to).expect("the replacement is live");
    assert!(
        (moved.point().position - truth.point_set.points[1].position).norm() < TRUTH_TOLERANCE,
        "{:?}",
        moved.point().position
    );
    // And nothing else moved.
    assert_eq!(map.forward(0), Some(0));
    assert_eq!(
        next.point(0).expect("still live").point().position,
        start.base.point_set.points[0].position
    );
}

#[test]
fn a_point_that_does_not_move_takes_no_new_index() {
    // The first pass settles the point on what its pixels state; the second
    // reads the same pixels at the same poses, so it has nothing to do and
    // the point keeps the index the first pass gave it.
    let start = edited(truth());
    let (once, first, _) = retriangulate_points(
        &start,
        RetriangulateWhich::These(&[1]),
        &RetriangulateOptions::default(),
        &Progress::none(),
    )
    .expect("the fixture retriangulates");
    let settled = first.forward(1).expect("the point survived");

    let (twice, map, report) = retriangulate_points(
        &once,
        RetriangulateWhich::These(&[settled]),
        &RetriangulateOptions::default(),
        &Progress::none(),
    )
    .expect("the fixture retriangulates");
    assert_eq!(report.read, 1);
    assert_eq!(report.moved, 0);
    assert_eq!(map.forward(settled), Some(settled));
    assert_eq!(twice.point_count(), POINTS);
    assert_eq!(twice.added.points.len(), once.added.points.len());
}

#[test]
fn a_point_too_few_observations_place_keeps_where_it_stood() {
    let mut recon = nudged();
    // Point 4 keeps one sighting, which states a direction and no place.
    let first = recon
        .point_set
        .tracks
        .iter()
        .position(|o| o.point_index == 4)
        .expect("point 4 is seen");
    let rows: Vec<usize> = (0..recon.point_set.tracks.len())
        .filter(|&row| recon.point_set.tracks[row].point_index != 4 || row == first)
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
    recon.point_set.observation_counts[4] = 1;
    recon.rebuild_derived_fields();
    let stood_at = recon.point_set.points[4].position;

    let start = edited(recon);
    let (next, _, report) = retriangulate_points(
        &start,
        RetriangulateWhich::All,
        &RetriangulateOptions::default(),
        &Progress::none(),
    )
    .expect("the fixture retriangulates");

    assert_eq!(report.kept, 1);
    assert_eq!(report.census.few, 1);
    assert_eq!(next.base.point_set.points[4].position, stood_at);
    assert_eq!(next.base.point_set.points[4].w, 1.0);
}

#[test]
fn a_floor_wide_enough_turns_every_point_into_a_direction() {
    let start = edited(nudged());
    let (next, _, report) = retriangulate_points(
        &start,
        RetriangulateWhich::All,
        &RetriangulateOptions {
            // Wider than the arc, so no pair of rays opens past it.
            floor_rad: Some(1.0),
            ..RetriangulateOptions::default()
        },
        &Progress::none(),
    )
    .expect("the fixture retriangulates");

    assert_eq!(report.census.thin, POINTS);
    assert_eq!(report.crossed, POINTS);
    assert_eq!(report.moved, POINTS);
    assert!(next
        .base
        .point_set
        .points
        .iter()
        .all(|p| p.is_at_infinity()));
    assert_eq!(next.base.point_set.infinity_point_count, POINTS);
    // A crossing has no distance in the scene, so none is reported.
    assert!(report.median_shift.is_nan());
}

#[test]
fn a_direction_is_read_as_one_and_stays_one() {
    let mut recon = nudged();
    recon.point_set.points[0].w = 0.0;
    recon.point_set.points[0].position = Point3::new(0.0, 0.0, -1.0);
    recon.rebuild_derived_fields();
    let start = edited(recon);
    let (next, _, report) = retriangulate_points(
        &start,
        RetriangulateWhich::All,
        &RetriangulateOptions::default(),
        &Progress::none(),
    )
    .expect("the fixture retriangulates");
    assert_eq!(report.census.marked, 1);
    assert_eq!(report.crossed, 0);
    assert!(next.base.point_set.points[0].is_at_infinity());
}

#[test]
fn the_patch_frame_of_a_moved_point_keeps_its_angular_size() {
    let start = edited(nudged());
    let before = start
        .base
        .point_set
        .patch_u_halfvec_xyz
        .as_ref()
        .expect("the fixture carries frames")[[0, 0]];
    let (next, _, _) = retriangulate_points(
        &start,
        RetriangulateWhich::All,
        &RetriangulateOptions::default(),
        &Progress::none(),
    )
    .expect("the fixture retriangulates");
    let after = next
        .base
        .point_set
        .patch_u_halfvec_xyz
        .as_ref()
        .expect("the frames survive")[[0, 0]];
    let ratio = f64::from(after / before);
    let was = start
        .base
        .image_table
        .placement_scale(&start.base.point_set.points[0].position);
    let now = next
        .base
        .image_table
        .placement_scale(&next.base.point_set.points[0].position);
    assert!(
        (ratio - now / was).abs() < 1e-5,
        "{ratio} is not {}",
        now / was
    );
}

#[test]
fn mixed_cameras_are_refused() {
    let mut recon = nudged();
    recon.image_table.cameras.push(pinhole());
    recon.image_table.images[1].camera_index = 1;
    let start = edited(recon);
    assert_eq!(
        retriangulate_points(
            &start,
            RetriangulateWhich::All,
            &RetriangulateOptions::default(),
            &Progress::none(),
        )
        .expect_err("two lenses are refused"),
        RetriangulateError::MixedCameras { cameras: 2 }
    );
}

#[test]
fn a_value_with_no_pose_is_refused() {
    let mut recon = nudged();
    for image in recon.image_table.images.iter_mut() {
        image.translation_xyz = Vector3::new(f64::NAN, f64::NAN, f64::NAN);
    }
    let start = edited(recon);
    assert_eq!(
        retriangulate_points(
            &start,
            RetriangulateWhich::All,
            &RetriangulateOptions::default(),
            &Progress::none(),
        )
        .expect_err("an unposed value is refused"),
        RetriangulateError::NoPosedImages
    );
}

#[test]
fn an_index_that_names_no_live_point_is_refused() {
    let start = edited(nudged());
    assert_eq!(
        retriangulate_points(
            &start,
            RetriangulateWhich::These(&[99]),
            &RetriangulateOptions::default(),
            &Progress::none(),
        )
        .expect_err("index 99 names nothing"),
        RetriangulateError::NoSuchPoint(99)
    );
}

#[test]
fn an_overlay_edit_is_folded_in_before_every_point_is_re_solved() {
    let mut start = edited(nudged());
    start.delete_point(0).expect("point 0 is live");
    let (next, map, report) = retriangulate_points(
        &start,
        RetriangulateWhich::All,
        &RetriangulateOptions::default(),
        &Progress::none(),
    )
    .expect("the fixture retriangulates");
    assert_eq!(report.read, POINTS - 1);
    assert_eq!(next.base.point_count(), POINTS - 1);
    assert!(next.deleted_points.is_empty());
    // The deleted index resolves to nothing, and the rest shift down.
    assert_eq!(map.forward(0), None);
    assert_eq!(map.forward(1), Some(0));
}

#[test]
fn the_census_of_one_point_names_its_one_verdict() {
    let start = edited(nudged());
    let (_, _, report) = retriangulate_points(
        &start,
        RetriangulateWhich::These(&[2]),
        &RetriangulateOptions::default(),
        &Progress::none(),
    )
    .expect("the fixture retriangulates");
    assert_eq!(report.census.sole_verdict(), Some(PointVerdict::Finite));

    let (_, _, whole) = retriangulate_points(
        &start,
        RetriangulateWhich::All,
        &RetriangulateOptions::default(),
        &Progress::none(),
    )
    .expect("the fixture retriangulates");
    assert_eq!(whole.census.sole_verdict(), None);
}

#[test]
fn a_free_constraint_column_leaves_every_point_free() {
    let mut recon = nudged();
    for p in 0..POINTS {
        constrain(&mut recon, p, POINT_CONSTRAINT_FREE, f64::NAN, u32::MAX);
    }
    let start = edited(recon);
    let (_, _, report) = retriangulate_points(
        &start,
        RetriangulateWhich::All,
        &RetriangulateOptions::default(),
        &Progress::none(),
    )
    .expect("the fixture retriangulates");
    assert_eq!(report.held, 0);
    assert_eq!(report.census.ranged, 0);
    assert_eq!(report.census.finite, POINTS);
}

#[test]
fn the_stages_are_named_for_a_reader_watching() {
    let seen = Mutex::new(Vec::new());
    let sink = |event: Event<'_>| {
        if let Event::Leave { phase, .. } = event {
            seen.lock().unwrap().push(phase);
        }
    };
    let start = edited(nudged());
    let _ = retriangulate_points(
        &start,
        RetriangulateWhich::All,
        &RetriangulateOptions::default(),
        &Progress::to(&sink),
    )
    .expect("the fixture retriangulates");
    let seen = seen.lock().unwrap();
    assert_eq!(
        *seen,
        ["gather observations", "retriangulate", "write back"],
        "{seen:?}"
    );
}
