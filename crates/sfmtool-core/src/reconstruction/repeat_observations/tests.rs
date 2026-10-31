// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Dropping repeat observations: which observation each image keeps, that
//! every per-observation column stays in step, that the changed points are
//! scored again, and that a value with no repeat comes back unchanged.
//!
//! The fixture is three cameras on a short arc looking at the origin and six
//! points, each observation at the exact projection plus an offset the test
//! chooses, so every reprojection error is known.

use nalgebra::{Point3, UnitQuaternion, Vector3};
use ndarray::Array2;

use crate::camera::{CameraIntrinsics, CameraModel};
use crate::progress::Progress;
use crate::reconstruction::data::{
    ObservationSource, Point3D, SfmrImage, SfmrReconstruction, TrackObservation,
};

use super::*;

const IMAGES: usize = 3;
const STORED_ERROR: f32 = 0.25;

/// One observation of the fixture: its image, and the pixel offset from the
/// point's exact projection, or `None` for a pixel that is not known (NaN).
type Row = (u32, Option<[f32; 2]>);

/// The fixture's points: `(x, y, z, w)` and their observations, in track
/// order.
fn scene_points() -> Vec<([f64; 4], Vec<Row>)> {
    vec![
        // 0: finite, two in image 1, the worse first.
        (
            [0.0, 0.0, 0.0, 1.0],
            vec![
                (0, Some([0.0, 0.0])),
                (1, Some([3.0, 0.0])),
                (1, Some([0.0, 0.5])),
                (2, Some([0.0, 0.0])),
            ],
        ),
        // 1: at infinity, looking down -z, two in image 2, the worse first.
        (
            [0.0, 0.0, -1.0, 0.0],
            vec![
                (0, Some([0.0, 0.0])),
                (1, Some([0.0, 0.0])),
                (2, Some([2.0, 0.0])),
                (2, Some([0.0, 1.0])),
            ],
        ),
        // 2: a tie in image 0, two observations on the same pixel.
        (
            [0.3, 0.2, 0.0, 1.0],
            vec![
                (0, Some([1.0, 0.0])),
                (0, Some([1.0, 0.0])),
                (1, Some([0.0, 0.0])),
            ],
        ),
        // 3: no repeat.
        (
            [-0.3, 0.1, 0.0, 1.0],
            vec![(0, Some([0.0, 0.0])), (1, Some([0.0, 0.0]))],
        ),
        // 4: an unknown pixel first, a far one second, in image 0.
        (
            [0.1, -0.3, 0.0, 1.0],
            vec![(0, None), (0, Some([50.0, 0.0])), (2, Some([0.0, 0.0]))],
        ),
        // 5: behind every camera, two in image 0: none projects.
        (
            [0.0, 0.0, 12.0, 1.0],
            vec![(0, Some([0.0, 0.0])), (0, Some([0.0, 0.0])), (1, None)],
        ),
    ]
}

fn pinhole() -> CameraIntrinsics {
    CameraIntrinsics {
        model: CameraModel::SimplePinhole {
            focal_length: 500.0,
            principal_point_x: 320.0,
            principal_point_y: 240.0,
        },
        width: 640,
        height: 480,
        rolling_shutter: None,
    }
}

/// The exact projection of `point` into image `i`, or the principal point for
/// one the camera cannot image.
fn project(recon: &SfmrReconstruction, i: usize, point: &Point3D) -> [f32; 2] {
    let image = &recon.image_set.images[i];
    let local = if point.is_at_infinity() {
        image.quaternion_wxyz * point.position.coords
    } else {
        image.quaternion_wxyz * point.position.coords + image.translation_xyz
    };
    pinhole()
        .ray_to_pixel([local.x, local.y, local.z])
        .map_or([320.0, 240.0], |(u, v)| [u as f32, v as f32])
}

/// The fixture as a `sift_files` value with the inline keypoint copy, a
/// distinct feature index per observation and an observation confidence equal
/// to the row, so a row's columns can be told apart after the selection.
fn scene(points: &[([f64; 4], Vec<Row>)]) -> SfmrReconstruction {
    let mut recon = SfmrReconstruction::demo(1);
    recon.image_set.cameras = vec![pinhole()];
    recon.image_set.images = (0..IMAGES)
        .map(|i| {
            let angle = 0.2 * (i as f64 - (IMAGES as f64 - 1.0) / 2.0);
            let centre = Vector3::new(8.0 * angle.sin(), 0.0, 8.0 * angle.cos());
            let rotation = UnitQuaternion::face_towards(&centre, &Vector3::y()).inverse();
            SfmrImage {
                name: format!("image_{i:03}.jpg"),
                camera_index: 0,
                quaternion_wxyz: rotation,
                translation_xyz: -(rotation * centre),
                intrinsic_adjustment: Default::default(),
            }
        })
        .collect();
    let stats = recon.image_set.depth_statistics.images[0].clone();
    recon.image_set.depth_statistics.images = vec![stats; IMAGES];
    recon.image_set.depth_histogram_counts =
        vec![recon.image_set.depth_histogram_counts[0].clone(); IMAGES];

    recon.point_set.points = points
        .iter()
        .map(|([x, y, z, w], _)| Point3D {
            position: Point3::new(*x, *y, *z),
            w: *w,
            color: [10, 20, 30],
            rms_reprojection_error: STORED_ERROR,
            normal: Vector3::new(0.0, 0.0, 1.0),
        })
        .collect();

    let mut tracks = Vec::new();
    let mut counts = Vec::new();
    let mut keypoints: Vec<[f32; 2]> = Vec::new();
    let mut feature_indexes = Vec::new();
    let mut next_feature = [0u32; IMAGES];
    for (p, (_, rows)) in points.iter().enumerate() {
        for &(image, offset) in rows {
            let exact = project(&recon, image as usize, &recon.point_set.points[p]);
            keypoints.push(match offset {
                Some([du, dv]) => [exact[0] + du, exact[1] + dv],
                None => [f32::NAN, f32::NAN],
            });
            tracks.push(TrackObservation {
                image_index: image,
                point_index: p as u32,
            });
            feature_indexes.push(next_feature[image as usize]);
            next_feature[image as usize] += 1;
        }
        counts.push(rows.len() as u32);
    }
    let mut keypoints_xy = Array2::<f32>::zeros((keypoints.len(), 2));
    for (row, uv) in keypoints.iter().enumerate() {
        keypoints_xy[[row, 0]] = uv[0];
        keypoints_xy[[row, 1]] = uv[1];
    }
    let set = &mut recon.point_set;
    set.observation_confidence = Some((0..tracks.len() as u8).collect());
    set.tracks = tracks;
    set.observation_counts = counts;
    set.observations = ObservationSource::SiftFiles {
        feature_indexes,
        keypoints_xy: Some(keypoints_xy),
        feature_tool_hashes: vec![[0u8; 16]; IMAGES],
        sift_content_hashes: vec![[0u8; 16]; IMAGES],
    };
    recon.metadata.feature_source = sfmtool_sfmr_format::FEATURE_SOURCE_SIFT_FILES.to_string();
    recon.rebuild_derived_fields();
    recon
}

/// The rows of the fixture each point keeps, by their original row, in track
/// order.
fn kept_original_rows(recon: &SfmrReconstruction) -> Vec<Vec<u8>> {
    let confidence = recon.point_set.observation_confidence.as_ref().unwrap();
    (0..recon.point_count())
        .map(|p| {
            let start = recon.point_set.observation_offsets[p];
            (0..recon.observations_for_point(p).len())
                .map(|k| confidence[start + k])
                .collect()
        })
        .collect()
}

#[test]
fn each_image_keeps_its_best_observation() {
    let before = scene(&scene_points());
    let (after, report) = before
        .drop_repeat_observations(&Progress::none())
        .expect("the fixture carries its pixels inline");

    // Original rows: point 0 is 0..4, point 1 is 4..8, point 2 is 8..11,
    // point 3 is 11..13, point 4 is 13..16, point 5 is 16..19.
    assert_eq!(
        kept_original_rows(&after),
        vec![
            // The lower error, though it comes second.
            vec![0, 2, 3],
            // At infinity, through the rotation alone: the lower error.
            vec![4, 5, 7],
            // A tie keeps the first.
            vec![8, 10],
            vec![11, 12],
            // A pixel that is not known ranks after one that projects.
            vec![14, 15],
            // Nothing projects, so the first is kept.
            vec![16, 18],
        ]
    );
    assert_eq!(
        report,
        RepeatObservationsReport {
            observations_dropped: 5,
            points_changed: 5,
        }
    );
    assert_eq!(after.observation_count(), before.observation_count() - 5);
    assert_eq!(after.point_count(), before.point_count());
    assert_eq!(
        after.metadata.observation_count,
        after.observation_count() as u32
    );

    // No point loses an image, and none moves.
    for p in 0..before.point_count() {
        let mut images_before = before.track_image_indices(p);
        images_before.dedup();
        let images_after = after.track_image_indices(p);
        assert_eq!(images_after, images_before, "point {p}");
        assert_eq!(
            after.point_set.points[p].position,
            before.point_set.points[p].position
        );
        assert_eq!(after.point_set.points[p].w, before.point_set.points[p].w);
    }
}

#[test]
fn every_observation_column_follows_its_row() {
    let before = scene(&scene_points());
    let (after, _) = before.drop_repeat_observations(&Progress::none()).unwrap();

    let before_features = before.feature_indexes().unwrap();
    let before_keypoints = before.keypoints_xy().unwrap();
    let after_features = after.feature_indexes().unwrap();
    let after_keypoints = after.keypoints_xy().unwrap();
    let confidence = after.point_set.observation_confidence.as_ref().unwrap();
    assert_eq!(after_features.len(), after.observation_count());
    assert_eq!(after_keypoints.nrows(), after.observation_count());
    for (row, &original) in confidence.iter().enumerate() {
        let original = original as usize;
        assert_eq!(
            after.point_set.tracks[row].image_index,
            before.point_set.tracks[original].image_index
        );
        assert_eq!(after_features[row], before_features[original]);
        let (a, b) = (after_keypoints.row(row), before_keypoints.row(original));
        assert!(a
            .iter()
            .zip(b.iter())
            .all(|(x, y)| x == y || (x.is_nan() && y.is_nan())));
    }
    after.validate_observation_columns().unwrap();
    // The feature map names only the features that are kept.
    let image0 = &after.point_set.image_feature_to_point[0];
    assert_eq!(image0.len(), confidence_rows_in_image(&after, 0));
}

fn confidence_rows_in_image(recon: &SfmrReconstruction, image: u32) -> usize {
    recon
        .point_set
        .tracks
        .iter()
        .filter(|t| t.image_index == image)
        .count()
}

#[test]
fn the_changed_points_are_scored_again() {
    let before = scene(&scene_points());
    let (after, _) = before.drop_repeat_observations(&Progress::none()).unwrap();
    let errors: Vec<f32> = after
        .point_set
        .points
        .iter()
        .map(|p| p.rms_reprojection_error)
        .collect();

    let close = |a: f32, b: f64| (a as f64 - b).abs() < 1e-3;
    // 0 keeps offsets 0, 0.5 and 0.
    assert!(close(errors[0], (0.25f64 / 3.0).sqrt()), "{}", errors[0]);
    // 1 keeps offsets 0, 0 and 1.
    assert!(close(errors[1], (1.0f64 / 3.0).sqrt()), "{}", errors[1]);
    // 2 keeps offsets 1 and 0.
    assert!(close(errors[2], (1.0f64 / 2.0).sqrt()), "{}", errors[2]);
    // 3 lost nothing, so keeps the value it stored.
    assert_eq!(errors[3], STORED_ERROR);
    // 4 keeps offsets 50 and 0.
    assert!(close(errors[4], (2500.0f64 / 2.0).sqrt()), "{}", errors[4]);
    // 5 has no observation that projects.
    assert_eq!(errors[5], 0.0);
}

#[test]
fn the_depth_statistics_follow_the_tracks_and_the_normals_stay() {
    let mut before = scene(&scene_points());
    before.point_set.has_normals = true;
    let (after, _) = before.drop_repeat_observations(&Progress::none()).unwrap();

    let mut derived = after.clone();
    derived.point_set.has_normals = false;
    derived.recompute_depth_statistics().unwrap();
    assert_eq!(
        after.image_set.depth_histogram_counts,
        derived.image_set.depth_histogram_counts
    );
    for (a, b) in after.point_set.points.iter().zip(&before.point_set.points) {
        assert_eq!(a.normal, b.normal);
    }
    assert!(after.point_set.has_normals);
}

#[test]
fn a_value_without_repeats_comes_back_unchanged() {
    let points: Vec<_> = scene_points()
        .into_iter()
        .map(|(xyzw, rows)| {
            let mut seen = Vec::new();
            let rows: Vec<Row> = rows
                .into_iter()
                .filter(|(image, _)| {
                    let first = !seen.contains(image);
                    seen.push(*image);
                    first
                })
                .collect();
            (xyzw, rows)
        })
        .collect();
    let before = scene(&points);
    let (after, report) = before.drop_repeat_observations(&Progress::none()).unwrap();

    assert_eq!(report, RepeatObservationsReport::default());
    assert_eq!(after.point_set.tracks, before.point_set.tracks);
    assert_eq!(after.feature_indexes(), before.feature_indexes());
    assert_eq!(
        after.point_set.observation_confidence,
        before.point_set.observation_confidence
    );
    for (a, b) in after.point_set.points.iter().zip(&before.point_set.points) {
        assert_eq!(a.rms_reprojection_error, b.rms_reprojection_error);
    }
}
