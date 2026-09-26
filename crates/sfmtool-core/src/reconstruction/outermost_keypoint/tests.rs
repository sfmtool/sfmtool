// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The outermost keypoint, observed and detected.
//!
//! The fixture is the demo reconstruction with a `.sift` file per image in a
//! temporary workspace, every feature on a horizontal line out from the
//! principal point at a radius that grows with its index, and one more feature
//! per image, further out than any, that no observation uses.

use ndarray::{Array2, Array3};
use tempfile::TempDir;

use super::*;
use crate::reconstruction::data::ObservationSource;

const POINTS: usize = 12;
/// The radius of the one feature per image no observation uses.
const UNOBSERVED_RADIUS: f64 = 150.0;

/// Where feature `f` of any image lies under `camera`.
fn feature_xy(camera: &CameraIntrinsics, f: usize) -> [f32; 2] {
    let (cx, cy) = camera.principal_point();
    [(cx + 10.0 + f as f64) as f32, cy as f32]
}

fn fixture(dir: &TempDir) -> SfmrReconstruction {
    let mut recon = SfmrReconstruction::demo(POINTS);
    recon.workspace_dir = dir.path().to_path_buf();
    recon.metadata.workspace.contents.feature_prefix_dir = "features".into();
    for image in 0..recon.image_table.images.len() {
        let camera =
            &recon.image_table.cameras[recon.image_table.images[image].camera_index as usize];
        let observed = recon.point_set.max_track_feature_index[image] as usize + 1;
        let count = observed + 1;
        let mut positions = Array2::<f32>::zeros((count, 2));
        for f in 0..observed {
            let [x, y] = feature_xy(camera, f);
            positions[[f, 0]] = x;
            positions[[f, 1]] = y;
        }
        let (cx, cy) = camera.principal_point();
        positions[[observed, 0]] = (cx + UNOBSERVED_RADIUS) as f32;
        positions[[observed, 1]] = cy as f32;
        let data = sfmtool_sift_format::SiftData {
            feature_tool_metadata: sfmtool_sift_format::FeatureToolMetadata {
                feature_tool: "test".into(),
                feature_type: "sift".into(),
                feature_options: serde_json::json!({}),
            },
            metadata: sfmtool_sift_format::SiftMetadata {
                version: sfmtool_sift_format::SIFT_FORMAT_VERSION,
                image_name: recon.image_table.images[image].name.clone(),
                image_file_xxh128: "0".repeat(32),
                image_file_size: 1,
                image_width: camera.width,
                image_height: camera.height,
                feature_count: count as u32,
            },
            content_hash: sfmtool_sift_format::SiftContentHash::default(),
            positions_xy: positions,
            affine_shapes: Array3::<f32>::zeros((count, 2, 2)),
            descriptors: Array2::<u8>::zeros((count, 128)),
            thumbnail_y_x_rgb: Array3::<u8>::zeros((128, 128, 3)),
        };
        let path = recon.sift_path_for_image(image);
        std::fs::create_dir_all(path.parent().expect("a feature directory")).unwrap();
        sfmtool_sift_format::write_sift(&path, &data, 3).unwrap();
    }
    recon
}

/// Every camera index of the fixture.
fn all_cameras(recon: &SfmrReconstruction) -> Vec<usize> {
    (0..recon.image_table.cameras.len()).collect()
}

/// The largest feature index any observation of `camera`'s images uses.
fn largest_observed_feature(recon: &SfmrReconstruction, camera: usize) -> Option<usize> {
    let features = recon.feature_indexes().expect("a sift_files value");
    recon
        .point_set
        .tracks
        .iter()
        .enumerate()
        .filter(|(_, o)| {
            recon.image_table.images[o.image_index as usize].camera_index as usize == camera
        })
        .map(|(row, _)| features[row] as usize)
        .max()
}

#[test]
fn the_detected_keypoint_reaches_past_the_observed_one() {
    let dir = tempfile::tempdir().unwrap();
    let recon = fixture(&dir);
    let reach = outermost_keypoints(&recon, &all_cameras(&recon), true);
    assert_eq!(reach.len(), recon.image_table.cameras.len());
    for entry in &reach {
        let camera = &recon.image_table.cameras[entry.camera];
        let images = recon
            .image_table
            .images
            .iter()
            .filter(|i| i.camera_index as usize == entry.camera)
            .count();
        assert_eq!(entry.images, images);
        assert_eq!(entry.detected_images, images);
        let Some(f) = largest_observed_feature(&recon, entry.camera) else {
            assert_eq!(entry.observed, None);
            continue;
        };
        let observed = entry.observed.expect("observations");
        assert!(
            (observed.radius_px - (10.0 + f as f64)).abs() < 1e-3,
            "{observed:?}"
        );
        let [x, y] = feature_xy(camera, f);
        assert_eq!(observed.xy, [f64::from(x), f64::from(y)]);
        assert!(
            (observed.theta_deg - off_axis_angle_deg(camera, observed.xy[0], observed.xy[1])).abs()
                < 1e-12
        );
        let detected = entry.detected.expect("the .sift files are readable");
        assert!(
            (detected.radius_px - UNOBSERVED_RADIUS).abs() < 1e-3,
            "{detected:?}"
        );
        assert!(detected.theta_deg > observed.theta_deg);
    }
}

#[test]
fn without_readable_sift_files_nothing_is_detected() {
    let dir = tempfile::tempdir().unwrap();
    let recon = fixture(&dir);
    // Not asked to read: a sift_files value has no observed pixel either.
    for entry in outermost_keypoints(&recon, &all_cameras(&recon), false) {
        assert_eq!(
            (entry.observed, entry.detected, entry.detected_images),
            (None, None, 0)
        );
    }
    // Asked, and the files are gone.
    let mut moved = recon.clone();
    moved.workspace_dir = dir.path().join("elsewhere");
    for entry in outermost_keypoints(&moved, &all_cameras(&moved), true) {
        assert_eq!(entry.detected, None);
        assert_eq!(entry.detected_images, 0);
    }
}

#[test]
fn inline_keypoints_are_observed_without_reading_a_file() {
    let dir = tempfile::tempdir().unwrap();
    let mut recon = fixture(&dir);
    let rows = recon.point_set.tracks.len();
    let mut keypoints_xy = Array2::<f32>::zeros((rows, 2));
    let camera = recon.image_table.cameras[0].clone();
    let (cx, cy) = camera.principal_point();
    for row in 0..rows {
        keypoints_xy[[row, 0]] = cx as f32;
        keypoints_xy[[row, 1]] = (cy + row as f64) as f32;
    }
    recon.point_set.observations = ObservationSource::EmbeddedPatches {
        keypoints_xy,
        image_file_hashes: vec![[0u8; 16]; recon.image_table.images.len()],
    };
    let reach = outermost_keypoints(&recon, &[0], false);
    let observed = reach[0].observed.expect("inline keypoints");
    let last = recon
        .point_set
        .tracks
        .iter()
        .rposition(|o| recon.image_table.images[o.image_index as usize].camera_index == 0)
        .expect("camera 0 is observed");
    assert!(
        (observed.radius_px - last as f64).abs() < 1e-3,
        "{observed:?}"
    );
    assert_eq!(reach[0].detected, None);
}

#[test]
fn a_camera_past_the_table_is_ignored() {
    let dir = tempfile::tempdir().unwrap();
    let recon = fixture(&dir);
    let count = recon.image_table.cameras.len();
    let reach = outermost_keypoints(&recon, &[count + 3, 0, 0], true);
    assert_eq!(reach.len(), 1);
    assert_eq!(reach[0].camera, 0);
}
