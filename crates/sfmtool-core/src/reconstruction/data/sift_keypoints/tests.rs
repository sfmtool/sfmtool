// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Filling the inline keypoint column from `.sift` files, and the loader doing
//! it for a file that lacks the column.

use ndarray::{Array2, Array3};
use tempfile::TempDir;

use super::*;

/// The demo reconstruction with a `.sift` file per image under `dir`, and its
/// `sift_content_hashes` set to the hashes of those files.
///
/// Feature `f` of image `i` sits at `(100 + f, 200 + i)`, so a filled row can
/// be checked against the observation's image and feature index.
fn fixture(dir: &TempDir) -> SfmrReconstruction {
    let mut recon = SfmrReconstruction::demo(12);
    recon.workspace_dir = dir.path().to_path_buf();
    recon.metadata.workspace.contents.feature_prefix_dir = "features".into();

    let mut hashes = Vec::new();
    for image in 0..recon.image_table.images.len() {
        let count = recon.point_set.max_track_feature_index[image] as usize + 1;
        let mut positions = Array2::<f32>::zeros((count, 2));
        for feature in 0..count {
            positions[[feature, 0]] = 100.0 + feature as f32;
            positions[[feature, 1]] = 200.0 + image as f32;
        }
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
                image_width: recon.image_table.cameras[0].width,
                image_height: recon.image_table.cameras[0].height,
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
        let (_, _, stored) = sfmtool_sift_format::read_sift_metadata(&path).unwrap();
        hashes.push(decode_xxh128_hex(&stored.content_xxh128).unwrap());
    }
    if let ObservationSource::SiftFiles {
        sift_content_hashes,
        ..
    } = &mut recon.point_set.observations
    {
        *sift_content_hashes = hashes;
    }
    recon
}

/// Each observation's row is the `.sift` position its feature index names.
fn assert_rows_match_features(recon: &SfmrReconstruction) {
    let keypoints = recon.keypoints_xy().expect("the column was filled");
    let features = recon.point_set.feature_indexes().unwrap();
    assert_eq!(keypoints.nrows(), recon.point_set.tracks.len());
    for (row, obs) in recon.point_set.tracks.iter().enumerate() {
        assert_eq!(keypoints[[row, 0]], 100.0 + features[row] as f32);
        assert_eq!(keypoints[[row, 1]], 200.0 + obs.image_index as f32);
    }
}

#[test]
fn fills_every_observation_from_its_feature() {
    let dir = tempfile::tempdir().unwrap();
    let mut recon = fixture(&dir);
    assert!(recon.keypoints_xy().is_none());

    assert_eq!(
        recon.fill_keypoints_from_sift(&Progress::none()),
        SiftKeypointFill::Filled
    );
    assert_rows_match_features(&recon);
    assert_eq!(
        recon.fill_keypoints_from_sift(&Progress::none()),
        SiftKeypointFill::AlreadyPresent
    );
}

#[test]
fn a_missing_sift_leaves_the_column_absent() {
    let dir = tempfile::tempdir().unwrap();
    let mut recon = fixture(&dir);
    std::fs::remove_file(recon.sift_path_for_image(3)).unwrap();

    match recon.fill_keypoints_from_sift(&Progress::none()) {
        SiftKeypointFill::Unavailable { image, reason } => {
            assert_eq!(image, 3);
            assert!(reason.ends_with("no such file"), "{reason}");
        }
        other => panic!("expected Unavailable, got {other:?}"),
    }
    assert!(recon.keypoints_xy().is_none());
}

#[test]
fn a_sift_from_another_extraction_leaves_the_column_absent() {
    let dir = tempfile::tempdir().unwrap();
    let mut recon = fixture(&dir);
    if let ObservationSource::SiftFiles {
        sift_content_hashes,
        ..
    } = &mut recon.point_set.observations
    {
        sift_content_hashes[0] = [7; 16];
    }

    match recon.fill_keypoints_from_sift(&Progress::none()) {
        SiftKeypointFill::Unavailable { image, reason } => {
            assert_eq!(image, 0);
            assert!(reason.contains("content hash differs"), "{reason}");
        }
        other => panic!("expected Unavailable, got {other:?}"),
    }
    assert!(recon.keypoints_xy().is_none());
}

#[test]
fn an_embedded_patches_value_is_not_touched() {
    let dir = tempfile::tempdir().unwrap();
    let mut recon = fixture(&dir);
    recon.fill_keypoints_from_sift(&Progress::none());
    let keypoints = recon.keypoints_xy().unwrap().clone();
    recon.point_set.observations = ObservationSource::EmbeddedPatches {
        keypoints_xy: keypoints,
        image_file_hashes: vec![[0; 16]; recon.image_table.images.len()],
    };
    assert_eq!(
        recon.fill_keypoints_from_sift(&Progress::none()),
        SiftKeypointFill::NotSiftFiles
    );
}

/// A `sift_files` file saved without the column loads with it, and a save of
/// the loaded value writes it.
#[test]
fn load_fills_a_file_without_the_column() {
    let dir = tempfile::tempdir().unwrap();
    let recon = fixture(&dir);
    // The marker is how the load finds the workspace the `.sift` files are in.
    std::fs::write(dir.path().join(".sfm-workspace.json"), "{}").unwrap();
    let path = dir.path().join("no-keypoints.sfmr");
    recon.save(&path).unwrap();
    assert!(!sfmtool_sfmr_format::read_sfmr(&path)
        .unwrap()
        .keypoints_xy
        .is_some());

    let loaded = SfmrReconstruction::load(&path, &Progress::none()).unwrap();
    assert_rows_match_features(&loaded);
    // The value still names the file it was read from.
    assert_eq!(
        loaded.content_hash.content_xxh128,
        sfmtool_sfmr_format::read_sfmr(&path)
            .unwrap()
            .content_hash
            .content_xxh128
    );

    let resaved = dir.path().join("with-keypoints.sfmr");
    loaded.save(&resaved).unwrap();
    assert!(sfmtool_sfmr_format::read_sfmr(&resaved)
        .unwrap()
        .keypoints_xy
        .is_some());
}
