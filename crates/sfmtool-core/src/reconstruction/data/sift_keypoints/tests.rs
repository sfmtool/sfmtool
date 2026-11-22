// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Filling the inline keypoint column from `.sift` files: the load doing it
//! for a version 16 file that lacks the column, and the save for a value that
//! lacks it.

use ndarray::{Array2, Array3};
use tempfile::TempDir;

use super::*;

/// The demo reconstruction without its inline keypoints, with a `.sift` file
/// per image under `dir`, and its `sift_content_hashes` set to the hashes of
/// those files.
///
/// Feature `f` of image `i` sits at `(100 + f, 200 + i)`, so a filled row can
/// be checked against the observation's image and feature index.
fn fixture(dir: &TempDir) -> SfmrReconstruction {
    let mut recon = SfmrReconstruction::demo_without_keypoints(12);
    recon.workspace_dir = dir.path().to_path_buf();
    recon.metadata.workspace.contents.feature_prefix_dir = "features".into();

    let mut hashes = Vec::new();
    for image in 0..recon.image_set.images.len() {
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
                feature_options: serde_json::json!({"image_to_gray": {"formula": "G"}}),
            },
            metadata: sfmtool_sift_format::SiftMetadata {
                version: sfmtool_sift_format::SIFT_FORMAT_VERSION,
                image_name: recon.image_set.images[image].name.clone(),
                image_file_xxh128: "0".repeat(32),
                image_file_size: 1,
                image_width: recon.image_set.cameras[0].width,
                image_height: recon.image_set.cameras[0].height,
                feature_count: count as u32,
                sharpness: 0.0,
                gradient_structure_tensor_xx_yy_xy: [0.0; 3],
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
        hashes.push(parse_hash_bytes(&stored.content_xxh128).unwrap());
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
    recon.point_set.keypoints_xy = Some(keypoints);
    recon.point_set.observations = ObservationSource::EmbeddedPatches {
        image_file_hashes: vec![[0; 16]; recon.image_set.images.len()],
    };
    assert_eq!(
        recon.fill_keypoints_from_sift(&Progress::none()),
        SiftKeypointFill::NotSiftFiles
    );
}

/// The last count a run reported: how many done, out of how many.
type LastCount = Option<(u64, Option<u64>)>;

/// The phases `run` closed, in order, and the last count reported, given the
/// `Progress` to report through.
fn recording<T>(run: impl FnOnce(&Progress<'_>) -> T) -> (T, Vec<&'static str>, LastCount) {
    let left = std::sync::Mutex::new(Vec::new());
    let count = std::sync::Mutex::new(None);
    let sink = |event: crate::progress::Event<'_>| match event {
        crate::progress::Event::Leave { phase, .. } => left.lock().unwrap().push(phase),
        crate::progress::Event::Count { done, total, .. } => {
            *count.lock().unwrap() = Some((done, total));
        }
        _ => {}
    };
    let out = run(&Progress::to(&sink));
    let phases = left.lock().unwrap().clone();
    let last = *count.lock().unwrap();
    (out, phases, last)
}

/// A save of a value without the column writes it, read from the `.sift`
/// files in a `keypoints` stage that counts the images, and leaves the value
/// as it was.
#[test]
fn save_writes_the_column_from_the_sift_files() {
    let dir = tempfile::tempdir().unwrap();
    let recon = fixture(&dir);
    let path = dir.path().join("saved.sfmr");
    let (saved, phases, count) = recording(|progress| recon.save(&path, progress));
    saved.unwrap();
    assert_eq!(phases, ["keypoints"]);
    let images = recon.image_set.images.len() as u64;
    assert_eq!(count, Some((images, Some(images))));
    assert!(recon.keypoints_xy().is_none());

    let mut written = recon.clone();
    written.point_set.keypoints_xy = sfmtool_sfmr_format::read_sfmr(&path).unwrap().keypoints_xy;
    assert_rows_match_features(&written);

    // A value that carries the column reads no `.sift` file on a save.
    let (saved, phases, count) = recording(|progress| written.save(&path, progress));
    saved.unwrap();
    assert!(phases.is_empty(), "{phases:?}");
    assert_eq!(count, None);
}

/// A version 16 `sift_files` file written without the column loads with it,
/// in a `keypoints` stage, and a save of the loaded value writes it.
#[test]
fn load_fills_a_version_16_file_without_the_column() {
    let dir = tempfile::tempdir().unwrap();
    let recon = fixture(&dir);
    // The marker is how the load finds the workspace the `.sift` files are in.
    std::fs::write(dir.path().join(".sfm-workspace.json"), "{}").unwrap();
    let path = dir.path().join("no-keypoints.sfmr");
    recon.save(&path, &Progress::none()).unwrap();
    super::super::demo::rewrite_as_version_16_without_keypoints(&path);
    let data = sfmtool_sfmr_format::read_sfmr(&path).unwrap();
    assert_eq!(data.metadata.version, 16);
    assert!(data.keypoints_xy.is_none());

    let (loaded, phases, _) = recording(|progress| SfmrReconstruction::load(&path, progress));
    let loaded = loaded.unwrap();
    assert_eq!(phases, ["read", "derive", "keypoints"]);
    assert_rows_match_features(&loaded);
    // The value still names the file it was read from.
    assert_eq!(
        loaded.content_hash.content_xxh128,
        data.content_hash.content_xxh128
    );

    let resaved = dir.path().join("with-keypoints.sfmr");
    loaded.save(&resaved, &Progress::none()).unwrap();
    let written = sfmtool_sfmr_format::read_sfmr(&resaved).unwrap();
    assert_eq!(
        written.metadata.version,
        sfmtool_sfmr_format::SFMR_FORMAT_VERSION
    );
    assert_eq!(written.keypoints_xy.as_ref(), loaded.keypoints_xy());
}

/// When the `.sift` files cannot supply the column, a save writes nothing and
/// says which file and what to do.
#[test]
fn save_refuses_a_value_whose_sift_files_cannot_supply_the_column() {
    let dir = tempfile::tempdir().unwrap();
    let recon = fixture(&dir);
    let missing = recon.sift_path_for_image(2);
    std::fs::remove_file(&missing).unwrap();
    let path = dir.path().join("refused.sfmr");

    match recon.save(&path, &Progress::none()) {
        Err(sfmtool_sfmr_format::SfmrError::InvalidFormat(message)) => {
            assert!(message.contains("tracks/keypoints_xy"), "{message}");
            assert!(
                message.contains(&format!("{}: no such file", missing.display())),
                "{message}"
            );
            assert!(message.contains("Restore the .sift files"), "{message}");
        }
        other => panic!("expected InvalidFormat, got {other:?}"),
    }
    assert!(!path.exists());
}
