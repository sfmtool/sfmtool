// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! What the baseline conversion reports, and what it does when it is asked to
//! stop.
//!
//! The fixture is the demo reconstruction with a real `.sift` file per image in
//! a temporary workspace: the default `FeatureSize` sizing reads each file's
//! affine shapes, and the keypoint copy reads each file's positions, so a
//! conversion with no files behind it would fail before it reported anything.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

use ndarray::{Array2, Array3};
use tempfile::TempDir;

use super::*;
use crate::progress::{Event, Progress};

/// How many points the fixture carries. Small: what is under test is the
/// reporting and the stopping, not the arithmetic of the frames.
const POINTS: usize = 12;

/// The demo reconstruction with a `.sift` file per image under `dir`.
///
/// Each feature's affine column-0 norm is a positive `σ`, so every observation
/// resolves a readable keypoint scale and `FeatureSize` can size every patch;
/// each position is a distinct pixel, so the keypoints the conversion copies
/// can be told apart from zeros.
fn fixture(dir: &TempDir) -> SfmrReconstruction {
    let mut recon = SfmrReconstruction::demo(POINTS);
    recon.workspace_dir = dir.path().to_path_buf();
    recon.metadata.workspace.contents.feature_prefix_dir = "features".into();

    for image in 0..recon.image_table.images.len() {
        let count = recon.point_set.max_track_feature_index[image] as usize + 1;
        let mut positions = Array2::<f32>::zeros((count, 2));
        let mut affine = Array3::<f32>::zeros((count, 2, 2));
        for feature in 0..count {
            positions[[feature, 0]] = 100.0 + feature as f32;
            positions[[feature, 1]] = 200.0 + image as f32;
            affine[[feature, 0, 0]] = 2.0;
            affine[[feature, 1, 1]] = 2.0;
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
            affine_shapes: affine,
            descriptors: Array2::<u8>::zeros((count, 128)),
            thumbnail_y_x_rgb: Array3::<u8>::zeros((128, 128, 3)),
        };
        let path = recon.sift_path_for_image(image);
        std::fs::create_dir_all(path.parent().expect("a feature directory")).unwrap();
        sfmtool_sift_format::write_sift(&path, &data, 3).unwrap();
    }
    recon
}

/// The conversion names its three stages, in order, and says how much each one
/// covered.
#[test]
fn the_conversion_names_its_stages() {
    let dir = tempfile::tempdir().unwrap();
    let recon = fixture(&dir);

    let left = Mutex::new(Vec::new());
    let images = Mutex::new(Vec::new());
    let sink = |event: Event<'_>| match event {
        Event::Leave { phase, note, .. } => {
            left.lock().unwrap().push((phase, note.map(str::to_string)))
        }
        Event::Count { done, total, unit } => images.lock().unwrap().push((done, total, unit)),
        _ => {}
    };
    let embedded = recon
        .to_embedded_patches(
            PatchNormal::MeanViewing,
            PatchExtent::default(),
            &Progress::to(&sink),
        )
        .expect("the fixture has a .sift file per image");

    let left = left.lock().unwrap();
    let names: Vec<&str> = left.iter().map(|(phase, _)| *phase).collect();
    assert_eq!(names, ["patch frames", "read keypoints", "assemble"]);
    assert_eq!(left[0].1.as_deref(), Some("12 points"));
    assert_eq!(left[1].1.as_deref(), Some("8 images"));
    assert_eq!(left[2].1.as_deref(), Some("24 observations"));
    // One count per image, in order, against the total the caller knows.
    assert_eq!(
        *images.lock().unwrap(),
        (1..=8).map(|i| (i, Some(8), "image")).collect::<Vec<_>>()
    );

    assert_eq!(embedded.metadata.feature_source, "embedded_patches");
    assert_eq!(embedded.point_count(), recon.point_count());
    assert_eq!(embedded.image_count(), recon.image_count());
    // The keypoints are the detections the fixture wrote, not zeros.
    let keypoints = embedded
        .point_set
        .keypoints_xy()
        .expect("an embedded_patches value carries its keypoints");
    assert_eq!(keypoints[[0, 0]], 100.0);
}

/// Cancelled after the first image, it stops with [`ReconstructionError::Cancelled`]
/// and leaves the reconstruction it was reading alone.
#[test]
fn a_cancel_between_the_images_stops_the_read() {
    let dir = tempfile::tempdir().unwrap();
    let recon = fixture(&dir);

    let flag = AtomicBool::new(false);
    let read = Mutex::new(0u64);
    // Set from the sink, so the flag is raised by the very count that says one
    // image is behind it: the next poll is the loop's own, one iteration later.
    let sink = |event: Event<'_>| {
        if let Event::Count { done, unit, .. } = event {
            if unit == "image" {
                *read.lock().unwrap() = done;
                flag.store(true, Ordering::Relaxed);
            }
        }
    };
    let progress = Progress::to(&sink).cancelled_by(&flag);

    let error = match recon.to_embedded_patches(
        PatchNormal::MeanViewing,
        PatchExtent::default(),
        &progress,
    ) {
        Err(error) => error,
        Ok(_) => panic!("the cancel flag was raised while it read, and it produced a value"),
    };
    assert!(
        matches!(error, ReconstructionError::Cancelled),
        "{error:?} is not a cancellation"
    );
    assert_eq!(
        error.to_string(),
        "the operation was asked to stop before it had an answer, so nothing was produced"
    );
    // It stopped where it was told to rather than running the read out.
    assert_eq!(*read.lock().unwrap(), 1);
    // The input is untouched: the conversion is a function of it and builds a
    // new value, so a stopped one leaves nothing half-written behind.
    assert_eq!(recon.metadata.feature_source, "sift_files");
    assert!(recon.point_set.feature_indexes().is_some());
    assert!(recon.point_set.patch_u_halfvec_xyz.is_none());
}

/// A reconstruction with no `.sift` files behind it is refused by the sizing
/// step, in the frame builder's own words.
#[test]
fn a_missing_sift_file_is_refused_while_the_frames_are_built() {
    let dir = tempfile::tempdir().unwrap();
    let mut recon = SfmrReconstruction::demo(POINTS);
    recon.workspace_dir = dir.path().to_path_buf();
    recon.metadata.workspace.contents.feature_prefix_dir = "features".into();

    let error = match recon.to_embedded_patches(
        PatchNormal::MeanViewing,
        PatchExtent::default(),
        &Progress::none(),
    ) {
        Err(error) => error,
        Ok(_) => panic!("there is no .sift file to size a patch from"),
    };
    assert!(
        error.to_string().contains("building patch frames failed"),
        "{error}"
    );
}
