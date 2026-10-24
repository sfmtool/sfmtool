// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::Mutex;

use super::*;
use crate::progress::Event;

const SEOUL_BULL: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../test-data/images/seoul_bull_sculpture"
);

/// A workspace in a temporary directory: seoul_bull's `.sfm-workspace.json`
/// (the `sfmtool` tool with its default options) at the root, and copies of
/// the first `count` seoul_bull images under `images/`.
fn workspace_with_images(count: usize) -> (tempfile::TempDir, Vec<PathBuf>) {
    let dir = tempfile::tempdir().unwrap();
    let source = Path::new(SEOUL_BULL);
    std::fs::copy(
        source.join(WORKSPACE_CONFIG_FILE),
        dir.path().join(WORKSPACE_CONFIG_FILE),
    )
    .unwrap();
    let images_dir = dir.path().join("images");
    std::fs::create_dir(&images_dir).unwrap();
    let images = (1..=count)
        .map(|i| {
            let name = format!("seoul_bull_sculpture_{i:02}.jpg");
            let to = images_dir.join(&name);
            std::fs::copy(source.join(&name), &to).unwrap();
            to
        })
        .collect();
    (dir, images)
}

#[test]
fn options_come_from_the_workspace_file() {
    let (dir, _) = workspace_with_images(0);
    let options = BuildSiftFilesOptions::from_workspace(dir.path()).unwrap();
    assert_eq!(
        options.feature_prefix_dir,
        PathBuf::from("features/sift-sfmtool-3dcd2b2f8c892d12c3ffe28cedce19c9")
    );
    assert_eq!(options.feature_options["max_num_features"], 8192);
    assert_eq!(options.concurrent_images, None);
    assert_eq!(
        options.sift_path_for_image(Path::new("a/b/x.jpg")),
        Path::new("a/b")
            .join(&options.feature_prefix_dir)
            .join("x.jpg.sift")
    );
}

/// The defaults are what `sfm ws init` wrote into seoul_bull's workspace file,
/// down to the hash in the prefix directory.
#[test]
fn the_sfmtool_defaults_are_those_ws_init_writes() {
    let (dir, _) = workspace_with_images(0);
    let written = BuildSiftFilesOptions::from_workspace(dir.path()).unwrap();
    let defaults = BuildSiftFilesOptions::sfmtool_defaults();
    assert_eq!(defaults.feature_options, written.feature_options);
    assert_eq!(defaults.feature_prefix_dir, written.feature_prefix_dir);
    assert_eq!(defaults.concurrent_images, None);
    let config: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(dir.path().join(WORKSPACE_CONFIG_FILE)).unwrap(),
    )
    .unwrap();
    assert_eq!(config["feature_tool"], SFMTOOL_FEATURE_TOOL);
    assert_eq!(config["feature_type"], SFMTOOL_FEATURE_TYPE);
}

#[test]
fn a_workspace_of_another_feature_tool_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join(WORKSPACE_CONFIG_FILE),
        r#"{"feature_tool": "colmap", "feature_options": {}, "feature_prefix_dir": "features/x"}"#,
    )
    .unwrap();
    assert_eq!(
        BuildSiftFilesOptions::from_workspace(dir.path()).unwrap_err(),
        BuildSiftFilesError::UnsupportedFeatureTool {
            feature_tool: "colmap".into()
        }
    );

    let missing = tempfile::tempdir().unwrap();
    let err = BuildSiftFilesOptions::from_workspace(missing.path()).unwrap_err();
    assert!(
        matches!(&err, BuildSiftFilesError::Workspace { problem, .. } if problem.contains("not initialized")),
        "{err}"
    );
}

#[test]
fn a_built_file_holds_what_the_extractor_finds() {
    let (dir, images) = workspace_with_images(2);
    let options = BuildSiftFilesOptions::from_workspace(dir.path()).unwrap();
    let report = build_sift_files(&images, &options, &Progress::none()).unwrap();
    assert_eq!(report.extracted, 2);
    assert_eq!(report.skipped, 0);
    assert_eq!(
        report.sift_paths,
        images
            .iter()
            .map(|image| options.sift_path_for_image(image))
            .collect::<Vec<_>>()
    );

    let image = &images[0];
    let data = sfmtool_sift_format::read_sift(&report.sift_paths[0]).unwrap();
    assert_eq!(data.feature_tool_metadata.feature_tool, "sfmtool");
    assert_eq!(data.feature_tool_metadata.feature_type, "sift");
    assert_eq!(
        data.feature_tool_metadata.feature_options,
        options.feature_options
    );
    let bytes = std::fs::read(image).unwrap();
    assert_eq!(data.metadata.image_name, "seoul_bull_sculpture_01.jpg");
    assert_eq!(data.metadata.image_file_size, bytes.len() as u64);
    assert_eq!(
        data.metadata.image_file_xxh128,
        format!("{:032x}", xxhash_rust::xxh3::xxh3_128(&bytes))
    );
    assert_eq!(
        (data.metadata.image_width, data.metadata.image_height),
        (270, 480)
    );

    // The features are the extractor's on the same decode, in its order.
    let rgb = ImageU8::read_rgb(image).unwrap();
    let params = SiftParams::from_feature_options(&options.feature_options).unwrap();
    let gray = gray_from_rgb(270, 480, rgb.data(), &params.image_to_gray);
    let expected = extract_sift(&gray, &params);
    let n = expected.keypoints.len();
    assert!(n > 100, "{n} features");
    assert_eq!(data.metadata.feature_count as usize, n);
    for (i, kp) in expected.keypoints.iter().enumerate() {
        assert_eq!(data.positions_xy.row(i).to_vec(), [kp.x, kp.y]);
        let shape = data.affine_shapes.index_axis(ndarray::Axis(0), i);
        assert_eq!(
            [
                [shape[[0, 0]], shape[[0, 1]]],
                [shape[[1, 0]], shape[[1, 1]]]
            ],
            kp.affine_shape
        );
    }
    let descriptors: Vec<u8> = expected.descriptors.rows().concat();
    assert_eq!(data.descriptors.as_slice().unwrap(), &descriptors[..]);
    assert_eq!(
        data.thumbnail_y_x_rgb.as_slice().unwrap(),
        &thumbnail_from_rgb(rgb.data(), 270, 480)[..]
    );
    let sizes: Vec<f32> = expected.keypoints.iter().map(|kp| kp.scale()).collect();
    assert!(sizes.windows(2).all(|w| w[0] >= w[1]), "sorted by size");
}

#[test]
fn an_up_to_date_file_is_skipped_and_a_newer_image_rebuilt() {
    let (dir, images) = workspace_with_images(2);
    let options = BuildSiftFilesOptions {
        concurrent_images: Some(2),
        ..BuildSiftFilesOptions::from_workspace(dir.path()).unwrap()
    };
    let first = build_sift_files(&images, &options, &Progress::none()).unwrap();
    assert_eq!((first.extracted, first.skipped), (2, 0));

    let again = build_sift_files(&images, &options, &Progress::none()).unwrap();
    assert_eq!((again.extracted, again.skipped), (0, 2));

    // An image modified after its .sift was written is extracted again.
    let sift_modified = std::fs::metadata(&first.sift_paths[1])
        .unwrap()
        .modified()
        .unwrap();
    std::fs::File::options()
        .write(true)
        .open(&images[1])
        .unwrap()
        .set_modified(sift_modified + std::time::Duration::from_secs(10))
        .unwrap();
    let rebuilt = build_sift_files(&images, &options, &Progress::none()).unwrap();
    assert_eq!((rebuilt.extracted, rebuilt.skipped), (1, 1));
}

#[test]
fn progress_counts_each_image_and_says_how_many_were_skipped() {
    let (dir, images) = workspace_with_images(3);
    let options = BuildSiftFilesOptions::from_workspace(dir.path()).unwrap();
    build_sift_files(&images[..1], &options, &Progress::none()).unwrap();

    let events = Mutex::new(Vec::new());
    let sink = |event: Event<'_>| {
        let line = match event {
            Event::Enter { phase, .. } => format!("enter {phase}"),
            Event::Message { text, .. } => format!("message {text}"),
            Event::Count { done, total, unit } => format!("count {done}/{total:?} {unit}"),
            _ => return,
        };
        events.lock().unwrap().push(line);
    };
    let report = build_sift_files(&images, &options, &Progress::to(&sink)).unwrap();
    assert_eq!((report.extracted, report.skipped), (2, 1));
    let events = events.into_inner().unwrap();
    assert_eq!(events[0], "enter build_sift_files");
    assert!(events.contains(&"message 1 of 3 images already have an up-to-date .sift file".into()));
    assert!(events.contains(&"count 0/Some(2) image".into()));
    assert_eq!(events.last().unwrap(), "count 2/Some(2) image");
}

#[test]
fn a_cancelled_build_writes_nothing_more() {
    let (dir, images) = workspace_with_images(2);
    let options = BuildSiftFilesOptions::from_workspace(dir.path()).unwrap();
    let cancel = AtomicBool::new(true);
    let progress = Progress::none().cancelled_by(&cancel);
    assert_eq!(
        build_sift_files(&images, &options, &progress).unwrap_err(),
        BuildSiftFilesError::Cancelled
    );
    for image in &images {
        assert!(!options.sift_path_for_image(image).exists());
    }
}

#[test]
fn the_first_failing_image_in_input_order_is_the_error() {
    let (dir, mut images) = workspace_with_images(2);
    let images_dir = dir.path().join("images");
    for name in ["broken_a.jpg", "broken_b.jpg"] {
        std::fs::write(images_dir.join(name), b"not an image").unwrap();
    }
    images.insert(1, images_dir.join("broken_a.jpg"));
    images.push(images_dir.join("broken_b.jpg"));
    let options = BuildSiftFilesOptions {
        concurrent_images: Some(4),
        ..BuildSiftFilesOptions::from_workspace(dir.path()).unwrap()
    };
    match build_sift_files(&images, &options, &Progress::none()).unwrap_err() {
        BuildSiftFilesError::Image { path, .. } => assert_eq!(path, images[1]),
        other => panic!("expected an image error, got {other}"),
    }
}

#[test]
fn invalid_feature_options_fail_before_anything_is_written() {
    let (dir, images) = workspace_with_images(1);
    let mut options = BuildSiftFilesOptions::from_workspace(dir.path()).unwrap();
    options.feature_options["sigma"] = serde_json::json!("wide");
    let err = build_sift_files(&images, &options, &Progress::none()).unwrap_err();
    assert!(
        matches!(err, BuildSiftFilesError::FeatureOptions(_)),
        "{err}"
    );
    assert!(!dir.path().join("images/features").exists());
}

#[test]
fn concurrency_is_the_core_count_capped_by_memory() {
    let seoul = 270 * 480;
    // Plenty of memory: one image per core.
    assert_eq!(concurrent_images_for(64, Some(seoul), 64 << 30), 64);
    // A 2 GiB budget holds 8 images of 1.4 megapixels at 192 bytes a pixel.
    let pixels = 1_400_000;
    let expected = ((2u64 << 30) / (pixels * BYTES_PER_SOURCE_PIXEL)) as usize;
    assert_eq!(concurrent_images_for(64, Some(pixels), 2 << 30), expected);
    // Never fewer than one, and at most four when the size is unknown.
    assert_eq!(concurrent_images_for(8, Some(u64::MAX / 1024), 1), 1);
    assert_eq!(concurrent_images_for(16, None, 1 << 30), 4);
    assert_eq!(concurrent_images_for(2, None, 1 << 30), 2);
}

#[test]
fn feature_options_override_the_defaults() {
    let defaults = SiftParams::from_feature_options(&serde_json::Value::Null).unwrap();
    assert_eq!(defaults.max_num_features, Some(8192));
    let params = SiftParams::from_feature_options(&serde_json::json!({
        "octave_layers": 4,
        "sigma": 2,
        "double_image": false,
        "max_num_features": null,
    }))
    .unwrap();
    assert_eq!(params.octave_layers, 4);
    assert_eq!(params.sigma, 2.0);
    assert!(!params.double_image);
    assert_eq!(params.max_num_features, None);

    let refused = |options: serde_json::Value| {
        SiftParams::from_feature_options(&options)
            .unwrap_err()
            .to_string()
    };
    assert_eq!(
        refused(serde_json::json!({"octave_layers": 1.5})),
        "feature option 'octave_layers' has the wrong type"
    );
    assert_eq!(
        refused(serde_json::json!({"colour": 1})),
        "unknown feature option 'colour'"
    );
    assert_eq!(
        refused(serde_json::json!({"orientation_bins": 0})),
        "feature option 'orientation_bins' must be >= 1"
    );
    assert!(refused(serde_json::json!({"descriptor_width": 3})).contains("fixed at 4 and 8"));
    assert!(refused(serde_json::json!({"gray_formula": "R +"})).contains("gray formula"));
    assert!(refused(serde_json::json!([1, 2])).contains("JSON object"));
}
