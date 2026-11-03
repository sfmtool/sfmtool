// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::Mutex;

use super::*;
use crate::features::sift::ImageToGray;
use crate::progress::Event;

const SEOUL_BULL: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../test-data/images/seoul_bull_sculpture"
);

/// The `feature_prefix_dir` `sfm ws init --feature-tool sfmtool` writes for
/// the default options. `tests/test_workspace.py` holds the Python side to the
/// same string, so the two default folder names stay equal.
const WS_INIT_SFMTOOL_PREFIX: &str = "features/sift-sfmtool-9017ec0d2c5dfc401da610578062b53d";

/// Copies of the first `count` seoul_bull images under `images/` of a
/// temporary directory.
fn workspace_with_images(count: usize) -> (tempfile::TempDir, Vec<PathBuf>) {
    let dir = tempfile::tempdir().unwrap();
    let source = Path::new(SEOUL_BULL);
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

/// The defaults are what `sfm ws init` writes for the `sfmtool` tool, down to
/// the hash in the prefix directory.
#[test]
fn the_sfmtool_defaults_are_those_ws_init_writes() {
    let defaults = BuildSiftFilesOptions::sfmtool_defaults();
    assert_eq!(
        defaults.feature_prefix_dir,
        PathBuf::from(WS_INIT_SFMTOOL_PREFIX)
    );
    assert_eq!(
        defaults.feature_options["image_to_gray"],
        serde_json::json!({"formula": "0.2126*R + 0.7152*G + 0.0722*B"})
    );
    assert!(defaults.feature_options.get("gray_formula").is_none());
    assert_eq!(defaults.feature_options["max_num_features"], 8192);
    assert_eq!(defaults.concurrent_images, None);
    assert_eq!(
        defaults.sift_path_for_image(Path::new("a/b/x.jpg")),
        Path::new("a/b")
            .join(&defaults.feature_prefix_dir)
            .join("x.jpg.sift")
    );
    // The options are a valid extraction.
    SiftParams::from_feature_options(&defaults.feature_options).unwrap();
}

#[test]
fn a_built_file_holds_what_the_extractor_finds() {
    let (_dir, images) = workspace_with_images(2);
    let options = BuildSiftFilesOptions::sfmtool_defaults();
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

    // A version 2 file, whose readings are those of the same decode through the
    // options' conversion.
    let rgb = ImageU8::read_rgb(image).unwrap();
    let params = SiftParams::from_feature_options(&options.feature_options).unwrap();
    assert_eq!(
        data.metadata.version,
        sfmtool_sift_format::SIFT_FORMAT_VERSION
    );
    let readings = image_readings_from_rgb(270, 480, rgb.data(), &params.image_to_gray).unwrap();
    assert_eq!(data.metadata.sharpness, Some(readings.sharpness));
    assert_eq!(
        data.metadata.gradient_structure_tensor_xx_yy_xy,
        Some(readings.gradient_structure_tensor_xx_yy_xy)
    );
    assert!(readings.sharpness > 0.0);

    // The features are the extractor's on the same decode, in its order.
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
    let (_dir, images) = workspace_with_images(2);
    let options = BuildSiftFilesOptions {
        concurrent_images: Some(2),
        ..BuildSiftFilesOptions::sfmtool_defaults()
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
    let (_dir, images) = workspace_with_images(3);
    let options = BuildSiftFilesOptions::sfmtool_defaults();
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
    let (_dir, images) = workspace_with_images(2);
    let options = BuildSiftFilesOptions::sfmtool_defaults();
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
        ..BuildSiftFilesOptions::sfmtool_defaults()
    };
    match build_sift_files(&images, &options, &Progress::none()).unwrap_err() {
        BuildSiftFilesError::Image { path, .. } => assert_eq!(path, images[1]),
        other => panic!("expected an image error, got {other}"),
    }
}

#[test]
fn invalid_feature_options_fail_before_anything_is_written() {
    let (dir, images) = workspace_with_images(1);
    let mut options = BuildSiftFilesOptions::sfmtool_defaults();
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
    assert!(
        refused(serde_json::json!({"image_to_gray": {"formula": "G", "quantize_bits": 0}}))
            .contains("feature option 'image_to_gray' is not valid")
    );
    assert!(refused(serde_json::json!({
        "image_to_gray": {"formula": "G"},
        "gray_formula": "G",
    }))
    .contains("cannot be given with 'image_to_gray'"));
    assert!(refused(serde_json::json!([1, 2])).contains("JSON object"));
}

#[test]
fn the_conversion_comes_from_image_to_gray_or_a_version_1_gray_formula() {
    let params = SiftParams::from_feature_options(&serde_json::json!({
        "image_to_gray": {"formula": "G", "quantize_bits": 8},
    }))
    .unwrap();
    assert_eq!(
        params.image_to_gray,
        ImageToGray::new("G", Some(8)).unwrap()
    );
    let v1 = SiftParams::from_feature_options(&serde_json::json!({"gray_formula": "R"})).unwrap();
    assert_eq!(
        v1.image_to_gray,
        ImageToGray::from_gray_formula("R").unwrap()
    );
}

#[test]
fn version_1_settings_are_refused_when_there_is_something_to_write() {
    let (dir, images) = workspace_with_images(1);
    let mut options = BuildSiftFilesOptions::sfmtool_defaults();
    let object = options
        .feature_options
        .as_object_mut()
        .unwrap()
        .remove("image_to_gray")
        .unwrap();
    options.feature_options["gray_formula"] = object["formula"].clone();
    let err = build_sift_files(&images, &options, &Progress::none()).unwrap_err();
    assert_eq!(err, BuildSiftFilesError::Version1Settings);
    assert!(err.to_string().contains("sfm ws upgrade"), "{err}");
    assert!(!dir.path().join("images/features").exists());

    // With every file up to date there is nothing to write, so nothing is
    // refused.
    let v2 = BuildSiftFilesOptions::sfmtool_defaults();
    build_sift_files(&images, &v2, &Progress::none()).unwrap();
    let up_to_date = BuildSiftFilesOptions {
        feature_prefix_dir: v2.feature_prefix_dir.clone(),
        ..options
    };
    let report = build_sift_files(&images, &up_to_date, &Progress::none()).unwrap();
    assert_eq!((report.extracted, report.skipped), (0, 1));
}

#[test]
fn a_folder_holding_version_1_files_is_refused() {
    let (dir, images) = workspace_with_images(2);
    let options = BuildSiftFilesOptions::sfmtool_defaults();
    // A version 1 file of the second image in the folder the first goes in.
    let v1_path = options.sift_path_for_image(&images[1]);
    std::fs::create_dir_all(v1_path.parent().unwrap()).unwrap();
    let mut data = sfmtool_sift_format::SiftData {
        feature_tool_metadata: FeatureToolMetadata {
            feature_tool: "sfmtool".into(),
            feature_type: "sift".into(),
            feature_options: serde_json::json!({"gray_formula": "G"}),
        },
        metadata: SiftMetadata {
            version: 1,
            image_name: "seoul_bull_sculpture_02.jpg".into(),
            image_file_xxh128: "0".repeat(32),
            image_file_size: 1,
            image_width: 270,
            image_height: 480,
            feature_count: 0,
            sharpness: None,
            gradient_structure_tensor_xx_yy_xy: None,
        },
        content_hash: SiftContentHash::default(),
        positions_xy: Array2::zeros((0, 2)),
        affine_shapes: Array3::zeros((0, 2, 2)),
        descriptors: Array2::zeros((0, 128)),
        thumbnail_y_x_rgb: Array3::zeros((THUMBNAIL_SIZE, THUMBNAIL_SIZE, 3)),
    };
    sfmtool_sift_format::write_sift(&v1_path, &data, 1).unwrap();
    let err = build_sift_files(&images[..1], &options, &Progress::none()).unwrap_err();
    assert_eq!(
        err,
        BuildSiftFilesError::Version1Files {
            folder: v1_path.parent().unwrap().to_path_buf()
        }
    );
    assert!(
        err.to_string().contains("holds version 1 .sift files"),
        "{err}"
    );
    assert!(!options.sift_path_for_image(&images[0]).exists());

    // A version 2 file beside it is no reason to refuse.
    data.metadata.version = 2;
    data.metadata.sharpness = Some(0.0);
    data.metadata.gradient_structure_tensor_xx_yy_xy = Some([0.0; 3]);
    data.feature_tool_metadata.feature_options =
        serde_json::json!({"image_to_gray": {"formula": "G"}});
    sfmtool_sift_format::write_sift(&v1_path, &data, 1).unwrap();
    build_sift_files(&images[..1], &options, &Progress::none()).unwrap();
    drop(dir);
}

#[test]
fn a_reading_that_is_not_finite_fails_naming_the_image_and_the_conversion() {
    let (_dir, images) = workspace_with_images(1);
    let mut options = BuildSiftFilesOptions::sfmtool_defaults();
    // R ** -1 is infinite where R is 0, and seoul_bull has black pixels.
    options.feature_options["image_to_gray"] =
        serde_json::json!({"formula": "R ** -1 + -1 ** 0.5"});
    let err = build_sift_files(&images, &options, &Progress::none()).unwrap_err();
    match &err {
        BuildSiftFilesError::Image { path, problem } => {
            assert_eq!(path, &images[0]);
            assert!(problem.contains("R ** -1 + -1 ** 0.5"), "{problem}");
        }
        other => panic!("expected an image error, got {other}"),
    }
    assert!(!options.sift_path_for_image(&images[0]).exists());
}
