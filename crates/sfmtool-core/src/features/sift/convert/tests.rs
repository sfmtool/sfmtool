// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;

use ndarray::{Array2, Array3};
use sfmtool_sift_format::{FeatureToolMetadata, SiftData, SiftMetadata, THUMBNAIL_SIZE};

use super::*;
use crate::features::sift::{image_readings_from_rgb, ImageToGray, DEFAULT_GRAY_FORMULA};
use crate::workspace::{tool_image_to_gray, FeatureOptionsError};

const SEOUL_BULL: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../test-data/images/seoul_bull_sculpture"
);

/// How many features the synthetic version 1 files hold.
const FEATURES: usize = 7;

/// A temporary directory with copies of the first two seoul_bull images in
/// `images/`.
fn images() -> (tempfile::TempDir, [PathBuf; 2]) {
    let dir = tempfile::tempdir().unwrap();
    let images_dir = dir.path().join("images");
    std::fs::create_dir(&images_dir).unwrap();
    let copy = |i: usize| {
        let name = format!("seoul_bull_sculpture_{i:02}.jpg");
        let to = images_dir.join(&name);
        std::fs::copy(Path::new(SEOUL_BULL).join(&name), &to).unwrap();
        to
    };
    (dir, [copy(1), copy(2)])
}

/// A version 1 `.sift` file for `image` in `images/features/v1/`, recording
/// `feature_tool` and `feature_options`, the image's hash, size and decoded
/// width and height, and features and a thumbnail that vary from entry to
/// entry. The features need not be the image's: the conversion copies them.
fn write_version_1(
    image: &Path,
    feature_tool: &str,
    feature_options: serde_json::Value,
) -> PathBuf {
    let decoded = ImageU8::read_rgb(image).unwrap();
    let (image_file_xxh128, image_file_size) = xxh128_of_file(image).unwrap();
    let name = image.file_name().unwrap().to_string_lossy().into_owned();
    let folder = image.parent().unwrap().join("features/v1");
    std::fs::create_dir_all(&folder).unwrap();
    let path = folder.join(format!("{name}.sift"));
    let data = SiftData {
        feature_tool_metadata: FeatureToolMetadata {
            feature_tool: feature_tool.to_owned(),
            feature_type: "sift".to_owned(),
            feature_options,
        },
        metadata: SiftMetadata {
            version: 1,
            image_name: name,
            image_file_xxh128,
            image_file_size,
            image_width: decoded.width(),
            image_height: decoded.height(),
            feature_count: FEATURES as u32,
            sharpness: None,
            gradient_structure_tensor_xx_yy_xy: None,
        },
        content_hash: SiftContentHash::default(),
        positions_xy: Array2::from_shape_fn((FEATURES, 2), |(i, j)| {
            10.5 + 13.25 * i as f32 + 0.125 * j as f32
        }),
        affine_shapes: Array3::from_shape_fn((FEATURES, 2, 2), |(i, j, k)| {
            1.5 + i as f32 - 0.25 * j as f32 + 0.0625 * k as f32
        }),
        descriptors: Array2::from_shape_fn((FEATURES, 128), |(i, j)| (i * 31 + j * 7) as u8),
        thumbnail_y_x_rgb: Array3::from_shape_fn(
            (THUMBNAIL_SIZE, THUMBNAIL_SIZE, 3),
            |(y, x, c)| (y + 2 * x + 3 * c) as u8,
        ),
    };
    sfmtool_sift_format::write_sift(&path, &data, SIFT_ZSTD_LEVEL).unwrap();
    path
}

/// The destination folder of the tests, beside the version 1 one.
fn destination(image: &Path) -> PathBuf {
    image.parent().unwrap().join("features/v2")
}

fn f32_bits<'a>(values: impl IntoIterator<Item = &'a f32>) -> Vec<u32> {
    values.into_iter().map(|v| v.to_bits()).collect()
}

fn assert_same_hashes(a: &SiftContentHash, b: &SiftContentHash) {
    assert_eq!(a.content_xxh128, b.content_xxh128);
    assert_eq!(a.feature_tool_xxh128, b.feature_tool_xxh128);
    assert_eq!(a.metadata_xxh128, b.metadata_xxh128);
}

/// Convert a version 1 file of `feature_tool` and check everything the
/// conversion promises; `image_to_gray` is the conversion the readings must
/// be taken through, stated by the test rather than read from the code.
fn check_conversion(feature_tool: &str, v1_options: serde_json::Value, image_to_gray: ImageToGray) {
    let (_dir, [image, _]) = images();
    let source = write_version_1(&image, feature_tool, v1_options.clone());
    let source_bytes = std::fs::read(&source).unwrap();
    let old = sfmtool_sift_format::read_sift(&source).unwrap();

    let folder = Version2SiftFolder::create(&destination(&image)).unwrap();
    let converted = folder
        .convert_version_1_sift_file(&source, &image, &Progress::none())
        .unwrap();
    assert_eq!(
        converted.path,
        destination(&image).join("seoul_bull_sculpture_01.jpg.sift")
    );
    let new = sfmtool_sift_format::read_sift(&converted.path).unwrap();

    // The features and the thumbnail are copied byte for byte.
    assert_eq!(
        f32_bits(&new.positions_xy),
        f32_bits(&old.positions_xy),
        "positions"
    );
    assert_eq!(
        f32_bits(&new.affine_shapes),
        f32_bits(&old.affine_shapes),
        "affine shapes"
    );
    assert_eq!(new.descriptors, old.descriptors);
    assert_eq!(new.thumbnail_y_x_rgb, old.thumbnail_y_x_rgb);

    // The record keeps its tool and type, and its options are upgraded.
    assert_eq!(new.feature_tool_metadata.feature_tool, feature_tool);
    assert_eq!(new.feature_tool_metadata.feature_type, "sift");
    assert_eq!(
        new.feature_tool_metadata.feature_options,
        crate::workspace::upgrade_feature_options(feature_tool, &v1_options).unwrap()
    );
    assert_eq!(
        new.feature_tool_metadata.feature_options["image_to_gray"],
        image_to_gray.to_json()
    );

    // The metadata keeps every field but the version, and gains the readings
    // a fresh computation from the image gives.
    assert_eq!(new.metadata.version, 2);
    assert_eq!(new.metadata.image_name, old.metadata.image_name);
    assert_eq!(
        new.metadata.image_file_xxh128,
        old.metadata.image_file_xxh128
    );
    assert_eq!(new.metadata.image_file_size, old.metadata.image_file_size);
    assert_eq!(new.metadata.image_width, old.metadata.image_width);
    assert_eq!(new.metadata.image_height, old.metadata.image_height);
    assert_eq!(new.metadata.feature_count, old.metadata.feature_count);
    let decoded = ImageU8::read_rgb(&image).unwrap();
    let expected = image_readings_from_rgb(
        decoded.width(),
        decoded.height(),
        decoded.data(),
        &image_to_gray,
    )
    .unwrap();
    assert!(expected.sharpness > 0.0);
    assert_eq!(new.metadata.sharpness, Some(f64::from(expected.sharpness)));
    assert_eq!(
        new.metadata.gradient_structure_tensor_xx_yy_xy,
        Some(expected.gradient_structure_tensor_xx_yy_xy.map(f64::from))
    );

    // The hashes returned are the ones the two files store.
    assert_same_hashes(&converted.source_content_hash, &old.content_hash);
    assert_same_hashes(&converted.content_hash, &new.content_hash);
    assert_ne!(
        converted.content_hash.content_xxh128,
        converted.source_content_hash.content_xxh128
    );

    // The new file verifies, and the source is untouched.
    assert_eq!(
        sfmtool_sift_format::verify_sift(&converted.path).unwrap(),
        (true, Vec::new())
    );
    assert_eq!(std::fs::read(&source).unwrap(), source_bytes);
}

#[test]
fn an_sfmtool_file_converts_losslessly() {
    // A formula other than the default, so the upgrade has to carry it over.
    let formula = "0.25*R + 0.5*G + 0.25*B";
    check_conversion(
        "sfmtool",
        serde_json::json!({"peak_ratio": 0.8, "max_num_features": 8192, "gray_formula": formula}),
        ImageToGray::new(formula, None).unwrap(),
    );
}

#[test]
fn an_sfmtool_file_without_a_formula_converts_through_the_default() {
    check_conversion(
        "sfmtool",
        serde_json::json!({"peak_ratio": 0.8}),
        ImageToGray::new(DEFAULT_GRAY_FORMULA, None).unwrap(),
    );
}

#[test]
fn a_colmap_file_converts_losslessly() {
    check_conversion(
        "colmap",
        serde_json::json!({"max_num_features": 3000, "peak_threshold": 0.006666666666666667,
                           "normalization": "L1_ROOT", "upright": false}),
        ImageToGray::new(DEFAULT_GRAY_FORMULA, Some(8)).unwrap(),
    );
}

#[test]
fn an_opencv_file_converts_losslessly() {
    check_conversion(
        "opencv",
        serde_json::json!({"nfeatures": 0, "contrastThreshold": 0.04}),
        tool_image_to_gray("opencv").unwrap(),
    );
}

/// The folder holds no file after a refused conversion.
fn assert_nothing_written(folder: &Version2SiftFolder) {
    let entries: Vec<_> = std::fs::read_dir(folder.path()).unwrap().collect();
    assert!(entries.is_empty(), "{entries:?}");
}

#[test]
fn a_version_2_file_is_refused() {
    let (dir, [image, _]) = images();
    let source = write_version_1(&image, "sfmtool", serde_json::json!({}));
    let first = Version2SiftFolder::create(&destination(&image)).unwrap();
    let converted = first
        .convert_version_1_sift_file(&source, &image, &Progress::none())
        .unwrap();
    let second = Version2SiftFolder::create(&dir.path().join("again")).unwrap();
    let error = second
        .convert_version_1_sift_file(&converted.path, &image, &Progress::none())
        .unwrap_err();
    assert!(
        matches!(error, ConvertSiftFileError::NotVersion1 { version: 2, .. }),
        "{error:?}"
    );
    assert!(
        error.to_string().contains("version 2 .sift file"),
        "{error}"
    );
    assert_nothing_written(&second);
}

#[test]
fn a_file_that_is_not_a_sift_file_is_refused() {
    let (_dir, [image, _]) = images();
    let folder = Version2SiftFolder::create(&destination(&image)).unwrap();
    let error = folder
        .convert_version_1_sift_file(&image, &image, &Progress::none())
        .unwrap_err();
    assert!(
        matches!(error, ConvertSiftFileError::Read { .. }),
        "{error:?}"
    );
    assert_nothing_written(&folder);
}

#[test]
fn a_missing_image_is_refused() {
    let (dir, [image, _]) = images();
    let source = write_version_1(&image, "sfmtool", serde_json::json!({}));
    let folder = Version2SiftFolder::create(&destination(&image)).unwrap();
    let missing = dir.path().join("images/gone.jpg");
    let error = folder
        .convert_version_1_sift_file(&source, &missing, &Progress::none())
        .unwrap_err();
    assert!(
        matches!(&error, ConvertSiftFileError::ImageMissing { image_path, .. } if *image_path == missing),
        "{error:?}"
    );
    assert_nothing_written(&folder);
}

#[test]
fn a_changed_image_is_refused() {
    let (_dir, [image, other]) = images();
    let source = write_version_1(&image, "colmap", serde_json::json!({}));
    let folder = Version2SiftFolder::create(&destination(&image)).unwrap();
    let error = folder
        .convert_version_1_sift_file(&source, &other, &Progress::none())
        .unwrap_err();
    let (other_hash, _) = xxh128_of_file(&other).unwrap();
    assert!(
        matches!(&error, ConvertSiftFileError::ImageChanged { image_xxh128, .. } if *image_xxh128 == other_hash),
        "{error:?}"
    );
    assert_nothing_written(&folder);
}

#[test]
fn an_image_that_does_not_decode_is_refused() {
    let (_dir, [image, _]) = images();
    let source = write_version_1(&image, "sfmtool", serde_json::json!({}));
    // Replace the image with bytes that are no image, and record their hash,
    // so the file passes the hash check and fails the decode.
    std::fs::write(&image, b"not an image").unwrap();
    let mut data = sfmtool_sift_format::read_sift(&source).unwrap();
    data.metadata.image_file_xxh128 = xxh128_of_file(&image).unwrap().0;
    sfmtool_sift_format::write_sift(&source, &data, SIFT_ZSTD_LEVEL).unwrap();
    let folder = Version2SiftFolder::create(&destination(&image)).unwrap();
    let error = folder
        .convert_version_1_sift_file(&source, &image, &Progress::none())
        .unwrap_err();
    assert!(
        matches!(error, ConvertSiftFileError::ImageUnreadable { .. }),
        "{error:?}"
    );
    assert!(
        error.to_string().contains("could not be decoded"),
        "{error}"
    );
    assert_nothing_written(&folder);
}

#[test]
fn an_image_of_another_size_is_refused() {
    let (_dir, [image, _]) = images();
    let source = write_version_1(&image, "sfmtool", serde_json::json!({}));
    let mut data = sfmtool_sift_format::read_sift(&source).unwrap();
    data.metadata.image_width += 1;
    sfmtool_sift_format::write_sift(&source, &data, SIFT_ZSTD_LEVEL).unwrap();
    let folder = Version2SiftFolder::create(&destination(&image)).unwrap();
    let error = folder
        .convert_version_1_sift_file(&source, &image, &Progress::none())
        .unwrap_err();
    assert!(
        matches!(
            error,
            ConvertSiftFileError::ImageSize {
                recorded: (271, 480),
                decoded: (270, 480),
                ..
            }
        ),
        "{error:?}"
    );
    assert_eq!(
        error.to_string(),
        format!(
            "the image {} decodes to 270 x 480 pixels, and its .sift file records 271 x 480",
            image.display()
        )
    );
    assert_nothing_written(&folder);
}

#[test]
fn readings_that_are_not_finite_are_refused() {
    let (_dir, [image, _]) = images();
    // The square root of -1 is NaN at every pixel, so every reading is NaN.
    let formula = "-1 ** 0.5";
    let source = write_version_1(
        &image,
        "sfmtool",
        serde_json::json!({"gray_formula": formula}),
    );
    let folder = Version2SiftFolder::create(&destination(&image)).unwrap();
    let error = folder
        .convert_version_1_sift_file(&source, &image, &Progress::none())
        .unwrap_err();
    assert!(
        matches!(
            &error,
            ConvertSiftFileError::Readings { image_path, error: ImageReadingsError::NotFinite { .. } }
                if *image_path == image
        ),
        "{error:?}"
    );
    let message = error.to_string();
    let prefix = format!(
        "failed to take the readings of the image {}: ",
        image.display()
    );
    assert!(message.starts_with(&prefix), "{message}");
    assert!(message[prefix.len()..].contains(formula), "{message}");
    assert_nothing_written(&folder);
}

#[test]
fn a_directory_in_place_of_the_converted_file_is_refused() {
    let (_dir, [image, _]) = images();
    let source = write_version_1(&image, "sfmtool", serde_json::json!({}));
    let folder = Version2SiftFolder::create(&destination(&image)).unwrap();
    let blocked = folder.path().join(source.file_name().unwrap());
    std::fs::create_dir(&blocked).unwrap();
    let error = folder
        .convert_version_1_sift_file(&source, &image, &Progress::none())
        .unwrap_err();
    assert!(
        matches!(&error, ConvertSiftFileError::Write { path, .. } if *path == blocked),
        "{error:?}"
    );
    let message = error.to_string();
    let prefix = format!("failed to write {}: ", blocked.display());
    assert!(
        message.starts_with(&prefix) && message.len() > prefix.len(),
        "{message}"
    );
    // The directory is left as it was, and no temporary file is left beside it.
    let entries: Vec<_> = std::fs::read_dir(folder.path())
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect();
    assert_eq!(entries, vec![blocked.clone()]);
    assert!(std::fs::read_dir(&blocked).unwrap().next().is_none());
}

#[test]
fn an_undistorted_file_is_refused_and_told_to_run_undistort_again() {
    let (_dir, [image, _]) = images();
    let source = write_version_1(
        &image,
        "sfmtool-undistort",
        serde_json::json!({"source_feature_tool": "colmap", "source_feature_options": {}}),
    );
    let folder = Version2SiftFolder::create(&destination(&image)).unwrap();
    let error = folder
        .convert_version_1_sift_file(&source, &image, &Progress::none())
        .unwrap_err();
    assert!(
        matches!(
            error,
            ConvertSiftFileError::FeatureOptions {
                error: FeatureOptionsError::Undistort,
                ..
            }
        ),
        "{error:?}"
    );
    assert!(
        error.to_string().contains("run 'sfm undistort' again"),
        "{error}"
    );
    assert_nothing_written(&folder);
}

#[test]
fn an_unknown_feature_tool_is_refused() {
    let (_dir, [image, _]) = images();
    let source = write_version_1(&image, "superpoint", serde_json::json!({}));
    let folder = Version2SiftFolder::create(&destination(&image)).unwrap();
    let error = folder
        .convert_version_1_sift_file(&source, &image, &Progress::none())
        .unwrap_err();
    assert!(
        matches!(
            &error,
            ConvertSiftFileError::FeatureOptions { error: FeatureOptionsError::UnknownTool(tool), .. }
                if tool == "superpoint"
        ),
        "{error:?}"
    );
    assert_nothing_written(&folder);
}

#[test]
fn a_folder_holding_version_1_files_is_refused() {
    let (_dir, [image, _]) = images();
    let source = write_version_1(&image, "sfmtool", serde_json::json!({}));
    let v1_folder = source.parent().unwrap();
    let error = Version2SiftFolder::create(v1_folder).unwrap_err();
    match &error {
        ConvertSiftFileError::Version1Files(found) => {
            assert_eq!(found.folder, v1_folder);
            assert_eq!(found.files, vec![source.clone()]);
        }
        other => panic!("expected Version1Files, got {other:?}"),
    }

    // A file in the folder's place cannot be listed.
    let error = Version2SiftFolder::create(&image).unwrap_err();
    assert!(
        matches!(&error, ConvertSiftFileError::ListFolder { folder, .. } if *folder == image),
        "{error:?}"
    );
    let message = error.to_string();
    let prefix = format!("failed to list the folder {}: ", image.display());
    assert!(
        message.starts_with(&prefix) && message.len() > prefix.len(),
        "{message}"
    );
}

#[test]
fn a_folder_that_cannot_be_made_is_refused() {
    let (dir, [image, _]) = images();
    // A path that is not there, so there is nothing to list, and that cannot
    // be made. On Windows a path under a regular file is reported as not
    // found; on Linux and macOS it is "not a directory", which is a failure
    // to list, so there the path is a symbolic link to nothing.
    #[cfg(windows)]
    let blocked = {
        let _ = &dir;
        image.join("v2")
    };
    #[cfg(unix)]
    let blocked = {
        let link = dir.path().join("v2");
        std::os::unix::fs::symlink(dir.path().join("nowhere"), &link).unwrap();
        let _ = &image;
        link
    };
    let error = Version2SiftFolder::create(&blocked).unwrap_err();
    assert!(
        matches!(&error, ConvertSiftFileError::CreateFolder { folder, .. } if *folder == blocked),
        "{error:?}"
    );
    let message = error.to_string();
    let prefix = format!("failed to create the folder {}: ", blocked.display());
    assert!(
        message.starts_with(&prefix) && message.len() > prefix.len(),
        "{message}"
    );
}

#[test]
fn a_cancelled_conversion_writes_nothing() {
    let (_dir, [image, _]) = images();
    let source = write_version_1(&image, "sfmtool", serde_json::json!({}));
    let folder = Version2SiftFolder::create(&destination(&image)).unwrap();
    let cancel = AtomicBool::new(true);
    let error = folder
        .convert_version_1_sift_file(&source, &image, &Progress::none().cancelled_by(&cancel))
        .unwrap_err();
    assert!(
        matches!(error, ConvertSiftFileError::Cancelled),
        "{error:?}"
    );
    assert_eq!(
        error.to_string(),
        "the conversion was cancelled; nothing was written"
    );
    assert_nothing_written(&folder);
}
