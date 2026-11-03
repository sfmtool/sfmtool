// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use crate::*;
use ndarray::{Array2, Array3};

/// Create minimal valid version 2 SiftData for testing.
fn make_test_data() -> SiftData {
    let feature_count: u32 = 4;
    let n = feature_count as usize;

    SiftData {
        feature_tool_metadata: FeatureToolMetadata {
            feature_tool: "colmap".into(),
            feature_type: "sift".into(),
            feature_options: serde_json::json!({
                "max_num_features": 8192,
                "image_to_gray": {"formula": "0.2126*R + 0.7152*G + 0.0722*B", "quantize_bits": 8},
            }),
        },
        metadata: SiftMetadata {
            version: 2,
            image_name: "test_image.jpg".into(),
            image_file_xxh128: "abcdef0123456789abcdef0123456789".into(),
            image_file_size: 12345,
            image_width: 640,
            image_height: 480,
            feature_count,
            sharpness: Some(0.004_120_6),
            gradient_structure_tensor_xx_yy_xy: Some([0.000_931_14, 0.000_714_02, -8.351e-5]),
        },
        content_hash: SiftContentHash::default(),
        positions_xy: Array2::from_shape_vec(
            (n, 2),
            vec![100.5, 200.5, 50.5, 75.5, 320.5, 240.5, 10.5, 10.5],
        )
        .unwrap(),
        affine_shapes: Array3::from_shape_vec(
            (n, 2, 2),
            vec![
                5.0, 0.0, 0.0, 5.0, // identity scaled by 5
                3.0, -1.0, 1.0, 3.0, // rotated + scaled
                10.0, 0.0, 0.0, 10.0, // identity scaled by 10
                2.0, 0.0, 0.0, 2.0, // identity scaled by 2
            ],
        )
        .unwrap(),
        descriptors: Array2::from_shape_fn((n, 128), |(i, j)| ((i * 128 + j) % 256) as u8),
        thumbnail_y_x_rgb: Array3::from_shape_fn((128, 128, 3), |(y, x, c)| {
            ((y * 128 + x + c) % 256) as u8
        }),
    }
}

#[test]
fn test_round_trip() {
    let data = make_test_data();
    let dir = std::env::temp_dir().join("sift_test_round_trip");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("test.sift");

    // Write
    write_sift(&path, &data, 3).unwrap();

    // Read back
    let loaded = read_sift(&path).unwrap();

    // Verify metadata
    assert_eq!(loaded.metadata.image_name, "test_image.jpg");
    assert_eq!(loaded.metadata.feature_count, 4);
    assert_eq!(loaded.metadata.image_width, 640);
    assert_eq!(loaded.metadata.image_height, 480);

    // Verify feature tool metadata
    assert_eq!(loaded.feature_tool_metadata.feature_tool, "colmap");

    // Verify arrays
    assert_eq!(loaded.positions_xy, data.positions_xy);
    assert_eq!(loaded.affine_shapes, data.affine_shapes);
    assert_eq!(loaded.descriptors, data.descriptors);
    assert_eq!(loaded.thumbnail_y_x_rgb, data.thumbnail_y_x_rgb);

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn test_round_trip_verify() {
    let data = make_test_data();
    let dir = std::env::temp_dir().join("sift_test_verify");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("test.sift");

    write_sift(&path, &data, 3).unwrap();

    let (valid, errors) = verify_sift(&path).unwrap();
    assert!(valid, "Verification failed: {:?}", errors);
    assert!(errors.is_empty());

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn test_read_metadata_only() {
    let data = make_test_data();
    let dir = std::env::temp_dir().join("sift_test_metadata");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("test.sift");

    write_sift(&path, &data, 3).unwrap();

    let (tool_meta, meta, hash) = read_sift_metadata(&path).unwrap();
    assert_eq!(meta.image_name, "test_image.jpg");
    assert_eq!(meta.feature_count, 4);
    assert_eq!(tool_meta.feature_tool, "colmap");
    assert_eq!(hash.feature_tool_xxh128.len(), 32);

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn test_read_thumbnail_only() {
    let data = make_test_data();
    let dir = std::env::temp_dir().join("sift_test_thumbnail");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("test.sift");

    write_sift(&path, &data, 3).unwrap();

    let (meta, hash, thumbnail) = read_sift_thumbnail(&path).unwrap();
    let full = read_sift(&path).unwrap();
    assert_eq!(thumbnail, data.thumbnail_y_x_rgb);
    assert_eq!(meta.image_file_xxh128, full.metadata.image_file_xxh128);
    assert_eq!(hash.content_xxh128, full.content_hash.content_xxh128);

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn test_content_hash_populated() {
    // Version 1 data, so the bytes are those a version 1 writer wrote.
    let data = make_version_1_data();
    let dir = std::env::temp_dir().join("sift_test_hash");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("test.sift");

    write_sift(&path, &data, 3).unwrap();
    let loaded = read_sift(&path).unwrap();
    // Frozen from the original writer at HEAD 6528c746.
    assert_eq!(
        loaded.content_hash.content_xxh128,
        "577fc9bd3d53068b8e198073ffa7d840"
    );

    // All hashes should be 32-char hex strings
    assert_eq!(loaded.content_hash.feature_tool_xxh128.len(), 32);
    assert_eq!(loaded.content_hash.content_xxh128.len(), 32);

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn test_partial_read() {
    let data = make_test_data();
    let dir = std::env::temp_dir().join("sift_test_partial");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("test.sift");

    write_sift(&path, &data, 3).unwrap();

    // Read only first 2 features
    let partial = read_sift_partial(&path, 2).unwrap();
    assert_eq!(partial.positions_xy.shape(), &[2, 2]);
    assert_eq!(partial.affine_shapes.shape(), &[2, 2, 2]);
    assert_eq!(partial.descriptors.shape(), &[2, 128]);

    // Verify data matches first 2 rows
    assert_eq!(
        partial.positions_xy,
        data.positions_xy.slice(ndarray::s![..2, ..]).to_owned()
    );
    assert_eq!(
        partial.descriptors,
        data.descriptors.slice(ndarray::s![..2, ..]).to_owned()
    );

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn test_partial_read_oversize() {
    let data = make_test_data();
    let dir = std::env::temp_dir().join("sift_test_partial_over");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("test.sift");

    write_sift(&path, &data, 3).unwrap();

    // Request more than available — should return all
    let partial = read_sift_partial(&path, 100).unwrap();
    assert_eq!(partial.positions_xy.shape(), &[4, 2]);

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn test_write_validation_wrong_positions_shape() {
    let mut data = make_test_data();
    // Wrong positions shape: 3 positions but feature_count = 4
    data.positions_xy = Array2::zeros((3, 2));

    let dir = std::env::temp_dir().join("sift_test_bad_pos");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("test.sift");

    let result = write_sift(&path, &data, 3);
    assert!(result.is_err());
    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("positions_xy") || err_msg.contains("Shape"),
        "Error should mention positions_xy shape, got: {err_msg}"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_write_validation_wrong_affine_shape() {
    let mut data = make_test_data();
    // Wrong affine shapes: 3 features instead of 4
    data.affine_shapes = Array3::zeros((3, 2, 2));

    let dir = std::env::temp_dir().join("sift_test_bad_aff");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("test.sift");

    let result = write_sift(&path, &data, 3);
    assert!(result.is_err());
    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("affine") || err_msg.contains("Shape"),
        "Error should mention affine shape, got: {err_msg}"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_write_validation_wrong_descriptors_shape() {
    let mut data = make_test_data();
    // Wrong descriptor shape: 3 features instead of 4
    data.descriptors = Array2::zeros((3, 128));

    let dir = std::env::temp_dir().join("sift_test_bad_desc");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("test.sift");

    let result = write_sift(&path, &data, 3);
    assert!(result.is_err());
    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("descriptor") || err_msg.contains("Shape"),
        "Error should mention descriptor shape, got: {err_msg}"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_write_validation_all_wrong_shapes() {
    // All arrays have wrong shapes simultaneously
    let mut data = make_test_data();
    data.positions_xy = Array2::zeros((2, 2));
    data.affine_shapes = Array3::zeros((5, 2, 2));
    data.descriptors = Array2::zeros((3, 128));

    let dir = std::env::temp_dir().join("sift_test_all_bad");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("test.sift");

    let result = write_sift(&path, &data, 3);
    assert!(result.is_err());

    let _ = std::fs::remove_dir_all(&dir);
}

/// Write `data` with `metadata.version` set to `version`, bypassing the
/// writer's own version check, and return the file's directory and path.
fn write_with_version(version: u32, name: &str) -> (std::path::PathBuf, std::path::PathBuf) {
    let mut data = make_test_data();
    data.metadata.version = version;
    let dir = std::env::temp_dir().join(name);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("test.sift");
    crate::write::write_sift_any_version(&path, &data, 3).unwrap();
    (dir, path)
}

#[test]
fn test_read_rejects_unsupported_versions() {
    for (version, name) in [
        (0, "sift_test_version_zero"),
        (SIFT_FORMAT_VERSION + 1, "sift_test_future_version"),
    ] {
        let (dir, path) = write_with_version(version, name);
        for result in [
            read_sift(&path).err(),
            read_sift_partial(&path, 2).err(),
            read_sift_positions(&path, 2).err(),
            read_sift_keypoints(&path, 2).err(),
            read_sift_features(&path).err(),
            read_sift_thumbnail(&path).err(),
            read_sift_metadata(&path).err(),
        ] {
            let err = result.unwrap_or_else(|| panic!("version {version} must be rejected"));
            let msg = err.to_string();
            assert!(
                msg.contains("unsupported .sift format version"),
                "unexpected error: {msg}"
            );
        }

        // The hashes are consistent, so the version is the only error.
        let (ok, errors) = verify_sift(&path).unwrap();
        assert!(!ok, "version {version} must fail verification");
        assert_eq!(errors.len(), 1, "{errors:?}");
        assert!(errors[0].contains("unsupported .sift format version"));

        std::fs::remove_dir_all(&dir).unwrap();
    }
}

#[test]
fn test_write_rejects_unsupported_versions() {
    for version in [0, SIFT_FORMAT_VERSION + 1] {
        let mut data = make_test_data();
        data.metadata.version = version;
        let dir = std::env::temp_dir().join(format!("sift_test_write_version_{version}"));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.sift");

        let err = write_sift(&path, &data, 3).expect_err("unsupported version must be rejected");
        assert!(err.to_string().contains("unsupported .sift format version"));
        assert!(!path.exists(), "a rejected write must leave no file");

        std::fs::remove_dir_all(&dir).unwrap();
    }
}

#[test]
fn test_read_nonexistent_file() {
    let result = read_sift(std::path::Path::new("nonexistent.sift"));
    match result {
        Err(err) => {
            let msg = err.to_string();
            assert!(
                msg.contains("nonexistent.sift"),
                "Error message should contain the file path, got: {msg}"
            );
        }
        Ok(_) => panic!("Expected error for nonexistent file"),
    }
}

// Every entry in a .sift archive MUST use ZIP's STORE method. Entries are already
// zstandard-compressed; applying ZIP-level DEFLATE would double-compress and break
// the spec guarantee of random access via simple seek.
#[test]
fn test_archive_uses_stored_compression() {
    let data = make_test_data();
    let dir = std::env::temp_dir().join("sift_test_stored_compression");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("test.sift");

    write_sift(&path, &data, 3).unwrap();

    let file = std::fs::File::open(&path).unwrap();
    let mut archive = zip::ZipArchive::new(file).unwrap();
    assert!(!archive.is_empty(), "expected a populated archive");
    for i in 0..archive.len() {
        let entry = archive.by_index(i).unwrap();
        assert_eq!(
            entry.compression(),
            zip::CompressionMethod::Stored,
            "entry '{}' uses {:?}, expected Stored",
            entry.name().unwrap(),
            entry.compression(),
        );
    }

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn test_validate_rejects_wrong_thumbnail_shape() {
    let mut data = make_test_data();
    data.thumbnail_y_x_rgb = Array3::zeros((64, 64, 3));
    let err = validate_sift_data(&data).expect_err("a 64x64 thumbnail must be rejected");
    assert!(matches!(err, SiftError::ShapeMismatch(_)), "{err}");
    assert!(err.to_string().contains("thumbnail_y_x_rgb"), "{err}");
}

#[test]
fn test_validate_rejects_shape_count_mismatch() {
    let mut data = make_test_data();
    data.metadata.feature_count = 5;
    let err = validate_sift_data(&data).expect_err("feature_count 5 with 4 rows must fail");
    assert!(matches!(err, SiftError::ShapeMismatch(_)), "{err}");
    assert!(err.to_string().contains("positions_xy"), "{err}");
}

#[test]
fn test_validate_rejects_non_object_feature_options() {
    let mut data = make_test_data();
    data.feature_tool_metadata.feature_options = serde_json::json!([1, 2]);
    let err = validate_sift_data(&data).expect_err("a list of options must be rejected");
    assert!(matches!(err, SiftError::InvalidMetadata(_)), "{err}");
    assert!(err.to_string().contains("feature_options"), "{err}");

    let dir = std::env::temp_dir().join("sift_test_bad_feature_options");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("test.sift");
    assert!(write_sift(&path, &data, 3).is_err());
    assert!(!path.exists(), "a rejected write must leave no file");
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn test_validate_accepts_valid_data() {
    validate_sift_data(&make_test_data()).unwrap();
}

fn valid_metadata_json() -> serde_json::Value {
    serde_json::to_value(make_test_data().metadata).unwrap()
}

#[test]
fn test_metadata_for_write_rejects_unknown_missing_and_mistyped_keys() {
    SiftMetadata::from_json_for_write(valid_metadata_json()).unwrap();

    let mut extra = valid_metadata_json();
    extra["colour"] = serde_json::json!("blue");
    let err = SiftMetadata::from_json_for_write(extra).unwrap_err();
    assert!(matches!(err, SiftError::InvalidMetadata(_)), "{err}");
    let msg = err.to_string();
    assert!(msg.contains("metadata: unknown field `colour`"), "{msg}");
    assert!(msg.contains("image_width"), "names the defined keys: {msg}");

    let mut missing = valid_metadata_json();
    missing.as_object_mut().unwrap().remove("image_width");
    let err = SiftMetadata::from_json_for_write(missing).unwrap_err();
    assert!(
        err.to_string().contains("missing field `image_width`"),
        "{err}"
    );

    let mut mistyped = valid_metadata_json();
    mistyped["feature_count"] = serde_json::json!("4");
    let err = SiftMetadata::from_json_for_write(mistyped).unwrap_err();
    assert!(err.to_string().contains("invalid type: string"), "{err}");

    let err = SiftMetadata::from_json_for_write(serde_json::json!([1])).unwrap_err();
    assert!(err.to_string().contains("must be a JSON object"), "{err}");

    let tool = serde_json::json!({
        "feature_tool": "colmap",
        "feature_type": "sift-colmap",
        "feature_options": {},
        "feature_version": 2,
    });
    let err = FeatureToolMetadata::from_json_for_write(tool).unwrap_err();
    assert!(
        err.to_string()
            .contains("feature_tool_metadata: unknown field `feature_version`"),
        "{err}"
    );
}

/// Write a `.sift` archive holding just the three JSON entries, with
/// `metadata.json` taken as given, for the readers that stop after them.
fn write_raw_metadata_archive(name: &str, metadata: &serde_json::Value) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(name);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("test.sift");
    let file = std::fs::File::create(&path).unwrap();
    let mut zip = zip::ZipWriter::new(file);
    let data = make_test_data();
    sfmtool_archive_io::write_json_entry(
        &mut zip,
        "feature_tool_metadata.json.zst",
        &data.feature_tool_metadata,
        3,
    )
    .unwrap();
    sfmtool_archive_io::write_json_entry(&mut zip, "metadata.json.zst", metadata, 3).unwrap();
    sfmtool_archive_io::write_json_entry(
        &mut zip,
        "content_hash.json.zst",
        &SiftContentHash::default(),
        3,
    )
    .unwrap();
    zip.finish().unwrap();
    path
}

/// The format asks a reader to ignore a metadata key it does not know, so a
/// later version can add one; only a writer is strict.
#[test]
fn test_read_ignores_unknown_metadata_keys() {
    let mut metadata = valid_metadata_json();
    metadata["added_in_a_later_version"] = serde_json::json!(true);
    let path = write_raw_metadata_archive("sift_test_read_extra_key", &metadata);
    let (_, read, _) = read_sift_metadata(&path).unwrap();
    assert_eq!(read.image_name, "test_image.jpg");
    std::fs::remove_dir_all(path.parent().unwrap()).unwrap();
}

/// [`make_test_data`] as version 1 data: no `image_to_gray` and no readings.
fn make_version_1_data() -> SiftData {
    let mut data = make_test_data();
    data.metadata.version = 1;
    data.metadata.sharpness = None;
    data.metadata.gradient_structure_tensor_xx_yy_xy = None;
    data.feature_tool_metadata.feature_options = serde_json::json!({"max_num_features": 8192});
    data
}

#[test]
fn test_version_2_round_trip_keeps_readings_exactly() {
    let data = make_test_data();
    let dir = std::env::temp_dir().join("sift_test_v2_round_trip");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("test.sift");
    write_sift(&path, &data, 3).unwrap();
    let loaded = read_sift(&path).unwrap();
    assert_eq!(loaded.metadata.version, 2);
    assert_eq!(loaded.metadata.sharpness, data.metadata.sharpness);
    assert_eq!(
        loaded.metadata.gradient_structure_tensor_xx_yy_xy,
        data.metadata.gradient_structure_tensor_xx_yy_xy
    );
    assert_eq!(
        loaded.feature_tool_metadata.feature_options["image_to_gray"],
        serde_json::json!({"formula": "0.2126*R + 0.7152*G + 0.0722*B", "quantize_bits": 8})
    );
    let (_, metadata, _) = read_sift_metadata(&path).unwrap();
    assert_eq!(metadata.sharpness, data.metadata.sharpness);
    assert_eq!(verify_sift(&path).unwrap(), (true, vec![]));
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn test_version_1_round_trip_reads_readings_absent() {
    let data = make_version_1_data();
    let dir = std::env::temp_dir().join("sift_test_v1_round_trip");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("test.sift");
    write_sift(&path, &data, 3).unwrap();
    let loaded = read_sift(&path).unwrap();
    assert_eq!(loaded.metadata.version, 1);
    assert_eq!(loaded.metadata.sharpness, None);
    assert_eq!(loaded.metadata.gradient_structure_tensor_xx_yy_xy, None);
    // The keys are not written at all, so a version 1 file holds only the
    // keys a version 1 writer wrote.
    let json = serde_json::to_value(&loaded.metadata).unwrap();
    assert!(json.get("sharpness").is_none());
    assert!(json.get("gradient_structure_tensor_xx_yy_xy").is_none());
    assert_eq!(verify_sift(&path).unwrap(), (true, vec![]));
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn test_validate_version_rules() {
    let invalid = |data: &SiftData| validate_sift_data(data).unwrap_err().to_string();

    let mut data = make_test_data();
    data.metadata.sharpness = None;
    assert!(invalid(&data).contains("has no 'sharpness'"));

    let mut data = make_test_data();
    data.metadata.gradient_structure_tensor_xx_yy_xy = None;
    assert!(invalid(&data).contains("has no 'gradient_structure_tensor_xx_yy_xy'"));

    let mut data = make_test_data();
    data.feature_tool_metadata.feature_options = serde_json::json!({"max_num_features": 8192});
    assert!(invalid(&data).contains("must carry 'image_to_gray'"));

    for (image_to_gray, problem) in [
        (serde_json::json!("G"), "must be a JSON object"),
        (serde_json::json!({}), "has no 'formula'"),
        (
            serde_json::json!({"formula": 3}),
            "must be a non-empty string",
        ),
        (
            serde_json::json!({"formula": " "}),
            "must be a non-empty string",
        ),
        (
            serde_json::json!({"formula": "G", "quantize_bits": 0}),
            "from 1 to 16",
        ),
        (
            serde_json::json!({"formula": "G", "quantize_bits": 17}),
            "from 1 to 16",
        ),
        (
            serde_json::json!({"formula": "G", "quantize_bits": 8.5}),
            "from 1 to 16",
        ),
        (
            serde_json::json!({"formula": "G", "gamma": 2.2}),
            "unknown key",
        ),
    ] {
        let mut data = make_test_data();
        data.feature_tool_metadata.feature_options["image_to_gray"] = image_to_gray.clone();
        let message = invalid(&data);
        assert!(message.contains(problem), "{image_to_gray}: {message}");
    }

    for sharpness in [f64::NAN, f64::INFINITY, -1e-9] {
        let mut data = make_test_data();
        data.metadata.sharpness = Some(sharpness);
        assert!(invalid(&data).contains("sharpness must be finite"));
    }
    for tensor in [
        [f64::NAN, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, f64::INFINITY],
    ] {
        let mut data = make_test_data();
        data.metadata.gradient_structure_tensor_xx_yy_xy = Some(tensor);
        assert!(invalid(&data).contains("gradient_structure_tensor_xx_yy_xy must be finite"));
    }
    // A negative cross term is an ordinary value.
    let mut data = make_test_data();
    data.metadata.gradient_structure_tensor_xx_yy_xy = Some([1.0, 1.0, -0.5]);
    validate_sift_data(&data).unwrap();

    let mut data = make_version_1_data();
    data.metadata.sharpness = Some(0.1);
    assert!(invalid(&data).contains("which belongs to version 2"));

    let mut data = make_version_1_data();
    data.feature_tool_metadata.feature_options["image_to_gray"] =
        serde_json::json!({"formula": "G"});
    assert!(invalid(&data).contains("which belongs to version 2"));

    // Only the top level of feature_options records the conversion.
    let mut data = make_version_1_data();
    data.feature_tool_metadata.feature_options =
        serde_json::json!({"source_feature_options": {"image_to_gray": {"formula": "G"}}});
    validate_sift_data(&data).unwrap();
}

#[test]
fn test_read_and_verify_reject_version_2_without_readings() {
    let mut data = make_test_data();
    data.metadata.sharpness = None;
    let dir = std::env::temp_dir().join("sift_test_v2_no_readings");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("test.sift");
    crate::write::write_sift_any_version(&path, &data, 3).unwrap();
    let message = read_sift(&path).err().unwrap().to_string();
    assert!(message.contains("has no 'sharpness'"), "{message}");
    let (ok, errors) = verify_sift(&path).unwrap();
    assert!(!ok);
    assert_eq!(errors.len(), 1, "{errors:?}");
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn test_verify_reports_version_2_without_image_to_gray() {
    let mut data = make_test_data();
    data.feature_tool_metadata.feature_options = serde_json::json!({});
    let dir = std::env::temp_dir().join("sift_test_v2_no_image_to_gray");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("test.sift");
    crate::write::write_sift_any_version(&path, &data, 3).unwrap();
    let (ok, errors) = verify_sift(&path).unwrap();
    assert!(!ok);
    assert!(
        errors[0].contains("must carry 'image_to_gray'"),
        "{errors:?}"
    );
    std::fs::remove_dir_all(&dir).unwrap();
}
