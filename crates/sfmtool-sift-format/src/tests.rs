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
            sharpness: 0.004_120_6,
            gradient_structure_tensor_xx_yy_xy: [0.000_931_14, 0.000_714_02, -8.351e-5],
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
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().to_path_buf();
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
}

#[test]
fn test_round_trip_verify() {
    let data = make_test_data();
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().to_path_buf();
    let path = dir.join("test.sift");

    write_sift(&path, &data, 3).unwrap();

    let (valid, errors) = verify_sift(&path).unwrap();
    assert!(valid, "Verification failed: {:?}", errors);
    assert!(errors.is_empty());
}

#[test]
fn test_read_metadata_only() {
    let data = make_test_data();
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().to_path_buf();
    let path = dir.join("test.sift");

    write_sift(&path, &data, 3).unwrap();

    let (tool_meta, meta, hash) = read_sift_metadata(&path).unwrap();
    assert_eq!(meta.image_name, "test_image.jpg");
    assert_eq!(meta.feature_count, 4);
    assert_eq!(tool_meta.feature_tool, "colmap");
    assert_eq!(hash.feature_tool_xxh128.len(), 32);
}

#[test]
fn test_read_thumbnail_only() {
    let data = make_test_data();
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().to_path_buf();
    let path = dir.join("test.sift");

    write_sift(&path, &data, 3).unwrap();

    let (meta, hash, thumbnail) = read_sift_thumbnail(&path).unwrap();
    let full = read_sift(&path).unwrap();
    assert_eq!(thumbnail, data.thumbnail_y_x_rgb);
    assert_eq!(meta.image_file_xxh128, full.metadata.image_file_xxh128);
    assert_eq!(hash.content_xxh128, full.content_hash.content_xxh128);
}

#[test]
fn test_content_hash_populated() {
    let data = make_test_data();
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().to_path_buf();
    let path = dir.join("test.sift");

    write_sift(&path, &data, 3).unwrap();
    let loaded = read_sift(&path).unwrap();
    // Frozen from the writer at 6ab59c8d, before the in-memory readings
    // stopped being optional, so a change to the bytes written shows here.
    assert_eq!(
        loaded.content_hash.content_xxh128,
        "eb7a54df84a0952b22f90a22a373b26d"
    );

    // All hashes should be 32-char hex strings
    assert_eq!(loaded.content_hash.feature_tool_xxh128.len(), 32);
    assert_eq!(loaded.content_hash.content_xxh128.len(), 32);
}

#[test]
fn test_partial_read() {
    let data = make_test_data();
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().to_path_buf();
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
}

#[test]
fn test_read_positions_descriptors_reads_what_read_sift_features_reads() {
    let data = make_test_data();
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().to_path_buf();
    let path = dir.join("test.sift");
    write_sift(&path, &data, 3).unwrap();

    let (positions_xy, descriptors) = read_sift_positions_descriptors(&path).unwrap();
    let features = read_sift_features(&path).unwrap();
    assert_eq!(positions_xy, features.positions_xy);
    assert_eq!(descriptors, features.descriptors);
    // And both are what was written.
    assert_eq!(
        positions_xy.as_flattened(),
        data.positions_xy.as_slice().unwrap()
    );
    assert_eq!(descriptors, data.descriptors.as_slice().unwrap());
}

/// Copy the `.sift` archive at `from` to `to`, storing `stored` as the raw
/// bytes of the entry `name` in place of its own.
fn replace_entry(from: &std::path::Path, to: &std::path::Path, name: &str, stored: &[u8]) {
    use std::io::{Read, Write};
    let mut source = zip::ZipArchive::new(std::fs::File::open(from).unwrap()).unwrap();
    let mut zip = zip::ZipWriter::new(std::fs::File::create(to).unwrap());
    let options =
        zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored);
    for i in 0..source.len() {
        let mut entry = source.by_index(i).unwrap();
        let entry_name = entry.name().unwrap().into_owned();
        let mut bytes = Vec::new();
        entry.read_to_end(&mut bytes).unwrap();
        zip.start_file(entry_name.as_str(), options).unwrap();
        zip.write_all(if entry_name == name { stored } else { &bytes })
            .unwrap();
    }
    zip.finish().unwrap();
}

/// The stored (compressed) bytes of the entry `name` of the archive at `path`.
fn stored_entry(path: &std::path::Path, name: &str) -> Vec<u8> {
    use std::io::Read;
    let mut archive = zip::ZipArchive::new(std::fs::File::open(path).unwrap()).unwrap();
    let mut bytes = Vec::new();
    archive
        .by_name(name)
        .unwrap()
        .read_to_end(&mut bytes)
        .unwrap();
    bytes
}

#[test]
fn test_corrupt_feature_entries_are_errors() {
    let data = make_test_data();
    let n = data.metadata.feature_count as usize;
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().to_path_buf();
    let path = dir.join("test.sift");
    write_sift(&path, &data, 3).unwrap();

    // A truncated descriptors entry is an error naming the entry, the same
    // error from both readers that read it.
    let descriptors = crate::types::descriptors_entry_name(n);
    let stored = stored_entry(&path, &descriptors);
    let truncated = dir.join("truncated.sift");
    replace_entry(&path, &truncated, &descriptors, &stored[..stored.len() - 3]);
    let err = read_sift_positions_descriptors(&truncated).unwrap_err();
    assert!(matches!(err, SiftError::InvalidFormat(_)), "{err}");
    assert!(err.to_string().contains(&descriptors), "{err}");
    assert_eq!(
        err.to_string(),
        read_sift_features(&truncated).err().unwrap().to_string()
    );

    // A descriptors entry holding one feature too few is the length check's
    // error.
    let short = dir.join("short.sift");
    let one_fewer = sfmtool_archive_io::zstd_compress(
        &data.descriptors.as_slice().unwrap()[DESCRIPTOR_DIM..],
        3,
    )
    .unwrap();
    replace_entry(&path, &short, &descriptors, &one_fewer);
    let err = read_sift_positions_descriptors(&short).unwrap_err();
    assert!(matches!(err, SiftError::ShapeMismatch(_)), "{err}");
    assert_eq!(
        err.to_string(),
        read_sift_features(&short).err().unwrap().to_string()
    );

    // A corrupt affine shapes entry fails the reader that reads it, and not
    // the one that skips it.
    let affine = crate::types::affine_shapes_entry_name(n);
    let bad_affine = dir.join("bad_affine.sift");
    replace_entry(&path, &bad_affine, &affine, b"not a zstd frame");
    let err = read_sift_features(&bad_affine).err().unwrap();
    assert!(err.to_string().contains(&affine), "{err}");
    let (positions_xy, descriptors) = read_sift_positions_descriptors(&bad_affine).unwrap();
    assert_eq!(
        positions_xy.as_flattened(),
        data.positions_xy.as_slice().unwrap()
    );
    assert_eq!(descriptors, data.descriptors.as_slice().unwrap());
}

#[test]
fn test_partial_read_oversize() {
    let data = make_test_data();
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().to_path_buf();
    let path = dir.join("test.sift");

    write_sift(&path, &data, 3).unwrap();

    // Request more than available — should return all
    let partial = read_sift_partial(&path, 100).unwrap();
    assert_eq!(partial.positions_xy.shape(), &[4, 2]);
}

#[test]
fn test_write_validation_wrong_positions_shape() {
    let mut data = make_test_data();
    // Wrong positions shape: 3 positions but feature_count = 4
    data.positions_xy = Array2::zeros((3, 2));

    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().to_path_buf();
    let path = dir.join("test.sift");

    let result = write_sift(&path, &data, 3);
    assert!(result.is_err());
    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("positions_xy") || err_msg.contains("Shape"),
        "Error should mention positions_xy shape, got: {err_msg}"
    );
}

#[test]
fn test_write_validation_wrong_affine_shape() {
    let mut data = make_test_data();
    // Wrong affine shapes: 3 features instead of 4
    data.affine_shapes = Array3::zeros((3, 2, 2));

    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().to_path_buf();
    let path = dir.join("test.sift");

    let result = write_sift(&path, &data, 3);
    assert!(result.is_err());
    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("affine") || err_msg.contains("Shape"),
        "Error should mention affine shape, got: {err_msg}"
    );
}

#[test]
fn test_write_validation_wrong_descriptors_shape() {
    let mut data = make_test_data();
    // Wrong descriptor shape: 3 features instead of 4
    data.descriptors = Array2::zeros((3, 128));

    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().to_path_buf();
    let path = dir.join("test.sift");

    let result = write_sift(&path, &data, 3);
    assert!(result.is_err());
    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("descriptor") || err_msg.contains("Shape"),
        "Error should mention descriptor shape, got: {err_msg}"
    );
}

#[test]
fn test_write_validation_all_wrong_shapes() {
    // All arrays have wrong shapes simultaneously
    let mut data = make_test_data();
    data.positions_xy = Array2::zeros((2, 2));
    data.affine_shapes = Array3::zeros((5, 2, 2));
    data.descriptors = Array2::zeros((3, 128));

    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().to_path_buf();
    let path = dir.join("test.sift");

    let result = write_sift(&path, &data, 3);
    assert!(result.is_err());
}

/// Write `data` with `metadata.version` set to `version`, bypassing the
/// writer's own version check, into a new temporary directory whose name
/// starts with `name`; return the directory, removed when it drops, and the
/// file's path.
fn write_with_version(version: u32, name: &str) -> (tempfile::TempDir, std::path::PathBuf) {
    let mut data = make_test_data();
    data.metadata.version = version;
    let dir = tempfile::Builder::new().prefix(name).tempdir().unwrap();
    let path = dir.path().join("test.sift");
    crate::write::write_sift_any_version(&path, &data, 3).unwrap();
    (dir, path)
}

#[test]
fn test_read_rejects_unsupported_versions() {
    for (version, name) in [
        (0, "sift_test_version_zero"),
        (SIFT_FORMAT_VERSION + 1, "sift_test_future_version"),
    ] {
        let (_dir, path) = write_with_version(version, name);
        for result in [
            read_sift(&path).err(),
            read_sift_partial(&path, 2).err(),
            read_sift_positions(&path, 2).err(),
            read_sift_keypoints(&path, 2).err(),
            read_sift_features(&path).err(),
            read_sift_positions_descriptors(&path).err(),
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
    }
}

/// Write a version 1 file at `path` as a version 1 writer wrote it: the
/// entries of [`make_test_data`], with `feature_options` holding no
/// `image_to_gray`, `metadata.json` holding version 1 and no readings, and
/// the content hashes over those bytes.
fn write_version_1_file(path: &std::path::Path) {
    use sfmtool_archive_io::{row_major_bytes, write_binary_entry, write_json_entry};
    let data = make_test_data();
    let n = data.metadata.feature_count as usize;
    let feature_tool = serde_json::json!({
        "feature_tool": "colmap",
        "feature_type": "sift",
        "feature_options": {"max_num_features": 8192},
    });
    let mut metadata = serde_json::to_value(&data.metadata).unwrap();
    let object = metadata.as_object_mut().unwrap();
    object.insert("version".into(), serde_json::json!(1));
    object.remove("sharpness");
    object.remove("gradient_structure_tensor_xx_yy_xy");

    let mut zip = zip::ZipWriter::new(std::fs::File::create(path).unwrap());
    let mut digests = sfmtool_archive_io::SectionDigests::new();
    let tool_bytes =
        write_json_entry(&mut zip, "feature_tool_metadata.json.zst", &feature_tool, 3).unwrap();
    let metadata_bytes = write_json_entry(&mut zip, "metadata.json.zst", &metadata, 3).unwrap();
    digests.push_bytes(&tool_bytes);
    digests.push_bytes(&metadata_bytes);
    for (name, bytes) in [
        (positions_entry_name(n), row_major_bytes(&data.positions_xy)),
        (
            affine_shapes_entry_name(n),
            row_major_bytes(&data.affine_shapes),
        ),
        (
            descriptors_entry_name(n),
            row_major_bytes(&data.descriptors),
        ),
        (
            thumbnail_entry_name(),
            row_major_bytes(&data.thumbnail_y_x_rgb),
        ),
    ] {
        write_binary_entry(&mut zip, &name, &bytes, 3).unwrap();
        digests.push_bytes(&bytes);
    }
    let content_hash = SiftContentHash {
        metadata_xxh128: sfmtool_archive_io::format_hash(xxhash_rust::xxh3::xxh3_128(
            &metadata_bytes,
        )),
        feature_tool_xxh128: sfmtool_archive_io::format_hash(xxhash_rust::xxh3::xxh3_128(
            &tool_bytes,
        )),
        content_xxh128: sfmtool_archive_io::format_hash(digests.finish()),
    };
    write_json_entry(&mut zip, "content_hash.json.zst", &content_hash, 3).unwrap();
    zip.finish().unwrap();
}

/// Every reader, `read_sift_metadata` and `verify_sift` refuse a version 1
/// file, which a release wrote, with the message that names the file, its
/// version, the oldest version read and the last release that reads it.
#[test]
fn test_readers_refuse_version_1_naming_the_last_release_that_reads_it() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().to_path_buf();
    let path = dir.join("image.jpg.sift");
    write_version_1_file(&path);
    let expected = format!(
        "{} is a version 1 .sift file, and this sfmtool reads .sift files of version 2 or \
         later. sfmtool 0.4.1 is the last release that reads version 1: install it, run `sfm \
         ws upgrade` on the workspace, then use the file with this release.",
        path.display()
    );
    for result in [
        read_sift(&path).err(),
        read_sift_partial(&path, 2).err(),
        read_sift_positions(&path, 2).err(),
        read_sift_keypoints(&path, 2).err(),
        read_sift_features(&path).err(),
        read_sift_positions_descriptors(&path).err(),
        read_sift_thumbnail(&path).err(),
        read_sift_metadata(&path).err(),
    ] {
        let err = result.expect("a version 1 file must be refused");
        assert!(
            matches!(err, SiftError::VersionNoLongerRead { version: 1, .. }),
            "{err:?}"
        );
        assert_eq!(err.to_string(), expected);
    }
    // The file is sound, so the version is the one error verification reports.
    assert_eq!(verify_sift(&path).unwrap(), (false, vec![expected]));
}

#[test]
fn test_write_rejects_unsupported_versions() {
    for version in [0, 1, SIFT_FORMAT_VERSION + 1] {
        let mut data = make_test_data();
        data.metadata.version = version;
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().to_path_buf();
        let path = dir.join("test.sift");

        let err = write_sift(&path, &data, 3).expect_err("unsupported version must be rejected");
        assert!(err.to_string().contains("unsupported .sift format version"));
        assert!(!path.exists(), "a rejected write must leave no file");
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
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().to_path_buf();
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

    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().to_path_buf();
    let path = dir.join("test.sift");
    assert!(write_sift(&path, &data, 3).is_err());
    assert!(!path.exists(), "a rejected write must leave no file");
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
/// `metadata.json` taken as given, for the readers that stop after them, in a
/// new temporary directory whose name starts with `name`. Returns the
/// directory, removed when it drops, and the file's path.
fn write_raw_metadata_archive(
    name: &str,
    metadata: &serde_json::Value,
) -> (tempfile::TempDir, std::path::PathBuf) {
    let dir = tempfile::Builder::new().prefix(name).tempdir().unwrap();
    let path = dir.path().join("test.sift");
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
    (dir, path)
}

/// The format asks a reader to ignore a metadata key it does not know, so a
/// later version can add one; only a writer is strict.
#[test]
fn test_read_ignores_unknown_metadata_keys() {
    let mut metadata = valid_metadata_json();
    metadata["added_in_a_later_version"] = serde_json::json!(true);
    let (_dir, path) = write_raw_metadata_archive("sift_test_read_extra_key", &metadata);
    let (_, read, _) = read_sift_metadata(&path).unwrap();
    assert_eq!(read.image_name, "test_image.jpg");
}

#[test]
fn test_version_2_round_trip_keeps_readings_exactly() {
    let data = make_test_data();
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().to_path_buf();
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
}

#[test]
fn test_validate_version_rules() {
    let invalid = |data: &SiftData| validate_sift_data(data).unwrap_err().to_string();

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
        data.metadata.sharpness = sharpness;
        assert!(invalid(&data).contains("sharpness must be finite"));
    }
    for tensor in [
        [f64::NAN, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, f64::INFINITY],
    ] {
        let mut data = make_test_data();
        data.metadata.gradient_structure_tensor_xx_yy_xy = tensor;
        assert!(invalid(&data).contains("gradient_structure_tensor_xx_yy_xy must be finite"));
    }
    // A negative cross term is an ordinary value.
    let mut data = make_test_data();
    data.metadata.gradient_structure_tensor_xx_yy_xy = [1.0, 1.0, -0.5];
    validate_sift_data(&data).unwrap();

    // Only the top level of feature_options records the conversion.
    let mut data = make_test_data();
    data.feature_tool_metadata.feature_options =
        serde_json::json!({"source_feature_options": {"image_to_gray": {"formula": "G"}}});
    assert!(invalid(&data).contains("must carry 'image_to_gray'"));
}

#[test]
fn test_read_rejects_version_2_without_readings() {
    for reading in ["sharpness", "gradient_structure_tensor_xx_yy_xy"] {
        let mut metadata = valid_metadata_json();
        metadata.as_object_mut().unwrap().remove(reading);
        let (_dir, path) = write_raw_metadata_archive("sift_test_v2_no_readings", &metadata);
        let message = read_sift_metadata(&path).err().unwrap().to_string();
        assert!(
            message.contains(&format!("missing field `{reading}`")),
            "{message}"
        );
    }
}

#[test]
fn test_verify_reports_version_2_without_image_to_gray() {
    let mut data = make_test_data();
    data.feature_tool_metadata.feature_options = serde_json::json!({});
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().to_path_buf();
    let path = dir.join("test.sift");
    crate::write::write_sift_any_version(&path, &data, 3).unwrap();
    let (ok, errors) = verify_sift(&path).unwrap();
    assert!(!ok);
    assert!(
        errors[0].contains("must carry 'image_to_gray'"),
        "{errors:?}"
    );
}

/// `array`'s shape and values in column-major (Fortran) order.
fn fortran_order<A: Clone, D: ndarray::Dimension>(
    array: &ndarray::Array<A, D>,
) -> ndarray::Array<A, D> {
    use ndarray::ShapeBuilder;
    let values: Vec<A> = array.t().iter().cloned().collect();
    let out = ndarray::Array::from_shape_vec(array.raw_dim().f(), values).unwrap();
    assert!(!out.is_standard_layout());
    out
}

/// `array`'s shape and values as every other row of a buffer twice as long.
fn strided<A: Clone, D: ndarray::Dimension>(array: &ndarray::Array<A, D>) -> ndarray::Array<A, D> {
    let mut dim = array.raw_dim();
    dim[0] *= 2;
    let mut out = ndarray::Array::from_elem(dim, array.first().unwrap().clone());
    out.slice_axis_inplace(ndarray::Axis(0), ndarray::Slice::new(0, None, 2));
    out.assign(array);
    assert!(!out.is_standard_layout());
    out
}

#[test]
fn test_write_arrays_out_of_standard_layout() {
    // A caller's arrays in Fortran order or strided are written in row-major
    // order, the same bytes a standard-layout copy gives, rather than
    // panicking.
    let standard = make_test_data();
    let mut altered = make_test_data();
    altered.positions_xy = fortran_order(&altered.positions_xy);
    altered.affine_shapes = fortran_order(&altered.affine_shapes);
    altered.descriptors = strided(&altered.descriptors);
    altered.thumbnail_y_x_rgb = fortran_order(&altered.thumbnail_y_x_rgb);

    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().to_path_buf();
    let (path_a, path_b) = (dir.join("standard.sift"), dir.join("altered.sift"));
    write_sift(&path_a, &standard, 3).unwrap();
    write_sift(&path_b, &altered, 3).unwrap();

    let loaded_a = read_sift(&path_a).unwrap();
    let loaded_b = read_sift(&path_b).unwrap();
    assert_eq!(
        loaded_b.content_hash.content_xxh128,
        loaded_a.content_hash.content_xxh128
    );
    assert_eq!(loaded_b.positions_xy, standard.positions_xy);
    assert_eq!(loaded_b.affine_shapes, standard.affine_shapes);
    assert_eq!(loaded_b.descriptors, standard.descriptors);
    assert_eq!(loaded_b.thumbnail_y_x_rgb, standard.thumbnail_y_x_rgb);
    let (valid, errors) = verify_sift(&path_b).unwrap();
    assert!(valid, "Verification failed: {errors:?}");
}
