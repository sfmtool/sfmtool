// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! `.sift` file format reading, writing, and verification.
//!
//! The `.sift` format stores SIFT feature descriptors extracted from images.
//! It is a ZIP archive with zstandard-compressed JSON metadata and binary
//! feature arrays (positions_xy, affine_shapes, descriptors, thumbnail_y_x_rgb).
//!
//! See `docs/sift-file-format.md` for the specification.

#[cfg(not(target_endian = "little"))]
compile_error!(
    "sift-format requires a little-endian target (binary arrays are stored as little-endian)"
);

pub(crate) mod archive_io;
mod read;
mod types;
mod verify;
mod write;

pub use read::{read_sift, read_sift_metadata, read_sift_partial, read_sift_positions};
pub use types::*;
pub use verify::verify_sift;
pub use write::write_sift;

#[cfg(test)]
mod tests {
    use crate::*;
    use ndarray::{Array2, Array3};

    /// Create minimal valid SiftData for testing.
    fn make_test_data() -> SiftData {
        let feature_count: u32 = 4;
        let n = feature_count as usize;

        SiftData {
            feature_tool_metadata: FeatureToolMetadata {
                feature_tool: "colmap".into(),
                feature_type: "sift".into(),
                feature_options: serde_json::json!({"max_num_features": 8192}),
            },
            metadata: SiftMetadata {
                version: 1,
                image_name: "test_image.jpg".into(),
                image_file_xxh128: "abcdef0123456789abcdef0123456789".into(),
                image_file_size: 12345,
                image_width: 640,
                image_height: 480,
                feature_count,
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
    fn test_content_hash_populated() {
        let data = make_test_data();
        let dir = std::env::temp_dir().join("sift_test_hash");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.sift");

        write_sift(&path, &data, 3).unwrap();
        let loaded = read_sift(&path).unwrap();

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
}