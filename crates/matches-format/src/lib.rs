// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! `.matches` file format reading, writing, and verification.
//!
//! The `.matches` format stores feature match correspondences between images.
//! It is a ZIP archive with zstandard-compressed JSON metadata and binary
//! arrays for match data, with an optional two-view geometries section.
//!
//! See `docs/matches-file-format.md` for the specification.

#[cfg(not(target_endian = "little"))]
compile_error!(
    "matches-format requires a little-endian target (binary arrays are stored as little-endian)"
);

pub(crate) mod archive_io;
mod read;
mod types;
mod verify;
mod write;

pub use read::{read_matches, read_matches_metadata};
pub use types::*;
pub use verify::verify_matches;
pub use write::write_matches;

#[cfg(test)]
mod tests {
    use crate::*;
    use ndarray::{Array1, Array2, Array3};
    use std::collections::HashMap;

    /// Create minimal valid MatchesData without TVGs for testing.
    fn make_test_data() -> MatchesData {
        // 3 images, 2 pairs, 5 total matches
        let image_count = 3u32;
        let pair_count = 2usize;
        let match_count = 5usize;

        MatchesData {
            metadata: MatchesMetadata {
                version: 1,
                matching_method: "sequential".into(),
                matching_tool: "colmap".into(),
                matching_tool_version: "4.02".into(),
                matching_options: {
                    let mut m = HashMap::new();
                    m.insert("overlap".into(), serde_json::json!(10));
                    m
                },
                workspace: WorkspaceMetadata {
                    absolute_path: "/tmp/workspace".into(),
                    relative_path: "..".into(),
                    contents: WorkspaceContents {
                        feature_tool: "colmap".into(),
                        feature_type: "sift".into(),
                        feature_options: serde_json::json!({}),
                        feature_prefix_dir: "features/sift-colmap-abc123".into(),
                    },
                },
                timestamp: "2026-03-29T10:00:00Z".into(),
                image_count,
                image_pair_count: pair_count as u32,
                match_count: match_count as u32,
                has_two_view_geometries: false,
            },
            content_hash: MatchesContentHash {
                metadata_xxh128: String::new(),
                images_xxh128: String::new(),
                image_pairs_xxh128: String::new(),
                two_view_geometries_xxh128: None,
                content_xxh128: String::new(),
            },
            image_names: vec![
                "frames/frame_000.jpg".into(),
                "frames/frame_001.jpg".into(),
                "frames/frame_002.jpg".into(),
            ],
            feature_tool_hashes: vec![[0u8; 16]; image_count as usize],
            sift_content_hashes: vec![[1u8; 16]; image_count as usize],
            feature_counts: Array1::from_vec(vec![100, 150, 200]),
            // Pair (0,1) has 3 matches, pair (0,2) has 2 matches
            image_index_pairs: Array2::from_shape_vec((pair_count, 2), vec![0, 1, 0, 2]).unwrap(),
            match_counts: Array1::from_vec(vec![3, 2]),
            match_feature_indexes: Array2::from_shape_vec(
                (match_count, 2),
                vec![
                    0, 0, // pair (0,1) match 0
                    1, 1, // pair (0,1) match 1
                    2, 3, // pair (0,1) match 2
                    5, 10, // pair (0,2) match 0
                    10, 50, // pair (0,2) match 1
                ],
            )
            .unwrap(),
            match_descriptor_distances: Array1::from_vec(vec![100.0, 120.0, 90.0, 200.0, 180.0]),
            two_view_geometries: None,
        }
    }

    /// Create test data with two-view geometries.
    fn make_test_data_with_tvg() -> MatchesData {
        let mut data = make_test_data();
        data.metadata.has_two_view_geometries = true;

        let pair_count = data.metadata.image_pair_count as usize;
        // 2 inliers from pair 0, 1 inlier from pair 1
        let inlier_count = 3usize;

        data.two_view_geometries = Some(TwoViewGeometryData {
            metadata: TvgMetadata {
                image_pair_count: pair_count as u32,
                inlier_count: inlier_count as u32,
                verification_tool: "colmap".into(),
                verification_options: {
                    let mut m = HashMap::new();
                    m.insert("min_num_inliers".into(), serde_json::json!(15));
                    m.insert("max_error".into(), serde_json::json!(4.0));
                    m
                },
            },
            config_types: vec![
                TwoViewGeometryConfig::Calibrated,
                TwoViewGeometryConfig::Degenerate,
            ],
            config_indexes: Array1::from_vec(vec![0, 1]), // pair 0: calibrated, pair 1: degenerate
            inlier_counts: Array1::from_vec(vec![2, 1]),
            inlier_feature_indexes: Array2::from_shape_vec(
                (inlier_count, 2),
                vec![
                    0, 0, // pair 0 inlier 0 (subset of candidate match 0)
                    1, 1, // pair 0 inlier 1 (subset of candidate match 1)
                    5, 10, // pair 1 inlier 0 (subset of candidate match 3)
                ],
            )
            .unwrap(),
            f_matrices: Array3::zeros((pair_count, 3, 3)),
            e_matrices: {
                let mut e = Array3::zeros((pair_count, 3, 3));
                // Set a non-trivial essential matrix for pair 0
                e[[0, 0, 1]] = -0.1;
                e[[0, 1, 0]] = 0.1;
                e[[0, 2, 2]] = 0.5;
                e
            },
            h_matrices: Array3::zeros((pair_count, 3, 3)),
            quaternions_wxyz: {
                let mut q = Array2::zeros((pair_count, 4));
                q[[0, 0]] = 1.0; // identity for pair 0
                q[[1, 0]] = 1.0; // identity for pair 1
                q
            },
            translations_xyz: Array2::zeros((pair_count, 3)),
        });

        data
    }

    #[test]
    fn test_round_trip_no_tvg() {
        let data = make_test_data();
        let dir = std::env::temp_dir().join("matches_test_round_trip");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.matches");

        write_matches(&path, &data, 3).unwrap();
        let loaded = read_matches(&path).unwrap();

        // Verify metadata
        assert_eq!(loaded.metadata.matching_method, "sequential");
        assert_eq!(loaded.metadata.matching_tool, "colmap");
        assert_eq!(loaded.metadata.matching_tool_version, "4.02");
        assert_eq!(loaded.metadata.image_count, 3);
        assert_eq!(loaded.metadata.image_pair_count, 2);
        assert_eq!(loaded.metadata.match_count, 5);
        assert!(!loaded.metadata.has_two_view_geometries);

        // Verify images
        assert_eq!(loaded.image_names, data.image_names);
        assert_eq!(loaded.feature_tool_hashes, data.feature_tool_hashes);
        assert_eq!(loaded.sift_content_hashes, data.sift_content_hashes);
        assert_eq!(loaded.feature_counts, data.feature_counts);

        // Verify pairs
        assert_eq!(loaded.image_index_pairs, data.image_index_pairs);
        assert_eq!(loaded.match_counts, data.match_counts);
        assert_eq!(loaded.match_feature_indexes, data.match_feature_indexes);
        assert_eq!(
            loaded.match_descriptor_distances,
            data.match_descriptor_distances
        );

        // No TVG
        assert!(loaded.two_view_geometries.is_none());

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_round_trip_with_tvg() {
        let data = make_test_data_with_tvg();
        let dir = std::env::temp_dir().join("matches_test_tvg_round_trip");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.matches");

        write_matches(&path, &data, 3).unwrap();
        let loaded = read_matches(&path).unwrap();

        assert!(loaded.metadata.has_two_view_geometries);
        let tvg = loaded.two_view_geometries.as_ref().unwrap();
        let orig_tvg = data.two_view_geometries.as_ref().unwrap();

        assert_eq!(tvg.metadata.inlier_count, 3);
        assert_eq!(tvg.config_types, orig_tvg.config_types);
        assert_eq!(tvg.config_indexes, orig_tvg.config_indexes);
        assert_eq!(tvg.inlier_counts, orig_tvg.inlier_counts);
        assert_eq!(tvg.inlier_feature_indexes, orig_tvg.inlier_feature_indexes);
        assert_eq!(tvg.f_matrices, orig_tvg.f_matrices);
        assert_eq!(tvg.e_matrices, orig_tvg.e_matrices);
        assert_eq!(tvg.h_matrices, orig_tvg.h_matrices);
        assert_eq!(tvg.quaternions_wxyz, orig_tvg.quaternions_wxyz);
        assert_eq!(tvg.translations_xyz, orig_tvg.translations_xyz);

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_verify_no_tvg() {
        let data = make_test_data();
        let dir = std::env::temp_dir().join("matches_test_verify");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.matches");

        write_matches(&path, &data, 3).unwrap();

        let (valid, errors) = verify_matches(&path).unwrap();
        assert!(valid, "Verification failed: {:?}", errors);
        assert!(errors.is_empty());

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_verify_with_tvg() {
        let data = make_test_data_with_tvg();
        let dir = std::env::temp_dir().join("matches_test_verify_tvg");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.matches");

        write_matches(&path, &data, 3).unwrap();

        let (valid, errors) = verify_matches(&path).unwrap();
        assert!(valid, "Verification failed: {:?}", errors);
        assert!(errors.is_empty());

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_read_metadata_only() {
        let data = make_test_data();
        let dir = std::env::temp_dir().join("matches_test_metadata");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.matches");

        write_matches(&path, &data, 3).unwrap();

        let metadata = read_matches_metadata(&path).unwrap();
        assert_eq!(metadata.matching_method, "sequential");
        assert_eq!(metadata.image_count, 3);
        assert_eq!(metadata.match_count, 5);
        assert!(!metadata.has_two_view_geometries);

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_content_hash_populated() {
        let data = make_test_data();
        let dir = std::env::temp_dir().join("matches_test_hash");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.matches");

        write_matches(&path, &data, 3).unwrap();
        let loaded = read_matches(&path).unwrap();

        assert_eq!(loaded.content_hash.metadata_xxh128.len(), 32);
        assert_eq!(loaded.content_hash.images_xxh128.len(), 32);
        assert_eq!(loaded.content_hash.image_pairs_xxh128.len(), 32);
        assert_eq!(loaded.content_hash.content_xxh128.len(), 32);
        assert!(loaded.content_hash.two_view_geometries_xxh128.is_none());

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_content_hash_with_tvg() {
        let data = make_test_data_with_tvg();
        let dir = std::env::temp_dir().join("matches_test_hash_tvg");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.matches");

        write_matches(&path, &data, 3).unwrap();
        let loaded = read_matches(&path).unwrap();

        assert!(loaded.content_hash.two_view_geometries_xxh128.is_some());
        assert_eq!(
            loaded
                .content_hash
                .two_view_geometries_xxh128
                .unwrap()
                .len(),
            32
        );

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_empty_matches() {
        let data = MatchesData {
            metadata: MatchesMetadata {
                version: 1,
                matching_method: "exhaustive".into(),
                matching_tool: "colmap".into(),
                matching_tool_version: "4.02".into(),
                matching_options: HashMap::new(),
                workspace: WorkspaceMetadata {
                    absolute_path: "/tmp/workspace".into(),
                    relative_path: "..".into(),
                    contents: WorkspaceContents {
                        feature_tool: "colmap".into(),
                        feature_type: "sift".into(),
                        feature_options: serde_json::json!({}),
                        feature_prefix_dir: String::new(),
                    },
                },
                timestamp: "2026-03-29T10:00:00Z".into(),
                image_count: 0,
                image_pair_count: 0,
                match_count: 0,
                has_two_view_geometries: false,
            },
            content_hash: MatchesContentHash {
                metadata_xxh128: String::new(),
                images_xxh128: String::new(),
                image_pairs_xxh128: String::new(),
                two_view_geometries_xxh128: None,
                content_xxh128: String::new(),
            },
            image_names: vec![],
            feature_tool_hashes: vec![],
            sift_content_hashes: vec![],
            feature_counts: Array1::from_vec(vec![]),
            image_index_pairs: Array2::zeros((0, 2)),
            match_counts: Array1::from_vec(vec![]),
            match_feature_indexes: Array2::zeros((0, 2)),
            match_descriptor_distances: Array1::from_vec(vec![]),
            two_view_geometries: None,
        };

        let dir = std::env::temp_dir().join("matches_test_empty");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.matches");

        write_matches(&path, &data, 3).unwrap();
        let loaded = read_matches(&path).unwrap();

        assert_eq!(loaded.metadata.image_count, 0);
        assert_eq!(loaded.metadata.match_count, 0);
        assert_eq!(loaded.image_names.len(), 0);

        let (valid, errors) = verify_matches(&path).unwrap();
        assert!(valid, "Empty matches verification failed: {:?}", errors);

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_write_validation_unsorted_pairs() {
        let mut data = make_test_data();
        // Swap pair order to make them unsorted
        data.image_index_pairs = Array2::from_shape_vec(
            (2, 2),
            vec![0, 2, 0, 1], // (0,2) before (0,1) — not sorted
        )
        .unwrap();

        let dir = std::env::temp_dir().join("matches_test_unsorted");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.matches");

        let result = write_matches(&path, &data, 3);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("not sorted"),
            "Error should mention sorting, got: {msg}"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_write_validation_idx_i_not_less_than_idx_j() {
        let mut data = make_test_data();
        // Make idx_i == idx_j
        data.image_index_pairs = Array2::from_shape_vec(
            (2, 2),
            vec![0, 0, 0, 2], // (0,0) is invalid
        )
        .unwrap();

        let dir = std::env::temp_dir().join("matches_test_bad_pair");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.matches");

        let result = write_matches(&path, &data, 3);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("idx_i must be < idx_j"),
            "Error should mention idx_i < idx_j, got: {msg}"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_write_validation_feature_index_out_of_bounds() {
        let mut data = make_test_data();
        // Set a feature index beyond feature_counts
        data.match_feature_indexes[[0, 0]] = 999; // feature_counts[0] = 100

        let dir = std::env::temp_dir().join("matches_test_oob_feat");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.matches");

        let result = write_matches(&path, &data, 3);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("999") && msg.contains("feature_counts"),
            "Error should mention feature index out of bounds, got: {msg}"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_write_validation_inlier_not_subset() {
        let mut data = make_test_data_with_tvg();
        // Make an inlier that isn't in the candidate matches
        let tvg = data.two_view_geometries.as_mut().unwrap();
        tvg.inlier_feature_indexes[[0, 0]] = 99; // (99, 0) is not a candidate match

        let dir = std::env::temp_dir().join("matches_test_inlier_subset");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.matches");

        let result = write_matches(&path, &data, 3);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("not in candidate matches"),
            "Error should mention inlier subset constraint, got: {msg}"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_read_nonexistent_file() {
        let result = read_matches(std::path::Path::new("nonexistent.matches"));
        match result {
            Err(err) => {
                let msg = err.to_string();
                assert!(
                    msg.contains("nonexistent.matches"),
                    "Error message should contain the file path, got: {msg}"
                );
            }
            Ok(_) => panic!("Expected error for nonexistent file"),
        }
    }

    #[test]
    fn test_two_view_geometry_config_round_trip() {
        let configs = [
            TwoViewGeometryConfig::Undefined,
            TwoViewGeometryConfig::Degenerate,
            TwoViewGeometryConfig::Calibrated,
            TwoViewGeometryConfig::Uncalibrated,
            TwoViewGeometryConfig::Planar,
            TwoViewGeometryConfig::PlanarOrPanoramic,
            TwoViewGeometryConfig::Panoramic,
            TwoViewGeometryConfig::Multiple,
            TwoViewGeometryConfig::WatermarkClean,
            TwoViewGeometryConfig::WatermarkBad,
        ];
        for config in &configs {
            let s = config.as_str();
            let parsed = s.parse::<TwoViewGeometryConfig>().unwrap();
            assert_eq!(*config, parsed, "Round-trip failed for {s}");
        }
    }

    #[test]
    fn test_invalid_config_string() {
        let result = "bogus".parse::<TwoViewGeometryConfig>();
        assert!(result.is_err());
    }

    // Every entry in a .matches archive MUST use ZIP's STORE method. Entries are already
    // zstandard-compressed; applying ZIP-level DEFLATE would double-compress and break
    // the spec guarantee of random access via simple seek.
    #[test]
    fn test_archive_uses_stored_compression() {
        let data = make_test_data_with_tvg();
        let dir = std::env::temp_dir().join("matches_test_stored_compression");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.matches");

        write_matches(&path, &data, 3).unwrap();

        let file = std::fs::File::open(&path).unwrap();
        let mut archive = zip::ZipArchive::new(file).unwrap();
        assert!(archive.len() > 0, "expected a populated archive");
        for i in 0..archive.len() {
            let entry = archive.by_index(i).unwrap();
            assert_eq!(
                entry.compression(),
                zip::CompressionMethod::Stored,
                "entry '{}' uses {:?}, expected Stored",
                entry.name(),
                entry.compression(),
            );
        }

        std::fs::remove_dir_all(&dir).unwrap();
    }
}
