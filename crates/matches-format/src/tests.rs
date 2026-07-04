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
            version: MATCHES_FORMAT_VERSION,
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
            // Non-trivial pose for pair 0 so convention tests can observe
            // the S-conjugation; identity for pair 1.
            q[[0, 0]] = 0.9;
            q[[0, 1]] = 0.1;
            q[[0, 2]] = 0.2;
            q[[0, 3]] = 0.3;
            q[[1, 0]] = 1.0;
            q
        },
        translations_xyz: {
            let mut t = Array2::zeros((pair_count, 3));
            t[[0, 0]] = 1.0;
            t[[0, 1]] = 0.5;
            t[[0, 2]] = 0.25;
            t
        },
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
            version: MATCHES_FORMAT_VERSION,
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

/// Copy a written `.matches` archive, rewriting `metadata.json.zst` so its
/// `version` field reads `version`, and recomputing the stored hashes so the
/// result is an internally consistent file of that version — for authoring
/// old- or future-version fixture bytes.
fn rewrite_matches_version(src: &std::path::Path, dst: &std::path::Path, version: u32) {
    use std::io::{Read, Write};

    use crate::archive_io::format_hash;

    let archive_file = std::fs::File::open(src).unwrap();
    let mut archive = zip::ZipArchive::new(archive_file).unwrap();
    let names: Vec<String> = archive.file_names().map(|s| s.to_string()).collect();

    // Author the new metadata JSON and its hash (hashes cover the
    // uncompressed JSON bytes).
    let mut meta_compressed = Vec::new();
    archive
        .by_name("metadata.json.zst")
        .unwrap()
        .read_to_end(&mut meta_compressed)
        .unwrap();
    let mut meta_json: serde_json::Value =
        serde_json::from_slice(&zstd::stream::decode_all(&meta_compressed[..]).unwrap()).unwrap();
    meta_json
        .as_object_mut()
        .unwrap()
        .insert("version".into(), serde_json::json!(version));
    let meta_bytes = serde_json::to_vec(&meta_json).unwrap();
    let meta_hash = xxhash_rust::xxh3::xxh3_128(&meta_bytes);

    // Rebuild the content hash from the stored per-section digests with the
    // metadata digest replaced (writer order: metadata, images, pairs, tvg).
    let mut hash_compressed = Vec::new();
    archive
        .by_name("content_hash.json.zst")
        .unwrap()
        .read_to_end(&mut hash_compressed)
        .unwrap();
    let mut stored_hashes: MatchesContentHash =
        serde_json::from_slice(&zstd::stream::decode_all(&hash_compressed[..]).unwrap()).unwrap();
    let parse = |hex: &str| u128::from_str_radix(hex, 16).unwrap();
    let mut digests = vec![meta_hash, parse(&stored_hashes.images_xxh128)];
    digests.push(parse(&stored_hashes.image_pairs_xxh128));
    if let Some(tvg) = &stored_hashes.two_view_geometries_xxh128 {
        digests.push(parse(tvg));
    }
    let all_digests_bytes: Vec<u8> = digests.iter().flat_map(|d| d.to_be_bytes()).collect();
    stored_hashes.metadata_xxh128 = format_hash(meta_hash);
    stored_hashes.content_xxh128 = format_hash(xxhash_rust::xxh3::xxh3_128(&all_digests_bytes));

    let out = std::fs::File::create(dst).unwrap();
    let mut zip_out = zip::ZipWriter::new(out);
    let stored =
        zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored);
    for name in &names {
        let mut compressed = Vec::new();
        archive
            .by_name(name)
            .unwrap()
            .read_to_end(&mut compressed)
            .unwrap();
        zip_out.start_file(name, stored).unwrap();
        if name == "metadata.json.zst" {
            let bytes = zstd::bulk::compress(&meta_bytes, 3).unwrap();
            zip_out.write_all(&bytes).unwrap();
        } else if name == "content_hash.json.zst" {
            let bytes =
                zstd::bulk::compress(&serde_json::to_vec(&stored_hashes).unwrap(), 3).unwrap();
            zip_out.write_all(&bytes).unwrap();
        } else {
            zip_out.write_all(&compressed).unwrap();
        }
    }
    zip_out.finish().unwrap();
}

#[test]
fn test_writer_always_writes_current_version() {
    let mut data = make_test_data();
    data.metadata.version = 1; // stale caller-supplied version is overridden
    let dir = std::env::temp_dir().join("matches_test_write_version");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("test.matches");

    write_matches(&path, &data, 3).unwrap();
    let metadata = read_matches_metadata(&path).unwrap();
    assert_eq!(metadata.version, MATCHES_FORMAT_VERSION);

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn test_version_1_relative_poses_upgrade_on_load() {
    let data = make_test_data_with_tvg();
    let dir = std::env::temp_dir().join("matches_test_v1_upgrade");
    std::fs::create_dir_all(&dir).unwrap();
    let v2_path = dir.join("v2.matches");
    let v1_path = dir.join("v1.matches");

    write_matches(&v2_path, &data, 3).unwrap();
    rewrite_matches_version(&v2_path, &v1_path, 1);
    assert_eq!(read_matches_metadata(&v1_path).unwrap().version, 1);

    // Hashes cover the stored bytes: the v1 file verifies as written,
    // before any in-memory conversion.
    let (valid, errors) = verify_matches(&v1_path).unwrap();
    assert!(valid, "v1 fixture failed verification: {errors:?}");

    let loaded = read_matches(&v1_path).unwrap();
    assert_eq!(loaded.metadata.version, MATCHES_FORMAT_VERSION);
    let tvg = loaded.two_view_geometries.as_ref().unwrap();
    let orig = data.two_view_geometries.as_ref().unwrap();

    // Relative poses are S-conjugated: w/x keep, y/z negate; same for t.
    for k in 0..tvg.quaternions_wxyz.nrows() {
        assert_eq!(tvg.quaternions_wxyz[[k, 0]], orig.quaternions_wxyz[[k, 0]]);
        assert_eq!(tvg.quaternions_wxyz[[k, 1]], orig.quaternions_wxyz[[k, 1]]);
        assert_eq!(tvg.quaternions_wxyz[[k, 2]], -orig.quaternions_wxyz[[k, 2]]);
        assert_eq!(tvg.quaternions_wxyz[[k, 3]], -orig.quaternions_wxyz[[k, 3]]);
        assert_eq!(tvg.translations_xyz[[k, 0]], orig.translations_xyz[[k, 0]]);
        assert_eq!(tvg.translations_xyz[[k, 1]], -orig.translations_xyz[[k, 1]]);
        assert_eq!(tvg.translations_xyz[[k, 2]], -orig.translations_xyz[[k, 2]]);
    }

    // Pixel-space F/E/H matrices are untouched by the upgrade.
    assert_eq!(tvg.f_matrices, orig.f_matrices);
    assert_eq!(tvg.e_matrices, orig.e_matrices);
    assert_eq!(tvg.h_matrices, orig.h_matrices);

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn test_unsupported_future_version_rejected() {
    let data = make_test_data();
    let dir = std::env::temp_dir().join("matches_test_future_version");
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("current.matches");
    let dst = dir.join("future.matches");

    write_matches(&src, &data, 3).unwrap();
    let future_version = MATCHES_FORMAT_VERSION + 1;
    rewrite_matches_version(&src, &dst, future_version);

    let expected = format!("unsupported .matches format version {future_version}");
    let err = read_matches(&dst).err().unwrap();
    assert!(format!("{err}").contains(&expected), "{err}");
    let (valid, errors) = verify_matches(&dst).unwrap();
    assert!(
        !valid && errors.iter().any(|e| e.contains(&expected)),
        "{errors:?}"
    );

    std::fs::remove_dir_all(&dir).unwrap();
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
    assert!(!archive.is_empty(), "expected a populated archive");
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
