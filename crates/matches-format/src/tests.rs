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
            image_pair_count: Some(pair_count as u32),
            match_count: Some(match_count as u32),
            cluster_count: None,
            cluster_member_count: None,
            has_two_view_geometries: false,
            has_clusters: false,
            has_cluster_patches: false,
        },
        content_hash: empty_content_hash(),
        image_names: vec![
            "frames/frame_000.jpg".into(),
            "frames/frame_001.jpg".into(),
            "frames/frame_002.jpg".into(),
        ],
        feature_tool_hashes: vec![[0u8; 16]; image_count as usize],
        sift_content_hashes: vec![[1u8; 16]; image_count as usize],
        feature_counts: Array1::from_vec(vec![100, 150, 200]),
        image_dims: Some(
            Array2::from_shape_vec((3, 2), vec![640, 480, 640, 480, 1024, 768]).unwrap(),
        ),
        image_pairs: Some(PairsData {
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
        }),
        clusters: None,
        cluster_patches: None,
        two_view_geometries: None,
    }
}

fn empty_content_hash() -> MatchesContentHash {
    MatchesContentHash {
        metadata_xxh128: String::new(),
        images_xxh128: String::new(),
        image_pairs_xxh128: None,
        clusters_xxh128: None,
        cluster_patches_xxh128: None,
        two_view_geometries_xxh128: None,
        content_xxh128: String::new(),
    }
}

/// Create test data with two-view geometries.
fn make_test_data_with_tvg() -> MatchesData {
    let mut data = make_test_data();
    data.metadata.has_two_view_geometries = true;

    let pair_count = data.metadata.image_pair_count.unwrap() as usize;
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

/// Create cluster-backbone test data: 3 images, 2 clusters, 5 members.
///
/// Cluster 0 = members 0..3 in images (0, 1, 2); cluster 1 = members 3..5
/// in images (0, 2).
fn make_cluster_test_data() -> MatchesData {
    let mut data = make_test_data();
    data.metadata.image_pair_count = None;
    data.metadata.match_count = None;
    data.metadata.cluster_count = Some(2);
    data.metadata.cluster_member_count = Some(5);
    data.metadata.has_clusters = true;
    data.image_pairs = None;
    data.clusters = Some(ClustersData {
        cluster_starts: Array1::from_vec(vec![0, 3, 5]),
        member_images: Array1::from_vec(vec![0, 1, 2, 0, 2]),
        member_features: Array1::from_vec(vec![0, 1, 2, 5, 10]),
        matcher_options: serde_json::json!({
            "d": 8, "alpha": 1.2, "min_size": 2, "preset": "default"
        }),
    });
    data
}

/// Create cluster test data with the cluster_patches enrichment.
///
/// Cluster 0: member 0 is the reference, member 1 kept, member 2 rejected
/// (low ZNCC). Cluster 1: unrefinable (all members not evaluated).
fn make_cluster_patch_test_data() -> MatchesData {
    let mut data = make_cluster_test_data();
    data.metadata.has_cluster_patches = true;
    let mut member_affines = Array3::zeros((5, 2, 3));
    // Reference row: identity | x_ref (the reference keypoint's own absolute
    // position).
    member_affines[[0, 0, 0]] = 1.0;
    member_affines[[0, 0, 2]] = 12.5;
    member_affines[[0, 1, 1]] = 1.0;
    member_affines[[0, 1, 2]] = 20.25;
    // Kept member: a non-trivial affine; the last column is the member's
    // refined absolute keypoint position p = A·x_ref + t.
    member_affines[[1, 0, 0]] = 1.1;
    member_affines[[1, 0, 1]] = -0.05;
    member_affines[[1, 0, 2]] = 42.5;
    member_affines[[1, 1, 0]] = 0.03;
    member_affines[[1, 1, 1]] = 0.95;
    member_affines[[1, 1, 2]] = 17.25;
    // Rejected member: still carries its refined affine + position.
    member_affines[[2, 0, 0]] = 1.0;
    member_affines[[2, 0, 2]] = 13.0;
    member_affines[[2, 1, 1]] = 1.0;
    member_affines[[2, 1, 2]] = 21.0;
    data.cluster_patches = Some(ClusterPatchData {
        reference_members: Array1::from_vec(vec![0, CLUSTER_REFERENCE_UNREFINABLE]),
        member_status: Array1::from_vec(vec![
            ClusterMemberStatus::Reference as u8,
            ClusterMemberStatus::Kept as u8,
            ClusterMemberStatus::RejectedLowZncc as u8,
            ClusterMemberStatus::NotEvaluated as u8,
            ClusterMemberStatus::NotEvaluated as u8,
        ]),
        member_affines,
        member_zncc: Array1::from_vec(vec![1.0, 0.93, 0.41, f32::NAN, f32::NAN]),
        member_shift_px: Array1::from_vec(vec![0.0, 1.25, 0.8, f32::NAN, f32::NAN]),
        member_consistency_residual: Array1::from_vec(vec![0.02, 0.05, 0.31, f32::NAN, f32::NAN]),
        refine_options: serde_json::json!({
            "radius": 4.0, "resolution": 15, "min_zncc": 0.85, "max_shift_px": 3.0
        }),
    });
    data
}

fn write_to_temp(name: &str, data: &MatchesData) -> (std::path::PathBuf, std::path::PathBuf) {
    let dir = std::env::temp_dir().join(name);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("test.matches");
    write_matches(&path, data, 3).unwrap();
    (dir, path)
}

/// Bit-exact f32 comparison (NaN == NaN).
fn assert_f32_bits_eq(actual: &Array1<f32>, expected: &Array1<f32>) {
    assert_eq!(actual.len(), expected.len());
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            e.to_bits(),
            "mismatch at index {i}: {a} vs {e}"
        );
    }
}

#[test]
fn test_round_trip_no_tvg() {
    let data = make_test_data();
    let (dir, path) = write_to_temp("matches_test_round_trip", &data);
    let loaded = read_matches(&path).unwrap();

    // Verify metadata
    assert_eq!(loaded.metadata.matching_method, "sequential");
    assert_eq!(loaded.metadata.matching_tool, "colmap");
    assert_eq!(loaded.metadata.matching_tool_version, "4.02");
    assert_eq!(loaded.metadata.image_count, 3);
    assert_eq!(loaded.metadata.image_pair_count, Some(2));
    assert_eq!(loaded.metadata.match_count, Some(5));
    assert!(!loaded.metadata.has_two_view_geometries);
    assert!(!loaded.metadata.has_clusters);
    assert!(!loaded.metadata.has_cluster_patches);
    assert_eq!(loaded.metadata.cluster_count, None);
    assert_eq!(loaded.metadata.cluster_member_count, None);

    // Verify images
    assert_eq!(loaded.image_names, data.image_names);
    assert_eq!(loaded.feature_tool_hashes, data.feature_tool_hashes);
    assert_eq!(loaded.sift_content_hashes, data.sift_content_hashes);
    assert_eq!(loaded.feature_counts, data.feature_counts);
    assert_eq!(loaded.image_dims, data.image_dims);

    // Verify pairs
    let pairs = loaded.image_pairs.as_ref().unwrap();
    let orig_pairs = data.image_pairs.as_ref().unwrap();
    assert_eq!(pairs.image_index_pairs, orig_pairs.image_index_pairs);
    assert_eq!(pairs.match_counts, orig_pairs.match_counts);
    assert_eq!(
        pairs.match_feature_indexes,
        orig_pairs.match_feature_indexes
    );
    assert_eq!(
        pairs.match_descriptor_distances,
        orig_pairs.match_descriptor_distances
    );

    // No clusters, no TVG
    assert!(loaded.clusters.is_none());
    assert!(loaded.cluster_patches.is_none());
    assert!(loaded.two_view_geometries.is_none());

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn test_round_trip_with_tvg() {
    let data = make_test_data_with_tvg();
    let (dir, path) = write_to_temp("matches_test_tvg_round_trip", &data);
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
fn test_round_trip_clusters_only() {
    let data = make_cluster_test_data();
    let (dir, path) = write_to_temp("matches_test_clusters_round_trip", &data);
    let loaded = read_matches(&path).unwrap();

    assert!(loaded.metadata.has_clusters);
    assert!(!loaded.metadata.has_cluster_patches);
    assert!(!loaded.metadata.has_two_view_geometries);
    assert_eq!(loaded.metadata.cluster_count, Some(2));
    assert_eq!(loaded.metadata.cluster_member_count, Some(5));
    assert_eq!(loaded.metadata.image_pair_count, None);
    assert_eq!(loaded.metadata.match_count, None);

    assert!(loaded.image_pairs.is_none());
    assert!(loaded.cluster_patches.is_none());
    assert!(loaded.two_view_geometries.is_none());

    let clusters = loaded.clusters.as_ref().unwrap();
    let orig = data.clusters.as_ref().unwrap();
    assert_eq!(clusters.cluster_starts, orig.cluster_starts);
    assert_eq!(clusters.member_images, orig.member_images);
    assert_eq!(clusters.member_features, orig.member_features);
    assert_eq!(clusters.matcher_options, orig.matcher_options);

    // Content hash: clusters digest present, pairs digest absent.
    assert!(loaded.content_hash.clusters_xxh128.is_some());
    assert!(loaded.content_hash.image_pairs_xxh128.is_none());
    assert!(loaded.content_hash.cluster_patches_xxh128.is_none());

    let (valid, errors) = verify_matches(&path).unwrap();
    assert!(valid, "Verification failed: {errors:?}");

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn test_round_trip_clusters_with_patches() {
    let data = make_cluster_patch_test_data();
    let (dir, path) = write_to_temp("matches_test_cluster_patches_round_trip", &data);
    let loaded = read_matches(&path).unwrap();

    assert!(loaded.metadata.has_clusters);
    assert!(loaded.metadata.has_cluster_patches);

    let cp = loaded.cluster_patches.as_ref().unwrap();
    let orig = data.cluster_patches.as_ref().unwrap();
    assert_eq!(cp.reference_members, orig.reference_members);
    assert_eq!(cp.member_status, orig.member_status);
    assert_eq!(cp.member_affines, orig.member_affines);
    assert_f32_bits_eq(&cp.member_zncc, &orig.member_zncc);
    assert_f32_bits_eq(&cp.member_shift_px, &orig.member_shift_px);
    assert_f32_bits_eq(
        &cp.member_consistency_residual,
        &orig.member_consistency_residual,
    );
    assert_eq!(cp.refine_options, orig.refine_options);

    assert!(loaded.content_hash.cluster_patches_xxh128.is_some());

    let (valid, errors) = verify_matches(&path).unwrap();
    assert!(valid, "Verification failed: {errors:?}");

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn test_verify_no_tvg() {
    let data = make_test_data();
    let (dir, path) = write_to_temp("matches_test_verify", &data);

    let (valid, errors) = verify_matches(&path).unwrap();
    assert!(valid, "Verification failed: {:?}", errors);
    assert!(errors.is_empty());

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn test_verify_with_tvg() {
    let data = make_test_data_with_tvg();
    let (dir, path) = write_to_temp("matches_test_verify_tvg", &data);

    let (valid, errors) = verify_matches(&path).unwrap();
    assert!(valid, "Verification failed: {:?}", errors);
    assert!(errors.is_empty());

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn test_read_metadata_only() {
    let data = make_test_data();
    let (dir, path) = write_to_temp("matches_test_metadata", &data);

    let metadata = read_matches_metadata(&path).unwrap();
    assert_eq!(metadata.matching_method, "sequential");
    assert_eq!(metadata.image_count, 3);
    assert_eq!(metadata.match_count, Some(5));
    assert!(!metadata.has_two_view_geometries);
    assert!(!metadata.has_clusters);

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn test_content_hash_populated() {
    let data = make_test_data();
    let (dir, path) = write_to_temp("matches_test_hash", &data);
    let loaded = read_matches(&path).unwrap();

    assert_eq!(loaded.content_hash.metadata_xxh128.len(), 32);
    assert_eq!(loaded.content_hash.images_xxh128.len(), 32);
    assert_eq!(loaded.content_hash.image_pairs_xxh128.unwrap().len(), 32);
    assert_eq!(loaded.content_hash.content_xxh128.len(), 32);
    assert!(loaded.content_hash.clusters_xxh128.is_none());
    assert!(loaded.content_hash.cluster_patches_xxh128.is_none());
    assert!(loaded.content_hash.two_view_geometries_xxh128.is_none());

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn test_content_hash_with_tvg() {
    let data = make_test_data_with_tvg();
    let (dir, path) = write_to_temp("matches_test_hash_tvg", &data);
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
    let mut data = make_test_data();
    data.metadata.image_count = 0;
    data.metadata.image_pair_count = Some(0);
    data.metadata.match_count = Some(0);
    data.image_names = vec![];
    data.feature_tool_hashes = vec![];
    data.sift_content_hashes = vec![];
    data.feature_counts = Array1::from_vec(vec![]);
    data.image_dims = Some(Array2::zeros((0, 2)));
    data.image_pairs = Some(PairsData {
        image_index_pairs: Array2::zeros((0, 2)),
        match_counts: Array1::from_vec(vec![]),
        match_feature_indexes: Array2::zeros((0, 2)),
        match_descriptor_distances: Array1::from_vec(vec![]),
    });

    let (dir, path) = write_to_temp("matches_test_empty", &data);
    let loaded = read_matches(&path).unwrap();

    assert_eq!(loaded.metadata.image_count, 0);
    assert_eq!(loaded.metadata.match_count, Some(0));
    assert_eq!(loaded.image_names.len(), 0);

    let (valid, errors) = verify_matches(&path).unwrap();
    assert!(valid, "Empty matches verification failed: {:?}", errors);

    std::fs::remove_dir_all(&dir).unwrap();
}

fn expect_write_error(name: &str, data: &MatchesData, expected_fragment: &str) {
    let dir = std::env::temp_dir().join(name);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("test.matches");

    let result = write_matches(&path, data, 3);
    assert!(
        result.is_err(),
        "expected write to fail: {expected_fragment}"
    );
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains(expected_fragment),
        "Error should mention {expected_fragment:?}, got: {msg}"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_write_validation_unsorted_pairs() {
    let mut data = make_test_data();
    // Swap pair order to make them unsorted
    data.image_pairs.as_mut().unwrap().image_index_pairs = Array2::from_shape_vec(
        (2, 2),
        vec![0, 2, 0, 1], // (0,2) before (0,1) — not sorted
    )
    .unwrap();
    expect_write_error("matches_test_unsorted", &data, "not sorted");
}

#[test]
fn test_write_validation_idx_i_not_less_than_idx_j() {
    let mut data = make_test_data();
    // Make idx_i == idx_j
    data.image_pairs.as_mut().unwrap().image_index_pairs = Array2::from_shape_vec(
        (2, 2),
        vec![0, 0, 0, 2], // (0,0) is invalid
    )
    .unwrap();
    expect_write_error("matches_test_bad_pair", &data, "idx_i must be < idx_j");
}

#[test]
fn test_write_validation_feature_index_out_of_bounds() {
    let mut data = make_test_data();
    // Set a feature index beyond feature_counts
    data.image_pairs.as_mut().unwrap().match_feature_indexes[[0, 0]] = 999; // feature_counts[0] = 100
    expect_write_error("matches_test_oob_feat", &data, "feature_counts");
}

#[test]
fn test_write_validation_inlier_not_subset() {
    let mut data = make_test_data_with_tvg();
    // Make an inlier that isn't in the candidate matches
    let tvg = data.two_view_geometries.as_mut().unwrap();
    tvg.inlier_feature_indexes[[0, 0]] = 99; // (99, 0) is not a candidate match
    expect_write_error(
        "matches_test_inlier_subset",
        &data,
        "not in candidate matches",
    );
}

#[test]
fn test_write_validation_both_backbones() {
    let mut data = make_test_data();
    data.clusters = make_cluster_test_data().clusters;
    expect_write_error(
        "matches_test_both_backbones",
        &data,
        "exactly one of image_pairs / clusters",
    );
}

#[test]
fn test_write_validation_neither_backbone() {
    let mut data = make_test_data();
    data.image_pairs = None;
    expect_write_error(
        "matches_test_neither_backbone",
        &data,
        "exactly one of image_pairs / clusters",
    );
}

#[test]
fn test_write_validation_cluster_patches_require_clusters() {
    let mut data = make_test_data();
    data.cluster_patches = make_cluster_patch_test_data().cluster_patches;
    expect_write_error(
        "matches_test_cp_requires_clusters",
        &data,
        "cluster_patches requires the clusters section",
    );
}

#[test]
fn test_write_validation_tvg_requires_pairs() {
    let mut data = make_cluster_test_data();
    data.two_view_geometries = make_test_data_with_tvg().two_view_geometries;
    expect_write_error(
        "matches_test_tvg_requires_pairs",
        &data,
        "two_view_geometries requires the image_pairs section",
    );
}

#[test]
fn test_write_validation_flag_mismatch() {
    let mut data = make_cluster_test_data();
    data.metadata.has_clusters = false;
    expect_write_error("matches_test_flag_mismatch", &data, "metadata.has_clusters");
}

#[test]
fn test_write_validation_cluster_file_with_pair_counts() {
    let mut data = make_cluster_test_data();
    data.metadata.image_pair_count = Some(2);
    data.metadata.match_count = Some(5);
    expect_write_error(
        "matches_test_cluster_pair_counts",
        &data,
        "must not set metadata.image_pair_count",
    );
}

#[test]
fn test_write_validation_bad_csr() {
    // starts[0] != 0
    let mut data = make_cluster_test_data();
    data.clusters.as_mut().unwrap().cluster_starts = Array1::from_vec(vec![1, 3, 5]);
    expect_write_error("matches_test_csr_start", &data, "cluster_starts[0]");

    // decreasing
    let mut data = make_cluster_test_data();
    data.clusters.as_mut().unwrap().cluster_starts = Array1::from_vec(vec![0, 4, 3]);
    // A 4→3 step is decreasing, but cluster 0 spanning 0..4 then 4..3 first
    // trips the non-decreasing check.
    expect_write_error("matches_test_csr_decreasing", &data, "non-decreasing");

    // final value != member count
    let mut data = make_cluster_test_data();
    data.clusters.as_mut().unwrap().cluster_starts = Array1::from_vec(vec![0, 2, 4]);
    expect_write_error(
        "matches_test_csr_final",
        &data,
        "cluster_starts final value 4 != member count 5",
    );

    // cluster with a single member
    let mut data = make_cluster_test_data();
    data.clusters.as_mut().unwrap().cluster_starts = Array1::from_vec(vec![0, 1, 5]);
    expect_write_error("matches_test_csr_singleton", &data, ">= 2");
}

#[test]
fn test_write_validation_missing_image_dims() {
    let mut data = make_test_data();
    data.image_dims = None;
    expect_write_error("matches_test_missing_dims", &data, "image_dims is required");
}

#[test]
fn test_write_validation_zero_image_dims() {
    let mut data = make_test_data();
    data.image_dims.as_mut().unwrap()[[1, 1]] = 0;
    expect_write_error(
        "matches_test_zero_dims",
        &data,
        "image_dims[1] has a zero height",
    );
}

#[test]
fn test_write_validation_reference_row_not_identity() {
    let mut data = make_cluster_patch_test_data();
    // Member 0 is cluster 0's reference; corrupt its leading 2×2.
    data.cluster_patches.as_mut().unwrap().member_affines[[0, 0, 0]] = 1.1;
    expect_write_error(
        "matches_test_cp_ref_not_identity",
        &data,
        "identity leading 2x2",
    );
}

#[test]
fn test_write_validation_member_feature_out_of_bounds() {
    let mut data = make_cluster_test_data();
    data.clusters.as_mut().unwrap().member_features[1] = 150; // feature_counts[1] = 150
    expect_write_error(
        "matches_test_member_feat_oob",
        &data,
        "member_features[1] = 150 >= feature_counts[1]",
    );
}

#[test]
fn test_write_validation_member_image_out_of_bounds() {
    let mut data = make_cluster_test_data();
    data.clusters.as_mut().unwrap().member_images[0] = 3; // image_count = 3
    expect_write_error(
        "matches_test_member_img_oob",
        &data,
        "member_images[0] = 3 >= image_count",
    );
}

#[test]
fn test_write_validation_cluster_patches_wrong_lengths() {
    let mut data = make_cluster_patch_test_data();
    data.cluster_patches.as_mut().unwrap().reference_members = Array1::from_vec(vec![0]);
    expect_write_error(
        "matches_test_cp_wrong_len",
        &data,
        "reference_members len 1 != cluster_count 2",
    );

    let mut data = make_cluster_patch_test_data();
    data.cluster_patches.as_mut().unwrap().member_zncc = Array1::from_vec(vec![1.0, 0.9, 0.4, 0.0]);
    expect_write_error(
        "matches_test_cp_wrong_zncc_len",
        &data,
        "member_zncc len 4 != cluster_member_count 5",
    );

    let mut data = make_cluster_patch_test_data();
    data.cluster_patches
        .as_mut()
        .unwrap()
        .member_consistency_residual = Array1::from_vec(vec![0.0, 0.1]);
    expect_write_error(
        "matches_test_cp_wrong_consistency_len",
        &data,
        "member_consistency_residual len 2 != cluster_member_count 5",
    );
}

#[test]
fn test_write_validation_invalid_status() {
    let mut data = make_cluster_patch_test_data();
    data.cluster_patches.as_mut().unwrap().member_status[2] = 7;
    expect_write_error(
        "matches_test_cp_bad_status",
        &data,
        "not a valid ClusterMemberStatus",
    );
}

#[test]
fn test_write_validation_reference_outside_cluster() {
    let mut data = make_cluster_patch_test_data();
    // Cluster 0 owns members [0, 3); member 4 is in cluster 1.
    data.cluster_patches.as_mut().unwrap().reference_members[0] = 4;
    expect_write_error(
        "matches_test_cp_ref_range",
        &data,
        "outside cluster member range",
    );
}

#[test]
fn test_write_validation_reference_wrong_status() {
    let mut data = make_cluster_patch_test_data();
    // Member 1 is Kept, not Reference.
    data.cluster_patches.as_mut().unwrap().reference_members[0] = 1;
    expect_write_error(
        "matches_test_cp_ref_status",
        &data,
        "expected 0 (reference)",
    );
}

#[test]
fn test_write_validation_two_kept_members_one_image() {
    let mut data = make_cluster_patch_test_data();
    // Cluster 1 = members 3, 4; put both in image 0 and mark both kept.
    data.clusters.as_mut().unwrap().member_images[4] = 0;
    let cp = data.cluster_patches.as_mut().unwrap();
    cp.member_status[3] = ClusterMemberStatus::Kept as u8;
    cp.member_status[4] = ClusterMemberStatus::Kept as u8;
    expect_write_error(
        "matches_test_cp_dup_kept",
        &data,
        "both reference/kept for image 0",
    );
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

#[test]
fn test_cluster_member_status_round_trip() {
    for value in 0u8..=6 {
        let status = ClusterMemberStatus::from_u8(value).unwrap();
        assert_eq!(status as u8, value);
    }
    assert!(ClusterMemberStatus::from_u8(7).is_none());
    assert!(ClusterMemberStatus::from_u8(255).is_none());
}

// ── Archive-crafting helpers for verify-side rejection tests ───────────────

/// Read every entry of a `.matches` archive as (name, decompressed bytes),
/// excluding `content_hash.json.zst` (rebuilt by [`rebuild_matches_archive`]).
fn load_archive_entries(src: &std::path::Path) -> Vec<(String, Vec<u8>)> {
    use std::io::Read;

    let file = std::fs::File::open(src).unwrap();
    let mut archive = zip::ZipArchive::new(file).unwrap();
    let names: Vec<String> = archive.file_names().map(String::from).collect();
    let mut entries = Vec::new();
    for name in names {
        if name == "content_hash.json.zst" {
            continue;
        }
        let mut compressed = Vec::new();
        archive
            .by_name(&name)
            .unwrap()
            .read_to_end(&mut compressed)
            .unwrap();
        let bytes = zstd::stream::decode_all(&compressed[..]).unwrap();
        entries.push((name, bytes));
    }
    entries
}

/// Mutate the decompressed bytes of the named entry in place.
fn mutate_entry(entries: &mut [(String, Vec<u8>)], name: &str, f: impl FnOnce(&mut Vec<u8>)) {
    let entry = entries
        .iter_mut()
        .find(|(n, _)| n == name)
        .unwrap_or_else(|| panic!("entry {name} not found"));
    f(&mut entry.1);
}

/// Mutate the top-level metadata JSON in place.
fn mutate_metadata(entries: &mut [(String, Vec<u8>)], f: impl FnOnce(&mut serde_json::Value)) {
    mutate_entry(entries, "metadata.json.zst", |bytes| {
        let mut json: serde_json::Value = serde_json::from_slice(bytes).unwrap();
        f(&mut json);
        *bytes = serde_json::to_vec(&json).unwrap();
    });
}

/// Overwrite the u32 at `index` in a little-endian uint32 entry.
fn set_u32(bytes: &mut [u8], index: usize, value: u32) {
    bytes[index * 4..index * 4 + 4].copy_from_slice(&value.to_le_bytes());
}

/// Overwrite the f64 at `index` in a little-endian float64 entry.
fn set_f64(bytes: &mut [u8], index: usize, value: f64) {
    bytes[index * 8..index * 8 + 8].copy_from_slice(&value.to_le_bytes());
}

/// Write entries to a `.matches` archive with a freshly recomputed
/// `content_hash.json.zst`, so the result is internally consistent except
/// for whatever the caller mutated. Section membership follows the metadata
/// flags (mirroring the writer/verifier), so hash checks stay green and the
/// structural error under test is the one that fires.
fn rebuild_matches_archive(entries: &[(String, Vec<u8>)], dst: &std::path::Path) {
    use crate::archive_io::format_hash;
    use std::io::Write;
    use xxhash_rust::xxh3::Xxh3;

    let metadata_bytes = &entries
        .iter()
        .find(|(n, _)| n == "metadata.json.zst")
        .unwrap()
        .1;
    let metadata: MatchesMetadata = serde_json::from_slice(metadata_bytes).unwrap();

    let section_digest = |prefix: &str| -> Option<u128> {
        let mut names: Vec<&(String, Vec<u8>)> = entries
            .iter()
            .filter(|(n, _)| n.starts_with(prefix))
            .collect();
        if names.is_empty() {
            return None;
        }
        names.sort_by(|a, b| a.0.cmp(&b.0));
        let mut hasher = Xxh3::new();
        for (_, bytes) in names {
            hasher.update(bytes);
        }
        Some(hasher.digest128())
    };

    let metadata_hash = xxhash_rust::xxh3::xxh3_128(metadata_bytes);
    let images_hash = section_digest("images/").unwrap();
    let pairs_hash = if metadata.has_clusters {
        None
    } else {
        section_digest("image_pairs/")
    };
    let clusters_hash = if metadata.has_clusters {
        section_digest("clusters/")
    } else {
        None
    };
    let cp_hash = if metadata.has_cluster_patches {
        section_digest("cluster_patches/")
    } else {
        None
    };
    let tvg_hash = if metadata.has_two_view_geometries {
        section_digest("two_view_geometries/")
    } else {
        None
    };

    let mut digests = vec![metadata_hash, images_hash];
    digests.extend(pairs_hash);
    digests.extend(clusters_hash);
    digests.extend(cp_hash);
    digests.extend(tvg_hash);
    let all_digests_bytes: Vec<u8> = digests.iter().flat_map(|d| d.to_be_bytes()).collect();

    let content_hash = MatchesContentHash {
        metadata_xxh128: format_hash(metadata_hash),
        images_xxh128: format_hash(images_hash),
        image_pairs_xxh128: pairs_hash.map(format_hash),
        clusters_xxh128: clusters_hash.map(format_hash),
        cluster_patches_xxh128: cp_hash.map(format_hash),
        two_view_geometries_xxh128: tvg_hash.map(format_hash),
        content_xxh128: format_hash(xxhash_rust::xxh3::xxh3_128(&all_digests_bytes)),
    };

    let out = std::fs::File::create(dst).unwrap();
    let mut zip_out = zip::ZipWriter::new(out);
    let stored =
        zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored);
    for (name, bytes) in entries {
        zip_out.start_file(name, stored).unwrap();
        zip_out
            .write_all(&zstd::bulk::compress(bytes, 3).unwrap())
            .unwrap();
    }
    zip_out.start_file("content_hash.json.zst", stored).unwrap();
    zip_out
        .write_all(&zstd::bulk::compress(&serde_json::to_vec(&content_hash).unwrap(), 3).unwrap())
        .unwrap();
    zip_out.finish().unwrap();
}

/// Copy a written `.matches` archive, rewriting `metadata.json.zst` so its
/// `version` field reads `version` (dropping the version-3 metadata fields
/// for `version <= 2` and the version-4 `images/image_dims` entry for
/// `version <= 3`, matching what old writers produced), then recomputing the
/// stored hashes so the result is an internally consistent file of that
/// version — for authoring old- or future-version fixture bytes.
fn rewrite_matches_version(src: &std::path::Path, dst: &std::path::Path, version: u32) {
    let mut entries = load_archive_entries(src);
    mutate_metadata(&mut entries, |json| {
        let obj = json.as_object_mut().unwrap();
        obj.insert("version".into(), serde_json::json!(version));
        if version <= 2 {
            obj.remove("has_clusters");
            obj.remove("has_cluster_patches");
            obj.remove("cluster_count");
            obj.remove("cluster_member_count");
        }
    });
    if version <= 3 {
        entries.retain(|(n, _)| !n.starts_with("images/image_dims."));
    }
    rebuild_matches_archive(&entries, dst);
}

/// Craft an invalid cluster-bearing file from a valid one and assert that
/// `verify_matches` reports the expected error (with `read+write` untested —
/// these bytes can't be produced through `write_matches`).
fn expect_verify_error(
    name: &str,
    data: &MatchesData,
    mutate: impl FnOnce(&mut Vec<(String, Vec<u8>)>),
    expected_fragment: &str,
) {
    let dir = std::env::temp_dir().join(name);
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("valid.matches");
    let dst = dir.join("invalid.matches");
    write_matches(&src, data, 3).unwrap();

    let mut entries = load_archive_entries(&src);
    mutate(&mut entries);
    rebuild_matches_archive(&entries, &dst);

    let (valid, errors) = verify_matches(&dst).unwrap();
    assert!(!valid, "expected verification failure: {expected_fragment}");
    assert!(
        errors.iter().any(|e| e.contains(expected_fragment)),
        "Expected an error containing {expected_fragment:?}, got: {errors:?}"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_verify_rejects_both_backbones() {
    // A cluster-bearing file that also contains image_pairs/ entries.
    let dir = std::env::temp_dir().join("matches_test_verify_both_backbones");
    std::fs::create_dir_all(&dir).unwrap();
    let pairwise_path = dir.join("pairwise.matches");
    write_matches(&pairwise_path, &make_test_data(), 3).unwrap();
    let pair_entries: Vec<(String, Vec<u8>)> = load_archive_entries(&pairwise_path)
        .into_iter()
        .filter(|(n, _)| n.starts_with("image_pairs/"))
        .collect();

    expect_verify_error(
        "matches_test_verify_both_backbones",
        &make_cluster_test_data(),
        move |entries| entries.extend(pair_entries),
        "contains image_pairs/ entries",
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_verify_rejects_neither_backbone() {
    // A cluster file whose metadata claims the pairwise backbone: no
    // image_pairs/ section exists, and the clusters/ entries are orphaned.
    expect_verify_error(
        "matches_test_verify_neither_backbone",
        &make_cluster_test_data(),
        |entries| {
            mutate_metadata(entries, |json| {
                let obj = json.as_object_mut().unwrap();
                obj.insert("has_clusters".into(), serde_json::json!(false));
            })
        },
        "no backbone",
    );
}

#[test]
fn test_verify_rejects_bad_csr() {
    expect_verify_error(
        "matches_test_verify_bad_csr",
        &make_cluster_test_data(),
        |entries| {
            mutate_entry(entries, "clusters/cluster_starts.3.uint32.zst", |bytes| {
                set_u32(bytes, 0, 1); // cluster_starts[0] != 0
            })
        },
        "cluster_starts[0] = 1 != 0",
    );

    expect_verify_error(
        "matches_test_verify_csr_final",
        &make_cluster_test_data(),
        |entries| {
            mutate_entry(entries, "clusters/cluster_starts.3.uint32.zst", |bytes| {
                set_u32(bytes, 1, 2); // clusters keep >= 2 members ...
                set_u32(bytes, 2, 4); // ... but the final value != member count 5
            })
        },
        "cluster_starts final value 4 != member count 5",
    );
}

#[test]
fn test_verify_rejects_member_feature_out_of_range() {
    expect_verify_error(
        "matches_test_verify_member_feat_oob",
        &make_cluster_test_data(),
        |entries| {
            mutate_entry(entries, "clusters/member_features.5.uint32.zst", |bytes| {
                set_u32(bytes, 1, 150); // feature_counts[1] = 150
            })
        },
        "member_features[1] = 150 >= feature_counts[1]",
    );
}

#[test]
fn test_verify_rejects_cluster_patches_without_clusters() {
    expect_verify_error(
        "matches_test_verify_cp_no_clusters",
        &make_test_data(),
        |entries| {
            mutate_metadata(entries, |json| {
                let obj = json.as_object_mut().unwrap();
                obj.insert("has_cluster_patches".into(), serde_json::json!(true));
            })
        },
        "has_cluster_patches requires has_clusters",
    );
}

#[test]
fn test_verify_rejects_wrong_array_lengths() {
    expect_verify_error(
        "matches_test_verify_cp_wrong_len",
        &make_cluster_patch_test_data(),
        |entries| {
            mutate_entry(
                entries,
                "cluster_patches/member_zncc.5.float32.zst",
                |bytes| bytes.truncate(4 * 4), // 4 values instead of 5
            )
        },
        "member_zncc byte length 16 != expected 20",
    );
}

#[test]
fn test_verify_rejects_invalid_status() {
    expect_verify_error(
        "matches_test_verify_bad_status",
        &make_cluster_patch_test_data(),
        |entries| {
            mutate_entry(
                entries,
                "cluster_patches/member_status.5.uint8.zst",
                |bytes| bytes[3] = 7,
            )
        },
        "not a valid ClusterMemberStatus",
    );
}

#[test]
fn test_verify_rejects_reference_outside_cluster() {
    expect_verify_error(
        "matches_test_verify_ref_range",
        &make_cluster_patch_test_data(),
        |entries| {
            mutate_entry(
                entries,
                "cluster_patches/reference_members.2.uint32.zst",
                |bytes| set_u32(bytes, 0, 4), // cluster 0 range is [0, 3)
            )
        },
        "outside cluster member range",
    );
}

#[test]
fn test_verify_rejects_zero_image_dims() {
    expect_verify_error(
        "matches_test_verify_zero_dims",
        &make_test_data(),
        |entries| {
            mutate_entry(entries, "images/image_dims.3.2.uint32.zst", |bytes| {
                set_u32(bytes, 4, 0); // image 2's width
            })
        },
        "image_dims[2] has a zero width",
    );
}

#[test]
fn test_verify_rejects_reference_row_not_identity() {
    expect_verify_error(
        "matches_test_verify_ref_not_identity",
        &make_cluster_patch_test_data(),
        |entries| {
            mutate_entry(
                entries,
                "cluster_patches/member_affines.5.2.3.float64.zst",
                |bytes| set_f64(bytes, 1, 0.25), // reference row's A01
            )
        },
        "identity leading 2x2",
    );
}

#[test]
fn test_verify_rejects_two_kept_members_one_image() {
    expect_verify_error(
        "matches_test_verify_dup_kept",
        &make_cluster_patch_test_data(),
        |entries| {
            // Put cluster 1's two members in the same image and keep both.
            mutate_entry(entries, "clusters/member_images.5.uint32.zst", |bytes| {
                set_u32(bytes, 4, 0); // member 3 is already in image 0
            });
            mutate_entry(
                entries,
                "cluster_patches/member_status.5.uint8.zst",
                |bytes| {
                    bytes[3] = ClusterMemberStatus::Kept as u8;
                    bytes[4] = ClusterMemberStatus::Kept as u8;
                },
            );
        },
        "both reference/kept for image 0",
    );
}

#[test]
fn test_writer_always_writes_current_version() {
    let mut data = make_test_data();
    data.metadata.version = 1; // stale caller-supplied version is overridden
    let (dir, path) = write_to_temp("matches_test_write_version", &data);

    let metadata = read_matches_metadata(&path).unwrap();
    assert_eq!(metadata.version, MATCHES_FORMAT_VERSION);

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn test_version_1_relative_poses_upgrade_on_load() {
    let data = make_test_data_with_tvg();
    let dir = std::env::temp_dir().join("matches_test_v1_upgrade");
    std::fs::create_dir_all(&dir).unwrap();
    let current_path = dir.join("current.matches");
    let v1_path = dir.join("v1.matches");

    write_matches(&current_path, &data, 3).unwrap();
    rewrite_matches_version(&current_path, &v1_path, 1);
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
fn test_version_2_loads_without_pose_conjugation() {
    let data = make_test_data_with_tvg();
    let dir = std::env::temp_dir().join("matches_test_v2_load");
    std::fs::create_dir_all(&dir).unwrap();
    let current_path = dir.join("current.matches");
    let v2_path = dir.join("v2.matches");

    write_matches(&current_path, &data, 3).unwrap();
    rewrite_matches_version(&current_path, &v2_path, 2);
    assert_eq!(read_matches_metadata(&v2_path).unwrap().version, 2);

    let (valid, errors) = verify_matches(&v2_path).unwrap();
    assert!(valid, "v2 fixture failed verification: {errors:?}");

    // Version 2 poses are already canonical: loaded unchanged, and the
    // in-memory version upgrades to current. Version ≤ 3 files never stored
    // per-image dimensions, so they load with image_dims = None.
    let loaded = read_matches(&v2_path).unwrap();
    assert_eq!(loaded.metadata.version, MATCHES_FORMAT_VERSION);
    assert_eq!(loaded.image_dims, None);
    let tvg = loaded.two_view_geometries.as_ref().unwrap();
    let orig = data.two_view_geometries.as_ref().unwrap();
    assert_eq!(tvg.quaternions_wxyz, orig.quaternions_wxyz);
    assert_eq!(tvg.translations_xyz, orig.translations_xyz);

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn test_version_3_clusters_load_without_image_dims() {
    // A version-3 cluster-backbone file (no image_dims, no cluster_patches)
    // verifies and loads unchanged; image_dims comes back None.
    let data = make_cluster_test_data();
    let dir = std::env::temp_dir().join("matches_test_v3_clusters_load");
    std::fs::create_dir_all(&dir).unwrap();
    let current_path = dir.join("current.matches");
    let v3_path = dir.join("v3.matches");

    write_matches(&current_path, &data, 3).unwrap();
    rewrite_matches_version(&current_path, &v3_path, 3);
    assert_eq!(read_matches_metadata(&v3_path).unwrap().version, 3);

    let (valid, errors) = verify_matches(&v3_path).unwrap();
    assert!(valid, "v3 fixture failed verification: {errors:?}");

    let loaded = read_matches(&v3_path).unwrap();
    assert_eq!(loaded.metadata.version, MATCHES_FORMAT_VERSION);
    assert_eq!(loaded.image_dims, None);
    let clusters = loaded.clusters.as_ref().unwrap();
    let orig = data.clusters.as_ref().unwrap();
    assert_eq!(clusters.cluster_starts, orig.cluster_starts);
    assert_eq!(clusters.member_images, orig.member_images);
    assert_eq!(clusters.member_features, orig.member_features);

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn test_version_3_cluster_patches_rejected_on_read() {
    // A version-3 cluster-patch file stores the affine translation in the
    // member_affines last column — not upgradable without the .sift
    // positions, so read_matches refuses it (with regeneration guidance)
    // while verify_matches still validates its stored bytes.
    let data = make_cluster_patch_test_data();
    let dir = std::env::temp_dir().join("matches_test_v3_patches_rejected");
    std::fs::create_dir_all(&dir).unwrap();
    let current_path = dir.join("current.matches");
    let v3_path = dir.join("v3.matches");

    write_matches(&current_path, &data, 3).unwrap();
    rewrite_matches_version(&current_path, &v3_path, 3);

    let (valid, errors) = verify_matches(&v3_path).unwrap();
    assert!(valid, "v3 patch fixture failed verification: {errors:?}");

    let err = read_matches(&v3_path).err().unwrap();
    let msg = format!("{err}");
    assert!(
        msg.contains("regenerate with `sfm cluster-patches`"),
        "{msg}"
    );

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn test_version_2_with_clusters_rejected() {
    // A version <= 2 file can never carry the cluster backbone.
    let data = make_cluster_test_data();
    let dir = std::env::temp_dir().join("matches_test_v2_clusters");
    std::fs::create_dir_all(&dir).unwrap();
    let current_path = dir.join("current.matches");
    let v2_path = dir.join("v2.matches");

    write_matches(&current_path, &data, 3).unwrap();
    let mut entries = load_archive_entries(&current_path);
    mutate_metadata(&mut entries, |json| {
        json.as_object_mut()
            .unwrap()
            .insert("version".into(), serde_json::json!(2));
    });
    rebuild_matches_archive(&entries, &v2_path);

    let expected = "claims clusters/cluster_patches (introduced in version 3)";
    let err = read_matches(&v2_path).err().unwrap();
    assert!(format!("{err}").contains(expected), "{err}");
    let (valid, errors) = verify_matches(&v2_path).unwrap();
    assert!(
        !valid && errors.iter().any(|e| e.contains(expected)),
        "{errors:?}"
    );

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
    let mut entries = load_archive_entries(&src);
    mutate_metadata(&mut entries, |json| {
        json.as_object_mut()
            .unwrap()
            .insert("version".into(), serde_json::json!(future_version));
    });
    rebuild_matches_archive(&entries, &dst);

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
    let (dir, path) = write_to_temp("matches_test_stored_compression", &data);

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

#[test]
fn test_cluster_archive_uses_stored_compression() {
    let data = make_cluster_patch_test_data();
    let (dir, path) = write_to_temp("matches_test_cluster_stored_compression", &data);

    let file = std::fs::File::open(&path).unwrap();
    let mut archive = zip::ZipArchive::new(file).unwrap();
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
