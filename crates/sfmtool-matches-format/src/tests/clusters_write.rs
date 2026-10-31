// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! `write_clusters_matches`, the writer for a cluster matcher's output.

use super::*;

/// The clusters fixture of `make_cluster_test_data` as the clusters writer
/// takes it.
fn make_clusters_matches_data() -> ClustersMatchesData {
    let data = make_cluster_test_data();
    ClustersMatchesData {
        matching_method: "cluster".into(),
        matching_tool: "sfmtool".into(),
        matching_tool_version: "0.3.0".into(),
        matching_options: {
            let mut m = BTreeMap::new();
            m.insert("mode".into(), serde_json::json!("background-floor"));
            m.insert("d".into(), serde_json::json!(10));
            m
        },
        workspace: data.metadata.workspace,
        timestamp: data.metadata.timestamp,
        image_names: data.image_names,
        feature_tool_hashes: data.feature_tool_hashes,
        sift_content_hashes: data.sift_content_hashes,
        feature_counts: data.feature_counts,
        image_dims: data.image_dims.unwrap(),
        clusters: data.clusters.unwrap(),
    }
}

/// Write `data` with the clusters writer into a fresh directory under the
/// system temp directory; return the directory and the result.
fn write_clusters_to_temp(
    name: &str,
    data: ClustersMatchesData,
) -> (std::path::PathBuf, Result<(), MatchesError>) {
    let dir = std::env::temp_dir().join(name);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let result = write_clusters_matches(&dir.join("test.matches"), data, 3);
    (dir, result)
}

fn expect_clusters_write_error(name: &str, data: ClustersMatchesData, expected_fragment: &str) {
    let (dir, result) = write_clusters_to_temp(name, data);
    let msg = result
        .expect_err(&format!("expected write to fail: {expected_fragment}"))
        .to_string();
    assert!(
        msg.contains(expected_fragment),
        "Error should mention {expected_fragment:?}, got: {msg}"
    );
    assert!(!dir.join("test.matches").exists());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_clusters_writer_round_trip() {
    let data = make_clusters_matches_data();
    let expected = make_clusters_matches_data();
    let (dir, result) = write_clusters_to_temp("matches_test_clusters_writer_round_trip", data);
    result.unwrap();
    let path = dir.join("test.matches");
    let loaded = read_matches(&path).unwrap();

    // The derived metadata.
    let meta = &loaded.metadata;
    assert_eq!(meta.version, MATCHES_FORMAT_VERSION);
    assert_eq!(meta.image_count, 3);
    assert_eq!(meta.cluster_count, Some(2));
    assert_eq!(meta.cluster_member_count, Some(5));
    assert_eq!(meta.image_pair_count, None);
    assert_eq!(meta.match_count, None);
    assert!(meta.has_clusters);
    assert!(!meta.has_cluster_patches);
    assert!(!meta.has_two_view_geometries);

    // The supplied metadata.
    assert_eq!(meta.matching_method, expected.matching_method);
    assert_eq!(meta.matching_tool, expected.matching_tool);
    assert_eq!(meta.matching_tool_version, expected.matching_tool_version);
    assert_eq!(meta.matching_options, expected.matching_options);
    assert_eq!(meta.timestamp, expected.timestamp);
    assert_eq!(
        meta.workspace.absolute_path,
        expected.workspace.absolute_path
    );
    assert_eq!(
        meta.workspace.relative_path,
        expected.workspace.relative_path
    );

    // The images.
    assert_eq!(loaded.image_names, expected.image_names);
    assert_eq!(loaded.feature_tool_hashes, expected.feature_tool_hashes);
    assert_eq!(loaded.sift_content_hashes, expected.sift_content_hashes);
    assert_eq!(loaded.feature_counts, expected.feature_counts);
    assert_eq!(loaded.image_dims.as_ref(), Some(&expected.image_dims));

    // The backbone, its geometry bit for bit, and no other section.
    assert!(loaded.image_pairs.is_none());
    assert!(loaded.cluster_patches.is_none());
    assert!(loaded.two_view_geometries.is_none());
    let clusters = loaded.clusters.as_ref().unwrap();
    let orig = &expected.clusters;
    assert_eq!(clusters.cluster_starts, orig.cluster_starts);
    assert_eq!(clusters.member_images, orig.member_images);
    assert_eq!(clusters.member_features, orig.member_features);
    assert_eq!(clusters.matcher_options, orig.matcher_options);
    let bits = |a: &[f32]| a.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    assert_eq!(
        bits(
            clusters
                .member_positions
                .as_ref()
                .unwrap()
                .as_slice()
                .unwrap()
        ),
        bits(orig.member_positions.as_ref().unwrap().as_slice().unwrap())
    );
    assert_eq!(
        bits(
            clusters
                .member_affine_shapes
                .as_ref()
                .unwrap()
                .as_slice()
                .unwrap()
        ),
        bits(
            orig.member_affine_shapes
                .as_ref()
                .unwrap()
                .as_slice()
                .unwrap()
        )
    );

    let (valid, errors) = verify_matches(&path).unwrap();
    assert!(valid, "Verification failed: {errors:?}");

    std::fs::remove_dir_all(&dir).unwrap();
}

/// The clusters writer and `write_matches` given the same file write the
/// same bytes, so there is one layout.
#[test]
fn test_clusters_writer_matches_general_writer() {
    let mut general = make_cluster_test_data();
    let clusters = make_clusters_matches_data();
    general.metadata.matching_method = clusters.matching_method.clone();
    general.metadata.matching_tool = clusters.matching_tool.clone();
    general.metadata.matching_tool_version = clusters.matching_tool_version.clone();
    general.metadata.matching_options = clusters.matching_options.clone();

    let (dir, general_path) = write_to_temp("matches_test_clusters_writer_same_bytes", &general);
    let clusters_path = dir.join("clusters.matches");
    write_clusters_matches(&clusters_path, clusters, 3).unwrap();
    assert_eq!(
        std::fs::read(&general_path).unwrap(),
        std::fs::read(&clusters_path).unwrap()
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn test_clusters_writer_rejects_member_image_out_of_range() {
    let mut data = make_clusters_matches_data();
    data.clusters.member_images[0] = 3; // three images
    expect_clusters_write_error(
        "matches_test_clusters_writer_member_image",
        data,
        "member_images[0] = 3 >= image_count 3",
    );
}

#[test]
fn test_clusters_writer_rejects_member_feature_out_of_range() {
    let mut data = make_clusters_matches_data();
    data.clusters.member_features[1] = 150; // feature_counts[1] = 150
    expect_clusters_write_error(
        "matches_test_clusters_writer_member_feature",
        data,
        "member_features[1] = 150 >= feature_counts[1] = 150",
    );
}

#[test]
fn test_clusters_writer_rejects_bad_cluster_starts() {
    let mut data = make_clusters_matches_data();
    data.clusters.cluster_starts = Array1::from_vec(vec![1, 3, 5]);
    expect_clusters_write_error(
        "matches_test_clusters_writer_starts_origin",
        data,
        "cluster_starts[0] = 1 != 0",
    );

    let mut data = make_clusters_matches_data();
    data.clusters.cluster_starts = Array1::from_vec(vec![0, 2, 4]);
    expect_clusters_write_error(
        "matches_test_clusters_writer_starts_end",
        data,
        "cluster_starts final value 4 != member count 5",
    );

    let mut data = make_clusters_matches_data();
    data.clusters.cluster_starts = Array1::from_vec(vec![]);
    expect_clusters_write_error(
        "matches_test_clusters_writer_starts_empty",
        data,
        "cluster_starts len 0 != cluster_count + 1 (1)",
    );
}

#[test]
fn test_clusters_writer_rejects_mismatched_lengths() {
    let mut data = make_clusters_matches_data();
    data.feature_counts = Array1::from_vec(vec![100, 150]);
    expect_clusters_write_error(
        "matches_test_clusters_writer_feature_counts",
        data,
        "feature_counts len 2 != image_count 3",
    );

    let mut data = make_clusters_matches_data();
    data.clusters.member_features = Array1::from_vec(vec![0, 1, 2, 5]);
    expect_clusters_write_error(
        "matches_test_clusters_writer_member_features",
        data,
        "member_features len 4 != cluster_member_count 5",
    );
}

#[test]
fn test_clusters_writer_requires_member_geometry() {
    let mut data = make_clusters_matches_data();
    data.clusters.member_positions = None;
    data.clusters.member_affine_shapes = None;
    expect_clusters_write_error(
        "matches_test_clusters_writer_geometry",
        data,
        "member_positions and clusters.member_affine_shapes are required",
    );
}
