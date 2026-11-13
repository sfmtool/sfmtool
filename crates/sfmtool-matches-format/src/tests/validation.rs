// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The writer's structural rules held by the reader and the verifier too.
//!
//! Each file here is a valid one with bytes changed after it was written, so
//! only damage or a hand edit could make it; [`rebuild_matches_archive`]
//! restores its hashes, so the broken rule is the only thing wrong with it.
//! `read_matches` used to open such files, and code that indexes by the
//! values they store, such as `select_clusters`, then panicked.

use super::*;

const CLUSTER_STARTS: &str = "clusters/cluster_starts.3.uint32.zst";
const MEMBER_IMAGES: &str = "clusters/member_images.5.uint32.zst";
const MEMBER_FEATURES: &str = "clusters/member_features.5.uint32.zst";
const MEMBER_POSITIONS: &str = "clusters/member_positions.5.2.float32.zst";
const MEMBER_SHAPES: &str = "clusters/member_affine_shapes.5.2.2.float32.zst";
const MEMBER_STATUS: &str = "cluster_patches/member_status.5.uint8.zst";
const REFERENCE_MEMBERS: &str = "cluster_patches/reference_members.2.uint32.zst";
const IMAGE_INDEX_PAIRS: &str = "image_pairs/image_index_pairs.2.2.uint32.zst";
const MATCH_COUNTS: &str = "image_pairs/match_counts.2.uint32.zst";
const MATCH_FEATURE_INDEXES: &str = "image_pairs/match_feature_indexes.5.2.uint32.zst";
const IMAGE_DIMS: &str = "images/image_dims.3.2.uint32.zst";
const INLIER_COUNTS: &str = "two_view_geometries/inlier_counts.2.uint32.zst";
const INLIER_FEATURE_INDEXES: &str = "two_view_geometries/inlier_feature_indexes.3.2.uint32.zst";

/// Write `data`, change its entries with `mutate`, and assert that
/// `read_matches` refuses the result as `InvalidFormat` with exactly
/// `expected`.
fn expect_read_error(
    name: &str,
    data: &MatchesData,
    mutate: impl FnOnce(&mut Vec<(String, Vec<u8>)>),
    expected: &str,
) {
    let dir = std::env::temp_dir().join(name);
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("valid.matches");
    let dst = dir.join("invalid.matches");
    write_matches(&src, data, 3).unwrap();
    read_matches(&src).unwrap();

    let mut entries = load_archive_entries(&src);
    mutate(&mut entries);
    rebuild_matches_archive(&entries, &dst);

    match read_matches(&dst) {
        Err(MatchesError::InvalidFormat(message)) => assert_eq!(message, expected),
        Err(other) => panic!("expected InvalidFormat({expected:?}), got {other}"),
        Ok(_) => panic!("expected InvalidFormat({expected:?}), but the file read"),
    }

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn read_refuses_a_cluster_run_past_the_members() {
    // Cluster 1 runs from member 3 to 9 of 5: `select_clusters` indexed
    // member 5 of 5 here.
    expect_read_error(
        "matches_test_read_csr_overrun",
        &make_cluster_test_data(),
        |entries| mutate_entry(entries, CLUSTER_STARTS, |bytes| set_u32(bytes, 2, 9)),
        "cluster_starts final value 9 != member count 5",
    );
    expect_read_error(
        "matches_test_read_csr_start",
        &make_cluster_test_data(),
        |entries| mutate_entry(entries, CLUSTER_STARTS, |bytes| set_u32(bytes, 0, 1)),
        "cluster_starts[0] = 1 != 0",
    );
}

#[test]
fn read_refuses_a_member_past_its_image_or_feature_list() {
    // Image 7 of 3: `select_clusters` with an image restriction indexed its
    // per-image map at 7.
    expect_read_error(
        "matches_test_read_member_image",
        &make_cluster_test_data(),
        |entries| mutate_entry(entries, MEMBER_IMAGES, |bytes| set_u32(bytes, 4, 7)),
        "member_images[4] = 7 >= image_count 3",
    );
    expect_read_error(
        "matches_test_read_member_feature",
        &make_cluster_test_data(),
        |entries| mutate_entry(entries, MEMBER_FEATURES, |bytes| set_u32(bytes, 1, 150)),
        "member_features[1] = 150 >= feature_counts[1] = 150",
    );
}

#[test]
fn read_refuses_nan_member_geometry() {
    expect_read_error(
        "matches_test_read_member_nan",
        &make_cluster_test_data(),
        |entries| {
            mutate_entry(entries, MEMBER_POSITIONS, |bytes| {
                set_f32(bytes, 5, f32::NAN)
            })
        },
        "member_positions[2] contains NaN",
    );
    expect_read_error(
        "matches_test_read_shape_nan",
        &make_cluster_test_data(),
        |entries| mutate_entry(entries, MEMBER_SHAPES, |bytes| set_f32(bytes, 13, f32::NAN)),
        "member_affine_shapes[3] contains NaN",
    );
}

#[test]
fn read_refuses_a_broken_cluster_reference() {
    expect_read_error(
        "matches_test_read_ref_range",
        &make_cluster_patch_test_data(),
        |entries| mutate_entry(entries, REFERENCE_MEMBERS, |bytes| set_u32(bytes, 0, 4)),
        "reference_members[0] = 4 outside cluster member range [0, 3)",
    );
    expect_read_error(
        "matches_test_read_ref_status",
        &make_cluster_patch_test_data(),
        |entries| mutate_entry(entries, REFERENCE_MEMBERS, |bytes| set_u32(bytes, 0, 1)),
        "reference_members[0] = 1 has status 1, expected 0 (reference)",
    );
    expect_read_error(
        "matches_test_read_ref_singular",
        &make_cluster_patch_test_data(),
        |entries| {
            mutate_entry(entries, MEMBER_SHAPES, |bytes| {
                // Row 1 of the reference's shape a multiple of row 0.
                set_f32(bytes, 2, 2.0 * CP_S_REF[0][0] as f32);
                set_f32(bytes, 3, 2.0 * CP_S_REF[0][1] as f32);
            })
        },
        "member_affine_shapes[0] (cluster 0's reference member) is singular (det 0)",
    );
    expect_read_error(
        "matches_test_read_two_kept",
        &make_cluster_patch_test_data(),
        |entries| {
            mutate_entry(entries, MEMBER_IMAGES, |bytes| set_u32(bytes, 4, 0));
            mutate_entry(entries, MEMBER_STATUS, |bytes| {
                bytes[3] = ClusterMemberStatus::Kept as u8;
                bytes[4] = ClusterMemberStatus::Kept as u8;
            });
        },
        "cluster 1: members 3 and 4 are both reference/kept for image 0",
    );
}

#[test]
fn read_refuses_a_broken_pair_table() {
    expect_read_error(
        "matches_test_read_pair_order",
        &make_test_data(),
        |entries| {
            mutate_entry(entries, IMAGE_INDEX_PAIRS, |bytes| {
                set_u32(bytes, 2, 0); // pair 1 = (0, 1), a repeat of pair 0
                set_u32(bytes, 3, 1);
            })
        },
        "image_index_pairs not sorted at index 1: (0, 1) >= (0, 1)",
    );
    expect_read_error(
        "matches_test_read_match_sum",
        &make_test_data(),
        |entries| mutate_entry(entries, MATCH_COUNTS, |bytes| set_u32(bytes, 0, 4)),
        "sum of match_counts (6) != match_count (5)",
    );
    expect_read_error(
        "matches_test_read_match_feature",
        &make_test_data(),
        |entries| {
            mutate_entry(entries, MATCH_FEATURE_INDEXES, |bytes| {
                set_u32(bytes, 0, 100)
            })
        },
        "match_feature_indexes[0][0] = 100 >= feature_counts[0] = 100",
    );
}

#[test]
fn read_refuses_inliers_that_disagree_with_the_pair_table() {
    expect_read_error(
        "matches_test_read_inlier_sum",
        &make_test_data_with_tvg(),
        |entries| mutate_entry(entries, INLIER_COUNTS, |bytes| set_u32(bytes, 1, 2)),
        "sum of inlier_counts (4) != inlier_count (3)",
    );
    expect_read_error(
        "matches_test_read_inlier_subset",
        &make_test_data_with_tvg(),
        |entries| {
            mutate_entry(entries, INLIER_FEATURE_INDEXES, |bytes| {
                set_u32(bytes, 0, 99)
            })
        },
        "inlier (99, 0) at index 0 for pair 0 is not in candidate matches",
    );
}

#[test]
fn read_refuses_a_zero_image_dimension() {
    expect_read_error(
        "matches_test_read_zero_dims",
        &make_test_data(),
        |entries| mutate_entry(entries, IMAGE_DIMS, |bytes| set_u32(bytes, 4, 0)),
        "image_dims[2] has a zero width (every dimension must be >= 1)",
    );
}

#[test]
fn verify_reports_an_inlier_that_is_not_a_candidate_match() {
    expect_verify_errors(
        "matches_test_verify_inlier_subset",
        &make_test_data_with_tvg(),
        |entries| {
            mutate_entry(entries, INLIER_FEATURE_INDEXES, |bytes| {
                set_u32(bytes, 0, 99)
            })
        },
        &["inlier (99, 0) at index 0 for pair 0 is not in candidate matches"],
    );
}

#[test]
fn verify_reports_a_short_inlier_array_instead_of_walking_it() {
    expect_verify_errors(
        "matches_test_verify_short_inliers",
        &make_test_data_with_tvg(),
        |entries| mutate_entry(entries, INLIER_FEATURE_INDEXES, |bytes| bytes.truncate(16)),
        &["inlier_feature_indexes byte length 16 != expected 24 (3 uint32 pairs)"],
    );
}

#[test]
fn inliers_in_another_order_than_their_candidates_are_accepted() {
    // The check walks the candidates in order first; inliers out of that
    // order take the search, which must find them all the same.
    let mut data = make_test_data_with_tvg();
    let tvg = data.two_view_geometries.as_mut().unwrap();
    tvg.inlier_feature_indexes = Array2::from_shape_vec((3, 2), vec![1, 1, 0, 0, 5, 10]).unwrap();
    let dir = std::env::temp_dir().join("matches_test_inliers_reordered");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("reordered.matches");
    write_matches(&path, &data, 3).unwrap();
    let read = read_matches(&path).unwrap();
    assert_eq!(
        read.two_view_geometries.unwrap().inlier_feature_indexes,
        data.two_view_geometries.unwrap().inlier_feature_indexes
    );
    assert_eq!(verify_matches(&path).unwrap(), (true, vec![]));
    let _ = std::fs::remove_dir_all(&dir);
}
