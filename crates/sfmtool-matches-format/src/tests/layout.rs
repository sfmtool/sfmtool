// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Arrays a caller hands in outside standard layout: the writer and the
//! cluster selection read them in row-major order, as they would a standard
//! layout copy, rather than panicking.

use super::cells::make_cell_test_data;
use super::*;
use ndarray::{Array, Axis, Dimension, ShapeBuilder, Slice};

/// `array`'s shape and values in a layout that is not standard: column-major
/// (Fortran) order for an array of two or more axes, and every other element
/// of a buffer twice as long for one axis.
fn non_standard<A: Clone, D: Dimension>(array: &Array<A, D>) -> Array<A, D> {
    let out = if array.ndim() >= 2 {
        let values: Vec<A> = array.t().iter().cloned().collect();
        Array::from_shape_vec(array.raw_dim().f(), values).unwrap()
    } else {
        let mut dim = array.raw_dim();
        dim[0] *= 2;
        let mut out = Array::from_elem(dim, array.first().unwrap().clone());
        out.slice_axis_inplace(Axis(0), Slice::new(0, None, 2));
        out.assign(array);
        out
    };
    assert!(
        !out.is_standard_layout(),
        "shape {:?} has no non-standard layout",
        array.shape()
    );
    out
}

/// Every array of `data` the pairs backbone, the two-view geometries, the
/// clusters backbone and the cluster patches carry, replaced by
/// [`non_standard`].
fn to_non_standard(data: &mut MatchesData) {
    fn swap<A: Clone, D: Dimension>(array: &mut Array<A, D>) {
        *array = non_standard(array);
    }
    swap(&mut data.feature_counts);
    swap(data.image_dims.as_mut().unwrap());
    if let Some(pairs) = data.image_pairs.as_mut() {
        swap(&mut pairs.image_index_pairs);
        swap(&mut pairs.match_counts);
        swap(&mut pairs.match_feature_indexes);
        swap(&mut pairs.match_descriptor_distances);
    }
    if let Some(tvg) = data.two_view_geometries.as_mut() {
        swap(&mut tvg.config_indexes);
        swap(&mut tvg.inlier_counts);
        swap(&mut tvg.inlier_feature_indexes);
        swap(&mut tvg.f_matrices);
        swap(&mut tvg.e_matrices);
        swap(&mut tvg.h_matrices);
        swap(&mut tvg.quaternions_wxyz);
        swap(&mut tvg.translations_xyz);
    }
    if let Some(clusters) = data.clusters.as_mut() {
        swap(&mut clusters.cluster_starts);
        swap(&mut clusters.member_images);
        swap(&mut clusters.member_features);
        swap(clusters.member_positions.as_mut().unwrap());
        swap(clusters.member_affine_shapes.as_mut().unwrap());
    }
    if let Some(cp) = data.cluster_patches.as_mut() {
        swap(&mut cp.reference_members);
        swap(&mut cp.member_status);
        swap(&mut cp.member_zncc);
        swap(&mut cp.member_shift_px);
        swap(&mut cp.member_consistency_residual);
        let cells = cp.member_cells.as_mut().unwrap();
        swap(&mut cells.shift_px);
        swap(&mut cells.zncc);
        swap(&mut cells.status);
        swap(&mut cells.iterations);
    }
}

/// Write `make()` as built and with its arrays out of standard layout, and
/// check that the two files hold the same bytes: every section hash agrees,
/// and the second reads back to the first. Returns both loaded values.
fn write_both(name: &str, make: fn() -> MatchesData) -> (MatchesData, MatchesData) {
    let standard = make();
    let mut altered = make();
    to_non_standard(&mut altered);
    let (dir_a, path_a) = write_to_temp(&format!("{name}_standard"), &standard);
    let (dir_b, path_b) = write_to_temp(&format!("{name}_non_standard"), &altered);
    let loaded_a = read_matches(&path_a).unwrap();
    let loaded_b = read_matches(&path_b).unwrap();
    assert_eq!(
        serde_json::to_value(&loaded_b.content_hash).unwrap(),
        serde_json::to_value(&loaded_a.content_hash).unwrap()
    );
    let (valid, errors) = verify_matches(&path_b).unwrap();
    assert!(valid, "Verification failed: {errors:?}");
    std::fs::remove_dir_all(&dir_a).unwrap();
    std::fs::remove_dir_all(&dir_b).unwrap();
    (loaded_a, loaded_b)
}

#[test]
fn test_write_pairs_and_tvg_out_of_standard_layout() {
    let (_, loaded) = write_both("matches_layout_pairs", make_test_data_with_tvg);
    let data = make_test_data_with_tvg();
    assert_eq!(loaded.feature_counts, data.feature_counts);
    assert_eq!(loaded.image_dims, data.image_dims);
    let (pairs, orig_pairs) = (
        loaded.image_pairs.as_ref().unwrap(),
        data.image_pairs.as_ref().unwrap(),
    );
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
    let (tvg, orig_tvg) = (
        loaded.two_view_geometries.as_ref().unwrap(),
        data.two_view_geometries.as_ref().unwrap(),
    );
    assert_eq!(tvg.config_indexes, orig_tvg.config_indexes);
    assert_eq!(tvg.inlier_counts, orig_tvg.inlier_counts);
    assert_eq!(tvg.inlier_feature_indexes, orig_tvg.inlier_feature_indexes);
    assert_eq!(tvg.e_matrices, orig_tvg.e_matrices);
    assert_eq!(tvg.quaternions_wxyz, orig_tvg.quaternions_wxyz);
    assert_eq!(tvg.translations_xyz, orig_tvg.translations_xyz);
}

#[test]
fn test_write_clusters_and_patches_out_of_standard_layout() {
    let (_, loaded) = write_both("matches_layout_clusters", make_cell_test_data);
    let data = make_cell_test_data();
    let (clusters, orig) = (
        loaded.clusters.as_ref().unwrap(),
        data.clusters.as_ref().unwrap(),
    );
    assert_eq!(clusters.cluster_starts, orig.cluster_starts);
    assert_eq!(clusters.member_images, orig.member_images);
    assert_eq!(clusters.member_features, orig.member_features);
    assert_eq!(clusters.member_positions, orig.member_positions);
    assert_eq!(clusters.member_affine_shapes, orig.member_affine_shapes);
    let (cp, orig_cp) = (
        loaded.cluster_patches.as_ref().unwrap(),
        data.cluster_patches.as_ref().unwrap(),
    );
    assert_eq!(cp.reference_members, orig_cp.reference_members);
    assert_eq!(cp.member_status, orig_cp.member_status);
    assert_f32_bits_eq(&cp.member_zncc, &orig_cp.member_zncc);
    let (cells, orig_cells) = (
        cp.member_cells.as_ref().unwrap(),
        orig_cp.member_cells.as_ref().unwrap(),
    );
    assert_eq!(cells.status, orig_cells.status);
    assert_eq!(cells.iterations, orig_cells.iterations);
}

#[test]
fn test_select_clusters_out_of_standard_layout() {
    let data = make_cell_test_data();
    let mut altered = make_cell_test_data();
    to_non_standard(&mut altered);
    let opts = ClusterSelect {
        accepted_statuses: vec![ClusterMemberStatus::Reference, ClusterMemberStatus::Kept],
        ..ClusterSelect::default()
    };
    let expected = data.select_clusters(&opts).unwrap();
    let selected = altered.select_clusters(&opts).unwrap();
    let (clusters, expected_clusters) = (
        selected.clusters.as_ref().unwrap(),
        expected.clusters.as_ref().unwrap(),
    );
    assert_eq!(clusters.cluster_starts, expected_clusters.cluster_starts);
    assert_eq!(clusters.member_images, expected_clusters.member_images);
    assert_eq!(clusters.member_features, expected_clusters.member_features);
    assert_eq!(
        altered.cluster_worst_consistency(),
        data.cluster_worst_consistency()
    );
}
