// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use matches_format::{ClusterPatchData, ClustersData, MatchesData};
use ndarray::{Array1, Array2, Array3};

use super::*;

/// A member shape `[[a, b], [c, d]]`, row-major, as the flat four the kernels
/// take.
fn shape(a: f32, b: f32, c: f32, d: f32) -> [f32; 4] {
    [a, b, c, d]
}

fn flat(rows: &[[f32; 4]]) -> Vec<f32> {
    rows.iter().flatten().copied().collect()
}

#[test]
fn a_members_radius_is_half_the_refine_radius_times_the_column_norm_sum() {
    // Columns of [[3, 0], [4, 10]] are (3, 4) and (0, 10): norms 5 and 10.
    // Radius = 0.5 * 6 * 15 = 45.
    let shapes = flat(&[shape(3.0, 0.0, 4.0, 10.0)]);
    assert_eq!(member_radii(&shapes, 6.0), vec![45.0]);
    // A caller holding the same shapes widened -- the source-cluster join's
    // selection arrays are f64 -- narrows them back at its own boundary and
    // reads the same number, so the widening is not a second convention.
    let wide: Vec<f64> = shapes.iter().map(|&v| f64::from(v)).collect();
    let narrowed: Vec<f32> = wide.iter().map(|&v| v as f32).collect();
    assert_eq!(member_radii(&narrowed, 6.0), vec![45.0]);
}

#[test]
fn a_cluster_takes_its_widest_members_radius() {
    let starts = [0u32, 2, 3];
    let shapes = flat(&[
        // Cluster 0: scaled identities at 0.25 and 0.75, so radius 6 * 0.75.
        shape(0.25, 0.0, 0.0, 0.25),
        shape(0.75, 0.0, 0.0, 0.75),
        // Cluster 1: a lone member at 1.0.
        shape(1.0, 0.0, 0.0, 1.0),
    ]);
    assert_eq!(
        cluster_radii(&starts, &shapes, 6.0),
        vec![6.0 * 0.75, 6.0 * 1.0]
    );
}

#[test]
fn a_cluster_with_no_members_reads_zero() {
    let starts = [0u32, 1, 1, 2];
    let shapes = flat(&[shape(1.0, 0.0, 0.0, 1.0), shape(2.0, 0.0, 0.0, 2.0)]);
    assert_eq!(cluster_radii(&starts, &shapes, 4.0), vec![4.0, 0.0, 8.0]);
}

#[test]
fn the_coarsest_cut_is_radius_descending_with_ties_by_ascending_id() {
    // Radii 2, 3, 2, 1 (times 5): the two coarsest are 1 then 0, and 0 beats 2
    // on the tie.
    let starts = [0u32, 1, 2, 3, 4];
    let shapes = flat(&[
        shape(2.0, 0.0, 0.0, 2.0),
        shape(3.0, 0.0, 0.0, 3.0),
        shape(2.0, 0.0, 0.0, 2.0),
        shape(1.0, 0.0, 0.0, 1.0),
    ]);
    // Ascending return, so the tie shows in WHICH ids are taken, not in order.
    assert_eq!(coarsest_clusters(&starts, &shapes, 5.0, 2), vec![0, 1]);
    assert_eq!(coarsest_clusters(&starts, &shapes, 5.0, 3), vec![0, 1, 2]);
    // Asking for more than exist yields all of them.
    assert_eq!(
        coarsest_clusters(&starts, &shapes, 5.0, 99),
        vec![0, 1, 2, 3]
    );
    assert!(coarsest_clusters(&starts, &shapes, 5.0, 0).is_empty());
}

/// A minimal cluster-backbone file with the `cluster_patches/` section the
/// radius reading needs.
fn file(starts: &[u32], shapes: &[f32], patch_size: f64, with_patches: bool) -> MatchesData {
    let m = shapes.len() / 4;
    let n_cl = starts.len() - 1;
    MatchesData {
        metadata: matches_format::MatchesMetadata {
            version: matches_format::MATCHES_FORMAT_VERSION,
            matching_method: "test".into(),
            matching_tool: "test".into(),
            matching_tool_version: "0".into(),
            matching_options: std::collections::BTreeMap::new(),
            workspace: matches_format::WorkspaceMetadata {
                absolute_path: String::new(),
                relative_path: ".".into(),
                contents: matches_format::WorkspaceContents {
                    feature_tool: "none".into(),
                    feature_type: "sift".into(),
                    feature_options: serde_json::json!({}),
                    feature_prefix_dir: String::new(),
                },
            },
            timestamp: String::new(),
            image_count: 2,
            image_pair_count: None,
            match_count: None,
            cluster_count: Some(n_cl as u32),
            cluster_member_count: Some(m as u32),
            has_two_view_geometries: false,
            has_clusters: true,
            has_cluster_patches: with_patches,
        },
        content_hash: matches_format::MatchesContentHash {
            metadata_xxh128: String::new(),
            images_xxh128: String::new(),
            image_pairs_xxh128: None,
            clusters_xxh128: None,
            cluster_patches_xxh128: None,
            two_view_geometries_xxh128: None,
            content_xxh128: String::new(),
        },
        image_names: vec!["a.jpg".into(), "b.jpg".into()],
        feature_tool_hashes: vec![[0u8; 16]; 2],
        sift_content_hashes: vec![[1u8; 16]; 2],
        feature_counts: Array1::from_vec(vec![m as u32; 2]),
        image_dims: Some(Array2::from_shape_vec((2, 2), vec![640, 480, 640, 480]).expect("(2, 2)")),
        image_pairs: None,
        clusters: Some(ClustersData {
            cluster_starts: Array1::from_vec(starts.to_vec()),
            member_images: Array1::from_vec(vec![0u32; m]),
            member_features: Array1::from_vec((0..m as u32).collect()),
            member_positions: None,
            member_affine_shapes: Some(
                Array3::from_shape_vec((m, 2, 2), shapes.to_vec()).expect("(m, 2, 2)"),
            ),
            matcher_options: serde_json::json!({}),
        }),
        cluster_patches: with_patches.then(|| ClusterPatchData {
            reference_members: Array1::from_vec(vec![0u32; n_cl]),
            member_status: Array1::from_vec(vec![0u8; m]),
            member_zncc: Array1::from_vec(vec![1.0f32; m]),
            member_shift_px: Array1::from_vec(vec![0.0f32; m]),
            member_consistency_residual: Array1::from_vec(vec![f32::NAN; m]),
            refine_options: serde_json::json!({ "patch_size": patch_size }),
        }),
        two_view_geometries: None,
    }
}

#[test]
fn the_file_form_answers_what_the_arrays_form_answers() {
    let starts = [0u32, 2, 3];
    let shapes = flat(&[
        shape(0.2, 0.1, 0.0, 0.2),
        shape(0.9, 0.0, 0.3, 0.9),
        shape(1.0, 0.0, 0.0, 1.7),
    ]);
    // `patch_size` is the full patch edge, so the refine radius is half of it.
    let data = file(&starts, &shapes, 12.0, true);
    assert_eq!(
        cluster_radii_from_matches(&data).expect("readable"),
        cluster_radii(&starts, &shapes, 6.0)
    );
    assert_eq!(
        coarsest_clusters_from_matches(&data, 1).expect("readable"),
        coarsest_clusters(&starts, &shapes, 6.0, 1)
    );
}

#[test]
fn a_file_without_the_patch_section_is_refused_by_name() {
    let starts = [0u32, 1];
    let shapes = flat(&[shape(1.0, 0.0, 0.0, 1.0)]);
    assert_eq!(
        cluster_radii_from_matches(&file(&starts, &shapes, 12.0, false)),
        Err(ClusterRadiiError::NoClusterPatches)
    );
    assert_eq!(
        coarsest_clusters_from_matches(&file(&starts, &shapes, 12.0, false), 1),
        Err(ClusterRadiiError::NoClusterPatches)
    );

    // No cluster backbone at all is its own refusal.
    let mut pairwise = file(&starts, &shapes, 12.0, true);
    pairwise.clusters = None;
    assert_eq!(
        cluster_radii_from_matches(&pairwise),
        Err(ClusterRadiiError::NoClusters)
    );

    // A backbone without member geometry is a third.
    let mut no_shapes = file(&starts, &shapes, 12.0, true);
    no_shapes.clusters.as_mut().unwrap().member_affine_shapes = None;
    assert_eq!(
        cluster_radii_from_matches(&no_shapes),
        Err(ClusterRadiiError::NoAffineShapes)
    );

    // Refine options that name no scale are a fourth.
    let mut no_radius = file(&starts, &shapes, 12.0, true);
    no_radius.cluster_patches.as_mut().unwrap().refine_options = serde_json::json!({});
    assert_eq!(
        cluster_radii_from_matches(&no_radius),
        Err(ClusterRadiiError::NoRefineRadius)
    );
}

#[test]
fn a_nan_shape_never_displaces_a_cluster_that_has_an_extent() {
    let starts = [0u32, 1, 2];
    let shapes = flat(&[shape(f32::NAN, 0.0, 0.0, 1.0), shape(1.0, 0.0, 0.0, 1.0)]);
    // Cluster 0's only member has no extent, so the finite cluster is taken.
    assert_eq!(coarsest_clusters(&starts, &shapes, 6.0, 1), vec![1]);
}
