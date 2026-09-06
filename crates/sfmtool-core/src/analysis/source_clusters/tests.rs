// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use super::*;

/// A selection row: `(image, feature, scale)`. The shape is a scaled identity,
/// so the row's radius is `refine_radius * scale`.
///
/// Every scale a test pins a radius on is a binary fraction: the reading
/// narrows the selection's `f64` shapes to its own `f32`, so an exactly
/// representable scale keeps the expected radius exact at both widths.
fn rows(spec: &[(u32, u32, f64)]) -> (Vec<u32>, Vec<u32>, Vec<f64>, Vec<f64>) {
    let mut img = Vec::new();
    let mut feat = Vec::new();
    let mut pos = Vec::new();
    let mut shp = Vec::new();
    for (k, &(i, f, s)) in spec.iter().enumerate() {
        img.push(i);
        feat.push(f);
        // Shape [[s, 0], [0, s]] with a distinguishable pixel per row.
        pos.extend_from_slice(&[k as f64, 10.0 + k as f64]);
        shp.extend_from_slice(&[s, 0.0, 0.0, s]);
    }
    (img, feat, pos, shp)
}

fn selection<'a>(
    starts: &'a [u32],
    img: &'a [u32],
    feat: &'a [u32],
    pos: &'a [f64],
    shp: &'a [f64],
    n_images: usize,
) -> SourceSelection<'a> {
    SourceSelection {
        cluster_starts: starts,
        member_images: img,
        member_features: feat,
        member_positions: pos,
        member_affine_shapes: shp,
        refine_radius: 6.0,
        n_images,
    }
}

/// Bands one octave apart, anchored at the floor, with the top band open.
const EDGES: [f64; 4] = [f64::INFINITY, 1.0, 0.5, 0.25];

#[test]
fn a_cluster_the_member_holds_is_not_a_candidate() {
    // Cluster 0 is the member's own; clusters 1 and 2 are not.
    let (img, feat, pos, shp) = rows(&[
        (0, 10, 1.0),
        (1, 11, 1.0),
        (0, 20, 0.5),
        (1, 21, 0.5),
        (0, 30, 0.25),
        (1, 31, 0.25),
    ]);
    let starts = [0u32, 2, 4, 6];
    let out = source_clusters(
        selection(&starts, &img, &feat, &pos, &shp, 2),
        MemberIdentity {
            obs_image: &[0, 1],
            obs_feature: &[10, 11],
        },
        &[0, 1],
        &EDGES,
    );
    assert_eq!(out.n_file_clusters, 3);
    assert_eq!(out.n_rows_matched, 2);
    assert_eq!(out.admission_radius, vec![6.0]);
    assert_eq!(out.admission_floor_px, 6.0);
    assert_eq!(out.candidates, vec![1, 2]);
    assert_eq!(out.candidate_radius, vec![3.0, 1.5]);
}

#[test]
fn a_cluster_only_one_placed_frame_sees_is_not_a_candidate() {
    let (img, feat, pos, shp) = rows(&[
        (0, 10, 1.0),
        (1, 11, 1.0),
        // Cluster 1 lives entirely on image 2, which is not placed.
        (2, 20, 0.5),
        (2, 21, 0.5),
        // Cluster 2 has one placed frame and one unplaced.
        (0, 30, 0.5),
        (2, 31, 0.5),
    ]);
    let starts = [0u32, 2, 4, 6];
    let out = source_clusters(
        selection(&starts, &img, &feat, &pos, &shp, 3),
        MemberIdentity {
            obs_image: &[0, 1],
            obs_feature: &[10, 11],
        },
        &[0, 1],
        &EDGES,
    );
    assert!(out.candidates.is_empty());
    assert!(out.obs_cluster.is_empty());
}

#[test]
fn the_selected_rows_are_the_candidates_own_rows_on_placed_frames() {
    let (img, feat, pos, shp) = rows(&[
        (0, 10, 1.0),
        (1, 11, 1.0),
        (0, 20, 0.5),
        (1, 21, 0.5),
        (2, 22, 0.5),
    ]);
    let starts = [0u32, 2, 5];
    let out = source_clusters(
        selection(&starts, &img, &feat, &pos, &shp, 3),
        MemberIdentity {
            obs_image: &[0, 1],
            obs_feature: &[10, 11],
        },
        &[0, 1],
        &EDGES,
    );
    assert_eq!(out.candidates, vec![1]);
    // Row 4 sits on image 2, which is not placed, so it is not selected.
    assert_eq!(out.obs_cluster, vec![1, 1]);
    assert_eq!(out.obs_image, vec![0, 1]);
    assert_eq!(out.obs_feature, vec![20, 21]);
    // Rows carry their own pixel, the affine's last column.
    assert_eq!(out.obs_uv, vec![2.0, 12.0, 3.0, 13.0]);
    assert_eq!(out.obs_shape, vec![0.5, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.5]);
}

#[test]
fn a_clusters_radius_is_its_widest_members() {
    let (img, feat, pos, shp) = rows(&[(0, 10, 1.0), (1, 11, 1.0), (0, 20, 0.25), (1, 21, 0.75)]);
    let starts = [0u32, 2, 4];
    let out = source_clusters(
        selection(&starts, &img, &feat, &pos, &shp, 2),
        MemberIdentity {
            obs_image: &[0, 1],
            obs_feature: &[10, 11],
        },
        &[0, 1],
        &EDGES,
    );
    assert_eq!(out.candidate_radius, vec![6.0 * 0.75]);
}

#[test]
fn the_bands_are_half_open_and_the_top_one_is_open_above_the_floor() {
    // Floor 1.0: a radius at the floor sits in band 0, one just under it in
    // band 1, one above the floor also in band 0, and one under the last edge
    // in no band at all.
    let radius = [1.0, 0.999, 4.0, 0.5, 0.4999, 0.2];
    let bands = assign_bands(&radius, 1.0, &EDGES);
    assert_eq!(bands, vec![0, 1, 0, 1, 2, -1]);
}

#[test]
fn a_member_row_takes_the_first_selection_row_carrying_its_key() {
    // The same (image, feature) appears in cluster 0 and again in cluster 1.
    // The member's row admits the first, and cluster 1 stays a candidate.
    let (img, feat, pos, shp) = rows(&[(0, 10, 1.0), (1, 11, 1.0), (0, 10, 0.5), (1, 12, 0.5)]);
    let starts = [0u32, 2, 4];
    let out = source_clusters(
        selection(&starts, &img, &feat, &pos, &shp, 2),
        MemberIdentity {
            obs_image: &[0],
            obs_feature: &[10],
        },
        &[0, 1],
        &EDGES,
    );
    assert_eq!(out.n_rows_matched, 1);
    assert_eq!(out.admission_radius, vec![6.0]);
    assert_eq!(out.candidates, vec![1]);
}

#[test]
fn a_member_row_the_selection_does_not_carry_matches_nothing() {
    let (img, feat, pos, shp) = rows(&[(0, 10, 1.0), (1, 11, 1.0)]);
    let starts = [0u32, 2];
    let out = source_clusters(
        selection(&starts, &img, &feat, &pos, &shp, 2),
        MemberIdentity {
            obs_image: &[0, 0],
            obs_feature: &[10, 99],
        },
        &[0, 1],
        &EDGES,
    );
    assert_eq!(out.n_rows_matched, 1);
}

#[test]
fn an_empty_admission_has_no_floor() {
    let (img, feat, pos, shp) = rows(&[(0, 10, 1.0), (1, 11, 1.0)]);
    let starts = [0u32, 2];
    let out = source_clusters(
        selection(&starts, &img, &feat, &pos, &shp, 2),
        MemberIdentity {
            obs_image: &[],
            obs_feature: &[],
        },
        &[0, 1],
        &EDGES,
    );
    assert!(out.admission_radius.is_empty());
    assert!(out.admission_floor_px.is_nan());
    assert_eq!(out.candidates, vec![0]);
    // Every band comparison against a NaN floor is false, so nothing lands in
    // a band and the caller sees the refusal in the floor itself.
    assert_eq!(out.candidate_band, vec![-1]);
}

#[test]
fn the_join_repeats_itself() {
    let (img, feat, pos, shp) = rows(&[
        (0, 10, 1.0),
        (1, 11, 1.0),
        (0, 20, 0.5),
        (1, 21, 0.5),
        (0, 30, 0.3),
        (1, 31, 0.3),
    ]);
    let starts = [0u32, 2, 4, 6];
    let sel = selection(&starts, &img, &feat, &pos, &shp, 2);
    let m = MemberIdentity {
        obs_image: &[0, 1],
        obs_feature: &[10, 11],
    };
    assert_eq!(
        source_clusters(sel, m, &[0, 1], &EDGES),
        source_clusters(sel, m, &[0, 1], &EDGES)
    );
}
