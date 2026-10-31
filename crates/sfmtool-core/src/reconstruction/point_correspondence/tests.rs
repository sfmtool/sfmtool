// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[test]
fn test_basic_correspondences() {
    // Source reconstruction: image 0 has features 10,20; image 1 has feature 30
    let src_img = [0, 0, 1];
    let src_feat = [10, 20, 30];
    let src_pts = [100, 200, 300];

    // Target reconstruction: image 0 has features 10,20; image 1 has feature 30
    let tgt_img = [0, 0, 1];
    let tgt_feat = [10, 20, 30];
    let tgt_pts = [500, 600, 700];

    // Shared images: source img 0 <-> target img 0, source img 1 <-> target img 1
    let shared_src = [0, 1];
    let shared_tgt = [0, 1];

    let result = find_point_correspondences(
        &src_img,
        &src_feat,
        &src_pts,
        &tgt_img,
        &tgt_feat,
        &tgt_pts,
        &shared_src,
        &shared_tgt,
    );

    assert_eq!(result.source_ids.len(), 3);
    assert_eq!(result.target_ids.len(), 3);

    // Verify all correspondences are present (order may vary due to HashMap)
    let corr: HashMap<u32, u32> = result
        .source_ids
        .iter()
        .zip(result.target_ids.iter())
        .map(|(&s, &t)| (s, t))
        .collect();
    assert_eq!(corr[&100], 500);
    assert_eq!(corr[&200], 600);
    assert_eq!(corr[&300], 700);
}

#[test]
fn test_no_shared_features() {
    // Source has feature 10 in image 0
    let src_img = [0];
    let src_feat = [10];
    let src_pts = [100];

    // Target has feature 20 in image 0 (different feature index)
    let tgt_img = [0];
    let tgt_feat = [20];
    let tgt_pts = [500];

    let shared_src = [0];
    let shared_tgt = [0];

    let result = find_point_correspondences(
        &src_img,
        &src_feat,
        &src_pts,
        &tgt_img,
        &tgt_feat,
        &tgt_pts,
        &shared_src,
        &shared_tgt,
    );

    assert!(result.source_ids.is_empty());
    assert!(result.target_ids.is_empty());
}

#[test]
fn test_deduplication_first_occurrence_wins() {
    // Source: point 100 is observed as feature 10 in both image 0 and image 1
    let src_img = [0, 1];
    let src_feat = [10, 10];
    let src_pts = [100, 100];

    // Target: feature 10 maps to point 500 in image 0, point 600 in image 1
    let tgt_img = [0, 1];
    let tgt_feat = [10, 10];
    let tgt_pts = [500, 600];

    // Both pairs are shared
    let shared_src = [0, 1];
    let shared_tgt = [0, 1];

    let result = find_point_correspondences(
        &src_img,
        &src_feat,
        &src_pts,
        &tgt_img,
        &tgt_feat,
        &tgt_pts,
        &shared_src,
        &shared_tgt,
    );

    // Source point 100 should appear only once
    assert_eq!(result.source_ids.len(), 1);
    assert_eq!(result.source_ids[0], 100);
    // First occurrence wins: mapped to target point 500 (from first shared pair)
    assert_eq!(result.target_ids[0], 500);
}

#[test]
fn test_missing_shared_image() {
    // Source has observations only in image 0
    let src_img = [0];
    let src_feat = [10];
    let src_pts = [100];

    // Target has observations only in image 0
    let tgt_img = [0];
    let tgt_feat = [10];
    let tgt_pts = [500];

    // Shared images reference image 5 which doesn't exist in either
    let shared_src = [5];
    let shared_tgt = [5];

    let result = find_point_correspondences(
        &src_img,
        &src_feat,
        &src_pts,
        &tgt_img,
        &tgt_feat,
        &tgt_pts,
        &shared_src,
        &shared_tgt,
    );

    assert!(result.source_ids.is_empty());
}

#[test]
fn test_merge_rejects_embedded_patches() {
    use crate::reconstruction::ObservationSource;

    // Merge keys on (image, feature_index), which embedded_patches lacks, so
    // the guard must refuse it up front rather than substitute a placeholder.
    let mut recon = SfmrReconstruction::demo(2);
    let n_obs = recon.point_set.tracks.len();
    let n_img = recon.image_set.images.len();
    recon.point_set.observations = ObservationSource::EmbeddedPatches {
        keypoints_xy: ndarray::Array2::zeros((n_obs, 2)),
        image_file_hashes: vec![[0u8; 16]; n_img],
    };

    let mapping = vec![HashMap::new()];
    // (MergedPointsAndTracks is not Debug, so match rather than expect_err.)
    let Err(err) = merge_points_and_tracks(&[&recon], &[], &mapping) else {
        panic!("merge must reject embedded_patches");
    };
    assert!(
        err.contains("embedded_patches"),
        "unexpected error message: {err}"
    );
}

/// Two copies of a four-point demo reconstruction whose points share every
/// observation, with the identity image mapping for each.
fn two_demo_copies() -> (
    SfmrReconstruction,
    SfmrReconstruction,
    Vec<HashMap<u32, u32>>,
) {
    let a = SfmrReconstruction::demo(4);
    let b = a.clone();
    let identity: HashMap<u32, u32> = (0..a.image_set.images.len() as u32)
        .map(|i| (i, i))
        .collect();
    (a, b, vec![identity.clone(), identity])
}

fn set_at_infinity(recon: &mut SfmrReconstruction, point_id: usize, direction: [f64; 3]) {
    let point = &mut recon.point_set.points[point_id];
    point.position = nalgebra::Point3::new(direction[0], direction[1], direction[2]);
    point.w = 0.0;
}

/// The merged position of the point that holds the observation of feature
/// `feature` in image `image`.
fn merged_xyzw(merged: &MergedPointsAndTracks, image: u32, feature: u32) -> [f64; 4] {
    let row = (0..merged.track_image_indexes.len())
        .find(|&j| {
            merged.track_image_indexes[j] == image && merged.track_feature_indexes[j] == feature
        })
        .expect("the observation is in the merged tracks");
    merged.positions_xyzw[merged.track_point_indexes[row] as usize]
}

#[test]
fn test_merge_keeps_points_at_infinity() {
    let (mut a, mut b, mapping) = two_demo_copies();
    // Demo point i is feature i in images i and i + 1, in both copies.
    // Point 0: finite in `a`, at infinity in `b`.
    set_at_infinity(&mut b, 0, [0.0, 0.0, 1.0]);
    // Point 1: at infinity in both, along different directions.
    set_at_infinity(&mut a, 1, [1.0, 0.0, 0.0]);
    set_at_infinity(&mut b, 1, [0.0, 1.0, 0.0]);
    // Point 2: at infinity in both, along opposite directions.
    set_at_infinity(&mut a, 2, [1.0, 0.0, 0.0]);
    set_at_infinity(&mut b, 2, [-1.0, 0.0, 0.0]);
    let finite_0 = a.point_set.points[0].position;
    let finite_3 = a.point_set.points[3].position;

    let Ok(merged) = merge_points_and_tracks(&[&a, &b], &[], &mapping) else {
        panic!("merge must succeed");
    };

    assert_eq!(merged.positions_xyzw.len(), 4);
    // A finite point and a point at infinity merge into the finite point.
    assert_eq!(
        merged_xyzw(&merged, 0, 0),
        [finite_0.x, finite_0.y, finite_0.z, 1.0]
    );
    // Points at infinity merge along the normalised sum of their directions.
    let h = std::f64::consts::FRAC_1_SQRT_2;
    let p1 = merged_xyzw(&merged, 1, 1);
    assert_eq!(p1[3], 0.0);
    for (got, want) in p1[..3].iter().zip([h, h, 0.0]) {
        assert!((got - want).abs() < 1e-12, "{p1:?}");
    }
    // Directions that cancel fall back to the first point's direction.
    assert_eq!(merged_xyzw(&merged, 2, 2), [1.0, 0.0, 0.0, 0.0]);
    // Two finite copies stay where they are.
    assert_eq!(
        merged_xyzw(&merged, 3, 3),
        [finite_3.x, finite_3.y, finite_3.z, 1.0]
    );
}

#[test]
fn test_merge_group_averages_only_its_finite_points() {
    let (a, mut b, mapping) = two_demo_copies();
    let mut c = a.clone();
    // Point 0 is finite in `a`, at infinity in `b`, and finite but moved in `c`.
    set_at_infinity(&mut b, 0, [0.0, 0.0, 1.0]);
    c.point_set.points[0].position.x += 2.0;
    let mapping = vec![mapping[0].clone(), mapping[1].clone(), mapping[0].clone()];
    let p = a.point_set.points[0].position;

    let groups = vec![vec![(0usize, 0u32), (1, 0), (2, 0)]];
    let Ok(merged) = merge_points_and_tracks(&[&a, &b, &c], &groups, &mapping) else {
        panic!("merge must succeed");
    };

    let got = merged_xyzw(&merged, 0, 0);
    assert_eq!(got[3], 1.0);
    for (got, want) in got[..3].iter().zip([p.x + 1.0, p.y, p.z]) {
        assert!((got - want).abs() < 1e-12);
    }
}
