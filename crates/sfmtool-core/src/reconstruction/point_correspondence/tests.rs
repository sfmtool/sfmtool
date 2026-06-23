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
    let n_obs = recon.tracks.len();
    let n_img = recon.images.len();
    recon.observations = ObservationSource::EmbeddedPatches {
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
