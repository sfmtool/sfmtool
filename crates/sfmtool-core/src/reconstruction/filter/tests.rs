use super::*;

#[test]
fn test_basic_filtering() {
    // 5 points, keep points 0, 2, 4
    let mask = [true, false, true, false, true];
    // 6 track observations
    let track_image_indexes = [0u32, 1, 0, 1, 2, 2];
    let track_feature_indexes = [10u32, 20, 30, 40, 50, 60];
    let track_point_indexes = [0u32, 1, 2, 3, 4, 0];

    let result = filter_tracks_by_point_mask(
        &mask,
        &track_image_indexes,
        &track_feature_indexes,
        &track_point_indexes,
    );

    // Should keep observations for points 0, 2, 4 (indices 0, 2, 4, 5)
    assert_eq!(result.track_image_indexes, vec![0, 0, 2, 2]);
    assert_eq!(result.track_feature_indexes, vec![10, 30, 50, 60]);
    // Point 0 -> 0, point 2 -> 1, point 4 -> 2
    assert_eq!(result.track_point_indexes, vec![0, 1, 2, 0]);
}

#[test]
fn test_keep_all() {
    let mask = [true, true, true];
    let track_image_indexes = [0u32, 1, 2];
    let track_feature_indexes = [10u32, 20, 30];
    let track_point_indexes = [0u32, 1, 2];

    let result = filter_tracks_by_point_mask(
        &mask,
        &track_image_indexes,
        &track_feature_indexes,
        &track_point_indexes,
    );

    assert_eq!(result.track_image_indexes, vec![0, 1, 2]);
    assert_eq!(result.track_feature_indexes, vec![10, 20, 30]);
    assert_eq!(result.track_point_indexes, vec![0, 1, 2]);
}

#[test]
fn test_keep_none() {
    let mask = [false, false, false];
    let track_image_indexes = [0u32, 1, 2];
    let track_feature_indexes = [10u32, 20, 30];
    let track_point_indexes = [0u32, 1, 2];

    let result = filter_tracks_by_point_mask(
        &mask,
        &track_image_indexes,
        &track_feature_indexes,
        &track_point_indexes,
    );

    assert!(result.track_image_indexes.is_empty());
    assert!(result.track_feature_indexes.is_empty());
    assert!(result.track_point_indexes.is_empty());
}

#[test]
fn test_remapping_is_contiguous() {
    // Keep only points 1 and 3 out of 5
    let mask = [false, true, false, true, false];
    let track_image_indexes = [0u32, 1, 2, 3];
    let track_feature_indexes = [10u32, 20, 30, 40];
    let track_point_indexes = [1u32, 3, 1, 3];

    let result = filter_tracks_by_point_mask(
        &mask,
        &track_image_indexes,
        &track_feature_indexes,
        &track_point_indexes,
    );

    // All observations kept (all reference points 1 or 3)
    assert_eq!(result.track_image_indexes.len(), 4);
    // Point 1 -> 0, point 3 -> 1
    assert_eq!(result.track_point_indexes, vec![0, 1, 0, 1]);
}
