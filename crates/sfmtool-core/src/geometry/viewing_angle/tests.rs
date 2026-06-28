use super::*;
use std::f64::consts::PI;

/// Helper: create identity quaternion (w=1, x=0, y=0, z=0)
fn identity_quat() -> [f64; 4] {
    [1.0, 0.0, 0.0, 0.0]
}

#[test]
fn test_cameras_at_same_position_filtered() {
    // Two cameras at the same position looking at a point - 0 angle, should be filtered
    let quaternions = [identity_quat(), identity_quat()].concat();
    let translations = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let positions = [0.0, 0.0, 5.0]; // point in front
    let track_point_indexes = [0_u32, 0];
    let track_image_indexes = [0_u32, 1];

    let keep = compute_narrow_track_mask(
        &quaternions,
        &translations,
        2,
        &positions,
        1,
        &track_point_indexes,
        &track_image_indexes,
        0.1, // any positive threshold
    );

    assert_eq!(keep.len(), 1);
    assert!(
        !keep[0],
        "Point observed from same position should be filtered"
    );
}

#[test]
fn test_cameras_90_degrees_apart_kept() {
    // Camera 0: at origin, identity rotation -> center at (0,0,0)
    // Camera 1: rotated 90° around Y, translated so center is at (5,0,0)
    // Point at (0,0,5)
    //
    // Ray from cam0: (0,0,5) - (0,0,0) = (0,0,1)
    // Ray from cam1: (0,0,5) - (5,0,0) = (-5,0,5), normalized ~ (-0.707, 0, 0.707)
    // Dot product: 0.707, angle ~ 45°
    //
    // Use identity for both cameras but place them apart via translation
    // Camera center = -R^T @ t, so t = -R @ center
    // Cam0: center=(0,0,0), R=I, t=(0,0,0)
    // Cam1: center=(5,0,0), R=I, t=(-5,0,0)

    let quaternions = [identity_quat(), identity_quat()].concat();
    let translations = [0.0, 0.0, 0.0, -5.0, 0.0, 0.0];
    let positions = [0.0, 0.0, 5.0];
    let track_point_indexes = [0_u32, 0];
    let track_image_indexes = [0_u32, 1];

    // Threshold at 30° (pi/6) - the actual angle is ~45°, so point should be kept
    let keep = compute_narrow_track_mask(
        &quaternions,
        &translations,
        2,
        &positions,
        1,
        &track_point_indexes,
        &track_image_indexes,
        PI / 6.0,
    );

    assert!(
        keep[0],
        "Point with ~45° angle should be kept with 30° threshold"
    );

    // Threshold at 80° - the actual angle is ~45°, so point should be filtered
    let keep2 = compute_narrow_track_mask(
        &quaternions,
        &translations,
        2,
        &positions,
        1,
        &track_point_indexes,
        &track_image_indexes,
        80.0_f64.to_radians(),
    );

    assert!(
        !keep2[0],
        "Point with ~45° angle should be filtered with 80° threshold"
    );
}

#[test]
fn test_single_observation_removed() {
    let quaternions = identity_quat().to_vec();
    let translations = [0.0, 0.0, 0.0];
    let positions = [1.0, 2.0, 3.0];
    let track_point_indexes = [0_u32];
    let track_image_indexes = [0_u32];

    let keep = compute_narrow_track_mask(
        &quaternions,
        &translations,
        1,
        &positions,
        1,
        &track_point_indexes,
        &track_image_indexes,
        0.01,
    );

    assert!(
        !keep[0],
        "Point with single observation should always be removed"
    );
}

#[test]
fn test_angle_threshold_boundary() {
    // Verify keep/remove decisions are correct just below and above the
    // actual viewing angle.
    // Set up cameras with known geometry
    let quaternions = [identity_quat(), identity_quat()].concat();
    let translations = [0.0, 0.0, 0.0, -10.0, 0.0, 0.0];
    let positions = [0.0, 0.0, 10.0];
    let track_point_indexes = [0_u32, 0];
    let track_image_indexes = [0_u32, 1];

    // The actual angle: atan2(10, 10) = 45°
    // Test at thresholds just below and above
    let threshold_below = 44.0_f64.to_radians();
    let threshold_above = 46.0_f64.to_radians();

    let keep_below = compute_narrow_track_mask(
        &quaternions,
        &translations,
        2,
        &positions,
        1,
        &track_point_indexes,
        &track_image_indexes,
        threshold_below,
    );
    let keep_above = compute_narrow_track_mask(
        &quaternions,
        &translations,
        2,
        &positions,
        1,
        &track_point_indexes,
        &track_image_indexes,
        threshold_above,
    );

    assert!(
        keep_below[0],
        "Should keep with threshold below actual angle"
    );
    assert!(
        !keep_above[0],
        "Should filter with threshold above actual angle"
    );
}

#[test]
fn test_multiple_points_mixed() {
    // 3 cameras, 3 points with different observation patterns
    let quaternions = [identity_quat(), identity_quat(), identity_quat()].concat();
    // Cam0 at origin, Cam1 at (10,0,0), Cam2 at (0,10,0)
    let translations = [0.0, 0.0, 0.0, -10.0, 0.0, 0.0, 0.0, -10.0, 0.0];
    // Point 0 at (0,0,1) - very close, large angle from cam0+cam1
    // Point 1 at (0,0,100) - very far, small angle
    // Point 2 at (5,5,5) - moderate
    let positions = [0.0, 0.0, 1.0, 0.0, 0.0, 100.0, 5.0, 5.0, 5.0];

    let track_point_indexes = [0_u32, 0, 1, 1, 2, 2];
    let track_image_indexes = [0_u32, 1, 0, 1, 0, 2];

    let keep = compute_narrow_track_mask(
        &quaternions,
        &translations,
        3,
        &positions,
        3,
        &track_point_indexes,
        &track_image_indexes,
        10.0_f64.to_radians(),
    );

    assert_eq!(keep.len(), 3);
    // Point 0: angle ~ atan2(10, 1) ~ 84° - should be kept
    assert!(keep[0]);
    // Point 1: angle ~ atan2(10, 100) ~ 5.7° - should be filtered
    assert!(!keep[1]);
}
