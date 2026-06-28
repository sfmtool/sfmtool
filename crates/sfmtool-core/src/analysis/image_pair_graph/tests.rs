use super::*;
use approx::assert_relative_eq;
use std::f64::consts::PI;

/// Identity quaternion: w=1, x=0, y=0, z=0.
fn identity_quat() -> [f64; 4] {
    [1.0, 0.0, 0.0, 0.0]
}

/// Quaternion for 180-degree rotation around Y axis.
fn quat_180_around_y() -> [f64; 4] {
    // 180° around Y: w=cos(90°)=0, x=0, y=sin(90°)=1, z=0
    [0.0, 0.0, 1.0, 0.0]
}

#[test]
fn test_camera_directions_identity() {
    let quaternions = identity_quat();
    let dirs = compute_camera_directions(&quaternions, 1);

    assert_eq!(dirs.len(), 3);
    // Identity rotation: camera looks down -Z in camera space.
    // R_world_from_cam = I, so direction = -I[:, 2] = (0, 0, -1)
    assert_relative_eq!(dirs[0], 0.0, epsilon = 1e-10);
    assert_relative_eq!(dirs[1], 0.0, epsilon = 1e-10);
    assert_relative_eq!(dirs[2], -1.0, epsilon = 1e-10);
}

#[test]
fn test_camera_directions_rotated() {
    let quaternions = quat_180_around_y();
    let dirs = compute_camera_directions(&quaternions, 1);

    assert_eq!(dirs.len(), 3);
    // 180° around Y: the Z axis flips, so direction = (0, 0, 1)
    assert_relative_eq!(dirs[0], 0.0, epsilon = 1e-10);
    assert_relative_eq!(dirs[1], 0.0, epsilon = 1e-10);
    assert_relative_eq!(dirs[2], 1.0, epsilon = 1e-10);
}

#[test]
fn test_covisibility_simple() {
    // 3 images, all identity quaternions (all pointing same direction)
    let quaternions: Vec<f64> = [identity_quat(), identity_quat(), identity_quat()].concat();

    // Point 0 observed by images 0 and 1
    // Point 1 observed by images 0 and 2
    let track_point_indexes = [0_u32, 0, 1, 1];
    let track_image_indexes = [0_u32, 1, 0, 2];

    let pairs = build_covisibility_pairs(
        &quaternions,
        3,
        &track_point_indexes,
        &track_image_indexes,
        90.0,
    );

    assert_eq!(pairs.len(), 2);

    // Both pairs have count 1; check we have the right pairs
    let pair_set: std::collections::HashSet<(u32, u32)> =
        pairs.iter().map(|&(i, j, _)| (i, j)).collect();
    assert!(pair_set.contains(&(0, 1)));
    assert!(pair_set.contains(&(0, 2)));

    // All counts should be 1
    for &(_, _, count) in &pairs {
        assert_eq!(count, 1);
    }
}

#[test]
fn test_covisibility_angle_culling() {
    // Two cameras pointing opposite directions: identity and 180° around Y
    let quaternions: Vec<f64> = [identity_quat(), quat_180_around_y()].concat();

    // Point 0 observed by both images
    let track_point_indexes = [0_u32, 0];
    let track_image_indexes = [0_u32, 1];

    // Angle between directions is 180° which exceeds 90° threshold
    let pairs = build_covisibility_pairs(
        &quaternions,
        2,
        &track_point_indexes,
        &track_image_indexes,
        90.0,
    );

    assert!(
        pairs.is_empty(),
        "Cameras pointing opposite directions should be filtered at 90° threshold"
    );
}

#[test]
fn test_estimate_z_from_histogram() {
    // Uniform histogram: 4 bins each with 10 counts, min=0, max=4
    let hist_counts = [10_u32, 10, 10, 10];
    let z = estimate_z_from_histogram(&hist_counts, 0.0, 4.0, 50.0);

    // 50th percentile of uniform [0, 4] should be ~2.0
    // Bin width = 1.0, target count = 20 out of 40
    // cumsum = [10, 20, ...], bin_idx = 1, center = 0 + (1 + 0.5) * 1.0 = 1.5
    // This is the bin center which is an approximation
    assert_relative_eq!(z, 1.5, epsilon = 1e-10);
}

#[test]
fn test_estimate_z_from_histogram_empty() {
    let hist_counts = [0_u32, 0, 0, 0];
    let z = estimate_z_from_histogram(&hist_counts, 1.0, 5.0, 50.0);
    assert_relative_eq!(z, 3.0, epsilon = 1e-10);
}

#[test]
fn test_build_frustum_intersection_pairs_identical_cameras() {
    let num_images = 2;
    let quaternions: Vec<f64> = [identity_quat(), identity_quat()].concat();
    let translations = vec![0.0; num_images * 3]; // both at origin

    let fx_vals = vec![500.0; num_images];
    let fy_vals = vec![500.0; num_images];
    let cx_vals = vec![320.0; num_images];
    let cy_vals = vec![240.0; num_images];
    let widths = vec![640_u32; num_images];
    let heights = vec![480_u32; num_images];

    // Uniform depth histograms
    let num_bins = 10;
    let hist_counts: Vec<u32> = vec![10; num_images * num_bins];
    let histogram_min_z = vec![1.0; num_images];
    let histogram_max_z = vec![10.0; num_images];

    let pairs = build_frustum_intersection_pairs(
        &quaternions,
        &translations,
        num_images,
        &fx_vals,
        &fy_vals,
        &cx_vals,
        &cy_vals,
        &widths,
        &heights,
        &hist_counts,
        &histogram_min_z,
        &histogram_max_z,
        num_bins,
        5.0,
        95.0,
        1000,
        90.0,
        42,
    );

    assert_eq!(pairs.len(), 1, "Should find exactly one pair");
    assert_eq!(pairs[0].0, 0);
    assert_eq!(pairs[0].1, 1);

    // Identical cameras should have high intersection volume
    let volume = pairs[0].2;
    assert!(
        volume > 0.0,
        "Identical cameras should have positive intersection volume, got {}",
        volume
    );

    // Compare to analytical frustum volume
    // With 5th and 95th percentile on uniform [1, 10], near ~ bin center,
    // far ~ bin center. The intersection should be a large fraction of the volume.
    let expected_near = estimate_z_from_histogram(&hist_counts[..num_bins], 1.0, 10.0, 5.0);
    let expected_far = estimate_z_from_histogram(&hist_counts[..num_bins], 1.0, 10.0, 95.0);
    let full_volume =
        frustum::compute_frustum_volume(640, 480, 500.0, 500.0, expected_near, expected_far);

    // Monte Carlo estimate should be within 50% of full volume for identical cameras
    assert!(
        volume > full_volume * 0.5,
        "Intersection volume ({:.4}) should be at least 50% of full volume ({:.4})",
        volume,
        full_volume
    );
}

#[test]
fn test_build_frustum_intersection_pairs_nan_skipped() {
    let num_images = 2;
    let quaternions: Vec<f64> = [identity_quat(), identity_quat()].concat();
    let translations = vec![0.0; num_images * 3];

    let fx_vals = vec![500.0; num_images];
    let fy_vals = vec![500.0; num_images];
    let cx_vals = vec![320.0; num_images];
    let cy_vals = vec![240.0; num_images];
    let widths = vec![640_u32; num_images];
    let heights = vec![480_u32; num_images];

    let num_bins = 4;
    let hist_counts = vec![10_u32; num_images * num_bins];
    // First image has NAN min_z -> should be skipped
    let histogram_min_z = vec![f64::NAN, 1.0];
    let histogram_max_z = vec![f64::NAN, 10.0];

    let pairs = build_frustum_intersection_pairs(
        &quaternions,
        &translations,
        num_images,
        &fx_vals,
        &fy_vals,
        &cx_vals,
        &cy_vals,
        &widths,
        &heights,
        &hist_counts,
        &histogram_min_z,
        &histogram_max_z,
        num_bins,
        5.0,
        95.0,
        100,
        90.0,
        42,
    );

    assert!(
        pairs.is_empty(),
        "Should find no pairs when one image has NAN depth stats"
    );
}

#[test]
fn test_camera_directions_multiple() {
    // Test with multiple images
    let quaternions: Vec<f64> = [identity_quat(), identity_quat(), quat_180_around_y()].concat();

    let dirs = compute_camera_directions(&quaternions, 3);
    assert_eq!(dirs.len(), 9);

    // First two should be (0, 0, -1)
    assert_relative_eq!(dirs[2], -1.0, epsilon = 1e-10);
    assert_relative_eq!(dirs[5], -1.0, epsilon = 1e-10);

    // Third should be (0, 0, 1)
    assert_relative_eq!(dirs[8], 1.0, epsilon = 1e-10);
}

#[test]
fn test_camera_directions_90_around_y() {
    // 90° rotation around Y axis: w=cos(45°), x=0, y=sin(45°), z=0
    let half = (PI / 4.0).cos();
    let quaternions = [half, 0.0, half, 0.0];

    let dirs = compute_camera_directions(&quaternions, 1);

    // R_cam_from_world = R_y(90°), R_world_from_cam = R_y(90°)^T
    // direction = -col2(R_world_from_cam) = (1, 0, 0)
    // Verified against Python numpy-quaternion implementation.
    assert_relative_eq!(dirs[0], 1.0, epsilon = 1e-10);
    assert_relative_eq!(dirs[1], 0.0, epsilon = 1e-10);
    assert_relative_eq!(dirs[2], 0.0, epsilon = 1e-10);
}
