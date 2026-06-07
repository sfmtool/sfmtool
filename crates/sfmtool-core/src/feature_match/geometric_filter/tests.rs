use super::*;
use approx::assert_relative_eq;

fn test_intrinsics() -> Matrix3<f64> {
    let mut k = Matrix3::identity();
    k[(0, 0)] = 500.0;
    k[(1, 1)] = 500.0;
    k[(0, 2)] = 320.0;
    k[(1, 2)] = 240.0;
    k
}

#[test]
fn test_extract_affine_size_identity() {
    // Identity matrix: col0 = (1,0), col1 = (0,1), size = 0.5*(1+1) = 1
    let affine = [1.0, 0.0, 0.0, 1.0];
    assert_relative_eq!(extract_affine_size(&affine), 1.0, epsilon = 1e-10);
}

#[test]
fn test_extract_affine_size_scaled() {
    // 5*I: col0 = (5,0), col1 = (0,5), size = 5
    let affine = [5.0, 0.0, 0.0, 5.0];
    assert_relative_eq!(extract_affine_size(&affine), 5.0, epsilon = 1e-10);
}

#[test]
fn test_extract_affine_size_asymmetric() {
    // col0 = (3,4) len=5, col1 = (0,1) len=1, size = 0.5*(5+1) = 3
    let affine = [3.0, 0.0, 4.0, 1.0];
    assert_relative_eq!(extract_affine_size(&affine), 3.0, epsilon = 1e-10);
}

#[test]
fn test_orientation_consistency_aligned() {
    // Same affine → cosine = 1.0, should pass any threshold
    let affine1 = [3.0, 1.0, 4.0, 2.0];
    let candidates = [3.0, 1.0, 4.0, 2.0]; // identical
    let result = check_orientation_consistency_batch(&affine1, &candidates, 1, 0.95);
    assert!(result[0]);
}

#[test]
fn test_orientation_consistency_perpendicular() {
    // affine1 first column: (1, 0), candidate first column: (0, 1)
    // These are perpendicular → cosine = 0
    let affine1 = [1.0, 0.0, 0.0, 1.0];
    let candidates = [0.0, 0.0, 1.0, 0.0]; // first col = (0, 1)
    let result = check_orientation_consistency_batch(&affine1, &candidates, 1, 0.5);
    assert!(!result[0]);
}

#[test]
fn test_orientation_consistency_zero_vector() {
    let affine1 = [0.0, 0.0, 0.0, 0.0];
    let candidates = [1.0, 0.0, 0.0, 1.0];
    let result = check_orientation_consistency_batch(&affine1, &candidates, 1, 0.0);
    assert!(!result[0]);
}

#[test]
fn test_camera_geometry_construction() {
    let k = test_intrinsics();
    let r = Matrix3::identity();
    let t1 = Vector3::new(0.0, 0.0, 0.0);
    let t2 = Vector3::new(1.0, 0.0, 0.0);

    let geom = StereoPairGeometry::new(&k, &k, &r, &r, &t1, &t2);

    // Camera centers: C = -R^T @ t
    assert_relative_eq!(geom.cam1.center, Vector3::zeros(), epsilon = 1e-10);
    assert_relative_eq!(
        geom.cam2.center,
        Vector3::new(-1.0, 0.0, 0.0),
        epsilon = 1e-10
    );

    // R_2d should be identity upper-left 2×2 (since both R are identity)
    assert_relative_eq!(geom.r_2d[0], 1.0, epsilon = 1e-10);
    assert_relative_eq!(geom.r_2d[1], 0.0, epsilon = 1e-10);
    assert_relative_eq!(geom.r_2d[2], 0.0, epsilon = 1e-10);
    assert_relative_eq!(geom.r_2d[3], 1.0, epsilon = 1e-10);
}

#[test]
fn test_compute_ray_angle_cosine_same_point_same_camera() {
    let k = test_intrinsics();
    let r = Matrix3::identity();
    let t = Vector3::zeros();

    let geom = StereoPairGeometry::new(&k, &k, &r, &r, &t, &t);

    // Same point in same camera → parallel rays → cosine = 1.0
    let cos_angle = compute_ray_angle_cosine([320.0, 240.0], [320.0, 240.0], &geom);
    assert_relative_eq!(cos_angle, 1.0, epsilon = 1e-10);
}

#[test]
fn test_compute_ray_angle_cosine_lateral_baseline() {
    let k = test_intrinsics();
    let r = Matrix3::identity();
    let t1 = Vector3::zeros();
    let t2 = Vector3::new(1.0, 0.0, 0.0);

    let geom = StereoPairGeometry::new(&k, &k, &r, &r, &t1, &t2);

    // Center point in both images: ray goes along z-axis → parallel → cosine ≈ 1.0
    let cos_angle = compute_ray_angle_cosine([320.0, 240.0], [320.0, 240.0], &geom);
    assert_relative_eq!(cos_angle, 1.0, epsilon = 1e-10);

    // Different points should give cosine < 1.0
    let cos_angle2 = compute_ray_angle_cosine([320.0, 240.0], [400.0, 240.0], &geom);
    assert!(cos_angle2 < 1.0);
    assert!(cos_angle2 > 0.0);
}

#[test]
fn test_triangulate_point_dlt_basic() {
    let k = test_intrinsics();
    let r = Matrix3::identity();
    let t1 = Vector3::zeros();
    let t2 = Vector3::new(1.0, 0.0, 0.0);

    // A 3D point at (0, 0, 10) should project to the principal point in camera 1,
    // and slightly offset in camera 2 (because of the baseline).
    // P1 = K [I | 0], P2 = K [I | t2]
    // For X=(0,0,10): P1*X_h = K*(0,0,10,1)^T -> u=K*(0/10,0/10,1)=(320,240)
    // For X=(0,0,10): P2*X_h = K*(0+1, 0, 10)^T -> K*(1/10, 0, 1) = (500*0.1+320, 240) = (370, 240)
    let x1 = [320.0, 240.0];
    let x2 = [370.0, 240.0];

    let result = triangulate_point_dlt(x1, x2, &k, &k, &r, &t1, &r, &t2);
    assert!(result.is_some());
    let pt = result.unwrap();

    assert_relative_eq!(pt[0], 0.0, epsilon = 0.1);
    assert_relative_eq!(pt[1], 0.0, epsilon = 0.1);
    assert_relative_eq!(pt[2], 10.0, epsilon = 0.1);
}

#[test]
fn test_compute_depth_from_camera_basic() {
    let r = Matrix3::identity();
    let t = Vector3::zeros();
    let x_world = [0.0, 0.0, 10.0];
    let depth = compute_depth_from_camera(&x_world, &r, &t);
    assert_relative_eq!(depth, 10.0, epsilon = 1e-10);
}

#[test]
fn test_compute_depth_from_camera_with_translation() {
    let r = Matrix3::identity();
    let t = Vector3::new(0.0, 0.0, 5.0);
    let x_world = [0.0, 0.0, 10.0];
    // X_cam = R * X + t = (0,0,10) + (0,0,5) = (0,0,15)
    let depth = compute_depth_from_camera(&x_world, &r, &t);
    assert_relative_eq!(depth, 15.0, epsilon = 1e-10);
}

#[test]
fn test_two_stage_geometric_filter_empty() {
    let k = test_intrinsics();
    let r = Matrix3::identity();
    let t = Vector3::zeros();
    let geom = StereoPairGeometry::new(&k, &k, &r, &r, &t, &t);
    let config = GeometricFilterConfig::default();

    let mask = two_stage_geometric_filter(
        [320.0, 240.0],
        &[1.0, 0.0, 0.0, 1.0],
        &[],
        &[],
        0,
        &geom,
        &config,
    );
    assert!(mask.is_empty());
}

#[test]
fn test_two_stage_geometric_filter_consistent_match() {
    let k = test_intrinsics();
    let r = Matrix3::identity();
    let t1 = Vector3::zeros();
    let t2 = Vector3::new(1.0, 0.0, 0.0);
    let geom = StereoPairGeometry::new(&k, &k, &r, &r, &t1, &t2);
    let config = GeometricFilterConfig::default();

    // Query at image center with identity affine
    let x1 = [320.0, 240.0];
    let affine1 = [5.0, 0.0, 0.0, 5.0];

    // Candidate also near center with same-ish affine (consistent match)
    let candidate_pos = [370.0, 240.0];
    let candidate_aff = [5.0, 0.0, 0.0, 5.0];

    let mask = two_stage_geometric_filter(
        x1,
        &affine1,
        &candidate_pos,
        &candidate_aff,
        1,
        &geom,
        &config,
    );
    assert_eq!(mask.len(), 1);
    assert!(mask[0], "Consistent match should pass the filter");
}

#[test]
fn test_two_stage_geometric_filter_rejects_bad_orientation() {
    let k = test_intrinsics();
    let r = Matrix3::identity();
    let t1 = Vector3::zeros();
    let t2 = Vector3::new(1.0, 0.0, 0.0);
    let geom = StereoPairGeometry::new(&k, &k, &r, &r, &t1, &t2);
    let config = GeometricFilterConfig::default();

    // Query with affine whose first column points right: (5, 0)
    let x1 = [320.0, 240.0];
    let affine1 = [5.0, 0.0, 0.0, 5.0];

    // Candidate with first column pointing up: (0, 5) — perpendicular, should fail
    let candidate_pos = [370.0, 240.0];
    let candidate_aff = [0.0, 5.0, 5.0, 0.0]; // first col = (0, 5)

    let mask = two_stage_geometric_filter(
        x1,
        &affine1,
        &candidate_pos,
        &candidate_aff,
        1,
        &geom,
        &config,
    );
    assert_eq!(mask.len(), 1);
    assert!(
        !mask[0],
        "Perpendicular orientation should be rejected by Stage 1"
    );
}

#[test]
fn test_geometric_filter_config_custom() {
    let config = GeometricFilterConfig {
        max_angle_difference: 20.0,
        min_triangulation_angle: 3.0,
        geometric_size_ratio_min: 0.75,
        geometric_size_ratio_max: 1.333,
    };
    assert_relative_eq!(config.max_angle_difference, 20.0);
    assert_relative_eq!(config.geometric_size_ratio_min, 0.75);
}

#[test]
fn test_two_stage_geometric_filter_large_size_mismatch() {
    let k = test_intrinsics();
    let r = Matrix3::identity();
    let t1 = Vector3::zeros();
    let t2 = Vector3::new(1.0, 0.0, 0.0);
    let geom = StereoPairGeometry::new(&k, &k, &r, &r, &t1, &t2);
    let config = GeometricFilterConfig::default();

    // Query with size 5, candidate with size 25 (5x mismatch)
    let x1 = [320.0, 240.0];
    let affine1 = [5.0, 0.0, 0.0, 5.0]; // size = 5

    let candidate_pos = [370.0, 240.0];
    let candidate_aff = [25.0, 0.0, 0.0, 25.0]; // size = 25, ratio = 5.0

    let mask = two_stage_geometric_filter(
        x1,
        &affine1,
        &candidate_pos,
        &candidate_aff,
        1,
        &geom,
        &config,
    );
    assert_eq!(mask.len(), 1);
    // Lateral baseline with 50px horizontal disparity gives a meaningful
    // triangulation angle, so stage 2 (size check) applies.
    // Size ratio 5.0 >> max 1.25 → rejected.
    assert!(
        !mask[0],
        "5x size mismatch should be rejected with lateral baseline"
    );
}

#[test]
fn test_two_stage_geometric_filter_forward_motion_depth_change() {
    // Camera 1 at origin, camera 2 moved forward by 5m
    // Object at depth 10m from cam1 → depth 5m from cam2
    // Feature should appear ~2x larger in cam2
    let k = test_intrinsics();
    let r = Matrix3::identity();
    let t1 = Vector3::zeros();
    let t2 = Vector3::new(0.0, 0.0, 5.0); // forward motion
    let geom = StereoPairGeometry::new(&k, &k, &r, &r, &t1, &t2);
    let config = GeometricFilterConfig::default();

    // Object at (0, 0, 10) in world coords
    // Projects to principal point in cam1: (320, 240)
    // In cam2: X_cam2 = R*(0,0,10) + (0,0,5) = (0,0,15)
    // Projects to (320, 240) in cam2 as well
    let x1 = [320.0, 240.0];
    let affine1 = [5.0, 0.0, 0.0, 5.0];

    // Candidate at same pixel but 2x larger (depth halved)
    let candidate_pos = [320.0, 240.0];
    let candidate_aff = [10.0, 0.0, 0.0, 10.0];

    let mask = two_stage_geometric_filter(
        x1,
        &affine1,
        &candidate_pos,
        &candidate_aff,
        1,
        &geom,
        &config,
    );
    assert_eq!(mask.len(), 1);
    // For forward motion, rays are nearly parallel, so stage 2 (size check)
    // is typically skipped — candidate should pass
    assert!(
        mask[0],
        "Forward motion with nearly parallel rays should skip size check and accept"
    );
}

#[test]
fn test_two_stage_geometric_filter_strict_config() {
    let k = test_intrinsics();
    let r = Matrix3::identity();
    let t1 = Vector3::zeros();
    let t2 = Vector3::new(1.0, 0.0, 0.0);
    let geom = StereoPairGeometry::new(&k, &k, &r, &r, &t1, &t2);

    // Strict config: only 5 degrees of angle tolerance
    let config = GeometricFilterConfig {
        max_angle_difference: 5.0,
        min_triangulation_angle: 5.0,
        geometric_size_ratio_min: 0.9,
        geometric_size_ratio_max: 1.1,
    };

    // Query with identity affine, candidate slightly rotated (~10 degrees)
    let x1 = [320.0, 240.0];
    let affine1 = [5.0, 0.0, 0.0, 5.0];

    // Rotate first column by ~10 degrees: cos(10°) ≈ 0.985, sin(10°) ≈ 0.174
    let angle = 10.0_f64.to_radians();
    let candidate_pos = [370.0, 240.0];
    let candidate_aff = [5.0 * angle.cos(), 0.0, 5.0 * angle.sin(), 5.0];

    let mask = two_stage_geometric_filter(
        x1,
        &affine1,
        &candidate_pos,
        &candidate_aff,
        1,
        &geom,
        &config,
    );
    assert_eq!(mask.len(), 1);
    // 10 degree difference > 5 degree threshold → should be rejected
    assert!(
        !mask[0],
        "10° rotation should be rejected with 5° max angle tolerance"
    );
}

#[test]
fn test_stereo_pair_geometry_swapped() {
    let k = test_intrinsics();
    let r = Matrix3::identity();
    let t1 = Vector3::zeros();
    let t2 = Vector3::new(1.0, 0.0, 0.0);
    let geom = StereoPairGeometry::new(&k, &k, &r, &r, &t1, &t2);
    let swapped = geom.swapped();

    // Swapped cam1 should be original cam2
    assert_relative_eq!(swapped.cam1.center, geom.cam2.center, epsilon = 1e-10);
    assert_relative_eq!(swapped.cam2.center, geom.cam1.center, epsilon = 1e-10);

    // R_2d should be transposed
    assert_relative_eq!(swapped.r_2d[0], geom.r_2d[0], epsilon = 1e-10);
    assert_relative_eq!(swapped.r_2d[1], geom.r_2d[2], epsilon = 1e-10);
    assert_relative_eq!(swapped.r_2d[2], geom.r_2d[1], epsilon = 1e-10);
    assert_relative_eq!(swapped.r_2d[3], geom.r_2d[3], epsilon = 1e-10);
}

#[test]
fn test_extract_affine_size_zero() {
    let affine = [0.0, 0.0, 0.0, 0.0];
    assert_relative_eq!(extract_affine_size(&affine), 0.0, epsilon = 1e-10);
}

#[test]
fn test_orientation_consistency_batch_multiple() {
    // Test with 4 candidates: 2 aligned, 2 perpendicular
    let affine1 = [5.0, 0.0, 0.0, 5.0]; // first col = (5, 0)
    let candidates = [
        5.0, 0.0, 0.0, 5.0, // aligned (cosine ≈ 1)
        0.0, 5.0, 5.0, 0.0, // perpendicular (cosine ≈ 0)
        4.8, 0.0, 1.0, 5.0, // slightly rotated (cosine > 0.9)
        0.0, 5.0, -5.0, 0.0, // perpendicular (cosine ≈ 0)
    ];
    let result = check_orientation_consistency_batch(&affine1, &candidates, 4, 0.8);
    assert!(result[0], "Aligned should pass");
    assert!(!result[1], "Perpendicular should fail");
    assert!(result[2], "Slightly rotated should pass");
    assert!(!result[3], "Perpendicular should fail");
}

#[test]
fn test_two_stage_geometric_filter_multiple_candidates() {
    let k = test_intrinsics();
    let r = Matrix3::identity();
    let t1 = Vector3::zeros();
    let t2 = Vector3::new(1.0, 0.0, 0.0);
    let geom = StereoPairGeometry::new(&k, &k, &r, &r, &t1, &t2);
    let config = GeometricFilterConfig::default();

    let x1 = [320.0, 240.0];
    let affine1 = [5.0, 0.0, 0.0, 5.0];

    // 3 candidates: good, bad orientation, good
    let candidate_pos = [
        370.0, 240.0, // candidate 0 (good)
        370.0, 240.0, // candidate 1 (bad orientation)
        370.0, 240.0, // candidate 2 (good)
    ];
    let candidate_aff = [
        5.0, 0.0, 0.0, 5.0, // candidate 0: same orientation
        0.0, 5.0, 5.0, 0.0, // candidate 1: perpendicular
        4.5, 0.0, 0.0, 4.5, // candidate 2: similar orientation
    ];

    let mask = two_stage_geometric_filter(
        x1,
        &affine1,
        &candidate_pos,
        &candidate_aff,
        3,
        &geom,
        &config,
    );
    assert_eq!(mask.len(), 3);
    assert!(mask[0], "Candidate 0 should pass");
    assert!(!mask[1], "Candidate 1 should fail orientation");
    assert!(mask[2], "Candidate 2 should pass");
}
