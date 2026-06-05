use super::*;

#[test]
fn test_cartesian_to_polar_basic() {
    // Point at (10, 0) relative to epipole at (0, 0)
    let points = [10.0, 0.0, 0.0, 10.0, -5.0, 0.0];
    let (theta, radius, valid) = cartesian_to_polar(&points, 3, [0.0, 0.0], 1.0);

    assert_eq!(valid.len(), 3);
    assert!((theta[0] - 0.0).abs() < 1e-10); // right
    assert!((theta[1] - PI / 2.0).abs() < 1e-10); // up
    assert!((theta[2] - PI).abs() < 1e-10); // left
    assert!((radius[0] - 10.0).abs() < 1e-10);
    assert!((radius[1] - 10.0).abs() < 1e-10);
    assert!((radius[2] - 5.0).abs() < 1e-10);
}

#[test]
fn test_cartesian_to_polar_min_radius() {
    // Point too close to epipole should be filtered
    let points = [1.0, 1.0, 100.0, 100.0];
    let (_, _, valid) = cartesian_to_polar(&points, 2, [0.0, 0.0], 10.0);
    assert_eq!(valid.len(), 1);
    assert_eq!(valid[0], 1);
}

#[test]
fn test_extend_for_wraparound() {
    // Features near boundaries should be duplicated based on angular threshold
    let theta = vec![-3.0, -2.0, 0.0, 2.0, 3.0];
    let desc_len = 4;
    let descs: Vec<u8> = (0..20).collect();

    // angular_threshold = min(π/4, 2/5 * 2π) = min(0.785, 2.513) = 0.785
    // near +π: theta > π - 0.785 ≈ 2.356 → only theta=3.0
    // near -π: theta < -π + 0.785 ≈ -2.356 → only theta=-3.0
    let (ext_theta, ext_descs, orig_len, n_prep) =
        extend_for_wraparound(&theta, &descs, desc_len, 2);

    assert_eq!(orig_len, 5);
    // Only theta=3.0 is near +π → prepended with θ-2π
    assert_eq!(n_prep, 1);
    // Only theta=-3.0 is near -π → appended with θ+2π
    assert_eq!(ext_theta.len(), 5 + 1 + 1);
    assert!((ext_theta[0] - (3.0 - 2.0 * PI)).abs() < 1e-10);
    assert!((ext_theta[6] - (-3.0 + 2.0 * PI)).abs() < 1e-10);
    // Extended descriptors should have correct length
    assert_eq!(ext_descs.len(), ext_theta.len() * desc_len);
}

#[test]
fn test_polar_match_self() {
    // Match features against themselves — should get identity matches
    let n = 10;
    let theta: Vec<f64> = (0..n)
        .map(|i| -PI + 2.0 * PI * i as f64 / n as f64)
        .collect();

    let desc_len = 128;
    let mut descs = vec![0u8; n * desc_len];
    for i in 0..n {
        descs[i * desc_len] = (i * 10) as u8;
        descs[i * desc_len + 1] = (i * 5) as u8;
    }

    let matches = polar_match_one_way(&theta, &descs, &theta, &descs, desc_len, 5, None);

    // Each feature should match itself
    for (&idx1, &(idx2, dist)) in &matches {
        assert_eq!(
            idx1, idx2,
            "Feature {idx1} matched to {idx2} instead of itself"
        );
        assert_eq!(dist, 0.0);
    }
}

#[test]
fn test_epipole_at_infinity_returns_none() {
    // F = [[0,0,0],[0,0,-1],[0,1,0]]
    // Null space of F is [1,0,0] (at infinity), so should return None.
    let f = [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0];
    assert!(compute_epipole_pair(&f).is_none());
}

#[test]
fn test_epipole_from_pure_translation() {
    // Pure translation: P1 = [I|0], P2 = [I|t] with t = (2, 3, 1).
    // F = [t]_x is skew-symmetric, so both null(F) and null(F^T) are t.
    // Both epipoles dehomogenize to t/t_z = (2, 3).
    //
    //   [t]_x = [[ 0, -1,  3],
    //            [ 1,  0, -2],
    //            [-3,  2,  0]]
    let f = [0.0, -1.0, 3.0, 1.0, 0.0, -2.0, -3.0, 2.0, 0.0];

    let (e1, e2) = compute_epipole_pair(&f).expect("both epipoles should be finite");
    assert!((e1[0] - 2.0).abs() < 1e-6, "e1.x = {}", e1[0]);
    assert!((e1[1] - 3.0).abs() < 1e-6, "e1.y = {}", e1[1]);
    assert!((e2[0] - 2.0).abs() < 1e-6, "e2.x = {}", e2[0]);
    assert!((e2[1] - 3.0).abs() < 1e-6, "e2.y = {}", e2[1]);
}

#[test]
fn test_epipole_from_diagonal_translation() {
    // Pure translation with t = (5, -4, 2).
    // Both epipoles = t/t_z = (2.5, -2).
    //
    //   [t]_x = [[ 0, -2, -4],
    //            [ 2,  0, -5],
    //            [ 4,  5,  0]]
    let f = [0.0, -2.0, -4.0, 2.0, 0.0, -5.0, 4.0, 5.0, 0.0];

    let (e1, e2) = compute_epipole_pair(&f).expect("both epipoles should be finite");
    assert!((e1[0] - 2.5).abs() < 1e-6, "e1.x = {}", e1[0]);
    assert!((e1[1] - (-2.0)).abs() < 1e-6, "e1.y = {}", e1[1]);
    assert!((e2[0] - 2.5).abs() < 1e-6, "e2.x = {}", e2[0]);
    assert!((e2[1] - (-2.0)).abs() < 1e-6, "e2.y = {}", e2[1]);
}

#[test]
fn test_polar_mutual_best_match_geometric_basic() {
    use nalgebra::{Matrix3, Vector3};

    // 5 features arranged in a circle around (320, 240) at radius 100
    let n = 5;
    let desc_len = 128;
    let mut positions = Vec::with_capacity(n * 2);
    let mut descs = vec![0u8; n * desc_len];
    let mut affines = Vec::with_capacity(n * 4);

    for i in 0..n {
        let angle = 2.0 * PI * i as f64 / n as f64;
        positions.push(320.0 + 100.0 * angle.cos());
        positions.push(240.0 + 100.0 * angle.sin());
        // Unique descriptors
        descs[i * desc_len] = (i * 30) as u8;
        descs[i * desc_len + 1] = (i * 17) as u8;
        // Identity affines
        affines.extend_from_slice(&[1.0, 0.0, 0.0, 1.0]);
    }

    // Identity cameras with small baseline for finite epipoles
    let mut k = Matrix3::identity();
    k[(0, 0)] = 500.0;
    k[(1, 1)] = 500.0;
    k[(0, 2)] = 320.0;
    k[(1, 2)] = 240.0;
    let r = Matrix3::identity();
    let t1 = Vector3::zeros();
    let t2 = Vector3::zeros();
    let geom = StereoPairGeometry::new(&k, &k, &r, &r, &t1, &t2);
    let config = GeometricFilterConfig::default();

    // F for pure forward translation t=(0,0,1):
    // [t]_x = [[0, -1, 0], [1, 0, 0], [0, 0, 0]]
    let f_matrix = [0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];

    // Epipoles should be at (0, 0) from null space of F (which is [0,0,1])
    // Actually for this F, null(F) = (0,0,1) -> dehomogenized = (0,0).
    // The epipole is at (0,0) which is far from features at ~(320,240).

    let result = polar_mutual_best_match_geometric(
        &positions, &descs, n, &positions, &descs, n, &affines, &affines, desc_len, &f_matrix, 10,
        None, 1.0, &geom, &config,
    );

    let mutual = result.expect("Should return Some since epipoles are finite");
    // All features should match themselves
    assert_eq!(
        mutual.len(),
        n,
        "Expected {n} mutual matches, got {}",
        mutual.len()
    );
    for (idx1, idx2, dist) in &mutual {
        assert_eq!(
            idx1, idx2,
            "Feature {idx1} matched to {idx2} instead of itself"
        );
        assert_eq!(*dist, 0.0);
    }
}

#[test]
fn test_polar_match_with_threshold() {
    // Match features against themselves but with a tight threshold
    let n = 10;
    let theta: Vec<f64> = (0..n)
        .map(|i| -PI + 2.0 * PI * i as f64 / n as f64)
        .collect();

    let desc_len = 128;
    let mut descs = vec![0u8; n * desc_len];
    for i in 0..n {
        descs[i * desc_len] = (i * 10) as u8;
        descs[i * desc_len + 1] = (i * 5) as u8;
    }

    // With no threshold, self-match should get all matches
    let matches_none = polar_match_one_way(&theta, &descs, &theta, &descs, desc_len, 5, None);
    assert_eq!(matches_none.len(), n);

    // With threshold = 0.0, only exact matches pass (self-match gives dist=0)
    let matches_zero = polar_match_one_way(&theta, &descs, &theta, &descs, desc_len, 5, Some(0.0));
    assert_eq!(matches_zero.len(), n);

    // With a very tight threshold < min inter-descriptor distance, and different descriptors
    let mut descs2 = vec![0u8; n * desc_len];
    for i in 0..n {
        descs2[i * desc_len] = (i * 10 + 100) as u8; // shifted away
    }
    let matches_tight =
        polar_match_one_way(&theta, &descs, &theta, &descs2, desc_len, 5, Some(5.0));
    // Most or all matches should be rejected since descriptors differ by >= 100
    assert!(
        matches_tight.len() < n,
        "Tight threshold should reject distant descriptors"
    );
}

#[test]
fn test_polar_match_multiple_window_sizes() {
    let n = 20;
    let theta: Vec<f64> = (0..n)
        .map(|i| -PI + 2.0 * PI * i as f64 / n as f64)
        .collect();

    let desc_len = 128;
    let mut descs = vec![0u8; n * desc_len];
    for i in 0..n {
        descs[i * desc_len] = (i * 12) as u8;
        descs[i * desc_len + 1] = (i * 7) as u8;
    }

    // Self-match with various window sizes should all find perfect matches
    for window in [3, 5, 10, 20] {
        let matches = polar_match_one_way(&theta, &descs, &theta, &descs, desc_len, window, None);
        assert_eq!(
            matches.len(),
            n,
            "Window size {window}: expected {n} matches, got {}",
            matches.len()
        );
        for (&idx1, &(idx2, dist)) in &matches {
            assert_eq!(idx1, idx2);
            assert_eq!(dist, 0.0);
        }
    }
}

#[test]
fn test_polar_match_larger_feature_set() {
    // Test with 50 features (matching Python test scale)
    let n = 50;
    let theta: Vec<f64> = (0..n)
        .map(|i| -PI + 2.0 * PI * i as f64 / n as f64)
        .collect();

    let desc_len = 128;
    let mut descs = vec![0u8; n * desc_len];
    for i in 0..n {
        // Use multiple bytes for more unique descriptors
        descs[i * desc_len] = (i % 256) as u8;
        descs[i * desc_len + 1] = ((i * 7) % 256) as u8;
        descs[i * desc_len + 2] = ((i * 13) % 256) as u8;
    }

    let matches = polar_match_one_way(&theta, &descs, &theta, &descs, desc_len, 10, None);
    assert_eq!(matches.len(), n);
    for (&idx1, &(idx2, dist)) in &matches {
        assert_eq!(idx1, idx2);
        assert_eq!(dist, 0.0);
    }
}

#[test]
fn test_polar_mutual_best_match_nongeometric() {
    // Test the non-geometric polar_mutual_best_match function
    let n = 10;
    let desc_len = 128;

    // Features arranged in a circle at radius 100 from epipole at (0, 0)
    let mut positions = Vec::with_capacity(n * 2);
    let mut descs = vec![0u8; n * desc_len];
    for i in 0..n {
        let angle = 2.0 * PI * i as f64 / n as f64;
        positions.push(100.0 * angle.cos());
        positions.push(100.0 * angle.sin());
        descs[i * desc_len] = (i * 25) as u8;
        descs[i * desc_len + 1] = (i * 13) as u8;
    }

    // F for pure forward translation t=(0,0,1):
    // [t]_x = [[0, -1, 0], [1, 0, 0], [0, 0, 0]]
    let f_matrix = [0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];

    let result = polar_mutual_best_match(
        &positions, &descs, n, &positions, &descs, n, desc_len, &f_matrix, 10, None, 1.0,
    );

    let mutual = result.expect("Should return Some since epipoles are finite");
    assert_eq!(mutual.len(), n);
    for (idx1, idx2, dist) in &mutual {
        assert_eq!(idx1, idx2);
        assert_eq!(*dist, 0.0);
    }
}

#[test]
fn test_polar_mutual_best_match_with_threshold() {
    let n = 10;
    let desc_len = 128;

    let mut positions = Vec::with_capacity(n * 2);
    let mut descs1 = vec![0u8; n * desc_len];
    let mut descs2 = vec![0u8; n * desc_len];
    for i in 0..n {
        let angle = 2.0 * PI * i as f64 / n as f64;
        positions.push(100.0 * angle.cos());
        positions.push(100.0 * angle.sin());
        // Make all descriptors in set 1 unique
        descs1[i * desc_len] = (i * 25) as u8;
        descs1[i * desc_len + 1] = (i * 13) as u8;
        // Set 2: all very far from set 1 (fill with 200)
        descs2[i * desc_len..i * desc_len + desc_len].fill(200);
    }

    let f_matrix = [0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];

    // No threshold: should find matches (nearest, even if far)
    let result_none = polar_mutual_best_match(
        &positions, &descs1, n, &positions, &descs2, n, desc_len, &f_matrix, 10, None, 1.0,
    );
    let mutual_none = result_none.expect("Should return Some");
    assert!(
        !mutual_none.is_empty(),
        "Without threshold, should find matches"
    );

    // Very tight threshold: should reject all (min distance ≈ sqrt(200^2 * 128) ≈ 2263)
    let result_tight = polar_mutual_best_match(
        &positions,
        &descs1,
        n,
        &positions,
        &descs2,
        n,
        desc_len,
        &f_matrix,
        10,
        Some(10.0),
        1.0,
    );
    let mutual_tight = result_tight.expect("Should return Some");
    assert!(
        mutual_tight.is_empty(),
        "Very tight threshold should reject all distant matches, got {} matches",
        mutual_tight.len()
    );
}

#[test]
fn test_polar_epipole_null_space_property() {
    // Verify F @ e1_h ≈ 0 and F^T @ e2_h ≈ 0
    let f_vals = [0.0, -1.0, 3.0, 1.0, 0.0, -2.0, -3.0, 2.0, 0.0];
    let f = Matrix3::from_row_slice(&f_vals);

    let (e1, e2) = compute_epipole_pair(&f_vals).expect("epipoles should be finite");

    // F @ e1_h ≈ 0 (left null space)
    let e1_h = nalgebra::Vector3::new(e1[0], e1[1], 1.0);
    let f_e1 = f * e1_h;
    assert!(f_e1.norm() < 1e-6, "F @ e1_h should be ≈ 0, got {:?}", f_e1);

    // F^T @ e2_h ≈ 0 (right null space)
    let e2_h = nalgebra::Vector3::new(e2[0], e2[1], 1.0);
    let ft_e2 = f.transpose() * e2_h;
    assert!(
        ft_e2.norm() < 1e-6,
        "F^T @ e2_h should be ≈ 0, got {:?}",
        ft_e2
    );
}

#[test]
fn test_polar_match_one_way_geometric_rejects_bad_orientation() {
    use nalgebra::{Matrix3, Vector3};

    // 5 features arranged in a circle at radius 100 from epipole at (0,0)
    let n = 5;
    let desc_len = 128;

    let mut theta = Vec::with_capacity(n);
    let mut positions = Vec::with_capacity(n * 2);

    for i in 0..n {
        let angle = -PI + 2.0 * PI * i as f64 / n as f64;
        theta.push(angle);
        positions.push(100.0 * angle.cos());
        positions.push(100.0 * angle.sin());
    }

    // All descriptors identical so descriptor matching always succeeds
    let descs = vec![1u8; n * desc_len];

    // Query affines: identity orientation
    let affines1: Vec<f64> = (0..n).flat_map(|_| vec![5.0, 0.0, 0.0, 5.0]).collect();
    // Target affines: perpendicular orientation (should be rejected)
    let affines2: Vec<f64> = (0..n).flat_map(|_| vec![0.0, 5.0, 5.0, 0.0]).collect();

    let mut k = Matrix3::identity();
    k[(0, 0)] = 500.0;
    k[(1, 1)] = 500.0;
    k[(0, 2)] = 320.0;
    k[(1, 2)] = 240.0;
    let r = Matrix3::identity();
    let t1 = Vector3::zeros();
    let t2 = Vector3::new(1.0, 0.0, 0.0);
    let geom = StereoPairGeometry::new(&k, &k, &r, &r, &t1, &t2);
    let config = GeometricFilterConfig::default();

    let matches = polar_match_one_way_geometric(
        &theta, &descs, &positions, &affines1, &theta, &descs, &positions, &affines2, desc_len, 5,
        None, &geom, &config,
    );

    assert!(
        matches.is_empty(),
        "All matches should be rejected due to perpendicular orientation, got {} matches",
        matches.len()
    );
}
