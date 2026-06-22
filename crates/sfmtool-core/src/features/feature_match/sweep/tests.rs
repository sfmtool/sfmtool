use super::*;

#[test]
fn test_argsort_by_y() {
    // 3 points: (0, 10), (0, 5), (0, 15)
    let kpts = [0.0, 10.0, 0.0, 5.0, 0.0, 15.0];
    let indices = argsort_by_y(&kpts, 3);
    assert_eq!(indices, vec![1, 0, 2]); // sorted by Y: 5, 10, 15
}

#[test]
fn test_one_way_sweep_basic() {
    // 3 features in image 1, 3 in image 2
    // Sorted by Y already
    let kpts1 = [0.0, 1.0, 0.0, 2.0, 0.0, 3.0];
    let kpts2 = [0.0, 1.0, 0.0, 2.0, 0.0, 3.0];

    // Descriptors: make feature 0 in img1 closest to feature 0 in img2, etc.
    let mut descs1 = vec![0u8; 3 * 128];
    let mut descs2 = vec![0u8; 3 * 128];
    // Feature 0: descs1[0] = [1, 0, ...], descs2[0] = [1, 0, ...]
    descs1[0] = 1;
    descs2[0] = 1;
    // Feature 1: descs1[128] = [0, 2, ...], descs2[128] = [0, 2, ...]
    descs1[129] = 2;
    descs2[129] = 2;
    // Feature 2: descs1[256] = [0, 0, 3, ...], descs2[256] = [0, 0, 3, ...]
    descs1[258] = 3;
    descs2[258] = 3;

    let matches = match_one_way_sweep(&kpts1, &descs1, 3, &kpts2, &descs2, 3, 3, None);

    assert_eq!(matches.len(), 3);
    assert_eq!(matches[&0].0, 0);
    assert_eq!(matches[&1].0, 1);
    assert_eq!(matches[&2].0, 2);
}

#[test]
fn test_mutual_match_basic() {
    // Two sets of identical features
    let kpts1 = [10.0, 1.0, 20.0, 2.0, 30.0, 3.0];
    let kpts2 = [10.0, 1.0, 20.0, 2.0, 30.0, 3.0];

    let mut descs1 = vec![0u8; 3 * 128];
    let mut descs2 = vec![0u8; 3 * 128];
    descs1[0] = 10;
    descs2[0] = 10;
    descs1[129] = 20;
    descs2[129] = 20;
    descs1[258] = 30;
    descs2[258] = 30;

    let mutual = mutual_best_match_sweep(&kpts1, &descs1, 3, &kpts2, &descs2, 3, 128, 3, None);

    assert_eq!(mutual.len(), 3);
    // All should be identity matches
    for (idx1, idx2, dist) in &mutual {
        assert_eq!(idx1, idx2);
        assert_eq!(*dist, 0.0);
    }
}

#[test]
fn test_empty_inputs() {
    let matches = match_one_way_sweep(&[], &[], 0, &[0.0, 1.0], &[0u8; 128], 1, 30, None);
    assert!(matches.is_empty());

    let matches = match_one_way_sweep(&[0.0, 1.0], &[0u8; 128], 1, &[], &[], 0, 30, None);
    assert!(matches.is_empty());
}

#[test]
fn test_mutual_match_geometric_basic() {
    use nalgebra::{Matrix3, Vector3};

    // Same setup as test_mutual_match_basic but with identity affines and identity cameras
    let kpts1 = [10.0, 1.0, 20.0, 2.0, 30.0, 3.0];
    let kpts2 = [10.0, 1.0, 20.0, 2.0, 30.0, 3.0];

    let mut descs1 = vec![0u8; 3 * 128];
    let mut descs2 = vec![0u8; 3 * 128];
    descs1[0] = 10;
    descs2[0] = 10;
    descs1[129] = 20;
    descs2[129] = 20;
    descs1[258] = 30;
    descs2[258] = 30;

    // Identity affines for all features
    let affines1 = [
        1.0, 0.0, 0.0, 1.0, // feature 0
        1.0, 0.0, 0.0, 1.0, // feature 1
        1.0, 0.0, 0.0, 1.0, // feature 2
    ];
    let affines2 = [
        1.0, 0.0, 0.0, 1.0, // feature 0
        1.0, 0.0, 0.0, 1.0, // feature 1
        1.0, 0.0, 0.0, 1.0, // feature 2
    ];

    // Identity cameras (same K, identity R, zero t)
    let mut k = Matrix3::identity();
    k[(0, 0)] = 500.0;
    k[(1, 1)] = 500.0;
    k[(0, 2)] = 320.0;
    k[(1, 2)] = 240.0;
    let r = Matrix3::identity();
    let t = Vector3::zeros();
    let geom = StereoPairGeometry::new(&k, &k, &r, &r, &t, &t);
    let config = GeometricFilterConfig::default();

    let mutual = mutual_best_match_sweep_geometric(
        &kpts1, &descs1, 3, &kpts2, &descs2, 3, &affines1, &affines2, 128, 3, None, &geom, &config,
    );

    assert_eq!(mutual.len(), 3);
    // All should be identity matches
    for (idx1, idx2, dist) in &mutual {
        assert_eq!(idx1, idx2);
        assert_eq!(*dist, 0.0);
    }
}

#[test]
fn test_one_way_sweep_with_threshold_rejects() {
    // 1 feature each, with descriptors far apart
    let kpts1 = [0.0, 1.0];
    let kpts2 = [0.0, 1.0];

    let descs1 = vec![0u8; 128];
    let descs2 = vec![100u8; 128];
    // Distance = sqrt(100^2 * 128) ≈ 1131.4

    // Tight threshold: should reject
    let matches = match_one_way_sweep(&kpts1, &descs1, 1, &kpts2, &descs2, 1, 1, Some(10.0));
    assert!(
        matches.is_empty(),
        "Tight threshold should reject distant descriptors"
    );

    // Generous threshold: should accept
    let matches = match_one_way_sweep(&kpts1, &descs1, 1, &kpts2, &descs2, 1, 1, Some(2000.0));
    assert_eq!(matches.len(), 1);
}

#[test]
fn test_one_way_sweep_with_threshold_accepts() {
    // With a generous threshold, all matches should pass
    let kpts1 = [0.0, 1.0, 0.0, 2.0, 0.0, 3.0];
    let kpts2 = [0.0, 1.0, 0.0, 2.0, 0.0, 3.0];

    let mut descs1 = vec![0u8; 3 * 128];
    let mut descs2 = vec![0u8; 3 * 128];
    descs1[0] = 1;
    descs2[0] = 1;
    descs1[129] = 2;
    descs2[129] = 2;
    descs1[258] = 3;
    descs2[258] = 3;

    let matches = match_one_way_sweep(&kpts1, &descs1, 3, &kpts2, &descs2, 3, 3, Some(2000.0));
    assert_eq!(matches.len(), 3);
}

#[test]
fn test_one_way_sweep_multiple_thresholds() {
    let kpts1 = [0.0, 1.0];
    let kpts2 = [0.0, 1.0];

    let descs1 = vec![0u8; 128];
    let mut descs2 = vec![0u8; 128];
    descs2[0] = 3;
    descs2[1] = 4;
    // Distance = sqrt(9 + 16) = 5.0

    // Threshold below distance: no match
    let matches = match_one_way_sweep(&kpts1, &descs1, 1, &kpts2, &descs2, 1, 1, Some(4.0));
    assert!(matches.is_empty());

    // Threshold at distance: accepted (<=)
    let matches = match_one_way_sweep(&kpts1, &descs1, 1, &kpts2, &descs2, 1, 1, Some(5.0));
    assert_eq!(matches.len(), 1);

    // Threshold above distance: accepted
    let matches = match_one_way_sweep(&kpts1, &descs1, 1, &kpts2, &descs2, 1, 1, Some(100.0));
    assert_eq!(matches.len(), 1);
}

#[test]
fn test_asymmetric_feature_counts() {
    // 5 features in img1, 3 in img2
    let n1 = 5;
    let n2 = 3;
    let mut kpts1 = vec![0.0; n1 * 2];
    let mut kpts2 = vec![0.0; n2 * 2];
    for i in 0..n1 {
        kpts1[i * 2 + 1] = i as f64;
    }
    for i in 0..n2 {
        kpts2[i * 2 + 1] = i as f64;
    }

    let mut descs1 = vec![0u8; n1 * 128];
    let mut descs2 = vec![0u8; n2 * 128];
    // Make each descriptor unique
    for i in 0..n1 {
        descs1[i * 128] = (i * 10) as u8;
    }
    for i in 0..n2 {
        descs2[i * 128] = (i * 10) as u8;
    }

    let matches = match_one_way_sweep(&kpts1, &descs1, n1, &kpts2, &descs2, n2, 5, None);
    // First 3 features in img1 should match perfectly to img2 features
    assert!(matches.len() >= 3);
    for i in 0..3 {
        assert_eq!(matches[&i].0, i);
        assert_eq!(matches[&i].1, 0.0);
    }
}

#[test]
fn test_mutual_match_asymmetric() {
    // 4 features in img1, 6 in img2
    let n1 = 4;
    let n2 = 6;
    let mut kpts1 = vec![0.0; n1 * 2];
    let mut kpts2 = vec![0.0; n2 * 2];
    for i in 0..n1 {
        kpts1[i * 2 + 1] = i as f64;
    }
    for i in 0..n2 {
        kpts2[i * 2 + 1] = i as f64;
    }

    let mut descs1 = vec![0u8; n1 * 128];
    let mut descs2 = vec![0u8; n2 * 128];
    // First n1 features match between the two sets
    for i in 0..n1 {
        descs1[i * 128] = (i * 20 + 10) as u8;
        descs2[i * 128] = (i * 20 + 10) as u8;
    }
    // Extra features in img2 are distinct
    for i in n1..n2 {
        descs2[i * 128] = 200;
        descs2[i * 128 + 1] = (i * 30) as u8;
    }

    let mutual = mutual_best_match_sweep(&kpts1, &descs1, n1, &kpts2, &descs2, n2, 128, 10, None);

    assert_eq!(mutual.len(), n1);
    for (idx1, idx2, dist) in &mutual {
        assert_eq!(idx1, idx2);
        assert_eq!(*dist, 0.0);
    }
}

#[test]
fn test_mutual_match_with_threshold() {
    let kpts1 = [10.0, 1.0, 20.0, 2.0, 30.0, 3.0];
    let kpts2 = [10.0, 1.0, 20.0, 2.0, 30.0, 3.0];

    // Feature 0: identical (dist=0), Feature 1: close (dist=5), Feature 2: far (dist≈1131)
    let mut descs1 = vec![0u8; 3 * 128];
    let mut descs2 = vec![0u8; 3 * 128];
    descs1[0] = 10;
    descs2[0] = 10;
    descs1[129] = 20;
    descs2[129] = 23; // diff = 3
    descs2[130] = 4; // diff = 4, distance = 5
    descs1[258] = 30;
    descs2[256..384].fill(100); // very far

    // Threshold that accepts feature 0 and 1, rejects feature 2
    let mutual =
        mutual_best_match_sweep(&kpts1, &descs1, 3, &kpts2, &descs2, 3, 128, 3, Some(10.0));
    assert_eq!(mutual.len(), 2);
}

#[test]
fn test_larger_window_sizes() {
    // 10 features, test window sizes 5, 10, 20
    let n = 10;
    let mut kpts = vec![0.0; n * 2];
    for i in 0..n {
        kpts[i * 2 + 1] = (i * 10) as f64;
    }
    let mut descs = vec![0u8; n * 128];
    for i in 0..n {
        descs[i * 128] = (i * 15) as u8;
    }

    for window in [5, 10, 20] {
        let mutual = mutual_best_match_sweep(&kpts, &descs, n, &kpts, &descs, n, 128, window, None);
        // All features should match themselves regardless of window size
        assert_eq!(
            mutual.len(),
            n,
            "Window size {window}: expected {n} matches, got {}",
            mutual.len()
        );
        for (idx1, idx2, dist) in &mutual {
            assert_eq!(idx1, idx2);
            assert_eq!(*dist, 0.0);
        }
    }
}

#[test]
fn test_match_one_way_sweep_geometric_rejects_bad_orientation() {
    use nalgebra::{Matrix3, Vector3};

    // 3 features in image 1, 3 in image 2, sorted by Y
    let kpts1 = [320.0, 1.0, 320.0, 2.0, 320.0, 3.0];
    let kpts2 = [320.0, 1.0, 320.0, 2.0, 320.0, 3.0];

    // Make all descriptors identical so descriptor matching always succeeds
    let descs1 = vec![1u8; 3 * 128];
    let descs2 = vec![1u8; 3 * 128];

    // Query affines: identity orientation
    let affines1 = [
        5.0, 0.0, 0.0, 5.0, // feature 0
        5.0, 0.0, 0.0, 5.0, // feature 1
        5.0, 0.0, 0.0, 5.0, // feature 2
    ];
    // Target affines: perpendicular orientation (first col rotated 90 degrees)
    let affines2 = [
        0.0, 5.0, 5.0, 0.0, // feature 0: first col = (0, 5), perpendicular
        0.0, 5.0, 5.0, 0.0, // feature 1
        0.0, 5.0, 5.0, 0.0, // feature 2
    ];

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

    let matches = match_one_way_sweep_geometric(
        &kpts1, &descs1, 3, &kpts2, &descs2, 3, &affines1, &affines2, 3, None, &geom, &config,
    );

    // All candidates should be rejected due to bad orientation
    assert!(
        matches.is_empty(),
        "All matches should be rejected due to perpendicular orientation"
    );
}
