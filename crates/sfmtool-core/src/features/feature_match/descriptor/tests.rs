use super::*;

#[test]
fn test_distance_identical() {
    let a = [42u8; 128];
    assert_eq!(descriptor_distance_l2(&a, &a), 0.0);
    assert_eq!(descriptor_distance_l2_squared(&a, &a), 0);
}

#[test]
fn test_distance_known_values() {
    let a = [0u8; 128];
    let mut b = [0u8; 128];
    b[0] = 3;
    b[1] = 4;
    // sqrt(9 + 16) = 5.0
    assert!((descriptor_distance_l2(&a, &b) - 5.0).abs() < 1e-10);
    assert_eq!(descriptor_distance_l2_squared(&a, &b), 25);
}

#[test]
fn test_distance_symmetric() {
    let a: Vec<u8> = (0..128).collect();
    let b: Vec<u8> = (128..=255).chain(0..1).take(128).collect();
    let d1 = descriptor_distance_l2(&a, &b);
    let d2 = descriptor_distance_l2(&b, &a);
    assert!((d1 - d2).abs() < 1e-10);
}

#[test]
fn test_find_best_match_empty() {
    let query = [0u8; 128];
    let candidates: Vec<&[u8]> = vec![];
    assert!(find_best_match(&query, &candidates, None).is_none());
}

#[test]
fn test_find_best_match_selects_closest() {
    let query = [0u8; 128];
    let far = [100u8; 128];
    let close = [1u8; 128];
    let mid = [50u8; 128];
    let candidates: Vec<&[u8]> = vec![&far, &close, &mid];
    let (idx, _dist) = find_best_match(&query, &candidates, None).unwrap();
    assert_eq!(idx, 1);
}

#[test]
fn test_find_best_match_threshold() {
    let query = [0u8; 128];
    let far = [100u8; 128];
    let candidates: Vec<&[u8]> = vec![&far];
    // Distance is sqrt(100^2 * 128) ≈ 1131.4
    assert!(find_best_match(&query, &candidates, Some(10.0)).is_none());
    assert!(find_best_match(&query, &candidates, Some(2000.0)).is_some());
}

#[test]
fn test_distance_max_values() {
    // All-zero vs all-255: maximum possible distance
    let a = [0u8; 128];
    let b = [255u8; 128];
    // sqrt(255^2 * 128) = 255 * sqrt(128) ≈ 2884.27
    let dist = descriptor_distance_l2(&a, &b);
    let expected = 255.0 * (128.0_f64).sqrt();
    assert!((dist - expected).abs() < 0.01);
    assert_eq!(descriptor_distance_l2_squared(&a, &b), 255_i64 * 255 * 128);
}

#[test]
fn test_find_best_match_multiple_thresholds() {
    let query = [0u8; 128];
    let mut candidate = [0u8; 128];
    candidate[0] = 3;
    candidate[1] = 4;
    // Distance = 5.0
    let candidates: Vec<&[u8]> = vec![&candidate];

    // Test various thresholds around the distance
    assert!(find_best_match(&query, &candidates, Some(4.0)).is_none());
    assert!(find_best_match(&query, &candidates, Some(4.9)).is_none());
    assert!(find_best_match(&query, &candidates, Some(5.0)).is_some());
    assert!(find_best_match(&query, &candidates, Some(100.0)).is_some());
    assert!(find_best_match(&query, &candidates, Some(500.0)).is_some());
    assert!(find_best_match(&query, &candidates, Some(1000.0)).is_some());
}

#[test]
fn test_find_best_match_single_candidate() {
    let query = [42u8; 128];
    let candidate = [42u8; 128];
    let candidates: Vec<&[u8]> = vec![&candidate];
    let (idx, dist) = find_best_match(&query, &candidates, None).unwrap();
    assert_eq!(idx, 0);
    assert_eq!(dist, 0.0);
}

#[test]
fn test_distance_l2_squared_matches_l2() {
    // Verify that l2 == sqrt(l2_squared) for multiple test vectors
    let test_cases: Vec<(Vec<u8>, Vec<u8>)> = vec![
        (vec![0; 128], vec![0; 128]),
        (vec![0; 128], vec![255; 128]),
        (
            (0..128).collect(),
            (128..=255).chain(0..1).take(128).collect(),
        ),
        (vec![42; 128], vec![43; 128]),
    ];

    for (a, b) in &test_cases {
        let dist_sq = descriptor_distance_l2_squared(a, b);
        let dist = descriptor_distance_l2(a, b);
        assert!(
            (dist - (dist_sq as f64).sqrt()).abs() < 1e-10,
            "l2 and sqrt(l2_squared) should match"
        );
    }
}

#[test]
fn test_find_best_match_contiguous() {
    let query = [0u8; 128];
    let mut candidates = vec![100u8; 128 * 3];
    // Make second candidate closest
    for b in &mut candidates[128..256] {
        *b = 1;
    }
    let (idx, _) = find_best_match_contiguous(&query, &candidates, 128, None).unwrap();
    assert_eq!(idx, 1);
}
