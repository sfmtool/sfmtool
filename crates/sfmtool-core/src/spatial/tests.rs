use super::*;

// ── f64 query tests ─────────────────────────────────────────────────

#[test]
fn test_2d_nearest() {
    let positions = [0.0_f64, 0.0, 3.0, 4.0, 10.0, 0.0];
    let cloud = PointCloud2::<f64>::new(&positions, 3);

    let result = cloud.nearest(&[1.0, 1.0], 1);
    assert_eq!(result, vec![0]);

    let result = cloud.nearest(&[9.0, 0.0], 1);
    assert_eq!(result, vec![2]);
}

#[test]
fn test_2d_nearest_k() {
    let positions = [0.0_f64, 0.0, 1.0, 0.0, 2.0, 0.0, 10.0, 0.0];
    let cloud = PointCloud2::<f64>::new(&positions, 4);

    let result = cloud.nearest_k(&[0.5, 0.0], 1, 2);
    assert_eq!(result.len(), 2);
    assert!(result.contains(&0));
    assert!(result.contains(&1));
}

#[test]
fn test_2d_within_radius() {
    let positions = [0.0_f64, 0.0, 1.0, 0.0, 2.0, 0.0, 10.0, 0.0];
    let cloud = PointCloud2::<f64>::new(&positions, 4);

    let (offsets, indices) = cloud.within_radius(&[0.5, 0.0], 1, 2.0);
    assert_eq!(offsets, vec![0, 3]);
    assert_eq!(indices.len(), 3);
    let mut sorted = indices.clone();
    sorted.sort();
    assert_eq!(sorted, vec![0, 1, 2]);
}

#[test]
fn test_2d_self_nearest_k() {
    let positions = [0.0_f64, 0.0, 1.0, 0.0, 5.0, 0.0];
    let cloud = PointCloud2::<f64>::new(&positions, 3);

    let result = cloud.self_nearest_k(1);
    assert_eq!(result.len(), 3);
    assert_eq!(result[0], 1);
    assert_eq!(result[1], 0);
    assert_eq!(result[2], 1);
}

#[test]
fn test_3d_nearest() {
    let positions = [0.0_f64, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 10.0, 0.0];
    let cloud = PointCloud3::<f64>::new(&positions, 3);

    let result = cloud.nearest(&[0.5, 0.0, 0.0], 1);
    assert_eq!(result, vec![0]);

    let result = cloud.nearest(&[0.9, 0.0, 0.0], 1);
    assert_eq!(result, vec![1]);
}

#[test]
fn test_3d_within_radius() {
    let positions = [0.0_f64, 0.0, 0.0, 1.0, 0.0, 0.0, 100.0, 0.0, 0.0];
    let cloud = PointCloud3::<f64>::new(&positions, 3);

    let (offsets, indices) = cloud.within_radius(&[0.5, 0.0, 0.0], 1, 2.0);
    assert_eq!(offsets, vec![0, 2]);
    let mut sorted = indices.clone();
    sorted.sort();
    assert_eq!(sorted, vec![0, 1]);
}

#[test]
fn test_batch_queries() {
    let positions = [0.0_f64, 0.0, 10.0, 0.0];
    let cloud = PointCloud2::<f64>::new(&positions, 2);

    let result = cloud.nearest(&[1.0, 0.0, 9.0, 0.0], 2);
    assert_eq!(result, vec![0, 1]);
}

#[test]
fn test_self_nearest_k_greater_than_available() {
    let positions = [0.0_f64, 0.0, 1.0, 0.0];
    let cloud = PointCloud2::<f64>::new(&positions, 2);
    let result = cloud.self_nearest_k(3);
    assert_eq!(result.len(), 6);
    assert_eq!(result[0], 1);
    assert_eq!(result[1], u32::MAX);
    assert_eq!(result[2], u32::MAX);
    assert_eq!(result[3], 0);
    assert_eq!(result[4], u32::MAX);
    assert_eq!(result[5], u32::MAX);
}

// ── f32 tests ───────────────────────────────────────────────────────

#[test]
fn test_2d_f32_nearest() {
    let positions: [f32; 6] = [0.0, 0.0, 3.0, 4.0, 10.0, 0.0];
    let cloud = PointCloud2::<f32>::new(&positions, 3);
    assert_eq!(cloud.len(), 3);

    let result = cloud.nearest(&[1.0, 1.0], 1);
    assert_eq!(result, vec![0]);

    let result = cloud.nearest(&[9.0, 0.0], 1);
    assert_eq!(result, vec![2]);
}

#[test]
fn test_2d_f32_nearest_k() {
    let positions: [f32; 8] = [0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 10.0, 0.0];
    let cloud = PointCloud2::<f32>::new(&positions, 4);

    let result = cloud.nearest_k(&[0.5_f32, 0.0], 1, 2);
    assert_eq!(result.len(), 2);
    assert!(result.contains(&0));
    assert!(result.contains(&1));
}

#[test]
fn test_2d_f32_within_radius() {
    let positions: [f32; 8] = [0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 10.0, 0.0];
    let cloud = PointCloud2::<f32>::new(&positions, 4);

    let (offsets, indices) = cloud.within_radius(&[0.5_f32, 0.0], 1, 2.0);
    assert_eq!(offsets, vec![0, 3]);
    let mut sorted = indices.clone();
    sorted.sort();
    assert_eq!(sorted, vec![0, 1, 2]);
}

#[test]
fn test_2d_f32_self_nearest_k() {
    let positions: [f32; 6] = [0.0, 0.0, 1.0, 0.0, 5.0, 0.0];
    let cloud = PointCloud2::<f32>::new(&positions, 3);

    let result = cloud.self_nearest_k(1);
    assert_eq!(result, vec![1, 0, 1]);
}

#[test]
fn test_3d_f32_nearest() {
    let positions: [f32; 9] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 10.0, 0.0];
    let cloud = PointCloud3::<f32>::new(&positions, 3);

    let result = cloud.nearest(&[0.5_f32, 0.0, 0.0], 1);
    assert_eq!(result, vec![0]);

    let result = cloud.nearest(&[0.9_f32, 0.0, 0.0], 1);
    assert_eq!(result, vec![1]);
}

#[test]
fn test_3d_f32_self_nearest_k() {
    let positions: [f32; 9] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 5.0, 0.0, 0.0];
    let cloud = PointCloud3::<f32>::new(&positions, 3);

    let result = cloud.self_nearest_k(1);
    assert_eq!(result, vec![1, 0, 1]);
}

// ── nearest_k_within_radius tests ────────────────────────────────────

#[test]
fn test_2d_nearest_k_within_radius() {
    // Points at x = 0, 1, 5, 10
    let positions = [0.0_f64, 0.0, 1.0, 0.0, 5.0, 0.0, 10.0, 0.0];
    let cloud = PointCloud2::<f64>::new(&positions, 4);

    // Query at 0.1: within radius 2.0, k=3 → should get points 0 (dist 0.1) and 1 (dist 0.9)
    let result = cloud.nearest_k_within_radius(&[0.1, 0.0], 1, 3, 2.0);
    assert_eq!(result.len(), 3);
    assert_eq!(result[0], 0); // nearest (dist 0.1)
    assert_eq!(result[1], 1); // 2nd nearest (dist 0.9)
    assert_eq!(result[2], u32::MAX); // point 2 at dist 4.9, beyond radius
}

#[test]
fn test_2d_nearest_k_within_radius_limits_by_k() {
    // Points at x = 0, 1, 2, 10 — query at 0.1 avoids equidistant ties
    let positions = [0.0_f64, 0.0, 1.0, 0.0, 2.0, 0.0, 10.0, 0.0];
    let cloud = PointCloud2::<f64>::new(&positions, 4);

    // 3 nearby points within radius 3, but k=2
    let result = cloud.nearest_k_within_radius(&[0.1, 0.0], 1, 2, 3.0);
    assert_eq!(result.len(), 2);
    assert_eq!(result[0], 0); // dist 0.1
    assert_eq!(result[1], 1); // dist 0.9
}

#[test]
fn test_2d_nearest_k_within_radius_limits_by_radius() {
    let positions = [0.0_f64, 0.0, 1.0, 0.0, 2.0, 0.0, 10.0, 0.0];
    let cloud = PointCloud2::<f64>::new(&positions, 4);

    // k=5 but radius 0.6 → only point 0 (at dist 0.1) and point 1 (at dist 0.9 > 0.6)
    let result = cloud.nearest_k_within_radius(&[0.1, 0.0], 1, 5, 0.6);
    assert_eq!(result.len(), 5);
    assert_eq!(result[0], 0);
    assert_eq!(result[1], u32::MAX);
}

#[test]
fn test_2d_f32_nearest_k_within_radius() {
    let positions: [f32; 8] = [0.0, 0.0, 1.0, 0.0, 5.0, 0.0, 10.0, 0.0];
    let cloud = PointCloud2::<f32>::new(&positions, 4);

    // Query at 0.1: k=3 within radius 2.0 → points 0 (dist 0.1) and 1 (dist 0.9)
    let result = cloud.nearest_k_within_radius(&[0.1_f32, 0.0], 1, 3, 2.0);
    assert_eq!(result.len(), 3);
    assert_eq!(result[0], 0);
    assert_eq!(result[1], 1);
    assert_eq!(result[2], u32::MAX);
}

#[test]
fn test_3d_nearest_k_within_radius() {
    let positions = [0.0_f64, 0.0, 0.0, 1.0, 0.0, 0.0, 100.0, 0.0, 0.0];
    let cloud = PointCloud3::<f64>::new(&positions, 3);

    let result = cloud.nearest_k_within_radius(&[0.1, 0.0, 0.0], 1, 5, 2.0);
    assert_eq!(result.len(), 5);
    assert_eq!(result[0], 0); // dist 0.1
    assert_eq!(result[1], 1); // dist 0.9
    assert_eq!(result[2], u32::MAX);
}

// ── nearest_neighbor_distances on different types ────────────────────

#[test]
fn test_nn_distances_2d_f64() {
    let positions = [0.0_f64, 0.0, 3.0, 4.0];
    let cloud = PointCloud2::<f64>::new(&positions, 2);
    let dists = cloud.nearest_neighbor_distances();
    assert_eq!(dists.len(), 2);
    assert!((dists[0] - 5.0).abs() < 1e-10);
    assert!((dists[1] - 5.0).abs() < 1e-10);
}

#[test]
fn test_nn_distances_3d_f32() {
    let positions: [f32; 6] = [0.0, 0.0, 0.0, 3.0, 4.0, 0.0];
    let cloud = PointCloud3::<f32>::new(&positions, 2);
    let dists = cloud.nearest_neighbor_distances();
    assert_eq!(dists.len(), 2);
    assert!((dists[0] - 5.0).abs() < 1e-5);
    assert!((dists[1] - 5.0).abs() < 1e-5);
}

#[test]
fn test_nn_distances_single_point_f32() {
    let positions: [f32; 2] = [1.0, 2.0];
    let cloud = PointCloud2::<f32>::new(&positions, 1);
    let dists = cloud.nearest_neighbor_distances();
    assert_eq!(dists.len(), 1);
    assert!(dists[0].is_infinite());
}
