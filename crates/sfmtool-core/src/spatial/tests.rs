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

/// Regression: kiddo 4.2's KD-tree constructions both fail on inputs with
/// many points sharing the same value on one axis — the mutable `KdTree`
/// panics with "Too many items with the same position on one axis" once
/// more than `BUCKET_SIZE - 1` items collide, and the `ImmutableKdTree`'s
/// balance optimizer hits a pivot underflow panic ("mid > len") for
/// heavily skewed inputs. `PointCloud::new` applies deterministic
/// per-index jitter so every axis value is unique, which sidesteps both
/// bugs. This reproduces the real-world failure seen during flow matching
/// on scenes with strong vertical edges.
#[test]
fn test_2d_construction_axis_cluster() {
    // 40 points with x=100.0 plus 60 scattered points. The 40 collinear
    // points exceed the default 32 bucket size.
    let mut positions = Vec::new();
    for i in 0..40 {
        positions.push(100.0_f32);
        positions.push(i as f32 * 0.5);
    }
    for i in 0..60 {
        positions.push((i as f32).sin() * 50.0 + 50.0);
        positions.push((i as f32).cos() * 50.0 + 50.0);
    }
    let n = 100;
    let cloud = PointCloud2::<f32>::new(&positions, n);
    assert_eq!(cloud.len(), n);
    // Query near one of the collinear points; it must land on a point with
    // x=100.0 (indices 0..40).
    let result = cloud.nearest(&[100.0, 5.1], 1);
    assert!(result[0] < 40);
}

/// Regression: many points at literally the same position (e.g. SIFT
/// features at the exact same quantized location) should not panic.
#[test]
fn test_2d_construction_exact_duplicates() {
    // 200 points all at the exact same coordinate. Without jitter this
    // triggers the mutable tree's bucket-overflow panic immediately.
    let n = 200;
    let mut positions = Vec::with_capacity(n * 2);
    for _ in 0..n {
        positions.push(42.0_f32);
        positions.push(17.0_f32);
    }
    let cloud = PointCloud2::<f32>::new(&positions, n);
    assert_eq!(cloud.len(), n);
    // All points coincide; nearest to the shared location must be valid.
    let result = cloud.nearest(&[42.0, 17.0], 1);
    assert!(result[0] < n as u32);
}
