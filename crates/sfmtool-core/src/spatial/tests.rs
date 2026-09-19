// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

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

/// Many points sharing the same value on one axis is the input that the
/// KD-tree's *mutable* form cannot hold: its leaf buckets take at most 32
/// items that collide on the split axis, and the rest are refused. The
/// bulk-built tree [`PointCloud`] uses splits on medians instead and indexes
/// every one of them, which is what this asserts — `cloud.len()` counts the
/// points handed in, but the query is what proves they are all reachable.
/// The shape reproduces a real failure seen during flow matching on scenes
/// with strong vertical edges.
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

/// Many points at literally the same position — SIFT features at the same
/// quantized location, or a rotation-only reconstruction's cameras all at the
/// origin — are indexed, not refused or collapsed.
#[test]
fn test_2d_construction_exact_duplicates() {
    // 200 points all at the exact same coordinate, well past the tree's
    // 32-item leaf bucket.
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
    // Every one of them is reachable, not just a bucket's worth.
    let (offsets, indices) = cloud.within_radius(&[42.0, 17.0], 1, 0.0);
    assert_eq!(offsets, vec![0, n as u32]);
    let mut seen = indices.clone();
    seen.sort_unstable();
    seen.dedup();
    assert_eq!(seen.len(), n, "every duplicate must be its own entry");
}

/// A regular grid duplicates every coordinate on every axis at once — the
/// worst case for a median split — and a plane duplicates one axis across the
/// whole cloud. Both index completely and answer radius queries exactly.
#[test]
fn test_3d_grid_and_plane_are_indexed_completely() {
    // 16^3 lattice points.
    let side = 16usize;
    let n = side * side * side;
    let mut positions = Vec::with_capacity(n * 3);
    for i in 0..n {
        positions.push((i % side) as f32);
        positions.push(((i / side) % side) as f32);
        positions.push((i / (side * side)) as f32);
    }
    let cloud = PointCloud3::<f32>::new(&positions, n);
    assert_eq!(cloud.len(), n);
    // The 6 face neighbours of an interior lattice point sit at distance 1.
    let (offsets, _) = cloud.within_radius(&[8.0, 8.0, 8.0], 1, 1.0);
    assert_eq!(offsets[1], 7, "the point itself plus 6 face neighbours");

    // 5000 points on the plane z = 0.
    let n = 5000usize;
    let mut positions = Vec::with_capacity(n * 3);
    for i in 0..n {
        positions.push((i % 97) as f32);
        positions.push((i / 97) as f32 * 0.5);
        positions.push(0.0);
    }
    let cloud = PointCloud3::<f32>::new(&positions, n);
    let (offsets, indices) = cloud.within_radius(&positions, n, 1e9);
    assert_eq!(offsets[n], (n * n) as u32);
    assert_eq!(indices.len(), n * n, "every point sees every other");
}

// ── Documented guarantees ───────────────────────────────────────────

/// `radius` is Euclidean, not squared, and inclusive at the boundary.
#[test]
fn test_radius_is_euclidean_and_inclusive() {
    // Points at Euclidean distance 3, 4 and 5 from the origin.
    let positions = [3.0_f64, 0.0, 0.0, 4.0, 3.0, 4.0];
    let cloud = PointCloud2::<f64>::new(&positions, 3);

    let (_, indices) = cloud.within_radius(&[0.0, 0.0], 1, 4.0);
    let mut got = indices.clone();
    got.sort_unstable();
    assert_eq!(got, vec![0, 1], "the point exactly at the radius is a hit");

    // A squared radius would have swept all three in at 4.0.
    let (offsets, _) = cloud.within_radius(&[0.0, 0.0], 1, 5.0);
    assert_eq!(offsets[1], 3);

    // Radius zero keeps only what is exactly at the query point.
    let (offsets, indices) = cloud.within_radius(&[3.0, 0.0], 1, 0.0);
    assert_eq!(offsets[1], 1);
    assert_eq!(indices, vec![0]);

    // The same boundary rule for the k-bounded form.
    let got = cloud.nearest_k_within_radius(&[0.0, 0.0], 1, 3, 4.0);
    assert_eq!(got, vec![0, 1, u32::MAX]);
}

/// k-nearest results come back nearest-first and short rows are padded at the
/// end, never in the middle.
#[test]
fn test_nearest_k_is_sorted_and_padded_as_a_suffix() {
    let positions = [0.0_f64, 0.0, 1.0, 0.0, 2.0, 0.0];
    let cloud = PointCloud2::<f64>::new(&positions, 3);

    assert_eq!(cloud.nearest_k(&[0.1, 0.0], 1, 3), vec![0, 1, 2]);
    assert_eq!(cloud.nearest_k(&[1.9, 0.0], 1, 3), vec![2, 1, 0]);

    // k greater than the cloud: the real answers first, then the padding.
    assert_eq!(
        cloud.nearest_k(&[0.1, 0.0], 1, 5),
        vec![0, 1, 2, u32::MAX, u32::MAX]
    );
    // Same for the radius-bounded form, where the radius is what runs out.
    assert_eq!(
        cloud.nearest_k_within_radius(&[0.1, 0.0], 1, 3, 1.5),
        vec![0, 1, u32::MAX]
    );
    // `self_nearest_k` excludes only the point's own index.
    assert_eq!(cloud.self_nearest_k(2), vec![1, 2, 0, 2, 1, 0]);
}

/// `k == 0` is a legal no-op rather than a panic, and an empty cloud answers
/// every query without one either.
#[test]
fn test_degenerate_k_and_empty_cloud() {
    let positions = [0.0_f64, 0.0];
    let cloud = PointCloud2::<f64>::new(&positions, 1);
    assert!(cloud.nearest_k(&[0.0, 0.0], 1, 0).is_empty());
    assert!(cloud
        .nearest_k_within_radius(&[0.0, 0.0], 1, 0, 1.0)
        .is_empty());

    let empty = PointCloud2::<f64>::new(&[], 0);
    assert!(empty.is_empty());
    assert_eq!(empty.len(), 0);
    assert_eq!(empty.nearest_k(&[0.0, 0.0], 1, 2), vec![u32::MAX; 2]);
    assert_eq!(
        empty.within_radius(&[0.0, 0.0], 1, 10.0),
        (vec![0, 0], vec![])
    );
    assert!(empty.self_nearest_k(3).is_empty());
    assert!(empty.nearest_neighbor_distances().is_empty());
}

/// A query point sitting exactly on a data point finds it at distance zero,
/// and coincident points report zero to each other rather than the distance to
/// the next distinct location.
#[test]
fn test_coincident_points_and_queries() {
    let positions = [0.0_f64, 0.0, 0.0, 0.0, 5.0, 0.0];
    let cloud = PointCloud2::<f64>::new(&positions, 3);

    assert_eq!(cloud.nearest(&[5.0, 0.0], 1), vec![2]);
    let distances = cloud.nearest_neighbor_distances();
    assert_eq!(distances[0], 0.0);
    assert_eq!(distances[1], 0.0);
    assert_eq!(distances[2], 5.0);

    // Zero radius around a query that coincides with the duplicated pair.
    let (offsets, _) = cloud.within_radius(&[0.0, 0.0], 1, 0.0);
    assert_eq!(offsets[1], 2);
}

/// `nearest_neighbor_distances` reports Euclidean distances, not the squared
/// ones the metric works in.
#[test]
fn test_nearest_neighbor_distances_are_euclidean() {
    let positions = [0.0_f32, 0.0, 3.0, 4.0];
    let cloud = PointCloud2::<f32>::new(&positions, 2);
    assert_eq!(cloud.nearest_neighbor_distances(), vec![5.0, 5.0]);
}
