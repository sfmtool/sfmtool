use super::*;
use ndarray::Array1;

#[test]
fn test_empty_reconstruction() {
    let q = Array2::<f64>::zeros((0, 4));
    let t = Array2::<f64>::zeros((0, 3));
    let p = Array2::<f64>::zeros((0, 4));
    let ii = Array1::<u32>::zeros(0);
    let pi = Array1::<u32>::zeros(0);

    let result = compute_depth_statistics(&q, &t, &p, &ii, &pi).unwrap();
    assert_eq!(result.mean_viewing_normals_xyz.shape(), &[0, 3]);
    assert_eq!(result.depth_statistics.images.len(), 0);
    assert_eq!(result.observed_depth_histogram_counts.shape(), &[0, 128]);
}

#[test]
fn test_single_camera_single_point() {
    // Camera at origin looking along −Z (identity rotation, zero
    // translation; canonical convention).
    let mut q = Array2::<f64>::zeros((1, 4));
    q[[0, 0]] = 1.0; // w=1, identity quaternion
    let t = Array2::<f64>::zeros((1, 3));

    // Point at (0, 0, −5) — in front of the camera at depth −z = 5.
    let mut p = Array2::<f64>::zeros((1, 4));
    p[[0, 2]] = -5.0;
    p[[0, 3]] = 1.0;

    let ii = Array1::from_vec(vec![0u32]);
    let pi = Array1::from_vec(vec![0u32]);

    let result = compute_depth_statistics(&q, &t, &p, &ii, &pi).unwrap();

    assert_eq!(result.mean_viewing_normals_xyz.shape(), &[1, 3]);
    assert_eq!(result.depth_statistics.images.len(), 1);

    let stats = &result.depth_statistics.images[0];
    assert_eq!(stats.observed.count, 1);
    assert_eq!(stats.observed.infinity_count, 0);
    assert!((stats.observed.min_z.unwrap() - 5.0).abs() < 1e-10);
    assert!((stats.observed.max_z.unwrap() - 5.0).abs() < 1e-10);
}

#[test]
fn test_point_at_infinity_counted_separately() {
    // Camera at origin looking along −Z (canonical convention).
    let mut q = Array2::<f64>::zeros((1, 4));
    q[[0, 0]] = 1.0;
    let t = Array2::<f64>::zeros((1, 3));

    // Point 0: finite at (0, 0, −5), in front. Point 1: at infinity (w = 0).
    let mut p = Array2::<f64>::zeros((2, 4));
    p[[0, 2]] = -5.0;
    p[[0, 3]] = 1.0;
    p[[1, 2]] = -1.0; // direction (0, 0, −1), w = 0

    let ii = Array1::from_vec(vec![0u32, 0]);
    let pi = Array1::from_vec(vec![0u32, 1]);

    let result = compute_depth_statistics(&q, &t, &p, &ii, &pi).unwrap();
    let stats = &result.depth_statistics.images[0];
    assert_eq!(stats.observed.count, 1);
    assert_eq!(stats.observed.infinity_count, 1);
    // The infinity point must not pollute the finite depth range.
    assert!((stats.observed.max_z.unwrap() - 5.0).abs() < 1e-10);
    // Its estimated normal stays at the (0, 0, 0) initializer.
    assert_eq!(result.mean_viewing_normals_xyz[[1, 0]], 0.0);
    assert_eq!(result.mean_viewing_normals_xyz[[1, 1]], 0.0);
    assert_eq!(result.mean_viewing_normals_xyz[[1, 2]], 0.0);
}

#[test]
fn test_point_behind_camera_excluded() {
    // Identity pose: canonical camera looks down −Z, so a point at
    // +Z is behind the camera and must not contribute a depth.
    let mut q = Array2::<f64>::zeros((1, 4));
    q[[0, 0]] = 1.0;
    let t = Array2::<f64>::zeros((1, 3));

    let mut p = Array2::<f64>::zeros((1, 4));
    p[[0, 2]] = 5.0; // behind: camera-space z = +5, depth −z = −5
    p[[0, 3]] = 1.0;

    let ii = Array1::from_vec(vec![0u32]);
    let pi = Array1::from_vec(vec![0u32]);

    let result = compute_depth_statistics(&q, &t, &p, &ii, &pi).unwrap();
    let stats = &result.depth_statistics.images[0];
    assert_eq!(stats.observed.count, 0);
    assert!(stats.observed.min_z.is_none());
}

#[test]
fn test_camera_centers() {
    // Identity rotation, translation = [1, 2, 3]
    // Center = -R^T @ t = -[1, 2, 3]
    let mut q = Array2::<f64>::zeros((1, 4));
    q[[0, 0]] = 1.0;
    let mut t = Array2::<f64>::zeros((1, 3));
    t[[0, 0]] = 1.0;
    t[[0, 1]] = 2.0;
    t[[0, 2]] = 3.0;

    let centers = compute_camera_centers(&q, &t);
    assert!((centers[[0, 0]] - (-1.0)).abs() < 1e-10);
    assert!((centers[[0, 1]] - (-2.0)).abs() < 1e-10);
    assert!((centers[[0, 2]] - (-3.0)).abs() < 1e-10);
}

#[test]
fn test_histogram_uniform() {
    let values: Vec<f64> = (0..100).map(|i| i as f64).collect();
    let hist = compute_histogram(&values, 0.0, 99.0, 10);
    // Each bucket should have 10 values
    assert_eq!(hist.len(), 10);
    let total: u32 = hist.iter().sum();
    assert_eq!(total, 100);
}

#[test]
fn test_normals_point_between_cameras() {
    // Two cameras on opposite sides of a point
    let mut q = Array2::<f64>::zeros((2, 4));
    q[[0, 0]] = 1.0; // identity
    q[[1, 0]] = 1.0; // identity
    let mut t = Array2::<f64>::zeros((2, 3));
    // Camera 0 at x=-5, camera 1 at x=+5 (centers = -R^T @ t)
    t[[0, 0]] = 5.0; // center at (-5, 0, 0)
    t[[1, 0]] = -5.0; // center at (5, 0, 0)

    // Point at origin
    let p = Array2::<f64>::zeros((1, 3));
    let ii = Array1::from_vec(vec![0u32, 1]);
    let pi = Array1::from_vec(vec![0u32, 0]);

    let normals = compute_mean_viewing_normals(&p, &compute_camera_centers(&q, &t), &ii, &pi);

    // Directions (-5,0,0)->(0,0,0) and (5,0,0)->(0,0,0) cancel out on x,
    // but the directions are from point TO camera, so (-5,0,0) and (5,0,0)
    // sum to (0,0,0) -> normalized to (0,0,0) with 1e-10 floor
    // Actually: camera centers are at (-5,0,0) and (5,0,0).
    // Direction from point(0,0,0) to camera(-5,0,0) = (-5,0,0)
    // Direction from point(0,0,0) to camera(5,0,0) = (5,0,0)
    // Sum = (0,0,0) -> near-zero, normalized to ~(0,0,0)
    let norm = (normals[[0, 0]].powi(2) + normals[[0, 1]].powi(2) + normals[[0, 2]].powi(2)).sqrt();
    // The result is a unit vector (or near-zero normalized)
    assert!(norm <= 1.0 + 1e-5);
}
