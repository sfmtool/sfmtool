use super::*;

fn assert_unit_norm(points: &[f32], tol: f32) {
    for chunk in points.chunks_exact(3) {
        let norm = (chunk[0] * chunk[0] + chunk[1] * chunk[1] + chunk[2] * chunk[2]).sqrt();
        assert!((norm - 1.0).abs() < tol, "point not unit norm: norm={norm}");
    }
}

fn std_dev(values: &[f32]) -> f32 {
    let mean = values.iter().sum::<f32>() / values.len() as f32;
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / values.len() as f32;
    variance.sqrt()
}

fn nn_distances(points: &[f32]) -> Vec<f32> {
    let n = points.len() / 3;
    PointCloud3::<f32>::new(points, n).nearest_neighbor_distances()
}

#[test]
fn random_points_have_unit_norm() {
    let points = random_sphere_points(200, None);
    assert_eq!(points.len(), 600);
    assert_unit_norm(&points, 1e-5);
}

#[test]
fn random_points_cover_sphere() {
    // A simple smoke check: octant occupancy should be roughly balanced.
    let points = random_sphere_points(8000, None);
    let mut counts = [0usize; 8];
    for chunk in points.chunks_exact(3) {
        let i = (chunk[0] >= 0.0) as usize
            | (((chunk[1] >= 0.0) as usize) << 1)
            | (((chunk[2] >= 0.0) as usize) << 2);
        counts[i] += 1;
    }
    // Uniform expectation per octant is 1000; allow plenty of slack.
    for c in counts {
        assert!(c > 700 && c < 1300, "octant count out of range: {c}");
    }
}

#[test]
fn relaxation_improves_uniformity() {
    let n = 500;
    let initial = random_sphere_points(n, None);
    let rough_std = std_dev(&nn_distances(&initial));

    let mut relaxed = initial.clone();
    relax_sphere_points(&mut relaxed, &RelaxConfig::default());
    let smooth_std = std_dev(&nn_distances(&relaxed));

    // Random uniform sampling has high NN variance; relaxation should
    // cut it dramatically (well over 2x in practice).
    assert!(
        smooth_std < rough_std * 0.5,
        "relaxation did not cut NN variance enough: rough_std={rough_std}, smooth_std={smooth_std}"
    );
}

#[test]
fn relaxed_points_remain_on_unit_sphere() {
    let mut points = random_sphere_points(300, None);
    relax_sphere_points(&mut points, &RelaxConfig::default());
    assert_unit_norm(&points, 1e-4);
}

#[test]
fn evenly_distributed_returns_unit_norm_points() {
    let points = evenly_distributed_sphere_points(50, &RelaxConfig::default());
    assert_eq!(points.len(), 150);
    assert_unit_norm(&points, 1e-4);
}

#[test]
fn empty_input_is_a_noop() {
    let mut empty: Vec<f32> = vec![];
    relax_sphere_points(&mut empty, &RelaxConfig::default());
    assert!(empty.is_empty());
}

#[test]
fn single_point_is_a_noop() {
    let mut single = vec![1.0, 0.0, 0.0];
    relax_sphere_points(&mut single, &RelaxConfig::default());
    assert_eq!(single, vec![1.0, 0.0, 0.0]);
}

#[test]
fn two_points_become_approximately_antipodal() {
    // With fixed step length the algorithm has a limit cycle around
    // antipodal (≈ one step of angular travel), so the threshold here
    // is loose. The point is just to confirm the repulsion drives them
    // into opposite hemispheres from a random start.
    let mut points = random_sphere_points(2, None);
    let config = RelaxConfig {
        iterations: 500,
        ..Default::default()
    };
    relax_sphere_points(&mut points, &config);
    let dot = points[0] * points[3] + points[1] * points[4] + points[2] * points[5];
    assert!(
        dot < -0.95,
        "two points did not converge near antipodal: dot={dot}"
    );
}

#[test]
fn zero_iterations_leaves_points_unchanged() {
    let mut points = random_sphere_points(10, None);
    let snapshot = points.clone();
    let config = RelaxConfig {
        iterations: 0,
        ..Default::default()
    };
    relax_sphere_points(&mut points, &config);
    assert_eq!(points, snapshot);
}

#[test]
fn same_seed_produces_identical_random_points() {
    let a = random_sphere_points(50, Some(42));
    let b = random_sphere_points(50, Some(42));
    assert_eq!(a, b);
}

#[test]
fn different_seeds_produce_different_random_points() {
    let a = random_sphere_points(50, Some(1));
    let b = random_sphere_points(50, Some(2));
    assert_ne!(a, b);
}

#[test]
fn seed_makes_evenly_distributed_deterministic() {
    let cfg = RelaxConfig {
        seed: Some(7),
        ..Default::default()
    };
    let a = evenly_distributed_sphere_points(80, &cfg);
    let b = evenly_distributed_sphere_points(80, &cfg);
    assert_eq!(a, b);
}
