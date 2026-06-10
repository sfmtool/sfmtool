use super::*;

/// Rays converging on a known point recover it, in front of all cameras and
/// well conditioned.
#[test]
fn finite_point_recovered() {
    let target = Point3::new(0.5, -1.0, 4.0);
    let centers = vec![
        Point3::new(-2.0, 0.0, 0.0),
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(2.0, 1.0, 0.0),
    ];
    let dirs: Vec<Vector3<f64>> = centers
        .iter()
        .map(|c| (target.coords - c.coords).normalize())
        .collect();
    let offsets = [0, 3];

    let tris = triangulate_batch(&dirs, &centers, &offsets);
    assert_eq!(tris.len(), 1);
    let tri = &tris[0];
    assert!(
        (tri.point.coords - target.coords).norm() < 1e-9,
        "recovered {:?} vs target {:?}",
        tri.point,
        target
    );
    assert!(tri.in_front_of_all_cameras);
    assert!(tri.condition_number.is_finite());
    // Σ eigenvalues = 2K.
    let sum: f64 = tri.eigenvalues.iter().sum();
    assert!((sum - 6.0).abs() < 1e-9, "Σλ = {sum}, expected 2K = 6");
}

/// Parallel rays are degenerate: the smallest eigenvalue collapses, the
/// condition number is infinite, and the point is not in front (depth
/// unobservable).
#[test]
fn parallel_rays_are_degenerate() {
    let d = Vector3::new(0.0, 0.0, 1.0);
    let dirs = vec![d, d, d];
    let centers = vec![
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(1.0, 0.0, 0.0),
        Point3::new(2.0, 0.0, 0.0),
    ];
    let offsets = [0, 3];

    let tris = triangulate_batch(&dirs, &centers, &offsets);
    let tri = &tris[0];
    assert!(tri.eigenvalues[0] < 1e-9, "λ_min should collapse to ~0");
    assert!(tri.condition_number.is_infinite());
    assert!(!tri.in_front_of_all_cameras);
    let sum: f64 = tri.eigenvalues.iter().sum();
    assert!((sum - 6.0).abs() < 1e-9, "Σλ = {sum}, expected 2K = 6");
}

/// A point behind the cameras is flagged. Rays pointing toward a target
/// put it in front; reversing those rays (same line, opposite bearing)
/// puts the same recovered point behind every camera.
#[test]
fn behind_camera_flagged() {
    let centers = vec![Point3::new(-1.0, 0.0, 0.0), Point3::new(1.0, 0.0, 0.0)];
    let target = Point3::new(0.0, 0.0, 3.0);
    let dirs: Vec<Vector3<f64>> = centers
        .iter()
        .map(|c| (target.coords - c.coords).normalize())
        .collect();
    let offsets = [0, 2];

    let tris = triangulate_batch(&dirs, &centers, &offsets);
    assert!(tris[0].in_front_of_all_cameras);

    let reversed: Vec<Vector3<f64>> = dirs.iter().map(|d| -d).collect();
    let tris2 = triangulate_batch(&reversed, &centers, &offsets);
    assert!(!tris2[0].in_front_of_all_cameras);
}

/// Multiple tracks in one batch are handled independently via the offsets.
#[test]
fn multiple_tracks_independent() {
    let t0 = Point3::new(0.0, 0.0, 5.0);
    let c0 = [Point3::new(-1.0, 0.0, 0.0), Point3::new(1.0, 0.0, 0.0)];
    let d0: Vec<Vector3<f64>> = c0
        .iter()
        .map(|c| (t0.coords - c.coords).normalize())
        .collect();
    let par = Vector3::new(1.0, 0.0, 0.0);
    let c1 = [Point3::new(0.0, 0.0, 0.0), Point3::new(0.0, 1.0, 0.0)];
    let d1 = [par, par];

    let dirs: Vec<Vector3<f64>> = d0.iter().chain(d1.iter()).cloned().collect();
    let centers: Vec<Point3<f64>> = c0.iter().chain(c1.iter()).cloned().collect();
    let offsets = [0, 2, 4];

    let tris = triangulate_batch(&dirs, &centers, &offsets);
    assert_eq!(tris.len(), 2);
    assert!(tris[0].condition_number.is_finite());
    assert!((tris[0].point.coords - t0.coords).norm() < 1e-9);
    assert!(tris[1].condition_number.is_infinite());
}

/// Depth uncertainty: a well-triangulated near point has a large z-score; a
/// near-parallel distant track has a small one.
#[test]
fn depth_uncertainty_separates_finite_from_infinity() {
    let sigma = 1.0 / 1000.0; // ~1 px at f = 1000.
    let make = |target: Point3<f64>, baseline: f64| {
        let centers = vec![
            Point3::new(-baseline, 0.0, 0.0),
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(baseline, 0.0, 0.0),
        ];
        let dirs: Vec<Vector3<f64>> = centers
            .iter()
            .map(|c| (target.coords - c.coords).normalize())
            .collect();
        (dirs, centers)
    };

    // Near point with wide baseline → well observed.
    let (dn, cn) = make(Point3::new(0.0, 0.0, 5.0), 2.0);
    let offsets = [0, 3];
    let sig = vec![sigma; 3];
    let trin = triangulate_batch(&dn, &cn, &offsets);
    let dun = depth_uncertainty_batch(&trin, &dn, &cn, &offsets, &sig);
    assert!(dun[0].sigma.is_finite());
    assert!(
        dun[0].inverse_depth_z > 10.0,
        "near point z = {}",
        dun[0].inverse_depth_z
    );

    // The near (wide-baseline) track resolves far further than the far
    // (tiny-baseline) one: D_max = B⊥/σ scales with the perpendicular
    // baseline.
    assert!(
        dun[0].resolvable_distance > 100.0,
        "wide baseline resolvable {}",
        dun[0].resolvable_distance
    );

    // Far point with tiny baseline → near-parallel, depth unconstrained.
    let (df, cf) = make(Point3::new(0.0, 0.0, 5000.0), 0.01);
    let trif = triangulate_batch(&df, &cf, &offsets);
    let duf = depth_uncertainty_batch(&trif, &df, &cf, &offsets, &sig);
    assert!(
        duf[0].inverse_depth_z < 4.0,
        "far point z = {}",
        duf[0].inverse_depth_z
    );
    assert!(
        duf[0].resolvable_distance < dun[0].resolvable_distance,
        "tiny baseline resolves less far: {} vs {}",
        duf[0].resolvable_distance,
        dun[0].resolvable_distance
    );
}

/// `resolvable_distance` is computed from camera geometry + noise, not the
/// solved point, so it scales linearly with the scene (B⊥ scales, σ fixed)
/// — unlike the scale-free z-score.
#[test]
fn resolvable_distance_scales_with_baseline() {
    let sigma = 1.0 / 800.0;
    let build = |scale: f64| {
        let target = Point3::new(0.3 * scale, 0.2 * scale, 6.0 * scale);
        let centers = vec![
            Point3::new(-1.5 * scale, 0.0, 0.0),
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.5 * scale, 0.0, 0.0),
        ];
        let dirs: Vec<Vector3<f64>> = centers
            .iter()
            .map(|c| (target.coords - c.coords).normalize())
            .collect();
        (dirs, centers)
    };
    let offsets = [0, 3];
    let sig = vec![sigma; 3];

    let (d1, c1) = build(1.0);
    let r1 = depth_uncertainty_batch(
        &triangulate_batch(&d1, &c1, &offsets),
        &d1,
        &c1,
        &offsets,
        &sig,
    )[0]
    .resolvable_distance;
    let (d2, c2) = build(10.0);
    let r2 = depth_uncertainty_batch(
        &triangulate_batch(&d2, &c2, &offsets),
        &d2,
        &c2,
        &offsets,
        &sig,
    )[0]
    .resolvable_distance;
    assert!(r1.is_finite() && r1 > 0.0);
    assert!(
        (r2 / r1 - 10.0).abs() < 1e-6,
        "resolvable_distance should scale 10×: {r1} -> {r2}"
    );
}

/// inverse_depth_z is scale-free: scaling the whole scene leaves it
/// unchanged.
#[test]
fn inverse_depth_z_is_scale_invariant() {
    let sigma = 1.0 / 800.0;
    let build = |scale: f64| {
        let target = Point3::new(0.3 * scale, 0.2 * scale, 6.0 * scale);
        let centers = vec![
            Point3::new(-1.5 * scale, 0.0, 0.0),
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.5 * scale, 0.0, 0.0),
        ];
        let dirs: Vec<Vector3<f64>> = centers
            .iter()
            .map(|c| (target.coords - c.coords).normalize())
            .collect();
        (dirs, centers)
    };
    let offsets = [0, 3];
    let sig = vec![sigma; 3];

    let (d1, c1) = build(1.0);
    let t1 = triangulate_batch(&d1, &c1, &offsets);
    let z1 = depth_uncertainty_batch(&t1, &d1, &c1, &offsets, &sig)[0].inverse_depth_z;

    let (d2, c2) = build(1000.0);
    let t2 = triangulate_batch(&d2, &c2, &offsets);
    let z2 = depth_uncertainty_batch(&t2, &d2, &c2, &offsets, &sig)[0].inverse_depth_z;

    assert!(
        (z1 - z2).abs() / z1 < 1e-6,
        "z-scores differ across scale: {z1} vs {z2}"
    );
}

/// An empty offsets slice yields no tracks; an empty track (offsets
/// `[0, 0]`) yields one degenerate result that is not in front.
#[test]
fn empty_track_handled() {
    assert!(triangulate_batch(&[], &[], &[]).is_empty());
    assert!(triangulate_batch(&[], &[], &[0]).is_empty());

    let tris = triangulate_batch(&[], &[], &[0, 0]);
    assert_eq!(tris.len(), 1);
    assert!(!tris[0].in_front_of_all_cameras);
    assert!(tris[0].condition_number.is_infinite());
}
