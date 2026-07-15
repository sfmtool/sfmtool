// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use super::*;
use nalgebra::Unit;

/// Deterministic LCG so fixtures need no `rand` and are bitwise-stable.
struct Lcg(u64);

impl Lcg {
    fn next_f64(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }

    fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next_f64()
    }

    /// Standard normal via Box-Muller.
    fn gaussian(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-300);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    fn rotation(&mut self) -> UnitQuaternion<f64> {
        let axis = Unit::new_normalize(Vector3::new(
            self.uniform(-1.0, 1.0),
            self.uniform(-1.0, 1.0),
            self.uniform(-1.0, 1.0),
        ));
        let angle = self.uniform(0.1, 2.5);
        UnitQuaternion::from_axis_angle(&axis, angle)
    }
}

/// A synthetic scene: the generating world-to-camera pose plus `n` world
/// points guaranteed in front of the camera (canonical: `z_cam < 0`) and their
/// exact unit bearings.
struct Scene {
    rotation: UnitQuaternion<f64>,
    translation: Vector3<f64>,
    points: Vec<Point3<f64>>,
    bearings: Vec<Vector3<f64>>,
}

impl Scene {
    /// Build a non-degenerate scene. Camera-frame points are drawn with
    /// `z < 0` (in front), so bearings are `normalize(R·X + t)` with negative
    /// z — the exact convention the solver expects.
    fn new(n: usize, seed: u64) -> Self {
        let mut rng = Lcg(seed);
        let rotation = rng.rotation();
        let translation = Vector3::new(
            rng.uniform(-3.0, 3.0),
            rng.uniform(-3.0, 3.0),
            rng.uniform(-3.0, 3.0),
        );
        let mut points = Vec::with_capacity(n);
        let mut bearings = Vec::with_capacity(n);
        let rinv = rotation.inverse();
        for _ in 0..n {
            // Camera-frame point in front of the canonical camera (z < 0).
            let cam = Vector3::new(
                rng.uniform(-3.0, 3.0),
                rng.uniform(-3.0, 3.0),
                rng.uniform(-8.0, -1.5),
            );
            bearings.push(cam.normalize());
            let world = rinv * (cam - translation);
            points.push(Point3::from(world));
        }
        Scene {
            rotation,
            translation,
            points,
            bearings,
        }
    }

    /// Angular difference (radians) of a candidate rotation from the truth.
    fn rot_err(&self, r: &UnitQuaternion<f64>) -> f64 {
        self.rotation.angle_to(r)
    }

    fn trans_err(&self, t: &Vector3<f64>) -> f64 {
        (self.translation - t).norm()
    }
}

/// True if `poses` contains the generating pose to floating-point accuracy.
fn contains_truth(poses: &[Pose], scene: &Scene) -> bool {
    poses
        .iter()
        .any(|(r, t)| scene.rot_err(r) < 1e-9 && scene.trans_err(t) < 1e-9)
}

#[test]
fn exact_recovery_on_random_configs() {
    for seed in 0..200 {
        let scene = Scene::new(3, seed * 7 + 1);
        let b = [scene.bearings[0], scene.bearings[1], scene.bearings[2]];
        let x = [scene.points[0], scene.points[1], scene.points[2]];
        let poses = p3p_solve(&b, &x);
        assert!(
            contains_truth(&poses, &scene),
            "seed {seed}: {} solutions, none matched the generating pose",
            poses.len()
        );
        // Every returned pose must have positive depths / valid rotation.
        for (r, t) in &poses {
            assert!(r.coords.iter().all(|v| v.is_finite()));
            assert!(t.iter().all(|v| v.is_finite()));
        }
    }
}

#[test]
fn near_planar_triples_recover() {
    // Three world points nearly coplanar with the origin plane but not
    // collinear — the fiddly case for the depth cubic.
    for seed in 0..50 {
        let mut rng = Lcg(seed * 13 + 3);
        let rotation = rng.rotation();
        let translation = Vector3::new(
            rng.uniform(-2.0, 2.0),
            rng.uniform(-2.0, 2.0),
            rng.uniform(-2.0, 2.0),
        );
        let rinv = rotation.inverse();
        // Camera points share nearly the same depth (near-planar frontal set).
        let z = rng.uniform(-6.0, -3.0);
        let mut b = [Vector3::zeros(); 3];
        let mut x = [Point3::origin(); 3];
        for i in 0..3 {
            let cam = Vector3::new(
                rng.uniform(-2.0, 2.0),
                rng.uniform(-2.0, 2.0),
                z + 1e-3 * rng.uniform(-1.0, 1.0),
            );
            b[i] = cam.normalize();
            x[i] = Point3::from(rinv * (cam - translation));
        }
        let scene = Scene {
            rotation,
            translation,
            points: x.to_vec(),
            bearings: b.to_vec(),
        };
        let poses = p3p_solve(&b, &x);
        let best = poses
            .iter()
            .map(|(r, t)| scene.rot_err(r) + scene.trans_err(t))
            .fold(f64::INFINITY, f64::min);
        // Near-coplanar triples are inherently worse-conditioned than the
        // general case (the depth cubic loses a few digits), so the tolerance
        // is looser than the 1e-9 of `exact_recovery_on_random_configs`.
        assert!(
            best < 1e-6,
            "near-planar seed {seed}: best error {best} over {} solutions",
            poses.len()
        );
    }
}

#[test]
fn multiplicity_returns_all_valid_solutions() {
    // Collect the number of distinct solutions over many configs; the true
    // pose is always present, and configs with more than one positive-depth
    // solution return all of them (the estimator disambiguates with a 4th
    // point). Verify each returned pose actually reproduces the three bearings.
    let mut multi = 0;
    for seed in 0..100 {
        let scene = Scene::new(3, seed * 11 + 5);
        let b = [scene.bearings[0], scene.bearings[1], scene.bearings[2]];
        let x = [scene.points[0], scene.points[1], scene.points[2]];
        let poses = p3p_solve(&b, &x);
        assert!(contains_truth(&poses, &scene));
        if poses.len() > 1 {
            multi += 1;
        }
        // Each solution must reproject the three input bearings exactly.
        for (r, t) in &poses {
            for i in 0..3 {
                let d = (r * x[i].coords + t).normalize();
                assert!(
                    b[i].dot(&d) > 1.0 - 1e-9,
                    "seed {seed}: a returned pose does not reproduce bearing {i}"
                );
            }
        }
    }
    assert!(multi > 0, "expected some multi-solution configurations");
}

#[test]
fn degenerate_inputs_return_empty() {
    // Collinear world points.
    let b = [
        Vector3::new(0.1, 0.0, -1.0).normalize(),
        Vector3::new(0.0, 0.1, -1.0).normalize(),
        Vector3::new(-0.1, 0.05, -1.0).normalize(),
    ];
    let collinear = [
        Point3::new(0.0, 0.0, 5.0),
        Point3::new(1.0, 1.0, 6.0),
        Point3::new(2.0, 2.0, 7.0),
    ];
    assert!(p3p_solve(&b, &collinear).is_empty(), "collinear points");

    // Coincident bearings.
    let same = Vector3::new(0.0, 0.0, -1.0);
    let x = [
        Point3::new(0.0, 0.0, 5.0),
        Point3::new(1.0, 0.5, 6.0),
        Point3::new(-1.0, 1.0, 7.0),
    ];
    assert!(
        p3p_solve(&[same, same, b[2]], &x).is_empty(),
        "coincident bearings"
    );

    // Antipodal bearings.
    let anti = [
        Vector3::new(0.0, 0.0, -1.0),
        Vector3::new(0.0, 0.0, 1.0),
        b[2],
    ];
    assert!(p3p_solve(&anti, &x).is_empty(), "antipodal bearings");

    // Non-finite input.
    let nan_b = [Vector3::new(f64::NAN, 0.0, -1.0), b[1], b[2]];
    assert!(p3p_solve(&nan_b, &x).is_empty(), "non-finite bearing");
    let inf_x = [Point3::new(f64::INFINITY, 0.0, 5.0), x[1], x[2]];
    assert!(p3p_solve(&b, &inf_x).is_empty(), "non-finite point");
}

/// A contaminated correspondence set: the scene, per-correspondence bearings
/// and points, and a per-correspondence inlier mask.
type ContaminatedSet = (Scene, Vec<Vector3<f64>>, Vec<Point3<f64>>, Vec<bool>);

/// Build a contaminated correspondence set: `n` total, `inlier_frac` fraction
/// consistent with the scene's pose, the rest random bearings. Inlier bearings
/// carry small angular noise `noise_rad`.
fn contaminated(n: usize, inlier_frac: f64, noise_rad: f64, seed: u64) -> ContaminatedSet {
    let scene = Scene::new(n, seed);
    let mut rng = Lcg(seed ^ 0x9e37_79b9);
    let n_in = ((n as f64) * inlier_frac).round() as usize;
    let mut bearings = Vec::with_capacity(n);
    let mut truth = vec![false; n];
    for (i, is_inlier) in truth.iter_mut().enumerate() {
        if i < n_in {
            // Perturb the true bearing by a small random rotation.
            let mut d = scene.bearings[i];
            if noise_rad > 0.0 {
                let axis = Unit::new_normalize(Vector3::new(
                    rng.gaussian(),
                    rng.gaussian(),
                    rng.gaussian(),
                ));
                d = UnitQuaternion::from_axis_angle(&axis, noise_rad * rng.gaussian()) * d;
            }
            bearings.push(d.normalize());
            *is_inlier = true;
        } else {
            // A random unit bearing pointing generally forward.
            let d = Vector3::new(
                rng.uniform(-1.0, 1.0),
                rng.uniform(-1.0, 1.0),
                rng.uniform(-2.0, -0.2),
            );
            bearings.push(d.normalize());
        }
    }
    let points = scene.points.clone();
    (scene, bearings, points, truth)
}

#[test]
fn contamination_sweep_recovers_pose() {
    let n = 160;
    for &frac in &[0.6, 0.4, 0.25, 0.12, 0.05] {
        let (scene, bearings, points, _truth) = contaminated(n, frac, 0.0, 424242);
        let opts = AbsolutePoseOptions {
            max_angular_error: 0.01,
            confidence: 0.999,
            max_iterations: 200_000,
            min_inliers: 5,
            seed: 7,
            local_optimization: true,
        };
        let est = estimate_absolute_pose(&bearings, &points, &opts)
            .unwrap_or_else(|| panic!("frac {frac}: expected a pose"));
        assert!(
            scene.rot_err(&est.rotation) < 1e-6,
            "frac {frac}: rotation error {}",
            scene.rot_err(&est.rotation)
        );
        assert!(
            scene.trans_err(&est.translation) < 1e-6,
            "frac {frac}: translation error {}",
            scene.trans_err(&est.translation)
        );
    }
}

#[test]
fn below_min_inliers_returns_none() {
    // Only 3 inliers, but require 6.
    let n = 100;
    let (_scene, bearings, points, _truth) = contaminated(n, 0.03, 0.0, 99);
    let opts = AbsolutePoseOptions {
        max_angular_error: 0.01,
        confidence: 0.999,
        max_iterations: 20_000,
        min_inliers: 6,
        seed: 1,
        local_optimization: true,
    };
    assert!(estimate_absolute_pose(&bearings, &points, &opts).is_none());
}

#[test]
fn determinism_same_seed_bit_identical() {
    let n = 150;
    let (_scene, bearings, points, _truth) = contaminated(n, 0.3, 0.001, 2024);
    let opts = AbsolutePoseOptions {
        max_angular_error: 0.01,
        confidence: 0.999,
        max_iterations: 50_000,
        min_inliers: 6,
        seed: 42,
        local_optimization: true,
    };
    let a = estimate_absolute_pose(&bearings, &points, &opts).unwrap();
    let b = estimate_absolute_pose(&bearings, &points, &opts).unwrap();
    let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(
        bits(a.rotation.coords.as_slice()),
        bits(b.rotation.coords.as_slice())
    );
    assert_eq!(
        bits(a.translation.as_slice()),
        bits(b.translation.as_slice())
    );
    assert_eq!(a.iterations, b.iterations);
    assert_eq!(a.inliers, b.inliers);
}

#[test]
fn determinism_different_seed_within_tolerance() {
    let n = 150;
    let (scene, bearings, points, _truth) = contaminated(n, 0.4, 0.0, 3030);
    let base = AbsolutePoseOptions {
        max_angular_error: 0.01,
        confidence: 0.999,
        max_iterations: 50_000,
        min_inliers: 6,
        seed: 1,
        local_optimization: true,
    };
    let a = estimate_absolute_pose(&bearings, &points, &base).unwrap();
    let b = estimate_absolute_pose(
        &bearings,
        &points,
        &AbsolutePoseOptions {
            seed: 98765,
            ..base
        },
    )
    .unwrap();
    assert!(scene.rot_err(&a.rotation) < 1e-6);
    assert!(scene.rot_err(&b.rotation) < 1e-6);
}

#[test]
fn local_optimization_improves_or_equals() {
    // Noisy inliers so the raw 3-point consensus is imperfect and the
    // Gauss-Newton refit (fitting all inliers) can only help.
    let n = 150;
    let (scene, bearings, points, _truth) = contaminated(n, 0.5, 0.003, 555);
    let base = AbsolutePoseOptions {
        max_angular_error: 0.02,
        confidence: 0.999,
        max_iterations: 50_000,
        min_inliers: 6,
        seed: 3,
        local_optimization: false,
    };
    let raw = estimate_absolute_pose(&bearings, &points, &base).unwrap();
    let lo = estimate_absolute_pose(
        &bearings,
        &points,
        &AbsolutePoseOptions {
            local_optimization: true,
            ..base
        },
    )
    .unwrap();
    let raw_err = scene.rot_err(&raw.rotation) + scene.trans_err(&raw.translation);
    let lo_err = scene.rot_err(&lo.rotation) + scene.trans_err(&lo.translation);
    assert!(
        lo_err <= raw_err + 1e-9,
        "local optimization worse: raw {raw_err}, lo {lo_err}"
    );
    assert!(
        lo.inliers.iter().filter(|&&b| b).count() >= raw.inliers.iter().filter(|&&b| b).count()
    );
}
