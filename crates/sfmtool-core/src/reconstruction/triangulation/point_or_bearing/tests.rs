// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::camera::{CameraIntrinsics, CameraModel};
use nalgebra::{Rotation2, UnitQuaternion};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use rand_distr::StandardNormal;

const PLAIN: PointBearingFitOptions = PointBearingFitOptions {
    soft_l1_scale: None,
    max_iterations: DEFAULT_POINT_FIT_MAX_ITERATIONS,
};

const ROBUST: PointBearingFitOptions = PointBearingFitOptions {
    soft_l1_scale: Some(DEFAULT_SOFT_L1_SCALE),
    max_iterations: DEFAULT_POINT_FIT_MAX_ITERATIONS,
};

const T: f64 = DEFAULT_DEPTH_LIKELIHOOD_RATIO_THRESHOLD;

/// `k` camera centres spread over a 2-unit baseline in x and a little in y.
fn spread_centers(k: usize) -> Vec<Point3<f64>> {
    (0..k)
        .map(|i| {
            let s = i as f64 / (k - 1) as f64;
            Point3::new(-1.0 + 2.0 * s, 0.3 * (3.0 * s).sin(), 0.1 * s)
        })
        .collect()
}

/// `k` camera centres on an arc of `arc_deg` degrees and radius 5 around the
/// origin, starting at a random angle, at slightly varying heights: an
/// object-centric capture.
fn arc_centers(k: usize, arc_deg: f64, rng: &mut StdRng) -> Vec<Point3<f64>> {
    let start: f64 = rng.random_range(0.0..360.0);
    (0..k)
        .map(|i| {
            let a = (start + arc_deg * i as f64 / (k - 1) as f64).to_radians();
            Point3::new(
                5.0 * a.cos(),
                rng.random_range(-0.5..0.5) + 1.0,
                5.0 * a.sin(),
            )
        })
        .collect()
}

/// `d` turned by an angle of Gaussian noise of standard deviation `sigma` in
/// each of the two directions perpendicular to it.
fn perturb(d: &Vector3<f64>, sigma: f64, rng: &mut StdRng) -> Vector3<f64> {
    let (b1, b2) = tangent_basis(d);
    let n1: f64 = rng.sample(StandardNormal);
    let n2: f64 = rng.sample(StandardNormal);
    (d + sigma * (n1 * b1 + n2 * b2)).normalize()
}

/// Rays from each centre towards `target`, perturbed by `sigma`.
fn rays_to(
    target: &Point3<f64>,
    centers: &[Point3<f64>],
    sigma: f64,
    rng: &mut StdRng,
) -> Vec<Vector3<f64>> {
    centers
        .iter()
        .map(|c| perturb(&(target - c).normalize(), sigma, rng))
        .collect()
}

/// Rays from each centre along the bearing `u`, perturbed by `sigma`.
fn rays_along(
    u: &Vector3<f64>,
    centers: &[Point3<f64>],
    sigma: f64,
    rng: &mut StdRng,
) -> Vec<Vector3<f64>> {
    centers.iter().map(|_| perturb(u, sigma, rng)).collect()
}

/// Isotropic weights at one angular noise for every ray.
fn iso(dirs: &[Vector3<f64>], sigma: f64) -> Vec<Matrix2x3<f64>> {
    isotropic_ray_weights(dirs, &vec![sigma; dirs.len()])
}

fn far_bearing() -> Vector3<f64> {
    Vector3::new(0.2, -0.1, 1.0).normalize()
}

fn score(dirs: &[Vector3<f64>], centers: &[Point3<f64>], sigma: f64) -> BearingScore {
    bearing_score(dirs, centers, &iso(dirs, sigma)).unwrap()
}

fn fit(
    dirs: &[Vector3<f64>],
    centers: &[Point3<f64>],
    sigma: f64,
    start: Option<Point3<f64>>,
    options: &PointBearingFitOptions,
) -> PointBearingFit {
    fit_point_and_bearing(dirs, centers, &iso(dirs, sigma), start, None, options).unwrap()
}

/// For any track the point fit costs no more than the bearing, and rays exactly
/// along one bearing from every camera give `Λ = 0` at `ρ = 0`.
#[test]
fn point_model_nests_the_bearing() {
    let centers = spread_centers(6);

    let u = far_bearing();
    let dirs = vec![u; 6];
    let f = fit(&dirs, &centers, 1e-3, None, &PLAIN);
    assert_eq!(f.inverse_depth, 0.0);
    assert!(f.point.is_none());
    assert_eq!(f.depth_likelihood_ratio, 0.0);
    assert!(f.bearing_cost < 1e-12, "{}", f.bearing_cost);
    assert!((f.bearing - u).norm() < 1e-9);
    let s = score(&dirs, &centers, 1e-3);
    assert!(s.depth_score < 1e-12, "{}", s.depth_score);
    assert_eq!(s.midpoint_bound, 0.0);

    let mut rng = StdRng::seed_from_u64(1);
    for trial in 0..200 {
        let distance = 10f64.powf(rng.random_range(0.5..5.0));
        let target = Point3::from(far_bearing() * distance);
        let dirs = rays_to(&target, &centers, 1e-3, &mut rng);
        for options in [PLAIN, ROBUST] {
            let f = fit(&dirs, &centers, 1e-3, None, &options);
            assert!(
                f.point_cost <= f.bearing_cost,
                "trial {trial}: {} > {}",
                f.point_cost,
                f.bearing_cost
            );
            assert!(f.depth_likelihood_ratio >= 0.0);
            assert!(f.inverse_depth >= 0.0);
        }
        let s = score(&dirs, &centers, 1e-3);
        assert!(s.depth_score >= 0.0);
        assert!(s.midpoint_bound >= 0.0 && s.midpoint_bound <= s.bearing_cost);
    }
}

/// The eigenvector bearing matches a Gauss-Newton fit of the bearing on the
/// same sine residuals, started away from it, and its cost matches.
#[test]
fn closed_form_bearing_is_the_least_squares_bearing() {
    let mut rng = StdRng::seed_from_u64(2);
    let centers = spread_centers(9);
    let sigma: Vec<f64> = (0..9).map(|i| 1e-3 * (1.0 + 0.2 * i as f64)).collect();
    let dirs = rays_along(&far_bearing(), &centers, 2e-3, &mut rng);
    let weights = isotropic_ray_weights(&dirs, &sigma);
    let track = Track::new(&dirs, &centers, &weights, None).unwrap();
    let (u, cost) = track.closed_form_bearing();

    let start = (far_bearing() + Vector3::new(0.05, 0.03, 0.0)).normalize();
    let gn = track.refine(start, 0.0, true, None, 50);
    assert!((gn.u - u).norm() < 1e-8, "{:?} vs {:?}", gn.u, u);
    assert!(
        (gn.cost - cost).abs() < 1e-8 * cost,
        "{} vs {cost}",
        gn.cost
    );

    // The cost is the smallest eigenvalue of Σ wᵢ (I − dᵢdᵢᵀ), wᵢ = 1/σᵢ².
    let mut m = Matrix3::zeros();
    for (d, s) in dirs.iter().zip(&sigma) {
        m += (Matrix3::identity() - d * d.transpose()) / (s * s);
    }
    let lambda_min = SymmetricEigen::new(m).eigenvalues.min();
    assert!((lambda_min - cost).abs() < 1e-6 * cost.max(1.0));
}

/// The analytic Jacobian of each ray's residual matches central differences in
/// the two tangent directions of `u` and in `ρ`, at `ρ > 0` and at `ρ = 0`,
/// for isotropic and anisotropic weights.
#[test]
fn analytic_jacobian_matches_finite_differences() {
    let mut rng = StdRng::seed_from_u64(3);
    let centers = spread_centers(5);
    let target = Point3::new(1.0, 0.5, 8.0);
    let dirs = rays_to(&target, &centers, 1e-2, &mut rng);
    let anisotropic: Vec<Matrix2x3<f64>> = dirs
        .iter()
        .map(|d| {
            let m = Matrix2::new(800.0, 150.0, -60.0, 300.0);
            m * isotropic_ray_weight(d, 1.0) + Matrix2x3::from_element(5.0)
        })
        .collect();
    for weights in [iso(&dirs, 2e-3), anisotropic] {
        let track = Track::new(&dirs, &centers, &weights, None).unwrap();
        let u = (target - track.anchor).normalize();
        let (t1, t2) = tangent_basis(&u);
        for rho in [0.0, 0.11] {
            for ray in &track.rays {
                let (_, j) = ray.residual_and_jacobian(&u, &t1, &t2, rho).unwrap();
                let h = 1e-7;
                let at = |d1: f64, d2: f64, dr: f64| {
                    let un = (u + d1 * t1 + d2 * t2).normalize();
                    ray.residual(&un, rho + dr).unwrap()
                };
                let cols = [
                    (at(h, 0.0, 0.0) - at(-h, 0.0, 0.0)) / (2.0 * h),
                    (at(0.0, h, 0.0) - at(0.0, -h, 0.0)) / (2.0 * h),
                    (at(0.0, 0.0, h) - at(0.0, 0.0, -h)) / (2.0 * h),
                ];
                for (c, fd) in cols.iter().enumerate() {
                    for r in 0..2 {
                        let a = j[(r, c)];
                        let tol = 1e-5 * a.abs().max(1.0);
                        assert!(
                            (a - fd[r]).abs() < tol,
                            "rho {rho}, J[{r},{c}] analytic {a} vs fd {}",
                            fd[r]
                        );
                    }
                }
            }
        }
    }
}

/// Fractions of `values` above each threshold against `½ P(χ²₁ > t)`, within
/// four standard deviations of the sampling error.
fn check_half_chi_square(name: &str, values: &[f64]) {
    // ½ P(χ²₁ > t) for t = 0, 2.706, 6.635.
    let thresholds = [(0.0, 0.5), (2.706, 0.05), (6.635, 0.005)];
    let n = values.len() as f64;
    for (t, expected) in thresholds {
        let frac = values.iter().filter(|&&v| v > t + 1e-9).count() as f64 / n;
        let sd = (expected * (1.0 - expected) / n).sqrt();
        assert!(
            (frac - expected).abs() < 4.0 * sd,
            "{name}: fraction above {t} is {frac}, expected {expected} ± {}",
            4.0 * sd
        );
    }
}

/// True bearings with Gaussian angular noise of the stated σ: the fraction
/// whose statistic exceeds `t` is `½ P(χ²₁ > t)`, for the score and for `Λ`,
/// in two geometries.
#[test]
fn statistic_is_half_chi_square_on_true_bearings() {
    let mut rng = StdRng::seed_from_u64(4);
    let sigma = 1e-3;
    let geometries = [
        (spread_centers(8), far_bearing()),
        // Three views, the bearing at 45° to the baseline.
        (spread_centers(3), Vector3::new(0.7, 0.0, 0.7).normalize()),
    ];
    for (g, (centers, u)) in geometries.iter().enumerate() {
        let scores: Vec<f64> = (0..40000)
            .map(|_| score(&rays_along(u, centers, sigma, &mut rng), centers, sigma).depth_score)
            .collect();
        check_half_chi_square(&format!("score, geometry {g}"), &scores);

        let ratios: Vec<f64> = (0..20000)
            .map(|_| {
                let dirs = rays_along(u, centers, sigma, &mut rng);
                fit(&dirs, centers, sigma, None, &PLAIN).depth_likelihood_ratio
            })
            .collect();
        check_half_chi_square(&format!("likelihood ratio, geometry {g}"), &ratios);
    }
}

/// Finite points at increasing distance: `Λ` falls as the distance grows, and
/// where `Λ` is large the fitted point is the true one.
#[test]
fn recovers_finite_points_and_ranks_them_by_distance() {
    let centers = spread_centers(10);
    let distances = [5.0, 20.0, 80.0, 320.0, 1280.0, 5120.0];

    // Noise-free: Λ falls strictly with distance, and the point is exact.
    let mut rng = StdRng::seed_from_u64(5);
    let mut last = f64::INFINITY;
    for &dist in &distances {
        let target = Point3::from(far_bearing() * dist);
        let dirs = rays_to(&target, &centers, 0.0, &mut rng);
        let f = fit(&dirs, &centers, 1e-3, None, &PLAIN);
        assert!(
            f.depth_likelihood_ratio < last,
            "Λ {} at {dist} not below {last}",
            f.depth_likelihood_ratio
        );
        last = f.depth_likelihood_ratio;
        let p = f.point.unwrap();
        assert!((p - target).norm() < 1e-6 * dist, "{p:?} vs {target:?}");
        assert!(f.in_front_of_all_cameras);
    }

    // Noisy: where Λ is large the point is within a few depth uncertainties of
    // the truth. With a 2-unit baseline and 1 mrad noise over 10 views, the
    // depth error at distance D is roughly D² × 1e-3 / (2 √10 × 0.6).
    let mut rng = StdRng::seed_from_u64(6);
    for &dist in &distances[..3] {
        let target = Point3::from(far_bearing() * dist);
        let dirs = rays_to(&target, &centers, 1e-3, &mut rng);
        let f = fit(&dirs, &centers, 1e-3, None, &PLAIN);
        assert!(f.depth_likelihood_ratio > 100.0);
        let depth_sd = dist * dist * 1e-3 / (2.0 * 10f64.sqrt() * 0.6);
        let err = (f.point.unwrap() - target).norm();
        assert!(err < 5.0 * depth_sd + 1e-3 * dist, "error {err} at {dist}");
    }
}

/// Over tracks from near to infinity, along a short baseline and around
/// object-centric arcs, the score and `Λ` agree closely wherever `Λ` is below
/// four times the threshold; a bearing cost below the threshold never comes
/// with `Λ` at or above it; the midpoint bound never exceeds `Λ`; and the
/// verdict agrees with `Λ` away from the threshold.
#[test]
fn score_agrees_with_likelihood_ratio_where_the_decision_is_close() {
    let mut rng = StdRng::seed_from_u64(7);
    // Trials where the decision is close, so the comparison is known to have
    // been exercised there.
    let mut near_threshold = 0;
    let mut wide_rescued = 0;
    for trial in 0..1500 {
        let k = rng.random_range(3..20);
        let (centers, target, geometry) = match trial % 3 {
            0 => {
                let d = 10f64.powf(rng.random_range(0.5..5.0));
                (
                    spread_centers(k),
                    Point3::from(far_bearing() * d),
                    "baseline",
                )
            }
            1 => {
                // Along the baseline, where the parallax is least.
                let d = 10f64.powf(rng.random_range(0.5..4.0));
                let dir = Vector3::new(1.0, 0.05, 0.1).normalize();
                (spread_centers(k), Point3::from(dir * d), "along baseline")
            }
            _ => {
                let arc = [30.0, 90.0, 180.0, 360.0][rng.random_range(0..4)];
                let centers = arc_centers(k, arc, &mut rng);
                // Mostly near the arc's centre, sometimes far out.
                let d = 10f64.powf(rng.random_range(-1.0..4.0));
                let dir = Vector3::new(
                    rng.random_range(-1.0..1.0),
                    rng.random_range(-0.3..0.3),
                    rng.random_range(-1.0..1.0),
                )
                .normalize();
                (centers, Point3::from(dir * d), "arc")
            }
        };
        let sigma: Vec<f64> = (0..k).map(|_| rng.random_range(5e-4..2e-3)).collect();
        let dirs: Vec<Vector3<f64>> = centers
            .iter()
            .zip(&sigma)
            .map(|(c, &s)| perturb(&(target - c).normalize(), s, &mut rng))
            .collect();
        let weights = isotropic_ray_weights(&dirs, &sigma);
        let s = bearing_score(&dirs, &centers, &weights).unwrap();
        let f = fit_point_and_bearing(&dirs, &centers, &weights, None, None, &PLAIN).unwrap();
        let lr = f.depth_likelihood_ratio;
        let ctx = format!("trial {trial} ({geometry}, k {k}, target {target:?})");
        assert!(
            (s.bearing_cost - f.bearing_cost).abs() < 1e-12 * f.bearing_cost.max(1.0),
            "{ctx}"
        );
        assert!(
            s.midpoint_bound <= lr * (1.0 + 1e-9) + 1e-9,
            "{ctx}: bound {} over Λ {lr}",
            s.midpoint_bound
        );
        if lr < 4.0 * T {
            if lr > T / 2.0 {
                near_threshold += 1;
            }
            let tol = 0.02 * lr.max(s.depth_score) + 0.05;
            assert!(
                (s.depth_score - lr).abs() < tol,
                "{ctx}: score {} vs Λ {lr}",
                s.depth_score
            );
        }
        if s.bearing_cost < T {
            assert!(lr < T, "{ctx}: bearing cost below threshold with Λ {lr}");
        }
        if lr > 1.5 * T && f.in_front_of_all_cameras {
            assert!(is_finite(&s, T), "{ctx}: Λ {lr} called a bearing");
            if s.depth_score < T {
                wide_rescued += 1;
            }
        }
        if lr < T / 1.5 {
            assert!(!is_finite(&s, T), "{ctx}: Λ {lr} called finite");
        }
    }
    assert!(
        near_threshold >= 50,
        "only {near_threshold} trials near the threshold"
    );
    assert!(
        wide_rescued > 0,
        "no trial needed the midpoint bound, so the wide geometries did not exercise it"
    );
}

/// Finite points inside object-centric arcs of cameras from 120° to a full
/// ring are all called finite. The score linearised at the bearing says
/// nothing there; the midpoint bound decides them.
#[test]
fn wide_arcs_around_a_point_are_finite() {
    let mut rng = StdRng::seed_from_u64(13);
    for arc in [120.0, 180.0, 270.0, 360.0] {
        let mut score_alone_fails = 0;
        for trial in 0..500 {
            let k = rng.random_range(3..12);
            let centers = arc_centers(k, arc, &mut rng);
            let target = Point3::new(
                rng.random_range(-1.0..1.0),
                rng.random_range(-1.0..1.0),
                rng.random_range(-1.0..1.0),
            );
            let dirs = rays_to(&target, &centers, 2e-4, &mut rng);
            let s = score(&dirs, &centers, 2e-4);
            assert!(
                is_finite(&s, T),
                "arc {arc}, trial {trial}: score {} bound {} bearing cost {}",
                s.depth_score,
                s.midpoint_bound,
                s.bearing_cost
            );
            if s.depth_score < T {
                score_alone_fails += 1;
            }
            let f = fit(&dirs, &centers, 2e-4, None, &PLAIN);
            assert!(f.in_front_of_all_cameras);
            assert!(f.depth_likelihood_ratio >= s.midpoint_bound * (1.0 - 1e-9));
            assert!((f.point.unwrap() - target).norm() < 1e-2);
        }
        if arc == 360.0 {
            assert!(
                score_alone_fails > 0,
                "the full ring never needed the midpoint bound"
            );
        }
    }
}

/// Rays that converge behind the cameras score 0 and are not called finite.
/// A fit warm-started at the point behind them, which fits the sine residuals
/// almost exactly, is not kept: the fit from the bearing in front is.
#[test]
fn rays_converging_behind_the_cameras_are_not_finite() {
    let mut rng = StdRng::seed_from_u64(8);
    let centers = spread_centers(6);
    let behind = Point3::new(0.0, 0.0, -20.0);
    // Each ray points away from the point behind the cameras.
    let dirs: Vec<Vector3<f64>> = centers
        .iter()
        .map(|c| perturb(&(c - behind).normalize(), 1e-4, &mut rng))
        .collect();
    let s = score(&dirs, &centers, 1e-3);
    assert_eq!(s.depth_score, 0.0);
    assert_eq!(s.midpoint_bound, 0.0);
    assert!(s.bearing_cost > T);
    assert!(!is_finite(&s, T));

    let cold = fit(&dirs, &centers, 1e-3, None, &PLAIN);
    assert!(
        cold.depth_likelihood_ratio < 1e-6,
        "{}",
        cold.depth_likelihood_ratio
    );
    assert!(cold.in_front_of_all_cameras);

    // The warm fit alone lands behind the cameras with a cost far below the
    // bearing's.
    let weights = iso(&dirs, 1e-3);
    let track = Track::new(&dirs, &centers, &weights, None).unwrap();
    let (u0, rho0) = track.to_inverse_depth(behind).unwrap();
    let warm_alone = track.refine(u0, rho0, false, None, 20);
    assert!(!track.in_front(&warm_alone));
    assert!(s.bearing_cost - warm_alone.cost > T);

    let warm = fit(&dirs, &centers, 1e-3, Some(behind), &PLAIN);
    assert!(warm.in_front_of_all_cameras);
    assert!(
        warm.depth_likelihood_ratio < 1e-6,
        "{}",
        warm.depth_likelihood_ratio
    );
}

/// A warm start whose fit ends above the bearing's cost is replaced by the
/// cold fit. With no iterations the warm fit stays at its start, which costs
/// more than the bearing, so the result is the cold one.
#[test]
fn warm_start_above_the_bearing_cost_falls_back_to_the_cold_fit() {
    let mut rng = StdRng::seed_from_u64(14);
    let centers = spread_centers(7);
    let dirs = rays_along(&far_bearing(), &centers, 1e-3, &mut rng);
    let no_iterations = PointBearingFitOptions {
        soft_l1_scale: None,
        max_iterations: 0,
    };
    let bad = Point3::new(30.0, -10.0, 5.0);
    let weights = iso(&dirs, 1e-3);
    let track = Track::new(&dirs, &centers, &weights, None).unwrap();
    let (u0, rho0) = track.to_inverse_depth(bad).unwrap();
    let (_, bearing_cost) = track.closed_form_bearing();
    assert!(track.cost(&u0, rho0, None) > bearing_cost);

    let warm = fit(&dirs, &centers, 1e-3, Some(bad), &no_iterations);
    let cold = fit(&dirs, &centers, 1e-3, None, &no_iterations);
    assert_eq!(warm, cold);
    assert!(warm.point_cost <= warm.bearing_cost);
}

/// One sighting displaced by 20σ on a true bearing, at the end camera and in
/// the direction a finite depth would move it, lifts the plain least-squares
/// `Λ` over the threshold; the soft-L1 fit keeps it under.
#[test]
fn one_bad_sighting_does_not_make_a_bearing_finite_under_soft_l1() {
    let mut rng = StdRng::seed_from_u64(9);
    let centers = spread_centers(10);
    let sigma = 1e-3;
    let u = far_bearing();
    let mut centroid = Vector3::zeros();
    for c in &centers {
        centroid += c.coords;
    }
    let centroid = Point3::from(centroid / centers.len() as f64);
    // A finite depth turns ray i towards the anchor: along (I − uuᵀ)(a − cᵢ).
    let last = centers.len() - 1;
    let towards = centroid - centers[last];
    let shift = (towards - towards.dot(&u) * u).normalize();
    for trial in 0..10 {
        let mut dirs = rays_along(&u, &centers, sigma, &mut rng);
        dirs[last] = (dirs[last] + 20.0 * sigma * shift).normalize();
        let plain = fit(&dirs, &centers, sigma, None, &PLAIN);
        let robust = fit(&dirs, &centers, sigma, None, &ROBUST);
        assert!(
            plain.depth_likelihood_ratio > T,
            "trial {trial}: plain Λ {}",
            plain.depth_likelihood_ratio
        );
        assert!(
            robust.depth_likelihood_ratio < T,
            "trial {trial}: robust Λ {}",
            robust.depth_likelihood_ratio
        );
        assert!(robust.point_cost <= robust.bearing_cost);
    }
}

/// A warm start reaches the same point as a cold fit, and a caller's anchor
/// changes the parametrisation but not the point or `Λ`.
#[test]
fn warm_start_and_anchor_reach_the_cold_fit() {
    let mut rng = StdRng::seed_from_u64(10);
    let centers = spread_centers(7);
    let target = Point3::new(2.0, -1.0, 30.0);
    let dirs = rays_to(&target, &centers, 1e-3, &mut rng);
    let weights = iso(&dirs, 1e-3);
    let cold = fit(&dirs, &centers, 1e-3, None, &PLAIN);
    let start = Point3::new(2.5, -1.2, 36.0);
    let warm = fit(&dirs, &centers, 1e-3, Some(start), &PLAIN);
    assert!((warm.point_cost - cold.point_cost).abs() < 1e-8 * cold.point_cost.max(1.0));
    assert!((warm.point.unwrap() - cold.point.unwrap()).norm() < 1e-5);

    let anchor = Point3::new(-3.0, 4.0, 1.0);
    let anchored =
        fit_point_and_bearing(&dirs, &centers, &weights, None, Some(anchor), &PLAIN).unwrap();
    assert_eq!(anchored.anchor, anchor);
    assert!(
        (anchored.depth_likelihood_ratio - cold.depth_likelihood_ratio).abs()
            < 1e-6 * cold.depth_likelihood_ratio
    );
    assert!((anchored.point.unwrap() - cold.point.unwrap()).norm() < 1e-5);
}

/// Isotropic weights built from any basis perpendicular to the ray give the
/// same costs as the scalar form `Σ ‖dᵢ × u‖² / σᵢ²`, and the same results.
#[test]
fn isotropic_weights_are_the_scalar_noise_form() {
    let mut rng = StdRng::seed_from_u64(15);
    let centers = spread_centers(8);
    let sigma: Vec<f64> = (0..8).map(|_| rng.random_range(5e-4..2e-3)).collect();
    let target = Point3::from(far_bearing() * 400.0);
    let dirs: Vec<Vector3<f64>> = centers
        .iter()
        .zip(&sigma)
        .map(|(c, &s)| perturb(&(target - c).normalize(), s, &mut rng))
        .collect();
    let weights = isotropic_ray_weights(&dirs, &sigma);
    // The same noise in a basis turned by a different angle per ray.
    let turned: Vec<Matrix2x3<f64>> = weights
        .iter()
        .map(|w| Rotation2::new(rng.random_range(0.0..std::f64::consts::TAU)).into_inner() * w)
        .collect();

    let a = bearing_score(&dirs, &centers, &weights).unwrap();
    let b = bearing_score(&dirs, &centers, &turned).unwrap();
    let scalar: f64 = dirs
        .iter()
        .zip(&sigma)
        .map(|(d, s)| d.cross(&a.bearing).norm_squared() / (s * s))
        .sum();
    assert!((a.bearing_cost - scalar).abs() < 1e-9 * scalar);
    assert!((a.bearing_cost - b.bearing_cost).abs() < 1e-9 * scalar);
    assert!((a.depth_score - b.depth_score).abs() < 1e-6 * a.depth_score.max(1.0));
    assert!((a.bearing - b.bearing).norm() < 1e-9);

    let fa = fit_point_and_bearing(&dirs, &centers, &weights, None, None, &PLAIN).unwrap();
    let fb = fit_point_and_bearing(&dirs, &centers, &turned, None, None, &PLAIN).unwrap();
    assert!(
        (fa.depth_likelihood_ratio - fb.depth_likelihood_ratio).abs()
            < 1e-6 * fa.depth_likelihood_ratio.max(1.0)
    );
}

/// The anisotropic weight for one ray: noise `major` along a tangent axis at
/// angle `phi` and `minor` across it.
fn anisotropic_weight(d: &Vector3<f64>, phi: f64, major: f64, minor: f64) -> Matrix2x3<f64> {
    let whiten = Matrix2::new(1.0 / major, 0.0, 0.0, 1.0 / minor);
    whiten * Rotation2::new(phi).into_inner() * isotropic_ray_weight(d, 1.0)
}

/// Noise three times larger along one tangent axis than across it, at a
/// different axis per ray. With the matching weights the statistic follows
/// the half-χ²₁ law of a true bearing, and a finite point is recovered.
#[test]
fn anisotropic_noise_is_calibrated_and_recovered() {
    let mut rng = StdRng::seed_from_u64(16);
    let centers = spread_centers(8);
    let (major, minor) = (3e-3, 1e-3);
    let u = far_bearing();
    let phis: Vec<f64> = (0..8)
        .map(|_| rng.random_range(0.0..std::f64::consts::PI))
        .collect();

    // Draw the noise in the axes the weight of the true direction uses.
    let noisy = |truth: &[Vector3<f64>], rng: &mut StdRng| -> Vec<Vector3<f64>> {
        truth
            .iter()
            .zip(&phis)
            .map(|(d, &phi)| {
                let (b1, b2) = tangent_basis(d);
                let rot = Rotation2::new(phi).into_inner();
                // Rows of rot·Bᵀ are the noise axes.
                let e1 = rot[(0, 0)] * b1 + rot[(0, 1)] * b2;
                let e2 = rot[(1, 0)] * b1 + rot[(1, 1)] * b2;
                let n1: f64 = rng.sample(StandardNormal);
                let n2: f64 = rng.sample(StandardNormal);
                (d + major * n1 * e1 + minor * n2 * e2).normalize()
            })
            .collect()
    };
    let weights_for = |dirs: &[Vector3<f64>]| -> Vec<Matrix2x3<f64>> {
        dirs.iter()
            .zip(&phis)
            .map(|(d, &phi)| anisotropic_weight(d, phi, major, minor))
            .collect()
    };

    let truth = vec![u; 8];
    let mut scores = Vec::new();
    let mut ratios = Vec::new();
    for i in 0..30000 {
        let dirs = noisy(&truth, &mut rng);
        let w = weights_for(&dirs);
        scores.push(bearing_score(&dirs, &centers, &w).unwrap().depth_score);
        if i < 10000 {
            let f = fit_point_and_bearing(&dirs, &centers, &w, None, None, &PLAIN).unwrap();
            ratios.push(f.depth_likelihood_ratio);
        }
    }
    check_half_chi_square("anisotropic score", &scores);
    check_half_chi_square("anisotropic likelihood ratio", &ratios);

    let target = Point3::from(u * 10.0);
    let truth: Vec<Vector3<f64>> = centers.iter().map(|c| (target - c).normalize()).collect();
    let dirs = noisy(&truth, &mut rng);
    let w = weights_for(&dirs);
    let s = bearing_score(&dirs, &centers, &w).unwrap();
    assert!(is_finite(&s, T));
    let f = fit_point_and_bearing(&dirs, &centers, &w, None, None, &PLAIN).unwrap();
    // The depth error at 10 units is about 0.1 here.
    assert!((f.point.unwrap() - target).norm() < 0.5, "{:?}", f.point);
}

/// [`observed_ray`] un-projects the pixel to the world ray towards the point,
/// and its weight maps a small move of the point to the pixel shift over σ,
/// for a pinhole, an analytic fisheye and a fisheye differentiated
/// numerically.
#[test]
fn observed_ray_weight_is_the_pixel_derivative() {
    let cameras = [
        CameraIntrinsics {
            model: CameraModel::SimplePinhole {
                focal_length: 500.0,
                principal_point_x: 320.0,
                principal_point_y: 240.0,
            },
            width: 640,
            height: 480,
        },
        CameraIntrinsics {
            model: CameraModel::SimpleRadialFisheye {
                focal_length: 150.0,
                principal_point_x: 240.0,
                principal_point_y: 240.0,
                radial_distortion_k1: 0.02,
            },
            width: 480,
            height: 480,
        },
        CameraIntrinsics {
            model: CameraModel::OpenCVFisheye {
                focal_length_x: 150.0,
                focal_length_y: 152.0,
                principal_point_x: 240.0,
                principal_point_y: 240.0,
                radial_distortion_k1: 0.01,
                radial_distortion_k2: -0.002,
                radial_distortion_k3: 0.0,
                radial_distortion_k4: 0.0,
            },
            width: 480,
            height: 480,
        },
    ];
    let rot = UnitQuaternion::from_euler_angles(0.3, -0.5, 1.1);
    let center = Point3::new(1.0, -2.0, 0.5);
    let t = -(rot * center.coords);
    let sigma_px = 0.25;
    for cam in &cameras {
        // A camera-frame ray well off the axis (the camera looks along −z).
        let off_axis = if matches!(cam.model, CameraModel::SimplePinhole { .. }) {
            Vector3::new(0.3, -0.2, -1.0)
        } else {
            Vector3::new(1.5, 1.0, -0.6)
        };
        let point = center + rot.inverse() * off_axis.normalize() * 7.0;
        let project = |p: &Point3<f64>| {
            let pc = rot * p.coords + t;
            let (u, v) = cam.ray_to_pixel([pc.x, pc.y, pc.z]).unwrap();
            Vector2::new(u, v)
        };
        let pixel = project(&point);
        let obs = observed_ray(cam, &rot, [pixel.x, pixel.y], sigma_px).unwrap();
        let toward = (point - center).normalize();
        assert!((obs.dir - toward).norm() < 1e-8, "{:?}", cam.model);

        let a = obs.weight * (Matrix3::identity() - obs.dir * obs.dir.transpose());
        for delta in [
            Vector3::new(1e-3, 0.0, 0.0),
            Vector3::new(0.0, 1e-3, 0.0),
            Vector3::new(0.0, 0.0, 1e-3),
        ] {
            let moved = point + delta;
            let predicted = a * (moved - center).normalize();
            let actual = (project(&moved) - pixel) / sigma_px;
            assert!(
                (predicted - actual).norm() < 1e-3 * actual.norm().max(1e-3),
                "{:?}: predicted {predicted:?}, actual {actual:?}",
                cam.model
            );
        }
    }
}

/// The batch forms agree with the single-track forms bit for bit, and tracks
/// with fewer than two usable rays are `None`.
#[test]
fn batch_matches_single_track_and_skips_short_tracks() {
    let mut rng = StdRng::seed_from_u64(11);
    let mut dirs = Vec::new();
    let mut centers = Vec::new();
    let mut weights = Vec::new();
    let mut offsets = vec![0];
    let mut starts = Vec::new();
    let mut anchors = Vec::new();
    for t in 0..40 {
        let k = match t % 10 {
            0 => 0,
            1 => 1,
            2 | 3 => 2,
            _ => 2 + t % 7,
        };
        let c = if k >= 2 {
            spread_centers(k)
        } else {
            vec![Point3::origin(); k]
        };
        let distance = 10f64.powf(rng.random_range(0.5..4.5));
        let target = Point3::from(far_bearing() * distance);
        let mut d = rays_to(&target, &c, 1e-3, &mut rng);
        let s: Vec<f64> = (0..k).map(|_| rng.random_range(5e-4..2e-3)).collect();
        if t % 10 == 2 {
            // Two rays, one unusable: fewer than two usable.
            d[1] = Vector3::new(f64::NAN, 0.0, 1.0);
        }
        let mut w = isotropic_ray_weights(&d, &s);
        if t % 10 == 3 {
            w[0] = Matrix2x3::zeros();
        }
        dirs.extend(d);
        centers.extend(c);
        weights.extend(w);
        offsets.push(dirs.len());
        starts.push(if t % 3 == 0 {
            Point3::new(f64::NAN, 0.0, 0.0)
        } else {
            target + Vector3::new(0.1, 0.0, 0.0)
        });
        anchors.push(if t % 4 == 0 {
            Point3::new(f64::NAN, 0.0, 0.0)
        } else {
            Point3::new(0.5, 0.2, -0.3)
        });
    }

    let scores = bearing_score_batch(&dirs, &centers, &offsets, &weights);
    let fits =
        fit_point_and_bearing_batch(&dirs, &centers, &offsets, &weights, None, None, &ROBUST);
    let warm = fit_point_and_bearing_batch(
        &dirs,
        &centers,
        &offsets,
        &weights,
        Some(&starts),
        Some(&anchors),
        &PLAIN,
    );
    assert_eq!(scores.len(), 40);
    for t in 0..40 {
        let r = offsets[t]..offsets[t + 1];
        let (d, c, w) = (&dirs[r.clone()], &centers[r.clone()], &weights[r]);
        assert_eq!(scores[t], bearing_score(d, c, w), "track {t}");
        assert_eq!(
            fits[t],
            fit_point_and_bearing(d, c, w, None, None, &ROBUST),
            "track {t}"
        );
        let start = Some(starts[t]).filter(|p| p.x.is_finite());
        let anchor = Some(anchors[t]).filter(|p| p.x.is_finite());
        assert_eq!(
            warm[t],
            fit_point_and_bearing(d, c, w, start, anchor, &PLAIN),
            "track {t}"
        );

        let short = matches!(t % 10, 0..=3);
        assert_eq!(scores[t].is_none(), short, "track {t}");
        assert_eq!(fits[t].is_none(), short, "track {t}");
    }
}

/// Cameras at one centre give the depth no leverage: the score is 0 and `Λ` is
/// 0, so the track is a bearing.
#[test]
fn coincident_cameras_give_no_depth() {
    let mut rng = StdRng::seed_from_u64(12);
    let centers = vec![Point3::new(1.0, 2.0, 3.0); 5];
    let target = Point3::new(1.5, 2.0, 8.0);
    let dirs = rays_to(&target, &centers, 1e-3, &mut rng);
    let s = score(&dirs, &centers, 1e-3);
    assert_eq!(s.depth_score, 0.0);
    assert!(s.midpoint_bound < 1e-9);
    assert!(!is_finite(&s, T));
    let f = fit(&dirs, &centers, 1e-3, None, &PLAIN);
    assert!(
        f.depth_likelihood_ratio < 1e-9,
        "{}",
        f.depth_likelihood_ratio
    );
}

/// A warm-started fit never reports a `Λ` below the midpoint bound, even
/// with few iterations, as bundle adjustment would run it: starts up to 4
/// units off a point inside object-centric arcs.
#[test]
fn warm_started_fit_never_falls_below_the_midpoint_bound() {
    let mut rng = StdRng::seed_from_u64(17);
    let few = PointBearingFitOptions {
        soft_l1_scale: None,
        max_iterations: 3,
    };
    for arc in [120.0, 180.0, 270.0, 360.0] {
        for trial in 0..300 {
            let k = rng.random_range(3..12);
            let centers = arc_centers(k, arc, &mut rng);
            let target = Point3::new(
                rng.random_range(-1.0..1.0),
                rng.random_range(-1.0..1.0),
                rng.random_range(-1.0..1.0),
            );
            let dirs = rays_to(&target, &centers, 2e-3, &mut rng);
            let weights = iso(&dirs, 2e-3);
            let s = bearing_score(&dirs, &centers, &weights).unwrap();
            for spread in [0.5, 2.0, 4.0] {
                let start = target
                    + Vector3::new(
                        rng.random_range(-spread..spread),
                        rng.random_range(-spread..spread),
                        rng.random_range(-spread..spread),
                    );
                for options in [PLAIN, few] {
                    let f = fit_point_and_bearing(
                        &dirs,
                        &centers,
                        &weights,
                        Some(start),
                        None,
                        &options,
                    )
                    .unwrap();
                    assert!(
                        f.depth_likelihood_ratio >= s.midpoint_bound * (1.0 - 1e-9) - 1e-9,
                        "arc {arc}, trial {trial}, {} iterations: Λ {} below bound {}",
                        options.max_iterations,
                        f.depth_likelihood_ratio,
                        s.midpoint_bound
                    );
                }
            }
        }
    }
}

/// Rays along `u`, `−u` and `u` fit the bearing `u` at no cost, since the sine
/// cannot tell a ray from its opposite; the score says the bearing is behind
/// one camera.
#[test]
fn bearing_behind_a_camera_is_flagged() {
    let centers = spread_centers(3);
    let u = far_bearing();
    let dirs = vec![u, -u, u];
    let s = score(&dirs, &centers, 1e-3);
    assert!(s.bearing_cost < 1e-12);
    assert!(!s.bearing_in_front_of_all_cameras);

    let s = score(&[u, u, u], &centers, 1e-3);
    assert!(s.bearing_in_front_of_all_cameras);
}

/// Weights so large that the bearing's cost overflows make the rays unusable
/// rather than giving an infinite cost.
#[test]
fn overflowing_weights_are_unusable() {
    let mut rng = StdRng::seed_from_u64(18);
    let centers = spread_centers(4);
    let dirs = rays_along(&far_bearing(), &centers, 1e-3, &mut rng);
    let weights = iso(&dirs, 1e-300);
    assert!(bearing_score(&dirs, &centers, &weights).is_none());
    assert!(fit_point_and_bearing(&dirs, &centers, &weights, None, None, &PLAIN).is_none());
}

/// [`observed_ray`] declines a noise level that is not finite and positive,
/// and a pixel past the fold of a fisheye's distortion, whose un-projected ray
/// projects somewhere else. The camera's numeric pixel Jacobian, which the
/// weight uses for models without an analytic one, declines a ray whose
/// difference probes cross the edge of the model's domain.
#[test]
fn observed_ray_declines_what_it_cannot_weight() {
    // θ_d = θ (1 − 0.25 θ²): rising to 0.770 at θ = 1.155, zero at θ = 2.
    let cam = CameraIntrinsics {
        model: CameraModel::OpenCVFisheye {
            focal_length_x: 200.0,
            focal_length_y: 200.0,
            principal_point_x: 300.0,
            principal_point_y: 300.0,
            radial_distortion_k1: -0.25,
            radial_distortion_k2: 0.0,
            radial_distortion_k3: 0.0,
            radial_distortion_k4: 0.0,
        },
        width: 600,
        height: 600,
    };
    assert!(!cam.model.supports_pixel_jacobian());
    let rot = UnitQuaternion::identity();
    let inside = [300.0 + 200.0 * 0.5, 300.0];
    assert!(observed_ray(&cam, &rot, inside, 0.5).is_some());
    for sigma in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(
            observed_ray(&cam, &rot, inside, sigma).is_none(),
            "σ {sigma}"
        );
    }

    // A pixel at θ_d = 0.85, beyond the largest θ_d the model reaches.
    let past_fold = [300.0 + 200.0 * 0.85, 300.0];
    assert!(observed_ray(&cam, &rot, past_fold, 0.5).is_none());

    // A camera-frame ray 1e-8 rad inside θ = 2, where θ_d reaches 0: the
    // forward projection is defined there, but a difference probe is not.
    // The camera looks along −z.
    let theta: f64 = 2.0 - 1e-8;
    let ray = [theta.sin(), 0.0, -theta.cos()];
    assert!(cam.ray_to_pixel(ray).is_some());
    assert!(cam.pixel_jacobian(ray).is_none());
}

/// One camera very close to the point and three far from it: the sine
/// residual lets a fit from the midpoint slide to a cheap point behind the
/// near camera. The fit stays in front instead, so `Λ` never falls below the
/// midpoint bound at any iteration count or start, and when the bound decides
/// the track finite the fit places the point.
#[test]
fn a_camera_close_to_the_point_keeps_the_bound() {
    let mut rng = StdRng::seed_from_u64(19);
    let target = Point3::origin();
    let mut centers = vec![Point3::new(0.0, 0.0, -0.02)];
    for a in [0.0f64, 120.0, 240.0] {
        let a = a.to_radians();
        centers.push(Point3::new(2.5 * a.cos(), 2.5 * a.sin(), -4.33));
    }
    let mut decided_by_bound = 0;
    for sigma in [1e-2, 1e-3] {
        for trial in 0..500 {
            let dirs = rays_to(&target, &centers, sigma, &mut rng);
            let weights = iso(&dirs, sigma);
            let s = bearing_score(&dirs, &centers, &weights).unwrap();
            for start in [None, Some(target)] {
                for max_iterations in [0, 2, 3, 20, 50] {
                    let options = PointBearingFitOptions {
                        soft_l1_scale: None,
                        max_iterations,
                    };
                    let f = fit_point_and_bearing(&dirs, &centers, &weights, start, None, &options)
                        .unwrap();
                    assert!(
                        f.depth_likelihood_ratio >= s.midpoint_bound * (1.0 - 1e-9) - 1e-9,
                        "σ {sigma}, trial {trial}, start {start:?}, {max_iterations} iterations: \
                         Λ {} below bound {}",
                        f.depth_likelihood_ratio,
                        s.midpoint_bound
                    );
                    if s.midpoint_bound >= T {
                        assert!(f.point.is_some() && f.in_front_of_all_cameras);
                    }
                }
            }
            if s.midpoint_bound >= T {
                decided_by_bound += 1;
            }
        }
    }
    assert!(decided_by_bound > 0);
}

/// One camera within 0.3 of the point and the others 1 to 300 away over a
/// wide spread of directions. Often neither the bearing nor the weighted
/// midpoint is in front of every camera, so no ordinary cold start leads in
/// front; the start found along the rays does, and every track called finite
/// gets a point in front of every camera with `Λ` over the threshold.
#[test]
fn a_near_camera_with_a_wide_spread_gets_a_point_in_front() {
    let mut rng = StdRng::seed_from_u64(20);
    let target = Point3::origin();
    let mut no_ordinary_start = 0;
    for trial in 0..4000 {
        let k = rng.random_range(2..10);
        let centers: Vec<Point3<f64>> = (0..k)
            .map(|i| {
                let jitter = Vector3::new(
                    rng.random_range(-1.0..1.0),
                    rng.random_range(-1.0..1.0),
                    rng.random_range(-1.0..1.0),
                );
                let dir = (Vector3::<f64>::new(0.0, 0.0, -1.0)
                    + rng.random_range(0.0f64..2.0) * jitter)
                    .normalize();
                let dist = if i == 0 {
                    rng.random_range(0.01..0.3)
                } else {
                    rng.random_range(1.0..300.0)
                };
                target + dir * dist
            })
            .collect();
        let sigma = [1e-4, 1e-3, 1e-2][trial % 3];
        let dirs = rays_to(&target, &centers, sigma, &mut rng);
        let weights = iso(&dirs, sigma);
        let s = bearing_score(&dirs, &centers, &weights).unwrap();
        if !is_finite(&s, T) {
            continue;
        }
        let f = fit_point_and_bearing(&dirs, &centers, &weights, None, None, &PLAIN).unwrap();
        let ctx = format!("trial {trial}, k {k}, σ {sigma}");
        assert!(
            f.point.is_some() && f.in_front_of_all_cameras,
            "{ctx}: {f:?}"
        );
        let track = Track::new(&dirs, &centers, &weights, None).unwrap();
        if !s.bearing_in_front_of_all_cameras && track.midpoint_in_front().is_none() {
            no_ordinary_start += 1;
            assert!(
                f.depth_likelihood_ratio >= T,
                "{ctx}: Λ {}",
                f.depth_likelihood_ratio
            );
        }
    }
    assert!(no_ordinary_start > 0, "no trial lacked an ordinary start");
}
