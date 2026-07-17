// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::camera::epipolar::compute_fundamental_matrix;
use nalgebra::{Rotation3, Unit};

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

    fn gaussian(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-300);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    fn rotation_matrix(&mut self, max_angle: f64) -> Matrix3<f64> {
        let axis = Unit::new_normalize(Vector3::new(
            self.uniform(-1.0, 1.0),
            self.uniform(-1.0, 1.0),
            self.uniform(-1.0, 1.0),
        ));
        let angle = self.uniform(0.05, max_angle);
        *Rotation3::from_axis_angle(&axis, angle).matrix()
    }
}

/// Pixel intrinsic matrix with square pixels and zero skew.
fn k_of(f: f64, cx: f64, cy: f64) -> Matrix3<f64> {
    Matrix3::new(f, 0.0, cx, 0.0, f, cy, 0.0, 0.0, 1.0)
}

/// A synthetic two-view correspondence set (optical +Z-forward convention).
struct Pair {
    f_true: Matrix3<f64>,
    f1: f64,
    pp1: [f64; 2],
    pp2: [f64; 2],
    x1: Vec<[f64; 2]>,
    x2: Vec<[f64; 2]>,
    inlier: Vec<bool>,
}

/// Build a non-degenerate camera pair and `n` correspondences, `outlier_frac`
/// of which are random pixel pairs; inliers carry `noise_px` Gaussian jitter.
fn make_pair(
    seed: u64,
    n: usize,
    focal1: f64,
    focal2: f64,
    noise_px: f64,
    outlier_frac: f64,
) -> Pair {
    let mut rng = Lcg(seed.wrapping_mul(0x9e3779b97f4a7c15).wrapping_add(1));
    let pp1 = [320.0, 240.0];
    let pp2 = [300.0, 260.0];
    let k1 = k_of(focal1, pp1[0], pp1[1]);
    let k2 = k_of(focal2, pp2[0], pp2[1]);

    let r1 = rng.rotation_matrix(0.6);
    let t1 = Vector3::new(
        rng.uniform(-0.3, 0.3),
        rng.uniform(-0.3, 0.3),
        rng.uniform(-0.3, 0.3),
    );
    let r2 = rng.rotation_matrix(0.6);
    // A real baseline so the epipolar geometry is well defined.
    let t2 = Vector3::new(
        rng.uniform(0.5, 1.5),
        rng.uniform(-0.5, 0.5),
        rng.uniform(-0.3, 0.3),
    );

    let f_true =
        compute_fundamental_matrix(&k1, &r1, &t1, &k2, &r2, &t2).expect("non-singular intrinsics");

    let n_out = (n as f64 * outlier_frac).round() as usize;
    let n_in = n - n_out;
    let mut x1 = Vec::with_capacity(n);
    let mut x2 = Vec::with_capacity(n);
    let mut inlier = Vec::with_capacity(n);

    let r1t = r1.transpose();
    while x1.len() < n_in {
        // Point in front of camera 1 (optical z > 0), then check camera 2.
        let cam1 = Vector3::new(
            rng.uniform(-2.0, 2.0),
            rng.uniform(-2.0, 2.0),
            rng.uniform(2.0, 8.0),
        );
        let xw = r1t * (cam1 - t1);
        let cam2 = r2 * xw + t2;
        if cam2.z <= 0.2 {
            continue;
        }
        let p1 = k1 * cam1;
        let p2 = k2 * cam2;
        let mut u1 = [p1.x / p1.z, p1.y / p1.z];
        let mut u2 = [p2.x / p2.z, p2.y / p2.z];
        if noise_px > 0.0 {
            u1[0] += noise_px * rng.gaussian();
            u1[1] += noise_px * rng.gaussian();
            u2[0] += noise_px * rng.gaussian();
            u2[1] += noise_px * rng.gaussian();
        }
        x1.push(u1);
        x2.push(u2);
        inlier.push(true);
    }
    for _ in 0..n_out {
        x1.push([rng.uniform(0.0, 640.0), rng.uniform(0.0, 480.0)]);
        x2.push([rng.uniform(0.0, 640.0), rng.uniform(0.0, 480.0)]);
        inlier.push(false);
    }

    Pair {
        f_true,
        f1: focal1,
        pp1,
        pp2,
        x1,
        x2,
        inlier,
    }
}

/// Frobenius distance between unit-normalized matrices, minimized over the sign
/// ambiguity — an "up to scale" comparison.
fn scale_diff(a: &Matrix3<f64>, b: &Matrix3<f64>) -> f64 {
    let an = a / a.norm();
    let bn = b / b.norm();
    (an - bn).norm().min((an + bn).norm())
}

/// Largest `|x̃₂ᵀ F x̃₁|` over the correspondences (algebraic residual).
fn max_algebraic_resid(f: &Matrix3<f64>, x1: &[[f64; 2]], x2: &[[f64; 2]]) -> f64 {
    (0..x1.len())
        .map(|i| {
            let p1 = Vector3::new(x1[i][0], x1[i][1], 1.0);
            let p2 = Vector3::new(x2[i][0], x2[i][1], 1.0);
            (p2.transpose() * f * p1)[(0, 0)].abs()
        })
        .fold(0.0, f64::max)
}

/// Smallest / largest singular value ratio.
fn rank_ratio(f: &Matrix3<f64>) -> f64 {
    let s = f.svd(false, false).singular_values;
    s[2] / s[0]
}

// ── 7-point solver ───────────────────────────────────────────────────────────

#[test]
fn exact_recovery_7pt() {
    for seed in 0..200u64 {
        let pair = make_pair(seed, 7, 600.0, 700.0, 0.0, 0.0);
        let s1: [[f64; 2]; 7] = core::array::from_fn(|k| pair.x1[k]);
        let s2: [[f64; 2]; 7] = core::array::from_fn(|k| pair.x2[k]);
        let cands = fundamental_7pt(&s1, &s2);
        assert!(!cands.is_empty(), "seed {seed}: no candidates");
        // One candidate satisfies the 7 constraints and matches the true F.
        let best_alg = cands
            .iter()
            .map(|f| max_algebraic_resid(f, &pair.x1, &pair.x2))
            .fold(f64::INFINITY, f64::min);
        assert!(
            best_alg < 1e-7,
            "seed {seed}: algebraic residual {best_alg}"
        );
        let best_scale = cands
            .iter()
            .map(|f| scale_diff(f, &pair.f_true))
            .fold(f64::INFINITY, f64::min);
        assert!(
            best_scale < 1e-6,
            "seed {seed}: no candidate matched true F (diff {best_scale})"
        );
    }
}

#[test]
fn cubic_multiplicity_returns_all_roots() {
    // Some configurations have three real roots; all are returned and one
    // matches the generating geometry. Every candidate is a valid rank-2 F.
    let mut triple = 0;
    for seed in 0..200u64 {
        let pair = make_pair(seed, 7, 550.0, 550.0, 0.0, 0.0);
        let s1: [[f64; 2]; 7] = core::array::from_fn(|k| pair.x1[k]);
        let s2: [[f64; 2]; 7] = core::array::from_fn(|k| pair.x2[k]);
        let cands = fundamental_7pt(&s1, &s2);
        assert!(!cands.is_empty());
        if cands.len() == 3 {
            triple += 1;
        }
        let best_scale = cands
            .iter()
            .map(|f| scale_diff(f, &pair.f_true))
            .fold(f64::INFINITY, f64::min);
        assert!(
            best_scale < 1e-6,
            "seed {seed}: true F not among candidates"
        );
    }
    assert!(triple > 0, "expected some three-root configurations");
}

#[test]
fn seven_point_candidates_are_rank2() {
    for seed in 0..100u64 {
        let pair = make_pair(seed, 7, 620.0, 480.0, 0.0, 0.0);
        let s1: [[f64; 2]; 7] = core::array::from_fn(|k| pair.x1[k]);
        let s2: [[f64; 2]; 7] = core::array::from_fn(|k| pair.x2[k]);
        for f in fundamental_7pt(&s1, &s2) {
            assert!(
                rank_ratio(&f) < 1e-6,
                "seed {seed}: rank ratio {}",
                rank_ratio(&f)
            );
        }
    }
}

#[test]
fn seven_point_degenerate_returns_empty() {
    let pair = make_pair(1, 7, 600.0, 600.0, 0.0, 0.0);
    // Repeated correspondence: two identical rows drop the rank.
    let mut s1: [[f64; 2]; 7] = core::array::from_fn(|k| pair.x1[k]);
    let mut s2: [[f64; 2]; 7] = core::array::from_fn(|k| pair.x2[k]);
    s1[1] = s1[0];
    s2[1] = s2[0];
    assert!(
        fundamental_7pt(&s1, &s2).is_empty(),
        "repeated correspondence"
    );

    // Non-finite value.
    let mut n1 = s1;
    n1[3][0] = f64::NAN;
    assert!(fundamental_7pt(&n1, &s2).is_empty(), "non-finite input");
}

// ── 8-point solver ───────────────────────────────────────────────────────────

#[test]
fn eight_point_recovers_and_is_rank2() {
    for seed in 0..100u64 {
        let pair = make_pair(seed, 40, 700.0, 640.0, 0.0, 0.0);
        let f = fundamental_8pt(&pair.x1, &pair.x2).expect("valid design");
        assert!(rank_ratio(&f) < 1e-9, "rank ratio {}", rank_ratio(&f));
        assert!(
            scale_diff(&f, &pair.f_true) < 1e-6,
            "seed {seed}: 8-point diff {}",
            scale_diff(&f, &pair.f_true)
        );
    }
}

#[test]
fn eight_point_degenerate_returns_none() {
    let pair = make_pair(2, 40, 600.0, 600.0, 0.0, 0.0);
    // Too few correspondences.
    assert!(fundamental_8pt(&pair.x1[..7], &pair.x2[..7]).is_none());
    // Non-finite.
    let mut bad = pair.x1.clone();
    bad[5][1] = f64::INFINITY;
    assert!(fundamental_8pt(&bad, &pair.x2).is_none());
    // Coincident points (zero spread).
    let same = vec![[100.0, 100.0]; 12];
    assert!(fundamental_8pt(&same, &same).is_none());
}

// ── Robust estimator ─────────────────────────────────────────────────────────

fn base_opts() -> FundamentalOptions {
    FundamentalOptions {
        max_error_px: 1.5,
        confidence: 0.999,
        max_iterations: 40_000,
        min_inliers: 12,
        seed: 7,
        local_optimization: true,
    }
}

#[test]
fn contamination_sweep_recovers_geometry() {
    // Floor at 0.35 keeps the (unoptimized) test-profile `w⁷` RANSAC tractable;
    // the 0.2 floor from the spec is exercised in the release-built Python
    // binding tests. See the spec's deviation note.
    let n = 120;
    for &frac in &[0.9, 0.7, 0.5, 0.35] {
        let outlier = 1.0 - frac;
        let pair = make_pair(4242, n, 650.0, 650.0, 0.0, outlier);
        let est = estimate_fundamental(&pair.x1, &pair.x2, &base_opts())
            .unwrap_or_else(|| panic!("frac {frac}: expected an estimate"));
        assert!(
            scale_diff(&est.f_matrix, &pair.f_true) < 5e-3,
            "frac {frac}: F diff {}",
            scale_diff(&est.f_matrix, &pair.f_true)
        );
        // The true inliers are recovered (allow a couple of misses).
        let true_in = pair.inlier.iter().filter(|&&b| b).count();
        let found = est.inliers.iter().filter(|&&b| b).count();
        assert!(
            found + 3 >= true_in,
            "frac {frac}: found {found} of {true_in} inliers"
        );
    }
}

#[test]
fn below_min_inliers_returns_none() {
    // Only ~15 true inliers but require 40.
    let pair = make_pair(9, 120, 600.0, 600.0, 0.0, 0.9);
    let opts = FundamentalOptions {
        min_inliers: 40,
        max_iterations: 3_000,
        ..base_opts()
    };
    assert!(estimate_fundamental(&pair.x1, &pair.x2, &opts).is_none());
}

#[test]
fn none_on_garbage() {
    let mut rng = Lcg(123);
    let n = 100;
    let x1: Vec<[f64; 2]> = (0..n)
        .map(|_| [rng.uniform(0.0, 640.0), rng.uniform(0.0, 480.0)])
        .collect();
    let x2: Vec<[f64; 2]> = (0..n)
        .map(|_| [rng.uniform(0.0, 640.0), rng.uniform(0.0, 480.0)])
        .collect();
    let opts = FundamentalOptions {
        min_inliers: 30,
        max_error_px: 1.0,
        max_iterations: 3_000,
        ..base_opts()
    };
    assert!(estimate_fundamental(&x1, &x2, &opts).is_none());
}

#[test]
fn returned_matrix_is_rank2() {
    let pair = make_pair(11, 150, 600.0, 600.0, 0.3, 0.3);
    let est = estimate_fundamental(&pair.x1, &pair.x2, &base_opts()).unwrap();
    assert!(rank_ratio(&est.f_matrix) < 1e-9);
    assert!(
        (est.f_matrix.norm() - 1.0).abs() < 1e-9,
        "unit Frobenius norm"
    );
}

#[test]
fn determinism_same_seed_bit_identical() {
    let pair = make_pair(55, 150, 600.0, 600.0, 0.4, 0.4);
    let opts = base_opts();
    let a = estimate_fundamental(&pair.x1, &pair.x2, &opts).unwrap();
    let b = estimate_fundamental(&pair.x1, &pair.x2, &opts).unwrap();
    let bits = |m: &Matrix3<f64>| m.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&a.f_matrix), bits(&b.f_matrix));
    assert_eq!(a.inliers, b.inliers);
    assert_eq!(a.iterations, b.iterations);
}

// ── Focal length (Bougnoux) ──────────────────────────────────────────────────

#[test]
fn focal_exact_recovery() {
    let mut recovered = 0;
    for seed in 0..100u64 {
        let f1 = 400.0 + (seed as f64) * 3.0;
        let f2 = 900.0 - (seed as f64) * 2.0;
        let pair = make_pair(seed + 1000, 20, f1, f2, 0.0, 0.0);
        // Exact F recovers the generating focal to floating-point accuracy.
        if let Some(f) = focal_from_fundamental(&pair.f_true, pair.pp1, pair.pp2) {
            assert!(
                (f - pair.f1).abs() / pair.f1 < 1e-6,
                "seed {seed}: recovered {f}, true {}",
                pair.f1
            );
            recovered += 1;
        }
    }
    // The overwhelming majority of random non-degenerate poses recover.
    assert!(recovered > 90, "only {recovered}/100 focals recovered");
}

#[test]
fn focal_noisy_median_within_tolerance() {
    // RANSAC-estimated F from noisy correspondences: the median focal over many
    // pairs lands within a few percent of the truth.
    let f1 = 620.0;
    let mut focals = Vec::new();
    for seed in 0..40u64 {
        let pair = make_pair(seed + 5000, 140, f1, f1, 0.5, 0.2);
        let opts = FundamentalOptions {
            max_error_px: 2.0,
            min_inliers: 20,
            max_iterations: 20_000,
            ..base_opts()
        };
        if let Some(est) = estimate_fundamental(&pair.x1, &pair.x2, &opts) {
            if let Some(f) = focal_from_fundamental(&est.f_matrix, pair.pp1, pair.pp2) {
                focals.push(f);
            }
        }
    }
    assert!(
        focals.len() > 20,
        "too few focal estimates: {}",
        focals.len()
    );
    focals.sort_by(f64::total_cmp);
    let median = focals[focals.len() / 2];
    assert!(
        (median - f1).abs() / f1 < 0.05,
        "median focal {median}, true {f1}"
    );
}

#[test]
fn focal_degenerate_returns_none() {
    // Rotation-only motion: shared camera center → F = 0 → None.
    let k1 = k_of(600.0, 320.0, 240.0);
    let k2 = k_of(650.0, 300.0, 260.0);
    let center = Vector3::new(0.2, -0.1, 0.3);
    let (r1, _) = look_at(center, Vector3::new(1.0, 0.0, 4.0));
    let (r2, _) = look_at(center, Vector3::new(-0.5, 0.5, 4.0));
    let t1 = -r1 * center;
    let t2 = -r2 * center;
    let f_rot = compute_fundamental_matrix(&k1, &r1, &t1, &k2, &r2, &t2).unwrap();
    assert!(
        focal_from_fundamental(&f_rot, [320.0, 240.0], [300.0, 260.0]).is_none(),
        "rotation-only should be degenerate"
    );

    // Pure forward translation along the shared optical axis.
    let eye1 = Vector3::new(0.0, 0.0, 0.0);
    let (r, _) = look_at(eye1, Vector3::new(0.0, 0.0, 5.0));
    let fwd_world = r.transpose() * Vector3::new(0.0, 0.0, 1.0);
    let eye2 = eye1 + 1.0 * fwd_world;
    let tf1 = -r * eye1;
    let tf2 = -r * eye2;
    let f_fwd = compute_fundamental_matrix(&k1, &r, &tf1, &k2, &r, &tf2).unwrap();
    assert!(
        focal_from_fundamental(&f_fwd, [320.0, 240.0], [300.0, 260.0]).is_none(),
        "forward translation should be degenerate"
    );

    // Fixating cameras: optical axes intersect at a common target.
    let target = Vector3::new(0.0, 0.0, 5.0);
    let (rf1, _) = look_at(Vector3::new(-1.0, 0.0, 0.0), target);
    let (rf2, _) = look_at(Vector3::new(1.0, 0.2, 0.0), target);
    let tfx1 = -rf1 * Vector3::new(-1.0, 0.0, 0.0);
    let tfx2 = -rf2 * Vector3::new(1.0, 0.2, 0.0);
    let f_fix = compute_fundamental_matrix(&k1, &rf1, &tfx1, &k2, &rf2, &tfx2).unwrap();
    assert!(
        focal_from_fundamental(&f_fix, [320.0, 240.0], [300.0, 260.0]).is_none(),
        "fixating cameras should be degenerate"
    );
}

/// World-to-camera optical rotation looking from `eye` toward `target`
/// (+Z forward). Returns `(R, C)` with the camera center `C = eye`.
fn look_at(eye: Vector3<f64>, target: Vector3<f64>) -> (Matrix3<f64>, Vector3<f64>) {
    let z = (target - eye).normalize();
    let a = if z[0].abs() < 0.9 {
        Vector3::new(1.0, 0.0, 0.0)
    } else {
        Vector3::new(0.0, 1.0, 0.0)
    };
    let x = (a - z * a.dot(&z)).normalize();
    let y = z.cross(&x);
    let r = Matrix3::from_rows(&[x.transpose(), y.transpose(), z.transpose()]);
    (r, eye)
}
