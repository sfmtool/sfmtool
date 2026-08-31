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

/// Correspondences from 3D points drawn in a slab of thickness `slab` about a
/// fronto-parallel plane at depth 5, with the second camera displaced by
/// `baseline`. `slab = 0` is exactly coplanar structure; `baseline = 0` is pure
/// rotation. Both make the 8-point design rank-deficient: a homography then
/// relates the views, so every `F = [e]ₓ·H` fits, for any epipole `e`.
fn slab_pair(seed: u64, n: usize, slab: f64, baseline: f64) -> (Vec<[f64; 2]>, Vec<[f64; 2]>) {
    let mut rng = Lcg(seed.wrapping_mul(0x9e3779b97f4a7c15).wrapping_add(1));
    let k1 = k_of(700.0, 320.0, 240.0);
    let k2 = k_of(640.0, 300.0, 260.0);
    let r1 = rng.rotation_matrix(0.4);
    let r2 = rng.rotation_matrix(0.4);
    let t2 = Vector3::new(baseline, 0.0, 0.0);
    let r1t = r1.transpose();
    let (mut x1, mut x2) = (Vec::with_capacity(n), Vec::with_capacity(n));
    while x1.len() < n {
        let cam1 = Vector3::new(
            rng.uniform(-2.0, 2.0),
            rng.uniform(-2.0, 2.0),
            5.0 + slab * rng.uniform(-1.0, 1.0),
        );
        let cam2 = r2 * (r1t * cam1) + t2;
        if cam2.z <= 0.2 {
            continue;
        }
        let p1 = k1 * cam1;
        let p2 = k2 * cam2;
        x1.push([p1.x / p1.z, p1.y / p1.z]);
        x2.push([p2.x / p2.z, p2.y / p2.z]);
    }
    (x1, x2)
}

/// A rank-deficient design carries fewer than eight independent constraints, so
/// its null space is more than one dimensional and the smallest eigenvector is
/// an arbitrary member of it. Rank-2 enforcement still yields a well-formed
/// matrix that scores a *full* consensus on the degenerate points, so nothing
/// downstream can catch it — the guard is the only line of defence.
#[test]
fn eight_point_rejects_rank_deficient_designs() {
    // Coplanar structure, from exactly planar to a slab thin against its depth.
    for slab in [0.0, 1e-6, 1e-4] {
        let (x1, x2) = slab_pair(7, 40, slab, 1.0);
        assert!(
            fundamental_8pt(&x1, &x2).is_none(),
            "coplanar structure (slab {slab}) must be rejected"
        );
    }

    // Zero and near-zero baseline: a homography relates the views.
    for baseline in [0.0, 1e-6, 1e-4] {
        let (x1, x2) = slab_pair(7, 40, 2.0, baseline);
        assert!(
            fundamental_8pt(&x1, &x2).is_none(),
            "baseline {baseline} must be rejected"
        );
    }

    // Fewer than eight *distinct* correspondences, padded out to a length that
    // passes the `n < 8` check.
    let pair = make_pair(3, 24, 700.0, 640.0, 0.0, 0.0);
    for k in [4usize, 6, 7] {
        let a: Vec<_> = (0..24).map(|i| pair.x1[i % k]).collect();
        let b: Vec<_> = (0..24).map(|i| pair.x2[i % k]).collect();
        assert!(
            fundamental_8pt(&a, &b).is_none(),
            "{k} distinct correspondences must be rejected"
        );
    }

    // Collinear image points in both views.
    let a: Vec<[f64; 2]> = (0..20)
        .map(|i| [20.0 * i as f64, 10.0 * i as f64])
        .collect();
    let b: Vec<[f64; 2]> = (0..20)
        .map(|i| [15.0 * i as f64 + 3.0, 9.0 * i as f64 - 1.0])
        .collect();
    assert!(
        fundamental_8pt(&a, &b).is_none(),
        "collinear points must be rejected"
    );
}

/// The guard is a rank test, not a conditioning test: over-determined designs
/// in general position survive it even under noise far past anything the
/// estimator would call an inlier. Their rank margin sits near `1e-3`, twelve
/// orders above every rejection above.
#[test]
fn eight_point_accepts_healthy_designs() {
    for seed in 0..100u64 {
        let exact = make_pair(seed, 12, 700.0, 640.0, 0.0, 0.0);
        let f = fundamental_8pt(&exact.x1, &exact.x2)
            .unwrap_or_else(|| panic!("seed {seed}: general-position design rejected"));
        assert!(scale_diff(&f, &exact.f_true) < 1e-6);

        let noisy = make_pair(seed, 40, 700.0, 640.0, 5.0, 0.0);
        assert!(
            fundamental_8pt(&noisy.x1, &noisy.x2).is_some(),
            "seed {seed}: 5px noise must not read as rank deficiency"
        );
    }
}

/// At exactly eight correspondences there is one constraint per unknown, so an
/// unlucky general-position draw is genuinely near-degenerate: measured over
/// 400 seeds, three land at a margin of `1e-10`-`1e-13` and are rejected, and
/// the answer the unguarded solver gives for those is itself only good to
/// `~1e-7` against the `1e-15` its accepted siblings reach. Rejecting them is
/// the intended trade — [`local_optimize_f`] simply keeps the minimal solution
/// it already had — but the rate belongs in a test, since a threshold change
/// that pushed it up would quietly disable local optimization at the floor.
#[test]
fn eight_point_minimal_designs_are_almost_always_accepted() {
    let mut accepted = 0;
    for seed in 0..400u64 {
        let pair = make_pair(seed, 8, 700.0, 640.0, 0.0, 0.0);
        if let Some(f) = fundamental_8pt(&pair.x1, &pair.x2) {
            accepted += 1;
            assert!(
                scale_diff(&f, &pair.f_true) < 1e-6,
                "seed {seed}: accepted minimal design off by {}",
                scale_diff(&f, &pair.f_true)
            );
        }
    }
    assert!(
        accepted >= 390,
        "only {accepted}/400 minimal designs accepted"
    );
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

/// Correspondences with `plane_frac` of the points on a common plane and the
/// rest in general position, plus the true `F`.
fn dominant_plane_pair(
    seed: u64,
    n: usize,
    plane_frac: f64,
) -> (Vec<[f64; 2]>, Vec<[f64; 2]>, Matrix3<f64>) {
    let mut rng = Lcg(seed.wrapping_mul(0x9e3779b97f4a7c15).wrapping_add(7));
    let k1 = k_of(700.0, 320.0, 240.0);
    let k2 = k_of(640.0, 300.0, 260.0);
    let r1 = rng.rotation_matrix(0.3);
    let t1 = Vector3::zeros();
    let r2 = rng.rotation_matrix(0.3);
    let t2 = Vector3::new(1.0, 0.1, 0.05);
    let f_true =
        compute_fundamental_matrix(&k1, &r1, &t1, &k2, &r2, &t2).expect("non-singular intrinsics");
    let n_plane = (n as f64 * plane_frac).round() as usize;
    let r1t = r1.transpose();
    let (mut x1, mut x2) = (Vec::with_capacity(n), Vec::with_capacity(n));
    while x1.len() < n {
        let z = if x1.len() < n_plane {
            5.0
        } else {
            rng.uniform(2.0, 9.0)
        };
        let cam1 = Vector3::new(rng.uniform(-2.0, 2.0), rng.uniform(-2.0, 2.0), z);
        let cam2 = r2 * (r1t * (cam1 - t1)) + t2;
        if cam2.z <= 0.2 {
            continue;
        }
        let p1 = k1 * cam1;
        let p2 = k2 * cam2;
        x1.push([p1.x / p1.z, p1.y / p1.z]);
        x2.push([p2.x / p2.z, p2.y / p2.z]);
    }
    (x1, x2, f_true)
}

/// A scene dominated by one plane is how a coplanar inlier set reaches the
/// 8-point refit: the minimal sample is drawn in general position, but its
/// consensus can collapse onto the plane. Run in isolation on such a set, the
/// unguarded refit returns a matrix that fits the plane perfectly — up to 76 of
/// 80 inliers — while sitting 2-4% from the true `F` and voting focals between
/// 215 and 1274 against a true 700.
///
/// Through the estimator the rank guard is a backstop rather than the active
/// defence, because [`fundamental_7pt`]'s own guard already rejects the
/// all-coplanar minimal samples: measured across these scenes, output is
/// bit-identical either side of the fix. What this test holds is the healthy
/// half of that claim — up to a `0.7` plane fraction the geometry is recovered
/// exactly, so a future change to either guard cannot quietly start rejecting
/// the refits that matter.
///
/// It stops at `0.7` deliberately. From about `0.85` the estimator's adaptive
/// termination will settle on a plane-only consensus and stop early (worst
/// measured: `7e-4` in `F`, `411px` in focal), which is a limitation of the
/// stopping rule, not of this guard — it predates the fix and is unchanged by
/// it.
#[test]
fn estimator_survives_a_dominant_plane() {
    for plane_frac in [0.5, 0.7] {
        for scene in 0..12u64 {
            let (x1, x2, f_true) = dominant_plane_pair(scene, 80, plane_frac);
            for seed in [7u64, 11, 23, 101] {
                let opts = FundamentalOptions {
                    seed,
                    ..base_opts()
                };
                let est = estimate_fundamental(&x1, &x2, &opts).unwrap_or_else(|| {
                    panic!("plane_frac {plane_frac}, scene {scene}, seed {seed}: no consensus")
                });
                let diff = scale_diff(&est.f_matrix, &f_true);
                assert!(
                    diff < 1e-8,
                    "plane_frac {plane_frac}, scene {scene}, seed {seed}: F off by {diff:.3e}"
                );
                let focal = focal_from_fundamental(&est.f_matrix, [320.0, 240.0], [300.0, 260.0])
                    .unwrap_or_else(|| {
                        panic!("plane_frac {plane_frac}, scene {scene}, seed {seed}: no focal")
                    });
                assert!(
                    (focal - 700.0).abs() < 0.5,
                    "plane_frac {plane_frac}, scene {scene}, seed {seed}: focal {focal}"
                );
            }
        }
    }
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
