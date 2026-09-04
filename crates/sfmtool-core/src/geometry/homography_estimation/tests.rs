// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use super::*;

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
}

/// Project a pixel through a homography.
fn apply_h(h: &Matrix3<f64>, p: [f64; 2]) -> [f64; 2] {
    let v = h * Vector3::new(p[0], p[1], 1.0);
    [v[0] / v[2], v[1] / v[2]]
}

/// A random well-conditioned homography (perspective + affine, near identity).
fn random_homography(rng: &mut Lcg) -> Matrix3<f64> {
    Matrix3::new(
        1.0 + rng.uniform(-0.2, 0.2),
        rng.uniform(-0.2, 0.2),
        rng.uniform(-40.0, 40.0),
        rng.uniform(-0.2, 0.2),
        1.0 + rng.uniform(-0.2, 0.2),
        rng.uniform(-40.0, 40.0),
        rng.uniform(-3e-4, 3e-4),
        rng.uniform(-3e-4, 3e-4),
        1.0,
    )
}

fn scale_diff(a: &Matrix3<f64>, b: &Matrix3<f64>) -> f64 {
    let an = a / a.norm();
    let bn = b / b.norm();
    (an - bn).norm().min((an + bn).norm())
}

#[test]
fn four_point_exact_recovery() {
    for seed in 0..200u64 {
        let mut rng = Lcg(seed.wrapping_mul(0x9e3779b9).wrapping_add(1));
        let h_true = random_homography(&mut rng);
        let x1: Vec<[f64; 2]> = (0..4)
            .map(|_| [rng.uniform(0.0, 640.0), rng.uniform(0.0, 480.0)])
            .collect();
        let x2: Vec<[f64; 2]> = x1.iter().map(|&p| apply_h(&h_true, p)).collect();
        let s1: [[f64; 2]; 4] = core::array::from_fn(|k| x1[k]);
        let s2: [[f64; 2]; 4] = core::array::from_fn(|k| x2[k]);
        let h = homography_4pt(&s1, &s2).expect("valid minimal sample");
        assert!(
            scale_diff(&h, &h_true) < 1e-6,
            "seed {seed}: H diff {}",
            scale_diff(&h, &h_true)
        );
    }
}

#[test]
fn dlt_degenerate_returns_none() {
    // Too few points.
    assert!(homography_dlt(&[[0.0, 0.0]; 3], &[[0.0, 0.0]; 3]).is_none());
    // Coincident points (zero spread).
    let same = vec![[100.0, 100.0]; 8];
    assert!(homography_dlt(&same, &same).is_none());
    // Non-finite.
    let mut bad = vec![[10.0, 20.0], [30.0, 40.0], [50.0, 60.0], [70.0, 80.0]];
    bad[2][1] = f64::NAN;
    assert!(homography_dlt(&bad, &bad).is_none());
}

#[test]
fn ransac_recovers_planted_h_under_contamination() {
    for &inlier_frac in &[0.9, 0.7, 0.5] {
        let mut rng = Lcg(777);
        let h_true = random_homography(&mut rng);
        let n = 160;
        let n_out = ((1.0 - inlier_frac) * n as f64).round() as usize;
        let mut x1 = Vec::new();
        let mut x2 = Vec::new();
        for _ in 0..(n - n_out) {
            let p = [rng.uniform(0.0, 640.0), rng.uniform(0.0, 480.0)];
            let mut q = apply_h(&h_true, p);
            q[0] += 0.2 * rng.gaussian();
            q[1] += 0.2 * rng.gaussian();
            x1.push(p);
            x2.push(q);
        }
        for _ in 0..n_out {
            x1.push([rng.uniform(0.0, 640.0), rng.uniform(0.0, 480.0)]);
            x2.push([rng.uniform(0.0, 640.0), rng.uniform(0.0, 480.0)]);
        }
        let opts = HomographyOptions {
            max_error_px: 2.0,
            min_inliers: 20,
            max_iterations: 40_000,
            seed: 3,
            ..Default::default()
        };
        let est = estimate_homography(&x1, &x2, &opts)
            .unwrap_or_else(|| panic!("frac {inlier_frac}: expected an estimate"));
        assert!(
            scale_diff(&est.h_matrix, &h_true) < 5e-2,
            "frac {inlier_frac}: H diff {}",
            scale_diff(&est.h_matrix, &h_true)
        );
        let found = est.inliers.iter().filter(|&&b| b).count();
        assert!(
            found + 5 >= n - n_out,
            "frac {inlier_frac}: found {found} of {} inliers",
            n - n_out
        );
    }
}

#[test]
fn ransac_none_on_garbage() {
    let mut rng = Lcg(42);
    let x1: Vec<[f64; 2]> = (0..100)
        .map(|_| [rng.uniform(0.0, 640.0), rng.uniform(0.0, 480.0)])
        .collect();
    let x2: Vec<[f64; 2]> = (0..100)
        .map(|_| [rng.uniform(0.0, 640.0), rng.uniform(0.0, 480.0)])
        .collect();
    let opts = HomographyOptions {
        max_error_px: 1.0,
        min_inliers: 40,
        max_iterations: 5_000,
        seed: 1,
        ..Default::default()
    };
    assert!(estimate_homography(&x1, &x2, &opts).is_none());
}

#[test]
fn determinism_same_seed_bit_identical() {
    let mut rng = Lcg(9);
    let h_true = random_homography(&mut rng);
    let mut x1 = Vec::new();
    let mut x2 = Vec::new();
    for _ in 0..120 {
        let p = [rng.uniform(0.0, 640.0), rng.uniform(0.0, 480.0)];
        let mut q = apply_h(&h_true, p);
        q[0] += 0.4 * rng.gaussian();
        q[1] += 0.4 * rng.gaussian();
        x1.push(p);
        x2.push(q);
    }
    let opts = HomographyOptions {
        seed: 55,
        ..Default::default()
    };
    let a = estimate_homography(&x1, &x2, &opts).unwrap();
    let b = estimate_homography(&x1, &x2, &opts).unwrap();
    let bits = |m: &Matrix3<f64>| m.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&a.h_matrix), bits(&b.h_matrix));
    assert_eq!(a.inliers, b.inliers);
    assert_eq!(a.iterations, b.iterations);
}

// ── SIMD parity ──────────────────────────────────────────────────────────────

/// The AVX2 `score_h` is a lane-per-correspondence repetition of the scalar
/// loop, so its mask and count must agree exactly. Lengths sweep the 4-wide
/// lane boundary and its tail; the point set mixes ordinary correspondences
/// with ones that drive the transfer onto the plane at infinity (`w ≈ 0`, the
/// `+∞` guard) and with non-finite input, which is where a compare that quieted
/// NaNs the other way would diverge.
#[test]
#[cfg(target_arch = "x86_64")]
fn score_h_simd_matches_scalar() {
    if !std::is_x86_feature_detected!("avx2") {
        return;
    }
    let mut rng = Lcg(4242);
    // A homography whose last row is large enough that some points really do
    // transfer through a near-zero third coordinate.
    let h = Matrix3::new(1.02, 0.03, 12.0, -0.04, 0.98, -7.0, 1e-3, -2e-3, 1.0);
    let h_inv = h.try_inverse().expect("invertible");
    for &len in &[0usize, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 64, 199] {
        let x1: Vec<[f64; 2]> = (0..len)
            .map(|i| match i % 13 {
                3 => [500.0, 500.0], // lands near the vanishing line of `h`
                7 => [f64::NAN, 1.0],
                _ => [rng.uniform(-600.0, 600.0), rng.uniform(-600.0, 600.0)],
            })
            .collect();
        let x2: Vec<[f64; 2]> = x1
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let q = apply_h(&h, *p);
                let n = if i % 5 == 0 { 40.0 } else { 0.4 };
                [q[0] + n * rng.gaussian(), q[1] + n * rng.gaussian()]
            })
            .collect();
        for &thresh2 in &[1.0, 9.0, 1e6] {
            let mut want = vec![false; len];
            let mut got = vec![false; len];
            let want_n = score_h_scalar(&h, &h_inv, &x1, &x2, thresh2, &mut want);
            // SAFETY: avx2 confirmed above; `x2` and the mask both cover `x1`.
            let got_n = unsafe { score_h_avx2(&h, &h_inv, &x1, &x2, thresh2, &mut got) };
            assert_eq!(got_n, want_n, "len {len} thresh2 {thresh2}");
            assert_eq!(got, want, "len {len} thresh2 {thresh2}");
            // ...and the dispatcher must reach the same place.
            let mut dispatched = vec![false; len];
            assert_eq!(
                score_h(&h, &x1, &x2, thresh2, &mut dispatched),
                want_n,
                "dispatch len {len}"
            );
            assert_eq!(dispatched, want, "dispatch len {len}");
        }
    }
}

/// The degenerate-infinity guard, pinned on its own: a point exactly on the
/// vanishing line of `h` transfers to `w = 0`, and the SIMD path must reproduce
/// the scalar early return's `+∞` (never an inlier) by blending rather than by
/// skipping the lane — including when its three lane-mates are inliers.
#[test]
#[cfg(target_arch = "x86_64")]
fn score_h_simd_reproduces_the_infinity_guard() {
    if !std::is_x86_feature_detected!("avx2") {
        return;
    }
    // Last row (0, 1, -1): the point y = 1 maps to w = 0 exactly.
    let h = Matrix3::new(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, -1.0);
    let h_inv = h.try_inverse().expect("invertible");
    let x1 = [[3.0, 1.0], [2.0, 5.0], [-4.0, 9.0], [7.0, 3.0]];
    let x2: Vec<[f64; 2]> = x1.iter().map(|p| apply_h(&h, *p)).collect();
    assert!(
        symmetric_transfer_sq(&h, &h_inv, x1[0], x2[0]).is_infinite(),
        "fixture must hit the degenerate branch"
    );
    let mut want = vec![false; 4];
    let mut got = vec![false; 4];
    let want_n = score_h_scalar(&h, &h_inv, &x1, &x2, 9.0, &mut want);
    // SAFETY: avx2 confirmed above; every slice holds the same four points.
    let got_n = unsafe { score_h_avx2(&h, &h_inv, &x1, &x2, 9.0, &mut got) };
    assert_eq!(want, [false, true, true, true], "guard marks only lane 0");
    assert_eq!(got, want);
    assert_eq!(got_n, want_n);
}
