// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use super::super::tests::{rx, ry, Lcg};
use super::*;

// ── SIMD parity ──────────────────────────────────────────────────────────────

/// Rays for the parity tests: unit rays over the whole sphere, plus a handful
/// of shapes that exercise the guards — a zero ray (so `‖E x‖` hits the `1e-15`
/// floor), a huge one, and a non-finite one (so the NaN branches of `min`,
/// `max` and `clamp` are compared, which is where a mis-ordered SIMD operand
/// would show).
fn parity_rays(seed: u64, n: usize) -> Vec<Vector3<f64>> {
    let mut rng = Lcg(seed);
    (0..n)
        .map(|i| match i % 17 {
            5 => Vector3::zeros(),
            11 => Vector3::new(1e200, -1e200, 1e200),
            13 => Vector3::new(f64::NAN, 1.0, 0.0),
            _ => Vector3::new(rng.gaussian(), rng.gaussian(), rng.gaussian()).normalize(),
        })
        .collect()
}

/// The AVX2 residual kernels are lane-per-point repetitions of the scalar loop,
/// so their agreement is exact, not approximate — compare the bits. Lengths
/// sweep the 4-wide lane boundary and its tail.
#[test]
#[cfg(target_arch = "x86_64")]
fn epipolar_residuals_simd_matches_scalar() {
    if !std::is_x86_feature_detected!("avx2") {
        return;
    }
    let mut rng = Lcg(77);
    let e = Matrix3::from_fn(|_, _| rng.gaussian());
    for &len in &[0usize, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 64, 199] {
        let r1 = parity_rays(11, len);
        let r2 = parity_rays(29, len);
        for side_two in [true, false] {
            let mut want = vec![0.0f64; len];
            let mut got = vec![0.0f64; len];
            epipolar_residuals_scalar(&e, &r1, &r2, side_two, &mut want);
            // SAFETY: avx2 confirmed above; all three slices have length `len`.
            unsafe { epipolar_residuals_avx2(&e, &r1, &r2, side_two, &mut got) };
            for i in 0..len {
                assert_eq!(
                    got[i].to_bits(),
                    want[i].to_bits(),
                    "len {len} side_two {side_two} index {i}"
                );
            }
            // ...and the dispatcher must reach the same place.
            let mut dispatched = vec![0.0f64; len];
            epipolar_residuals(&e, &r1, &r2, side_two, &mut dispatched);
            for i in 0..len {
                assert_eq!(dispatched[i].to_bits(), want[i].to_bits(), "dispatch {i}");
            }
        }
    }
}

#[test]
#[cfg(target_arch = "x86_64")]
fn rotation_residuals_simd_matches_scalar() {
    if !std::is_x86_feature_detected!("avx2") {
        return;
    }
    let rot = rx(0.7) * ry(-1.3);
    for &len in &[0usize, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 64, 199] {
        let r1 = parity_rays(5, len);
        let r2 = parity_rays(23, len);
        let all: Vec<usize> = (0..len).collect();
        let mut want = vec![0.0f64; len];
        let mut got = vec![0.0f64; len];
        rotation_residuals_scalar(&rot, &r1, &r2, &all, &mut want);
        rotation_residuals(&rot, &r1, &r2, &all, &mut got);
        for i in 0..len {
            assert_eq!(got[i].to_bits(), want[i].to_bits(), "len {len} index {i}");
        }

        // A consecutive run that does not start at zero takes the same path;
        // an arbitrary subset falls back to the scalar loop. Both must agree
        // with the scalar reference over their own index list.
        for idx in [
            (len / 3..len).collect::<Vec<usize>>(),
            (0..len).filter(|i| i % 3 != 1).collect::<Vec<usize>>(),
        ] {
            let mut want = vec![0.0f64; idx.len()];
            let mut got = vec![0.0f64; idx.len()];
            rotation_residuals_scalar(&rot, &r1, &r2, &idx, &mut want);
            rotation_residuals(&rot, &r1, &r2, &idx, &mut got);
            for i in 0..idx.len() {
                assert_eq!(got[i].to_bits(), want[i].to_bits(), "len {len} sub {i}");
            }
        }
    }
}

/// The `f32` rays for the single-precision parity tests, narrowed from the
/// same shapes the `f64` kernels are exercised on so the guard branches (zero
/// ray, overflow, NaN) are compared there too.
fn parity_rays_f32(seed: u64, n: usize) -> RaysF32 {
    let mut out = RaysF32::zeros(n);
    for (i, v) in parity_rays(seed, n).into_iter().enumerate() {
        out.set(i, v);
    }
    out
}

/// The `f32` kernels claim exactness only against their own scalar twins —
/// they are a different arithmetic from the `f64` ones by construction — and
/// that twinning is what makes a slice length off the 8-wide boundary
/// harmless. Compare the bits, sweeping the lane boundary and its tail.
#[test]
#[cfg(target_arch = "x86_64")]
fn epipolar_residuals_f32_simd_matches_scalar() {
    if !std::is_x86_feature_detected!("avx2") {
        return;
    }
    let mut rng = Lcg(77);
    let e = Matrix3::from_fn(|_, _| rng.gaussian());
    for &len in &[0usize, 1, 3, 7, 8, 9, 15, 16, 17, 64, 199] {
        let r1 = parity_rays_f32(11, len);
        let r2 = parity_rays_f32(29, len);
        for side_two in [true, false] {
            let mut want = vec![0.0f32; len];
            let mut got = vec![0.0f32; len];
            epipolar_residuals_f32_scalar(&e, &r1, &r2, side_two, &mut want);
            // SAFETY: avx2 confirmed above; all three slices have length `len`.
            unsafe { epipolar_residuals_f32_avx2(&e, &r1, &r2, side_two, &mut got) };
            for i in 0..len {
                assert_eq!(
                    got[i].to_bits(),
                    want[i].to_bits(),
                    "len {len} side_two {side_two} index {i}"
                );
            }
            let mut dispatched = vec![0.0f32; len];
            epipolar_residuals_f32(&e, &r1, &r2, side_two, &mut dispatched);
            for i in 0..len {
                assert_eq!(dispatched[i].to_bits(), want[i].to_bits(), "dispatch {i}");
            }
        }
    }
}

#[test]
#[cfg(target_arch = "x86_64")]
fn rotation_residuals_f32_simd_matches_scalar() {
    if !std::is_x86_feature_detected!("avx2") {
        return;
    }
    let rot = rx(0.7) * ry(-1.3);
    for &len in &[0usize, 1, 3, 7, 8, 9, 15, 16, 17, 64, 199] {
        let r1 = parity_rays_f32(5, len);
        let r2 = parity_rays_f32(23, len);
        for idx in [
            (0..len).collect::<Vec<usize>>(),
            (len / 3..len).collect::<Vec<usize>>(),
            (0..len).filter(|i| i % 3 != 1).collect::<Vec<usize>>(),
        ] {
            let mut want = vec![0.0f32; idx.len()];
            let mut got = vec![0.0f32; idx.len()];
            rotation_residuals_f32_scalar(&rot, &r1, &r2, &idx, &mut want);
            rotation_residuals_f32(&rot, &r1, &r2, &idx, &mut got);
            for i in 0..idx.len() {
                assert_eq!(got[i].to_bits(), want[i].to_bits(), "len {len} index {i}");
            }
        }
    }
}

/// The constant-time consecutiveness test answers the same question the
/// element scan it replaced did, for every strictly increasing list — which is
/// the only shape the residual kernels are ever handed.
#[test]
#[cfg(target_arch = "x86_64")]
fn consecutive_run_matches_an_element_scan() {
    let lists: Vec<Vec<usize>> = vec![
        vec![],
        vec![0],
        vec![7],
        (0..40).collect(),
        (13..40).collect(),
        (0..40).filter(|i| i % 3 != 1).collect(),
        (0..40).filter(|&i| i != 39).collect(),
        (0..40).filter(|&i| i != 0).collect(),
        (0..80).step_by(2).collect(),
    ];
    for idx in &lists {
        for len in 0..=idx.len() {
            let scanned = len > 0 && idx[..len].iter().enumerate().all(|(j, &i)| i == idx[0] + j);
            assert_eq!(consecutive_run(idx, len), scanned, "idx {idx:?} len {len}");
        }
    }
}

/// Why the `f32` rotation residual measures `asin|r₁ × r₂|` and not `acos` of
/// the dot, stated as a measurement against angles that are known exactly.
///
/// Both forms are handed the same `f32`-narrowed rays. The cross product IS
/// `sin θ`, so it carries a small angle in its own leading digits; the dot is
/// `1 − θ²/2`, and below a milliradian that difference from `1` is smaller
/// than the rays' own representation error — so the `acos` form returns
/// nonsense (often exactly zero) at precisely the angles the consensus bounds
/// sit at. The `f64` arithmetic in `rotation_residuals` does not rescue it:
/// the information is gone before the `acos` is reached. Measured over the
/// `1e-4 .. 1e-3` rad band below: `4.6e-4` worst relative error through the
/// cross product, `2.2e0` through the dot — four orders apart. Out at
/// `1e-2 .. 1` rad the cross form measures `5.2e-6`.
#[test]
fn the_cross_product_form_carries_small_angles_the_dot_cannot() {
    let n = 4096;
    let rot = Matrix3::identity();
    let mut rng = Lcg(4242);
    let mut r1 = RaysF32::zeros(n);
    let mut r2 = RaysF32::zeros(n);
    let (mut a64, mut b64, mut truth) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..n {
        let base = Vector3::new(rng.gaussian(), rng.gaussian(), rng.gaussian()).normalize();
        // Angles spanning the inlier band (1e-4 .. 1e-3 rad) and out to 1.
        let ang = 10f64.powf(-5.0 + 5.0 * (i as f64 / n as f64));
        let perp = base.cross(&Vector3::new(0.3, -0.7, 0.6)).normalize();
        let other = (base * ang.cos() + perp * ang.sin()).normalize();
        a64.push(r1.set(i, base));
        b64.push(r2.set(i, other));
        truth.push(ang);
    }
    let idx: Vec<usize> = (0..n).collect();
    let mut cross = vec![0.0f32; n];
    let mut dot = vec![0.0f64; n];
    rotation_residuals_f32(&rot, &r1, &r2, &idx, &mut cross);
    rotation_residuals(&rot, &a64, &b64, &idx, &mut dot);

    let band = |lo: f64, hi: f64, v: &dyn Fn(usize) -> f64| -> f64 {
        let mut worst: f64 = 0.0;
        for (i, &t) in truth.iter().enumerate() {
            if t >= lo && t < hi {
                worst = worst.max((v(i) - t).abs() / t);
            }
        }
        worst
    };
    let cross_band = band(1e-4, 1e-3, &|i| f64::from(cross[i]));
    let dot_band = band(1e-4, 1e-3, &|i| dot[i]);
    assert!(
        cross_band < 2e-3,
        "cross-product form in the inlier band: {cross_band:e}"
    );
    assert!(
        dot_band > 1.0,
        "the dot form is supposed to be unusable in the inlier band, not {dot_band:e}"
    );
    // Out where the angle is comfortably above the rays' own precision, the
    // cross form is exact to what `f32` rays can express.
    let cross_wide = band(1e-2, 1.0, &|i| f64::from(cross[i]));
    assert!(
        cross_wide < 5e-5,
        "cross-product form at wide angles {cross_wide:e}"
    );
}
