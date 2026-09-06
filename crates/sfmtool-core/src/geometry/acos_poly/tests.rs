// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::numeric::splitmix64;

/// Uniform `[0, 1)` from the crate's own RNG, so the fixtures below need no
/// dependency and reproduce exactly.
fn unit(state: &mut u64) -> f64 {
    (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64
}

/// The polynomial replaces a libm call in the focal vote's hottest loop, so
/// the bar is that no *decision* can move: an absolute error under `5e-16` is
/// below the ULP of `π` and far below any residual tolerance the vote applies.
/// The near-`±1` populations matter more than the bulk grid — that is where
/// the branch changes and where `acos` is steepest.
#[test]
fn acos_poly_tracks_libm_over_the_domain() {
    let mut worst = 0.0f64;
    let n = 200_001usize;
    for k in 0..n {
        let d = -1.0 + 2.0 * (k as f64) / ((n - 1) as f64);
        worst = worst.max((acos_poly_scalar(d) - d.acos()).abs());
    }
    let mut state = 0x5f3a_11c7u64;
    for _ in 0..20_000 {
        // Crowd the ends: `1 - u³` puts most samples inside 1e-3 of `±1`.
        let u = unit(&mut state);
        let mag = 1.0 - u * u * u;
        for d in [mag, -mag] {
            worst = worst.max((acos_poly_scalar(d) - d.acos()).abs());
        }
    }
    assert!(worst < 5e-16, "worst acos error {worst:e}");
}

/// The three arguments the kernel actually hits often — a clamped cosine of
/// `±1` from a coincident or antipodal ray pair, and `0` — plus the branch
/// boundary at `±0.5`, are exact or within one ULP of libm.
#[test]
fn acos_poly_is_exact_at_the_landmarks() {
    assert_eq!(acos_poly_scalar(1.0), 0.0);
    assert_eq!(acos_poly_scalar(-1.0), std::f64::consts::PI);
    assert_eq!(acos_poly_scalar(0.0), std::f64::consts::FRAC_PI_2);
    assert_eq!(acos_poly_scalar(-0.0), std::f64::consts::FRAC_PI_2);
    for d in [0.5f64, -0.5] {
        let (got, want) = (acos_poly_scalar(d), d.acos());
        assert!(
            (got.to_bits() as i64 - want.to_bits() as i64).abs() <= 1,
            "acos({d}) = {got:?}, libm {want:?}"
        );
    }
}

/// A NaN cosine can only come from a NaN ray, and the vote's downstream
/// comparisons rely on it staying NaN rather than becoming a finite angle —
/// which is what libm did.
#[test]
fn acos_poly_propagates_nan() {
    assert!(acos_poly_scalar(f64::NAN).is_nan());
    assert!(acos_poly_scalar(-f64::NAN).is_nan());
}
