// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use super::*;
use nalgebra::SMatrix;

/// Uniform `[0, 1)` from the module's own RNG, so the fixtures below need no
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

/// Random 8×9 design of full row rank.
fn random_rank8(state: &mut u64) -> [[f64; 9]; 8] {
    let mut a = [[0.0f64; 9]; 8];
    for row in a.iter_mut() {
        for v in row.iter_mut() {
            *v = 2.0 * unit(state) - 1.0;
        }
    }
    a
}

/// Smallest eigenvector of `AᵀA` — the solver this replaces, kept here as the
/// reference the elimination is checked against.
fn null9_via_eigen(a: &[[f64; 9]; 8]) -> [f64; 9] {
    let mut ata = SMatrix::<f64, 9, 9>::zeros();
    for row in a.iter() {
        for i in 0..9 {
            for j in 0..9 {
                ata[(i, j)] += row[i] * row[j];
            }
        }
    }
    let eig = ata.symmetric_eigen();
    let mut best = 0usize;
    for j in 1..9 {
        if eig.eigenvalues[j] < eig.eigenvalues[best] {
            best = j;
        }
    }
    let c = eig.eigenvectors.column(best);
    std::array::from_fn(|i| c[i])
}

/// The elimination has to return *the* null direction, not merely a small
/// residual: the span check against the eigen path is what would catch a
/// back-substitution that solved a different system and still looked small.
#[test]
fn null9_from_8rows_spans_the_null_space() {
    let mut state = 0xc0ffee_u64;
    for _ in 0..200 {
        let a = random_rank8(&mut state);
        let v = null9_from_8rows(a).expect("generic design has a null space");
        let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-12, "norm {norm}");
        for row in a.iter() {
            let r: f64 = (0..9).map(|c| row[c] * v[c]).sum();
            assert!(r.abs() < 1e-9, "residual {r:e}");
        }
        let e = null9_via_eigen(&a);
        let dot: f64 = (0..9).map(|i| v[i] * e[i]).sum::<f64>().abs();
        assert!((dot - 1.0).abs() < 1e-6, "|v·v_eig| = {dot}");
    }
}

/// A design that constrains nothing has no direction to return.
#[test]
fn null9_from_8rows_rejects_the_zero_design() {
    assert!(null9_from_8rows([[0.0f64; 9]; 8]).is_none());
}

/// A rank-deficient sample (a duplicated correspondence, a degenerate
/// configuration) still has to yield *some* member of its null space rather
/// than a NaN — the RANSAC scores it and moves on.
#[test]
fn null9_from_8rows_handles_rank_deficiency() {
    let mut state = 0x1234_5678u64;
    let mut a = random_rank8(&mut state);
    a[7] = a[3];
    a[6] = a[2];
    let v = null9_from_8rows(a).expect("rank-6 design still has a null space");
    assert!(v.iter().all(|x| x.is_finite()));
    let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    assert!((norm - 1.0).abs() < 1e-12, "norm {norm}");
    for row in a.iter() {
        let r: f64 = (0..9).map(|c| row[c] * v[c]).sum();
        assert!(r.abs() < 1e-9, "residual {r:e}");
    }
}
