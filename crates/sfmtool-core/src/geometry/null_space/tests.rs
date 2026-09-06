// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::numeric::splitmix64;

/// Uniform `[0, 1)` from the crate's own RNG, so the fixtures below need no
/// dependency and reproduce exactly.
fn unit(state: &mut u64) -> f64 {
    (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64
}
use nalgebra::SMatrix;

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
