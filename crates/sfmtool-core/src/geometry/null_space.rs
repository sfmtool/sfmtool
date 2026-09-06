// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The minimal solvers' null space.
//!
//! A rank-8 minimal sample of a 9-parameter model — the epipolar and
//! homography kernels' 8-row designs — has a one-dimensional null space, and
//! [`null9_from_8rows`] returns it directly rather than through an eigen
//! decomposition of `AᵀA`. The pivot rule is part of the determinism
//! contract, so this lives in one place: a drifted copy would change
//! reconstruction results without failing to compile.

/// Diagnostic switch restoring the `AᵀA` + 9×9 `symmetric_eigen` minimal
/// solvers in place of [`null9_from_8rows`], set by
/// `SFMTOOL_FOCAL_VOTE_EIGEN_MINSOLVE`.
static EIGEN_MINSOLVE: std::sync::LazyLock<bool> =
    std::sync::LazyLock::new(|| std::env::var_os("SFMTOOL_FOCAL_VOTE_EIGEN_MINSOLVE").is_some());

/// Whether the minimal samples take the eigen solvers instead of
/// [`null9_from_8rows`]. See [`EIGEN_MINSOLVE`].
#[inline]
pub(crate) fn eigen_minsolve_enabled() -> bool {
    *EIGEN_MINSOLVE
}

/// Unit-norm right null vector of an 8×9 system by Gaussian elimination with
/// partial pivoting.
///
/// A generic rank-8 minimal sample has a one-dimensional null space, and this
/// returns it directly instead of taking the smallest eigenvector of `AᵀA`:
/// better conditioned (no squaring of the condition number), about an order of
/// magnitude cheaper, and free of an iterative decomposition. The pivot rule is
/// part of the determinism contract — strict `>` when scanning a column, so the
/// *first* maximal pivot is kept.
///
/// Rank-deficient input takes the last free column with the other free
/// coordinates zero: a deterministic member of the null space, and such
/// degenerate samples score few inliers and lose the RANSAC regardless.
/// `None` when the design carries no pivot at all, or the result is
/// non-finite.
pub(crate) fn null9_from_8rows(mut a: [[f64; 9]; 8]) -> Option<[f64; 9]> {
    let mut pivot_cols = [usize::MAX; 8];
    let mut rank = 0usize;
    let mut col = 0usize;
    while rank < 8 && col < 9 {
        let mut best = rank;
        let mut best_v = a[rank][col].abs();
        for (r, row) in a.iter().enumerate().skip(rank + 1) {
            let v = row[col].abs();
            if v > best_v {
                best = r;
                best_v = v;
            }
        }
        if best_v <= 0.0 {
            col += 1;
            continue;
        }
        a.swap(rank, best);
        let (upper, lower) = a.split_at_mut(rank + 1);
        let pivot_row = &upper[rank];
        for row in lower.iter_mut() {
            let f = row[col] / pivot_row[col];
            if f != 0.0 {
                row[col] = 0.0;
                for (x, &p) in row.iter_mut().zip(pivot_row.iter()).skip(col + 1) {
                    *x -= f * p;
                }
            }
        }
        pivot_cols[rank] = col;
        rank += 1;
        col += 1;
    }
    if rank == 0 {
        return None;
    }
    let mut is_pivot = [false; 9];
    for &pc in &pivot_cols[..rank] {
        is_pivot[pc] = true;
    }
    let free = (0..9).rev().find(|&c| !is_pivot[c])?;
    let mut v = [0.0f64; 9];
    v[free] = 1.0;
    for r in (0..rank).rev() {
        let pc = pivot_cols[r];
        let mut s = 0.0;
        for c in pc + 1..9 {
            if v[c] != 0.0 {
                s += a[r][c] * v[c];
            }
        }
        v[pc] = -s / a[r][pc];
    }
    let n = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if !n.is_finite() || n <= 0.0 {
        return None;
    }
    for x in v.iter_mut() {
        *x /= n;
    }
    v.iter().all(|x| x.is_finite()).then_some(v)
}

#[cfg(test)]
mod tests;
