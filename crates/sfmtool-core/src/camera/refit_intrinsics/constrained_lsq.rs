// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Linear least squares under linear inequality constraints.
//!
//! The spline fits minimize `‖A·x − b‖²` subject to `G·x ≥ 0`, with `x` at
//! most 33 wide and a few thousand constraint rows. The method is the one in
//! Lawson & Hanson, *Solving Least Squares Problems* (1974), chapter 23: the
//! SVD of `A` turns the problem into a least-distance problem, `min ‖z‖`
//! subject to `E·z ≥ f`, and that problem's dual is a non-negative least
//! squares problem solved by the Lawson–Hanson active-set algorithm. The
//! constraints a positive dual variable rests on are the active ones.
//!
//! The caller solves the unconstrained problem first and comes here only when
//! that solution breaks a constraint, so an unconstrained fit is not taken
//! through this arithmetic at all.

use nalgebra::{DMatrix, DVector};

/// Relative rank threshold of the small solves inside [`nnls`].
const RANK_TOLERANCE: f64 = 1e-14;

/// The solution of an inequality-constrained least-squares problem.
pub(super) struct Constrained {
    /// The minimizer.
    pub(super) x: DVector<f64>,
    /// For each constraint row, whether the solution rests on it: its dual
    /// variable is positive.
    pub(super) active: Vec<bool>,
}

/// Minimize `‖A·x − b‖²` subject to `G·x ≥ 0`, where `A = U·diag(s)·Vᵀ` is
/// given by its thin SVD (`u`, `singular_values`, `v_t`), every singular value
/// positive, and `unconstrained` is the minimizer without constraints,
/// `V·diag(s)⁻¹·Uᵀ·b`.
///
/// With `z = diag(s)·Vᵀ·x − Uᵀ·b` the objective is `‖z‖²` plus a constant,
/// and `x = unconstrained + V·diag(s)⁻¹·z`, so the constraints become
/// `E·z ≥ f` with `E = G·V·diag(s)⁻¹` and `f = −G·unconstrained`. Each row of
/// `E` is scaled to unit length, with its `f`, which changes neither the
/// feasible set nor the solution.
///
/// Returns `None` when the constraints admit no solution or the dual solve
/// does not converge.
pub(super) fn least_squares_with_inequalities(
    singular_values: &DVector<f64>,
    v_t: &DMatrix<f64>,
    unconstrained: &DVector<f64>,
    g: &DMatrix<f64>,
) -> Option<Constrained> {
    let n = unconstrained.len();
    let mut k = v_t.transpose();
    for (j, s) in singular_values.iter().enumerate() {
        k.column_mut(j).scale_mut(1.0 / s);
    }
    let mut e = g * &k;
    let mut f = -(g * unconstrained);
    for row in 0..e.nrows() {
        let norm = e.row(row).norm();
        if norm > 0.0 {
            e.row_mut(row).scale_mut(1.0 / norm);
            f[row] /= norm;
        }
    }
    debug_assert_eq!(e.ncols(), n);
    let (z, active) = least_distance(&e, &f)?;
    Some(Constrained {
        x: unconstrained + k * z,
        active,
    })
}

/// Minimize `‖z‖` subject to `E·z ≥ f` (Lawson & Hanson, algorithm LDP).
///
/// The dual is `min ‖M·u − eₙ‖` over `u ≥ 0` with `M = [Eᵀ; fᵀ]`; its residual
/// `r` gives `z = −r[..n] / r[n]`, and a residual of zero means no `z`
/// satisfies the constraints. Returns `z` and which constraints carry a
/// positive dual variable.
fn least_distance(e: &DMatrix<f64>, f: &DVector<f64>) -> Option<(DVector<f64>, Vec<bool>)> {
    let (m, n) = e.shape();
    let mut dual = DMatrix::<f64>::zeros(n + 1, m);
    for k in 0..m {
        for i in 0..n {
            dual[(i, k)] = e[(k, i)];
        }
        dual[(n, k)] = f[k];
    }
    let mut target = DVector::<f64>::zeros(n + 1);
    target[n] = 1.0;
    let u = nnls(&dual, &target)?;
    let r = &dual * &u - &target;
    // `r[n] = f·u − 1`; at a solution of the dual it is `−‖r‖²`, and it is
    // zero only where the constraints are infeasible.
    if r[n].abs() <= 1e-12 {
        return None;
    }
    let z = DVector::from_iterator(n, (0..n).map(|i| -r[i] / r[n]));
    let active = u.iter().map(|&v| v > 0.0).collect();
    Some((z, active))
}

/// Minimize `‖M·u − g‖` subject to `u ≥ 0` by the Lawson–Hanson active-set
/// algorithm (Lawson & Hanson, algorithm NNLS).
///
/// A column enters the passive set when its gradient component is the largest
/// positive one; the least-squares solve on the passive columns is then
/// followed along the segment from the current point until a passive variable
/// would turn negative, which leaves the set. A column whose own solve comes
/// back non-positive the moment it enters, which exact arithmetic rules out
/// and rounding does not, is set aside until the passive set next grows, so
/// the loop cannot pick it forever. Returns `None` when the iteration budget
/// runs out.
fn nnls(m: &DMatrix<f64>, g: &DVector<f64>) -> Option<DVector<f64>> {
    let (rows, cols) = m.shape();
    let mut u = DVector::<f64>::zeros(cols);
    let mut passive = vec![false; cols];
    let mut set_aside = vec![false; cols];
    let norm1 = m
        .column_iter()
        .map(|c| c.iter().map(|v| v.abs()).sum::<f64>())
        .fold(0.0, f64::max);
    let tolerance = 10.0 * f64::EPSILON * norm1 * rows.max(cols) as f64;

    for _ in 0..3 * cols.max(1) {
        let gradient = m.transpose() * (g - m * &u);
        let entering = (0..cols)
            .filter(|&j| !passive[j] && !set_aside[j] && gradient[j] > tolerance)
            .max_by(|&a, &b| gradient[a].total_cmp(&gradient[b]));
        let Some(entering) = entering else {
            return Some(u);
        };
        passive[entering] = true;

        let mut first = true;
        loop {
            let indices: Vec<usize> = (0..cols).filter(|&j| passive[j]).collect();
            let s = solve_on(m, g, &indices)?;
            if s.iter().all(|&v| v > 0.0) {
                u.fill(0.0);
                for (k, &j) in indices.iter().enumerate() {
                    u[j] = s[k];
                }
                set_aside.fill(false);
                break;
            }
            if first {
                let at = indices.iter().position(|&j| j == entering).unwrap();
                if s[at] <= 0.0 {
                    passive[entering] = false;
                    set_aside[entering] = true;
                    break;
                }
            }
            first = false;
            // Step from `u` toward `s` until the first passive variable
            // reaches zero, and move it (and any other at zero) out.
            let mut alpha = f64::INFINITY;
            let mut leaving = usize::MAX;
            for (k, &j) in indices.iter().enumerate() {
                if s[k] <= 0.0 {
                    let a = u[j] / (u[j] - s[k]);
                    if a < alpha {
                        alpha = a;
                        leaving = j;
                    }
                }
            }
            for (k, &j) in indices.iter().enumerate() {
                u[j] += alpha * (s[k] - u[j]);
            }
            for &j in &indices {
                if j == leaving || u[j] <= 0.0 {
                    u[j] = 0.0;
                    passive[j] = false;
                }
            }
        }
    }
    None
}

/// The least-squares solution of `M[:, indices]·s ≈ g`.
fn solve_on(m: &DMatrix<f64>, g: &DVector<f64>, indices: &[usize]) -> Option<Vec<f64>> {
    let sub = m.select_columns(indices);
    let svd = sub.svd(true, true);
    let largest = svd.singular_values.max();
    let s = svd.solve(g, largest * RANK_TOLERANCE).ok()?;
    Some(s.iter().copied().collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nnls_matches_a_hand_solved_problem() {
        // min ‖M·u − g‖ with u ≥ 0; the unconstrained solution has a negative
        // second component, so the constrained one sets it to zero and solves
        // for the first alone.
        let m = DMatrix::from_row_slice(3, 2, &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
        let g = DVector::from_row_slice(&[2.0, -1.0, 1.0]);
        let u = nnls(&m, &g).unwrap();
        assert!((u[0] - 1.5).abs() < 1e-12, "{u}");
        assert_eq!(u[1], 0.0);
    }

    #[test]
    fn least_distance_projects_onto_a_half_plane() {
        // min ‖z‖ subject to z₀ + z₁ ≥ 2: the point (1, 1), resting on the
        // one constraint; the second, z₀ ≥ −5, is slack.
        let e = DMatrix::from_row_slice(2, 2, &[1.0, 1.0, 1.0, 0.0]);
        let f = DVector::from_row_slice(&[2.0, -5.0]);
        let (z, active) = least_distance(&e, &f).unwrap();
        assert!(
            (z[0] - 1.0).abs() < 1e-12 && (z[1] - 1.0).abs() < 1e-12,
            "{z}"
        );
        assert_eq!(active, vec![true, false]);
    }

    #[test]
    fn least_distance_reports_infeasible_constraints() {
        // z₀ ≥ 1 and −z₀ ≥ 1 cannot both hold.
        let e = DMatrix::from_row_slice(2, 1, &[1.0, -1.0]);
        let f = DVector::from_row_slice(&[1.0, 1.0]);
        assert!(least_distance(&e, &f).is_none());
    }
}
