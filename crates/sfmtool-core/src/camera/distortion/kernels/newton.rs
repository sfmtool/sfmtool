// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The damped 2D Newton solve that inverts the two thin-prism fisheye models
//! in equidistant space.
//!
//! [`thin_prism`](super::thin_prism) and [`rad_tan`](super::rad_tan) differ
//! only in their forward map `F(uu, vv)` and its Jacobian; the iteration around
//! them (residual test, 2×2 solve, step-halving line search, iteration caps) is
//! this one function. Each model passes its own forward map and a combined
//! forward-and-Jacobian evaluation, so neither model's residual is merged into
//! the other.

use crate::camera::distortion::{UNDISTORT_EPS, UNDISTORT_MAX_ITER};

/// A 2×2 Jacobian `∂F/∂(uu, vv)`, row-major: `[[∂Fu/∂uu, ∂Fu/∂vv], [∂Fv/∂uu, ∂Fv/∂vv]]`.
pub(super) type Jacobian2 = [[f64; 2]; 2];

/// Solve `F(uu, vv) = (x_d, y_d)` by Newton's method from `(uu_init, vv_init)`.
///
/// `forward_and_jacobian` returns `F` and its Jacobian at a point; `forward`
/// returns `F` alone and is what the line search evaluates. The two must agree
/// bit for bit on `F`, since the line search compares its trial residual with
/// the one taken from `forward_and_jacobian`.
///
/// Each of at most `UNDISTORT_MAX_ITER` iterations stops when the residual's
/// L1 norm is below `UNDISTORT_EPS`, or when the Jacobian's determinant is
/// below `1e-30` in magnitude. Otherwise it takes the full Newton step, halved
/// up to ten times until the squared residual decreases; after the tenth
/// halving the step is taken whether or not it did.
pub(super) fn newton_2d(
    x_d: f64,
    y_d: f64,
    uu_init: f64,
    vv_init: f64,
    forward: impl Fn(f64, f64) -> (f64, f64),
    forward_and_jacobian: impl Fn(f64, f64) -> ((f64, f64), Jacobian2),
) -> (f64, f64) {
    let mut uu = uu_init;
    let mut vv = vv_init;
    for _ in 0..UNDISTORT_MAX_ITER {
        let ((fu, fv), [[j00, j01], [j10, j11]]) = forward_and_jacobian(uu, vv);

        // Residual: F(uu, vv) - (x_d, y_d)
        let res_u = fu - x_d;
        let res_v = fv - y_d;

        let res_norm = res_u * res_u + res_v * res_v;
        if res_u.abs() + res_v.abs() < UNDISTORT_EPS {
            break;
        }

        // Solve J * delta = residual via 2x2 inverse
        let det = j00 * j11 - j01 * j10;
        if det.abs() < 1e-30 {
            break;
        }
        let inv_det = 1.0 / det;
        let delta_uu = (j11 * res_u - j01 * res_v) * inv_det;
        let delta_vv = (-j10 * res_u + j00 * res_v) * inv_det;

        // Backtracking line search: halve the step until the residual decreases.
        let mut alpha = 1.0;
        for _ in 0..10 {
            let (fu_t, fv_t) = forward(uu - alpha * delta_uu, vv - alpha * delta_vv);
            let ru = fu_t - x_d;
            let rv = fv_t - y_d;
            if ru * ru + rv * rv < res_norm {
                break;
            }
            alpha *= 0.5;
        }

        uu -= alpha * delta_uu;
        vv -= alpha * delta_vv;
    }
    (uu, vv)
}
