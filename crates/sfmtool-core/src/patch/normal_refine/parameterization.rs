// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Sphere parameterization for the normal search: a deterministic tangent basis
//! and the angle-uniform exponential map used to walk candidate normals.

use nalgebra::Vector3;

/// Deterministic orthonormal tangent basis of the unit normal `n` — a pure
/// function of `n` (least-aligned world axis + Gram-Schmidt), so refinements
/// are reproducible.
pub fn tangent_basis(n: &Vector3<f64>) -> (Vector3<f64>, Vector3<f64>) {
    let n = n.normalize();
    let (ax, ay, az) = (n.x.abs(), n.y.abs(), n.z.abs());
    let a = if ax <= ay && ax <= az {
        Vector3::x()
    } else if ay <= az {
        Vector3::y()
    } else {
        Vector3::z()
    };
    let u = (a - n * a.dot(&n)).normalize();
    let v = n.cross(&u);
    (u, v)
}

/// Exponential map on the sphere: tilt the unit normal `n0` by angle `‖δ‖`
/// toward the tangent direction `δ` (expressed in [`tangent_basis`]`(n0)`):
/// `n(δ) = cos‖δ‖·n₀ + sin‖δ‖·δ̂`. Angle-uniform — equal steps in `δ` are
/// equal angles.
pub fn exp_map_normal(n0: &Vector3<f64>, delta: [f64; 2]) -> Vector3<f64> {
    let n = n0.normalize();
    let (u, v) = tangent_basis(&n);
    exp_map_in_basis(&n, &u, &v, delta)
}

pub(super) fn exp_map_in_basis(
    n0: &Vector3<f64>,
    u: &Vector3<f64>,
    v: &Vector3<f64>,
    delta: [f64; 2],
) -> Vector3<f64> {
    let theta = delta[0].hypot(delta[1]);
    if theta < 1e-12 {
        return *n0;
    }
    let dir = (u * delta[0] + v * delta[1]) / theta;
    n0 * theta.cos() + dir * theta.sin()
}
