// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The ray Jacobian that the equidistant-family fisheye models share.
//!
//! [`equidistant`](super::equidistant)'s `radial_fisheye_ray_jacobian`
//! (`θ_d = θ·(1 + k1·θ²)`) and [`sfmtool_fisheye`](super::sfmtool_fisheye)'s
//! `sfmtool_fisheye_ray_jacobian` (`θ_d = θ + δ(θ)`) differ only in the radial
//! map `θ ↦ (θ_d, θ_d')`. The chain rule through `θ = atan2(ρ, rz)` and the unit
//! direction `(rx, ry)/ρ`, the on-axis limit and the antipode and fold gates
//! are [`radial_ray_jacobian`], and each model passes its own radial map.

/// Angular width, relative to the ray norm, of the on-axis band where the
/// 2D direction `(rx, ry)/r_xy` is numerically meaningless and the Jacobian
/// is evaluated from its axis limit instead.
pub(in crate::camera::distortion) const EQUIDISTANT_AXIS_EPS: f64 = 1e-12;

/// Distorted normalized coordinate `(x_d, y_d)` paired with the 2×3
/// `∂(x_d, y_d)/∂(rx, ry, rz)`, row-major — the pre-intrinsics half of a
/// [`PixelJacobian`](crate::camera::distortion::PixelJacobian), in the optical frame.
pub(in crate::camera::distortion) type NormalizedRayJacobian = ((f64, f64), [[f64; 3]; 2]);

/// Distorted coordinate and the analytic `∂(x_d, y_d)/∂(rx, ry, rz)` of a
/// radial fisheye map, row-major, all in the optical frame.
///
/// The map sends a ray at incidence angle `θ` to `θ_d(θ)` times the unit 2D
/// direction. `theta_d_and_deriv(θ)` returns `(θ_d, θ_d')` with
/// `θ_d' = dθ_d/dθ`; it is called once, and only off the axis.
///
/// With `ρ = r_xy`, `n² = ρ² + rz²`, unit direction `(ux, uy) = (rx, ry)/ρ` and
/// `θ = atan2(ρ, rz)`:
///
/// ```text
/// ∂θ/∂rx = ux·rz/n²   ∂θ/∂ry = uy·rz/n²   ∂θ/∂rz = −ρ/n²
/// ∂ux/∂rx = uy²/ρ     ∂ux/∂ry = −ux·uy/ρ  (and the mirror for uy)
/// ```
///
/// so, chaining `x_d = θ_d(θ)·ux` and writing `c = θ_d'·rz/n² − θ_d/ρ` for
/// the shared off-diagonal factor,
///
/// ```text
/// ∂x_d/∂rx = θ_d·uy²/ρ + θ_d'·ux²·rz/n²   ∂x_d/∂ry = ux·uy·c
///                                         ∂x_d/∂rz = −θ_d'·rx/n²
/// ∂y_d/∂rx = ux·uy·c                      ∂y_d/∂ry = θ_d·ux²/ρ + θ_d'·uy²·rz/n²
///                                         ∂y_d/∂rz = −θ_d'·ry/n²
/// ```
///
/// Nothing here is guarded on `rz`: the expressions are finite and correct
/// past 90°, which is the whole point of a fisheye-native derivative.
///
/// Two limits, for a radial map with `θ_d(0) = 0` and `θ_d'(0) = 1`:
///
/// - **On axis, in front** (`ρ → 0`, `rz > 0`): `θ_d/ρ → 1/rz` and
///   `θ_d' → 1`, so the off-diagonal factor `c → 0` and the third column
///   vanishes, leaving `diag(1/rz, 1/rz)` — the pinhole small-angle
///   Jacobian, independent of the radial map and of the direction `(ux, uy)`
///   that is undefined there. Within [`EQUIDISTANT_AXIS_EPS`] of the axis that
///   limit is returned directly.
/// - **At the antipode** (`ρ → 0`, `rz < 0`): `θ → π` while `ρ → 0`, so
///   `θ_d/ρ` diverges and no finite Jacobian exists. Returns `None`.
///
/// `None` also past the fold, where `θ > 0` but `θ_d ≤ 0`: the forward maps
/// reject those rays, so there is no projection to differentiate.
#[inline]
pub(in crate::camera::distortion) fn radial_ray_jacobian(
    rx: f64,
    ry: f64,
    rz: f64,
    theta_d_and_deriv: impl FnOnce(f64) -> (f64, f64),
) -> Option<NormalizedRayJacobian> {
    let rho2 = rx * rx + ry * ry;
    let rho = rho2.sqrt();
    let n2 = rho2 + rz * rz;
    if n2 == 0.0 {
        return None;
    }
    if rho <= EQUIDISTANT_AXIS_EPS * n2.sqrt() {
        // On the optical axis: only the forward limit is finite.
        if rz <= 0.0 {
            return None;
        }
        let inv = 1.0 / rz;
        return Some(((0.0, 0.0), [[inv, 0.0, 0.0], [0.0, inv, 0.0]]));
    }
    let theta = rho.atan2(rz);
    let (theta_d, dtheta_d) = theta_d_and_deriv(theta);
    if theta > 0.0 && theta_d <= 0.0 {
        return None;
    }
    let (ux, uy) = (rx / rho, ry / rho);
    let rz_n2 = rz / n2;
    let theta_rho = theta_d / rho;
    let cross = ux * uy * (dtheta_d * rz_n2 - theta_rho);
    Some((
        (theta_d * ux, theta_d * uy),
        [
            [
                theta_rho * uy * uy + dtheta_d * (ux * ux * rz_n2),
                cross,
                -dtheta_d * rx / n2,
            ],
            [
                cross,
                theta_rho * ux * ux + dtheta_d * (uy * uy * rz_n2),
                -dtheta_d * ry / n2,
            ],
        ],
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A radial map with no special structure, `θ_d = θ + 0.1·sin(θ)·θ²`,
    /// and its derivative.
    fn wobble(theta: f64) -> (f64, f64) {
        let s = theta.sin();
        (
            theta + 0.1 * s * theta * theta,
            1.0 + 0.1 * (theta.cos() * theta * theta + 2.0 * s * theta),
        )
    }

    /// The forward map written independently of [`radial_ray_jacobian`]:
    /// `θ_d(θ)` times the unit 2D direction.
    fn forward(r: [f64; 3]) -> (f64, f64) {
        let rho = (r[0] * r[0] + r[1] * r[1]).sqrt();
        let (theta_d, _) = wobble(rho.atan2(r[2]));
        (theta_d * r[0] / rho, theta_d * r[1] / rho)
    }

    /// The analytic Jacobian and the distorted coordinate agree with a
    /// central difference of the forward map, at incidence angles from near
    /// the axis to past 90° and at several ray lengths.
    #[test]
    fn radial_ray_jacobian_matches_central_difference() {
        let h = 1e-6;
        let mut checked = 0;
        for deg in [0.5_f64, 10.0, 45.0, 89.0, 91.0, 120.0, 170.0] {
            for az_deg in [0.0_f64, 30.0, 135.0, 250.0] {
                for len in [0.3_f64, 1.0, 4.0] {
                    let (t, a) = (deg.to_radians(), az_deg.to_radians());
                    let ray = [
                        len * t.sin() * a.cos(),
                        len * t.sin() * a.sin(),
                        len * t.cos(),
                    ];
                    let (xd, jac) =
                        radial_ray_jacobian(ray[0], ray[1], ray[2], wobble).expect("in domain");
                    let f0 = forward(ray);
                    assert!((xd.0 - f0.0).abs() < 1e-12 && (xd.1 - f0.1).abs() < 1e-12);
                    for c in 0..3 {
                        let (mut rp, mut rm) = (ray, ray);
                        rp[c] += h;
                        rm[c] -= h;
                        let (fp, fm) = (forward(rp), forward(rm));
                        let fd = [(fp.0 - fm.0) / (2.0 * h), (fp.1 - fm.1) / (2.0 * h)];
                        for row in 0..2 {
                            assert!(
                                (jac[row][c] - fd[row]).abs() <= 1e-6 * (1.0 + fd[row].abs()),
                                "θ={deg}° az={az_deg}° len={len} [{row}][{c}]: \
                                 analytic {} vs central-diff {}",
                                jac[row][c],
                                fd[row],
                            );
                        }
                        checked += 1;
                    }
                }
            }
        }
        assert_eq!(checked, 7 * 4 * 3 * 3);
    }

    /// On the axis in front the Jacobian is `diag(1/rz, 1/rz)` without calling
    /// the radial map; at the antipode, at the origin and past the fold there
    /// is none.
    #[test]
    fn radial_ray_jacobian_limits_and_gates() {
        let unused = |_: f64| -> (f64, f64) { panic!("radial map called on the axis") };
        let (xd, jac) = radial_ray_jacobian(0.0, 0.0, 2.0, unused).unwrap();
        assert_eq!(xd, (0.0, 0.0));
        assert_eq!(jac, [[0.5, 0.0, 0.0], [0.0, 0.5, 0.0]]);
        assert!(radial_ray_jacobian(0.0, 0.0, -2.0, unused).is_none());
        assert!(radial_ray_jacobian(0.0, 0.0, 0.0, unused).is_none());
        // A radial map folded non-positive at this angle.
        assert!(radial_ray_jacobian(1.0, 0.0, 1.0, |t| (-t, -1.0)).is_none());
    }
}
