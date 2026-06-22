// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Batch triangulation with observability diagnostics.
//!
//! The least-squares closest point to a track's observation rays (the midpoint
//! estimate) is the solution of `A p = b` with
//! `A = Σ(I − dᵢdᵢᵀ)` and `b = Σ(I − dᵢdᵢᵀ)cᵢ`, where `dᵢ` is the unit
//! world-space ray of observation `i` and `cᵢ` its camera center. The solve
//! already reveals how well the depth is observed: the spectrum of `A` is `0`
//! along a direction with no parallax and grows with the spread of the rays.
//! [`triangulate_batch`] returns that spectrum (and the condition number) for
//! free alongside the point, and [`depth_uncertainty_batch`] layers the
//! noise-calibrated depth uncertainty on top when a caller supplies a per-ray
//! angular noise model.
//!
//! Tracks are flattened CSR-style, the same shape as the reconstruction's
//! `observation_offsets`: track `t` owns `dirs[offsets[t]..offsets[t+1]]` and
//! the matching `centers`. The functions are pure and IO-free — building the
//! rays (un-projection, distortion, pose) is the caller's concern.
//!
//! See `specs/core/batch-triangulation-api.md` for the design.

use nalgebra::{Matrix3, Point3, SymmetricEigen, Vector3};

/// One track's triangulation and the observability diagnostics the linear solve
/// computes alongside the point. Geometric fields are always populated.
#[derive(Debug, Clone, Copy)]
pub struct Triangulation {
    /// Least-squares closest point to the rays (the midpoint estimate). When
    /// the rays are exactly parallel (depth unobservable) this is the
    /// minimum-norm solution in the observable subspace; `condition_number`
    /// and `eigenvalues` flag that case.
    pub point: Point3<f64>,
    /// Eigenvalues of `A = Σ(I − dᵢdᵢᵀ)`, ascending. `eigenvalues[0] → 0` marks
    /// parallel rays (depth unobservable). `Σ eigenvalues = 2·K`.
    pub eigenvalues: [f64; 3],
    /// Condition number `λ_max / λ_min` of `A` (`∞` when exactly degenerate). A
    /// cheap geometric indicator — but note it scales with track length `K`, so
    /// it is a proxy, not the decision variable (see [`depth_uncertainty_batch`]).
    pub condition_number: f64,
    /// `point` has positive depth in every observing camera (lies in front of
    /// each, not behind). False means the least-squares point landed behind a
    /// camera, so the finite position is non-physical. Also false for an empty
    /// track.
    pub in_front_of_all_cameras: bool,
}

/// Depth uncertainty along the mean viewing direction, from the inverse-
/// variance-weighted normal matrix.
#[derive(Debug, Clone, Copy)]
pub struct DepthUncertainty {
    /// Depth of the point along the mean viewing direction, measured from the
    /// camera-cloud centroid.
    pub depth: f64,
    /// 1σ depth uncertainty along the mean viewing direction (`∞` when the
    /// depth is unobservable).
    pub sigma: f64,
    /// Inverse-depth z-score `depth / sigma`. Small (≲ 3-4) ⇒ statistically
    /// indistinguishable from infinity. Scale-free; the finite-vs-∞ test, but
    /// reliable only on a non-degenerate solve (it divides by the solved depth,
    /// which is noise when the rays are near-parallel).
    pub inverse_depth_z: f64,
    /// Farthest depth this track's geometry can tell from infinity:
    /// `B⊥ / σ_ray`, the perpendicular camera baseline over the RMS per-ray
    /// angular noise — equivalently the depth at which `inverse_depth_z` would
    /// fall to 1. Computed from the camera geometry and noise alone, *not* the
    /// solved point, so it stays meaningful where `inverse_depth_z` goes
    /// unstable. Gate the finite-vs-∞ decision on this against a policy
    /// `finite_horizon` (default: the camera extents).
    pub resolvable_distance: f64,
}

/// Relative tolerance for treating an eigenvalue as zero (degenerate direction).
const EIGENVALUE_REL_TOL: f64 = 1e-12;

/// Triangulate a batch of tracks. `dirs` are unit world-space rays; `centers`
/// the matching camera centers (`dirs.len() == centers.len()`); `offsets`
/// (len `M+1`) delimits the `M` tracks CSR-style.
///
/// The cost is `O(K)` to assemble `A` per track plus one fixed 3×3 symmetric
/// eigensolve, so the whole-reconstruction cost is just that summed over tracks.
pub fn triangulate_batch(
    dirs: &[Vector3<f64>],
    centers: &[Point3<f64>],
    offsets: &[usize],
) -> Vec<Triangulation> {
    assert_eq!(
        dirs.len(),
        centers.len(),
        "dirs and centers must have equal length"
    );
    let m = offsets.len().saturating_sub(1);
    let identity = Matrix3::<f64>::identity();
    let mut out = Vec::with_capacity(m);

    for t in 0..m {
        let start = offsets[t];
        let end = offsets[t + 1];

        // Normal matrix A = Σ(I − dᵢdᵢᵀ) and rhs b = Σ(I − dᵢdᵢᵀ)cᵢ.
        let mut a = Matrix3::<f64>::zeros();
        let mut b = Vector3::<f64>::zeros();
        for i in start..end {
            let d = dirs[i];
            let proj = identity - d * d.transpose();
            a += proj;
            b += proj * centers[i].coords;
        }

        let eig = SymmetricEigen::new(a);

        // Sort the eigenpairs ascending. A is PSD, so clamp tiny negative
        // round-off to zero for the reported spectrum.
        let mut order = [0usize, 1, 2];
        order.sort_by(|&i, &j| {
            eig.eigenvalues[i]
                .partial_cmp(&eig.eigenvalues[j])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let eigenvalues = [
            eig.eigenvalues[order[0]].max(0.0),
            eig.eigenvalues[order[1]].max(0.0),
            eig.eigenvalues[order[2]].max(0.0),
        ];

        // Minimum-norm least-squares point via the eigendecomposition: invert
        // only the observable directions (λ above the relative tolerance).
        let lambda_max = eigenvalues[2];
        let tol = lambda_max * EIGENVALUE_REL_TOL;
        let mut p = Vector3::<f64>::zeros();
        for &k in &order {
            let lambda = eig.eigenvalues[k];
            if lambda > tol {
                let v = eig.eigenvectors.column(k).into_owned();
                p += (v.dot(&b) / lambda) * v;
            }
        }
        let point = Point3::from(p);

        let condition_number = if eigenvalues[0] > 0.0 {
            eigenvalues[2] / eigenvalues[0]
        } else {
            f64::INFINITY
        };

        // The point must lie in front of every observing camera: the ray from
        // the camera to the point agrees with the un-projected direction.
        let mut in_front_of_all_cameras = end > start;
        for i in start..end {
            if (point.coords - centers[i].coords).dot(&dirs[i]) <= 0.0 {
                in_front_of_all_cameras = false;
                break;
            }
        }

        out.push(Triangulation {
            point,
            eigenvalues,
            condition_number,
            in_front_of_all_cameras,
        });
    }

    out
}

/// Depth uncertainty along the mean viewing direction for each track.
///
/// `sigma_rad` is the per-ray angular noise (e.g. `noise_px / fᵢ`), indexed the
/// same as `dirs`/`centers`. The weighted normal matrix
/// `M = Σ (1/(ρᵢσᵢ)²)(I − dᵢdᵢᵀ)` (with range `ρᵢ = ‖point − cᵢ‖`) is the
/// Fisher information of the point; its inverse along the mean viewing
/// direction gives the depth variance. `tris` are the corresponding
/// [`triangulate_batch`] results, supplying each point and providing the same
/// CSR layout via `offsets`.
pub fn depth_uncertainty_batch(
    tris: &[Triangulation],
    dirs: &[Vector3<f64>],
    centers: &[Point3<f64>],
    offsets: &[usize],
    sigma_rad: &[f64],
) -> Vec<DepthUncertainty> {
    assert_eq!(
        dirs.len(),
        centers.len(),
        "dirs and centers must have equal length"
    );
    assert_eq!(
        sigma_rad.len(),
        dirs.len(),
        "sigma_rad must have one entry per ray"
    );
    let m = offsets.len().saturating_sub(1);
    assert_eq!(tris.len(), m, "tris must have one entry per track");
    let identity = Matrix3::<f64>::identity();
    let mut out = Vec::with_capacity(m);

    for t in 0..m {
        let start = offsets[t];
        let end = offsets[t + 1];
        let p = tris[t].point.coords;

        // Mean viewing direction and camera-cloud centroid for this track.
        let mut mean = Vector3::<f64>::zeros();
        let mut centroid = Vector3::<f64>::zeros();
        let mut m_mat = Matrix3::<f64>::zeros();
        for i in start..end {
            let d = dirs[i];
            mean += d;
            centroid += centers[i].coords;
            let rho = (p - centers[i].coords).norm();
            let denom = rho * sigma_rad[i];
            let w = if denom > 0.0 {
                1.0 / (denom * denom)
            } else {
                0.0
            };
            m_mat += w * (identity - d * d.transpose());
        }
        let count = (end - start) as f64;
        let mean_norm = mean.norm();
        let m_dir = if mean_norm > 0.0 {
            mean / mean_norm
        } else if end > start {
            dirs[start]
        } else {
            Vector3::z()
        };
        let centroid = if count > 0.0 {
            centroid / count
        } else {
            Vector3::zeros()
        };
        let depth = (p - centroid).dot(&m_dir);

        // Resolvable distance D_max = B⊥ / σ_ray — the farthest depth this
        // track's geometry can tell from infinity, independent of the (possibly
        // degenerate) solved point. B⊥ is the camera spread perpendicular to the
        // mean viewing direction (2·RMS offset, matching the 2-view baseline);
        // σ_ray is the RMS per-ray angular noise.
        let mut perp_sq_sum = 0.0_f64;
        let mut sigma_sq_sum = 0.0_f64;
        for i in start..end {
            let off = centers[i].coords - centroid;
            let perp = off - off.dot(&m_dir) * m_dir;
            perp_sq_sum += perp.norm_squared();
            sigma_sq_sum += sigma_rad[i] * sigma_rad[i];
        }
        let resolvable_distance = if count > 0.0 {
            let b_perp = 2.0 * (perp_sq_sum / count).sqrt();
            let sigma_ray = (sigma_sq_sum / count).sqrt();
            if sigma_ray > 0.0 {
                b_perp / sigma_ray
            } else {
                f64::INFINITY
            }
        } else {
            0.0
        };

        // Variance along m_dir is mᵀ M⁻¹ m = Σ (m·vⱼ)² / λⱼ. A near-zero λ with
        // non-trivial projection blows the variance up — the depth is
        // unobservable (infinity).
        let eig = SymmetricEigen::new(m_mat);
        let lambda_max = eig.eigenvalues.iter().cloned().fold(0.0_f64, f64::max);
        let tol = lambda_max * EIGENVALUE_REL_TOL;
        let mut variance = 0.0_f64;
        for k in 0..3 {
            let lambda = eig.eigenvalues[k];
            let proj = eig.eigenvectors.column(k).dot(&m_dir);
            let proj_sq = proj * proj;
            if lambda > tol {
                variance += proj_sq / lambda;
            } else if proj_sq > 1e-18 {
                variance = f64::INFINITY;
                break;
            }
        }
        let sigma = variance.sqrt();
        let inverse_depth_z = if sigma > 0.0 && sigma.is_finite() {
            depth / sigma
        } else if sigma == 0.0 {
            // Depth perfectly determined (no noise) — treat as fully finite.
            f64::INFINITY
        } else {
            // sigma is infinite: depth indistinguishable from infinity.
            0.0
        };

        out.push(DepthUncertainty {
            depth,
            sigma,
            inverse_depth_z,
            resolvable_distance,
        });
    }

    out
}

#[cfg(test)]
mod tests;
