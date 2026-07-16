// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Least-squares alignment of two corresponded point sets.
//!
//! Estimates the similarity (or rigid) transform — rotation, translation, and
//! optional uniform scale — that best maps source points onto their target
//! counterparts, via an SVD of the cross-covariance (the classical
//! Kabsch/Umeyama construction). Optional trimming iteratively refits on the
//! best-fitting fraction of correspondences to reject gross mismatches.

use nalgebra::{Matrix3, Vector3};

use crate::geometry::RotQuaternion;
use crate::geometry::Se3Transform;

/// Parameters for [`estimate_alignment`].
///
/// [`Default`] is a single-shot similarity fit over every correspondence
/// (`rounds = 1`, `keep_fraction = 1.0`, `estimate_scale = true`).
#[derive(Clone, Copy, Debug)]
pub struct AlignmentParams {
    /// Refit iterations. Each round after the first re-selects the inlier
    /// subset from the smallest-residual correspondences; `1` disables trimming.
    pub rounds: usize,
    /// Fraction of correspondences retained each round after the first.
    /// Ignored when `rounds == 1`.
    pub keep_fraction: f64,
    /// Fit a similarity (scale + rotation + translation) when `true`, or a rigid
    /// transform (scale fixed at 1.0) when `false`.
    pub estimate_scale: bool,
}

impl Default for AlignmentParams {
    fn default() -> Self {
        Self {
            rounds: 1,
            keep_fraction: 1.0,
            estimate_scale: true,
        }
    }
}

/// Estimate the similarity-or-rigid transform aligning `source_points` to
/// `target_points` by least squares over corresponded points.
///
/// Points are flat slices of length `n_points * 3`, row-major
/// (`x0, y0, z0, x1, y1, z1, …`). With the default [`AlignmentParams`] this is
/// the classical single-shot similarity fit; raising `rounds` above 1 trims
/// gross mismatches by iteratively refitting on the `keep_fraction` of
/// correspondences with the smallest residual `‖s·R·src + t − tgt‖` under the
/// previous fit.
///
/// Returns an [`Se3Transform`] or an error string on degenerate input
/// (`n_points < 1`, slice length ≠ `n_points * 3`, or SVD failure).
pub fn estimate_alignment(
    source_points: &[f64],
    target_points: &[f64],
    n_points: usize,
    params: AlignmentParams,
) -> Result<Se3Transform, String> {
    if n_points < 1 {
        return Err("Need at least 1 point".to_string());
    }
    if source_points.len() != n_points * 3 || target_points.len() != n_points * 3 {
        return Err("Point slice length does not match n_points * 3".to_string());
    }
    let keep = (params.keep_fraction.clamp(0.0, 1.0) * n_points as f64).round() as usize;
    let keep = keep.max(1).min(n_points);

    let mut indices: Vec<usize> = (0..n_points).collect();
    let mut transform = fit_indexed(
        source_points,
        target_points,
        &indices,
        params.estimate_scale,
    )?;
    for _ in 0..params.rounds.saturating_sub(1) {
        // Residual of every correspondence under the current transform.
        let mut resid: Vec<(f64, usize)> = (0..n_points)
            .map(|i| {
                let s = point_at(source_points, i);
                let mapped = transform.scale * (transform.rotation.to_rotation_matrix() * s)
                    + transform.translation;
                let d = (mapped - point_at(target_points, i)).norm();
                (d, i)
            })
            .collect();
        resid.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        indices = resid[..keep].iter().map(|&(_, i)| i).collect();
        indices.sort_unstable();
        transform = fit_indexed(
            source_points,
            target_points,
            &indices,
            params.estimate_scale,
        )?;
    }
    Ok(transform)
}

/// Point at row `i` of a flat `[x0,y0,z0, x1,y1,z1, ...]` slice.
fn point_at(points: &[f64], i: usize) -> Vector3<f64> {
    let base = i * 3;
    Vector3::new(points[base], points[base + 1], points[base + 2])
}

/// SVD similarity/rigid fit over the observations in `indices` (a subset of the
/// points). `estimate_scale = false` yields a rigid transform (scale 1).
///
/// This is the single SVD implementation underlying [`estimate_alignment`].
fn fit_indexed(
    source_points: &[f64],
    target_points: &[f64],
    indices: &[usize],
    estimate_scale: bool,
) -> Result<Se3Transform, String> {
    let n = indices.len();
    if n < 1 {
        return Err("Need at least 1 point".to_string());
    }
    let w = 1.0 / n as f64;

    // Compute centroids
    let mut src_c = Vector3::zeros();
    let mut tgt_c = Vector3::zeros();
    for &i in indices {
        src_c += point_at(source_points, i);
        tgt_c += point_at(target_points, i);
    }
    src_c *= w;
    tgt_c *= w;

    // Compute cross-covariance matrix H = sum(w * (src - src_c) * (tgt - tgt_c)^T)
    let mut h = Matrix3::zeros();
    for &i in indices {
        let s = point_at(source_points, i) - src_c;
        let t = point_at(target_points, i) - tgt_c;
        // H += w * s * t^T  (equivalent to weighted_source.T @ weighted_target with uniform weights)
        h += w * s * t.transpose();
    }

    // SVD
    let svd = h.svd(true, true);
    let u = svd.u.ok_or("SVD failed to compute U")?;
    let mut vt = svd.v_t.ok_or("SVD failed to compute V^T")?;

    // For rank-deficient H, SVD may pick arbitrary orthonormal completions
    // for nullspace columns of U and V.  We only need to fix these when the
    // rotation is genuinely unconstrained:
    //
    //   Rank 0: all unconstrained → R = I
    //   Rank 1: one axis constrained (collinear) → force the two nullspace
    //           columns of V to match U so the cross-axis rotation is identity
    //   Rank 2+: fully determined by SVD + det correction (the third axis is
    //            fixed by orthonormality), so don't touch it
    let sv = &svd.singular_values;
    let max_sv = sv[0].max(sv[1]).max(sv[2]);
    let rank = if max_sv > 0.0 {
        sv.iter().filter(|&&s| s / max_sv > 1e-10).count()
    } else {
        0
    };
    if rank == 0 {
        vt = u.transpose();
    } else if rank == 1 {
        for i in 0..3 {
            if sv[i] / max_sv < 1e-10 {
                for j in 0..3 {
                    vt[(i, j)] = u[(j, i)];
                }
            }
        }
    }

    // R = V * U^T
    let mut rot = vt.transpose() * u.transpose();

    // Ensure proper rotation (det = +1)
    if rot.determinant() < 0.0 {
        let mut vt_fixed = vt;
        vt_fixed.row_mut(2).scale_mut(-1.0);
        rot = vt_fixed.transpose() * u.transpose();
    }

    // Compute scale (skipped for a rigid fit, where scale is fixed at 1.0)
    let scale = if !estimate_scale {
        1.0
    } else {
        let mut numerator = 0.0;
        let mut denominator = 0.0;
        for &i in indices {
            let s = point_at(source_points, i) - src_c;
            let t = point_at(target_points, i) - tgt_c;
            let s_rot = rot * s;
            numerator += w * t.dot(&s_rot);
            denominator += w * s.dot(&s);
        }
        // When source points have zero variance (single point or all
        // coincident), scale is undetermined — default to 1.0.
        if denominator <= 0.0 {
            1.0
        } else {
            numerator / denominator
        }
    };

    // translation = tgt_centroid - scale * R * src_centroid
    let translation = tgt_c - scale * (rot * src_c);

    let rotation = RotQuaternion::from_rotation_matrix(rot);
    Ok(Se3Transform::new(rotation, translation, scale))
}

#[cfg(test)]
mod tests;
