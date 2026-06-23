// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Kabsch algorithm for optimal rotation, translation, and scale alignment.
//!
//! Finds the similarity transform (rotation, translation, scale) that best
//! aligns source points to target points using SVD decomposition.

use nalgebra::{Matrix3, Vector3};

use crate::geometry::RotQuaternion;
use crate::geometry::Se3Transform;

/// Compute the optimal rotation, translation, and scale to align source points
/// to target points using the Kabsch algorithm.
///
/// Points are passed as flat slices of length `n_points * 3`, stored in
/// row-major order (x0, y0, z0, x1, y1, z1, ...).
///
/// Returns an [`Se3Transform`] or an error string on degenerate input.
pub fn kabsch_algorithm(
    source_points: &[f64],
    target_points: &[f64],
    n_points: usize,
) -> Result<Se3Transform, String> {
    if n_points < 1 {
        return Err("Need at least 1 point".to_string());
    }
    if source_points.len() != n_points * 3 || target_points.len() != n_points * 3 {
        return Err("Point slice length does not match n_points * 3".to_string());
    }

    let w = 1.0 / n_points as f64;

    // Compute centroids
    let mut src_c = Vector3::zeros();
    let mut tgt_c = Vector3::zeros();
    for i in 0..n_points {
        let base = i * 3;
        src_c += Vector3::new(
            source_points[base],
            source_points[base + 1],
            source_points[base + 2],
        );
        tgt_c += Vector3::new(
            target_points[base],
            target_points[base + 1],
            target_points[base + 2],
        );
    }
    src_c *= w;
    tgt_c *= w;

    // Compute cross-covariance matrix H = sum(w * (src - src_c) * (tgt - tgt_c)^T)
    let mut h = Matrix3::zeros();
    for i in 0..n_points {
        let base = i * 3;
        let s = Vector3::new(
            source_points[base] - src_c[0],
            source_points[base + 1] - src_c[1],
            source_points[base + 2] - src_c[2],
        );
        let t = Vector3::new(
            target_points[base] - tgt_c[0],
            target_points[base + 1] - tgt_c[1],
            target_points[base + 2] - tgt_c[2],
        );
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

    // Compute scale
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    for i in 0..n_points {
        let base = i * 3;
        let s = Vector3::new(
            source_points[base] - src_c[0],
            source_points[base + 1] - src_c[1],
            source_points[base + 2] - src_c[2],
        );
        let t = Vector3::new(
            target_points[base] - tgt_c[0],
            target_points[base + 1] - tgt_c[1],
            target_points[base + 2] - tgt_c[2],
        );
        let s_rot = rot * s;
        numerator += w * t.dot(&s_rot);
        denominator += w * s.dot(&s);
    }

    // When source points have zero variance (single point or all coincident),
    // scale is undetermined — default to 1.0.
    let scale = if denominator <= 0.0 {
        1.0
    } else {
        numerator / denominator
    };

    // translation = tgt_centroid - scale * R * src_centroid
    let translation = tgt_c - scale * (rot * src_c);

    let rotation = RotQuaternion::from_rotation_matrix(rot);
    Ok(Se3Transform::new(rotation, translation, scale))
}

#[cfg(test)]
mod tests;
