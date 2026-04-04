// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Kabsch algorithm for optimal rotation, translation, and scale alignment.
//!
//! Finds the similarity transform (rotation, translation, scale) that best
//! aligns source points to target points using SVD decomposition.

use nalgebra::{Matrix3, Vector3};

use crate::rot_quaternion::RotQuaternion;
use crate::se3_transform::Se3Transform;

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
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_zero_points_is_error() {
        let result = kabsch_algorithm(&[], &[], 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_single_point_gives_translation() {
        // With a single point, rotation is identity, scale is 1, and
        // translation bridges the two points.
        let source = [1.0, 2.0, 3.0];
        let target = [4.0, 5.0, 6.0];
        let t = kabsch_algorithm(&source, &target, 1).unwrap();
        let rot = t.rotation.to_rotation_matrix();

        assert_relative_eq!(rot, Matrix3::identity(), epsilon = 1e-10);
        assert_relative_eq!(t.scale, 1.0, epsilon = 1e-10);
        assert_relative_eq!(t.translation, Vector3::new(3.0, 3.0, 3.0), epsilon = 1e-10);
    }

    #[test]
    fn test_coincident_source_points() {
        // All identical source points — rotation/scale undetermined,
        // defaults to identity rotation, scale 1, translation to target centroid.
        let source = [1.0, 2.0, 3.0, 1.0, 2.0, 3.0];
        let target = [4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let t = kabsch_algorithm(&source, &target, 2).unwrap();
        let rot = t.rotation.to_rotation_matrix();

        assert_relative_eq!(rot, Matrix3::identity(), epsilon = 1e-10);
        assert_relative_eq!(t.scale, 1.0, epsilon = 1e-10);
        // Target centroid is (5.5, 6.5, 7.5), source centroid is (1, 2, 3)
        assert_relative_eq!(t.translation, Vector3::new(4.5, 4.5, 4.5), epsilon = 1e-10);
    }

    #[test]
    fn test_collinear_points_identity() {
        // Collinear points mapped to themselves — should recover identity transform.
        let points = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let t = kabsch_algorithm(&points, &points, 3).unwrap();
        let rot = t.rotation.to_rotation_matrix();

        assert_relative_eq!(rot, Matrix3::identity(), epsilon = 1e-10);
        assert_relative_eq!(t.translation, Vector3::zeros(), epsilon = 1e-10);
        assert_relative_eq!(t.scale, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_collinear_points_with_transform() {
        // Collinear points with a known translation + scale (no rotation component
        // along the line).  The cross-axis rotation is unconstrained but the
        // transform should still map source to target correctly.
        let source = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0];
        let target: Vec<f64> = source.iter().map(|&x| x * 2.0 + 5.0).collect();

        let t = kabsch_algorithm(&source, &target, 3).unwrap();

        // Verify the transform maps source to target
        for i in 0..3 {
            let s = nalgebra::Point3::new(source[i * 3], source[i * 3 + 1], source[i * 3 + 2]);
            let expected = Vector3::new(target[i * 3], target[i * 3 + 1], target[i * 3 + 2]);
            let transformed = t.apply_to_point(&s);
            assert_relative_eq!(transformed.coords, expected, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_identity_alignment() {
        // Same non-collinear points should give identity transform
        let points = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let t = kabsch_algorithm(&points, &points, 4).unwrap();
        let rot = t.rotation.to_rotation_matrix();

        assert_relative_eq!(rot, Matrix3::identity(), epsilon = 1e-10);
        assert_relative_eq!(t.translation, Vector3::zeros(), epsilon = 1e-10);
        assert_relative_eq!(t.scale, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_known_rotation() {
        // 90-degree rotation around Z axis: (x, y, z) -> (-y, x, z)
        let source = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let target = [0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0];

        let t = kabsch_algorithm(&source, &target, 3).unwrap();
        let rot = t.rotation.to_rotation_matrix();

        assert_relative_eq!(t.scale, 1.0, epsilon = 1e-10);
        assert_relative_eq!(t.translation.norm(), 0.0, epsilon = 1e-10);

        // Check that rotation maps source to target
        for i in 0..3 {
            let s = Vector3::new(source[i * 3], source[i * 3 + 1], source[i * 3 + 2]);
            let expected = Vector3::new(target[i * 3], target[i * 3 + 1], target[i * 3 + 2]);
            let transformed = rot * s;
            assert_relative_eq!(transformed, expected, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_known_scale() {
        // Scale by factor of 2 with translation
        let source = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let target: Vec<f64> = source
            .chunks(3)
            .flat_map(|p| [p[0] * 2.0 + 1.0, p[1] * 2.0 + 2.0, p[2] * 2.0 + 3.0])
            .collect();

        let t = kabsch_algorithm(&source, &target, 4).unwrap();
        let rot = t.rotation.to_rotation_matrix();

        assert_relative_eq!(t.scale, 2.0, epsilon = 1e-10);
        assert_relative_eq!(rot, Matrix3::identity(), epsilon = 1e-10);
        assert_relative_eq!(t.translation, Vector3::new(1.0, 2.0, 3.0), epsilon = 1e-10);
    }

    #[test]
    fn test_random_similarity_round_trip() {
        use rand::rngs::StdRng;
        use rand::Rng;
        use rand::SeedableRng;

        let mut rng = StdRng::seed_from_u64(12345);
        let n_points = 20;

        for _ in 0..10 {
            // Random rotation via Rodrigues formula on a random axis/angle
            let axis = Vector3::new(
                rng.gen_range(-1.0..1.0),
                rng.gen_range(-1.0..1.0),
                rng.gen_range(-1.0..1.0),
            )
            .normalize();
            let angle: f64 = rng.gen_range(-std::f64::consts::PI..std::f64::consts::PI);
            let k = Matrix3::new(
                0.0, -axis[2], axis[1], axis[2], 0.0, -axis[0], -axis[1], axis[0], 0.0,
            );
            let expected_rot = Matrix3::identity() + angle.sin() * k + (1.0 - angle.cos()) * k * k;

            let expected_scale: f64 = rng.gen_range(0.1..10.0);
            let expected_trans = Vector3::new(
                rng.gen_range(-10.0..10.0),
                rng.gen_range(-10.0..10.0),
                rng.gen_range(-10.0..10.0),
            );

            // Generate random 3D source points
            let mut source = vec![0.0f64; n_points * 3];
            for v in source.iter_mut() {
                *v = rng.gen_range(-5.0..5.0);
            }

            // Apply similarity transform: target = scale * R * source + translation
            let mut target = vec![0.0f64; n_points * 3];
            for i in 0..n_points {
                let s = Vector3::new(source[i * 3], source[i * 3 + 1], source[i * 3 + 2]);
                let t = expected_scale * (expected_rot * s) + expected_trans;
                target[i * 3] = t[0];
                target[i * 3 + 1] = t[1];
                target[i * 3 + 2] = t[2];
            }

            let result = kabsch_algorithm(&source, &target, n_points).unwrap();
            let rot = result.rotation.to_rotation_matrix();

            assert_relative_eq!(result.scale, expected_scale, epsilon = 1e-8);
            assert_relative_eq!(rot, expected_rot, epsilon = 1e-8);
            assert_relative_eq!(result.translation, expected_trans, epsilon = 1e-8);
        }
    }
}