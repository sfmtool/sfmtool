// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Two-stage geometric filtering for sort-and-sweep feature matching.
//!
//! This module implements the two-stage geometric filtering approach:
//!
//! **Stage 1 (Always):** Orientation check using rotation (well-conditioned)
//! **Stage 2 (Conditional):** Size check using depth ratio (only when triangulation is reliable)
//!
//! This provides motion-adaptive filtering that gracefully handles all camera motion types:
//! - Forward motion (parallel rays) → orientation check only
//! - Lateral motion (good angles) → orientation + size check

use nalgebra::{Matrix3, Matrix4, Vector3, Vector4};

/// Configuration for two-stage geometric filtering.
///
/// # Fields
///
/// * `max_angle_difference` – Maximum allowed angle difference in degrees (Stage 1).
/// * `min_triangulation_angle` – Minimum ray angle in degrees for reliable triangulation (Stage 2).
/// * `geometric_size_ratio_min` – Minimum allowed size ratio (Stage 2).
/// * `geometric_size_ratio_max` – Maximum allowed size ratio (Stage 2).
#[derive(Debug, Clone)]
pub struct GeometricFilterConfig {
    /// Maximum allowed angle difference in degrees (Stage 1).
    pub max_angle_difference: f64,
    /// Minimum ray angle in degrees for reliable triangulation (Stage 2).
    pub min_triangulation_angle: f64,
    /// Minimum allowed size ratio (Stage 2).
    pub geometric_size_ratio_min: f64,
    /// Maximum allowed size ratio (Stage 2).
    pub geometric_size_ratio_max: f64,
}

impl Default for GeometricFilterConfig {
    fn default() -> Self {
        Self {
            max_angle_difference: 15.0,
            min_triangulation_angle: 5.0,
            geometric_size_ratio_min: 0.8,
            geometric_size_ratio_max: 1.25,
        }
    }
}

/// Intrinsic and extrinsic parameters for a single camera, with precomputed
/// derived quantities (inverse intrinsics, world-frame rotation, camera center).
#[derive(Debug, Clone)]
pub struct CameraParams {
    /// 3×3 intrinsic matrix K.
    pub k: Matrix3<f64>,
    /// 3×3 cam_from_world rotation R.
    pub r: Matrix3<f64>,
    /// cam_from_world translation t.
    pub t: Vector3<f64>,
    /// Inverse intrinsic matrix K⁻¹.
    pub k_inv: Matrix3<f64>,
    /// Camera center in world coordinates: C = −R^T t.
    pub center: Vector3<f64>,
    /// Rotation to world coordinates: R^T.
    pub r_world: Matrix3<f64>,
}

impl CameraParams {
    /// Construct camera parameters and precompute derived quantities.
    pub fn new(k: &Matrix3<f64>, r: &Matrix3<f64>, t: &Vector3<f64>) -> Self {
        let k_inv = k
            .try_inverse()
            .expect("Intrinsic matrix K must be invertible");
        let r_world = r.transpose();
        let center = -(r_world * t);
        Self {
            k: *k,
            r: *r,
            t: *t,
            k_inv,
            center,
            r_world,
        }
    }
}

/// Precomputed geometry for a stereo image pair.
///
/// Composed of two [`CameraParams`] plus a 2×2 in-plane rotation matrix
/// derived from the relative rotation between the cameras.
#[derive(Debug, Clone)]
pub struct StereoPairGeometry {
    /// Camera 1 (the "query" camera in forward matching).
    pub cam1: CameraParams,
    /// Camera 2 (the "target" camera in forward matching).
    pub cam2: CameraParams,
    /// 2×2 in-plane rotation matrix (upper-left 2×2 of R2 @ R1^T), row-major `[a00, a01, a10, a11]`.
    pub r_2d: [f64; 4],
}

impl StereoPairGeometry {
    /// Construct stereo pair geometry from individual camera parameters.
    ///
    /// # Parameters
    ///
    /// * `k1`, `k2` – 3×3 intrinsic matrices.
    /// * `r1`, `r2` – 3×3 cam_from_world rotation matrices.
    /// * `t1`, `t2` – cam_from_world translation vectors.
    pub fn new(
        k1: &Matrix3<f64>,
        k2: &Matrix3<f64>,
        r1: &Matrix3<f64>,
        r2: &Matrix3<f64>,
        t1: &Vector3<f64>,
        t2: &Vector3<f64>,
    ) -> Self {
        let cam1 = CameraParams::new(k1, r1, t1);
        let cam2 = CameraParams::new(k2, r2, t2);

        // R_rel = R2 @ R1^T, extract upper-left 2×2
        let r_rel = r2 * cam1.r_world;
        let r_2d = [r_rel[(0, 0)], r_rel[(0, 1)], r_rel[(1, 0)], r_rel[(1, 1)]];

        Self { cam1, cam2, r_2d }
    }

    /// Create a new pair with cameras 1 and 2 swapped.
    ///
    /// Used for backward matching where the roles of query and target are reversed.
    /// The `r_2d` matrix is transposed since R1 @ R2^T = (R2 @ R1^T)^T.
    pub fn swapped(&self) -> Self {
        Self {
            cam1: self.cam2.clone(),
            cam2: self.cam1.clone(),
            r_2d: [self.r_2d[0], self.r_2d[2], self.r_2d[1], self.r_2d[3]],
        }
    }
}

/// Extract average feature size from a 2×2 affine shape matrix.
///
/// The affine is stored row-major as `[a00, a01, a10, a11]`.
/// Size is `0.5 * (||col0|| + ||col1||)` where col0 = (a00, a10) and col1 = (a01, a11).
///
/// # Parameters
///
/// * `affine` – Row-major 2×2 affine shape: `[a00, a01, a10, a11]`.
pub fn extract_affine_size(affine: &[f64; 4]) -> f64 {
    let col0_len = (affine[0] * affine[0] + affine[2] * affine[2]).sqrt();
    let col1_len = (affine[1] * affine[1] + affine[3] * affine[3]).sqrt();
    0.5 * (col0_len + col1_len)
}

/// Batch orientation consistency check (Stage 1).
///
/// For each candidate, extracts the first column vector from the rotated query
/// affine and the candidate affine, computes the cosine of the angle between
/// them, and checks whether it meets the threshold.
///
/// # Parameters
///
/// * `affine1_rotated` – Query affine pre-rotated by R_2d, row-major `[a00, a01, a10, a11]`.
/// * `candidate_affines` – Flat row-major array of N candidate affines (N×4).
/// * `n` – Number of candidates.
/// * `min_cos_angle` – Minimum cosine of the allowed angle difference.
///
/// # Returns
///
/// `Vec<bool>` of length `n`, true if the candidate passes the orientation check.
pub fn check_orientation_consistency_batch(
    affine1_rotated: &[f64; 4],
    candidate_affines: &[f64],
    n: usize,
    min_cos_angle: f64,
) -> Vec<bool> {
    debug_assert_eq!(candidate_affines.len(), n * 4);

    // First column of the rotated query affine: (a00, a10)
    let v1_x = affine1_rotated[0];
    let v1_y = affine1_rotated[2];
    let len1 = (v1_x * v1_x + v1_y * v1_y).sqrt();

    if len1 == 0.0 {
        return vec![false; n];
    }

    let mut result = Vec::with_capacity(n);
    for i in 0..n {
        let base = i * 4;
        // First column of candidate affine: (a00, a10)
        let v2_x = candidate_affines[base];
        let v2_y = candidate_affines[base + 2];
        let len2 = (v2_x * v2_x + v2_y * v2_y).sqrt();

        if len2 == 0.0 {
            result.push(false);
            continue;
        }

        let dot = v1_x * v2_x + v1_y * v2_y;
        let cos_angle = dot / (len1 * len2);
        result.push(cos_angle >= min_cos_angle);
    }

    result
}

/// Compute cosine of angle between unprojected rays in world coordinates.
///
/// # Parameters
///
/// * `x1` – 2D point in image 1 (pixel coordinates), `[x, y]`.
/// * `x2` – 2D point in image 2 (pixel coordinates), `[x, y]`.
/// * `geom` – Precomputed camera geometry.
///
/// # Returns
///
/// Cosine of the angle between the two rays, clamped to `[-1, 1]`.
pub fn compute_ray_angle_cosine(x1: [f64; 2], x2: [f64; 2], geom: &StereoPairGeometry) -> f64 {
    // Homogeneous coordinates
    let x1_hom = Vector3::new(x1[0], x1[1], 1.0);
    let x2_hom = Vector3::new(x2[0], x2[1], 1.0);

    // Unproject to camera coordinates, then rotate to world coordinates.
    // Rotation preserves vector length, so we normalize once at the end.
    let v1_world = geom.cam1.r_world * (geom.cam1.k_inv * x1_hom);
    let v2_world = geom.cam2.r_world * (geom.cam2.k_inv * x2_hom);

    let dot = v1_world.dot(&v2_world);
    let denom = v1_world.norm() * v2_world.norm();
    (dot / denom).clamp(-1.0, 1.0)
}

/// Triangulate a 3D point from two 2D correspondences using DLT (Direct Linear Transform).
///
/// Builds a 4×4 linear system from the projection equations and solves
/// via SVD. Returns `None` if the homogeneous coordinate is degenerate (≈0).
///
/// # Parameters
///
/// * `x1` – 2D point in image 1, `[x, y]`.
/// * `x2` – 2D point in image 2, `[x, y]`.
/// * `k1` – 3×3 intrinsic matrix for camera 1.
/// * `k2` – 3×3 intrinsic matrix for camera 2.
/// * `r1` – 3×3 cam_from_world rotation for camera 1.
/// * `t1` – cam_from_world translation for camera 1.
/// * `r2` – 3×3 cam_from_world rotation for camera 2.
/// * `t2` – cam_from_world translation for camera 2.
///
/// # Returns
///
/// `Some([x, y, z])` in world coordinates, or `None` if degenerate.
#[allow(clippy::too_many_arguments)]
pub fn triangulate_point_dlt(
    x1: [f64; 2],
    x2: [f64; 2],
    k1: &Matrix3<f64>,
    k2: &Matrix3<f64>,
    r1: &Matrix3<f64>,
    t1: &Vector3<f64>,
    r2: &Matrix3<f64>,
    t2: &Vector3<f64>,
) -> Option<[f64; 3]> {
    // Build 3×4 projection matrices: P = K [R | t]
    // nalgebra stores column-major, so we construct carefully.
    let p1 = build_projection_matrix(k1, r1, t1);
    let p2 = build_projection_matrix(k2, r2, t2);

    // Build 4×4 system A from cross-product formulation:
    //   x * P[2,:] - P[0,:]
    //   y * P[2,:] - P[1,:]
    let row0: nalgebra::RowVector4<f64> = x1[0] * p1.row(2) - p1.row(0);
    let row1: nalgebra::RowVector4<f64> = x1[1] * p1.row(2) - p1.row(1);
    let row2: nalgebra::RowVector4<f64> = x2[0] * p2.row(2) - p2.row(0);
    let row3: nalgebra::RowVector4<f64> = x2[1] * p2.row(2) - p2.row(1);

    let a = Matrix4::from_rows(&[row0, row1, row2, row3]);

    // Solve via SVD: solution is last column of V (smallest singular value)
    let svd = nalgebra::SVD::new(a, true, true);
    let v_t = svd.v_t?;

    // Last row of V^T = last column of V
    let x_hom = Vector4::new(v_t[(3, 0)], v_t[(3, 1)], v_t[(3, 2)], v_t[(3, 3)]);

    if x_hom[3].abs() < 1e-12 {
        return None;
    }

    Some([
        x_hom[0] / x_hom[3],
        x_hom[1] / x_hom[3],
        x_hom[2] / x_hom[3],
    ])
}

/// Build a 3×4 projection matrix P = K [R | t].
fn build_projection_matrix(
    k: &Matrix3<f64>,
    r: &Matrix3<f64>,
    t: &Vector3<f64>,
) -> nalgebra::Matrix3x4<f64> {
    // [R | t] is a 3×4 matrix
    let mut rt = nalgebra::Matrix3x4::zeros();
    for i in 0..3 {
        for j in 0..3 {
            rt[(i, j)] = r[(i, j)];
        }
        rt[(i, 3)] = t[i];
    }
    k * rt
}

/// Compute depth (Z coordinate) of a world point in camera space.
///
/// # Parameters
///
/// * `x_world` – 3D point in world coordinates.
/// * `r` – 3×3 cam_from_world rotation.
/// * `t` – cam_from_world translation.
///
/// # Returns
///
/// Z coordinate in camera space: `(R @ X + t)[2]`.
pub fn compute_depth_from_camera(x_world: &[f64; 3], r: &Matrix3<f64>, t: &Vector3<f64>) -> f64 {
    let x = Vector3::new(x_world[0], x_world[1], x_world[2]);
    let x_cam = r * x + t;
    x_cam[2]
}

/// Multiply two 2×2 matrices in row-major `[a00, a01, a10, a11]` form.
///
/// Returns the product in the same row-major layout.
fn mat2x2_mul(m: &[f64; 4], a: &[f64; 4]) -> [f64; 4] {
    [
        m[0] * a[0] + m[1] * a[2],
        m[0] * a[1] + m[1] * a[3],
        m[2] * a[0] + m[3] * a[2],
        m[2] * a[1] + m[3] * a[3],
    ]
}

/// Check size consistency for a single candidate (Stage 2).
///
/// First checks the ray angle cosine against the minimum triangulation angle.
/// If rays are nearly parallel, triangulation is unreliable and the candidate
/// is accepted (returns `true`). Otherwise, triangulates the point, computes
/// depths, scales the query affine by the inverse depth ratio, and compares
/// sizes.
///
/// # Parameters
///
/// * `x1` – 2D query point in image 1.
/// * `x2` – 2D candidate point in image 2.
/// * `affine1_rotated` – Query affine pre-rotated by R_2d (row-major 2×2).
/// * `affine2` – Candidate affine shape (row-major 2×2, `[a00, a01, a10, a11]`).
/// * `geom` – Precomputed stereo pair geometry.
/// * `config` – Filter configuration.
/// * `min_ray_angle_cos` – Precomputed `cos(min_triangulation_angle)`.
fn check_size_consistency(
    x1: [f64; 2],
    x2: [f64; 2],
    affine1_rotated: &[f64; 4],
    affine2: &[f64; 4],
    geom: &StereoPairGeometry,
    config: &GeometricFilterConfig,
    min_ray_angle_cos: f64,
) -> bool {
    let ray_cos = compute_ray_angle_cosine(x1, x2, geom);

    if ray_cos > min_ray_angle_cos {
        // Rays nearly parallel — triangulation unreliable, accept candidate.
        return true;
    }

    // Triangulate
    let x_world = match triangulate_point_dlt(
        x1,
        x2,
        &geom.cam1.k,
        &geom.cam2.k,
        &geom.cam1.r,
        &geom.cam1.t,
        &geom.cam2.r,
        &geom.cam2.t,
    ) {
        Some(pt) => pt,
        None => return false,
    };

    // Compute depths
    let d1 = compute_depth_from_camera(&x_world, &geom.cam1.r, &geom.cam1.t);
    let d2 = compute_depth_from_camera(&x_world, &geom.cam2.r, &geom.cam2.t);

    if d1 <= 0.0 || d2 <= 0.0 {
        return false;
    }

    // Scale the rotated query affine by the inverse depth ratio
    let scale = d1 / d2;
    let affine1_transformed = [
        affine1_rotated[0] * scale,
        affine1_rotated[1] * scale,
        affine1_rotated[2] * scale,
        affine1_rotated[3] * scale,
    ];

    let size1 = extract_affine_size(&affine1_transformed);
    let size2 = extract_affine_size(affine2);

    if size1 == 0.0 {
        return false;
    }

    let size_ratio = size2 / size1;
    config.geometric_size_ratio_min <= size_ratio && size_ratio <= config.geometric_size_ratio_max
}

/// Apply two-stage geometric filtering to candidate matches.
///
/// **Stage 1 (Always):** Check orientation consistency using rotation.
/// **Stage 2 (Conditional):** Check size consistency using depth ratio,
/// only when triangulation is reliable.
///
/// # Parameters
///
/// * `x1` – 2D query point in image 1, `[x, y]`.
/// * `affine1` – 2×2 affine shape for the query feature, row-major `[a00, a01, a10, a11]`.
/// * `candidate_positions` – Flat row-major array of N candidate positions (N×2).
/// * `candidate_affines` – Flat row-major array of N candidate affines (N×4).
/// * `n` – Number of candidates.
/// * `geom` – Precomputed stereo pair geometry.
/// * `config` – Filter configuration.
///
/// # Returns
///
/// `Vec<bool>` mask of length `n`, true if the candidate passes both stages.
pub fn two_stage_geometric_filter(
    x1: [f64; 2],
    affine1: &[f64; 4],
    candidate_positions: &[f64],
    candidate_affines: &[f64],
    n: usize,
    geom: &StereoPairGeometry,
    config: &GeometricFilterConfig,
) -> Vec<bool> {
    debug_assert_eq!(candidate_positions.len(), n * 2);
    debug_assert_eq!(candidate_affines.len(), n * 4);

    if n == 0 {
        return Vec::new();
    }

    // Precompute: rotate affine1 by R_2d
    let affine1_rotated = mat2x2_mul(&geom.r_2d, affine1);

    // Precompute cosine threshold for orientation check
    let min_cos_angle = config.max_angle_difference.to_radians().cos();

    // Stage 1: orientation consistency (batch)
    let mut mask =
        check_orientation_consistency_batch(&affine1_rotated, candidate_affines, n, min_cos_angle);

    // Stage 2: size consistency (per-candidate, only for those passing Stage 1)
    let min_tri_angle_cos = config.min_triangulation_angle.to_radians().cos();

    for i in 0..n {
        if !mask[i] {
            continue;
        }

        let x2 = [candidate_positions[i * 2], candidate_positions[i * 2 + 1]];
        let aff2: &[f64; 4] = candidate_affines[i * 4..i * 4 + 4].try_into().unwrap();

        if !check_size_consistency(
            x1,
            x2,
            &affine1_rotated,
            aff2,
            geom,
            config,
            min_tri_angle_cos,
        ) {
            mask[i] = false;
        }
    }

    mask
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn test_intrinsics() -> Matrix3<f64> {
        let mut k = Matrix3::identity();
        k[(0, 0)] = 500.0;
        k[(1, 1)] = 500.0;
        k[(0, 2)] = 320.0;
        k[(1, 2)] = 240.0;
        k
    }

    #[test]
    fn test_extract_affine_size_identity() {
        // Identity matrix: col0 = (1,0), col1 = (0,1), size = 0.5*(1+1) = 1
        let affine = [1.0, 0.0, 0.0, 1.0];
        assert_relative_eq!(extract_affine_size(&affine), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_extract_affine_size_scaled() {
        // 5*I: col0 = (5,0), col1 = (0,5), size = 5
        let affine = [5.0, 0.0, 0.0, 5.0];
        assert_relative_eq!(extract_affine_size(&affine), 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_extract_affine_size_asymmetric() {
        // col0 = (3,4) len=5, col1 = (0,1) len=1, size = 0.5*(5+1) = 3
        let affine = [3.0, 0.0, 4.0, 1.0];
        assert_relative_eq!(extract_affine_size(&affine), 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_orientation_consistency_aligned() {
        // Same affine → cosine = 1.0, should pass any threshold
        let affine1 = [3.0, 1.0, 4.0, 2.0];
        let candidates = [3.0, 1.0, 4.0, 2.0]; // identical
        let result = check_orientation_consistency_batch(&affine1, &candidates, 1, 0.95);
        assert!(result[0]);
    }

    #[test]
    fn test_orientation_consistency_perpendicular() {
        // affine1 first column: (1, 0), candidate first column: (0, 1)
        // These are perpendicular → cosine = 0
        let affine1 = [1.0, 0.0, 0.0, 1.0];
        let candidates = [0.0, 0.0, 1.0, 0.0]; // first col = (0, 1)
        let result = check_orientation_consistency_batch(&affine1, &candidates, 1, 0.5);
        assert!(!result[0]);
    }

    #[test]
    fn test_orientation_consistency_zero_vector() {
        let affine1 = [0.0, 0.0, 0.0, 0.0];
        let candidates = [1.0, 0.0, 0.0, 1.0];
        let result = check_orientation_consistency_batch(&affine1, &candidates, 1, 0.0);
        assert!(!result[0]);
    }

    #[test]
    fn test_camera_geometry_construction() {
        let k = test_intrinsics();
        let r = Matrix3::identity();
        let t1 = Vector3::new(0.0, 0.0, 0.0);
        let t2 = Vector3::new(1.0, 0.0, 0.0);

        let geom = StereoPairGeometry::new(&k, &k, &r, &r, &t1, &t2);

        // Camera centers: C = -R^T @ t
        assert_relative_eq!(geom.cam1.center, Vector3::zeros(), epsilon = 1e-10);
        assert_relative_eq!(
            geom.cam2.center,
            Vector3::new(-1.0, 0.0, 0.0),
            epsilon = 1e-10
        );

        // R_2d should be identity upper-left 2×2 (since both R are identity)
        assert_relative_eq!(geom.r_2d[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(geom.r_2d[1], 0.0, epsilon = 1e-10);
        assert_relative_eq!(geom.r_2d[2], 0.0, epsilon = 1e-10);
        assert_relative_eq!(geom.r_2d[3], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_compute_ray_angle_cosine_same_point_same_camera() {
        let k = test_intrinsics();
        let r = Matrix3::identity();
        let t = Vector3::zeros();

        let geom = StereoPairGeometry::new(&k, &k, &r, &r, &t, &t);

        // Same point in same camera → parallel rays → cosine = 1.0
        let cos_angle = compute_ray_angle_cosine([320.0, 240.0], [320.0, 240.0], &geom);
        assert_relative_eq!(cos_angle, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_compute_ray_angle_cosine_lateral_baseline() {
        let k = test_intrinsics();
        let r = Matrix3::identity();
        let t1 = Vector3::zeros();
        let t2 = Vector3::new(1.0, 0.0, 0.0);

        let geom = StereoPairGeometry::new(&k, &k, &r, &r, &t1, &t2);

        // Center point in both images: ray goes along z-axis → parallel → cosine ≈ 1.0
        let cos_angle = compute_ray_angle_cosine([320.0, 240.0], [320.0, 240.0], &geom);
        assert_relative_eq!(cos_angle, 1.0, epsilon = 1e-10);

        // Different points should give cosine < 1.0
        let cos_angle2 = compute_ray_angle_cosine([320.0, 240.0], [400.0, 240.0], &geom);
        assert!(cos_angle2 < 1.0);
        assert!(cos_angle2 > 0.0);
    }

    #[test]
    fn test_triangulate_point_dlt_basic() {
        let k = test_intrinsics();
        let r = Matrix3::identity();
        let t1 = Vector3::zeros();
        let t2 = Vector3::new(1.0, 0.0, 0.0);

        // A 3D point at (0, 0, 10) should project to the principal point in camera 1,
        // and slightly offset in camera 2 (because of the baseline).
        // P1 = K [I | 0], P2 = K [I | t2]
        // For X=(0,0,10): P1*X_h = K*(0,0,10,1)^T -> u=K*(0/10,0/10,1)=(320,240)
        // For X=(0,0,10): P2*X_h = K*(0+1, 0, 10)^T -> K*(1/10, 0, 1) = (500*0.1+320, 240) = (370, 240)
        let x1 = [320.0, 240.0];
        let x2 = [370.0, 240.0];

        let result = triangulate_point_dlt(x1, x2, &k, &k, &r, &t1, &r, &t2);
        assert!(result.is_some());
        let pt = result.unwrap();

        assert_relative_eq!(pt[0], 0.0, epsilon = 0.1);
        assert_relative_eq!(pt[1], 0.0, epsilon = 0.1);
        assert_relative_eq!(pt[2], 10.0, epsilon = 0.1);
    }

    #[test]
    fn test_compute_depth_from_camera_basic() {
        let r = Matrix3::identity();
        let t = Vector3::zeros();
        let x_world = [0.0, 0.0, 10.0];
        let depth = compute_depth_from_camera(&x_world, &r, &t);
        assert_relative_eq!(depth, 10.0, epsilon = 1e-10);
    }

    #[test]
    fn test_compute_depth_from_camera_with_translation() {
        let r = Matrix3::identity();
        let t = Vector3::new(0.0, 0.0, 5.0);
        let x_world = [0.0, 0.0, 10.0];
        // X_cam = R * X + t = (0,0,10) + (0,0,5) = (0,0,15)
        let depth = compute_depth_from_camera(&x_world, &r, &t);
        assert_relative_eq!(depth, 15.0, epsilon = 1e-10);
    }

    #[test]
    fn test_two_stage_geometric_filter_empty() {
        let k = test_intrinsics();
        let r = Matrix3::identity();
        let t = Vector3::zeros();
        let geom = StereoPairGeometry::new(&k, &k, &r, &r, &t, &t);
        let config = GeometricFilterConfig::default();

        let mask = two_stage_geometric_filter(
            [320.0, 240.0],
            &[1.0, 0.0, 0.0, 1.0],
            &[],
            &[],
            0,
            &geom,
            &config,
        );
        assert!(mask.is_empty());
    }

    #[test]
    fn test_two_stage_geometric_filter_consistent_match() {
        let k = test_intrinsics();
        let r = Matrix3::identity();
        let t1 = Vector3::zeros();
        let t2 = Vector3::new(1.0, 0.0, 0.0);
        let geom = StereoPairGeometry::new(&k, &k, &r, &r, &t1, &t2);
        let config = GeometricFilterConfig::default();

        // Query at image center with identity affine
        let x1 = [320.0, 240.0];
        let affine1 = [5.0, 0.0, 0.0, 5.0];

        // Candidate also near center with same-ish affine (consistent match)
        let candidate_pos = [370.0, 240.0];
        let candidate_aff = [5.0, 0.0, 0.0, 5.0];

        let mask = two_stage_geometric_filter(
            x1,
            &affine1,
            &candidate_pos,
            &candidate_aff,
            1,
            &geom,
            &config,
        );
        assert_eq!(mask.len(), 1);
        assert!(mask[0], "Consistent match should pass the filter");
    }

    #[test]
    fn test_two_stage_geometric_filter_rejects_bad_orientation() {
        let k = test_intrinsics();
        let r = Matrix3::identity();
        let t1 = Vector3::zeros();
        let t2 = Vector3::new(1.0, 0.0, 0.0);
        let geom = StereoPairGeometry::new(&k, &k, &r, &r, &t1, &t2);
        let config = GeometricFilterConfig::default();

        // Query with affine whose first column points right: (5, 0)
        let x1 = [320.0, 240.0];
        let affine1 = [5.0, 0.0, 0.0, 5.0];

        // Candidate with first column pointing up: (0, 5) — perpendicular, should fail
        let candidate_pos = [370.0, 240.0];
        let candidate_aff = [0.0, 5.0, 5.0, 0.0]; // first col = (0, 5)

        let mask = two_stage_geometric_filter(
            x1,
            &affine1,
            &candidate_pos,
            &candidate_aff,
            1,
            &geom,
            &config,
        );
        assert_eq!(mask.len(), 1);
        assert!(
            !mask[0],
            "Perpendicular orientation should be rejected by Stage 1"
        );
    }

    #[test]
    fn test_geometric_filter_config_custom() {
        let config = GeometricFilterConfig {
            max_angle_difference: 20.0,
            min_triangulation_angle: 3.0,
            geometric_size_ratio_min: 0.75,
            geometric_size_ratio_max: 1.333,
        };
        assert_relative_eq!(config.max_angle_difference, 20.0);
        assert_relative_eq!(config.geometric_size_ratio_min, 0.75);
    }

    #[test]
    fn test_two_stage_geometric_filter_large_size_mismatch() {
        let k = test_intrinsics();
        let r = Matrix3::identity();
        let t1 = Vector3::zeros();
        let t2 = Vector3::new(1.0, 0.0, 0.0);
        let geom = StereoPairGeometry::new(&k, &k, &r, &r, &t1, &t2);
        let config = GeometricFilterConfig::default();

        // Query with size 5, candidate with size 25 (5x mismatch)
        let x1 = [320.0, 240.0];
        let affine1 = [5.0, 0.0, 0.0, 5.0]; // size = 5

        let candidate_pos = [370.0, 240.0];
        let candidate_aff = [25.0, 0.0, 0.0, 25.0]; // size = 25, ratio = 5.0

        let mask = two_stage_geometric_filter(
            x1,
            &affine1,
            &candidate_pos,
            &candidate_aff,
            1,
            &geom,
            &config,
        );
        assert_eq!(mask.len(), 1);
        // Lateral baseline with 50px horizontal disparity gives a meaningful
        // triangulation angle, so stage 2 (size check) applies.
        // Size ratio 5.0 >> max 1.25 → rejected.
        assert!(
            !mask[0],
            "5x size mismatch should be rejected with lateral baseline"
        );
    }

    #[test]
    fn test_two_stage_geometric_filter_forward_motion_depth_change() {
        // Camera 1 at origin, camera 2 moved forward by 5m
        // Object at depth 10m from cam1 → depth 5m from cam2
        // Feature should appear ~2x larger in cam2
        let k = test_intrinsics();
        let r = Matrix3::identity();
        let t1 = Vector3::zeros();
        let t2 = Vector3::new(0.0, 0.0, 5.0); // forward motion
        let geom = StereoPairGeometry::new(&k, &k, &r, &r, &t1, &t2);
        let config = GeometricFilterConfig::default();

        // Object at (0, 0, 10) in world coords
        // Projects to principal point in cam1: (320, 240)
        // In cam2: X_cam2 = R*(0,0,10) + (0,0,5) = (0,0,15)
        // Projects to (320, 240) in cam2 as well
        let x1 = [320.0, 240.0];
        let affine1 = [5.0, 0.0, 0.0, 5.0];

        // Candidate at same pixel but 2x larger (depth halved)
        let candidate_pos = [320.0, 240.0];
        let candidate_aff = [10.0, 0.0, 0.0, 10.0];

        let mask = two_stage_geometric_filter(
            x1,
            &affine1,
            &candidate_pos,
            &candidate_aff,
            1,
            &geom,
            &config,
        );
        assert_eq!(mask.len(), 1);
        // For forward motion, rays are nearly parallel, so stage 2 (size check)
        // is typically skipped — candidate should pass
        assert!(
            mask[0],
            "Forward motion with nearly parallel rays should skip size check and accept"
        );
    }

    #[test]
    fn test_two_stage_geometric_filter_strict_config() {
        let k = test_intrinsics();
        let r = Matrix3::identity();
        let t1 = Vector3::zeros();
        let t2 = Vector3::new(1.0, 0.0, 0.0);
        let geom = StereoPairGeometry::new(&k, &k, &r, &r, &t1, &t2);

        // Strict config: only 5 degrees of angle tolerance
        let config = GeometricFilterConfig {
            max_angle_difference: 5.0,
            min_triangulation_angle: 5.0,
            geometric_size_ratio_min: 0.9,
            geometric_size_ratio_max: 1.1,
        };

        // Query with identity affine, candidate slightly rotated (~10 degrees)
        let x1 = [320.0, 240.0];
        let affine1 = [5.0, 0.0, 0.0, 5.0];

        // Rotate first column by ~10 degrees: cos(10°) ≈ 0.985, sin(10°) ≈ 0.174
        let angle = 10.0_f64.to_radians();
        let candidate_pos = [370.0, 240.0];
        let candidate_aff = [5.0 * angle.cos(), 0.0, 5.0 * angle.sin(), 5.0];

        let mask = two_stage_geometric_filter(
            x1,
            &affine1,
            &candidate_pos,
            &candidate_aff,
            1,
            &geom,
            &config,
        );
        assert_eq!(mask.len(), 1);
        // 10 degree difference > 5 degree threshold → should be rejected
        assert!(
            !mask[0],
            "10° rotation should be rejected with 5° max angle tolerance"
        );
    }

    #[test]
    fn test_stereo_pair_geometry_swapped() {
        let k = test_intrinsics();
        let r = Matrix3::identity();
        let t1 = Vector3::zeros();
        let t2 = Vector3::new(1.0, 0.0, 0.0);
        let geom = StereoPairGeometry::new(&k, &k, &r, &r, &t1, &t2);
        let swapped = geom.swapped();

        // Swapped cam1 should be original cam2
        assert_relative_eq!(swapped.cam1.center, geom.cam2.center, epsilon = 1e-10);
        assert_relative_eq!(swapped.cam2.center, geom.cam1.center, epsilon = 1e-10);

        // R_2d should be transposed
        assert_relative_eq!(swapped.r_2d[0], geom.r_2d[0], epsilon = 1e-10);
        assert_relative_eq!(swapped.r_2d[1], geom.r_2d[2], epsilon = 1e-10);
        assert_relative_eq!(swapped.r_2d[2], geom.r_2d[1], epsilon = 1e-10);
        assert_relative_eq!(swapped.r_2d[3], geom.r_2d[3], epsilon = 1e-10);
    }

    #[test]
    fn test_extract_affine_size_zero() {
        let affine = [0.0, 0.0, 0.0, 0.0];
        assert_relative_eq!(extract_affine_size(&affine), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_orientation_consistency_batch_multiple() {
        // Test with 4 candidates: 2 aligned, 2 perpendicular
        let affine1 = [5.0, 0.0, 0.0, 5.0]; // first col = (5, 0)
        let candidates = [
            5.0, 0.0, 0.0, 5.0, // aligned (cosine ≈ 1)
            0.0, 5.0, 5.0, 0.0, // perpendicular (cosine ≈ 0)
            4.8, 0.0, 1.0, 5.0, // slightly rotated (cosine > 0.9)
            0.0, 5.0, -5.0, 0.0, // perpendicular (cosine ≈ 0)
        ];
        let result = check_orientation_consistency_batch(&affine1, &candidates, 4, 0.8);
        assert!(result[0], "Aligned should pass");
        assert!(!result[1], "Perpendicular should fail");
        assert!(result[2], "Slightly rotated should pass");
        assert!(!result[3], "Perpendicular should fail");
    }

    #[test]
    fn test_two_stage_geometric_filter_multiple_candidates() {
        let k = test_intrinsics();
        let r = Matrix3::identity();
        let t1 = Vector3::zeros();
        let t2 = Vector3::new(1.0, 0.0, 0.0);
        let geom = StereoPairGeometry::new(&k, &k, &r, &r, &t1, &t2);
        let config = GeometricFilterConfig::default();

        let x1 = [320.0, 240.0];
        let affine1 = [5.0, 0.0, 0.0, 5.0];

        // 3 candidates: good, bad orientation, good
        let candidate_pos = [
            370.0, 240.0, // candidate 0 (good)
            370.0, 240.0, // candidate 1 (bad orientation)
            370.0, 240.0, // candidate 2 (good)
        ];
        let candidate_aff = [
            5.0, 0.0, 0.0, 5.0, // candidate 0: same orientation
            0.0, 5.0, 5.0, 0.0, // candidate 1: perpendicular
            4.5, 0.0, 0.0, 4.5, // candidate 2: similar orientation
        ];

        let mask = two_stage_geometric_filter(
            x1,
            &affine1,
            &candidate_pos,
            &candidate_aff,
            3,
            &geom,
            &config,
        );
        assert_eq!(mask.len(), 3);
        assert!(mask[0], "Candidate 0 should pass");
        assert!(!mask[1], "Candidate 1 should fail orientation");
        assert!(mask[2], "Candidate 2 should pass");
    }
}
