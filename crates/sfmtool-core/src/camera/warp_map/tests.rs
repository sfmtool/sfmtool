use super::*;
use crate::camera_intrinsics::CameraModel;

/// Helper: build a simple pinhole camera.
fn pinhole(width: u32, height: u32, focal: f64) -> CameraIntrinsics {
    CameraIntrinsics {
        model: CameraModel::Pinhole {
            focal_length_x: focal,
            focal_length_y: focal,
            principal_point_x: width as f64 / 2.0,
            principal_point_y: height as f64 / 2.0,
        },
        width,
        height,
    }
}

/// Helper: build a simple radial camera with distortion.
fn simple_radial(width: u32, height: u32, focal: f64, k1: f64) -> CameraIntrinsics {
    CameraIntrinsics {
        model: CameraModel::SimpleRadial {
            focal_length: focal,
            principal_point_x: width as f64 / 2.0,
            principal_point_y: height as f64 / 2.0,
            radial_distortion_k1: k1,
        },
        width,
        height,
    }
}

/// Helper: build a simple radial fisheye camera.
fn simple_radial_fisheye(width: u32, height: u32, focal: f64, k1: f64) -> CameraIntrinsics {
    CameraIntrinsics {
        model: CameraModel::SimpleRadialFisheye {
            focal_length: focal,
            principal_point_x: width as f64 / 2.0,
            principal_point_y: height as f64 / 2.0,
            radial_distortion_k1: k1,
        },
        width,
        height,
    }
}

/// Helper: build an equirectangular camera (full sphere).
fn equirectangular(width: u32, height: u32) -> CameraIntrinsics {
    let fx = width as f64 / (2.0 * std::f64::consts::PI);
    let fy = height as f64 / std::f64::consts::PI;
    CameraIntrinsics {
        model: CameraModel::Equirectangular {
            focal_length_x: fx,
            focal_length_y: fy,
            principal_point_x: width as f64 / 2.0,
            principal_point_y: height as f64 / 2.0,
        },
        width,
        height,
    }
}

// -----------------------------------------------------------------------
// Identity map
// -----------------------------------------------------------------------

#[test]
fn identity_map_produces_pixel_centers() {
    let cam = pinhole(64, 48, 100.0);
    let warp = WarpMap::from_cameras(&cam, &cam);

    assert_eq!(warp.width(), 64);
    assert_eq!(warp.height(), 48);

    for row in 0..warp.height() {
        for col in 0..warp.width() {
            assert!(
                warp.is_valid(col, row),
                "pixel ({col}, {row}) should be valid"
            );
            let (x, y) = warp.get(col, row);
            let expected_x = col as f32 + 0.5;
            let expected_y = row as f32 + 0.5;
            assert!(
                (x - expected_x).abs() < 1e-3,
                "col={col} row={row}: x={x} expected {expected_x}"
            );
            assert!(
                (y - expected_y).abs() < 1e-3,
                "col={col} row={row}: y={y} expected {expected_y}"
            );
        }
    }
}

// -----------------------------------------------------------------------
// Round-trip: undistort then redistort recovers original
// -----------------------------------------------------------------------

#[test]
fn round_trip_undistort_redistort() {
    let distorted = simple_radial(64, 48, 100.0, 0.1);
    let undistorted = pinhole(64, 48, 100.0);

    // Forward: distorted → undistorted (undistort map)
    let undistort_map = WarpMap::from_cameras(&distorted, &undistorted);
    // Reverse: undistorted → distorted (redistort map)
    let redistort_map = WarpMap::from_cameras(&undistorted, &distorted);

    // For pixels near the center (avoid boundary issues), composing the
    // two maps should recover roughly the pixel center.
    let margin = 8;
    for row in margin..(undistorted.height - margin) {
        for col in margin..(undistorted.width - margin) {
            if !undistort_map.is_valid(col, row) {
                continue;
            }
            let (sx, sy) = undistort_map.get(col, row);
            // Look up the redistort map at the (fractional) source pixel.
            // Use nearest-neighbour for simplicity.
            let sc = (sx - 0.5).round() as u32;
            let sr = (sy - 0.5).round() as u32;
            if sc >= redistort_map.width() || sr >= redistort_map.height() {
                continue;
            }
            if !redistort_map.is_valid(sc, sr) {
                continue;
            }
            let (rx, ry) = redistort_map.get(sc, sr);
            let expected_x = col as f32 + 0.5;
            let expected_y = row as f32 + 0.5;
            // Tolerance is generous because of nearest-neighbour sampling.
            assert!(
                (rx - expected_x).abs() < 2.0,
                "col={col} row={row}: rx={rx} expected ~{expected_x}"
            );
            assert!(
                (ry - expected_y).abs() < 2.0,
                "col={col} row={row}: ry={ry} expected ~{expected_y}"
            );
        }
    }
}

// -----------------------------------------------------------------------
// Equirectangular target — centre should have valid pixels
// -----------------------------------------------------------------------

#[test]
fn equirectangular_target_center_valid() {
    let fisheye = simple_radial_fisheye(200, 200, 100.0, 0.0);
    let equirect = equirectangular(400, 200);

    let warp = WarpMap::from_cameras(&fisheye, &equirect);

    // The centre of the equirectangular image corresponds to the forward
    // direction, which should definitely map into the fisheye image.
    let cx = warp.width() / 2;
    let cy = warp.height() / 2;
    assert!(
        warp.is_valid(cx, cy),
        "centre of equirectangular target should be valid"
    );

    // Check a small region around centre.
    let margin = 10;
    let mut valid_count = 0u32;
    let total = (2 * margin + 1) * (2 * margin + 1);
    for dr in 0..=(2 * margin) {
        for dc in 0..=(2 * margin) {
            let c = cx - margin + dc;
            let r = cy - margin + dr;
            if warp.is_valid(c, r) {
                valid_count += 1;
            }
        }
    }
    assert!(
        valid_count == total,
        "expected all {total} centre pixels valid, got {valid_count}"
    );
}

// -----------------------------------------------------------------------
// SVD computation
// -----------------------------------------------------------------------

#[test]
fn compute_svd_runs_without_panic() {
    let cam = pinhole(32, 24, 50.0);
    let mut warp = WarpMap::from_cameras(&cam, &cam);
    assert!(!warp.has_svd());

    warp.compute_svd();
    assert!(warp.has_svd());

    let svd = warp.svd().unwrap();
    let n = 32 * 24;
    assert_eq!(svd.sigma_major.len(), n);
    assert_eq!(svd.sigma_minor.len(), n);
    assert_eq!(svd.major_dir.len(), 2 * n);
}

#[test]
fn identity_svd_values_near_one() {
    let cam = pinhole(32, 24, 50.0);
    let mut warp = WarpMap::from_cameras(&cam, &cam);
    warp.compute_svd();
    let svd = warp.svd().unwrap();

    // Interior pixels (away from boundary) should have singular values
    // very close to 1.0 for an identity map.
    for row in 2..22 {
        for col in 2..30 {
            let idx = row * 32 + col;
            assert!(
                (svd.sigma_major[idx] - 1.0).abs() < 0.01,
                "sigma_major at ({col},{row}) = {} expected ~1",
                svd.sigma_major[idx]
            );
            assert!(
                (svd.sigma_minor[idx] - 1.0).abs() < 0.01,
                "sigma_minor at ({col},{row}) = {} expected ~1",
                svd.sigma_minor[idx]
            );
        }
    }
}

// -----------------------------------------------------------------------
// is_valid
// -----------------------------------------------------------------------

#[test]
fn is_valid_returns_correct_values() {
    // Build a map where some pixels are intentionally out of bounds.
    // Use a small source image and a large destination with the same
    // intrinsics so that edge pixels in dst land outside src.
    let src = pinhole(32, 24, 50.0);
    let dst = pinhole(64, 48, 50.0);

    let warp = WarpMap::from_cameras(&src, &dst);

    // Centre of dst should map to centre of src and be valid.
    assert!(warp.is_valid(32, 24));

    // Far corners of dst should map outside the small src and be invalid.
    // The dst is 64x48 with cx=32, cy=24 and src is 32x24 with cx=16, cy=12.
    // dst pixel (0, 0) → unproject → project → will be at src pixel
    // (0 + 0.5 - 32) * 50/50 + 16 = -15.5 which is < 0 → invalid.
    assert!(
        !warp.is_valid(0, 0),
        "corner pixel (0,0) should be out of source bounds"
    );
    assert!(
        !warp.is_valid(63, 47),
        "corner pixel (63,47) should be out of source bounds"
    );
}

// -----------------------------------------------------------------------
// SVD of scaled map
// -----------------------------------------------------------------------

#[test]
fn svd_of_scaled_map() {
    // A 2x zoom should produce singular values near 2.0.
    // Use a large source so that dst pixels don't go out of bounds.
    let src = pinhole(128, 96, 100.0);
    let dst = pinhole(64, 48, 50.0); // half the focal length → 2x zoom out

    let mut warp = WarpMap::from_cameras(&src, &dst);
    warp.compute_svd();
    let svd = warp.svd().unwrap();

    // Check interior (skip boundary pixels where central differences
    // fall back to identity).
    for row in 2..46 {
        for col in 2..62 {
            let idx = row * 64 + col;
            assert!(
                (svd.sigma_major[idx] - 2.0).abs() < 0.1,
                "sigma_major at ({col},{row}) = {} expected ~2",
                svd.sigma_major[idx]
            );
            assert!(
                (svd.sigma_minor[idx] - 2.0).abs() < 0.1,
                "sigma_minor at ({col},{row}) = {} expected ~2",
                svd.sigma_minor[idx]
            );
        }
    }
}

// -----------------------------------------------------------------------
// 2x2 SVD helper
// -----------------------------------------------------------------------

#[test]
fn svd_2x2_identity() {
    let (s1, s2, dx, dy) = svd_2x2(1.0, 0.0, 0.0, 1.0);
    assert!((s1 - 1.0).abs() < 1e-6);
    assert!((s2 - 1.0).abs() < 1e-6);
    // Direction can be either (1,0) or (0,1) since both singular values
    // are equal — just check it's unit length.
    assert!((dx * dx + dy * dy - 1.0).abs() < 1e-6);
}

#[test]
fn svd_2x2_diagonal() {
    let (s1, s2, _dx, _dy) = svd_2x2(3.0, 0.0, 0.0, 2.0);
    assert!((s1 - 3.0).abs() < 1e-5, "s1 = {s1}");
    assert!((s2 - 2.0).abs() < 1e-5, "s2 = {s2}");
}

#[test]
fn svd_2x2_rotation() {
    // Pure rotation should have both singular values = 1.
    let angle = 0.7_f32;
    let (s1, s2, _, _) = svd_2x2(angle.cos(), -angle.sin(), angle.sin(), angle.cos());
    assert!((s1 - 1.0).abs() < 1e-5, "s1 = {s1}");
    assert!((s2 - 1.0).abs() < 1e-5, "s2 = {s2}");
}

// -----------------------------------------------------------------------
// Pose-aware constructors
// -----------------------------------------------------------------------

/// Compare two warp maps pixel-by-pixel with a per-pixel tolerance.
///
/// Both maps are required to agree on which pixels are valid (NaN vs not).
fn assert_maps_equal(a: &WarpMap, b: &WarpMap, tol: f32) {
    assert_eq!(a.width(), b.width());
    assert_eq!(a.height(), b.height());
    for row in 0..a.height() {
        for col in 0..a.width() {
            let va = a.is_valid(col, row);
            let vb = b.is_valid(col, row);
            assert_eq!(va, vb, "validity mismatch at ({col}, {row})");
            if !va {
                continue;
            }
            let (ax, ay) = a.get(col, row);
            let (bx, by) = b.get(col, row);
            let err = ((ax - bx).powi(2) + (ay - by).powi(2)).sqrt();
            assert!(
                err < tol,
                "mismatch at ({col},{row}): ({ax},{ay}) vs ({bx},{by}), err={err}"
            );
        }
    }
}

#[test]
fn from_cameras_with_rotation_identity_matches_from_cameras() {
    // rotation-aware with identity rotation must equal the baseline.
    let src = pinhole(64, 48, 100.0);
    let dst = pinhole(64, 48, 100.0);

    let baseline = WarpMap::from_cameras(&src, &dst);
    let rotated = WarpMap::from_cameras_with_rotation(&src, &dst, &RotQuaternion::identity());

    assert_maps_equal(&baseline, &rotated, 1e-4);
}

#[test]
fn from_cameras_with_rotation_identity_matches_from_cameras_equirect() {
    // Same invariance must hold for equirectangular / fisheye, which go
    // through the ray-based path.
    let src = simple_radial_fisheye(200, 200, 100.0, 0.0);
    let dst = equirectangular(400, 200);

    let baseline = WarpMap::from_cameras(&src, &dst);
    let rotated = WarpMap::from_cameras_with_rotation(&src, &dst, &RotQuaternion::identity());

    assert_maps_equal(&baseline, &rotated, 1e-3);
}

#[test]
fn from_cameras_with_pose_infinity_matches_rotation_only() {
    // With depth = +INF the translation drops out of the formulation.
    let src = pinhole(64, 48, 100.0);
    let dst = pinhole(64, 48, 100.0);

    let rot = RotQuaternion::from_axis_angle(Vector3::new(0.0, 1.0, 0.0), 0.2).unwrap();
    let src_from_world = RigidTransform::new(rot.clone(), Vector3::new(0.5, -0.2, 3.0));
    let dst_from_world =
        RigidTransform::new(RotQuaternion::identity(), Vector3::new(-0.1, 0.3, 2.5));
    // Relative rotation from dst-frame to src-frame: R_sw * R_dw^T.
    let r_sd_mat =
        src_from_world.to_rotation_matrix() * dst_from_world.to_rotation_matrix().transpose();
    let r_sd = RotQuaternion::from_rotation_matrix(r_sd_mat);

    let pose_inf = WarpMap::from_cameras_with_pose(
        &src,
        &dst,
        &src_from_world,
        &dst_from_world,
        f64::INFINITY,
    );
    let rot_only = WarpMap::from_cameras_with_rotation(&src, &dst, &r_sd);
    assert_maps_equal(&pose_inf, &rot_only, 1e-3);
}

#[test]
fn from_cameras_with_pose_coincident_pose_matches_from_cameras() {
    // If src and dst cameras sit at the same world pose, the pose
    // construction must agree with from_cameras (scene points on the
    // radius-r sphere around dst land in the same place in src).
    let src = pinhole(64, 48, 100.0);
    let dst = pinhole(64, 48, 100.0);

    let pose = RigidTransform::new(
        RotQuaternion::from_axis_angle(Vector3::new(1.0, 0.5, 0.2), 0.4).unwrap(),
        Vector3::new(2.0, -1.0, 0.5),
    );

    let baseline = WarpMap::from_cameras(&src, &dst);
    let posed = WarpMap::from_cameras_with_pose(&src, &dst, &pose, &pose, 5.0);
    assert_maps_equal(&baseline, &posed, 1e-2);
}

#[test]
fn from_cameras_with_pose_known_depth_synthetic_sphere() {
    // Synthetic "scene" = a sphere of radius R centered at dst camera.
    // Every dst pixel center traces a ray hitting a point on that sphere.
    // When we transform that point into src coordinates and project, we
    // should get a pixel which is in-bounds and matches the warp exactly.
    let src = pinhole(160, 120, 200.0);
    let dst = pinhole(160, 120, 200.0);

    let src_from_world = RigidTransform::new(
        RotQuaternion::from_axis_angle(Vector3::new(0.1, 1.0, 0.05), 0.15).unwrap(),
        Vector3::new(0.5, 0.0, 0.0),
    );
    let dst_from_world = RigidTransform::new(
        RotQuaternion::from_axis_angle(Vector3::new(0.0, 1.0, 0.1), -0.05).unwrap(),
        Vector3::new(0.0, 0.0, 0.0),
    );
    let radius = 10.0_f64;

    let warp =
        WarpMap::from_cameras_with_pose(&src, &dst, &src_from_world, &dst_from_world, radius);

    // Precompute R_sd and T_sd just like the implementation does.
    let r_sw = src_from_world.to_rotation_matrix();
    let r_dw = dst_from_world.to_rotation_matrix();
    let r_sd = r_sw * r_dw.transpose();
    let t_sd = src_from_world.translation - r_sd * dst_from_world.translation;

    // Spot-check the middle 80% of pixels — edges may miss src bounds.
    let w = warp.width();
    let h = warp.height();
    let mut checked = 0usize;
    for row in (h / 10)..(9 * h / 10) {
        for col in (w / 10)..(9 * w / 10) {
            if !warp.is_valid(col, row) {
                continue;
            }
            let u = col as f64 + 0.5;
            let v = row as f64 + 0.5;
            let d_dst = dst.pixel_to_ray(u, v);
            let d_dst_vec = Vector3::new(d_dst[0], d_dst[1], d_dst[2]);
            let p_dst = radius * d_dst_vec;
            let p_src = r_sd * p_dst + t_sd;
            let (exp_x, exp_y) = src.ray_to_pixel([p_src.x, p_src.y, p_src.z]).unwrap();

            let (gx, gy) = warp.get(col, row);
            assert!(
                (gx as f64 - exp_x).abs() < 1e-3,
                "x mismatch at ({col},{row}): got {gx}, expected {exp_x}"
            );
            assert!(
                (gy as f64 - exp_y).abs() < 1e-3,
                "y mismatch at ({col},{row}): got {gy}, expected {exp_y}"
            );
            checked += 1;
        }
    }
    assert!(checked > 100, "too few pixels in-bounds: {checked}");
}

#[test]
fn from_cameras_with_pose_baseline_comparable_to_depth() {
    // Spec: at r ≈ B_max, a small-angle approximation (rotation-only,
    // ignoring translation) errs by noticeable amounts. The exact ray
    // formula must still be pixel-accurate. We construct a synthetic
    // scene on a sphere and verify the two formulations disagree.

    let cam = pinhole(320, 240, 300.0);
    // Baseline = 0.3, depth = 1.0: comparable scale, but mild enough that
    // most pixels still project inside the image. The small-angle approx
    // (rotation-only) disagrees with the exact formulation by many
    // pixels under these conditions.
    let src_from_world =
        RigidTransform::new(RotQuaternion::identity(), Vector3::new(0.3, 0.0, 0.0));
    let dst_from_world =
        RigidTransform::new(RotQuaternion::identity(), Vector3::new(0.0, 0.0, 0.0));
    let radius = 1.0_f64;

    let exact =
        WarpMap::from_cameras_with_pose(&cam, &cam, &src_from_world, &dst_from_world, radius);
    // The rotation-only (infinite-depth) approximation ignores the
    // translation entirely — it should be obviously wrong here.
    let approx = WarpMap::from_cameras_with_rotation(&cam, &cam, &RotQuaternion::identity());

    // Find the maximum error between the two maps across valid pixels.
    let mut max_err_sq = 0.0_f32;
    let mut counted = 0usize;
    for row in (cam.height / 4)..(3 * cam.height / 4) {
        for col in (cam.width / 4)..(3 * cam.width / 4) {
            if !exact.is_valid(col, row) || !approx.is_valid(col, row) {
                continue;
            }
            let (ax, ay) = exact.get(col, row);
            let (bx, by) = approx.get(col, row);
            let err_sq = (ax - bx).powi(2) + (ay - by).powi(2);
            if err_sq > max_err_sq {
                max_err_sq = err_sq;
            }
            counted += 1;
        }
    }
    assert!(counted > 100);
    let max_err = max_err_sq.sqrt();
    // Sanity: the approximation should be off by many pixels. If the two
    // agreed to <0.1 px we'd know one of the two was silently ignoring
    // its inputs.
    assert!(
        max_err > 5.0,
        "rotation-only approximation should disagree with exact; got max_err={max_err} px",
    );

    // Cross-check the exact map against a hand-computed reprojection at
    // the centre pixel. src_from_world.translation = (0.3, 0, 0) means
    // src camera sits at world (-0.3, 0, 0). Ray from dst centre is
    // (0, 0, 1); at depth 1 it lands at world point (0, 0, 1). In src
    // frame: p_src = R_sw * p_w + t_sw = (0, 0, 1) + (0.3, 0, 0)
    // = (0.3, 0, 1), which projects to pixel (cx + 0.3*f, cy) =
    // (160 + 90, 120) = (250, 120).
    let cx = cam.width / 2;
    let cy = cam.height / 2;
    assert!(exact.is_valid(cx, cy));
    let (gx, gy) = exact.get(cx, cy);
    // Allow ±1 px since (cx, cy) is the integer pixel index and the
    // ray traces from the pixel centre at (cx + 0.5, cy + 0.5).
    assert!(
        (gx - 250.5).abs() < 1.0,
        "centre sx = {gx}, expected ~250.5"
    );
    assert!(
        (gy - 120.5).abs() < 1.0,
        "centre sy = {gy}, expected ~120.5"
    );

    // The rotation-only approximation at the same pixel is the identity
    // map: pixel centre (160.5, 120.5).
    let (ax, ay) = approx.get(cx, cy);
    assert!((ax - 160.5).abs() < 0.5);
    assert!((ay - 120.5).abs() < 0.5);
    let center_err = ((gx - ax).powi(2) + (gy - ay).powi(2)).sqrt();
    assert!(
        center_err > 50.0,
        "rotation-only vs exact must disagree by many pixels; got {center_err}"
    );
}

#[test]
fn from_cameras_with_pose_svd_still_works() {
    // compute_svd must work identically on a pose-built map.
    let src = pinhole(64, 48, 100.0);
    let dst = pinhole(64, 48, 100.0);
    let src_from_world = RigidTransform::new(
        RotQuaternion::from_axis_angle(Vector3::new(0.0, 1.0, 0.0), 0.05).unwrap(),
        Vector3::new(0.02, 0.0, 0.0),
    );
    let dst_from_world = RigidTransform::identity();
    let mut warp =
        WarpMap::from_cameras_with_pose(&src, &dst, &src_from_world, &dst_from_world, 100.0);
    assert!(!warp.has_svd());
    warp.compute_svd();
    assert!(warp.has_svd());

    // Singular values must be finite and positive somewhere away from
    // the boundary.
    let svd = warp.svd().unwrap();
    let idx = (24 * 64 + 32) as usize;
    assert!(svd.sigma_major[idx].is_finite() && svd.sigma_major[idx] > 0.0);
    assert!(svd.sigma_minor[idx].is_finite() && svd.sigma_minor[idx] > 0.0);
}

#[test]
fn from_cameras_with_pose_equirect_dst() {
    // Equirectangular dst: every ray is valid and all pixels with a
    // proper src projection should be in-bounds.
    let src = pinhole(200, 200, 100.0);
    let dst = equirectangular(400, 200);
    let pose = RigidTransform::identity();

    let warp = WarpMap::from_cameras_with_pose(&src, &dst, &pose, &pose, 1e6);

    // Centre of equirect (forward) must be valid in src.
    let cx = warp.width() / 2;
    let cy = warp.height() / 2;
    assert!(warp.is_valid(cx, cy));
}
