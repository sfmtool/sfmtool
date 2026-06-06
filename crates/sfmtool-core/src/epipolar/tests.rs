use super::*;
use crate::{CameraModel, RotQuaternion};
use approx::assert_relative_eq;

fn pinhole_cam() -> CameraIntrinsics {
    CameraIntrinsics {
        model: CameraModel::Pinhole {
            focal_length_x: 500.0,
            focal_length_y: 500.0,
            principal_point_x: 320.0,
            principal_point_y: 240.0,
        },
        width: 640,
        height: 480,
    }
}

fn fisheye_cam() -> CameraIntrinsics {
    CameraIntrinsics {
        model: CameraModel::OpenCVFisheye {
            focal_length_x: 300.0,
            focal_length_y: 300.0,
            principal_point_x: 320.0,
            principal_point_y: 240.0,
            radial_distortion_k1: 0.05,
            radial_distortion_k2: -0.01,
            radial_distortion_k3: 0.002,
            radial_distortion_k4: 0.0,
        },
        width: 640,
        height: 480,
    }
}

#[test]
fn epipolar_curve_pinhole_satisfies_fundamental_constraint() {
    let cam = pinhole_cam();
    let pose1 = RigidTransform::identity();
    let pose2 = RigidTransform::new(RotQuaternion::identity(), Vector3::new(1.0, 0.0, 0.0));
    let p1 = [400.0, 305.0];
    let curve = plot_epipolar_curve(
        p1,
        &cam,
        &pose1,
        &cam,
        &pose2,
        5.0,
        &EpipolarCurveOptions::default(),
    );
    assert!(curve.len() >= 2);

    let k = cam.intrinsic_matrix();
    let f = compute_fundamental_matrix(
        &k,
        &pose1.to_rotation_matrix(),
        &pose1.translation,
        &k,
        &pose2.to_rotation_matrix(),
        &pose2.translation,
    );
    let p1h = Vector3::new(p1[0], p1[1], 1.0);
    for q in &curve {
        let p2h = Vector3::new(q[0], q[1], 1.0);
        let c = (p2h.transpose() * f * p1h)[(0, 0)];
        assert!(c.abs() < 1e-6, "epipolar constraint = {c}");
    }
}

#[test]
fn epipolar_curve_pinhole_is_two_vertices() {
    // A pinhole epipolar "curve" is exactly a straight line, so the
    // adaptive subdivision should accept the initial chord immediately.
    let cam = pinhole_cam();
    let pose1 = RigidTransform::identity();
    let pose2 = RigidTransform::new(RotQuaternion::identity(), Vector3::new(0.7, 0.2, 0.0));
    let curve = plot_epipolar_curve(
        [400.0, 305.0],
        &cam,
        &pose1,
        &cam,
        &pose2,
        5.0,
        &EpipolarCurveOptions::default(),
    );
    assert_eq!(curve.len(), 2, "pinhole curve should not be subdivided");
}

#[test]
fn epipolar_curve_vertices_are_inside_image_rect() {
    let cam = fisheye_cam();
    let pose1 = RigidTransform::identity();
    let pose2 = RigidTransform::new(RotQuaternion::identity(), Vector3::new(0.5, 0.0, 0.0));
    let curve = plot_epipolar_curve(
        [380.0, 250.0],
        &cam,
        &pose1,
        &cam,
        &pose2,
        5.0,
        &EpipolarCurveOptions::default(),
    );
    assert!(!curve.is_empty());
    for q in &curve {
        assert!(
            q[0] >= 0.0 && q[0] < cam.width as f64,
            "vertex u={} outside [0, {})",
            q[0],
            cam.width
        );
        assert!(
            q[1] >= 0.0 && q[1] < cam.height as f64,
            "vertex v={} outside [0, {})",
            q[1],
            cam.height
        );
    }
}

/// Squared distance from `q` to the polyline `curve`, evaluated against
/// every segment (not just vertices). Returns `f64::INFINITY` for a curve
/// with fewer than 2 vertices.
fn polyline_distance(curve: &[[f64; 2]], q: [f64; 2]) -> f64 {
    if curve.len() < 2 {
        return f64::INFINITY;
    }
    let mut best = f64::INFINITY;
    for w in curve.windows(2) {
        let a = w[0];
        let b = w[1];
        let dx = b[0] - a[0];
        let dy = b[1] - a[1];
        let len2 = dx * dx + dy * dy;
        let d2 = if len2 < 1e-12 {
            (q[0] - a[0]).powi(2) + (q[1] - a[1]).powi(2)
        } else {
            let t = (((q[0] - a[0]) * dx + (q[1] - a[1]) * dy) / len2).clamp(0.0, 1.0);
            let px = a[0] + t * dx;
            let py = a[1] + t * dy;
            (q[0] - px).powi(2) + (q[1] - py).powi(2)
        };
        if d2 < best {
            best = d2;
        }
    }
    best.sqrt()
}

#[test]
fn epipolar_curve_fisheye_subdivides_under_tight_tolerance() {
    // The chord-deviation criterion should produce more than two vertices
    // for any non-trivial fisheye curve when the tolerance is well below
    // the curve's natural sagitta. 0.01 px ensures subdivision triggers
    // regardless of how mild the curvature happens to be.
    let cam = fisheye_cam();
    let pose1 = RigidTransform::identity();
    let pose2 = RigidTransform::new(RotQuaternion::identity(), Vector3::new(0.5, 0.3, 0.0));
    let opts = EpipolarCurveOptions {
        curvature_tolerance: 0.01,
        max_vertices: 256,
    };
    let curve = plot_epipolar_curve([180.0, 360.0], &cam, &pose1, &cam, &pose2, 5.0, &opts);
    assert!(
        curve.len() > 2,
        "fisheye curve at 0.01px tolerance should subdivide; got {} vertices",
        curve.len()
    );
}

#[test]
fn epipolar_curve_passes_through_true_correspondence_fisheye() {
    let cam = fisheye_cam();
    let pose1 = RigidTransform::identity();
    let pose2 = RigidTransform::new(RotQuaternion::identity(), Vector3::new(0.5, 0.1, 0.0));

    let x_world = Point3::new(0.3, -0.2, 4.0);
    let xc1 = pose1.transform_point(&x_world);
    let xc2 = pose2.transform_point(&x_world);
    let (u1, v1) = cam.ray_to_pixel([xc1.x, xc1.y, xc1.z]).unwrap();
    let (u2, v2) = cam.ray_to_pixel([xc2.x, xc2.y, xc2.z]).unwrap();

    let curve = plot_epipolar_curve(
        [u1, v1],
        &cam,
        &pose1,
        &cam,
        &pose2,
        4.0,
        &EpipolarCurveOptions::default(),
    );
    assert!(!curve.is_empty());
    // Measure to the polyline (segments), not just vertices — the adaptive
    // sampler emits sparse vertices where the curve is locally straight,
    // so a nearest-vertex test would be overly strict.
    let d = polyline_distance(&curve, [u2, v2]);
    assert!(
        d < 1.0,
        "polyline is {d}px from true match (curve has {} vertices)",
        curve.len()
    );
}

#[test]
fn epipolar_curve_empty_for_zero_baseline() {
    let cam = pinhole_cam();
    let pose = RigidTransform::identity();
    let curve = plot_epipolar_curve(
        [400.0, 300.0],
        &cam,
        &pose,
        &cam,
        &pose,
        1.0,
        &Default::default(),
    );
    assert!(curve.is_empty());
}

#[test]
fn epipolar_curve_empty_when_ray_entirely_behind_cam2() {
    // Camera 2 rotated 180° about Y and pushed behind camera 1: the
    // back-projected ray of (320,240) — straight down +Z in world — sits
    // entirely behind camera 2, so the in-image predicate is never true.
    let cam = pinhole_cam();
    let pose1 = RigidTransform::identity();
    let pose2 = RigidTransform::new(
        RotQuaternion::from_axis_angle(Vector3::y(), std::f64::consts::PI).unwrap(),
        Vector3::new(0.0, 0.0, -10.0),
    );
    let curve = plot_epipolar_curve(
        [320.0, 240.0],
        &cam,
        &pose1,
        &cam,
        &pose2,
        5.0,
        &EpipolarCurveOptions::default(),
    );
    assert!(
        curve.is_empty(),
        "expected empty polyline, got {} vertices",
        curve.len()
    );
}

#[test]
fn epipolar_curve_respects_max_vertices_cap() {
    let cam = fisheye_cam();
    let pose1 = RigidTransform::identity();
    let pose2 = RigidTransform::new(RotQuaternion::identity(), Vector3::new(0.5, 0.0, 0.0));
    let opts = EpipolarCurveOptions {
        curvature_tolerance: 0.001, // unreachable tightness
        max_vertices: 8,
    };
    let curve = plot_epipolar_curve([120.0, 360.0], &cam, &pose1, &cam, &pose2, 5.0, &opts);
    assert!(!curve.is_empty());
    assert!(
        curve.len() <= opts.max_vertices,
        "vertex count {} exceeds cap {}",
        curve.len(),
        opts.max_vertices
    );
}

#[test]
fn epipolar_curves_batch_matches_scalar() {
    let cam = pinhole_cam();
    let pose1 = RigidTransform::identity();
    let pose2 = RigidTransform::new(RotQuaternion::identity(), Vector3::new(0.3, 0.4, 0.0));
    let pts = [[400.0, 300.0], [200.0, 150.0], [500.0, 400.0]];
    let anchors = [5.0, 3.0, 8.0];
    let opts = EpipolarCurveOptions::default();
    let batch = plot_epipolar_curves_batch(&pts, &anchors, &cam, &pose1, &cam, &pose2, &opts);
    assert_eq!(batch.len(), pts.len());
    for (i, (&p1, &anchor)) in pts.iter().zip(anchors.iter()).enumerate() {
        let scalar = plot_epipolar_curve(p1, &cam, &pose1, &cam, &pose2, anchor, &opts);
        assert_eq!(batch[i], scalar);
    }
}

#[test]
fn test_fundamental_matrix_identity_cameras() {
    // Two cameras at same pose: R_rel = I, t_rel = 0
    // E = [0]_x * I = 0, so F = 0
    let k = Matrix3::new(500.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0);
    let r = Matrix3::identity();
    let t = Vector3::zeros();

    let f = compute_fundamental_matrix(&k, &r, &t, &k, &r, &t);
    assert_relative_eq!(f, Matrix3::zeros(), epsilon = 1e-10);
}

#[test]
fn test_fundamental_matrix_lateral_baseline() {
    // Camera 1 at origin, camera 2 translated along X axis
    let k = Matrix3::new(500.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0);
    let r = Matrix3::identity();
    let t1 = Vector3::zeros();
    let t2 = Vector3::new(1.0, 0.0, 0.0);

    let f = compute_fundamental_matrix(&k, &r, &t1, &k, &r, &t2);

    // F should not be zero
    assert!(f.norm() > 1e-10);

    // F should be rank 2 (det = 0)
    assert_relative_eq!(f.determinant(), 0.0, epsilon = 1e-10);

    // Epipolar constraint: for a 3D point, its projections satisfy p2^T F p1 = 0
    // Point at (0, 0, 10): projects to (320, 240) in cam1, (370, 240) in cam2
    let p1 = Vector3::new(320.0, 240.0, 1.0);
    let p2 = Vector3::new(370.0, 240.0, 1.0);
    let epipolar_constraint = p2.transpose() * f * p1;
    assert_relative_eq!(epipolar_constraint[(0, 0)], 0.0, epsilon = 1e-6);
}

#[test]
fn test_fundamental_matrix_vertical_baseline() {
    let k = Matrix3::new(500.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0);
    let r = Matrix3::identity();
    let t1 = Vector3::zeros();
    let t2 = Vector3::new(0.0, 1.0, 0.0);

    let f = compute_fundamental_matrix(&k, &r, &t1, &k, &r, &t2);

    assert!(f.norm() > 1e-10);
    assert_relative_eq!(f.determinant(), 0.0, epsilon = 1e-10);
}

#[test]
fn test_epipole_at_infinity() {
    // F = [[0,0,0],[0,0,-1],[0,1,0]]
    // Null space of F is [1,0,0] (at infinity), so pair should return None.
    let f = Matrix3::from_row_slice(&[0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0]);
    assert!(compute_epipole_pair(&f).is_none());
}

#[test]
fn test_epipole_from_pure_translation() {
    // Pure translation: P1 = [I|0], P2 = [I|t] with t = (2, 3, 1).
    // F = [t]_x is skew-symmetric, so both null(F) and null(F^T) are t.
    // Both epipoles dehomogenize to t/t_z = (2, 3).
    let f = Matrix3::from_row_slice(&[0.0, -1.0, 3.0, 1.0, 0.0, -2.0, -3.0, 2.0, 0.0]);

    let (e1, e2) = compute_epipole_pair(&f).expect("both epipoles should be finite");
    assert!((e1[0] - 2.0).abs() < 1e-6, "e1.x = {}", e1[0]);
    assert!((e1[1] - 3.0).abs() < 1e-6, "e1.y = {}", e1[1]);
    assert!((e2[0] - 2.0).abs() < 1e-6, "e2.x = {}", e2[0]);
    assert!((e2[1] - 3.0).abs() < 1e-6, "e2.y = {}", e2[1]);
}

#[test]
fn test_epipole_from_diagonal_translation() {
    // Pure translation with t = (5, -4, 2).
    // Both epipoles = t/t_z = (2.5, -2).
    let f = Matrix3::from_row_slice(&[0.0, -2.0, -4.0, 2.0, 0.0, -5.0, 4.0, 5.0, 0.0]);

    let (e1, e2) = compute_epipole_pair(&f).expect("both epipoles should be finite");
    assert!((e1[0] - 2.5).abs() < 1e-6, "e1.x = {}", e1[0]);
    assert!((e1[1] - (-2.0)).abs() < 1e-6, "e1.y = {}", e1[1]);
    assert!((e2[0] - 2.5).abs() < 1e-6, "e2.x = {}", e2[0]);
    assert!((e2[1] - (-2.0)).abs() < 1e-6, "e2.y = {}", e2[1]);
}

#[test]
fn test_single_epipole_lateral_motion() {
    // For pure lateral motion, epipole is at infinity.
    let k = Matrix3::new(500.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0);
    let r = Matrix3::identity();
    let t1 = Vector3::zeros();
    let t2 = Vector3::new(1.0, 0.0, 0.0);
    let f = compute_fundamental_matrix(&k, &r, &t1, &k, &r, &t2);

    let (_epipole, is_at_infinity) = compute_epipole(&f);
    assert!(is_at_infinity);
}

#[test]
fn test_fundamental_equals_essential_when_k_identity() {
    // When K = I, F = E = [t]_x R_rel
    let k = Matrix3::identity();
    let r = Matrix3::identity();
    let t1 = Vector3::zeros();
    let t2 = Vector3::new(1.0, 0.0, 0.0);

    let f = compute_fundamental_matrix(&k, &r, &t1, &k, &r, &t2);

    // E = [t_rel]_x * R_rel
    //   = [t2]_x * I
    //   = skew(t2)
    let e = skew_symmetric(&t2);

    // F should equal E when K = I
    for i in 0..3 {
        for j in 0..3 {
            assert!(
                (f[(i, j)] - e[(i, j)]).abs() < 1e-10,
                "F[{i},{j}] = {} != E[{i},{j}] = {}",
                f[(i, j)],
                e[(i, j)]
            );
        }
    }
}

#[test]
fn test_essential_matrix_epipolar_constraint() {
    // Verify p2^T E p1 = 0 for a known 3D point
    let k = Matrix3::new(500.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0);
    let r1 = Matrix3::identity();
    let t1 = Vector3::zeros();
    let r2 = Matrix3::identity();
    let t2 = Vector3::new(1.0, 0.0, 0.0);

    let f = compute_fundamental_matrix(&k, &r1, &t1, &k, &r2, &t2);

    // 3D point at (2, 3, 10):
    // cam1: K * (2, 3, 10) / 10 = K * (0.2, 0.3, 1) = (420, 390, 1) (homogeneous)
    // cam2: K * (2+1, 3, 10) / 10 = K * (0.3, 0.3, 1) = (470, 390, 1)
    let p1 = Vector3::new(500.0 * 0.2 + 320.0, 500.0 * 0.3 + 240.0, 1.0);
    let p2 = Vector3::new(500.0 * 0.3 + 320.0, 500.0 * 0.3 + 240.0, 1.0);

    let constraint = p2.transpose() * f * p1;
    assert!(
        constraint[(0, 0)].abs() < 1e-6,
        "Epipolar constraint p2^T F p1 should be ≈ 0, got {}",
        constraint[(0, 0)]
    );
}

#[test]
fn test_epipole_from_90_degree_rotation() {
    // Camera 2 rotated 90° around Y from camera 1
    let k = Matrix3::new(500.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0);
    let r1 = Matrix3::identity();
    let t1 = Vector3::zeros();

    let angle = std::f64::consts::FRAC_PI_2;
    let r2 = Matrix3::new(
        angle.cos(),
        0.0,
        angle.sin(),
        0.0,
        1.0,
        0.0,
        -angle.sin(),
        0.0,
        angle.cos(),
    );
    let t2 = Vector3::new(1.0, 0.0, 0.0);

    let f = compute_fundamental_matrix(&k, &r1, &t1, &k, &r2, &t2);

    // F should be rank 2
    assert!(f.norm() > 1e-10);
    assert!(f.determinant().abs() < 1e-6, "F should be rank 2");
}

#[test]
fn test_single_epipole_forward_motion() {
    // For forward motion along Z, epipole is at principal point.
    let k = Matrix3::new(500.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0);
    let r = Matrix3::identity();
    let t1 = Vector3::zeros();
    let t2 = Vector3::new(0.0, 0.0, 1.0);
    let f = compute_fundamental_matrix(&k, &r, &t1, &k, &r, &t2);

    let (epipole, is_at_infinity) = compute_epipole(&f);
    assert!(!is_at_infinity);
    assert!((epipole[0] - 320.0).abs() < 1.0);
    assert!((epipole[1] - 240.0).abs() < 1.0);
}
