// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The rolling-shutter step: parity with a camera without rates, the forward
//! map against the formulas of `specs/core/camera/sfmtool-pinhole-kernels.md`
//! written out independently in the `.sfmr` frame, the round trip through the
//! fixed-point inverse, and the analytic ray Jacobian against differences.

use crate::camera::{
    CameraIntrinsics, CameraModel, IntrinsicAdjustment, RollingShutter, RollingShutterScanlines,
};
use crate::geometry::RigidTransform;

const SCANLINES: [RollingShutterScanlines; 2] = [
    RollingShutterScanlines::Rows,
    RollingShutterScanlines::Columns,
];

/// Ten times the rates measured on a handheld phone capture (about `2·10⁻³`).
const RATES: [f64; 3] = [2e-2, -1.5e-2, 1e-2];

/// An `SFMTOOL_PINHOLE` camera of the measured phone's size and focal, with an
/// eight-coefficient barrel spline.
fn phone_camera() -> CameraIntrinsics {
    let camera = CameraIntrinsics {
        model: CameraModel::SfmtoolPinhole {
            focal_length: 1447.8,
            principal_point_x: 1021.5,
            principal_point_y: 765.0,
            bspline_rho_max: 0.84,
            bspline: vec![
                -0.002, -0.006, -0.012, -0.018, -0.024, -0.03, -0.036, -0.042,
            ],
        },
        width: 2040,
        height: 1536,
        rolling_shutter: None,
    };
    let CameraModel::SfmtoolPinhole {
        bspline,
        bspline_rho_max,
        ..
    } = &camera.model
    else {
        unreachable!()
    };
    assert!(crate::camera::distortion::bspline::bspline_is_monotone(
        bspline,
        *bspline_rho_max,
        *bspline_rho_max
    ));
    camera
}

/// The phone camera as one image of it reads: a focal scale, a principal-point
/// offset and a radial correction of ten times the measured size, and the
/// rates [`RATES`] on `scanlines`.
fn image_camera(scanlines: RollingShutterScanlines) -> CameraIntrinsics {
    phone_camera().with_adjustment(
        &IntrinsicAdjustment {
            focal: 0.013,
            principal_point: [0.004, -0.006],
            radial_bspline: [0.02, -0.03],
            rolling_shutter: RATES,
        },
        Some(scanlines),
    )
}

/// A grid of pixels over the whole image, corners included.
fn pixel_grid(camera: &CameraIntrinsics, n: usize) -> Vec<[f64; 2]> {
    let (w, h) = (f64::from(camera.width), f64::from(camera.height));
    let mut out = Vec::new();
    for i in 0..n {
        for j in 0..n {
            out.push([w * i as f64 / (n - 1) as f64, h * j as f64 / (n - 1) as f64]);
        }
    }
    out
}

/// The `.sfmr`-frame reference of the whole map, written from the formulas
/// alone: the radial map of the camera without its rolling shutter gives the
/// pixel before the step, `t` is that pixel's distance from the principal
/// point's scanline over `min(width, height)`, and the step adds
/// `t·(ẋ, ẏ)` to `(x_d, y_d)` before `u = cx + f·x_r`, `v = cy − f·y_r`.
fn reference_pixel(camera: &CameraIntrinsics, ray: [f64; 3]) -> Option<(f64, f64)> {
    let rolling_shutter = camera.rolling_shutter.expect("a rolling shutter");
    let lens = CameraIntrinsics {
        rolling_shutter: None,
        ..camera.clone()
    };
    let (u0, v0) = lens.ray_to_pixel(ray)?;
    let f = camera.focal_lengths().0;
    let (cx, cy) = camera.principal_point();
    let m = f64::from(camera.width.min(camera.height));
    let (x_d, y_d) = ((u0 - cx) / f, (cy - v0) / f);
    let (x, y) = (ray[0] / -ray[2], ray[1] / -ray[2]);
    let t = match rolling_shutter.scanlines {
        RollingShutterScanlines::Rows => (v0 - cy) / m,
        RollingShutterScanlines::Columns => (u0 - cx) / m,
    };
    let [wx, wy, wz] = rolling_shutter.rates;
    let x_dot = wy * (1.0 + x * x) - wx * x * y + wz * y;
    let y_dot = -wx * (1.0 + y * y) + wy * x * y - wz * x;
    let (x_r, y_r) = (x_d + t * x_dot, y_d + t * y_dot);
    Some((cx + f * x_r, cy - f * y_r))
}

/// Rays through the pixel grid of the camera without its rolling shutter:
/// directions that cover the image.
fn rays_over_image(camera: &CameraIntrinsics) -> Vec<[f64; 3]> {
    let lens = CameraIntrinsics {
        rolling_shutter: None,
        ..camera.clone()
    };
    pixel_grid(&lens, 15)
        .into_iter()
        .map(|[u, v]| lens.pixel_to_ray(u, v))
        .collect()
}

#[test]
fn zero_rates_project_unproject_and_differentiate_bit_for_bit() {
    let plain = phone_camera();
    for scanlines in SCANLINES {
        let zero = CameraIntrinsics {
            rolling_shutter: Some(RollingShutter {
                scanlines,
                rates: [0.0; 3],
            }),
            ..plain.clone()
        };
        assert!(zero.rolling_shutter_step().is_none());
        for [u, v] in pixel_grid(&plain, 9) {
            let bits = |r: [f64; 3]| r.map(f64::to_bits);
            assert_eq!(
                bits(zero.pixel_to_ray(u, v)),
                bits(plain.pixel_to_ray(u, v))
            );
            let (a, b) = (zero.unproject(u, v), plain.unproject(u, v));
            assert_eq!(
                (a.0.to_bits(), a.1.to_bits()),
                (b.0.to_bits(), b.1.to_bits())
            );
            let ray = plain.pixel_to_ray(u, v);
            assert_eq!(zero.ray_to_pixel(ray), plain.ray_to_pixel(ray));
            assert_eq!(
                zero.ray_to_pixel_with_jacobian(ray),
                plain.ray_to_pixel_with_jacobian(ray)
            );
            let (x, y) = plain.unproject(u, v);
            assert_eq!(zero.project(x, y), plain.project(x, y));
        }
    }
    // A zero adjustment gives the camera back exactly, whatever the scanlines.
    for scanlines in SCANLINES {
        assert_eq!(
            plain.with_adjustment(&IntrinsicAdjustment::default(), Some(scanlines)),
            plain
        );
    }
}

#[test]
fn the_forward_map_matches_the_formulas_in_the_sfmr_frame() {
    for scanlines in SCANLINES {
        let camera = image_camera(scanlines);
        let mut largest_shift = 0.0f64;
        for ray in rays_over_image(&camera) {
            let (u, v) = camera.ray_to_pixel(ray).expect("in front");
            let (ru, rv) = reference_pixel(&camera, ray).expect("in front");
            assert!(
                (u - ru).abs() < 1e-9 && (v - rv).abs() < 1e-9,
                "{scanlines:?}: ({u}, {v}) against the reference ({ru}, {rv})"
            );
            let lens = CameraIntrinsics {
                rolling_shutter: None,
                ..camera.clone()
            };
            let (u0, v0) = lens.ray_to_pixel(ray).unwrap();
            largest_shift = largest_shift.max((u - u0).hypot(v - v0));
            // `project` reads the same map from the undistorted point.
            let (x, y) = (ray[0] / -ray[2], ray[1] / -ray[2]);
            let (pu, pv) = camera.project(x, y);
            assert!((pu - u).abs() < 1e-9 && (pv - v).abs() < 1e-9);
        }
        // Rates ten times the measured size move the edge by tens of pixels,
        // so the comparison above is not of two identities.
        assert!(largest_shift > 5.0, "{scanlines:?}: {largest_shift}");
    }
}

#[test]
fn the_velocity_is_the_image_motion_of_a_small_camera_rotation() {
    // With no distortion and no rolling shutter, a camera turning at `ω` moves
    // a fixed point as `Ṗ = −ω × P`; its image-plane velocity is the
    // derivative of `(Px, Py)/(−Pz)`, which the step's `ṗ` must equal.
    let camera = image_camera(RollingShutterScanlines::Rows);
    let step = camera.rolling_shutter_step().unwrap();
    let w = nalgebra::Vector3::from(RATES);
    for p in [
        [0.3, -0.2, -1.0],
        [-0.5, 0.4, -1.3],
        [0.0, 0.0, -2.0],
        [0.7, 0.6, -1.1],
    ] {
        let p = nalgebra::Vector3::from(p);
        let image = |q: nalgebra::Vector3<f64>| (q.x / -q.z, q.y / -q.z);
        let h = 1e-6;
        let (xp, yp) = image(p - h * w.cross(&p));
        let (xm, ym) = image(p + h * w.cross(&p));
        let (x_dot, y_dot) = ((xp - xm) / (2.0 * h), (yp - ym) / (2.0 * h));
        let (x, y) = image(p);
        // `velocity` takes and gives the kernels' optical frame, y down.
        let [vx, vy] = step.velocity(x, -y);
        assert!((vx - x_dot).abs() < 1e-8, "{vx} against {x_dot}");
        assert!((-vy - y_dot).abs() < 1e-8, "{} against {y_dot}", -vy);
    }
}

#[test]
fn pixel_to_ray_inverts_ray_to_pixel_over_the_image() {
    for scanlines in SCANLINES {
        let camera = image_camera(scanlines);
        let mut worst = 0.0f64;
        for [u, v] in pixel_grid(&camera, 21) {
            let ray = camera.pixel_to_ray(u, v);
            let (u2, v2) = camera.ray_to_pixel(ray).expect("in front");
            worst = worst.max((u2 - u).hypot(v2 - v));
            let (x, y) = camera.unproject(u, v);
            let (u3, v3) = camera.project(x, y);
            worst = worst.max((u3 - u).hypot(v3 - v));
        }
        assert!(worst < 1e-9, "{scanlines:?}: {worst} px");
    }
}

#[test]
fn the_ray_jacobian_matches_central_differences() {
    for scanlines in SCANLINES {
        let camera = image_camera(scanlines);
        for ray in rays_over_image(&camera) {
            let ((u, v), jac) = camera.ray_to_pixel_with_jacobian(ray).expect("in front");
            assert_eq!(Some((u, v)), camera.ray_to_pixel(ray));
            let n = (ray[0] * ray[0] + ray[1] * ray[1] + ray[2] * ray[2]).sqrt();
            let h = 1e-6 * n;
            for c in 0..3 {
                let (mut plus, mut minus) = (ray, ray);
                plus[c] += h;
                minus[c] -= h;
                let (up, vp) = camera.ray_to_pixel(plus).unwrap();
                let (um, vm) = camera.ray_to_pixel(minus).unwrap();
                let (du, dv) = ((up - um) / (2.0 * h), (vp - vm) / (2.0 * h));
                let scale = 1.0 + jac[0][c].abs().max(jac[1][c].abs());
                assert!(
                    (jac[0][c] - du).abs() < 1e-6 * scale && (jac[1][c] - dv).abs() < 1e-6 * scale,
                    "{scanlines:?} column {c}: ({}, {}) against ({du}, {dv})",
                    jac[0][c],
                    jac[1][c]
                );
            }
        }
    }
}

#[test]
fn the_batch_forms_and_the_ray_grid_agree_with_the_scalar_forms() {
    for scanlines in SCANLINES {
        let camera = image_camera(scanlines);
        let pixels = pixel_grid(&camera, 7);
        let rays = camera.pixel_to_ray_batch(&pixels);
        for (p, r) in pixels.iter().zip(&rays) {
            assert_eq!(*r, camera.pixel_to_ray(p[0], p[1]));
        }
        let back = camera.ray_to_pixel_batch(&rays);
        for (r, b) in rays.iter().zip(&back) {
            assert_eq!(*b, camera.ray_to_pixel(*r).map(|(u, v)| [u, v]));
        }
        let points = camera.unproject_batch(&pixels);
        for (p, x) in pixels.iter().zip(&points) {
            let (a, b) = camera.unproject(p[0], p[1]);
            assert_eq!(*x, [a, b]);
        }
        for (x, q) in points.iter().zip(camera.project_batch(&points)) {
            let (a, b) = camera.project(x[0], x[1]);
            assert_eq!(q, [a, b]);
        }
        // The ray grid's per-node projection.
        let (cols, rows) = (24u32, 20u32);
        let origin = [-0.5, 0.35, -1.0];
        let (col_step, row_step) = ([0.04, 0.0, 0.0], [0.0, -0.035, 0.0]);
        let mut out = vec![0.0f32; 2 * (cols * rows) as usize];
        camera.ray_to_pixel_grid(origin, col_step, row_step, cols, rows, &mut out);
        for row in 0..rows as usize {
            for col in 0..cols as usize {
                let ray = [
                    origin[0] + col as f64 * col_step[0],
                    origin[1] + row as f64 * row_step[1],
                    origin[2],
                ];
                let k = 2 * (row * cols as usize + col);
                let (u, v) = camera.ray_to_pixel(ray).unwrap();
                assert_eq!([out[k], out[k + 1]], [u as f32, v as f32]);
            }
        }
    }
}

#[test]
fn a_point_at_infinity_projects_as_its_direction() {
    let camera = image_camera(RollingShutterScanlines::Rows);
    let pose =
        RigidTransform::from_axis_angle(nalgebra::Vector3::new(0.3, -1.0, 0.2), 0.1).unwrap();
    let pose = RigidTransform::new(pose.rotation, nalgebra::Vector3::new(0.3, -0.2, 4.0));
    let direction = nalgebra::Vector3::new(0.1, -0.05, -1.0).normalize();
    let projected = camera
        .project_homogeneous(&pose, direction, 0.0)
        .expect("in front");
    let rotated = pose.to_rotation_matrix() * direction;
    let (u, v) = camera
        .ray_to_pixel([rotated.x, rotated.y, rotated.z])
        .unwrap();
    assert_eq!(projected, [u, v]);
    // The translation does not move a direction, and the ray round-trips.
    let back = camera.pixel_to_ray(u, v);
    let back = nalgebra::Vector3::new(back[0], back[1], back[2]);
    assert!((back - rotated).norm() < 1e-12);
    assert!(camera.project_homogeneous(&pose, -direction, 0.0).is_none());
}

#[test]
#[should_panic(expected = "only on an SFMTOOL_PINHOLE camera")]
fn rates_on_another_model_panic_rather_than_project_without_them() {
    let camera = CameraIntrinsics {
        model: CameraModel::SimplePinhole {
            focal_length: 500.0,
            principal_point_x: 320.0,
            principal_point_y: 240.0,
        },
        width: 640,
        height: 480,
        rolling_shutter: Some(RollingShutter {
            scanlines: RollingShutterScanlines::Rows,
            rates: RATES,
        }),
    };
    let _ = camera.ray_to_pixel([0.1, 0.1, -1.0]);
}
