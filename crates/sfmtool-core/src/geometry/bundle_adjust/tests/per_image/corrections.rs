// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The per-image corrections: each image's radial correction and
//! rolling-shutter rates solved under a prior centred on its camera's mean,
//! that mean held through the round, the mean radial correction moved into the
//! camera's spline after it, and the corrections' step guards.

use super::*;
use crate::camera::distortion::bspline::delta as spline_delta;

/// A 60° `SFMTOOL_PINHOLE` of 640×480 (short side 480) with a mild barrel
/// spline on a domain that reaches past the image's corners.
fn spline_pinhole() -> CameraIntrinsics {
    CameraIntrinsics {
        model: CameraModel::SfmtoolPinhole {
            focal_length: 320.0 / 30f64.to_radians().tan(),
            principal_point_x: 326.0,
            principal_point_y: 236.0,
            bspline_rho_max: 0.8,
            bspline: vec![-0.001, -0.004, -0.009, -0.016, -0.025, -0.036],
        },
        width: 640,
        height: 480,
        rolling_shutter: None,
    }
}

/// The camera's spline coefficients and domain end.
fn spline_of(cam: &CameraIntrinsics) -> (Vec<f64>, f64) {
    let (c, d_max, _) = cam.model.radial_spline().expect("a spline model");
    (c.to_vec(), d_max)
}

/// The release of each image's radial correction and rolling-shutter rates.
const CORRECTIONS: CameraRelease = CameraRelease {
    image_radial_bspline: true,
    image_rolling_shutter: true,
    ..CameraRelease::HELD
};

const BOTH_SCANLINES: [RollingShutterScanlines; 2] = [
    RollingShutterScanlines::Rows,
    RollingShutterScanlines::Columns,
];

/// Run the per-image release over `s` from the camera `start` with every
/// release flag on, narrowed by `release`, the images' rates read on
/// `scanlines`, every image starting at `start_adj` (zero where empty).
fn run_corrections(
    s: &mut Scene,
    start: &CameraIntrinsics,
    start_adj: &[IntrinsicAdjustment],
    release: CameraRelease,
    scanlines: Option<RollingShutterScanlines>,
    prior: Option<&ImagePriorOptions>,
    schedule: &[BaSchedule],
) -> BundleAdjustment {
    let n_img = s.quats.len();
    let adjustments: Vec<IntrinsicAdjustment> = if start_adj.is_empty() {
        vec![IntrinsicAdjustment::default(); n_img]
    } else {
        start_adj.to_vec()
    };
    let releases = [release];
    let scanlines = [scanlines];
    bundle_adjust(
        &BaCameras {
            cameras: std::slice::from_ref(start),
            image_camera: Cow::Owned(vec![0; n_img]),
            releases: Some(&releases),
            image_adjustments: Some(Cow::Owned(adjustments)),
            rolling_shutter_scanlines: Some(&scanlines),
        },
        &mut s.quats,
        &mut s.trans,
        &mut s.points,
        &s.uv,
        &s.obs_img,
        &s.obs_pt,
        None,
        None,
        FreePointPolicy::default(),
        None,
        DEFAULT_PROTECTED_LOSS_SCALE,
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        prior,
        schedule,
        60,
        2,
        12,
        &Progress::none(),
    )
}

/// `count` corrections drawn about zero, each component of the radial
/// correction with spread `tau_e` and each rate with spread `tau_w`, with their
/// mean over the images removed so that it is exactly zero.
fn correction_draws(count: usize, tau_e: f64, tau_w: f64, salt: u64) -> Vec<IntrinsicAdjustment> {
    let mut out: Vec<IntrinsicAdjustment> = (0..count)
        .map(|i| IntrinsicAdjustment {
            radial_bspline: [tau_e * gaussian(i, salt + 1), tau_e * gaussian(i, salt + 2)],
            rolling_shutter: [
                tau_w * gaussian(i, salt + 3),
                tau_w * gaussian(i, salt + 4),
                tau_w * gaussian(i, salt + 5),
            ],
            ..IntrinsicAdjustment::default()
        })
        .collect();
    let n = count as f64;
    let mean_e: [f64; 2] =
        std::array::from_fn(|k| out.iter().map(|a| a.radial_bspline[k]).sum::<f64>() / n);
    let mean_w: [f64; 3] =
        std::array::from_fn(|k| out.iter().map(|a| a.rolling_shutter[k]).sum::<f64>() / n);
    for a in &mut out {
        for (e, m) in a.radial_bspline.iter_mut().zip(mean_e) {
            *e -= m;
        }
        for (w, m) in a.rolling_shutter.iter_mut().zip(mean_w) {
            *w -= m;
        }
    }
    out
}

/// The root-mean-square of `value(got) − value(want)` over images and
/// components, and of `value(want)` alone.
fn rms_error(
    got: &[IntrinsicAdjustment],
    want: &[IntrinsicAdjustment],
    value: impl Fn(&IntrinsicAdjustment) -> Vec<f64>,
) -> (f64, f64) {
    let (mut err, mut size, mut n) = (0.0, 0.0, 0.0);
    for (g, w) in got.iter().zip(want) {
        for (a, b) in value(g).iter().zip(value(w)) {
            err += (a - b).powi(2);
            size += b * b;
            n += 1.0;
        }
    }
    ((err / n).sqrt(), (size / n).sqrt())
}

/// The mean of each image's `value` over `adjustments`.
fn mean_of<const K: usize>(
    adjustments: &[IntrinsicAdjustment],
    value: impl Fn(&IntrinsicAdjustment) -> [f64; K],
) -> [f64; K] {
    let n = adjustments.len() as f64;
    std::array::from_fn(|k| adjustments.iter().map(|a| value(a)[k]).sum::<f64>() / n)
}

#[test]
fn correction_columns_match_a_central_difference() {
    // One image at the identity pose and one point toward a corner, read
    // through a camera whose image carries a radial correction and rates, on
    // rows and on columns, and through one whose image carries neither, which
    // is where a fresh release starts. The five correction columns agree with
    // a central difference of the projection in each case.
    let cam = spline_pinhole();
    let point = Vector3::new(1.5, -1.1, -4.0);
    let held = IntrinsicAdjustment {
        focal: 0.012,
        principal_point: [0.008, -0.006],
        radial_bspline: [0.004, -0.006],
        rolling_shutter: [2e-2, -1.5e-2, 1e-2],
    };
    let h = 1e-6;
    for scanlines in BOTH_SCANLINES {
        for adjustment in [held, IntrinsicAdjustment::default()] {
            let image = cam.with_adjustment(&adjustment, Some(scanlines));
            let (u0, v0) = image
                .ray_to_pixel([point.x, point.y, point.z])
                .expect("in front");
            let (cx, cy) = image.principal_point();
            let lens = LinLens {
                cam: &image,
                analytic: true,
                cx,
                cy,
                f: cam.focal_lengths().0,
                f_image: image.focal_lengths().0,
                m: 480.0,
                opt_f: false,
                opt_k1: false,
                opt_bspline: false,
                opt_pp: false,
                opt_image_f: false,
                opt_image_pp: false,
                opt_image_radial_bspline: true,
                opt_image_rolling_shutter: true,
                scanlines: Some(scanlines),
                rho_max: 0.8,
                n_coeffs: 0,
                d_max: 0.0,
                radial: SplineRadial::ImagePlaneRadius,
                slot0: 0,
            };
            let st = LinState {
                lenses: std::slice::from_ref(&lens),
                ci_cam: &[0],
                q: &[UnitQuaternion::identity()],
                t: &[Vector3::zeros()],
                xp: &[point],
                bases: &[(Vector3::zeros(), Vector3::zeros())],
                cp_dir: &[false],
                cp_held: &[false],
                cp_origin: &[None],
                cp_inverse: &[],
                inv_depth: &[],
                uv: &[[u0 + 0.5, v0 - 0.5]],
                n_shared: lens_block_width(0),
                img_cols: POSE_COLS + IMG_PARAMS,
            };
            let b = observation_blocks::<BASE_CORR_CAM_COLS>(&st, 0, 0, 0, 1e18);
            let pose_c = BASE_CORR_CAM_COLS - (POSE_COLS + IMG_PARAMS);
            for k in 0..IMG_CORR_COLS {
                assert_eq!(
                    b.idx[pose_c + IMG_E_SLOT + k],
                    lens_block_width(0) + IMG_E_SLOT + k,
                    "correction column {k}'s slot"
                );
            }
            for k in 0..IMG_CORR_COLS {
                let moved = |d: f64| {
                    let mut a = adjustment;
                    if k < 2 {
                        a.radial_bspline[k] += d;
                    } else {
                        a.rolling_shutter[k - 2] += d;
                    }
                    cam.with_adjustment(&a, Some(scanlines))
                        .ray_to_pixel([point.x, point.y, point.z])
                        .expect("in front")
                };
                let (up, vp) = moved(h);
                let (um, vm) = moved(-h);
                let fd = [(up - um) / (2.0 * h), (vp - vm) / (2.0 * h)];
                for (row, want) in fd.iter().enumerate() {
                    let got = b.cam_j[(row, pose_c + IMG_E_SLOT + k)];
                    assert!(
                        (got - want).abs() <= 1e-5 * want.abs().max(1.0),
                        "{scanlines:?}, {adjustment:?}: column {k}, row {row} is {got}, the \
                         difference {want}"
                    );
                }
            }
        }
    }
}

#[test]
fn planted_corrections_are_recovered_under_rows_and_under_columns() {
    // Thirty images, each seeing 80 points through its own radial correction
    // and rates, drawn about a mean of zero, with 0.1 px of noise. The solve
    // starts every image at zero with the shared camera at the truth, and
    // recovers each image's correction and rates within their noise, on rows
    // and on columns. The mean rate stays at its start, zero, and the mean
    // radial correction is moved into the spline, which barely moves.
    let truth = spline_pinhole();
    for (n, scanlines) in BOTH_SCANLINES.into_iter().enumerate() {
        let planted = correction_draws(30, 3e-3, 4e-3, 70 + 10 * n as u64);
        let mut s = per_image_scene_on(&truth, Some(scanlines), 30, 400, &planted, 0.1, 21, |_| 80);
        perturb_poses(&mut s);
        let out = run_corrections(
            &mut s,
            &truth,
            &[],
            CORRECTIONS,
            Some(scanlines),
            None,
            &DEFAULT_SCHEDULE,
        );
        let (err_e, size_e) = rms_error(&out.image_adjustments, &planted, |a| {
            a.radial_bspline.to_vec()
        });
        let (err_w, size_w) = rms_error(&out.image_adjustments, &planted, |a| {
            a.rolling_shutter.to_vec()
        });
        assert!(
            err_e < 0.25 * size_e,
            "{scanlines:?}: radial correction off by {err_e} (rms), planted {size_e}"
        );
        assert!(
            err_w < 0.15 * size_w,
            "{scanlines:?}: rates off by {err_w} (rms), planted {size_w}"
        );
        let mean_w = mean_of(&out.image_adjustments, |a| a.rolling_shutter);
        assert!(
            mean_w.iter().all(|w| w.abs() < 1e-7),
            "{scanlines:?}: mean rate {mean_w:?}"
        );
        let mean_e = mean_of(&out.image_adjustments, |a| a.radial_bspline);
        assert!(
            mean_e.iter().all(|e| e.abs() < 1e-15),
            "{scanlines:?}: mean radial correction {mean_e:?}"
        );
        let (c_out, _) = spline_of(&out.cameras[0]);
        let (c_in, _) = spline_of(&truth);
        for (a, b) in c_out.iter().zip(&c_in) {
            assert!((a - b).abs() < 1e-7, "{scanlines:?}: spline {c_out:?}");
        }
        let worst = out
            .residual_norms
            .iter()
            .copied()
            .filter(|r| r.is_finite())
            .fold(0.0f64, f64::max);
        assert!(worst < 2.0, "{scanlines:?}: largest residual {worst} px");
        let prior = out.image_priors[0].expect("a per-image release");
        assert!(prior.derived);
        assert!(prior.widths.focal.is_nan() && prior.widths.principal_point.is_nan());
        assert!(
            prior.widths.radial_bspline.is_finite() && prior.widths.rolling_shutter.is_finite()
        );
    }
}

#[test]
fn the_camera_spline_takes_a_common_radial_correction() {
    // Every image's radial correction carries a common part, (4, 8)·10⁻³ at
    // the domain's two coefficients, and the camera's spline is released with
    // them. The common part is the camera's: the mean correction is held
    // through each round and moved into the spline after it, so each image's
    // own spline comes out as it was planted while its correction is only its
    // difference from the mean.
    let truth = spline_pinhole();
    let (c_truth, rho_max) = spline_of(&truth);
    let f = truth.focal_lengths().0;
    let common = [0.004, 0.008];
    let planted: Vec<IntrinsicAdjustment> = correction_draws(30, 3e-3, 0.0, 90)
        .into_iter()
        .map(|a| IntrinsicAdjustment {
            radial_bspline: [
                a.radial_bspline[0] + common[0],
                a.radial_bspline[1] + common[1],
            ],
            rolling_shutter: [0.0; 3],
            ..a
        })
        .collect();
    let mut s = per_image_scene_on(&truth, None, 30, 400, &planted, 0.1, 21, |_| 80);
    perturb_poses(&mut s);
    let release = CameraRelease {
        distortion: true,
        image_radial_bspline: true,
        ..CameraRelease::HELD
    };
    let out = run_corrections(&mut s, &truth, &[], release, None, None, &DEFAULT_SCHEDULE);
    let (c_out, _) = spline_of(&out.cameras[0]);
    let mean_e = mean_of(&out.image_adjustments, |a| a.radial_bspline);
    assert!(
        mean_e.iter().all(|e| e.abs() < 1e-15),
        "mean radial correction {mean_e:?}"
    );
    // The camera took the common part.
    let shifted = image_spline(&c_truth, common);
    for &rho in &[0.3, 0.5, 0.7] {
        let got = spline_delta(&c_out, rho_max, rho);
        let want = spline_delta(&shifted, rho_max, rho);
        assert!(
            f * (got - want).abs() < 0.3,
            "the camera's δ({rho}) is {got}, the truth's with the common part {want}"
        );
    }
    // Each image reads its own planted spline.
    for (i, (got, want)) in out.image_adjustments.iter().zip(&planted).enumerate() {
        let image = image_spline(&c_out, got.radial_bspline);
        let planted = image_spline(&c_truth, want.radial_bspline);
        for &rho in &[0.3, 0.5, 0.7] {
            let (a, b) = (
                spline_delta(&image, rho_max, rho),
                spline_delta(&planted, rho_max, rho),
            );
            assert!(
                f * (a - b).abs() < 0.5,
                "image {i}: δ({rho}) is {a}, planted {b}"
            );
        }
    }
}

#[test]
fn the_mean_rate_is_held_at_its_start() {
    // The planted rates share a common part of 3·10⁻³ in ωx, and every image
    // starts at rates whose mean is not zero. The mean is held where it
    // started, so the common part the data ask for is not taken up by it,
    // while the images' differences from the mean are recovered.
    let truth = spline_pinhole();
    let scanlines = RollingShutterScanlines::Rows;
    let draws = correction_draws(30, 0.0, 4e-3, 110);
    let planted: Vec<IntrinsicAdjustment> = draws
        .iter()
        .map(|a| IntrinsicAdjustment {
            rolling_shutter: [
                a.rolling_shutter[0] + 3e-3,
                a.rolling_shutter[1],
                a.rolling_shutter[2],
            ],
            ..*a
        })
        .collect();
    let mut s = per_image_scene_on(&truth, Some(scanlines), 30, 400, &planted, 0.3, 21, |_| 80);
    perturb_poses(&mut s);
    let start_rates = [1e-3, -5e-4, 2e-4];
    let start: Vec<IntrinsicAdjustment> = (0..30)
        .map(|_| IntrinsicAdjustment {
            rolling_shutter: start_rates,
            ..IntrinsicAdjustment::default()
        })
        .collect();
    let release = CameraRelease {
        image_rolling_shutter: true,
        ..CameraRelease::HELD
    };
    let out = run_corrections(
        &mut s,
        &truth,
        &start,
        release,
        Some(scanlines),
        None,
        &DEFAULT_SCHEDULE,
    );
    let mean = mean_of(&out.image_adjustments, |a| a.rolling_shutter);
    for k in 0..3 {
        assert!(
            (mean[k] - start_rates[k]).abs() < 1e-7,
            "mean rate {mean:?}, started at {start_rates:?}"
        );
    }
    // The differences from the mean are the planted ones.
    let centred: Vec<IntrinsicAdjustment> = out
        .image_adjustments
        .iter()
        .map(|a| IntrinsicAdjustment {
            rolling_shutter: std::array::from_fn(|k| a.rolling_shutter[k] - mean[k]),
            ..*a
        })
        .collect();
    let (err, size) = rms_error(&centred, &draws, |a| a.rolling_shutter.to_vec());
    assert!(
        err < 0.3 * size,
        "rates off by {err} (rms) about their mean, planted {size}"
    );
}

#[test]
fn the_derived_widths_recover_planted_spreads() {
    // Forty images each seeing 60 points through its own correction and
    // rates, with half a pixel of noise. Each derived width is its field's
    // planted spread, within what forty images can say about it.
    let truth = spline_pinhole();
    let scanlines = RollingShutterScanlines::Rows;
    let planted = correction_draws(40, 3e-3, 4e-3, 130);
    let (_, tau_e) = rms_error(&planted, &planted, |a| a.radial_bspline.to_vec());
    let (_, tau_w) = rms_error(&planted, &planted, |a| a.rolling_shutter.to_vec());
    let mut s = per_image_scene_on(&truth, Some(scanlines), 40, 400, &planted, 0.5, 101, |_| 60);
    perturb_poses(&mut s);
    let out = run_corrections(
        &mut s,
        &truth,
        &[],
        CORRECTIONS,
        Some(scanlines),
        None,
        &DEFAULT_SCHEDULE,
    );
    let prior = out.image_priors[0].expect("a per-image release");
    assert!(prior.derived);
    let (w_e, w_w) = (prior.widths.radial_bspline, prior.widths.rolling_shutter);
    assert!(
        (0.7 * tau_e..1.4 * tau_e).contains(&w_e),
        "derived radial width {w_e}, planted spread {tau_e}"
    );
    assert!(
        (0.7 * tau_w..1.4 * tau_w).contains(&w_w),
        "derived rate width {w_w}, planted spread {tau_w}"
    );
    assert_eq!(
        prior.spread.radial_bspline, w_e,
        "a width the clamp did not set"
    );
    assert_eq!(
        prior.spread.rolling_shutter, w_w,
        "a width the clamp did not set"
    );
}

#[test]
fn corrections_that_do_not_vary_derive_the_floor() {
    // Every image through the camera itself: the spreads round to nothing,
    // and both widths are the floor.
    let truth = spline_pinhole();
    let scanlines = RollingShutterScanlines::Columns;
    let planted = vec![IntrinsicAdjustment::default(); 30];
    let mut s = per_image_scene_on(&truth, Some(scanlines), 30, 400, &planted, 0.3, 21, |_| 60);
    perturb_poses(&mut s);
    let out = run_corrections(
        &mut s,
        &truth,
        &[],
        CORRECTIONS,
        Some(scanlines),
        None,
        &DEFAULT_SCHEDULE,
    );
    let floor = ImagePriorOptions::default().floor;
    let prior = out.image_priors[0].expect("a per-image release");
    assert_eq!(prior.widths.radial_bspline, floor.radial_bspline);
    assert_eq!(prior.widths.rolling_shutter, floor.rolling_shutter);
    assert!(prior.spread.radial_bspline < floor.radial_bspline);
    assert!(prior.spread.rolling_shutter < floor.rolling_shutter);
}

#[test]
fn fixed_widths_cover_the_corrections() {
    let truth = spline_pinhole();
    let scanlines = RollingShutterScanlines::Rows;
    let planted = correction_draws(12, 3e-3, 4e-3, 150);
    let mut s = per_image_scene_on(&truth, Some(scanlines), 12, 300, &planted, 0.3, 21, |_| 60);
    let fixed = ImagePriorWidths {
        focal: 0.0123,
        principal_point: 0.0456,
        radial_bspline: 0.0078,
        rolling_shutter: 0.0091,
    };
    let options = ImagePriorOptions {
        fixed: Some(fixed),
        ..ImagePriorOptions::default()
    };
    let out = run_corrections(
        &mut s,
        &truth,
        &[],
        CORRECTIONS,
        Some(scanlines),
        Some(&options),
        &DEFAULT_SCHEDULE,
    );
    let prior = out.image_priors[0].unwrap();
    assert!(!prior.derived);
    assert_eq!(prior.widths.radial_bspline, fixed.radial_bspline);
    assert_eq!(prior.widths.rolling_shutter, fixed.rolling_shutter);
    assert!(prior.widths.focal.is_nan() && prior.widths.principal_point.is_nan());
    assert!(prior.spread.radial_bspline.is_nan() && prior.spread.rolling_shutter.is_nan());
}

#[test]
fn the_mean_radial_correction_moves_into_the_spline_without_changing_an_image() {
    // The re-centring after a round: the camera's spline takes the mean of
    // the solved images' corrections, each image keeps its difference, and
    // every image's spline `c + K·e`, the unsolved one included, is the one it
    // had to rounding.
    let cam = spline_pinhole();
    let (c0, _) = spline_of(&cam);
    let mut lenses = vec![Lens::new(
        &cam, None, false, false, false, false, false, false, true, false,
    )];
    let image_camera = vec![0u32; 5];
    let mut adj: Vec<IntrinsicAdjustment> = (0..5)
        .map(|i| IntrinsicAdjustment {
            radial_bspline: [0.003 + 1e-3 * jitter(i, 1), 0.006 + 1e-3 * jitter(i, 2)],
            ..IntrinsicAdjustment::default()
        })
        .collect();
    let before: Vec<Vec<f64>> = adj
        .iter()
        .map(|a| image_spline(&c0, a.radial_bspline))
        .collect();
    // Image 4 was not solved.
    recentre_adjustments(&mut lenses, &image_camera, &mut adj, &[0, 1, 2, 3]);
    assert_ne!(lenses[0].bspline, c0);
    let mean = mean_of(&adj[..4], |a| a.radial_bspline);
    assert!(mean.iter().all(|m| m.abs() < 1e-17), "mean {mean:?}");
    for (i, a) in adj.iter().enumerate() {
        let after = image_spline(&lenses[0].bspline, a.radial_bspline);
        for (x, y) in after.iter().zip(&before[i]) {
            assert!(
                (x - y).abs() < 1e-17,
                "image {i}: {after:?}, before {:?}",
                before[i]
            );
        }
    }
}

#[test]
fn a_radial_correction_that_would_fold_its_spline_is_kept_monotone() {
    // Image 0's observations are made through a radial correction that folds
    // its spline before the domain's end. Under a prior wide enough not to hold
    // it, the solve pulls that image's correction toward it, rejects every step
    // that would break its spline's monotonicity, and ends with every image's
    // spline monotone.
    let truth = spline_pinhole();
    let (c, rho_max) = spline_of(&truth);
    let mut planted = vec![IntrinsicAdjustment::default(); 8];
    planted[0].radial_bspline = [0.0, -0.4];
    assert!(!bspline_is_monotone(
        &image_spline(&c, planted[0].radial_bspline),
        rho_max,
        rho_max
    ));
    let mut s = per_image_scene_on(&truth, None, 8, 300, &planted, 0.0, 21, |_| 80);
    let options = ImagePriorOptions {
        fixed: Some(ImagePriorWidths {
            focal: 1.0,
            principal_point: 1.0,
            radial_bspline: 10.0,
            rolling_shutter: 1.0,
        }),
        ..ImagePriorOptions::default()
    };
    let release = CameraRelease {
        image_radial_bspline: true,
        ..CameraRelease::HELD
    };
    let out = run_corrections(
        &mut s,
        &truth,
        &[],
        release,
        None,
        Some(&options),
        &[BaSchedule {
            trim_px: f64::INFINITY,
            loss_scale: 10.0,
        }],
    );
    let (c_out, _) = spline_of(&out.cameras[0]);
    for (i, a) in out.image_adjustments.iter().enumerate() {
        assert!(
            bspline_is_monotone(&image_spline(&c_out, a.radial_bspline), rho_max, rho_max),
            "image {i}'s spline folds: {:?}",
            a.radial_bspline
        );
    }
    let e0 =
        out.image_adjustments[0].radial_bspline[1] - out.image_adjustments[1].radial_bspline[1];
    assert!(
        e0 < -0.05,
        "image 0's correction {e0} did not move toward the planted one"
    );
}

#[test]
fn the_rolling_shutter_contraction_grows_with_the_rates() {
    // Zero without rates, about `|ω|·f/m` for small ones, and linear in them.
    let cam = spline_pinhole();
    let at = |rates: [f64; 3], scanlines| {
        cam.with_adjustment(
            &IntrinsicAdjustment {
                rolling_shutter: rates,
                ..IntrinsicAdjustment::default()
            },
            Some(scanlines),
        )
        .rolling_shutter_contraction()
    };
    for scanlines in BOTH_SCANLINES {
        assert_eq!(cam.rolling_shutter_contraction(), 0.0);
        let small = at([1e-3, 0.0, 0.0], scanlines);
        let f_over_m = cam.focal_lengths().0 / 480.0;
        assert!(
            (0.5 * f_over_m * 1e-3..4.0 * f_over_m * 1e-3).contains(&small),
            "{scanlines:?}: {small}"
        );
        let double = at([2e-3, 0.0, 0.0], scanlines);
        assert!(
            (double / small - 2.0).abs() < 1e-6,
            "{scanlines:?}: {double} {small}"
        );
        assert!(at([0.8, 0.0, 0.0], scanlines) > ROLLING_SHUTTER_CONTRACTION_LIMIT);
    }
}

#[test]
fn rates_far_from_the_identity_are_refused() {
    // Image 0's observations are made through rates of 0.8 per short side, at
    // which the rolling-shutter step at the corners is farther from the
    // identity than the guard allows. Under a prior wide enough not to hold
    // them, the solve pulls that image's rates toward them and stops at the
    // guard.
    let truth = spline_pinhole();
    let scanlines = RollingShutterScanlines::Rows;
    let mut planted = vec![IntrinsicAdjustment::default(); 8];
    planted[0].rolling_shutter = [0.8, 0.0, 0.0];
    let mut s = per_image_scene_on(&truth, Some(scanlines), 8, 300, &planted, 0.0, 21, |_| 80);
    let options = ImagePriorOptions {
        fixed: Some(ImagePriorWidths {
            focal: 1.0,
            principal_point: 1.0,
            radial_bspline: 1.0,
            rolling_shutter: 10.0,
        }),
        ..ImagePriorOptions::default()
    };
    let release = CameraRelease {
        image_rolling_shutter: true,
        ..CameraRelease::HELD
    };
    let out = run_corrections(
        &mut s,
        &truth,
        &[],
        release,
        Some(scanlines),
        Some(&options),
        &[BaSchedule {
            trim_px: f64::INFINITY,
            loss_scale: 10.0,
        }],
    );
    for (i, a) in out.image_adjustments.iter().enumerate() {
        let c = out.cameras[0]
            .with_adjustment(a, Some(scanlines))
            .rolling_shutter_contraction();
        assert!(
            c <= ROLLING_SHUTTER_CONTRACTION_LIMIT,
            "image {i}'s step is {c} from the identity: {:?}",
            a.rolling_shutter
        );
    }
    let w0 =
        out.image_adjustments[0].rolling_shutter[0] - out.image_adjustments[1].rolling_shutter[0];
    assert!(
        w0 > 0.1,
        "image 0's rate {w0} did not move toward the planted one"
    );
}

#[test]
fn correction_flags_with_nothing_released_per_image_change_nothing() {
    // The correction flags on, with the camera's release holding its images'
    // corrections, against the same solve with the flags off: bit for bit.
    let truth = spline_pinhole();
    let planted = principal_point_draws(8, 0.004);
    let scene = per_image_scene(&truth, 8, 200, &planted, 0.3, 21, |_| 50);
    let mut a = scene.clone();
    let mut b = scene.clone();
    let out_a = run_corrections(
        &mut a,
        &truth,
        &[],
        IMAGE_PP,
        Some(RollingShutterScanlines::Rows),
        None,
        &DEFAULT_SCHEDULE,
    );
    let out_b = run_image(&mut b, &truth, &[], IMAGE_PP, None, &DEFAULT_SCHEDULE);
    assert_bitwise_equal(&a, &out_a, &b, &out_b);
    assert_eq!(out_a.cameras, out_b.cameras);
    assert_eq!(out_a.image_adjustments, out_b.image_adjustments);
    assert_eq!(
        format!("{:?}", out_a.image_priors),
        format!("{:?}", out_b.image_priors)
    );
}

#[test]
fn corrections_on_another_model_are_held() {
    // A `SIMPLE_PINHOLE` cannot take either correction, so releasing them
    // there is the solve that holds them, bit for bit.
    let truth = wide_pinhole(326.0, 236.0);
    let scene = per_image_scene(
        &truth,
        8,
        200,
        &vec![IntrinsicAdjustment::default(); 8],
        0.3,
        21,
        |_| 50,
    );
    let mut a = scene.clone();
    let mut b = scene.clone();
    let rows = Some(RollingShutterScanlines::Rows);
    let out_a = run_corrections(
        &mut a,
        &truth,
        &[],
        CORRECTIONS,
        rows,
        None,
        &DEFAULT_SCHEDULE,
    );
    let out_b = run_corrections(
        &mut b,
        &truth,
        &[],
        CameraRelease::HELD,
        rows,
        None,
        &DEFAULT_SCHEDULE,
    );
    assert_bitwise_equal(&a, &out_a, &b, &out_b);
    assert_eq!(out_a.image_priors, vec![None]);
}

#[test]
#[should_panic(expected = "rolling_shutter_scanlines")]
fn releasing_rates_without_a_scanlines_setting_panics() {
    let truth = spline_pinhole();
    let mut s = per_image_scene(
        &truth,
        6,
        200,
        &vec![IntrinsicAdjustment::default(); 6],
        0.3,
        21,
        |_| 50,
    );
    run_corrections(
        &mut s,
        &truth,
        &[],
        CORRECTIONS,
        None,
        None,
        &DEFAULT_SCHEDULE,
    );
}
