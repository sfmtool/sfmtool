// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use super::*;

/// The first `kerry_park` rig sensor's lens, to the digits the switch draft
/// quotes: `OPENCV_FISHEYE`, 480 × 480, principal point held at the centre.
fn kerry_cam0() -> CameraIntrinsics {
    CameraIntrinsics {
        model: CameraModel::OpenCVFisheye {
            focal_length_x: 129.718,
            focal_length_y: 129.430,
            principal_point_x: 240.0,
            principal_point_y: 240.0,
            radial_distortion_k1: 0.02865,
            radial_distortion_k2: -0.00228,
            radial_distortion_k3: 0.00902,
            radial_distortion_k4: -0.00355,
        },
        width: 480,
        height: 480,
    }
}

fn equidistant(f: f64) -> CameraIntrinsics {
    CameraIntrinsics {
        model: CameraModel::EquidistantFisheye {
            focal_length: f,
            principal_point_x: 240.0,
            principal_point_y: 236.0,
        },
        width: 480,
        height: 480,
    }
}

fn sfmtool_fisheye(f: f64, theta_max: f64, bspline: Vec<f64>) -> CameraIntrinsics {
    CameraIntrinsics {
        model: CameraModel::SfmtoolFisheye {
            focal_length: f,
            principal_point_x: 240.0,
            principal_point_y: 240.0,
            bspline_theta_max: theta_max,
            bspline,
        },
        width: 480,
        height: 480,
    }
}

fn spline_of(camera: &CameraIntrinsics) -> (f64, f64, Vec<f64>) {
    match &camera.model {
        CameraModel::SfmtoolFisheye {
            focal_length,
            bspline_theta_max,
            bspline,
            ..
        } => (*focal_length, *bspline_theta_max, bspline.clone()),
        other => panic!("expected SFMTOOL_FISHEYE, got {}", other.model_name()),
    }
}

#[test]
fn equidistant_to_sfmtool_fisheye_is_exact_with_zero_coefficients() {
    let source = equidistant(130.0);
    let target = RefitTarget::from_name("SFMTOOL_FISHEYE", Some(8)).unwrap();
    let refit = refit_camera(&source, &target, &RefitOptions::default()).unwrap();

    let (f, _, coeffs) = spline_of(&refit.camera);
    assert!((f - 130.0).abs() < 1e-9, "focal {f}");
    assert_eq!(coeffs.len(), 8);
    assert!(coeffs.iter().all(|c| c.abs() < 1e-12), "{coeffs:?}");
    assert!(refit.max_px < 1e-8, "max {}", refit.max_px);
    assert_eq!(refit.camera.principal_point(), (240.0, 236.0));
    assert_eq!(refit.theta_fit_source, ThetaFitSource::ImageCorner);
    assert!(refit.dropped.is_empty());
}

#[test]
fn equidistant_target_is_the_spline_with_no_coefficients() {
    let source = sfmtool_fisheye(125.0, 2.5, vec![0.0; 6]);
    let refit = refit_camera(
        &source,
        &RefitTarget::EquidistantFisheye,
        &RefitOptions::default(),
    )
    .unwrap();
    assert_eq!(refit.camera.model_name(), "EQUIDISTANT_FISHEYE");
    assert!((refit.camera.focal_lengths().0 - 125.0).abs() < 1e-9);
    assert_eq!(refit.spline_domain_deg, None);
}

#[test]
fn simple_radial_to_radial_reproduces_the_copy() {
    let source = CameraIntrinsics {
        model: CameraModel::SimpleRadial {
            focal_length: 400.0,
            principal_point_x: 135.0,
            principal_point_y: 240.0,
            radial_distortion_k1: -0.05,
        },
        width: 270,
        height: 480,
    };
    let target = RefitTarget::from_name("radial", None).unwrap();
    let refit = refit_camera(&source, &target, &RefitOptions::default()).unwrap();
    match refit.camera.model {
        CameraModel::Radial {
            focal_length,
            principal_point_x,
            principal_point_y,
            radial_distortion_k1,
            radial_distortion_k2,
        } => {
            assert!((focal_length - 400.0).abs() < 1e-9);
            assert_eq!((principal_point_x, principal_point_y), (135.0, 240.0));
            assert!((radial_distortion_k1 + 0.05).abs() < 1e-12);
            assert!(radial_distortion_k2.abs() < 1e-12);
        }
        other => panic!("expected RADIAL, got {}", other.model_name()),
    }
    assert!(refit.max_px < 1e-9);
}

#[test]
fn a_synthetic_spline_is_recovered() {
    let theta_max = 2.4;
    let truth = vec![0.001, -0.004, -0.012, -0.025, -0.04, -0.058];
    let source = sfmtool_fisheye(128.0, theta_max, truth.clone());
    let target = RefitTarget::from_name("SFMTOOL_FISHEYE", Some(truth.len())).unwrap();
    let options = RefitOptions {
        theta_fit_deg: Some(theta_max.to_degrees()),
        spline_domain_deg: Some(theta_max.to_degrees()),
    };
    let refit = refit_camera(&source, &target, &options).unwrap();
    let (f, domain, coeffs) = spline_of(&refit.camera);
    assert!((domain - theta_max).abs() < 1e-12);
    assert!((f - 128.0).abs() < 1e-3, "focal {f}");
    for (fitted, want) in coeffs.iter().zip(&truth) {
        assert!((fitted - want).abs() < 1e-4, "{coeffs:?} vs {truth:?}");
    }
    assert!(refit.max_px < 0.01, "max {}", refit.max_px);
}

#[test]
fn kerry_default_fit_stops_at_the_trusted_bound_short_of_the_fold() {
    let source = kerry_cam0();
    let target = RefitTarget::from_name("SFMTOOL_FISHEYE", Some(8)).unwrap();
    let refit = refit_camera(&source, &target, &RefitOptions::default()).unwrap();

    assert_eq!(refit.theta_fit_source, ThetaFitSource::TrustedBound);
    let fold = refit.extent.source_fold_deg.expect("cam0 folds");
    assert!((fold - 101.6).abs() < 0.5, "fold {fold}");
    assert!(refit.theta_fit_deg < 90.0 && refit.theta_fit_deg < fold);

    let (f, domain, coeffs) = spline_of(&refit.camera);
    assert!((f - 129.56).abs() < 0.1, "focal {f}");
    // The far corner at 339 px over a 129.6 focal: about 150°.
    assert!(
        (domain.to_degrees() - 150.0).abs() < 1.0,
        "{}",
        domain.to_degrees()
    );
    assert!(bspline_is_monotone(&coeffs, domain, domain));
    assert!(refit.radial_rms_px < 0.05, "radial {}", refit.radial_rms_px);
    assert!(refit.rms_px < 0.2, "rms {}", refit.rms_px);
    assert!(refit.max_px < 0.5, "max {}", refit.max_px);
    assert_eq!(refit.dropped.len(), 1);
    match refit.dropped[0] {
        DroppedTerm::FocalAspect { fy_over_fx } => {
            assert!((fy_over_fx - 129.430 / 129.718).abs() < 1e-12)
        }
        ref other => panic!("unexpected {other:?}"),
    }
    assert!(refit.dropped[0].to_string().contains("0.9978"));
    // The new model reaches the corners, where the old one had folded.
    assert!(
        refit.extent.corner_deg > 140.0,
        "{}",
        refit.extent.corner_deg
    );
}

#[test]
fn a_colmap_target_equal_to_its_source_is_unchanged() {
    let source = kerry_cam0();
    let target = RefitTarget::from_name("OPENCV_FISHEYE", None).unwrap();
    let refit = refit_camera(&source, &target, &RefitOptions::default()).unwrap();
    assert_eq!(refit.camera, source);
    assert!(refit.dropped.is_empty());
}

#[test]
fn a_polynomial_is_fitted_to_a_spline() {
    let source = sfmtool_fisheye(130.0, 2.6, vec![0.0, -0.003, -0.01, -0.02, -0.035]);
    let target = RefitTarget::from_name("OPENCV_FISHEYE", None).unwrap();
    let options = RefitOptions {
        theta_fit_deg: Some(80.0),
        spline_domain_deg: None,
    };
    let refit = refit_camera(&source, &target, &options).unwrap();
    assert_eq!(refit.camera.model_name(), "OPENCV_FISHEYE");
    assert!(refit.max_px < 0.05, "max {}", refit.max_px);
}

#[test]
fn a_perspective_target_past_90_degrees_is_refused() {
    let source = equidistant(130.0);
    let target = RefitTarget::from_name("SFMTOOL_PINHOLE", Some(4)).unwrap();
    let err = refit_camera(&source, &target, &RefitOptions::default()).unwrap_err();
    assert!(matches!(err, RefitError::PerspectivePast90 { .. }), "{err}");

    // A perspective source fits it.
    let source = CameraIntrinsics {
        model: CameraModel::SimpleRadial {
            focal_length: 400.0,
            principal_point_x: 135.0,
            principal_point_y: 240.0,
            radial_distortion_k1: -0.05,
        },
        width: 270,
        height: 480,
    };
    let refit = refit_camera(&source, &target, &RefitOptions::default()).unwrap();
    assert_eq!(refit.camera.model_name(), "SFMTOOL_PINHOLE");
    assert_eq!(refit.theta_fit_source, ThetaFitSource::ImageCorner);
    assert!(refit.max_px < 0.05, "max {}", refit.max_px);
}

#[test]
fn a_fit_past_the_trusted_bound_is_refused() {
    let options = RefitOptions {
        theta_fit_deg: Some(100.0),
        spline_domain_deg: None,
    };
    let target = RefitTarget::from_name("SFMTOOL_FISHEYE", None).unwrap();
    let err = refit_camera(&kerry_cam0(), &target, &options).unwrap_err();
    match err {
        RefitError::BeyondTrustedBound {
            theta_fit_deg,
            trusted_deg,
        } => {
            assert_eq!(theta_fit_deg, 100.0);
            assert!(trusted_deg < 90.0);
        }
        other => panic!("unexpected {other}"),
    }
}

#[test]
fn a_non_monotone_spline_is_refused() {
    // A spline that turns over and falls back while its radius stays positive:
    // the source projects, and a fit that follows it has no inverse.
    let folded = vec![0.0, -0.2, -0.6, -1.0];
    let source = sfmtool_fisheye(130.0, 2.4, folded.clone());
    assert!(!bspline_is_monotone(&folded, 2.4, 2.4));
    let target = RefitTarget::from_name("SFMTOOL_FISHEYE", Some(4)).unwrap();
    let options = RefitOptions {
        theta_fit_deg: Some(120.0),
        spline_domain_deg: Some(2.4f64.to_degrees()),
    };
    let err = refit_camera(&source, &target, &options).unwrap_err();
    assert_eq!(err, RefitError::NotMonotone);
}

#[test]
fn a_polynomial_trusted_short_of_the_fit_is_refused() {
    // Past 90° of distorted angle the fisheye polynomial's inverse blends, so a
    // polynomial fitted to 120° of a near-equidistant lens is trusted only to
    // about 90°.
    let source = sfmtool_fisheye(130.0, 2.6, vec![0.0, -0.001, -0.002, -0.003]);
    let target = RefitTarget::from_name("OPENCV_FISHEYE", None).unwrap();
    let options = RefitOptions {
        theta_fit_deg: Some(120.0),
        spline_domain_deg: None,
    };
    let err = refit_camera(&source, &target, &options).unwrap_err();
    assert!(matches!(err, RefitError::TrustedBoundShort { .. }), "{err}");
}

#[test]
fn targets_are_checked_by_name() {
    assert!(matches!(
        RefitTarget::from_name("EQUIRECTANGULAR", None),
        Err(RefitError::UnknownTarget { .. })
    ));
    assert!(matches!(
        RefitTarget::from_name("NOT_A_MODEL", None),
        Err(RefitError::UnknownTarget { .. })
    ));
    assert!(matches!(
        RefitTarget::from_name("SFMTOOL_FISHEYE", Some(1)),
        Err(RefitError::CoeffCount { count: 1, .. })
    ));
    assert!(matches!(
        RefitTarget::from_name("RADIAL", Some(4)),
        Err(RefitError::CoeffCountNotApplicable { .. })
    ));
    assert_eq!(
        RefitTarget::from_name("sfmtool_fisheye", None).unwrap(),
        RefitTarget::SfmtoolFisheye {
            coeff_count: DEFAULT_COEFF_COUNT
        }
    );
    assert_eq!(
        RefitTarget::from_name("opencv_fisheye", None).unwrap(),
        RefitTarget::Colmap("OPENCV_FISHEYE")
    );
}
