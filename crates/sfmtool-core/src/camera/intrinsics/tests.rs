use super::*;
use approx::assert_relative_eq;

// -----------------------------------------------------------------------
// Helper: build test instances for each model
// -----------------------------------------------------------------------

fn pinhole() -> CameraIntrinsics {
    CameraIntrinsics {
        model: CameraModel::Pinhole {
            focal_length_x: 500.0,
            focal_length_y: 502.0,
            principal_point_x: 320.0,
            principal_point_y: 240.0,
        },
        width: 640,
        height: 480,
    }
}

fn simple_pinhole() -> CameraIntrinsics {
    CameraIntrinsics {
        model: CameraModel::SimplePinhole {
            focal_length: 500.0,
            principal_point_x: 320.0,
            principal_point_y: 240.0,
        },
        width: 640,
        height: 480,
    }
}

fn simple_radial() -> CameraIntrinsics {
    CameraIntrinsics {
        model: CameraModel::SimpleRadial {
            focal_length: 500.0,
            principal_point_x: 320.0,
            principal_point_y: 240.0,
            radial_distortion_k1: 0.1,
        },
        width: 640,
        height: 480,
    }
}

fn radial() -> CameraIntrinsics {
    CameraIntrinsics {
        model: CameraModel::Radial {
            focal_length: 500.0,
            principal_point_x: 320.0,
            principal_point_y: 240.0,
            radial_distortion_k1: 0.1,
            radial_distortion_k2: -0.05,
        },
        width: 640,
        height: 480,
    }
}

fn opencv() -> CameraIntrinsics {
    CameraIntrinsics {
        model: CameraModel::OpenCV {
            focal_length_x: 500.0,
            focal_length_y: 502.0,
            principal_point_x: 320.0,
            principal_point_y: 240.0,
            radial_distortion_k1: 0.1,
            radial_distortion_k2: -0.05,
            tangential_distortion_p1: 0.001,
            tangential_distortion_p2: -0.002,
        },
        width: 640,
        height: 480,
    }
}

fn opencv_fisheye() -> CameraIntrinsics {
    CameraIntrinsics {
        model: CameraModel::OpenCVFisheye {
            focal_length_x: 500.0,
            focal_length_y: 502.0,
            principal_point_x: 320.0,
            principal_point_y: 240.0,
            radial_distortion_k1: 0.1,
            radial_distortion_k2: -0.05,
            radial_distortion_k3: 0.01,
            radial_distortion_k4: -0.005,
        },
        width: 640,
        height: 480,
    }
}

fn full_opencv() -> CameraIntrinsics {
    CameraIntrinsics {
        model: CameraModel::FullOpenCV {
            focal_length_x: 500.0,
            focal_length_y: 502.0,
            principal_point_x: 320.0,
            principal_point_y: 240.0,
            radial_distortion_k1: 0.1,
            radial_distortion_k2: -0.05,
            tangential_distortion_p1: 0.001,
            tangential_distortion_p2: -0.002,
            radial_distortion_k3: 0.01,
            radial_distortion_k4: -0.005,
            radial_distortion_k5: 0.002,
            radial_distortion_k6: -0.001,
        },
        width: 640,
        height: 480,
    }
}

fn simple_radial_fisheye() -> CameraIntrinsics {
    CameraIntrinsics {
        model: CameraModel::SimpleRadialFisheye {
            focal_length: 500.0,
            principal_point_x: 320.0,
            principal_point_y: 240.0,
            radial_distortion_k1: 0.05,
        },
        width: 640,
        height: 480,
    }
}

fn radial_fisheye() -> CameraIntrinsics {
    CameraIntrinsics {
        model: CameraModel::RadialFisheye {
            focal_length: 500.0,
            principal_point_x: 320.0,
            principal_point_y: 240.0,
            radial_distortion_k1: 0.05,
            radial_distortion_k2: -0.02,
        },
        width: 640,
        height: 480,
    }
}

fn thin_prism_fisheye() -> CameraIntrinsics {
    CameraIntrinsics {
        model: CameraModel::ThinPrismFisheye {
            focal_length_x: 500.0,
            focal_length_y: 502.0,
            principal_point_x: 320.0,
            principal_point_y: 240.0,
            radial_distortion_k1: 0.05,
            radial_distortion_k2: -0.02,
            tangential_distortion_p1: 0.001,
            tangential_distortion_p2: -0.001,
            radial_distortion_k3: 0.005,
            radial_distortion_k4: -0.003,
            thin_prism_sx1: 0.001,
            thin_prism_sy1: -0.001,
        },
        width: 640,
        height: 480,
    }
}

fn rad_tan_thin_prism_fisheye() -> CameraIntrinsics {
    CameraIntrinsics {
        model: CameraModel::RadTanThinPrismFisheye {
            focal_length_x: 500.0,
            focal_length_y: 502.0,
            principal_point_x: 320.0,
            principal_point_y: 240.0,
            radial_distortion_k0: 0.01,
            radial_distortion_k1: 0.02,
            radial_distortion_k2: -0.01,
            radial_distortion_k3: 0.005,
            radial_distortion_k4: -0.003,
            radial_distortion_k5: 0.001,
            tangential_distortion_p0: 0.001,
            tangential_distortion_p1: -0.001,
            thin_prism_s0: 0.001,
            thin_prism_s1: -0.001,
            thin_prism_s2: 0.0005,
            thin_prism_s3: -0.0005,
        },
        width: 640,
        height: 480,
    }
}

fn equirectangular() -> CameraIntrinsics {
    // Equirectangular with same principal point (320, 240) as other test cameras
    CameraIntrinsics {
        model: CameraModel::Equirectangular {
            focal_length_x: 640.0 / (2.0 * std::f64::consts::PI),
            focal_length_y: 480.0 / std::f64::consts::PI,
            principal_point_x: 320.0,
            principal_point_y: 240.0,
        },
        width: 640,
        height: 480,
    }
}

fn equidistant_fisheye() -> CameraIntrinsics {
    CameraIntrinsics {
        model: CameraModel::EquidistantFisheye {
            focal_length: 500.0,
            principal_point_x: 320.0,
            principal_point_y: 240.0,
        },
        width: 640,
        height: 480,
    }
}

fn sfmtool_fisheye() -> CameraIntrinsics {
    CameraIntrinsics {
        model: CameraModel::SfmtoolFisheye {
            focal_length: 500.0,
            principal_point_x: 320.0,
            principal_point_y: 240.0,
            bspline_theta_max: 2.0,
            bspline: vec![-0.001, -0.004, -0.01, -0.02, -0.03, -0.05, -0.07, -0.09],
        },
        width: 640,
        height: 480,
    }
}

fn sfmtool_pinhole() -> CameraIntrinsics {
    CameraIntrinsics {
        model: CameraModel::SfmtoolPinhole {
            focal_length: 500.0,
            principal_point_x: 320.0,
            principal_point_y: 240.0,
            bspline_rho_max: 0.9,
            bspline: vec![
                0.0008, 0.0031, 0.0075, 0.0142, 0.0236, 0.0361, 0.052, 0.0718,
            ],
        },
        width: 640,
        height: 480,
    }
}

fn all_cameras() -> Vec<CameraIntrinsics> {
    vec![
        pinhole(),
        simple_pinhole(),
        simple_radial(),
        radial(),
        opencv(),
        opencv_fisheye(),
        simple_radial_fisheye(),
        radial_fisheye(),
        thin_prism_fisheye(),
        rad_tan_thin_prism_fisheye(),
        full_opencv(),
        equirectangular(),
        equidistant_fisheye(),
        sfmtool_fisheye(),
        sfmtool_pinhole(),
    ]
}

// -----------------------------------------------------------------------
// Intrinsic matrix: K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
// -----------------------------------------------------------------------

#[test]
fn intrinsic_matrix_pinhole() {
    let cam = pinhole();
    let k = cam.intrinsic_matrix();
    assert_relative_eq!(k[(0, 0)], 500.0);
    assert_relative_eq!(k[(1, 1)], 502.0);
    assert_relative_eq!(k[(0, 2)], 320.0);
    assert_relative_eq!(k[(1, 2)], 240.0);
    assert_relative_eq!(k[(0, 1)], 0.0);
    assert_relative_eq!(k[(1, 0)], 0.0);
    assert_relative_eq!(k[(2, 0)], 0.0);
    assert_relative_eq!(k[(2, 1)], 0.0);
    assert_relative_eq!(k[(2, 2)], 1.0);
}

// Single-focal models use fx = fy = f

#[test]
fn intrinsic_matrix_simple_pinhole() {
    let cam = simple_pinhole();
    let k = cam.intrinsic_matrix();
    assert_relative_eq!(k[(0, 0)], 500.0);
    assert_relative_eq!(k[(1, 1)], 500.0);
    assert_relative_eq!(k[(0, 2)], 320.0);
    assert_relative_eq!(k[(1, 2)], 240.0);
    assert_relative_eq!(k[(2, 2)], 1.0);
}

// Distortion parameters do not affect K

#[test]
fn intrinsic_matrix_ignores_distortion() {
    // All distortion models should produce the same K when they share fx, fy, cx, cy.
    let cam_opencv = opencv();
    let cam_fisheye = opencv_fisheye();
    let cam_full = full_opencv();

    for cam in [&cam_opencv, &cam_fisheye, &cam_full] {
        let k = cam.intrinsic_matrix();
        assert_relative_eq!(k[(0, 0)], 500.0);
        assert_relative_eq!(k[(1, 1)], 502.0);
        assert_relative_eq!(k[(0, 2)], 320.0);
        assert_relative_eq!(k[(1, 2)], 240.0);
        assert_relative_eq!(k[(2, 2)], 1.0);
        // Off-diagonal zeros
        assert_relative_eq!(k[(0, 1)], 0.0);
        assert_relative_eq!(k[(1, 0)], 0.0);
        assert_relative_eq!(k[(2, 0)], 0.0);
        assert_relative_eq!(k[(2, 1)], 0.0);
    }

    // Single-focal models with distortion
    let cam_sr = simple_radial();
    let cam_r = radial();
    for cam in [&cam_sr, &cam_r] {
        let k = cam.intrinsic_matrix();
        assert_relative_eq!(k[(0, 0)], 500.0);
        assert_relative_eq!(k[(1, 1)], 500.0);
        assert_relative_eq!(k[(0, 2)], 320.0);
        assert_relative_eq!(k[(1, 2)], 240.0);
    }
}

// -----------------------------------------------------------------------
// model_name() returns COLMAP-compatible string for each variant
// -----------------------------------------------------------------------

#[test]
fn model_name_all_variants() {
    let expected = [
        "PINHOLE",
        "SIMPLE_PINHOLE",
        "SIMPLE_RADIAL",
        "RADIAL",
        "OPENCV",
        "OPENCV_FISHEYE",
        "SIMPLE_RADIAL_FISHEYE",
        "RADIAL_FISHEYE",
        "THIN_PRISM_FISHEYE",
        "RAD_TAN_THIN_PRISM_FISHEYE",
        "FULL_OPENCV",
        "EQUIRECTANGULAR",
        "EQUIDISTANT_FISHEYE",
        "SFMTOOL_FISHEYE",
        "SFMTOOL_PINHOLE",
    ];
    assert_eq!(all_cameras().len(), expected.len());
    for (cam, name) in all_cameras().iter().zip(expected.iter()) {
        assert_eq!(cam.model_name(), *name);
    }
}

// -----------------------------------------------------------------------
// focal_lengths(): dual-focal models return (fx, fy), single-focal return (f, f)
// -----------------------------------------------------------------------

#[test]
fn focal_lengths_dual_focal() {
    let cam = pinhole();
    assert_eq!(cam.focal_lengths(), (500.0, 502.0));
}

#[test]
fn focal_lengths_single_focal() {
    for cam in [simple_pinhole(), simple_radial(), radial()] {
        let (fx, fy) = cam.focal_lengths();
        assert_relative_eq!(fx, 500.0);
        assert_relative_eq!(fy, 500.0);
    }
}

// -----------------------------------------------------------------------
// principal_point() extracts (cx, cy) from all model variants
// -----------------------------------------------------------------------

#[test]
fn principal_point_all_models() {
    for cam in all_cameras() {
        assert_eq!(cam.principal_point(), (320.0, 240.0));
    }
}

// -----------------------------------------------------------------------
// has_distortion(): false for pure pinhole, true for models with k/p params
// -----------------------------------------------------------------------

#[test]
fn has_distortion_false_for_pinhole_models() {
    assert!(!pinhole().has_distortion());
    assert!(!simple_pinhole().has_distortion());
}

#[test]
fn has_distortion_true_for_distortion_models() {
    assert!(simple_radial().has_distortion());
    assert!(radial().has_distortion());
    assert!(opencv().has_distortion());
    assert!(opencv_fisheye().has_distortion());
    assert!(full_opencv().has_distortion());
}

#[test]
fn has_distortion_false_for_zero_coefficient_models() {
    // Distortion-capable models with all-zero coefficients are effectively pinhole
    let sr = CameraIntrinsics {
        model: CameraModel::SimpleRadial {
            focal_length: 500.0,
            principal_point_x: 320.0,
            principal_point_y: 240.0,
            radial_distortion_k1: 0.0,
        },
        width: 640,
        height: 480,
    };
    assert!(!sr.has_distortion());

    let r = CameraIntrinsics {
        model: CameraModel::Radial {
            focal_length: 500.0,
            principal_point_x: 320.0,
            principal_point_y: 240.0,
            radial_distortion_k1: 0.0,
            radial_distortion_k2: 0.0,
        },
        width: 640,
        height: 480,
    };
    assert!(!r.has_distortion());

    let cv = CameraIntrinsics {
        model: CameraModel::OpenCV {
            focal_length_x: 500.0,
            focal_length_y: 502.0,
            principal_point_x: 320.0,
            principal_point_y: 240.0,
            radial_distortion_k1: 0.0,
            radial_distortion_k2: 0.0,
            tangential_distortion_p1: 0.0,
            tangential_distortion_p2: 0.0,
        },
        width: 640,
        height: 480,
    };
    assert!(!cv.has_distortion());

    let fe = CameraIntrinsics {
        model: CameraModel::OpenCVFisheye {
            focal_length_x: 500.0,
            focal_length_y: 502.0,
            principal_point_x: 320.0,
            principal_point_y: 240.0,
            radial_distortion_k1: 0.0,
            radial_distortion_k2: 0.0,
            radial_distortion_k3: 0.0,
            radial_distortion_k4: 0.0,
        },
        width: 640,
        height: 480,
    };
    assert!(!fe.has_distortion());

    let full = CameraIntrinsics {
        model: CameraModel::FullOpenCV {
            focal_length_x: 500.0,
            focal_length_y: 502.0,
            principal_point_x: 320.0,
            principal_point_y: 240.0,
            radial_distortion_k1: 0.0,
            radial_distortion_k2: 0.0,
            tangential_distortion_p1: 0.0,
            tangential_distortion_p2: 0.0,
            radial_distortion_k3: 0.0,
            radial_distortion_k4: 0.0,
            radial_distortion_k5: 0.0,
            radial_distortion_k6: 0.0,
        },
        width: 640,
        height: 480,
    };
    assert!(!full.has_distortion());
}

#[test]
fn has_distortion_on_camera_model_directly() {
    // Test CameraModel::has_distortion directly
    assert!(!CameraModel::Pinhole {
        focal_length_x: 500.0,
        focal_length_y: 500.0,
        principal_point_x: 320.0,
        principal_point_y: 240.0,
    }
    .has_distortion());

    assert!(CameraModel::SimpleRadial {
        focal_length: 500.0,
        principal_point_x: 320.0,
        principal_point_y: 240.0,
        radial_distortion_k1: 0.01,
    }
    .has_distortion());

    // Zero k1 → no effective distortion
    assert!(!CameraModel::SimpleRadial {
        focal_length: 500.0,
        principal_point_x: 320.0,
        principal_point_y: 240.0,
        radial_distortion_k1: 0.0,
    }
    .has_distortion());
}

// -----------------------------------------------------------------------
// SfmrCamera serialization round-trip preserves all parameters
// -----------------------------------------------------------------------

#[test]
fn sfmr_camera_round_trip_all_models() {
    for cam in all_cameras() {
        let sfmr: SfmrCamera = SfmrCamera::from(&cam);
        let restored = CameraIntrinsics::try_from(&sfmr)
            .unwrap_or_else(|e| panic!("round-trip failed for {}: {e}", cam.model_name()));
        assert_eq!(
            cam,
            restored,
            "round-trip mismatch for {}",
            cam.model_name()
        );
    }
}

// -----------------------------------------------------------------------
// TryFrom rejects unknown models and missing parameters
// -----------------------------------------------------------------------

#[test]
fn try_from_unknown_model() {
    let sfmr = SfmrCamera {
        model: "UNKNOWN_MODEL".to_string(),
        width: 640,
        height: 480,
        parameters: BTreeMap::new(),
    };
    let err = CameraIntrinsics::try_from(&sfmr).unwrap_err();
    assert!(
        matches!(err, CameraIntrinsicsError::UnknownModel(ref name) if name == "UNKNOWN_MODEL")
    );
}

#[test]
fn try_from_missing_parameter() {
    let mut params = BTreeMap::new();
    params.insert("focal_length_x".to_string(), 500.0);
    // Missing focal_length_y, principal_point_x, principal_point_y
    let sfmr = SfmrCamera {
        model: "PINHOLE".to_string(),
        width: 640,
        height: 480,
        parameters: params,
    };
    let err = CameraIntrinsics::try_from(&sfmr).unwrap_err();
    assert!(matches!(
        err,
        CameraIntrinsicsError::MissingParameter {
            ref model,
            ref parameter,
        } if model == "PINHOLE" && parameter == "focal_length_y"
    ));
}

// -----------------------------------------------------------------------
// 10. RADIAL uses single focal length (matching COLMAP definition)
// -----------------------------------------------------------------------
//
// COLMAP's RADIAL model has a single focal length parameter, not two.
// Verify that intrinsic_matrix() correctly uses (f, f) for both fx and fy.

#[test]
fn radial_uses_single_focal_length() {
    let cam = radial();
    let k = cam.intrinsic_matrix();
    // fx and fy should both equal the single focal_length parameter
    assert_relative_eq!(k[(0, 0)], k[(1, 1)], epsilon = 1e-12);
    assert_relative_eq!(k[(0, 0)], 500.0);
}

// -----------------------------------------------------------------------
// CameraModel::model_name() delegates correctly
// -----------------------------------------------------------------------

#[test]
fn camera_model_model_name() {
    let cam = pinhole();
    assert_eq!(cam.model.model_name(), "PINHOLE");

    let cam = simple_pinhole();
    assert_eq!(cam.model.model_name(), "SIMPLE_PINHOLE");

    let cam = simple_radial();
    assert_eq!(cam.model.model_name(), "SIMPLE_RADIAL");

    let cam = radial();
    assert_eq!(cam.model.model_name(), "RADIAL");

    let cam = opencv();
    assert_eq!(cam.model.model_name(), "OPENCV");

    let cam = opencv_fisheye();
    assert_eq!(cam.model.model_name(), "OPENCV_FISHEYE");

    let cam = full_opencv();
    assert_eq!(cam.model.model_name(), "FULL_OPENCV");
}

// -----------------------------------------------------------------------
// Debug formatting includes type name, variant, and values
// -----------------------------------------------------------------------

#[test]
fn debug_formatting() {
    let cam = simple_pinhole();
    let debug_str = format!("{cam:?}");
    assert!(debug_str.contains("CameraIntrinsics"));
    assert!(debug_str.contains("SimplePinhole"));
    assert!(debug_str.contains("500"));
    assert!(debug_str.contains("640"));
    assert!(debug_str.contains("480"));
}

// -----------------------------------------------------------------------
// Error messages are human-readable
// -----------------------------------------------------------------------

#[test]
fn error_display_unknown_model() {
    let err = CameraIntrinsicsError::UnknownModel("FANCY".to_string());
    let msg = format!("{err}");
    assert_eq!(msg, "unknown camera model: FANCY");
}

#[test]
fn error_display_missing_parameter() {
    let err = CameraIntrinsicsError::MissingParameter {
        model: "PINHOLE".to_string(),
        parameter: "focal_length_x".to_string(),
    };
    let msg = format!("{err}");
    assert_eq!(
        msg,
        "missing parameter 'focal_length_x' for camera model 'PINHOLE'"
    );
}

#[test]
fn error_display_invalid_parameter() {
    let err = CameraIntrinsicsError::InvalidParameter {
        model: "SFMTOOL_FISHEYE".to_string(),
        parameter: "bspline_coeff_count".to_string(),
    };
    let msg = format!("{err}");
    assert_eq!(
        msg,
        "invalid parameter 'bspline_coeff_count' for camera model 'SFMTOOL_FISHEYE'"
    );
}

// -----------------------------------------------------------------------
// Dual-focal distortion models preserve separate fx, fy
// -----------------------------------------------------------------------

#[test]
fn focal_lengths_dual_focal_distortion_models() {
    for cam in [opencv(), opencv_fisheye(), full_opencv()] {
        let (fx, fy) = cam.focal_lengths();
        assert_relative_eq!(fx, 500.0);
        assert_relative_eq!(fy, 502.0);
    }
}

// -----------------------------------------------------------------------
// is_fisheye(): true for all fisheye variants, false for perspective models
// -----------------------------------------------------------------------

#[test]
fn is_fisheye_true_for_fisheye_models() {
    assert!(equidistant_fisheye().model.is_fisheye());
    assert!(sfmtool_fisheye().model.is_fisheye());
    assert!(simple_radial_fisheye().model.is_fisheye());
    assert!(radial_fisheye().model.is_fisheye());
    assert!(opencv_fisheye().model.is_fisheye());
    assert!(thin_prism_fisheye().model.is_fisheye());
    assert!(rad_tan_thin_prism_fisheye().model.is_fisheye());
}

#[test]
fn is_fisheye_false_for_perspective_models() {
    assert!(!pinhole().model.is_fisheye());
    assert!(!simple_pinhole().model.is_fisheye());
    assert!(!simple_radial().model.is_fisheye());
    assert!(!radial().model.is_fisheye());
    assert!(!opencv().model.is_fisheye());
    assert!(!full_opencv().model.is_fisheye());
    assert!(!equirectangular().model.is_fisheye());
    // The spline PINHOLE is a perspective model: same spline machinery as its
    // fisheye sibling, on the pinhole's radial coordinate.
    assert!(!sfmtool_pinhole().model.is_fisheye());
}

#[test]
fn is_equirectangular() {
    assert!(equirectangular().model.is_equirectangular());
    assert!(!pinhole().model.is_equirectangular());
    assert!(!opencv_fisheye().model.is_equirectangular());
}

#[test]
fn equirectangular_has_no_distortion() {
    assert!(!equirectangular().has_distortion());
}

// -----------------------------------------------------------------------
// has_distortion for new fisheye models
// -----------------------------------------------------------------------

#[test]
fn has_distortion_false_for_zero_coefficient_fisheye() {
    // OpenCVFisheye with all-zero coefficients should report no distortion
    let fe = CameraIntrinsics {
        model: CameraModel::OpenCVFisheye {
            focal_length_x: 500.0,
            focal_length_y: 502.0,
            principal_point_x: 320.0,
            principal_point_y: 240.0,
            radial_distortion_k1: 0.0,
            radial_distortion_k2: 0.0,
            radial_distortion_k3: 0.0,
            radial_distortion_k4: 0.0,
        },
        width: 640,
        height: 480,
    };
    assert!(!fe.has_distortion());
}

#[test]
fn has_distortion_true_for_distortion_fisheye_models() {
    assert!(simple_radial_fisheye().has_distortion());
    assert!(radial_fisheye().has_distortion());
    assert!(thin_prism_fisheye().has_distortion());
    assert!(rad_tan_thin_prism_fisheye().has_distortion());
}

// -----------------------------------------------------------------------
// EquidistantFisheye classification: a fisheye with no distortion, and one
// of the two ray-path models carrying an analytic pixel Jacobian.
// -----------------------------------------------------------------------

#[test]
fn equidistant_fisheye_classification() {
    let m = equidistant_fisheye().model;
    assert_eq!(m.model_name(), "EQUIDISTANT_FISHEYE");
    assert!(m.is_fisheye());
    assert!(!m.is_equirectangular());
    assert!(m.needs_ray_path());
    // The `θ = r/f` map carries no distortion coefficients at all.
    assert!(!m.has_distortion());
    // …yet differentiates in closed form, as does the one-coefficient
    // `θ_d = θ·(1 + k1·θ²)` that extends it. The multi-coefficient fisheye
    // models and equirectangular do not.
    assert!(m.supports_pixel_jacobian());
    assert!(simple_radial_fisheye().model.supports_pixel_jacobian());
    assert!(!radial_fisheye().model.supports_pixel_jacobian());
    assert!(!equirectangular().model.supports_pixel_jacobian());
    assert!(simple_pinhole().model.supports_pixel_jacobian());
}

// -----------------------------------------------------------------------
// SfmtoolFisheye: equidistant base + radial spline. Classification,
// variable-length serialization, and the gapped-spline error.
// -----------------------------------------------------------------------

#[test]
fn sfmtool_fisheye_classification() {
    let m = sfmtool_fisheye().model;
    assert_eq!(m.model_name(), "SFMTOOL_FISHEYE");
    assert!(m.is_fisheye());
    assert!(!m.is_equirectangular());
    assert!(m.needs_ray_path());
    // The radial spline counts as distortion when any coefficient is live…
    assert!(m.has_distortion());
    // …and the θ-map differentiates in closed form like the rest of the trio.
    assert!(m.supports_pixel_jacobian());
}

#[test]
fn sfmtool_fisheye_zero_or_empty_bspline_has_no_distortion() {
    for bspline in [vec![], vec![0.0; 8]] {
        let m = CameraModel::SfmtoolFisheye {
            focal_length: 500.0,
            principal_point_x: 320.0,
            principal_point_y: 240.0,
            bspline_theta_max: 2.0,
            bspline,
        };
        assert!(!m.has_distortion());
        // Still classified by shape, not by coefficient values.
        assert!(m.is_fisheye());
        assert!(m.supports_pixel_jacobian());
    }
}

#[test]
fn sfmtool_fisheye_inactive_spline_has_no_distortion_however_live_the_coefficients() {
    // `has_distortion` must agree with what the kernels actually run. They
    // short-circuit an INACTIVE spline — one below the cubic minimum, or one
    // whose domain end is not positive and finite — to the exact equidistant
    // arithmetic, so neither shape distorts anything.
    let live = vec![-0.001, -0.004, -0.01, -0.02, -0.03, -0.05, -0.07, -0.09];
    let cases: Vec<(f64, Vec<f64>)> = vec![
        // One coefficient is below the cubic minimum: no spline to evaluate.
        (2.0, vec![0.4]),
        // Live coefficients, degenerate domain end.
        (0.0, live.clone()),
        (-1.0, live.clone()),
        (f64::INFINITY, live.clone()),
        (f64::NEG_INFINITY, live.clone()),
        (f64::NAN, live.clone()),
    ];
    for (bspline_theta_max, bspline) in cases {
        let m = CameraModel::SfmtoolFisheye {
            focal_length: 500.0,
            principal_point_x: 320.0,
            principal_point_y: 240.0,
            bspline_theta_max,
            bspline,
        };
        assert!(
            !m.has_distortion(),
            "θ_max = {bspline_theta_max} reported distortion"
        );
        // Still classified by shape, not by coefficient values.
        assert!(m.is_fisheye());
        assert!(m.supports_pixel_jacobian());
    }
    // A real spline on a real domain still reports distortion.
    assert!(CameraModel::SfmtoolFisheye {
        focal_length: 500.0,
        principal_point_x: 320.0,
        principal_point_y: 240.0,
        bspline_theta_max: 2.0,
        bspline: live,
    }
    .has_distortion());
}

#[test]
fn sfmtool_fisheye_serializes_the_bspline_as_indexed_parameters() {
    let cam = sfmtool_fisheye();
    let stored = SfmrCamera::from(&cam);
    assert_eq!(stored.model, "SFMTOOL_FISHEYE");
    // The five-parameter head plus one key per coefficient.
    assert_eq!(stored.parameters.len(), 5 + 8);
    assert_eq!(stored.parameters["bspline_theta_max"], 2.0);
    assert_eq!(stored.parameters["bspline_coeff_count"], 8.0);
    assert_eq!(stored.parameters["bspline_c0"], -0.001);
    assert_eq!(stored.parameters["bspline_c7"], -0.09);
    // Round trip through the on-disk representation, all 8 coefficients.
    assert_eq!(CameraIntrinsics::try_from(&stored).unwrap(), cam);
}

#[test]
fn sfmtool_fisheye_empty_bspline_round_trips() {
    let cam = CameraIntrinsics {
        model: CameraModel::SfmtoolFisheye {
            focal_length: 500.0,
            principal_point_x: 320.0,
            principal_point_y: 240.0,
            bspline_theta_max: 2.0,
            bspline: vec![],
        },
        width: 640,
        height: 480,
    };
    let stored = SfmrCamera::from(&cam);
    // The declared length carries the empty spline; no coefficient keys.
    assert_eq!(stored.parameters["bspline_coeff_count"], 0.0);
    assert!(!stored.parameters.keys().any(|k| k
        .strip_prefix("bspline_c")
        .is_some_and(|i| i.parse::<usize>().is_ok())));
    assert_eq!(CameraIntrinsics::try_from(&stored).unwrap(), cam);
}

#[test]
fn sfmtool_fisheye_gapped_bspline_is_a_missing_parameter() {
    // bspline_c1 is absent while bspline_c2 exists under a declared length of
    // four: the read must report the hole rather than silently truncating or
    // skipping it.
    let mut stored = SfmrCamera::from(&sfmtool_fisheye());
    stored.parameters.clear();
    stored.parameters.insert("focal_length".to_string(), 500.0);
    stored
        .parameters
        .insert("principal_point_x".to_string(), 320.0);
    stored
        .parameters
        .insert("principal_point_y".to_string(), 240.0);
    stored
        .parameters
        .insert("bspline_theta_max".to_string(), 2.0);
    stored
        .parameters
        .insert("bspline_coeff_count".to_string(), 4.0);
    stored.parameters.insert("bspline_c0".to_string(), -0.001);
    stored.parameters.insert("bspline_c2".to_string(), -0.01);
    stored.parameters.insert("bspline_c3".to_string(), -0.02);
    let err = CameraIntrinsics::try_from(&stored).unwrap_err();
    assert!(matches!(
        err,
        CameraIntrinsicsError::MissingParameter {
            ref model,
            ref parameter,
        } if model == "SFMTOOL_FISHEYE" && parameter == "bspline_c1"
    ));
}

#[test]
fn sfmtool_fisheye_missing_theta_max_is_a_missing_parameter() {
    let mut stored = SfmrCamera::from(&sfmtool_fisheye());
    stored.parameters.remove("bspline_theta_max");
    let err = CameraIntrinsics::try_from(&stored).unwrap_err();
    assert!(matches!(
        err,
        CameraIntrinsicsError::MissingParameter { ref parameter, .. }
            if parameter == "bspline_theta_max"
    ));
}

#[test]
fn sfmtool_fisheye_missing_coeff_count_is_a_missing_parameter() {
    // Without the declared length there is no spline to read: the key count
    // is deliberately NOT a fallback.
    let mut stored = SfmrCamera::from(&sfmtool_fisheye());
    stored.parameters.remove("bspline_coeff_count");
    let err = CameraIntrinsics::try_from(&stored).unwrap_err();
    assert!(matches!(
        err,
        CameraIntrinsicsError::MissingParameter {
            ref model,
            ref parameter,
        } if model == "SFMTOOL_FISHEYE" && parameter == "bspline_coeff_count"
    ));
}

#[test]
fn sfmtool_fisheye_non_integer_coeff_count_is_an_invalid_parameter() {
    // A count must be a finite non-negative integer; each of these is stored
    // as an f64 that is not one.
    for bad in [2.5, -1.0, f64::NAN, f64::INFINITY] {
        let mut stored = SfmrCamera::from(&sfmtool_fisheye());
        stored
            .parameters
            .insert("bspline_coeff_count".to_string(), bad);
        let err = CameraIntrinsics::try_from(&stored).unwrap_err();
        assert!(
            matches!(
                err,
                CameraIntrinsicsError::InvalidParameter {
                    ref model,
                    ref parameter,
                } if model == "SFMTOOL_FISHEYE" && parameter == "bspline_coeff_count"
            ),
            "bspline_coeff_count = {bad} was accepted: {err}"
        );
    }
}

#[test]
fn sfmtool_fisheye_single_coefficient_count_is_an_invalid_parameter() {
    // A clamped cubic basis needs at least two coefficients. Zero is the
    // empty spline and stays valid; exactly one is a length the model does not
    // define, and reading it as the identity would hide a corrupt file.
    let mut stored = SfmrCamera::from(&sfmtool_fisheye());
    stored.parameters.retain(|k, _| {
        !k.strip_prefix("bspline_c")
            .is_some_and(|i| i.parse::<usize>().is_ok())
    });
    stored
        .parameters
        .insert("bspline_coeff_count".to_string(), 1.0);
    stored.parameters.insert("bspline_c0".to_string(), -0.001);
    let err = CameraIntrinsics::try_from(&stored).unwrap_err();
    assert!(
        matches!(
            err,
            CameraIntrinsicsError::InvalidParameter {
                ref model,
                ref parameter,
            } if model == "SFMTOOL_FISHEYE" && parameter == "bspline_coeff_count"
        ),
        "a one-coefficient spline was accepted: {err}"
    );
}

#[test]
fn sfmtool_fisheye_degenerate_theta_max_is_an_invalid_parameter() {
    // The domain end must be a real interval. Zero and negative leave the
    // basis nothing to live on; `±∞` and NaN are not a domain at all.
    for bad in [0.0, -1.0, f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
        let mut stored = SfmrCamera::from(&sfmtool_fisheye());
        stored
            .parameters
            .insert("bspline_theta_max".to_string(), bad);
        let err = CameraIntrinsics::try_from(&stored).unwrap_err();
        assert!(
            matches!(
                err,
                CameraIntrinsicsError::InvalidParameter {
                    ref model,
                    ref parameter,
                } if model == "SFMTOOL_FISHEYE" && parameter == "bspline_theta_max"
            ),
            "bspline_theta_max = {bad} was accepted: {err}"
        );
    }
}

#[test]
fn sfmtool_fisheye_coefficient_beyond_the_declared_length_is_an_invalid_parameter() {
    // Eight coefficients declared, nine present: the extra key is named,
    // rather than read as a ninth coefficient.
    let mut stored = SfmrCamera::from(&sfmtool_fisheye());
    stored.parameters.insert("bspline_c8".to_string(), -0.11);
    let err = CameraIntrinsics::try_from(&stored).unwrap_err();
    assert!(matches!(
        err,
        CameraIntrinsicsError::InvalidParameter {
            ref model,
            ref parameter,
        } if model == "SFMTOOL_FISHEYE" && parameter == "bspline_c8"
    ));
}

// -----------------------------------------------------------------------
// SfmtoolPinhole: pinhole base + radial spline. The same variable-length
// serialization as its fisheye sibling under `bspline_rho_max`, and the
// classification of a perspective model.
// -----------------------------------------------------------------------

#[test]
fn sfmtool_pinhole_classification() {
    let m = sfmtool_pinhole().model;
    assert_eq!(m.model_name(), "SFMTOOL_PINHOLE");
    // Perspective, not fisheye: the map has a perspective divide and its
    // domain is the half space in front of the camera.
    assert!(!m.is_fisheye());
    assert!(!m.is_equirectangular());
    assert!(!m.needs_ray_path());
    assert!(m.has_distortion());
    // The spline enters the perspective family's radial factor, so the
    // analytic pixel Jacobian is the family's.
    assert!(m.supports_pixel_jacobian());
}

#[test]
fn sfmtool_pinhole_zero_or_empty_bspline_has_no_distortion() {
    for bspline in [vec![], vec![0.0; 8]] {
        let m = CameraModel::SfmtoolPinhole {
            focal_length: 500.0,
            principal_point_x: 320.0,
            principal_point_y: 240.0,
            bspline_rho_max: 0.9,
            bspline,
        };
        assert!(!m.has_distortion());
        // Still classified by shape, not by coefficient values.
        assert!(!m.is_fisheye());
        assert!(m.supports_pixel_jacobian());
    }
}

#[test]
fn sfmtool_pinhole_inactive_spline_has_no_distortion_however_live_the_coefficients() {
    // `has_distortion` must agree with what the kernels actually run: an
    // INACTIVE spline — one below the cubic minimum, or one whose domain end
    // is not positive and finite — short-circuits to the exact pinhole
    // arithmetic, so neither shape distorts anything.
    let live = vec![
        0.0008, 0.0031, 0.0075, 0.0142, 0.0236, 0.0361, 0.052, 0.0718,
    ];
    let cases: Vec<(f64, Vec<f64>)> = vec![
        (0.9, vec![0.4]),
        (0.0, live.clone()),
        (-1.0, live.clone()),
        (f64::INFINITY, live.clone()),
        (f64::NEG_INFINITY, live.clone()),
        (f64::NAN, live.clone()),
    ];
    for (bspline_rho_max, bspline) in cases {
        let m = CameraModel::SfmtoolPinhole {
            focal_length: 500.0,
            principal_point_x: 320.0,
            principal_point_y: 240.0,
            bspline_rho_max,
            bspline,
        };
        assert!(
            !m.has_distortion(),
            "ρ_max = {bspline_rho_max} reported distortion"
        );
        assert!(!m.is_fisheye());
        assert!(m.supports_pixel_jacobian());
    }
    assert!(CameraModel::SfmtoolPinhole {
        focal_length: 500.0,
        principal_point_x: 320.0,
        principal_point_y: 240.0,
        bspline_rho_max: 0.9,
        bspline: live,
    }
    .has_distortion());
}

#[test]
fn sfmtool_pinhole_serializes_the_bspline_as_indexed_parameters() {
    let cam = sfmtool_pinhole();
    let stored = SfmrCamera::from(&cam);
    assert_eq!(stored.model, "SFMTOOL_PINHOLE");
    // The five-parameter head plus one key per coefficient.
    assert_eq!(stored.parameters.len(), 5 + 8);
    assert_eq!(stored.parameters["bspline_rho_max"], 0.9);
    assert_eq!(stored.parameters["bspline_coeff_count"], 8.0);
    assert_eq!(stored.parameters["bspline_c0"], 0.0008);
    assert_eq!(stored.parameters["bspline_c7"], 0.0718);
    // The domain end is named per model: the fisheye's key is absent.
    assert!(!stored.parameters.contains_key("bspline_theta_max"));
    assert_eq!(CameraIntrinsics::try_from(&stored).unwrap(), cam);
}

#[test]
fn sfmtool_pinhole_empty_bspline_round_trips() {
    let cam = CameraIntrinsics {
        model: CameraModel::SfmtoolPinhole {
            focal_length: 500.0,
            principal_point_x: 320.0,
            principal_point_y: 240.0,
            bspline_rho_max: 0.9,
            bspline: vec![],
        },
        width: 640,
        height: 480,
    };
    let stored = SfmrCamera::from(&cam);
    assert_eq!(stored.parameters["bspline_coeff_count"], 0.0);
    assert!(!stored.parameters.keys().any(|k| k
        .strip_prefix("bspline_c")
        .is_some_and(|i| i.parse::<usize>().is_ok())));
    assert_eq!(CameraIntrinsics::try_from(&stored).unwrap(), cam);
}

#[test]
fn sfmtool_pinhole_gapped_bspline_is_a_missing_parameter() {
    let mut stored = SfmrCamera::from(&sfmtool_pinhole());
    stored.parameters.clear();
    stored.parameters.insert("focal_length".to_string(), 500.0);
    stored
        .parameters
        .insert("principal_point_x".to_string(), 320.0);
    stored
        .parameters
        .insert("principal_point_y".to_string(), 240.0);
    stored.parameters.insert("bspline_rho_max".to_string(), 0.9);
    stored
        .parameters
        .insert("bspline_coeff_count".to_string(), 4.0);
    stored.parameters.insert("bspline_c0".to_string(), 0.0008);
    stored.parameters.insert("bspline_c2".to_string(), 0.0075);
    stored.parameters.insert("bspline_c3".to_string(), 0.0142);
    let err = CameraIntrinsics::try_from(&stored).unwrap_err();
    assert!(matches!(
        err,
        CameraIntrinsicsError::MissingParameter {
            ref model,
            ref parameter,
        } if model == "SFMTOOL_PINHOLE" && parameter == "bspline_c1"
    ));
}

#[test]
fn sfmtool_pinhole_missing_rho_max_is_a_missing_parameter() {
    let mut stored = SfmrCamera::from(&sfmtool_pinhole());
    stored.parameters.remove("bspline_rho_max");
    let err = CameraIntrinsics::try_from(&stored).unwrap_err();
    assert!(matches!(
        err,
        CameraIntrinsicsError::MissingParameter { ref parameter, .. }
            if parameter == "bspline_rho_max"
    ));
}

#[test]
fn sfmtool_pinhole_missing_coeff_count_is_a_missing_parameter() {
    let mut stored = SfmrCamera::from(&sfmtool_pinhole());
    stored.parameters.remove("bspline_coeff_count");
    let err = CameraIntrinsics::try_from(&stored).unwrap_err();
    assert!(matches!(
        err,
        CameraIntrinsicsError::MissingParameter {
            ref model,
            ref parameter,
        } if model == "SFMTOOL_PINHOLE" && parameter == "bspline_coeff_count"
    ));
}

#[test]
fn sfmtool_pinhole_non_integer_coeff_count_is_an_invalid_parameter() {
    for bad in [2.5, -1.0, f64::NAN, f64::INFINITY] {
        let mut stored = SfmrCamera::from(&sfmtool_pinhole());
        stored
            .parameters
            .insert("bspline_coeff_count".to_string(), bad);
        let err = CameraIntrinsics::try_from(&stored).unwrap_err();
        assert!(
            matches!(
                err,
                CameraIntrinsicsError::InvalidParameter {
                    ref model,
                    ref parameter,
                } if model == "SFMTOOL_PINHOLE" && parameter == "bspline_coeff_count"
            ),
            "bspline_coeff_count = {bad} was accepted: {err}"
        );
    }
}

#[test]
fn sfmtool_pinhole_single_coefficient_count_is_an_invalid_parameter() {
    let mut stored = SfmrCamera::from(&sfmtool_pinhole());
    stored.parameters.retain(|k, _| {
        !k.strip_prefix("bspline_c")
            .is_some_and(|i| i.parse::<usize>().is_ok())
    });
    stored
        .parameters
        .insert("bspline_coeff_count".to_string(), 1.0);
    stored.parameters.insert("bspline_c0".to_string(), 0.0008);
    let err = CameraIntrinsics::try_from(&stored).unwrap_err();
    assert!(
        matches!(
            err,
            CameraIntrinsicsError::InvalidParameter {
                ref model,
                ref parameter,
            } if model == "SFMTOOL_PINHOLE" && parameter == "bspline_coeff_count"
        ),
        "a one-coefficient spline was accepted: {err}"
    );
}

#[test]
fn sfmtool_pinhole_degenerate_rho_max_is_an_invalid_parameter() {
    for bad in [0.0, -1.0, f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
        let mut stored = SfmrCamera::from(&sfmtool_pinhole());
        stored.parameters.insert("bspline_rho_max".to_string(), bad);
        let err = CameraIntrinsics::try_from(&stored).unwrap_err();
        assert!(
            matches!(
                err,
                CameraIntrinsicsError::InvalidParameter {
                    ref model,
                    ref parameter,
                } if model == "SFMTOOL_PINHOLE" && parameter == "bspline_rho_max"
            ),
            "bspline_rho_max = {bad} was accepted: {err}"
        );
    }
}

#[test]
fn sfmtool_pinhole_coefficient_beyond_the_declared_length_is_an_invalid_parameter() {
    let mut stored = SfmrCamera::from(&sfmtool_pinhole());
    stored.parameters.insert("bspline_c8".to_string(), 0.09);
    let err = CameraIntrinsics::try_from(&stored).unwrap_err();
    assert!(matches!(
        err,
        CameraIntrinsicsError::InvalidParameter {
            ref model,
            ref parameter,
        } if model == "SFMTOOL_PINHOLE" && parameter == "bspline_c8"
    ));
}

#[test]
fn radial_spline_reports_the_models_radial_coordinate() {
    use crate::camera::intrinsics::SplineRadial;
    let fisheye = sfmtool_fisheye();
    let (coeffs, d_max, radial) = fisheye.model.radial_spline().unwrap();
    assert_eq!(coeffs.len(), 8);
    assert_eq!(d_max, 2.0);
    assert_eq!(radial, SplineRadial::IncidenceAngle);
    let pinhole = sfmtool_pinhole();
    let (coeffs, d_max, radial) = pinhole.model.radial_spline().unwrap();
    assert_eq!(coeffs.len(), 8);
    assert_eq!(d_max, 0.9);
    assert_eq!(radial, SplineRadial::ImagePlaneRadius);
    // Every other model carries no spline.
    for cam in all_cameras() {
        let spline = cam.model.radial_spline();
        assert_eq!(
            spline.is_some(),
            matches!(
                cam.model,
                CameraModel::SfmtoolFisheye { .. } | CameraModel::SfmtoolPinhole { .. }
            ),
            "{} reported the wrong spline presence",
            cam.model_name()
        );
    }
}

#[test]
fn equidistant_fisheye_has_simple_pinhole_parameter_list() {
    let cam = equidistant_fisheye();
    let stored = SfmrCamera::from(&cam);
    assert_eq!(stored.model, "EQUIDISTANT_FISHEYE");
    let names: Vec<&str> = stored.parameters.keys().map(String::as_str).collect();
    assert_eq!(
        names,
        ["focal_length", "principal_point_x", "principal_point_y"]
    );
    assert_eq!(cam.focal_lengths(), (500.0, 500.0));
    assert_eq!(cam.principal_point(), (320.0, 240.0));
    // Round trip through the on-disk representation.
    assert_eq!(CameraIntrinsics::try_from(&stored).unwrap(), cam);
}
