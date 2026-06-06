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
    ];
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
        parameters: HashMap::new(),
    };
    let err = CameraIntrinsics::try_from(&sfmr).unwrap_err();
    assert!(
        matches!(err, CameraIntrinsicsError::UnknownModel(ref name) if name == "UNKNOWN_MODEL")
    );
}

#[test]
fn try_from_missing_parameter() {
    let mut params = HashMap::new();
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
