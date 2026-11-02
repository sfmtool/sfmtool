// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The camera row's text, on its own.
//!
//! No frame and no fixture: a camera row is a single formatted line, and what
//! it must say about a model — one focal length or two, and whether the
//! parameterization is still moving — is decided entirely by the intrinsics
//! handed to it. That the row is drawn, hoverable and selectable is the
//! panel's business and is covered in `scene_graph/tests.rs`.

use sfmtool_core::camera::RollingShutterScanlines;
use sfmtool_core::reconstruction::PerImageAdjustments;
use sfmtool_core::{CameraIntrinsics, CameraModel};

use super::{camera_row_text, per_image_names};

/// One focal length reads `f`; two that differ read `fx/fy`. Written from the
/// model's own focal lengths, so a fisheye's `px/rad` and a pinhole's `px` are
/// both just the number the file carries.
#[test]
fn a_camera_row_names_one_focal_length_or_two() {
    let square = CameraIntrinsics {
        model: CameraModel::SimplePinhole {
            focal_length: 240.14,
            principal_point_x: 240.0,
            principal_point_y: 240.0,
        },
        width: 480,
        height: 480,
        rolling_shutter: None,
    };
    assert_eq!(
        camera_row_text(0, &square, 26, PerImageAdjustments::default()),
        "#0  SIMPLE_PINHOLE  480×480  f 240.1  26 images"
    );

    let anamorphic = CameraIntrinsics {
        model: CameraModel::Pinhole {
            focal_length_x: 240.14,
            focal_length_y: 239.72,
            principal_point_x: 240.0,
            principal_point_y: 240.0,
        },
        width: 480,
        height: 480,
        rolling_shutter: None,
    };
    assert_eq!(
        camera_row_text(1, &anamorphic, 1, PerImageAdjustments::default()),
        "#1  PINHOLE  480×480  f 240.1/239.7  1 image"
    );
}

/// A model whose parameterization is not yet frozen says so on the row; the
/// registry's note itself is in the hover tooltip, which egui hangs off the
/// whole row rather than off the `β`.
#[test]
fn a_beta_model_row_carries_the_beta_marker() {
    let beta = CameraIntrinsics {
        model: CameraModel::SfmtoolFisheye {
            focal_length: 240.0,
            principal_point_x: 240.0,
            principal_point_y: 240.0,
            bspline_theta_max: 1.6,
            bspline: vec![0.01, 0.02, 0.03, 0.04],
        },
        width: 480,
        height: 480,
        rolling_shutter: None,
    };
    assert_eq!(
        camera_row_text(0, &beta, 0, PerImageAdjustments::default()),
        "#0  SFMTOOL_FISHEYE β  480×480  f 240.0  0 images"
    );
    assert!(beta.model.beta_note().is_some());

    let settled = CameraIntrinsics {
        model: CameraModel::SimplePinhole {
            focal_length: 240.0,
            principal_point_x: 240.0,
            principal_point_y: 240.0,
        },
        width: 480,
        height: 480,
        rolling_shutter: None,
    };
    assert!(
        !camera_row_text(0, &settled, 3, PerImageAdjustments::default()).contains('β'),
        "a settled model must not be marked beta"
    );
}

/// A camera that varies parameters per image says which after its image
/// count, abbreviated in the file's order, and the tooltip names them in full
/// with the scanlines setting; a camera that varies none keeps its row as it
/// was and has no such line.
#[test]
fn a_camera_row_names_its_per_image_parameters() {
    let camera = CameraIntrinsics {
        model: CameraModel::SimplePinhole {
            focal_length: 240.14,
            principal_point_x: 240.0,
            principal_point_y: 240.0,
        },
        width: 480,
        height: 480,
        rolling_shutter: None,
    };
    let focal_pp = PerImageAdjustments {
        focal: true,
        principal_point: true,
        ..Default::default()
    };
    assert_eq!(
        camera_row_text(0, &camera, 26, focal_pp),
        "#0  SIMPLE_PINHOLE  480×480  f 240.1  26 images  per image: f, pp"
    );
    assert_eq!(
        per_image_names(focal_pp).as_deref(),
        Some("Varies per image: focal, principal point")
    );

    let every = PerImageAdjustments {
        focal: true,
        principal_point: true,
        radial_bspline: true,
        rolling_shutter: true,
        rolling_shutter_scanlines: Some(RollingShutterScanlines::Columns),
    };
    assert!(camera_row_text(0, &camera, 17, every)
        .ends_with("17 images  per image: f, pp, radial, shutter"));
    assert_eq!(
        per_image_names(every).as_deref(),
        Some(
            "Varies per image: focal, principal point, radial correction, \
             rolling-shutter rates (read on columns)"
        )
    );

    // A scanlines setting alone varies nothing.
    let scanlines_only = PerImageAdjustments {
        rolling_shutter_scanlines: Some(RollingShutterScanlines::Rows),
        ..Default::default()
    };
    assert_eq!(
        camera_row_text(0, &camera, 26, scanlines_only),
        "#0  SIMPLE_PINHOLE  480×480  f 240.1  26 images"
    );
    assert_eq!(per_image_names(scanlines_only), None);
}
