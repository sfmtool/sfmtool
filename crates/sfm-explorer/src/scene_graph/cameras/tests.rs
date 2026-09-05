// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The camera row's text, on its own.
//!
//! No frame and no fixture: a camera row is a single formatted line, and what
//! it must say about a model — one focal length or two, and whether the
//! parameterization is still moving — is decided entirely by the intrinsics
//! handed to it. That the row is drawn, hoverable and selectable is the
//! panel's business and is covered in `scene_graph/tests.rs`.

use sfmtool_core::{CameraIntrinsics, CameraModel};

use super::camera_row_text;

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
    };
    assert_eq!(
        camera_row_text(0, &square, 26),
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
    };
    assert_eq!(
        camera_row_text(1, &anamorphic, 1),
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
    };
    assert_eq!(
        camera_row_text(0, &beta, 0),
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
    };
    assert!(
        !camera_row_text(0, &settled, 3).contains('β'),
        "a settled model must not be marked beta"
    );
}
