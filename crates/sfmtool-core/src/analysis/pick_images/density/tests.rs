// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::analysis::pick_images::essential_split::{EssentialSplit, SplitNotMeasured};
use crate::camera::{CameraModel, RollingShutter, RollingShutterScanlines};

fn split(inliers: usize, p50: f64, p90: f64) -> EssentialSplitOutcome {
    EssentialSplitOutcome::Measured(EssentialSplit {
        matches: inliers + 10,
        inliers,
        rotation_deg: 3.0,
        parallax_p50_deg: p50,
        parallax_p90_deg: p90,
    })
}

#[test]
fn a_step_counts_only_with_enough_inliers_a_wide_p90_and_a_narrow_p50() {
    // Counts: its ratio.
    assert_eq!(step_measure(&split(50, 0.125, 1.5)), 12.0);
    assert_eq!(step_measure(&split(400, 0.25, 0.375)), 1.5);
    // Each condition at its edge, then just past it.
    assert_eq!(step_measure(&split(49, 0.1, 1.2)), 1.0);
    assert_eq!(step_measure(&split(80, 0.1, 0.2)), 2.0);
    assert_eq!(step_measure(&split(80, 0.05, 0.199)), 1.0);
    assert_eq!(step_measure(&split(80, 0.499, 0.998)), 2.0);
    assert_eq!(step_measure(&split(80, 0.5, 5.0)), 1.0);
    // A P50 of 0 has no ratio, and a P90 that is not a number fails its floor.
    assert_eq!(step_measure(&split(80, 0.0, 1.0)), 1.0);
    assert_eq!(step_measure(&split(80, 0.1, f64::NAN)), 1.0);
    // Not measured.
    for reason in [
        SplitNotMeasured::TooFewMatches,
        SplitNotMeasured::NoConsensus,
        SplitNotMeasured::NoPoseInFront,
    ] {
        let outcome = EssentialSplitOutcome::NotMeasured {
            matches: 300,
            reason,
        };
        assert_eq!(step_measure(&outcome), 1.0);
    }
}

#[test]
fn the_gain_is_the_median_measure_over_its_unit_kept_within_its_bounds() {
    // Before the first step.
    assert_eq!(gain(&[], 3.0), (1.0, 1.0));
    // One or two measures: their median.
    assert_eq!(gain(&[5.0], 3.0), (5.0, 2.0));
    assert_eq!(gain(&[1.0, 6.0], 3.0), (3.5, 1.4));
    // Three: the middle one, whatever their order.
    assert_eq!(gain(&[20.0, 1.0, 5.0], 3.0), (5.0, 2.0));
    // Below the unit the gain stays 1, above the maximum it is the maximum.
    assert_eq!(gain(&[2.0, 2.4, 1.0], 3.0), (2.0, 1.0));
    assert_eq!(gain(&[30.0, 40.0, 1.0], 3.0), (30.0, 3.0));
    assert_eq!(gain(&[30.0, 40.0, 1.0], 1.5), (30.0, 1.5));
    assert_eq!(gain(&[30.0, 40.0, 1.0], 1.0), (30.0, 1.0));
}

fn pinhole(width: u32, height: u32, f: f64) -> CameraIntrinsics {
    CameraIntrinsics {
        model: CameraModel::SimplePinhole {
            focal_length: f,
            principal_point_x: f64::from(width) / 2.0,
            principal_point_y: f64::from(height) / 2.0,
        },
        width,
        height,
        rolling_shutter: None,
    }
}

#[test]
fn the_camera_is_scaled_to_the_frames_or_the_term_says_why_it_is_off() {
    let camera = pinhole(1920, 1080, 1500.0);
    // At the frames' size it is used as it is.
    let same = split_options(Some(&camera), 1920, 1080, 3.0).unwrap();
    assert_eq!(same.camera, camera);
    assert_eq!(same, EssentialSplitOptions::new(camera.clone()));
    // At half the size, scaled by a half.
    let half = split_options(Some(&camera), 960, 540, 3.0).unwrap();
    assert_eq!(half.camera, pinhole(960, 540, 750.0));
    // A gain of 1 measures nothing, with or without a camera.
    assert_eq!(
        split_options(Some(&camera), 1920, 1080, 1.0),
        Err(DensityTermOff::MaxGainOne)
    );
    assert_eq!(
        split_options(None, 1920, 1080, 3.0),
        Err(DensityTermOff::NoCamera)
    );
    // Another aspect cannot be scaled to.
    assert_eq!(
        split_options(Some(&camera), 1080, 1920, 3.0),
        Err(DensityTermOff::CameraShape {
            camera: [1920, 1080],
            frames: [1080, 1920],
        })
    );
    assert_eq!(
        split_options(Some(&pinhole(0, 1080, 1500.0)), 1920, 1080, 3.0),
        Err(DensityTermOff::CameraShape {
            camera: [0, 1080],
            frames: [1920, 1080],
        })
    );
    // A camera the split refuses, named by its problem.
    let mut shutter = camera.clone();
    shutter.rolling_shutter = Some(RollingShutter {
        scanlines: RollingShutterScanlines::Rows,
        rates: [0.01, 0.0, 0.0],
    });
    match split_options(Some(&shutter), 1920, 1080, 3.0) {
        Err(DensityTermOff::CameraRefused { problem }) => {
            assert!(problem.contains("SFMTOOL_PINHOLE"), "{problem}")
        }
        other => panic!("{other:?}"),
    }
    match split_options(Some(&pinhole(1920, 1080, 0.0)), 1920, 1080, 3.0) {
        Err(DensityTermOff::CameraRefused { problem }) => {
            assert!(problem.contains("focal"), "{problem}")
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn an_equidistant_fisheye_is_scaled_and_measured_with() {
    let camera = CameraIntrinsics {
        model: CameraModel::EquidistantFisheye {
            focal_length: 300.0,
            principal_point_x: 482.0,
            principal_point_y: 478.0,
        },
        width: 960,
        height: 960,
        rolling_shutter: None,
    };
    let options = split_options(Some(&camera), 480, 480, 3.0).unwrap();
    assert_eq!(
        options.camera.model,
        CameraModel::EquidistantFisheye {
            focal_length: 150.0,
            principal_point_x: 241.0,
            principal_point_y: 239.0,
        }
    );
}

#[test]
fn the_term_serializes_with_its_status_and_reason() {
    let json = |term: &DensityTerm| serde_json::to_value(term).unwrap();
    assert_eq!(
        json(&DensityTerm::On { focal_px: 750.0 }),
        serde_json::json!({"status": "on", "focal_px": 750.0})
    );
    assert_eq!(
        json(&DensityTerm::Off(DensityTermOff::NoCamera)),
        serde_json::json!({"status": "off", "reason": "no_camera"})
    );
    assert_eq!(
        json(&DensityTerm::Off(DensityTermOff::CameraShape {
            camera: [4, 3],
            frames: [3, 4],
        })),
        serde_json::json!({"status": "off", "reason": "camera_shape",
                           "camera": [4, 3], "frames": [3, 4]})
    );
}
