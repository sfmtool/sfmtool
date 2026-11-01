// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Dropping per-image parameters from one camera: each fold, the parameters a
//! camera does not vary, other cameras, the scanlines setting and the
//! refusals.
//!
//! The fixture is the demo reconstruction through two `SFMTOOL_PINHOLE`
//! cameras, images 0 to 3 through camera 0 and 4 to 7 through camera 1, with
//! inline keypoints at the exact projections through each image's intrinsics.
//! Camera 0's posed images 0 to 2 carry `m + d`, `m − d` and `m` of each
//! parameter, so image 2 holds their mean exactly in binary, with the focal
//! scale's `m ± d` and `m` in log space, `ln(1 + s)`, the space its mean is
//! taken in; image 3 is unposed and carries a value far from the others.

use ndarray::Array2;

use crate::camera::{CameraIntrinsics, CameraModel, IntrinsicAdjustment, RollingShutterScanlines};
use crate::reconstruction::data::adjusted_fixture::adjust_and_reproject;
use crate::reconstruction::data::ObservationSource;
use crate::reconstruction::PerImageAdjustments;

use super::*;

use ImageAdjustmentParameter::{Focal, PrincipalPoint, RadialBspline, RollingShutter};

const ALL: [ImageAdjustmentParameter; 4] = [Focal, PrincipalPoint, RadialBspline, RollingShutter];

/// The image that holds the mean of camera 0's posed images.
const MEAN_IMAGE: usize = 2;
/// Camera 0's unposed image.
const UNPOSED: usize = 3;

fn pinhole(focal: f64) -> CameraIntrinsics {
    CameraIntrinsics {
        model: CameraModel::SfmtoolPinhole {
            focal_length: focal,
            principal_point_x: 960.0,
            principal_point_y: 540.0,
            bspline_rho_max: 1.1,
            bspline: vec![0.002, 0.006, 0.012, 0.02],
        },
        width: 1920,
        height: 1080,
        rolling_shutter: None,
    }
}

/// `m + d`, `m − d` and `m` for slots 0, 1 and 2, and `far` for slot 3.
fn spread(slot: usize, m: f64, d: f64, far: f64) -> f64 {
    match slot {
        0 => m + d,
        1 => m - d,
        2 => m,
        _ => far,
    }
}

fn adjustment(i: usize) -> IntrinsicAdjustment {
    // Camera 1's images get values of their own, to show they are left alone.
    let (slot, k) = if i < 4 { (i, 1.0) } else { (i - 4, -0.5) };
    IntrinsicAdjustment {
        focal: (k * spread(slot, 0.03125, 0.015625, 0.5)).exp_m1(),
        principal_point: [
            k * spread(slot, 0.0078125, 0.00390625, 0.25),
            k * spread(slot, -0.00390625, 0.001953125, -0.25),
        ],
        radial_bspline: [
            k * spread(slot, 0.0009765625, 0.00048828125, 0.125),
            k * spread(slot, -0.001953125, 0.0009765625, -0.125),
        ],
        rolling_shutter: [1e-3 * (i as f64 + 1.0), -2e-3, 5e-4 * i as f64],
    }
}

fn every_parameter() -> PerImageAdjustments {
    PerImageAdjustments {
        focal: true,
        principal_point: true,
        radial_bspline: true,
        rolling_shutter: true,
        rolling_shutter_scanlines: Some(RollingShutterScanlines::Rows),
    }
}

fn scene() -> SfmrReconstruction {
    let mut recon = SfmrReconstruction::demo(40);
    recon.image_set.cameras = vec![pinhole(1000.0), pinhole(1200.0)];
    for (i, image) in recon.image_set.images.iter_mut().enumerate() {
        image.camera_index = u32::from(i >= 4);
    }
    recon.image_set.images[UNPOSED].translation_xyz.x = f64::NAN;
    recon.image_set.per_image_adjustments = vec![every_parameter(); 2];
    let rows = recon.point_set.tracks.len();
    let images = recon.image_set.images.len();
    recon.point_set.observations = ObservationSource::EmbeddedPatches {
        keypoints_xy: Array2::zeros((rows, 2)),
        image_file_hashes: vec![[0u8; 16]; images],
    };
    recon.metadata.feature_source =
        sfmtool_sfmr_format::FEATURE_SOURCE_EMBEDDED_PATCHES.to_string();
    recon.rebuild_derived_fields();
    adjust_and_reproject(&mut recon, adjustment);
    recon.recompute_point_errors().unwrap();
    recon.image_set.check_image_intrinsics().unwrap();
    recon
}

#[test]
fn each_fold_keeps_the_mean_image_as_it_was() {
    // The fold is the mean adjustment applied to the camera, so the image that
    // held the mean is read through the same arithmetic before and after. The
    // principal point and the radial correction are linear means of values
    // exact in binary, so those intrinsics agree exactly; the focal's mean is
    // taken through `ln(1 + s)` and back, so it agrees to rounding. The rates
    // are not folded, so they are dropped in their own test.
    let recon = scene();
    let before = recon.image_intrinsics(MEAN_IMAGE).into_owned();
    for parameters in [
        &[Focal][..],
        &[PrincipalPoint],
        &[RadialBspline],
        &[Focal, PrincipalPoint, RadialBspline],
    ] {
        let out = drop_image_adjustments(&recon, 0, parameters).unwrap();
        let after = out.image_intrinsics(MEAN_IMAGE).into_owned();
        if parameters.contains(&Focal) {
            let (f_after, f_before) = (after.focal_lengths().0, before.focal_lengths().0);
            assert!(
                (f_after - f_before).abs() <= 1e-12 * f_before,
                "{parameters:?}: {f_after} against {f_before}"
            );
            assert_eq!(after.principal_point(), before.principal_point());
            assert_eq!(after.model.radial_spline(), before.model.radial_spline());
        } else {
            assert_eq!(after, before, "{parameters:?}");
        }
        // The other posed images move: their values were not the mean.
        assert_ne!(
            out.image_intrinsics(0).into_owned(),
            recon.image_intrinsics(0).into_owned(),
            "{parameters:?}"
        );
        let flags = out.image_set.camera_per_image_adjustments(0);
        for &p in parameters {
            let (flag, zero) = match p {
                Focal => (
                    flags.focal,
                    (0..4).all(|i| out.image_set.images[i].intrinsic_adjustment.focal == 0.0),
                ),
                PrincipalPoint => (
                    flags.principal_point,
                    (0..4).all(|i| {
                        out.image_set.images[i].intrinsic_adjustment.principal_point == [0.0; 2]
                    }),
                ),
                RadialBspline => (
                    flags.radial_bspline,
                    (0..4).all(|i| {
                        out.image_set.images[i].intrinsic_adjustment.radial_bspline == [0.0; 2]
                    }),
                ),
                RollingShutter => unreachable!(),
            };
            assert!(!flag && zero, "{p:?}");
        }
        // What was not named is kept.
        assert!(flags.rolling_shutter);
        assert_eq!(
            out.image_set.images[0].intrinsic_adjustment.rolling_shutter,
            recon.image_set.images[0]
                .intrinsic_adjustment
                .rolling_shutter
        );
    }
}

#[test]
fn the_fold_is_the_mean_over_the_posed_images() {
    // The unposed image's far value is left out of the mean.
    let recon = scene();
    let out = drop_image_adjustments(&recon, 0, &[Focal, PrincipalPoint, RadialBspline]).unwrap();
    let CameraModel::SfmtoolPinhole {
        focal_length,
        principal_point_x,
        principal_point_y,
        bspline,
        ..
    } = &out.image_set.cameras[0].model
    else {
        panic!("still an SFMTOOL_PINHOLE");
    };
    // The focal takes the geometric mean of `1 + s`, `exp(m)`, not the
    // arithmetic one, which the spread of the scales puts measurably above it.
    let geometric = 1000.0 * 0.03125f64.exp();
    assert!(
        (*focal_length - geometric).abs() <= 1e-12 * geometric,
        "{focal_length} against {geometric}"
    );
    let arithmetic = 1000.0
        * (0..3)
            .map(|i| 1.0 + recon.image_set.images[i].intrinsic_adjustment.focal)
            .sum::<f64>()
        / 3.0;
    assert!(
        (arithmetic - geometric).abs() > 1e-6 * geometric,
        "{arithmetic} against {geometric}"
    );
    assert_eq!(*principal_point_x, 960.0 + 0.0078125 * 1080.0);
    assert_eq!(*principal_point_y, 540.0 - 0.00390625 * 1080.0);
    let k = crate::camera::distortion::bspline::radial_bspline_insertion(4);
    let c = [0.002, 0.006, 0.012, 0.02];
    let e = [0.0009765625, -0.001953125];
    for (j, (&got, row)) in bspline.iter().zip(&k).enumerate() {
        assert_eq!(
            got,
            c[j] + (row[0] * e[0] + row[1] * e[1]),
            "coefficient {j}"
        );
    }
    // The unposed image reads the folded camera with nothing of its own.
    assert_eq!(
        out.image_set.images[UNPOSED].intrinsic_adjustment.focal,
        0.0
    );
}

#[test]
fn a_drop_after_a_focal_solve_leaves_the_camera_focal_as_it_was() {
    // The bundle adjustment re-centres a per-image focal so that its images'
    // `ln(1 + s)` sum to zero. Their geometric mean is then 1, and the drop
    // folds nothing into the focal.
    let mut recon = scene();
    for (i, t) in [0.02, -0.02, 0.0].into_iter().enumerate() {
        recon.image_set.images[i].intrinsic_adjustment.focal = f64::exp_m1(t);
    }
    let sum: f64 = (0..3)
        .map(|i| recon.image_set.images[i].intrinsic_adjustment.focal.ln_1p())
        .sum();
    assert!(sum.abs() < 1e-15, "{sum}");
    // The arithmetic mean of `1 + s` is not 1 for these images.
    let arithmetic: f64 = (0..3)
        .map(|i| recon.image_set.images[i].intrinsic_adjustment.focal)
        .sum::<f64>()
        / 3.0;
    assert!(arithmetic > 1e-6, "{arithmetic}");

    let out = drop_image_adjustments(&recon, 0, &[Focal]).unwrap();
    assert_eq!(
        out.image_set.cameras[0].focal_lengths(),
        recon.image_set.cameras[0].focal_lengths()
    );
}

#[test]
fn dropping_the_rates_keeps_the_camera_and_its_scanlines_setting() {
    // A mean rate describes no photograph, so nothing is folded. The scanlines
    // setting says how the sensor reads out, which the format allows without
    // the flag, so it stays for a later solve of the rates.
    let recon = scene();
    let out = drop_image_adjustments(&recon, 0, &[RollingShutter]).unwrap();
    assert_eq!(out.image_set.cameras[0], recon.image_set.cameras[0]);
    let flags = out.image_set.camera_per_image_adjustments(0);
    assert!(!flags.rolling_shutter);
    assert!(flags.focal && flags.principal_point && flags.radial_bspline);
    assert_eq!(
        flags.rolling_shutter_scanlines,
        Some(RollingShutterScanlines::Rows)
    );
    for i in 0..4 {
        let a = &out.image_set.images[i].intrinsic_adjustment;
        let b = &recon.image_set.images[i].intrinsic_adjustment;
        assert_eq!(a.rolling_shutter, [0.0; 3]);
        assert_eq!(a.focal, b.focal);
        assert_eq!(a.radial_bspline, b.radial_bspline);
    }
    assert!(out.image_intrinsics(MEAN_IMAGE).rolling_shutter.is_none());
}

#[test]
fn other_cameras_and_their_images_are_untouched() {
    let recon = scene();
    let out = drop_image_adjustments(&recon, 0, &ALL).unwrap();
    assert_eq!(out.image_set.cameras[1], recon.image_set.cameras[1]);
    assert_eq!(
        out.image_set.camera_per_image_adjustments(1),
        every_parameter()
    );
    for i in 4..8 {
        assert_eq!(
            out.image_set.images[i].intrinsic_adjustment,
            recon.image_set.images[i].intrinsic_adjustment
        );
    }
    // Camera 0 keeps only its scanlines setting.
    assert_eq!(
        out.image_set.camera_per_image_adjustments(0),
        PerImageAdjustments {
            rolling_shutter_scanlines: Some(RollingShutterScanlines::Rows),
            ..Default::default()
        }
    );
    for (a, b) in out.image_set.images.iter().zip(&recon.image_set.images) {
        assert_eq!(a.quaternion_wxyz, b.quaternion_wxyz);
        assert_eq!(a.translation_xyz.y, b.translation_xyz.y);
    }
    for (a, b) in out.point_set.points.iter().zip(&recon.point_set.points) {
        assert_eq!(a.position, b.position);
    }
}

#[test]
fn a_camera_left_with_nothing_empties_the_column() {
    let mut recon = scene();
    recon.image_set.per_image_adjustments = vec![
        PerImageAdjustments {
            focal: true,
            ..Default::default()
        },
        PerImageAdjustments::default(),
    ];
    for image in &mut recon.image_set.images {
        image.intrinsic_adjustment = IntrinsicAdjustment {
            focal: image.intrinsic_adjustment.focal,
            ..Default::default()
        };
    }
    for image in &mut recon.image_set.images[4..] {
        image.intrinsic_adjustment.focal = 0.0;
    }
    let out = drop_image_adjustments(&recon, 0, &[Focal]).unwrap();
    assert!(out.image_set.per_image_adjustments.is_empty());
    assert!(out
        .image_set
        .images
        .iter()
        .all(|im| im.intrinsic_adjustment.is_zero()));
}

#[test]
fn dropping_what_the_camera_does_not_vary_changes_nothing() {
    let mut recon = scene();
    recon.image_set.per_image_adjustments = vec![
        PerImageAdjustments {
            focal: true,
            ..Default::default()
        },
        every_parameter(),
    ];
    for image in &mut recon.image_set.images[..4] {
        image.intrinsic_adjustment = IntrinsicAdjustment {
            focal: image.intrinsic_adjustment.focal,
            ..Default::default()
        };
    }
    for parameters in [&[][..], &[PrincipalPoint, RadialBspline, RollingShutter]] {
        let out = drop_image_adjustments(&recon, 0, parameters).unwrap();
        assert_eq!(out.image_set.cameras, recon.image_set.cameras);
        assert_eq!(
            out.image_set.per_image_adjustments,
            recon.image_set.per_image_adjustments
        );
        for (a, b) in out.image_set.images.iter().zip(&recon.image_set.images) {
            assert_eq!(a.intrinsic_adjustment, b.intrinsic_adjustment);
        }
    }
    // A value a camera does not list is still dropped and folded.
    recon.image_set.per_image_adjustments[0].focal = false;
    let out = drop_image_adjustments(&recon, 0, &[Focal]).unwrap();
    assert_ne!(out.image_set.cameras[0], recon.image_set.cameras[0]);
    assert_eq!(out.image_set.images[0].intrinsic_adjustment.focal, 0.0);
}

#[test]
fn the_stored_errors_are_rescored_through_the_new_lens() {
    let recon = scene();
    let out = drop_image_adjustments(&recon, 0, &ALL).unwrap();
    let mut rescored = out.clone();
    rescored.recompute_point_errors().unwrap();
    let mut moved = 0;
    for ((a, b), was) in out
        .point_set
        .points
        .iter()
        .zip(&rescored.point_set.points)
        .zip(&recon.point_set.points)
    {
        assert_eq!(a.rms_reprojection_error, b.rms_reprojection_error);
        if (a.rms_reprojection_error - was.rms_reprojection_error).abs() > 1e-3 {
            moved += 1;
        }
    }
    assert!(moved > 0, "no stored error changed");
}

#[test]
fn an_unknown_camera_is_refused() {
    let recon = scene();
    assert_eq!(
        drop_image_adjustments(&recon, 2, &[Focal]).err(),
        Some(DropImageAdjustmentsError::UnknownCamera {
            camera: 2,
            count: 2
        })
    );
}

#[test]
fn a_folded_lens_that_is_not_valid_is_refused() {
    // A mean radial correction that bends the spline back on itself.
    let mut recon = scene();
    for image in &mut recon.image_set.images[..4] {
        image.intrinsic_adjustment.radial_bspline = [0.0, -1.0];
    }
    let err = drop_image_adjustments(&recon, 0, &[RadialBspline]).err();
    assert_eq!(
        err,
        Some(DropImageAdjustmentsError::FoldedSplineNotMonotone { camera: 0 })
    );
    // A focal scale at or below −1 has no `ln(1 + s)`, so the mean is NaN and
    // leaves the camera no focal.
    let mut recon = scene();
    for image in &mut recon.image_set.images[..4] {
        image.intrinsic_adjustment.focal = -1.5;
    }
    let err = drop_image_adjustments(&recon, 0, &[Focal]).err().unwrap();
    assert!(
        matches!(err, DropImageAdjustmentsError::Fold { camera: 0, .. }),
        "{err}"
    );
    assert!(err.to_string().contains("not positive"), "{err}");
}
