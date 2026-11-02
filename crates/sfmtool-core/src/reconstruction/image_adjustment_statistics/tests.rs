// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The statistics of a camera's per-image parameters: the arithmetic of one
//! component, which images they are over, the focal's log-space mean, and
//! which parameters are reported.
//!
//! The fixture is the demo reconstruction with images 0 to 3 through camera 0
//! and the rest through camera 1. Camera 0's posed images 0 to 2 carry values
//! whose statistics are exact in binary, and its image 3 is unposed and
//! carries a value far from the others.

use crate::camera::IntrinsicAdjustment;
use crate::reconstruction::bundle_adjust::ImageAdjustmentParameter;
use crate::reconstruction::drop_image_adjustments::drop_image_adjustments;
use crate::reconstruction::PerImageAdjustments;
use crate::SfmrReconstruction;

use super::*;

const UNPOSED: usize = 3;

fn scene(flags: PerImageAdjustments) -> SfmrReconstruction {
    let mut recon = SfmrReconstruction::demo(40);
    let camera = recon.image_set.cameras[0].clone();
    recon.image_set.cameras = vec![camera.clone(), camera];
    for (i, image) in recon.image_set.images.iter_mut().enumerate() {
        image.camera_index = u32::from(i >= 4);
    }
    recon.image_set.images[UNPOSED].translation_xyz.x = f64::NAN;
    recon.image_set.per_image_adjustments = vec![flags, PerImageAdjustments::default()];
    let values: [f64; 4] = [0.25, -0.25, 0.0, 4.0];
    for (i, image) in recon.image_set.images.iter_mut().enumerate().take(4) {
        let v = values[i];
        image.intrinsic_adjustment = IntrinsicAdjustment {
            focal: (v / 8.0).exp_m1(),
            principal_point: [v / 16.0, -v / 32.0],
            radial_bspline: [v / 64.0, v / 128.0],
            rolling_shutter: [v / 256.0, -v / 256.0, v / 512.0],
        };
    }
    recon
}

fn every_parameter() -> PerImageAdjustments {
    PerImageAdjustments {
        focal: true,
        principal_point: true,
        radial_bspline: true,
        rolling_shutter: true,
        rolling_shutter_scanlines: None,
    }
}

#[test]
fn one_components_statistics() {
    let s = ComponentStatistics::of(&[1.0, 2.0, 3.0, 4.0, 100.0]).unwrap();
    assert_eq!(s.mean, 22.0);
    assert_eq!((s.min, s.max), (1.0, 100.0));
    // Population deviation: the root of the mean squared distance.
    let var = [21.0f64, 20.0, 19.0, 18.0, 78.0]
        .iter()
        .map(|d| d * d)
        .sum::<f64>()
        / 5.0;
    assert_eq!(s.std, var.sqrt());
    // Five values cannot put one past two deviations: the 100 is 78 from the
    // mean, just inside 2 · 39.01.
    assert_eq!(s.beyond_two_std, 0);
    // Nine zeros and a 10: mean 1, deviation 3, and the 10 is 9 out.
    let mut values = vec![0.0; 9];
    values.push(10.0);
    let s = ComponentStatistics::of(&values).unwrap();
    assert_eq!((s.mean, s.std, s.beyond_two_std), (1.0, 3.0, 1));

    let flat = ComponentStatistics::of(&[0.0, 0.0, 0.0]).unwrap();
    assert_eq!(
        flat,
        ComponentStatistics {
            mean: 0.0,
            std: 0.0,
            min: 0.0,
            max: 0.0,
            beyond_two_std: 0
        }
    );
    assert_eq!(ComponentStatistics::of(&[]), None);
}

#[test]
fn the_statistics_are_over_the_posed_images_in_stored_units() {
    let recon = scene(every_parameter());
    let s = image_adjustment_statistics(&recon.image_set, 0).unwrap();
    assert_eq!(s.images, vec![0, 1, 2]);
    // The unposed image's 4.0 would move every mean and the maximum.
    let pp = s.principal_point.unwrap();
    assert_eq!(pp[0].mean, 0.0);
    assert_eq!((pp[0].min, pp[0].max), (-0.25 / 16.0, 0.25 / 16.0));
    assert_eq!((pp[1].min, pp[1].max), (-0.25 / 32.0, 0.25 / 32.0));
    let radial = s.radial_bspline.unwrap();
    assert_eq!(radial[1].max, 0.25 / 128.0);
    let rates = s.rolling_shutter.unwrap();
    assert_eq!(rates[2].max, 0.25 / 512.0);
    assert_eq!(rates[1].min, -0.25 / 256.0);
}

#[test]
fn a_camera_with_no_posed_image_is_described_by_all_its_images() {
    let mut recon = scene(every_parameter());
    for image in recon.image_set.images.iter_mut().take(4) {
        image.translation_xyz.x = f64::NAN;
    }
    let s = image_adjustment_statistics(&recon.image_set, 0).unwrap();
    assert_eq!(s.images, vec![0, 1, 2, 3]);
    assert_eq!(s.principal_point.unwrap()[0].max, 4.0 / 16.0);
}

#[test]
fn the_focal_mean_is_the_scale_a_drop_folds_into_the_camera() {
    let recon = scene(every_parameter());
    let s = image_adjustment_statistics(&recon.image_set, 0).unwrap();
    let focal = s.focal.unwrap();
    // The values are `exp(v/8) − 1` for v = ±0.25 and 0, so the geometric
    // mean of `1 + s` is 1 and the deviation is that of v/8, while the
    // arithmetic mean of `s` is above zero.
    assert!(focal.mean.abs() < 1e-15, "{}", focal.mean);
    let arithmetic = [0usize, 1, 2]
        .iter()
        .map(|&i| recon.image_set.images[i].intrinsic_adjustment.focal)
        .sum::<f64>()
        / 3.0;
    assert!(arithmetic > 1e-4, "{arithmetic}");
    let expected_std = ((2.0 * (0.25f64 / 8.0).powi(2)) / 3.0).sqrt();
    assert!((focal.std - expected_std).abs() < 1e-15);
    assert!((focal.min - (-0.25f64 / 8.0).exp_m1()).abs() < 1e-15);
    assert!((focal.max - (0.25f64 / 8.0).exp_m1()).abs() < 1e-15);

    // Against the drop itself, on a mean that is not zero, with the camera
    // varying the focal alone, which the demo's `PINHOLE` can carry.
    let mut recon = scene(PerImageAdjustments {
        focal: true,
        ..Default::default()
    });
    for image in recon.image_set.images.iter_mut() {
        let focal = (1.0 + image.intrinsic_adjustment.focal) * 1.02 - 1.0;
        image.intrinsic_adjustment = IntrinsicAdjustment {
            focal,
            ..Default::default()
        };
    }
    recon.image_set.images[UNPOSED].intrinsic_adjustment.focal = 0.0;
    for image in recon.image_set.images.iter_mut().skip(4) {
        image.intrinsic_adjustment.focal = 0.0;
    }
    let s = image_adjustment_statistics(&recon.image_set, 0).unwrap();
    let dropped =
        drop_image_adjustments(&recon, 0, &[ImageAdjustmentParameter::Focal]).expect("drops");
    let before = recon.image_set.cameras[0].focal_lengths().0;
    let after = dropped.image_set.cameras[0].focal_lengths().0;
    let mean = s.focal.unwrap().mean;
    assert!((mean - 0.02).abs() < 1e-12, "{mean}");
    assert!((after - before * (1.0 + mean)).abs() < 1e-9 * before);
}

#[test]
fn only_the_parameters_the_camera_varies_are_reported() {
    let flags = PerImageAdjustments {
        focal: true,
        principal_point: true,
        ..Default::default()
    };
    let mut recon = scene(flags);
    // A varied parameter whose values are all zero is reported as zero.
    for image in recon.image_set.images.iter_mut() {
        image.intrinsic_adjustment.principal_point = [0.0; 2];
    }
    let s = image_adjustment_statistics(&recon.image_set, 0).unwrap();
    assert!(s.focal.is_some());
    let pp = s.principal_point.unwrap();
    assert_eq!((pp[0].mean, pp[0].std, pp[0].max), (0.0, 0.0, 0.0));
    // The images carry radial and rate values the camera does not vary.
    assert_eq!(s.radial_bspline, None);
    assert_eq!(s.rolling_shutter, None);

    let other = image_adjustment_statistics(&recon.image_set, 1).unwrap();
    assert_eq!(other.images, vec![4, 5, 6, 7]);
    assert_eq!(
        (
            other.focal,
            other.principal_point,
            other.radial_bspline,
            other.rolling_shutter
        ),
        (None, None, None, None)
    );
}

#[test]
fn an_unknown_camera_is_refused() {
    let recon = scene(every_parameter());
    assert_eq!(
        image_adjustment_statistics(&recon.image_set, 2),
        Err(UnknownCamera {
            camera: 2,
            count: 2
        })
    );
}
