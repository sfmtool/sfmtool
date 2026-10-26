// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! An image's intrinsic adjustment: applied by `image_intrinsics`, carried with
//! its image through the edits that move images, and through a `.sfmr` file.

use std::borrow::Cow;

use super::*;
use crate::camera::{CameraModel, IntrinsicAdjustment};
use crate::reconstruction::PerImageAdjustments;

/// A distinct adjustment for image `i`, so one moved to the wrong image shows
/// up as the wrong value.
fn tagged(i: usize) -> IntrinsicAdjustment {
    let t = i as f64 + 1.0;
    IntrinsicAdjustment {
        focal: 0.001 * t,
        principal_point: [0.002 * t, -0.003 * t],
    }
}

/// The demo with every image's adjustment tagged and its one camera varying
/// both parameters.
fn adjusted_demo() -> SfmrReconstruction {
    let mut recon = SfmrReconstruction::demo(16);
    for (i, image) in recon.image_set.images.iter_mut().enumerate() {
        image.intrinsic_adjustment = tagged(i);
    }
    recon.image_set.per_image_adjustments = vec![PerImageAdjustments {
        focal: true,
        principal_point: true,
    }];
    recon
}

#[test]
fn a_zero_adjustment_borrows_the_camera() {
    let recon = SfmrReconstruction::demo(16);
    for i in 0..recon.image_count() {
        let intrinsics = recon.image_intrinsics(i);
        assert!(matches!(intrinsics, Cow::Borrowed(_)));
        assert!(std::ptr::eq(
            intrinsics.as_ref(),
            &recon.image_set.cameras[recon.image_set.images[i].camera_index as usize]
        ));
    }
}

#[test]
fn a_non_zero_adjustment_is_applied_exactly_on_a_two_focal_model() {
    // The demo camera: PINHOLE, 1920 x 1080, so the short side is 1080.
    let recon = adjusted_demo();
    let adjustment = tagged(3);
    let intrinsics = recon.image_intrinsics(3);
    assert!(matches!(intrinsics, Cow::Owned(_)));
    let CameraModel::Pinhole {
        focal_length_x,
        focal_length_y,
        principal_point_x,
        principal_point_y,
    } = intrinsics.model
    else {
        panic!("the model changed");
    };
    let scale = 1.0 + adjustment.focal;
    assert_eq!(focal_length_x, 1000.0 * scale);
    assert_eq!(focal_length_y, 1000.0 * scale);
    assert_eq!(
        principal_point_x,
        960.0 + adjustment.principal_point[0] * 1080.0
    );
    assert_eq!(
        principal_point_y,
        540.0 + adjustment.principal_point[1] * 1080.0
    );
    assert_eq!((intrinsics.width, intrinsics.height), (1920, 1080));
}

#[test]
fn a_non_zero_adjustment_is_applied_exactly_on_a_single_focal_model() {
    // Portrait, so the short side is the width.
    let camera = CameraIntrinsics {
        model: CameraModel::SimpleRadial {
            focal_length: 400.0,
            principal_point_x: 135.0,
            principal_point_y: 240.0,
            radial_distortion_k1: -0.05,
        },
        width: 270,
        height: 480,
    };
    let adjustment = IntrinsicAdjustment {
        focal: -0.0125,
        principal_point: [0.011, -0.014],
    };
    let adjusted = camera.with_adjustment(&adjustment);
    assert_eq!(
        adjusted.model,
        CameraModel::SimpleRadial {
            focal_length: 400.0 * (1.0 - 0.0125),
            principal_point_x: 135.0 + 0.011 * 270.0,
            principal_point_y: 240.0 - 0.014 * 270.0,
            radial_distortion_k1: -0.05,
        }
    );
    // Zero gives the camera back exactly.
    assert_eq!(
        camera.with_adjustment(&IntrinsicAdjustment::default()),
        camera
    );
}

#[test]
fn an_image_filter_and_reorder_carry_the_adjustment_with_its_image() {
    let recon = adjusted_demo();
    let subset = recon.subset_by_image_indices(&[5, 1, 6], false).unwrap();
    for (new, old) in [(0, 5), (1, 1), (2, 6)] {
        assert_eq!(
            subset.image_set.images[new].intrinsic_adjustment,
            tagged(old)
        );
        assert_eq!(*subset.image_intrinsics(new), *recon.image_intrinsics(old));
    }
    assert_eq!(
        subset.image_set.per_image_adjustments,
        recon.image_set.per_image_adjustments
    );

    // A similarity of the scene leaves every adjustment where it was.
    let moved = recon.apply_se3_transform(&crate::Se3Transform::identity());
    for i in 0..recon.image_count() {
        assert_eq!(moved.image_set.images[i].intrinsic_adjustment, tagged(i));
    }
}

#[test]
fn the_columns_and_the_flags_cross_the_sfmr_data_boundary() {
    let recon = adjusted_demo();
    let data = recon.to_sfmr_data();
    assert_eq!(
        data.cameras[0].per_image_adjustments,
        vec!["focal".to_string(), "principal_point".to_string()]
    );
    let focal = data.focal_adjustments.as_ref().unwrap();
    let principal_point = data.principal_point_adjustments_xy.as_ref().unwrap();
    for i in 0..recon.image_count() {
        assert_eq!(focal[i], tagged(i).focal);
        assert_eq!(
            [principal_point[[i, 0]], principal_point[[i, 1]]],
            tagged(i).principal_point
        );
    }
    let back = SfmrReconstruction::from_sfmr_data(data).unwrap();
    assert_eq!(
        back.image_set.per_image_adjustments,
        recon.image_set.per_image_adjustments
    );
    for i in 0..recon.image_count() {
        assert_eq!(back.image_set.images[i].intrinsic_adjustment, tagged(i));
    }

    // A value without adjustments writes neither column nor key, and reads
    // back with no flags.
    let plain = SfmrReconstruction::demo(16).to_sfmr_data();
    assert!(plain.focal_adjustments.is_none());
    assert!(plain.principal_point_adjustments_xy.is_none());
    assert!(plain.cameras[0].per_image_adjustments.is_empty());
    let back = SfmrReconstruction::from_sfmr_data(plain).unwrap();
    assert!(back.image_set.per_image_adjustments.is_empty());
}

#[test]
fn a_flag_without_a_value_writes_a_zero_column() {
    let mut recon = SfmrReconstruction::demo(16);
    recon.image_set.per_image_adjustments = vec![PerImageAdjustments {
        focal: false,
        principal_point: true,
    }];
    let data = recon.to_sfmr_data();
    assert!(data.focal_adjustments.is_none());
    assert_eq!(
        data.principal_point_adjustments_xy,
        Some(ndarray::Array2::zeros((recon.image_count(), 2)))
    );
    assert_eq!(
        data.cameras[0].per_image_adjustments,
        vec!["principal_point".to_string()]
    );
}

#[test]
fn a_save_keeps_every_adjustment_and_refuses_one_its_camera_does_not_vary() {
    let dir = std::env::temp_dir().join("sfmr_core_per_image_adjustments_save");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join(".sfm-workspace.json"), "{}").unwrap();

    // A non-zero adjustment comes back on load bit for bit, with its
    // camera's flags.
    let path = dir.join("adjusted.sfmr");
    let adjusted = adjusted_demo();
    adjusted.save(&path).unwrap();
    let loaded = SfmrReconstruction::load(&path, &Progress::none()).unwrap();
    assert_eq!(
        loaded.image_set.per_image_adjustments,
        adjusted.image_set.per_image_adjustments
    );
    for (a, b) in loaded
        .image_set
        .images
        .iter()
        .zip(&adjusted.image_set.images)
    {
        assert_eq!(
            a.intrinsic_adjustment.focal.to_bits(),
            b.intrinsic_adjustment.focal.to_bits()
        );
        assert_eq!(
            a.intrinsic_adjustment.principal_point.map(f64::to_bits),
            b.intrinsic_adjustment.principal_point.map(f64::to_bits)
        );
    }
    assert_eq!(
        loaded.content_xxh128().unwrap().content_xxh128,
        adjusted.content_xxh128().unwrap().content_xxh128
    );

    // An adjustment under a camera that does not vary it is refused as well,
    // rather than dropped by the save.
    let mut unflagged = adjusted_demo();
    unflagged.image_set.per_image_adjustments.clear();
    assert!(unflagged.save(&dir.join("unflagged.sfmr")).is_err());

    // The flags with every adjustment zero come back on load as well.
    let mut flagged = SfmrReconstruction::demo(16);
    flagged.image_set.per_image_adjustments = vec![PerImageAdjustments {
        focal: true,
        principal_point: false,
    }];
    let zero = dir.join("zero.sfmr");
    flagged.save(&zero).unwrap();
    let loaded = SfmrReconstruction::load(&zero, &Progress::none()).unwrap();
    assert_eq!(
        loaded.image_set.per_image_adjustments,
        flagged.image_set.per_image_adjustments
    );
    assert!(loaded
        .image_set
        .images
        .iter()
        .all(|im| im.intrinsic_adjustment.is_zero()));

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn the_flags_name_their_parameters_in_the_formats_order() {
    let both = PerImageAdjustments {
        focal: true,
        principal_point: true,
    };
    assert_eq!(both.names(), vec!["focal", "principal_point"]);
    assert_eq!(PerImageAdjustments::from_names(&both.names()), Ok(both));
    assert!(PerImageAdjustments::default().names().is_empty());
    assert!(!PerImageAdjustments::default().any());
    assert!(PerImageAdjustments::from_names(&["skew"]).is_err());
}

#[test]
fn the_intrinsics_list_holds_one_entry_per_adjusted_image() {
    // With no adjustment the list is the cameras themselves, borrowed, and
    // the camera column.
    let plain = SfmrReconstruction::demo(16);
    let list = plain.image_set.image_intrinsics_list();
    assert!(matches!(list.intrinsics, Cow::Borrowed(_)));
    assert_eq!(list.intrinsics.len(), plain.camera_count());
    for i in 0..plain.image_count() {
        assert_eq!(list.image_entry[i], plain.image_set.images[i].camera_index);
    }

    // With every image but one adjusted, each adjusted image has an entry of
    // its own after the cameras, and every image reads its own intrinsics.
    let mut recon = adjusted_demo();
    recon.image_set.images[2].intrinsic_adjustment = IntrinsicAdjustment::default();
    let list = recon.image_set.image_intrinsics_list();
    assert_eq!(
        list.intrinsics.len(),
        recon.camera_count() + recon.image_count() - 1
    );
    assert_eq!(list.image_entry[2], 0);
    let all = recon.image_set.all_image_intrinsics();
    assert_eq!(all.len(), recon.image_count());
    for (i, intrinsics) in all.iter().enumerate() {
        assert_eq!(list.of(i), recon.image_intrinsics(i).as_ref());
        assert_eq!(intrinsics.as_ref(), recon.image_intrinsics(i).as_ref());
    }
}
