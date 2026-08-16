// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Alignment round-trip tests.
//!
//! The fixture is the one the spec asks for: reconstruction B is reconstruction
//! A put through a **known** similarity, so a correct fit is the one that maps
//! B's coordinates back onto A's — and the assertion can be stated on points
//! rather than on the transform's components, which sidesteps the quaternion
//! sign ambiguity entirely.

use nalgebra::{Point3, Vector3};

use crate::reconstruction::ObservationSource;
use crate::{RotQuaternion, Se3Transform, SfmrReconstruction};

use super::*;

/// A similarity with all three parts non-trivial: a rotation about a slanted
/// axis, a translation, and a scale change. Anything that only happened to work
/// for one of the three fails here.
fn known_similarity() -> Se3Transform {
    Se3Transform::new(
        RotQuaternion::from_axis_angle(Vector3::new(0.3, -0.7, 0.6), 0.9).unwrap(),
        Vector3::new(4.0, -2.5, 1.25),
        2.0,
    )
}

/// `recon` with every point and every camera pose put through `t` — the same
/// reconstruction seen from a different frame.
fn transformed(recon: &SfmrReconstruction, t: &Se3Transform) -> SfmrReconstruction {
    let mut out = recon.clone();
    for p in &mut out.points {
        p.position = t.apply_to_point(&p.position);
    }
    for image in &mut out.images {
        let (rotation, translation) = t.apply_to_camera_pose(
            &RotQuaternion::from_nalgebra(image.quaternion_wxyz),
            &image.translation_xyz,
        );
        image.quaternion_wxyz = *rotation.as_nalgebra();
        image.translation_xyz = translation;
    }
    out
}

/// Rename every image so the two reconstructions share nothing.
fn renamed(recon: &SfmrReconstruction, prefix: &str) -> SfmrReconstruction {
    let mut out = recon.clone();
    for (i, image) in out.images.iter_mut().enumerate() {
        image.name = format!("{prefix}_{i:03}.jpg");
    }
    out
}

/// Swap in embedded keypoints, which carry no feature index — the case the
/// point mode cannot match on.
fn embedded(recon: &SfmrReconstruction) -> SfmrReconstruction {
    let mut out = recon.clone();
    out.observations = ObservationSource::EmbeddedPatches {
        keypoints_xy: ndarray::Array2::<f32>::from_elem((out.tracks.len(), 2), 100.0),
        image_file_hashes: vec![[0u8; 16]; out.images.len()],
    };
    out
}

fn options(source: AlignSource, estimate_scale: bool) -> AlignOptions {
    AlignOptions {
        source,
        estimate_scale,
    }
}

/// Largest distance between `transform(source point)` and the target point it
/// should have landed on.
fn worst_point_error(
    source: &SfmrReconstruction,
    target: &SfmrReconstruction,
    transform: &Se3Transform,
) -> f64 {
    source
        .points
        .iter()
        .zip(target.points.iter())
        .map(|(s, t)| (transform.apply_to_point(&s.position) - t.position).norm())
        .fold(0.0, f64::max)
}

// ── Round trip ──────────────────────────────────────────────────────────

#[test]
fn aligning_by_cameras_recovers_a_known_similarity() {
    let a = SfmrReconstruction::demo(64);
    let t = known_similarity();
    let b = transformed(&a, &t);

    let fit = align_reconstructions(&b, &a, options(AlignSource::Cameras, true)).expect("a fit");

    // The fit maps B back onto A, so it is `t` inverted — asserted where it
    // matters, on where B's points land.
    assert!(
        worst_point_error(&b, &a, &fit.transform) < 1e-9,
        "worst point error {} after a camera-mode round trip",
        worst_point_error(&b, &a, &fit.transform),
    );
    assert!(
        (fit.transform.scale - 0.5).abs() < 1e-9,
        "scale {} should invert the fixture's 2.0",
        fit.transform.scale,
    );
    assert_eq!(fit.correspondences, a.images.len());
    assert!(fit.rms < 1e-9, "RMS {} over an exact fixture", fit.rms);
}

/// A camera fit is a fit of *centres*, so the cameras it lands have to end up
/// facing the way the target's do — the geometry a viewer draws as an image
/// plane in front of each camera rather than behind it.
#[test]
fn aligning_by_cameras_lands_the_cameras_facing_the_same_way() {
    let a = SfmrReconstruction::demo(64);
    let b = transformed(&a, &known_similarity());

    let fit = align_reconstructions(&b, &a, options(AlignSource::Cameras, true)).expect("a fit");

    let rotation = fit.transform.rotation.to_rotation_matrix();
    for (source, target) in b.images.iter().zip(a.images.iter()) {
        // Viewing direction: the camera-to-world rotation's third axis, put
        // through the fit's rotation (a direction ignores translation, and the
        // positive scale leaves it pointing where it did).
        let forward = |image: &crate::SfmrImage| {
            image
                .quaternion_wxyz
                .inverse()
                .transform_vector(&Vector3::new(0.0, 0.0, 1.0))
        };
        let landed = rotation * forward(source);
        assert!(
            landed.dot(&forward(target)) > 0.999,
            "camera {} lands facing {landed:?}, not the target's {:?}",
            source.name,
            forward(target),
        );
    }
}

#[test]
fn aligning_by_points_recovers_a_known_similarity() {
    let a = SfmrReconstruction::demo(64);
    let t = known_similarity();
    let b = transformed(&a, &t);

    let fit = align_reconstructions(&b, &a, options(AlignSource::Points, true)).expect("a fit");

    assert!(
        worst_point_error(&b, &a, &fit.transform) < 1e-9,
        "worst point error {} after a point-mode round trip",
        worst_point_error(&b, &a, &fit.transform),
    );
    // Every point is matched through its own feature index in both shared
    // images, and every one of them survives RANSAC on an exact fixture.
    assert_eq!(fit.correspondences, 64);
    assert_eq!(fit.inliers, 64);
    assert!(fit.rms < 1e-9);
}

#[test]
fn a_rigid_fit_leaves_the_scale_at_one() {
    let a = SfmrReconstruction::demo(64);
    // Rotation and translation only, so a rigid fit can still recover it
    // exactly — the point is that the *scale* is not fitted.
    let t = Se3Transform::new(
        RotQuaternion::from_axis_angle(Vector3::new(0.0, 0.0, 1.0), 0.4).unwrap(),
        Vector3::new(1.0, 2.0, -3.0),
        1.0,
    );
    let b = transformed(&a, &t);

    let fit = align_reconstructions(&b, &a, options(AlignSource::Cameras, false)).expect("a fit");

    assert_eq!(fit.transform.scale, 1.0);
    assert!(worst_point_error(&b, &a, &fit.transform) < 1e-9);
}

#[test]
fn a_rigid_fit_of_a_scaled_pair_does_not_absorb_the_scale() {
    let a = SfmrReconstruction::demo(64);
    let b = transformed(&a, &known_similarity());

    let fit = align_reconstructions(&b, &a, options(AlignSource::Cameras, false)).expect("a fit");

    // Refusing to fit scale is the whole point of the Rigid option: the result
    // is deliberately a poor fit of a scaled pair rather than a silent
    // similarity.
    assert_eq!(fit.transform.scale, 1.0);
    assert!(
        fit.rms > 1e-3,
        "a rigid fit of a 2x-scaled pair should not be tight (RMS {})",
        fit.rms,
    );
}

#[test]
fn the_infinity_points_are_left_out_of_a_point_fit() {
    let mut a = SfmrReconstruction::demo(64);
    // A bearing, not a location: kept in the cloud, dropped from the fit.
    a.points[0].w = 0.0;
    a.points[0].position = Point3::new(0.0, 0.0, 1.0);
    let b = transformed(&a, &known_similarity());

    let fit = align_reconstructions(&b, &a, options(AlignSource::Points, true)).expect("a fit");

    assert_eq!(
        fit.correspondences, 63,
        "the point at infinity was fed to the fit"
    );
}

// ── Failures leave everything alone ─────────────────────────────────────

#[test]
fn two_reconstructions_with_no_shared_images_cannot_be_aligned() {
    let a = SfmrReconstruction::demo(64);
    let b = renamed(&transformed(&a, &known_similarity()), "other");

    for source in [AlignSource::Cameras, AlignSource::Points] {
        let err =
            align_reconstructions(&b, &a, options(source, true)).expect_err("no correspondences");
        assert!(
            err.contains("share no image names"),
            "unhelpful reason for {source:?}: {err}"
        );
    }
}

#[test]
fn too_few_shared_cameras_is_reported_rather_than_fitted() {
    let mut a = SfmrReconstruction::demo(64);
    let b = transformed(&a, &known_similarity());
    // Leave two images sharing a name — one short of what pins a rotation down.
    for (i, image) in a.images.iter_mut().enumerate().skip(2) {
        image.name = format!("only_in_a_{i:03}.jpg");
    }

    let err =
        align_reconstructions(&b, &a, options(AlignSource::Cameras, true)).expect_err("too few");
    assert!(err.contains("only 2 shared image(s)"), "{err}");
}

#[test]
fn the_point_mode_refuses_a_reconstruction_without_feature_indexes() {
    let a = SfmrReconstruction::demo(64);
    let b = embedded(&transformed(&a, &known_similarity()));

    let err =
        align_reconstructions(&b, &a, options(AlignSource::Points, true)).expect_err("no indexes");
    assert!(err.contains("feature indexes"), "{err}");

    // The camera mode is unaffected: it never looks at observations.
    assert!(align_reconstructions(&b, &a, options(AlignSource::Cameras, true)).is_ok());
}

#[test]
fn a_pair_sharing_images_but_no_3d_points_is_reported() {
    let a = SfmrReconstruction::demo(64);
    let mut b = transformed(&a, &known_similarity());
    // Same images, but every observation now names a feature index that the
    // other side never used, so nothing joins.
    if let ObservationSource::SiftFiles {
        feature_indexes, ..
    } = &mut b.observations
    {
        for f in feature_indexes.iter_mut() {
            *f += 10_000;
        }
    }

    let err =
        align_reconstructions(&b, &a, options(AlignSource::Points, true)).expect_err("no matches");
    assert!(err.contains("point correspondences"), "{err}");
}

// ── Shared images ───────────────────────────────────────────────────────

#[test]
fn a_repeated_image_name_pairs_once_on_either_side() {
    let a = SfmrReconstruction::demo(64);
    let mut b = a.clone();
    b.images[1].name = b.images[0].name.clone();

    // First occurrence wins on both sides, so the duplicate contributes one
    // pair, not two — and never a pair with the *second* index.
    let pairs = shared_images(&b, &a);
    assert_eq!(pairs.iter().filter(|&&(s, _)| s == 0).count(), 1);
    assert!(!pairs.iter().any(|&(s, _)| s == 1));
    assert_eq!(pairs.len(), a.images.len() - 1);
}
