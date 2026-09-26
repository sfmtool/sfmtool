// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Switching cameras of a whole reconstruction: what changes, what does not,
//! the fixed-set comparison, and the refusals.
//!
//! The fixture is a synthetic scene through the `kerry_park` rig's first
//! fisheye lens: cameras on a shallow arc looking at a cloud of points, every
//! keypoint the exact projection, plus one point seen by the first image at
//! 95° off its axis, past the lens's trusted bound.

use nalgebra::{Point3, UnitQuaternion, Vector3};
use ndarray::Array2;

use crate::camera::{CameraIntrinsics, CameraModel};
use crate::reconstruction::data::{ObservationSource, Point3D, SfmrImage, TrackObservation};

use super::*;

const IMAGES: usize = 5;
const POINTS: usize = 30;
const WIDE_DEG: f64 = 95.0;

fn jitter(i: usize, salt: u64) -> f64 {
    let mut z = (i as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ salt;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z ^= z >> 27;
    ((z % 20001) as f64 / 10000.0) - 1.0
}

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

fn equidistant() -> CameraIntrinsics {
    CameraIntrinsics {
        model: CameraModel::EquidistantFisheye {
            focal_length: 131.0,
            principal_point_x: 240.0,
            principal_point_y: 240.0,
        },
        width: 480,
        height: 480,
    }
}

/// The scene through `cameras`, image `i` through camera `camera_of(i)`.
fn scene(cameras: Vec<CameraIntrinsics>, camera_of: impl Fn(usize) -> u32) -> SfmrReconstruction {
    let mut recon = SfmrReconstruction::demo(1);
    recon.image_table.cameras = cameras;
    recon.image_table.images = (0..IMAGES)
        .map(|i| {
            let angle = 0.25 * (i as f64 - (IMAGES as f64 - 1.0) / 2.0);
            let centre = Vector3::new(8.0 * angle.sin(), 0.5 * jitter(i, 11), 8.0 * angle.cos());
            let rotation = UnitQuaternion::face_towards(&centre, &Vector3::y()).inverse();
            SfmrImage {
                name: format!("image_{i:03}.jpg"),
                camera_index: camera_of(i),
                quaternion_wxyz: rotation,
                translation_xyz: -(rotation * centre),
            }
        })
        .collect();
    let stats = recon.image_table.depth_statistics.images[0].clone();
    recon.image_table.depth_statistics.images = vec![stats; IMAGES];
    recon.image_table.depth_histogram_counts =
        vec![recon.image_table.depth_histogram_counts[0].clone(); IMAGES];

    let mut positions: Vec<Point3<f64>> = (0..POINTS)
        .map(|p| Point3::new(2.0 * jitter(p, 1), 2.0 * jitter(p, 2), 1.5 * jitter(p, 3)))
        .collect();
    // One point 95° off the first image's axis, three units out.
    let first = &recon.image_table.images[0];
    let wide = WIDE_DEG.to_radians();
    let local = Vector3::new(wide.sin(), 0.0, -wide.cos()) * 3.0;
    let world = first.quaternion_wxyz.inverse() * (local - first.translation_xyz);
    positions.push(Point3::from(world));

    recon.point_set.points = positions
        .iter()
        .map(|&position| Point3D {
            position,
            w: 1.0,
            color: [10, 20, 30],
            error: 0.25,
            normal: Vector3::new(0.0, 0.0, -1.0),
        })
        .collect();

    let mut tracks = Vec::new();
    let mut counts = Vec::new();
    let mut keypoints: Vec<[f32; 2]> = Vec::new();
    for (p, position) in positions.iter().enumerate() {
        let mut count = 0u32;
        let images: Vec<usize> = if p == POINTS {
            vec![0]
        } else {
            (0..IMAGES).collect()
        };
        for i in images {
            let image = &recon.image_table.images[i];
            let camera = &recon.image_table.cameras[image.camera_index as usize];
            let local = image.quaternion_wxyz * position.coords + image.translation_xyz;
            let Some((u, v)) = camera.ray_to_pixel([local.x, local.y, local.z]) else {
                continue;
            };
            tracks.push(TrackObservation {
                image_index: i as u32,
                point_index: p as u32,
            });
            keypoints.push([u as f32, v as f32]);
            count += 1;
        }
        counts.push(count);
    }
    let mut keypoints_xy = Array2::<f32>::zeros((keypoints.len(), 2));
    for (row, uv) in keypoints.iter().enumerate() {
        keypoints_xy[[row, 0]] = uv[0];
        keypoints_xy[[row, 1]] = uv[1];
    }
    let set = &mut recon.point_set;
    set.tracks = tracks;
    set.observation_counts = counts;
    set.observations = ObservationSource::EmbeddedPatches {
        keypoints_xy,
        image_file_hashes: vec![[0u8; 16]; IMAGES],
    };
    recon.metadata.feature_source =
        sfmtool_sfmr_format::FEATURE_SOURCE_EMBEDDED_PATCHES.to_string();
    recon.rebuild_derived_fields();
    recon
}

fn target(name: &str) -> RefitTarget {
    RefitTarget::from_name(name, None).unwrap()
}

#[test]
fn poses_points_and_keypoints_are_unchanged() {
    let recon = scene(vec![kerry_cam0()], |_| 0);
    let (out, report) = switch_camera_model(
        &recon,
        &[0],
        &target("SFMTOOL_FISHEYE"),
        &RefitOptions::default(),
    )
    .unwrap();

    for (a, b) in out.image_table.images.iter().zip(&recon.image_table.images) {
        assert_eq!(a.quaternion_wxyz, b.quaternion_wxyz);
        assert_eq!(a.translation_xyz, b.translation_xyz);
        assert_eq!(a.camera_index, b.camera_index);
    }
    assert_eq!(out.point_set.tracks, recon.point_set.tracks);
    assert_eq!(out.point_set.keypoints_xy(), recon.point_set.keypoints_xy());
    for (a, b) in out.point_set.points.iter().zip(&recon.point_set.points) {
        assert_eq!(a.position, b.position);
        assert_eq!(a.w, b.w);
        // The planted 0.25 is replaced by the recomputed mean error.
        assert_ne!(a.error, 0.25);
    }
    assert_eq!(out.image_table.cameras[0].model_name(), "SFMTOOL_FISHEYE");
    assert_eq!(out.image_table.cameras[0], report.cameras[0].refit.camera);
    // The input is untouched.
    assert_eq!(recon.image_table.cameras[0], kerry_cam0());

    let entry = &report.cameras[0];
    assert_eq!(entry.camera, 0);
    assert_eq!(entry.images, IMAGES);
    assert_eq!(entry.source, kerry_cam0());
    let obs = &entry.observations;
    assert_eq!(obs.observations + obs.unmeasured, recon.observation_count());
    assert_eq!(obs.unmeasured, 0);
    assert!(obs.before.median_px < 1e-3, "{:?}", obs.before);
    assert!(obs.after.median_px < 0.5, "{:?}", obs.after);
    assert!((obs.max_theta_deg - WIDE_DEG).abs() < 1e-6);
    // The one observation past the source's trusted bound is the wide one.
    assert!(obs.trusted_deg.unwrap() < WIDE_DEG);
    assert_eq!(obs.past_trusted, 1);
    assert!(obs.past_trusted_before.max_px < 1e-3);

    // The outermost observation is the wide one, measured under the switched
    // camera; the fixture sits beside no .sift file, so nothing is detected.
    let outermost = &entry.outermost;
    assert_eq!((outermost.camera, outermost.images), (0, IMAGES));
    let observed = outermost.observed.expect("inline keypoints");
    assert_eq!(observed.image, 0);
    // Past the source's trusted bound the two models disagree by a degree or
    // two about the angle of the same pixel.
    assert!((observed.theta_deg - WIDE_DEG).abs() < 3.0, "{observed:?}");
    assert_eq!((outermost.detected, outermost.detected_images), (None, 0));
}

#[test]
fn only_the_named_cameras_change() {
    let recon = scene(vec![kerry_cam0(), kerry_cam0()], |i| (i % 2) as u32);
    let (out, report) = switch_camera_model(
        &recon,
        &[1],
        &target("SFMTOOL_FISHEYE"),
        &RefitOptions::default(),
    )
    .unwrap();
    assert_eq!(out.image_table.cameras[0], kerry_cam0());
    assert_eq!(out.image_table.cameras[1].model_name(), "SFMTOOL_FISHEYE");
    assert_eq!(report.cameras.len(), 1);
    assert_eq!(report.cameras[0].camera, 1);
    assert_eq!(report.cameras[0].images, IMAGES / 2);
}

#[test]
fn without_a_trusted_bound_the_fit_reaches_the_observations() {
    let recon = scene(vec![equidistant()], |_| 0);
    let (_, report) = switch_camera_model(
        &recon,
        &[0],
        &target("SFMTOOL_FISHEYE"),
        &RefitOptions::default(),
    )
    .unwrap();
    let refit = &report.cameras[0].refit;
    assert_eq!(refit.theta_fit_source, ThetaFitSource::Observations);
    assert!((refit.theta_fit_deg - WIDE_DEG).abs() < 1e-6);
    // The equidistant lens is exact under the spline.
    assert!(report.cameras[0].observations.after.max_px < 1e-3);
}

#[test]
fn a_perspective_target_for_a_camera_observed_past_90_is_refused() {
    let recon = scene(vec![equidistant()], |_| 0);
    let options = RefitOptions {
        theta_fit_deg: Some(60.0),
        spline_domain_deg: None,
    };
    let err = switch_camera_model(&recon, &[0], &target("SFMTOOL_PINHOLE"), &options)
        .err()
        .expect("refused");
    match err {
        SwitchCameraModelError::Refit {
            camera: 0,
            error: RefitError::ObservationsPast90 { max_theta_deg },
        } => assert!((max_theta_deg - WIDE_DEG).abs() < 1e-6),
        other => panic!("unexpected {other}"),
    }
}

#[test]
fn a_refusal_names_the_camera() {
    let recon = scene(vec![kerry_cam0()], |_| 0);
    let options = RefitOptions {
        theta_fit_deg: Some(100.0),
        spline_domain_deg: None,
    };
    let err = switch_camera_model(&recon, &[0], &target("SFMTOOL_FISHEYE"), &options)
        .err()
        .expect("refused");
    assert!(matches!(
        err,
        SwitchCameraModelError::Refit {
            camera: 0,
            error: RefitError::BeyondTrustedBound { .. }
        }
    ));
    assert!(err.to_string().starts_with("camera 0: "), "{err}");
}

#[test]
fn camera_lists_are_checked() {
    let recon = scene(vec![kerry_cam0()], |_| 0);
    let t = target("SFMTOOL_FISHEYE");
    let options = RefitOptions::default();
    assert_eq!(
        switch_camera_model(&recon, &[], &t, &options)
            .err()
            .expect("refused"),
        SwitchCameraModelError::NoCameras
    );
    assert_eq!(
        switch_camera_model(&recon, &[3], &t, &options)
            .err()
            .expect("refused"),
        SwitchCameraModelError::UnknownCamera {
            camera: 3,
            count: 1
        }
    );
}

/// The `kerry_park` first lens switched to an eight-coefficient
/// `SFMTOOL_FISHEYE`, the camera a spline refit starts from.
fn kerry_spline() -> CameraIntrinsics {
    crate::camera::refit_intrinsics::refit_camera_intrinsics(
        &kerry_cam0(),
        &RefitTarget::from_name("SFMTOOL_FISHEYE", Some(8)).unwrap(),
        &RefitOptions::default(),
    )
    .unwrap()
    .camera
}

#[test]
fn a_spline_switched_to_its_own_model_is_refitted_over_its_whole_domain() {
    let source = kerry_spline();
    let recon = scene(vec![source.clone(), kerry_cam0()], |i| (i % 2) as u32);
    let ten = RefitTarget::from_name("SFMTOOL_FISHEYE", Some(10)).unwrap();
    let options = RefitOptions {
        theta_fit_deg: None,
        spline_domain_deg: Some(108.0),
    };

    let (out, report) = switch_camera_model(&recon, &[0], &ten, &options).unwrap();

    // Exactly the lens-only refit of the spline, which samples the whole new
    // domain rather than the observations' extent.
    let expected = refit_spline(&source, 10, Some(108.0)).unwrap();
    let entry = &report.cameras[0];
    assert_eq!(entry.refit, expected);
    assert_eq!(entry.refit.theta_fit_source, ThetaFitSource::SplineDomain);
    assert!((entry.refit.theta_fit_deg - 108.0).abs() < 1e-9);
    assert_eq!(out.image_table.cameras[0], expected.camera);
    let Some((coeffs, _, _)) = out.image_table.cameras[0].model.radial_spline() else {
        panic!("still a spline");
    };
    assert_eq!(coeffs.len(), 10);
    assert!(
        (crate::camera::refit_intrinsics::spline_domain_deg(&out.image_table.cameras[0]).unwrap()
            - 108.0)
            .abs()
            < 1e-9
    );
    // The other camera is not touched, and the comparison still covers the
    // switched camera's observations.
    assert_eq!(out.image_table.cameras[1], kerry_cam0());
    assert!(entry.observations.observations > 0);

    // With no domain given the domain end is kept bit for bit.
    let (out, report) = switch_camera_model(&recon, &[0], &ten, &RefitOptions::default()).unwrap();
    assert_eq!(
        report.cameras[0].refit,
        refit_spline(&source, 10, None).unwrap()
    );
    let (Some((_, before, _)), Some((_, after, _))) = (
        source.model.radial_spline(),
        out.image_table.cameras[0].model.radial_spline(),
    ) else {
        panic!("spline models");
    };
    assert_eq!(before.to_bits(), after.to_bits());

    // A fit angle given is a fit like any other source's, over that angle.
    let given = RefitOptions {
        theta_fit_deg: Some(80.0),
        spline_domain_deg: None,
    };
    let (_, report) = switch_camera_model(&recon, &[0], &ten, &given).unwrap();
    assert_eq!(
        report.cameras[0].refit.theta_fit_source,
        ThetaFitSource::Given
    );

    // The other spline model is a switch between models, not a refit.
    let pinhole = RefitTarget::from_name("SFMTOOL_PINHOLE", Some(8)).unwrap();
    let scene_narrow = scene(vec![source.clone()], |_| 0);
    let error = switch_camera_model(&scene_narrow, &[0], &pinhole, &RefitOptions::default())
        .err()
        .expect("observed past 90°");
    assert!(matches!(
        error,
        SwitchCameraModelError::Refit {
            camera: 0,
            error: RefitError::ObservationsPast90 { .. }
        }
    ));
}

#[test]
fn a_spline_refit_the_model_cannot_have_is_refused_naming_the_camera() {
    let recon = scene(vec![kerry_cam0(), kerry_spline()], |i| (i % 2) as u32);
    let options = RefitOptions {
        theta_fit_deg: None,
        spline_domain_deg: Some(200.0),
    };
    let error = switch_camera_model(
        &recon,
        &[1],
        &RefitTarget::from_name("SFMTOOL_FISHEYE", Some(8)).unwrap(),
        &options,
    )
    .err();
    assert_eq!(
        error,
        Some(SwitchCameraModelError::Refit {
            camera: 1,
            error: RefitError::SplineDomainInvalid {
                spline_domain_deg: 200.0
            }
        })
    );
    assert!(error.unwrap().to_string().starts_with("camera 1:"));
}
