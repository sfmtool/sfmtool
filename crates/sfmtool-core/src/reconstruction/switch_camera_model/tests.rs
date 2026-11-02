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
        rolling_shutter: None,
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
        rolling_shutter: None,
    }
}

/// The scene through `cameras`, image `i` through camera `camera_of(i)`.
fn scene(cameras: Vec<CameraIntrinsics>, camera_of: impl Fn(usize) -> u32) -> SfmrReconstruction {
    let mut recon = SfmrReconstruction::demo(1);
    recon.image_set.cameras = cameras;
    recon.image_set.images = (0..IMAGES)
        .map(|i| {
            let angle = 0.25 * (i as f64 - (IMAGES as f64 - 1.0) / 2.0);
            let centre = Vector3::new(8.0 * angle.sin(), 0.5 * jitter(i, 11), 8.0 * angle.cos());
            let rotation = UnitQuaternion::face_towards(&centre, &Vector3::y()).inverse();
            SfmrImage {
                name: format!("image_{i:03}.jpg"),
                camera_index: camera_of(i),
                quaternion_wxyz: rotation,
                translation_xyz: -(rotation * centre),
                intrinsic_adjustment: Default::default(),
            }
        })
        .collect();
    let stats = recon.image_set.depth_statistics.images[0].clone();
    recon.image_set.depth_statistics.images = vec![stats; IMAGES];
    recon.image_set.depth_histogram_counts =
        vec![recon.image_set.depth_histogram_counts[0].clone(); IMAGES];

    let mut positions: Vec<Point3<f64>> = (0..POINTS)
        .map(|p| Point3::new(2.0 * jitter(p, 1), 2.0 * jitter(p, 2), 1.5 * jitter(p, 3)))
        .collect();
    // One point 95° off the first image's axis, three units out.
    let first = &recon.image_set.images[0];
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
            rms_reprojection_error: 0.25,
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
            let image = &recon.image_set.images[i];
            let camera = &recon.image_set.cameras[image.camera_index as usize];
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

/// The switch's options with the fit's set to `refit`.
fn fit(refit: RefitOptions) -> SwitchCameraModelOptions {
    SwitchCameraModelOptions {
        refit,
        ..SwitchCameraModelOptions::default()
    }
}

#[test]
fn poses_points_and_keypoints_are_unchanged() {
    let recon = scene(vec![kerry_cam0()], |_| 0);
    let (out, report) = switch_camera_model(
        &recon,
        &[0],
        &target("SFMTOOL_FISHEYE"),
        &SwitchCameraModelOptions::default(),
    )
    .unwrap();

    for (a, b) in out.image_set.images.iter().zip(&recon.image_set.images) {
        assert_eq!(a.quaternion_wxyz, b.quaternion_wxyz);
        assert_eq!(a.translation_xyz, b.translation_xyz);
        assert_eq!(a.camera_index, b.camera_index);
    }
    assert_eq!(out.point_set.tracks, recon.point_set.tracks);
    assert_eq!(out.point_set.keypoints_xy(), recon.point_set.keypoints_xy());
    for (a, b) in out.point_set.points.iter().zip(&recon.point_set.points) {
        assert_eq!(a.position, b.position);
        assert_eq!(a.w, b.w);
        // The planted 0.25 is replaced by the recomputed RMS error.
        assert_ne!(a.rms_reprojection_error, 0.25);
    }
    // Every point here is seen by the switched camera, so each stored error is
    // the root mean square `recompute_point_errors` writes.
    let mut rescored = out.clone();
    rescored.recompute_point_errors().unwrap();
    for (a, b) in out.point_set.points.iter().zip(&rescored.point_set.points) {
        assert!(
            (a.rms_reprojection_error - b.rms_reprojection_error).abs() < 1e-5,
            "{} against {}",
            a.rms_reprojection_error,
            b.rms_reprojection_error
        );
    }
    assert_eq!(out.image_set.cameras[0].model_name(), "SFMTOOL_FISHEYE");
    assert_eq!(out.image_set.cameras[0], report.cameras[0].refit.camera);
    // The input is untouched.
    assert_eq!(recon.image_set.cameras[0], kerry_cam0());

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
    let outermost = entry.outermost.as_ref().expect("asked for by default");
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
        &SwitchCameraModelOptions::default(),
    )
    .unwrap();
    assert_eq!(out.image_set.cameras[0], kerry_cam0());
    assert_eq!(out.image_set.cameras[1].model_name(), "SFMTOOL_FISHEYE");
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
        &SwitchCameraModelOptions::default(),
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
    let err = switch_camera_model(&recon, &[0], &target("SFMTOOL_PINHOLE"), &fit(options))
        .err()
        .expect("refused");
    // The refusal names the point whose ray is wide, the one the fixture put
    // 95° off the first image's axis, and says the angle is the ray's rather
    // than a keypoint's.
    let sentence = err.to_string();
    assert!(
        sentence.contains(&format!(
            "the ray to point {POINTS} is 95.00° off the camera's axis"
        )),
        "{sentence}"
    );
    match err {
        SwitchCameraModelError::Refit {
            camera: 0,
            error:
                RefitError::ObservationsPast90 {
                    max_theta_deg,
                    point,
                },
        } => {
            assert!((max_theta_deg - WIDE_DEG).abs() < 1e-6);
            assert_eq!(point, POINTS as u32);
        }
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
    let err = switch_camera_model(&recon, &[0], &target("SFMTOOL_FISHEYE"), &fit(options))
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
    let options = SwitchCameraModelOptions::default();
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

    let (out, report) = switch_camera_model(&recon, &[0], &ten, &fit(options)).unwrap();

    // Exactly the lens-only refit of the spline, which samples the whole new
    // domain rather than the observations' extent.
    let expected = refit_spline(&source, 10, Some(108.0)).unwrap();
    let entry = &report.cameras[0];
    assert_eq!(entry.refit, expected);
    assert_eq!(entry.refit.theta_fit_source, ThetaFitSource::SplineDomain);
    assert!((entry.refit.theta_fit_deg - 108.0).abs() < 1e-9);
    assert_eq!(out.image_set.cameras[0], expected.camera);
    let Some((coeffs, _, _)) = out.image_set.cameras[0].model.radial_spline() else {
        panic!("still a spline");
    };
    assert_eq!(coeffs.len(), 10);
    assert!(
        (crate::camera::refit_intrinsics::spline_domain_deg(&out.image_set.cameras[0]).unwrap()
            - 108.0)
            .abs()
            < 1e-9
    );
    // The other camera is not touched, and the comparison still covers the
    // switched camera's observations.
    assert_eq!(out.image_set.cameras[1], kerry_cam0());
    assert!(entry.observations.observations > 0);

    // With no domain given the domain end is kept bit for bit.
    let (out, report) =
        switch_camera_model(&recon, &[0], &ten, &SwitchCameraModelOptions::default()).unwrap();
    assert_eq!(
        report.cameras[0].refit,
        refit_spline(&source, 10, None).unwrap()
    );
    let (Some((_, before, _)), Some((_, after, _))) = (
        source.model.radial_spline(),
        out.image_set.cameras[0].model.radial_spline(),
    ) else {
        panic!("spline models");
    };
    assert_eq!(before.to_bits(), after.to_bits());

    // A fit angle given is a fit like any other source's, over that angle.
    let given = RefitOptions {
        theta_fit_deg: Some(80.0),
        spline_domain_deg: None,
    };
    let (_, report) = switch_camera_model(&recon, &[0], &ten, &fit(given)).unwrap();
    assert_eq!(
        report.cameras[0].refit.theta_fit_source,
        ThetaFitSource::Given
    );

    // The other spline model is a switch between models, not a refit.
    let pinhole = RefitTarget::from_name("SFMTOOL_PINHOLE", Some(8)).unwrap();
    let scene_narrow = scene(vec![source.clone()], |_| 0);
    let error = switch_camera_model(
        &scene_narrow,
        &[0],
        &pinhole,
        &SwitchCameraModelOptions::default(),
    )
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
        &fit(options),
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

/// Write a `.sift` file for every image of `recon`, under a workspace in
/// `dir`, holding one feature at the principal point.
fn write_sift_files(recon: &mut SfmrReconstruction, dir: &tempfile::TempDir) {
    recon.workspace_dir = dir.path().to_path_buf();
    recon.metadata.workspace.contents.feature_prefix_dir = "features".into();
    for image in 0..recon.image_set.images.len() {
        let camera = &recon.image_set.cameras[recon.image_set.images[image].camera_index as usize];
        let (cx, cy) = camera.principal_point();
        let positions = Array2::from_shape_vec((1, 2), vec![cx as f32, cy as f32]).unwrap();
        let data = sfmtool_sift_format::SiftData {
            feature_tool_metadata: sfmtool_sift_format::FeatureToolMetadata {
                feature_tool: "test".into(),
                feature_type: "sift".into(),
                feature_options: serde_json::json!({}),
            },
            metadata: sfmtool_sift_format::SiftMetadata {
                version: sfmtool_sift_format::SIFT_FORMAT_VERSION,
                image_name: recon.image_set.images[image].name.clone(),
                image_file_xxh128: "0".repeat(32),
                image_file_size: 1,
                image_width: camera.width,
                image_height: camera.height,
                feature_count: 1,
            },
            content_hash: sfmtool_sift_format::SiftContentHash::default(),
            positions_xy: positions,
            affine_shapes: ndarray::Array3::<f32>::zeros((1, 2, 2)),
            descriptors: Array2::<u8>::zeros((1, 128)),
            thumbnail_y_x_rgb: ndarray::Array3::<u8>::zeros((128, 128, 3)),
        };
        let path = recon.sift_path_for_image(image);
        std::fs::create_dir_all(path.parent().expect("a feature directory")).unwrap();
        sfmtool_sift_format::write_sift(&path, &data, 3).unwrap();
    }
}

#[test]
fn the_outermost_keypoint_can_be_left_out_of_the_report() {
    let dir = tempfile::tempdir().unwrap();
    let mut recon = scene(vec![kerry_cam0()], |_| 0);
    write_sift_files(&mut recon, &dir);
    let t = target("SFMTOOL_FISHEYE");

    // Asked for, the keypoint is detected from every image's `.sift` file.
    let (with, with_report) =
        switch_camera_model(&recon, &[0], &t, &SwitchCameraModelOptions::default()).unwrap();
    let outermost = with_report.cameras[0]
        .outermost
        .as_ref()
        .expect("asked for");
    assert_eq!(outermost.detected_images, IMAGES);
    assert!(outermost.detected.is_some());

    // Left out, the entry carries none, and everything else is the same.
    let options = SwitchCameraModelOptions {
        outermost_keypoint: false,
        ..SwitchCameraModelOptions::default()
    };
    let (without, without_report) = switch_camera_model(&recon, &[0], &t, &options).unwrap();
    let entry = &without_report.cameras[0];
    assert_eq!(entry.outermost, None);
    let asked = &with_report.cameras[0];
    assert_eq!(
        (entry.camera, &entry.source, &entry.refit, entry.images),
        (asked.camera, &asked.source, &asked.refit, asked.images)
    );
    assert_eq!(entry.observations, asked.observations);
    assert_eq!(without.image_set.cameras, with.image_set.cameras);
    for (a, b) in without.point_set.points.iter().zip(&with.point_set.points) {
        assert_eq!(a.rms_reprojection_error, b.rms_reprojection_error);
    }

    // And it reads no `.sift` file for it: with every file gone, the switch
    // that leaves it out answers exactly as before.
    std::fs::remove_dir_all(dir.path().join("features")).unwrap();
    let (_, again) = switch_camera_model(&recon, &[0], &t, &options).unwrap();
    assert_eq!(again, without_report);
}

/// The fit alone is the camera the switch installs, for a change of model, a
/// refit of a spline and a source without a trusted bound, and it refuses what
/// the switch refuses.
#[test]
fn the_fit_alone_is_the_camera_the_switch_installs() {
    let cases: Vec<(SfmrReconstruction, usize, RefitTarget, RefitOptions)> = vec![
        (
            scene(vec![kerry_cam0(), kerry_cam0()], |i| (i % 2) as u32),
            1,
            target("SFMTOOL_FISHEYE"),
            RefitOptions::default(),
        ),
        (
            scene(vec![equidistant()], |_| 0),
            0,
            target("SFMTOOL_FISHEYE"),
            RefitOptions::default(),
        ),
        (
            scene(vec![kerry_spline()], |_| 0),
            0,
            RefitTarget::from_name("SFMTOOL_FISHEYE", Some(10)).unwrap(),
            RefitOptions {
                theta_fit_deg: None,
                spline_domain_deg: Some(120.0),
            },
        ),
    ];
    for (recon, camera, target, refit) in cases {
        let alone = fit_camera_switch(&recon, camera, &target, &refit).expect("a fit");
        let (_, report) = switch_camera_model(&recon, &[camera], &target, &fit(refit)).unwrap();
        assert_eq!(alone, report.cameras[0].refit);
    }

    let recon = scene(vec![equidistant()], |_| 0);
    let options = RefitOptions {
        theta_fit_deg: Some(60.0),
        spline_domain_deg: None,
    };
    let pinhole = target("SFMTOOL_PINHOLE");
    assert_eq!(
        fit_camera_switch(&recon, 0, &pinhole, &options),
        Err(
            switch_camera_model(&recon, &[0], &pinhole, &fit(options.clone()))
                .err()
                .expect("refused")
        ),
    );
    assert_eq!(
        fit_camera_switch(&recon, 2, &pinhole, &options),
        Err(SwitchCameraModelError::UnknownCamera {
            camera: 2,
            count: 1
        })
    );
}

/// The default fit angle is the one a fit given none uses, and is reported for
/// a target whose fit at that angle is refused.
#[test]
fn the_default_fit_angle_is_the_fits_and_survives_a_refusal() {
    let recon = scene(vec![equidistant()], |_| 0);
    let (deg, source) = default_fit_angle(&recon, 0).expect("a default");
    assert_eq!(source, ThetaFitSource::Observations);
    let refit = fit_camera_switch(
        &recon,
        0,
        &target("SFMTOOL_FISHEYE"),
        &RefitOptions::default(),
    )
    .expect("a fit");
    assert_eq!((refit.theta_fit_deg, refit.theta_fit_source), (deg, source));
    // A perspective target is refused for the point 95° off the axis; the
    // default is still there.
    assert!(fit_camera_switch(
        &recon,
        0,
        &target("SFMTOOL_PINHOLE"),
        &RefitOptions::default()
    )
    .is_err());
    assert!((deg - WIDE_DEG).abs() < 1e-6);

    let kerry = scene(vec![kerry_cam0()], |_| 0);
    let (_, source) = default_fit_angle(&kerry, 0).expect("a default");
    assert_eq!(source, ThetaFitSource::TrustedBound);
    assert_eq!(
        default_fit_angle(&kerry, 3),
        Err(SwitchCameraModelError::UnknownCamera {
            camera: 3,
            count: 1
        })
    );
}

#[test]
fn every_image_keeps_its_adjustment_through_a_switch() {
    // The switch refits the shared camera and leaves each image's intrinsic
    // adjustment as it was, since the adjustment is relative to whatever the
    // camera is. The errors it reports, and the ones it stores, are read
    // through each image's intrinsics: before the switch they are the exact
    // projections the keypoints were made at.
    use crate::reconstruction::data::adjusted_fixture::{adjust_and_reproject, varied_adjustment};
    let mut recon = scene(vec![kerry_cam0()], |_| 0);
    adjust_and_reproject(&mut recon, varied_adjustment);
    let (out, report) = switch_camera_model(
        &recon,
        &[0],
        &target("SFMTOOL_FISHEYE"),
        &SwitchCameraModelOptions::default(),
    )
    .unwrap();
    for (a, b) in out.image_set.images.iter().zip(&recon.image_set.images) {
        assert_eq!(a.intrinsic_adjustment, b.intrinsic_adjustment);
    }
    let obs = &report.cameras[0].observations;
    assert!(obs.before.median_px < 1e-3, "{:?}", obs.before);
    assert!(obs.after.median_px < 0.5, "{:?}", obs.after);
    let mut rescored = out.clone();
    rescored.recompute_point_errors().unwrap();
    for (a, b) in out.point_set.points.iter().zip(&rescored.point_set.points) {
        assert!(
            (a.rms_reprojection_error - b.rms_reprojection_error).abs() < 1e-5,
            "{} against {}",
            a.rms_reprojection_error,
            b.rms_reprojection_error
        );
    }
}

#[test]
fn a_camera_whose_images_carry_lens_corrections_is_refused() {
    // The refusal is decided on the image set alone, before any fit, so the
    // camera's model does not matter here.
    let corrected = |adjustment: crate::camera::IntrinsicAdjustment| {
        let mut recon = scene(vec![kerry_cam0()], |_| 0);
        recon.image_set.images[1].intrinsic_adjustment = adjustment;
        recon
    };
    for adjustment in [
        crate::camera::IntrinsicAdjustment {
            radial_bspline: [0.001, 0.0],
            ..Default::default()
        },
        crate::camera::IntrinsicAdjustment {
            rolling_shutter: [0.0, 1e-3, 0.0],
            ..Default::default()
        },
    ] {
        let recon = corrected(adjustment);
        let err = switch_camera_model(
            &recon,
            &[0],
            &target("SFMTOOL_FISHEYE"),
            &SwitchCameraModelOptions::default(),
        )
        .err()
        .expect("refused");
        assert!(
            matches!(
                err,
                SwitchCameraModelError::LensCorrections { camera: 0, .. }
            ),
            "{err}"
        );
        assert!(err.to_string().contains("does not carry over"), "{err}");
    }
    // A camera that lists either correction is refused even at zero.
    let mut recon = scene(vec![kerry_cam0()], |_| 0);
    recon.image_set.per_image_adjustments = vec![crate::reconstruction::PerImageAdjustments {
        rolling_shutter: true,
        ..Default::default()
    }];
    assert!(matches!(
        fit_camera_switch(
            &recon,
            0,
            &target("SFMTOOL_FISHEYE"),
            &RefitOptions::default()
        ),
        Err(SwitchCameraModelError::LensCorrections { camera: 0, .. })
    ));
}

use crate::camera::{IntrinsicAdjustment, RollingShutterScanlines};
use crate::reconstruction::PerImageAdjustments;
use ImageAdjustmentParameter::{Focal, PrincipalPoint, RadialBspline, RollingShutter};

/// The scene with every image carrying its own focal scale and
/// principal-point offset, flagged on the camera, and its keypoints made
/// through them.
fn varied_scene(camera: CameraIntrinsics) -> SfmrReconstruction {
    use crate::reconstruction::data::adjusted_fixture::{adjust_and_reproject, varied_adjustment};
    let mut recon = scene(vec![camera], |_| 0);
    recon.image_set.per_image_adjustments = vec![PerImageAdjustments {
        focal: true,
        principal_point: true,
        ..Default::default()
    }];
    adjust_and_reproject(&mut recon, varied_adjustment);
    recon
}

/// A four-coefficient `SFMTOOL_PINHOLE` on a 60° domain whose images vary all
/// four per-image parameters, read on rows, with their keypoints made through
/// them. The fixture's 95° point has no pixel under it, so it has no
/// observation.
fn pinhole_corrected_scene() -> SfmrReconstruction {
    use crate::reconstruction::data::adjusted_fixture::adjust_and_reproject;
    let camera = CameraIntrinsics {
        model: CameraModel::SfmtoolPinhole {
            focal_length: 300.0,
            principal_point_x: 240.0,
            principal_point_y: 240.0,
            bspline_rho_max: 60f64.to_radians().tan(),
            bspline: vec![0.0, 0.002, -0.004, 0.003],
        },
        width: 480,
        height: 480,
        rolling_shutter: None,
    };
    let mut recon = scene(vec![camera], |_| 0);
    recon.image_set.per_image_adjustments = vec![PerImageAdjustments {
        focal: true,
        principal_point: true,
        radial_bspline: true,
        rolling_shutter: true,
        rolling_shutter_scanlines: Some(RollingShutterScanlines::Rows),
    }];
    adjust_and_reproject(&mut recon, |i| {
        let t = i as f64;
        IntrinsicAdjustment {
            focal: 0.01 * (0.9 * t + 0.3).sin(),
            principal_point: [0.004 * (1.3 * t).cos(), 0.004 * (0.7 * t).sin()],
            radial_bspline: [0.001 * (t - 2.0), -0.0005 * (t - 2.0)],
            rolling_shutter: [1e-3 * (t - 2.0), 0.0, 5e-4],
        }
    });
    recon.recompute_point_errors().unwrap();
    recon
}

fn drops(parameters: &[ImageAdjustmentParameter]) -> SwitchCameraModelOptions {
    SwitchCameraModelOptions {
        drop_image_adjustments: parameters.to_vec(),
        ..SwitchCameraModelOptions::default()
    }
}

#[test]
fn each_dropped_parameter_is_dropped_before_the_fit() {
    let recon = varied_scene(kerry_cam0());
    let fisheye = target("SFMTOOL_FISHEYE");
    for parameter in [Focal, PrincipalPoint] {
        let (out, report) =
            switch_camera_model(&recon, &[0], &fisheye, &drops(&[parameter])).unwrap();
        let entry = &report.cameras[0];
        assert_eq!(entry.dropped_image_adjustments, vec![parameter]);
        assert_eq!(entry.source, recon.image_set.cameras[0]);
        // The camera installed is the fit of the camera the drop leaves.
        let folded = drop_image_adjustments(&recon, 0, &[parameter]).unwrap();
        let expected = fit_camera_switch(&folded, 0, &fisheye, &RefitOptions::default()).unwrap();
        assert_eq!(out.image_set.cameras[0], expected.camera);
        assert_eq!(entry.refit, expected);
        let flags = out.image_set.camera_per_image_adjustments(0);
        assert_eq!(
            (flags.focal, flags.principal_point),
            (parameter != Focal, parameter != PrincipalPoint)
        );
        for (a, b) in out.image_set.images.iter().zip(&recon.image_set.images) {
            let (a, b) = (&a.intrinsic_adjustment, &b.intrinsic_adjustment);
            if parameter == Focal {
                assert_eq!((a.focal, a.principal_point), (0.0, b.principal_point));
            } else {
                assert_eq!((a.focal, a.principal_point), (b.focal, [0.0; 2]));
            }
        }
        // `before` is the value as it was handed in, the exact projections.
        let o = &entry.observations;
        assert!(o.before.median_px < 1e-3, "{:?}", o.before);
        assert!(o.after.median_px > o.before.median_px, "{o:?}");
    }
}

#[test]
fn nothing_dropped_leaves_the_switch_as_it_was() {
    let recon = varied_scene(kerry_cam0());
    let fisheye = target("SFMTOOL_FISHEYE");
    let plain = switch_camera_model(&recon, &[0], &fisheye, &drops(&[])).unwrap();
    assert!(plain.1.cameras[0].dropped_image_adjustments.is_empty());
    assert_eq!(
        plain.1.cameras[0].refit,
        fit_camera_switch(&recon, 0, &fisheye, &RefitOptions::default()).unwrap()
    );
    // A parameter the camera does not vary is passed over.
    let passed_over = switch_camera_model(
        &recon,
        &[0],
        &fisheye,
        &drops(&[RadialBspline, RollingShutter]),
    )
    .unwrap();
    assert!(passed_over.1.cameras[0]
        .dropped_image_adjustments
        .is_empty());
    assert_eq!(passed_over.1, plain.1);
    for (a, b) in passed_over
        .0
        .image_set
        .images
        .iter()
        .zip(&plain.0.image_set.images)
    {
        assert_eq!(
            a.intrinsic_adjustment, b.intrinsic_adjustment,
            "every adjustment kept"
        );
    }
    assert_eq!(passed_over.0.image_set.cameras, plain.0.image_set.cameras);
}

#[test]
fn what_a_switch_carries_depends_on_the_target_and_the_domain() {
    let recon = pinhole_corrected_scene();
    let carried = |name: &str, options: &RefitOptions| -> Vec<(ImageAdjustmentParameter, bool)> {
        image_adjustments_carried(&recon, 0, &target(name), options)
            .unwrap()
            .into_iter()
            .map(|c| (c.parameter, c.carried))
            .collect()
    };
    let default = RefitOptions::default();
    assert_eq!(
        carried("SFMTOOL_FISHEYE", &default),
        vec![
            (Focal, true),
            (PrincipalPoint, true),
            (RadialBspline, false),
            (RollingShutter, false)
        ]
    );
    // A refit of the spline that keeps its domain end carries all four.
    assert!(carried("SFMTOOL_PINHOLE", &default).iter().all(|&(_, c)| c));
    // A new domain end, or a fit over a given angle, moves the domain the
    // radial correction is written on; the rates still ride on the model.
    for options in [
        RefitOptions {
            spline_domain_deg: Some(50.0),
            ..default.clone()
        },
        RefitOptions {
            theta_fit_deg: Some(40.0),
            ..default.clone()
        },
    ] {
        assert_eq!(
            carried("SFMTOOL_PINHOLE", &options),
            vec![
                (Focal, true),
                (PrincipalPoint, true),
                (RadialBspline, false),
                (RollingShutter, true)
            ],
            "{options:?}"
        );
    }
    // A camera varying nothing lists nothing, and a stale index is refused.
    let bare = scene(vec![kerry_cam0()], |_| 0);
    assert!(
        image_adjustments_carried(&bare, 0, &target("SFMTOOL_FISHEYE"), &default)
            .unwrap()
            .is_empty()
    );
    assert!(matches!(
        image_adjustments_carried(&bare, 3, &target("SFMTOOL_FISHEYE"), &default),
        Err(SwitchCameraModelError::UnknownCamera {
            camera: 3,
            count: 1
        })
    ));
}

#[test]
fn lens_corrections_kept_on_a_target_that_cannot_carry_them_are_refused() {
    let recon = pinhole_corrected_scene();
    let fisheye = target("SFMTOOL_FISHEYE");
    let err = switch_camera_model(&recon, &[0], &fisheye, &drops(&[Focal]))
        .err()
        .expect("refused");
    assert_eq!(
        err,
        SwitchCameraModelError::LensCorrections {
            camera: 0,
            target: "SFMTOOL_FISHEYE",
            kept: vec![RadialBspline, RollingShutter],
        }
    );
    let sentence = err.to_string();
    assert!(
        sentence.contains("radial correction and rolling-shutter rates")
            && sentence.contains("SFMTOOL_FISHEYE")
            && sentence.contains("drop them per image"),
        "{sentence}"
    );
    // Dropping one still refuses for the other.
    let err = switch_camera_model(&recon, &[0], &fisheye, &drops(&[RadialBspline]))
        .err()
        .expect("refused");
    assert!(
        matches!(&err, SwitchCameraModelError::LensCorrections { kept, .. } if *kept == vec![RollingShutter]),
        "{err}"
    );
    // A refit over a new domain end keeps the rates and refuses the radial
    // correction.
    let pinhole = target("SFMTOOL_PINHOLE");
    let moved = SwitchCameraModelOptions {
        refit: RefitOptions {
            spline_domain_deg: Some(50.0),
            ..RefitOptions::default()
        },
        ..SwitchCameraModelOptions::default()
    };
    assert!(matches!(
        switch_camera_model(&recon, &[0], &pinhole, &moved),
        Err(SwitchCameraModelError::LensCorrections { kept, .. }) if kept == vec![RadialBspline]
    ));
}

#[test]
fn dropping_the_lens_corrections_lifts_the_refusal() {
    let recon = pinhole_corrected_scene();
    let fisheye = target("SFMTOOL_FISHEYE");
    let (out, report) = switch_camera_model(
        &recon,
        &[0],
        &fisheye,
        &drops(&[RollingShutter, RadialBspline]),
    )
    .unwrap();
    let entry = &report.cameras[0];
    assert_eq!(
        entry.dropped_image_adjustments,
        vec![RadialBspline, RollingShutter]
    );
    assert_eq!(out.image_set.cameras[0].model_name(), "SFMTOOL_FISHEYE");
    let flags = out.image_set.camera_per_image_adjustments(0);
    assert!(flags.focal && flags.principal_point);
    assert!(!flags.radial_bspline && !flags.rolling_shutter);
    assert_eq!(
        flags.rolling_shutter_scanlines,
        Some(RollingShutterScanlines::Rows),
        "the scanlines setting is kept"
    );
    for (a, b) in out.image_set.images.iter().zip(&recon.image_set.images) {
        let (a, b) = (&a.intrinsic_adjustment, &b.intrinsic_adjustment);
        assert_eq!((a.focal, a.principal_point), (b.focal, b.principal_point));
        assert_eq!((a.radial_bspline, a.rolling_shutter), ([0.0; 2], [0.0; 3]));
    }
    out.image_set.check_image_intrinsics().unwrap();
    let o = &entry.observations;
    assert_eq!(o.unmeasured, 0);
    assert!(o.before.median_px < 1e-3, "{:?}", o.before);
    // What the drops cost is in the comparison: the keypoints were made
    // through the rates and the radial correction.
    assert!(o.after.median_px > 1e-3, "{:?}", o.after);
}

#[test]
fn a_refit_that_only_drops_a_parameter_runs() {
    // An `SFMTOOL_PINHOLE` refit as itself carries its corrections and drops
    // what it is asked to.
    let recon = pinhole_corrected_scene();
    let pinhole = target("SFMTOOL_PINHOLE");
    let (out, report) =
        switch_camera_model(&recon, &[0], &pinhole, &drops(&[PrincipalPoint])).unwrap();
    let entry = &report.cameras[0];
    assert_eq!(entry.dropped_image_adjustments, vec![PrincipalPoint]);
    assert_eq!(
        entry.refit.theta_fit_source,
        crate::camera::refit_intrinsics::ThetaFitSource::SplineDomain
    );
    let flags = out.image_set.camera_per_image_adjustments(0);
    assert!(flags.focal && !flags.principal_point && flags.radial_bspline && flags.rolling_shutter);
    for (a, b) in out.image_set.images.iter().zip(&recon.image_set.images) {
        let (a, b) = (&a.intrinsic_adjustment, &b.intrinsic_adjustment);
        assert_eq!(a.principal_point, [0.0; 2]);
        assert_eq!(
            (a.focal, a.radial_bspline, a.rolling_shutter),
            (b.focal, b.radial_bspline, b.rolling_shutter)
        );
    }
    // The shared principal point moved to the images' mean, and the spline
    // kept its domain end.
    let folded = drop_image_adjustments(&recon, 0, &[PrincipalPoint]).unwrap();
    let (cx, cy) = out.image_set.cameras[0].principal_point();
    let (fx, fy) = folded.image_set.cameras[0].principal_point();
    assert!((cx - fx).abs() < 1e-6 && (cy - fy).abs() < 1e-6);
    assert_eq!(
        crate::camera::refit_intrinsics::spline_domain_deg(&out.image_set.cameras[0]),
        crate::camera::refit_intrinsics::spline_domain_deg(&recon.image_set.cameras[0])
    );
    // A spline fisheye refit as itself drops its principal point the same way.
    let recon = varied_scene(kerry_spline());
    let (out, report) = switch_camera_model(
        &recon,
        &[0],
        &target("SFMTOOL_FISHEYE"),
        &drops(&[PrincipalPoint]),
    )
    .unwrap();
    assert_eq!(
        report.cameras[0].dropped_image_adjustments,
        vec![PrincipalPoint]
    );
    assert_eq!(out.image_set.cameras[0].model_name(), "SFMTOOL_FISHEYE");
    assert!(
        !out.image_set
            .camera_per_image_adjustments(0)
            .principal_point
    );
}
