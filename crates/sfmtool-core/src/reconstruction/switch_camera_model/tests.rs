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
    let (out, report) =
        switch_camera_model(&recon, &[0], &ten, &SwitchCameraModelOptions::default()).unwrap();
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
    for image in 0..recon.image_table.images.len() {
        let camera =
            &recon.image_table.cameras[recon.image_table.images[image].camera_index as usize];
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
                image_name: recon.image_table.images[image].name.clone(),
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
    assert_eq!(without.image_table.cameras, with.image_table.cameras);
    for (a, b) in without.point_set.points.iter().zip(&with.point_set.points) {
        assert_eq!(a.error, b.error);
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
