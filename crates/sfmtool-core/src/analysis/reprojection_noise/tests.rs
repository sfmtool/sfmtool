// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The measured reprojection noise: a known noise level recovered from both
//! observation sources, per camera, with points at infinity left out.

use nalgebra::Point3;
use ndarray::{Array2, Array3};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use rand_distr::StandardNormal;
use tempfile::TempDir;

use crate::camera::{CameraIntrinsics, CameraModel};
use crate::{ObservationSource, SfmrReconstruction};

/// The demo reconstruction with a second camera (the odd images use it) and an
/// inline keypoint per observation: its point's projection plus Gaussian noise
/// of `sigma_px[camera]` per axis.
///
/// The result is `sift_files` with the inline column, which every reader takes
/// in preference to the `.sift` files; [`as_embedded`] and [`as_sift_only`]
/// state the same pixels in the other two ways.
pub(crate) fn noisy_demo(num_points: usize, sigma_px: [f64; 2], seed: u64) -> SfmrReconstruction {
    let mut recon = SfmrReconstruction::demo(num_points);
    let second = recon.image_table.cameras[0].clone();
    recon.image_table.cameras.push(second);
    for (i, image) in recon.image_table.images.iter_mut().enumerate() {
        image.camera_index = (i % 2) as u32;
    }
    observe(&mut recon, sigma_px, seed);
    recon
}

/// Replace `recon`'s inline keypoints with each observation's projection of
/// its point as stored (a direction through the rotation alone for a point at
/// infinity) plus Gaussian noise of `sigma_px[camera]` per axis. `recon` must
/// be `sift_files`, and every point in front of the cameras observing it.
pub(crate) fn observe(recon: &mut SfmrReconstruction, sigma_px: [f64; 2], seed: u64) {
    let mut rng = StdRng::seed_from_u64(seed);
    let tracks = &recon.point_set.tracks;
    let mut keypoints = Array2::<f32>::zeros((tracks.len(), 2));
    for (row, obs) in tracks.iter().enumerate() {
        let image = &recon.image_table.images[obs.image_index as usize];
        let camera = &recon.image_table.cameras[image.camera_index as usize];
        let point = &recon.point_set.points[obs.point_index as usize];
        let p_cam = if point.is_at_infinity() {
            image.quaternion_wxyz * point.position.coords
        } else {
            image.quaternion_wxyz * point.position.coords + image.translation_xyz
        };
        let (u, v) = camera
            .ray_to_pixel([p_cam.x, p_cam.y, p_cam.z])
            .expect("the point is in front of its cameras");
        let s = sigma_px[(image.camera_index as usize).min(1)];
        let du: f64 = rng.sample(StandardNormal);
        let dv: f64 = rng.sample(StandardNormal);
        keypoints[[row, 0]] = (u + s * du) as f32;
        keypoints[[row, 1]] = (v + s * dv) as f32;
    }
    if let ObservationSource::SiftFiles { keypoints_xy, .. } = &mut recon.point_set.observations {
        *keypoints_xy = Some(keypoints);
    }
}

/// `recon` as an `embedded_patches` reconstruction with the same pixels.
pub(crate) fn as_embedded(recon: &SfmrReconstruction) -> SfmrReconstruction {
    let mut out = recon.clone();
    let keypoints = recon.keypoints_xy().expect("inline pixels").clone();
    out.point_set.observations = ObservationSource::EmbeddedPatches {
        keypoints_xy: keypoints,
        image_file_hashes: vec![[7u8; 16]; recon.image_count()],
    };
    out.rebuild_derived_fields();
    out
}

/// `recon` with no inline column and its pixels in a `.sift` file per image
/// under `dir`, at the feature indexes its observations name.
pub(crate) fn as_sift_only(recon: &SfmrReconstruction, dir: &TempDir) -> SfmrReconstruction {
    let mut out = recon.clone();
    out.workspace_dir = dir.path().to_path_buf();
    out.metadata.workspace.contents.feature_prefix_dir = "features".into();
    let keypoints = recon.keypoints_xy().expect("inline pixels").clone();
    let features = recon.feature_indexes().expect("sift_files").to_vec();
    for image in 0..out.image_count() {
        let count = out.point_set.max_track_feature_index[image] as usize + 1;
        let mut positions = Array2::<f32>::zeros((count, 2));
        for (row, obs) in out.point_set.tracks.iter().enumerate() {
            if obs.image_index as usize == image {
                positions[[features[row] as usize, 0]] = keypoints[[row, 0]];
                positions[[features[row] as usize, 1]] = keypoints[[row, 1]];
            }
        }
        let data = sfmtool_sift_format::SiftData {
            feature_tool_metadata: sfmtool_sift_format::FeatureToolMetadata {
                feature_tool: "test".into(),
                feature_type: "sift".into(),
                feature_options: serde_json::json!({}),
            },
            metadata: sfmtool_sift_format::SiftMetadata {
                version: sfmtool_sift_format::SIFT_FORMAT_VERSION,
                image_name: out.image_table.images[image].name.clone(),
                image_file_xxh128: "0".repeat(32),
                image_file_size: 1,
                image_width: out.image_table.cameras[0].width,
                image_height: out.image_table.cameras[0].height,
                feature_count: count as u32,
            },
            content_hash: sfmtool_sift_format::SiftContentHash::default(),
            positions_xy: positions,
            affine_shapes: Array3::<f32>::zeros((count, 2, 2)),
            descriptors: Array2::<u8>::zeros((count, 128)),
            thumbnail_y_x_rgb: Array3::<u8>::zeros((128, 128, 3)),
        };
        let path = out.sift_path_for_image(image);
        std::fs::create_dir_all(path.parent().expect("a feature directory")).unwrap();
        sfmtool_sift_format::write_sift(&path, &data, 3).unwrap();
    }
    if let ObservationSource::SiftFiles { keypoints_xy, .. } = &mut out.point_set.observations {
        *keypoints_xy = None;
    }
    out
}

#[test]
fn recovers_a_known_noise_level() {
    let recon = noisy_demo(3000, [0.4, 0.4], 1);
    let noise = recon.reprojection_noise().unwrap();
    assert_eq!(noise.observation_count, 6000);
    let sigma = noise.sigma_px.unwrap();
    // 12,000 residual components: the RMS is within about 1% of 0.4.
    assert!((sigma - 0.4).abs() < 0.015, "sigma {sigma}");
    assert_eq!(recon.reprojection_noise_px().unwrap(), Some(sigma));
}

#[test]
fn measures_each_camera_separately() {
    let recon = noisy_demo(3000, [0.3, 0.9], 2);
    let noise = recon.reprojection_noise().unwrap();
    let [n0, n1] = [
        noise.per_camera_observation_count[0],
        noise.per_camera_observation_count[1],
    ];
    assert_eq!(n0 + n1, noise.observation_count);
    let s0 = noise.per_camera_sigma_px[0].unwrap();
    let s1 = noise.per_camera_sigma_px[1].unwrap();
    assert!((s0 - 0.3).abs() < 0.015, "camera 0: {s0}");
    assert!((s1 - 0.9).abs() < 0.04, "camera 1: {s1}");
    // The overall value pools the squared residuals of both.
    let pooled = ((n0 as f64 * s0 * s0 + n1 as f64 * s1 * s1) / (n0 + n1) as f64).sqrt();
    assert!((noise.sigma_px.unwrap() - pooled).abs() < 1e-12);
}

#[test]
fn every_observation_source_gives_the_same_value() {
    let recon = noisy_demo(200, [0.5, 0.5], 3);
    let inline = recon.reprojection_noise().unwrap();
    let embedded_recon = as_embedded(&recon);
    assert_eq!(embedded_recon.feature_source(), "embedded_patches");
    let embedded = embedded_recon.reprojection_noise().unwrap();
    let dir = tempfile::tempdir().unwrap();
    let sift_only = as_sift_only(&recon, &dir);
    assert!(sift_only.keypoints_xy().is_none());
    let from_sift = sift_only.reprojection_noise().unwrap();
    assert_eq!(inline, embedded);
    assert_eq!(inline, from_sift);
}

#[test]
fn a_sift_only_reconstruction_without_its_files_is_an_error() {
    let recon = noisy_demo(20, [0.5, 0.5], 4);
    let dir = tempfile::tempdir().unwrap();
    let mut sift_only = as_sift_only(&recon, &dir);
    sift_only.workspace_dir = dir.path().join("elsewhere");
    assert!(sift_only.reprojection_noise().is_err());
}

#[test]
fn points_at_infinity_are_left_out() {
    let recon = noisy_demo(400, [0.5, 0.5], 5);
    let before = recon.reprojection_noise().unwrap();

    // Demote every third point to a bearing whose observations are far off:
    // the measure then counts only the remaining finite points' observations,
    // with their residuals unchanged.
    let mut demoted = recon.clone();
    let mut kept_sq = 0.0;
    let mut kept = 0usize;
    for (p, point) in demoted.point_set.points.iter_mut().enumerate() {
        if p % 3 == 0 {
            point.position = Point3::new(0.0, 0.0, 1.0);
            point.w = 0.0;
        }
    }
    demoted.rebuild_derived_fields();
    let keypoints = recon.keypoints_xy().unwrap();
    for (row, obs) in recon.point_set.tracks.iter().enumerate() {
        if obs.point_index % 3 == 0 {
            continue;
        }
        let image = &recon.image_table.images[obs.image_index as usize];
        let camera = &recon.image_table.cameras[image.camera_index as usize];
        let p = &recon.point_set.points[obs.point_index as usize].position;
        let p_cam = image.quaternion_wxyz * p.coords + image.translation_xyz;
        let (u, v) = camera.ray_to_pixel([p_cam.x, p_cam.y, p_cam.z]).unwrap();
        let du = u - keypoints[[row, 0]] as f64;
        let dv = v - keypoints[[row, 1]] as f64;
        kept_sq += du * du + dv * dv;
        kept += 1;
    }
    let after = demoted.reprojection_noise().unwrap();
    assert_eq!(after.observation_count, kept);
    assert!(after.observation_count < before.observation_count);
    let expected = (kept_sq / (2 * kept) as f64).sqrt();
    assert!((after.sigma_px.unwrap() - expected).abs() < 1e-12);
}

#[test]
fn no_finite_point_gives_none() {
    let mut recon = noisy_demo(10, [0.5, 0.5], 6);
    for point in &mut recon.point_set.points {
        point.position = Point3::new(1.0, 0.0, 0.0);
        point.w = 0.0;
    }
    recon.rebuild_derived_fields();
    let noise = recon.reprojection_noise().unwrap();
    assert_eq!(noise.sigma_px, None);
    assert_eq!(noise.observation_count, 0);
    assert_eq!(noise.per_camera_sigma_px, vec![None, None]);
    assert_eq!(noise.per_camera_observation_count, vec![0, 0]);

    let empty = SfmrReconstruction::demo(0);
    assert_eq!(empty.reprojection_noise_px().unwrap(), None);
}

#[test]
fn a_non_finite_pixel_is_left_out() {
    let mut recon = noisy_demo(50, [0.5, 0.5], 7);
    let before = recon.reprojection_noise().unwrap();
    if let ObservationSource::SiftFiles {
        keypoints_xy: Some(k),
        ..
    } = &mut recon.point_set.observations
    {
        k[[0, 0]] = f32::NAN;
    }
    let after = recon.reprojection_noise().unwrap();
    assert_eq!(after.observation_count, before.observation_count - 1);
}

/// Set the inline keypoint of observation `row`.
fn set_keypoint(recon: &mut SfmrReconstruction, row: usize, xy: [f32; 2]) {
    if let ObservationSource::SiftFiles {
        keypoints_xy: Some(k),
        ..
    } = &mut recon.point_set.observations
    {
        k[[row, 0]] = xy[0];
        k[[row, 1]] = xy[1];
    }
}

#[test]
fn a_gross_outlier_is_left_out_of_the_rms() {
    let clean = noisy_demo(1000, [0.5, 0.5], 8);
    let before = clean.reprojection_noise().unwrap();
    assert_eq!(before.outlier_count, 0);
    assert_eq!(before.observation_count, 2000);

    // One mismatched keypoint 40 px off its projection (80 robust spreads):
    // the plain RMS would rise to about 0.97 px. It is counted as an outlier
    // and the rest give the noise level.
    let mut planted = clean.clone();
    let k = clean.keypoints_xy().unwrap();
    set_keypoint(&mut planted, 7, [k[[7, 0]] + 40.0, k[[7, 1]]]);
    let after = planted.reprojection_noise().unwrap();
    assert_eq!(after.outlier_count, 1);
    assert_eq!(after.observation_count, 1999);
    let sigma = after.sigma_px.unwrap();
    assert!((sigma - 0.5).abs() < 0.03, "sigma {sigma}");
    assert!((sigma - before.sigma_px.unwrap()).abs() < 0.01);

    // The case that made σ 7e7 px: two keypoints set to 1e9.
    let mut huge = clean.clone();
    set_keypoint(&mut huge, 3, [1.0e9, 1.0e9]);
    set_keypoint(&mut huge, 1500, [1.0e9, 1.0e9]);
    let huge_noise = huge.reprojection_noise().unwrap();
    assert_eq!(huge_noise.outlier_count, 2);
    assert_eq!(huge_noise.observation_count, 1998);
    assert!((huge_noise.sigma_px.unwrap() - 0.5).abs() < 0.03);
}

#[test]
fn the_gate_reads_each_cameras_own_spread() {
    // Camera 1 is three times as noisy as camera 0. Gated at a spread pooled
    // over both, camera 1's own tail would be cut; gated per camera, neither
    // loses an observation and both are recovered.
    let recon = noisy_demo(3000, [0.3, 0.9], 9);
    let noise = recon.reprojection_noise().unwrap();
    assert_eq!(noise.outlier_count, 0);
    let s1 = noise.per_camera_sigma_px[1].unwrap();
    assert!((s1 - 0.9).abs() < 0.04, "camera 1: {s1}");
}

#[test]
fn an_observation_the_ray_construction_declines_is_left_out() {
    // A fisheye whose distortion folds at θ_d = 0.770 (radius 385 px): a pixel
    // past it un-projects to a ray that projects elsewhere, so observed_ray
    // declines it and the point-or-bearing test gives it no ray. The measure
    // leaves it out too, rather than counting it as an outlier.
    let mut recon = SfmrReconstruction::demo(200);
    recon.image_table.cameras[0] = CameraIntrinsics {
        model: CameraModel::OpenCVFisheye {
            focal_length_x: 500.0,
            focal_length_y: 500.0,
            principal_point_x: 960.0,
            principal_point_y: 540.0,
            radial_distortion_k1: -0.25,
            radial_distortion_k2: 0.0,
            radial_distortion_k3: 0.0,
            radial_distortion_k4: 0.0,
        },
        width: 1920,
        height: 1080,
    };
    observe(&mut recon, [0.5, 0.5], 10);
    let before = recon.reprojection_noise().unwrap();
    assert_eq!(before.observation_count, 400);

    set_keypoint(&mut recon, 11, [960.0 + 500.0 * 0.85, 540.0]);
    let after = recon.reprojection_noise().unwrap();
    assert_eq!(after.observation_count, 399);
    assert_eq!(after.outlier_count, 0);
}

#[test]
fn round_off_residuals_are_not_outliers() {
    // Keypoints exact up to their f32 storage: every residual is round-off,
    // spread evenly over a few hundred-thousandths of a pixel, and the gate
    // leaves every one in.
    let mut recon = SfmrReconstruction::demo(50);
    observe(&mut recon, [0.0, 0.0], 11);
    let noise = recon.reprojection_noise().unwrap();
    assert_eq!(noise.outlier_count, 0);
    assert_eq!(noise.observation_count, 100);
    assert!(noise.sigma_px.unwrap() < 1e-3);
}
