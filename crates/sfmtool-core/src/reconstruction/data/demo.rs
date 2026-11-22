// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Synthetic reconstruction fixture.
//!
//! [`SfmrReconstruction::demo`] builds an in-memory reconstruction with no
//! files behind it — points on a relaxed unit sphere observed by a ring of
//! cameras. Used by the SfM Explorer's "Load Demo Data" and by tests that need
//! a reconstruction without a workspace on disk. Split out of [`super`]: it is
//! a data *generator*, not part of the data model.

use std::path::PathBuf;

use nalgebra::{Point3, UnitQuaternion, Vector3};

use sfmtool_sfmr_format::{
    ContentHash, DepthStatistics, ImageDepthStats, ObservedDepthStats, SfmrMetadata,
    FEATURE_SOURCE_SIFT_FILES,
};

use crate::camera::CameraIntrinsics;
use crate::features::sift::{ImageReadings, ImageToGray};

use super::{
    compute_observation_offsets, count_points_at_infinity, ImageSet, ObservationSource, Point3D,
    PointSet, SfmrImage, SfmrReconstruction, TrackObservation,
};

impl SfmrReconstruction {
    /// Creates a demo reconstruction with synthetic data for testing.
    ///
    /// Generates `num_points` 3D points evenly distributed on a unit sphere
    /// (offset to sit in front of the camera arc), observed by 8 cameras
    /// arranged in a circle around the origin. Every image carries invented
    /// image readings under the default grey conversion, and every
    /// observation its pixel, the point's projection, so the demo saves as a
    /// `.sfmr` with no photographs or `.sift` files behind it.
    pub fn demo(num_points: usize) -> Self {
        use crate::camera::CameraModel;
        use crate::spherical::sphere_points::{evenly_distributed_sphere_points, RelaxConfig};
        use std::collections::{BTreeMap, HashMap};

        let num_images = 8;
        let num_buckets: u32 = 128;

        // Camera
        let cameras = vec![CameraIntrinsics {
            model: CameraModel::Pinhole {
                focal_length_x: 1000.0,
                focal_length_y: 1000.0,
                principal_point_x: 960.0,
                principal_point_y: 540.0,
            },
            width: 1920,
            height: 1080,
            rolling_shutter: None,
        }];

        // Images: cameras in an arc
        let mut images = Vec::with_capacity(num_images);
        for i in 0..num_images {
            let angle = (i as f64) * std::f64::consts::PI / 4.0;
            let radius = 5.0;
            let position = Point3::new(radius * angle.cos(), radius * angle.sin(), 1.5);

            // Look at origin: forward = -position.normalize()
            let forward = (Point3::origin() - position).normalize();
            let world_up = Vector3::z();
            let right = forward.cross(&world_up).normalize();
            let up = right.cross(&forward).normalize();

            // Build rotation matrix (world-to-camera, canonical convention:
            // camera looks down −Z with +Y up). Rows are the camera axes
            // expressed in world coordinates: X=right, Y=up, Z=−forward.
            let r = nalgebra::Matrix3::new(
                right.x, right.y, right.z, up.x, up.y, up.z, -forward.x, -forward.y, -forward.z,
            );
            let rotation = UnitQuaternion::from_rotation_matrix(
                &nalgebra::Rotation3::from_matrix_unchecked(r),
            );
            let translation = rotation * (-position.coords);

            images.push(SfmrImage {
                name: format!("image_{:03}.jpg", i),
                camera_index: 0,
                quaternion_wxyz: rotation,
                translation_xyz: translation,
                intrinsic_adjustment: Default::default(),
                // Invented like the poses, so the demo saves as the format
                // requires: a softer frame every other image, with the
                // horizontal gradients stronger than the vertical.
                readings: Some(ImageReadings {
                    sharpness: if i % 2 == 0 { 0.02 } else { 0.01 },
                    gradient_structure_tensor_xx_yy_xy: [0.004, 0.003, 0.0005 * i as f32],
                }),
            });
        }

        // Points: evenly distributed on the unit sphere via Thomson relaxation,
        // then offset by +1 in z so they sit in front of the camera arc. The
        // sampling is seeded so the demo is the same reconstruction on every
        // call: tests built on it reason about particular points.
        let sphere = evenly_distributed_sphere_points(
            num_points,
            &RelaxConfig {
                seed: Some(0x0d3a_0c1d),
                ..RelaxConfig::default()
            },
        );
        let mut points = Vec::with_capacity(num_points);
        for i in 0..num_points {
            let x = sphere[3 * i] as f64;
            let y = sphere[3 * i + 1] as f64;
            let z = sphere[3 * i + 2] as f64;

            let r = ((x + 1.0) * 127.5) as u8;
            let g = ((y + 1.0) * 127.5) as u8;
            let b = ((z + 1.0) * 127.5) as u8;

            points.push(Point3D {
                position: Point3::new(x, y, z + 1.0),
                w: 1.0,
                color: [r, g, b],
                rms_reprojection_error: 0.5 + (i as f64 / num_points.max(1) as f64) as f32 * 0.5,
                normal: Vector3::new(x as f32, y as f32, z as f32).normalize(),
            });
        }

        // Simple tracks: each point observed by 2 adjacent cameras. The parallel
        // feature_indexes column goes into the sift_files ObservationSource.
        let mut tracks = Vec::new();
        let mut feature_indexes = Vec::new();
        let mut observation_counts = Vec::with_capacity(num_points);
        for i in 0..num_points {
            let cam1 = (i % num_images) as u32;
            let cam2 = ((i + 1) % num_images) as u32;
            let (first, second) = if cam1 <= cam2 {
                (cam1, cam2)
            } else {
                (cam2, cam1)
            };
            tracks.push(TrackObservation {
                image_index: first,
                point_index: i as u32,
            });
            tracks.push(TrackObservation {
                image_index: second,
                point_index: i as u32,
            });
            feature_indexes.push(i as u32);
            feature_indexes.push(i as u32);
            observation_counts.push(2);
        }

        // Each observation's pixel is its point projected through its image's
        // pose, as a solve would have detected it.
        let keypoints_xy = ndarray::Array2::from_shape_fn((tracks.len(), 2), |(row, k)| {
            let obs = &tracks[row];
            let image = &images[obs.image_index as usize];
            let p = image.quaternion_wxyz * points[obs.point_index as usize].position.coords
                + image.translation_xyz;
            let (u, v) = cameras[0].project(p.x / -p.z, p.y / -p.z);
            [u as f32, v as f32][k]
        });

        // Empty depth statistics
        let depth_statistics = DepthStatistics {
            num_histogram_buckets: num_buckets,
            images: (0..num_images)
                .map(|_| ImageDepthStats {
                    histogram_min_z: None,
                    histogram_max_z: None,
                    observed: ObservedDepthStats {
                        count: 0,
                        infinity_count: 0,
                        min_z: None,
                        max_z: None,
                        median_z: None,
                        mean_z: None,
                    },
                })
                .collect(),
        };
        let depth_histogram_counts = vec![vec![0u32; num_buckets as usize]; num_images];

        let metadata = SfmrMetadata {
            version: 2,
            operation: "demo".into(),
            tool: "sfmtool".into(),
            tool_version: "0.1.0".into(),
            tool_options: BTreeMap::new(),
            workspace: sfmtool_sfmr_format::WorkspaceMetadata {
                absolute_path: String::new(),
                relative_path: ".".into(),
                contents: sfmtool_sfmr_format::WorkspaceContents {
                    feature_tool: "none".into(),
                    feature_type: "sift".into(),
                    feature_options: serde_json::json!({}),
                    feature_prefix_dir: String::new(),
                },
            },
            timestamp: String::new(),
            image_count: num_images as u32,
            point_count: num_points as u32,
            infinity_point_count: 0,
            observation_count: (num_points * 2) as u32,
            camera_count: 1,
            rig_count: None,
            sensor_count: None,
            frame_count: None,
            distance_measurement_count: None,
            world_space_unit: None,
            feature_source: FEATURE_SOURCE_SIFT_FILES.to_string(),
        };

        let observation_offsets = compute_observation_offsets(&observation_counts);

        // Build per-image feature→point mapping
        let mut image_feature_to_point = vec![HashMap::new(); num_images];
        let mut max_track_feature_index = vec![0u32; num_images];
        for (obs, &feat) in tracks.iter().zip(&feature_indexes) {
            let img = obs.image_index as usize;
            image_feature_to_point[img].insert(feat, obs.point_index);
            max_track_feature_index[img] = max_track_feature_index[img].max(feat);
        }

        let infinity_point_count = count_points_at_infinity(&points);
        SfmrReconstruction {
            workspace_dir: PathBuf::new(),
            metadata,
            // No file behind this value, which is what the default state says.
            content_hash: ContentHash::default(),
            image_set: ImageSet {
                cameras,
                per_image_adjustments: Vec::new(),
                images,
                // Demo data has no photographs, and a zero column would be the
                // placeholder rows the format forbids.
                thumbnails_y_x_rgb: None,
                image_to_gray: Some(ImageToGray::default()),
                depth_statistics,
                depth_histogram_counts,
                rig_frame_data: None,
                measurements: Default::default(),
            },
            point_set: PointSet {
                infinity_point_count,
                patch_u_halfvec_xyz: None,
                patch_v_halfvec_xyz: None,
                patch_bitmaps_y_x_rgba: None,
                patch_bitmaps_for_display: false,
                has_normals: true,
                normal_confidence: None,
                point_constraints: None,
                observation_confidence: None,
                observation_readings: None,
                reference_observations: None,
                display_only_references: None,
                observations: ObservationSource::SiftFiles {
                    feature_indexes,
                    feature_tool_hashes: vec![[0u8; 16]; num_images],
                    sift_content_hashes: vec![[0u8; 16]; num_images],
                },
                keypoints_xy: Some(keypoints_xy),
                points,
                tracks,
                observation_counts,
                observation_offsets,
                image_feature_to_point,
                max_track_feature_index,
            },
        }
    }
}

#[cfg(test)]
impl SfmrReconstruction {
    /// [`Self::demo`] without its inline keypoints: the value a load gives for
    /// a version 16 `sift_files` file written without `tracks/keypoints_xy`
    /// whose `.sift` files cannot be read, which a save refuses to write.
    pub(crate) fn demo_without_keypoints(num_points: usize) -> Self {
        let mut recon = Self::demo(num_points);
        recon.point_set.keypoints_xy = None;
        recon
    }
}

/// Rewrite the version 17 `sift_files` `.sfmr` at `path` in place as a version
/// 16 file written without `tracks/keypoints_xy`, which version 16 allowed:
/// the entry left out, `has_keypoints_xy` false and `metadata.json` at version
/// 16. The readers do not check the content hashes, so they are left as they
/// are.
#[cfg(test)]
pub(crate) fn rewrite_as_version_16_without_keypoints(path: &std::path::Path) {
    use std::io::{Read, Write};

    let mut archive = zip::ZipArchive::new(std::fs::File::open(path).unwrap()).unwrap();
    let mut entries = Vec::new();
    for i in 0..archive.len() {
        let mut entry = archive.by_index(i).unwrap();
        let name = entry.name().unwrap().to_string();
        let mut compressed = Vec::new();
        entry.read_to_end(&mut compressed).unwrap();
        entries.push((name, compressed));
    }
    drop(archive);
    let edit_json = |compressed: &[u8], key: &str, value: serde_json::Value| {
        let raw = zstd::stream::decode_all(compressed).unwrap();
        let mut json: serde_json::Value = serde_json::from_slice(&raw).unwrap();
        json[key] = value;
        zstd::bulk::compress(&serde_json::to_vec(&json).unwrap(), 3).unwrap()
    };
    let mut zip = zip::ZipWriter::new(std::fs::File::create(path).unwrap());
    let stored =
        zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored);
    for (name, compressed) in entries {
        let bytes = match name.as_str() {
            n if n.starts_with("tracks/keypoints_xy.") => continue,
            "metadata.json.zst" => edit_json(&compressed, "version", serde_json::json!(16)),
            "tracks/metadata.json.zst" => {
                edit_json(&compressed, "has_keypoints_xy", serde_json::json!(false))
            }
            _ => compressed,
        };
        zip.start_file(&name, stored).unwrap();
        zip.write_all(&bytes).unwrap();
    }
    zip.finish().unwrap();
}
