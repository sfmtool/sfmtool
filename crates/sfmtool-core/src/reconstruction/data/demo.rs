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
use ndarray::Array4;

use sfmr_format::{
    ContentHash, DepthStatistics, ImageDepthStats, ObservedDepthStats, SfmrMetadata,
    FEATURE_SOURCE_SIFT_FILES,
};

use crate::camera::CameraIntrinsics;

use super::{
    compute_observation_offsets, count_points_at_infinity, ObservationSource, Point3D, SfmrImage,
    SfmrReconstruction, TrackObservation,
};

impl SfmrReconstruction {
    /// Creates a demo reconstruction with synthetic data for testing.
    ///
    /// Generates `num_points` 3D points evenly distributed on a unit sphere
    /// (offset to sit in front of the camera arc), observed by 8 cameras
    /// arranged in a circle around the origin.
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
            });
        }

        // Points: evenly distributed on the unit sphere via Thomson relaxation,
        // then offset by +1 in z so they sit in front of the camera arc.
        let sphere = evenly_distributed_sphere_points(num_points, &RelaxConfig::default());
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
                error: 0.5 + (i as f64 / num_points.max(1) as f64) as f32 * 0.5,
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
            workspace: sfmr_format::WorkspaceMetadata {
                absolute_path: String::new(),
                relative_path: ".".into(),
                contents: sfmr_format::WorkspaceContents {
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
            infinity_point_count,
            workspace_dir: PathBuf::new(),
            metadata,
            rig_frame_data: None,
            patch_u_halfvec_xyz: None,
            patch_v_halfvec_xyz: None,
            patch_bitmaps_y_x_rgba: None,
            has_normals: true,
            normal_confidence: None,
            observation_confidence: None,
            observations: ObservationSource::SiftFiles {
                feature_indexes,
                feature_tool_hashes: vec![[0u8; 16]; num_images],
                sift_content_hashes: vec![[0u8; 16]; num_images],
            },
            content_hash: ContentHash {
                metadata_xxh128: String::new(),
                cameras_xxh128: String::new(),
                rigs_xxh128: None,
                frames_xxh128: None,
                images_xxh128: String::new(),
                points3d_xxh128: String::new(),
                tracks_xxh128: String::new(),
                content_xxh128: String::new(),
            },
            cameras,
            images,
            points,
            tracks,
            observation_counts,
            observation_offsets,
            thumbnails_y_x_rgb: Array4::zeros((num_images, 128, 128, 3)),
            depth_statistics,
            depth_histogram_counts,
            image_feature_to_point,
            max_track_feature_index,
        }
    }
}
