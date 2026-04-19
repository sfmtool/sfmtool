// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! `.sfmr` file format reading and writing.
//!
//! The `.sfmr` format is sfmtool's native reconstruction file format.
//! It stores SfM reconstructions as ZIP archives with zstandard-compressed
//! columnar binary data and JSON metadata.
//!
//! See `docs/sfmr-file-format.md` for the specification.

#[cfg(not(target_endian = "little"))]
compile_error!(
    "sfmr-format requires a little-endian target (binary arrays are stored as little-endian)"
);

pub(crate) mod archive_io;
mod depth_stats;
mod read;
mod types;
mod verify;
mod write;

pub use depth_stats::{compute_depth_statistics, DepthStatsResult};
pub use read::{read_sfmr, read_sfmr_metadata, resolve_workspace_dir};
pub use types::*;
pub use verify::verify_sfmr;
pub use write::{write_sfmr, write_sfmr_with_options, WriteOptions};

#[cfg(test)]
mod tests {
    use crate::*;
    use ndarray::{Array1, Array2, Array4};
    use std::collections::HashMap;

    /// Create minimal valid SfmrData for testing.
    fn make_test_data() -> SfmrData {
        let image_count = 3;
        let points3d_count = 5;
        let observation_count = 8;
        let num_buckets = 128;

        // Build track arrays sorted by (points3d_indexes, image_indexes)
        // Point 0: observed by images 0, 1 (2 obs)
        // Point 1: observed by images 0, 1, 2 (3 obs)
        // Point 2: observed by image 2 (1 obs)
        // Point 3: observed by image 1 (1 obs)
        // Point 4: observed by image 0 (1 obs)
        let points3d_indexes = Array1::from_vec(vec![0, 0, 1, 1, 1, 2, 3, 4]);
        let image_indexes = Array1::from_vec(vec![0, 1, 0, 1, 2, 2, 1, 0]);
        let feature_indexes = Array1::from_vec(vec![0, 0, 1, 1, 0, 1, 2, 2]);
        let observation_counts = Array1::from_vec(vec![2, 3, 1, 1, 1]);

        SfmrData {
            workspace_dir: None,
            metadata: SfmrMetadata {
                version: 1,
                operation: "sfm_solve".into(),
                tool: "colmap".into(),
                tool_version: "3.10".into(),
                tool_options: HashMap::new(),
                workspace: WorkspaceMetadata {
                    absolute_path: "/tmp/workspace".into(),
                    relative_path: "..".into(),
                    contents: WorkspaceContents {
                        feature_tool: "colmap".into(),
                        feature_type: "sift".into(),
                        feature_options: serde_json::json!({}),
                        feature_prefix_dir: String::new(),
                    },
                },
                timestamp: "2025-12-21T14:32:15.123456Z".into(),
                image_count: image_count as u32,
                points3d_count: points3d_count as u32,
                observation_count: observation_count as u32,
                camera_count: 1,
                rig_count: None,
                sensor_count: None,
                frame_count: None,
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
            rig_frame_data: None,
            cameras: vec![SfmrCamera {
                model: "PINHOLE".into(),
                width: 1920,
                height: 1080,
                parameters: [
                    ("focal_length_x".into(), 1000.0),
                    ("focal_length_y".into(), 1000.0),
                    ("principal_point_x".into(), 960.0),
                    ("principal_point_y".into(), 540.0),
                ]
                .into_iter()
                .collect(),
            }],
            image_names: vec![
                "frame_001.jpg".into(),
                "frame_002.jpg".into(),
                "frame_003.jpg".into(),
            ],
            camera_indexes: Array1::from_vec(vec![0, 0, 0]),
            quaternions_wxyz: Array2::from_shape_vec(
                (image_count, 4),
                vec![
                    1.0, 0.0, 0.0, 0.0, 0.9239, 0.0, 0.3827, 0.0, 0.7071, 0.0, 0.7071, 0.0,
                ],
            )
            .unwrap(),
            translations_xyz: Array2::zeros((image_count, 3)),
            feature_tool_hashes: vec![[0u8; 16]; image_count],
            sift_content_hashes: vec![[1u8; 16]; image_count],
            thumbnails_y_x_rgb: Array4::zeros((image_count, 128, 128, 3)),
            positions_xyz: Array2::from_shape_vec(
                (points3d_count, 3),
                vec![
                    0.0, 0.0, 5.0, 1.0, 0.0, 6.0, -1.0, 1.0, 4.0, 0.5, -0.5, 7.0, -0.5, 0.5, 3.0,
                ],
            )
            .unwrap(),
            colors_rgb: Array2::from_shape_vec(
                (points3d_count, 3),
                vec![255, 0, 0, 0, 255, 0, 0, 0, 255, 128, 128, 0, 0, 128, 128],
            )
            .unwrap(),
            reprojection_errors: Array1::from_vec(vec![0.5, 0.6, 0.7, 0.8, 0.4]),
            estimated_normals_xyz: Array2::from_shape_vec(
                (points3d_count, 3),
                vec![
                    0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                ],
            )
            .unwrap(),
            image_indexes,
            feature_indexes,
            points3d_indexes,
            observation_counts,
            depth_statistics: DepthStatistics {
                num_histogram_buckets: num_buckets as u32,
                images: vec![
                    ImageDepthStats {
                        histogram_min_z: Some(3.0),
                        histogram_max_z: Some(7.0),
                        observed: ObservedDepthStats {
                            count: 3,
                            min_z: Some(3.0),
                            max_z: Some(7.0),
                            median_z: Some(5.0),
                            mean_z: Some(5.0),
                        },
                    },
                    ImageDepthStats {
                        histogram_min_z: Some(4.0),
                        histogram_max_z: Some(7.0),
                        observed: ObservedDepthStats {
                            count: 3,
                            min_z: Some(4.0),
                            max_z: Some(7.0),
                            median_z: Some(5.5),
                            mean_z: Some(5.5),
                        },
                    },
                    ImageDepthStats {
                        histogram_min_z: Some(4.0),
                        histogram_max_z: Some(6.0),
                        observed: ObservedDepthStats {
                            count: 2,
                            min_z: Some(4.0),
                            max_z: Some(6.0),
                            median_z: Some(5.0),
                            mean_z: Some(5.0),
                        },
                    },
                ],
            },
            observed_depth_histogram_counts: Array2::zeros((image_count, num_buckets)),
        }
    }

    #[test]
    fn test_round_trip() {
        let mut data = make_test_data();
        let dir = std::env::temp_dir().join("sfmr_test_round_trip");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.sfmr");

        // Write with skip_recompute_depth_stats to test exact round-trip fidelity
        let options = WriteOptions {
            skip_recompute_depth_stats: true,
            ..Default::default()
        };
        write_sfmr_with_options(&path, &mut data, &options).unwrap();

        // Read back
        let loaded = read_sfmr(&path).unwrap();

        // Verify metadata
        assert_eq!(loaded.metadata.operation, "sfm_solve");
        assert_eq!(loaded.metadata.tool, "colmap");
        assert_eq!(loaded.metadata.image_count, 3);
        assert_eq!(loaded.metadata.points3d_count, 5);
        assert_eq!(loaded.metadata.observation_count, 8);
        assert_eq!(loaded.metadata.camera_count, 1);

        // Verify cameras
        assert_eq!(loaded.cameras.len(), 1);
        assert_eq!(loaded.cameras[0].model, "PINHOLE");
        assert_eq!(loaded.cameras[0].width, 1920);

        // Verify images
        assert_eq!(loaded.image_names, data.image_names);
        assert_eq!(loaded.camera_indexes, data.camera_indexes);
        assert_eq!(loaded.quaternions_wxyz, data.quaternions_wxyz);
        assert_eq!(loaded.translations_xyz, data.translations_xyz);
        assert_eq!(loaded.feature_tool_hashes, data.feature_tool_hashes);
        assert_eq!(loaded.sift_content_hashes, data.sift_content_hashes);
        assert_eq!(loaded.thumbnails_y_x_rgb, data.thumbnails_y_x_rgb);

        // Verify points3d
        assert_eq!(loaded.positions_xyz, data.positions_xyz);
        assert_eq!(loaded.colors_rgb, data.colors_rgb);
        assert_eq!(loaded.reprojection_errors, data.reprojection_errors);
        assert_eq!(loaded.estimated_normals_xyz, data.estimated_normals_xyz);

        // Verify tracks
        assert_eq!(loaded.image_indexes, data.image_indexes);
        assert_eq!(loaded.feature_indexes, data.feature_indexes);
        assert_eq!(loaded.points3d_indexes, data.points3d_indexes);
        assert_eq!(loaded.observation_counts, data.observation_counts);

        // Verify depth statistics
        assert_eq!(
            loaded.depth_statistics.num_histogram_buckets,
            data.depth_statistics.num_histogram_buckets
        );
        assert_eq!(
            loaded.observed_depth_histogram_counts,
            data.observed_depth_histogram_counts
        );

        // Clean up
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_round_trip_verify() {
        let mut data = make_test_data();
        let dir = std::env::temp_dir().join("sfmr_test_verify");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.sfmr");

        write_sfmr(&path, &mut data).unwrap();

        let (valid, errors) = verify_sfmr(&path).unwrap();
        assert!(valid, "Verification failed: {:?}", errors);
        assert!(errors.is_empty());

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_read_metadata_only() {
        let mut data = make_test_data();
        let dir = std::env::temp_dir().join("sfmr_test_metadata");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.sfmr");

        write_sfmr(&path, &mut data).unwrap();

        let metadata = read_sfmr_metadata(&path).unwrap();
        assert_eq!(metadata.operation, "sfm_solve");
        assert_eq!(metadata.image_count, 3);

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_content_hash_populated() {
        let mut data = make_test_data();
        let dir = std::env::temp_dir().join("sfmr_test_hash");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.sfmr");

        write_sfmr(&path, &mut data).unwrap();
        let loaded = read_sfmr(&path).unwrap();

        // All hashes should be non-empty 32-char hex strings
        assert_eq!(loaded.content_hash.metadata_xxh128.len(), 32);
        assert_eq!(loaded.content_hash.cameras_xxh128.len(), 32);
        assert_eq!(loaded.content_hash.images_xxh128.len(), 32);
        assert_eq!(loaded.content_hash.points3d_xxh128.len(), 32);
        assert_eq!(loaded.content_hash.tracks_xxh128.len(), 32);
        assert_eq!(loaded.content_hash.content_xxh128.len(), 32);

        std::fs::remove_dir_all(&dir).unwrap();
    }

    // NOTE: test_reconstruction_round_trip is intentionally omitted here.
    // It tests SfmrReconstruction which lives in sfmtool-core, not sfmr-format.
    // That test remains in sfmtool-core.

    #[test]
    fn test_empty_reconstruction() {
        let num_buckets = 128;
        let mut data = SfmrData {
            workspace_dir: None,
            metadata: SfmrMetadata {
                version: 1,
                operation: "sfm_solve".into(),
                tool: "colmap".into(),
                tool_version: "3.10".into(),
                tool_options: HashMap::new(),
                workspace: WorkspaceMetadata {
                    absolute_path: "/tmp/workspace".into(),
                    relative_path: "..".into(),
                    contents: WorkspaceContents {
                        feature_tool: "colmap".into(),
                        feature_type: "sift".into(),
                        feature_options: serde_json::json!({}),
                        feature_prefix_dir: String::new(),
                    },
                },
                timestamp: "2025-12-21T14:32:15.123456Z".into(),
                image_count: 0,
                points3d_count: 0,
                observation_count: 0,
                camera_count: 0,
                rig_count: None,
                sensor_count: None,
                frame_count: None,
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
            rig_frame_data: None,
            cameras: vec![],
            image_names: vec![],
            camera_indexes: Array1::from_vec(vec![]),
            quaternions_wxyz: Array2::zeros((0, 4)),
            translations_xyz: Array2::zeros((0, 3)),
            feature_tool_hashes: vec![],
            sift_content_hashes: vec![],
            thumbnails_y_x_rgb: Array4::zeros((0, 128, 128, 3)),
            positions_xyz: Array2::zeros((0, 3)),
            colors_rgb: Array2::zeros((0, 3)),
            reprojection_errors: Array1::from_vec(vec![]),
            estimated_normals_xyz: Array2::zeros((0, 3)),
            image_indexes: Array1::from_vec(vec![]),
            feature_indexes: Array1::from_vec(vec![]),
            points3d_indexes: Array1::from_vec(vec![]),
            observation_counts: Array1::from_vec(vec![]),
            depth_statistics: DepthStatistics {
                num_histogram_buckets: num_buckets as u32,
                images: vec![],
            },
            observed_depth_histogram_counts: Array2::zeros((0, num_buckets)),
        };

        let dir = std::env::temp_dir().join("sfmr_test_empty");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.sfmr");

        let options = WriteOptions {
            skip_recompute_depth_stats: true,
            ..Default::default()
        };
        write_sfmr_with_options(&path, &mut data, &options).unwrap();

        // Read back
        let loaded = read_sfmr(&path).unwrap();
        assert_eq!(loaded.metadata.image_count, 0);
        assert_eq!(loaded.metadata.points3d_count, 0);
        assert_eq!(loaded.metadata.observation_count, 0);
        assert_eq!(loaded.cameras.len(), 0);
        assert_eq!(loaded.image_names.len(), 0);
        assert_eq!(loaded.positions_xyz.shape(), &[0, 3]);

        // Verify
        let (valid, errors) = verify_sfmr(&path).unwrap();
        assert!(
            valid,
            "Empty reconstruction verification failed: {:?}",
            errors
        );

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_multiple_cameras() {
        let mut data = make_test_data();
        // Add a second camera (OPENCV model)
        data.cameras.push(SfmrCamera {
            model: "OPENCV".into(),
            width: 3840,
            height: 2160,
            parameters: [
                ("focal_length_x".into(), 2000.0),
                ("focal_length_y".into(), 2000.0),
                ("principal_point_x".into(), 1920.0),
                ("principal_point_y".into(), 1080.0),
                ("radial_distortion_k1".into(), 0.01),
                ("radial_distortion_k2".into(), -0.02),
                ("tangential_distortion_p1".into(), 0.001),
                ("tangential_distortion_p2".into(), -0.001),
            ]
            .into_iter()
            .collect(),
        });
        data.metadata.camera_count = 2;
        // Assign image 2 to camera 1
        data.camera_indexes[2] = 1;

        let dir = std::env::temp_dir().join("sfmr_test_multi_cam");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.sfmr");

        write_sfmr(&path, &mut data).unwrap();

        let loaded = read_sfmr(&path).unwrap();
        assert_eq!(loaded.cameras.len(), 2);
        assert_eq!(loaded.cameras[0].model, "PINHOLE");
        assert_eq!(loaded.cameras[1].model, "OPENCV");
        assert_eq!(loaded.cameras[1].width, 3840);
        assert_eq!(loaded.camera_indexes[2], 1);

        let (valid, errors) = verify_sfmr(&path).unwrap();
        assert!(valid, "Multi-camera verification failed: {:?}", errors);

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_normals_are_unit_vectors() {
        let mut data = make_test_data();
        let dir = std::env::temp_dir().join("sfmr_test_normals");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.sfmr");

        write_sfmr(&path, &mut data).unwrap();
        let loaded = read_sfmr(&path).unwrap();

        // Verify all normals are unit vectors
        for i in 0..loaded.estimated_normals_xyz.shape()[0] {
            let nx = loaded.estimated_normals_xyz[[i, 0]] as f64;
            let ny = loaded.estimated_normals_xyz[[i, 1]] as f64;
            let nz = loaded.estimated_normals_xyz[[i, 2]] as f64;
            let norm = (nx * nx + ny * ny + nz * nz).sqrt();
            assert!(
                (norm - 1.0).abs() < 0.01,
                "Normal {i} should be unit vector, got norm = {norm}"
            );
        }

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_depth_statistics_structure() {
        let mut data = make_test_data();
        let dir = std::env::temp_dir().join("sfmr_test_depth_stats");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.sfmr");

        write_sfmr(&path, &mut data).unwrap();
        let loaded = read_sfmr(&path).unwrap();

        // Verify structure
        assert_eq!(loaded.depth_statistics.num_histogram_buckets, 128);
        assert_eq!(
            loaded.depth_statistics.images.len(),
            loaded.metadata.image_count as usize
        );
        assert_eq!(
            loaded.observed_depth_histogram_counts.shape(),
            &[loaded.metadata.image_count as usize, 128]
        );

        // Each image should have observed stats with count
        for (i, img_stats) in loaded.depth_statistics.images.iter().enumerate() {
            assert!(
                img_stats.observed.count > 0 || loaded.metadata.observation_count == 0,
                "Image {i} should have observations"
            );
        }

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_read_nonexistent_file() {
        let result = read_sfmr(std::path::Path::new("nonexistent.sfmr"));
        match result {
            Err(err) => {
                let msg = err.to_string();
                assert!(
                    msg.contains("nonexistent.sfmr"),
                    "Error message should contain the file path, got: {msg}"
                );
            }
            Ok(_) => panic!("Expected error for nonexistent file"),
        }
    }

    #[test]
    fn test_round_trip_with_rigs() {
        let mut data = make_test_data();

        // Add rig and frame data: a stereo rig with 2 sensors
        data.metadata.rig_count = Some(1);
        data.metadata.sensor_count = Some(2);
        data.metadata.frame_count = Some(2);

        data.rig_frame_data = Some(RigFrameData {
            rigs_metadata: RigsMetadata {
                rig_count: 1,
                sensor_count: 2,
                rigs: vec![RigDefinition {
                    name: "stereo".into(),
                    sensor_count: 2,
                    sensor_offset: 0,
                    ref_sensor_name: "left".into(),
                    sensor_names: vec!["left".into(), "right".into()],
                }],
            },
            sensor_camera_indexes: Array1::from_vec(vec![0, 0]),
            sensor_quaternions_wxyz: Array2::from_shape_vec(
                (2, 4),
                vec![
                    1.0, 0.0, 0.0, 0.0, // identity for ref sensor
                    0.9239, 0.0, 0.3827, 0.0, // rotation for second sensor
                ],
            )
            .unwrap(),
            sensor_translations_xyz: Array2::from_shape_vec(
                (2, 3),
                vec![
                    0.0, 0.0, 0.0, // zero for ref sensor
                    0.1, 0.0, 0.0, // baseline for second sensor
                ],
            )
            .unwrap(),
            frames_metadata: FramesMetadata { frame_count: 2 },
            rig_indexes: Array1::from_vec(vec![0, 0]),
            // 3 images: img0 → sensor 0, frame 0; img1 → sensor 1, frame 0; img2 → sensor 0, frame 1
            image_sensor_indexes: Array1::from_vec(vec![0, 1, 0]),
            image_frame_indexes: Array1::from_vec(vec![0, 0, 1]),
        });

        let dir = std::env::temp_dir().join("sfmr_test_rigs_round_trip");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.sfmr");

        let options = WriteOptions {
            skip_recompute_depth_stats: true,
            ..Default::default()
        };
        write_sfmr_with_options(&path, &mut data, &options).unwrap();

        // Read back
        let loaded = read_sfmr(&path).unwrap();

        // Verify rig data was preserved
        let rf = loaded
            .rig_frame_data
            .as_ref()
            .expect("rig_frame_data should be present");
        assert_eq!(rf.rigs_metadata.rig_count, 1);
        assert_eq!(rf.rigs_metadata.sensor_count, 2);
        assert_eq!(rf.rigs_metadata.rigs[0].name, "stereo");
        assert_eq!(
            rf.rigs_metadata.rigs[0].sensor_names,
            vec!["left".to_string(), "right".to_string()]
        );
        assert_eq!(rf.sensor_camera_indexes, Array1::from_vec(vec![0, 0]));
        assert_eq!(rf.sensor_quaternions_wxyz.shape(), &[2, 4]);
        assert_eq!(rf.sensor_translations_xyz.shape(), &[2, 3]);

        // Verify frame data was preserved
        assert_eq!(rf.frames_metadata.frame_count, 2);
        assert_eq!(rf.rig_indexes, Array1::from_vec(vec![0, 0]));
        assert_eq!(rf.image_sensor_indexes, Array1::from_vec(vec![0, 1, 0]));
        assert_eq!(rf.image_frame_indexes, Array1::from_vec(vec![0, 0, 1]));

        // Verify metadata counts
        assert_eq!(loaded.metadata.rig_count, Some(1));
        assert_eq!(loaded.metadata.sensor_count, Some(2));
        assert_eq!(loaded.metadata.frame_count, Some(2));

        // Verify content hashes
        assert!(loaded.content_hash.rigs_xxh128.is_some());
        assert!(loaded.content_hash.frames_xxh128.is_some());

        // Verify integrity
        let (valid, errors) = verify_sfmr(&path).unwrap();
        assert!(valid, "Rig round-trip verification failed: {:?}", errors);

        std::fs::remove_dir_all(&dir).unwrap();
    }

    // Every entry in a .sfmr archive MUST use ZIP's STORE method. Entries are already
    // zstandard-compressed; applying ZIP-level DEFLATE would double-compress and break
    // the spec guarantee of random access via simple seek.
    #[test]
    fn test_archive_uses_stored_compression() {
        let mut data = make_test_data();
        let dir = std::env::temp_dir().join("sfmr_test_stored_compression");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.sfmr");

        write_sfmr(&path, &mut data).unwrap();

        let file = std::fs::File::open(&path).unwrap();
        let mut archive = zip::ZipArchive::new(file).unwrap();
        assert!(archive.len() > 10, "expected a populated archive");
        for i in 0..archive.len() {
            let entry = archive.by_index(i).unwrap();
            assert_eq!(
                entry.compression(),
                zip::CompressionMethod::Stored,
                "entry '{}' uses {:?}, expected Stored",
                entry.name(),
                entry.compression(),
            );
        }

        std::fs::remove_dir_all(&dir).unwrap();
    }
}
