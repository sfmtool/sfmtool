// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Read and write COLMAP binary reconstruction files.
//!
//! This module handles the three binary files that comprise a COLMAP reconstruction:
//! `cameras.bin`, `images.bin`, and `points3D.bin`.

mod read;
mod types;
mod write;

pub use read::read_colmap_binary;
pub use types::{
    camera_params_to_array, colmap_model_id, ColmapDataId, ColmapFrame, ColmapIoError,
    ColmapReconstruction, ColmapRig, ColmapRigSensor, ColmapSensor, ColmapSensorType,
    ColmapWriteData, Keypoint2D,
};
pub use write::{write_colmap_binary, write_frames_bin, write_rigs_bin};

#[cfg(test)]
mod tests {
    use sfmr_format::SfmrCamera;

    use super::*;

    /// Helper to create a PINHOLE camera.
    fn make_pinhole_camera(
        width: u32,
        height: u32,
        fx: f64,
        fy: f64,
        cx: f64,
        cy: f64,
    ) -> SfmrCamera {
        SfmrCamera {
            model: "PINHOLE".into(),
            width,
            height,
            parameters: [
                ("focal_length_x".into(), fx),
                ("focal_length_y".into(), fy),
                ("principal_point_x".into(), cx),
                ("principal_point_y".into(), cy),
            ]
            .into_iter()
            .collect(),
        }
    }

    #[test]
    fn test_round_trip() {
        let cameras = vec![make_pinhole_camera(
            1920, 1080, 1000.0, 1000.0, 960.0, 540.0,
        )];
        let image_names = vec![
            "frame_001.jpg".to_string(),
            "frame_002.jpg".to_string(),
            "frame_003.jpg".to_string(),
        ];
        let camera_indexes = vec![0u32, 0, 0];
        let quaternions_wxyz = vec![
            [1.0, 0.0, 0.0, 0.0],
            [0.9239, 0.0, 0.3827, 0.0],
            [
                std::f64::consts::FRAC_1_SQRT_2,
                0.0,
                std::f64::consts::FRAC_1_SQRT_2,
                0.0,
            ],
        ];
        let translations_xyz = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]];

        // 3D points
        let positions_xyz = vec![[0.0, 0.0, 5.0], [1.0, 0.0, 6.0]];
        let colors_rgb = vec![[255u8, 0, 0], [0, 255, 0]];
        let reprojection_errors = vec![0.5, 0.6];

        // Tracks: point 0 seen by images 0,1; point 1 seen by images 1,2
        let track_point3d_indexes = vec![0u32, 0, 1, 1];
        let track_image_indexes = vec![0u32, 1, 1, 2];
        let track_feature_indexes = vec![0u32, 0, 1, 0];

        // Per-image keypoints: image 0 has 2 kps, image 1 has 2 kps, image 2 has 1 kp
        let keypoints_per_image = vec![
            vec![[100.0, 200.0], [300.0, 400.0]],
            vec![[110.0, 210.0], [310.0, 410.0]],
            vec![[120.0, 220.0]],
        ];

        let write_data = ColmapWriteData {
            cameras: &cameras,
            image_names: &image_names,
            camera_indexes: &camera_indexes,
            quaternions_wxyz: &quaternions_wxyz,
            translations_xyz: &translations_xyz,
            positions_xyz: &positions_xyz,
            colors_rgb: &colors_rgb,
            reprojection_errors: &reprojection_errors,
            track_image_indexes: &track_image_indexes,
            track_feature_indexes: &track_feature_indexes,
            track_point3d_indexes: &track_point3d_indexes,
            keypoints_per_image: &keypoints_per_image,
            rigs: None,
            frames: None,
        };

        let dir = std::env::temp_dir().join("colmap_io_round_trip");
        std::fs::create_dir_all(&dir).unwrap();

        write_colmap_binary(&dir, &write_data).unwrap();
        let recon = read_colmap_binary(&dir).unwrap();

        // Images are sorted by name, which is already sorted here
        assert_eq!(recon.image_names, image_names);
        assert_eq!(recon.cameras.len(), 1);
        assert_eq!(recon.cameras[0].model, "PINHOLE");
        assert_eq!(recon.cameras[0].width, 1920);
        assert_eq!(recon.cameras[0].height, 1080);
        assert!((recon.cameras[0].parameters["focal_length_x"] - 1000.0).abs() < 1e-10);

        assert_eq!(recon.camera_indexes, camera_indexes);
        assert_eq!(recon.quaternions_wxyz, quaternions_wxyz);
        assert_eq!(recon.translations_xyz, translations_xyz);

        assert_eq!(recon.positions_xyz, positions_xyz);
        assert_eq!(recon.colors_rgb, colors_rgb);
        assert!((recon.reprojection_errors[0] - 0.5).abs() < 1e-10);
        assert!((recon.reprojection_errors[1] - 0.6).abs() < 1e-10);

        // Verify tracks
        assert_eq!(recon.tracks.len(), 2);
        // Point 0 seen by images 0,1 at feature 0,0
        assert!(recon.tracks[0].contains(&(0, 0)));
        assert!(recon.tracks[0].contains(&(1, 0)));
        // Point 1 seen by images 1,2 at features 1,0
        assert!(recon.tracks[1].contains(&(1, 1)));
        assert!(recon.tracks[1].contains(&(2, 0)));

        // Verify keypoints
        assert_eq!(recon.keypoints_per_image.len(), 3);
        assert_eq!(recon.keypoints_per_image[0].len(), 2);
        assert!((recon.keypoints_per_image[0][0].x - 100.0).abs() < 1e-10);
        assert!((recon.keypoints_per_image[0][0].y - 200.0).abs() < 1e-10);
        // Image 0, feature 0 -> point 0
        assert_eq!(recon.keypoints_per_image[0][0].point3d_index, Some(0));
        // Image 0, feature 1 -> no point
        assert_eq!(recon.keypoints_per_image[0][1].point3d_index, None);
        // Image 1, feature 0 -> point 0
        assert_eq!(recon.keypoints_per_image[1][0].point3d_index, Some(0));
        // Image 1, feature 1 -> point 1
        assert_eq!(recon.keypoints_per_image[1][1].point3d_index, Some(1));
        // Image 2, feature 0 -> point 1
        assert_eq!(recon.keypoints_per_image[2][0].point3d_index, Some(1));

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_unsorted_image_names() {
        // Write images in non-alphabetical order (C, A, B) and verify
        // that read sorts them to (A, B, C) with tracks remapped correctly.
        let cameras = vec![make_pinhole_camera(640, 480, 500.0, 500.0, 320.0, 240.0)];

        // Written in order C, A, B
        let image_names = vec![
            "img_C.jpg".to_string(),
            "img_A.jpg".to_string(),
            "img_B.jpg".to_string(),
        ];
        let camera_indexes = vec![0u32, 0, 0];
        let quaternions_wxyz = vec![
            [1.0, 0.0, 0.0, 0.0], // C
            [0.9, 0.1, 0.0, 0.0], // A
            [0.8, 0.2, 0.0, 0.0], // B
        ];
        let translations_xyz = vec![
            [3.0, 0.0, 0.0], // C
            [1.0, 0.0, 0.0], // A
            [2.0, 0.0, 0.0], // B
        ];

        // Point 0 seen by image 0 (C) at feat 0, and image 1 (A) at feat 0
        let track_point3d_indexes = vec![0u32, 0];
        let track_image_indexes = vec![0u32, 1]; // C=0, A=1
        let track_feature_indexes = vec![0u32, 0];

        let positions_xyz = vec![[5.0, 5.0, 5.0]];
        let colors_rgb = vec![[128u8, 128, 128]];
        let reprojection_errors = vec![0.3];

        let keypoints_per_image = vec![vec![[10.0, 20.0]], vec![[30.0, 40.0]], vec![[50.0, 60.0]]];

        let write_data = ColmapWriteData {
            cameras: &cameras,
            image_names: &image_names,
            camera_indexes: &camera_indexes,
            quaternions_wxyz: &quaternions_wxyz,
            translations_xyz: &translations_xyz,
            positions_xyz: &positions_xyz,
            colors_rgb: &colors_rgb,
            reprojection_errors: &reprojection_errors,
            track_image_indexes: &track_image_indexes,
            track_feature_indexes: &track_feature_indexes,
            track_point3d_indexes: &track_point3d_indexes,
            keypoints_per_image: &keypoints_per_image,
            rigs: None,
            frames: None,
        };

        let dir = std::env::temp_dir().join("colmap_io_unsorted");
        std::fs::create_dir_all(&dir).unwrap();

        write_colmap_binary(&dir, &write_data).unwrap();
        let recon = read_colmap_binary(&dir).unwrap();

        // After sorting: A=0, B=1, C=2
        assert_eq!(
            recon.image_names,
            vec!["img_A.jpg", "img_B.jpg", "img_C.jpg"]
        );

        // Verify poses are remapped: A had quat [0.9,...], B had [0.8,...], C had [1.0,...]
        assert_eq!(recon.quaternions_wxyz[0], [0.9, 0.1, 0.0, 0.0]); // A
        assert_eq!(recon.quaternions_wxyz[1], [0.8, 0.2, 0.0, 0.0]); // B
        assert_eq!(recon.quaternions_wxyz[2], [1.0, 0.0, 0.0, 0.0]); // C

        assert_eq!(recon.translations_xyz[0], [1.0, 0.0, 0.0]); // A
        assert_eq!(recon.translations_xyz[1], [2.0, 0.0, 0.0]); // B
        assert_eq!(recon.translations_xyz[2], [3.0, 0.0, 0.0]); // C

        // Verify keypoints are remapped
        assert!((recon.keypoints_per_image[0][0].x - 30.0).abs() < 1e-10); // A's kp
        assert!((recon.keypoints_per_image[1][0].x - 50.0).abs() < 1e-10); // B's kp
        assert!((recon.keypoints_per_image[2][0].x - 10.0).abs() < 1e-10); // C's kp

        // Track: point 0 was seen by C (write idx 0 -> sorted idx 2) and A (write idx 1 -> sorted idx 0)
        assert_eq!(recon.tracks.len(), 1);
        assert!(recon.tracks[0].contains(&(2, 0))); // C at feat 0
        assert!(recon.tracks[0].contains(&(0, 0))); // A at feat 0

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_camera_model_round_trip() {
        let cameras = vec![
            SfmrCamera {
                model: "SIMPLE_PINHOLE".into(),
                width: 640,
                height: 480,
                parameters: [
                    ("focal_length".into(), 500.0),
                    ("principal_point_x".into(), 320.0),
                    ("principal_point_y".into(), 240.0),
                ]
                .into_iter()
                .collect(),
            },
            SfmrCamera {
                model: "PINHOLE".into(),
                width: 1920,
                height: 1080,
                parameters: [
                    ("focal_length_x".into(), 1000.0),
                    ("focal_length_y".into(), 1001.0),
                    ("principal_point_x".into(), 960.0),
                    ("principal_point_y".into(), 540.0),
                ]
                .into_iter()
                .collect(),
            },
            SfmrCamera {
                model: "SIMPLE_RADIAL".into(),
                width: 800,
                height: 600,
                parameters: [
                    ("focal_length".into(), 600.0),
                    ("principal_point_x".into(), 400.0),
                    ("principal_point_y".into(), 300.0),
                    ("radial_distortion_k1".into(), 0.01),
                ]
                .into_iter()
                .collect(),
            },
            SfmrCamera {
                model: "RADIAL".into(),
                width: 800,
                height: 600,
                parameters: [
                    ("focal_length".into(), 600.0),
                    ("principal_point_x".into(), 400.0),
                    ("principal_point_y".into(), 300.0),
                    ("radial_distortion_k1".into(), 0.01),
                    ("radial_distortion_k2".into(), -0.02),
                ]
                .into_iter()
                .collect(),
            },
            SfmrCamera {
                model: "OPENCV".into(),
                width: 1920,
                height: 1080,
                parameters: [
                    ("focal_length_x".into(), 1000.0),
                    ("focal_length_y".into(), 1000.0),
                    ("principal_point_x".into(), 960.0),
                    ("principal_point_y".into(), 540.0),
                    ("radial_distortion_k1".into(), 0.1),
                    ("radial_distortion_k2".into(), -0.2),
                    ("tangential_distortion_p1".into(), 0.001),
                    ("tangential_distortion_p2".into(), -0.001),
                ]
                .into_iter()
                .collect(),
            },
            SfmrCamera {
                model: "FULL_OPENCV".into(),
                width: 3840,
                height: 2160,
                parameters: [
                    ("focal_length_x".into(), 2000.0),
                    ("focal_length_y".into(), 2000.0),
                    ("principal_point_x".into(), 1920.0),
                    ("principal_point_y".into(), 1080.0),
                    ("radial_distortion_k1".into(), 0.1),
                    ("radial_distortion_k2".into(), -0.2),
                    ("tangential_distortion_p1".into(), 0.001),
                    ("tangential_distortion_p2".into(), -0.001),
                    ("radial_distortion_k3".into(), 0.03),
                    ("radial_distortion_k4".into(), -0.04),
                    ("radial_distortion_k5".into(), 0.05),
                    ("radial_distortion_k6".into(), -0.06),
                ]
                .into_iter()
                .collect(),
            },
            SfmrCamera {
                model: "OPENCV_FISHEYE".into(),
                width: 1280,
                height: 720,
                parameters: [
                    ("focal_length_x".into(), 500.0),
                    ("focal_length_y".into(), 500.0),
                    ("principal_point_x".into(), 640.0),
                    ("principal_point_y".into(), 360.0),
                    ("radial_distortion_k1".into(), 0.1),
                    ("radial_distortion_k2".into(), -0.2),
                    ("radial_distortion_k3".into(), 0.03),
                    ("radial_distortion_k4".into(), -0.04),
                ]
                .into_iter()
                .collect(),
            },
        ];

        // Each camera assigned to one image with no keypoints
        let num_images = cameras.len();
        let image_names: Vec<String> = (0..num_images)
            .map(|i| format!("img_{:03}.jpg", i))
            .collect();
        let camera_indexes: Vec<u32> = (0..num_images as u32).collect();
        let quaternions_wxyz: Vec<[f64; 4]> = vec![[1.0, 0.0, 0.0, 0.0]; num_images];
        let translations_xyz: Vec<[f64; 3]> = vec![[0.0, 0.0, 0.0]; num_images];
        let keypoints_per_image: Vec<Vec<[f64; 2]>> = vec![vec![]; num_images];

        let write_data = ColmapWriteData {
            cameras: &cameras,
            image_names: &image_names,
            camera_indexes: &camera_indexes,
            quaternions_wxyz: &quaternions_wxyz,
            translations_xyz: &translations_xyz,
            positions_xyz: &[],
            colors_rgb: &[],
            reprojection_errors: &[],
            track_image_indexes: &[],
            track_feature_indexes: &[],
            track_point3d_indexes: &[],
            keypoints_per_image: &keypoints_per_image,
            rigs: None,
            frames: None,
        };

        let dir = std::env::temp_dir().join("colmap_io_camera_models");
        std::fs::create_dir_all(&dir).unwrap();

        write_colmap_binary(&dir, &write_data).unwrap();
        let recon = read_colmap_binary(&dir).unwrap();

        assert_eq!(recon.cameras.len(), cameras.len());
        for (orig, loaded) in cameras.iter().zip(recon.cameras.iter()) {
            assert_eq!(
                orig.model, loaded.model,
                "Model mismatch for {}",
                orig.model
            );
            assert_eq!(orig.width, loaded.width);
            assert_eq!(orig.height, loaded.height);
            for (key, &val) in &orig.parameters {
                let loaded_val = loaded.parameters[key];
                assert!(
                    (val - loaded_val).abs() < 1e-10,
                    "Param {} mismatch for {}: {} vs {}",
                    key,
                    orig.model,
                    val,
                    loaded_val
                );
            }
        }

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_empty_reconstruction() {
        let cameras: Vec<SfmrCamera> = vec![];
        let image_names: Vec<String> = vec![];
        let camera_indexes: Vec<u32> = vec![];
        let quaternions_wxyz: Vec<[f64; 4]> = vec![];
        let translations_xyz: Vec<[f64; 3]> = vec![];
        let keypoints_per_image: Vec<Vec<[f64; 2]>> = vec![];

        let write_data = ColmapWriteData {
            cameras: &cameras,
            image_names: &image_names,
            camera_indexes: &camera_indexes,
            quaternions_wxyz: &quaternions_wxyz,
            translations_xyz: &translations_xyz,
            positions_xyz: &[],
            colors_rgb: &[],
            reprojection_errors: &[],
            track_image_indexes: &[],
            track_feature_indexes: &[],
            track_point3d_indexes: &[],
            keypoints_per_image: &keypoints_per_image,
            rigs: None,
            frames: None,
        };

        let dir = std::env::temp_dir().join("colmap_io_empty");
        std::fs::create_dir_all(&dir).unwrap();

        write_colmap_binary(&dir, &write_data).unwrap();
        let recon = read_colmap_binary(&dir).unwrap();

        assert_eq!(recon.cameras.len(), 0);
        assert_eq!(recon.image_names.len(), 0);
        assert_eq!(recon.positions_xyz.len(), 0);
        assert_eq!(recon.tracks.len(), 0);
        assert_eq!(recon.keypoints_per_image.len(), 0);

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_rigs_frames_round_trip() {
        use types::{
            ColmapDataId, ColmapFrame, ColmapRig, ColmapRigSensor, ColmapSensor, ColmapSensorType,
        };
        use write::{write_frames_bin, write_rigs_bin};

        let dir = std::env::temp_dir().join("colmap_io_rigs_frames");
        std::fs::create_dir_all(&dir).unwrap();

        // Write a rig with a reference sensor (camera_id=1) and one non-ref sensor (camera_id=2).
        // COLMAP uses 1-based camera IDs.
        let rigs = vec![ColmapRig {
            rig_id: 1,
            ref_sensor: Some(ColmapSensor {
                sensor_type: ColmapSensorType::Camera,
                id: 1,
            }),
            non_ref_sensors: vec![ColmapRigSensor {
                sensor: ColmapSensor {
                    sensor_type: ColmapSensorType::Camera,
                    id: 2,
                },
                sensor_from_rig: Some((
                    [0.9239, 0.0, 0.3827, 0.0], // ~45° rotation around Y
                    [0.5, 0.0, 0.0],
                )),
            }],
        }];

        // Frame data_ids use 1-based COLMAP image IDs.
        // write_colmap_binary writes images sorted by name, so:
        //   image_id 1 -> "front/img_001.jpg" (sorted index 0)
        //   image_id 2 -> "front/img_002.jpg" (sorted index 1)
        //   image_id 3 -> "right/img_001.jpg" (sorted index 2)
        // But our writer writes in the order given, not sorted. The image names
        // below are already sorted, so image_id matches position + 1.
        let frames = vec![
            ColmapFrame {
                frame_id: 1,
                rig_id: 1,
                quaternion_wxyz: [1.0, 0.0, 0.0, 0.0],
                translation_xyz: [0.0, 0.0, 0.0],
                data_ids: vec![
                    ColmapDataId {
                        sensor_type: ColmapSensorType::Camera,
                        sensor_id: 1,
                        data_id: 1, // image_id 1
                    },
                    ColmapDataId {
                        sensor_type: ColmapSensorType::Camera,
                        sensor_id: 2,
                        data_id: 2, // image_id 2
                    },
                ],
            },
            ColmapFrame {
                frame_id: 2,
                rig_id: 1,
                quaternion_wxyz: [0.9239, 0.0, 0.3827, 0.0],
                translation_xyz: [1.0, 0.0, 0.0],
                data_ids: vec![ColmapDataId {
                    sensor_type: ColmapSensorType::Camera,
                    sensor_id: 1,
                    data_id: 3, // image_id 3
                }],
            },
        ];

        write_rigs_bin(&dir.join("rigs.bin"), &rigs).unwrap();
        write_frames_bin(&dir.join("frames.bin"), &frames).unwrap();

        // Also write minimal cameras/images/points3D for read_colmap_binary
        let cameras = vec![
            make_pinhole_camera(640, 480, 320.0, 320.0, 320.0, 240.0),
            make_pinhole_camera(640, 480, 320.0, 320.0, 320.0, 240.0),
        ];
        let image_names = vec![
            "front/img_001.jpg".to_string(),
            "right/img_001.jpg".to_string(),
            "front/img_002.jpg".to_string(),
        ];
        let camera_indexes = vec![0u32, 1, 0];
        let quaternions_wxyz = vec![
            [1.0, 0.0, 0.0, 0.0],
            [0.9239, 0.0, 0.3827, 0.0],
            [0.9239, 0.0, 0.3827, 0.0],
        ];
        let translations_xyz = vec![[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let keypoints_per_image = vec![vec![], vec![], vec![]];

        let write_data = ColmapWriteData {
            cameras: &cameras,
            image_names: &image_names,
            camera_indexes: &camera_indexes,
            quaternions_wxyz: &quaternions_wxyz,
            translations_xyz: &translations_xyz,
            positions_xyz: &[],
            colors_rgb: &[],
            reprojection_errors: &[],
            track_image_indexes: &[],
            track_feature_indexes: &[],
            track_point3d_indexes: &[],
            keypoints_per_image: &keypoints_per_image,
            rigs: None,
            frames: None,
        };
        write_colmap_binary(&dir, &write_data).unwrap();

        // Read back and verify rig data
        let recon = read_colmap_binary(&dir).unwrap();

        // Verify rigs — sensor IDs should be remapped to 0-based camera indexes
        let read_rigs = recon.rigs.as_ref().unwrap();
        assert_eq!(read_rigs.len(), 1);
        assert_eq!(read_rigs[0].rig_id, 1);
        assert!(read_rigs[0].ref_sensor.is_some());
        assert_eq!(read_rigs[0].ref_sensor.as_ref().unwrap().id, 0); // camera_id 1 -> index 0
        assert_eq!(read_rigs[0].non_ref_sensors.len(), 1);
        let non_ref = &read_rigs[0].non_ref_sensors[0];
        assert_eq!(non_ref.sensor.id, 1); // camera_id 2 -> index 1
        let (quat, trans) = non_ref.sensor_from_rig.unwrap();
        assert!((quat[0] - 0.9239).abs() < 1e-10);
        assert!((quat[2] - 0.3827).abs() < 1e-10);
        assert!((trans[0] - 0.5).abs() < 1e-10);

        // Verify frames — data_id values should be remapped to 0-based sorted image indexes
        let read_frames = recon.frames.as_ref().unwrap();
        assert_eq!(read_frames.len(), 2);
        assert_eq!(read_frames[0].frame_id, 1);
        assert_eq!(read_frames[0].rig_id, 1);
        assert_eq!(read_frames[0].data_ids.len(), 2);
        // sensor_id remapped to 0-based camera index
        assert_eq!(read_frames[0].data_ids[0].sensor_id, 0); // camera_id 1 -> index 0
        assert_eq!(read_frames[0].data_ids[1].sensor_id, 1); // camera_id 2 -> index 1

        // data_id remapped to 0-based sorted image index.
        // Write order: ["front/img_001.jpg", "right/img_001.jpg", "front/img_002.jpg"]
        // Sorted order: ["front/img_001.jpg", "front/img_002.jpg", "right/img_001.jpg"]
        // So: image_id 1 -> sorted 0, image_id 2 -> sorted 2, image_id 3 -> sorted 1
        assert_eq!(read_frames[0].data_ids[0].data_id, 0); // image_id 1 -> sorted index 0
        assert_eq!(read_frames[0].data_ids[1].data_id, 2); // image_id 2 -> sorted index 2
        assert_eq!(read_frames[1].frame_id, 2);
        assert_eq!(read_frames[1].data_ids.len(), 1);
        assert_eq!(read_frames[1].data_ids[0].data_id, 1); // image_id 3 -> sorted index 1

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_no_rigs_files() {
        // When rigs.bin/frames.bin don't exist, rigs and frames should be None
        let cameras = vec![make_pinhole_camera(640, 480, 320.0, 320.0, 320.0, 240.0)];
        let image_names = vec!["img.jpg".to_string()];
        let keypoints_per_image = vec![vec![]];

        let write_data = ColmapWriteData {
            cameras: &cameras,
            image_names: &image_names,
            camera_indexes: &[0],
            quaternions_wxyz: &[[1.0, 0.0, 0.0, 0.0]],
            translations_xyz: &[[0.0, 0.0, 0.0]],
            positions_xyz: &[],
            colors_rgb: &[],
            reprojection_errors: &[],
            track_image_indexes: &[],
            track_feature_indexes: &[],
            track_point3d_indexes: &[],
            keypoints_per_image: &keypoints_per_image,
            rigs: None,
            frames: None,
        };

        let dir = std::env::temp_dir().join("colmap_io_no_rigs");
        std::fs::create_dir_all(&dir).unwrap();
        write_colmap_binary(&dir, &write_data).unwrap();

        let recon = read_colmap_binary(&dir).unwrap();
        assert!(recon.rigs.is_none());
        assert!(recon.frames.is_none());

        std::fs::remove_dir_all(&dir).unwrap();
    }
}