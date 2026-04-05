// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Tests for COLMAP SQLite database I/O.

use rusqlite::Connection;
use sfmr_format::SfmrCamera;

use super::types::{PosePrior, TwoViewGeometry, TwoViewGeometryConfig};
use super::write::write_colmap_db;
use super::ColmapDbWriteData;

/// Helper to create a PINHOLE camera.
fn make_pinhole_camera(width: u32, height: u32, fx: f64, fy: f64, cx: f64, cy: f64) -> SfmrCamera {
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
fn test_write_and_read_back_cameras() {
    let dir = std::env::temp_dir().join("colmap_db_cameras");
    std::fs::create_dir_all(&dir).unwrap();
    let db_path = dir.join("test.db");

    let cameras = vec![make_pinhole_camera(
        1920, 1080, 1000.0, 1000.0, 960.0, 540.0,
    )];
    let image_names = vec!["frame_001.jpg".to_string()];
    let camera_indexes = vec![0u32];
    let quaternions_wxyz = vec![[1.0, 0.0, 0.0, 0.0]];
    let translations_xyz = vec![[0.0, 0.0, 0.0]];
    let keypoints_per_image = vec![vec![[100.0, 200.0], [300.0, 400.0]]];
    let descriptors_per_image = vec![vec![0u8; 2 * 128]];

    let data = ColmapDbWriteData {
        cameras: &cameras,
        image_names: &image_names,
        camera_indexes: &camera_indexes,
        quaternions_wxyz: &quaternions_wxyz,
        translations_xyz: &translations_xyz,
        keypoints_per_image: &keypoints_per_image,
        descriptors_per_image: &descriptors_per_image,
        descriptor_dim: 128,
        pose_priors: None,
        two_view_geometries: None,
        rigs: None,
        frames: None,
    };

    let image_ids = write_colmap_db(&db_path, &data).unwrap();
    assert_eq!(image_ids.len(), 1);

    // Read back and verify
    let conn = Connection::open(&db_path).unwrap();

    // Check cameras
    let (model, width, height): (i32, i32, i32) = conn
        .query_row("SELECT model, width, height FROM cameras", [], |row| {
            Ok((row.get(0)?, row.get(1)?, row.get(2)?))
        })
        .unwrap();
    assert_eq!(model, 1); // PINHOLE
    assert_eq!(width, 1920);
    assert_eq!(height, 1080);

    // Check camera params
    let params_blob: Vec<u8> = conn
        .query_row("SELECT params FROM cameras", [], |row| row.get(0))
        .unwrap();
    assert_eq!(params_blob.len(), 4 * 8); // 4 f64 params
    let params: Vec<f64> = params_blob
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert!((params[0] - 1000.0).abs() < 1e-10); // fx
    assert!((params[1] - 1000.0).abs() < 1e-10); // fy
    assert!((params[2] - 960.0).abs() < 1e-10); // cx
    assert!((params[3] - 540.0).abs() < 1e-10); // cy

    // Check images
    let (name, camera_id): (String, i64) = conn
        .query_row("SELECT name, camera_id FROM images", [], |row| {
            Ok((row.get(0)?, row.get(1)?))
        })
        .unwrap();
    assert_eq!(name, "frame_001.jpg");
    assert_eq!(camera_id, 1); // 1-based

    // Check keypoints
    let (rows, cols): (i32, i32) = conn
        .query_row("SELECT rows, cols FROM keypoints", [], |row| {
            Ok((row.get(0)?, row.get(1)?))
        })
        .unwrap();
    assert_eq!(rows, 2); // 2 keypoints
    assert_eq!(cols, 2); // x, y

    let kp_blob: Vec<u8> = conn
        .query_row("SELECT data FROM keypoints", [], |row| row.get(0))
        .unwrap();
    assert_eq!(kp_blob.len(), 2 * 2 * 4); // 2 keypoints × 2 coords × f32
    let kp_values: Vec<f32> = kp_blob
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert!((kp_values[0] - 100.0).abs() < 1e-5);
    assert!((kp_values[1] - 200.0).abs() < 1e-5);
    assert!((kp_values[2] - 300.0).abs() < 1e-5);
    assert!((kp_values[3] - 400.0).abs() < 1e-5);

    // Check descriptors
    let (desc_rows, desc_cols): (i32, i32) = conn
        .query_row("SELECT rows, cols FROM descriptors", [], |row| {
            Ok((row.get(0)?, row.get(1)?))
        })
        .unwrap();
    assert_eq!(desc_rows, 2);
    assert_eq!(desc_cols, 128);

    drop(conn);
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn test_write_pose_priors() {
    let dir = std::env::temp_dir().join("colmap_db_priors");
    std::fs::create_dir_all(&dir).unwrap();
    let db_path = dir.join("test.db");

    let cameras = vec![make_pinhole_camera(640, 480, 500.0, 500.0, 320.0, 240.0)];
    let image_names = vec!["img_001.jpg".to_string(), "img_002.jpg".to_string()];
    let camera_indexes = vec![0u32, 0];
    let quaternions_wxyz = vec![[1.0, 0.0, 0.0, 0.0], [0.9, 0.1, 0.0, 0.0]];
    let translations_xyz = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
    let keypoints_per_image = vec![vec![], vec![]];
    let descriptors_per_image = vec![vec![], vec![]];

    let pose_priors = vec![
        PosePrior {
            position: [1.0, 2.0, 3.0],
            position_covariance: [0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01],
            coordinate_system: 2, // CARTESIAN
        },
        PosePrior {
            position: [4.0, 5.0, 6.0],
            position_covariance: [0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01],
            coordinate_system: 2,
        },
    ];

    let data = ColmapDbWriteData {
        cameras: &cameras,
        image_names: &image_names,
        camera_indexes: &camera_indexes,
        quaternions_wxyz: &quaternions_wxyz,
        translations_xyz: &translations_xyz,
        keypoints_per_image: &keypoints_per_image,
        descriptors_per_image: &descriptors_per_image,
        descriptor_dim: 128,
        pose_priors: Some(&pose_priors),
        two_view_geometries: None,
        rigs: None,
        frames: None,
    };

    write_colmap_db(&db_path, &data).unwrap();

    // Read back and verify
    let conn = Connection::open(&db_path).unwrap();
    let count: i32 = conn
        .query_row("SELECT COUNT(*) FROM pose_priors", [], |row| row.get(0))
        .unwrap();
    assert_eq!(count, 2);

    // Check first prior position
    let pos_blob: Vec<u8> = conn
        .query_row(
            "SELECT position FROM pose_priors WHERE image_id = 1",
            [],
            |row| row.get(0),
        )
        .unwrap();
    let pos: Vec<f64> = pos_blob
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert!((pos[0] - 1.0).abs() < 1e-10);
    assert!((pos[1] - 2.0).abs() < 1e-10);
    assert!((pos[2] - 3.0).abs() < 1e-10);

    // Check coordinate system
    let coord_sys: i32 = conn
        .query_row(
            "SELECT coordinate_system FROM pose_priors WHERE image_id = 1",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(coord_sys, 2);

    drop(conn);
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn test_write_two_view_geometries() {
    let dir = std::env::temp_dir().join("colmap_db_tvg");
    std::fs::create_dir_all(&dir).unwrap();
    let db_path = dir.join("test.db");

    let cameras = vec![make_pinhole_camera(640, 480, 500.0, 500.0, 320.0, 240.0)];
    let image_names = vec![
        "img_001.jpg".to_string(),
        "img_002.jpg".to_string(),
        "img_003.jpg".to_string(),
    ];
    let camera_indexes = vec![0u32, 0, 0];
    let quaternions_wxyz = vec![
        [1.0, 0.0, 0.0, 0.0],
        [0.9, 0.1, 0.0, 0.0],
        [0.8, 0.2, 0.0, 0.0],
    ];
    let translations_xyz = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
    let keypoints_per_image = vec![vec![], vec![], vec![]];
    let descriptors_per_image = vec![vec![], vec![], vec![]];

    // Create a fundamental matrix (identity-like for testing)
    let f_matrix = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

    let tvgs = vec![
        TwoViewGeometry {
            image_idx1: 0,
            image_idx2: 1,
            matches: vec![],
            config: TwoViewGeometryConfig::Calibrated,
            f_matrix: Some(f_matrix),
            e_matrix: None,
            h_matrix: None,
            qvec_wxyz: Some([1.0, 0.0, 0.0, 0.0]),
            tvec: Some([1.0, 0.0, 0.0]),
        },
        TwoViewGeometry {
            image_idx1: 1,
            image_idx2: 2,
            matches: vec![],
            config: TwoViewGeometryConfig::Calibrated,
            f_matrix: Some(f_matrix),
            e_matrix: None,
            h_matrix: None,
            qvec_wxyz: Some([0.9, 0.1, 0.0, 0.0]),
            tvec: Some([1.0, 0.0, 0.0]),
        },
    ];

    let data = ColmapDbWriteData {
        cameras: &cameras,
        image_names: &image_names,
        camera_indexes: &camera_indexes,
        quaternions_wxyz: &quaternions_wxyz,
        translations_xyz: &translations_xyz,
        keypoints_per_image: &keypoints_per_image,
        descriptors_per_image: &descriptors_per_image,
        descriptor_dim: 128,
        pose_priors: None,
        two_view_geometries: Some(&tvgs),
        rigs: None,
        frames: None,
    };

    write_colmap_db(&db_path, &data).unwrap();

    // Read back and verify
    let conn = Connection::open(&db_path).unwrap();
    let count: i32 = conn
        .query_row("SELECT COUNT(*) FROM two_view_geometries", [], |row| {
            row.get(0)
        })
        .unwrap();
    assert_eq!(count, 2);

    // Check config values
    let configs: Vec<i32> = {
        let mut stmt = conn
            .prepare("SELECT config FROM two_view_geometries ORDER BY pair_id")
            .unwrap();
        stmt.query_map([], |row| row.get(0))
            .unwrap()
            .map(|r| r.unwrap())
            .collect()
    };
    assert_eq!(configs, vec![2, 2]); // CALIBRATED = 2

    // Check F matrix blob for first pair
    let f_blob: Vec<u8> = {
        let mut stmt = conn
            .prepare("SELECT F FROM two_view_geometries ORDER BY pair_id LIMIT 1")
            .unwrap();
        stmt.query_row([], |row| row.get(0)).unwrap()
    };
    assert_eq!(f_blob.len(), 9 * 8); // 3x3 matrix of f64
    let f_values: Vec<f64> = f_blob
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect();
    // Should be identity
    assert!((f_values[0] - 1.0).abs() < 1e-10);
    assert!((f_values[4] - 1.0).abs() < 1e-10);
    assert!((f_values[8] - 1.0).abs() < 1e-10);

    // Check qvec blob
    let qvec_blob: Vec<u8> = {
        let mut stmt = conn
            .prepare("SELECT qvec FROM two_view_geometries ORDER BY pair_id LIMIT 1")
            .unwrap();
        stmt.query_row([], |row| row.get(0)).unwrap()
    };
    let qvec: Vec<f64> = qvec_blob
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert!((qvec[0] - 1.0).abs() < 1e-10); // w
    assert!((qvec[1]).abs() < 1e-10); // x
    assert!((qvec[2]).abs() < 1e-10); // y
    assert!((qvec[3]).abs() < 1e-10); // z

    drop(conn);
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn test_pair_id_encoding() {
    // Verify pair_id encoding matches COLMAP's formula
    let dir = std::env::temp_dir().join("colmap_db_pair_id");
    std::fs::create_dir_all(&dir).unwrap();
    let db_path = dir.join("test.db");

    let cameras = vec![make_pinhole_camera(640, 480, 500.0, 500.0, 320.0, 240.0)];
    let image_names = vec!["a.jpg".to_string(), "b.jpg".to_string()];
    let camera_indexes = vec![0u32, 0];
    let quaternions_wxyz = vec![[1.0, 0.0, 0.0, 0.0]; 2];
    let translations_xyz = vec![[0.0, 0.0, 0.0]; 2];
    let keypoints_per_image = vec![vec![]; 2];
    let descriptors_per_image = vec![vec![]; 2];

    let tvgs = vec![TwoViewGeometry {
        image_idx1: 0,
        image_idx2: 1,
        matches: vec![],
        config: TwoViewGeometryConfig::Calibrated,
        f_matrix: None,
        e_matrix: None,
        h_matrix: None,
        qvec_wxyz: None,
        tvec: None,
    }];

    let data = ColmapDbWriteData {
        cameras: &cameras,
        image_names: &image_names,
        camera_indexes: &camera_indexes,
        quaternions_wxyz: &quaternions_wxyz,
        translations_xyz: &translations_xyz,
        keypoints_per_image: &keypoints_per_image,
        descriptors_per_image: &descriptors_per_image,
        descriptor_dim: 128,
        pose_priors: None,
        two_view_geometries: Some(&tvgs),
        rigs: None,
        frames: None,
    };

    let image_ids = write_colmap_db(&db_path, &data).unwrap();

    // DB assigns image_id=1 and image_id=2
    assert_eq!(image_ids[0], 1);
    assert_eq!(image_ids[1], 2);

    // Expected pair_id: kMaxNumImages * 1 + 2 = 2147483647 + 2 = 2147483649
    let conn = Connection::open(&db_path).unwrap();
    let pair_id: i64 = conn
        .query_row("SELECT pair_id FROM two_view_geometries", [], |row| {
            row.get(0)
        })
        .unwrap();
    assert_eq!(pair_id, 2_147_483_647i64 + 2);

    drop(conn);
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn test_multiple_cameras() {
    let dir = std::env::temp_dir().join("colmap_db_multi_cam");
    std::fs::create_dir_all(&dir).unwrap();
    let db_path = dir.join("test.db");

    let cameras = vec![
        make_pinhole_camera(1920, 1080, 1000.0, 1000.0, 960.0, 540.0),
        SfmrCamera {
            model: "SIMPLE_RADIAL".into(),
            width: 640,
            height: 480,
            parameters: [
                ("focal_length".into(), 500.0),
                ("principal_point_x".into(), 320.0),
                ("principal_point_y".into(), 240.0),
                ("radial_distortion_k1".into(), 0.01),
            ]
            .into_iter()
            .collect(),
        },
    ];
    let image_names = vec![
        "cam1_001.jpg".to_string(),
        "cam2_001.jpg".to_string(),
        "cam1_002.jpg".to_string(),
    ];
    let camera_indexes = vec![0u32, 1, 0]; // First and third use camera 0, second uses camera 1
    let quaternions_wxyz = vec![[1.0, 0.0, 0.0, 0.0]; 3];
    let translations_xyz = vec![[0.0, 0.0, 0.0]; 3];
    let keypoints_per_image = vec![vec![]; 3];
    let descriptors_per_image = vec![vec![]; 3];

    let data = ColmapDbWriteData {
        cameras: &cameras,
        image_names: &image_names,
        camera_indexes: &camera_indexes,
        quaternions_wxyz: &quaternions_wxyz,
        translations_xyz: &translations_xyz,
        keypoints_per_image: &keypoints_per_image,
        descriptors_per_image: &descriptors_per_image,
        descriptor_dim: 128,
        pose_priors: None,
        two_view_geometries: None,
        rigs: None,
        frames: None,
    };

    write_colmap_db(&db_path, &data).unwrap();

    // Verify
    let conn = Connection::open(&db_path).unwrap();

    let cam_count: i32 = conn
        .query_row("SELECT COUNT(*) FROM cameras", [], |row| row.get(0))
        .unwrap();
    assert_eq!(cam_count, 2);

    let img_count: i32 = conn
        .query_row("SELECT COUNT(*) FROM images", [], |row| row.get(0))
        .unwrap();
    assert_eq!(img_count, 3);

    // Check camera model IDs
    let models: Vec<i32> = {
        let mut stmt = conn
            .prepare("SELECT model FROM cameras ORDER BY camera_id")
            .unwrap();
        stmt.query_map([], |row| row.get(0))
            .unwrap()
            .map(|r| r.unwrap())
            .collect()
    };
    assert_eq!(models, vec![1, 2]); // PINHOLE=1, SIMPLE_RADIAL=2

    // Check image-camera assignments
    let assignments: Vec<(String, i64)> = {
        let mut stmt = conn
            .prepare("SELECT name, camera_id FROM images ORDER BY image_id")
            .unwrap();
        stmt.query_map([], |row| Ok((row.get(0)?, row.get(1)?)))
            .unwrap()
            .map(|r| r.unwrap())
            .collect()
    };
    assert_eq!(assignments[0], ("cam1_001.jpg".to_string(), 1));
    assert_eq!(assignments[1], ("cam2_001.jpg".to_string(), 2));
    assert_eq!(assignments[2], ("cam1_002.jpg".to_string(), 1));

    drop(conn);
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn test_overwrite_existing_db() {
    let dir = std::env::temp_dir().join("colmap_db_overwrite");
    std::fs::create_dir_all(&dir).unwrap();
    let db_path = dir.join("test.db");

    let cameras = vec![make_pinhole_camera(640, 480, 500.0, 500.0, 320.0, 240.0)];
    let image_names = vec!["img.jpg".to_string()];
    let camera_indexes = vec![0u32];
    let quaternions_wxyz = vec![[1.0, 0.0, 0.0, 0.0]];
    let translations_xyz = vec![[0.0, 0.0, 0.0]];
    let keypoints_per_image = vec![vec![]];
    let descriptors_per_image = vec![vec![]];

    let data = ColmapDbWriteData {
        cameras: &cameras,
        image_names: &image_names,
        camera_indexes: &camera_indexes,
        quaternions_wxyz: &quaternions_wxyz,
        translations_xyz: &translations_xyz,
        keypoints_per_image: &keypoints_per_image,
        descriptors_per_image: &descriptors_per_image,
        descriptor_dim: 128,
        pose_priors: None,
        two_view_geometries: None,
        rigs: None,
        frames: None,
    };

    // Write twice — second should overwrite cleanly
    write_colmap_db(&db_path, &data).unwrap();
    write_colmap_db(&db_path, &data).unwrap();

    let conn = Connection::open(&db_path).unwrap();
    let count: i32 = conn
        .query_row("SELECT COUNT(*) FROM images", [], |row| row.get(0))
        .unwrap();
    assert_eq!(count, 1);

    drop(conn);
    std::fs::remove_dir_all(&dir).unwrap();
}

// ── Matches-format interop tests ────────────────────────────────────────

use matches_format::{
    MatchesContentHash, MatchesData, MatchesMetadata, TvgMetadata, TwoViewGeometryData,
    WorkspaceContents as MatchesWorkspaceContents, WorkspaceMetadata as MatchesWorkspaceMetadata,
};
use ndarray::{Array1, Array2, Array3};
use std::collections::HashMap;

use super::read::read_colmap_db_matches;
use super::write::{write_colmap_db_features, write_colmap_db_matches};
use super::ColmapDbFeatureData;

/// Create feature data for 3 images with some keypoints.
fn make_feature_data() -> (
    Vec<SfmrCamera>,
    Vec<String>,
    Vec<u32>,
    Vec<Vec<[f64; 2]>>,
    Vec<Vec<u8>>,
) {
    let cameras = vec![make_pinhole_camera(640, 480, 500.0, 500.0, 320.0, 240.0)];
    let image_names = vec![
        "frame_000.jpg".to_string(),
        "frame_001.jpg".to_string(),
        "frame_002.jpg".to_string(),
    ];
    let camera_indexes = vec![0u32, 0, 0];
    let keypoints_per_image = vec![
        vec![[10.0, 20.0], [30.0, 40.0], [50.0, 60.0], [70.0, 80.0]],
        vec![[15.0, 25.0], [35.0, 45.0], [55.0, 65.0], [75.0, 85.0]],
        vec![[12.0, 22.0], [32.0, 42.0], [52.0, 62.0], [72.0, 82.0]],
    ];
    let descriptors_per_image = vec![vec![0u8; 4 * 128]; 3];
    (
        cameras,
        image_names,
        camera_indexes,
        keypoints_per_image,
        descriptors_per_image,
    )
}

/// Create a MatchesData with 2 pairs, 5 total matches.
fn make_matches_data() -> MatchesData {
    MatchesData {
        metadata: MatchesMetadata {
            version: 1,
            matching_method: "sequential".into(),
            matching_tool: "colmap".into(),
            matching_tool_version: "4.02".into(),
            matching_options: HashMap::new(),
            workspace: MatchesWorkspaceMetadata {
                absolute_path: "/tmp/ws".into(),
                relative_path: "..".into(),
                contents: MatchesWorkspaceContents {
                    feature_tool: "colmap".into(),
                    feature_type: "sift".into(),
                    feature_options: serde_json::Value::Object(Default::default()),
                    feature_prefix_dir: "features/sift".into(),
                },
            },
            timestamp: "2026-03-29T10:00:00Z".into(),
            image_count: 3,
            image_pair_count: 2,
            match_count: 5,
            has_two_view_geometries: false,
        },
        content_hash: MatchesContentHash {
            metadata_xxh128: String::new(),
            images_xxh128: String::new(),
            image_pairs_xxh128: String::new(),
            two_view_geometries_xxh128: None,
            content_xxh128: String::new(),
        },
        image_names: vec![
            "frame_000.jpg".into(),
            "frame_001.jpg".into(),
            "frame_002.jpg".into(),
        ],
        feature_tool_hashes: vec![[0u8; 16]; 3],
        sift_content_hashes: vec![[1u8; 16]; 3],
        feature_counts: Array1::from_vec(vec![4, 4, 4]),
        image_index_pairs: Array2::from_shape_vec((2, 2), vec![0, 1, 0, 2]).unwrap(),
        match_counts: Array1::from_vec(vec![3, 2]),
        match_feature_indexes: Array2::from_shape_vec((5, 2), vec![0, 0, 1, 1, 2, 3, 0, 0, 3, 2])
            .unwrap(),
        match_descriptor_distances: Array1::from_vec(vec![100.0, 120.0, 90.0, 200.0, 180.0]),
        two_view_geometries: None,
    }
}

fn make_matches_data_with_tvg() -> MatchesData {
    let mut data = make_matches_data();
    data.metadata.has_two_view_geometries = true;

    data.two_view_geometries = Some(TwoViewGeometryData {
        metadata: TvgMetadata {
            image_pair_count: 2,
            inlier_count: 3,
            verification_tool: "colmap".into(),
            verification_options: HashMap::new(),
        },
        config_types: vec![
            matches_format::TwoViewGeometryConfig::Calibrated,
            matches_format::TwoViewGeometryConfig::Degenerate,
        ],
        config_indexes: Array1::from_vec(vec![0, 1]),
        inlier_counts: Array1::from_vec(vec![2, 1]),
        inlier_feature_indexes: Array2::from_shape_vec((3, 2), vec![0, 0, 1, 1, 0, 0]).unwrap(),
        f_matrices: {
            let mut f = Array3::zeros((2, 3, 3));
            f[[0, 0, 1]] = 0.5;
            f[[0, 1, 0]] = -0.5;
            f
        },
        e_matrices: {
            let mut e = Array3::zeros((2, 3, 3));
            e[[0, 0, 2]] = 0.1;
            e[[0, 2, 0]] = -0.1;
            e
        },
        h_matrices: Array3::zeros((2, 3, 3)),
        quaternions_wxyz: {
            let mut q = Array2::zeros((2, 4));
            q[[0, 0]] = 0.9;
            q[[0, 1]] = 0.1;
            q[[0, 2]] = 0.2;
            q[[0, 3]] = 0.3;
            q[[1, 0]] = 1.0; // identity
            q
        },
        translations_xyz: {
            let mut t = Array2::zeros((2, 3));
            t[[0, 0]] = 1.0;
            t[[0, 1]] = 0.5;
            t
        },
    });

    data
}

#[test]
fn test_feature_then_matches_round_trip() {
    let dir = std::env::temp_dir().join("colmap_db_feat_match_rt");
    std::fs::create_dir_all(&dir).unwrap();
    let db_path = dir.join("test.db");

    let (cameras, image_names, camera_indexes, keypoints, descriptors) = make_feature_data();
    let feature_data = ColmapDbFeatureData {
        cameras: &cameras,
        image_names: &image_names,
        camera_indexes: &camera_indexes,
        keypoints_per_image: &keypoints,
        descriptors_per_image: &descriptors,
        descriptor_dim: 128,
        pose_priors: None,
        rigs: None,
        frames: None,
    };

    let id_map = write_colmap_db_features(&db_path, &feature_data).unwrap();
    assert_eq!(id_map.index_to_db_id.len(), 3);

    let matches_data = make_matches_data();
    write_colmap_db_matches(&db_path, &matches_data, &id_map).unwrap();

    let loaded = read_colmap_db_matches(&db_path, false).unwrap();
    assert_eq!(loaded.metadata.image_count, 3);
    assert_eq!(loaded.metadata.image_pair_count, 2);
    assert_eq!(loaded.metadata.match_count, 5);
    assert!(loaded.two_view_geometries.is_none());
    assert_eq!(loaded.image_names, image_names);
    assert_eq!(loaded.feature_counts, Array1::from_vec(vec![4, 4, 4]));
    assert_eq!(loaded.image_index_pairs, matches_data.image_index_pairs);
    assert_eq!(loaded.match_counts, matches_data.match_counts);
    assert_eq!(
        loaded.match_feature_indexes,
        matches_data.match_feature_indexes
    );
    assert!(loaded.match_descriptor_distances.iter().all(|&d| d == 0.0));

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn test_feature_then_matches_with_tvg_round_trip() {
    let dir = std::env::temp_dir().join("colmap_db_feat_match_tvg_rt");
    std::fs::create_dir_all(&dir).unwrap();
    let db_path = dir.join("test.db");

    let (cameras, image_names, camera_indexes, keypoints, descriptors) = make_feature_data();
    let feature_data = ColmapDbFeatureData {
        cameras: &cameras,
        image_names: &image_names,
        camera_indexes: &camera_indexes,
        keypoints_per_image: &keypoints,
        descriptors_per_image: &descriptors,
        descriptor_dim: 128,
        pose_priors: None,
        rigs: None,
        frames: None,
    };

    let id_map = write_colmap_db_features(&db_path, &feature_data).unwrap();
    let matches_data = make_matches_data_with_tvg();
    write_colmap_db_matches(&db_path, &matches_data, &id_map).unwrap();

    let loaded = read_colmap_db_matches(&db_path, true).unwrap();
    assert!(loaded.metadata.has_two_view_geometries);

    let tvg = loaded.two_view_geometries.as_ref().unwrap();
    let orig_tvg = matches_data.two_view_geometries.as_ref().unwrap();

    assert_eq!(tvg.metadata.inlier_count, 3);
    assert_eq!(tvg.inlier_counts, orig_tvg.inlier_counts);
    assert_eq!(tvg.inlier_feature_indexes, orig_tvg.inlier_feature_indexes);

    // Config round-trip
    let pair0_config = tvg.config_types[tvg.config_indexes[0] as usize];
    let pair1_config = tvg.config_types[tvg.config_indexes[1] as usize];
    assert_eq!(
        pair0_config,
        matches_format::TwoViewGeometryConfig::Calibrated
    );
    assert_eq!(
        pair1_config,
        matches_format::TwoViewGeometryConfig::Degenerate
    );

    // Matrix round-trips
    assert!((tvg.f_matrices[[0, 0, 1]] - 0.5).abs() < 1e-10);
    assert!((tvg.f_matrices[[0, 1, 0]] - (-0.5)).abs() < 1e-10);
    assert!((tvg.e_matrices[[0, 0, 2]] - 0.1).abs() < 1e-10);
    assert!(tvg.h_matrices[[0, 0, 0]].abs() < 1e-10);

    // Quaternion: pair 0 non-identity, pair 1 identity→NULL→identity
    assert!((tvg.quaternions_wxyz[[0, 0]] - 0.9).abs() < 1e-10);
    assert!((tvg.quaternions_wxyz[[1, 0]] - 1.0).abs() < 1e-10);
    assert!(tvg.quaternions_wxyz[[1, 1]].abs() < 1e-10);

    // Translation: pair 0 non-zero, pair 1 zero→NULL→zero
    assert!((tvg.translations_xyz[[0, 0]] - 1.0).abs() < 1e-10);
    assert!(tvg.translations_xyz[[1, 0]].abs() < 1e-10);

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn test_read_with_include_tvg_false() {
    let dir = std::env::temp_dir().join("colmap_db_no_tvg_read");
    std::fs::create_dir_all(&dir).unwrap();
    let db_path = dir.join("test.db");

    let (cameras, image_names, camera_indexes, keypoints, descriptors) = make_feature_data();
    let feature_data = ColmapDbFeatureData {
        cameras: &cameras,
        image_names: &image_names,
        camera_indexes: &camera_indexes,
        keypoints_per_image: &keypoints,
        descriptors_per_image: &descriptors,
        descriptor_dim: 128,
        pose_priors: None,
        rigs: None,
        frames: None,
    };

    let id_map = write_colmap_db_features(&db_path, &feature_data).unwrap();
    let matches_data = make_matches_data_with_tvg();
    write_colmap_db_matches(&db_path, &matches_data, &id_map).unwrap();

    let loaded = read_colmap_db_matches(&db_path, false).unwrap();
    assert!(!loaded.metadata.has_two_view_geometries);
    assert!(loaded.two_view_geometries.is_none());
    assert_eq!(loaded.metadata.match_count, 5);

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn test_empty_matches_from_db() {
    let dir = std::env::temp_dir().join("colmap_db_empty_matches");
    std::fs::create_dir_all(&dir).unwrap();
    let db_path = dir.join("test.db");

    let (cameras, image_names, camera_indexes, keypoints, descriptors) = make_feature_data();
    let feature_data = ColmapDbFeatureData {
        cameras: &cameras,
        image_names: &image_names,
        camera_indexes: &camera_indexes,
        keypoints_per_image: &keypoints,
        descriptors_per_image: &descriptors,
        descriptor_dim: 128,
        pose_priors: None,
        rigs: None,
        frames: None,
    };

    write_colmap_db_features(&db_path, &feature_data).unwrap();

    let loaded = read_colmap_db_matches(&db_path, true).unwrap();
    assert_eq!(loaded.metadata.image_count, 3);
    assert_eq!(loaded.metadata.image_pair_count, 0);
    assert_eq!(loaded.metadata.match_count, 0);
    assert!(loaded.two_view_geometries.is_none());

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn test_db_contents_verification() {
    let dir = std::env::temp_dir().join("colmap_db_contents_verify");
    std::fs::create_dir_all(&dir).unwrap();
    let db_path = dir.join("test.db");

    let (cameras, image_names, camera_indexes, keypoints, descriptors) = make_feature_data();
    let feature_data = ColmapDbFeatureData {
        cameras: &cameras,
        image_names: &image_names,
        camera_indexes: &camera_indexes,
        keypoints_per_image: &keypoints,
        descriptors_per_image: &descriptors,
        descriptor_dim: 128,
        pose_priors: None,
        rigs: None,
        frames: None,
    };

    let id_map = write_colmap_db_features(&db_path, &feature_data).unwrap();
    let matches_data = make_matches_data();
    write_colmap_db_matches(&db_path, &matches_data, &id_map).unwrap();

    let conn = Connection::open(&db_path).unwrap();
    let match_count: i32 = conn
        .query_row("SELECT COUNT(*) FROM matches", [], |row| row.get(0))
        .unwrap();
    assert_eq!(match_count, 2);

    let (rows, cols): (i32, i32) = conn
        .query_row(
            "SELECT rows, cols FROM matches ORDER BY pair_id LIMIT 1",
            [],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .unwrap();
    assert_eq!(rows, 3);
    assert_eq!(cols, 2);

    drop(conn);
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn test_config_exhaustive_round_trip() {
    use matches_format::TwoViewGeometryConfig as MConfig;

    let all_configs = [
        MConfig::Undefined,
        MConfig::Degenerate,
        MConfig::Calibrated,
        MConfig::Uncalibrated,
        MConfig::Planar,
        MConfig::PlanarOrPanoramic,
        MConfig::Panoramic,
        MConfig::Multiple,
        MConfig::WatermarkClean,
        MConfig::WatermarkBad,
    ];

    for &config in &all_configs {
        let colmap_config: TwoViewGeometryConfig = config.into();
        let colmap_int = colmap_config as i32;
        let back = TwoViewGeometryConfig::from_colmap_int(colmap_int).unwrap();
        let matches_config: MConfig = back.into();
        assert_eq!(
            config, matches_config,
            "Config round-trip failed for {:?} (int={colmap_int})",
            config
        );
    }
}

#[test]
fn test_image_id_map_consistency() {
    let dir = std::env::temp_dir().join("colmap_db_idmap_consistency");
    std::fs::create_dir_all(&dir).unwrap();
    let db_path = dir.join("test.db");

    let (cameras, image_names, camera_indexes, keypoints, descriptors) = make_feature_data();
    let feature_data = ColmapDbFeatureData {
        cameras: &cameras,
        image_names: &image_names,
        camera_indexes: &camera_indexes,
        keypoints_per_image: &keypoints,
        descriptors_per_image: &descriptors,
        descriptor_dim: 128,
        pose_priors: None,
        rigs: None,
        frames: None,
    };

    let id_map = write_colmap_db_features(&db_path, &feature_data).unwrap();

    for (idx, &db_id) in id_map.index_to_db_id.iter().enumerate() {
        assert_eq!(id_map.db_id_to_index[&db_id], idx);
    }

    let conn = Connection::open(&db_path).unwrap();
    for (idx, name) in image_names.iter().enumerate() {
        let db_id: i64 = conn
            .query_row(
                "SELECT image_id FROM images WHERE name = ?1",
                [name],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(db_id, id_map.index_to_db_id[idx]);
    }

    drop(conn);
    std::fs::remove_dir_all(&dir).unwrap();
}

// ── Rig and frame tests ─────────────────────────────────────────────────

use super::types::{DbFrame, DbFrameDataId, DbRig, DbRigSensor, DbSensor, DbSensorType};

#[test]
fn test_write_single_camera_rig_with_frame() {
    let dir = std::env::temp_dir().join("colmap_db_rig_single");
    std::fs::create_dir_all(&dir).unwrap();
    let db_path = dir.join("test.db");

    let cameras = vec![make_pinhole_camera(640, 480, 500.0, 500.0, 320.0, 240.0)];
    let image_names = vec!["frame_001.jpg".to_string()];
    let camera_indexes = vec![0u32];
    let quaternions_wxyz = vec![[1.0, 0.0, 0.0, 0.0]];
    let translations_xyz = vec![[0.0, 0.0, 0.0]];
    let keypoints_per_image = vec![vec![[100.0, 200.0]]];
    let descriptors_per_image = vec![vec![0u8; 128]];

    // Single-camera trivial rig
    let rigs = vec![DbRig {
        ref_sensor: DbSensor {
            sensor_type: DbSensorType::Camera,
            id: 1, // camera_id in DB (1-based)
        },
        sensors: vec![],
    }];

    let frames = vec![DbFrame {
        rig_index: 0,
        data_ids: vec![DbFrameDataId {
            sensor_type: DbSensorType::Camera,
            sensor_id: 1,
            data_id: 0, // 0-based image index
        }],
    }];

    let data = ColmapDbWriteData {
        cameras: &cameras,
        image_names: &image_names,
        camera_indexes: &camera_indexes,
        quaternions_wxyz: &quaternions_wxyz,
        translations_xyz: &translations_xyz,
        keypoints_per_image: &keypoints_per_image,
        descriptors_per_image: &descriptors_per_image,
        descriptor_dim: 128,
        pose_priors: None,
        two_view_geometries: None,
        rigs: Some(&rigs),
        frames: Some(&frames),
    };

    let image_ids = write_colmap_db(&db_path, &data).unwrap();
    assert_eq!(image_ids.len(), 1);

    let conn = Connection::open(&db_path).unwrap();

    // Check rigs table
    let (rig_id, ref_sensor_id, ref_sensor_type): (i64, i64, i32) = conn
        .query_row(
            "SELECT rig_id, ref_sensor_id, ref_sensor_type FROM rigs",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .unwrap();
    assert_eq!(rig_id, 1);
    assert_eq!(ref_sensor_id, 1);
    assert_eq!(ref_sensor_type, 0); // Camera

    // Check no rig_sensors (single-camera rig has no non-ref sensors)
    let sensor_count: i32 = conn
        .query_row("SELECT COUNT(*) FROM rig_sensors", [], |row| row.get(0))
        .unwrap();
    assert_eq!(sensor_count, 0);

    // Check frames table
    let (frame_id, frame_rig_id): (i64, i64) = conn
        .query_row("SELECT frame_id, rig_id FROM frames", [], |row| {
            Ok((row.get(0)?, row.get(1)?))
        })
        .unwrap();
    assert_eq!(frame_id, 1);
    assert_eq!(frame_rig_id, 1);

    // Check frame_data
    let (fd_frame_id, fd_data_id, fd_sensor_id, fd_sensor_type): (i64, i64, i64, i32) = conn
        .query_row(
            "SELECT frame_id, data_id, sensor_id, sensor_type FROM frame_data",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
        )
        .unwrap();
    assert_eq!(fd_frame_id, 1);
    assert_eq!(fd_data_id, image_ids[0]); // DB image_id
    assert_eq!(fd_sensor_id, 1);
    assert_eq!(fd_sensor_type, 0);

    drop(conn);
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn test_write_multi_sensor_rig() {
    let dir = std::env::temp_dir().join("colmap_db_rig_multi");
    std::fs::create_dir_all(&dir).unwrap();
    let db_path = dir.join("test.db");

    // Two cameras: left (ref) and right (non-ref)
    let cameras = vec![
        make_pinhole_camera(640, 480, 500.0, 500.0, 320.0, 240.0),
        make_pinhole_camera(640, 480, 500.0, 500.0, 320.0, 240.0),
    ];
    let image_names = vec![
        "left_001.jpg".to_string(),
        "right_001.jpg".to_string(),
        "left_002.jpg".to_string(),
        "right_002.jpg".to_string(),
    ];
    let camera_indexes = vec![0, 1, 0, 1];
    let quaternions_wxyz = vec![[1.0, 0.0, 0.0, 0.0]; 4];
    let translations_xyz = vec![[0.0, 0.0, 0.0]; 4];
    let keypoints_per_image = vec![vec![]; 4];
    let descriptors_per_image = vec![vec![]; 4];

    // Stereo rig: camera 1 is ref, camera 2 is offset by 10cm in x
    let rigs = vec![DbRig {
        ref_sensor: DbSensor {
            sensor_type: DbSensorType::Camera,
            id: 1,
        },
        sensors: vec![DbRigSensor {
            sensor: DbSensor {
                sensor_type: DbSensorType::Camera,
                id: 2,
            },
            sensor_from_rig: Some(([1.0, 0.0, 0.0, 0.0], [0.1, 0.0, 0.0])),
        }],
    }];

    // Two frames, each with both sensors
    let frames = vec![
        DbFrame {
            rig_index: 0,
            data_ids: vec![
                DbFrameDataId {
                    sensor_type: DbSensorType::Camera,
                    sensor_id: 1,
                    data_id: 0,
                },
                DbFrameDataId {
                    sensor_type: DbSensorType::Camera,
                    sensor_id: 2,
                    data_id: 1,
                },
            ],
        },
        DbFrame {
            rig_index: 0,
            data_ids: vec![
                DbFrameDataId {
                    sensor_type: DbSensorType::Camera,
                    sensor_id: 1,
                    data_id: 2,
                },
                DbFrameDataId {
                    sensor_type: DbSensorType::Camera,
                    sensor_id: 2,
                    data_id: 3,
                },
            ],
        },
    ];

    let data = ColmapDbWriteData {
        cameras: &cameras,
        image_names: &image_names,
        camera_indexes: &camera_indexes,
        quaternions_wxyz: &quaternions_wxyz,
        translations_xyz: &translations_xyz,
        keypoints_per_image: &keypoints_per_image,
        descriptors_per_image: &descriptors_per_image,
        descriptor_dim: 128,
        pose_priors: None,
        two_view_geometries: None,
        rigs: Some(&rigs),
        frames: Some(&frames),
    };

    let image_ids = write_colmap_db(&db_path, &data).unwrap();
    assert_eq!(image_ids.len(), 4);

    let conn = Connection::open(&db_path).unwrap();

    // Check rigs
    let rig_count: i32 = conn
        .query_row("SELECT COUNT(*) FROM rigs", [], |row| row.get(0))
        .unwrap();
    assert_eq!(rig_count, 1);

    // Check rig_sensors (1 non-ref sensor)
    let sensor_count: i32 = conn
        .query_row("SELECT COUNT(*) FROM rig_sensors", [], |row| row.get(0))
        .unwrap();
    assert_eq!(sensor_count, 1);

    // Check sensor_from_rig blob
    let pose_blob: Vec<u8> = conn
        .query_row("SELECT sensor_from_rig FROM rig_sensors", [], |row| {
            row.get(0)
        })
        .unwrap();
    assert_eq!(pose_blob.len(), 7 * 8); // qw,qx,qy,qz,tx,ty,tz as f64
    let values: Vec<f64> = pose_blob
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert!((values[0] - 1.0).abs() < 1e-10); // qw
    assert!((values[4] - 0.1).abs() < 1e-10); // tx

    // Check frames
    let frame_count: i32 = conn
        .query_row("SELECT COUNT(*) FROM frames", [], |row| row.get(0))
        .unwrap();
    assert_eq!(frame_count, 2);

    // Check frame_data
    let fd_count: i32 = conn
        .query_row("SELECT COUNT(*) FROM frame_data", [], |row| row.get(0))
        .unwrap();
    assert_eq!(fd_count, 4); // 2 frames x 2 sensors

    // Verify frame 1 has the correct image IDs
    let frame1_data_ids: Vec<i64> = {
        let mut stmt = conn
            .prepare("SELECT data_id FROM frame_data WHERE frame_id = 1 ORDER BY sensor_id")
            .unwrap();
        stmt.query_map([], |row| row.get(0))
            .unwrap()
            .map(|r| r.unwrap())
            .collect()
    };
    assert_eq!(frame1_data_ids, vec![image_ids[0], image_ids[1]]);

    drop(conn);
    std::fs::remove_dir_all(&dir).unwrap();
}
