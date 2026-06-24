use crate::*;
use ndarray::{Array1, Array2, Array4};
use std::collections::HashMap;

/// Create minimal valid SfmrData for testing.
fn make_test_data() -> SfmrData {
    let image_count = 3;
    let point_count = 5;
    let observation_count = 8;
    let num_buckets = 128;

    // Build track arrays sorted by (point_indexes, image_indexes)
    // Point 0: observed by images 0, 1 (2 obs)
    // Point 1: observed by images 0, 1, 2 (3 obs)
    // Point 2: observed by image 2 (1 obs)
    // Point 3: observed by image 1 (1 obs)
    // Point 4: observed by image 0 (1 obs)
    let point_indexes = Array1::from_vec(vec![0, 0, 1, 1, 1, 2, 3, 4]);
    let image_indexes = Array1::from_vec(vec![0, 1, 0, 1, 2, 2, 1, 0]);
    let feature_indexes = Array1::from_vec(vec![0, 0, 1, 1, 0, 1, 2, 2]);
    let observation_counts = Array1::from_vec(vec![2, 3, 1, 1, 1]);

    SfmrData {
        workspace_dir: None,
        metadata: SfmrMetadata {
            version: 2,
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
            point_count: point_count as u32,
            infinity_point_count: 0,
            observation_count: observation_count as u32,
            camera_count: 1,
            rig_count: None,
            sensor_count: None,
            frame_count: None,
            world_space_unit: None,
            feature_source: FEATURE_SOURCE_SIFT_FILES.to_string(),
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
        quaternions_wxyz: {
            const FRAC_1_SQRT_2: f64 = std::f64::consts::FRAC_1_SQRT_2;
            Array2::from_shape_vec(
                (image_count, 4),
                vec![
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.9239,
                    0.0,
                    0.3827,
                    0.0,
                    FRAC_1_SQRT_2,
                    0.0,
                    FRAC_1_SQRT_2,
                    0.0,
                ],
            )
            .unwrap()
        },
        translations_xyz: Array2::zeros((image_count, 3)),
        feature_tool_hashes: Some(vec![[0u8; 16]; image_count]),
        sift_content_hashes: Some(vec![[1u8; 16]; image_count]),
        image_file_hashes: None,
        thumbnails_y_x_rgb: Array4::zeros((image_count, 128, 128, 3)),
        positions_xyzw: Array2::from_shape_vec(
            (point_count, 4),
            vec![
                0.0, 0.0, 5.0, 1.0, // finite point, w=1
                1.0, 0.0, 6.0, 1.0, //
                -1.0, 1.0, 4.0, 1.0, //
                0.5, -0.5, 7.0, 1.0, //
                -0.5, 0.5, 3.0, 1.0, //
            ],
        )
        .unwrap(),
        colors_rgb: Array2::from_shape_vec(
            (point_count, 3),
            vec![255, 0, 0, 0, 255, 0, 0, 0, 255, 128, 128, 0, 0, 128, 128],
        )
        .unwrap(),
        reprojection_errors: Array1::from_vec(vec![0.5, 0.6, 0.7, 0.8, 0.4]),
        normals_xyz: Some(
            Array2::from_shape_vec(
                (point_count, 3),
                vec![
                    0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                ],
            )
            .unwrap(),
        ),
        patch_u_halfvec_xyz: None,
        patch_v_halfvec_xyz: None,
        patch_bitmaps_y_x_rgba: None,
        image_indexes,
        feature_indexes: Some(feature_indexes),
        keypoints_xy: None,
        point_indexes,
        observation_counts,
        depth_statistics: DepthStatistics {
            num_histogram_buckets: num_buckets as u32,
            images: vec![
                ImageDepthStats {
                    histogram_min_z: Some(3.0),
                    histogram_max_z: Some(7.0),
                    observed: ObservedDepthStats {
                        count: 3,
                        infinity_count: 0,
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
                        infinity_count: 0,
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
                        infinity_count: 0,
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
    assert_eq!(loaded.metadata.point_count, 5);
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
    assert_eq!(loaded.positions_xyzw, data.positions_xyzw);
    assert_eq!(loaded.colors_rgb, data.colors_rgb);
    assert_eq!(loaded.reprojection_errors, data.reprojection_errors);
    assert_eq!(loaded.normals_xyz, data.normals_xyz);

    // Verify tracks
    assert_eq!(loaded.image_indexes, data.image_indexes);
    assert_eq!(loaded.feature_indexes, data.feature_indexes);
    assert_eq!(loaded.point_indexes, data.point_indexes);
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

/// Convert the SIFT-referenced test fixture into an `embedded_patches` one:
/// drop the `.sift`-link arrays, add inline keypoints and a direct image hash.
fn make_embedded_test_data() -> SfmrData {
    let mut data = make_test_data();
    let m = data.metadata.observation_count as usize;
    let n = data.metadata.image_count as usize;
    let p = data.metadata.point_count as usize;
    data.metadata.feature_source = FEATURE_SOURCE_EMBEDDED_PATCHES.to_string();
    data.feature_tool_hashes = None;
    data.sift_content_hashes = None;
    data.image_file_hashes = Some(vec![[2u8; 16]; n]);
    data.feature_indexes = None;
    // Sub-pixel keypoints inside the 1920x1080 camera, one per observation.
    let kp: Vec<f32> = (0..m)
        .flat_map(|i| [100.5 + i as f32, 200.25 + i as f32])
        .collect();
    data.keypoints_xy = Some(Array2::from_shape_vec((m, 2), kp).unwrap());
    // An embedded_patches file requires the per-point patch frame (has_uv_frames).
    let u: Vec<f32> = (0..p).flat_map(|_| [0.5f32, 0.0, 0.0]).collect();
    let v: Vec<f32> = (0..p).flat_map(|_| [0.0f32, 0.5, 0.0]).collect();
    data.patch_u_halfvec_xyz = Some(Array2::from_shape_vec((p, 3), u).unwrap());
    data.patch_v_halfvec_xyz = Some(Array2::from_shape_vec((p, 3), v).unwrap());
    data
}

#[test]
fn test_embedded_patches_round_trip() {
    let mut data = make_embedded_test_data();
    let dir = std::env::temp_dir().join("sfmr_test_embedded_round_trip");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("test.sfmr");

    let options = WriteOptions {
        skip_recompute_depth_stats: true,
        ..Default::default()
    };
    write_sfmr_with_options(&path, &mut data, &options).unwrap();

    // Written as version 4 with the embedded_patches source.
    let loaded = read_sfmr(&path).unwrap();
    assert_eq!(loaded.metadata.version, 4);
    assert_eq!(
        loaded.metadata.feature_source,
        FEATURE_SOURCE_EMBEDDED_PATCHES
    );

    // The inline keypoints and direct image hash round-trip; the .sift-link
    // arrays are absent.
    assert_eq!(loaded.keypoints_xy, data.keypoints_xy);
    assert_eq!(loaded.image_file_hashes, data.image_file_hashes);
    assert!(loaded.feature_indexes.is_none());
    assert!(loaded.feature_tool_hashes.is_none());
    assert!(loaded.sift_content_hashes.is_none());

    // The shared columns are unchanged.
    assert_eq!(loaded.image_indexes, data.image_indexes);
    assert_eq!(loaded.point_indexes, data.point_indexes);
    assert_eq!(loaded.observation_counts, data.observation_counts);

    // The required patch frame round-trips.
    assert_eq!(loaded.patch_u_halfvec_xyz, data.patch_u_halfvec_xyz);
    assert_eq!(loaded.patch_v_halfvec_xyz, data.patch_v_halfvec_xyz);

    // Integrity verification passes for the embedded-patches layout.
    let (valid, errors) = verify_sfmr(&path).unwrap();
    assert!(valid, "verification failed: {errors:?}");

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn test_embedded_patches_requires_patch_frame() {
    // An embedded_patches file without the per-point patch frame is rejected
    // (the keypoint anchors a patch that only exists with a u/v frame).
    let mut data = make_embedded_test_data();
    data.patch_u_halfvec_xyz = None;
    data.patch_v_halfvec_xyz = None;
    let dir = std::env::temp_dir().join("sfmr_test_embedded_no_frame");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("test.sfmr");
    let options = WriteOptions {
        skip_recompute_depth_stats: true,
        ..Default::default()
    };
    let err = write_sfmr_with_options(&path, &mut data, &options).unwrap_err();
    assert!(
        err.to_string().contains("patch_u_halfvec_xyz"),
        "expected a patch-frame requirement error, got: {err}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_write_rejects_contradictory_columns() {
    // sift_files must not also carry keypoints_xy.
    let mut d = make_test_data();
    d.keypoints_xy = Some(Array2::zeros((d.metadata.observation_count as usize, 2)));
    let dir = std::env::temp_dir().join("sfmr_test_contradiction1");
    std::fs::create_dir_all(&dir).unwrap();
    let err = write_sfmr(&dir.join("a.sfmr"), &mut d).unwrap_err();
    assert!(
        format!("{err}").contains("must not carry keypoints_xy"),
        "{err}"
    );

    // embedded_patches must not also carry feature_indexes.
    let mut e = make_embedded_test_data();
    e.feature_indexes = Some(Array1::from_vec(vec![
        0u32;
        e.metadata.observation_count as usize
    ]));
    let err = write_sfmr(&dir.join("b.sfmr"), &mut e).unwrap_err();
    assert!(
        format!("{err}").contains("must not carry feature_indexes"),
        "{err}"
    );

    // An unknown feature_source is rejected outright.
    let mut f = make_test_data();
    f.metadata.feature_source = "bogus".into();
    let err = write_sfmr(&dir.join("c.sfmr"), &mut f).unwrap_err();
    assert!(format!("{err}").contains("unknown feature_source"), "{err}");
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn test_embedded_keypoints_validated_on_read_and_verify() {
    let dir = std::env::temp_dir().join("sfmr_test_kp_validate");
    std::fs::create_dir_all(&dir).unwrap();

    // Out-of-bounds keypoint (camera is 1920x1080) is rejected on read+verify.
    let mut oob = make_embedded_test_data();
    oob.keypoints_xy.as_mut().unwrap()[[0, 0]] = 5000.0;
    let path = dir.join("oob.sfmr");
    write_sfmr(&path, &mut oob).unwrap(); // write does not bounds-check
    let err = read_sfmr(&path).err().unwrap();
    assert!(format!("{err}").contains("image bounds"), "{err}");
    let (valid, errors) = verify_sfmr(&path).unwrap();
    assert!(
        !valid && errors.iter().any(|e| e.contains("image bounds")),
        "{errors:?}"
    );

    // A non-finite keypoint is rejected too.
    let mut nan = make_embedded_test_data();
    nan.keypoints_xy.as_mut().unwrap()[[1, 1]] = f32::NAN;
    let path = dir.join("nan.sfmr");
    write_sfmr(&path, &mut nan).unwrap();
    let err = read_sfmr(&path).err().unwrap();
    assert!(format!("{err}").contains("not finite"), "{err}");
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn test_keypoints_are_folded_into_tracks_hash() {
    let dir = std::env::temp_dir().join("sfmr_test_kp_hash");
    std::fs::create_dir_all(&dir).unwrap();

    let mut a = make_embedded_test_data();
    write_sfmr(&dir.join("a.sfmr"), &mut a).unwrap();
    let ha = read_sfmr(&dir.join("a.sfmr"))
        .unwrap()
        .content_hash
        .tracks_xxh128;

    let mut b = make_embedded_test_data();
    b.keypoints_xy.as_mut().unwrap()[[0, 0]] += 1.0; // still in bounds
    write_sfmr(&dir.join("b.sfmr"), &mut b).unwrap();
    let hb = read_sfmr(&dir.join("b.sfmr"))
        .unwrap()
        .content_hash
        .tracks_xxh128;

    assert_ne!(ha, hb, "changing a keypoint must change the tracks hash");
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn test_embedded_sort_reorders_keypoints_in_lockstep() {
    // Deliberately unsorted tracks, each keypoint tagged (image+0.25,
    // point+0.25). After the write-time sort, every row's keypoint must still
    // match its own (point, image) — proving keypoints are permuted with the
    // index arrays.
    let mut d = make_embedded_test_data();
    let pts = [2u32, 0, 4, 1, 0, 3, 1, 1];
    let imgs = [0u32, 0, 2, 0, 1, 1, 1, 2];
    d.point_indexes = Array1::from_vec(pts.to_vec());
    d.image_indexes = Array1::from_vec(imgs.to_vec());
    d.observation_counts = Array1::from_vec(vec![2, 3, 1, 1, 1]); // per point 0..4
    let kp: Vec<f32> = (0..8)
        .flat_map(|j| [imgs[j] as f32 + 0.25, pts[j] as f32 + 0.25])
        .collect();
    d.keypoints_xy = Some(Array2::from_shape_vec((8, 2), kp).unwrap());

    let dir = std::env::temp_dir().join("sfmr_test_kp_reorder");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("u.sfmr");
    write_sfmr(&path, &mut d).unwrap();
    let loaded = read_sfmr(&path).unwrap();

    let kp = loaded.keypoints_xy.unwrap();
    for j in 0..8 {
        let p = loaded.point_indexes[j] as f32 + 0.25;
        let i = loaded.image_indexes[j] as f32 + 0.25;
        assert_eq!(
            (kp[[j, 0]], kp[[j, 1]]),
            (i, p),
            "row {j} keypoint misaligned"
        );
    }
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

#[test]
fn test_unsupported_future_version_rejected() {
    use std::io::{Read, Write};

    let mut data = make_test_data();
    let dir = std::env::temp_dir().join("sfmr_test_future_version");
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("v4.sfmr");
    write_sfmr(&src, &mut data).unwrap();

    // Copy the archive verbatim except for metadata.json.zst, whose version
    // is bumped to a value this build does not understand.
    let dst = dir.join("v5.sfmr");
    let archive_file = std::fs::File::open(&src).unwrap();
    let mut archive = zip::ZipArchive::new(archive_file).unwrap();
    let names: Vec<String> = archive.file_names().map(|s| s.to_string()).collect();
    let out = std::fs::File::create(&dst).unwrap();
    let mut zip = zip::ZipWriter::new(out);
    let stored =
        zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored);
    for name in &names {
        let mut compressed = Vec::new();
        archive
            .by_name(name)
            .unwrap()
            .read_to_end(&mut compressed)
            .unwrap();
        zip.start_file(name, stored).unwrap();
        if name == "metadata.json.zst" {
            let mut json: serde_json::Value =
                serde_json::from_slice(&zstd::stream::decode_all(&compressed[..]).unwrap())
                    .unwrap();
            json.as_object_mut()
                .unwrap()
                .insert("version".into(), serde_json::json!(5));
            let bytes = zstd::bulk::compress(&serde_json::to_vec(&json).unwrap(), 3).unwrap();
            zip.write_all(&bytes).unwrap();
        } else {
            zip.write_all(&compressed).unwrap();
        }
    }
    zip.finish().unwrap();

    let err = read_sfmr(&dst).err().unwrap();
    assert!(
        format!("{err}").contains("unsupported .sfmr format version 5"),
        "{err}"
    );
    let (valid, errors) = verify_sfmr(&dst).unwrap();
    assert!(
        !valid
            && errors
                .iter()
                .any(|e| e.contains("unsupported .sfmr format version 5")),
        "{errors:?}"
    );

    std::fs::remove_dir_all(&dir).unwrap();
}

/// Rewrite a version-2 `.sfmr` file into the version-1 on-disk layout, so
/// the version-1 compatibility path in `read_sfmr` can be exercised.
/// Version 1 stored Euclidean `positions_xyz` `(P, 3)`, named the track
/// point-index array `points3d_indexes`, and used the `points3d_count`
/// metadata key.
fn rewrite_v2_as_v1(v2_path: &std::path::Path, v1_path: &std::path::Path) {
    use std::io::{Read, Write};

    let v2 = std::fs::File::open(v2_path).unwrap();
    let mut archive = zip::ZipArchive::new(v2).unwrap();
    let names: Vec<String> = archive.file_names().map(|s| s.to_string()).collect();

    let out = std::fs::File::create(v1_path).unwrap();
    let mut zip = zip::ZipWriter::new(out);
    let stored =
        zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored);

    let rename_json_key = |compressed: &[u8], set_version: bool| -> Vec<u8> {
        let mut json: serde_json::Value =
            serde_json::from_slice(&zstd::stream::decode_all(compressed).unwrap()).unwrap();
        let obj = json.as_object_mut().unwrap();
        if let Some(pc) = obj.remove("point_count") {
            obj.insert("points3d_count".into(), pc);
        }
        obj.remove("infinity_point_count");
        if set_version {
            obj.insert("version".into(), serde_json::json!(1));
        }
        zstd::bulk::compress(&serde_json::to_vec(&json).unwrap(), 3).unwrap()
    };

    for name in &names {
        let mut compressed = Vec::new();
        archive
            .by_name(name)
            .unwrap()
            .read_to_end(&mut compressed)
            .unwrap();

        if name == "metadata.json.zst" {
            zip.start_file(name, stored).unwrap();
            zip.write_all(&rename_json_key(&compressed, true)).unwrap();
        } else if name == "points3d/metadata.json.zst" {
            zip.start_file(name, stored).unwrap();
            zip.write_all(&rename_json_key(&compressed, false)).unwrap();
        } else if name.starts_with("points3d/positions_xyzw.") {
            // Drop the homogeneous `w` column: keep the first 3 of every
            // 4 little-endian f64 values (24 of every 32 bytes).
            let xyzw = zstd::stream::decode_all(&compressed[..]).unwrap();
            let mut xyz = Vec::with_capacity(xyzw.len() / 4 * 3);
            for row in xyzw.chunks_exact(32) {
                xyz.extend_from_slice(&row[..24]);
            }
            let v1_name = name
                .replace("positions_xyzw", "positions_xyz")
                .replace(".4.float64", ".3.float64");
            zip.start_file(&v1_name, stored).unwrap();
            zip.write_all(&zstd::bulk::compress(&xyz, 3).unwrap())
                .unwrap();
        } else if name.starts_with("tracks/point_indexes.") {
            // Identical bytes under the version-1 name.
            let v1_name = name.replace("point_indexes", "points3d_indexes");
            zip.start_file(&v1_name, stored).unwrap();
            zip.write_all(&compressed).unwrap();
        } else if name.starts_with("points3d/normals_xyz.") {
            // Identical bytes under the version-1/2 normals name.
            let v1_name = name.replace("normals_xyz", "estimated_normals_xyz");
            zip.start_file(&v1_name, stored).unwrap();
            zip.write_all(&compressed).unwrap();
        } else {
            zip.start_file(name, stored).unwrap();
            zip.write_all(&compressed).unwrap();
        }
    }
    zip.finish().unwrap();
}

#[test]
fn test_read_version_1_layout() {
    let mut data = make_test_data();
    let dir = std::env::temp_dir().join("sfmr_test_v1_compat");
    std::fs::create_dir_all(&dir).unwrap();
    let v2_path = dir.join("v2.sfmr");
    let v1_path = dir.join("v1.sfmr");

    write_sfmr(&v2_path, &mut data).unwrap();
    rewrite_v2_as_v1(&v2_path, &v1_path);

    // Metadata-only read must accept the version-1 `points3d_count` key.
    let meta = read_sfmr_metadata(&v1_path).unwrap();
    assert_eq!(meta.version, 1);
    assert_eq!(meta.point_count, 5);

    // Full read upgrades version 1 to the current in-memory model.
    let loaded = read_sfmr(&v1_path).unwrap();
    assert_eq!(loaded.metadata.version, 4);
    // A legacy file upgrades to the sift_files source with its .sift-link
    // columns present and the embedded columns absent.
    assert_eq!(loaded.metadata.feature_source, FEATURE_SOURCE_SIFT_FILES);
    assert!(loaded.feature_indexes.is_some());
    assert!(loaded.feature_tool_hashes.is_some());
    assert!(loaded.keypoints_xy.is_none());
    assert!(loaded.image_file_hashes.is_none());
    assert_eq!(loaded.metadata.point_count, 5);
    assert_eq!(loaded.metadata.infinity_point_count, 0);
    assert_eq!(loaded.positions_xyzw.shape(), &[5, 4]);
    // Every version-1 point is finite: the appended `w` column is 1.
    for i in 0..5 {
        assert_eq!(loaded.positions_xyzw[[i, 3]], 1.0);
    }
    // Euclidean coordinates survive the round trip.
    assert_eq!(loaded.positions_xyzw[[0, 0]], 0.0);
    assert_eq!(loaded.positions_xyzw[[0, 2]], 5.0);
    // The renamed track point-index array is read correctly.
    assert_eq!(loaded.point_indexes, data.point_indexes);

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn test_point_at_infinity_round_trip() {
    let mut data = make_test_data();
    // Turn point 4 (observed only by image 0) into a point at infinity:
    // a unit direction with `w == 0`.
    data.positions_xyzw[[4, 0]] = 0.0;
    data.positions_xyzw[[4, 1]] = 0.0;
    data.positions_xyzw[[4, 2]] = 1.0;
    data.positions_xyzw[[4, 3]] = 0.0;

    let dir = std::env::temp_dir().join("sfmr_test_infinity_round_trip");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("test.sfmr");

    write_sfmr(&path, &mut data).unwrap();

    let (valid, errors) = verify_sfmr(&path).unwrap();
    assert!(valid, "Verification failed: {:?}", errors);

    let loaded = read_sfmr(&path).unwrap();
    assert_eq!(loaded.metadata.infinity_point_count, 1);
    assert_eq!(loaded.positions_xyzw[[4, 3]], 0.0);

    // Depth statistics count the infinity point separately. Image 0 also
    // observes finite points 0 and 1, so its finite `count` stays 2.
    let img0 = &loaded.depth_statistics.images[0];
    assert_eq!(img0.observed.infinity_count, 1);
    assert_eq!(img0.observed.count, 2);

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
            version: 2,
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
            point_count: 0,
            infinity_point_count: 0,
            observation_count: 0,
            camera_count: 0,
            rig_count: None,
            sensor_count: None,
            frame_count: None,
            world_space_unit: None,
            feature_source: FEATURE_SOURCE_SIFT_FILES.to_string(),
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
        feature_tool_hashes: Some(vec![]),
        sift_content_hashes: Some(vec![]),
        image_file_hashes: None,
        thumbnails_y_x_rgb: Array4::zeros((0, 128, 128, 3)),
        positions_xyzw: Array2::zeros((0, 4)),
        colors_rgb: Array2::zeros((0, 3)),
        reprojection_errors: Array1::from_vec(vec![]),
        normals_xyz: Some(Array2::zeros((0, 3))),
        patch_u_halfvec_xyz: None,
        patch_v_halfvec_xyz: None,
        patch_bitmaps_y_x_rgba: None,
        image_indexes: Array1::from_vec(vec![]),
        feature_indexes: Some(Array1::from_vec(vec![])),
        keypoints_xy: None,
        point_indexes: Array1::from_vec(vec![]),
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
    assert_eq!(loaded.metadata.point_count, 0);
    assert_eq!(loaded.metadata.observation_count, 0);
    assert_eq!(loaded.cameras.len(), 0);
    assert_eq!(loaded.image_names.len(), 0);
    assert_eq!(loaded.positions_xyzw.shape(), &[0, 4]);

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
    let normals = loaded.normals_xyz.expect("normals present");
    for i in 0..normals.shape()[0] {
        let nx = normals[[i, 0]] as f64;
        let ny = normals[[i, 1]] as f64;
        let nz = normals[[i, 2]] as f64;
        let norm = (nx * nx + ny * ny + nz * nz).sqrt();
        assert!(
            (norm - 1.0).abs() < 0.01,
            "Normal {i} should be unit vector, got norm = {norm}"
        );
    }

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn test_write_preserves_set_normals_and_fills_missing() {
    // The default write preserves any normal already set and recomputes only
    // the missing (zero) rows from geometry. In make_test_data all cameras
    // sit at the origin (zero translations), so the mean-viewing recompute
    // for a point P points roughly toward -P — distinguishable from a set
    // normal that disagrees with it.
    let mut data = make_test_data();

    // Point 0 is at +z (0, 0, 5); its mean-viewing normal would be (0, 0, -1).
    // Store the opposite, (0, 0, 1): a *set* normal that recompute would flip.
    let set = data.normals_xyz.as_mut().unwrap();
    set[[0, 0]] = 0.0;
    set[[0, 1]] = 0.0;
    set[[0, 2]] = 1.0;
    // Point 1 has no normal yet — the zero vector. It must be filled.
    set[[1, 0]] = 0.0;
    set[[1, 1]] = 0.0;
    set[[1, 2]] = 0.0;

    let dir = std::env::temp_dir().join("sfmr_test_preserve_normals");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("test.sfmr");
    write_sfmr(&path, &mut data).unwrap();
    let loaded = read_sfmr(&path).unwrap();
    let normals = loaded.normals_xyz.expect("normals present");

    // Point 0's set normal is preserved (z = +1), not overwritten with the
    // recomputed mean-viewing normal (which would have z = -1).
    assert!(
        normals[[0, 2]] > 0.5,
        "set normal should be preserved, got {:?}",
        normals.row(0)
    );

    // Point 1's missing normal is filled from the recompute: a non-zero unit
    // vector pointing roughly toward -P = -(1, 0, 6).
    let n1 = normals.row(1);
    let norm1 = (n1[0] * n1[0] + n1[1] * n1[1] + n1[2] * n1[2]).sqrt();
    assert!(
        (norm1 - 1.0).abs() < 0.01,
        "missing normal should be filled"
    );
    assert!(
        n1[0] < 0.0 && n1[2] < 0.0,
        "filled normal should point toward -P"
    );

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn test_round_trip_without_normals() {
    // Normals are optional: `None` opts out entirely, so no normals are
    // written and the reloaded data carries `None`.
    let mut data = make_test_data();
    data.normals_xyz = None;

    let dir = std::env::temp_dir().join("sfmr_test_no_normals");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("test.sfmr");
    write_sfmr(&path, &mut data).unwrap();

    let loaded = read_sfmr(&path).unwrap();
    assert!(loaded.normals_xyz.is_none());

    // The normals entry is absent from the archive.
    let file = std::fs::File::open(&path).unwrap();
    let mut archive = zip::ZipArchive::new(file).unwrap();
    assert!(archive
        .by_name("points3d/normals_xyz.5.3.float32.zst")
        .is_err());

    let (valid, errors) = verify_sfmr(&path).unwrap();
    assert!(valid, "no-normals verification failed: {errors:?}");

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

#[test]
fn test_round_trip_with_patches_and_bitmaps() {
    let mut data = make_test_data();
    // Per-point patch frame over all 5 points, with R=4 bitmaps. Distinct
    // per-cell values catch any reshape/transposition bug on round trip.
    let point_count = 5;
    let r = 4;
    let u_halfvec_xyz = Array2::from_shape_fn((point_count, 3), |(i, j)| (i * 3 + j) as f32 * 0.1);
    let v_halfvec_xyz =
        Array2::from_shape_fn((point_count, 3), |(i, j)| (i * 3 + j) as f32 * 0.01 + 1.0);
    // 4 channels: RGB + an alpha confidence plane.
    let bitmaps = Array4::<u8>::from_shape_fn((point_count, r, r, 4), |(i, y, x, c)| {
        ((i * 37 + y * 11 + x * 5 + c) % 256) as u8
    });
    data.patch_u_halfvec_xyz = Some(u_halfvec_xyz.clone());
    data.patch_v_halfvec_xyz = Some(v_halfvec_xyz.clone());
    data.patch_bitmaps_y_x_rgba = Some(bitmaps.clone());

    // Snapshot every points3d array the writer may touch so we can assert
    // the patch frame round-trips without disturbing its neighbours.
    let positions_xyzw = data.positions_xyzw.clone();
    let colors_rgb = data.colors_rgb.clone();
    let reprojection_errors = data.reprojection_errors.clone();
    let normals_xyz = data.normals_xyz.clone().unwrap();

    let dir = std::env::temp_dir().join("sfmr_test_patches_round_trip");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("test.sfmr");
    write_sfmr(&path, &mut data).unwrap();

    let meta = read_sfmr_metadata(&path).unwrap();
    assert_eq!(meta.version, 4);

    let loaded = read_sfmr(&path).unwrap();
    // Patch arrays are parallel to the points and round-trip exactly.
    assert_eq!(loaded.patch_u_halfvec_xyz.unwrap(), u_halfvec_xyz);
    assert_eq!(loaded.patch_v_halfvec_xyz.unwrap(), v_halfvec_xyz);
    assert_eq!(loaded.patch_bitmaps_y_x_rgba.unwrap(), bitmaps);
    // The rest of the points3d section round-trips unchanged alongside it.
    assert_eq!(loaded.positions_xyzw, positions_xyzw);
    assert_eq!(loaded.colors_rgb, colors_rgb);
    assert_eq!(loaded.reprojection_errors, reprojection_errors);
    assert_eq!(loaded.normals_xyz.unwrap(), normals_xyz);
    // The patch frame is hashed inside the points3d section.
    assert_eq!(loaded.content_hash.points3d_xxh128.len(), 32);

    let (valid, errors) = verify_sfmr(&path).unwrap();
    assert!(valid, "patch round-trip verification failed: {errors:?}");

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn test_round_trip_with_patches_no_bitmaps() {
    let mut data = make_test_data();
    // A patch frame without bitmaps exercises the has_patch_bitmaps=false path.
    let u_halfvec_xyz = Array2::from_shape_fn((5, 3), |(i, j)| (i * 3 + j) as f32 * 0.1);
    let v_halfvec_xyz = Array2::from_shape_fn((5, 3), |(i, j)| (i * 3 + j) as f32 * 0.05 - 0.5);
    data.patch_u_halfvec_xyz = Some(u_halfvec_xyz.clone());
    data.patch_v_halfvec_xyz = Some(v_halfvec_xyz.clone());

    let dir = std::env::temp_dir().join("sfmr_test_patches_no_bitmaps");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("test.sfmr");
    write_sfmr(&path, &mut data).unwrap();

    let loaded = read_sfmr(&path).unwrap();
    // The frame round-trips by value; bitmaps stay absent.
    assert_eq!(loaded.patch_u_halfvec_xyz.unwrap(), u_halfvec_xyz);
    assert_eq!(loaded.patch_v_halfvec_xyz.unwrap(), v_halfvec_xyz);
    assert!(loaded.patch_bitmaps_y_x_rgba.is_none());

    let (valid, errors) = verify_sfmr(&path).unwrap();
    assert!(valid, "{errors:?}");

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn test_patch_frame_without_normals_round_trips() {
    // A patch frame is independent of normals: storing the frame with
    // `normals_xyz = None` is well-formed (the normal is implied by the
    // frame), so it must round-trip.
    let mut data = make_test_data();
    data.normals_xyz = None;
    let u_halfvec_xyz = Array2::from_shape_fn((5, 3), |(i, j)| (i * 3 + j) as f32 * 0.1);
    let v_halfvec_xyz = Array2::from_shape_fn((5, 3), |(i, j)| (i * 3 + j) as f32 * 0.05 - 0.5);
    data.patch_u_halfvec_xyz = Some(u_halfvec_xyz.clone());
    data.patch_v_halfvec_xyz = Some(v_halfvec_xyz.clone());

    let dir = std::env::temp_dir().join("sfmr_test_patch_frame_no_normals");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("test.sfmr");
    write_sfmr(&path, &mut data).unwrap();

    let loaded = read_sfmr(&path).unwrap();
    assert!(loaded.normals_xyz.is_none());
    assert_eq!(loaded.patch_u_halfvec_xyz.unwrap(), u_halfvec_xyz);
    assert_eq!(loaded.patch_v_halfvec_xyz.unwrap(), v_halfvec_xyz);

    let (valid, errors) = verify_sfmr(&path).unwrap();
    assert!(valid, "{errors:?}");

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn test_patch_bitmaps_without_frame_rejected() {
    // The one cross-restriction: bitmaps require the patch frame. Bitmaps
    // with no `u`/`v` must be rejected at write time.
    let mut data = make_test_data();
    data.patch_bitmaps_y_x_rgba = Some(Array4::<u8>::zeros((5, 4, 4, 4)));

    let dir = std::env::temp_dir().join("sfmr_test_bitmaps_no_frame");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("test.sfmr");
    let result = write_sfmr(&path, &mut data);
    assert!(
        matches!(result, Err(SfmrError::ShapeMismatch(_))),
        "bitmaps without a frame should be rejected, got {result:?}"
    );

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn test_patch_frame_half_present_rejected() {
    // `u` and `v` form the frame as a pair: one without the other is not a
    // frame and must be rejected.
    let mut data = make_test_data();
    data.patch_u_halfvec_xyz = Some(Array2::from_shape_fn((5, 3), |(i, j)| {
        (i * 3 + j) as f32 * 0.1
    }));

    let dir = std::env::temp_dir().join("sfmr_test_frame_half");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("test.sfmr");
    let result = write_sfmr(&path, &mut data);
    assert!(
        matches!(result, Err(SfmrError::ShapeMismatch(_))),
        "a half-present frame should be rejected, got {result:?}"
    );

    std::fs::remove_dir_all(&dir).unwrap();
}
