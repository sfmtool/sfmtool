use sfmr_format::{RigFrameData, FEATURE_SOURCE_EMBEDDED_PATCHES, FEATURE_SOURCE_SIFT_FILES};

use super::*;

#[test]
fn test_observations_for_point() {
    let recon = SfmrReconstruction::demo(1000);
    // Demo creates 1000 points, each observed by 2 cameras
    assert_eq!(recon.observation_offsets.len(), recon.points.len() + 1);
    assert_eq!(
        *recon.observation_offsets.last().unwrap(),
        recon.tracks.len()
    );

    for i in 0..recon.points.len() {
        let obs = recon.observations_for_point(i);
        assert_eq!(obs.len(), 2, "point {i} should have 2 observations");
        for o in obs {
            assert_eq!(o.point_index, i as u32);
        }
    }
}

#[test]
fn test_track_image_indices() {
    let recon = SfmrReconstruction::demo(1000);
    // Point 0 is observed by cameras 0 and 1 in the demo
    let images = recon.track_image_indices(0);
    assert_eq!(images.len(), 2);
}

#[test]
fn test_observation_affine_shape_frontoparallel() {
    use nalgebra::{Point3, UnitQuaternion, Vector3};
    use ndarray::Array2;

    let mut recon = SfmrReconstruction::demo(16);

    // Pinhole with principal point at the origin, for easy arithmetic.
    recon.cameras[0] = crate::CameraIntrinsics {
        model: crate::camera::CameraModel::Pinhole {
            focal_length_x: 100.0,
            focal_length_y: 100.0,
            principal_point_x: 0.0,
            principal_point_y: 0.0,
        },
        width: 4096,
        height: 4096,
    };
    // Image 0 at the origin looking down −Z (identity world->camera, the
    // canonical convention).
    recon.images[0].camera_index = 0;
    recon.images[0].quaternion_wxyz = UnitQuaternion::identity();
    recon.images[0].translation_xyz = Vector3::zeros();
    // Point 0 at (0, 0, −2) — in front — with a fronto-parallel patch:
    // half-extent 0.05 along world X (u) and Y (v), normal u × v = +Z
    // (toward the camera).
    recon.points[0].position = Point3::new(0.0, 0.0, -2.0);
    recon.points[0].w = 1.0;
    let n = recon.points.len();
    let mut u = Array2::<f32>::zeros((n, 3));
    let mut v = Array2::<f32>::zeros((n, 3));
    u[[0, 0]] = 0.05;
    v[[0, 1]] = 0.05;
    recon.patch_u_halfvec_xyz = Some(u);
    recon.patch_v_halfvec_xyz = Some(v);

    // Keypoint = projection of the point centre = (0, 0). Expected shape is
    // axis-aligned with scale f * half_extent / depth = 100 * 0.05 / 2 = 2.5.
    // The v half-vector (world +Y = camera +Y) points *up* on screen, so its
    // pixel column is (0, −2.5) — pixel v grows down.
    let shape = recon
        .observation_affine_shape(0, 0, [0.0, 0.0])
        .expect("a finite patch projects to a shape");
    assert!((shape[0][0] - 2.5).abs() < 1e-3, "a11 = {}", shape[0][0]);
    assert!((shape[1][1] + 2.5).abs() < 1e-3, "a22 = {}", shape[1][1]);
    assert!(shape[0][1].abs() < 1e-3, "a12 = {}", shape[0][1]);
    assert!(shape[1][0].abs() < 1e-3, "a21 = {}", shape[1][0]);

    // A point at infinity still has a shape: its patch is tangent to the
    // direction sphere, and here the tangent frame projects to the same
    // axis-aligned (roughly circular) footprint.
    recon.points[0].w = 0.0;
    let inf_shape = recon
        .observation_affine_shape(0, 0, [0.0, 0.0])
        .expect("an infinity patch projects to a shape");
    assert!(
        (inf_shape[0][0] - 2.5).abs() < 1e-3,
        "inf a11 = {}",
        inf_shape[0][0]
    );
    assert!(
        (inf_shape[1][1] + 2.5).abs() < 1e-3,
        "inf a22 = {}",
        inf_shape[1][1]
    );

    // No patch arrays at all -> None.
    recon.points[0].w = 1.0;
    recon.patch_u_halfvec_xyz = None;
    assert!(recon.observation_affine_shape(0, 0, [0.0, 0.0]).is_none());
}

#[test]
fn test_observation_affine_shape_rotated_camera() {
    use nalgebra::{Point3, Rotation3, UnitQuaternion, Vector3};
    use ndarray::Array2;

    let mut recon = SfmrReconstruction::demo(16);
    recon.cameras[0] = crate::CameraIntrinsics {
        model: crate::camera::CameraModel::Pinhole {
            focal_length_x: 100.0,
            focal_length_y: 100.0,
            principal_point_x: 0.0,
            principal_point_y: 0.0,
        },
        width: 4096,
        height: 4096,
    };
    // World->camera rotation = +90° about Z; camera at the origin. The
    // rotated canonical camera still sees a point at world −Z in front.
    let rot = Rotation3::from_axis_angle(&Vector3::z_axis(), std::f64::consts::FRAC_PI_2);
    recon.images[0].camera_index = 0;
    recon.images[0].quaternion_wxyz = UnitQuaternion::from_rotation_matrix(&rot);
    recon.images[0].translation_xyz = Vector3::zeros();
    recon.points[0].position = Point3::new(0.0, 0.0, -2.0);
    recon.points[0].w = 1.0;
    let n = recon.points.len();
    let mut u = Array2::<f32>::zeros((n, 3));
    let mut v = Array2::<f32>::zeros((n, 3));
    u[[0, 0]] = 0.05; // world +X
    v[[0, 1]] = 0.05; // world +Y
    recon.patch_u_halfvec_xyz = Some(u);
    recon.patch_v_halfvec_xyz = Some(v);

    // The point still projects to (0, 0). The world->camera rotation R sends
    // world +X -> camera +Y (image up, pixel v *decreasing*) and world +Y ->
    // camera −X, so the columns rotate 90°: u -> (0, −2.5), v -> (−2.5, 0).
    // Projecting with R^T (a transpose bug) would rotate the other way, so
    // this pins the rotation direction.
    let s = recon
        .observation_affine_shape(0, 0, [0.0, 0.0])
        .expect("shape");
    assert!(s[0][0].abs() < 1e-3, "a11 = {}", s[0][0]);
    assert!((s[1][0] + 2.5).abs() < 1e-3, "a21 = {}", s[1][0]);
    assert!((s[0][1] + 2.5).abs() < 1e-3, "a12 = {}", s[0][1]);
    assert!(s[1][1].abs() < 1e-3, "a22 = {}", s[1][1]);

    // A point whose frame row is all-zero (no patch) -> None.
    assert!(recon.observation_affine_shape(1, 0, [0.0, 0.0]).is_none());
}

#[test]
fn test_subset_keep_all_images_is_identity() {
    let recon = SfmrReconstruction::demo(1000);
    let indices: Vec<u32> = (0..recon.images.len() as u32).collect();
    let subset = recon.subset_by_image_indices(&indices, false).unwrap();
    assert_eq!(subset.images.len(), recon.images.len());
    assert_eq!(subset.points.len(), recon.points.len());
    assert_eq!(subset.tracks.len(), recon.tracks.len());
    assert_eq!(subset.observation_counts, recon.observation_counts);
}

#[test]
fn test_subset_keeps_all_points_by_default() {
    let recon = SfmrReconstruction::demo(1000);
    // Keep only image 0. In the demo, point i is observed by images
    // (i % 8) and ((i + 1) % 8), so ~2 points out of every 8 touch image 0.
    let subset = recon.subset_by_image_indices(&[0], false).unwrap();

    assert_eq!(subset.images.len(), 1);
    // Default: all points kept even if their track dropped to zero.
    assert_eq!(subset.points.len(), recon.points.len());
    assert_eq!(subset.observation_counts.len(), recon.points.len());

    // Observations that survived are the ones referencing image 0.
    let expected_surviving: usize = recon.tracks.iter().filter(|t| t.image_index == 0).count();
    assert_eq!(subset.tracks.len(), expected_surviving);
    // Every surviving track now references the new image index 0.
    for obs in &subset.tracks {
        assert_eq!(obs.image_index, 0);
    }
    // Per-point observation_counts sum to the surviving track count.
    assert_eq!(
        subset.observation_counts.iter().sum::<u32>() as usize,
        expected_surviving
    );
    // And some points have zero observations.
    assert!(subset.observation_counts.contains(&0));
}

#[test]
fn test_subset_drops_orphaned_points_when_requested() {
    let recon = SfmrReconstruction::demo(1000);
    let subset = recon.subset_by_image_indices(&[0], true).unwrap();

    assert_eq!(subset.images.len(), 1);
    // All surviving points have at least one observation.
    assert!(subset.observation_counts.iter().all(|&c| c > 0));
    assert_eq!(
        subset.points.len(),
        subset.observation_counts.iter().filter(|&&c| c > 0).count()
    );
    // Point IDs in tracks are contiguous.
    let max_pt = subset
        .tracks
        .iter()
        .map(|t| t.point_index)
        .max()
        .unwrap_or(0);
    assert!((max_pt as usize) < subset.points.len());
    // Observation offsets round-trip.
    assert_eq!(
        *subset.observation_offsets.last().unwrap(),
        subset.tracks.len()
    );
}

#[test]
fn test_subset_rejects_out_of_bounds_and_duplicates() {
    let recon = SfmrReconstruction::demo(1000);
    let n = recon.images.len() as u32;
    assert!(recon.subset_by_image_indices(&[n], false).is_err());
    assert!(recon.subset_by_image_indices(&[0, 0], false).is_err());
}

#[test]
fn test_subset_filters_rig_frame_data() {
    use ndarray::{Array1, Array2};
    use sfmr_format::{FramesMetadata, RigDefinition, RigsMetadata};

    // Start from the demo (8 images) and attach a trivial rig/frame
    // structure: one single-sensor rig, one frame per image.
    let mut recon = SfmrReconstruction::demo(1000);
    let n_images = recon.images.len();
    let rig_def = RigDefinition {
        name: "rig0".to_string(),
        sensor_count: 1,
        sensor_offset: 0,
        ref_sensor_name: "sensor0".to_string(),
        sensor_names: vec!["sensor0".to_string()],
    };
    recon.rig_frame_data = Some(RigFrameData {
        rigs_metadata: RigsMetadata {
            rig_count: 1,
            sensor_count: 1,
            rigs: vec![rig_def],
        },
        sensor_camera_indexes: Array1::from_vec(vec![0u32]),
        sensor_quaternions_wxyz: Array2::from_shape_vec((1, 4), vec![1.0, 0.0, 0.0, 0.0]).unwrap(),
        sensor_translations_xyz: Array2::from_shape_vec((1, 3), vec![0.0, 0.0, 0.0]).unwrap(),
        frames_metadata: FramesMetadata {
            frame_count: n_images as u32,
        },
        rig_indexes: Array1::from_vec(vec![0u32; n_images]),
        image_sensor_indexes: Array1::from_vec(vec![0u32; n_images]),
        image_frame_indexes: Array1::from_vec((0..n_images as u32).collect()),
    });

    // Keep images 0, 3, 5 — three frames survive and must be remapped to 0,1,2.
    let subset = recon.subset_by_image_indices(&[0, 3, 5], false).unwrap();
    let rf = subset.rig_frame_data.as_ref().unwrap();
    assert_eq!(rf.frames_metadata.frame_count, 3);
    assert_eq!(rf.rig_indexes.len(), 3);
    assert_eq!(rf.image_frame_indexes.to_vec(), vec![0, 1, 2]);
    // Sensor definitions are unchanged.
    assert_eq!(rf.rigs_metadata.rig_count, 1);
    assert_eq!(rf.sensor_camera_indexes.to_vec(), vec![0]);
}

/// Build an `embedded_patches` reconstruction from the sift_files `demo`, by
/// dropping the `.sift`-link arrays and substituting inline keypoints + a
/// per-image hash. Keypoint row `i` holds `[2i, 2i+1]` so a test can assert
/// exactly which observation rows survive an edit. Every image hash is `[7; 16]`.
fn demo_embedded(num_points: usize) -> SfmrReconstruction {
    let mut data = SfmrReconstruction::demo(num_points).to_sfmr_data();
    let m = data.metadata.observation_count as usize;
    let n = data.metadata.image_count as usize;
    data.metadata.feature_source = FEATURE_SOURCE_EMBEDDED_PATCHES.to_string();
    data.feature_indexes = None;
    data.feature_tool_hashes = None;
    data.sift_content_hashes = None;
    data.image_file_hashes = Some(vec![[7u8; 16]; n]);
    data.keypoints_xy = Some(ndarray::Array2::from_shape_fn((m, 2), |(i, c)| {
        (i * 2 + c) as f32
    }));
    SfmrReconstruction::from_sfmr_data(data).expect("embedded should load")
}

#[test]
fn test_embedded_patches_round_trips_through_reconstruction() {
    // Build an embedded_patches SfmrData by hand (drop the .sift-link arrays, add
    // inline keypoints + image hash), load it, and round-trip it back.
    let recon = demo_embedded(10);
    let n = recon.images.len();
    let kp_expected = recon.keypoints_xy().unwrap().clone();

    assert_eq!(recon.feature_source(), FEATURE_SOURCE_EMBEDDED_PATCHES);
    assert_eq!(recon.keypoints_xy().unwrap(), &kp_expected);
    assert_eq!(recon.image_file_hashes().unwrap().len(), n);
    // Feature-index machinery is empty for embedded (placeholders only).
    assert!(recon.image_feature_to_point.iter().all(|m| m.is_empty()));

    // Round-trips back to embedded columns, .sift-link arrays absent.
    let out = recon.to_sfmr_data();
    assert_eq!(out.metadata.feature_source, FEATURE_SOURCE_EMBEDDED_PATCHES);
    assert_eq!(out.keypoints_xy.unwrap(), kp_expected);
    assert!(out.image_file_hashes.is_some());
    assert!(out.feature_indexes.is_none());
    assert!(out.feature_tool_hashes.is_none());
    assert!(out.sift_content_hashes.is_none());
}

#[test]
fn test_recompute_point_errors_embedded_uses_inline_keypoints() {
    // An embedded_patches recon has no `.sift` files; recomputing errors must
    // use the inline keypoints instead of trying (and failing) to read `.sift`.
    // The sift_files path on the same synthetic recon would error on the missing
    // `.sift`, so a successful Ok here proves the embedded branch is taken.
    let mut recon = demo_embedded(10);
    recon
        .recompute_point_errors()
        .expect("embedded recompute must not read .sift");
    assert!(recon.points.iter().all(|p| p.error.is_finite()));
}

#[test]
fn test_recompute_infinity_point_errors_embedded() {
    // Flag one point as at-infinity; the infinity-only recompute must run
    // against the inline keypoints (no `.sift`) and leave a finite error.
    let mut recon = demo_embedded(10);
    recon.points[0].position = nalgebra::Point3::new(0.0, 0.0, -1.0);
    recon.points[0].w = 0.0;
    recon
        .recompute_infinity_point_errors()
        .expect("embedded infinity recompute must not read .sift");
    assert!(recon.points[0].error.is_finite());
}

#[test]
fn test_to_sfmr_data_is_sift_files() {
    // A reconstruction round-trips as a sift_files v4 file.
    let data = SfmrReconstruction::demo(10).to_sfmr_data();
    assert_eq!(data.metadata.feature_source, FEATURE_SOURCE_SIFT_FILES);
    assert!(data.feature_indexes.is_some());
    assert!(data.keypoints_xy.is_none());
    assert!(data.image_file_hashes.is_none());
}

#[test]
fn test_filter_points_keeps_embedded_keypoints_parallel() {
    // demo observes each point by 2 cameras, so observations are grouped by
    // point: point i owns keypoint rows 2i and 2i+1 (values [2k] / [2k+1]).
    let recon = demo_embedded(4);
    let mask = vec![true, false, true, false];
    let out = recon.filter_points_by_mask(&mask);

    assert_eq!(out.feature_source(), FEATURE_SOURCE_EMBEDDED_PATCHES);
    assert_eq!(out.point_count(), 2);

    // Surviving observations are the rows for points 0 and 2 — source rows
    // [0, 1, 4, 5] — kept in order and still parallel to the new tracks.
    let kp = out.keypoints_xy().unwrap();
    assert_eq!(kp.nrows(), out.tracks.len());
    assert_eq!(kp.nrows(), 4);
    assert_eq!([kp[[0, 0]], kp[[0, 1]]], [0.0, 1.0]); // source row 0
    assert_eq!([kp[[1, 0]], kp[[1, 1]]], [2.0, 3.0]); // source row 1
    assert_eq!([kp[[2, 0]], kp[[2, 1]]], [8.0, 9.0]); // source row 4
    assert_eq!([kp[[3, 0]], kp[[3, 1]]], [10.0, 11.0]); // source row 5

    // Images are untouched, so the per-image hashes pass through unchanged.
    assert_eq!(
        out.image_file_hashes().unwrap(),
        recon.image_file_hashes().unwrap()
    );
    out.validate_observation_columns().unwrap();
}

#[test]
fn test_se3_transform_preserves_embedded_columns() {
    use crate::geometry::RotQuaternion;
    use crate::Se3Transform;
    use nalgebra::{UnitQuaternion, Vector3};

    let recon = demo_embedded(4);
    let kp0 = recon.keypoints_xy().unwrap().clone();

    // 2D features are pose-invariant: a similarity transform must not touch the
    // keypoints or the per-image hashes.
    let rot = RotQuaternion::from_nalgebra(UnitQuaternion::from_axis_angle(
        &Vector3::z_axis(),
        std::f64::consts::FRAC_PI_3,
    ));
    let t = Se3Transform::new(rot, Vector3::new(1.0, -2.0, 0.5), 1.5);
    let out = recon.apply_se3_transform(&t);

    assert_eq!(out.feature_source(), FEATURE_SOURCE_EMBEDDED_PATCHES);
    assert_eq!(out.keypoints_xy().unwrap(), &kp0);
    assert_eq!(
        out.image_file_hashes().unwrap(),
        recon.image_file_hashes().unwrap()
    );
    out.validate_observation_columns().unwrap();
}

#[test]
fn test_subset_by_image_indices_rejects_embedded() {
    let recon = demo_embedded(4);
    let all: Vec<u32> = (0..recon.images.len() as u32).collect();
    // (SfmrReconstruction is not Debug, so match rather than expect_err.)
    let Err(err) = recon.subset_by_image_indices(&all, true) else {
        panic!("subset must reject embedded_patches");
    };
    assert!(
        err.contains("embedded_patches"),
        "unexpected error message: {err}"
    );
}

#[test]
fn test_find_points_at_infinity_rejects_embedded() {
    let recon = demo_embedded(4);
    let Err(err) = recon.find_points_at_infinity(1.0, 0.7, 0.8, 2, None, 1.0) else {
        panic!("find_points_at_infinity must reject embedded_patches");
    };
    let msg = err.to_string();
    assert!(
        msg.contains("embedded_patches"),
        "unexpected message: {msg}"
    );
}

#[test]
fn test_validate_observation_columns_detects_desync() {
    // Healthy reconstructions in both modes pass.
    SfmrReconstruction::demo(4)
        .validate_observation_columns()
        .unwrap();
    demo_embedded(4).validate_observation_columns().unwrap();

    // Truncating keypoints below the observation count is caught.
    let mut recon = demo_embedded(4);
    if let ObservationSource::EmbeddedPatches { keypoints_xy, .. } = &mut recon.observations {
        *keypoints_xy = keypoints_xy.select(ndarray::Axis(0), &[0, 1]);
    }
    let err = recon
        .validate_observation_columns()
        .expect_err("desynced keypoints must be rejected");
    assert!(err.contains("keypoints_xy"), "unexpected message: {err}");
}

// ── .sfmr v4 → v5 convention upgrade on load (plan D1) ─────────────────────

/// Convert a canonical `SfmrData` to the COLMAP convention in place — the
/// exact inverse of `convention::sfmr_data_colmap_to_canonical` — so tests
/// can author version-4 (COLMAP-convention) content from a canonical fixture.
fn sfmr_data_canonical_to_colmap(data: &mut SfmrData) {
    use crate::geometry::convention::{pose_canonical_to_colmap, world_rotate_w_inverse};
    use crate::geometry::RotQuaternion;

    for i in 0..data.quaternions_wxyz.nrows() {
        let q = RotQuaternion::from_wxyz_array([
            data.quaternions_wxyz[[i, 0]],
            data.quaternions_wxyz[[i, 1]],
            data.quaternions_wxyz[[i, 2]],
            data.quaternions_wxyz[[i, 3]],
        ]);
        let t = Vector3::new(
            data.translations_xyz[[i, 0]],
            data.translations_xyz[[i, 1]],
            data.translations_xyz[[i, 2]],
        );
        let (q_new, t_new) = pose_canonical_to_colmap(&q, &t);
        for (k, &v) in q_new.to_wxyz_array().iter().enumerate() {
            data.quaternions_wxyz[[i, k]] = v;
        }
        for k in 0..3 {
            data.translations_xyz[[i, k]] = t_new[k];
        }
    }
    for i in 0..data.positions_xyzw.nrows() {
        let v = world_rotate_w_inverse(&Vector3::new(
            data.positions_xyzw[[i, 0]],
            data.positions_xyzw[[i, 1]],
            data.positions_xyzw[[i, 2]],
        ));
        for k in 0..3 {
            data.positions_xyzw[[i, k]] = v[k];
        }
    }
    let rotate_rows = |arr: &mut Array2<f32>| {
        for i in 0..arr.nrows() {
            let y = arr[[i, 1]];
            let z = arr[[i, 2]];
            arr[[i, 1]] = -z;
            arr[[i, 2]] = y;
        }
    };
    if let Some(n) = data.normals_xyz.as_mut() {
        rotate_rows(n);
    }
    if let Some(u) = data.patch_u_halfvec_xyz.as_mut() {
        rotate_rows(u);
    }
    if let Some(v) = data.patch_v_halfvec_xyz.as_mut() {
        rotate_rows(v);
    }
}

/// Copy a written `.sfmr` archive, rewriting only `metadata.json.zst` so its
/// `version` field reads `4` — producing genuine version-4 bytes on disk.
fn rewrite_sfmr_version_4(src: &Path, dst: &Path) {
    use std::io::{Read, Write};

    let archive_file = std::fs::File::open(src).unwrap();
    let mut archive = zip::ZipArchive::new(archive_file).unwrap();
    let names: Vec<String> = archive.file_names().map(|s| s.to_string()).collect();
    let out = std::fs::File::create(dst).unwrap();
    let mut zip_out = zip::ZipWriter::new(out);
    let stored =
        zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored);
    for name in &names {
        let mut compressed = Vec::new();
        archive
            .by_name(name)
            .unwrap()
            .read_to_end(&mut compressed)
            .unwrap();
        zip_out.start_file(name, stored).unwrap();
        if name == "metadata.json.zst" {
            let mut json: serde_json::Value =
                serde_json::from_slice(&zstd::stream::decode_all(&compressed[..]).unwrap())
                    .unwrap();
            json.as_object_mut()
                .unwrap()
                .insert("version".into(), serde_json::json!(4));
            let bytes = zstd::bulk::compress(&serde_json::to_vec(&json).unwrap(), 3).unwrap();
            zip_out.write_all(&bytes).unwrap();
        } else {
            zip_out.write_all(&compressed).unwrap();
        }
    }
    zip_out.finish().unwrap();
}

fn assert_rotations_close(a: &UnitQuaternion<f64>, b: &UnitQuaternion<f64>, ctx: &str) {
    let ra = a.to_rotation_matrix();
    let rb = b.to_rotation_matrix();
    for r in 0..3 {
        for c in 0..3 {
            assert!(
                (ra[(r, c)] - rb[(r, c)]).abs() < 1e-9,
                "{ctx}: rotation mismatch at ({r}, {c}): {} vs {}",
                ra[(r, c)],
                rb[(r, c)]
            );
        }
    }
}

#[test]
fn test_sfmr_data_colmap_to_canonical_converts_every_section() {
    use crate::geometry::convention::{
        pose_colmap_to_canonical, relative_pose_conjugate_s, sfmr_data_colmap_to_canonical,
    };
    use crate::geometry::RotQuaternion;
    use ndarray::Array1;
    use sfmr_format::{FramesMetadata, RigDefinition, RigsMetadata};

    let mut recon = SfmrReconstruction::demo(6);
    // Exercise the w = 0 branch: point 0 becomes an infinity direction.
    recon.points[0].position = Point3::new(0.6, 0.0, 0.8);
    recon.points[0].w = 0.0;
    recon.infinity_point_count = 1;

    let mut data = recon.to_sfmr_data();
    let p = recon.points.len();

    // Attach a patch frame so the u/v half-vector rotation is exercised.
    let mut u = Array2::<f32>::zeros((p, 3));
    let mut v = Array2::<f32>::zeros((p, 3));
    for i in 0..p {
        u[[i, 0]] = 1.0;
        u[[i, 1]] = 0.25;
        v[[i, 1]] = -0.25;
        v[[i, 2]] = 1.0;
    }
    data.patch_u_halfvec_xyz = Some(u.clone());
    data.patch_v_halfvec_xyz = Some(v.clone());

    // Attach a two-sensor rig so the sensor_from_rig conjugation is exercised.
    let sensor_q = Array2::from_shape_vec(
        (2, 4),
        vec![
            1.0, 0.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0, // 180° about Y
        ],
    )
    .unwrap();
    let sensor_t = Array2::from_shape_vec((2, 3), vec![0.0, 0.0, 0.0, 0.1, 0.2, 0.3]).unwrap();
    data.rig_frame_data = Some(RigFrameData {
        rigs_metadata: RigsMetadata {
            rig_count: 1,
            sensor_count: 2,
            rigs: vec![RigDefinition {
                name: "rig".into(),
                sensor_count: 2,
                sensor_offset: 0,
                ref_sensor_name: "s0".into(),
                sensor_names: vec!["s0".into(), "s1".into()],
            }],
        },
        sensor_camera_indexes: Array1::from_vec(vec![0, 0]),
        sensor_quaternions_wxyz: sensor_q.clone(),
        sensor_translations_xyz: sensor_t.clone(),
        frames_metadata: FramesMetadata { frame_count: 0 },
        rig_indexes: Array1::from_vec(vec![]),
        image_sensor_indexes: Array1::from_vec(vec![0; recon.images.len()]),
        image_frame_indexes: Array1::from_vec(vec![0; recon.images.len()]),
    });

    let orig_quats = data.quaternions_wxyz.clone();
    let orig_trans = data.translations_xyz.clone();
    let orig_positions = data.positions_xyzw.clone();
    let orig_normals = data.normals_xyz.clone().unwrap();
    let orig_depth_stats = serde_json::to_string(&data.depth_statistics).unwrap();

    sfmr_data_colmap_to_canonical(&mut data);

    // Camera poses: R' = S·R·Wᵀ, t' = S·t per row.
    for i in 0..orig_quats.nrows() {
        let q = RotQuaternion::from_wxyz_array([
            orig_quats[[i, 0]],
            orig_quats[[i, 1]],
            orig_quats[[i, 2]],
            orig_quats[[i, 3]],
        ]);
        let t = Vector3::new(orig_trans[[i, 0]], orig_trans[[i, 1]], orig_trans[[i, 2]]);
        let (q_exp, t_exp) = pose_colmap_to_canonical(&q, &t);
        let got = RotQuaternion::from_wxyz_array([
            data.quaternions_wxyz[[i, 0]],
            data.quaternions_wxyz[[i, 1]],
            data.quaternions_wxyz[[i, 2]],
            data.quaternions_wxyz[[i, 3]],
        ]);
        assert_rotations_close(got.as_nalgebra(), q_exp.as_nalgebra(), "camera pose");
        for k in 0..3 {
            assert!((data.translations_xyz[[i, k]] - t_exp[k]).abs() < 1e-12);
        }
    }

    // Rig sensor poses: S-conjugation. Identity stays identity; the 180°-Y
    // rotation is S-invariant while the translation flips its y/z signs.
    let rig = data.rig_frame_data.as_ref().unwrap();
    for i in 0..2 {
        let q = RotQuaternion::from_wxyz_array([
            sensor_q[[i, 0]],
            sensor_q[[i, 1]],
            sensor_q[[i, 2]],
            sensor_q[[i, 3]],
        ]);
        let t = Vector3::new(sensor_t[[i, 0]], sensor_t[[i, 1]], sensor_t[[i, 2]]);
        let (q_exp, t_exp) = relative_pose_conjugate_s(&q, &t);
        let got = RotQuaternion::from_wxyz_array([
            rig.sensor_quaternions_wxyz[[i, 0]],
            rig.sensor_quaternions_wxyz[[i, 1]],
            rig.sensor_quaternions_wxyz[[i, 2]],
            rig.sensor_quaternions_wxyz[[i, 3]],
        ]);
        assert_rotations_close(got.as_nalgebra(), q_exp.as_nalgebra(), "sensor pose");
        for k in 0..3 {
            assert!((rig.sensor_translations_xyz[[i, k]] - t_exp[k]).abs() < 1e-12);
        }
    }
    assert_eq!(rig.sensor_translations_xyz[[1, 0]], 0.1);
    assert_eq!(rig.sensor_translations_xyz[[1, 1]], -0.2);
    assert_eq!(rig.sensor_translations_xyz[[1, 2]], -0.3);

    // World points: (x, y, z) → (x, z, −y), w carried through — including the
    // w = 0 infinity direction in row 0.
    for i in 0..p {
        assert_eq!(data.positions_xyzw[[i, 0]], orig_positions[[i, 0]]);
        assert_eq!(data.positions_xyzw[[i, 1]], orig_positions[[i, 2]]);
        assert_eq!(data.positions_xyzw[[i, 2]], -orig_positions[[i, 1]]);
        assert_eq!(data.positions_xyzw[[i, 3]], orig_positions[[i, 3]]);
    }
    assert_eq!(data.positions_xyzw[[0, 3]], 0.0, "row 0 stays at infinity");

    // Normals and patch half-vectors rotate by W (exact on f32).
    let normals = data.normals_xyz.as_ref().unwrap();
    for i in 0..p {
        assert_eq!(normals[[i, 0]], orig_normals[[i, 0]]);
        assert_eq!(normals[[i, 1]], orig_normals[[i, 2]]);
        assert_eq!(normals[[i, 2]], -orig_normals[[i, 1]]);
    }
    let u_new = data.patch_u_halfvec_xyz.as_ref().unwrap();
    let v_new = data.patch_v_halfvec_xyz.as_ref().unwrap();
    for i in 0..p {
        assert_eq!(u_new[[i, 0]], u[[i, 0]]);
        assert_eq!(u_new[[i, 1]], u[[i, 2]]);
        assert_eq!(u_new[[i, 2]], -u[[i, 1]]);
        assert_eq!(v_new[[i, 0]], v[[i, 0]]);
        assert_eq!(v_new[[i, 1]], v[[i, 2]]);
        assert_eq!(v_new[[i, 2]], -v[[i, 1]]);
    }

    // Depth statistics are invariant under the conversion and must not be
    // touched (canonical depth −z' equals the stored COLMAP +z depth).
    assert_eq!(
        serde_json::to_string(&data.depth_statistics).unwrap(),
        orig_depth_stats
    );
}

#[test]
fn test_v4_file_upgrades_to_canonical_on_load_and_saves_as_v5() {
    let dir = std::env::temp_dir().join("sfmr_core_v4_upgrade");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    // A workspace marker so SfmrReconstruction::load can resolve the dir.
    std::fs::write(dir.join(".sfm-workspace.json"), "{}").unwrap();

    // Canonical ground truth.
    let recon = SfmrReconstruction::demo(16);

    // Author genuine v4 bytes: canonical → COLMAP content, written with the
    // current writer, then the archive's version field rewritten to 4.
    let mut colmap_data = recon.to_sfmr_data();
    sfmr_data_canonical_to_colmap(&mut colmap_data);
    let staged = dir.join("staged.sfmr");
    let options = sfmr_format::WriteOptions {
        skip_recompute_depth_stats: true,
        ..Default::default()
    };
    sfmr_format::write_sfmr_with_options(&staged, &mut colmap_data, &options).unwrap();
    let v4_path = dir.join("legacy_v4.sfmr");
    rewrite_sfmr_version_4(&staged, &v4_path);
    assert_eq!(
        sfmr_format::read_sfmr_metadata(&v4_path).unwrap().version,
        4
    );

    // Load applies the COLMAP→canonical upgrade: the result matches the
    // canonical ground truth and reports the current version.
    let loaded = SfmrReconstruction::load(&v4_path).unwrap();
    assert_eq!(loaded.metadata.version, sfmr_format::SFMR_FORMAT_VERSION);
    assert_eq!(loaded.images.len(), recon.images.len());
    for (li, ri) in loaded.images.iter().zip(&recon.images) {
        assert_rotations_close(&li.quaternion_wxyz, &ri.quaternion_wxyz, &li.name);
        for k in 0..3 {
            assert!(
                (li.translation_xyz[k] - ri.translation_xyz[k]).abs() < 1e-9,
                "{}: translation[{k}]",
                li.name
            );
        }
    }
    for (lp, rp) in loaded.points.iter().zip(&recon.points) {
        for k in 0..3 {
            assert!((lp.position[k] - rp.position[k]).abs() < 1e-9);
            assert!((lp.normal[k] - rp.normal[k]).abs() < 1e-6);
        }
        assert_eq!(lp.w, rp.w);
    }

    // Saving writes a v5 file that reloads unchanged (no double conversion).
    let saved = dir.join("upgraded_v5.sfmr");
    loaded.save(&saved).unwrap();
    assert_eq!(
        sfmr_format::read_sfmr_metadata(&saved).unwrap().version,
        sfmr_format::SFMR_FORMAT_VERSION
    );
    let reloaded = SfmrReconstruction::load(&saved).unwrap();
    for (li, ri) in reloaded.images.iter().zip(&loaded.images) {
        assert_rotations_close(&li.quaternion_wxyz, &ri.quaternion_wxyz, &li.name);
        for k in 0..3 {
            assert!((li.translation_xyz[k] - ri.translation_xyz[k]).abs() < 1e-12);
        }
    }
    for (lp, rp) in reloaded.points.iter().zip(&loaded.points) {
        for k in 0..3 {
            assert!((lp.position[k] - rp.position[k]).abs() < 1e-12);
        }
        assert_eq!(lp.w, rp.w);
    }

    std::fs::remove_dir_all(&dir).ok();
}
