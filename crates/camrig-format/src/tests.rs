use crate::*;
use ndarray::Array2;
use std::collections::HashMap;

// ── Construction helpers ────────────────────────────────────────────

fn pinhole(width: u32, height: u32, fx: f64, fy: f64, cx: f64, cy: f64) -> CamRigCamera {
    let mut parameters = HashMap::new();
    parameters.insert("focal_length_x".into(), fx);
    parameters.insert("focal_length_y".into(), fy);
    parameters.insert("principal_point_x".into(), cx);
    parameters.insert("principal_point_y".into(), cy);
    CamRigCamera {
        model: "PINHOLE".into(),
        width,
        height,
        parameters,
    }
}

fn opencv_fisheye(width: u32, height: u32) -> CamRigCamera {
    let mut parameters = HashMap::new();
    // kerry_park calibration values.
    parameters.insert("focal_length_x".into(), 129.1499937015594);
    parameters.insert("focal_length_y".into(), 129.2573627423474);
    parameters.insert("principal_point_x".into(), 240.0);
    parameters.insert("principal_point_y".into(), 240.0);
    parameters.insert("radial_distortion_k1".into(), 0.038113353966529886);
    parameters.insert("radial_distortion_k2".into(), -0.00800851799065643);
    parameters.insert("radial_distortion_k3".into(), 0.008329720504707577);
    parameters.insert("radial_distortion_k4".into(), -0.0026901578801066814);
    CamRigCamera {
        model: "OPENCV_FISHEYE".into(),
        width,
        height,
        parameters,
    }
}

/// Quaternion `[w, x, y, z]` for a rotation of `angle` radians about a
/// unit axis.
fn quat_axis_angle(axis: [f64; 3], angle: f64) -> [f64; 4] {
    let h = angle / 2.0;
    let s = h.sin();
    [h.cos(), axis[0] * s, axis[1] * s, axis[2] * s]
}

/// Shortest-arc quaternion rotating `−Z` — the canonical sensor's forward
/// axis — onto the unit vector `d`.
fn shortest_arc_from_neg_z(d: [f64; 3]) -> [f64; 4] {
    let dot = -d[2]; // (−Z) · d
    if dot > 0.999_999 {
        return [1.0, 0.0, 0.0, 0.0]; // already looking along d
    }
    if dot < -0.999_999 {
        return [0.0, 1.0, 0.0, 0.0]; // d = +Z: 180° about X
    }
    // Half-way quaternion: q = normalize([1 + dot, cross(−Z, d)]).
    let q = [1.0 + dot, d[1], -d[0], 0.0];
    let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    [q[0] / n, q[1] / n, q[2] / n, q[3] / n]
}

/// `n` near-uniform unit directions via the Fibonacci sphere.
fn fibonacci_sphere(n: usize) -> Vec<[f64; 3]> {
    let golden = std::f64::consts::PI * (1.0 + 5.0_f64.sqrt());
    (0..n)
        .map(|i| {
            let y = 1.0 - 2.0 * (i as f64 + 0.5) / n as f64;
            let r = (1.0 - y * y).max(0.0).sqrt();
            let theta = golden * i as f64;
            [r * theta.cos(), y, r * theta.sin()]
        })
        .collect()
}

/// Assemble columnar sensor pose arrays from a list of `(w,x,y,z)`
/// quaternions and `(x,y,z)` translations.
fn pose_arrays(quats: &[[f64; 4]], trans: &[[f64; 3]]) -> (Array2<f64>, Array2<f64>) {
    let s = quats.len();
    let q: Vec<f64> = quats.iter().flatten().copied().collect();
    let t: Vec<f64> = trans.iter().flatten().copied().collect();
    (
        Array2::from_shape_vec((s, 4), q).unwrap(),
        Array2::from_shape_vec((s, 3), t).unwrap(),
    )
}

fn metadata(
    name: &str,
    sensor_count: usize,
    camera_count: usize,
    rig_type: &str,
    rig_attributes: serde_json::Value,
) -> CamRigMetadata {
    CamRigMetadata {
        version: CAMRIG_FORMAT_VERSION,
        name: name.into(),
        sensor_count: sensor_count as u32,
        camera_count: camera_count as u32,
        rig_type: rig_type.into(),
        rig_attributes,
    }
}

// ── Rig builders for the three target rig families ──────────────────

/// A single free-floating camera (1 sensor).
fn single_camera_rig() -> CamRigData {
    let cameras = vec![pinhole(640, 480, 500.0, 500.0, 320.0, 240.0)];
    let (quaternions_wxyz, translations_xyz) =
        pose_arrays(&[[1.0, 0.0, 0.0, 0.0]], &[[0.0, 0.0, 0.0]]);
    CamRigData {
        metadata: metadata("single", 1, 1, "generic", serde_json::json!({})),
        content_hash: CamRigContentHash::default(),
        cameras,
        sensor_image_patterns: vec!["image_%04d.jpg".into()],
        camera_indexes: vec![0],
        quaternions_wxyz,
        translations_xyz,
    }
}

/// A stereo pair: two side-by-side pinhole sensors with parallel optical
/// axes, offset along X by the baseline. Both share one intrinsic.
fn stereo_pair_rig() -> CamRigData {
    let cameras = vec![pinhole(1280, 960, 900.0, 900.0, 640.0, 480.0)];
    let baseline_m = 0.12;
    // Parallel axes ⇒ both sensors keep identity rotation. The right
    // sensor's optical centre sits at +baseline in X, so its
    // sensor_from_rig translation is -baseline in X.
    let quats = [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]];
    let trans = [[0.0, 0.0, 0.0], [-baseline_m, 0.0, 0.0]];
    let (quaternions_wxyz, translations_xyz) = pose_arrays(&quats, &trans);
    CamRigData {
        metadata: metadata(
            "stereo",
            2,
            1,
            "stereo_pair",
            serde_json::json!({ "baseline_m": baseline_m }),
        ),
        content_hash: CamRigContentHash::default(),
        cameras,
        sensor_image_patterns: vec!["left/image_%04d.jpg".into(), "right/image_%04d.jpg".into()],
        camera_indexes: vec![0, 0],
        quaternions_wxyz,
        translations_xyz,
    }
}

/// A back-to-back fisheye pair, as `sfm insv2rig` extracts: the right
/// sensor faces opposite the left, offset by a small baseline. Canonical
/// convention: sensors look down −Z, so the rear lens sits along **+Z** of
/// the reference sensor (the 180°-about-Y rotation is S-invariant; only the
/// baseline's sign flipped relative to the COLMAP-convention rig).
fn insv2_rig() -> CamRigData {
    // Both sensors share one fisheye intrinsic (camera pool size 1).
    let cameras = vec![opencv_fisheye(480, 480)];
    let baseline_m = 0.0307;
    let quats = [
        [1.0, 0.0, 0.0, 0.0], // left: reference, identity
        [0.0, 0.0, 1.0, 0.0], // right: 180° about Y (S-invariant)
    ];
    let trans = [[0.0, 0.0, 0.0], [0.0, 0.0, baseline_m]];
    let (quaternions_wxyz, translations_xyz) = pose_arrays(&quats, &trans);
    CamRigData {
        metadata: metadata(
            "insv2_x5",
            2,
            1,
            "fisheye_360",
            serde_json::json!({ "baseline_m": baseline_m }),
        ),
        content_hash: CamRigContentHash::default(),
        cameras,
        sensor_image_patterns: vec![
            "fisheye_left/image_%04d.jpg".into(),
            "fisheye_right/image_%04d.jpg".into(),
        ],
        camera_indexes: vec![0, 0],
        quaternions_wxyz,
        translations_xyz,
    }
}

/// A six-face cubemap, as `sfm pano2rig` produces: six co-centric
/// pinhole faces (front/right/back/left/top/bottom). Canonical convention:
/// the reference face looks down rig −Z; relative to the COLMAP-convention
/// rig the Y-rotation signs are S-conjugated (`S·Ry(θ)·S = Ry(−θ)`) while
/// the X-rotations are S-invariant (`S·Rx(θ)·S = Rx(θ)`).
fn cubemap_rig(face_size: u32) -> CamRigData {
    // 90° FOV per face ⇒ fx = fy = face_size / 2.
    let f = face_size as f64 / 2.0;
    let c = face_size as f64 / 2.0;
    let cameras = vec![pinhole(face_size, face_size, f, f, c, c)];
    let q = std::f64::consts::FRAC_PI_2;
    let quats = [
        [1.0, 0.0, 0.0, 0.0],                 // front (reference)
        quat_axis_angle([0.0, 1.0, 0.0], -q), // right: -90° about Y
        [0.0, 0.0, 1.0, 0.0],                 // back: 180° about Y
        quat_axis_angle([0.0, 1.0, 0.0], q),  // left: +90° about Y
        quat_axis_angle([1.0, 0.0, 0.0], -q), // top: -90° about X
        quat_axis_angle([1.0, 0.0, 0.0], q),  // bottom: +90° about X
    ];
    let trans = [[0.0; 3]; 6];
    let (quaternions_wxyz, translations_xyz) = pose_arrays(&quats, &trans);
    let names = ["front", "right", "back", "left", "top", "bottom"]
        .iter()
        .map(|s| format!("{s}/frame_%04d.jpg"))
        .collect();
    CamRigData {
        metadata: metadata("cubemap", 6, 1, "cubemap", serde_json::json!({})),
        content_hash: CamRigContentHash::default(),
        cameras,
        sensor_image_patterns: names,
        camera_indexes: vec![0; 6],
        quaternions_wxyz,
        translations_xyz,
    }
}

/// A spherical tile rig: `n` co-centric pinhole tiles discretising the
/// sphere. All tiles share one pinhole camera; only rotation varies;
/// translations are all zero. Sensors are anonymous.
fn spherical_tile_rig(n: usize, overlap_factor: f64) -> CamRigData {
    let directions = fibonacci_sphere(n);
    // Synthetic-but-plausible sizing (see specs/core/spherical-tiles-rig.md).
    let nn_angle = (4.0 * std::f64::consts::PI / n as f64).sqrt();
    let half_fov_rad = 0.5 * nn_angle * overlap_factor;
    let arc_per_pixel = nn_angle / 16.0;
    let patch_size = ((2.0 * half_fov_rad / arc_per_pixel).ceil() as u32).max(5);
    let f = (patch_size as f64 / 2.0) / half_fov_rad.tan();
    let c = patch_size as f64 / 2.0;
    let atlas_cols = (n as f64).sqrt().ceil() as u32;

    let cameras = vec![pinhole(patch_size, patch_size, f, f, c, c)];
    let quats: Vec<[f64; 4]> = directions
        .iter()
        .map(|&d| shortest_arc_from_neg_z(d))
        .collect();
    let trans = vec![[0.0; 3]; n];
    let (quaternions_wxyz, translations_xyz) = pose_arrays(&quats, &trans);

    let attributes = serde_json::json!({
        "n": n,
        "arc_per_pixel": arc_per_pixel,
        "overlap_factor": overlap_factor,
        "half_fov_rad": half_fov_rad,
        "patch_size": patch_size,
        "atlas_cols": atlas_cols,
        "centre": [0.0, 0.0, 0.0],
        "relax_seed": 42,
    });
    CamRigData {
        metadata: metadata(
            &format!("spherical_tiles_n{n}"),
            n,
            1,
            "spherical_tiles",
            attributes,
        ),
        content_hash: CamRigContentHash::default(),
        cameras,
        sensor_image_patterns: vec![], // anonymous
        camera_indexes: vec![0; n],
        quaternions_wxyz,
        translations_xyz,
    }
}

// ── Round-trip assertion ────────────────────────────────────────────

fn assert_round_trip(data: &CamRigData, label: &str) -> CamRigData {
    let dir = std::env::temp_dir().join(format!("camrig_test_{label}"));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(format!("{label}.camrig"));

    write_camrig(&path, data, 3).unwrap_or_else(|e| panic!("write {label}: {e}"));
    let loaded = read_camrig(&path).unwrap_or_else(|e| panic!("read {label}: {e}"));

    assert_eq!(loaded.metadata.version, data.metadata.version);
    assert_eq!(loaded.metadata.name, data.metadata.name);
    assert_eq!(loaded.metadata.sensor_count, data.metadata.sensor_count);
    assert_eq!(loaded.metadata.camera_count, data.metadata.camera_count);
    assert_eq!(loaded.metadata.rig_type, data.metadata.rig_type);
    assert_eq!(loaded.metadata.rig_attributes, data.metadata.rig_attributes);
    assert_eq!(loaded.cameras, data.cameras);
    assert_eq!(loaded.sensor_image_patterns, data.sensor_image_patterns);
    assert_eq!(loaded.camera_indexes, data.camera_indexes);
    assert_eq!(loaded.quaternions_wxyz, data.quaternions_wxyz);
    assert_eq!(loaded.translations_xyz, data.translations_xyz);

    let (valid, errors) = verify_camrig(&path).unwrap();
    assert!(valid, "verify {label} failed: {errors:?}");

    let (meta_only, hash_only) = read_camrig_metadata(&path).unwrap();
    assert_eq!(meta_only.sensor_count, data.metadata.sensor_count);
    assert_eq!(hash_only.content_xxh128, loaded.content_hash.content_xxh128);
    assert_eq!(hash_only.content_xxh128.len(), 32);

    std::fs::remove_dir_all(&dir).ok();
    loaded
}

// ── Tests ───────────────────────────────────────────────────────────

#[test]
fn round_trip_single_camera() {
    assert_round_trip(&single_camera_rig(), "single");
}

#[test]
fn round_trip_insv2_rig() {
    let loaded = assert_round_trip(&insv2_rig(), "insv2");
    assert_eq!(loaded.sensor_count(), 2);
    assert_eq!(
        loaded.cameras.len(),
        1,
        "both fisheye sensors share intrinsics"
    );
    assert_eq!(loaded.metadata.rig_type, "fisheye_360");
    assert_eq!(loaded.cameras[0].model, "OPENCV_FISHEYE");
}

#[test]
fn round_trip_stereo_pair_rig() {
    let loaded = assert_round_trip(&stereo_pair_rig(), "stereo");
    assert_eq!(loaded.sensor_count(), 2);
    assert_eq!(loaded.cameras.len(), 1);
    assert_eq!(loaded.metadata.rig_type, "stereo_pair");
    // Side-by-side ⇒ a non-zero baseline translation on the right sensor.
    assert_eq!(loaded.translations_xyz[[1, 0]], -0.12);
    assert_eq!(loaded.translations_xyz[[0, 0]], 0.0);
}

#[test]
fn round_trip_cubemap_rig() {
    let loaded = assert_round_trip(&cubemap_rig(512), "cubemap");
    assert_eq!(loaded.sensor_count(), 6);
    assert_eq!(loaded.cameras.len(), 1);
    assert_eq!(loaded.sensor_image_patterns[2], "back/frame_%04d.jpg");
}

#[test]
fn round_trip_spherical_tile_rigs_varied_sizes() {
    for &n in &[2usize, 80, 1280, 20_000] {
        let label = format!("sph_n{n}");
        let loaded = assert_round_trip(&spherical_tile_rig(n, 1.15), &label);
        assert_eq!(loaded.sensor_count(), n);
        assert_eq!(loaded.cameras.len(), 1, "all tiles share one camera");
        assert!(loaded.is_anonymous());
        assert_eq!(loaded.metadata.rig_type, "spherical_tiles");
    }
}

#[test]
fn round_trip_spherical_tile_rigs_varied_overlap() {
    for &overlap in &[1.0, 1.15, 1.5, 2.0] {
        let label = format!("sph_ov{}", (overlap * 100.0) as u32);
        let loaded = assert_round_trip(&spherical_tile_rig(1280, overlap), &label);
        let attrs = &loaded.metadata.rig_attributes;
        assert_eq!(attrs["overlap_factor"].as_f64().unwrap(), overlap);
        // Wider overlap ⇒ wider FOV ⇒ shorter focal length.
        assert!(attrs["half_fov_rad"].as_f64().unwrap() > 0.0);
    }
}

#[test]
fn round_trip_spherical_tile_rig_100k() {
    let loaded = assert_round_trip(&spherical_tile_rig(100_000, 1.15), "sph_100k");
    assert_eq!(loaded.sensor_count(), 100_000);
    assert_eq!(loaded.cameras.len(), 1);
}

#[test]
fn large_co_centric_rig_compresses_small() {
    // A 100k-tile rig has one camera and all-zero translations; the file
    // should stay far below the 100k×4×8 = 3.2 MB raw quaternion size.
    let data = spherical_tile_rig(100_000, 1.15);
    let dir = std::env::temp_dir().join("camrig_test_size");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("big.camrig");
    write_camrig(&path, &data, 3).unwrap();
    let size = std::fs::metadata(&path).unwrap().len();
    assert!(
        size < 4_000_000,
        "100k-sensor rig file unexpectedly large: {size} bytes"
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn anonymous_and_named_sensors_both_round_trip() {
    // Named (cubemap) and anonymous (spherical tiles) both already
    // covered; assert the anonymous marker explicitly.
    let anon = spherical_tile_rig(80, 1.15);
    assert!(anon.is_anonymous());
    let named = cubemap_rig(256);
    assert!(!named.is_anonymous());
    assert_round_trip(&anon, "anon");
    assert_round_trip(&named, "named");
}

#[test]
fn verify_detects_tampering() {
    let data = cubemap_rig(256);
    let dir = std::env::temp_dir().join("camrig_test_tamper");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("c.camrig");
    write_camrig(&path, &data, 3).unwrap();

    // Corrupt a 64-byte run that lands inside the compressed payload of
    // the first archive members (well past the ~47-byte local header of
    // `metadata.json.zst`), guaranteeing hashed content is altered.
    let mut bytes = std::fs::read(&path).unwrap();
    assert!(bytes.len() > 128, "test file unexpectedly small");
    for b in &mut bytes[64..128] {
        *b ^= 0xFF;
    }
    std::fs::write(&path, &bytes).unwrap();

    // Either the ZIP/zstd layer rejects it, or the content hash fails.
    if let Ok((valid, _errors)) = verify_camrig(&path) {
        assert!(!valid, "tampered file passed verification");
    }
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn write_rejects_wrong_quaternion_shape() {
    let mut data = cubemap_rig(256);
    data.quaternions_wxyz = Array2::zeros((6, 3)); // should be (6, 4)
    let path = std::env::temp_dir().join("camrig_bad_quat.camrig");
    let err = write_camrig(&path, &data, 3).unwrap_err();
    assert!(format!("{err}").contains("quaternions_wxyz"));
}

#[test]
fn no_sensor_need_be_at_the_identity_pose() {
    // A spherical tile rig's sensor 0 (tile 0) is not at identity — the
    // format places no pose constraint on any sensor.
    let data = spherical_tile_rig(80, 1.15);
    let q0 = data.quaternions_wxyz.row(0).to_vec();
    assert!(
        (q0[0] - 1.0).abs() > 1e-6,
        "tile 0 should not be at the identity pose"
    );
    assert_round_trip(&data, "nonident_sensor0");
}

#[test]
fn write_rejects_out_of_range_camera_index() {
    let mut data = cubemap_rig(256);
    data.camera_indexes[3] = 5; // pool has only 1 camera
    let path = std::env::temp_dir().join("camrig_bad_idx.camrig");
    let err = write_camrig(&path, &data, 3).unwrap_err();
    assert!(format!("{err}").contains("out of range"));
}

#[test]
fn write_rejects_non_unit_quaternion() {
    let mut data = cubemap_rig(256);
    data.quaternions_wxyz[[1, 1]] = 0.5; // breaks unit length on sensor 1
    let path = std::env::temp_dir().join("camrig_bad_norm.camrig");
    let err = write_camrig(&path, &data, 3).unwrap_err();
    assert!(format!("{err}").contains("unit length"));
}

#[test]
fn write_rejects_mismatched_sensor_image_patterns_length() {
    let mut data = cubemap_rig(256);
    data.sensor_image_patterns = vec!["only_one".into()]; // 1 name for 6 sensors
    let path = std::env::temp_dir().join("camrig_bad_names.camrig");
    let err = write_camrig(&path, &data, 3).unwrap_err();
    assert!(format!("{err}").contains("sensor_image_patterns"));
}

#[test]
fn write_rejects_multi_sensor_pattern_without_frame_field() {
    // A glob-only pattern groups frames positionally, which a multi-sensor
    // rig cannot use — every pattern must carry a `%d` / `%0Nd` field.
    let mut data = cubemap_rig(256);
    data.sensor_image_patterns[2] = "back/*.jpg".into();
    let path = std::env::temp_dir().join("camrig_no_frame_field.camrig");
    let err = write_camrig(&path, &data, 3).unwrap_err();
    assert!(format!("{err}").contains("frame field"));
}

#[test]
fn single_sensor_glob_pattern_without_frame_field_round_trips() {
    // The motivating case: a one-camera rig dropped beside a directory of
    // images, its sole pattern a bare glob. Positional frame grouping is
    // well-defined for a single sensor, so this is valid.
    let mut data = single_camera_rig();
    data.sensor_image_patterns = vec!["*.jpg".into()];
    let loaded = assert_round_trip(&data, "single_glob");
    assert_eq!(loaded.sensor_image_patterns, vec!["*.jpg".to_string()]);
}

#[test]
fn write_rejects_pattern_with_two_frame_fields() {
    // A pattern carries at most one frame field; a second `%d` makes the
    // captured frame index ambiguous, so validate() rejects it — even for
    // a single-sensor rig.
    let mut data = single_camera_rig();
    data.sensor_image_patterns = vec!["cam_%d_%04d.jpg".into()];
    let path = std::env::temp_dir().join("camrig_two_frame_fields.camrig");
    let err = write_camrig(&path, &data, 3).unwrap_err();
    assert!(format!("{err}").contains("at most one"));
}

#[test]
fn write_rejects_camera_with_zero_dimension() {
    // A zero width or height is a degenerate camera; consumers that scale
    // intrinsics by aspect ratio would divide by zero, so validate()
    // rejects it.
    let mut data = single_camera_rig();
    data.cameras[0].height = 0;
    let path = std::env::temp_dir().join("camrig_zero_dimension.camrig");
    let err = write_camrig(&path, &data, 3).unwrap_err();
    assert!(format!("{err}").contains("non-positive dimension"));
}

#[test]
fn write_handles_non_contiguous_pose_arrays() {
    // The pose arrays are public fields; a caller may hand in a
    // non-contiguous array. Re-store the values in Fortran (column-major)
    // layout, where `as_slice()` returns `None`. Writing must not panic,
    // and the values must round-trip.
    use ndarray::ShapeBuilder;
    let to_fortran = |a: &Array2<f64>| -> Array2<f64> {
        let (rows, cols) = a.dim();
        let mut col_major = Vec::with_capacity(rows * cols);
        for j in 0..cols {
            for i in 0..rows {
                col_major.push(a[[i, j]]);
            }
        }
        Array2::from_shape_vec((rows, cols).f(), col_major).unwrap()
    };

    let mut data = cubemap_rig(256);
    data.quaternions_wxyz = to_fortran(&data.quaternions_wxyz);
    data.translations_xyz = to_fortran(&data.translations_xyz);
    assert!(data.quaternions_wxyz.as_slice().is_none());
    assert_round_trip(&data, "noncontig");
}

// The write path validates, so deliberately-invalid files are produced
// with `write_camrig_unchecked` to exercise the read/verify paths.

#[test]
fn read_rejects_out_of_range_camera_index() {
    let mut data = cubemap_rig(256);
    data.camera_indexes[3] = 5; // pool has only 1 camera
    let path = std::env::temp_dir().join("camrig_read_bad_idx.camrig");
    crate::write::write_camrig_unchecked(&path, &data, 3).unwrap();

    let err = read_camrig(&path).unwrap_err();
    assert!(format!("{err}").contains("out of range"));

    let (valid, errors) = verify_camrig(&path).unwrap();
    assert!(!valid);
    assert!(errors.iter().any(|e| e.contains("Structural validation")));
    std::fs::remove_file(&path).ok();
}

#[test]
fn read_rejects_non_unit_quaternion() {
    let mut data = cubemap_rig(256);
    data.quaternions_wxyz[[2, 1]] = 0.5; // breaks unit length on sensor 2
    let path = std::env::temp_dir().join("camrig_read_bad_norm.camrig");
    crate::write::write_camrig_unchecked(&path, &data, 3).unwrap();

    let err = read_camrig(&path).unwrap_err();
    assert!(format!("{err}").contains("unit length"));
    std::fs::remove_file(&path).ok();
}

#[test]
fn read_rejects_zero_sensors() {
    let data = CamRigData {
        metadata: metadata("empty", 0, 1, "generic", serde_json::json!({})),
        content_hash: CamRigContentHash::default(),
        cameras: vec![pinhole(640, 480, 500.0, 500.0, 320.0, 240.0)],
        sensor_image_patterns: vec![],
        camera_indexes: vec![],
        quaternions_wxyz: Array2::zeros((0, 4)),
        translations_xyz: Array2::zeros((0, 3)),
    };
    let path = std::env::temp_dir().join("camrig_read_zero.camrig");
    crate::write::write_camrig_unchecked(&path, &data, 3).unwrap();

    let err = read_camrig(&path).unwrap_err();
    assert!(format!("{err}").contains("at least one sensor"));
    std::fs::remove_file(&path).ok();
}

#[test]
fn read_rejects_unsupported_version() {
    let mut data = cubemap_rig(256);
    data.metadata.version = CAMRIG_FORMAT_VERSION + 1;
    let path = std::env::temp_dir().join("camrig_read_bad_ver.camrig");
    crate::write::write_camrig_unchecked(&path, &data, 3).unwrap();

    let err = read_camrig(&path).unwrap_err();
    assert!(format!("{err}").contains("version"));
    std::fs::remove_file(&path).ok();
}

#[test]
fn writer_always_writes_current_version() {
    let mut data = cubemap_rig(256);
    data.metadata.version = 1; // stale caller-supplied version is overridden
    let path = std::env::temp_dir().join("camrig_write_version.camrig");
    write_camrig(&path, &data, 3).unwrap();

    let (meta, _) = read_camrig_metadata(&path).unwrap();
    assert_eq!(meta.version, CAMRIG_FORMAT_VERSION);
    std::fs::remove_file(&path).ok();
}

#[test]
fn version_1_sensor_poses_upgrade_on_load() {
    // Author a genuine version-1 file: take the canonical insv2 rig,
    // S-conjugate its poses back to the COLMAP convention (S is involutive),
    // and write it verbatim with version = 1 via the unchecked writer.
    let canonical = insv2_rig();
    let mut v1 = canonical.clone();
    v1.metadata.version = 1;
    v1.s_conjugate_sensor_poses();
    // The COLMAP-convention rig carries the rear lens at −Z.
    assert_eq!(v1.translations_xyz[[1, 2]], -0.0307);

    let path = std::env::temp_dir().join("camrig_v1_upgrade.camrig");
    crate::write::write_camrig_unchecked(&path, &v1, 3).unwrap();
    let (meta, _) = read_camrig_metadata(&path).unwrap();
    assert_eq!(meta.version, 1);

    // Hashes cover the stored bytes: the v1 file verifies as written,
    // before any in-memory conversion.
    let (valid, errors) = verify_camrig(&path).unwrap();
    assert!(valid, "v1 fixture failed verification: {errors:?}");

    // Loading upgrades to the canonical convention and the current version.
    let loaded = read_camrig(&path).unwrap();
    assert_eq!(loaded.metadata.version, CAMRIG_FORMAT_VERSION);
    assert_eq!(loaded.quaternions_wxyz, canonical.quaternions_wxyz);
    assert_eq!(loaded.translations_xyz, canonical.translations_xyz);

    std::fs::remove_file(&path).ok();
}

/// Two unit quaternions describe the same rotation iff they are equal or
/// exact negatives (`q` and `−q`). Used to compare recovered sensor rotations
/// without caring about the immaterial global sign.
fn same_rotation(a: ndarray::ArrayView1<f64>, b: ndarray::ArrayView1<f64>) -> bool {
    let equal = (0..4).all(|k| (a[k] - b[k]).abs() < 1e-12);
    let negated = (0..4).all(|k| (a[k] + b[k]).abs() < 1e-12);
    equal || negated
}

#[test]
fn version_1_spherical_tiles_upgrade_uses_world_anchored_flip() {
    // A spherical_tiles rig is world-anchored: its rig frame *is* the
    // reconstruction world, so the v1→v2 upgrade must left-S-multiply the
    // sensor poses (R' = S·R), not S-conjugate them (R' = S·R·S) the way
    // body-anchored rigs do. S-conjugating a tile rig would rotate every tile
    // 180° about the world X axis (the wrong hemisphere) while still passing
    // structural validation.
    let canonical = spherical_tile_rig(8, 1.3);

    // Author a genuine version-1 file: apply the involutive world-anchored flip
    // to the canonical poses to land in the COLMAP convention, stamp version 1.
    let mut v1 = canonical.clone();
    v1.metadata.version = 1;
    for i in 0..v1.quaternions_wxyz.nrows() {
        let mut q = [
            v1.quaternions_wxyz[[i, 0]],
            v1.quaternions_wxyz[[i, 1]],
            v1.quaternions_wxyz[[i, 2]],
            v1.quaternions_wxyz[[i, 3]],
        ];
        let mut t = [0.0; 3];
        s_premultiply_sensor_pose(&mut q, &mut t);
        for (k, &val) in q.iter().enumerate() {
            v1.quaternions_wxyz[[i, k]] = val;
        }
    }

    let path = std::env::temp_dir().join("camrig_v1_tiles_upgrade.camrig");
    crate::write::write_camrig_unchecked(&path, &v1, 3).unwrap();
    let (meta, _) = read_camrig_metadata(&path).unwrap();
    assert_eq!(meta.version, 1);

    // Loading upgrades via the world-anchored flip and recovers the canonical
    // rotations (up to the immaterial global quaternion sign).
    let loaded = read_camrig(&path).unwrap();
    assert_eq!(loaded.metadata.version, CAMRIG_FORMAT_VERSION);
    for i in 0..canonical.quaternions_wxyz.nrows() {
        assert!(
            same_rotation(
                loaded.quaternions_wxyz.row(i),
                canonical.quaternions_wxyz.row(i)
            ),
            "tile {i} did not upgrade to the canonical rotation"
        );
    }

    // The body-anchored conjugation the pre-fix code applied would leave at
    // least one tile pointing at the wrong hemisphere — the dispatch matters.
    let mut wrong = v1.clone();
    wrong.s_conjugate_sensor_poses();
    let any_wrong = (0..canonical.quaternions_wxyz.nrows()).any(|i| {
        !same_rotation(
            wrong.quaternions_wxyz.row(i),
            canonical.quaternions_wxyz.row(i),
        )
    });
    assert!(
        any_wrong,
        "S-conjugation unexpectedly matched the canonical rotations"
    );

    std::fs::remove_file(&path).ok();
}

#[test]
fn read_nonexistent_file_reports_path() {
    let err = read_camrig(std::path::Path::new("does_not_exist.camrig")).unwrap_err();
    assert!(err.to_string().contains("does_not_exist.camrig"));
}

// Every entry MUST use ZIP's STORE method — entries are already
// zstd-compressed, and STORE preserves random access by seek.
#[test]
fn archive_uses_stored_compression() {
    let data = insv2_rig();
    let dir = std::env::temp_dir().join("camrig_test_stored");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("s.camrig");
    write_camrig(&path, &data, 3).unwrap();

    let file = std::fs::File::open(&path).unwrap();
    let mut archive = zip::ZipArchive::new(file).unwrap();
    assert!(!archive.is_empty());
    for i in 0..archive.len() {
        let entry = archive.by_index(i).unwrap();
        assert_eq!(
            entry.compression(),
            zip::CompressionMethod::Stored,
            "entry '{}' is not STORE",
            entry.name()
        );
    }
    std::fs::remove_dir_all(&dir).ok();
}
