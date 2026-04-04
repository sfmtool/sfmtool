// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Read COLMAP binary reconstruction files (cameras.bin, images.bin, points3D.bin).

use std::collections::HashMap;
use std::io::{BufReader, Read};
use std::path::Path;

use sfmr_format::SfmrCamera;

use super::types::{
    camera_params_from_array, colmap_model_name, colmap_num_params, ColmapDataId, ColmapFrame,
    ColmapIoError, ColmapReconstruction, ColmapRig, ColmapRigSensor, ColmapSensor,
    ColmapSensorType, Keypoint2D,
};

/// Sentinel value for unobserved 3D point references in COLMAP binary format.
const INVALID_POINT3D_ID: u64 = u64::MAX;

fn open_file(path: &Path) -> Result<std::fs::File, ColmapIoError> {
    std::fs::File::open(path).map_err(|e| ColmapIoError::IoPath {
        operation: "Failed to open file",
        path: path.to_path_buf(),
        source: e,
    })
}

/// Read a complete COLMAP binary reconstruction from a directory containing
/// `cameras.bin`, `images.bin`, and `points3D.bin`. Optionally reads `rigs.bin`
/// and `frames.bin` if present.
///
/// All COLMAP IDs are remapped to 0-based indices: camera IDs become sequential
/// camera indexes, image IDs become name-sorted image indexes, and images are
/// returned sorted by name. Rig sensor IDs and frame data IDs are remapped
/// accordingly.
///
/// Only camera sensors are supported. Non-camera sensors (e.g. IMU) are silently
/// dropped from rigs and frame data_ids.
pub fn read_colmap_binary(dir: &Path) -> Result<ColmapReconstruction, ColmapIoError> {
    let (cameras, camera_id_to_index) = read_cameras_bin(&dir.join("cameras.bin"))?;
    let raw_images = read_images_bin(&dir.join("images.bin"))?;
    let (positions, colors, errors, tracks, point3d_id_map) =
        read_points3d_bin(&dir.join("points3D.bin"))?;

    // Sort images by name for consistent ordering
    let mut sorted_indices: Vec<usize> = (0..raw_images.len()).collect();
    sorted_indices.sort_by(|&a, &b| raw_images[a].name.cmp(&raw_images[b].name));
    let num_images = sorted_indices.len();
    let mut image_names = Vec::with_capacity(num_images);
    let mut camera_indexes = Vec::with_capacity(num_images);
    let mut quaternions_wxyz = Vec::with_capacity(num_images);
    let mut translations_xyz = Vec::with_capacity(num_images);
    let mut keypoints_per_image = Vec::with_capacity(num_images);

    for &old_idx in &sorted_indices {
        let img = &raw_images[old_idx];
        image_names.push(img.name.clone());
        camera_indexes.push(*camera_id_to_index.get(&img.camera_id).ok_or_else(|| {
            ColmapIoError::InvalidData(format!(
                "Image '{}' references unknown camera_id {}",
                img.name, img.camera_id
            ))
        })?);
        quaternions_wxyz.push(img.quat_wxyz);
        translations_xyz.push(img.trans_xyz);

        // Remap point3d IDs to 0-based indices
        let kps: Vec<Keypoint2D> = img
            .keypoints
            .iter()
            .map(|kp| {
                let point3d_index = if kp.point3d_id == INVALID_POINT3D_ID {
                    None
                } else {
                    point3d_id_map.get(&kp.point3d_id).copied()
                };
                Keypoint2D {
                    x: kp.x,
                    y: kp.y,
                    point3d_index,
                }
            })
            .collect();
        keypoints_per_image.push(kps);
    }

    // Build a map from COLMAP image_id to sorted 0-based index.
    // COLMAP image IDs are arbitrary (not necessarily sequential), so we must
    // map through the actual IDs rather than assuming image_id == index + 1.
    let mut image_id_to_sorted_index: HashMap<u32, u32> = HashMap::with_capacity(num_images);
    for (sorted_idx, &old_idx) in sorted_indices.iter().enumerate() {
        let colmap_id = raw_images[old_idx].image_id;
        image_id_to_sorted_index.insert(colmap_id, sorted_idx as u32);
    }

    // Remap track image IDs from COLMAP IDs to sorted 0-based indices
    let remapped_tracks: Vec<Vec<(u32, u32)>> = tracks
        .into_iter()
        .map(|track| {
            track
                .into_iter()
                .filter_map(|(colmap_img_id, feat_idx)| {
                    // Filter out track entries referencing images not in this
                    // reconstruction (can happen with partial reconstructions).
                    image_id_to_sorted_index
                        .get(&colmap_img_id)
                        .map(|&new_idx| (new_idx, feat_idx))
                })
                .collect()
        })
        .collect();

    // Optionally read rigs.bin and frames.bin, remapping IDs to 0-based indexes
    // and dropping non-camera sensors.
    let rigs_path = dir.join("rigs.bin");
    let rigs = if rigs_path.exists() {
        let mut raw_rigs = read_rigs_bin(&rigs_path)?;
        for rig in &mut raw_rigs {
            // Drop non-camera ref sensor
            if let Some(ref ref_sensor) = rig.ref_sensor {
                if ref_sensor.sensor_type != ColmapSensorType::Camera {
                    rig.ref_sensor = None;
                }
            }
            // Remap ref sensor camera_id to 0-based camera index
            if let Some(ref mut ref_sensor) = rig.ref_sensor {
                if let Some(&idx) = camera_id_to_index.get(&ref_sensor.id) {
                    ref_sensor.id = idx;
                }
            }
            // Drop non-camera sensors and remap camera_ids
            rig.non_ref_sensors
                .retain(|s| s.sensor.sensor_type == ColmapSensorType::Camera);
            for non_ref in &mut rig.non_ref_sensors {
                if let Some(&idx) = camera_id_to_index.get(&non_ref.sensor.id) {
                    non_ref.sensor.id = idx;
                }
            }
        }
        Some(raw_rigs)
    } else {
        None
    };

    let frames_path = dir.join("frames.bin");
    let frames = if frames_path.exists() {
        let mut raw_frames = read_frames_bin(&frames_path)?;
        for frame in &mut raw_frames {
            // Drop non-camera data_ids and remap camera/image IDs
            frame
                .data_ids
                .retain(|d| d.sensor_type == ColmapSensorType::Camera);
            for data_id in &mut frame.data_ids {
                if let Some(&idx) = camera_id_to_index.get(&data_id.sensor_id) {
                    data_id.sensor_id = idx;
                }
                if let Some(&idx) = image_id_to_sorted_index.get(&(data_id.data_id as u32)) {
                    data_id.data_id = idx as u64;
                }
            }
        }
        Some(raw_frames)
    } else {
        None
    };

    Ok(ColmapReconstruction {
        cameras,
        image_names,
        camera_indexes,
        quaternions_wxyz,
        translations_xyz,
        keypoints_per_image,
        positions_xyz: positions,
        colors_rgb: colors,
        reprojection_errors: errors,
        tracks: remapped_tracks,
        rigs,
        frames,
    })
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn read_u32(r: &mut impl Read) -> Result<u32, ColmapIoError> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_i32(r: &mut impl Read) -> Result<i32, ColmapIoError> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(i32::from_le_bytes(buf))
}

fn read_u64(r: &mut impl Read) -> Result<u64, ColmapIoError> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_f64(r: &mut impl Read) -> Result<f64, ColmapIoError> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(f64::from_le_bytes(buf))
}

fn read_u8(r: &mut impl Read) -> Result<u8, ColmapIoError> {
    let mut buf = [0u8; 1];
    r.read_exact(&mut buf)?;
    Ok(buf[0])
}

/// Read a null-terminated UTF-8 string.
fn read_null_terminated_string(r: &mut impl Read) -> Result<String, ColmapIoError> {
    let mut bytes = Vec::new();
    loop {
        let b = read_u8(r)?;
        if b == 0 {
            break;
        }
        bytes.push(b);
    }
    String::from_utf8(bytes)
        .map_err(|e| ColmapIoError::InvalidData(format!("Invalid UTF-8 string: {}", e)))
}

/// Intermediate raw image data before sorting.
struct RawImage {
    image_id: u32,
    name: String,
    camera_id: u32,
    quat_wxyz: [f64; 4],
    trans_xyz: [f64; 3],
    keypoints: Vec<RawKeypoint>,
}

struct RawKeypoint {
    x: f64,
    y: f64,
    point3d_id: u64,
}

/// Read `cameras.bin`, returning cameras and a camera_id -> 0-based index map.
fn read_cameras_bin(path: &Path) -> Result<(Vec<SfmrCamera>, HashMap<u32, u32>), ColmapIoError> {
    let file = open_file(path)?;
    let mut r = BufReader::new(file);

    let num_cameras = read_u64(&mut r)?;
    let mut cameras = Vec::with_capacity(num_cameras as usize);
    let mut id_map = HashMap::with_capacity(num_cameras as usize);

    for idx in 0..num_cameras {
        let camera_id = read_u32(&mut r)?;
        let model_id = read_i32(&mut r)?;
        let width = read_u64(&mut r)?;
        let height = read_u64(&mut r)?;

        let model_name = colmap_model_name(model_id)?;
        let num_params = colmap_num_params(model_name)?;

        let mut params = Vec::with_capacity(num_params);
        for _ in 0..num_params {
            params.push(read_f64(&mut r)?);
        }

        let parameters = camera_params_from_array(model_name, &params)?;
        cameras.push(SfmrCamera {
            model: model_name.to_string(),
            width: width as u32,
            height: height as u32,
            parameters,
        });
        id_map.insert(camera_id, idx as u32);
    }

    Ok((cameras, id_map))
}

/// Read `images.bin`, returning raw images (unsorted).
fn read_images_bin(path: &Path) -> Result<Vec<RawImage>, ColmapIoError> {
    let file = open_file(path)?;
    let mut r = BufReader::new(file);

    let num_images = read_u64(&mut r)?;
    let mut images = Vec::with_capacity(num_images as usize);

    for _ in 0..num_images {
        let image_id = read_u32(&mut r)?;
        let qw = read_f64(&mut r)?;
        let qx = read_f64(&mut r)?;
        let qy = read_f64(&mut r)?;
        let qz = read_f64(&mut r)?;
        let tx = read_f64(&mut r)?;
        let ty = read_f64(&mut r)?;
        let tz = read_f64(&mut r)?;
        let camera_id = read_u32(&mut r)?;
        let name = read_null_terminated_string(&mut r)?;
        let num_points2d = read_u64(&mut r)?;

        let mut keypoints = Vec::with_capacity(num_points2d as usize);
        for _ in 0..num_points2d {
            let x = read_f64(&mut r)?;
            let y = read_f64(&mut r)?;
            let point3d_id = read_u64(&mut r)?;
            keypoints.push(RawKeypoint { x, y, point3d_id });
        }

        images.push(RawImage {
            image_id,
            name,
            camera_id,
            quat_wxyz: [qw, qx, qy, qz],
            trans_xyz: [tx, ty, tz],
            keypoints,
        });
    }

    Ok(images)
}

/// Parsed output of `points3D.bin`.
type Points3dResult = (
    Vec<[f64; 3]>,
    Vec<[u8; 3]>,
    Vec<f64>,
    Vec<Vec<(u32, u32)>>,
    HashMap<u64, u32>,
);

/// Read `points3D.bin`, returning positions, colors, errors, tracks, and a
/// point3D_id -> 0-based index map.
fn read_points3d_bin(path: &Path) -> Result<Points3dResult, ColmapIoError> {
    let file = open_file(path)?;
    let mut r = BufReader::new(file);

    let num_points = read_u64(&mut r)?;
    let mut positions = Vec::with_capacity(num_points as usize);
    let mut colors = Vec::with_capacity(num_points as usize);
    let mut errors = Vec::with_capacity(num_points as usize);
    let mut tracks = Vec::with_capacity(num_points as usize);
    let mut id_map = HashMap::with_capacity(num_points as usize);

    for idx in 0..num_points {
        let point3d_id = read_u64(&mut r)?;
        let x = read_f64(&mut r)?;
        let y = read_f64(&mut r)?;
        let z = read_f64(&mut r)?;
        let red = read_u8(&mut r)?;
        let green = read_u8(&mut r)?;
        let blue = read_u8(&mut r)?;
        let error = read_f64(&mut r)?;
        let track_length = read_u64(&mut r)?;

        let mut track = Vec::with_capacity(track_length as usize);
        for _ in 0..track_length {
            let image_id = read_u32(&mut r)?;
            let point2d_idx = read_u32(&mut r)?;
            // Store raw COLMAP image_id; remapping to 0-based index happens in
            // read_colmap_binary using the image_id -> index map.
            track.push((image_id, point2d_idx));
        }

        id_map.insert(point3d_id, idx as u32);
        positions.push([x, y, z]);
        colors.push([red, green, blue]);
        errors.push(error);
        tracks.push(track);
    }

    Ok((positions, colors, errors, tracks, id_map))
}

/// Read `rigs.bin`.
fn read_rigs_bin(path: &Path) -> Result<Vec<ColmapRig>, ColmapIoError> {
    let file = open_file(path)?;
    let mut r = BufReader::new(file);

    let num_rigs = read_u64(&mut r)?;
    let mut rigs = Vec::with_capacity(num_rigs as usize);

    for _ in 0..num_rigs {
        let rig_id = read_u32(&mut r)?;
        let num_sensors = read_u32(&mut r)?;

        let ref_sensor = if num_sensors > 0 {
            let sensor_type = ColmapSensorType::from_i32(read_i32(&mut r)?)?;
            let sensor_id = read_u32(&mut r)?;
            Some(ColmapSensor {
                sensor_type,
                id: sensor_id,
            })
        } else {
            None
        };

        let mut non_ref_sensors = Vec::new();
        if num_sensors > 1 {
            for _ in 0..num_sensors - 1 {
                let sensor_type = ColmapSensorType::from_i32(read_i32(&mut r)?)?;
                let sensor_id = read_u32(&mut r)?;
                let has_pose = read_u8(&mut r)? != 0;

                let sensor_from_rig = if has_pose {
                    let qw = read_f64(&mut r)?;
                    let qx = read_f64(&mut r)?;
                    let qy = read_f64(&mut r)?;
                    let qz = read_f64(&mut r)?;
                    let tx = read_f64(&mut r)?;
                    let ty = read_f64(&mut r)?;
                    let tz = read_f64(&mut r)?;
                    Some(([qw, qx, qy, qz], [tx, ty, tz]))
                } else {
                    None
                };

                non_ref_sensors.push(ColmapRigSensor {
                    sensor: ColmapSensor {
                        sensor_type,
                        id: sensor_id,
                    },
                    sensor_from_rig,
                });
            }
        }

        rigs.push(ColmapRig {
            rig_id,
            ref_sensor,
            non_ref_sensors,
        });
    }

    Ok(rigs)
}

/// Read `frames.bin`.
fn read_frames_bin(path: &Path) -> Result<Vec<ColmapFrame>, ColmapIoError> {
    let file = open_file(path)?;
    let mut r = BufReader::new(file);

    let num_frames = read_u64(&mut r)?;
    let mut frames = Vec::with_capacity(num_frames as usize);

    for _ in 0..num_frames {
        let frame_id = read_u32(&mut r)?;
        let rig_id = read_u32(&mut r)?;

        let qw = read_f64(&mut r)?;
        let qx = read_f64(&mut r)?;
        let qy = read_f64(&mut r)?;
        let qz = read_f64(&mut r)?;
        let tx = read_f64(&mut r)?;
        let ty = read_f64(&mut r)?;
        let tz = read_f64(&mut r)?;

        let num_data_ids = read_u32(&mut r)?;
        let mut data_ids = Vec::with_capacity(num_data_ids as usize);
        for _ in 0..num_data_ids {
            let sensor_type = ColmapSensorType::from_i32(read_i32(&mut r)?)?;
            let sensor_id = read_u32(&mut r)?;
            let data_id = read_u64(&mut r)?;
            data_ids.push(ColmapDataId {
                sensor_type,
                sensor_id,
                data_id,
            });
        }

        frames.push(ColmapFrame {
            frame_id,
            rig_id,
            quaternion_wxyz: [qw, qx, qy, qz],
            translation_xyz: [tx, ty, tz],
            data_ids,
        });
    }

    Ok(frames)
}