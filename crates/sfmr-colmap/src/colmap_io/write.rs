// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Write COLMAP binary reconstruction files (cameras.bin, images.bin, points3D.bin,
//! rigs.bin, frames.bin).

use std::collections::HashMap;
use std::io::{BufWriter, Write};
use std::path::Path;

use super::types::{
    camera_params_to_array, colmap_model_id, ColmapFrame, ColmapIoError, ColmapRig, ColmapWriteData,
};

/// Sentinel value for unobserved 3D point references in COLMAP binary format.
const INVALID_POINT3D_ID: u64 = u64::MAX;

/// Write a complete COLMAP binary reconstruction to a directory, creating
/// `cameras.bin`, `images.bin`, and `points3D.bin`.
pub fn write_colmap_binary(dir: &Path, data: &ColmapWriteData) -> Result<(), ColmapIoError> {
    validate_write_data(data)?;
    std::fs::create_dir_all(dir)?;
    write_cameras_bin(&dir.join("cameras.bin"), data)?;
    write_images_bin(&dir.join("images.bin"), data)?;
    write_points3d_bin(&dir.join("points3D.bin"), data)?;
    if let Some(rigs) = data.rigs {
        write_rigs_bin(&dir.join("rigs.bin"), rigs)?;
    }
    if let Some(frames) = data.frames {
        write_frames_bin(&dir.join("frames.bin"), frames)?;
    }
    Ok(())
}

/// Validate that all arrays in `ColmapWriteData` have consistent lengths.
fn validate_write_data(data: &ColmapWriteData) -> Result<(), ColmapIoError> {
    let n = data.image_names.len();
    if data.camera_indexes.len() != n {
        return Err(ColmapIoError::InvalidData(format!(
            "camera_indexes length {} != image_names length {}",
            data.camera_indexes.len(),
            n
        )));
    }
    if data.quaternions_wxyz.len() != n {
        return Err(ColmapIoError::InvalidData(format!(
            "quaternions_wxyz length {} != image_names length {}",
            data.quaternions_wxyz.len(),
            n
        )));
    }
    if data.translations_xyz.len() != n {
        return Err(ColmapIoError::InvalidData(format!(
            "translations_xyz length {} != image_names length {}",
            data.translations_xyz.len(),
            n
        )));
    }
    if data.keypoints_per_image.len() != n {
        return Err(ColmapIoError::InvalidData(format!(
            "keypoints_per_image length {} != image_names length {}",
            data.keypoints_per_image.len(),
            n
        )));
    }

    let p = data.positions_xyz.len();
    if data.colors_rgb.len() != p {
        return Err(ColmapIoError::InvalidData(format!(
            "colors_rgb length {} != positions_xyz length {}",
            data.colors_rgb.len(),
            p
        )));
    }
    if data.reprojection_errors.len() != p {
        return Err(ColmapIoError::InvalidData(format!(
            "reprojection_errors length {} != positions_xyz length {}",
            data.reprojection_errors.len(),
            p
        )));
    }

    let m = data.track_image_indexes.len();
    if data.track_feature_indexes.len() != m {
        return Err(ColmapIoError::InvalidData(format!(
            "track_feature_indexes length {} != track_image_indexes length {}",
            data.track_feature_indexes.len(),
            m
        )));
    }
    if data.track_point3d_indexes.len() != m {
        return Err(ColmapIoError::InvalidData(format!(
            "track_point3d_indexes length {} != track_image_indexes length {}",
            data.track_point3d_indexes.len(),
            m
        )));
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn create_file(path: &Path) -> Result<std::fs::File, ColmapIoError> {
    std::fs::File::create(path).map_err(|e| ColmapIoError::IoPath {
        operation: "Failed to create file",
        path: path.to_path_buf(),
        source: e,
    })
}

fn write_u32(w: &mut impl Write, val: u32) -> Result<(), ColmapIoError> {
    w.write_all(&val.to_le_bytes())?;
    Ok(())
}

fn write_i32(w: &mut impl Write, val: i32) -> Result<(), ColmapIoError> {
    w.write_all(&val.to_le_bytes())?;
    Ok(())
}

fn write_u64(w: &mut impl Write, val: u64) -> Result<(), ColmapIoError> {
    w.write_all(&val.to_le_bytes())?;
    Ok(())
}

fn write_f64(w: &mut impl Write, val: f64) -> Result<(), ColmapIoError> {
    w.write_all(&val.to_le_bytes())?;
    Ok(())
}

fn write_u8(w: &mut impl Write, val: u8) -> Result<(), ColmapIoError> {
    w.write_all(&[val])?;
    Ok(())
}

/// Write `cameras.bin`.
fn write_cameras_bin(path: &Path, data: &ColmapWriteData) -> Result<(), ColmapIoError> {
    let file = create_file(path)?;
    let mut w = BufWriter::new(file);

    write_u64(&mut w, data.cameras.len() as u64)?;

    for (idx, camera) in data.cameras.iter().enumerate() {
        let camera_id = (idx as u32) + 1; // 1-based
        let model_id = colmap_model_id(&camera.model)?;
        let params = camera_params_to_array(camera)?;

        write_u32(&mut w, camera_id)?;
        write_i32(&mut w, model_id)?;
        write_u64(&mut w, camera.width as u64)?;
        write_u64(&mut w, camera.height as u64)?;

        for &p in &params {
            write_f64(&mut w, p)?;
        }
    }

    w.flush()?;
    Ok(())
}

/// Write `images.bin`.
fn write_images_bin(path: &Path, data: &ColmapWriteData) -> Result<(), ColmapIoError> {
    let file = create_file(path)?;
    let mut w = BufWriter::new(file);

    let num_images = data.image_names.len();
    write_u64(&mut w, num_images as u64)?;

    // Build a mapping from (0-based image_index, 0-based feature_index) -> 0-based point3d_index
    // from the flat track arrays.
    let mut keypoint_to_point3d: HashMap<(u32, u32), u32> = HashMap::new();
    for i in 0..data.track_image_indexes.len() {
        let img_idx = data.track_image_indexes[i];
        let feat_idx = data.track_feature_indexes[i];
        let pt_idx = data.track_point3d_indexes[i];
        keypoint_to_point3d.insert((img_idx, feat_idx), pt_idx);
    }

    for img_idx in 0..num_images {
        let image_id = (img_idx as u32) + 1; // 1-based
        let q = &data.quaternions_wxyz[img_idx];
        let t = &data.translations_xyz[img_idx];

        write_u32(&mut w, image_id)?;
        write_f64(&mut w, q[0])?; // qw
        write_f64(&mut w, q[1])?; // qx
        write_f64(&mut w, q[2])?; // qy
        write_f64(&mut w, q[3])?; // qz
        write_f64(&mut w, t[0])?; // tx
        write_f64(&mut w, t[1])?; // ty
        write_f64(&mut w, t[2])?; // tz

        // camera_id is 1-based
        let camera_id = data.camera_indexes[img_idx] + 1;
        write_u32(&mut w, camera_id)?;

        // Null-terminated image name
        w.write_all(data.image_names[img_idx].as_bytes())?;
        write_u8(&mut w, 0)?;

        // 2D keypoints
        let keypoints = &data.keypoints_per_image[img_idx];
        write_u64(&mut w, keypoints.len() as u64)?;

        for (feat_idx, kp) in keypoints.iter().enumerate() {
            write_f64(&mut w, kp[0])?; // x
            write_f64(&mut w, kp[1])?; // y

            // Look up point3d_id: convert 0-based index to 1-based ID, or INVALID
            let point3d_id = match keypoint_to_point3d.get(&(img_idx as u32, feat_idx as u32)) {
                Some(&pt_idx) => (pt_idx as u64) + 1, // 1-based
                None => INVALID_POINT3D_ID,
            };
            write_u64(&mut w, point3d_id)?;
        }
    }

    w.flush()?;
    Ok(())
}

/// Write `points3D.bin`.
fn write_points3d_bin(path: &Path, data: &ColmapWriteData) -> Result<(), ColmapIoError> {
    let file = create_file(path)?;
    let mut w = BufWriter::new(file);

    let num_points = data.positions_xyz.len();
    write_u64(&mut w, num_points as u64)?;

    // Group track observations by point3d_index.
    // Build a vec of (image_index, feature_index) per point.
    let mut tracks_by_point: Vec<Vec<(u32, u32)>> = vec![Vec::new(); num_points];
    for i in 0..data.track_image_indexes.len() {
        let pt_idx = data.track_point3d_indexes[i] as usize;
        let img_idx = data.track_image_indexes[i];
        let feat_idx = data.track_feature_indexes[i];
        if pt_idx < num_points {
            tracks_by_point[pt_idx].push((img_idx, feat_idx));
        }
    }

    for (pt_idx, track) in tracks_by_point.iter().enumerate() {
        let point3d_id = (pt_idx as u64) + 1; // 1-based
        let pos = &data.positions_xyz[pt_idx];
        let col = &data.colors_rgb[pt_idx];
        let err = data.reprojection_errors[pt_idx];

        write_u64(&mut w, point3d_id)?;
        write_f64(&mut w, pos[0])?;
        write_f64(&mut w, pos[1])?;
        write_f64(&mut w, pos[2])?;
        write_u8(&mut w, col[0])?;
        write_u8(&mut w, col[1])?;
        write_u8(&mut w, col[2])?;
        write_f64(&mut w, err)?;
        write_u64(&mut w, track.len() as u64)?;

        for &(img_idx, feat_idx) in track {
            let image_id = img_idx + 1; // 1-based
            write_u32(&mut w, image_id)?;
            write_u32(&mut w, feat_idx)?; // 0-based point2D_idx
        }
    }

    w.flush()?;
    Ok(())
}

/// Write `rigs.bin`.
pub fn write_rigs_bin(path: &Path, rigs: &[ColmapRig]) -> Result<(), ColmapIoError> {
    let file = create_file(path)?;
    let mut w = BufWriter::new(file);

    write_u64(&mut w, rigs.len() as u64)?;

    for rig in rigs {
        write_u32(&mut w, rig.rig_id)?;
        let num_sensors = if rig.ref_sensor.is_some() {
            1 + rig.non_ref_sensors.len() as u32
        } else {
            0
        };
        write_u32(&mut w, num_sensors)?;

        if let Some(ref ref_sensor) = rig.ref_sensor {
            write_i32(&mut w, ref_sensor.sensor_type as i32)?;
            write_u32(&mut w, ref_sensor.id)?;
        }

        for non_ref in &rig.non_ref_sensors {
            write_i32(&mut w, non_ref.sensor.sensor_type as i32)?;
            write_u32(&mut w, non_ref.sensor.id)?;

            match non_ref.sensor_from_rig {
                Some((quat, trans)) => {
                    write_u8(&mut w, 1)?;
                    write_f64(&mut w, quat[0])?; // qw
                    write_f64(&mut w, quat[1])?; // qx
                    write_f64(&mut w, quat[2])?; // qy
                    write_f64(&mut w, quat[3])?; // qz
                    write_f64(&mut w, trans[0])?; // tx
                    write_f64(&mut w, trans[1])?; // ty
                    write_f64(&mut w, trans[2])?; // tz
                }
                None => {
                    write_u8(&mut w, 0)?;
                }
            }
        }
    }

    w.flush()?;
    Ok(())
}

/// Write `frames.bin`.
pub fn write_frames_bin(path: &Path, frames: &[ColmapFrame]) -> Result<(), ColmapIoError> {
    let file = create_file(path)?;
    let mut w = BufWriter::new(file);

    write_u64(&mut w, frames.len() as u64)?;

    for frame in frames {
        write_u32(&mut w, frame.frame_id)?;
        write_u32(&mut w, frame.rig_id)?;

        // rig_from_world quaternion WXYZ + translation XYZ
        let q = &frame.quaternion_wxyz;
        let t = &frame.translation_xyz;
        write_f64(&mut w, q[0])?;
        write_f64(&mut w, q[1])?;
        write_f64(&mut w, q[2])?;
        write_f64(&mut w, q[3])?;
        write_f64(&mut w, t[0])?;
        write_f64(&mut w, t[1])?;
        write_f64(&mut w, t[2])?;

        write_u32(&mut w, frame.data_ids.len() as u32)?;
        for data_id in &frame.data_ids {
            write_i32(&mut w, data_id.sensor_type as i32)?;
            write_u32(&mut w, data_id.sensor_id)?;
            write_u64(&mut w, data_id.data_id)?;
        }
    }

    w.flush()?;
    Ok(())
}