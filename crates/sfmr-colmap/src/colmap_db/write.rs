// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Write COLMAP SQLite databases.

use std::path::Path;

use matches_format::MatchesData;
use rusqlite::Connection;

use crate::colmap_io::{camera_params_to_array, colmap_model_id};

use super::types::{
    ColmapDbError, ColmapDbFeatureData, ColmapDbWriteData, ImageIdMap, PosePrior, TwoViewGeometry,
    TwoViewGeometryConfig,
};

/// COLMAP's maximum number of images for pair ID encoding.
/// `kMaxNumImages = std::numeric_limits<int32_t>::max() = 2^31 - 1`
const K_MAX_NUM_IMAGES: i64 = 2_147_483_647;

/// Create the COLMAP database schema.
fn create_schema(conn: &Connection) -> Result<(), ColmapDbError> {
    conn.execute_batch(
        "
        CREATE TABLE IF NOT EXISTS cameras (
            camera_id   INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
            model       INTEGER NOT NULL,
            width       INTEGER NOT NULL,
            height      INTEGER NOT NULL,
            params      BLOB,
            prior_focal_length  INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS images (
            image_id    INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
            name        TEXT NOT NULL DEFAULT '',
            camera_id   INTEGER NOT NULL,
            prior_qw    REAL,
            prior_qx    REAL,
            prior_qy    REAL,
            prior_qz    REAL,
            prior_tx    REAL,
            prior_ty    REAL,
            prior_tz    REAL
        );

        CREATE TABLE IF NOT EXISTS keypoints (
            image_id    INTEGER PRIMARY KEY NOT NULL,
            rows        INTEGER NOT NULL,
            cols        INTEGER NOT NULL,
            data        BLOB,
            FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS descriptors (
            image_id    INTEGER PRIMARY KEY NOT NULL,
            rows        INTEGER NOT NULL,
            cols        INTEGER NOT NULL,
            data        BLOB,
            FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS matches (
            pair_id     INTEGER PRIMARY KEY NOT NULL,
            rows        INTEGER NOT NULL,
            cols        INTEGER NOT NULL,
            data        BLOB
        );

        CREATE TABLE IF NOT EXISTS two_view_geometries (
            pair_id     INTEGER PRIMARY KEY NOT NULL,
            rows        INTEGER NOT NULL,
            cols        INTEGER NOT NULL,
            data        BLOB,
            config      INTEGER NOT NULL DEFAULT 0,
            F           BLOB,
            E           BLOB,
            H           BLOB,
            qvec        BLOB,
            tvec        BLOB
        );

        CREATE TABLE IF NOT EXISTS pose_priors (
            image_id                INTEGER PRIMARY KEY NOT NULL,
            position                BLOB,
            coordinate_system       INTEGER NOT NULL DEFAULT 0,
            position_covariance     BLOB,
            FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE
        );
        ",
    )?;
    Ok(())
}

/// Compute pair ID from two 1-based image IDs using COLMAP's encoding.
///
/// Formula: `kMaxNumImages * smaller_id + larger_id`
fn compute_pair_id(image_id1: i64, image_id2: i64) -> Result<i64, ColmapDbError> {
    if image_id1 <= 0 || image_id2 <= 0 {
        return Err(ColmapDbError::InvalidPairId(format!(
            "Image IDs must be positive, got ({image_id1}, {image_id2})"
        )));
    }
    if image_id1 >= K_MAX_NUM_IMAGES || image_id2 >= K_MAX_NUM_IMAGES {
        return Err(ColmapDbError::InvalidPairId(format!(
            "Image IDs must be < {K_MAX_NUM_IMAGES}, got ({image_id1}, {image_id2})"
        )));
    }
    let (smaller, larger) = if image_id1 < image_id2 {
        (image_id1, image_id2)
    } else {
        (image_id2, image_id1)
    };
    Ok(K_MAX_NUM_IMAGES * smaller + larger)
}

/// Write cameras to the database, returning a map from 0-based camera index to database camera_id.
fn write_cameras(
    conn: &Connection,
    cameras: &[sfmr_format::SfmrCamera],
) -> Result<Vec<i64>, ColmapDbError> {
    let mut camera_ids = Vec::with_capacity(cameras.len());
    let mut stmt = conn.prepare(
        "INSERT INTO cameras (model, width, height, params, prior_focal_length)
         VALUES (?1, ?2, ?3, ?4, ?5)",
    )?;

    for camera in cameras {
        let model_id = colmap_model_id(&camera.model)
            .map_err(|_| ColmapDbError::UnknownModelName(camera.model.clone()))?;
        let params = camera_params_to_array(camera).map_err(|e| match e {
            crate::colmap_io::ColmapIoError::InvalidData(msg) => ColmapDbError::InvalidData(msg),
            crate::colmap_io::ColmapIoError::UnknownModelName(name) => {
                ColmapDbError::UnknownModelName(name)
            }
            other => ColmapDbError::InvalidData(other.to_string()),
        })?;
        let params_blob: Vec<u8> = params.iter().flat_map(|v: &f64| v.to_le_bytes()).collect();

        stmt.execute(rusqlite::params![
            model_id,
            camera.width,
            camera.height,
            params_blob,
            0i32, // prior_focal_length = false
        ])?;
        camera_ids.push(conn.last_insert_rowid());
    }

    Ok(camera_ids)
}

/// Write images to the database, returning a vec of database image_ids.
fn write_images(
    conn: &Connection,
    image_names: &[String],
    camera_indexes: &[u32],
    camera_ids: &[i64],
) -> Result<Vec<i64>, ColmapDbError> {
    let mut image_ids = Vec::with_capacity(image_names.len());
    let mut stmt = conn.prepare(
        "INSERT INTO images (name, camera_id)
         VALUES (?1, ?2)",
    )?;

    for (i, name) in image_names.iter().enumerate() {
        let cam_idx = camera_indexes[i] as usize;
        let db_camera_id = camera_ids[cam_idx];
        stmt.execute(rusqlite::params![name, db_camera_id])?;
        image_ids.push(conn.last_insert_rowid());
    }

    Ok(image_ids)
}

/// Write keypoints for all images.
fn write_keypoints(
    conn: &Connection,
    keypoints_per_image: &[Vec<[f64; 2]>],
    image_ids: &[i64],
) -> Result<(), ColmapDbError> {
    let mut stmt = conn.prepare(
        "INSERT INTO keypoints (image_id, rows, cols, data)
         VALUES (?1, ?2, ?3, ?4)",
    )?;

    for (i, keypoints) in keypoints_per_image.iter().enumerate() {
        let num_keypoints = keypoints.len() as i32;
        let cols = 2i32;
        let blob: Vec<u8> = keypoints
            .iter()
            .flat_map(|[x, y]| {
                let mut bytes = Vec::with_capacity(16);
                bytes.extend_from_slice(&(*x as f32).to_le_bytes());
                bytes.extend_from_slice(&(*y as f32).to_le_bytes());
                bytes
            })
            .collect();

        stmt.execute(rusqlite::params![image_ids[i], num_keypoints, cols, blob])?;
    }

    Ok(())
}

/// Write descriptors for all images.
fn write_descriptors(
    conn: &Connection,
    keypoints_per_image: &[Vec<[f64; 2]>],
    descriptors_per_image: &[Vec<u8>],
    descriptor_dim: u32,
    image_ids: &[i64],
) -> Result<(), ColmapDbError> {
    let mut stmt = conn.prepare(
        "INSERT INTO descriptors (image_id, rows, cols, data)
         VALUES (?1, ?2, ?3, ?4)",
    )?;

    let dim = descriptor_dim as usize;

    for (i, desc_data) in descriptors_per_image.iter().enumerate() {
        let num_keypoints = keypoints_per_image[i].len();
        let expected_len = num_keypoints * dim;
        if desc_data.len() != expected_len {
            return Err(ColmapDbError::InvalidData(format!(
                "Image {} descriptor data length {} does not match expected {} ({}×{})",
                i,
                desc_data.len(),
                expected_len,
                num_keypoints,
                dim
            )));
        }

        stmt.execute(rusqlite::params![
            image_ids[i],
            num_keypoints as i32,
            descriptor_dim as i32,
            desc_data.as_slice(),
        ])?;
    }

    Ok(())
}

/// Write pose priors for images.
fn write_pose_priors(
    conn: &Connection,
    pose_priors: &[PosePrior],
    image_ids: &[i64],
) -> Result<(), ColmapDbError> {
    let mut stmt = conn.prepare(
        "INSERT INTO pose_priors (image_id, position, coordinate_system, position_covariance)
         VALUES (?1, ?2, ?3, ?4)",
    )?;

    for (i, prior) in pose_priors.iter().enumerate() {
        let position_blob: Vec<u8> = prior
            .position
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let covariance_blob: Vec<u8> = prior
            .position_covariance
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();

        stmt.execute(rusqlite::params![
            image_ids[i],
            position_blob,
            prior.coordinate_system,
            covariance_blob,
        ])?;
    }

    Ok(())
}

/// Convert an f64 array to a little-endian byte blob.
fn f64_array_to_blob(values: &[f64]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// Write two-view geometries (legacy format with per-entry struct).
fn write_two_view_geometries(
    conn: &Connection,
    tvgs: &[TwoViewGeometry],
    image_ids: &[i64],
) -> Result<(), ColmapDbError> {
    let mut stmt = conn.prepare(
        "INSERT INTO two_view_geometries (pair_id, rows, cols, data, config, F, E, H, qvec, tvec)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
    )?;

    for tvg in tvgs {
        let db_id1 = image_ids[tvg.image_idx1 as usize];
        let db_id2 = image_ids[tvg.image_idx2 as usize];
        let pair_id = compute_pair_id(db_id1, db_id2)?;

        let num_matches = tvg.matches.len() / 2;
        let match_blob: Vec<u8> = tvg.matches.iter().flat_map(|v| v.to_le_bytes()).collect();

        let f_blob = tvg.f_matrix.as_ref().map(|m| f64_array_to_blob(m));
        let e_blob = tvg.e_matrix.as_ref().map(|m| f64_array_to_blob(m));
        let h_blob = tvg.h_matrix.as_ref().map(|m| f64_array_to_blob(m));
        let qvec_blob = tvg.qvec_wxyz.as_ref().map(|q| f64_array_to_blob(q));
        let tvec_blob = tvg.tvec.as_ref().map(|t| f64_array_to_blob(t));

        stmt.execute(rusqlite::params![
            pair_id,
            num_matches as i32,
            2i32,
            match_blob,
            tvg.config as i32,
            f_blob,
            e_blob,
            h_blob,
            qvec_blob,
            tvec_blob,
        ])?;
    }

    Ok(())
}

/// Check if a 3×3 matrix (stored as 9-element row-major) is all zeros.
fn is_zero_matrix_3x3(m: &[f64; 9]) -> bool {
    m.iter().all(|&v| v == 0.0)
}

/// Convert an Option<[f64; N]> to a blob, returning None if the array is a sentinel.
fn matrix_to_optional_blob(m: &[f64; 9]) -> Option<Vec<u8>> {
    if is_zero_matrix_3x3(m) {
        None
    } else {
        Some(f64_array_to_blob(m))
    }
}

fn quat_to_optional_blob(q: &[f64; 4]) -> Option<Vec<u8>> {
    // Identity quaternion [1, 0, 0, 0] → NULL
    if (q[0] - 1.0).abs() < f64::EPSILON
        && q[1].abs() < f64::EPSILON
        && q[2].abs() < f64::EPSILON
        && q[3].abs() < f64::EPSILON
    {
        None
    } else {
        Some(f64_array_to_blob(q))
    }
}

fn tvec_to_optional_blob(t: &[f64; 3]) -> Option<Vec<u8>> {
    if t[0].abs() < f64::EPSILON && t[1].abs() < f64::EPSILON && t[2].abs() < f64::EPSILON {
        None
    } else {
        Some(f64_array_to_blob(t))
    }
}

// ── Public API ──────────────────────────────────────────────────────────

/// Create a COLMAP SQLite database and populate it with reconstruction data.
///
/// This creates a database file at `db_path` with the standard COLMAP schema and
/// populates it with cameras, images, keypoints, descriptors, and optionally
/// pose priors and two-view geometries.
///
/// If the file already exists, it will be overwritten.
pub fn write_colmap_db(
    db_path: &Path,
    data: &ColmapDbWriteData,
) -> Result<Vec<i64>, ColmapDbError> {
    if db_path.exists() {
        std::fs::remove_file(db_path)?;
    }
    if let Some(parent) = db_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let conn = Connection::open(db_path)?;
    conn.execute_batch("PRAGMA journal_mode=WAL;")?;
    create_schema(&conn)?;

    let tx = conn.unchecked_transaction()?;

    let camera_ids = write_cameras(&tx, data.cameras)?;
    let image_ids = write_images(&tx, data.image_names, data.camera_indexes, &camera_ids)?;
    write_keypoints(&tx, data.keypoints_per_image, &image_ids)?;
    write_descriptors(
        &tx,
        data.keypoints_per_image,
        data.descriptors_per_image,
        data.descriptor_dim,
        &image_ids,
    )?;

    if let Some(priors) = data.pose_priors {
        write_pose_priors(&tx, priors, &image_ids)?;
    }

    if let Some(tvgs) = data.two_view_geometries {
        write_two_view_geometries(&tx, tvgs, &image_ids)?;
    }

    tx.commit()?;
    Ok(image_ids)
}

/// Create a COLMAP database with cameras, images, keypoints, and descriptors.
///
/// Does NOT write matches or two-view geometries. Returns an [`ImageIdMap`]
/// for use with [`write_colmap_db_matches`].
///
/// If the file already exists, it will be overwritten.
pub fn write_colmap_db_features(
    db_path: &Path,
    data: &ColmapDbFeatureData,
) -> Result<ImageIdMap, ColmapDbError> {
    if db_path.exists() {
        std::fs::remove_file(db_path)?;
    }
    if let Some(parent) = db_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let conn = Connection::open(db_path)?;
    conn.execute_batch("PRAGMA journal_mode=WAL;")?;
    create_schema(&conn)?;

    let tx = conn.unchecked_transaction()?;

    let camera_ids = write_cameras(&tx, data.cameras)?;
    let image_ids = write_images(&tx, data.image_names, data.camera_indexes, &camera_ids)?;
    write_keypoints(&tx, data.keypoints_per_image, &image_ids)?;
    write_descriptors(
        &tx,
        data.keypoints_per_image,
        data.descriptors_per_image,
        data.descriptor_dim,
        &image_ids,
    )?;

    if let Some(priors) = data.pose_priors {
        write_pose_priors(&tx, priors, &image_ids)?;
    }

    tx.commit()?;
    Ok(ImageIdMap::from_db_ids(image_ids))
}

/// Write the `matches` and optionally `two_view_geometries` tables to an
/// existing COLMAP database from a [`MatchesData`].
///
/// The `id_map` must come from the [`write_colmap_db_features`] call that
/// created this database, ensuring image indexes are consistent.
pub fn write_colmap_db_matches(
    db_path: &Path,
    matches_data: &MatchesData,
    id_map: &ImageIdMap,
) -> Result<(), ColmapDbError> {
    let conn = Connection::open(db_path)?;
    let tx = conn.unchecked_transaction()?;

    let pair_count = matches_data.metadata.image_pair_count as usize;

    // Write matches table
    {
        let mut stmt =
            tx.prepare("INSERT INTO matches (pair_id, rows, cols, data) VALUES (?1, ?2, ?3, ?4)")?;

        let mut match_offset: usize = 0;
        for k in 0..pair_count {
            let idx_i = matches_data.image_index_pairs[[k, 0]] as usize;
            let idx_j = matches_data.image_index_pairs[[k, 1]] as usize;

            if idx_i >= id_map.index_to_db_id.len() || idx_j >= id_map.index_to_db_id.len() {
                return Err(ColmapDbError::InvalidData(format!(
                    "Image index ({idx_i}, {idx_j}) out of range for id_map (len {})",
                    id_map.index_to_db_id.len()
                )));
            }

            let db_id_i = id_map.index_to_db_id[idx_i];
            let db_id_j = id_map.index_to_db_id[idx_j];
            let pair_id = compute_pair_id(db_id_i, db_id_j)?;

            let count = matches_data.match_counts[k] as usize;

            // Build u32 LE blob for match feature indexes
            let mut blob = Vec::with_capacity(count * 2 * 4);
            for m in match_offset..match_offset + count {
                blob.extend_from_slice(&matches_data.match_feature_indexes[[m, 0]].to_le_bytes());
                blob.extend_from_slice(&matches_data.match_feature_indexes[[m, 1]].to_le_bytes());
            }

            stmt.execute(rusqlite::params![pair_id, count as i32, 2i32, blob])?;
            match_offset += count;
        }
    }

    // Write two_view_geometries table
    if let Some(tvg) = &matches_data.two_view_geometries {
        let mut stmt = tx.prepare(
            "INSERT INTO two_view_geometries (pair_id, rows, cols, data, config, F, E, H, qvec, tvec)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
        )?;

        let mut inlier_offset: usize = 0;
        for k in 0..pair_count {
            let idx_i = matches_data.image_index_pairs[[k, 0]] as usize;
            let idx_j = matches_data.image_index_pairs[[k, 1]] as usize;
            let db_id_i = id_map.index_to_db_id[idx_i];
            let db_id_j = id_map.index_to_db_id[idx_j];
            let pair_id = compute_pair_id(db_id_i, db_id_j)?;

            let inlier_count = tvg.inlier_counts[k] as usize;

            // Build inlier blob
            let mut inlier_blob = Vec::with_capacity(inlier_count * 2 * 4);
            for i in inlier_offset..inlier_offset + inlier_count {
                inlier_blob.extend_from_slice(&tvg.inlier_feature_indexes[[i, 0]].to_le_bytes());
                inlier_blob.extend_from_slice(&tvg.inlier_feature_indexes[[i, 1]].to_le_bytes());
            }

            // Config
            let config_enum = tvg.config_types[tvg.config_indexes[k] as usize];
            let colmap_config: TwoViewGeometryConfig = config_enum.into();

            // Matrices — extract row k from (P, 3, 3) arrays
            let f: [f64; 9] = std::array::from_fn(|i| tvg.f_matrices[[k, i / 3, i % 3]]);
            let e: [f64; 9] = std::array::from_fn(|i| tvg.e_matrices[[k, i / 3, i % 3]]);
            let h: [f64; 9] = std::array::from_fn(|i| tvg.h_matrices[[k, i / 3, i % 3]]);

            // Quaternion and translation — extract row k
            let q: [f64; 4] = std::array::from_fn(|i| tvg.quaternions_wxyz[[k, i]]);
            let t: [f64; 3] = std::array::from_fn(|i| tvg.translations_xyz[[k, i]]);

            stmt.execute(rusqlite::params![
                pair_id,
                inlier_count as i32,
                2i32,
                inlier_blob,
                colmap_config as i32,
                matrix_to_optional_blob(&f),
                matrix_to_optional_blob(&e),
                matrix_to_optional_blob(&h),
                quat_to_optional_blob(&q),
                tvec_to_optional_blob(&t),
            ])?;

            inlier_offset += inlier_count;
        }
    }

    tx.commit()?;
    Ok(())
}