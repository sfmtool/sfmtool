// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Read matches and two-view geometries from COLMAP SQLite databases.

use std::collections::HashMap;
use std::path::Path;

use matches_format::{
    MatchesContentHash, MatchesData, MatchesMetadata, TvgMetadata, TwoViewGeometryData,
    WorkspaceContents, WorkspaceMetadata,
};
use ndarray::{Array1, Array2, Array3};
use rusqlite::Connection;

use super::types::{ColmapDbError, TwoViewGeometryConfig};

/// COLMAP's maximum number of images for pair ID encoding.
const K_MAX_NUM_IMAGES: i64 = 2_147_483_647;

/// Decode a COLMAP pair_id back to (smaller_id, larger_id).
fn decode_pair_id(pair_id: i64) -> (i64, i64) {
    let smaller = pair_id / K_MAX_NUM_IMAGES;
    let larger = pair_id % K_MAX_NUM_IMAGES;
    (smaller, larger)
}

/// Per-pair match data used during reading.
struct PairMatches {
    idx_i: u32,
    idx_j: u32,
    feature_indexes: Vec<[u32; 2]>,
}

/// Read matches and two-view geometries from a COLMAP database.
///
/// Reads the `images` and `keypoints` tables to establish the image list
/// and feature counts, then reads the `matches` table for candidate matches
/// and optionally the `two_view_geometries` table.
///
/// Descriptor distances are not stored in the COLMAP DB, so
/// `match_descriptor_distances` is filled with `0.0` for every match.
/// The caller should compute real distances from `.sift` files if needed.
///
/// Metadata fields not stored in the DB (workspace, tool hashes, content
/// hash) are set to placeholder values. The caller fills these in before
/// writing to a `.matches` file via `write_matches`.
pub fn read_colmap_db_matches(
    db_path: &Path,
    include_tvg: bool,
) -> Result<MatchesData, ColmapDbError> {
    let conn = Connection::open(db_path)?;

    // Read images, sorted by name (lexicographic, matching .matches spec recommendation)
    let mut img_stmt = conn.prepare("SELECT image_id, name FROM images ORDER BY name")?;
    let image_rows: Vec<(i64, String)> = img_stmt
        .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?
        .collect::<Result<Vec<_>, _>>()?;

    let image_count = image_rows.len();
    let image_names: Vec<String> = image_rows.iter().map(|(_, name)| name.clone()).collect();

    // Build db_id → 0-based index map
    let db_id_to_index: HashMap<i64, usize> = image_rows
        .iter()
        .enumerate()
        .map(|(idx, (db_id, _))| (*db_id, idx))
        .collect();

    // Read keypoints to get feature counts
    let mut feature_counts_vec = vec![0u32; image_count];
    {
        let mut kp_stmt = conn.prepare("SELECT image_id, rows FROM keypoints")?;
        let kp_rows =
            kp_stmt.query_map([], |row| Ok((row.get::<_, i64>(0)?, row.get::<_, i32>(1)?)))?;
        for kp in kp_rows {
            let (db_id, rows) = kp?;
            if let Some(&idx) = db_id_to_index.get(&db_id) {
                feature_counts_vec[idx] = rows as u32;
            }
        }
    }

    // Read matches table
    let mut all_pairs: Vec<PairMatches> = Vec::new();
    {
        let mut match_stmt = conn.prepare("SELECT pair_id, rows, data FROM matches")?;
        let match_rows = match_stmt.query_map([], |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, i32>(1)?,
                row.get::<_, Option<Vec<u8>>>(2)?,
            ))
        })?;

        for m in match_rows {
            let (pair_id, rows, data) = m?;

            // Skip pairs with no match data (NULL blob or 0 rows)
            let count = rows as usize;
            if count == 0 {
                continue;
            }
            let data = match data {
                Some(d) => d,
                None => continue,
            };

            let (db_smaller, db_larger) = decode_pair_id(pair_id);

            let idx_i = match db_id_to_index.get(&db_smaller) {
                Some(&idx) => idx,
                None => continue, // Skip pairs with unknown images
            };
            let idx_j = match db_id_to_index.get(&db_larger) {
                Some(&idx) => idx,
                None => continue,
            };

            let expected_bytes = count * 2 * 4;
            if data.len() != expected_bytes {
                return Err(ColmapDbError::InvalidData(format!(
                    "Match blob for pair_id {pair_id}: expected {expected_bytes} bytes, got {}",
                    data.len()
                )));
            }

            let mut feature_indexes = Vec::with_capacity(count);
            for i in 0..count {
                let fi = u32::from_le_bytes(data[i * 8..i * 8 + 4].try_into().unwrap());
                let fj = u32::from_le_bytes(data[i * 8 + 4..i * 8 + 8].try_into().unwrap());
                feature_indexes.push([fi, fj]);
            }

            // Ensure idx_i < idx_j (should already be the case from pair_id encoding)
            let (final_i, final_j, final_fi) = if idx_i < idx_j {
                (idx_i as u32, idx_j as u32, feature_indexes)
            } else {
                // Swap pair and swap feature indexes within each match
                let swapped: Vec<[u32; 2]> =
                    feature_indexes.iter().map(|[a, b]| [*b, *a]).collect();
                (idx_j as u32, idx_i as u32, swapped)
            };

            all_pairs.push(PairMatches {
                idx_i: final_i,
                idx_j: final_j,
                feature_indexes: final_fi,
            });
        }
    }

    // Sort pairs by (idx_i, idx_j)
    all_pairs.sort_by(|a, b| (a.idx_i, a.idx_j).cmp(&(b.idx_i, b.idx_j)));

    // Build flat arrays
    let pair_count = all_pairs.len();
    let total_matches: usize = all_pairs.iter().map(|p| p.feature_indexes.len()).sum();

    let mut image_index_pairs_vec = Vec::with_capacity(pair_count * 2);
    let mut match_counts_vec = Vec::with_capacity(pair_count);
    let mut match_feature_indexes_vec = Vec::with_capacity(total_matches * 2);

    for pair in &all_pairs {
        image_index_pairs_vec.push(pair.idx_i);
        image_index_pairs_vec.push(pair.idx_j);
        match_counts_vec.push(pair.feature_indexes.len() as u32);
        for [fi, fj] in &pair.feature_indexes {
            match_feature_indexes_vec.push(*fi);
            match_feature_indexes_vec.push(*fj);
        }
    }

    let image_index_pairs = Array2::from_shape_vec((pair_count, 2), image_index_pairs_vec)
        .map_err(|e| ColmapDbError::InvalidData(format!("image_index_pairs reshape: {e}")))?;
    let match_counts = Array1::from_vec(match_counts_vec);
    let match_feature_indexes =
        Array2::from_shape_vec((total_matches, 2), match_feature_indexes_vec).map_err(|e| {
            ColmapDbError::InvalidData(format!("match_feature_indexes reshape: {e}"))
        })?;
    let match_descriptor_distances = Array1::from_vec(vec![0.0f32; total_matches]);

    // Read two-view geometries (optional)
    let two_view_geometries = if include_tvg {
        read_tvg_from_db(&conn, &all_pairs, &db_id_to_index, pair_count)?
    } else {
        None
    };

    let feature_counts = Array1::from_vec(feature_counts_vec);

    Ok(MatchesData {
        metadata: MatchesMetadata {
            version: 1,
            matching_method: "unknown".into(),
            matching_tool: "colmap".into(),
            matching_tool_version: String::new(),
            matching_options: HashMap::new(),
            workspace: WorkspaceMetadata {
                absolute_path: String::new(),
                relative_path: String::new(),
                contents: WorkspaceContents {
                    feature_tool: String::new(),
                    feature_type: String::new(),
                    feature_options: serde_json::Value::Object(Default::default()),
                    feature_prefix_dir: String::new(),
                },
            },
            timestamp: String::new(),
            image_count: image_count as u32,
            image_pair_count: pair_count as u32,
            match_count: total_matches as u32,
            has_two_view_geometries: two_view_geometries.is_some(),
        },
        content_hash: MatchesContentHash {
            metadata_xxh128: String::new(),
            images_xxh128: String::new(),
            image_pairs_xxh128: String::new(),
            two_view_geometries_xxh128: None,
            content_xxh128: String::new(),
        },
        image_names,
        feature_tool_hashes: vec![[0u8; 16]; image_count],
        sift_content_hashes: vec![[0u8; 16]; image_count],
        feature_counts,
        image_index_pairs,
        match_counts,
        match_feature_indexes,
        match_descriptor_distances,
        two_view_geometries,
    })
}

/// Read the two_view_geometries table and build TwoViewGeometryData aligned
/// with the sorted pair order from the matches table.
fn read_tvg_from_db(
    conn: &Connection,
    sorted_pairs: &[PairMatches],
    db_id_to_index: &HashMap<i64, usize>,
    pair_count: usize,
) -> Result<Option<TwoViewGeometryData>, ColmapDbError> {
    // Check if the table has any rows
    let tvg_count: i32 = conn.query_row("SELECT COUNT(*) FROM two_view_geometries", [], |row| {
        row.get(0)
    })?;
    if tvg_count == 0 {
        return Ok(None);
    }

    // Read all TVG rows into a map keyed by (idx_i, idx_j)
    struct TvgRow {
        config: i32,
        inlier_indexes: Vec<[u32; 2]>,
        f_matrix: [f64; 9],
        e_matrix: [f64; 9],
        h_matrix: [f64; 9],
        qvec: [f64; 4],
        tvec: [f64; 3],
    }

    let mut tvg_map: HashMap<(u32, u32), TvgRow> = HashMap::new();
    {
        let mut stmt = conn.prepare(
            "SELECT pair_id, rows, data, config, F, E, H, qvec, tvec FROM two_view_geometries",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, i32>(1)?,
                row.get::<_, Option<Vec<u8>>>(2)?,
                row.get::<_, i32>(3)?,
                row.get::<_, Option<Vec<u8>>>(4)?,
                row.get::<_, Option<Vec<u8>>>(5)?,
                row.get::<_, Option<Vec<u8>>>(6)?,
                row.get::<_, Option<Vec<u8>>>(7)?,
                row.get::<_, Option<Vec<u8>>>(8)?,
            ))
        })?;

        for r in rows {
            let (pair_id, inlier_rows, data, config, f_blob, e_blob, h_blob, qvec_blob, tvec_blob) =
                r?;
            let (db_smaller, db_larger) = decode_pair_id(pair_id);

            let idx_i = match db_id_to_index.get(&db_smaller) {
                Some(&idx) => idx as u32,
                None => continue,
            };
            let idx_j = match db_id_to_index.get(&db_larger) {
                Some(&idx) => idx as u32,
                None => continue,
            };

            let (final_i, final_j) = if idx_i < idx_j {
                (idx_i, idx_j)
            } else {
                (idx_j, idx_i)
            };

            let count = inlier_rows as usize;
            let mut inlier_indexes = Vec::with_capacity(count);
            if let Some(ref data) = data {
                for i in 0..count {
                    let fi = u32::from_le_bytes(data[i * 8..i * 8 + 4].try_into().unwrap());
                    let fj = u32::from_le_bytes(data[i * 8 + 4..i * 8 + 8].try_into().unwrap());
                    if idx_i < idx_j {
                        inlier_indexes.push([fi, fj]);
                    } else {
                        inlier_indexes.push([fj, fi]);
                    }
                }
            }

            let f_matrix = blob_to_matrix_3x3(&f_blob);
            let e_matrix = blob_to_matrix_3x3(&e_blob);
            let h_matrix = blob_to_matrix_3x3(&h_blob);
            let qvec = blob_to_quat(&qvec_blob);
            let tvec = blob_to_tvec(&tvec_blob);

            tvg_map.insert(
                (final_i, final_j),
                TvgRow {
                    config,
                    inlier_indexes,
                    f_matrix,
                    e_matrix,
                    h_matrix,
                    qvec,
                    tvec,
                },
            );
        }
    }

    // Build arrays aligned with sorted_pairs order
    let mut config_set: Vec<matches_format::TwoViewGeometryConfig> = Vec::new();
    let mut config_index_map: HashMap<matches_format::TwoViewGeometryConfig, u8> = HashMap::new();

    let mut config_indexes_vec = Vec::with_capacity(pair_count);
    let mut inlier_counts_vec = Vec::with_capacity(pair_count);
    let mut all_inlier_fi: Vec<u32> = Vec::new();
    let mut f_data = Vec::with_capacity(pair_count * 9);
    let mut e_data = Vec::with_capacity(pair_count * 9);
    let mut h_data = Vec::with_capacity(pair_count * 9);
    let mut q_data = Vec::with_capacity(pair_count * 4);
    let mut t_data = Vec::with_capacity(pair_count * 3);

    for pair in sorted_pairs {
        let tvg_row = tvg_map.get(&(pair.idx_i, pair.idx_j));

        let empty_inliers = Vec::new();
        let (config, inliers, f, e, h, q, t) = match tvg_row {
            Some(row) => {
                let cfg = TwoViewGeometryConfig::from_colmap_int(row.config)?;
                let matches_cfg: matches_format::TwoViewGeometryConfig = cfg.into();
                (
                    matches_cfg,
                    &row.inlier_indexes,
                    row.f_matrix,
                    row.e_matrix,
                    row.h_matrix,
                    row.qvec,
                    row.tvec,
                )
            }
            None => (
                matches_format::TwoViewGeometryConfig::Undefined,
                &empty_inliers,
                [0.0; 9],
                [0.0; 9],
                [0.0; 9],
                [1.0, 0.0, 0.0, 0.0],
                [0.0; 3],
            ),
        };

        // Config index
        let ci = if let Some(&idx) = config_index_map.get(&config) {
            idx
        } else {
            let idx = config_set.len() as u8;
            config_set.push(config);
            config_index_map.insert(config, idx);
            idx
        };
        config_indexes_vec.push(ci);

        inlier_counts_vec.push(inliers.len() as u32);
        for [fi, fj] in inliers {
            all_inlier_fi.push(*fi);
            all_inlier_fi.push(*fj);
        }

        f_data.extend_from_slice(&f);
        e_data.extend_from_slice(&e);
        h_data.extend_from_slice(&h);
        q_data.extend_from_slice(&q);
        t_data.extend_from_slice(&t);
    }

    let total_inliers = all_inlier_fi.len() / 2;

    Ok(Some(TwoViewGeometryData {
        metadata: TvgMetadata {
            image_pair_count: pair_count as u32,
            inlier_count: total_inliers as u32,
            verification_tool: "colmap".into(),
            verification_options: HashMap::new(),
        },
        config_types: config_set,
        config_indexes: Array1::from_vec(config_indexes_vec),
        inlier_counts: Array1::from_vec(inlier_counts_vec),
        inlier_feature_indexes: Array2::from_shape_vec((total_inliers, 2), all_inlier_fi).map_err(
            |e| ColmapDbError::InvalidData(format!("inlier_feature_indexes reshape: {e}")),
        )?,
        f_matrices: Array3::from_shape_vec((pair_count, 3, 3), f_data)
            .map_err(|e| ColmapDbError::InvalidData(format!("f_matrices reshape: {e}")))?,
        e_matrices: Array3::from_shape_vec((pair_count, 3, 3), e_data)
            .map_err(|e| ColmapDbError::InvalidData(format!("e_matrices reshape: {e}")))?,
        h_matrices: Array3::from_shape_vec((pair_count, 3, 3), h_data)
            .map_err(|e| ColmapDbError::InvalidData(format!("h_matrices reshape: {e}")))?,
        quaternions_wxyz: Array2::from_shape_vec((pair_count, 4), q_data)
            .map_err(|e| ColmapDbError::InvalidData(format!("quaternions reshape: {e}")))?,
        translations_xyz: Array2::from_shape_vec((pair_count, 3), t_data)
            .map_err(|e| ColmapDbError::InvalidData(format!("translations reshape: {e}")))?,
    }))
}

/// Parse an optional f64 blob into a 3×3 matrix. NULL → all zeros.
fn blob_to_matrix_3x3(blob: &Option<Vec<u8>>) -> [f64; 9] {
    match blob {
        Some(b) if b.len() == 72 => {
            let mut m = [0.0; 9];
            for (i, chunk) in b.chunks_exact(8).enumerate() {
                m[i] = f64::from_le_bytes(chunk.try_into().unwrap());
            }
            m
        }
        _ => [0.0; 9],
    }
}

/// Parse an optional f64 blob into a quaternion. NULL → identity [1,0,0,0].
fn blob_to_quat(blob: &Option<Vec<u8>>) -> [f64; 4] {
    match blob {
        Some(b) if b.len() == 32 => {
            let mut q = [0.0; 4];
            for (i, chunk) in b.chunks_exact(8).enumerate() {
                q[i] = f64::from_le_bytes(chunk.try_into().unwrap());
            }
            q
        }
        _ => [1.0, 0.0, 0.0, 0.0],
    }
}

/// Parse an optional f64 blob into a translation. NULL → [0,0,0].
fn blob_to_tvec(blob: &Option<Vec<u8>>) -> [f64; 3] {
    match blob {
        Some(b) if b.len() == 24 => {
            let mut t = [0.0; 3];
            for (i, chunk) in b.chunks_exact(8).enumerate() {
                t[i] = f64::from_le_bytes(chunk.try_into().unwrap());
            }
            t
        }
        _ => [0.0; 3],
    }
}
