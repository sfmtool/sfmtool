// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Read and write COLMAP SQLite databases.
//!
//! This module handles creating and populating COLMAP-compatible SQLite databases
//! with cameras, images, keypoints, descriptors, pose priors, and two-view geometries.
//!
//! The database schema matches COLMAP's `database.cc` implementation. Key conventions:
//! - Camera and image IDs are 1-based in the database
//! - Pair IDs encode image pairs as `kMaxNumImages * smaller_id + larger_id`
//!   where `kMaxNumImages = 2^31 - 1`
//! - Matrices (F, E, H) are stored as row-major f64 BLOBs
//! - Quaternions are stored in WXYZ order
//! - Keypoints and descriptors are stored as BLOBs with (rows, cols) dimensions

mod read;
mod types;
mod write;

pub use read::read_colmap_db_matches;
pub use types::{
    ColmapDbError, ColmapDbFeatureData, ColmapDbWriteData, DbFrame, DbFrameDataId, DbRig,
    DbRigSensor, DbSensor, DbSensorType, ImageIdMap, PosePrior, TwoViewGeometry,
    TwoViewGeometryConfig,
};
pub use write::{write_colmap_db, write_colmap_db_features, write_colmap_db_matches};

#[cfg(test)]
mod tests;
