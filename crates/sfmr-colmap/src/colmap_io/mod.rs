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
mod tests;
