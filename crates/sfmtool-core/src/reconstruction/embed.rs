// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Baseline `sift_files` → `embedded_patches` conversion (no photometric
//! adaptation).
//!
//! [`SfmrReconstruction::to_embedded_patches`] changes a reconstruction's
//! observation representation without registering anything photometrically: it
//! gives each point a `(u, v)` patch frame from a chosen normal/extent policy
//! (e.g. the mean viewing direction), copies each observation's 2D keypoint
//! straight from its `.sift` feature, and copies each image's identity hash from
//! the `.sift` metadata. The result is a valid `embedded_patches` reconstruction
//! whose keypoints are exactly the original SIFT detections, with no photometric
//! [sift→patch pipeline](../../patch) (normal refinement + view selection +
//! keypoint localization) involved.

use ndarray::Array2;

use sift_format::{read_sift_metadata, read_sift_positions};

use super::data::ReconstructionError;
use super::{ObservationSource, SfmrReconstruction};
use crate::patch::cloud::{PatchCloud, PatchExtent, PatchNormal};

impl SfmrReconstruction {
    /// Convert this `sift_files` reconstruction into an `embedded_patches` one
    /// **without photometric adaptation**, returning a new reconstruction (the
    /// input is unchanged).
    ///
    /// - **Patch frame:** each point gets a `(u, v)` half-vector frame from
    ///   [`PatchCloud::from_reconstruction`] (with `exclude_points_at_infinity =
    ///   false`) built from `normal` (e.g. [`PatchNormal::MeanViewing`]) and
    ///   `extent`, with no normal refinement. Finite points get a planar surfel
    ///   frame; points at infinity get a tangent-sphere frame around their
    ///   direction `d` (`u, v ⊥ d`, normal `normalize(-d)`), so **every** point
    ///   carries a real frame and the point set is preserved.
    /// - **Keypoints:** each observation's inline `keypoints_xy` is copied
    ///   verbatim from its `.sift` feature (`sift.positions_xy[feature_index]`),
    ///   so the 2D coordinate is exactly the original SIFT detection.
    /// - **Image hashes:** each image's `image_file_hashes` entry is read from its
    ///   `.sift` metadata (`image_file_xxh128`) — a minimal metadata read, no
    ///   re-hashing of the image bytes.
    ///
    /// Errors with [`ReconstructionError::Unsupported`] if the reconstruction is
    /// already `embedded_patches` (no `.sift` to copy from) or the patch frame
    /// cannot be built, and with [`ReconstructionError::SiftRead`] if a `.sift`
    /// file cannot be read or lacks a feature an observation references.
    pub fn to_embedded_patches(
        &self,
        normal: PatchNormal,
        extent: PatchExtent,
    ) -> Result<Self, ReconstructionError> {
        if let ObservationSource::EmbeddedPatches { .. } = &self.observations {
            return Err(ReconstructionError::Unsupported(
                "to_embedded_patches: reconstruction is already embedded_patches; \
                 there is no .sift to copy keypoints from"
                    .to_string(),
            ));
        }

        let feature_indexes = match &self.observations {
            ObservationSource::SiftFiles {
                feature_indexes, ..
            } => feature_indexes,
            // Unreachable: the embedded case returned above.
            ObservationSource::EmbeddedPatches { .. } => unreachable!(),
        };

        // Patch frames from the chosen normal/extent policy — no refinement.
        // Build frames for every point: finite surfels plus the tangent-sphere
        // frames for points at infinity (exclude_points_at_infinity = false), so
        // every point ends up with a real (non-zero) frame.
        let cloud = PatchCloud::from_reconstruction(self, normal, extent, false).map_err(|e| {
            ReconstructionError::Unsupported(format!(
                "to_embedded_patches: building patch frames failed: {e}"
            ))
        })?;
        let (patch_u, patch_v) = cloud.to_halfvec_arrays(self.points.len());

        // Per-image: a minimal keypoint read plus the source-image identity hash.
        let n_images = self.images.len();
        let mut positions_per_image: Vec<Vec<[f32; 2]>> = Vec::with_capacity(n_images);
        let mut image_file_hashes: Vec<[u8; 16]> = Vec::with_capacity(n_images);
        for i in 0..n_images {
            let path = self.sift_path_for_image(i);
            let count = self.max_track_feature_index[i] as usize + 1;
            let positions =
                read_sift_positions(&path, count).map_err(|e| ReconstructionError::SiftRead {
                    path: path.clone(),
                    source: e.to_string(),
                })?;
            let (_, meta, _) =
                read_sift_metadata(&path).map_err(|e| ReconstructionError::SiftRead {
                    path: path.clone(),
                    source: e.to_string(),
                })?;
            let hash = decode_xxh128_hex(&meta.image_file_xxh128).ok_or_else(|| {
                ReconstructionError::SiftRead {
                    path: path.clone(),
                    source: format!(
                        "invalid image_file_xxh128 {:?} (expected 32 hex chars)",
                        meta.image_file_xxh128
                    ),
                }
            })?;
            positions_per_image.push(positions);
            image_file_hashes.push(hash);
        }

        // Per-observation keypoints, parallel to `tracks` (and thus to the
        // feature_indexes column), so the existing track ordering is preserved.
        let m = self.tracks.len();
        let mut keypoints_xy = Array2::<f32>::zeros((m, 2));
        for (j, obs) in self.tracks.iter().enumerate() {
            let img = obs.image_index as usize;
            let fidx = feature_indexes[j] as usize;
            let pos = positions_per_image[img].get(fidx).ok_or_else(|| {
                ReconstructionError::SiftRead {
                    path: self.sift_path_for_image(img),
                    source: format!(
                        "observation {j} references feature {fidx} of image {img}, but only \
                         {} features were read",
                        positions_per_image[img].len()
                    ),
                }
            })?;
            keypoints_xy[[j, 0]] = pos[0];
            keypoints_xy[[j, 1]] = pos[1];
        }

        let mut out = self.clone();
        out.observations = ObservationSource::EmbeddedPatches {
            keypoints_xy,
            image_file_hashes,
        };
        out.metadata.feature_source = out.observations.name().to_string();
        out.patch_u_halfvec_xyz = Some(patch_u);
        out.patch_v_halfvec_xyz = Some(patch_v);
        out.rebuild_derived_fields();
        out.validate_observation_columns()
            .map_err(ReconstructionError::Unsupported)?;
        Ok(out)
    }
}

/// Decode a 32-character lowercase/uppercase hex string (an XXH128 digest, as
/// `.sift` records `image_file_xxh128`) into 16 bytes, byte `i` from hex pair
/// `[2i, 2i+2)` — the same `bytes.fromhex` convention the format's image hashes
/// use. `None` if the string is not exactly 32 hex characters.
fn decode_xxh128_hex(s: &str) -> Option<[u8; 16]> {
    if s.len() != 32 {
        return None;
    }
    let mut out = [0u8; 16];
    for (i, byte) in out.iter_mut().enumerate() {
        *byte = u8::from_str_radix(s.get(2 * i..2 * i + 2)?, 16).ok()?;
    }
    Some(out)
}
