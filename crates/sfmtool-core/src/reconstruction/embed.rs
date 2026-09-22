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
//!
//! It reads a `.sift` file per image, twice over in the default sizing policy,
//! so it is long enough to report and long enough to want stopping: it takes a
//! [`Progress`] like every other kernel that can outlast a frame, names three
//! stages under it, moves the bar within each of them, and polls the cancel
//! flag between the images and through the passes over the points.

use ndarray::Array2;

use sfmtool_sift_format::{read_sift_metadata, read_sift_positions};

use super::data::ReconstructionError;
use super::{ObservationSource, SfmrReconstruction};
use crate::patch::cloud::{PatchCloud, PatchCloudError, PatchExtent, PatchNormal};
use crate::progress::Progress;
use crate::progress_note;

#[cfg(test)]
mod tests;

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
    /// - **Keypoints:** each observation's inline `keypoints_xy` is carried over
    ///   verbatim. A `sift_files` reconstruction that already holds the optional
    ///   inline column keeps exactly those coordinates; otherwise each one is
    ///   copied from its `.sift` feature (`sift.positions_xy[feature_index]`), so
    ///   the 2D coordinate is the original SIFT detection.
    /// - **Image hashes:** each image's `image_file_hashes` entry is read from its
    ///   `.sift` metadata (`image_file_xxh128`) — a minimal metadata read, no
    ///   re-hashing of the image bytes.
    ///
    /// `progress` is where this call names its three stages -- `patch frames`
    /// (the [`PatchCloud::from_reconstruction`] build, which names stages of
    /// its own underneath), `read keypoints` (one count per image over the
    /// `.sift` detections and image hashes), and `assemble` (the
    /// per-observation keypoint column and the validated output) -- and it is
    /// also how the call is asked to stop: the flag is polled between the
    /// stages, between the images of both `.sift` walks, and at intervals
    /// through the passes over the points and the observations, and a cancelled
    /// conversion returns [`ReconstructionError::Cancelled`] with nothing
    /// built. Every stage reports a fraction, so the bar moves throughout. Pass
    /// `&Progress::none()` to report nothing and never stop.
    ///
    /// Errors with [`ReconstructionError::Unsupported`] if the reconstruction is
    /// already `embedded_patches` (no `.sift` to copy from) or the patch frame
    /// cannot be built, and with [`ReconstructionError::SiftRead`] if a `.sift`
    /// file cannot be read or lacks a feature an observation references.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use sfmtool_core::patch::cloud::{PatchExtent, PatchNormal};
    /// use sfmtool_core::progress::Progress;
    /// use sfmtool_core::SfmrReconstruction;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let recon = SfmrReconstruction::load("run.sfmr".as_ref(), &Progress::none())?;
    /// let embedded = recon.to_embedded_patches(
    ///     PatchNormal::MeanViewing,
    ///     PatchExtent::default(),
    ///     &Progress::none(),
    /// )?;
    /// # let _ = embedded;
    /// # Ok(())
    /// # }
    /// ```
    pub fn to_embedded_patches(
        &self,
        normal: PatchNormal,
        extent: PatchExtent,
        progress: &Progress<'_>,
    ) -> Result<Self, ReconstructionError> {
        if let ObservationSource::EmbeddedPatches { .. } = &self.point_set.observations {
            return Err(ReconstructionError::Unsupported(
                "to_embedded_patches: reconstruction is already embedded_patches; \
                 there is no .sift to copy keypoints from"
                    .to_string(),
            ));
        }

        let (feature_indexes, inline_keypoints) = match &self.point_set.observations {
            ObservationSource::SiftFiles {
                feature_indexes,
                keypoints_xy,
                ..
            } => (feature_indexes, keypoints_xy.as_ref()),
            // Unreachable: the embedded case returned above.
            ObservationSource::EmbeddedPatches { .. } => unreachable!(),
        };

        // The three stages share the bar in proportion to what they cost, and
        // the frame build is the largest. Both file-walking stages read a
        // `.sift` per image and neither expands a descriptor: the frame build's
        // walk takes each file's affine shapes for the keypoint scales
        // `FeatureSize` sizes from, where the keypoint read takes the positions
        // and the metadata, so the two are within a factor of two of each other
        // and what separates the stages is the sizing and framing the first one
        // does afterwards. On a 4054-image, 1.07M-point, 16.3M-observation
        // capture that is 4.5 s of framing against 1.9 s of reading and 0.92 s
        // of assembly with the files in cache, and 5.5 s against 2.5 s and
        // 1.1 s on a colder one. The weights sit between the two, since what
        // the cache changes is how much the first stage dominates by and not
        // which one does, and they are exact eighths because three weights that
        // sum to one only approximately leave the finished bar a hair short of
        // its end.
        let [framing, reading, assembling] = progress.split([0.625, 0.25, 0.125]);
        progress.check_cancel()?;

        // Patch frames from the chosen normal/extent policy — no refinement.
        // Build frames for every point: finite surfels plus the tangent-sphere
        // frames for points at infinity (exclude_points_at_infinity = false), so
        // every point ends up with a real (non-zero) frame.
        let (patch_u, patch_v) = {
            let mut phase = framing.phase("patch frames");
            let cloud = PatchCloud::from_reconstruction(self, normal, extent, false, &phase)
                .map_err(|e| match e {
                    // The one failure that is not a refusal: the build was
                    // asked to stop, which this call reports in its own words.
                    PatchCloudError::Cancelled => ReconstructionError::Cancelled,
                    e => ReconstructionError::Unsupported(format!(
                        "to_embedded_patches: building patch frames failed: {e}"
                    )),
                })?;
            progress_note!(phase, "{} points", self.point_set.points.len());
            cloud.to_halfvec_arrays(self.point_set.points.len())
        };

        // Per-image: a minimal keypoint read plus the source-image identity hash.
        let n_images = self.image_table.images.len();
        let mut reading = reading.phase("read keypoints");
        progress_note!(reading, "{n_images} images");
        let mut positions_per_image: Vec<Vec<[f32; 2]>> = Vec::with_capacity(n_images);
        let mut image_file_hashes: Vec<[u8; 16]> = Vec::with_capacity(n_images);
        for i in 0..n_images {
            // Between the images rather than inside one file's read: a `.sift`
            // read is one call, so this is where a cancel can land.
            reading.check_cancel()?;
            let path = self.sift_path_for_image(i);
            // The detections are needed only when the reconstruction states no
            // coordinates of its own; the image hash below is read either way.
            let positions = match inline_keypoints {
                Some(_) => Vec::new(),
                None => {
                    let count = self.point_set.max_track_feature_index[i] as usize + 1;
                    read_sift_positions(&path, count).map_err(|e| {
                        ReconstructionError::SiftRead {
                            path: path.clone(),
                            source: e.to_string(),
                        }
                    })?
                }
            };
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
            reading.count(i as u64 + 1, Some(n_images as u64), "image");
        }
        drop(reading);

        // Per-observation keypoints, parallel to `tracks` (and thus to the
        // feature_indexes column), so the existing track ordering is preserved.
        progress.check_cancel()?;
        let mut assembling = assembling.phase("assemble");
        let m = self.point_set.tracks.len();
        let keypoints_xy = match inline_keypoints {
            Some(inline) => inline.clone(),
            None => {
                let mut keypoints_xy = Array2::<f32>::zeros((m, 2));
                // A scatter over sixteen million observations, so the bar moves
                // and the flag is read on a boundary rather than per row: one
                // report per `step` is two hundred across the stage.
                let step = (m / 200).max(1);
                for (j, obs) in self.point_set.tracks.iter().enumerate() {
                    if j.is_multiple_of(step) {
                        assembling.check_cancel()?;
                        assembling.set_fraction(j as f32 / m.max(1) as f32);
                    }
                    let img = obs.image_index as usize;
                    let fidx = feature_indexes[j] as usize;
                    let pos = positions_per_image[img].get(fidx).ok_or_else(|| {
                        ReconstructionError::SiftRead {
                            path: self.sift_path_for_image(img),
                            source: format!(
                                "observation {j} references feature {fidx} of image {img}, but \
                                 only {} features were read",
                                positions_per_image[img].len()
                            ),
                        }
                    })?;
                    keypoints_xy[[j, 0]] = pos[0];
                    keypoints_xy[[j, 1]] = pos[1];
                }
                keypoints_xy
            }
        };

        let mut out = self.clone_for_edit();
        out.point_set.observations = ObservationSource::EmbeddedPatches {
            keypoints_xy,
            image_file_hashes,
        };
        out.metadata.feature_source = out.point_set.observations.name().to_string();
        out.point_set.patch_u_halfvec_xyz = Some(patch_u);
        out.point_set.patch_v_halfvec_xyz = Some(patch_v);
        out.rebuild_derived_fields();
        out.validate_observation_columns()
            .map_err(ReconstructionError::Unsupported)?;
        progress_note!(assembling, "{m} observations");
        // The stage is over, and its last boundary was up to `step` rows short
        // of the end; this is also the operation's own end, so the bar fills.
        assembling.set_fraction(1.0);
        Ok(out)
    }
}

/// Decode a 32-character lowercase/uppercase hex string (an XXH128 digest, as
/// `.sift` records `image_file_xxh128`) into 16 bytes, byte `i` from hex pair
/// `[2i, 2i+2)` — the same `bytes.fromhex` convention the format's image hashes
/// use. `None` if the string is not exactly 32 hex characters.
pub(crate) fn decode_xxh128_hex(s: &str) -> Option<[u8; 16]> {
    if s.len() != 32 {
        return None;
    }
    let mut out = [0u8; 16];
    for (i, byte) in out.iter_mut().enumerate() {
        *byte = u8::from_str_radix(s.get(2 * i..2 * i + 2)?, 16).ok()?;
    }
    Some(out)
}
