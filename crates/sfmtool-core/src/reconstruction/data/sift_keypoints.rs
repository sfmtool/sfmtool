// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Filling a `sift_files` reconstruction's inline `keypoints_xy` column from its
//! `.sift` companions.
//!
//! A `sift_files` reconstruction names each observation by a feature index into
//! its image's `.sift` file, and may also carry an inline copy of each
//! observation's pixel. Consumers that read pixels (the bench's points source,
//! bundle adjustment, reprojection errors) answer from the inline copy when it
//! is there, so [`SfmrReconstruction::load`] fills it in when a file lacks it,
//! and every later save writes it.

use ndarray::Array2;

use crate::progress::Progress;
use crate::ObservationSource;

use super::super::embed::decode_xxh128_hex;
use super::SfmrReconstruction;

/// What [`SfmrReconstruction::fill_keypoints_from_sift`] did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SiftKeypointFill {
    /// The column was built from the `.sift` files, one row per observation.
    Filled,
    /// The reconstruction already carried the column; nothing was read.
    AlreadyPresent,
    /// The reconstruction is `embedded_patches`, which always carries it.
    NotSiftFiles,
    /// An image's `.sift` could not supply its observations' pixels, so the
    /// column stays absent. `reason` says which file and why, in a sentence.
    Unavailable {
        /// The index of the first image whose `.sift` could not be used.
        image: usize,
        /// Why, naming the file.
        reason: String,
    },
    /// The progress was cancelled before every file was read; the column stays
    /// absent.
    Cancelled,
}

impl SfmrReconstruction {
    /// Build the inline `keypoints_xy` column of a `sift_files` reconstruction
    /// that does not carry one, reading each observation's pixel from its
    /// image's `.sift` file.
    ///
    /// The column is all or nothing, since it is one row per observation: when
    /// any image with observations has no `.sift` at
    /// [`sift_path_for_image`](Self::sift_path_for_image), has one whose stored
    /// content hash differs from the reconstruction's `sift_content_hashes`
    /// entry for it, or has fewer features than its observations index, the
    /// reconstruction is left unchanged and the result says which image stopped
    /// it. A `.sift` whose hash differs is a different extraction, so its
    /// feature indexes name other features.
    ///
    /// The stored content hashes are left as they are. They name the file the
    /// value was read from, and the filled pixels are the ones that file's
    /// feature indexes already denote.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use sfmtool_core::progress::Progress;
    /// use sfmtool_core::SfmrReconstruction;
    /// # fn run(recon: &mut SfmrReconstruction) {
    /// let outcome = recon.fill_keypoints_from_sift(&Progress::none());
    /// println!("{outcome:?}; inline keypoints: {}", recon.keypoints_xy().is_some());
    /// # }
    /// ```
    pub fn fill_keypoints_from_sift(&mut self, progress: &Progress<'_>) -> SiftKeypointFill {
        let (feature_indexes, sift_content_hashes) = match &self.point_set.observations {
            ObservationSource::EmbeddedPatches { .. } => return SiftKeypointFill::NotSiftFiles,
            ObservationSource::SiftFiles {
                keypoints_xy: Some(_),
                ..
            } => return SiftKeypointFill::AlreadyPresent,
            ObservationSource::SiftFiles {
                feature_indexes,
                sift_content_hashes,
                ..
            } => (feature_indexes, sift_content_hashes),
        };

        let n_images = self.image_table.images.len();
        let mut observed = vec![false; n_images];
        for obs in &self.point_set.tracks {
            observed[obs.image_index as usize] = true;
        }
        let images: Vec<usize> = (0..n_images).filter(|&i| observed[i]).collect();
        if !images.is_empty() && self.workspace_dir.as_os_str().is_empty() {
            return SiftKeypointFill::Unavailable {
                image: images[0],
                reason: "the reconstruction has no workspace directory to find .sift files in"
                    .to_string(),
            };
        }

        let mut positions: Vec<Vec<[f32; 2]>> = vec![Vec::new(); n_images];
        for (done, &image) in images.iter().enumerate() {
            if progress.check_cancel().is_err() {
                return SiftKeypointFill::Cancelled;
            }
            let path = self.sift_path_for_image(image);
            let unavailable = |reason: String| SiftKeypointFill::Unavailable {
                image,
                reason: format!("{}: {reason}", path.display()),
            };
            if !path.is_file() {
                return unavailable("no such file".to_string());
            }
            let stored = match sfmtool_sift_format::read_sift_metadata(&path) {
                Ok((_, _, content_hash)) => decode_xxh128_hex(&content_hash.content_xxh128),
                Err(e) => return unavailable(e.to_string()),
            };
            if stored != Some(sift_content_hashes[image]) {
                return unavailable(
                    "its content hash differs from the one the reconstruction records".to_string(),
                );
            }
            let needed = self.point_set.max_track_feature_index[image] as usize + 1;
            match sfmtool_sift_format::read_sift_positions(&path, needed) {
                Ok(read) if read.len() == needed => positions[image] = read,
                Ok(read) => {
                    return unavailable(format!(
                        "it has {} features, and the observations index feature {}",
                        read.len(),
                        needed - 1
                    ))
                }
                Err(e) => return unavailable(e.to_string()),
            }
            progress.count(done as u64 + 1, Some(images.len() as u64), "image");
        }

        let mut column = Array2::<f32>::zeros((self.point_set.tracks.len(), 2));
        for (row, (obs, &feature)) in self
            .point_set
            .tracks
            .iter()
            .zip(feature_indexes)
            .enumerate()
        {
            let [x, y] = positions[obs.image_index as usize][feature as usize];
            column[[row, 0]] = x;
            column[[row, 1]] = y;
        }
        if let ObservationSource::SiftFiles { keypoints_xy, .. } = &mut self.point_set.observations
        {
            *keypoints_xy = Some(column);
        }
        SiftKeypointFill::Filled
    }
}

#[cfg(test)]
mod tests;
