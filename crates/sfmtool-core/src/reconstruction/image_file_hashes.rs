// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The hash of each image's source file, wherever the reconstruction keeps it.
//!
//! An `embedded_patches` reconstruction stores it as the `image_file_hashes`
//! column. A `sift_files` one does not, but each image's `.sift` file records
//! the same value, the XXH128 of the photograph's bytes, in its metadata as
//! `image_file_xxh128`. [`SfmrReconstruction::find_image_file_hashes`] reads
//! whichever applies, so a caller that wants to know which photograph an image
//! is, such as image pairing for alignment
//! ([`crate::analysis::alignment::image_pairing`]), reads it the same way for
//! both kinds. No photograph is read or hashed.

use sfmtool_archive_io::parse_hash_bytes;

use crate::progress::{Cancelled, Progress};
use crate::SfmrReconstruction;

impl SfmrReconstruction {
    /// Each image's source-file hash (XXH128 of the photograph's bytes), or
    /// `None` for an image whose hash is not known.
    ///
    /// - `embedded_patches`: the stored `image_file_hashes` column, so every
    ///   image has one.
    /// - `sift_files`: the `image_file_xxh128` the image's `.sift` file records,
    ///   read from the feature folder the reconstruction records
    ///   ([`SfmrReconstruction::sift_path_for_image`]). Only the file's metadata
    ///   is read. An image's hash is `None` when its `.sift` file is missing or
    ///   cannot be read, when its content hash is not the image's
    ///   `sift_content_hashes` entry (it is another extraction, which may be of
    ///   another photograph), or when its `image_file_xxh128` is not a hash; and
    ///   every image's is `None` when the reconstruction has no workspace
    ///   directory. None of these is an error: a caller treats such an image as
    ///   one whose file is unknown.
    ///
    /// Reading a `sift_files` reconstruction's hashes opens one `.sift` file
    /// per image, so `progress` counts the images and can cancel between them.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use sfmtool_core::progress::Progress;
    /// use sfmtool_core::SfmrReconstruction;
    /// # fn run(path: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
    /// let recon = SfmrReconstruction::load(path, &Progress::none())?;
    /// let hashes = recon.find_image_file_hashes(&Progress::none())?;
    /// let known = hashes.iter().filter(|h| h.is_some()).count();
    /// println!("{known} of {} image file hashes are known", hashes.len());
    /// # Ok(())
    /// # }
    /// ```
    pub fn find_image_file_hashes(
        &self,
        progress: &Progress,
    ) -> Result<Vec<Option<[u8; 16]>>, Cancelled> {
        if let Some(hashes) = self.image_file_hashes() {
            return Ok(hashes.iter().copied().map(Some).collect());
        }
        let n = self.image_set.images.len();
        if self.workspace_dir.as_os_str().is_empty() {
            return Ok(vec![None; n]);
        }
        let recorded = self.sift_content_hashes();
        let reading = progress.phase("read image file hashes");
        let mut hashes = Vec::with_capacity(n);
        for i in 0..n {
            reading.check_cancel()?;
            let expected = recorded.and_then(|r| r.get(i).copied());
            hashes.push(self.sift_image_file_hash(i, expected));
            reading.count(i as u64 + 1, Some(n as u64), "image");
        }
        Ok(hashes)
    }

    /// The `image_file_xxh128` image `index`'s `.sift` file records, when the
    /// file can be read, is the extraction `expected` names (when given) and
    /// records a well-formed hash.
    fn sift_image_file_hash(&self, index: usize, expected: Option<[u8; 16]>) -> Option<[u8; 16]> {
        let path = self.sift_path_for_image(index);
        let (_, metadata, content_hash) = sfmtool_sift_format::read_sift_metadata(&path).ok()?;
        if let Some(expected) = expected {
            if parse_hash_bytes(&content_hash.content_xxh128).ok() != Some(expected) {
                return None;
            }
        }
        parse_hash_bytes(&metadata.image_file_xxh128).ok()
    }
}

#[cfg(test)]
mod tests;
