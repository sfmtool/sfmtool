// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Convert a version 1 `.sift` file to version 2 without extracting its
//! features again.
//!
//! A version 1 file records no `image_to_gray` and no image readings. The
//! conversion copies its positions, affine shapes, descriptors and thumbnail
//! byte for byte, upgrades its `feature_options` with
//! [`upgrade_feature_options`], and
//! takes the two readings from the image file through the upgraded options'
//! `image_to_gray`, after checking the image against the file's
//! `image_file_xxh128`. See `specs/formats/sift-file-format.md` § "Converting
//! a version 1 file", and `specs/drafts/ws-upgrade-files-amendment.md`, which
//! plans a workspace upgrade that converts every version 1 file this way.
//!
//! The files are written into a [`Version2SiftFolder`], a folder checked once
//! to hold no version 1 `.sift` file, since the two versions never share a
//! folder. The check opens every `.sift` file in the folder, so it is made when
//! the folder is prepared rather than for each file converted into it.

use std::fmt;
use std::path::{Path, PathBuf};

use sfmtool_sift_format::{SiftContentHash, SiftError, SIFT_FORMAT_VERSION, SIFT_ZSTD_LEVEL};

use super::files::{version_1_sift_files, xxh128_of_file, Version1SiftFiles};
use super::image_readings::{image_readings_from_rgb, ImageReadingsError};
use crate::camera::image::ImageU8;
use crate::progress::{Cancelled, Progress};
use crate::workspace::{recorded_image_to_gray, upgrade_feature_options};

/// Why a version 1 `.sift` file could not be converted, or a folder could not
/// be prepared for the converted files.
#[derive(Debug)]
pub enum ConvertSiftFileError {
    /// The destination folder holds version 1 `.sift` files, and a version 2
    /// file never goes beside one.
    Version1Files(Version1SiftFiles),
    /// The destination folder is there but could not be listed.
    ListFolder {
        /// The folder.
        folder: PathBuf,
        /// What went wrong.
        problem: String,
    },
    /// The destination folder could not be made.
    CreateFolder {
        /// The folder.
        folder: PathBuf,
        /// What went wrong.
        problem: String,
    },
    /// The `.sift` file could not be read.
    Read {
        /// The `.sift` file.
        path: PathBuf,
        /// The format crate's error.
        source: SiftError,
    },
    /// The `.sift` file is not version 1, so there is nothing to convert.
    NotVersion1 {
        /// The `.sift` file.
        path: PathBuf,
        /// The version it is.
        version: u32,
    },
    /// The file's `feature_options` cannot be made version 2 ones: its
    /// feature tool is `sfmtool-undistort` or one this build does not know,
    /// or its `gray_formula` does not parse.
    FeatureOptions {
        /// The `.sift` file.
        path: PathBuf,
        /// Why the options could not be upgraded.
        error: crate::workspace::FeatureOptionsError,
    },
    /// The image file the `.sift` file was extracted from is not there.
    ImageMissing {
        /// The `.sift` file.
        sift_path: PathBuf,
        /// The image file.
        image_path: PathBuf,
    },
    /// The image file is there but could not be read or decoded.
    ImageUnreadable {
        /// The image file.
        image_path: PathBuf,
        /// What went wrong.
        problem: String,
    },
    /// The image file's bytes do not hash to the `.sift` file's
    /// `image_file_xxh128`, so it is not the image the features were
    /// extracted from.
    ImageChanged {
        /// The `.sift` file.
        sift_path: PathBuf,
        /// The image file.
        image_path: PathBuf,
        /// The XXH128 the `.sift` file records.
        recorded_xxh128: String,
        /// The XXH128 of the image file's bytes.
        image_xxh128: String,
    },
    /// The image decodes to another size than the `.sift` file records.
    ImageSize {
        /// The image file.
        image_path: PathBuf,
        /// The width and height the `.sift` file records.
        recorded: (u32, u32),
        /// The width and height of the decoded image.
        decoded: (u32, u32),
    },
    /// The image's readings are not finite under the upgraded conversion.
    Readings {
        /// The image file.
        image_path: PathBuf,
        /// The error of [`image_readings_from_rgb`].
        error: ImageReadingsError,
    },
    /// The version 2 file could not be written, or its hashes read back.
    Write {
        /// The version 2 file.
        path: PathBuf,
        /// The format crate's error.
        source: SiftError,
    },
    /// The caller asked the conversion to stop. Nothing was written.
    Cancelled,
}

impl fmt::Display for ConvertSiftFileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Version1Files(files) => write!(f, "{files}"),
            Self::ListFolder { folder, problem } => {
                write!(
                    f,
                    "failed to list the folder {}: {problem}",
                    folder.display()
                )
            }
            Self::CreateFolder { folder, problem } => write!(
                f,
                "failed to create the folder {}: {problem}",
                folder.display()
            ),
            Self::Read { path, source } => {
                write!(f, "failed to read {}: {source}", path.display())
            }
            Self::NotVersion1 { path, version } => write!(
                f,
                "{} is a version {version} .sift file, and only a version 1 file is converted",
                path.display()
            ),
            Self::FeatureOptions { path, error } => write!(
                f,
                "the feature options of {} cannot be converted to version 2: {error}",
                path.display()
            ),
            Self::ImageMissing {
                sift_path,
                image_path,
            } => write!(
                f,
                "the image {} that {} was extracted from is not there, so its readings cannot \
                 be taken",
                image_path.display(),
                sift_path.display()
            ),
            Self::ImageUnreadable {
                image_path,
                problem,
            } => write!(
                f,
                "failed to read the image {}: {problem}",
                image_path.display()
            ),
            Self::ImageChanged {
                sift_path,
                image_path,
                recorded_xxh128,
                image_xxh128,
            } => write!(
                f,
                "the image {} hashes to {image_xxh128}, and {} records {recorded_xxh128}, so it \
                 is not the image the features were extracted from",
                image_path.display(),
                sift_path.display()
            ),
            Self::ImageSize {
                image_path,
                recorded: (rw, rh),
                decoded: (dw, dh),
            } => write!(
                f,
                "the image {} decodes to {dw} x {dh} pixels, and its .sift file records \
                 {rw} x {rh}",
                image_path.display()
            ),
            Self::Readings { image_path, error } => write!(
                f,
                "failed to take the readings of the image {}: {error}",
                image_path.display()
            ),
            Self::Write { path, source } => {
                write!(f, "failed to write {}: {source}", path.display())
            }
            Self::Cancelled => write!(f, "the conversion was cancelled; nothing was written"),
        }
    }
}

impl std::error::Error for ConvertSiftFileError {}

impl From<Cancelled> for ConvertSiftFileError {
    fn from(_: Cancelled) -> Self {
        Self::Cancelled
    }
}

/// The hashes of a version 1 `.sift` file and of the version 2 file converted
/// from it, from [`Version2SiftFolder::convert_version_1_sift_file`].
///
/// A `.sfmr` or `.matches` file records each `.sift` file's `content_xxh128`
/// and `feature_tool_xxh128`; both change in the conversion, so a caller
/// rewriting those records looks up each old `content_xxh128` here.
#[derive(Debug, Clone)]
pub struct ConvertedSiftFile {
    /// The version 2 file written.
    pub path: PathBuf,
    /// The content hashes the version 1 file stores.
    pub source_content_hash: SiftContentHash,
    /// The content hashes the version 2 file stores.
    pub content_hash: SiftContentHash,
}

/// A folder that version 1 `.sift` files are converted into, checked to hold
/// no version 1 `.sift` file.
///
/// [`Version2SiftFolder::create`] makes the check, with
/// [`version_1_sift_files`], the check every writer of version 2 files makes
/// before it writes into a folder, and makes the folder when it is not there.
/// [`Version2SiftFolder::convert_version_1_sift_file`] then converts one file
/// into it; it takes `&self`, so several threads can convert into one folder
/// at once. The check is not made again for each file, since it opens every
/// `.sift` file in the folder; a version 1 file put into the folder by another
/// program after the check is not noticed.
#[derive(Debug, Clone)]
pub struct Version2SiftFolder {
    folder: PathBuf,
}

impl Version2SiftFolder {
    /// Check that `folder` holds no version 1 `.sift` file and make it when it
    /// is not there.
    ///
    /// # Errors
    ///
    /// [`ConvertSiftFileError::Version1Files`] when it holds version 1 files,
    /// [`ConvertSiftFileError::ListFolder`] when it is there but cannot be
    /// listed, and [`ConvertSiftFileError::CreateFolder`] when it cannot be
    /// made.
    pub fn create(folder: &Path) -> Result<Self, ConvertSiftFileError> {
        let found = version_1_sift_files(folder).map_err(|e| ConvertSiftFileError::ListFolder {
            folder: folder.to_path_buf(),
            problem: e.to_string(),
        })?;
        if let Some(found) = found {
            return Err(ConvertSiftFileError::Version1Files(found));
        }
        std::fs::create_dir_all(folder).map_err(|e| ConvertSiftFileError::CreateFolder {
            folder: folder.to_path_buf(),
            problem: e.to_string(),
        })?;
        Ok(Self {
            folder: folder.to_path_buf(),
        })
    }

    /// The folder.
    pub fn path(&self) -> &Path {
        &self.folder
    }

    /// Convert the version 1 `.sift` file at `sift_path`, extracted from the
    /// image file at `image_path`, to a version 2 file of the same name in
    /// this folder, and return the two files' content hashes.
    ///
    /// The positions, affine shapes, descriptors and thumbnail are copied byte
    /// for byte, and the feature tool, the feature type and every
    /// `metadata` field but the version are kept. `feature_options` becomes
    /// [`upgrade_feature_options`] of the file's own, and `metadata` gains
    /// the image's `sharpness` and `gradient_structure_tensor_xx_yy_xy`:
    /// [`image_readings_from_rgb`] of the image decoded by
    /// [`ImageU8::read_rgb`], through the upgraded options' `image_to_gray`,
    /// the decode and the function extraction and the `.sfmr` readings gather
    /// use, so the readings equal what extracting the image would record. The
    /// image is first checked against the file's `image_file_xxh128` and its
    /// decoded size against the file's `image_width` and `image_height`. The
    /// file is written by `sfmtool_sift_format::write_sift` at
    /// [`SIFT_ZSTD_LEVEL`], through a temporary file renamed over any file
    /// already at the path, and the source file is not changed.
    ///
    /// `image_path` is the caller's because the `.sift` file does not say
    /// where the image is: the image is in the directory the workspace's
    /// `feature_prefix_dir` is relative to, which the file does not record,
    /// and the `image_name` that version 1 extractors wrote is sometimes a
    /// path relative to another directory (`fisheye_left/frame_001039.jpg`)
    /// rather than the file name.
    ///
    /// It reads a whole `.sift` file, hashes and decodes an image and writes a
    /// whole `.sift` file, which `specs/STANDARDS.md` counts as long-running
    /// work, though one call takes about 0.04 s for a 2040 x 1536 photograph
    /// with 8,000 features; so it takes a `Progress`, through which a caller
    /// converting a workspace's files cancels it. It opens a
    /// `"convert_version_1_sift_file"` detail phase and checks for
    /// cancellation after reading the file's options and before writing.
    ///
    /// ```no_run
    /// use std::path::Path;
    /// use sfmtool_core::features::sift::Version2SiftFolder;
    /// use sfmtool_core::Progress;
    ///
    /// let folder = Version2SiftFolder::create(Path::new("images/features/sift-sfmtool-new"))?;
    /// let converted = folder.convert_version_1_sift_file(
    ///     Path::new("images/features/sift-sfmtool-old/0001.jpg.sift"),
    ///     Path::new("images/0001.jpg"),
    ///     &Progress::none(),
    /// )?;
    /// println!(
    ///     "{} -> {}",
    ///     converted.source_content_hash.content_xxh128, converted.content_hash.content_xxh128
    /// );
    /// # Ok::<(), sfmtool_core::features::sift::ConvertSiftFileError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// [`ConvertSiftFileError::Read`] when the file cannot be read,
    /// [`ConvertSiftFileError::NotVersion1`] when it is not version 1, and
    /// [`ConvertSiftFileError::FeatureOptions`] when its options cannot be
    /// upgraded: an `sfmtool-undistort` file, whose message says to run
    /// `sfm undistort` again, a feature tool this build does not know, or a
    /// `gray_formula` that does not parse.
    /// [`ConvertSiftFileError::ImageMissing`],
    /// [`ConvertSiftFileError::ImageUnreadable`],
    /// [`ConvertSiftFileError::ImageChanged`] and
    /// [`ConvertSiftFileError::ImageSize`] when the image is not there,
    /// cannot be read or decoded, is not the one recorded, or decodes to
    /// another size; [`ConvertSiftFileError::Readings`] when its readings are
    /// not finite; [`ConvertSiftFileError::Write`] when the file cannot be
    /// written or its hashes read back; and
    /// [`ConvertSiftFileError::Cancelled`]. Nothing is written when it fails
    /// before the write.
    pub fn convert_version_1_sift_file(
        &self,
        sift_path: &Path,
        image_path: &Path,
        progress: &Progress,
    ) -> Result<ConvertedSiftFile, ConvertSiftFileError> {
        let progress = progress.detail_phase("convert_version_1_sift_file");
        let mut data = sfmtool_sift_format::read_sift(sift_path).map_err(|source| {
            ConvertSiftFileError::Read {
                path: sift_path.to_path_buf(),
                source,
            }
        })?;
        if data.metadata.version != 1 {
            return Err(ConvertSiftFileError::NotVersion1 {
                path: sift_path.to_path_buf(),
                version: data.metadata.version,
            });
        }
        let options_error = |error| ConvertSiftFileError::FeatureOptions {
            path: sift_path.to_path_buf(),
            error,
        };
        let tool = &data.feature_tool_metadata.feature_tool;
        let feature_options =
            upgrade_feature_options(tool, &data.feature_tool_metadata.feature_options)
                .map_err(options_error)?;
        let image_to_gray =
            recorded_image_to_gray(tool, &feature_options).map_err(options_error)?;
        progress.check_cancel()?;

        if !image_path.is_file() {
            return Err(ConvertSiftFileError::ImageMissing {
                sift_path: sift_path.to_path_buf(),
                image_path: image_path.to_path_buf(),
            });
        }
        let unreadable = |problem: String| ConvertSiftFileError::ImageUnreadable {
            image_path: image_path.to_path_buf(),
            problem,
        };
        let (image_xxh128, _) =
            xxh128_of_file(image_path).map_err(|e| unreadable(e.to_string()))?;
        if image_xxh128 != data.metadata.image_file_xxh128 {
            return Err(ConvertSiftFileError::ImageChanged {
                sift_path: sift_path.to_path_buf(),
                image_path: image_path.to_path_buf(),
                recorded_xxh128: data.metadata.image_file_xxh128.clone(),
                image_xxh128,
            });
        }
        let image = ImageU8::read_rgb(image_path)
            .map_err(|e| unreadable(format!("it could not be decoded: {e}")))?;
        let recorded = (data.metadata.image_width, data.metadata.image_height);
        let decoded = (image.width(), image.height());
        if decoded != recorded {
            return Err(ConvertSiftFileError::ImageSize {
                image_path: image_path.to_path_buf(),
                recorded,
                decoded,
            });
        }
        let readings =
            image_readings_from_rgb(image.width(), image.height(), image.data(), &image_to_gray)
                .map_err(|error| ConvertSiftFileError::Readings {
                    image_path: image_path.to_path_buf(),
                    error,
                })?;
        drop(image);
        progress.check_cancel()?;

        let source_content_hash = std::mem::take(&mut data.content_hash);
        data.feature_tool_metadata.feature_options = feature_options;
        data.metadata.version = SIFT_FORMAT_VERSION;
        data.metadata.sharpness = Some(f64::from(readings.sharpness));
        data.metadata.gradient_structure_tensor_xx_yy_xy =
            Some(readings.gradient_structure_tensor_xx_yy_xy.map(f64::from));
        let path = self.folder.join(sift_path.file_name().unwrap_or_default());
        let write_error = |source| ConvertSiftFileError::Write {
            path: path.clone(),
            source,
        };
        sfmtool_sift_format::write_sift(&path, &data, SIFT_ZSTD_LEVEL).map_err(write_error)?;
        let (_, _, content_hash) =
            sfmtool_sift_format::read_sift_metadata(&path).map_err(write_error)?;
        Ok(ConvertedSiftFile {
            path,
            source_content_hash,
            content_hash,
        })
    }
}

#[cfg(test)]
mod tests;
