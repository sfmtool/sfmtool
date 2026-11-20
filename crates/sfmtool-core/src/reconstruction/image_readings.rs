// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The image readings a `.sfmr` save writes, gathered from where they already
//! are.
//!
//! A `.sfmr` file stores a sharpness and a gradient structure tensor for
//! every image, under one `image_to_gray` conversion (see
//! `specs/formats/sfmr-file-format.md` § "Where a writer gets the values").
//! [`SfmrReconstruction::gather_image_readings`] takes each image's row from
//! the first of these that applies:
//!
//! 1. the reconstruction itself, when the image already holds a row under the
//!    conversion the save records ([`SfmrImage::readings`]);
//! 2. the image's `.sift` file, when it verifiably belongs to the
//!    image and records the same conversion: its two readings are copied, the
//!    way [`verified_sift_thumbnail`](super::thumbnail::verified_sift_thumbnail)
//!    copies its thumbnail;
//! 3. the photograph, decoded in stored pixel order ([`ImageU8::read_rgb`]),
//!    checked against the reconstruction's record of its hash, and read by
//!    [`image_readings_from_rgb`].
//!
//! When none applies to an image, the gather fails and names each such image
//! with what it tried. [`SfmrReconstruction::save`] gathers the rows a value
//! lacks before it writes, and [`SfmrReconstruction::fill_image_readings`]
//! stores them in the value.
//!
//! [`SfmrImage::readings`]: crate::SfmrImage::readings

use std::fmt;
use std::path::Path;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Mutex;

use rayon::prelude::*;
use sfmtool_archive_io::parse_hash_bytes;

use crate::camera::image::{image_dimensions, ImageU8};
use crate::features::sift::{image_readings_from_rgb, ImageReadings, ImageToGray};
use crate::progress::Progress;
use crate::workspace::recorded_image_to_gray;
use crate::SfmrReconstruction;

/// Where one image's row came from in [`SfmrReconstruction::gather_image_readings`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageReadingSource {
    /// The reconstruction already held it under the conversion recorded.
    Reconstruction,
    /// Copied from the image's verified `.sift` file.
    Sift,
    /// Computed from the verified photograph.
    Photograph,
}

/// Every image's row and the conversion they were computed through, from
/// [`SfmrReconstruction::gather_image_readings`].
#[derive(Debug, Clone, PartialEq)]
pub struct GatheredImageReadings {
    /// The conversion the rows were computed through, which a save records.
    pub image_to_gray: ImageToGray,
    /// One row per image, in image order.
    pub rows: Vec<ImageReadings>,
    /// Where each row came from, parallel to `rows`.
    pub sources: Vec<ImageReadingSource>,
}

impl GatheredImageReadings {
    /// How many rows came from `source`.
    pub fn count(&self, source: ImageReadingSource) -> usize {
        self.sources.iter().filter(|&&s| s == source).count()
    }

    /// The rows as the `.sfmr` image-readings columns.
    pub fn to_columns(&self) -> sfmtool_sfmr_format::ImageReadingColumns {
        image_reading_columns(&self.image_to_gray, &self.rows)
    }
}

/// The `.sfmr` image-readings columns of `rows`, computed through
/// `image_to_gray`.
pub(crate) fn image_reading_columns(
    image_to_gray: &ImageToGray,
    rows: &[ImageReadings],
) -> sfmtool_sfmr_format::ImageReadingColumns {
    sfmtool_sfmr_format::ImageReadingColumns {
        image_to_gray: image_to_gray.to_json(),
        sharpness: rows.iter().map(|r| r.sharpness).collect(),
        gradient_structure_tensors_xx_yy_xy: ndarray::Array2::from_shape_fn(
            (rows.len(), 3),
            |(i, k)| rows[i].gradient_structure_tensor_xx_yy_xy[k],
        ),
    }
}

/// One image no source could supply a row for, with what was tried.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnavailableImageReadings {
    /// The image's name, as the reconstruction records it.
    pub name: String,
    /// Why the reconstruction's own row was not used: it holds none, or holds
    /// one under no recorded conversion.
    pub reconstruction: String,
    /// Why its `.sift` file was not used: the path tried and the reason.
    pub sift: String,
    /// Why its photograph was not used: the path tried and the reason.
    pub photograph: String,
}

/// Why [`SfmrReconstruction::gather_image_readings`] gathered no readings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ImageReadingsError {
    /// No source could supply a row for these images; nothing was gathered.
    Unavailable {
        /// The conversion the rows were wanted under, as JSON text.
        image_to_gray: String,
        /// The image count of the reconstruction.
        image_count: usize,
        /// Each image without a row, in image order.
        images: Vec<UnavailableImageReadings>,
    },
    /// The progress was cancelled.
    Cancelled,
}

impl fmt::Display for ImageReadingsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unavailable {
                image_to_gray,
                image_count,
                images,
            } => {
                write!(
                    f,
                    "a .sfmr file stores a sharpness and a gradient structure tensor for every \
                     image, and {} of {image_count} images have none in the reconstruction, in \
                     a matching .sift file or from their photograph (image_to_gray \
                     {image_to_gray}):",
                    images.len()
                )?;
                for image in images {
                    write!(
                        f,
                        "\n    {}: {}; {}; {}",
                        image.name, image.reconstruction, image.sift, image.photograph
                    )?;
                }
                Ok(())
            }
            Self::Cancelled => write!(f, "gathering the image readings was cancelled"),
        }
    }
}

impl std::error::Error for ImageReadingsError {}

impl SfmrReconstruction {
    /// The `image_to_gray` a save of this value records for its image
    /// readings.
    ///
    /// The value's own [`ImageSet::image_to_gray`](crate::ImageSet::image_to_gray)
    /// when it records one; otherwise the workspace record's
    /// `feature_options.image_to_gray` (`metadata.workspace.contents`);
    /// otherwise the one the first image whose `.sift` file reads records;
    /// otherwise the conversion the workspace record states for its feature
    /// tool ([`recorded_image_to_gray`]); and otherwise the default,
    /// BT.709 luma unrounded ([`ImageToGray::default`]).
    pub fn image_to_gray_for_save(&self) -> ImageToGray {
        if let Some(conversion) = &self.image_set.image_to_gray {
            return conversion.clone();
        }
        let contents = &self.metadata.workspace.contents;
        if let Some(value) = contents.feature_options.get("image_to_gray") {
            if let Ok(conversion) = ImageToGray::from_json(value) {
                return conversion;
            }
        }
        if !self.workspace_dir.as_os_str().is_empty() {
            for index in 0..self.image_set.images.len() {
                let path = self.sift_path_for_image(index);
                if !path.is_file() {
                    continue;
                }
                let Ok((feature_tool, _, _)) = sfmtool_sift_format::read_sift_metadata(&path)
                else {
                    continue;
                };
                if let Some(conversion) = feature_tool
                    .feature_options
                    .get("image_to_gray")
                    .and_then(|v| ImageToGray::from_json(v).ok())
                {
                    return conversion;
                }
            }
        }
        recorded_image_to_gray(&contents.feature_tool, &contents.feature_options)
            .unwrap_or_default()
    }

    /// Every image's sharpness and gradient structure tensor under
    /// [`Self::image_to_gray_for_save`], from the first source that has it: the
    /// value itself, the image's verified `.sift` file recorded
    /// under the same conversion, or the verified photograph (see the module
    /// docs).
    ///
    /// The `.sift` file is the one [`Self::sift_path_for_image`] names. It is
    /// the image's when its stored content hash is the image's
    /// `sift_content_hashes` entry (`sift_files`), or when its recorded
    /// `image_file_xxh128` is the image's `image_file_hashes` entry
    /// (`embedded_patches`). The photograph is `workspace_dir` joined with
    /// the image name, and it is used only when its bytes hash to the
    /// reconstruction's record of it: the matching `.sift`'s
    /// `image_file_xxh128` in a `sift_files` value, which a value without its
    /// `.sift` files therefore cannot verify, and `image_file_hashes` in an
    /// `embedded_patches` one.
    ///
    /// The images are read on the rayon pool in two passes, and `progress`
    /// counts them. The first takes every row that needs no decode, all in
    /// parallel, and reads the size of each photograph left from its header.
    /// The second decodes those photographs, at most as many at once as fit
    /// in half of physical memory at about 16 bytes per pixel of the largest
    /// of them, and never more than the core count: the rule `.sift`
    /// building sizes its images in flight by. A JPEG's decode and readings
    /// take under a tenth of a second at 12 megapixels and under a
    /// second at 200, when one is counted at about 3 GiB.
    ///
    /// # Errors
    ///
    /// [`ImageReadingsError::Unavailable`] naming every image no source could
    /// supply, with the `.sift` path and the photograph path tried and why
    /// each was not used; [`ImageReadingsError::Cancelled`] when `progress`
    /// is cancelled.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use sfmtool_core::progress::Progress;
    /// use sfmtool_core::reconstruction::image_readings::ImageReadingSource;
    /// use sfmtool_core::SfmrReconstruction;
    /// # fn run(path: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
    /// let recon = SfmrReconstruction::load(path, &Progress::none())?;
    /// let gathered = recon.gather_image_readings(&Progress::none())?;
    /// println!(
    ///     "{} rows from .sift files, {} from photographs",
    ///     gathered.count(ImageReadingSource::Sift),
    ///     gathered.count(ImageReadingSource::Photograph)
    /// );
    /// # Ok(())
    /// # }
    /// ```
    pub fn gather_image_readings(
        &self,
        progress: &Progress<'_>,
    ) -> Result<GatheredImageReadings, ImageReadingsError> {
        self.gather_image_readings_decoding(progress, None)
    }

    /// [`Self::gather_image_readings`] with at most `concurrent_decodes`
    /// photographs decoded at once, or the number
    /// [`concurrent_decodes_for`] gives when `None`.
    fn gather_image_readings_decoding(
        &self,
        progress: &Progress<'_>,
        concurrent_decodes: Option<usize>,
    ) -> Result<GatheredImageReadings, ImageReadingsError> {
        let image_to_gray = self.image_to_gray_for_save();
        let own_rows = self.image_set.image_to_gray.as_ref() == Some(&image_to_gray);
        let total = self.image_set.images.len() as u64;
        let done = AtomicU64::new(0);
        let counted = || {
            let n = done.fetch_add(1, Ordering::Relaxed) + 1;
            progress.count(n, Some(total), "image");
        };
        // A placeholder for an image skipped after a cancel, which the cancel
        // check after each pass discards.
        let skipped = || {
            Err(UnavailableImageReadings {
                name: String::new(),
                reconstruction: String::new(),
                sift: String::new(),
                photograph: String::new(),
            })
        };

        // First every row that needs no decode, all in parallel, which leaves
        // the photographs to decode.
        let first: Vec<FirstPass> = (0..self.image_set.images.len())
            .into_par_iter()
            .map(|index| {
                if progress.is_cancelled() {
                    return FirstPass::Done(skipped());
                }
                let image = &self.image_set.images[index];
                let found = match image.readings.filter(|_| own_rows) {
                    Some(row) => FirstPass::Done(Ok((row, ImageReadingSource::Reconstruction))),
                    None => {
                        let reconstruction = if image.readings.is_some() {
                            "the reconstruction's row is under no recorded image_to_gray"
                        } else {
                            "the reconstruction holds no row for it"
                        };
                        match self.image_readings_before_decode(index, &image_to_gray) {
                            FirstPass::Done(found) => {
                                FirstPass::Done(found.map_err(|mut unavailable| {
                                    unavailable.reconstruction = reconstruction.into();
                                    unavailable
                                }))
                            }
                            FirstPass::Photograph(mut photograph) => {
                                photograph.unavailable.reconstruction = reconstruction.into();
                                return FirstPass::Photograph(photograph);
                            }
                        }
                    }
                };
                counted();
                found
            })
            .collect();
        if progress.check_cancel().is_err() {
            return Err(ImageReadingsError::Cancelled);
        }

        // Then the photographs, with the number decoded at once bounded by
        // memory.
        let mut results = Vec::with_capacity(first.len());
        let mut photographs = Vec::new();
        for (index, pass) in first.into_iter().enumerate() {
            match pass {
                FirstPass::Done(found) => results.push(found),
                FirstPass::Photograph(photograph) => {
                    results.push(skipped());
                    photographs.push((index, photograph));
                }
            }
        }
        if !photographs.is_empty() {
            let k = concurrent_decodes
                .unwrap_or_else(|| {
                    concurrent_decodes_for(photographs.iter().map(|(_, p)| p.dimensions))
                })
                .clamp(1, photographs.len());
            let decoded = map_at_most(&photographs, k, |(_, photograph)| {
                if progress.is_cancelled() {
                    return skipped();
                }
                let found = photograph.readings(&image_to_gray);
                counted();
                found
            });
            for ((index, _), found) in photographs.iter().zip(decoded) {
                results[*index] = found;
            }
            if progress.check_cancel().is_err() {
                return Err(ImageReadingsError::Cancelled);
            }
        }
        let mut rows = Vec::with_capacity(results.len());
        let mut sources = Vec::with_capacity(results.len());
        let mut unavailable = Vec::new();
        for result in results {
            match result {
                Ok((row, source)) => {
                    rows.push(row);
                    sources.push(source);
                }
                Err(image) => unavailable.push(image),
            }
        }
        if !unavailable.is_empty() {
            return Err(ImageReadingsError::Unavailable {
                image_to_gray: image_to_gray.to_json().to_string(),
                image_count: self.image_set.images.len(),
                images: unavailable,
            });
        }
        Ok(GatheredImageReadings {
            image_to_gray,
            rows,
            sources,
        })
    }

    /// Give every image of this value a readings row, gathered by
    /// [`Self::gather_image_readings`], and record their conversion. A value
    /// that already holds a row for every image is left as it is.
    ///
    /// # Errors
    ///
    /// As [`Self::gather_image_readings`]; the value is then unchanged.
    pub fn fill_image_readings(
        &mut self,
        progress: &Progress<'_>,
    ) -> Result<GatheredImageReadings, ImageReadingsError> {
        let gathered = self.gather_image_readings(progress)?;
        for (image, row) in self.image_set.images.iter_mut().zip(&gathered.rows) {
            image.readings = Some(*row);
        }
        self.image_set.image_to_gray = Some(gathered.image_to_gray.clone());
        Ok(gathered)
    }

    /// Image `index`'s readings under `image_to_gray` from its `.sift` file,
    /// or its photograph to read them from, or what was tried for each when
    /// there is no workspace directory to look in.
    fn image_readings_before_decode(&self, index: usize, image_to_gray: &ImageToGray) -> FirstPass {
        let name = self.image_set.images[index].name.clone();
        if self.workspace_dir.as_os_str().is_empty() {
            return FirstPass::Done(Err(UnavailableImageReadings {
                name,
                reconstruction: String::new(),
                sift: "the reconstruction has no workspace directory to find a .sift file in"
                    .into(),
                photograph: "nor a photograph".into(),
            }));
        }
        // The hash the photograph must have, once known.
        let mut image_hash = self.image_file_hashes().map(|hashes| hashes[index]);
        let sift = self.sift_path_for_image(index);
        let sift_reason = match self.sift_readings(index, &sift, image_to_gray, &mut image_hash) {
            Ok(row) => return FirstPass::Done(Ok((row, ImageReadingSource::Sift))),
            Err(reason) => format!("{}: {reason}", sift.display()),
        };
        let path = self.workspace_dir.join(&name);
        FirstPass::Photograph(PhotographToRead {
            dimensions: image_dimensions(&path).ok(),
            path,
            image_hash,
            unavailable: UnavailableImageReadings {
                name,
                reconstruction: String::new(),
                sift: sift_reason,
                photograph: String::new(),
            },
        })
    }

    /// The readings image `index`'s `.sift` at `path` records under
    /// `image_to_gray`, or why it does not supply them. A `.sift` that belongs
    /// to the image sets `image_hash` to the photograph hash it records, when
    /// `image_hash` is not already known.
    fn sift_readings(
        &self,
        index: usize,
        path: &Path,
        image_to_gray: &ImageToGray,
        image_hash: &mut Option<[u8; 16]>,
    ) -> Result<ImageReadings, String> {
        if !path.is_file() {
            return Err("no such file".into());
        }
        let (feature_tool, metadata, content_hash) =
            sfmtool_sift_format::read_sift_metadata(path).map_err(|e| e.to_string())?;
        let recorded_image_hash = parse_hash_bytes(&metadata.image_file_xxh128).ok();
        if let Some(expected) = *image_hash {
            if recorded_image_hash != Some(expected) {
                return Err(
                    "its image_file_xxh128 is not the image's image_file_hashes entry, \
                     so it was extracted from another photograph"
                        .into(),
                );
            }
        } else {
            let expected = self.sift_content_hashes().map(|hashes| hashes[index]);
            if parse_hash_bytes(&content_hash.content_xxh128).ok() != expected {
                return Err(
                    "its content hash differs from the one the reconstruction records, \
                     so it is another extraction"
                        .into(),
                );
            }
            *image_hash = recorded_image_hash;
        }
        let recorded = feature_tool.feature_options.get("image_to_gray");
        let same = recorded
            .and_then(|v| ImageToGray::from_json(v).ok())
            .is_some_and(|c| c == *image_to_gray);
        if !same {
            return Err(format!(
                "its readings were computed through image_to_gray {}, not {}",
                recorded.map_or_else(|| "(none)".to_string(), |v| v.to_string()),
                image_to_gray.to_json()
            ));
        }
        // A `.sift` stores each reading as a double that holds the `f32`
        // exactly, so the conversion back is exact.
        let [xx, yy, xy] = metadata.gradient_structure_tensor_xx_yy_xy;
        Ok(ImageReadings {
            sharpness: metadata.sharpness as f32,
            gradient_structure_tensor_xx_yy_xy: [xx as f32, yy as f32, xy as f32],
        })
    }
}

/// The readings of the photograph at `path` under `image_to_gray`, after
/// checking that its bytes hash to `image_hash`, or why they cannot be taken.
fn photograph_readings(
    path: &Path,
    image_hash: Option<[u8; 16]>,
    image_to_gray: &ImageToGray,
) -> Result<ImageReadings, String> {
    if !path.is_file() {
        return Err("no such file".into());
    }
    let Some(expected) = image_hash else {
        return Err(
            "no record of the photograph's hash to check it against, since the image's \
             .sift file was not usable"
                .into(),
        );
    };
    let (hash, _) = crate::features::sift::xxh128_of_file(path).map_err(|e| e.to_string())?;
    if parse_hash_bytes(&hash).ok() != Some(expected) {
        return Err(
            "its bytes do not hash to the reconstruction's record, so it is not the \
             photograph the reconstruction was built from"
                .into(),
        );
    }
    let image = ImageU8::read_rgb(path).map_err(|e| format!("it could not be decoded: {e}"))?;
    image_readings_from_rgb(image.width(), image.height(), image.data(), image_to_gray)
        .map_err(|e| e.to_string())
}

/// About how many bytes one photograph holds per source pixel while
/// [`photograph_readings`] reads it: the decoded RGB frame, 3 bytes, and what
/// the decoder holds beside it, such as a JPEG's component planes or a PNG's
/// RGBA or 16-bit frame before its conversion to 8-bit RGB. The readings make
/// no grey copy of the frame, only the 480-pixel reading image. Measured at
/// 4.7 to 5.4 for baseline JPEGs and up to 11.1 for a 16-bit RGBA PNG, rounded
/// up, since guessing high only gives up a few concurrent decodes while
/// guessing low risks running out of memory. The measurements are in
/// `specs/formats/sfmr-file-format.md` § "Implementations".
const BYTES_PER_SOURCE_PIXEL: u64 = 16;

/// One image's outcome in the first pass of
/// [`SfmrReconstruction::gather_image_readings`], which decodes nothing.
enum FirstPass {
    /// Its row, or what was tried, with nothing left to try.
    Done(Result<(ImageReadings, ImageReadingSource), UnavailableImageReadings>),
    /// Its row is to come from its photograph, in the second pass.
    Photograph(PhotographToRead),
}

/// A photograph the second pass of
/// [`SfmrReconstruction::gather_image_readings`] decodes.
struct PhotographToRead {
    /// Where it is.
    path: std::path::PathBuf,
    /// The hash its bytes must have, when the reconstruction records one.
    image_hash: Option<[u8; 16]>,
    /// Its size from its header, when that can be read.
    dimensions: Option<(u32, u32)>,
    /// What was tried before it, which the photograph's own reason completes
    /// when it cannot be read either.
    unavailable: UnavailableImageReadings,
}

impl PhotographToRead {
    /// Its readings under `image_to_gray`, or what was tried for its image.
    fn readings(
        &self,
        image_to_gray: &ImageToGray,
    ) -> Result<(ImageReadings, ImageReadingSource), UnavailableImageReadings> {
        photograph_readings(&self.path, self.image_hash, image_to_gray)
            .map(|row| (row, ImageReadingSource::Photograph))
            .map_err(|reason| UnavailableImageReadings {
                photograph: format!("{}: {reason}", self.path.display()),
                ..self.unavailable.clone()
            })
    }
}

/// About how many bytes the largest of the photographs of `dimensions` holds
/// while it is read, at [`BYTES_PER_SOURCE_PIXEL`], or `None` when none of
/// their sizes is known and above zero. It saturates rather than overflows.
fn decode_footprint(dimensions: impl IntoIterator<Item = Option<(u32, u32)>>) -> Option<u64> {
    dimensions
        .into_iter()
        .flatten()
        .map(|(width, height)| u64::from(width) * u64::from(height))
        .filter(|&pixels| pixels > 0)
        .max()
        .map(|pixels| pixels.saturating_mul(BYTES_PER_SOURCE_PIXEL))
}

/// How many of the photographs of `dimensions` to decode at once: as many as
/// fit in half of physical memory at the largest one's
/// [`decode_footprint`], at most the core count and at least 1, the rule
/// `.sift` building is sized by.
fn concurrent_decodes_for(dimensions: impl IntoIterator<Item = Option<(u32, u32)>>) -> usize {
    crate::features::sift::concurrent_images_in_memory(decode_footprint(dimensions))
}

/// `f` of each of `items`, in order, run on the rayon pool with at most `k`
/// calls in flight at once.
///
/// It spawns `k` jobs in a [`rayon::scope`], and each takes the next item
/// until none is left. The bound is on the jobs rather than on threads: a
/// thread that waits inside one call for rayon work the call started may
/// steal another of the jobs and start a call of its own, but no more than
/// `k` jobs exist, and each runs one call at a time. Nothing in it blocks but
/// the scope, which a rayon thread waits on by running the pool's work, so it
/// can be called from inside the rayon pool, even one of a single thread, and
/// it adds no threads of its own.
fn map_at_most<T: Sync, R: Send>(items: &[T], k: usize, f: impl Fn(&T) -> R + Sync) -> Vec<R> {
    let next = AtomicUsize::new(0);
    let found = Mutex::new(Vec::with_capacity(items.len()));
    rayon::scope(|scope| {
        for _ in 0..k.max(1) {
            scope.spawn(|_| loop {
                let index = next.fetch_add(1, Ordering::Relaxed);
                let Some(item) = items.get(index) else {
                    return;
                };
                let result = f(item);
                found.lock().unwrap().push((index, result));
            });
        }
    });
    let mut found = found.into_inner().unwrap();
    found.sort_unstable_by_key(|(index, _)| *index);
    found.into_iter().map(|(_, result)| result).collect()
}

#[cfg(test)]
mod tests;
