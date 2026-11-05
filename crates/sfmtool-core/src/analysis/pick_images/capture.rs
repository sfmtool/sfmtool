// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! What [`super::pick_images`] is given: the capture's sequences of frames,
//! the source that makes their `.sift` files exist, the options, and the
//! errors it returns. See `specs/core/analysis/pick-images.md`.

use std::fmt;
use std::path::PathBuf;

use serde::Serialize;
use sfmtool_sift_format::SiftError;

use super::frame_pair::{FramePairError, FramePairOptions};
use crate::features::kdforest::KdForestParams;
use crate::features::sift::Version1SiftFiles;
use crate::progress::Progress;

/// Where one image of a capture is and where its `.sift` file goes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CaptureImage {
    /// The image file.
    pub image_path: PathBuf,
    /// Its `.sift` file, which the [`FrameSource`] makes exist and
    /// [`super::pick_images`] reads.
    pub sift_path: PathBuf,
}

/// One instant of a capture: one image, or one image per rig sensor with the
/// reference sensor's first.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CaptureFrame {
    /// The file number the frame's images carry.
    pub number: i64,
    /// The frame's images, one per sensor, the reference sensor's first.
    pub images: Vec<CaptureImage>,
}

/// One sequence of frames, in file-number order, with the pattern each
/// sensor's list entry uses.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CaptureSequence {
    /// One pattern per sensor, in the order of [`CaptureFrame::images`]. The
    /// walk does not read them; they say which sequence an error is about,
    /// and a caller writing a list uses them.
    pub patterns: Vec<String>,
    /// The frames, in file-number order.
    pub frames: Vec<CaptureFrame>,
}

/// One frame of a capture: the index of its sequence and its position in that
/// sequence, from 0. A position, not a file number, so that the rate of a
/// sequence numbered in steps of 2 is that of one numbered in steps of 1.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
pub struct FrameId {
    /// The index of the sequence in the slice [`super::pick_images`] is given.
    pub sequence: usize,
    /// The position of the frame in its sequence, from 0.
    pub frame: usize,
}

/// Why a [`FrameSource`] could not make the files exist, in its own words.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FrameSourceError {
    /// The source's message.
    pub message: String,
}

impl fmt::Display for FrameSourceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for FrameSourceError {}

/// Makes sure the `.sift` files of the given frames exist and are current.
///
/// [`super::pick_images`] calls it with every frame it is about to read and
/// with no other, each frame once, and then reads every `.sift` file of each
/// frame, so a frame gets a `.sift` file exactly when the walk reads it. It
/// asks for several frames in one call where it knows them at once (the
/// baseline's batches, a blurred run's frames), so a source can extract them
/// as one batch.
pub trait FrameSource {
    /// Make the `.sift` file of every image of every frame in `frames` exist,
    /// extracting those that are missing or older than their image.
    ///
    /// # Errors
    ///
    /// The source's own failure, which [`super::pick_images`] returns as
    /// [`PickImagesError::Source`].
    fn ensure_sift_files(
        &mut self,
        frames: &[FrameId],
        progress: &Progress,
    ) -> Result<(), FrameSourceError>;
}

/// How [`super::pick_images`] chooses frames.
#[derive(Clone, Debug)]
pub struct PickImagesOptions {
    /// The step to aim for between two chosen frames: the median feature
    /// displacement, in percent of the image's long side. Positive.
    pub target_step_percent: f64,
    /// How far a step may be from the target and still be accepted: within
    /// `target × (1 ± step_band)`. In `(0, 1)`.
    pub step_band: f64,
    /// A frame whose sharpness is below `blur_ratio` times its sequence's
    /// median sharpness counts as blurred. In `(0, 1)`.
    pub blur_ratio: f64,
    /// How many frames the baseline samples, spread evenly over the capture
    /// and shared among the sequences by their frame counts. At least 2.
    pub baseline_frames: usize,
    /// A step whose matches fit a fundamental matrix with a smaller share of
    /// inliers does not match well. In `[0, 1]`.
    pub min_inlier_share: f64,
    /// The forest each frame's descriptors are indexed with.
    pub forest: KdForestParams,
    /// How each pair of frames is matched and fitted. Its `min_matches` is
    /// also the fewest matches a step that matches well has.
    pub pair: FramePairOptions,
}

impl Default for PickImagesOptions {
    fn default() -> Self {
        Self {
            target_step_percent: 10.0,
            step_band: 0.3,
            blur_ratio: 0.6,
            baseline_frames: 200,
            min_inlier_share: 0.5,
            forest: KdForestParams::balanced(),
            pair: FramePairOptions::default(),
        }
    }
}

impl PickImagesOptions {
    /// Every option's range, with the forest's and the pair options' checks
    /// made once here rather than at the first frame or pair.
    pub(super) fn validate(&self) -> Result<(), PickImagesError> {
        let options = |name: &str, reason: String| PickImagesError::Options {
            name: name.to_string(),
            reason,
        };
        if !(self.target_step_percent.is_finite() && self.target_step_percent > 0.0) {
            return Err(options(
                "target_step_percent",
                format!(
                    "must be positive and finite, got {}",
                    self.target_step_percent
                ),
            ));
        }
        for (name, value) in [
            ("step_band", self.step_band),
            ("blur_ratio", self.blur_ratio),
        ] {
            if !(value > 0.0 && value < 1.0) {
                return Err(options(name, format!("must be in (0, 1), got {value}")));
            }
        }
        if self.baseline_frames < 2 {
            return Err(options(
                "baseline_frames",
                format!(
                    "must be at least 2, the first and last frames, got {}",
                    self.baseline_frames
                ),
            ));
        }
        if !(0.0..=1.0).contains(&self.min_inlier_share) {
            return Err(options(
                "min_inlier_share",
                format!("must be in [0, 1], got {}", self.min_inlier_share),
            ));
        }
        let nested = |prefix: &str, e: FramePairError| match e {
            FramePairError::Options { name, reason } => PickImagesError::Options {
                name: if name.starts_with("forest.") {
                    name.to_string()
                } else {
                    format!("{prefix}{name}")
                },
                reason,
            },
            other => PickImagesError::Options {
                name: prefix.trim_end_matches('.').to_string(),
                reason: other.to_string(),
            },
        };
        super::frame_pair::validate_forest(&self.forest).map_err(|e| nested("forest.", e))?;
        self.pair.validate().map_err(|e| nested("pair.", e))
    }
}

/// Why [`super::pick_images`] stopped without a report.
#[derive(Debug)]
pub enum PickImagesError {
    /// An option out of range. `name` is the field, `forest.` or `pair.` and
    /// the field for the nested options.
    Options { name: String, reason: String },
    /// No sequences, an empty sequence, a frame with no images or with a
    /// different number of images than its sequence's first frame, a pattern
    /// count that differs from the sensor count, frame numbers not in
    /// increasing order, or two images with one `.sift` path.
    Sequence { reason: String },
    /// A folder the capture's `.sift` files go in holds version 1 `.sift`
    /// files, found before anything is extracted.
    Version1Files(Version1SiftFiles),
    /// A folder the capture's `.sift` files go in is there but cannot be
    /// listed, such as a file in its place.
    ListFolder {
        folder: PathBuf,
        source: std::io::Error,
    },
    /// The [`FrameSource`] failed.
    Source { message: String },
    /// A `.sift` file the source made exist cannot be read, or holds readings
    /// or features that break the format's rules.
    ReadSift { path: PathBuf, source: SiftError },
    /// A `.sift` file read is version 1, which records no readings: one put
    /// in its folder after the up-front check.
    Version1Sift { path: PathBuf },
    /// Two frames could not be matched: their reference sensor's images
    /// differ in size.
    Pair {
        a: FrameId,
        b: FrameId,
        /// The reference sensor's image of each frame.
        images: Box<[PathBuf; 2]>,
        error: FramePairError,
    },
    /// The progress's cancel flag was set.
    Cancelled,
}

impl fmt::Display for PickImagesError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Options { name, reason } => write!(f, "{name} {reason}"),
            Self::Sequence { reason } => f.write_str(reason),
            Self::Version1Files(found) => write!(f, "{found}"),
            Self::ListFolder { folder, source } => write!(
                f,
                "the folder {} that .sift files go in cannot be listed: {source}",
                folder.display()
            ),
            Self::Source { message } => f.write_str(message),
            Self::ReadSift { path, source } => {
                write!(f, "cannot read {}: {source}", path.display())
            }
            Self::Version1Sift { path } => write!(
                f,
                "{} is a version 1 .sift file, which records no sharpness or gradient \
                 structure tensor; remove it and run again, which extracts a version 2 file \
                 in its place",
                path.display()
            ),
            Self::Pair { images, error, .. } => write!(
                f,
                "cannot match {} with {}: {error}",
                images[0].display(),
                images[1].display()
            ),
            Self::Cancelled => f.write_str("picking images was cancelled"),
        }
    }
}

impl std::error::Error for PickImagesError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::ListFolder { source, .. } => Some(source),
            Self::ReadSift { source, .. } => Some(source),
            Self::Pair { error, .. } => Some(error),
            _ => None,
        }
    }
}

impl From<crate::progress::Cancelled> for PickImagesError {
    fn from(_: crate::progress::Cancelled) -> Self {
        Self::Cancelled
    }
}

/// Check the sequences' shape: the [`PickImagesError::Sequence`] cases.
pub(super) fn validate_sequences(sequences: &[CaptureSequence]) -> Result<(), PickImagesError> {
    let refuse = |reason: String| Err(PickImagesError::Sequence { reason });
    if sequences.is_empty() {
        return refuse("there are no sequences to pick images from".to_string());
    }
    let mut sift_paths = std::collections::HashSet::new();
    for (s, sequence) in sequences.iter().enumerate() {
        let name = match sequence.patterns.first() {
            Some(pattern) => format!("sequence {s} ({pattern})"),
            None => format!("sequence {s}"),
        };
        let Some(first) = sequence.frames.first() else {
            return refuse(format!("{name} has no frames"));
        };
        let sensors = first.images.len();
        if sensors == 0 {
            return refuse(format!("{name}: frame {} has no images", first.number));
        }
        if sequence.patterns.len() != sensors {
            return refuse(format!(
                "{name} has {} patterns but {sensors} images per frame",
                sequence.patterns.len()
            ));
        }
        let mut previous: Option<i64> = None;
        for frame in &sequence.frames {
            if frame.images.len() != sensors {
                return refuse(format!(
                    "{name}: frame {} has {} images but the sequence's first frame has \
                     {sensors}",
                    frame.number,
                    frame.images.len()
                ));
            }
            if previous.is_some_and(|p| frame.number <= p) {
                return refuse(format!(
                    "{name}: frame {} follows frame {}, but frames must be in increasing \
                     file-number order",
                    frame.number,
                    previous.unwrap_or_default()
                ));
            }
            previous = Some(frame.number);
            for image in &frame.images {
                if !sift_paths.insert(&image.sift_path) {
                    return refuse(format!(
                        "{} is the .sift path of two images",
                        image.sift_path.display()
                    ));
                }
            }
        }
    }
    Ok(())
}
