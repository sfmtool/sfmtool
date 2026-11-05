// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The frames [`super::pick_images`] has read: it asks the source for each
//! frame once, reads every sensor's readings from the `.sift` metadata, and
//! keeps each pair's statistics so no pair is matched twice. A frame's
//! features are read again from its `.sift` file whenever a pair needs them,
//! and dropped when the pairs that need them are matched, so at most a few
//! forests are held at once.

use std::collections::hash_map::Entry;
use std::collections::HashMap;

use sfmtool_sift_format::{
    read_sift_features, read_sift_metadata, SiftError, SiftMetadata, SIFT_READINGS_VERSION,
};

use super::capture::{CaptureSequence, FrameId, FrameSource, PickImagesError, PickImagesOptions};
use super::frame_pair::{match_frame_pair, FrameFeatures, FramePairStats};
use super::report::{MeasuredPair, Stage};
use crate::features::sift::ImageReadings;
use crate::progress::Progress;

/// One frame read.
pub(super) struct ReadFrame {
    pub(super) id: FrameId,
    pub(super) stage: Stage,
    /// One per sensor, the reference sensor's first.
    pub(super) readings: Vec<ImageReadings>,
    /// The reference sensor's metadata, for building its features.
    metadata: SiftMetadata,
}

pub(super) struct Frames<'a> {
    sequences: &'a [CaptureSequence],
    source: &'a mut dyn FrameSource,
    options: &'a PickImagesOptions,
    /// Per sequence and position, the index in `read` of a frame read.
    index: Vec<Vec<Option<usize>>>,
    read: Vec<ReadFrame>,
    pair_index: HashMap<(FrameId, FrameId), usize>,
    pairs: Vec<MeasuredPair>,
}

impl<'a> Frames<'a> {
    pub(super) fn new(
        sequences: &'a [CaptureSequence],
        source: &'a mut dyn FrameSource,
        options: &'a PickImagesOptions,
    ) -> Self {
        Self {
            sequences,
            source,
            options,
            index: sequences
                .iter()
                .map(|s| vec![None; s.frames.len()])
                .collect(),
            read: Vec::new(),
            pair_index: HashMap::new(),
            pairs: Vec::new(),
        }
    }

    /// Whether `id` has been read.
    pub(super) fn is_read(&self, id: FrameId) -> bool {
        self.index[id.sequence][id.frame].is_some()
    }

    /// Ask the source for the frames of `ids` not yet read, in one call, and
    /// read their readings. A frame already read is not asked for again.
    pub(super) fn ensure(
        &mut self,
        ids: &[FrameId],
        stage: Stage,
        progress: &Progress,
    ) -> Result<(), PickImagesError> {
        let mut new = Vec::new();
        for &id in ids {
            if !self.is_read(id) && !new.contains(&id) {
                new.push(id);
            }
        }
        if new.is_empty() {
            return Ok(());
        }
        progress.check_cancel()?;
        self.source
            .ensure_sift_files(&new, progress)
            .map_err(|e| PickImagesError::Source { message: e.message })?;
        for id in new {
            let frame = &self.sequences[id.sequence].frames[id.frame];
            let mut readings = Vec::with_capacity(frame.images.len());
            let mut reference = None;
            for image in &frame.images {
                let (metadata, image_readings) = read_readings(&image.sift_path)?;
                readings.push(image_readings);
                reference.get_or_insert(metadata);
            }
            let metadata = reference.expect("a validated frame has at least one image");
            self.index[id.sequence][id.frame] = Some(self.read.len());
            self.read.push(ReadFrame {
                id,
                stage,
                readings,
                metadata,
            });
        }
        progress.count(self.read.len() as u64, None, "frames");
        Ok(())
    }

    /// The readings of a frame read, one per sensor.
    pub(super) fn readings(&self, id: FrameId) -> &[ImageReadings] {
        let i = self.index[id.sequence][id.frame].expect("the frame was read");
        &self.read[i].readings
    }

    /// Every frame read, in the order read.
    pub(super) fn read_frames(&self) -> &[ReadFrame] {
        &self.read
    }

    /// The features of a frame read: its reference sensor's, with the forest
    /// built.
    pub(super) fn features(&self, id: FrameId) -> Result<FrameFeatures, PickImagesError> {
        let i = self.index[id.sequence][id.frame].expect("the frame was read");
        let path = &self.sequences[id.sequence].frames[id.frame].images[0].sift_path;
        let features = read_sift_features(path).map_err(|source| PickImagesError::ReadSift {
            path: path.clone(),
            source,
        })?;
        FrameFeatures::from_sift(features, &self.read[i].metadata, self.options.forest).map_err(
            |e| PickImagesError::ReadSift {
                path: path.clone(),
                source: SiftError::InvalidFormat(e.to_string()),
            },
        )
    }

    /// The statistics of `a` against `b` when that pair has been matched.
    pub(super) fn pair(&self, a: FrameId, b: FrameId) -> Option<&FramePairStats> {
        self.pair_index.get(&(a, b)).map(|&i| &self.pairs[i].stats)
    }

    /// Match `a` with `b`, whose features are `fa` and `fb`, and keep the
    /// statistics. Checks for cancellation first, and reports the count of
    /// pairs matched as the value `pairs_matched` after.
    pub(super) fn match_pair(
        &mut self,
        a: FrameId,
        fa: &FrameFeatures,
        b: FrameId,
        fb: &FrameFeatures,
        stage: Stage,
        progress: &Progress,
    ) -> Result<FramePairStats, PickImagesError> {
        if let Some(stats) = self.pair(a, b) {
            return Ok(stats.clone());
        }
        progress.check_cancel()?;
        let stats = match_frame_pair(fa, fb, &self.options.pair).map_err(|error| {
            let image = |id: FrameId| {
                self.sequences[id.sequence].frames[id.frame].images[0]
                    .image_path
                    .clone()
            };
            PickImagesError::Pair {
                a,
                b,
                images: Box::new([image(a), image(b)]),
                error,
            }
        })?;
        self.pair_index.insert((a, b), self.pairs.len());
        self.pairs.push(MeasuredPair {
            a,
            b,
            stats: stats.clone(),
            stage,
        });
        progress.value("pairs_matched", self.pairs.len() as f64);
        Ok(stats)
    }

    /// Match every pair of `pairs` not yet matched, in order. A frame's
    /// features are built when its first pair comes up and dropped after its
    /// last, so only the frames of the pairs near the current one are held.
    /// Every frame of `pairs` has been read.
    pub(super) fn match_pairs(
        &mut self,
        pairs: &[(FrameId, FrameId)],
        stage: Stage,
        progress: &Progress,
    ) -> Result<(), PickImagesError> {
        let todo: Vec<(FrameId, FrameId)> = pairs
            .iter()
            .copied()
            .filter(|&(a, b)| self.pair(a, b).is_none())
            .collect();
        let mut uses: HashMap<FrameId, usize> = HashMap::new();
        for &(a, b) in &todo {
            *uses.entry(a).or_default() += 1;
            *uses.entry(b).or_default() += 1;
        }
        let mut live: HashMap<FrameId, FrameFeatures> = HashMap::new();
        for (a, b) in todo {
            if self.pair(a, b).is_some() {
                continue;
            }
            for id in [a, b] {
                if let Entry::Vacant(slot) = live.entry(id) {
                    slot.insert(self.features(id)?);
                }
            }
            self.match_pair(a, &live[&a], b, &live[&b], stage, progress)?;
            for id in [a, b] {
                let left = uses.get_mut(&id).expect("counted above");
                *left -= 1;
                if *left == 0 {
                    live.remove(&id);
                }
            }
        }
        Ok(())
    }

    /// Every pair matched, in order.
    pub(super) fn into_pairs(self) -> Vec<MeasuredPair> {
        self.pairs
    }
}

/// The metadata of a `.sift` file and its readings, refusing a version 1 file
/// and readings that break the format's rules.
fn read_readings(path: &std::path::Path) -> Result<(SiftMetadata, ImageReadings), PickImagesError> {
    let read_error = |source| PickImagesError::ReadSift {
        path: path.to_path_buf(),
        source,
    };
    let (_, metadata, _) = read_sift_metadata(path).map_err(read_error)?;
    if metadata.version < SIFT_READINGS_VERSION {
        return Err(PickImagesError::Version1Sift {
            path: path.to_path_buf(),
        });
    }
    // A version 2 file that reads has both readings: the reader refuses one
    // that lacks either.
    let sharpness = metadata.sharpness.unwrap_or(f64::NAN);
    let [xx, yy, xy] = metadata
        .gradient_structure_tensor_xx_yy_xy
        .unwrap_or([f64::NAN; 3]);
    // The readings are compared within a capture to about three digits, and
    // `ImageReadings` holds them in `f32` for that reason.
    let readings = ImageReadings::new(sharpness as f32, [xx as f32, yy as f32, xy as f32])
        .map_err(|e| read_error(SiftError::InvalidFormat(e.to_string())))?;
    Ok((metadata, readings))
}
