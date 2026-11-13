// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The frames [`super::pick_images`] has read: it waits on the source for
//! each frame once, reads every sensor's readings from the `.sift` metadata,
//! and keeps each pair's statistics so no pair is matched twice. A frame's
//! features are read again from its `.sift` file whenever a pair needs them,
//! and dropped when the pairs that need them are matched, so at most a few
//! forests are held at once. The frames [`super::pick_images`] predicts it
//! will read are passed to the source's prefetch, which only starts work:
//! nothing here reads a frame it did not wait for. While the walk asks for
//! them, the matched positions of the pairs it may still accept are kept
//! too, so that an accepted step's essential split is measured from the
//! walk's own match of the pair. Each of these steps adds its wall time to
//! the [`Clock`] the report's times come from.

use std::cell::Cell;
use std::collections::HashMap;
use std::time::{Duration, Instant};

use sfmtool_sift_format::{
    read_sift_metadata, read_sift_positions_descriptors, SiftError, SiftMetadata, DESCRIPTOR_DIM,
    SIFT_READINGS_VERSION,
};

use super::capture::{CaptureFrame, CaptureSequence, FrameId, PickImagesError, PickImagesOptions};
use super::essential_split::{EssentialSplitOptions, EssentialSplitOutcome};
use super::frame_pair::{
    match_frame_pair_positions, FrameFeatures, FramePairError, FramePairStats, MatchedPositions,
};
use super::report::MeasuredPair;
use super::source::{FrameSource, SourceWork};
use crate::features::sift::ImageReadings;
use crate::progress::Progress;

/// One frame read.
pub(super) struct ReadFrame {
    pub(super) id: FrameId,
    /// One per sensor, the reference sensor's first.
    pub(super) readings: Vec<ImageReadings>,
    /// The reference sensor's metadata, for building its features.
    metadata: SiftMetadata,
}

/// The wall time the walk's thread spent in each kind of work, summed over
/// the calls. Each call is timed once, by the step that makes it, and no
/// timed step calls another, so the times never count one moment twice.
#[derive(Default)]
pub(super) struct Clock {
    /// Blocked in the source's `wait`.
    pub(super) waiting_for_frames: Cell<Duration>,
    /// Reading each frame's `.sift` metadata and readings.
    pub(super) reading_sift_metadata: Cell<Duration>,
    /// Reading features from `.sift` files, for a pair or for the pilot vote.
    pub(super) reading_sift_features: Cell<Duration>,
    /// Building each frame's KD-forest.
    pub(super) building_forests: Cell<Duration>,
    /// Matching pairs of frames.
    pub(super) matching: Cell<Duration>,
    /// Measuring the essential splits of accepted steps.
    pub(super) essential_split: Cell<Duration>,
    /// Clustering the pilot frames' features and running the focal vote.
    pub(super) pilot_vote: Cell<Duration>,
}

/// Run `f`, adding its wall time to `total`.
pub(super) fn timed<T>(total: &Cell<Duration>, f: impl FnOnce() -> T) -> T {
    let start = Instant::now();
    let value = f();
    total.set(total.get() + start.elapsed());
    value
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
    /// Whether the matched positions of each pair matched are kept.
    keep_positions: bool,
    /// The matched positions kept, by pair.
    positions: HashMap<(FrameId, FrameId), MatchedPositions>,
    /// The wall time spent in each kind of work.
    pub(super) clock: Clock,
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
            keep_positions: false,
            positions: HashMap::new(),
            clock: Clock::default(),
        }
    }

    /// Whether `id` has been read.
    pub(super) fn is_read(&self, id: FrameId) -> bool {
        self.index[id.sequence][id.frame].is_some()
    }

    /// The frames of `ids` not yet read, each once, in order.
    fn unread(&self, ids: &[FrameId]) -> Vec<FrameId> {
        let mut new: Vec<FrameId> = Vec::new();
        for &id in ids {
            if !self.is_read(id) && !new.contains(&id) {
                new.push(id);
            }
        }
        new
    }

    fn requests(&self, ids: &[FrameId]) -> Vec<(FrameId, &'a CaptureFrame)> {
        let sequences: &'a [CaptureSequence] = self.sequences;
        ids.iter()
            .map(|&id| (id, &sequences[id.sequence].frames[id.frame]))
            .collect()
    }

    /// How many frames the source works on at once when the walk prefetches,
    /// and 0 when the options turn prefetching off or the source does no
    /// work ahead.
    pub(super) fn prefetch_window(&self) -> usize {
        if self.options.prefetch {
            self.source.concurrency()
        } else {
            0
        }
    }

    /// Ask the source to start on the frames of `ids` not yet read, in
    /// order, in place of the last frames asked for. Nothing is read.
    pub(super) fn prefetch(&mut self, ids: &[FrameId]) {
        if self.prefetch_window() == 0 {
            return;
        }
        let new = self.unread(ids);
        let requests = self.requests(&new);
        self.source.prefetch(&requests);
    }

    /// Wait on the source for the frames of `ids` not yet read, in one call,
    /// and read their readings. A frame already read is not waited for
    /// again.
    pub(super) fn ensure(
        &mut self,
        ids: &[FrameId],
        progress: &Progress,
    ) -> Result<(), PickImagesError> {
        let new = self.unread(ids);
        if new.is_empty() {
            return Ok(());
        }
        progress.check_cancel()?;
        let requests = self.requests(&new);
        timed(&self.clock.waiting_for_frames, || {
            self.source.wait(&requests, progress)
        })?;
        let start = Instant::now();
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
                readings,
                metadata,
            });
        }
        let reading = &self.clock.reading_sift_metadata;
        reading.set(reading.get() + start.elapsed());
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

    /// How many frames of sequence `sequence` have been read.
    pub(super) fn read_count(&self, sequence: usize) -> usize {
        self.index[sequence].iter().flatten().count()
    }

    /// The width and height of a frame read's reference image.
    pub(super) fn image_size(&self, id: FrameId) -> (u32, u32) {
        let i = self.index[id.sequence][id.frame].expect("the frame was read");
        let metadata = &self.read[i].metadata;
        (metadata.image_width, metadata.image_height)
    }

    /// The positions and descriptors of the first `count` features of a
    /// frame's reference image, or all of them when it has fewer.
    pub(super) fn first_features(
        &self,
        id: FrameId,
        count: usize,
    ) -> Result<(Vec<[f32; 2]>, Vec<u8>), PickImagesError> {
        let path = &self.sequences[id.sequence].frames[id.frame].images[0].sift_path;
        let (mut positions_xy, mut descriptors) = timed(&self.clock.reading_sift_features, || {
            read_sift_positions_descriptors(path)
        })
        .map_err(|source| PickImagesError::ReadSift {
            path: path.clone(),
            source,
        })?;
        let n = count.min(positions_xy.len());
        if descriptors.len() < n * DESCRIPTOR_DIM {
            return Err(PickImagesError::ReadSift {
                path: path.clone(),
                source: SiftError::InvalidFormat(format!(
                    "{} descriptor bytes for {n} features",
                    descriptors.len()
                )),
            });
        }
        positions_xy.truncate(n);
        descriptors.truncate(n * DESCRIPTOR_DIM);
        Ok((positions_xy, descriptors))
    }

    /// The features of a frame read: its reference sensor's, with the forest
    /// built.
    pub(super) fn features(&self, id: FrameId) -> Result<FrameFeatures, PickImagesError> {
        let i = self.index[id.sequence][id.frame].expect("the frame was read");
        let path = &self.sequences[id.sequence].frames[id.frame].images[0].sift_path;
        let features = timed(&self.clock.reading_sift_features, || {
            read_sift_positions_descriptors(path)
        })
        .map_err(|source| PickImagesError::ReadSift {
            path: path.clone(),
            source,
        })?;
        timed(&self.clock.building_forests, || {
            FrameFeatures::from_sift(features, &self.read[i].metadata, self.options.forest)
        })
        .map_err(|e| PickImagesError::ReadSift {
            path: path.clone(),
            source: SiftError::InvalidFormat(e.to_string()),
        })
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
        progress: &Progress,
    ) -> Result<FramePairStats, PickImagesError> {
        if let Some(stats) = self.pair(a, b) {
            return Ok(stats.clone());
        }
        progress.check_cancel()?;
        let (stats, positions) = timed(&self.clock.matching, || {
            match_frame_pair_positions(fa, fb, &self.options.pair)
        })
        .map_err(|error| self.pair_error(a, b, error))?;
        if self.keep_positions {
            self.positions.insert((a, b), positions);
        }
        self.pair_index.insert((a, b), self.pairs.len());
        self.pairs.push(MeasuredPair {
            a,
            b,
            stats: stats.clone(),
        });
        progress.value("pairs_matched", self.pairs.len() as f64);
        Ok(stats)
    }

    /// Keep the matched positions of the pairs matched from now on, or, with
    /// `keep` false, keep none and drop those kept.
    pub(super) fn keep_positions(&mut self, keep: bool) {
        self.keep_positions = keep;
        if !keep {
            self.positions.clear();
        }
    }

    /// Drop the matched positions kept of every pair whose first frame is
    /// before `from` in its sequence: the pairs a walk at `from` no longer
    /// measures.
    pub(super) fn drop_positions_before(&mut self, from: FrameId) {
        self.positions
            .retain(|(a, _), _| a.sequence != from.sequence || a.frame >= from.frame);
    }

    /// The essential split of `a` with `b`, measured with `split` from the
    /// matched positions kept of that pair, the ones its statistics were
    /// measured from. The pair has been matched while positions were kept,
    /// and its positions not dropped since. Checks for cancellation first.
    pub(super) fn measure_split(
        &self,
        a: FrameId,
        b: FrameId,
        split: &EssentialSplitOptions,
        progress: &Progress,
    ) -> Result<EssentialSplitOutcome, PickImagesError> {
        progress.check_cancel()?;
        let positions = self
            .positions
            .get(&(a, b))
            .expect("the walk keeps the positions of every pair it may accept");
        timed(&self.clock.essential_split, || {
            positions.essential_split(split)
        })
        .map_err(|error| self.pair_error(a, b, error))
    }

    /// The error of a pair that could not be matched, naming the reference
    /// sensor's image of each frame.
    fn pair_error(&self, a: FrameId, b: FrameId, error: FramePairError) -> PickImagesError {
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
    }

    /// Stop the source's work and return what it did: every frame it wrote
    /// `.sift` files for, sorted, and its workers' time.
    pub(super) fn finish_source(&mut self) -> SourceWork {
        self.source.finish()
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
