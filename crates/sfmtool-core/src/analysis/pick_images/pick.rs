// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! [`pick_images`]: the checks of the input, the walk of each sequence, and
//! the report.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use super::capture::{
    validate_sequences, CaptureSequence, FrameId, PickImagesError, PickImagesOptions,
};
use super::frames::Frames;
use super::report::{FrameReading, PickImagesReport, SensorReading, SequenceReport};
use super::source::FrameSource;
use super::walk::{walk_sequence, Walked};
use crate::features::sift::version_1_sift_files;
use crate::progress::Progress;

/// Choose the frames of a capture a solve should use.
///
/// Walks each sequence from its first frame, choosing frames about
/// `target_step_percent` of the image's long side apart, with no step beyond
/// the next frame keeping fewer than `min_step_matches` mutual matches, and
/// a frame blurred against its neighbouring frames swapped for a sharper one.
///
/// Every frame is read through `source`: [`pick_images`] waits on it for each
/// frame just before reading the frame and, unless `options.prefetch` is
/// off, asks it to prefetch the frames it predicts it will read: the next
/// step's first candidate and its neighbours, the floor search's next tries
/// and, where it reads every frame, the frames ahead. It decides only from
/// frames it waited for, so the report is the same whatever the source does
/// ahead of it, except for
/// [`PickImagesReport::frames_extracted_unread`]. When the run ends, with a
/// report or an error, the source's work is stopped with
/// [`FrameSource::finish`].
///
/// Before anything is extracted, every folder the capture's `.sift` files go
/// in is checked for version 1 files with [`version_1_sift_files`].
///
/// # Errors
///
/// [`PickImagesError::Options`] and [`PickImagesError::Sequence`] for
/// input out of range, [`PickImagesError::Version1Files`] and
/// [`PickImagesError::ListFolder`] from the folder check,
/// [`PickImagesError::Source`] when the source cannot make a frame read
/// exist, naming the frame, [`PickImagesError::ReadSift`] and
/// [`PickImagesError::Version1Sift`] for a `.sift` file that cannot be used,
/// [`PickImagesError::Pair`] for two frames whose reference images differ in
/// size, and [`PickImagesError::Cancelled`] when `progress` is cancelled,
/// which is checked before every wait on the source, while it waits, and
/// before every pair matched.
pub fn pick_images(
    sequences: &[CaptureSequence],
    source: &mut dyn FrameSource,
    options: &PickImagesOptions,
    progress: &Progress,
) -> Result<PickImagesReport, PickImagesError> {
    options.validate()?;
    validate_sequences(sequences)?;
    check_folders(sequences)?;
    progress.check_cancel()?;

    let mut frames = Frames::new(sequences, source, options);
    let walked = walk_all(&mut frames, sequences, options, progress);
    // Stopped on every path, so no worker goes on extracting after the run.
    let written = frames.finish_source();
    Ok(assemble(walked?, frames, &written))
}

/// The walk of each sequence, one after another.
fn walk_all(
    frames: &mut Frames,
    sequences: &[CaptureSequence],
    options: &PickImagesOptions,
    progress: &Progress,
) -> Result<Vec<Walked>, PickImagesError> {
    let phase = progress.phase("walk");
    sequences
        .iter()
        .enumerate()
        .map(|(s, sequence)| walk_sequence(frames, s, sequence.frames.len(), options, &phase))
        .collect()
}

/// Refuse a capture whose `.sift` folders hold version 1 files.
fn check_folders(sequences: &[CaptureSequence]) -> Result<(), PickImagesError> {
    let folders: BTreeSet<PathBuf> = sequences
        .iter()
        .flat_map(|s| &s.frames)
        .flat_map(|f| &f.images)
        .map(|image| match image.sift_path.parent() {
            Some(parent) if !parent.as_os_str().is_empty() => parent.to_path_buf(),
            _ => Path::new(".").to_path_buf(),
        })
        .collect();
    for folder in folders {
        match version_1_sift_files(&folder) {
            Ok(None) => {}
            Ok(Some(found)) => return Err(PickImagesError::Version1Files(found)),
            Err(source) => return Err(PickImagesError::ListFolder { folder, source }),
        }
    }
    Ok(())
}

/// The report from what was read and walked, and the frames the source
/// wrote `.sift` files for.
fn assemble(walked: Vec<Walked>, frames: Frames, written: &[FrameId]) -> PickImagesReport {
    let readings: Vec<FrameReading> = frames
        .read_frames()
        .iter()
        .map(|f| FrameReading {
            id: f.id,
            sensors: f
                .readings
                .iter()
                .map(|r| SensorReading {
                    sharpness: r.sharpness,
                    gradient_structure_tensor_xx_yy_xy: r.gradient_structure_tensor_xx_yy_xy,
                    axis_ratio: r.axis_ratio(),
                    coherence: r.coherence(),
                })
                .collect(),
        })
        .collect();
    let sequences = walked
        .into_iter()
        .enumerate()
        .map(|(s, walk)| SequenceReport {
            chosen: walk.chosen,
            steps: walk.steps,
            swaps: walk.swaps,
            frames_read: readings.iter().filter(|f| f.id.sequence == s).count(),
        })
        .collect();
    let written: BTreeSet<FrameId> = written.iter().copied().collect();
    let frames_extracted: Vec<bool> = readings.iter().map(|f| written.contains(&f.id)).collect();
    let extracted_read = frames_extracted.iter().filter(|&&w| w).count();
    PickImagesReport {
        sequences,
        frames: readings,
        pairs: frames.into_pairs(),
        frames_extracted,
        frames_extracted_unread: written.len() - extracted_read,
    }
}
