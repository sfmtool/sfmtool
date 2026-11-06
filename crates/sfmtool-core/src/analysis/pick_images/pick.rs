// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! [`pick_images`]: the checks of the input, the baseline, and the walk of
//! each sequence.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use super::baseline::{
    rate_p90, sample_counts, sample_positions, sequence_baseline, SequenceBaseline, BASELINE_GAPS,
};
use super::capture::{
    validate_sequences, CaptureSequence, FrameId, PickImagesError, PickImagesOptions,
};
use super::frames::Frames;
use super::report::{
    BaselineReport, FrameReading, PickImagesReport, SequenceReport, Stage, WalkReport,
};
use super::source::FrameSource;
use super::walk::walk_sequence;
use crate::features::sift::version_1_sift_files;
use crate::progress::Progress;

/// Choose the frames of a capture a solve should use.
///
/// Reads a baseline sample of each sequence, then walks each sequence
/// choosing frames about `target_step_percent` of the image's long side
/// apart, with no step beyond the next frame keeping fewer than
/// `min_step_matches` mutual matches, and sharp frames in place of blurred
/// ones.
///
/// Every frame is read through `source`: [`pick_images`] waits on it for each
/// frame just before reading the frame and, unless `options.prefetch` is
/// off, asks it to prefetch the frames it knows it will read: the baseline,
/// and the frames ahead of the walk through a break, where it reads every
/// frame. It decides only from frames it waited for, so the
/// report is the same whatever the source does ahead of it, except for
/// [`PickImagesReport::frames_extracted_unread`]. When the run
/// ends, with a report or an error, the source's work is stopped with
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
    let decided = decide(&mut frames, sequences, options, progress);
    // Stopped on every path, so no worker goes on extracting after the run.
    let written = frames.finish_source();
    Ok(assemble(decided?, frames, &written, options))
}

/// What [`decide`] found, for [`assemble`].
struct Decided {
    baselines: Vec<SequenceBaseline>,
    walks: Vec<WalkReport>,
}

/// The baseline and the walk of each sequence.
fn decide(
    frames: &mut Frames,
    sequences: &[CaptureSequence],
    options: &PickImagesOptions,
    progress: &Progress,
) -> Result<Decided, PickImagesError> {
    let lengths: Vec<usize> = sequences.iter().map(|s| s.frames.len()).collect();
    let samples: Vec<Vec<usize>> = sample_counts(&lengths, options.baseline_frames)
        .into_iter()
        .zip(&lengths)
        .map(|(count, &len)| sample_positions(len, count))
        .collect();

    // The baseline: the sampled frames and the frames 1 and 3 after them.
    {
        let phase = progress.phase("baseline");
        let groups = baseline_groups(&samples, &lengths, &BASELINE_GAPS);
        read_baseline(frames, &groups, &phase)?;
    }
    let baselines: Vec<SequenceBaseline> = samples
        .into_iter()
        .enumerate()
        .map(|(s, positions)| sequence_baseline(frames, s, positions, lengths[s], options))
        .collect();

    let mut walks = Vec::with_capacity(sequences.len());
    {
        let phase = progress.phase("walk");
        for (s, baseline) in baselines.iter().enumerate() {
            walks.push(walk_sequence(
                frames, baseline, s, lengths[s], options, &phase,
            )?);
        }
    }
    Ok(Decided { baselines, walks })
}

/// Read the baseline: prefetch all its frames, then wait for and match each
/// sampled frame's in turn, so that matching one sample's pairs overlaps the
/// source's work on the next ones. The frames are read, and the pairs
/// matched, in the baseline's order.
fn read_baseline(
    frames: &mut Frames,
    groups: &[SampleGroup],
    progress: &Progress,
) -> Result<(), PickImagesError> {
    let all: Vec<FrameId> = groups
        .iter()
        .flat_map(|g| g.frames.iter().copied())
        .collect();
    frames.prefetch(&all);
    for group in groups {
        frames.ensure(&group.frames, Stage::Baseline, progress)?;
        frames.match_pairs(&group.pairs, Stage::Baseline, progress)?;
    }
    Ok(())
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

/// One sampled frame's part of the baseline: the frames it reads
/// and the pairs it matches.
struct SampleGroup {
    frames: Vec<FrameId>,
    pairs: Vec<(FrameId, FrameId)>,
}

/// The baseline, per sampled frame in sample order: the sampled frame, the
/// frames `gaps` after it that exist, and the pairs from the sampled frame to
/// them.
fn baseline_groups(samples: &[Vec<usize>], lengths: &[usize], gaps: &[usize]) -> Vec<SampleGroup> {
    let mut groups = Vec::new();
    for (sequence, positions) in samples.iter().enumerate() {
        let id = |frame| FrameId { sequence, frame };
        for &s in positions {
            let mut group = SampleGroup {
                frames: vec![id(s)],
                pairs: Vec::new(),
            };
            for &gap in gaps {
                if s + gap < lengths[sequence] {
                    group.frames.push(id(s + gap));
                    group.pairs.push((id(s), id(s + gap)));
                }
            }
            groups.push(group);
        }
    }
    groups
}

/// The report from what was read and decided, and the frames the source
/// wrote `.sift` files for.
fn assemble(
    decided: Decided,
    frames: Frames,
    written: &[FrameId],
    options: &PickImagesOptions,
) -> PickImagesReport {
    let Decided { baselines, walks } = decided;
    let readings: Vec<FrameReading> = frames
        .read_frames()
        .iter()
        .map(|f| {
            let sensors = baselines[f.id.sequence].judge_frame(&f.readings, options.blur_ratio);
            FrameReading {
                id: f.id,
                stage: f.stage,
                blurred: sensors.iter().any(|s| s.blurred),
                sensors,
            }
        })
        .collect();
    let sequences = baselines
        .into_iter()
        .zip(walks)
        .enumerate()
        .map(|(s, (baseline, walk))| {
            let sampled: BTreeSet<usize> = baseline.samples.iter().copied().collect();
            let in_sequence = |f: &&FrameReading| f.id.sequence == s;
            let frames_read = readings
                .iter()
                .filter(in_sequence)
                .filter(|f| f.stage != Stage::Walk)
                .count();
            let blurred_samples = readings
                .iter()
                .filter(in_sequence)
                .filter(|f| sampled.contains(&f.id.frame) && f.blurred)
                .count();
            SequenceReport {
                baseline: BaselineReport {
                    rate_p90: rate_p90(&baseline),
                    samples: baseline.samples,
                    frames_read,
                    sensors: baseline.sensors,
                    blurred_samples,
                    next_frame_matches_median: baseline.next_frame_matches_median,
                    sample_rates: baseline.sample_rates,
                    rate_median: baseline.rate_median,
                    total_motion_percent: baseline.rate_median * baseline.len as f64,
                    flagged_regions: baseline.flagged_regions,
                },
                walk,
            }
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
