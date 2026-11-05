// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! [`pick_images`]: the checks of the input, the baseline in its two batches,
//! the verdict, and the walk of each sequence.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use super::baseline::{
    rate_p90, sample_counts, sample_positions, sequence_baseline, solvability_check, video_check,
    SequenceBaseline, FAR_GAPS, NEAR_GAPS,
};
use super::capture::{
    validate_sequences, CaptureSequence, FrameId, FrameSource, PickImagesError, PickImagesOptions,
};
use super::frames::Frames;
use super::report::{
    BaselineReport, FrameReading, PickImagesReport, SequenceReport, SolvabilityCheck, Stage,
    Verdict, VideoCheck, WalkReport,
};
use super::walk::walk_sequence;
use crate::features::sift::version_1_sift_files;
use crate::progress::Progress;

/// Choose the frames of a capture a solve should use.
///
/// Reads a baseline sample of each sequence, judges from it whether each
/// sequence is a video and whether the capture can be solved, and, when it
/// can, walks each sequence choosing frames about `target_step_percent` of
/// the image's long side apart, sharp ones in place of blurred ones. Every
/// frame is read through `source`, which is asked to make a frame's `.sift`
/// files exist just before the frame is read and for no frame that is not
/// read. A verdict that stops the capture is a report, not an error.
///
/// Before anything is extracted, every folder the capture's `.sift` files go
/// in is checked for version 1 files with [`version_1_sift_files`].
///
/// # Errors
///
/// [`PickImagesError::Options`] and [`PickImagesError::Sequence`] for
/// input out of range, [`PickImagesError::Version1Files`] and
/// [`PickImagesError::ListFolder`] from the folder check,
/// [`PickImagesError::Source`] when the source fails,
/// [`PickImagesError::ReadSift`] and [`PickImagesError::Version1Sift`] for a
/// `.sift` file that cannot be used, [`PickImagesError::Pair`] for two frames
/// whose reference images differ in size, and [`PickImagesError::Cancelled`]
/// when `progress` is cancelled, which is checked before every call to the
/// source and every pair matched.
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

    let lengths: Vec<usize> = sequences.iter().map(|s| s.frames.len()).collect();
    let samples: Vec<Vec<usize>> = sample_counts(&lengths, options.baseline_frames)
        .into_iter()
        .zip(&lengths)
        .map(|(count, &len)| sample_positions(len, count))
        .collect();
    let mut frames = Frames::new(sequences, source, options);

    // The first batch: the sampled frames and the frames 1 and 3 after them.
    let (near_frames, near_pairs) = batch(&samples, &lengths, &NEAR_GAPS, true);
    {
        let phase = progress.phase("baseline");
        frames.ensure(&near_frames, Stage::BaselineNear, &phase)?;
        frames.match_pairs(&near_pairs, Stage::BaselineNear, &phase)?;
    }
    let baselines: Vec<SequenceBaseline> = samples
        .into_iter()
        .enumerate()
        .map(|(s, positions)| sequence_baseline(&frames, s, positions, lengths[s], options))
        .collect();
    let videos: Vec<VideoCheck> = baselines
        .iter()
        .enumerate()
        .map(|(s, b)| video_check(&frames, s, b, options))
        .collect();
    if videos.iter().any(|v| !v.is_video) {
        return Ok(assemble(
            Verdict::NotAVideo,
            frames,
            baselines,
            videos,
            None,
            None,
            options,
        ));
    }

    // The second batch: the frames 10 and 30 after the sampled frames.
    let sampled: Vec<Vec<usize>> = baselines.iter().map(|b| b.samples.clone()).collect();
    let (far_frames, far_pairs) = batch(&sampled, &lengths, &FAR_GAPS, false);
    {
        let phase = progress.phase("baseline far pairs");
        frames.ensure(&far_frames, Stage::BaselineFar, &phase)?;
        frames.match_pairs(&far_pairs, Stage::BaselineFar, &phase)?;
    }
    let checks: Vec<SolvabilityCheck> = baselines
        .iter()
        .enumerate()
        .map(|(s, b)| solvability_check(&frames, s, b))
        .collect();
    if checks.iter().any(|c| !c.solvable) {
        return Ok(assemble(
            Verdict::NotSolvable,
            frames,
            baselines,
            videos,
            Some(checks),
            None,
            options,
        ));
    }

    let mut walks = Vec::with_capacity(sequences.len());
    {
        let phase = progress.phase("walk");
        for (s, baseline) in baselines.iter().enumerate() {
            walks.push(walk_sequence(
                &mut frames,
                baseline,
                s,
                lengths[s],
                options,
                &phase,
            )?);
        }
    }
    Ok(assemble(
        Verdict::Solvable,
        frames,
        baselines,
        videos,
        Some(checks),
        Some(walks),
        options,
    ))
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

/// One batch of the baseline: the frames `gaps` after each sampled frame
/// that exist, with the sampled frames themselves when `with_samples`, and
/// the pairs from each sampled frame to them, in sample order.
fn batch(
    samples: &[Vec<usize>],
    lengths: &[usize],
    gaps: &[usize],
    with_samples: bool,
) -> (Vec<FrameId>, Vec<(FrameId, FrameId)>) {
    let mut frames = Vec::new();
    let mut pairs = Vec::new();
    for (sequence, positions) in samples.iter().enumerate() {
        let id = |frame| FrameId { sequence, frame };
        for &s in positions {
            if with_samples {
                frames.push(id(s));
            }
            for &gap in gaps {
                if s + gap < lengths[sequence] {
                    frames.push(id(s + gap));
                    pairs.push((id(s), id(s + gap)));
                }
            }
        }
    }
    (frames, pairs)
}

/// The report from what was read and decided.
fn assemble(
    verdict: Verdict,
    frames: Frames,
    baselines: Vec<SequenceBaseline>,
    videos: Vec<VideoCheck>,
    checks: Option<Vec<SolvabilityCheck>>,
    walks: Option<Vec<WalkReport>>,
    options: &PickImagesOptions,
) -> PickImagesReport {
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
    let mut checks = checks.map(Vec::into_iter);
    let mut walks = walks.map(Vec::into_iter);
    let sequences = baselines
        .into_iter()
        .zip(videos)
        .enumerate()
        .map(|(s, (baseline, video))| {
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
                    sample_rates: baseline.sample_rates,
                    rate_median: baseline.rate_median,
                    total_motion_percent: baseline.rate_median * baseline.len as f64,
                    flagged_regions: baseline.flagged_regions,
                },
                video,
                solvability: checks.as_mut().and_then(Iterator::next),
                walk: walks.as_mut().and_then(Iterator::next),
            }
        })
        .collect();
    PickImagesReport {
        verdict,
        sequences,
        frames: readings,
        pairs: frames.into_pairs(),
    }
}
