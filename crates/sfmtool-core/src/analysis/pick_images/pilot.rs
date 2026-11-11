// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The pilot vote: the camera of a sequence that has none, from a focal vote
//! on the first frames the walk chooses without the local density term.
//! See `specs/core/analysis/pick-images.md` § "The pilot vote".

use std::time::Instant;

use ndarray::ArrayView2;
use serde::Serialize;
use sfmtool_sift_format::DESCRIPTOR_DIM;

use super::capture::{FrameId, PickImagesError, PickImagesOptions};
use super::frames::Frames;
use super::walk::walk_sequence;
use crate::camera::{CameraIntrinsics, CameraModel as LensModel};
use crate::features::cluster_match::{background_floor_clusters, BackgroundFloorParams};
use crate::geometry::estimate_intrinsics::{
    estimate_intrinsics, ColumnPolicy, IntrinsicsEstimate, IntrinsicsOptions,
};
use crate::geometry::focal_vote::{CameraModel, FocalVoteOptions};
use crate::progress::Progress;

/// How many frames the pilot walk chooses before the vote.
pub(super) const PILOT_FRAMES: usize = 30;

/// How many features of each pilot frame, its first, are clustered.
pub(super) const PILOT_MAX_FEATURES: usize = 1500;

/// The fewest pilot frames a vote is run on. The vote's consensus needs two
/// pooled votes and each image pair casts at most one, so two frames, one
/// pair, never reach one.
pub(super) const MIN_PILOT_FRAMES: usize = 3;

/// The largest pool spread, the log interquartile range of the votes behind
/// the consensus, at which the pilot's camera is used.
pub(super) const MAX_POOL_SPREAD: f64 = 0.35;

/// What the pilot vote of a sequence found, and whether the walk measured
/// with its camera.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PilotVote {
    /// The positions of the frames voted on: the first 30 frames the walk
    /// chooses without the local density term, or all it chooses in a
    /// shorter sequence.
    pub frames: Vec<usize>,
    /// How many frames the pilot walk read.
    pub frames_read: usize,
    /// How many features were clustered, at most 1,500 per frame; 0 when no
    /// vote was run.
    pub features: usize,
    /// How many track clusters they formed.
    pub clusters: usize,
    /// The vote's camera-model verdict, `"Pinhole"` or
    /// `"EquidistantFisheye"`, `None` without a vote or a verdict.
    pub camera_model: Option<&'static str>,
    /// Whether an `"EquidistantFisheye"` verdict carries certified rotation
    /// votes, `None` for any other verdict. An unconfirmed one is still
    /// measured with as a fisheye.
    pub confirmed: Option<bool>,
    /// The consensus focal at the frames' size, `None` without one.
    pub focal_px: Option<f64>,
    /// The pool spread of the votes behind `focal_px`, `None` without a
    /// consensus.
    pub pool_spread: Option<f64>,
    /// The pooled votes, and how many of them were epipolar and rotation
    /// votes.
    pub n_pool: usize,
    pub n_epipolar: usize,
    pub n_rotation: usize,
    /// Whether the walk measured with the vote's camera, and why not.
    pub outcome: PilotOutcome,
    /// The pilot's wall time in seconds, its walk, matching and vote: the one
    /// figure of the pilot that differs from run to run.
    pub seconds: f64,
}

/// Whether the walk measured with the pilot vote's camera, and why not.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum PilotOutcome {
    /// The walk measured with the vote's camera.
    Used,
    /// The pilot walk chose fewer than `min_frames` frames, and no vote ran.
    TooFewFrames { min_frames: usize },
    /// The vote reached no consensus focal.
    NoConsensus,
    /// The pool spread is above `max_pool_spread`.
    WideSpread { max_pool_spread: f64 },
}

/// The outcome of a vote with consensus focal `focal_px` and pool spread
/// `pool_spread`.
pub(super) fn judge(focal_px: Option<f64>, pool_spread: f64) -> PilotOutcome {
    match focal_px {
        None => PilotOutcome::NoConsensus,
        Some(_) if pool_spread > MAX_POOL_SPREAD => PilotOutcome::WideSpread {
            max_pool_spread: MAX_POOL_SPREAD,
        },
        Some(_) => PilotOutcome::Used,
    }
}

/// The camera of a verdict `model` at `focal_px`, for frames of `width` ×
/// `height`, with its principal point at the image centre, where the vote
/// places it: an equidistant fisheye for a fisheye verdict, confirmed or
/// not, and a pinhole otherwise.
pub(super) fn verdict_camera(
    model: Option<CameraModel>,
    focal_px: f64,
    width: u32,
    height: u32,
) -> CameraIntrinsics {
    let principal_point_x = f64::from(width) / 2.0;
    let principal_point_y = f64::from(height) / 2.0;
    let model = match model {
        Some(CameraModel::EquidistantFisheye) => LensModel::EquidistantFisheye {
            focal_length: focal_px,
            principal_point_x,
            principal_point_y,
        },
        _ => LensModel::SimplePinhole {
            focal_length: focal_px,
            principal_point_x,
            principal_point_y,
        },
    };
    CameraIntrinsics {
        model,
        width,
        height,
        rolling_shutter: None,
    }
}

/// Run the pilot vote of sequence `sequence`, of `len` frames: walk it
/// without the local density term until it has chosen [`PILOT_FRAMES`]
/// frames, cluster the first [`PILOT_MAX_FEATURES`] features of each, and
/// vote with the camera-model columns chosen by the evidence. Returns the
/// pilot's report and the camera the walk measures with, `None` when the
/// outcome is not [`PilotOutcome::Used`].
pub(super) fn pilot_vote(
    frames: &mut Frames,
    sequence: usize,
    len: usize,
    options: &PickImagesOptions,
    progress: &Progress,
) -> Result<(PilotVote, Option<CameraIntrinsics>), PickImagesError> {
    let start = Instant::now();
    let read_before = frames.read_count(sequence);
    let walked = walk_sequence(
        frames,
        sequence,
        len,
        None,
        options,
        progress,
        Some(PILOT_FRAMES),
    )?;
    let mut vote = PilotVote {
        frames: walked.chosen,
        frames_read: frames.read_count(sequence) - read_before,
        features: 0,
        clusters: 0,
        camera_model: None,
        confirmed: None,
        focal_px: None,
        pool_spread: None,
        n_pool: 0,
        n_epipolar: 0,
        n_rotation: 0,
        outcome: PilotOutcome::TooFewFrames {
            min_frames: MIN_PILOT_FRAMES,
        },
        seconds: 0.0,
    };
    let mut camera = None;
    if vote.frames.len() >= MIN_PILOT_FRAMES {
        let ids: Vec<FrameId> = vote
            .frames
            .iter()
            .map(|&frame| FrameId { sequence, frame })
            .collect();
        let (width, height) = frames.image_size(ids[0]);
        let (estimate, features, clusters) = vote_on(frames, &ids, width, height, progress)?;
        let result = &estimate.vote;
        vote.features = features;
        vote.clusters = clusters;
        vote.camera_model = estimate.camera_model.map(CameraModel::as_str);
        vote.confirmed = estimate.confirmed;
        vote.focal_px = estimate.focal_px;
        vote.pool_spread = estimate.focal_px.map(|_| result.pool_spread);
        vote.n_pool = result.n_pool;
        vote.n_epipolar = result.n_epipolar;
        vote.n_rotation = result.n_rotation;
        vote.outcome = judge(estimate.focal_px, result.pool_spread);
        if let (PilotOutcome::Used, Some(focal_px)) = (vote.outcome, estimate.focal_px) {
            camera = Some(verdict_camera(
                estimate.camera_model,
                focal_px,
                width,
                height,
            ));
        }
    }
    vote.seconds = start.elapsed().as_secs_f64();
    Ok((vote, camera))
}

/// Cluster the first [`PILOT_MAX_FEATURES`] features of each frame of `ids`,
/// as `sfm match --cluster --max-features` does, and run the focal vote
/// `sfm estimate-intrinsics` runs on the clusters, the model chosen by the
/// evidence. Returns the estimate, the number of features clustered and the
/// number of clusters.
fn vote_on(
    frames: &Frames,
    ids: &[FrameId],
    width: u32,
    height: u32,
    progress: &Progress,
) -> Result<(IntrinsicsEstimate, usize, usize), PickImagesError> {
    let mut positions: Vec<Vec<[f32; 2]>> = Vec::with_capacity(ids.len());
    let mut corpus: Vec<u8> = Vec::new();
    let mut image_starts: Vec<u32> = vec![0];
    for &id in ids {
        progress.check_cancel()?;
        let (frame_positions, descriptors) = frames.first_features(id, PILOT_MAX_FEATURES)?;
        corpus.extend_from_slice(&descriptors);
        image_starts.push((corpus.len() / DESCRIPTOR_DIM) as u32);
        positions.push(frame_positions);
    }
    progress.check_cancel()?;
    let features = corpus.len() / DESCRIPTOR_DIM;
    let params = BackgroundFloorParams::default();
    // A corpus too small for the matcher's background rank forms no
    // clusters, and the vote on none reaches no consensus.
    let (cluster_starts, member_images, member_features) = if features > params.d {
        let view = ArrayView2::from_shape((features, DESCRIPTOR_DIM), &corpus)
            .expect("the corpus holds whole descriptors");
        let clusters = background_floor_clusters(view, &image_starts, &params)
            .expect("the offsets are built here and the corpus holds more rows than d");
        (
            clusters.cluster_starts.to_vec(),
            clusters.member_images.to_vec(),
            clusters.member_features.to_vec(),
        )
    } else {
        (vec![0], Vec::new(), Vec::new())
    };
    let member_positions: Vec<[f32; 2]> = member_images
        .iter()
        .zip(&member_features)
        .map(|(&image, &feature)| positions[image as usize][feature as usize])
        .collect();
    progress.check_cancel()?;
    let estimate = estimate_intrinsics(
        &cluster_starts,
        &member_images,
        &member_positions,
        width,
        height,
        &IntrinsicsOptions {
            vote: FocalVoteOptions::default(),
            columns: ColumnPolicy::Auto,
            ..IntrinsicsOptions::default()
        },
    );
    Ok((estimate, features, cluster_starts.len() - 1))
}

#[cfg(test)]
mod tests;
