// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The commit: the one step that takes an editable track off the bench and into
//! the reconstruction.
//!
//! `specs/core/bench/editable-track.md` is the design. Everything else about the
//! bench leaves the reconstruction exactly as it was; this builds a
//! [`PointRecord`] out of the track's payload and its `in` observations and
//! makes one ordinary point edit with it.

use nalgebra::Vector3;

use crate::reconstruction::data::Point3D;
use crate::reconstruction::edited::{
    EditError, EditedReconstruction, PointMap, PointRecord, RecordObservation,
};

use super::track::{EditableTrack, Provenance, StageKind, TrackPayload};

/// What one commit wrote.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommitReport {
    /// The index the written point took.
    pub point: u32,
    /// The index it replaced, which is now deleted, or `None` when the track
    /// had no origin that resolves and its point was appended.
    pub replaced: Option<u32>,
    /// What the commit did to point indexes: the write, and the absorbed
    /// points' removal chained after it when there was one.
    ///
    /// A caller carrying a selection, a point id or an undo across the commit
    /// reads this rather than reassembling it from the fields above, and it is
    /// the same vocabulary every other edit answers in.
    pub map: PointMap,
    /// How many observations were written, which is how many were `in`.
    pub observation_count: usize,
}

impl CommitReport {
    /// The points the commit deleted because `in` observations had been pulled
    /// from them, ascending. A merge is what this list being non-empty means.
    pub fn absorbed(&self) -> &[u32] {
        match &self.map {
            PointMap::Chain(steps) => match steps.last() {
                Some(PointMap::Removed(absorbed)) => absorbed,
                _ => &[],
            },
            _ => &[],
        }
    }

    /// The sentence the log records, given the name the caller knows the
    /// reconstruction by.
    ///
    /// The node's name is the caller's because core has no scene: a
    /// reconstruction value does not know what it is called in a window or in a
    /// script's output.
    pub fn label(&self, node: &str) -> String {
        let mut text = format!(
            "Committed track: {} observations in {node}",
            self.observation_count
        );
        if let Some(replaced) = self.replaced {
            text.push_str(&format!(", replacing point {replaced}"));
        }
        let absorbed = self.absorbed().len();
        if absorbed > 0 {
            text.push_str(&format!(", absorbing {absorbed} points"));
        }
        text
    }
}

impl std::fmt::Display for CommitReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.label("the reconstruction"))
    }
}

/// Why a track could not be committed. Every variant names what did not hold,
/// because the caller is a button that has to say so in one sentence.
#[derive(Debug, Clone, PartialEq)]
pub enum CommitError {
    /// The track is a cluster. There is no position to store and the format has
    /// no row for a track that is not a point.
    ClusterStage,
    /// The reconstruction's observations are `.sift` feature indexes. An
    /// observation the localizer placed is a keypoint and not a feature index,
    /// so a track is committed to an `embedded_patches` reconstruction only.
    NotEmbeddedPatches,
    /// Fewer than two observations are `in`. One sighting fixes a bearing and
    /// no point, and none fixes nothing.
    TooFewObservations(usize),
    /// The track carries no position, so nothing has triangulated it yet.
    NoPosition,
    /// The reconstruction carries a patch frame per point and the track has
    /// none.
    NoFrame,
    /// The reconstruction carries a patch bitmap per point and the track has
    /// none.
    NoBitmap,
    /// An `in` observation carries no keypoint, so there is no pixel to store
    /// for it.
    NoKeypoint {
        /// Which observation of the track.
        observation: usize,
        /// The image it names.
        image: u32,
    },
    /// An `in` observation names an image the reconstruction does not hold.
    ImageOutOfRange {
        /// The index named.
        image: u32,
        /// How many images the reconstruction holds.
        image_count: usize,
    },
    /// The record the commit built was refused by the overlay.
    Edit(EditError),
}

impl std::fmt::Display for CommitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CommitError::ClusterStage => write!(
                f,
                "the track is at the cluster stage; upgrade it before committing"
            ),
            CommitError::NotEmbeddedPatches => write!(
                f,
                "committing a track needs an embedded_patches reconstruction; \
                 this one's observations are .sift features"
            ),
            CommitError::TooFewObservations(n) => {
                write!(f, "{n} observations are in, and a point needs two or more")
            }
            CommitError::NoPosition => {
                write!(f, "the track has no position; fit it before committing")
            }
            CommitError::NoFrame => write!(
                f,
                "the reconstruction stores a patch frame per point and the track has none"
            ),
            CommitError::NoBitmap => write!(
                f,
                "the reconstruction stores a patch bitmap per point and the track has none"
            ),
            CommitError::NoKeypoint { observation, image } => write!(
                f,
                "observation {observation}, in image {image}, has no keypoint to store"
            ),
            CommitError::ImageOutOfRange { image, image_count } => write!(
                f,
                "image {image} is past the {image_count} images of the reconstruction"
            ),
            CommitError::Edit(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for CommitError {}

impl From<EditError> for CommitError {
    fn from(e: EditError) -> Self {
        CommitError::Edit(e)
    }
}

/// Write `track` into `edited` as one point.
///
/// The record is the track's payload plus its `in` observations' keypoints: the
/// position it carries, the frame it stands on, the consensus bitmap, the
/// colour read from that bitmap's centre, the normal the frame states, and one
/// observation per `in` observation with its keypoint and its leave-one-out
/// ZNCC in `observation_confidence` where the column exists.
///
/// With **no origin that resolves**, the point is appended. With **an origin
/// that resolves** in this value, it takes that point's place, which is what
/// makes putting a point on the bench and committing it a modification rather
/// than a duplication. An origin whose point this value no longer holds is an
/// origin that names nothing, and the commit creates rather than refusing: the
/// person is looking at a track, and the point it came from has been deleted
/// under it.
///
/// **`in` observations pulled from other points absorb them.** An `in`
/// observation whose provenance names a point other than the origin means that
/// point's sighting now belongs to this track, and a reconstruction should not
/// hold two points for one surface, so that point is deleted. A candidate or an
/// `out` observation pulled from a point leaves that point alone.
///
/// Nothing here triangulates. The track commits with the position it carries,
/// and a track that carries none refuses naming the fit as the step that
/// is missing, so the record that is written is one the numbers on screen
/// describe.
///
/// The base behind `edited`'s `Arc` is not written: the returned value shares
/// it.
///
/// # Example
///
/// ```no_run
/// # use sfmtool_core::bench::{commit, EditableTrack};
/// # use sfmtool_core::EditedReconstruction;
/// # fn run(edited: &EditedReconstruction, track: &EditableTrack)
/// # -> Result<(), Box<dyn std::error::Error>> {
/// let (next, report) = commit(edited, track)?;
/// println!("{}", report.label("bull"));
/// let settled = track.with_origin(1, report.point);
/// # let _ = (next, settled);
/// # Ok(())
/// # }
/// ```
pub fn commit(
    edited: &EditedReconstruction,
    track: &EditableTrack,
) -> Result<(EditedReconstruction, CommitReport), CommitError> {
    // The cheap refusals first, so a caller greying a button gets the same
    // answers without building anything.
    let payload: &TrackPayload = match track.stage_kind() {
        StageKind::Cluster => return Err(CommitError::ClusterStage),
        StageKind::Track => track.track().expect("the track stage carries a payload"),
    };
    if edited.has_feature_indexes() {
        return Err(CommitError::NotEmbeddedPatches);
    }
    let kept = track.in_observations();
    if kept.len() < 2 {
        return Err(CommitError::TooFewObservations(kept.len()));
    }
    let position = payload.position.ok_or(CommitError::NoPosition)?;
    if edited.has_patch_frames() && payload.frame.is_none() {
        return Err(CommitError::NoFrame);
    }
    if edited.has_patch_bitmaps() && payload.bitmap.is_none() {
        return Err(CommitError::NoBitmap);
    }
    let image_count = edited.image_count();

    // ---- The track, in image order ----
    //
    // A stored track is in image order and every reader of one relies on that,
    // so the observations are sorted here rather than left in the order the
    // person happened to add them to the bench.
    let mut rows: Vec<(u32, usize)> = kept
        .iter()
        .map(|&i| (track.observations[i].image, i))
        .collect();
    rows.sort_by_key(|&(image, i)| (image, i));
    let mut observations = Vec::with_capacity(rows.len());
    for &(image, i) in &rows {
        if image as usize >= image_count {
            return Err(CommitError::ImageOutOfRange { image, image_count });
        }
        let measurement = track.observations[i]
            .track
            .as_ref()
            .ok_or(CommitError::NoKeypoint {
                observation: i,
                image,
            })?;
        let keypoint = measurement.keypoint.ok_or(CommitError::NoKeypoint {
            observation: i,
            image,
        })?;
        observations.push(RecordObservation {
            image_index: image,
            feature_index: None,
            keypoint_xy: edited.has_keypoints().then_some(keypoint),
            confidence: edited.has_observation_confidence().then(|| {
                let zncc = measurement.zncc.unwrap_or(0.0);
                if zncc.is_nan() {
                    0
                } else {
                    (zncc.clamp(0.0, 1.0) * f64::from(u8::MAX)).round() as u8
                }
            }),
        });
    }

    // ---- The point the record stands on ----
    let frame = payload.frame.as_ref();
    let normal = frame.map_or_else(Vector3::zeros, |patch| {
        let n = patch.normal();
        Vector3::new(n.x as f32, n.y as f32, n.z as f32)
    });
    let record = PointRecord {
        point: Point3D {
            position,
            w: frame.map_or(1.0, |patch| patch.w),
            color: bitmap_color(payload),
            error: mean_reprojection_error(track, &kept),
            normal,
        },
        observations,
        patch_u_halfvec: frame
            .filter(|_| edited.has_patch_frames())
            .map(|patch| halfvec(patch.u_axis * patch.half_extent[0])),
        patch_v_halfvec: frame
            .filter(|_| edited.has_patch_frames())
            .map(|patch| halfvec(patch.v_axis * patch.half_extent[1])),
        patch_bitmap: edited
            .has_patch_bitmaps()
            .then(|| payload.bitmap.clone().expect("checked above")),
        normal_confidence: edited
            .has_normal_confidence()
            .then(|| payload.normal_confidence.unwrap_or(0)),
        // A committed track states nothing about a distance a caller owns, so
        // its point is free.
        constraint: edited.has_point_constraints().then_some((
            sfmtool_sfmr_format::POINT_CONSTRAINT_FREE,
            f64::NAN,
            sfmtool_sfmr_format::NO_REFERENCE_IMAGE,
        )),
    };

    // ---- The edit ----
    let origin = track
        .origin
        .map(|o| o.point)
        .filter(|&p| edited.point(p).is_some());
    let mut next = edited.clone();
    let (point, written) = match origin {
        Some(replaced) => {
            let point = next.replace_point(replaced, record)?;
            (point, PointMap::Replaced(vec![(replaced, point)]))
        }
        None => {
            let point = next.add_point(record)?;
            (point, PointMap::Created(vec![point]))
        }
    };

    // ---- The points the kept observations were pulled from ----
    let mut absorbed: Vec<u32> = kept
        .iter()
        .filter_map(|&i| match track.observations[i].provenance {
            Provenance::Point { point } => Some(point),
            _ => None,
        })
        .filter(|point| Some(*point) != origin)
        .collect();
    absorbed.sort_unstable();
    absorbed.dedup();
    // A point another version already deleted is nothing to absorb, and a
    // commit is not the place to complain about it.
    absorbed.retain(|&point| next.delete_point(point).is_ok());

    // The write first, then what it absorbed: the absorbed indexes are the ones
    // this value held before the write, and a point edit leaves indexes where
    // they were, so the two steps compose in the order they were applied.
    let map = if absorbed.is_empty() {
        written
    } else {
        PointMap::Chain(vec![written, PointMap::Removed(absorbed)])
    };

    Ok((
        next,
        CommitReport {
            point,
            replaced: origin,
            map,
            observation_count: rows.len(),
        },
    ))
}

/// The colour the committed point carries: the centre of the consensus bitmap
/// when there is one, and the colour the payload carries otherwise.
///
/// The bitmap is the appearance every `in` observation agreed on, so its centre
/// is the colour of the surface at the point rather than the colour one
/// photograph happened to show there.
fn bitmap_color(payload: &TrackPayload) -> [u8; 3] {
    let Some(bitmap) = &payload.bitmap else {
        return payload.color;
    };
    let shape = bitmap.shape();
    if shape[0] == 0 || shape[1] == 0 || shape[2] == 0 {
        return payload.color;
    }
    let (row, col) = (shape[0] / 2, shape[1] / 2);
    let mut color = [0u8; 3];
    for (c, out) in color.iter_mut().enumerate() {
        let channel = if shape[2] >= 3 { c } else { 0 };
        *out = bitmap[[row, col, channel]];
    }
    color
}

/// The `error` column of the committed point: the mean of what the last
/// evaluation measured at each `in` observation.
///
/// The column is a mean pixel reprojection error and the track stage has
/// measured exactly that, against the position the commit is about to write, so
/// the number is there to be carried across rather than left at zero. Zero is
/// what a point with nothing measured gets, which is the same thing
/// `recompute_point_errors` leaves on a point no observation could be scored
/// for.
fn mean_reprojection_error(track: &EditableTrack, kept: &[usize]) -> f32 {
    let measured: Vec<f64> = kept
        .iter()
        .filter_map(|&i| track.observations[i].track.as_ref())
        .filter_map(|m| m.reprojection_error)
        .filter(|e| e.is_finite())
        .collect();
    if measured.is_empty() {
        return 0.0;
    }
    (measured.iter().sum::<f64>() / measured.len() as f64) as f32
}

/// A world half-vector as the column's `f32` triple.
fn halfvec(v: Vector3<f64>) -> [f32; 3] {
    [v.x as f32, v.y as f32, v.z as f32]
}
