// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! An [`EditedReconstruction`]: a shared immutable base plus the point edits
//! made on top of it, and the materialisation that turns the pair back into a
//! plain [`SfmrReconstruction`].
//!
//! [`PointMap`] is what one edit says it did to point indexes, whatever kind of
//! edit it was: a whole-value edit's [`RowMap`] is one of its cases, and a
//! caller carrying an index across an edit reads every case the same way.
//!
//! `specs/core/reconstruction/edited-reconstruction.md` is the design; this
//! module is the whole of it, apart from the `.sfmr` writer's hashing, which
//! `sfmtool_sfmr_format::content_hash_of` supplies.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, OnceLock};

use nalgebra::Vector3;
use ndarray::{Array2, Array3, Array4, ArrayView3, Axis};
use xxhash_rust::xxh3::Xxh3;

use sfmtool_archive_io::format_hash;
use sfmtool_sfmr_format::{
    ContentHash, SfmrError, NO_REFERENCE_IMAGE, NO_REFERENCE_OBSERVATION, POINT_CONSTRAINT_FREE,
    POINT_CONSTRAINT_HELD, POINT_CONSTRAINT_RANGED,
};

use crate::patch::cloud::OrientedPatch;

use super::data::{
    ImageTable, ObservationReading, ObservationReadings, ObservationSource, Point3D,
    PointConstraintColumns, PointSet, SfmrReconstruction, TrackObservation,
};

mod row_map;

pub use row_map::RowMap;

/// Why an edit was refused. Every variant names the index or the column that
/// did not hold, because a caller building a record has no other way to tell
/// which of its many parallel pieces was wrong.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EditError {
    /// The edited index names no live point: past the end, or already deleted.
    NoSuchPoint(u32),
    /// An observation names an image the base's image table does not hold.
    ImageOutOfRange {
        /// Which observation of the record.
        observation: usize,
        /// The image index it named.
        image_index: u32,
        /// How many images the base holds.
        image_count: usize,
    },
    /// The record carries a column the base does not, or omits one it does.
    /// An addition set never introduces or removes a column.
    ColumnMismatch {
        /// The column's name, as the point set spells it.
        column: &'static str,
        /// Whether the base carries it.
        base_has: bool,
    },
    /// A record with no observations. A point that nothing sees is not a point.
    NoObservations,
    /// A patch bitmap whose resolution is not the base's.
    PatchBitmapShape {
        /// The record's `(rows, cols, channels)`.
        got: (usize, usize, usize),
        /// The base's.
        expected: (usize, usize, usize),
    },
    /// A constraint code the format does not define.
    UnknownConstraint(u8),
    /// A constraint whose reference image the base's image table does not hold.
    ConstraintImageOutOfRange(u32),
    /// A reference observation that is neither `-1` nor an index into the
    /// record's observations.
    ReferenceObservationOutOfRange {
        /// The record's reference observation.
        reference: i32,
        /// How many observations the record carries.
        observations: usize,
    },
    /// [`RowMap::by_scan`] was given an image map that is not one entry per
    /// image of the *before* reconstruction.
    ScanImageMap {
        /// The map's length.
        got: usize,
        /// The before reconstruction's image count.
        expected: usize,
    },
}

impl std::fmt::Display for EditError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EditError::NoSuchPoint(i) => write!(f, "no live point at edited index {i}"),
            EditError::ImageOutOfRange {
                observation,
                image_index,
                image_count,
            } => write!(
                f,
                "observation {observation} names image {image_index}, past the {image_count} \
                 images of the base"
            ),
            EditError::ColumnMismatch { column, base_has } => {
                if *base_has {
                    write!(f, "the base carries '{column}' and the record does not")
                } else {
                    write!(f, "the record carries '{column}' and the base does not")
                }
            }
            EditError::NoObservations => write!(f, "a point record carries no observations"),
            EditError::PatchBitmapShape { got, expected } => write!(
                f,
                "patch bitmap is {got:?}, and the base's bitmaps are {expected:?}"
            ),
            EditError::UnknownConstraint(k) => write!(f, "constraint {k} is not defined"),
            EditError::ConstraintImageOutOfRange(i) => {
                write!(
                    f,
                    "constraint measures its distance from image {i}, which the base \
                          does not hold"
                )
            }
            EditError::ReferenceObservationOutOfRange {
                reference,
                observations,
            } => write!(
                f,
                "reference observation {reference} is neither -1 nor one of the record's \
                 {observations} observations"
            ),
            EditError::ScanImageMap { got, expected } => write!(
                f,
                "the image map has {got} entries and the original reconstruction has \
                 {expected} images"
            ),
        }
    }
}

impl std::error::Error for EditError {}

/// One observation of a [`PointRecord`]: the image that sees the point, and
/// whatever the base's columns say about that sighting.
///
/// It names no point index. An observation's point is the record it belongs to,
/// and the index that identifies that point differs between the overlay and the
/// materialisation, so carrying one here would be a second answer to a question
/// the record already answers.
#[derive(Debug, Clone, PartialEq)]
pub struct RecordObservation {
    /// The image, as an index into the base's image table.
    pub image_index: u32,
    /// The `.sift` feature this observation is, for a `sift_files` base.
    pub feature_index: Option<u32>,
    /// The sub-pixel `(u, v)`, when the base carries the inline column.
    pub keypoint_xy: Option<[f32; 2]>,
    /// The photometric-sharpness confidence, when the base carries the column.
    pub confidence: Option<u8>,
    /// The observation's readings on its own render, when the base carries
    /// them ([`PointSet::observation_readings`]).
    pub reading: Option<crate::reconstruction::ObservationReading>,
}

/// A whole point: its geometry, its whole track, and every per-point column the
/// base carries.
///
/// This is the unit a point edit trades in. Delete-and-re-add means a
/// modification is expressed as a complete record rather than as a diff, so
/// there is one code path for "a point that changed" whatever changed about it,
/// and no partial state in which a track and a per-observation column disagree.
#[derive(Debug, Clone, PartialEq)]
pub struct PointRecord {
    /// Position, colour, error and normal.
    pub point: Point3D,
    /// The whole track, in the order it is stored.
    pub observations: Vec<RecordObservation>,
    /// The patch frame's in-plane half-vectors, when the base carries them.
    pub patch_u_halfvec: Option<[f32; 3]>,
    /// The other half-vector; present exactly when `patch_u_halfvec` is.
    pub patch_v_halfvec: Option<[f32; 3]>,
    /// The `(R, R, 4)` patch bitmap, when the base carries the column.
    pub patch_bitmap: Option<Array3<u8>>,
    /// Confidence in the normal, when the base carries the column.
    pub normal_confidence: Option<u8>,
    /// The constraint triple `(constraint, distance, reference image)`, when the
    /// base carries the columns.
    pub constraint: Option<(u8, f64, u32)>,
    /// Which of [`Self::observations`] the patch bitmap is, or is to be,
    /// rendered from, as an index into them, or `-1` for no reference chosen,
    /// when the base carries the column (`tracks/reference_observations`,
    /// present with the patch frame).
    pub reference_observation: Option<i32>,
    /// Whether [`Self::reference_observation`] is a pick only the display
    /// render made ([`PointSet::display_only_references`]), which a save writes
    /// as `-1`. Carried so that a point rewritten through
    /// [`EditedReconstruction::replace_point`] keeps the mark, as the same edit
    /// made in place does. Not compared by [`Self::agrees_with`]: a record
    /// that agrees leaves the point untouched, mark and all.
    pub display_only_reference: bool,
    /// The options the observations' readings
    /// ([`RecordObservation::reading`]) were taken under, `None` where no
    /// observation carries one. A record whose options differ from those of
    /// the readings the value already holds adds rows with nothing measured,
    /// so no column mixes readings taken under different options.
    pub reading_options: Option<crate::reconstruction::ObservationReadingOptions>,
}

impl PointRecord {
    /// Whether this record and `other` say the same thing in every column.
    ///
    /// **Exact on the stored representation**, column for column: the `f64`
    /// coordinate as stored, the `f32` keypoints, the `u8` confidences, the
    /// whole bitmap. Nothing here is approximate, because what this answers is
    /// whether writing this record would leave the point exactly as it is, and
    /// a position that moved by a stored amount is a point that moved.
    ///
    /// The one concession is `NaN`, which is not equal to itself: two `NaN`s in
    /// one column agree here. A free point's constraint distance is `NaN` by
    /// definition, so a structural comparison would report a record as
    /// differing from a copy of itself for no reason other than that neither
    /// constrains anything -- the same reading `constraints_agree` takes of the
    /// column form, held to for every float so that no column can make a record
    /// differ from itself forever.
    pub fn agrees_with(&self, other: &Self) -> bool {
        points_agree(&self.point, &other.point)
            && self.observations.len() == other.observations.len()
            && self
                .observations
                .iter()
                .zip(&other.observations)
                .all(|(a, b)| observations_agree(a, b))
            && halfvecs_agree(self.patch_u_halfvec, other.patch_u_halfvec)
            && halfvecs_agree(self.patch_v_halfvec, other.patch_v_halfvec)
            && self.patch_bitmap == other.patch_bitmap
            && self.normal_confidence == other.normal_confidence
            && constraint_triples_agree(self.constraint, other.constraint)
            && self.reference_observation == other.reference_observation
    }
}

/// Two floats that say the same thing: equal, or both `NaN`.
///
/// `f32` columns are widened to `f64` to be read, which is exact and carries a
/// `NaN` across as one, so one rule covers every float a record holds.
fn floats_agree(a: f64, b: f64) -> bool {
    a == b || (a.is_nan() && b.is_nan())
}

/// Whether two geometries agree: position, `w`, colour, error and normal.
fn points_agree(a: &Point3D, b: &Point3D) -> bool {
    a.color == b.color
        && floats_agree(a.w, b.w)
        && floats_agree(f64::from(a.error), f64::from(b.error))
        && a.position
            .coords
            .iter()
            .zip(b.position.coords.iter())
            .all(|(x, y)| floats_agree(*x, *y))
        && a.normal
            .iter()
            .zip(b.normal.iter())
            .all(|(x, y)| floats_agree(f64::from(*x), f64::from(*y)))
}

/// Whether two observations of a record agree: the image, whatever names the
/// sighting, and its confidence.
fn observations_agree(a: &RecordObservation, b: &RecordObservation) -> bool {
    a.image_index == b.image_index
        && a.feature_index == b.feature_index
        && a.confidence == b.confidence
        && a.reading.unwrap_or(ObservationReading::NOT_MEASURED)
            == b.reading.unwrap_or(ObservationReading::NOT_MEASURED)
        && match (a.keypoint_xy, b.keypoint_xy) {
            (Some(x), Some(y)) => x
                .iter()
                .zip(&y)
                .all(|(u, v)| floats_agree(f64::from(*u), f64::from(*v))),
            (x, y) => x.is_none() && y.is_none(),
        }
}

/// Whether two half-vector columns agree, both absent included.
fn halfvecs_agree(a: Option<[f32; 3]>, b: Option<[f32; 3]>) -> bool {
    match (a, b) {
        (Some(x), Some(y)) => x
            .iter()
            .zip(&y)
            .all(|(u, v)| floats_agree(f64::from(*u), f64::from(*v))),
        (x, y) => x.is_none() && y.is_none(),
    }
}

/// Whether two constraint triples say the same thing, `NaN` distances included.
fn constraint_triples_agree(a: Option<(u8, f64, u32)>, b: Option<(u8, f64, u32)>) -> bool {
    match (a, b) {
        (Some((ka, da, ra)), Some((kb, db, rb))) => ka == kb && ra == rb && floats_agree(da, db),
        (x, y) => x.is_none() && y.is_none(),
    }
}

/// A point read through the overlay: a borrow into whichever point set holds
/// it, with the columns addressed by that set's own local index.
///
/// Handing back a view rather than a record is what lets the viewer's Track View
/// and the track rays read an edited reconstruction at the cost of a
/// slice, with no allocation and no materialisation; [`Self::to_record`] is
/// there for the caller that wants the owned form.
#[derive(Clone, Copy)]
pub struct PointView<'a> {
    set: &'a PointSet,
    local: usize,
}

impl<'a> PointView<'a> {
    /// The point's geometry.
    pub fn point(&self) -> &'a Point3D {
        &self.set.points[self.local]
    }

    /// The point's whole track, in stored order.
    pub fn observations(&self) -> &'a [TrackObservation] {
        self.set.observations_for_point(self.local)
    }

    /// The observation rows this point's track occupies in its own set, which
    /// is what every per-observation column here is addressed by.
    fn rows(&self) -> std::ops::Range<usize> {
        self.set.observation_offsets[self.local]..self.set.observation_offsets[self.local + 1]
    }

    /// The `.sift` feature index of each observation, for a `sift_files` base.
    pub fn feature_indexes(&self) -> Option<&'a [u32]> {
        let rows = self.rows();
        self.set.feature_indexes().map(|f| &f[rows])
    }

    /// The photometric-sharpness confidence of each observation, when the
    /// column is carried.
    pub fn observation_confidence(&self) -> Option<&'a [u8]> {
        let rows = self.rows();
        self.set.observation_confidence.as_ref().map(|c| &c[rows])
    }

    /// The readings of each observation on its own render, when the column
    /// is carried ([`PointSet::observation_readings`]).
    pub fn observation_readings(&self) -> Option<&'a [crate::reconstruction::ObservationReading]> {
        let rows = self.rows();
        self.set
            .observation_readings
            .as_ref()
            .map(|r| &r.rows[rows])
    }

    /// The options [`Self::observation_readings`] were taken under.
    pub fn observation_reading_options(
        &self,
    ) -> Option<crate::reconstruction::ObservationReadingOptions> {
        self.set.observation_readings.as_ref().map(|r| r.options)
    }

    /// The sub-pixel `(u, v)` of observation `k` of this track.
    pub fn keypoint_xy(&self, k: usize) -> Option<[f32; 2]> {
        let row = self.set.observation_offsets[self.local] + k;
        let kp = self.set.keypoints_xy()?;
        Some([kp[[row, 0]], kp[[row, 1]]])
    }

    /// The patch frame's `u` half-vector, when the base carries the frame.
    pub fn patch_u_halfvec(&self) -> Option<[f32; 3]> {
        halfvec_row(&self.set.patch_u_halfvec_xyz, self.local)
    }

    /// The patch frame's `v` half-vector, when the base carries the frame.
    pub fn patch_v_halfvec(&self) -> Option<[f32; 3]> {
        halfvec_row(&self.set.patch_v_halfvec_xyz, self.local)
    }

    /// The point's patch placement, when the base carries the frame columns
    /// and this row's half-vectors are non-zero.
    ///
    /// The stored half-vectors are split into a unit axis and a half-size, the
    /// centre is the point's position, and `w` is the **point's** own, since a
    /// bearing's patch is tangent to the direction sphere and the two stored
    /// half-vectors do not say which kind it is.
    pub fn placement(&self) -> Option<OrientedPatch> {
        let axis = |h: [f32; 3]| Vector3::new(f64::from(h[0]), f64::from(h[1]), f64::from(h[2]));
        let u = axis(self.patch_u_halfvec()?);
        let v = axis(self.patch_v_halfvec()?);
        let (hu, hv) = (u.norm(), v.norm());
        if !(hu > 0.0 && hv > 0.0) {
            return None;
        }
        let point = self.point();
        let mut patch = OrientedPatch::new(point.position, u / hu, v / hv, [hu, hv]);
        patch.w = if point.is_at_infinity() { 0.0 } else { 1.0 };
        Some(patch)
    }

    /// The `(R, R, 4)` patch bitmap, when the base carries the column.
    pub fn patch_bitmap(&self) -> Option<ArrayView3<'a, u8>> {
        self.set
            .patch_bitmaps_y_x_rgba
            .as_ref()
            .map(|b| b.index_axis(Axis(0), self.local))
    }

    /// The confidence in this point's normal, when the column is carried.
    pub fn normal_confidence(&self) -> Option<u8> {
        self.set.normal_confidence.as_ref().map(|c| c[self.local])
    }

    /// The constraint triple, when the columns are carried.
    pub fn constraint(&self) -> Option<(u8, f64, u32)> {
        self.set.point_constraints.as_ref().map(|c| {
            (
                c.point_constraints[self.local],
                c.constraint_distances[self.local],
                c.constraint_reference_images[self.local],
            )
        })
    }

    /// The index, among [`Self::observations`], of the observation the patch
    /// bitmap is, or is to be, rendered from, `-1` for no reference chosen,
    /// when the base carries the column.
    pub fn reference_observation(&self) -> Option<i32> {
        self.set
            .reference_observations
            .as_ref()
            .map(|r| r[self.local])
    }

    /// Whether this point's reference is a pick only the display render made
    /// ([`PointSet::display_only_references`]), which a save writes as `-1`.
    pub fn display_only_reference(&self) -> bool {
        self.set
            .display_only_references
            .as_ref()
            .is_some_and(|m| m[self.local])
    }

    /// The owned form of everything above.
    pub fn to_record(&self) -> PointRecord {
        let observations = self
            .observations()
            .iter()
            .enumerate()
            .map(|(k, obs)| RecordObservation {
                image_index: obs.image_index,
                feature_index: self.feature_indexes().map(|f| f[k]),
                keypoint_xy: self.keypoint_xy(k),
                confidence: self.observation_confidence().map(|c| c[k]),
                reading: self.observation_readings().map(|r| r[k]),
            })
            .collect();
        PointRecord {
            point: self.point().clone(),
            observations,
            patch_u_halfvec: self.patch_u_halfvec(),
            patch_v_halfvec: self.patch_v_halfvec(),
            patch_bitmap: self.patch_bitmap().map(|b| b.to_owned()),
            normal_confidence: self.normal_confidence(),
            constraint: self.constraint(),
            reference_observation: self.reference_observation(),
            display_only_reference: self.display_only_reference(),
            reading_options: self.set.observation_readings.as_ref().map(|r| r.options),
        }
    }
}

/// Row `i` of an optional `(P, 3)` half-vector column.
fn halfvec_row(arr: &Option<Array2<f32>>, i: usize) -> Option<[f32; 3]> {
    arr.as_ref().map(|a| [a[[i, 0]], a[[i, 1]], a[[i, 2]]])
}

/// A reconstruction value that is a shared base plus what this version changed.
///
/// The base is never written. A point edit deletes an index of the base and, if
/// the point survives the edit, re-adds its whole record to `added`, so the
/// cost of an edit is the size of the records it touches and every version in a
/// run of edits shares one base. Indexes are stable for the base's lifetime:
/// a base point keeps its index, a deleted index resolves to nothing and is
/// never reused, and an addition takes the next index at or after the base's
/// point count. [`Self::materialize`] is the one operation that renumbers.
///
/// Equality is over the base's *contents* and the edits, so two values built
/// from equal bases with equal edits are equal even when the bases are separate
/// allocations. It compares the point side only, which is what an overlay edit
/// can change, and it reads two `NaN` constraint distances as agreeing, since
/// both say the point is at no distance from anything.
pub struct EditedReconstruction {
    /// What was loaded, or the last materialisation. Shared by every version in
    /// a run of point edits.
    pub base: Arc<SfmrReconstruction>,
    /// The edited indexes that are gone. Below `base.point_count()` these are
    /// base indexes; at or above it they are additions this overlay has since
    /// deleted or replaced, whose rows stay in `added` so that the indexes
    /// after them do not move.
    pub deleted_points: HashSet<u32>,
    /// The points this version holds and the base does not, as a point set with
    /// exactly the base's columns, whose observations index the base's image
    /// table.
    pub added: PointSet,
    /// For each added point, the base index it replaces (a modified point), or
    /// `None` for a point that is new. What puts a modified point back in its
    /// place at materialisation.
    pub replaces: Vec<Option<u32>>,
    /// The base's content hashes, computed on first request. Not part of the
    /// value: two equal values agree on it whether or not either has asked.
    base_hash: OnceLock<ContentHash>,
    /// What the point-or-bearing test reads off the base, measured on first
    /// request and shared by every version cloned from this one, since they
    /// share the base. Not part of the value, for the reason `base_hash` is
    /// not.
    base_measures: Arc<BaseMeasures>,
}

/// The base's measured noise level and minimum point depth, each computed
/// once. See [`EditedReconstruction::base_reprojection_noise_px`].
#[derive(Default)]
struct BaseMeasures {
    /// `reprojection_noise_px`, or why there is none, as a sentence.
    noise_px: OnceLock<Result<f64, String>>,
    /// `min_point_depth`.
    min_point_depth: OnceLock<f64>,
}

impl Clone for EditedReconstruction {
    fn clone(&self) -> Self {
        Self {
            base: Arc::clone(&self.base),
            deleted_points: self.deleted_points.clone(),
            added: self.added.clone(),
            replaces: self.replaces.clone(),
            base_hash: self.base_hash.clone(),
            base_measures: Arc::clone(&self.base_measures),
        }
    }
}

impl std::fmt::Debug for EditedReconstruction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EditedReconstruction")
            .field("base_point_count", &self.base.point_count())
            .field("deleted", &self.deleted_points.len())
            .field("added", &self.added.point_count())
            .finish()
    }
}

impl PartialEq for EditedReconstruction {
    fn eq(&self, other: &Self) -> bool {
        // The overlay first: it is the small half, and it is what differs
        // between two versions of the same base.
        self.deleted_points == other.deleted_points
            && self.replaces == other.replaces
            && point_sets_equal(&self.added, &other.added)
            && (Arc::ptr_eq(&self.base, &other.base)
                || point_sets_equal(&self.base.point_set, &other.base.point_set))
    }
}

/// Whether two point sets hold the same points, tracks and columns. The derived
/// indexes are a function of those, so they are not compared.
fn point_sets_equal(a: &PointSet, b: &PointSet) -> bool {
    a.points == b.points
        && a.observation_counts == b.observation_counts
        && a.tracks == b.tracks
        && a.feature_indexes() == b.feature_indexes()
        && a.keypoints_xy() == b.keypoints_xy()
        && a.observation_confidence == b.observation_confidence
        && a.observation_readings == b.observation_readings
        && a.patch_u_halfvec_xyz == b.patch_u_halfvec_xyz
        && a.patch_v_halfvec_xyz == b.patch_v_halfvec_xyz
        && a.patch_bitmaps_y_x_rgba.as_deref() == b.patch_bitmaps_y_x_rgba.as_deref()
        && a.normal_confidence == b.normal_confidence
        && constraints_agree(&a.point_constraints, &b.point_constraints)
        && a.reference_observations == b.reference_observations
        && a.display_only_references == b.display_only_references
}

/// Whether two constraint columns say the same thing.
///
/// A free point's distance is `NaN`, which is not equal to itself, so a
/// structural comparison would report two identical reconstructions as
/// different for no reason other than that neither constrains anything. Two
/// `NaN` distances say the same thing -- no distance -- so here they agree.
fn constraints_agree(
    a: &Option<PointConstraintColumns>,
    b: &Option<PointConstraintColumns>,
) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(a), Some(b)) => {
            a.point_constraints == b.point_constraints
                && a.constraint_reference_images == b.constraint_reference_images
                && a.constraint_distances.len() == b.constraint_distances.len()
                && a.constraint_distances
                    .iter()
                    .zip(&b.constraint_distances)
                    .all(|(x, y)| floats_agree(*x, *y))
        }
        _ => false,
    }
}

impl EditedReconstruction {
    /// A version that is exactly its base: nothing deleted, nothing added.
    pub fn new(base: Arc<SfmrReconstruction>) -> Self {
        let added = empty_like(&base.point_set, base.image_count());
        Self {
            base,
            deleted_points: HashSet::new(),
            added,
            replaces: Vec::new(),
            base_hash: OnceLock::new(),
            base_measures: Arc::default(),
        }
    }

    /// The base's reprojection noise level, in px: its
    /// [`reprojection_noise_px`](SfmrReconstruction::reprojection_noise_px),
    /// measured on the first call and kept for every version that shares this
    /// base.
    ///
    /// The noise level the bench weights a track's rays by in the
    /// point-or-bearing test. It is the base's and not this version's because a
    /// version differs from its base by a handful of edited points, while the
    /// measure is over every observation of every finite point; and the base is
    /// what a run of edits shares, so the measure is taken once per base rather
    /// than once per edit. A new base (a load, a materialisation, a whole-value
    /// edit) starts with none.
    ///
    /// # Errors
    ///
    /// A sentence saying why there is no level: no observation of a finite
    /// point to measure it from, or pixels that could not be read (a
    /// `sift_files` base without its `.sift` files). The error is kept too, so
    /// a base without a level does not re-read its `.sift` files on every call.
    /// A base whose keypoints sit at the exact projections of its points
    /// measures the resolution an `f32` keypoint is stored at
    /// ([`keypoint_resolution_px`](crate::analysis::reprojection_noise::keypoint_resolution_px)).
    pub fn base_reprojection_noise_px(&self) -> Result<f64, String> {
        self.base_measures
            .noise_px
            .get_or_init(|| match self.base.reprojection_noise_px() {
                Ok(Some(s)) if s.is_finite() => Ok(s),
                Ok(Some(s)) => Err(format!(
                    "the reconstruction's reprojection noise measures {s} px"
                )),
                Ok(None) => Err(
                    "the reconstruction has no observation of a finite point to \
                                 measure the reprojection noise from"
                        .to_string(),
                ),
                Err(e) => Err(format!("the reprojection noise could not be measured: {e}")),
            })
            .clone()
    }

    /// The base's [`min_point_depth`](SfmrReconstruction::min_point_depth),
    /// computed on the first call and kept as
    /// [`Self::base_reprojection_noise_px`] is.
    pub fn base_min_point_depth(&self) -> f64 {
        *self
            .base_measures
            .min_point_depth
            .get_or_init(|| self.base.min_point_depth())
    }

    /// How many points this version holds: the base's, less the deletions, plus
    /// the live additions. O(1).
    pub fn point_count(&self) -> usize {
        self.base.point_count() + self.added.point_count() - self.deleted_points.len()
    }

    /// How many of this version's points are at infinity: the base's, plus the
    /// additions at infinity, less the deleted or replaced points at infinity,
    /// whether a deleted index is a base row or an addition. A replacement that
    /// turns a position into a bearing, or a bearing into a position, is
    /// counted by its new record. Linear in the additions and the deletions,
    /// which are the number of hand edits a version stands on.
    pub fn infinity_point_count(&self) -> usize {
        let base_count = self.base.point_count();
        let added_at_infinity = self
            .added
            .points
            .iter()
            .filter(|p| p.is_at_infinity())
            .count();
        let deleted_at_infinity = self
            .deleted_points
            .iter()
            .filter(|&&index| {
                let i = index as usize;
                let point = if i < base_count {
                    self.base.point_set.points.get(i)
                } else {
                    self.added.points.get(i - base_count)
                };
                point.is_some_and(|p| p.is_at_infinity())
            })
            .count();
        self.base.point_set.infinity_point_count + added_at_infinity - deleted_at_infinity
    }

    /// How many observations this version holds: the base's and the additions',
    /// less every observation of a point the version has deleted.
    ///
    /// The count a version reports rather than the base's `tracks.len()`: a
    /// point edit adds and removes whole tracks, so a reader that asks the base
    /// is told what the file held before the session started. Linear in the
    /// deleted set, which is the number of hand edits a version stands on
    /// rather than anything that scales with the scene.
    pub fn observation_count(&self) -> usize {
        let base_count = self.base.point_count();
        let live = self.base.point_set.tracks.len() + self.added.tracks.len();
        live - self
            .deleted_points
            .iter()
            .map(|&index| {
                let i = index as usize;
                let (set, local) = if i < base_count {
                    (&self.base.point_set, i)
                } else {
                    (&self.added, i - base_count)
                };
                set.observations_for_point(local).len()
            })
            .sum::<usize>()
    }

    /// How many images: the base's, always. An image edit is not an overlay
    /// edit.
    pub fn image_count(&self) -> usize {
        self.base.image_count()
    }

    /// How many points the base holds, which is where the addition indexes
    /// start.
    pub fn base_point_count(&self) -> usize {
        self.base.point_count()
    }

    /// One past the largest index this version has ever handed out. Every live
    /// point's index is below it, and no index below it is ever reused.
    pub fn index_bound(&self) -> u32 {
        (self.base.point_count() + self.added.point_count()) as u32
    }

    /// Whether `index` named a point that has since been deleted or replaced.
    pub fn is_deleted(&self, index: u32) -> bool {
        self.deleted_points.contains(&index)
    }

    /// Where the point a *base row* holds lives in this version, or `None` when
    /// the version deleted it outright.
    ///
    /// The base index itself when no edit has touched the point, and the
    /// addition that superseded it when one replaced it. `replace_point`
    /// carries the base index a record descends from forward through a chain,
    /// so a point modified twice still answers here, and the deleted set picks
    /// the one link of that chain which is live.
    ///
    /// The accessor for a caller holding an index *into the base* rather than
    /// into the version. `PointSet::image_feature_to_point` is the one in the
    /// viewer: it is built once per base and says nothing about what the
    /// versions above it did, so reading it without this turns a deleted point
    /// into a feature that still draws and still selects.
    ///
    /// Linear in the addition count, which is the number of hand edits a
    /// version stands on rather than anything that scales with the scene.
    pub fn live_index_of_base(&self, base_index: u32) -> Option<u32> {
        let base_count = self.base.point_count() as u32;
        if base_index >= base_count {
            return None;
        }
        if !self.deleted_points.contains(&base_index) {
            return Some(base_index);
        }
        self.replaces
            .iter()
            .enumerate()
            .map(|(a, replaced)| (base_count + a as u32, replaced))
            .find(|(index, replaced)| {
                **replaced == Some(base_index) && !self.deleted_points.contains(index)
            })
            .map(|(index, _)| index)
    }

    /// The point at `index`, or `None` when that index names no live point.
    ///
    /// Below the base's point count the index is a base index; at or above it,
    /// an addition. The caller does not learn which, which is the point: the
    /// panel, the picker and the ray builder read one accessor and never
    /// materialise.
    pub fn point(&self, index: u32) -> Option<PointView<'_>> {
        if self.deleted_points.contains(&index) {
            return None;
        }
        let base_count = self.base.point_count();
        let i = index as usize;
        if i < base_count {
            Some(PointView {
                set: &self.base.point_set,
                local: i,
            })
        } else if i < base_count + self.added.point_count() {
            Some(PointView {
                set: &self.added,
                local: i - base_count,
            })
        } else {
            None
        }
    }

    /// The images that observe the point at `index`, in stored order, or an
    /// empty list when the index names no live point.
    ///
    /// The overlay's answer to `SfmrReconstruction::track_image_indices`, which
    /// reads the base and so cannot see an addition's track.
    pub fn track_image_indices(&self, index: u32) -> Vec<usize> {
        self.point(index).map_or_else(Vec::new, |view| {
            view.observations()
                .iter()
                .map(|o| o.image_index as usize)
                .collect()
        })
    }

    /// The affine shape of one observation's keypoint, read through the
    /// overlay.
    ///
    /// The overlay's answer to `SfmrReconstruction::observation_affine_shape`:
    /// the same projection algebra, over the frame and position this version
    /// holds for the point rather than the base's.
    pub fn observation_affine_shape(
        &self,
        index: u32,
        image_index: usize,
        keypoint_xy: [f32; 2],
    ) -> Option<[[f32; 2]; 2]> {
        let view = self.point(index)?;
        let u = view.patch_u_halfvec()?;
        let v = view.patch_v_halfvec()?;
        super::data::patch_affine_shape(
            view.point(),
            nalgebra::Vector3::new(u[0] as f64, u[1] as f64, u[2] as f64),
            nalgebra::Vector3::new(v[0] as f64, v[1] as f64, v[2] as f64),
            &self.base.image_table,
            image_index,
            keypoint_xy,
        )
    }

    /// Every live index, ascending. Base points first, then additions in the
    /// order they were made -- which is materialisation's order for the
    /// appended points and not for the modified ones.
    pub fn live_indexes(&self) -> impl Iterator<Item = u32> + '_ {
        (0..self.index_bound()).filter(|i| !self.deleted_points.contains(i))
    }

    // ── Column presence, answered from the base ──────────────────────

    /// The `feature_source` discriminator, the base's.
    pub fn feature_source(&self) -> &str {
        self.base.feature_source()
    }

    /// Whether observations carry `.sift` feature indexes.
    pub fn has_feature_indexes(&self) -> bool {
        self.base.feature_indexes().is_some()
    }

    /// Whether observations carry inline keypoints.
    pub fn has_keypoints(&self) -> bool {
        self.base.keypoints_xy().is_some()
    }

    /// Whether observations carry a confidence.
    pub fn has_observation_confidence(&self) -> bool {
        self.base.point_set.observation_confidence.is_some()
    }

    /// Whether observations carry their readings
    /// ([`PointSet::observation_readings`]).
    pub fn has_observation_readings(&self) -> bool {
        self.base.point_set.observation_readings.is_some()
    }

    /// The options this version's readings stand under: the base's where it
    /// carries readings, else those of the first record that brought readings
    /// to it, `None` where neither has any. A record whose readings stand
    /// under other options adds rows with nothing measured.
    pub fn observation_reading_options(
        &self,
    ) -> Option<crate::reconstruction::ObservationReadingOptions> {
        self.base
            .point_set
            .observation_readings
            .as_ref()
            .or(self.added.observation_readings.as_ref())
            .map(|r| r.options)
    }

    /// Whether points carry a patch frame.
    pub fn has_patch_frames(&self) -> bool {
        self.base.point_set.patch_u_halfvec_xyz.is_some()
    }

    /// Whether points carry a patch bitmap.
    pub fn has_patch_bitmaps(&self) -> bool {
        self.base.point_set.patch_bitmaps_y_x_rgba.is_some()
    }

    /// Whether points carry a normal confidence.
    pub fn has_normal_confidence(&self) -> bool {
        self.base.point_set.normal_confidence.is_some()
    }

    /// Whether points carry the constraint triple.
    pub fn has_point_constraints(&self) -> bool {
        self.base.point_set.point_constraints.is_some()
    }

    /// Whether points carry a reference observation, which they do exactly
    /// when they carry a patch frame.
    pub fn has_reference_observations(&self) -> bool {
        self.base.point_set.reference_observations.is_some()
    }

    /// How many camera models the **posed** images of this value are taken
    /// through, which is zero when none of them carries a pose.
    ///
    /// Zero means nothing is posed. The viewer's menu gates for the bundle
    /// adjustment, the retriangulation and the covered-observation prune ask
    /// that question here, and so does `prune_covered_observations` before it
    /// starts, so for the prune the sentence its menu entry is greyed with and
    /// the refusal the operation produces are read off one count. An unposed
    /// image states no ray, so its lens is not counted.
    pub fn posed_lens_count(&self) -> usize {
        self.posed_lenses().len()
    }

    /// The camera-table indexes the **posed** images of this value are taken
    /// through, ascending and each once: the cameras a bundle adjustment
    /// solves, and so the ones a caller offering its focal release checks.
    pub fn posed_lenses(&self) -> Vec<u32> {
        let mut lenses: Vec<u32> = self
            .base
            .image_table
            .images
            .iter()
            .filter(|image| {
                image.quaternion_wxyz.coords.iter().all(|c| c.is_finite())
                    && image.translation_xyz.iter().all(|c| c.is_finite())
            })
            .map(|image| image.camera_index)
            .collect();
        lenses.sort_unstable();
        lenses.dedup();
        lenses
    }

    // ── The point edits ──────────────────────────────────────────────

    /// Delete the point at `index`.
    ///
    /// The base is untouched: the index joins the deleted set, and every other
    /// index keeps resolving to the record it resolved to before.
    pub fn delete_point(&mut self, index: u32) -> Result<(), EditError> {
        if self.point(index).is_none() {
            return Err(EditError::NoSuchPoint(index));
        }
        self.deleted_points.insert(index);
        Ok(())
    }

    /// Replace the point at `index` with `record`, and give back the index the
    /// replacement took.
    ///
    /// This is delete-and-re-add: whatever changed about the point -- its
    /// position, its constraint, its frame, its track -- the old index is
    /// deleted and the whole new record is appended. The base index the point
    /// occupies is recorded against the replacement, so materialisation puts it
    /// back in its place; replacing a point that is itself a replacement keeps
    /// the original base index rather than chaining.
    pub fn replace_point(&mut self, index: u32, record: PointRecord) -> Result<u32, EditError> {
        let base_count = self.base.point_count() as u32;
        if self.point(index).is_none() {
            return Err(EditError::NoSuchPoint(index));
        }
        let replaces = if index < base_count {
            Some(index)
        } else {
            self.replaces[(index - base_count) as usize]
        };
        let new_index = self.push_record(record, replaces)?;
        self.deleted_points.insert(index);
        Ok(new_index)
    }

    /// Add a point that the base does not hold, and give back its index.
    pub fn add_point(&mut self, record: PointRecord) -> Result<u32, EditError> {
        self.push_record(record, None)
    }

    /// Append `record` to the addition set and record what it replaces.
    ///
    /// The validation happens before anything is pushed, so a rejected record
    /// leaves the addition set exactly as it was -- otherwise a caller that
    /// recovers from an error would be reading a set whose columns had stopped
    /// being parallel.
    fn push_record(
        &mut self,
        record: PointRecord,
        replaces: Option<u32>,
    ) -> Result<u32, EditError> {
        self.validate_record(&record)?;

        let image_count = self.base.image_count();
        let base_count = self.base.point_count();
        let set = &mut self.added;
        set.points.push(record.point);
        set.observation_counts
            .push(record.observations.len() as u32);
        let local = set.points.len() - 1;
        for obs in &record.observations {
            set.tracks.push(TrackObservation {
                image_index: obs.image_index,
                point_index: local as u32,
            });
        }
        match &mut set.observations {
            ObservationSource::SiftFiles {
                feature_indexes,
                keypoints_xy,
                ..
            } => {
                for obs in &record.observations {
                    feature_indexes.push(obs.feature_index.expect("validated present"));
                }
                if let Some(kp) = keypoints_xy {
                    append_keypoints(kp, &record.observations);
                }
            }
            ObservationSource::EmbeddedPatches { keypoints_xy, .. } => {
                append_keypoints(keypoints_xy, &record.observations);
            }
        }
        if let Some(c) = &mut set.observation_confidence {
            for obs in &record.observations {
                c.push(obs.confidence.expect("validated present"));
            }
        }
        // A record that brings readings to a value without them creates the
        // column under its options, the rows added before it not measured. A
        // row is kept only where the record's options are the column's; a
        // record without readings, or under other options, adds rows with
        // nothing measured.
        let brings = record
            .observations
            .iter()
            .any(|o| o.reading.is_some_and(|r| r.is_measured()));
        if set.observation_readings.is_none() && brings {
            if let Some(options) = record.reading_options {
                let before = set.tracks.len() - record.observations.len();
                set.observation_readings = Some(ObservationReadings::not_measured(before, options));
            }
        }
        if let Some(r) = &mut set.observation_readings {
            let same = record.reading_options == Some(r.options);
            for obs in &record.observations {
                r.rows.push(
                    obs.reading
                        .filter(|_| same)
                        .unwrap_or(ObservationReading::NOT_MEASURED),
                );
            }
        }
        if let Some(u) = &mut set.patch_u_halfvec_xyz {
            push_halfvec(u, record.patch_u_halfvec.expect("validated present"));
        }
        if let Some(v) = &mut set.patch_v_halfvec_xyz {
            push_halfvec(v, record.patch_v_halfvec.expect("validated present"));
        }
        if let Some(b) = &mut set.patch_bitmaps_y_x_rgba {
            // The addition set's bitmaps are its own array, never the base's,
            // so taking it out of the `Arc` copies only on the first push after
            // a clone and never touches a shared value.
            let placeholder = Arc::new(Array4::<u8>::zeros((0, 0, 0, 0)));
            let mut owned = Arc::unwrap_or_clone(std::mem::replace(b, placeholder));
            let bitmap = record.patch_bitmap.expect("validated present");
            owned
                .push(Axis(0), bitmap.view())
                .expect("a validated bitmap has the column's row shape");
            *b = Arc::new(owned);
        }
        if let Some(c) = &mut set.normal_confidence {
            c.push(record.normal_confidence.expect("validated present"));
        }
        if let Some(c) = &mut set.point_constraints {
            let (k, d, r) = record.constraint.expect("validated present");
            c.point_constraints.push(k);
            c.constraint_distances.push(d);
            c.constraint_reference_images.push(r);
        }
        if let Some(r) = &mut set.reference_observations {
            r.push(record.reference_observation.expect("validated present"));
        }
        if let Some(m) = &mut set.display_only_references {
            m.push(record.display_only_reference);
        }
        set.rebuild_derived_fields(image_count);
        self.replaces.push(replaces);
        Ok(base_count as u32 + local as u32)
    }

    /// Check a record against the base: its observations name images the base
    /// holds, and it carries exactly the base's columns.
    fn validate_record(&self, record: &PointRecord) -> Result<(), EditError> {
        if record.observations.is_empty() {
            return Err(EditError::NoObservations);
        }
        let image_count = self.base.image_count();
        for (k, obs) in record.observations.iter().enumerate() {
            if obs.image_index as usize >= image_count {
                return Err(EditError::ImageOutOfRange {
                    observation: k,
                    image_index: obs.image_index,
                    image_count,
                });
            }
            check_column(
                "feature_indexes",
                self.has_feature_indexes(),
                obs.feature_index.is_some(),
            )?;
            check_column(
                "keypoints_xy",
                self.has_keypoints(),
                obs.keypoint_xy.is_some(),
            )?;
            check_column(
                "observation_confidence",
                self.has_observation_confidence(),
                obs.confidence.is_some(),
            )?;
        }
        check_column(
            "patch_u_halfvec_xyz",
            self.has_patch_frames(),
            record.patch_u_halfvec.is_some(),
        )?;
        check_column(
            "patch_v_halfvec_xyz",
            self.has_patch_frames(),
            record.patch_v_halfvec.is_some(),
        )?;
        check_column(
            "patch_bitmaps_y_x_rgba",
            self.has_patch_bitmaps(),
            record.patch_bitmap.is_some(),
        )?;
        check_column(
            "normal_confidence",
            self.has_normal_confidence(),
            record.normal_confidence.is_some(),
        )?;
        check_column(
            "point_constraints",
            self.has_point_constraints(),
            record.constraint.is_some(),
        )?;
        check_column(
            "reference_observations",
            self.has_reference_observations(),
            record.reference_observation.is_some(),
        )?;
        if let Some(r) = record.reference_observation {
            if r != NO_REFERENCE_OBSERVATION
                && !(r >= 0 && (r as usize) < record.observations.len())
            {
                return Err(EditError::ReferenceObservationOutOfRange {
                    reference: r,
                    observations: record.observations.len(),
                });
            }
        }
        if let (Some(bitmap), Some(base)) = (
            &record.patch_bitmap,
            &self.base.point_set.patch_bitmaps_y_x_rgba,
        ) {
            let s = base.shape();
            let expected = (s[1], s[2], s[3]);
            let g = bitmap.shape();
            let got = (g[0], g[1], g[2]);
            if got != expected {
                return Err(EditError::PatchBitmapShape { got, expected });
            }
        }
        if let Some((k, _, r)) = record.constraint {
            if !matches!(
                k,
                POINT_CONSTRAINT_FREE | POINT_CONSTRAINT_RANGED | POINT_CONSTRAINT_HELD
            ) {
                return Err(EditError::UnknownConstraint(k));
            }
            if r != NO_REFERENCE_IMAGE && r as usize >= image_count {
                return Err(EditError::ConstraintImageOutOfRange(r));
            }
        }
        Ok(())
    }

    // ── Hashes ───────────────────────────────────────────────────────

    /// The base's content hashes: the ones it arrived with, or the ones a write
    /// of it would store.
    ///
    /// **A base that came from a file is taken at its word.** The file carries
    /// its own hashes and `read` keeps them, so recomputing them would spend a
    /// serialisation of the whole value to arrive at a number already in hand.
    /// On a large reconstruction that is seconds, and it is seconds spent to
    /// learn nothing: the hashes name the file this value was read out of,
    /// which is exactly what a caller asking for a base's identity means.
    ///
    /// Re-deriving them would also answer a subtly different question. It says
    /// what a *save of this value* would store, which is not the same thing
    /// after a convention upgrade, and it is the file on disk that a point id
    /// has to name.
    ///
    /// Checking the stored hashes against the bytes is verification, and
    /// verification is [`sfmtool_sfmr_format::verify_sfmr`]'s job, asked for
    /// deliberately. It is not something a value does to itself every time it
    /// is asked who it is.
    ///
    /// **A base with no stored hash is computed and kept.** That is a value
    /// materialised from an edit, which `EditedReconstruction::materialise`
    /// clears the hashes on precisely because they would name a file whose
    /// content it no longer is.
    pub fn base_content_hash(&self) -> Result<&ContentHash, SfmrError> {
        if let Some(h) = self.base_hash.get() {
            return Ok(h);
        }
        if !self.base.content_hash.content_xxh128.is_empty() {
            let _ = self.base_hash.set(self.base.content_hash.clone());
            return Ok(self.base_hash.get().expect("just set"));
        }
        let hash = self.base.content_xxh128()?;
        // A racing computation produces the same bytes, so whichever lands
        // first is the answer and the other is dropped.
        let _ = self.base_hash.set(hash);
        Ok(self.base_hash.get().expect("just set"))
    }

    /// The content hash of a point edit that creates `records` on this version's
    /// base.
    ///
    /// It is a function of what was added and where, and of nothing else: the
    /// base's `content_xxh128`, then each record's observations as the content
    /// hash of the image that sees it and the pixel it sees it at, then the
    /// point that pixel set triangulates to. So two different additions on one
    /// base hash differently, and the same addition made twice -- in two
    /// sessions, or after an undo -- hashes the same, which is right, since it
    /// is the same point.
    ///
    /// The image's content hash is the base's own per-image hash for that image
    /// (`image_file_hashes` for `embedded_patches`, `sift_content_hashes` for
    /// `sift_files`). The pixel is the inline keypoint when the base carries
    /// one and the `.sift` feature index otherwise, which is what identifies the
    /// sighting in each mode.
    pub fn point_edit_hash(&self, records: &[PointRecord]) -> Result<String, SfmrError> {
        let base = &self.base_content_hash()?.content_xxh128;
        let image_hashes = self
            .base
            .image_file_hashes()
            .or_else(|| self.base.sift_content_hashes());
        let mut h = Xxh3::new();
        h.update(base.as_bytes());
        for record in records {
            h.update(&(record.observations.len() as u64).to_be_bytes());
            for obs in &record.observations {
                match image_hashes {
                    // No per-image hash column at all: the image's name is what
                    // this reconstruction says the image is.
                    None => h.update(
                        self.base.image_table.images[obs.image_index as usize]
                            .name
                            .as_bytes(),
                    ),
                    Some(hashes) => h.update(&hashes[obs.image_index as usize]),
                }
                match obs.keypoint_xy {
                    Some([x, y]) => {
                        h.update(&x.to_be_bytes());
                        h.update(&y.to_be_bytes());
                    }
                    None => h.update(&obs.feature_index.unwrap_or(u32::MAX).to_be_bytes()),
                }
            }
            let p = &record.point;
            for v in [p.position.x, p.position.y, p.position.z, p.w] {
                h.update(&v.to_be_bytes());
            }
        }
        Ok(format_hash(h.digest128()))
    }

    // ── Materialisation ──────────────────────────────────────────────

    /// The plain reconstruction this version is, with every point in its place,
    /// and the map from this version's indexes to that value's.
    ///
    /// A modified point goes back to the base index it replaced, carrying its
    /// new record; a deleted point's slot closes up and the points after it
    /// shift down; a point that is new to this base is appended after the last
    /// base point, in the order the additions were made. The tracks come out of
    /// one merge pass over the base's already-sorted runs, never a re-sort, and
    /// the derived indexes are rebuilt at the end.
    ///
    /// Deterministic (the same value materialises to the same value) and
    /// idempotent (materialising the result, with an empty overlay, changes
    /// nothing and gives the identity map).
    pub fn materialize(&self) -> (SfmrReconstruction, RowMap) {
        let base_count = self.base.point_count();
        // Which record occupies each base slot, and which additions are new.
        let mut occupant: Vec<Option<Source>> = (0..base_count)
            .map(|b| {
                let b = b as u32;
                (!self.deleted_points.contains(&b)).then_some(Source::Base(b))
            })
            .collect();
        let mut appended: Vec<u32> = Vec::new();
        for (a, replaces) in self.replaces.iter().enumerate() {
            let edited = base_count as u32 + a as u32;
            if self.deleted_points.contains(&edited) {
                continue;
            }
            match replaces {
                Some(b) => occupant[*b as usize] = Some(Source::Added(a as u32)),
                None => appended.push(a as u32),
            }
        }

        // The emission order is the materialised order, and the row map falls
        // out of it.
        let mut sources: Vec<Source> = Vec::with_capacity(self.point_count());
        let mut holes: Vec<u32> = Vec::new();
        let mut replaced: Vec<u32> = Vec::new();
        let mut moved: Vec<(u32, u32)> = Vec::new();
        for (b, slot) in occupant.iter().enumerate() {
            match slot {
                None => holes.push(b as u32),
                Some(src) => {
                    if let Source::Added(a) = src {
                        replaced.push(b as u32);
                        moved.push((base_count as u32 + a, sources.len() as u32));
                    }
                    sources.push(*src);
                }
            }
        }
        for a in appended {
            moved.push((base_count as u32 + a, sources.len() as u32));
            sources.push(Source::Added(a));
        }

        let point_set = self.build_point_set(&sources);
        let mut metadata = self.base.metadata.clone();
        metadata.point_count = point_set.points.len() as u32;
        metadata.observation_count = point_set.tracks.len() as u32;
        metadata.image_count = self.base.image_count() as u32;
        metadata.camera_count = self.base.camera_count() as u32;
        metadata.infinity_point_count = point_set.infinity_point_count as u32;

        let recon = SfmrReconstruction {
            workspace_dir: self.base.workspace_dir.clone(),
            metadata,
            // The hashes on a value describe the file it came from, and this
            // value is not that file: it holds the base's points only where the
            // edits left them alone. Carrying them over would name a file whose
            // content this is not, so they are cleared to the state a
            // never-written reconstruction carries.
            // `SfmrReconstruction::content_xxh128` is the live answer.
            content_hash: ContentHash::default(),
            // No overlay edit touches an image, so the whole table is the
            // base's, thumbnails shared rather than copied.
            image_table: ImageTable {
                thumbnails_y_x_rgb: self.base.image_table.thumbnails_y_x_rgb.clone(),
                ..self.base.image_table.clone()
            },
            point_set,
        };
        // `moved` comes out in emission order, which is base-slot order for the
        // modified points and addition order for the appended ones; both
        // lookups binary-search, so each gets its own sorting.
        let mut by_new = moved.clone();
        by_new.sort_unstable_by_key(|&(_, n)| n);
        moved.sort_unstable_by_key(|&(e, _)| e);
        (
            recon,
            RowMap::materialized(base_count as u32, holes, replaced, moved, by_new),
        )
    }

    /// The materialised point set: every column selected in `sources` order.
    fn build_point_set(&self, sources: &[Source]) -> PointSet {
        let base = &self.base.point_set;
        let n = sources.len();

        let mut points = Vec::with_capacity(n);
        let mut observation_counts = Vec::with_capacity(n);
        let mut tracks = Vec::with_capacity(base.tracks.len());
        let mut obs_rows: Vec<(bool, usize)> = Vec::with_capacity(base.tracks.len());
        for (new_index, src) in sources.iter().enumerate() {
            let (set, local, is_base) = match src {
                Source::Base(b) => (base, *b as usize, true),
                Source::Added(a) => (&self.added, *a as usize, false),
            };
            points.push(set.points[local].clone());
            let start = set.observation_offsets[local];
            let end = set.observation_offsets[local + 1];
            observation_counts.push((end - start) as u32);
            for row in start..end {
                tracks.push(TrackObservation {
                    image_index: set.tracks[row].image_index,
                    point_index: new_index as u32,
                });
                obs_rows.push((is_base, row));
            }
        }

        let point_rows: Vec<(bool, usize)> = sources
            .iter()
            .map(|s| match s {
                Source::Base(b) => (true, *b as usize),
                Source::Added(a) => (false, *a as usize),
            })
            .collect();

        let observations = self.merge_observation_source(&obs_rows);
        let observation_confidence = base.observation_confidence.as_ref().map(|c| {
            obs_rows
                .iter()
                .map(|&(is_base, row)| {
                    if is_base {
                        c[row]
                    } else {
                        self.added
                            .observation_confidence
                            .as_ref()
                            .expect("column parity")[row]
                    }
                })
                .collect()
        });

        // Present where the base carries readings or an added record brought
        // a measured one; a base row the base has none for is not measured.
        // Present where the base or the addition set carries readings; the
        // addition set's stand under the base's options wherever the base has
        // any (`push_record`).
        let added_readings = self.added.observation_readings.as_ref();
        let observation_readings = base
            .observation_readings
            .as_ref()
            .or(added_readings)
            .map(|r| r.options)
            .map(|options| ObservationReadings {
                rows: obs_rows
                    .iter()
                    .map(|&(is_base, row)| {
                        let from = if is_base {
                            base.observation_readings.as_ref()
                        } else {
                            added_readings
                        };
                        from.map_or(ObservationReading::NOT_MEASURED, |r| r.rows[row])
                    })
                    .collect(),
                options,
            });

        // A point's observations are copied as one run in their stored order,
        // so its index within the run is unchanged.
        let reference_observations = base.reference_observations.as_ref().map(|r| {
            point_rows
                .iter()
                .map(|&(is_base, i)| {
                    if is_base {
                        r[i]
                    } else {
                        self.added
                            .reference_observations
                            .as_ref()
                            .expect("column parity")[i]
                    }
                })
                .collect()
        });
        // An added point carries the mark its record did: a point rewritten
        // through `replace_point` keeps it, and a record an edit built fresh
        // is unmarked.
        let display_only_references = base.display_only_references.as_ref().map(|marks| {
            let added = self
                .added
                .display_only_references
                .as_ref()
                .expect("column parity");
            point_rows
                .iter()
                .map(|&(is_base, i)| if is_base { marks[i] } else { added[i] })
                .collect()
        });

        PointSet {
            points,
            observation_counts,
            tracks,
            observations,
            observation_confidence,
            observation_readings,
            reference_observations,
            display_only_references,
            patch_u_halfvec_xyz: self.merge_halfvec(&base.patch_u_halfvec_xyz, &point_rows, true),
            patch_v_halfvec_xyz: self.merge_halfvec(&base.patch_v_halfvec_xyz, &point_rows, false),
            patch_bitmaps_y_x_rgba: self.merge_bitmaps(&point_rows),
            patch_bitmaps_for_display: base.patch_bitmaps_for_display,
            has_normals: base.has_normals,
            normal_confidence: base.normal_confidence.as_ref().map(|c| {
                point_rows
                    .iter()
                    .map(|&(is_base, i)| {
                        if is_base {
                            c[i]
                        } else {
                            self.added
                                .normal_confidence
                                .as_ref()
                                .expect("column parity")[i]
                        }
                    })
                    .collect()
            }),
            point_constraints: base.point_constraints.as_ref().map(|c| {
                let added = self
                    .added
                    .point_constraints
                    .as_ref()
                    .expect("column parity");
                let pick = |is_base: bool| if is_base { c } else { added };
                PointConstraintColumns {
                    point_constraints: point_rows
                        .iter()
                        .map(|&(b, i)| pick(b).point_constraints[i])
                        .collect(),
                    constraint_distances: point_rows
                        .iter()
                        .map(|&(b, i)| pick(b).constraint_distances[i])
                        .collect(),
                    constraint_reference_images: point_rows
                        .iter()
                        .map(|&(b, i)| pick(b).constraint_reference_images[i])
                        .collect(),
                }
            }),
            // Restored below from everything above.
            observation_offsets: Vec::new(),
            image_feature_to_point: Vec::new(),
            max_track_feature_index: Vec::new(),
            infinity_point_count: 0,
        }
        .with_derived_fields(self.base.image_count())
    }

    /// The merged observation source: the per-observation columns selected in
    /// `obs_rows` order, the per-image hashes taken from the base unchanged
    /// (no overlay edit touches an image).
    fn merge_observation_source(&self, obs_rows: &[(bool, usize)]) -> ObservationSource {
        let base = &self.base.point_set;
        let merge_keypoints = |b: &Array2<f32>, a: &Array2<f32>| {
            let mut out = Array2::<f32>::zeros((obs_rows.len(), 2));
            for (i, &(is_base, row)) in obs_rows.iter().enumerate() {
                let src = if is_base { b } else { a };
                out[[i, 0]] = src[[row, 0]];
                out[[i, 1]] = src[[row, 1]];
            }
            out
        };
        match (&base.observations, &self.added.observations) {
            (
                ObservationSource::SiftFiles {
                    feature_indexes,
                    keypoints_xy,
                    feature_tool_hashes,
                    sift_content_hashes,
                },
                ObservationSource::SiftFiles {
                    feature_indexes: added_features,
                    keypoints_xy: added_keypoints,
                    ..
                },
            ) => ObservationSource::SiftFiles {
                feature_indexes: obs_rows
                    .iter()
                    .map(|&(is_base, row)| {
                        if is_base {
                            feature_indexes[row]
                        } else {
                            added_features[row]
                        }
                    })
                    .collect(),
                keypoints_xy: keypoints_xy
                    .as_ref()
                    .map(|b| merge_keypoints(b, added_keypoints.as_ref().expect("column parity"))),
                feature_tool_hashes: feature_tool_hashes.clone(),
                sift_content_hashes: sift_content_hashes.clone(),
            },
            (
                ObservationSource::EmbeddedPatches {
                    keypoints_xy,
                    image_file_hashes,
                },
                ObservationSource::EmbeddedPatches {
                    keypoints_xy: added_keypoints,
                    ..
                },
            ) => ObservationSource::EmbeddedPatches {
                keypoints_xy: merge_keypoints(keypoints_xy, added_keypoints),
                image_file_hashes: image_file_hashes.clone(),
            },
            // `new` builds the addition set from the base's variant and nothing
            // can change either afterwards.
            _ => unreachable!("the addition set is always the base's mode"),
        }
    }

    /// The merged `(P, 3)` half-vector column, `u` when `is_u` and `v`
    /// otherwise.
    fn merge_halfvec(
        &self,
        base: &Option<Array2<f32>>,
        point_rows: &[(bool, usize)],
        is_u: bool,
    ) -> Option<Array2<f32>> {
        let base = base.as_ref()?;
        let added = if is_u {
            self.added.patch_u_halfvec_xyz.as_ref()
        } else {
            self.added.patch_v_halfvec_xyz.as_ref()
        }
        .expect("column parity");
        let mut out = Array2::<f32>::zeros((point_rows.len(), 3));
        for (i, &(is_base, row)) in point_rows.iter().enumerate() {
            let src = if is_base { base } else { added };
            for c in 0..3 {
                out[[i, c]] = src[[row, c]];
            }
        }
        Some(out)
    }

    /// The merged patch-bitmap column.
    ///
    /// The base's array is shared outright when the materialisation neither
    /// moves a row nor replaces one, which is the common case of a version that
    /// edited no patch: the bitmaps are the heaviest column in the value and a
    /// copy of them costs more than everything else here put together.
    fn merge_bitmaps(&self, point_rows: &[(bool, usize)]) -> Option<Arc<Array4<u8>>> {
        let base = self.base.point_set.patch_bitmaps_y_x_rgba.as_ref()?;
        let added = self
            .added
            .patch_bitmaps_y_x_rgba
            .as_ref()
            .expect("column parity");
        let in_place = point_rows.len() == base.shape()[0]
            && point_rows
                .iter()
                .enumerate()
                .all(|(i, &(is_base, row))| !is_base || row == i);
        if in_place {
            let changed: Vec<usize> = point_rows
                .iter()
                .enumerate()
                .filter(|&(i, &(is_base, row))| {
                    !is_base && added.index_axis(Axis(0), row) != base.index_axis(Axis(0), i)
                })
                .map(|(i, _)| i)
                .collect();
            if changed.is_empty() {
                return Some(Arc::clone(base));
            }
            let mut out = (**base).clone();
            for i in changed {
                let (_, row) = point_rows[i];
                out.index_axis_mut(Axis(0), i)
                    .assign(&added.index_axis(Axis(0), row));
            }
            return Some(Arc::new(out));
        }
        let s = base.shape();
        let mut out = Array4::<u8>::zeros((point_rows.len(), s[1], s[2], s[3]));
        for (i, &(is_base, row)) in point_rows.iter().enumerate() {
            let src = if is_base { base } else { added };
            out.index_axis_mut(Axis(0), i)
                .assign(&src.index_axis(Axis(0), row));
        }
        Some(Arc::new(out))
    }
}

/// Where a materialised row comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Source {
    /// The base's point at this index.
    Base(u32),
    /// The addition set's point at this local index.
    Added(u32),
}

/// What one edit did to point indexes.
///
/// It is what a caller holding an index across an edit follows: a selection, a
/// copied id, a row of a panel's prepared state. Every case is stored as what
/// it is rather than as a pair of dense arrays, so a map is the size of the
/// edit that made it: a list of indexes, or a row map, or the few steps one
/// bulk edit took.
///
/// A map is **per step, not per index space**. [`PointMap::Removed`] is what a
/// point edit did (indexes are stable, so it says only which ones stopped
/// resolving); a [`PointMap::Chain`] of two [`PointMap::Rows`] is what a bulk
/// edit's materialisation and renumbering did. Both answer the same question,
/// [`PointMap::forward`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PointMap {
    /// A point edit. Indexes are stable across it, so the map is only the
    /// indexes that stopped resolving, ascending.
    Removed(Vec<u32>),
    /// A point edit that **modified** points: each pair is the index a point
    /// held before the step and the one it took after it.
    ///
    /// Delete-and-re-add gives a modified point a new index while it stays the
    /// same point, so this is what carries a selection, a copied id and a
    /// panel's prepared state across the edit. Every index not named is
    /// unchanged, which is what makes the map the size of the edit.
    Replaced(Vec<(u32, u32)>),
    /// A point edit that **created** points, naming the indexes they took.
    ///
    /// Every index the version already held is unchanged, so the forward
    /// direction is the identity; the created ones are what the inverse has no
    /// answer for, which is how an undo drops a selection that sits on one
    /// rather than carrying it back to an index that held nothing.
    Created(Vec<u32>),
    /// A whole-value edit's row map: a materialisation's, or the one
    /// [`RowMap::by_scan`] reads off a bulk edit's input and output.
    Rows(RowMap),
    /// The steps one edit took, applied in order.
    Chain(Vec<PointMap>),
}

impl PointMap {
    /// Where the index `before` this step lands after it, or `None` when the
    /// point it named is gone.
    pub fn forward(&self, before: u32) -> Option<u32> {
        match self {
            PointMap::Removed(removed) => is_live(removed, before).then_some(before),
            PointMap::Replaced(moves) => Some(
                moves
                    .iter()
                    .find(|&&(from, _)| from == before)
                    .map_or(before, |&(_, to)| to),
            ),
            PointMap::Created(_) => Some(before),
            PointMap::Rows(map) => map.forward(before),
            PointMap::Chain(steps) => steps
                .iter()
                .try_fold(before, |index, step| step.forward(index)),
        }
    }

    /// Where the index `after` this step came from, or `None` when this step
    /// created the point it names.
    pub fn inverse(&self, after: u32) -> Option<u32> {
        match self {
            PointMap::Removed(removed) => is_live(removed, after).then_some(after),
            PointMap::Replaced(moves) => Some(
                moves
                    .iter()
                    .find(|&&(_, to)| to == after)
                    .map_or(after, |&(from, _)| from),
            ),
            PointMap::Created(created) => (!created.contains(&after)).then_some(after),
            PointMap::Rows(map) => map.inverse(after),
            PointMap::Chain(steps) => steps
                .iter()
                .rev()
                .try_fold(after, |index, step| step.inverse(index)),
        }
    }
}

/// Whether `index` survived a step that removed `removed` (ascending).
fn is_live(removed: &[u32], index: u32) -> bool {
    removed.binary_search(&index).is_err()
}

/// Whether a record's column presence matches the base's.
fn check_column(column: &'static str, base_has: bool, record_has: bool) -> Result<(), EditError> {
    if base_has == record_has {
        Ok(())
    } else {
        Err(EditError::ColumnMismatch { column, base_has })
    }
}

/// Append each observation's keypoint to a `(M, 2)` column.
fn append_keypoints(kp: &mut Array2<f32>, observations: &[RecordObservation]) {
    for obs in observations {
        let [x, y] = obs.keypoint_xy.expect("validated present");
        kp.push_row(ndarray::ArrayView1::from(&[x, y]))
            .expect("a two-column row always fits a two-column array");
    }
}

/// Append one row to a `(P, 3)` half-vector column.
fn push_halfvec(arr: &mut Array2<f32>, v: [f32; 3]) {
    arr.push_row(ndarray::ArrayView1::from(&v))
        .expect("a three-column row always fits a three-column array");
}

/// An empty point set carrying exactly `base`'s columns, over `image_count`
/// images.
///
/// The per-image hash vectors are the base's: they are measured per image, the
/// overlay changes no image, and a point set whose hash vectors did not match
/// the image count would fail its own validation.
fn empty_like(base: &PointSet, image_count: usize) -> PointSet {
    let observations = match &base.observations {
        ObservationSource::SiftFiles {
            keypoints_xy,
            feature_tool_hashes,
            sift_content_hashes,
            ..
        } => ObservationSource::SiftFiles {
            feature_indexes: Vec::new(),
            keypoints_xy: keypoints_xy.as_ref().map(|_| Array2::<f32>::zeros((0, 2))),
            feature_tool_hashes: feature_tool_hashes.clone(),
            sift_content_hashes: sift_content_hashes.clone(),
        },
        ObservationSource::EmbeddedPatches {
            image_file_hashes, ..
        } => ObservationSource::EmbeddedPatches {
            keypoints_xy: Array2::<f32>::zeros((0, 2)),
            image_file_hashes: image_file_hashes.clone(),
        },
    };
    PointSet {
        points: Vec::new(),
        tracks: Vec::new(),
        observation_counts: Vec::new(),
        observations,
        patch_u_halfvec_xyz: base
            .patch_u_halfvec_xyz
            .as_ref()
            .map(|_| Array2::<f32>::zeros((0, 3))),
        patch_v_halfvec_xyz: base
            .patch_v_halfvec_xyz
            .as_ref()
            .map(|_| Array2::<f32>::zeros((0, 3))),
        patch_bitmaps_y_x_rgba: base.patch_bitmaps_y_x_rgba.as_ref().map(|b| {
            let s = b.shape();
            Arc::new(Array4::<u8>::zeros((0, s[1], s[2], s[3])))
        }),
        patch_bitmaps_for_display: base.patch_bitmaps_for_display,
        has_normals: base.has_normals,
        normal_confidence: base.normal_confidence.as_ref().map(|_| Vec::new()),
        point_constraints: base
            .point_constraints
            .as_ref()
            .map(|_| PointConstraintColumns::all_free(0)),
        observation_confidence: base.observation_confidence.as_ref().map(|_| Vec::new()),
        // Present with the base's options where it has readings; otherwise
        // created by the first record that brings readings, under its options
        // (`add_record`), so an edit can bring readings to a base without them.
        observation_readings: base
            .observation_readings
            .as_ref()
            .map(|r| ObservationReadings::not_measured(0, r.options)),
        reference_observations: base.reference_observations.as_ref().map(|_| Vec::new()),
        // Present with the base's marks, so a rewritten point keeps its own.
        display_only_references: base.display_only_references.as_ref().map(|_| Vec::new()),
        observation_offsets: vec![0],
        image_feature_to_point: vec![HashMap::new(); image_count],
        max_track_feature_index: vec![0; image_count],
        infinity_point_count: 0,
    }
}

impl PointSet {
    /// This set with its derived indexes restored, as an expression.
    ///
    /// The materialisation builds its columns and then needs the indexes; a
    /// method that consumes and returns keeps that a single expression rather
    /// than a `let mut` whose intermediate value is invalid.
    fn with_derived_fields(mut self, image_count: usize) -> Self {
        self.rebuild_derived_fields(image_count);
        self
    }
}

#[cfg(test)]
mod tests;
