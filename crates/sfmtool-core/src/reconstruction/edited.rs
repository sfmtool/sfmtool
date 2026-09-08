// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! An [`EditedReconstruction`]: a shared immutable base plus the point edits
//! made on top of it, and the materialisation that turns the pair back into a
//! plain [`SfmrReconstruction`].
//!
//! `specs/core/reconstruction/edited-reconstruction.md` is the design; this
//! module is the whole of it, apart from the `.sfmr` writer's hashing, which
//! `sfmr_format::content_hash_of` supplies.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, OnceLock};

use ndarray::{Array2, Array3, Array4, ArrayView3, Axis};
use xxhash_rust::xxh3::Xxh3;

use sfmr_format::{
    ContentHash, SfmrError, NO_REFERENCE_IMAGE, POINT_CONSTRAINT_FREE, POINT_CONSTRAINT_HELD,
    POINT_CONSTRAINT_RANGED,
};
use sfmtool_archive_io::format_hash;

use super::data::{
    ImageTable, ObservationSource, Point3D, PointConstraintColumns, PointSet, SfmrReconstruction,
    TrackObservation,
};

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
}

/// A point read through the overlay: a borrow into whichever point set holds
/// it, with the columns addressed by that set's own local index.
///
/// Handing back a view rather than a record is what lets the Point Track Detail
/// panel and the track rays read an edited reconstruction at the cost of a
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
}

impl Clone for EditedReconstruction {
    fn clone(&self) -> Self {
        Self {
            base: Arc::clone(&self.base),
            deleted_points: self.deleted_points.clone(),
            added: self.added.clone(),
            replaces: self.replaces.clone(),
            base_hash: self.base_hash.clone(),
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
        && a.patch_u_halfvec_xyz == b.patch_u_halfvec_xyz
        && a.patch_v_halfvec_xyz == b.patch_v_halfvec_xyz
        && a.patch_bitmaps_y_x_rgba.as_deref() == b.patch_bitmaps_y_x_rgba.as_deref()
        && a.normal_confidence == b.normal_confidence
        && constraints_agree(&a.point_constraints, &b.point_constraints)
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
                    .all(|(x, y)| x == y || (x.is_nan() && y.is_nan()))
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
        }
    }

    /// How many points this version holds: the base's, less the deletions, plus
    /// the live additions. O(1).
    pub fn point_count(&self) -> usize {
        self.base.point_count() + self.added.point_count() - self.deleted_points.len()
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

    /// The base's content hashes, computed once and kept.
    ///
    /// Computed from the base value rather than from a file, so a base that was
    /// never written has the hash a write of it would store.
    pub fn base_content_hash(&self) -> Result<&ContentHash, SfmrError> {
        if let Some(h) = self.base_hash.get() {
            return Ok(h);
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
                thumbnails_y_x_rgb: Arc::clone(&self.base.image_table.thumbnails_y_x_rgb),
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
            RowMap {
                base_count: base_count as u32,
                holes,
                replaced,
                by_edited: moved,
                by_new,
                // A materialisation knows what it did, so it needs no scan.
                scan: None,
            },
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

        PointSet {
            points,
            observation_counts,
            tracks,
            observations,
            observation_confidence,
            patch_u_halfvec_xyz: self.merge_halfvec(&base.patch_u_halfvec_xyz, &point_rows, true),
            patch_v_halfvec_xyz: self.merge_halfvec(&base.patch_v_halfvec_xyz, &point_rows, false),
            patch_bitmaps_y_x_rgba: self.merge_bitmaps(&point_rows),
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

/// The map a materialisation produces, from this version's indexes to the
/// materialised value's, and back.
///
/// It is stored as what it is rather than as two arrays: base points shift down
/// by the number of slots that emptied before them, which is a prefix count
/// over a sorted list of holes, and only the additions -- a handful -- need an
/// entry each. Both directions are a binary search.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RowMap {
    /// The base's point count, which is where the addition indexes start.
    base_count: u32,
    /// The base slots the materialisation emptied, ascending. A base point
    /// shifts down by how many of these sit below it.
    holes: Vec<u32>,
    /// The base slots a modified point took over, ascending. The slot survives
    /// -- it is the modification's row -- but the base index that named it does
    /// not, because the point that lives there is now the addition.
    replaced: Vec<u32>,
    /// `(edited index, new index)` for every live addition, ascending by
    /// edited.
    by_edited: Vec<(u32, u32)>,
    /// The same pairs, ascending by new index.
    by_new: Vec<(u32, u32)>,
    /// A scan's answer, when the map came from [`RowMap::by_scan`]; `None` for
    /// a materialisation's, which the four fields above describe exactly.
    ///
    /// Dense rather than a list of holes because a scan has to express rows the
    /// edit **created**, interleaved anywhere: a survivor's new index is then
    /// not its old one less the holes below it, and no amount of hole counting
    /// recovers it. Two arrays, so both directions are a lookup.
    scan: Option<ScanRows>,
}

/// The two directions of a scanned map, one entry per point of each side.
#[derive(Debug, Clone, PartialEq, Eq)]
struct ScanRows {
    /// Per point of the before value: where it landed, or `None` if it was
    /// deleted.
    forward: Vec<Option<u32>>,
    /// Per point of the after value: where it came from, or `None` if the edit
    /// created it.
    inverse: Vec<Option<u32>>,
}

impl RowMap {
    /// The map a whole-value edit performed, read off its input and its output.
    ///
    /// A bulk edit produces a new reconstruction rather than a description of
    /// what it did, so a caller holding a point index across one -- a
    /// selection, a copied id, a constraint row -- has nothing to follow it
    /// through. Rather than every such edit growing a second spelling that
    /// returns a map, this derives the map once, from the two values, by
    /// walking their point lists side by side.
    ///
    /// `image_map` says where each image of `before` went: one entry per image
    /// of `before`, holding its index in `after` or `None` for an image the
    /// edit dropped. `None` for the whole argument means the image table did
    /// not move, which is every bulk edit but an image subset.
    ///
    /// ## The invariant it rests on
    ///
    /// **A bulk edit never reorders the points that survive it.** Every one of
    /// them is a selection or a per-point rewrite over the existing list, in
    /// its existing order: the image subset filters and renumbers, the point
    /// mask filters, a similarity transform and a bundle adjustment rewrite
    /// every point in place, and a materialisation puts each point back in the
    /// slot it came from. A bulk edit may **drop** points (an image subset
    /// orphans them, a mask removes them) and may **create** them (a densify,
    /// a fresh triangulation pass, a materialisation's appended additions), and
    /// the scan reports both; what it cannot follow is a list whose survivors
    /// changed places.
    ///
    /// ## What counts as the same point
    ///
    /// Two heads walk the two point lists. The after point at the write head
    /// matches the before point at the read head when its track is a non-empty
    /// subsequence of that before point's track, once the before track is put
    /// through `image_map` and the observations of dropped images are removed.
    /// On a match both heads advance. Otherwise the scan looks ahead through
    /// the remaining before points for the first one the after point matches:
    /// found, everything skipped over is a deletion; not found, the after point
    /// is one the edit **created**, and only the write head advances. Whatever
    /// the write head never accounts for by the end is a deletion too.
    ///
    /// A created point has no old index: [`RowMap::inverse`] answers `None` for
    /// it, exactly as it does for a materialisation's appended additions.
    ///
    /// An observation is compared on the image it is in, plus its **feature
    /// index** when the reconstruction carries one (`sift_files`). The inline
    /// keypoint of an `embedded_patches` reconstruction is deliberately not
    /// compared: a refinement moves a keypoint by a fraction of a pixel without
    /// making it a different sighting, so matching on it would report a point
    /// the adjustment merely improved as one the edit deleted. So an
    /// `embedded_patches` point is identified by the images that see it and
    /// nothing finer, and two adjacent points seen by the same images in the
    /// same order are indistinguishable to the scan. That ambiguity is reachable
    /// only when one of the two is deleted and the other kept, and the answer it
    /// gives then is one of the two rows, both of which hold a point the same
    /// images saw. A point left with no observations at all is likewise
    /// unidentifiable, and is reported as created rather than carried over.
    ///
    /// ## Cost
    ///
    /// One pass over both point lists when the edit created nothing, or
    /// appended what it created, which is the case every bulk edit here is.
    /// Each created point that is *interleaved* costs a look-ahead to the end
    /// of the before list, so a value whose new points are scattered through it
    /// in quantity degrades toward quadratic.
    pub fn by_scan(
        before: &SfmrReconstruction,
        after: &SfmrReconstruction,
        image_map: Option<&[Option<u32>]>,
    ) -> Result<Self, EditError> {
        if let Some(map) = image_map {
            if map.len() != before.image_count() {
                return Err(EditError::ScanImageMap {
                    got: map.len(),
                    expected: before.image_count(),
                });
            }
        }
        // A feature index identifies a sighting when the value carries one; an
        // `embedded_patches` value is matched on the image alone (above).
        let before_features = before.point_set.feature_indexes();
        let after_features = after.point_set.feature_indexes();
        let keyed = before_features.is_some() && after_features.is_some();

        // The before track put through the image map, per point, built lazily
        // as the read head advances rather than all at once.
        let mapped = |point: usize| -> Vec<(u32, Option<u32>)> {
            let offset = before.point_set.observation_offsets[point];
            before
                .point_set
                .observations_for_point(point)
                .iter()
                .enumerate()
                .filter_map(|(k, observation)| {
                    let image = match image_map {
                        Some(map) => map[observation.image_index as usize]?,
                        None => observation.image_index,
                    };
                    Some((
                        image,
                        keyed.then(|| before_features.expect("keyed")[offset + k]),
                    ))
                })
                .collect()
        };

        let mut forward: Vec<Option<u32>> = vec![None; before.point_count()];
        let mut inverse: Vec<Option<u32>> = vec![None; after.point_count()];
        let mut read = 0usize;
        // The index is the subject here: it addresses three parallel things
        // (the offsets, the track and the row being written) and is the value
        // recorded in the map.
        #[allow(clippy::needless_range_loop)]
        for write in 0..after.point_count() {
            let offset = after.point_set.observation_offsets[write];
            let wanted: Vec<(u32, Option<u32>)> = after
                .point_set
                .observations_for_point(write)
                .iter()
                .enumerate()
                .map(|(k, observation)| {
                    (
                        observation.image_index,
                        keyed.then(|| after_features.expect("keyed")[offset + k]),
                    )
                })
                .collect();
            // An empty track identifies nothing, so it never matches: such a
            // point is reported as created. A bare subsequence test would
            // instead match it against whatever the read head happened to be
            // on, since nothing is a subsequence of everything.
            if wanted.is_empty() {
                continue;
            }
            // The read head first, then a look-ahead. Nothing before the head
            // is revisited: a survivor never moves ahead of one that precedes
            // it, which is the invariant above.
            let found = (read..before.point_count()).find(|&k| is_subsequence(&wanted, &mapped(k)));
            if let Some(k) = found {
                forward[k] = Some(write as u32);
                inverse[write] = Some(k as u32);
                read = k + 1;
            }
            // Otherwise `inverse[write]` stays `None`: the edit created it.
        }
        Ok(RowMap {
            base_count: before.point_count() as u32,
            holes: Vec::new(),
            replaced: Vec::new(),
            by_edited: Vec::new(),
            by_new: Vec::new(),
            scan: Some(ScanRows { forward, inverse }),
        })
    }

    /// Where `edited` landed, or `None` when that index named no live point.
    pub fn forward(&self, edited: u32) -> Option<u32> {
        if let Some(scan) = &self.scan {
            return scan.forward.get(edited as usize).copied().flatten();
        }
        if edited >= self.base_count {
            return self
                .by_edited
                .binary_search_by_key(&edited, |&(e, _)| e)
                .ok()
                .map(|k| self.by_edited[k].1);
        }
        if self.holes.binary_search(&edited).is_ok() || self.replaced.binary_search(&edited).is_ok()
        {
            return None;
        }
        Some(edited - self.holes_below(edited))
    }

    /// Which edited index landed at `new`, or `None` when the row is one the
    /// edit created and when `new` is past the point count this map came from.
    pub fn inverse(&self, new: u32) -> Option<u32> {
        if let Some(scan) = &self.scan {
            return scan.inverse.get(new as usize).copied().flatten();
        }
        if let Ok(k) = self.by_new.binary_search_by_key(&new, |&(_, n)| n) {
            return Some(self.by_new[k].0);
        }
        self.new_to_base_slot(new)
    }

    /// The base slot a materialised row sits in, when it sits in one.
    ///
    /// Inverts `slot - holes_below(slot) == new`. Writing `slot = new + m`,
    /// `f(m) = holes_below(new + m) - m` falls by nought or one per step, so it
    /// is non-increasing and a binary search finds where it crosses zero.
    ///
    /// The crossing is a run, not a point: a hole and the live slot after it
    /// both satisfy the equation, since the hole contributes nothing to the
    /// count below itself. The answer is the **last** `m` of the run, which is
    /// the only one that is not a hole.
    fn new_to_base_slot(&self, new: u32) -> Option<u32> {
        let (mut lo, mut hi) = (0usize, self.holes.len() + 1);
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if self.holes_below(new + mid as u32) >= mid as u32 {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        let slot = new + (lo - 1) as u32;
        (slot < self.base_count).then_some(slot)
    }

    /// How many holes sit below `slot`.
    fn holes_below(&self, slot: u32) -> u32 {
        self.holes.partition_point(|&h| h < slot) as u32
    }

    /// The forward map as a dense array over `0..index_bound`, `None` where the
    /// index named no live point. What a caller crossing the language boundary
    /// wants, and what the GPU buffers consume.
    pub fn forward_dense(&self, index_bound: u32) -> Vec<Option<u32>> {
        (0..index_bound).map(|e| self.forward(e)).collect()
    }

    /// The inverse map as a dense array over the materialised points.
    ///
    /// For a materialisation's map, where every row has a source. A scanned
    /// map ([`RowMap::by_scan`]) can hold rows the edit created, which have
    /// none, so read that one through [`RowMap::inverse`] instead.
    ///
    /// # Panics
    /// If a row in `0..point_count` has no source.
    pub fn inverse_dense(&self, point_count: u32) -> Vec<u32> {
        (0..point_count)
            .map(|n| {
                self.inverse(n)
                    .expect("every materialised row has a source")
            })
            .collect()
    }
}

/// Whether `wanted` appears in `available` in order, allowing gaps.
///
/// Only called with a non-empty `wanted`: an empty one is a subsequence of
/// everything, which would make a point that identifies nothing match whatever
/// the scan's read head happened to be on.
fn is_subsequence<T: PartialEq>(wanted: &[T], available: &[T]) -> bool {
    let mut it = available.iter();
    wanted.iter().all(|w| it.any(|a| a == w))
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
        has_normals: base.has_normals,
        normal_confidence: base.normal_confidence.as_ref().map(|_| Vec::new()),
        point_constraints: base
            .point_constraints
            .as_ref()
            .map(|_| PointConstraintColumns::all_free(0)),
        observation_confidence: base.observation_confidence.as_ref().map(|_| Vec::new()),
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
