// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The track table: named per-point values computed for every live point of a
//! reconstruction, and the query that filters and sorts points by them.
//!
//! The table's rows are tracks, keyed by point index; it is computed on
//! demand and never stored. A [`TrackTableColumn`] names one value per point
//! (its track length, its stored reprojection error, the widest angle between
//! its viewing rays, ...).
//! [`TrackTable::compute`] fills the columns a caller asks for, for every
//! live point of an [`EditedReconstruction`], and [`query_points`] returns the
//! indexes of the points that pass a set of bounds, in a chosen order. The
//! columns are defined here and nowhere else, so every surface that shows one
//! shows the same number. See `specs/core/analysis/track-table.md`.

use rayon::prelude::*;
use std::borrow::Cow;

use crate::progress::{Cancelled, Progress};
use crate::reconstruction::bundle_adjust::is_posed;
use crate::reconstruction::{EditedReconstruction, PointView};
use crate::CameraIntrinsics;

mod observation_quality;
mod query;
mod track_geometry;

#[cfg(test)]
mod tests;

pub use query::{
    query_points, ColumnBounds, SortOrder, TrackTableQueryError, TrackTableQueryOptions,
};
pub use track_geometry::{
    compute_max_pairwise_angle, compute_max_ray_angle_deg, compute_point_diagnostics,
    compute_position_diagnostics, observation_rays,
};

/// One named per-point value. The wire name is the stable identifier.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum TrackTableColumn {
    /// Number of observations in the track.
    TrackLength,
    /// The track's observations beyond one per image: the track length less
    /// the number of distinct images among its observations. Non-zero is a
    /// defect, an image holding several observations of the point, which a
    /// GLOMAP solve can produce.
    RepeatObservations,
    /// The number of the track's observations that see the point's patch at
    /// a viewing angle of `90°` or more, edge on or from behind
    /// (back-facing); no value for a point with no patch frame, absent
    /// without patch frames.
    BackFacingObservations,
    /// The point's stored RMS reprojection error in pixels,
    /// [`Point3D::rms_reprojection_error`](crate::reconstruction::Point3D::rms_reprojection_error).
    RmsErrorPx,
    /// The largest reprojection error among the track's observations, in
    /// pixels, each observation's inline keypoint against the point's
    /// projection; absent without inline keypoints.
    MaxErrorPx,
    /// The stored position's x, in the stored frame. No value at infinity.
    X,
    /// The stored position's y. No value at infinity.
    Y,
    /// The stored position's z. No value at infinity.
    Z,
    /// Distance from the posed cameras' centroid, as
    /// [`ImageSet::placement_scale`](crate::reconstruction::ImageSet::placement_scale)
    /// measures it; `+∞` for a point at infinity, absent with no posed image.
    CentroidDistance,
    /// The stored `normal_confidence` byte, 0 to 255; absent without the
    /// column.
    NormalConfidence,
    /// The lowest stored per-observation ZNCC in the track, `byte / 255` over
    /// the non-zero `observation_confidence` bytes; no value when no
    /// observation was measured, absent without the column.
    MinZncc,
    /// The lowest blur-matched ZNCC against the stored bitmap among the
    /// track's observation readings (`blur_matched_bitmap_zncc`); no value
    /// when no score was read, absent without readings.
    MinBlurMatchedBitmapZncc,
    /// The largest ZNCC self-similarity radius among the track's measured
    /// observation readings, in patch px; no value when none is measured,
    /// absent without readings. A lower bound where any measured reading's
    /// radius is ([`TrackTable::is_at_least`]).
    MaxZnccSelfSimilarityRadius,
    /// The ZNCC self-similarity radius of the point's patch as its reference
    /// view sees it, in patch px: the stored reading at the reference
    /// observation where there is one, else the reading of the stored patch
    /// bitmap. Absent with neither readings nor bitmaps. A lower bound where
    /// the reading's major axis is ([`TrackTable::is_at_least`]).
    ReferenceZnccSelfSimilarityRadius,
    /// The elongation of the same reading's ellipse, its semi-major axis over
    /// its semi-minor axis, `1` for a circle; large for an edge, which slides
    /// along itself and is fixed only across it. No value where the minor
    /// axis may be longer than it reads, which leaves the ratio unknown; a
    /// lower bound where only the major axis may be.
    ReferenceZnccSelfSimilarityElongation,
    /// The largest angle between two of the track's viewing rays, in degrees,
    /// `0` at infinity ([`compute_max_ray_angle_deg`]).
    MaxRayAngleDeg,
    /// The largest viewing angle of the point's patch over the track's
    /// observations, in degrees, as Track View measures it; no value for a
    /// point with no patch frame, absent without patch frames.
    MaxViewingAngleDeg,
    /// Condition number of the point's triangulation
    /// ([`compute_point_diagnostics`]). No value at infinity.
    ConditionNumber,
    /// Inverse-depth z-score of the point's triangulation
    /// ([`compute_point_diagnostics`]). No value at infinity.
    InverseDepthZ,
}

/// Where a column's work runs: what it costs per point.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ColumnCost {
    /// Read from data the reconstruction already holds, in O(1) or O(track
    /// length) per point.
    Stored,
    /// A triangulation, a pass over every pair of observations, a
    /// projection of every observation or a self-similarity reading, per
    /// point.
    Computed,
}

/// The number of columns, the length of [`TrackTableColumn::ALL`].
const COLUMN_COUNT: usize = 19;

impl TrackTableColumn {
    /// Every column, in the order the spec's catalogue lists them.
    pub const ALL: [TrackTableColumn; COLUMN_COUNT] = [
        TrackTableColumn::TrackLength,
        TrackTableColumn::RepeatObservations,
        TrackTableColumn::BackFacingObservations,
        TrackTableColumn::RmsErrorPx,
        TrackTableColumn::MaxErrorPx,
        TrackTableColumn::X,
        TrackTableColumn::Y,
        TrackTableColumn::Z,
        TrackTableColumn::CentroidDistance,
        TrackTableColumn::NormalConfidence,
        TrackTableColumn::MinZncc,
        TrackTableColumn::MinBlurMatchedBitmapZncc,
        TrackTableColumn::MaxZnccSelfSimilarityRadius,
        TrackTableColumn::ReferenceZnccSelfSimilarityRadius,
        TrackTableColumn::ReferenceZnccSelfSimilarityElongation,
        TrackTableColumn::MaxRayAngleDeg,
        TrackTableColumn::MaxViewingAngleDeg,
        TrackTableColumn::ConditionNumber,
        TrackTableColumn::InverseDepthZ,
    ];

    /// The column's stable name on the wire and in the Python bindings.
    pub fn wire_name(self) -> &'static str {
        match self {
            TrackTableColumn::TrackLength => "track_length",
            TrackTableColumn::RepeatObservations => "repeat_observations",
            TrackTableColumn::BackFacingObservations => "back_facing_observations",
            TrackTableColumn::RmsErrorPx => "rms_error_px",
            TrackTableColumn::MaxErrorPx => "max_error_px",
            TrackTableColumn::X => "x",
            TrackTableColumn::Y => "y",
            TrackTableColumn::Z => "z",
            TrackTableColumn::CentroidDistance => "centroid_distance",
            TrackTableColumn::NormalConfidence => "normal_confidence",
            TrackTableColumn::MinZncc => "min_zncc",
            TrackTableColumn::MinBlurMatchedBitmapZncc => "min_blur_matched_bitmap_zncc",
            TrackTableColumn::MaxZnccSelfSimilarityRadius => "max_zncc_self_similarity_radius",
            TrackTableColumn::ReferenceZnccSelfSimilarityRadius => {
                "reference_zncc_self_similarity_radius"
            }
            TrackTableColumn::ReferenceZnccSelfSimilarityElongation => {
                "reference_zncc_self_similarity_elongation"
            }
            TrackTableColumn::MaxRayAngleDeg => "max_ray_angle_deg",
            TrackTableColumn::MaxViewingAngleDeg => "max_viewing_angle_deg",
            TrackTableColumn::ConditionNumber => "condition_number",
            TrackTableColumn::InverseDepthZ => "inverse_depth_z",
        }
    }

    /// The column whose [`Self::wire_name`] is `name`, if any.
    pub fn from_wire_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|c| c.wire_name() == name)
    }

    /// Whether the column is read from stored data or computed per point.
    pub fn cost(self) -> ColumnCost {
        match self {
            TrackTableColumn::BackFacingObservations
            | TrackTableColumn::MaxErrorPx
            | TrackTableColumn::ReferenceZnccSelfSimilarityRadius
            | TrackTableColumn::ReferenceZnccSelfSimilarityElongation
            | TrackTableColumn::MaxRayAngleDeg
            | TrackTableColumn::MaxViewingAngleDeg
            | TrackTableColumn::ConditionNumber
            | TrackTableColumn::InverseDepthZ => ColumnCost::Computed,
            _ => ColumnCost::Stored,
        }
    }

    /// Whether the column holds `f64` values ([`TrackTable::values_f64`])
    /// rather than `f32` ones ([`TrackTable::values`]): the three position
    /// columns, which a georeferenced reconstruction needs `f64` for.
    pub fn is_f64(self) -> bool {
        matches!(
            self,
            TrackTableColumn::X | TrackTableColumn::Y | TrackTableColumn::Z
        )
    }

    /// Whether `recon` can supply this column at all. A column it cannot
    /// supply is **absent** for the whole reconstruction, which is different
    /// from a single point having no value (a NaN).
    pub fn is_available(self, recon: &EditedReconstruction) -> bool {
        match self {
            TrackTableColumn::CentroidDistance => recon
                .base
                .image_set
                .images
                .iter()
                .any(|image| is_posed(&image.quaternion_wxyz, &image.translation_xyz)),
            TrackTableColumn::NormalConfidence => recon.has_normal_confidence(),
            TrackTableColumn::MinZncc => recon.has_observation_confidence(),
            TrackTableColumn::MinBlurMatchedBitmapZncc
            | TrackTableColumn::MaxZnccSelfSimilarityRadius => {
                recon.observation_reading_options().is_some()
            }
            TrackTableColumn::ReferenceZnccSelfSimilarityRadius
            | TrackTableColumn::ReferenceZnccSelfSimilarityElongation => {
                recon.has_patch_bitmaps() || recon.observation_reading_options().is_some()
            }
            TrackTableColumn::MaxErrorPx => recon.has_keypoints(),
            TrackTableColumn::BackFacingObservations | TrackTableColumn::MaxViewingAngleDeg => {
                recon.has_patch_frames()
            }
            TrackTableColumn::TrackLength
            | TrackTableColumn::RepeatObservations
            | TrackTableColumn::RmsErrorPx
            | TrackTableColumn::X
            | TrackTableColumn::Y
            | TrackTableColumn::Z
            | TrackTableColumn::MaxRayAngleDeg
            | TrackTableColumn::ConditionNumber
            | TrackTableColumn::InverseDepthZ => true,
        }
    }

    /// Whether a value of the column can be a lower bound, the true value
    /// possibly larger, so that the table carries a flag per row beside it
    /// ([`TrackTable::is_at_least`]): the two self-similarity radii and the
    /// elongation, read off ellipses whose axes carry `axes_is_at_least`.
    pub fn has_lower_bounds(self) -> bool {
        matches!(
            self,
            TrackTableColumn::MaxZnccSelfSimilarityRadius
                | TrackTableColumn::ReferenceZnccSelfSimilarityRadius
                | TrackTableColumn::ReferenceZnccSelfSimilarityElongation
        )
    }

    /// The column's position in [`Self::ALL`].
    fn ordinal(self) -> usize {
        self as usize
    }
}

/// The track table: column values for every live point of one version, one
/// row per track keyed by its point index, in
/// [`EditedReconstruction::live_indexes`] order. Holds only the columns it
/// was asked for that the reconstruction could supply.
///
/// "No value" for a single point is a NaN. A column whose values can be lower
/// bounds ([`TrackTableColumn::has_lower_bounds`]) carries a flag per row
/// beside its values ([`Self::is_at_least`]). The table is computed on demand
/// by [`Self::compute`] and never stored.
#[derive(Clone, Debug, Default)]
pub struct TrackTable {
    point_indexes: Vec<u32>,
    values_f32: [Option<Vec<f32>>; COLUMN_COUNT],
    values_f64: [Option<Vec<f64>>; COLUMN_COUNT],
    is_at_least: [Option<Vec<bool>>; COLUMN_COUNT],
}

/// Why [`TrackTable::merge`] refused a table: its rows are not this table's,
/// so it was computed from another version.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TrackTableMergeError {
    /// The number of rows of the table merged into.
    pub rows: usize,
    /// The number of rows of the refused table.
    pub other_rows: usize,
}

impl std::fmt::Display for TrackTableMergeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "the track tables were computed from different versions: \
             their point indexes differ ({} rows and {} rows)",
            self.rows, self.other_rows
        )
    }
}

impl std::error::Error for TrackTableMergeError {}

/// Points per block between cancellation polls and progress counts.
const BLOCK: usize = 8192;

/// What every point's values are read against, computed once per call.
struct Context<'a> {
    recon: &'a EditedReconstruction,
    centroid: nalgebra::Point3<f64>,
    /// Every image's intrinsics, built once rather than once per observation.
    intrinsics: Vec<Cow<'a, CameraIntrinsics>>,
}

impl TrackTable {
    /// Compute `columns` for every live point of `recon`.
    ///
    /// Columns the reconstruction cannot supply ([`TrackTableColumn::is_available`])
    /// are skipped, not an error, and [`Self::has`] says which were filled; a
    /// column named twice is computed once. The stored columns are filled
    /// first and the computed ones after them, each pass reporting its count
    /// of points within its own share of `progress`.
    ///
    /// # Errors
    ///
    /// [`Cancelled`] when `progress` is cancelled; it is polled between blocks
    /// of points.
    pub fn compute(
        recon: &EditedReconstruction,
        columns: &[TrackTableColumn],
        progress: &Progress<'_>,
    ) -> Result<TrackTable, Cancelled> {
        let mut wanted = [false; COLUMN_COUNT];
        for &column in columns {
            if column.is_available(recon) {
                wanted[column.ordinal()] = true;
            }
        }
        let pick = |cost: ColumnCost| -> Vec<TrackTableColumn> {
            TrackTableColumn::ALL
                .into_iter()
                .filter(|c| wanted[c.ordinal()] && c.cost() == cost)
                .collect()
        };
        let stored = pick(ColumnCost::Stored);
        let computed = pick(ColumnCost::Computed);

        let point_indexes: Vec<u32> = recon.live_indexes().collect();
        let mut out = TrackTable {
            point_indexes,
            ..TrackTable::default()
        };
        let context = Context {
            recon,
            centroid: recon.base.image_set.camera_centroid(),
            intrinsics: recon.base.image_set.all_image_intrinsics(),
        };

        // The computed columns cost a triangulation or a pass over every pair
        // of rays per point, the stored ones a read, so the computed pass gets
        // most of the bar when there is one.
        let weights = [
            if stored.is_empty() { 0.0 } else { 1.0 },
            if computed.is_empty() { 0.0 } else { 9.0 },
        ];
        let [p_stored, p_computed] = progress.split(weights);
        if !stored.is_empty() {
            let phase = p_stored.phase("stored track table columns");
            out.fill(&context, &stored, &phase)?;
        }
        if !computed.is_empty() {
            let phase = p_computed.phase("computed track table columns");
            out.fill(&context, &computed, &phase)?;
        }
        Ok(out)
    }

    /// Fill `columns` for every point, a block at a time, each block's
    /// points in parallel.
    fn fill(
        &mut self,
        context: &Context<'_>,
        columns: &[TrackTableColumn],
        progress: &Progress<'_>,
    ) -> Result<(), Cancelled> {
        let n = self.point_indexes.len();
        for &column in columns {
            if column.is_f64() {
                self.values_f64[column.ordinal()] = Some(Vec::with_capacity(n));
            } else {
                self.values_f32[column.ordinal()] = Some(Vec::with_capacity(n));
            }
            if column.has_lower_bounds() {
                self.is_at_least[column.ordinal()] = Some(Vec::with_capacity(n));
            }
        }
        let total = n as u64;
        progress.count(0, Some(total), "points");
        for (block_index, block) in self.point_indexes.chunks(BLOCK).enumerate() {
            progress.check_cancel()?;
            let rows: Vec<RowValues> = block
                .par_iter()
                .map(|&index| {
                    let view = context
                        .recon
                        .point(index)
                        .expect("a live index names a live point");
                    point_values(context, &view, columns)
                })
                .collect();
            for &column in columns {
                let k = column.ordinal();
                if column.is_f64() {
                    let values = self.values_f64[k].as_mut().expect("allocated above");
                    values.extend(rows.iter().map(|row| row.values[k]));
                } else {
                    let values = self.values_f32[k].as_mut().expect("allocated above");
                    values.extend(rows.iter().map(|row| row.values[k] as f32));
                }
                if let Some(flags) = self.is_at_least[k].as_mut() {
                    flags.extend(rows.iter().map(|row| row.is_at_least[k]));
                }
            }
            let done = (block_index * BLOCK + block.len()) as u64;
            progress.count(done, Some(total), "points");
        }
        progress.check_cancel()
    }

    /// The point indexes the rows are for, ascending.
    pub fn point_indexes(&self) -> &[u32] {
        &self.point_indexes
    }

    /// Whether `column` was filled.
    pub fn has(&self, column: TrackTableColumn) -> bool {
        let k = column.ordinal();
        self.values_f32[k].is_some() || self.values_f64[k].is_some()
    }

    /// The values of an `f32` column, one per row; `None` when the column was
    /// not filled or is a position column, which is read through
    /// [`Self::values_f64`].
    pub fn values(&self, column: TrackTableColumn) -> Option<&[f32]> {
        self.values_f32[column.ordinal()].as_deref()
    }

    /// The values of a position column (`x`, `y`, `z`), one per row; `None`
    /// when the column was not filled or is not a position column.
    pub fn values_f64(&self, column: TrackTableColumn) -> Option<&[f64]> {
        self.values_f64[column.ordinal()].as_deref()
    }

    /// Per row of a column whose values can be lower bounds
    /// ([`TrackTableColumn::has_lower_bounds`]), whether the true value may be
    /// larger than the one held: `true` where the reading the value comes from
    /// flags the axis it is read off (`axes_is_at_least`), which is set at the
    /// largest radius searched, where the region runs off the square of shifts
    /// searched, and where a gap with no reading could hide more of it.
    /// `false` where the value is exact and where there is no value. `None`
    /// when the column was not filled or has no flags.
    ///
    /// The values sort and bound by their number whatever the flag says.
    pub fn is_at_least(&self, column: TrackTableColumn) -> Option<&[bool]> {
        self.is_at_least[column.ordinal()].as_deref()
    }

    /// Move `other`'s columns into the columns this table does not hold yet.
    ///
    /// The column arrays are moved, not copied. A column this table already
    /// holds is kept and `other`'s copy of it is dropped, so merging a table
    /// computed at the same time for the same columns changes nothing.
    ///
    /// The two tables must describe the same version of a reconstruction.
    /// Their rows are compared, and a table with other rows is refused; that
    /// catches a point added or deleted between the two computes but not an
    /// edit that changed values and kept every point, which the caller rules
    /// out by computing both from one version.
    ///
    /// # Errors
    ///
    /// [`TrackTableMergeError`] when `other`'s point indexes differ from this
    /// table's. This table is then unchanged.
    pub fn merge(&mut self, other: TrackTable) -> Result<(), TrackTableMergeError> {
        if other.point_indexes != self.point_indexes {
            return Err(TrackTableMergeError {
                rows: self.point_indexes.len(),
                other_rows: other.point_indexes.len(),
            });
        }
        let TrackTable {
            values_f32,
            values_f64,
            is_at_least,
            ..
        } = other;
        for (mine, theirs) in self.values_f32.iter_mut().zip(values_f32) {
            if mine.is_none() {
                *mine = theirs;
            }
        }
        for (mine, theirs) in self.values_f64.iter_mut().zip(values_f64) {
            if mine.is_none() {
                *mine = theirs;
            }
        }
        for (mine, theirs) in self.is_at_least.iter_mut().zip(is_at_least) {
            if mine.is_none() {
                *mine = theirs;
            }
        }
        Ok(())
    }

    /// The value of `column` at `row` as `f64`, NaN for no value. The column
    /// must have been filled.
    pub(crate) fn value(&self, column: TrackTableColumn, row: usize) -> f64 {
        let k = column.ordinal();
        match (&self.values_f32[k], &self.values_f64[k]) {
            (Some(v), _) => f64::from(v[row]),
            (_, Some(v)) => v[row],
            (None, None) => f64::NAN,
        }
    }
}

/// One point's values, at their columns' ordinals, and the lower-bound flag of
/// each; every slot not computed is NaN and `false`.
struct RowValues {
    values: [f64; COLUMN_COUNT],
    is_at_least: [bool; COLUMN_COUNT],
}

/// The values of `columns` for one point.
fn point_values(
    context: &Context<'_>,
    view: &PointView<'_>,
    columns: &[TrackTableColumn],
) -> RowValues {
    let image_set = &context.recon.base.image_set;
    let point = view.point();
    let at_infinity = point.is_at_infinity();
    let finite = |v: f64| if at_infinity { f64::NAN } else { v };
    let mut diagnostics: Option<(f32, f32)> = None;
    // The viewing angles and the reference reading each serve two columns,
    // so a point reads them once.
    let mut angles: Option<Option<(f64, f64)>> = None;
    let mut reference: Option<observation_quality::Axes> = None;
    let mut row = RowValues {
        values: [f64::NAN; COLUMN_COUNT],
        is_at_least: [false; COLUMN_COUNT],
    };
    for &column in columns {
        let mut is_at_least = false;
        let value = match column {
            TrackTableColumn::TrackLength => view.observations().len() as f64,
            TrackTableColumn::RepeatObservations => {
                let mut images: Vec<u32> =
                    view.observations().iter().map(|o| o.image_index).collect();
                let length = images.len();
                images.sort_unstable();
                images.dedup();
                (length - images.len()) as f64
            }
            TrackTableColumn::RmsErrorPx => f64::from(point.rms_reprojection_error),
            TrackTableColumn::MaxErrorPx => {
                observation_quality::max_error_px(image_set, &context.intrinsics, view)
            }
            TrackTableColumn::BackFacingObservations => angles
                .get_or_insert_with(|| {
                    observation_quality::viewing_angles(image_set, &context.intrinsics, view)
                })
                .map_or(f64::NAN, |(_, count)| count),
            TrackTableColumn::MaxViewingAngleDeg => angles
                .get_or_insert_with(|| {
                    observation_quality::viewing_angles(image_set, &context.intrinsics, view)
                })
                .map_or(f64::NAN, |(largest, _)| largest),
            TrackTableColumn::X => finite(point.position.x),
            TrackTableColumn::Y => finite(point.position.y),
            TrackTableColumn::Z => finite(point.position.z),
            // A point at infinity is farther than any finite point, so its
            // distance is `+∞`, a value that sorts and bounds like one.
            TrackTableColumn::CentroidDistance => {
                if at_infinity {
                    f64::INFINITY
                } else {
                    crate::reconstruction::ImageSet::placement_scale_from(
                        &context.centroid,
                        &point.position,
                    )
                }
            }
            TrackTableColumn::NormalConfidence => {
                view.normal_confidence().map_or(f64::NAN, f64::from)
            }
            TrackTableColumn::MinZncc => min_zncc(view),
            TrackTableColumn::MinBlurMatchedBitmapZncc => {
                observation_quality::min_blur_matched_bitmap_zncc(view)
            }
            TrackTableColumn::ReferenceZnccSelfSimilarityRadius => {
                let axes = reference.get_or_insert_with(|| {
                    observation_quality::reference_self_similarity_axes(view)
                });
                is_at_least = axes.is_at_least[0];
                axes.lengths[0]
            }
            TrackTableColumn::ReferenceZnccSelfSimilarityElongation => {
                let axes = reference.get_or_insert_with(|| {
                    observation_quality::reference_self_similarity_axes(view)
                });
                let (value, flag) = elongation(axes);
                is_at_least = flag;
                value
            }
            TrackTableColumn::MaxZnccSelfSimilarityRadius => {
                let (value, flag) = max_self_similarity_radius(view);
                is_at_least = flag;
                value
            }
            TrackTableColumn::MaxRayAngleDeg => {
                f64::from(compute_max_ray_angle_deg(image_set, view))
            }
            TrackTableColumn::ConditionNumber => f64::from(
                diagnostics
                    .get_or_insert_with(|| compute_point_diagnostics(image_set, view))
                    .0,
            ),
            TrackTableColumn::InverseDepthZ => f64::from(
                diagnostics
                    .get_or_insert_with(|| compute_point_diagnostics(image_set, view))
                    .1,
            ),
        };
        row.values[column.ordinal()] = value;
        row.is_at_least[column.ordinal()] = is_at_least && !value.is_nan();
    }
    row
}

/// The elongation of a reading's ellipse, its semi-major axis over its
/// semi-minor axis, and whether it is a lower bound.
///
/// The ratio is exact where both axes are. Where only the major axis may be
/// longer, the ratio may be larger, so it is a lower bound. Where the minor
/// axis may be longer, the ratio may be smaller or, with the major axis also
/// flagged, anything, so it has no value: a flat patch, whose axes both read
/// the cap, is one.
fn elongation(axes: &observation_quality::Axes) -> (f64, bool) {
    let [major, minor] = axes.lengths;
    let [major_at_least, minor_at_least] = axes.is_at_least;
    if minor_at_least {
        return (f64::NAN, false);
    }
    (major / minor, major_at_least)
}

/// The lowest measured `observation_confidence` byte of the track, over 255:
/// a `0` byte says the observation was never measured, so it is skipped.
fn min_zncc(view: &PointView<'_>) -> f64 {
    view.observation_confidence()
        .and_then(|bytes| bytes.iter().copied().filter(|&b| b != 0).min())
        .map_or(f64::NAN, |b| f64::from(b) / 255.0)
}

/// The largest semi-major axis of the self-similarity ellipse among the
/// track's measured readings, and whether it is a lower bound. It is where any
/// measured reading's major axis is flagged, whatever that reading's own
/// value: its true radius may be larger than the largest shown.
fn max_self_similarity_radius(view: &PointView<'_>) -> (f64, bool) {
    let mut largest = f64::NAN;
    let mut is_at_least = false;
    for row in view.observation_readings().into_iter().flatten() {
        if row.is_measured() {
            largest = largest.max(f64::from(row.ellipse_axes[0]));
            is_at_least |= row.ellipse_axes_is_at_least[0];
        }
    }
    (largest, is_at_least)
}
