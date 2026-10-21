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

use crate::progress::{Cancelled, Progress};
use crate::reconstruction::bundle_adjust::is_posed;
use crate::reconstruction::{EditedReconstruction, PointView};

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
    /// Number of distinct images among the track's observations, less than
    /// the track length where one image holds several observations.
    ImageCount,
    /// The point's stored RMS reprojection error in pixels,
    /// [`Point3D::rms_reprojection_error`](crate::reconstruction::Point3D::rms_reprojection_error).
    RmsErrorPx,
    /// `1` for a point at infinity (`w = 0`), `0` for a finite one.
    AtInfinity,
    /// The stored position's x, in the stored frame. No value at infinity.
    X,
    /// The stored position's y. No value at infinity.
    Y,
    /// The stored position's z. No value at infinity.
    Z,
    /// Distance from the posed cameras' centroid, as
    /// [`ImageSet::placement_scale`](crate::reconstruction::ImageSet::placement_scale)
    /// measures it. No value at infinity; absent with no posed image.
    CentroidDistance,
    /// The stored `normal_confidence` byte, 0 to 255; absent without the
    /// column.
    NormalConfidence,
    /// The lowest stored per-observation ZNCC in the track, `byte / 255` over
    /// the non-zero `observation_confidence` bytes; no value when no
    /// observation was measured, absent without the column.
    LowestZncc,
    /// The largest ZNCC self-similarity radius among the track's measured
    /// observation readings, in patch px; no value when none is measured,
    /// absent without readings.
    LargestZnccSelfSimilarityRadius,
    /// The largest angle between two of the track's viewing rays, in degrees,
    /// `0` at infinity ([`compute_max_ray_angle_deg`]).
    MaxRayAngleDeg,
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
    /// A triangulation or a pass over every pair of observations, per point.
    Computed,
}

/// The number of columns, the length of [`TrackTableColumn::ALL`].
const COLUMN_COUNT: usize = 14;

impl TrackTableColumn {
    /// Every column, in the order the spec's catalogue lists them.
    pub const ALL: [TrackTableColumn; COLUMN_COUNT] = [
        TrackTableColumn::TrackLength,
        TrackTableColumn::ImageCount,
        TrackTableColumn::RmsErrorPx,
        TrackTableColumn::AtInfinity,
        TrackTableColumn::X,
        TrackTableColumn::Y,
        TrackTableColumn::Z,
        TrackTableColumn::CentroidDistance,
        TrackTableColumn::NormalConfidence,
        TrackTableColumn::LowestZncc,
        TrackTableColumn::LargestZnccSelfSimilarityRadius,
        TrackTableColumn::MaxRayAngleDeg,
        TrackTableColumn::ConditionNumber,
        TrackTableColumn::InverseDepthZ,
    ];

    /// The column's stable name on the wire and in the Python bindings.
    pub fn wire_name(self) -> &'static str {
        match self {
            TrackTableColumn::TrackLength => "track_length",
            TrackTableColumn::ImageCount => "image_count",
            TrackTableColumn::RmsErrorPx => "rms_error_px",
            TrackTableColumn::AtInfinity => "at_infinity",
            TrackTableColumn::X => "x",
            TrackTableColumn::Y => "y",
            TrackTableColumn::Z => "z",
            TrackTableColumn::CentroidDistance => "centroid_distance",
            TrackTableColumn::NormalConfidence => "normal_confidence",
            TrackTableColumn::LowestZncc => "lowest_zncc",
            TrackTableColumn::LargestZnccSelfSimilarityRadius => {
                "largest_zncc_self_similarity_radius"
            }
            TrackTableColumn::MaxRayAngleDeg => "max_ray_angle_deg",
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
            TrackTableColumn::MaxRayAngleDeg
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
            TrackTableColumn::LowestZncc => recon.has_observation_confidence(),
            TrackTableColumn::LargestZnccSelfSimilarityRadius => {
                recon.observation_reading_options().is_some()
            }
            _ => true,
        }
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
/// "No value" for a single point is a NaN. The table is computed on demand by
/// [`Self::compute`] and never stored.
#[derive(Clone, Debug, Default)]
pub struct TrackTable {
    point_indexes: Vec<u32>,
    values_f32: [Option<Vec<f32>>; COLUMN_COUNT],
    values_f64: [Option<Vec<f64>>; COLUMN_COUNT],
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
        }
        let total = n as u64;
        progress.count(0, Some(total), "points");
        for (block_index, block) in self.point_indexes.chunks(BLOCK).enumerate() {
            progress.check_cancel()?;
            let rows: Vec<[f64; COLUMN_COUNT]> = block
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
                    values.extend(rows.iter().map(|row| row[k]));
                } else {
                    let values = self.values_f32[k].as_mut().expect("allocated above");
                    values.extend(rows.iter().map(|row| row[k] as f32));
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

/// The values of `columns` for one point, at their ordinals; every other slot
/// is NaN.
fn point_values(
    context: &Context<'_>,
    view: &PointView<'_>,
    columns: &[TrackTableColumn],
) -> [f64; COLUMN_COUNT] {
    let image_set = &context.recon.base.image_set;
    let point = view.point();
    let at_infinity = point.is_at_infinity();
    let finite = |v: f64| if at_infinity { f64::NAN } else { v };
    let mut diagnostics: Option<(f32, f32)> = None;
    let mut row = [f64::NAN; COLUMN_COUNT];
    for &column in columns {
        let value = match column {
            TrackTableColumn::TrackLength => view.observations().len() as f64,
            TrackTableColumn::ImageCount => {
                let mut images: Vec<u32> =
                    view.observations().iter().map(|o| o.image_index).collect();
                images.sort_unstable();
                images.dedup();
                images.len() as f64
            }
            TrackTableColumn::RmsErrorPx => f64::from(point.rms_reprojection_error),
            TrackTableColumn::AtInfinity => {
                if at_infinity {
                    1.0
                } else {
                    0.0
                }
            }
            TrackTableColumn::X => finite(point.position.x),
            TrackTableColumn::Y => finite(point.position.y),
            TrackTableColumn::Z => finite(point.position.z),
            TrackTableColumn::CentroidDistance => {
                finite(crate::reconstruction::ImageSet::placement_scale_from(
                    &context.centroid,
                    &point.position,
                ))
            }
            TrackTableColumn::NormalConfidence => {
                view.normal_confidence().map_or(f64::NAN, f64::from)
            }
            TrackTableColumn::LowestZncc => lowest_zncc(view),
            TrackTableColumn::LargestZnccSelfSimilarityRadius => {
                largest_self_similarity_radius(view)
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
        row[column.ordinal()] = value;
    }
    row
}

/// The lowest measured `observation_confidence` byte of the track, over 255:
/// a `0` byte says the observation was never measured, so it is skipped.
fn lowest_zncc(view: &PointView<'_>) -> f64 {
    view.observation_confidence()
        .and_then(|bytes| bytes.iter().copied().filter(|&b| b != 0).min())
        .map_or(f64::NAN, |b| f64::from(b) / 255.0)
}

/// The largest semi-major axis of the self-similarity ellipse among the
/// track's measured readings.
fn largest_self_similarity_radius(view: &PointView<'_>) -> f64 {
    view.observation_readings()
        .and_then(|rows| {
            rows.iter()
                .filter(|r| r.is_measured())
                .map(|r| f64::from(r.ellipse_axes[0]))
                .reduce(f64::max)
        })
        .unwrap_or(f64::NAN)
}
