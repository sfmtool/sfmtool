// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The query over the track table: which points pass a set of bounds on its
//! columns, and in which order.

use std::cmp::Ordering;
use std::fmt;

use super::{TrackTable, TrackTableColumn};
use crate::reconstruction::EditedReconstruction;

/// A minimum and a maximum on one column, either of which may be left open.
/// A row passes when its value is at least `min` and at most `max`; a row
/// with no value (NaN) fails a bound that sets either. A bound that sets
/// neither filters nothing. `+∞` (the `centroid_distance` of a point at
/// infinity) is a value: it passes any `min` and fails any finite `max`.
#[derive(Clone, Debug, PartialEq)]
pub struct ColumnBounds {
    pub column: TrackTableColumn,
    pub min: Option<f64>,
    pub max: Option<f64>,
}

/// The direction of a sort.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum SortOrder {
    Ascending,
    Descending,
}

/// What to match and in which order.
///
/// Settings chosen by the caller, so `*Options` by the type-name rule. (The
/// viewer's Go to Point dialog has an unrelated `PointQuery`, the parse of
/// what was typed; this type is named apart from it.)
#[derive(Clone, Debug, Default)]
pub struct TrackTableQueryOptions {
    /// Every bound must pass.
    pub bounds: Vec<ColumnBounds>,
    /// Keep only points with an observation in this image.
    pub seen_in_image: Option<u32>,
    /// The column and direction to sort by; `None` keeps
    /// [`EditedReconstruction::live_indexes`] order.
    pub order_by: Option<(TrackTableColumn, SortOrder)>,
}

/// Why a query was refused.
#[derive(Clone, Debug, PartialEq)]
pub enum TrackTableQueryError {
    /// A bound or the sort names a column the [`TrackTable`] does not hold.
    ColumnNotComputed(TrackTableColumn),
    /// `seen_in_image` names an image the reconstruction does not hold.
    ImageOutOfRange { image: u32, image_count: usize },
}

impl fmt::Display for TrackTableQueryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TrackTableQueryError::ColumnNotComputed(column) => {
                write!(f, "the column {} was not computed", column.wire_name())
            }
            TrackTableQueryError::ImageOutOfRange { image, image_count } => write!(
                f,
                "image {image} is out of range: the reconstruction has {image_count} images"
            ),
        }
    }
}

impl std::error::Error for TrackTableQueryError {}

/// The point indexes that match `options`, in order.
///
/// The full list, not a page: a point index is what selecting, reading and
/// deleting a point take, so the result passes to any of them unchanged.
/// `table` must have been computed from `recon` as it stands.
///
/// The sort is stable and breaks ties by ascending point index, whichever way
/// it runs, and rows with no value sort after every row that has one, in
/// either direction. An infinite value is a value, not "no value": `+∞` sorts
/// after every finite value ascending and before them descending.
///
/// # Errors
///
/// [`TrackTableQueryError::ColumnNotComputed`] when a bound or the sort names a
/// column `table` does not hold, and [`TrackTableQueryError::ImageOutOfRange`]
/// when `seen_in_image` is not an image of `recon`.
pub fn query_points(
    recon: &EditedReconstruction,
    table: &TrackTable,
    options: &TrackTableQueryOptions,
) -> Result<Vec<u32>, TrackTableQueryError> {
    let named = options
        .bounds
        .iter()
        .map(|b| b.column)
        .chain(options.order_by.map(|(c, _)| c));
    for column in named {
        if !table.has(column) {
            return Err(TrackTableQueryError::ColumnNotComputed(column));
        }
    }
    if let Some(image) = options.seen_in_image {
        let image_count = recon.image_count();
        if image as usize >= image_count {
            return Err(TrackTableQueryError::ImageOutOfRange { image, image_count });
        }
    }

    let passes = |row: usize| -> bool {
        options.bounds.iter().all(|b| {
            if b.min.is_none() && b.max.is_none() {
                return true;
            }
            let v = table.value(b.column, row);
            !v.is_nan() && b.min.is_none_or(|m| v >= m) && b.max.is_none_or(|m| v <= m)
        })
    };
    let seen = |index: u32| -> bool {
        options.seen_in_image.is_none_or(|image| {
            recon.point(index).is_some_and(|view| {
                view.observations()
                    .iter()
                    .any(|obs| obs.image_index == image)
            })
        })
    };
    let mut rows: Vec<usize> = table
        .point_indexes()
        .iter()
        .enumerate()
        .filter(|&(row, &index)| passes(row) && seen(index))
        .map(|(row, _)| row)
        .collect();

    if let Some((column, order)) = options.order_by {
        // The rows are in ascending point index, so a stable sort on the
        // value alone leaves ties in that order.
        rows.sort_by(|&a, &b| {
            let (va, vb) = (table.value(column, a), table.value(column, b));
            match (va.is_nan(), vb.is_nan()) {
                (true, true) => Ordering::Equal,
                (true, false) => Ordering::Greater,
                (false, true) => Ordering::Less,
                // Neither is NaN, so the comparison is total, infinities
                // included; `-0` and `0` compare equal and keep their
                // point-index order.
                (false, false) => {
                    let ascending = va.partial_cmp(&vb).unwrap_or(Ordering::Equal);
                    match order {
                        SortOrder::Ascending => ascending,
                        SortOrder::Descending => ascending.reverse(),
                    }
                }
            }
        });
    }
    let indexes = table.point_indexes();
    Ok(rows.into_iter().map(|row| indexes[row]).collect())
}
