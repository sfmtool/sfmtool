// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Which other keypoints lie inside a keypoint's own disk, per image.
//!
//! A KEYPOINT here is a row of a track set: an image index, a pixel position,
//! and its own query radius, its REACH, in pixels. Several rules read that
//! neighbourhood and differ only in what they then test, so the enumeration is
//! stated once and the tests stay with the callers.
//!
//! This is the pixel-domain counterpart of the world-space queries on
//! [`PointCloud`](super::PointCloud): that index answers proximity between 3D
//! points in world units at a shared radius; this one answers proximity between
//! 2D keypoints in pixels, per image, each keypoint carrying its own radius.
//!
//! The relation is DIRECTED: the disk is row `i`'s, so `(i, j)` says nothing
//! about `(j, i)`. A row is never its own candidate, and a row whose reach is not
//! finite asks nothing while still appearing as a candidate of others.
//!
//! Within an image the rows are ordered by column. A disk of radius `reach`
//! cannot contain a centre whose column is further than `reach` away, so a
//! row's candidates are one contiguous run of that order, found by binary
//! search at `x - reach` and `x + reach` and then filtered by true Euclidean
//! distance. Cost is the sort plus the output size: no tree, no grid, and no
//! quadratic pass over an image unless the answer itself is quadratic.
//!
//! See `specs/core/analysis/keypoint-reach.md` for the design.

use std::cmp::Ordering;
use std::fmt;

use rayon::prelude::*;

/// How many rows one unit of the expansion covers.
///
/// A work bound and not a threshold: the pairs are the same pairs in the same
/// order at any value of it, so it trades scheduling grain against per-unit
/// overhead and nothing else.
pub const BATCH_ROWS: usize = 256;

/// One row per keypoint over a whole track set.
#[derive(Debug, Clone, Copy)]
pub struct KeypointRows<'a> {
    /// `n` image index per row.
    pub image_of_row: &'a [i64],
    /// `n * 2` pixel positions, `[x, y, x, y, ...]`.
    pub xy_px: &'a [f64],
    /// `n` query radius per row, in pixels.
    pub reach_px: &'a [f64],
}

/// What the enumeration refuses to answer.
#[derive(Debug, Clone, PartialEq)]
pub enum KeypointReachError {
    /// The three inputs disagree on how many rows there are.
    LengthMismatch {
        /// Rows the image index states.
        images: usize,
        /// Rows the position array states.
        positions: usize,
        /// Rows the reach array states.
        reaches: usize,
    },
    /// A row asks with a radius below zero, which names no disk.
    NegativeReach {
        /// The offending row.
        row: usize,
        /// The radius it stated.
        reach: f64,
    },
}

impl fmt::Display for KeypointReachError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LengthMismatch {
                images,
                positions,
                reaches,
            } => write!(
                f,
                "image_of_row states {images} rows, xy_px {positions} and reach_px {reaches}"
            ),
            Self::NegativeReach { row, reach } => {
                write!(f, "row {row} states a negative reach {reach}")
            }
        }
    }
}

impl std::error::Error for KeypointReachError {}

/// The candidate pairs, as three parallel arrays in the defined order.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ReachPairs {
    /// The row whose disk was asked.
    pub row: Vec<i64>,
    /// The row that lies inside it.
    pub candidate: Vec<i64>,
    /// Their separation in pixels.
    pub distance_px: Vec<f64>,
}

impl ReachPairs {
    /// How many pairs the enumeration found.
    pub fn len(&self) -> usize {
        self.row.len()
    }

    /// Whether no pair was found.
    pub fn is_empty(&self) -> bool {
        self.row.is_empty()
    }
}

/// Every row inside every row's own reach, per image.
///
/// Rows of different images are never paired. The pair stream runs rows in
/// their given order within an image, each row's candidates in the sorted
/// column order of its run, and images in ascending image index.
///
/// # Errors
///
/// [`KeypointReachError::LengthMismatch`] where the three inputs disagree on
/// the row count, [`KeypointReachError::NegativeReach`] where a row states a
/// radius below zero. A NaN reach is the documented "asks nothing" value and
/// is not an error.
pub fn pairs_within_reach(rows: KeypointRows<'_>) -> Result<ReachPairs, KeypointReachError> {
    pairs_within_reach_batch(rows, BATCH_ROWS, true)
}

/// [`pairs_within_reach`] with the work grain and the parallelism named.
///
/// The output is identical at every `batch_rows` and with `parallel` either
/// way; both exist so a test can say so.
///
/// # Errors
///
/// The same refusals as [`pairs_within_reach`].
pub fn pairs_within_reach_batch(
    rows: KeypointRows<'_>,
    batch_rows: usize,
    parallel: bool,
) -> Result<ReachPairs, KeypointReachError> {
    let n = rows.image_of_row.len();
    if rows.xy_px.len() != 2 * n || rows.reach_px.len() != n {
        return Err(KeypointReachError::LengthMismatch {
            images: n,
            positions: rows.xy_px.len() / 2,
            reaches: rows.reach_px.len(),
        });
    }
    if let Some((row, &reach)) = rows.reach_px.iter().enumerate().find(|(_, &r)| r < 0.0) {
        return Err(KeypointReachError::NegativeReach { row, reach });
    }
    if n == 0 {
        return Ok(ReachPairs::default());
    }

    let groups = group_by_image(rows);
    let grain = batch_rows.max(1);
    let units: Vec<(usize, usize, usize)> = groups
        .iter()
        .enumerate()
        .flat_map(|(g, grp)| {
            let m = grp.rows.len();
            (0..m)
                .step_by(grain)
                .map(move |lo| (g, lo, (lo + grain).min(m)))
        })
        .collect();

    let parts: Vec<ReachPairs> = if parallel {
        units
            .par_iter()
            .map(|&(g, lo, hi)| expand(&groups[g], lo, hi))
            .collect()
    } else {
        units
            .iter()
            .map(|&(g, lo, hi)| expand(&groups[g], lo, hi))
            .collect()
    };

    let total: usize = parts.iter().map(ReachPairs::len).sum();
    let mut out = ReachPairs {
        row: Vec::with_capacity(total),
        candidate: Vec::with_capacity(total),
        distance_px: Vec::with_capacity(total),
    };
    for part in parts {
        out.row.extend_from_slice(&part.row);
        out.candidate.extend_from_slice(&part.candidate);
        out.distance_px.extend_from_slice(&part.distance_px);
    }
    Ok(out)
}

/// One image's rows, gathered in row order and indexed by column.
struct ImageRows {
    /// Global row index of each of the image's rows, in row order.
    rows: Vec<i64>,
    /// Column of each of those rows.
    x: Vec<f64>,
    /// Row coordinate of each of those rows.
    y: Vec<f64>,
    /// Query radius of each of those rows.
    reach: Vec<f64>,
    /// Local positions ordered by column, ties in row order.
    by_column: Vec<u32>,
    /// The columns themselves in that order, so the search reads one slice.
    columns: Vec<f64>,
}

/// The rows of each image, images in ascending image index.
fn group_by_image(rows: KeypointRows<'_>) -> Vec<ImageRows> {
    let n = rows.image_of_row.len();
    let mut order: Vec<u32> = (0..n as u32).collect();
    order.sort_by_key(|&r| rows.image_of_row[r as usize]);

    let mut bounds = vec![0usize];
    for k in 1..n {
        if rows.image_of_row[order[k] as usize] != rows.image_of_row[order[k - 1] as usize] {
            bounds.push(k);
        }
    }
    bounds.push(n);

    (0..bounds.len() - 1)
        .into_par_iter()
        .map(|b| {
            let members = &order[bounds[b]..bounds[b + 1]];
            let m = members.len();
            let mut grp = ImageRows {
                rows: Vec::with_capacity(m),
                x: Vec::with_capacity(m),
                y: Vec::with_capacity(m),
                reach: Vec::with_capacity(m),
                by_column: (0..m as u32).collect(),
                columns: Vec::with_capacity(m),
            };
            for &r in members {
                let r = r as usize;
                grp.rows.push(r as i64);
                grp.x.push(rows.xy_px[2 * r]);
                grp.y.push(rows.xy_px[2 * r + 1]);
                grp.reach.push(rows.reach_px[r]);
            }
            let xs = &grp.x;
            grp.by_column
                .sort_by(|&a, &b| cmp_nan_last(xs[a as usize], xs[b as usize]));
            grp.columns = grp.by_column.iter().map(|&k| grp.x[k as usize]).collect();
            grp
        })
        .collect()
}

/// The pairs the rows `lo..hi` of one image ask for.
fn expand(grp: &ImageRows, lo: usize, hi: usize) -> ReachPairs {
    let mut out = ReachPairs::default();
    for i in lo..hi {
        let reach = grp.reach[i];
        if !reach.is_finite() {
            continue;
        }
        let (xi, yi) = (grp.x[i], grp.y[i]);
        let first = lower_bound(&grp.columns, xi - reach);
        let last = upper_bound(&grp.columns, xi + reach);
        for k in first..last {
            let j = grp.by_column[k] as usize;
            if j == i {
                continue;
            }
            let dx = grp.x[j] - xi;
            let dy = grp.y[j] - yi;
            // Written out rather than fused: the callers' masks are compared
            // against a NumPy `sqrt(dx * dx + dy * dy)`, and an FMA would round
            // the sum once instead of twice.
            let d = (dx * dx + dy * dy).sqrt();
            if d <= reach {
                out.row.push(grp.rows[i]);
                out.candidate.push(grp.rows[j]);
                out.distance_px.push(d);
            }
        }
    }
    out
}

/// Total order over `f64` that puts NaN last, so a column of NaN sorts to the
/// end of an image's order and is never found by a search.
fn cmp_nan_last(a: f64, b: f64) -> Ordering {
    match a.partial_cmp(&b) {
        Some(order) => order,
        None if a.is_nan() && b.is_nan() => Ordering::Equal,
        None if a.is_nan() => Ordering::Greater,
        None => Ordering::Less,
    }
}

/// First position whose column is not below `key`, under [`cmp_nan_last`].
fn lower_bound(columns: &[f64], key: f64) -> usize {
    columns.partition_point(|&v| cmp_nan_last(v, key) == Ordering::Less)
}

/// First position whose column is above `key`, under [`cmp_nan_last`].
fn upper_bound(columns: &[f64], key: f64) -> usize {
    columns.partition_point(|&v| cmp_nan_last(v, key) != Ordering::Greater)
}

#[cfg(test)]
mod tests;
