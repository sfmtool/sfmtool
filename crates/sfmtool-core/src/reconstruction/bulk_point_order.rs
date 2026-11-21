// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The order a bulk kernel takes a reconstruction's points in.
//!
//! A bulk kernel (the stored bitmaps' whole-cloud render, *Refine all
//! normals*, *Refit all points*) renders every point it is given from the
//! point's views. Taking the points grouped by the image of each point's
//! reference observation, and within an image by that observation's keypoint,
//! makes the renders of one photograph follow each other, and those of points
//! near each other in it, so the photograph tiles a render reads are the ones
//! the renders before it read. The order uses only the reconstruction, so it
//! does not depend on how photographs are cut into tiles.
//!
//! A kernel runs its parallel loop over [`BulkPointOrder::positions`], which
//! rayon splits into runs of consecutive entries, so the workers are in at
//! most as many reference images at once as there are workers; and it writes
//! each result at its point's place with [`BulkPointOrder::restore`], so its
//! answer is the one any other order gives.

use std::cmp::Ordering;

use super::data::PointSet;

/// The points of a bulk kernel's loop, in the order it takes them: by the
/// image of each point's reference observation, then by that observation's
/// keypoint, its pixel row `floor(y)` and then its `x`, with ties broken by
/// point index; the points with no reference observation come last, by point
/// index.
///
/// A point's reference observation is the one
/// [`PointSet::reference_observations`] names for it, where that is `≥ 0` and
/// within its track. A point set with no keypoints column orders the points
/// of one reference image by point index.
pub(crate) struct BulkPointOrder {
    /// Positions within the points the order was made for, in order.
    positions: Vec<usize>,
}

/// Where a point's reference observation is: its image, and its keypoint's
/// row and `x` where the point set holds keypoints.
type ReferencePlace = (u32, Option<(f32, f32)>);

impl BulkPointOrder {
    /// The order of every point of `point_set`, as positions that are point
    /// indexes.
    pub(crate) fn of_all_points(point_set: &PointSet) -> Self {
        let points: Vec<u32> = (0..point_set.points.len() as u32).collect();
        Self::of_points(point_set, &points)
    }

    /// The order of `points`, indexes into `point_set`, as positions within
    /// `points`. A point index past the point set's points, or a point whose
    /// track the derived offsets do not hold, sorts as a point with no
    /// reference; two entries naming one point keep their order.
    pub(crate) fn of_points(point_set: &PointSet, points: &[u32]) -> Self {
        let mut keyed: Vec<(Option<ReferencePlace>, u32, usize)> = points
            .iter()
            .enumerate()
            .map(|(position, &point)| (reference_place(point_set, point), point, position))
            .collect();
        keyed.sort_unstable_by(|a, b| {
            compare_places(a.0, b.0)
                .then(a.1.cmp(&b.1))
                .then(a.2.cmp(&b.2))
        });
        Self {
            positions: keyed.into_iter().map(|(_, _, position)| position).collect(),
        }
    }

    /// The positions `0..len` in their own order: the order a loop over the
    /// points by index takes.
    #[cfg(test)]
    pub(crate) fn identity(len: usize) -> Self {
        Self {
            positions: (0..len).collect(),
        }
    }

    /// The positions, in the order the kernel takes them.
    pub(crate) fn positions(&self) -> &[usize] {
        &self.positions
    }

    /// `in_order`, one result per entry of [`Self::positions`] in that order,
    /// put back at the positions: the result at `positions()[k]` is
    /// `in_order[k]`. A result past the positions is dropped, and a position
    /// with no result is left out.
    pub(crate) fn restore<T>(&self, in_order: impl IntoIterator<Item = T>) -> Vec<T> {
        let mut slots: Vec<Option<T>> = (0..self.positions.len()).map(|_| None).collect();
        for (&position, result) in self.positions.iter().zip(in_order) {
            if let Some(slot) = slots.get_mut(position) {
                *slot = Some(result);
            }
        }
        slots.into_iter().flatten().collect()
    }
}

/// The place of `point`'s reference observation, `None` where it has none.
fn reference_place(point_set: &PointSet, point: u32) -> Option<ReferencePlace> {
    let p = point as usize;
    let start = *point_set.observation_offsets.get(p)?;
    let end = *point_set.observation_offsets.get(p + 1)?;
    let r = usize::try_from(*point_set.reference_observations.as_ref()?.get(p)?).ok()?;
    let j = start.checked_add(r).filter(|&j| j < end)?;
    let image = point_set.tracks.get(j)?.image_index;
    let keypoint = point_set
        .keypoints_xy()
        .filter(|xy| j < xy.nrows() && xy.ncols() >= 2)
        .map(|xy| (xy[[j, 1]].floor(), xy[[j, 0]]));
    Some((image, keypoint))
}

/// Places with a reference first, by image, then by row and `x`.
fn compare_places(a: Option<ReferencePlace>, b: Option<ReferencePlace>) -> Ordering {
    match (a, b) {
        (Some((image_a, at_a)), Some((image_b, at_b))) => {
            image_a.cmp(&image_b).then(match (at_a, at_b) {
                (Some((row_a, x_a)), Some((row_b, x_b))) => {
                    row_a.total_cmp(&row_b).then(x_a.total_cmp(&x_b))
                }
                _ => Ordering::Equal,
            })
        }
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        (None, None) => Ordering::Equal,
    }
}

#[cfg(test)]
pub(crate) mod tests;
