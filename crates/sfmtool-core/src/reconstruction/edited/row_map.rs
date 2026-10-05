// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! [`RowMap`], the map a whole-value edit leaves from one point numbering to
//! another: built by a materialisation, by a scan of an edit's input and output
//! ([`RowMap::by_scan`]), or from the indexes an edit removed
//! ([`RowMap::by_removal`]).

use crate::reconstruction::data::SfmrReconstruction;

use super::EditError;

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
    /// The map [`EditedReconstruction::materialize`] returns, from the five
    /// lists it builds while emitting rows: `by_edited` ascending by edited
    /// index and `by_new` the same pairs ascending by new index.
    ///
    /// [`EditedReconstruction::materialize`]: super::EditedReconstruction::materialize
    pub(super) fn materialized(
        base_count: u32,
        holes: Vec<u32>,
        replaced: Vec<u32>,
        by_edited: Vec<(u32, u32)>,
        by_new: Vec<(u32, u32)>,
    ) -> Self {
        RowMap {
            base_count,
            holes,
            replaced,
            by_edited,
            by_new,
            // A materialisation knows what it did, so it needs no scan.
            scan: None,
        }
    }

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

    /// The map a whole-value edit that **only drops points** performed, stated
    /// by the edit rather than read back off its two values.
    ///
    /// `point_count` is how many points the edit was given and `removed` the
    /// indexes it dropped, in any order and with repeats allowed. Every
    /// surviving point moves down by how many dropped indexes sit below it,
    /// which is the whole of what such an edit does.
    ///
    /// [`RowMap::by_scan`] answers the same question for an edit that cannot
    /// say what it did, and pays for it: it identifies a point by the images
    /// that see it, so on a value whose points are all seen by the same images
    /// it cannot tell a dropped point from the one after it. An edit that
    /// **knows** its answer says so here instead of having it guessed.
    pub fn by_removal(point_count: u32, removed: &[u32]) -> Self {
        let mut holes: Vec<u32> = removed
            .iter()
            .copied()
            .filter(|&index| index < point_count)
            .collect();
        holes.sort_unstable();
        holes.dedup();
        RowMap {
            base_count: point_count,
            holes,
            replaced: Vec::new(),
            by_edited: Vec::new(),
            by_new: Vec::new(),
            scan: None,
        }
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

#[cfg(test)]
mod tests;
