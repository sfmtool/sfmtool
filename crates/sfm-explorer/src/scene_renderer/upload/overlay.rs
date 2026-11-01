// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Base identity and the deleted mask: how a node's *edits* reach the GPU
//! without touching what its base uploaded.
//!
//! See `specs/gui/document-model.md`. A bundle remembers the base `Arc` its
//! buffers were built from, so the frame's upload phase asks
//! [`SceneRenderer::base_changed`] rather than reading a flag, and a run of
//! point edits -- which shares one base -- answers "no" every time. What such
//! an edit does change is the version's deleted set, and that reaches the point
//! and patch shaders through [`SceneRenderer::update_point_mask`], as one
//! `u32` per instance written only where the set moved.

use std::collections::HashSet;
use std::sync::Arc;

use super::super::SceneRenderer;
use crate::filter_3d::KeptPoints;
use crate::scene::ReconId;
use sfmtool_core::SfmrReconstruction;

/// The mask word of a point the version in view has deleted: the shaders emit a
/// clipped vertex for it, so it draws nothing, occludes nothing and answers no
/// pick.
pub(crate) const MASK_DELETED: u32 = 0;
/// The mask word of an ordinary live point.
pub(crate) const MASK_ALIVE: u32 = 1;
/// The mask word of a live point the viewport is calling out: drawn in the
/// hover tint for as long as whatever is calling it out lasts.
pub(crate) const MASK_HIGHLIGHTED: u32 = 2;
/// The mask word of a live point the eraser brush has painted in the stroke
/// being held: drawn in the painted red.
pub(crate) const MASK_PAINTED: u32 = 3;
/// The mask word of a live point under the eraser brush's circle before a
/// press: drawn in the pale red.
pub(crate) const MASK_UNDER_BRUSH: u32 = 4;
/// The bit set on a live point's word, over whichever of the words above it
/// takes, when the Track Table's *Filter the 3D view* rejects the point
/// ([`crate::filter_3d`]). The node's `ReconUniforms::filter_mode` says
/// whether such a point is dimmed or hidden. A deleted point's word never
/// carries it.
pub(crate) const MASK_REJECTED: u32 = 0x100;

/// The viewport's marks on one node's points, in the index space the mask is
/// keyed on (edited indexes). Each set picks a mask word other than
/// [`MASK_ALIVE`] for a live point; [`mask_word`] says which wins when a point
/// is in more than one. The Track Table's filter adds [`MASK_REJECTED`] on
/// top.
#[derive(Clone, Copy)]
pub(crate) struct PointMarks<'a> {
    /// The points a camera being moved observes ([`crate::camera_lock`]).
    pub(crate) highlighted: &'a HashSet<u32>,
    /// The points the stroke being held has painted.
    pub(crate) painted: &'a HashSet<u32>,
    /// The points under the brush's circle before a press.
    pub(crate) under_brush: &'a HashSet<u32>,
    /// The points the Track Table's filter keeps, while it is on this node:
    /// every live point outside it takes [`MASK_REJECTED`]. Built for the
    /// version in view and no other ([`crate::filter_3d::Filter3d::kept_for`]).
    pub(crate) kept: Option<&'a Arc<KeptPoints>>,
    /// Whether the filter hides the points it rejects rather than dims them.
    pub(crate) filter_hides: bool,
}

/// The set every mark of [`PointMarks::none`] borrows.
static NO_POINTS: std::sync::LazyLock<HashSet<u32>> = std::sync::LazyLock::new(HashSet::new);

impl PointMarks<'static> {
    /// No marks: every live point takes [`MASK_ALIVE`].
    pub(crate) fn none() -> Self {
        Self {
            highlighted: &NO_POINTS,
            painted: &NO_POINTS,
            under_brush: &NO_POINTS,
            kept: None,
            filter_hides: false,
        }
    }
}

/// The mask word one point index takes under a deleted set and the viewport's
/// marks.
///
/// The one place the words are chosen, so a buffer *built* from the sets and a
/// buffer *corrected* entry by entry cannot come to disagree about what a given
/// index means. Deleted wins over every mark, since a deleted point draws
/// nothing. Painted wins over under-the-brush, so a painted point keeps its red
/// when the circle passes over it again. The brush's two marks win over the
/// highlight, though the brush and the Move Camera lock are never on together.
/// A live point the Track Table's filter rejects carries [`MASK_REJECTED`]
/// beside its word, so a rejected point keeps the brush's red.
pub(crate) fn mask_word(index: u32, deleted: &HashSet<u32>, marks: PointMarks<'_>) -> u32 {
    let word = if deleted.contains(&index) {
        return MASK_DELETED;
    } else if marks.painted.contains(&index) {
        MASK_PAINTED
    } else if marks.under_brush.contains(&index) {
        MASK_UNDER_BRUSH
    } else if marks.highlighted.contains(&index) {
        MASK_HIGHLIGHTED
    } else {
        MASK_ALIVE
    };
    if marks.kept.is_some_and(|kept| !kept.contains(index)) {
        word | MASK_REJECTED
    } else {
        word
    }
}

/// Whether two kept sets are the same set, by pointer: what the mask was
/// written from against what the frame asks for.
fn same_kept(a: Option<&Arc<KeptPoints>>, b: Option<&Arc<KeptPoints>>) -> bool {
    match (a, b) {
        (Some(a), Some(b)) => Arc::ptr_eq(a, b),
        (None, None) => true,
        _ => false,
    }
}

/// Split `(position, word)` entries, sorted by position with no repeats, into
/// runs of consecutive positions: each run is its first position and the words
/// to write from there, one `write_buffer` apiece.
pub(super) fn contiguous_runs(entries: &[(u32, u32)]) -> Vec<(u32, Vec<u32>)> {
    let mut runs: Vec<(u32, Vec<u32>)> = Vec::new();
    for &(position, word) in entries {
        match runs.last_mut() {
            Some((start, words)) if *start + words.len() as u32 == position => words.push(word),
            _ => runs.push((position, vec![word])),
        }
    }
    runs
}

/// Write `entries` into `buffer`, one `write_buffer` per run of consecutive
/// positions. Sorts them first, since a patch slot's order need not follow its
/// point index's.
fn write_runs(queue: &wgpu::Queue, buffer: &wgpu::Buffer, entries: &mut [(u32, u32)]) {
    entries.sort_unstable_by_key(|&(position, _)| position);
    for (start, words) in contiguous_runs(entries) {
        queue.write_buffer(buffer, u64::from(start) * 4, bytemuck::cast_slice(&words));
    }
}

impl SceneRenderer {
    /// Whether `id`'s buffers were built from a base other than `base`.
    ///
    /// True for a node with no bundle yet, which is how a first upload is
    /// asked for.
    pub(crate) fn base_changed(&self, id: ReconId, base: &Arc<SfmrReconstruction>) -> bool {
        match self.recons.get(&id) {
            Some(bundle) => !bundle
                .uploaded_base
                .as_ref()
                .is_some_and(|uploaded| Arc::ptr_eq(uploaded, base)),
            None => true,
        }
    }

    /// Whether `id` has ever had a base uploaded.
    ///
    /// The difference between a node the renderer is meeting for the first time
    /// and one whose base an edit replaced. Both reach the upload phase as a
    /// [`Self::base_changed`], and they are the same work; what separates them is
    /// what the *view* owes them. A node arriving in the scene has no scene scale
    /// yet and gets one derived from its data; a node the viewer has been looking
    /// at already has whatever scale the viewer chose, and an edit is not a reason
    /// to take it away.
    pub(crate) fn has_uploaded_base(&self, id: ReconId) -> bool {
        self.recons
            .get(&id)
            .is_some_and(|bundle| bundle.uploaded_base.is_some())
    }

    /// Record which base `id`'s buffers now hold. Called by the upload phase
    /// once the three per-node uploads have run.
    pub(crate) fn set_uploaded_base(&mut self, id: ReconId, base: Arc<SfmrReconstruction>) {
        if let Some(bundle) = self.recons.get_mut(&id) {
            bundle.uploaded_base = Some(base);
        }
    }

    /// Bring `id`'s point mask in line with `deleted` and `marks`.
    ///
    /// One `u32` per point, carrying every answer because they are one word to
    /// the shader: `0` deleted, `1` alive, `2` alive and drawn in the hover
    /// tint, `3` alive and painted by the eraser brush, `4` alive and under the
    /// brush's circle ([`mask_word`] picks it). Written only for the indexes
    /// that entered or left any of the sets, in the point buffer and -- for a
    /// point that carries a surfel -- the patch buffer, so an edit costs its own
    /// size rather than the node's. An index at or above the base's point count
    /// is an addition, and is masked in the additions' own buffers, which the
    /// overlay owns alongside the mask. The changed entries of each buffer go
    /// out as runs of consecutive positions, one `write_buffer` per run.
    ///
    /// The marks are *viewport* statements rather than document ones -- the
    /// points a camera being moved observes ([`crate::camera_lock`]), and the
    /// eraser brush's painted points and the points under its circle -- and
    /// they ride in this word because a second per-point buffer would mean a
    /// second vertex attribute and a second pipeline layout for a temporary
    /// colour.
    ///
    /// The Track Table's filter rides in the same word, as [`MASK_REJECTED`],
    /// for the same reason, and so it reaches the surfels as well as the
    /// splats at no memory beyond the word already there. Its kept set is
    /// compared by pointer; a new one, or the filter coming or going, writes
    /// every word of the node once.
    ///
    /// Returns how many indexes it wrote, which is what the frame's phase note
    /// says beside the time: this runs on every frame for every node, so zero
    /// is the usual answer and is the one that means nothing happened.
    pub(crate) fn update_point_mask(
        &mut self,
        queue: &wgpu::Queue,
        id: ReconId,
        deleted: &HashSet<u32>,
        marks: PointMarks<'_>,
    ) -> usize {
        let Some(bundle) = self.recons.get_mut(&id) else {
            return 0;
        };
        // Dim or hide is a uniform, so switching between them writes no word.
        bundle.filter_hides = marks.kept.is_some() && marks.filter_hides;
        let kept_moved = !same_kept(bundle.masked_kept.as_ref(), marks.kept);
        if !kept_moved
            && bundle.masked_deleted == *deleted
            && bundle.masked_highlighted == *marks.highlighted
            && bundle.masked_painted == *marks.painted
            && bundle.masked_under_brush == *marks.under_brush
        {
            return 0;
        }
        // A new kept set, or a filter switched on or off, can change any
        // point's bit, so every word is written again: once per change of the
        // Track Table's matched rows, never per frame.
        let mut changed: Vec<u32> = if kept_moved {
            (0..bundle.point_count + bundle.addition_count()).collect()
        } else {
            bundle
                .masked_deleted
                .symmetric_difference(deleted)
                .copied()
                .collect()
        };
        if !kept_moved {
            for (was, now) in [
                (&bundle.masked_highlighted, marks.highlighted),
                (&bundle.masked_painted, marks.painted),
                (&bundle.masked_under_brush, marks.under_brush),
            ] {
                changed.extend(was.symmetric_difference(now).copied());
            }
            changed.sort_unstable();
            changed.dedup();
        }
        let written = changed.len();

        // Each buffer's entries, as (position in that buffer, word). `changed`
        // is ascending, so the point rows already are; a patch slot need not
        // follow its point's order, and `write_runs` sorts.
        let mut points = Vec::new();
        let mut patches = Vec::new();
        let mut addition_points = Vec::new();
        let mut addition_patches = Vec::new();
        for index in changed {
            let word = mask_word(index, deleted, marks);
            if index < bundle.point_count {
                points.push((index, word));
            }
            if let Some(&slot) = bundle
                .patch
                .as_ref()
                .and_then(|patch| patch.slot_of_point.get(&index))
            {
                patches.push((slot, word));
            }
            // An addition's own instance, whose row is its edited index less
            // the base's point count. Its surfel is keyed on the edited index,
            // like the base's, because the instances carry that.
            if let Some(additions) = &bundle.additions {
                if let Some(row) = index.checked_sub(bundle.point_count) {
                    if row < additions.point_count {
                        addition_points.push((row, word));
                    }
                }
                if let Some(&slot) = additions
                    .patch
                    .as_ref()
                    .and_then(|patch| patch.slot_of_point.get(&index))
                {
                    addition_patches.push((slot, word));
                }
            }
        }
        if let Some(buffer) = &bundle.point_alive_buffer {
            write_runs(queue, buffer, &mut points);
        }
        if let Some(patch) = &bundle.patch {
            write_runs(queue, &patch.alive_buffer, &mut patches);
        }
        if let Some(additions) = &bundle.additions {
            write_runs(queue, &additions.point_alive_buffer, &mut addition_points);
            if let Some(patch) = &additions.patch {
                write_runs(queue, &patch.alive_buffer, &mut addition_patches);
            }
        }
        bundle.masked_deleted.clone_from(deleted);
        bundle.masked_highlighted.clone_from(marks.highlighted);
        bundle.masked_painted.clone_from(marks.painted);
        bundle.masked_under_brush.clone_from(marks.under_brush);
        bundle.masked_kept = marks.kept.cloned();
        written
    }

    /// The word `id`'s mask holds for `index`, as the last
    /// [`Self::update_point_mask`] left it. For the tests, which cannot read a
    /// buffer back from the `noop` backend and so assert on the sets the
    /// writes were diffed against.
    #[cfg(test)]
    pub(crate) fn masked_word(&self, id: ReconId, index: u32) -> Option<u32> {
        self.recons
            .get(&id)
            .map(|bundle| mask_word(index, &bundle.masked_deleted, bundle.masked_marks()))
    }

    /// How many base indexes `id`'s mask currently marks as deleted. For the
    /// tests, which assert on what reached the GPU rather than on what the
    /// document holds.
    #[cfg(test)]
    pub(crate) fn masked_deleted_count(&self, id: ReconId) -> usize {
        self.recons
            .get(&id)
            .map_or(0, |bundle| bundle.masked_deleted.len())
    }
}
