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

impl SceneRenderer {
    /// Whether `id`'s buffers were built from a base other than `base`.
    ///
    /// True for a node with no bundle yet, which is how a first upload is
    /// asked for.
    pub fn base_changed(&self, id: ReconId, base: &Arc<SfmrReconstruction>) -> bool {
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
    pub fn has_uploaded_base(&self, id: ReconId) -> bool {
        self.recons
            .get(&id)
            .is_some_and(|bundle| bundle.uploaded_base.is_some())
    }

    /// Record which base `id`'s buffers now hold. Called by the upload phase
    /// once the three per-node uploads have run.
    pub fn set_uploaded_base(&mut self, id: ReconId, base: Arc<SfmrReconstruction>) {
        if let Some(bundle) = self.recons.get_mut(&id) {
            bundle.uploaded_base = Some(base);
        }
    }

    /// Bring `id`'s point mask in line with `deleted` and `highlighted`.
    ///
    /// One `u32` per point, carrying both answers because they are one word to
    /// the shader: `0` deleted, `1` alive, `2` alive and drawn in the hover
    /// tint. Written only for the indexes that entered or left either set, in
    /// the point buffer and -- for a point that carries a surfel -- the patch
    /// buffer, so an edit costs its own size rather than the node's. An index at
    /// or above the base's point count is an addition, and is masked in the
    /// additions' own buffers, which the overlay owns alongside the mask.
    ///
    /// The highlight is a *viewport* statement rather than a document one --
    /// today, the points a camera being moved observes
    /// ([`crate::camera_lock`]) -- and it rides in this word because a second
    /// per-point buffer would mean a second vertex attribute and a second
    /// pipeline layout for one temporary colour.
    ///
    /// Returns how many indexes it wrote, which is what the frame's phase note
    /// says beside the time: this runs on every frame for every node, so zero
    /// is the usual answer and is the one that means nothing happened.
    pub fn update_point_mask(
        &mut self,
        queue: &wgpu::Queue,
        id: ReconId,
        deleted: &HashSet<u32>,
        highlighted: &HashSet<u32>,
    ) -> usize {
        let Some(bundle) = self.recons.get_mut(&id) else {
            return 0;
        };
        if bundle.masked_deleted == *deleted && bundle.masked_highlighted == *highlighted {
            return 0;
        }
        let mut changed: Vec<u32> = bundle
            .masked_deleted
            .symmetric_difference(deleted)
            .copied()
            .collect();
        changed.extend(
            bundle
                .masked_highlighted
                .symmetric_difference(highlighted)
                .copied(),
        );
        changed.sort_unstable();
        changed.dedup();
        let written = changed.len();
        for index in changed {
            let alive: u32 = if deleted.contains(&index) {
                MASK_DELETED
            } else if highlighted.contains(&index) {
                MASK_HIGHLIGHTED
            } else {
                MASK_ALIVE
            };
            if index < bundle.point_count {
                if let Some(buffer) = &bundle.point_alive_buffer {
                    queue.write_buffer(buffer, u64::from(index) * 4, bytemuck::bytes_of(&alive));
                }
            }
            if let Some(patch) = &bundle.patch {
                if let Some(&slot) = patch.slot_of_point.get(&index) {
                    queue.write_buffer(
                        &patch.alive_buffer,
                        u64::from(slot) * 4,
                        bytemuck::bytes_of(&alive),
                    );
                }
            }
            // An addition's own instance, whose row is its edited index less
            // the base's point count. Its surfel is keyed on the edited index,
            // like the base's, because the instances carry that.
            if let Some(additions) = &bundle.additions {
                if let Some(row) = index.checked_sub(bundle.point_count) {
                    if row < additions.point_count {
                        queue.write_buffer(
                            &additions.point_alive_buffer,
                            u64::from(row) * 4,
                            bytemuck::bytes_of(&alive),
                        );
                    }
                }
                if let Some(patch) = &additions.patch {
                    if let Some(&slot) = patch.slot_of_point.get(&index) {
                        queue.write_buffer(
                            &patch.alive_buffer,
                            u64::from(slot) * 4,
                            bytemuck::bytes_of(&alive),
                        );
                    }
                }
            }
        }
        bundle.masked_deleted = deleted.clone();
        bundle.masked_highlighted = highlighted.clone();
        written
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
