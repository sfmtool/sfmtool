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
//! and patch shaders through [`SceneRenderer::update_deleted_mask`], as one
//! `u32` per instance written only where the set moved.

use std::collections::HashSet;
use std::sync::Arc;

use super::super::SceneRenderer;
use crate::scene::ReconId;
use sfmtool_core::SfmrReconstruction;

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

    /// Record which base `id`'s buffers now hold. Called by the upload phase
    /// once the three per-node uploads have run.
    pub fn set_uploaded_base(&mut self, id: ReconId, base: Arc<SfmrReconstruction>) {
        if let Some(bundle) = self.recons.get_mut(&id) {
            bundle.uploaded_base = Some(base);
        }
    }

    /// Bring `id`'s deleted mask in line with `deleted`.
    ///
    /// Writes one `u32` per index that entered or left the set, in both the
    /// point buffer and -- for a point that carries a surfel -- the patch
    /// buffer, so an edit costs its own size rather than the node's. An index at
    /// or above the base's point count is an addition, and is masked in the
    /// additions' own buffers, which the overlay owns alongside the mask.
    pub fn update_deleted_mask(
        &mut self,
        queue: &wgpu::Queue,
        id: ReconId,
        deleted: &HashSet<u32>,
    ) {
        let Some(bundle) = self.recons.get_mut(&id) else {
            return;
        };
        if bundle.masked_deleted == *deleted {
            return;
        }
        let changed: Vec<u32> = bundle
            .masked_deleted
            .symmetric_difference(deleted)
            .copied()
            .collect();
        for index in changed {
            let alive: u32 = u32::from(!deleted.contains(&index));
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
