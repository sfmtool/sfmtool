// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Pick-buffer encoding, and the tables that decode an id back to a typed ref.
//!
//! The pick target is a single `R32Uint` texture, so one `u32` has to carry
//! both *what kind* of entity was hit and *which* entity, across every loaded
//! reconstruction. See `specs/gui/scene-graph.md` ("Picking"):
//!
//! ```text
//! bits 31..30  tag: 0 = none, 1 = frustum/camera, 2 = point   (3 reserved)
//! bits 29..0   global index: recon pick base + local index    (2^30 ≈ 1.07B)
//! ```
//!
//! Each loaded reconstruction owns a contiguous range in two independent index
//! spaces — one for points, one for images. The base arrives in the shader via
//! the per-recon uniform block, so **instance buffers store nothing new** and
//! never need rewriting when bases move.

use crate::scene::{ImageRef, PointRef, ReconId};

/// Bit position of the 2-bit entity tag.
pub const PICK_TAG_SHIFT: u32 = 30;

/// Pick ID for "nothing" (background / no entity).
pub const PICK_TAG_NONE: u32 = 0;
/// Pick ID tag for frustum / camera image entities.
pub const PICK_TAG_FRUSTUM: u32 = 1 << PICK_TAG_SHIFT;
/// Pick ID tag for 3D point entities.
pub const PICK_TAG_POINT: u32 = 2 << PICK_TAG_SHIFT;
/// Mask to extract the entity type tag (top 2 bits).
pub const PICK_TAG_MASK: u32 = 0b11 << PICK_TAG_SHIFT;
/// Mask to extract the global entity index (bottom 30 bits).
pub const PICK_INDEX_MASK: u32 = !PICK_TAG_MASK;

/// One past the largest addressable global index, per entity kind.
pub const PICK_INDEX_CAPACITY: u32 = PICK_INDEX_MASK + 1;

/// Uniform sentinel for "no selection / no hover". Outside the 30-bit index
/// space, so it can never equal a real global index.
pub const PICK_INDEX_NONE: u32 = u32::MAX;

/// What a pick id resolved to: an entity in a specific reconstruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PickTarget {
    Image(ImageRef),
    Point(PointRef),
}

/// Encode a tag and a global index into a pick id, the way the shaders do.
///
/// Test-only: in the app the pick id is only ever *produced* on the GPU (five
/// WGSL fragment shaders write it) and only ever *consumed* through
/// [`PickTables::resolve`]. This exists so a round-trip is assertable from the
/// CPU without a render pass.
#[cfg(test)]
pub fn encode(tag: u32, global_index: u32) -> u32 {
    (tag & PICK_TAG_MASK) | (global_index & PICK_INDEX_MASK)
}

/// One reconstruction's slice of one global index space.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PickRange {
    base: u32,
    count: u32,
    recon: ReconId,
}

/// The sorted `(base, ReconId)` tables a pick id is decoded against.
///
/// Rebuilt whenever a node is added or removed; kept sorted by `base` (which,
/// since bases are handed out in `ReconId` order, is also load order) so decode
/// is a binary search rather than a scan.
#[derive(Default)]
pub(super) struct PickTables {
    points: Vec<PickRange>,
    images: Vec<PickRange>,
}

impl PickTables {
    /// Drop every range, ready for a fresh assignment pass.
    pub(super) fn clear(&mut self) {
        self.points.clear();
        self.images.clear();
    }

    /// Append one reconstruction's ranges. Callers must push in increasing base
    /// order (i.e. iterate nodes in `ReconId` order), which keeps both tables
    /// sorted without a re-sort.
    pub(super) fn push(
        &mut self,
        recon: ReconId,
        point_base: u32,
        point_count: u32,
        image_base: u32,
        image_count: u32,
    ) {
        self.points.push(PickRange {
            base: point_base,
            count: point_count,
            recon,
        });
        self.images.push(PickRange {
            base: image_base,
            count: image_count,
            recon,
        });
    }

    /// Decode a pick id read back from the GPU.
    ///
    /// Returns `None` for the background value, for the reserved tag, and for
    /// an index that falls outside every assigned range (a stale readback from
    /// a node that has since been released).
    pub(super) fn resolve(&self, pick_id: u32) -> Option<PickTarget> {
        let index = pick_id & PICK_INDEX_MASK;
        match pick_id & PICK_TAG_MASK {
            PICK_TAG_POINT => {
                let (recon, local) = lookup(&self.points, index)?;
                Some(PickTarget::Point(PointRef::new(recon, local)))
            }
            PICK_TAG_FRUSTUM => {
                let (recon, local) = lookup(&self.images, index)?;
                Some(PickTarget::Image(ImageRef::new(recon, local)))
            }
            // PICK_TAG_NONE, and the reserved tag 3 (which is what the
            // all-ones sentinel decodes to).
            _ => None,
        }
    }
}

/// Binary-search `ranges` (sorted by base) for the range containing `index`.
fn lookup(ranges: &[PickRange], index: u32) -> Option<(ReconId, usize)> {
    let i = ranges.partition_point(|r| r.base <= index).checked_sub(1)?;
    let range = ranges[i];
    let local = index - range.base;
    (local < range.count).then_some((range.recon, local as usize))
}

#[cfg(test)]
mod tests;
