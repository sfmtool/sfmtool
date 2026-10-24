// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! **Filter the 3D view**: the Track Table's matched set applied to the
//! selected reconstruction's point cloud.
//!
//! See `specs/gui/track-table.md` § "Filtering the 3D view". The Track Table
//! panel owns the toggle, which is panel state like its query, and publishes
//! what it asks of the 3D view as a [`Filter3d`] on
//! [`AppState::filter_3d`](crate::state::AppState::filter_3d). The frame's
//! upload phase hands the kept set to the scene renderer, which carries it as
//! one bit of each point's mask word
//! ([`SceneRenderer::update_point_mask`](crate::scene_renderer::SceneRenderer::update_point_mask)),
//! and the viewport's stats line counts the points it keeps.

use std::sync::Arc;

use crate::document::VersionSerial;
use crate::scene::ReconId;

/// How the 3D view draws a point the Track Table's filter rejects.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum Filter3dMode {
    /// At [`DIM_ALPHA`], after the points the filter keeps, and still picked.
    #[default]
    Dim,
    /// Not at all: out of the draw and out of picking.
    Hide,
}

/// The alpha a rejected point is drawn at under [`Filter3dMode::Dim`]. Its
/// WGSL twin is `DIM_ALPHA` in `shaders/common.wgsl`.
/// Read by the pipeline test that holds the two equal.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) const DIM_ALPHA: f32 = 0.25;

/// The points the filter keeps, as one bit per edited point index.
///
/// Built from the Track Table's row list when that list changes, never per
/// frame. The bits are in the index space of the version the row list was
/// computed for, and are applied to that version only ([`Filter3d::kept`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct KeptPoints {
    /// Bit `i % 64` of word `i / 64` is set when point `i` is kept.
    words: Vec<u64>,
    /// How many points are kept.
    count: usize,
}

impl KeptPoints {
    /// The set holding `points`, which are distinct point indexes in any
    /// order.
    pub(crate) fn from_points(points: &[u32]) -> Self {
        let len = points.iter().max().map_or(0, |&max| max as usize / 64 + 1);
        let mut words = vec![0u64; len];
        for &point in points {
            words[point as usize / 64] |= 1 << (point % 64);
        }
        Self {
            words,
            count: points.len(),
        }
    }

    /// Whether point `index` is kept. An index past the last kept point is
    /// not.
    pub(crate) fn contains(&self, index: u32) -> bool {
        self.words
            .get(index as usize / 64)
            .is_some_and(|word| word & (1 << (index % 64)) != 0)
    }

    /// How many points are kept.
    pub(crate) fn count(&self) -> usize {
        self.count
    }
}

/// What the Track Table asks of the 3D view while its *Filter the 3D view*
/// box is ticked and the panel is open.
#[derive(Debug, Clone)]
pub(crate) struct Filter3d {
    /// The reconstruction the table shows, the only one the filter touches.
    pub(crate) node: ReconId,
    /// Dim or hide.
    pub(crate) mode: Filter3dMode,
    /// The points kept and the version their indexes belong to, or `None`
    /// while the table has no row list for the version at the node's cursor:
    /// after an edit until the new table lands, or before the first one has.
    /// The 3D view then draws every point.
    pub(crate) kept: Option<(VersionSerial, Arc<KeptPoints>)>,
}

impl Filter3d {
    /// The kept set, when it was built for `document`, the serial of the
    /// version at the node's cursor. A set built for any other version is
    /// never applied, since point indexes can shift across edits.
    pub(crate) fn kept_for(&self, document: VersionSerial) -> Option<&Arc<KeptPoints>> {
        self.kept
            .as_ref()
            .filter(|(serial, _)| *serial == document)
            .map(|(_, kept)| kept)
    }
}

#[cfg(test)]
mod tests {
    use super::KeptPoints;

    #[test]
    fn kept_points_hold_exactly_the_points_given() {
        let kept = KeptPoints::from_points(&[130, 0, 63, 64]);
        assert_eq!(kept.count(), 4);
        for index in 0..200 {
            assert_eq!(
                kept.contains(index),
                [0, 63, 64, 130].contains(&index),
                "{index}"
            );
        }
        let none = KeptPoints::from_points(&[]);
        assert_eq!(none.count(), 0);
        assert!(!none.contains(0));
    }
}
