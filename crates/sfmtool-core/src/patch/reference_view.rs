// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The per-view measurements that say what each view of a track can
//! contribute to its patch bitmap, and the reference-view rule, which picks
//! the one view whose tile could stand as the bitmap.
//!
//! `specs/core/patch/reference-view.md` is the design. The measurements are
//! taken on each view's `R×R` tile ([`render_view_tile`]): its coverage, the
//! clipped share of the photograph under it ([`clipped_share`]) and the angle
//! the view sees the patch at
//! ([`viewing_angle`](crate::patch::normal_refine::viewing_angle)); and across
//! the views, the ZNCC between each pair over each cell of the ZNCC grid
//! ([`cell_agreement`]). The rule ([`choose_reference_view`]) is a pure
//! function over those readings, the median pairwise ZNCC from
//! [member coherence](crate::patch::member_coherence) and each tile's
//! self-similarity ellipse.
//!
//! [`read_track`] takes those readings over a track's views' tiles and runs the
//! rule; the bench reports its pick beside its other readings, and the stored
//! patch bitmap is the picked view's tile
//! ([`crate::patch::stored_bitmap`]). Every reading is taken on the `R×R`
//! tiles and nothing else: no coarser grid, and no pixels of the photograph
//! outside the tile.

mod agreement;
mod rule;
mod tile;
mod track;

#[cfg(test)]
mod tests;

pub use agreement::{cell_agreement, pair_zncc_grid, CellAgreement};
#[cfg(test)]
pub(crate) use agreement::{cell_agreement_from_pairs, finite_middle};
pub use rule::{
    choose_reference_view, ReferenceChoice, ReferenceCost, ReferenceExclusion, ReferenceReadings,
    ReferenceStanding, ReferenceTerm, ReferenceViewOptions,
};
pub use tile::{clipped_share, render_view_tile, ViewTile};
pub use track::{read_track, readings_of, tile_max_radius, tile_semi_axes, TrackReading};

/// The typical agreement a cell needs before the cell check judges it.
///
/// A cell where the track's views do not agree, because it holds no texture
/// they share, says nothing about any one view.
pub const REFERENCE_MIN_JUDGED_CELL_ZNCC: f64 = 0.5;

/// The fewest samples with data in both tiles a cell of [`pair_zncc_grid`] is
/// correlated over. Below it a ZNCC is read off a handful of samples.
pub const REFERENCE_MIN_CELL_SAMPLES: usize = 16;
