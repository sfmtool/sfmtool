// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The reference-view rule over one track: the readings across its views'
//! tiles, and the view the rule picks from them.

use super::agreement::{cell_agreement, finite_middle, CellAgreement};
use super::tile::ViewTile;
use super::{choose_reference_view, ReferenceChoice, ReferenceReadings};
use crate::camera::sampler::SamplerChoice;
use crate::patch::cloud::OrientedPatch;
use crate::patch::member_coherence::{member_zncc_matrix_reporting, MemberCoherenceParams};
use crate::patch::normal_refine::ProjectedImage;
use crate::patch::self_similarity::{zncc_self_similarity_parts, PatchTile, SelfSimilarityParams};
use crate::progress::Progress;

/// What the reference-view rule read across one track's views, and what it
/// picked ([`read_track`]).
#[derive(Debug, Clone, PartialEq)]
pub struct TrackReading {
    /// Per view: the median of its pairwise ZNCCs with the other views, from
    /// member coherence's matrix; `None` where it has none.
    pub pair_zncc: Vec<Option<f64>>,
    /// Each view's agreement with the others over the ZNCC grid's cells.
    pub cells: CellAgreement,
    /// Per view: what the rule read of it.
    pub readings: Vec<ReferenceReadings>,
    /// What the rule decided.
    pub choice: ReferenceChoice,
}

/// What the rule reads of one view: the coverage, clipped share and viewing
/// angle `tile` was read with, its pair ZNCC and cell deficit, and its
/// self-similarity ellipse's semi-axes `[major, minor]` in grid px.
pub fn readings_of(
    tile: &ViewTile,
    pair_zncc: Option<f64>,
    cell_deficit: Option<f64>,
    semi_axes: Option<[f64; 2]>,
) -> ReferenceReadings {
    ReferenceReadings {
        coverage: Some(tile.coverage),
        clipped_share: tile.clipped_share,
        viewing_angle_deg: tile.viewing_angle.map(|a| a.angle_deg),
        cell_deficit,
        pair_zncc,
        semi_major: semi_axes.map(|a| a[0]),
        semi_minor: semi_axes.map(|a| a[1]),
    }
}

/// The semi-axes `[major, minor]`, in grid px, of the self-similarity ellipse
/// of the whole of `tile`, read the overlap way over its samples on the
/// photograph with the default parameters: the reading the bench takes of
/// each view's tile and the rule ranks the candidates by. `None` for a tile
/// under 3 a side.
pub fn tile_semi_axes(tile: &ViewTile) -> Option<[f64; 2]> {
    let resolution = tile.resolution();
    let channels = tile.channels();
    if channels == 0 || resolution < 3 {
        return None;
    }
    let samples: Vec<f32> = tile.samples.iter().map(|&v| f32::from(v)).collect();
    let (planes, colour) =
        PatchTile::planes_from_interleaved(&samples, resolution, resolution, channels);
    let parts = zncc_self_similarity_parts(
        &PatchTile {
            values: &planes,
            channels: colour,
            width: resolution,
            height: resolution,
        },
        Some(&tile.valid),
        &SelfSimilarityParams::default(),
    );
    Some(parts.whole.ellipse.axes)
}

/// Take the readings the reference-view rule needs across a track's views and
/// run the rule ([`choose_reference_view`]).
///
/// View `v` is image `members[v]` of `views`, its keypoint `keypoints[v]`;
/// `tiles[v]` is its `R×R` tile ([`render_view_tile`](super::render_view_tile)
/// of `patch` at that keypoint), and `semi_axes[v]` its tile's self-similarity
/// semi-axes ([`tile_semi_axes`]).
///
/// The pair ZNCC is the median of the view's row of member coherence's matrix
/// ([`member_zncc_matrix_reporting`]), rendered over the views' common support
/// with member coherence's window at the tiles' resolution `R` and with
/// `sampler`, anchored at each view's keypoint. The cell readings are taken on
/// the tiles themselves ([`cell_agreement`]): member coherence renders only
/// the samples inside its window's disk and common to every member, so its
/// corner cells would hold part of their square, and the cell check was
/// measured on whole cells with each pair's own support. Nothing coarser than
/// `R`, and nothing outside the tile, is read.
///
/// Two views of one image share the matrix row of the first of them, as
/// member coherence counts an image once.
///
/// # Panics
///
/// Panics if `members`, `keypoints`, `tiles` and `semi_axes` are not parallel,
/// or the tiles differ in resolution or channels.
#[allow(clippy::too_many_arguments)]
pub fn read_track(
    patch: &OrientedPatch,
    views: &[ProjectedImage<'_>],
    members: &[u32],
    keypoints: &[Option<[f64; 2]>],
    tiles: &[&ViewTile],
    semi_axes: &[Option<[f64; 2]>],
    sampler: SamplerChoice,
    progress: &Progress<'_>,
) -> TrackReading {
    let k = members.len();
    assert!(
        keypoints.len() == k && tiles.len() == k && semi_axes.len() == k,
        "read_track: members, keypoints, tiles and semi_axes must be parallel"
    );
    let resolution = tiles.first().map_or(2, |t| t.resolution()) as u32;
    let params = MemberCoherenceParams {
        resolution: resolution.max(2),
        sampler,
        ..MemberCoherenceParams::default()
    };
    let matrix =
        member_zncc_matrix_reporting(patch, views, members, Some(keypoints), &params, progress);
    let pair_zncc: Vec<Option<f64>> = members
        .iter()
        .map(|image| {
            let row = matrix.members.iter().position(|m| m == image)?;
            let others: Vec<f64> = (0..matrix.len())
                .filter(|&j| j != row)
                .map(|j| matrix.get(row, j))
                .collect();
            let middle = finite_middle(&others);
            middle.is_finite().then_some(middle)
        })
        .collect();
    let cells = cell_agreement(tiles);
    let readings: Vec<ReferenceReadings> = (0..k)
        .map(|v| {
            let deficit = cells.deficit[v];
            readings_of(
                tiles[v],
                pair_zncc[v],
                deficit.is_finite().then_some(deficit),
                semi_axes[v],
            )
        })
        .collect();
    let choice = choose_reference_view(&readings);
    TrackReading {
        pair_zncc,
        cells,
        readings,
        choice,
    }
}
