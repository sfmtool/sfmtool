// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Where in the tile each view agrees with the others: the ZNCC between every
//! pair of views over each cell of the ZNCC grid's three-by-three split, each
//! view's median over the others, and how far that falls below the track's
//! typical agreement, the median over the views, in its worst cell.

use super::pair_readings::{blur_matched_pairs, BlurMatchedPairs};
use super::tile::ViewTile;
use super::{REFERENCE_MIN_CELL_SAMPLES, REFERENCE_MIN_JUDGED_CELL_ZNCC};
use crate::numeric::median_in_place;
use crate::patch::blur_matched::TilePlanes;
use crate::patch::normal_refine::{grid_bounds, PatchWindow};
use crate::patch::pair_sharpness::PairMatching;
use crate::progress::Progress;

/// Each view's agreement with the others over the ZNCC grid's cells.
#[derive(Debug, Clone, PartialEq)]
pub struct CellAgreement {
    /// Per view, per cell (`grid[row][col]` from the top-left cell): the median
    /// over the other views of the pair's ZNCC in that cell, `NaN` where no
    /// other view could be correlated with it there.
    pub pair_zncc_grid: Vec<[[f64; 3]; 3]>,
    /// Per cell: the track's typical agreement there, the median of
    /// [`Self::pair_zncc_grid`] over the views, `NaN` where no view has a
    /// reading.
    pub typical: [[f64; 3]; 3],
    /// Per view: the cell deficit, the largest amount by which the view's
    /// agreement falls below the typical agreement, over the cells whose
    /// typical agreement is at least [`REFERENCE_MIN_JUDGED_CELL_ZNCC`]. `0`
    /// where no cell is judged, and `NaN` where the view has no reading in any
    /// judged cell.
    pub deficit: Vec<f64>,
}

/// The ZNCC of two views' tiles over each cell of the ZNCC grid's split, rows
/// and columns cut at `R/3` and `R − R/3` (`grid_bounds`), `grid[row][col]`
/// from the top-left cell.
///
/// Each cell is read over the samples that carry image data in both tiles,
/// with every sample weighted equally, per colour channel, and the channels'
/// ZNCCs averaged. A channel flat in either tile over the cell is left out of
/// the average, and a cell is `NaN` where no channel is left or fewer than
/// [`REFERENCE_MIN_CELL_SAMPLES`] samples carry data in both.
///
/// # Panics
///
/// Panics if the two tiles differ in resolution or channel count.
pub fn pair_zncc_grid(a: &ViewTile, b: &ViewTile) -> [[f64; 3]; 3] {
    let r = a.resolution();
    let channels = a.channels().min(3);
    assert_eq!(r, b.resolution(), "the tiles differ in resolution");
    assert_eq!(a.channels(), b.channels(), "the tiles differ in channels");
    let bounds = grid_bounds(r as u32);
    let mut grid = [[f64::NAN; 3]; 3];
    for (i, grid_row) in grid.iter_mut().enumerate() {
        for (j, cell) in grid_row.iter_mut().enumerate() {
            let mut total = 0.0;
            let mut used = 0usize;
            for c in 0..channels {
                let (mut sa, mut sb, mut saa, mut sbb, mut sab, mut n) =
                    (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64, 0usize);
                for row in bounds[i]..bounds[i + 1] {
                    for col in bounds[j]..bounds[j + 1] {
                        let k = row * r + col;
                        if !(a.valid[k] && b.valid[k]) {
                            continue;
                        }
                        let x = f64::from(a.samples[[row, col, c]]);
                        let y = f64::from(b.samples[[row, col, c]]);
                        sa += x;
                        sb += y;
                        saa += x * x;
                        sbb += y * y;
                        sab += x * y;
                        n += 1;
                    }
                }
                if n < REFERENCE_MIN_CELL_SAMPLES {
                    continue;
                }
                let n = n as f64;
                let va = saa - sa * sa / n;
                let vb = sbb - sb * sb / n;
                if va <= 1e-6 || vb <= 1e-6 {
                    continue;
                }
                total += (sab - sa * sb / n) / (va * vb).sqrt();
                used += 1;
            }
            if used > 0 {
                *cell = total / used as f64;
            }
        }
    }
    grid
}

/// Each view's agreement with the other views over the ZNCC grid's cells
/// ([`CellAgreement`]), from `tiles`, one per view.
///
/// Every pair is correlated once ([`pair_zncc_grid`]), `k(k − 1)/2` pairs of
/// nine cells, on tiles already rendered.
pub fn cell_agreement(tiles: &[&ViewTile]) -> CellAgreement {
    let k = tiles.len();
    let mut pairs = vec![[[f64::NAN; 3]; 3]; k * k];
    for a in 0..k {
        for b in (a + 1)..k {
            let grid = pair_zncc_grid(tiles[a], tiles[b]);
            pairs[a * k + b] = grid;
            pairs[b * k + a] = grid;
        }
    }
    cell_agreement_from_pairs(&pairs, k)
}

/// [`cell_agreement`] from the pairs' grids already computed: `pairs` is the
/// row-major `k×k` table of [`pair_zncc_grid`] readings, whose diagonal is
/// not read.
///
/// # Panics
///
/// Panics if `pairs` is not `k×k`.
pub fn cell_agreement_from_pairs(pairs: &[[[f64; 3]; 3]], k: usize) -> CellAgreement {
    assert_eq!(pairs.len(), k * k, "pairs must be a k*k row-major table");
    let pair_zncc_grid: Vec<[[f64; 3]; 3]> = (0..k)
        .map(|v| {
            let mut grid = [[f64::NAN; 3]; 3];
            for (i, row) in grid.iter_mut().enumerate() {
                for (j, cell) in row.iter_mut().enumerate() {
                    let others: Vec<f64> = (0..k)
                        .filter(|&w| w != v)
                        .map(|w| pairs[v * k + w][i][j])
                        .collect();
                    *cell = finite_middle(&others);
                }
            }
            grid
        })
        .collect();
    let mut typical = [[f64::NAN; 3]; 3];
    for (i, row) in typical.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            let views: Vec<f64> = pair_zncc_grid.iter().map(|g| g[i][j]).collect();
            *cell = finite_middle(&views);
        }
    }
    let judged: Vec<(usize, usize)> = (0..3)
        .flat_map(|i| (0..3).map(move |j| (i, j)))
        .filter(|&(i, j)| typical[i][j] >= REFERENCE_MIN_JUDGED_CELL_ZNCC)
        .collect();
    let deficit = pair_zncc_grid
        .iter()
        .map(|grid| {
            if judged.is_empty() {
                return 0.0;
            }
            judged
                .iter()
                .map(|&(i, j)| typical[i][j] - grid[i][j])
                .filter(|d| d.is_finite())
                .fold(f64::NAN, f64::max)
        })
        .collect();
    CellAgreement {
        pair_zncc_grid,
        typical,
        deficit,
    }
}

/// The blur-matched readings of a track's views: each view's median
/// blur-matched ZNCC with the others over the whole tile, and its agreement
/// over the ZNCC grid's cells on the same blur-matched pairs.
#[derive(Debug, Clone, PartialEq)]
pub struct BlurMatchedAgreement {
    /// Per view: the median of its blur-matched whole-tile ZNCCs with the
    /// other views, `NaN` where it has none.
    pub pair_zncc: Vec<f64>,
    /// The cell readings ([`cell_agreement_from_pairs`]) of the blur-matched
    /// pair grids.
    pub cells: CellAgreement,
    /// The pairs themselves, with how many were blurred.
    pub pairs: BlurMatchedPairs,
}

/// [`BlurMatchedAgreement`] of `tiles`, one per view, with `ellipses[v]` view
/// `v`'s self-similarity ellipse matrix in grid px² (`None` where it has
/// none, which leaves its pairs plain), each pair matched under `matching`
/// ([`blur_matched_pairs`]).
///
/// The whole-tile reading is windowed by `window` (member coherence's
/// default is the reference rule's), over the samples with data in both tiles
/// of each pair; the cells are read as [`pair_zncc_grid`] reads them. With
/// [`PairMatching::Plain`] the cells are [`cell_agreement`]'s.
///
/// # Panics
///
/// Panics if `ellipses` is not parallel to `tiles`, or the tiles differ in
/// resolution.
pub fn blur_matched_agreement(
    tiles: &[&ViewTile],
    ellipses: &[Option<[[f64; 2]; 2]>],
    matching: PairMatching,
    window: PatchWindow,
    progress: &Progress<'_>,
) -> BlurMatchedAgreement {
    let planes: Vec<TilePlanes> = tiles.iter().map(|t| t.planes()).collect();
    let refs: Vec<&TilePlanes> = planes.iter().collect();
    let pairs = blur_matched_pairs(&refs, ellipses, matching, window, progress);
    let k = tiles.len();
    let pair_zncc = (0..k).map(|v| pairs.row_middle(v)).collect();
    let cells = cell_agreement_from_pairs(&pairs.grid, k);
    BlurMatchedAgreement {
        pair_zncc,
        cells,
        pairs,
    }
}

/// The median ([`median_in_place`]) of the finite values, `NaN` when there are
/// none: a reading that is missing is left out rather than counted.
pub(crate) fn finite_middle(values: &[f64]) -> f64 {
    let mut finite: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    median_in_place(&mut finite)
}
