// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Where in the tile each view agrees with the others: the ZNCC between every
//! pair of views over each cell of the ZNCC grid's three-by-three split, each
//! view's median over the others, and how far that falls below the track's
//! typical agreement, the median over the views, in its worst cell.
//!
//! A view's cell is judged only where every one of its samples is on the
//! photograph and it holds texture by its self-similarity reading; a cell with
//! a sample off the photograph is also left out of the typical agreement
//! ([`cell_agreement`]).

use super::tile::ViewTile;
use super::track::{tile_max_radius, TileSemiAxes};
use super::{REFERENCE_MIN_CELL_SAMPLES, REFERENCE_MIN_JUDGED_CELL_ZNCC};
use crate::numeric::median_in_place;
use crate::patch::normal_refine::grid_bounds;

/// Each view's agreement with the others over the ZNCC grid's cells.
#[derive(Debug, Clone, PartialEq)]
pub struct CellAgreement {
    /// Per view, per cell (`grid[row][col]` from the top-left cell): the median
    /// over the other views of the pair's ZNCC in that cell, `NaN` where no
    /// other view could be correlated with it there.
    pub pair_zncc_grid: Vec<[[f64; 3]; 3]>,
    /// Per cell: the track's typical agreement there, the median of
    /// [`Self::pair_zncc_grid`] over the views whose tile has every sample of
    /// the cell on the photograph, `NaN` where no such view has a reading.
    pub typical: [[f64; 3]; 3],
    /// Per view, per cell: whether the cell check judges the view there: the
    /// typical agreement is at least [`REFERENCE_MIN_JUDGED_CELL_ZNCC`], and
    /// the view's own tile has every sample of the cell on the photograph and
    /// holds texture there.
    pub judged: Vec<[[bool; 3]; 3]>,
    /// Per view: the cell deficit, the largest amount by which the view's
    /// agreement falls below the typical agreement, over the cells it is
    /// judged in ([`Self::judged`]). `0` where no cell is judged, and `NaN`
    /// where the view has no reading in any judged cell.
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
/// ([`CellAgreement`]), from `tiles`, one per view, and `semi_axes`, each
/// tile's self-similarity semi-axes ([`tile_semi_axes`](super::tile_semi_axes)).
///
/// Every pair is correlated once ([`pair_zncc_grid`]), `k(k − 1)/2` pairs of
/// nine cells, on tiles already rendered.
///
/// Two kinds of view-cell are not judged:
///
/// - **No texture** (`cells_with_texture`): the cell's self-similarity
///   ellipse reads the cap of its reading ([`tile_max_radius`]) along both
///   axes, so the cell matches itself shifted anywhere in the search, and a
///   ZNCC against another view there reads noise or shading. The view's
///   reading still counts in the typical agreement.
/// - **Off the photograph** (`cells_on_photograph`): some of the cell's
///   samples are off the photograph, so its ZNCCs are read over part of the
///   cell. The coverage penalty already charges the view for the samples it
///   lacks. Its reading is also left out of the typical agreement.
///
/// A view with no self-similarity reading counts as holding texture in every
/// cell.
///
/// # Panics
///
/// Panics if `semi_axes` is not parallel to `tiles`.
pub fn cell_agreement(tiles: &[&ViewTile], semi_axes: &[Option<TileSemiAxes>]) -> CellAgreement {
    let k = tiles.len();
    assert_eq!(
        semi_axes.len(),
        k,
        "cell_agreement: tiles and semi_axes must be parallel"
    );
    let mut pairs = vec![[[f64::NAN; 3]; 3]; k * k];
    for a in 0..k {
        for b in (a + 1)..k {
            let grid = pair_zncc_grid(tiles[a], tiles[b]);
            pairs[a * k + b] = grid;
            pairs[b * k + a] = grid;
        }
    }
    let on_photograph: Vec<[[bool; 3]; 3]> =
        tiles.iter().map(|tile| cells_on_photograph(tile)).collect();
    let textured: Vec<[[bool; 3]; 3]> = semi_axes
        .iter()
        .map(|axes| cells_with_texture(axes.as_ref()))
        .collect();
    cell_agreement_from_pairs(&pairs, k, &on_photograph, &textured)
}

/// Per cell of `tile`'s split, whether every sample of the cell is on the
/// photograph.
pub(crate) fn cells_on_photograph(tile: &ViewTile) -> [[bool; 3]; 3] {
    let r = tile.resolution();
    let bounds = grid_bounds(r as u32);
    std::array::from_fn(|i| {
        std::array::from_fn(|j| {
            (bounds[i]..bounds[i + 1])
                .all(|row| (bounds[j]..bounds[j + 1]).all(|col| tile.valid[row * r + col]))
        })
    })
}

/// Per cell, whether it holds texture by the cell's self-similarity ellipse
/// in `semi_axes`: its semi-minor axis is shorter than the cap of the reading
/// ([`tile_max_radius`]). A cell along a straight edge reads at the cap along
/// the edge only and holds texture, since a view that disagrees across the
/// edge is told apart there. A cell with no reading, and every cell where
/// `semi_axes` is `None`, counts as holding texture.
pub(crate) fn cells_with_texture(semi_axes: Option<&TileSemiAxes>) -> [[bool; 3]; 3] {
    let cap = tile_max_radius();
    std::array::from_fn(|i| {
        std::array::from_fn(|j| {
            semi_axes.is_none_or(|axes| axes.grid[i][j][1].is_nan() || axes.grid[i][j][1] < cap)
        })
    })
}

/// [`cell_agreement`] from the pairs' grids already computed: `pairs` is the
/// row-major `k×k` table of [`pair_zncc_grid`] readings, whose diagonal is
/// not read, and `on_photograph` and `textured` each view's
/// [`cells_on_photograph`] and [`cells_with_texture`].
pub(crate) fn cell_agreement_from_pairs(
    pairs: &[[[f64; 3]; 3]],
    k: usize,
    on_photograph: &[[[bool; 3]; 3]],
    textured: &[[[bool; 3]; 3]],
) -> CellAgreement {
    debug_assert_eq!(pairs.len(), k * k, "pairs must be a k*k row-major table");
    debug_assert!(
        on_photograph.len() == k && textured.len() == k,
        "on_photograph and textured must hold one grid per view"
    );
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
            let views: Vec<f64> = (0..k)
                .filter(|&v| on_photograph[v][i][j])
                .map(|v| pair_zncc_grid[v][i][j])
                .collect();
            *cell = finite_middle(&views);
        }
    }
    let judged: Vec<[[bool; 3]; 3]> = (0..k)
        .map(|v| {
            std::array::from_fn(|i| {
                std::array::from_fn(|j| {
                    on_photograph[v][i][j]
                        && textured[v][i][j]
                        && typical[i][j] >= REFERENCE_MIN_JUDGED_CELL_ZNCC
                })
            })
        })
        .collect();
    let deficit = pair_zncc_grid
        .iter()
        .zip(&judged)
        .map(|(grid, judged)| {
            let cells: Vec<(usize, usize)> = (0..3)
                .flat_map(|i| (0..3).map(move |j| (i, j)))
                .filter(|&(i, j)| judged[i][j])
                .collect();
            if cells.is_empty() {
                return 0.0;
            }
            cells
                .iter()
                .map(|&(i, j)| typical[i][j] - grid[i][j])
                .filter(|d| d.is_finite())
                .fold(f64::NAN, f64::max)
        })
        .collect();
    CellAgreement {
        pair_zncc_grid,
        typical,
        judged,
        deficit,
    }
}

/// The median ([`median_in_place`]) of the finite values, `NaN` when there are
/// none: a reading that is missing is left out rather than counted.
pub(crate) fn finite_middle(values: &[f64]) -> f64 {
    let mut finite: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    median_in_place(&mut finite)
}
