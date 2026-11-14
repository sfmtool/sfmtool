// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Observation coverage: which image pixels existing tracks already claim.
//!
//! [`ObservationCoverage::build`] rasterizes every observation's image-space
//! footprint — a disk at its keypoint, with the radius supplied per observation —
//! into per-image grids of saturating counts once, and the batch queries answer
//! against them.
//!
//! See `specs/core/analysis/observation-coverage.md` for the design.

use std::f64::consts::TAU;

use rayon::prelude::*;

/// The spec's default cell size in pixels.
pub const DEFAULT_CELL_PX: u32 = 4;

/// The most sectors [`ObservationCoverage::uncovered_sectors`] can report, set
/// by the width of the bitmask it returns.
pub const MAX_SECTORS: u32 = 32;

/// The most cells one image's grid may have: 2³⁰, one GiB of counts, which an
/// image 131,072 pixels square reaches at the default 4-pixel cell.
/// [`ObservationCoverage::build`] refuses a larger grid rather than ask for
/// the memory.
pub const MAX_GRID_CELLS: u64 = 1 << 30;

/// What [`ObservationCoverage`] refuses.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ObservationCoverageError {
    /// `cell_px` is zero.
    ZeroCellPx,
    /// A slice does not have one entry per row of the slice it runs parallel
    /// to.
    Length {
        /// The slice with the wrong length.
        name: &'static str,
        /// Its length.
        len: usize,
        /// The slice it runs parallel to.
        against: &'static str,
        /// That slice's length.
        expected: usize,
    },
    /// An image index names no image.
    ImageIndexOutOfRange {
        /// The slice holding the index.
        name: &'static str,
        /// The index.
        index: u32,
        /// How many images the coverage spans.
        n_images: usize,
    },
    /// An image's grid would have more than [`MAX_GRID_CELLS`] cells.
    GridTooLarge {
        /// The image.
        image: usize,
        /// Its `[width, height]` in pixels.
        size: [u32; 2],
        /// The cell size in pixels.
        cell_px: u32,
    },
    /// An image's grid could not be allocated.
    OutOfMemory {
        /// The image.
        image: usize,
        /// The cells its grid needed.
        cells: u64,
    },
    /// `n_sectors` is outside `1..=`[`MAX_SECTORS`].
    SectorCount {
        /// The sector count asked for.
        n_sectors: u32,
    },
}

impl std::fmt::Display for ObservationCoverageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroCellPx => write!(f, "cell_px must be positive"),
            Self::Length {
                name,
                len,
                against,
                expected,
            } => write!(f, "{name} has {len} entries but {against} has {expected}"),
            Self::ImageIndexOutOfRange {
                name,
                index,
                n_images,
            } => write!(
                f,
                "{name} contains {index}, out of range for {n_images} images"
            ),
            Self::GridTooLarge {
                image,
                size,
                cell_px,
            } => write!(
                f,
                "image {image} of {} x {} pixels needs more than {MAX_GRID_CELLS} cells of \
                 {cell_px} pixels",
                size[0], size[1]
            ),
            Self::OutOfMemory { image, cells } => write!(
                f,
                "could not allocate the {cells} cells of the grid of image {image}"
            ),
            Self::SectorCount { n_sectors } => write!(
                f,
                "n_sectors must be between 1 and {MAX_SECTORS}, got {n_sectors}"
            ),
        }
    }
}

impl std::error::Error for ObservationCoverageError {}

/// Refuse a slice whose length is not `expected`.
fn require_len(
    name: &'static str,
    len: usize,
    against: &'static str,
    expected: usize,
) -> Result<(), ObservationCoverageError> {
    if len == expected {
        Ok(())
    } else {
        Err(ObservationCoverageError::Length {
            name,
            len,
            against,
            expected,
        })
    }
}

/// Refuse an image index that names no image.
fn require_images(
    name: &'static str,
    image_indexes: &[u32],
    n_images: usize,
) -> Result<(), ObservationCoverageError> {
    match image_indexes.iter().find(|&&i| i as usize >= n_images) {
        Some(&index) => Err(ObservationCoverageError::ImageIndexOutOfRange {
            name,
            index,
            n_images,
        }),
        None => Ok(()),
    }
}

/// Per-image occupancy grids over the observations' image-space footprints.
///
/// Per image, a grid of saturating `u8` counts at `1/cell_px` resolution: cell
/// `(cx, cy)` covers the pixel square `[cx·cell_px, (cx+1)·cell_px) ×
/// [cy·cell_px, (cy+1)·cell_px)` and its count is the number of observation
/// footprints containing the cell's **center**. Grid dimensions are
/// `ceil(width / cell_px) × ceil(height / cell_px)`, so cells at the right and
/// bottom edges may extend past the image; clipping never removes them and the
/// queries treat them like any other in-grid cell.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ObservationCoverage {
    cell_px: u32,
    /// Per image, `[width_cells, height_cells]`.
    dims: Vec<[u32; 2]>,
    /// Per image, `width_cells · height_cells` row-major counts.
    grids: Vec<Vec<u8>>,
}

impl ObservationCoverage {
    /// Rasterize every observation's footprint into its image's grid.
    ///
    /// # Arguments
    /// * `image_sizes` — per image, `[width, height]` in pixels.
    /// * `track_image_indexes` — per observation, the image it was seen in.
    /// * `keypoints_xy` — per observation, the footprint center in that image.
    /// * `radii_px` — per observation, the footprint radius. A non-positive or
    ///   non-finite radius contributes nothing, as does a non-finite keypoint.
    /// * `cell_px` — grid cell size in pixels ([`DEFAULT_CELL_PX`] is the spec's
    ///   default).
    ///
    /// Every cell of the observation's image whose center lies within `radius`
    /// of the keypoint is incremented once, saturating at 255; a footprint that
    /// reaches past the grid is clipped to it, and one that misses every cell
    /// contributes nothing. Counts are commutative, so the per-image
    /// rasterization parallelizes over images with results that do not depend on
    /// thread scheduling.
    ///
    /// # Errors
    /// [`ObservationCoverageError::ZeroCellPx`] when `cell_px` is zero,
    /// [`ObservationCoverageError::Length`] when the per-observation slices
    /// disagree on length, [`ObservationCoverageError::ImageIndexOutOfRange`]
    /// when an observation names an image past `image_sizes`,
    /// [`ObservationCoverageError::GridTooLarge`] when an image's grid would
    /// have more than [`MAX_GRID_CELLS`] cells, and
    /// [`ObservationCoverageError::OutOfMemory`] when a grid cannot be
    /// allocated.
    pub fn build(
        image_sizes: &[[u32; 2]],
        track_image_indexes: &[u32],
        keypoints_xy: &[[f64; 2]],
        radii_px: &[f32],
        cell_px: u32,
    ) -> Result<Self, ObservationCoverageError> {
        if cell_px == 0 {
            return Err(ObservationCoverageError::ZeroCellPx);
        }
        let n_obs = keypoints_xy.len();
        require_len(
            "track_image_indexes",
            track_image_indexes.len(),
            "keypoints_xy",
            n_obs,
        )?;
        require_len("radii_px", radii_px.len(), "keypoints_xy", n_obs)?;
        let n_images = image_sizes.len();
        require_images("track_image_indexes", track_image_indexes, n_images)?;

        let dims: Vec<[u32; 2]> = image_sizes
            .iter()
            .map(|&[w, h]| [w.div_ceil(cell_px), h.div_ceil(cell_px)])
            .collect();
        // Every grid is sized and allocated before any is filled, so a capture
        // too large to hold is refused before the rasterization starts.
        let mut grids: Vec<Vec<u8>> = Vec::with_capacity(n_images);
        for (image, &[w, h]) in dims.iter().enumerate() {
            let cells = u64::from(w) * u64::from(h);
            if cells > MAX_GRID_CELLS {
                return Err(ObservationCoverageError::GridTooLarge {
                    image,
                    size: image_sizes[image],
                    cell_px,
                });
            }
            let mut grid = Vec::new();
            grid.try_reserve_exact(cells as usize)
                .map_err(|_| ObservationCoverageError::OutOfMemory { image, cells })?;
            grid.resize(cells as usize, 0u8);
            grids.push(grid);
        }

        // Group first, then rasterize one image per task: no two tasks ever
        // touch the same grid, so no synchronization and no nested parallelism.
        let mut by_image: Vec<Vec<u32>> = vec![Vec::new(); n_images];
        for (o, &image) in track_image_indexes.iter().enumerate() {
            by_image[image as usize].push(o as u32);
        }

        grids
            .par_iter_mut()
            .zip(by_image.par_iter().zip(dims.par_iter()))
            .for_each(|(grid, (obs, &dim))| {
                rasterize(grid, obs, dim, keypoints_xy, radii_px, cell_px)
            });

        Ok(Self {
            cell_px,
            dims,
            grids,
        })
    }

    /// Number of images the coverage spans.
    pub fn image_count(&self) -> usize {
        self.dims.len()
    }

    /// Grid cell size in pixels.
    pub fn cell_px(&self) -> u32 {
        self.cell_px
    }

    /// One image's counts as `(cells, width_cells, height_cells)`, row-major, so
    /// callers can run analyses the queries do not anticipate.
    ///
    /// `None` when `image_index` names no image.
    pub fn grid(&self, image_index: usize) -> Option<(&[u8], u32, u32)> {
        let [w, h] = *self.dims.get(image_index)?;
        Some((&self.grids[image_index], w, h))
    }

    /// The count of the cell containing each pixel coordinate; 0 for a
    /// coordinate outside the grid.
    ///
    /// # Errors
    /// [`ObservationCoverageError::Length`] when the two slices disagree on
    /// length, and [`ObservationCoverageError::ImageIndexOutOfRange`] when an
    /// image index names no image.
    pub fn counts_at(
        &self,
        image_indexes: &[u32],
        xy: &[[f64; 2]],
    ) -> Result<Vec<u8>, ObservationCoverageError> {
        self.check_batch(image_indexes, xy, None)?;
        Ok(image_indexes
            .par_iter()
            .zip(xy.par_iter())
            .map(|(&image, &[x, y])| {
                self.cell_index(image as usize, x, y)
                    .map_or(0, |c| self.grids[image as usize][c])
            })
            .collect())
    }

    /// Refuse a batch query whose slices disagree on length or whose image
    /// indexes name no image.
    fn check_batch(
        &self,
        image_indexes: &[u32],
        xy: &[[f64; 2]],
        radius_px: Option<&[f32]>,
    ) -> Result<(), ObservationCoverageError> {
        require_len("image_indexes", image_indexes.len(), "xy", xy.len())?;
        if let Some(radius_px) = radius_px {
            require_len("radius_px", radius_px.len(), "xy", xy.len())?;
        }
        require_images("image_indexes", image_indexes, self.image_count())
    }

    /// Of the cells whose centers lie within `radius` of the query point, the
    /// fraction with count ≥ 1; 0 when no cell center falls in the disk.
    ///
    /// # Errors
    /// As for [`Self::counts_at`], and [`ObservationCoverageError::Length`]
    /// when `radius_px` disagrees with `xy` on length.
    pub fn covered_fraction(
        &self,
        image_indexes: &[u32],
        xy: &[[f64; 2]],
        radius_px: &[f32],
    ) -> Result<Vec<f32>, ObservationCoverageError> {
        self.check_batch(image_indexes, xy, Some(radius_px))?;
        Ok((0..xy.len())
            .into_par_iter()
            .map(|q| {
                let (mut total, mut covered) = (0u64, 0u64);
                self.for_each_disk_cell(
                    image_indexes[q] as usize,
                    xy[q],
                    radius_px[q] as f64,
                    |_, _, count| {
                        total += 1;
                        covered += u64::from(count > 0);
                    },
                );
                if total == 0 {
                    0.0
                } else {
                    covered as f32 / total as f32
                }
            })
            .collect())
    }

    /// Bitmask of the angular sectors around each query point that hold at least
    /// one uncovered cell — the directions worth reaching into.
    ///
    /// The disk of `radius` around the query point is divided into `n_sectors`
    /// equal sectors, sector `k` spanning `[k, k + 1) · 2π / n_sectors` of
    /// `atan2(dy, dx)` of the cell center relative to the query point. Bit `k`
    /// is set when sector `k` contains at least one in-grid cell with count 0.
    /// A sector none of whose cells fall inside the grid contributes no bit —
    /// outside the image is not spawnable, so it is never reported as
    /// uncovered — and a cell at exactly the query point has no direction and is
    /// skipped.
    ///
    /// # Errors
    /// As for [`Self::covered_fraction`], and
    /// [`ObservationCoverageError::SectorCount`] when `n_sectors` is outside
    /// `1..=`[`MAX_SECTORS`], which the bitmask cannot represent.
    pub fn uncovered_sectors(
        &self,
        image_indexes: &[u32],
        xy: &[[f64; 2]],
        radius_px: &[f32],
        n_sectors: u32,
    ) -> Result<Vec<u32>, ObservationCoverageError> {
        if n_sectors == 0 || n_sectors > MAX_SECTORS {
            return Err(ObservationCoverageError::SectorCount { n_sectors });
        }
        self.check_batch(image_indexes, xy, Some(radius_px))?;
        Ok((0..xy.len())
            .into_par_iter()
            .map(|q| {
                let mut mask = 0u32;
                self.for_each_disk_cell(
                    image_indexes[q] as usize,
                    xy[q],
                    radius_px[q] as f64,
                    |dx, dy, count| {
                        if count > 0 || (dx == 0.0 && dy == 0.0) {
                            return;
                        }
                        mask |= 1u32 << sector_of(dx, dy, n_sectors);
                    },
                );
                mask
            })
            .collect())
    }

    /// Fraction of one image's cells with count ≥ 1, an overall claim summary.
    ///
    /// 0 for an image that does not exist or has no cells.
    pub fn image_covered_fraction(&self, image_index: usize) -> f32 {
        let Some(grid) = self.grids.get(image_index) else {
            return 0.0;
        };
        if grid.is_empty() {
            return 0.0;
        }
        let covered = grid.iter().filter(|&&c| c > 0).count();
        covered as f32 / grid.len() as f32
    }

    /// Flat index of the cell containing pixel `(x, y)` in `image`, or `None`
    /// when the coordinate or the image lies outside the grid.
    fn cell_index(&self, image: usize, x: f64, y: f64) -> Option<usize> {
        let [w, h] = *self.dims.get(image)?;
        // Checked here rather than left to the cast, which saturates.
        if !x.is_finite() || !y.is_finite() || x < 0.0 || y < 0.0 {
            return None;
        }
        let cell = self.cell_px as f64;
        let (cx, cy) = ((x / cell).floor(), (y / cell).floor());
        if cx >= w as f64 || cy >= h as f64 {
            return None;
        }
        Some(cy as usize * w as usize + cx as usize)
    }

    /// Visit every in-grid cell of `image` whose center lies within `radius` of
    /// `xy`, passing the center's offset from `xy` and the cell's count.
    fn for_each_disk_cell(
        &self,
        image: usize,
        xy: [f64; 2],
        radius: f64,
        mut visit: impl FnMut(f64, f64, u8),
    ) {
        let Some(&[w, h]) = self.dims.get(image) else {
            return;
        };
        let [x, y] = xy;
        if !radius.is_finite() || radius <= 0.0 || !x.is_finite() || !y.is_finite() {
            return;
        }
        let cell = self.cell_px as f64;
        let (Some((cx0, cx1)), Some((cy0, cy1))) = (
            center_span(x, radius, cell, w),
            center_span(y, radius, cell, h),
        ) else {
            return;
        };
        let grid = &self.grids[image];
        let r2 = radius * radius;
        for cy in cy0..=cy1 {
            let dy = (cy as f64 + 0.5) * cell - y;
            let row = cy as usize * w as usize;
            for cx in cx0..=cx1 {
                let dx = (cx as f64 + 0.5) * cell - x;
                if dx * dx + dy * dy <= r2 {
                    visit(dx, dy, grid[row + cx as usize]);
                }
            }
        }
    }
}

/// Rasterize one image's observations into its zeroed grid of
/// `dim[0] · dim[1]` cells.
fn rasterize(
    grid: &mut [u8],
    obs: &[u32],
    dim: [u32; 2],
    keypoints_xy: &[[f64; 2]],
    radii_px: &[f32],
    cell_px: u32,
) {
    let (w, h) = (dim[0], dim[1]);
    if w == 0 || h == 0 {
        return;
    }
    let cell = cell_px as f64;
    for &o in obs {
        let o = o as usize;
        let radius = radii_px[o] as f64;
        let [x, y] = keypoints_xy[o];
        if !radius.is_finite() || radius <= 0.0 || !x.is_finite() || !y.is_finite() {
            continue;
        }
        let (Some((cx0, cx1)), Some((cy0, cy1))) = (
            center_span(x, radius, cell, w),
            center_span(y, radius, cell, h),
        ) else {
            continue;
        };
        let r2 = radius * radius;
        for cy in cy0..=cy1 {
            let dy = (cy as f64 + 0.5) * cell - y;
            let row = cy as usize * w as usize;
            for cx in cx0..=cx1 {
                let dx = (cx as f64 + 0.5) * cell - x;
                if dx * dx + dy * dy <= r2 {
                    let slot = &mut grid[row + cx as usize];
                    *slot = slot.saturating_add(1);
                }
            }
        }
    }
}

/// Inclusive cell-index range along one axis whose centers can lie within
/// `radius` of `p`, clipped to `[0, n_cells - 1]`; `None` when it is empty.
///
/// Cell `c`'s center sits at `(c + 0.5) · cell`, so the unclipped range is
/// `[(p - radius) / cell - 0.5, (p + radius) / cell - 0.5]`.
fn center_span(p: f64, radius: f64, cell: f64, n_cells: u32) -> Option<(u32, u32)> {
    if n_cells == 0 {
        return None;
    }
    let lo = ((p - radius) / cell - 0.5).ceil().max(0.0);
    let hi = ((p + radius) / cell - 0.5)
        .floor()
        .min((n_cells - 1) as f64);
    // Both ends are non-NaN: the caller has already vetted `p` and `radius`.
    if lo > hi {
        return None;
    }
    Some((lo as u32, hi as u32))
}

/// Sector index of the displacement `(dx, dy)` among `n_sectors` equal sectors,
/// sector `k` spanning `[k, k + 1) · 2π / n_sectors`.
///
/// `atan2` lands in `(-π, π]`, so `angle / 2π + 1` lands in `(0.5, 1.5]`; the
/// scaled value is therefore always positive and truncation toward zero is a
/// floor. `n_sectors` must be non-zero.
fn sector_of(dx: f64, dy: f64, n_sectors: u32) -> u32 {
    let angle = dy.atan2(dx);
    let scaled = (angle / TAU + 1.0) * n_sectors as f64;
    (scaled as u32) % n_sectors
}

#[cfg(test)]
mod tests;
