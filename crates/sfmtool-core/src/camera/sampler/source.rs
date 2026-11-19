// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The render entry points: how a kernel reads a [`TiledPhotograph`].
//!
//! Each reads one view of a patch through its [`ViewRegions`]: it works out
//! the regions its read needs and reads the view-region pyramid the view
//! kept when that holds them; otherwise it reserves the regions, grown by
//! [`REGION_REUSE_MARGIN`], from the kernel call's [`RegionPool`], assembles
//! them from the photograph's tiles and keeps them; when the reservation
//! fails it reads through photograph tiles ([`TileWindow`]). Every way gives
//! the values a whole pyramid gives, bit for bit.

use crate::camera::image::{ImageF32WithGrad, ImageU8, ImageU8Pyramid, PyramidLevel};
use crate::camera::remap::{
    aniso_grad_row, aniso_pixel, bilinear_cell, bilinear_grad_row, bilinear_mip_grad_row,
    bilinear_mip_row, bilinear_row, count_call, AnisoTally, GradRow, HeldLevel, LevelTexels,
    PyramidTexels,
};
use crate::camera::tiled_photograph::{
    check_rect, out_of_range, PhotographReadError, TiledPhotograph,
};
use crate::camera::WarpMap;

use super::grid::{
    affine_regions, grid_cells, grid_regions, sample_grid, sample_support_affine,
    sample_support_affine_aniso, GridEdge, GridPoints,
};
use super::pool::{RegionPool, Reservation};
use super::regions::{required_regions, LevelRect, LevelRegions, REGION_REUSE_MARGIN};
use super::tile_window::{TileWindow, WindowStats};
use super::{
    prepare_for_grad, render_prepared, render_with_grad_prepared, Sampler, MAX_ANISOTROPY,
};

/// How a read reached its pixels, for the tests.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ReadPath {
    /// Regions of these bytes, reserved from the pool and assembled from the
    /// photograph's tiles.
    Copied { bytes: u64 },
    /// The regions the view kept from an earlier read, of these bytes.
    Reused { bytes: u64 },
    /// The tile path.
    Tiles(WindowStats),
}

/// One view of the patch a kernel is working on, as the render entry points
/// read it: the view's photograph, the kernel call's region pool, and the
/// view-region pyramid kept from the view's last copied read.
///
/// A kernel makes one for each view of a patch and passes it to every read
/// of that view while it works on the patch: a fit's rounds, a refinement's
/// steps and an evaluation's views of one track share their views' kept
/// regions, and the kernel drops them when it moves to the next patch, so
/// the next starts with none. A read whose required regions lie inside the
/// kept ones reads the kept pyramid; any other read drops it and assembles a
/// new one, or takes the tile path when the pool cannot hold it. The kept
/// regions stay reserved in the pool until they are replaced or the
/// `ViewRegions` is dropped. Reuse changes no value, only how often regions
/// are assembled.
pub struct ViewRegions<'a> {
    source: &'a TiledPhotograph,
    pool: &'a RegionPool,
    kept: Option<Kept<'a>>,
}

/// A view-region pyramid kept for reuse, with its reservation.
struct Kept<'a> {
    /// The regions the pyramid holds: the required regions of the read that
    /// assembled it, grown by [`REGION_REUSE_MARGIN`] and clipped to their
    /// levels.
    regions: LevelRegions,
    pyramid: ImageU8Pyramid,
    reservation: Reservation<'a>,
}

impl<'a> ViewRegions<'a> {
    /// A view of `source` that reads through `pool`, keeping no regions yet.
    pub fn new(source: &'a TiledPhotograph, pool: &'a RegionPool) -> Self {
        Self {
            source,
            pool,
            kept: None,
        }
    }

    /// The view's photograph.
    pub fn source(&self) -> &'a TiledPhotograph {
        self.source
    }

    /// The kernel call's region pool.
    pub fn pool(&self) -> &'a RegionPool {
        self.pool
    }

    /// The regions the kept pyramid holds, or `None` when it keeps none.
    pub fn kept_regions(&self) -> Option<&LevelRegions> {
        self.kept.as_ref().map(|k| &k.regions)
    }

    /// The bytes the kept pyramid reserves from the pool, 0 when it keeps
    /// none.
    pub fn kept_bytes(&self) -> u64 {
        self.kept.as_ref().map_or(0, |k| k.reservation.bytes())
    }
}

/// Whether every region of `required` lies inside the same level's region of
/// `held`.
fn holds(held: &LevelRegions, required: &LevelRegions) -> bool {
    held.levels() == required.levels()
        && (0..required.levels()).all(|level| match required.region(level) {
            None => true,
            Some(rect) => held.region(level).is_some_and(|h| h.contains(&rect)),
        })
}

/// Refuse a photograph with no pixels.
pub(crate) fn check_source(source: &TiledPhotograph) -> Result<(), PhotographReadError> {
    let (w, h) = source.layout().level_size(0);
    if w == 0 || h == 0 || source.channels() == 0 {
        return Err(out_of_range(format!(
            "the photograph is {w} x {h} with {} channels",
            source.channels()
        )));
    }
    Ok(())
}

/// The pixels a read reaches: copied regions, or the tile path's window.
enum Held<'a, 's> {
    Copied(&'a ImageU8Pyramid),
    Tiles(&'a TileWindow<'s>),
}

/// Read `regions` of `view`'s photograph with `read`: from the pyramid the
/// view kept when it holds them; otherwise from a new pyramid of the regions
/// grown by [`REGION_REUSE_MARGIN`], reserved from the pool, assembled from
/// the photograph's tiles and kept; or, when the reservation fails or there
/// is nothing to read, through the tile path.
fn read_regions<R>(
    view: &mut ViewRegions<'_>,
    regions: &LevelRegions,
    read: impl FnOnce(Held<'_, '_>) -> R,
) -> Result<(R, ReadPath), PhotographReadError> {
    let (source, pool) = (view.source, view.pool);
    if !regions.is_empty() {
        if let Some(kept) = view.kept.as_ref().filter(|k| holds(&k.regions, regions)) {
            pool.count_reused();
            let bytes = kept.reservation.bytes();
            return Ok((
                read(Held::Copied(&kept.pyramid)),
                ReadPath::Reused { bytes },
            ));
        }
        // Give the kept regions back before reserving their replacement.
        view.kept = None;
        let grown = regions.grown(REGION_REUSE_MARGIN);
        let bytes = grown.byte_len(source.channels());
        if let Some(reservation) = pool.try_reserve(bytes) {
            let pyramid = copy_regions(source, &grown)?;
            pool.count_copied();
            let kept = view.kept.insert(Kept {
                regions: grown,
                pyramid,
                reservation,
            });
            return Ok((
                read(Held::Copied(&kept.pyramid)),
                ReadPath::Copied { bytes },
            ));
        }
    }
    let window = TileWindow::new(source);
    let result = read(Held::Tiles(&window));
    let stats = window.finish()?;
    pool.count_tiled();
    Ok((result, ReadPath::Tiles(stats)))
}

/// A region pyramid of `regions`, each assembled from `source`'s tiles.
fn copy_regions(
    source: &TiledPhotograph,
    regions: &LevelRegions,
) -> Result<ImageU8Pyramid, PhotographReadError> {
    let layout = source.layout();
    let mut levels = Vec::with_capacity(regions.levels());
    for level in 0..regions.levels() {
        levels.push(match regions.region(level) {
            Some(rect) => {
                let (w, h) = layout.level_size(level);
                Some(PyramidLevel {
                    image: source.region(level, rect)?,
                    origin: [rect.x0, rect.y0],
                    full_size: [w, h],
                })
            }
            None => None,
        });
    }
    ImageU8Pyramid::from_regions(levels).map_err(|e| out_of_range(e.to_string()))
}

/// The rows of a `map`-sized `u8` render on the tile path, in order.
fn tile_rows(map: &WarpMap, channels: usize, mut fill_row: impl FnMut(u32, &mut [u8])) -> ImageU8 {
    let stride = map.width() as usize * channels;
    let mut data = vec![0u8; stride * map.height() as usize];
    if stride > 0 {
        for (row, row_data) in data.chunks_exact_mut(stride).enumerate() {
            fill_row(row as u32, row_data);
        }
    }
    count_call(map);
    ImageU8::new(map.width(), map.height(), channels as u32, data)
}

/// [`render_tile`](super::render_tile) of `source`: the tile of `map` rendered with `sampler`.
///
/// Prepares the map as `render_tile` does, and gives `render_tile`'s values on the whole pyramid, bit for bit,
/// through copied regions or through tiles.
pub fn render_source_tile(
    view: &mut ViewRegions<'_>,
    map: &mut WarpMap,
    sampler: Sampler,
) -> Result<ImageU8, PhotographReadError> {
    render_source_tile_path(view, map, sampler).map(|(image, _)| image)
}

/// [`render_source_tile`], and how it read.
pub(crate) fn render_source_tile_path(
    view: &mut ViewRegions<'_>,
    map: &mut WarpMap,
    sampler: Sampler,
) -> Result<(ImageU8, ReadPath), PhotographReadError> {
    let source = view.source();
    check_source(source)?;
    // Computes the SVD the sampler needs, as `render_tile` does.
    let regions = required_regions(map, sampler, source.layout());
    let map = &*map;
    let channels = source.channels() as usize;
    read_regions(view, &regions, |held| match held {
        Held::Copied(pyramid) => render_prepared(pyramid, map, sampler),
        Held::Tiles(window) => match sampler {
            Sampler::Bilinear => tile_rows(map, channels, |row, row_data| {
                bilinear_row(&window.texels(0), map, row, row_data)
            }),
            Sampler::BilinearMip => tile_rows(map, channels, |row, row_data| {
                bilinear_mip_row(window, map, row, row_data)
            }),
            Sampler::Anisotropic => tile_rows(map, channels, |row, row_data| {
                let mut tally = AnisoTally::default();
                for col in 0..map.width() {
                    aniso_pixel(window, map, col, row, MAX_ANISOTROPY, row_data, &mut tally);
                }
                tally.count();
            }),
        },
    })
}

/// [`render_tile_with_grad_into`](super::render_tile_with_grad_into) of
/// `source`: the tile of `map` and its
/// gradients rendered with `sampler` into `out`, with the values the whole
/// pyramid gives, bit for bit.
pub fn render_source_tile_with_grad_into(
    view: &mut ViewRegions<'_>,
    map: &mut WarpMap,
    sampler: Sampler,
    out: &mut ImageF32WithGrad,
) -> Result<(), PhotographReadError> {
    render_source_tile_with_grad_path(view, map, sampler, out).map(|_| ())
}

/// [`render_source_tile_with_grad_into`], and how it read.
pub(crate) fn render_source_tile_with_grad_path(
    view: &mut ViewRegions<'_>,
    map: &mut WarpMap,
    sampler: Sampler,
    out: &mut ImageF32WithGrad,
) -> Result<ReadPath, PhotographReadError> {
    let source = view.source();
    check_source(source)?;
    prepare_for_grad(map, sampler);
    let regions = required_regions(map, sampler, source.layout());
    let channels = source.channels();
    let (out_w, out_h) = (map.width(), map.height());
    let map_ref = &*map;
    let ((), path) = read_regions(view, &regions, |held| match held {
        Held::Copied(pyramid) => render_with_grad_prepared(pyramid, map_ref, sampler, out),
        Held::Tiles(window) => {
            out.resize(out_w, out_h, channels);
            let stride = out_w as usize * channels as usize;
            if stride == 0 {
                return;
            }
            let rows = out
                .value
                .chunks_exact_mut(stride)
                .zip(out.grad_x.chunks_exact_mut(stride))
                .zip(out.grad_y.chunks_exact_mut(stride));
            for (row, ((value, grad_x), grad_y)) in rows.enumerate() {
                let row_out = GradRow {
                    value,
                    grad_x,
                    grad_y,
                };
                let row = row as u32;
                match sampler {
                    Sampler::Bilinear => {
                        bilinear_grad_row(&window.texels(0), map_ref, row, row_out)
                    }
                    Sampler::BilinearMip => bilinear_mip_grad_row(window, map_ref, row, row_out),
                    Sampler::Anisotropic => {
                        aniso_grad_row(window, map_ref, MAX_ANISOTROPY, row, row_out)
                    }
                }
            }
            count_call(map_ref);
        }
    })?;
    Ok(path)
}

/// Refuse a level `source` does not have.
fn check_level(source: &TiledPhotograph, level: usize) -> Result<(), PhotographReadError> {
    if level >= source.layout().levels() {
        return Err(out_of_range(format!(
            "level {level} of a pyramid of {} levels",
            source.layout().levels()
        )));
    }
    Ok(())
}

/// The view selection's affine fast path on `source`: the support's raw
/// samples, planar (`[channel · n + support_index]`), into `out`.
///
/// `a` is the affine map from grid indices to level pixels at `level` under
/// [`Sampler::Bilinear`] and [`Sampler::BilinearMip`], each support pixel one
/// bilinear sample with `u8` rounding; under [`Sampler::Anisotropic`] it is
/// in level-0 pixels, `level` is 0, and each support pixel takes the
/// anisotropic sample of the map's constant SVD. `pixels` are row-major
/// indices into a grid of `resolution` columns, and `out` holds
/// `channels · pixels.len()` values. Refuses a level the photograph does not
/// have, a `resolution` of 0 and an `out` of the wrong length.
pub fn render_source_affine(
    view: &mut ViewRegions<'_>,
    a: &[f64; 6],
    level: usize,
    pixels: &[usize],
    resolution: usize,
    sampler: Sampler,
    out: &mut [f32],
) -> Result<(), PhotographReadError> {
    render_source_affine_path(view, a, level, pixels, resolution, sampler, out).map(|_| ())
}

/// [`render_source_affine`], and how it read.
pub(crate) fn render_source_affine_path(
    view: &mut ViewRegions<'_>,
    a: &[f64; 6],
    level: usize,
    pixels: &[usize],
    resolution: usize,
    sampler: Sampler,
    out: &mut [f32],
) -> Result<ReadPath, PhotographReadError> {
    let source = view.source();
    check_source(source)?;
    check_level(source, level)?;
    if resolution == 0 || out.len() != source.channels() as usize * pixels.len() {
        return Err(out_of_range(format!(
            "an affine read of {} pixels of a grid of {resolution} columns into {} values",
            pixels.len(),
            out.len()
        )));
    }
    let aniso = sampler == Sampler::Anisotropic;
    let mut regions = LevelRegions::new(source.layout());
    affine_regions(&mut regions, a, level, aniso, pixels, resolution);
    let ((), path) = read_regions(view, &regions, |held| match held {
        Held::Copied(pyramid) if aniso => {
            sample_support_affine_aniso(pyramid, a, pixels, resolution, out)
        }
        Held::Copied(pyramid) => {
            sample_support_affine(&HeldLevel::of(pyramid, level), a, pixels, resolution, out)
        }
        Held::Tiles(window) if aniso => {
            sample_support_affine_aniso(window, a, pixels, resolution, out)
        }
        Held::Tiles(window) => {
            sample_support_affine(&window.texels(level), a, pixels, resolution, out)
        }
    })?;
    Ok(path)
}

/// A cluster grid sampled on `source`: the interleaved `n × C` `f32` raw
/// samples of `points` through the affine map `a`, from grid indices to
/// pixels of level `level`, by `edge`'s rule, as the cluster refinement's
/// `sample_grid_window` and `build_template` sample it. `Ok(None)` where the
/// grid is refused: a level under 2 px a side, a non-finite coordinate under
/// [`GridEdge::Clamp`], or any sample out of frame under [`GridEdge::InFrame`].
pub fn sample_source_grid(
    view: &mut ViewRegions<'_>,
    a: &[f64; 6],
    level: usize,
    points: GridPoints<'_>,
    edge: GridEdge,
) -> Result<Option<Vec<f32>>, PhotographReadError> {
    sample_source_grid_path(view, a, level, points, edge).map(|r| r.map(|(v, _)| v))
}

/// [`sample_source_grid`], and how it read.
pub(crate) fn sample_source_grid_path(
    view: &mut ViewRegions<'_>,
    a: &[f64; 6],
    level: usize,
    points: GridPoints<'_>,
    edge: GridEdge,
) -> Result<Option<(Vec<f32>, ReadPath)>, PhotographReadError> {
    let source = view.source();
    check_source(source)?;
    check_level(source, level)?;
    if let GridPoints::Window { size, .. } = points {
        if size.checked_mul(size).is_none() {
            return Err(out_of_range(format!("a grid window of {size} x {size}")));
        }
    }
    if let GridPoints::Pixels { resolution: 0, .. } = points {
        return Err(out_of_range("a grid of 0 columns".to_string()));
    }
    let Some(cells) = grid_cells(a, points, edge, source.layout().level_size(level)) else {
        return Ok(None);
    };
    let mut regions = LevelRegions::new(source.layout());
    grid_regions(&mut regions, level, &cells);
    let (samples, path) = read_regions(view, &regions, |held| match held {
        Held::Copied(pyramid) => sample_grid(&HeldLevel::of(pyramid, level), &cells),
        Held::Tiles(window) => sample_grid(&window.texels(level), &cells),
    })?;
    Ok(Some((samples, path)))
}

/// The pixels about each position of a [`read_source_samples`] read.
pub struct SourceSamples {
    side: usize,
    channels: usize,
    pixels: Vec<u8>,
    fractions: Vec<[f32; 2]>,
}

impl SourceSamples {
    /// The window's side, `2 + 2 · border`.
    pub fn side(&self) -> usize {
        self.side
    }

    /// The channels of each pixel.
    pub fn channels(&self) -> usize {
        self.channels
    }

    /// The number of positions.
    pub fn len(&self) -> usize {
        self.fractions.len()
    }

    /// Whether there are no positions.
    pub fn is_empty(&self) -> bool {
        self.fractions.is_empty()
    }

    /// Position `k`'s window: `side` rows of `side` pixels of `channels`
    /// bytes, row-major, from `floor(y − 0.5) − border` and
    /// `floor(x − 0.5) − border`.
    pub fn window(&self, k: usize) -> &[u8] {
        let len = self.side * self.side * self.channels;
        &self.pixels[k * len..][..len]
    }

    /// Position `k`'s in-cell fractions `(fx, fy)`, as the bilinear samplers
    /// compute them.
    pub fn fractions(&self, k: usize) -> [f32; 2] {
        self.fractions[k]
    }
}

/// The pixels a bilinear read (`border` 0) or a read of the 4 x 4 pixels
/// (`border` 1) at each of `positions` on level `level` takes: for each
/// position the `(2 + 2 · border)²` pixels from `floor(x − 0.5) − border`
/// and `floor(y − 0.5) − border`, each address clamped to the level as
/// `sample_bilinear_u8` and the grown plane's `sobel_gradient` clamp it, with
/// the bilinear in-cell fractions. Refuses a level the photograph does not
/// have and a border over 1.
pub fn read_source_samples(
    view: &mut ViewRegions<'_>,
    level: usize,
    positions: &[[f32; 2]],
    border: u32,
) -> Result<SourceSamples, PhotographReadError> {
    read_source_samples_path(view, level, positions, border).map(|(s, _)| s)
}

/// [`read_source_samples`], and how it read.
pub(crate) fn read_source_samples_path(
    view: &mut ViewRegions<'_>,
    level: usize,
    positions: &[[f32; 2]],
    border: u32,
) -> Result<(SourceSamples, ReadPath), PhotographReadError> {
    let source = view.source();
    check_source(source)?;
    check_level(source, level)?;
    if border > 1 {
        return Err(out_of_range(format!("a border of {border} px, over 1")));
    }
    let regions = LevelRegions::for_samples(source.layout(), level, positions, border);
    let side = 2 + 2 * border as usize;
    let channels = source.channels() as usize;
    let gather = |texels: &dyn Fn(i32, i32, &mut Vec<u8>)| {
        let mut pixels = Vec::with_capacity(positions.len() * side * side * channels);
        let mut fractions = Vec::with_capacity(positions.len());
        for &[x, y] in positions {
            let (x0, y0, fx, fy) = bilinear_cell(x, y);
            let (bx, by) = (
                x0.saturating_sub(border as i32),
                y0.saturating_sub(border as i32),
            );
            for r in 0..side as i32 {
                for q in 0..side as i32 {
                    texels(bx.saturating_add(q), by.saturating_add(r), &mut pixels);
                }
            }
            fractions.push([fx, fy]);
        }
        SourceSamples {
            side,
            channels,
            pixels,
            fractions,
        }
    };
    // Each address is clamped to the full level first, then read.
    let (w, h) = source.layout().level_size(level);
    let (w_max, h_max) = ((w as i32 - 1).max(0), (h as i32 - 1).max(0));
    let clamp = move |x: i32, y: i32| (x.clamp(0, w_max), y.clamp(0, h_max));
    read_regions(view, &regions, |held| match held {
        Held::Copied(pyramid) => {
            let held = HeldLevel::of(pyramid, level);
            gather(&|x, y, out| {
                let (x, y) = clamp(x, y);
                held.with_pixel(x, y, |p| out.extend_from_slice(p));
            })
        }
        Held::Tiles(window) => {
            let texels = window.texels(level);
            gather(&|x, y, out| {
                let (x, y) = clamp(x, y);
                texels.with_pixel(x, y, |p| out.extend_from_slice(p));
            })
        }
    })
}

/// A rectangle of a level read by [`read_source_rect`], holding its
/// reservation from the kernel call's pool until dropped.
pub struct SourceRect<'p> {
    image: ImageU8,
    _reservation: Option<Reservation<'p>>,
}

impl SourceRect<'_> {
    /// The rectangle's pixels.
    pub fn image(&self) -> &ImageU8 {
        &self.image
    }
}

/// The pixels of `rect` of level `level`, a rectangle the kernel's own
/// arithmetic defines.
///
/// Reserves the rectangle's bytes from `pool` and copies it, or, when the
/// reservation fails, reads it out of the photograph's tiles; either way the
/// rectangle is copied, and the reservation, if any, is held until the
/// returned [`SourceRect`] is dropped. Refuses a level the photograph does
/// not have and an empty rectangle or one past its level.
pub fn read_source_rect<'p>(
    source: &TiledPhotograph,
    pool: &'p RegionPool,
    level: usize,
    rect: LevelRect,
) -> Result<SourceRect<'p>, PhotographReadError> {
    check_rect(source.layout(), level, rect)?;
    let bytes = u64::from(rect.width()) * u64::from(rect.height()) * u64::from(source.channels());
    if let Some(reservation) = pool.try_reserve(bytes) {
        pool.count_copied();
        return Ok(SourceRect {
            image: source.region(level, rect)?,
            _reservation: Some(reservation),
        });
    }
    let window = TileWindow::new(source);
    let texels = window.texels(level);
    let mut data = Vec::with_capacity(bytes as usize);
    for y in rect.y0..rect.y1 {
        for x in rect.x0..rect.x1 {
            texels.with_pixel(x as i32, y as i32, |p| data.extend_from_slice(p));
        }
    }
    window.finish()?;
    pool.count_tiled();
    Ok(SourceRect {
        image: ImageU8::new(rect.width(), rect.height(), source.channels(), data),
        _reservation: None,
    })
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod kept_tests;
