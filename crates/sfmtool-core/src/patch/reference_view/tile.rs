// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! One view's `R×R` tile and the readings taken while rendering it: which of
//! its samples carry image data, the share of the photograph under it that is
//! clipped, and the angle the view sees the patch at.

use ndarray::Array3;

use crate::camera::image::ImageU8Pyramid;
use crate::camera::sampler::{
    check_source, render_phase, render_source_tile, Sampler, SamplerChoice, ViewRegions,
};
use crate::camera::tiled_photograph::{
    PhotographReadError, PhotographTile, TileKey, TiledPhotograph,
};
use crate::camera::warp_map::patch_grid_jacobian;
use crate::camera::WarpMap;
use crate::patch::blur_matched::TilePlanes;
use crate::patch::cloud::OrientedPatch;
use crate::patch::normal_refine::{viewing_angle, ProjectedImage, ViewingAngle};
use crate::progress::Progress;

/// One view's `R×R` tile of a patch, rendered as the bench and every stored
/// patch bitmap render it, with what was read alongside it.
#[derive(Debug, Clone, PartialEq)]
pub struct ViewTile {
    /// The `(R, R, C)` samples, `C` the photograph's channel count. A sample
    /// the warp cannot place on the photograph is black. The second of two
    /// channels and the fourth of four are alpha, opaque everywhere, as a
    /// stored bitmap has it
    /// ([`InterleavedTile`](crate::patch::self_similarity::InterleavedTile)'s
    /// channel rule).
    pub samples: Array3<u8>,
    /// Per sample, row-major: whether the warp places it on the photograph,
    /// so that it carries image data.
    pub valid: Vec<bool>,
    /// The patch frame the tile is rendered through: the patch re-anchored on
    /// the keypoint ([`OrientedPatch::anchored_at_keypoint`]), or the patch
    /// itself where there is no keypoint or the slide refuses it.
    pub patch_frame: OrientedPatch,
    /// The Jacobian of the warp at the tile's centre, in image px per grid px
    /// ([`patch_grid_jacobian`]), where the centre projects.
    pub jacobian: Option<[[f64; 2]; 2]>,
    /// The sampler the tile is rendered with.
    pub sampler: Sampler,
    /// The share of the tile's samples that carry image data, `0 ..= 1`.
    pub coverage: f64,
    /// The share of the photograph's pixels under the tile that are clipped
    /// ([`clipped_share`]), or `None` where no sample lands on the photograph.
    pub clipped_share: Option<f64>,
    /// The angle the view sees the patch frame at, at its centre
    /// ([`viewing_angle`]), so at the keypoint where there is one.
    pub viewing_angle: Option<ViewingAngle>,
}

impl ViewTile {
    /// The tile's side, `R`.
    pub fn resolution(&self) -> usize {
        self.samples.shape()[0]
    }

    /// The tile's channel count, `C`.
    pub fn channels(&self) -> usize {
        self.samples.shape()[2]
    }

    /// The tile's colour planes in `f32`, its samples on the photograph
    /// flagged as carrying data: what blur matching reads
    /// ([`crate::patch::blur_matched`]).
    pub fn planes(&self) -> TilePlanes {
        let copied: Vec<u8>;
        let samples = match self.samples.as_slice() {
            Some(samples) => samples,
            None => {
                copied = self.samples.iter().copied().collect();
                &copied
            }
        };
        TilePlanes::from_interleaved(samples, self.resolution(), self.channels(), &self.valid)
    }
}

/// Render `patch` in `view` at `resolution`, anchored on `keypoint` where
/// there is one, with the sampler `sampler` picks from the patch frame's
/// Jacobian, and read its coverage, the clipped share of the photograph under
/// it and the viewing angle.
///
/// The tile is the one the bench's self-similarity reading takes and the one
/// every stored patch bitmap is rendered as: the keypoint-anchored patch frame,
/// the sampler rule's choice at the patch resolution, and black where the warp
/// leaves the photograph. The render is timed in its sampler's detail phase of
/// `progress` ([`render_phase`]).
///
/// `regions` reads `view`'s photograph: the tile through
/// [`render_source_tile`], and the clipped share through
/// [`source_clipped_share`]. A caller that reads the view again while it works
/// on the same patch, as the reference-view rule does for its pair ZNCCs,
/// passes the same `regions`, so the later read reuses the regions this one
/// assembled.
///
/// ```no_run
/// # use sfmtool_core::camera::sampler::{RegionPool, SamplerChoice, ViewRegions};
/// # use sfmtool_core::patch::cloud::OrientedPatch;
/// # use sfmtool_core::patch::normal_refine::ProjectedImage;
/// # use sfmtool_core::patch::reference_view::render_view_tile;
/// # use sfmtool_core::progress::Progress;
/// # fn run(patch: &OrientedPatch, view: &ProjectedImage<'_>)
/// # -> Result<(), Box<dyn std::error::Error>> {
/// let pool = RegionPool::for_kernel_call();
/// let tile = render_view_tile(
///     patch,
///     view,
///     &mut ViewRegions::new(view.source, &pool),
///     Some([812.4, 377.9]),
///     24,
///     SamplerChoice::per_view(),
///     &Progress::none(),
/// )?;
/// println!("coverage {:.2}, clipped {:?}", tile.coverage, tile.clipped_share);
/// # Ok(())
/// # }
/// ```
///
/// # Errors
///
/// The [`PhotographReadError`] of a read of the photograph that failed.
pub fn render_view_tile(
    patch: &OrientedPatch,
    view: &ProjectedImage<'_>,
    regions: &mut ViewRegions<'_>,
    keypoint: Option<[f64; 2]>,
    resolution: usize,
    sampler: SamplerChoice,
    progress: &Progress<'_>,
) -> Result<ViewTile, PhotographReadError> {
    let resolution = resolution.max(2);
    let patch_frame = patch.seen_at_keypoint(view.camera, view.cam_from_world, keypoint);
    let jacobian = patch_grid_jacobian(&patch_frame, view.camera, view.cam_from_world, resolution);
    let sampler = sampler.for_jacobian(jacobian);
    let channels = view.source.channels() as usize;
    let mut map = WarpMap::from_patch(
        &patch_frame,
        view.camera,
        view.cam_from_world,
        resolution as u32,
    );
    let r = resolution as u32;
    let valid: Vec<bool> = (0..r * r).map(|k| map.is_valid(k % r, k / r)).collect();
    let outline = tile_outline(&map);
    let tile = {
        let _phase = render_phase(progress, sampler, 1);
        render_source_tile(regions, &mut map, sampler)?
    };
    let src_channels = tile.channels();
    // The second of two channels is alpha, as the fourth of four is.
    let alpha = (channels == 2).then_some(1);
    let mut samples = Array3::<u8>::zeros((resolution, resolution, channels));
    for row in 0..resolution {
        for col in 0..resolution {
            for c in 0..channels {
                samples[[row, col, c]] = if c >= 3 || Some(c) == alpha {
                    u8::MAX
                } else if src_channels >= 3 {
                    tile.get_pixel(col as u32, row as u32, c as u32)
                } else {
                    tile.get_pixel(col as u32, row as u32, 0)
                };
            }
        }
    }
    let coverage = valid.iter().filter(|&&v| v).count() as f64 / valid.len() as f64;
    Ok(ViewTile {
        samples,
        valid,
        clipped_share: source_clipped_share(regions.source(), &outline)?,
        viewing_angle: viewing_angle(&patch_frame, view.cam_from_world),
        patch_frame,
        jacobian,
        sampler,
        coverage,
    })
}

/// The tile's outline in the photograph: the photograph positions of the
/// samples on the grid's border that land on it, in order round the square
/// from the top-left corner, clockwise in the grid.
fn tile_outline(map: &WarpMap) -> Vec<[f64; 2]> {
    let r = map.width() as usize;
    let mut outline = Vec::with_capacity(4 * r);
    let mut push = |col: usize, row: usize| {
        if map.is_valid(col as u32, row as u32) {
            let (x, y) = map.get(col as u32, row as u32);
            outline.push([f64::from(x), f64::from(y)]);
        }
    };
    for col in 0..r {
        push(col, 0);
    }
    for row in 1..r {
        push(r - 1, row);
    }
    for col in (0..r.saturating_sub(1)).rev() {
        push(col, r - 1);
    }
    for row in (1..r.saturating_sub(1)).rev() {
        push(0, row);
    }
    outline
}

/// The share of the photograph's pixels inside `outline` that are clipped:
/// `0` or `255` in any of the first three channels. `None` when the outline
/// has fewer than three points.
///
/// Read from the photograph's own full-resolution pixels rather than from a
/// rendered tile, because the sampler's blending moves a clipped value off the
/// limit. A pixel is inside when its centre, `(x + 0.5, y + 0.5)`, is inside
/// the polygon by the even-odd rule. An outline smaller than one pixel holds no
/// centre, and then the pixels under its points are read instead.
///
/// ```
/// use sfmtool_core::camera::image::{ImageU8, ImageU8Pyramid};
/// use sfmtool_core::patch::reference_view::clipped_share;
///
/// // A 10 x 10 grey photograph whose left half is blown out to 255.
/// let data: Vec<u8> = (0..100).map(|i| if i % 10 < 5 { 255 } else { 128 }).collect();
/// let pyramid = ImageU8Pyramid::build(&ImageU8::new(10, 10, 1, data), 1);
/// let square = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]];
/// assert_eq!(clipped_share(&pyramid, &square), Some(0.5));
/// ```
pub fn clipped_share(pyramid: &ImageU8Pyramid, outline: &[[f64; 2]]) -> Option<f64> {
    if outline.len() < 3 {
        return None;
    }
    let image = pyramid.level(0);
    let (w, h) = (i64::from(image.width()), i64::from(image.height()));
    let ch = image.channels() as usize;
    let data = image.data();
    let is_clipped = |x: i64, y: i64| clipped_pixel(&data[(y * w + x) as usize * ch..][..ch]);
    let mut count = ClippedCount::default();
    count.add_spans(outline, outline_rows(outline, h), 0..w, is_clipped);
    if count.total == 0 {
        for p in outline {
            let (x, y) = pixel_under(p, w, h);
            count.add(is_clipped(x, y));
        }
    }
    Some(count.share())
}

/// [`clipped_share`] of a photograph read through its tiles: the same share
/// of level 0's pixel centres inside `outline`, read from the level-0 tiles
/// the outline's bounding box covers, one at a time.
///
/// Each tile's pixel centres inside the outline are counted by the same
/// even-odd rule, row by row, and the counts summed, so the share equals the
/// whole level's. A footprint of `F` level-0 pixels covers about
/// `(ceil(F / T) + 1)²` tiles of side `T`; this is the one read whose cost
/// grows with the footprint rather than with the patch resolution. Refuses a
/// photograph with no pixels, and passes up a tile read that fails.
pub fn source_clipped_share(
    source: &TiledPhotograph,
    outline: &[[f64; 2]],
) -> Result<Option<f64>, PhotographReadError> {
    check_source(source)?;
    if outline.len() < 3 {
        return Ok(None);
    }
    let (w, h) = source.layout().level_size(0);
    let (w, h) = (i64::from(w), i64::from(h));
    let ts = i64::from(source.layout().tile_size());
    let ch = source.channels() as usize;
    let tile_of = |x: i64, y: i64| TileKey {
        level: 0,
        col: (x / ts) as u32,
        row: (y / ts) as u32,
    };
    // The pixel of `tile` at level-0 `(x, y)`, inside its core.
    let pixel = |tile: &PhotographTile, x: i64, y: i64| {
        let lx = (x - i64::from(tile.origin[0]) + 1) as usize;
        let ly = (y - i64::from(tile.origin[1]) + 1) as usize;
        let at = (ly * tile.pixels.width() as usize + lx) * ch;
        clipped_pixel(&tile.pixels.data()[at..at + ch])
    };
    let rows = outline_rows(outline, h);
    // The columns of the centres inside the outline: a centre counted lies at
    // or past one crossing and before another, and a crossing lies within the
    // outline's bounding box up to rounding far under half a pixel, so the
    // centre's column lies within `floor(x0)..=ceil(x1)`, as its row lies
    // within `outline_rows`.
    let (mut x0, mut x1) = (f64::MAX, f64::MIN);
    for p in outline {
        x0 = x0.min(p[0]);
        x1 = x1.max(p[0]);
    }
    let cols = (x0.floor() as i64).max(0)..=(x1.ceil() as i64).min(w - 1);
    let mut count = ClippedCount::default();
    if !rows.is_empty() && !cols.is_empty() {
        for row in rows.start() / ts..=rows.end() / ts {
            for col in cols.start() / ts..=cols.end() / ts {
                let tile = source.tile(tile_of(col * ts, row * ts))?;
                let (ox, oy) = (i64::from(tile.origin[0]), i64::from(tile.origin[1]));
                let tile_rows =
                    (*rows.start()).max(oy)..=(*rows.end()).min(oy + i64::from(tile.size[1]) - 1);
                let tile_cols = ox..ox + i64::from(tile.size[0]);
                count.add_spans(outline, tile_rows, tile_cols, |x, y| pixel(&tile, x, y));
            }
        }
    }
    if count.total == 0 {
        for p in outline {
            let (x, y) = pixel_under(p, w, h);
            let tile = source.tile(tile_of(x, y))?;
            count.add(pixel(&tile, x, y));
        }
    }
    Ok(Some(count.share()))
}

/// Whether a pixel is clipped: `0` or `255` in any of its first three
/// channels.
fn clipped_pixel(pixel: &[u8]) -> bool {
    pixel[..pixel.len().min(3)]
        .iter()
        .any(|&v| v == 0 || v == u8::MAX)
}

/// The rows of pixel centres an outline can hold on a level `h` rows high.
fn outline_rows(outline: &[[f64; 2]], h: i64) -> std::ops::RangeInclusive<i64> {
    let (mut y0, mut y1) = (f64::MAX, f64::MIN);
    for p in outline {
        y0 = y0.min(p[1]);
        y1 = y1.max(p[1]);
    }
    (y0.floor() as i64).max(0)..=(y1.ceil() as i64).min(h - 1)
}

/// The pixel under an outline's point, clamped to a `w` x `h` level.
fn pixel_under(p: &[f64; 2], w: i64, h: i64) -> (i64, i64) {
    (
        (p[0].floor() as i64).clamp(0, w - 1),
        (p[1].floor() as i64).clamp(0, h - 1),
    )
}

/// The pixels counted inside an outline, and the clipped ones among them.
#[derive(Default)]
struct ClippedCount {
    total: usize,
    clipped: usize,
    /// Where the current row crosses the outline's edges.
    crossings: Vec<f64>,
}

impl ClippedCount {
    fn add(&mut self, clipped: bool) {
        self.total += 1;
        self.clipped += usize::from(clipped);
    }

    fn share(&self) -> f64 {
        self.clipped as f64 / self.total as f64
    }

    /// Count the pixels of rows `rows` and columns `cols` whose centres,
    /// `(x + 0.5, y + 0.5)`, are inside `outline` by the even-odd rule.
    ///
    /// One row of pixel centres at a time: where the row crosses the
    /// outline's edges, sorted. A centre is inside when an odd number of
    /// crossings lie to its right, so the inside of a row is a set of spans
    /// between crossings and each pixel is read once. A tile far from its
    /// camera covers hundreds of thousands of photograph pixels, which a test
    /// of every pixel against every edge would make the evaluation's largest
    /// cost. The crossings do not depend on `cols`, so counts over columns
    /// that partition a row sum to the count over the whole row.
    fn add_spans(
        &mut self,
        outline: &[[f64; 2]],
        rows: std::ops::RangeInclusive<i64>,
        cols: std::ops::Range<i64>,
        is_clipped: impl Fn(i64, i64) -> bool,
    ) {
        for y in rows {
            let yc = y as f64 + 0.5;
            self.crossings.clear();
            let mut j = outline.len() - 1;
            for i in 0..outline.len() {
                let ([xi, yi], [xj, yj]) = (outline[i], outline[j]);
                if (yi > yc) != (yj > yc) {
                    self.crossings.push((xj - xi) * (yc - yi) / (yj - yi) + xi);
                }
                j = i;
            }
            self.crossings.sort_by(f64::total_cmp);
            let m = self.crossings.len();
            // The centres at or past crossing `k - 1` and before crossing `k`
            // have `m - k` crossings to their right.
            for k in 1..m {
                if (m - k).is_multiple_of(2) {
                    continue;
                }
                let (from, to) = (self.crossings[k - 1], self.crossings[k]);
                let mut x = ((from - 0.5).ceil() as i64).max(cols.start);
                while (x as f64 + 0.5) < from {
                    x += 1;
                }
                while x < cols.end && (x as f64 + 0.5) < to {
                    self.total += 1;
                    self.clipped += usize::from(is_clipped(x, y));
                    x += 1;
                }
            }
        }
    }
}
