// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! One view's `R×R` tile and the readings taken while rendering it: which of
//! its samples carry image data, the share of the photograph under it that is
//! clipped, and the angle the view sees the patch at.

use ndarray::Array3;

use crate::camera::image::ImageU8Pyramid;
use crate::camera::sampler::{render_phase, render_tile, Sampler, SamplerChoice};
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
    /// the warp cannot place on the photograph is black, and a fourth channel
    /// is opaque everywhere, as a stored bitmap has it.
    pub samples: Array3<u8>,
    /// Per sample, row-major: whether the warp places it on the photograph,
    /// so that it carries image data.
    pub valid: Vec<bool>,
    /// The placement the tile is rendered through: the patch re-anchored on
    /// the keypoint ([`OrientedPatch::anchored_at_keypoint`]), or the patch
    /// itself where there is no keypoint or its ray misses the patch's plane.
    pub placement: OrientedPatch,
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
    /// The angle the view sees the placement at, at its centre
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
/// there is one, with the sampler `sampler` picks from the placement's
/// Jacobian, and read its coverage, the clipped share of the photograph under
/// it and the viewing angle.
///
/// The tile is the one the bench's self-similarity reading takes and the one
/// every stored patch bitmap is rendered as: the keypoint-anchored placement,
/// the sampler rule's choice at the patch resolution, and black where the warp
/// leaves the photograph. The render is timed in its sampler's detail phase of
/// `progress` ([`render_phase`]).
///
/// ```no_run
/// # use sfmtool_core::camera::sampler::SamplerChoice;
/// # use sfmtool_core::patch::cloud::OrientedPatch;
/// # use sfmtool_core::patch::normal_refine::ProjectedImage;
/// # use sfmtool_core::patch::reference_view::render_view_tile;
/// # use sfmtool_core::progress::Progress;
/// # fn run(patch: &OrientedPatch, view: &ProjectedImage<'_>) {
/// let tile = render_view_tile(
///     patch,
///     view,
///     Some([812.4, 377.9]),
///     24,
///     SamplerChoice::per_view(),
///     &Progress::none(),
/// );
/// println!("coverage {:.2}, clipped {:?}", tile.coverage, tile.clipped_share);
/// # }
/// ```
pub fn render_view_tile(
    patch: &OrientedPatch,
    view: &ProjectedImage<'_>,
    keypoint: Option<[f64; 2]>,
    resolution: usize,
    sampler: SamplerChoice,
    progress: &Progress<'_>,
) -> ViewTile {
    let resolution = resolution.max(2);
    let placement = keypoint
        .and_then(|kp| patch.anchored_at_keypoint(view.camera, view.cam_from_world, kp))
        .unwrap_or_else(|| patch.clone());
    let jacobian = patch_grid_jacobian(&placement, view.camera, view.cam_from_world, resolution);
    let sampler = sampler.for_jacobian(jacobian);
    let channels = view.pyramid.level(0).channels() as usize;
    let mut map = WarpMap::from_patch(
        &placement,
        view.camera,
        view.cam_from_world,
        resolution as u32,
    );
    let r = resolution as u32;
    let valid: Vec<bool> = (0..r * r).map(|k| map.is_valid(k % r, k / r)).collect();
    let outline = tile_outline(&map);
    let tile = {
        let _phase = render_phase(progress, sampler, 1);
        render_tile(view.pyramid, &mut map, sampler)
    };
    let src_channels = tile.channels();
    let mut samples = Array3::<u8>::zeros((resolution, resolution, channels));
    for row in 0..resolution {
        for col in 0..resolution {
            for c in 0..channels {
                samples[[row, col, c]] = if c >= 3 {
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
    ViewTile {
        samples,
        valid,
        clipped_share: clipped_share(view.pyramid, &outline),
        viewing_angle: viewing_angle(&placement, view.cam_from_world),
        placement,
        jacobian,
        sampler,
        coverage,
    }
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
    let is_clipped = |x: i64, y: i64| {
        let at = (y * w + x) as usize * ch;
        data[at..at + ch.min(3)]
            .iter()
            .any(|&v| v == 0 || v == u8::MAX)
    };
    let (mut y0, mut y1) = (f64::MAX, f64::MIN);
    for p in outline {
        y0 = y0.min(p[1]);
        y1 = y1.max(p[1]);
    }
    // One row of pixel centres at a time: where the row crosses the outline's
    // edges, sorted. A centre is inside when an odd number of crossings lie to
    // its right, the even-odd rule, so the inside of a row is a set of spans
    // between crossings and each pixel is read once. A tile far from its
    // camera covers hundreds of thousands of photograph pixels, which a test
    // of every pixel against every edge would make the evaluation's largest
    // cost.
    let (mut total, mut clipped) = (0usize, 0usize);
    let mut crossings: Vec<f64> = Vec::new();
    for y in (y0.floor() as i64).max(0)..=(y1.ceil() as i64).min(h - 1) {
        let yc = y as f64 + 0.5;
        crossings.clear();
        let mut j = outline.len() - 1;
        for i in 0..outline.len() {
            let ([xi, yi], [xj, yj]) = (outline[i], outline[j]);
            if (yi > yc) != (yj > yc) {
                crossings.push((xj - xi) * (yc - yi) / (yj - yi) + xi);
            }
            j = i;
        }
        crossings.sort_by(f64::total_cmp);
        let m = crossings.len();
        // The centres at or past crossing `k - 1` and before crossing `k` have
        // `m - k` crossings to their right.
        for k in 1..m {
            if (m - k).is_multiple_of(2) {
                continue;
            }
            let (from, to) = (crossings[k - 1], crossings[k]);
            let mut x = ((from - 0.5).ceil() as i64).max(0);
            while (x as f64 + 0.5) < from {
                x += 1;
            }
            while x < w && (x as f64 + 0.5) < to {
                total += 1;
                clipped += usize::from(is_clipped(x, y));
                x += 1;
            }
        }
    }
    if total == 0 {
        for p in outline {
            let x = (p[0].floor() as i64).clamp(0, w - 1);
            let y = (p[1].floor() as i64).clamp(0, h - 1);
            total += 1;
            clipped += usize::from(is_clipped(x, y));
        }
    }
    Some(clipped as f64 / total as f64)
}
