// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The ZNCC self-similarity radius: how far a patch can slide over itself, by
//! whole pixels, and still look as much like itself as a true match between
//! two views would.
//!
//! See `specs/core/patch/zncc-self-similarity-radius.md`. For a template
//! rectangle inside a tile, [`zncc_self_similarity_radius`] computes the
//! channel-averaged ZNCC of the template against the window of the same size
//! moved by every shift `d` in the disk `dx² + dy² ≤ r²`, counts a shift as
//! indistinguishable when `1 − z(d) ≤ ε + mean_c (n / s_c)²`, and reports the
//! length of the furthest such shift (saturating at `r` when one lies in the
//! disk's outer ring) and the direction the indistinguishable shifts line up
//! in. [`zncc_self_similarity_parts`] reads a whole `R×R` core, its middle
//! square and the nine cells of the ZNCC grid's split from one tile, sharing
//! the cross sums between them.
//!
//! The tile is centred by its own mean per channel before the kernels run, so
//! the `f32` cross sums keep their precision; the window moments come from
//! per-channel summed-area tables in `f64`, and the combine, the tolerance
//! test and the slide run in `f64`.

mod kernels;

#[cfg(test)]
mod tests;

use crate::patch::normal_refine::{grid_bounds, middle_span, FLAT_NORM_SQ_EPS};

/// The template spread, in grey levels, under which a channel carries no
/// texture: it is left out of the channel average, and a template with no
/// channel at or above it scores `max_radius`.
pub const FLAT_FLOOR: f64 = 0.5;

/// How far a patch can slide over itself and stay indistinguishable from its
/// true position.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SelfSimilarityParams {
    /// The length of the largest shift searched, in tile pixels; a radius of
    /// `max_radius` reads "this far or further".
    pub max_radius: u32,
    /// ε: the ZNCC deficit two views of the same surface show from warp, blur
    /// and lighting, as a fraction.
    pub relative_tolerance: f64,
    /// n: the noise between two views, in grey levels.
    pub noise: f64,
}

impl Default for SelfSimilarityParams {
    fn default() -> Self {
        Self {
            max_radius: 3,
            relative_tolerance: 0.05,
            noise: 2.0,
        }
    }
}

/// One template's reading.
#[derive(Debug, Clone, PartialEq)]
pub struct SelfSimilarity {
    /// The length of the furthest indistinguishable shift, `0 ..= max_radius`,
    /// in tile pixels; `max_radius` when one lies in the window's outer ring,
    /// or when the template has no textured channel.
    pub radius: f64,
    /// The direction of the indistinguishable shifts, in the grid frame (`x`
    /// column-right, `y` row-down), scaled by how strongly they line up:
    /// the unit eigenvector of their second moment's larger eigenvalue `μ₁`
    /// times `1 − μ₂/μ₁`. `[0, 0]` when there are none, and for a template
    /// with no textured channel. Its sign means nothing.
    pub slide: [f64; 2],
    /// The tolerance τ the template was judged by; `f64::INFINITY` for a
    /// template with no textured channel, where every shift counts.
    pub tolerance: f64,
    /// The channel-averaged ZNCC `z(d)` for every shift of the `(2r + 1)²`
    /// square, row-major from `(dx, dy) = (−r, −r)`: `z(0, 0) = 1`, and `NaN`
    /// for the shifts outside the disk. Every value is `NaN` for a template
    /// with no textured channel.
    pub surface: Vec<f64>,
}

/// A tile's whole core, its middle square and the nine cells of the ZNCC
/// grid's split, each read as its own template.
#[derive(Debug, Clone, PartialEq)]
pub struct SelfSimilarityParts {
    /// The whole `R×R` core.
    pub whole: SelfSimilarity,
    /// The middle square, the core's rows and columns `R/4 .. R − R/4`.
    pub middle: SelfSimilarity,
    /// Each cell of the core's split at `R/3` and `R − R/3`, `grid[row][col]`
    /// from the top-left cell.
    pub grid: [[SelfSimilarity; 3]; 3],
}

/// A tile, planar: channel `c`'s row-major `width × height` plane at
/// `values[c * width * height ..]`. Up to three colour channels.
#[derive(Debug, Clone, Copy)]
pub struct PatchTile<'a> {
    pub values: &'a [f32],
    pub channels: usize,
    pub width: usize,
    pub height: usize,
}

impl<'a> PatchTile<'a> {
    /// The planar colour channels of an interleaved `height × width × C`
    /// patch, the layout a rendered bitmap arrives in, dropping a fourth
    /// channel (alpha): returns the planes and the colour channel count.
    ///
    /// # Panics
    ///
    /// Panics if `patch.len() != width * height * channels` or `channels` is 0.
    pub fn planes_from_interleaved(
        patch: &[f32],
        width: usize,
        height: usize,
        channels: usize,
    ) -> (Vec<f32>, usize) {
        assert!(channels > 0, "planes_from_interleaved: no channels");
        assert_eq!(
            patch.len(),
            width * height * channels,
            "planes_from_interleaved: a {width}×{height}×{channels} patch has {} values, not {}",
            width * height * channels,
            patch.len()
        );
        let colour = channels.min(3);
        let n = width * height;
        let mut planes = vec![0.0f32; colour * n];
        for (p, pixel) in patch.chunks_exact(channels).enumerate() {
            for c in 0..colour {
                planes[c * n + p] = pixel[c];
            }
        }
        (planes, colour)
    }

    /// Check the tile's own shape: one to three channels and a value for
    /// every pixel of every channel.
    fn validate(&self) {
        assert!(
            (1..=3).contains(&self.channels),
            "PatchTile: {} channels, expected 1 to 3 colour channels",
            self.channels
        );
        assert_eq!(
            self.values.len(),
            self.channels * self.width * self.height,
            "PatchTile: a {}×{}×{} tile has {} values, not {}",
            self.width,
            self.height,
            self.channels,
            self.channels * self.width * self.height,
            self.values.len()
        );
    }
}

/// One template `Ω = (x, y, w, h)` inside `tile`, with `max_radius` pixels of
/// tile around it on every side.
///
/// # Panics
///
/// Panics if the tile is malformed, the template is empty, or the template has
/// less than `max_radius` pixels of tile on some side; the message names the
/// sizes.
pub fn zncc_self_similarity_radius(
    tile: &PatchTile<'_>,
    template: [usize; 4],
    params: &SelfSimilarityParams,
) -> SelfSimilarity {
    radius_with(tile, template, params, Kernel::Dispatch)
}

/// The core `R×R` square centred in a `(R + 2r)²` tile, read whole, over its
/// middle `R/4 .. R − R/4` and over each cell of the `R/3`, `R − R/3` split.
///
/// # Panics
///
/// Panics if the tile is malformed, `resolution < 3`, or the tile is not
/// `(resolution + 2·max_radius)` square; the message names the sizes.
pub fn zncc_self_similarity_parts(
    tile: &PatchTile<'_>,
    resolution: usize,
    params: &SelfSimilarityParams,
) -> SelfSimilarityParts {
    parts_with(tile, resolution, params, Kernel::Dispatch)
}

/// Which cross-sum kernel to run: the dispatching one, or the scalar reference
/// (for the equivalence test).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kernel {
    Dispatch,
    #[cfg_attr(not(test), allow(dead_code))]
    Scalar,
}

fn radius_with(
    tile: &PatchTile<'_>,
    template: [usize; 4],
    params: &SelfSimilarityParams,
    kernel: Kernel,
) -> SelfSimilarity {
    tile.validate();
    let r = params.max_radius as usize;
    let [x, y, w, h] = template;
    assert!(
        w > 0 && h > 0,
        "zncc_self_similarity_radius: the template is {w}×{h}, it needs at least one pixel"
    );
    assert!(
        x >= r && y >= r && x + w + r <= tile.width && y + h + r <= tile.height,
        "zncc_self_similarity_radius: the {w}×{h} template at ({x}, {y}) needs {r} px of tile \
         on every side, but the tile is {}×{}",
        tile.width,
        tile.height
    );
    let prepared = Prepared::new(tile);
    let cross = prepared.cross_sums(template, r, kernel);
    prepared.judge(template, r, &cross, params)
}

fn parts_with(
    tile: &PatchTile<'_>,
    resolution: usize,
    params: &SelfSimilarityParams,
    kernel: Kernel,
) -> SelfSimilarityParts {
    tile.validate();
    let r = params.max_radius as usize;
    assert!(
        resolution >= 3,
        "zncc_self_similarity_parts: resolution {resolution} is too small to split in three"
    );
    let side = resolution + 2 * r;
    assert!(
        tile.width == side && tile.height == side,
        "zncc_self_similarity_parts: an R = {resolution} core with r = {r} needs a {side}×{side} \
         tile, but the tile is {}×{}",
        tile.width,
        tile.height
    );
    let prepared = Prepared::new(tile);
    let bounds = grid_bounds(resolution as u32);
    let cell_rect = |row: usize, col: usize| {
        [
            r + bounds[col],
            r + bounds[row],
            bounds[col + 1] - bounds[col],
            bounds[row + 1] - bounds[row],
        ]
    };
    // The nine cells tile the core, so the core's cross sums are theirs added.
    let cell_sums: [[Vec<f64>; 3]; 3] = std::array::from_fn(|row| {
        std::array::from_fn(|col| prepared.cross_sums(cell_rect(row, col), r, kernel))
    });
    let mut whole_sums = vec![0.0f64; cell_sums[0][0].len()];
    for sums in cell_sums.iter().flatten() {
        for (total, value) in whole_sums.iter_mut().zip(sums) {
            *total += value;
        }
    }
    let middle = middle_span(resolution as u32);
    let middle_rect = [
        r + middle.start,
        r + middle.start,
        middle.len(),
        middle.len(),
    ];
    let middle_sums = prepared.cross_sums(middle_rect, r, kernel);
    SelfSimilarityParts {
        whole: prepared.judge([r, r, resolution, resolution], r, &whole_sums, params),
        middle: prepared.judge(middle_rect, r, &middle_sums, params),
        grid: std::array::from_fn(|row| {
            std::array::from_fn(|col| {
                prepared.judge(cell_rect(row, col), r, &cell_sums[row][col], params)
            })
        }),
    }
}

/// The most template rows one `f32` kernel call accumulates before its sums
/// are added into the `f64` totals, which bounds the `f32` rounding a large
/// template accumulates.
const BAND_ROWS: usize = 8;

/// A tile ready to score: each channel centred by its own mean and copied into
/// a plane with 8 zero columns of padding per row, and the summed-area tables
/// of the centred values and their squares.
struct Prepared {
    /// `channels` planes of `stride × height`, centred.
    planes: Vec<f32>,
    stride: usize,
    channels: usize,
    width: usize,
    height: usize,
    /// Per channel, the `(width + 1) × (height + 1)` summed-area table of the
    /// centred values, and of their squares.
    sat1: Vec<f64>,
    sat2: Vec<f64>,
}

impl Prepared {
    fn new(tile: &PatchTile<'_>) -> Self {
        let (width, height, channels) = (tile.width, tile.height, tile.channels);
        let n = width * height;
        let stride = width + 8;
        let sat_side = (width + 1) * (height + 1);
        let mut planes = vec![0.0f32; channels * stride * height];
        let mut sat1 = vec![0.0f64; channels * sat_side];
        let mut sat2 = vec![0.0f64; channels * sat_side];
        for c in 0..channels {
            let values = &tile.values[c * n..][..n];
            let mean = if n == 0 {
                0.0
            } else {
                values.iter().map(|&v| f64::from(v)).sum::<f64>() / n as f64
            };
            let plane = &mut planes[c * stride * height..][..stride * height];
            let s1 = &mut sat1[c * sat_side..][..sat_side];
            let s2 = &mut sat2[c * sat_side..][..sat_side];
            for y in 0..height {
                let (mut row1, mut row2) = (0.0f64, 0.0f64);
                for x in 0..width {
                    let v = (f64::from(values[y * width + x]) - mean) as f32;
                    plane[y * stride + x] = v;
                    let v = f64::from(v);
                    row1 += v;
                    row2 += v * v;
                    let at = (y + 1) * (width + 1) + x + 1;
                    s1[at] = s1[at - (width + 1)] + row1;
                    s2[at] = s2[at - (width + 1)] + row2;
                }
            }
        }
        Self {
            planes,
            stride,
            channels,
            width,
            height,
            sat1,
            sat2,
        }
    }

    fn plane(&self, c: usize) -> &[f32] {
        &self.planes[c * self.stride * self.height..][..self.stride * self.height]
    }

    /// `Σ v` and `Σ v²` of channel `c`'s centred values over the rectangle
    /// `[x, y, w, h]`.
    fn rect_sums(&self, c: usize, x: usize, y: usize, w: usize, h: usize) -> (f64, f64) {
        let side = (self.width + 1) * (self.height + 1);
        let at = |sat: &[f64], xx: usize, yy: usize| sat[c * side + yy * (self.width + 1) + xx];
        let sum = |sat: &[f64]| {
            at(sat, x + w, y + h) - at(sat, x, y + h) - at(sat, x + w, y) + at(sat, x, y)
        };
        (sum(&self.sat1), sum(&self.sat2))
    }

    /// The raw cross sums of a template rectangle, per channel, `(2r + 1)²`
    /// shifts each: channel `c`'s block at `c·(2r + 1)²`, in `f64`, the kernel
    /// run over bands of at most [`BAND_ROWS`] template rows.
    fn cross_sums(&self, rect: [usize; 4], r: usize, kernel: Kernel) -> Vec<f64> {
        let [x, y, w, h] = rect;
        let shifts = (2 * r + 1) * (2 * r + 1);
        let mut totals = vec![0.0f64; self.channels * shifts];
        let mut band = vec![0.0f32; shifts];
        for c in 0..self.channels {
            let plane = self.plane(c);
            let mut top = y;
            while top < y + h {
                let rows = BAND_ROWS.min(y + h - top);
                let band_rect = [x, top, w, rows];
                match kernel {
                    Kernel::Dispatch => {
                        kernels::cross_sums(plane, self.stride, band_rect, r, &mut band)
                    }
                    Kernel::Scalar => {
                        kernels::cross_sums_scalar(plane, self.stride, band_rect, r, &mut band)
                    }
                }
                for (total, &value) in totals[c * shifts..][..shifts].iter_mut().zip(&band) {
                    *total += f64::from(value);
                }
                top += rows;
            }
        }
        totals
    }

    /// Judge one template rectangle from its raw cross sums: its ZNCC surface,
    /// the tolerance, the radius and the slide.
    fn judge(
        &self,
        rect: [usize; 4],
        r: usize,
        cross: &[f64],
        params: &SelfSimilarityParams,
    ) -> SelfSimilarity {
        let [x, y, w, h] = rect;
        let side = 2 * r + 1;
        let shifts = side * side;
        let count = (w * h) as f64;
        // The textured channels: (channel, template mean, template Σ t̃²).
        let textured: Vec<(usize, f64, f64)> = (0..self.channels)
            .filter_map(|c| {
                let (s1, s2) = self.rect_sums(c, x, y, w, h);
                let mean = s1 / count;
                let norm = (s2 - s1 * mean).max(0.0);
                ((norm / count).sqrt() >= FLAT_FLOOR).then_some((c, mean, norm))
            })
            .collect();
        if textured.is_empty() {
            return SelfSimilarity {
                radius: r as f64,
                slide: [0.0; 2],
                tolerance: f64::INFINITY,
                surface: vec![f64::NAN; shifts],
            };
        }
        let noise_term = textured
            .iter()
            .map(|&(_, _, norm)| params.noise * params.noise * count / norm)
            .sum::<f64>()
            / textured.len() as f64;
        let tolerance = params.relative_tolerance + noise_term;

        let ri = r as i64;
        let inner = (ri - 1) * (ri - 1);
        let mut surface = vec![f64::NAN; shifts];
        let mut furthest: i64 = 0;
        let mut saturated = false;
        let (mut sxx, mut sxy, mut syy) = (0.0f64, 0.0f64, 0.0f64);
        for dy in -ri..=ri {
            for dx in -ri..=ri {
                let d2 = dx * dx + dy * dy;
                if d2 > ri * ri {
                    continue;
                }
                let index = ((dy + ri) as usize) * side + (dx + ri) as usize;
                if d2 == 0 {
                    surface[index] = 1.0;
                    continue;
                }
                let (wx, wy) = ((x as i64 + dx) as usize, (y as i64 + dy) as usize);
                let z = textured
                    .iter()
                    .map(|&(c, mean, norm)| {
                        let (s1, s2) = self.rect_sums(c, wx, wy, w, h);
                        let window_norm = s2 - s1 * s1 / count;
                        if window_norm < FLAT_NORM_SQ_EPS {
                            // A flat window is plainly different from a
                            // textured template.
                            0.0
                        } else {
                            (cross[c * shifts + index] - mean * s1) / (norm * window_norm).sqrt()
                        }
                    })
                    .sum::<f64>()
                    / textured.len() as f64;
                surface[index] = z;
                if 1.0 - z <= tolerance {
                    furthest = furthest.max(d2);
                    saturated |= d2 > inner;
                    let (fx, fy) = (dx as f64, dy as f64);
                    sxx += fx * fx;
                    sxy += fx * fy;
                    syy += fy * fy;
                }
            }
        }
        let radius = if saturated {
            r as f64
        } else {
            (furthest as f64).sqrt()
        };
        SelfSimilarity {
            radius,
            slide: slide_of(sxx, sxy, syy),
            tolerance,
            surface,
        }
    }
}

/// The slide of a set of shifts from the sums of their second moments: the
/// unit eigenvector of the larger eigenvalue `μ₁` scaled by `1 − μ₂/μ₁`, or
/// `[0, 0]` for an empty set. The common `1/|A|` factor cancels, so the raw
/// sums serve.
fn slide_of(sxx: f64, sxy: f64, syy: f64) -> [f64; 2] {
    let half_trace = 0.5 * (sxx + syy);
    let spread = (0.25 * (sxx - syy) * (sxx - syy) + sxy * sxy).sqrt();
    let (mu1, mu2) = (half_trace + spread, half_trace - spread);
    if mu1 <= 0.0 {
        return [0.0; 2];
    }
    let strength = (1.0 - mu2.max(0.0) / mu1).clamp(0.0, 1.0);
    let theta = 0.5 * (2.0 * sxy).atan2(sxx - syy);
    [strength * theta.cos(), strength * theta.sin()]
}
