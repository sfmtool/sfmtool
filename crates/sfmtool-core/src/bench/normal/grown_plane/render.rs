// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The plane-induced render of one view onto the reference's working grid, and
//! the scores [`grown_plane_normal`](super::grown_plane_normal) reads from it.
//!
//! Every sample of the working grid is a ray from the reference camera through
//! a point of the starting patch frame re-anchored on the reference keypoint. A plane
//! through the anchor, the starting normal tilted by `(a, b)` along the patch frame's
//! `u` and `v` axes, cuts each ray at one world point, which is projected into
//! a view through its own lens model and sampled from one level of the view's
//! pyramid, after the view's image shift is added. The reference itself is
//! rendered the same way at no tilt and no shift; its rays meet every plane at
//! the reference pixels, so its tile is the same at every tilt.

use std::cell::{Cell, RefCell};

use nalgebra::Vector3;

use crate::camera::image::ImageU8;
use crate::camera::remap::sample_bilinear_u8;
use crate::camera::sampler::bilinear_mip_level;
use crate::camera::warp_map::{patch_grid_jacobian, singular_values_2x2};
use crate::camera::CameraIntrinsics;
use crate::geometry::RigidTransform;
use crate::numeric::median_in_place;
use crate::patch::cell_registration::CellLayout;
use crate::patch::cloud::OrientedPatch;
use crate::patch::keypoint_localize::shifted_center;
use crate::patch::normal_refine::ProjectedImage;

/// A reference channel whose spread over the samples read is under this, in
/// grey levels, is flat: it is not scored, and a cell flat in every channel
/// is not scored at all.
pub(super) const FLAT_SPREAD: f64 = 0.5;

/// A normalised vector shorter than this is treated as flat.
const TINY_NORM: f64 = 1e-9;

/// The reference's working grid: one ray per sample, the anchor every plane
/// passes through, and the starting patch frame the tilt is stated against.
pub(super) struct Grid {
    /// Samples per side, `W`.
    pub(super) side: usize,
    /// The starting patch frame re-anchored on the reference keypoint: the plane at
    /// no tilt, whose square the grid's samples cover.
    pub(super) anchored_patch_frame: OrientedPatch,
    /// The reference camera's centre.
    origin: Vector3<f64>,
    /// Each sample's ray direction from `origin`, row-major, unnormalised.
    rays: Vec<Vector3<f64>>,
    /// Where the reference keypoint's ray meets the starting patch frame.
    pub(super) anchor: Vector3<f64>,
    /// The starting patch frame's outward normal and in-plane axes.
    normal: Vector3<f64>,
    u: Vector3<f64>,
    v: Vector3<f64>,
    /// The samples of each of the nine cells, `[row * 3 + col]` from the
    /// top-left, over the split `[0, W/3, W − W/3, W]`.
    pub(super) cells: [Vec<usize>; 9],
}

impl Grid {
    /// The grid of `side` samples a side over `frame` re-anchored on the
    /// reference's `keypoint` (on `frame` itself where there is none or its
    /// ray misses the plane), seen from the reference view `reference`.
    pub(super) fn new(
        patch_frame: &OrientedPatch,
        reference: &ProjectedImage<'_>,
        keypoint: Option<[f64; 2]>,
        side: usize,
    ) -> Grid {
        let anchored_patch_frame = anchored(patch_frame, reference, keypoint);
        let origin = reference.cam_from_world.inverse_translation_origin().coords;
        let wpp = 2.0 * anchored_patch_frame.half_extent[0] / side as f64;
        let half = side as f64 / 2.0;
        let mut rays = Vec::with_capacity(side * side);
        for row in 0..side {
            for col in 0..side {
                let (gx, gy) = (col as f64 + 0.5 - half, row as f64 + 0.5 - half);
                rays.push(shifted_center(&anchored_patch_frame, gx, gy, wpp, wpp).coords - origin);
            }
        }
        let bounds = CellLayout::new(side as u32, 0.0).bounds;
        let cells = std::array::from_fn(|k| {
            let (row, col) = (k / 3, k % 3);
            let mut samples = Vec::new();
            for y in bounds[row]..bounds[row + 1] {
                for x in bounds[col]..bounds[col + 1] {
                    samples.push(y * side + x);
                }
            }
            samples
        });
        Grid {
            side,
            origin,
            rays,
            anchor: anchored_patch_frame.center.coords,
            anchored_patch_frame,
            normal: patch_frame.normal(),
            u: patch_frame.u_axis,
            v: patch_frame.v_axis,
            cells,
        }
    }

    /// The plane's unit normal at `tilt`: the starting normal tilted by `a`
    /// along `u` and `b` along `v`.
    pub(super) fn plane_normal(&self, tilt: [f64; 2]) -> Vector3<f64> {
        (self.normal + self.u * tilt[0] + self.v * tilt[1]).normalize()
    }

    /// The samples of the window `2 · half + 1` samples a side centred on the
    /// grid's centre sample `(W / 2, W / 2)`, clipped to the grid. On an even
    /// grid that sample's centre is half a sample right of and below the
    /// anchor.
    pub(super) fn window(&self, half: usize) -> Vec<usize> {
        let c = self.side / 2;
        let lo = c.saturating_sub(half);
        let hi = (c + half + 1).min(self.side);
        let mut rows = Vec::with_capacity((hi - lo) * (hi - lo));
        for y in lo..hi {
            for x in lo..hi {
                rows.push(y * self.side + x);
            }
        }
        rows
    }

    /// Every sample of the grid.
    pub(super) fn all(&self) -> Vec<usize> {
        (0..self.side * self.side).collect()
    }

    /// Where each of `rows` lands in `view`'s photograph, full-resolution px,
    /// through the plane at `tilt`, before any shift; `None` where the ray
    /// meets the plane behind the reference camera or not at all, or the
    /// point does not project.
    pub(super) fn project(
        &self,
        view: &ViewImage<'_>,
        tilt: [f64; 2],
        rows: &[usize],
    ) -> Vec<Option<[f64; 2]>> {
        let n = self.plane_normal(tilt);
        let reach = (self.anchor - self.origin).dot(&n);
        rows.iter()
            .map(|&k| {
                let d = self.rays[k];
                let along = d.dot(&n);
                if along.abs() < 1e-12 {
                    return None;
                }
                let lambda = reach / along;
                if !(lambda > 0.0 && lambda.is_finite()) {
                    return None;
                }
                view.camera
                    .project_homogeneous(view.cam_from_world, self.origin + d * lambda, 1.0)
            })
            .collect()
    }

    /// `view`'s photograph px per world unit at the anchor: the square root
    /// of the area scale of the projection of the starting patch frame there, read
    /// by a central difference of a hundredth of the patch frame's half-extent
    /// along each axis. `NaN` where the anchor's neighbourhood does not
    /// project.
    pub(super) fn scale_px(&self, view: &ViewImage<'_>) -> f64 {
        let h = 0.01 * self.anchored_patch_frame.half_extent[0].max(1e-12);
        let at = |d: Vector3<f64>| {
            view.camera
                .project_homogeneous(view.cam_from_world, self.anchor + d, 1.0)
        };
        let (Some(up), Some(un), Some(vp), Some(vn)) = (
            at(self.u * h),
            at(-self.u * h),
            at(self.v * h),
            at(-self.v * h),
        ) else {
            return f64::NAN;
        };
        let du = [(up[0] - un[0]) / (2.0 * h), (up[1] - un[1]) / (2.0 * h)];
        let dv = [(vp[0] - vn[0]) / (2.0 * h), (vp[1] - vn[1]) / (2.0 * h)];
        (du[0] * dv[1] - du[1] * dv[0]).abs().sqrt()
    }

    /// The working grid's samples per world unit on the starting patch frame.
    pub(super) fn density(&self) -> f64 {
        self.side as f64 / (2.0 * self.anchored_patch_frame.half_extent[0])
    }

    /// Where the anchor lands in `view`'s photograph, full-resolution px.
    pub(super) fn anchor_pixel(&self, view: &ViewImage<'_>) -> Option<[f64; 2]> {
        view.camera
            .project_homogeneous(view.cam_from_world, self.anchor, 1.0)
    }
}

/// `frame` re-anchored on `keypoint` in `view`, or `frame` itself where there
/// is no keypoint or its ray misses the plane.
pub(super) fn anchored(
    patch_frame: &OrientedPatch,
    view: &ProjectedImage<'_>,
    keypoint: Option<[f64; 2]>,
) -> OrientedPatch {
    keypoint
        .and_then(|kp| patch_frame.anchored_at_keypoint(view.camera, view.cam_from_world, kp))
        .unwrap_or_else(|| patch_frame.clone())
}

/// One view's photograph as the step reads it: one pyramid level, chosen once.
pub(super) struct ViewImage<'a> {
    pub(super) camera: &'a CameraIntrinsics,
    pub(super) cam_from_world: &'a RigidTransform,
    /// The pyramid level every sample of this view is read from.
    level: &'a ImageU8,
    /// `2^l` for level `l`: full-resolution px per level px.
    scale: f64,
    /// Channels scored: the photograph's, an alpha channel dropped.
    pub(super) channels: usize,
}

impl<'a> ViewImage<'a> {
    /// `view` read at the level the bilinear-mip rule picks for the map from a
    /// working grid of `side` samples over `patch_frame` into the photograph,
    /// at the grid's centre: `round(log2 σ_major)` for the map's larger
    /// singular value, clamped to the pyramid; level 0 where the centre does
    /// not project.
    pub(super) fn new(view: &ProjectedImage<'a>, patch_frame: &OrientedPatch, side: usize) -> Self {
        let jacobian = patch_grid_jacobian(patch_frame, view.camera, view.cam_from_world, side);
        let level = jacobian
            .map(|j| bilinear_mip_level(singular_values_2x2(j)[0]) as usize)
            .unwrap_or(0)
            .min(view.pyramid.num_levels().saturating_sub(1));
        let image = view.pyramid.level(level);
        ViewImage {
            camera: view.camera,
            cam_from_world: view.cam_from_world,
            level: image,
            scale: (1u64 << level) as f64,
            channels: (image.channels() as usize).clamp(1, 3),
        }
    }

    /// Sample `pixels` (full-resolution px, one per row of `rows`) moved by
    /// `shift` into `out`. A sample is valid where its pixel exists and all
    /// four bilinear taps lie on the photograph; an invalid sample still
    /// carries the edge-clamped value, which only a frozen residual reads.
    pub(super) fn sample(
        &self,
        pixels: &[Option<[f64; 2]>],
        rows: &[usize],
        shift: [f64; 2],
        out: &mut Samples,
    ) {
        let (w, h) = (
            f64::from(self.level.width()),
            f64::from(self.level.height()),
        );
        let ch = out.channels;
        for (&k, pixel) in rows.iter().zip(pixels) {
            let Some(p) = pixel else {
                out.valid[k] = false;
                out.values[k * ch..(k + 1) * ch].fill(0.0);
                continue;
            };
            let x = (p[0] + shift[0]) / self.scale;
            let y = (p[1] + shift[1]) / self.scale;
            out.valid[k] = (0.5..=w - 0.5).contains(&x) && (0.5..=h - 0.5).contains(&y);
            let (xf, yf) = (x as f32, y as f32);
            for c in 0..ch {
                out.values[k * ch + c] = sample_bilinear_u8(self.level, xf, yf, c as u32);
            }
        }
    }
}

/// A render of every sample of the grid: values interleaved by channel, and
/// whether each sample landed on the photograph. Only the rows a render was
/// asked for are written.
#[derive(Clone)]
pub(super) struct Samples {
    pub(super) channels: usize,
    pub(super) values: Vec<f32>,
    pub(super) valid: Vec<bool>,
}

impl Samples {
    pub(super) fn new(n: usize, channels: usize) -> Samples {
        Samples {
            channels,
            values: vec![0.0; n * channels],
            valid: vec![false; n],
        }
    }

    fn at(&self, k: usize, c: usize) -> f64 {
        f64::from(self.values[k * self.channels + c])
    }
}

/// One view rendered on the grid, with the projection of its last tilt kept,
/// so that renders at one tilt and several shifts project once.
pub(super) struct ViewRender<'g, 'a> {
    pub(super) grid: &'g Grid,
    pub(super) view: &'g ViewImage<'a>,
    cache: RefCell<Option<Projected>>,
    /// How many times [`Self::render`] has run, for the detail phases' notes.
    renders: Cell<u64>,
}

/// The projection of one tilt over one set of rows.
struct Projected {
    tilt: [f64; 2],
    rows: Vec<usize>,
    pixels: Vec<Option<[f64; 2]>>,
}

impl<'g, 'a> ViewRender<'g, 'a> {
    pub(super) fn new(grid: &'g Grid, view: &'g ViewImage<'a>) -> Self {
        ViewRender {
            grid,
            view,
            cache: RefCell::new(None),
            renders: Cell::new(0),
        }
    }

    /// How many renders this view has made so far.
    pub(super) fn renders(&self) -> u64 {
        self.renders.get()
    }

    /// The view's samples over `rows` through the plane at `tilt`, shifted by
    /// `shift`.
    pub(super) fn render(&self, tilt: [f64; 2], shift: [f64; 2], rows: &[usize]) -> Samples {
        self.renders.set(self.renders.get() + 1);
        let mut out = Samples::new(self.grid.side * self.grid.side, self.view.channels);
        let mut cache = self.cache.borrow_mut();
        let hit = cache
            .as_ref()
            .is_some_and(|c| c.tilt == tilt && c.rows.as_slice() == rows);
        if !hit {
            *cache = Some(Projected {
                tilt,
                rows: rows.to_vec(),
                pixels: self.grid.project(self.view, tilt, rows),
            });
        }
        let projected = cache.as_ref().expect("filled above");
        self.view.sample(&projected.pixels, rows, shift, &mut out);
        out
    }
}

/// The mean per-channel ZNCC of `view` against `reference` over `rows`, read
/// where both samples are valid; `−1` when under `min_coverage` of `rows` are.
/// A channel the reference is flat in over those samples is not scored; with
/// no channel scored the ZNCC is `0`.
pub(super) fn zncc(reference: &Samples, view: &Samples, rows: &[usize], min_coverage: f64) -> f64 {
    let kept: Vec<usize> = rows
        .iter()
        .copied()
        .filter(|&k| reference.valid[k] && view.valid[k])
        .collect();
    if rows.is_empty() || (kept.len() as f64) < min_coverage * rows.len() as f64 || kept.len() < 2 {
        return -1.0;
    }
    let channels = reference.channels.min(view.channels);
    let (mut sum, mut scored) = (0.0, 0usize);
    for c in 0..channels {
        let Some(z) = channel_zncc(reference, view, &kept, c) else {
            continue;
        };
        sum += z;
        scored += 1;
    }
    if scored == 0 {
        0.0
    } else {
        sum / scored as f64
    }
}

/// One channel's ZNCC over `kept`, or `None` where the reference is flat.
fn channel_zncc(reference: &Samples, view: &Samples, kept: &[usize], c: usize) -> Option<f64> {
    let n = kept.len() as f64;
    let (mut ma, mut mb) = (0.0, 0.0);
    for &k in kept {
        ma += reference.at(k, c);
        mb += view.at(k, c);
    }
    let (ma, mb) = (ma / n, mb / n);
    let (mut saa, mut sbb, mut sab) = (0.0, 0.0, 0.0);
    for &k in kept {
        let (a, b) = (reference.at(k, c) - ma, view.at(k, c) - mb);
        saa += a * a;
        sbb += b * b;
        sab += a * b;
    }
    if (saa / n).sqrt() < FLAT_SPREAD {
        return None;
    }
    let d = (saa * sbb).sqrt();
    Some(if d > TINY_NORM { sab / d } else { 0.0 })
}

/// Whether the reference cell `rows` is textured: its spread is at least
/// [`FLAT_SPREAD`] in some channel, over its valid samples.
pub(super) fn textured(reference: &Samples, rows: &[usize]) -> bool {
    let kept: Vec<usize> = rows
        .iter()
        .copied()
        .filter(|&k| reference.valid[k])
        .collect();
    if kept.len() < 2 {
        return false;
    }
    let n = kept.len() as f64;
    (0..reference.channels).any(|c| {
        let mean = kept.iter().map(|&k| reference.at(k, c)).sum::<f64>() / n;
        let var = kept
            .iter()
            .map(|&k| (reference.at(k, c) - mean).powi(2))
            .sum::<f64>()
            / n;
        var.sqrt() >= FLAT_SPREAD
    })
}

/// The robust combination of the cells' ZNCC: `None` for a flat cell. Returns
/// the score and each cell's weight; the score is `−1` with no live cell.
///
/// One-sided Tukey: `m` is the median of the live cells' ZNCC, the scale is
/// `1.4826` times the median absolute deviation from `m` of the cells at or
/// above `m`, floored at `0.05`, and the cut is `4.685` scales. A cell at or
/// above `m` weighs 1 and one below weighs `(1 − ((m − z)/cut)²)²`, `0` past
/// the cut. The scale is read from the upper cells only because with several
/// cells that do not match, the spread of all of them is so wide that none
/// falls past the cut.
pub(super) fn robust_score(cells: &[Option<f64>; 9]) -> (f64, [f64; 9]) {
    let mut weights = [0.0; 9];
    let live: Vec<(usize, f64)> = cells
        .iter()
        .enumerate()
        .filter_map(|(k, z)| z.map(|z| (k, z)))
        .collect();
    if live.is_empty() {
        return (-1.0, weights);
    }
    let mut values: Vec<f64> = live.iter().map(|&(_, z)| z).collect();
    let m = median_in_place(&mut values);
    let mut upper: Vec<f64> = values
        .iter()
        .filter(|&&z| z >= m)
        .map(|&z| (z - m).abs())
        .collect();
    let mad = if upper.is_empty() {
        0.0
    } else {
        median_in_place(&mut upper)
    };
    let cut = 4.685 * (1.4826 * mad).max(0.05);
    let (mut total, mut weighted) = (0.0, 0.0);
    for &(k, z) in &live {
        let w = if z >= m {
            1.0
        } else {
            let t = (m - z) / cut;
            (1.0 - t * t).max(0.0).powi(2)
        };
        weights[k] = w;
        total += w;
        weighted += w * z;
    }
    if total <= 0.0 {
        return (-1.0, weights);
    }
    (weighted / total, weights)
}

/// The normalisation of one block of samples frozen at one render: which
/// samples and channels it reads, the reference's normalised values, and the
/// block's weight in the residual.
///
/// For a channel with centred reference `a` and view `b` over the same
/// samples, `â = a/‖a‖` and `b̂ = b/‖b‖` are unit vectors and
/// `‖â − b̂‖² = 2 − 2·ZNCC`. Scaled by `√(weight / channels)`, a block's
/// residuals sum to `weight · (2 − 2·z)` with `z` its mean-channel ZNCC, so a
/// set of blocks whose weights sum to 1 has a residual sum of squares of
/// `2 · (1 − score)` for the weighted mean score.
pub(super) struct Block {
    rows: Vec<usize>,
    channels: Vec<usize>,
    scale: f64,
    /// `channels.len() × rows.len()` normalised reference values.
    reference: Vec<f64>,
}

impl Block {
    /// The block over the samples of `rows` valid in both renders, with the
    /// channels the reference is not flat in, weighted `weight`; `None` under
    /// `min_coverage` or with no channel to read.
    pub(super) fn freeze(
        reference: &Samples,
        view: &Samples,
        rows: &[usize],
        weight: f64,
        min_coverage: f64,
    ) -> Option<Block> {
        let kept: Vec<usize> = rows
            .iter()
            .copied()
            .filter(|&k| reference.valid[k] && view.valid[k])
            .collect();
        if rows.is_empty()
            || (kept.len() as f64) < min_coverage * rows.len() as f64
            || kept.len() < 2
        {
            return None;
        }
        let mut channels = Vec::new();
        let mut normalised = Vec::new();
        for c in 0..reference.channels.min(view.channels) {
            let values: Vec<f64> = kept.iter().map(|&k| reference.at(k, c)).collect();
            let unit = unit_centred(&values);
            let spread = (values.iter().map(|v| v * v).sum::<f64>() / values.len() as f64
                - (values.iter().sum::<f64>() / values.len() as f64).powi(2))
            .max(0.0)
            .sqrt();
            if spread < FLAT_SPREAD {
                continue;
            }
            channels.push(c);
            normalised.extend(unit);
        }
        if channels.is_empty() || weight <= 0.0 {
            return None;
        }
        Some(Block {
            scale: (weight / channels.len() as f64).sqrt(),
            rows: kept,
            channels,
            reference: normalised,
        })
    }

    /// Append the block's residuals for `view` to `out`.
    pub(super) fn residuals(&self, view: &Samples, out: &mut Vec<f64>) {
        let n = self.rows.len();
        for (j, &c) in self.channels.iter().enumerate() {
            let values: Vec<f64> = self.rows.iter().map(|&k| view.at(k, c)).collect();
            let unit = unit_centred(&values);
            let reference = &self.reference[j * n..(j + 1) * n];
            out.extend(
                reference
                    .iter()
                    .zip(&unit)
                    .map(|(a, b)| self.scale * (a - b)),
            );
        }
    }
}

impl Block {
    /// Append the block's residuals between two views' renders to `out`:
    /// `first` normalised afresh over the block's samples and channels in
    /// place of the frozen reference, less `second` normalised the same way.
    pub(super) fn pair_residuals(&self, first: &Samples, second: &Samples, out: &mut Vec<f64>) {
        for &c in &self.channels {
            let a: Vec<f64> = self.rows.iter().map(|&k| first.at(k, c)).collect();
            let b: Vec<f64> = self.rows.iter().map(|&k| second.at(k, c)).collect();
            out.extend(
                unit_centred(&a)
                    .iter()
                    .zip(&unit_centred(&b))
                    .map(|(x, y)| self.scale * (x - y)),
            );
        }
    }
}

/// `values` less their mean, divided by the result's length; zeros where
/// that length is under [`TINY_NORM`].
fn unit_centred(values: &[f64]) -> Vec<f64> {
    let mean = values.iter().sum::<f64>() / values.len().max(1) as f64;
    let centred: Vec<f64> = values.iter().map(|v| v - mean).collect();
    let norm = centred.iter().map(|v| v * v).sum::<f64>().sqrt();
    if norm < TINY_NORM {
        vec![0.0; values.len()]
    } else {
        centred.into_iter().map(|v| v / norm).collect()
    }
}
