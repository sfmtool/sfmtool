// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The crops of one round: every view rendered onto the round's plane through
//! the point, on one grid all the views share.
//!
//! [`CropGrid::new`] sizes the grid from the views' pixels on the plane at the
//! point and leaves out the views too oblique to it; [`render_crops`] renders
//! each remaining view's crop through its camera model with
//! [`WarpMap::from_patch`], read with the sampler rule
//! ([`SamplerChoice::per_view`]), so a view finer than the grid is read
//! through its photograph's pyramid and one coarser is upsampled bilinearly;
//! [`CropPyramid`] is a crop's Gaussian pyramid, the flow's own
//! [`ImagePyramid`], and [`PairLevel`] the level of it a pair is read at.
//!
//! A crop's grey values are `f32`, as the flow's [`GrayImage`] is: flow errors
//! are 0.04 px or more, far above `f32`'s rounding at crop coordinates under
//! a few thousand samples. The geometry that sizes the grid is `f64`, as the
//! camera models are.

use nalgebra::{Point3, Vector3};
use rayon::prelude::*;

use crate::camera::sampler::{render_tile, SamplerChoice};
use crate::camera::warp_map::{patch_grid_jacobian, singular_values_2x2};
use crate::camera::WarpMap;
use crate::features::optical_flow::{matcher_gray_from_rgb, GrayImage, ImagePyramid};
use crate::patch::cloud::OrientedPatch;
use crate::patch::normal_refine::ProjectedImage;

use super::model::CommonPlane;
use super::PairwiseFlowNormalOptions;

/// The side, in samples, of the probe tile whose central Jacobian measures a
/// view's pixel on the plane. Its samples are a hundredth of the patch's
/// half-extent apart, so the finite difference reads the camera model over a
/// small fraction of the patch.
const PROBE_RESOLUTION: u32 = 200;

/// A patch axis within this sine of the plane's normal is too close to it to
/// give the crops' first in-plane axis; the other patch axis is used.
const MIN_AXIS_SINE: f64 = 0.1; // about 6 degrees

/// How a view stands to a round's plane at the point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum ViewSight {
    /// The view sees the plane at the point within the round's largest view
    /// angle.
    Seen {
        /// Degrees between the view's line of sight and the plane's normal.
        angle_deg: f64,
        /// The world size, on the plane at the point, of one of the view's own
        /// pixels along its coarser axis there: the reciprocal of the smaller
        /// singular value of the plane-to-photograph Jacobian.
        pixel_size: f64,
    },
    /// More than `max_view_angle_deg` from the plane's normal.
    TooOblique {
        /// Degrees between the view's line of sight and the plane's normal.
        angle_deg: f64,
    },
    /// The camera's centre is at the point, or the point does not project
    /// into the camera's model.
    NoSight,
}

/// The grid one round's crops share: the plane they are rendered on, its
/// sample spacing and the crops' half-width.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct CropGrid {
    /// The round's plane through the point, its axes and the grid spacing.
    pub(crate) plane: CommonPlane,
    /// Samples along one side of the patch: by the rule, or fewer where the
    /// crop cap coarsened the grid.
    pub(crate) samples_per_patch: f64,
    /// The crops' half-width, grid samples; a crop is twice this a side.
    pub(crate) half_width: u32,
}

impl CropGrid {
    /// The grid for the plane of unit normal `normal` through `point`, for a
    /// patch of half-extent `half_extent` whose in-plane axis `u` (and `v`
    /// where `u` is within a few degrees of `normal`) sets the crops' axes,
    /// with each of `views`' sight of it. `None` where no view sees the plane
    /// or the inputs are not finite.
    ///
    /// The crops' first axis is `u` projected onto the plane and normalised,
    /// the second `normal × u`; where `u` lies within about 6° of `normal`, `v`
    /// is projected instead and the first axis is `v × normal`. Either way the
    /// crop's `u × v` is `normal`.
    ///
    /// The spacing is one photograph pixel of the finest view, the seen view
    /// with the smallest `pixel_size`, but the patch's side is never fewer
    /// than `min_samples_per_patch` samples nor more than
    /// `max_samples_per_patch`. The crop's half-width in the world is the
    /// larger of `crop_half_extents` half-extents and `min_crop_half_width_px`
    /// pixels of the finest view, rounded up to whole samples. Only where that
    /// would make the crop more than `max_crop_samples` a side is the grid
    /// coarsened, keeping the world half-width.
    pub(crate) fn new(
        point: Point3<f64>,
        normal: Vector3<f64>,
        axes: [Vector3<f64>; 2],
        half_extent: f64,
        views: &[ProjectedImage<'_>],
        options: &PairwiseFlowNormalOptions,
    ) -> (Option<CropGrid>, Vec<ViewSight>) {
        let nothing = |views: &[ProjectedImage<'_>]| (None, vec![ViewSight::NoSight; views.len()]);
        let finite = point
            .coords
            .iter()
            .chain(normal.iter())
            .all(|c| c.is_finite())
            && half_extent.is_finite()
            && half_extent > 0.0;
        if !finite || normal.norm() == 0.0 {
            return nothing(views);
        }
        let normal = normal.normalize();
        let Some((x_axis, v_axis)) = crop_axes(normal, axes) else {
            return nothing(views);
        };
        let probe = OrientedPatch::new(point, x_axis, v_axis, [half_extent, half_extent]);
        let sights: Vec<ViewSight> = views
            .iter()
            .map(|view| sight(&probe, normal, view, options.max_view_angle_deg))
            .collect();
        let finest = sights
            .iter()
            .filter_map(|s| match s {
                ViewSight::Seen { pixel_size, .. } => Some(*pixel_size),
                _ => None,
            })
            .fold(f64::INFINITY, f64::min);
        if !finest.is_finite() {
            return (None, sights);
        }
        let by_rule = (2.0 * half_extent / finest).clamp(
            f64::from(options.min_samples_per_patch),
            f64::from(options.max_samples_per_patch),
        );
        let mut spacing = 2.0 * half_extent / by_rule;
        let world_half_width =
            (options.crop_half_extents * half_extent).max(options.min_crop_half_width_px * finest);
        let mut half_width = (world_half_width / spacing).ceil().max(1.0);
        let cap = f64::from(options.max_crop_samples / 2).max(1.0);
        if half_width > cap {
            half_width = cap;
            spacing = world_half_width / half_width;
        }
        let grid = CropGrid {
            plane: CommonPlane {
                point: point.coords,
                normal,
                x_axis,
                // The rows step along `-v`, as `WarpMap::from_patch` lays them.
                y_axis: -v_axis,
                spacing,
            },
            samples_per_patch: 2.0 * half_extent / spacing,
            half_width: half_width as u32,
        };
        (Some(grid), sights)
    }

    /// The crops' side, grid samples.
    pub(crate) fn side(&self) -> u32 {
        2 * self.half_width
    }

    /// The crop as an oriented patch: centred on the point, on the round's
    /// plane, half `half_width` samples a side.
    fn crop_patch(&self) -> OrientedPatch {
        let plane = &self.plane;
        let half = f64::from(self.half_width) * plane.spacing;
        OrientedPatch::new(
            Point3::from(plane.point),
            plane.x_axis,
            -plane.y_axis,
            [half, half],
        )
    }
}

/// The crops' two in-plane axes `(u, v)` with `u × v = normal`, from the
/// patch's axes as [`CropGrid::new`] says.
fn crop_axes(
    normal: Vector3<f64>,
    axes: [Vector3<f64>; 2],
) -> Option<(Vector3<f64>, Vector3<f64>)> {
    let off = |a: Vector3<f64>| a - normal * normal.dot(&a);
    let [u, v] = axes;
    let (pu, pv) = (off(u), off(v));
    if pu.norm() >= MIN_AXIS_SINE * u.norm() && pu.norm() > 0.0 {
        let x = pu.normalize();
        Some((x, normal.cross(&x)))
    } else if pv.norm() >= MIN_AXIS_SINE * v.norm() && pv.norm() > 0.0 {
        let y = pv.normalize();
        Some((y.cross(&normal), y))
    } else {
        None
    }
}

/// How `view` sees the plane `probe` lies in, at its centre.
fn sight(
    probe: &OrientedPatch,
    normal: Vector3<f64>,
    view: &ProjectedImage<'_>,
    max_view_angle_deg: f64,
) -> ViewSight {
    let centre = view.cam_from_world.inverse_translation_origin().coords;
    let line = centre - probe.center.coords;
    let length = line.norm();
    if !(length.is_finite() && length > 0.0) {
        return ViewSight::NoSight;
    }
    let angle_deg = line
        .normalize()
        .dot(&normal)
        .clamp(-1.0, 1.0)
        .acos()
        .to_degrees();
    if angle_deg > max_view_angle_deg {
        return ViewSight::TooOblique { angle_deg };
    }
    let Some(jacobian) = patch_grid_jacobian(
        probe,
        view.camera,
        view.cam_from_world,
        PROBE_RESOLUTION as usize,
    ) else {
        return ViewSight::NoSight;
    };
    let [_, minor] = singular_values_2x2(jacobian);
    // Photograph px per probe sample, and a probe sample is this much world.
    let probe_step = 2.0 * probe.half_extent[0] / f64::from(PROBE_RESOLUTION);
    let pixel_size = probe_step / minor;
    if !(pixel_size.is_finite() && pixel_size > 0.0) {
        return ViewSight::NoSight;
    }
    ViewSight::Seen {
        angle_deg,
        pixel_size,
    }
}

/// One view's crop on a round's grid.
pub(crate) struct Crop {
    /// The crop's grey values, as the flow reads them, `side × side`; a
    /// sample off the photograph holds the mean of those on it.
    pub(crate) image: GrayImage,
    /// Whether each sample fell on the photograph.
    pub(crate) valid: Vec<bool>,
    /// The camera's centre, in world space.
    pub(crate) centre: Vector3<f64>,
    /// Degrees between the view's line of sight and the plane's normal.
    pub(crate) angle_deg: f64,
    /// How many grid samples one of the view's own pixels spans at the point,
    /// along its coarser axis on the plane: over 1 the view was upsampled,
    /// under 1 it was read through the photograph's pyramid.
    pub(crate) samples_per_pixel: f64,
    /// The share of the crop's samples that fell on the photograph.
    pub(crate) coverage: f64,
}

/// Render the crop of every view `sights` says is seen, on `grid`, through
/// its camera model; `None` for the others. `sights` is the second half of
/// [`CropGrid::new`]'s answer for the same `views`.
///
/// The views are rendered in parallel; each crop depends on its own view
/// alone, and they are returned in the views' order.
pub(crate) fn render_crops(
    grid: &CropGrid,
    views: &[ProjectedImage<'_>],
    sights: &[ViewSight],
) -> Vec<Option<Crop>> {
    views
        .par_iter()
        .zip(sights)
        .map(|(view, sight)| match *sight {
            ViewSight::Seen {
                angle_deg,
                pixel_size,
            } => Some(render_crop(grid, view, angle_deg, pixel_size)),
            _ => None,
        })
        .collect()
}

fn render_crop(
    grid: &CropGrid,
    view: &ProjectedImage<'_>,
    angle_deg: f64,
    pixel_size: f64,
) -> Crop {
    let side = grid.side();
    let patch = grid.crop_patch();
    let sampler =
        SamplerChoice::per_view().for_patch_frame(&patch, view.camera, view.cam_from_world, side);
    let mut map = WarpMap::from_patch(&patch, view.camera, view.cam_from_world, side);
    let tile = render_tile(view.pyramid, &mut map, sampler);
    let n = side as usize * side as usize;
    let channels = tile.channels() as usize;
    let data = tile.data();
    // The flow's grey reads interleaved RGB; a grey photograph gives its one
    // channel to all three, and an alpha channel carries no texture.
    let mut rgb = Vec::with_capacity(3 * n);
    for k in 0..n {
        let at = k * channels;
        if channels >= 3 {
            rgb.extend_from_slice(&data[at..at + 3]);
        } else {
            rgb.extend_from_slice(&[data[at]; 3]);
        }
    }
    let mut image = matcher_gray_from_rgb(side, side, &rgb);
    let valid: Vec<bool> = (0..side)
        .flat_map(|row| (0..side).map(move |col| (col, row)))
        .map(|(col, row)| map.is_valid(col, row))
        .collect();
    let on = valid.iter().filter(|&&v| v).count();
    // Off the photograph the render holds nothing; filling it with the mean
    // keeps the flow's pyramid from blurring an edge into the crop.
    if on < n {
        // Each row summed and then the row sums, in `f32`: over a crop of at
        // most `MAX_CROP_SAMPLES` a side the rounding is bounded by about
        // `(rows + columns) · 2⁻²⁴` of the sum, `5 · 10⁻⁴`, and the fill only
        // has to be near the crop's grey.
        let mean = if on == 0 {
            0.5
        } else {
            let row = side as usize;
            let sum: f32 = image
                .data()
                .chunks_exact(row)
                .zip(valid.chunks_exact(row))
                .map(|(grey, ok)| {
                    grey.iter()
                        .zip(ok)
                        .filter(|(_, &v)| v)
                        .map(|(&g, _)| g)
                        .sum::<f32>()
                })
                .sum();
            sum / on as f32
        };
        for (g, &v) in image.data_mut().iter_mut().zip(&valid) {
            if !v {
                *g = mean;
            }
        }
    }
    Crop {
        image,
        valid,
        centre: view.cam_from_world.inverse_translation_origin().coords,
        angle_deg,
        samples_per_pixel: pixel_size / grid.plane.spacing,
        coverage: on as f64 / n as f64,
    }
}

/// A crop's Gaussian pyramid, the flow's own [`ImagePyramid`] (a blur by its
/// 6-tap σ = 1 kernel and a 2× downsample per level), with which of each
/// level's samples lie wholly on the photograph.
pub(crate) struct CropPyramid {
    pyramid: ImagePyramid,
    valid: Vec<Vec<bool>>,
}

impl CropPyramid {
    /// The pyramid of `crop` down to level `deepest`.
    pub(crate) fn new(crop: &Crop, deepest: u32) -> Self {
        let pyramid = ImagePyramid::build(&crop.image, deepest + 1);
        let mut valid = vec![crop.valid.clone()];
        for level in 1..=deepest as usize {
            let (w, h) = (
                pyramid.level(level - 1).width() as usize,
                pyramid.level(level - 1).height() as usize,
            );
            let next = downsample_valid(&valid[level - 1], w, h);
            valid.push(next);
        }
        Self { pyramid, valid }
    }

    /// Level `level`'s image and which of its samples are valid.
    pub(crate) fn level(&self, level: u32) -> (&GrayImage, &[bool]) {
        (
            self.pyramid.level(level as usize),
            &self.valid[level as usize],
        )
    }
}

/// The valid mask one pyramid level down: a sample is valid where every
/// sample under its 6-tap kernel, clamped at the border as the blur clamps
/// it, is.
fn downsample_valid(valid: &[bool], width: usize, height: usize) -> Vec<bool> {
    let (out_w, out_h) = (width / 2, height / 2);
    let clamp = |i: isize, n: usize| i.clamp(0, n as isize - 1) as usize;
    let mut rows = vec![true; height * out_w];
    for row in 0..height {
        for oc in 0..out_w {
            rows[row * out_w + oc] =
                (-2..4).all(|k| valid[row * width + clamp(2 * oc as isize + k, width)]);
        }
    }
    let mut out = vec![true; out_h * out_w];
    for or in 0..out_h {
        for col in 0..out_w {
            out[or * out_w + col] =
                (-2..4).all(|k| rows[clamp(2 * or as isize + k, height) * out_w + col]);
        }
    }
    out
}

/// The pyramid level a pair of crops is read at, and how much the coarser
/// view is upsampled there.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct PairLevel {
    /// `ℓ = floor(log2 k)`, `k` the coarser view's samples per pixel; 0 where
    /// neither view is coarser than the grid.
    pub(crate) level: u32,
    /// `m = k / 2^ℓ`, between 1 and 2; 1 where neither view is coarser than
    /// the grid.
    pub(crate) upsampling: f64,
}

impl PairLevel {
    /// The level for two crops of `samples_per_pixel` `k_i` and `k_j`.
    pub(crate) fn new(k_i: f64, k_j: f64) -> Self {
        let k = k_i.max(k_j);
        if !(k.is_finite() && k > 1.0) {
            return Self {
                level: 0,
                upsampling: 1.0,
            };
        }
        let level = k.log2().floor() as u32;
        Self {
            level,
            upsampling: k / f64::from(level).exp2(),
        }
    }

    /// Grid samples per sample at this level, `2^ℓ`.
    pub(crate) fn scale(&self) -> f64 {
        f64::from(self.level).exp2()
    }
}

#[cfg(test)]
mod tests;
