// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Cluster-patch refinement: turn SIFT feature clusters into patch clusters.
//!
//! See `specs/core/patch/cluster-patch-refinement.md` (implementation) and
//! `specs/core/patch/cluster-patches.md` (design). Given per-image pyramids, SIFT
//! feature geometry, and CSR clusters, [`refine_cluster_patches`] first
//! gates each member on the ZNCC self-similarity radius of its own patch
//! (`specs/core/patch/zncc-self-similarity-radius.md`; members above
//! `max_member_zncc_self_similarity_radius` are excluded up front), then picks a
//! reference member per cluster (largest SIFT scale, deterministic
//! tie-breaks), builds a Gaussian-windowed z-normalized template around the
//! reference detection, refines an affine warp to every other member by a
//! shift → similarity → affine Nelder-Mead cascade on the windowed ZNCC
//! (seeded from the SIFT affine shapes, `M₀ = A_mem · A_ref⁻¹`), vets by
//! achieved ZNCC and translation drift, dedupes to one kept member per image,
//! and emits member-parallel arrays that map 1:1 onto the `.matches`
//! `cluster_patches/` section.
//!
//! The refinement's working unknown is the relative warp `W`, but what the
//! member arrays STORE is the absolute affine shape `S = W · A_ref` — the map
//! from the detector's canonical unit frame onto that member's image pixels,
//! `.matches` format version 5. Every member is then self-contained for
//! extent (`S`'s column norms) as well as position, and `W = S · S_ref⁻¹`
//! recovers the relative warp through the cluster's `S_ref | x_ref` reference
//! row.
//!
//! Sampling reuses the house conventions: the `bilinear_geometry` pixel-center
//! convention (`x − 0.5`), the shared window `Support`, and the
//! `weighted_moments_pub` / `znorm_write` z-normalization kernels. Pyramid
//! levels follow the standard mip rule — per sampled image, level
//! `ℓ = clamp(⌊log₂ s_min⌋, 0, L−1)` where `s_min` is the smaller singular
//! value of the support map's linear part (sample spacing in source px), with
//! the map divided by `2^ℓ` before sampling.

mod consistency;
mod kernels;
mod params;
mod piecewise;
pub mod prof;

#[cfg(test)]
mod tests;

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

use ndarray::{Array2, Array3};
use rayon::prelude::*;

use crate::camera::image::ImageU8Pyramid;
use crate::patch::normal_refine::{
    build_support, weighted_moments_pub, znorm_write, PartZncc, Support, FLAT_NORM_SQ_EPS,
};
use crate::patch::self_similarity::{zncc_self_similarity_radius, PatchTile, SelfSimilarityParams};
use crate::patch::view_selection::AffineCoreMap;

use kernels::{
    eval_zncc, eval_zncc_parts, grid_bbox, nelder_mead, SupportTables, TemplateKernel, TileCache,
};

pub use consistency::warp_consistency_residuals;
pub use params::{
    ClusterRefineParams, ClusterRefineResult, FeatureGeometry, MemberStatus, REFERENCE_UNREFINABLE,
};
pub use piecewise::{
    member_cell_data, CellRefinement, CellStatus, LoopStop, PiecewiseParams, ACCEPT_ZNCC_TOLERANCE,
    DEFAULT_MIN_CELL_CURVATURE, DEFAULT_MIN_CELL_ZNCC,
};

/// A member's SIFT affine shape is usable when `|det A|` clears this floor.
const MIN_ABS_DET: f64 = 1e-9;

/// Log-scale clamp of the similarity stage (`σ ∈ [−1.5, 1.5]`, the
/// prototype's bound).
const SIGMA_CLAMP: f64 = 1.5;

// ── Small 2×2 matrix helpers ────────────────────────────────────────────────

type Mat2 = [[f64; 2]; 2];

fn mul2(a: &Mat2, b: &Mat2) -> Mat2 {
    [
        [
            a[0][0] * b[0][0] + a[0][1] * b[1][0],
            a[0][0] * b[0][1] + a[0][1] * b[1][1],
        ],
        [
            a[1][0] * b[0][0] + a[1][1] * b[1][0],
            a[1][0] * b[0][1] + a[1][1] * b[1][1],
        ],
    ]
}

fn det2(a: &Mat2) -> f64 {
    a[0][0] * a[1][1] - a[0][1] * a[1][0]
}

/// Inverse of a 2×2 with a non-degenerate determinant (callers gate on
/// [`MIN_ABS_DET`]).
fn inv2(a: &Mat2) -> Mat2 {
    let inv_det = 1.0 / det2(a);
    [
        [a[1][1] * inv_det, -a[0][1] * inv_det],
        [-a[1][0] * inv_det, a[0][0] * inv_det],
    ]
}

// ── Warp parameterization (the prototype's `PairWarp`) ─────────────────────

/// Cascade stage: which parameters `θ` carries.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Stage {
    /// `θ = (tx, ty)`.
    Shift,
    /// `θ = (tx, ty, σ, φ)`; `D = e^σ R(φ) − I` with `σ` clamped to
    /// ±[`SIGMA_CLAMP`].
    Sim,
    /// `θ = (tx, ty, D00, D01, D10, D11)`.
    Affine,
}

impl Stage {
    fn dims(self) -> usize {
        match self {
            Stage::Shift => 2,
            Stage::Sim => 4,
            Stage::Affine => 6,
        }
    }

    /// Simplex seed scales: 0.5 px for translations, 0.05 for σ/φ/D entries.
    fn scales(self) -> &'static [f64] {
        match self {
            Stage::Shift => &[0.5, 0.5],
            Stage::Sim => &[0.5, 0.5, 0.05, 0.05],
            Stage::Affine => &[0.5, 0.5, 0.05, 0.05, 0.05, 0.05],
        }
    }
}

/// `D = e^σ R(φ) − I` (σ clamped).
fn sim_d(sigma: f64, phi: f64) -> Mat2 {
    let s = sigma.clamp(-SIGMA_CLAMP, SIGMA_CLAMP).exp();
    let (sin, cos) = phi.sin_cos();
    [[s * cos - 1.0, -s * sin], [s * sin, s * cos - 1.0]]
}

/// Split a stage vector into `(t, D)`.
fn unpack(theta: &[f64], stage: Stage) -> ([f64; 2], Mat2) {
    let t = [theta[0], theta[1]];
    let d = match stage {
        Stage::Shift => [[0.0; 2]; 2],
        Stage::Sim => sim_d(theta[2], theta[3]),
        Stage::Affine => [[theta[2], theta[3]], [theta[4], theta[5]]],
    };
    (t, d)
}

/// Re-express a converged stage vector as the next stage's start (σ, φ
/// promoted to `D` entries; new degrees of freedom start at 0).
fn promote(theta: &[f64], from: Stage, to: Stage) -> Vec<f64> {
    let mut out = vec![0.0; to.dims()];
    out[0] = theta[0];
    out[1] = theta[1];
    match from {
        Stage::Sim => {
            let d = sim_d(theta[2], theta[3]);
            out[2] = d[0][0];
            out[3] = d[0][1];
            out[4] = d[1][0];
            out[5] = d[1][1];
        }
        Stage::Shift | Stage::Affine => {
            let upto = theta.len().min(out.len());
            out[2..upto].copy_from_slice(&theta[2..upto]);
        }
    }
    out
}

// ── Pyramid-level selection (the mip rule) ─────────────────────────────────

/// Level `ℓ = clamp(⌊log₂ s_min⌋, 0, L−1)` from the smaller singular value of
/// the map's linear part (closed form). A non-finite or shrinking map stays
/// at level 0 (the degenerate warp is rejected by the sampler's frame test).
fn level_for_map(map: &AffineCoreMap, num_levels: usize) -> usize {
    let a = &map.a;
    let e = (a[0] + a[4]) * 0.5;
    let f = (a[0] - a[4]) * 0.5;
    let g = (a[3] + a[1]) * 0.5;
    let h = (a[3] - a[1]) * 0.5;
    let q = (e * e + h * h).sqrt();
    let r = (f * f + g * g).sqrt();
    let s_min = (q - r).abs();
    if !s_min.is_finite() || s_min < 2.0 {
        return 0;
    }
    (s_min.log2().floor() as usize).min(num_levels.saturating_sub(1))
}

/// Divide the map by `2^level` (level coordinates are full-resolution
/// coordinates over `2^ℓ` under the shared pixel-center convention).
fn map_at_level(map: &AffineCoreMap, level: usize) -> AffineCoreMap {
    if level == 0 {
        return AffineCoreMap::from_coeffs(map.a);
    }
    let inv = 1.0 / (1u64 << level) as f64;
    let mut a = map.a;
    for c in a.iter_mut() {
        *c *= inv;
    }
    AffineCoreMap::from_coeffs(a)
}

// ── Per-cluster machinery ───────────────────────────────────────────────────

/// One usable member's decoded SIFT geometry.
#[derive(Clone)]
struct MemberGeo {
    k_global: u32,
    image: usize,
    pos: [f64; 2],
    a: Mat2,
    /// `√|det A|`, the reference-selection key.
    scale: f64,
}

/// Per-member scatter payload.
#[derive(Clone)]
struct MemberOutcome {
    status: MemberStatus,
    affine: [[f64; 3]; 2],
    zncc: f32,
    /// The middle ZNCC at the same final map (see [`kernels::eval_zncc_parts`]).
    zncc_middle: f32,
    /// The ZNCC grid at the same final map.
    zncc_grid: [[f32; 3]; 3],
    shift: f32,
    /// The piecewise refinement's cells, for a kept member when the stage
    /// runs.
    cells: Option<CellRefinement>,
}

impl Default for MemberOutcome {
    fn default() -> Self {
        MemberOutcome {
            status: MemberStatus::NotEvaluated,
            affine: [[0.0; 3]; 2],
            zncc: f32::NAN,
            zncc_middle: f32::NAN,
            zncc_grid: [[f32::NAN; 3]; 3],
            shift: f32::NAN,
            cells: None,
        }
    }
}

impl MemberOutcome {
    /// The part readings in the outcome's own `f32`.
    fn set_parts(&mut self, parts: PartZncc) {
        self.zncc_middle = parts.middle as f32;
        self.zncc_grid = parts.grid.map(|row| row.map(|z| z as f32));
    }
}

struct ClusterOutcome {
    reference: u32,
    members: Vec<MemberOutcome>,
}

/// The affine grid→source map of the anchored warp
/// `W(u) = pos_mem + t + (I + D)·M₀·A_ref·u` over grid indices, full-res
/// coordinates. For the reference template itself pass `t = 0`, `D = 0`,
/// `M₀ = I`, `pos_mem = pos_ref` — the map collapses to
/// `pos_ref + A_ref·u`.
fn warp_map(pos: [f64; 2], t: [f64; 2], b: &Mat2, step: f64, off: f64) -> AffineCoreMap {
    AffineCoreMap::from_coeffs([
        b[0][0] * step,
        b[0][1] * step,
        pos[0] + t[0] + (b[0][0] + b[0][1]) * off,
        b[1][0] * step,
        b[1][1] * step,
        pos[1] + t[1] + (b[1][0] + b[1][1]) * off,
    ])
}

/// One member's own tile on the template grid: the `R×R×C` interleaved `f32`
/// samples at the member's own position and affine shape, the grid the
/// reference's template is cut from.
///
/// `position` is the member's keypoint in source-image pixels and
/// `affine_shape` its absolute affine shape `S` (the map from the detector's
/// canonical unit frame onto that image's pixels) -- a seed's, or the
/// refinement's answer for it. The grid, the mip rule and the border clamp are
/// the kernel's own, so a caller that wants the reference's template to draw
/// gets the tile the cascade registers against. The member gate reads the
/// same grid ([`member_zncc_self_similarity_radius`]).
///
/// `None` for a degenerate shape, a non-finite coordinate, or a pyramid whose
/// selected level is too small to bilinear-sample.
pub fn sample_member_grid(
    pyramid: &ImageU8Pyramid,
    position: [f64; 2],
    affine_shape: [[f64; 2]; 2],
    params: &ClusterRefineParams,
) -> Option<Vec<f32>> {
    let det = det2(&affine_shape);
    if !det.is_finite() || det.abs() < MIN_ABS_DET {
        return None;
    }
    let resolution = params.resolution.max(2);
    let step = 2.0 * params.radius / resolution as f64;
    let off = 0.5 * step - params.radius;
    sample_patch_grid(pyramid, position, &affine_shape, resolution, step, off)
}

/// The ZNCC self-similarity radius of one member's own `R×R` grid
/// ([`sample_member_grid`]) at `position` and `affine_shape`, in template-grid
/// px, read the overlap way with the default [`SelfSimilarityParams`]: at each
/// shift only the samples both windows hold are correlated, so the reading
/// depends on no pixel outside the grid. This is the number
/// [`ClusterRefineParams::max_member_zncc_self_similarity_radius`] judges.
/// Only the leading three channels are read, as a patch tile's colour. `None`
/// where the grid cannot be sampled.
pub fn member_zncc_self_similarity_radius(
    pyramid: &ImageU8Pyramid,
    position: [f64; 2],
    affine_shape: [[f64; 2]; 2],
    params: &ClusterRefineParams,
) -> Option<f64> {
    let samples = sample_member_grid(pyramid, position, affine_shape, params)?;
    Some(grid_self_similarity_radius(
        &samples,
        params.resolution.max(2) as usize,
    ))
}

/// The overlap reading's ZNCC self-similarity radius of an interleaved
/// `R×R×C` grid from [`sample_member_grid`]; `NaN` for a grid with no
/// channels.
fn grid_self_similarity_radius(samples: &[f32], resolution: usize) -> f64 {
    let channels = samples.len() / (resolution * resolution);
    if channels == 0 {
        return f64::NAN;
    }
    let (planes, colour) =
        PatchTile::planes_from_interleaved(samples, resolution, resolution, channels);
    let tile = PatchTile {
        values: &planes,
        channels: colour,
        width: resolution,
        height: resolution,
    };
    zncc_self_similarity_radius(
        &tile,
        None,
        [0, 0, resolution, resolution],
        &SelfSimilarityParams::default(),
    )
    .radius
}

/// Sample a member's own full `R×R` grid at its SIFT geometry (identity
/// warp, mip-selected level, bit-exact `bilinear_geometry` convention) into
/// an interleaved `R×R×C` f32 patch. Unlike [`build_template`], every grid
/// pixel is sampled, not just the windowed support, and samples outside the
/// frame clamp to the nearest valid pixel (border replicate) so a border
/// member is read on its visible content instead of skipping the member
/// gate. `None` only for non-finite coordinates (degenerate geometry) or a
/// level too small to bilinear-sample.
fn sample_patch_grid(
    pyramid: &ImageU8Pyramid,
    pos: [f64; 2],
    a: &Mat2,
    resolution: u32,
    step: f64,
    off: f64,
) -> Option<Vec<f32>> {
    sample_grid_window(
        pyramid,
        pos,
        a,
        step,
        off,
        0,
        resolution as usize,
        GridEdge::Clamp,
    )
}

/// What [`sample_grid_window`] does with a sample whose taps leave the image.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum GridEdge {
    /// Clamp to the nearest valid pixel (border replicate), and refuse the
    /// whole window on a non-finite coordinate.
    Clamp,
    /// Write `NaN` in every channel of that sample, non-finite coordinates
    /// included.
    Missing,
}

/// Sample a `size × size` window of the template grid's positions, starting at
/// grid index `first` on both axes, at position `pos` and shape `a`, into an
/// interleaved `size × size × C` f32 patch. The map, the pyramid level and the
/// bilinear convention are those of the `R×R` grid whose spacing is `step`
/// and whose first sample is at `off` (keypoint-frame units), so a
/// window with `first = 0` and `size = R` is that grid, and a window reaching
/// past it continues the same map. `None` for a level too small to
/// bilinear-sample, or, under [`GridEdge::Clamp`], a non-finite coordinate.
#[allow(clippy::too_many_arguments)]
fn sample_grid_window(
    pyramid: &ImageU8Pyramid,
    pos: [f64; 2],
    a: &Mat2,
    step: f64,
    off: f64,
    first: i64,
    size: usize,
    edge: GridEdge,
) -> Option<Vec<f32>> {
    let map = warp_map(pos, [0.0, 0.0], a, step, off);
    let level = level_for_map(&map, pyramid.num_levels());
    let lmap = map_at_level(&map, level);
    let img = pyramid.level(level);
    let ch = img.channels() as usize;
    let stride = img.width() as usize * ch;
    let data = img.data();
    let (w_img, h_img) = (img.width() as i64, img.height() as i64);
    if w_img < 2 || h_img < 2 {
        return None;
    }
    let (max_gx, max_gy) = ((w_img - 1) as f64, (h_img - 1) as f64);
    let mut out = vec![0f32; size * size * ch];
    for px in 0..size * size {
        let col = (first + (px % size) as i64) as f64;
        let row = (first + (px / size) as i64) as f64;
        let x = lmap.a[0] * col + lmap.a[1] * row + lmap.a[2];
        let y = lmap.a[3] * col + lmap.a[4] * row + lmap.a[5];
        // `bilinear_geometry`'s pixel-center convention.
        let gx = x - 0.5;
        let gy = y - 0.5;
        let (ix, iy, fx, fy) = match edge {
            GridEdge::Clamp => {
                if !gx.is_finite() || !gy.is_finite() {
                    return None;
                }
                // Nearest-valid-pixel clamp (border replicate): the coordinate
                // clamps to the outermost pixel center and the tap base to the
                // last valid 2x2 cell, so fx/fy saturate and the blend reads
                // the edge pixel.
                let gx = gx.clamp(0.0, max_gx);
                let gy = gy.clamp(0.0, max_gy);
                let ix = (gx.floor() as i64).min(w_img - 2);
                let iy = (gy.floor() as i64).min(h_img - 2);
                (ix, iy, (gx - ix as f64) as f32, (gy - iy as f64) as f32)
            }
            GridEdge::Missing => {
                let in_frame = gx.is_finite()
                    && gy.is_finite()
                    && gx >= 0.0
                    && gy >= 0.0
                    && (gx.floor() as i64) + 1 < w_img
                    && (gy.floor() as i64) + 1 < h_img;
                if !in_frame {
                    out[px * ch..][..ch].fill(f32::NAN);
                    continue;
                }
                let (x0, y0) = (gx.floor(), gy.floor());
                (x0 as i64, y0 as i64, (gx - x0) as f32, (gy - y0) as f32)
            }
        };
        let base = iy as usize * stride + ix as usize * ch;
        for c in 0..ch {
            let v00 = data[base + c] as f32;
            let v10 = data[base + ch + c] as f32;
            let v01 = data[base + stride + c] as f32;
            let v11 = data[base + stride + ch + c] as f32;
            out[px * ch + c] = (1.0 - fx) * (1.0 - fy) * v00
                + fx * (1.0 - fy) * v10
                + (1.0 - fx) * fy * v01
                + fx * fy * v11;
        }
    }
    Some(out)
}

/// Build the reference's z-normalized template kernel: sample every image
/// channel over the support grid at the mip-selected level (bit-exact
/// `bilinear_geometry` convention, all-in-frame required), z-normalize each
/// channel with the sqrt-window fold, and drop flat channels. `None` when any
/// support sample leaves the frame or every channel is flat — the candidate
/// reference is unusable.
fn build_template(
    pyramid: &ImageU8Pyramid,
    geo: &MemberGeo,
    support: &Support,
    tables: &SupportTables,
    resolution: u32,
    step: f64,
    off: f64,
) -> Option<TemplateKernel> {
    let map = warp_map(geo.pos, [0.0, 0.0], &geo.a, step, off);
    let level = level_for_map(&map, pyramid.num_levels());
    let lmap = map_at_level(&map, level);
    let img = pyramid.level(level);
    let n = support.pixels.len();
    let ch = img.channels() as usize;
    let r = resolution as usize;
    let stride = img.width() as usize * ch;
    let data = img.data();
    let (w_img, h_img) = (img.width() as i64, img.height() as i64);

    // Raw support samples, planar per channel `[c·n + k]` (the ContextTile
    // layout), values un-rounded f32.
    let mut raw = vec![0f32; ch * n];
    for (k, &p) in support.pixels.iter().enumerate() {
        let col = (p % r) as f64;
        let row = (p / r) as f64;
        let x = lmap.a[0] * col + lmap.a[1] * row + lmap.a[2];
        let y = lmap.a[3] * col + lmap.a[4] * row + lmap.a[5];
        // `bilinear_geometry`'s pixel-center convention.
        let gx = x - 0.5;
        let gy = y - 0.5;
        if !gx.is_finite() || !gy.is_finite() {
            return None;
        }
        let x0 = gx.floor();
        let y0 = gy.floor();
        let (ix, iy) = (x0 as i64, y0 as i64);
        if ix < 0 || iy < 0 || ix + 1 >= w_img || iy + 1 >= h_img {
            return None;
        }
        let (fx, fy) = ((gx - x0) as f32, (gy - y0) as f32);
        let base = iy as usize * stride + ix as usize * ch;
        for c in 0..ch {
            let v00 = data[base + c] as f32;
            let v10 = data[base + ch + c] as f32;
            let v01 = data[base + stride + c] as f32;
            let v11 = data[base + stride + ch + c] as f32;
            raw[c * n + k] = (1.0 - fx) * (1.0 - fy) * v00
                + fx * (1.0 - fy) * v10
                + (1.0 - fx) * fy * v01
                + fx * fy * v11;
        }
    }

    // Z-normalize each channel over the windowed support (f64 moments →
    // `znorm_write`), dropping flat channels; fold the second `√w` into the
    // correlation kernel so `Σ kern·v` realizes the windowed inner product
    // against raw member samples.
    let mut kern = Vec::new();
    let mut kern_sums = Vec::new();
    let mut src_channels = Vec::new();
    let mut samples = Vec::new();
    let mut znormed = vec![0f32; n];
    for c in 0..ch {
        let col_vals = &raw[c * n..][..n];
        let (s1, s2) = weighted_moments_pub(col_vals, &support.weights);
        let mean = s1 / support.total_weight;
        let norm_sq = s2 - s1 * mean;
        if norm_sq < FLAT_NORM_SQ_EPS {
            continue;
        }
        let inv_norm = 1.0 / norm_sq.sqrt();
        znorm_write(
            col_vals,
            &support.sqrt_weights,
            mean as f32,
            inv_norm as f32,
            &mut znormed,
        );
        let base = kern.len();
        kern.resize(base + tables.n_padded, 0.0);
        let mut sum = 0f64;
        for k in 0..n {
            let kv = support.sqrt_weights[k] * znormed[k];
            kern[base + k] = kv;
            sum += kv as f64;
        }
        kern_sums.push(sum);
        src_channels.push(c);
        samples.extend_from_slice(col_vals);
    }
    if src_channels.is_empty() {
        return None;
    }

    // The same channels over the whole square, for the ZNCC grid. A pixel off
    // the image is `NaN`, which leaves its cell unread rather than refusing the
    // template: the square reaches past the support the refinement scores.
    let square = r * r;
    let mut square_samples = vec![f32::NAN; src_channels.len() * square];
    for p in 0..square {
        let x = lmap.a[0] * (p % r) as f64 + lmap.a[1] * (p / r) as f64 + lmap.a[2] - 0.5;
        let y = lmap.a[3] * (p % r) as f64 + lmap.a[4] * (p / r) as f64 + lmap.a[5] - 0.5;
        if !x.is_finite() || !y.is_finite() {
            continue;
        }
        let (x0, y0) = (x.floor(), y.floor());
        let (ix, iy) = (x0 as i64, y0 as i64);
        if ix < 0 || iy < 0 || ix + 1 >= w_img || iy + 1 >= h_img {
            continue;
        }
        let (fx, fy) = ((x - x0) as f32, (y - y0) as f32);
        let base = iy as usize * stride + ix as usize * ch;
        for (tc, &c) in src_channels.iter().enumerate() {
            let v00 = data[base + c] as f32;
            let v10 = data[base + ch + c] as f32;
            let v01 = data[base + stride + c] as f32;
            let v11 = data[base + stride + ch + c] as f32;
            square_samples[tc * square + p] = (1.0 - fx) * (1.0 - fy) * v00
                + fx * (1.0 - fy) * v10
                + (1.0 - fx) * fy * v01
                + fx * fy * v11;
        }
    }
    Some(TemplateKernel {
        channels: src_channels.len(),
        src_channels,
        kern,
        kern_sums,
        samples,
        square_samples,
    })
}

/// Refine one non-reference member: the shift → similarity → affine
/// Nelder-Mead cascade on the negated windowed ZNCC. Returns
/// `(zncc, parts, shift_px, absolute 2×3 affine)`, where `parts` is the same
/// final map read over the middle of the grid and over each cell of the ZNCC
/// grid ([`eval_zncc_parts`]) and the affine's leading 2×2 is the member's
/// absolute affine shape `S = W·S_ref`, last column its refined absolute
/// keypoint position; `None` when the seed support is out of frame (→
/// `NotEvaluated`).
#[allow(clippy::too_many_arguments)]
fn refine_member(
    pyramid: &ImageU8Pyramid,
    ref_geo: &MemberGeo,
    mem_geo: &MemberGeo,
    tmpl: &TemplateKernel,
    tables: &SupportTables,
    resolution: u32,
    step: f64,
    off: f64,
    params: &ClusterRefineParams,
) -> Option<(f64, PartZncc, f64, [[f64; 3]; 2])> {
    let a_ref_inv = inv2(&ref_geo.a);
    let m0 = mul2(&mem_geo.a, &a_ref_inv);
    let num_levels = pyramid.num_levels();
    let mut tiles = TileCache::default();

    let mut eval_raw = |t: [f64; 2], d: Mat2| -> Option<f64> {
        let id = [[1.0 + d[0][0], d[0][1]], [d[1][0], 1.0 + d[1][1]]];
        let b = mul2(&mul2(&id, &m0), &ref_geo.a);
        let map = warp_map(mem_geo.pos, t, &b, step, off);
        let level = level_for_map(&map, num_levels);
        let lmap = map_at_level(&map, level);
        let bbox = grid_bbox(&lmap, resolution);
        let tile = tiles.get_or_build(pyramid, level, bbox)?;
        prof::count(&prof::N_EVALS, 1);
        prof::EVAL.time(|| eval_zncc(&lmap, tile, tables, tmpl))
    };

    // Seed support out of frame → the member is not evaluated.
    eval_raw([0.0, 0.0], [[0.0; 2]; 2])?;
    prof::count(&prof::N_EVALS_SHIFT, 1);

    let mut theta = vec![0.0f64; 2];
    let mut prev = Stage::Shift;
    let mut best_val = 1.0f64;
    for stage in [Stage::Shift, Stage::Sim, Stage::Affine] {
        let th0 = if stage == Stage::Shift {
            theta.clone()
        } else {
            promote(&theta, prev, stage)
        };
        // The affine stage's result is stored; the shift/sim stages only seed
        // the next stage and get the looser intermediate tolerance.
        let tol = if stage == Stage::Affine {
            params.convergence
        } else {
            params.intermediate_convergence
        };
        let stage_evals = std::cell::Cell::new(0u64);
        let (th, val) = nelder_mead(
            |x| {
                if prof::enabled() {
                    stage_evals.set(stage_evals.get() + 1);
                }
                let (t, d) = unpack(x, stage);
                match eval_raw(t, d) {
                    // Any support sample out of frame scores worst (+1.0) so
                    // the simplex retreats — the all-in-frame rule.
                    Some(z) => -z,
                    None => 1.0,
                }
            },
            &th0,
            stage.scales(),
            params.max_iters,
            tol,
            params.stall_iters,
            params.stall_tol,
        );
        prof::count(
            match stage {
                Stage::Shift => &prof::N_EVALS_SHIFT,
                Stage::Sim => &prof::N_EVALS_SIM,
                Stage::Affine => &prof::N_EVALS_AFFINE,
            },
            stage_evals.get(),
        );
        theta = th;
        prev = stage;
        best_val = val;
    }

    let (t, d) = unpack(&theta, Stage::Affine);
    let zncc = -best_val;
    // The part readings, at the map the winning evaluation sampled: one more
    // pass over the tile that evaluation read.
    let parts = {
        let id = [[1.0 + d[0][0], d[0][1]], [d[1][0], 1.0 + d[1][1]]];
        let b = mul2(&mul2(&id, &m0), &ref_geo.a);
        let map = warp_map(mem_geo.pos, t, &b, step, off);
        let level = level_for_map(&map, num_levels);
        let lmap = map_at_level(&map, level);
        let bbox = grid_bbox(&lmap, resolution);
        tiles
            .get_or_build(pyramid, level, bbox)
            .map_or(PartZncc::NAN, |tile| {
                eval_zncc_parts(&lmap, tile, tables, tmpl)
            })
    };
    let shift = (t[0] * t[0] + t[1] * t[1]).sqrt();
    // Absolute affine shape: the refined warp `W = (I + D)·M₀` composed onto
    // the reference feature's detector shape, `S = W·S_ref` — literally the
    // `b` matrix the winning evaluation sampled with, so the stored shape IS
    // the shape that was measured. `S` maps the detector's canonical unit
    // frame onto this member's image pixels, so its column norms are the
    // member's own image-space extent with no `.sift` read. The stored last
    // column is the member's refined absolute keypoint position
    // `p = pos_mem + t`. The reference-relative warp stays recoverable as
    // `W = S·S_ref⁻¹` and then reads `x_mem = W·(x − x_ref) + p`, with
    // `S_ref | x_ref` the cluster's reference row.
    let id = [[1.0 + d[0][0], d[0][1]], [d[1][0], 1.0 + d[1][1]]];
    let s_abs = mul2(&mul2(&id, &m0), &ref_geo.a);
    let p = [mem_geo.pos[0] + t[0], mem_geo.pos[1] + t[1]];
    Some((
        zncc,
        parts,
        shift,
        [
            [s_abs[0][0], s_abs[0][1], p[0]],
            [s_abs[1][0], s_abs[1][1], p[1]],
        ],
    ))
}

/// Refine one cluster (the per-cluster algorithm of the spec).
#[allow(clippy::too_many_arguments)]
fn refine_cluster(
    k0: usize,
    k1: usize,
    pyramids: &[&ImageU8Pyramid],
    features: &[FeatureGeometry<'_>],
    member_images: &[u32],
    member_features: &[u32],
    params: &ClusterRefineParams,
    support: &Support,
    tables: &SupportTables,
    resolution: u32,
) -> ClusterOutcome {
    let size = k1 - k0;
    let mut members = vec![MemberOutcome::default(); size];
    let unrefinable = |members| ClusterOutcome {
        reference: REFERENCE_UNREFINABLE,
        members,
    };

    // 1. Validate members: feature index in bounds, |det A| ≥ MIN_ABS_DET.
    let mut geo: Vec<Option<MemberGeo>> = vec![None; size];
    for (j, slot) in geo.iter_mut().enumerate() {
        let k = k0 + j;
        let img = member_images[k] as usize;
        if img >= features.len() {
            continue;
        }
        let f = &features[img];
        let fi = member_features[k] as usize;
        if fi >= f.positions_xy.nrows() || fi >= f.affine_shapes.shape()[0] {
            continue;
        }
        let a = [
            [
                f.affine_shapes[[fi, 0, 0]] as f64,
                f.affine_shapes[[fi, 0, 1]] as f64,
            ],
            [
                f.affine_shapes[[fi, 1, 0]] as f64,
                f.affine_shapes[[fi, 1, 1]] as f64,
            ],
        ];
        let det = det2(&a);
        if det.abs() < MIN_ABS_DET || !det.is_finite() {
            continue;
        }
        *slot = Some(MemberGeo {
            k_global: k as u32,
            image: img,
            pos: [
                f.positions_xy[[fi, 0]] as f64,
                f.positions_xy[[fi, 1]] as f64,
            ],
            a,
            scale: det.abs().sqrt(),
        });
    }
    if prof::enabled() {
        prof::count(
            &prof::N_MEMBERS,
            geo.iter().filter(|s| s.is_some()).count() as u64,
        );
    }
    let step = 2.0 * params.radius / resolution as f64;
    let off = 0.5 * step - params.radius;

    // 1b. Member self-similarity gate: read each usable member's own patch
    // and exclude members that match themselves further than the bar away —
    // before reference selection, so a patch that pins no position can
    // neither anchor nor join the cluster. Border patches sample with a
    // nearest-valid-pixel clamp, so they are read on their visible content;
    // only degenerate geometry (non-finite coordinates, or a sub-2px pyramid
    // level) skips the gate.
    if params.member_self_similarity_gate_is_on() {
        for (j, slot) in geo.iter_mut().enumerate() {
            let Some(g) = slot.as_ref() else {
                continue;
            };
            let Some(grid) = prof::GATE_SAMPLE
                .time(|| sample_member_grid(pyramids[g.image], g.pos, g.a, params))
            else {
                continue;
            };
            prof::count(&prof::N_GATED, 1);
            let radius =
                prof::GATE_SCORE.time(|| grid_self_similarity_radius(&grid, resolution as usize));
            if !params.admits_member_zncc_self_similarity_radius(radius) {
                prof::count(&prof::N_GATE_REJECTED, 1);
                members[j].status = MemberStatus::RejectedUnlocalizable;
                *slot = None;
            }
        }
    }

    let usable: Vec<usize> = (0..size).filter(|&j| geo[j].is_some()).collect();
    if usable.len() < 2 {
        return unrefinable(members);
    }

    // 2–3. Reference selection (largest scale, ties to the lowest global
    // member index) with template-usability fallback to the next candidate.
    let mut cands = usable;
    cands.sort_by(|&i, &j| {
        let (gi, gj) = (geo[i].as_ref().unwrap(), geo[j].as_ref().unwrap());
        gj.scale
            .partial_cmp(&gi.scale)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(gi.k_global.cmp(&gj.k_global))
    });
    let mut reference: Option<(usize, TemplateKernel)> = None;
    for &j in &cands {
        let g = geo[j].as_ref().unwrap();
        if let Some(t) = prof::TEMPLATE
            .time(|| build_template(pyramids[g.image], g, support, tables, resolution, step, off))
        {
            reference = Some((j, t));
            break;
        }
    }
    let Some((ref_j, tmpl)) = reference else {
        return unrefinable(members);
    };
    let ref_geo = geo[ref_j].clone().unwrap();
    // Reference row: `S_ref | x_ref` — the reference feature's own detector
    // affine shape and `.sift` keypoint position. Every member (the
    // reference included) then reads its absolute shape and position straight
    // from its own row, and the row is also what a consumer inverts to
    // recover a member's reference-relative warp.
    members[ref_j] = MemberOutcome {
        status: MemberStatus::Reference,
        affine: [
            [ref_geo.a[0][0], ref_geo.a[0][1], ref_geo.pos[0]],
            [ref_geo.a[1][0], ref_geo.a[1][1], ref_geo.pos[1]],
        ],
        zncc: 1.0,
        shift: 0.0,
        ..MemberOutcome::default()
    };
    // The template against itself: 1 wherever it is not flat.
    members[ref_j].set_parts(tables.parts.read(
        (&tmpl.samples, &tmpl.samples),
        (&tmpl.square_samples, &tmpl.square_samples),
        tmpl.channels,
        &tables.weights,
    ));

    // 5. Refine every other member (in member order) and vet.
    for j in 0..size {
        if j == ref_j {
            continue;
        }
        let Some(g) = geo[j].as_ref() else {
            // Invalid members stay NotEvaluated; gated members keep their
            // RejectedUnlocalizable status.
            continue;
        };
        if g.image == ref_geo.image {
            members[j].status = MemberStatus::DuplicateImage;
            continue;
        }
        prof::count(&prof::N_REFINES, 1);
        if let Some((zncc, parts, shift, affine)) = prof::REFINE.time(|| {
            refine_member(
                pyramids[g.image],
                &ref_geo,
                g,
                &tmpl,
                tables,
                resolution,
                step,
                off,
                params,
            )
        }) {
            let status = if zncc < params.min_zncc {
                MemberStatus::RejectedLowZncc
            } else if shift > params.max_shift_px {
                MemberStatus::RejectedShift
            } else {
                MemberStatus::Kept
            };
            members[j] = MemberOutcome {
                status,
                affine,
                zncc: zncc as f32,
                shift: shift as f32,
                ..MemberOutcome::default()
            };
            members[j].set_parts(parts);
        }
    }

    // 6. One kept member per image: highest ZNCC wins, ties to the lowest
    // member index (strict `>` keeps the earlier member).
    let mut best_per_image: HashMap<usize, usize> = HashMap::new();
    for j in 0..size {
        if members[j].status != MemberStatus::Kept {
            continue;
        }
        let img = geo[j].as_ref().unwrap().image;
        match best_per_image.entry(img) {
            std::collections::hash_map::Entry::Occupied(mut e) => {
                let cur = *e.get();
                if members[j].zncc > members[cur].zncc {
                    members[cur].status = MemberStatus::DuplicateImage;
                    e.insert(j);
                } else {
                    members[j].status = MemberStatus::DuplicateImage;
                }
            }
            std::collections::hash_map::Entry::Vacant(e) => {
                e.insert(j);
            }
        }
    }

    // 7. The piecewise refinement of every kept member, from its cascade
    // shape.
    if let Some(pp) = params.piecewise.as_ref() {
        let kept: Vec<usize> = (0..size)
            .filter(|&j| members[j].status == MemberStatus::Kept)
            .collect();
        if !kept.is_empty() {
            let layout = piecewise::CellLayout::new(resolution, pp.cell_shift_bound_px);
            let templates = piecewise::CellTemplates::new(&tmpl, &layout);
            for j in kept {
                let g = geo[j].as_ref().unwrap();
                prof::count(&prof::N_PIECEWISE, 1);
                prof::PIECEWISE.time(|| {
                    refine_kept_member_cells(
                        &mut members[j],
                        pyramids[g.image],
                        g.pos,
                        &tmpl,
                        &templates,
                        &layout,
                        tables,
                        resolution,
                        step,
                        off,
                        params,
                        pp,
                    )
                });
            }
        }
    }

    ClusterOutcome {
        reference: ref_geo.k_global,
        members,
    }
}

/// Run the piecewise refinement on one kept member and store its outcome:
/// the cells always, and, when the loop moved the shape, the new shape and
/// position with the whole-patch ZNCC, its parts and the shift from the seed
/// read again at the new map. The member's status stays the cascade's.
///
/// The member was kept on the cascade's readings, so the stage's shape must
/// pass the same gates to replace it: the ZNCC read again at the new map must
/// be at least [`ClusterRefineParams::min_zncc`] and at least the cascade's own
/// stored ZNCC, and the new position's shift from the seed at most
/// [`ClusterRefineParams::max_shift_px`]. When any fails, or when the new map's support leaves the frame so nothing
/// can be read, the member keeps its cascade shape, position and readings,
/// and every cell is stored as [`CellStatus::NotAttempted`] with the loop's
/// iteration count.
#[allow(clippy::too_many_arguments)]
fn refine_kept_member_cells(
    member: &mut MemberOutcome,
    pyramid: &ImageU8Pyramid,
    seed_pos: [f64; 2],
    tmpl: &TemplateKernel,
    templates: &piecewise::CellTemplates,
    layout: &piecewise::CellLayout,
    tables: &SupportTables,
    resolution: u32,
    step: f64,
    off: f64,
    params: &ClusterRefineParams,
    pp: &PiecewiseParams,
) {
    let s = [
        [member.affine[0][0], member.affine[0][1]],
        [member.affine[1][0], member.affine[1][1]],
    ];
    let p = [member.affine[0][2], member.affine[1][2]];
    // The cascade's own objective, which an update must not lower.
    let score = |shape: &Mat2, position: [f64; 2]| {
        member_zncc_at(
            pyramid, position, shape, tmpl, tables, resolution, step, off,
        )
    };
    let out = piecewise::refine_member_cells(
        pyramid,
        &tmpl.src_channels,
        templates,
        layout,
        s,
        p,
        step,
        off,
        pp,
        score,
    );
    if !out.updated {
        member.cells = Some(out.cells);
        return;
    }
    let (sh, ps) = (out.shape, out.position);
    let shift = (ps[0] - seed_pos[0]).hypot(ps[1] - seed_pos[1]);
    // The stored ZNCC must not fall below the cascade's: the loop's floor is
    // its own reading of the starting shape, which can differ from the
    // cascade's stored value in the last bits, so the stored value is the bar.
    let cascade_zncc = member.zncc;
    let reread = read_member_at(pyramid, ps, &sh, tmpl, tables, resolution, step, off).filter(
        |&(zncc, _)| {
            zncc >= params.min_zncc && zncc as f32 >= cascade_zncc && shift <= params.max_shift_px
        },
    );
    let Some((zncc, parts)) = reread else {
        member.cells = Some(CellRefinement::not_attempted(out.cells.iterations));
        return;
    };
    member.affine = [[sh[0][0], sh[0][1], ps[0]], [sh[1][0], sh[1][1], ps[1]]];
    member.zncc = zncc as f32;
    member.set_parts(parts);
    member.shift = shift as f32;
    member.cells = Some(out.cells);
}

/// The whole-patch windowed ZNCC and its part readings of a member at
/// absolute position `p` and shape `s`; `None` when the support leaves the
/// frame.
#[allow(clippy::too_many_arguments)]
fn read_member_at(
    pyramid: &ImageU8Pyramid,
    p: [f64; 2],
    s: &Mat2,
    tmpl: &TemplateKernel,
    tables: &SupportTables,
    resolution: u32,
    step: f64,
    off: f64,
) -> Option<(f64, PartZncc)> {
    let map = warp_map(p, [0.0, 0.0], s, step, off);
    let level = level_for_map(&map, pyramid.num_levels());
    let lmap = map_at_level(&map, level);
    let bbox = grid_bbox(&lmap, resolution);
    let mut tiles = TileCache::default();
    let tile = tiles.get_or_build(pyramid, level, bbox)?;
    let zncc = eval_zncc(&lmap, tile, tables, tmpl)?;
    Some((zncc, eval_zncc_parts(&lmap, tile, tables, tmpl)))
}

/// The whole-patch windowed ZNCC of a member at absolute position `p` and
/// shape `s`, the objective the cascade maximises; `None` when the support
/// leaves the frame.
#[allow(clippy::too_many_arguments)]
fn member_zncc_at(
    pyramid: &ImageU8Pyramid,
    p: [f64; 2],
    s: &Mat2,
    tmpl: &TemplateKernel,
    tables: &SupportTables,
    resolution: u32,
    step: f64,
    off: f64,
) -> Option<f64> {
    let map = warp_map(p, [0.0, 0.0], s, step, off);
    let level = level_for_map(&map, pyramid.num_levels());
    let lmap = map_at_level(&map, level);
    let bbox = grid_bbox(&lmap, resolution);
    let mut tiles = TileCache::default();
    let tile = tiles.get_or_build(pyramid, level, bbox)?;
    eval_zncc(&lmap, tile, tables, tmpl)
}

/// Refine every cluster into a patch cluster: per cluster, a reference member
/// plus a vetted absolute affine for every other member — leading 2×2 the
/// member's absolute affine shape `S = W·S_ref` (the refined reference→member
/// warp composed onto the reference feature's detector shape), last column the
/// member's refined absolute keypoint position `p`. Reference rows are
/// `S_ref | x_ref`, so `W = S·S_ref⁻¹` recovers the relative warp (see
/// [`warp_consistency_residuals`], which does exactly that).
/// Parallel over clusters (rayon); results are deterministic
/// under any thread schedule (each cluster's work is self-contained and the
/// scatter preserves cluster order). `progress` is bumped once per finished
/// cluster.
///
/// The kernel is pure: no I/O, no `.sift` reads — the caller supplies decoded
/// pyramids and feature geometry, one [`FeatureGeometry`] per pyramid.
///
/// # Panics
///
/// Panics when `features` is not parallel to `pyramids` or the CSR arrays are
/// inconsistent (`cluster_starts` must start at 0, be non-decreasing, and end
/// at the member count; the two member arrays must have equal length). A
/// member whose image index is out of range for `pyramids`, whose feature
/// index is out of range for its image, or whose affine shape is degenerate
/// is reported as [`MemberStatus::NotEvaluated`] rather than panicking.
pub fn refine_cluster_patches(
    pyramids: &[ImageU8Pyramid],
    features: &[FeatureGeometry<'_>],
    cluster_starts: &[u32],
    member_images: &[u32],
    member_features: &[u32],
    params: &ClusterRefineParams,
    progress: Option<&AtomicUsize>,
) -> ClusterRefineResult {
    let borrowed: Vec<&ImageU8Pyramid> = pyramids.iter().collect();
    refine_cluster_patches_borrowed(
        &borrowed,
        features,
        cluster_starts,
        member_images,
        member_features,
        params,
        progress,
    )
}

/// [`refine_cluster_patches`] over **borrowed** pyramids: the same kernel, for
/// a caller that holds one pyramid per image behind a reference rather than a
/// table of its own.
///
/// A pyramid is a decoded image and cloning one copies every pixel, so a caller
/// whose views are
/// [`ProjectedImage`](crate::patch::normal_refine::ProjectedImage)s -- the
/// bench's evaluation, which registers an in-memory cluster over the same views
/// the patch kernels read -- reaches the kernel through this entry instead of
/// building a table it would have to own. Everything else, the panics included,
/// is [`refine_cluster_patches`].
pub fn refine_cluster_patches_borrowed(
    pyramids: &[&ImageU8Pyramid],
    features: &[FeatureGeometry<'_>],
    cluster_starts: &[u32],
    member_images: &[u32],
    member_features: &[u32],
    params: &ClusterRefineParams,
    progress: Option<&AtomicUsize>,
) -> ClusterRefineResult {
    assert_eq!(
        features.len(),
        pyramids.len(),
        "features must be parallel to pyramids"
    );
    let m = member_images.len();
    assert_eq!(
        member_features.len(),
        m,
        "member_images and member_features must have equal length"
    );
    assert!(
        !cluster_starts.is_empty() && cluster_starts[0] == 0,
        "cluster_starts must begin at 0"
    );
    assert!(
        cluster_starts.windows(2).all(|w| w[0] <= w[1]),
        "cluster_starts must be non-decreasing"
    );
    assert_eq!(
        *cluster_starts.last().unwrap() as usize,
        m,
        "cluster_starts must end at the member count"
    );

    let c_count = cluster_starts.len() - 1;
    let resolution = params.resolution.max(2);
    let support = build_support(params.window, resolution);
    let tables = SupportTables::new(&support, resolution);

    if prof::enabled() {
        prof::reset();
    }
    let wall_start = std::time::Instant::now();
    let outcomes: Vec<ClusterOutcome> = (0..c_count)
        .into_par_iter()
        .map(|c| {
            let out = prof::TOTAL.time(|| {
                refine_cluster(
                    cluster_starts[c] as usize,
                    cluster_starts[c + 1] as usize,
                    pyramids,
                    features,
                    member_images,
                    member_features,
                    params,
                    &support,
                    &tables,
                    resolution,
                )
            });
            if let Some(p) = progress {
                p.fetch_add(1, Ordering::Relaxed);
            }
            out
        })
        .collect();
    if prof::enabled() {
        prof::report(c_count, wall_start.elapsed().as_secs_f64());
    }

    let mut result = ClusterRefineResult {
        reference_members: vec![REFERENCE_UNREFINABLE; c_count],
        member_status: vec![MemberStatus::NotEvaluated; m],
        member_positions: Array2::zeros((m, 2)),
        member_affine_shapes: Array3::zeros((m, 2, 2)),
        member_zncc: vec![f32::NAN; m],
        member_zncc_middle: vec![f32::NAN; m],
        member_zncc_grid: vec![[[f32::NAN; 3]; 3]; m],
        member_shift_px: vec![f32::NAN; m],
        cells: vec![None; m],
    };
    for (c, out) in outcomes.into_iter().enumerate() {
        result.reference_members[c] = out.reference;
        let k0 = cluster_starts[c] as usize;
        for (j, mo) in out.members.into_iter().enumerate() {
            let k = k0 + j;
            result.member_status[k] = mo.status;
            result.member_zncc[k] = mo.zncc;
            result.member_zncc_middle[k] = mo.zncc_middle;
            result.member_zncc_grid[k] = mo.zncc_grid;
            result.member_shift_px[k] = mo.shift;
            if mo.status == MemberStatus::Kept {
                result.cells[k] = mo.cells;
            }
            // The 2×3 the cascade carries splits into the two arrays the
            // format stores: leading 2×2 the absolute shape, last column the
            // absolute position.
            for (r, row) in mo.affine.iter().enumerate() {
                result.member_affine_shapes[[k, r, 0]] = row[0];
                result.member_affine_shapes[[k, r, 1]] = row[1];
                result.member_positions[[k, r]] = row[2];
            }
        }
    }
    result
}
