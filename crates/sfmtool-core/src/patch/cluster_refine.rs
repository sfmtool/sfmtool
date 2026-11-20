// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Cluster-patch refinement: turn SIFT feature clusters into patch clusters.
//!
//! See `specs/core/patch/cluster-patch-refinement.md` (implementation) and
//! `specs/core/patch/cluster-patches.md` (design). Given per-image photographs, SIFT
//! feature geometry, and CSR clusters, [`refine_cluster_patches`] first
//! gates each member on the ZNCC self-similarity radius of its own patch
//! (`specs/core/patch/zncc-self-similarity-radius.md`; members above
//! `max_member_zncc_self_similarity_radius` are excluded up front), then picks a
//! reference member per cluster (largest SIFT scale, deterministic
//! tie-breaks), builds a Gaussian-windowed z-normalized template around the
//! reference detection, refines an affine warp to every other member by a
//! shift → similarity → affine Nelder-Mead cascade on the windowed ZNCC
//! (seeded from the SIFT affine shapes, `M₀ = A_mem · A_ref⁻¹`), vets by
//! achieved ZNCC and translation drift, reads each surviving member's own
//! patch again at its refined shape (the whole patch's self-similarity radius
//! and how many of its nine cells read the largest), dedupes to one kept
//! member per image, and emits member-parallel arrays that map 1:1 onto the
//! `.matches` `cluster_patches/` section.
//!
//! The refinement's working unknown is the relative warp `W`, but what the
//! member arrays STORE is the absolute affine shape `S = W · A_ref` — the map
//! from the detector's canonical unit frame onto that member's image pixels,
//! `.matches` format version 5. Every member is then self-contained for
//! extent (`S`'s column norms) as well as position, and `W = S · S_ref⁻¹`
//! recovers the relative warp through the cluster's `S_ref | x_ref` reference
//! row.
//!
//! Sampling reuses the house conventions: the `bilinear_cell` pixel-center
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

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

use ndarray::{Array2, Array3};
use rayon::prelude::*;

use crate::camera::image::ImageU8Pyramid;
use crate::camera::remap::HeldLevel;
use crate::camera::sampler::{
    grid_cells, sample_grid, sample_source_grid, GridEdge, GridPoints, RegionPool, ViewRegions,
};
use crate::camera::{PhotographReadError, TiledPhotograph};
use crate::patch::cell_registration::{CellLayout, CellTemplates};
use crate::patch::normal_refine::{
    build_support, weighted_moments_pub, znorm_write, PartZncc, Support, FLAT_NORM_SQ_EPS,
};
use crate::patch::self_similarity::{
    zncc_self_similarity_parts, zncc_self_similarity_radius, InterleavedTile, SelfSimilarityParams,
    SelfSimilarityParts,
};
use crate::patch::view_selection::AffineCoreMap;

use kernels::{
    eval_zncc, eval_zncc_parts, grid_bbox, nelder_mead, SupportTables, TemplateKernel, TileCache,
};

pub use crate::patch::cell_registration::CellStatus;
pub use consistency::warp_consistency_residuals;
pub use params::{
    capped_cell_count, ClusterRefineParams, ClusterRefineResult, FeatureGeometry, MemberStatus,
    CELL_COUNT, DEFAULT_MAX_CAPPED_CELLS, DEFAULT_MAX_SHIFT_SCALES,
    DEFAULT_REGATE_AT_REFINED_SHAPE, DEFAULT_ROTATION_STARTS, MAX_PATCH_ELONGATION,
    REFERENCE_UNREFINABLE,
};
pub use piecewise::{
    member_cell_data, CellRefinement, LoopStop, PiecewiseParams, ACCEPT_ZNCC_TOLERANCE,
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

// ── Singular values: the mip rule and the elongation limit ─────────────────

/// The singular values `(s_max, s_min)` of the 2×2 linear map
/// `[[m00, m01], [m10, m11]]`, in closed form: with `q` and `r` the lengths of
/// its similarity and anti-similarity parts, `s_max = q + r` and
/// `s_min = |q − r|`.
fn singular_values(m00: f64, m01: f64, m10: f64, m11: f64) -> (f64, f64) {
    let e = (m00 + m11) * 0.5;
    let f = (m00 - m11) * 0.5;
    let g = (m10 + m01) * 0.5;
    let h = (m10 - m01) * 0.5;
    let q = (e * e + h * h).sqrt();
    let r = (f * f + g * g).sqrt();
    (q + r, (q - r).abs())
}

/// The elongation of the 2×2 linear map `[[m00, m01], [m10, m11]]`: its larger
/// singular value over its smaller, `1` for a similarity. Infinite or `NaN`
/// for a singular map.
fn elongation(m00: f64, m01: f64, m10: f64, m11: f64) -> f64 {
    let (s_max, s_min) = singular_values(m00, m01, m10, m11);
    s_max / s_min
}

/// The members' elongation limit as one cluster's refinement applies it, with
/// the count of the evaluations it turned away.
struct ElongationLimit {
    /// The largest elongation admitted: [`MAX_PATCH_ELONGATION`] in every
    /// refinement but the test that runs the same clusters without the limit,
    /// which passes infinity.
    max: f64,
    /// Evaluations turned away for their map's elongation
    /// ([`ClusterRefineResult::elongated_evaluations`]).
    elongated_evaluations: Cell<u64>,
}

impl ElongationLimit {
    fn new(max: f64) -> Self {
        Self {
            max,
            elongated_evaluations: Cell::new(0),
        }
    }

    /// Whether an elongation is within the limit: at most [`Self::max`]. A
    /// `NaN` elongation is not, and an infinite one is not unless the limit
    /// is infinite.
    fn admits(&self, elongation: f64) -> bool {
        elongation <= self.max
    }

    /// Whether a member's affine shape `a` is within the limit, the test
    /// validation applies to a detected shape.
    fn admits_shape(&self, a: &Mat2) -> bool {
        self.admits(elongation(a[0][0], a[0][1], a[1][0], a[1][1]))
    }

    /// Whether an evaluation may read `map`: `false`, after counting the
    /// evaluation in [`Self::elongated_evaluations`], when the map's linear
    /// part is elongated past the limit. Such an evaluation scores as a
    /// support out of frame does, and is turned away before it asks for any
    /// tile. The map is a shape scaled by the grid's step, so it has the
    /// shape's elongation.
    fn admits_map(&self, map: &AffineCoreMap) -> bool {
        let a = &map.a;
        let within = self.admits(elongation(a[0], a[1], a[3], a[4]));
        if !within {
            let n = &self.elongated_evaluations;
            n.set(n.get() + 1);
        }
        within
    }
}

/// Level `ℓ = clamp(⌊log₂ s_min⌋, 0, L−1)` from the smaller singular value of
/// the map's linear part (closed form). A non-finite or shrinking map stays
/// at level 0 (the degenerate warp is rejected by the sampler's frame test).
fn level_for_map(map: &AffineCoreMap, num_levels: usize) -> usize {
    let a = &map.a;
    let (_, s_min) = singular_values(a[0], a[1], a[3], a[4]);
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
    /// The ZNCC self-similarity radius of the member's own grid at its
    /// refined shape, and its nine cells' radii, when the gates at the
    /// refined shape read it.
    refined_radius: f32,
    refined_radius_grid: [[f32; 3]; 3],
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
            refined_radius: f32::NAN,
            refined_radius_grid: [[f32::NAN; 3]; 3],
        }
    }
}

impl MemberOutcome {
    /// The part readings in the outcome's own `f32`.
    fn set_parts(&mut self, parts: PartZncc) {
        self.zncc_middle = parts.middle as f32;
        self.zncc_grid = parts.grid.map(|row| row.map(|z| z as f32));
    }

    /// The member's refined shape and position, from the stored affine.
    fn shape_and_position(&self) -> (Mat2, [f64; 2]) {
        let a = &self.affine;
        ([[a[0][0], a[0][1]], [a[1][0], a[1][1]]], [a[0][2], a[1][2]])
    }

    /// Read the member's own grid at its refined shape and position, store
    /// the readings, and return the verdict of the two gates at the refined
    /// shape ([`ClusterRefineParams::refined_shape_verdict`]): `None` when it
    /// passes, or when the grid cannot be sampled there, which skips the
    /// gates as non-finite geometry skips the up-front gate.
    ///
    /// The whole grid is read by the up-front gate's own reading
    /// ([`grid_self_similarity_radius`]), so a grid gives the same radius
    /// before and after refinement; the cells come from the parts reading of
    /// the same grid ([`grid_self_similarity_parts`]).
    fn read_refined_shape(
        &mut self,
        view: &mut ViewRegions<'_>,
        params: &ClusterRefineParams,
    ) -> Result<Option<MemberStatus>, PhotographReadError> {
        let (shape, position) = self.shape_and_position();
        let Some(samples) = sample_source_member_grid(view, position, shape, params)? else {
            return Ok(None);
        };
        let resolution = params.resolution.max(2) as usize;
        let Some(parts) = grid_self_similarity_parts(&samples, resolution) else {
            return Ok(None);
        };
        let radius = grid_self_similarity_radius(&samples, resolution);
        prof::count(&prof::N_REFINED_GATED, 1);
        let cells = parts
            .grid
            .each_ref()
            .map(|row| row.each_ref().map(|c| c.radius));
        self.refined_radius = radius as f32;
        self.refined_radius_grid = cells.map(|row| row.map(|r| r as f32));
        Ok(params.refined_shape_verdict(radius, &cells))
    }
}

struct ClusterOutcome {
    reference: u32,
    members: Vec<MemberOutcome>,
    /// Members refused at validation for their elongation
    /// ([`ClusterRefineResult::elongated_members`]).
    elongated_members: usize,
    /// Evaluations scored out of frame for their map's elongation
    /// ([`ClusterRefineResult::elongated_evaluations`]).
    elongated_evaluations: u64,
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

/// [`sample_member_grid`] read through `view`, one view of a tiled
/// photograph: the same grid, level and samples, bit for bit, read by
/// [`sample_source_grid`] from the region the grid's cells take, copied from
/// the view's region pool or read through the photograph's tiles.
///
/// Track View's cluster-stage tiles and their hover views call it.
/// `Ok(None)` where [`sample_member_grid`] gives `None`.
pub fn sample_source_member_grid(
    view: &mut ViewRegions<'_>,
    position: [f64; 2],
    affine_shape: [[f64; 2]; 2],
    params: &ClusterRefineParams,
) -> Result<Option<Vec<f32>>, PhotographReadError> {
    let det = det2(&affine_shape);
    if !det.is_finite() || det.abs() < MIN_ABS_DET {
        return Ok(None);
    }
    let resolution = params.resolution.max(2);
    let step = 2.0 * params.radius / resolution as f64;
    let off = 0.5 * step - params.radius;
    sample_source_grid_window(
        view,
        position,
        &affine_shape,
        step,
        off,
        0,
        resolution as usize,
        GridEdge::Clamp,
    )
}

/// The ZNCC self-similarity radius of one member's own `R×R` grid
/// ([`sample_member_grid`]) at `position` and `affine_shape`, in template-grid
/// px, read the overlap way with the default [`SelfSimilarityParams`]: at each
/// shift only the samples both windows hold are correlated, so the reading
/// depends on no pixel outside the grid. This is the number
/// [`ClusterRefineParams::max_member_zncc_self_similarity_radius`] judges.
/// The channels are read by [`InterleavedTile`]'s rule: a second or fourth
/// channel is the image's alpha, and a sample where it is 0 carries no data.
/// `None` where the grid cannot be sampled.
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

/// The overlap reading of one member's own `R×R` grid
/// ([`sample_member_grid`]) at `position` and `affine_shape`, as a whole, its
/// middle square and its nine cells, with the default
/// [`SelfSimilarityParams`]. Its `grid` is what
/// [`ClusterRefineParams::max_capped_cells`] judges at the refined shape. Its
/// `whole` is the template [`member_zncc_self_similarity_radius`] reads, read
/// with the parts' shared sums, so the two can differ in the last bits; the
/// gates judge the whole grid by [`member_zncc_self_similarity_radius`]'s
/// reading. `None` where the grid cannot be sampled, or has a channel count
/// [`InterleavedTile`] refuses.
pub fn member_zncc_self_similarity_parts(
    pyramid: &ImageU8Pyramid,
    position: [f64; 2],
    affine_shape: [[f64; 2]; 2],
    params: &ClusterRefineParams,
) -> Option<SelfSimilarityParts> {
    let samples = sample_member_grid(pyramid, position, affine_shape, params)?;
    grid_self_similarity_parts(&samples, params.resolution.max(2) as usize)
}

/// An interleaved `R×R×C` grid from [`sample_member_grid`] read by
/// [`InterleavedTile`]'s channel rule; `None` for a grid of a channel count
/// that rule refuses.
fn grid_tile(samples: &[f32], resolution: usize) -> Option<InterleavedTile> {
    let channels = samples.len() / (resolution * resolution);
    InterleavedTile::new(samples, resolution, resolution, channels, None).ok()
}

/// The overlap reading of an interleaved `R×R×C` grid from
/// [`sample_member_grid`] as a whole, its middle and its nine cells; `None`
/// for a grid of a channel count [`InterleavedTile`] refuses.
fn grid_self_similarity_parts(samples: &[f32], resolution: usize) -> Option<SelfSimilarityParts> {
    let tile = grid_tile(samples, resolution)?;
    Some(zncc_self_similarity_parts(
        &tile.tile(),
        tile.data(),
        &SelfSimilarityParams::default(),
    ))
}

/// The overlap reading's ZNCC self-similarity radius of an interleaved
/// `R×R×C` grid from [`sample_member_grid`]; `NaN` for a grid of a channel
/// count [`InterleavedTile`] refuses.
fn grid_self_similarity_radius(samples: &[f32], resolution: usize) -> f64 {
    let Some(tile) = grid_tile(samples, resolution) else {
        return f64::NAN;
    };
    zncc_self_similarity_radius(
        &tile.tile(),
        tile.data(),
        [0, 0, resolution, resolution],
        &SelfSimilarityParams::default(),
    )
    .radius
}

/// Sample a member's own full `R×R` grid at its SIFT geometry (identity
/// warp, mip-selected level, bit-exact `bilinear_cell` convention) into
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
    let (level, lmap) = grid_level_map(pos, a, step, off, pyramid.num_levels());
    let [w, h] = pyramid.level_full_size(level);
    let cells = grid_cells(&lmap.a, GridPoints::Window { first, size }, edge, (w, h))?;
    Some(sample_grid(&HeldLevel::of(pyramid, level), &cells))
}

/// [`sample_grid_window`] read through `view`, one view of a tiled
/// photograph, by [`sample_source_grid`]: the same grid, level and samples,
/// bit for bit. `Ok(None)` where [`sample_grid_window`] gives `None`.
#[allow(clippy::too_many_arguments)]
fn sample_source_grid_window(
    view: &mut ViewRegions<'_>,
    pos: [f64; 2],
    a: &Mat2,
    step: f64,
    off: f64,
    first: i64,
    size: usize,
    edge: GridEdge,
) -> Result<Option<Vec<f32>>, PhotographReadError> {
    let (level, lmap) = grid_level_map(pos, a, step, off, view.source().layout().levels());
    sample_source_grid(
        view,
        &lmap.a,
        level,
        GridPoints::Window { first, size },
        edge,
    )
}

/// The level a template grid at position `pos` and shape `a` is read from,
/// out of `levels`, by [`level_for_map`], and the grid's map in that level's
/// pixels.
fn grid_level_map(
    pos: [f64; 2],
    a: &Mat2,
    step: f64,
    off: f64,
    levels: usize,
) -> (usize, AffineCoreMap) {
    let map = warp_map(pos, [0.0, 0.0], a, step, off);
    let level = level_for_map(&map, levels);
    (level, map_at_level(&map, level))
}

/// Build the reference's z-normalized template kernel: sample every image
/// channel over the support grid at the mip-selected level (bit-exact
/// `bilinear_cell` convention, all-in-frame required), z-normalize each
/// channel with the sqrt-window fold, and drop flat channels. `Ok(None)` when
/// any support sample leaves the frame or every channel is flat — the
/// candidate reference is unusable. Both of its reads go through `view`, the
/// candidate's view.
fn build_template(
    view: &mut ViewRegions<'_>,
    geo: &MemberGeo,
    support: &Support,
    tables: &SupportTables,
    resolution: u32,
    step: f64,
    off: f64,
) -> Result<Option<TemplateKernel>, PhotographReadError> {
    let (level, lmap) = grid_level_map(geo.pos, &geo.a, step, off, view.source().layout().levels());
    let n = support.pixels.len();
    let ch = view.source().channels() as usize;
    let points = GridPoints::Pixels {
        pixels: &support.pixels,
        resolution: resolution as usize,
    };
    let Some(samples) = sample_source_grid(view, &lmap.a, level, points, GridEdge::InFrame)? else {
        return Ok(None);
    };

    // Raw support samples, planar per channel `[c·n + k]` (the ContextTile
    // layout), values un-rounded f32.
    let mut raw = vec![0f32; ch * n];
    for (k, sample) in samples.chunks_exact(ch).enumerate() {
        for (c, &v) in sample.iter().enumerate() {
            raw[c * n + k] = v;
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
        return Ok(None);
    }

    // The same channels over the whole square, for the ZNCC grid. A pixel off
    // the image is `NaN`, which leaves its cell unread rather than refusing the
    // template: the square reaches past the support the refinement scores.
    let r = resolution as usize;
    let square = r * r;
    let mut square_samples = vec![f32::NAN; src_channels.len() * square];
    let window = GridPoints::Window { first: 0, size: r };
    if let Some(all) = sample_source_grid(view, &lmap.a, level, window, GridEdge::Missing)? {
        for (p, sample) in all.chunks_exact(ch).enumerate() {
            for (tc, &c) in src_channels.iter().enumerate() {
                square_samples[tc * square + p] = sample[c];
            }
        }
    }
    Ok(Some(TemplateKernel {
        channels: src_channels.len(),
        src_channels,
        kern,
        kern_sums,
        samples,
        square_samples,
    }))
}

/// The seed warps the cascade may start from: `M₀` turned by each of the
/// [`ClusterRefineParams::rotation_starts`] rotations evenly spaced over a
/// full turn, `R(2πk/K)·M₀` for `k = 0 .. K`. The first is `M₀` itself.
/// Turning on the left turns the member's samples about its tile centre and
/// keeps the seed's scale, skew and translation.
fn rotation_start_warps(m0: &Mat2, starts: u32) -> Vec<Mat2> {
    let starts = starts.max(1);
    (0..starts)
        .map(|k| {
            if k == 0 {
                return *m0;
            }
            let angle = std::f64::consts::TAU * k as f64 / starts as f64;
            let (sin, cos) = angle.sin_cos();
            mul2(&[[cos, -sin], [sin, cos]], m0)
        })
        .collect()
}

/// What [`refine_member`] returns for a member it measured: `(zncc, parts,
/// shift_px, absolute 2×3 affine)`.
type MemberFit = (f64, PartZncc, f64, [[f64; 3]; 2]);

/// Refine one non-reference member: score the template against the member at
/// each rotation start ([`rotation_start_warps`]) with no optimisation, then
/// run the shift → similarity → affine
/// Nelder-Mead cascade on the negated windowed ZNCC from the best-scoring
/// start. Returns
/// `(zncc, parts, shift_px, absolute 2×3 affine)`, where `parts` is the same
/// final map read over the middle of the grid and over each cell of the ZNCC
/// grid ([`eval_zncc_parts`]) and the affine's leading 2×2 is the member's
/// absolute affine shape `S = W·S_ref`, last column its refined absolute
/// keypoint position; `Ok(None)` when the support is out of frame at every
/// rotation start (→ `NotEvaluated`).
///
/// An evaluation whose map is elongated past [`MAX_PATCH_ELONGATION`] scores
/// as one whose support is out of frame, before it reads a pixel, and is
/// counted in `limit`. The member's tiles are read from `source` through
/// `pool`; a read that fails stops the cascade, and its error is returned.
#[allow(clippy::too_many_arguments)]
fn refine_member<'p>(
    source: &TiledPhotograph,
    pool: &'p RegionPool,
    ref_geo: &MemberGeo,
    mem_geo: &MemberGeo,
    tmpl: &TemplateKernel,
    tables: &SupportTables,
    resolution: u32,
    step: f64,
    off: f64,
    params: &ClusterRefineParams,
    limit: &ElongationLimit,
) -> Result<Option<MemberFit>, PhotographReadError> {
    let a_ref_inv = inv2(&ref_geo.a);
    let num_levels = source.layout().levels();

    let eval_in = |tiles: &mut TileCache<'p>,
                   m0: &Mat2,
                   t: [f64; 2],
                   d: Mat2|
     -> Result<Option<f64>, PhotographReadError> {
        let id = [[1.0 + d[0][0], d[0][1]], [d[1][0], 1.0 + d[1][1]]];
        let b = mul2(&mul2(&id, m0), &ref_geo.a);
        let map = warp_map(mem_geo.pos, t, &b, step, off);
        if !limit.admits_map(&map) {
            return Ok(None);
        }
        let level = level_for_map(&map, num_levels);
        let lmap = map_at_level(&map, level);
        let bbox = grid_bbox(&lmap, resolution);
        let Some(tile) = tiles.get_or_build(source, pool, level, bbox)? else {
            return Ok(None);
        };
        prof::count(&prof::N_EVALS, 1);
        Ok(prof::EVAL.time(|| eval_zncc(&lmap, tile, tables, tmpl)))
    };

    // The rotation starts: one score each at the seed's translation and
    // scale. A start whose support leaves the frame is skipped; the
    // best-scoring start seeds the cascade, ties to the earliest (the SIFT
    // seed itself is the first). Out of frame at every start → the member is
    // not evaluated. The scores read tiles of their own: a tile is centred on
    // the mean of the region it covers, so tiles grown by the other starts'
    // renders would change the last bits of every cascade score, and the
    // cascade from the seed would no longer be the single-start one.
    let seed_m0 = mul2(&mem_geo.a, &a_ref_inv);
    let m0 = if params.rotation_starts <= 1 {
        seed_m0
    } else {
        let mut start_tiles = TileCache::default();
        let mut best_start: Option<(f64, Mat2)> = None;
        for m0 in rotation_start_warps(&seed_m0, params.rotation_starts) {
            let Some(z) = eval_in(&mut start_tiles, &m0, [0.0, 0.0], [[0.0; 2]; 2])? else {
                continue;
            };
            prof::count(&prof::N_EVALS_START, 1);
            let better = match best_start {
                None => true,
                Some((best, _)) => z > best || (best.is_nan() && !z.is_nan()),
            };
            if better {
                best_start = Some((z, m0));
            }
        }
        let Some((_, m0)) = best_start else {
            return Ok(None);
        };
        m0
    };
    let mut tiles = TileCache::default();
    let mut eval_raw = |t: [f64; 2], d: Mat2| eval_in(&mut tiles, &m0, t, d);

    // Start support out of frame → the member is not evaluated.
    if eval_raw([0.0, 0.0], [[0.0; 2]; 2])?.is_none() {
        return Ok(None);
    }
    prof::count(&prof::N_EVALS_SHIFT, 1);

    // The first read that failed in the cascade. Every evaluation after it
    // scores worst without reading, and the cascade returns it.
    let failure: RefCell<Option<PhotographReadError>> = RefCell::new(None);

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
                if failure.borrow().is_some() {
                    return 1.0;
                }
                let (t, d) = unpack(x, stage);
                match eval_raw(t, d) {
                    Ok(Some(z)) => -z,
                    // Any support sample out of frame, or a map elongated
                    // past the limit, scores worst (+1.0) so the simplex
                    // retreats — the all-in-frame rule.
                    Ok(None) => 1.0,
                    Err(e) => {
                        *failure.borrow_mut() = Some(e);
                        1.0
                    }
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
        if let Some(e) = failure.take() {
            return Err(e);
        }
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
        if limit.admits_map(&map) {
            let level = level_for_map(&map, num_levels);
            let lmap = map_at_level(&map, level);
            let bbox = grid_bbox(&lmap, resolution);
            tiles
                .get_or_build(source, pool, level, bbox)?
                .map_or(PartZncc::NAN, |tile| {
                    eval_zncc_parts(&lmap, tile, tables, tmpl)
                })
        } else {
            PartZncc::NAN
        }
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
    Ok(Some((
        zncc,
        parts,
        shift,
        [
            [s_abs[0][0], s_abs[0][1], p[0]],
            [s_abs[1][0], s_abs[1][1], p[1]],
        ],
    )))
}

/// Refine one cluster (the per-cluster algorithm of the spec).
///
/// Each usable member reads its photograph through a [`ViewRegions`] of its
/// own, which its gate grid, its template, its grid at the refined shape and
/// its piecewise renders share, and the cascade's tiles through `pool`; all
/// of them are dropped when the cluster is done. A read that fails stops the
/// cluster, and its error is returned.
#[allow(clippy::too_many_arguments)]
fn refine_cluster(
    k0: usize,
    k1: usize,
    sources: &[&TiledPhotograph],
    pool: &RegionPool,
    features: &[FeatureGeometry<'_>],
    member_images: &[u32],
    member_features: &[u32],
    params: &ClusterRefineParams,
    support: &Support,
    tables: &SupportTables,
    resolution: u32,
    max_elongation: f64,
) -> Result<ClusterOutcome, PhotographReadError> {
    let size = k1 - k0;
    let mut members = vec![MemberOutcome::default(); size];
    let unrefinable = |members, elongated_members| ClusterOutcome {
        reference: REFERENCE_UNREFINABLE,
        members,
        elongated_members,
        elongated_evaluations: 0,
    };

    // 1. Validate members: feature index in bounds, |det A| ≥ MIN_ABS_DET,
    // and the shape's elongation within MAX_PATCH_ELONGATION. A member that
    // fails stays NotEvaluated.
    let mut geo: Vec<Option<MemberGeo>> = vec![None; size];
    let mut elongated_members = 0;
    let limit = ElongationLimit::new(max_elongation);
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
        if !limit.admits_shape(&a) {
            elongated_members += 1;
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
    // One view per usable member, for every read of its photograph but the
    // cascade's tiles.
    let mut views: Vec<Option<ViewRegions<'_>>> = geo
        .iter()
        .map(|g| g.as_ref().map(|g| ViewRegions::new(sources[g.image], pool)))
        .collect();

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
            let view = views[j].as_mut().expect("a usable member has a view");
            let Some(grid) =
                prof::GATE_SAMPLE.time(|| sample_source_member_grid(view, g.pos, g.a, params))?
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
        return Ok(unrefinable(members, elongated_members));
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
        let view = views[j].as_mut().expect("a usable member has a view");
        if let Some(t) = prof::TEMPLATE
            .time(|| build_template(view, g, support, tables, resolution, step, off))?
        {
            reference = Some((j, t));
            break;
        }
    }
    let Some((ref_j, tmpl)) = reference else {
        return Ok(unrefinable(members, elongated_members));
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
                sources[g.image],
                pool,
                &ref_geo,
                g,
                &tmpl,
                tables,
                resolution,
                step,
                off,
                params,
                &limit,
            )
        })? {
            let status = if zncc < params.min_zncc {
                MemberStatus::RejectedLowZncc
            } else if !params.admits_shift(
                shift,
                &[[affine[0][0], affine[0][1]], [affine[1][0], affine[1][1]]],
            ) {
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
            // The member gate again, at the shape the cascade returned:
            // before the per-image dedupe, so a refused member cannot cost
            // its image a member that would have passed.
            if status == MemberStatus::Kept && params.reads_refined_shape() {
                let member = &mut members[j];
                let view = views[j].as_mut().expect("a usable member has a view");
                if let Some(refused) =
                    prof::REFINED_GATE.time(|| member.read_refined_shape(view, params))?
                {
                    prof::count(&prof::N_REFINED_GATE_REJECTED, 1);
                    member.status = refused;
                }
            }
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
            let layout = CellLayout::new(resolution, pp.cell_shift_bound_px);
            let templates = CellTemplates::new(&tmpl, &layout);
            for j in kept {
                let g = geo[j].as_ref().unwrap();
                prof::count(&prof::N_PIECEWISE, 1);
                let view = views[j].as_mut().expect("a usable member has a view");
                prof::PIECEWISE.time(|| {
                    refine_kept_member_cells(
                        &mut members[j],
                        view,
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
                        &limit,
                    )
                })?;
            }
        }
    }

    Ok(ClusterOutcome {
        reference: ref_geo.k_global,
        members,
        elongated_members,
        elongated_evaluations: limit.elongated_evaluations.get(),
    })
}

/// Run the piecewise refinement on one kept member and store its outcome:
/// the cells always, and, when the loop moved the shape, the new shape and
/// position with the whole-patch ZNCC, its parts and the shift from the seed
/// read again at the new map. Without [`PiecewiseParams::move_shape`] the
/// shape never moves, so only the cells are stored and every other reading of
/// the member is left untouched.
///
/// The member was kept on the cascade's readings, so the stage's shape must
/// pass the same gates to replace it: the ZNCC read again at the new map must
/// be at least [`ClusterRefineParams::min_zncc`] and at least the cascade's own
/// stored ZNCC, the new position's shift from the seed must pass the shift
/// gate at the new shape ([`ClusterRefineParams::admits_shift`]), and, when either gate at the refined
/// shape is on ([`ClusterRefineParams::reads_refined_shape`]), the member's
/// own grid read at the new shape must pass both. When any fails, or when the
/// new map's support leaves the frame so nothing can be read, the member
/// keeps its cascade shape, position and readings, the readings at the
/// refined shape included, and every cell is stored as
/// [`CellStatus::NotAttempted`] with the loop's iteration count. A shape
/// elongated past [`MAX_PATCH_ELONGATION`] cannot be read either, and is
/// counted in `limit`.
///
/// The stage never changes a member's status. The member passed every gate
/// at its cascade shape, and the per-image dedupe has already run, so a
/// refusal here could leave its image with no member, where reverting keeps
/// the member that passed.
///
/// The working patch and the grid at the moved shape are read through
/// `view`, the member's view, and the scores' tiles through its pool. A read
/// that fails stops the stage, and its error is returned.
#[allow(clippy::too_many_arguments)]
fn refine_kept_member_cells(
    member: &mut MemberOutcome,
    view: &mut ViewRegions<'_>,
    seed_pos: [f64; 2],
    tmpl: &TemplateKernel,
    templates: &CellTemplates,
    layout: &CellLayout,
    tables: &SupportTables,
    resolution: u32,
    step: f64,
    off: f64,
    params: &ClusterRefineParams,
    pp: &PiecewiseParams,
    limit: &ElongationLimit,
) -> Result<(), PhotographReadError> {
    let s = [
        [member.affine[0][0], member.affine[0][1]],
        [member.affine[1][0], member.affine[1][1]],
    ];
    let p = [member.affine[0][2], member.affine[1][2]];
    let (source, pool) = (view.source(), view.pool());
    // The cascade's own objective, which an update must not lower.
    let score = |shape: &Mat2, position: [f64; 2]| {
        member_zncc_at(
            source, pool, position, shape, tmpl, tables, resolution, step, off, limit,
        )
    };
    let out = piecewise::refine_member_cells(
        view,
        &tmpl.src_channels,
        templates,
        layout,
        s,
        p,
        step,
        off,
        pp,
        score,
    )?;
    if !out.updated {
        member.cells = Some(out.cells);
        return Ok(());
    }
    let (sh, ps) = (out.shape, out.position);
    let shift = (ps[0] - seed_pos[0]).hypot(ps[1] - seed_pos[1]);
    // The stored ZNCC must not fall below the cascade's: the loop's floor is
    // its own reading of the starting shape, which can differ from the
    // cascade's stored value in the last bits, so the stored value is the bar.
    let cascade_zncc = member.zncc;
    let reread = read_member_at(
        source, pool, ps, &sh, tmpl, tables, resolution, step, off, limit,
    )?
    .filter(|&(zncc, _)| {
        zncc >= params.min_zncc && zncc as f32 >= cascade_zncc && params.admits_shift(shift, &sh)
    });
    let Some((zncc, parts)) = reread else {
        member.cells = Some(CellRefinement::not_attempted(out.cells.iterations));
        return Ok(());
    };
    let cascade = member.clone();
    member.affine = [[sh[0][0], sh[0][1], ps[0]], [sh[1][0], sh[1][1], ps[1]]];
    member.zncc = zncc as f32;
    member.set_parts(parts);
    member.shift = shift as f32;
    if params.reads_refined_shape() && member.read_refined_shape(view, params)?.is_some() {
        // The moved shape fails a gate the cascade shape passed: keep the
        // cascade shape and its readings.
        prof::count(&prof::N_REFINED_GATE_REVERTED, 1);
        *member = cascade;
        member.cells = Some(CellRefinement::not_attempted(out.cells.iterations));
        return Ok(());
    }
    member.cells = Some(out.cells);
    Ok(())
}

/// The whole-patch windowed ZNCC and its part readings of a member at
/// absolute position `p` and shape `s`, its tile read from `source` through
/// `pool`; `Ok(None)` when the support leaves the frame, or when the shape is
/// elongated past [`MAX_PATCH_ELONGATION`], which is counted in `limit`.
#[allow(clippy::too_many_arguments)]
fn read_member_at(
    source: &TiledPhotograph,
    pool: &RegionPool,
    p: [f64; 2],
    s: &Mat2,
    tmpl: &TemplateKernel,
    tables: &SupportTables,
    resolution: u32,
    step: f64,
    off: f64,
    limit: &ElongationLimit,
) -> Result<Option<(f64, PartZncc)>, PhotographReadError> {
    let map = warp_map(p, [0.0, 0.0], s, step, off);
    if !limit.admits_map(&map) {
        return Ok(None);
    }
    let level = level_for_map(&map, source.layout().levels());
    let lmap = map_at_level(&map, level);
    let bbox = grid_bbox(&lmap, resolution);
    let mut tiles = TileCache::default();
    let Some(tile) = tiles.get_or_build(source, pool, level, bbox)? else {
        return Ok(None);
    };
    Ok(eval_zncc(&lmap, tile, tables, tmpl)
        .map(|zncc| (zncc, eval_zncc_parts(&lmap, tile, tables, tmpl))))
}

/// The whole-patch windowed ZNCC of a member at absolute position `p` and
/// shape `s`, the objective the cascade maximises, its tile read from
/// `source` through `pool`; `Ok(None)` when the support leaves the frame, or
/// when the shape is elongated past [`MAX_PATCH_ELONGATION`], which is
/// counted in `limit`.
#[allow(clippy::too_many_arguments)]
fn member_zncc_at(
    source: &TiledPhotograph,
    pool: &RegionPool,
    p: [f64; 2],
    s: &Mat2,
    tmpl: &TemplateKernel,
    tables: &SupportTables,
    resolution: u32,
    step: f64,
    off: f64,
    limit: &ElongationLimit,
) -> Result<Option<f64>, PhotographReadError> {
    let map = warp_map(p, [0.0, 0.0], s, step, off);
    if !limit.admits_map(&map) {
        return Ok(None);
    }
    let level = level_for_map(&map, source.layout().levels());
    let lmap = map_at_level(&map, level);
    let bbox = grid_bbox(&lmap, resolution);
    let mut tiles = TileCache::default();
    let Some(tile) = tiles.get_or_build(source, pool, level, bbox)? else {
        return Ok(None);
    };
    Ok(eval_zncc(&lmap, tile, tables, tmpl))
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
/// The kernel does no I/O and reads no `.sift` file -- the caller supplies the
/// photographs and feature geometry, one [`FeatureGeometry`] per photograph. A
/// colour photograph is RGB, as [`ImageU8::read_rgb`](crate::camera::image::ImageU8::read_rgb)
/// decodes it. The score averages the channels, so their order changes only
/// the order of that floating-point sum. Every read of a photograph goes
/// through the render entry points with the region pool the caller made for
/// this call (`specs/core/camera/view-region-pyramids.md`), so the results are
/// the same whichever way a read reaches its pixels.
///
/// # Errors
///
/// The [`PhotographReadError`] of the first read of a photograph that failed;
/// a photograph made in memory fails only for pixels it does not hold.
///
/// # Panics
///
/// Panics when `features` is not parallel to `sources` or the CSR arrays are
/// inconsistent (`cluster_starts` must start at 0, be non-decreasing, and end
/// at the member count; the two member arrays must have equal length). A
/// member whose image index is out of range for `sources`, whose feature
/// index is out of range for its image, or whose affine shape is degenerate
/// or elongated past [`MAX_PATCH_ELONGATION`] is reported as
/// [`MemberStatus::NotEvaluated`] rather than panicking.
#[allow(clippy::too_many_arguments)]
pub fn refine_cluster_patches(
    sources: &[TiledPhotograph],
    pool: &RegionPool,
    features: &[FeatureGeometry<'_>],
    cluster_starts: &[u32],
    member_images: &[u32],
    member_features: &[u32],
    params: &ClusterRefineParams,
    progress: Option<&AtomicUsize>,
) -> Result<ClusterRefineResult, PhotographReadError> {
    let borrowed: Vec<&TiledPhotograph> = sources.iter().collect();
    refine_cluster_patches_borrowed(
        &borrowed,
        pool,
        features,
        cluster_starts,
        member_images,
        member_features,
        params,
        progress,
    )
}

/// [`refine_cluster_patches`] over **borrowed** photographs: the same kernel,
/// for a caller that holds one photograph per image behind a reference rather
/// than a table of its own.
///
/// A caller whose views are
/// [`ProjectedImage`](crate::patch::normal_refine::ProjectedImage)s -- the
/// bench's evaluation, which registers an in-memory cluster over the same views
/// the patch kernels read -- reaches the kernel through this entry instead of
/// building a table it would have to own. Everything else, the errors and the
/// panics included, is [`refine_cluster_patches`].
#[allow(clippy::too_many_arguments)]
pub fn refine_cluster_patches_borrowed(
    sources: &[&TiledPhotograph],
    pool: &RegionPool,
    features: &[FeatureGeometry<'_>],
    cluster_starts: &[u32],
    member_images: &[u32],
    member_features: &[u32],
    params: &ClusterRefineParams,
    progress: Option<&AtomicUsize>,
) -> Result<ClusterRefineResult, PhotographReadError> {
    refine_cluster_patches_with_elongation_limit(
        sources,
        pool,
        features,
        cluster_starts,
        member_images,
        member_features,
        params,
        MAX_PATCH_ELONGATION,
        progress,
    )
}

/// [`refine_cluster_patches_borrowed`] with the members' elongation limit
/// given: `max_elongation` in place of [`MAX_PATCH_ELONGATION`]. Every
/// refinement passes [`MAX_PATCH_ELONGATION`] but the test that runs the same
/// clusters with the limit switched off (`f64::INFINITY`) and checks that the
/// limit changes nothing where it turns nothing away.
#[allow(clippy::too_many_arguments)]
fn refine_cluster_patches_with_elongation_limit(
    sources: &[&TiledPhotograph],
    pool: &RegionPool,
    features: &[FeatureGeometry<'_>],
    cluster_starts: &[u32],
    member_images: &[u32],
    member_features: &[u32],
    params: &ClusterRefineParams,
    max_elongation: f64,
    progress: Option<&AtomicUsize>,
) -> Result<ClusterRefineResult, PhotographReadError> {
    assert_eq!(
        features.len(),
        sources.len(),
        "features must be parallel to sources"
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
                    sources,
                    pool,
                    features,
                    member_images,
                    member_features,
                    params,
                    &support,
                    &tables,
                    resolution,
                    max_elongation,
                )
            })?;
            if let Some(p) = progress {
                p.fetch_add(1, Ordering::Relaxed);
            }
            Ok(out)
        })
        .collect::<Result<_, PhotographReadError>>()?;
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
        refined_zncc_self_similarity_radius: vec![f32::NAN; m],
        refined_zncc_self_similarity_radius_grid: vec![[[f32::NAN; 3]; 3]; m],
        elongated_members: 0,
        elongated_evaluations: 0,
    };
    for (c, out) in outcomes.into_iter().enumerate() {
        result.reference_members[c] = out.reference;
        result.elongated_members += out.elongated_members;
        result.elongated_evaluations += out.elongated_evaluations;
        let k0 = cluster_starts[c] as usize;
        for (j, mo) in out.members.into_iter().enumerate() {
            let k = k0 + j;
            result.member_status[k] = mo.status;
            result.member_zncc[k] = mo.zncc;
            result.member_zncc_middle[k] = mo.zncc_middle;
            result.member_zncc_grid[k] = mo.zncc_grid;
            result.member_shift_px[k] = mo.shift;
            result.refined_zncc_self_similarity_radius[k] = mo.refined_radius;
            result.refined_zncc_self_similarity_radius_grid[k] = mo.refined_radius_grid;
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
    Ok(result)
}
