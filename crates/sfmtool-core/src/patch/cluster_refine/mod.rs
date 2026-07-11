// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Cluster-patch refinement: turn SIFT feature clusters into patch clusters.
//!
//! See `specs/core/cluster-patch-refinement.md` (implementation) and
//! `specs/core/cluster-patches.md` (design). Given per-image pyramids, SIFT
//! feature geometry, and CSR clusters, [`refine_cluster_patches`] first
//! gates each member on the localizability of its own patch (the
//! noise-normalized structure-tensor uncertainty of
//! `specs/core/patch-localizability.md`; members above
//! `max_keypoint_uncertainty` are excluded up front), then picks a
//! reference member per cluster (largest SIFT scale, deterministic
//! tie-breaks), builds a Gaussian-windowed z-normalized template around the
//! reference detection, refines an affine warp to every other member by a
//! shift → similarity → affine Nelder-Mead cascade on the windowed ZNCC
//! (seeded from the SIFT affine shapes, `M₀ = A_mem · A_ref⁻¹`), vets by
//! achieved ZNCC and translation drift, dedupes to one kept member per image,
//! and emits member-parallel arrays that map 1:1 onto the `.matches`
//! `cluster_patches/` section.
//!
//! Sampling reuses the house conventions: the `bilinear_geometry` pixel-center
//! convention (`x − 0.5`), the shared window [`Support`], and the
//! `weighted_moments_pub` / `znorm_write` z-normalization kernels. Pyramid
//! levels follow the standard mip rule — per sampled image, level
//! `ℓ = clamp(⌊log₂ s_min⌋, 0, L−1)` where `s_min` is the smaller singular
//! value of the support map's linear part (sample spacing in source px), with
//! the map divided by `2^ℓ` before sampling.

mod consistency;
mod kernels;
mod params;
pub mod prof;

#[cfg(test)]
mod tests;

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

use ndarray::Array3;
use rayon::prelude::*;

use crate::camera::remap::ImageU8Pyramid;
use crate::patch::localizability::patch_localizability;
use crate::patch::normal_refine::{
    build_support, weighted_moments_pub, znorm_write, Support, FLAT_NORM_SQ_EPS,
};
use crate::patch::view_selection::AffineCoreMap;

use kernels::{eval_zncc, grid_bbox, nelder_mead, SupportTables, TemplateKernel, TileCache};

pub use consistency::warp_consistency_residuals;
pub use params::{
    ClusterRefineParams, ClusterRefineResult, FeatureGeometry, MemberStatus, REFERENCE_UNREFINABLE,
};

/// A member's SIFT affine shape is usable when `|det A|` clears this floor.
const MIN_ABS_DET: f64 = 1e-9;

/// Log-scale clamp of the similarity stage (`σ ∈ [−1.5, 1.5]`, the
/// prototype's bound).
const SIGMA_CLAMP: f64 = 1.5;

/// Global photometric-noise constant (intensity units) for the
/// localizability gate — sets the absolute px scale of `σ_pos`, matching
/// the `score_localizability` / `embed-patches` default (see
/// `specs/core/patch-localizability.md`, "σ_noise (v1: global constant)").
const LOCALIZABILITY_SIGMA_NOISE: f64 = 3.0;

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
    shift: f32,
}

impl Default for MemberOutcome {
    fn default() -> Self {
        MemberOutcome {
            status: MemberStatus::NotEvaluated,
            affine: [[0.0; 3]; 2],
            zncc: f32::NAN,
            shift: f32::NAN,
        }
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

/// Sample a member's own full `R×R` grid at its SIFT geometry (identity
/// warp, mip-selected level, bit-exact `bilinear_geometry` convention) into
/// an interleaved `R×R×C` f32 patch — the layout [`patch_localizability`]
/// scores. Unlike [`build_template`], every grid pixel is sampled (the
/// scorer's gradients cover the full grid, not just the windowed support),
/// and samples outside the frame clamp to the nearest valid pixel (border
/// replicate) so a border member is scored on its visible content instead
/// of skipping the gate. `None` only for non-finite coordinates (degenerate
/// geometry) or a level too small to bilinear-sample.
fn sample_patch_grid(
    pyramid: &ImageU8Pyramid,
    geo: &MemberGeo,
    resolution: u32,
    step: f64,
    off: f64,
) -> Option<Vec<f32>> {
    let map = warp_map(geo.pos, [0.0, 0.0], &geo.a, step, off);
    let level = level_for_map(&map, pyramid.num_levels());
    let lmap = map_at_level(&map, level);
    let img = pyramid.level(level);
    let r = resolution as usize;
    let ch = img.channels() as usize;
    let stride = img.width() as usize * ch;
    let data = img.data();
    let (w_img, h_img) = (img.width() as i64, img.height() as i64);
    if w_img < 2 || h_img < 2 {
        return None;
    }
    let (max_gx, max_gy) = ((w_img - 1) as f64, (h_img - 1) as f64);
    let mut out = vec![0f32; r * r * ch];
    for px in 0..r * r {
        let col = (px % r) as f64;
        let row = (px / r) as f64;
        let x = lmap.a[0] * col + lmap.a[1] * row + lmap.a[2];
        let y = lmap.a[3] * col + lmap.a[4] * row + lmap.a[5];
        // `bilinear_geometry`'s pixel-center convention.
        let gx = x - 0.5;
        let gy = y - 0.5;
        if !gx.is_finite() || !gy.is_finite() {
            return None;
        }
        // Nearest-valid-pixel clamp (border replicate): the coordinate
        // clamps to the outermost pixel center and the tap base to the last
        // valid 2x2 cell, so fx/fy saturate and the blend reads the edge
        // pixel.
        let gx = gx.clamp(0.0, max_gx);
        let gy = gy.clamp(0.0, max_gy);
        let ix = (gx.floor() as i64).min(w_img - 2);
        let iy = (gy.floor() as i64).min(h_img - 2);
        let (fx, fy) = ((gx - ix as f64) as f32, (gy - iy as f64) as f32);
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
    }
    if src_channels.is_empty() {
        return None;
    }
    Some(TemplateKernel {
        channels: src_channels.len(),
        src_channels,
        kern,
        kern_sums,
    })
}

/// Refine one non-reference member: the shift → similarity → affine
/// Nelder-Mead cascade on the negated windowed ZNCC. Returns
/// `(zncc, shift_px, absolute 2×3 affine)`; `None` when the seed support is
/// out of frame (→ `NotEvaluated`).
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
) -> Option<(f64, f64, [[f64; 3]; 2])> {
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
    let shift = (t[0] * t[0] + t[1] * t[1]).sqrt();
    // Absolute affine: `A = (I + D)·M₀`, `t_abs = pos_mem + t − A·pos_ref`,
    // so `x_mem = A·x_ref + t_abs` composes without the seed.
    let id = [[1.0 + d[0][0], d[0][1]], [d[1][0], 1.0 + d[1][1]]];
    let a_abs = mul2(&id, &m0);
    let t_abs = [
        mem_geo.pos[0] + t[0] - (a_abs[0][0] * ref_geo.pos[0] + a_abs[0][1] * ref_geo.pos[1]),
        mem_geo.pos[1] + t[1] - (a_abs[1][0] * ref_geo.pos[0] + a_abs[1][1] * ref_geo.pos[1]),
    ];
    Some((
        zncc,
        shift,
        [
            [a_abs[0][0], a_abs[0][1], t_abs[0]],
            [a_abs[1][0], a_abs[1][1], t_abs[1]],
        ],
    ))
}

/// Refine one cluster (the per-cluster algorithm of the spec).
#[allow(clippy::too_many_arguments)]
fn refine_cluster(
    k0: usize,
    k1: usize,
    pyramids: &[ImageU8Pyramid],
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

    // 1b. Localizability gate: score each usable member's own patch and
    // exclude members whose weak-axis positional uncertainty exceeds the
    // threshold — before reference selection, so an unlocalizable patch can
    // neither anchor nor join the cluster. Border patches sample with a
    // nearest-valid-pixel clamp, so they are scored on their visible
    // content; only degenerate geometry (non-finite coordinates, or a
    // sub-2px pyramid level) skips the gate.
    if params.max_keypoint_uncertainty > 0.0 {
        for (j, slot) in geo.iter_mut().enumerate() {
            let Some(g) = slot.as_ref() else {
                continue;
            };
            let Some(raw) = prof::GATE_SAMPLE
                .time(|| sample_patch_grid(&pyramids[g.image], g, resolution, step, off))
            else {
                continue;
            };
            let r = resolution as usize;
            let channels = raw.len() / (r * r);
            prof::count(&prof::N_GATED, 1);
            let loc = prof::GATE_SCORE.time(|| {
                patch_localizability(&raw, r, channels, support, LOCALIZABILITY_SIGMA_NOISE)
            });
            // NaN (empty patch) compares false -> kept, like embed-patches.
            if loc.sigma_pos_grid > params.max_keypoint_uncertainty {
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
        if let Some(t) = prof::TEMPLATE.time(|| {
            build_template(
                &pyramids[g.image],
                g,
                support,
                tables,
                resolution,
                step,
                off,
            )
        }) {
            reference = Some((j, t));
            break;
        }
    }
    let Some((ref_j, tmpl)) = reference else {
        return unrefinable(members);
    };
    let ref_geo = geo[ref_j].clone().unwrap();
    members[ref_j] = MemberOutcome {
        status: MemberStatus::Reference,
        affine: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        zncc: 1.0,
        shift: 0.0,
    };

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
        if let Some((zncc, shift, affine)) = prof::REFINE.time(|| {
            refine_member(
                &pyramids[g.image],
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
            };
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

    ClusterOutcome {
        reference: ref_geo.k_global,
        members,
    }
}

/// Refine every cluster into a patch cluster: per cluster, a reference member
/// plus a vetted absolute affine warp (`x_member = A·x_ref + t`) to every
/// other member. Parallel over clusters (rayon); results are deterministic
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
        member_affines: Array3::zeros((m, 2, 3)),
        member_zncc: vec![f32::NAN; m],
        member_shift_px: vec![f32::NAN; m],
    };
    for (c, out) in outcomes.into_iter().enumerate() {
        result.reference_members[c] = out.reference;
        let k0 = cluster_starts[c] as usize;
        for (j, mo) in out.members.into_iter().enumerate() {
            let k = k0 + j;
            result.member_status[k] = mo.status;
            result.member_zncc[k] = mo.zncc;
            result.member_shift_px[k] = mo.shift;
            for (r, row) in mo.affine.iter().enumerate() {
                for (cc, &v) in row.iter().enumerate() {
                    result.member_affines[[k, r, cc]] = v;
                }
            }
        }
    }
    result
}
