// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Fronto-parallel patch cache for normal refinement (see
//! `specs/core/fronto-parallel-patch-cache.md`).
//!
//! Within one refinement the patch centre, size, and 3D point are fixed — only
//! the normal moves — so the texture each view sees is fixed up to a planar
//! re-warp. Instead of re-rendering every candidate normal from the source
//! image (the cost centre, ~80% of CPU), this renders **one** supersampled base
//! patch per view *up front*, oriented fronto-parallel to that camera, and
//! resamples each candidate from it by an **affine** map. The source image is
//! touched exactly once per view for the whole refinement.
//!
//! The affine map is built from the cameras' **undistorted-normalized** corner
//! projections, so the lens distortion cancels in the base↔candidate
//! correspondence and the cache is exact for any camera model. The base is
//! packed `u32`/pixel (RGB in the low 24 bits) and replicate-padded by a small
//! guard so the resample can clamp with a single branch-free integer bound; the
//! candidate support is resampled straight into the scorer's planar layout.

use nalgebra::{SMatrix, Vector3};

use super::{consensus_phi, repose_patch, LevelContext, NormalRefineParams, ProjectedImage};
use crate::patch_cloud::OrientedPatch;
use crate::remap::remap_bilinear;
use crate::warp_map::WarpMap;

/// Replicate-padded border (base pixels) around each fronto base. A tilted
/// candidate's footprint *mostly* shrinks inside the fronto base, but a
/// near-frontal or grazing candidate can poke a little past the edge; the guard
/// plus a single integer coordinate clamp keep every gather in-bounds. 2 px is
/// enough for the realistic sub-pixel overshoot; the clamp covers the rest.
const FRONTO_GUARD: u32 = 2;

/// The packed channel count the cache assumes (RGB). Non-3-channel views make
/// [`prerender`] bail so the caller falls back to source rendering.
const CHANNELS: usize = 3;

/// One fronto-parallel base patch for a view: the supersampled RGB patch packed
/// as `u32`/pixel and replicate-padded by [`FRONTO_GUARD`], plus the **affine**
/// grid→undistorted-normalized map that produced it (fit on the *unpadded*
/// `bw`×`bw` grid).
struct FrontoBase {
    /// Padded buffer, `bw_pad × bw_pad`.
    patch: Vec<u32>,
    /// Padded width `bw + 2·FRONTO_GUARD` (the buffer's row stride).
    bw_pad: u32,
    /// Affine grid→undistorted-normalized map (homogeneous 3×3, last row 0 0 1).
    a0: SMatrix<f64, 3, 3>,
}

/// Per-view fronto bases, parallel to the caller's `views` slice (`None` where
/// the patch is back-facing in that view or the base could not be built).
pub(super) struct FrontoCache {
    bases: Vec<Option<FrontoBase>>,
}

/// Undistorted-normalized projection `(x/z, y/z)` of the four patch grid corners
/// in `view` (`(0,0)`, `(r-1,0)`, `(0,r-1)`, `(r-1,r-1)`). Omitting the lens
/// distortion makes the base↔candidate map distortion-independent. `None` if any
/// corner is behind the camera.
fn corner_norm_pts(
    patch: &OrientedPatch,
    view: &ProjectedImage<'_>,
    r: u32,
) -> Option<[(f64, f64); 4]> {
    let step = 2.0 / r as f64;
    let corners = [(0u32, 0u32), (r - 1, 0), (0, r - 1), (r - 1, r - 1)];
    let mut out = [(0.0, 0.0); 4];
    for (k, &(c, rw)) in corners.iter().enumerate() {
        let s = (c as f64 + 0.5) * step - 1.0;
        let t = (rw as f64 + 0.5) * step - 1.0;
        let world = patch.to_world(s, t);
        let p = view.cam_from_world.transform_point(&world);
        if p.z <= 1e-9 {
            return None;
        }
        out[k] = (p.x / p.z, p.y / p.z);
    }
    Some(out)
}

/// Affine grid→image map from three projected grid corners — exact for an
/// affine, and the orthographic/differential approximation of the patch
/// homography. Homogeneous 3×3 with last row `[0,0,1]` so it composes like the
/// homography path.
fn affine_grid_to_img(pts: &[(f64, f64); 4], r: u32) -> SMatrix<f64, 3, 3> {
    let d = (r - 1) as f64;
    let (x00, y00) = pts[0];
    let (x10, y10) = pts[1];
    let (x01, y01) = pts[2];
    SMatrix::<f64, 3, 3>::new(
        (x10 - x00) / d,
        (x01 - x00) / d,
        x00,
        (y10 - y00) / d,
        (y01 - y00) / d,
        y00,
        0.0,
        0.0,
        1.0,
    )
}

/// Replicate-pad a `rb×rb` packed-`u32` patch by `guard` pixels on every side.
fn pad_replicate(src: &[u32], rb: usize, guard: usize) -> Vec<u32> {
    let w = rb + 2 * guard;
    let mut out = vec![0u32; w * w];
    for r in 0..w {
        let sr = r.saturating_sub(guard).min(rb - 1);
        for c in 0..w {
            let sc = c.saturating_sub(guard).min(rb - 1);
            out[r * w + c] = src[sr * rb + sc];
        }
    }
    out
}

/// Render the fronto-parallel base, once, for every view that observes the patch
/// around the search `seed`. `supersample` renders the base denser than the
/// candidate grid (sharper resample → fewer argmax flips). Returns `None` —
/// caller falls back to source rendering — if fewer than `min_views` usable
/// bases survive (e.g. non-RGB imagery).
///
/// The front-face cull is against the patch *at the seed* (the search centre),
/// not the input normal: `coarse_to_fine` builds each level's kept-view set at
/// its centre normal (which starts at the seed — possibly the mean-viewing
/// direction, far from the input normal), so culling against the seed is what
/// guarantees every level-0 kept view has a base. A view that only becomes
/// front-facing at a *drifted* later-level centre is rare (it must also pass the
/// validity gate) and merely ends that patch's search one level early.
pub(super) fn prerender(
    base: &OrientedPatch,
    seed: &Vector3<f64>,
    views: &[ProjectedImage<'_>],
    resolution: u32,
    supersample: f64,
    params: &NormalRefineParams,
) -> Option<FrontoCache> {
    let ss = supersample.max(1.0);
    let rb = (((resolution as f64) * ss).round() as u32).max(resolution);
    let seed_patch = repose_patch(base, seed);
    let mut bases: Vec<Option<FrontoBase>> = Vec::with_capacity(views.len());
    for view in views {
        if !seed_patch.is_front_facing(view.cam_from_world) {
            bases.push(None);
            continue;
        }
        // Patch plane faces this camera (least foreshortening → most resolution).
        let cam_c = view.cam_from_world.inverse_translation_origin();
        let fronto_n = (cam_c - base.center).normalize();
        let fp =
            OrientedPatch::from_center_normal(base.center, fronto_n, base.u_axis, base.half_extent);
        let map = WarpMap::from_patch(&fp, view.camera, view.cam_from_world, rb);
        let img = remap_bilinear(view.pyramid.level(0), &map);
        if img.channels() as usize != CHANNELS {
            bases.push(None);
            continue;
        }
        let mut packed = vec![0u32; (rb * rb) as usize];
        for (p, slot) in packed.iter_mut().enumerate() {
            let (c, rw) = ((p as u32) % rb, (p as u32) / rb);
            let r = img.get_pixel(c, rw, 0) as u32;
            let g = img.get_pixel(c, rw, 1) as u32;
            let b = img.get_pixel(c, rw, 2) as u32;
            *slot = r | (g << 8) | (b << 16);
        }
        let fb = corner_norm_pts(&fp, view, rb).map(|pts| FrontoBase {
            patch: pad_replicate(&packed, rb as usize, FRONTO_GUARD as usize),
            bw_pad: rb + 2 * FRONTO_GUARD,
            a0: affine_grid_to_img(&pts, rb),
        });
        bases.push(fb);
    }
    if bases.iter().filter(|b| b.is_some()).count() < params.min_views.max(2) as usize {
        return None;
    }
    Some(FrontoCache { bases })
}

/// Resample the `cols`/`rows` support pixels of a padded base by the affine map
/// `phi` (the 2×3 candidate-grid → unpadded base-grid map, row-major
/// `[a,b,c, d,e,f]`) into the planar `out` (`[R-plane | G-plane | B-plane]`,
/// each `n = cols.len()`). Bilinear, packed `u32` source; a single branch-free
/// integer clamp on the pixel coordinate keeps escaped/degenerate candidates
/// in-bounds (the guard absorbs the sub-pixel overshoot of covered ones).
fn resample_support(
    base: &FrontoBase,
    phi: &[f64; 6],
    cols: &[i32],
    rows: &[i32],
    out: &mut [f32],
) {
    let n = cols.len();
    let bw = base.bw_pad as usize;
    let g = FRONTO_GUARD as f32;
    let hi = (bw - 2) as i32; // x0/y0 ceiling so the +1 neighbour stays in-bounds
    for i in 0..n {
        let cf = cols[i] as f32;
        let rf = rows[i] as f32;
        let bx = (phi[0] as f32) * cf + (phi[1] as f32) * rf + phi[2] as f32 + g;
        let by = (phi[3] as f32) * cf + (phi[4] as f32) * rf + phi[5] as f32 + g;
        let x0 = (bx as i32).clamp(0, hi) as usize;
        let y0 = (by as i32).clamp(0, hi) as usize;
        let fx = bx - x0 as f32;
        let fy = by - y0 as f32;
        let (row0, row1) = (y0 * bw, (y0 + 1) * bw);
        for sh in 0..CHANNELS {
            let unpack = |idx: usize| ((base.patch[idx] >> (8 * sh as u32)) & 0xFF) as f32;
            let top = unpack(row0 + x0) * (1.0 - fx) + unpack(row0 + x0 + 1) * fx;
            let bot = unpack(row1 + x0) * (1.0 - fx) + unpack(row1 + x0 + 1) * fx;
            out[sh * n + i] = top * (1.0 - fy) + bot * fy;
        }
    }
}

/// `Φ` for `candidate_n` evaluated by affine-resampling the cached fronto bases
/// instead of re-rendering from the source images. `cols`/`rows` are the level's
/// frozen support grid coords (computed once per level by the caller). Mirrors
/// [`super::normalized_stack`] + [`consensus_phi`]; `None` if a candidate map
/// fails or no channel survives.
#[allow(clippy::too_many_arguments)]
pub(super) fn eval_phi(
    base: &OrientedPatch,
    candidate_n: &Vector3<f64>,
    cache: &FrontoCache,
    ctx: &LevelContext,
    views: &[ProjectedImage<'_>],
    resolution: u32,
    cols: &[i32],
    rows: &[i32],
    params: &NormalRefineParams,
) -> Option<f64> {
    super::prof::count(&super::prof::N_EVAL, 1);
    let cp = repose_patch(base, candidate_n);
    let n = cols.len();
    let total_w: f64 = ctx.weights.iter().sum();
    if total_w <= 0.0 {
        return None;
    }
    let sqrt_w: Vec<f64> = ctx.weights.iter().map(|&w| w.sqrt()).collect();
    let mut masked = vec![0f32; n * CHANNELS];

    // raw[view][channel][support-pixel], in cols/rows (= ctx.weights) order.
    let mut raw: Vec<Vec<Vec<f64>>> = Vec::with_capacity(ctx.kept.len());
    for &vi in &ctx.kept {
        let fb = cache.bases[vi].as_ref()?;
        let view = &views[vi];
        let ap =
            corner_norm_pts(&cp, view, resolution).map(|p| affine_grid_to_img(&p, resolution))?;
        // Both operands are affine (last row [0,0,1]), so the composition is too;
        // the resampler only needs the 2×3 part (candidate-grid → base-grid).
        let m = fb.a0.try_inverse()? * ap;
        let phi = [
            m[(0, 0)],
            m[(0, 1)],
            m[(0, 2)],
            m[(1, 0)],
            m[(1, 1)],
            m[(1, 2)],
        ];
        resample_support(fb, &phi, cols, rows, &mut masked);
        let per_channel: Vec<Vec<f64>> = (0..CHANNELS)
            .map(|c| (0..n).map(|i| masked[c * n + i] as f64).collect())
            .collect();
        raw.push(per_channel);
    }

    // Same z-normalization as the source-render path, then the shared consensus.
    let xs = super::znormalize_stack(&raw, &ctx.weights, total_w, &sqrt_w)?;
    consensus_phi(&xs, params.objective)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn affine_fit_recovers_known_affine() {
        // A 3-corner fit of points generated by a known affine recovers it
        // exactly, and composing the map with its inverse is the identity (the
        // φ a same-orientation candidate sees → a no-op resample).
        let r = 16u32;
        let d = (r - 1) as f64;
        let (a, b, c, dd, e, f) = (1.3, -0.2, 4.0, 0.1, 0.9, -3.0);
        let grid = [(0.0, 0.0), (d, 0.0), (0.0, d), (d, d)];
        let mut pts = [(0.0, 0.0); 4];
        for (k, &(gx, gy)) in grid.iter().enumerate() {
            pts[k] = (a * gx + b * gy + c, dd * gx + e * gy + f);
        }
        let m = affine_grid_to_img(&pts, r);
        let got = [
            m[(0, 0)],
            m[(0, 1)],
            m[(0, 2)],
            m[(1, 0)],
            m[(1, 1)],
            m[(1, 2)],
        ];
        for (g, w) in got.iter().zip(&[a, b, c, dd, e, f]) {
            assert!((g - w).abs() < 1e-9, "{g} vs {w}");
        }
        let id = m.try_inverse().unwrap() * m;
        for i in 0..3 {
            for j in 0..3 {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((id[(i, j)] - want).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn pad_replicate_borders_and_center() {
        let rb = 3usize;
        let src: Vec<u32> = (0..9).collect();
        let guard = 2usize;
        let w = rb + 2 * guard;
        let p = pad_replicate(&src, rb, guard);
        // Interior maps back to the source; the border replicates the nearest edge.
        assert_eq!(p[guard * w + guard], src[0]); // (g,g) -> src(0,0)
        assert_eq!(p[(guard + 1) * w + (guard + 1)], src[rb + 1]); // -> src(1,1)
        assert_eq!(p[0], src[0]); // top-left corner replicates src(0,0)
        assert_eq!(p[w * w - 1], src[rb * rb - 1]); // bottom-right replicates src(2,2)
    }
}
