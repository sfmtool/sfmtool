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

use super::consensus::{consensus_phi, ConsensusScratch};
use super::level::LevelContext;
use super::params::{NormalRefineParams, Objective, ProjectedImage};
use super::support::repose_patch;
use super::znorm::znormalize_into;
use crate::camera::remap::remap_bilinear;
use crate::camera::WarpMap;
use crate::patch::cloud::OrientedPatch;
use crate::patch::keypoint_localize::{seed_offset, shifted_center};

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
/// as `u32`/pixel and replicate-padded by [`FRONTO_GUARD`], plus the **inverse**
/// of the affine grid→undistorted-normalized map that produced it (fit on the
/// *unpadded* `bw`×`bw` grid).
struct FrontoBase {
    /// Padded buffer, `bw_pad × bw_pad`.
    patch: Vec<u32>,
    /// Padded width `bw + 2·FRONTO_GUARD` (the buffer's row stride).
    bw_pad: u32,
    /// Inverse of the base's affine grid→undistorted-normalized map (homogeneous
    /// 3×3). Precomputed once per view: it is candidate-independent, so the
    /// per-candidate composition `a0⁻¹·ap` is a matmul, not an inverse.
    a0_inv: SMatrix<f64, 3, 3>,
    /// World-space offset of this view's surfel center from `base.center`,
    /// anchoring the base on the view's stored keypoint (zero when none). Computed
    /// once at the seed normal in [`prerender`]; [`eval_phi`] recenters each
    /// candidate patch by it so base and candidate stay registered. Holding it at
    /// the seed normal (rather than recomputing per candidate, as the exact path
    /// does) is a second-order approximation well inside the cache's resampling
    /// budget.
    center_offset: Vector3<f64>,
}

/// Per-view fronto bases, parallel to the caller's `views` slice (`None` where
/// the patch is back-facing in that view or the base could not be built).
pub(super) struct FrontoCache {
    bases: Vec<Option<FrontoBase>>,
}

/// Reusable per-candidate scratch buffers, owned by the caller for the whole
/// `coarse_to_fine` and threaded through [`eval_phi`]. The flat buffers keep
/// their capacity across the ~hundreds of candidate evaluations per patch, so
/// scoring a candidate allocates nothing after warm-up.
#[derive(Default)]
pub(super) struct Scratch {
    /// Flat raw support `[(view*CHANNELS + channel)*n + pixel]`, f32. Each view's
    /// resample writes straight into its slice — no separate planar buffer and no
    /// f32→f64 widening.
    raw: Vec<f32>,
    /// Flat z-normalized stack handed to `consensus_phi`.
    xs: Vec<f32>,
    /// Reused buffers for the consensus / IRLS reductions.
    cons: ConsensusScratch,
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
    view_keypoints: Option<&[Option<[f64; 2]>]>,
) -> Option<FrontoCache> {
    let ss = supersample.max(1.0);
    let rb = (((resolution as f64) * ss).round() as u32).max(resolution);
    let seed_patch = repose_patch(base, seed);
    let mut bases: Vec<Option<FrontoBase>> = Vec::with_capacity(views.len());
    for (vi, view) in views.iter().enumerate() {
        if !seed_patch.is_front_facing(view.cam_from_world) {
            bases.push(None);
            continue;
        }
        // Anchor on this view's stored keypoint (when given): the surfel center
        // for this view is the keypoint's ray ∩ the *seed* surfel plane, held fixed
        // across candidates. Render the fronto base — and later recenter every
        // candidate (`eval_phi`) — from that point so they stay registered.
        let center_offset = view_keypoints
            .and_then(|k| k[vi])
            .and_then(|kp| seed_offset(&seed_patch, view, kp, 1.0, 1.0))
            .map(|[au, av]| shifted_center(&seed_patch, au, av, 1.0, 1.0) - base.center)
            .unwrap_or_else(Vector3::zeros);
        let center_pt = base.center + center_offset;
        // Patch plane faces this camera (least foreshortening → most resolution).
        let cam_c = view.cam_from_world.inverse_translation_origin();
        let fronto_n = (cam_c - center_pt).normalize();
        let fp =
            OrientedPatch::from_center_normal(center_pt, fronto_n, base.u_axis, base.half_extent);
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
        // Invert the base affine once per view (None → skip this base, as a
        // degenerate map could not be used per-candidate anyway).
        let fb = corner_norm_pts(&fp, view, rb)
            .and_then(|pts| affine_grid_to_img(&pts, rb).try_inverse())
            .map(|a0_inv| FrontoBase {
                patch: pad_replicate(&packed, rb as usize, FRONTO_GUARD as usize),
                bw_pad: rb + 2 * FRONTO_GUARD,
                a0_inv,
                center_offset,
            });
        bases.push(fb);
    }
    // A non-invertible base (a degenerate, collinear fronto projection) is `None`
    // and so does not count here — it could not be resampled from anyway.
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
///
/// Runtime-dispatched to an AVX2 kernel (the `kdforest` pattern); the scalar
/// path is the reference and the `n % 8` AVX2 tail.
fn resample_support(
    base: &FrontoBase,
    phi: &[f64; 6],
    cols: &[i32],
    rows: &[i32],
    out: &mut [f32],
) {
    let n = cols.len();
    let p = [
        phi[0] as f32,
        phi[1] as f32,
        phi[2] as f32,
        phi[3] as f32,
        phi[4] as f32,
        phi[5] as f32,
    ];
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: guarded by the runtime feature check above.
            unsafe {
                return resample_support_avx2(base, &p, cols, rows, out);
            }
        }
    }
    resample_support_scalar(base, &p, cols, rows, 0, n, out);
}

/// Scalar reference resample of support pixels `[i0, i1)` (also the AVX2 tail).
fn resample_support_scalar(
    base: &FrontoBase,
    phi: &[f32; 6],
    cols: &[i32],
    rows: &[i32],
    i0: usize,
    i1: usize,
    out: &mut [f32],
) {
    let n = cols.len();
    let bw = base.bw_pad as usize;
    let g = FRONTO_GUARD as f32;
    let hi = (bw - 2) as i32; // x0/y0 ceiling so the +1 neighbour stays in-bounds
    for i in i0..i1 {
        let cf = cols[i] as f32;
        let rf = rows[i] as f32;
        let bx = phi[0] * cf + phi[1] * rf + phi[2] + g;
        let by = phi[3] * cf + phi[4] * rf + phi[5] + g;
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

/// AVX2 resample (8 support pixels per iteration). One packed `vpgatherdd` per
/// bilinear tap fetches all three channels of 8 pixels (4 gathers, not 12);
/// channels are unpacked in-register and written planar. The `n % 8` tail falls
/// back to the scalar reference. Produces the same result as
/// [`resample_support_scalar`] up to f32 rounding.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn resample_support_avx2(
    base: &FrontoBase,
    phi: &[f32; 6],
    cols: &[i32],
    rows: &[i32],
    out: &mut [f32],
) {
    use std::arch::x86_64::*;
    let n = cols.len();
    let bw = base.bw_pad as usize;
    let bptr = base.patch.as_ptr() as *const i32;
    let optr = out.as_mut_ptr();
    let one = _mm256_set1_ps(1.0);
    let gv = _mm256_set1_ps(FRONTO_GUARD as f32);
    let bw_i = _mm256_set1_epi32(bw as i32);
    let one_i = _mm256_set1_epi32(1);
    let zero_i = _mm256_setzero_si256();
    let hi_i = _mm256_set1_epi32((bw - 2) as i32);
    let mask = _mm256_set1_epi32(0xFF);
    let h: [__m256; 6] = std::array::from_fn(|i| _mm256_set1_ps(phi[i]));
    let mut i = 0usize;
    while i + 8 <= n {
        let colf = _mm256_cvtepi32_ps(_mm256_loadu_si256(cols.as_ptr().add(i) as *const __m256i));
        let rowf = _mm256_cvtepi32_ps(_mm256_loadu_si256(rows.as_ptr().add(i) as *const __m256i));
        // Affine + guard offset; no divide, no per-pixel float clamp.
        let bx = _mm256_add_ps(
            _mm256_fmadd_ps(h[0], colf, _mm256_fmadd_ps(h[1], rowf, h[2])),
            gv,
        );
        let by = _mm256_add_ps(
            _mm256_fmadd_ps(h[3], colf, _mm256_fmadd_ps(h[4], rowf, h[5])),
            gv,
        );
        // One branch-free integer clamp keeps the +1 neighbour in-bounds (and
        // makes any escaped/degenerate candidate memory-safe: cvtt maps
        // NaN/overflow to i32::MIN → clamps to 0).
        let x0i = _mm256_min_epi32(_mm256_max_epi32(_mm256_cvttps_epi32(bx), zero_i), hi_i);
        let y0i = _mm256_min_epi32(_mm256_max_epi32(_mm256_cvttps_epi32(by), zero_i), hi_i);
        let fx = _mm256_sub_ps(bx, _mm256_cvtepi32_ps(x0i));
        let fy = _mm256_sub_ps(by, _mm256_cvtepi32_ps(y0i));
        let row0 = _mm256_mullo_epi32(y0i, bw_i);
        let row1 = _mm256_mullo_epi32(_mm256_add_epi32(y0i, one_i), bw_i);
        let x1i = _mm256_add_epi32(x0i, one_i);
        let i00 = _mm256_add_epi32(row0, x0i);
        let i10 = _mm256_add_epi32(row0, x1i);
        let i01 = _mm256_add_epi32(row1, x0i);
        let i11 = _mm256_add_epi32(row1, x1i);
        let g00 = _mm256_i32gather_epi32::<4>(bptr, i00);
        let g10 = _mm256_i32gather_epi32::<4>(bptr, i10);
        let g01 = _mm256_i32gather_epi32::<4>(bptr, i01);
        let g11 = _mm256_i32gather_epi32::<4>(bptr, i11);
        let omfx = _mm256_sub_ps(one, fx);
        let omfy = _mm256_sub_ps(one, fy);
        let ch = |sh: i32| -> __m256 {
            let unpack = |g: __m256i| {
                let s = _mm256_srlv_epi32(g, _mm256_set1_epi32(sh));
                _mm256_cvtepi32_ps(_mm256_and_si256(s, mask))
            };
            let top = _mm256_fmadd_ps(unpack(g10), fx, _mm256_mul_ps(unpack(g00), omfx));
            let bot = _mm256_fmadd_ps(unpack(g11), fx, _mm256_mul_ps(unpack(g01), omfx));
            _mm256_fmadd_ps(bot, fy, _mm256_mul_ps(top, omfy))
        };
        _mm256_storeu_ps(optr.add(i), ch(0));
        _mm256_storeu_ps(optr.add(n + i), ch(8));
        _mm256_storeu_ps(optr.add(2 * n + i), ch(16));
        i += 8;
    }
    resample_support_scalar(base, phi, cols, rows, i, n, out);
}

/// `Φ` for `candidate_n` evaluated by affine-resampling the cached fronto bases
/// instead of re-rendering from the source images. `cols`/`rows` are the level's
/// frozen support grid coords and `sqrt_weights`/`total_weight` its window weights —
/// candidate-independent, so the caller computes them once per level. `scratch`
/// holds reused buffers so a candidate allocates nothing after warm-up. Mirrors
/// [`super::znorm::normalized_stack`] + [`consensus_phi`]; `None` if a candidate map
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
    sqrt_weights: &[f32],
    total_weight: f64,
    scratch: &mut Scratch,
    objective: Objective,
) -> Option<f64> {
    super::prof::count(&super::prof::N_EVAL, 1);
    if total_weight <= 0.0 {
        return None;
    }
    let cp = repose_patch(base, candidate_n);
    let n = cols.len();
    let vn = ctx.kept.len();
    let stride = CHANNELS * n;
    scratch.raw.resize(vn * stride, 0.0);

    // Flat raw `[(view*CHANNELS + channel)*n + pixel]`: each view's resample
    // writes its `CHANNELS*n` slice in the planar `[R|G|B]` layout directly.
    for (k, &vi) in ctx.kept.iter().enumerate() {
        let fb = cache.bases[vi].as_ref()?;
        let view = &views[vi];
        // Recenter the candidate to this view's keypoint-anchored center so it
        // registers with the base (which was rendered there). Zero offset → `cp`
        // unchanged, no allocation.
        let cp_anchored = (fb.center_offset.norm_squared() != 0.0).then(|| {
            let mut p = cp.clone();
            p.center += fb.center_offset;
            p
        });
        let cp_ref = cp_anchored.as_ref().unwrap_or(&cp);
        // Both operands are affine (last row [0,0,1]), so the composition is too;
        // the resampler only needs the 2×3 part (candidate-grid → base-grid). The
        // base inverse is precomputed, so this is a matmul, not a per-candidate
        // inverse.
        let phi = super::prof::CACHE_MAP.time(|| {
            let ap = corner_norm_pts(cp_ref, view, resolution)
                .map(|p| affine_grid_to_img(&p, resolution))?;
            let m = fb.a0_inv * ap;
            Some([
                m[(0, 0)],
                m[(0, 1)],
                m[(0, 2)],
                m[(1, 0)],
                m[(1, 1)],
                m[(1, 2)],
            ])
        })?;
        super::prof::CACHE_RESAMPLE.time(|| {
            resample_support(
                fb,
                &phi,
                cols,
                rows,
                &mut scratch.raw[k * stride..(k + 1) * stride],
            )
        });
    }

    // Same z-normalization + consensus as the source-render path, on the shared
    // flat f32 buffers.
    let kept = super::prof::CACHE_ZNORM.time(|| {
        znormalize_into(
            &scratch.raw[..vn * stride],
            vn,
            CHANNELS,
            n,
            &ctx.weights,
            total_weight,
            sqrt_weights,
            &mut scratch.xs,
        )
    })?;
    super::prof::CACHE_CONSENSUS
        .time(|| consensus_phi(&scratch.xs, vn, kept, n, objective, &mut scratch.cons))
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

    /// A synthetic padded base + scattered support, for the kernel tests/bench.
    fn synthetic_base(rb: usize) -> FrontoBase {
        let guard = FRONTO_GUARD as usize;
        let bw = rb + 2 * guard;
        let patch: Vec<u32> = (0..bw * bw)
            .map(|k| {
                let r = (k * 7) % 256;
                let g = (k * 13 + 5) % 256;
                let b = (k * 29 + 11) % 256;
                (r | (g << 8) | (b << 16)) as u32
            })
            .collect();
        FrontoBase {
            patch,
            bw_pad: bw as u32,
            a0_inv: SMatrix::identity(),
            center_offset: Vector3::zeros(),
        }
    }

    fn synthetic_support(rb: usize) -> (Vec<i32>, Vec<i32>) {
        // 173 support pixels (not a multiple of 8, so the AVX2 tail is exercised).
        let mut cols = Vec::new();
        let mut rows = Vec::new();
        for k in 0..173 {
            cols.push((k % rb) as i32);
            rows.push((k * 3 % rb) as i32);
        }
        (cols, rows)
    }

    #[test]
    fn resample_avx2_matches_scalar() {
        let rb = 24usize;
        let base = synthetic_base(rb);
        let (cols, rows) = synthetic_support(rb);
        let n = cols.len();
        // A few affine maps incl. one that pushes some pixels off the base edge
        // (exercises the clamp); the dispatcher picks AVX2 where available.
        for phi in [
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [1.07, 0.04, 1.5, -0.03, 0.96, -0.8],
            [1.6, 0.2, 6.0, -0.25, 1.55, -5.0],
        ] {
            let p32 = [
                phi[0] as f32,
                phi[1] as f32,
                phi[2] as f32,
                phi[3] as f32,
                phi[4] as f32,
                phi[5] as f32,
            ];
            let mut want = vec![0f32; n * 3];
            resample_support_scalar(&base, &p32, &cols, &rows, 0, n, &mut want);
            let mut got = vec![0f32; n * 3];
            resample_support(&base, &phi, &cols, &rows, &mut got);
            for (a, b) in want.iter().zip(&got) {
                assert!((a - b).abs() < 5e-3, "{a} vs {b}");
            }
        }
    }

    /// On-demand micro-benchmark: `cargo test -p sfmtool-core resample_bench
    /// -- --ignored --nocapture`. Times the dispatched (AVX2) kernel vs the
    /// scalar reference on a 32×32-ish support.
    #[test]
    #[ignore]
    fn resample_bench() {
        use std::time::Instant;
        let rb = 64usize;
        let base = synthetic_base(rb);
        let mut cols: Vec<i32> = Vec::new();
        let mut rows: Vec<i32> = Vec::new();
        for k in 0..812 {
            cols.push(k % 32);
            rows.push(k / 32 % 32);
        }
        let n = cols.len();
        let phi = [1.05, 0.03, 2.0, -0.02, 0.98, 1.5];
        let p32 = [
            phi[0] as f32,
            phi[1] as f32,
            phi[2] as f32,
            phi[3] as f32,
            phi[4] as f32,
            phi[5] as f32,
        ];
        let iters = 200_000;
        let mut out = vec![0f32; n * 3];
        let t = Instant::now();
        for _ in 0..iters {
            resample_support_scalar(&base, &p32, &cols, &rows, 0, n, &mut out);
            std::hint::black_box(&out);
        }
        let scalar = t.elapsed().as_nanos() as f64 / iters as f64;
        let t = Instant::now();
        for _ in 0..iters {
            resample_support(&base, &phi, &cols, &rows, &mut out);
            std::hint::black_box(&out);
        }
        let dispatch = t.elapsed().as_nanos() as f64 / iters as f64;
        println!(
            "resample {n} support px: scalar {scalar:.0} ns, dispatch {dispatch:.0} ns ({:.2}x)",
            scalar / dispatch
        );
    }
}
