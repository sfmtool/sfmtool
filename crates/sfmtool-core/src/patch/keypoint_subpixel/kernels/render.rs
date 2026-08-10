// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Rendering and the render-once tile for subpixel keypoint refinement.
//!
//! Direct projective renders ([`render_core`] / [`render_core_with_jg`], the
//! out-of-tile fallback), the [`RefineTile`] prerender and its
//! cubic-B-spline reads ([`render_refine_tile`] / [`try_render_refine_tile`] /
//! [`core_value`] / [`core_value_with_jg`]), and the coarse-grid gate
//! ([`grid_to_source_scale`] / [`TILE_MAX_GRID_TO_SOURCE`]).

use crate::camera::remap::{
    remap_aniso_with_grad_into, remap_aniso_with_pyramid, remap_bilinear, remap_bilinear_mip,
    remap_bilinear_mip_with_grad_into, remap_bilinear_with_grad_into, ImageF32WithGrad,
};
use crate::camera::WarpMap;
use crate::patch::cloud::OrientedPatch;
use crate::patch::keypoint_localize::shifted_center;
use crate::patch::normal_refine::{ProjectedImage, Sampler, Support};

use crate::patch::keypoint_subpixel::prof;

/// `remap_aniso` sample cap along the major axis (mirrors `normal_refine`).
const MAX_ANISOTROPY: u32 = 16;

/// Render one view's `R×R` core at in-plane offset `(au, av)` (patch-grid px) into
/// `out` (flat `[channel * n + support_index]`), reading only the window-support
/// pixels. Returns `false` (leaving `out` untouched) when any support pixel is out
/// of frame — a δ whose core left the frame is invalid and can't be scored.
///
/// Direct (full projective render) value path. In the solver this is only the
/// **fallback** for an offset outside the [`RefineTile`]'s coverage — the hot
/// path reads the pair's render-once tile ([`core_value`]). The GN
/// normal-equations fallback is [`render_core_with_jg`].
#[allow(clippy::too_many_arguments)]
pub(in crate::patch::keypoint_subpixel) fn render_core(
    patch: &OrientedPatch,
    view: &ProjectedImage<'_>,
    au: f64,
    av: f64,
    wpp_u: f64,
    wpp_v: f64,
    resolution: u32,
    sampler: Sampler,
    support: &Support,
    channels: usize,
    out: &mut [f32],
) -> bool {
    prof::RENDER_VALUE.time(|| {
        let center = shifted_center(patch, au, av, wpp_u, wpp_v);
        let mut core_patch = OrientedPatch::from_center_normal(
            center,
            patch.normal(),
            patch.v_axis,
            patch.half_extent,
        );
        // Preserve the homogeneous weight so a point at infinity renders as a
        // direction patch (corners are directions), not a finite surfel.
        core_patch.w = patch.w;
        let mut map =
            WarpMap::from_patch(&core_patch, view.camera, view.cam_from_world, resolution);
        let img = match sampler {
            Sampler::Anisotropic => {
                map.compute_svd();
                remap_aniso_with_pyramid(view.pyramid, &map, MAX_ANISOTROPY)
            }
            Sampler::BilinearMip => {
                map.compute_svd();
                remap_bilinear_mip(view.pyramid, &map)
            }
            Sampler::Bilinear => remap_bilinear(view.pyramid.level(0), &map),
        };
        let n = support.pixels.len();
        for (k, &p) in support.pixels.iter().enumerate() {
            let col = (p % resolution as usize) as u32;
            let row = (p / resolution as usize) as u32;
            if !map.is_valid(col, row) {
                return false;
            }
            for c in 0..channels {
                out[c * n + k] = img.get_pixel(col, row, c as u32) as f32;
            }
        }
        true
    })
}

/// Render one view's `R×R` core at offset `(au, av)` and also fill the analytic
/// image Jacobian `∂I/∂δ` in patch-grid coords per support pixel and channel
/// — one render that returns value + gradient (instead of the previous 5×
/// finite-difference pattern). Returns `false` (leaving outputs untouched) when
/// any support pixel is out of frame. Like [`render_core`], this is now the
/// out-of-tile-coverage **fallback** ([`core_value_with_jg`] reads the tile's
/// pre-composed gradient planes on the hot path); it also defines the
/// value+gradient convention the tile prerender stores.
///
/// Per pixel the sampler returns `(I, ∂I/∂x, ∂I/∂y)` in **source-pixel** coords;
/// composing with the warp Jacobian `J = ∂(source)/∂(grid)` gives `∂I/∂δ` in
/// **patch-grid** coords (`δ = (δ_col, δ_row)`):
///
/// ```text
/// Jg_u = J[0][0] · dI_dx + J[1][0] · dI_dy   (column = u axis)
/// Jg_v = J[0][1] · dI_dx + J[1][1] · dI_dy   (row    = v axis)
/// ```
///
/// where `J[0][0] = dx/dcol`, `J[0][1] = dx/drow`, `J[1][0] = dy/dcol`,
/// `J[1][1] = dy/drow` (the convention `WarpMap::get_jacobian` stores).
#[allow(clippy::too_many_arguments)]
pub(in crate::patch::keypoint_subpixel) fn render_core_with_jg(
    patch: &OrientedPatch,
    view: &ProjectedImage<'_>,
    au: f64,
    av: f64,
    wpp_u: f64,
    wpp_v: f64,
    resolution: u32,
    sampler: Sampler,
    support: &Support,
    channels: usize,
    g: &mut [f32],
    jg_u: &mut [f32],
    jg_v: &mut [f32],
    img_scratch: &mut ImageF32WithGrad,
) -> bool {
    prof::RENDER_GRAD.time(|| {
        let center = shifted_center(patch, au, av, wpp_u, wpp_v);
        let mut core_patch = OrientedPatch::from_center_normal(
            center,
            patch.normal(),
            patch.v_axis,
            patch.half_extent,
        );
        core_patch.w = patch.w;
        let mut map =
            WarpMap::from_patch(&core_patch, view.camera, view.cam_from_world, resolution);
        match sampler {
            Sampler::Anisotropic => {
                map.compute_svd(); // also populates jacobians as a by-product
                remap_aniso_with_grad_into(view.pyramid, &map, MAX_ANISOTROPY, img_scratch);
            }
            Sampler::BilinearMip => {
                map.compute_svd(); // also populates jacobians as a by-product
                remap_bilinear_mip_with_grad_into(view.pyramid, &map, img_scratch);
            }
            Sampler::Bilinear => {
                map.compute_jacobians();
                remap_bilinear_with_grad_into(view.pyramid.level(0), &map, img_scratch);
            }
        };
        let n = support.pixels.len();
        let stride = img_scratch.width() as usize * channels;
        let value = img_scratch.value();
        let grad_x = img_scratch.grad_x();
        let grad_y = img_scratch.grad_y();
        for (k, &p) in support.pixels.iter().enumerate() {
            let col = (p % resolution as usize) as u32;
            let row = (p / resolution as usize) as u32;
            if !map.is_valid(col, row) {
                return false;
            }
            let j = map.get_jacobian(col, row);
            let row_off = row as usize * stride + col as usize * channels;
            for c in 0..channels {
                let idx = row_off + c;
                let v = value[idx];
                let gx = grad_x[idx];
                let gy = grad_y[idx];
                g[c * n + k] = v;
                jg_u[c * n + k] = j[0][0] * gx + j[1][0] * gy;
                jg_v[c * n + k] = j[0][1] * gx + j[1][1] * gy;
            }
        }
        true
    })
}

/// A **render-once context tile** for one (point, view) pair — the refiner's
/// counterpart of the localizer's `ContextTile` (see
/// `specs/core/keypoint-localization-search-cache.md`), adapted to the GN
/// solve's **fractional** reads. The patch frame is fixed during refinement and
/// only the 2-DOF in-plane offset `δ` moves, so every render of the pair is the
/// same patch→image map at a slightly different sub-pixel shift. This tile is
/// that map, rendered once over an expanded patch-grid-aligned grid (`R + 2·pad`
/// per side, one tile texel = one patch-grid pixel in world units, centred at
/// the view's **seed** offset); a core render at offset `(au, av)` then becomes
/// a continuous **cubic-spline read** of the tile at the (shared-per-call)
/// fractional tile shift `(au, av) − seed + pad`. At an exactly-integer shift
/// the read reproduces the tile texels (the cardinal spline interpolates its
/// samples exactly) and hence a direct render, up to the direct path's `u8`
/// output rounding (the tile keeps the sampler's unquantized `f32`) and the
/// prefilter's f32 round-trip.
///
/// **Interpolator choice (load-bearing for quality).** The planes are stored
/// as **cubic B-spline coefficients** (Unser's two-pass IIR prefilter at
/// prerender) and reads evaluate the cardinal cubic spline with the 4×4
/// B-spline kernel. Cheaper interpolators displace the ECC optimum measurably
/// on high-frequency content because their amplitude response varies with the
/// sub-texel phase: bilinear reads (phase-dependent smoothing, maximal at
/// half-integer shifts) planted up to ~0.065 px recovery error on the
/// synthetic fixtures, and Catmull-Rom still left ~0.02–0.045 px on the
/// near-Nyquist ones; the prefiltered spline's near-flat passband brings the
/// recovery back inside the spec's < 0.02 px target at the same 4×4 read
/// cost. A 2× supersampled bilinear tile was the measured-and-rejected
/// alternative — it quadruples the prerender, which is the new dominant cost.
///
/// **Gradients.** The tile stores the pre-composed patch-grid image Jacobian
/// `Jg = ∇_src I · J` per texel — the sampler's analytic value+gradient render
/// (`remap_*_with_grad_into`, LOD-consistent with the value) composed with the
/// warp Jacobian `WarpMap::get_jacobian`, exactly as [`render_core_with_jg`]
/// does — and the GN read bilinearly interpolates those planes. This is chosen
/// over central differences of the tile's value plane: the stored planes keep
/// the analytic, LOD-consistent character of the direct path (differencing the
/// already-bilinear value field would double-smooth and make the gradient
/// piecewise-constant per texel cell), and the interior texels use central
/// differences of the warp where the direct path's `R×R` map falls back to
/// one-sided differences on its boundary ring. The interpolated `Jg` is not the
/// exact derivative of the interpolated value field; that mismatch only
/// perturbs the GN step direction — acceptance stays guarded by the ECC line
/// search on the value reads, so the never-worse-than-seed guarantee is
/// untouched.
///
/// **Coverage / sizing.** `pad = ⌈max_offset_px⌉ + 2` patch-grid px, so every
/// offset the line search can accept (`|δ − δ_seed| ≤ max_offset_px`, and the
/// tile is centred at the seed) reads in-bounds, including the cubic kernel's
/// `±1/+2` tap ring. A read outside the coverage returns `None` and the caller
/// falls back to a direct render (counted in [`prof::N_TILE_FALLBACK`];
/// expected ~0).
///
/// **Coarse-grid gate (no tile).** A tile is built only when the patch grid
/// samples at least as densely as the source
/// (`grid_to_source_scale ≤ ~1.2` source px per grid px —
/// [`TILE_MAX_GRID_TO_SOURCE`]). A *coarser* grid (large-scale keypoints)
/// would freeze the sampling phase of above-grid-Nyquist source content that
/// the direct path samples continuously — measured as spurious sub-pixel
/// displacement (up to ~0.15 px) on a coarse-grid near-Nyquist synthetic
/// fixture — and a source-density supersampled tile was measured on dino to
/// cost *more* than direct rendering (a 3× tile is a ~9× prerender vs ~14
/// direct renders per pair). So coarse-grid views simply keep the exact
/// direct-render path ([`try_render_refine_tile`] returns `None`, counted in
/// [`prof::N_TILE_SKIPPED`]); on dino that is ~1/3 of the (point, view)
/// pairs.
///
/// **Validity.** An exactly-integer read (the seed) gates on the texels it
/// reproduces, matching the direct path's gate; a fractional read requires the
/// full 4×4 tap neighbourhood ([`valid4`](Self::valid4), eroded once at
/// prerender), which is slightly conservative at frame edges (an
/// edge-straddling candidate scores as out-of-frame where a direct fractional
/// render might still squeak by — the line search then simply keeps the
/// previous offset). Out-of-frame texels are rendered black before the
/// prefilter, whose IIR spreads a small (`|√3−2|^d`-decaying) imprint of that
/// step a few texels into the valid region — only patches straddling the
/// frame edge see it, and their fractional reads are already gated by
/// `valid4`.
///
/// **Accepted loss.** Double interpolation — the tile render resamples the
/// source once, the read cubically resamples the tile — can still move
/// converged keypoints by small fractions of a px vs the direct path
/// (measured on dino: see the spec's status note).
pub(in crate::patch::keypoint_subpixel) struct RefineTile {
    /// Tile side in texels (`resolution + 2·pad`).
    pub(in crate::patch::keypoint_subpixel) res: usize,
    /// Border margin in patch-grid px (`⌈max_offset_px⌉ + 2`).
    pub(in crate::patch::keypoint_subpixel) pad: usize,
    /// Channel count.
    pub(in crate::patch::keypoint_subpixel) channels: usize,
    /// The seed offset `(au, av)` (patch-grid px) the tile is centred at.
    pub(in crate::patch::keypoint_subpixel) seed: [f64; 2],
    /// Sampler values as **cubic B-spline coefficients** (prefiltered in
    /// place), **planar per channel**: `[c · res² + row · res + col]`.
    /// Planar (not interleaved) is load-bearing for read speed: the support's
    /// row runs then read contiguous texel segments, so each of the 16 tap
    /// accumulations in [`read_planes`](Self::read_planes) is a vectorizable
    /// contiguous SAXPY. Evaluating the 4×4 B-spline kernel on these
    /// reproduces the sampler's unquantized `f32` values exactly at texels and
    /// the cardinal spline in between.
    pub(in crate::patch::keypoint_subpixel) value: Vec<f32>,
    /// Pre-composed patch-grid image Jacobian `∂I/∂δ_col` per texel/channel
    /// (B-spline coefficients, same layout and prefilter as
    /// [`value`](Self::value)).
    pub(in crate::patch::keypoint_subpixel) jg_u: Vec<f32>,
    /// `∂I/∂δ_row` per texel/channel, same layout.
    pub(in crate::patch::keypoint_subpixel) jg_v: Vec<f32>,
    /// Per-texel validity (`true` in frame), `row · res + col`.
    pub(in crate::patch::keypoint_subpixel) valid: Vec<bool>,
    /// Eroded validity for fractional reads: `true` where the whole 4×4
    /// spline tap neighbourhood `[row−1, row+2] × [col−1, col+2]` is in
    /// frame (and inside the tile). Indexed by the read's base texel.
    pub(in crate::patch::keypoint_subpixel) valid4: Vec<bool>,
}

impl RefineTile {
    /// Resolve a core read at offset `(au, av)`: the integer tile origin of the
    /// core's `(0, 0)` pixel plus the shared in-cell fractions. `None` when the
    /// offset falls outside the tile's coverage (caller falls back to a direct
    /// render).
    fn read_geometry(
        &self,
        au: f64,
        av: f64,
        resolution: usize,
    ) -> Option<(usize, usize, f32, f32)> {
        let ou = self.pad as f64 + (au - self.seed[0]);
        let ov = self.pad as f64 + (av - self.seed[1]);
        let (iu, iv) = (ou.floor(), ov.floor());
        // The cubic kernel taps `base − 1 ..= base + 2` per axis; the widest
        // read touches texel `resolution − 1 + iu + 2`.
        if iu < 1.0 || iv < 1.0 {
            return None;
        }
        let (fu, fv) = ((ou - iu) as f32, (ov - iv) as f32);
        let (iu, iv) = (iu as usize, iv as usize);
        if resolution + iu + 1 > self.res - 1 || resolution + iv + 1 > self.res - 1 {
            return None;
        }
        Some((iu, iv, fu, fv))
    }

    /// Cubic-read the `R×R` core at offset `(au, av)` into `out`
    /// (`[channel * n + support_index]`) — the tile counterpart of
    /// [`render_core`]. Returns `None` when the offset is outside the tile's
    /// coverage (fall back to a direct render); `Some(false)` when a needed
    /// tile texel is out of frame (the core can't be scored there, matching the
    /// direct path's validity gate; an exactly-integer read requires only the
    /// texels it reproduces).
    pub(in crate::patch::keypoint_subpixel) fn read_core(
        &self,
        au: f64,
        av: f64,
        resolution: usize,
        support: &Support,
        out: &mut [f32],
    ) -> Option<bool> {
        self.read_planes(au, av, resolution, support, &[&self.value], &mut [out])
    }

    /// Like [`read_core`](Self::read_core), but reading the pre-composed
    /// patch-grid image Jacobian planes into `jg_u` / `jg_v` — the tile
    /// counterpart of [`render_core_with_jg`]'s gradient half. The **value**
    /// core is *not* re-read: the GN loop always holds the value core at the
    /// current offset from its preceding score evaluation (see
    /// [`refine_one_view`](crate::patch::keypoint_subpixel::refine_one_view)'s invariant note).
    pub(in crate::patch::keypoint_subpixel) fn read_jg(
        &self,
        au: f64,
        av: f64,
        resolution: usize,
        support: &Support,
        jg_u: &mut [f32],
        jg_v: &mut [f32],
    ) -> Option<bool> {
        self.read_planes(
            au,
            av,
            resolution,
            support,
            &[&self.jg_u, &self.jg_v],
            &mut [jg_u, jg_v],
        )
    }

    /// Shared read body: cubic-blend each of `planes` (value, or the two
    /// Jacobian planes) into the parallel `outs` buffers over the support.
    ///
    /// The support is walked as **row runs** (maximal spans of consecutive
    /// pixels within one core row — the disk window is a stack of such runs):
    /// for a run, each of the 16 spline taps contributes one **contiguous**
    /// texel segment of a planar channel, so the inner loop is a plain
    /// vectorizable segment multiply-accumulate, and the per-channel output
    /// span `dst[c·n + k .. + len]` is written contiguously too.
    fn read_planes(
        &self,
        au: f64,
        av: f64,
        resolution: usize,
        support: &Support,
        planes: &[&[f32]],
        outs: &mut [&mut [f32]],
    ) -> Option<bool> {
        debug_assert_eq!(planes.len(), outs.len());
        let (iu, iv, fu, fv) = self.read_geometry(au, av, resolution)?;
        let n = support.pixels.len();
        // Integer shifts reproduce texel values exactly (cardinal spline), so
        // they gate on the texel itself — the direct path's validity
        // semantics; fractional reads gate on the full tap neighbourhood.
        let exact = fu == 0.0 && fv == 0.0;
        let gate = if exact { &self.valid } else { &self.valid4 };
        let wu = bspline3_weights(fu);
        let wv = bspline3_weights(fv);
        let mut w16 = [0.0f32; 16];
        for j in 0..4 {
            for i in 0..4 {
                w16[j * 4 + i] = wv[j] * wu[i];
            }
        }
        let ch = self.channels;
        let res = self.res;
        let area = res * res;
        let pixels = &support.pixels;
        let mut k = 0usize;
        while k < n {
            let p = pixels[k];
            let row = p / resolution;
            // Extend the run: consecutive support pixels along this core row.
            let mut len = 1usize;
            while k + len < n && pixels[k + len] == p + len && (p + len) / resolution == row {
                len += 1;
            }
            let trow = row + iv; // texel row of the run's base
            let tcol = p % resolution + iu; // texel col of the run's first pixel
            let gbase = trow * res + tcol;
            if gate[gbase..gbase + len].iter().any(|&v| !v) {
                return Some(false);
            }
            for (plane, dst) in planes.iter().zip(outs.iter_mut()) {
                for c in 0..ch {
                    let seg = &mut dst[c * n + k..][..len];
                    let pbase = c * area + (trow - 1) * res + (tcol - 1);
                    for (j, wrow) in w16.chunks_exact(4).enumerate() {
                        // One pass per tap row: the four horizontal taps are
                        // overlapping shifted reads of the same contiguous
                        // texel row (`windows(4)`), fused into a single
                        // multiply-accumulate over the run.
                        let src = &plane[pbase + j * res..][..len + 3];
                        if j == 0 {
                            for (s, w4) in seg.iter_mut().zip(src.windows(4)) {
                                *s = wrow[0] * w4[0]
                                    + wrow[1] * w4[1]
                                    + wrow[2] * w4[2]
                                    + wrow[3] * w4[3];
                            }
                        } else {
                            for (s, w4) in seg.iter_mut().zip(src.windows(4)) {
                                *s += wrow[0] * w4[0]
                                    + wrow[1] * w4[1]
                                    + wrow[2] * w4[2]
                                    + wrow[3] * w4[3];
                            }
                        }
                    }
                }
            }
            k += len;
        }
        Some(true)
    }
}

/// Cubic B-spline basis weights `β³` for the four taps `−1, 0, +1, +2` at
/// in-cell fraction `f ∈ [0, 1)`. Applied to **prefiltered coefficients**
/// this evaluates the cardinal (interpolating) cubic spline; `f = 0` yields
/// `(1/6, 4/6, 1/6, 0)`, which reproduces the original sample exactly.
#[inline]
fn bspline3_weights(f: f32) -> [f32; 4] {
    let f2 = f * f;
    let f3 = f2 * f;
    let omf = 1.0 - f;
    [
        omf * omf * omf / 6.0,
        (4.0 - 6.0 * f2 + 3.0 * f3) / 6.0,
        (1.0 + 3.0 * (f + f2 - f3)) / 6.0,
        f3 / 6.0,
    ]
}

/// In-place cubic-B-spline prefilter of one line (Unser's two-pass IIR,
/// mirror boundary, horizon-truncated causal init): turns samples into the
/// interpolation coefficients the cardinal spline evaluates. `f64` throughout
/// (the caller round-trips through the `f32` planes once).
fn bspline3_prefilter_line(line: &mut [f64]) {
    const Z1: f64 = -0.267_949_192_431_122_7; // √3 − 2
    let n = line.len();
    if n < 2 {
        return;
    }
    // Causal init: c⁺(0) = Σ_k z₁ᵏ s(k), truncated once z₁ᵏ < 1e-16.
    let horizon = n.min(28);
    let mut sum = line[0];
    let mut zk = Z1;
    for &s in line.iter().take(horizon).skip(1) {
        sum += zk * s;
        zk *= Z1;
    }
    line[0] = sum;
    for k in 1..n {
        line[k] += Z1 * line[k - 1];
    }
    // Anticausal init + pass, then the (1 − z₁)(1 − 1/z₁) = 6 gain.
    line[n - 1] = (Z1 / (Z1 * Z1 - 1.0)) * (line[n - 1] + Z1 * line[n - 2]);
    for k in (0..n - 1).rev() {
        line[k] = Z1 * (line[k + 1] - line[k]);
    }
    for v in line.iter_mut() {
        *v *= 6.0;
    }
}

/// Prefilter one planar tile plane (rows then columns, per channel) into
/// B-spline coefficients, in place.
fn bspline3_prefilter_plane(plane: &mut [f32], res: usize, channels: usize) {
    let mut line = vec![0f64; res];
    let area = res * res;
    for c in 0..channels {
        let pl = &mut plane[c * area..][..area];
        for row in 0..res {
            let prow = &mut pl[row * res..][..res];
            for (slot, &v) in line.iter_mut().zip(prow.iter()) {
                *slot = v as f64;
            }
            bspline3_prefilter_line(&mut line);
            for (dst, &v) in prow.iter_mut().zip(line.iter()) {
                *dst = v as f32;
            }
        }
        for col in 0..res {
            for (row, slot) in line.iter_mut().enumerate() {
                *slot = pl[col + row * res] as f64;
            }
            bspline3_prefilter_line(&mut line);
            for (row, &v) in line.iter().enumerate() {
                pl[col + row * res] = v as f32;
            }
        }
    }
}

/// A [`RefineTile`] is only built for a view whose patch grid samples at least
/// this densely relative to the source ([`grid_to_source_scale`] ≤ this many
/// source px per grid px). Coarser views keep the exact direct-render path —
/// see the "Coarse-grid gate" note on [`RefineTile`]. The 1.2 slack over 1.0
/// tolerates mild undersampling (the top of the source band, already
/// attenuated by the sampler's bilinear footprint) before the measured
/// aliasing displacement matters.
pub(in crate::patch::keypoint_subpixel) const TILE_MAX_GRID_TO_SOURCE: f64 = 1.2;

/// Estimate the patch-grid → source scale (source px per grid px, the larger
/// of the two axes) from the projected core corners — the [`RefineTile`]
/// coarse-grid gate. Uses the same affine ray basis as `WarpMap::from_patch`
/// (so `w = 0` folds in identically); `None` when a corner fails to project
/// (grazing / behind camera — the view then keeps the direct path, which
/// handles partial visibility exactly).
///
/// **An approximation, and a model-conditional one.** This is a SECANT over the
/// whole core — two edge chords from one corner — not the local scale anywhere
/// in particular. It therefore reports the patch's average stretch and is blind
/// to how that stretch varies across the grid. Under the perspective family the
/// variation is small over a patch; under the equidistant map the radial and
/// azimuthal scales differ by `θ/sin θ` (1.57 at 90°, 2.42 at 120°) and the
/// azimuthal one grows across a peripheral patch, so the secant can sit either
/// side of the true local maximum. The consequence is bounded: the value only
/// chooses tile-vs-direct render, the direct path is exact, and a
/// mis-classified view renders correctly either way — at worst it aliases
/// slightly (tile taken when it should not have been) or costs the tile's
/// speed-up (direct taken when the tile would have done).
///
/// The model-correct measure is the warp map's per-pixel `σ_major`
/// (`WarpMap::compute_svd`), which is **not** a drop-in here: no `WarpMap`
/// exists at this call site. The first one on this path is built inside
/// `render_refine_tile`, i.e. after the gate has already decided, and building
/// one to decide would cost a full `(resolution + 2·pad)²` per-pixel projection
/// where this costs three `ray_to_pixel` calls — inverting the point of a gate
/// whose job is to SKIP the tile. `view_selection` faces the same trade and
/// resolves it the same way, with its own closed-form `affine_sigma_major`.
/// Deferred deliberately; if a cheap per-view curvature bound is ever wanted
/// here, the precedent is `view_selection`'s fourth-corner residual (one more
/// projection), not the SVD.
pub(in crate::patch::keypoint_subpixel) fn grid_to_source_scale(
    patch: &OrientedPatch,
    view: &ProjectedImage<'_>,
    resolution: u32,
) -> Option<f64> {
    let rot = view.cam_from_world.rotation.to_rotation_matrix();
    let q0 = rot * patch.center.coords + view.cam_from_world.translation * patch.w;
    let qu = (rot * patch.u_axis) * patch.half_extent[0];
    let qv = (rot * patch.v_axis) * (-patch.half_extent[1]);
    let step = 2.0 / resolution as f64;
    let s0 = 0.5 * step - 1.0;
    let origin = q0 + (qu + qv) * s0;
    let cs = qu * step;
    let rs = qv * step;
    let r1 = (resolution.max(2) - 1) as f64;
    let project_corner = |c: f64, r: f64| -> Option<(f64, f64)> {
        let ray = origin + cs * c + rs * r;
        view.camera.ray_to_pixel([ray.x, ray.y, ray.z])
    };
    let p00 = project_corner(0.0, 0.0)?;
    let p10 = project_corner(r1, 0.0)?;
    let p01 = project_corner(0.0, r1)?;
    let du = (p10.0 - p00.0).hypot(p10.1 - p00.1) / r1;
    let dv = (p01.0 - p00.0).hypot(p01.1 - p00.1) / r1;
    Some(du.max(dv))
}

/// [`render_refine_tile`] behind the coarse-grid gate: `None` (no tile — the
/// view keeps the exact direct-render path, counted in
/// [`prof::N_TILE_SKIPPED`]) when the patch grid is coarser than
/// [`TILE_MAX_GRID_TO_SOURCE`] source px per grid px, or a core corner fails
/// to project.
#[allow(clippy::too_many_arguments)]
pub(in crate::patch::keypoint_subpixel) fn try_render_refine_tile(
    patch: &OrientedPatch,
    view: &ProjectedImage<'_>,
    seed: [f64; 2],
    wpp_u: f64,
    wpp_v: f64,
    resolution: u32,
    pad: u32,
    sampler: Sampler,
    img: &mut ImageF32WithGrad,
) -> Option<RefineTile> {
    match grid_to_source_scale(patch, view, resolution) {
        Some(s) if s <= TILE_MAX_GRID_TO_SOURCE => Some(render_refine_tile(
            patch, view, seed, wpp_u, wpp_v, resolution, pad, sampler, img,
        )),
        _ => {
            prof::count(&prof::N_TILE_SKIPPED, 1);
            None
        }
    }
}

/// Render one view's [`RefineTile`]: the expanded patch-grid-aligned tile
/// centred at the view's `seed` offset, rendered with the value+gradient
/// variant of `sampler`, composed with the warp Jacobian into the stored
/// patch-grid gradient planes, and prefiltered into B-spline coefficients.
/// `img` is a reused scratch for the value+gradient render. Production goes
/// through [`try_render_refine_tile`] (the coarse-grid gate); this
/// unconditional entry is what the unit tests drive directly.
#[allow(clippy::too_many_arguments)]
pub(in crate::patch::keypoint_subpixel) fn render_refine_tile(
    patch: &OrientedPatch,
    view: &ProjectedImage<'_>,
    seed: [f64; 2],
    wpp_u: f64,
    wpp_v: f64,
    resolution: u32,
    pad: u32,
    sampler: Sampler,
    img: &mut ImageF32WithGrad,
) -> RefineTile {
    prof::TILE_PRERENDER.time(|| {
        let tile_res = resolution + 2 * pad;
        let center = shifted_center(patch, seed[0], seed[1], wpp_u, wpp_v);
        let ext = tile_res as f64 / resolution as f64;
        let mut tile_patch = OrientedPatch::from_center_normal(
            center,
            patch.normal(),
            patch.v_axis,
            [patch.half_extent[0] * ext, patch.half_extent[1] * ext],
        );
        // Preserve the homogeneous weight so a point at infinity renders as a
        // direction patch, exactly as the direct render paths do.
        tile_patch.w = patch.w;
        let mut map = WarpMap::from_patch(&tile_patch, view.camera, view.cam_from_world, tile_res);
        match sampler {
            Sampler::Anisotropic => {
                map.compute_svd(); // also populates jacobians as a by-product
                remap_aniso_with_grad_into(view.pyramid, &map, MAX_ANISOTROPY, img);
            }
            Sampler::BilinearMip => {
                map.compute_svd(); // also populates jacobians as a by-product
                remap_bilinear_mip_with_grad_into(view.pyramid, &map, img);
            }
            Sampler::Bilinear => {
                map.compute_jacobians();
                remap_bilinear_with_grad_into(view.pyramid.level(0), &map, img);
            }
        }
        let t = tile_res as usize;
        let ch = img.channels() as usize;
        let area = t * t;
        // De-interleave the render into the planar-per-channel tile layout
        // (see the `RefineTile::value` doc), composing the gradient planes
        // with the per-texel warp Jacobian (`Jg = ∇_src I · J`, the same
        // convention as `render_core_with_jg`) on the way.
        let mut value = vec![0f32; ch * area];
        let mut jg_u = vec![0f32; ch * area];
        let mut jg_v = vec![0f32; ch * area];
        let mut valid = vec![false; area];
        let vsrc = img.value();
        let grad_x = img.grad_x();
        let grad_y = img.grad_y();
        for row in 0..tile_res {
            for col in 0..tile_res {
                let p = row as usize * t + col as usize;
                valid[p] = map.is_valid(col, row);
                let j = map.get_jacobian(col, row);
                let base = p * ch;
                for c in 0..ch {
                    let gx = grad_x[base + c];
                    let gy = grad_y[base + c];
                    value[c * area + p] = vsrc[base + c];
                    jg_u[c * area + p] = j[0][0] * gx + j[1][0] * gy;
                    jg_v[c * area + p] = j[0][1] * gx + j[1][1] * gy;
                }
            }
        }
        // Turn the planes into B-spline interpolation coefficients (in place;
        // a few % of the render cost).
        bspline3_prefilter_plane(&mut value, t, ch);
        bspline3_prefilter_plane(&mut jg_u, t, ch);
        bspline3_prefilter_plane(&mut jg_v, t, ch);
        // Eroded validity for fractional reads: base texel `(row, col)` is
        // readable iff its whole 4×4 tap neighbourhood `[row−1, row+2] ×
        // [col−1, col+2]` is in frame (border texels whose neighbourhood exits
        // the tile stay `false`; coverage bounds keep fractional reads off them
        // anyway). The all-valid case (patch fully in frame — the common one)
        // skips the erosion entirely.
        let mut valid4 = vec![false; t * t];
        if valid.iter().all(|&v| v) {
            for row in 1..t.saturating_sub(2) {
                valid4[row * t + 1..row * t + t - 2].fill(true);
            }
        } else {
            for row in 1..t.saturating_sub(2) {
                for col in 1..t.saturating_sub(2) {
                    let mut ok = true;
                    'probe: for j in 0..4 {
                        for i in 0..4 {
                            if !valid[(row - 1 + j) * t + (col - 1 + i)] {
                                ok = false;
                                break 'probe;
                            }
                        }
                    }
                    valid4[row * t + col] = ok;
                }
            }
        }
        RefineTile {
            res: t,
            pad: pad as usize,
            channels: ch,
            seed,
            value,
            jg_u,
            jg_v,
            valid,
            valid4,
        }
    })
}

/// Value core at offset `(au, av)`: a [`RefineTile::read_core`] when the view
/// has a tile and the offset is inside its coverage (the expected path), else
/// the direct [`render_core`] fallback (a coarse-grid view has no tile — see
/// [`try_render_refine_tile`]; an out-of-coverage read on a tiled view is
/// counted in [`prof::N_TILE_FALLBACK`]).
#[allow(clippy::too_many_arguments)]
pub(in crate::patch::keypoint_subpixel) fn core_value(
    patch: &OrientedPatch,
    view: &ProjectedImage<'_>,
    tile: Option<&RefineTile>,
    au: f64,
    av: f64,
    wpp_u: f64,
    wpp_v: f64,
    resolution: u32,
    sampler: Sampler,
    support: &Support,
    channels: usize,
    out: &mut [f32],
) -> bool {
    if let Some(tile) = tile {
        debug_assert_eq!(tile.channels, channels);
        match prof::VALUE_READ.time(|| tile.read_core(au, av, resolution as usize, support, out)) {
            Some(ok) => return ok,
            None => prof::count(&prof::N_TILE_FALLBACK, 1),
        }
    }
    render_core(
        patch, view, au, av, wpp_u, wpp_v, resolution, sampler, support, channels, out,
    )
}

/// Value + patch-grid-Jacobian core at offset `(au, av)` for the GN step.
///
/// Tile path: reads **only** the two Jacobian planes ([`RefineTile::read_jg`])
/// — the value core in `g` is required to already hold the tile read at
/// `(au, av)` (the GN loop's invariant: the last successful score evaluation
/// was at the current offset — see [`refine_one_view`](crate::patch::keypoint_subpixel::refine_one_view)).
/// Direct fallback ([`render_core_with_jg`]) fills `g` too, keeping value and
/// Jacobian mutually consistent per path.
#[allow(clippy::too_many_arguments)]
pub(in crate::patch::keypoint_subpixel) fn core_value_with_jg(
    patch: &OrientedPatch,
    view: &ProjectedImage<'_>,
    tile: Option<&RefineTile>,
    au: f64,
    av: f64,
    wpp_u: f64,
    wpp_v: f64,
    resolution: u32,
    sampler: Sampler,
    support: &Support,
    channels: usize,
    g: &mut [f32],
    jg_u: &mut [f32],
    jg_v: &mut [f32],
    img_scratch: &mut ImageF32WithGrad,
) -> bool {
    if let Some(tile) = tile {
        debug_assert_eq!(tile.channels, channels);
        match prof::GRAD_READ
            .time(|| tile.read_jg(au, av, resolution as usize, support, jg_u, jg_v))
        {
            Some(ok) => return ok,
            None => prof::count(&prof::N_TILE_FALLBACK, 1),
        }
    }
    render_core_with_jg(
        patch,
        view,
        au,
        av,
        wpp_u,
        wpp_v,
        resolution,
        sampler,
        support,
        channels,
        g,
        jg_u,
        jg_v,
        img_scratch,
    )
}
