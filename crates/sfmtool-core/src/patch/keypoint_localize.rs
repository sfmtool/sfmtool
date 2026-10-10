// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Patch-keypoint localization: every view of a point aligned to its
//! reference render, in one pass.
//!
//! See `specs/core/patch/patch-keypoint-localization.md` and
//! `specs/core/patch/keypoint-localization-search-cache.md`. Given one 3D point
//! with its oriented patch, a view set, and a starting keypoint per view,
//! [`localize_patch_keypoints`] moves each view's keypoint to where its tile best
//! matches the point's reference render, and reports which views it kept.
//!
//! The template is the point's stored bitmap as the starting keypoints render
//! it: the `R×R` render of the reference observation at its own keypoint, never
//! blurred. The reference is the caller's (the point's stored reference, or the
//! reference a bench track holds), or where there is none the reference-view
//! rule picks one from the views' renders at their starting keypoints. Where
//! the rule picks none it would store, the template is the fused mean of the
//! views, which is then the point's stored bitmap. The reference observation's
//! keypoint is returned as given. Every other view
//! has one context tile rendered around its starting keypoint, its own tile is
//! read by the member self-similarity gate, and the tile is searched once for
//! the shift whose ZNCC against the template is highest, refined to sub-pixel
//! by a quadratic fit over the 3×3 neighbourhood of the integer peak. Views
//! whose tile does not fix a 2D position of its own, move too far from
//! the point's projection, leave the frame, or match the template too poorly
//! are dropped. Nothing is iterated: the template does not change while the
//! views are aligned.
//!
//! The render and z-normalization machinery is the same as
//! [normal refinement](super::normal_refine) and
//! [view selection](super::view_selection); the kernel here is the per-view
//! windowed-ZNCC translation search against a fixed template.

pub mod prof;

mod align;
mod kernels;
mod params;
mod reference;
mod search;
mod seed;

use crate::camera::sampler::{render_phase, render_tile};
use crate::camera::WarpMap;
use crate::patch::cloud::{OrientedPatch, PatchCloud};
#[cfg(test)]
use crate::patch::normal_refine::FLAT_NORM_SQ_EPS;
use crate::patch::normal_refine::{ProjectedImage, Sampler, Support};
use crate::patch::self_similarity::{zncc_self_similarity_radius, PatchTile, SelfSimilarityParams};
use crate::patch::PatchCounter;
use crate::progress::{Cancelled, Progress};
// Only the reference scorer (`znorm_core`, test-only) needs the moment helper.
#[cfg(test)]
use crate::patch::normal_refine::{build_support, weighted_moments_pub, PatchWindow};
use crate::reconstruction::SfmrReconstruction;
use nalgebra::Point3;
use rayon::prelude::*;

// Public API, re-exported at the historical `keypoint_localize::` paths.
pub use params::{
    KeypointLocalization, KeypointLocalizeParams, SearchStrategy,
    DEFAULT_MAX_MEMBER_ZNCC_SELF_SIMILARITY_RADIUS,
};
pub use reference::{TemplateKind, TrackReferences, ViewScore, ViewSearch};
pub use seed::keypoint_grid_offset;

pub(super) use align::{resolve_reference, ResolvedReference};
pub(super) use seed::seed_offset;

// Search machinery, re-exported into this module's namespace for the sibling
// test module's `use super::*`.
#[cfg(test)]
use search::{search_shift, search_shift_plus_descent, SearchScratch, ShiftResult};

// Correlation kernels re-exported into this module's namespace only for the
// sibling test module's `use super::*`; production callers reach them through
// `search`, so the re-export is test-gated to stay warning-clean in release.
#[cfg(test)]
use kernels::{
    compute_channel_grids, compute_channel_grids_scalar, score_cell_one_channel,
    score_cell_one_channel_scalar, support_offsets,
};

/// `f32` lanes per destination pixel the render scratch behind one context tile
/// holds: the warp map's interleaved `(x, y)`, its per-pixel 2x2 Jacobian, and
/// the SVD's two singular values and major direction.
///
/// Counted so a tile's cost can be stated before it is asked for
/// ([`view_cache_bytes`]) and so the size can be tested fallibly in front of the
/// render, which allocates it where no refusal can be threaded.
const RENDER_SCRATCH_LANES: usize = 10;

/// Tile size above which the render scratch is probed before it is built, in
/// bytes.
///
/// Sixteen MB is well past every tile a solve renders -- the production default
/// is tens of KB -- and well short of any size an allocator would refuse, so
/// the probe costs nothing where it would never have fired and is made
/// everywhere it might.
const PROBE_ABOVE_BYTES: usize = 16 << 20;

/// Why a localization could not be run to its end.
///
/// The kernel's buffers are sized by the caller's `search` radius and they grow
/// as its square, so a caller that widens the window far enough asks for a tile
/// no machine has the memory for. An allocation that fails inside a global
/// allocator **aborts the process**, which takes the window and everything
/// unsaved in it; every buffer whose size the caller controls is therefore
/// reserved fallibly, and what would have been an abort arrives here as a
/// refusal a step can report.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LocalizeError {
    /// A buffer the search needs could not be allocated.
    OutOfMemory {
        /// What was asked for, in bytes.
        bytes: usize,
    },
    /// The caller asked the operation to stop.
    Cancelled,
}

impl std::fmt::Display for LocalizeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LocalizeError::OutOfMemory { bytes } => write!(
                f,
                "the localizer could not allocate {bytes} bytes for its search buffers"
            ),
            LocalizeError::Cancelled => write!(f, "the operation was cancelled"),
        }
    }
}

impl std::error::Error for LocalizeError {}

impl From<Cancelled> for LocalizeError {
    fn from(_: Cancelled) -> Self {
        LocalizeError::Cancelled
    }
}

/// A zeroed `f32` buffer of `len` lanes, or [`LocalizeError::OutOfMemory`].
///
/// `try_reserve_exact` asks the allocator the same question `vec![0.0; len]`
/// asks and hands back the refusal instead of aborting on it.
pub(super) fn try_zeroed_f32(len: usize) -> Result<Vec<f32>, LocalizeError> {
    let mut buffer: Vec<f32> = Vec::new();
    buffer
        .try_reserve_exact(len)
        .map_err(|_| LocalizeError::OutOfMemory {
            bytes: len.saturating_mul(std::mem::size_of::<f32>()),
        })?;
    buffer.resize(len, 0.0);
    Ok(buffer)
}

/// A `false`-filled `bool` buffer of `len` entries, or
/// [`LocalizeError::OutOfMemory`].
fn try_false_bools(len: usize) -> Result<Vec<bool>, LocalizeError> {
    let mut buffer: Vec<bool> = Vec::new();
    buffer
        .try_reserve_exact(len)
        .map_err(|_| LocalizeError::OutOfMemory { bytes: len })?;
    buffer.resize(len, false);
    Ok(buffer)
}

/// The bytes one view's search costs at `params`, over a photograph of
/// `channels` channels.
///
/// Every buffer sized by the search radius, counted once: the rendered context
/// tile the search reads from (its centered planes, its invalidity plane and
/// its validity map), and the warp map and remapped image the render builds on
/// the way to it. The side of that tile is `R + 2 · margin`, so the answer
/// grows as the **square** of the search radius, which is what makes a widened
/// window worth budgeting for before it is attempted rather than after.
///
/// The shift grids of the search scratch are not here: there is one set of them
/// per call rather than one per view, and they are an order smaller than the
/// tiles they slide over. The views are searched one after another, so one
/// tile is held at a time.
///
/// # Example
///
/// ```
/// # use sfmtool_core::patch::keypoint_localize::{view_cache_bytes, KeypointLocalizeParams};
/// let params = KeypointLocalizeParams::default();
/// // A three-channel photograph at the production defaults: tens of KB.
/// assert!(view_cache_bytes(&params, 3) < 1 << 20);
/// ```
pub fn view_cache_bytes(params: &KeypointLocalizeParams, channels: usize) -> usize {
    let resolution = params.resolution.max(2);
    let margin = params.search.ceil().max(1.0) as i64;
    let side = (resolution as usize).saturating_add(2 * margin.max(0) as usize);
    let pixels = side.saturating_mul(side);
    let istride = cache_istride(side);
    let f32_size = std::mem::size_of::<f32>();
    // The tile: one centered plane per channel plus the invalidity plane, each
    // `istride · side` lanes, and the `bool` validity map.
    let tile = istride
        .saturating_mul(side)
        .saturating_mul(channels + 1)
        .saturating_mul(f32_size)
        .saturating_add(pixels);
    // The render scratch it is built from: the warp map (two lanes of `(x, y)`,
    // four of Jacobian and four of SVD) and the remapped `u8` image.
    let render = pixels
        .saturating_mul(RENDER_SCRATCH_LANES)
        .saturating_mul(f32_size)
        .saturating_add(pixels.saturating_mul(channels));
    tile.saturating_add(render)
}

/// A rendered context tile for one view: source colour over a
/// `cache_res × cache_res` grid (larger than the scored `R×R` core so the shift
/// search can slide), plus per-pixel validity from the warp map (an invalid pixel
/// is out of frame, rendered black, and must not be scored).
///
/// The localizer renders it **once per view**, centred on the view's starting
/// keypoint and `2 · margin` wider than the core, and scores every shift of the
/// search window from it. Because the patch frame is fixed during localization,
/// an integer in-plane shift is an integer tile-index shift, so a read at an
/// integer offset is bit-identical to re-warping the patch at that offset (see
/// `specs/core/patch/keypoint-localization-search-cache.md`).
///
/// **Layout (stage 1 of the SIMD search kernel).** The cache is **planar per
/// channel** in **centered `f32`**: `planes[c][row · istride + col]` holds
/// `I − means[c]` (the channel mean over the cache). Centering is load-bearing —
/// the windowed-ZNCC denominator `S2 − S1²/W` is a catastrophic-cancellation trap
/// in `f32` when `I ~ 10²` (`S2 ~ 10⁷`); centering makes `S1 ≈ 0` and `S2 ≈
/// variance · W`, so `f32` accumulation is accurate. The numerator is recovered
/// exactly by `Ncross = Ncross' + mean · Σ kern` — algebraically identical to
/// z-normalize-then-dot for any template. Rows are padded to `istride =
/// align_up(cacheW − 1 + 16, 8)` so a 16-wide aligned `f32` load from any support
/// column stays in bounds; the pad columns hold `0` (= the mean after centering,
/// harmless — they only feed discarded grid cells). The per-pixel invalidity
/// plane (`1.0` out of frame, else `0.0`) lives alongside in the same `istride`
/// row layout for the SIMD validity pass; the `bool` `valid` map is kept too for
/// the integer-tracked core read (`extract_core`).
struct ContextTile {
    /// Side length of the (square) tile, in patch-grid px.
    res: usize,
    /// Row stride of each plane in `f32` lanes: `align_up(res − 1 + 16, 8)` so a
    /// 16-wide aligned load from any support column stays in bounds.
    istride: usize,
    /// Channel count.
    channels: usize,
    /// Per-channel mean over the cache (the value subtracted to produce
    /// [`planes`](Self::planes)). Used to recover original-scale values on the
    /// core read (`extract_core`) and to fold back into the numerator
    /// (`Ncross = Ncross' + mean · Σ kern`).
    means: Vec<f32>,
    /// Centered per-channel planes: `planes[c][row · istride + col] = I_c − means[c]`.
    /// Length `channels`, each plane length `istride · res`.
    planes: Vec<Vec<f32>>,
    /// Per-pixel invalidity plane in the same `istride` row layout (`1.0` invalid,
    /// `0.0` valid). Drives the SIMD validity count pass that gates `−∞` shifts.
    invalid_plane: Vec<f32>,
    /// Per-pixel validity (`true` in frame), `[row · res + col]`. The `bool` form
    /// is what the core read (`extract_core`) checks.
    valid: Vec<bool>,
    /// Whether every pixel of [`valid`](Self::valid) is in frame, recorded when
    /// the tile is built. Then [`invalid_plane`](Self::invalid_plane) is all
    /// zero, so the search skips its validity count.
    all_valid: bool,
}

/// Compute the row stride for the centered planar cache: enough lanes to admit a
/// 16-wide aligned `f32` load starting at any column in `[0, res)`. `align_up`
/// to 8 lanes (one `__m256`) keeps row starts naturally aligned for AVX2.
#[inline]
fn cache_istride(res: usize) -> usize {
    // The widest load needed is 16 f32s (two `__m256`s) starting at column
    // `res − 1`, which reads through `res − 1 + 15`; the buffer must hold
    // through `res − 1 + 16` (cap). Round up to a multiple of 8.
    let need = res - 1 + 16;
    need.div_ceil(8) * 8
}

/// Project a homogeneous world point `(p, w)` into a view **without** the
/// frame-bounds test: `None` only when the point falls behind the camera or
/// outside the camera model's valid domain. `w = 1` is a finite point; `w = 0` is
/// a direction (a point at infinity), rotated into the camera frame without
/// translation and projected as a ray.
///
/// This is the form a *residual* wants: a reprojection a pixel outside the frame
/// is a small error, not a missing measurement, so [candidate
/// spawning](super::spawn) measures against this while the visibility-gated
/// [`project`] decides which views a patch can be rendered in at all.
pub(crate) fn project_unclipped(
    view: &ProjectedImage<'_>,
    p: &Point3<f64>,
    w: f64,
) -> Option<(f64, f64)> {
    view.camera
        .project_homogeneous(view.cam_from_world, p.coords, w)
        .map(|[x, y]| (x, y))
}

/// Project a homogeneous world point `(p, w)` into a view; `None` when it falls
/// behind the camera or outside the frame. See [`project_unclipped`] for the
/// variant that skips the frame test.
pub(super) fn project(view: &ProjectedImage<'_>, p: &Point3<f64>, w: f64) -> Option<(f64, f64)> {
    let (px, py) = project_unclipped(view, p, w)?;
    let (iw, ih) = (view.camera.width as f64, view.camera.height as f64);
    (px >= 0.0 && py >= 0.0 && px < iw && py < ih).then_some((px, py))
}

/// The patch centre re-anchored on the plane by an in-plane offset `(au, av)` in
/// patch-grid px: `X_p + au·wpp_u·û − av·wpp_v·v̂`. Grid rows count *downward*
/// from `+v̂` (they map to `−v_axis`, matching `WarpMap::from_patch`), so a
/// positive `av` steps along `−v̂`.
pub(super) fn shifted_center(
    patch: &OrientedPatch,
    au: f64,
    av: f64,
    wpp_u: f64,
    wpp_v: f64,
) -> Point3<f64> {
    patch.center + patch.u_axis * (au * wpp_u) - patch.v_axis * (av * wpp_v)
}

/// Render one view's context tile / cache with the patch centre at in-plane
/// offset `(au, av)` (patch-grid px). The context patch spans `context_res / R`
/// times the core extent, rendered at `context_res`, so each context pixel equals
/// one core pixel in world units. The localizer calls it **once per view**
/// with `(au, av)` at the view's starting keypoint: the core at shift `(0, 0)`
/// then sits at tile offset `(context_res − R) / 2`.
///
/// **The buffers are reserved before the render, and fallibly.** Their side is
/// `R + 2 · margin`, so a caller that widens its search window asks for a tile
/// that grows as the square of the radius; an allocation the global allocator
/// refuses aborts the process, and an abort takes the window with it. So the
/// planes this will fill are reserved first, through
/// [`try_zeroed_f32`], and the render scratch -- the warp map and the remapped
/// image, which are allocated inside kernels no refusal can be threaded through
/// -- is asked for at its own size and released first whenever the tile is big
/// enough for the answer to be no, so a refusal comes back as
/// [`LocalizeError::OutOfMemory`] before either is built.
#[allow(clippy::too_many_arguments)]
fn render_context(
    patch: &OrientedPatch,
    view: &ProjectedImage<'_>,
    au: f64,
    av: f64,
    wpp_u: f64,
    wpp_v: f64,
    resolution: u32,
    context_res: u32,
    sampler: Sampler,
    progress: &Progress<'_>,
) -> Result<ContextTile, LocalizeError> {
    // The tile the search will read, allocated **before** the render: the
    // warp map and the remapped image are of the same order, and asking for
    // this first means an impossible size is refused here rather than aborting
    // the process inside one of them. `try_zeroed_f32` asks for exactly what
    // will be filled below.
    let cr = context_res as usize;
    let channels = view.pyramid.level(0).channels() as usize;
    let istride = cache_istride(cr);
    let mut planes: Vec<Vec<f32>> = Vec::new();
    planes
        .try_reserve_exact(channels)
        .map_err(|_| LocalizeError::OutOfMemory {
            bytes: channels.saturating_mul(std::mem::size_of::<Vec<f32>>()),
        })?;
    for _ in 0..channels {
        planes.push(try_zeroed_f32(istride.saturating_mul(cr))?);
    }
    let mut invalid_plane = try_zeroed_f32(istride.saturating_mul(cr))?;
    let mut valid = try_false_bools(cr.saturating_mul(cr))?;

    // The render's own scratch is bigger than the tile it produces -- the warp
    // map carries ten `f32` lanes per pixel against the tile's four per channel
    // -- and it is allocated inside `WarpMap` and the remap, where no refusal
    // can be threaded. So a tile big enough to be refused has that size asked
    // for here first and released: what the allocator says about it is what it
    // will say a line later, and this is the line that can still report it.
    //
    // Only above the threshold, because this runs once per view per point of a
    // whole cloud and an allocator does not refuse a few hundred KB: a probe at
    // the production tile size would be an allocation and a free per render,
    // bought against a refusal that cannot happen.
    let scratch_lanes = cr.saturating_mul(cr).saturating_mul(RENDER_SCRATCH_LANES);
    if scratch_lanes.saturating_mul(std::mem::size_of::<f32>()) > PROBE_ABOVE_BYTES {
        let mut probe: Vec<f32> = Vec::new();
        probe
            .try_reserve_exact(scratch_lanes)
            .map_err(|_| LocalizeError::OutOfMemory {
                bytes: scratch_lanes.saturating_mul(std::mem::size_of::<f32>()),
            })?;
    }

    let center = shifted_center(patch, au, av, wpp_u, wpp_v);
    let scale = context_res as f64 / resolution as f64;
    let mut ctx_patch = OrientedPatch::from_center_normal(
        center,
        patch.normal(),
        patch.v_axis,
        [patch.half_extent[0] * scale, patch.half_extent[1] * scale],
    );
    // Preserve the anchor's homogeneous weight so a point at infinity renders as
    // a direction patch (corners are directions), not a finite surfel.
    ctx_patch.w = patch.w;
    let mut map = prof::RENDER_PROJECT
        .time(|| WarpMap::from_patch(&ctx_patch, view.camera, view.cam_from_world, context_res));
    if sampler.needs_svd() {
        prof::RENDER_SVD.time(|| map.compute_svd());
    }
    let img = prof::RENDER_REMAP.time(|| {
        let _phase = render_phase(progress, sampler, 1);
        render_tile(view.pyramid, &mut map, sampler)
    });
    debug_assert_eq!(channels, img.channels() as usize);

    // Per-channel sum → mean over the cache. We accumulate in `f64` to keep the
    // centering exact to the last `f32` ulp (one mean per channel; cheap).
    let means: Vec<f32> = prof::RENDER_MEAN.time(|| {
        let mut sums = vec![0.0f64; channels];
        let total = (cr * cr) as f64;
        for row in 0..context_res {
            for col in 0..context_res {
                for (ch, slot) in sums.iter_mut().enumerate() {
                    *slot += img.get_pixel(col, row, ch as u32) as f64;
                }
            }
        }
        sums.iter().map(|&s| (s / total) as f32).collect()
    });

    // Centered planar planes. Pad columns past `cr` stay at `0.0` (= the mean
    // after centering, harmless — those columns only feed discarded grid cells
    // past the search window).
    let mut all_valid = true;
    prof::RENDER_CENTER.time(|| {
        for row in 0..context_res {
            for col in 0..context_res {
                let r = row as usize;
                let c = col as usize;
                let row_off = r * istride + c;
                let v = map.is_valid(col, row);
                valid[r * cr + c] = v;
                all_valid &= v;
                invalid_plane[row_off] = if v { 0.0 } else { 1.0 };
                for ch in 0..channels {
                    let p = img.get_pixel(col, row, ch as u32) as f32;
                    planes[ch][row_off] = p - means[ch];
                }
            }
        }
    });
    Ok(ContextTile {
        res: cr,
        istride,
        channels,
        means,
        planes,
        invalid_plane,
        valid,
        all_valid,
    })
}

/// Extract the raw (un-normalized) core of `tile` at window offset `(oy, ox)`
/// into `out`, flat `[channel * n + support_index]`. Returns `false` (leaving
/// `out` untouched) when any support pixel is invalid (out of frame) — the slid
/// core then can't be scored.
fn extract_core(
    tile: &ContextTile,
    support: &Support,
    resolution: usize,
    oy: usize,
    ox: usize,
    out: &mut [f32],
) -> bool {
    let ch = tile.channels;
    let n = support.pixels.len();
    let tile_res = tile.res;
    let istride = tile.istride;
    for (k, &p) in support.pixels.iter().enumerate() {
        let (r, c) = (p / resolution, p % resolution);
        // The `valid` plane stays in the tight `tile_res`-stride layout (it's a
        // per-pixel mask for the core read, not part of the SIMD hot
        // loop); the centered planes live in the padded `istride` layout.
        if !tile.valid[(oy + r) * tile_res + (ox + c)] {
            return false;
        }
        let cp = (oy + r) * istride + (ox + c);
        // Add back the per-channel mean so a core read recovers the original
        // source value (the centered representation is an internal optimization
        // for the SIMD search; `znorm_core` and the rest of the pipeline see
        // the same values as the old interleaved cache).
        for cc in 0..ch {
            out[cc * n + k] = tile.planes[cc][cp] + tile.means[cc];
        }
    }
    true
}

/// Whether a ZNCC against the template fails the **absolute** floor: finite and
/// below `floor`. A `NaN` (the view was not scored) has no verdict to fail, and a
/// `floor` of `0.0` or below disables the gate exactly — a negative correlation
/// is only refused when the caller asks for a positive floor.
#[inline]
fn below_absolute_floor(zncc: f64, floor: f64) -> bool {
    floor > 0.0 && zncc.is_finite() && zncc < floor
}

/// One view's **own** core's ZNCC self-similarity radius, the number the
/// member gate ([`KeypointLocalizeParams::max_member_zncc_self_similarity_radius`])
/// judges: the `R×R` core at window offset `(oy, ox)` of `tile`, the tile the
/// caller already rendered for that view, in the core's grid px, read the
/// overlap way with the default [`SelfSimilarityParams`].
///
/// Only the core is read: at each shift the reading correlates the samples
/// both windows hold inside the core, so no pixel of the tile around it enters
/// the number, and it does not depend on how wide a tile the caller rendered.
/// Every sample counts as data, a pixel out of frame as the black it was
/// rendered, as it does for every other read of the tile. Only the leading
/// three channels are read, as a patch tile's colour.
///
/// # Panics
///
/// Panics if the core does not lie inside the tile.
fn member_self_similarity_radius(
    tile: &ContextTile,
    resolution: usize,
    oy: usize,
    ox: usize,
    scratch: &mut Vec<f32>,
) -> f64 {
    assert!(
        oy + resolution <= tile.res && ox + resolution <= tile.res,
        "member_self_similarity_radius: the {resolution}×{resolution} core at ({ox}, {oy}) \
         does not fit in the {}×{} tile",
        tile.res,
        tile.res
    );
    let colour = tile.channels.min(3);
    if colour == 0 {
        return f64::NAN;
    }
    let n = resolution * resolution;
    scratch.clear();
    scratch.resize(colour * n, 0.0);
    for (c, plane) in tile.planes.iter().take(colour).enumerate() {
        for row in 0..resolution {
            let src = &plane[(oy + row) * tile.istride + ox..][..resolution];
            let dst = &mut scratch[c * n + row * resolution..][..resolution];
            for (out, &v) in dst.iter_mut().zip(src) {
                // The planes are centred; add the channel mean back so the
                // reading sees the source values (its flat test is on spread,
                // so this only keeps the numbers recognisable).
                *out = v + tile.means[c];
            }
        }
    }
    let core = PatchTile {
        values: scratch,
        channels: colour,
        width: resolution,
        height: resolution,
    };
    zncc_self_similarity_radius(
        &core,
        None,
        [0, 0, resolution, resolution],
        &SelfSimilarityParams::default(),
    )
    .radius
}

/// z-normalize a raw core (`raw[channel * n + k]`) over the kept original
/// channels into `out` (compacted, `out[kept_c * n + k]`), folding `√w` in so a
/// plain dot is a windowed ZNCC. A kept channel that is flat in this core
/// (windowed norm² below [`FLAT_NORM_SQ_EPS`]) is written as zeros (contributes
/// `0` to the ZNCC rather than a misaligned dot), matching view selection.
///
/// Production [`search_shift`] folds this z-normalization into its correlation
/// maps; this remains the reference the equivalence test scores against.
#[cfg(test)]
fn znorm_core(raw: &[f32], support: &Support, keep_mask: &[bool], out: &mut [f32]) {
    let n = support.pixels.len();
    let mut kc = 0;
    for (c, &keep) in keep_mask.iter().enumerate() {
        if !keep {
            continue;
        }
        let col = &raw[c * n..][..n];
        let (s1, s2) = weighted_moments_pub(col, &support.weights);
        let mean = (s1 / support.total_weight) as f32;
        let norm_sq = s2 - s1 * (mean as f64);
        let dst = &mut out[kc * n..][..n];
        if norm_sq < FLAT_NORM_SQ_EPS {
            dst.fill(0.0);
        } else {
            let inv = (1.0 / norm_sq.sqrt()) as f32;
            for (d, (&x, &sw)) in dst.iter_mut().zip(col.iter().zip(&support.sqrt_weights)) {
                *d = sw * (x - mean) * inv;
            }
        }
        kc += 1;
    }
}

/// Per-channel-averaged ZNCC of a z-normalized core against a unit-norm template
/// (both laid out `[c * n + k]`). Reference scoring for the equivalence test;
/// production [`search_shift`] computes the same value by accumulation.
#[cfg(test)]
fn template_zncc(core: &[f32], tmpl: &[f32], channels: usize, n: usize) -> f64 {
    let mut s = 0.0;
    for c in 0..channels {
        let a = &core[c * n..][..n];
        let b = &tmpl[c * n..][..n];
        s += a
            .iter()
            .zip(b)
            .map(|(&x, &y)| (x as f64) * (y as f64))
            .sum::<f64>();
    }
    s / channels as f64
}

/// Sub-sample peak offset in `[-1, 1]` from a 3-point parabola (scores at `-1`,
/// `0`, `+1` around an integer maximum).
fn parabolic(mid: f64, left: f64, right: f64) -> f64 {
    let denom = left - 2.0 * mid + right;
    if denom.abs() < 1e-12 {
        return 0.0;
    }
    (0.5 * (left - right) / denom).clamp(-1.0, 1.0)
}

/// Sub-pixel offset `(sy, sx)` of a score surface's peak from its integer
/// maximum `(py, px)`, each in `[-1, 1]`.
///
/// `peak` is the score at `(py, px)` and `nb(dy, dx)` the score at another
/// cell, `None` where that cell was not scored. With all eight neighbours
/// scored, the offset is the vertex `−H⁻¹g` of the quadratic through the 3×3
/// neighbourhood, with the gradient `g` and Hessian `H` from central
/// differences. The cross term `H_xy = (f(1,1) − f(1,−1) − f(−1,1) + f(−1,−1))/4`
/// keeps a shift along one axis from showing up on the other on a diagonal
/// texture, which a separate parabola per axis does not. When a neighbour is
/// missing, or `H` is not negative definite (the surface is not a peak to
/// second order), the offset falls back to a 3-point parabola on each axis, and
/// an axis with a missing neighbour stays at the integer cell.
fn subpixel_peak(peak: f64, py: i64, px: i64, nb: impl Fn(i64, i64) -> Option<f64>) -> (f64, f64) {
    let up = nb(py - 1, px);
    let down = nb(py + 1, px);
    let left = nb(py, px - 1);
    let right = nb(py, px + 1);
    if let (Some(u), Some(d), Some(l), Some(r)) = (up, down, left, right) {
        let diag = (
            nb(py - 1, px - 1),
            nb(py - 1, px + 1),
            nb(py + 1, px - 1),
            nb(py + 1, px + 1),
        );
        if let (Some(mm), Some(mp), Some(pm), Some(pp)) = diag {
            let gy = 0.5 * (d - u);
            let gx = 0.5 * (r - l);
            let hyy = u - 2.0 * peak + d;
            let hxx = l - 2.0 * peak + r;
            let hxy = 0.25 * (pp - pm - mp + mm);
            let det = hyy * hxx - hxy * hxy;
            // Negative definite: both diagonal terms negative and `det > 0`.
            if hyy < 0.0 && hxx < 0.0 && det > 1e-12 {
                let sy = -(hxx * gy - hxy * gx) / det;
                let sx = -(hyy * gx - hxy * gy) / det;
                if sy.is_finite() && sx.is_finite() {
                    return (sy.clamp(-1.0, 1.0), sx.clamp(-1.0, 1.0));
                }
            }
        }
    }
    let sy = match (up, down) {
        (Some(l), Some(r)) => parabolic(peak, l, r),
        _ => 0.0,
    };
    let sx = match (left, right) {
        (Some(l), Some(r)) => parabolic(peak, l, r),
        _ => 0.0,
    };
    (sy, sx)
}

/// Localize the keypoints of one oriented patch over a view set by aligning
/// every view to the point's reference render.
///
/// `views` is one [`ProjectedImage`] per reconstruction image (indexed by image
/// index); `view_set` lists the views to localize (the output of
/// [view selection](super::view_selection), or a track). `starting_keypoints`,
/// when given, is one **optional** seed per `view_set` entry (source-image px),
/// parallel to it: `Some([x, y])` starts that view there, `None` starts it at
/// the point's own projection `project_i(X_p)`, and so does `None` for the
/// whole slice. `reference` is the position in `view_set` of the point's
/// reference observation, whose render at its starting keypoint is the template
/// and whose keypoint is not moved; `None`, or a reference the grazing
/// pre-filter turns away, has the reference-view rule pick one from the renders
/// at the starting keypoints. Returns the kept views, their keypoints and their
/// scores against the template; see [`KeypointLocalization`] and
/// `specs/core/patch/patch-keypoint-localization.md`.
///
/// # Panics
///
/// Panics if a search buffer cannot be allocated, which a `search` radius wide
/// enough to size the per-view tile past the machine's memory can do. A caller
/// whose radius is the person's rather than a constant runs
/// [`try_localize_patch_keypoints`] and reports the [`LocalizeError`] instead.
pub fn localize_patch_keypoints(
    patch: &OrientedPatch,
    views: &[ProjectedImage<'_>],
    view_set: &[u32],
    starting_keypoints: Option<&[Option<[f64; 2]>]>,
    reference: Option<usize>,
    params: &KeypointLocalizeParams,
) -> KeypointLocalization {
    try_localize_patch_keypoints(
        patch,
        views,
        view_set,
        starting_keypoints,
        reference,
        params,
        &Progress::none(),
    )
    .expect("the localizer's buffers fit and no cancellation is possible without a Progress")
}

/// [`localize_patch_keypoints`] as a fallible call: the same localization, with
/// the two things that can stop it reported rather than raised.
///
/// The buffers the search reads from are sized by
/// [`search`](KeypointLocalizeParams::search) and grow as its square, so a
/// caller that takes that radius from a person can ask for a tile no machine
/// can hold; an allocation the global allocator refuses **aborts the process**,
/// and this is the entry point that hands back
/// [`LocalizeError::OutOfMemory`] instead. `progress` is polled between views,
/// so a caller that can be cancelled gets [`LocalizeError::Cancelled`] rather
/// than running the whole set out.
///
/// # Example
///
/// ```no_run
/// # use sfmtool_core::patch::keypoint_localize::{
/// #     try_localize_patch_keypoints, KeypointLocalizeParams,
/// # };
/// # use sfmtool_core::patch::cloud::OrientedPatch;
/// # use sfmtool_core::patch::normal_refine::ProjectedImage;
/// # use sfmtool_core::progress::Progress;
/// # fn run(
/// #     patch: &OrientedPatch,
/// #     views: &[ProjectedImage<'_>],
/// #     view_set: &[u32],
/// # ) -> Result<(), Box<dyn std::error::Error>> {
/// // The first view of the set is the point's reference observation. It is
/// // the reference reported unless the grazing pre-filter turns it away or its
/// // tile cannot be rendered at its keypoint.
/// let localized = try_localize_patch_keypoints(
///     patch,
///     views,
///     view_set,
///     None,
///     Some(0),
///     &KeypointLocalizeParams::default(),
///     &Progress::none(),
/// )?;
/// if let Some(reference) = localized.reference {
///     let k = localized.views.iter().position(|&v| v == reference).unwrap();
///     assert_eq!(localized.zncc[k], 1.0);
/// }
/// # Ok(())
/// # }
/// ```
pub fn try_localize_patch_keypoints(
    patch: &OrientedPatch,
    views: &[ProjectedImage<'_>],
    view_set: &[u32],
    starting_keypoints: Option<&[Option<[f64; 2]>]>,
    reference: Option<usize>,
    params: &KeypointLocalizeParams,
    progress: &Progress<'_>,
) -> Result<KeypointLocalization, LocalizeError> {
    prof::TOTAL.time(|| {
        align::align_to_reference(
            patch,
            views,
            view_set,
            starting_keypoints,
            reference,
            params,
            progress,
        )
    })
}

/// Batch [`localize_patch_keypoints`] over a [`PatchCloud`], parallel across
/// patches (rayon). `view_sets[i]` lists, for patch `i`, the views to localize.
/// `starting_keypoints`, when given, is parallel to `view_sets` in both
/// dimensions (one **optional** seed per view); `None` seeds every view at the
/// point's projection, and so does an **empty** per-patch entry -- the batch
/// form has no per-patch `Option`, so it says "this patch is unseeded" with an
/// empty list -- and so does a `None` for one view inside a seeded patch's list.
/// `references`, when given, is parallel to the cloud: per patch, the position
/// in its view set of its reference observation, or `None` to have the
/// reference-view rule pick one. Results are returned in cloud order.
///
/// `done`, when given, is bumped once per patch, for a caller polling progress
/// from another thread. `progress` receives a `patches` count about every
/// hundredth of the way through, is polled for cancellation before each patch
/// and between a patch's views, and, when detailed, times the renders of each
/// sampler in its own detail phase.
///
/// # Errors
///
/// [`Cancelled`] when `progress` was cancelled before every patch was
/// localized.
///
/// # Panics
///
/// Panics if `view_sets.len() != cloud.len()` (or `starting_keypoints` /
/// `references` are given and not parallel), if an index is out of range, or
/// if a search buffer cannot be allocated.
#[allow(clippy::too_many_arguments)]
pub fn localize_patch_cloud_keypoints(
    cloud: &PatchCloud,
    views: &[ProjectedImage<'_>],
    view_sets: &[Vec<u32>],
    starting_keypoints: Option<&[Vec<Option<[f64; 2]>>]>,
    references: Option<&[Option<usize>]>,
    params: &KeypointLocalizeParams,
    done: Option<&std::sync::atomic::AtomicUsize>,
    progress: &Progress<'_>,
) -> Result<Vec<KeypointLocalization>, Cancelled> {
    assert_eq!(
        view_sets.len(),
        cloud.len(),
        "view_sets must be parallel to the cloud"
    );
    if let Some(seeds) = starting_keypoints {
        assert_eq!(
            seeds.len(),
            cloud.len(),
            "starting_keypoints must be parallel to the cloud"
        );
    }
    if let Some(refs) = references {
        assert_eq!(
            refs.len(),
            cloud.len(),
            "references must be parallel to the cloud"
        );
    }
    if prof::enabled() {
        prof::reset();
    }
    let wall_start = std::time::Instant::now();
    let counter = PatchCounter::new(cloud.len(), done, progress);
    let out: Vec<Option<KeypointLocalization>> = cloud
        .patches
        .par_iter()
        .enumerate()
        .map(|(i, patch)| {
            if progress.is_cancelled() {
                return None;
            }
            let seeds = starting_keypoints
                .map(|s| s[i].as_slice())
                .filter(|s| !s.is_empty());
            let reference = references.and_then(|r| r[i]);
            let out = try_localize_patch_keypoints(
                patch,
                views,
                &view_sets[i],
                seeds,
                reference,
                params,
                progress,
            );
            let out = match out {
                Ok(out) => out,
                // A cancellation mid-patch: the check below returns it.
                Err(LocalizeError::Cancelled) => return None,
                Err(e) => panic!("the localizer's buffers fit: {e}"),
            };
            counter.finished();
            Some(out)
        })
        .collect();
    if prof::enabled() {
        prof::report(cloud.len(), wall_start.elapsed().as_secs_f64());
    }
    progress.check_cancel()?;
    Ok(out
        .into_iter()
        .map(|o| o.expect("every patch ran when nothing was cancelled"))
        .collect())
}

/// For each patch of `cloud` (linked to `recon` via `point_indexes`), the track image
/// indices observing its source 3D point — a convenience default `view_sets` for
/// [`localize_patch_cloud_keypoints`] when no view selection has been run.
/// Identical to
/// [`view_indices_from_reconstruction`](super::normal_refine::view_indices_from_reconstruction).
///
/// # Panics
///
/// Panics if `cloud.point_indexes` is not parallel to its patches.
pub fn track_views_from_reconstruction(
    recon: &SfmrReconstruction,
    cloud: &PatchCloud,
) -> Vec<Vec<u32>> {
    super::normal_refine::view_indices_from_reconstruction(recon, cloud)
}

/// Reference (pre-optimization) translation search: score each candidate by
/// extract → z-normalize → template dot. Kept as the oracle the accumulation
/// [`search_shift`] is checked against in [`tests`]; not used in production.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn search_shift_ref(
    tile: &ContextTile,
    tmpl: &[f32],
    support: &Support,
    keep_mask: &[bool],
    channels: usize,
    resolution: usize,
    margin: i64,
    base_y: usize,
    base_x: usize,
) -> Option<ShiftResult> {
    let n = support.pixels.len();
    let mut raw = vec![0f32; tile.channels * n];
    let mut core = vec![0f32; channels * n];
    let span = (2 * margin + 1) as usize;
    let mut grid = vec![f64::NEG_INFINITY; span * span];
    let at =
        |dy: i64, dx: i64| -> usize { ((dy + margin) as usize) * span + (dx + margin) as usize };
    let mut best = (f64::NEG_INFINITY, 0i64, 0i64);
    for dy in -margin..=margin {
        for dx in -margin..=margin {
            // The shift `(dy, dx)` window's top-left support pixel sits at the base
            // offset plus the shift — matching `search_shift`'s grid (whose `gy =
            // margin` row is `dy = 0`, reading at `win_oy + margin = base_y`).
            let oy = (base_y as i64 + dy) as usize;
            let ox = (base_x as i64 + dx) as usize;
            if !extract_core(tile, support, resolution, oy, ox, &mut raw) {
                continue;
            }
            znorm_core(&raw, support, keep_mask, &mut core);
            let z = template_zncc(&core, tmpl, channels, n);
            grid[at(dy, dx)] = z;
            // An exact tie goes to the cell nearer the start, as in
            // `search_shift`.
            let nearer = dy.abs() + dx.abs() < best.1.abs() + best.2.abs();
            if z > best.0 || (z == best.0 && nearer) {
                best = (z, dy, dx);
            }
        }
    }
    if !best.0.is_finite() {
        return None;
    }
    let (peak, py, px) = best;
    let nb = |dy: i64, dx: i64| -> Option<f64> {
        if dy.abs() <= margin && dx.abs() <= margin {
            let g = grid[at(dy, dx)];
            g.is_finite().then_some(g)
        } else {
            None
        }
    };
    let (sy, sx) = subpixel_peak(peak, py, px, nb);
    Some(ShiftResult {
        dx: px as f64 + sx,
        dy: py as f64 + sy,
        ix: px,
        iy: py,
        peak,
    })
}

#[cfg(test)]
mod tests;
