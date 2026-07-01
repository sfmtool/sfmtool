// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Configuration and result types for patch-normal refinement: the per-view
//! [`ProjectedImage`], the [`Objective`] / [`PatchWindow`] / [`Sampler`] /
//! [`CacheMode`] knobs, the [`NormalRefineParams`] bundle and its
//! [`NormalRefineResult`], plus the shared numeric thresholds.

use crate::camera::remap::ImageU8Pyramid;
use crate::camera::CameraIntrinsics;
use crate::geometry::RigidTransform;
use crate::patch::cloud::OrientedPatch;

/// A source image together with the projection that maps world points into it:
/// the camera intrinsics, its world-to-camera pose, and a prebuilt pyramid for
/// sampling colour. Everything a patch needs to be rendered from one view. The
/// pyramid is built once and borrowed for every candidate render, so a refinement
/// allocates no per-candidate source image data.
#[derive(Clone, Copy)]
pub struct ProjectedImage<'a> {
    pub camera: &'a CameraIntrinsics,
    pub cam_from_world: &'a RigidTransform,
    pub pyramid: &'a ImageU8Pyramid,
}

/// Photoconsistency `Φ`: the consensus all-pairs mean ZNCC (see the spec's
/// "Objective" for the closed form and why there is no reference-view
/// variant).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Objective {
    /// Unweighted consensus `ρ̄ = (V‖x̄‖² − 1)/(V − 1)`.
    MeanPairwise,
    /// IRLS-weighted consensus that down-weights outlier (occluded /
    /// wrong-surface) views by a Tukey weight on each view's residual
    /// `‖xᵢ − x̄_w‖`, re-weighting `iters` times. Recommended default.
    RobustWeighted { iters: u32 },
}

/// Per-pixel weight applied to the `R×R` patch when scoring (the NCC window).
///
/// `sigma` is expressed in normalized patch coordinates (the patch spans
/// `(s, t) ∈ [-1, 1]²`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PatchWindow {
    /// Uniform weight over the whole square grid (rotation-leaky; mainly a
    /// baseline).
    Uniform,
    /// Gaussian center weight over the square grid.
    Gaussian { sigma: f64 },
    /// Gaussian weight confined to the inscribed disk — radial, so in-plane
    /// rotation is exactly free and grazing corners don't leak in.
    /// Recommended default.
    GaussianDisk { sigma: f64 },
    // Beyond-v1: `Alpha` — an explicit per-(s, t) mask carried by the patch
    // (spec item 6); deferred until a producer exists.
}

/// How to sample a [`ProjectedImage`]'s pyramid when rendering a patch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sampler {
    /// Plain bilinear from the full-resolution level. The **default**: it barely
    /// moves the found normal (within ~1° of anisotropic on pinhole cameras) at a
    /// fraction of the cost.
    Bilinear,
    /// Anisotropic sampling over the pyramid — the patch warp's Jacobian SVD picks
    /// the level, de-aliasing oblique / grazing views. Costs ~1.6–3× more but keeps
    /// the reported `Φ`/confidence unbiased (bilinear depresses `Φ` on oblique
    /// views) and helps slightly on distorted/fisheye rigs.
    Anisotropic,
}

/// How candidate normals are scored: re-rendered from the source images, or
/// resampled from a cached base patch (see
/// `specs/core/fronto-parallel-patch-cache.md`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheMode {
    /// Re-render every candidate from the source pyramid — the exact path; opt
    /// in for the tightest tail when wall time doesn't matter.
    Off,
    /// Render one supersampled fronto-parallel base per view up front and
    /// affine-resample each candidate from it. **The default** (with
    /// `cache_supersample = 2`): ~2× faster at Φ-equivalent *median* accuracy;
    /// the affine approximation costs a little in the tail on flat-`Φ` data. Note
    /// the returned `photoconsistency` is always source-scored in the final pass,
    /// and `≥ init_photoconsistency` by construction, regardless of the cache.
    /// Falls back to `Off` per-patch when the base can't be built (e.g. non-RGB
    /// imagery).
    FrontoParallel,
}

/// Tunables for [`refine_patch_normal`].
#[derive(Debug, Clone)]
pub struct NormalRefineParams {
    /// Half-extent of the level-0 search cone, degrees.
    pub angular_range_deg: f64,
    /// Grid resolution per tangent axis at each level.
    pub init_steps: u32,
    /// Coarse-to-fine passes; each recenters on the level's best normal and
    /// shrinks the cone to one previous grid spacing.
    pub refine_levels: u32,
    /// Consensus objective.
    pub objective: Objective,
    /// Per-pixel scoring weight / support.
    pub window: PatchWindow,
    /// Per-view floor on the window-weighted valid-pixel fraction; views below
    /// it are dropped for the level.
    pub min_valid_fraction: f64,
    /// Minimum number of (effective) views; below it the candidate / patch is
    /// not refined.
    pub min_views: u32,
    /// How to sample the source pyramids when rendering candidates.
    pub sampler: Sampler,
    /// Candidate-scoring strategy (source re-render vs. fronto-parallel cache).
    pub cache: CacheMode,
    /// Base-patch supersample factor for the cache (≥ 1; `1.0` = candidate
    /// density). Denser bases sharpen the resample and tighten accuracy at the
    /// cost of a bigger up-front render. Ignored when `cache == Off`.
    pub cache_supersample: f64,
    /// Whether to compute the per-patch confidence (the `Φ`-peakedness stencil).
    /// Off by default: it is an un-cached extra source-render pass per patch
    /// (~⅙ of the cached runtime) and is purely informational — when `false`,
    /// [`NormalRefineResult::confidence`] is `NaN`.
    pub compute_confidence: bool,
    /// Cheaper consensus objective for the coarse-to-fine *search* ranking only
    /// (the bulk of the consensus evaluations). The final pass that picks the
    /// winner and reports `Φ` always uses [`objective`](Self::objective), so the
    /// returned `photoconsistency` stays honest regardless of this knob. `None`
    /// (the default) uses `objective` for the search too; `Some(0)` searches
    /// with [`Objective::MeanPairwise`]; `Some(k)` searches with
    /// `RobustWeighted { iters: k }`. Lowering it trades a little tail accuracy
    /// in the found normal for a faster search. It only *reduces* search cost
    /// relative to a robust [`objective`](Self::objective); under
    /// `objective = MeanPairwise` the final pass is already the cheapest, so the
    /// knob has no benefit (and `Some(k ≥ 1)` would make the search dearer than
    /// the final pass).
    pub search_robust_iters: Option<u32>,
    /// Exponent `p` of the multiplicative **obliquity view-weight** `|v̂·n|^p`
    /// folded into the robust IRLS consensus weights (see
    /// [`obliquity`](super::obliquity), use A). `0` (default) disables it — the
    /// consensus runs exactly as before. `2` is the `cos²θ` foreshortening weight:
    /// it softly down-weights a view the more oblique it sees the surfel, a
    /// continuous replacement for a hard grazing-view cut. Only affects
    /// [`Objective::RobustWeighted`] (the [`Objective::MeanPairwise`] search
    /// ranking ignores it); the default `objective` is robust.
    pub obliquity_weight_power: f64,
    /// Weight `λ` of the additive **fronto-parallel prior** `λ·mean_v (v̂·n)²`
    /// added to a candidate normal's `Φ` when ranking (see
    /// [`obliquity`](super::obliquity), use B). `0` (default) disables it. It
    /// rewards normals that face the observing cameras, supplying the missing
    /// constraint on a low-parallax point (flat `Φ`) so the normal settles
    /// fronto-parallel instead of drifting to a photometrically-equivalent tilt;
    /// wherever real parallax curves `Φ` the (small) prior is overruled. Applied in
    /// the coarse-to-fine search and the final winner pass, but **not** in the
    /// confidence stencil (which reports the data curvature alone).
    pub fronto_prior_weight: f64,
    /// Whether to render the per-patch RGBA representative texture
    /// ([`NormalRefineResult::representative`]) at the found normal. Off by
    /// default: it is one extra full-grid source render per kept view per patch
    /// (the search and scoring only touch the masked common support), so it is
    /// computed only when a caller wants to persist the patch bitmaps. When
    /// `false`, [`NormalRefineResult::representative`] is `None`.
    pub render_bitmap: bool,
}

impl NormalRefineParams {
    /// The consensus objective used to rank candidates during the
    /// coarse-to-fine search. Derived from
    /// [`search_robust_iters`](Self::search_robust_iters); the final winner /
    /// `Φ` pass always uses [`objective`](Self::objective).
    pub(super) fn search_objective(&self) -> Objective {
        match self.search_robust_iters {
            None => self.objective,
            Some(0) => Objective::MeanPairwise,
            Some(iters) => Objective::RobustWeighted { iters },
        }
    }
}

impl Default for NormalRefineParams {
    fn default() -> Self {
        Self {
            angular_range_deg: 25.0,
            init_steps: 7,
            refine_levels: 3,
            objective: Objective::RobustWeighted { iters: 3 },
            window: PatchWindow::GaussianDisk { sigma: 0.6 },
            min_valid_fraction: 0.6,
            min_views: 3,
            sampler: Sampler::Bilinear,
            cache: CacheMode::FrontoParallel,
            cache_supersample: 2.0,
            compute_confidence: false,
            search_robust_iters: None,
            obliquity_weight_power: 0.0,
            fronto_prior_weight: 0.0,
            render_bitmap: false,
        }
    }
}

/// Result of [`refine_patch_normal`].
#[derive(Debug, Clone)]
pub struct NormalRefineResult {
    /// The input patch with its normal replaced by the optimum; `center`,
    /// `half_extent`, and the in-plane convention are preserved (the input
    /// `u_axis` is reprojected onto the new plane, `v = n × u`).
    pub patch: OrientedPatch,
    /// Consensus photoconsistency `Φ` at the returned normal (NaN if it could
    /// not be evaluated).
    pub photoconsistency: f64,
    /// `Φ` at the input normal under the same frozen support (NaN if it could
    /// not be evaluated).
    pub init_photoconsistency: f64,
    /// Number of views kept by the validity gates at the final evaluation.
    pub valid_view_count: u32,
    /// Peakedness of `Φ` at the optimum: the smaller eigenvalue of the
    /// finite-difference curvature of `Φ(δ)`, normalized against the larger
    /// one (see [`grid_confidence`] internals). `0` when the patch was not
    /// refined or `Φ` is flat (e.g. the narrow-baseline degeneracy). `NaN` when
    /// not requested ([`NormalRefineParams::compute_confidence`] is `false`).
    pub confidence: f64,
    /// The canonical robust-template appearance at the found normal: a fused
    /// `R×R` RGBA texture, flat in row-major `(row, col, channel)` order (length
    /// `R·R·4`). RGB is the cross-view fused colour (a robust IRLS-weighted mean
    /// under [`Objective::RobustWeighted`], an unweighted mean under
    /// [`Objective::MeanPairwise`]); the alpha channel is a per-pixel cross-view
    /// agreement confidence (`0` where no kept view covers the pixel). `None`
    /// when [`NormalRefineParams::render_bitmap`] is `false`, or the patch was
    /// not refined (too few valid views).
    pub representative: Option<Vec<u8>>,
}

/// Windowed norm² below which a colour channel counts as flat (no texture
/// signal); flat channels are dropped from the consensus so the z-normalized
/// identity stays well-defined.
pub(in crate::patch) const FLAT_NORM_SQ_EPS: f64 = 1e-6;

/// Minimum number of commonly-valid pixels for a support to be scoreable.
pub(super) const MIN_MASK_PIXELS: usize = 8;

/// Minimum robust *effective* view count `1/Σwᵢ²` for the weighted consensus to
/// be meaningful (a pair) and avoid the `0/0` at `Σwᵢ² → 1`. This is a degeneracy
/// floor only — the *count* of kept views is gated separately by `min_views`, so
/// a clean `V == min_views` track (weights near- but not exactly uniform, hence
/// `1/Σwᵢ² < V`) still scores.
pub(super) const MIN_EFFECTIVE_VIEWS: f64 = 2.0;

/// `remap_aniso` sample cap along the major axis.
pub(super) const MAX_ANISOTROPY: u32 = 16;
