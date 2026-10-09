// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Tunables and result types for subpixel keypoint refinement, split out of the Gauss–Newton orchestration ([`super`]).
//!
//! The render/window knobs on [`KeypointSubpixelParams`] mirror
//! [`KeypointLocalizeParams`](crate::patch::keypoint_localize::KeypointLocalizeParams).

use crate::patch::normal_refine::{PatchWindow, SamplerChoice};

/// Tunables for [`refine_patch_keypoints`](super::refine_patch_keypoints).
///
/// The render/window knobs mirror
/// [`KeypointLocalizeParams`](crate::patch::keypoint_localize::KeypointLocalizeParams)
/// so the template is built on the same conventions as the discrete search that
/// typically seeds this refiner.
#[derive(Debug, Clone)]
pub struct KeypointSubpixelParams {
    /// The `R×R` patch grid the template and per-view ECC are scored on.
    pub resolution: u32,
    /// Per-pixel scoring weight / support.
    pub window: PatchWindow,
    /// Which sampler renders each view's tiles: the sampler rule by default
    /// ([`SamplerChoice::per_view`]). Under the rule each view's sampler is
    /// chosen once, at its seed keypoint, from the patch re-anchored there at
    /// the patch resolution ([`SamplerChoice::for_observation`]), and every
    /// render of the view as the refiner moves it uses that sampler. The GN inner
    /// step uses the **value+gradient** variant of the chosen sampler — one render
    /// returns `(value, ∂I/∂x, ∂I/∂y)` per support pixel and channel — composed
    /// per-pixel with the warp Jacobian `J = WarpMap::get_jacobian(col, row)` to
    /// give the analytic `∂I/∂δ` the GN normal equations need. The
    /// `Sampler::Anisotropic` and `Sampler::BilinearMip` gradients are
    /// computed at the same LOD(s) / footprint as the value (per-level bilinear
    /// gradient **divided** by the level's `2^level` to convert from level-pixel
    /// to level-0 source-pixel coords; the anisotropic path additionally blends
    /// with the same `frac` the value uses), so value and gradient stay
    /// LOD-consistent.
    pub sampler: SamplerChoice,
    /// IRLS reweighting passes for the fused mean, which is the template where
    /// the point has no reference observation, and the stored bitmap there.
    pub robust_iters: u32,
    /// Maximum forward-additive Gauss–Newton steps per view.
    pub max_gn_steps: u32,
    /// Stop a view's GN solve once the accepted step magnitude falls below this
    /// many patch-grid px.
    pub convergence_px: f64,
    /// Maximum total per-view drift from the seed, in patch-grid px. A step that
    /// would carry `|δ − δ_seed|` past this is rejected (keeps the local-refiner
    /// contract; the seed must already be in the basin).
    pub max_offset_px: f64,
    /// Backtracking line-search shrink factor (`0 < γ < 1`) and attempt cap: a
    /// rejected step is retried at `γ·step`, up to [`line_search_max`](Self::line_search_max).
    pub line_search_shrink: f64,
    /// Maximum backtracking attempts before a GN step is abandoned (the seed/δ is
    /// kept for that step).
    pub line_search_max: u32,
    /// Also render each point's **stored bitmap** (see
    /// [`KeypointRefinement::representative`]): the reference observation's
    /// `R×R` tile at its keypoint, which the refinement does not move, named in
    /// [`KeypointRefinement::reference`] ([`crate::patch::stored_bitmap`]).
    /// Where the point has no reference observation and the reference-view
    /// rule picks none it would store
    /// ([`ReferenceRender::stored_reference`](crate::patch::stored_bitmap::ReferenceRender::stored_reference)),
    /// the views are re-rendered at their final offsets, IRLS view weights
    /// built from those cores, and the views rendered full-grid and fused
    /// (weighted-mean RGB + agreement·coverage alpha). Points at infinity go
    /// through the same path (`w = 0` rendering is first-class here). Off by
    /// default.
    pub render_bitmaps: bool,
}

impl Default for KeypointSubpixelParams {
    fn default() -> Self {
        Self {
            resolution: 24,
            window: PatchWindow::GaussianDisk { sigma: 0.6 },
            sampler: SamplerChoice::per_view(),
            robust_iters: 3,
            max_gn_steps: 10,
            convergence_px: 0.01,
            max_offset_px: 2.0,
            line_search_shrink: 0.5,
            line_search_max: 8,
            render_bitmaps: false,
        }
    }
}

/// The refined keypoints for one point — parallel arrays over the views, in the
/// **input order**. The one membership change is the projection gate: a view in
/// which the patch centre fails to project (behind the camera or outside the
/// frame) has no projection-anchored offset to report, so it is dropped from the
/// returned set (matching the sibling localizer). Otherwise the set is preserved
/// — a view whose GN solve fails the guard stays at its seed.
#[derive(Debug, Clone, Default)]
pub struct KeypointRefinement {
    /// The image indices, in the input `view_set` order (deduplicated; a repeated
    /// image is refined once).
    pub views: Vec<u32>,
    /// The refined keypoint `project_i(X_p) + δ_v` per view, in source-image
    /// pixels (`[x, y]`), parallel to [`views`](Self::views).
    pub keypoints: Vec<[f64; 2]>,
    /// Per view, the keypoint's offset from the point's projection
    /// `project_i(X_p)` in source-image pixels, parallel to [`views`](Self::views).
    pub offsets_px: Vec<f64>,
    /// Per view, the final ECC score (channel-averaged windowed ZNCC of the
    /// refined core against the template): `1.0` for the reference
    /// observation, whose render the template is. `NaN` when the view could not
    /// be scored (fewer than two views, or no template rendered).
    pub scores: Vec<f64>,
    /// The point's stored bitmap (`R·R·4` RGBA, row-major), only when
    /// [`KeypointSubpixelParams::render_bitmaps`] is set: the tile of the
    /// reference observation [`Self::reference`] names, at its keypoint, or the
    /// fused mean of the views at their final keypoints where there is none.
    /// `None` when fewer than two views survive the projection gate, or when
    /// there is no reference and fewer than two views render in frame at their
    /// final offsets for the fused mean -- the uniform "culled point" signal,
    /// for finite and infinity points alike.
    pub representative: Option<Vec<u8>>,
    /// The reference observation the views were aligned to, as an index into
    /// [`Self::views`]: the view whose tile [`Self::representative`] is. `None`
    /// where the template was the fused mean, or there was none.
    pub reference: Option<usize>,
}
