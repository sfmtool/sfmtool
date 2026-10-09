// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Tunables, strategy config, and result types for patch-keypoint localization.
//!
//! Split out of the localization orchestration ([`super`]); the render/window
//! knobs on [`KeypointLocalizeParams`] mirror
//! [`NormalRefineParams`](crate::patch::normal_refine::NormalRefineParams).

use crate::patch::normal_refine::{PatchWindow, SamplerChoice};

/// How each view's shift grid is traversed when it is aligned to the template.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SearchStrategy {
    /// "+"-descent on the integer shift grid: starts at the view's starting
    /// keypoint, evaluates the 4 axis neighbours per step, moves to the best
    /// improver, and stops when no neighbour beats the current cell. Each cell
    /// is scored at most once via a per-cell ZNCC kernel
    /// (`score_cell_one_channel`, AVX2-gather when available, scalar
    /// otherwise); the visited cache stores the combined ZNCC per cell. The
    /// final cell's sub-pixel fit is a quadratic over its 3×3 neighbourhood,
    /// which reuses the 4 cardinal neighbours already in the cache and scores
    /// the 4 diagonal ones.
    ///
    /// **The default.** It climbs to the correlation peak nearest the starting
    /// keypoint, which is the evidence for which of several similar peaks is
    /// meant. On the seoul_bull and kerry_park ground truths it places views
    /// closer to the truth than [`Exhaustive`](Self::Exhaustive) from starting
    /// keypoints within 1 px of it, and is the faster of the two; see
    /// `specs/core/patch/patch-keypoint-localization.md`, "How the alignment
    /// was measured".
    #[default]
    PlusDescent,
    /// Score every cell of the `(2·margin+1) × (2·margin+1)` shift grid via
    /// the hand-rolled SIMD SAXPY accumulator (`compute_channel_grids`), then
    /// argmax + the same 3×3 quadratic sub-pixel fit. The global maximum over the window; it
    /// recovers from a starting keypoint 2 to 3 px off better than the
    /// descent, and loses to it nearer the truth, where a side peak of a
    /// repeated texture can score higher than the true one.
    Exhaustive,
}

/// Tunables for [`localize_patch_keypoints`](super::localize_patch_keypoints).
///
/// The render/window knobs mirror
/// [`NormalRefineParams`](crate::patch::normal_refine::NormalRefineParams) so the
/// template is built on the same conventions as refinement and selection.
#[derive(Debug, Clone)]
pub struct KeypointLocalizeParams {
    /// The reach of each view's search around its starting keypoint, in
    /// patch-grid px: the context tile is rendered this much larger than the
    /// scored core on every side, and the shift is searched over `±search`.
    pub search: f64,
    /// Drop a view whose keypoint sits more than this many *source-image*
    /// pixels from the point's projection `project_i(X_p)` (an absolute
    /// distance, not the move from the starting keypoint). Never applied to the
    /// reference observation, which is not moved.
    pub max_shift_px: f64,
    /// Drop a view whose ZNCC against the template falls below this fraction
    /// of the median over the point's other views (the reference left out),
    /// so a uniformly low-texture patch is not over-dropped. `0.0` (or a
    /// non-finite value) disables it exactly.
    pub min_relative_zncc: f64,
    /// Drop a view whose ZNCC against the template is finite and **below this
    /// absolute floor**, however many views remain. `0.0` (or a non-finite
    /// value) disables it exactly.
    pub min_absolute_zncc: f64,
    /// Drop a view whose **own** rendered core tile does not pin a 2D position:
    /// its [ZNCC self-similarity radius](crate::patch::self_similarity), how far
    /// the core can slide over itself and still match itself as well as a true
    /// match between two views would, is above this bar, in patch-grid px. A
    /// flat or edge-only view matches itself along the edge or everywhere, so
    /// its ZNCC to anything cannot place it. Read once per view, before its
    /// search, on the `R×R` core of the tile already rendered for that view at
    /// its starting keypoint, the overlap way: no pixel of the tile around the
    /// core enters it. Not applied to the reference observation, which is not
    /// moved.
    ///
    /// A view passes when its radius is at or below the bar; a `NaN` radius
    /// fails. The radius reads at most the default
    /// [`SelfSimilarityParams::max_radius`](crate::patch::self_similarity::SelfSimilarityParams::max_radius)
    /// (`3`), which stands for "that far or further", so a bar at or above it
    /// turns nothing out. `0.0` (or a non-finite value) disables the gate
    /// exactly. See [`Self::admits_member_zncc_self_similarity_radius`].
    pub max_member_zncc_self_similarity_radius: f64,
    /// Grazing cutoff: drop a view whose viewing ray is near-parallel to the
    /// patch plane (`|d̂ · n̂|` below this), where the in-plane anchor is
    /// ill-conditioned.
    pub min_grazing_cos: f64,
    /// The `R×R` patch grid the template and the per-view ZNCC are scored on.
    pub resolution: u32,
    /// Per-pixel scoring weight / support.
    pub window: PatchWindow,
    /// Which sampler renders each view's tiles: the sampler rule by default
    /// ([`SamplerChoice::per_view`]). Under the rule each view's sampler is
    /// chosen once, at its starting keypoint, from the patch re-anchored there
    /// at the patch resolution ([`SamplerChoice::for_observation`]), and every
    /// tile of the view, the wider context tile included, is rendered with it.
    pub sampler: SamplerChoice,
    /// IRLS reweighting passes for the fused mean that stands as the template
    /// where the reference-view rule picks no reference it would store.
    pub robust_iters: u32,
    /// How each view's shift grid is traversed; see [`SearchStrategy`].
    pub search_strategy: SearchStrategy,
}

/// The default [`KeypointLocalizeParams::max_member_zncc_self_similarity_radius`]:
/// `2.5` grid px, the same bar as the bench's
/// [`BENCH_MAX_ZNCC_SELF_SIMILARITY_RADIUS`](crate::bench::BENCH_MAX_ZNCC_SELF_SIMILARITY_RADIUS).
///
/// The user chose `2.5` from a sweep on seoul_bull and kerry_park (see
/// `specs/core/patch/patch-keypoint-localization.md`, "The member gate's
/// default", which gives the sweep's figures). The bar sits under the largest
/// radius read (`3`), so it rejects flat and edge-only views, which read `3`.
pub const DEFAULT_MAX_MEMBER_ZNCC_SELF_SIMILARITY_RADIUS: f64 = 2.5;

impl KeypointLocalizeParams {
    /// Whether [`Self::max_member_zncc_self_similarity_radius`] is on: finite
    /// and above `0`.
    pub fn member_self_similarity_gate_is_on(&self) -> bool {
        let bar = self.max_member_zncc_self_similarity_radius;
        bar.is_finite() && bar > 0.0
    }

    /// Whether a view whose own core reads `radius` passes the member gate:
    /// always when the gate is off, otherwise when `radius` is at or below
    /// [`Self::max_member_zncc_self_similarity_radius`]. A `NaN` radius fails
    /// an active gate.
    pub fn admits_member_zncc_self_similarity_radius(&self, radius: f64) -> bool {
        !self.member_self_similarity_gate_is_on()
            || radius <= self.max_member_zncc_self_similarity_radius
    }
}

impl Default for KeypointLocalizeParams {
    fn default() -> Self {
        Self {
            search: 6.0,
            max_shift_px: 3.0,
            min_relative_zncc: 0.7,
            min_absolute_zncc: 0.5,
            max_member_zncc_self_similarity_radius: DEFAULT_MAX_MEMBER_ZNCC_SELF_SIMILARITY_RADIUS,
            min_grazing_cos: 0.1,
            resolution: 24,
            window: PatchWindow::GaussianDisk { sigma: 0.6 },
            sampler: SamplerChoice::per_view(),
            robust_iters: 3,
            search_strategy: SearchStrategy::PlusDescent,
        }
    }
}

/// The localized keypoints for one point -- parallel arrays over the **kept**
/// views (a subset of the input view set, in the input's order; grazing,
/// out-of-frame, unlocalizable, large-shift and low-agreement views are
/// dropped).
///
/// The reference observation is always kept when it renders, at the keypoint
/// it was given. Every other view can be dropped, so the kept set can be a
/// single view, or none where the reference itself could not be rendered,
/// for the caller's `min_views` cull to remove.
#[derive(Debug, Clone, Default)]
pub struct KeypointLocalization {
    /// The kept image indices (into the `views` slice), a subset of the input
    /// view set preserving its order.
    pub views: Vec<u32>,
    /// The keypoint per kept view, in source-image pixels (`[x, y]`), parallel
    /// to [`views`](Self::views). The reference's is its starting keypoint.
    pub keypoints: Vec<[f64; 2]>,
    /// Per kept view, the keypoint's offset from the point's projection
    /// `project_i(X_p)` in source-image pixels, parallel to
    /// [`views`](Self::views).
    pub offsets_px: Vec<f64>,
    /// Per kept view, its plain ZNCC against the template at the search's
    /// integer peak, parallel to [`views`](Self::views): `1.0` for the
    /// reference observation, whose render the template is, and `NaN` for a
    /// view that was not searched because there was no template.
    pub zncc: Vec<f64>,
    /// The image index of the reference observation the views were aligned
    /// to. `None` where there was none to align to: the reference-view rule
    /// picked no reference it would store, so the template was the fused mean
    /// of the views, or nothing rendered to align to.
    pub reference: Option<u32>,
}
