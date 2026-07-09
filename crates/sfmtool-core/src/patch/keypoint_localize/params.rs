// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Tunables, strategy config, and result types for patch-keypoint localization.
//!
//! Split out of the localization orchestration ([`super`]); the render/window
//! knobs on [`KeypointLocalizeParams`] mirror
//! [`NormalRefineParams`](crate::patch::normal_refine::NormalRefineParams).

use crate::patch::normal_refine::{PatchWindow, Sampler};

/// How the per-(view, round) shift grid is traversed inside `search_shift`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SearchStrategy {
    /// "+"-descent on the integer shift grid: starts at `(dy, dx) = (0, 0)`,
    /// evaluates the 4 axis neighbors per step, moves to the best improver,
    /// and stops when no neighbor beats the current cell. Each cell is scored
    /// at most once via a per-cell ZNCC kernel
    /// ([`score_cell_one_channel`](super::kernels::score_cell_one_channel) —
    /// AVX2-gather when available, scalar otherwise); the visited cache stores
    /// `(n, s1, s2, ginv)` per cell. The final-cell separable parabolic
    /// sub-pixel fit reuses the 4 cardinal-neighbor cells already in the cache
    /// (each was evaluated to discover the STOP condition).
    ///
    /// **The default.** Late-round congealing typically leaves a view's
    /// argmax at `(0, 0)`, so the descent stops after 5 cell scores; on the
    /// dino dataset this drives per-`search_shift` cost from ~145 µs
    /// ([`Exhaustive`](Self::Exhaustive)) to ~32 µs and total localize wall
    /// down ~1.9× at comparable accuracy (median per-observation keypoint
    /// shift vs `Exhaustive` is ~0.05 px, 91 % of observations within 1 px on
    /// dino). The accuracy tail is the local-optima failure mode of any
    /// descent on a multi-modal ZNCC landscape; pick
    /// [`Exhaustive`](Self::Exhaustive) when the tail matters or for
    /// bit-equivalent comparisons.
    #[default]
    PlusDescent,
    /// Score every cell of the `(2·margin+1) × (2·margin+1)` shift grid via
    /// the hand-rolled SIMD SAXPY accumulator
    /// ([`compute_channel_grids`](super::kernels::compute_channel_grids)), then
    /// argmax + separable parabolic. The original whole-grid path; retained
    /// as the global-argmax fallback (no local-optima risk) and as the
    /// per-cell reference both equivalence tests and the per-cell ZNCC kernel
    /// ([`score_cell_one_channel`](super::kernels::score_cell_one_channel))
    /// check against.
    Exhaustive,
}

/// Tunables for [`localize_patch_keypoints`](super::localize_patch_keypoints).
///
/// The render/window knobs mirror
/// [`NormalRefineParams`](crate::patch::normal_refine::NormalRefineParams) so the
/// consensus is built on the same conventions as refinement and selection.
#[derive(Debug, Clone)]
pub struct KeypointLocalizeParams {
    /// Maximum congealing rounds; the loop stops early once the mean per-view
    /// position change of a round falls below
    /// [`convergence_px`](Self::convergence_px).
    pub max_iters: u32,
    /// Max total per-view drift from the point's projection (patch-grid px). The
    /// accumulated offset is clipped to `±search` each round, bounding runaway, and
    /// the context tile is rendered this much larger than the scored core so the
    /// shift search can slide without running off the edge. (The drift, the
    /// `max_shift_px` gate, and the reported offset are all anchored at the
    /// projection `project_i(X_p)`, i.e. `acc = 0`.)
    pub search: f64,
    /// Drop a view whose refined keypoint sits more than this many *source-image*
    /// pixels from the point's projection `project_i(X_p)` (an absolute distance,
    /// not the per-round move).
    pub max_shift_px: f64,
    /// Drop a view whose leave-one-out ZNCC falls below this fraction of the
    /// views' *median* leave-one-out ZNCC (relative, so a uniformly low-texture
    /// patch is not over-dropped).
    pub min_relative_zncc: f64,
    /// Grazing cutoff: drop a view whose viewing ray is near-parallel to the
    /// patch plane (`|d̂ · n̂|` below this), where the in-plane anchor is
    /// ill-conditioned and the view would only contaminate the consensus.
    pub min_grazing_cos: f64,
    /// The `R×R` patch grid the consensus and per-view ZNCC are scored on.
    pub resolution: u32,
    /// Per-pixel scoring weight / support.
    pub window: PatchWindow,
    /// How to sample the source pyramids when rendering patch tiles.
    pub sampler: Sampler,
    /// IRLS reweighting passes for the robust consensus.
    pub robust_iters: u32,
    /// Convergence threshold: stop once a round's mean **round-over-round
    /// change** of the per-view refined positions (integer accumulator +
    /// sub-pixel residual, this round vs the previous one) is below this many
    /// patch-grid px.
    pub convergence_px: f64,
    /// Search-resolution multiplier `m` (default `1.0`, a no-op). The discrete
    /// cross-view search is run at resolution `R_s = round(m·R)`: the per-view
    /// cache, window support, and shift grid are all built at `R_s`, so one
    /// integer step in the search grid is `1/m` patch-grid px. The found shift is
    /// scaled by `1/m` back to patch-grid px before it moves the accumulator and
    /// is reported. `m < 1` smooths the correlation surface for a speed fallback;
    /// `m > 1` resolves sub-pixel offsets directly on a finer grid. See
    /// `specs/core/keypoint-localization-search-cache.md`.
    pub search_resolution_multiplier: f32,
    /// Per-(view, round) shift-grid traversal — see [`SearchStrategy`].
    /// Defaults to [`SearchStrategy::PlusDescent`].
    pub search_strategy: SearchStrategy,
}

impl Default for KeypointLocalizeParams {
    fn default() -> Self {
        Self {
            max_iters: 5,
            search: 6.0,
            max_shift_px: 3.0,
            min_relative_zncc: 0.7,
            min_grazing_cos: 0.1,
            resolution: 24,
            window: PatchWindow::GaussianDisk { sigma: 0.6 },
            sampler: Sampler::Bilinear,
            robust_iters: 3,
            convergence_px: 0.05,
            search_resolution_multiplier: 1.0,
            search_strategy: SearchStrategy::PlusDescent,
        }
    }
}

/// The localized keypoints for one point — parallel arrays over the **kept**
/// views (a subset of the input view set, in the input's order; grazing /
/// out-of-frame / large-shift / low-agreement views are dropped in-loop).
#[derive(Debug, Clone, Default)]
pub struct KeypointLocalization {
    /// The kept image indices (into the `views` slice), a subset of the input
    /// view set preserving its order.
    pub views: Vec<u32>,
    /// The refined keypoint `project_i(X_p) + δ_j` per kept view, in source-image
    /// pixels (`[x, y]`), parallel to [`views`](Self::views).
    pub keypoints: Vec<[f64; 2]>,
    /// Per kept view, the keypoint's offset from the point's projection
    /// `project_i(X_p)` in source-image pixels (`|δ_j|`), parallel to
    /// [`views`](Self::views).
    pub offsets_px: Vec<f64>,
    /// Per kept view, the leave-one-out ZNCC against the other views' consensus
    /// from the last round that scored it (the integer-peak value of that round's
    /// shift search), parallel to [`views`](Self::views). `NaN` for a view no round
    /// ever scored — e.g. a lone input view, or a view kept by the early
    /// "fewer than two views remain" exit before any consensus was built.
    pub loo_zncc: Vec<f64>,
    /// Congealing rounds actually executed (`<= max_iters`; `0` when the input
    /// had fewer than two views and the loop never ran). Diagnostic: lets tests
    /// and callers observe the `convergence_px` early exit directly.
    pub rounds: u32,
}
