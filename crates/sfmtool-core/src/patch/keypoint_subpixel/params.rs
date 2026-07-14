// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Tunables, refresh-granularity config, and result types for subpixel keypoint
//! refinement, split out of the Gauss–Newton orchestration ([`super`]).
//!
//! The render/window knobs on [`KeypointSubpixelParams`] mirror
//! [`KeypointLocalizeParams`](crate::patch::keypoint_localize::KeypointLocalizeParams).

use crate::patch::normal_refine::{PatchWindow, Sampler};

/// Within-sweep granularity of consensus refresh — the spec's "Consensus
/// refresh granularity" axis. Across sweeps the consensus is always rebuilt
/// from scratch (the [`KeypointSubpixelParams::max_outer_sweeps`] loop); this
/// enum chooses what happens **inside** a sweep as views move.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ConsensusRefresh {
    /// `T` is held fixed for the duration of a sweep. Every view refines
    /// against the same consensus the sweep was built with; the next sweep
    /// rebuilds `T` from the moved views. With `max_outer_sweeps = 1` this is
    /// the spec's **single-pass frozen** variant; with `> 1` it is the
    /// **per-sweep refresh** variant. The default — preserves existing
    /// behavior.
    #[default]
    PerSweep,
    /// The spec's **per-move (Gauss–Seidel) incremental** variant. The
    /// consensus is maintained as a running weighted sum
    /// `S = Σ_v w_v · ẑ_v` of the z-normalized view cores; when view `v` moves
    /// from `δ` to `δ'`, `S += w_v · (ẑ_v' − ẑ_v)` and `T = normalize(S)`. The
    /// next view's GN solve aligns to a consensus that already reflects the
    /// previous view's move. The IRLS view weights are refreshed at the lower
    /// **per-sweep** frequency (the spec's two-frequency design); within a
    /// sweep the weights are held fixed so the delta update is exact.
    ///
    /// **Shared (not LOO) consensus.** The spec lists leave-one-out as the
    /// "free with the running sum" bonus, avoiding the self-pollution where a
    /// view aligns to a `T` that includes itself. Direct measurement on
    /// `dino_dog_toy` (see `RunningConsensus::write_shared_template`) found
    /// LOO regressed mean ECC (0.82 vs 0.87 for shared) at the small view
    /// counts of real tracks: the chain of LOO updates amplifies drift more
    /// than the self-pollution it removes. PerMove therefore uses shared `T`.
    ///
    /// **Limitation at `N = 2` views.** A direct consequence of the shared-`T`
    /// choice is that with only two views one view's own contribution dominates
    /// the consensus the other view aligns to. On the minimal 2-view planted-
    /// offset fixture PerMove underestimates the relative offset by ~3%
    /// (`per_move_two_views_known_underestimate_at_n2` pins the actual bias);
    /// PerSweep at `N = 2` is unaffected. **`N ≥ 3` is recommended** when
    /// opting into PerMove.
    PerMove,
}

/// Tunables for [`refine_patch_keypoints`](super::refine_patch_keypoints).
///
/// The render/window knobs mirror
/// [`KeypointLocalizeParams`](crate::patch::keypoint_localize::KeypointLocalizeParams)
/// so the consensus is built on the same conventions as the discrete search that
/// typically seeds this refiner.
#[derive(Debug, Clone)]
pub struct KeypointSubpixelParams {
    /// The `R×R` patch grid the consensus and per-view ECC are scored on.
    pub resolution: u32,
    /// Per-pixel scoring weight / support.
    pub window: PatchWindow,
    /// How to sample the source pyramids when rendering patch tiles. The GN inner
    /// step uses the **value+gradient** variant of the chosen sampler — one render
    /// returns `(value, ∂I/∂x, ∂I/∂y)` per support pixel and channel — composed
    /// per-pixel with the warp Jacobian `J = WarpMap::get_jacobian(col, row)` to
    /// give the analytic `∂I/∂δ` the GN normal equations need. The
    /// [`Sampler::Anisotropic`] and [`Sampler::BilinearMip`] gradients are
    /// computed at the same LOD(s) / footprint as the value (per-level bilinear
    /// gradient **divided** by the level's `2^level` to convert from level-pixel
    /// to level-0 source-pixel coords; the anisotropic path additionally blends
    /// with the same `frac` the value uses), so value and gradient stay
    /// LOD-consistent.
    pub sampler: Sampler,
    /// IRLS reweighting passes for the robust consensus.
    pub robust_iters: u32,
    /// Maximum **outer sweeps** of the alternating loop (refresh consensus → move
    /// every view). `1` is the spec's single-pass-frozen variant (build `T` once at
    /// the seed, hold it fixed) — the cheapest, and the default. `> 1` is the
    /// per-sweep-refresh variant: each subsequent sweep re-renders the views at
    /// their current offsets and rebuilds `T` from those. The outer loop early-exits
    /// when the mean per-view move of a sweep falls below
    /// [`outer_convergence_px`](Self::outer_convergence_px).
    ///
    /// The "never worse than the seed" guard is **per sweep**: within a sweep each
    /// accepted step is non-decreasing in the ECC score against THAT sweep's `T`.
    /// Across sweeps `T` changes, so the final score against the final `T` is not
    /// bit-bounded below by the seed score against the seed `T`. With
    /// `max_outer_sweeps = 1` (the default) the two coincide, so the guarantee is
    /// the spec's strict form.
    pub max_outer_sweeps: u32,
    /// Stop the outer (consensus-refresh) loop once the mean per-view move across a
    /// completed sweep falls below this many patch-grid px. Ignored when
    /// `max_outer_sweeps == 1`.
    pub outer_convergence_px: f64,
    /// Maximum forward-additive Gauss–Newton steps per view per outer sweep.
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
    /// Within-sweep consensus refresh granularity (the spec's "Consensus refresh
    /// granularity" choice). [`ConsensusRefresh::PerSweep`] (default) holds `T`
    /// fixed for the sweep — preserves existing behavior. [`ConsensusRefresh::PerMove`]
    /// is the spec's Gauss–Seidel incremental variant: after each view moves,
    /// the consensus is delta-updated from the running weighted sum
    /// `S = Σ_v w_v · ẑ_v` so the next view aligns to a `T` that already
    /// reflects the previous move. IRLS weights are still refreshed only at the
    /// per-sweep boundary (the spec's two-frequency design — fixed weights make
    /// the delta exact). Per-move uses the **shared** consensus `normalize(S)`;
    /// the spec's leave-one-out alternative was measured-and-rejected (regressed
    /// mean ECC on real tracks — see [`ConsensusRefresh::PerMove`]).
    /// **Limitation:** at `N = 2` views PerMove underestimates the relative
    /// offset by ~3% (the moved view's own contribution dominates the shared
    /// `T`); `N ≥ 3` is recommended.
    pub consensus_refresh: ConsensusRefresh,
    /// Also fuse each point's **representative RGBA texture** at the FINAL
    /// per-view keypoints (see [`KeypointRefinement::representative`]): after the
    /// last sweep the views are re-rendered at their final offsets, the final IRLS
    /// view weights are rebuilt from those cores, and the kept views are rendered
    /// full-grid
    /// ([`PatchViewStack`](crate::patch::normal_refine::PatchViewStack)) at the
    /// final keypoints and fused (weighted-mean RGB + agreement·coverage alpha,
    /// exactly the normal-refine representative). Points at infinity go through the
    /// same path (`w = 0` rendering is first-class here). Costs one extra full-grid
    /// source render per live view per point, so it is off by default.
    pub render_bitmaps: bool,
}

impl Default for KeypointSubpixelParams {
    fn default() -> Self {
        Self {
            resolution: 24,
            window: PatchWindow::GaussianDisk { sigma: 0.6 },
            sampler: Sampler::Bilinear,
            robust_iters: 3,
            max_outer_sweeps: 1,
            outer_convergence_px: 0.005,
            max_gn_steps: 10,
            convergence_px: 0.01,
            max_offset_px: 2.0,
            line_search_shrink: 0.5,
            line_search_max: 8,
            consensus_refresh: ConsensusRefresh::PerSweep,
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
    /// refined core against the frozen consensus). `NaN` when the view could not be
    /// scored (e.g. fewer than two views, so no consensus was built).
    pub scores: Vec<f64>,
    /// The point's fused representative RGBA texture (`R·R·4`, row-major), rendered
    /// at the **final** per-view keypoints and fused with the final IRLS view
    /// weights — only when [`KeypointSubpixelParams::render_bitmaps`] is set.
    /// `None` when the point produced no valid cross-view consensus (fewer than
    /// two views survive the projection gate / render at their final offsets) —
    /// the uniform "culled point" signal, for finite and infinity points alike.
    pub representative: Option<Vec<u8>>,
}
