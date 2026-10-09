// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Opt-in phase timing for keypoint localization.
//!
//! Set `SFMTOOL_PROFILE=1` to accumulate per-phase wall time (atomic nanosecond
//! counters, summed across rayon threads) during
//! [`localize_patch_cloud_keypoints`](super::localize_patch_cloud_keypoints); a
//! summary is printed to stderr when the batch finishes. With the variable unset
//! the timers compile to a single branch on a cached flag, so the hot path is
//! unaffected. Mirrors `normal_refine::prof`.
//!
//! Phase times are *thread-summed* (CPU-seconds, not wall-clock): with N rayon
//! threads busy, one wall second accumulates up to N phase-seconds. Shares of the
//! total are therefore meaningful; absolute values exceed wall time.

use std::sync::atomic::{AtomicU64, Ordering};

pub use crate::profiling::{count, enabled, Phase};
use crate::profiling::{report_overhead, report_phases, reset_all, PATCH_ROWS};

// Enclosing phases (overlap the leaves; TOTAL is the 100% denominator).
/// Whole `localize_patch_keypoints` calls.
pub static TOTAL: Phase = Phase::new("localize_total");

// Leaf phases (non-overlapping; they partition the bulk of TOTAL).
/// Per-point decision of what the views are aligned to: nothing where the
/// caller names the reference, else the reference-view rule over the renders
/// at the starting keypoints, and the fused mean where the rule picks no
/// reference it would store. Its renders are timed here, not in [`RENDER`].
pub static REFERENCE: Phase = Phase::new("reference");
/// Per-view context tile render (`render_context`), one per view, and the
/// reference's own tile.
pub static RENDER: Phase = Phase::new("render_context");
/// Sub-phase of [`RENDER`]: `WarpMap::from_patch` -- the per-pixel camera
/// projection of the context tile.
pub static RENDER_PROJECT: Phase = Phase::new("render_project");
/// Sub-phase of [`RENDER`]: `compute_svd` -- the per-pixel anisotropic Jacobian
/// SVD (anisotropic sampler only).
pub static RENDER_SVD: Phase = Phase::new("render_svd");
/// Sub-phase of [`RENDER`]: the resample into the tile
/// (`remap_aniso_with_pyramid` / `remap_bilinear`).
pub static RENDER_REMAP: Phase = Phase::new("render_remap");
/// Sub-phase of [`RENDER`]: per-channel mean accumulation over the rendered tile.
pub static RENDER_MEAN: Phase = Phase::new("render_mean");
/// Sub-phase of [`RENDER`]: building the centered/padded planes + valid/invalid
/// masks.
pub static RENDER_CENTER: Phase = Phase::new("render_center");
/// Per-view sub-pixel translation search against the template
/// (`search_shift` / `search_shift_plus_descent`).
pub static SEARCH: Phase = Phase::new("search_shift");
/// Sub-phase of [`SEARCH`]: the per-channel correlation-grid accumulation
/// (`compute_channel_grids`, or the per-cell kernel under the descent).
/// Includes the validity-count pass on the invalidity plane.
pub static SEARCH_ACC: Phase = Phase::new("search_acc");
/// Sub-phase of [`SEARCH`]: the per-channel grid combine into the per-shift ZNCC
/// (denominator + numerator-fold) and the cross-channel sum into the combined
/// grid. Captures the non-vectorized "after the inner loop" cost.
pub static SEARCH_COMBINE: Phase = Phase::new("search_combine");
/// Sub-phase of [`SEARCH`]: the argmax pass + separable parabolic sub-pixel fit
/// on the combined grid.
pub static SEARCH_ARGMAX: Phase = Phase::new("search_argmax");

// Event counters (no time attached).
/// Context tile renders performed (one per view, and the reference's tile).
pub static N_RENDER: AtomicU64 = AtomicU64::new(0);
/// Sub-pixel shift searches (one per view the gates let through).
pub static N_SEARCH: AtomicU64 = AtomicU64::new(0);
/// Cells scored by [`SearchStrategy::PlusDescent`](super::SearchStrategy::PlusDescent)
/// (one per visited-cache **miss** inside `search_shift_plus_descent`; visited-
/// cache **hits** do not count, since they re-use a prior cell's score). Always
/// `0` for [`SearchStrategy::Exhaustive`](super::SearchStrategy::Exhaustive) --
/// the SAXPY accumulator scores all `(2·margin+1)²` cells in one streaming
/// pass and has no per-cell event. Average cells per search call is
/// `N_CELLS / N_SEARCH`.
pub static N_CELLS: AtomicU64 = AtomicU64::new(0);
/// Views dropped by the member self-similarity gate
/// ([`max_member_zncc_self_similarity_radius`](super::KeypointLocalizeParams::max_member_zncc_self_similarity_radius)),
/// summed over points -- their own tile pins no 2D position.
pub static N_DROP_UNLOCALIZABLE: AtomicU64 = AtomicU64::new(0);
/// Views dropped by [`max_shift_px`](super::KeypointLocalizeParams::max_shift_px).
pub static N_DROP_SHIFT: AtomicU64 = AtomicU64::new(0);
/// Views dropped by the absolute floor
/// ([`min_absolute_zncc`](super::KeypointLocalizeParams::min_absolute_zncc)).
pub static N_DROP_ABS_ZNCC: AtomicU64 = AtomicU64::new(0);
/// Views dropped by the relative bar
/// ([`min_relative_zncc`](super::KeypointLocalizeParams::min_relative_zncc)).
pub static N_DROP_REL_ZNCC: AtomicU64 = AtomicU64::new(0);

const PHASES: [&Phase; 12] = [
    &TOTAL,
    &REFERENCE,
    &RENDER,
    &RENDER_PROJECT,
    &RENDER_SVD,
    &RENDER_REMAP,
    &RENDER_MEAN,
    &RENDER_CENTER,
    &SEARCH,
    &SEARCH_ACC,
    &SEARCH_COMBINE,
    &SEARCH_ARGMAX,
];

/// Zero all counters (start of a profiled batch).
pub fn reset() {
    reset_all(
        &PHASES,
        &[
            &N_RENDER,
            &N_SEARCH,
            &N_CELLS,
            &N_DROP_UNLOCALIZABLE,
            &N_DROP_SHIFT,
            &N_DROP_ABS_ZNCC,
            &N_DROP_REL_ZNCC,
        ],
    );
    crate::camera::remap::prof::reset();
}

/// Print the accumulated summary to stderr (end of a profiled batch).
pub fn report(patches: usize, wall_secs: f64) {
    let total_ns = TOTAL.ns().max(1);
    eprintln!(
        "[sfmtool-profile] localize_patch_cloud_keypoints: {patches} patches, wall {wall_secs:.3}s \
         (phase times are thread-summed CPU time; % of localize_total)"
    );
    report_phases(PHASES, total_ns, &PATCH_ROWS);
    report_overhead(
        &[&REFERENCE, &RENDER, &SEARCH],
        total_ns,
        "localize_total",
        &PATCH_ROWS,
    );
    let n_search = N_SEARCH.load(Ordering::Relaxed);
    let n_cells = N_CELLS.load(Ordering::Relaxed);
    eprintln!(
        "[sfmtool-profile]   renders {}  searches {}  cells {} ({:.2}/search)  \
         dropped: unlocalizable {}, shift {}, abs-zncc {}, rel-zncc {}",
        N_RENDER.load(Ordering::Relaxed),
        n_search,
        n_cells,
        if n_search > 0 {
            n_cells as f64 / n_search as f64
        } else {
            0.0
        },
        N_DROP_UNLOCALIZABLE.load(Ordering::Relaxed),
        N_DROP_SHIFT.load(Ordering::Relaxed),
        N_DROP_ABS_ZNCC.load(Ordering::Relaxed),
        N_DROP_REL_ZNCC.load(Ordering::Relaxed),
    );
    crate::camera::remap::prof::report();
}
