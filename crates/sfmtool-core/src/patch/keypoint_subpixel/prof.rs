// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Opt-in phase timing for photometric subpixel keypoint refinement.
//!
//! Set `SFMTOOL_PROFILE=1` to accumulate per-phase wall time (atomic nanosecond
//! counters, summed across rayon threads) during
//! [`refine_patch_cloud_keypoints`](super::refine_patch_cloud_keypoints); a
//! summary is printed to stderr when the batch finishes. With the variable
//! unset the timers compile to a single branch on a cached flag, so the hot
//! path is unaffected. Mirrors `keypoint_localize::prof` / `normal_refine::prof`.
//!
//! Phase times are *thread-summed* (CPU-seconds, not wall-clock): with N rayon
//! threads busy, one wall second accumulates up to N phase-seconds. Shares of
//! the total are therefore meaningful; absolute values exceed wall time.
//!
//! Callers: the Rust batch entry
//! [`refine_patch_cloud_keypoints`](super::refine_patch_cloud_keypoints) and the
//! PyO3 binding (which inlines its own per-patch loop for lazy seed
//! construction) both bracket their work with [`reset`]/[`report`]; the
//! per-patch [`TOTAL`] phase is timed inside
//! [`refine_patch_keypoints`](super::refine_patch_keypoints) itself so both
//! entries are covered. The shared `camera::remap` tap counters are bracketed
//! too; with the render-once context tile they mostly count the per-(point,
//! view) tile prerenders (plus any direct-render fallbacks; the non-default
//! anisotropic-gradient path is uncounted).

use std::sync::atomic::{AtomicU64, Ordering};

pub use crate::profiling::{count, enabled, Phase};
use crate::profiling::{report_overhead, report_phases, reset_all, PATCH_ROWS};

// Enclosing phase (overlaps the leaves; reported as the 100% denominator).
/// Whole `refine_patch_keypoints` calls.
pub static TOTAL: Phase = Phase::new("subpixel_total");

// Leaf phases (non-overlapping; they partition the bulk of TOTAL).
/// Per-(point, view) context-tile prerenders (`render_refine_tile`): one
/// expanded value+gradient source render plus the warp-Jacobian composition
/// into patch-grid gradient planes. Every value / GN-gradient evaluation of
/// that (point, view) pair then reads this tile.
pub static TILE_PRERENDER: Phase = Phase::new("tile_prerender");
/// Value-only tile reads (`RefineTile::read_core`): the seed score, every
/// line-search candidate, and the representative's final-offset cores.
pub static VALUE_READ: Phase = Phase::new("value_read");
/// Jacobian-plane tile reads for the GN normal equations
/// (`RefineTile::read_jg`), one per Gauss–Newton step (the value core is
/// reused from the preceding score read).
pub static GRAD_READ: Phase = Phase::new("gn_grad_read");
/// Value-only direct core renders (`render_core`): every evaluation of a
/// coarse-grid (no-tile) view, plus the rare out-of-coverage fallback on a
/// tiled view (see [`N_TILE_SKIPPED`] / [`N_TILE_FALLBACK`]).
pub static RENDER_VALUE: Phase = Phase::new("value_render");
/// Value+gradient direct renders for the GN normal equations
/// (`render_core_with_jg`) — the no-tile / out-of-coverage counterpart of
/// [`GRAD_READ`].
pub static RENDER_GRAD: Phase = Phase::new("gn_grad_render");
/// z-normalization of a raw core (`znorm_core`), wherever it runs (the
/// template, candidate scoring, the representative).
pub static ZNORM: Phase = Phase::new("znormalize");
/// What the views are aligned to: nothing where the caller names the
/// reference, else the reference-view rule over the renders at the starting
/// keypoints, and the fused mean where the rule picks no reference it would
/// store.
pub static REFERENCE: Phase = Phase::new("reference");
/// The fused mean's IRLS view weights, where the representative is the fused
/// mean.
pub static CONSENSUS: Phase = Phase::new("consensus_build");
/// The analytic GN normal-equations build (`view_jacobian`: ∂ẑ/∂δ composition
/// and the H/b accumulation; the 2×2 solve is a handful of flops and is left
/// to overhead).
pub static JACOBIAN: Phase = Phase::new("gn_jacobian");
/// ECC scoring (`ecc_score`: the z-normalized-core × template dot).
pub static ECC: Phase = Phase::new("ecc_score");
/// Representative fusion (`render_bitmaps` only): the full-grid
/// `PatchViewStack::render` + `fuse` at the final keypoints. The stack's
/// support-only re-render/IRLS legs are attributed to the value-render /
/// znorm / consensus leaves above.
pub static REPR_FUSE: Phase = Phase::new("repr_stack_fuse");

// Event counters (no time attached).
/// Gauss–Newton steps taken (one `render_core_with_jg` + solve each).
pub static N_GN_STEPS: AtomicU64 = AtomicU64::new(0);
/// Line-search candidate evaluations (value render + znorm + ECC each).
pub static N_LINE_SEARCH: AtomicU64 = AtomicU64::new(0);
/// Evaluations whose offset fell outside the tile's coverage and took the
/// direct-render fallback (`render_core` / `render_core_with_jg`). Expected to
/// stay at ~0: the tile is sized to cover the line-search's `max_offset_px`
/// bound.
pub static N_TILE_FALLBACK: AtomicU64 = AtomicU64::new(0);
/// (point, view) pairs that got **no tile** (coarse-grid gate: the patch grid
/// is coarser than the source, so the pair keeps the exact direct-render
/// path — see `try_render_refine_tile`).
pub static N_TILE_SKIPPED: AtomicU64 = AtomicU64::new(0);

const PHASES: [&Phase; 12] = [
    &TOTAL,
    &TILE_PRERENDER,
    &VALUE_READ,
    &GRAD_READ,
    &RENDER_VALUE,
    &RENDER_GRAD,
    &ZNORM,
    &REFERENCE,
    &CONSENSUS,
    &JACOBIAN,
    &ECC,
    &REPR_FUSE,
];

/// Zero all counters (start of a profiled batch).
pub fn reset() {
    reset_all(
        &PHASES,
        &[
            &N_GN_STEPS,
            &N_LINE_SEARCH,
            &N_TILE_FALLBACK,
            &N_TILE_SKIPPED,
        ],
    );
    crate::camera::remap::prof::reset();
}

/// Print the accumulated summary to stderr (end of a profiled batch). No-op
/// when profiling is off. `wall_secs` is the batch wall time measured by the
/// caller.
pub fn report(patches: usize, wall_secs: f64) {
    if !enabled() {
        return;
    }
    let total_ns = TOTAL.ns().max(1);
    eprintln!(
        "[sfmtool-profile] refine_patch_keypoints: {patches} patches, wall {wall_secs:.3}s \
         (phase times are thread-summed CPU time; % of subpixel_total)"
    );
    report_phases(PHASES, total_ns, &PATCH_ROWS);
    report_overhead(
        &[
            &TILE_PRERENDER,
            &VALUE_READ,
            &GRAD_READ,
            &RENDER_VALUE,
            &RENDER_GRAD,
            &ZNORM,
            &REFERENCE,
            &CONSENSUS,
            &JACOBIAN,
            &ECC,
            &REPR_FUSE,
        ],
        total_ns,
        "subpixel_total",
        &PATCH_ROWS,
    );
    eprintln!(
        "[sfmtool-profile]   gn-steps {}  line-search-evals {}  tile-fallbacks {}  \
         tiles-skipped-coarse {}",
        N_GN_STEPS.load(Ordering::Relaxed),
        N_LINE_SEARCH.load(Ordering::Relaxed),
        N_TILE_FALLBACK.load(Ordering::Relaxed),
        N_TILE_SKIPPED.load(Ordering::Relaxed),
    );
    crate::camera::remap::prof::report();
}
