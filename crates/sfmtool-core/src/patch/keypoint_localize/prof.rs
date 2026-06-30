// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Opt-in phase timing for keypoint localization (congealing).
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
use std::sync::OnceLock;
use std::time::Instant;

/// Whether `SFMTOOL_PROFILE` is set (cached on first query).
pub fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED
        .get_or_init(|| std::env::var("SFMTOOL_PROFILE").is_ok_and(|v| !v.is_empty() && v != "0"))
}

/// One accumulating phase counter: total nanoseconds and number of events.
pub struct Phase {
    name: &'static str,
    ns: AtomicU64,
    calls: AtomicU64,
}

impl Phase {
    const fn new(name: &'static str) -> Self {
        Self {
            name,
            ns: AtomicU64::new(0),
            calls: AtomicU64::new(0),
        }
    }

    fn reset(&self) {
        self.ns.store(0, Ordering::Relaxed);
        self.calls.store(0, Ordering::Relaxed);
    }

    /// Run `f`, attributing its wall time to this phase when profiling is on.
    #[inline]
    pub fn time<T>(&self, f: impl FnOnce() -> T) -> T {
        if !enabled() {
            return f();
        }
        let t0 = Instant::now();
        let r = f();
        self.ns
            .fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        self.calls.fetch_add(1, Ordering::Relaxed);
        r
    }
}

// Enclosing phase (overlaps the leaves; reported as the 100% denominator).
/// Whole `localize_patch_keypoints` calls.
pub static TOTAL: Phase = Phase::new("localize_total");

// Leaf phases (non-overlapping; they partition the bulk of TOTAL).
/// Per-view render-once cache build (`render_context`), run a single time per view
/// before the round loop rather than per (view, round).
pub static RENDER: Phase = Phase::new("render_context");
/// Sub-phase of [`RENDER`]: `WarpMap::from_patch` — the per-pixel camera
/// projection of the context tile.
pub static RENDER_PROJECT: Phase = Phase::new("render_project");
/// Sub-phase of [`RENDER`]: `compute_svd` — the per-pixel anisotropic Jacobian
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
/// Per-round z-normalization of the live cores into shared channel space.
pub static ZNORM: Phase = Phase::new("znormalize");
/// Per-view leave-one-out consensus template (LOO copy + IRLS reweight +
/// weighted unit template).
pub static TEMPLATE: Phase = Phase::new("loo_template");
/// Per-view sub-pixel translation search against the LOO reference
/// (`search_shift`).
pub static SEARCH: Phase = Phase::new("search_shift");
/// Sub-phase of [`SEARCH`]: the per-channel correlation-grid accumulation
/// (`compute_channel_grids` — the dispatched inner kernel, `f32` or `i16`).
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
/// Congealing rounds executed (summed over points).
pub static N_ROUNDS: AtomicU64 = AtomicU64::new(0);
/// Cache renders performed (one per view, total — the render-once collapse means
/// this no longer scales with rounds).
pub static N_RENDER: AtomicU64 = AtomicU64::new(0);
/// Sub-pixel shift searches (one per live view per round).
pub static N_SEARCH: AtomicU64 = AtomicU64::new(0);
/// Cells scored by [`SearchStrategy::PlusDescent`](super::SearchStrategy::PlusDescent)
/// (one per visited-cache **miss** inside `search_shift_plus_descent`; visited-
/// cache **hits** do not count, since they re-use a prior cell's score). Always
/// `0` for [`SearchStrategy::Exhaustive`](super::SearchStrategy::Exhaustive) —
/// the SAXPY accumulator scores all `(2·margin+1)²` cells in one streaming
/// pass and has no per-cell event. Average cells per search call is
/// `N_CELLS / N_SEARCH`; the spec's per-call cells claim is derived here.
pub static N_CELLS: AtomicU64 = AtomicU64::new(0);

/// Count one event on `c` when profiling is on.
#[inline]
pub fn count(c: &AtomicU64, n: u64) {
    if enabled() {
        c.fetch_add(n, Ordering::Relaxed);
    }
}

const PHASES: [&Phase; 13] = [
    &TOTAL,
    &RENDER,
    &RENDER_PROJECT,
    &RENDER_SVD,
    &RENDER_REMAP,
    &RENDER_MEAN,
    &RENDER_CENTER,
    &ZNORM,
    &TEMPLATE,
    &SEARCH,
    &SEARCH_ACC,
    &SEARCH_COMBINE,
    &SEARCH_ARGMAX,
];

/// Zero all counters (start of a profiled batch).
pub fn reset() {
    for p in PHASES {
        p.reset();
    }
    for c in [&N_ROUNDS, &N_RENDER, &N_SEARCH, &N_CELLS] {
        c.store(0, Ordering::Relaxed);
    }
}

/// Print the accumulated summary to stderr (end of a profiled batch).
pub fn report(patches: usize, wall_secs: f64) {
    let total_ns = TOTAL.ns.load(Ordering::Relaxed).max(1);
    eprintln!(
        "[sfmtool-profile] localize_patch_cloud_keypoints: {patches} patches, wall {wall_secs:.3}s \
         (phase times are thread-summed CPU time; % of localize_total)"
    );
    for p in PHASES {
        let ns = p.ns.load(Ordering::Relaxed);
        let calls = p.calls.load(Ordering::Relaxed);
        eprintln!(
            "[sfmtool-profile]   {:<16} {:>9.3}s  {:>5.1}%  {:>10} calls  {:>8.2}us/call",
            p.name,
            ns as f64 * 1e-9,
            100.0 * ns as f64 / total_ns as f64,
            calls,
            if calls > 0 {
                ns as f64 * 1e-3 / calls as f64
            } else {
                0.0
            },
        );
    }
    let leaves: u64 = [&RENDER, &ZNORM, &TEMPLATE, &SEARCH]
        .iter()
        .map(|p| p.ns.load(Ordering::Relaxed))
        .sum();
    eprintln!(
        "[sfmtool-profile]   {:<16} {:>9.3}s  {:>5.1}%  (localize_total minus leaf phases)",
        "other/overhead",
        (total_ns.saturating_sub(leaves)) as f64 * 1e-9,
        100.0 * total_ns.saturating_sub(leaves) as f64 / total_ns as f64,
    );
    let n_search = N_SEARCH.load(Ordering::Relaxed);
    let n_cells = N_CELLS.load(Ordering::Relaxed);
    eprintln!(
        "[sfmtool-profile]   rounds {}  renders {}  searches {}  cells {} ({:.2}/search)",
        N_ROUNDS.load(Ordering::Relaxed),
        N_RENDER.load(Ordering::Relaxed),
        n_search,
        n_cells,
        if n_search > 0 {
            n_cells as f64 / n_search as f64
        } else {
            0.0
        },
    );
}
