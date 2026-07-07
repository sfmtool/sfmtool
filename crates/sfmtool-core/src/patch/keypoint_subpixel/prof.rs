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
//! too (they cover the value renders + the bilinear GN-gradient renders; the
//! non-default anisotropic-gradient path is uncounted).

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
/// Whole `refine_patch_keypoints` calls.
pub static TOTAL: Phase = Phase::new("subpixel_total");

// Leaf phases (non-overlapping; they partition the bulk of TOTAL).
/// Value-only core renders (`render_core`): the seed score, every line-search
/// candidate, the per-sweep re-render, the PerMove kept-offset refresh, and the
/// representative's final-offset cores.
pub static RENDER_VALUE: Phase = Phase::new("value_render");
/// Value+gradient renders for the GN normal equations (`render_core_with_jg`),
/// one per Gauss–Newton step.
pub static RENDER_GRAD: Phase = Phase::new("gn_grad_render");
/// z-normalization of a raw core (`znorm_core`), wherever it runs (sweep
/// stack build, candidate scoring, PerMove refresh, representative).
pub static ZNORM: Phase = Phase::new("znormalize");
/// Per-sweep consensus (re)build: the IRLS view weights + weighted unit
/// template (+ the PerMove running-sum rebuild), and the representative's
/// final-weights IRLS.
pub static CONSENSUS: Phase = Phase::new("consensus_build");
/// PerMove within-sweep consensus maintenance: the per-move delta update of
/// the running sum and the shared-template realization. Zero under the default
/// PerSweep refresh.
pub static CONSENSUS_UPDATE: Phase = Phase::new("consensus_update");
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
/// Outer sweeps executed (summed over points).
pub static N_SWEEPS: AtomicU64 = AtomicU64::new(0);
/// Gauss–Newton steps taken (one `render_core_with_jg` + solve each).
pub static N_GN_STEPS: AtomicU64 = AtomicU64::new(0);
/// Line-search candidate evaluations (value render + znorm + ECC each).
pub static N_LINE_SEARCH: AtomicU64 = AtomicU64::new(0);

/// Count one event on `c` when profiling is on.
#[inline]
pub fn count(c: &AtomicU64, n: u64) {
    if enabled() {
        c.fetch_add(n, Ordering::Relaxed);
    }
}

const PHASES: [&Phase; 9] = [
    &TOTAL,
    &RENDER_VALUE,
    &RENDER_GRAD,
    &ZNORM,
    &CONSENSUS,
    &CONSENSUS_UPDATE,
    &JACOBIAN,
    &ECC,
    &REPR_FUSE,
];

/// Zero all counters (start of a profiled batch).
pub fn reset() {
    for p in PHASES {
        p.reset();
    }
    for c in [&N_SWEEPS, &N_GN_STEPS, &N_LINE_SEARCH] {
        c.store(0, Ordering::Relaxed);
    }
    crate::camera::remap::prof::reset();
}

/// Print the accumulated summary to stderr (end of a profiled batch). No-op
/// when profiling is off. `wall_secs` is the batch wall time measured by the
/// caller.
pub fn report(patches: usize, wall_secs: f64) {
    if !enabled() {
        return;
    }
    let total_ns = TOTAL.ns.load(Ordering::Relaxed).max(1);
    eprintln!(
        "[sfmtool-profile] refine_patch_keypoints: {patches} patches, wall {wall_secs:.3}s \
         (phase times are thread-summed CPU time; % of subpixel_total)"
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
    let leaves: u64 = [
        &RENDER_VALUE,
        &RENDER_GRAD,
        &ZNORM,
        &CONSENSUS,
        &CONSENSUS_UPDATE,
        &JACOBIAN,
        &ECC,
        &REPR_FUSE,
    ]
    .iter()
    .map(|p| p.ns.load(Ordering::Relaxed))
    .sum();
    eprintln!(
        "[sfmtool-profile]   {:<16} {:>9.3}s  {:>5.1}%  (subpixel_total minus leaf phases)",
        "other/overhead",
        (total_ns.saturating_sub(leaves)) as f64 * 1e-9,
        100.0 * total_ns.saturating_sub(leaves) as f64 / total_ns as f64,
    );
    eprintln!(
        "[sfmtool-profile]   sweeps {}  gn-steps {}  line-search-evals {}",
        N_SWEEPS.load(Ordering::Relaxed),
        N_GN_STEPS.load(Ordering::Relaxed),
        N_LINE_SEARCH.load(Ordering::Relaxed),
    );
    crate::camera::remap::prof::report();
}
