// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Opt-in phase timing for patch-view selection.
//!
//! Set `SFMTOOL_PROFILE=1` to accumulate per-phase wall time (atomic nanosecond
//! counters, summed across rayon threads) during
//! [`select_patch_cloud_views`](super::select_patch_cloud_views); a summary is
//! printed to stderr when the batch finishes. With the variable unset the
//! timers compile to a single branch on a cached flag, so the hot path is
//! unaffected. Mirrors `keypoint_localize::prof` / `normal_refine::prof`.
//!
//! Phase times are *thread-summed* (CPU-seconds, not wall-clock): with N rayon
//! threads busy, one wall second accumulates up to N phase-seconds. Shares of
//! the total are therefore meaningful; absolute values exceed wall time.
//!
//! Selection drives the shared support/render machinery of `normal_refine`
//! (`build_level_context` / `normalized_stack`), whose own `normal_refine::prof`
//! timers also tick while a selection batch runs; those counters are reset at
//! the start of the next refine batch, so the leakage is harmless. The phases
//! here are at selection's own altitude: the reference build, and the per-view
//! ZNCC scoring split into its render and dot legs.

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
/// Whole `select_patch_views` calls.
pub static TOTAL: Phase = Phase::new("select_total");

// Leaf phases (non-overlapping; they partition the bulk of TOTAL).
/// Reference-template build (`build_reference`): the frozen track support, the
/// track renders, the IRLS reference consensus, and the self-agreement score.
pub static REFERENCE: Phase = Phase::new("reference_build");
/// Track-view ZNCC scoring against the reference (diagnostics; always-admitted
/// views).
pub static TRACK_SCORE: Phase = Phase::new("track_zncc");
/// Candidate-view ZNCC scoring against the reference (the expansion vetting).
pub static CAND_SCORE: Phase = Phase::new("candidate_zncc");

// Sub-phases (nest inside REFERENCE / TRACK_SCORE / CAND_SCORE; not part of
// the leaf partition).
/// Sub-phase of [`REFERENCE`]: the frozen track support (`build_level_context`).
pub static REF_SUPPORT: Phase = Phase::new("ref_support");
/// Sub-phase of [`REFERENCE`]: the track renders (`normalized_stack`).
pub static REF_RENDER: Phase = Phase::new("ref_render");
/// Sub-phase of [`REFERENCE`]: z-normalize + IRLS weights + unit template +
/// self-agreement.
pub static REF_CONSENSUS: Phase = Phase::new("ref_consensus");
/// Sub-phase of the two SCORE leaves: the scored view's support render
/// (`normalized_stack` inside `candidate_zncc`).
pub static ZNCC_RENDER: Phase = Phase::new("zncc_render");
/// Sub-phase of the two SCORE leaves: the z-normalize + template dot of a
/// scored view.
pub static ZNCC_DOT: Phase = Phase::new("zncc_dot");

// Event counters (no time attached).
/// Points admitted verbatim (no reference could be built, or self-agreement
/// below the trust gate) — no candidate expansion ran.
pub static N_VERBATIM: AtomicU64 = AtomicU64::new(0);
/// Candidate views that passed the geometric gate and were ZNCC-scored.
pub static N_CANDIDATES: AtomicU64 = AtomicU64::new(0);
/// Candidate views admitted (ZNCC cleared the relative bar).
pub static N_ADMITTED: AtomicU64 = AtomicU64::new(0);

/// Count one event on `c` when profiling is on.
#[inline]
pub fn count(c: &AtomicU64, n: u64) {
    if enabled() {
        c.fetch_add(n, Ordering::Relaxed);
    }
}

const PHASES: [&Phase; 9] = [
    &TOTAL,
    &REFERENCE,
    &REF_SUPPORT,
    &REF_RENDER,
    &REF_CONSENSUS,
    &TRACK_SCORE,
    &CAND_SCORE,
    &ZNCC_RENDER,
    &ZNCC_DOT,
];

/// Zero all counters (start of a profiled batch).
pub fn reset() {
    for p in PHASES {
        p.reset();
    }
    for c in [&N_VERBATIM, &N_CANDIDATES, &N_ADMITTED] {
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
        "[sfmtool-profile] select_patch_cloud_views: {patches} patches, wall {wall_secs:.3}s \
         (phase times are thread-summed CPU time; % of select_total)"
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
    let leaves: u64 = [&REFERENCE, &TRACK_SCORE, &CAND_SCORE]
        .iter()
        .map(|p| p.ns.load(Ordering::Relaxed))
        .sum();
    eprintln!(
        "[sfmtool-profile]   {:<16} {:>9.3}s  {:>5.1}%  (select_total minus leaf phases)",
        "other/overhead",
        (total_ns.saturating_sub(leaves)) as f64 * 1e-9,
        100.0 * total_ns.saturating_sub(leaves) as f64 / total_ns as f64,
    );
    eprintln!(
        "[sfmtool-profile]   verbatim {}  candidates-scored {}  candidates-admitted {}",
        N_VERBATIM.load(Ordering::Relaxed),
        N_CANDIDATES.load(Ordering::Relaxed),
        N_ADMITTED.load(Ordering::Relaxed),
    );
    crate::camera::remap::prof::report();
}
