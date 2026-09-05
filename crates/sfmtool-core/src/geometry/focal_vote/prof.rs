// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Opt-in phase timing for the focal-vote kernel.
//!
//! Set `SFMTOOL_PROFILE=1` to accumulate per-phase wall time (atomic nanosecond
//! counters, summed across rayon threads) during
//! [`focal_vote_with_options`](super::focal_vote_with_options); a summary goes
//! to stderr when the vote finishes. With the variable unset every timer is one
//! branch on a cached flag, so the hot path is unaffected. Mirrors
//! `crate::patch::cluster_refine::prof`.
//!
//! Phase times are *thread-summed* (CPU-seconds, not wall-clock): with `N`
//! rayon threads busy, one wall second accumulates up to `N` phase-seconds.
//! Read the shares, not the absolute values — and read the shares at
//! `RAYON_NUM_THREADS=1`, where the atomics also carry no contention.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::Instant;

/// Whether `SFMTOOL_PROFILE` is set (cached on first query).
pub(crate) fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED
        .get_or_init(|| std::env::var("SFMTOOL_PROFILE").is_ok_and(|v| !v.is_empty() && v != "0"))
}

/// One accumulating phase counter: total nanoseconds and number of events.
pub(crate) struct Phase {
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
    pub(crate) fn time<T>(&self, f: impl FnOnce() -> T) -> T {
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

// Enclosing phase (overlaps everything else; the 100% denominator).
/// The whole `focal_vote_with_options` call.
pub(crate) static TOTAL: Phase = Phase::new("total");

// ── Closed-form (pinhole) phases ─────────────────────────────────────────────
/// The one pass over cluster runs building the pair tables.
pub(crate) static PAIRTABLE: Phase = Phase::new("pair_table");
/// `pair_correspondences` merge-joins (every caller).
pub(crate) static CORR: Phase = Phase::new("correspondences");
/// `estimate_fundamental` (encloses [`SCORE_F`]).
pub(crate) static EST_F: Phase = Phase::new("estimate_F");
/// `estimate_homography` (encloses [`SCORE_H`]).
pub(crate) static EST_H: Phase = Phase::new("estimate_H");
/// Sub-phase of [`EST_F`]: Sampson inlier scoring.
pub(crate) static SCORE_F: Phase = Phase::new("  score_F");
/// Sub-phase of [`EST_H`]: symmetric-transfer inlier scoring.
pub(crate) static SCORE_H: Phase = Phase::new("  score_H");
/// Sub-phase of [`EST_H`]: the DLT solvers (4-point minimal and N-point refit).
pub(crate) static H_DLT: Phase = Phase::new("  homography_dlt");
/// `focal_from_fundamental` (Bougnoux, with its SVD).
pub(crate) static BOUGNOUX: Phase = Phase::new("bougnoux");
/// `rotation_self_calib_focal` (the 48-point orthogonality grid).
pub(crate) static ORTHO: Phase = Phase::new("ortho_scan");

// ── Column-scan phases ───────────────────────────────────────────────────────
/// `ScanCandidate::new` minus its merge-join (which lands in [`CORR`]).
pub(crate) static SCAN_CAND: Phase = Phase::new("scan_candidates");
/// Per-grid-point ray / tolerance / pixel-scale rebuild loops.
pub(crate) static RAYS: Phase = Phase::new("ray_rebuild");
/// `epipolar_rows` design-row builds.
pub(crate) static EPI_ROWS: Phase = Phase::new("epipolar_rows");
/// Minimal-sample solvers (`null9_from_8rows` elimination).
pub(crate) static MINSOLVE: Phase = Phase::new("minimal_solve");
/// `epipolar_residuals` (the AVX2 f64 residual kernel).
pub(crate) static EPI_RESID: Phase = Phase::new("epipolar_resid");
/// Consensus counting / masking loops of the epipolar cell.
pub(crate) static EPI_MASK: Phase = Phase::new("epipolar_mask");
/// Consensus refits of the epipolar cell (`null_from_rows`, 9×9 eigen).
pub(crate) static EPI_REFIT: Phase = Phase::new("epipolar_refit");
/// `singular_values_desc` of the essentialness cost.
pub(crate) static EPI_SVD: Phase = Phase::new("epipolar_svd");
/// `kabsch` orthogonal Procrustes fits (3×3 SVD).
pub(crate) static KABSCH: Phase = Phase::new("kabsch");
/// `rotation_residuals` (the AVX2 f64 cosine + polynomial `acos` kernel).
pub(crate) static ROT_RESID: Phase = Phase::new("rotation_resid");
/// Consensus counting / masking loops of the rotation cell.
pub(crate) static ROT_MASK: Phase = Phase::new("rotation_mask");
/// Trimming sorts and RMS accumulation of `fit_rotation`.
pub(crate) static ROT_TRIM: Phase = Phase::new("rotation_trim");

const PHASES: [&Phase; 19] = [
    &TOTAL, &PAIRTABLE, &CORR, &EST_F, &SCORE_F, &EST_H, &SCORE_H, &BOUGNOUX, &ORTHO, &SCAN_CAND,
    &RAYS, &EPI_ROWS, &MINSOLVE, &EPI_RESID, &EPI_MASK, &EPI_REFIT, &EPI_SVD, &KABSCH, &ROT_RESID,
];
/// The remaining leaves, split out only because `PHASES` is a fixed-size array.
const PHASES2: [&Phase; 3] = [&ROT_MASK, &ROT_TRIM, &H_DLT];

/// Leaves that partition the bulk of [`TOTAL`] (the `score_*` sub-phases are
/// nested inside `EST_*` and are excluded so the sum stays a partition).
const LEAVES: [&Phase; 18] = [
    &ROT_MASK, &ROT_TRIM, &PAIRTABLE, &CORR, &EST_F, &EST_H, &BOUGNOUX, &ORTHO, &SCAN_CAND, &RAYS,
    &EPI_ROWS, &MINSOLVE, &EPI_RESID, &EPI_MASK, &EPI_REFIT, &EPI_SVD, &KABSCH, &ROT_RESID,
];

// ── f32 residual audit ───────────────────────────────────────────────────────

/// Whether `SFMTOOL_FOCAL_VOTE_F32_AUDIT` is set (cached on first query).
///
/// With it on, every `f32` residual pass also runs the `f64` kernel over the
/// **same rays** and files `|Δ|` in [`ResidAudit`], which isolates what the
/// narrower arithmetic costs from what narrowing the rays costs. It roughly
/// triples the cost of the arms it audits, so it is a measurement mode, not
/// something to leave on.
pub(crate) fn audit_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("SFMTOOL_FOCAL_VOTE_F32_AUDIT").is_some())
}

/// Decade histogram of one residual loop's `f32`-against-`f64` absolute
/// differences, plus the exact maximum.
///
/// Bucket `k < 16` counts `|Δ| ∈ [10^-k-1, 10^-k)`; bucket 16 counts exact
/// agreement. Quantiles read off decade boundaries, which is the resolution
/// this question deserves — the answer wanted is "how many digits", not a
/// number.
pub(crate) struct ResidAudit {
    name: &'static str,
    buckets: [AtomicU64; 17],
    /// The largest `|Δ|` seen, as its `f64` bit pattern (nonnegative floats
    /// order the same as their bits, so `fetch_max` on the bits is a maximum).
    max_bits: AtomicU64,
}

impl ResidAudit {
    const fn new(name: &'static str) -> Self {
        #[allow(clippy::declare_interior_mutable_const)]
        const Z: AtomicU64 = AtomicU64::new(0);
        Self {
            name,
            buckets: [Z; 17],
            max_bits: AtomicU64::new(0),
        }
    }

    fn reset(&self) {
        for b in &self.buckets {
            b.store(0, Ordering::Relaxed);
        }
        self.max_bits.store(0, Ordering::Relaxed);
    }

    /// File one difference.
    #[inline]
    pub(crate) fn record(&self, d: f64) {
        let d = d.abs();
        let k = if d > 0.0 {
            (-d.log10()).floor().clamp(0.0, 15.0) as usize
        } else {
            16
        };
        self.buckets[k].fetch_add(1, Ordering::Relaxed);
        self.max_bits.fetch_max(d.to_bits(), Ordering::Relaxed);
    }

    fn report(&self) {
        let counts: Vec<u64> = self
            .buckets
            .iter()
            .map(|b| b.load(Ordering::Relaxed))
            .collect();
        let total: u64 = counts.iter().sum();
        if total == 0 {
            return;
        }
        // Buckets run largest-difference-first, so a cumulative sweep in order
        // gives the upper quantiles directly.
        let q = |p: f64| -> String {
            let want = (p * total as f64).ceil() as u64;
            let mut seen = 0u64;
            for (k, &c) in counts.iter().enumerate() {
                seen += c;
                if seen > total.saturating_sub(want) {
                    return if k == 16 {
                        "0".into()
                    } else {
                        format!("<1e-{k}")
                    };
                }
            }
            "?".into()
        };
        eprintln!(
            "[sfmtool-profile]   audit {:<16} n {:>10}  p50 {:>7}  p90 {:>7}  p99 {:>7}  \
             max {:.3e}",
            self.name,
            total,
            q(0.50),
            q(0.90),
            q(0.99),
            f64::from_bits(self.max_bits.load(Ordering::Relaxed)),
        );
    }
}

/// `epipolar_residuals` in `f32` against `f64` on the same rays.
pub(crate) static AUDIT_EPI: ResidAudit = ResidAudit::new("epipolar_resid");
/// `rotation_residuals` in `f32` (cross-product form) against `f64` (`acos` of
/// the dot) on the same rays. Radians either way.
pub(crate) static AUDIT_ROT: ResidAudit = ResidAudit::new("rotation_resid");

/// Zero all counters (start of a profiled vote).
pub(crate) fn reset() {
    for p in PHASES.iter().chain(PHASES2.iter()) {
        p.reset();
    }
    AUDIT_EPI.reset();
    AUDIT_ROT.reset();
}

/// Print the accumulated summary to stderr (end of a profiled vote).
pub(crate) fn report() {
    let total_ns = TOTAL.ns.load(Ordering::Relaxed).max(1);
    eprintln!(
        "[sfmtool-profile] focal_vote: total {:.3}s (phase times are thread-summed CPU time; \
         % of total)",
        total_ns as f64 * 1e-9
    );
    for p in PHASES.iter().chain(PHASES2.iter()) {
        let ns = p.ns.load(Ordering::Relaxed);
        let calls = p.calls.load(Ordering::Relaxed);
        eprintln!(
            "[sfmtool-profile]   {:<16} {:>9.4}s  {:>5.1}%  {:>11} calls  {:>9.3}us/call",
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
    let leaves: u64 = LEAVES.iter().map(|p| p.ns.load(Ordering::Relaxed)).sum();
    eprintln!(
        "[sfmtool-profile]   {:<16} {:>9.4}s  {:>5.1}%  (total minus leaf phases)",
        "other/overhead",
        total_ns.saturating_sub(leaves) as f64 * 1e-9,
        100.0 * total_ns.saturating_sub(leaves) as f64 / total_ns as f64,
    );
    AUDIT_EPI.report();
    AUDIT_ROT.report();
}
