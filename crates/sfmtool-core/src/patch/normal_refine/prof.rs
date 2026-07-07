// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Opt-in phase timing for patch-normal refinement.
//!
//! Set `SFMTOOL_PROFILE=1` to accumulate per-phase wall time (atomic
//! nanosecond counters, summed across rayon threads) during
//! [`refine_patch_cloud_normals`](super::refine_patch_cloud_normals); a summary is printed to
//! stderr when the batch finishes. With the variable unset the timers compile
//! to a single branch on a cached flag, so the hot path is unaffected.
//!
//! Phase times are *thread-summed* (CPU-seconds, not wall-clock): with N rayon
//! threads busy, one wall second accumulates up to N phase-seconds. Shares of
//! the total are therefore meaningful; absolute values exceed wall time.

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

// Leaf phases (non-overlapping; they partition the bulk of TOTAL).
/// `WarpMap::from_patch` for candidate renders.
pub static WARP: Phase = Phase::new("warp_from_patch");
/// `WarpMap::compute_svd` (anisotropic sampler only).
pub static SVD: Phase = Phase::new("warp_svd");
/// `remap_bilinear` / `remap_aniso_with_pyramid`.
pub static REMAP: Phase = Phase::new("remap");
/// Masked-pixel gather + z-normalization in `normalized_stack`.
pub static ZNORM: Phase = Phase::new("gather_znorm");
/// `consensus_phi` (closed-form consensus + IRLS).
pub static CONSENSUS: Phase = Phase::new("consensus_phi");
/// `view_valid_mask` (support warps + masks; level/final context builds).
pub static MASK: Phase = Phase::new("valid_mask");
// Fronto-cache leaf phases (the per-candidate cached scoring path).
/// Candidate→base affine map: corner projection + 3-corner fit + compose.
pub static CACHE_MAP: Phase = Phase::new("cache_map");
/// `resample_support` (the AVX2 gather kernel).
pub static CACHE_RESAMPLE: Phase = Phase::new("cache_resample");
/// `znormalize_into` on the cached stack.
pub static CACHE_ZNORM: Phase = Phase::new("cache_znorm");
/// `consensus_phi` on the cached stack.
pub static CACHE_CONSENSUS: Phase = Phase::new("cache_consensus");
/// Fronto-base packing in `prerender`: the u32 pack, replicate pad, and
/// base-affine corner fit/inverse (the base *render* itself is attributed to
/// [`WARP`] / [`REMAP`]).
pub static CACHE_PACK: Phase = Phase::new("cache_pack");
/// Per-candidate setup on the cached scoring path: candidate repose, obliquity
/// prior fill, raw-buffer resize (the keypoint-anchored recenter + affine map
/// build is attributed to [`CACHE_MAP`]).
pub static CACHE_SETUP: Phase = Phase::new("cache_setup");
/// D-optimal refinement-basis subset selection
/// (`view_subset::select_refine_subset`, the `max_refine_views` cap).
pub static SUBSET: Phase = Phase::new("view_subset");

// Consensus sub-phases (nest inside CONSENSUS / CACHE_CONSENSUS for the
// consensus deep-dive; not partitioned into the leaf total).
/// IRLS weighted-consensus (xbar) SAXPY accumulation.
pub static IRLS_XBAR: Phase = Phase::new("irls_xbar");
/// IRLS per-view residual `‖xᵢ − x̄_w‖` computation.
pub static IRLS_RESID: Phase = Phase::new("irls_resid");
/// IRLS median/MAD/Tukey reweight (O(views)).
pub static IRLS_REWEIGHT: Phase = Phase::new("irls_reweight");
/// Final closed-form weighted consensus (post-IRLS).
pub static CONS_FINAL: Phase = Phase::new("cons_final");

// Enclosing phases (overlap the leaves; reported as shares of TOTAL).
/// Whole `refine_patch_normal` calls.
pub static TOTAL: Phase = Phase::new("refine_total");
/// Whole `grid_confidence` calls (9-point stencil; nests the leaves).
pub static CONFIDENCE: Phase = Phase::new("confidence");
/// Whole `fronto_cache::prerender` calls (per patch; nests WARP/REMAP).
pub static PRERENDER: Phase = Phase::new("cache_prerender");

// Event counters (no time attached).
/// `Φ` evaluations (each renders every kept view).
pub static N_EVAL: AtomicU64 = AtomicU64::new(0);
/// Per-view candidate renders (warp + remap + z-normalize).
pub static N_RENDER: AtomicU64 = AtomicU64::new(0);
/// Candidates rejected by the frozen-support validity re-check.
pub static N_REJECT: AtomicU64 = AtomicU64::new(0);
/// Patches whose refinement basis was capped by the D-optimal view subset.
pub static N_SUBSET: AtomicU64 = AtomicU64::new(0);
/// Patches where a cap was requested but the subset selection could not cap
/// (no front-facing view to anchor on), so all views were used.
pub static N_SUBSET_NO_ANCHOR: AtomicU64 = AtomicU64::new(0);

/// Count one event on `c` when profiling is on.
#[inline]
pub fn count(c: &AtomicU64, n: u64) {
    if enabled() {
        c.fetch_add(n, Ordering::Relaxed);
    }
}

const PHASES: [&Phase; 20] = [
    &TOTAL,
    &WARP,
    &SVD,
    &REMAP,
    &ZNORM,
    &CONSENSUS,
    &MASK,
    &CACHE_MAP,
    &CACHE_RESAMPLE,
    &CACHE_ZNORM,
    &CACHE_CONSENSUS,
    &CACHE_PACK,
    &CACHE_SETUP,
    &SUBSET,
    &IRLS_XBAR,
    &IRLS_RESID,
    &IRLS_REWEIGHT,
    &CONS_FINAL,
    &CONFIDENCE,
    &PRERENDER,
];

/// Zero all counters (start of a profiled batch).
pub fn reset() {
    for p in PHASES {
        p.reset();
    }
    for c in [
        &N_EVAL,
        &N_RENDER,
        &N_REJECT,
        &N_SUBSET,
        &N_SUBSET_NO_ANCHOR,
    ] {
        c.store(0, Ordering::Relaxed);
    }
    crate::camera::remap::prof::reset();
}

/// Print the accumulated summary to stderr (end of a profiled batch).
pub fn report(patches: usize, wall_secs: f64) {
    let total_ns = TOTAL.ns.load(Ordering::Relaxed).max(1);
    eprintln!(
        "[sfmtool-profile] refine_patch_cloud_normals: {patches} patches, wall {wall_secs:.3}s \
         (phase times are thread-summed CPU time; % of refine_total)"
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
        &WARP,
        &SVD,
        &REMAP,
        &ZNORM,
        &CONSENSUS,
        &MASK,
        &CACHE_MAP,
        &CACHE_RESAMPLE,
        &CACHE_ZNORM,
        &CACHE_CONSENSUS,
        &CACHE_PACK,
        &CACHE_SETUP,
        &SUBSET,
    ]
    .iter()
    .map(|p| p.ns.load(Ordering::Relaxed))
    .sum();
    eprintln!(
        "[sfmtool-profile]   {:<16} {:>9.3}s  {:>5.1}%  (refine_total minus leaf phases)",
        "other/overhead",
        (total_ns.saturating_sub(leaves)) as f64 * 1e-9,
        100.0 * total_ns.saturating_sub(leaves) as f64 / total_ns as f64,
    );
    eprintln!(
        "[sfmtool-profile]   evals {}  renders {}  support-rejects {}  \
         view-subsets {}  subset-no-anchor {}",
        N_EVAL.load(Ordering::Relaxed),
        N_RENDER.load(Ordering::Relaxed),
        N_REJECT.load(Ordering::Relaxed),
        N_SUBSET.load(Ordering::Relaxed),
        N_SUBSET_NO_ANCHOR.load(Ordering::Relaxed),
    );
    crate::camera::remap::prof::report();
}
