// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Opt-in phase timing for cluster-patch refinement.
//!
//! Set `SFMTOOL_PROFILE=1` to accumulate per-phase wall time (atomic nanosecond
//! counters, summed across rayon threads) during
//! [`refine_cluster_patches`](super::refine_cluster_patches); a summary is
//! printed to stderr when the batch finishes. With the variable unset the
//! timers compile to a single branch on a cached flag, so the hot path is
//! unaffected. Mirrors `keypoint_localize::prof`.
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

// Enclosing phase (overlaps the leaves; reported as the 100% denominator).
/// Whole per-cluster `refine_cluster` calls.
pub static TOTAL: Phase = Phase::new("cluster_total");

// Leaf phases (non-overlapping; they partition the bulk of TOTAL).
/// Localizability gate: per-member full-grid patch sample
/// (`sample_patch_grid`).
pub static GATE_SAMPLE: Phase = Phase::new("gate_sample");
/// Localizability gate: per-member `patch_localizability` score.
pub static GATE_SCORE: Phase = Phase::new("gate_score");
/// Reference template builds (`build_template`), including fallback retries.
pub static TEMPLATE: Phase = Phase::new("build_template");
/// Whole per-member refinement cascades (`refine_member`), enclosing
/// [`TILE`] and [`EVAL`].
pub static REFINE: Phase = Phase::new("refine_member");
/// Sub-phase of [`REFINE`]: `LevelTile` builds and rebuilds inside
/// `TileCache::get_or_build` (cache hits are not timed).
pub static TILE: Phase = Phase::new("tile_build");
/// Sub-phase of [`REFINE`]: fused windowed-ZNCC objective evaluations
/// (`eval_zncc`).
pub static EVAL: Phase = Phase::new("eval_zncc");

// Event counters (no time attached).
/// Members carrying usable geometry (the gate + refinement population).
pub static N_MEMBERS: AtomicU64 = AtomicU64::new(0);
/// Members scored by the localizability gate.
pub static N_GATED: AtomicU64 = AtomicU64::new(0);
/// Members the gate rejected.
pub static N_GATE_REJECTED: AtomicU64 = AtomicU64::new(0);
/// `refine_member` cascades run.
pub static N_REFINES: AtomicU64 = AtomicU64::new(0);
/// Objective evaluations (calls of `eval_zncc`, all cascade stages).
pub static N_EVALS: AtomicU64 = AtomicU64::new(0);
/// Objective evaluations spent in the shift stage (includes the seed check).
pub static N_EVALS_SHIFT: AtomicU64 = AtomicU64::new(0);
/// Objective evaluations spent in the similarity stage.
pub static N_EVALS_SIM: AtomicU64 = AtomicU64::new(0);
/// Objective evaluations spent in the affine stage.
pub static N_EVALS_AFFINE: AtomicU64 = AtomicU64::new(0);
/// `LevelTile` (re)builds.
pub static N_TILE_BUILDS: AtomicU64 = AtomicU64::new(0);
/// Pixels copied into (re)built `LevelTile`s (tile area × channels).
pub static N_TILE_PIXELS: AtomicU64 = AtomicU64::new(0);

/// Count `n` events on `c` when profiling is on.
#[inline]
pub fn count(c: &AtomicU64, n: u64) {
    if enabled() {
        c.fetch_add(n, Ordering::Relaxed);
    }
}

const PHASES: [&Phase; 7] = [
    &TOTAL,
    &GATE_SAMPLE,
    &GATE_SCORE,
    &TEMPLATE,
    &REFINE,
    &TILE,
    &EVAL,
];

/// Zero all counters (start of a profiled batch).
pub fn reset() {
    for p in PHASES {
        p.reset();
    }
    for c in [
        &N_MEMBERS,
        &N_GATED,
        &N_GATE_REJECTED,
        &N_REFINES,
        &N_EVALS,
        &N_EVALS_SHIFT,
        &N_EVALS_SIM,
        &N_EVALS_AFFINE,
        &N_TILE_BUILDS,
        &N_TILE_PIXELS,
    ] {
        c.store(0, Ordering::Relaxed);
    }
}

/// Print the accumulated summary to stderr (end of a profiled batch).
pub fn report(clusters: usize, wall_secs: f64) {
    let total_ns = TOTAL.ns.load(Ordering::Relaxed).max(1);
    eprintln!(
        "[sfmtool-profile] refine_cluster_patches: {clusters} clusters, wall {wall_secs:.3}s \
         (phase times are thread-summed CPU time; % of cluster_total)"
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
    let leaves: u64 = [&GATE_SAMPLE, &GATE_SCORE, &TEMPLATE, &REFINE]
        .iter()
        .map(|p| p.ns.load(Ordering::Relaxed))
        .sum();
    eprintln!(
        "[sfmtool-profile]   {:<16} {:>9.3}s  {:>5.1}%  (cluster_total minus leaf phases)",
        "other/overhead",
        (total_ns.saturating_sub(leaves)) as f64 * 1e-9,
        100.0 * total_ns.saturating_sub(leaves) as f64 / total_ns as f64,
    );
    let n_refines = N_REFINES.load(Ordering::Relaxed);
    let n_evals = N_EVALS.load(Ordering::Relaxed);
    let n_tiles = N_TILE_BUILDS.load(Ordering::Relaxed);
    eprintln!(
        "[sfmtool-profile]   members {}  gated {} (rejected {})  refines {}  evals {} \
         ({:.1}/refine)  tile builds {} ({:.1} px/build)",
        N_MEMBERS.load(Ordering::Relaxed),
        N_GATED.load(Ordering::Relaxed),
        N_GATE_REJECTED.load(Ordering::Relaxed),
        n_refines,
        n_evals,
        if n_refines > 0 {
            n_evals as f64 / n_refines as f64
        } else {
            0.0
        },
        n_tiles,
        if n_tiles > 0 {
            N_TILE_PIXELS.load(Ordering::Relaxed) as f64 / n_tiles as f64
        } else {
            0.0
        },
    );
    let per_refine = |c: &AtomicU64| {
        let v = c.load(Ordering::Relaxed);
        (
            v,
            if n_refines > 0 {
                v as f64 / n_refines as f64
            } else {
                0.0
            },
        )
    };
    let (sh, sh_r) = per_refine(&N_EVALS_SHIFT);
    let (si, si_r) = per_refine(&N_EVALS_SIM);
    let (af, af_r) = per_refine(&N_EVALS_AFFINE);
    eprintln!(
        "[sfmtool-profile]   evals by stage: shift {sh} ({sh_r:.1}/refine)  \
         sim {si} ({si_r:.1}/refine)  affine {af} ({af_r:.1}/refine)",
    );
}
