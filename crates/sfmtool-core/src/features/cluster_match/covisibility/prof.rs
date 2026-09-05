// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Opt-in phase timing for the cluster-covisibility build.
//!
//! Set `SFMTOOL_PROFILE=1` to accumulate per-phase wall time (atomic nanosecond
//! counters) during
//! [`from_clusters_with_positions`](super::ClusterCovisibility::from_clusters_with_positions);
//! a summary goes to stderr when the build finishes. With the variable unset
//! every timer is one branch on a cached flag, so the hot path is unaffected.
//! Mirrors `crate::geometry::focal_vote::prof`.
//!
//! The phases here are the ones a caller cannot separate from outside: the
//! positioned build runs three passes over the same arrays and returns one
//! object, so only an internal timer says which pass the wall time went to.
//! Each phase is timed once per build, so the `Instant` pair costs nothing
//! measurable even when profiling is on.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::Instant;

/// Whether `SFMTOOL_PROFILE` is set (cached on first query).
pub(crate) fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED
        .get_or_init(|| std::env::var("SFMTOOL_PROFILE").is_ok_and(|v| !v.is_empty() && v != "0"))
}

/// One accumulating phase counter: total nanoseconds.
pub(crate) struct Phase {
    name: &'static str,
    ns: AtomicU64,
}

impl Phase {
    const fn new(name: &'static str) -> Self {
        Self {
            name,
            ns: AtomicU64::new(0),
        }
    }

    fn reset(&self) {
        self.ns.store(0, Ordering::Relaxed);
    }

    /// Run `f`, attributing its wall time to this phase when profiling is on.
    #[inline]
    pub(crate) fn time<T>(&self, f: impl FnOnce() -> T) -> T {
        if !enabled() {
            return f();
        }
        let t0 = Instant::now();
        let r = f();
        self.record(t0);
        r
    }

    /// Attribute an already-started span to this phase. For a region that is
    /// not expressible as a closure -- the enclosing build, whose body ends in
    /// a fallible construction.
    #[inline]
    pub(crate) fn record(&self, t0: Instant) {
        if !enabled() {
            return;
        }
        self.ns
            .fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
}

/// The whole positioned build (the 100% denominator).
pub(crate) static TOTAL: Phase = Phase::new("total");
/// The one pass over clusters that dedupes each span, votes the shared counts,
/// and draws the sampled displacement pair.
pub(crate) static CLUSTER_PASS: Phase = Phase::new("cluster_pass");
/// The `num_images^2` divide-and-mirror fold over the sampled tables.
pub(crate) static MEAN_FOLD: Phase = Phase::new("mean_fold");
/// The neighborhood's exhaustive cross-image member-pair accumulation.
pub(crate) static NBR_ACCUM: Phase = Phase::new("neighborhood_accum");
/// The neighborhood's pair sort and CSR assembly.
pub(crate) static NBR_SORT: Phase = Phase::new("neighborhood_sort");

const PHASES: [&Phase; 5] = [&TOTAL, &CLUSTER_PASS, &MEAN_FOLD, &NBR_ACCUM, &NBR_SORT];
/// Leaves that partition [`TOTAL`].
const LEAVES: [&Phase; 4] = [&CLUSTER_PASS, &MEAN_FOLD, &NBR_ACCUM, &NBR_SORT];

/// Zero all counters (start of a profiled build).
pub(crate) fn reset() {
    for p in PHASES {
        p.reset();
    }
}

/// Print the accumulated summary to stderr (end of a profiled build).
pub(crate) fn report(num_images: usize, n_clusters: usize, n_members: usize) {
    let total_ns = TOTAL.ns.load(Ordering::Relaxed).max(1);
    eprintln!(
        "[sfmtool-profile] cluster_covisibility: total {:.3}s over {num_images} images, \
         {n_clusters} clusters, {n_members} members",
        total_ns as f64 * 1e-9
    );
    for p in PHASES {
        let ns = p.ns.load(Ordering::Relaxed);
        eprintln!(
            "[sfmtool-profile]   {:<20} {:>9.4}s  {:>5.1}%",
            p.name,
            ns as f64 * 1e-9,
            100.0 * ns as f64 / total_ns as f64,
        );
    }
    let leaves: u64 = LEAVES.iter().map(|p| p.ns.load(Ordering::Relaxed)).sum();
    eprintln!(
        "[sfmtool-profile]   {:<20} {:>9.4}s  {:>5.1}%  (total minus leaf phases)",
        "other/overhead",
        total_ns.saturating_sub(leaves) as f64 * 1e-9,
        100.0 * total_ns.saturating_sub(leaves) as f64 / total_ns as f64,
    );
}
