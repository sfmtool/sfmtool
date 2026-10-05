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

pub(crate) use crate::profiling::{enabled, Phase};
use crate::profiling::{report_overhead, report_phases, reset_all, RowFormat};

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

/// The phase-row layout of this summary.
const ROWS: RowFormat = RowFormat {
    name_width: 20,
    secs_precision: 4,
    calls: None,
};

/// Zero all counters (start of a profiled build).
pub(crate) fn reset() {
    reset_all(&PHASES, &[]);
}

/// Print the accumulated summary to stderr (end of a profiled build).
pub(crate) fn report(num_images: usize, n_clusters: usize, n_members: usize) {
    let total_ns = TOTAL.ns().max(1);
    eprintln!(
        "[sfmtool-profile] cluster_covisibility: total {:.3}s over {num_images} images, \
         {n_clusters} clusters, {n_members} members",
        total_ns as f64 * 1e-9
    );
    report_phases(PHASES, total_ns, &ROWS);
    report_overhead(&LEAVES, total_ns, "total", &ROWS);
}
