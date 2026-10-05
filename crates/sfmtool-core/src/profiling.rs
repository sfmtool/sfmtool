// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The counters, timer and report rows that the per-algorithm `prof` modules
//! share.
//!
//! Each `prof` module (`geometry::focal_vote::prof`, `camera::remap::prof`,
//! `features::cluster_match::covisibility::prof` and the five under `patch`)
//! declares its own phase statics, event counters and report text, and builds
//! them from the pieces here: the cached `SFMTOOL_PROFILE` gate ([`enabled`]),
//! the accumulating [`Phase`] timer, the [`count`] increment, and the two
//! report rows every summary prints ([`report_phases`] and
//! [`report_overhead`]). The row layout is passed in as a [`RowFormat`], so
//! each profiler keeps the column widths and precisions it has always printed.
//!
//! With `SFMTOOL_PROFILE` unset (or empty, or `0`) every timer and counter is
//! one branch on a cached flag, so the hot paths are unaffected.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::Instant;

/// Whether `SFMTOOL_PROFILE` is set to a value other than empty or `0`
/// (cached on first query).
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
    /// A zeroed phase that reports under `name`.
    pub(crate) const fn new(name: &'static str) -> Self {
        Self {
            name,
            ns: AtomicU64::new(0),
            calls: AtomicU64::new(0),
        }
    }

    /// Zero the time and the event count.
    pub(crate) fn reset(&self) {
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
        self.record(t0);
        r
    }

    /// Attribute an already-started span to this phase, for a region that is
    /// not expressible as a closure (for example one whose body ends in a
    /// fallible construction).
    #[inline]
    pub fn record(&self, t0: Instant) {
        if !enabled() {
            return;
        }
        self.ns
            .fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        self.calls.fetch_add(1, Ordering::Relaxed);
    }

    /// Accumulated nanoseconds.
    pub(crate) fn ns(&self) -> u64 {
        self.ns.load(Ordering::Relaxed)
    }
}

/// Add `n` to the event counter `c` when profiling is on.
#[inline]
pub fn count(c: &AtomicU64, n: u64) {
    if enabled() {
        c.fetch_add(n, Ordering::Relaxed);
    }
}

/// Zero every phase in `phases` and every event counter in `counters`.
pub(crate) fn reset_all(phases: &[&Phase], counters: &[&AtomicU64]) {
    for p in phases {
        p.reset();
    }
    for c in counters {
        c.store(0, Ordering::Relaxed);
    }
}

/// The column layout of one profiler's phase rows.
pub(crate) struct RowFormat {
    /// Width the phase name is left-aligned in.
    pub name_width: usize,
    /// Decimal places of the seconds column (right-aligned in 9 columns).
    pub secs_precision: usize,
    /// The call-count columns, or `None` for a profiler that prints none.
    pub calls: Option<CallColumns>,
}

/// The call-count and per-call-time columns of a phase row.
pub(crate) struct CallColumns {
    /// Width the call count is right-aligned in.
    pub count_width: usize,
    /// Width the microseconds-per-call value is right-aligned in.
    pub per_call_width: usize,
    /// Decimal places of the microseconds-per-call value.
    pub per_call_precision: usize,
}

/// The layout every `patch` profiler prints.
pub(crate) const PATCH_ROWS: RowFormat = RowFormat {
    name_width: 16,
    secs_precision: 3,
    calls: Some(CallColumns {
        count_width: 10,
        per_call_width: 8,
        per_call_precision: 2,
    }),
};

/// Print one stderr row per phase: its seconds, its share of `total_ns`, and
/// (when `fmt` has them) its call count and mean microseconds per call.
pub(crate) fn report_phases<'a>(
    phases: impl IntoIterator<Item = &'a Phase>,
    total_ns: u64,
    fmt: &RowFormat,
) {
    let (w, sp) = (fmt.name_width, fmt.secs_precision);
    for p in phases {
        let ns = p.ns();
        let secs = ns as f64 * 1e-9;
        let pct = 100.0 * ns as f64 / total_ns as f64;
        match &fmt.calls {
            None => eprintln!(
                "[sfmtool-profile]   {:<w$} {:>9.sp$}s  {:>5.1}%",
                p.name, secs, pct,
            ),
            Some(c) => {
                let calls = p.calls.load(Ordering::Relaxed);
                let (cw, uw, up) = (c.count_width, c.per_call_width, c.per_call_precision);
                eprintln!(
                    "[sfmtool-profile]   {:<w$} {:>9.sp$}s  {:>5.1}%  {:>cw$} calls  \
                     {:>uw$.up$}us/call",
                    p.name,
                    secs,
                    pct,
                    calls,
                    if calls > 0 {
                        ns as f64 * 1e-3 / calls as f64
                    } else {
                        0.0
                    },
                );
            }
        }
    }
}

/// Print the stderr row for the time in `total_ns` that no phase of `leaves`
/// covers, labelled "(`total_label` minus leaf phases)".
pub(crate) fn report_overhead(
    leaves: &[&Phase],
    total_ns: u64,
    total_label: &str,
    fmt: &RowFormat,
) {
    let (w, sp) = (fmt.name_width, fmt.secs_precision);
    let leaf_ns: u64 = leaves.iter().map(|p| p.ns()).sum();
    let other = total_ns.saturating_sub(leaf_ns);
    eprintln!(
        "[sfmtool-profile]   {:<w$} {:>9.sp$}s  {:>5.1}%  ({total_label} minus leaf phases)",
        "other/overhead",
        other as f64 * 1e-9,
        100.0 * other as f64 / total_ns as f64,
    );
}
