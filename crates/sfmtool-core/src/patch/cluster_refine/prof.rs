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

pub use crate::profiling::{count, enabled, Phase};
use crate::profiling::{report_overhead, report_phases, reset_all, PATCH_ROWS};

// Enclosing phase (overlaps the leaves; reported as the 100% denominator).
/// Whole per-cluster `refine_cluster` calls.
pub static TOTAL: Phase = Phase::new("cluster_total");

// Leaf phases (non-overlapping; they partition the bulk of TOTAL).
/// Member self-similarity gate: per-member sample of its own `R×R` grid
/// (`sample_member_grid`).
pub static GATE_SAMPLE: Phase = Phase::new("gate_sample");
/// Member self-similarity gate: per-member ZNCC self-similarity radius
/// reading.
pub static GATE_SCORE: Phase = Phase::new("gate_score");
/// Reference template builds (`build_template`), including fallback retries.
pub static TEMPLATE: Phase = Phase::new("build_template");
/// Whole per-member refinement cascades (`refine_member`), enclosing
/// [`TILE`] and [`EVAL`].
pub static REFINE: Phase = Phase::new("refine_member");
/// Piecewise refinements of kept members (`refine_kept_member_cells`): the
/// working-patch renders, the cell searches and update fits, and the
/// whole-patch readings taken again at the refined shape. Encloses the
/// [`TILE`] build that re-reading makes; it encloses no [`EVAL`], because the
/// re-reading's `eval_zncc` call is not timed as one.
pub static PIECEWISE: Phase = Phase::new("piecewise");
/// Sub-phase of [`REFINE`] and of [`PIECEWISE`]: `LevelTile` builds and
/// rebuilds inside `TileCache::get_or_build` (cache hits are not timed), by
/// the cascade's objective and by the piecewise stage's re-reading alike.
pub static TILE: Phase = Phase::new("tile_build");
/// Sub-phase of [`REFINE`] only: the cascade's fused windowed-ZNCC objective
/// evaluations (`eval_zncc`).
pub static EVAL: Phase = Phase::new("eval_zncc");

// Event counters (no time attached).
/// Members carrying usable geometry (the gate + refinement population).
pub static N_MEMBERS: AtomicU64 = AtomicU64::new(0);
/// Members read by the member self-similarity gate.
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
/// Kept members the piecewise refinement ran on.
pub static N_PIECEWISE: AtomicU64 = AtomicU64::new(0);
/// `LevelTile` (re)builds, by the cascade and the piecewise stage.
pub static N_TILE_BUILDS: AtomicU64 = AtomicU64::new(0);
/// Pixels copied into (re)built `LevelTile`s (tile area × channels).
pub static N_TILE_PIXELS: AtomicU64 = AtomicU64::new(0);

const PHASES: [&Phase; 8] = [
    &TOTAL,
    &GATE_SAMPLE,
    &GATE_SCORE,
    &TEMPLATE,
    &REFINE,
    &PIECEWISE,
    &TILE,
    &EVAL,
];

/// Zero all counters (start of a profiled batch).
pub fn reset() {
    reset_all(
        &PHASES,
        &[
            &N_MEMBERS,
            &N_GATED,
            &N_GATE_REJECTED,
            &N_REFINES,
            &N_EVALS,
            &N_EVALS_SHIFT,
            &N_EVALS_SIM,
            &N_EVALS_AFFINE,
            &N_PIECEWISE,
            &N_TILE_BUILDS,
            &N_TILE_PIXELS,
        ],
    );
}

/// Print the accumulated summary to stderr (end of a profiled batch).
pub fn report(clusters: usize, wall_secs: f64) {
    let total_ns = TOTAL.ns().max(1);
    eprintln!(
        "[sfmtool-profile] refine_cluster_patches: {clusters} clusters, wall {wall_secs:.3}s \
         (phase times are thread-summed CPU time; % of cluster_total)"
    );
    report_phases(PHASES, total_ns, &PATCH_ROWS);
    report_overhead(
        &[&GATE_SAMPLE, &GATE_SCORE, &TEMPLATE, &REFINE, &PIECEWISE],
        total_ns,
        "cluster_total",
        &PATCH_ROWS,
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
    eprintln!(
        "[sfmtool-profile]   piecewise members {}",
        N_PIECEWISE.load(Ordering::Relaxed),
    );
}
