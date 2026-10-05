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

pub use crate::profiling::{count, enabled, Phase};
use crate::profiling::{report_overhead, report_phases, reset_all, PATCH_ROWS};

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
/// Sub-phase of the two SCORE leaves: the scored view's **exact** support
/// render (`normalized_stack` inside `candidate_zncc`) — the fallback when the
/// affine fast path declines a view (see [`N_AFFINE_FALLBACK`]).
pub static ZNCC_RENDER: Phase = Phase::new("zncc_render");
/// Sub-phase of the two SCORE leaves: the affine fast path's corner
/// projection + fit + gates (`affine_core_map`).
pub static AFFINE_MAP: Phase = Phase::new("affine_map");
/// Sub-phase of the two SCORE leaves: the affine fast path's support sampling
/// (`sample_support_affine` — bilinear gathers at the affine-mapped support
/// positions, skipping the per-pixel projective warp).
pub static AFFINE_SAMPLE: Phase = Phase::new("zncc_affine");
/// Sub-phase of the two SCORE leaves: the z-normalize + template dot of a
/// scored view (shared by the affine and exact paths).
pub static ZNCC_DOT: Phase = Phase::new("zncc_dot");

// Event counters (no time attached).
/// Points admitted verbatim (no reference could be built, or self-agreement
/// below the trust gate) — no candidate expansion ran.
pub static N_VERBATIM: AtomicU64 = AtomicU64::new(0);
/// Candidate views that passed the geometric gate and were ZNCC-scored.
pub static N_CANDIDATES: AtomicU64 = AtomicU64::new(0);
/// Candidate views admitted (ZNCC cleared the relative bar).
pub static N_ADMITTED: AtomicU64 = AtomicU64::new(0);
/// Views scored via the affine fast path (track diagnostics + candidates).
pub static N_AFFINE: AtomicU64 = AtomicU64::new(0);
/// Subset of [`N_AFFINE`] whose affine map selected a pyramid level above 0
/// (the `BilinearMip` sampler minifying the patch); the remainder sampled
/// level 0, where the mip and bilinear fast paths coincide.
pub static N_AFFINE_MIP: AtomicU64 = AtomicU64::new(0);
/// Views the affine fast path declined — corner projection failed, the
/// 4th-corner residual exceeded the bound (heavy distortion / wide angle), or
/// the mapped patch came too close to the frame border — scored by the exact
/// warp instead.
pub static N_AFFINE_FALLBACK: AtomicU64 = AtomicU64::new(0);

const PHASES: [&Phase; 11] = [
    &TOTAL,
    &REFERENCE,
    &REF_SUPPORT,
    &REF_RENDER,
    &REF_CONSENSUS,
    &TRACK_SCORE,
    &CAND_SCORE,
    &ZNCC_RENDER,
    &AFFINE_MAP,
    &AFFINE_SAMPLE,
    &ZNCC_DOT,
];

/// Zero all counters (start of a profiled batch).
pub fn reset() {
    reset_all(
        &PHASES,
        &[
            &N_VERBATIM,
            &N_CANDIDATES,
            &N_ADMITTED,
            &N_AFFINE,
            &N_AFFINE_MIP,
            &N_AFFINE_FALLBACK,
        ],
    );
    crate::camera::remap::prof::reset();
}

/// Print the accumulated summary to stderr (end of a profiled batch). No-op
/// when profiling is off. `wall_secs` is the batch wall time measured by the
/// caller.
pub fn report(patches: usize, wall_secs: f64) {
    if !enabled() {
        return;
    }
    let total_ns = TOTAL.ns().max(1);
    eprintln!(
        "[sfmtool-profile] select_patch_cloud_views: {patches} patches, wall {wall_secs:.3}s \
         (phase times are thread-summed CPU time; % of select_total)"
    );
    report_phases(PHASES, total_ns, &PATCH_ROWS);
    report_overhead(
        &[&REFERENCE, &TRACK_SCORE, &CAND_SCORE],
        total_ns,
        "select_total",
        &PATCH_ROWS,
    );
    eprintln!(
        "[sfmtool-profile]   verbatim {}  candidates-scored {}  candidates-admitted {}  \
         affine-scored {} (mip level>0: {})  affine-fallbacks {}",
        N_VERBATIM.load(Ordering::Relaxed),
        N_CANDIDATES.load(Ordering::Relaxed),
        N_ADMITTED.load(Ordering::Relaxed),
        N_AFFINE.load(Ordering::Relaxed),
        N_AFFINE_MIP.load(Ordering::Relaxed),
        N_AFFINE_FALLBACK.load(Ordering::Relaxed),
    );
    crate::camera::remap::prof::report();
}
