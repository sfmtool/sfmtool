// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Opt-in resample-sampler counters, shared by both patch phases (localization's
//! `render_remap` and refinement's `remap`). Gated on `SFMTOOL_PROFILE`; with the
//! variable unset the increments compile to a single branch on a cached flag, so
//! the hot path is unaffected. The per-phase prof modules
//! (`keypoint_localize::prof`, `normal_refine::prof`) call [`reset`] at the start
//! of their batch and [`report`] at the end, so each phase reads its own totals.
//!
//! Counting happens per output row (one atomic add per row, not per pixel), so the
//! inner bilinear loop is untouched; the small per-row atomics still inflate the
//! `render_remap`/`remap` *timing* slightly when profiling is on, so take the
//! counts from an instrumented run and the timings from a clean run.
//!
//! Counted paths: `remap_bilinear`, `remap_bilinear_mip`,
//! `remap_aniso_with_pyramid`, and the two bilinear gradient paths
//! `remap_bilinear_with_grad_into` and `remap_bilinear_mip_with_grad_into` (the
//! latter is the default subpixel sampler). The anisotropic *gradient* path
//! (`remap_aniso_with_grad_into`) is not tap-counted — its per-pixel tap count
//! depends on the footprint walk, which the value path already characterizes
//! via [`ANISO_FAST`] / [`ANISO_MULTI`] / [`ANISO_SUM_N`].

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

/// Whether `SFMTOOL_PROFILE` is set (cached on first query).
pub fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED
        .get_or_init(|| std::env::var("SFMTOOL_PROFILE").is_ok_and(|v| !v.is_empty() && v != "0"))
}

/// `remap_*` calls (one per (patch, view) render).
pub static CALLS: AtomicU64 = AtomicU64::new(0);
/// Output pixels visited (`out_w * out_h`, summed over calls).
pub static PX_TOTAL: AtomicU64 = AtomicU64::new(0);
/// Output pixels actually sampled (source coord not NaN — the rest are
/// out-of-frame / behind-camera and written black for free).
pub static PX_SAMPLED: AtomicU64 = AtomicU64::new(0);
/// Bilinear taps issued: one per sampled pixel per channel for
/// `remap_bilinear`; for the anisotropic sampler also × the footprint walk ×
/// pyramid levels. A "tap" is one `bilinear_taps` 4-corner fetch.
pub static TAPS: AtomicU64 = AtomicU64::new(0);
/// Anisotropic-only: pixels taking the single-bilinear fast path
/// (`sigma_major <= 1`).
pub static ANISO_FAST: AtomicU64 = AtomicU64::new(0);
/// Anisotropic-only: pixels taking the multi-tap footprint walk.
pub static ANISO_MULTI: AtomicU64 = AtomicU64::new(0);
/// Anisotropic-only: summed footprint length `n` over multi-tap pixels
/// (mean `n = ANISO_SUM_N / ANISO_MULTI`).
pub static ANISO_SUM_N: AtomicU64 = AtomicU64::new(0);

/// Add `n` to `c` when profiling is on.
#[inline]
pub fn add(c: &AtomicU64, n: u64) {
    if enabled() {
        c.fetch_add(n, Ordering::Relaxed);
    }
}

/// Zero all sampler counters (start of a profiled batch).
pub fn reset() {
    for c in [
        &CALLS,
        &PX_TOTAL,
        &PX_SAMPLED,
        &TAPS,
        &ANISO_FAST,
        &ANISO_MULTI,
        &ANISO_SUM_N,
    ] {
        c.store(0, Ordering::Relaxed);
    }
}

/// Print the sampler summary to stderr (end of a profiled batch). No-op when
/// profiling is off or no remap ran.
pub fn report() {
    if !enabled() {
        return;
    }
    let calls = CALLS.load(Ordering::Relaxed);
    if calls == 0 {
        return;
    }
    let total = PX_TOTAL.load(Ordering::Relaxed).max(1);
    let sampled = PX_SAMPLED.load(Ordering::Relaxed);
    let taps = TAPS.load(Ordering::Relaxed);
    let fast = ANISO_FAST.load(Ordering::Relaxed);
    let multi = ANISO_MULTI.load(Ordering::Relaxed);
    let sum_n = ANISO_SUM_N.load(Ordering::Relaxed);
    eprintln!(
        "[sfmtool-profile]   remap-sampler: {calls} calls, {sampled}/{total} px sampled \
         ({:.1}%), {taps} taps ({:.2} taps/sampled-px)",
        100.0 * sampled as f64 / total as f64,
        if sampled > 0 {
            taps as f64 / sampled as f64
        } else {
            0.0
        },
    );
    if fast > 0 || multi > 0 {
        eprintln!(
            "[sfmtool-profile]   remap-aniso: {fast} fast-path px, {multi} multi-tap px, \
             mean footprint n {:.2}",
            if multi > 0 {
                sum_n as f64 / multi as f64
            } else {
                0.0
            },
        );
    }
}
