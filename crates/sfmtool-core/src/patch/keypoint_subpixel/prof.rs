// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Opt-in remap accounting for photometric subpixel keypoint refinement.
//!
//! Unlike [`keypoint_localize::prof`](crate::patch::keypoint_localize) and
//! [`normal_refine::prof`](crate::patch::normal_refine), this stage does not (yet)
//! carry its own per-phase timers. It exists so that when `SFMTOOL_PROFILE=1` the
//! resample taps this stage's Gauss–Newton gradient renders issue (via
//! [`camera::remap`](crate::camera::remap)) are **reset before** and **reported
//! after** the batch, rather than leaking into a neighbouring phase's report or
//! going unaccounted. It simply brackets the shared `camera::remap::prof`
//! counters and prints a one-line header with the patch count and wall time.
//!
//! Callers: the Rust batch entry
//! [`refine_patch_cloud_keypoints`](super::refine_patch_cloud_keypoints) and the
//! PyO3 binding (which inlines its own per-patch loop for lazy seed construction)
//! both bracket their work with [`reset`]/[`report`].

use std::sync::OnceLock;

/// Whether `SFMTOOL_PROFILE` is set (cached on first query). Same semantics as
/// the sibling prof modules.
pub fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED
        .get_or_init(|| std::env::var("SFMTOOL_PROFILE").is_ok_and(|v| !v.is_empty() && v != "0"))
}

/// Zero the shared remap-sampler counters at the start of a profiled batch.
pub fn reset() {
    crate::camera::remap::prof::reset();
}

/// Print the subpixel remap summary to stderr at the end of a profiled batch.
/// No-op when profiling is off. `wall_secs` is the batch wall time measured by
/// the caller.
pub fn report(patches: usize, wall_secs: f64) {
    if !enabled() {
        return;
    }
    eprintln!(
        "[sfmtool-profile] refine_patch_keypoints: {patches} patches, wall {wall_secs:.3}s \
         (remap-sampler taps below cover the value renders + the bilinear GN-gradient \
         renders; the non-default anisotropic-gradient path is uncounted)"
    );
    crate::camera::remap::prof::report();
}
