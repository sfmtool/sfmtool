// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Patch localizability — the keypoint self-similarity score.
//!
//! See `specs/core/patch-localizability.md`. A per-point grade of how well a
//! patch pins its own keypoint: the curvature of the ZNCC self-similarity
//! surface, measured as the classical Harris/Shi–Tomasi structure tensor on the
//! point's cross-view **consensus** patch. The noise-normalized weak-axis
//! eigenvalue comes out as a positional uncertainty `σ_pos` (in patch-grid px),
//! which catches the aperture blind spot the cross-view agreement gate misses
//! (a patch straddling a straight edge agrees perfectly yet slides along it).
//!
//! [`patch_localizability`](scorer::patch_localizability) is the pure per-patch
//! scorer (structure tensor + 2×2 eig); [`score_localizability_stack`] batches it
//! over a `(P, R, R, C)` consensus stack in parallel. The grid→source-px mapping
//! that turns `sigma_pos_grid` into a source-pixel `σ_pos` lives at the binding /
//! Python layer (it needs recon geometry, not the consensus).

mod scorer;

#[cfg(test)]
mod tests;

pub use scorer::{score_localizability_stack, Localizability};

use crate::patch::normal_refine::{window_weights as kernel_window_weights, PatchWindow};

/// The scorer's `R×R` window weights (row-major) for `window` — the shared patch
/// kernel, exposed so callers (the Python binding, tests) score against the same
/// window the scorer uses rather than reimplementing it.
pub fn window_weights(window: PatchWindow, resolution: u32) -> Vec<f64> {
    kernel_window_weights(window, resolution)
}
