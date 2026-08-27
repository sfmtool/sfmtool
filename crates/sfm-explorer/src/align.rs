// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The Scene Graph panel's `Align to ▸ <node>` action: what the viewer adds on
//! top of the shared fit.
//!
//! The fit itself — correspondence gathering, the trimmed least-squares
//! estimate, the RANSAC pass over point correspondences, and the counts and RMS
//! that describe the outcome — lives in
//! [`sfmtool_core::analysis::alignment::reconstructions`], so the GUI and any
//! other caller align two reconstructions exactly the same way. See
//! `specs/gui/scene-graph.md`, "Node Transforms and Alignment", for the
//! behaviour this drives.
//!
//! What stays here is the viewer's own share: the popup's options (re-exported
//! from core, since they *are* the fit's options) and the status-line text the
//! outcome is reported in. Composing the fitted transform into the target's
//! displayed frame is the caller's job (see [`crate::state::AppState::align_node`]).

pub use sfmtool_core::analysis::alignment::{
    align_reconstructions, AlignFit, AlignOptions, AlignSource,
};

/// `Aligned run_b → run_a: 214/243 cameras, RMS 0.031`.
pub fn success_message(source_label: &str, target_label: &str, fit: &AlignFit) -> String {
    format!(
        "Aligned {source_label} → {target_label}: {}/{} {}, RMS {:.3}",
        fit.inliers,
        fit.correspondences,
        fit.source.noun(),
        fit.rms,
    )
}

/// `Align run_b → run_a failed: <reason>`. The node's transform is left exactly
/// as it was.
pub fn failure_message(source_label: &str, target_label: &str, reason: &str) -> String {
    format!("Align {source_label} → {target_label} failed: {reason}")
}

#[cfg(test)]
mod tests;
