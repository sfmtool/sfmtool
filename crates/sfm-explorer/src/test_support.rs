// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Helpers shared by the headless UI tests.

/// Run one egui frame headlessly and discard its output.
///
/// The wrapper exists for the discarding: since epaint 0.36 a
/// [`TexturesDelta`](egui::TexturesDelta) panics on drop if it still holds
/// unapplied deltas, on the grounds that a real integration losing them is a
/// bug ("texture has not been allocated yet" on a later partial update — the
/// same hazard `app.rs` documents around its `update_texture` loop). These
/// frames have no painter at all, so the deltas are dropped deliberately, which
/// is exactly the case epaint asks be spelled out with an explicit `clear`.
pub(crate) fn run_frame_headless(
    ctx: &egui::Context,
    input: egui::RawInput,
    run_ui: impl FnMut(&mut egui::Ui),
) {
    let mut output = ctx.run_ui(input, run_ui);
    output.textures_delta.clear();
}
