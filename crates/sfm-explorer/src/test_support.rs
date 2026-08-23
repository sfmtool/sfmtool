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

/// Every string painted in one headless frame, in paint order.
///
/// A panel that elides its own text has to be asked what it actually drew, and
/// egui's frame output carries the galleys: laying out needs no GPU, so the
/// strings are real even with no painter behind them. Nested `Shape::Vec`s are
/// walked, since a widget's shapes arrive grouped.
pub(crate) fn painted_texts(
    ctx: &egui::Context,
    input: egui::RawInput,
    run_ui: impl FnMut(&mut egui::Ui),
) -> Vec<String> {
    let mut output = ctx.run_ui(input, run_ui);
    output.textures_delta.clear();
    let mut texts = Vec::new();
    for clipped in &output.shapes {
        collect_texts(&clipped.shape, &mut texts);
    }
    texts
}

fn collect_texts(shape: &egui::Shape, out: &mut Vec<String>) {
    match shape {
        egui::Shape::Text(text) => out.push(text.galley.text().to_owned()),
        egui::Shape::Vec(shapes) => {
            for shape in shapes {
                collect_texts(shape, out);
            }
        }
        _ => {}
    }
}
