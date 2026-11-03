// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The fonts the viewer draws its text in.
//!
//! These are egui's bundled fonts, in egui's order, with one addition: Hack,
//! the first face of the monospace family, is also the last face of the
//! proportional family. None of the proportional family's own three faces,
//! Ubuntu-Light and the two emoji faces behind it, has a glyph for `→`, `▸`
//! or `●`, so without Hack a tooltip or a label that says `0.101 → 0.105 px`
//! draws the arrow as an empty box, while the same string in the monospace
//! font draws it. As the last face, Hack draws only what the
//! proportional family's own faces lack, so no text that drew before changes
//! its font.
//!
//! See `specs/gui/architecture.md` § "Fonts".

/// The bundled face added at the end of the proportional family: the
/// monospace family's first face, which has `→`, `▸` and `●`.
const SHARED_FACE: &str = "Hack";

/// egui's bundled font definitions, with [`SHARED_FACE`] added as the
/// proportional family's last fallback.
fn font_definitions() -> egui::FontDefinitions {
    let mut fonts = egui::FontDefinitions::default();
    let proportional = fonts
        .families
        .entry(egui::FontFamily::Proportional)
        .or_default();
    if !proportional.iter().any(|face| face == SHARED_FACE) {
        proportional.push(SHARED_FACE.to_owned());
    }
    fonts
}

/// Give `ctx` the viewer's fonts. Called once, when the viewer's egui
/// context is made.
pub(crate) fn install(ctx: &egui::Context) {
    ctx.set_fonts(font_definitions());
}

/// The characters of `text`, other than whitespace, that `family` in the
/// viewer's fonts has no glyph for, each once and in order.
///
/// It asks the family's faces for their character maps rather than calling
/// egui's `has_glyph`, which answers no for every character of the face that
/// also holds the replacement glyph: Hack, the monospace family's first face.
#[cfg(test)]
pub(crate) fn lacking_glyphs(family: &egui::FontFamily, text: &str) -> String {
    let ctx = egui::Context::default();
    install(&ctx);
    // A frame makes the context build its fonts from the definitions.
    crate::test_support::run_frame_headless(&ctx, egui::RawInput::default(), |_| {});
    ctx.fonts_mut(|fonts| {
        let mut font = fonts.fonts.font(family);
        let characters = font.characters();
        let mut lacking = String::new();
        for c in text.chars() {
            if !c.is_whitespace() && !characters.contains_key(&c) && !lacking.contains(c) {
                lacking.push(c);
            }
        }
        lacking
    })
}

#[cfg(test)]
mod tests;
