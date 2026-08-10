// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Go to Point: type or paste a point index — or a whole `pt3d_<hash>_<index>`
//! id — and land on that point.
//!
//! See `specs/gui/gui-goto-point.md`. Every other way to reach a 3D point in
//! this viewer is a *click*: on a splat in the 3D viewport, on a feature in the
//! Image Detail overlay, on a row of the Scene panel. That leaves no way back
//! in from the outside — from a Point ID copied into a notes file, a constraints
//! table or a CLI run — which is exactly what the ID format exists for. This
//! module is that way back in.
//!
//! The parse ([`PointQuery`]) and the scene lookup ([`resolve_point_query`])
//! are plain functions over the scene slice, so the interesting behaviour is
//! testable without a frame; [`GotoPointDialog`] is the thin egui shell that
//! collects the text and reports the [`PointRef`] it resolved to.

use crate::scene::{hash_prefix, node_by_id, selected_node, PointRef, ReconId, SceneNode};

/// The keyboard shortcut that opens the dialog: Ctrl+G, or Cmd+G on macOS.
///
/// `COMMAND` rather than `CTRL` so it follows the platform's own convention,
/// the same way egui's own built-in shortcuts do.
pub const SHORTCUT: egui::KeyboardShortcut =
    egui::KeyboardShortcut::new(egui::Modifiers::COMMAND, egui::Key::G);

/// What the user typed, once it has been recognized as a point reference.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PointQuery {
    /// A bare index, which names a point but not a file — so it resolves
    /// against whichever reconstruction is currently selected.
    Index(usize),
    /// A full `pt3d_<hash>_<index>` id: the hash names the reconstruction, so
    /// this resolves on its own and may *change* the selected reconstruction.
    Qualified {
        /// The hash as typed, lowercased. Usually the 8 characters a displayed
        /// Point ID carries, but any prefix of a `content_xxh128` is accepted.
        hash: String,
        index: usize,
    },
}

/// Recognize `input` as a point index or a full Point ID.
///
/// Tolerates surrounding whitespace, a leading `#` (the Image Detail tooltip
/// prints `Point3D #12345`, and that `#12345` is a natural thing to retype) and
/// a Point ID in any case. Anything else is rejected with a message that shows
/// both accepted shapes rather than trying to guess what was meant.
pub fn parse_point_query(input: &str) -> Result<PointQuery, String> {
    let text = input.trim();
    let bare = text.strip_prefix('#').unwrap_or(text).trim();

    if !bare.is_empty() && bare.bytes().all(|b| b.is_ascii_digit()) {
        return bare
            .parse::<usize>()
            .map(PointQuery::Index)
            .map_err(|_| format!("Point index {bare} is too large."));
    }

    // `splitn(3, '_')` rather than a full split: only the first two separators
    // are structural, and a trailing part that is not a number is caught below
    // as "not a number" rather than silently ignored.
    let mut parts = text.splitn(3, '_');
    if let (Some(prefix), Some(hash), Some(index)) = (parts.next(), parts.next(), parts.next()) {
        if prefix.eq_ignore_ascii_case("pt3d") {
            if hash.is_empty() || !hash.bytes().all(|b| b.is_ascii_hexdigit()) {
                return Err(format!(
                    "{hash:?} is not a reconstruction hash — expected hex digits, as in \
                     pt3d_a1b2c3d4_12345."
                ));
            }
            let Ok(index) = index.parse::<usize>() else {
                return Err(format!(
                    "{index:?} is not a point index — expected a number, as in \
                     pt3d_a1b2c3d4_12345."
                ));
            };
            return Ok(PointQuery::Qualified {
                hash: hash.to_ascii_lowercase(),
                index,
            });
        }
    }

    Err(format!(
        "{text:?} is not a point index or ID. Expected 12345 or pt3d_a1b2c3d4_12345."
    ))
}

/// Find the point `query` names among the loaded reconstructions.
///
/// A bare index resolves against `selected`; a qualified id resolves against
/// whichever loaded node carries that content hash, which is what lets a pasted
/// ID select a *different* reconstruction than the one in front of you.
///
/// The index is bounds-checked here rather than left to the panels: a selection
/// that points past the end of its own reconstruction would show as an empty
/// Point Track panel with nothing to say why.
pub fn resolve_point_query(
    scene: &[SceneNode],
    selected: Option<ReconId>,
    query: &PointQuery,
) -> Result<PointRef, String> {
    let node = match query {
        PointQuery::Index(_) => selected_node(scene, selected)
            .ok_or_else(|| "No reconstruction is loaded — use File ▸ Open first.".to_string())?,
        PointQuery::Qualified { hash, .. } => {
            find_by_hash(scene, selected, hash).ok_or_else(|| {
                format!(
                    "No loaded reconstruction has content hash {hash} — open the .sfmr file \
                     this ID came from."
                )
            })?
        }
    };

    let index = match *query {
        PointQuery::Index(index) => index,
        PointQuery::Qualified { index, .. } => index,
    };
    let count = node.recon.points.len();
    if index >= count {
        return Err(format!(
            "{} has {count} points — index {index} is out of range.",
            node.label
        ));
    }
    Ok(PointRef::new(node.id, index))
}

/// The loaded node whose content hash starts with `hash`, preferring the
/// selected one.
///
/// Several nodes can match: the same file opened from two paths shares a
/// content hash, and so do any two reconstructions carrying *no* hash, which
/// both display as `00000000`. Every match holds the same content, so the index
/// means the same thing in each — preferring the selected node just keeps the
/// answer where the user is already looking instead of jumping them elsewhere
/// for no visible reason.
fn find_by_hash<'a>(
    scene: &'a [SceneNode],
    selected: Option<ReconId>,
    hash: &str,
) -> Option<&'a SceneNode> {
    selected
        .and_then(|id| node_by_id(scene, id))
        .filter(|node| hash_matches(node, hash))
        .or_else(|| scene.iter().find(|node| hash_matches(node, hash)))
}

/// Whether `hash` is a prefix of this node's `content_xxh128`.
///
/// The displayed 8-character prefix is the common case, but a full 32-character
/// hash out of `content_hash.json.zst` matches too — the format spec offers it
/// for exact disambiguation, so pasting one should work. The `hash_prefix`
/// fallback covers a reconstruction with no hash at all, which displays (and so
/// must resolve) as `00000000`.
fn hash_matches(node: &SceneNode, hash: &str) -> bool {
    let full = &node.recon.content_hash.content_xxh128;
    full.len() >= hash.len() && full[..hash.len()].eq_ignore_ascii_case(hash)
        || hash_prefix(&node.recon).eq_ignore_ascii_case(hash)
}

/// The modal that collects the text.
///
/// Opened from `Go ▸ Go to Point…`, from the shortcut, or from the Point Track
/// panel's own button; it stays open on a bad query so the message sits right
/// under the text that caused it and can be corrected in place.
#[derive(Default)]
pub struct GotoPointDialog {
    /// Whether the window is showing.
    open: bool,
    /// The text field's contents, kept across opens so a mistyped index can be
    /// corrected rather than retyped.
    input: String,
    /// What went wrong with the last submission, cleared on the next one.
    error: Option<String>,
    /// Set when the dialog opens, consumed by the first frame that draws the
    /// text field — which is the frame that can actually focus it.
    focus_pending: bool,
}

impl GotoPointDialog {
    /// Show the dialog, ready for typing.
    ///
    /// Idempotent: opening an already-open dialog just re-focuses the field,
    /// so the menu item and the shortcut cannot fight over it.
    pub fn open(&mut self) {
        self.open = true;
        self.error = None;
        self.focus_pending = true;
    }

    fn close(&mut self) {
        self.open = false;
        self.error = None;
        self.focus_pending = false;
    }

    /// Draw one frame of the dialog, returning the point to select when the
    /// user submits a query that resolves.
    ///
    /// Returning the [`PointRef`] rather than applying it keeps this free of
    /// `AppState`: the caller owns what "go there" means (select the point,
    /// and raise the Point Track tab so the result is visible).
    pub fn show(
        &mut self,
        ctx: &egui::Context,
        scene: &[SceneNode],
        selected: Option<ReconId>,
    ) -> Option<PointRef> {
        if !self.open {
            return None;
        }

        let mut still_open = true;
        let mut submitted = false;
        let mut cancelled = false;

        egui::Window::new("Go to Point")
            .open(&mut still_open)
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .show(ctx, |ui| {
                ui.label("Point index, or full ID with hash:");
                let edit = ui.add(
                    egui::TextEdit::singleline(&mut self.input)
                        .hint_text("12345   or   pt3d_a1b2c3d4_12345")
                        .desired_width(280.0),
                );
                if std::mem::take(&mut self.focus_pending) {
                    edit.request_focus();
                }
                // `lost_focus` rather than `has_focus`: egui reports Enter on
                // the frame the field gives focus up, and pairing the two is
                // what distinguishes submitting from clicking away.
                submitted |= edit.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));

                ui.label(
                    egui::RichText::new(
                        "A bare index refers to the selected reconstruction; a full ID \
                         selects the one it names.",
                    )
                    .weak()
                    .small(),
                );

                if let Some(error) = &self.error {
                    ui.add_space(4.0);
                    ui.colored_label(ui.visuals().error_fg_color, error);
                }

                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    let ready = !self.input.trim().is_empty();
                    if ui
                        .add_enabled(ready, egui::Button::new("Go"))
                        .on_disabled_hover_text("Type a point index or ID first")
                        .clicked()
                    {
                        submitted = true;
                    }
                    if ui.button("Cancel").clicked() {
                        cancelled = true;
                    }
                });

                cancelled |= ui.input(|i| i.key_pressed(egui::Key::Escape));
            });

        if !still_open || cancelled {
            self.close();
            return None;
        }
        if !submitted || self.input.trim().is_empty() {
            return None;
        }

        match parse_point_query(&self.input)
            .and_then(|query| resolve_point_query(scene, selected, &query))
        {
            Ok(point) => {
                self.close();
                Some(point)
            }
            Err(message) => {
                self.error = Some(message);
                // Enter took focus off the field to submit; put it back so the
                // correction can be typed straight away.
                self.focus_pending = true;
                None
            }
        }
    }
}

#[cfg(test)]
mod tests;
