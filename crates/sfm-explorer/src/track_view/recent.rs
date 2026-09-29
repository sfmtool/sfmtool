// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The recent items strip: the items most recently focused, on the *Edit*
//! box's row to its right, each one click away.
//!
//! See `specs/gui/track-view.md` § "The recent items strip". The entries are
//! [`AppState::recent_items`] read at the cursor: most recent first, the
//! focused item left out, an entry whose item is not on its node's bench
//! skipped rather than removed, and at most [`MAX_CHIPS`] of them. The strip
//! draws as many as fit whole in the width the row has left.
//!
//! A chip is the item's patch, the picture the header's patch slot draws
//! ([`crate::track_view::body::track_patch_image`]), and its label cut in the
//! middle to [`CHIP_LABEL_WIDTH`]. A click lands in the value [`RecentStrip::show`]
//! returns, and the dock applies it through `AppState::focus_bench_item`.

use std::collections::HashMap;
use std::sync::Arc;

use sfmtool_core::bench::{EditableTrack, ItemId, Stage};

use crate::bench::live::Evaluation;
use crate::scene::ReconId;
use crate::state::AppState;

/// The most chips the strip draws, however wide the row is.
pub(crate) const MAX_CHIPS: usize = 8;

/// A chip's patch, in points a side.
const CHIP_PATCH_SIZE: f32 = 24.0;

/// The patch in a chip's hover text, in points a side.
const HOVER_PATCH_SIZE: f32 = 64.0;

/// The widest a chip's label is drawn. A longer label loses its middle.
const CHIP_LABEL_WIDTH: f32 = 72.0;

/// The space between a chip's patch and its label, and at each end of the
/// chip.
const CHIP_PAD: f32 = 4.0;

/// One item the strip can draw: its node, its ID, its label at the cursor and
/// its track.
struct Entry {
    node: ReconId,
    item: ItemId,
    label: String,
    track: Arc<EditableTrack>,
}

/// One chip the strip drew on the last frame.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct DrawnChip {
    /// The item's node.
    pub(crate) node: ReconId,
    /// The item's whole label.
    pub(crate) label: String,
    /// The label as the chip drew it, cut in the middle where it was too long.
    pub(crate) shown: String,
    /// The texture the chip drew its patch with, or `None` for the empty
    /// frame drawn for an item with neither a bitmap nor a template.
    pub(crate) patch: Option<egui::TextureId>,
}

/// The strip's state: the uploaded patches, and what it drew last frame.
#[derive(Default)]
pub(crate) struct RecentStrip {
    /// Each drawn item's patch, uploaded, with the track value it was made
    /// from. Held against the track's `Arc` rather than its address, so a
    /// freed track's address cannot be taken for a new one; a step on the
    /// item gives it a new `Arc`, which is what says the upload is stale.
    /// An entry the last frame did not draw is dropped.
    patches: HashMap<(ReconId, ItemId), (Arc<EditableTrack>, Option<egui::TextureHandle>)>,
    /// What the strip drew last frame, in order.
    ///
    /// Recorded unconditionally rather than under `cfg(test)`, so that what
    /// the tests read is the very strip the app draws.
    drawn: Vec<DrawnChip>,
}

impl RecentStrip {
    /// What the strip drew last frame, in order.
    #[cfg(test)]
    pub(crate) fn drawn(&self) -> &[DrawnChip] {
        &self.drawn
    }

    /// Draw the strip into the rest of the row and return the item whose chip
    /// was clicked, as its node and its label.
    pub(crate) fn show(
        &mut self,
        ui: &mut egui::Ui,
        state: &AppState,
    ) -> Option<(ReconId, String)> {
        self.drawn.clear();
        let font = egui::TextStyle::Body.resolve(ui.style());
        let color = ui.visuals().text_color();
        let spacing = ui.spacing().item_spacing.x;
        let mut room = ui.available_width();
        let mut clicked = None;
        let mut drawn = Vec::new();
        for entry in entries(state) {
            let shown = crate::elide::middle(&entry.label, CHIP_LABEL_WIDTH, |value| {
                ui.ctx().fonts_mut(|fonts| {
                    fonts
                        .layout_no_wrap(value.to_owned(), font.clone(), color)
                        .rect
                        .width()
                })
            });
            let galley = ui
                .painter()
                .layout_no_wrap(shown.clone(), font.clone(), color);
            let width = CHIP_PAD + CHIP_PATCH_SIZE + CHIP_PAD + galley.size().x + CHIP_PAD;
            // A chip that would not fit whole is not drawn, and neither is any
            // after it, so the order stays most recent first.
            if width > room {
                break;
            }
            room -= width + spacing;
            drawn.push((entry.node, entry.item));
            let patch = self.ensure_patch(ui.ctx(), &entry);
            let (rect, response) =
                ui.allocate_exact_size(egui::vec2(width, CHIP_PATCH_SIZE), egui::Sense::click());
            if response.hovered() {
                ui.painter()
                    .rect_filled(rect, 3.0, ui.visuals().widgets.hovered.weak_bg_fill);
            }
            let patch_rect = egui::Rect::from_min_size(
                rect.min + egui::vec2(CHIP_PAD, 0.0),
                egui::vec2(CHIP_PATCH_SIZE, CHIP_PATCH_SIZE),
            );
            draw_patch(ui, patch_rect, patch);
            ui.painter().galley(
                egui::pos2(
                    patch_rect.max.x + CHIP_PAD,
                    rect.center().y - galley.size().y / 2.0,
                ),
                galley,
                color,
            );
            let response = response.on_hover_ui(|ui| show_hover(ui, state, &entry, patch));
            if response.clicked() {
                clicked = Some((entry.node, entry.label.clone()));
            }
            self.drawn.push(DrawnChip {
                node: entry.node,
                label: entry.label,
                shown,
                patch,
            });
        }
        self.patches.retain(|key, _| drawn.contains(key));
        clicked
    }

    /// The item's patch, uploaded on the first frame that draws it since its
    /// track moved.
    fn ensure_patch(&mut self, ctx: &egui::Context, entry: &Entry) -> Option<egui::TextureId> {
        let key = (entry.node, entry.item);
        let stale = self
            .patches
            .get(&key)
            .is_none_or(|(track, _)| !Arc::ptr_eq(track, &entry.track));
        if stale {
            let texture = crate::track_view::body::track_patch_image(&entry.track).map(|image| {
                ctx.load_texture(
                    format!("recent_item_patch_{:?}_{}", entry.node, entry.label),
                    image,
                    egui::TextureOptions::NEAREST,
                )
            });
            self.patches
                .insert(key, (Arc::clone(&entry.track), texture));
        }
        self.patches
            .get(&key)
            .and_then(|(_, texture)| texture.as_ref().map(|texture| texture.id()))
    }
}

/// The items the strip can draw, in order: [`AppState::recent_items`] with
/// the focused item left out and every entry not on its node's bench at the
/// cursor skipped, cut at [`MAX_CHIPS`].
fn entries(state: &AppState) -> Vec<Entry> {
    let focused = state.focused_item().copied();
    state
        .recent_items
        .iter()
        .filter(|recent| Some(**recent) != focused)
        .filter_map(|recent| {
            let bench = state.bench(recent.node)?;
            let label = bench.label_of(recent.item)?;
            Some(Entry {
                node: recent.node,
                item: recent.item,
                label: label.to_string(),
                track: Arc::clone(bench.track(label)?),
            })
        })
        .take(MAX_CHIPS)
        .collect()
}

/// A patch at `rect`, or the empty frame drawn where there is none.
fn draw_patch(ui: &egui::Ui, rect: egui::Rect, patch: Option<egui::TextureId>) {
    match patch {
        Some(texture) => {
            ui.painter().image(
                texture,
                rect,
                egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                egui::Color32::WHITE,
            );
        }
        None => {
            ui.painter()
                .rect_filled(rect, 2.0, ui.visuals().faint_bg_color);
            ui.painter().rect_stroke(
                rect,
                2.0,
                egui::Stroke::new(1.0_f32, ui.visuals().weak_text_color()),
                egui::StrokeKind::Inside,
            );
        }
    }
}

/// A chip's hover text: the patch at [`HOVER_PATCH_SIZE`], then what the item
/// is, where it came from, where it is, where its evaluation stands, and what a
/// click does.
fn show_hover(ui: &mut egui::Ui, state: &AppState, entry: &Entry, patch: Option<egui::TextureId>) {
    let (rect, _) = ui.allocate_exact_size(
        egui::vec2(HOVER_PATCH_SIZE, HOVER_PATCH_SIZE),
        egui::Sense::hover(),
    );
    draw_patch(ui, rect, patch);
    for line in hover_lines(state, entry) {
        ui.label(line);
    }
    ui.weak(CLICK_TO_EDIT);
}

/// The last line of a chip's hover text.
pub(crate) const CLICK_TO_EDIT: &str = "Click to edit";

/// The lines of a chip's hover text between the patch and [`CLICK_TO_EDIT`].
fn hover_lines(state: &AppState, entry: &Entry) -> Vec<String> {
    let track = &entry.track;
    let mut lines = vec![entry.label.clone()];
    let node = state.node(entry.node);
    if state.scene.len() > 1 {
        if let Some(node) = node {
            lines.push(format!("On {}", node.label));
        }
    }
    lines.push(format!("The {} stage", track.stage_kind()));
    let (kept, out) = track.verdict_counts();
    let pinned = track.observations.iter().filter(|o| o.pinned).count();
    lines.push(format!("{kept} kept · {out} out · {pinned} pinned"));
    let resolved = node.and_then(|node| {
        let point = state.resolved_origin(node, track)?;
        Some(crate::scene::point_id(node, point as usize))
    });
    lines.push(match (track.origin, resolved) {
        (_, Some(id)) => format!("Point {id}"),
        (Some(origin), None) => format!("From point {}, which is gone", origin.point),
        (None, None) => "New: read from no point".to_string(),
    });
    if let Stage::Track(payload) = &track.stage {
        lines.push(crate::track_view::body::position_text(payload));
    }
    lines.push(
        match state
            .bench_evaluation(entry.node, &entry.label)
            .unwrap_or(Evaluation::Evaluating)
        {
            Evaluation::Current => crate::track_view::body::EVALUATED_LABEL.to_string(),
            Evaluation::Evaluating => crate::track_view::body::EVALUATING_LABEL.to_string(),
            Evaluation::Refused(why) | Evaluation::Failed(why) => why,
        },
    );
    lines
}
