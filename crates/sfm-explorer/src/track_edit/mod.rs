// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The Track Edit panel: the bench's active track, and the steps that act on
//! it.
//!
//! See `specs/gui/track-edit.md`. This is the first **bench panel**: it shows
//! the active track of the selected node's bench ([`crate::bench`]) and every
//! gesture in it names that track. The Point Track Detail panel beside it stays
//! what it is -- the view of the *selected point's* committed track -- and this
//! panel's table carries that panel's columns first, so a reader who knows one
//! reads the other.
//!
//! The panel decides nothing. Each gesture lands in [`TrackEditResponse`] and
//! the dock applies it through the `AppState` method that pushes the version,
//! for the reason every other panel's response works that way: the panel holds
//! `&AppState` while it draws, and a step needs it mutably.
//!
//! Almost no state lives here. The bench is the node's, at its cursor, so what
//! the panel owns is the slider positions, the row selection a split reads, the
//! thumbnails it has loaded, and the painting the sliders produce -- which is
//! cached against the track's own `Arc` rather than recomputed per frame,
//! because the painting is `apply_thresholds` run over a copy and a copy of a
//! track carries its consensus bitmap.

use std::collections::HashMap;

use ndarray::Axis;
use sfmtool_core::bench::{
    apply_thresholds, Bench, EditableTrack, Observation, Provenance, StageKind, Thresholds, Verdict,
};
use sfmtool_core::SfmrReconstruction;

use crate::scene::{ImageRef, ReconId, SceneNode};
use crate::state::AppState;
use crate::texture::thumbnail_color_image;

mod table;

#[cfg(test)]
mod tests;

pub(crate) use table::RowSummary;

/// What one frame of the panel asks the dock to do.
///
/// Every field is one gesture, and at most one of them is set on a frame: the
/// entries that push a version are buttons, and a button is clicked once.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct TrackEditResponse {
    /// A tab or a Scene-tree row asked for this item to become the active one.
    pub activate: Option<String>,
    /// A tab's close mark, or the toolbar's *Discard*.
    pub discard: Option<String>,
    /// *Rename* was committed: the item, and the label it should take.
    pub rename: Option<(String, String)>,
    /// *Put selected point on bench*.
    pub put_selected_point_on_bench: bool,
    /// *Start cluster here*, at the Image Detail panel's current pixel.
    pub start_cluster: bool,
    /// *Add observation here*, at the same pixel, on the active track.
    pub add_observation: bool,
    /// *Evaluate*.
    pub evaluate: bool,
    /// The *Stage* toggle, carrying the stage it asks for.
    pub set_stage: Option<StageKind>,
    /// *Apply thresholds*, carrying the bars the sliders stand at.
    pub apply_thresholds: Option<Thresholds>,
    /// *Split off selected rows*, carrying the rows.
    pub split: Option<Vec<usize>>,
    /// *Commit*.
    pub commit: bool,
    /// A verdict control was clicked: the observation, and the verdict it
    /// cycled to.
    pub set_verdict: Option<(usize, Verdict)>,
    /// A row was clicked -- select this image, as the view-only panel does.
    pub select_image: Option<usize>,
    /// The image under the pointer, for cross-panel hover.
    pub hovered_image: Option<usize>,
    /// Whether the pointer is inside the panel.
    pub has_pointer: bool,
}

/// Track Edit panel state.
pub struct TrackEdit {
    /// Where the threshold sliders stand. Panel state: a slider proposes and
    /// *Apply thresholds* is what makes the proposal verdicts, so moving one
    /// pushes no version and closing the panel keeps where they were left.
    thresholds: Thresholds,
    /// The verdicts the sliders propose for the active track, one per
    /// observation, which is what the rows are painted by.
    painted: Vec<Verdict>,
    /// The item and the exact track value [`TrackEdit::painted`] was computed
    /// from: the label, and the address of the track's `Arc`. A step on the
    /// track gives it a new `Arc`, which is what says the painting is stale.
    painted_for: Option<(String, usize, Thresholds)>,
    /// Why the active track cannot be committed, or `None` when it can, as of
    /// the value and the track [`TrackEdit::commit_refusal_for`] last asked.
    ///
    /// Cached rather than asked per frame: the question is
    /// `sfmtool_core::bench::commit` itself, asked of the very track the button
    /// would commit so that the button and the step cannot disagree, and that
    /// builds a point record. It is stale exactly when the track's `Arc` or the
    /// node's version moves, which is what the key below holds.
    commit_refusal: Option<String>,
    /// The item, the track's `Arc` and the version [`TrackEdit::commit_refusal`]
    /// was asked at.
    commit_refusal_for: Option<(String, usize, u64)>,
    /// The rows the user has selected, by observation index, ascending. Panel
    /// state rather than a version: it is what *Split off selected rows* reads,
    /// and nothing else.
    selected_rows: Vec<usize>,
    /// Which item those rows belong to, so a change of the active item clears
    /// them rather than applying them to another track's observations.
    selection_of: Option<(ReconId, String)>,
    /// A rename in progress: the item, and the text typed so far.
    renaming: Option<(String, String)>,
    /// Thumbnail textures, keyed by image, as the view-only panel keys its own.
    thumbnail_textures: HashMap<ImageRef, egui::TextureHandle>,
    /// What the table drew last frame, in row order.
    ///
    /// Recorded unconditionally rather than under `cfg(test)`, so that what the
    /// tests read is the very table the app draws.
    rows: Vec<RowSummary>,
    /// Tracked vertical scroll offset.
    scroll_offset_y: Option<f32>,
}

impl Default for TrackEdit {
    fn default() -> Self {
        Self::new()
    }
}

impl TrackEdit {
    pub fn new() -> Self {
        Self {
            thresholds: Thresholds::default(),
            painted: Vec::new(),
            painted_for: None,
            commit_refusal: None,
            commit_refusal_for: None,
            selected_rows: Vec::new(),
            selection_of: None,
            renaming: None,
            thumbnail_textures: HashMap::new(),
            rows: Vec::new(),
            scroll_offset_y: None,
        }
    }

    /// What the table drew last frame, in row order.
    #[cfg(test)]
    pub(crate) fn rows(&self) -> &[RowSummary] {
        &self.rows
    }

    /// Where the sliders stand.
    #[cfg(test)]
    pub(crate) fn thresholds(&self) -> &Thresholds {
        &self.thresholds
    }

    /// Drop everything cached for a reconstruction that has left the scene.
    pub fn forget_recon(&mut self, id: ReconId) {
        self.thumbnail_textures.retain(|image, _| image.recon != id);
        if self.selection_of.as_ref().is_some_and(|(of, _)| *of == id) {
            self.selected_rows.clear();
            self.selection_of = None;
        }
        self.painted_for = None;
        self.commit_refusal_for = None;
        self.painted.clear();
        self.rows.clear();
    }

    /// Draw the panel and report what the user did with it.
    pub fn show(&mut self, ui: &mut egui::Ui, state: &AppState) -> TrackEditResponse {
        let mut response = TrackEditResponse::default();
        let panel_rect = ui.available_rect_before_wrap();
        if let Some(pos) = ui.input(|i| i.pointer.hover_pos()) {
            response.has_pointer = panel_rect.contains(pos);
        }

        let Some(node) = crate::scene::selected_node(&state.scene, state.selected_recon) else {
            self.rows.clear();
            ui.centered_and_justified(|ui| {
                ui.label("No reconstruction selected");
            });
            return response;
        };
        let id = node.id;
        let bench = node.history.current_bench();

        self.show_item_tabs(ui, bench, &mut response);

        let active = crate::bench::active_track_label(bench).map(str::to_string);
        let Some(label) = active else {
            self.rows.clear();
            self.selected_rows.clear();
            self.selection_of = None;
            show_empty_state(ui, state, node, &mut response);
            return response;
        };
        let track = bench.track(&label).expect("the active label names a track");

        // A change of item takes the row selection with it: the rows are
        // observation indexes, and another track's observations are not these.
        if self.selection_of.as_ref() != Some(&(id, label.clone())) {
            self.selected_rows.clear();
            self.selection_of = Some((id, label.clone()));
        }
        self.repaint_if_stale(&label, track);
        self.recheck_commit_if_stale(&label, track, node);

        show_header(ui, &label, track);
        self.show_toolbar(ui, state, node, &label, track, &mut response);
        self.show_thresholds(ui);
        ui.separator();
        self.show_table(ui, node.recon(), id, state, track, &mut response);
        response
    }

    /// The row of tabs, one per track on the bench, the active one raised.
    fn show_item_tabs(
        &mut self,
        ui: &mut egui::Ui,
        bench: &Bench,
        response: &mut TrackEditResponse,
    ) {
        if bench.is_empty() {
            return;
        }
        let active = crate::bench::active_track_label(bench);
        ui.horizontal_wrapped(|ui| {
            for entry in bench.entries() {
                let Some(track) = entry.item.as_track() else {
                    continue;
                };
                let selected = active == Some(entry.label.as_str());
                let text = format!("{} ({} in)", entry.label, track.verdict_counts().0);
                if ui
                    .add(egui::Button::selectable(selected, text))
                    .on_hover_text(format!(
                        "{} · the {} stage",
                        entry.label,
                        track.stage_kind()
                    ))
                    .clicked()
                {
                    response.activate = Some(entry.label.clone());
                }
                if ui
                    .small_button("x")
                    .on_hover_text(format!("Discard {} from the bench", entry.label))
                    .clicked()
                {
                    response.discard = Some(entry.label.clone());
                }
            }
        });
        ui.separator();
    }

    /// The toolbar: every step that acts on the active track, each greyed with
    /// the sentence naming what is missing.
    fn show_toolbar(
        &mut self,
        ui: &mut egui::Ui,
        state: &AppState,
        node: &SceneNode,
        label: &str,
        track: &EditableTrack,
        response: &mut TrackEditResponse,
    ) {
        let id = node.id;
        let busy = state.busy_refusal(id);
        let pixel = pixel_refusal(state, id);
        ui.horizontal_wrapped(|ui| {
            if entry(
                ui,
                "Evaluate",
                busy.clone(),
                "Measure every observation at the track's own stage",
            ) {
                response.evaluate = true;
            }
            let (next, stage_label) = match track.stage_kind() {
                StageKind::Cluster => (StageKind::Track, "Stage: cluster \u{2192} track"),
                StageKind::Track => (StageKind::Cluster, "Stage: track \u{2192} cluster"),
            };
            if entry(
                ui,
                stage_label,
                busy.clone(),
                "Move the track between its two representations",
            ) {
                response.set_stage = Some(next);
            }
            if entry(
                ui,
                "Apply thresholds",
                busy.clone(),
                "Turn the painting into verdicts, leaving the pinned ones alone",
            ) {
                response.apply_thresholds = Some(self.thresholds.clone());
            }
            let split_refusal = busy.clone().or_else(|| split_refusal(self, track));
            if entry(
                ui,
                &format!("Split off {} rows", self.selected_rows.len()),
                split_refusal,
                "Move the selected rows onto a second track beside this one",
            ) {
                response.split = Some(self.selected_rows.clone());
            }
            let commit_refusal = busy.clone().or_else(|| self.commit_refusal.clone());
            if entry(
                ui,
                "Commit",
                commit_refusal,
                "Write the track into the reconstruction",
            ) {
                response.commit = true;
            }
            if entry(ui, "Discard", busy.clone(), "Take this track off the bench") {
                response.discard = Some(label.to_string());
            }
        });
        ui.horizontal_wrapped(|ui| {
            let point_refusal = busy.clone().or_else(|| {
                (state.selected_point.filter(|p| p.recon == id).is_none())
                    .then(|| "No point is selected.".to_string())
            });
            if entry(
                ui,
                PUT_ON_BENCH_LABEL,
                point_refusal,
                "Put the selected point's track on the bench and work on it",
            ) {
                response.put_selected_point_on_bench = true;
            }
            if entry(
                ui,
                "Start cluster here",
                busy.clone().or_else(|| pixel.clone()),
                "Start a cluster-stage track at the pixel named in Image Detail",
            ) {
                response.start_cluster = true;
            }
            if entry(
                ui,
                "Add observation here",
                busy.clone().or(pixel),
                "Add a candidate at the pixel named in Image Detail",
            ) {
                response.add_observation = true;
            }
            self.show_rename(ui, label, busy, response);
        });
    }

    /// *Rename*, which opens a field in place and commits on Enter.
    fn show_rename(
        &mut self,
        ui: &mut egui::Ui,
        label: &str,
        busy: Option<String>,
        response: &mut TrackEditResponse,
    ) {
        match self.renaming.as_mut() {
            None => {
                if entry(ui, "Rename", busy, "Give this item a name of your own") {
                    self.renaming = Some((label.to_string(), label.to_string()));
                }
            }
            Some((item, text)) => {
                let item = item.clone();
                let field = ui.add(egui::TextEdit::singleline(text).desired_width(140.0));
                let commit = field.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
                let typed = text.clone();
                if commit || ui.button("Set").clicked() {
                    response.rename = Some((item, typed));
                    self.renaming = None;
                } else if ui.button("Cancel").clicked() {
                    self.renaming = None;
                }
            }
        }
    }

    /// The threshold sliders, which paint the table live and change nothing
    /// until *Apply thresholds* is pressed.
    fn show_thresholds(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_wrapped(|ui| {
            ui.label("Thresholds");
            ui.add(
                egui::Slider::new(&mut self.thresholds.min_zncc, 0.0..=1.0)
                    .text("min ZNCC")
                    .max_decimals(2),
            );
            ui.add(
                egui::Slider::new(&mut self.thresholds.max_shift_px, 0.0..=20.0)
                    .text("max shift px")
                    .max_decimals(2),
            );
            ui.add(
                egui::Slider::new(&mut self.thresholds.max_keypoint_uncertainty, 0.0..=2.0)
                    .text("max \u{3c3}_pos")
                    .max_decimals(2),
            );
        });
    }

    /// Recompute the painting when the track or the bars have moved.
    ///
    /// The painting **is** what applying the thresholds would do, computed by
    /// the same core function the button calls, so a row can never be painted
    /// one way and painted another when the button is pressed. A pinned verdict
    /// comes back unchanged from that call, which is what leaves it alone.
    fn repaint_if_stale(&mut self, label: &str, track: &std::sync::Arc<EditableTrack>) {
        let key = (
            label.to_string(),
            std::sync::Arc::as_ptr(track) as usize,
            self.thresholds.clone(),
        );
        if self.painted_for.as_ref() == Some(&key) {
            return;
        }
        let mut with_bars = (**track).clone();
        with_bars.thresholds = self.thresholds.clone();
        let (painted, _) = apply_thresholds(&with_bars);
        self.painted = painted.observations.iter().map(|o| o.verdict).collect();
        self.painted_for = Some(key);
    }

    /// Ask again why the active track cannot be committed, when the track or the
    /// node's version has moved since the last time.
    ///
    /// The question is the core commit itself, asked of the very track the
    /// button would commit, so the button and the step cannot disagree about
    /// when it can run; asking it per frame would build a point record per
    /// frame, and the answer changes only when one of the two things it is a
    /// function of does.
    fn recheck_commit_if_stale(
        &mut self,
        label: &str,
        track: &std::sync::Arc<EditableTrack>,
        node: &SceneNode,
    ) {
        let key = (
            label.to_string(),
            std::sync::Arc::as_ptr(track) as usize,
            node.history.current_version().serial.as_u64(),
        );
        if self.commit_refusal_for.as_ref() == Some(&key) {
            return;
        }
        self.commit_refusal = sfmtool_core::bench::commit(node.edited(), track)
            .err()
            .map(|why| why.to_string());
        self.commit_refusal_for = Some(key);
    }

    /// Load one thumbnail texture into the cache.
    fn load_thumbnail(&mut self, ctx: &egui::Context, recon: &SfmrReconstruction, image: ImageRef) {
        let color_image = thumbnail_color_image(
            recon
                .image_table
                .thumbnails_y_x_rgb
                .index_axis(Axis(0), image.index()),
        );
        let texture = ctx.load_texture(
            format!("bench_thumb_{}", image.index()),
            color_image,
            egui::TextureOptions::LINEAR,
        );
        self.thumbnail_textures.insert(image, texture);
    }
}

/// The button the Point Track Detail panel names as the way onto the bench, and
/// the toolbar's own entry, quoted from one constant so the two agree.
pub(crate) const PUT_ON_BENCH_LABEL: &str = "Put selected point on bench";

/// The header: what the active track is, and what the last evaluation of it
/// made of it.
fn show_header(ui: &mut egui::Ui, label: &str, track: &EditableTrack) {
    let (kept, candidates, out) = track.verdict_counts();
    ui.horizontal_wrapped(|ui| {
        ui.label(egui::RichText::new(label).strong());
        ui.weak(format!("· the {} stage", track.stage_kind()));
        ui.weak(match track.origin {
            Some(origin) => format!("· from point {}", origin.point),
            None => "· new".to_string(),
        });
        ui.label(format!("{kept} in · {candidates} candidates · {out} out"));
    });
    match &track.stage {
        sfmtool_core::bench::Stage::Cluster(payload) => {
            ui.weak(format!(
                "Reference: observation {}{}",
                payload.reference,
                match &payload.template {
                    Some(template) => format!(", template {}", template.samples.shape()[0]),
                    None => ", no template cut yet".to_string(),
                }
            ));
        }
        sfmtool_core::bench::Stage::Track(payload) => {
            ui.weak(match payload.position {
                Some(position) => format!(
                    "Position ({:.3}, {:.3}, {:.3}){}",
                    position.x,
                    position.y,
                    position.z,
                    match payload.condition_number {
                        Some(condition) => format!(", condition {condition:.1}"),
                        None => String::new(),
                    }
                ),
                None => "No position: nothing has triangulated this track yet".to_string(),
            });
        }
    }
}

/// The panel with nothing on the bench: the ways in, each naming its gesture.
fn show_empty_state(
    ui: &mut egui::Ui,
    state: &AppState,
    node: &SceneNode,
    response: &mut TrackEditResponse,
) {
    ui.centered_and_justified(|ui| {
        ui.vertical_centered(|ui| {
            ui.label("No track on the bench");
            ui.add_space(8.0);
            let id = node.id;
            let point_refusal = state.busy_refusal(id).or_else(|| {
                (state.selected_point.filter(|p| p.recon == id).is_none())
                    .then(|| "No point is selected.".to_string())
            });
            if entry(
                ui,
                PUT_ON_BENCH_LABEL,
                point_refusal,
                "Put the selected point's track on the bench and work on it",
            ) {
                response.put_selected_point_on_bench = true;
            }
            if entry(
                ui,
                "Start cluster here",
                state.busy_refusal(id).or_else(|| pixel_refusal(state, id)),
                "Start a track at the pixel last named in Image Detail",
            ) {
                response.start_cluster = true;
            }
            ui.add_space(4.0);
            ui.weak(
                "To name a pixel: right-click it in the Image Detail panel, then press \
                 Start cluster here.",
            );
        });
    });
}

/// Why the two pixel entries cannot run, or `None`.
///
/// The pixel is the one the Image Detail panel's context menu was last opened
/// at -- the same value the Create 3D Point prompt opens on -- so naming a
/// pixel is the gesture that panel already has and this one borrows.
fn pixel_refusal(state: &AppState, id: ReconId) -> Option<String> {
    if state.selected_image.filter(|i| i.recon == id).is_none() {
        return Some("No image of this reconstruction is selected.".to_string());
    }
    if state.pending_observation_pixel.is_none() {
        return Some(
            "No pixel is named yet: right-click one in the Image Detail panel first.".to_string(),
        );
    }
    None
}

/// Why *Split off selected rows* cannot run, or `None`.
fn split_refusal(panel: &TrackEdit, track: &EditableTrack) -> Option<String> {
    match panel.selected_rows.len() {
        0 => Some("No rows are selected: click a row, or Ctrl-click several.".to_string()),
        n if n == track.observations.len() => {
            Some("Every row is selected, which would leave one track rather than two.".to_string())
        }
        _ => None,
    }
}

/// One toolbar entry: enabled, or greyed with the sentence saying what is
/// missing, in the style of the Image Detail menu entries.
fn entry(ui: &mut egui::Ui, text: &str, refusal: Option<String>, hint: &str) -> bool {
    let button = egui::Button::new(text);
    match refusal {
        None => ui.add(button).on_hover_text(hint).clicked(),
        Some(why) => {
            ui.add_enabled(false, button).on_disabled_hover_text(why);
            false
        }
    }
}

/// The word a provenance shows in the *From* column.
fn provenance_text(provenance: Provenance) -> String {
    match provenance {
        Provenance::Origin => "origin".to_string(),
        Provenance::Descriptor { feature } => format!("feature {feature}"),
        Provenance::Sweep => "sweep".to_string(),
        Provenance::Pixel => "pixel".to_string(),
        Provenance::Point { point } => format!("point {point}"),
    }
}

/// The measurements one observation shows at `stage`, as the table prints them:
/// ZNCC, shift, localizability, reprojection error, ray angle, status.
fn measurements(observation: &Observation, stage: StageKind) -> [String; 6] {
    let number = |value: Option<f64>, digits: usize| match value {
        Some(v) if v.is_finite() => format!("{v:.digits$}"),
        Some(_) => "NaN".to_string(),
        None => "-".to_string(),
    };
    match stage {
        StageKind::Cluster => {
            let m = observation.cluster.as_ref();
            [
                number(m.and_then(|m| m.zncc), 3),
                number(m.and_then(|m| m.shift_px), 2),
                number(m.and_then(|m| m.localizability), 3),
                "-".to_string(),
                "-".to_string(),
                m.and_then(|m| m.status)
                    .map_or_else(|| "not evaluated".to_string(), |s| format!("{s:?}")),
            ]
        }
        StageKind::Track => {
            let m = observation.track.as_ref();
            [
                number(m.and_then(|m| m.zncc), 3),
                number(m.and_then(|m| m.shift_px), 2),
                number(m.and_then(|m| m.localizability), 3),
                number(m.and_then(|m| m.reprojection_error), 2),
                number(m.and_then(|m| m.ray_angle_deg), 2),
                match m.and_then(|m| m.zncc) {
                    Some(_) => "localized".to_string(),
                    None => "not evaluated".to_string(),
                },
            ]
        }
    }
}
