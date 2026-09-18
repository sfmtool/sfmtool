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
//! tiles it has rendered, and the painting the sliders produce -- the last two
//! cached against the track's own `Arc` rather than recomputed per frame,
//! because the painting is `apply_thresholds` run over a copy (and a copy of a
//! track carries its consensus bitmap) and a tile is a warp of a
//! full-resolution photograph.

use std::collections::HashMap;

use sfmtool_core::bench::{
    apply_thresholds, Bench, EditableTrack, Observation, Provenance, StageKind, Thresholds, Verdict,
};
use sfmtool_core::SfmrReconstruction;

use crate::scene::{ImageRef, ReconId, SceneNode};
use crate::state::AppState;

mod table;
mod tile;

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
    /// *Evaluate*, carrying the search radius the control stands at: the
    /// reading moves nothing, so the one thing it needs from the panel is how
    /// far around each observation to look.
    pub evaluate: Option<f64>,
    /// *Fit*, carrying the same search radius for the reading it ends with.
    pub fit: Option<f64>,
    /// The *Stage* toggle, carrying the stage it asks for.
    pub set_stage: Option<StageKind>,
    /// *Apply thresholds*, carrying the bars the sliders stand at.
    pub apply_thresholds: Option<Thresholds>,
    /// *Split off selected rows*, carrying the rows.
    pub split: Option<Vec<usize>>,
    /// *Duplicate*: put a copy of the active track on the bench beside it.
    pub duplicate: bool,
    /// *Commit*.
    pub commit: bool,
    /// The Descriptor index row's *Open...*, which asks the dock for a file
    /// chooser. The panel names no path: a chooser is not an egui widget, so it
    /// lives where the other file questions do.
    pub open_descriptor_index: bool,
    /// The Descriptor index row's *Build*.
    pub build_descriptor_index: bool,
    /// A row's *Search for matching features*, carrying the observation it was
    /// opened on.
    pub search_descriptors: Option<usize>,
    /// A verdict control was clicked: the observation, and the verdict it
    /// cycled to.
    pub set_verdict: Option<(usize, Verdict)>,
    /// A row was clicked -- select this image, as the view-only panel does.
    pub select_image: Option<usize>,
    /// The clicked row's observation, in that image's own pixels: the place the
    /// Image Detail panel is asked to bring into view along with the image.
    /// `None` for an observation nothing has placed yet.
    pub reveal_feature: Option<[f32; 2]>,
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
    ///
    /// Seeded from the **active track's own bars**, not from the defaults: the
    /// track carries the bars it was last applied, and sliders that said
    /// something else would paint the table by a rule the track does not hold
    /// and hand that rule to the next press of the button.
    thresholds: Thresholds,
    /// The item and the track's own bars [`TrackEdit::thresholds`] was seeded
    /// from, so the seeding happens again when the active track changes or when
    /// a step moves that track's bars -- and not while the person is dragging a
    /// slider, which moves the panel's copy and leaves the track's alone.
    seeded_from: Option<(ReconId, String, Thresholds)>,
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
    /// The rendered tile of each observation, by observation index.
    ///
    /// Keyed by the row rather than by the image, because two observations can
    /// name one image and they are two pictures: at the cluster stage each has
    /// its own position and shape. A tile is a warp of a full-resolution
    /// photograph, so it is rendered once and kept; what says it is stale is
    /// [`TrackEdit::tiles_for`].
    tiles: HashMap<usize, Option<egui::TextureHandle>>,
    /// The item and the exact track value [`TrackEdit::tiles`] was rendered
    /// from: the label, and the address of the track's `Arc`. Any step on the
    /// track gives it a new `Arc`, and every step that moves a tile is one.
    tiles_for: Option<(String, usize)>,
    /// What the table drew last frame, in row order.
    ///
    /// Recorded unconditionally rather than under `cfg(test)`, so that what the
    /// tests read is the very table the app draws.
    rows: Vec<RowSummary>,
    /// Why the selected node's descriptor index cannot be built, or `None`
    /// when it can, as of the node it was last asked about.
    ///
    /// Cached rather than asked per frame: the question stats a `.sift` path
    /// per image of the capture, and what it is a function of -- which files
    /// sit beside the workspace -- is not something the viewer changes. The
    /// busy half of the refusal is asked every frame, in front of this, because
    /// that one is free and does move.
    build_refusal: Option<(ReconId, Option<String>)>,
    /// How far around each observation the next reading looks for its
    /// correlation peak, in patch-grid px. Panel state beside the sliders, and
    /// what *Evaluate* and *Fit* carry: it is an input to the measurement
    /// rather than a bar the painting judges by, so moving it repaints nothing
    /// and changes no number until the next reading runs.
    search_px: f64,
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
            seeded_from: None,
            painted: Vec::new(),
            painted_for: None,
            commit_refusal: None,
            commit_refusal_for: None,
            selected_rows: Vec::new(),
            selection_of: None,
            renaming: None,
            tiles: HashMap::new(),
            tiles_for: None,
            rows: Vec::new(),
            build_refusal: None,
            search_px: crate::bench::default_search_px(),
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

    /// Where the search control stands, in patch-grid px.
    #[cfg(test)]
    pub(crate) fn search_px(&self) -> f64 {
        self.search_px
    }

    /// Select one observation row from outside the panel.
    ///
    /// What the Image Detail panel's bench layer reports a click on a mark
    /// through: a mark there and a row here are one observation, so clicking
    /// either is the one gesture. It replaces the selection rather than
    /// extending it, which is what a plain click on a row does.
    pub(crate) fn select_row(&mut self, id: ReconId, label: &str, observation: usize) {
        self.selection_of = Some((id, label.to_string()));
        self.selected_rows = vec![observation];
    }

    /// Drop everything cached for a reconstruction that has left the scene.
    pub fn forget_recon(&mut self, id: ReconId) {
        self.tiles.clear();
        self.tiles_for = None;
        if self.selection_of.as_ref().is_some_and(|(of, _)| *of == id) {
            self.selected_rows.clear();
            self.selection_of = None;
        }
        self.painted_for = None;
        self.commit_refusal_for = None;
        self.build_refusal = None;
        self.seeded_from = None;
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
        self.reseat_thresholds(id, &label, track);
        self.repaint_if_stale(&label, track);
        self.recheck_commit_if_stale(&label, track, node);
        self.retile_if_stale(&label, track);

        show_header(ui, &label, track);
        self.show_toolbar(ui, state, node, &label, track, &mut response);
        self.show_thresholds(ui);
        self.show_descriptor_index(ui, state, id, &mut response);
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
        ui.horizontal_wrapped(|ui| {
            if entry(
                ui,
                "Evaluate",
                busy.clone(),
                "Measure every observation where it sits, moving nothing: no gate \
                 drops a row, and a row that cannot be read says why",
            ) {
                response.evaluate = Some(self.search_px);
            }
            let fit_refusal = busy.clone().or_else(|| {
                sfmtool_core::bench::fit_preconditions(track)
                    .err()
                    .map(|why| why.to_string())
            });
            if entry(
                ui,
                "Fit",
                fit_refusal,
                "Localize every sighting against the surfel, re-triangulate the \
                 in ones and re-fuse: this one moves the track",
            ) {
                response.fit = Some(self.search_px);
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
            if entry(
                ui,
                "Duplicate",
                busy.clone(),
                "Put a copy of this track on the bench and work on that: a patch \
                 already fitted to one piece of surface is most of the way to the \
                 piece beside it",
            ) {
                response.duplicate = true;
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
            // The fourth bar of `Thresholds`, which view selection scores a
            // candidate by as a fraction of the track's own self-agreement: a
            // bar with no slider is a bar only the wire can move.
            ui.add(
                egui::Slider::new(&mut self.thresholds.min_relative_zncc, 0.0..=1.0)
                    .text("min relative ZNCC")
                    .max_decimals(2),
            );
            // Not a threshold: this one is an input to the next reading rather
            // than a bar the painting judges by, which is why it stands apart
            // and why moving it repaints nothing.
            ui.separator();
            ui.add(
                egui::Slider::new(&mut self.search_px, 1.0..=24.0)
                    .text("search px")
                    .max_decimals(1),
            )
            .on_hover_text(
                "How far around each observation Evaluate looks for the correlation \
                 peak, in patch-grid px",
            );
        });
    }

    /// Put the sliders where the active track's own bars are, when the track
    /// they were seeded from is no longer the one being shown or its bars have
    /// moved under them.
    ///
    /// A slider drag moves [`TrackEdit::thresholds`] and not the track's, so
    /// the key below is unchanged and the drag survives; a step that sets the
    /// track's bars -- *Apply thresholds* here, `apply_bench_track_thresholds`
    /// over the wire, an undo of either -- moves them, and the sliders follow.
    fn reseat_thresholds(&mut self, id: ReconId, label: &str, track: &EditableTrack) {
        let key = (id, label.to_string(), track.thresholds.clone());
        if self.seeded_from.as_ref() == Some(&key) {
            return;
        }
        self.thresholds = track.thresholds.clone();
        self.seeded_from = Some(key);
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

    /// The tile one row draws, rendering it if this is the first frame that has
    /// asked for it since the track moved.
    ///
    /// `None` is a real answer and is cached as one: an observation with no
    /// surfel behind it yet, or in a photograph the node's cache has not
    /// decoded, has no tile, and re-attempting the warp every frame would be
    /// the cost the cache exists to avoid.
    fn ensure_tile(
        &mut self,
        ctx: &egui::Context,
        recon: &SfmrReconstruction,
        track: &EditableTrack,
        observation: usize,
        state: &AppState,
    ) -> Option<egui::TextureId> {
        if let Some(cached) = self.tiles.get(&observation) {
            return cached.as_ref().map(|texture| texture.id());
        }
        let id = self.selection_of.as_ref().map(|(id, _)| *id)?;
        let image = ImageRef::new(id, track.observations.get(observation)?.image as usize);
        let tile = state
            .full_res_cache
            .get(&image)
            .and_then(|slot| slot.as_ref())
            .and_then(|src| {
                tile::render(
                    ctx,
                    recon,
                    track,
                    observation,
                    src,
                    format!("bench_tile_{}_{observation}", image.index()),
                )
            });
        let texture_id = tile.as_ref().map(|texture| texture.id());
        self.tiles.insert(observation, tile);
        texture_id
    }

    /// Drop the rendered tiles when the track they were rendered from has
    /// moved, so a row never shows a picture of a position the observation has
    /// left.
    fn retile_if_stale(&mut self, label: &str, track: &std::sync::Arc<EditableTrack>) {
        let key = (label.to_string(), std::sync::Arc::as_ptr(track) as usize);
        if self.tiles_for.as_ref() == Some(&key) {
            return;
        }
        self.tiles.clear();
        self.tiles_for = Some(key);
    }
}

/// The button the Point Track Detail panel names as the way onto the bench, and
/// the toolbar's own entry, quoted from one constant so the two agree.
pub(crate) const PUT_ON_BENCH_LABEL: &str = "Put selected point on bench";

/// The observation row's context-menu entry, in one constant, as the Image
/// Detail menu's entries are: the label is quoted in a refusal and read back by
/// a test, and three spellings of one entry would drift.
pub(crate) const SEARCH_DESCRIPTORS_LABEL: &str = "Search for matching features";

impl TrackEdit {
    /// The **Descriptor index** row: which `.kdf` a search would query, and the
    /// two ways to give it one.
    ///
    /// Above the table because it is about the node and not about a row: every
    /// row's search goes through the same index, and a row that offered its own
    /// file chooser would suggest otherwise. The path is the whole of the
    /// state; the forest itself is the node's
    /// ([`crate::descriptor_index`]).
    fn show_descriptor_index(
        &mut self,
        ui: &mut egui::Ui,
        state: &AppState,
        id: ReconId,
        response: &mut TrackEditResponse,
    ) {
        if self.build_refusal.as_ref().map(|(of, _)| *of) != Some(id) {
            self.build_refusal = Some((id, state.descriptor_sources_refusal(id)));
        }
        let sources = self.build_refusal.as_ref().and_then(|(_, why)| why.clone());
        let busy = state.busy_refusal(id);
        ui.horizontal_wrapped(|ui| {
            ui.label("Descriptor index");
            match state.descriptor_index(id) {
                Some(index) => {
                    let path = index.path.display().to_string();
                    let shown = crate::elide::middle(&path, INDEX_PATH_WIDTH, |value| {
                        ui.ctx().fonts_mut(|fonts| {
                            fonts
                                .layout_no_wrap(
                                    value.to_owned(),
                                    egui::TextStyle::Body.resolve(ui.style()),
                                    ui.visuals().weak_text_color(),
                                )
                                .rect
                                .width()
                        })
                    });
                    ui.weak(shown).on_hover_text(format!(
                        "{path}
{} descriptors",
                        index.feature_count()
                    ));
                }
                None => {
                    ui.weak("none");
                }
            }
            if entry(
                ui,
                "Open...",
                busy.clone(),
                "Search a .kdf of your own choosing",
            ) {
                response.open_descriptor_index = true;
            }
            if entry(
                ui,
                "Build",
                busy.or(sources),
                "Index every .sift file of this reconstruction and write it beside them",
            ) {
                response.build_descriptor_index = true;
            }
        });
    }
}

/// How wide the Descriptor index row elides the path it names to.
const INDEX_PATH_WIDTH: f32 = 280.0;

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
            ui.add_space(4.0);
            ui.weak(format!(
                "Or start one from a pixel: right-click it in the Image Detail panel and \
                 choose \"{}\".",
                crate::image_detail::START_CLUSTER_LABEL,
            ));
        });
    });
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
        Provenance::Search { inliers } => format!("search ({inliers})"),
        Provenance::Sweep => "sweep".to_string(),
        Provenance::Pixel => "pixel".to_string(),
        Provenance::Point { point } => format!("point {point}"),
    }
}

/// The measurements one observation shows at `stage`, as the table prints them:
/// ZNCC, seed shift, projection offset, localizability, reprojection error, ray
/// angle, status.
///
/// The two distances are two questions and get two columns. **Seed shift** is
/// how far the correlation peak sits from the observation itself -- the
/// sighting's own evidence, and what the `max shift px` bar paints on. **Proj.
/// offset** is how far the observation sits from the point's projection, which
/// is a statement about the *point*: a mis-triangulated track shows a column of
/// large offsets beside a column of zero shifts, and one number could not say
/// that. The cluster stage has one of them -- the drift from its seed -- and
/// prints `-` for the other.
fn measurements(observation: &Observation, stage: StageKind) -> [String; 7] {
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
                "-".to_string(),
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
                number(m.and_then(|m| m.seed_shift_px), 2),
                number(m.and_then(|m| m.projection_offset_px), 2),
                number(m.and_then(|m| m.localizability), 3),
                number(m.and_then(|m| m.reprojection_error), 2),
                number(m.and_then(|m| m.ray_angle_deg), 2),
                // A row without a score says which of the reading's refusals it
                // was, in the evaluation's own sentence. An evaluation drops
                // nothing, so "no ZNCC" always has one of those answers behind
                // it, and a row that has never been read says that instead.
                match m {
                    Some(m) if m.zncc.is_some() => "localized".to_string(),
                    Some(m) => match m.reason {
                        Some(reason) => reason.to_string(),
                        None => "not evaluated".to_string(),
                    },
                    None => "not evaluated".to_string(),
                },
            ]
        }
    }
}
