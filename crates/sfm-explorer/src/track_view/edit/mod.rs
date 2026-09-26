// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Track View's edit mode: the bench's active track, and the steps that act on
//! it.
//!
//! See `specs/gui/track-view.md`. This body is drawn while Track View's *Edit*
//! box is ticked, which is while the selected node's bench ([`crate::bench`])
//! has an active track, and every gesture in it names that track. It shows the
//! active item and nothing else on the bench: the bench as a list is the Scene
//! tree's. The table carries view mode's columns first, so a reader who knows
//! one reads the other.
//!
//! The body decides nothing. Each gesture lands in [`TrackEditResponse`] and
//! the dock applies it through the `AppState` method that pushes the version,
//! for the reason every other panel's response works that way: the panel holds
//! `&AppState` while it draws, and a step needs it mutably.
//!
//! Almost no state lives here. The bench is the node's, at its cursor, so what
//! the panel owns is the slider positions during a drag, the *Lock* box Image Detail reads a
//! dot drag by, the row selection a split reads, the tiles it has rendered,
//! and the painting the sliders produce. The last two are cached against the
//! track's own `Arc` rather than recomputed per frame, because the painting is
//! `apply_thresholds` run over a copy (and a copy of a track carries its
//! consensus bitmap) and a tile is a warp of a full-resolution photograph.

use std::collections::HashMap;

use sfmtool_core::bench::{
    apply_thresholds, EditableTrack, Observation, Provenance, StageKind, Thresholds, Verdict,
};
use sfmtool_core::SfmrReconstruction;

use crate::bench::live::Evaluation;
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
/// entries that push a version are buttons and slider releases, and each is
/// made once.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct TrackEditResponse {
    /// The toolbar's *Discard*.
    pub discard: Option<String>,
    /// *Rename* was committed: the item, and the label it should take.
    pub rename: Option<(String, String)>,
    /// *Fit*, which runs at the radius the live evaluation reads at.
    pub fit: bool,
    /// The *search px* slider was released, or a value typed into it was
    /// committed: the radius the bench's evaluation should read at. Set only
    /// when it differs from the viewer's own.
    pub search_px: Option<f64>,
    /// The *Stage* toggle, carrying the stage it asks for.
    pub set_stage: Option<StageKind>,
    /// A threshold slider was released, or a value typed into one was
    /// committed: the bars the four sliders stand at, for the active track.
    /// Set only when they differ from the track's own.
    pub apply_thresholds: Option<Thresholds>,
    /// A kept-at-seed row's *Accept walk*, carrying the observation: put its
    /// sighting where the last fit's walk would have taken it.
    pub accept_walk: Option<usize>,
    /// *Split off selected rows*, carrying the rows.
    pub split: Option<Vec<usize>>,
    /// *Duplicate*: put a copy of the active track on the bench beside it.
    pub duplicate: bool,
    /// *Commit*.
    pub commit: bool,
    /// A row's *Build Index Files* / *Rebuild Index Files*: the entry a row
    /// offers where the search would be when the node's index is absent or out
    /// of date.
    pub build_index_files: bool,
    /// A row's *Find matches by SIFT query*, carrying the observation it was
    /// opened on.
    pub search_descriptors: Option<usize>,
    /// A track-stage row's *Find matches by geometry*, carrying the
    /// observation whose appearance is the explicit reference.
    pub search_geometry: Option<usize>,
    /// A verdict control was clicked: the observation, and the verdict it
    /// cycled to.
    pub set_verdict: Option<(usize, Verdict)>,
    /// A row was clicked -- select this image, as a view-mode row does.
    pub select_image: Option<usize>,
    /// A row was double-clicked -- enter camera view for this image, as a
    /// view-mode row's double-click does.
    pub request_camera_view: Option<usize>,
    /// The clicked row's observation, in that image's own pixels: the place the
    /// Image Detail panel is asked to bring into view along with the image.
    /// `None` for an observation nothing has placed yet.
    pub reveal_feature: Option<[f32; 2]>,
    /// The image under the pointer, for cross-panel hover.
    pub hovered_image: Option<usize>,
    /// Whether the pointer is inside the panel.
    pub has_pointer: bool,
}

/// Track View's edit-mode state.
pub struct TrackEdit {
    /// Where the threshold sliders stand.
    ///
    /// The **active track's own bars**, copied from it on every frame no slider
    /// is being dragged, so an undo, a redo, a step over the wire or a change
    /// of active item moves the sliders with it. Only during a drag does this
    /// hold a value the track does not: the drag repaints the table live, and
    /// its release applies the bars to the track as one version. The panel
    /// therefore never holds bars a *Fit* would not use.
    thresholds: Thresholds,
    /// Whether a threshold slider was being dragged on the last frame, which is
    /// what keeps [`TrackEdit::thresholds`] from being reset to the track's bars
    /// in the middle of the drag.
    sliding: bool,
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
    /// Why the selected node's SIFT index cannot be built, or `None` when it
    /// can, as of the node it was last asked about.
    ///
    /// Cached rather than asked per frame: the question stats `.sift` paths
    /// beside the workspace, and what it is a function of -- which of those
    /// files exist -- is not something the viewer changes. The busy half of the
    /// refusal is asked every frame, in front of this, because that one is free
    /// and does move.
    build_refusal: Option<(ReconId, Option<String>)>,
    /// Where the *search px* slider stands, in patch-grid px.
    ///
    /// The viewer's own radius ([`AppState::bench_search_px`]), copied on
    /// every frame the slider is not being dragged, as the threshold sliders
    /// copy the track's bars. A release hands the new radius to the dock, and
    /// every track is then evaluated again at it.
    search_px: f64,
    /// Whether the *search px* slider was being dragged on the last frame.
    searching: bool,
    /// Where the active track's evaluation stood when this frame drew it:
    /// what the status at the head of the toolbar says, and how the rows print
    /// their numbers.
    evaluation: Evaluation,
    /// Whether a dot drag in Image Detail at the track stage moves the patch,
    /// every sighting following (`true`, the default), or that one sighting's
    /// keypoint alone (`false`). The *Lock* checkbox.
    ///
    /// A tool setting and not bench state: it says what the next gesture will
    /// mean rather than anything about the track, so toggling it is no step,
    /// pushes no version and is not undone by Undo. It lives here for the
    /// session and is not saved with the layout, because a lock left off by
    /// the last session would turn the next one's first drag into an edit of
    /// one sighting nobody asked for.
    lock: bool,
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
            sliding: false,
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
            searching: false,
            evaluation: Evaluation::Evaluating,
            lock: true,
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

    /// Where the active track's evaluation stood when the panel last drew it.
    #[cfg(test)]
    pub(crate) fn evaluation(&self) -> &Evaluation {
        &self.evaluation
    }

    /// Whether the *Lock* box is ticked: a track-stage dot drag in Image Detail
    /// moves the patch when it is, and one sighting's keypoint when it is not.
    ///
    /// The box's own state, whatever stage the active track is in. At the
    /// cluster stage the box is greyed and every handle is already one
    /// sighting's, so what it holds there is only what the next track stage
    /// will be edited with.
    pub(crate) fn lock(&self) -> bool {
        self.lock
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

    /// The one observation row selected on `label`'s track of `id`, when
    /// exactly one is.
    ///
    /// What the 3D viewer's bench layer draws larger, so a row picked here can
    /// be found out in the world and a mark picked there can be seen to be this
    /// row. A multi-row selection names no single mark, so it names none.
    pub(crate) fn selected_row(&self, id: ReconId, label: &str) -> Option<usize> {
        let (of, item) = self.selection_of.as_ref()?;
        if *of != id || item != label {
            return None;
        }
        match self.selected_rows.as_slice() {
            [one] => Some(*one),
            _ => None,
        }
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
        self.sliding = false;
        self.painted.clear();
        self.rows.clear();
    }

    /// Draw the active track and report what the user did with it.
    ///
    /// Track View calls this only while a track is active; with none, or with
    /// no node selected, it draws nothing and forgets the rows it drew.
    pub fn show(&mut self, ui: &mut egui::Ui, state: &AppState) -> TrackEditResponse {
        let mut response = TrackEditResponse::default();
        let panel_rect = ui.available_rect_before_wrap();
        if let Some(pos) = ui.input(|i| i.pointer.hover_pos()) {
            response.has_pointer = panel_rect.contains(pos);
        }

        let Some(node) = crate::scene::selected_node(&state.scene, state.selected_recon) else {
            self.rows.clear();
            return response;
        };
        let id = node.id;
        let bench = node.history.current_bench();

        let active = crate::bench::active_track_label(bench).map(str::to_string);
        let Some(label) = active else {
            self.rows.clear();
            self.selected_rows.clear();
            self.selection_of = None;
            return response;
        };
        let track = bench.track(&label).expect("the active label names a track");

        // A change of item takes the row selection with it: the rows are
        // observation indexes, and another track's observations are not these.
        if self.selection_of.as_ref() != Some(&(id, label.clone())) {
            self.selected_rows.clear();
            self.selection_of = Some((id, label.clone()));
        }
        self.reseat_thresholds(track);
        if !self.searching {
            self.search_px = state.bench_search_px();
        }
        self.evaluation = state
            .bench_evaluation(id, &label)
            .unwrap_or(Evaluation::Evaluating);
        self.repaint_if_stale(&label, track);
        self.recheck_commit_if_stale(&label, track, node);
        self.retile_if_stale(&label, track);

        // Why an index cannot be built beside this node, cached against the
        // node: every row's menu asks it, and the question stats files.
        if self.build_refusal.as_ref().map(|(of, _)| *of) != Some(id) {
            self.build_refusal = Some((id, state.sift_sources_refusal(id)));
        }

        show_header(ui, &label, track);
        self.show_toolbar(ui, state, node, &label, track, &mut response);
        // A release applies what the drag left the sliders at. Painted by this
        // frame's value from the next frame on, which is the frame the dock
        // has applied it by.
        response.apply_thresholds = self.show_thresholds(ui, state.busy_refusal(id), track);
        response.search_px = self.show_search_px(ui, state.bench_search_px());
        ui.separator();
        self.show_table(ui, node.recon(), id, state, track, &mut response);
        response
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
            let (next, stage_label) = match track.stage_kind() {
                StageKind::Cluster => (StageKind::Track, "Stage: cluster \u{2192} track"),
                StageKind::Track => (StageKind::Cluster, "Stage: track \u{2192} cluster"),
            };
            let refusals = photometric_refusals(busy.as_deref(), track, next);
            show_evaluation(ui, &self.evaluation);
            if entry(
                ui,
                "Fit",
                refusals.fit,
                "Localize every sighting against the patch, re-triangulate the \
                 in ones and re-fuse: this one moves the track",
            ) {
                response.fit = true;
            }
            if entry(
                ui,
                stage_label,
                refusals.stage,
                "Move the track between its two representations",
            ) {
                response.set_stage = Some(next);
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
                 piece beside it (Ctrl+D)",
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
            self.show_lock(ui, track);
            self.show_rename(ui, label, busy, response);
        });
    }

    /// The *Lock* box: whether dragging a sighting's dot in Image Detail moves
    /// the patch or that sighting alone.
    ///
    /// Never greyed by a busy node, because toggling it is no step: it changes
    /// what the next drag will mean, and a drag on a busy node is refused on
    /// its own. Greyed at the cluster stage instead, where there is nothing
    /// for it to decide: a cluster has no shared geometry, so every handle is
    /// already one sighting's own.
    fn show_lock(&mut self, ui: &mut egui::Ui, track: &EditableTrack) {
        let at_cluster = track.stage_kind() == StageKind::Cluster;
        let checkbox = ui.add_enabled(!at_cluster, egui::Checkbox::new(&mut self.lock, LOCK_LABEL));
        if at_cluster {
            checkbox.on_disabled_hover_text(LOCK_AT_CLUSTER);
        } else if self.lock {
            checkbox.on_hover_text(
                "Locked: dragging a sighting's dot in Image Detail slides the patch, and                  every sighting follows it. Clear to move one keypoint on its own",
            );
        } else {
            checkbox.on_hover_text(
                "Unlocked: dragging a sighting's dot in Image Detail moves that keypoint                  alone, and the patch and every other sighting stay where they are. The                  outline's edges and corners take no drag while unlocked, a sighting                  having no size or turn of its own",
            );
        }
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

    /// The threshold sliders, which apply to the active track: a drag paints
    /// the table live, and its release (or a typed value's commit) hands back
    /// the bars to apply as one version. `None` on every other frame, and on a
    /// release that left the bars where the track has them.
    ///
    /// Greyed while the node is busy, with the busy sentence: a release there
    /// would be refused, and a slider that snapped back after a drag would say
    /// less than one that could not be dragged.
    fn show_thresholds(
        &mut self,
        ui: &mut egui::Ui,
        busy: Option<String>,
        track: &EditableTrack,
    ) -> Option<Thresholds> {
        let enabled = busy.is_none();
        let mut sliding = false;
        let mut released = false;
        ui.horizontal_wrapped(|ui| {
            ui.label("Thresholds");
            let bars = &mut self.thresholds;
            let sliders = [
                egui::Slider::new(&mut bars.min_zncc, 0.0..=1.0).text(MIN_ZNCC_LABEL),
                egui::Slider::new(&mut bars.max_shift_px, 0.0..=20.0).text(MAX_SHIFT_LABEL),
                egui::Slider::new(&mut bars.max_keypoint_uncertainty, 0.0..=2.0)
                    .text("max \u{3c3}_pos"),
                // The fourth bar of `Thresholds`, which view selection scores a
                // candidate by as a fraction of the track's own self-agreement:
                // a bar with no slider is a bar only the wire can move.
                egui::Slider::new(&mut bars.min_relative_zncc, 0.0..=1.0).text("min relative ZNCC"),
            ];
            for slider in sliders {
                // A typed value lands when the field is left, not per
                // keystroke, so typing "0.9" is one version and not three.
                let slider = slider.max_decimals(2).update_while_editing(false);
                let r = ui.add_enabled(enabled, slider);
                let r = match &busy {
                    Some(why) => r.on_disabled_hover_text(why),
                    None => r.on_hover_text(
                        "Applies to the track when released: one version, which Undo reverses",
                    ),
                };
                sliding |= r.dragged();
                // A drag ends in its release; a typed value or an arrow key
                // changes the value with no drag at all.
                released |= r.drag_stopped() || (r.changed() && !r.dragged());
            }
        });
        self.sliding = sliding;
        (released && !sliding && self.thresholds != track.thresholds)
            .then(|| self.thresholds.clone())
    }

    /// The *search px* slider: how far around each observation the evaluation
    /// looks for its correlation peak. Its release, or a typed value's commit,
    /// hands back the radius to set; `None` on every other frame, and on a
    /// release that left it at `current`.
    ///
    /// Not a threshold: it is an input to the evaluation rather than a bar the
    /// painting judges by, which is why it stands on a row of its own, and it
    /// applies to every track rather than to the active one. Never greyed by a
    /// busy node, because setting it is no step on the node: the evaluations
    /// it asks for wait until the node is free.
    fn show_search_px(&mut self, ui: &mut egui::Ui, current: f64) -> Option<f64> {
        let r = ui
            .add(
                egui::Slider::new(&mut self.search_px, 1.0..=24.0)
                    .text(SEARCH_PX_LABEL)
                    .max_decimals(1)
                    .update_while_editing(false),
            )
            .on_hover_text(
                "How far around each observation the evaluation looks for the correlation \
                 peak, in patch-grid px. Every track is evaluated again at it when it is \
                 released",
            );
        self.searching = r.dragged();
        let released = r.drag_stopped() || (r.changed() && !r.dragged());
        (released && !self.searching && self.search_px.to_bits() != current.to_bits())
            .then_some(self.search_px)
    }

    /// Put the sliders where the active track's own bars are, unless a slider
    /// is being dragged.
    ///
    /// Every frame, rather than when something is seen to change: the sliders
    /// show the track's bars and nothing else, so whatever moved them -- a
    /// slider's release here, `apply_bench_track_thresholds` over the wire, an
    /// undo or redo of either, another item made active -- the sliders follow.
    fn reseat_thresholds(&mut self, track: &EditableTrack) {
        if !self.sliding {
            self.thresholds = track.thresholds.clone();
        }
    }

    /// Recompute the painting when the track or the bars have moved.
    ///
    /// The painting **is** what applying the thresholds would do, computed by
    /// the same core function a slider's release applies, so a row can never be
    /// painted one way and turned another when the slider is let go. A pinned verdict
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
    /// patch behind it yet, or in a photograph the node's cache has not
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

/// The minimum-ZNCC slider's label, in one constant so the tests aim at the
/// label drawn.
pub(crate) const MIN_ZNCC_LABEL: &str = "min ZNCC";

/// The maximum-shift slider's label: the bar the painting judges a seed shift
/// by and the bound on how far a fit may move a sighting.
pub(crate) const MAX_SHIFT_LABEL: &str = "max shift px";

/// A kept-at-seed row's menu entry, which puts the sighting where the fit's
/// walk would have taken it.
pub(crate) const ACCEPT_WALK_LABEL: &str = "Accept walk";

/// The edit-mode checkbox that says whether Image Detail's dot drag moves the
/// patch or one sighting, in one constant so the tests aim at the label drawn.
pub(crate) const LOCK_LABEL: &str = "Lock";

/// Why *Lock* is greyed at the cluster stage.
pub(crate) const LOCK_AT_CLUSTER: &str = "A cluster has no shared patch: every sighting is                                           already moved on its own, locked or not.";

/// The observation row's context-menu entry, in one constant, as the Image
/// Detail menu's entries are: the label is quoted in a refusal and read back by
/// a test, and three spellings of one entry would drift.
pub(crate) const SEARCH_DESCRIPTORS_LABEL: &str = "Find matches by SIFT query";

/// The track-stage geometry search entry. It is separate from the SIFT label
/// because it reads poses and photographs, and requires no descriptor index.
pub(crate) const SEARCH_GEOMETRY_LABEL: &str = "Find matches by geometry";

/// The *search px* slider's label, in one constant so the tests aim at the
/// label drawn.
pub(crate) const SEARCH_PX_LABEL: &str = "search px";

/// What the toolbar says while an evaluation of the active track's current
/// inputs is running or waiting to start, and what each row's status cell says
/// then.
pub(crate) const EVALUATING_LABEL: &str = "Evaluating\u{2026}";

/// What the toolbar says once the numbers are the evaluation of the track as
/// it stands.
pub(crate) const EVALUATED_LABEL: &str = "Evaluated";

/// The status cell of a row whose track has no evaluation of its current
/// inputs and gets none until a step changes them.
pub(crate) const NOT_EVALUATED: &str = "not evaluated";

/// Where the active track's evaluation stands, at the head of the toolbar.
///
/// There is no *Evaluate* button: every change to an input of the evaluation
/// evaluates the track again (`specs/gui/bench.md` § "Live evaluation"), so
/// what the panel owes the person is which state the numbers below are in. A
/// track that cannot be evaluated, or whose evaluation failed, says why in the
/// sentence the refusal or the failure carries.
fn show_evaluation(ui: &mut egui::Ui, evaluation: &Evaluation) {
    match evaluation {
        Evaluation::Current => {
            ui.weak(EVALUATED_LABEL).on_hover_text(
                "The numbers below are the evaluation of the track as it stands. Every \
                 change to it is evaluated again as it is made",
            );
        }
        Evaluation::Evaluating => {
            ui.spinner();
            ui.weak(EVALUATING_LABEL).on_hover_text(
                "The track has changed since its numbers were measured, and it is being \
                 evaluated again. The greyed numbers below are the last evaluation's",
            );
        }
        Evaluation::Refused(why) | Evaluation::Failed(why) => {
            ui.colored_label(ui.visuals().warn_fg_color, why.as_str());
        }
    }
}

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
            // A bearing and a position are the same three numbers and different
            // statements, so the word in front of them is what tells a reader
            // which they are looking at. "at infinity" is view mode's own word
            // for the same row.
            //
            // The track's own flag and not its patch's `w`: a point put on the
            // bench from a node that stores no patch frames has no patch to read
            // a `w` off, and a bearing it came from is still a bearing.
            let at_infinity = payload.at_infinity;
            ui.weak(match payload.position {
                Some(position) => format!(
                    "{} ({:.3}, {:.3}, {:.3}){}{}",
                    if at_infinity { "Bearing" } else { "Position" },
                    position.x,
                    position.y,
                    position.z,
                    if at_infinity { ", at infinity" } else { "" },
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

/// What each of the two photometric entries is greyed with, or `None` where
/// the step can run.
pub(super) struct PhotometricRefusals {
    /// *Fit*.
    pub(super) fit: Option<String>,
    /// The *Stage* toggle, for the stage it would move to.
    pub(super) stage: Option<String>,
}

/// The sentence each of the two photometric entries is greyed with.
///
/// The busy refusal first, and then core's **own** half of that step's
/// validation -- the half that reads no photograph
/// ([`sfmtool_core::bench::fit_preconditions`],
/// [`sfmtool_core::bench::set_stage_preconditions`]). Asking them here is what
/// keeps a button that cannot work from starting a task whose only act would be
/// to decode a dozen photographs and fail for a reason the track already knew,
/// and it is why the button and the step cannot disagree about what is missing.
///
/// Split out of [`TrackEdit::show_toolbar`] so the rule can be read without a
/// frame to draw it in.
pub(super) fn photometric_refusals(
    busy: Option<&str>,
    track: &EditableTrack,
    next: StageKind,
) -> PhotometricRefusals {
    let refused = |why: Option<String>| busy.map(str::to_string).or(why);
    PhotometricRefusals {
        fit: refused(
            sfmtool_core::bench::fit_preconditions(track)
                .err()
                .map(|why| why.to_string()),
        ),
        stage: refused(
            sfmtool_core::bench::set_stage_preconditions(track, next)
                .err()
                .map(|why| why.to_string()),
        ),
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
///
/// The cells follow where the track's evaluation stands. Current, they are the
/// numbers the track carries. While an evaluation of new inputs is on its way
/// the numbers are the last evaluation's, which the table greys, and the status
/// cell says the row is being evaluated. A track that has no evaluation of its
/// inputs and will not get one prints no number at all.
fn measurements(
    observation: &Observation,
    stage: StageKind,
    evaluation: &Evaluation,
) -> [String; 7] {
    match evaluation {
        Evaluation::Current => measured(observation, stage),
        Evaluation::Evaluating => {
            let mut cells = measured(observation, stage);
            cells[6] = EVALUATING_LABEL.to_string();
            cells
        }
        Evaluation::Refused(_) | Evaluation::Failed(_) => {
            let mut cells: [String; 7] = Default::default();
            cells[..6].fill("-".to_string());
            cells[6] = NOT_EVALUATED.to_string();
            cells
        }
    }
}

/// The cells of [`measurements`] for the numbers the track carries.
fn measured(observation: &Observation, stage: StageKind) -> [String; 7] {
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
                //
                // The walk comes first among the answers a scored row can give:
                // it says the sighting did *not* move where the correlation
                // wanted it, which is the one thing about the row a person
                // reading "localized" would get wrong.
                // With the ZNCC the walk would have bought where the fit
                // scored one, beside the row's own ZNCC read at the seed: the
                // two numbers a person accepting the walk or not decides by.
                match m {
                    Some(m) if m.walked_px.is_some() => format!(
                        "walked {:.0} px{}, kept at seed",
                        m.walked_px.expect("just matched"),
                        match m.walked_zncc {
                            Some(z) if z.is_finite() => format!(" (ZNCC {z:.3} there)"),
                            _ => String::new(),
                        }
                    ),
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
