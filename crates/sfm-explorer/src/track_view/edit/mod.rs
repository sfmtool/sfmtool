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
//! the panel owns is the threshold boxes' values during a drag, the *Lock* box Image Detail reads a
//! dot drag by, the tiles it has rendered and their hover views, and the
//! judgement the boxes produce. The last two are cached against the
//! track's own `Arc` rather than recomputed per frame, because the judgement is
//! `verdicts_if_unpinned` run over a copy (and a copy of a track carries its
//! consensus bitmap) and a tile is a warp of a full-resolution photograph.
//! The row selection is not the panel's: it is the bench's, in
//! `AppState::bench_rows`, and a row click reports through
//! [`TrackEditResponse::pick_row`].

use std::collections::HashMap;

use sfmtool_core::bench::{
    bar_checks, verdicts_if_unpinned, BarChecks, EditableTrack, Observation, Provenance, StageKind,
    Thresholds, Verdict,
};
use sfmtool_core::SfmrReconstruction;

use crate::bench::live::Evaluation;
use crate::scene::{ImageRef, ReconId, SceneNode};
use crate::state::AppState;

mod surface_plot;
mod table;
mod tile;

#[cfg(test)]
mod tests;

pub(crate) use table::RowSummary;

/// What one frame of the panel asks the dock to do.
///
/// Every field is one gesture, and at most one of them is set on a frame: the
/// entries that push a version are buttons and box releases, and each is
/// made once.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct TrackEditResponse {
    /// The toolbar's *Discard*.
    pub discard: Option<String>,
    /// *Rename* was committed: the item, and the label it should take.
    pub rename: Option<(String, String)>,
    /// *Fit*.
    pub fit: bool,
    /// The *Stage* toggle, carrying the stage it asks for.
    pub set_stage: Option<StageKind>,
    /// A threshold box was released, or a value typed into one was
    /// committed: the bars the four boxes stand at, for the active track.
    /// Set only when they differ from the track's own.
    pub apply_thresholds: Option<Thresholds>,
    /// A kept-at-seed row's *Accept walk*, carrying the observation: put its
    /// sighting where the last fit's walk would have taken it.
    pub accept_walk: Option<usize>,
    /// *Split off selected rows*, carrying the rows.
    pub split: Option<Vec<usize>>,
    /// A row was clicked: the observation, and whether Ctrl or Shift was held
    /// to extend the selection rather than replace it. Applied through
    /// `AppState::pick_bench_observation`.
    pub pick_row: Option<(usize, bool)>,
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
    /// A row's *Keep* switch was clicked: the observation, and the verdict it
    /// switched to.
    pub set_verdict: Option<(usize, Verdict)>,
    /// Verdicts handed back to the thresholds in one step: a row's pin or
    /// its *Unpin, let the thresholds decide*, the row menu's unpin of a
    /// selection, or the *Keep* heading's pin, which names every pinned row.
    pub unpin_verdicts: Option<Vec<usize>>,
    /// The header's go-to button: open the *Go to Point* dialog.
    pub request_goto_point: bool,
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
    /// Where the threshold boxes stand.
    ///
    /// The **active track's own bars**, copied from it on every frame no box
    /// is being dragged, so an undo, a redo, a step over the wire or a change
    /// of active item moves the boxes with it. Only during a drag does this
    /// hold a value the track does not: the drag repaints the table live, and
    /// its release applies the bars to the track as one version. The panel
    /// therefore never holds bars a *Fit* would not use.
    thresholds: Thresholds,
    /// Whether a threshold box was being dragged on the last frame, which is
    /// what keeps [`TrackEdit::thresholds`] from being reset to the track's bars
    /// in the middle of the drag.
    sliding: bool,
    /// What the boxes say about each observation of the active track, which
    /// is what its readings and its *Keep* cell are coloured by: `None` for an
    /// observation nothing at the track's stage has measured, which the bars
    /// do not judge.
    judged: Vec<Option<Judgement>>,
    /// The item, the exact track value and the bars [`TrackEdit::judged`] was
    /// computed from: the label, the address of the track's `Arc`, and the
    /// boxes. A step on the track gives it a new `Arc`, which is what says the
    /// judgement is stale.
    judged_for: Option<(String, usize, Thresholds)>,
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
    /// The node and the item the panel last drew, which the row menus and the
    /// tiles are read against.
    showing: Option<(ReconId, String)>,
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
    /// The hover view of each row's tile, by observation index: rendered the
    /// first time the pointer rests on that tile, and kept and dropped with
    /// [`TrackEdit::tiles`], since it is the same picture made wider and goes
    /// stale exactly when the tile does. `None` is cached as the tile's is.
    contexts: HashMap<usize, Option<tile::DrawnContext>>,
    /// The self-similarity surface plot of each observation, by observation
    /// index, with the surface and tolerance it was drawn from. Checked
    /// against the row's own reading every frame and redrawn when that
    /// changes, since an evaluation replaces the readings without moving the
    /// tile.
    plots: HashMap<usize, (Vec<f64>, f64, Option<surface_plot::DrawnPlot>)>,
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
            judged: Vec::new(),
            judged_for: None,
            commit_refusal: None,
            commit_refusal_for: None,
            showing: None,
            renaming: None,
            tiles: HashMap::new(),
            tiles_for: None,
            contexts: HashMap::new(),
            plots: HashMap::new(),
            rows: Vec::new(),
            build_refusal: None,
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

    /// Where the boxes stand.
    #[cfg(test)]
    pub(crate) fn thresholds(&self) -> &Thresholds {
        &self.thresholds
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

    /// Drop everything cached for a reconstruction that has left the scene.
    pub fn forget_recon(&mut self, id: ReconId) {
        self.tiles.clear();
        self.tiles_for = None;
        self.contexts.clear();
        if self.showing.as_ref().is_some_and(|(of, _)| *of == id) {
            self.showing = None;
        }
        self.judged_for = None;
        self.commit_refusal_for = None;
        self.build_refusal = None;
        self.sliding = false;
        self.judged.clear();
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
            self.showing = None;
            return response;
        };
        let track = bench.track(&label).expect("the active label names a track");

        self.showing = Some((id, label.clone()));
        self.reseat_thresholds(track);
        self.evaluation = state
            .bench_evaluation(id, &label)
            .unwrap_or(Evaluation::Evaluating);
        self.rejudge_if_stale(&label, track);
        self.recheck_commit_if_stale(&label, track, node);
        self.retile_if_stale(&label, track);

        // Why an index cannot be built beside this node, cached against the
        // node: every row's menu asks it, and the question stats files.
        if self.build_refusal.as_ref().map(|(of, _)| *of) != Some(id) {
            self.build_refusal = Some((id, state.sift_sources_refusal(id)));
        }

        // The ID of the point the track was read from, where that point is
        // still in the version at the cursor: what the header's copy button
        // copies, and what the *Go to Point* dialog takes back.
        let point_id = state
            .resolved_origin(node, track)
            .map(|index| crate::scene::point_id(node, index as usize));
        response.request_goto_point = show_header(ui, &label, track, point_id.as_deref());
        self.show_toolbar(ui, state, node, &label, track, &mut response);
        // A release applies what the drag left the boxes at. Painted by this
        // frame's value from the next frame on, which is the frame the dock
        // has applied it by.
        response.apply_thresholds = self.show_thresholds(ui, state.busy_refusal(id), track);
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
            // The arrow is U+23F5, which egui's bundled fonts draw; they have no
            // glyph for U+2192, which draws as a box.
            let (next, stage_label) = match track.stage_kind() {
                StageKind::Cluster => (StageKind::Track, "Stage: cluster \u{23f5} track"),
                StageKind::Track => (StageKind::Cluster, "Stage: track \u{23f5} cluster"),
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
            let selected = state.selected_bench_observations(id, label);
            let split_refusal = busy.clone().or_else(|| split_refusal(selected, track));
            if entry(
                ui,
                &format!("Split off {} rows", selected.len()),
                split_refusal,
                "Move the selected rows onto a second track beside this one",
            ) {
                response.split = Some(selected.to_vec());
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

    /// The threshold boxes, which apply to the active track: a drag paints
    /// the table live, and its release (or a typed value's commit) hands back
    /// the bars to apply as one version. `None` on every other frame, and on a
    /// release that left the bars where the track has them.
    ///
    /// Greyed while the node is busy, with the busy sentence: a release there
    /// would be refused, and a box that snapped back after a drag would say
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
            // Each bar is a label and a box that is dragged left and right to
            // change it, or clicked to type into: a slider's rail beside the
            // box would say nothing the box does not. The ZNCC bars read in
            // percent, as the table's ZNCC column does; the track stores them
            // on the 0 to 1 scale.
            let boxes = [
                (
                    MIN_ZNCC_LABEL,
                    percent(egui::DragValue::new(&mut bars.min_zncc)),
                ),
                // The bar on the middle reading; 0 turns it off.
                (
                    MIN_ZNCC_MIDDLE_LABEL,
                    percent(egui::DragValue::new(&mut bars.min_zncc_middle)),
                ),
                // In patch-grid px, and also the radius the evaluation looks
                // for each peak within.
                (
                    MAX_SHIFT_LABEL,
                    egui::DragValue::new(&mut bars.max_shift_px)
                        .range(0.0..=24.0)
                        .speed(0.05)
                        .max_decimals(1),
                ),
                // In patch-grid px, the unit of the self-similarity column; at
                // the largest radius read it turns nothing out.
                (
                    MAX_SELF_SIMILARITY_LABEL,
                    egui::DragValue::new(&mut bars.max_zncc_self_similarity_radius)
                        .range(0.0..=max_self_similarity_radius())
                        .speed(0.02)
                        .max_decimals(1),
                ),
                // The fourth bar of `Thresholds`, which view selection scores a
                // candidate by as a fraction of the track's own self-agreement:
                // a bar with no box is a bar only the wire can move.
                (
                    MIN_RELATIVE_ZNCC_LABEL,
                    percent(egui::DragValue::new(&mut bars.min_relative_zncc)),
                ),
            ];
            for (label, value) in boxes {
                let named = ui.add_enabled(enabled, egui::Label::new(label));
                if label == MAX_SHIFT_LABEL {
                    named.on_hover_text(MAX_SHIFT_TIP);
                } else if label == MAX_SELF_SIMILARITY_LABEL {
                    named.on_hover_text(MAX_SELF_SIMILARITY_TIP);
                }
                // A typed value lands when the field is left, not per
                // keystroke, so typing "90" is one version and not two.
                let r = ui.add_enabled(enabled, value.update_while_editing(false));
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

    /// Put the boxes where the active track's own bars are, unless a box
    /// is being dragged.
    ///
    /// Every frame, rather than when something is seen to change: the boxes
    /// show the track's bars and nothing else, so whatever moved them -- a
    /// box's release here, `apply_bench_track_thresholds` over the wire, an
    /// undo or redo of either, another item made active -- the boxes follow.
    fn reseat_thresholds(&mut self, track: &EditableTrack) {
        if !self.sliding {
            self.thresholds = track.thresholds.clone();
        }
    }

    /// Recompute the judgement when the track or the bars have moved.
    ///
    /// Both halves are the core's own: the readings are judged by
    /// `bar_checks`, and the proposal is `verdicts_if_unpinned`, which gives an
    /// unpinned row what applying the bars would do -- the same core step a
    /// box's release applies, so a row can never be shown one way and turned
    /// another when the box is let go -- and a pinned row what unpinning it
    /// would. Both run with the boxes' bars, so they follow a drag live.
    fn rejudge_if_stale(&mut self, label: &str, track: &std::sync::Arc<EditableTrack>) {
        let key = (
            label.to_string(),
            std::sync::Arc::as_ptr(track) as usize,
            self.thresholds.clone(),
        );
        if self.judged_for.as_ref() == Some(&key) {
            return;
        }
        let mut with_bars = (**track).clone();
        with_bars.thresholds = self.thresholds.clone();
        let stage = track.stage_kind();
        self.judged = verdicts_if_unpinned(&with_bars)
            .into_iter()
            .zip(&with_bars.observations)
            .map(|(proposal, observation)| {
                Some(Judgement {
                    checks: bar_checks(observation, stage, &self.thresholds)?,
                    proposal: proposal?,
                })
            })
            .collect();
        self.judged_for = Some(key);
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
        let id = self.showing.as_ref().map(|(id, _)| *id)?;
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

    /// The hover view of one row's tile, rendering it if this is the first
    /// frame that has asked for it since the track moved.
    ///
    /// Asked for only while the pointer rests on the tile, so a table of many
    /// rows renders the wider picture for the rows a person looks at and no
    /// others.
    fn ensure_context(
        &mut self,
        ctx: &egui::Context,
        recon: &SfmrReconstruction,
        track: &EditableTrack,
        observation: usize,
        state: &AppState,
    ) -> Option<&tile::DrawnContext> {
        if !self.contexts.contains_key(&observation) {
            let id = self.showing.as_ref().map(|(id, _)| *id)?;
            let image = ImageRef::new(id, track.observations.get(observation)?.image as usize);
            let drawn = state
                .full_res_cache
                .get(&image)
                .and_then(|slot| slot.as_ref())
                .and_then(|src| tile::context(recon, track, observation, src))
                .map(|context| {
                    tile::DrawnContext::new(
                        ctx,
                        context,
                        format!("bench_tile_context_{}_{observation}", image.index()),
                    )
                });
            self.contexts.insert(observation, drawn);
        }
        self.contexts.get(&observation).and_then(Option::as_ref)
    }

    /// The self-similarity surface plot of `observation` for `surface` read
    /// at `tolerance`, drawing it if the row's reading has changed since it
    /// was last drawn, or `None` when the reading has nothing to draw.
    fn ensure_plot(
        &mut self,
        ctx: &egui::Context,
        observation: usize,
        surface: &[f64],
        tolerance: f64,
    ) -> Option<&surface_plot::DrawnPlot> {
        let stale = self
            .plots
            .get(&observation)
            .is_none_or(|(drawn_from, at, _)| {
                drawn_from.as_slice() != surface || at.to_bits() != tolerance.to_bits()
            });
        if stale {
            let drawn = surface_plot::SurfacePlot::new(surface, tolerance).map(|plot| {
                surface_plot::DrawnPlot::new(
                    ctx,
                    plot,
                    format!("bench_self_similarity_{observation}"),
                )
            });
            self.plots
                .insert(observation, (surface.to_vec(), tolerance, drawn));
        }
        self.plots
            .get(&observation)
            .and_then(|(_, _, drawn)| drawn.as_ref())
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
        self.contexts.clear();
        self.tiles_for = Some(key);
    }
}

/// A patch ZNCC as a table cell prints it, in percent: the whole-patch
/// reading over the middle one (`92% whole` over `61% mid`).
///
/// The middle ZNCC is the same samples read over the middle square of the
/// patch only, so the pair says whether an agreement is carried by the
/// pixel's own neighbourhood or by its surroundings. Percent carries the same
/// two digits as `0.92` in fewer characters, which keeps the column narrow.
/// `-` stands for a reading that is not there: no whole-patch ZNCC at all, or
/// no middle one beside it, as on a track read back from a committed point. A
/// reading that was taken and came out non-finite prints `NaN`.
pub(crate) fn zncc_text(whole: Option<f64>, middle: Option<f64>) -> String {
    stacked(whole, middle, |value| format!("{:.0}%", 100.0 * value))
}

/// [`zncc_text`] for a sentence, on one line (`92% / 61%`).
pub(crate) fn zncc_sentence(whole: Option<f64>, middle: Option<f64>) -> String {
    let number = |value: f64| finite_or_nan(value, |v| format!("{:.0}%", 100.0 * v));
    match whole {
        None => "-".to_string(),
        Some(whole) => format!(
            "{} / {}",
            number(whole),
            middle.map_or_else(|| "-".to_string(), number)
        ),
    }
}

/// A ZNCC self-similarity radius as a table cell prints it, in grid px: the
/// whole tile's over its middle square's (`0.4 px whole` over `3+ px mid`), to
/// one decimal, and `3+` for the largest radius the reading searches, which
/// stands for that far or further.
///
/// `-` stands for a reading that is not there, as [`zncc_text`] has it.
fn self_similarity_text(whole: Option<f64>, middle: Option<f64>) -> String {
    stacked(whole, middle, |value| {
        format!("{} px", radius_number(value))
    })
}

/// A whole-patch reading over its middle's, as a two-reading cell prints them:
/// each through `number`, with the name of the part it reads (`93% whole` over
/// `89% mid`).
///
/// A missing whole reading prints `-` alone, since there is nothing to name;
/// a missing middle beside a whole prints `- mid`, as on a track read back from
/// a committed point. A reading that was taken and came out non-finite prints
/// `NaN`.
fn stacked(whole: Option<f64>, middle: Option<f64>, number: impl Fn(f64) -> String) -> String {
    let Some(whole) = whole else {
        return "-".to_string();
    };
    let number = |value: f64| finite_or_nan(value, &number);
    format!(
        "{} whole\n{} mid",
        number(whole),
        middle.map_or_else(|| "-".to_string(), number)
    )
}

/// `number(value)` for a finite value, and `NaN` for one that is not.
fn finite_or_nan(value: f64, number: impl Fn(f64) -> String) -> String {
    if value.is_finite() {
        number(value)
    } else {
        "NaN".to_string()
    }
}

/// The largest radius the self-similarity reading searches, in grid px: the
/// value that reads "this far or further".
fn max_self_similarity_radius() -> f64 {
    f64::from(sfmtool_core::patch::self_similarity::SelfSimilarityParams::default().max_radius)
}

/// One self-similarity radius as the cell and the grid's hover print it: to
/// one decimal (`0.4`, `1.3`), and `3+` at the maximum, which reads "that far
/// or further".
fn radius_number(value: f64) -> String {
    if !value.is_finite() {
        return "NaN".to_string();
    }
    let max = max_self_similarity_radius();
    if value >= max {
        return format!("{max:.0}+");
    }
    format!("{value:.1}")
}

/// What the bars say about one measured observation: each reading's check,
/// and the verdict the bars propose for it were its verdict unpinned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Judgement {
    /// What each bar says about the reading it judges.
    pub checks: BarChecks,
    /// The verdict the bars propose, with one `in` per image kept.
    pub proposal: Verdict,
}

/// The two three-by-three grids a row draws: the ZNCC grid and the
/// self-similarity grid of the measurement its stage carries.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub(crate) struct RowGrids {
    /// The ZNCC of each cell, `grid[row][col]` from the top-left cell.
    pub zncc: Option<[[f64; 3]; 3]>,
    /// The ZNCC self-similarity radius of each cell, in grid px.
    pub radius: Option<[[f64; 3]; 3]>,
    /// Per cell, the direction the cell's indistinguishable shifts line up
    /// in, scaled by how strongly.
    pub radius_slide: Option<[[[f64; 2]; 3]; 3]>,
}

/// The whole core's self-similarity radius of `observation` at `stage`.
fn row_radius(observation: &Observation, stage: StageKind) -> Option<f64> {
    match stage {
        StageKind::Cluster => observation.cluster.as_ref()?.zncc_self_similarity_radius,
        StageKind::Track => observation.track.as_ref()?.zncc_self_similarity_radius,
    }
}

/// The whole core's self-similarity surface of `observation` at `stage` and
/// the tolerance it was judged by, or `None` when there is none, including
/// while the evaluation is refused or failed.
fn row_surface<'a>(
    observation: &'a Observation,
    stage: StageKind,
    evaluation: &Evaluation,
) -> Option<(&'a [f64], f64)> {
    if matches!(evaluation, Evaluation::Refused(_) | Evaluation::Failed(_)) {
        return None;
    }
    let (surface, tolerance) = match stage {
        StageKind::Cluster => observation.cluster.as_ref().map(|m| {
            (
                m.zncc_self_similarity_surface.as_deref(),
                m.zncc_self_similarity_tolerance,
            )
        })?,
        StageKind::Track => observation.track.as_ref().map(|m| {
            (
                m.zncc_self_similarity_surface.as_deref(),
                m.zncc_self_similarity_tolerance,
            )
        })?,
    };
    Some((surface?, tolerance?))
}

/// The grids of `observation` at `stage`, or none while the evaluation is
/// refused or failed, when the row's cells print `-` too.
fn row_grids(observation: &Observation, stage: StageKind, evaluation: &Evaluation) -> RowGrids {
    if matches!(evaluation, Evaluation::Refused(_) | Evaluation::Failed(_)) {
        return RowGrids::default();
    }
    match stage {
        StageKind::Cluster => observation
            .cluster
            .as_ref()
            .map_or_else(RowGrids::default, |m| RowGrids {
                zncc: m.zncc_grid,
                radius: m.zncc_self_similarity_radius_grid,
                radius_slide: m.zncc_self_similarity_slide_grid,
            }),
        StageKind::Track => observation
            .track
            .as_ref()
            .map_or_else(RowGrids::default, |m| RowGrids {
                zncc: m.zncc_grid,
                radius: m.zncc_self_similarity_radius_grid,
                radius_slide: m.zncc_self_similarity_slide_grid,
            }),
    }
}

/// The colour a ZNCC grid cell is drawn in: red at `0.5` and below, green at
/// `1`, through yellow between. `None` for a cell with no reading.
pub(crate) fn zncc_cell_color(zncc: f64) -> Option<egui::Color32> {
    zncc.is_finite()
        .then(|| red_to_green(((zncc - 0.5) / 0.5).clamp(0.0, 1.0)))
}

/// The colour a self-similarity grid cell is drawn in: green under 1, where
/// a match locks within a pixel; yellow from 1 to 2; orange from 2 to under
/// the largest radius searched; and red at it, which reads "that far or
/// further". `None` for a cell with no reading.
fn self_similarity_cell_color(radius: f64) -> Option<egui::Color32> {
    if !radius.is_finite() {
        return None;
    }
    let t = if radius >= max_self_similarity_radius() {
        0.0
    } else if radius >= 2.0 {
        0.25
    } else if radius >= 1.0 {
        0.5
    } else {
        1.0
    };
    Some(red_to_green(t))
}

/// Red at `0`, yellow at `0.5`, green at `1`.
fn red_to_green(t: f64) -> egui::Color32 {
    let t = t.clamp(0.0, 1.0);
    let red = (2.0 * (1.0 - t)).min(1.0);
    let green = (2.0 * t).min(1.0);
    egui::Color32::from_rgb((220.0 * red) as u8, (200.0 * green) as u8, 40)
}

/// The minimum-ZNCC box's label, in one constant so the tests aim at the
/// label drawn.
pub(crate) const MIN_ZNCC_LABEL: &str = "min ZNCC (%)";

/// The minimum-middle-ZNCC box's label.
pub(crate) const MIN_ZNCC_MIDDLE_LABEL: &str = "min middle ZNCC (%)";

/// The minimum-relative-ZNCC box's label.
pub(crate) const MIN_RELATIVE_ZNCC_LABEL: &str = "min relative ZNCC (%)";

/// A box over a `0 ..= 1` bar that shows and takes the value in percent, in
/// whole steps: `70` for a stored `0.7`, half a percent per point dragged. A
/// typed value may carry a trailing `%`.
fn percent(value: egui::DragValue<'_>) -> egui::DragValue<'_> {
    value
        .range(0.0..=1.0)
        .speed(0.005)
        .max_decimals(2)
        .custom_formatter(|value, _| format!("{:.0}", 100.0 * value))
        .custom_parser(parse_percent)
}

/// A typed percent as the `0 ..= 1` value it stands for, or `None` when it is
/// not a number.
fn parse_percent(text: &str) -> Option<f64> {
    let number = text.trim().trim_end_matches('%').trim();
    number.parse::<f64>().ok().map(|v| v / 100.0)
}

/// The shift box's label: the bar the painting judges a shift by, the radius
/// the evaluation looks for each peak within, and the bound on how far a fit
/// may move a sighting.
pub(crate) const MAX_SHIFT_LABEL: &str = "shift px";

/// The shift box's hover text.
const MAX_SHIFT_TIP: &str = "The largest shift a sighting may have, in patch-grid px: how \
    far the correlation peak may sit from where the sighting is. The evaluation looks for each \
    peak within this distance, a row whose peak is further is painted out, and a fit moves no \
    sighting further than this.";

/// The self-similarity box's label: the largest ZNCC self-similarity radius
/// an observation's tile may have.
pub(crate) const MAX_SELF_SIMILARITY_LABEL: &str = "self-sim. px";

/// The self-similarity box's hover text.
const MAX_SELF_SIMILARITY_TIP: &str = "The largest ZNCC self-similarity radius a sighting's \
    own tile may have, in patch-grid px: how far the tile can slide over itself and still \
    match. A row whose whole tile reads further than this is painted out, since a match \
    cannot pin its position, as along a straight edge or over a flat patch. At 3, the largest \
    radius read, it turns nothing out.";

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
/// made of it. Returns whether its go-to button was clicked.
///
/// `point_id` is the ID of the point the track was read from, when that point
/// is still in the version at the cursor. It carries the copy and the go-to
/// buttons view mode's header draws beside a point ID, so an ID copied here is
/// one the *Go to Point* dialog takes back. A track put on the bench from a
/// point is labelled with that ID unless it was renamed, and the ID is then
/// printed once, as the label.
fn show_header(
    ui: &mut egui::Ui,
    label: &str,
    track: &EditableTrack,
    point_id: Option<&str>,
) -> bool {
    use crate::track_view::header_buttons::{copy_button, goto_button};
    let (kept, out) = track.verdict_counts();
    let pinned = track.observations.iter().filter(|o| o.pinned).count();
    let mut goto_clicked = false;
    ui.horizontal_wrapped(|ui| {
        match point_id {
            Some(id) if id == label => {
                ui.label(egui::RichText::new(label).monospace().strong());
            }
            Some(id) => {
                ui.label(egui::RichText::new(label).strong());
                ui.weak("· point");
                ui.label(egui::RichText::new(id).monospace());
            }
            None => {
                ui.label(egui::RichText::new(label).strong());
            }
        }
        if let Some(id) = point_id {
            if copy_button(ui, "Copy Point ID") {
                ui.ctx().copy_text(id.to_string());
            }
            goto_clicked = goto_button(ui);
        }
        ui.weak(format!("· the {} stage", track.stage_kind()));
        match (track.origin, point_id) {
            (None, _) => {
                ui.weak("· new");
            }
            // The point is gone from this version: say which it was.
            (Some(origin), None) => {
                ui.weak(format!("· from point {}", origin.point));
            }
            (Some(_), Some(_)) => {}
        }
        ui.label(format!("{kept} kept · {out} out · {pinned} pinned"));
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
    goto_clicked
}

/// Why *Split off selected rows* cannot run, or `None`.
fn split_refusal(selected: &[usize], track: &EditableTrack) -> Option<String> {
    match selected.len() {
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
/// ZNCC, seed shift, the reprojection error over the ray angle, the ZNCC
/// self-similarity radius, status.
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
) -> [String; 5] {
    match evaluation {
        Evaluation::Current => measured(observation, stage),
        Evaluation::Evaluating => {
            let mut cells = measured(observation, stage);
            cells[4] = EVALUATING_LABEL.to_string();
            cells
        }
        Evaluation::Refused(_) | Evaluation::Failed(_) => {
            let mut cells: [String; 5] = Default::default();
            cells[..4].fill("-".to_string());
            cells[4] = NOT_EVALUATED.to_string();
            cells
        }
    }
}

/// A reprojection error as its table cell prints it: in pixels over the
/// same residual as an angle in degrees, each to two decimals (`0.65 px` over
/// `0.08°`). `-` stands for a reading that is not there, as [`zncc_text`] has
/// it.
fn projection_error_text(px: Option<f64>, deg: Option<f64>) -> String {
    let px = px.map_or_else(
        || "-".to_string(),
        |v| finite_or_nan(v, |v| format!("{v:.2} px")),
    );
    let deg = deg.map_or_else(
        || "-".to_string(),
        |v| finite_or_nan(v, |v| format!("{v:.2}\u{b0}")),
    );
    if px == "-" && deg == "-" {
        return "-".to_string();
    }
    format!("{px}\n{deg}")
}

/// The cells of [`measurements`] for the numbers the track carries.
fn measured(observation: &Observation, stage: StageKind) -> [String; 5] {
    let px = |value: Option<f64>| match value {
        Some(v) if v.is_finite() => format!("{v:.2} px"),
        Some(_) => "NaN".to_string(),
        None => "-".to_string(),
    };
    match stage {
        StageKind::Cluster => {
            let m = observation.cluster.as_ref();
            [
                zncc_text(m.and_then(|m| m.zncc), m.and_then(|m| m.zncc_middle)),
                px(m.and_then(|m| m.shift_px)),
                "-".to_string(),
                self_similarity_text(
                    m.and_then(|m| m.zncc_self_similarity_radius),
                    m.and_then(|m| m.zncc_self_similarity_radius_middle),
                ),
                m.and_then(|m| m.status)
                    .map_or_else(|| "not evaluated".to_string(), |s| format!("{s:?}")),
            ]
        }
        StageKind::Track => {
            let m = observation.track.as_ref();
            [
                zncc_text(m.and_then(|m| m.zncc), m.and_then(|m| m.zncc_middle)),
                px(m.and_then(|m| m.seed_shift_px)),
                // One column for the reprojection error: in px to the
                // triangulated point, or before there is one to the patch's
                // centre, which is kept on the point once it exists, so the
                // two are one number wherever both are measured; then the same
                // residual in degrees.
                projection_error_text(
                    m.and_then(|m| m.reprojection_error.or(m.projection_offset_px)),
                    m.and_then(|m| m.ray_angle_deg),
                ),
                self_similarity_text(
                    m.and_then(|m| m.zncc_self_similarity_radius),
                    m.and_then(|m| m.zncc_self_similarity_radius_middle),
                ),
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
                        "walked {:.0} grid px{}, kept at seed",
                        m.walked_px.expect("just matched"),
                        match m.walked_zncc {
                            Some(z) if z.is_finite() => format!(
                                " (ZNCC {} there)",
                                zncc_sentence(Some(z), m.walked_zncc_middle)
                            ),
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
