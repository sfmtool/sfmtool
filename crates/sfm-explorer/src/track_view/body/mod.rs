// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Track View's body: one track, drawn in one of two modes.
//!
//! See `specs/gui/track-view.md`. The body draws an `EditableTrack` with a
//! header, a toolbar, the threshold boxes and a table of its observations, in
//! one of two modes ([`BodyMode`]):
//!
//! - **Viewed**, while Track View's *Edit* box is clear: the viewed track
//!   ([`crate::bench::viewed`]), the selected point read as an editable track
//!   and held off every bench. Nothing about it can be changed. The header
//!   carries the point's own summary, the toolbar only where its evaluation
//!   stands, the boxes judge the readings by the session's read-only bars, and
//!   the *Keep* column is a *Verdict* column saying what those bars give each
//!   row.
//! - **Edited**, while the box is ticked: the focused item on the selected
//!   node's bench ([`crate::bench`]), with every step that acts on it.
//!
//! Every column, tile, crop and reading is drawn the same way in both modes,
//! so a committed point and its copy on the bench are read in one layout with
//! one set of numbers. The differences are branches on the mode.
//!
//! The body decides nothing. Each gesture lands in [`TrackBodyResponse`] and
//! the dock applies it through the `AppState` method that pushes the version,
//! or in Viewed mode moves the selection, for the reason every other panel's
//! response works that way: the panel holds `&AppState` while it draws, and a
//! step needs it mutably.
//!
//! Almost no state lives here. The bench is the node's, at its cursor, and the
//! viewed track and its bars are `AppState`'s, so what the panel owns is the
//! threshold boxes' values during a drag, the *Lock* box Image Detail reads a
//! dot drag by, the tiles it has rendered and their hover views, the header's
//! summary, and the judgement the boxes produce. These are cached against the
//! track's own `Arc` rather than recomputed per frame, because the judgement is
//! `verdicts_if_unpinned` run over a copy (and a copy of a track carries its
//! consensus bitmap) and a tile is a warp of a full-resolution photograph.
//! The row selection is not the panel's: it is the bench's, in
//! `AppState::bench_rows`, and a row click in Edited mode reports through
//! [`TrackBodyResponse::pick_row`].

use std::collections::HashMap;
use std::sync::Arc;

use sfmtool_core::bench::{
    bar_checks, verdicts_if_unpinned, BarChecks, EditableTrack, Observation, Provenance, Stage,
    StageKind, Thresholds, Verdict,
};
use sfmtool_core::{EditedReconstruction, Point3D, SfmrReconstruction};

use crate::bench::live::Evaluation;
use crate::bench::viewed::ViewedTrack;
use crate::bench::{NormalStep, SplitSettings};
use crate::scene::{ImageRef, PointRef, ReconId, SceneNode};
use crate::state::AppState;
use crate::track_view::EDIT_LABEL;

mod crop;
mod patch;
mod surface_plot;
mod table;
mod tile;

#[cfg(test)]
mod tests;

pub(crate) use patch::track_patch_image;
pub(crate) use table::RowSummary;

/// Display size of the track's own patch, left of the toolbar.
const STORED_PATCH_SIZE: f32 = 64.0;

/// Which of the two ways the body draws a track.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum BodyMode {
    /// The viewed track: the selected point, read-only. *Edit* is clear.
    Viewed,
    /// The focused item on the bench, with the steps that act on it. *Edit*
    /// is ticked.
    Edited,
}

/// What the body is drawing: the node, the mode, the track's label, and the
/// point the track was read from where that point is live at the cursor.
#[derive(Debug, Clone, PartialEq)]
struct Showing {
    node: ReconId,
    mode: BodyMode,
    label: String,
    /// The origin followed to the cursor, which is the viewed point itself in
    /// Viewed mode: what the crop's hover caption reads a feature index
    /// through.
    origin: Option<u32>,
}

/// The whole-track numbers the header prints beside the label: the error, the
/// track's length and the triangulation diagnostics, and for the viewed track
/// the point's own record, whose colour and homogeneous coordinates the header
/// shows.
#[derive(Debug, Clone, PartialEq)]
struct HeaderSummary {
    /// The point's stored record, for the viewed track only.
    point: Option<Point3D>,
    /// The reprojection error in pixels: the point's stored RMS error for the
    /// viewed track, the RMS of the kept rows' measured errors for a bench
    /// track. `None` where nothing has measured one.
    error_px: Option<f64>,
    /// How many observations the track keeps.
    track_length: usize,
    /// Whether the track is a direction rather than a place.
    at_infinity: bool,
    /// The largest angle between two observing rays, in degrees: `0` for a
    /// point at infinity or a single ray.
    max_angle_deg: f32,
    /// Inverse-depth z-score (`depth / σ_depth`); NaN when undefined.
    depth_z: f32,
    /// Condition number of the triangulation's normal matrix; NaN when
    /// undefined.
    condition: f32,
}

/// What a [`HeaderSummary`] was computed from: the mode, the node, the label,
/// the address of the track's `Arc` (zero for the viewed track, whose summary
/// reads the point and not the track) and the serial it was read at (the
/// document serial for the viewed track, the version serial for a bench one).
type SummaryKey = (BodyMode, ReconId, String, usize, u64);

/// What one frame of the body asks the dock to do.
///
/// Every field is one gesture, and at most one of them is set on a frame: the
/// entries that push a version are buttons and box releases, and each is
/// made once. In Viewed mode only the selection and hover fields,
/// `request_goto_point` and `viewed_thresholds` are ever set.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct TrackBodyResponse {
    /// The mode the body drew in, or `None` when it drew no track.
    pub mode: Option<BodyMode>,
    /// Viewed mode: a threshold box moved, carrying the read-only bars the
    /// boxes now stand at, for `AppState::set_viewed_thresholds`. Set on every
    /// frame of a drag, since the bars change nothing but what is drawn.
    pub viewed_thresholds: Option<Thresholds>,
    /// The toolbar's *Discard*.
    pub discard: Option<String>,
    /// *Rename* was committed: the item, and the label it should take.
    pub rename: Option<(String, String)>,
    /// *Fit*.
    pub fit: bool,
    /// *Fit Normal*, *Finite Diff Normal* or *Grid Plane Normal*, carrying
    /// which, and for the last two the *per axis* and *overlap* settings it
    /// was pressed at.
    pub(crate) normal: Option<NormalStep>,
    /// The *Stage* toggle, carrying the stage it asks for.
    pub set_stage: Option<StageKind>,
    /// A threshold box was released, or a value typed into one was
    /// committed: the bars the four boxes stand at, for the focused item.
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
    /// *Duplicate*: put a copy of the focused item on the bench beside it.
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
    /// Verdicts pinned as they stand in one step: the *Keep* heading's pin
    /// when no row is pinned, which names every row.
    pub pin_verdicts: Option<Vec<usize>>,
    /// The header's go-to button, or the empty state's *Go to Point...*: open
    /// the *Go to Point* dialog.
    pub request_goto_point: bool,
    /// A row was clicked -- select this image.
    pub select_image: Option<usize>,
    /// A row was double-clicked -- enter camera view for this image.
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

/// Track View's body state.
pub struct TrackBody {
    /// Where the threshold boxes stand.
    ///
    /// In Edited mode, the **focused item's own bars**, copied from it on
    /// every frame no box is being dragged, so an undo, a redo, a step over
    /// the wire or a change of focused item moves the boxes with it. Only
    /// during a drag does this hold a value the track does not: the drag
    /// repaints the table live, and its release applies the bars to the track
    /// as one version. The panel therefore never holds bars a *Fit* would not
    /// use.
    ///
    /// In Viewed mode, the session's read-only bars
    /// (`AppState::viewed_thresholds`), copied from the state every frame; a
    /// box that moves reports the new bars through
    /// [`TrackBodyResponse::viewed_thresholds`].
    thresholds: Thresholds,
    /// Whether a threshold box was being dragged on the last frame in Edited
    /// mode, which is what keeps [`TrackBody::thresholds`] from being reset to
    /// the track's bars in the middle of the drag.
    sliding: bool,
    /// What the boxes say about each observation of the track drawn, which is
    /// what its readings and its *Keep* or *Verdict* cell are coloured by:
    /// `None` for an observation nothing at the track's stage has measured,
    /// which the bars do not judge.
    judged: Vec<Option<Judgement>>,
    /// The mode, the track's label, the address of its `Arc` and the bars
    /// [`TrackBody::judged`] was computed from. A step on the track, or an
    /// evaluation of the viewed track landing, gives it a new `Arc`, which is
    /// what says the judgement is stale.
    judged_for: Option<(BodyMode, String, usize, Thresholds)>,
    /// Why the focused item cannot be committed, or `None` when it can, as of
    /// the value and the track [`TrackBody::commit_refusal_for`] last asked.
    ///
    /// Cached rather than asked per frame: the question is
    /// `sfmtool_core::bench::commit` itself, asked of the very track the button
    /// would commit so that the button and the step cannot disagree, and that
    /// builds a point record. It is stale exactly when the track's `Arc` or the
    /// node's version moves, which is what the key below holds.
    commit_refusal: Option<String>,
    /// The item, the track's `Arc` and the version [`TrackBody::commit_refusal`]
    /// was asked at.
    commit_refusal_for: Option<(String, usize, u64)>,
    /// What the panel last drew, which the row menus, the tiles and the crop
    /// captions are read against.
    showing: Option<Showing>,
    /// The header's summary, and what it was computed from. Computed once per
    /// key rather than per frame, because the triangulation diagnostics
    /// triangulate.
    summary: Option<(SummaryKey, Option<HeaderSummary>)>,
    /// A rename in progress: the item, and the text typed so far.
    renaming: Option<(String, String)>,
    /// The rendered tile of each observation, by observation index.
    ///
    /// Keyed by the row rather than by the image, because two observations can
    /// name one image and they are two pictures: at the cluster stage each has
    /// its own position and shape. A tile is a warp of a full-resolution
    /// photograph, so it is rendered once and kept; what says it is stale is
    /// [`TrackBody::tiles_for`].
    tiles: HashMap<usize, Option<egui::TextureHandle>>,
    /// The track and the exact track value [`TrackBody::tiles`] was rendered
    /// from: the label, and the address of the track's `Arc`. Any step on the
    /// track gives it a new `Arc`, and every step that moves a tile is one.
    tiles_for: Option<(String, usize)>,
    /// The hover view of each row's tile, by observation index: rendered the
    /// first time the pointer rests on that tile, and kept and dropped with
    /// [`TrackBody::tiles`], since it is the same picture made wider and goes
    /// stale exactly when the tile does. `None` is cached as the tile's is.
    contexts: HashMap<usize, Option<tile::DrawnContext>>,
    /// The crop of each row's photograph around the patch's outline, by
    /// observation index, kept and dropped with [`TrackBody::tiles`]: it moves
    /// exactly when the tile does. `None` is cached as the tile's is.
    crops: HashMap<usize, Option<crop::DrawnCrop>>,
    /// The hover view of each row's crop, rendered the first time the pointer
    /// rests on that crop and dropped with [`TrackBody::crops`].
    crop_contexts: HashMap<usize, Option<crop::DrawnCrop>>,
    /// The track's own patch drawn left of the toolbar, uploaded, or `None`
    /// until it has been asked for since the track moved. The inner `None` is
    /// a track with nothing to show, cached as the tile's is. Dropped with
    /// [`TrackBody::tiles`], since a step that moves the track gives it a new
    /// `Arc`.
    track_patch: Option<Option<egui::TextureHandle>>,
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
    /// Where the drawn track's evaluation stood when this frame drew it:
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
    /// How *Finite Diff Normal* and *Grid Plane Normal* cut the patch: the
    /// *per axis* and *overlap* boxes beside them. A tool setting for the reason [`TrackBody::lock`] is,
    /// kept for the session.
    split_settings: SplitSettings,
    /// Where the table's rows are scrolled to, both ways; the headings and the
    /// threshold row follow its sideways part. Kept to add a drag of the
    /// headings to.
    scroll_offset: egui::Vec2,
    /// The order the table's rows are drawn in, set by a click on a heading.
    /// A tool setting for the reason [`TrackBody::lock`] is, kept for the
    /// session and across tracks.
    sort: table::TableSort,
}

impl Default for TrackBody {
    fn default() -> Self {
        Self::new()
    }
}

impl TrackBody {
    pub fn new() -> Self {
        Self {
            thresholds: Thresholds::default(),
            sliding: false,
            judged: Vec::new(),
            judged_for: None,
            commit_refusal: None,
            commit_refusal_for: None,
            showing: None,
            summary: None,
            renaming: None,
            tiles: HashMap::new(),
            tiles_for: None,
            contexts: HashMap::new(),
            crops: HashMap::new(),
            crop_contexts: HashMap::new(),
            track_patch: None,
            plots: HashMap::new(),
            rows: Vec::new(),
            build_refusal: None,
            evaluation: Evaluation::Evaluating,
            lock: true,
            split_settings: SplitSettings::default(),
            scroll_offset: egui::Vec2::ZERO,
            sort: table::TableSort::default(),
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

    /// Where the drawn track's evaluation stood when the panel last drew it.
    #[cfg(test)]
    pub(crate) fn evaluation(&self) -> &Evaluation {
        &self.evaluation
    }

    /// The header's summary as the panel last drew it, or `None` when the
    /// track it drew has none (a cluster).
    #[cfg(test)]
    fn summary(&self) -> Option<&HeaderSummary> {
        self.summary
            .as_ref()
            .and_then(|(_, summary)| summary.as_ref())
    }

    /// Whether the *Lock* box is ticked: a track-stage dot drag in Image Detail
    /// moves the patch when it is, and one sighting's keypoint when it is not.
    ///
    /// The box's own state, whatever stage the focused item is in. At the
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
        self.crops.clear();
        self.crop_contexts.clear();
        self.track_patch = None;
        if self
            .showing
            .as_ref()
            .is_some_and(|showing| showing.node == id)
        {
            self.showing = None;
        }
        self.summary = None;
        self.judged_for = None;
        self.commit_refusal_for = None;
        self.build_refusal = None;
        self.sliding = false;
        self.judged.clear();
        self.rows.clear();
    }

    /// Forget what the last frame drew, for a frame that draws no track.
    pub(crate) fn draw_nothing(&mut self) {
        self.rows.clear();
        self.showing = None;
    }

    /// Draw the track Track View shows and report what the user did with it:
    /// the focused item in Edited mode when one is focused on the selected
    /// node, and otherwise the viewed track in Viewed mode. With neither, or
    /// with no node selected, it draws nothing and forgets the rows it drew.
    pub fn show(&mut self, ui: &mut egui::Ui, state: &AppState) -> TrackBodyResponse {
        let mut response = TrackBodyResponse::default();
        let panel_rect = ui.available_rect_before_wrap();
        if let Some(pos) = ui.input(|i| i.pointer.hover_pos()) {
            response.has_pointer = panel_rect.contains(pos);
        }

        let Some(node) = crate::scene::selected_node(&state.scene, state.selected_recon) else {
            self.draw_nothing();
            return response;
        };
        let id = node.id;
        if let Some(label) = state.focused_item_label(id).map(str::to_string) {
            let track = Arc::clone(
                node.history
                    .current_bench()
                    .track(&label)
                    .expect("the focused label names a track"),
            );
            response.mode = Some(BodyMode::Edited);
            self.show_edited(ui, state, node, &label, &track, &mut response);
        } else if let Some(viewed) = state.viewed_track().filter(|viewed| viewed.node == id) {
            response.mode = Some(BodyMode::Viewed);
            self.show_viewed(ui, state, node, viewed, &mut response);
        } else {
            self.draw_nothing();
        }
        response
    }

    /// What both modes do before drawing: record what is shown, and bring the
    /// judgement and the tiles up to the track.
    fn enter(&mut self, showing: Showing, track: &Arc<EditableTrack>, evaluation: Evaluation) {
        let (mode, label) = (showing.mode, showing.label.clone());
        self.showing = Some(showing);
        self.evaluation = evaluation;
        self.rejudge_if_stale(mode, &label, track);
        self.retile_if_stale(&label, track);
    }

    /// Edited mode: the focused item, its steps, and its bars.
    fn show_edited(
        &mut self,
        ui: &mut egui::Ui,
        state: &AppState,
        node: &SceneNode,
        label: &str,
        track: &Arc<EditableTrack>,
        response: &mut TrackBodyResponse,
    ) {
        let id = node.id;
        self.reseat_thresholds(track);
        // The ID of the point the track was read from, where that point is
        // still in the version at the cursor: what the header's copy button
        // copies, and what the *Go to Point* dialog takes back.
        let origin = state.resolved_origin(node, track);
        self.enter(
            Showing {
                node: id,
                mode: BodyMode::Edited,
                label: label.to_string(),
                origin,
            },
            track,
            state
                .bench_evaluation(id, label)
                .unwrap_or(Evaluation::Evaluating),
        );
        self.recheck_commit_if_stale(label, track, node);

        // Why an index cannot be built beside this node, cached against the
        // node: every row's menu asks it, and the question stats files.
        if self.build_refusal.as_ref().map(|(of, _)| *of) != Some(id) {
            self.build_refusal = Some((id, state.sift_sources_refusal(id)));
        }

        let key: SummaryKey = (
            BodyMode::Edited,
            id,
            label.to_string(),
            Arc::as_ptr(track) as usize,
            node.history.current_version().serial.as_u64(),
        );
        if self.summary.as_ref().map(|(of, _)| of) != Some(&key) {
            self.summary = Some((key, track_summary(node.recon(), track)));
        }
        let summary = self.summary.as_ref().and_then(|(_, s)| s.clone());

        let point_id = origin.map(|index| crate::scene::point_id(node, index as usize));
        response.request_goto_point =
            show_header(ui, label, track, point_id.as_deref(), summary.as_ref());
        // The track's own patch at the left, under the title, and the stage's
        // line, the toolbar and the boxes beside it, so the table's separator
        // runs straight under the patch.
        let patch = self.ensure_track_patch(ui.ctx(), label, track);
        // Greyed while the node is busy or its bench is view-only, with that
        // sentence: a release there would be refused, and a box that snapped
        // back after a drag would say less than one that could not be dragged.
        let busy = state.bench_edit_refusal(id);
        let hover = match &busy {
            Some(why) => BoxHover::Refused(why),
            None => BoxHover::Tip(EDITED_BARS_TIP),
        };
        let mut moved = BoxesMoved::default();
        ui.horizontal_top(|ui| {
            show_track_patch(ui, patch, track.stage_kind(), BodyMode::Edited);
            ui.vertical(|ui| {
                show_headline(ui, track);
                self.show_toolbar(ui, state, node, label, track, response);
                moved = geometry_search_box(ui, &mut self.thresholds, hover);
            });
        });
        ui.separator();
        moved.merge(self.show_table(ui, node.recon(), id, state, track, hover, response));
        // A release applies what the drag left the boxes at. Painted by this
        // frame's value from the next frame on, which is the frame the dock
        // has applied it by.
        response.apply_thresholds = self.bars_to_apply(moved, track);
    }

    /// Viewed mode: the viewed track, read-only, with the point's own summary
    /// and the read-only bars.
    fn show_viewed(
        &mut self,
        ui: &mut egui::Ui,
        state: &AppState,
        node: &SceneNode,
        viewed: &ViewedTrack,
        response: &mut TrackBodyResponse,
    ) {
        let id = node.id;
        let track = &viewed.track;
        // The session's bars, every frame: the dock applies what a box
        // reports at the end of the frame, so the next frame reads it back.
        self.thresholds = state.viewed_thresholds.clone();
        self.sliding = false;
        self.enter(
            Showing {
                node: id,
                mode: BodyMode::Viewed,
                label: viewed.label.clone(),
                origin: Some(viewed.point),
            },
            track,
            viewed.evaluation.clone(),
        );

        let key: SummaryKey = (
            BodyMode::Viewed,
            id,
            viewed.label.clone(),
            0,
            viewed.document.as_u64(),
        );
        if self.summary.as_ref().map(|(of, _)| of) != Some(&key) {
            self.summary = Some((key, point_summary(node.edited(), viewed.point)));
        }
        let summary = self.summary.as_ref().and_then(|(_, s)| s.clone());

        // Minted every frame rather than kept from the build: an edit or an
        // undo can change which content the earliest rule mints against
        // without moving the selection, and the header must show the ID that
        // resolves now.
        let point_id = crate::scene::point_id(node, viewed.point as usize);
        response.request_goto_point = show_viewed_header(ui, &point_id, summary.as_ref());
        let on_bench = state.bench_item_from_point(PointRef::new(id, viewed.point as usize));
        let patch = self.ensure_track_patch(ui.ctx(), &viewed.label, track);
        ui.horizontal_top(|ui| {
            show_track_patch(ui, patch, track.stage_kind(), BodyMode::Viewed);
            ui.vertical(|ui| {
                ui.weak(bench_line(on_bench.as_deref()));
                ui.horizontal_wrapped(|ui| show_evaluation(ui, &self.evaluation));
                geometry_search_box(ui, &mut self.thresholds, BoxHover::Tip(VIEWED_BARS_TIP));
            });
        });
        ui.separator();
        let hover = BoxHover::Tip(VIEWED_BARS_TIP);
        self.show_table(ui, node.recon(), id, state, track, hover, response);
        // The boxes in Viewed mode hold the session's read-only bars, and are
        // never greyed: a drag or a typed value recolours the readings and the
        // *Verdict* column and changes nothing else. Any frame a box moved
        // them hands the new bars to the dock, for
        // `AppState::set_viewed_thresholds`.
        response.viewed_thresholds =
            (self.thresholds != state.viewed_thresholds).then(|| self.thresholds.clone());
    }

    /// The mode the body last drew in, or `None` when it drew no track.
    fn mode(&self) -> Option<BodyMode> {
        self.showing.as_ref().map(|showing| showing.mode)
    }

    /// The toolbar: every step that acts on the focused item, each greyed with
    /// the sentence naming what is missing.
    fn show_toolbar(
        &mut self,
        ui: &mut egui::Ui,
        state: &AppState,
        node: &SceneNode,
        label: &str,
        track: &EditableTrack,
        response: &mut TrackBodyResponse,
    ) {
        let id = node.id;
        // Every step here edits the track, and is greyed by the busy node or
        // a view-only bench; Discard and Rename are not edits of the track and
        // are greyed by the busy node alone.
        let busy = state.bench_edit_refusal(id);
        let housekeeping = state.busy_refusal(id);
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
                FIT_NORMAL_LABEL,
                refusals.normal.clone(),
                "Turn the patch to the normal at which its sightings agree best, \
                keeping its centre",
            ) {
                response.normal = Some(NormalStep::Photometric);
            }
            if entry(
                ui,
                FINITE_DIFF_NORMAL_LABEL,
                refusals.normal.clone(),
                "Cut the patch into smaller pieces along each of its axes, fit each \
                piece, and turn the patch to the plane through where they land, \
                keeping its centre",
            ) {
                response.normal = Some(NormalStep::FiniteDifference(self.split_settings));
            }
            if entry(
                ui,
                GRID_PLANE_NORMAL_LABEL,
                refusals.normal.clone(),
                "Cut the whole patch into a grid of smaller pieces, fit each piece, \
                 and turn the patch to the plane through where they land, keeping \
                 its centre",
            ) {
                response.normal = Some(NormalStep::GridPlane(self.split_settings));
            }
            self.show_split_settings(ui, refusals.normal.is_some());
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
            let duplicate_refusal = busy.clone().or_else(|| state.duplicate_refusal(id, label));
            if entry(
                ui,
                "Duplicate",
                duplicate_refusal,
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
            if entry(
                ui,
                "Discard",
                housekeeping.clone(),
                "Take this track off the bench",
            ) {
                response.discard = Some(label.to_string());
            }
        });
        ui.horizontal_wrapped(|ui| {
            self.show_lock(ui, track);
            self.show_rename(ui, label, housekeeping, response);
        });
    }

    /// The *per axis* and *overlap* boxes that say how *Finite Diff Normal*
    /// and *Grid Plane Normal* cut the patch. Greyed with the buttons, since
    /// they say nothing while those cannot run.
    fn show_split_settings(&mut self, ui: &mut egui::Ui, greyed: bool) {
        use sfmtool_core::bench::normal::{MAX_OVERLAP, MAX_PIECES, MIN_PIECES};
        ui.add_enabled_ui(!greyed, |ui| {
            ui.add(
                egui::DragValue::new(&mut self.split_settings.pieces)
                    .range(MIN_PIECES..=MAX_PIECES)
                    .suffix(" per axis"),
            )
            .on_hover_text(
                "How many pieces the patch is cut into along each of its two axes: a row \
                 of that many along each axis for Finite Diff Normal, a grid of that many \
                 by that many for Grid Plane Normal",
            );
            ui.add(
                egui::DragValue::new(&mut self.split_settings.overlap_percent)
                    .range(0.0..=MAX_OVERLAP * 100.0)
                    .speed(1.0)
                    .max_decimals(0)
                    .suffix("% overlap"),
            )
            .on_hover_text("How much neighbouring pieces overlap, as a share of a piece's side");
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
        response: &mut TrackBodyResponse,
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

    /// The bars to apply after the boxes moved as `moved` on this frame, in
    /// Edited mode: a drag paints the table live, and its release (or a typed
    /// value's commit) hands back the bars to apply as one version. `None` on
    /// every other frame, and on a release that left the bars where the track
    /// has them.
    fn bars_to_apply(&mut self, moved: BoxesMoved, track: &EditableTrack) -> Option<Thresholds> {
        self.sliding = moved.sliding;
        (moved.released && !moved.sliding && self.thresholds != track.thresholds)
            .then(|| self.thresholds.clone())
    }

    /// Put the boxes where the focused item's own bars are, unless a box
    /// is being dragged.
    ///
    /// Every frame, rather than when something is seen to change: the boxes
    /// show the track's bars and nothing else, so whatever moved them -- a
    /// box's release here, `apply_bench_track_thresholds` over the wire, an
    /// undo or redo of either, another item focused -- the boxes follow.
    fn reseat_thresholds(&mut self, track: &EditableTrack) {
        if !self.sliding || self.mode() != Some(BodyMode::Edited) {
            self.thresholds = track.thresholds.clone();
            self.sliding = false;
        }
    }

    /// Recompute the judgement when the track or the bars have moved.
    ///
    /// Both halves are the core's own: the readings are judged by
    /// `bar_checks`, and the proposal is `verdicts_if_unpinned`, which gives an
    /// unpinned row what applying the bars would do -- the same core step a
    /// box's release applies, so a row can never be shown one way and turned
    /// another when the box is let go -- and a pinned row what unpinning it
    /// would. Both run with the boxes' bars, so they follow a drag live. The
    /// viewed track's rows are all pinned, so its proposals are exactly the
    /// verdicts its *Verdict* column shows: what the bench's own evaluation
    /// would give each row once the point is on the bench and unpinned.
    fn rejudge_if_stale(&mut self, mode: BodyMode, label: &str, track: &Arc<EditableTrack>) {
        let key = (
            mode,
            label.to_string(),
            Arc::as_ptr(track) as usize,
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

    /// Ask again why the focused item cannot be committed, when the track or the
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
    /// `None` from the render is a real answer and is cached as one: an
    /// observation with no patch behind it yet has no tile, and re-attempting
    /// the warp every frame would be the cost the cache exists to avoid. A
    /// photograph the photograph cache has not decoded yet is not cached as a
    /// miss: its decode is running off the GUI thread and repaints when it
    /// ends, and the next frame's look finds it.
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
        let id = self.showing.as_ref().map(|showing| showing.node)?;
        let image = ImageRef::new(id, track.observations.get(observation)?.image as usize);
        let src = crate::state::peek_full_res_pyramid(&state.photographs, recon, image.index())?;
        let tile = tile::render(
            ctx,
            recon,
            track,
            observation,
            &src,
            format!("bench_tile_{}_{observation}", image.index()),
        );
        let texture_id = tile.as_ref().map(|texture| texture.id());
        self.tiles.insert(observation, tile);
        texture_id
    }

    /// The hover view of one row's tile, rendering it if this is the first
    /// frame that has asked for it since the track moved.
    ///
    /// Asked for only while the pointer rests on the tile, so a table of many
    /// rows renders the wider picture for the rows a person looks at and no
    /// others. As with [`Self::ensure_tile`], a photograph not decoded yet is
    /// not cached as a miss.
    fn ensure_context(
        &mut self,
        ctx: &egui::Context,
        recon: &SfmrReconstruction,
        track: &EditableTrack,
        observation: usize,
        state: &AppState,
    ) -> Option<&tile::DrawnContext> {
        if !self.contexts.contains_key(&observation) {
            let id = self.showing.as_ref().map(|showing| showing.node)?;
            let image = ImageRef::new(id, track.observations.get(observation)?.image as usize);
            let src =
                crate::state::peek_full_res_pyramid(&state.photographs, recon, image.index())?;
            let drawn = tile::context(recon, track, observation, &src).map(|context| {
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

    /// The track's own patch, uploading it if this is the first frame that has
    /// asked for it since the track moved: the consensus bitmap at the track
    /// stage, the template at the cluster stage, `None` where there is
    /// neither.
    fn ensure_track_patch(
        &mut self,
        ctx: &egui::Context,
        label: &str,
        track: &EditableTrack,
    ) -> Option<egui::TextureId> {
        let texture = self.track_patch.get_or_insert_with(|| {
            let image = patch::track_patch_image(track)?;
            Some(ctx.load_texture(
                format!("bench_track_patch_{label}"),
                image,
                egui::TextureOptions::NEAREST,
            ))
        });
        texture.as_ref().map(|texture| texture.id())
    }

    /// The crop one row draws beside its tile, cutting it out of the
    /// photograph if this is the first frame that has asked for it since the
    /// track moved. `None` is cached as the tile's is, and, as with the tile,
    /// a photograph not decoded yet is not cached as a miss.
    fn ensure_crop(
        &mut self,
        ctx: &egui::Context,
        recon: &SfmrReconstruction,
        track: &EditableTrack,
        observation: usize,
        state: &AppState,
    ) -> Option<&crop::DrawnCrop> {
        if !self.crops.contains_key(&observation) {
            let drawn = self.drawn_crop(ctx, recon, track, observation, state, false)?;
            self.crops.insert(observation, drawn);
        }
        self.crops.get(&observation).and_then(Option::as_ref)
    }

    /// The hover view of one row's crop, rendering it if this is the first
    /// frame that has asked for it since the track moved. Asked for only while
    /// the pointer rests on the crop, as the tile's hover view is.
    fn ensure_crop_context(
        &mut self,
        ctx: &egui::Context,
        recon: &SfmrReconstruction,
        track: &EditableTrack,
        observation: usize,
        state: &AppState,
    ) -> Option<&crop::DrawnCrop> {
        if !self.crop_contexts.contains_key(&observation) {
            let drawn = self.drawn_crop(ctx, recon, track, observation, state, true)?;
            self.crop_contexts.insert(observation, drawn);
        }
        self.crop_contexts
            .get(&observation)
            .and_then(Option::as_ref)
    }

    /// A row's crop, or with `in_context` its hover view, cut from the
    /// photograph cache's pyramid and uploaded.
    ///
    /// The outer `None` is "not yet": no row, or a photograph the cache has
    /// not decoded, which the caller does not remember. The inner `None` is a
    /// crop that cannot be cut from the photograph, which it does.
    fn drawn_crop(
        &self,
        ctx: &egui::Context,
        recon: &SfmrReconstruction,
        track: &EditableTrack,
        observation: usize,
        state: &AppState,
        in_context: bool,
    ) -> Option<Option<crop::DrawnCrop>> {
        let id = self.showing.as_ref().map(|showing| showing.node)?;
        let row = track.observations.get(observation)?;
        let image = ImageRef::new(id, row.image as usize);
        let src = crate::state::peek_full_res_pyramid(&state.photographs, recon, image.index())?;
        let cut = || {
            let (picture, name, options) = if in_context {
                (
                    crop::context(recon, track, observation, &src)?,
                    format!("bench_crop_context_{}_{observation}", image.index()),
                    egui::TextureOptions::LINEAR,
                )
            } else {
                (
                    crop::image(recon, track, observation, &src)?,
                    format!("bench_crop_{}_{observation}", image.index()),
                    // Each photograph pixel magnified to a block, as the tile
                    // is, so the crop shows the photograph's own resolution.
                    egui::TextureOptions {
                        magnification: egui::TextureFilter::Nearest,
                        minification: egui::TextureFilter::Linear,
                        ..egui::TextureOptions::NEAREST
                    },
                )
            };
            Some(crop::DrawnCrop::new(
                ctx,
                picture,
                row.verdict,
                name,
                options,
            ))
        };
        Some(cut())
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
        self.crops.clear();
        self.crop_contexts.clear();
        self.track_patch = None;
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

/// The text after the minimum-ZNCC box in the threshold row: the unit and the
/// name the ZNCC cell's first line prints with. In one constant so the tests
/// aim at the text drawn.
pub(crate) const MIN_ZNCC_LABEL: &str = "% whole";

/// The minimum-ZNCC box's hover text.
const MIN_ZNCC_TIP: &str = "The lowest whole-patch ZNCC a sighting may have. A row whose \
    whole ZNCC is under it is painted out.";

/// The text after the minimum-middle-ZNCC box, under the one above.
pub(crate) const MIN_ZNCC_MIDDLE_LABEL: &str = "% mid";

/// The minimum-middle-ZNCC box's hover text.
const MIN_ZNCC_MIDDLE_TIP: &str = "The lowest middle ZNCC a sighting may have, read over the \
    centred square half the patch's width. A row whose middle ZNCC is under it is painted out. \
    0 turns the bar off.";

/// The geometry search box's label, above the table: its bar judges no
/// column, so it does not sit in the threshold row.
pub(crate) const GEOMETRY_SEARCH_LABEL: &str = "geometry search min relative ZNCC (%)";

/// The geometry search box's hover text.
const GEOMETRY_SEARCH_TIP: &str = "The bar Find matches by geometry admits a photograph by: its \
    ZNCC to the track's reference appearance, as a percent of the track's own agreement with \
    that reference. It judges no row, so moving it changes no verdict.";

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

/// What the boxes' value fields say when hovered: the mode's own hint, or the
/// sentence they are greyed with.
#[derive(Clone, Copy)]
pub(super) enum BoxHover<'a> {
    /// Enabled, with this hint.
    Tip(&'a str),
    /// Greyed, with this refusal.
    Refused(&'a str),
}

impl BoxHover<'_> {
    /// Whether the boxes take input.
    pub(super) fn enabled(self) -> bool {
        matches!(self, Self::Tip(_))
    }
}

/// How the boxes moved on one frame.
#[derive(Default)]
pub(super) struct BoxesMoved {
    /// A box is being dragged.
    pub(super) sliding: bool,
    /// A drag ended, or a typed value or an arrow key changed a box.
    pub(super) released: bool,
}

impl BoxesMoved {
    /// Fold in how another box moved on the same frame.
    pub(super) fn merge(&mut self, other: BoxesMoved) {
        self.sliding |= other.sliding;
        self.released |= other.released;
    }
}

/// One threshold box over `value`, greyed when `hover` is a refusal.
///
/// Each bar is a box that is dragged left and right to change it, or clicked
/// to type into: a slider's rail beside the box would say nothing the box
/// does not.
pub(super) fn bar_box(
    ui: &mut egui::Ui,
    value: egui::DragValue<'_>,
    hover: BoxHover<'_>,
) -> BoxesMoved {
    // A typed value lands when the field is left, not per keystroke, so
    // typing "90" is one change and not two.
    let r = ui.add_enabled(hover.enabled(), value.update_while_editing(false));
    let r = match hover {
        BoxHover::Refused(why) => r.on_disabled_hover_text(why),
        BoxHover::Tip(tip) => r.on_hover_text(tip),
    };
    BoxesMoved {
        sliding: r.dragged(),
        // A drag ends in its release; a typed value or an arrow key changes
        // the value with no drag at all.
        released: r.drag_stopped() || (r.changed() && !r.dragged()),
    }
}

/// The geometry search's bar, above the table, after its label. It judges
/// no column, so it has no place in the threshold row under the headings;
/// it is still one of the track's bars, and applies with them.
fn geometry_search_box(
    ui: &mut egui::Ui,
    bars: &mut Thresholds,
    hover: BoxHover<'_>,
) -> BoxesMoved {
    ui.horizontal(|ui| {
        ui.add_enabled(hover.enabled(), egui::Label::new(GEOMETRY_SEARCH_LABEL))
            .on_hover_text(GEOMETRY_SEARCH_TIP);
        bar_box(
            ui,
            percent(egui::DragValue::new(
                &mut bars.geometry_search_min_relative_zncc,
            )),
            hover,
        )
    })
    .inner
}

/// The hover text of a box in Edited mode.
const EDITED_BARS_TIP: &str = "Applies to the track when released: one version, which Undo \
    reverses";

/// The hover text of a box in Viewed mode.
const VIEWED_BARS_TIP: &str = "Judges the readings and the Verdict column by this bar, and \
    changes nothing on the point. The bars are kept for the session, and a point put on the \
    bench from here takes them.";

/// The text after the shift box in the threshold row, the unit the Shift
/// cell prints in. The box is the bar the painting judges a shift by, the
/// radius the evaluation looks for each peak within, and the bound on how far
/// a fit may move a sighting.
pub(crate) const MAX_SHIFT_LABEL: &str = "px";

/// The shift box's hover text.
const MAX_SHIFT_TIP: &str = "The largest shift a sighting may have, in patch-grid px: how \
    far the correlation peak may sit from where the sighting is. The evaluation looks for each \
    peak within this distance, a row whose peak is further is painted out, and a fit moves no \
    sighting further than this.";

/// The text after the self-similarity box in the threshold row, the unit and
/// the name of the self-similarity cell's first line. The box is the largest
/// ZNCC self-similarity radius an observation's whole tile may have.
pub(crate) const MAX_SELF_SIMILARITY_LABEL: &str = "px whole";

/// The self-similarity box's hover text.
const MAX_SELF_SIMILARITY_TIP: &str = "The largest ZNCC self-similarity radius a sighting's \
    own tile may have, in patch-grid px: how far the tile can slide over itself and still \
    match. A row whose whole tile reads further than this is painted out, since a match \
    cannot pin its position, as along a straight edge or over a flat patch. At 3, the largest \
    radius read, it turns nothing out.";

/// The text after the projection error box in the threshold row, the unit of
/// the *Proj. err* cell's first line. The box is the largest reprojection
/// error an observation may have.
pub(crate) const MAX_PROJECTION_ERROR_LABEL: &str = "px";

/// The projection error box's hover text.
const MAX_PROJECTION_ERROR_TIP: &str = "The largest reprojection error a sighting may have, in \
    the photograph's pixels: how far it may sit from where the track's point projects. A row \
    further than this is painted out. The bar judges the point as much as the sighting: when \
    every row fails it, the point is off rather than the sightings. Track stage only, and 0 \
    turns it off.";

/// A kept-at-seed row's menu entry, which puts the sighting where the fit's
/// walk would have taken it.
pub(crate) const ACCEPT_WALK_LABEL: &str = "Accept walk";

/// The toolbar entry that turns the patch to its photometric normal, in one
/// constant so the tests aim at the label drawn.
pub(crate) const FIT_NORMAL_LABEL: &str = "Fit Normal";

/// The toolbar entry that turns the patch to the plane its fitted pieces lie
/// on.
pub(crate) const FINITE_DIFF_NORMAL_LABEL: &str = "Finite Diff Normal";

/// The toolbar entry that turns the patch to the plane through a grid of its
/// fitted pieces.
pub(crate) const GRID_PLANE_NORMAL_LABEL: &str = "Grid Plane Normal";

/// The Edited-mode checkbox that says whether Image Detail's dot drag moves the
/// patch or one sighting, in one constant so the tests aim at the label drawn.
pub(crate) const LOCK_LABEL: &str = "Lock";

/// Why *Lock* is greyed at the cluster stage.
pub(crate) const LOCK_AT_CLUSTER: &str = "A cluster has no shared patch: every sighting is \
    already moved on its own, locked or not.";

/// The observation row's context-menu entry, in one constant, as the Image
/// Detail menu's entries are: the label is quoted in a refusal and read back by
/// a test, and three spellings of one entry would drift.
pub(crate) const SEARCH_DESCRIPTORS_LABEL: &str = "Find matches by SIFT query";

/// The track-stage geometry search entry. It is separate from the SIFT label
/// because it reads poses and photographs, and requires no descriptor index.
pub(crate) const SEARCH_GEOMETRY_LABEL: &str = "Find matches by geometry";

/// What the toolbar says while an evaluation of the focused item's current
/// inputs is running or waiting to start, and what each row's status cell says
/// then.
pub(crate) const EVALUATING_LABEL: &str = "Evaluating\u{2026}";

/// What the toolbar says once the numbers are the evaluation of the track as
/// it stands.
pub(crate) const EVALUATED_LABEL: &str = "Evaluated";

/// The status cell of a row whose track has no evaluation of its current
/// inputs and gets none until a step changes them.
pub(crate) const NOT_EVALUATED: &str = "not evaluated";

/// Where the focused item's evaluation stands, at the head of the toolbar.
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

/// The header in Edited mode: what the focused item is, what the last
/// evaluation of it made of it, and at the track stage its summary. Returns
/// whether its go-to button was clicked.
///
/// `point_id` is the ID of the point the track was read from, when that point
/// is still in the version at the cursor. It carries the copy and the go-to
/// buttons the Viewed header draws beside a point ID, so an ID copied here is
/// one the *Go to Point* dialog takes back. A track put on the bench from a
/// point is labelled with that ID unless it was renamed, and the ID is then
/// printed once, as the label.
fn show_header(
    ui: &mut egui::Ui,
    label: &str,
    track: &EditableTrack,
    point_id: Option<&str>,
    summary: Option<&HeaderSummary>,
) -> bool {
    use crate::track_view::header_buttons::{copy_button, goto_button};
    let (kept, out) = track.verdict_counts();
    let pinned = track.observations.iter().filter(|o| o.pinned).count();
    let mut goto_clicked = false;
    ui.horizontal_wrapped(|ui| {
        if matches!(&track.stage, Stage::Track(p) if p.at_infinity) {
            crate::track_view::infinity_mark(ui);
        }
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
        if let Some(summary) = summary {
            show_summary(ui, summary);
        }
    });
    goto_clicked
}

/// The header in Viewed mode: the point's colour, its ID with the copy and
/// go-to buttons, its homogeneous coordinates with their own copy button, and
/// its summary. Returns whether the go-to button was clicked.
fn show_viewed_header(ui: &mut egui::Ui, point_id: &str, summary: Option<&HeaderSummary>) -> bool {
    use crate::track_view::header_buttons::{copy_button, goto_button};
    let point = summary.and_then(|summary| summary.point.as_ref());
    let mut goto_clicked = false;
    ui.horizontal_wrapped(|ui| {
        if let Some(point) = point {
            let [r, g, b] = point.color;
            let (rect, swatch) =
                ui.allocate_exact_size(egui::vec2(16.0, 16.0), egui::Sense::hover());
            ui.painter()
                .rect_filled(rect, 2.0, egui::Color32::from_rgb(r, g, b));
            ui.painter().rect_stroke(
                rect,
                2.0,
                egui::Stroke::new(1.0_f32, ui.visuals().weak_text_color()),
                egui::StrokeKind::Outside,
            );
            swatch.on_hover_text(format!("rgb({r}, {g}, {b})"));
            if point.w == 0.0 {
                crate::track_view::infinity_mark(ui);
            }
        }
        ui.label(egui::RichText::new(point_id).monospace().strong());
        if copy_button(ui, "Copy Point ID") {
            ui.ctx().copy_text(point_id.to_string());
        }
        // Beside Copy, because these are the two halves of one round trip:
        // copy an ID out of this header, paste it back into the dialog this
        // button opens -- here, or in another session entirely.
        goto_clicked = goto_button(ui);
        if let Some(point) = point {
            ui.label("|");
            // Homogeneous, because `w` is the whole difference between a
            // position and a direction: it is `1` for a finite point and `0`
            // for one at infinity, whose `position` is then a unit direction
            // rather than a place.
            let coords = format!(
                "{:.3}, {:.3}, {:.3}, {:.0}",
                point.position.x, point.position.y, point.position.z, point.w
            );
            ui.label(format!("xyzw: ({coords})"));
            if copy_button(ui, "Copy coordinates") {
                ui.ctx().copy_text(coords);
            }
        }
        if let Some(summary) = summary {
            show_summary(ui, summary);
        }
    });
    goto_clicked
}

/// The summary's readings, each after a `|`: the error, the track's length,
/// `at infinity` for a direction, and the three triangulation diagnostics,
/// each left out where it is not defined.
fn show_summary(ui: &mut egui::Ui, summary: &HeaderSummary) {
    ui.label("|");
    ui.label(match summary.error_px {
        Some(px) if px.is_finite() => format!("error: {px:.2}px"),
        _ => "error: -".to_string(),
    });
    ui.label("|");
    ui.label(format!("track: {} obs", summary.track_length));
    // Said in words as well as in `w`, because the rest of this row is about
    // a point that has a place and this one does not: the lines that would
    // say where it is are absent rather than zero, and a reader is owed the
    // reason.
    if summary.at_infinity {
        ui.label("|");
        ui.label(egui::RichText::new("at infinity").color(ui.visuals().warn_fg_color));
    }
    if summary.max_angle_deg > 0.0 {
        ui.label("|");
        ui.label(format!("max pair angle: {:.1}°", summary.max_angle_deg));
    }
    // Complementary to the max angle: scale-free, and correct in the
    // near-infinity regime.
    if summary.depth_z.is_finite() {
        ui.label("|");
        ui.label(format!("depth z: {:.1}", summary.depth_z));
    }
    if summary.condition.is_finite() {
        ui.label("|");
        ui.label(format!("cond: {:.0}", summary.condition));
    }
}

/// The summary of the committed point at `point`, read from the version at the
/// cursor, or `None` when it is not a live point there.
fn point_summary(edited: &EditedReconstruction, point: u32) -> Option<HeaderSummary> {
    let view = edited.point(point)?;
    let stored = view.point();
    let table = &edited.base.image_table;
    let images: Vec<usize> = view
        .observations()
        .iter()
        .map(|obs| obs.image_index as usize)
        .collect();
    let rays = crate::metrics::observation_rays(
        table,
        &stored.position,
        stored.is_at_infinity(),
        images.iter().copied(),
    );
    let (condition, depth_z) = crate::metrics::compute_point_diagnostics(table, &view);
    Some(HeaderSummary {
        point: Some(stored.clone()),
        error_px: Some(f64::from(stored.error)),
        track_length: images.len(),
        at_infinity: stored.is_at_infinity(),
        max_angle_deg: crate::metrics::compute_max_pairwise_angle(&rays),
        depth_z,
        condition,
    })
}

/// The summary of a bench track at the track stage, from its own position and
/// its `in` observations, or `None` at the cluster stage, which has no
/// position.
///
/// The error is the root mean square of the kept rows' measured reprojection
/// errors, since a bench track carries no stored error of its own; `None`
/// until a reading has measured one. Without a position nothing can be
/// triangulated, and the angle and the diagnostics are left undefined.
fn track_summary(recon: &SfmrReconstruction, track: &EditableTrack) -> Option<HeaderSummary> {
    let Stage::Track(payload) = &track.stage else {
        return None;
    };
    let kept: Vec<&Observation> = track
        .observations
        .iter()
        .filter(|o| o.verdict == Verdict::In)
        .collect();
    let errors: Vec<f64> = kept
        .iter()
        .filter_map(|o| o.track.as_ref()?.reprojection_error)
        .filter(|e| e.is_finite())
        .collect();
    let error_px = (!errors.is_empty())
        .then(|| (errors.iter().map(|e| e * e).sum::<f64>() / errors.len() as f64).sqrt());
    let images = || kept.iter().map(|o| o.image as usize);
    let table = &recon.image_table;
    let (max_angle_deg, condition, depth_z) = match payload.position {
        Some(position) => {
            let rays =
                crate::metrics::observation_rays(table, &position, payload.at_infinity, images());
            let (condition, depth_z) = if payload.at_infinity {
                (f32::NAN, f32::NAN)
            } else {
                crate::metrics::compute_position_diagnostics(
                    table,
                    &position,
                    error_px.unwrap_or(1.0),
                    images(),
                )
            };
            (
                crate::metrics::compute_max_pairwise_angle(&rays),
                condition,
                depth_z,
            )
        }
        None => (0.0, f32::NAN, f32::NAN),
    };
    Some(HeaderSummary {
        point: None,
        error_px,
        track_length: kept.len(),
        at_infinity: payload.at_infinity,
        max_angle_deg,
        depth_z,
        condition,
    })
}

/// The line under the Viewed header saying how to change the track: tick
/// *Edit*, or, when the point already has an item on the bench, that the
/// tick opens that item.
fn bench_line(on_bench: Option<&str>) -> String {
    match on_bench {
        Some(label) => format!("On the bench as {label}. Tick {EDIT_LABEL} to open it."),
        None => format!("Tick {EDIT_LABEL} to work on this track."),
    }
}

/// The stage's own line under the header: at the cluster stage the reference
/// observation and whether a template has been cut, and at the track stage the
/// coordinate and the last triangulation's condition number, or the sentence
/// saying nothing has triangulated it yet.
fn show_headline(ui: &mut egui::Ui, track: &EditableTrack) {
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
            // which they are looking at. "at infinity" is the Viewed header's own word
            // for the same row.
            //
            // The track's own flag and not its patch's `w`: a point put on the
            // bench from a node that stores no patch frames has no patch to read
            // a `w` off, and a bearing it came from is still a bearing.
            ui.weak(match payload.position {
                Some(_) => format!(
                    "{}{}",
                    position_text(payload),
                    match payload.condition_number {
                        Some(condition) => format!(", condition {condition:.1}"),
                        None => String::new(),
                    }
                ),
                None => position_text(payload),
            });
        }
    }
}

/// Where a track-stage track is: `Position (x, y, z)`, `Bearing (x, y, z), at
/// infinity`, or the sentence saying nothing has triangulated it yet. The
/// headline prints it with the condition number after it, and a recent item's
/// hover text prints it alone.
pub(crate) fn position_text(payload: &sfmtool_core::bench::TrackPayload) -> String {
    let at_infinity = payload.at_infinity;
    match payload.position {
        Some(position) => format!(
            "{} ({:.3}, {:.3}, {:.3}){}",
            if at_infinity { "Bearing" } else { "Position" },
            position.x,
            position.y,
            position.z,
            if at_infinity { ", at infinity" } else { "" },
        ),
        None => "No position: nothing has triangulated this track yet".to_string(),
    }
}

/// The track's own patch, at [`STORED_PATCH_SIZE`], left of the headline, the
/// toolbar and the boxes: at the track stage the consensus bitmap the
/// observations were fused into, which for the viewed track is the point's
/// stored patch and which a commit writes as it, and at the cluster stage the
/// template the members register onto. It carries no label, since the picture
/// says what it is. With nothing to show -- a point with no stored patch, a
/// track not yet fused, a cluster with no template cut -- the slot is an empty
/// frame of the same size, so the controls beside it do not move when a fit or
/// a stage change fills it.
fn show_track_patch(
    ui: &mut egui::Ui,
    texture: Option<egui::TextureId>,
    stage: StageKind,
    mode: BodyMode,
) {
    let size = STORED_PATCH_SIZE;
    let (rect, response) = ui.allocate_exact_size(egui::vec2(size, size), egui::Sense::hover());
    match texture {
        Some(texture) => {
            ui.painter().image(
                texture,
                rect,
                egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                egui::Color32::WHITE,
            );
            response.on_hover_text(match (stage, mode) {
                (StageKind::Track, BodyMode::Viewed) => {
                    "The point's stored patch: the consensus of its observations"
                }
                (StageKind::Track, BodyMode::Edited) => {
                    "The track's patch: the consensus of its observations, which a commit \
                     writes as the point's stored patch"
                }
                (StageKind::Cluster, _) => {
                    "The cluster's template, which every member registers onto"
                }
            });
        }
        None => {
            ui.painter()
                .rect_filled(rect, 2.0, ui.visuals().faint_bg_color);
            response.on_hover_text(match (stage, mode) {
                (StageKind::Track, BodyMode::Viewed) => "The point has no stored patch",
                (StageKind::Track, BodyMode::Edited) => {
                    "No patch yet: fit the track to fuse its observations into one"
                }
                (StageKind::Cluster, _) => {
                    "No template yet: the cluster's first evaluation cuts one"
                }
            });
        }
    }
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
    /// The three normal entries, which share their refusals.
    pub(super) normal: Option<String>,
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
/// Split out of [`TrackBody::show_toolbar`] so the rule can be read without a
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
        normal: refused(
            sfmtool_core::bench::normal_preconditions(track)
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

/// The two bullets the crop's hover view adds under its picture: the pixel the
/// observation sits at, and its feature index, or `None` for an observation
/// nothing has placed.
///
/// The feature index is the `.sift` feature the observation is: named by a
/// row put on by index, or, for a row read from the point the track came
/// from, looked up in that point's observation of the same image. A
/// reconstruction that stores its keypoints itself has no `.sift` feature
/// behind them, and the index given is then the observation's place in the
/// point's track. `origin` is that point, followed to the version `edited`
/// holds.
fn crop_caption(
    edited: &EditedReconstruction,
    origin: Option<u32>,
    row: &Observation,
) -> Option<String> {
    let pixel = crate::bench::observation_site(row)?.pixel;
    let feature = match row.provenance {
        Provenance::Descriptor { feature } => format!("Index: feature {feature} of its .sift file"),
        Provenance::Origin => {
            let found = origin.and_then(|point| {
                let view = edited.point(point)?;
                let k = view
                    .observations()
                    .iter()
                    .position(|obs| obs.image_index == row.image)?;
                Some((view.feature_indexes().map(|f| f[k]), k))
            });
            match found {
                Some((Some(feature), _)) => format!("Index: feature {feature} of its .sift file"),
                Some((None, k)) => format!("Index: observation {k} of the point"),
                None => "No feature index: the point it was read from is not in this version"
                    .to_string(),
            }
        }
        Provenance::Search { .. } => "No feature index: a descriptor search placed it".to_string(),
        Provenance::Sweep => "No feature index: the view sweep placed it".to_string(),
        Provenance::Pixel => "No feature index: it was placed by hand".to_string(),
        Provenance::Point { point } => {
            format!("No feature index: it was taken from point {point}")
        }
    };
    Some(format!(
        "\u{2022} At pixel ({:.1}, {:.1})\n\u{2022} {feature}",
        pixel[0], pixel[1]
    ))
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
