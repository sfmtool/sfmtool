// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! An opt-in [Model Context Protocol][mcp] control surface for the running
//! viewer, so an agent can drive the window the human is looking at.
//!
//! See `specs/gui/mcp-server.md`. Started with `sfm-explorer --mcp`, the viewer
//! hosts a small HTTP server on loopback; a connected agent can enumerate the
//! scene graph, open and close `.sfmr` files, move the selection and the 3D
//! camera, choose what the Image Detail panel draws over its photograph, edit a
//! loaded reconstruction and walk its history, save it, and take a screenshot
//! of the viewport.
//!
//! ## The shape of this module, and why
//!
//! Application state stays single-threaded on the GUI thread. `AppState`,
//! [`Viewer3D`] and the renderer are threaded through the frame as `&mut`, and
//! the panels rely on that — [`crate::dock::TabContext`] hands out seven
//! simultaneous `&mut` borrows, and the SIFT and full-res caches are
//! split-borrowed against the scene on purpose. So the server never touches app
//! state: it builds a [`Command`], hands it to the GUI thread over a channel,
//! wakes the event loop, and waits for the answer.
//!
//! That gives the module three layers, in dependency order:
//!
//! - [`tools`] — the tool table and the wire parse. Names, descriptions,
//!   `inputSchema`, and JSON arguments to [`Command`].
//! - [`apply_with_window`] and [`render`] — the whole command vocabulary, applied to
//!   `(&mut AppState, &mut Viewer3D)` and a [`crate::window::WindowHost`].
//!   **No `App`, no GPU handle**, which is what keeps every tool but
//!   `screenshot` under headless test.
//! - [`server`] — the `rmcp` handler and the `axum`/`tokio` plumbing that
//!   carries a [`Request`] to the GUI thread and its [`Reply`] back.
//!
//! [mcp]: https://modelcontextprotocol.io/

use std::path::PathBuf;

use serde_json::Value;

use crate::scene::{CameraRef, ImageRef, PointRef, ReconId};
use crate::state::AppState;
use crate::viewer_3d::Viewer3D;

mod bench;
mod display;
mod edit;
mod frame;
mod layout;
mod logged;
mod panel_rect;
mod read;
pub(crate) mod server;
pub(crate) mod tools;
mod view;
pub(crate) mod window;
mod write;

pub(crate) mod render;

#[cfg(test)]
mod tests;

pub(crate) use server::serve;

use logged::{query_text, screenshot_size};
pub(crate) use panel_rect::panel_body_points;
use panel_rect::panel_crop;

/// Everything the MCP surface can ask the viewer to do. One variant per tool.
///
/// A reconstruction is named by its **label**, so these carry a `String` that
/// [`apply_with_window`] resolves against `AppState::scene`. `Option<String>` means "the
/// selected reconstruction if omitted"; the editing tools take a plain
/// `String`, because an edit names the node it edits rather than landing on
/// whatever the human last clicked. The `ReconId` never crosses the wire:
/// a label is unique across the scene and survives every edit of the node it
/// names (see "Addressing" in `specs/gui/mcp-server.md`).
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Command {
    GetScene,
    ListCameraImages {
        reconstruction_label: Option<String>,
        offset: usize,
        limit: usize,
    },
    GetCameraImage {
        reconstruction_label: Option<String>,
        camera_image: CameraImageSel,
    },
    GetCameraIntrinsics {
        reconstruction_label: Option<String>,
        camera_intrinsics_index: usize,
    },
    GetPoint {
        point: crate::goto_point::PointQuery,
    },
    /// The Action Log from a revision onward: what happened while the agent
    /// was not looking, whoever did it.
    ///
    /// `actors` is never empty — the parse refuses `[]`, since a call that can
    /// return nothing by construction has asked no question, and fills an
    /// omitted field with every actor.
    GetActionLog {
        since_revision: u64,
        limit: usize,
        actors: Vec<crate::action_log::Actor>,
        /// Whether each row carries the breakdown of where its time went.
        ///
        /// What *was* recorded, which is a different question from the level
        /// [`Command::SetTimingDetail`] sets: an entry keeps the detail it was
        /// recorded with, so asking for it here re-times nothing.
        detail: bool,
    },
    OpenReconstruction {
        path: PathBuf,
    },
    CloseReconstruction {
        target: CloseTarget,
    },
    SelectReconstruction {
        reconstruction_label: String,
    },
    SelectCameraImage {
        reconstruction_label: Option<String>,
        camera_image: CameraImageSel,
    },
    SelectCameraIntrinsics {
        reconstruction_label: Option<String>,
        camera_intrinsics_index: usize,
    },
    SelectPoint {
        point: crate::goto_point::PointQuery,
    },
    ClearSelection {
        scope: SelectionScope,
    },
    SetReconstructionDisplay {
        reconstruction_label: String,
        change: DisplayChange,
    },
    /// Set one node's display transform outright, as the three pieces of a
    /// similarity the wire carries it in.
    SetReconstructionTransform {
        reconstruction_label: String,
        rotation_wxyz: [f64; 4],
        translation: [f64; 3],
        scale: f64,
    },
    /// Set it from the bench's active patch, in one of the four ways the
    /// viewport's patch menu offers.
    SetReconstructionTransformFromPatch {
        reconstruction_label: String,
        mode: crate::display_transform::PatchReframe,
    },
    /// Write the display transform into the node's data, as one version.
    BakeReconstructionTransform {
        reconstruction_label: String,
    },
    SetSolo {
        reconstruction_label: Option<String>,
    },
    GetImageDetailDisplay,
    /// Every field an `Option`, `None` meaning "leave it alone".
    ///
    /// The parse has already resolved the mode name, checked the two ladders
    /// and the size bounds, so [`apply_with_window`] only writes and records —
    /// which is what makes a refusal atomic without a rollback.
    SetImageDetailDisplay {
        change: ImageDetailDisplayChange,
    },
    /// Where the Image Detail panel is looking: the photograph, the zoom, and
    /// the rectangle of it on screen.
    GetImageDetailView,
    /// Point the Image Detail panel at one place in one photograph.
    ///
    /// The 2D counterpart of [`Command::SetView`], and one target per call for
    /// the same reason: "centre this pixel" and "fit this rectangle" are
    /// different questions, and a call carrying both would have no answer.
    SetImageDetailView {
        request: ImageDetailViewRequest,
    },
    /// Whether the operations to come record their finer stages.
    GetTimingDetail,
    /// The same switch the Action Log toolbar's **Detailed timing** checkbox
    /// throws, and through the same call, so the level cannot be raised
    /// without the window saying who raised it.
    ///
    /// This decides what gets *recorded*; `GetActionLog`'s `detail` asks for
    /// what was.
    SetTimingDetail {
        enabled: bool,
    },
    SetView {
        view: ViewCommand,
    },
    GetWindowLayout,
    SetWindowLayout {
        /// The document as it arrived, unparsed, so that one the viewer will
        /// not accept is a *domain* error in the layout parser's own words —
        /// path and all — rather than a malformed request.
        document: Value,
    },
    ShowPanel {
        panel: crate::dock::Tab,
    },
    HidePanel {
        panel: crate::dock::Tab,
    },
    /// One node's version list, its cursor, and what a save would find.
    GetHistory {
        reconstruction_label: String,
    },
    Undo {
        reconstruction_label: String,
    },
    Redo {
        reconstruction_label: String,
    },
    /// The Edit History panel's jump, by the serial the panel and the log
    /// spell (`"v12"`).
    JumpToVersion {
        reconstruction_label: String,
        serial: String,
    },
    /// Write the node out: over its own path when the call named none, and to
    /// a named path otherwise, which re-points the node at it.
    SaveReconstruction {
        reconstruction_label: String,
        path: Option<PathBuf>,
        /// Write a minimal copy to `path` instead, leaving the node as it is.
        minimal: bool,
        /// State the `workspace.relative_path` the file records, rather than
        /// measuring it; needs `path`.
        workspace_path: Option<String>,
    },
    DeletePoint {
        reconstruction_label: String,
        point: crate::goto_point::PointQuery,
    },
    /// Re-solve one point from its own observations, at the poses and the lens
    /// the reconstruction already holds.
    RetriangulatePoint {
        reconstruction_label: String,
        point: crate::goto_point::PointQuery,
    },
    /// Re-solve every point of one node the same way.
    RetriangulateAllPoints {
        reconstruction_label: String,
    },
    /// Retire every observation of one node a finer tracked one covers.
    PruneCoveredObservations {
        reconstruction_label: String,
        options: sfmtool_core::reconstruction::prune_covered::PruneCoveredOptions,
    },
    DeleteCameraImage {
        reconstruction_label: String,
        camera_image: CameraImageSel,
    },
    /// Put one camera image at a pose, as one version of its reconstruction.
    ///
    /// The pose is world-from-camera in the reconstruction's own frame, in the
    /// pieces the wire carries them: a rotation quaternion and a camera centre.
    MoveCameraImage {
        reconstruction_label: String,
        camera_image: CameraImageSel,
        quaternion_wxyz: [f64; 4],
        translation: [f64; 3],
    },
    ResectCameraImage {
        reconstruction_label: String,
        camera_image: CameraImageSel,
        from_matches: bool,
    },
    BundleAdjust {
        reconstruction_label: String,
        release_focal: bool,
    },
    /// Convert one node's observations from `sift_files` to
    /// `embedded_patches`, then render bitmaps from readable photographs
    /// without photometric adaptation.
    ConvertToEmbeddedPatches {
        reconstruction_label: String,
    },
    /// One node's bench: the items, their kinds, origins, stages and counts,
    /// and which is active.
    GetBench {
        reconstruction_label: String,
    },
    /// One track on it, with every observation's provenance, verdict and
    /// measurements.
    GetBenchTrack {
        reconstruction_label: String,
        /// `None` is the active track, which is what a bench panel's gesture
        /// means when it names no item.
        track: Option<String>,
    },
    /// Start a cluster-stage track from a place in one camera image.
    CreateBenchCluster {
        reconstruction_label: String,
        camera_image: CameraImageSel,
        seed: crate::bench::Seed,
    },
    /// Put a point of the reconstruction on the bench as a track-stage track.
    CreateBenchTrack {
        reconstruction_label: String,
        point: crate::goto_point::PointQuery,
    },
    ActivateBenchItem {
        reconstruction_label: String,
        item: String,
    },
    /// Leave every item on the bench and make none active: Track View's *Edit*
    /// box cleared.
    DeactivateBenchItem {
        reconstruction_label: String,
    },
    RenameBenchItem {
        reconstruction_label: String,
        item: String,
        /// The label it should take.
        label: String,
    },
    DiscardBenchItem {
        reconstruction_label: String,
        item: String,
    },
    /// Put a copy of one item on the bench beside it. A copy has no origin, so
    /// a commit of it creates a point rather than replacing one.
    DuplicateBenchItem {
        reconstruction_label: String,
        /// Omitted means the active track.
        item: Option<String>,
    },
    AddBenchTrackObservation {
        reconstruction_label: String,
        track: Option<String>,
        camera_image: CameraImageSel,
        seed: crate::bench::Seed,
    },
    /// Move the track-stage patch, by a displacement on its own axes or to a
    /// pixel of one photograph. Every sighting follows.
    TranslateBenchPatch {
        reconstruction_label: String,
        track: Option<String>,
        /// Exactly one of the two ways to name where it goes.
        to: TranslateTarget,
    },
    /// Resize the track-stage patch, by a world half-length or to a pixel of
    /// one photograph.
    ResizeBenchPatch {
        reconstruction_label: String,
        track: Option<String>,
        /// Exactly one of the two ways to name the size.
        to: ResizeTarget,
    },
    /// Put one edge of a cluster sighting's parallelogram under a pixel, with
    /// the opposite edge left where it is.
    ResizeBenchShape {
        reconstruction_label: String,
        track: Option<String>,
        /// The observation whose parallelogram is being dragged.
        observation: usize,
        /// Which edge of its square.
        edge: sfmtool_core::bench::Edge,
        /// Where its midpoint should land, in that image's own px.
        pixel: [f64; 2],
    },
    /// Turn the track-stage patch about its own outward normal.
    SpinBenchPatch {
        reconstruction_label: String,
        track: Option<String>,
        /// How far, positive about the patch's outward normal.
        degrees: f64,
    },
    /// Turn one cluster sighting's parallelogram in its own image's pixels.
    SpinBenchShape {
        reconstruction_label: String,
        track: Option<String>,
        /// The observation whose shape turns.
        observation: usize,
        /// How far, positive from `+x` toward `+y` of the image raster.
        degrees: f64,
    },
    /// Turn the track-stage patch to face a new outward normal. Every
    /// sighting is rebuilt on the turned axes.
    TiltBenchPatch {
        reconstruction_label: String,
        track: Option<String>,
        /// The outward normal wanted, in the reconstruction's own coordinates.
        normal: [f64; 3],
    },
    /// Put one observation's own sighting at a pixel, by hand.
    SightBenchObservation {
        reconstruction_label: String,
        track: Option<String>,
        /// The observation's position in the track's list.
        observation: usize,
        /// Where, in that observation's own image's px.
        pixel: [f64; 2],
    },
    /// Give one cluster sighting its affine shape outright.
    ShapeBenchObservation {
        reconstruction_label: String,
        track: Option<String>,
        /// The observation's position in the track's list.
        observation: usize,
        /// Keypoint-frame units to that image's pixels.
        shape: [[f64; 2]; 2],
    },
    SetBenchTrackVerdict {
        reconstruction_label: String,
        track: Option<String>,
        /// The observation's position in the track's list, which is stable for
        /// the life of the track.
        observation: usize,
        verdict: sfmtool_core::bench::Verdict,
    },
    /// Set the track's bars and paint the proposed verdicts onto its unpinned
    /// observations, which is the one gesture the panel's button is.
    ApplyBenchTrackThresholds {
        reconstruction_label: String,
        track: Option<String>,
        thresholds: ThresholdChange,
    },
    /// Move the named observations onto a second track beside this one.
    SplitBenchTrack {
        reconstruction_label: String,
        track: Option<String>,
        observations: Vec<usize>,
    },
    /// Write the track into the node's reconstruction.
    CommitBenchTrack {
        reconstruction_label: String,
        track: Option<String>,
    },
    /// Read every observation at the stage the track is in, on a worker,
    /// moving nothing.
    EvaluateBenchTrack {
        reconstruction_label: String,
        track: Option<String>,
        /// How far around each observation the correlation peak is looked for,
        /// in patch-grid px, or `None` for the reading own default.
        search_px: Option<f64>,
    },
    /// Fit the track at the stage it is in, on a worker: the step that moves
    /// it, and which ends by reading its own result.
    FitBenchTrack {
        reconstruction_label: String,
        track: Option<String>,
        /// The search radius the reading a fit ends with runs at.
        search_px: Option<f64>,
    },
    /// Move the track between its two representations, on a worker.
    SetBenchTrackStage {
        reconstruction_label: String,
        track: Option<String>,
        stage: sfmtool_core::bench::StageKind,
    },
    /// Ask the node's SIFT index which other photographs hold the patch
    /// around one observation, and add each as a candidate, on a worker.
    SearchBenchTrackDescriptors {
        reconstruction_label: String,
        track: Option<String>,
        /// Which observation to search from, by its position in
        /// `get_bench_track`'s list.
        observation: usize,
        /// The constellation's radius in the searched image's own pixels, or
        /// `None` for the radius that holds about fifty of its keypoints.
        radius_px: Option<f64>,
        /// Fewest agreeing correspondences an image needs, or `None` for the
        /// query's own bar.
        min_inliers: Option<usize>,
    },
    /// Project the track's patch into every camera of the node and add each
    /// photometrically admitted photograph as a candidate, on a worker.
    SearchBenchTrackGeometry {
        reconstruction_label: String,
        track: Option<String>,
        /// Which observation supplies the reference appearance, by its
        /// position in `get_bench_track`'s list.
        observation: usize,
    },
    /// Adopt a `.kdf` as the node's SIFT index.
    OpenSiftIndex {
        reconstruction_label: String,
        /// The file, or `None` for the node's own index path beside its
        /// `.sfmr`.
        path: Option<String>,
    },
    /// Build a SIFT index over the node's `.sift` files, on a worker.
    BuildSiftIndex {
        reconstruction_label: String,
        /// Where to write it, or `None` for the node's own index path.
        path: Option<String>,
    },
    /// Let go of the node's open SIFT index.
    CloseSiftIndex {
        reconstruction_label: String,
    },
    /// What the background operation is doing, or what the last one did.
    ///
    /// Names no operation, for the reason [`Command::CancelBackgroundTask`] does
    /// not: one runs at a time, viewer-wide.
    GetBackgroundTask,
    /// Ask the running background operation to stop.
    ///
    /// Names no operation: one runs at a time, viewer-wide, so "the one that is
    /// running" is unambiguous.
    CancelBackgroundTask,
    /// A picture of the presented window, or of one panel's body cropped from
    /// it.
    ///
    /// `hud: false` is only reachable with `panel: Some(Tab::Viewer3D)`; the
    /// parse refuses it anywhere else, because no other panel has a picture
    /// underneath what is drawn on it.
    Screenshot {
        panel: Option<crate::dock::Tab>,
        hud: bool,
        max_dimension: Option<u32>,
    },
}

/// How a tool named a camera image: by its index in the reconstruction, or by
/// the `.sfmr` relative path that is its name.
///
/// Both, because the surface hands out both — a track observation reports an
/// index, a `list_camera_images` row reports both — and an agent arrives
/// holding whichever it last read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum CameraImageSel {
    Index(usize),
    Name(String),
}

/// What `close_reconstruction` was asked to close.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum CloseTarget {
    One(String),
    All,
}

/// How much of the selection `clear_selection` drops.
///
/// Follows the viewer's own rule that dismissing a photograph says nothing
/// about the lens: [`SelectionScope::CameraImage`] leaves the intrinsics
/// selected, exactly as `AppState::select_image(None)` does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SelectionScope {
    All,
    CameraImage,
    CameraIntrinsics,
    Point,
}

/// A `set_reconstruction_display` request: every field is one of `SceneNode`'s
/// own, and every `None` is left alone.
///
/// `tint` is doubly optional on purpose. The outer `None` is "the call did not
/// mention the tint"; the inner `None` is "clear it back to the node's own
/// colors", which is what a JSON `null` asks for.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct DisplayChange {
    pub(crate) visible: Option<bool>,
    pub(crate) interactive: Option<bool>,
    pub(crate) show_points: Option<bool>,
    pub(crate) show_camera_images: Option<bool>,
    pub(crate) show_patches: Option<bool>,
    pub(crate) show_points_at_infinity: Option<bool>,
    pub(crate) tint: Option<Option<String>>,
}

/// A `set_image_detail_display` request: the Image Detail panel's controls,
/// with every field a call did not name left alone.
///
/// The feature overlay and its filters at the top level and the intrinsics
/// layer in a sub-struct, which is the shape of the document on the wire and
/// the shape of the toolbar it drives.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct ImageDetailDisplayChange {
    pub(crate) overlay_mode: Option<crate::state::OverlayMode>,
    /// Doubly optional: the outer `None` is "the call did not mention the
    /// cap", the inner one is "lift it", which is what a JSON `null` asks for.
    pub(crate) max_features: Option<Option<usize>>,
    pub(crate) feature_size_px: Option<display::FeatureSize>,
    pub(crate) tracked_only: Option<bool>,
    pub(crate) intrinsics: IntrinsicsChange,
}

/// A `set_image_detail_view` request: which photograph to look at, the one
/// thing to look at in it, and how close.
///
/// The photograph is doubly implied on purpose. A call that names one selects
/// it; a call that names none looks in the one already selected -- except where
/// the target itself names a photograph, which `bench_observation` does, and
/// then that is the one selected. So an agent walking a bench track's
/// observations names only the observation.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ImageDetailViewRequest {
    pub(crate) reconstruction_label: Option<String>,
    pub(crate) camera_image: Option<CameraImageSel>,
    pub(crate) target: ImageDetailTarget,
    /// The magnification to look at the target at, where the target takes one.
    /// Absolute, with 1.0 the fit; clamped to the panel's range, and the reply
    /// says where it landed.
    pub(crate) zoom: Option<f32>,
}

/// The one thing a `set_image_detail_view` call asks to look at.
///
/// An enum rather than optional fields, as [`ViewCommand`] is: these are
/// intents with different arithmetic behind them, and a call carrying two of
/// them has asked two questions.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum ImageDetailTarget {
    /// A place in the photograph's own pixels, brought to the panel centre.
    Pixel([f32; 2]),
    /// A rectangle of the photograph, `[x0, y0, x1, y1]` in its own pixels,
    /// fitted to the panel. The one target that settles the zoom itself.
    Rect([f32; 4]),
    /// A 3D point, at its observation in the photograph being looked at.
    Point(crate::goto_point::PointQuery),
    /// A `.sift` feature of that photograph, by its index in the file.
    Feature(u32),
    /// One observation of a bench track, which names its own photograph.
    BenchObservation {
        /// `None` is the active track, as everywhere else on the bench.
        track: Option<String>,
        /// Its position in the track's observation list.
        observation: usize,
    },
    /// The whole photograph: zoom 1, centred.
    Fit,
}

/// The `intrinsics` sub-block of an [`ImageDetailDisplayChange`].
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct IntrinsicsChange {
    pub(crate) enabled: Option<bool>,
    pub(crate) axes: Option<bool>,
    pub(crate) rings: Option<bool>,
    pub(crate) distortion: Option<bool>,
    /// Doubly optional, as `max_features` is: an explicit `null` is "back to
    /// the automatic scale".
    pub(crate) distortion_scale: Option<Option<f32>>,
    pub(crate) grid_cols: Option<usize>,
}

/// An `apply_bench_track_thresholds` request: the bars the painting judges an
/// observation against, with every bar a call did not name left where the track
/// has it.
///
/// Left where the track has it rather than reset to the default, because the
/// bars are the track's own state and a call that moves one bar has said
/// nothing about the other three. The panel's sliders are the same statement
/// made with a hand, and they start from the same place.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct ThresholdChange {
    pub(crate) min_zncc: Option<f64>,
    pub(crate) max_shift_px: Option<f64>,
    pub(crate) max_keypoint_uncertainty: Option<f64>,
    pub(crate) min_relative_zncc: Option<f64>,
}

impl ThresholdChange {
    /// `thresholds` with the bars this call named moved.
    pub(super) fn applied_to(
        &self,
        thresholds: &sfmtool_core::bench::Thresholds,
    ) -> sfmtool_core::bench::Thresholds {
        let mut next = thresholds.clone();
        if let Some(value) = self.min_zncc {
            next.min_zncc = value;
        }
        if let Some(value) = self.max_shift_px {
            next.max_shift_px = value;
        }
        if let Some(value) = self.max_keypoint_uncertainty {
            next.max_keypoint_uncertainty = value;
        }
        if let Some(value) = self.min_relative_zncc {
            next.min_relative_zncc = value;
        }
        next
    }
}

/// How `translate_bench_patch` named where the patch goes.
///
/// One enum and not a bag of optional fields, because these are two *intents*
/// and not two spellings of one: a displacement on the patch's own axes is a
/// statement out in the world, where the normal part is a statement about depth
/// no sighting could make, and a pixel is a statement in one photograph. A
/// call carrying both would have no answer, so the wire takes exactly one.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum TranslateTarget {
    /// `[u, v, n]` on the patch's own orthonormal axes, in world units.
    By([f64; 3]),
    /// A pixel of one photograph, which the centre of the square drawn there
    /// lands under.
    Pixel {
        /// The photograph the pixel is in, and so the square it is read
        /// against.
        viewpoint: ViewpointSel,
        /// Where, in that image's own px.
        pixel: [f64; 2],
    },
}

/// Which photograph a pixel form names its pixel in: the wire's spelling of
/// `sfmtool_core::bench::Viewpoint`.
///
/// Either an observation, whose image shows the patch re-anchored on its
/// keypoint, or a camera image in either of its two spellings, which shows the
/// patch as it stands: the ghost outline's square in an image the track has no
/// sighting in. The camera image is resolved against the node when the call
/// runs, as every other `camera_image` is.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum ViewpointSel {
    /// An observation of the track, by its position in the track's list.
    Observation(usize),
    /// A camera image of the reconstruction.
    CameraImage(CameraImageSel),
}

/// How `resize_bench_patch` named the size, for the reason
/// [`TranslateTarget`] is an enum.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum ResizeTarget {
    /// A world half-length, with the edge that moves or `None` for both about a
    /// held centre.
    HalfLength {
        /// The new half-length, in the reconstruction's own units.
        half_length: f64,
        /// Which edge moves, the far one held.
        moved_edge: Option<sfmtool_core::bench::Edge>,
    },
    /// One edge of the outline drawn in a photograph, put under a pixel.
    Pixel {
        /// The photograph whose outline is being dragged.
        viewpoint: ViewpointSel,
        /// Which edge of the patch's square.
        edge: sfmtool_core::bench::Edge,
        /// Where its midpoint should land, in that image's own px.
        pixel: [f64; 2],
    },
}

/// The five things `set_view` can be asked for.
///
/// One enum rather than a bag of optional fields, because these are *intents*
/// and not representations: "frame the scene" and "put the camera exactly
/// here" are different questions, and a call that carried both would have no
/// answer.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum ViewCommand {
    /// Frame everything drawn, or one named reconstruction.
    Fit {
        reconstruction_label: Option<String>,
    },
    /// Look through a camera image, as double-click and `Z` do.
    LookThrough {
        reconstruction_label: Option<String>,
        camera_image: CameraImageSel,
    },
    /// Leave camera-view mode, keeping the camera where it is.
    ExitCameraView,
    /// The explicit camera, in whatever pieces the call carried.
    Place(Placement),
    /// The field of view alone.
    Fov { fov_short_axis_deg: f64 },
}

/// The explicit camera -- a position, an orientation and a target distance --
/// as the pieces one call carried.
///
/// Every field is optional and every absent one is preserved from the standing
/// view, which is what makes the look-at form (`position` with `target`), the
/// exact form (`orientation_wxyz` with its companions) and a single piece on
/// its own the same command with different fields filled in, rather than three
/// commands that would each need their own placement arithmetic.
///
/// Which combinations determine a camera at all is settled in
/// [`tools::parse`], which sees what the call named; what a combination means
/// is settled in `view::set_view`, which sees the view it is changing.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct Placement {
    pub(crate) position: Option<[f64; 3]>,
    pub(crate) target: Option<[f64; 3]>,
    pub(crate) forward: Option<[f64; 3]>,
    pub(crate) orientation_wxyz: Option<[f64; 4]>,
    pub(crate) target_distance: Option<f64>,
    pub(crate) up: Option<[f64; 3]>,
    pub(crate) world_up: Option<[f64; 3]>,
    pub(crate) fov_short_axis_deg: Option<f64>,
}

/// A tool's refusal: everything the viewer can say no to, in the style its own
/// status line uses — what was asked, and what is actually there.
///
/// Distinct from a protocol error. This becomes a `CallToolResult` with
/// `isError: true`, which tells the client the request was well-formed and the
/// *viewer* declined; a malformed request never reaches [`apply_with_window`] at all.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ToolError(pub(crate) String);

impl ToolError {
    pub(crate) fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl std::fmt::Display for ToolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// What a tool produced.
///
/// Two shapes rather than one, because `screenshot` answers with a picture and
/// the other fifty-five answer with JSON, and squeezing an image through a JSON
/// field would mean a magic key that the transport has to know to look for.
pub(crate) enum ToolOutput {
    Json(Value),
    Png {
        bytes: Vec<u8>,
        width: u32,
        height: u32,
        /// One line describing what was in frame, for the text block that
        /// accompanies the image.
        caption: String,
    },
}

/// A tool's answer: what it produced, or a message for `isError: true`.
pub(crate) type Reply = Result<ToolOutput, ToolError>;

/// The answer of the fifty-five tools that speak only JSON.
///
/// Widened to a [`Reply`] at the [`apply_with_window`] dispatch, so nothing below it has to
/// name the shape it is not.
pub(super) type JsonReply = Result<Value, ToolError>;

/// Whether [`apply_with_window`] finished the job, or needs the frame to complete first.
pub(crate) enum Outcome {
    Done(Reply),
    Deferred(Deferred),
}

/// A command whose answer cannot exist yet.
///
/// Three tools are in here, waiting on different things. `App` holds them until
/// the readback phase and answers them there, after the frame they were waiting
/// on has been drawn and presented -- which is also where the `wgpu::Device`
/// already is, and so what keeps [`apply_with_window`] free of a GPU handle.
pub(crate) enum Deferred {
    Screenshot {
        /// Which pixels to read once the frame has been presented.
        source: ScreenshotSource,
        max_dimension: Option<u32>,
        /// The one-line description of what is in frame, built while the state
        /// was still borrowed. Held here so the readback phase does not have to
        /// reach back into `AppState` to describe a picture it already took.
        caption: String,
    },
    /// A background operation this call started, whose answer is either its
    /// result or a handle, whichever the clock reaches first.
    Background(BackgroundReply),
    /// A view request the Image Detail panel has not drawn yet, whose answer is
    /// the view the frame that draws it settles on.
    ImageDetailView(PendingView),
}

/// A `set_image_detail_view` call waiting for the panel to draw the photograph
/// it named.
///
/// The look is already standing on `AppState` and the panel applies it on the
/// frame it draws that photograph; what is waited for is the *reading*, since
/// the panel is the only thing that knows how big its body is and therefore
/// what fit means in it. As with [`BackgroundReply`], nothing blocks and no
/// frame is held: each frame asks [`display::pending_view_reply`] whether the
/// answer is there yet.
pub(crate) struct PendingView {
    /// The photograph whose frame the reply is waiting on.
    pub(crate) image: ImageRef,
    /// When the call was made, which the deadline is measured from.
    pub(crate) started: std::time::Instant,
    /// The publication serial the panel stood at when the call was made.
    ///
    /// The reply is owed a reading **newer** than this. "The panel has drawn
    /// that photograph" is not enough on its own: it may have drawn it before
    /// the call and before the layout moved, and answering from that reading is
    /// how a reply comes to report a panel size the next frame contradicts.
    pub(crate) after: u64,
}

/// A tool call waiting on the operation it started.
///
/// The wait is not a wait: nothing blocks and no frame is held. Each frame asks
/// [`edit::background_reply`] whether this can be answered yet, which it can as
/// soon as the operation finishes or as soon as
/// [`REPLY_DIRECTLY_WITHIN`] has passed, whichever comes first.
pub(crate) struct BackgroundReply {
    /// Which operation this call started, so a poll can still tell it from the
    /// one that replaced it.
    pub(crate) operation_id: u64,
    /// What it is called, for the handle.
    pub(crate) operation_name: &'static str,
    /// What the call answers with once the operation has landed.
    pub(crate) answer: Answer,
    /// The label of the node it runs on, or the one an open's file suggests,
    /// for the handle.
    pub(crate) label: String,
    /// When the call started it, which the window below is measured from.
    pub(crate) started: std::time::Instant,
}

/// What a call that started a background operation answers with once the
/// operation has landed, the handle aside.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Answer {
    /// The version the operation left on this node, as every edit answers.
    Version(ReconId),
    /// The reconstruction the open made, as `open_reconstruction` answers, with
    /// whether the path was already open when the call was made.
    Opened { already_open: bool },
}

/// How long a tool that starts a background operation waits for it before
/// answering with a handle instead of a result.
///
/// A threshold about **when a wait stops feeling immediate**, not about what
/// any particular reconstruction costs. Below roughly a tenth of a second a
/// response reads as instantaneous; up to about a second a caller stays in the
/// flow of what it was doing and simply sees the system working; past that,
/// attention wanders and the wait wants explaining. Two tenths sits just past
/// instantaneous and well short of anything anyone would call slow, so a handle
/// comes back only for an operation that genuinely is slow, and an operation
/// that answers normally has answered inside the window where nobody had begun
/// to wonder.
pub(crate) const REPLY_DIRECTLY_WITHIN: std::time::Duration = std::time::Duration::from_millis(200);

/// Which pixels a deferred screenshot reads.
///
/// The first two are the one presented surface texture, whole or cropped; the
/// third is the viewport's own render target, which is what `hud: false` asks
/// for — the same view with nothing egui painted over it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ScreenshotSource {
    /// The presented surface, whole: the window as the human sees it.
    Window,
    /// The presented surface, cropped to one panel's body.
    ///
    /// The rectangle is **not** resolved here. A panel a `show_panel` earlier
    /// in the same batch opened has no rectangle until this frame's egui pass
    /// has laid the dock out, so the tab travels and the frame reads
    /// `LeafNode::viewport` after the pass.
    Panel(crate::dock::Tab),
    /// The viewport's own `edl output` texture — `viewer_3d` with `hud: false`.
    ViewportRender,
}

/// One tool call in flight: what to do, and where the answer goes.
pub(crate) struct Request {
    pub(crate) command: Command,
    pub(crate) reply: tokio::sync::oneshot::Sender<Reply>,
}

/// Apply one command to the viewer, with no window behind it.
///
/// [`apply_with_window`] against a `test_support::NoWindow` host, which is the
/// windowless case: `get_window` and `set_window` refuse with "no window", and
/// every other tool behaves exactly as it does in the viewer. Kept as its own
/// function for the callers that have no window to offer and should not have to
/// invent one — which today is the tests, the real frame always having a window
/// by the time it drains, so this is compiled for them.
#[cfg(test)]
pub(crate) fn apply(state: &mut AppState, viewer: &mut Viewer3D, command: Command) -> Outcome {
    apply_with_window(state, viewer, &mut crate::test_support::NoWindow, command)
}

/// Apply one command to the viewer.
///
/// Takes no `App` and no GPU handle, which is what makes every tool but
/// `screenshot` testable in a headless `cargo test`: `App` owns a
/// `wgpu::Device`, a surface and a window, and constructing one needs a GPU and
/// a display that this crate's lib tests deliberately do without. The one
/// GPU-shaped command leaves through [`Outcome::Deferred`] instead, and the one
/// window-shaped command through [`crate::window::WindowHost`], which a fake can stand
/// in for.
pub(crate) fn apply_with_window(
    state: &mut AppState,
    viewer: &mut Viewer3D,
    host: &mut dyn crate::window::WindowHost,
    command: Command,
) -> Outcome {
    match command {
        Command::GetScene => done(Ok(render::scene(state, viewer))),
        Command::ListCameraImages {
            reconstruction_label,
            offset,
            limit,
        } => done(read::list_camera_images(
            state,
            reconstruction_label.as_deref(),
            offset,
            limit,
        )),
        Command::GetCameraImage {
            reconstruction_label,
            camera_image,
        } => done(read::get_camera_image(
            state,
            reconstruction_label.as_deref(),
            &camera_image,
        )),
        Command::GetCameraIntrinsics {
            reconstruction_label,
            camera_intrinsics_index,
        } => done(read::get_camera_intrinsics(
            state,
            reconstruction_label.as_deref(),
            camera_intrinsics_index,
        )),
        Command::GetPoint { point } => done(read::get_point(state, &point)),
        Command::GetActionLog {
            since_revision,
            limit,
            actors,
            detail,
        } => done(read::get_action_log(
            state,
            since_revision,
            limit,
            &actors,
            detail,
        )),
        Command::OpenReconstruction { path } => write::open_reconstruction(state, &path),
        Command::CloseReconstruction { target } => done(write::close_reconstruction(state, target)),
        Command::SelectReconstruction {
            reconstruction_label,
        } => done(write::select_reconstruction(state, &reconstruction_label)),
        Command::SelectCameraImage {
            reconstruction_label,
            camera_image,
        } => done(write::select_camera_image(
            state,
            reconstruction_label.as_deref(),
            &camera_image,
        )),
        Command::SelectCameraIntrinsics {
            reconstruction_label,
            camera_intrinsics_index,
        } => done(write::select_camera_intrinsics(
            state,
            reconstruction_label.as_deref(),
            camera_intrinsics_index,
        )),
        Command::SelectPoint { point } => done(write::select_point(state, &point)),
        Command::ClearSelection { scope } => done(write::clear_selection(state, scope)),
        Command::SetReconstructionDisplay {
            reconstruction_label,
            change,
        } => done(write::set_reconstruction_display(
            state,
            &reconstruction_label,
            change,
        )),
        Command::SetReconstructionTransform {
            reconstruction_label,
            rotation_wxyz,
            translation,
            scale,
        } => done(edit::set_reconstruction_transform(
            state,
            &reconstruction_label,
            rotation_wxyz,
            translation,
            scale,
        )),
        Command::SetReconstructionTransformFromPatch {
            reconstruction_label,
            mode,
        } => done(edit::set_reconstruction_transform_from_patch(
            state,
            &reconstruction_label,
            mode,
        )),
        Command::BakeReconstructionTransform {
            reconstruction_label,
        } => done(edit::bake_reconstruction_transform(
            state,
            &reconstruction_label,
        )),
        Command::SetSolo {
            reconstruction_label,
        } => done(write::set_solo(state, reconstruction_label.as_deref())),
        Command::GetImageDetailDisplay => done(display::get(state)),
        Command::SetImageDetailDisplay { change } => done(display::set(state, &change)),
        Command::GetImageDetailView => done(display::get_view(state)),
        Command::SetImageDetailView { request } => display::set_view(state, &request),
        Command::GetTimingDetail => done(display::get_timing_detail(state)),
        Command::SetTimingDetail { enabled } => done(display::set_timing_detail(state, enabled)),
        Command::SetView { view } => done(view::set_view(state, viewer, view)),
        Command::GetWindowLayout => done(layout::get_window_layout(state, host)),
        Command::SetWindowLayout { document } => {
            done(layout::set_window_layout(state, host, &document))
        }
        Command::ShowPanel { panel } => done(layout::show_panel(state, host, panel)),
        Command::HidePanel { panel } => done(layout::hide_panel(state, host, panel)),
        Command::GetHistory {
            reconstruction_label,
        } => done(edit::get_history(state, &reconstruction_label)),
        Command::Undo {
            reconstruction_label,
        } => {
            let reply = edit::undo(state, &reconstruction_label);
            moved_cursor(state, viewer, reply)
        }
        Command::Redo {
            reconstruction_label,
        } => {
            let reply = edit::redo(state, &reconstruction_label);
            moved_cursor(state, viewer, reply)
        }
        Command::JumpToVersion {
            reconstruction_label,
            serial,
        } => {
            let reply = edit::jump_to_version(state, &reconstruction_label, &serial);
            moved_cursor(state, viewer, reply)
        }
        Command::SaveReconstruction {
            reconstruction_label,
            path,
            minimal,
            workspace_path,
        } => done(edit::save_reconstruction(
            state,
            &reconstruction_label,
            path.as_deref(),
            minimal,
            workspace_path.as_deref(),
        )),
        Command::DeletePoint {
            reconstruction_label,
            point,
        } => done(edit::delete_point(state, &reconstruction_label, &point)),
        Command::RetriangulatePoint {
            reconstruction_label,
            point,
        } => done(edit::retriangulate_point(
            state,
            &reconstruction_label,
            &point,
        )),
        Command::RetriangulateAllPoints {
            reconstruction_label,
        } => edit::retriangulate_all_points(state, &reconstruction_label),
        Command::PruneCoveredObservations {
            reconstruction_label,
            options,
        } => edit::prune_covered_observations(state, &reconstruction_label, &options),
        Command::DeleteCameraImage {
            reconstruction_label,
            camera_image,
        } => done(edit::delete_camera_image(
            state,
            &reconstruction_label,
            &camera_image,
        )),
        Command::MoveCameraImage {
            reconstruction_label,
            camera_image,
            quaternion_wxyz,
            translation,
        } => done(edit::move_camera_image(
            state,
            &reconstruction_label,
            &camera_image,
            quaternion_wxyz,
            translation,
        )),
        Command::ResectCameraImage {
            reconstruction_label,
            camera_image,
            from_matches,
        } => done(edit::resect_camera_image(
            state,
            &reconstruction_label,
            &camera_image,
            from_matches,
        )),
        Command::BundleAdjust {
            reconstruction_label,
            release_focal,
        } => edit::bundle_adjust(state, &reconstruction_label, release_focal),
        Command::ConvertToEmbeddedPatches {
            reconstruction_label,
        } => edit::convert_to_embedded_patches(state, &reconstruction_label),
        Command::GetBench {
            reconstruction_label,
        } => done(bench::get_bench(state, &reconstruction_label)),
        Command::GetBenchTrack {
            reconstruction_label,
            track,
        } => done(bench::get_bench_track(
            state,
            &reconstruction_label,
            track.as_deref(),
        )),
        Command::CreateBenchCluster {
            reconstruction_label,
            camera_image,
            seed,
        } => done(bench::create_bench_cluster(
            state,
            &reconstruction_label,
            &camera_image,
            &seed,
        )),
        Command::CreateBenchTrack {
            reconstruction_label,
            point,
        } => done(bench::create_bench_track(
            state,
            &reconstruction_label,
            &point,
        )),
        Command::ActivateBenchItem {
            reconstruction_label,
            item,
        } => done(bench::activate_bench_item(
            state,
            &reconstruction_label,
            &item,
        )),
        Command::DeactivateBenchItem {
            reconstruction_label,
        } => done(bench::deactivate_bench_item(state, &reconstruction_label)),
        Command::RenameBenchItem {
            reconstruction_label,
            item,
            label,
        } => done(bench::rename_bench_item(
            state,
            &reconstruction_label,
            &item,
            &label,
        )),
        Command::DiscardBenchItem {
            reconstruction_label,
            item,
        } => done(bench::discard_bench_item(
            state,
            &reconstruction_label,
            &item,
        )),
        Command::DuplicateBenchItem {
            reconstruction_label,
            item,
        } => done(bench::duplicate_bench_item(
            state,
            &reconstruction_label,
            item.as_deref(),
        )),
        Command::AddBenchTrackObservation {
            reconstruction_label,
            track,
            camera_image,
            seed,
        } => done(bench::add_bench_track_observation(
            state,
            &reconstruction_label,
            track.as_deref(),
            &camera_image,
            &seed,
        )),
        Command::TranslateBenchPatch {
            reconstruction_label,
            track,
            to,
        } => done(bench::translate_bench_patch(
            state,
            &reconstruction_label,
            track.as_deref(),
            &to,
        )),
        Command::ResizeBenchPatch {
            reconstruction_label,
            track,
            to,
        } => done(bench::resize_bench_patch(
            state,
            &reconstruction_label,
            track.as_deref(),
            &to,
        )),
        Command::ResizeBenchShape {
            reconstruction_label,
            track,
            observation,
            edge,
            pixel,
        } => done(bench::resize_bench_shape(
            state,
            &reconstruction_label,
            track.as_deref(),
            observation,
            edge,
            pixel,
        )),
        Command::SpinBenchPatch {
            reconstruction_label,
            track,
            degrees,
        } => done(bench::spin_bench_patch(
            state,
            &reconstruction_label,
            track.as_deref(),
            degrees,
        )),
        Command::SpinBenchShape {
            reconstruction_label,
            track,
            observation,
            degrees,
        } => done(bench::spin_bench_shape(
            state,
            &reconstruction_label,
            track.as_deref(),
            observation,
            degrees,
        )),
        Command::TiltBenchPatch {
            reconstruction_label,
            track,
            normal,
        } => done(bench::tilt_bench_patch(
            state,
            &reconstruction_label,
            track.as_deref(),
            normal,
        )),
        Command::SightBenchObservation {
            reconstruction_label,
            track,
            observation,
            pixel,
        } => done(bench::sight_bench_observation(
            state,
            &reconstruction_label,
            track.as_deref(),
            observation,
            pixel,
        )),
        Command::ShapeBenchObservation {
            reconstruction_label,
            track,
            observation,
            shape,
        } => done(bench::shape_bench_observation(
            state,
            &reconstruction_label,
            track.as_deref(),
            observation,
            shape,
        )),
        Command::SetBenchTrackVerdict {
            reconstruction_label,
            track,
            observation,
            verdict,
        } => done(bench::set_bench_track_verdict(
            state,
            &reconstruction_label,
            track.as_deref(),
            observation,
            verdict,
        )),
        Command::ApplyBenchTrackThresholds {
            reconstruction_label,
            track,
            thresholds,
        } => done(bench::apply_bench_track_thresholds(
            state,
            &reconstruction_label,
            track.as_deref(),
            &thresholds,
        )),
        Command::SplitBenchTrack {
            reconstruction_label,
            track,
            observations,
        } => done(bench::split_bench_track(
            state,
            &reconstruction_label,
            track.as_deref(),
            &observations,
        )),
        Command::CommitBenchTrack {
            reconstruction_label,
            track,
        } => done(bench::commit_bench_track(
            state,
            &reconstruction_label,
            track.as_deref(),
        )),
        Command::EvaluateBenchTrack {
            reconstruction_label,
            track,
            search_px,
        } => bench::evaluate_bench_track(state, &reconstruction_label, track.as_deref(), search_px),
        Command::FitBenchTrack {
            reconstruction_label,
            track,
            search_px,
        } => bench::fit_bench_track(state, &reconstruction_label, track.as_deref(), search_px),
        Command::SetBenchTrackStage {
            reconstruction_label,
            track,
            stage,
        } => bench::set_bench_track_stage(state, &reconstruction_label, track.as_deref(), stage),
        Command::SearchBenchTrackDescriptors {
            reconstruction_label,
            track,
            observation,
            radius_px,
            min_inliers,
        } => bench::search_bench_track_descriptors(
            state,
            &reconstruction_label,
            track.as_deref(),
            observation,
            radius_px,
            min_inliers,
        ),
        Command::SearchBenchTrackGeometry {
            reconstruction_label,
            track,
            observation,
        } => bench::search_bench_track_geometry(
            state,
            &reconstruction_label,
            track.as_deref(),
            observation,
        ),
        Command::OpenSiftIndex {
            reconstruction_label,
            path,
        } => done(bench::open_sift_index(
            state,
            &reconstruction_label,
            path.as_deref(),
        )),
        Command::BuildSiftIndex {
            reconstruction_label,
            path,
        } => bench::build_sift_index(state, &reconstruction_label, path.as_deref()),
        Command::CloseSiftIndex {
            reconstruction_label,
        } => done(bench::close_sift_index(state, &reconstruction_label)),
        Command::GetBackgroundTask => done(read::get_background_task(state)),
        Command::CancelBackgroundTask => done(edit::cancel_background_task(state)),
        Command::Screenshot {
            panel,
            hud,
            max_dimension,
        } => {
            // A minimized window renders nothing to photograph, and whether its
            // swapchain still presents at all is platform-dependent — so this
            // is refused rather than attempted, naming the call that fixes it.
            // Checked here, against the window snapshot, so it is under
            // headless test with the rest of the vocabulary.
            if state.window.as_ref().map(|info| info.state)
                == Some(crate::window::WindowState::Minimized)
            {
                return done(Err(ToolError::new(MINIMIZED)));
            }
            // A panel that is not drawn cannot be photographed, and both ways
            // that happens have an answer the agent can act on. Checked against
            // the dock at *apply* time, so a `show_panel` earlier in the same
            // batch satisfies them — and headlessly, since the dock is a plain
            // field.
            if let Some(panel) = panel {
                if let Err(error) = drawn_panel(state, panel) {
                    return done(Err(error));
                }
            }
            let source = match (panel, hud) {
                (None, _) => ScreenshotSource::Window,
                (Some(crate::dock::Tab::Viewer3D), false) => ScreenshotSource::ViewportRender,
                (Some(panel), _) => ScreenshotSource::Panel(panel),
            };
            Outcome::Deferred(Deferred::Screenshot {
                source,
                max_dimension,
                caption: screenshot_caption(state, viewer, source),
            })
        }
    }
}

/// Whether `panel` is on screen to be photographed, or why it is not.
///
/// Closed and behind-a-sibling are different mistakes with different fixes, so
/// they get different messages — and a screenshot of a tab that is not in front
/// would be a picture of the tab that is.
fn drawn_panel(state: &AppState, panel: crate::dock::Tab) -> Result<(), ToolError> {
    let Some(path) = state.dock.find_tab(&panel) else {
        return Err(ToolError::new(format!(
            "The {} panel is closed, so there is nothing of it to photograph. Send show_panel \
             {{ \"panel_name\": \"{}\" }} first.",
            panel.title(),
            panel.wire_name(),
        )));
    };
    let leaf = state
        .dock
        .leaf(path.node_path())
        .map_err(|_| ToolError::new("The panel is no longer docked."))?;
    let front = leaf.tabs.get(leaf.active.0).copied();
    if front != Some(panel) {
        let in_front = front.map(|tab| tab.title()).unwrap_or("another tab");
        return Err(ToolError::new(format!(
            "The {} panel is behind {} in its node, so a picture of it would be a picture of {}. \
             Send show_panel {{ \"panel_name\": \"{}\" }} first.",
            panel.title(),
            in_front,
            in_front,
            panel.wire_name(),
        )));
    }
    Ok(())
}

/// Why a `screenshot` of a minimized window is refused.
///
/// A picture of a window the human cannot see answers nothing an agent asked
/// of a shared viewer, so the refusal names the call that makes one possible.
pub(super) const MINIMIZED: &str =
    "The window is minimized, so nothing is being rendered to photograph. Send \
     set_window_layout { \"window\": { \"state\": \"normal\" } } first.";

fn done(reply: JsonReply) -> Outcome {
    Outcome::Done(reply.map(ToolOutput::Json))
}

/// A cursor move's answer, with the viewport following the value it landed on.
///
/// A step of the cursor can change the pose of the very camera the viewport is
/// looking through, and camera view follows the value rather than remembering
/// where a hand left it -- so the re-snap the Edit menu's own Undo and Redo make
/// around these `AppState` calls is made here too, the viewport being
/// [`Viewer3D`]'s and not the state's. `camera_lock::resnap_camera_view` is a
/// no-op with no camera view open, and while a camera is held in hand.
///
/// Only where the move succeeded: a refusal landed on no version, and snapping
/// then would take a free-look offset away from the human for nothing.
fn moved_cursor(state: &mut AppState, viewer: &mut Viewer3D, reply: JsonReply) -> Outcome {
    if reply.is_ok() {
        crate::camera_lock::resnap_camera_view(viewer, state);
    }
    done(reply)
}

/// The sentence that rides along with a screenshot: what was photographed, at
/// what size, and — where the 3D view is in it — what was in frame.
///
/// Built here, while `AppState` is still borrowed, rather than at readback:
/// the picture and the description of it should be of the same instant, and
/// the readback phase runs after the UI has had a whole frame to change things.
/// The size is the panel's *last* laid-out size, which is the size the picture
/// comes back at in every frame but the one that opened the panel.
fn screenshot_caption(state: &AppState, viewer: &Viewer3D, source: ScreenshotSource) -> String {
    let [width, height] = screenshot_size(state, viewer, source, None);
    let subject = match source {
        ScreenshotSource::Window => format!("The window, {width}×{height}"),
        ScreenshotSource::ViewportRender => {
            format!("The 3D Viewer panel without its HUD, {width}×{height}")
        }
        ScreenshotSource::Panel(panel) => {
            format!("The {} panel, {width}×{height}", panel.title())
        }
    };
    // The frame description belongs to the pictures the 3D view is in: for
    // every other panel it would describe something the picture does not show.
    let shows_scene = matches!(
        source,
        ScreenshotSource::Window
            | ScreenshotSource::ViewportRender
            | ScreenshotSource::Panel(crate::dock::Tab::Viewer3D)
    );
    if !shows_scene {
        return format!("{subject}.");
    }
    format!("{subject}. {}", frame_description(state, viewer))
}

/// What the 3D viewport has in it: which reconstructions are drawn, how much of
/// them, and the camera image being looked through.
fn frame_description(state: &AppState, viewer: &Viewer3D) -> String {
    let stats = crate::scene::visible_stats(&state.scene, state.solo);
    let drawn: Vec<&str> = state
        .scene
        .iter()
        .filter(|node| crate::scene::is_visible(node, state.solo))
        .map(|node| node.label.as_str())
        .collect();
    let looking_through = viewer
        .camera_view
        .as_ref()
        .and_then(|camera_view| {
            state.node(camera_view.image.recon).and_then(|node| {
                node.recon()
                    .image_table
                    .images
                    .get(camera_view.image.index())
            })
        })
        .map(|image| format!(", looking through {}", image.name))
        .unwrap_or_default();
    format!(
        "In the 3D viewport: {} drawn ({}), {} points, {} camera images{}.",
        drawn.len(),
        if drawn.is_empty() {
            "nothing loaded".to_string()
        } else {
            drawn.join(", ")
        },
        stats.points,
        stats.images,
        looking_through,
    )
}

// ── Resolution: a wire handle to the thing it names ──────────────────────

/// The reconstruction a tool named, or the selected one when it named none.
///
/// The error names what *is* loaded, because the two ways to get here — a
/// typo, and a reconstruction the human closed in between — both want the same
/// list to recover from.
pub(super) fn resolve_reconstruction(
    state: &AppState,
    label: Option<&str>,
) -> Result<ReconId, ToolError> {
    let Some(label) = label else {
        return state.selected_recon.ok_or_else(|| {
            ToolError::new(
                "No reconstruction is selected — name one with reconstruction_label, or open \
                 a file first.",
            )
        });
    };
    state
        .scene
        .iter()
        .find(|node| node.label == label)
        .map(|node| node.id)
        .ok_or_else(|| {
            ToolError::new(format!(
                "No loaded reconstruction is labelled {label:?}{}",
                loaded_list(state)
            ))
        })
}

/// The camera image a tool named, as an index into `reconstruction`.
pub(super) fn resolve_camera_image(
    state: &AppState,
    reconstruction: ReconId,
    selector: &CameraImageSel,
) -> Result<ImageRef, ToolError> {
    let node = state
        .node(reconstruction)
        .ok_or_else(|| ToolError::new("The reconstruction is no longer loaded."))?;
    let index = match selector {
        CameraImageSel::Index(index) => {
            if *index >= node.recon().image_table.images.len() {
                return Err(ToolError::new(format!(
                    "{} has {} camera images — index {index} is out of range.",
                    node.label,
                    node.recon().image_table.images.len()
                )));
            }
            *index
        }
        CameraImageSel::Name(name) => node
            .recon()
            .image_table
            .images
            .iter()
            .position(|image| image.name == *name)
            .ok_or_else(|| {
                ToolError::new(format!(
                    "{} has no camera image named {name:?} — names are .sfmr relative paths, as \
                     in \"images/IMG_0042.jpg\"; list_camera_images reports them.",
                    node.label
                ))
            })?,
    };
    Ok(ImageRef::new(reconstruction, index))
}

/// The camera intrinsics record a tool named, as an index into
/// `reconstruction`.
pub(super) fn resolve_camera_intrinsics(
    state: &AppState,
    reconstruction: ReconId,
    index: usize,
) -> Result<CameraRef, ToolError> {
    let node = state
        .node(reconstruction)
        .ok_or_else(|| ToolError::new("The reconstruction is no longer loaded."))?;
    if index >= node.recon().image_table.cameras.len() {
        return Err(ToolError::new(format!(
            "{} has {} camera intrinsics records — index {index} is out of range.",
            node.label,
            node.recon().image_table.cameras.len()
        )));
    }
    Ok(CameraRef::new(reconstruction, index))
}

/// The 3D point a tool named.
///
/// Goes through the same parse and the same lookup the Go to Point dialog uses
/// ([`crate::goto_point`]), so a point id a human copied out of Track View
/// pastes straight into a tool call, and the two paths cannot disagree
/// about what an id means.
pub(super) fn resolve_point(
    state: &AppState,
    query: &crate::goto_point::PointQuery,
) -> Result<PointRef, ToolError> {
    crate::goto_point::resolve_point_query(&state.scene, state.selected_recon, query)
        .map_err(ToolError)
}

/// The 3D point a tool named, **inside** the reconstruction the same call
/// named.
///
/// The editing tools name their reconstruction and their point separately, and
/// the two have to agree: a bare index is a coordinate in that node's value,
/// and a qualified id that resolves somewhere else is a refusal rather than an
/// edit quietly applied to the wrong file. Goes through the same parse and
/// lookup [`resolve_point`] does, with the named node standing where the
/// selected one usually does, so an id means the same thing here as it does in
/// `get_point`.
pub(super) fn resolve_point_in(
    state: &AppState,
    reconstruction: ReconId,
    query: &crate::goto_point::PointQuery,
) -> Result<PointRef, ToolError> {
    let point = crate::goto_point::resolve_point_query(&state.scene, Some(reconstruction), query)
        .map_err(ToolError)?;
    if point.recon != reconstruction {
        let named = state.node(reconstruction).map(|node| node.label.as_str());
        let holder = state.node(point.recon).map(|node| node.label.as_str());
        return Err(ToolError::new(format!(
            "That point id belongs to {}, not to {}.",
            holder.unwrap_or("another reconstruction"),
            named.unwrap_or("the reconstruction named"),
        )));
    }
    Ok(point)
}

/// `" — loaded: a, b."`, or a note that nothing is, to hang off a
/// "no such reconstruction" message.
fn loaded_list(state: &AppState) -> String {
    if state.scene.is_empty() {
        return " — nothing is loaded.".to_string();
    }
    let labels: Vec<String> = state
        .scene
        .iter()
        .map(|node| format!("{:?}", node.label))
        .collect();
    format!(" — loaded: {}.", labels.join(", "))
}

/// Apply a frame's worth of commands **as the agent**, recording what they did.
///
/// The drain's application phase, with the channel left out so it is reachable
/// from a headless test. Three things happen here that [`apply_with_window`] cannot do for
/// itself:
///
/// - the Action Log's ambient actor is moved to
///   [`Mcp`](crate::action_log::Actor::Mcp) for the whole batch and restored
///   afterwards — one move per frame rather than an `Actor` argument on every
///   `AppState` method;
/// - a **read** is recorded, because it changes no state and so has no state
///   method to log through — and it is recorded from the command *before* the
///   command is applied, so a deferred `screenshot` lines up in order with the
///   commands around it rather than at readback;
/// - a **refusal** is recorded, in the same words the agent receives, because
///   the methods below return their failures rather than logging them.
///
/// A mutating tool that succeeds writes nothing here: the `AppState` and
/// `Viewer3D` methods it called already did, in the same words the GUI's own
/// path produces. Three of them word their own **refusal** as well -- the
/// resection, the adjustment and the camera move own the vocabulary of
/// the operation they refused -- so a failed entry is written here only when the
/// batch's application recorded none, which is what keeps one failure to one
/// entry.
///
/// A fourth thing happens before an **editing** command is applied: a camera the
/// human is holding on the node it names ([`crate::camera_lock`]) is ended, as a
/// commit when it has been moved. An edit landing under a held lock would leave
/// the reviewer holding a camera whose stored pose had moved beneath them, and
/// the lock is the viewport's rather than the state's -- which is why the step
/// is here, where the GUI thread applies the command, and not in the tool.
pub(crate) fn apply_as_agent(
    state: &mut AppState,
    viewer: &mut Viewer3D,
    host: &mut dyn crate::window::WindowHost,
    commands: Vec<Command>,
) -> Applied {
    debug_assert_eq!(
        state.action_log.actor(),
        crate::action_log::Actor::User,
        "the agent's commands were applied with the actor already moved",
    );
    state.action_log.set_actor(crate::action_log::Actor::Mcp);
    let mut stale: Vec<ReconId> = Vec::new();
    let outcomes = commands
        .into_iter()
        .map(|command| {
            log::debug!("MCP: applying {command:?}");
            let tool = command.tool_name();
            let kind = command.kind();
            let run = command.run();
            let query = query_text(state, viewer, &command);
            let renumbers = command.renumbers().map(str::to_string);
            // Before the revision is read, so that the sentence a committed
            // lock records belongs to the commit rather than to the edit that
            // displaced it -- and so that its refusal, if it has one, does not
            // stand in for the tool's own.
            let edits = command
                .edits()
                .and_then(|label| resolve_reconstruction(state, Some(label)).ok());
            stale.extend(
                edits.and_then(|id| crate::camera_lock::exit_implicitly_for(viewer, state, id)),
            );
            // A view command that leaves camera view (a fit, a look-through, a
            // placement, an explicit exit) is a step away from a camera in
            // hand, exactly as `,` and `.` are: the lock ends first, as a
            // commit when it has been moved. A field-of-view change keeps
            // camera view and so keeps the lock, as the zoom controls do.
            if matches!(command, Command::SetView { view: ref v } if !matches!(v, ViewCommand::Fov { .. }))
            {
                stale.extend(crate::camera_lock::exit_implicitly(viewer, state));
            }
            let before = state.action_log.revision();
            let outcome = apply_with_window(state, viewer, host, command);
            if matches!(outcome, Outcome::Done(Ok(_))) {
                stale.extend(
                    renumbers
                        .as_deref()
                        .and_then(|label| resolve_reconstruction(state, Some(label)).ok()),
                );
            }
            match (&outcome, query) {
                (Outcome::Done(Err(error)), _) if !recorded_a_failure(state, before) => state
                    .action_log
                    .fail(kind, format!("{tool} failed: {error}")),
                (Outcome::Done(Err(_)), _) => {}
                // A read that folds goes through `query`, which puts the
                // tool in both the kind and the run; `screenshot` has no run
                // and is recorded as the discrete act it is.
                (_, Some(text)) => match run {
                    Some(run) => state.action_log.query(run, text),
                    None => state.action_log.record(kind, text),
                },
                (_, None) => {}
            }
            outcome
        })
        .collect();
    state.action_log.set_actor(crate::action_log::Actor::User);
    Applied { outcomes, stale }
}

/// What a frame's batch of commands did.
pub(crate) struct Applied {
    /// One outcome per command, in the order they were applied.
    pub(crate) outcomes: Vec<Outcome>,
    /// The nodes a command renumbered ([`Command::renumbers`]), for the caller
    /// to drop what its panels cached about the table they had.
    pub(crate) stale: Vec<ReconId>,
}

/// Whether anything applied since revision `before` recorded a refusal of its
/// own.
///
/// The two bulk edits word their own refusals, because the vocabulary of "this
/// resection could not be attempted" belongs to the resection and not to the
/// tool that asked for it. The drain's `{tool} failed: …` row would be a second
/// line saying the same thing, so it is written only where the state wrote
/// none.
fn recorded_a_failure(state: &AppState, before: u64) -> bool {
    state.action_log.since(before).any(|entry| entry.failed)
}
