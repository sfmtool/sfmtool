// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The Action Log's view of MCP commands.

use crate::action_log::Kind;
use crate::state::AppState;
use crate::viewer_3d::Viewer3D;

use super::panel_rect::panel_body_size;
use super::{render, CameraImageSel, Command, ScreenshotSource};

// ── The Action Log's view of a command ───────────────────────────────────
//
// A mutating tool writes no entry of its own: it calls the same `AppState` and
// `Viewer3D` methods the GUI calls, and those record — which is what makes the
// actor column trustworthy, since two rows reading the same did the same thing.
// What is left for the drain is the two things no state method can know: the
// name of the tool, for a refusal, and the fact that a read happened at all.
//
// Which reads fold is the drain's knowledge too: a poll is one row per tool,
// but a screenshot is a picture the agent took and presumably looked at, so
// ten in a second are ten lines.

impl Command {
    /// The tool this command came from, as the wire names it.
    pub(crate) fn tool_name(&self) -> &'static str {
        match self {
            Command::GetScene => "get_scene",
            Command::ListCameraImages { .. } => "list_camera_images",
            Command::GetCameraImage { .. } => "get_camera_image",
            Command::GetCameraIntrinsics { .. } => "get_camera_intrinsics",
            Command::GetPoint { .. } => "get_point",
            Command::GetActionLog { .. } => "get_action_log",
            Command::OpenReconstruction { .. } => "open_reconstruction",
            Command::CloseReconstruction { .. } => "close_reconstruction",
            Command::SelectReconstruction { .. } => "select_reconstruction",
            Command::SelectCameraImage { .. } => "select_camera_image",
            Command::SelectCameraIntrinsics { .. } => "select_camera_intrinsics",
            Command::SelectPoint { .. } => "select_point",
            Command::ClearSelection { .. } => "clear_selection",
            Command::SetReconstructionDisplay { .. } => "set_reconstruction_display",
            Command::SetReconstructionTransform { .. } => "set_reconstruction_transform",
            Command::SetReconstructionTransformFromPatch { .. } => {
                "set_reconstruction_transform_from_patch"
            }
            Command::BakeReconstructionTransform { .. } => "bake_reconstruction_transform",
            Command::SetSolo { .. } => "set_solo",
            Command::GetImageDetailDisplay => "get_image_detail_display",
            Command::SetImageDetailDisplay { .. } => "set_image_detail_display",
            Command::GetImageDetailView => "get_image_detail_view",
            Command::SetImageDetailView { .. } => "set_image_detail_view",
            Command::GetTimingDetail => "get_timing_detail",
            Command::SetTimingDetail { .. } => "set_timing_detail",
            Command::SetView { .. } => "set_view",
            Command::GetWindowLayout => "get_window_layout",
            Command::SetWindowLayout { .. } => "set_window_layout",
            Command::ShowPanel { .. } => "show_panel",
            Command::HidePanel { .. } => "hide_panel",
            Command::GetHistory { .. } => "get_history",
            Command::Undo { .. } => "undo",
            Command::Redo { .. } => "redo",
            Command::JumpToVersion { .. } => "jump_to_version",
            Command::SaveReconstruction { .. } => "save_reconstruction",
            Command::DeletePoint { .. } => "delete_point",
            Command::RetriangulatePoint { .. } => "retriangulate_point",
            Command::RetriangulateAllPoints { .. } => "retriangulate_all_points",
            Command::PruneCoveredObservations { .. } => "prune_covered_observations",
            Command::DeleteCameraImage { .. } => "delete_camera_image",
            Command::MoveCameraImage { .. } => "move_camera_image",
            Command::ResectCameraImage { .. } => "resect_camera_image",
            Command::AddCameraImageToTracks { .. } => "add_camera_image_to_tracks",
            Command::BundleAdjust { .. } => "bundle_adjust",
            Command::ConvertToEmbeddedPatches { .. } => "convert_to_embedded_patches",
            Command::GetBench { .. } => "get_bench",
            Command::GetBenchTrack { .. } => "get_bench_track",
            Command::CreateBenchCluster { .. } => "create_bench_cluster",
            Command::CreateBenchTrack { .. } => "create_bench_track",
            Command::CreateTrackAtPixel { .. } => "create_track_at_pixel",
            Command::ActivateBenchItem { .. } => "activate_bench_item",
            Command::DeactivateBenchItem { .. } => "deactivate_bench_item",
            Command::RenameBenchItem { .. } => "rename_bench_item",
            Command::DiscardBenchItem { .. } => "discard_bench_item",
            Command::DuplicateBenchItem { .. } => "duplicate_bench_item",
            Command::AddBenchTrackObservation { .. } => "add_bench_track_observation",
            Command::TranslateBenchPatch { .. } => "translate_bench_patch",
            Command::ResizeBenchPatch { .. } => "resize_bench_patch",
            Command::ResizeBenchShape { .. } => "resize_bench_shape",
            Command::SpinBenchPatch { .. } => "spin_bench_patch",
            Command::SpinBenchShape { .. } => "spin_bench_shape",
            Command::TiltBenchPatch { .. } => "tilt_bench_patch",
            Command::SightBenchObservation { .. } => "sight_bench_observation",
            Command::ShapeBenchObservation { .. } => "shape_bench_observation",
            Command::SetBenchTrackVerdict { .. } => "set_bench_track_verdict",
            Command::ApplyBenchTrackThresholds { .. } => "apply_bench_track_thresholds",
            Command::SplitBenchTrack { .. } => "split_bench_track",
            Command::CommitBenchTrack { .. } => "commit_bench_track",
            Command::EvaluateBenchTrack { .. } => "evaluate_bench_track",
            Command::FitBenchTrack { .. } => "fit_bench_track",
            Command::SetBenchTrackStage { .. } => "set_bench_track_stage",
            Command::SearchBenchTrackDescriptors { .. } => "search_bench_track_descriptors",
            Command::SearchBenchTrackGeometry { .. } => "search_bench_track_geometry",
            Command::OpenIndexFiles { .. } => "open_index_files",
            Command::BuildIndexFiles { .. } => "build_index_files",
            Command::CloseIndexFiles { .. } => "close_index_files",
            Command::GetBackgroundTask => "get_background_task",
            Command::CancelBackgroundTask => "cancel_background_task",
            Command::Screenshot { .. } => "screenshot",
        }
    }

    /// The node whose **data** this command is about to change, read before it
    /// is applied.
    ///
    /// What a camera held in hand is ended for: the value under it is about to
    /// move. Distinct from [`Self::renumbers`], which is read afterwards and is
    /// about the caches a succeeded command invalidated -- the cursor moves are
    /// there and not here, because a cursor move under a held lock is what
    /// `camera_lock::resnap_camera_view` follows rather than something the lock
    /// has to be given up for.
    pub(super) fn edits(&self) -> Option<&str> {
        match self {
            Command::DeletePoint {
                reconstruction_label,
                ..
            }
            | Command::RetriangulatePoint {
                reconstruction_label,
                ..
            }
            | Command::RetriangulateAllPoints {
                reconstruction_label,
            }
            | Command::DeleteCameraImage {
                reconstruction_label,
                ..
            }
            | Command::MoveCameraImage {
                reconstruction_label,
                ..
            }
            | Command::ResectCameraImage {
                reconstruction_label,
                ..
            }
            | Command::AddCameraImageToTracks {
                reconstruction_label,
                ..
            }
            | Command::BundleAdjust {
                reconstruction_label,
                ..
            }
            | Command::BakeReconstructionTransform {
                reconstruction_label,
            }
            | Command::ConvertToEmbeddedPatches {
                reconstruction_label,
            }
            // The one bench step that writes the reconstruction. Every other
            // one changes the bench beside it, which is not the value a lock
            // is held over.
            | Command::CommitBenchTrack {
                reconstruction_label,
                ..
            } => Some(reconstruction_label),
            _ => None,
        }
    }

    /// The node this command may have renumbered, once it has succeeded.
    ///
    /// A bulk edit gives the node a whole new base and a cursor move lands on
    /// one, so an image index or a point index a panel cached is afterwards a
    /// statement about something else. The drain drops those caches for the
    /// nodes named here, exactly as the menus and the Edit History panel do
    /// around the same `AppState` calls. A point edit is not here, for the same
    /// reason the GUI keeps its caches across one: the base does not move.
    ///
    /// A camera move renumbers nothing -- the image table stays put and no
    /// point is deleted -- but it installs a whole new base, so what the panels
    /// cached *about* the geometry describes a value the node no longer holds.
    /// It is here for that, which is the same drop the lock's own commit makes
    /// in the window.
    ///
    /// The bundle adjustment is **not** here, though it renumbers as hard as
    /// anything does: it runs in the background, so the node it renumbers has
    /// not been renumbered yet when this is read. The frame drops those caches
    /// when the version actually lands, off `Polled::installed`.
    pub(super) fn renumbers(&self) -> Option<&str> {
        match self {
            Command::DeleteCameraImage {
                reconstruction_label,
                ..
            }
            // The bake installs a whole new base and renumbers nothing, which
            // is the camera move's case.
            | Command::BakeReconstructionTransform {
                reconstruction_label,
            }
            | Command::MoveCameraImage {
                reconstruction_label,
                ..
            }
            | Command::ResectCameraImage {
                reconstruction_label,
                ..
            }
            | Command::Undo {
                reconstruction_label,
            }
            | Command::Redo {
                reconstruction_label,
            }
            | Command::JumpToVersion {
                reconstruction_label,
                ..
            }
            // A commit writes a point where its origin was, and a replaced
            // point takes a new index, so the panels' cached tables are a
            // statement about a value the node no longer holds -- which is the
            // drop the window makes around the same call.
            | Command::CommitBenchTrack {
                reconstruction_label,
                ..
            }
            // A retriangulated point is a delete-and-re-add and takes a new
            // index for the same reason, so the same drop applies. Its
            // whole-value sibling is not here, for the reason the adjustment is
            // not: it runs in the background and has renumbered nothing yet.
            | Command::RetriangulatePoint {
                reconstruction_label,
                ..
            } => Some(reconstruction_label),
            _ => None,
        }
    }

    /// What the entry the drain writes for this command folds with.
    ///
    /// A read folds per tool, so a poll is one row however often it asks —
    /// except `screenshot`, which is discrete: a reader wants to know how many
    /// pictures were taken and of what. A mutating command records through the
    /// `AppState` / `Viewer3D` method it calls, which chooses its own run, and
    /// a refusal never folds, so `None` is the answer for everything else.
    pub(crate) fn run(&self) -> crate::action_log::Run {
        match self.kind() {
            Kind::Query(tool) if !matches!(self, Command::Screenshot { .. }) => Some(tool),
            _ => None,
        }
    }

    /// The Action Log kind a refusal of this command is filed under.
    ///
    /// A failed entry never coalesces, so this is about where the row belongs
    /// rather than about folding — but it should still be the kind the tool's
    /// success would have been.
    pub(crate) fn kind(&self) -> Kind {
        match self {
            Command::GetScene
            | Command::ListCameraImages { .. }
            | Command::GetCameraImage { .. }
            | Command::GetCameraIntrinsics { .. }
            | Command::GetPoint { .. }
            | Command::GetActionLog { .. }
            | Command::GetWindowLayout
            | Command::GetImageDetailDisplay
            | Command::GetImageDetailView
            | Command::GetTimingDetail
            | Command::GetHistory { .. }
            | Command::GetBackgroundTask
            | Command::GetBench { .. }
            | Command::GetBenchTrack { .. }
            | Command::Screenshot { .. } => Kind::Query(self.tool_name()),
            Command::OpenReconstruction { .. }
            | Command::CloseReconstruction { .. }
            | Command::SaveReconstruction { .. } => Kind::File,
            // The kind every edit, cursor move and refusal of one is filed
            // under, whoever asked for it.
            Command::Undo { .. }
            | Command::Redo { .. }
            | Command::JumpToVersion { .. }
            | Command::DeletePoint { .. }
            | Command::RetriangulatePoint { .. }
            | Command::RetriangulateAllPoints { .. }
            | Command::PruneCoveredObservations { .. }
            | Command::DeleteCameraImage { .. }
            | Command::MoveCameraImage { .. }
            | Command::ResectCameraImage { .. }
            | Command::AddCameraImageToTracks { .. }
            | Command::BundleAdjust { .. }
            | Command::BakeReconstructionTransform { .. }
            | Command::ConvertToEmbeddedPatches { .. }
            // The one bench step whose row is an `Edit`, because it is one
            // (`specs/gui/edits/commit-track.md`).
            | Command::CommitBenchTrack { .. }
            | Command::CancelBackgroundTask => Kind::Edit,
            // Every other bench step writes the bench beside the
            // reconstruction, which is the kind the panel's own refusals carry.
            Command::CreateBenchCluster { .. }
            | Command::CreateBenchTrack { .. }
            // Its row puts the track on the bench; the commit that follows
            // writes its own `Edit` row.
            | Command::CreateTrackAtPixel { .. }
            | Command::ActivateBenchItem { .. }
            | Command::DeactivateBenchItem { .. }
            | Command::RenameBenchItem { .. }
            | Command::DiscardBenchItem { .. }
            | Command::DuplicateBenchItem { .. }
            | Command::AddBenchTrackObservation { .. }
            | Command::TranslateBenchPatch { .. }
            | Command::ResizeBenchPatch { .. }
            | Command::ResizeBenchShape { .. }
            | Command::SpinBenchPatch { .. }
            | Command::SpinBenchShape { .. }
            | Command::TiltBenchPatch { .. }
            | Command::SightBenchObservation { .. }
            | Command::ShapeBenchObservation { .. }
            | Command::SetBenchTrackVerdict { .. }
            | Command::ApplyBenchTrackThresholds { .. }
            | Command::SplitBenchTrack { .. }
            | Command::EvaluateBenchTrack { .. }
            | Command::FitBenchTrack { .. }
            | Command::SetBenchTrackStage { .. }
            | Command::SearchBenchTrackDescriptors { .. }
            | Command::SearchBenchTrackGeometry { .. }
            // The three that are about the index rather than about a track:
            // nothing on the bench moves, and the row belongs beside the search
            // that will use them.
            | Command::OpenIndexFiles { .. }
            | Command::BuildIndexFiles { .. }
            | Command::CloseIndexFiles { .. } => Kind::Bench,
            Command::SelectReconstruction { .. }
            | Command::SelectCameraImage { .. }
            | Command::SelectCameraIntrinsics { .. }
            | Command::SelectPoint { .. }
            | Command::ClearSelection { .. } => Kind::Selection,
            Command::SetReconstructionDisplay { .. }
            | Command::SetSolo { .. }
            // A reframe pushes a version and is still not an edit: nothing it
            // does reaches the file, which is the bench step's case exactly.
            | Command::SetReconstructionTransform { .. }
            | Command::SetReconstructionTransformFromPatch { .. } => Kind::Scene,
            // The kind the HUD's own controls record under: the Image Detail
            // toolbar is the same sort of thing on a different panel, and the
            // Action Log toolbar's timing checkbox on a third.
            // The panel's view is a panel control like the rest of them: the
            // zoom and the pan are what its own wheel and drag move, and
            // nothing about them reaches the reconstruction.
            Command::SetImageDetailDisplay { .. }
            | Command::SetImageDetailView { .. }
            | Command::SetTimingDetail { .. } => Kind::Display,
            Command::SetView { .. } => Kind::View,
            Command::ShowPanel { .. } | Command::HidePanel { .. } => Kind::Layout,
            // One call, two portions, and a refusal has to be filed somewhere:
            // under the panels when it carried a panel portion — the coarser of
            // the two, and the one a reader looks for a layout refusal in —
            // and under the window otherwise.
            Command::SetWindowLayout { document } => {
                if document.get("layout").is_some() {
                    Kind::Layout
                } else {
                    Kind::Window
                }
            }
        }
    }
}

/// The Action Log text for a **read-only** tool, or `None` for a mutating one.
///
/// Written by the drain from the command rather than by the tool, because a
/// read changes no state and so has no state method to log through. Built
/// *before* the command is applied, so a deferred `screenshot` lines up in
/// order with the commands around it rather than at readback.
pub(crate) fn query_text(state: &AppState, viewer: &Viewer3D, command: &Command) -> Option<String> {
    // A call that named no reconstruction read the selected one, and the log
    // should say which that was rather than leaving the row ambiguous.
    let named = |label: &Option<String>| match label {
        Some(label) => label.clone(),
        None => state
            .selected_recon
            .and_then(|id| render::label_of(state, id))
            .unwrap_or_else(|| "(none selected)".to_string()),
    };
    Some(match command {
        Command::GetScene => "get_scene".to_string(),
        Command::ListCameraImages {
            reconstruction_label,
            offset,
            limit,
        } => format!(
            "list_camera_images {} {offset}..{}",
            named(reconstruction_label),
            offset + limit
        ),
        Command::GetCameraImage {
            reconstruction_label,
            camera_image,
        } => format!(
            "get_camera_image {} {}",
            named(reconstruction_label),
            camera_image_text(camera_image)
        ),
        Command::GetCameraIntrinsics {
            reconstruction_label,
            camera_intrinsics_index,
        } => format!(
            "get_camera_intrinsics {} #{camera_intrinsics_index}",
            named(reconstruction_label)
        ),
        Command::GetPoint { point } => format!("get_point {}", point_text(point)),
        Command::GetActionLog { since_revision, .. } => {
            format!("get_action_log since {since_revision}")
        }
        Command::GetWindowLayout => "get_window_layout".to_string(),
        Command::GetBackgroundTask => "get_background_task".to_string(),
        Command::GetImageDetailDisplay => "get_image_detail_display".to_string(),
        Command::GetImageDetailView => "get_image_detail_view".to_string(),
        Command::GetTimingDetail => "get_timing_detail".to_string(),
        Command::GetHistory {
            reconstruction_label,
        } => format!("get_history {reconstruction_label}"),
        Command::GetBench {
            reconstruction_label,
        } => format!("get_bench {reconstruction_label}"),
        Command::GetBenchTrack {
            reconstruction_label,
            track,
        } => format!(
            "get_bench_track {reconstruction_label} {}",
            track.as_deref().unwrap_or("(the active track)")
        ),
        Command::Screenshot {
            panel,
            hud,
            max_dimension,
        } => {
            let source = match (panel, hud) {
                (None, _) => ScreenshotSource::Window,
                (Some(crate::dock::Tab::Viewer3D), false) => ScreenshotSource::ViewportRender,
                (Some(panel), _) => ScreenshotSource::Panel(*panel),
            };
            let [width, height] = screenshot_size(state, viewer, source, *max_dimension);
            let target = match source {
                ScreenshotSource::Window => "window",
                ScreenshotSource::ViewportRender => crate::dock::Tab::Viewer3D.wire_name(),
                ScreenshotSource::Panel(panel) => panel.wire_name(),
            };
            let without_hud = if matches!(source, ScreenshotSource::ViewportRender) {
                " without HUD"
            } else {
                ""
            };
            format!("screenshot {target} {width}×{height}{without_hud}")
        }
        _ => return None,
    })
}

/// `images/IMG_0007.jpg`, or `#7` where the call named an index.
fn camera_image_text(selector: &CameraImageSel) -> String {
    match selector {
        CameraImageSel::Index(index) => format!("#{index}"),
        CameraImageSel::Name(name) => name.clone(),
    }
}

/// A point query as the user would have typed it.
fn point_text(query: &crate::goto_point::PointQuery) -> String {
    match query {
        crate::goto_point::PointQuery::Index(index) => format!("#{index}"),
        crate::goto_point::PointQuery::Qualified { hash, index } => format!("pt3d_{hash}_{index}"),
    }
}

/// The size a screenshot taken this frame would come back at.
///
/// The window, the panel or the viewport as last laid out, shrunk by
/// `max_dimension` exactly as the readback shrinks the pixels — so the log
/// line, the caption and the image agree. A panel's rectangle is the one this
/// frame's predecessor laid out, which is the right size in every frame but the
/// one that opened the panel.
pub(super) fn screenshot_size(
    state: &AppState,
    viewer: &Viewer3D,
    source: ScreenshotSource,
    max_dimension: Option<u32>,
) -> [u32; 2] {
    let [width, height] = match source {
        ScreenshotSource::Window => state
            .window
            .as_ref()
            .map(|info| info.inner_size)
            .unwrap_or([0, 0]),
        ScreenshotSource::ViewportRender => viewer.panel_size,
        ScreenshotSource::Panel(panel) => {
            let scale = state
                .window
                .as_ref()
                .map(|info| info.scale_factor)
                .unwrap_or(1.0);
            panel_body_size(&state.dock, panel, scale as f32).unwrap_or([0, 0])
        }
    };
    let Some(limit) = max_dimension else {
        return [width, height];
    };
    let longest = width.max(height);
    if longest <= limit || longest == 0 {
        return [width, height];
    }
    let scale = f64::from(limit) / f64::from(longest);
    [
        ((f64::from(width) * scale).round() as u32).max(1),
        ((f64::from(height) * scale).round() as u32).max(1),
    ]
}
