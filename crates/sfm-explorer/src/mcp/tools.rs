// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The tool table, and the parse from a `tools/call` argument object to a
//! [`Command`].
//!
//! Tool names and argument names *are* the API — they live in client configs
//! and in the prompts people write against them — so the vocabulary here obeys
//! one rule without exception: **one entity, one spelled-out word, in tool
//! names, arguments and reply fields alike.** No abbreviations, and no word
//! that names two things. See "The wire vocabulary" in
//! `specs/gui/mcp-server.md` for what that buys and what it costs.
//!
//! The catalog families live in [`mod@catalog`] and the parser stays here. Together
//! they state what each tool accepts; a catalog-wide test walks both halves so
//! an advertised schema and its parser cannot drift.

use serde_json::{json, Map, Value};
use sfmtool_core::reconstruction::prune_covered::PruneCoveredOptions;

use super::{
    CameraImageSel, CloseTarget, Command, DisplayChange, Placement, SelectionScope, ToolError,
    ViewCommand,
};
use crate::action_log::Actor;
use crate::dock::Tab;
use crate::goto_point::{parse_point_query, PointQuery};

/// What a tool does to the viewer, which is all the MCP annotations need to
/// know.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ToolKind {
    /// Changes nothing.
    Read,
    /// Changes the scene, the selection, the view, or what a loaded
    /// reconstruction holds. Never a file on disk: an edit makes a new version
    /// in memory, and undo puts it back.
    Write,
    /// Writes a file. One tool, `save_reconstruction`, and the reason the kind
    /// exists at all: it is the only call on this surface that can overwrite
    /// something the human cannot undo.
    Save,
}

/// One advertised tool.
pub(crate) struct ToolSpec {
    pub(crate) name: &'static str,
    pub(crate) description: &'static str,
    pub(crate) kind: ToolKind,
    pub(crate) schema: Value,
}

/// Every tool this surface advertises, in the order `tools/list` reports them.
mod catalog;

pub(crate) fn catalog() -> &'static [ToolSpec] {
    static CATALOG: std::sync::OnceLock<Vec<ToolSpec>> = std::sync::OnceLock::new();
    CATALOG.get_or_init(catalog::build_catalog)
}

// ── The parse ────────────────────────────────────────────────────────────

/// Build the [`Command`] a `tools/call` asked for.
///
/// The schemas in `catalog` are closed and typed, so a compliant client will not
/// reach most of these errors; they are here because a tool call arrives from
/// whatever the agent actually sent, and "silently did something else" is the
/// one answer this surface must never give.
pub(crate) fn parse(
    name: &str,
    arguments: Option<&Map<String, Value>>,
) -> Result<Command, ToolError> {
    static EMPTY: std::sync::OnceLock<Map<String, Value>> = std::sync::OnceLock::new();
    let map = arguments.unwrap_or_else(|| EMPTY.get_or_init(Map::new));
    let args = Args { tool: name, map };

    // A tool's closed input schema is the single source of truth for its
    // top-level vocabulary. Unknown names deliberately skip this lookup and
    // reach the match's existing unknown-tool error below.
    if let Some(spec) = catalog().iter().find(|spec| spec.name == name) {
        let allowed = spec.schema["properties"]
            .as_object()
            .expect("every tool schema has object properties")
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>();
        args.reject_unknown(&allowed)?;
    }

    let command = match name {
        "get_scene" => Command::GetScene,
        "list_camera_images" => Command::ListCameraImages {
            reconstruction_label: args.optional_string("reconstruction_label")?,
            offset: args.optional_usize("offset")?.unwrap_or(0),
            limit: args
                .optional_usize("limit")?
                .unwrap_or(super::read::DEFAULT_LIMIT),
        },
        "get_camera_image" => Command::GetCameraImage {
            reconstruction_label: args.optional_string("reconstruction_label")?,
            camera_image: args.camera_image("camera_image")?,
        },
        "get_camera_intrinsics" => Command::GetCameraIntrinsics {
            reconstruction_label: args.optional_string("reconstruction_label")?,
            camera_intrinsics_index: args.required_usize("camera_intrinsics_index")?,
        },
        "get_point" => Command::GetPoint {
            point: args.point("point")?,
        },
        "get_action_log" => Command::GetActionLog {
            since_revision: args.optional_u64("since_revision")?.unwrap_or(0),
            limit: args
                .optional_usize("limit")?
                .unwrap_or(super::read::ACTION_LOG_DEFAULT_LIMIT),
            actors: args.actors("actors")?,
            detail: args.optional_bool("detail")?.unwrap_or(false),
        },
        "open_reconstruction" => Command::OpenReconstruction {
            path: std::path::PathBuf::from(args.required_string("path")?),
        },
        "close_reconstruction" => {
            let all = args.optional_bool("all")?.unwrap_or(false);
            let label = args.optional_string("reconstruction_label")?;
            match (all, label) {
                (true, None) => Command::CloseReconstruction {
                    target: CloseTarget::All,
                },
                (false, Some(label)) => Command::CloseReconstruction {
                    target: CloseTarget::One(label),
                },
                (true, Some(_)) => {
                    return Err(args.error(
                        "takes either reconstruction_label or all: true, not both — \"close this \
                         one\" and \"close everything\" are different requests.",
                    ))
                }
                (false, None) => {
                    return Err(args
                        .error("needs a reconstruction_label, or all: true to clear the scene."))
                }
            }
        }
        "select_reconstruction" => Command::SelectReconstruction {
            reconstruction_label: args.required_string("reconstruction_label")?,
        },
        "select_camera_image" => Command::SelectCameraImage {
            reconstruction_label: args.optional_string("reconstruction_label")?,
            camera_image: args.camera_image("camera_image")?,
        },
        "select_camera_intrinsics" => Command::SelectCameraIntrinsics {
            reconstruction_label: args.optional_string("reconstruction_label")?,
            camera_intrinsics_index: args.required_usize("camera_intrinsics_index")?,
        },
        "select_point" => Command::SelectPoint {
            point: args.point("point")?,
        },
        "clear_selection" => {
            let scope = match args.optional_string("scope")?.as_deref() {
                None | Some("all") => SelectionScope::All,
                Some("camera_image") => SelectionScope::CameraImage,
                Some("camera_intrinsics") => SelectionScope::CameraIntrinsics,
                Some("point") => SelectionScope::Point,
                Some(other) => {
                    return Err(args.error(format!(
                        "does not know the scope {other:?} — expected all, camera_image, \
                         camera_intrinsics or point."
                    )))
                }
            };
            Command::ClearSelection { scope }
        }
        "set_reconstruction_display" => {
            let change = DisplayChange {
                visible: args.optional_bool("visible")?,
                interactive: args.optional_bool("interactive")?,
                show_points: args.optional_bool("show_points")?,
                show_camera_images: args.optional_bool("show_camera_images")?,
                show_patches: args.optional_bool("show_patches")?,
                show_points_at_infinity: args.optional_bool("show_points_at_infinity")?,
                // Doubly optional: absent leaves the tint alone, an explicit
                // null clears it.
                tint: match args.map.get("tint") {
                    None => None,
                    Some(Value::Null) => Some(None),
                    Some(Value::String(name)) => Some(Some(name.clone())),
                    Some(_) => return Err(args.error("wants tint to be a palette name or null.")),
                },
            };
            if change == DisplayChange::default() {
                return Err(args.error("was given nothing to change."));
            }
            Command::SetReconstructionDisplay {
                reconstruction_label: args.required_string("reconstruction_label")?,
                change,
            }
        }
        "set_reconstruction_transform" => {
            let transform = args
                .map
                .get("transform")
                .and_then(Value::as_object)
                .ok_or_else(|| {
                    args.error(
                        "needs transform, an object carrying rotation_wxyz, translation and \
                         scale.",
                    )
                })?;
            let inner = Args {
                tool: "set_reconstruction_transform.transform",
                map: transform,
            };
            inner.reject_unknown(&["rotation_wxyz", "translation", "scale"])?;
            let rotation_wxyz = inner.required_vec4("rotation_wxyz")?;
            let translation = inner.required_vec3("translation")?;
            let scale = inner.required_f64("scale")?;
            if rotation_wxyz.iter().all(|c| *c == 0.0) {
                return Err(inner.error("needs a rotation_wxyz with some length to normalise."));
            }
            if !(scale.is_finite() && scale > 0.0) {
                return Err(inner.error(format!("needs a positive, finite scale, not {scale}.")));
            }
            Command::SetReconstructionTransform {
                reconstruction_label: args.required_string("reconstruction_label")?,
                rotation_wxyz,
                translation,
                scale,
            }
        }
        "set_reconstruction_transform_from_patch" => {
            let mode = args.required_string("mode")?;
            let mode =
                crate::display_transform::PatchReframe::from_wire_name(&mode).ok_or_else(|| {
                    args.error(format!(
                        "does not know the mode {mode:?}: expected set_to_origin, \
                         align_normal_to_z, translate_to_origin or translate_to_xy_plane."
                    ))
                })?;
            Command::SetReconstructionTransformFromPatch {
                reconstruction_label: args.required_string("reconstruction_label")?,
                mode,
            }
        }
        "bake_reconstruction_transform" => Command::BakeReconstructionTransform {
            reconstruction_label: args.required_string("reconstruction_label")?,
        },
        "set_solo" => Command::SetSolo {
            reconstruction_label: args.optional_string("reconstruction_label")?,
        },
        "get_image_detail_display" => Command::GetImageDetailDisplay,
        // Every vocabulary this tool has is static — seven modes, two ladders,
        // the bounds on a size filter — so the whole call is validated here and
        // `apply` cannot fail. That is what makes a refusal atomic.
        "set_image_detail_display" => Command::SetImageDetailDisplay {
            change: super::display::parse_change(&args)?,
        },
        "get_image_detail_view" => Command::GetImageDetailView,
        // The one-target rule, the zoom's range and the targets that refuse a
        // zoom are all settled here, before a `Command` exists: what is left
        // for the tool body is resolving the handles, which is the part that
        // needs the scene.
        "set_image_detail_view" => Command::SetImageDetailView {
            request: super::display::parse_view(&args)?,
        },
        "get_timing_detail" => Command::GetTimingDetail,
        // Required rather than a toggle, for the reason `set_solo` takes the
        // state it wants: an agent issuing a toggle cannot know the outcome
        // without reading first, and a retried call would undo itself.
        "set_timing_detail" => Command::SetTimingDetail {
            enabled: args.required_bool("enabled")?,
        },
        "set_view" => parse_set_view(&args)?,
        "get_window_layout" => Command::GetWindowLayout,
        "set_window_layout" => {
            let document = Value::Object(args.map.clone());
            // Carried through unparsed: `WindowLayout::from_value` reads it in
            // the tool body, so a document the viewer will not accept is a
            // refusal the agent and the Action Log both see, in the layout
            // parser's own words with its path. The one thing checked here is
            // the tool's own rule — that a call has to ask for something —
            // which is the parser's own definition of empty, so the two cannot
            // come to disagree about what an empty document is.
            if crate::layout::WindowLayout::from_value(&document)
                .is_ok_and(|document| document.is_empty())
            {
                return Err(args.error("was given nothing to do — pass window, layout, or both."));
            }
            Command::SetWindowLayout { document }
        }
        "show_panel" => Command::ShowPanel {
            panel: args.panel("panel_name")?,
        },
        "hide_panel" => Command::HidePanel {
            panel: args.panel("panel_name")?,
        },
        "get_history" => Command::GetHistory {
            reconstruction_label: args.required_string("reconstruction_label")?,
        },
        "undo" => Command::Undo {
            reconstruction_label: args.required_string("reconstruction_label")?,
        },
        "redo" => Command::Redo {
            reconstruction_label: args.required_string("reconstruction_label")?,
        },
        "jump_to_version" => Command::JumpToVersion {
            reconstruction_label: args.required_string("reconstruction_label")?,
            serial: args.required_string("serial")?,
        },
        "save_reconstruction" => Command::SaveReconstruction {
            reconstruction_label: args.required_string("reconstruction_label")?,
            path: args.optional_string("path")?.map(std::path::PathBuf::from),
            minimal: args.optional_bool("minimal")?.unwrap_or(false),
            workspace_path: args.optional_string("workspace_path")?,
        },
        "delete_point" => Command::DeletePoint {
            reconstruction_label: args.required_string("reconstruction_label")?,
            point: args.point("point")?,
        },
        "delete_camera_image" => Command::DeleteCameraImage {
            reconstruction_label: args.required_string("reconstruction_label")?,
            camera_image: args.camera_image("camera_image")?,
        },
        "move_camera_image" => {
            let pose = args
                .map
                .get("world_from_camera")
                .and_then(Value::as_object)
                .ok_or_else(|| {
                    args.error(
                        "needs world_from_camera, an object carrying quaternion_wxyz and \
                         translation.",
                    )
                })?;
            let inner = Args {
                tool: "move_camera_image.world_from_camera",
                map: pose,
            };
            inner.reject_unknown(&["quaternion_wxyz", "translation"])?;
            Command::MoveCameraImage {
                reconstruction_label: args.required_string("reconstruction_label")?,
                camera_image: args.camera_image("camera_image")?,
                quaternion_wxyz: inner.required_vec4("quaternion_wxyz")?,
                translation: inner.required_vec3("translation")?,
            }
        }
        "resect_camera_image" => Command::ResectCameraImage {
            reconstruction_label: args.required_string("reconstruction_label")?,
            camera_image: args.camera_image("camera_image")?,
            from_matches: args.optional_bool("from_matches")?.unwrap_or(false),
        },
        "bundle_adjust" => Command::BundleAdjust {
            reconstruction_label: args.required_string("reconstruction_label")?,
            release_focal: args.optional_bool("release_focal")?.unwrap_or(false),
        },
        "convert_to_embedded_patches" => Command::ConvertToEmbeddedPatches {
            reconstruction_label: args.required_string("reconstruction_label")?,
        },
        "retriangulate_point" => Command::RetriangulatePoint {
            reconstruction_label: args.required_string("reconstruction_label")?,
            point: args.point("point")?,
        },
        "retriangulate_all_points" => Command::RetriangulateAllPoints {
            reconstruction_label: args.required_string("reconstruction_label")?,
        },
        "prune_covered_observations" => {
            let defaults = PruneCoveredOptions::default();
            Command::PruneCoveredObservations {
                reconstruction_label: args.required_string("reconstruction_label")?,
                options: PruneCoveredOptions {
                    footprint_fraction: args
                        .optional_f64("footprint_fraction")?
                        .unwrap_or(defaults.footprint_fraction),
                    ratio: args.optional_f64("ratio")?.unwrap_or(defaults.ratio),
                    min_fine_radius_px: args
                        .optional_f64("min_fine_radius_px")?
                        .unwrap_or(defaults.min_fine_radius_px),
                    min_observations: defaults.min_observations,
                },
            }
        }
        "get_bench" => Command::GetBench {
            reconstruction_label: args.required_string("reconstruction_label")?,
        },
        "get_bench_track" => Command::GetBenchTrack {
            reconstruction_label: args.required_string("reconstruction_label")?,
            track: args.optional_string("track")?,
        },
        "create_bench_cluster" => Command::CreateBenchCluster {
            reconstruction_label: args.required_string("reconstruction_label")?,
            camera_image: args.camera_image("camera_image")?,
            seed: parse_seed(&args)?,
        },
        "create_bench_track" => Command::CreateBenchTrack {
            reconstruction_label: args.required_string("reconstruction_label")?,
            point: args.point("point")?,
        },
        "activate_bench_item" => Command::ActivateBenchItem {
            reconstruction_label: args.required_string("reconstruction_label")?,
            item: args.required_string("item")?,
        },
        "deactivate_bench_item" => Command::DeactivateBenchItem {
            reconstruction_label: args.required_string("reconstruction_label")?,
        },
        "rename_bench_item" => Command::RenameBenchItem {
            reconstruction_label: args.required_string("reconstruction_label")?,
            item: args.required_string("item")?,
            label: args.required_string("label")?,
        },
        "discard_bench_item" => Command::DiscardBenchItem {
            reconstruction_label: args.required_string("reconstruction_label")?,
            item: args.required_string("item")?,
        },
        "add_bench_track_observation" => Command::AddBenchTrackObservation {
            reconstruction_label: args.required_string("reconstruction_label")?,
            track: args.optional_string("track")?,
            camera_image: args.camera_image("camera_image")?,
            seed: parse_seed(&args)?,
        },
        "duplicate_bench_item" => Command::DuplicateBenchItem {
            reconstruction_label: args.required_string("reconstruction_label")?,
            item: args.optional_string("item")?,
        },
        "translate_bench_patch" => Command::TranslateBenchPatch {
            reconstruction_label: args.required_string("reconstruction_label")?,
            track: args.optional_string("track")?,
            to: translate_target(&args)?,
        },
        "sight_bench_observation" => Command::SightBenchObservation {
            reconstruction_label: args.required_string("reconstruction_label")?,
            track: args.optional_string("track")?,
            observation: args.required_usize("observation")?,
            pixel: args.required_pixel_f64("pixel")?,
        },
        "shape_bench_observation" => Command::ShapeBenchObservation {
            reconstruction_label: args.required_string("reconstruction_label")?,
            track: args.optional_string("track")?,
            observation: args.required_usize("observation")?,
            shape: args
                .optional_affine("shape")?
                .ok_or_else(|| args.error("needs shape: a 2x2 affine.".to_string()))?,
        },
        "resize_bench_patch" => Command::ResizeBenchPatch {
            reconstruction_label: args.required_string("reconstruction_label")?,
            track: args.optional_string("track")?,
            to: resize_target(&args)?,
        },
        "resize_bench_shape" => Command::ResizeBenchShape {
            reconstruction_label: args.required_string("reconstruction_label")?,
            track: args.optional_string("track")?,
            observation: args.required_usize("observation")?,
            edge: args.edge("edge")?,
            pixel: args.required_pixel_f64("pixel")?,
        },
        "tilt_bench_patch" => Command::TiltBenchPatch {
            reconstruction_label: args.required_string("reconstruction_label")?,
            track: args.optional_string("track")?,
            normal: args.required_vec3("normal")?,
        },
        "spin_bench_patch" => Command::SpinBenchPatch {
            reconstruction_label: args.required_string("reconstruction_label")?,
            track: args.optional_string("track")?,
            degrees: args.required_f64("degrees")?,
        },
        "spin_bench_shape" => Command::SpinBenchShape {
            reconstruction_label: args.required_string("reconstruction_label")?,
            track: args.optional_string("track")?,
            observation: args.required_usize("observation")?,
            degrees: args.required_f64("degrees")?,
        },
        "set_bench_track_verdict" => Command::SetBenchTrackVerdict {
            reconstruction_label: args.required_string("reconstruction_label")?,
            track: args.optional_string("track")?,
            observation: args.required_usize("observation")?,
            verdict: args.verdict("verdict")?,
        },
        "apply_bench_track_thresholds" => Command::ApplyBenchTrackThresholds {
            reconstruction_label: args.required_string("reconstruction_label")?,
            track: args.optional_string("track")?,
            thresholds: super::ThresholdChange {
                min_zncc: args.optional_f64("min_zncc")?,
                max_shift_px: args.optional_f64("max_shift_px")?,
                max_keypoint_uncertainty: args.optional_f64("max_keypoint_uncertainty")?,
                min_relative_zncc: args.optional_f64("min_relative_zncc")?,
            },
        },
        "split_bench_track" => Command::SplitBenchTrack {
            reconstruction_label: args.required_string("reconstruction_label")?,
            track: args.optional_string("track")?,
            observations: args.observations("observations")?,
        },
        "commit_bench_track" => Command::CommitBenchTrack {
            reconstruction_label: args.required_string("reconstruction_label")?,
            track: args.optional_string("track")?,
        },
        "evaluate_bench_track" => Command::EvaluateBenchTrack {
            reconstruction_label: args.required_string("reconstruction_label")?,
            track: args.optional_string("track")?,
            search_px: args.optional_f64("search_px")?,
        },
        "fit_bench_track" => Command::FitBenchTrack {
            reconstruction_label: args.required_string("reconstruction_label")?,
            track: args.optional_string("track")?,
            search_px: args.optional_f64("search_px")?,
        },
        "set_bench_track_stage" => Command::SetBenchTrackStage {
            reconstruction_label: args.required_string("reconstruction_label")?,
            track: args.optional_string("track")?,
            stage: args.stage("stage")?,
        },
        "search_bench_track_descriptors" => Command::SearchBenchTrackDescriptors {
            reconstruction_label: args.required_string("reconstruction_label")?,
            track: args.optional_string("track")?,
            observation: args.required_usize("observation")?,
            radius_px: args.optional_f64("radius_px")?,
            min_inliers: args.optional_usize("min_inliers")?,
        },
        "search_bench_track_geometry" => Command::SearchBenchTrackGeometry {
            reconstruction_label: args.required_string("reconstruction_label")?,
            track: args.optional_string("track")?,
            observation: args.required_usize("observation")?,
        },
        "open_sift_index" => Command::OpenSiftIndex {
            reconstruction_label: args.required_string("reconstruction_label")?,
            path: args.optional_string("path")?,
        },
        "build_sift_index" => Command::BuildSiftIndex {
            reconstruction_label: args.required_string("reconstruction_label")?,
            path: args.optional_string("path")?,
        },
        "close_sift_index" => Command::CloseSiftIndex {
            reconstruction_label: args.required_string("reconstruction_label")?,
        },
        "get_background_task" => Command::GetBackgroundTask,
        "cancel_background_task" => Command::CancelBackgroundTask,
        "screenshot" => {
            let panel = match args.map.get("panel_name") {
                None | Some(Value::Null) => None,
                Some(_) => Some(args.panel("panel_name")?),
            };
            let hud = args.optional_bool("hud")?.unwrap_or(true);
            // `hud` is a statement about the picture *underneath* what egui
            // painted, and only the 3D Viewer has one. Refused elsewhere rather
            // than read as a request to draw the frame differently: a panel
            // drawn differently for a screenshot would hand the agent a picture
            // the human never saw.
            if !hud && panel != Some(Tab::Viewer3D) {
                return Err(args.error(
                    "takes hud only with panel_name \"viewer_3d\": hud applies to the 3D Viewer \
                     only; the other panels have no picture underneath what is drawn on them.",
                ));
            }
            Command::Screenshot {
                panel,
                hud,
                max_dimension: args
                    .optional_usize("max_dimension")?
                    .map(|d| d.min(u32::MAX as usize) as u32),
            }
        }
        other => {
            return Err(ToolError::new(format!(
                "There is no tool named {other:?}. Call tools/list for what this viewer offers."
            )))
        }
    };
    Ok(command)
}

/// The three seed forms the two bench creates share, told apart by which field
/// is present.
///
/// A `.sift` feature carries its own position and its own keypoint frame, so a
/// call that named one has said everything there is to say about where the
/// observation goes: a pixel or a shape alongside it would be a second answer
/// to a question already answered, and is refused rather than silently losing
/// to one or the other. A pixel takes at most one statement of size, since an
/// affine shape already says how large the patch is.
fn parse_seed(args: &Args) -> Result<crate::bench::Seed, ToolError> {
    let present = |key: &str| args.map.contains_key(key);
    if let Some(feature) = args.optional_u32("feature")? {
        let with: Vec<&str> = ["pixel", "radius_px", "affine"]
            .into_iter()
            .filter(|key| present(key))
            .collect();
        if !with.is_empty() {
            return Err(args.error(format!(
                "was given {} with feature — a .sift feature carries its own position and its \
                 own shape.",
                with.join(" and ")
            )));
        }
        return Ok(crate::bench::Seed::Feature { feature });
    }
    let Some(pixel) = args.optional_numbers::<2>("pixel")? else {
        return Err(args.error(
            "needs somewhere to seed from: a pixel, a pixel with radius_px or affine, or a \
             feature.",
        ));
    };
    if present("radius_px") && present("affine") {
        return Err(args.error(
            "was given both radius_px and affine — an affine shape already says how large the \
             patch is.",
        ));
    }
    if let Some(shape) = args.optional_affine("affine")? {
        return Ok(crate::bench::Seed::Affine { pixel, shape });
    }
    Ok(crate::bench::Seed::Pixel {
        pixel,
        radius_px: args.radius("radius_px")?.map(f64::from),
    })
}

/// `set_view`'s five forms, told apart by which field is present.
///
/// The forms are exclusive and the check is up front, because they are
/// *intents* rather than representations: a call carrying both `fit` and
/// `position` has no answer, and guessing one would move the camera somewhere
/// the agent did not ask for. The explicit camera is one form however many of
/// its pieces a call carries, so any of them puts the call in it.
fn parse_set_view(args: &Args) -> Result<Command, ToolError> {
    let present = |key: &str| args.map.contains_key(key);
    let explicit: Vec<&str> = PLACEMENT_KEYS
        .into_iter()
        .filter(|key| present(key))
        .collect();
    let forms: Vec<&str> = ["fit", "look_through", "exit_camera_view"]
        .into_iter()
        .filter(|key| present(key))
        .chain(explicit.first().copied())
        .collect();
    if forms.len() > 1 {
        return Err(args.error(format!(
            "was given {} at once — fit, look_through, exit_camera_view and the explicit camera \
             are exclusive, one per call.",
            forms.join(" and ")
        )));
    }

    let fov = args.optional_f64("fov_short_axis_deg")?;

    if present("fit") {
        return Ok(Command::SetView {
            view: ViewCommand::Fit {
                reconstruction_label: args.optional_string("fit")?,
            },
        });
    }
    if let Some(look_through) = args.map.get("look_through") {
        let map = look_through.as_object().ok_or_else(|| {
            args.error("wants look_through to be an object naming a camera image.")
        })?;
        let inner = Args {
            tool: "set_view.look_through",
            map,
        };
        inner.reject_unknown(&["reconstruction_label", "camera_image"])?;
        return Ok(Command::SetView {
            view: ViewCommand::LookThrough {
                reconstruction_label: inner.optional_string("reconstruction_label")?,
                camera_image: inner.camera_image("camera_image")?,
            },
        });
    }
    if present("exit_camera_view") {
        if args.optional_bool("exit_camera_view")? != Some(true) {
            return Err(args.error(
                "reads exit_camera_view: false as no request at all — omit it, or pass true.",
            ));
        }
        return Ok(Command::SetView {
            view: ViewCommand::ExitCameraView,
        });
    }
    if !explicit.is_empty() {
        return Ok(Command::SetView {
            view: ViewCommand::Place(parse_placement(args, fov)?),
        });
    }
    match fov {
        Some(fov_short_axis_deg) => Ok(Command::SetView {
            view: ViewCommand::Fov { fov_short_axis_deg },
        }),
        None => Err(args.error(
            "was given nothing to do — pass fit, look_through, exit_camera_view, a piece of the \
             explicit camera (position, target, forward, target_distance or orientation_wxyz), \
             or fov_short_axis_deg alone.",
        )),
    }
}

/// Every argument that puts a `set_view` call in the explicit camera form.
///
/// `up` and `world_up` are in the list even though neither determines a
/// camera: a call carrying one of them alone has asked for a roll and nothing
/// to roll, and the refusal that says so belongs with the rest of the family
/// rather than in the catch-all at the end of [`parse_set_view`].
const PLACEMENT_KEYS: [&str; 7] = [
    "position",
    "target",
    "forward",
    "orientation_wxyz",
    "target_distance",
    "up",
    "world_up",
];

/// The pieces of the explicit camera one call carried.
///
/// What a call does not carry is preserved, so this parse is not about which
/// pieces are missing but about which combinations *cannot* be honoured: a
/// piece that would over-determine the camera, and a piece the resolved form
/// would never read. Both are refused. An argument silently ignored leaves the
/// agent believing it asked for something it did not, which is the same reason
/// the schemas are closed.
fn parse_placement(args: &Args, fov: Option<f64>) -> Result<Placement, ToolError> {
    let present = |key: &str| args.map.contains_key(key);
    if present("orientation_wxyz") {
        // The exact form states the orientation outright, so nothing that
        // would derive one may ride along, and its roll travels in world_up.
        if present("target") {
            return Err(args.error(
                "was given both target and orientation_wxyz — the look-at form and the exact \
                 form are exclusive.",
            ));
        }
        if present("forward") {
            return Err(args.error(
                "was given both forward and orientation_wxyz -- the exact form states the \
                 orientation, so there is no direction to derive one from.",
            ));
        }
        if present("up") {
            return Err(args.error(
                "was given up with orientation_wxyz -- the exact form carries its roll in \
                 world_up.",
            ));
        }
        return Ok(Placement {
            position: Some(args.required_vec3("position")?),
            orientation_wxyz: Some(args.required_vec4("orientation_wxyz")?),
            target_distance: Some(args.required_f64("target_distance")?),
            world_up: args.optional_vec3("world_up")?,
            fov_short_axis_deg: fov,
            ..Placement::default()
        });
    }
    let pair = present("position") && present("target");
    if pair && present("target_distance") {
        return Err(args.error(
            "was given position, target and target_distance -- the separation of position and \
             target is the distance.",
        ));
    }
    if pair && present("forward") {
        return Err(args.error(
            "was given position, target and forward -- the pair already fixes the view \
             direction.",
        ));
    }
    if present("world_up") {
        return Err(args.error(
            "was given world_up outside the exact form -- pass up to roll a view whose \
             direction is being derived.",
        ));
    }
    if present("up") && !pair && !present("forward") {
        return Err(args.error(
            "was given up with nothing to roll -- up steers the roll only where the orientation \
             is being derived, from forward or from position with target.",
        ));
    }
    Ok(Placement {
        position: args.optional_vec3("position")?,
        target: args.optional_vec3("target")?,
        forward: args.optional_vec3("forward")?,
        target_distance: args.optional_f64("target_distance")?,
        up: args.optional_vec3("up")?,
        fov_short_axis_deg: fov,
        ..Placement::default()
    })
}

/// One tool call's argument object, with the accessors that turn a JSON value
/// into a typed argument or into a message saying what was wrong with it.
pub(super) struct Args<'a> {
    tool: &'a str,
    map: &'a Map<String, Value>,
}

impl<'a> Args<'a> {
    /// The accessors over a nested object, named for the path that reaches it
    /// — `set_image_detail_display.intrinsics` — so a refusal from inside one
    /// says which sub-object it is about.
    pub(super) fn new(tool: &'a str, map: &'a Map<String, Value>) -> Self {
        Self { tool, map }
    }

    /// The raw value under `key`, for the doubly-optional arguments where an
    /// explicit `null` means something other than "absent".
    pub(super) fn get(&self, key: &str) -> Option<&Value> {
        self.map.get(key)
    }
}

impl Args<'_> {
    /// `"<tool> <complaint>"`, so every message from this module reads as a
    /// sentence about the tool that was called.
    pub(super) fn error(&self, complaint: impl std::fmt::Display) -> ToolError {
        ToolError::new(format!("{} {complaint}", self.tool))
    }

    fn wrong_type(&self, key: &str, expected: &str, got: &Value) -> ToolError {
        self.error(format!(
            "wants {key} to be {expected} — got {}.",
            describe(got)
        ))
    }

    /// Refuse an argument the tool does not have.
    ///
    /// The schemas say `additionalProperties: false`, but a schema is only
    /// enforced by clients that enforce it. An ignored typo would leave the
    /// agent believing it asked for something it did not, and the whole reason
    /// this surface returns its resulting state is so that never happens.
    pub(super) fn reject_unknown(&self, allowed: &[&str]) -> Result<(), ToolError> {
        let unknown: Vec<String> = self
            .map
            .keys()
            .filter(|key| !allowed.contains(&key.as_str()))
            .map(|key| format!("{key:?}"))
            .collect();
        if unknown.is_empty() {
            return Ok(());
        }
        let known = if allowed.is_empty() {
            "it takes none".to_string()
        } else {
            format!("it takes {}", allowed.join(", "))
        };
        Err(self.error(format!("has no argument {} — {known}.", unknown.join(", "))))
    }

    pub(super) fn optional_string(&self, key: &str) -> Result<Option<String>, ToolError> {
        match self.map.get(key) {
            None | Some(Value::Null) => Ok(None),
            Some(Value::String(s)) => Ok(Some(s.clone())),
            Some(other) => Err(self.wrong_type(key, "a string", other)),
        }
    }

    fn required_string(&self, key: &str) -> Result<String, ToolError> {
        self.optional_string(key)?
            .ok_or_else(|| self.error(format!("needs {key}.")))
    }

    pub(super) fn optional_bool(&self, key: &str) -> Result<Option<bool>, ToolError> {
        match self.map.get(key) {
            None | Some(Value::Null) => Ok(None),
            Some(Value::Bool(b)) => Ok(Some(*b)),
            Some(other) => Err(self.wrong_type(key, "true or false", other)),
        }
    }

    fn required_bool(&self, key: &str) -> Result<bool, ToolError> {
        self.optional_bool(key)?
            .ok_or_else(|| self.error(format!("needs {key}.")))
    }

    pub(super) fn optional_usize(&self, key: &str) -> Result<Option<usize>, ToolError> {
        match self.map.get(key) {
            None | Some(Value::Null) => Ok(None),
            Some(value) => value
                .as_u64()
                .map(|n| Some(n as usize))
                .ok_or_else(|| self.wrong_type(key, "a whole number, zero or more", value)),
        }
    }

    pub(super) fn required_usize(&self, key: &str) -> Result<usize, ToolError> {
        self.optional_usize(key)?
            .ok_or_else(|| self.error(format!("needs {key}.")))
    }

    fn optional_u64(&self, key: &str) -> Result<Option<u64>, ToolError> {
        match self.map.get(key) {
            None | Some(Value::Null) => Ok(None),
            Some(value) => value
                .as_u64()
                .map(Some)
                .ok_or_else(|| self.wrong_type(key, "a whole number, zero or more", value)),
        }
    }

    /// The actors a call named, or every actor where it named none.
    ///
    /// An empty array is refused rather than read as "everything": a call that
    /// can return nothing by construction has not asked a question, and reading
    /// it as its opposite would be the surface guessing.
    fn actors(&self, key: &str) -> Result<Vec<Actor>, ToolError> {
        let value = match self.map.get(key) {
            None | Some(Value::Null) => return Ok(Actor::ALL.to_vec()),
            Some(value) => value,
        };
        let array = value
            .as_array()
            .ok_or_else(|| self.wrong_type(key, "an array of actor names", value))?;
        if array.is_empty() {
            return Err(self.error(format!(
                "was given an empty {key}, which can return nothing — omit it for every actor, or \
                 name some of {}.",
                Actor::all_wire_names()
            )));
        }
        let mut actors = Vec::with_capacity(array.len());
        for element in array {
            let name = element
                .as_str()
                .ok_or_else(|| self.wrong_type(key, "an array of actor names", value))?;
            let actor = Actor::from_wire_name(name).ok_or_else(|| {
                self.error(format!(
                    "does not know the actor {name:?} — the actors are {}.",
                    Actor::all_wire_names()
                ))
            })?;
            if !actors.contains(&actor) {
                actors.push(actor);
            }
        }
        Ok(actors)
    }

    pub(super) fn optional_f64(&self, key: &str) -> Result<Option<f64>, ToolError> {
        match self.map.get(key) {
            None | Some(Value::Null) => Ok(None),
            Some(value) => value
                .as_f64()
                .map(Some)
                .ok_or_else(|| self.wrong_type(key, "a number", value)),
        }
    }

    pub(super) fn required_f64(&self, key: &str) -> Result<f64, ToolError> {
        self.optional_f64(key)?
            .ok_or_else(|| self.error(format!("needs {key}.")))
    }

    fn optional_vec3(&self, key: &str) -> Result<Option<[f64; 3]>, ToolError> {
        self.optional_numbers::<3>(key)
    }

    fn required_vec3(&self, key: &str) -> Result<[f64; 3], ToolError> {
        self.optional_vec3(key)?
            .ok_or_else(|| self.error(format!("needs {key}.")))
    }

    fn required_vec4(&self, key: &str) -> Result<[f64; 4], ToolError> {
        self.optional_numbers::<4>(key)?
            .ok_or_else(|| self.error(format!("needs {key}.")))
    }

    pub(super) fn optional_numbers<const N: usize>(
        &self,
        key: &str,
    ) -> Result<Option<[f64; N]>, ToolError> {
        let value = match self.map.get(key) {
            None | Some(Value::Null) => return Ok(None),
            Some(value) => value,
        };
        let expected = format!("an array of {N} numbers");
        let array = value
            .as_array()
            .ok_or_else(|| self.wrong_type(key, &expected, value))?;
        if array.len() != N {
            return Err(self.error(format!(
                "wants {key} to be {expected} — got {}.",
                array.len()
            )));
        }
        let mut out = [0.0; N];
        for (slot, element) in out.iter_mut().zip(array) {
            *slot = element
                .as_f64()
                .filter(|n| n.is_finite())
                .ok_or_else(|| self.wrong_type(key, &expected, value))?;
        }
        Ok(Some(out))
    }

    /// A pixel in a camera image, as the edits that take one want it.
    ///
    /// `f32` because that is what a `.sfmr` keypoint is and what every edit
    /// below this takes; the wire's number is `f64` and narrows here rather
    /// than in each tool body.
    pub(super) fn pixel(&self, key: &str) -> Result<[f32; 2], ToolError> {
        let [x, y] = self
            .optional_numbers::<2>(key)?
            .ok_or_else(|| self.error(format!("needs {key}: a pixel [x, y].")))?;
        Ok([x as f32, y as f32])
    }

    /// A pixel in a camera image, in the wire's own `f64`.
    ///
    /// Distinct from [`Self::pixel`], which narrows to the `f32` a `.sfmr`
    /// keypoint is: the patch tools read their pixel against a lens and a plane
    /// in `f64`, and narrowing on the way in would put a rounding of the
    /// caller's number between the ray and the answer.
    pub(super) fn required_pixel_f64(&self, key: &str) -> Result<[f64; 2], ToolError> {
        self.optional_numbers::<2>(key)?
            .ok_or_else(|| self.error(format!("needs {key}: a pixel [x, y].")))
    }

    /// Which edge of a patch's square, by its name on the wire.
    fn edge(&self, key: &str) -> Result<sfmtool_core::bench::Edge, ToolError> {
        let word = self.required_string(key)?;
        word.parse().map_err(|why: String| self.error(why))
    }

    /// [`Self::edge`] for the resize that may name none, where no edge means
    /// both of them about a held centre.
    fn optional_edge(&self, key: &str) -> Result<Option<sfmtool_core::bench::Edge>, ToolError> {
        match self.optional_string(key)? {
            None => Ok(None),
            Some(word) => word
                .parse()
                .map(Some)
                .map_err(|why: String| self.error(why)),
        }
    }

    /// A patch radius in pixels: positive, or absent for the viewer's own
    /// default.
    ///
    /// Zero and negative are refused rather than passed on, because a patch
    /// with no extent is not a smaller patch: it is a request the edit has no
    /// answer for, and the prompt the human uses cannot express it either.
    fn radius(&self, key: &str) -> Result<Option<f32>, ToolError> {
        match self.optional_f64(key)? {
            None => Ok(None),
            Some(radius) if radius.is_finite() && radius > 0.0 => Ok(Some(radius as f32)),
            Some(_) => Err(self.error(format!(
                "wants {key} to be a radius greater than zero, or absent for the median radius \
                 the image's own patches project to."
            ))),
        }
    }

    /// A `.sift` feature index, which is a `u32` because that is what a
    /// feature index is everywhere below this.
    fn optional_u32(&self, key: &str) -> Result<Option<u32>, ToolError> {
        match self.optional_u64(key)? {
            None => Ok(None),
            Some(index) => u32::try_from(index)
                .map(Some)
                .map_err(|_| self.wrong_type(key, "a feature index", &json!(index))),
        }
    }

    /// A 2x2 affine shape, row by row.
    fn optional_affine(&self, key: &str) -> Result<Option<[[f64; 2]; 2]>, ToolError> {
        let expected = "a 2x2 array of numbers";
        let value = match self.map.get(key) {
            None | Some(Value::Null) => return Ok(None),
            Some(value) => value,
        };
        let rows = value
            .as_array()
            .filter(|rows| rows.len() == 2)
            .ok_or_else(|| self.wrong_type(key, expected, value))?;
        let mut shape = [[0.0; 2]; 2];
        for (out, row) in shape.iter_mut().zip(rows) {
            let row = row
                .as_array()
                .filter(|row| row.len() == 2)
                .ok_or_else(|| self.wrong_type(key, expected, value))?;
            for (slot, element) in out.iter_mut().zip(row) {
                *slot = element
                    .as_f64()
                    .filter(|n| n.is_finite())
                    .ok_or_else(|| self.wrong_type(key, expected, value))?;
            }
        }
        Ok(Some(shape))
    }

    /// The observations a split names, by their positions in the track's list.
    fn observations(&self, key: &str) -> Result<Vec<usize>, ToolError> {
        let expected = "an array of observation indexes";
        let value = self
            .map
            .get(key)
            .ok_or_else(|| self.error(format!("needs {key}: {expected}.")))?;
        let array = value
            .as_array()
            .ok_or_else(|| self.wrong_type(key, expected, value))?;
        array
            .iter()
            .map(|element| {
                element
                    .as_u64()
                    .map(|index| index as usize)
                    .ok_or_else(|| self.wrong_type(key, expected, value))
            })
            .collect()
    }

    /// A verdict, in the three words the bench spells them with.
    fn verdict(&self, key: &str) -> Result<sfmtool_core::bench::Verdict, ToolError> {
        use sfmtool_core::bench::Verdict;
        match self.optional_string(key)?.as_deref() {
            Some("in") => Ok(Verdict::In),
            Some("out") => Ok(Verdict::Out),
            Some("candidate") => Ok(Verdict::Candidate),
            Some(other) => Err(self.error(format!(
                "does not know the verdict {other:?} — the verdicts are in, out and candidate."
            ))),
            None => Err(self.error(format!("needs {key} — one of in, out and candidate."))),
        }
    }

    /// A stage, in the two words the bench spells them with.
    fn stage(&self, key: &str) -> Result<sfmtool_core::bench::StageKind, ToolError> {
        use sfmtool_core::bench::StageKind;
        match self.optional_string(key)?.as_deref() {
            Some("cluster") => Ok(StageKind::Cluster),
            Some("track") => Ok(StageKind::Track),
            Some(other) => Err(self.error(format!(
                "does not know the stage {other:?} — the stages are cluster and track."
            ))),
            None => Err(self.error(format!("needs {key} — cluster or track."))),
        }
    }

    /// A panel argument, by the name the layout file spells it with.
    fn panel(&self, key: &str) -> Result<Tab, ToolError> {
        let name = self.optional_string(key)?.ok_or_else(|| {
            self.error(format!("needs {key} — one of {}.", Tab::all_wire_names()))
        })?;
        Tab::from_wire_name(&name).ok_or_else(|| {
            self.error(format!(
                "does not know the panel {name:?} — the panels are {}.",
                Tab::all_wire_names()
            ))
        })
    }

    /// A camera image argument, in either of its two spellings.
    pub(super) fn camera_image(&self, key: &str) -> Result<CameraImageSel, ToolError> {
        match self.map.get(key) {
            Some(Value::String(name)) => Ok(CameraImageSel::Name(name.clone())),
            Some(value) if value.as_u64().is_some() => Ok(CameraImageSel::Index(
                value.as_u64().expect("just checked") as usize,
            )),
            Some(value) => Err(self.wrong_type(key, "an image index or an image name", value)),
            None => Err(self.error(format!(
                "needs {key} — an index, or the image's .sfmr relative path."
            ))),
        }
    }

    /// A point argument, through the same parser the Go to Point dialog uses.
    ///
    /// A bare JSON integer is the index form spelled as a number rather than as
    /// a string, which is what a caller reading an index out of a track will
    /// naturally send; everything else goes to
    /// [`parse_point_query`], whose error messages already show both accepted
    /// shapes.
    pub(super) fn point(&self, key: &str) -> Result<PointQuery, ToolError> {
        match self.map.get(key) {
            Some(value) if value.as_u64().is_some() => Ok(PointQuery::Index(
                value.as_u64().expect("just checked") as usize,
            )),
            Some(Value::String(text)) => parse_point_query(text).map_err(ToolError),
            Some(value) => Err(self.wrong_type(key, "a point index or a point id", value)),
            None => Err(self.error(format!(
                "needs {key} — an index, or a pt3d_<hash>_<index> id."
            ))),
        }
    }
}

/// What a value is, for a message that has to say what arrived instead.
///
/// The kind and not the value: an argument that was wrong is usually long, and
/// a message that quotes the whole of it buries the part that matters.
fn describe(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "true or false",
        Value::Number(_) => "a number",
        Value::String(_) => "a string",
        Value::Array(_) => "an array",
        Value::Object(_) => "an object",
    }
}

/// Which of `translate_bench_patch`'s two forms a call made, refusing one that
/// made both or neither.
///
/// The refusal is here rather than in the schema because JSON Schema's `oneOf`
/// would report "does not match the schema" and leave the caller to work out
/// which half it got wrong; a sentence naming both spellings is what an agent
/// can act on.
fn translate_target(args: &Args<'_>) -> Result<super::TranslateTarget, ToolError> {
    let by = args.optional_vec3("by")?;
    let viewpoint = pixel_viewpoint(args)?;
    let pixel = args.optional_numbers::<2>("pixel")?;
    match (by, viewpoint, pixel) {
        (Some(by), None, None) => Ok(super::TranslateTarget::By(by)),
        (None, Some(viewpoint), Some(pixel)) => {
            Ok(super::TranslateTarget::Pixel { viewpoint, pixel })
        }
        (Some(_), _, _) => Err(args.error(
            "was given by as well as a pixel or the photograph it is in. A displacement on \
             the patch's own axes and a pixel of one photograph are two ways to say where the \
             patch goes, so give one."
                .to_string(),
        )),
        _ => Err(args.error(
            "needs either by, a displacement [u, v, n] on the patch's own axes in world units, \
             or a pixel with the photograph it is in, as observation or camera_image."
                .to_string(),
        )),
    }
}

/// The photograph a pixel form names, as exactly one of `observation` and
/// `camera_image`, or `None` when the call named neither.
///
/// Both at once is refused here, with a sentence naming the two, rather than
/// one quietly winning: they name different squares, the patch re-anchored on
/// a sighting and the patch as it stands, so a call carrying both has no one
/// answer.
fn pixel_viewpoint(args: &Args<'_>) -> Result<Option<super::ViewpointSel>, ToolError> {
    let observation = args.optional_usize("observation")?;
    let camera_image = match args.map.get("camera_image") {
        None | Some(Value::Null) => None,
        Some(_) => Some(args.camera_image("camera_image")?),
    };
    match (observation, camera_image) {
        (Some(_), Some(_)) => Err(args.error(
            "was given both observation and camera_image. The first reads the pixel against \
             the patch re-anchored on that sighting and the second against the patch as it \
             stands, so give one."
                .to_string(),
        )),
        (Some(observation), None) => Ok(Some(super::ViewpointSel::Observation(observation))),
        (None, Some(image)) => Ok(Some(super::ViewpointSel::CameraImage(image))),
        (None, None) => Ok(None),
    }
}

/// Which of `resize_bench_patch`'s two forms a call made, for the reason
/// [`translate_target`] is its own function.
fn resize_target(args: &Args<'_>) -> Result<super::ResizeTarget, ToolError> {
    let half_length = args.optional_f64("half_length")?;
    let viewpoint = pixel_viewpoint(args)?;
    let pixel = args.optional_numbers::<2>("pixel")?;
    match (half_length, viewpoint, pixel) {
        (Some(half_length), None, None) => Ok(super::ResizeTarget::HalfLength {
            half_length,
            moved_edge: args.optional_edge("moved_edge")?,
        }),
        (None, Some(viewpoint), Some(pixel)) => Ok(super::ResizeTarget::Pixel {
            viewpoint,
            edge: args.edge("edge")?,
            pixel,
        }),
        (Some(_), _, _) => Err(args.error(
            "was given half_length as well as a pixel or the photograph it is in. A world \
             half-length and an edge under a pixel are two ways to say how large the patch is, \
             so give one."
                .to_string(),
        )),
        _ => Err(args.error(
            "needs either half_length, a world half-length, or an edge and a pixel with the \
             photograph it is in, as observation or camera_image."
                .to_string(),
        )),
    }
}
