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
//! The catalog and the parser sit in one module because they are two halves of
//! one statement: [`catalog`] advertises what a tool accepts and [`parse`]
//! accepts it, and a test walks the pair so a schema and its parser cannot
//! drift.

use serde_json::{json, Map, Value};

use super::{
    CameraImageSel, CloseTarget, Command, DisplayChange, Placement, SelectionScope, ToolError,
    ViewCommand,
};
use crate::action_log::Actor;
use crate::dock::Tab;
use crate::goto_point::{parse_point_query, PointQuery};
use crate::window::WindowState;

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

/// Every tool this surface advertises, in the order `tools/list` reports them:
/// the reads first, then the writes, then the one that hands back a picture.
pub(crate) fn catalog() -> Vec<ToolSpec> {
    use ToolKind::{Read, Save, Write};
    vec![
        ToolSpec {
            name: "get_scene",
            description: "The whole scene graph: every loaded reconstruction with its counts and \
                          display state, the current selection, which reconstruction is soloed, \
                          the 3D viewport camera, and the window title. Call this first — the \
                          labels it reports are the handles every other tool takes. Counts only: \
                          no tool here returns point arrays or track tables in bulk, so read the \
                          .sfmr file itself (or `sfm inspect`) for data and ask the viewer for \
                          state.",
            kind: Read,
            schema: object(&[], &[]),
        },
        ToolSpec {
            name: "list_camera_images",
            description: "One reconstruction's camera images — index, name, the intrinsics record \
                          each was shot through, camera centre, and observation count — a page at \
                          a time.",
            kind: Read,
            schema: object(
                &[
                    ("reconstruction_label", reconstruction_label_schema()),
                    (
                        "offset",
                        json!({
                            "type": "integer",
                            "minimum": 0,
                            "description": "First image index to return. Defaults to 0.",
                        }),
                    ),
                    (
                        "limit",
                        json!({
                            "type": "integer",
                            "minimum": 1,
                            "maximum": super::read::MAX_LIMIT,
                            "description":
                                "How many images to return. Defaults to 50, capped at 500.",
                        }),
                    ),
                ],
                &[],
            ),
        },
        ToolSpec {
            name: "get_camera_image",
            description: "One camera image: its pose (world-to-camera quaternion and translation, \
                          plus the camera centre those imply), the intrinsics record it was shot \
                          through, how many observations it carries, and a reprojection-error \
                          summary. The error summary is null when it cannot be computed — an \
                          embedded-patches reconstruction has no .sift file to read features \
                          from.",
            kind: Read,
            schema: object(
                &[("reconstruction_label", reconstruction_label_schema())],
                &[("camera_image", camera_image_schema())],
            ),
        },
        ToolSpec {
            name: "get_camera_intrinsics",
            description: "One camera intrinsics record — the lens: model, sensor size, every \
                          stored parameter by name, and the camera images that use it. The \
                          parameters are a name-to-value map in the model's own declaration \
                          order, which is the order `sfm inspect` prints.",
            kind: Read,
            schema: object(
                &[("reconstruction_label", reconstruction_label_schema())],
                &[(
                    "camera_intrinsics_index",
                    json!({
                        "type": "integer",
                        "minimum": 0,
                        "description":
                            "Index of the intrinsics record. An intrinsics record has no name to \
                             address it by; get_camera_image reports the index each image uses.",
                    }),
                )],
            ),
        },
        ToolSpec {
            name: "get_point",
            description: "One 3D point: position, colour, RMS error, whether it is a point at \
                          infinity, and its full track — every observing camera image with the \
                          pixel it was seen at and that observation's reprojection error.",
            kind: Read,
            schema: object(&[], &[("point", point_schema())]),
        },
        ToolSpec {
            name: "get_action_log",
            description: "What has happened in the viewer, oldest first — the human's selections \
                          and file loads, the agent's own calls, and every refusal, each with the \
                          revision it was written at. Pass the revision from a previous reply back \
                          as since_revision to read only what has happened since; the log's \
                          revision goes up on every entry and on every fold of a run of like \
                          entries into one, so a line that changed is reported again. \
                          oldest_revision says how far back the log still goes: a since_revision \
                          below it means entries were missed. actors filters by who did it, and \
                          actors: [\"user\"] is the read that answers \"what did the human do while \
                          I was working\". An entry carries took_ms once the viewer has drawn the \
                          frame that showed its result, and detail: true adds the stage-by-stage \
                          breakdown of where that time went.",
            kind: Read,
            schema: object(
                &[
                    (
                        "since_revision",
                        json!({
                            "type": "integer",
                            "minimum": 0,
                            "description":
                                "Return entries written or changed after this revision. Defaults \
                                 to 0, the start of the log.",
                        }),
                    ),
                    (
                        "limit",
                        json!({
                            "type": "integer",
                            "minimum": 1,
                            "maximum": super::read::ACTION_LOG_MAX_LIMIT,
                            "description":
                                "How many entries to return. Defaults to 200, capped at 1000; \
                                 truncated in the reply says there are more, and the last entry's \
                                 revision is where to continue from.",
                        }),
                    ),
                    (
                        "actors",
                        json!({
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": crate::action_log::Actor::ALL.map(|a| a.wire_name()),
                            },
                            "minItems": 1,
                            "description":
                                "Which actors' entries to return. Omit for all of them; an empty \
                                 array is refused, since it can return nothing by construction.",
                        }),
                    ),
                    (
                        "detail",
                        flag(
                            "Carry each entry's breakdown: the stages it spent its time in and \
                             what it said, in the order the Action Log panel draws them, with \
                             elsewhere_ms beside took_ms for the time no stage claimed. Off by \
                             default, since a breakdown is several times the size of the row it \
                             hangs off. The read this is for: find a slow row, then ask again \
                             with detail set and since_revision just below that row's revision. \
                             It reports what was recorded, which is what set_timing_detail \
                             decided at the time.",
                        ),
                    ),
                ],
                &[],
            ),
        },
        ToolSpec {
            name: "get_timing_detail",
            description: "Whether the viewer is recording the finer stages inside each \
                          operation: the Action Log toolbar's Detailed timing checkbox, wherever \
                          it was last left. Off by default, and it lives only as long as the \
                          session.",
            kind: Read,
            schema: object(&[], &[]),
        },
        ToolSpec {
            name: "get_window_layout",
            description: "Where the window is and how its panels are arranged, as one document — \
                          the layout file the Panels menu saves and the viewer reads at startup, \
                          which set_window_layout takes back unchanged. Beside it: the live \
                          window block (state, focus, scale factor, current position and sizes in \
                          physical pixels, and every monitor, the current one first) and one \
                          entry per panel saying whether it is open and whether it is the front \
                          tab of its node. The document's window section is the *normal* \
                          rectangle — what the window restores to — so it and the live block \
                          differ for a maximized window, on purpose.",
            kind: Read,
            schema: object(&[], &[]),
        },
        ToolSpec {
            name: "get_image_detail_display",
            description: "What the Image Detail panel draws over its photograph: the feature \
                          overlay mode, the three filters on the features, and the intrinsics \
                          layer with its own sub-toggles, as one document. These are scene-level \
                          controls rather than properties of any image, so they decide what a \
                          screenshot of the image_detail panel shows whichever camera image is \
                          selected.",
            kind: Read,
            schema: object(&[], &[]),
        },
        ToolSpec {
            name: "get_image_detail_view",
            description: "Where the Image Detail panel is looking: the camera image on screen, \
                          the zoom (1.0 is the whole photograph fitted to the panel), the \
                          rectangle of the photograph the panel shows in its own pixels, the \
                          panel's size in points and the photograph's size in pixels. This is \
                          what the last drawn frame settled on, so it is what a screenshot of \
                          the panel would show — a select_camera_image sent a moment ago is not \
                          in it until the next frame. Every field is null before the panel has \
                          drawn a photograph at all.",
            kind: Read,
            schema: object(&[], &[]),
        },
        ToolSpec {
            name: "get_history",
            description: "One reconstruction's versions, oldest first: every edit anyone has \
                          made to it this session, with the sentence the edit recorded as each \
                          version's label and the time it was made. cursor names the version the \
                          viewer is showing, disk_serial the one its file holds (null when it \
                          came from no file), and dirty says the two differ. A version whose \
                          value the history budget released is listed with held: false: what \
                          happened there is still known, but the cursor can no longer go to it, \
                          and neither can jump_to_version.",
            kind: Read,
            schema: object(&[], &[("reconstruction_label", edited_label_schema())]),
        },
        ToolSpec {
            name: "get_bench",
            description: "The bench beside one reconstruction: every item on it by label, with \
                          its kind, the stage it is at, the point it came from where it came \
                          from one, its observation and verdict counts, and which item of each \
                          kind is active. The bench is a place beside the reconstruction rather \
                          than part of it — nothing on it is saved, and a commit is how it \
                          reaches the file.",
            kind: Read,
            schema: object(&[], &[("reconstruction_label", edited_label_schema())]),
        },
        ToolSpec {
            name: "get_bench_track",
            description: "One track on the bench, as its Track Edit table: the stage, the point \
                          it came from, the thresholds, and every observation with what put it \
                          there, the verdict on it and whatever each stage has measured about \
                          it. An observation is addressed by its position in the list, which is \
                          stable for the life of the track — observations are appended and never \
                          renumbered, so an index read here still names the same observation \
                          after a verdict or an evaluation.",
            kind: Read,
            schema: object(
                &[("track", bench_track_schema())],
                &[("reconstruction_label", edited_label_schema())],
            ),
        },
        ToolSpec {
            name: "open_reconstruction",
            description: "Load an .sfmr file into the scene as a new reconstruction, and select \
                          it. Opening a path that is already open adds a second node for it, \
                          with a history of its own; `already_open` says whether that happened. \
                          Read the returned `label` back rather \
                          than assuming it — a colliding file stem is disambiguated as \
                          \"name (2)\".",
            kind: Write,
            schema: object(
                &[],
                &[(
                    "path",
                    json!({
                        "type": "string",
                        "description": "Path to an .sfmr file, as the viewer's process can see it.",
                    }),
                )],
            ),
        },
        ToolSpec {
            name: "close_reconstruction",
            description: "Unload one reconstruction, or all of them. This removes it from the \
                          viewer; it does not delete or modify any file.",
            kind: Write,
            schema: json!({
                "type": "object",
                "description":
                    "Name one reconstruction to close, or pass all: true to clear the scene.",
                "properties": {
                    "reconstruction_label": {
                        "type": "string",
                        "description": "The reconstruction to close.",
                    },
                    "all": {
                        "type": "boolean",
                        "description": "Close every loaded reconstruction.",
                    },
                },
                "additionalProperties": false,
            }),
        },
        ToolSpec {
            name: "select_reconstruction",
            description: "Make one reconstruction the one the file- and sequence-shaped panels \
                          follow. Selections belonging to other reconstructions are dropped, \
                          which is the invariant that stops two panels showing two different \
                          files' selections.",
            kind: Write,
            schema: object(
                &[],
                &[(
                    "reconstruction_label",
                    json!({ "type": "string", "description": "The reconstruction to select." }),
                )],
            ),
        },
        ToolSpec {
            name: "select_camera_image",
            description: "Select a camera image — and with it the intrinsics record it was shot \
                          through, and the reconstruction that owns it. The selected image's \
                          frustum is drawn in cyan, so this is visible in a screenshot.",
            kind: Write,
            schema: object(
                &[("reconstruction_label", reconstruction_label_schema())],
                &[("camera_image", camera_image_schema())],
            ),
        },
        ToolSpec {
            name: "select_camera_intrinsics",
            description: "Select a camera intrinsics record. Clears the selected camera image \
                          unless that image uses these intrinsics.",
            kind: Write,
            schema: object(
                &[("reconstruction_label", reconstruction_label_schema())],
                &[(
                    "camera_intrinsics_index",
                    json!({
                        "type": "integer",
                        "minimum": 0,
                        "description": "Index of the intrinsics record to select.",
                    }),
                )],
            ),
        },
        ToolSpec {
            name: "select_point",
            description: "Select a 3D point, and with it the reconstruction that owns it. A \
                          qualified pt3d_<hash>_<index> id names its own reconstruction, so this \
                          can move the selection to a different one. The selected point's track \
                          rays are drawn in orange.",
            kind: Write,
            schema: object(&[], &[("point", point_schema())]),
        },
        ToolSpec {
            name: "clear_selection",
            description: "Drop the selection, wholly or one kind of it. Selection is visible — a \
                          selected image tints its frustum, a selected point draws its track — so \
                          this is how to get a clean render before a screenshot. Clearing just \
                          the camera image keeps its intrinsics selected: dismissing a photograph \
                          says nothing about the lens.",
            kind: Write,
            schema: object(
                &[(
                    "scope",
                    json!({
                        "type": "string",
                        "enum": ["all", "camera_image", "camera_intrinsics", "point"],
                        "description": "How much to clear. Defaults to all.",
                    }),
                )],
                &[],
            ),
        },
        ToolSpec {
            name: "set_reconstruction_display",
            description: "Change how one reconstruction is drawn: its master eye, whether pointer \
                          picks reach it, the per-group eyes, and its comparison tint. Every \
                          field is optional and every omitted one is left alone. None of this \
                          touches the reconstruction's data.",
            kind: Write,
            schema: object(
                &[
                    ("visible", flag("Master eye. Off draws nothing of it.")),
                    (
                        "interactive",
                        flag(
                            "Whether pointer hover and click-pick in the 3D viewport reach this \
                             reconstruction. Off is display-only: it still renders and occludes.",
                        ),
                    ),
                    ("show_points", flag("Group eye: the 3D points.")),
                    (
                        "show_camera_images",
                        flag("Group eye: the camera frustums and image quads."),
                    ),
                    ("show_patches", flag("Group eye: the patch surfels.")),
                    (
                        "show_points_at_infinity",
                        flag("Sub-toggle of show_points: the w = 0 directions."),
                    ),
                    (
                        "tint",
                        json!({
                            "type": ["string", "null"],
                            "enum": tint_names_with_null(),
                            "description":
                                "A comparison tint mixed into everything this reconstruction \
                                 draws, from a fixed colour-blind-safe palette, or null for its \
                                 own colours.",
                        }),
                    ),
                ],
                &[(
                    "reconstruction_label",
                    json!({ "type": "string", "description": "The reconstruction to restyle." }),
                )],
            ),
        },
        ToolSpec {
            name: "set_solo",
            description: "Draw only one reconstruction, or end the solo. At most one is soloed at \
                          a time and soloing a second moves the solo. Solo is independent of \
                          selection and never writes the per-reconstruction eyes, so ending it \
                          restores exactly the visibility that was set by hand.",
            kind: Write,
            schema: object(
                &[(
                    "reconstruction_label",
                    json!({
                        "type": ["string", "null"],
                        "description":
                            "The reconstruction to draw alone, or null to end the solo.",
                    }),
                )],
                &[],
            ),
        },
        ToolSpec {
            name: "set_image_detail_display",
            description: "Change what the Image Detail panel draws over its photograph. Every \
                          field is optional at either level and every omitted one is left alone; \
                          the reply is the whole document, as get_image_detail_display returns \
                          it. The panel does not have to be open — this is what it will show when \
                          show_panel opens it — and nothing here selects an image: \
                          select_camera_image chooses the photograph the overlay is drawn on. An \
                          unknown overlay_mode, an off-ladder distortion_scale or grid_cols, a \
                          max_features below 1 and a feature_size_px whose min exceeds its max \
                          are refused whole, leaving the call's other fields unapplied.",
            kind: Write,
            schema: object(
                &[
                    (
                        "overlay_mode",
                        json!({
                            "type": "string",
                            "enum": crate::state::OverlayMode::ALL.map(|mode| mode.wire_name()),
                            "description":
                                "Which feature overlay to draw: \"none\" for the clean \
                                 photograph, \"features\" for the keypoints themselves, and the \
                                 five heatmaps for one metric each over the tracked features.",
                        }),
                    ),
                    (
                        "max_features",
                        json!({
                            "type": ["integer", "null"],
                            "minimum": 1,
                            "description":
                                "Show at most this many features per image — the largest ones, \
                                 since features are stored largest first — or null for all of \
                                 them. 0 is refused: \"no features\" is overlay_mode \"none\".",
                        }),
                    ),
                    ("feature_size_px", super::display::feature_size_schema()),
                    (
                        "tracked_only",
                        flag(
                            "Show only features with a 3D point behind them, which is the CLI's \
                             --filter-sfm.",
                        ),
                    ),
                    ("intrinsics", super::display::intrinsics_schema()),
                ],
                &[],
            ),
        },
        ToolSpec {
            name: "set_image_detail_view",
            description: "Point the Image Detail panel at one place in one photograph — the 2D \
                          counterpart of set_view, and the tool to call before a screenshot of \
                          the image_detail panel. Exactly one target per call: pixel centres a \
                          place, rect fits a region, point centres a 3D point's observation in \
                          the photograph being looked at, feature centres a .sift feature of it, \
                          bench_observation centres one sighting of a bench track (and selects \
                          the camera image it is in), and fit shows the whole photograph. zoom \
                          is absolute, 1.0 being the fit, and applies to every target but rect \
                          and fit, which settle their own; it is clamped to the panel's range \
                          and the reply says where it landed. The reply is get_image_detail_view's \
                          document for the view that will be applied on the next frame.",
            kind: Write,
            schema: object(
                &[
                    ("reconstruction_label", reconstruction_label_schema()),
                    (
                        "camera_image",
                        json!({
                            "type": ["integer", "string"],
                            "description":
                                "The camera image to look at, by index or by its .sfmr relative \
                                 path. Selected first, as select_camera_image would. Omitted, \
                                 the target's own camera image is used where it names one \
                                 (bench_observation), and the selected one otherwise.",
                        }),
                    ),
                    (
                        "pixel",
                        json!({
                            "type": "array",
                            "items": { "type": "number" },
                            "minItems": 2,
                            "maxItems": 2,
                            "description":
                                "[x, y] in the photograph's own pixels, brought to the centre of \
                                 the panel.",
                        }),
                    ),
                    (
                        "rect",
                        json!({
                            "type": "array",
                            "items": { "type": "number" },
                            "minItems": 4,
                            "maxItems": 4,
                            "description":
                                "[x0, y0, x1, y1] in the photograph's own pixels: zoom so this \
                                 rectangle fills the panel, centred. It settles its own zoom, so \
                                 a zoom argument beside it is refused.",
                        }),
                    ),
                    (
                        "point",
                        json!({
                            "type": ["integer", "string"],
                            "description":
                                "A 3D point, by index or by a pt3d_<hash>_<index> id: centre its \
                                 observation in the camera image being looked at. Refused when \
                                 its track holds no sighting there — this never chooses the \
                                 camera image for you.",
                        }),
                    ),
                    (
                        "feature",
                        json!({
                            "type": "integer",
                            "minimum": 0,
                            "description":
                                "A .sift feature of the camera image being looked at, by its \
                                 index in that file: centre it. Read through the same cache the \
                                 panel draws its ellipses from.",
                        }),
                    ),
                    (
                        "bench_observation",
                        json!({
                            "type": "integer",
                            "minimum": 0,
                            "description":
                                "One observation of a bench track, by its position in \
                                 get_bench_track's list: centre where it sits, and select the \
                                 camera image it is a sighting in.",
                        }),
                    ),
                    ("track", bench_track_schema()),
                    (
                        "fit",
                        flag(
                            "True for the whole photograph: zoom 1.0, centred — what Z and a \
                             double-click in the panel do. It settles its own zoom, so a zoom \
                             argument beside it is refused.",
                        ),
                    ),
                    (
                        "zoom",
                        json!({
                            "type": "number",
                            "exclusiveMinimum": 0,
                            "description":
                                "Absolute magnification, 1.0 being the whole photograph fitted to \
                                 the panel and 32.0 the closest the panel goes. Clamped to that \
                                 range, and the reply reports the zoom that was applied.",
                        }),
                    ),
                ],
                &[],
            ),
        },
        ToolSpec {
            name: "set_timing_detail",
            description: "Record the finer stages inside each operation, or stop recording them. \
                          This is the level the next operation is timed at; get_action_log's \
                          detail argument asks for what was already recorded. The case it exists \
                          for: an agent that has found a slow row turns detail on, runs the \
                          operation again, reads the log back with detail set, and turns it off. \
                          It takes effect on the next operation, so nothing already recorded is \
                          re-timed and an entry keeps the detail it was recorded with. It is the \
                          Action Log toolbar's Detailed timing checkbox, and setting it writes \
                          the same Action Log entry ticking that checkbox does, so the human at \
                          the window can see that an agent raised the level.",
            kind: Write,
            schema: object(
                &[],
                &[(
                    "enabled",
                    flag("True to record the detailed stages, false to record only the overview."),
                )],
            ),
        },
        ToolSpec {
            name: "set_view",
            description: "Move the 3D viewport camera — the tool to call immediately before \
                          screenshot. Five forms, exactly one per call: frame everything or one \
                          reconstruction (fit), look through a camera image (look_through), leave \
                          camera view (exit_camera_view), place the explicit camera, or set \
                          fov_short_axis_deg alone. The explicit camera takes its pieces one at a \
                          time and preserves what a call does not carry: position with target is \
                          the look-at form and orientation_wxyz with target_distance restores a \
                          view read from get_scene, while target alone re-centres the view, \
                          forward alone orbits the camera around what it is looking at, and \
                          target_distance alone dollies. fov_short_axis_deg may ride along with \
                          any of them. View changes jump rather than animating, so a screenshot \
                          taken straight afterward shows the new view.",
            kind: Write,
            schema: set_view_schema(),
        },
        ToolSpec {
            name: "set_window_layout",
            description: "Place the window, arrange the panels, or both: the argument is a \
                          layout document, in the shape get_window_layout returns and the Panels \
                          menu saves, so a whole reply — or a file a human saved — can be sent \
                          back unedited. Both sections are optional and a call must carry one. \
                          The window section is applied first: its rectangle is the window's \
                          *normal* rectangle, so a size sent to a maximized window changes what \
                          it restores to and leaves it maximized, and what the section does not \
                          carry is preserved. The layout section replaces the whole arrangement \
                          — a panel it does not mention is closed, though every panel keeps its \
                          own state — or is the string \"default\" for the stock seven-panel \
                          grid. A document that does not validate is refused whole, naming what \
                          was wrong and where, and nothing is applied.",
            kind: Write,
            schema: object(
                &[
                    (
                        "sfm_explorer_layout",
                        json!({
                            "type": "integer",
                            "description":
                                "The document version, 2. Optional: send it when passing a file \
                                 or a whole get_window_layout reply back unedited.",
                        }),
                    ),
                    ("window", window_section_schema()),
                    (
                        "layout",
                        json!({
                            "description":
                                "The panel arrangement, as get_window_layout's document carries \
                                 one, or \"default\" for the stock arrangement.",
                            "anyOf": [
                                { "type": "object" },
                                { "type": "string", "enum": ["default"] },
                            ],
                        }),
                    ),
                ],
                &[],
            ),
        },
        ToolSpec {
            name: "show_panel",
            description: "Open a panel at its home position, or — if it is already open — raise \
                          it, making it the front tab of its node without moving anything. Where \
                          an opened panel lands is the viewer's own home rule; send a layout \
                          document through set_window_layout to put one somewhere specific.",
            kind: Write,
            schema: object(&[], &[("panel_name", panel_name_schema())]),
        },
        ToolSpec {
            name: "hide_panel",
            description: "Close a panel. Hiding one that is already closed succeeds and changes \
                          nothing. The panel keeps its state while it is closed, and show_panel \
                          brings it back.",
            kind: Write,
            schema: object(&[], &[("panel_name", panel_name_schema())]),
        },
        ToolSpec {
            name: "undo",
            description: "Step one reconstruction's history back a version, exactly as the Edit \
                          menu's Undo does. Refused when it is already on its first version. The \
                          reply is the version now showing.",
            kind: Write,
            schema: object(&[], &[("reconstruction_label", edited_label_schema())]),
        },
        ToolSpec {
            name: "redo",
            description: "Step one reconstruction's history forward a version. Refused when \
                          there is nothing ahead of the cursor, since an edit made after an undo \
                          discards what was ahead. The reply is the version now showing.",
            kind: Write,
            schema: object(&[], &[("reconstruction_label", edited_label_schema())]),
        },
        ToolSpec {
            name: "jump_to_version",
            description: "Move one reconstruction's cursor straight to a version, which is what \
                          clicking a row of the Edit History panel does. The move is the run of \
                          undos or redos between the two, so the selection lands where stepping \
                          would have put it, and it is refused whole when any version on the way, \
                          the destination included, has had its value released.",
            kind: Write,
            schema: object(
                &[],
                &[
                    ("reconstruction_label", edited_label_schema()),
                    (
                        "serial",
                        json!({
                            "type": "string",
                            "description":
                                "Which version, spelled as get_history and the Action Log spell \
                                 it: \"v12\".",
                        }),
                    ),
                ],
            ),
        },
        ToolSpec {
            name: "save_reconstruction",
            description: "Write one reconstruction to disk. With no path it is written over the \
                          file it came from, and a reconstruction that came from no file is \
                          refused; with a path it is written there and the node is re-pointed at \
                          it, taking that file's name as its label, so read the reply's \
                          reconstruction_label back before the next call. The version written is \
                          the one at the cursor, and it becomes the version the history calls \
                          clean.",
            kind: Save,
            schema: object(
                &[(
                    "path",
                    json!({
                        "type": "string",
                        "description":
                            "Where to write it, as the viewer's process can see it. Omit to \
                             write over the file the reconstruction came from.",
                    }),
                )],
                &[("reconstruction_label", edited_label_schema())],
            ),
        },
        ToolSpec {
            name: "delete_point",
            description: "Delete one 3D point and its whole track. A point edit: every other \
                          point keeps the index it had, so indexes an agent is holding stay \
                          good, and undo puts it back.",
            kind: Write,
            schema: object(
                &[],
                &[
                    ("reconstruction_label", edited_label_schema()),
                    ("point", point_schema()),
                ],
            ),
        },
        ToolSpec {
            name: "delete_camera_image",
            description: "Delete one camera image, its observations, and any track left with \
                          none. A bulk edit: every image index at or after the deleted one moves \
                          down by one and the surviving points are renumbered, so indexes read \
                          before the call no longer mean what they meant. Deleting the only image \
                          is refused.",
            kind: Write,
            schema: object(
                &[],
                &[
                    ("reconstruction_label", edited_label_schema()),
                    ("camera_image", camera_image_schema()),
                ],
            ),
        },
        ToolSpec {
            name: "add_observation",
            description: "Add one observation of a 3D point to a camera image that does not \
                          already see it, at a named pixel. The pixel is a starting point: the \
                          embed pass's photometric kernel places the keypoint from there and the \
                          track is re-triangulated, and the reply's report says how far it moved \
                          and how well it matched. Needs an embedded_patches reconstruction, one \
                          whose observations carry inline keypoints, and the photographs, which \
                          are decoded on demand.",
            kind: Write,
            schema: object(
                &[],
                &[
                    ("reconstruction_label", edited_label_schema()),
                    ("point", point_schema()),
                    ("camera_image", camera_image_schema()),
                    ("pixel", pixel_schema()),
                ],
            ),
        },
        ToolSpec {
            name: "create_point",
            description: "Create a 3D point at a pixel of one camera image. The point is made at \
                          infinity along that pixel's ray, since one sighting fixes a bearing and no \
                          distance, with a one-observation track, its colour read from the \
                          photograph and a patch of the named radius; add_observation in a second \
                          image is what brings it to a finite depth. Needs an embedded_patches \
                          reconstruction.",
            kind: Write,
            schema: object(
                &[(
                    "radius_px",
                    json!({
                        "type": "number",
                        "exclusiveMinimum": 0,
                        "description":
                            "The patch's radius in this image's pixels. Omit for the radius the \
                             viewer's own prompt would offer: the median radius the image's \
                             existing patches project to.",
                    }),
                )],
                &[
                    ("reconstruction_label", edited_label_schema()),
                    ("camera_image", camera_image_schema()),
                    ("pixel", pixel_schema()),
                ],
            ),
        },
        ToolSpec {
            name: "remove_observation",
            description: "Remove one camera image's observation from a point's track, and \
                          re-triangulate what is left. The reply's report says what became of the \
                          point: fewer observations, a bearing at infinity where one sighting is \
                          left, or deleted where none is.",
            kind: Write,
            schema: object(
                &[],
                &[
                    ("reconstruction_label", edited_label_schema()),
                    ("point", point_schema()),
                    ("camera_image", camera_image_schema()),
                ],
            ),
        },
        ToolSpec {
            name: "move_camera_image",
            description: "Put one camera image at a pose, as one version of its reconstruction. \
                          The pose is world-from-camera in the reconstruction's own frame: \
                          quaternion_wxyz carries camera axes onto world axes and translation is \
                          the camera centre, which is what get_camera_image reports as center. \
                          The tracks that image observes are re-triangulated around the new pose \
                          where two or more pixels see them; a bearing only it sees turns with \
                          it, and anything else keeps its position. The image table stays put and \
                          nothing is renumbered. Undo (Ctrl+Z in the window) puts the stored pose \
                          back.",
            kind: Write,
            schema: object(
                &[],
                &[
                    ("reconstruction_label", edited_label_schema()),
                    ("camera_image", camera_image_schema()),
                    (
                        "world_from_camera",
                        json!({
                            "type": "object",
                            "description":
                                "Where to put the camera, in the reconstruction's own frame.",
                            "properties": {
                                "quaternion_wxyz": {
                                    "type": "array",
                                    "items": { "type": "number" },
                                    "minItems": 4,
                                    "maxItems": 4,
                                    "description":
                                        "Camera-to-world rotation, WXYZ. Normalised on arrival.",
                                },
                                "translation": vec3_schema(
                                    "The camera centre in world coordinates.",
                                ),
                            },
                            "required": ["quaternion_wxyz", "translation"],
                            "additionalProperties": false,
                        }),
                    ),
                ],
            ),
        },
        ToolSpec {
            name: "resect_camera_image_in_place",
            description: "Re-estimate one camera image's pose against structure held out from \
                          it, and install the answer as the reconstruction's next version rather \
                          than as a derived node beside it. A bulk edit: the points the image \
                          observes are re-triangulated and the surviving ones are renumbered, \
                          while the image table stays put. A refused estimate pushes no version.",
            kind: Write,
            schema: object(
                &[(
                    "from_matches",
                    flag(
                        "Estimate against a .matches file rather than against the \
                         reconstruction's own observations. The file is the one already chosen \
                         for this reconstruction in the viewer; with none chosen the call is \
                         refused, since this surface opens no file dialog.",
                    ),
                )],
                &[
                    ("reconstruction_label", edited_label_schema()),
                    ("camera_image", camera_image_schema()),
                ],
            ),
        },
        ToolSpec {
            name: "bundle_adjust",
            description: "Refine every pose and every point of one reconstruction against its \
                          observations, as one version. A bulk edit, and it runs on a worker \
                          thread, so the window stays live while it solves. An adjustment that \
                          finishes quickly replies with the version it pushed and a report \
                          carrying the counts and the median residual before and after; one that \
                          is still going after 200 ms replies instead with running: true and an \
                          operation_id, and the outcome is then read out of get_action_log or \
                          stopped with cancel_background_task. Needs inline keypoints and one shared \
                          lens.",
            kind: Write,
            schema: object(
                &[(
                    "release_focal",
                    flag(
                        "Solve the shared focal length as well as the poses and points. Defaults \
                         to false, which holds it where it is.",
                    ),
                )],
                &[("reconstruction_label", edited_label_schema())],
            ),
        },
        ToolSpec {
            name: "create_bench_cluster",
            description: "Start a cluster-stage track on the bench from a place in one camera \
                          image, and make it the active track. A cluster is a set of image \
                          patches that register onto one template, with no geometry behind them: \
                          it wants more observations and an evaluation, and set_bench_track_stage \
                          \"track\" is what triangulates it. Seed it with a pixel (and a \
                          radius_px or an affine shape for the patch), or with a .sift feature, \
                          which carries its own position and shape. The reply names the item the \
                          rest of the bench tools take.",
            kind: Write,
            schema: object(
                &seed_properties(),
                &[
                    ("reconstruction_label", edited_label_schema()),
                    ("camera_image", camera_image_schema()),
                ],
            ),
        },
        ToolSpec {
            name: "create_bench_track",
            description: "Put one 3D point of the reconstruction on the bench as a track-stage \
                          track, and make it the active track, so its observations can be judged \
                          one at a time and the result committed back over the point. Putting on \
                          a point a track already came from activates that track rather than \
                          putting a second one on. The reply names the item.",
            kind: Write,
            schema: object(
                &[],
                &[
                    ("reconstruction_label", edited_label_schema()),
                    ("point", point_schema()),
                ],
            ),
        },
        ToolSpec {
            name: "activate_bench_item",
            description: "Make one item on the bench the active one of its kind, which is the \
                          item the Track Edit panel shows and the item a bench tool acts on when \
                          it names none.",
            kind: Write,
            schema: object(
                &[],
                &[
                    ("reconstruction_label", edited_label_schema()),
                    ("item", bench_item_schema()),
                ],
            ),
        },
        ToolSpec {
            name: "rename_bench_item",
            description: "Give one item on the bench a label of your own, which is what names it \
                          in every later call, in the Scene tree and in each Action Log row. The \
                          reply carries the new label. A label another item already holds is \
                          refused.",
            kind: Write,
            schema: object(
                &[],
                &[
                    ("reconstruction_label", edited_label_schema()),
                    ("item", bench_item_schema()),
                    (
                        "label",
                        json!({
                            "type": "string",
                            "description":
                                "The label the item should take. Unique on this bench, and \
                                 something other than whitespace.",
                        }),
                    ),
                ],
            ),
        },
        ToolSpec {
            name: "discard_bench_item",
            description: "Take one item off the bench. It is a version like any other step, so \
                          undo puts the item back where it was and active as it was; nothing of \
                          the reconstruction is touched, since nothing on the bench is part of \
                          it.",
            kind: Write,
            schema: object(
                &[],
                &[
                    ("reconstruction_label", edited_label_schema()),
                    ("item", bench_item_schema()),
                ],
            ),
        },
        ToolSpec {
            name: "add_bench_track_observation",
            description: "Add one candidate observation of a bench track, in a camera image, at \
                          the place the seed names — a pixel, a pixel with a radius_px or an \
                          affine shape, or a .sift feature. It joins as a candidate and unpinned: \
                          something proposed it and nobody has ruled on it. A second observation \
                          in an image the track already sees is allowed and is measured like any \
                          other; what it cannot do is be turned in while the other is. The reply \
                          names the index it took.",
            kind: Write,
            schema: object(
                &{
                    let mut optional = seed_properties();
                    optional.push(("track", bench_track_schema()));
                    optional
                },
                &[
                    ("reconstruction_label", edited_label_schema()),
                    ("camera_image", camera_image_schema()),
                ],
            ),
        },
        ToolSpec {
            name: "set_bench_track_verdict",
            description: "Rule on one observation of a bench track by hand: in, out, or back to \
                          candidate. A verdict set this way is pinned, which is what leaves it \
                          alone when the thresholds are applied. A track cannot see one image \
                          twice, so turning an observation in while another observation of the \
                          same image is in is refused. Setting the verdict an observation already \
                          has changes nothing and pushes no version.",
            kind: Write,
            schema: object(
                &[("track", bench_track_schema())],
                &[
                    ("reconstruction_label", edited_label_schema()),
                    ("observation", observation_schema()),
                    (
                        "verdict",
                        json!({
                            "type": "string",
                            "enum": ["in", "out", "candidate"],
                            "description":
                                "in: the observation belongs to the track, and a commit writes \
                                 it. out: it was refused, and stays in the list so the refusal \
                                 is visible. candidate: proposed and not ruled on.",
                        }),
                    ),
                ],
            ),
        },
        ToolSpec {
            name: "apply_bench_track_thresholds",
            description: "Set a bench track's bars and turn them into verdicts in one step: \
                          every unpinned observation is judged against them and takes the verdict \
                          they propose, and an observation ruled on by hand is left alone. A bar \
                          the call does not name stays where the track has it. The reply's report \
                          says how many were turned in, turned out, left pinned and left \
                          unmeasured.",
            kind: Write,
            schema: object(
                &[
                    ("track", bench_track_schema()),
                    (
                        "min_zncc",
                        threshold_schema(
                            "The ZNCC an observation has to reach: the achieved template ZNCC at \
                             the cluster stage, the leave-one-out ZNCC at the track stage.",
                        ),
                    ),
                    (
                        "max_shift_px",
                        threshold_schema(
                            "How far an observation may sit from its seed (cluster stage) or \
                             from the surfel's projection (track stage), in source-image px.",
                        ),
                    ),
                    (
                        "max_keypoint_uncertainty",
                        threshold_schema(
                            "The largest tile localizability sigma_pos an observation may carry, \
                             in template-grid px.",
                        ),
                    ),
                    (
                        "min_relative_zncc",
                        threshold_schema(
                            "The fraction of the track's own self-agreement a candidate's ZNCC \
                             has to reach.",
                        ),
                    ),
                ],
                &[("reconstruction_label", edited_label_schema())],
            ),
        },
        ToolSpec {
            name: "split_bench_track",
            description: "Move the named observations off a bench track onto a second track \
                          beside it, whatever their verdicts were, and answer with the label the \
                          new item took. The half that comes off arrives at the cluster stage: a \
                          split questions the 3D hypothesis fitted to both halves, so carrying it \
                          onto the new half would state it as fact. Splitting off every \
                          observation, or none, is refused.",
            kind: Write,
            schema: object(
                &[("track", bench_track_schema())],
                &[
                    ("reconstruction_label", edited_label_schema()),
                    (
                        "observations",
                        json!({
                            "type": "array",
                            "items": { "type": "integer", "minimum": 0 },
                            "description":
                                "Which observations to take off, by their positions in \
                                 get_bench_track's list.",
                        }),
                    ),
                ],
            ),
        },
        ToolSpec {
            name: "commit_bench_track",
            description: "Write a bench track into the reconstruction as one version: its in \
                          observations become the track of a point, over the point it came from \
                          where it came from one and as a new point otherwise. Refused for a \
                          track still at the cluster stage, for one with fewer than two in \
                          observations, and on a reconstruction whose observations are .sift \
                          feature indexes rather than inline keypoints — each in the bench's own \
                          words. The track stays on the bench, seated on the point it wrote.",
            kind: Write,
            schema: object(
                &[("track", bench_track_schema())],
                &[("reconstruction_label", edited_label_schema())],
            ),
        },
        ToolSpec {
            name: "evaluate_bench_track",
            description: "Read every observation of a bench track at the stage the track is in \
                          and MOVE NOTHING — the position, the frame and every keypoint come \
                          back as they were. It is the refinement against the template at the \
                          cluster stage, and at the track stage one round of the localizer at \
                          the pixels the observations already sit at: each gets its \
                          leave-one-out ZNCC, seed_shift_px (how far the correlation peak sits \
                          from the observation), projection_offset_px (how far the observation \
                          sits from the point's projection — the number that says the point is \
                          off, not the sighting), the reprojection error, the ray angle and its \
                          tile localizability. No gate drops a row: an observation that cannot \
                          be read carries a reason sentence instead of a score. Read the \
                          numbers back with get_bench_track. It runs on a worker thread, so a \
                          reading still going after 200 ms replies with running: true and an \
                          operation_id to poll with get_background_task instead of the version \
                          it pushed. Needs the photographs, which are decoded on demand.",
            kind: Write,
            schema: object(
                &[
                    ("track", bench_track_schema()),
                    ("search_px", search_px_schema()),
                ],
                &[("reconstruction_label", edited_label_schema())],
            ),
        },
        ToolSpec {
            name: "fit_bench_track",
            description: "Fit a bench track at the stage it is in — the step that MOVES it. At \
                          the track stage it localizes every sighting against the surfel, \
                          refines each to sub-pixel, re-triangulates the in ones, re-centres \
                          the frame there and re-fuses the consensus bitmap; at the cluster \
                          stage it is the refinement, which is what a reading is too. Nothing \
                          is dropped by a gate: a sighting that does not belong is turned out \
                          with set_bench_track_verdict or by the thresholds, not deleted from \
                          the evidence. The fit ends by evaluating its own result, so the \
                          numbers it leaves behind are the ones evaluate_bench_track reports. \
                          Refused for a track stage with fewer than two in observations, which \
                          a reading permits. Answers as evaluate_bench_track does.",
            kind: Write,
            schema: object(
                &[
                    ("track", bench_track_schema()),
                    ("search_px", search_px_schema()),
                ],
                &[("reconstruction_label", edited_label_schema())],
            ),
        },
        ToolSpec {
            name: "set_bench_track_stage",
            description: "Move a bench track between its two representations. \"track\" \
                          triangulates the in observations, fits a surfel to them and localizes \
                          each keypoint against it, which is what a commit needs; \"cluster\" \
                          drops the geometry and leaves the patches registering onto one \
                          template, which is what questioning a wrong position looks like. Runs \
                          on a worker thread and answers as evaluate_bench_track does. Setting \
                          the stage a track is already at changes nothing.",
            kind: Write,
            schema: object(
                &[("track", bench_track_schema())],
                &[
                    ("reconstruction_label", edited_label_schema()),
                    (
                        "stage",
                        json!({
                            "type": "string",
                            "enum": ["cluster", "track"],
                            "description": "Which representation the track should be in.",
                        }),
                    ),
                ],
            ),
        },
        ToolSpec {
            name: "search_bench_track_descriptors",
            description: "Ask the node's descriptor index which OTHER photographs hold the patch \
                          around one observation of a bench track, and add each as a candidate. \
                          It is a constellation query, not a lookup of one descriptor: the \
                          detected keypoints within radius_px of the observation are looked up in \
                          the index, the hits are grouped by image, and an image whose hits agree \
                          on a single affine warp with at least min_inliers of them is found. That \
                          warp applied to the observation's own pixel and shape is the seed the \
                          new candidate takes, so it arrives where and at the size the warp says \
                          the patch is — evaluate_bench_track is what then scores it. An image the \
                          track already has an observation in is left alone whatever its verdict, \
                          and so is the searched image itself. Needs an index: open_descriptor_index \
                          or build_descriptor_index first, and get_bench reports whether one is \
                          open. Runs on a worker thread and answers as evaluate_bench_track does.",
            kind: Write,
            schema: object(
                &[
                    ("track", bench_track_schema()),
                    (
                        "radius_px",
                        json!({
                            "type": "number",
                            "exclusiveMinimum": 0,
                            "description":
                                "The constellation's radius around the observation, in the \
                                 searched image's own pixels. Omit for the radius that holds \
                                 about fifty of that image's keypoints, which is the size the \
                                 query is worth asking at: the affine is a first-order \
                                 approximation about the patch centre, so a wider patch buys \
                                 recall and loses the accuracy of the warp.",
                        }),
                    ),
                    (
                        "min_inliers",
                        json!({
                            "type": "integer",
                            "minimum": 3,
                            "description":
                                "Fewest agreeing correspondences an image needs to be found. \
                                 Omit for the query's own bar.",
                        }),
                    ),
                ],
                &[
                    ("reconstruction_label", edited_label_schema()),
                    ("observation", observation_schema()),
                ],
            ),
        },
        ToolSpec {
            name: "open_descriptor_index",
            description: "Adopt a .kdf descriptor index for one reconstruction, which is what \
                          search_bench_track_descriptors queries. Omit path for the default, \
                          which is index.kdf in the directory the node's .sift files live in. \
                          The index has to be over THIS reconstruction's images in THIS \
                          reconstruction's order — a match names a corpus image and the \
                          candidate it becomes names a reconstruction image — and one that is \
                          not is refused naming the first image it disagrees on. Nothing about \
                          the reconstruction or the bench moves, so this pushes no version and \
                          undo has nothing to take back. The viewer opens the default index on \
                          its own when the file is there; this is for one somewhere else.",
            kind: Write,
            schema: object(
                &[(
                    "path",
                    json!({
                        "type": "string",
                        "description":
                            "The .kdf to open. Omit for the default path, which get_bench \
                             reports under descriptor_index.",
                    }),
                )],
                &[("reconstruction_label", edited_label_schema())],
            ),
        },
        ToolSpec {
            name: "build_descriptor_index",
            description: "Build a descriptor index over every .sift file of one reconstruction, \
                          write it and open it. The corpus carries one image-table row per image \
                          of the node, in the node's own order, including images with no .sift \
                          file, which is what lets a search name node images directly. Omit path \
                          for the default, beside the .sift files. Refused when no .sift file of \
                          the node can be found. Reading every descriptor of a capture takes a \
                          while, so it runs on a worker thread and answers as \
                          evaluate_bench_track does.",
            kind: Write,
            schema: object(
                &[(
                    "path",
                    json!({
                        "type": "string",
                        "description":
                            "Where to write the .kdf. Omit for the default path, which \
                             get_bench reports under descriptor_index. An existing file there \
                             is replaced.",
                    }),
                )],
                &[("reconstruction_label", edited_label_schema())],
            ),
        },
        ToolSpec {
            name: "get_background_task",
            description: "What the viewer is busy with: the operation, the reconstruction it is \
                          running on, how long it has been going, how far along it is, the stage \
                          it is in, and the stages it has finished, in the shape get_action_log's \
                          detail returns them in. One operation runs at a time, viewer-wide, so \
                          this names none. With nothing running it reports the last operation of \
                          the session instead, with running: false and finished: true, so one \
                          call answers both whether the solve is done and what it cost. Test \
                          running to tell the two apart.",
            kind: Read,
            schema: object(&[], &[]),
        },
        ToolSpec {
            name: "cancel_background_task",
            description: "Ask the background operation that is running to stop. One runs at a \
                          time, viewer-wide, so this names none. The operation stops at its next \
                          safe point, pushes no version, and writes a cancelled row to the Action \
                          Log. Refused when nothing is running, and when the operation never asks \
                          whether it should stop.",
            kind: Write,
            schema: object(&[], &[]),
        },
        ToolSpec {
            name: "screenshot",
            description: "A PNG of the window as the human sees it — menu bar, every panel, \
                          status line — or, with panel_name, of one panel's body cropped from the \
                          same frame. The 3D viewport is panel_name \"viewer_3d\", and hud: false \
                          returns its render alone with nothing drawn over it. A panel that is \
                          closed, or behind another tab in its node, is refused naming show_panel. \
                          Answered after the next frame has been rendered, so it reflects any \
                          change made in the same batch of calls.",
            kind: Read,
            schema: object(
                &[
                    (
                        "panel_name",
                        json!({
                            "type": "string",
                            "enum": Tab::ALL.map(|tab| tab.wire_name()),
                            "description":
                                "Photograph one panel's body instead of the whole window, by the \
                                 name get_window_layout and the layout file use.",
                        }),
                    ),
                    (
                        "hud",
                        json!({
                            "type": "boolean",
                            "description":
                                "With panel_name \"viewer_3d\" only: false returns the 3D render \
                                 target itself, without the HUD, the stats and the status line \
                                 painted over it. Defaults to true.",
                        }),
                    ),
                    (
                        "max_dimension",
                        json!({
                            "type": "integer",
                            "minimum": 16,
                            "description":
                                "Scale the image down so neither side exceeds this many pixels. \
                                 Omit for the native size of whatever was photographed.",
                        }),
                    ),
                ],
                &[],
            ),
        },
    ]
}

// ── Schema fragments ─────────────────────────────────────────────────────

/// An object schema from its optional and required properties.
fn object(optional: &[(&str, Value)], required: &[(&str, Value)]) -> Value {
    let mut properties = Map::new();
    for (name, schema) in optional.iter().chain(required) {
        properties.insert((*name).to_string(), schema.clone());
    }
    json!({
        "type": "object",
        "properties": properties,
        "required": required.iter().map(|(name, _)| *name).collect::<Vec<_>>(),
        // Closed on purpose. A misspelled argument that is silently ignored
        // leaves the agent believing it asked for something it did not.
        "additionalProperties": false,
    })
}

fn flag(description: &str) -> Value {
    json!({ "type": "boolean", "description": description })
}

fn reconstruction_label_schema() -> Value {
    json!({
        "type": "string",
        "description":
            "Which reconstruction, by the label get_scene reports. Omit for the selected one. A \
             label is unique across the scene and survives every edit, which is why it rather \
             than any internal id is the handle.",
    })
}

/// The reconstruction argument of a tool that edits one, reads its history or
/// writes it out.
///
/// Required rather than defaulting to the selection, which is the one place
/// this surface departs from "omit for the selected one": the selection is the
/// *human's*, it moves under the agent between calls, and an edit that landed
/// on whatever was last clicked would be an edit the agent could not check it
/// had asked for. A read of the history is required for the same reason its
/// answer would otherwise be about a node the caller did not name.
fn edited_label_schema() -> Value {
    json!({
        "type": "string",
        "description":
            "Which reconstruction, by the label get_scene reports. Named rather than defaulting \
             to the selected one: the selection belongs to the human at the window and can move \
             between calls.",
    })
}

/// A pixel in one camera image's own pixel coordinates.
fn pixel_schema() -> Value {
    json!({
        "type": "array",
        "items": { "type": "number" },
        "minItems": 2,
        "maxItems": 2,
        "description":
            "A pixel [x, y] in the camera image's own coordinates, as get_point reports an \
             observation's xy.",
    })
}

/// A camera image argument, which takes either of the two handles the surface
/// hands out.
///
/// The field is named for the entity rather than for an attribute, because it
/// has no single spelling: a track observation reports an index, a
/// `list_camera_images` row reports both, and an agent arrives holding
/// whichever it read. Contrast `camera_intrinsics_index`, which can name its
/// attribute because an intrinsics record has exactly one handle.
fn camera_image_schema() -> Value {
    json!({
        "description":
            "Which camera image: its index in the reconstruction, or its name — the .sfmr \
             relative path, as in \"images/IMG_0042.jpg\".",
        "anyOf": [
            { "type": "integer", "minimum": 0 },
            { "type": "string" },
        ],
    })
}

/// An item on the bench, by its label.
///
/// A label and not an index: an item is named by its label everywhere -- in the
/// Scene tree, on the panel's tabs and in every Action Log row -- and the
/// bench's own refusal for one that names nothing is in those terms.
fn bench_item_schema() -> Value {
    json!({
        "type": "string",
        "description":
            "Which item on the bench, by the label get_bench reports. A label is minted from \
             what the item was made from — a point id, or an image and a pixel — until \
             rename_bench_item gives it one of your own.",
    })
}

/// The track a bench tool acts on, which is optional: a call that names none
/// acts on the active track, as a gesture in the Track Edit panel does.
fn bench_track_schema() -> Value {
    json!({
        "type": "string",
        "description":
            "Which track on the bench, by its label. Omit for the active track, which is what \
             the Track Edit panel is showing and what a create or an activate last made active.",
    })
}

/// How far around each observation a reading looks for its correlation peak.
fn search_px_schema() -> Value {
    json!({
        "type": "number",
        "description":
            "How far from each observation's own pixel the correlation peak is looked for, in \
             patch-grid px. Omit for the reading's own radius. A wider window finds a feature \
             the sighting sits further from, and says so in seed_shift_px.",
    })
}

/// One observation of a bench track, by its position in the track's list.
fn observation_schema() -> Value {
    json!({
        "type": "integer",
        "minimum": 0,
        "description":
            "Which observation, by its position in get_bench_track's observations list. \
             Observations are appended and never renumbered, so the position is stable for the \
             life of the track.",
    })
}

/// One bar of the threshold painting: a number, or absent to leave the bar
/// where the track has it.
fn threshold_schema(description: &str) -> Value {
    json!({ "type": "number", "description": description })
}

/// The seed forms `create_bench_cluster` and `add_bench_track_observation`
/// share, as optional properties.
///
/// Optional rather than required because the three forms are alternatives and a
/// schema cannot say "one of these"; which combinations are a seed at all is
/// settled in [`parse_seed`], which sees what the call named.
fn seed_properties() -> Vec<(&'static str, Value)> {
    vec![
        ("pixel", pixel_schema()),
        (
            "radius_px",
            json!({
                "type": "number",
                "exclusiveMinimum": 0,
                "description":
                    "The patch's half-width at that pixel, in this image's own pixels. With \
                     create_bench_cluster, omit for the radius the viewer's own Create 3D Point \
                     prompt would offer; with add_bench_track_observation, omit for the scale \
                     the track already works at.",
            }),
        ),
        (
            "affine",
            json!({
                "type": "array",
                "items": {
                    "type": "array",
                    "items": { "type": "number" },
                    "minItems": 2,
                    "maxItems": 2,
                },
                "minItems": 2,
                "maxItems": 2,
                "description":
                    "The affine shape at that pixel, as [[a11, a12], [a21, a22]]: the detector's \
                     canonical keypoint frame mapped onto this image's pixels, which is what a \
                     .sift feature's shape is. A shape is a scale rather than a size, so it is \
                     read over the cluster's own radius.",
            }),
        ),
        (
            "feature",
            json!({
                "type": "integer",
                "minimum": 0,
                "description":
                    "Seed from a .sift feature of this camera image instead, by its index in \
                     that file. It carries its own position and its own shape, so pixel, \
                     radius_px and affine are not accepted with it.",
            }),
        ),
    ]
}

fn point_schema() -> Value {
    json!({
        "description":
            "Which 3D point: a bare index into the selected reconstruction, or a full \
             pt3d_<hash>_<index> id, which names its own reconstruction and so resolves \
             wherever the selection happens to be.",
        "anyOf": [
            { "type": "integer", "minimum": 0 },
            { "type": "string" },
        ],
    })
}

/// A panel argument: one of the seven names, spelled as the layout file spells
/// them.
///
/// `panel_name` rather than `panel`, because the field carries a name and not
/// a panel — the same rule that makes the reconstruction argument
/// `reconstruction_label`.
fn panel_name_schema() -> Value {
    json!({
        "type": "string",
        "enum": Tab::ALL.map(|tab| tab.wire_name()),
        "description":
            "Which panel, by the name get_window_layout and the layout file use.",
    })
}

fn window_state_schema() -> Value {
    json!({
        "type": "string",
        "enum": WindowState::ALL.map(|state| state.wire_name()),
        "description":
            "What the window should be. \"normal\" restores it from all three of minimized, \
             maximized and fullscreen.",
    })
}

/// The `window` section of a layout document: the same five keys the file
/// carries, since the file and the wire are one document with one parser.
fn window_section_schema() -> Value {
    json!({
        "type": "object",
        "additionalProperties": false,
        "description":
            "Where the window goes. Every key is optional and what the section does not carry is \
             preserved; omit the section to leave the window alone.",
        "properties": {
            "state": window_state_schema(),
            "outer_position": int_pair_schema(
                "The top-left corner of the window's normal rectangle, in physical pixels in \
                 desktop coordinates.",
                None,
            ),
            "inner_size": int_pair_schema(
                "The drawable area of the window's normal rectangle, in physical pixels. The \
                 platform has the last word on it, so read the reply rather than assuming the \
                 request.",
                Some(1),
            ),
            "monitor": {
                "type": "object",
                "additionalProperties": false,
                "description":
                    "The monitor the rectangle was measured on, as get_window_layout writes it. \
                     A rectangle that lands nowhere visible on this desktop is mapped onto the \
                     current monitor by the share of it the window occupied. Needs a rectangle \
                     to fit.",
                "properties": {
                    "position": int_pair_schema(
                        "The monitor's top-left corner, physical pixels.",
                        None,
                    ),
                    "size": int_pair_schema("The monitor's size, physical pixels.", Some(1)),
                },
                "required": ["position", "size"],
            },
            "focus": {
                "type": "boolean",
                "description":
                    "Bring the window to the front. A platform may decline to let an application \
                     take focus; the reply's focused says whether it worked.",
            },
        },
    })
}

/// A two-element array of whole numbers: a size, or a position.
fn int_pair_schema(description: &str, minimum: Option<i64>) -> Value {
    let mut items = json!({ "type": "integer" });
    if let Some(minimum) = minimum {
        items["minimum"] = json!(minimum);
    }
    json!({
        "type": "array",
        "items": items,
        "minItems": 2,
        "maxItems": 2,
        "description": description,
    })
}

fn tint_names_with_null() -> Vec<Value> {
    crate::scene::TINT_PALETTE
        .iter()
        .map(|color| json!(color.name))
        .chain(std::iter::once(Value::Null))
        .collect()
}

fn vec3_schema(description: &str) -> Value {
    json!({
        "type": "array",
        "items": { "type": "number" },
        "minItems": 3,
        "maxItems": 3,
        "description": description,
    })
}

fn set_view_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "fit": {
                "type": ["string", "null"],
                "description":
                    "Frame the scene: a reconstruction label to frame that one, or null to frame \
                     everything drawn.",
            },
            "look_through": {
                "type": "object",
                "description": "Look through one camera image, as double-clicking its frustum does.",
                "properties": {
                    "reconstruction_label": reconstruction_label_schema(),
                    "camera_image": camera_image_schema(),
                },
                "required": ["camera_image"],
                "additionalProperties": false,
            },
            "exit_camera_view": {
                "type": "boolean",
                "description": "Leave camera view, keeping the camera where it is.",
            },
            "position": vec3_schema(
                "Camera position in world coordinates. On its own it moves the camera and \
                 carries the target along, keeping the orientation.",
            ),
            "target": vec3_schema(
                "The point the camera looks at. With position, the look-at form; on its own it \
                 re-centres the view on that point, keeping the orientation and the distance.",
            ),
            "forward": vec3_schema(
                "The direction the camera looks, the derived.forward the view block reports. \
                 Need not be a unit vector. On its own it swings the camera around what it is \
                 looking at rather than turning it in place.",
            ),
            "up": vec3_schema(
                "The roll, where the orientation is being derived -- with forward, or with \
                 position and target. Defaults to the view's current world_up.",
            ),
            "orientation_wxyz": {
                "type": "array",
                "items": { "type": "number" },
                "minItems": 4,
                "maxItems": 4,
                "description":
                    "World-to-camera rotation. Present makes this the exact form, which restores \
                     a view read from get_scene verbatim.",
            },
            "target_distance": {
                "type": "number",
                "exclusiveMinimum": 0,
                "description":
                    "Distance to the orbit target along the camera's forward axis. On its own it \
                     dollies: the target stays put and the camera moves. Not accepted alongside \
                     position and target, whose separation is already the distance.",
            },
            "world_up": vec3_schema(
                "Navigation up, which carries the roll, for the exact form. Elsewhere the roll \
                 is up.",
            ),
            "fov_short_axis_deg": {
                "type": "number",
                "minimum": 5,
                "maximum": 160,
                "description":
                    "Field of view of the shorter viewport dimension — vertical in a landscape \
                     window, horizontal in a portrait one. May accompany an explicit camera \
                     placement, or be sent alone.",
            },
        },
        "additionalProperties": false,
    })
}

// ── The parse ────────────────────────────────────────────────────────────

/// Build the [`Command`] a `tools/call` asked for.
///
/// The schemas above are closed and typed, so a compliant client will not
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

    let command = match name {
        "get_scene" => {
            args.reject_unknown(&[])?;
            Command::GetScene
        }
        "list_camera_images" => {
            args.reject_unknown(&["reconstruction_label", "offset", "limit"])?;
            Command::ListCameraImages {
                reconstruction_label: args.optional_string("reconstruction_label")?,
                offset: args.optional_usize("offset")?.unwrap_or(0),
                limit: args
                    .optional_usize("limit")?
                    .unwrap_or(super::read::DEFAULT_LIMIT),
            }
        }
        "get_camera_image" => {
            args.reject_unknown(&["reconstruction_label", "camera_image"])?;
            Command::GetCameraImage {
                reconstruction_label: args.optional_string("reconstruction_label")?,
                camera_image: args.camera_image("camera_image")?,
            }
        }
        "get_camera_intrinsics" => {
            args.reject_unknown(&["reconstruction_label", "camera_intrinsics_index"])?;
            Command::GetCameraIntrinsics {
                reconstruction_label: args.optional_string("reconstruction_label")?,
                camera_intrinsics_index: args.required_usize("camera_intrinsics_index")?,
            }
        }
        "get_point" => {
            args.reject_unknown(&["point"])?;
            Command::GetPoint {
                point: args.point("point")?,
            }
        }
        "get_action_log" => {
            args.reject_unknown(&["since_revision", "limit", "actors", "detail"])?;
            Command::GetActionLog {
                since_revision: args.optional_u64("since_revision")?.unwrap_or(0),
                limit: args
                    .optional_usize("limit")?
                    .unwrap_or(super::read::ACTION_LOG_DEFAULT_LIMIT),
                actors: args.actors("actors")?,
                detail: args.optional_bool("detail")?.unwrap_or(false),
            }
        }
        "open_reconstruction" => {
            args.reject_unknown(&["path"])?;
            Command::OpenReconstruction {
                path: std::path::PathBuf::from(args.required_string("path")?),
            }
        }
        "close_reconstruction" => {
            args.reject_unknown(&["reconstruction_label", "all"])?;
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
        "select_reconstruction" => {
            args.reject_unknown(&["reconstruction_label"])?;
            Command::SelectReconstruction {
                reconstruction_label: args.required_string("reconstruction_label")?,
            }
        }
        "select_camera_image" => {
            args.reject_unknown(&["reconstruction_label", "camera_image"])?;
            Command::SelectCameraImage {
                reconstruction_label: args.optional_string("reconstruction_label")?,
                camera_image: args.camera_image("camera_image")?,
            }
        }
        "select_camera_intrinsics" => {
            args.reject_unknown(&["reconstruction_label", "camera_intrinsics_index"])?;
            Command::SelectCameraIntrinsics {
                reconstruction_label: args.optional_string("reconstruction_label")?,
                camera_intrinsics_index: args.required_usize("camera_intrinsics_index")?,
            }
        }
        "select_point" => {
            args.reject_unknown(&["point"])?;
            Command::SelectPoint {
                point: args.point("point")?,
            }
        }
        "clear_selection" => {
            args.reject_unknown(&["scope"])?;
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
            args.reject_unknown(&[
                "reconstruction_label",
                "visible",
                "interactive",
                "show_points",
                "show_camera_images",
                "show_patches",
                "show_points_at_infinity",
                "tint",
            ])?;
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
        "set_solo" => {
            args.reject_unknown(&["reconstruction_label"])?;
            Command::SetSolo {
                reconstruction_label: args.optional_string("reconstruction_label")?,
            }
        }
        "get_image_detail_display" => {
            args.reject_unknown(&[])?;
            Command::GetImageDetailDisplay
        }
        // Every vocabulary this tool has is static — seven modes, two ladders,
        // the bounds on a size filter — so the whole call is validated here and
        // `apply` cannot fail. That is what makes a refusal atomic.
        "set_image_detail_display" => Command::SetImageDetailDisplay {
            change: super::display::parse_change(&args)?,
        },
        "get_image_detail_view" => {
            args.reject_unknown(&[])?;
            Command::GetImageDetailView
        }
        // The one-target rule, the zoom's range and the targets that refuse a
        // zoom are all settled here, before a `Command` exists: what is left
        // for the tool body is resolving the handles, which is the part that
        // needs the scene.
        "set_image_detail_view" => Command::SetImageDetailView {
            request: super::display::parse_view(&args)?,
        },
        "get_timing_detail" => {
            args.reject_unknown(&[])?;
            Command::GetTimingDetail
        }
        // Required rather than a toggle, for the reason `set_solo` takes the
        // state it wants: an agent issuing a toggle cannot know the outcome
        // without reading first, and a retried call would undo itself.
        "set_timing_detail" => {
            args.reject_unknown(&["enabled"])?;
            Command::SetTimingDetail {
                enabled: args.required_bool("enabled")?,
            }
        }
        "set_view" => parse_set_view(&args)?,
        "get_window_layout" => {
            args.reject_unknown(&[])?;
            Command::GetWindowLayout
        }
        "set_window_layout" => {
            args.reject_unknown(&["sfm_explorer_layout", "window", "layout"])?;
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
        "show_panel" => {
            args.reject_unknown(&["panel_name"])?;
            Command::ShowPanel {
                panel: args.panel("panel_name")?,
            }
        }
        "hide_panel" => {
            args.reject_unknown(&["panel_name"])?;
            Command::HidePanel {
                panel: args.panel("panel_name")?,
            }
        }
        "get_history" => {
            args.reject_unknown(&["reconstruction_label"])?;
            Command::GetHistory {
                reconstruction_label: args.required_string("reconstruction_label")?,
            }
        }
        "undo" => {
            args.reject_unknown(&["reconstruction_label"])?;
            Command::Undo {
                reconstruction_label: args.required_string("reconstruction_label")?,
            }
        }
        "redo" => {
            args.reject_unknown(&["reconstruction_label"])?;
            Command::Redo {
                reconstruction_label: args.required_string("reconstruction_label")?,
            }
        }
        "jump_to_version" => {
            args.reject_unknown(&["reconstruction_label", "serial"])?;
            Command::JumpToVersion {
                reconstruction_label: args.required_string("reconstruction_label")?,
                serial: args.required_string("serial")?,
            }
        }
        "save_reconstruction" => {
            args.reject_unknown(&["reconstruction_label", "path"])?;
            Command::SaveReconstruction {
                reconstruction_label: args.required_string("reconstruction_label")?,
                path: args.optional_string("path")?.map(std::path::PathBuf::from),
            }
        }
        "delete_point" => {
            args.reject_unknown(&["reconstruction_label", "point"])?;
            Command::DeletePoint {
                reconstruction_label: args.required_string("reconstruction_label")?,
                point: args.point("point")?,
            }
        }
        "delete_camera_image" => {
            args.reject_unknown(&["reconstruction_label", "camera_image"])?;
            Command::DeleteCameraImage {
                reconstruction_label: args.required_string("reconstruction_label")?,
                camera_image: args.camera_image("camera_image")?,
            }
        }
        "add_observation" => {
            args.reject_unknown(&["reconstruction_label", "point", "camera_image", "pixel"])?;
            Command::AddObservation {
                reconstruction_label: args.required_string("reconstruction_label")?,
                point: args.point("point")?,
                camera_image: args.camera_image("camera_image")?,
                pixel: args.pixel("pixel")?,
            }
        }
        "create_point" => {
            args.reject_unknown(&["reconstruction_label", "camera_image", "pixel", "radius_px"])?;
            Command::CreatePoint {
                reconstruction_label: args.required_string("reconstruction_label")?,
                camera_image: args.camera_image("camera_image")?,
                pixel: args.pixel("pixel")?,
                radius_px: args.radius("radius_px")?,
            }
        }
        "remove_observation" => {
            args.reject_unknown(&["reconstruction_label", "point", "camera_image"])?;
            Command::RemoveObservation {
                reconstruction_label: args.required_string("reconstruction_label")?,
                point: args.point("point")?,
                camera_image: args.camera_image("camera_image")?,
            }
        }
        "move_camera_image" => {
            args.reject_unknown(&["reconstruction_label", "camera_image", "world_from_camera"])?;
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
        "resect_camera_image_in_place" => {
            args.reject_unknown(&["reconstruction_label", "camera_image", "from_matches"])?;
            Command::ResectCameraImageInPlace {
                reconstruction_label: args.required_string("reconstruction_label")?,
                camera_image: args.camera_image("camera_image")?,
                from_matches: args.optional_bool("from_matches")?.unwrap_or(false),
            }
        }
        "bundle_adjust" => {
            args.reject_unknown(&["reconstruction_label", "release_focal"])?;
            Command::BundleAdjust {
                reconstruction_label: args.required_string("reconstruction_label")?,
                release_focal: args.optional_bool("release_focal")?.unwrap_or(false),
            }
        }
        "get_bench" => {
            args.reject_unknown(&["reconstruction_label"])?;
            Command::GetBench {
                reconstruction_label: args.required_string("reconstruction_label")?,
            }
        }
        "get_bench_track" => {
            args.reject_unknown(&["reconstruction_label", "track"])?;
            Command::GetBenchTrack {
                reconstruction_label: args.required_string("reconstruction_label")?,
                track: args.optional_string("track")?,
            }
        }
        "create_bench_cluster" => {
            args.reject_unknown(&[
                "reconstruction_label",
                "camera_image",
                "pixel",
                "radius_px",
                "affine",
                "feature",
            ])?;
            Command::CreateBenchCluster {
                reconstruction_label: args.required_string("reconstruction_label")?,
                camera_image: args.camera_image("camera_image")?,
                seed: parse_seed(&args)?,
            }
        }
        "create_bench_track" => {
            args.reject_unknown(&["reconstruction_label", "point"])?;
            Command::CreateBenchTrack {
                reconstruction_label: args.required_string("reconstruction_label")?,
                point: args.point("point")?,
            }
        }
        "activate_bench_item" => {
            args.reject_unknown(&["reconstruction_label", "item"])?;
            Command::ActivateBenchItem {
                reconstruction_label: args.required_string("reconstruction_label")?,
                item: args.required_string("item")?,
            }
        }
        "rename_bench_item" => {
            args.reject_unknown(&["reconstruction_label", "item", "label"])?;
            Command::RenameBenchItem {
                reconstruction_label: args.required_string("reconstruction_label")?,
                item: args.required_string("item")?,
                label: args.required_string("label")?,
            }
        }
        "discard_bench_item" => {
            args.reject_unknown(&["reconstruction_label", "item"])?;
            Command::DiscardBenchItem {
                reconstruction_label: args.required_string("reconstruction_label")?,
                item: args.required_string("item")?,
            }
        }
        "add_bench_track_observation" => {
            args.reject_unknown(&[
                "reconstruction_label",
                "track",
                "camera_image",
                "pixel",
                "radius_px",
                "affine",
                "feature",
            ])?;
            Command::AddBenchTrackObservation {
                reconstruction_label: args.required_string("reconstruction_label")?,
                track: args.optional_string("track")?,
                camera_image: args.camera_image("camera_image")?,
                seed: parse_seed(&args)?,
            }
        }
        "set_bench_track_verdict" => {
            args.reject_unknown(&["reconstruction_label", "track", "observation", "verdict"])?;
            Command::SetBenchTrackVerdict {
                reconstruction_label: args.required_string("reconstruction_label")?,
                track: args.optional_string("track")?,
                observation: args.required_usize("observation")?,
                verdict: args.verdict("verdict")?,
            }
        }
        "apply_bench_track_thresholds" => {
            args.reject_unknown(&[
                "reconstruction_label",
                "track",
                "min_zncc",
                "max_shift_px",
                "max_keypoint_uncertainty",
                "min_relative_zncc",
            ])?;
            Command::ApplyBenchTrackThresholds {
                reconstruction_label: args.required_string("reconstruction_label")?,
                track: args.optional_string("track")?,
                thresholds: super::ThresholdChange {
                    min_zncc: args.optional_f64("min_zncc")?,
                    max_shift_px: args.optional_f64("max_shift_px")?,
                    max_keypoint_uncertainty: args.optional_f64("max_keypoint_uncertainty")?,
                    min_relative_zncc: args.optional_f64("min_relative_zncc")?,
                },
            }
        }
        "split_bench_track" => {
            args.reject_unknown(&["reconstruction_label", "track", "observations"])?;
            Command::SplitBenchTrack {
                reconstruction_label: args.required_string("reconstruction_label")?,
                track: args.optional_string("track")?,
                observations: args.observations("observations")?,
            }
        }
        "commit_bench_track" => {
            args.reject_unknown(&["reconstruction_label", "track"])?;
            Command::CommitBenchTrack {
                reconstruction_label: args.required_string("reconstruction_label")?,
                track: args.optional_string("track")?,
            }
        }
        "evaluate_bench_track" => {
            args.reject_unknown(&["reconstruction_label", "track", "search_px"])?;
            Command::EvaluateBenchTrack {
                reconstruction_label: args.required_string("reconstruction_label")?,
                track: args.optional_string("track")?,
                search_px: args.optional_f64("search_px")?,
            }
        }
        "fit_bench_track" => {
            args.reject_unknown(&["reconstruction_label", "track", "search_px"])?;
            Command::FitBenchTrack {
                reconstruction_label: args.required_string("reconstruction_label")?,
                track: args.optional_string("track")?,
                search_px: args.optional_f64("search_px")?,
            }
        }
        "set_bench_track_stage" => {
            args.reject_unknown(&["reconstruction_label", "track", "stage"])?;
            Command::SetBenchTrackStage {
                reconstruction_label: args.required_string("reconstruction_label")?,
                track: args.optional_string("track")?,
                stage: args.stage("stage")?,
            }
        }
        "search_bench_track_descriptors" => {
            args.reject_unknown(&[
                "reconstruction_label",
                "track",
                "observation",
                "radius_px",
                "min_inliers",
            ])?;
            Command::SearchBenchTrackDescriptors {
                reconstruction_label: args.required_string("reconstruction_label")?,
                track: args.optional_string("track")?,
                observation: args.required_usize("observation")?,
                radius_px: args.optional_f64("radius_px")?,
                min_inliers: args.optional_usize("min_inliers")?,
            }
        }
        "open_descriptor_index" => {
            args.reject_unknown(&["reconstruction_label", "path"])?;
            Command::OpenDescriptorIndex {
                reconstruction_label: args.required_string("reconstruction_label")?,
                path: args.optional_string("path")?,
            }
        }
        "build_descriptor_index" => {
            args.reject_unknown(&["reconstruction_label", "path"])?;
            Command::BuildDescriptorIndex {
                reconstruction_label: args.required_string("reconstruction_label")?,
                path: args.optional_string("path")?,
            }
        }
        "get_background_task" => {
            args.reject_unknown(&[])?;
            Command::GetBackgroundTask
        }
        "cancel_background_task" => {
            args.reject_unknown(&[])?;
            Command::CancelBackgroundTask
        }
        "screenshot" => {
            args.reject_unknown(&["panel_name", "hud", "max_dimension"])?;
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
    args.reject_unknown(&[
        "fit",
        "look_through",
        "exit_camera_view",
        "position",
        "target",
        "forward",
        "up",
        "orientation_wxyz",
        "target_distance",
        "world_up",
        "fov_short_axis_deg",
    ])?;

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
