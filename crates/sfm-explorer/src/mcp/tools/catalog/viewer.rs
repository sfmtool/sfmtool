// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use super::*;
use ToolKind::Write;

pub(super) fn specs() -> Vec<ToolSpec> {
    vec![
        ToolSpec {
            name: "open_reconstruction",
            description: "Load an .sfmr file into the scene as a new reconstruction, and select \
                          it. Opening a path that is already open adds a second node for it, \
                          with a history of its own; `already_open` says whether that happened. \
                          The open is a background task: it reads the file, builds every \
                          thumbnail the file does not carry (from each image's .sift, else its \
                          photograph) and renders every patch bitmap it does not carry, for \
                          display only. One that finishes within 200 ms replies with the \
                          reconstruction; one still going replies with running: true and an \
                          operation_id to poll with get_background_task, and the node appears \
                          when it lands. Refused while another background operation runs. \
                          Read the returned `label` back rather than assuming it: a colliding \
                          file stem is disambiguated as \"name (2)\".",
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
            name: "set_reconstruction_transform",
            description: "Set the display transform one reconstruction is drawn under: the \
                          similarity from its own coordinates into the shared world, as \
                          get_scene reports it in transform. The identity returns it to its own \
                          frame, which is the Scene panel's Reset Transform. One version of the \
                          reconstruction whose data is untouched, so the reconstruction does not \
                          go dirty and undo steps back out of it; nothing reaches a file until \
                          bake_reconstruction_transform writes the transform into the data and a \
                          save writes that.",
            kind: Write,
            schema: object(
                &[],
                &[
                    ("reconstruction_label", edited_label_schema()),
                    (
                        "transform",
                        json!({
                            "type": "object",
                            "description":
                                "The similarity p' = scale * (rotation p) + translation, from \
                                 the reconstruction's own coordinates into the world.",
                            "properties": {
                                "rotation_wxyz": {
                                    "type": "array",
                                    "items": { "type": "number" },
                                    "minItems": 4,
                                    "maxItems": 4,
                                    "description": "The rotation, WXYZ. Normalised on arrival.",
                                },
                                "translation": vec3_schema("The translation, in world units."),
                                "scale": {
                                    "type": "number",
                                    "exclusiveMinimum": 0,
                                    "description": "The uniform scale.",
                                },
                            },
                            "required": ["rotation_wxyz", "translation", "scale"],
                            "additionalProperties": false,
                        }),
                    ),
                ],
            ),
        },
        ToolSpec {
            name: "set_reconstruction_transform_from_patch",
            description: "Set one reconstruction's display transform from the active patch on \
                          its bench, in one of the four ways the 3D viewport's patch menu offers: \
                          set_to_origin draws the scene in the patch's own frame (centre at the \
                          origin, u on +X, normal on +Z); align_normal_to_z tips the scene about \
                          the patch's centre until its normal is +Z; translate_to_origin moves the \
                          patch's centre to the origin; translate_to_xy_plane moves the scene \
                          along Z alone so the patch's centre sits at z = 0. Each composes onto \
                          the transform the reconstruction already carries and keeps its scale. \
                          One version, like set_reconstruction_transform. Refused when nothing on \
                          the bench is active, the active item is a cluster, the track has no \
                          patch frame, or the track is at infinity.",
            kind: Write,
            schema: object(
                &[],
                &[
                    ("reconstruction_label", edited_label_schema()),
                    (
                        "mode",
                        json!({
                            "type": "string",
                            "enum": crate::display_transform::PatchReframe::ALL
                                .map(|mode| mode.wire_name()),
                            "description": "Which of the four, as the menu labels them, \
                                            snake-cased.",
                        }),
                    ),
                ],
            ),
        },
        ToolSpec {
            name: "bake_reconstruction_transform",
            description: "Write one reconstruction's display transform into its points, camera \
                          poses and bench, and return it to its own frame, as one version. The \
                          picture does not move: what changes is which side of the display the \
                          numbers live on, so a save_reconstruction afterwards carries the new \
                          frame. A bulk edit that renumbers nothing, and undo puts both the data \
                          and the transform back. Refused on a reconstruction whose transform is \
                          the identity.",
            kind: Write,
            schema: object(&[], &[("reconstruction_label", edited_label_schema())]),
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
                    (
                        "feature_size_px",
                        crate::mcp::display::feature_size_schema(),
                    ),
                    (
                        "tracked_only",
                        flag(
                            "Show only features with a 3D point behind them, which is the CLI's \
                             --filter-sfm.",
                        ),
                    ),
                    ("intrinsics", crate::mcp::display::intrinsics_schema()),
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
                          and the reply says where it landed. The panel is opened if it was \
                          closed, and a call that arrives before it has drawn a photograph is \
                          answered once it has. The reply is get_image_detail_view's document \
                          for the view the panel applies.",
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
                            "True for the whole photograph: zoom 1.0, centred — what Z does in \
                             the panel. It settles its own zoom, so a zoom argument beside it \
                             is refused.",
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
    ]
}
