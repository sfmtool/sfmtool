// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use super::*;
use ToolKind::{Read, Write};

pub(super) fn specs() -> Vec<ToolSpec> {
    vec![
        ToolSpec {
            name: "get_background_tasks",
            description: "What the viewer is busy with. Background tasks run from two queues, \
                          slow (the whole-reconstruction edits and Build index files) and fast \
                          (opening files and the other bench steps); each runs one task at a \
                          time and the two run at the same time. With no argument, for each \
                          queue (slow, fast): running, \
                          the task it runs or null, and last_finished, the last task that \
                          finished from it or null. A running task carries the operation, its \
                          queue, the reconstruction it runs on, how long it has been going, how \
                          far along it is, the stage it is in, and the stages it has finished, in \
                          the shape get_action_log's detail returns them in; a finished one \
                          carries running: false, finished: true, what it cost and how it ended. \
                          With operation_id, that one task in the same shape, running or \
                          finished, so one call answers both whether the solve is done and what \
                          it cost. Test running to tell the two apart.",
            kind: Read,
            schema: object(
                &[(
                    "operation_id",
                    json!({
                        "type": "integer",
                        "minimum": 0,
                        "description":
                            "The task to report on, from the handle the call that started it \
                             returned. Omit for both queues.",
                    }),
                )],
                &[],
            ),
        },
        ToolSpec {
            name: "cancel_background_tasks",
            description: "Ask running background tasks to stop: the ones operation_ids names, \
                          or every running task. Each stops at its next safe point, pushes no \
                          version, and writes a cancelled row to the Action Log. The reply lists \
                          cancelling, the tasks asked to stop, and refused, each task that \
                          cannot be stopped (one whose operation never asks whether it should, \
                          or one that is not running) with the reason. Refused as a whole when \
                          it names nothing and nothing is running.",
            kind: Write,
            schema: object(
                &[(
                    "operation_ids",
                    json!({
                        "type": "array",
                        "items": { "type": "integer", "minimum": 0 },
                        "minItems": 1,
                        "description":
                            "The tasks to stop, by the operation_id their handles carry. Omit \
                             to stop every running task.",
                    }),
                )],
                &[],
            ),
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
                            "minimum": crate::mcp::frame::MIN_MAX_DIMENSION,
                            "description":
                                "Scale the image down so neither side exceeds this many pixels. \
                                 Omit for the native size of whatever was photographed. \
                                 Rectangles in the widget listing are never scaled with it.",
                        }),
                    ),
                    ("crop_px", crop_px_schema("Photograph only this rectangle of the target")),
                    (
                        "widgets",
                        json!({
                            "type": "boolean",
                            "description":
                                "true adds the get_widgets listing of the same target and crop, \
                                 taken from the frame the picture is of, as a JSON block after \
                                 the image. Refused with hud: false, since the render target \
                                 under the HUD has no widgets on it.",
                        }),
                    ),
                ],
                &[],
            ),
        },
        ToolSpec {
            name: "get_widgets",
            description: "Every widget egui drew in the window, or with panel_name in one \
                          panel's body: its id, role, name, rectangle, enabled and toggled \
                          state, value, what a click on it can do, and its path of named \
                          ancestors. Rectangles are rect_px, [x, y, width, height] in physical \
                          pixels of the target, the same space a screenshot of that target is \
                          in. The dialogs and menus open above the dock come first, in dialogs \
                          and menus, each with its own widgets, even when both are empty. The \
                          dock's tabs are listed as role tab with the panel_name they raise. \
                          A panel that is closed or behind another tab is refused naming \
                          show_panel. Answered after the next frame has been laid out.",
            kind: Read,
            schema: object(
                &[
                    (
                        "panel_name",
                        json!({
                            "type": "string",
                            "enum": Tab::ALL.map(|tab| tab.wire_name()),
                            "description":
                                "List one panel's body instead of the whole window, by the name \
                                 get_window_layout and the layout file use.",
                        }),
                    ),
                    ("crop_px", crop_px_schema("List only the widgets that overlap this rectangle of the target, each with its whole rectangle")),
                ],
                &[],
            ),
        },
    ]
}

/// A rectangle of a tool's target, in its physical pixels.
fn crop_px_schema(what: &str) -> Value {
    json!({
        "type": "array",
        "items": { "type": "integer", "minimum": 0 },
        "minItems": 4,
        "maxItems": 4,
        "description": format!(
            "{what}: [x, y, width, height] in the target's \
             physical pixels, the space rect_px is in. A rectangle that reaches outside the \
             target is refused, and the refusal names the target's size."
        ),
    })
}
