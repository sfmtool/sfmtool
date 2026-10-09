// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use super::*;
use ToolKind::{Read, Write};

pub(super) fn specs() -> Vec<ToolSpec> {
    vec![
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
