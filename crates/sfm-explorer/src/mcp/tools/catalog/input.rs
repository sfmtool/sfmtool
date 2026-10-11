// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The five input tools: the pointer and the keyboard, as a person uses them.

use super::*;
use crate::mcp::input::MAX_DRAG_POINTS;
use ToolKind::Input;

pub(super) fn specs() -> Vec<ToolSpec> {
    vec![
        ToolSpec {
            name: "click",
            description: "Click in the window the way a person's mouse does: the pointer moves \
                          to the point and a frame is drawn, the button is pressed and released \
                          in the next, and the reply is built from the frame after that. Aim \
                          with at_px, in the pixels of the window or of panel_name's body (the \
                          space get_widgets and screenshot use), or with widget, an id from \
                          get_widgets, which is aimed at its centre and refused naming what is \
                          on top when a menu or dialog covers it. The reply gives at_px, hit \
                          (the clickable widget under the pointer, or null) and the dialogs and \
                          menus open after the click; what the click did is read back with the \
                          tool that reports that state, and the Action Log records it as it \
                          would a person's. The pointer stays where it was left.",
            kind: Input,
            schema: object(
                &[
                    ("panel_name", panel_name_schema("at_px is a point of")),
                    ("at_px", at_px_schema()),
                    ("widget", widget_schema()),
                    (
                        "mouse_button",
                        json!({
                            "type": "string",
                            "enum": ["left", "middle", "right"],
                            "description": "Which button. Omit for left.",
                        }),
                    ),
                    (
                        "count",
                        json!({
                            "type": "integer",
                            "enum": [1, 2],
                            "description":
                                "2 for a double click: a second press and release in the frame \
                                 after the first. Omit for 1.",
                        }),
                    ),
                    ("modifiers", modifiers_schema("during the click")),
                ],
                &[],
            ),
        },
        ToolSpec {
            name: "hover",
            description: "Move the pointer to a point and leave it there, which is what opens a \
                          submenu and what a key read over a panel needs. Aimed as click is, \
                          with at_px and an optional panel_name, or with widget. The reply gives \
                          at_px, hit (the clickable widget the pointer arrived over, or null) \
                          and the dialogs and menus open once it has settled.",
            kind: Input,
            schema: object(
                &[
                    ("panel_name", panel_name_schema("at_px is a point of")),
                    ("at_px", at_px_schema()),
                    ("widget", widget_schema()),
                ],
                &[],
            ),
        },
        ToolSpec {
            name: "drag",
            description: "Drag in the window the way a person's mouse does: the pointer moves \
                          to the first point of path_px and a frame is drawn, the button goes \
                          down there in the next, the pointer moves to each later point in a \
                          frame of its own with the button held, the button is released at the \
                          last point in the frame after, and the reply is built from the frame \
                          after that. The points are in the pixels of the window or of \
                          panel_name's body, as click's at_px is. It orbits, pans and zooms \
                          the 3D Viewer as a person's drag does, with the modifiers the \
                          viewport reads, and moves what a person can drag. The reply gives \
                          at_px (the last point, where the pointer stays), hit (the widget \
                          under the press, or null) and the dialogs and menus open afterwards.",
            kind: Input,
            schema: object(
                &[
                    ("panel_name", panel_name_schema("path_px is a path in")),
                    (
                        "mouse_button",
                        json!({
                            "type": "string",
                            "enum": ["left", "middle", "right"],
                            "description":
                                "Which button is held. Omit for left. On Windows the 3D Viewer \
                                 reads which button is down from the operating system, which \
                                 this input does not reach, so there a middle or right drag \
                                 moves the camera as a left drag does.",
                        }),
                    ),
                    ("modifiers", modifiers_schema("for the whole drag")),
                ],
                &[(
                    "path_px",
                    json!({
                        "type": "array",
                        "items": at_px_schema(),
                        "minItems": 2,
                        "maxItems": MAX_DRAG_POINTS,
                        "description":
                            "The points [x, y] the pointer passes through, one per frame: the \
                             press at the first and the release at the last. Every point has \
                             to be inside the target, or the call is refused naming its size \
                             before the button goes down. egui reads a drag from how far the \
                             pointer moves between frames, so a step of a few pixels moves \
                             little, and a long step moves as far as a fast hand.",
                    }),
                )],
            ),
        },
        ToolSpec {
            name: "press_key",
            description: "Press one key and release it in the next frame, with any modifiers \
                          held across both. It goes where a person's key would: to a menu bar \
                          shortcut (command Z undoes, command G opens Go to Point), to an open \
                          dialog (Enter runs it, Escape cancels it), to the text input with \
                          keyboard focus, or with panel_name or at_px to the panel the pointer \
                          is moved over first (I over image_detail toggles its intrinsics \
                          layer). A letter key is a key, not text: type_text types. The reply \
                          gives the dialogs and menus open afterwards and focused, the widget \
                          with keyboard focus, or null.",
            kind: Input,
            schema: object(
                &[
                    ("modifiers", modifiers_schema("while the key is down")),
                    (
                        "panel_name",
                        panel_name_schema(
                            "Move the pointer over this panel's body first, to its centre \
                             unless at_px says where; at_px is a point of",
                        ),
                    ),
                    ("at_px", at_px_schema()),
                ],
                &[(
                    "key",
                    json!({
                        "type": "string",
                        "description":
                            "egui's name for the key: a letter (\"M\"), a digit (\"0\"), \
                             \"Enter\", \"Escape\", \"Tab\", \"Backspace\", \"Delete\", \
                             \"Space\", the arrows (\"ArrowUp\"), \"F1\" to \"F35\", or a \
                             punctuation name (\"Comma\", \"Period\", \"OpenBracket\"). An \
                             unknown name is refused with the whole list.",
                    }),
                )],
            ),
        },
        ToolSpec {
            name: "type_text",
            description: "Type a string into a text input, as one text event, which is what an \
                          input method delivers when it commits. With widget, the text input is \
                          clicked first to focus it; a widget that is not a text input is \
                          refused. Without it, the text goes to the widget with keyboard focus, \
                          and the call is refused when nothing has focus or what has it takes no \
                          text. It does not press Enter: send press_key Enter after it, so the \
                          field's value can be checked first. The reply gives the dialogs and \
                          menus open afterwards and focused, the widget with keyboard focus. The \
                          Action Log records how many characters were typed, never the text.",
            kind: Input,
            schema: object(
                &[("widget", widget_schema())],
                &[(
                    "text",
                    json!({
                        "type": "string",
                        "minLength": 1,
                        "description": "What to type.",
                    }),
                )],
            ),
        },
    ]
}

/// The panel whose body a point is in.
fn panel_name_schema(what: &str) -> Value {
    json!({
        "type": "string",
        "enum": Tab::ALL.map(|tab| tab.wire_name()),
        "description": format!(
            "{what} this panel's body instead of the window, by the name get_window_layout and \
             the layout file use. A panel that is closed or behind another tab is refused \
             naming show_panel."
        ),
    })
}

/// A point of the target.
fn at_px_schema() -> Value {
    json!({
        "type": "array",
        "items": { "type": "number" },
        "minItems": 2,
        "maxItems": 2,
        "description":
            "A point [x, y] in the physical pixels of the target, the space rect_px is in: the \
             window, or panel_name's body. A point outside the target is refused naming its \
             size.",
    })
}

/// A widget from a listing.
fn widget_schema() -> Value {
    json!({
        "type": "string",
        "pattern": "^[0-9a-f]{16}$",
        "description":
            "A widget id from get_widgets or a screenshot's listing. An id no longer drawn is \
             refused naming what now carries the name it was listed under.",
    })
}

/// The modifier keys held down.
fn modifiers_schema(when: &str) -> Value {
    json!({
        "type": "array",
        "items": { "type": "string", "enum": ["shift", "control", "alt", "command"] },
        "uniqueItems": true,
        "description": format!(
            "Keys held {when}. command is Control on Windows and Linux and the Command key on \
             macOS, which is how the viewer's shortcuts are declared."
        ),
    })
}
