// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The six input tools: the pointer and the keyboard, as a person uses them.

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
                          would a person's. The pointer stays where it was left; hover with \
                          leave: true takes it out of the window.",
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
                          and the dialogs and menus open once it has settled. With leave: true \
                          instead, the pointer leaves the window as a person's mouse moving off \
                          it does: nothing is hovered any more, so the 3D Viewer draws no hover \
                          label and no panel highlights what was under the pointer, which is the \
                          state to take a screenshot in; the reply's at_px and hit are null.",
            kind: Input,
            schema: object(
                &[
                    ("panel_name", panel_name_schema("at_px is a point of")),
                    ("at_px", at_px_schema()),
                    ("widget", widget_schema()),
                    (
                        "leave",
                        json!({
                            "type": "boolean",
                            "description":
                                "true moves the pointer out of the window instead of to a \
                                 point, and is given alone, without at_px, widget or \
                                 panel_name. The next click, hover, drag or scroll brings it \
                                 back.",
                        }),
                    ),
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
                                "Which button is held. Omit for left. The 3D Viewer tells a \
                                 middle or right drag apart only by the operating system's \
                                 button state on Windows, which this input does not reach, so \
                                 over it a middle or right drag moves the camera as a left \
                                 drag does.",
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
            name: "scroll",
            description: "Scroll the way a person's mouse wheel or trackpad does: the pointer \
                          moves to the point and a frame is drawn, one scroll event is \
                          delivered there in the next, and the reply is built once egui has \
                          finished spreading the scroll over frames. Aimed as click is, with \
                          at_px and an optional panel_name, or with widget. Give delta_lines \
                          for a mouse wheel (one notch is one line) or delta_px for a trackpad \
                          (physical pixels), as [x, y]: a positive y is the wheel turned away \
                          from you or two fingers moved down, which moves the content down \
                          and shows what is above, and a positive x moves the content right. \
                          The panel under the pointer reads it as it reads a person's: a \
                          wheel zooms the 3D Viewer and Image Detail, a trackpad scroll \
                          orbits the 3D Viewer and pans Image Detail, either moves the Image \
                          Browser's strip, and lists and tables scroll. modifiers are held \
                          for the scroll: control zooms where a panel zooms, shift turns a \
                          list's vertical scroll horizontal, and alt is the 3D Viewer's \
                          target push and pull. The reply gives at_px, hit (the clickable \
                          widget under the pointer, or null) and the dialogs and menus open \
                          afterwards.",
            kind: Input,
            schema: object(
                &[
                    ("panel_name", panel_name_schema("at_px is a point of")),
                    ("at_px", at_px_schema()),
                    ("widget", widget_schema()),
                    (
                        "delta_lines",
                        scroll_delta_schema(
                            "How far a mouse wheel turns, in lines: one notch is 1, and egui \
                             scrolls a list 40 points a line. Give this or delta_px.",
                        ),
                    ),
                    (
                        "delta_px",
                        scroll_delta_schema(
                            "How far two fingers move on a trackpad, in the window's physical \
                             pixels, which egui reads in points as it reads a trackpad's. Give \
                             this or delta_lines.",
                        ),
                    ),
                    ("modifiers", modifiers_schema("during the scroll")),
                ],
                &[],
            ),
        },
        ToolSpec {
            name: "press_key",
            description: "Press one key and release it in the next frame, with any modifiers \
                          held across both. It goes where a person's key would: to a menu bar \
                          shortcut (command Z undoes, command G opens Go to Point), to an open \
                          dialog (Enter runs it, Escape cancels it), to the text input or \
                          number box with keyboard focus, or with panel_name or at_px to the \
                          panel the pointer is moved over first (I over image_detail toggles \
                          its intrinsics layer). A letter key is a key, not text: type_text types. The reply \
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
            description: "Type a string into a text input or a number box (role text_input or \
                          spin_button), as one text event, which is what an input method \
                          delivers when it commits. With widget, the field is clicked first to \
                          focus it, as a person's click does: a number box opens for typing with \
                          its number selected, so the text replaces it, and a text input takes \
                          the text where the click put the cursor. A widget of any other role is \
                          refused. Without widget, the text goes to the widget with keyboard \
                          focus, and the call is refused when nothing has focus or what has it \
                          takes no text. It does not press Enter: send press_key Enter after it, \
                          so the field can be checked first. A number box takes a typed number \
                          as it is typed, or, like the Track View threshold boxes, only when \
                          Enter is pressed or focus leaves it; text that is not a number leaves \
                          the value as it was, and a number is clamped to the box's range. Type \
                          the number as the box shows it, which can differ from its value: a \
                          Track View ZNCC box with value 0.7 shows 70 percent and takes 85 for \
                          0.85. The reply gives the dialogs and menus open afterwards and \
                          focused, the widget with keyboard focus. The Action Log records how many characters were typed, never the text.",
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

/// A scroll's distance, `[x, y]`, in the units `what` names.
fn scroll_delta_schema(what: &str) -> Value {
    json!({
        "type": "array",
        "items": { "type": "number" },
        "minItems": 2,
        "maxItems": 2,
        "description": format!(
            "{what} Positive y moves the content down, showing what is above (the wheel turned \
             away from you); positive x moves it right, showing what is to the left. Not both \
             0."
        ),
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
