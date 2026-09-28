// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Synthetic pointer and keyboard input: `click`, `hover`, `press_key` and
//! `type_text`.
//!
//! The input goes into egui's `RawInput.events` as the same `egui::Event`s
//! `egui_winit` makes from a real mouse and keyboard, so a click reaches the
//! widget through egui's own hit test and a key reaches whatever reads it,
//! exactly as a person's would. Nothing here calls into a panel.
//!
//! A command is a short list of [`Step`]s, one per frame. Before each egui
//! pass [`PendingInput::before_pass`] turns the next step into that frame's
//! events; after the pass [`PendingInput::after_pass`] is handed the widgets
//! the frame drew. The first step of a command that aims at something only
//! looks: a widget id and a point in a panel are resolved against a frame
//! that has been laid out, and egui resolves a press against the widget
//! rectangles of the frame before it, so every event is aimed with the frame
//! it will be resolved against. The reply is built from the frame drawn after
//! the last event, which is the frame a menu or dialog the input opened first
//! appears in.
//!
//! [`InputQueue`] is the part that lives on `App`: commands run one after
//! another, each to its reply, and a refused one is answered in the frame it
//! was refused in so the next can start there.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use serde_json::{json, Map, Value};

use super::widgets::{widget_id_text, Widget, WidgetFrame};
use super::{Reply, ToolError, ToolOutput};
use crate::action_log::{Actor, Kind};
use crate::dock::Tab;
use crate::state::AppState;

/// One of the four input tools, parsed.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum InputCommand {
    /// Press and release a mouse button, once or twice.
    Click {
        target: PointerTarget,
        button: egui::PointerButton,
        /// 1 or 2; the parse refuses anything else.
        count: u8,
        modifiers: ModifierKeys,
    },
    /// Move the pointer and leave it there.
    Hover { target: PointerTarget },
    /// Press one key and release it in the next frame.
    PressKey {
        key: egui::Key,
        modifiers: ModifierKeys,
        /// Where the pointer goes first, for a key read over a panel. `None`
        /// leaves the pointer where it is.
        over: Option<PointerTarget>,
    },
    /// Enter a string as one `Text` event.
    TypeText {
        text: String,
        /// The text input to click first, by the value of its id. `None`
        /// types into whatever has keyboard focus.
        widget: Option<u64>,
    },
}

impl InputCommand {
    /// The tool this came from, as the wire names it.
    pub(crate) fn tool_name(&self) -> &'static str {
        match self {
            InputCommand::Click { .. } => "click",
            InputCommand::Hover { .. } => "hover",
            InputCommand::PressKey { .. } => "press_key",
            InputCommand::TypeText { .. } => "type_text",
        }
    }

    /// The panel the command names, if any.
    pub(crate) fn panel(&self) -> Option<Tab> {
        match self {
            InputCommand::Click { target, .. } | InputCommand::Hover { target } => target.panel(),
            InputCommand::PressKey { over, .. } => over.as_ref().and_then(PointerTarget::panel),
            InputCommand::TypeText { .. } => None,
        }
    }

    /// The point the command names in its target's pixels, where it names one.
    pub(crate) fn at_px(&self) -> Option<[f64; 2]> {
        match self {
            InputCommand::Click { target, .. }
            | InputCommand::Hover { target }
            | InputCommand::PressKey {
                over: Some(target), ..
            } => match target {
                PointerTarget::At { at_px, .. } => *at_px,
                PointerTarget::Widget(_) => None,
            },
            _ => None,
        }
    }

    fn steps(&self) -> VecDeque<Step> {
        use Step::*;
        match self {
            InputCommand::Click { count, .. } => {
                let mut steps = vec![Look, Move, Press];
                if *count == 2 {
                    steps.push(PressAgain);
                }
                steps.push(Settle);
                steps.into()
            }
            InputCommand::Hover { .. } => [Look, Move, Settle].into(),
            InputCommand::PressKey { over: Some(_), .. } => {
                [Look, Move, KeyDown, KeyUp, Settle].into()
            }
            InputCommand::PressKey { over: None, .. } => [KeyDown, KeyUp, Settle].into(),
            InputCommand::TypeText {
                widget: Some(_), ..
            } => [Look, Move, Press, Text, Settle].into(),
            InputCommand::TypeText { widget: None, .. } => [Look, Text, Settle].into(),
        }
    }

    /// Where the pointer goes, for the steps that move it.
    fn target(&self) -> Option<PointerTarget> {
        match self {
            InputCommand::Click { target, .. } | InputCommand::Hover { target } => {
                Some(target.clone())
            }
            InputCommand::PressKey { over, .. } => over.clone(),
            InputCommand::TypeText { widget, .. } => widget.map(PointerTarget::Widget),
        }
    }

    fn modifiers(&self) -> ModifierKeys {
        match self {
            InputCommand::Click { modifiers, .. } | InputCommand::PressKey { modifiers, .. } => {
                *modifiers
            }
            InputCommand::Hover { .. } | InputCommand::TypeText { .. } => ModifierKeys::default(),
        }
    }
}

/// Where a pointer tool aims.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum PointerTarget {
    /// A point of the window, or of one panel's body, in that target's
    /// physical pixels. `at_px: None` is the centre of the target, which only
    /// `press_key` with a `panel_name` and no point asks for.
    At {
        panel: Option<Tab>,
        at_px: Option<[f64; 2]>,
    },
    /// The centre of a widget from a listing, by the value of its id.
    Widget(u64),
}

impl PointerTarget {
    fn panel(&self) -> Option<Tab> {
        match self {
            PointerTarget::At { panel, .. } => *panel,
            PointerTarget::Widget(_) => None,
        }
    }
}

/// The modifier keys a call holds down, by the wire's four names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct ModifierKeys {
    pub(crate) shift: bool,
    pub(crate) control: bool,
    pub(crate) alt: bool,
    /// Control on Windows and Linux and ⌘ on macOS, which is how the viewer's
    /// own shortcuts are declared.
    pub(crate) command: bool,
}

impl ModifierKeys {
    /// The four names, in the order a refusal lists them.
    pub(crate) const WIRE_NAMES: [&'static str; 4] = ["shift", "control", "alt", "command"];

    /// Hold the key `name` spells, or say it spells none.
    pub(crate) fn add(&mut self, name: &str) -> bool {
        match name {
            "shift" => self.shift = true,
            "control" => self.control = true,
            "alt" => self.alt = true,
            "command" => self.command = true,
            _ => return false,
        }
        true
    }

    fn is_empty(self) -> bool {
        self == Self::default()
    }

    /// The modifiers as `egui_winit` reports them on this platform: `command`
    /// is Control everywhere but macOS, where it is ⌘.
    pub(crate) fn to_egui(self) -> egui::Modifiers {
        let mac = cfg!(target_os = "macos");
        let ctrl = self.control || (self.command && !mac);
        let mac_cmd = self.command && mac;
        egui::Modifiers {
            alt: self.alt,
            ctrl,
            shift: self.shift,
            mac_cmd,
            command: if mac { mac_cmd } else { ctrl },
        }
    }

    /// `Ctrl+Shift`, as egui names the keys on this platform, or nothing.
    fn text(self) -> String {
        egui::ModifierNames::NAMES.format(&self.to_egui(), cfg!(target_os = "macos"))
    }
}

/// One frame's worth of a command.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Step {
    /// Deliver nothing: draw a frame to aim against.
    Look,
    /// Move the pointer to the target.
    Move,
    /// Press and release the button, in one frame.
    Press,
    /// The second press and release of a double click.
    PressAgain,
    /// Press the key.
    KeyDown,
    /// Release it.
    KeyUp,
    /// Deliver the text.
    Text,
    /// Release the modifiers, and draw the frame the reply is built from.
    Settle,
}

/// Where the pointer was sent.
#[derive(Debug, Clone, Copy)]
struct Aim {
    /// In window points, as egui's input takes it.
    pos: egui::Pos2,
    /// In window pixels, for the hit test against a captured frame.
    window_px: [f32; 2],
    /// In the target's pixels, for the reply and the Action Log.
    at_px: [f64; 2],
}

/// An input command on its way through the frames.
pub(crate) struct PendingInput {
    command: InputCommand,
    steps: VecDeque<Step>,
    /// The step whose events went into the frame being drawn, until that
    /// frame's widgets arrive.
    drawn: Option<Step>,
    /// The widgets of the last frame drawn for this command.
    frame: Option<Arc<WidgetFrame>>,
    aim: Option<Aim>,
    /// The `hit` block of the reply.
    hit: Value,
    /// The modifiers egui held before this command pressed its own, restored
    /// when it is done.
    before: Option<egui::Modifiers>,
}

impl PendingInput {
    pub(crate) fn new(command: InputCommand) -> Self {
        Self {
            steps: command.steps(),
            command,
            drawn: None,
            frame: None,
            aim: None,
            hit: Value::Null,
            before: None,
        }
    }

    /// The events of the coming frame, or the refusal that ends the command.
    ///
    /// `held` is the modifiers egui holds now. A refusal records one failed
    /// Action Log row, as the agent's; it is reached before any input has been
    /// delivered, so nothing the command did is left half done.
    pub(crate) fn before_pass(
        &mut self,
        state: &mut AppState,
        held: egui::Modifiers,
        names: &WidgetNames,
    ) -> Result<Vec<egui::Event>, ToolError> {
        let events = self.step(state, held, names);
        if let Err(error) = &events {
            state.action_log.fail_as(
                Actor::Mcp,
                Kind::Input,
                format!("{} failed: {error}", self.command.tool_name()),
            );
        }
        events
    }

    /// Whether the frame now being drawn carries one of this command's steps,
    /// and so has to have its widgets read.
    pub(crate) fn wants_frame(&self) -> bool {
        self.drawn.is_some()
    }

    /// Take the widgets of the frame just drawn. The reply, once that frame
    /// was the last one the command needed.
    pub(crate) fn after_pass(&mut self, frame: Arc<WidgetFrame>) -> Option<Reply> {
        let drawn = self.drawn.take()?;
        self.frame = Some(frame);
        (drawn == Step::Settle).then(|| self.reply())
    }

    fn step(
        &mut self,
        state: &mut AppState,
        held: egui::Modifiers,
        names: &WidgetNames,
    ) -> Result<Vec<egui::Event>, ToolError> {
        let Some(step) = self.steps.pop_front() else {
            return Ok(Vec::new());
        };
        self.drawn = Some(step);
        let modifiers = self.command.modifiers().to_egui();
        match step {
            Step::Look => Ok(Vec::new()),
            Step::Move => {
                let aim = self.aim(names)?;
                self.aim = Some(aim);
                // What the pointer arrives over, from the frame egui resolves
                // the move against.
                let frame = self.frame.clone().expect("a look precedes every move");
                if let InputCommand::Hover { .. } = self.command {
                    self.hit = frame.hit_block(frame.topmost_at(aim.window_px));
                    let on = on_text(frame.name_at(aim.window_px));
                    let text = format!("hover {}{on}", self.where_text(aim));
                    state.action_log.record_as(Actor::Mcp, Kind::Input, text);
                }
                Ok(vec![egui::Event::PointerMoved(aim.pos)])
            }
            Step::Press | Step::PressAgain => {
                let aim = self.aim.expect("a move precedes every press");
                let (button, count) = match &self.command {
                    InputCommand::Click { button, count, .. } => (*button, *count),
                    _ => (egui::PointerButton::Primary, 1),
                };
                let mut events = Vec::new();
                if step == Step::Press {
                    // The press is resolved against the frame the pointer
                    // arrived in, so that is the frame the hit is read from.
                    let frame = self.frame.clone().expect("a move precedes every press");
                    self.hit = frame.hit_block(frame.topmost_at(aim.window_px));
                    if let InputCommand::Click { modifiers, .. } = &self.command {
                        let mut text = String::from("click");
                        if button == egui::PointerButton::Secondary {
                            text.push_str(" right");
                        } else if button == egui::PointerButton::Middle {
                            text.push_str(" middle");
                        }
                        if count == 2 {
                            text.push_str(" twice");
                        }
                        text.push(' ');
                        text.push_str(&self.where_text(aim));
                        if !modifiers.is_empty() {
                            text.push_str(&format!(" with {}", modifiers.text()));
                        }
                        text.push_str(&on_text(frame.name_at(aim.window_px)));
                        state.action_log.record_as(Actor::Mcp, Kind::Input, text);
                    }
                    events.extend(self.hold(held));
                }
                for pressed in [true, false] {
                    events.push(egui::Event::PointerButton {
                        pos: aim.pos,
                        button,
                        pressed,
                        modifiers,
                    });
                }
                Ok(events)
            }
            Step::KeyDown | Step::KeyUp => {
                let InputCommand::PressKey {
                    key, modifiers: m, ..
                } = &self.command
                else {
                    return Ok(Vec::new());
                };
                let key = *key;
                let mut events = Vec::new();
                if step == Step::KeyDown {
                    let mut text = String::from("press_key ");
                    if !m.is_empty() {
                        text.push_str(&m.text());
                        text.push('+');
                    }
                    text.push_str(key.name());
                    if let Some(aim) = self.aim {
                        text.push_str(" over ");
                        // A key sent over a panel's centre names only the
                        // panel: the point is where the call put it, not what
                        // it asked for.
                        match self.command.at_px() {
                            Some(_) => text.push_str(&self.where_text(aim)),
                            None => text.push_str(
                                self.command
                                    .panel()
                                    .map_or("window", |panel| panel.wire_name()),
                            ),
                        }
                    }
                    state.action_log.record_as(Actor::Mcp, Kind::Input, text);
                    events.extend(self.hold(held));
                }
                events.push(egui::Event::Key {
                    key,
                    physical_key: None,
                    pressed: step == Step::KeyDown,
                    repeat: false,
                    modifiers,
                });
                Ok(events)
            }
            Step::Text => {
                let InputCommand::TypeText { text, widget } = &self.command else {
                    return Ok(Vec::new());
                };
                let frame = self.frame.clone().expect("a frame precedes the text");
                let into =
                    match widget {
                        // The click before this frame should have focused it.
                        Some(value) => match frame.focused() {
                            Some(focused) if focused.id.value() == *value => focused,
                            _ => {
                                return Err(ToolError::new(format!(
                                "Clicking widget {value:016x} did not give it keyboard focus, so \
                                 the text would have gone nowhere. Nothing was typed."
                            )))
                            }
                        },
                        None => match frame.focused() {
                            Some(focused) if focused.takes_text() => focused,
                            Some(focused) => {
                                return Err(ToolError::new(format!(
                                    "The widget with keyboard focus is {}, which is not a text \
                                 input, so the text would be dropped. Name the text input with \
                                 widget, or click it first.",
                                    frame.describe(focused)
                                )))
                            }
                            None => return Err(ToolError::new(
                                "Nothing has keyboard focus, so the text would be dropped with \
                                 no sign. Name the text input with widget, or click it first.",
                            )),
                        },
                    };
                let count = text.chars().count();
                let characters = if count == 1 {
                    "character"
                } else {
                    "characters"
                };
                let input = match into.name() {
                    Some(name) => format!("the text input \"{name}\""),
                    None => "the text input".to_string(),
                };
                state.action_log.record_as(
                    Actor::Mcp,
                    Kind::Input,
                    format!(
                        "type_text {count} {characters} into {input} {}",
                        text_place(&frame, into)
                    ),
                );
                Ok(vec![egui::Event::Text(text.clone())])
            }
            Step::Settle => Ok(self
                .before
                .map(egui::Event::ModifiersChanged)
                .into_iter()
                .collect()),
        }
    }

    /// Press this command's modifiers, remembering what egui held before, or
    /// nothing when it holds none.
    fn hold(&mut self, held: egui::Modifiers) -> Option<egui::Event> {
        let modifiers = self.command.modifiers();
        if modifiers.is_empty() {
            return None;
        }
        self.before.get_or_insert(held);
        Some(egui::Event::ModifiersChanged(modifiers.to_egui()))
    }

    /// Resolve the target against the frame last drawn.
    fn aim(&self, names: &WidgetNames) -> Result<Aim, ToolError> {
        let frame = self.frame.as_deref().expect("a look precedes every move");
        let target = self
            .command
            .target()
            .expect("only a command with a target moves the pointer");
        match target {
            PointerTarget::At { panel, at_px } => {
                let [left, top, width, height] = frame.target_px(panel).map_err(|_| {
                    ToolError::new(format!(
                        "The {} panel was not laid out in the frame this call was aimed in. It \
                         may have been closed since the call was made.",
                        panel.map_or("window's", |panel| panel.title())
                    ))
                })?;
                let at = at_px.unwrap_or([f64::from(width) / 2.0, f64::from(height) / 2.0]);
                check_inside(at, [width, height], panel)?;
                let window_px = [
                    (f64::from(left) + at[0]) as f32,
                    (f64::from(top) + at[1]) as f32,
                ];
                Ok(Aim {
                    pos: frame.to_points(window_px),
                    window_px,
                    at_px: at,
                })
            }
            PointerTarget::Widget(value) => {
                let widget = frame
                    .find_value(value)
                    .ok_or_else(|| unknown_widget(frame, value, names))?;
                if matches!(self.command, InputCommand::TypeText { .. }) && !widget.takes_text() {
                    return Err(ToolError::new(format!(
                        "type_text types into a text input, and widget {value:016x} is {}{}.",
                        frame.describe(widget),
                        widget
                            .role_text()
                            .map(|role| format!(", a {}", role.replace('_', " ")))
                            .unwrap_or_default(),
                    )));
                }
                let window_px = frame.centre_px(widget);
                if let Some(over) = frame.covering(widget, window_px) {
                    return Err(ToolError::new(format!(
                        "Widget {value:016x} ({}) is covered at its centre by {over}, so a \
                         person could not press it there either. Close what is over it first.",
                        frame.describe(widget)
                    )));
                }
                Ok(Aim {
                    pos: frame.to_points(window_px),
                    window_px,
                    at_px: [
                        f64::from(window_px[0].round()),
                        f64::from(window_px[1].round()),
                    ],
                })
            }
        }
    }

    /// `scene 46,53`, or `window 400,12`: the target and the point in it.
    fn where_text(&self, aim: Aim) -> String {
        let target = self
            .command
            .target()
            .and_then(|target| target.panel())
            .map_or("window", |panel| panel.wire_name());
        format!(
            "{target} {},{}",
            number_text(aim.at_px[0]),
            number_text(aim.at_px[1])
        )
    }

    /// The reply, built from the frame drawn after the last event.
    fn reply(&self) -> Reply {
        let frame = self
            .frame
            .as_deref()
            .expect("the reply follows a drawn frame");
        // A click can close the panel it was aimed at; the dialogs and menus
        // are then those of the window.
        let panel = self
            .command
            .panel()
            .filter(|panel| frame.target_px(Some(*panel)).is_ok());
        let listing = frame.listing(panel, None)?;
        let mut out = Map::new();
        match &self.command {
            InputCommand::Click { .. } | InputCommand::Hover { .. } => {
                let at = self.aim.map_or([0.0, 0.0], |aim| aim.at_px);
                out.insert(
                    "at_px".into(),
                    json!([number_value(at[0]), number_value(at[1])]),
                );
                out.insert("hit".into(), self.hit.clone());
                out.insert("dialogs".into(), listing["dialogs"].clone());
                out.insert("menus".into(), listing["menus"].clone());
            }
            InputCommand::PressKey { .. } | InputCommand::TypeText { .. } => {
                out.insert("dialogs".into(), listing["dialogs"].clone());
                out.insert("menus".into(), listing["menus"].clone());
                out.insert("focused".into(), frame.focused_entry(panel));
            }
        }
        Ok(ToolOutput::Json(Value::Object(out)))
    }
}

/// Refuse a point outside its target, naming the target's size.
pub(crate) fn check_inside(
    at: [f64; 2],
    [width, height]: [u32; 2],
    panel: Option<Tab>,
) -> Result<(), ToolError> {
    let inside =
        at[0] >= 0.0 && at[1] >= 0.0 && at[0] < f64::from(width) && at[1] < f64::from(height);
    if inside {
        return Ok(());
    }
    let target = match panel {
        None => "The window".to_string(),
        Some(panel) => format!("The {} panel's body", panel.title()),
    };
    Err(ToolError::new(format!(
        "at_px [{}, {}] is outside the target. {target} is {width}×{height} pixels, so a point \
         has to be at least 0 and less than {width} across and {height} down.",
        number_text(at[0]),
        number_text(at[1]),
    )))
}

/// The refusal for a widget id the frame did not draw, naming what now carries
/// the name it was listed under.
fn unknown_widget(frame: &WidgetFrame, value: u64, names: &WidgetNames) -> ToolError {
    let id = format!("{value:016x}");
    let Some(name) = names.get(value) else {
        return ToolError::new(format!(
            "No widget {id} is drawn in the window, and no listing has reported one. get_widgets \
             lists what is drawn."
        ));
    };
    let now: Vec<String> = frame
        .named(name)
        .into_iter()
        .take(3)
        .map(|widget| format!("{} {}", widget_id_text(widget.id), frame.place(widget)))
        .collect();
    let closest = match now.as_slice() {
        [] => format!("and nothing named \"{name}\" is drawn now"),
        [one] => format!("and the widget named \"{name}\" now is {one}"),
        many => format!(
            "and the widgets named \"{name}\" now are {}",
            many.join("; ")
        ),
    };
    ToolError::new(format!(
        "No widget {id} is drawn in the window. It was named \"{name}\" when it was listed, \
         {closest}."
    ))
}

/// ` on "demo"`, or nothing for a point with no name on it.
fn on_text(name: Option<&str>) -> String {
    name.map(|name| format!(" on \"{name}\""))
        .unwrap_or_default()
}

/// Where a text input is, for its Action Log row: the dialog it is in by its
/// title, or the panel.
fn text_place(frame: &WidgetFrame, widget: &Widget) -> String {
    let place = frame.place(widget);
    // `in the Go to Point dialog` reads as `in Go to Point` in a row that has
    // already said what the widget is.
    match place
        .strip_prefix("in the ")
        .and_then(|rest| rest.strip_suffix(" dialog"))
    {
        Some(title) => format!("in {title}"),
        None => place,
    }
}

/// A coordinate as a person writes it: `46`, or `46.5`.
fn number_text(n: f64) -> String {
    if n.fract() == 0.0 && n.abs() < 1e15 {
        format!("{}", n as i64)
    } else {
        format!("{n}")
    }
}

/// A coordinate on the wire: an integer when it is whole.
fn number_value(n: f64) -> Value {
    if n.fract() == 0.0 && n.abs() < 1e15 {
        json!(n as i64)
    } else {
        json!(n)
    }
}

/// The names widgets had when a listing reported them, by the value of their
/// id, so a call naming a widget that has since gone can say what now carries
/// its name.
#[derive(Default)]
pub(crate) struct WidgetNames(HashMap<u64, String>);

impl WidgetNames {
    /// More ids than any session lists before it forgets them all and starts
    /// again: ids from a listing long past are not worth the memory.
    const LIMIT: usize = 20_000;

    /// Remember every named widget of a captured frame.
    pub(crate) fn remember(&mut self, frame: &WidgetFrame) {
        if self.0.len() > Self::LIMIT {
            self.0.clear();
        }
        for (value, name) in frame.names() {
            self.0.insert(value, name.to_string());
        }
    }

    fn get(&self, value: u64) -> Option<&str> {
        self.0.get(&value).map(String::as_str)
    }
}

/// The input commands waiting on `App`, run one after another.
#[derive(Default)]
pub(crate) struct InputQueue {
    pending: VecDeque<(PendingInput, tokio::sync::oneshot::Sender<Reply>)>,
    names: WidgetNames,
}

impl InputQueue {
    pub(crate) fn push(&mut self, input: PendingInput, reply: tokio::sync::oneshot::Sender<Reply>) {
        self.pending.push_back((input, reply));
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }

    /// This frame's events: the next step of the command at the front. A
    /// command refused here is answered now, and the one behind it starts in
    /// the same frame.
    pub(crate) fn before_pass(
        &mut self,
        state: &mut AppState,
        held: egui::Modifiers,
    ) -> Vec<egui::Event> {
        while let Some((input, _)) = self.pending.front_mut() {
            match input.before_pass(state, held, &self.names) {
                Ok(events) => return events,
                Err(error) => {
                    if let Some((_, reply)) = self.pending.pop_front() {
                        let _ = reply.send(Err(error));
                    }
                }
            }
        }
        Vec::new()
    }

    /// Whether this frame delivered a step, and so has to be read.
    pub(crate) fn wants_frame(&self) -> bool {
        self.pending
            .front()
            .is_some_and(|(input, _)| input.wants_frame())
    }

    /// Hand the frame just drawn to the command at the front, and answer it
    /// if that was its last frame.
    pub(crate) fn after_pass(&mut self, frame: &Arc<WidgetFrame>) {
        let Some((input, _)) = self.pending.front_mut() else {
            return;
        };
        if let Some(answer) = input.after_pass(frame.clone()) {
            if let Some((_, reply)) = self.pending.pop_front() {
                let _ = reply.send(answer);
            }
        }
    }

    /// Remember the names a listing that is not an input's reported.
    pub(crate) fn remember(&mut self, frame: &WidgetFrame) {
        self.names.remember(frame);
    }
}
