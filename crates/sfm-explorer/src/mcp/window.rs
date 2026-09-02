// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The window tools: what the window is, and the one command that changes it.
//!
//! Everything an agent can learn or change about the window the human is
//! looking at — its state, size, position, focus, and the monitors it could
//! sit on — travels through the two types here: [`WindowInfo`], the snapshot
//! `AppState::window` holds and every reply renders, and [`WindowHost`], the
//! two-method seam the one `winit` call goes through.
//!
//! **Why a host trait rather than a third [`super::Outcome`] variant.**
//! `screenshot` defers because its answer does not exist until the frame has
//! been rendered. A window change is the other case: it can be applied on the
//! spot, its effect is wanted in *this* frame's layout, and its answer is a
//! read-back a fake can produce as well as a real window. Deferring it would
//! apply it after the present — a frame late — and would leave a `set_window`
//! followed by a `get_window` in one batch answering with the old window. A
//! trait with two methods keeps [`super::apply_with_window`] free of `winit` exactly as it
//! is free of `wgpu`, and puts the refusals, the application order and the
//! Action Log line under headless test against a fake.

use serde_json::{json, Value};

use super::{JsonReply, ToolError};
use crate::action_log::Kind;
use crate::state::AppState;

/// What the user sees the window as: one word for winit's three flags.
///
/// A window can be minimized *and* maximized at once — a maximized window the
/// user minimized, which comes back maximized — so the flags are collapsed in
/// a fixed order of precedence ([`WindowState::of`]). The question an agent
/// asks is "can the human see this window, and how much of the desktop is it";
/// the flags underneath are the viewer's business.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum WindowState {
    /// None of the three flags: an ordinary window with a size and a position
    /// of its own. The word the user and Win32's `SW_SHOWNORMAL` both use.
    #[default]
    Normal,
    Maximized,
    Minimized,
    Fullscreen,
}

impl WindowState {
    /// Every state, in the order the wire lists them.
    pub(crate) const ALL: [WindowState; 4] = [
        WindowState::Normal,
        WindowState::Maximized,
        WindowState::Minimized,
        WindowState::Fullscreen,
    ];

    /// The one word that governs what the user sees: minimized over
    /// fullscreen over maximized over normal.
    pub(crate) fn of(minimized: bool, fullscreen: bool, maximized: bool) -> Self {
        if minimized {
            WindowState::Minimized
        } else if fullscreen {
            WindowState::Fullscreen
        } else if maximized {
            WindowState::Maximized
        } else {
            WindowState::Normal
        }
    }

    pub(crate) fn wire_name(self) -> &'static str {
        match self {
            WindowState::Normal => "normal",
            WindowState::Maximized => "maximized",
            WindowState::Minimized => "minimized",
            WindowState::Fullscreen => "fullscreen",
        }
    }

    pub(crate) fn from_wire_name(name: &str) -> Option<Self> {
        WindowState::ALL
            .into_iter()
            .find(|state| state.wire_name() == name)
    }

    /// Every state name, as a refusal lists them.
    pub(crate) fn all_wire_names() -> String {
        WindowState::ALL
            .iter()
            .map(|state| state.wire_name())
            .collect::<Vec<_>>()
            .join(", ")
    }

    /// The Action Log phrase for arriving in this state.
    ///
    /// `Restored window` rather than anything with "normal" in it: what the
    /// person watching sees is the window coming back from wherever it was.
    fn log_text(self) -> &'static str {
        match self {
            WindowState::Normal => "Restored window",
            WindowState::Maximized => "Maximized window",
            WindowState::Minimized => "Minimized window",
            WindowState::Fullscreen => "Made window fullscreen",
        }
    }
}

/// One monitor, as the wire reports it.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct MonitorInfo {
    /// The platform's name for it, where it has one.
    pub(crate) name: Option<String>,
    /// Top-left corner in physical pixels, in desktop coordinates.
    pub(crate) position: [i32; 2],
    /// Physical pixels.
    pub(crate) size: [u32; 2],
    pub(crate) scale_factor: f64,
}

/// The window as last observed: the block `get_window` renders, minus the
/// monitor list.
///
/// **Physical pixels throughout**, with the scale factor beside them. They are
/// what `winit` reports natively, what the view block's `viewport_px` and a
/// screenshot's dimensions are in, and what a monitor's size is in — so an
/// agent comparing "the window" against "the picture I was handed" compares
/// like with like. The logical size is rendered under `derived`, next to the
/// scale factor it comes from.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct WindowInfo {
    pub(crate) state: WindowState,
    pub(crate) focused: bool,
    pub(crate) scale_factor: f64,
    /// `None` where the platform will not say — Wayland does not tell a window
    /// where it is. Reported as `null` rather than as `[0, 0]`, and a
    /// `set_window` `outer_position` fails there for the same reason.
    pub(crate) outer_position: Option<[i32; 2]>,
    pub(crate) outer_size: [u32; 2],
    pub(crate) inner_size: [u32; 2],
    pub(crate) monitor: Option<MonitorInfo>,
}

/// The pieces one `set_window` call carried. `None` is "leave it".
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct WindowChange {
    pub(crate) state: Option<WindowState>,
    /// Physical pixels, desktop coordinates.
    pub(crate) outer_position: Option<[i32; 2]>,
    /// Physical pixels, the drawable area.
    pub(crate) inner_size: Option<[u32; 2]>,
    /// Whether to ask for the foreground. Not an `Option`: `focus: false` is
    /// refused at the parse, so the only thing this field can say is "yes".
    pub(crate) focus: bool,
}

impl WindowChange {
    /// Whether the call asked for anything at all.
    pub(crate) fn is_empty(&self) -> bool {
        *self == WindowChange::default()
    }

    /// Whether the call carries geometry, which needs a normal window.
    pub(crate) fn has_geometry(&self) -> bool {
        self.outer_position.is_some() || self.inner_size.is_some()
    }
}

/// What [`super::apply_with_window`] needs from the window, and all it needs.
///
/// Two methods, so the one command that must reach `winit` goes through a seam
/// a test can stand in for. Implemented for `Arc<winit::window::Window>` in
/// `super::frame`, by `NoWindow` for the windowless case, and by a fake in
/// the tests.
pub(crate) trait WindowHost {
    /// Apply the pieces in the fixed order the spec states: state, position,
    /// size, focus.
    ///
    /// `Err` only for something the platform cannot do at all — a position on
    /// Wayland, or no window to change. A clamp is not an error: the read-back
    /// reports what the window became.
    fn apply(&mut self, change: &WindowChange) -> Result<(), ToolError>;

    /// The window as it is now, and every monitor, the current one first.
    fn observe(&self) -> Option<(WindowInfo, Vec<MonitorInfo>)>;
}

/// The host `super::apply` uses: there is no window here.
///
/// Which is the headless case, and it refuses rather than pretending — a
/// caller that forgot to pass a real host fails loudly rather than silently
/// exercising nothing. Compiled with `super::apply`, its only caller.
#[cfg(test)]
pub(crate) struct NoWindow;

#[cfg(test)]
impl WindowHost for NoWindow {
    fn apply(&mut self, _change: &WindowChange) -> Result<(), ToolError> {
        Err(ToolError::new(NO_WINDOW))
    }

    fn observe(&self) -> Option<(WindowInfo, Vec<MonitorInfo>)> {
        None
    }
}

/// What a window tool says when there is no window behind it.
pub(super) const NO_WINDOW: &str =
    "This viewer has no window to read or change — the window tools need a running viewer.";

// ── The tools ────────────────────────────────────────────────────────────

/// `get_window`: the window block, with every monitor.
pub(super) fn get_window(state: &mut AppState, host: &dyn WindowHost) -> JsonReply {
    let (info, monitors) = host.observe().ok_or_else(|| ToolError::new(NO_WINDOW))?;
    // A read is also the freshest observation anyone has, so the snapshot the
    // rest of the surface reads is brought up to date with it.
    state.window = Some(info.clone());
    Ok(json!({ "window": block(&info, Some(&monitors)) }))
}

/// `set_window`: the pieces one call carried, applied in order and read back.
pub(super) fn set_window(
    state: &mut AppState,
    host: &mut dyn WindowHost,
    change: &WindowChange,
) -> JsonReply {
    // Geometry belongs to the window manager unless the window is normal, and
    // a size the OS will immediately overrule is a call with no honest answer.
    // The half of that rule the parse cannot see is this one: the state the
    // window is *already* in. The other half — `minimized` with a size in one
    // call — is refused in `tools::parse`, where the call's own fields are.
    if change.has_geometry() {
        let resulting = change
            .state
            .or_else(|| state.window.as_ref().map(|info| info.state))
            .unwrap_or_default();
        if resulting != WindowState::Normal {
            return Err(ToolError::new(format!(
                "The window is {}; send state: \"normal\" in the same call to move or resize it.",
                resulting.wire_name()
            )));
        }
    }

    host.apply(change)?;
    let (info, monitors) = host.observe().ok_or_else(|| ToolError::new(NO_WINDOW))?;
    state.window = Some(info.clone());
    // What was *asked for*, which is what the Action Log records: the
    // read-back below can be a frame early on a platform that defers window
    // changes, and a row reading 1598×898 for a call that said 1600×900 would
    // be reporting the platform rather than the action.
    state.action_log.record(Kind::Window, log_text(change));
    Ok(json!({ "window": block(&info, Some(&monitors)) }))
}

/// One row for one call: the pieces it carried, in application order, joined
/// with `; ` — `Restored window; resized window to 1600×900`.
pub(super) fn log_text(change: &WindowChange) -> String {
    let mut pieces: Vec<String> = Vec::new();
    if let Some(state) = change.state {
        pieces.push(state.log_text().to_string());
    }
    if let Some([x, y]) = change.outer_position {
        pieces.push(format!("Moved window to ({x}, {y})"));
    }
    if let Some([width, height]) = change.inner_size {
        pieces.push(format!("Resized window to {width}×{height}"));
    }
    if change.focus {
        pieces.push("Focused window".to_string());
    }
    pieces
        .into_iter()
        .enumerate()
        .map(|(index, piece)| if index == 0 { piece } else { lower(&piece) })
        .collect::<Vec<_>>()
        .join("; ")
}

/// A phrase that is no longer first in its sentence.
fn lower(text: &str) -> String {
    let mut chars = text.chars();
    match chars.next() {
        Some(first) => first.to_lowercase().collect::<String>() + chars.as_str(),
        None => String::new(),
    }
}

// ── The JSON ─────────────────────────────────────────────────────────────

/// The `window` block: what `get_scene` embeds, and — with `monitors`
/// alongside — what the two window tools return.
pub(super) fn block(info: &WindowInfo, monitors: Option<&[MonitorInfo]>) -> Value {
    // A scale factor of zero cannot come from winit, but the logical size is a
    // division and a reply is not the place to find that out.
    let scale = if info.scale_factor > 0.0 {
        info.scale_factor
    } else {
        1.0
    };
    let [outer_width, outer_height] = info.outer_size;
    let fraction = info.monitor.as_ref().map(|monitor| {
        let [monitor_width, monitor_height] = monitor.size;
        [
            f64::from(outer_width) / f64::from(monitor_width.max(1)),
            f64::from(outer_height) / f64::from(monitor_height.max(1)),
        ]
    });
    let mut value = json!({
        "state": info.state.wire_name(),
        "focused": info.focused,
        "scale_factor": info.scale_factor,
        "outer_position": info.outer_position,
        "outer_size": info.outer_size,
        "inner_size": info.inner_size,
        "monitor": info.monitor.as_ref().map(monitor_block),
        "derived": {
            "inner_size_logical": [
                f64::from(info.inner_size[0]) / scale,
                f64::from(info.inner_size[1]) / scale,
            ],
            // How much of the desktop this window is, per axis and by area —
            // the question a size alone cannot answer, since it depends on the
            // monitor the window is on.
            "monitor_fraction": fraction,
            "monitor_area_fraction": fraction.map(|[x, y]| x * y),
        },
    });
    if let Some(monitors) = monitors {
        value
            .as_object_mut()
            .expect("a window block is an object")
            .insert(
                "monitors".into(),
                Value::Array(monitors.iter().map(monitor_block).collect()),
            );
    }
    value
}

fn monitor_block(monitor: &MonitorInfo) -> Value {
    json!({
        "name": monitor.name,
        "position": monitor.position,
        "size": monitor.size,
        "scale_factor": monitor.scale_factor,
    })
}
