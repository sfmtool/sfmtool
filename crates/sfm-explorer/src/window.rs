// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The window's placement: what it is, how a change to it is applied, and how
//! that change is spelled in the layout file.
//!
//! See `specs/gui/panel-layout.md` § "The window layout file". Everything the
//! window portion of a layout needs in every build lives here: [`WindowInfo`],
//! the snapshot [`AppState::window`](crate::state::AppState) holds and every
//! reply renders; [`WindowChange`], the `window` section of a document;
//! [`NormalRect`], the rectangle the window comes back to; and
//! [`WindowHost`], the seam every `winit` call goes through.
//!
//! **Why a host trait rather than reaching for the window.** A window change
//! has to be appliable from `AppState` — the Panels menu loads a document, the
//! startup load applies one, an MCP tool sends one — and none of those has a
//! `winit::Window` in reach that a headless test could stand up. Five
//! primitives and one provided [`WindowHost::apply`] keep the *order* the pieces
//! are applied in written once, so the fake in the tests exercises the real
//! rule rather than a copy of it.

use serde_json::Value;

use crate::layout::{known_keys, LayoutError};

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

/// The window as last observed: the block `get_window_layout` renders, minus
/// the monitor list.
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
    /// where it is. Reported as `null` rather than as `[0, 0]`, and an
    /// `outer_position` fails there for the same reason.
    pub(crate) outer_position: Option<[i32; 2]>,
    pub(crate) outer_size: [u32; 2],
    pub(crate) inner_size: [u32; 2],
    pub(crate) monitor: Option<MonitorInfo>,
}

/// The window's rectangle when it is normal.
///
/// Remembered across maximize, fullscreen and minimize, since `winit` reports
/// only the *current* rectangle and a maximized window's current rectangle is
/// the monitor's. This is the rectangle a saved layout needs: "maximized on the
/// left monitor" is a normal rectangle on that monitor plus the word
/// `maximized`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct NormalRect {
    /// `None` on a platform that will not say where a window is.
    pub(crate) outer_position: Option<[i32; 2]>,
    pub(crate) inner_size: [u32; 2],
}

/// A monitor's rectangle on the desktop, in physical pixels — the two fields
/// of [`MonitorInfo`] a fit needs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MonitorRect {
    pub(crate) position: [i32; 2],
    pub(crate) size: [u32; 2],
}

impl MonitorRect {
    /// The monitor a [`MonitorInfo`] describes, without its name or scale
    /// factor.
    pub(crate) fn of(monitor: &MonitorInfo) -> Self {
        MonitorRect {
            position: monitor.position,
            size: monitor.size,
        }
    }

    /// Whether `position` + `size` lies entirely inside this monitor.
    fn contains(&self, position: [i32; 2], size: [u32; 2]) -> bool {
        let [x, y] = position;
        let right = i64::from(x) + i64::from(size[0]);
        let bottom = i64::from(y) + i64::from(size[1]);
        x >= self.position[0]
            && y >= self.position[1]
            && right <= i64::from(self.position[0]) + i64::from(self.size[0])
            && bottom <= i64::from(self.position[1]) + i64::from(self.size[1])
    }
}

/// The `window` section of a layout document, and the pieces one
/// `set_window_layout` window portion carries. `None` is "leave it".
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct WindowChange {
    pub(crate) state: Option<WindowState>,
    /// Physical pixels, desktop coordinates: the top-left corner of the
    /// window's *normal* rectangle.
    pub(crate) outer_position: Option<[i32; 2]>,
    /// Physical pixels: the drawable area of the window's *normal* rectangle.
    pub(crate) inner_size: Option<[u32; 2]>,
    /// The monitor the rectangle was measured on.
    ///
    /// Read by [`fit_to_monitor`] and never by a host:
    /// `AppState::apply_window_layout` resolves it into a plain rectangle
    /// before the change reaches [`WindowHost::apply`].
    pub(crate) monitor: Option<MonitorRect>,
    /// Whether to ask for the foreground. Not an `Option`: `focus: false` is
    /// refused, so the only thing this field can say is "yes". Never written to
    /// a file; accepted on read so the file and the tool argument have one
    /// parser.
    pub(crate) focus: bool,
}

/// The bound a position is accepted within, either way: what `winit` takes.
const POSITION_LIMIT: i64 = i32::MAX as i64;

impl WindowChange {
    /// Whether the section asked for anything at all.
    pub(crate) fn is_empty(&self) -> bool {
        *self == WindowChange::default()
    }

    /// Whether the section carries a rectangle, which is applied to a normal
    /// window ([`WindowHost::apply`]).
    pub(crate) fn has_geometry(&self) -> bool {
        self.outer_position.is_some() || self.inner_size.is_some()
    }

    /// The `window` section, with the document's rules and its path-carrying
    /// messages.
    pub(crate) fn from_json(value: &Value, path: &str) -> Result<Self, LayoutError> {
        let Some(object) = value.as_object() else {
            return Err(LayoutError::at(path, "must be an object or null"));
        };
        known_keys(
            object,
            path,
            &["state", "outer_position", "inner_size", "monitor", "focus"],
        )?;

        let state = match object.get("state") {
            None | Some(Value::Null) => None,
            Some(Value::String(name)) => {
                Some(WindowState::from_wire_name(name).ok_or_else(|| {
                    LayoutError::at(
                        format!("{path}.state"),
                        format!(
                            "unknown window state \"{name}\"; the states are {}",
                            WindowState::all_wire_names()
                        ),
                    )
                })?)
            }
            Some(_) => {
                return Err(LayoutError::at(
                    format!("{path}.state"),
                    format!("must be one of {}", WindowState::all_wire_names()),
                ))
            }
        };
        let outer_position = int_pair(
            object.get("outer_position"),
            &format!("{path}.outer_position"),
            -POSITION_LIMIT,
            POSITION_LIMIT,
            "must be two whole numbers",
        )?
        .map(|[x, y]| [x as i32, y as i32]);
        let inner_size = int_pair(
            object.get("inner_size"),
            &format!("{path}.inner_size"),
            1,
            u32::MAX as i64,
            "must be two whole numbers greater than zero",
        )?
        .map(|[width, height]| [width as u32, height as u32]);
        let monitor = match object.get("monitor") {
            None | Some(Value::Null) => None,
            Some(value) => Some(monitor_rect_from_json(value, &format!("{path}.monitor"))?),
        };
        if monitor.is_some() && outer_position.is_none() && inner_size.is_none() {
            return Err(LayoutError::at(
                format!("{path}.monitor"),
                "has nothing to fit — send outer_position or inner_size with it",
            ));
        }
        let focus = match object.get("focus") {
            None | Some(Value::Null) => false,
            Some(Value::Bool(true)) => true,
            // As `set_view` reads `exit_camera_view: false`: a field that can
            // only ask for one thing has not asked for it.
            Some(_) => {
                return Err(LayoutError::at(
                    format!("{path}.focus"),
                    "can only ask for the foreground; omit it to leave focus alone",
                ))
            }
        };

        Ok(WindowChange {
            state,
            outer_position,
            inner_size,
            monitor,
            focus,
        })
    }

    /// The section as the file writes it: `state`, `outer_position` (omitted
    /// when there is none), `inner_size`, `monitor`; never `focus`, which is a
    /// request rather than a placement.
    pub(crate) fn write_json(&self, out: &mut String, depth: usize) {
        let inner = "  ".repeat(depth + 1);
        let outer = "  ".repeat(depth);
        let mut lines: Vec<String> = Vec::new();
        if let Some(state) = self.state {
            lines.push(format!("{inner}\"state\": \"{}\"", state.wire_name()));
        }
        if let Some([x, y]) = self.outer_position {
            lines.push(format!("{inner}\"outer_position\": [{x}, {y}]"));
        }
        if let Some([width, height]) = self.inner_size {
            lines.push(format!("{inner}\"inner_size\": [{width}, {height}]"));
        }
        if let Some(monitor) = self.monitor {
            let [x, y] = monitor.position;
            let [width, height] = monitor.size;
            lines.push(format!(
                "{inner}\"monitor\": {{ \"position\": [{x}, {y}], \"size\": [{width}, {height}] }}"
            ));
        }
        if lines.is_empty() {
            out.push_str("{}");
            return;
        }
        out.push_str("{\n");
        out.push_str(&lines.join(",\n"));
        out.push('\n');
        out.push_str(&outer);
        out.push('}');
    }

    /// The Action Log phrase: the pieces the call carried, in the order
    /// [`WindowHost::apply`] applies them, joined with `; `.
    ///
    /// A state the viewer put *back* after geometry is not in here: it was not
    /// asked for. The numbers are the call's rather than the read-back's, which
    /// is the rule the Action Log already keeps — it records what was asked.
    /// `fitted` is the saved monitor when [`fit_to_monitor`] changed the
    /// rectangle, since those numbers were the viewer's decision rather than
    /// the caller's and the person watching should be told so.
    ///
    /// Only the MCP surface words a window row: the menu and the startup load
    /// say which file they loaded, the file being the action there.
    #[cfg_attr(not(feature = "mcp"), allow(dead_code))]
    pub(crate) fn log_text(&self, fitted: Option<MonitorRect>) -> String {
        let mut pieces: Vec<String> = Vec::new();
        if let Some([x, y]) = self.outer_position {
            pieces.push(format!("Moved window to ({x}, {y})"));
        }
        if let Some([width, height]) = self.inner_size {
            pieces.push(format!("Resized window to {width}×{height}"));
        }
        // Hung off the last of the geometry pieces, which is the one whose
        // numbers the fit produced.
        if let (Some(monitor), Some(last)) = (fitted, pieces.last_mut()) {
            last.push_str(&format!(
                ", fitted from a {}×{} monitor",
                monitor.size[0], monitor.size[1]
            ));
        }
        if let Some(state) = self.state {
            pieces.push(state.log_text().to_string());
        }
        if self.focus {
            pieces.push("Focused window".to_string());
        }
        pieces
            .into_iter()
            .enumerate()
            .map(|(index, piece)| if index == 0 { piece } else { lower(&piece) })
            .collect::<Vec<_>>()
            .join("; ")
    }
}

/// Fit a saved rectangle onto the desktop it is being loaded at.
///
/// See `specs/gui/panel-layout.md` § "Fitting a rectangle to the desktop". A
/// layout saved at one desk and loaded at another — a laptop undocked from its
/// 4K monitor, a file an agent wrote on another machine — carries a rectangle
/// that may be nowhere the human can see; `monitor` says what share of a
/// monitor the window occupied, which is what makes it recoverable.
///
/// Returns the change with its rectangle fitted and `monitor` cleared, and the
/// saved monitor when a fit actually happened. A pure function over the section
/// and the monitor list, called from `AppState::apply_window_layout` before the
/// change reaches a host — so this is under headless test and
/// [`WindowHost::apply`] never sees a `monitor`.
pub(crate) fn fit_to_monitor(
    change: &WindowChange,
    monitors: &[MonitorInfo],
    target: Option<&MonitorInfo>,
) -> (WindowChange, Option<MonitorRect>) {
    let mut fitted = change.clone();
    fitted.monitor = None;
    let (Some(saved), true) = (change.monitor, change.has_geometry()) else {
        return (fitted, None);
    };
    // The desktop the file was written at, snapped and straddling windows
    // included: the rectangle is where it was put.
    if monitors
        .iter()
        .any(|monitor| MonitorRect::of(monitor) == saved)
    {
        return (fitted, None);
    }
    // Or somewhere the human can see it anyway, which is all the fit was for.
    if let (Some(position), Some(size)) = (change.outer_position, change.inner_size) {
        if monitors
            .iter()
            .any(|monitor| MonitorRect::of(monitor).contains(position, size))
        {
            return (fitted, None);
        }
    }
    let Some(target) = target else {
        // No monitor to map onto. Applied as written, and left to the window
        // manager, exactly as a section with no `monitor` is.
        return (fitted, None);
    };
    let scale_x = f64::from(target.size[0]) / f64::from(saved.size[0].max(1));
    let scale_y = f64::from(target.size[1]) / f64::from(saved.size[1].max(1));
    fitted.outer_position = change.outer_position.map(|[x, y]| {
        [
            target.position[0]
                + ((f64::from(x) - f64::from(saved.position[0])) * scale_x).round() as i32,
            target.position[1]
                + ((f64::from(y) - f64::from(saved.position[1])) * scale_y).round() as i32,
        ]
    });
    fitted.inner_size = change.inner_size.map(|[width, height]| {
        [
            ((f64::from(width) * scale_x).round() as u32).max(1),
            ((f64::from(height) * scale_y).round() as u32).max(1),
        ]
    });
    (fitted, Some(saved))
}

/// One `monitor` object: a position and a size, both required.
fn monitor_rect_from_json(value: &Value, path: &str) -> Result<MonitorRect, LayoutError> {
    const SHAPE: &str =
        "must be an object with \"position\" (two whole numbers) and \"size\" (two whole numbers \
         greater than zero)";
    let Some(object) = value.as_object() else {
        return Err(LayoutError::at(path, SHAPE));
    };
    known_keys(object, path, &["position", "size"])?;
    let position = int_pair(
        object.get("position"),
        path,
        -POSITION_LIMIT,
        POSITION_LIMIT,
        SHAPE,
    )?
    .ok_or_else(|| LayoutError::at(path, SHAPE))?;
    let size = int_pair(object.get("size"), path, 1, u32::MAX as i64, SHAPE)?
        .ok_or_else(|| LayoutError::at(path, SHAPE))?;
    Ok(MonitorRect {
        position: [position[0] as i32, position[1] as i32],
        size: [size[0] as u32, size[1] as u32],
    })
}

/// A pair of whole numbers in `[min, max]`, or `None` where the key is absent.
fn int_pair(
    value: Option<&Value>,
    path: &str,
    min: i64,
    max: i64,
    message: &str,
) -> Result<Option<[i64; 2]>, LayoutError> {
    let value = match value {
        None | Some(Value::Null) => return Ok(None),
        Some(value) => value,
    };
    let Some(array) = value.as_array() else {
        return Err(LayoutError::at(path, message));
    };
    if array.len() != 2 {
        return Err(LayoutError::at(path, message));
    }
    let mut out = [0i64; 2];
    for (slot, element) in out.iter_mut().zip(array) {
        *slot = element
            .as_i64()
            .filter(|n| (min..=max).contains(n))
            .ok_or_else(|| LayoutError::at(path, message))?;
    }
    Ok(Some(out))
}

/// A phrase that is no longer first in its sentence.
fn lower(text: &str) -> String {
    let mut chars = text.chars();
    match chars.next() {
        Some(first) => first.to_lowercase().collect::<String>() + chars.as_str(),
        None => String::new(),
    }
}

/// A refusal from the window itself: something the platform cannot do.
///
/// Distinct from a validation failure, which the document's parser produces
/// before anything is applied. This one can only come from the host, mid-apply,
/// which is why applying a document stops at the piece the platform refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct WindowError(pub(crate) String);

impl std::fmt::Display for WindowError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for WindowError {}

/// What a window read or change says when there is no window behind it.
pub(crate) const NO_WINDOW: &str =
    "This viewer has no window to read or change — the window tools need a running viewer.";

/// What applying a [`WindowChange`] needs from the window, and all it needs.
///
/// Five primitives, and one provided method that holds the application order —
/// written once, so the fake in the tests exercises the real rule rather than a
/// copy of it.
pub(crate) trait WindowHost {
    /// Show the window as `state`. `Normal` is all three flags off, minimized
    /// first: restoring a minimized window can bring a maximized one back, and
    /// the caller asked for normal rather than for whatever was under it.
    fn set_state(&mut self, state: WindowState);

    /// Put the window's top-left corner here, in physical pixels.
    ///
    /// `Err` only where the platform does not let an application place its own
    /// window; a position the window manager overrules is not an error, and the
    /// read-back reports what happened.
    fn set_outer_position(&mut self, position: [i32; 2]) -> Result<(), WindowError>;

    /// Ask for this drawable size, in physical pixels. What the platform does
    /// with the request is not knowable in advance, which is why nothing here
    /// returns it: the read-back reports what the window became.
    fn set_inner_size(&mut self, size: [u32; 2]);

    /// Ask for the foreground. A platform may decline.
    fn focus(&mut self);

    /// The window as it is now, and every monitor, the current one first.
    fn observe(&self) -> Option<(WindowInfo, Vec<MonitorInfo>)>;

    /// Apply a `window` section: **geometry first, on a normal window; then the
    /// state; then focus.**
    ///
    /// The geometry goes on a normal window whatever the window was showing as,
    /// so a rectangle always sets the *normal* rectangle — which is the one a
    /// saved layout carries, and the one a maximized window comes back to. The
    /// state that follows is the one the section names, or, where it names none
    /// and geometry was applied, the state the window was in before, put back.
    /// So `{ "inner_size": … }` against a maximized window changes what it will
    /// restore to and leaves it maximized.
    ///
    /// The previous state is read through [`WindowHost::observe`] rather than
    /// guessed, and a host that observes nothing has no window to apply to.
    fn apply(&mut self, change: &WindowChange) -> Result<(), WindowError> {
        let Some((info, _)) = self.observe() else {
            return Err(WindowError(NO_WINDOW.to_string()));
        };
        let previous = info.state;
        if change.has_geometry() {
            self.set_state(WindowState::Normal);
            if let Some(position) = change.outer_position {
                // An `Err` stops here, with the size and the state untouched:
                // a platform refusal is a refusal of the call.
                self.set_outer_position(position)?;
            }
            if let Some(size) = change.inner_size {
                self.set_inner_size(size);
            }
        }
        match change.state {
            Some(state) => self.set_state(state),
            // Nothing to put back where the window was already normal, or where
            // the section carried no geometry to make it normal for.
            None if change.has_geometry() && previous != WindowState::Normal => {
                self.set_state(previous)
            }
            None => {}
        }
        if change.focus {
            self.focus();
        }
        Ok(())
    }
}

// ── The real window ──────────────────────────────────────────────────────

/// The window behind the host trait.
///
/// Implemented on the `Arc<Window>` the frame already holds, so the frame has
/// nothing to construct and no lifetime to thread: the whole of what a window
/// change needs from `winit` is these five methods.
impl WindowHost for std::sync::Arc<winit::window::Window> {
    fn set_state(&mut self, state: WindowState) {
        match state {
            WindowState::Normal => {
                self.set_minimized(false);
                self.set_fullscreen(None);
                self.set_maximized(false);
            }
            WindowState::Maximized => {
                self.set_minimized(false);
                self.set_fullscreen(None);
                self.set_maximized(true);
            }
            WindowState::Minimized => self.set_minimized(true),
            // Borderless on the current monitor. A video mode is a different
            // feature, and the one thing anyone wants here is the window
            // filling the screen.
            WindowState::Fullscreen => {
                self.set_minimized(false);
                self.set_fullscreen(Some(winit::window::Fullscreen::Borderless(None)));
            }
        }
    }

    fn set_outer_position(&mut self, [x, y]: [i32; 2]) -> Result<(), WindowError> {
        // A platform that cannot say where a window is cannot be told where to
        // put one either, and `set_outer_position` reports nothing — so the
        // readable half of the pair is what answers for both.
        if self.outer_position().is_err() {
            return Err(WindowError(
                "This platform does not let an application position its own window.".to_string(),
            ));
        }
        winit::window::Window::set_outer_position(self, winit::dpi::PhysicalPosition::new(x, y));
        Ok(())
    }

    fn set_inner_size(&mut self, [width, height]: [u32; 2]) {
        // The return is the size the platform settled on where it settles
        // synchronously, and `None` where a resize arrives later as an event.
        // Either way what anyone reads is the next observation.
        let _ = self.request_inner_size(winit::dpi::PhysicalSize::new(width, height));
    }

    fn focus(&mut self) {
        self.focus_window();
    }

    fn observe(&self) -> Option<(WindowInfo, Vec<MonitorInfo>)> {
        let current = self.current_monitor();
        let info = WindowInfo {
            state: WindowState::of(
                // A platform that cannot say reads as not minimized: the
                // question this answers is "can the human see the window", and
                // the honest default is that they can.
                self.is_minimized().unwrap_or(false),
                self.fullscreen().is_some(),
                self.is_maximized(),
            ),
            focused: self.has_focus(),
            scale_factor: self.scale_factor(),
            outer_position: self
                .outer_position()
                .ok()
                .map(|position| [position.x, position.y]),
            outer_size: size_of(self.outer_size()),
            inner_size: size_of(self.inner_size()),
            monitor: current.as_ref().map(monitor_info),
        };
        // The current monitor first, and not repeated: whoever reads this list
        // is choosing where to put the window, and "the one it is on" is the
        // entry they compare the rest against.
        let mut monitors: Vec<MonitorInfo> = current.iter().map(monitor_info).collect();
        monitors.extend(
            self.available_monitors()
                .filter(|monitor| Some(monitor) != current.as_ref())
                .map(|monitor| monitor_info(&monitor)),
        );
        Some((info, monitors))
    }
}

fn monitor_info(monitor: &winit::monitor::MonitorHandle) -> MonitorInfo {
    let position = monitor.position();
    MonitorInfo {
        name: monitor.name(),
        position: [position.x, position.y],
        size: size_of(monitor.size()),
        scale_factor: monitor.scale_factor(),
    }
}

fn size_of(size: winit::dpi::PhysicalSize<u32>) -> [u32; 2] {
    [size.width, size.height]
}
