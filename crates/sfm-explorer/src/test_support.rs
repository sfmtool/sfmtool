// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Helpers shared by the headless UI tests.

use crate::window::{MonitorInfo, WindowChange, WindowError, WindowHost, WindowInfo, WindowState};

/// A [`WindowHost`] with no window behind it: it records the primitives it was
/// asked for, in order, and reports what it then is.
///
/// Lives here rather than in one test module because two need it — `mcp::tests`
/// for the tools, and `layout::tests` because applying a document and drawing
/// the Panels menu both take a host. Recording the *primitives* is the point:
/// the application order is [`WindowHost::apply`]'s provided body, so a test
/// asserting the sequence is asserting the real rule rather than a fake's
/// re-implementation of it.
///
/// The three flags are kept apart rather than collapsed, because a window can
/// be minimized *and* maximized at once and the collapse into one
/// [`WindowState`] is exactly what is under test.
#[derive(Debug, Clone)]
pub(crate) struct FakeWindow {
    pub(crate) minimized: bool,
    pub(crate) maximized: bool,
    pub(crate) fullscreen: bool,
    pub(crate) focused: bool,
    /// `None` stands in for a platform that will not say where a window is,
    /// and refuses a position for the same reason.
    pub(crate) position: Option<[i32; 2]>,
    pub(crate) inner_size: [u32; 2],
    /// A size it will not go below, standing in for whatever a real platform
    /// does to a request it will not honour: clamped, not refused, so that the
    /// read-back has something to differ from the request about.
    pub(crate) minimum: [u32; 2],
    /// Whether it lets an application take focus.
    pub(crate) focusable: bool,
    /// The monitors it reports, the current one first.
    pub(crate) monitors: Vec<MonitorInfo>,
    /// What it was asked, in the order it was asked.
    pub(crate) applied: Vec<String>,
}

impl Default for FakeWindow {
    fn default() -> Self {
        FakeWindow {
            minimized: false,
            maximized: false,
            fullscreen: false,
            focused: true,
            position: Some([120, 64]),
            inner_size: [1920, 1080],
            minimum: [1600, 1200],
            focusable: true,
            monitors: vec![
                MonitorInfo {
                    name: Some("DISPLAY1".to_string()),
                    position: [0, 0],
                    size: [3840, 2160],
                    scale_factor: 1.5,
                },
                MonitorInfo {
                    name: Some("DISPLAY2".to_string()),
                    position: [3840, 0],
                    size: [1920, 1080],
                    scale_factor: 1.0,
                },
            ],
            applied: Vec::new(),
        }
    }
}

impl FakeWindow {
    /// A fake showing one of the four states, everything else as default.
    pub(crate) fn in_state(state: WindowState) -> Self {
        FakeWindow {
            minimized: state == WindowState::Minimized,
            maximized: state == WindowState::Maximized,
            fullscreen: state == WindowState::Fullscreen,
            ..FakeWindow::default()
        }
    }

    pub(crate) fn info(&self) -> WindowInfo {
        WindowInfo {
            state: WindowState::of(self.minimized, self.fullscreen, self.maximized),
            focused: self.focused,
            scale_factor: 1.5,
            outer_position: self.position,
            outer_size: [self.inner_size[0] + 16, self.inner_size[1] + 39],
            inner_size: self.inner_size,
            monitor: self.monitors.first().cloned(),
        }
    }
}

impl WindowHost for FakeWindow {
    fn set_state(&mut self, state: WindowState) {
        self.applied.push(format!("state {}", state.wire_name()));
        match state {
            WindowState::Normal => {
                self.minimized = false;
                self.fullscreen = false;
                self.maximized = false;
            }
            WindowState::Maximized => {
                self.minimized = false;
                self.fullscreen = false;
                self.maximized = true;
            }
            WindowState::Minimized => self.minimized = true,
            WindowState::Fullscreen => {
                self.minimized = false;
                self.fullscreen = true;
            }
        }
    }

    fn set_outer_position(&mut self, [x, y]: [i32; 2]) -> Result<(), WindowError> {
        self.applied.push(format!("position {x},{y}"));
        if self.position.is_none() {
            return Err(WindowError(
                "This platform does not let an application position its own window.".to_string(),
            ));
        }
        self.position = Some([x, y]);
        Ok(())
    }

    fn set_inner_size(&mut self, [width, height]: [u32; 2]) {
        self.applied.push(format!("size {width}x{height}"));
        self.inner_size = [width.max(self.minimum[0]), height.max(self.minimum[1])];
    }

    fn focus(&mut self) {
        self.applied.push("focus".to_string());
        self.focused = self.focusable;
    }

    fn observe(&self) -> Option<(WindowInfo, Vec<MonitorInfo>)> {
        Some((self.info(), self.monitors.clone()))
    }
}

/// The host for the windowless case: there is no window here.
///
/// It refuses rather than pretending — a caller that forgot to pass a real host
/// fails loudly rather than silently exercising nothing. Every primitive is
/// unreachable, because [`WindowHost::apply`] stops at the observation.
pub(crate) struct NoWindow;

impl WindowHost for NoWindow {
    fn set_state(&mut self, _state: WindowState) {
        unreachable!("apply stops at the observation where there is no window")
    }

    fn set_outer_position(&mut self, _position: [i32; 2]) -> Result<(), WindowError> {
        unreachable!("apply stops at the observation where there is no window")
    }

    fn set_inner_size(&mut self, _size: [u32; 2]) {
        unreachable!("apply stops at the observation where there is no window")
    }

    fn focus(&mut self) {
        unreachable!("apply stops at the observation where there is no window")
    }

    fn observe(&self) -> Option<(WindowInfo, Vec<MonitorInfo>)> {
        None
    }
}

/// A change that carries nothing but geometry, for the fits and the orders.
#[allow(dead_code)] // Used by `layout::tests`; `mcp::tests` builds its own.
pub(crate) fn geometry(position: Option<[i32; 2]>, size: Option<[u32; 2]>) -> WindowChange {
    WindowChange {
        outer_position: position,
        inner_size: size,
        ..WindowChange::default()
    }
}

/// Run one egui frame headlessly and discard its output.
///
/// The wrapper exists for the discarding: since epaint 0.36 a
/// [`TexturesDelta`](egui::TexturesDelta) panics on drop if it still holds
/// unapplied deltas, on the grounds that a real integration losing them is a
/// bug ("texture has not been allocated yet" on a later partial update — the
/// same hazard `app.rs` documents around its `update_texture` loop). These
/// frames have no painter at all, so the deltas are dropped deliberately, which
/// is exactly the case epaint asks be spelled out with an explicit `clear`.
pub(crate) fn run_frame_headless(
    ctx: &egui::Context,
    input: egui::RawInput,
    run_ui: impl FnMut(&mut egui::Ui),
) {
    let mut output = ctx.run_ui(input, run_ui);
    output.textures_delta.clear();
}

/// Every string painted in one headless frame, in paint order.
///
/// A panel that elides its own text has to be asked what it actually drew, and
/// egui's frame output carries the galleys: laying out needs no GPU, so the
/// strings are real even with no painter behind them. Nested `Shape::Vec`s are
/// walked, since a widget's shapes arrive grouped.
pub(crate) fn painted_texts(
    ctx: &egui::Context,
    input: egui::RawInput,
    run_ui: impl FnMut(&mut egui::Ui),
) -> Vec<String> {
    let mut output = ctx.run_ui(input, run_ui);
    output.textures_delta.clear();
    let mut texts = Vec::new();
    for clipped in &output.shapes {
        collect_texts(&clipped.shape, &mut texts);
    }
    texts
}

fn collect_texts(shape: &egui::Shape, out: &mut Vec<String>) {
    match shape {
        egui::Shape::Text(text) => out.push(text.galley.text().to_owned()),
        egui::Shape::Vec(shapes) => {
            for shape in shapes {
                collect_texts(shape, out);
            }
        }
        _ => {}
    }
}
