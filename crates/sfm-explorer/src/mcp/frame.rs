// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Where MCP meets the frame: the two phases `App::run_ui_and_paint` calls,
//! and the real window behind [`super::window::WindowHost`].
//!
//! [`App::drain_mcp`] runs first in the frame, before the uploads, so a
//! command's effect is in the very frame the agent's request woke — including
//! a `set_window`, which is applied before egui reads the window size for that
//! frame's layout. [`App::resolve_mcp_deferred`] runs last, after the present,
//! and answers the one tool whose reply is a picture of that frame.
//!
//! These are `impl App` and live here rather than in [`crate::app`] so the
//! whole surface — including the parts that have to know about `wgpu` and
//! `winit` — stays in one feature-gated module.

use std::sync::Arc;

use image::{ImageEncoder, ImageFormat};
use winit::window::{Fullscreen, Window};

use super::window::{MonitorInfo, WindowChange, WindowHost, WindowInfo, WindowState};
use super::{apply_as_agent, Deferred, Outcome, Reply, Request, ToolError, ToolOutput};
use crate::App;

/// The real window behind the host trait.
///
/// Implemented on the `Arc<Window>` the frame already holds, so the drain has
/// nothing to construct and no lifetime to thread: the whole of what MCP needs
/// from `winit` is these two methods.
impl WindowHost for Arc<Window> {
    fn apply(&mut self, change: &WindowChange) -> Result<(), ToolError> {
        // One fixed order — state, position, size, focus — so a call carrying
        // several pieces reads as one sentence: "make it normal, put it here,
        // this big, and in front."
        if let Some(state) = change.state {
            match state {
                // All three flags off, and minimized first: restoring a
                // minimized window can bring a maximized one back, and the
                // agent asked for normal rather than for whatever was under it.
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
                // Borderless on the current monitor. A video mode is a
                // different feature, and the one thing an agent wants here is
                // the window filling the screen.
                WindowState::Fullscreen => {
                    self.set_minimized(false);
                    self.set_fullscreen(Some(Fullscreen::Borderless(None)));
                }
            }
        }
        if let Some([x, y]) = change.outer_position {
            // A platform that cannot say where a window is cannot be told
            // where to put one either, and `set_outer_position` reports
            // nothing — so the readable half of the pair is what answers for
            // both.
            if self.outer_position().is_err() {
                return Err(ToolError::new(
                    "This platform does not let an application position its own window.",
                ));
            }
            self.set_outer_position(winit::dpi::PhysicalPosition::new(x, y));
        }
        if let Some([width, height]) = change.inner_size {
            // The return is the size the platform settled on where it settles
            // synchronously, and `None` where a resize arrives later as an
            // event. Either way the reply is read back from the window rather
            // than from here.
            let _ = self.request_inner_size(winit::dpi::PhysicalSize::new(width, height));
        }
        if change.focus {
            self.focus_window();
        }
        Ok(())
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
        // The current monitor first, and not repeated: an agent reading this
        // list is choosing where to put the window, and "the one it is on"
        // is the entry it compares the rest against.
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

impl App {
    /// Apply every tool call that has arrived since the last frame.
    ///
    /// Drained into a vector first, then applied: `apply` needs `&mut` on two
    /// `App` fields, and the receiver is a third.
    ///
    /// This is the only point in the process where MCP touches application
    /// state. Everything a reply reports was read here, at one instant, with
    /// exclusive access — which is why a `get_scene` can never straddle a load.
    pub(crate) fn drain_mcp(&mut self, window: &Arc<Window>) {
        if self.mcp_rx.is_none() {
            return;
        }
        // The window snapshot every read answers from, taken before the batch
        // is applied — so `get_scene`'s window block and a `screenshot`'s
        // minimized check see this frame's window rather than the one before
        // it. `set_window` refreshes it again after a change of its own.
        self.state.window = window.observe().map(|(info, _)| info);
        let rx = self.mcp_rx.as_mut().expect("just checked");
        let mut requests: Vec<Request> = Vec::new();
        while let Ok(request) = rx.try_recv() {
            requests.push(request);
        }
        if requests.is_empty() {
            return;
        }
        // Split so the application phase can be a plain function over
        // `(&mut AppState, &mut Viewer3D)` — which is what puts the whole of
        // it, the Action Log's actor switch included, under headless test.
        let mut commands = Vec::with_capacity(requests.len());
        let mut replies = Vec::with_capacity(requests.len());
        for Request { command, reply } in requests {
            commands.push(command);
            replies.push(reply);
        }
        let mut host = window.clone();
        let outcomes = apply_as_agent(&mut self.state, &mut self.viewer_3d, &mut host, commands);
        for (outcome, reply) in outcomes.into_iter().zip(replies) {
            match outcome {
                // A dropped receiver means the client hung up mid-call, which
                // is normal and not worth a log line.
                Outcome::Done(answer) => {
                    let _ = reply.send(answer);
                }
                Outcome::Deferred(deferred) => self.mcp_deferred.push((deferred, reply)),
            }
            if let Some(status) = self.state.mcp.as_mut() {
                status.requests += 1;
            }
        }
    }

    /// Answer the tool calls that needed this frame to have been rendered.
    pub(crate) fn resolve_mcp_deferred(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        for (deferred, reply) in std::mem::take(&mut self.mcp_deferred) {
            let Deferred::Screenshot {
                max_dimension,
                caption,
            } = deferred;
            let _ = reply.send(self.screenshot(device, queue, max_dimension, caption));
        }
    }

    /// Read the 3D viewport back and encode it as a PNG.
    fn screenshot(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        max_dimension: Option<u32>,
        caption: String,
    ) -> Reply {
        let capture = self
            .scene_renderer
            .capture_edl_output(device, queue)
            .ok_or_else(|| {
                ToolError::new(
                    "The 3D viewport has not rendered yet, so there is nothing to photograph.",
                )
            })?;

        let mut rgba = image::RgbaImage::from_raw(capture.width, capture.height, capture.rgba)
            .ok_or_else(|| ToolError::new("The captured viewport image was the wrong size."))?;
        if let Some(limit) = max_dimension {
            let longest = capture.width.max(capture.height);
            if longest > limit {
                // Lanczos3 rather than a nearest or triangle filter: what a
                // downscaled point cloud is asked to answer is "is this noisy",
                // and a cheap filter's aliasing invents exactly that.
                let scale = f64::from(limit) / f64::from(longest);
                let width = ((f64::from(capture.width) * scale).round() as u32).max(1);
                let height = ((f64::from(capture.height) * scale).round() as u32).max(1);
                rgba = image::imageops::resize(
                    &rgba,
                    width,
                    height,
                    image::imageops::FilterType::Lanczos3,
                );
            }
        }

        let (width, height) = (rgba.width(), rgba.height());
        let mut bytes = Vec::new();
        image::codecs::png::PngEncoder::new(&mut std::io::Cursor::new(&mut bytes))
            .write_image(
                rgba.as_raw(),
                width,
                height,
                image::ExtendedColorType::Rgba8,
            )
            .map_err(|e| ToolError::new(format!("The screenshot could not be encoded: {e}")))?;
        debug_assert_eq!(image::guess_format(&bytes).ok(), Some(ImageFormat::Png));

        Ok(ToolOutput::Png {
            bytes,
            width,
            height,
            caption,
        })
    }
}
