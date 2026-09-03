// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Where MCP meets the frame: the two phases `App::run_ui_and_paint` calls.
//!
//! [`App::drain_mcp`] runs first in the frame, before the uploads, so a
//! command's effect is in the very frame the agent's request woke — including
//! a `set_window_layout`, which is applied before egui reads the window size
//! for that frame's layout. [`App::resolve_mcp_deferred`] runs last, after the
//! present, and answers the one tool whose reply is a picture of that frame.
//!
//! These are `impl App` and live here rather than in [`crate::app`] so the
//! whole surface — including the parts that have to know about `wgpu` and
//! `winit` — stays in one feature-gated module.

use std::sync::Arc;

use image::{ImageEncoder, ImageFormat};
use winit::window::Window;

use super::{apply_as_agent, Deferred, Outcome, Reply, Request, ToolError, ToolOutput};
use crate::App;

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
        // The window snapshot every read answers from was taken by
        // `run_ui_and_paint` just above, before this drain — every frame, not
        // only the ones the endpoint is live for. Applying a window layout
        // refreshes it again after its change, so a later call in this batch
        // sees the window the earlier one asked for.
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
