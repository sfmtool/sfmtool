// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Where MCP meets the frame: the three phases `App::run_ui_and_paint` calls.
//!
//! [`App::drain_mcp`] runs first in the frame, before the uploads, so a
//! command's effect is in the very frame the agent's request woke — including
//! a `set_window_layout`, which is applied before egui reads the window size
//! for that frame's layout. [`App::encode_screenshot_copy`] runs in the middle,
//! after the egui pass and before the present, because a surface texture is
//! only readable while it is still the frame's. [`App::resolve_mcp_deferred`]
//! runs last, after the present, and answers the one tool whose reply is a
//! picture of that frame.
//!
//! These are `impl App` and live here rather than in [`crate::app`] so the
//! whole surface — including the parts that have to know about `wgpu` and
//! `winit` — stays in one feature-gated module.

use std::sync::Arc;

use image::{ImageEncoder, ImageFormat};
use winit::window::Window;

use super::{
    apply_as_agent, panel_crop, Deferred, Outcome, Reply, Request, ScreenshotSource, ToolError,
    ToolOutput,
};
use crate::App;

/// Why a `screenshot` of the window is refused where the platform will not let
/// the swapchain be copied from.
///
/// The alternative — rendering every frame through an intermediate texture and
/// blitting it to the surface — is a per-frame cost for a case none of the three
/// supported backends has shown, so the refusal says what is true and points at
/// the one picture that is still available.
pub(super) const UNREADABLE_SURFACE: &str =
    "This platform's window surface cannot be read back, so there is nothing to photograph. \
     screenshot { \"panel_name\": \"viewer_3d\", \"hud\": false } reads the 3D render target \
     instead, which does not go through the surface.";

/// One row of a `copy_texture_to_buffer` destination must start on a 256-byte
/// boundary, so a capture buffer is padded per row and unpadded on the way out.
const COPY_ALIGNMENT: u32 = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;

/// Four bytes per pixel, for every surface format this reads back.
const BYTES_PER_PIXEL: u32 = 4;

/// A copy of this frame's surface texture, encoded before the present and
/// mapped after it.
///
/// Carries the geometry with the buffer because the surface can be reconfigured
/// between the copy and the map, and what was photographed is the size it was
/// then.
pub(crate) struct SurfaceCopy {
    staging: wgpu::Buffer,
    width: u32,
    height: u32,
    padded_bytes_per_row: u32,
    format: wgpu::TextureFormat,
}

/// Tightly packed RGBA8, top row first: what both readback paths produce and
/// what the PNG encoder takes.
struct Rgba {
    width: u32,
    height: u32,
    pixels: Vec<u8>,
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

    /// Encode a copy of the surface texture into this frame's encoder, when a
    /// screenshot of the window or of a panel is waiting for it.
    ///
    /// Called between the egui pass and `present`, into the same encoder, for
    /// the reason the copy exists at all: the surface texture is the frame the
    /// human is about to see, and it is not readable once it has been handed
    /// back to the presentation engine. `None` in every frame that has no such
    /// screenshot pending, which is nearly all of them — the cost of this
    /// feature is one copy per screenshot and nothing in any other frame.
    pub(crate) fn encode_screenshot_copy(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        surface: &wgpu::Texture,
    ) -> Option<SurfaceCopy> {
        if !self.surface_readable || !self.mcp_deferred.iter().any(reads_the_surface) {
            return None;
        }
        let (width, height) = (surface.width(), surface.height());
        if width == 0 || height == 0 {
            return None;
        }
        let padded_bytes_per_row =
            (width * BYTES_PER_PIXEL).div_ceil(COPY_ALIGNMENT) * COPY_ALIGNMENT;
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("window screenshot staging"),
            size: u64::from(padded_bytes_per_row) * u64::from(height),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: surface,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bytes_per_row),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        Some(SurfaceCopy {
            staging,
            width,
            height,
            padded_bytes_per_row,
            format: surface.format(),
        })
    }

    /// Answer the tool calls that needed this frame to have been rendered.
    ///
    /// `surface` is what [`App::encode_screenshot_copy`] enqueued, mapped once
    /// however many screenshots of this frame are waiting: they are all
    /// pictures of the same pixels, differing only in where they are cropped.
    pub(crate) fn resolve_mcp_deferred(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        surface: Option<SurfaceCopy>,
    ) {
        if self.mcp_deferred.is_empty() {
            return;
        }
        // Mapped once, and only if something asked for it. The result is shared
        // by reference below, refusal included, so a platform that cannot be
        // read back says so once per frame rather than once per copy.
        let window_image = surface.map(|copy| copy.read(device));
        // Physical pixels per logical point as *this frame* used them, which is
        // the window's scale factor composed with egui's zoom — the same number
        // the frame handed the renderer, and the only one the dock's rectangles
        // are in step with.
        let pixels_per_point = self.egui_ctx.pixels_per_point();
        for (deferred, reply) in std::mem::take(&mut self.mcp_deferred) {
            let Deferred::Screenshot {
                source,
                max_dimension,
                caption,
            } = deferred;
            let image = match source {
                ScreenshotSource::ViewportRender => self.viewport_render(device, queue),
                ScreenshotSource::Window | ScreenshotSource::Panel(_) => {
                    match window_image.as_ref() {
                        None => Err(ToolError::new(UNREADABLE_SURFACE)),
                        Some(Err(error)) => Err(error.clone()),
                        Some(Ok(frame)) => self.crop(frame, source, pixels_per_point),
                    }
                }
            };
            let _ = reply.send(image.and_then(|image| encode(image, max_dimension, caption)));
        }
    }

    /// The 3D viewport's own render target — `viewer_3d` with `hud: false`.
    ///
    /// A separate copy of a separate texture at a separate point in the frame,
    /// which is why it stays its own path: it reads the `edl output` texture
    /// the viewport rendered into, so the picture carries nothing egui painted
    /// over it.
    fn viewport_render(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<Rgba, ToolError> {
        let capture = self
            .scene_renderer
            .capture_edl_output(device, queue)
            .ok_or_else(|| {
                ToolError::new(
                    "The 3D viewport has not rendered yet, so there is nothing to photograph.",
                )
            })?;
        Ok(Rgba {
            width: capture.width,
            height: capture.height,
            pixels: capture.rgba,
        })
    }

    /// The whole frame, or the part of it one panel's body occupies.
    ///
    /// The rectangle is read *here* rather than when the command was applied: a
    /// panel a `show_panel` earlier in the same batch opened has no rectangle
    /// until this frame's egui pass has laid the dock out.
    fn crop(
        &self,
        frame: &Rgba,
        source: ScreenshotSource,
        pixels_per_point: f32,
    ) -> Result<Rgba, ToolError> {
        let ScreenshotSource::Panel(panel) = source else {
            return Ok(Rgba {
                width: frame.width,
                height: frame.height,
                pixels: frame.pixels.clone(),
            });
        };
        let [x, y, width, height] = panel_crop(
            &self.state.dock,
            panel,
            pixels_per_point,
            [frame.width, frame.height],
        )
        .ok_or_else(|| {
            ToolError::new(format!(
                "The {} panel has not been laid out in the window, so there is no rectangle to \
                 crop to. It may have been closed since the call was made.",
                panel.title()
            ))
        })?;
        let mut pixels = Vec::with_capacity((width * height * BYTES_PER_PIXEL) as usize);
        for row in y..y + height {
            let start = ((row * frame.width + x) * BYTES_PER_PIXEL) as usize;
            let end = start + (width * BYTES_PER_PIXEL) as usize;
            pixels.extend_from_slice(&frame.pixels[start..end]);
        }
        Ok(Rgba {
            width,
            height,
            pixels,
        })
    }
}

/// Whether a deferred screenshot needs the presented surface.
fn reads_the_surface((deferred, _): &(Deferred, tokio::sync::oneshot::Sender<Reply>)) -> bool {
    let Deferred::Screenshot { source, .. } = deferred;
    !matches!(source, ScreenshotSource::ViewportRender)
}

impl SurfaceCopy {
    /// Map the copy and unpack it into tightly packed RGBA8.
    ///
    /// The unpad and the swizzle are one pass, because they are one walk over
    /// the same bytes: each source row is `padded_bytes_per_row` long and only
    /// its first `width * 4` bytes are pixels, and a BGRA surface — which is
    /// what DX12 hands out on Windows — has to have its channels swapped, since
    /// `image` has no BGRA8 colour type to encode from.
    fn read(&self, device: &wgpu::Device) -> Result<Rgba, ToolError> {
        let swizzle = match self.format {
            wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb => true,
            wgpu::TextureFormat::Rgba8Unorm | wgpu::TextureFormat::Rgba8UnormSrgb => false,
            other => {
                return Err(ToolError::new(format!(
                    "The window surface is {other:?}, which this viewer does not know how to turn \
                     into a PNG. screenshot {{ \"panel_name\": \"viewer_3d\", \"hud\": false }} \
                     reads the 3D render target instead."
                )))
            }
        };
        let slice = self.staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        if !rx.recv().map(|r| r.is_ok()).unwrap_or(false) {
            return Err(ToolError::new(
                "The window could not be read back off the GPU this frame.",
            ));
        }
        let data = slice
            .get_mapped_range()
            .map_err(|_| ToolError::new("The window screenshot buffer could not be read."))?;
        let pixels = unpad(
            &data,
            self.width,
            self.height,
            self.padded_bytes_per_row,
            swizzle,
        );
        drop(data);
        self.staging.unmap();
        Ok(Rgba {
            width: self.width,
            height: self.height,
            pixels,
        })
    }
}

/// Strip the row padding a `copy_texture_to_buffer` destination carries, and
/// swap B and R on the way out where the surface was BGRA.
///
/// Split out from the map so the arithmetic is testable over a synthetic
/// buffer, which is the only way to test it without a window.
fn unpad(
    data: &[u8],
    width: u32,
    height: u32,
    padded_bytes_per_row: u32,
    swizzle: bool,
) -> Vec<u8> {
    let unpadded = (width * BYTES_PER_PIXEL) as usize;
    let mut out = Vec::with_capacity(unpadded * height as usize);
    for row in 0..height {
        let start = (row * padded_bytes_per_row) as usize;
        out.extend_from_slice(&data[start..start + unpadded]);
    }
    if swizzle {
        for pixel in out.chunks_exact_mut(BYTES_PER_PIXEL as usize) {
            pixel.swap(0, 2);
        }
    }
    out
}

/// Shrink if asked to, encode as a PNG, and wrap it with its caption.
///
/// Shared by both readback paths: what a picture is a picture *of* differs, and
/// nothing after that does.
fn encode(picture: Rgba, max_dimension: Option<u32>, caption: String) -> Reply {
    let (source_width, source_height) = (picture.width, picture.height);
    let mut rgba = image::RgbaImage::from_raw(source_width, source_height, picture.pixels)
        .ok_or_else(|| ToolError::new("The captured image was the wrong size."))?;
    if let Some(limit) = max_dimension {
        let longest = source_width.max(source_height);
        if longest > limit {
            // Lanczos3 rather than a nearest or triangle filter: what a
            // downscaled point cloud is asked to answer is "is this noisy",
            // and a cheap filter's aliasing invents exactly that.
            let scale = f64::from(limit) / f64::from(longest);
            let width = ((f64::from(source_width) * scale).round() as u32).max(1);
            let height = ((f64::from(source_height) * scale).round() as u32).max(1);
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A padded buffer of `width × height` pixels whose bytes say where they
    /// came from, so an unpad that reads the wrong row or the wrong column is
    /// visible in the output rather than merely wrong in size.
    fn padded(width: u32, height: u32, padded_bytes_per_row: u32) -> Vec<u8> {
        let mut data = vec![0xEE; (padded_bytes_per_row * height) as usize];
        for row in 0..height {
            for column in 0..width {
                let at = (row * padded_bytes_per_row + column * BYTES_PER_PIXEL) as usize;
                data[at] = 10 + row as u8; // B, or R
                data[at + 1] = 20 + column as u8; // G
                data[at + 2] = 30 + row as u8; // R, or B
                data[at + 3] = 255;
            }
        }
        data
    }

    #[test]
    fn unpadding_drops_the_row_padding_and_keeps_the_pixels() {
        let (width, height, stride) = (3, 2, 256);
        let out = unpad(&padded(width, height, stride), width, height, stride, false);
        assert_eq!(out.len() as u32, width * height * BYTES_PER_PIXEL);
        // Second row, third pixel: row 1, column 2, unswizzled.
        let at = ((width + 2) * BYTES_PER_PIXEL) as usize;
        assert_eq!(&out[at..at + 4], &[11, 22, 31, 255]);
    }

    #[test]
    fn a_bgra_surface_is_swizzled_to_rgba() {
        let (width, height, stride) = (3, 2, 256);
        let out = unpad(&padded(width, height, stride), width, height, stride, true);
        let at = ((width + 2) * BYTES_PER_PIXEL) as usize;
        // The same pixel, B and R exchanged and G and A left alone.
        assert_eq!(&out[at..at + 4], &[31, 22, 11, 255]);
    }
}
