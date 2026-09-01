// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Reading the finished 3D viewport back off the GPU, for the MCP `screenshot`
//! tool.
//!
//! What is captured is the **EDL output texture** — the offscreen image the
//! viewport panel displays — and not the window. The window is a surface
//! texture: copying that back would mean configuring the surface with
//! `COPY_SRC` and compositing egui's own pass over it, which is a much larger
//! change for a picture whose interesting part is the 3D view. See
//! `specs/gui/mcp-server.md`.
//!
//! The texture is `Rgba8UnormSrgb`, so its bytes are already display-ready:
//! what comes back here is exactly what the human is looking at, with no
//! colour conversion in between.

use super::SceneRenderer;

/// One row of a `copy_texture_to_buffer` destination must start on a 256-byte
/// boundary, so a capture buffer is padded per row and unpadded on the way out.
const COPY_ALIGNMENT: u32 = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;

/// Four bytes per pixel, RGBA8.
const BYTES_PER_PIXEL: u32 = 4;

/// The pixels of the 3D viewport as last rendered.
pub(crate) struct Capture {
    pub(crate) width: u32,
    pub(crate) height: u32,
    /// Tightly packed RGBA8, `width * height * 4` bytes, top row first.
    pub(crate) rgba: Vec<u8>,
}

impl SceneRenderer {
    /// Copy the finished viewport image back to the CPU.
    ///
    /// `None` before the viewport has been sized and rendered once — there is
    /// no texture to read, and an all-black image would be a worse answer than
    /// saying so.
    ///
    /// Submits its own encoder and blocks on the map: this runs in the frame's
    /// readback phase, after the scene pass has already been submitted and
    /// presented, alongside the pick readback that blocks the same way.
    pub(crate) fn capture_edl_output(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Option<Capture> {
        let texture = self.edl_output_texture.as_ref()?;
        let (width, height) = self.current_size;
        if width == 0 || height == 0 {
            return None;
        }

        let unpadded_bytes_per_row = width * BYTES_PER_PIXEL;
        let padded_bytes_per_row = unpadded_bytes_per_row.div_ceil(COPY_ALIGNMENT) * COPY_ALIGNMENT;
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("screenshot staging"),
            size: u64::from(padded_bytes_per_row) * u64::from(height),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("screenshot encoder"),
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture,
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
        queue.submit(std::iter::once(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        if !rx.recv().map(|r| r.is_ok()).unwrap_or(false) {
            log::warn!("MCP: the screenshot staging buffer could not be mapped");
            return None;
        }
        let data = slice.get_mapped_range().ok()?;

        // Unpad: each source row is `padded_bytes_per_row` long and only its
        // first `unpadded_bytes_per_row` bytes are pixels.
        let mut rgba = Vec::with_capacity((unpadded_bytes_per_row * height) as usize);
        for row in 0..height {
            let start = (row * padded_bytes_per_row) as usize;
            let end = start + unpadded_bytes_per_row as usize;
            rgba.extend_from_slice(&data[start..end]);
        }
        drop(data);
        staging.unmap();

        Some(Capture {
            width,
            height,
            rgba,
        })
    }
}
