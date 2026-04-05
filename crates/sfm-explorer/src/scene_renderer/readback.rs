// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! GPU readback logic for 5×5 depth and pick buffer sampling.

use super::gpu_types::*;
use super::SceneRenderer;

/// Result of a 5×5 readback from both depth and pick buffers.
pub struct ReadbackResult {
    /// Nearest valid depth in the 5×5 region, or `None` if background.
    pub depth: Option<f32>,
    /// Picked entity as `(tag, index)`, or `None` if background.
    pub pick: Option<(u32, u32)>,
}

/// Read a 5x5 grid of f32 values from a 256-byte-aligned staging buffer.
/// Returns the first valid value using center → 3x3 → 5x5 search order.
fn search_5x5_f32(data: &[u8], valid: impl Fn(f32) -> bool) -> Option<f32> {
    let read = |row: usize, col: usize| -> f32 {
        let offset = row * 256 + col * 4;
        if offset + 4 <= data.len() {
            f32::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ])
        } else {
            0.0
        }
    };

    // Center first
    let center = read(2, 2);
    if valid(center) {
        return Some(center);
    }

    // 3x3 neighborhood — pick nearest (smallest valid value)
    let mut best = None;
    for row in 1..=3 {
        for col in 1..=3 {
            if row == 2 && col == 2 {
                continue;
            }
            let v = read(row, col);
            if valid(v) && best.is_none_or(|b| v < b) {
                best = Some(v);
            }
        }
    }
    if best.is_some() {
        return best;
    }

    // Full 5x5
    for row in 0..5 {
        for col in 0..5 {
            let v = read(row, col);
            if valid(v) && best.is_none_or(|b| v < b) {
                best = Some(v);
            }
        }
    }
    best
}

/// Read a 5x5 grid of u32 values from a 256-byte-aligned staging buffer.
/// Returns the first valid value using center → 3x3 → 5x5 search order.
fn search_5x5_u32(data: &[u8], valid: impl Fn(u32) -> bool) -> Option<u32> {
    let read = |row: usize, col: usize| -> u32 {
        let offset = row * 256 + col * 4;
        if offset + 4 <= data.len() {
            u32::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ])
        } else {
            0
        }
    };

    // Center first
    let center = read(2, 2);
    if valid(center) {
        return Some(center);
    }

    // 3x3 neighborhood — return first valid hit
    for row in 1..=3 {
        for col in 1..=3 {
            if row == 2 && col == 2 {
                continue;
            }
            let v = read(row, col);
            if valid(v) {
                return Some(v);
            }
        }
    }

    // Full 5x5
    for row in 0..5 {
        for col in 0..5 {
            let v = read(row, col);
            if valid(v) {
                return Some(v);
            }
        }
    }
    None
}

impl SceneRenderer {
    /// Enqueue copies of a 5×5 pixel region from both the linear depth and
    /// pick buffer textures to their staging buffers. Used for both hover
    /// (every frame) and click (on demand) — same buffers, same logic.
    pub fn copy_readback_region(&mut self, encoder: &mut wgpu::CommandEncoder, cx: u32, cy: u32) {
        let (w, h) = self.current_size;
        if w == 0 || h == 0 {
            return;
        }

        let x0 = cx.saturating_sub(2).min(w.saturating_sub(5));
        let y0 = cy.saturating_sub(2).min(h.saturating_sub(5));
        let copy_w = 5.min(w - x0);
        let copy_h = 5.min(h - y0);
        let copy_extent = wgpu::Extent3d {
            width: copy_w,
            height: copy_h,
            depth_or_array_layers: 1,
        };
        let layout = wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(256),
            rows_per_image: None,
        };
        let origin = wgpu::Origin3d { x: x0, y: y0, z: 0 };

        if let (Some(texture), Some(staging)) = (&self.linear_depth_texture, &self.depth_staging) {
            encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture,
                    mip_level: 0,
                    origin,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: staging,
                    layout,
                },
                copy_extent,
            );
        }

        if let (Some(texture), Some(staging)) = (&self.pick_texture, &self.pick_staging) {
            encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture,
                    mip_level: 0,
                    origin,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: staging,
                    layout,
                },
                copy_extent,
            );
        }

        self.readback_pending = true;
    }

    /// Map both staging buffers and read the 5×5 depth + pick results.
    /// Updates the cached hover result and returns the full result for click handling.
    pub fn read_readback_result(&mut self, device: &wgpu::Device) -> Option<ReadbackResult> {
        if !self.readback_pending {
            return None;
        }
        self.readback_pending = false;

        let depth = self.read_5x5_depth(device);
        let pick = self.read_5x5_pick(device);

        // Cache for hover overlay
        self.hover_depth = depth;
        self.hover_pick_id = match pick {
            Some((tag, index)) => tag | index,
            None => PICK_TAG_NONE,
        };

        Some(ReadbackResult { depth, pick })
    }

    /// Returns the most recently read-back hover depth value.
    pub fn hover_depth(&self) -> Option<f32> {
        self.hover_depth
    }

    /// Returns the most recently read-back hover pick ID (tag | index).
    pub fn hover_pick_id(&self) -> u32 {
        self.hover_pick_id
    }

    /// Read a 5×5 depth region from the staging buffer.
    fn read_5x5_depth(&self, device: &wgpu::Device) -> Option<f32> {
        let staging = self.depth_staging.as_ref()?;
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        if !rx.recv().map(|r| r.is_ok()).unwrap_or(false) {
            return None;
        }
        let data = slice.get_mapped_range();
        let result = search_5x5_f32(&data, |d| d > 0.0);
        drop(data);
        staging.unmap();
        result
    }

    /// Read a 5×5 pick region from the staging buffer.
    fn read_5x5_pick(&self, device: &wgpu::Device) -> Option<(u32, u32)> {
        let staging = self.pick_staging.as_ref()?;
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        if !rx.recv().map(|r| r.is_ok()).unwrap_or(false) {
            return None;
        }
        let data = slice.get_mapped_range();
        let result = search_5x5_u32(&data, |id| id & PICK_TAG_MASK != PICK_TAG_NONE);
        drop(data);
        staging.unmap();
        result.map(|id| (id & PICK_TAG_MASK, id & PICK_INDEX_MASK))
    }
}
