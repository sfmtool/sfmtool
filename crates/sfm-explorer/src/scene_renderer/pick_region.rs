// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Reading a rectangle of the pick buffer, and finding the points in it that a
//! capsule covers: what the eraser brush reads where a click reads 5×5 pixels.
//!
//! See `specs/gui/camera-views.md` § "Region reads". The copy and the read run
//! through [`SceneRenderer`]; the rectangle, the decoded region and the capsule
//! test are plain values and a pure function, so they are tested without a GPU.

use std::collections::HashSet;

use super::picking::{PickTarget, PICK_TAG_MASK, PICK_TAG_POINT};
use super::SceneRenderer;
use crate::scene::{PointRef, ReconId};

/// A rectangle of the scene texture, in physical pixels from its top-left
/// corner: the columns `x .. x + width` and the rows `y .. y + height`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PixelRect {
    pub(crate) x: u32,
    pub(crate) y: u32,
    pub(crate) width: u32,
    pub(crate) height: u32,
}

impl PixelRect {
    /// The smallest rectangle holding every pixel whose centre lies within
    /// `radius` of the segment from `from` to `to`, which is every pixel
    /// [`points_in_capsule`] can select for the same capsule. Positions are in
    /// physical pixels of the scene texture, as fractional values: the pixel
    /// `(x, y)` covers `x .. x + 1` and has its centre at `x + 0.5`. Cut at the
    /// texture's top and left edges; [`SceneRenderer::copy_pick_region`] cuts it
    /// at the other two.
    pub(crate) fn around_capsule(from: [f32; 2], to: [f32; 2], radius: f32) -> Self {
        let radius = radius.max(0.0);
        // A centre `c` is at most `radius` from the segment only if it is within
        // the segment's bounding box grown by `radius`, so the pixel indexes
        // run from `ceil(min - radius - 0.5)` to `floor(max + radius - 0.5)`.
        let span = |a: f32, b: f32| -> (u32, u32) {
            let first = (a.min(b) - radius - 0.5).ceil().max(0.0);
            let last = (a.max(b) + radius - 0.5).floor();
            if last < first {
                (first as u32, 0)
            } else {
                (first as u32, (last - first) as u32 + 1)
            }
        };
        let (x, width) = span(from[0], to[0]);
        let (y, height) = span(from[1], to[1]);
        Self {
            x,
            y,
            width,
            height,
        }
    }

    /// How many pixels the rectangle holds.
    fn area(self) -> usize {
        self.width as usize * self.height as usize
    }

    /// This rectangle cut to a texture of `size`, or `None` when nothing of it
    /// is left.
    fn clamped(self, (texture_width, texture_height): (u32, u32)) -> Option<Self> {
        let x_end = self.x.saturating_add(self.width).min(texture_width);
        let y_end = self.y.saturating_add(self.height).min(texture_height);
        (self.x < x_end && self.y < y_end).then(|| Self {
            x: self.x,
            y: self.y,
            width: x_end - self.x,
            height: y_end - self.y,
        })
    }
}

/// A rectangle of the pick buffer, decoded: for each pixel, the point it shows,
/// or `None` for the background, a frustum or image quad, a non-interactive
/// node, and an id that no longer decodes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PickRegion {
    /// The rectangle the caller asked for, before it was cut to the texture.
    requested: PixelRect,
    rect: PixelRect,
    /// Row-major, `rect.width * rect.height` entries.
    points: Vec<Option<PointRef>>,
}

impl PickRegion {
    /// A region from its rectangle and its pixels, row-major, read for a
    /// request of that same rectangle. Panics if the pixel count is not the
    /// rectangle's area.
    pub(crate) fn new(rect: PixelRect, points: Vec<Option<PointRef>>) -> Self {
        assert_eq!(
            points.len(),
            rect.area(),
            "a pick region holds one entry per pixel of its rectangle"
        );
        Self {
            requested: rect,
            rect,
            points,
        }
    }

    /// The rectangle the request named, which is what a caller that asks on
    /// every frame matches an answer to its question by.
    pub(crate) fn requested(&self) -> PixelRect {
        self.requested
    }

    /// The rectangle this region was read from, which is the requested one cut
    /// to the texture. Read by the region tests; the brush matches an answer
    /// by [`Self::requested`] alone.
    #[cfg(test)]
    pub(crate) fn rect(&self) -> PixelRect {
        self.rect
    }

    /// The point the pixel `(x, y)` of the scene texture shows, or `None`
    /// when it shows none or lies outside the region.
    #[cfg(test)]
    pub(crate) fn point_at(&self, x: u32, y: u32) -> Option<PointRef> {
        let col = x.checked_sub(self.rect.x)?;
        let row = y.checked_sub(self.rect.y)?;
        if col >= self.rect.width || row >= self.rect.height {
            return None;
        }
        self.points[row as usize * self.rect.width as usize + col as usize]
    }
}

/// The indexes of `recon`'s points shown by a pixel of `region` whose centre
/// lies within `radius` of the segment from `from` to `to`.
///
/// The capsule is the area the brush's circle sweeps moving from `from` to
/// `to`; `from == to` is the circle itself. Positions and the radius are in
/// physical pixels of the scene texture, with the pixel `(x, y)` centred at
/// `(x + 0.5, y + 0.5)`. Pixels of other reconstructions are skipped, as are
/// capsule pixels outside the region. The indexes are the edited indexes the
/// pick buffer carries.
pub(crate) fn points_in_capsule(
    region: &PickRegion,
    recon: ReconId,
    from: [f32; 2],
    to: [f32; 2],
    radius: f32,
) -> HashSet<u32> {
    let mut found = HashSet::new();
    let radius_sq = radius * radius;
    let [ax, ay] = from;
    let (dx, dy) = (to[0] - ax, to[1] - ay);
    let length_sq = dx * dx + dy * dy;
    let rect = region.rect;
    for row in 0..rect.height {
        let cy = (rect.y + row) as f32 + 0.5;
        for col in 0..rect.width {
            let Some(point) = region.points[row as usize * rect.width as usize + col as usize]
            else {
                continue;
            };
            if point.recon != recon {
                continue;
            }
            let cx = (rect.x + col) as f32 + 0.5;
            // The nearest point of the segment to the centre: the projection,
            // held to the segment's ends.
            let t = if length_sq > 0.0 {
                (((cx - ax) * dx + (cy - ay) * dy) / length_sq).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let (ex, ey) = (cx - (ax + t * dx), cy - (ay + t * dy));
            if ex * ex + ey * ey <= radius_sq {
                found.insert(point.point);
            }
        }
    }
    found
}

/// The bytes one row of a region copy takes in the staging buffer: four per
/// pixel, padded to wgpu's row alignment.
fn padded_row_bytes(width: u32) -> u32 {
    (width * 4).div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT
}

impl SceneRenderer {
    /// Enqueue a copy of `rect` of the pick texture to the region staging
    /// buffer, for [`Self::read_pick_region`] to read after the frame is
    /// submitted. `rect` is cut to the texture first; a rectangle with nothing
    /// left enqueues nothing. The staging buffer grows to the largest region
    /// asked for and is kept.
    pub(crate) fn copy_pick_region(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        rect: PixelRect,
    ) {
        let Some(texture) = &self.pick_texture else {
            return;
        };
        let requested = rect;
        let Some(rect) = rect.clamped(self.current_size) else {
            return;
        };
        let row_bytes = padded_row_bytes(rect.width);
        let needed = u64::from(row_bytes) * u64::from(rect.height);
        if self
            .region_staging
            .as_ref()
            .is_none_or(|staging| staging.size() < needed)
        {
            self.region_staging = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("pick region staging"),
                size: needed,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }));
        }
        let staging = self.region_staging.as_ref().expect("just ensured");
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: rect.x,
                    y: rect.y,
                    z: 0,
                },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(row_bytes),
                    rows_per_image: None,
                },
            },
            wgpu::Extent3d {
                width: rect.width,
                height: rect.height,
                depth_or_array_layers: 1,
            },
        );
        self.region_pending = Some((requested, rect));
    }

    /// Whether a region copy was enqueued this frame and not yet read.
    pub(crate) fn pick_region_pending(&self) -> bool {
        self.region_pending.is_some()
    }

    /// Map the region staging buffer and decode the region the last
    /// [`Self::copy_pick_region`] copied, or `None` when none is waiting. Each
    /// pixel tagged as a point decodes through the per-reconstruction pick
    /// ranges as a click's pick does; every other pixel is `None`. Blocks on the
    /// map, as the 5×5 read does.
    pub(crate) fn read_pick_region(&mut self, device: &wgpu::Device) -> Option<PickRegion> {
        let (requested, rect) = self.region_pending.take()?;
        let staging = self.region_staging.as_ref()?;
        let row_bytes = padded_row_bytes(rect.width) as usize;
        let slice = staging.slice(..(row_bytes * rect.height as usize) as u64);
        let (tx, rx) = std::sync::mpsc::channel();
        let waited = std::time::Instant::now();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        if !rx.recv().map(|r| r.is_ok()).unwrap_or(false) {
            return None;
        }
        // What `specs/gui/edits/eraser-brush.md` § "The map blocks" was
        // measured with.
        log::debug!(
            "pick region {}x{}: map wait {:.3} ms",
            rect.width,
            rect.height,
            waited.elapsed().as_secs_f64() * 1e3
        );
        let data = slice.get_mapped_range().ok()?;
        let points = decode_rows(&data, rect, row_bytes, |id| {
            match self.pick_tables.resolve(id) {
                Some(PickTarget::Point(point)) => Some(point),
                _ => None,
            }
        });
        drop(data);
        staging.unmap();
        Some(PickRegion {
            requested,
            ..PickRegion::new(rect, points)
        })
    }
}

/// Decode the padded rows of a region copy, one pick id per pixel, keeping
/// only ids tagged as points. A run of pixels showing one splat carries one id,
/// so the last id's answer is reused rather than searched for again.
fn decode_rows(
    data: &[u8],
    rect: PixelRect,
    row_bytes: usize,
    resolve: impl Fn(u32) -> Option<PointRef>,
) -> Vec<Option<PointRef>> {
    let mut points = Vec::with_capacity(rect.area());
    let mut last: Option<(u32, Option<PointRef>)> = None;
    for row in 0..rect.height as usize {
        let start = row * row_bytes;
        for pixel in data[start..start + rect.width as usize * 4]
            .as_chunks::<4>()
            .0
        {
            let id = u32::from_le_bytes(*pixel);
            let point = match last {
                Some((last_id, point)) if last_id == id => point,
                _ => {
                    let point = if id & PICK_TAG_MASK == PICK_TAG_POINT {
                        resolve(id)
                    } else {
                        None
                    };
                    last = Some((id, point));
                    point
                }
            };
            points.push(point);
        }
    }
    points
}

#[cfg(test)]
mod tests;
