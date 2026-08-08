// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Dense optical flow field containers: the owned [`FlowField`] and the
//! zero-copy [`FlowFieldRef`] view.
//!
//! Both store `dx` / `dy` as separate row-major (H, W) planes for
//! SIMD-friendly access. The flow at pixel `(x, y)` means the point at
//! `(x, y)` in image A corresponds to `(x + dx, y + dy)` in image B.

/// Borrowed view of a dense optical flow field.
///
/// Zero-copy reference into existing flow data (e.g. numpy arrays).
/// Provides read-only operations: `sample` and `advect_points`.
#[derive(Clone, Copy)]
pub struct FlowFieldRef<'a> {
    width: u32,
    height: u32,
    data_u: &'a [f32],
    data_v: &'a [f32],
}

impl<'a> FlowFieldRef<'a> {
    /// Create a borrowed flow field view from raw slices.
    ///
    /// Each slice must have length `width * height`.
    pub fn from_slices(width: u32, height: u32, data_u: &'a [f32], data_v: &'a [f32]) -> Self {
        let n = (width as usize) * (height as usize);
        assert_eq!(data_u.len(), n);
        assert_eq!(data_v.len(), n);
        Self {
            width,
            height,
            data_u,
            data_v,
        }
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    /// Get flow for the pixel at grid position (col, row).
    pub fn get(&self, col: u32, row: u32) -> (f32, f32) {
        let idx = (row as usize) * (self.width as usize) + (col as usize);
        (self.data_u[idx], self.data_v[idx])
    }

    /// Bilinear interpolation of flow at fractional coordinates.
    /// Uses the pixel-center-at-0.5 convention.
    pub fn sample(&self, x: f32, y: f32) -> (f32, f32) {
        let gx = x - 0.5;
        let gy = y - 0.5;

        let x0 = gx.floor() as i32;
        let y0 = gy.floor() as i32;
        let x1 = x0.saturating_add(1);
        let y1 = y0.saturating_add(1);

        let fx = gx - x0 as f32;
        let fy = gy - y0 as f32;

        let w = self.width as i32;
        let h = self.height as i32;

        let clamp_x = |v: i32| v.clamp(0, w - 1) as u32;
        let clamp_y = |v: i32| v.clamp(0, h - 1) as u32;

        let (dx00, dy00) = self.get(clamp_x(x0), clamp_y(y0));
        let (dx10, dy10) = self.get(clamp_x(x1), clamp_y(y0));
        let (dx01, dy01) = self.get(clamp_x(x0), clamp_y(y1));
        let (dx11, dy11) = self.get(clamp_x(x1), clamp_y(y1));

        let w00 = (1.0 - fx) * (1.0 - fy);
        let w10 = fx * (1.0 - fy);
        let w01 = (1.0 - fx) * fy;
        let w11 = fx * fy;

        let dx = w00 * dx00 + w10 * dx10 + w01 * dx01 + w11 * dx11;
        let dy = w00 * dy00 + w10 * dy10 + w01 * dy01 + w11 * dy11;

        (dx, dy)
    }

    /// Advect a set of 2D points through this flow field.
    /// Points use the pixel-center-at-0.5 convention.
    /// Returns new positions: point + flow(point).
    pub fn advect_points(&self, points: &[(f32, f32)]) -> Vec<(f32, f32)> {
        use rayon::prelude::*;
        points
            .par_iter()
            .map(|&(x, y)| {
                let (dx, dy) = self.sample(x, y);
                (x + dx, y + dy)
            })
            .collect()
    }
}

/// Dense optical flow field.
///
/// Stores per-pixel (dx, dy) displacements in two separate arrays for
/// SIMD-friendly contiguous access. Each array is row-major (H, W) order.
/// The flow at pixel (x, y) means: the point at (x, y) in image A corresponds
/// to (x + dx, y + dy) in image B.
#[derive(Clone)]
pub struct FlowField {
    width: u32,
    height: u32,
    /// Horizontal displacements, row-major, length = width * height.
    pub(super) data_u: Vec<f32>,
    /// Vertical displacements, row-major, length = width * height.
    pub(super) data_v: Vec<f32>,
}

impl FlowField {
    pub fn new(width: u32, height: u32) -> Self {
        let n = (width as usize) * (height as usize);
        Self {
            width,
            height,
            data_u: vec![0.0; n],
            data_v: vec![0.0; n],
        }
    }

    /// Get a borrowed view of this flow field.
    pub fn as_ref(&self) -> FlowFieldRef<'_> {
        FlowFieldRef {
            width: self.width,
            height: self.height,
            data_u: &self.data_u,
            data_v: &self.data_v,
        }
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    /// Get flow for the pixel at grid position (col, row).
    pub fn get(&self, col: u32, row: u32) -> (f32, f32) {
        self.as_ref().get(col, row)
    }

    /// Set flow for the pixel at grid position (col, row).
    pub fn set(&mut self, col: u32, row: u32, dx: f32, dy: f32) {
        let idx = (row as usize) * (self.width as usize) + (col as usize);
        self.data_u[idx] = dx;
        self.data_v[idx] = dy;
    }

    /// Bilinear interpolation of flow at fractional coordinates.
    /// Uses the pixel-center-at-0.5 convention.
    pub fn sample(&self, x: f32, y: f32) -> (f32, f32) {
        self.as_ref().sample(x, y)
    }

    /// Advect a set of 2D points through this flow field.
    /// Points use the pixel-center-at-0.5 convention.
    /// Returns new positions: point + flow(point).
    pub fn advect_points(&self, points: &[(f32, f32)]) -> Vec<(f32, f32)> {
        self.as_ref().advect_points(points)
    }

    /// Create from pre-split u/v data vectors.
    ///
    /// Each vector must have length width * height.
    pub fn from_split(width: u32, height: u32, data_u: Vec<f32>, data_v: Vec<f32>) -> Self {
        let n = (width as usize) * (height as usize);
        assert_eq!(data_u.len(), n);
        assert_eq!(data_v.len(), n);
        Self {
            width,
            height,
            data_u,
            data_v,
        }
    }

    /// Access horizontal displacement data as a slice.
    pub fn u_slice(&self) -> &[f32] {
        &self.data_u
    }

    /// Access horizontal displacement data as a mutable slice.
    pub fn u_slice_mut(&mut self) -> &mut [f32] {
        &mut self.data_u
    }

    /// Access vertical displacement data as a slice.
    pub fn v_slice(&self) -> &[f32] {
        &self.data_v
    }

    /// Access vertical displacement data as a mutable slice.
    pub fn v_slice_mut(&mut self) -> &mut [f32] {
        &mut self.data_v
    }

    /// Downsample flow field by 2x (averaging, magnitudes halved).
    pub fn downsample_2x(&self) -> FlowField {
        let new_w = self.width.div_ceil(2);
        let new_h = self.height.div_ceil(2);
        let mut result = FlowField::new(new_w, new_h);

        for row in 0..new_h {
            for col in 0..new_w {
                // Map new pixel center back to old coordinates
                let src_x = (col as f32 + 0.5) * 2.0;
                let src_y = (row as f32 + 0.5) * 2.0;
                let (dx, dy) = self.sample(src_x, src_y);
                // Halve the magnitude since we're at half resolution
                result.set(col, row, dx * 0.5, dy * 0.5);
            }
        }

        result
    }

    /// Upsample flow field by 2x (bilinear, magnitudes doubled).
    pub fn upsample_2x(&self) -> FlowField {
        let new_w = self.width * 2;
        let new_h = self.height * 2;
        let mut result = FlowField::new(new_w, new_h);

        for row in 0..new_h {
            for col in 0..new_w {
                // Map new pixel center to old pixel center coordinates
                let src_x = (col as f32 + 0.5) * 0.5;
                let src_y = (row as f32 + 0.5) * 0.5;
                let (dx, dy) = self.sample(src_x, src_y);
                // Double the magnitude since we're at 2x resolution
                result.set(col, row, dx * 2.0, dy * 2.0);
            }
        }

        result
    }
}
