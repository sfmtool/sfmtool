// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Bilinear interpolation, image warping, and flow densification.
//!
//! All coordinates use the sfmtool convention: pixel centers at (col + 0.5, row + 0.5).

use super::dis::PatchResult;
use super::{FlowField, GrayImage};

/// Bilinear sample from a grayscale image.
///
/// Coordinates use pixel-center-at-0.5 convention: sampling at (0.5, 0.5)
/// returns the exact value of the top-left pixel. Out-of-bounds coordinates
/// are clamped to the nearest edge pixel.
pub fn sample_bilinear(img: &GrayImage, x: f32, y: f32) -> f32 {
    let gx = x - 0.5;
    let gy = y - 0.5;

    let w = img.width() as i32;
    let h = img.height() as i32;

    let x0 = gx.floor() as i32;
    let y0 = gy.floor() as i32;
    let x1 = x0.saturating_add(1);
    let y1 = y0.saturating_add(1);

    let fx = gx - x0 as f32;
    let fy = gy - y0 as f32;

    let cx0 = x0.clamp(0, w - 1) as u32;
    let cx1 = x1.clamp(0, w - 1) as u32;
    let cy0 = y0.clamp(0, h - 1) as u32;
    let cy1 = y1.clamp(0, h - 1) as u32;

    let v00 = img.get_pixel(cx0, cy0);
    let v10 = img.get_pixel(cx1, cy0);
    let v01 = img.get_pixel(cx0, cy1);
    let v11 = img.get_pixel(cx1, cy1);

    (1.0 - fx) * (1.0 - fy) * v00 + fx * (1.0 - fy) * v10 + (1.0 - fx) * fy * v01 + fx * fy * v11
}

/// Warp an image by a flow field (for computing residuals).
///
/// For each pixel (col, row) in the output, samples img at
/// (col + 0.5 + flow_dx, row + 0.5 + flow_dy) using bilinear interpolation.
#[cfg(test)]
pub(crate) fn warp_image(img: &GrayImage, flow: &FlowField) -> GrayImage {
    let w = flow.width();
    let h = flow.height();
    let mut data = vec![0.0f32; (w as usize) * (h as usize)];

    for row in 0..h {
        for col in 0..w {
            let (dx, dy) = flow.get(col, row);
            let sx = col as f32 + 0.5 + dx;
            let sy = row as f32 + 0.5 + dy;
            data[(row as usize) * (w as usize) + (col as usize)] = sample_bilinear(img, sx, sy);
        }
    }

    GrayImage::new(w, h, data)
}

/// Densify sparse patch flow updates to a full dense field using
/// photometric-error-weighted averaging (Eq. 3 from the paper).
///
/// Each patch's contribution at pixel x is weighted by:
///   w = 1 / max(1, ||d_i(x)||_2)
/// where d_i(x) is the per-pixel intensity difference between warped query and template.
pub(crate) fn densify_flow(
    patches: &[PatchResult],
    ref_image: &GrayImage,
    tgt_image: &GrayImage,
    width: u32,
    height: u32,
    patch_size: u32,
) -> FlowField {
    let w = width as usize;
    let h = height as usize;
    let ps = patch_size as usize;

    // Accumulate weighted flow contributions
    let mut flow_dx = vec![0.0f32; w * h];
    let mut flow_dy = vec![0.0f32; w * h];
    let mut weight_sum = vec![0.0f32; w * h];

    for patch in patches {
        let gx = patch.grid_x as usize;
        let gy = patch.grid_y as usize;
        let (fdx, fdy) = patch.final_flow;

        for py in 0..ps {
            let row = gy + py;
            if row >= h {
                break;
            }
            for px in 0..ps {
                let col = gx + px;
                if col >= w {
                    break;
                }

                // Compute photometric error at this pixel
                let ref_val = ref_image.get_pixel(col as u32, row as u32);
                let sx = col as f32 + 0.5 + fdx;
                let sy = row as f32 + 0.5 + fdy;
                let tgt_val = sample_bilinear(tgt_image, sx, sy);
                let diff = (tgt_val - ref_val).abs();
                let weight = 1.0 / diff.max(1.0);

                let idx = row * w + col;
                flow_dx[idx] += weight * fdx;
                flow_dy[idx] += weight * fdy;
                weight_sum[idx] += weight;
            }
        }
    }

    // Normalize
    let mut result = FlowField::new(width, height);
    for row in 0..h {
        for col in 0..w {
            let idx = row * w + col;
            if weight_sum[idx] > 0.0 {
                result.set(
                    col as u32,
                    row as u32,
                    flow_dx[idx] / weight_sum[idx],
                    flow_dy[idx] / weight_sum[idx],
                );
            }
        }
    }

    result
}

#[cfg(test)]
mod tests;
