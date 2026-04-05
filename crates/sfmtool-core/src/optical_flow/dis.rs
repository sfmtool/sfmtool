// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! DIS (Dense Inverse Search) algorithm core.
//!
//! Implements the per-level refinement step: grid creation, patch initialization,
//! inverse search, outlier rejection, and densification.

use super::gpu::GpuFlowContext;
use super::interp::{densify_flow, sample_bilinear};
use super::variational::{variational_refine, VariationalParams};
use super::{DisFlowParams, FlowField, GrayImage};
use rayon::prelude::*;

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// Result of inverse search for a single patch.
pub(crate) struct PatchResult {
    /// Grid position (top-left corner of patch in reference image).
    pub grid_x: u32,
    pub grid_y: u32,
    /// Final displacement after inverse search iterations (with outlier rejection applied).
    pub final_flow: (f32, f32),
}

/// Timing breakdown for a single pyramid level.
#[derive(Clone, Debug, Default)]
pub(crate) struct LevelTiming {
    /// DIS inverse search + densification time in seconds.
    pub dis_secs: f64,
    /// Variational refinement time in seconds (0 if disabled).
    pub var_secs: f64,
}

/// Run DIS at a single pyramid level.
///
/// Takes the current flow estimate and refines it via inverse search on a
/// regular patch grid, outlier rejection, densification, and optional
/// variational refinement.
pub(crate) fn refine_flow_at_level(
    ref_image: &GrayImage,
    tgt_image: &GrayImage,
    flow: &mut FlowField,
    params: &DisFlowParams,
    scale_index: u32,
    gpu: Option<&GpuFlowContext>,
) -> LevelTiming {
    refine_flow_at_level_inner(ref_image, tgt_image, flow, params, scale_index, gpu)
}

fn refine_flow_at_level_inner(
    ref_image: &GrayImage,
    tgt_image: &GrayImage,
    flow: &mut FlowField,
    params: &DisFlowParams,
    scale_index: u32,
    gpu: Option<&GpuFlowContext>,
) -> LevelTiming {
    use std::time::Instant;

    let w = ref_image.width();
    let h = ref_image.height();
    let mut timing = LevelTiming::default();

    // GPU path: run gradient computation, inverse search, and densification on GPU.
    // Skip GPU for small levels where dispatch overhead exceeds compute savings.
    #[cfg(feature = "gpu")]
    let use_gpu = gpu.is_some() && (w * h) >= params.gpu_min_pixels;
    #[cfg(not(feature = "gpu"))]
    let use_gpu = false;

    if use_gpu {
        #[cfg(feature = "gpu")]
        {
            let gpu_ctx = gpu.unwrap();

            if params.variational_refinement {
                // Combined DIS + variational in a single GPU submission.
                // Eliminates redundant image uploads and flow round-trips.
                let var_params = VariationalParams {
                    delta: params.variational_delta,
                    gamma: params.variational_gamma,
                    alpha: params.variational_alpha,
                    jacobi_iterations: params.variational_jacobi_iterations,
                    outer_iterations: params.variational_outer_iterations_base * (scale_index + 1),
                };
                let t_combined = Instant::now();
                gpu_ctx.run_dis_and_variational(ref_image, tgt_image, flow, params, &var_params);
                let total = t_combined.elapsed().as_secs_f64();
                // Cannot separate DIS/variational timing in combined submission
                timing.dis_secs = total;
            } else {
                let t_dis = Instant::now();
                gpu_ctx.run_dis_level(ref_image, tgt_image, flow, params);
                timing.dis_secs = t_dis.elapsed().as_secs_f64();
            }
            return timing;
        }
    }

    // CPU path: gradient computation, inverse search, and densification
    let t_dis = Instant::now();
    let ps = params.patch_size;
    let stride = params.patch_stride();

    // Precompute image gradients for the reference image
    let (grad_x, grad_y) = compute_gradients(ref_image);

    // Build grid positions, then run inverse search in parallel
    let mut grid: Vec<(u32, u32, (f32, f32))> = Vec::new();
    let mut gy = 0u32;
    while gy + ps <= h {
        let mut gx = 0u32;
        while gx + ps <= w {
            let cx = gx + ps / 2;
            let cy = gy + ps / 2;
            let initial_flow = flow.get(cx.min(w - 1), cy.min(h - 1));
            grid.push((gx, gy, initial_flow));
            gx += stride;
        }
        gy += stride;
    }

    let ps_sq = (ps * ps) as f32;
    let patches: Vec<PatchResult> = grid
        .par_iter()
        .map(|&(gx, gy, initial_flow)| {
            let mut final_flow = inverse_search(
                ref_image,
                tgt_image,
                &grad_x,
                &grad_y,
                gx,
                gy,
                initial_flow,
                params,
            );

            // Reject outlier: reset to initial flow if update exceeds patch_size
            let ddx = final_flow.0 - initial_flow.0;
            let ddy = final_flow.1 - initial_flow.1;
            if ddx * ddx + ddy * ddy > ps_sq {
                final_flow = initial_flow;
            }

            PatchResult {
                grid_x: gx,
                grid_y: gy,
                final_flow,
            }
        })
        .collect();

    // Densify: create full flow field from sparse patch results
    let dense = densify_flow(&patches, ref_image, tgt_image, w, h, ps);

    // Copy dense flow into the output
    flow.u_slice_mut().copy_from_slice(dense.u_slice());
    flow.v_slice_mut().copy_from_slice(dense.v_slice());
    timing.dis_secs = t_dis.elapsed().as_secs_f64();

    // Optionally apply variational refinement (CPU path only — GPU path returns early above)
    #[cfg(not(feature = "gpu"))]
    let _ = gpu;
    if params.variational_refinement {
        let var_params = VariationalParams {
            delta: params.variational_delta,
            gamma: params.variational_gamma,
            alpha: params.variational_alpha,
            jacobi_iterations: params.variational_jacobi_iterations,
            outer_iterations: params.variational_outer_iterations_base * (scale_index + 1),
        };
        let t_var = Instant::now();
        variational_refine(ref_image, tgt_image, flow, &var_params);
        timing.var_secs = t_var.elapsed().as_secs_f64();
    }

    timing
}

/// Compute image gradients using central differences.
///
/// Returns (grad_x, grad_y) images.
fn compute_gradients(img: &GrayImage) -> (GrayImage, GrayImage) {
    let w = img.width();
    let h = img.height();
    let mut gx_data = vec![0.0f32; (w as usize) * (h as usize)];
    let mut gy_data = vec![0.0f32; (w as usize) * (h as usize)];

    for row in 0..h {
        for col in 0..w {
            let left = if col > 0 {
                img.get_pixel(col - 1, row)
            } else {
                img.get_pixel(0, row)
            };
            let right = if col + 1 < w {
                img.get_pixel(col + 1, row)
            } else {
                img.get_pixel(w - 1, row)
            };
            let up = if row > 0 {
                img.get_pixel(col, row - 1)
            } else {
                img.get_pixel(col, 0)
            };
            let down = if row + 1 < h {
                img.get_pixel(col, row + 1)
            } else {
                img.get_pixel(col, h - 1)
            };

            let idx = (row as usize) * (w as usize) + (col as usize);
            gx_data[idx] = (right - left) * 0.5;
            gy_data[idx] = (down - up) * 0.5;
        }
    }

    (GrayImage::new(w, h, gx_data), GrayImage::new(w, h, gy_data))
}

/// Perform inverse search for a single patch.
///
/// Uses the inverse compositional image alignment of Baker and Matthews (2001):
/// - Precompute S' = nabla(T) (steepest descent images = template gradients)
/// - Precompute H' = sum(S'^T S') (2x2 Hessian)
/// - Iterate: du = H'^{-1} sum(S'^T * [I(x+u) - T(x)]), u <- u - du
#[allow(clippy::too_many_arguments)]
fn inverse_search(
    ref_image: &GrayImage,
    tgt_image: &GrayImage,
    grad_x: &GrayImage,
    grad_y: &GrayImage,
    patch_x: u32,
    patch_y: u32,
    initial_flow: (f32, f32),
    params: &DisFlowParams,
) -> (f32, f32) {
    let ps = params.patch_size as usize;

    // Precompute the Hessian H' = sum(S'^T * S') where S' = [gx, gy]
    // H' is a 2x2 matrix: [[sum(gx*gx), sum(gx*gy)], [sum(gx*gy), sum(gy*gy)]]
    let mut h00 = 0.0f32;
    let mut h01 = 0.0f32;
    let mut h11 = 0.0f32;

    // Also precompute template patch (with optional mean subtraction)
    let mut template = vec![0.0f32; ps * ps];
    let mut template_mean = 0.0f32;

    // Store gradients for the patch
    let mut patch_gx = vec![0.0f32; ps * ps];
    let mut patch_gy = vec![0.0f32; ps * ps];

    for py in 0..ps {
        for px in 0..ps {
            let col = patch_x as usize + px;
            let row = patch_y as usize + py;
            let idx = py * ps + px;

            template[idx] = ref_image.get_pixel(col as u32, row as u32);
            template_mean += template[idx];

            let gx = grad_x.get_pixel(col as u32, row as u32);
            let gy = grad_y.get_pixel(col as u32, row as u32);
            patch_gx[idx] = gx;
            patch_gy[idx] = gy;

            h00 += gx * gx;
            h01 += gx * gy;
            h11 += gy * gy;
        }
    }

    let n = (ps * ps) as f32;
    template_mean /= n;

    // Invert 2x2 Hessian
    let det = h00 * h11 - h01 * h01;
    if det.abs() < 1e-10 {
        // Singular Hessian — no gradient structure, return initial flow
        return initial_flow;
    }
    let inv_det = 1.0 / det;
    let ih00 = h11 * inv_det;
    let ih01 = -h01 * inv_det;
    let ih11 = h00 * inv_det;

    let mut u = initial_flow;

    for _ in 0..params.grad_descent_iterations {
        let (b0, b1) = compute_iteration(
            tgt_image,
            &template,
            template_mean,
            &patch_gx,
            &patch_gy,
            patch_x,
            patch_y,
            ps,
            u,
            params.normalize_patches,
        );

        // Compute update: du = H'^{-1} * b
        let du0 = ih00 * b0 + ih01 * b1;
        let du1 = ih01 * b0 + ih11 * b1;

        // Apply update (inverse compositional: u <- u - du)
        u.0 -= du0;
        u.1 -= du1;
    }

    u
}

/// Compute (b0, b1) for one DIS iteration, dispatching to SIMD when possible.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn compute_iteration(
    tgt_image: &GrayImage,
    template: &[f32],
    template_mean: f32,
    patch_gx: &[f32],
    patch_gy: &[f32],
    patch_x: u32,
    patch_y: u32,
    ps: usize,
    u: (f32, f32),
    normalize: bool,
) -> (f32, f32) {
    #[cfg(target_arch = "x86_64")]
    {
        if ps.is_multiple_of(4) {
            let ux_floor = u.0.floor() as i32;
            let uy_floor = u.1.floor() as i32;
            let x0_base = patch_x as i32 + ux_floor;
            let y0_base = patch_y as i32 + uy_floor;
            if x0_base >= 0
                && (x0_base + ps as i32) < tgt_image.width() as i32
                && y0_base >= 0
                && (y0_base + ps as i32) < tgt_image.height() as i32
            {
                return unsafe {
                    compute_iteration_sse2(
                        tgt_image.data().as_ptr(),
                        tgt_image.width() as usize,
                        template,
                        template_mean,
                        patch_gx,
                        patch_gy,
                        ps,
                        x0_base as usize,
                        y0_base as usize,
                        u.0 - u.0.floor(),
                        u.1 - u.1.floor(),
                        normalize,
                    )
                };
            }
        }
    }

    compute_iteration_scalar(
        tgt_image,
        template,
        template_mean,
        patch_gx,
        patch_gy,
        patch_x,
        patch_y,
        ps,
        u,
        normalize,
    )
}

/// Scalar iteration body: compute warp mean and gradient-weighted residuals.
#[allow(clippy::too_many_arguments)]
fn compute_iteration_scalar(
    tgt_image: &GrayImage,
    template: &[f32],
    template_mean: f32,
    patch_gx: &[f32],
    patch_gy: &[f32],
    patch_x: u32,
    patch_y: u32,
    ps: usize,
    u: (f32, f32),
    normalize: bool,
) -> (f32, f32) {
    let n = (ps * ps) as f32;
    let mut warp_mean = 0.0f32;

    if normalize {
        for py in 0..ps {
            for px in 0..ps {
                let col = patch_x as usize + px;
                let row = patch_y as usize + py;
                let sx = col as f32 + 0.5 + u.0;
                let sy = row as f32 + 0.5 + u.1;
                warp_mean += sample_bilinear(tgt_image, sx, sy);
            }
        }
        warp_mean /= n;
    }

    let mut b0 = 0.0f32;
    let mut b1 = 0.0f32;

    for py in 0..ps {
        for px in 0..ps {
            let col = patch_x as usize + px;
            let row = patch_y as usize + py;
            let idx = py * ps + px;
            let sx = col as f32 + 0.5 + u.0;
            let sy = row as f32 + 0.5 + u.1;
            let tgt_val = sample_bilinear(tgt_image, sx, sy);
            let residual = if normalize {
                (tgt_val - warp_mean) - (template[idx] - template_mean)
            } else {
                tgt_val - template[idx]
            };
            b0 += patch_gx[idx] * residual;
            b1 += patch_gy[idx] * residual;
        }
    }

    (b0, b1)
}

/// Horizontal sum of 4 f32 lanes in an SSE register.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn hsum_sse(v: __m128) -> f32 {
    let hi = _mm_movehl_ps(v, v);
    let sum = _mm_add_ps(v, hi);
    let shuf = _mm_shuffle_ps::<1>(sum, sum);
    _mm_cvtss_f32(_mm_add_ss(sum, shuf))
}

/// SSE2-optimized iteration body: bilinear-sample 4 pixels at a time.
///
/// Exploits the fact that for 4 consecutive horizontal pixels in a patch row,
/// all bilinear samples share the same fractional coordinates (fx, fy) and
/// have consecutive integer base coordinates, enabling 4-wide SIMD loads.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
#[allow(clippy::too_many_arguments)]
unsafe fn compute_iteration_sse2(
    tgt_data: *const f32,
    tgt_width: usize,
    template: &[f32],
    template_mean: f32,
    patch_gx: &[f32],
    patch_gy: &[f32],
    ps: usize,
    x0_base: usize,
    y0_base: usize,
    fx: f32,
    fy: f32,
    normalize: bool,
) -> (f32, f32) {
    let n = (ps * ps) as f32;

    let fx_v = _mm_set1_ps(fx);
    let fy_v = _mm_set1_ps(fy);
    let one = _mm_set1_ps(1.0);
    let omfx = _mm_sub_ps(one, fx_v);
    let omfy = _mm_sub_ps(one, fy_v);

    // Bilinear-sample 4 consecutive horizontal pixels starting at (x0, y0).
    // Loads pixels from rows y0 and y0+1, columns x0..x0+4.
    macro_rules! bilerp4 {
        ($x0:expr, $y0:expr) => {{
            let off_t = $y0 * tgt_width + $x0;
            let off_b = off_t + tgt_width;
            let tl = _mm_loadu_ps(tgt_data.add(off_t));
            let tr = _mm_loadu_ps(tgt_data.add(off_t + 1));
            let bl = _mm_loadu_ps(tgt_data.add(off_b));
            let br = _mm_loadu_ps(tgt_data.add(off_b + 1));
            let top = _mm_add_ps(_mm_mul_ps(omfx, tl), _mm_mul_ps(fx_v, tr));
            let bot = _mm_add_ps(_mm_mul_ps(omfx, bl), _mm_mul_ps(fx_v, br));
            _mm_add_ps(_mm_mul_ps(omfy, top), _mm_mul_ps(fy_v, bot))
        }};
    }

    // Phase 1: Compute warp mean
    let mut warp_mean = 0.0f32;
    if normalize {
        let mut sum_v = _mm_setzero_ps();
        for py in 0..ps {
            let y0 = y0_base + py;
            let mut px = 0;
            while px + 4 <= ps {
                sum_v = _mm_add_ps(sum_v, bilerp4!(x0_base + px, y0));
                px += 4;
            }
        }
        warp_mean = hsum_sse(sum_v) / n;
    }

    // Phase 2: Accumulate gradient-weighted residuals
    let mut b0_v = _mm_setzero_ps();
    let mut b1_v = _mm_setzero_ps();
    let wm_v = _mm_set1_ps(warp_mean);
    let tm_v = _mm_set1_ps(template_mean);

    for py in 0..ps {
        let y0 = y0_base + py;
        let row_off = py * ps;
        let mut px = 0;
        while px + 4 <= ps {
            let tgt_val = bilerp4!(x0_base + px, y0);
            let idx = row_off + px;
            let t = _mm_loadu_ps(template.as_ptr().add(idx));
            let gx = _mm_loadu_ps(patch_gx.as_ptr().add(idx));
            let gy = _mm_loadu_ps(patch_gy.as_ptr().add(idx));

            let residual = if normalize {
                _mm_sub_ps(_mm_sub_ps(tgt_val, wm_v), _mm_sub_ps(t, tm_v))
            } else {
                _mm_sub_ps(tgt_val, t)
            };

            b0_v = _mm_add_ps(b0_v, _mm_mul_ps(gx, residual));
            b1_v = _mm_add_ps(b1_v, _mm_mul_ps(gy, residual));
            px += 4;
        }
    }

    (hsum_sse(b0_v), hsum_sse(b1_v))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_gradients_constant() {
        let img = GrayImage::new_constant(8, 8, 0.5);
        let (gx, gy) = compute_gradients(&img);
        for &v in gx.data() {
            assert!(v.abs() < 1e-6);
        }
        for &v in gy.data() {
            assert!(v.abs() < 1e-6);
        }
    }

    #[test]
    fn test_compute_gradients_horizontal_ramp() {
        // Linear ramp in x: pixel values = col / (w-1)
        let w = 8u32;
        let h = 4u32;
        let data: Vec<f32> = (0..h)
            .flat_map(|_| (0..w).map(|c| c as f32 / (w - 1) as f32))
            .collect();
        let img = GrayImage::new(w, h, data);
        let (gx, gy) = compute_gradients(&img);

        // Interior pixels should have gx ≈ 1/(w-1) * 0.5 * 2 = 1/(w-1)
        // which is the central difference of a linear ramp with step 1/(w-1)
        let expected_gx = 1.0 / (w - 1) as f32;
        for row in 0..h {
            for col in 1..w - 1 {
                let val = gx.get_pixel(col, row);
                assert!(
                    (val - expected_gx).abs() < 1e-5,
                    "gx at ({},{}) = {}, expected {}",
                    col,
                    row,
                    val,
                    expected_gx
                );
            }
        }

        // gy should be ~0 for horizontal ramp
        for row in 1..h - 1 {
            for col in 0..w {
                let val = gy.get_pixel(col, row);
                assert!(val.abs() < 1e-5, "gy at ({},{}) = {}", col, row, val);
            }
        }
    }

    /// Generate a deterministic pseudo-random image for SIMD vs scalar testing.
    /// Uses a simple LCG to avoid pulling in a rand crate dependency.
    fn pseudo_random_image(width: u32, height: u32, seed: u32) -> GrayImage {
        let mut state = seed;
        let data: Vec<f32> = (0..(width as usize * height as usize))
            .map(|_| {
                // LCG: state = state * 1103515245 + 12345
                state = state.wrapping_mul(1103515245).wrapping_add(12345);
                // Map to [0, 1]
                (state >> 16) as f32 / 65535.0
            })
            .collect();
        GrayImage::new(width, height, data)
    }

    /// Helper: call both scalar and SSE2 paths and assert they match.
    #[cfg(target_arch = "x86_64")]
    fn assert_scalar_simd_equivalent(
        tgt_image: &GrayImage,
        template: &[f32],
        template_mean: f32,
        patch_gx: &[f32],
        patch_gy: &[f32],
        patch_x: u32,
        patch_y: u32,
        ps: usize,
        u: (f32, f32),
        normalize: bool,
    ) {
        let (b0_scalar, b1_scalar) = compute_iteration_scalar(
            tgt_image,
            template,
            template_mean,
            patch_gx,
            patch_gy,
            patch_x,
            patch_y,
            ps,
            u,
            normalize,
        );

        let ux_floor = u.0.floor() as i32;
        let uy_floor = u.1.floor() as i32;
        let x0_base = patch_x as i32 + ux_floor;
        let y0_base = patch_y as i32 + uy_floor;
        assert!(x0_base >= 0);
        assert!((x0_base + ps as i32) < tgt_image.width() as i32);
        assert!(y0_base >= 0);
        assert!((y0_base + ps as i32) < tgt_image.height() as i32);

        let (b0_simd, b1_simd) = unsafe {
            compute_iteration_sse2(
                tgt_image.data().as_ptr(),
                tgt_image.width() as usize,
                template,
                template_mean,
                patch_gx,
                patch_gy,
                ps,
                x0_base as usize,
                y0_base as usize,
                u.0 - u.0.floor(),
                u.1 - u.1.floor(),
                normalize,
            )
        };

        assert!(
            (b0_scalar - b0_simd).abs() < 1e-4,
            "b0 mismatch: scalar={b0_scalar}, simd={b0_simd}, diff={}",
            (b0_scalar - b0_simd).abs()
        );
        assert!(
            (b1_scalar - b1_simd).abs() < 1e-4,
            "b1 mismatch: scalar={b1_scalar}, simd={b1_simd}, diff={}",
            (b1_scalar - b1_simd).abs()
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_simd_vs_scalar_ps4_zero_flow() {
        let tgt = pseudo_random_image(32, 32, 42);
        let ref_img = pseudo_random_image(32, 32, 99);
        let ps = 4usize;
        let patch_x = 10u32;
        let patch_y = 10u32;

        // Build template and gradients from ref image
        let (grad_x, grad_y) = compute_gradients(&ref_img);
        let mut template = vec![0.0f32; ps * ps];
        let mut patch_gx = vec![0.0f32; ps * ps];
        let mut patch_gy = vec![0.0f32; ps * ps];
        let mut template_mean = 0.0f32;
        for py in 0..ps {
            for px in 0..ps {
                let idx = py * ps + px;
                let col = patch_x as usize + px;
                let row = patch_y as usize + py;
                template[idx] = ref_img.get_pixel(col as u32, row as u32);
                template_mean += template[idx];
                patch_gx[idx] = grad_x.get_pixel(col as u32, row as u32);
                patch_gy[idx] = grad_y.get_pixel(col as u32, row as u32);
            }
        }
        template_mean /= (ps * ps) as f32;

        assert_scalar_simd_equivalent(
            &tgt,
            &template,
            template_mean,
            &patch_gx,
            &patch_gy,
            patch_x,
            patch_y,
            ps,
            (0.0, 0.0),
            true,
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_simd_vs_scalar_ps8_fractional_flow() {
        let tgt = pseudo_random_image(64, 64, 123);
        let ref_img = pseudo_random_image(64, 64, 456);
        let ps = 8usize;
        let patch_x = 16u32;
        let patch_y = 20u32;

        let (grad_x, grad_y) = compute_gradients(&ref_img);
        let mut template = vec![0.0f32; ps * ps];
        let mut patch_gx = vec![0.0f32; ps * ps];
        let mut patch_gy = vec![0.0f32; ps * ps];
        let mut template_mean = 0.0f32;
        for py in 0..ps {
            for px in 0..ps {
                let idx = py * ps + px;
                let col = patch_x as usize + px;
                let row = patch_y as usize + py;
                template[idx] = ref_img.get_pixel(col as u32, row as u32);
                template_mean += template[idx];
                patch_gx[idx] = grad_x.get_pixel(col as u32, row as u32);
                patch_gy[idx] = grad_y.get_pixel(col as u32, row as u32);
            }
        }
        template_mean /= (ps * ps) as f32;

        // Test several fractional flow values
        for &u in &[
            (0.0, 0.0),
            (1.3, -0.7),
            (-2.5, 3.2),
            (0.0, 4.9),
            (-3.0, -1.0),
        ] {
            for normalize in [true, false] {
                assert_scalar_simd_equivalent(
                    &tgt,
                    &template,
                    template_mean,
                    &patch_gx,
                    &patch_gy,
                    patch_x,
                    patch_y,
                    ps,
                    u,
                    normalize,
                );
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_simd_vs_scalar_ps8_negative_fractional_flow() {
        // Specifically test negative fractional parts, where floor(-0.3) = -1
        let tgt = pseudo_random_image(64, 64, 789);
        let ref_img = pseudo_random_image(64, 64, 321);
        let ps = 8usize;
        let patch_x = 20u32;
        let patch_y = 20u32;

        let (grad_x, grad_y) = compute_gradients(&ref_img);
        let mut template = vec![0.0f32; ps * ps];
        let mut patch_gx = vec![0.0f32; ps * ps];
        let mut patch_gy = vec![0.0f32; ps * ps];
        let mut template_mean = 0.0f32;
        for py in 0..ps {
            for px in 0..ps {
                let idx = py * ps + px;
                let col = patch_x as usize + px;
                let row = patch_y as usize + py;
                template[idx] = ref_img.get_pixel(col as u32, row as u32);
                template_mean += template[idx];
                patch_gx[idx] = grad_x.get_pixel(col as u32, row as u32);
                patch_gy[idx] = grad_y.get_pixel(col as u32, row as u32);
            }
        }
        template_mean /= (ps * ps) as f32;

        for &u in &[(-0.3, -0.7), (-0.9, -0.1), (-5.5, 2.3)] {
            assert_scalar_simd_equivalent(
                &tgt,
                &template,
                template_mean,
                &patch_gx,
                &patch_gy,
                patch_x,
                patch_y,
                ps,
                u,
                true,
            );
        }
    }

    #[test]
    fn test_zero_flow_for_identical_images() {
        let img = GrayImage::checkerboard(32, 32);
        let mut flow = FlowField::new(32, 32);
        let params = DisFlowParams {
            variational_refinement: false,
            ..DisFlowParams::fast()
        };
        refine_flow_at_level(&img, &img, &mut flow, &params, 0, None);

        let mut max_flow = 0.0f32;
        for row in 0..32 {
            for col in 0..32 {
                let (dx, dy) = flow.get(col, row);
                max_flow = max_flow.max(dx.abs()).max(dy.abs());
            }
        }
        assert!(
            max_flow < 0.5,
            "Identical images should give near-zero flow, got max {}",
            max_flow
        );
    }
}
