// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Variational refinement for optical flow.
//!
//! Minimizes an energy with intensity constancy, gradient constancy,
//! and smoothness terms using a Jacobi iterative solver.
//!
//! Reference: Kroeger et al. Section 2.3, Brox et al. ECCV 2004.

use super::interp::sample_bilinear;
use super::{FlowField, GrayImage};
use rayon::prelude::*;

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// Wrapper to send a raw mutable pointer across threads.
/// Safety: the caller must ensure non-overlapping access per thread.
struct SendPtr(*mut f32);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}

impl SendPtr {
    fn ptr(&self) -> *mut f32 {
        self.0
    }
}

/// Parameters for variational refinement.
pub(crate) struct VariationalParams {
    /// Intensity data term weight (delta).
    pub delta: f32,
    /// Gradient data term weight (gamma).
    pub gamma: f32,
    /// Smoothness weight (alpha).
    pub alpha: f32,
    /// Inner Jacobi solver iterations per outer fixed-point iteration.
    ///
    /// This is NOT equivalent to the SOR iterations (θ_vi) in Kroeger et al. or
    /// OpenCV's DIS implementation. Jacobi converges slower per iteration than
    /// Gauss-Seidel SOR, so this value should be ~1.3-2× the equivalent SOR
    /// count. The tradeoff is that Jacobi iterations are fully parallelizable
    /// (SIMD, multi-core) while SOR iterations are sequential.
    pub jacobi_iterations: u32,
    /// Outer fixed-point iterations for this scale.
    pub outer_iterations: u32,
}

/// Robust penalizer derivative: Psi'(s^2) = 1 / (2 * sqrt(s^2 + eps^2)).
///
/// Used to weight the data and smoothness terms.
fn psi_deriv(s_sq: f32) -> f32 {
    let eps_sq = 1e-6; // epsilon^2 = 0.001^2
    1.0 / (2.0 * (s_sq + eps_sq).sqrt())
}

/// Apply variational refinement to smooth and improve the flow field.
///
/// Minimizes: E(U) = integral [ delta * Psi(E_I) + gamma * Psi(E_G) + alpha * Psi(E_S) ] dx
///
/// where:
/// - E_I = intensity constancy term
/// - E_G = gradient constancy term
/// - E_S = smoothness term (||nabla u||^2 + ||nabla v||^2)
pub(crate) fn variational_refine(
    ref_image: &GrayImage,
    tgt_image: &GrayImage,
    flow: &mut FlowField,
    params: &VariationalParams,
) {
    let w = flow.width() as usize;
    let h = flow.height() as usize;

    if w < 3 || h < 3 {
        return;
    }

    // Compute reference image gradients and their second derivatives.
    // These are constant across outer iterations — compute once.
    let (ref_ix, ref_iy) = compute_image_gradients(ref_image);
    let (ref_ixx, ref_ixy) = compute_image_gradients(&ref_ix);
    let (ref_iyx, ref_iyy) = compute_image_gradients(&ref_iy);

    for _outer in 0..params.outer_iterations {
        // Warp target image and its gradients by current flow
        let warped = warp_by_flow(tgt_image, flow);
        let (warped_ix, warped_iy) = compute_image_gradients(&warped);

        // Compute second derivatives of warped gradients (ref ones are precomputed)
        let (warped_ixx, warped_ixy) = compute_image_gradients(&warped_ix);
        let (warped_iyx, warped_iyy) = compute_image_gradients(&warped_iy);

        // Precompute per-pixel data for the SOR solver
        // Each pixel has coefficients for the linear system Au = b
        let mut a11 = vec![0.0f32; w * h]; // coefficient of du
        let mut a12 = vec![0.0f32; w * h]; // cross-coupling
        let mut a22 = vec![0.0f32; w * h]; // coefficient of dv
        let mut b1 = vec![0.0f32; w * h]; // right-hand side for u
        let mut b2 = vec![0.0f32; w * h]; // right-hand side for v

        for row in 0..h {
            for col in 0..w {
                let idx = row * w + col;

                // Intensity constancy term
                let iz = warped.get_pixel(col as u32, row as u32)
                    - ref_image.get_pixel(col as u32, row as u32);
                // Average gradients
                let ix = 0.5
                    * (ref_ix.get_pixel(col as u32, row as u32)
                        + warped_ix.get_pixel(col as u32, row as u32));
                let iy = 0.5
                    * (ref_iy.get_pixel(col as u32, row as u32)
                        + warped_iy.get_pixel(col as u32, row as u32));

                // Normalization factor for intensity term
                let grad_sq = ix * ix + iy * iy;
                let beta0 = 1.0 / (grad_sq + 0.01);

                // Psi weight for intensity term
                let psi_i = psi_deriv(beta0 * iz * iz);
                let wi = params.delta * beta0 * psi_i;

                a11[idx] += wi * ix * ix;
                a12[idx] += wi * ix * iy;
                a22[idx] += wi * iy * iy;
                b1[idx] -= wi * ix * iz;
                b2[idx] -= wi * iy * iz;

                // Gradient constancy term (x-derivative)
                let ixz = warped_ix.get_pixel(col as u32, row as u32)
                    - ref_ix.get_pixel(col as u32, row as u32);
                let ixx = 0.5
                    * (ref_ixx.get_pixel(col as u32, row as u32)
                        + warped_ixx.get_pixel(col as u32, row as u32));
                let ixy = 0.5
                    * (ref_ixy.get_pixel(col as u32, row as u32)
                        + warped_ixy.get_pixel(col as u32, row as u32));

                let grad_sq_x = ixx * ixx + ixy * ixy;
                let beta_x = 1.0 / (grad_sq_x + 0.01);
                let psi_gx = psi_deriv(beta_x * ixz * ixz);
                let wgx = params.gamma * beta_x * psi_gx;

                a11[idx] += wgx * ixx * ixx;
                a12[idx] += wgx * ixx * ixy;
                a22[idx] += wgx * ixy * ixy;
                b1[idx] -= wgx * ixx * ixz;
                b2[idx] -= wgx * ixy * ixz;

                // Gradient constancy term (y-derivative)
                let iyz = warped_iy.get_pixel(col as u32, row as u32)
                    - ref_iy.get_pixel(col as u32, row as u32);
                let iyx = 0.5
                    * (ref_iyx.get_pixel(col as u32, row as u32)
                        + warped_iyx.get_pixel(col as u32, row as u32));
                let iyy = 0.5
                    * (ref_iyy.get_pixel(col as u32, row as u32)
                        + warped_iyy.get_pixel(col as u32, row as u32));

                let grad_sq_y = iyx * iyx + iyy * iyy;
                let beta_y = 1.0 / (grad_sq_y + 0.01);
                let psi_gy = psi_deriv(beta_y * iyz * iyz);
                let wgy = params.gamma * beta_y * psi_gy;

                a11[idx] += wgy * iyx * iyx;
                a12[idx] += wgy * iyx * iyy;
                a22[idx] += wgy * iyy * iyy;
                b1[idx] -= wgy * iyx * iyz;
                b2[idx] -= wgy * iyy * iyz;
            }
        }

        // Jacobi iterations for the smoothness-regularized system.
        //
        // Unlike Gauss-Seidel SOR (which reads freshly-updated neighbors and has
        // sequential dependencies), Jacobi reads only from the previous iteration's
        // values. This makes all pixels within an iteration fully independent,
        // enabling SIMD vectorization and rayon parallelism.
        let mut du = vec![0.0f32; w * h];
        let mut dv = vec![0.0f32; w * h];
        let mut du_new = vec![0.0f32; w * h];
        let mut dv_new = vec![0.0f32; w * h];

        {
            let flow_u = flow.u_slice();
            let flow_v = flow.v_slice();

            for _inner in 0..params.jacobi_iterations {
                jacobi_iteration(
                    &du,
                    &dv,
                    &mut du_new,
                    &mut dv_new,
                    flow_u,
                    flow_v,
                    &a11,
                    &a12,
                    &a22,
                    &b1,
                    &b2,
                    w,
                    h,
                    params.alpha,
                );
                std::mem::swap(&mut du, &mut du_new);
                std::mem::swap(&mut dv, &mut dv_new);
            }
        }

        // Apply the computed update to the flow
        let flow_u_mut = flow.u_slice_mut();
        for i in 0..w * h {
            flow_u_mut[i] += du[i];
        }
        let flow_v_mut = flow.v_slice_mut();
        for i in 0..w * h {
            flow_v_mut[i] += dv[i];
        }
    }
}

/// Run one Jacobi iteration over all pixels, dispatching rows in parallel.
///
/// Each row writes only to its own `[row*w .. (row+1)*w]` slice of `du_new`/`dv_new`,
/// so rows can execute concurrently without data races. Interior rows use SSE2 SIMD
/// for the interior columns.
#[allow(clippy::too_many_arguments)]
fn jacobi_iteration(
    du: &[f32],
    dv: &[f32],
    du_new: &mut [f32],
    dv_new: &mut [f32],
    flow_u: &[f32],
    flow_v: &[f32],
    a11: &[f32],
    a12: &[f32],
    a22: &[f32],
    b1: &[f32],
    b2: &[f32],
    w: usize,
    h: usize,
    alpha: f32,
) {
    // Safety: each row index maps to a non-overlapping slice of du_new/dv_new.
    // The raw pointers allow parallel mutable access to disjoint regions.
    let du_new_ptr = SendPtr(du_new.as_mut_ptr());
    let dv_new_ptr = SendPtr(dv_new.as_mut_ptr());

    (0..h).into_par_iter().for_each(|row| {
        // Safety: row*w..(row+1)*w is exclusive to this iteration.
        let du_new_local =
            unsafe { std::slice::from_raw_parts_mut(du_new_ptr.ptr().add(row * w), w) };
        let dv_new_local =
            unsafe { std::slice::from_raw_parts_mut(dv_new_ptr.ptr().add(row * w), w) };

        if row == 0 || row == h - 1 {
            // Border rows: scalar with boundary-aware neighbor count
            for col in 0..w {
                jacobi_pixel_scalar_to_row(
                    du,
                    dv,
                    du_new_local,
                    dv_new_local,
                    flow_u,
                    flow_v,
                    a11,
                    a12,
                    a22,
                    b1,
                    b2,
                    w,
                    h,
                    col,
                    row,
                    alpha,
                );
            }
        } else {
            // Interior row: left border pixel
            jacobi_pixel_scalar_to_row(
                du,
                dv,
                du_new_local,
                dv_new_local,
                flow_u,
                flow_v,
                a11,
                a12,
                a22,
                b1,
                b2,
                w,
                h,
                0,
                row,
                alpha,
            );

            // Interior columns — SIMD where possible
            #[cfg(target_arch = "x86_64")]
            {
                // Safety: interior row guarantees up/down neighbors exist,
                // and we process cols 1..w-1 only.
                unsafe {
                    jacobi_row_interior_sse2(
                        du,
                        dv,
                        du_new_local,
                        dv_new_local,
                        flow_u,
                        flow_v,
                        a11,
                        a12,
                        a22,
                        b1,
                        b2,
                        w,
                        row,
                        alpha,
                    );
                }
            }

            #[cfg(not(target_arch = "x86_64"))]
            {
                for col in 1..w - 1 {
                    jacobi_pixel_scalar_to_row(
                        du,
                        dv,
                        du_new_local,
                        dv_new_local,
                        flow_u,
                        flow_v,
                        a11,
                        a12,
                        a22,
                        b1,
                        b2,
                        w,
                        h,
                        col,
                        row,
                        alpha,
                    );
                }
            }

            // Right border pixel
            jacobi_pixel_scalar_to_row(
                du,
                dv,
                du_new_local,
                dv_new_local,
                flow_u,
                flow_v,
                a11,
                a12,
                a22,
                b1,
                b2,
                w,
                h,
                w - 1,
                row,
                alpha,
            );
        }
    });
}

/// Scalar Jacobi update for a single pixel, writing to row-local output slices.
///
/// `du_row`/`dv_row` are the row-local output slices (length `w`), indexed by `col`.
/// All other arrays are global (length `w*h`), indexed by `row*w + col`.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn jacobi_pixel_scalar_to_row(
    du: &[f32],
    dv: &[f32],
    du_row: &mut [f32],
    dv_row: &mut [f32],
    flow_u: &[f32],
    flow_v: &[f32],
    a11: &[f32],
    a12: &[f32],
    a22: &[f32],
    b1: &[f32],
    b2: &[f32],
    w: usize,
    h: usize,
    col: usize,
    row: usize,
    alpha: f32,
) {
    let idx = row * w + col;
    let cu = flow_u[idx];
    let cv = flow_v[idx];

    let mut lap_u = 0.0f32;
    let mut lap_v = 0.0f32;
    let mut nn = 0u32;

    if col > 0 {
        let ni = idx - 1;
        lap_u += (flow_u[ni] + du[ni]) - cu;
        lap_v += (flow_v[ni] + dv[ni]) - cv;
        nn += 1;
    }
    if col + 1 < w {
        let ni = idx + 1;
        lap_u += (flow_u[ni] + du[ni]) - cu;
        lap_v += (flow_v[ni] + dv[ni]) - cv;
        nn += 1;
    }
    if row > 0 {
        let ni = idx - w;
        lap_u += (flow_u[ni] + du[ni]) - cu;
        lap_v += (flow_v[ni] + dv[ni]) - cv;
        nn += 1;
    }
    if row + 1 < h {
        let ni = idx + w;
        lap_u += (flow_u[ni] + du[ni]) - cu;
        lap_v += (flow_v[ni] + dv[ni]) - cv;
        nn += 1;
    }

    let diag_u = a11[idx] + alpha * nn as f32;
    let diag_v = a22[idx] + alpha * nn as f32;

    du_row[col] = if diag_u.abs() > 1e-10 {
        (b1[idx] + alpha * lap_u - a12[idx] * dv[idx]) / diag_u
    } else {
        du[idx]
    };

    dv_row[col] = if diag_v.abs() > 1e-10 {
        (b2[idx] + alpha * lap_v - a12[idx] * du[idx]) / diag_v
    } else {
        dv[idx]
    };
}

/// SSE2-vectorized Jacobi update for interior columns of a single row.
///
/// Processes columns 1..w-1 in chunks of 4. For interior pixels all 4 neighbors
/// exist, so num_neighbors=4 for every pixel. The split FlowField layout makes
/// all component loads contiguous.
///
/// `du_row`/`dv_row` are the row-local output slices (length `w`), indexed by column.
/// All other arrays are global, indexed by `row*w + col`.
///
/// # Safety
/// Caller must ensure `row` is an interior row (0 < row < h-1) and `w >= 6`
/// (so interior columns exist for SIMD processing).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
#[allow(clippy::too_many_arguments)]
unsafe fn jacobi_row_interior_sse2(
    du: &[f32],
    dv: &[f32],
    du_row: &mut [f32],
    dv_row: &mut [f32],
    flow_u: &[f32],
    flow_v: &[f32],
    a11: &[f32],
    a12: &[f32],
    a22: &[f32],
    b1: &[f32],
    b2: &[f32],
    w: usize,
    row: usize,
    alpha: f32,
) {
    let alpha_v = _mm_set1_ps(alpha);
    let alpha4_v = _mm_set1_ps(alpha * 4.0);
    let eps_v = _mm_set1_ps(1e-10);

    let base = row * w;
    let mut col = 1usize;

    // Process 4 columns at a time
    while col + 4 <= w - 1 {
        let idx = base + col;

        // Load flow_u neighbors for Laplacian
        let fu_c = _mm_loadu_ps(flow_u.as_ptr().add(idx));
        let fu_l = _mm_loadu_ps(flow_u.as_ptr().add(idx - 1));
        let fu_r = _mm_loadu_ps(flow_u.as_ptr().add(idx + 1));
        let fu_u = _mm_loadu_ps(flow_u.as_ptr().add(idx - w));
        let fu_d = _mm_loadu_ps(flow_u.as_ptr().add(idx + w));

        // Load du neighbors
        let du_c = _mm_loadu_ps(du.as_ptr().add(idx));
        let du_l = _mm_loadu_ps(du.as_ptr().add(idx - 1));
        let du_r = _mm_loadu_ps(du.as_ptr().add(idx + 1));
        let du_u = _mm_loadu_ps(du.as_ptr().add(idx - w));
        let du_d = _mm_loadu_ps(du.as_ptr().add(idx + w));

        // lap_u = sum((flow_u[n] + du[n]) - flow_u[c]) for 4 neighbors
        // = (fu_l + du_l + fu_r + du_r + fu_u + du_u + fu_d + du_d) - 4*fu_c
        let sum_fu_n = _mm_add_ps(_mm_add_ps(fu_l, fu_r), _mm_add_ps(fu_u, fu_d));
        let sum_du_n = _mm_add_ps(_mm_add_ps(du_l, du_r), _mm_add_ps(du_u, du_d));
        let four_fu_c = _mm_mul_ps(_mm_set1_ps(4.0), fu_c);
        let lap_u = _mm_sub_ps(_mm_add_ps(sum_fu_n, sum_du_n), four_fu_c);

        // Same for v component
        let fv_c = _mm_loadu_ps(flow_v.as_ptr().add(idx));
        let fv_l = _mm_loadu_ps(flow_v.as_ptr().add(idx - 1));
        let fv_r = _mm_loadu_ps(flow_v.as_ptr().add(idx + 1));
        let fv_u = _mm_loadu_ps(flow_v.as_ptr().add(idx - w));
        let fv_d = _mm_loadu_ps(flow_v.as_ptr().add(idx + w));

        let dv_c = _mm_loadu_ps(dv.as_ptr().add(idx));
        let dv_l = _mm_loadu_ps(dv.as_ptr().add(idx - 1));
        let dv_r = _mm_loadu_ps(dv.as_ptr().add(idx + 1));
        let dv_u = _mm_loadu_ps(dv.as_ptr().add(idx - w));
        let dv_d = _mm_loadu_ps(dv.as_ptr().add(idx + w));

        let sum_fv_n = _mm_add_ps(_mm_add_ps(fv_l, fv_r), _mm_add_ps(fv_u, fv_d));
        let sum_dv_n = _mm_add_ps(_mm_add_ps(dv_l, dv_r), _mm_add_ps(dv_u, dv_d));
        let four_fv_c = _mm_mul_ps(_mm_set1_ps(4.0), fv_c);
        let lap_v = _mm_sub_ps(_mm_add_ps(sum_fv_n, sum_dv_n), four_fv_c);

        // Load precomputed coefficients
        let a11_v = _mm_loadu_ps(a11.as_ptr().add(idx));
        let a12_v = _mm_loadu_ps(a12.as_ptr().add(idx));
        let a22_v = _mm_loadu_ps(a22.as_ptr().add(idx));
        let b1_v = _mm_loadu_ps(b1.as_ptr().add(idx));
        let b2_v = _mm_loadu_ps(b2.as_ptr().add(idx));

        // diag = a + alpha * 4 (interior pixels always have 4 neighbors)
        let diag_u = _mm_add_ps(a11_v, alpha4_v);
        let diag_v = _mm_add_ps(a22_v, alpha4_v);

        // rhs_u = b1 + alpha * lap_u - a12 * dv
        let rhs_u = _mm_sub_ps(
            _mm_add_ps(b1_v, _mm_mul_ps(alpha_v, lap_u)),
            _mm_mul_ps(a12_v, dv_c),
        );

        // rhs_v = b2 + alpha * lap_v - a12 * du
        let rhs_v = _mm_sub_ps(
            _mm_add_ps(b2_v, _mm_mul_ps(alpha_v, lap_v)),
            _mm_mul_ps(a12_v, du_c),
        );

        // du_new = (|diag_u| > eps) ? rhs_u / diag_u : du
        // Use a mask to select between division result and old value
        let abs_diag_u = _mm_andnot_ps(_mm_set1_ps(-0.0), diag_u); // abs via clear sign bit
        let mask_u = _mm_cmpgt_ps(abs_diag_u, eps_v);
        // Safe division: replace zero diags with 1.0 to avoid NaN, then mask
        let safe_diag_u = _mm_or_ps(
            _mm_and_ps(mask_u, diag_u),
            _mm_andnot_ps(mask_u, _mm_set1_ps(1.0)),
        );
        let div_u = _mm_div_ps(rhs_u, safe_diag_u);
        let result_u = _mm_or_ps(_mm_and_ps(mask_u, div_u), _mm_andnot_ps(mask_u, du_c));
        _mm_storeu_ps(du_row.as_mut_ptr().add(col), result_u);

        // Same for v
        let abs_diag_v = _mm_andnot_ps(_mm_set1_ps(-0.0), diag_v);
        let mask_v = _mm_cmpgt_ps(abs_diag_v, eps_v);
        let safe_diag_v = _mm_or_ps(
            _mm_and_ps(mask_v, diag_v),
            _mm_andnot_ps(mask_v, _mm_set1_ps(1.0)),
        );
        let div_v = _mm_div_ps(rhs_v, safe_diag_v);
        let result_v = _mm_or_ps(_mm_and_ps(mask_v, div_v), _mm_andnot_ps(mask_v, dv_c));
        _mm_storeu_ps(dv_row.as_mut_ptr().add(col), result_v);

        col += 4;
    }

    // Scalar tail for remaining interior columns
    while col < w - 1 {
        let idx = base + col;
        let cu = flow_u[idx];
        let cv = flow_v[idx];

        let lap_u = (flow_u[idx - 1] + du[idx - 1])
            + (flow_u[idx + 1] + du[idx + 1])
            + (flow_u[idx - w] + du[idx - w])
            + (flow_u[idx + w] + du[idx + w])
            - 4.0 * cu;
        let lap_v = (flow_v[idx - 1] + dv[idx - 1])
            + (flow_v[idx + 1] + dv[idx + 1])
            + (flow_v[idx - w] + dv[idx - w])
            + (flow_v[idx + w] + dv[idx + w])
            - 4.0 * cv;

        let diag_u = a11[idx] + alpha * 4.0;
        let diag_v = a22[idx] + alpha * 4.0;

        du_row[col] = if diag_u.abs() > 1e-10 {
            (b1[idx] + alpha * lap_u - a12[idx] * dv[idx]) / diag_u
        } else {
            du[idx]
        };
        dv_row[col] = if diag_v.abs() > 1e-10 {
            (b2[idx] + alpha * lap_v - a12[idx] * du[idx]) / diag_v
        } else {
            dv[idx]
        };

        col += 1;
    }
}

/// Compute image gradients using central differences.
///
/// gx = (right - left) * 0.5, gy = (down - up) * 0.5.
/// Border pixels use replicated (clamped) boundary conditions.
fn compute_image_gradients(img: &GrayImage) -> (GrayImage, GrayImage) {
    let w = img.width() as usize;
    let h = img.height() as usize;
    let data = img.data();
    let mut gx = vec![0.0f32; w * h];
    let mut gy = vec![0.0f32; w * h];

    // Top row (row=0): up is clamped to row 0
    gradient_row_scalar(data, &mut gx, &mut gy, w, 0, 0, 1);

    // Interior rows — SIMD for interior columns
    for row in 1..h - 1 {
        let base = row * w;
        let up_row = base - w;
        let dn_row = base + w;

        // Left border pixel (col=0): left clamped to col 0
        gx[base] = (data[base + 1] - data[base]) * 0.5;
        gy[base] = (data[dn_row] - data[up_row]) * 0.5;

        // Interior columns — SIMD
        #[cfg(target_arch = "x86_64")]
        {
            // Safety: interior row/col guarantees all offsets are in bounds.
            unsafe {
                gradient_row_interior_sse2(data, &mut gx, &mut gy, w, row);
            }
        }

        #[cfg(not(target_arch = "x86_64"))]
        {
            for col in 1..w - 1 {
                let idx = base + col;
                gx[idx] = (data[idx + 1] - data[idx - 1]) * 0.5;
                gy[idx] = (data[idx + w] - data[idx - w]) * 0.5;
            }
        }

        // Right border pixel (col=w-1): right clamped to col w-1
        let idx = base + w - 1;
        gx[idx] = (data[idx] - data[idx - 1]) * 0.5;
        gy[idx] = (data[dn_row + w - 1] - data[up_row + w - 1]) * 0.5;
    }

    // Bottom row (row=h-1): down is clamped to row h-1
    if h > 1 {
        gradient_row_scalar(data, &mut gx, &mut gy, w, h - 1, h - 2, h - 1);
    }

    (
        GrayImage::new(w as u32, h as u32, gx),
        GrayImage::new(w as u32, h as u32, gy),
    )
}

/// Scalar gradient for a full row. `up_row` and `dn_row` are the row indices
/// for the up/down neighbors (clamped at boundaries).
fn gradient_row_scalar(
    data: &[f32],
    gx: &mut [f32],
    gy: &mut [f32],
    w: usize,
    row: usize,
    up_row: usize,
    dn_row: usize,
) {
    let base = row * w;
    let up_base = up_row * w;
    let dn_base = dn_row * w;

    // col=0: left clamped
    gx[base] = (data[base + 1] - data[base]) * 0.5;
    gy[base] = (data[dn_base] - data[up_base]) * 0.5;

    for col in 1..w - 1 {
        let idx = base + col;
        gx[idx] = (data[idx + 1] - data[idx - 1]) * 0.5;
        gy[idx] = (data[dn_base + col] - data[up_base + col]) * 0.5;
    }

    // col=w-1: right clamped
    if w > 1 {
        let idx = base + w - 1;
        gx[idx] = (data[idx] - data[idx - 1]) * 0.5;
        gy[idx] = (data[dn_base + w - 1] - data[up_base + w - 1]) * 0.5;
    }
}

/// SSE2-vectorized gradient for interior columns of an interior row.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn gradient_row_interior_sse2(
    data: &[f32],
    gx: &mut [f32],
    gy: &mut [f32],
    w: usize,
    row: usize,
) {
    let half = _mm_set1_ps(0.5);
    let base = row * w;
    let mut col = 1usize;

    while col + 4 <= w - 1 {
        let idx = base + col;

        // gx = (right - left) * 0.5
        let right = _mm_loadu_ps(data.as_ptr().add(idx + 1));
        let left = _mm_loadu_ps(data.as_ptr().add(idx - 1));
        let gx_v = _mm_mul_ps(_mm_sub_ps(right, left), half);
        _mm_storeu_ps(gx.as_mut_ptr().add(idx), gx_v);

        // gy = (down - up) * 0.5
        let down = _mm_loadu_ps(data.as_ptr().add(idx + w));
        let up = _mm_loadu_ps(data.as_ptr().add(idx - w));
        let gy_v = _mm_mul_ps(_mm_sub_ps(down, up), half);
        _mm_storeu_ps(gy.as_mut_ptr().add(idx), gy_v);

        col += 4;
    }

    // Scalar tail
    while col < w - 1 {
        let idx = base + col;
        gx[idx] = (data[idx + 1] - data[idx - 1]) * 0.5;
        gy[idx] = (data[idx + w] - data[idx - w]) * 0.5;
        col += 1;
    }
}

/// Warp an image by the current flow field.
fn warp_by_flow(img: &GrayImage, flow: &FlowField) -> GrayImage {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_variational_refine_doesnt_crash() {
        let img = GrayImage::checkerboard(16, 16);
        let mut flow = FlowField::new(16, 16);
        let params = VariationalParams {
            delta: 5.0,
            gamma: 10.0,
            alpha: 10.0,
            jacobi_iterations: 3,
            outer_iterations: 1,
        };
        variational_refine(&img, &img, &mut flow, &params);

        // Flow should remain near zero for identical images
        let mut max_flow = 0.0f32;
        for row in 0..16 {
            for col in 0..16 {
                let (dx, dy) = flow.get(col, row);
                max_flow = max_flow.max(dx.abs()).max(dy.abs());
            }
        }
        assert!(
            max_flow < 0.5,
            "Expected near-zero flow, got max {}",
            max_flow
        );
    }

    #[test]
    fn test_psi_deriv() {
        // psi_deriv(0) = 1 / (2 * sqrt(eps^2)) = 1 / (2 * eps) = 500
        let val = psi_deriv(0.0);
        assert!(val > 400.0 && val < 600.0, "psi_deriv(0) = {}", val);

        // For large values, psi_deriv should be small
        let val_large = psi_deriv(100.0);
        assert!(val_large < 0.1, "psi_deriv(100) = {}", val_large);
    }

    #[test]
    fn test_variational_small_image_noop() {
        // Images smaller than 3x3 should be skipped
        let img = GrayImage::new_constant(2, 2, 0.5);
        let mut flow = FlowField::new(2, 2);
        let params = VariationalParams {
            delta: 5.0,
            gamma: 10.0,
            alpha: 10.0,
            jacobi_iterations: 5,
            outer_iterations: 1,
        };
        variational_refine(&img, &img, &mut flow, &params);
        // Should not crash, flow should remain zero
        let (dx, dy) = flow.get(0, 0);
        assert!((dx).abs() < 1e-6);
        assert!((dy).abs() < 1e-6);
    }
}