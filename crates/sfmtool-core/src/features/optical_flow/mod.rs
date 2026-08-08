// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Dense optical flow computation using the Dense Inverse Search (DIS) algorithm.
//!
//! Reference: Kroeger et al., "Fast Optical Flow using Dense Inverse Search," ECCV 2016.
//!
//! # Example
//! ```
//! use sfmtool_core::features::optical_flow::{compute_optical_flow, DisFlowParams, GrayImage};
//!
//! let img_a = GrayImage::new_constant(64, 64, 0.5);
//! let img_b = GrayImage::new_constant(64, 64, 0.5);
//! let params = DisFlowParams::default_quality();
//! let flow = compute_optical_flow(&img_a, &img_b, &params, None);
//! assert_eq!(flow.width(), img_a.width());
//! ```
//!
//! # Flow composition and initialization
//! ```
//! use sfmtool_core::features::optical_flow::{
//!     compute_optical_flow, compute_optical_flow_with_init, compose_flow,
//!     DisFlowParams, GrayImage,
//! };
//!
//! let img_a = GrayImage::new_constant(64, 64, 0.5);
//! let img_b = GrayImage::new_constant(64, 64, 0.5);
//! let img_c = GrayImage::new_constant(64, 64, 0.5);
//! let params = DisFlowParams::fast();
//!
//! let flow_ab = compute_optical_flow(&img_a, &img_b, &params, None);
//! let flow_bc = compute_optical_flow(&img_b, &img_c, &params, None);
//! let chained = compose_flow(&flow_ab, &flow_bc);
//! let refined = compute_optical_flow_with_init(&img_a, &img_c, &params, &chained, None);
//! assert_eq!(refined.width(), img_a.width());
//! ```

mod dis;
mod flow_field;
#[cfg(feature = "gpu")]
pub mod gpu;
mod image;
mod interp;
mod params;
mod pyramid;
mod variational;

// When the gpu feature is off, provide a stub type so the public API is consistent.
#[cfg(not(feature = "gpu"))]
pub mod gpu {
    /// Stub type when GPU support is not compiled in. Cannot be constructed.
    pub struct GpuFlowContext(());
}

// The container types live in their own modules but keep their original
// paths: `optical_flow::GrayImage` and friends are imported crate-wide and
// through the PyO3 bindings.
pub use flow_field::{FlowField, FlowFieldRef};
pub use image::GrayImage;
pub use params::{DisFlowParams, FlowTiming};

use pyramid::ImagePyramid;

/// Internals exposed for benchmarking only. Not part of the public API.
#[doc(hidden)]
pub mod bench {
    pub use super::interp::sample_bilinear;
    pub use super::pyramid::ImagePyramid;

    use super::{DisFlowParams, FlowField, GrayImage};

    /// Build a Gaussian pyramid (for benchmarking pyramid construction).
    pub fn build_pyramid(img: &GrayImage, num_levels: u32) -> Vec<(u32, u32)> {
        let pyr = ImagePyramid::build(img, num_levels);
        (0..num_levels as usize)
            .map(|i| (pyr.level(i).width(), pyr.level(i).height()))
            .collect()
    }

    /// Run a single level of DIS refinement (for benchmarking the core loop).
    pub fn refine_flow_at_level(
        ref_image: &GrayImage,
        tgt_image: &GrayImage,
        flow: &mut FlowField,
        params: &DisFlowParams,
        scale_index: u32,
    ) {
        super::dis::refine_flow_at_level(ref_image, tgt_image, flow, params, scale_index, None);
    }

    /// Run variational refinement (for benchmarking).
    pub fn variational_refine(
        ref_image: &GrayImage,
        tgt_image: &GrayImage,
        flow: &mut FlowField,
        params: &DisFlowParams,
    ) {
        let var_params = super::variational::VariationalParams {
            delta: params.variational_delta,
            gamma: params.variational_gamma,
            alpha: params.variational_alpha,
            jacobi_iterations: params.variational_jacobi_iterations,
            outer_iterations: params.variational_outer_iterations_base,
        };
        super::variational::variational_refine(ref_image, tgt_image, flow, &var_params);
    }
}

/// Compute dense optical flow from image A to image B.
///
/// Input images are single-channel grayscale f32 normalized to [0, 1].
/// Returns a FlowField the same size as the input images.
///
/// Pass a [`gpu::GpuFlowContext`] to run variational refinement on the GPU,
/// or `None` for the CPU path.
pub fn compute_optical_flow(
    img_a: &GrayImage,
    img_b: &GrayImage,
    params: &DisFlowParams,
    gpu: Option<&gpu::GpuFlowContext>,
) -> FlowField {
    compute_optical_flow_timed(img_a, img_b, params, gpu).0
}

/// Compute dense optical flow with per-stage timing breakdown.
///
/// Returns `(flow, timing)` where `timing` contains precise measurements
/// of each pipeline stage.
pub fn compute_optical_flow_timed(
    img_a: &GrayImage,
    img_b: &GrayImage,
    params: &DisFlowParams,
    gpu: Option<&gpu::GpuFlowContext>,
) -> (FlowField, FlowTiming) {
    use std::time::Instant;

    assert_eq!(img_a.width(), img_b.width());
    assert_eq!(img_a.height(), img_b.height());

    let t_total_start = Instant::now();
    let mut timing = FlowTiming::default();

    let width = img_a.width();

    // Determine pyramid levels
    let coarsest = params
        .coarsest_scale
        .unwrap_or_else(|| params.compute_coarsest_scale(width));
    let finest = params
        .finest_scale
        .unwrap_or_else(|| coarsest.saturating_sub(2));

    // Ensure we have at least one level to process
    let coarsest = coarsest.max(finest);

    // Build pyramids (need coarsest + 1 levels, indexed 0..=coarsest)
    let num_levels = coarsest + 1;

    // Compute pyramid level dimensions (w/2, h/2 at each level).
    let level_dims: Vec<(u32, u32)> = {
        let mut dims = Vec::with_capacity(num_levels as usize);
        let mut w = img_a.width();
        let mut h = img_a.height();
        for _ in 0..num_levels {
            dims.push((w, h));
            w /= 2;
            h /= 2;
        }
        dims
    };

    timing.levels_processed = coarsest - finest + 1;

    // Determine whether to use the multi-level GPU path (keeps flow on GPU between
    // levels, eliminating per-level CPU↔GPU flow transfers and CPU upsample cost).
    #[cfg(feature = "gpu")]
    let gpu_start_scale: Option<u32> = if gpu.is_some() && params.variational_refinement {
        // Find the first GPU-eligible scale in coarse-to-fine processing order.
        (finest..=coarsest).rev().find(|&s| {
            let (w, h) = level_dims[s as usize];
            (w * h) >= params.gpu_min_pixels
        })
    } else {
        None
    };
    #[cfg(not(feature = "gpu"))]
    let gpu_start_scale: Option<u32> = None;

    // Track whether the GPU path handled the final upsample to full resolution.
    let mut gpu_did_final_upsample = false;

    let mut flow = if let Some(gpu_start) = gpu_start_scale {
        #[cfg(feature = "gpu")]
        {
            let gpu_ctx = gpu.unwrap();
            let has_cpu_levels = gpu_start < coarsest;

            // Phase 0: Build GPU pyramid and optionally read back a seed level
            // for the CPU pyramid. This avoids building the expensive fine levels
            // on CPU when the GPU already builds them.
            let seed_level = if has_cpu_levels {
                let (w, h) = level_dims[gpu_start as usize + 1];
                Some((gpu_start as usize + 1, w, h))
            } else {
                None
            };

            let t_pyr = Instant::now();
            let (gpu_pyr_pool, seed_images) =
                gpu_ctx.build_gpu_pyramid(img_a, img_b, num_levels as usize, seed_level);

            // Build CPU pyramid from seed (only coarse levels beyond the seed).
            let (pyr_a, pyr_b) = if let Some((ref_seed, tgt_seed)) = seed_images {
                let seed_idx = gpu_start + 1;
                let additional = coarsest - seed_idx;
                let pyr_a = ImagePyramid::build_from_level(&ref_seed, seed_idx, additional);
                let pyr_b = ImagePyramid::build_from_level(&tgt_seed, seed_idx, additional);
                (pyr_a, pyr_b)
            } else {
                // All levels are GPU — build minimal CPU pyramids (unused).
                (ImagePyramid::build(img_a, 1), ImagePyramid::build(img_b, 1))
            };
            timing.pyramid_build = t_pyr.elapsed().as_secs_f64();

            // Initialize flow at the coarsest level.
            let (coarsest_w, coarsest_h) = level_dims[coarsest as usize];
            let mut flow = FlowField::new(coarsest_w, coarsest_h);

            // Phase 1: CPU levels (coarsest down to gpu_start + 1).
            for scale in ((gpu_start + 1)..=coarsest).rev() {
                let ref_img = pyr_a.level(scale as usize);
                let tgt_img = pyr_b.level(scale as usize);

                if scale < coarsest {
                    let t_up = Instant::now();
                    flow = flow.upsample_2x();
                    flow = resize_flow_to(flow, ref_img.width(), ref_img.height());
                    timing.upsample_total += t_up.elapsed().as_secs_f64();
                }

                let level_timing =
                    dis::refine_flow_at_level(ref_img, tgt_img, &mut flow, params, scale, None);
                timing.dis_total += level_timing.dis_secs;
                timing.variational_total += level_timing.var_secs;
                timing.per_level.push((
                    scale,
                    ref_img.width(),
                    ref_img.height(),
                    level_timing.dis_secs,
                    level_timing.var_secs,
                ));
            }

            // Phase 2: Transition upsample from last CPU level to first GPU level.
            if has_cpu_levels {
                let (gpu_start_w, gpu_start_h) = level_dims[gpu_start as usize];
                let t_up = Instant::now();
                flow = flow.upsample_2x();
                flow = resize_flow_to(flow, gpu_start_w, gpu_start_h);
                timing.upsample_total += t_up.elapsed().as_secs_f64();
            }

            // Phase 3: GPU levels using pre-built GPU pyramid.
            let gpu_scales: Vec<(u32, u32, u32)> = (finest..=gpu_start)
                .rev()
                .map(|s| {
                    let (w, h) = level_dims[s as usize];
                    (s, w, h)
                })
                .collect();

            let t_gpu = Instant::now();
            flow =
                gpu_ctx.run_gpu_levels_prebuilt(&gpu_pyr_pool, &gpu_scales, flow, params, finest);
            let gpu_elapsed = t_gpu.elapsed().as_secs_f64();

            if finest > 0 {
                gpu_did_final_upsample = true;
            }

            // Attribute total GPU time to dis_total (can't separate DIS/variational/upsample).
            timing.dis_total += gpu_elapsed;
            for &(s, w, h) in &gpu_scales {
                timing.per_level.push((s, w, h, 0.0, 0.0));
            }

            flow
        }
        #[cfg(not(feature = "gpu"))]
        {
            unreachable!("gpu_start_scale is always None without gpu feature");
        }
    } else {
        // All-CPU path (original loop, also used when GPU is unavailable).
        let t_pyr = Instant::now();
        let pyr_a = ImagePyramid::build(img_a, num_levels);
        let pyr_b = ImagePyramid::build(img_b, num_levels);
        timing.pyramid_build = t_pyr.elapsed().as_secs_f64();

        let (coarsest_w, coarsest_h) = level_dims[coarsest as usize];
        let mut flow = FlowField::new(coarsest_w, coarsest_h);

        for scale in (finest..=coarsest).rev() {
            let ref_img = pyr_a.level(scale as usize);
            let tgt_img = pyr_b.level(scale as usize);

            if scale < coarsest {
                let t_up = Instant::now();
                flow = flow.upsample_2x();
                flow = resize_flow_to(flow, ref_img.width(), ref_img.height());
                timing.upsample_total += t_up.elapsed().as_secs_f64();
            }

            let level_timing =
                dis::refine_flow_at_level(ref_img, tgt_img, &mut flow, params, scale, gpu);
            timing.dis_total += level_timing.dis_secs;
            timing.variational_total += level_timing.var_secs;
            timing.per_level.push((
                scale,
                ref_img.width(),
                ref_img.height(),
                level_timing.dis_secs,
                level_timing.var_secs,
            ));
        }

        flow
    };

    // Upsample to full resolution if finest_scale > 0
    // (skip if GPU already did the final upsample)
    let t_final_up = Instant::now();
    if !gpu_did_final_upsample {
        for _ in 0..finest {
            flow = flow.upsample_2x();
        }
    }

    // Final resize to match input dimensions
    let flow = resize_flow_to(flow, img_a.width(), img_a.height());
    timing.upsample_total += t_final_up.elapsed().as_secs_f64();

    timing.total = t_total_start.elapsed().as_secs_f64();

    (flow, timing)
}

/// Compose two flow fields: result(x) = flow_ab(x) + flow_bc(x + flow_ab(x)).
///
/// The composed field maps points from image A to image C via B.
/// `flow_bc` is sampled at the advected position using bilinear interpolation.
/// Both fields must have the same dimensions.
///
/// Rows are processed in parallel via Rayon.
pub fn compose_flow(flow_ab: &FlowField, flow_bc: &FlowField) -> FlowField {
    compose_flow_ref(&flow_ab.as_ref(), &flow_bc.as_ref())
}

/// Compose two flow fields from borrowed views. See [`compose_flow`].
pub fn compose_flow_ref(flow_ab: &FlowFieldRef<'_>, flow_bc: &FlowFieldRef<'_>) -> FlowField {
    use rayon::prelude::*;

    assert_eq!(flow_ab.width(), flow_bc.width());
    assert_eq!(flow_ab.height(), flow_bc.height());

    let w = flow_ab.width();
    let h = flow_ab.height();
    let ws = w as usize;
    let mut result = FlowField::new(w, h);

    result
        .data_u
        .par_chunks_mut(ws)
        .zip(result.data_v.par_chunks_mut(ws))
        .enumerate()
        .for_each(|(row, (row_u, row_v))| {
            let row = row as u32;
            for col in 0..w {
                let (dx_ab, dy_ab) = flow_ab.get(col, row);
                let mid_x = col as f32 + 0.5 + dx_ab;
                let mid_y = row as f32 + 0.5 + dy_ab;
                let (dx_bc, dy_bc) = flow_bc.sample(mid_x, mid_y);
                row_u[col as usize] = dx_ab + dx_bc;
                row_v[col as usize] = dy_ab + dy_bc;
            }
        });

    result
}

/// Compute dense optical flow from image A to image B, starting from an initial
/// flow estimate instead of zero.
///
/// The initial flow is downsampled into the coarsest pyramid level, and the solver
/// refines from there. This is useful when a chained or approximate flow is available
/// as a starting point — the solver only needs to compute the residual correction.
///
/// Input images are single-channel grayscale f32 normalized to [0, 1].
/// The initial flow must have the same dimensions as the input images.
pub fn compute_optical_flow_with_init(
    img_a: &GrayImage,
    img_b: &GrayImage,
    params: &DisFlowParams,
    initial_flow: &FlowField,
    gpu: Option<&gpu::GpuFlowContext>,
) -> FlowField {
    assert_eq!(img_a.width(), img_b.width());
    assert_eq!(img_a.height(), img_b.height());
    assert_eq!(initial_flow.width(), img_a.width());
    assert_eq!(initial_flow.height(), img_a.height());

    let width = img_a.width();

    // Determine pyramid levels
    let coarsest = params
        .coarsest_scale
        .unwrap_or_else(|| params.compute_coarsest_scale(width));
    let finest = params
        .finest_scale
        .unwrap_or_else(|| coarsest.saturating_sub(2));

    let coarsest = coarsest.max(finest);

    // Build pyramids
    let num_levels = coarsest + 1;
    let pyr_a = ImagePyramid::build(img_a, num_levels);
    let pyr_b = ImagePyramid::build(img_b, num_levels);

    // Downsample initial flow to the coarsest level
    let mut flow = initial_flow.clone();
    for _ in 0..coarsest {
        flow = flow.downsample_2x();
    }
    // Resize to match the coarsest pyramid level exactly
    let coarsest_img = pyr_a.level(coarsest as usize);
    flow = resize_flow_to(flow, coarsest_img.width(), coarsest_img.height());

    // Process from coarsest to finest (same as compute_optical_flow)
    for scale in (finest..=coarsest).rev() {
        let ref_img = pyr_a.level(scale as usize);
        let tgt_img = pyr_b.level(scale as usize);

        if scale < coarsest {
            flow = flow.upsample_2x();
            flow = resize_flow_to(flow, ref_img.width(), ref_img.height());
        }

        dis::refine_flow_at_level(ref_img, tgt_img, &mut flow, params, scale, gpu);
    }

    // Upsample to full resolution if finest_scale > 0
    for _ in 0..finest {
        flow = flow.upsample_2x();
    }

    resize_flow_to(flow, img_a.width(), img_a.height())
}

/// Resize a flow field to target dimensions, cropping or zero-padding as needed.
/// Takes ownership to avoid cloning when dimensions already match.
fn resize_flow_to(flow: FlowField, target_w: u32, target_h: u32) -> FlowField {
    if flow.width() == target_w && flow.height() == target_h {
        return flow;
    }

    let mut result = FlowField::new(target_w, target_h);
    let copy_w = flow.width().min(target_w) as usize;
    let copy_h = flow.height().min(target_h) as usize;
    let src_w = flow.width() as usize;
    let dst_w = target_w as usize;

    for row in 0..copy_h {
        let src_start = row * src_w;
        let dst_start = row * dst_w;
        result.data_u[dst_start..dst_start + copy_w]
            .copy_from_slice(&flow.data_u[src_start..src_start + copy_w]);
        result.data_v[dst_start..dst_start + copy_w]
            .copy_from_slice(&flow.data_v[src_start..src_start + copy_w]);
    }

    result
}

#[cfg(test)]
mod tests;
