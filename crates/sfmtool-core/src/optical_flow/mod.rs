// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Dense optical flow computation using the Dense Inverse Search (DIS) algorithm.
//!
//! Reference: Kroeger et al., "Fast Optical Flow using Dense Inverse Search," ECCV 2016.
//!
//! # Example
//! ```
//! use sfmtool_core::optical_flow::{compute_optical_flow, DisFlowParams, GrayImage};
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
//! use sfmtool_core::optical_flow::{
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
#[cfg(feature = "gpu")]
pub mod gpu;
mod interp;
mod pyramid;
mod variational;

// When the gpu feature is off, provide a stub type so the public API is consistent.
#[cfg(not(feature = "gpu"))]
pub mod gpu {
    /// Stub type when GPU support is not compiled in. Cannot be constructed.
    pub struct GpuFlowContext(());
}

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
    data_u: Vec<f32>,
    /// Vertical displacements, row-major, length = width * height.
    data_v: Vec<f32>,
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

/// Simple wrapper for grayscale image data.
///
/// Coordinate convention: pixel centers are at (col + 0.5, row + 0.5), matching the
/// .sfmr/.sift format convention (COLMAP convention). Sampling at (0.5, 0.5)
/// returns the exact value of the top-left pixel.
pub struct GrayImage {
    width: u32,
    height: u32,
    /// Pixel data normalized to [0, 1], row-major.
    data: Vec<f32>,
}

impl GrayImage {
    /// Create a new image from raw f32 data (must be width * height elements).
    pub fn new(width: u32, height: u32, data: Vec<f32>) -> Self {
        assert_eq!(data.len(), (width as usize) * (height as usize));
        Self {
            width,
            height,
            data,
        }
    }

    /// Create a constant-valued image.
    pub fn new_constant(width: u32, height: u32, value: f32) -> Self {
        Self {
            width,
            height,
            data: vec![value; (width as usize) * (height as usize)],
        }
    }

    /// Create from u8 data, normalizing to [0, 1].
    pub fn from_u8(width: u32, height: u32, data: &[u8]) -> Self {
        assert_eq!(data.len(), (width as usize) * (height as usize));
        Self {
            width,
            height,
            data: data.iter().map(|&v| v as f32 / 255.0).collect(),
        }
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    /// Get pixel value at grid position (col, row).
    pub fn get_pixel(&self, col: u32, row: u32) -> f32 {
        self.data[(row as usize) * (self.width as usize) + (col as usize)]
    }

    /// Set pixel value at grid position (col, row).
    pub fn set_pixel(&mut self, col: u32, row: u32, value: f32) {
        self.data[(row as usize) * (self.width as usize) + (col as usize)] = value;
    }

    /// Access raw data as a slice.
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    /// Access raw data as a mutable slice.
    pub fn data_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }

    /// Create a synthetic checkerboard pattern for testing.
    pub fn checkerboard(width: u32, height: u32) -> Self {
        let mut data = vec![0.0f32; (width as usize) * (height as usize)];
        for row in 0..height {
            for col in 0..width {
                let checker = ((col / 8) + (row / 8)) % 2;
                data[(row as usize) * (width as usize) + (col as usize)] =
                    if checker == 0 { 0.2 } else { 0.8 };
            }
        }
        Self {
            width,
            height,
            data,
        }
    }

    /// Create an image shifted by integer pixels from a source image.
    /// Pixels that fall outside the source are filled with 0.
    pub fn shifted(src: &GrayImage, shift_x: f32, shift_y: f32) -> Self {
        let w = src.width;
        let h = src.height;
        let mut data = vec![0.0f32; (w as usize) * (h as usize)];
        for row in 0..h {
            for col in 0..w {
                let sx = col as f32 + 0.5 - shift_x;
                let sy = row as f32 + 0.5 - shift_y;
                data[(row as usize) * (w as usize) + (col as usize)] =
                    interp::sample_bilinear(src, sx, sy);
            }
        }
        Self {
            width: w,
            height: h,
            data,
        }
    }
}

/// Algorithm parameters for DIS optical flow.
///
/// Notation follows Kroeger et al., "Fast Optical Flow using Dense Inverse Search," ECCV 2016.
pub struct DisFlowParams {
    /// Patch size (square edge length in pixels). Default: 8.
    pub patch_size: u32,
    /// Patch overlap fraction (0.0-1.0). Default: 0.4.
    pub patch_overlap: f32,
    /// Gradient descent iterations per patch. Default: 12.
    pub grad_descent_iterations: u32,
    /// Finest pyramid level (0 = full res). If None, computed as coarsest_scale - 2.
    pub finest_scale: Option<u32>,
    /// Coarsest pyramid level. If None, auto-computed from image width.
    pub coarsest_scale: Option<u32>,
    /// Enable variational refinement. Default: true.
    pub variational_refinement: bool,
    /// Variational refinement inner Jacobi solver iterations. Default: 7.
    ///
    /// Not directly comparable to the SOR iterations (θ_vi) in Kroeger et al.
    /// or OpenCV's DIS. Jacobi converges slower per iteration than Gauss-Seidel
    /// SOR but each iteration is fully parallelizable (SIMD, multi-core).
    /// Roughly 4/3× the SOR count gives equivalent convergence.
    pub variational_jacobi_iterations: u32,
    /// Variational outer iterations base multiplier. Default: 1.
    pub variational_outer_iterations_base: u32,
    /// Variational smoothness weight (alpha). Default: 10.
    pub variational_alpha: f32,
    /// Variational gradient weight (gamma). Default: 10.
    pub variational_gamma: f32,
    /// Variational intensity weight (delta). Default: 5.
    pub variational_delta: f32,
    /// Mean-normalize patches before matching. Default: true.
    pub normalize_patches: bool,
    /// Minimum pixel count (width × height) for GPU dispatch at a given pyramid level.
    /// Levels smaller than this fall back to CPU, avoiding GPU dispatch overhead on
    /// tiny images where CPU is faster. Default: 50000 (~224×224).
    /// Set to 0 to always use GPU when available.
    pub gpu_min_pixels: u32,
}

impl DisFlowParams {
    /// Operating Point 2 (recommended default).
    pub fn default_quality() -> Self {
        Self {
            patch_size: 8,
            patch_overlap: 0.4,
            grad_descent_iterations: 12,
            finest_scale: None,
            coarsest_scale: None,
            variational_refinement: true,
            variational_jacobi_iterations: 7,
            variational_outer_iterations_base: 1,
            variational_alpha: 10.0,
            variational_gamma: 10.0,
            variational_delta: 5.0,
            normalize_patches: true,
            gpu_min_pixels: 50_000,
        }
    }

    /// Operating Point 1 (fast, 600Hz): no variational refinement.
    pub fn fast() -> Self {
        Self {
            patch_size: 8,
            patch_overlap: 0.3,
            grad_descent_iterations: 16,
            finest_scale: Some(3),
            coarsest_scale: None,
            variational_refinement: false,
            variational_jacobi_iterations: 7,
            variational_outer_iterations_base: 1,
            variational_alpha: 10.0,
            variational_gamma: 10.0,
            variational_delta: 5.0,
            normalize_patches: true,
            gpu_min_pixels: 50_000,
        }
    }

    /// Operating Point 3 (high quality, 10Hz): larger patches, more overlap.
    pub fn high_quality() -> Self {
        Self {
            patch_size: 12,
            patch_overlap: 0.75,
            grad_descent_iterations: 16,
            finest_scale: Some(1),
            coarsest_scale: None,
            variational_refinement: true,
            variational_jacobi_iterations: 7,
            variational_outer_iterations_base: 1,
            variational_alpha: 10.0,
            variational_gamma: 10.0,
            variational_delta: 5.0,
            normalize_patches: true,
            gpu_min_pixels: 50_000,
        }
    }

    /// Compute the patch stride from patch_size and patch_overlap.
    fn patch_stride(&self) -> u32 {
        (self.patch_size as f32 * (1.0 - self.patch_overlap))
            .floor()
            .max(1.0) as u32
    }

    /// Auto-compute coarsest scale from image width.
    fn compute_coarsest_scale(&self, width: u32) -> u32 {
        let f = 5.0;
        let val = 2.0 * width as f64 / (f * self.patch_size as f64);
        val.log2().floor() as u32
    }
}

/// Per-stage timing breakdown for optical flow computation.
///
/// All times are in seconds. Only populated when
/// [`compute_optical_flow_timed`] is used.
#[derive(Clone, Debug, Default)]
pub struct FlowTiming {
    /// Time to build both Gaussian pyramids.
    pub pyramid_build: f64,
    /// Total time spent in DIS inverse search + densification (all levels).
    pub dis_total: f64,
    /// Total time spent in variational refinement (all levels).
    pub variational_total: f64,
    /// Total time spent upsampling flow between levels.
    pub upsample_total: f64,
    /// Total wall-clock time (should ≈ sum of above + overhead).
    pub total: f64,
    /// Number of pyramid levels processed.
    pub levels_processed: u32,
    /// Per-level breakdown: (scale, width, height, dis_time, variational_time).
    pub per_level: Vec<(u32, u32, u32, f64, f64)>,
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
mod tests {
    use super::*;

    #[test]
    fn test_flow_field_new() {
        let flow = FlowField::new(10, 20);
        assert_eq!(flow.width(), 10);
        assert_eq!(flow.height(), 20);
        assert_eq!(flow.u_slice().len(), 10 * 20);
        assert_eq!(flow.v_slice().len(), 10 * 20);
    }

    #[test]
    fn test_flow_field_get_set() {
        let mut flow = FlowField::new(10, 10);
        flow.set(3, 5, 1.5, -2.5);
        let (dx, dy) = flow.get(3, 5);
        assert!((dx - 1.5).abs() < 1e-6);
        assert!((dy - (-2.5)).abs() < 1e-6);
    }

    #[test]
    fn test_flow_field_sample_at_pixel_center() {
        let mut flow = FlowField::new(4, 4);
        flow.set(0, 0, 3.0, 4.0);
        let (dx, dy) = flow.sample(0.5, 0.5);
        assert!((dx - 3.0).abs() < 1e-6);
        assert!((dy - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_advect_points_constant_flow() {
        let mut flow = FlowField::new(10, 10);
        for row in 0..10 {
            for col in 0..10 {
                flow.set(col, row, 2.0, 3.0);
            }
        }
        let points = vec![(1.5, 1.5), (5.5, 5.5)];
        let advected = flow.advect_points(&points);
        assert!((advected[0].0 - 3.5).abs() < 1e-5);
        assert!((advected[0].1 - 4.5).abs() < 1e-5);
        assert!((advected[1].0 - 7.5).abs() < 1e-5);
        assert!((advected[1].1 - 8.5).abs() < 1e-5);
    }

    #[test]
    fn test_upsample_2x() {
        let mut flow = FlowField::new(4, 4);
        for row in 0..4 {
            for col in 0..4 {
                flow.set(col, row, 1.0, 2.0);
            }
        }
        let up = flow.upsample_2x();
        assert_eq!(up.width(), 8);
        assert_eq!(up.height(), 8);
        // Magnitude should be doubled
        let (dx, dy) = up.get(4, 4);
        assert!((dx - 2.0).abs() < 0.1);
        assert!((dy - 4.0).abs() < 0.1);
    }

    #[test]
    fn test_gray_image_from_u8() {
        let data = vec![0u8, 128, 255];
        let img = GrayImage::from_u8(3, 1, &data);
        assert!((img.get_pixel(0, 0) - 0.0).abs() < 1e-6);
        assert!((img.get_pixel(1, 0) - 128.0 / 255.0).abs() < 1e-6);
        assert!((img.get_pixel(2, 0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_identical_images_near_zero_flow() {
        let img = GrayImage::checkerboard(64, 64);
        let params = DisFlowParams::fast();
        let flow = compute_optical_flow(&img, &img, &params, None);

        // Flow should be near zero for identical images
        let mut max_flow = 0.0f32;
        for row in 0..flow.height() {
            for col in 0..flow.width() {
                let (dx, dy) = flow.get(col, row);
                max_flow = max_flow.max(dx.abs()).max(dy.abs());
            }
        }
        assert!(
            max_flow < 1.0,
            "Identical images should produce near-zero flow, got max {}",
            max_flow
        );
    }

    #[test]
    fn test_horizontal_shift() {
        let img_a = GrayImage::checkerboard(128, 128);
        let img_b = GrayImage::shifted(&img_a, 3.0, 0.0);
        let params = DisFlowParams {
            finest_scale: Some(0),
            coarsest_scale: Some(3),
            variational_refinement: false,
            ..DisFlowParams::default_quality()
        };
        let flow = compute_optical_flow(&img_a, &img_b, &params, None);

        // Check flow in the center region (away from borders)
        let mut sum_dx = 0.0;
        let mut sum_dy = 0.0;
        let mut count = 0;
        let margin = 20;
        for row in margin..flow.height() - margin {
            for col in margin..flow.width() - margin {
                let (dx, dy) = flow.get(col, row);
                sum_dx += dx;
                sum_dy += dy;
                count += 1;
            }
        }
        let avg_dx = sum_dx / count as f32;
        let avg_dy = sum_dy / count as f32;

        assert!(
            (avg_dx - 3.0).abs() < 2.0,
            "Expected avg dx ~3.0, got {}",
            avg_dx
        );
        assert!(avg_dy.abs() < 2.0, "Expected avg dy ~0.0, got {}", avg_dy);
    }

    #[test]
    fn test_presets() {
        // Just verify presets don't panic and produce valid params
        let fast = DisFlowParams::fast();
        assert_eq!(fast.patch_size, 8);
        assert!(!fast.variational_refinement);

        let default = DisFlowParams::default_quality();
        assert_eq!(default.patch_size, 8);
        assert!(default.variational_refinement);

        let hq = DisFlowParams::high_quality();
        assert_eq!(hq.patch_size, 12);
        assert!(hq.variational_refinement);
    }

    #[test]
    fn test_coarsest_scale_computation() {
        let params = DisFlowParams::default_quality();
        // For 2880 width: floor(log2(2*2880/(5*8))) = floor(log2(144)) = 7
        assert_eq!(params.compute_coarsest_scale(2880), 7);
        // For 640 width: floor(log2(2*640/(5*8))) = floor(log2(32)) = 5
        assert_eq!(params.compute_coarsest_scale(640), 5);
    }

    #[test]
    fn test_downsample_2x() {
        let mut flow = FlowField::new(8, 8);
        for row in 0..8 {
            for col in 0..8 {
                flow.set(col, row, 4.0, 6.0);
            }
        }
        let down = flow.downsample_2x();
        assert_eq!(down.width(), 4);
        assert_eq!(down.height(), 4);
        // Magnitude should be halved
        let (dx, dy) = down.get(2, 2);
        assert!((dx - 2.0).abs() < 0.1, "Expected dx ~2.0, got {}", dx);
        assert!((dy - 3.0).abs() < 0.1, "Expected dy ~3.0, got {}", dy);
    }

    #[test]
    fn test_downsample_upsample_roundtrip() {
        let mut flow = FlowField::new(8, 8);
        for row in 0..8 {
            for col in 0..8 {
                flow.set(col, row, 4.0, -2.0);
            }
        }
        let roundtrip = flow.downsample_2x().upsample_2x();
        assert_eq!(roundtrip.width(), 8);
        assert_eq!(roundtrip.height(), 8);
        // Center pixel should recover roughly the original values
        let (dx, dy) = roundtrip.get(4, 4);
        assert!((dx - 4.0).abs() < 0.5, "Expected dx ~4.0, got {}", dx);
        assert!((dy - (-2.0)).abs() < 0.5, "Expected dy ~-2.0, got {}", dy);
    }

    #[test]
    fn test_compose_flow_with_zero() {
        let mut flow_ab = FlowField::new(10, 10);
        for row in 0..10 {
            for col in 0..10 {
                flow_ab.set(col, row, 2.0, 3.0);
            }
        }
        let flow_bc = FlowField::new(10, 10); // zero flow
        let composed = compose_flow(&flow_ab, &flow_bc);
        // Composing with zero should give the original
        let (dx, dy) = composed.get(5, 5);
        assert!((dx - 2.0).abs() < 1e-5);
        assert!((dy - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_compose_flow_additive() {
        // Two constant flows should add
        let mut flow_ab = FlowField::new(10, 10);
        let mut flow_bc = FlowField::new(10, 10);
        for row in 0..10 {
            for col in 0..10 {
                flow_ab.set(col, row, 1.0, 0.0);
                flow_bc.set(col, row, 0.0, 2.0);
            }
        }
        let composed = compose_flow(&flow_ab, &flow_bc);
        let (dx, dy) = composed.get(5, 5);
        assert!((dx - 1.0).abs() < 1e-5, "Expected dx ~1.0, got {}", dx);
        assert!((dy - 2.0).abs() < 1e-5, "Expected dy ~2.0, got {}", dy);
    }

    #[test]
    fn test_compute_optical_flow_with_init_identical() {
        // With identical images, init flow should be refined toward zero
        let img = GrayImage::checkerboard(64, 64);
        let params = DisFlowParams::fast();

        let mut init = FlowField::new(64, 64);
        for row in 0..64 {
            for col in 0..64 {
                init.set(col, row, 1.0, 1.0);
            }
        }

        let flow = compute_optical_flow_with_init(&img, &img, &params, &init, None);

        // Should be near zero (solver corrects the bad init)
        let mut max_flow = 0.0f32;
        let margin = 10;
        for row in margin..flow.height() - margin {
            for col in margin..flow.width() - margin {
                let (dx, dy) = flow.get(col, row);
                max_flow = max_flow.max(dx.abs()).max(dy.abs());
            }
        }
        assert!(
            max_flow < 2.0,
            "Expected near-zero flow for identical images with init, got max {}",
            max_flow
        );
    }

    #[test]
    fn test_compute_optical_flow_with_init_shift() {
        let img_a = GrayImage::checkerboard(128, 128);
        let img_b = GrayImage::shifted(&img_a, 3.0, 0.0);
        let params = DisFlowParams {
            finest_scale: Some(0),
            coarsest_scale: Some(3),
            variational_refinement: false,
            ..DisFlowParams::default_quality()
        };

        // Provide a good initial flow
        let mut init = FlowField::new(128, 128);
        for row in 0..128 {
            for col in 0..128 {
                init.set(col, row, 3.0, 0.0);
            }
        }

        let flow = compute_optical_flow_with_init(&img_a, &img_b, &params, &init, None);

        // Check center region
        let mut sum_dx = 0.0;
        let mut count = 0;
        let margin = 20;
        for row in margin..flow.height() - margin {
            for col in margin..flow.width() - margin {
                sum_dx += flow.get(col, row).0;
                count += 1;
            }
        }
        let avg_dx = sum_dx / count as f32;
        assert!(
            (avg_dx - 3.0).abs() < 1.5,
            "Expected avg dx ~3.0, got {}",
            avg_dx
        );
    }
}