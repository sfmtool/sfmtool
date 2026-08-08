// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! [`DisFlowParams`] (algorithm parameters, following Kroeger et al.,
//! "Fast Optical Flow using Dense Inverse Search", ECCV 2016) and
//! [`FlowTiming`] (per-stage timing breakdown).

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
    pub(super) fn patch_stride(&self) -> u32 {
        (self.patch_size as f32 * (1.0 - self.patch_overlap))
            .floor()
            .max(1.0) as u32
    }

    /// Auto-compute coarsest scale from image width.
    pub(super) fn compute_coarsest_scale(&self, width: u32) -> u32 {
        let f = 5.0;
        let val = 2.0 * width as f64 / (f * self.patch_size as f64);
        val.log2().floor() as u32
    }
}

/// Per-stage timing breakdown for optical flow computation.
///
/// All times are in seconds. Only populated when
/// [`compute_optical_flow_timed`](super::compute_optical_flow_timed) is used.
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
