// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! [`DisFlowParams`] (algorithm parameters, following Kroeger et al.,
//! "Fast Optical Flow using Dense Inverse Search", ECCV 2016), the
//! [`OpticalFlowError`] their validation returns, and [`FlowTiming`]
//! (per-stage timing breakdown).

/// Algorithm parameters for DIS optical flow.
///
/// Notation follows Kroeger et al., "Fast Optical Flow using Dense Inverse Search," ECCV 2016.
/// The flow functions check every field against the image and return an
/// [`OpticalFlowError`] for a value outside the range its doc gives.
#[derive(Debug, Clone)]
pub struct DisFlowParams {
    /// Patch size (square edge length in pixels), at least 1 and no larger
    /// than the image's width or height. Default: 8.
    pub patch_size: u32,
    /// Spacing in pixels between the top-left corners of neighbouring patches
    /// on the grid, along both axes; at least 1. The paper's overlap
    /// `theta_ov` is `1 - patch_stride / patch_size`. Default: 4.
    pub patch_stride: u32,
    /// Gradient descent iterations per patch, at least 1. Default: 12.
    pub grad_descent_iterations: u32,
    /// Finest pyramid level (0 = full res). If None, `coarsest_scale - 2`
    /// (saturating at 0). Flow is solved down to this level and then
    /// upsampled bilinearly to full resolution, so it has no detail finer
    /// than `2^finest_scale` pixels.
    pub finest_scale: Option<u32>,
    /// Coarsest pyramid level. If None, computed from the image width (see
    /// "Automatic coarsest scale" in `specs/core/features/optical-flow.md`),
    /// lowered to the deepest level that is at least one pixel on each side.
    /// Raised to `finest_scale` when that is coarser.
    pub coarsest_scale: Option<u32>,
    /// Enable variational refinement. Default: true.
    pub variational_refinement: bool,
    /// Variational refinement inner Jacobi solver iterations, at least 1.
    /// Default: 7.
    ///
    /// Not directly comparable to the SOR iterations (θ_vi) in Kroeger et al.
    /// or OpenCV's DIS. Jacobi converges slower per iteration than Gauss-Seidel
    /// SOR but each iteration is fully parallelizable (SIMD, multi-core).
    /// Roughly 4/3× the SOR count gives equivalent convergence.
    pub variational_jacobi_iterations: u32,
    /// Variational outer iterations base multiplier, at least 1: level `s`
    /// runs `variational_outer_iterations_base * (s + 1)` outer iterations.
    /// Default: 1.
    pub variational_outer_iterations_base: u32,
    /// Variational smoothness weight (alpha), finite and at least 0. Default: 10.
    pub variational_alpha: f32,
    /// Variational gradient weight (gamma), finite and at least 0. Default: 10.
    pub variational_gamma: f32,
    /// Variational intensity weight (delta), finite and at least 0. Default: 5.
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
            patch_stride: 4,
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
            patch_stride: 5,
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
    ///
    /// Its finest scale is 1, so the flow is solved at half resolution at
    /// most and upsampled to full resolution.
    pub fn high_quality() -> Self {
        Self {
            patch_size: 12,
            patch_stride: 3,
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

    /// Auto-compute coarsest scale from image width.
    pub(super) fn compute_coarsest_scale(&self, width: u32) -> u32 {
        let f = 5.0;
        let val = 2.0 * width as f64 / (f * self.patch_size as f64);
        val.log2().floor() as u32
    }

    /// Check the parameters against a `width` x `height` image and return the
    /// pyramid levels to process as `(finest, coarsest)`, with
    /// `finest <= coarsest` and the coarsest level at least one pixel on each
    /// side.
    pub(super) fn resolve_scales(
        &self,
        width: u32,
        height: u32,
    ) -> Result<(u32, u32), OpticalFlowError> {
        if self.patch_size == 0 {
            return Err(OpticalFlowError::PatchSizeZero);
        }
        if self.patch_stride == 0 {
            return Err(OpticalFlowError::PatchStrideZero);
        }
        if self.patch_size > width || self.patch_size > height {
            return Err(OpticalFlowError::PatchLargerThanImage {
                patch_size: self.patch_size,
                width,
                height,
            });
        }
        for (field, value) in [
            ("grad_descent_iterations", self.grad_descent_iterations),
            (
                "variational_jacobi_iterations",
                self.variational_jacobi_iterations,
            ),
            (
                "variational_outer_iterations_base",
                self.variational_outer_iterations_base,
            ),
        ] {
            if value == 0 {
                return Err(OpticalFlowError::ZeroIterations { field });
            }
        }
        for (field, value) in [
            ("variational_alpha", self.variational_alpha),
            ("variational_gamma", self.variational_gamma),
            ("variational_delta", self.variational_delta),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(OpticalFlowError::InvalidWeight { field, value });
            }
        }

        // Level `s` is `width >> s` by `height >> s` pixels, so the deepest
        // level that is at least one pixel on each side is floor(log2(min)).
        // The patch check above makes both sides at least 1.
        let max_scale = 31 - width.min(height).leading_zeros();
        for (field, scale) in [
            ("finest_scale", self.finest_scale),
            ("coarsest_scale", self.coarsest_scale),
        ] {
            if let Some(scale) = scale {
                if scale > max_scale {
                    return Err(OpticalFlowError::ScaleTooCoarse {
                        field,
                        scale,
                        max_scale,
                    });
                }
            }
        }
        if let (Some(finest), Some(coarsest)) = (self.finest_scale, self.coarsest_scale) {
            if finest > coarsest {
                return Err(OpticalFlowError::FinestAboveCoarsest { finest, coarsest });
            }
        }

        let coarsest = self
            .coarsest_scale
            .unwrap_or_else(|| self.compute_coarsest_scale(width).min(max_scale));
        let finest = self
            .finest_scale
            .unwrap_or_else(|| coarsest.saturating_sub(2));
        Ok((finest, coarsest.max(finest)))
    }

    /// The checks of [`Self::resolve_scales`] that no image passes: those it
    /// makes against the largest image a `u32` side can hold, so every refusal
    /// but [`OpticalFlowError::PatchLargerThanImage`] and a scale the image at
    /// hand is too small for. A caller that checks its settings before it has
    /// an image, such as a bench step's options, refuses with the errors the
    /// flow would.
    pub(crate) fn check_fields(&self) -> Result<(), OpticalFlowError> {
        self.resolve_scales(u32::MAX, u32::MAX).map(|_| ())
    }
}

/// Why [`compute_optical_flow`](super::compute_optical_flow) or one of its
/// variants refused its input.
///
/// A `field` names the [`DisFlowParams`] field at fault, which is also the
/// keyword argument the Python flow bindings take for it, and
/// `sfmtool.bench.pairwise_flow_normal` takes with a `flow_` prefix.
#[derive(Clone, Debug, PartialEq)]
pub enum OpticalFlowError {
    /// The two images differ in size; each is `(width, height)`.
    ImageSizeMismatch { a: (u32, u32), b: (u32, u32) },
    /// The initial flow differs in size from the images; each is
    /// `(width, height)`.
    InitialFlowSizeMismatch { image: (u32, u32), flow: (u32, u32) },
    /// `patch_size` is 0.
    PatchSizeZero,
    /// `patch_stride` is 0.
    PatchStrideZero,
    /// `patch_size` is larger than the image's width or height.
    PatchLargerThanImage {
        patch_size: u32,
        width: u32,
        height: u32,
    },
    /// An iteration count is 0.
    ZeroIterations { field: &'static str },
    /// A variational weight is negative, infinite or NaN.
    InvalidWeight { field: &'static str, value: f32 },
    /// `finest_scale` or `coarsest_scale` names a level smaller than one
    /// pixel on a side; `max_scale` is the deepest level that is not.
    ScaleTooCoarse {
        field: &'static str,
        scale: u32,
        max_scale: u32,
    },
    /// `finest_scale` and `coarsest_scale` are both given and `finest_scale`
    /// is the larger.
    FinestAboveCoarsest { finest: u32, coarsest: u32 },
}

impl std::fmt::Display for OpticalFlowError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ImageSizeMismatch { a, b } => write!(
                f,
                "the two images differ in size: {}x{} and {}x{}",
                a.0, a.1, b.0, b.1
            ),
            Self::InitialFlowSizeMismatch { image, flow } => write!(
                f,
                "the initial flow is {}x{} but the images are {}x{}",
                flow.0, flow.1, image.0, image.1
            ),
            Self::PatchSizeZero => write!(f, "patch_size must be at least 1"),
            Self::PatchStrideZero => write!(f, "patch_stride must be at least 1"),
            Self::PatchLargerThanImage {
                patch_size,
                width,
                height,
            } => write!(
                f,
                "patch_size {patch_size} is larger than the {width}x{height} image"
            ),
            Self::ZeroIterations { field } => write!(f, "{field} must be at least 1"),
            Self::InvalidWeight { field, value } => {
                write!(f, "{field} must be finite and at least 0, got {value}")
            }
            Self::ScaleTooCoarse {
                field,
                scale,
                max_scale,
            } => write!(
                f,
                "{field} {scale} is a pyramid level less than one pixel on a side; \
                 the deepest level of this image is {max_scale}"
            ),
            Self::FinestAboveCoarsest { finest, coarsest } => write!(
                f,
                "finest_scale {finest} is larger than coarsest_scale {coarsest}"
            ),
        }
    }
}

impl std::error::Error for OpticalFlowError {}

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
