// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for optical flow computation.

use numpy::{IntoPyArray, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use std::sync::OnceLock;

use sfmtool_core::features::optical_flow;

use crate::sift::image_readings::PyImageToGray;

/// The grey the optical flow and the patch matchers read: BT.601 luma
/// rounded to 8 bits, ``ImageToGray("0.299*R + 0.587*G + 0.114*B",
/// quantize_bits=8)``.
///
/// The flow functions take an (H, W) uint8 grey image; this conversion,
/// applied with ``sfmtool.sift.gray_from_rgb`` and scaled by 255, makes it
/// from an RGB photograph, the same grey Find nearby tracks reads.
///
/// Returns:
///     ImageToGray.
#[pyfunction]
pub fn matcher_image_to_gray() -> PyImageToGray {
    PyImageToGray {
        inner: optical_flow::matcher_image_to_gray(),
    }
}

/// Lazily initialized GPU context, shared across all calls.
/// Returns `None` if no GPU is available.
static GPU_CONTEXT: OnceLock<Option<optical_flow::gpu::GpuFlowContext>> = OnceLock::new();

fn get_gpu_context() -> Option<&'static optical_flow::gpu::GpuFlowContext> {
    GPU_CONTEXT
        .get_or_init(optical_flow::gpu::GpuFlowContext::new)
        .as_ref()
}

/// Resolve the `use_gpu` parameter: None=auto (use if available), true=require, false=skip.
fn resolve_gpu(
    use_gpu: Option<bool>,
) -> PyResult<Option<&'static optical_flow::gpu::GpuFlowContext>> {
    match use_gpu {
        Some(false) => Ok(None),
        Some(true) => {
            let ctx = get_gpu_context();
            if ctx.is_none() {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "use_gpu=True but no GPU is available",
                ));
            }
            Ok(ctx)
        }
        None => Ok(get_gpu_context()),
    }
}

/// Returns True if a GPU is available for optical flow computation.
#[pyfunction]
pub fn gpu_available() -> bool {
    get_gpu_context().is_some()
}

/// Compute dense optical flow between two grayscale images.
///
/// Args:
///     img_a: (H, W) uint8 numpy array (grayscale)
///     img_b: (H, W) uint8 numpy array, the same shape as img_a
///     preset: "default" (default), "fast", or "high_quality"
///     use_gpu: None (default, auto-detect), True (require GPU), or False (force CPU)
///     patch_size, patch_stride, grad_descent_iterations, finest_scale,
///     coarsest_scale, variational_refinement, variational_jacobi_iterations,
///     variational_outer_iterations_base, variational_alpha, variational_gamma,
///     variational_delta, normalize_patches, gpu_min_pixels: keyword-only
///         overrides of the preset's DIS parameters; None keeps the preset's
///         value. "Python Bindings" in specs/core/features/optical-flow.md
///         gives each one's meaning and range.
///
/// Returns:
///     Tuple of two (H, W) numpy arrays of float32: (flow_u, flow_v)
///     where flow_u is horizontal displacement and flow_v is vertical displacement.
///     Stack with `np.stack((flow_u, flow_v), axis=-1)` to get (H, W, 2) if needed.
///
/// Raises:
///     TypeError: If an image is not a 2-D uint8 array.
///     ValueError: If the two images differ in shape, the preset is unknown,
///         or a parameter is out of its range for the images.
///     OverflowError: If an integer parameter is negative.
///     RuntimeError: If use_gpu is True and no GPU is available.
#[pyfunction]
#[pyo3(signature = (img_a, img_b, preset=None, use_gpu=None, *, patch_size=None, patch_stride=None, grad_descent_iterations=None, finest_scale=None, coarsest_scale=None, variational_refinement=None, variational_jacobi_iterations=None, variational_outer_iterations_base=None, variational_alpha=None, variational_gamma=None, variational_delta=None, normalize_patches=None, gpu_min_pixels=None))]
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
pub fn compute_optical_flow(
    py: Python<'_>,
    img_a: PyReadonlyArray2<'_, u8>,
    img_b: PyReadonlyArray2<'_, u8>,
    preset: Option<&str>,
    use_gpu: Option<bool>,
    patch_size: Option<u32>,
    patch_stride: Option<u32>,
    grad_descent_iterations: Option<u32>,
    finest_scale: Option<u32>,
    coarsest_scale: Option<u32>,
    variational_refinement: Option<bool>,
    variational_jacobi_iterations: Option<u32>,
    variational_outer_iterations_base: Option<u32>,
    variational_alpha: Option<f32>,
    variational_gamma: Option<f32>,
    variational_delta: Option<f32>,
    normalize_patches: Option<bool>,
    gpu_min_pixels: Option<u32>,
) -> PyResult<(Py<numpy::PyArray2<f32>>, Py<numpy::PyArray2<f32>>)> {
    let shape_a = img_a.shape();
    let shape_b = img_b.shape();
    if shape_a != shape_b {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "img_a and img_b must have the same shape",
        ));
    }

    let h = shape_a[0] as u32;
    let w = shape_a[1] as u32;

    let params = flow_params(
        preset,
        FlowParamOverrides {
            patch_size,
            patch_stride,
            grad_descent_iterations,
            finest_scale,
            coarsest_scale,
            variational_refinement,
            variational_jacobi_iterations,
            variational_outer_iterations_base,
            variational_alpha,
            variational_gamma,
            variational_delta,
            normalize_patches,
            gpu_min_pixels,
        },
    )?;
    let gpu = resolve_gpu(use_gpu)?;

    let data_a: Vec<u8> = to_contiguous!(img_a).into_owned();
    let data_b: Vec<u8> = to_contiguous!(img_b).into_owned();

    let gray_a = optical_flow::GrayImage::from_u8(w, h, &data_a);
    let gray_b = optical_flow::GrayImage::from_u8(w, h, &data_b);

    let flow = py
        .detach(|| optical_flow::compute_optical_flow(&gray_a, &gray_b, &params, gpu))
        .map_err(flow_error)?;

    let shape = (h as usize, w as usize);
    let arr_u = ndarray::Array2::from_shape_vec(shape, flow.u_slice().to_vec())
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    let arr_v = ndarray::Array2::from_shape_vec(shape, flow.v_slice().to_vec())
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    Ok((arr_u.into_pyarray(py).into(), arr_v.into_pyarray(py).into()))
}

/// Compute dense optical flow with an initial flow estimate.
///
/// The initial flow is downsampled into the coarsest pyramid level, and the solver
/// refines from there. Useful when a chained or approximate flow is available.
///
/// Args:
///     img_a: (H, W) uint8 numpy array (grayscale)
///     img_b: (H, W) uint8 numpy array, the same shape as img_a
///     initial_flow_u: (H, W) float32 initial horizontal flow estimate
///     initial_flow_v: (H, W) float32 initial vertical flow estimate
///     preset: "default" (default), "fast", or "high_quality"
///     use_gpu: None (default, auto-detect), True (require GPU), or False (force CPU)
///     patch_size, patch_stride, grad_descent_iterations, finest_scale,
///     coarsest_scale, variational_refinement, variational_jacobi_iterations,
///     variational_outer_iterations_base, variational_alpha, variational_gamma,
///     variational_delta, normalize_patches, gpu_min_pixels: keyword-only
///         overrides of the preset's DIS parameters; None keeps the preset's
///         value. "Python Bindings" in specs/core/features/optical-flow.md
///         gives each one's meaning and range.
///
/// Returns:
///     Tuple of two (H, W) numpy arrays of float32: (flow_u, flow_v)
///
/// Raises:
///     TypeError: If an image is not a 2-D uint8 array.
///     ValueError: If the images or the initial flow arrays differ in shape,
///         the preset is unknown, or a parameter is out of its range for the
///         images.
///     OverflowError: If an integer parameter is negative.
///     RuntimeError: If use_gpu is True and no GPU is available.
#[pyfunction]
#[pyo3(signature = (img_a, img_b, initial_flow_u, initial_flow_v, preset=None, use_gpu=None, *, patch_size=None, patch_stride=None, grad_descent_iterations=None, finest_scale=None, coarsest_scale=None, variational_refinement=None, variational_jacobi_iterations=None, variational_outer_iterations_base=None, variational_alpha=None, variational_gamma=None, variational_delta=None, normalize_patches=None, gpu_min_pixels=None))]
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
pub fn compute_optical_flow_with_init(
    py: Python<'_>,
    img_a: PyReadonlyArray2<'_, u8>,
    img_b: PyReadonlyArray2<'_, u8>,
    initial_flow_u: PyReadonlyArray2<'_, f32>,
    initial_flow_v: PyReadonlyArray2<'_, f32>,
    preset: Option<&str>,
    use_gpu: Option<bool>,
    patch_size: Option<u32>,
    patch_stride: Option<u32>,
    grad_descent_iterations: Option<u32>,
    finest_scale: Option<u32>,
    coarsest_scale: Option<u32>,
    variational_refinement: Option<bool>,
    variational_jacobi_iterations: Option<u32>,
    variational_outer_iterations_base: Option<u32>,
    variational_alpha: Option<f32>,
    variational_gamma: Option<f32>,
    variational_delta: Option<f32>,
    normalize_patches: Option<bool>,
    gpu_min_pixels: Option<u32>,
) -> PyResult<(Py<numpy::PyArray2<f32>>, Py<numpy::PyArray2<f32>>)> {
    let shape_a = img_a.shape();
    let shape_b = img_b.shape();
    if shape_a != shape_b {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "img_a and img_b must have the same shape",
        ));
    }
    if initial_flow_u.shape() != shape_a || initial_flow_v.shape() != shape_a {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Initial flow arrays must have the same shape as images",
        ));
    }

    let h = shape_a[0] as u32;
    let w = shape_a[1] as u32;

    let params = flow_params(
        preset,
        FlowParamOverrides {
            patch_size,
            patch_stride,
            grad_descent_iterations,
            finest_scale,
            coarsest_scale,
            variational_refinement,
            variational_jacobi_iterations,
            variational_outer_iterations_base,
            variational_alpha,
            variational_gamma,
            variational_delta,
            normalize_patches,
            gpu_min_pixels,
        },
    )?;
    let gpu = resolve_gpu(use_gpu)?;

    let data_a: Vec<u8> = to_contiguous!(img_a).into_owned();
    let data_b: Vec<u8> = to_contiguous!(img_b).into_owned();
    let init_u: Vec<f32> = to_contiguous!(initial_flow_u).into_owned();
    let init_v: Vec<f32> = to_contiguous!(initial_flow_v).into_owned();

    let gray_a = optical_flow::GrayImage::from_u8(w, h, &data_a);
    let gray_b = optical_flow::GrayImage::from_u8(w, h, &data_b);
    let init_flow = optical_flow::FlowField::from_split(w, h, init_u, init_v);

    let flow = py
        .detach(|| {
            optical_flow::compute_optical_flow_with_init(&gray_a, &gray_b, &params, &init_flow, gpu)
        })
        .map_err(flow_error)?;

    let shape = (h as usize, w as usize);
    let arr_u = ndarray::Array2::from_shape_vec(shape, flow.u_slice().to_vec())
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    let arr_v = ndarray::Array2::from_shape_vec(shape, flow.v_slice().to_vec())
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    Ok((arr_u.into_pyarray(py).into(), arr_v.into_pyarray(py).into()))
}

/// Compose two flow fields: result(x) = flow_ab(x) + flow_bc(x + flow_ab(x)).
///
/// The composed field maps points from image A to image C via B.
/// flow_bc is sampled at the advected position using bilinear interpolation.
///
/// Args:
///     flow_ab_u: (H, W) float32 horizontal flow A→B
///     flow_ab_v: (H, W) float32 vertical flow A→B
///     flow_bc_u: (H, W) float32 horizontal flow B→C
///     flow_bc_v: (H, W) float32 vertical flow B→C
///
/// Returns:
///     Tuple of two (H, W) float32 arrays: (flow_ac_u, flow_ac_v)
#[pyfunction]
#[allow(clippy::type_complexity)]
pub fn compose_flow(
    py: Python<'_>,
    flow_ab_u: PyReadonlyArray2<'_, f32>,
    flow_ab_v: PyReadonlyArray2<'_, f32>,
    flow_bc_u: PyReadonlyArray2<'_, f32>,
    flow_bc_v: PyReadonlyArray2<'_, f32>,
) -> PyResult<(Py<numpy::PyArray2<f32>>, Py<numpy::PyArray2<f32>>)> {
    let shape_ab = flow_ab_u.shape();
    if flow_ab_v.shape() != shape_ab
        || flow_bc_u.shape() != shape_ab
        || flow_bc_v.shape() != shape_ab
    {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "All flow arrays must have the same shape",
        ));
    }

    let h = shape_ab[0] as u32;
    let w = shape_ab[1] as u32;

    // Borrow flow data directly from numpy arrays (zero-copy)
    let ab_u = to_contiguous!(flow_ab_u);
    let ab_v = to_contiguous!(flow_ab_v);
    let bc_u = to_contiguous!(flow_bc_u);
    let bc_v = to_contiguous!(flow_bc_v);

    let ref_ab = optical_flow::FlowFieldRef::from_slices(w, h, &ab_u, &ab_v);
    let ref_bc = optical_flow::FlowFieldRef::from_slices(w, h, &bc_u, &bc_v);

    let result = py.detach(|| optical_flow::compose_flow_ref(&ref_ab, &ref_bc));

    let shape = (h as usize, w as usize);
    let arr_u = ndarray::Array2::from_shape_vec(shape, result.u_slice().to_vec())
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    let arr_v = ndarray::Array2::from_shape_vec(shape, result.v_slice().to_vec())
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    Ok((arr_u.into_pyarray(py).into(), arr_v.into_pyarray(py).into()))
}

/// Advect 2D points through a flow field using bilinear interpolation.
///
/// Args:
///     points: (N, 2) numpy array of (x, y) float32
///     flow_u: (H, W) numpy array of float32 (horizontal displacements)
///     flow_v: (H, W) numpy array of float32 (vertical displacements)
///
/// Returns:
///     (N, 2) numpy array of advected (x, y) float32
#[pyfunction]
pub fn advect_points(
    py: Python<'_>,
    points: PyReadonlyArray2<'_, f32>,
    flow_u: PyReadonlyArray2<'_, f32>,
    flow_v: PyReadonlyArray2<'_, f32>,
) -> PyResult<Py<numpy::PyArray2<f32>>> {
    let u_shape = flow_u.shape();
    let v_shape = flow_v.shape();
    if u_shape != v_shape {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "flow_u and flow_v must have the same shape",
        ));
    }

    let h = u_shape[0] as u32;
    let w = u_shape[1] as u32;

    let points_shape = points.shape();
    if points_shape.len() != 2 || points_shape[1] != 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "points must have shape (N, 2)",
        ));
    }

    // Borrow flow data directly from numpy arrays (zero-copy)
    let u_slice = to_contiguous!(flow_u);
    let v_slice = to_contiguous!(flow_v);
    let flow_ref = optical_flow::FlowFieldRef::from_slices(w, h, &u_slice, &v_slice);

    // Borrow points directly from numpy (zero-copy); reinterpret as (f32, f32) pairs
    let n = points_shape[0];
    let points_slice = to_contiguous!(points);
    // Safety: (f32, f32) has the same layout as [f32; 2] with no padding
    let point_pairs: &[(f32, f32)] =
        unsafe { std::slice::from_raw_parts(points_slice.as_ptr().cast(), n) };

    let advected = py.detach(|| flow_ref.advect_points(point_pairs));

    let mut result = Vec::with_capacity(n * 2);
    for (x, y) in advected {
        result.push(x);
        result.push(y);
    }

    let array = ndarray::Array2::from_shape_vec((n, 2), result)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    Ok(array.into_pyarray(py).into())
}

/// Compute dense optical flow with per-stage timing breakdown.
///
/// Takes the arguments of ``compute_optical_flow`` and raises its errors.
///
/// Returns a dict with timing info for each pipeline stage:
///   - pyramid_build: time to build Gaussian pyramids (seconds)
///   - dis_total: total DIS inverse search + densification time (seconds)
///   - variational_total: total variational refinement time (seconds)
///   - upsample_total: total flow upsampling time (seconds)
///   - total: total wall-clock time (seconds)
///   - levels_processed: number of pyramid levels processed
///   - per_level: list of (scale, width, height, dis_time, var_time) tuples
///   - flow_u, flow_v: the computed flow arrays
#[pyfunction]
#[pyo3(signature = (img_a, img_b, preset=None, use_gpu=None, *, patch_size=None, patch_stride=None, grad_descent_iterations=None, finest_scale=None, coarsest_scale=None, variational_refinement=None, variational_jacobi_iterations=None, variational_outer_iterations_base=None, variational_alpha=None, variational_gamma=None, variational_delta=None, normalize_patches=None, gpu_min_pixels=None))]
#[allow(clippy::too_many_arguments)]
pub fn compute_optical_flow_timed(
    py: Python<'_>,
    img_a: PyReadonlyArray2<'_, u8>,
    img_b: PyReadonlyArray2<'_, u8>,
    preset: Option<&str>,
    use_gpu: Option<bool>,
    patch_size: Option<u32>,
    patch_stride: Option<u32>,
    grad_descent_iterations: Option<u32>,
    finest_scale: Option<u32>,
    coarsest_scale: Option<u32>,
    variational_refinement: Option<bool>,
    variational_jacobi_iterations: Option<u32>,
    variational_outer_iterations_base: Option<u32>,
    variational_alpha: Option<f32>,
    variational_gamma: Option<f32>,
    variational_delta: Option<f32>,
    normalize_patches: Option<bool>,
    gpu_min_pixels: Option<u32>,
) -> PyResult<Py<PyAny>> {
    let shape_a = img_a.shape();
    let shape_b = img_b.shape();
    if shape_a != shape_b {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "img_a and img_b must have the same shape",
        ));
    }

    let h = shape_a[0] as u32;
    let w = shape_a[1] as u32;

    let params = flow_params(
        preset,
        FlowParamOverrides {
            patch_size,
            patch_stride,
            grad_descent_iterations,
            finest_scale,
            coarsest_scale,
            variational_refinement,
            variational_jacobi_iterations,
            variational_outer_iterations_base,
            variational_alpha,
            variational_gamma,
            variational_delta,
            normalize_patches,
            gpu_min_pixels,
        },
    )?;
    let gpu = resolve_gpu(use_gpu)?;

    let data_a: Vec<u8> = to_contiguous!(img_a).into_owned();
    let data_b: Vec<u8> = to_contiguous!(img_b).into_owned();

    let gray_a = optical_flow::GrayImage::from_u8(w, h, &data_a);
    let gray_b = optical_flow::GrayImage::from_u8(w, h, &data_b);

    let (flow, timing) = py
        .detach(|| optical_flow::compute_optical_flow_timed(&gray_a, &gray_b, &params, gpu))
        .map_err(flow_error)?;

    let shape = (h as usize, w as usize);
    let arr_u = ndarray::Array2::from_shape_vec(shape, flow.u_slice().to_vec())
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    let arr_v = ndarray::Array2::from_shape_vec(shape, flow.v_slice().to_vec())
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("pyramid_build", timing.pyramid_build)?;
    dict.set_item("dis_total", timing.dis_total)?;
    dict.set_item("variational_total", timing.variational_total)?;
    dict.set_item("upsample_total", timing.upsample_total)?;
    dict.set_item("total", timing.total)?;
    dict.set_item("levels_processed", timing.levels_processed)?;

    let per_level: Vec<(u32, u32, u32, f64, f64)> = timing.per_level;
    dict.set_item("per_level", per_level)?;

    dict.set_item("flow_u", arr_u.into_pyarray(py))?;
    dict.set_item("flow_v", arr_v.into_pyarray(py))?;

    Ok(dict.into())
}

/// The keyword-only overrides of a `DisFlowParams`' fields that the flow
/// functions take, each named as the field it sets, and that
/// `sfmtool.bench.pairwise_flow_normal` takes with a `flow_` prefix; `None`
/// keeps the value it is applied to. The core checks the values that result.
pub(crate) struct FlowParamOverrides {
    pub(crate) patch_size: Option<u32>,
    pub(crate) patch_stride: Option<u32>,
    pub(crate) grad_descent_iterations: Option<u32>,
    pub(crate) finest_scale: Option<u32>,
    pub(crate) coarsest_scale: Option<u32>,
    pub(crate) variational_refinement: Option<bool>,
    pub(crate) variational_jacobi_iterations: Option<u32>,
    pub(crate) variational_outer_iterations_base: Option<u32>,
    pub(crate) variational_alpha: Option<f32>,
    pub(crate) variational_gamma: Option<f32>,
    pub(crate) variational_delta: Option<f32>,
    pub(crate) normalize_patches: Option<bool>,
    pub(crate) gpu_min_pixels: Option<u32>,
}

impl FlowParamOverrides {
    /// `params` with each override that is given in place of its field.
    pub(crate) fn apply(
        self,
        mut params: optical_flow::DisFlowParams,
    ) -> optical_flow::DisFlowParams {
        let FlowParamOverrides {
            patch_size,
            patch_stride,
            grad_descent_iterations,
            finest_scale,
            coarsest_scale,
            variational_refinement,
            variational_jacobi_iterations,
            variational_outer_iterations_base,
            variational_alpha,
            variational_gamma,
            variational_delta,
            normalize_patches,
            gpu_min_pixels,
        } = self;
        params.patch_size = patch_size.unwrap_or(params.patch_size);
        params.patch_stride = patch_stride.unwrap_or(params.patch_stride);
        params.grad_descent_iterations =
            grad_descent_iterations.unwrap_or(params.grad_descent_iterations);
        params.finest_scale = finest_scale.or(params.finest_scale);
        params.coarsest_scale = coarsest_scale.or(params.coarsest_scale);
        params.variational_refinement =
            variational_refinement.unwrap_or(params.variational_refinement);
        params.variational_jacobi_iterations =
            variational_jacobi_iterations.unwrap_or(params.variational_jacobi_iterations);
        params.variational_outer_iterations_base =
            variational_outer_iterations_base.unwrap_or(params.variational_outer_iterations_base);
        params.variational_alpha = variational_alpha.unwrap_or(params.variational_alpha);
        params.variational_gamma = variational_gamma.unwrap_or(params.variational_gamma);
        params.variational_delta = variational_delta.unwrap_or(params.variational_delta);
        params.normalize_patches = normalize_patches.unwrap_or(params.normalize_patches);
        params.gpu_min_pixels = gpu_min_pixels.unwrap_or(params.gpu_min_pixels);
        params
    }
}

/// The preset's `DisFlowParams` with `overrides` applied.
fn flow_params(
    preset: Option<&str>,
    overrides: FlowParamOverrides,
) -> PyResult<optical_flow::DisFlowParams> {
    let params = match preset.unwrap_or("default") {
        "fast" => optical_flow::DisFlowParams::fast(),
        "default" => optical_flow::DisFlowParams::default_quality(),
        "high_quality" => optical_flow::DisFlowParams::high_quality(),
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown preset '{}'. Use 'fast', 'default', or 'high_quality'.",
                other
            )))
        }
    };
    Ok(overrides.apply(params))
}

/// Raise an `OpticalFlowError` as a Python `ValueError`.
fn flow_error(e: optical_flow::OpticalFlowError) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(e.to_string())
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(gpu_available, m)?)?;
    m.add_function(wrap_pyfunction!(matcher_image_to_gray, m)?)?;
    m.add_function(wrap_pyfunction!(compute_optical_flow, m)?)?;
    m.add_function(wrap_pyfunction!(compute_optical_flow_timed, m)?)?;
    m.add_function(wrap_pyfunction!(compute_optical_flow_with_init, m)?)?;
    m.add_function(wrap_pyfunction!(compose_flow, m)?)?;
    m.add_function(wrap_pyfunction!(advect_points, m)?)?;
    Ok(())
}
