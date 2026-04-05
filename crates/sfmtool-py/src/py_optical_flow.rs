// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for optical flow computation.

use numpy::{IntoPyArray, PyArrayMethods, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use std::sync::OnceLock;

use sfmtool_core::feature_match::descriptor;
use sfmtool_core::optical_flow;

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
///     img_a: (H, W) numpy array, uint8 or float32
///     img_b: (H, W) numpy array, uint8 or float32
///     preset: "default" (default), "fast", or "high_quality"
///     use_gpu: None (default, auto-detect), True (require GPU), or False (force CPU)
///
/// Returns:
///     Tuple of two (H, W) numpy arrays of float32: (flow_u, flow_v)
///     where flow_u is horizontal displacement and flow_v is vertical displacement.
///     Stack with `np.stack((flow_u, flow_v), axis=-1)` to get (H, W, 2) if needed.
#[pyfunction]
#[pyo3(signature = (img_a, img_b, preset=None, use_gpu=None))]
#[allow(clippy::type_complexity)]
pub fn compute_optical_flow(
    py: Python<'_>,
    img_a: PyReadonlyArray2<'_, u8>,
    img_b: PyReadonlyArray2<'_, u8>,
    preset: Option<&str>,
    use_gpu: Option<bool>,
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

    let params = parse_flow_preset(preset)?;
    let gpu = resolve_gpu(use_gpu)?;

    let data_a: Vec<u8> = img_a
        .as_slice()
        .map_or_else(|_| img_a.to_vec().unwrap(), |s| s.to_vec());
    let data_b: Vec<u8> = img_b
        .as_slice()
        .map_or_else(|_| img_b.to_vec().unwrap(), |s| s.to_vec());

    let gray_a = optical_flow::GrayImage::from_u8(w, h, &data_a);
    let gray_b = optical_flow::GrayImage::from_u8(w, h, &data_b);

    let flow = py.detach(|| optical_flow::compute_optical_flow(&gray_a, &gray_b, &params, gpu));

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
///     img_a: (H, W) numpy array, uint8 or float32
///     img_b: (H, W) numpy array, uint8 or float32
///     initial_flow_u: (H, W) float32 initial horizontal flow estimate
///     initial_flow_v: (H, W) float32 initial vertical flow estimate
///     preset: "default" (default), "fast", or "high_quality"
///     use_gpu: None (default, auto-detect), True (require GPU), or False (force CPU)
///
/// Returns:
///     Tuple of two (H, W) numpy arrays of float32: (flow_u, flow_v)
#[pyfunction]
#[pyo3(signature = (img_a, img_b, initial_flow_u, initial_flow_v, preset=None, use_gpu=None))]
#[allow(clippy::type_complexity)]
pub fn compute_optical_flow_with_init(
    py: Python<'_>,
    img_a: PyReadonlyArray2<'_, u8>,
    img_b: PyReadonlyArray2<'_, u8>,
    initial_flow_u: PyReadonlyArray2<'_, f32>,
    initial_flow_v: PyReadonlyArray2<'_, f32>,
    preset: Option<&str>,
    use_gpu: Option<bool>,
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

    let params = parse_flow_preset(preset)?;
    let gpu = resolve_gpu(use_gpu)?;

    let data_a: Vec<u8> = img_a
        .as_slice()
        .map_or_else(|_| img_a.to_vec().unwrap(), |s| s.to_vec());
    let data_b: Vec<u8> = img_b
        .as_slice()
        .map_or_else(|_| img_b.to_vec().unwrap(), |s| s.to_vec());
    let init_u: Vec<f32> = initial_flow_u
        .as_slice()
        .map_or_else(|_| initial_flow_u.to_vec().unwrap(), |s| s.to_vec());
    let init_v: Vec<f32> = initial_flow_v
        .as_slice()
        .map_or_else(|_| initial_flow_v.to_vec().unwrap(), |s| s.to_vec());

    let gray_a = optical_flow::GrayImage::from_u8(w, h, &data_a);
    let gray_b = optical_flow::GrayImage::from_u8(w, h, &data_b);
    let init_flow = optical_flow::FlowField::from_split(w, h, init_u, init_v);

    let flow = py.detach(|| {
        optical_flow::compute_optical_flow_with_init(&gray_a, &gray_b, &params, &init_flow, gpu)
    });

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
    let ab_u = flow_ab_u
        .as_slice()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("flow_ab_u must be C-contiguous"))?;
    let ab_v = flow_ab_v
        .as_slice()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("flow_ab_v must be C-contiguous"))?;
    let bc_u = flow_bc_u
        .as_slice()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("flow_bc_u must be C-contiguous"))?;
    let bc_v = flow_bc_v
        .as_slice()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("flow_bc_v must be C-contiguous"))?;

    let ref_ab = optical_flow::FlowFieldRef::from_slices(w, h, ab_u, ab_v);
    let ref_bc = optical_flow::FlowFieldRef::from_slices(w, h, bc_u, bc_v);

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
    let u_slice = flow_u
        .as_slice()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("flow_u must be C-contiguous"))?;
    let v_slice = flow_v
        .as_slice()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("flow_v must be C-contiguous"))?;
    let flow_ref = optical_flow::FlowFieldRef::from_slices(w, h, u_slice, v_slice);

    // Borrow points directly from numpy (zero-copy); reinterpret as (f32, f32) pairs
    let n = points_shape[0];
    let points_slice = points
        .as_slice()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("points must be C-contiguous"))?;
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
#[pyo3(signature = (img_a, img_b, preset=None, use_gpu=None))]
pub fn compute_optical_flow_timed(
    py: Python<'_>,
    img_a: PyReadonlyArray2<'_, u8>,
    img_b: PyReadonlyArray2<'_, u8>,
    preset: Option<&str>,
    use_gpu: Option<bool>,
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

    let params = parse_flow_preset(preset)?;
    let gpu = resolve_gpu(use_gpu)?;

    let data_a: Vec<u8> = img_a
        .as_slice()
        .map_or_else(|_| img_a.to_vec().unwrap(), |s| s.to_vec());
    let data_b: Vec<u8> = img_b
        .as_slice()
        .map_or_else(|_| img_b.to_vec().unwrap(), |s| s.to_vec());

    let gray_a = optical_flow::GrayImage::from_u8(w, h, &data_a);
    let gray_b = optical_flow::GrayImage::from_u8(w, h, &data_b);

    let (flow, timing) =
        py.detach(|| optical_flow::compute_optical_flow_timed(&gray_a, &gray_b, &params, gpu));

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

/// Parse a flow preset string into DisFlowParams.
fn parse_flow_preset(preset: Option<&str>) -> PyResult<optical_flow::DisFlowParams> {
    match preset.unwrap_or("default") {
        "fast" => Ok(optical_flow::DisFlowParams::fast()),
        "default" => Ok(optical_flow::DisFlowParams::default_quality()),
        "high_quality" => Ok(optical_flow::DisFlowParams::high_quality()),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Unknown preset '{}'. Use 'fast', 'default', or 'high_quality'.",
            other
        ))),
    }
}

/// Match source features to target features using candidate indices and
/// descriptor distance, with deduplication.
///
/// For each query point, examines its candidate target indices, computes
/// descriptor L2 distances, and picks the best match under the threshold.
/// If multiple source features match the same target, keeps the one with
/// the lowest descriptor distance.
///
/// Args:
///     candidates: (Q, K) uint32 array of candidate target indices into
///         descriptors2 (u32::MAX = empty slot). Typically produced by
///         KdTree2d.nearest_k_within_radius.
///     in_bounds_idx: (Q,) uint32 array mapping query index to source feature index.
///     descriptors1: (N1, 128) uint8 source descriptors.
///     descriptors2: (N2, 128) uint8 target descriptors.
///     descriptor_threshold: Maximum L2 descriptor distance for a valid match.
///
/// Returns:
///     (M, 2) uint32 array of deduplicated (src_idx, dst_idx) matched pairs.
#[pyfunction]
pub fn match_candidates_by_descriptor(
    py: Python<'_>,
    candidates: PyReadonlyArray2<'_, u32>,
    in_bounds_idx: numpy::PyReadonlyArray1<'_, u32>,
    descriptors1: PyReadonlyArray2<'_, u8>,
    descriptors2: PyReadonlyArray2<'_, u8>,
    descriptor_threshold: f64,
) -> PyResult<Py<numpy::PyArray2<u32>>> {
    let cand_shape = candidates.shape();
    let n_queries = cand_shape[0];
    let k = cand_shape[1];

    let ibi_shape = in_bounds_idx.shape();
    if ibi_shape[0] != n_queries {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "in_bounds_idx length must match candidates row count",
        ));
    }

    let desc1_shape = descriptors1.shape();
    let desc2_shape = descriptors2.shape();
    let desc_len = desc1_shape[1];
    if desc2_shape[1] != desc_len {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "descriptors1 and descriptors2 must have the same descriptor length",
        ));
    }

    // Borrow all arrays zero-copy
    let cand_data = candidates
        .as_slice()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("candidates must be C-contiguous"))?;
    let ibi_data = in_bounds_idx.as_slice().map_err(|_| {
        pyo3::exceptions::PyValueError::new_err("in_bounds_idx must be C-contiguous")
    })?;
    let desc1_data = descriptors1.as_slice().map_err(|_| {
        pyo3::exceptions::PyValueError::new_err("descriptors1 must be C-contiguous")
    })?;
    let desc2_data = descriptors2.as_slice().map_err(|_| {
        pyo3::exceptions::PyValueError::new_err("descriptors2 must be C-contiguous")
    })?;

    let matches = py.detach(|| {
        descriptor::match_candidates_and_deduplicate(
            cand_data,
            ibi_data,
            desc1_data,
            desc2_data,
            n_queries,
            k,
            desc_len,
            descriptor_threshold,
        )
    });

    let n_matches = matches.len();
    let flat: Vec<u32> = matches.into_iter().flatten().collect();
    let array = ndarray::Array2::from_shape_vec((n_matches, 2), flat)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    Ok(array.into_pyarray(py).into())
}