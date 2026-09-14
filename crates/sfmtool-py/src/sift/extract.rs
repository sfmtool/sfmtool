// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for the sfmtool Rust SIFT detector and descriptor.
//!
//! Wraps [`sfmtool_core::features::sift::detect_keypoints`] and
//! [`sfmtool_core::features::sift::extract_sift`]. Mirrors the `py_optical_flow.rs` house
//! conventions: numpy in, `IntoPyArray` out, the heavy compute wrapped in
//! `py.detach(...)`, and `PyValueError` for bad inputs.

use numpy::{
    IntoPyArray, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, PyReadonlyArrayDyn,
    PyUntypedArrayMethods,
};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use sfmtool_core::features::sift::{
    self, affine_shape_from_similarity, describe_keypoints as core_describe_keypoints,
    detect_keypoints, extract_sift_partial as core_extract_sift_partial, gray, QueryKeypoint,
    SiftKeypoint, SiftParams,
};

/// Build the `(positions, affine_shapes)` numpy arrays shared by both entry
/// points. `positions` is `(N, 2)` float32 `(x, y)`, `affine_shapes` is
/// `(N, 2, 2)` float32 `[[a11, a12], [a21, a22]]`.
#[allow(clippy::type_complexity)]
fn keypoints_to_arrays(
    py: Python<'_>,
    keypoints: &[SiftKeypoint],
) -> PyResult<(Py<numpy::PyArray2<f32>>, Py<numpy::PyArray3<f32>>)> {
    let n = keypoints.len();

    let mut positions = Vec::with_capacity(n * 2);
    let mut affine = Vec::with_capacity(n * 4);
    for kp in keypoints {
        positions.push(kp.x);
        positions.push(kp.y);
        let [[a11, a12], [a21, a22]] = kp.affine_shape;
        affine.push(a11);
        affine.push(a12);
        affine.push(a21);
        affine.push(a22);
    }

    let positions_arr = ndarray::Array2::from_shape_vec((n, 2), positions)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    let affine_arr = ndarray::Array3::from_shape_vec((n, 2, 2), affine)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    Ok((
        positions_arr.into_pyarray(py).into(),
        affine_arr.into_pyarray(py).into(),
    ))
}

/// Convert a numpy uint8 image (`(H, W, 3)` RGB or `(H, W)` grayscale) into a
/// [`sift::GrayImage`], using `params.image_to_gray` for the RGB path.
fn image_to_gray(
    image: &PyReadonlyArrayDyn<'_, u8>,
    params: &SiftParams,
) -> PyResult<sift::GrayImage> {
    let shape = image.shape();
    let data = to_contiguous!(image);
    let data = &*data;

    match shape {
        // (H, W, 3) uint8 RGB -> grayscale via the image-to-gray formula.
        [h, w, 3] => {
            let (h, w) = (*h as u32, *w as u32);
            Ok(gray::gray_from_rgb(w, h, data, &params.image_to_gray))
        }
        // (H, W) uint8 grayscale -> GrayImage directly (normalized by /255).
        [h, w] => {
            let (h, w) = (*h as u32, *w as u32);
            Ok(sift::GrayImage::from_u8(w, h, data))
        }
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "image must be (H, W, 3) uint8 RGB or (H, W) uint8 grayscale, got shape {:?}",
            other
        ))),
    }
}

/// Apply a Python dict of overrides onto `SiftParams::default()`.
///
/// Keys map onto [`SiftParams`] fields by name; `gray_formula` is a string that
/// is parsed into [`SiftParams::image_to_gray`]. An unknown key or a value of
/// the wrong type raises `PyValueError`.
fn parse_sift_params(params: Option<&Bound<'_, PyDict>>) -> PyResult<SiftParams> {
    let mut out = SiftParams::default();
    let Some(dict) = params else {
        return Ok(out);
    };

    for (key, value) in dict.iter() {
        let key: String = key
            .extract()
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("params keys must be strings"))?;

        macro_rules! extract_field {
            ($field:ident, $ty:ty) => {{
                out.$field = value.extract::<$ty>().map_err(|_| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "params['{}'] has the wrong type",
                        key
                    ))
                })?;
            }};
        }

        match key.as_str() {
            "octave_layers" => extract_field!(octave_layers, u32),
            "sigma" => extract_field!(sigma, f64),
            "blur_radius_factor" => extract_field!(blur_radius_factor, f64),
            "input_sigma" => extract_field!(input_sigma, f64),
            "double_image" => extract_field!(double_image, bool),
            "contrast_threshold" => extract_field!(contrast_threshold, f64),
            "edge_threshold" => extract_field!(edge_threshold, f64),
            "max_num_features" => {
                // `None` means unlimited; an integer caps the feature count.
                out.max_num_features = if value.is_none() {
                    None
                } else {
                    Some(value.extract::<usize>().map_err(|_| {
                        pyo3::exceptions::PyValueError::new_err(format!(
                            "params['{}'] must be a non-negative integer or None",
                            key
                        ))
                    })?)
                };
            }
            "orientation_bins" => extract_field!(orientation_bins, u32),
            "peak_ratio" => extract_field!(peak_ratio, f64),
            "descriptor_width" => extract_field!(descriptor_width, u32),
            "descriptor_bins" => extract_field!(descriptor_bins, u32),
            "descriptor_magnification" => extract_field!(descriptor_magnification, f64),
            "descriptor_clamp" => extract_field!(descriptor_clamp, f64),
            "gray_formula" => {
                let formula: String = value.extract().map_err(|_| {
                    pyo3::exceptions::PyValueError::new_err(
                        "params['gray_formula'] must be a string",
                    )
                })?;
                out.image_to_gray = gray::parse_gray_formula(&formula).map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "params['gray_formula']: {}",
                        e
                    ))
                })?;
            }
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "unknown params key '{}'",
                    other
                )));
            }
        }
    }

    // Reject degenerate values that would otherwise panic deep in the pipeline
    // (e.g. `orientation_bins = 0` → `rem_euclid(0)`, `octave_layers = 0` →
    // `k = 2^∞` → a huge kernel allocation) so callers get a clean error.
    let bad = |name: &str, msg: &str| {
        Err(pyo3::exceptions::PyValueError::new_err(format!(
            "params['{name}'] {msg}"
        )))
    };
    if out.octave_layers < 1 {
        return bad("octave_layers", "must be >= 1");
    }
    if out.orientation_bins < 1 {
        return bad("orientation_bins", "must be >= 1");
    }
    if !(out.sigma.is_finite() && out.sigma > 0.0) {
        return bad("sigma", "must be a positive, finite number");
    }
    if !(out.blur_radius_factor.is_finite() && out.blur_radius_factor > 0.0) {
        return bad("blur_radius_factor", "must be a positive, finite number");
    }
    if !(out.input_sigma.is_finite() && out.input_sigma >= 0.0) {
        return bad("input_sigma", "must be a non-negative, finite number");
    }
    if !(out.descriptor_magnification.is_finite() && out.descriptor_magnification > 0.0) {
        return bad(
            "descriptor_magnification",
            "must be a positive, finite number",
        );
    }
    if !(out.descriptor_clamp.is_finite() && out.descriptor_clamp > 0.0) {
        return bad("descriptor_clamp", "must be a positive, finite number");
    }

    // The sfmtool descriptor emits a fixed 128-D vector, so the grid dimensions
    // are not configurable. Reject non-default values rather than silently
    // ignoring them. (`descriptor_magnification` / `descriptor_clamp` are
    // honored, so they need no such guard.)
    if out.descriptor_width != 4 || out.descriptor_bins != 8 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "descriptor_width and descriptor_bins are fixed at 4 and 8 (the sfmtool \
             descriptor produces a 128-D vector); got {} and {}",
            out.descriptor_width, out.descriptor_bins
        )));
    }

    Ok(out)
}

/// Detect and orient SIFT keypoints (no descriptors).
///
/// Args:
///     image: (H, W, 3) uint8 RGB or (H, W) uint8 grayscale numpy array.
///         RGB is converted to gray via the image-to-gray formula
///         (params['gray_formula'] if given, else the BT.709 default);
///         grayscale is normalized by /255.
///     params: optional dict of overrides onto SiftParams::default(). Keys:
///         octave_layers, sigma, blur_radius_factor, input_sigma, double_image,
///         contrast_threshold,
///         edge_threshold, max_num_features (int or None), orientation_bins,
///         peak_ratio, descriptor_width, descriptor_bins,
///         descriptor_magnification, descriptor_clamp (numeric/bool), and
///         gray_formula (str).
///
/// Returns:
///     Tuple of:
///       positions: (N, 2) float32 — (x, y) full-resolution, pixel-center coords.
///       affine_shapes: (N, 2, 2) float32 — [[a11, a12], [a21, a22]].
///       responses: (N,) float32 — contrast response |D(x̂)|.
#[pyfunction]
#[pyo3(signature = (image, params=None))]
#[allow(clippy::type_complexity)]
pub fn detect_sift_keypoints(
    py: Python<'_>,
    image: PyReadonlyArrayDyn<'_, u8>,
    params: Option<&Bound<'_, PyDict>>,
) -> PyResult<(
    Py<numpy::PyArray2<f32>>,
    Py<numpy::PyArray3<f32>>,
    Py<numpy::PyArray1<f32>>,
)> {
    let sift_params = parse_sift_params(params)?;
    let gray_image = image_to_gray(&image, &sift_params)?;

    let detection = py.detach(|| detect_keypoints(&gray_image, &sift_params));
    let keypoints = detection.keypoints;

    let (positions, affine_shapes) = keypoints_to_arrays(py, &keypoints)?;
    let responses: Vec<f32> = keypoints.iter().map(|kp| kp.response).collect();
    let responses_arr = responses.into_pyarray(py).into();

    Ok((positions, affine_shapes, responses_arr))
}

/// Detect, orient, and describe SIFT keypoints.
///
/// Args:
///     image: (H, W, 3) uint8 RGB or (H, W) uint8 grayscale numpy array
///         (see `detect_sift_keypoints` for the conversion rules).
///     params: optional dict of overrides onto SiftParams::default()
///         (see `detect_sift_keypoints` for accepted keys).
///     max_described: optional cap on how many keypoints to describe. Detection
///         still finds every keypoint, but only the top `max_described` (by
///         feature size) get a descriptor, so `descriptors` has
///         `min(max_described, N)` rows for `N` keypoints. `None` describes all.
///
/// Returns:
///     Tuple of:
///       positions: (N, 2) float32 — (x, y) full-resolution, pixel-center coords.
///       affine_shapes: (N, 2, 2) float32 — [[a11, a12], [a21, a22]].
///       descriptors: (K, 128) uint8, K = min(max_described, N) (K = N if None).
#[pyfunction]
#[pyo3(signature = (image, params=None, max_described=None))]
#[allow(clippy::type_complexity)]
pub fn extract_sift(
    py: Python<'_>,
    image: PyReadonlyArrayDyn<'_, u8>,
    params: Option<&Bound<'_, PyDict>>,
    max_described: Option<usize>,
) -> PyResult<(
    Py<numpy::PyArray2<f32>>,
    Py<numpy::PyArray3<f32>>,
    Py<numpy::PyArray2<u8>>,
)> {
    let sift_params = parse_sift_params(params)?;
    let gray_image = image_to_gray(&image, &sift_params)?;

    let features =
        py.detach(|| core_extract_sift_partial(&gray_image, &sift_params, max_described));

    let (positions, affine_shapes) = keypoints_to_arrays(py, &features.keypoints)?;

    let n = features.descriptors.len();
    let mut desc_flat = Vec::with_capacity(n * 128);
    for row in features.descriptors.rows() {
        desc_flat.extend_from_slice(row);
    }
    let desc_arr = ndarray::Array2::from_shape_vec((n, 128), desc_flat)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    Ok((positions, affine_shapes, desc_arr.into_pyarray(py).into()))
}

/// Describe caller-supplied keypoints.
///
/// A descriptor is a pure function of the image's scale space and a keypoint, so
/// a keypoint nothing detected -- a pixel a person pointed at, a feature carried
/// over from another image -- is described exactly as `extract_sift` describes
/// its own detections: the same descriptor kernel, the octave and pyramid level
/// derived from the keypoint's size.
///
/// Args:
///     image: (H, W, 3) uint8 RGB or (H, W) uint8 grayscale numpy array
///         (see `detect_sift_keypoints` for the conversion rules).
///     positions: (N, 2) float32 -- (x, y) full-resolution, pixel-center coords.
///     affine_shapes: (N, 2, 2) float32 -- [[a11, a12], [a21, a22]], the same
///         layout `extract_sift` returns. Its column-norm average is the
///         keypoint size and its first column's angle the orientation; build one
///         from a size and an angle with `affine_shapes_from_similarity`.
///     params: optional dict of overrides onto SiftParams::default()
///         (see `detect_sift_keypoints` for accepted keys).
///
/// Returns:
///     descriptors: (N, 128) uint8 -- row `i` describes keypoint `i`.
///
/// Raises:
///     ValueError: if a keypoint is outside the image or has a non-positive
///         size, if the two arrays disagree on N, or if params are invalid.
#[pyfunction]
#[pyo3(signature = (image, positions, affine_shapes, params=None))]
pub fn describe_keypoints(
    py: Python<'_>,
    image: PyReadonlyArrayDyn<'_, u8>,
    positions: PyReadonlyArray2<'_, f32>,
    affine_shapes: PyReadonlyArray3<'_, f32>,
    params: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<numpy::PyArray2<u8>>> {
    let sift_params = parse_sift_params(params)?;
    let gray_image = image_to_gray(&image, &sift_params)?;

    let n = positions.shape()[0];
    if positions.shape()[1] != 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "positions must be (N, 2) float32, got shape {:?}",
            positions.shape()
        )));
    }
    if affine_shapes.shape() != [n, 2, 2] {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "affine_shapes must be (N, 2, 2) float32 with the same N as positions ({n}), got              shape {:?}",
            affine_shapes.shape()
        )));
    }
    let pos = positions.as_array();
    let aff = affine_shapes.as_array();
    let keypoints: Vec<QueryKeypoint> = (0..n)
        .map(|i| QueryKeypoint {
            x: pos[[i, 0]],
            y: pos[[i, 1]],
            affine_shape: [
                [aff[[i, 0, 0]], aff[[i, 0, 1]]],
                [aff[[i, 1, 0]], aff[[i, 1, 1]]],
            ],
        })
        .collect();

    let descriptors = py
        .detach(|| core_describe_keypoints(&gray_image, &sift_params, &keypoints))
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    let mut flat = Vec::with_capacity(n * 128);
    for row in descriptors.rows() {
        flat.extend_from_slice(row);
    }
    let arr = ndarray::Array2::from_shape_vec((n, 128), flat)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(arr.into_pyarray(py).into())
}

/// The affine shapes of similarity keypoints: the scaled rotations
/// `[[s·cosθ, -s·sinθ], [s·sinθ, s·cosθ]]` that COLMAP and the `.sift` format
/// store, from a size and an angle. The conversion `describe_keypoints` and the
/// detector both state, so a caller with a size and an angle in hand does not
/// restate it.
///
/// Args:
///     scales: (N,) float32 -- keypoint sizes (the affine-shape column norm).
///     orientations: (N,) float32 -- orientations in radians.
///
/// Returns:
///     affine_shapes: (N, 2, 2) float32, ready for `describe_keypoints`.
#[pyfunction]
#[pyo3(signature = (scales, orientations))]
pub fn affine_shapes_from_similarity(
    py: Python<'_>,
    scales: PyReadonlyArray1<'_, f32>,
    orientations: PyReadonlyArray1<'_, f32>,
) -> PyResult<Py<numpy::PyArray3<f32>>> {
    let n = scales.shape()[0];
    if orientations.shape()[0] != n {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "scales and orientations must have the same length, got {n} and {}",
            orientations.shape()[0]
        )));
    }
    let s = scales.as_array();
    let o = orientations.as_array();
    let mut flat = Vec::with_capacity(n * 4);
    for i in 0..n {
        let [[a11, a12], [a21, a22]] = affine_shape_from_similarity(s[i], o[i]);
        flat.extend_from_slice(&[a11, a12, a21, a22]);
    }
    let arr = ndarray::Array3::from_shape_vec((n, 2, 2), flat)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(arr.into_pyarray(py).into())
}

// ── Registration ──────────────────────────────────────────────────────────

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(detect_sift_keypoints, m)?)?;
    m.add_function(wrap_pyfunction!(extract_sift, m)?)?;
    m.add_function(wrap_pyfunction!(describe_keypoints, m)?)?;
    m.add_function(wrap_pyfunction!(affine_shapes_from_similarity, m)?)?;
    Ok(())
}
