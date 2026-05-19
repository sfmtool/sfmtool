// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Python binding for the photometric RANSAC refinement.
//!
//! See ``specs/drafts/photometric-subsets-ransac.md`` for the algorithm.
//! This binding accepts a float32-backed
//! [`crate::PyPerSphericalTileSourceStack`] and returns a
//! :class:`RansacPhotometricOutput` whose fields are owned NumPy arrays.

use numpy::PyArray1;
use pyo3::prelude::*;

use sfmtool_core::photometric_ransac::{
    refine_photometric_ransac, RansacPhotometricError, RansacPhotometricOutput,
    RansacPhotometricParams,
};

use crate::py_per_spherical_tile_source_stack::Inner;
use crate::PyPerSphericalTileSourceStack;

fn err_to_py(e: RansacPhotometricError) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(format!("{e}"))
}

/// Result of :func:`refine_photometric_ransac`. Fields are NumPy arrays and
/// match the spec's ``RansacPhotometricOutput`` shape:
///
/// - ``primary_mask`` : bool ``[R]``
/// - ``secondary_mask`` : bool ``[R]``
/// - ``tile_primary_count`` : int32 ``[n_tiles]``
/// - ``tile_secondary_count`` : int32 ``[n_tiles]``
/// - ``tile_primary_lum_mad`` : float32 ``[n_tiles]`` (NaN where skipped)
/// - ``tile_secondary_lum_mad`` : float32 ``[n_tiles]`` (NaN where skipped)
#[pyclass(name = "RansacPhotometricOutput", module = "sfmtool")]
pub struct PyRansacPhotometricOutput {
    inner: RansacPhotometricOutput,
}

#[pymethods]
impl PyRansacPhotometricOutput {
    #[getter]
    fn primary_mask<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        bool_array(py, &self.inner.primary_mask)
    }

    #[getter]
    fn secondary_mask<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        bool_array(py, &self.inner.secondary_mask)
    }

    #[getter]
    fn tile_primary_count<'py>(&self, py: Python<'py>) -> Py<PyAny> {
        PyArray1::from_slice(py, &self.inner.tile_primary_count)
            .into_any()
            .unbind()
    }

    #[getter]
    fn tile_secondary_count<'py>(&self, py: Python<'py>) -> Py<PyAny> {
        PyArray1::from_slice(py, &self.inner.tile_secondary_count)
            .into_any()
            .unbind()
    }

    #[getter]
    fn tile_primary_lum_mad<'py>(&self, py: Python<'py>) -> Py<PyAny> {
        PyArray1::from_slice(py, &self.inner.tile_primary_lum_mad)
            .into_any()
            .unbind()
    }

    #[getter]
    fn tile_secondary_lum_mad<'py>(&self, py: Python<'py>) -> Py<PyAny> {
        PyArray1::from_slice(py, &self.inner.tile_secondary_lum_mad)
            .into_any()
            .unbind()
    }

    fn __repr__(&self) -> String {
        format!(
            "RansacPhotometricOutput(primary_mask[{}], secondary_mask[{}], \
             tile_primary_count[{}])",
            self.inner.primary_mask.len(),
            self.inner.secondary_mask.len(),
            self.inner.tile_primary_count.len(),
        )
    }
}

fn bool_array<'py>(py: Python<'py>, data: &[bool]) -> PyResult<Py<PyAny>> {
    // numpy doesn't have a direct bool ndarray ctor in this binding; route
    // through u8 + .view(bool_).
    let bytes: Vec<u8> = data.iter().map(|&b| if b { 1 } else { 0 }).collect();
    let arr_u8 = PyArray1::from_vec(py, bytes);
    let np = py.import("numpy")?;
    let bool_dtype = np.getattr("bool_")?;
    let arr_bool = arr_u8.into_any().call_method1("view", (bool_dtype,))?;
    Ok(arr_bool.unbind())
}

/// Refine photometric agreement on ``stack`` per the spec.
///
/// The stack must be float-backed — either ``"float16"`` or ``"float32"``.
/// ``uint8`` stacks are rejected because the gamma exponentiation in Step 1
/// requires a floating-point representation. For ``"float16"`` storage the
/// chosen pyramid level is cast to f32 once internally (cheap: that level is
/// `target_patch_size = 4` by default, regardless of base patch size).
///
/// All keyword arguments mirror the spec's ``RansacPhotometricParams``;
/// defaults match the spec's recommendations.
///
/// Returns:
///     :class:`RansacPhotometricOutput`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(name = "refine_photometric_ransac", signature = (
    stack,
    inlier_threshold = 8.0,
    gamma = 1.0,
    target_patch_size = 4,
    scoring_patch_size = 2,
    subset_size = 2,
    max_subsets_per_tile = 64,
    min_inliers = 2,
    saturation_threshold = 254,
    seed = 0,
))]
pub fn refine_photometric_ransac_py(
    py: Python<'_>,
    stack: &PyPerSphericalTileSourceStack,
    inlier_threshold: f32,
    gamma: f32,
    target_patch_size: u32,
    scoring_patch_size: u32,
    subset_size: u32,
    max_subsets_per_tile: u32,
    min_inliers: u32,
    saturation_threshold: u8,
    seed: u64,
) -> PyResult<PyRansacPhotometricOutput> {
    let params = RansacPhotometricParams {
        inlier_threshold,
        gamma,
        target_patch_size,
        scoring_patch_size,
        subset_size,
        max_subsets_per_tile,
        min_inliers,
        saturation_threshold,
        seed,
        tile_index_base: 0,
    };
    let inner = match &stack.inner {
        Inner::F16(s) => py
            .detach(|| refine_photometric_ransac(s, &params))
            .map_err(err_to_py)?,
        Inner::F32(s) => py
            .detach(|| refine_photometric_ransac(s, &params))
            .map_err(err_to_py)?,
        Inner::U8(_) => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "refine_photometric_ransac requires a float16- or float32-backed stack; \
                 rebuild via PerSphericalTileSourceStack.build_rotation_only(\
                 ..., dtype=\"float16\" | \"float32\")",
            ));
        }
    };
    Ok(PyRansacPhotometricOutput { inner })
}
