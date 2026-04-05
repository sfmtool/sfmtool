// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for 2D and 3D KD-tree spatial queries.
//!
//! Each Python class (`KdTree2d`, `KdTree3d`) automatically selects an f32 or
//! f64 backing tree based on the dtype of the positions array passed to the
//! constructor.  Query arrays must match the tree's dtype.

use numpy::{PyArrayMethods, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use std::borrow::Cow;

use sfmtool_core::spatial::{PointCloud2, PointCloud3};

// ── Helper: dtype tag ───────────────────────────────────────────────────

/// Extract a simple dtype tag from a numpy array's dtype string.
fn dtype_tag(_py: Python<'_>, obj: &Bound<'_, pyo3::types::PyAny>) -> PyResult<String> {
    let dtype = obj.getattr("dtype")?;
    let name: String = dtype.getattr("name")?.extract()?;
    Ok(name)
}

// ── 2D KD-tree ──────────────────────────────────────────────────────────

enum Inner2d {
    F32(PointCloud2<f32>),
    F64(PointCloud2<f64>),
}

/// A 2D KD-tree for spatial point queries.
///
/// Build once from an (N, 2) float32 or float64 array, then issue batch queries.
/// The dtype of the positions array determines the internal precision; query
/// arrays must use the same dtype.
#[pyclass(name = "KdTree2d")]
pub struct PyKdTree2d {
    inner: Inner2d,
}

#[pymethods]
impl PyKdTree2d {
    /// Build a KD-tree from an (N, 2) float32 or float64 array of positions.
    #[new]
    fn new(py: Python<'_>, positions: &Bound<'_, pyo3::types::PyAny>) -> PyResult<Self> {
        let tag = dtype_tag(py, positions)?;
        match tag.as_str() {
            "float32" => {
                let arr: PyReadonlyArray2<f32> = positions.extract()?;
                let shape = arr.shape();
                if shape[1] != 2 {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "expected (N, 2) array, got (N, {})",
                        shape[1]
                    )));
                }
                let n = shape[0];
                let data: Cow<[f32]> = match arr.as_slice() {
                    Ok(s) => Cow::Borrowed(s),
                    Err(_) => Cow::Owned(arr.as_array().iter().copied().collect()),
                };
                Ok(Self {
                    inner: Inner2d::F32(PointCloud2::<f32>::new(&data, n)),
                })
            }
            "float64" => {
                let arr: PyReadonlyArray2<f64> = positions.extract()?;
                let shape = arr.shape();
                if shape[1] != 2 {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "expected (N, 2) array, got (N, {})",
                        shape[1]
                    )));
                }
                let n = shape[0];
                let data: Cow<[f64]> = match arr.as_slice() {
                    Ok(s) => Cow::Borrowed(s),
                    Err(_) => Cow::Owned(arr.as_array().iter().copied().collect()),
                };
                Ok(Self {
                    inner: Inner2d::F64(PointCloud2::<f64>::new(&data, n)),
                })
            }
            other => Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "expected float32 or float64 array, got {other}"
            ))),
        }
    }

    /// Number of points in the tree.
    #[getter]
    fn len(&self) -> usize {
        match &self.inner {
            Inner2d::F32(c) => c.len(),
            Inner2d::F64(c) => c.len(),
        }
    }

    /// The numpy dtype string of the tree ("float32" or "float64").
    #[getter]
    fn dtype(&self) -> &'static str {
        match &self.inner {
            Inner2d::F32(_) => "float32",
            Inner2d::F64(_) => "float64",
        }
    }

    /// Find the nearest point index for each query point.
    ///
    /// Args:
    ///     query_points: (M, 2) array matching the tree's dtype.
    ///
    /// Returns:
    ///     (M,) uint32 array of nearest point indices.
    fn nearest<'py>(
        &self,
        py: Python<'py>,
        query_points: &Bound<'py, pyo3::types::PyAny>,
    ) -> PyResult<Py<PyAny>> {
        match &self.inner {
            Inner2d::F32(cloud) => {
                let arr: PyReadonlyArray2<f32> = query_points.extract()?;
                check_dim(&arr, 2)?;
                let n = arr.shape()[0];
                let data = to_cow_slice(&arr);
                let result = cloud.nearest(&data, n);
                Ok(numpy::PyArray1::from_vec(py, result).into_any().unbind())
            }
            Inner2d::F64(cloud) => {
                let arr: PyReadonlyArray2<f64> = query_points.extract()?;
                check_dim(&arr, 2)?;
                let n = arr.shape()[0];
                let data = to_cow_slice(&arr);
                let result = cloud.nearest(&data, n);
                Ok(numpy::PyArray1::from_vec(py, result).into_any().unbind())
            }
        }
    }

    /// Find the nearest K point indices for each query point.
    ///
    /// Args:
    ///     query_points: (M, 2) array matching the tree's dtype.
    ///     k: Number of nearest neighbors to find.
    ///
    /// Returns:
    ///     (M, K) uint32 array of point indices, ordered by distance.
    ///     Slots beyond the available neighbors are filled with 2**32 - 1.
    fn nearest_k<'py>(
        &self,
        py: Python<'py>,
        query_points: &Bound<'py, pyo3::types::PyAny>,
        k: usize,
    ) -> PyResult<Py<PyAny>> {
        match &self.inner {
            Inner2d::F32(cloud) => {
                let arr: PyReadonlyArray2<f32> = query_points.extract()?;
                check_dim(&arr, 2)?;
                let n = arr.shape()[0];
                let data = to_cow_slice(&arr);
                let flat = cloud.nearest_k(&data, n, k);
                let out = numpy::PyArray1::from_vec(py, flat).reshape([n, k])?;
                Ok(out.into_any().unbind())
            }
            Inner2d::F64(cloud) => {
                let arr: PyReadonlyArray2<f64> = query_points.extract()?;
                check_dim(&arr, 2)?;
                let n = arr.shape()[0];
                let data = to_cow_slice(&arr);
                let flat = cloud.nearest_k(&data, n, k);
                let out = numpy::PyArray1::from_vec(py, flat).reshape([n, k])?;
                Ok(out.into_any().unbind())
            }
        }
    }

    /// Find up to K nearest point indices within a radius for each query point.
    ///
    /// Combines nearest_k and within_radius: returns the closest K points that
    /// are also within the given Euclidean distance. Useful for finding spatial
    /// candidates in dense point sets where you want both a count limit and a
    /// distance limit.
    ///
    /// Args:
    ///     query_points: (M, 2) array matching the tree's dtype.
    ///     k: Maximum number of nearest neighbors to find.
    ///     radius: Maximum Euclidean distance.
    ///
    /// Returns:
    ///     (M, K) uint32 array of point indices, ordered by distance.
    ///     Slots beyond the available neighbors within radius are filled with 2**32 - 1.
    fn nearest_k_within_radius<'py>(
        &self,
        py: Python<'py>,
        query_points: &Bound<'py, pyo3::types::PyAny>,
        k: usize,
        radius: f64,
    ) -> PyResult<Py<PyAny>> {
        match &self.inner {
            Inner2d::F32(cloud) => {
                let arr: PyReadonlyArray2<f32> = query_points.extract()?;
                check_dim(&arr, 2)?;
                let n = arr.shape()[0];
                let data = to_cow_slice(&arr);
                let flat = py.detach(|| cloud.nearest_k_within_radius(&data, n, k, radius as f32));
                let out = numpy::PyArray1::from_vec(py, flat).reshape([n, k])?;
                Ok(out.into_any().unbind())
            }
            Inner2d::F64(cloud) => {
                let arr: PyReadonlyArray2<f64> = query_points.extract()?;
                check_dim(&arr, 2)?;
                let n = arr.shape()[0];
                let data = to_cow_slice(&arr);
                let flat = py.detach(|| cloud.nearest_k_within_radius(&data, n, k, radius));
                let out = numpy::PyArray1::from_vec(py, flat).reshape([n, k])?;
                Ok(out.into_any().unbind())
            }
        }
    }

    /// Find all points within a Euclidean radius of each query point.
    ///
    /// Args:
    ///     query_points: (M, 2) array matching the tree's dtype.
    ///     radius: Search radius (Euclidean distance).
    ///
    /// Returns:
    ///     Tuple (offsets, indices):
    ///     - offsets: (M+1,) uint32 array with offsets[0] == 0, offsets[-1] == R.
    ///     - indices: (R,) uint32 array of point indices.
    ///     Results for query i are indices[offsets[i]:offsets[i+1]].
    fn within_radius<'py>(
        &self,
        py: Python<'py>,
        query_points: &Bound<'py, pyo3::types::PyAny>,
        radius: f64,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        match &self.inner {
            Inner2d::F32(cloud) => {
                let arr: PyReadonlyArray2<f32> = query_points.extract()?;
                check_dim(&arr, 2)?;
                let n = arr.shape()[0];
                let data = to_cow_slice(&arr);
                let (offsets, indices) = cloud.within_radius(&data, n, radius as f32);
                Ok((
                    numpy::PyArray1::from_vec(py, offsets).into_any().unbind(),
                    numpy::PyArray1::from_vec(py, indices).into_any().unbind(),
                ))
            }
            Inner2d::F64(cloud) => {
                let arr: PyReadonlyArray2<f64> = query_points.extract()?;
                check_dim(&arr, 2)?;
                let n = arr.shape()[0];
                let data = to_cow_slice(&arr);
                let (offsets, indices) = cloud.within_radius(&data, n, radius);
                Ok((
                    numpy::PyArray1::from_vec(py, offsets).into_any().unbind(),
                    numpy::PyArray1::from_vec(py, indices).into_any().unbind(),
                ))
            }
        }
    }

    /// Find the nearest K neighbors (excluding self) for every point in the tree.
    ///
    /// Args:
    ///     k: Number of nearest neighbors per point (default 1).
    ///
    /// Returns:
    ///     (N, K) uint32 array of point indices.
    ///     Slots beyond the available neighbors are filled with 2**32 - 1.
    #[pyo3(signature = (k=1))]
    fn self_nearest_k<'py>(&self, py: Python<'py>, k: usize) -> PyResult<Py<PyAny>> {
        match &self.inner {
            Inner2d::F32(cloud) => {
                let n = cloud.len();
                let flat = cloud.self_nearest_k(k);
                let out = numpy::PyArray1::from_vec(py, flat).reshape([n, k])?;
                Ok(out.into_any().unbind())
            }
            Inner2d::F64(cloud) => {
                let n = cloud.len();
                let flat = cloud.self_nearest_k(k);
                let out = numpy::PyArray1::from_vec(py, flat).reshape([n, k])?;
                Ok(out.into_any().unbind())
            }
        }
    }

    /// Compute the nearest-neighbor Euclidean distance for each point.
    ///
    /// Returns:
    ///     (N,) float32 or float64 array of NN distances (matches tree dtype).
    ///     Single-point trees return `[inf]`.
    fn nearest_neighbor_distances<'py>(&self, py: Python<'py>) -> Py<PyAny> {
        match &self.inner {
            Inner2d::F32(cloud) => {
                numpy::PyArray1::from_vec(py, cloud.nearest_neighbor_distances())
                    .into_any()
                    .unbind()
            }
            Inner2d::F64(cloud) => {
                numpy::PyArray1::from_vec(py, cloud.nearest_neighbor_distances())
                    .into_any()
                    .unbind()
            }
        }
    }
}

// ── 3D KD-tree ──────────────────────────────────────────────────────────

enum Inner3d {
    F32(PointCloud3<f32>),
    F64(PointCloud3<f64>),
}

/// A 3D KD-tree for spatial point queries.
///
/// Build once from an (N, 3) float32 or float64 array, then issue batch queries.
/// The dtype of the positions array determines the internal precision; query
/// arrays must use the same dtype.
#[pyclass(name = "KdTree3d")]
pub struct PyKdTree3d {
    inner: Inner3d,
}

#[pymethods]
impl PyKdTree3d {
    /// Build a KD-tree from an (N, 3) float32 or float64 array of positions.
    #[new]
    fn new(py: Python<'_>, positions: &Bound<'_, pyo3::types::PyAny>) -> PyResult<Self> {
        let tag = dtype_tag(py, positions)?;
        match tag.as_str() {
            "float32" => {
                let arr: PyReadonlyArray2<f32> = positions.extract()?;
                let shape = arr.shape();
                if shape[1] != 3 {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "expected (N, 3) array, got (N, {})",
                        shape[1]
                    )));
                }
                let n = shape[0];
                let data: Cow<[f32]> = match arr.as_slice() {
                    Ok(s) => Cow::Borrowed(s),
                    Err(_) => Cow::Owned(arr.as_array().iter().copied().collect()),
                };
                Ok(Self {
                    inner: Inner3d::F32(PointCloud3::<f32>::new(&data, n)),
                })
            }
            "float64" => {
                let arr: PyReadonlyArray2<f64> = positions.extract()?;
                let shape = arr.shape();
                if shape[1] != 3 {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "expected (N, 3) array, got (N, {})",
                        shape[1]
                    )));
                }
                let n = shape[0];
                let data: Cow<[f64]> = match arr.as_slice() {
                    Ok(s) => Cow::Borrowed(s),
                    Err(_) => Cow::Owned(arr.as_array().iter().copied().collect()),
                };
                Ok(Self {
                    inner: Inner3d::F64(PointCloud3::<f64>::new(&data, n)),
                })
            }
            other => Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "expected float32 or float64 array, got {other}"
            ))),
        }
    }

    /// Number of points in the tree.
    #[getter]
    fn len(&self) -> usize {
        match &self.inner {
            Inner3d::F32(c) => c.len(),
            Inner3d::F64(c) => c.len(),
        }
    }

    /// The numpy dtype string of the tree ("float32" or "float64").
    #[getter]
    fn dtype(&self) -> &'static str {
        match &self.inner {
            Inner3d::F32(_) => "float32",
            Inner3d::F64(_) => "float64",
        }
    }

    /// Find the nearest point index for each query point.
    fn nearest<'py>(
        &self,
        py: Python<'py>,
        query_points: &Bound<'py, pyo3::types::PyAny>,
    ) -> PyResult<Py<PyAny>> {
        match &self.inner {
            Inner3d::F32(cloud) => {
                let arr: PyReadonlyArray2<f32> = query_points.extract()?;
                check_dim(&arr, 3)?;
                let n = arr.shape()[0];
                let data = to_cow_slice(&arr);
                let result = cloud.nearest(&data, n);
                Ok(numpy::PyArray1::from_vec(py, result).into_any().unbind())
            }
            Inner3d::F64(cloud) => {
                let arr: PyReadonlyArray2<f64> = query_points.extract()?;
                check_dim(&arr, 3)?;
                let n = arr.shape()[0];
                let data = to_cow_slice(&arr);
                let result = cloud.nearest(&data, n);
                Ok(numpy::PyArray1::from_vec(py, result).into_any().unbind())
            }
        }
    }

    /// Find the nearest K point indices for each query point.
    fn nearest_k<'py>(
        &self,
        py: Python<'py>,
        query_points: &Bound<'py, pyo3::types::PyAny>,
        k: usize,
    ) -> PyResult<Py<PyAny>> {
        match &self.inner {
            Inner3d::F32(cloud) => {
                let arr: PyReadonlyArray2<f32> = query_points.extract()?;
                check_dim(&arr, 3)?;
                let n = arr.shape()[0];
                let data = to_cow_slice(&arr);
                let flat = cloud.nearest_k(&data, n, k);
                let out = numpy::PyArray1::from_vec(py, flat).reshape([n, k])?;
                Ok(out.into_any().unbind())
            }
            Inner3d::F64(cloud) => {
                let arr: PyReadonlyArray2<f64> = query_points.extract()?;
                check_dim(&arr, 3)?;
                let n = arr.shape()[0];
                let data = to_cow_slice(&arr);
                let flat = cloud.nearest_k(&data, n, k);
                let out = numpy::PyArray1::from_vec(py, flat).reshape([n, k])?;
                Ok(out.into_any().unbind())
            }
        }
    }

    /// Find up to K nearest point indices within a radius for each query point.
    ///
    /// Args:
    ///     query_points: (M, 3) array matching the tree's dtype.
    ///     k: Maximum number of nearest neighbors to find.
    ///     radius: Maximum Euclidean distance.
    ///
    /// Returns:
    ///     (M, K) uint32 array of point indices, ordered by distance.
    ///     Slots beyond the available neighbors within radius are filled with 2**32 - 1.
    fn nearest_k_within_radius<'py>(
        &self,
        py: Python<'py>,
        query_points: &Bound<'py, pyo3::types::PyAny>,
        k: usize,
        radius: f64,
    ) -> PyResult<Py<PyAny>> {
        match &self.inner {
            Inner3d::F32(cloud) => {
                let arr: PyReadonlyArray2<f32> = query_points.extract()?;
                check_dim(&arr, 3)?;
                let n = arr.shape()[0];
                let data = to_cow_slice(&arr);
                let flat = cloud.nearest_k_within_radius(&data, n, k, radius as f32);
                let out = numpy::PyArray1::from_vec(py, flat).reshape([n, k])?;
                Ok(out.into_any().unbind())
            }
            Inner3d::F64(cloud) => {
                let arr: PyReadonlyArray2<f64> = query_points.extract()?;
                check_dim(&arr, 3)?;
                let n = arr.shape()[0];
                let data = to_cow_slice(&arr);
                let flat = cloud.nearest_k_within_radius(&data, n, k, radius);
                let out = numpy::PyArray1::from_vec(py, flat).reshape([n, k])?;
                Ok(out.into_any().unbind())
            }
        }
    }

    /// Find all points within a Euclidean radius of each query point.
    fn within_radius<'py>(
        &self,
        py: Python<'py>,
        query_points: &Bound<'py, pyo3::types::PyAny>,
        radius: f64,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        match &self.inner {
            Inner3d::F32(cloud) => {
                let arr: PyReadonlyArray2<f32> = query_points.extract()?;
                check_dim(&arr, 3)?;
                let n = arr.shape()[0];
                let data = to_cow_slice(&arr);
                let (offsets, indices) = cloud.within_radius(&data, n, radius as f32);
                Ok((
                    numpy::PyArray1::from_vec(py, offsets).into_any().unbind(),
                    numpy::PyArray1::from_vec(py, indices).into_any().unbind(),
                ))
            }
            Inner3d::F64(cloud) => {
                let arr: PyReadonlyArray2<f64> = query_points.extract()?;
                check_dim(&arr, 3)?;
                let n = arr.shape()[0];
                let data = to_cow_slice(&arr);
                let (offsets, indices) = cloud.within_radius(&data, n, radius);
                Ok((
                    numpy::PyArray1::from_vec(py, offsets).into_any().unbind(),
                    numpy::PyArray1::from_vec(py, indices).into_any().unbind(),
                ))
            }
        }
    }

    /// Find the nearest K neighbors (excluding self) for every point in the tree.
    #[pyo3(signature = (k=1))]
    fn self_nearest_k<'py>(&self, py: Python<'py>, k: usize) -> PyResult<Py<PyAny>> {
        match &self.inner {
            Inner3d::F32(cloud) => {
                let n = cloud.len();
                let flat = cloud.self_nearest_k(k);
                let out = numpy::PyArray1::from_vec(py, flat).reshape([n, k])?;
                Ok(out.into_any().unbind())
            }
            Inner3d::F64(cloud) => {
                let n = cloud.len();
                let flat = cloud.self_nearest_k(k);
                let out = numpy::PyArray1::from_vec(py, flat).reshape([n, k])?;
                Ok(out.into_any().unbind())
            }
        }
    }

    /// Compute the nearest-neighbor Euclidean distance for each point.
    ///
    /// Returns:
    ///     (N,) float32 or float64 array of NN distances (matches tree dtype).
    ///     Single-point trees return `[inf]`.
    fn nearest_neighbor_distances<'py>(&self, py: Python<'py>) -> Py<PyAny> {
        match &self.inner {
            Inner3d::F32(cloud) => {
                numpy::PyArray1::from_vec(py, cloud.nearest_neighbor_distances())
                    .into_any()
                    .unbind()
            }
            Inner3d::F64(cloud) => {
                numpy::PyArray1::from_vec(py, cloud.nearest_neighbor_distances())
                    .into_any()
                    .unbind()
            }
        }
    }
}

// ── Shared helpers ──────────────────────────────────────────────────────

/// Check that the second dimension of a 2D array matches `expected_dim`.
fn check_dim<T: numpy::Element>(arr: &PyReadonlyArray2<T>, expected_dim: usize) -> PyResult<()> {
    let actual = arr.shape()[1];
    if actual != expected_dim {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "expected (M, {expected_dim}) array, got (M, {actual})"
        )));
    }
    Ok(())
}

/// Zero-copy slice if contiguous, otherwise copy.
fn to_cow_slice<'a, T: numpy::Element + Copy>(arr: &'a PyReadonlyArray2<'a, T>) -> Cow<'a, [T]> {
    match arr.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(arr.as_array().iter().copied().collect()),
    }
}
