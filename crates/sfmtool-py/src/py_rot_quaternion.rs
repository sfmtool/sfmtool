// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Python wrapper for the sfmtool-core RotQuaternion type.

use nalgebra::{Matrix3, Vector3};
use numpy::{IntoPyArray, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;

/// Extract a Vec<f64> from a Python object that is either a list or a 1D numpy array.
pub(crate) fn extract_f64_vec(obj: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    // Try numpy array first
    if let Ok(arr) = obj.extract::<numpy::PyReadonlyArray1<f64>>() {
        // Use as_slice for contiguous arrays, fall back to element-wise copy
        if let Ok(slice) = arr.as_slice() {
            return Ok(slice.to_vec());
        }
        // Non-contiguous array (e.g. column slice from 2D array): copy element-wise
        return Ok((0..arr.len()).map(|i| *arr.get(i).unwrap()).collect());
    }
    // Fall back to Python sequence (list, tuple, etc.)
    let seq: Vec<f64> = obj.extract().map_err(|_| {
        pyo3::exceptions::PyTypeError::new_err(
            "expected a list, tuple, or 1D numpy array of floats",
        )
    })?;
    Ok(seq)
}

/// Rotation quaternion for 3D rotations (WXYZ order).
#[pyclass(name = "RotQuaternion")]
#[derive(Clone)]
pub struct PyRotQuaternion {
    pub(crate) inner: sfmtool_core::RotQuaternion,
}

#[pymethods]
impl PyRotQuaternion {
    /// Create a quaternion from (w, x, y, z) components, normalizing to unit length.
    #[new]
    fn new(w: f64, x: f64, y: f64, z: f64) -> Self {
        Self {
            inner: sfmtool_core::RotQuaternion::new(w, x, y, z),
        }
    }

    /// Create the identity quaternion (no rotation).
    #[staticmethod]
    fn identity() -> Self {
        Self {
            inner: sfmtool_core::RotQuaternion::identity(),
        }
    }

    /// Create a quaternion from a [w, x, y, z] array (list or 1D numpy array).
    #[staticmethod]
    fn from_wxyz_array(arr: &Bound<'_, PyAny>) -> PyResult<Self> {
        let vals = extract_f64_vec(arr)?;
        if vals.len() != 4 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "expected array of length 4",
            ));
        }
        Ok(Self {
            inner: sfmtool_core::RotQuaternion::from_wxyz_array([
                vals[0], vals[1], vals[2], vals[3],
            ]),
        })
    }

    /// Create a quaternion from a 3x3 rotation matrix (numpy array).
    #[staticmethod]
    fn from_rotation_matrix(r: PyReadonlyArray2<f64>) -> PyResult<Self> {
        let shape = r.shape();
        if shape[0] != 3 || shape[1] != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "expected 3x3 matrix",
            ));
        }
        let mat = if let Ok(slice) = r.as_slice() {
            Matrix3::from_row_slice(slice)
        } else {
            // Non-contiguous array: copy element-wise (row-major)
            let mut data = [0.0f64; 9];
            for row in 0..3 {
                for col in 0..3 {
                    data[row * 3 + col] = *r.get([row, col]).ok_or_else(|| {
                        pyo3::exceptions::PyValueError::new_err("index out of bounds")
                    })?;
                }
            }
            Matrix3::from_row_slice(&data)
        };
        Ok(Self {
            inner: sfmtool_core::RotQuaternion::from_rotation_matrix(mat),
        })
    }

    /// Create a quaternion from an axis vector and angle in radians.
    #[staticmethod]
    fn from_axis_angle(axis: &Bound<'_, PyAny>, angle_rad: f64) -> PyResult<Self> {
        let vals = extract_f64_vec(axis)?;
        if vals.len() != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "expected axis of length 3",
            ));
        }
        let v = Vector3::new(vals[0], vals[1], vals[2]);
        let q = sfmtool_core::RotQuaternion::from_axis_angle(v, angle_rad)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(Self { inner: q })
    }

    /// The scalar (w) component.
    #[getter]
    fn w(&self) -> f64 {
        self.inner.w()
    }

    /// The first imaginary (x) component.
    #[getter]
    fn x(&self) -> f64 {
        self.inner.x()
    }

    /// The second imaginary (y) component.
    #[getter]
    fn y(&self) -> f64 {
        self.inner.y()
    }

    /// The third imaginary (z) component.
    #[getter]
    fn z(&self) -> f64 {
        self.inner.z()
    }

    /// Convert to a 3x3 rotation matrix (numpy array).
    fn to_rotation_matrix<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, numpy::PyArray2<f64>>> {
        let mat = self.inner.to_rotation_matrix();
        let data: Vec<Vec<f64>> = (0..3)
            .map(|r| (0..3).map(|c| mat[(r, c)]).collect())
            .collect();
        numpy::PyArray2::from_vec2(py, &data)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    /// Return as a [w, x, y, z] numpy array.
    fn to_wxyz_array<'py>(&self, py: Python<'py>) -> Bound<'py, numpy::PyArray1<f64>> {
        let arr = self.inner.to_wxyz_array();
        arr.to_vec().into_pyarray(py)
    }

    /// Convert to Euler angles, returning (roll, pitch, yaw) in radians.
    fn to_euler_angles(&self) -> (f64, f64, f64) {
        self.inner.to_euler_angles()
    }

    /// Compute the camera center in world coordinates from a world-to-camera pose.
    ///
    /// Given a world-to-camera transform where ``p_camera = R * p_world + t``,
    /// returns the camera center ``C = -R^T * t``.
    ///
    /// Args:
    ///     translation: 3-element list/array (the camera translation vector)
    ///
    /// Returns:
    ///     3-element numpy array of the camera center in world coordinates
    fn camera_center<'py>(
        &self,
        py: Python<'py>,
        translation: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
        let vals = extract_f64_vec(translation)?;
        if vals.len() != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "translation must have exactly 3 elements",
            ));
        }
        let t = Vector3::new(vals[0], vals[1], vals[2]);
        let center = self.inner.camera_center(&t);
        Ok(vec![center.x, center.y, center.z].into_pyarray(py))
    }

    /// Return the conjugate quaternion.
    fn conjugate(&self) -> Self {
        Self {
            inner: self.inner.conjugate(),
        }
    }

    /// Return the inverse rotation.
    fn inverse(&self) -> Self {
        Self {
            inner: self.inner.inverse(),
        }
    }

    /// Quaternion multiplication (composition of rotations).
    fn __mul__(&self, other: &PyRotQuaternion) -> Self {
        Self {
            inner: self.inner.clone() * other.inner.clone(),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "RotQuaternion(w={:.6}, x={:.6}, y={:.6}, z={:.6})",
            self.inner.w(),
            self.inner.x(),
            self.inner.y(),
            self.inner.z()
        )
    }

    fn __eq__(&self, other: &PyRotQuaternion) -> bool {
        self.inner == other.inner
    }
}
