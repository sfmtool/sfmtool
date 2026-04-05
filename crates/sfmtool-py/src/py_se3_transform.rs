// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Python wrapper for the sfmtool-core Se3Transform type.

use nalgebra::{Point3, Vector3};
use numpy::{IntoPyArray, PyArrayMethods, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::py_rot_quaternion::{extract_f64_vec, PyRotQuaternion};

/// SE(3) similarity transform: rotation, translation, and uniform scale.
///
/// Applies as: ``p' = scale * (R * p) + t``
#[pyclass(name = "SE3Transform")]
#[derive(Clone)]
pub struct PySe3Transform {
    pub(crate) inner: sfmtool_core::Se3Transform,
}

#[pymethods]
impl PySe3Transform {
    /// Create a new SE(3) transform.
    ///
    /// Args:
    ///     rotation: RotQuaternion (or None for identity rotation)
    ///     translation: 3-element list/array (or None for zero translation)
    ///     scale: uniform scale factor (default 1.0)
    #[new]
    #[pyo3(signature = (rotation=None, translation=None, scale=1.0))]
    fn new(
        rotation: Option<&Bound<'_, PyAny>>,
        translation: Option<&Bound<'_, PyAny>>,
        scale: f64,
    ) -> PyResult<Self> {
        let rot = match rotation {
            Some(obj) => {
                let pyq: PyRotQuaternion = obj.extract().map_err(|_| {
                    pyo3::exceptions::PyTypeError::new_err(
                        "rotation must be a RotQuaternion instance or None",
                    )
                })?;
                pyq.inner
            }
            None => sfmtool_core::RotQuaternion::identity(),
        };

        let trans = match translation {
            Some(obj) => {
                let vals = extract_f64_vec(obj)?;
                if vals.len() != 3 {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "translation must have exactly 3 elements",
                    ));
                }
                Vector3::new(vals[0], vals[1], vals[2])
            }
            None => Vector3::zeros(),
        };

        Ok(Self {
            inner: sfmtool_core::Se3Transform::new(rot, trans, scale),
        })
    }

    /// Create the identity transform (no rotation, no translation, scale=1).
    #[staticmethod]
    fn identity() -> Self {
        Self {
            inner: sfmtool_core::Se3Transform::identity(),
        }
    }

    /// Create a rotation-only transform from an axis and angle (radians).
    #[staticmethod]
    fn from_axis_angle(axis: &Bound<'_, PyAny>, angle_rad: f64) -> PyResult<Self> {
        let vals = extract_f64_vec(axis)?;
        if vals.len() != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "expected axis of length 3",
            ));
        }
        let v = Vector3::new(vals[0], vals[1], vals[2]);
        let t = sfmtool_core::Se3Transform::from_axis_angle(v, angle_rad)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(Self { inner: t })
    }

    /// Create from a dictionary ``{rotation: {w, x, y, z}, translation: [...], scale: ...}``.
    #[classmethod]
    fn from_dict(_cls: &Bound<'_, pyo3::types::PyType>, d: &Bound<'_, PyDict>) -> PyResult<Self> {
        let rot_obj = d
            .get_item("rotation")?
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("missing 'rotation' key"))?;
        let rot_dict: &Bound<'_, PyDict> = rot_obj
            .downcast()
            .map_err(|_| pyo3::exceptions::PyTypeError::new_err("'rotation' must be a dict"))?;

        let w: f64 = rot_dict
            .get_item("w")?
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("missing 'w'"))?
            .extract()?;
        let x: f64 = rot_dict
            .get_item("x")?
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("missing 'x'"))?
            .extract()?;
        let y: f64 = rot_dict
            .get_item("y")?
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("missing 'y'"))?
            .extract()?;
        let z: f64 = rot_dict
            .get_item("z")?
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("missing 'z'"))?
            .extract()?;

        let rot = sfmtool_core::RotQuaternion::new(w, x, y, z);

        let trans_obj = d
            .get_item("translation")?
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("missing 'translation' key"))?;
        let trans_vals = extract_f64_vec(&trans_obj)?;
        if trans_vals.len() != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "translation must have exactly 3 elements",
            ));
        }
        let translation = Vector3::new(trans_vals[0], trans_vals[1], trans_vals[2]);

        let scale: f64 = match d.get_item("scale")? {
            Some(s) => s.extract()?,
            None => 1.0,
        };

        Ok(Self {
            inner: sfmtool_core::Se3Transform::new(rot, translation, scale),
        })
    }

    /// The rotation component as a RotQuaternion.
    #[getter]
    fn rotation(&self) -> PyRotQuaternion {
        PyRotQuaternion {
            inner: self.inner.rotation.clone(),
        }
    }

    /// The translation component as a 3-element numpy array.
    #[getter]
    fn get_translation<'py>(&self, py: Python<'py>) -> Bound<'py, numpy::PyArray1<f64>> {
        let t = &self.inner.translation;
        vec![t.x, t.y, t.z].into_pyarray(py)
    }

    /// Set the translation component from a list/array.
    #[setter]
    fn set_translation(&mut self, obj: &Bound<'_, PyAny>) -> PyResult<()> {
        let vals = extract_f64_vec(obj)?;
        if vals.len() != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "translation must have exactly 3 elements",
            ));
        }
        self.inner.translation = Vector3::new(vals[0], vals[1], vals[2]);
        Ok(())
    }

    /// The uniform scale factor.
    #[getter]
    fn get_scale(&self) -> f64 {
        self.inner.scale
    }

    /// Set the uniform scale factor.
    #[setter]
    fn set_scale(&mut self, value: f64) {
        self.inner.scale = value;
    }

    /// Apply the transform to a single 3D point.
    ///
    /// Args:
    ///     point: 3-element list/array
    ///
    /// Returns:
    ///     3-element numpy array
    fn apply_to_point<'py>(
        &self,
        py: Python<'py>,
        point: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
        let vals = extract_f64_vec(point)?;
        if vals.len() != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "point must have exactly 3 elements",
            ));
        }
        let p = Point3::new(vals[0], vals[1], vals[2]);
        let result = self.inner.apply_to_point(&p);
        Ok(vec![result.x, result.y, result.z].into_pyarray(py))
    }

    /// Apply the transform to an (N, 3) array of 3D points.
    ///
    /// Args:
    ///     points: (N, 3) numpy array
    ///
    /// Returns:
    ///     (N, 3) numpy array
    fn apply_to_points<'py>(
        &self,
        py: Python<'py>,
        points: PyReadonlyArray2<f64>,
    ) -> PyResult<Bound<'py, numpy::PyArray2<f64>>> {
        let shape = points.shape();
        if shape[1] != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "points must have shape (N, 3)",
            ));
        }
        let n = shape[0];
        let slice = points.as_slice().map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("array not contiguous: {e}"))
        })?;

        let pts: Vec<Point3<f64>> = (0..n)
            .map(|i| {
                let off = i * 3;
                Point3::new(slice[off], slice[off + 1], slice[off + 2])
            })
            .collect();

        let results = self.inner.apply_to_points(&pts);

        let mut out = Vec::with_capacity(n * 3);
        for p in &results {
            out.push(p.x);
            out.push(p.y);
            out.push(p.z);
        }

        let arr = numpy::PyArray1::from_vec(py, out);
        Ok(arr.reshape([n, 3])?.to_owned())
    }

    /// Apply the transform to camera poses given as flat numpy arrays.
    ///
    /// Args:
    ///     quaternions_wxyz: (N, 4) numpy array of quaternions in WXYZ order
    ///     translations: (N, 3) numpy array of translations
    ///
    /// Returns:
    ///     Tuple of ((N, 4) numpy array, (N, 3) numpy array)
    #[allow(clippy::type_complexity)]
    fn apply_to_camera_poses<'py>(
        &self,
        py: Python<'py>,
        quaternions_wxyz: PyReadonlyArray2<f64>,
        translations: PyReadonlyArray2<f64>,
    ) -> PyResult<(
        Bound<'py, numpy::PyArray2<f64>>,
        Bound<'py, numpy::PyArray2<f64>>,
    )> {
        let q_shape = quaternions_wxyz.shape();
        let t_shape = translations.shape();

        if q_shape[1] != 4 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "quaternions must have shape (N, 4)",
            ));
        }
        if t_shape[1] != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "translations must have shape (N, 3)",
            ));
        }
        if q_shape[0] != t_shape[0] {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "quaternions and translations must have the same number of rows",
            ));
        }

        let n = q_shape[0];
        let q_in = quaternions_wxyz.as_slice().map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("quaternions not contiguous: {e}"))
        })?;
        let t_in = translations.as_slice().map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("translations not contiguous: {e}"))
        })?;

        let mut q_out = vec![0.0f64; n * 4];
        let mut t_out = vec![0.0f64; n * 3];

        self.inner
            .apply_to_camera_poses_flat(q_in, t_in, &mut q_out, &mut t_out);

        let q_arr = numpy::PyArray1::from_vec(py, q_out);
        let t_arr = numpy::PyArray1::from_vec(py, t_out);

        Ok((
            q_arr.reshape([n, 4])?.to_owned(),
            t_arr.reshape([n, 3])?.to_owned(),
        ))
    }

    /// Compose two transforms: apply self first, then other.
    fn compose(&self, other: &PySe3Transform) -> Self {
        Self {
            inner: self.inner.compose(&other.inner),
        }
    }

    /// Compute the inverse transform.
    ///
    /// Raises:
    ///     ValueError: If scale is zero.
    fn inverse(&self) -> PyResult<Self> {
        let inv = self
            .inner
            .inverse()
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(Self { inner: inv })
    }

    /// Convert to a dictionary representation.
    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);

        let rot_dict = PyDict::new(py);
        rot_dict.set_item("w", self.inner.rotation.w())?;
        rot_dict.set_item("x", self.inner.rotation.x())?;
        rot_dict.set_item("y", self.inner.rotation.y())?;
        rot_dict.set_item("z", self.inner.rotation.z())?;
        dict.set_item("rotation", rot_dict)?;

        let t = &self.inner.translation;
        let trans_list: Vec<f64> = vec![t.x, t.y, t.z];
        dict.set_item("translation", trans_list)?;

        dict.set_item("scale", self.inner.scale)?;

        Ok(dict)
    }

    /// Compose (SE3Transform) or apply to points (numpy array) via @.
    fn __matmul__<'py>(&self, py: Python<'py>, other: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
        // Try SE3Transform first
        if let Ok(other_t) = other.extract::<PySe3Transform>() {
            let composed = self.inner.compose(&other_t.inner);
            return Ok(PySe3Transform { inner: composed }
                .into_pyobject(py)?
                .into_any()
                .unbind());
        }

        // Try 2D numpy array (batch of points)
        if let Ok(arr) = other.extract::<numpy::PyReadonlyArray2<f64>>() {
            let shape = arr.shape();
            if shape[1] != 3 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Array must have shape (N, 3)",
                ));
            }
            let n = shape[0];
            let slice = arr.as_slice().map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!("array not contiguous: {e}"))
            })?;

            let pts: Vec<Point3<f64>> = (0..n)
                .map(|i| {
                    let off = i * 3;
                    Point3::new(slice[off], slice[off + 1], slice[off + 2])
                })
                .collect();

            let results = self.inner.apply_to_points(&pts);
            let mut out = Vec::with_capacity(n * 3);
            for p in &results {
                out.push(p.x);
                out.push(p.y);
                out.push(p.z);
            }

            let flat = numpy::PyArray1::from_vec(py, out);
            let reshaped = flat.reshape([n, 3])?;
            return Ok(reshaped.to_owned().into_any().unbind());
        }

        // Try 1D array (single point)
        if let Ok(arr) = other.extract::<numpy::PyReadonlyArray1<f64>>() {
            let slice = arr.as_slice().map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!("array not contiguous: {e}"))
            })?;
            if slice.len() != 3 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Array must have shape (3,) or (N, 3)",
                ));
            }
            let p = Point3::new(slice[0], slice[1], slice[2]);
            let result = self.inner.apply_to_point(&p);
            let out = vec![result.x, result.y, result.z].into_pyarray(py);
            return Ok(out.into_any().unbind());
        }

        Err(pyo3::exceptions::PyTypeError::new_err(
            "unsupported operand type for @: expected SE3Transform or numpy array",
        ))
    }

    fn __repr__(&self) -> String {
        let t = &self.inner.translation;
        format!(
            "SE3Transform(rotation={}, translation=[{:.6}, {:.6}, {:.6}], scale={:.6})",
            {
                let q = &self.inner.rotation;
                format!(
                    "RotQuaternion(w={:.6}, x={:.6}, y={:.6}, z={:.6})",
                    q.w(),
                    q.x(),
                    q.y(),
                    q.z()
                )
            },
            t.x,
            t.y,
            t.z,
            self.inner.scale,
        )
    }

    fn __eq__(&self, other: &PySe3Transform) -> bool {
        self.inner.rotation == other.inner.rotation
            && (self.inner.translation - other.inner.translation).norm() < 1e-12
            && (self.inner.scale - other.inner.scale).abs() < 1e-12
    }
}