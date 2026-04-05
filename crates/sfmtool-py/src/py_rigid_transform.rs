// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Python wrapper for the sfmtool-core RigidTransform type.

use nalgebra::Vector3;
use numpy::IntoPyArray;
use pyo3::prelude::*;

use crate::py_rot_quaternion::{extract_f64_vec, PyRotQuaternion};

/// Rigid body transformation: rotation + translation, no scale.
///
/// Transforms a point via: ``p' = R * p + t``
///
/// The rotation is a unit quaternion (det(R) = +1, no reflections).
/// Distances are preserved. For transforms with uniform scaling,
/// use :class:`SE3Transform`.
#[pyclass(name = "RigidTransform")]
#[derive(Clone)]
pub struct PyRigidTransform {
    pub(crate) inner: sfmtool_core::RigidTransform,
}

#[pymethods]
impl PyRigidTransform {
    /// Create a new rigid transform.
    ///
    /// Args:
    ///     rotation: RotQuaternion (or None for identity rotation)
    ///     translation: 3-element list/array (or None for zero translation)
    #[new]
    #[pyo3(signature = (rotation=None, translation=None))]
    fn new(
        rotation: Option<&Bound<'_, PyAny>>,
        translation: Option<&Bound<'_, PyAny>>,
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
            inner: sfmtool_core::RigidTransform::new(rot, trans),
        })
    }

    /// Create the identity transform (no rotation, no translation).
    #[staticmethod]
    fn identity() -> Self {
        Self {
            inner: sfmtool_core::RigidTransform::identity(),
        }
    }

    /// Create from raw WXYZ quaternion array and translation array.
    #[staticmethod]
    fn from_wxyz_translation(
        quaternion_wxyz: &Bound<'_, PyAny>,
        translation: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let q = extract_f64_vec(quaternion_wxyz)?;
        if q.len() != 4 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "quaternion must have exactly 4 elements",
            ));
        }
        let t = extract_f64_vec(translation)?;
        if t.len() != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "translation must have exactly 3 elements",
            ));
        }
        Ok(Self {
            inner: sfmtool_core::RigidTransform::from_wxyz_translation(
                [q[0], q[1], q[2], q[3]],
                [t[0], t[1], t[2]],
            ),
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

    /// Convert rotation to a 3x3 rotation matrix (numpy array).
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

    /// Compute the inverse translation origin: -R^T * t.
    ///
    /// When used as a world-to-camera transform, this gives the camera
    /// center in world coordinates.
    fn inverse_translation_origin<'py>(&self, py: Python<'py>) -> Bound<'py, numpy::PyArray1<f64>> {
        let c = self.inner.inverse_translation_origin();
        vec![c.x, c.y, c.z].into_pyarray(py)
    }

    /// Apply the transform to a 3D point: R * p + t.
    fn transform_point<'py>(
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
        let p = nalgebra::Point3::new(vals[0], vals[1], vals[2]);
        let result = self.inner.transform_point(&p);
        Ok(vec![result.x, result.y, result.z].into_pyarray(py))
    }

    fn __repr__(&self) -> String {
        let t = &self.inner.translation;
        let q = &self.inner.rotation;
        format!(
            "RigidTransform(rotation=RotQuaternion(w={:.6}, x={:.6}, y={:.6}, z={:.6}), translation=[{:.6}, {:.6}, {:.6}])",
            q.w(), q.x(), q.y(), q.z(),
            t.x, t.y, t.z,
        )
    }

    fn __eq__(&self, other: &PyRigidTransform) -> bool {
        self.inner == other.inner
    }
}