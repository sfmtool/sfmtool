// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for the coordinate-convention conversion primitives
//! (COLMAP/OpenCV ↔ canonical Z-up / −Z-forward; see
//! `sfmtool_core::geometry::convention` and
//! `specs/formats/sfmr-file-format.md` § "Coordinate System Conventions").
//!
//! All functions are batch-oriented: quaternions are `(N, 4)` WXYZ arrays,
//! translations and world vectors are `(N, 3)` arrays. The thin Python
//! wrapper `sfmtool.colmap.convention` adds single-pose ergonomics.

use nalgebra::Vector3;
use numpy::{IntoPyArray, PyArray2, PyArrayMethods, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;

use sfmtool_core::geometry::convention;
use sfmtool_core::RotQuaternion;

/// A converted pose batch: `(N, 4)` WXYZ quaternions + `(N, 3)` translations.
type PosePairArrays<'py> = (Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>);

/// Validate an `(N, k)` array shape, returning N.
fn check_rows(name: &str, arr: &PyReadonlyArray2<f64>, k: usize) -> PyResult<usize> {
    let shape = arr.shape();
    if shape[1] != k {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{name} must have shape (N, {k}), got (N, {})",
            shape[1]
        )));
    }
    Ok(shape[0])
}

/// Apply a per-pose conversion function over `(N, 4)` quats + `(N, 3)` translations.
fn map_poses<'py>(
    py: Python<'py>,
    quats_wxyz: PyReadonlyArray2<f64>,
    translations_xyz: PyReadonlyArray2<f64>,
    f: fn(&RotQuaternion, &Vector3<f64>) -> (RotQuaternion, Vector3<f64>),
) -> PyResult<PosePairArrays<'py>> {
    let n = check_rows("quats_wxyz", &quats_wxyz, 4)?;
    let n_t = check_rows("translations_xyz", &translations_xyz, 3)?;
    if n != n_t {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "quats_wxyz has {n} rows but translations_xyz has {n_t}"
        )));
    }

    let q = quats_wxyz.as_array();
    let t = translations_xyz.as_array();
    let mut q_out = Vec::with_capacity(n * 4);
    let mut t_out = Vec::with_capacity(n * 3);
    for i in 0..n {
        let rot = RotQuaternion::from_wxyz_array([q[[i, 0]], q[[i, 1]], q[[i, 2]], q[[i, 3]]]);
        let trans = Vector3::new(t[[i, 0]], t[[i, 1]], t[[i, 2]]);
        let (rot_new, trans_new) = f(&rot, &trans);
        q_out.extend_from_slice(&rot_new.to_wxyz_array());
        t_out.extend_from_slice(&[trans_new.x, trans_new.y, trans_new.z]);
    }

    Ok((
        q_out.into_pyarray(py).reshape([n, 4])?,
        t_out.into_pyarray(py).reshape([n, 3])?,
    ))
}

/// Apply a per-vector rotation over an `(N, 3)` array.
fn map_vectors<'py>(
    py: Python<'py>,
    vectors_xyz: PyReadonlyArray2<f64>,
    f: fn(&Vector3<f64>) -> Vector3<f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let n = check_rows("vectors_xyz", &vectors_xyz, 3)?;
    let v = vectors_xyz.as_array();
    let mut out = Vec::with_capacity(n * 3);
    for i in 0..n {
        let rotated = f(&Vector3::new(v[[i, 0]], v[[i, 1]], v[[i, 2]]));
        out.extend_from_slice(&[rotated.x, rotated.y, rotated.z]);
    }
    out.into_pyarray(py).reshape([n, 3])
}

/// Convert COLMAP-convention world-to-camera poses to the canonical
/// convention: ``R' = S·R·Wᵀ``, ``t' = S·t``.
///
/// Args:
///     quats_wxyz: ``(N, 4)`` WXYZ rotation quaternions
///     translations_xyz: ``(N, 3)`` translations
///
/// Returns:
///     Tuple of converted ``(N, 4)`` quaternions and ``(N, 3)`` translations.
#[pyfunction]
pub fn poses_colmap_to_canonical<'py>(
    py: Python<'py>,
    quats_wxyz: PyReadonlyArray2<f64>,
    translations_xyz: PyReadonlyArray2<f64>,
) -> PyResult<PosePairArrays<'py>> {
    map_poses(
        py,
        quats_wxyz,
        translations_xyz,
        convention::pose_colmap_to_canonical,
    )
}

/// Convert canonical-convention world-to-camera poses back to COLMAP:
/// ``R = S·R'·W``, ``t = S·t'``. Inverse of ``poses_colmap_to_canonical``.
#[pyfunction]
pub fn poses_canonical_to_colmap<'py>(
    py: Python<'py>,
    quats_wxyz: PyReadonlyArray2<f64>,
    translations_xyz: PyReadonlyArray2<f64>,
) -> PyResult<PosePairArrays<'py>> {
    map_poses(
        py,
        quats_wxyz,
        translations_xyz,
        convention::pose_canonical_to_colmap,
    )
}

/// Conjugate relative poses (``cam2_from_cam1`` / ``sensor_from_rig``) by the
/// camera flip ``S``: ``R' = S·R·S``, ``t' = S·t``. Involutive — the same
/// call converts COLMAP → canonical and back.
#[pyfunction]
pub fn relative_poses_conjugate_s<'py>(
    py: Python<'py>,
    quats_wxyz: PyReadonlyArray2<f64>,
    translations_xyz: PyReadonlyArray2<f64>,
) -> PyResult<PosePairArrays<'py>> {
    map_poses(
        py,
        quats_wxyz,
        translations_xyz,
        convention::relative_pose_conjugate_s,
    )
}

/// Rotate world-space vectors by ``W``: ``(x, y, z) → (x, z, −y)``.
///
/// Applies to finite point xyz, infinity directions, normals, and patch
/// ``u``/``v`` half-vectors on COLMAP → canonical import. For homogeneous
/// xyzw points, rotate the xyz columns and carry ``w`` unchanged.
#[pyfunction]
pub fn world_rotate_w<'py>(
    py: Python<'py>,
    vectors_xyz: PyReadonlyArray2<f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    map_vectors(py, vectors_xyz, convention::world_rotate_w)
}

/// Rotate world-space vectors by ``W⁻¹ = Wᵀ``: ``(x, y, z) → (x, −z, y)``.
/// The canonical → COLMAP (export) counterpart of ``world_rotate_w``.
#[pyfunction]
pub fn world_rotate_w_inverse<'py>(
    py: Python<'py>,
    vectors_xyz: PyReadonlyArray2<f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    map_vectors(py, vectors_xyz, convention::world_rotate_w_inverse)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(poses_colmap_to_canonical, m)?)?;
    m.add_function(wrap_pyfunction!(poses_canonical_to_colmap, m)?)?;
    m.add_function(wrap_pyfunction!(relative_poses_conjugate_s, m)?)?;
    m.add_function(wrap_pyfunction!(world_rotate_w, m)?)?;
    m.add_function(wrap_pyfunction!(world_rotate_w_inverse, m)?)?;
    Ok(())
}
