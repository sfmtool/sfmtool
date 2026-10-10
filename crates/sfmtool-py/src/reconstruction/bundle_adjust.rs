// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for the per-camera release checks of the reconstruction
//! bundle adjustment (`sfmtool_core::reconstruction::bundle_adjust`), which
//! `EditedReconstruction.bundle_adjust` runs.

use pyo3::prelude::*;

use sfmtool_core::reconstruction::bundle_adjust as core_bundle_adjust;

use crate::geometry::PyCameraIntrinsics;

/// Whether ``EditedReconstruction.bundle_adjust`` can release this camera's
/// focal length: true for ``SIMPLE_PINHOLE``, ``EQUIDISTANT_FISHEYE``,
/// ``SIMPLE_RADIAL_FISHEYE``, ``SFMTOOL_FISHEYE`` and ``SFMTOOL_PINHOLE``.
#[pyfunction]
fn focal_is_releasable(camera: PyRef<'_, PyCameraIntrinsics>) -> bool {
    core_bundle_adjust::focal_is_releasable(&camera.inner)
}

/// Whether ``EditedReconstruction.bundle_adjust`` can release this camera's
/// lens distortion: ``k1`` on ``SIMPLE_RADIAL_FISHEYE``, and the spline on an
/// ``SFMTOOL_FISHEYE`` or ``SFMTOOL_PINHOLE`` with at least two coefficients
/// on a positive domain.
#[pyfunction]
fn distortion_is_releasable(camera: PyRef<'_, PyCameraIntrinsics>) -> bool {
    core_bundle_adjust::distortion_is_releasable(&camera.inner)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(focal_is_releasable, m)?)?;
    m.add_function(wrap_pyfunction!(distortion_is_releasable, m)?)?;
    Ok(())
}
