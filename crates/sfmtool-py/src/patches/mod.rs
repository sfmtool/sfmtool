// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Bindings for the patch (surfel) pipeline: the `OrientedPatch` and
//! `PatchCloud` types, the `CameraViews`/`ImagePyramidSet` scene inputs, the
//! photometric RANSAC refiner, and the consensus-atlas compositor.
//!
//! `PatchCloud`'s heavy per-point kernels each live in their own module as an
//! additional `#[pymethods]` block (enabled by pyo3's `multiple-pymethods`
//! feature): `refine_normals`, `select_views`, `localize_keypoints`,
//! `refine_keypoints`, and `localizability`.

use pyo3::prelude::*;

pub mod args;
pub mod cloud;
pub mod consensus_atlas;
pub mod localizability;
pub mod localize_keypoints;
pub mod oriented_patch;
pub mod photometric_ransac;
pub mod refine_keypoints;
pub mod refine_normals;
pub mod select_views;
pub mod views;

pub use cloud::PyPatchCloud;
pub use oriented_patch::PyOrientedPatch;
pub use photometric_ransac::PyRansacPhotometricOutput;
pub use views::{PyCameraViews, PyImagePyramidSet};

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyOrientedPatch>()?;
    m.add_class::<PyPatchCloud>()?;
    m.add_class::<PyCameraViews>()?;
    m.add_class::<PyImagePyramidSet>()?;
    m.add_class::<PyRansacPhotometricOutput>()?;
    m.add_function(wrap_pyfunction!(
        photometric_ransac::refine_photometric_ransac_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        consensus_atlas::render_consensus_atlas_py,
        m
    )?)?;
    Ok(())
}
