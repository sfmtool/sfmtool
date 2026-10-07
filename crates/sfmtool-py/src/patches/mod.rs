// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Bindings for the patch (surfel) pipeline: the `OrientedPatch` and
//! `PatchCloud` types, the `CameraViews`/`ImagePyramidSet` scene inputs, the
//! photometric RANSAC refiner, the consensus-atlas compositor, candidate
//! track spawning, the ZNCC self-similarity radius of one tile or of a stack
//! of bitmaps, and the blur-matched ZNCC between a track's views' tiles.
//!
//! `PatchCloud`'s heavy per-point kernels each live in their own module as an
//! additional `#[pymethods]` block (enabled by pyo3's `multiple-pymethods`
//! feature): `refine_normals`, `select_views`, `localize_keypoints`,
//! `refine_keypoints`, `render_bitmaps`, and `member_coherence`.

use pyo3::prelude::*;

pub mod args;
pub mod blur_matched;
pub mod cloud;
pub mod consensus_atlas;
pub mod localize_keypoints;
pub mod member_coherence;
pub mod oriented_patch;
pub mod photometric_ransac;
pub mod refine_keypoints;
pub mod refine_normals;
pub mod render_bitmaps;
pub mod select_views;
pub mod self_similarity;
pub mod spawn;
pub mod view_tile;
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
    m.add_function(wrap_pyfunction!(spawn::spawn_candidate_tracks, m)?)?;
    m.add_function(wrap_pyfunction!(
        self_similarity::zncc_self_similarity_parts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        self_similarity::zncc_self_similarity_parts_stack,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(blur_matched::blur_matched_zncc_matrix, m)?)?;
    // The default bar on the ZNCC self-similarity radius, shared by the member
    // gates, the bench and the batch culls on a point's consensus bitmap.
    m.add(
        "DEFAULT_MAX_MEMBER_ZNCC_SELF_SIMILARITY_RADIUS",
        sfmtool_core::patch::keypoint_localize::DEFAULT_MAX_MEMBER_ZNCC_SELF_SIMILARITY_RADIUS,
    )?;
    // The sampler rule's threshold `a` that `sampler="per_view"` applies, so a
    // caller can record what its renders were made under.
    m.add(
        "DEFAULT_ANISOTROPIC_THRESHOLD",
        sfmtool_core::camera::sampler::DEFAULT_ANISOTROPIC_THRESHOLD,
    )?;
    Ok(())
}
