// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for sfmtool core functionality.
//!
//! Exposes file I/O (`.sfmr`, `.sift`, and COLMAP formats), geometric types,
//! feature matching, alignment, optical flow, and GUI viewer to Python via PyO3.
//!
//! File-format I/O, feature matching, optical flow, spatial indices, and
//! spherical-rig bindings each live on their own PyO3 submodule
//! (`_sfmtool.io`, `_sfmtool.matching`, `_sfmtool.flow`, `_sfmtool.spatial`,
//! `_sfmtool.spherical`); their `__name__` reads as `sfmtool.io` /
//! `sfmtool.matching` / `sfmtool.flow` / `sfmtool.spatial` /
//! `sfmtool.spherical` so binding objects report the public location in
//! tracebacks, IPython, and Sphinx. Everything else is registered flat on
//! `_sfmtool` for now (see hygiene audit #4 for the rest).
//!
//! # Example
//!
//! ```python
//! from sfmtool._sfmtool.io import read_sfmr, write_sfmr, verify_sfmr
//! from sfmtool._sfmtool.io import read_sift, write_sift, verify_sift
//!
//! data = read_sfmr("reconstruction.sfmr")
//! valid, errors = verify_sfmr("reconstruction.sfmr")
//! ```

use pyo3::prelude::*;

/// Try zero-copy `as_slice()` for contiguous arrays, fall back to copying for non-contiguous.
macro_rules! to_contiguous {
    ($arr:expr) => {
        match $arr.as_slice() {
            Ok(s) => Cow::Borrowed(s),
            Err(_) => Cow::Owned($arr.as_array().iter().copied().collect::<Vec<_>>()),
        }
    };
}

// ── Shared helpers ────────────────────────────────────────────────────────

pub(crate) mod helpers;

// ── Geometric types ───────────────────────────────────────────────────────

mod py_rot_quaternion;
pub use py_rot_quaternion::PyRotQuaternion;

mod py_camera_intrinsics;
pub use py_camera_intrinsics::PyCameraIntrinsics;

mod py_rigid_transform;
pub use py_rigid_transform::PyRigidTransform;

mod py_se3_transform;
pub use py_se3_transform::PySe3Transform;

mod py_sfmr_reconstruction;
pub use py_sfmr_reconstruction::PySfmrReconstruction;
mod recon_clone;

mod py_range_expr;
pub use py_range_expr::PyRangeExpr;

// ── File I/O ──────────────────────────────────────────────────────────────

mod io;

// ── Image inspection ──────────────────────────────────────────────────────

mod py_image;

// ── Feature matching ──────────────────────────────────────────────────────

mod matching;

mod py_sift;

// ── Image warping ────────────────────────────────────────────────────────

mod py_patch_cloud;
pub use py_patch_cloud::{PyOrientedPatch, PyPatchCloud};

// ── Optical flow + warp maps ─────────────────────────────────────────────

mod flow;

// ── Analysis & algorithms ─────────────────────────────────────────────────

mod py_analysis;
mod py_consensus_atlas;
mod py_epipolar;
mod py_image_pair_graph;
mod py_photometric_ransac;
mod py_triangulation;
pub use py_photometric_ransac::PyRansacPhotometricOutput;

// ── Spatial indices (KD-trees, kd-tree forest) ───────────────────────────

mod spatial;

// ── Spherical (tile rigs, source stacks, sphere points) ──────────────────

mod spherical;

// ── Module registration ───────────────────────────────────────────────────

/// Cargo build profile this extension was compiled with: `"debug"` or `"release"`.
///
/// The Rust numeric kernels (SIFT, optical flow, matching) run roughly an order
/// of magnitude slower without optimizations, so performance-sensitive callers
/// (benchmarks) check this to refuse accidental debug builds, whose timings are
/// meaningless. `cfg!(debug_assertions)` is the standard release/debug split.
#[pyfunction]
fn build_profile() -> &'static str {
    if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    }
}

/// Python module for sfmtool core functionality.
#[pymodule]
fn _sfmtool(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Build introspection
    m.add_function(wrap_pyfunction!(build_profile, m)?)?;

    // File-format I/O: `.sfmr`, `.sift`, `.matches`, `.camrig`, COLMAP binary + db.
    helpers::install_submodule(m, "sfmtool.io", io::register)?;

    // sfmtool SIFT detection / extraction
    m.add_function(wrap_pyfunction!(py_sift::detect_sift_keypoints, m)?)?;
    m.add_function(wrap_pyfunction!(py_sift::extract_sift, m)?)?;

    // Image inspection
    m.add_function(wrap_pyfunction!(py_image::image_dimensions, m)?)?;

    // Feature matching: descriptor + image-pair + sweep + cluster.
    helpers::install_submodule(m, "sfmtool.matching", matching::register)?;

    // SE3 transform acceleration
    m.add_function(wrap_pyfunction!(
        py_analysis::apply_se3_to_camera_poses_py,
        m
    )?)?;

    // Viewing angle analysis
    m.add_function(wrap_pyfunction!(py_analysis::compute_narrow_track_mask, m)?)?;

    // Batch triangulation
    m.add_function(wrap_pyfunction!(py_triangulation::triangulate_batch, m)?)?;

    // Alignment (Kabsch + RANSAC)
    m.add_function(wrap_pyfunction!(py_analysis::kabsch_algorithm_rs, m)?)?;
    m.add_function(wrap_pyfunction!(py_analysis::ransac_alignment_rs, m)?)?;

    // Point correspondence
    m.add_function(wrap_pyfunction!(
        py_analysis::find_point_correspondences_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        py_analysis::merge_points_and_tracks_py,
        m
    )?)?;

    // Track filtering
    m.add_function(wrap_pyfunction!(
        py_analysis::filter_tracks_by_point_mask_py,
        m
    )?)?;

    // Image pair graph
    m.add_function(wrap_pyfunction!(
        py_image_pair_graph::build_covisibility_pairs_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        py_image_pair_graph::build_frustum_intersection_pairs_py,
        m
    )?)?;

    // Optical flow + warp maps.
    helpers::install_submodule(m, "sfmtool.flow", flow::register)?;

    // Spatial indices: 2D/3D KD-trees + randomized kd-tree forest.
    helpers::install_submodule(m, "sfmtool.spatial", spatial::register)?;

    // Spherical: tile rigs + per-tile source stacks + sphere-point generation.
    helpers::install_submodule(m, "sfmtool.spherical", spherical::register)?;

    // Epipolar curves (distortion-aware epipolar lines)
    m.add_function(wrap_pyfunction!(py_epipolar::epipolar_curves_py, m)?)?;

    // Types
    m.add_class::<PyCameraIntrinsics>()?;
    m.add_class::<PyRigidTransform>()?;
    m.add_class::<PyRotQuaternion>()?;
    m.add_class::<PySe3Transform>()?;
    m.add_class::<PySfmrReconstruction>()?;
    m.add_class::<PyRangeExpr>()?;
    m.add_class::<PyRansacPhotometricOutput>()?;
    m.add_class::<PyOrientedPatch>()?;
    m.add_class::<PyPatchCloud>()?;

    // Photometric refinement.
    m.add_function(wrap_pyfunction!(
        py_photometric_ransac::refine_photometric_ransac_py,
        m
    )?)?;

    // Tile-batched consensus atlas compositing.
    m.add_function(wrap_pyfunction!(
        py_consensus_atlas::render_consensus_atlas_py,
        m
    )?)?;

    Ok(())
}
