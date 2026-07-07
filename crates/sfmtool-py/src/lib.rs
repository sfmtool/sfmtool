// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for sfmtool core functionality.
//!
//! Exposes file I/O (`.sfmr`, `.sift`, and COLMAP formats), geometric types,
//! feature matching, alignment, optical flow, and GUI viewer to Python via PyO3.
//!
//! Geometric value types, file-format I/O, SIFT extraction, feature matching,
//! reconstruction analysis, optical flow, spatial indices, and spherical-rig
//! bindings each live on their own PyO3 submodule (`_sfmtool.geometry`,
//! `_sfmtool.io`, `_sfmtool.sift`, `_sfmtool.matching`, `_sfmtool.analysis`,
//! `_sfmtool.flow`, `_sfmtool.spatial`, `_sfmtool.spherical`); their
//! `__name__` reads as `sfmtool.geometry` / `sfmtool.io` / `sfmtool.sift` /
//! `sfmtool.matching` / `sfmtool.analysis` / `sfmtool.flow` /
//! `sfmtool.spatial` / `sfmtool.spherical` so binding objects report the
//! public location in tracebacks, IPython, and Sphinx. Everything else is
//! registered flat on `_sfmtool` for now (see hygiene audit #4 for the rest).
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

mod geometry;
// Re-exported at the crate root so intra-crate users keep referring to these
// value types as `crate::PyCameraIntrinsics` etc.; the public Python surface
// lives on the `_sfmtool.geometry` submodule (see `geometry::register`).
pub use geometry::{PyCameraIntrinsics, PyRigidTransform, PyRotQuaternion, PySe3Transform};

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

mod sift;

// ── Image warping ────────────────────────────────────────────────────────

mod py_patch_cloud;
pub use py_patch_cloud::{PyImagePyramidSet, PyOrientedPatch, PyPatchCloud};

mod py_progress;
pub use py_progress::ProgressCounter;

// ── Optical flow + warp maps ─────────────────────────────────────────────

mod flow;

// ── Analysis & algorithms ─────────────────────────────────────────────────

mod analysis;
mod py_consensus_atlas;
mod py_photometric_ransac;
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

    // Geometric value types: camera intrinsics, quaternions, rigid + SE3 transforms.
    helpers::install_submodule(m, "sfmtool.geometry", geometry::register)?;

    // File-format I/O: `.sfmr`, `.sift`, `.matches`, `.camrig`, COLMAP binary + db.
    helpers::install_submodule(m, "sfmtool.io", io::register)?;

    // sfmtool SIFT detection / extraction.
    helpers::install_submodule(m, "sfmtool.sift", sift::register)?;

    // Image inspection
    m.add_function(wrap_pyfunction!(py_image::image_dimensions, m)?)?;

    // Feature matching: descriptor + image-pair + sweep + cluster.
    helpers::install_submodule(m, "sfmtool.matching", matching::register)?;

    // Reconstruction analysis: pose/track ops, alignment, correspondence,
    // triangulation, epipolar curves, image-pair graphs.
    helpers::install_submodule(m, "sfmtool.analysis", analysis::register)?;

    // Optical flow + warp maps.
    helpers::install_submodule(m, "sfmtool.flow", flow::register)?;

    // Spatial indices: 2D/3D KD-trees + randomized kd-tree forest.
    helpers::install_submodule(m, "sfmtool.spatial", spatial::register)?;

    // Spherical: tile rigs + per-tile source stacks + sphere-point generation.
    helpers::install_submodule(m, "sfmtool.spherical", spherical::register)?;

    // Types
    m.add_class::<PySfmrReconstruction>()?;
    m.add_class::<PyRangeExpr>()?;
    m.add_class::<PyRansacPhotometricOutput>()?;
    m.add_class::<PyOrientedPatch>()?;
    m.add_class::<PyPatchCloud>()?;
    m.add_class::<PyImagePyramidSet>()?;
    m.add_class::<ProgressCounter>()?;

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
