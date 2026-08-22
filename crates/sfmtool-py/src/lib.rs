// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for sfmtool core functionality.
//!
//! Exposes file I/O (`.sfmr`, `.sift`, and COLMAP formats), geometric types,
//! feature matching, alignment, optical flow, and GUI viewer to Python via PyO3.
//!
//! Every binding lives on a PyO3 submodule (`_sfmtool.geometry`,
//! `_sfmtool.io`, `_sfmtool.sift`, `_sfmtool.reconstruction`,
//! `_sfmtool.patches`, `_sfmtool.matching`, `_sfmtool.analysis`,
//! `_sfmtool.flow`, `_sfmtool.spatial`, `_sfmtool.spherical`); each
//! submodule's `__name__` reads as the public `sfmtool.<name>` so binding
//! objects report the public location in tracebacks, IPython, and Sphinx.
//! The only root-level registrations are `build_profile` (build
//! introspection of this extension), `ProgressCounter` (cross-cutting
//! progress instrumentation shared by patch and matching kernels) and
//! `THUMBNAIL_SIZE` (the edge both on-disk formats pin, which the Python
//! extractors resize to); all are re-exported explicitly by
//! `sfmtool/__init__.py`. That file additionally
//! re-exports each submodule wholesale (`from sfmtool._sfmtool.<sub> import
//! *`), so the flat `sfmtool.*` surface still exists — it is now assembled on
//! the Python side from named submodules rather than registered flat here.
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

/// Try zero-copy `as_slice()` for C-contiguous arrays, fall back to copying
/// in logical (row-major) order otherwise. The C-contiguity guard matters:
/// `as_slice()` also succeeds for Fortran-contiguous arrays and returns the
/// raw column-major memory, silently transposing every row of a 2-D input.
///
/// **Use this for every numpy input whose buffer you read positionally.** A
/// bare `as_slice()` whose result is indexed as `s[i * cols + j]`, or handed to
/// a kernel as a flat row-major buffer, is wrong for Fortran-ordered input, and
/// wrong *silently* — no error, just transposed data. A
/// `map_err(… "must be C-contiguous")` does not save you: F-contiguous arrays
/// pass that check.
///
/// Expands to a `std::borrow::Cow`, fully qualified so no import is needed at
/// the call site.
///
/// # Do not reason from rank
///
/// It is tempting to write off 1-D inputs as "always contiguous, so exempt".
/// That is false twice over, and the false version of this rule is how the bug
/// kept being re-introduced:
///
/// * A 1-D *numpy* array need not be contiguous at all — `a[::2]`, `a[::-1]`
///   and `np.broadcast_to(...)` are all 1-D and neither C nor F. At an
///   `as_slice()` site those are rejected rather than corrupted, which is safe
///   but is not the same thing as exempt.
/// * ndarray treats a **negative** stride as contiguous
///   (`Dimension::is_contiguous`: `dim[0] <= 1 || strides[0] == -1`, and the
///   N-D branch compares `unsigned_abs()`). So `a[::-1]` — 1-D, no Fortran
///   order anywhere — survives `to_owned()` with its `-1` stride intact and
///   then panics in a format writer.
///
/// The rule is about how the buffer is *read*, never about the array's rank.
///
/// # The other spelling of this bug
///
/// `as_slice()` is not the only way in. `arr.as_array().to_owned()` **also**
/// carries a non-standard layout through: ndarray's `to_owned` preserves
/// strides whenever the source is contiguous in memory order — which covers
/// both F-contiguous and negative-stride input. The resulting `ArrayN` is not
/// standard-layout, and the format writers call `.as_slice().unwrap()` on it,
/// so the failure surfaces as a `PanicException` from inside `sfmr-format`, far
/// from the binding that let it through. Always write
/// `.as_array().as_standard_layout().into_owned()` instead; it costs nothing
/// extra when the input is already C-ordered. Note this spelling can be split
/// across statements (`let v = arr.as_array();` … `v.to_owned()`), so grepping
/// for the literal `as_array().to_owned()` will miss sites — grep for
/// `to_owned` and check each one.
macro_rules! to_contiguous {
    ($arr:expr) => {
        match $arr.as_slice() {
            Ok(s) if $arr.as_array().is_standard_layout() => ::std::borrow::Cow::Borrowed(s),
            _ => ::std::borrow::Cow::Owned($arr.as_array().iter().copied().collect::<Vec<_>>()),
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

// ── Reconstruction core types ─────────────────────────────────────────────

mod reconstruction;
pub use reconstruction::range_expr::PyRangeExpr;
pub use reconstruction::sfmr_reconstruction::PySfmrReconstruction;

// ── File I/O (incl. header-only image inspection) ─────────────────────────

mod io;

// ── Feature matching ──────────────────────────────────────────────────────

mod matching;

mod sift;

// ── Patches (surfels) + photometric refinement ───────────────────────────

mod patches;
pub use patches::{
    PyCameraViews, PyImagePyramidSet, PyOrientedPatch, PyPatchCloud, PyRansacPhotometricOutput,
};

mod py_progress;
pub use py_progress::ProgressCounter;

// ── Optical flow + warp maps ─────────────────────────────────────────────

mod flow;

// ── Analysis & algorithms ─────────────────────────────────────────────────

mod analysis;

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

    // File-format I/O: `.sfmr`, `.sift`, `.matches`, `.camrig`, COLMAP binary
    // + db, header-only image inspection.
    helpers::install_submodule(m, "sfmtool.io", io::register)?;

    // sfmtool SIFT detection / extraction.
    helpers::install_submodule(m, "sfmtool.sift", sift::register)?;

    // Reconstruction core types: SfmrReconstruction + RangeExpr.
    helpers::install_submodule(m, "sfmtool.reconstruction", reconstruction::register)?;

    // Patches (surfels): OrientedPatch, PatchCloud + its kernels, scene
    // inputs, photometric RANSAC, consensus atlas.
    helpers::install_submodule(m, "sfmtool.patches", patches::register)?;

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

    // Deliberately root-level: `ProgressCounter` is cross-cutting progress
    // instrumentation consumed by patch AND matching kernels, so no single
    // submodule is its honest home. `sfmtool/__init__.py` re-exports both
    // root names explicitly (no wildcard).
    m.add_class::<ProgressCounter>()?;

    // Also deliberately root-level: the thumbnail edge is one fact shared by
    // both on-disk formats and by the Python side that *produces* the pixels
    // (the SIFT extractors and `_undistort_images` resize to it; `sift/file.py`
    // validates against it). Exposing the Rust constant is what stops the
    // producer from drifting away from the format that stores its output — a
    // drift no Rust-side check could catch, because it would arrive as
    // correctly-shaped data of the wrong size.
    m.add("THUMBNAIL_SIZE", sfmtool_core::THUMBNAIL_SIZE)?;

    Ok(())
}
