// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for the per-cluster feature radius and the coarsest-N cut
//! ordered by it (``sfmtool._sfmtool.analysis.cluster_radii`` /
//! ``coarsest_clusters``; see ``specs/core/analysis/source-clusters.md``).

use std::borrow::Cow;

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray3, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use matches_format::MatchesData;
use sfmtool_core::analysis::cluster_radii::{
    cluster_radii as core_cluster_radii, cluster_radii_from_matches,
    coarsest_clusters as core_coarsest_clusters, coarsest_clusters_from_matches, ClusterRadiiError,
};

use crate::io::matches_file::PyMatchesFile;

/// What a radius-shaped binding was called with: a parsed `.matches` handle,
/// or the CSR index and the shapes spelled out.
///
/// Both forms reach the same kernel; the object form hands the whole file to
/// the core's own `from_matches` entry rather than taking the file apart here,
/// so the reading (the shapes, the refine radius the file records them
/// against) has exactly one implementation and both languages get it.
enum RadiiSource<'a> {
    /// A `.matches` file, read by the core entry point.
    Matches(&'a MatchesData),
    /// Explicit CSR index, flat row-major shapes, and their refine radius.
    Arrays {
        cluster_starts: Vec<u32>,
        member_affine_shapes: Vec<f32>,
        refine_radius: f32,
    },
}

/// Resolve a radius-shaped binding's positional arguments into one of the two
/// forms.
///
/// `source` is either a `MatchesFile` -- and then nothing else may be given --
/// or the `cluster_starts` array, and then both remaining arguments are
/// required.
fn radii_source<'a, 'py>(
    source: &'a Bound<'py, PyAny>,
    member_affine_shapes: Option<Bound<'py, PyAny>>,
    refine_radius: Option<f32>,
) -> PyResult<RadiiSource<'a>> {
    if let Ok(file) = source.cast::<PyMatchesFile>() {
        if member_affine_shapes.is_some() || refine_radius.is_some() {
            return Err(PyValueError::new_err(
                "the MatchesFile form takes no shape arrays: the file states its own \
                 member shapes and the refine radius they are expressed against",
            ));
        }
        return Ok(RadiiSource::Matches(file.get().data()));
    }
    let cluster_starts: PyReadonlyArray1<'py, u32> = source.extract().map_err(|_| {
        pyo3::exceptions::PyTypeError::new_err(
            "the first argument must be a MatchesFile or a (n_clusters + 1,) uint32 \
             cluster_starts array",
        )
    })?;
    let (Some(shapes), Some(refine_radius)) = (member_affine_shapes, refine_radius) else {
        return Err(PyValueError::new_err(
            "the array form takes cluster_starts, member_affine_shapes and refine_radius",
        ));
    };
    let starts = to_contiguous!(cluster_starts);
    if starts.is_empty() {
        return Err(PyValueError::new_err(
            "cluster_starts must carry at least one boundary",
        ));
    }
    if starts.windows(2).any(|w| w[1] < w[0]) {
        return Err(PyValueError::new_err(
            "cluster_starts must be nondecreasing",
        ));
    }
    let shapes = member_shapes_f32(&shapes)?;
    if starts.last().copied().unwrap_or(0) as usize != shapes.len() / 4 {
        return Err(PyValueError::new_err(format!(
            "cluster_starts must close at the member count ({}), not {}",
            shapes.len() / 4,
            starts.last().copied().unwrap_or(0)
        )));
    }
    Ok(RadiiSource::Arrays {
        cluster_starts: starts.into_owned(),
        member_affine_shapes: shapes,
        refine_radius,
    })
}

/// The `member_affine_shapes` argument as the flat `f32` fours the kernel
/// takes.
///
/// `float32` is the array's own width and is taken as it lies. `float64` is
/// accepted and cast, because a caller holding shapes read out of a
/// `.matches` file has `f32`-originated values in a `float64` array and the
/// cast is exact for every one of them.
fn member_shapes_f32(obj: &Bound<'_, PyAny>) -> PyResult<Vec<f32>> {
    let shape_err =
        || PyValueError::new_err("member_affine_shapes must have shape (n_member, 2, 2)");
    if let Ok(a) = obj.extract::<PyReadonlyArray3<'_, f32>>() {
        if a.shape()[1] != 2 || a.shape()[2] != 2 {
            return Err(shape_err());
        }
        let flat: Cow<'_, [f32]> = to_contiguous!(a);
        return Ok(flat.into_owned());
    }
    let a: PyReadonlyArray3<'_, f64> = obj.extract().map_err(|_| {
        pyo3::exceptions::PyTypeError::new_err(
            "member_affine_shapes must be a (n_member, 2, 2) float32 or float64 array",
        )
    })?;
    if a.shape()[1] != 2 || a.shape()[2] != 2 {
        return Err(shape_err());
    }
    let flat = to_contiguous!(a);
    Ok(flat.iter().map(|&v| v as f32).collect())
}

/// Map a core radius-reading refusal onto the Python exception the caller
/// sees. Every one of them is a property of the file, so they are all value
/// errors.
fn radii_err_to_py(e: ClusterRadiiError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// Each cluster's feature radius in pixels (see
/// ``specs/core/analysis/source-clusters.md``).
///
/// A member's radius is the refine radius times the MEAN of its stored affine's
/// two column norms, computed as half the refine radius times their sum; a
/// cluster's radius is its widest member's, and a cluster with no members reads
/// ``0.0``. This is the same reading the source-cluster join bands against.
///
/// Takes its input in either of two forms, and only these two:
///
/// * ``cluster_radii(matches_file)`` -- a ``MatchesFile`` (a selection
///   included), whose cluster backbone carries the shapes and whose
///   ``cluster_patches/`` section carries the refine radius they are expressed
///   against. A file missing either raises ``ValueError``.
/// * ``cluster_radii(cluster_starts, member_affine_shapes, refine_radius)`` --
///   the same input spelled out.
///
/// Args:
///     cluster_starts: A ``MatchesFile``, or the (n_clusters + 1,) uint32 CSR
///         offsets into the member arrays: nondecreasing and closing at the
///         member count.
///     member_affine_shapes: (n_member, 2, 2) float32 absolute affine shapes
///         -- the width the ``.matches`` backbone stores them at. A float64
///         array is accepted and cast, which is exact for the
///         ``f32``-originated values a caller reads out of such a file.
///         Omitted in the ``MatchesFile`` form.
///     refine_radius: The refine radius the shapes are expressed against. A
///         Python float is float64 and narrows to the float32 the reading
///         runs at. Omitted in the ``MatchesFile`` form.
///
/// Returns:
///     (n_clusters,) float32 radius per cluster, in cluster order -- the width
///     the shapes are stored at and the reading is computed at.
// This is a Python docstring (rendered by `help()`), not Rust prose: its
// indented `Args:` / `Returns:` continuation paragraphs read as Markdown
// indented code blocks, which rustdoc then tries to parse as Rust.
#[allow(rustdoc::invalid_rust_codeblocks)]
#[pyfunction]
#[pyo3(signature = (cluster_starts, member_affine_shapes=None, refine_radius=None))]
pub fn cluster_radii<'py>(
    py: Python<'py>,
    cluster_starts: Bound<'py, PyAny>,
    member_affine_shapes: Option<Bound<'py, PyAny>>,
    refine_radius: Option<f32>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let source = radii_source(&cluster_starts, member_affine_shapes, refine_radius)?;
    let out = py
        .detach(move || match source {
            RadiiSource::Matches(matches) => cluster_radii_from_matches(matches),
            RadiiSource::Arrays {
                cluster_starts,
                member_affine_shapes,
                refine_radius,
            } => Ok(core_cluster_radii(
                &cluster_starts,
                &member_affine_shapes,
                refine_radius,
            )),
        })
        .map_err(radii_err_to_py)?;
    Ok(PyArray1::from_vec(py, out))
}

/// Ids of the ``n`` coarsest clusters: radius descending, id ascending on ties,
/// returned sorted ASCENDING (see ``specs/core/analysis/source-clusters.md``).
///
/// The ascending return is what makes the cut composable -- it is a cluster-id
/// set to restrict a selection by, not a ranking -- while the descending
/// ordering behind it decides membership. Fewer than ``n`` clusters yields all
/// of them.
///
/// Takes its input in the same two forms as ``cluster_radii``:
///
/// * ``coarsest_clusters(matches_file, n)``
/// * ``coarsest_clusters(cluster_starts, n, member_affine_shapes,
///   refine_radius)``
///
/// Args:
///     cluster_starts: A ``MatchesFile``, or the (n_clusters + 1,) uint32 CSR
///         offsets into the member arrays.
///     n: How many of the coarsest clusters to name.
///     member_affine_shapes: (n_member, 2, 2) float32 absolute affine shapes.
///         A float64 array is accepted and cast. Omitted in the
///         ``MatchesFile`` form.
///     refine_radius: The refine radius the shapes are expressed against. A
///         Python float is float64 and narrows to the float32 the reading
///         runs at. Omitted in the ``MatchesFile`` form.
///
/// Returns:
///     (min(n, n_clusters),) uint32 cluster ids, ascending.
// This is a Python docstring (rendered by `help()`), not Rust prose: its
// indented `Args:` / `Returns:` continuation paragraphs read as Markdown
// indented code blocks, which rustdoc then tries to parse as Rust.
#[allow(rustdoc::invalid_rust_codeblocks)]
#[pyfunction]
#[pyo3(signature = (cluster_starts, n, member_affine_shapes=None, refine_radius=None))]
pub fn coarsest_clusters<'py>(
    py: Python<'py>,
    cluster_starts: Bound<'py, PyAny>,
    n: usize,
    member_affine_shapes: Option<Bound<'py, PyAny>>,
    refine_radius: Option<f32>,
) -> PyResult<Bound<'py, PyArray1<u32>>> {
    let source = radii_source(&cluster_starts, member_affine_shapes, refine_radius)?;
    let out = py
        .detach(move || match source {
            RadiiSource::Matches(matches) => coarsest_clusters_from_matches(matches, n),
            RadiiSource::Arrays {
                cluster_starts,
                member_affine_shapes,
                refine_radius,
            } => Ok(core_coarsest_clusters(
                &cluster_starts,
                &member_affine_shapes,
                refine_radius,
                n,
            )),
        })
        .map_err(radii_err_to_py)?;
    Ok(PyArray1::from_vec(py, out))
}

// ── Registration ──────────────────────────────────────────────────────────

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(cluster_radii, m)?)?;
    m.add_function(wrap_pyfunction!(coarsest_clusters, m)?)?;
    Ok(())
}
