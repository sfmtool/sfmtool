// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Argument resolution shared by the bindings that take cluster-track
//! observations either as a `MatchesFile` or as CSR arrays spelled out:
//! `focal_vote`, `estimate_intrinsics`, `cluster_radii` and
//! `coarsest_cluster_ids`.
//!
//! Each binding keeps its own error messages and the order its checks run in;
//! this module holds the steps they share, so the accepted input forms and
//! dtype casts have one implementation.

use numpy::ndarray::Dimension;
use numpy::{PyReadonlyArray, PyReadonlyArray1, PyUntypedArrayMethods};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

use sfmtool_matches_format::MatchesData;

use crate::io::matches_file::PyMatchesFile;

/// The first positional argument of a CSR-taking binding, resolved.
pub(crate) enum CsrSource<'a> {
    /// A `.matches` file (a selection included); the binding hands it to the
    /// core's own `from_matches` entry.
    Matches(&'a MatchesData),
    /// The `cluster_starts` CSR offsets, copied into a contiguous vector. Not
    /// yet validated: each binding checks the index in its own order.
    Starts(Vec<u32>),
}

/// Resolve the first positional argument into a `.matches` file or the
/// `cluster_starts` array.
///
/// A `MatchesFile` states its own observations, so when `extra_given` is true
/// (the caller also passed any of the array-form arguments) the call is
/// refused with `ValueError(matches_form_error)`. Anything that is neither a
/// `MatchesFile` nor a 1-D `uint32` array is a `TypeError`. The array form's
/// own "every argument is required" check is left to the caller, because it
/// runs after this one and names that binding's arguments.
pub(crate) fn resolve_csr_source<'a>(
    source: &'a Bound<'_, PyAny>,
    extra_given: bool,
    matches_form_error: &'static str,
) -> PyResult<CsrSource<'a>> {
    if let Ok(file) = source.cast::<PyMatchesFile>() {
        if extra_given {
            return Err(PyValueError::new_err(matches_form_error));
        }
        return Ok(CsrSource::Matches(file.get().data()));
    }
    let cluster_starts: PyReadonlyArray1<'_, u32> = source.extract().map_err(|_| {
        PyTypeError::new_err(
            "the first argument must be a MatchesFile or a (n_clusters + 1,) uint32 \
             cluster_starts array",
        )
    })?;
    Ok(CsrSource::Starts(
        to_contiguous!(cluster_starts).into_owned(),
    ))
}

/// Refuse a `cluster_starts` index that decreases anywhere.
pub(crate) fn check_starts_nondecreasing(starts: &[u32]) -> PyResult<()> {
    if starts.windows(2).any(|w| w[1] < w[0]) {
        return Err(PyValueError::new_err(
            "cluster_starts must be nondecreasing",
        ));
    }
    Ok(())
}

/// Refuse a `cluster_starts` index whose last offset is not `n_members`.
pub(crate) fn check_starts_close(starts: &[u32], n_members: usize) -> PyResult<()> {
    let last = starts.last().copied().unwrap_or(0);
    if last as usize != n_members {
        return Err(PyValueError::new_err(format!(
            "cluster_starts must close at the member count ({n_members}), not {last}"
        )));
    }
    Ok(())
}

/// A floating-point array argument read as `f32`, with its shape.
pub(crate) struct F32Array {
    /// The array's shape, for the caller's own shape checks.
    pub shape: Vec<usize>,
    /// The elements in C order.
    pub data: Vec<f32>,
}

/// Read a `D`-dimensional `float32` or `float64` array argument as `f32`.
///
/// `float32` is the width the `.matches` backbone stores positions and shapes
/// at, and is taken as it lies. `float64` is accepted and cast, because a
/// caller holding values read out of a `.matches` file has `f32`-originated
/// values in a `float64` array and the cast is exact for every one of them; a
/// caller that computed in double precision loses the bits below `f32` here,
/// which the kernel would do at its own read anyway. Any other dtype or
/// dimension count is `TypeError(type_error)`. Shape checks are the caller's.
pub(crate) fn float_array_f32<D: Dimension>(
    obj: &Bound<'_, PyAny>,
    type_error: &'static str,
) -> PyResult<F32Array> {
    if let Ok(a) = obj.extract::<PyReadonlyArray<'_, f32, D>>() {
        return Ok(F32Array {
            shape: a.shape().to_vec(),
            data: to_contiguous!(a).into_owned(),
        });
    }
    let a: PyReadonlyArray<'_, f64, D> = obj
        .extract()
        .map_err(|_| PyTypeError::new_err(type_error))?;
    Ok(F32Array {
        shape: a.shape().to_vec(),
        data: to_contiguous!(a).iter().map(|&v| v as f32).collect(),
    })
}
