// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for the per-image keypoint reach enumeration: which other
//! keypoints lie inside a keypoint's own disk.

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use sfmtool_core::spatial::keypoint_reach::{pairs_within_reach, KeypointRows, ReachPairs};

/// The three arrays the enumeration hands back: `(i, j, d_px)`.
type PyReachPairs<'py> = (
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<f64>>,
);

/// Every row inside every row's own reach, per image.
///
/// A KEYPOINT is a row of a track set: an image index, a pixel position, and
/// its own query radius, its REACH, in pixels. The relation is per image, and
/// it is DIRECTED: the disk is row ``i``'s, so ``(i, j)`` says nothing about
/// ``(j, i)``. A caller wanting a symmetric relation reads both directions,
/// which this already emits.
///
/// A row is never its own candidate. A row whose reach is not finite asks
/// nothing, and still appears as a candidate of other rows.
///
/// One distance is reported per pair, so a caller testing against
/// ``reach_px[i]``, against a bound of its own, or against a function of both
/// radii reads the same ``d_px``.
///
/// The pair stream runs rows in their given order within an image, each row's
/// candidates in the sorted column order of its run, and images in ascending
/// image index.
///
/// Args:
///     image_of_row: (n,) int64 image index per row.
///     xy_px: (n, 2) float64 pixel positions.
///     reach_px: (n,) float64 query radius per row. NaN asks nothing; a
///         negative radius is refused.
///
/// Returns:
///     ``(i, j, d_px)``: (n_pairs,) int64 row indices, (n_pairs,) int64
///     candidate row indices, and (n_pairs,) float64 separations in pixels,
///     all C-contiguous.
///
/// Raises:
///     ValueError: where the three inputs disagree on the row count, where
///         ``xy_px`` is not ``(n, 2)``, or where a row states a negative reach.
#[pyfunction]
#[pyo3(signature = (image_of_row, xy_px, reach_px))]
pub fn keypoint_pairs_within_reach<'py>(
    py: Python<'py>,
    image_of_row: PyReadonlyArray1<'py, i64>,
    xy_px: PyReadonlyArray2<'py, f64>,
    reach_px: PyReadonlyArray1<'py, f64>,
) -> PyResult<PyReachPairs<'py>> {
    let n = image_of_row.shape()[0];
    if xy_px.shape() != [n, 2] {
        return Err(PyValueError::new_err(format!(
            "xy_px must have shape ({n}, 2), got {:?}",
            xy_px.shape()
        )));
    }
    if reach_px.shape()[0] != n {
        return Err(PyValueError::new_err(format!(
            "image_of_row states {n} rows and reach_px {}",
            reach_px.shape()[0]
        )));
    }

    let images = to_contiguous!(image_of_row);
    let xy = to_contiguous!(xy_px);
    let reach = to_contiguous!(reach_px);

    let ReachPairs {
        row,
        candidate,
        distance_px,
    } = py
        .detach(|| {
            pairs_within_reach(KeypointRows {
                image_of_row: &images,
                xy_px: &xy,
                reach_px: &reach,
            })
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok((
        PyArray1::from_vec(py, row),
        PyArray1::from_vec(py, candidate),
        PyArray1::from_vec(py, distance_px),
    ))
}

// ── Registration ──────────────────────────────────────────────────────────

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(keypoint_pairs_within_reach, m)?)?;
    Ok(())
}
