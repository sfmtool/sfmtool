// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Binding for the point columns, the named per-point values of a
//! reconstruction (``sfmtool.analysis.point_columns``; see
//! ``specs/core/analysis/point-columns.md``).

use std::borrow::Cow;
use std::sync::Arc;

use numpy::PyArray1;
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use sfmtool_core::analysis::point_columns::{PointColumn, PointColumns};
use sfmtool_core::progress::Progress;
use sfmtool_core::EditedReconstruction;

use crate::reconstruction::edited::PyEditedReconstruction;
use crate::PySfmrReconstruction;

/// The wire names of every column, in catalogue order, for an error message.
fn all_names() -> String {
    PointColumn::ALL
        .iter()
        .map(|c| c.wire_name())
        .collect::<Vec<_>>()
        .join(", ")
}

/// The named per-point values of every live point of a reconstruction.
///
/// Each column is one value per point under a fixed name: ``track_length``,
/// ``image_count``, ``error_px``, ``at_infinity``, ``x``, ``y``, ``z``,
/// ``centroid_distance``, ``normal_confidence``, ``lowest_zncc``,
/// ``largest_zncc_self_similarity_radius``, ``max_ray_angle_deg``,
/// ``condition_number`` and ``inverse_depth_z``. These are the numbers the
/// viewer shows for the same points. A point with no value in a column (a
/// point at infinity has no ``x``) holds NaN there.
///
/// Args:
///     recon: An :class:`SfmrReconstruction`, or an
///         :class:`EditedReconstruction` read through its edits.
///     columns: The wire names to compute, or None for every column the
///         reconstruction can supply.
///
/// Returns:
///     A dict mapping each computed column's name to an (n,) array,
///     ``float32``, or ``float64`` for ``x``, ``y`` and ``z``, plus
///     ``point_index``, the (n,) ``uint32`` point index of each row, ascending.
///
/// Raises:
///     ValueError: where a name is not a column, or names a column the
///         reconstruction cannot supply (``normal_confidence`` without that
///         column, ``lowest_zncc`` without observation confidences,
///         ``largest_zncc_self_similarity_radius`` without observation
///         readings, ``centroid_distance`` with no posed image).
///     TypeError: where ``recon`` is neither reconstruction type.
#[pyfunction]
#[pyo3(signature = (recon, columns = None))]
fn point_columns<'py>(
    py: Python<'py>,
    recon: &Bound<'py, PyAny>,
    columns: Option<Vec<String>>,
) -> PyResult<Bound<'py, PyDict>> {
    let edited_ref;
    let wrapped;
    let edited: &EditedReconstruction = if let Ok(e) = recon.cast::<PyEditedReconstruction>() {
        edited_ref = e.borrow();
        &edited_ref.inner
    } else if let Ok(r) = recon.cast::<PySfmrReconstruction>() {
        wrapped = EditedReconstruction::new(Arc::new(r.borrow().inner.clone()));
        &wrapped
    } else {
        return Err(PyTypeError::new_err(
            "recon must be an SfmrReconstruction or an EditedReconstruction",
        ));
    };

    let wanted: Cow<'_, [PointColumn]> = match columns {
        None => Cow::Borrowed(&PointColumn::ALL),
        Some(names) => {
            let mut wanted = Vec::with_capacity(names.len());
            for name in &names {
                let column = PointColumn::from_wire_name(name).ok_or_else(|| {
                    PyValueError::new_err(format!(
                        "{name:?} is not a point column; the columns are {}",
                        all_names()
                    ))
                })?;
                if !column.is_available(edited) {
                    let available: Vec<&str> = PointColumn::ALL
                        .iter()
                        .filter(|c| c.is_available(edited))
                        .map(|c| c.wire_name())
                        .collect();
                    return Err(PyValueError::new_err(format!(
                        "this reconstruction cannot supply the column {name}; it has {}",
                        available.join(", ")
                    )));
                }
                wanted.push(column);
            }
            Cow::Owned(wanted)
        }
    };

    let computed = py
        .detach(|| PointColumns::compute(edited, &wanted, &Progress::none()))
        .map_err(|_| PyRuntimeError::new_err("point_columns was cancelled"))?;

    let out = PyDict::new(py);
    out.set_item(
        "point_index",
        PyArray1::from_slice(py, computed.point_indexes()),
    )?;
    for column in PointColumn::ALL {
        if let Some(values) = computed.values(column) {
            out.set_item(column.wire_name(), PyArray1::from_slice(py, values))?;
        } else if let Some(values) = computed.values_f64(column) {
            out.set_item(column.wire_name(), PyArray1::from_slice(py, values))?;
        }
    }
    Ok(out)
}

// ── Registration ──────────────────────────────────────────────────────────

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(point_columns, m)?)?;
    Ok(())
}
