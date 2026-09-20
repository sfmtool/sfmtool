// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Binding for the rule that retires a coarse observation a finer one covers
//! (``sfmtool._sfmtool.analysis.covered_by_finer``; see
//! ``specs/core/analysis/covered-by-finer.md``).

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use sfmtool_core::analysis::covered_by_finer::{
    covered_by_finer as core_covered_by_finer, CoveredByFiner, CoveredOptions, CoveredRows,
};
use sfmtool_core::progress::Progress;

/// Which observations a finer one, on another owner, covers.
///
/// A ROW is one observation: the image it was seen in, the OWNER it belongs to,
/// its pixel position, the REACH of its drawn footprint and its own feature
/// RADIUS. The reach and the radius are two different lengths and the rule
/// needs both: containment is asked at the reach, and "finer" is asked of the
/// radius.
///
/// A row is retired where another row in the same image, on another owner, has
/// its centre inside the first row's footprint and a radius at least ``ratio``
/// times smaller. The coarse side is the one retired, never the fine one, and
/// the verdict is by existence, so the order rows arrive in cannot change it.
///
/// An owner left with fewer than ``min_observations`` surviving rows is dropped
/// and its survivors go with it.
///
/// Args:
///     image_of_row: (n,) int64 image index per row.
///     owner_of_row: (n,) int64 owner per row, in ``[0, n_owners)``.
///     xy_px: (n, 2) float64 pixel positions.
///     reach_px: (n,) float64 footprint radius per row. NaN asks nothing and
///         still covers; a negative radius is refused.
///     radius_px: (n,) float64 feature radius per row. NaN neither covers nor
///         is covered.
///     n_owners: How many owners the verdict indexes.
///     ratio: How many times finer the covering row has to be (default 2.0,
///         one octave). The comparison is non-strict.
///     min_fine_radius_px: A covering row below this says nothing (default
///         0.0, which is off).
///     min_observations: Surviving rows an owner needs to be kept (default 2).
///     protected: (n,) bool rows that are never retired, or None. A protected
///         row still covers.
///
/// Returns:
///     A dict with ``flagged`` (n,) bool, ``keep_row`` (n,) bool,
///     ``keep_owner`` (n_owners,) bool, and ``census``, a dict of ``rows``,
///     ``pairs_contained``, ``pairs_finer``, ``rows_flagged``, ``rows_spared``,
///     ``rows_removed``, ``owners_dropped_all_covered``,
///     ``owners_dropped_by_sweep`` and ``owners_kept``.
///
/// Raises:
///     ValueError: where the per-row inputs disagree on the row count, where
///         ``xy_px`` is not ``(n, 2)``, where a row names an owner outside
///         ``[0, n_owners)``, where a threshold is unusable, or where a row
///         states a negative reach.
// This is a Python docstring (rendered by `help()`), not Rust prose: its
// indented `Args:` / `Returns:` continuation paragraphs read as Markdown
// indented code blocks, which rustdoc then tries to parse as Rust.
#[allow(rustdoc::invalid_rust_codeblocks)]
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    image_of_row,
    owner_of_row,
    xy_px,
    reach_px,
    radius_px,
    n_owners,
    *,
    ratio=2.0,
    min_fine_radius_px=0.0,
    min_observations=2,
    protected=None,
))]
pub fn covered_by_finer<'py>(
    py: Python<'py>,
    image_of_row: PyReadonlyArray1<'py, i64>,
    owner_of_row: PyReadonlyArray1<'py, i64>,
    xy_px: PyReadonlyArray2<'py, f64>,
    reach_px: PyReadonlyArray1<'py, f64>,
    radius_px: PyReadonlyArray1<'py, f64>,
    n_owners: usize,
    ratio: f64,
    min_fine_radius_px: f64,
    min_observations: usize,
    protected: Option<PyReadonlyArray1<'py, bool>>,
) -> PyResult<Bound<'py, PyDict>> {
    let n = image_of_row.shape()[0];
    if xy_px.shape() != [n, 2] {
        return Err(PyValueError::new_err(format!(
            "xy_px must have shape ({n}, 2), got {:?}",
            xy_px.shape()
        )));
    }
    for (name, len) in [
        ("owner_of_row", owner_of_row.shape()[0]),
        ("reach_px", reach_px.shape()[0]),
        ("radius_px", radius_px.shape()[0]),
    ] {
        if len != n {
            return Err(PyValueError::new_err(format!(
                "image_of_row states {n} rows and {name} {len}"
            )));
        }
    }
    if let Some(mask) = &protected {
        if mask.shape()[0] != n {
            return Err(PyValueError::new_err(format!(
                "image_of_row states {n} rows and protected {}",
                mask.shape()[0]
            )));
        }
    }

    let images = to_contiguous!(image_of_row);
    let owners = to_contiguous!(owner_of_row);
    let xy = to_contiguous!(xy_px);
    let reach = to_contiguous!(reach_px);
    let radius = to_contiguous!(radius_px);
    let mask = protected.as_ref().map(|m| to_contiguous!(m));

    let options = CoveredOptions {
        ratio,
        min_fine_radius_px,
        min_observations,
    };
    let CoveredByFiner {
        flagged,
        keep_row,
        keep_owner,
        census,
    } = py
        .detach(|| {
            core_covered_by_finer(
                CoveredRows {
                    image_of_row: &images,
                    owner_of_row: &owners,
                    xy_px: &xy,
                    reach_px: &reach,
                    radius_px: &radius,
                    protected: mask.as_deref(),
                },
                n_owners,
                &options,
                &Progress::none(),
            )
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let c = PyDict::new(py);
    c.set_item("rows", census.rows)?;
    c.set_item("pairs_contained", census.pairs_contained)?;
    c.set_item("pairs_finer", census.pairs_finer)?;
    c.set_item("rows_flagged", census.rows_flagged)?;
    c.set_item("rows_spared", census.rows_spared)?;
    c.set_item("rows_removed", census.rows_removed)?;
    c.set_item(
        "owners_dropped_all_covered",
        census.owners_dropped_all_covered,
    )?;
    c.set_item("owners_dropped_by_sweep", census.owners_dropped_by_sweep)?;
    c.set_item("owners_kept", census.owners_kept)?;

    let out = PyDict::new(py);
    out.set_item("flagged", PyArray1::from_vec(py, flagged))?;
    out.set_item("keep_row", PyArray1::from_vec(py, keep_row))?;
    out.set_item("keep_owner", PyArray1::from_vec(py, keep_owner))?;
    out.set_item("census", c)?;
    Ok(out)
}

// ── Registration ──────────────────────────────────────────────────────────

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(covered_by_finer, m)?)?;
    Ok(())
}
