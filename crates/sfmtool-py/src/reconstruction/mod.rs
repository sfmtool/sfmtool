// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Bindings for the reconstruction core types: the `SfmrReconstruction`
//! wrapper (with its `clone_with_changes` editor) and the `RangeExpr`
//! integer-range parser used for image/frame selection.

use pyo3::prelude::*;

pub mod clone;
pub mod range_expr;
pub mod sfmr_reconstruction;

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<sfmr_reconstruction::PySfmrReconstruction>()?;
    m.add_class::<range_expr::PyRangeExpr>()?;
    Ok(())
}
