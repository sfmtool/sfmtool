// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Bindings for the reconstruction core types: the `SfmrReconstruction`
//! wrapper (with its `clone_with_changes` editor), the `EditedReconstruction`
//! overlay of point edits on a shared base with the `PointMap` its edits report
//! their index effect in, and the `RangeExpr` integer-range parser used for
//! image/frame selection, plus the rule-carrying triangulation of a track set.

use pyo3::prelude::*;

pub mod add_image_to_tracks;
pub mod clone;
pub mod edited;
pub mod range_expr;
pub mod sfmr_reconstruction;
pub mod switch_camera_model;
pub mod triangulate_points;

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<sfmr_reconstruction::PySfmrReconstruction>()?;
    m.add_class::<edited::PyEditedReconstruction>()?;
    m.add_class::<edited::PyPointMap>()?;
    m.add_class::<range_expr::PyRangeExpr>()?;
    triangulate_points::register(m)?;
    Ok(())
}
