// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Bindings for spatial indices: 2D/3D KD-trees and the randomized kd-tree
//! forest ANN index.

use pyo3::prelude::*;

pub mod kdforest;
pub mod kdtree;

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    kdtree::register(m)?;
    kdforest::register(m)?;
    Ok(())
}
