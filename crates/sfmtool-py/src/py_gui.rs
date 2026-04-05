// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for the GUI viewer.

use numpy::{PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;

/// Opens a 3D viewer window with the given point cloud data.
#[pyfunction]
pub fn view3d(
    _py: Python<'_>,
    positions: PyReadonlyArray2<f64>,
    colors: PyReadonlyArray2<u8>,
) -> PyResult<()> {
    let _positions = positions.as_array();
    let _colors = colors.as_array();

    // TODO: Launch GUI window with the point cloud
    println!("view3d called with {} points", positions.shape()[0]);

    Ok(())
}

/// Opens a 3D viewer window with camera frustums.
#[pyfunction]
pub fn view3d_with_cameras(
    _py: Python<'_>,
    positions: PyReadonlyArray2<f64>,
    colors: PyReadonlyArray2<u8>,
    camera_positions: PyReadonlyArray2<f64>,
    camera_quaternions: PyReadonlyArray2<f64>,
) -> PyResult<()> {
    let _positions = positions.as_array();
    let _colors = colors.as_array();
    let _camera_positions = camera_positions.as_array();
    let _camera_quaternions = camera_quaternions.as_array();

    // TODO: Launch GUI window with point cloud and cameras
    println!(
        "view3d_with_cameras called with {} points and {} cameras",
        positions.shape()[0],
        camera_positions.shape()[0]
    );

    Ok(())
}
