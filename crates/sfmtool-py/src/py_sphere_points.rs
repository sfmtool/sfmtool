// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Python binding for evenly-distributed sphere point generation.

use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;

use sfmtool_core::sphere_points::{evenly_distributed_sphere_points as core_generate, RelaxConfig};

/// Generate N points evenly distributed on the unit sphere.
///
/// Samples points uniformly on the sphere, then runs iterative Thomson-style
/// 1/r² repulsion relaxation (with KD-tree neighbor lookup) to push them
/// towards an evenly-distributed configuration.
///
/// Args:
///     n: Number of points to generate.
///     iterations: Number of relaxation iterations (default 50).
///     step_size: Step length per iteration, as a fraction of the
///         characteristic NN spacing √(4π/n) (default 0.05).
///     cutoff_multiplier: Per-point neighbor cutoff radius, as a multiple
///         of the characteristic NN spacing (default 5.0).
///     seed: Optional u64 seed for the initial uniform random sampling.
///         When provided the result is reproducible across runs; when `None`
///         (default) the thread-local RNG is used.
///
/// Returns:
///     (n, 3) float32 array of unit-norm points.
#[pyfunction]
#[pyo3(signature = (n, iterations=50, step_size=0.05, cutoff_multiplier=5.0, seed=None))]
pub fn evenly_distributed_sphere_points<'py>(
    py: Python<'py>,
    n: usize,
    iterations: usize,
    step_size: f32,
    cutoff_multiplier: f32,
    seed: Option<u64>,
) -> PyResult<Py<PyAny>> {
    let config = RelaxConfig {
        iterations,
        step_size,
        cutoff_multiplier,
        seed,
    };
    let points = py.detach(|| core_generate(n, &config));
    let arr = PyArray1::from_vec(py, points).reshape([n, 3])?;
    Ok(arr.into_any().unbind())
}
