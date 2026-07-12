// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for the alternating-least-squares affine factorization
//! and its metric upgrade (see `specs/core/affine-factorization.md`).

use std::borrow::Cow;

use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;

use sfmtool_core::geometry::affine_factorization::{
    factorize_affine as core_factorize_affine, metric_upgrade as core_metric_upgrade,
    AffineFactorization, AffineFactorizationParams, MetricHypothesis,
};

use crate::matching::cluster::extract_u32_1d;

/// Result of ``factorize_affine``: per-image affine cameras, per-cluster 3D
/// points in a shared affine frame (defined up to an invertible 3×3 gauge),
/// and the per-observation residuals and keep mask from the final round.
#[pyclass(name = "AffineFactorization", module = "sfmtool.geometry", frozen)]
pub struct PyAffineFactorization {
    inner: AffineFactorization,
}

#[pymethods]
impl PyAffineFactorization {
    /// Per-image affine camera matrices ``M_i`` as numpy (N, 2, 3) float64.
    #[getter]
    fn cameras<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        let n = self.inner.cameras.len();
        let flat: Vec<f64> = self
            .inner
            .cameras
            .iter()
            .flatten()
            .flatten()
            .copied()
            .collect();
        Ok(PyArray1::from_vec(py, flat)
            .reshape([n, 2, 3])?
            .into_any()
            .unbind())
    }

    /// Per-image translations ``t_i`` as numpy (N, 2) float64.
    #[getter]
    fn translations<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        let n = self.inner.translations.len();
        let flat: Vec<f64> = self.inner.translations.iter().flatten().copied().collect();
        Ok(PyArray1::from_vec(py, flat)
            .reshape([n, 2])?
            .into_any()
            .unbind())
    }

    /// Per-cluster 3D points (affine frame) as numpy (C, 3) float64.
    #[getter]
    fn points<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        let c = self.inner.points.len();
        let flat: Vec<f64> = self.inner.points.iter().flatten().copied().collect();
        Ok(PyArray1::from_vec(py, flat)
            .reshape([c, 3])?
            .into_any()
            .unbind())
    }

    /// Final-round per-observation residuals as numpy (K, 2) float64.
    #[getter]
    fn residuals<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        let k = self.inner.residuals.len();
        let flat: Vec<f64> = self.inner.residuals.iter().flatten().copied().collect();
        Ok(PyArray1::from_vec(py, flat)
            .reshape([k, 2])?
            .into_any()
            .unbind())
    }

    /// Per-observation keep mask as numpy (K,) bool.
    #[getter]
    fn keep<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<bool>> {
        PyArray1::from_slice(py, &self.inner.keep)
    }

    /// Per-image used mask (≥ 4 kept observations) as numpy (N,) bool.
    #[getter]
    fn used_images<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<bool>> {
        PyArray1::from_slice(py, &self.inner.used_images)
    }

    /// Metric upgrade: the 3×3 gauge making the affine cameras
    /// rotation-times-scale, as both reflection hypotheses (the
    /// factorization cannot distinguish them).
    ///
    /// Returns:
    ///     A pair of ``MetricHypothesis`` objects, or ``None`` when no image
    ///     is used or the constraint system is degenerate.
    fn metric_upgrade(&self) -> Option<(PyMetricHypothesis, PyMetricHypothesis)> {
        let [a, b] = core_metric_upgrade(&self.inner)?;
        Some((
            PyMetricHypothesis { inner: a },
            PyMetricHypothesis { inner: b },
        ))
    }
}

/// One reflection hypothesis of the metric upgrade: the gauge ``A`` and the
/// per-image rotation/scale decomposition of ``M_i·A`` (identity rotation
/// and zero scale where the image is unused).
#[pyclass(name = "MetricHypothesis", module = "sfmtool.geometry", frozen)]
pub struct PyMetricHypothesis {
    inner: MetricHypothesis,
}

#[pymethods]
impl PyMetricHypothesis {
    /// The gauge ``A`` as numpy (3, 3) float64.
    #[getter]
    fn gauge<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        let flat: Vec<f64> = self.inner.gauge.iter().flatten().copied().collect();
        Ok(PyArray1::from_vec(py, flat)
            .reshape([3, 3])?
            .into_any()
            .unbind())
    }

    /// Per-image rotations as numpy (N, 3, 3) float64; identity where unused.
    #[getter]
    fn rotations<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        let n = self.inner.rotations.len();
        let flat: Vec<f64> = self
            .inner
            .rotations
            .iter()
            .flatten()
            .flatten()
            .copied()
            .collect();
        Ok(PyArray1::from_vec(py, flat)
            .reshape([n, 3, 3])?
            .into_any()
            .unbind())
    }

    /// Per-image scales as numpy (N,) float64; 0 where unused.
    #[getter]
    fn scales<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.inner.scales)
    }
}

/// Alternating-least-squares affine factorization with residual trimming
/// (Tomasi-Kanade with missing data; see
/// ``specs/core/affine-factorization.md``). Jointly estimates an affine
/// camera per image, a 3D point per cluster (shared affine frame, defined up
/// to a 3×3 gauge), and a per-observation keep mask. Deterministic.
///
/// Args:
///     obs_clusters: (K,) uint32 per-observation cluster index.
///     obs_images: (K,) uint32 per-observation image index.
///     obs_xy: (K, 2) float64 centered 2D positions (the caller subtracts
///         its chosen image center).
///     num_images: Number of images the indexes refer to.
///     num_clusters: Number of clusters the indexes refer to.
///     rounds: Fixed alternation round count (default 25); trimming runs
///         from round ``rounds // 2`` onward.
///     trim_fraction: Per-trim fraction (default 0.05): each trimming round
///         keeps the observations with residual norm strictly below the
///         ``1 - trim_fraction`` quantile (numpy-default linear
///         interpolation) of the currently-kept norms.
///
/// Returns:
///     An ``AffineFactorization``.
#[pyfunction]
#[pyo3(signature = (obs_clusters, obs_images, obs_xy, num_images, num_clusters, rounds=25, trim_fraction=0.05))]
#[allow(clippy::too_many_arguments)]
pub fn factorize_affine(
    obs_clusters: &Bound<'_, PyAny>,
    obs_images: &Bound<'_, PyAny>,
    obs_xy: PyReadonlyArray2<'_, f64>,
    num_images: usize,
    num_clusters: usize,
    rounds: usize,
    trim_fraction: f64,
) -> PyResult<PyAffineFactorization> {
    let obs_clusters = extract_u32_1d(obs_clusters, "obs_clusters")?;
    let obs_images = extract_u32_1d(obs_images, "obs_images")?;
    let clusters: Cow<'_, [u32]> = to_contiguous!(obs_clusters);
    let images: Cow<'_, [u32]> = to_contiguous!(obs_images);
    if obs_xy.shape()[1] != 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "obs_xy must have shape (K, 2), got (K, {})",
            obs_xy.shape()[1]
        )));
    }
    let xy_view = obs_xy.as_array();
    let xy: Vec<[f64; 2]> = xy_view.rows().into_iter().map(|r| [r[0], r[1]]).collect();
    let params = AffineFactorizationParams {
        rounds,
        trim_fraction,
    };
    let inner = core_factorize_affine(&clusters, &images, &xy, num_images, num_clusters, &params)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(PyAffineFactorization { inner })
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyAffineFactorization>()?;
    m.add_class::<PyMetricHypothesis>()?;
    m.add_function(pyo3::wrap_pyfunction!(factorize_affine, m)?)?;
    Ok(())
}
