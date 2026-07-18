// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for cluster covisibility (see
//! `specs/core/cluster-covisibility.md`): per-image-pair shared-cluster
//! counts plus the grouping queries built on them, and the selection queries
//! (pair displacement, banded thinning, reach; see
//! `specs/core/covisibility-selection.md`).

use std::borrow::Cow;
use std::path::PathBuf;

use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1, PyUntypedArrayMethods};
use pyo3::prelude::*;

use matches_format::ClusterMemberStatus;
use sfmtool_core::features::cluster_match::covisibility::{
    ClusterCovisibility as CoreClusterCovisibility, SeedGroupParams,
};

use super::cluster::extract_u32_1d;

/// Extract an `(M, 2)` `float64` positions array as `Vec<[f64; 2]>`, with a
/// clear error if the dtype or shape is wrong.
fn extract_positions_xy(arr: &Bound<'_, PyAny>, what: &str) -> PyResult<Vec<[f64; 2]>> {
    let arr = arr.extract::<numpy::PyReadonlyArray2<f64>>().map_err(|_| {
        let dtype = arr
            .getattr("dtype")
            .and_then(|d| d.getattr("name"))
            .and_then(|n| n.extract::<String>())
            .unwrap_or_else(|_| "?".to_string());
        pyo3::exceptions::PyTypeError::new_err(format!(
            "{what} must be an (M, 2) float64 array, got {dtype}"
        ))
    })?;
    if arr.shape()[1] != 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{what} must be an (M, 2) array; got width {}",
            arr.shape()[1]
        )));
    }
    Ok(arr
        .as_array()
        .rows()
        .into_iter()
        .map(|r| [r[0], r[1]])
        .collect())
}

/// Extract a 1-D `bool` array, with a clear error if the dtype is wrong.
fn extract_bool_1d<'py>(
    arr: &Bound<'py, PyAny>,
    what: &str,
) -> PyResult<PyReadonlyArray1<'py, bool>> {
    arr.extract::<PyReadonlyArray1<bool>>().map_err(|_| {
        let dtype = arr
            .getattr("dtype")
            .and_then(|d| d.getattr("name"))
            .and_then(|n| n.extract::<String>())
            .unwrap_or_else(|_| "?".to_string());
        pyo3::exceptions::PyTypeError::new_err(format!(
            "{what} must be a 1-D bool array, got {dtype}"
        ))
    })
}

/// Per-image-pair shared-cluster counts (symmetric, zero diagonal) with the
/// grouping queries built on them: greedy mutually-covisible seed groups and
/// candidate ranking. Construct via ``from_arrays`` or ``from_matches_file``.
///
/// This is *cluster* covisibility — a pre-reconstruction quantity computed
/// from match clusters, distinct from the post-reconstruction shared-3D-track
/// covisibility of ``sfm analyze --coviz``.
#[pyclass(name = "ClusterCovisibility", module = "sfmtool.matching", frozen)]
pub struct PyClusterCovisibility {
    inner: CoreClusterCovisibility,
}

#[pymethods]
impl PyClusterCovisibility {
    /// Build the count matrix from CSR cluster arrays (the layout
    /// ``read_matches`` returns for the ``clusters/`` section).
    ///
    /// Args:
    ///     cluster_starts: (C+1,) uint32 CSR offsets; cluster c owns members
    ///         cluster_starts[c]:cluster_starts[c+1].
    ///     member_images: (M,) uint32 member image index.
    ///     num_images: Number of images the indexes refer to.
    ///     member_accepted: Optional (M,) bool mask; only accepted members
    ///         count. None (default) counts every member. Each cluster
    ///         contributes at most 1 to any pair; clusters spanning fewer
    ///         than 2 accepted images contribute nothing.
    ///     positions_xy: Optional (M, 2) float64 per-member observation
    ///         positions (pixels), parallel to member_images. Enables the
    ///         displacement queries (``pair_displacement``,
    ///         ``pair_displacement_counts``) and the isolation-ordered
    ///         thinning sweep: one seeded sampled pass at construction draws
    ///         one uniform distinct-member pair per multi-member cluster
    ///         (same-image pairs skipped). The counts are unchanged.
    ///     seed: Seed for the displacement sampling pass (default 0).
    #[staticmethod]
    #[pyo3(signature = (cluster_starts, member_images, num_images, member_accepted=None,
                        positions_xy=None, seed=0))]
    fn from_arrays(
        cluster_starts: &Bound<'_, PyAny>,
        member_images: &Bound<'_, PyAny>,
        num_images: usize,
        member_accepted: Option<&Bound<'_, PyAny>>,
        positions_xy: Option<&Bound<'_, PyAny>>,
        seed: u64,
    ) -> PyResult<Self> {
        let cluster_starts = extract_u32_1d(cluster_starts, "cluster_starts")?;
        let member_images = extract_u32_1d(member_images, "member_images")?;
        let starts: Cow<'_, [u32]> = to_contiguous!(cluster_starts);
        let images: Cow<'_, [u32]> = to_contiguous!(member_images);
        let accepted = member_accepted
            .map(|arr| extract_bool_1d(arr, "member_accepted"))
            .transpose()?;
        let mask: Option<Cow<'_, [bool]>> = accepted.as_ref().map(|a| to_contiguous!(a));
        let positions = positions_xy
            .map(|arr| extract_positions_xy(arr, "positions_xy"))
            .transpose()?;
        let inner = CoreClusterCovisibility::from_clusters_with_positions(
            &starts,
            &images,
            mask.as_deref(),
            num_images,
            positions.as_deref(),
            seed,
        )
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Build the count matrix from a cluster-bearing ``.matches`` file.
    ///
    /// When the file carries a ``cluster_patches/`` section, the default
    /// acceptance mask is status ∈ {reference, kept}; otherwise every member
    /// counts. Custom masks: use ``read_matches`` + numpy + ``from_arrays``.
    ///
    /// Args:
    ///     path: ``.matches`` file path (str or Path). The file must store
    ///         the cluster backbone (``clusters/``), not the pairwise one.
    #[staticmethod]
    fn from_matches_file(path: PathBuf) -> PyResult<Self> {
        let data = matches_format::read_matches(&path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        let clusters = data.clusters.ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "{}: no clusters/ section — cluster covisibility needs the cluster \
                 backbone, not pairwise matches",
                path.display()
            ))
        })?;
        let mask: Option<Vec<bool>> = data.cluster_patches.as_ref().map(|cp| {
            cp.member_status
                .iter()
                .map(|&s| {
                    s == ClusterMemberStatus::Reference as u8
                        || s == ClusterMemberStatus::Kept as u8
                })
                .collect()
        });
        let inner = CoreClusterCovisibility::from_clusters(
            clusters.cluster_starts.as_slice().expect("contiguous"),
            clusters.member_images.as_slice().expect("contiguous"),
            mask.as_deref(),
            data.image_names.len(),
        )
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Number of images the matrix covers.
    #[getter]
    fn num_images(&self) -> usize {
        self.inner.num_images()
    }

    /// The full count matrix as a numpy (N, N) uint32 copy.
    #[getter]
    fn counts<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        let n = self.inner.num_images();
        let mut flat: Vec<u32> = Vec::with_capacity(n * n);
        for i in 0..n {
            flat.extend_from_slice(self.inner.row(i as u32));
        }
        Ok(PyArray1::from_vec(py, flat)
            .reshape([n, n])?
            .into_any()
            .unbind())
    }

    /// Mean sampled feature displacement per covisible pair as a numpy
    /// (N, N) float64 copy (symmetric, 0 where no sample landed).
    ///
    /// Raises ValueError when constructed without ``positions_xy``.
    fn pair_displacement<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        let mean = self.inner.pair_displacement().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "constructed without positions_xy — displacement queries are unavailable",
            )
        })?;
        let n = self.inner.num_images();
        Ok(PyArray1::from_slice(py, mean)
            .reshape([n, n])?
            .into_any()
            .unbind())
    }

    /// Samples behind each ``pair_displacement`` mean as a numpy (N, N)
    /// uint32 copy, for callers that gate on support.
    ///
    /// Raises ValueError when constructed without ``positions_xy``.
    fn pair_displacement_counts<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        let count = self.inner.pair_displacement_counts().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "constructed without positions_xy — displacement queries are unavailable",
            )
        })?;
        let n = self.inner.num_images();
        Ok(PyArray1::from_slice(py, count)
            .reshape([n, n])?
            .into_any()
            .unbind())
    }

    /// Redundancy-thinned subset as a sorted uint32 numpy array: a greedy
    /// sweep in decreasing isolation (largest nearest-covisible-partner
    /// displacement first; construction order without positions) keeps an
    /// image only when its best shared-cluster count against the kept set
    /// falls in the band [tau/8, tau). The first swept image is always kept.
    ///
    /// Args:
    ///     tau: Band upper bound (exclusive); must be finite.
    fn thin<'py>(&self, py: Python<'py>, tau: f64) -> PyResult<Py<PyAny>> {
        if !tau.is_finite() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "tau must be finite, got {tau}"
            )));
        }
        Ok(PyArray1::from_vec(py, self.inner.thin(tau))
            .into_any()
            .unbind())
    }

    /// Thin to approximately ``target`` images (sorted uint32 numpy array):
    /// binary-searches ``tau`` (the kept count grows monotonically with tau)
    /// and returns the subset whose size lands closest to ``target``.
    ///
    /// Args:
    ///     target: Requested subset size.
    fn thin_to<'py>(&self, py: Python<'py>, target: usize) -> PyResult<Py<PyAny>> {
        Ok(PyArray1::from_vec(py, self.inner.thin_to(target))
            .into_any()
            .unbind())
    }

    /// Fraction of all images sharing at least ``min_shared`` clusters with
    /// at least one image of ``images`` (subset members count as reached).
    /// An empty subset reaches nothing (0.0).
    ///
    /// Args:
    ///     images: (K,) uint32 subset image indexes.
    ///     min_shared: Shared-cluster bar (default 8).
    #[pyo3(signature = (images, min_shared=8))]
    fn reach(&self, images: &Bound<'_, PyAny>, min_shared: u32) -> PyResult<f64> {
        let images = extract_u32_1d(images, "images")?;
        let subset: Cow<'_, [u32]> = to_contiguous!(images);
        let n = self.inner.num_images();
        if let Some(&bad) = subset.iter().find(|&&i| i as usize >= n) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "subset image index {bad} out of range for {n} images"
            )));
        }
        Ok(self.inner.reach(&subset, min_shared))
    }

    /// ``candidates`` reordered by descending covisibility with ``image``
    /// (ties: ascending index); zero-covisibility candidates are dropped.
    ///
    /// Args:
    ///     image: Query image index.
    ///     candidates: (K,) uint32 candidate image indexes.
    ///
    /// Returns:
    ///     uint32 numpy array (at most K entries).
    fn rank_by_covisibility<'py>(
        &self,
        py: Python<'py>,
        image: u32,
        candidates: &Bound<'py, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let candidates = extract_u32_1d(candidates, "candidates")?;
        let cands: Cow<'_, [u32]> = to_contiguous!(candidates);
        let n = self.inner.num_images();
        if image as usize >= n {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "image index {image} out of range for {n} images"
            )));
        }
        if let Some(&bad) = cands.iter().find(|&&c| c as usize >= n) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "candidate image index {bad} out of range for {n} images"
            )));
        }
        let ranked = self.inner.rank_by_covisibility(image, &cands);
        Ok(PyArray1::from_vec(py, ranked).into_any().unbind())
    }

    /// Lazy iterator of greedy mutually-covisible seed groups (list[int],
    /// sorted ascending), per the spec's Seed-group algorithm: each step
    /// takes the strongest remaining edge and greedily extends it maximizing
    /// the minimum shared count vs the group, then excludes the yielded
    /// images. Deterministic; groups are disjoint; ends when the strongest
    /// remaining edge is below ``min_shared``.
    ///
    /// Args:
    ///     group_size: Maximum images per group (default 5).
    ///     min_shared: Minimum within-group pairwise covisibility (default 8).
    #[pyo3(signature = (group_size=5, min_shared=8))]
    fn seed_groups(slf: PyRef<'_, Self>, group_size: usize, min_shared: u32) -> PySeedGroups {
        let num_images = slf.inner.num_images();
        PySeedGroups {
            parent: slf.into(),
            excluded: vec![false; num_images],
            params: SeedGroupParams {
                group_size,
                min_shared,
            },
        }
    }
}

/// Lazy iterator over a [`PyClusterCovisibility`]'s seed groups. Holds a
/// reference to its parent plus the excluded-image mask; each ``__next__``
/// re-runs the shared core step function against the parent's counts, so the
/// sequence is identical to the core `SeedGroups` iterator.
#[pyclass(name = "ClusterCovisibilitySeedGroups", module = "sfmtool.matching")]
pub struct PySeedGroups {
    parent: Py<PyClusterCovisibility>,
    excluded: Vec<bool>,
    params: SeedGroupParams,
}

#[pymethods]
impl PySeedGroups {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self) -> Option<Vec<u32>> {
        self.parent
            .get()
            .inner
            .next_seed_group(&mut self.excluded, &self.params)
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyClusterCovisibility>()?;
    m.add_class::<PySeedGroups>()?;
    Ok(())
}
