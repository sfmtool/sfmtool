// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Opaque `.matches` handle: one parse, cheap array access, and file-level
//! cluster selection (`select_clusters`) producing new handles.
//!
//! The dict-shaped `read_matches` stays for whole-file consumers; this class
//! serves pipelines that parse once and then slice — array accessors copy on
//! access, and `select_clusters` runs the `sfmtool-matches-format` derivation
//! (`specs/formats/matches-file-format.md` § "Cluster Selection").

use std::borrow::Cow;
use std::path::PathBuf;
use std::str::FromStr;

use numpy::{PyReadonlyArray1, ToPyArray};
use pyo3::prelude::*;

use sfmtool_matches_format::{ClusterCellStatus, ClusterMemberStatus, ClusterSelect, MatchesData};

use crate::helpers::serde_to_py;

/// A parsed `.matches` file (or an in-memory selection derived from one).
///
/// Construct from a path; array accessors return numpy copies. The file must
/// use the cluster backbone for the cluster accessors and `select_clusters`;
/// pairwise files can still be opened for the image-table accessors.
#[pyclass(name = "MatchesFile", module = "sfmtool.fileio", frozen)]
pub struct PyMatchesFile {
    inner: MatchesData,
}

impl PyMatchesFile {
    /// The parsed file behind the handle, for the bindings that hand a whole
    /// `.matches` to a core entry point taking `&MatchesData` (the vote's
    /// object form) instead of re-deriving its arrays through Python.
    pub(crate) fn data(&self) -> &MatchesData {
        &self.inner
    }

    fn clusters(&self) -> PyResult<&sfmtool_matches_format::ClustersData> {
        self.inner.clusters.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "no clusters/ section — this file stores the pairwise backbone",
            )
        })
    }

    /// The one resolution every image of the file shares, as the Python
    /// error a file that states none deserves.
    fn shared_image_dims(&self) -> PyResult<(u32, u32)> {
        self.inner.shared_image_dims().map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "this file states no single image resolution: {e}"
            ))
        })
    }

    fn cluster_patches(&self) -> PyResult<&sfmtool_matches_format::ClusterPatchData> {
        self.inner.cluster_patches.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("no cluster_patches/ section in this file")
        })
    }
}

/// Parse one accepted-status item: a `member_status` discriminant int or a
/// canonical status name string (e.g. `"kept"`).
fn parse_status(item: &Bound<'_, PyAny>) -> PyResult<ClusterMemberStatus> {
    if let Ok(v) = item.extract::<u8>() {
        return ClusterMemberStatus::from_u8(v).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "invalid ClusterMemberStatus discriminant {v} (valid: 0..=6)"
            ))
        });
    }
    let s: String = item.extract().map_err(|_| {
        pyo3::exceptions::PyTypeError::new_err(
            "accepted_statuses items must be status ints (0..=6) or names (e.g. 'kept')",
        )
    })?;
    ClusterMemberStatus::from_str(&s)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}

/// The `restrict_cluster_ids` argument as the ids the selection takes.
///
/// A `uint32` array — what `analysis.coarsest_cluster_ids` hands back — is read
/// straight off its buffer. Any other sequence of ints falls through to PyO3's
/// element-wise extraction, which is where a negative or too-wide value is
/// refused.
fn cluster_ids(obj: &Bound<'_, PyAny>) -> PyResult<Vec<u32>> {
    if let Ok(a) = obj.extract::<PyReadonlyArray1<'_, u32>>() {
        let ids: Cow<'_, [u32]> = to_contiguous!(a);
        return Ok(ids.into_owned());
    }
    obj.extract()
}

#[pymethods]
impl PyMatchesFile {
    /// Read a `.matches` file into a handle.
    ///
    /// Args:
    ///     path: `.matches` file path (str or Path).
    #[new]
    fn new(path: PathBuf) -> PyResult<Self> {
        let inner = sfmtool_matches_format::read_matches(&path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Top-level metadata as a dict (same shape as `read_matches_metadata`).
    #[getter]
    fn metadata(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        serde_to_py(py, &self.inner.metadata)
    }

    /// The whole-file `content_xxh128` hex string (empty for an in-memory
    /// selection that has not been saved).
    #[getter]
    fn content_xxh128(&self) -> &str {
        &self.inner.content_hash.content_xxh128
    }

    /// Image paths relative to the workspace directory (POSIX format).
    #[getter]
    fn image_names(&self) -> Vec<String> {
        self.inner.image_names.clone()
    }

    /// `(N, 2)` per-image (width, height) uint32 array, or None for a
    /// version <= 3 file that never stored dimensions.
    #[getter]
    fn image_dims<'py>(&self, py: Python<'py>) -> Option<Py<PyAny>> {
        self.inner
            .image_dims
            .as_ref()
            .map(|d| d.to_pyarray(py).into_any().unbind())
    }

    /// `(N,)` uint32 feature count per image as used during matching.
    #[getter]
    fn feature_counts<'py>(&self, py: Python<'py>) -> Py<PyAny> {
        self.inner.feature_counts.to_pyarray(py).into_any().unbind()
    }

    /// Number of images in the image table.
    #[getter]
    fn image_count(&self) -> u32 {
        self.inner.metadata.image_count
    }

    /// The pixel width every image of the file shares. Raises `ValueError`
    /// when the file records no dimensions or its images differ in
    /// resolution.
    #[getter]
    fn image_width(&self) -> PyResult<u32> {
        Ok(self.shared_image_dims()?.0)
    }

    /// The pixel height every image of the file shares, under the same rule
    /// as `image_width`.
    #[getter]
    fn image_height(&self) -> PyResult<u32> {
        Ok(self.shared_image_dims()?.1)
    }

    /// Number of clusters (`cluster_starts` length minus one).
    #[getter]
    fn cluster_count(&self) -> PyResult<u32> {
        self.clusters()?;
        Ok(self
            .inner
            .metadata
            .cluster_count
            .expect("a cluster-bearing file carries cluster_count"))
    }

    /// Total number of cluster members across all clusters.
    #[getter]
    fn member_count(&self) -> PyResult<u32> {
        self.clusters()?;
        Ok(self
            .inner
            .metadata
            .cluster_member_count
            .expect("a cluster-bearing file carries cluster_member_count"))
    }

    /// Whether the file stores the cluster backbone.
    #[getter]
    fn has_clusters(&self) -> bool {
        self.inner.clusters.is_some()
    }

    /// Whether the `cluster_patches/` enrichment is present.
    #[getter]
    fn has_cluster_patches(&self) -> bool {
        self.inner.cluster_patches.is_some()
    }

    /// `(C+1,)` uint32 CSR offsets into the member arrays.
    #[getter]
    fn cluster_starts<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        Ok(self
            .clusters()?
            .cluster_starts
            .to_pyarray(py)
            .into_any()
            .unbind())
    }

    /// `(M,)` uint32 image index per member.
    #[getter]
    fn member_images<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        Ok(self
            .clusters()?
            .member_images
            .to_pyarray(py)
            .into_any()
            .unbind())
    }

    /// `(M,)` uint32 feature index (into the image's `.sift`) per member.
    #[getter]
    fn member_features<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        Ok(self
            .clusters()?
            .member_features
            .to_pyarray(py)
            .into_any()
            .unbind())
    }

    /// Cluster matcher options recorded in `clusters/metadata.json.zst`.
    #[getter]
    fn matcher_options(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        serde_to_py(py, &self.clusters()?.matcher_options)
    }

    /// `(C,)` uint32 global member index of each cluster's reference;
    /// `0xFFFFFFFF` when absent (source-unrefinable, or — in a selection —
    /// a reference outside the restriction).
    #[getter]
    fn reference_members<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        Ok(self
            .cluster_patches()?
            .reference_members
            .to_pyarray(py)
            .into_any()
            .unbind())
    }

    /// `(M,)` uint8 ClusterMemberStatus discriminants.
    #[getter]
    fn member_status<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        Ok(self
            .cluster_patches()?
            .member_status
            .to_pyarray(py)
            .into_any()
            .unbind())
    }

    /// `(M,)` float32 achieved windowed ZNCC vs the reference (NaN where
    /// not evaluated).
    #[getter]
    fn member_zncc<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        Ok(self
            .cluster_patches()?
            .member_zncc
            .to_pyarray(py)
            .into_any()
            .unbind())
    }

    /// `(M,)` float32 translation drift from the SIFT seed (NaN where not
    /// evaluated).
    #[getter]
    fn member_shift_px<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        Ok(self
            .cluster_patches()?
            .member_shift_px
            .to_pyarray(py)
            .into_any()
            .unbind())
    }

    /// `(M,)` float32 warp-consistency residual (NaN where the member did
    /// not enter the fit).
    #[getter]
    fn member_consistency_residual<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        Ok(self
            .cluster_patches()?
            .member_consistency_residual
            .to_pyarray(py)
            .into_any()
            .unbind())
    }

    /// Whether the file carries the piecewise refinement's per-cell columns
    /// (format version 8). False for a file without a cluster_patches/
    /// section, for a file below version 8, and for one whose refinement did
    /// not run the piecewise stage.
    #[getter]
    fn has_member_cells(&self) -> bool {
        self.inner
            .cluster_patches
            .as_ref()
            .is_some_and(|cp| cp.member_cells.is_some())
    }

    /// `(M, 3, 3, 2)` float32 each cell's displacement `[x, y]` from where the
    /// member's affine shape places it, in patch grid px, cells `[m, row,
    /// col]` from the top-left; NaN where no shift was measured.
    ///
    /// None when the cluster_patches/ section carries no cells; raises
    /// `ValueError`, like the section's other getters, when the file has no
    /// cluster_patches/ section.
    #[getter]
    fn member_cell_shift_px<'py>(&self, py: Python<'py>) -> PyResult<Option<Py<PyAny>>> {
        Ok(self
            .cluster_patches()?
            .member_cells
            .as_ref()
            .map(|cells| cells.shift_px.to_pyarray(py).into_any().unbind()))
    }

    /// `(M, 3, 3)` float32 each cell's ZNCC against the reference at its best
    /// shift, averaged over the template's textured colour channels; NaN
    /// where nothing was read.
    ///
    /// None when the cluster_patches/ section carries no cells; raises
    /// `ValueError`, like the section's other getters, when the file has no
    /// cluster_patches/ section.
    #[getter]
    fn member_cell_zncc<'py>(&self, py: Python<'py>) -> PyResult<Option<Py<PyAny>>> {
        Ok(self
            .cluster_patches()?
            .member_cells
            .as_ref()
            .map(|cells| cells.zncc.to_pyarray(py).into_any().unbind()))
    }

    /// `(M, 3, 3)` uint8 cell statuses in the canonical numbering, whatever
    /// legend the file stated: 0 fitted, 1 refused_curvature, 2 refused_zncc,
    /// 3 not_attempted, 4 refused_bound, 5 refused_outlier
    /// (`member_cell_status_names`).
    ///
    /// None when the cluster_patches/ section carries no cells; raises
    /// `ValueError`, like the section's other getters, when the file has no
    /// cluster_patches/ section.
    #[getter]
    fn member_cell_status<'py>(&self, py: Python<'py>) -> PyResult<Option<Py<PyAny>>> {
        Ok(self
            .cluster_patches()?
            .member_cells
            .as_ref()
            .map(|cells| cells.status.to_pyarray(py).into_any().unbind()))
    }

    /// The canonical names of the `member_cell_status` codes, one per code in
    /// code order.
    ///
    /// None when the cluster_patches/ section carries no cells; raises
    /// `ValueError`, like the section's other getters, when the file has no
    /// cluster_patches/ section.
    #[getter]
    fn member_cell_status_names(&self) -> PyResult<Option<Vec<&'static str>>> {
        Ok(self
            .cluster_patches()?
            .member_cells
            .as_ref()
            .map(|_| ClusterCellStatus::NAMES.to_vec()))
    }

    /// `(M,)` uint8 renders the piecewise refinement made per member; 0 for a
    /// member it did not run on.
    ///
    /// None when the cluster_patches/ section carries no cells; raises
    /// `ValueError`, like the section's other getters, when the file has no
    /// cluster_patches/ section.
    #[getter]
    fn member_cell_iterations<'py>(&self, py: Python<'py>) -> PyResult<Option<Py<PyAny>>> {
        Ok(self
            .cluster_patches()?
            .member_cells
            .as_ref()
            .map(|cells| cells.iterations.to_pyarray(py).into_any().unbind()))
    }

    /// Refinement options recorded in `cluster_patches/metadata.json.zst`.
    #[getter]
    fn refine_options(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        serde_to_py(py, &self.cluster_patches()?.refine_options)
    }

    /// The refinement patch half-width in pixels, read from whichever
    /// `refine_options` key the file carries (`patch_size` full edge / 2, or
    /// the legacy `radius` half-width as-is); None when neither is recorded.
    #[getter]
    fn refine_radius(&self) -> PyResult<Option<f64>> {
        Ok(self.cluster_patches()?.refine_radius())
    }

    /// `(M, 2)` float32 per-member keypoint positions, at THIS file's stage:
    /// the detections in a matcher output; in a cluster-patches output the
    /// refined absolute position for every member its cascade measured, and
    /// the detection for the rest. Never NaN -- `member_status` says which
    /// reading a row carries and which members stand. None for a pairwise
    /// file.
    fn member_positions<'py>(&self, py: Python<'py>) -> Option<Py<PyAny>> {
        self.inner
            .member_positions()
            .map(|p| p.to_pyarray(py).into_any().unbind())
    }

    /// `(M, 2, 2)` float32 per-member affine shapes, at the same stage and
    /// under the same reading as `member_positions`. Each shape maps the
    /// detector's canonical unit frame onto that member's image pixels, so a
    /// member's image-space extent is its column norms; the reference->member
    /// warp is ``S·S_ref**-1`` through the cluster's reference member.
    fn member_affine_shapes<'py>(&self, py: Python<'py>) -> Option<Py<PyAny>> {
        self.inner
            .member_affine_shapes()
            .map(|s| s.to_pyarray(py).into_any().unbind())
    }

    /// `(C,)` float64 per-cluster worst (maximum) finite warp-consistency
    /// residual; `inf` for clusters with no finite residual.
    fn cluster_worst_consistency<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        self.clusters()?;
        self.cluster_patches()?;
        Ok(self
            .inner
            .cluster_worst_consistency()
            .expect("sections checked above")
            .to_pyarray(py)
            .into_any()
            .unbind())
    }

    /// Derive a new handle holding only the clusters and members that pass
    /// the selection, densely renumbered. This handle is left unchanged and
    /// nothing is written; call `save` to write the result. The steps, the
    /// `0xFFFFFFFF` absent-reference sentinel and the provenance record are
    /// specified in `specs/core/features/cluster-selection.md`.
    ///
    /// Raises ValueError on a pairwise file, on `min_span < 2`, on a
    /// `restrict_images` name not in this file's image table, or on a
    /// `restrict_cluster_ids` id outside this file's cluster range.
    ///
    /// Args:
    ///     min_span: Minimum distinct selected images per cluster (>= 2,
    ///         default 2).
    ///     restrict_images: Optional collection of image NAMES; every name
    ///         must exist in this file.
    ///     restrict_cluster_ids: Optional cluster ids of THIS file (its
    ///         source ids); every id must be in range. A (n,) uint32 array —
    ///         what `analysis.coarsest_cluster_ids` returns — is read off its
    ///         buffer; any other sequence of ints is taken element-wise.
    ///     accepted_statuses: Optional member statuses to keep, as ints
    ///         (0..=6) or names ("reference", "kept", ...). Default:
    ///         reference + kept. Ignored when the file has no
    ///         cluster_patches/ section.
    #[pyo3(signature = (min_span=2, restrict_images=None, restrict_cluster_ids=None, accepted_statuses=None))]
    fn select_clusters(
        &self,
        min_span: u32,
        restrict_images: Option<Vec<String>>,
        restrict_cluster_ids: Option<Bound<'_, PyAny>>,
        accepted_statuses: Option<Vec<Bound<'_, PyAny>>>,
    ) -> PyResult<Self> {
        let mut opts = ClusterSelect {
            min_span,
            restrict_images,
            restrict_cluster_ids: restrict_cluster_ids.as_ref().map(cluster_ids).transpose()?,
            ..ClusterSelect::default()
        };
        if let Some(items) = accepted_statuses {
            opts.accepted_statuses = items
                .iter()
                .map(parse_status)
                .collect::<PyResult<Vec<_>>>()?;
        }
        let inner = self
            .inner
            .select_clusters(&opts)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Write this handle's data to a new `.matches` file (content hashes
    /// recomputed).
    ///
    /// Args:
    ///     path: Output path (str or Path).
    ///     zstd_level: Entry compression level (default 3).
    #[pyo3(signature = (path, zstd_level=3))]
    fn save(&self, path: PathBuf, zstd_level: i32) -> PyResult<()> {
        sfmtool_matches_format::write_matches(&path, &self.inner, zstd_level)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMatchesFile>()?;
    Ok(())
}
