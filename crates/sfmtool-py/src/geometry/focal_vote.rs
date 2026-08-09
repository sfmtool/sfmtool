// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Python binding for the structure-free focal vote
//! (``sfmtool._sfmtool.geometry.focal_vote``; see ``specs/core/focal-vote.md``).

use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use sfmtool_core::geometry::focal_vote::{
    focal_vote_with_options, CameraModel, ColumnDiagnostics, FocalVoteOptions, ScanCell, ScanVote,
};

/// One column's `scan_votes` entry as a Python dict.
fn scan_vote_dict<'py>(py: Python<'py>, v: &ScanVote) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item(
        "cell",
        match v.cell {
            ScanCell::Epipolar => "Epipolar",
            ScanCell::Rotation => "Rotation",
        },
    )?;
    d.set_item("image_a", v.image_a)?;
    d.set_item("image_b", v.image_b)?;
    d.set_item("focal_px", v.focal_px)?;
    d.set_item("cost", v.cost)?;
    d.set_item("sharpness", v.sharpness)?;
    d.set_item("dir_disagreement", v.dir_disagreement)?;
    d.set_item("rotation_dominated", v.rotation_dominated)?;
    d.set_item("rotation_ratio", v.rotation_ratio)?;
    d.set_item("coverage_p90", v.coverage_p90)?;
    d.set_item("n_inliers", v.n_inliers)?;
    d.set_item("in_fov_band", v.in_fov_band)?;
    d.set_item("at_grid_edge", v.at_grid_edge)?;
    d.set_item("angular_focal_px", v.angular_focal_px)?;
    d.set_item("certified", v.certified)?;
    d.set_item("model_informative", v.model_informative)?;
    Ok(d)
}

/// One [`ColumnDiagnostics`] as a Python dict.
fn column_dict<'py>(py: Python<'py>, c: &ColumnDiagnostics) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("camera_model", c.model.as_str())?;
    d.set_item("focal_px", c.focal_px)?;
    d.set_item("family", c.family.map(|f| f.as_str()))?;
    d.set_item("epipolar_focal_px", c.epipolar_focal_px)?;
    d.set_item("rotation_focal_px", c.rotation_focal_px)?;
    d.set_item("n_epipolar", c.n_epipolar)?;
    d.set_item("n_rotation", c.n_rotation)?;
    d.set_item("n_pool", c.n_pool)?;
    d.set_item("pool_spread", c.pool_spread)?;
    d.set_item("family_disagreement", c.family_disagreement)?;
    d.set_item("epipolar_spread", c.epipolar_spread)?;
    d.set_item("rotation_spread", c.rotation_spread)?;
    d.set_item("parallax_poverty", c.parallax_poverty)?;
    d.set_item("n_rotation_dominated", c.n_rotation_dominated)?;
    d.set_item("n_scanned_epipolar", c.n_scanned_epipolar)?;
    d.set_item("n_scanned_rotation", c.n_scanned_rotation)?;
    d.set_item("n_certified_epipolar", c.n_certified_epipolar)?;
    d.set_item("n_certified_rotation", c.n_certified_rotation)?;
    d.set_item("n_informative_epipolar", c.n_informative_epipolar)?;
    d.set_item("n_informative_rotation", c.n_informative_rotation)?;
    d.set_item("n_certified", c.n_certified)?;
    d.set_item("n_informative", c.n_informative)?;
    let votes = pyo3::types::PyList::empty(py);
    for v in &c.scan_votes {
        votes.append(scan_vote_dict(py, v)?)?;
    }
    d.set_item("scan_votes", votes)?;
    Ok(d)
}

/// Estimate a shared focal length from cluster-track observations without any
/// reconstruction (see ``specs/core/focal-vote.md``).
///
/// Image pairs drawn from the cluster tracks each cast one focal vote through
/// whichever estimator their geometry can observe — the Bougnoux focal of a
/// fundamental matrix (parallax-rich pairs, one vote per pair: the geometric
/// mean of its two direction-consistent focals) or rotation self-calibration of
/// a parallax-free homography (far-field pairs, one vote per unordered image
/// pair) — and the consensus focal is the log-space median of the pooled votes
/// from both families. No structure is estimated, so the vote cannot be biased
/// by the depth/focal (bas-relief) compensation of structure-based focal
/// estimation.
///
/// Args:
///     cluster_indexes: (n_obs,) uint32 cluster id per observation,
///         nondecreasing (each distinct cluster is a contiguous run).
///     image_indexes: (n_obs,) uint32 image id per observation.
///     positions_xy: (n_obs, 2) float64 full-pixel keypoint positions.
///     width: Shared image width; the principal point is the image centre.
///     height: Shared image height.
///     seed: SplitMix64 seed for the sampled pair-table pass and the RANSAC
///         estimators; same inputs + seed => bit-identical output (default 0).
///     epipolar_min_disp_frac: Wide-baseline gate for epipolar candidate
///         pairs, as a fraction of the image diagonal their mean feature
///         displacement must reach (default 0.02). Too low admits
///         near-static pairs whose fundamental matrices vote junk focals.
///     columns: Camera-model columns to evaluate, as a sequence of names
///         (``"pinhole"``, ``"equidistant"`` / ``"fisheye"``). The default
///         ``("pinhole",)`` runs no scan and reproduces the closed-form
///         kernel exactly; asking for both columns adds the self-consistency
///         scans that arbitrate between them.
///
/// Returns:
///     A dict mirroring the output table: ``{"focal_px": float | None,
///     "family": "Epipolar" | "Rotation" | None, "epipolar_focal_px":
///     float | None, "rotation_focal_px": float | None, "n_epipolar": int,
///     "n_rotation": int, "n_pool": int, "pool_spread": float,
///     "family_disagreement": float | None, "parallax_poverty": float,
///     "epipolar_spread": float, "rotation_spread": float,
///     "epipolar_votes": list[dict], "rotation_votes": list[dict],
///     "n_h_dominated": int, "n_estimator_failed": int, "n_band_rejected":
///     int, "n_degenerate": int, "n_inconsistent_pairs": int, "camera_model":
///     str | None, "columns": list[dict]}``.
///
///     ``focal_px`` is the log-space median of the pooled votes — one per
///     direction-consistent epipolar pair plus one per unordered rotation pair
///     — and is ``None`` with fewer than 2 pooled votes; ``n_pool`` is
///     ``n_epipolar + n_rotation``. Every focal median here is taken in log
///     space (an even-length median is the geometric mean of the two central
///     votes). When both families voted and ``family_disagreement`` (their
///     log-focal gap) exceeds ``0.25`` the pool is bimodal, and ``focal_px`` is
///     the majority family's median instead of a blend. ``family`` is the
///     pool's majority contributor (ties go to ``"Rotation"``), ``None`` when
///     there is no consensus; under the disagreement rule ``focal_px`` is
///     exactly that family's median. ``pool_spread`` is the log-focal
///     interquartile range of the votes behind ``focal_px`` (the whole pool, or
///     the majority family's votes under the disagreement rule).
///
///     ``n_h_dominated``, ``n_estimator_failed`` and ``n_inconsistent_pairs``
///     count candidate PAIRS; ``n_band_rejected`` and ``n_degenerate`` count
///     DIRECTIONS (a Bougnoux focal outside the plausibility band, and an
///     extraction that produced no value at all).
///
///     ``epipolar_votes`` is the diagnostic detail layer, independent of what
///     pools: every in-band directional Bougnoux focal, both directions, with
///     ``image_a``, ``image_b``, ``shared_clusters``, ``mean_disp_px``,
///     ``n_f_inliers``, ``n_h_inliers``, ``transposed``, ``focal_px``. Each
///     ``rotation_votes`` entry carries ``image``, ``partner``,
///     ``mean_disp_px``, ``n_inliers``, ``focal_px``, one entry per unordered
///     image pair — two images that are each other's widest partner are
///     reached twice by the scan and vote only on the first.
///
///     ``camera_model`` is the model verdict (the requested column with the
///     greater certified mass of model-informative scan votes, ties going to
///     ``"Pinhole"``); the top-level focal fields are the winning column's
///     consensus, and ``columns`` carries every requested column's own
///     consensus, spreads and certificate counts (plus its per-pair
///     ``scan_votes``). Column focals are never blended: a pinhole focal and
///     an equidistant focal parameterize different maps. With the default
///     pinhole-only column set no scan runs, ``columns`` is empty and
///     ``camera_model`` is ``"Pinhole"``.
// This is a Python docstring (rendered by `help()`), not Rust prose: its
// indented `Args:` / `Returns:` continuation paragraphs read as Markdown
// indented code blocks, which rustdoc then tries to parse as Rust.
#[allow(rustdoc::invalid_rust_codeblocks)]
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (cluster_indexes, image_indexes, positions_xy, width, height, *, seed=0, epipolar_min_disp_frac=0.02, columns=None))]
pub fn focal_vote<'py>(
    py: Python<'py>,
    cluster_indexes: PyReadonlyArray1<'py, u32>,
    image_indexes: PyReadonlyArray1<'py, u32>,
    positions_xy: PyReadonlyArray2<'py, f64>,
    width: u32,
    height: u32,
    seed: u64,
    epipolar_min_disp_frac: f64,
    columns: Option<Vec<String>>,
) -> PyResult<Bound<'py, PyDict>> {
    if positions_xy.shape()[1] != 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "positions_xy must have shape (n_obs, 2)",
        ));
    }
    let n_obs = cluster_indexes.shape()[0];
    if image_indexes.shape()[0] != n_obs || positions_xy.shape()[0] != n_obs {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "cluster_indexes, image_indexes, and positions_xy must share n_obs",
        ));
    }

    let clusters = to_contiguous!(cluster_indexes);
    let images = to_contiguous!(image_indexes);
    let pos_flat = to_contiguous!(positions_xy);
    let positions: Vec<[f64; 2]> = pos_flat.chunks_exact(2).map(|c| [c[0], c[1]]).collect();

    // Cluster ids must be nondecreasing (contiguous runs).
    if clusters.windows(2).any(|w| w[1] < w[0]) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "cluster_indexes must be nondecreasing",
        ));
    }

    let models = match columns {
        None => vec![CameraModel::Pinhole],
        Some(names) => names
            .iter()
            .map(|n| {
                CameraModel::from_str_name(n).ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "unknown camera-model column {n:?} (expected \"pinhole\" or \
                         \"equidistant\")"
                    ))
                })
            })
            .collect::<PyResult<Vec<_>>>()?,
    };
    let options = FocalVoteOptions {
        seed,
        epipolar_min_disp_frac,
        columns: models,
    };

    let result = py.detach(move || {
        focal_vote_with_options(&clusters, &images, &positions, width, height, &options)
    });

    let d = PyDict::new(py);
    d.set_item("focal_px", result.focal_px)?;
    d.set_item("family", result.family.map(|f| f.as_str()))?;
    d.set_item("epipolar_focal_px", result.epipolar_focal_px)?;
    d.set_item("rotation_focal_px", result.rotation_focal_px)?;
    d.set_item("n_epipolar", result.n_epipolar)?;
    d.set_item("n_rotation", result.n_rotation)?;
    d.set_item("n_pool", result.n_pool)?;
    d.set_item("pool_spread", result.pool_spread)?;
    d.set_item("family_disagreement", result.family_disagreement)?;
    d.set_item("parallax_poverty", result.parallax_poverty)?;
    d.set_item("epipolar_spread", result.epipolar_spread)?;
    d.set_item("rotation_spread", result.rotation_spread)?;
    let evotes = pyo3::types::PyList::empty(py);
    for v in &result.epipolar_votes {
        let e = PyDict::new(py);
        e.set_item("image_a", v.image_a)?;
        e.set_item("image_b", v.image_b)?;
        e.set_item("shared_clusters", v.shared_clusters)?;
        e.set_item("mean_disp_px", v.mean_disp_px)?;
        e.set_item("n_f_inliers", v.n_f_inliers)?;
        e.set_item("n_h_inliers", v.n_h_inliers)?;
        e.set_item("transposed", v.transposed)?;
        e.set_item("focal_px", v.focal_px)?;
        evotes.append(e)?;
    }
    d.set_item("epipolar_votes", evotes)?;
    let rvotes = pyo3::types::PyList::empty(py);
    for v in &result.rotation_votes {
        let e = PyDict::new(py);
        e.set_item("image", v.image)?;
        e.set_item("partner", v.partner)?;
        e.set_item("mean_disp_px", v.mean_disp_px)?;
        e.set_item("n_inliers", v.n_inliers)?;
        e.set_item("focal_px", v.focal_px)?;
        rvotes.append(e)?;
    }
    d.set_item("rotation_votes", rvotes)?;
    d.set_item("n_h_dominated", result.n_h_dominated)?;
    d.set_item("n_estimator_failed", result.n_estimator_failed)?;
    d.set_item("n_band_rejected", result.n_band_rejected)?;
    d.set_item("n_degenerate", result.n_degenerate)?;
    d.set_item("n_inconsistent_pairs", result.n_inconsistent_pairs)?;
    d.set_item("camera_model", result.camera_model.map(|m| m.as_str()))?;
    let cols = pyo3::types::PyList::empty(py);
    for c in &result.columns {
        cols.append(column_dict(py, c)?)?;
    }
    d.set_item("columns", cols)?;
    Ok(d)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(pyo3::wrap_pyfunction!(focal_vote, m)?)?;
    Ok(())
}
