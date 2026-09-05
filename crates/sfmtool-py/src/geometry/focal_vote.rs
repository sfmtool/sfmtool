// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Python binding for the structure-free focal vote
//! (``sfmtool._sfmtool.geometry.focal_vote``; see ``specs/core/geometry/focal-vote.md``).

use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use matches_format::MatchesData;
use sfmtool_core::geometry::focal_vote::{
    focal_vote_from_matches, focal_vote_with_options, CameraModel, ColumnDiagnostics,
    FocalVoteOptions, FocalVoteResult, MatchesInputError, ScanCell, ScanVote,
};

use crate::io::matches_file::PyMatchesFile;

/// One column's `scan_votes` entry as a Python dict.
pub(crate) fn scan_vote_dict<'py>(py: Python<'py>, v: &ScanVote) -> PyResult<Bound<'py, PyDict>> {
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

/// The CSR observation arrays a vote-shaped binding takes in its array form,
/// validated and copied into the layout the kernel wants.
pub(crate) struct VoteArrays {
    /// `n_clusters + 1` CSR offsets into the member arrays.
    pub cluster_starts: Vec<u32>,
    /// Image id per member.
    pub member_images: Vec<u32>,
    /// Full-pixel keypoint position per member.
    pub member_positions: Vec<[f64; 2]>,
    /// Shared image width.
    pub width: u32,
    /// Shared image height.
    pub height: u32,
}

/// What a vote-shaped binding was called with: a parsed `.matches` handle, or
/// the CSR arrays spelled out.
///
/// Both forms reach the same kernel; the object form hands the whole file to
/// the core's own `from_matches` entry rather than taking the file apart here,
/// so the reading (the `f32 → f64` widening, the shared-dimensions rule) has
/// exactly one implementation and both languages get it.
pub(crate) enum VoteSource<'a> {
    /// A `.matches` file, read by the core entry point.
    Matches(&'a MatchesData),
    /// Explicit CSR observations.
    Arrays(VoteArrays),
}

/// Resolve a vote-shaped binding's positional arguments into one of the two
/// forms.
///
/// `source` is either a `MatchesFile` — and then nothing else may be given —
/// or the `cluster_starts` array, and then all four remaining arguments are
/// required.
pub(crate) fn vote_source<'a, 'py>(
    source: &'a Bound<'py, PyAny>,
    member_images: Option<PyReadonlyArray1<'py, u32>>,
    member_positions: Option<PyReadonlyArray2<'py, f64>>,
    width: Option<u32>,
    height: Option<u32>,
) -> PyResult<VoteSource<'a>> {
    if let Ok(file) = source.cast::<PyMatchesFile>() {
        if member_images.is_some()
            || member_positions.is_some()
            || width.is_some()
            || height.is_some()
        {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "the MatchesFile form takes no observation arrays: the file states \
                 its own members and image size",
            ));
        }
        return Ok(VoteSource::Matches(file.get().data()));
    }
    let cluster_starts: PyReadonlyArray1<'py, u32> = source.extract().map_err(|_| {
        pyo3::exceptions::PyTypeError::new_err(
            "the first argument must be a MatchesFile or a (n_clusters + 1,) uint32 \
             cluster_starts array",
        )
    })?;
    let (Some(member_images), Some(member_positions), Some(width), Some(height)) =
        (member_images, member_positions, width, height)
    else {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "the array form takes cluster_starts, member_images, member_positions, \
             width and height",
        ));
    };
    Ok(VoteSource::Arrays(vote_arrays(
        cluster_starts,
        member_images,
        member_positions,
        width,
        height,
    )?))
}

/// Validate and unpack the CSR observation arrays.
///
/// The index contract is checked here, in `O(n_clusters)`, so a caller learns
/// what is wrong with its arrays instead of getting the empty vote back.
fn vote_arrays(
    cluster_starts: PyReadonlyArray1<'_, u32>,
    member_images: PyReadonlyArray1<'_, u32>,
    member_positions: PyReadonlyArray2<'_, f64>,
    width: u32,
    height: u32,
) -> PyResult<VoteArrays> {
    if member_positions.shape()[1] != 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "member_positions must have shape (n_members, 2)",
        ));
    }
    let n_members = member_images.shape()[0];
    if member_positions.shape()[0] != n_members {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "member_images and member_positions must share n_members",
        ));
    }

    let starts = to_contiguous!(cluster_starts);
    let images = to_contiguous!(member_images);
    let pos_flat = to_contiguous!(member_positions);
    let positions: Vec<[f64; 2]> = pos_flat
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| [c[0], c[1]])
        .collect();

    if starts.first() != Some(&0) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "cluster_starts must have at least one entry and open at 0",
        ));
    }
    if starts.windows(2).any(|w| w[1] < w[0]) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "cluster_starts must be nondecreasing",
        ));
    }
    if starts.last().copied().unwrap_or(0) as usize != n_members {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "cluster_starts must close at the member count ({n_members}), not {}",
            starts.last().copied().unwrap_or(0)
        )));
    }
    Ok(VoteArrays {
        cluster_starts: starts.into_owned(),
        member_images: images.into_owned(),
        member_positions: positions,
        width,
        height,
    })
}

/// Map a core matches-reading refusal onto the Python exception the caller
/// sees. Every one of them is a property of the file, so they are all value
/// errors.
pub(crate) fn matches_err_to_py(e: MatchesInputError) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(e.to_string())
}

/// Resolve the `columns` argument; `None` is the pinhole-only default.
pub(crate) fn vote_columns(columns: Option<Vec<String>>) -> PyResult<Vec<CameraModel>> {
    match columns {
        None => Ok(vec![CameraModel::Pinhole]),
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
            .collect(),
    }
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
/// reconstruction (see ``specs/core/geometry/focal-vote.md``).
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
/// Takes its observations in either of two forms, and only these two:
///
/// * ``focal_vote(matches_file, ...)`` -- a ``MatchesFile`` (a selection
///   included), whose cluster backbone already IS the layout below and whose
///   image table supplies the shared image size. Every image of the file must
///   carry the same dimensions, because the vote estimates one shared camera.
/// * ``focal_vote(cluster_starts, member_images, member_positions, width,
///   height, ...)`` -- the same observations spelled out.
///
/// Args:
///     cluster_starts: A ``MatchesFile``, or the (n_clusters + 1,) uint32 CSR
///         offsets into the member arrays: opening at 0, nondecreasing, and
///         closing at the member count.
///     member_images: (n_members,) uint32 image id per member. Omitted in the
///         ``MatchesFile`` form.
///     member_positions: (n_members, 2) float64 full-pixel keypoint
///         positions. Omitted in the ``MatchesFile`` form.
///     width: Shared image width; the principal point is the image centre.
///         Omitted in the ``MatchesFile`` form.
///     height: Shared image height. Omitted in the ``MatchesFile`` form.
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
#[pyo3(signature = (cluster_starts, member_images=None, member_positions=None, width=None, height=None, *, seed=0, epipolar_min_disp_frac=0.02, columns=None))]
pub fn focal_vote<'py>(
    py: Python<'py>,
    cluster_starts: Bound<'py, PyAny>,
    member_images: Option<PyReadonlyArray1<'py, u32>>,
    member_positions: Option<PyReadonlyArray2<'py, f64>>,
    width: Option<u32>,
    height: Option<u32>,
    seed: u64,
    epipolar_min_disp_frac: f64,
    columns: Option<Vec<String>>,
) -> PyResult<Bound<'py, PyDict>> {
    let source = vote_source(
        &cluster_starts,
        member_images,
        member_positions,
        width,
        height,
    )?;
    let options = FocalVoteOptions {
        seed,
        epipolar_min_disp_frac,
        columns: vote_columns(columns)?,
    };

    let result = py.detach(move || match source {
        VoteSource::Matches(matches) => focal_vote_from_matches(matches, &options),
        VoteSource::Arrays(a) => Ok(focal_vote_with_options(
            &a.cluster_starts,
            &a.member_images,
            &a.member_positions,
            a.width,
            a.height,
            &options,
        )),
    });

    vote_dict(py, &result.map_err(matches_err_to_py)?)
}

/// A [`FocalVoteResult`] as the Python dict the binding documents.
pub(crate) fn vote_dict<'py>(
    py: Python<'py>,
    result: &FocalVoteResult,
) -> PyResult<Bound<'py, PyDict>> {
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
