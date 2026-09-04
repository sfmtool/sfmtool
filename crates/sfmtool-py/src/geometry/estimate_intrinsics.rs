// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Python binding for the intrinsics estimate over the focal vote
//! (``sfmtool._sfmtool.geometry.estimate_intrinsics``; see
//! ``specs/core/geometry/estimate-intrinsics.md``).

use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use sfmtool_core::geometry::estimate_intrinsics::{
    estimate_intrinsics as core_estimate_intrinsics, IntrinsicsOptions,
};
use sfmtool_core::geometry::focal_vote::{CameraModel, FocalVoteOptions};

use super::focal_vote::{scan_vote_dict, vote_columns, vote_dict, vote_inputs};

/// Estimate a camera from cluster-track observations: the model verdict, its
/// corroboration, the consensus focal, and the votes behind them (see
/// ``specs/core/geometry/estimate-intrinsics.md``).
///
/// This is the high-level face of ``focal_vote``: it runs the same vote and
/// returns one typed answer instead of a diagnostic table. The raw vote comes
/// back nested under ``"vote"``, so nothing is lost.
///
/// Args:
///     cluster_indexes: (n_obs,) uint32 cluster id per observation,
///         nondecreasing (each distinct cluster is a contiguous run).
///     image_indexes: (n_obs,) uint32 image id per observation.
///     positions_xy: (n_obs, 2) float64 full-pixel keypoint positions.
///     width: Shared image width; the principal point is the image centre.
///     height: Shared image height.
///     seed: SplitMix64 seed for the RANSAC estimators and the column scans;
///         same inputs + seed => bit-identical output (default 0).
///     epipolar_min_disp_frac: Wide-baseline gate for epipolar candidate
///         pairs, as a fraction of the image diagonal their mean feature
///         displacement must reach (default 0.02).
///     columns: Camera-model columns to evaluate, as a sequence of names
///         (``"pinhole"``, ``"equidistant"`` / ``"fisheye"``). The default
///         ``None`` runs both columns, because arbitrating between them is
///         what this function is for; a single named column is the verdict by
///         construction and arbitrates nothing.
///     min_rotation_mass: Certified rotation-cell votes an equidistant
///         verdict needs in the equidistant column before ``confirmed`` is
///         True (default 1). The default is structural rather than tuned: a
///         wrong ray map cannot fake a pure rotation of rays, so any certified
///         rotation mass at all separates a real fisheye from an arbitration
///         artifact. Raise it when running a reduced cell set, whose geometry
///         the default was not measured on.
///
/// Returns:
///     ``{"camera_model": str | None, "confirmed": bool | None, "focal_px":
///     float | None, "verdict_votes": list[dict], "vote": dict}``.
///
///     ``camera_model`` is the model verdict and ``focal_px`` the winning
///     column's consensus focal. ``confirmed`` answers whether an
///     ``"EquidistantFisheye"`` verdict is corroborated by certified rotation
///     mass, and is ``None`` when the question does not arise: a ``"Pinhole"``
///     verdict, no verdict at all, or a single-column run. An unconfirmed
///     verdict is still returned as the verdict -- refusing to act on it is
///     the caller's policy.
///
///     ``verdict_votes`` is the winning column's certified scan votes, in that
///     column's stored order (epipolar cell first, then rotation), with the
///     same per-vote keys as a column's ``scan_votes``. The nested ``vote``
///     dict's own ``epipolar_votes`` / ``rotation_votes`` always describe the
///     pinhole closed-form kernel regardless of the verdict, so those are not
///     the evidence behind a fisheye answer -- these are.
///
///     ``vote`` is the full ``focal_vote`` result dict, untouched.
// This is a Python docstring (rendered by `help()`), not Rust prose: its
// indented `Args:` / `Returns:` continuation paragraphs read as Markdown
// indented code blocks, which rustdoc then tries to parse as Rust.
#[allow(rustdoc::invalid_rust_codeblocks)]
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (cluster_indexes, image_indexes, positions_xy, width, height, *, seed=0, epipolar_min_disp_frac=0.02, columns=None, min_rotation_mass=1))]
pub fn estimate_intrinsics<'py>(
    py: Python<'py>,
    cluster_indexes: PyReadonlyArray1<'py, u32>,
    image_indexes: PyReadonlyArray1<'py, u32>,
    positions_xy: PyReadonlyArray2<'py, f64>,
    width: u32,
    height: u32,
    seed: u64,
    epipolar_min_disp_frac: f64,
    columns: Option<Vec<String>>,
    min_rotation_mass: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let inputs = vote_inputs(cluster_indexes, image_indexes, positions_xy)?;
    // Unlike `focal_vote`, whose default is the closed-form pinhole kernel,
    // this function's default is both columns: the verdict is the product.
    let models = match columns {
        None => vec![CameraModel::Pinhole, CameraModel::EquidistantFisheye],
        some => vote_columns(some)?,
    };
    let options = IntrinsicsOptions {
        vote: FocalVoteOptions {
            seed,
            epipolar_min_disp_frac,
            columns: models,
        },
        min_rotation_mass,
    };

    let estimate = py.detach(move || {
        core_estimate_intrinsics(
            &inputs.clusters,
            &inputs.images,
            &inputs.positions,
            width,
            height,
            &options,
        )
    });

    let d = PyDict::new(py);
    d.set_item("camera_model", estimate.camera_model.map(|m| m.as_str()))?;
    d.set_item("confirmed", estimate.confirmed)?;
    d.set_item("focal_px", estimate.focal_px)?;
    let votes = pyo3::types::PyList::empty(py);
    for v in &estimate.verdict_votes {
        votes.append(scan_vote_dict(py, v)?)?;
    }
    d.set_item("verdict_votes", votes)?;
    d.set_item("vote", vote_dict(py, &estimate.vote)?)?;
    Ok(d)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(pyo3::wrap_pyfunction!(estimate_intrinsics, m)?)?;
    Ok(())
}
