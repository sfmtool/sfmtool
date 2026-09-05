// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Python binding for the intrinsics estimate over the focal vote
//! (``sfmtool._sfmtool.geometry.estimate_intrinsics``; see
//! ``specs/core/geometry/estimate-intrinsics.md``).

use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString};

use sfmtool_core::geometry::estimate_intrinsics::{
    estimate_intrinsics as core_estimate_intrinsics, estimate_intrinsics_from_matches,
    ColumnPolicy, IntrinsicsOptions,
};
use sfmtool_core::geometry::focal_vote::{CameraModel, FocalVoteOptions};

use super::focal_vote::{
    matches_err_to_py, scan_vote_dict, vote_columns, vote_dict, vote_source, VoteSource,
};

/// Estimate a camera from cluster-track observations: the model verdict, its
/// corroboration, the consensus focal, and the votes behind them (see
/// ``specs/core/geometry/estimate-intrinsics.md``).
///
/// This is the high-level face of ``focal_vote``: it runs the same vote and
/// returns one typed answer instead of a diagnostic table. The raw vote comes
/// back nested under ``"vote"``, so nothing is lost.
///
/// Takes its observations in either of two forms, and only these two:
///
/// * ``estimate_intrinsics(matches_file, ...)`` -- a ``MatchesFile`` (a
///   selection included), whose cluster backbone already IS the layout below
///   and whose image table supplies the shared image size. Every image of the
///   file must carry the same dimensions, because the estimate is of ONE
///   shared camera with a centred principal point; a file mixing resolutions
///   raises ``ValueError`` rather than being answered from its first image.
/// * ``estimate_intrinsics(cluster_starts, member_images, member_positions,
///   width, height, ...)`` -- the same observations spelled out.
///
/// Args:
///     cluster_starts: A ``MatchesFile``, or the (n_clusters + 1,) uint32 CSR
///         offsets into the member arrays: opening at 0, nondecreasing, and
///         closing at the member count.
///     member_images: (n_members,) uint32 image id per member. Omitted in the
///         ``MatchesFile`` form.
///     member_positions: (n_members, 2) float32 full-pixel keypoint positions
///         -- the width the ``.matches`` backbone stores them at. A float64
///         array is accepted and cast, which is exact for the
///         ``f32``-originated values a caller reads out of such a file.
///         Omitted in the ``MatchesFile`` form.
///     width: Shared image width; the principal point is the image centre.
///         Omitted in the ``MatchesFile`` form.
///     height: Shared image height. Omitted in the ``MatchesFile`` form.
///     seed: SplitMix64 seed for the RANSAC estimators and the column scans;
///         same inputs + seed => bit-identical output (default 0).
///     epipolar_min_disp_frac: Wide-baseline gate for epipolar candidate
///         pairs, as a fraction of the image diagonal their mean feature
///         displacement must reach (default 0.02).
///     columns: Camera-model columns to evaluate, as a sequence of names
///         (``"pinhole"``, ``"equidistant"`` / ``"fisheye"``). The default
///         ``None`` runs both columns, because arbitrating between them is
///         what this function is for; a single named column is the verdict by
///         construction and arbitrates nothing. The string ``"auto"`` instead
///         lets the estimator choose: it runs the pinhole-only vote and pays
///         for the two-column run only when that vote comes back weak (see
///         ``escalation`` below).
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
///     float | None, "verdict_votes": list[dict], "escalation": list[str] |
///     None, "screening_vote": dict | None, "vote": dict}``.
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
///     ``escalation`` is what ``columns="auto"`` decided: the weak-vote reason
///     names that fired, in check order (``"no_consensus"``,
///     ``"rotation_railed"``, ``"family_disagreement"``, ``"thin_pool"``). An
///     empty list means the pinhole-only vote stood on its own, no scan ran,
///     and ``vote`` is that pinhole-only result -- no columns and a
///     ``"Pinhole"`` verdict. It is ``None`` whenever the columns were named
///     outright, which never escalates.
///
///     ``screening_vote`` is the pinhole-only vote that decision was read off,
///     present only when the estimate then re-ran with both columns. Read the
///     capture's PINHOLE numbers off it: a two-column ``vote`` reports the
///     winning column at the top level, which is the fisheye answer whenever
///     the escalation paid off.
///
///     ``vote`` is the full ``focal_vote`` result dict behind the verdict,
///     untouched.
// This is a Python docstring (rendered by `help()`), not Rust prose: its
// indented `Args:` / `Returns:` continuation paragraphs read as Markdown
// indented code blocks, which rustdoc then tries to parse as Rust.
#[allow(rustdoc::invalid_rust_codeblocks)]
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (cluster_starts, member_images=None, member_positions=None, width=None, height=None, *, seed=0, epipolar_min_disp_frac=0.02, columns=None, min_rotation_mass=1))]
pub fn estimate_intrinsics<'py>(
    py: Python<'py>,
    cluster_starts: Bound<'py, PyAny>,
    member_images: Option<PyReadonlyArray1<'py, u32>>,
    member_positions: Option<Bound<'py, PyAny>>,
    width: Option<u32>,
    height: Option<u32>,
    seed: u64,
    epipolar_min_disp_frac: f64,
    columns: Option<Bound<'py, PyAny>>,
    min_rotation_mass: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let source = vote_source(
        &cluster_starts,
        member_images,
        member_positions,
        width,
        height,
    )?;
    // Unlike `focal_vote`, whose default is the closed-form pinhole kernel,
    // this function's default is both columns: the verdict is the product.
    // `"auto"` hands the choice to the estimator instead, and then the column
    // set it passes here is the one it never reads.
    let (policy, models) = match columns {
        None => (
            ColumnPolicy::Fixed,
            vec![CameraModel::Pinhole, CameraModel::EquidistantFisheye],
        ),
        Some(obj) if obj.is_instance_of::<PyString>() => {
            let name: String = obj.extract()?;
            if !name.eq_ignore_ascii_case("auto") {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "columns must be \"auto\" or a sequence of column names \
                     (\"pinhole\", \"equidistant\"); got {name:?}"
                )));
            }
            (ColumnPolicy::Auto, Vec::new())
        }
        Some(obj) => (ColumnPolicy::Fixed, vote_columns(Some(obj.extract()?))?),
    };
    let options = IntrinsicsOptions {
        vote: FocalVoteOptions {
            seed,
            epipolar_min_disp_frac,
            columns: models,
        },
        columns: policy,
        min_rotation_mass,
    };

    let estimate = py
        .detach(move || match source {
            VoteSource::Matches(matches) => estimate_intrinsics_from_matches(matches, &options),
            VoteSource::Arrays(a) => Ok(core_estimate_intrinsics(
                &a.cluster_starts,
                &a.member_images,
                &a.member_positions,
                a.width,
                a.height,
                &options,
            )),
        })
        .map_err(matches_err_to_py)?;

    let d = PyDict::new(py);
    d.set_item("camera_model", estimate.camera_model.map(|m| m.as_str()))?;
    d.set_item("confirmed", estimate.confirmed)?;
    d.set_item("focal_px", estimate.focal_px)?;
    let votes = pyo3::types::PyList::empty(py);
    for v in &estimate.verdict_votes {
        votes.append(scan_vote_dict(py, v)?)?;
    }
    d.set_item("verdict_votes", votes)?;
    d.set_item(
        "escalation",
        estimate
            .escalation
            .as_ref()
            .map(|rs| rs.iter().map(|r| r.as_str()).collect::<Vec<_>>()),
    )?;
    d.set_item(
        "screening_vote",
        estimate
            .screening_vote
            .as_ref()
            .map(|v| vote_dict(py, v))
            .transpose()?,
    )?;
    d.set_item("vote", vote_dict(py, &estimate.vote)?)?;
    Ok(d)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(pyo3::wrap_pyfunction!(estimate_intrinsics, m)?)?;
    Ok(())
}
