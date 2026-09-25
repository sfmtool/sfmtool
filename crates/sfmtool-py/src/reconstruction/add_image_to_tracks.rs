// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! `EditedReconstruction.add_image_to_tracks`: adding one image's observations
//! to the tracks it can see (see ``specs/core/reconstruction/add-image-to-tracks.md``).
//!
//! The rule and the gates are spelled as strings plus their numbers, so a
//! script can sweep them without building Rust enums; the report's per-point
//! numbers come back as columns, one numpy array per scalar and one list per
//! ragged field, because a call reports every point the image does not observe.

use std::sync::Arc;

use numpy::{PyArray1, PyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use sfmtool_core::progress::Progress;
use sfmtool_core::reconstruction::add_image_to_tracks::{
    add_image_to_tracks as core_add_image_to_tracks, AcceptRule, AddImageToTracksOptions,
    AddImageToTracksReport, BasisStatistic, PairRule, PairStatistic, PositionGate, TemplateSource,
};
use sfmtool_core::reconstruction::edited::EditedReconstruction;

use super::edited::{materialised, PyEditedReconstruction};
use crate::patches::views::{resolve_pyramids, PosedViews};

fn basis_statistic(name: &str, k: f64, fraction: f64) -> PyResult<BasisStatistic> {
    match name {
        "min" => Ok(BasisStatistic::Min),
        "median_minus_mad" => Ok(BasisStatistic::MedianMinusMad { k }),
        "fraction_of_median" => Ok(BasisStatistic::FractionOfMedian { fraction }),
        other => Err(PyValueError::new_err(format!(
            "basis must be \"min\", \"median_minus_mad\" or \"fraction_of_median\", not {other:?}"
        ))),
    }
}

fn pair_statistic(name: &str) -> PyResult<PairStatistic> {
    match name {
        "min" => Ok(PairStatistic::Min),
        "mean" => Ok(PairStatistic::Mean),
        "max" => Ok(PairStatistic::Max),
        other => Err(PyValueError::new_err(format!(
            "pair_statistic must be \"min\", \"mean\" or \"max\", not {other:?}"
        ))),
    }
}

/// Four columns of `[x, y]` pairs as an `(N, 2)` array, `NaN` where absent.
fn xy_column<'py>(
    py: Python<'py>,
    rows: impl Iterator<Item = Option<[f64; 2]>>,
) -> Bound<'py, PyArray2<f64>> {
    let flat: Vec<f64> = rows
        .flat_map(|r| r.unwrap_or([f64::NAN, f64::NAN]))
        .collect();
    let n = flat.len() / 2;
    let array = ndarray::Array2::from_shape_vec((n, 2), flat).expect("two values per row");
    PyArray2::from_owned_array(py, array)
}

fn report_to_py<'py>(py: Python<'py>, r: &AddImageToTracksReport) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("image", r.image)?;
    d.set_item("accepted", r.accepted)?;
    d.set_item("observations_before", r.observations_before)?;
    d.set_item("observations_after", r.observations_after)?;
    d.set_item("pooled_bar", r.pooled_bar)?;
    d.set_item("position_bound_px", r.position_bound_px)?;
    let counts = PyDict::new(py);
    for (refusal, n) in r.refusal_counts() {
        counts.set_item(refusal.name(), n)?;
    }
    d.set_item("refusal_counts", counts)?;

    let c = &r.candidates;
    let cands = PyDict::new(py);
    cands.set_item(
        "point",
        PyArray1::from_vec(py, c.iter().map(|x| x.point).collect()),
    )?;
    cands.set_item(
        "accepted",
        PyArray1::from_vec(py, c.iter().map(|x| x.refusal.is_none()).collect()),
    )?;
    cands.set_item(
        "refusal",
        PyList::new(py, c.iter().map(|x| x.refusal.map(|r| r.name())))?,
    )?;
    cands.set_item("projection", xy_column(py, c.iter().map(|x| x.projection)))?;
    cands.set_item(
        "search_keypoint",
        xy_column(py, c.iter().map(|x| x.search_keypoint)),
    )?;
    cands.set_item("keypoint", xy_column(py, c.iter().map(|x| x.keypoint)))?;
    for (key, value) in [
        (
            "offset_px",
            c.iter().map(|x| x.offset_px).collect::<Vec<_>>(),
        ),
        ("sigma_pos", c.iter().map(|x| x.sigma_pos).collect()),
        ("peak_zncc", c.iter().map(|x| x.peak_zncc).collect()),
        ("zncc", c.iter().map(|x| x.zncc).collect()),
        ("judged", c.iter().map(|x| x.judged).collect()),
        ("bar", c.iter().map(|x| x.bar).collect()),
    ] {
        cands.set_item(key, PyArray1::from_vec(py, value))?;
    }
    cands.set_item(
        "references",
        PyList::new(py, c.iter().map(|x| x.references.clone()))?,
    )?;
    cands.set_item(
        "reference_loo_zncc",
        PyList::new(py, c.iter().map(|x| x.reference_loo_zncc.clone()))?,
    )?;
    cands.set_item(
        "reference_pair_zncc",
        PyList::new(py, c.iter().map(|x| x.reference_pair_zncc.clone()))?,
    )?;
    cands.set_item(
        "pair_zncc",
        PyList::new(py, c.iter().map(|x| x.pair_zncc.clone()))?,
    )?;
    d.set_item("candidates", cands)?;
    Ok(d)
}

#[pymethods]
impl PyEditedReconstruction {
    /// Add observations of `image` to the points of this version it does not
    /// observe, where it sees them and its view agrees with theirs, and give
    /// back the answer as this version's successor.
    ///
    /// For every such point: the point must project into the image, in front
    /// of the camera and inside the frame, not grazing, and (with
    /// ``require_facing``) with the camera on the same side of the patch plane
    /// as most of the cameras that observe it. The point's existing
    /// observations are rendered on the patch grid at their keypoints and
    /// combined into a robust consensus; the image is searched once against it
    /// within ``search`` patch-grid pixels of the projection, refined to
    /// sub-pixel (``subpixel``), and scored. The rule then judges the score
    /// (see ``specs/core/reconstruction/add-image-to-tracks.md``). Nothing but the added
    /// observations changes: no point, frame, bitmap or camera moves, and
    /// nothing is re-triangulated. A **bulk** edit, so the value that comes
    /// back is a whole new base with an empty overlay, every point at the index
    /// it has in this version's materialisation.
    ///
    /// Args:
    ///     image: The image's index.
    ///     images: One decoded image per image of this version, or an
    ///         ``ImagePyramidSet``. The references are measured, so the
    ///         photographs of the observing images are read as well as the
    ///         target's.
    ///     rule: ``"pooled_or_track"`` (default: a candidate passes the image's
    ///         pooled bar or its own track's bar), ``"pooled_basis"`` (one bar
    ///         from every candidate's references), ``"track_basis"`` (the
    ///         point's own references set the bar) or ``"fixed"``
    ///         (``min_zncc`` alone).
    ///     basis: The statistic of the references' leave-one-out ZNCCs for the
    ///         pooled bar, and for ``"track_basis"``: ``"median_minus_mad"``
    ///         (default, with ``basis_k``, default 3), ``"min"`` or
    ///         ``"fraction_of_median"`` (with ``basis_fraction``).
    ///     track_basis, track_basis_k, track_basis_fraction: The track's own
    ///         statistic under ``"pooled_or_track"`` (default
    ///         ``"fraction_of_median"`` at 0.9).
    ///     pair_statistic, pair_factor: For a two-reference track under
    ///         ``"track_basis"``: ``"min"``, ``"mean"`` or ``"max"`` of the
    ///         image's ZNCC against each reference must reach ``pair_factor``
    ///         times the ZNCC between the two references.
    ///     min_zncc: A floor on the ZNCC every rule applies (default 0.5;
    ///         ``0`` disables it for the basis rules); the whole of ``"fixed"``.
    ///     position_gate: ``"image_mad"`` (default: median plus ``position_k``
    ///         scaled MADs of the photometrically accepted offsets, never below
    ///         ``position_floor_px``), ``"max_px"`` (``position_max_px``) or
    ///         ``"off"``.
    ///     template: ``"rendered"`` or ``"stored_bitmap"``.
    ///     require_facing, subpixel, ascend_on_edge,
    ///         min_keypoint_separation_px: see the spec.
    ///     search: The search radius in patch-grid pixels.
    ///     max_keypoint_uncertainty: The member localizability gate's ``τ``
    ///         (``0`` disables it).
    ///     min_grazing_cos: The grazing cutoff.
    ///     resolution: The patch grid (a stored bitmap's own grid overrides it
    ///         under ``"stored_bitmap"``).
    ///
    /// Returns:
    ///     ``(EditedReconstruction, report)``. The report carries ``image``,
    ///     ``accepted``, ``observations_before``, ``observations_after``,
    ///     ``pooled_bar``, ``position_bound_px``, ``refusal_counts`` (name to
    ///     count) and ``candidates``, a dict of columns over every point the
    ///     image did not observe: ``point``, ``accepted``, ``refusal`` (a name
    ///     or ``None``), ``projection``, ``search_keypoint`` and ``keypoint``
    ///     (``(N, 2)``, ``NaN`` where absent), ``offset_px``, ``sigma_pos``,
    ///     ``peak_zncc``, ``zncc``, ``judged``, ``bar``, and the lists
    ///     ``references``, ``reference_loo_zncc``, ``reference_pair_zncc``
    ///     (row-major ``n × n``) and ``pair_zncc``. Raises ``ValueError`` with
    ///     the reason when the call is refused.
    #[allow(rustdoc::invalid_rust_codeblocks)]
    #[pyo3(signature = (
        image,
        images,
        *,
        rule = "pooled_or_track",
        basis = "median_minus_mad",
        basis_k = 3.0,
        basis_fraction = 0.9,
        track_basis = "fraction_of_median",
        track_basis_k = 3.0,
        track_basis_fraction = 0.9,
        pair_statistic = "mean",
        pair_factor = 0.9,
        min_zncc = 0.5,
        position_gate = "image_mad",
        position_max_px = 3.0,
        position_k = 3.0,
        position_floor_px = 1.0,
        template = "rendered",
        require_facing = true,
        subpixel = true,
        ascend_on_edge = false,
        min_keypoint_separation_px = 1.0,
        search = 6.0,
        max_keypoint_uncertainty = 0.35,
        min_grazing_cos = 0.1,
        resolution = 24,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn add_image_to_tracks(
        &self,
        py: Python<'_>,
        image: usize,
        images: &Bound<'_, PyAny>,
        rule: &str,
        basis: &str,
        basis_k: f64,
        basis_fraction: f64,
        track_basis: &str,
        track_basis_k: f64,
        track_basis_fraction: f64,
        pair_statistic: &str,
        pair_factor: f64,
        min_zncc: f64,
        position_gate: &str,
        position_max_px: f64,
        position_k: f64,
        position_floor_px: f64,
        template: &str,
        require_facing: bool,
        subpixel: bool,
        ascend_on_edge: bool,
        min_keypoint_separation_px: f64,
        search: f64,
        max_keypoint_uncertainty: f64,
        min_grazing_cos: f64,
        resolution: u32,
    ) -> PyResult<(PyEditedReconstruction, Py<PyDict>)> {
        let statistic = basis_statistic(basis, basis_k, basis_fraction)?;
        let rule = match rule {
            "fixed" => AcceptRule::FixedZncc,
            "track_basis" => AcceptRule::TrackBasis {
                statistic,
                pair: PairRule {
                    statistic: self::pair_statistic(pair_statistic)?,
                    factor: pair_factor,
                },
            },
            "pooled_basis" => AcceptRule::PooledBasis { statistic },
            "pooled_or_track" => AcceptRule::PooledOrTrack {
                pooled: statistic,
                track: basis_statistic(track_basis, track_basis_k, track_basis_fraction)?,
                pair: PairRule {
                    statistic: self::pair_statistic(pair_statistic)?,
                    factor: pair_factor,
                },
            },
            other => {
                return Err(PyValueError::new_err(format!(
                    "rule must be \"pooled_basis\", \"pooled_or_track\", \"track_basis\" or \"fixed\", not {other:?}"
                )))
            }
        };
        let position_gate = match position_gate {
            "off" => PositionGate::Off,
            "max_px" => PositionGate::MaxPx(position_max_px),
            "image_mad" => PositionGate::ImageMad {
                k: position_k,
                floor_px: position_floor_px,
            },
            other => {
                return Err(PyValueError::new_err(format!(
                    "position_gate must be \"off\", \"max_px\" or \"image_mad\", not {other:?}"
                )))
            }
        };
        let template = match template {
            "rendered" => TemplateSource::Rendered,
            "stored_bitmap" => TemplateSource::StoredBitmap,
            other => {
                return Err(PyValueError::new_err(format!(
                    "template must be \"rendered\" or \"stored_bitmap\", not {other:?}"
                )))
            }
        };
        let defaults = AddImageToTracksOptions::default();
        let options = AddImageToTracksOptions {
            rule,
            min_zncc,
            position_gate,
            template,
            require_facing,
            subpixel,
            ascend_on_edge,
            min_keypoint_separation_px,
            localize: sfmtool_core::patch::keypoint_localize::KeypointLocalizeParams {
                search,
                max_member_keypoint_uncertainty: max_keypoint_uncertainty,
                min_grazing_cos,
                resolution,
                ..defaults.localize.clone()
            },
            ..defaults
        };

        let value = materialised(&self.inner);
        let posed = PosedViews::from_reconstruction(&value);
        let pyramids = resolve_pyramids(&posed, images)?;
        let slots: Vec<_> = pyramids.as_slice().iter().map(Some).collect();
        let (next, report) = py
            .detach(|| core_add_image_to_tracks(&value, image, &slots, &options, &Progress::none()))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let d = report_to_py(py, &report)?;
        Ok((
            PyEditedReconstruction {
                inner: EditedReconstruction::new(Arc::new(next)),
            },
            d.unbind(),
        ))
    }
}
