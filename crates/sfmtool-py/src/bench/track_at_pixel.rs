// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for building a track at a pixel.
//!
//! [`PyTrackAtPixelSources`] holds what a query reads beside the reconstruction
//! and the photographs, built once per capture: the SIFT index, every image's
//! keypoints and the `.matches` clusters indexed onto the reconstruction's
//! images. [`build_track_at_pixel`] runs one query and gives back the track and
//! a report dict, or raises [`TrackAtPixelError`] carrying the stage, the
//! sentence and what was measured.

use numpy::{PyReadonlyArray2, PyReadonlyArray3};
use pyo3::create_exception;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use sfmtool_core::bench::{
    build_track_at_pixel as core_build_track_at_pixel, CandidateKind, CandidateRecord,
    CascadeMember, ClassificationReason, MatchesClusters, MemberRefusal, SiftIndexSource,
    StageRecord, TiltRecord, TrackAtPixelError as CoreError, TrackAtPixelOptions,
    TrackAtPixelSources,
};
use sfmtool_core::features::kdforest::ImageKeypoints;
use sfmtool_core::progress::Progress;

use super::{read_keypoints, views_of, PyEditableTrack};
use crate::io::matches_file::PyMatchesFile;
use crate::patches::views::{resolve_pyramids, PosedViews};
use crate::reconstruction::edited::PyEditedReconstruction;
use crate::spatial::kdf::PyLazyKdForest;

create_exception!(
    sfmtool.bench,
    TrackAtPixelError,
    PyValueError,
    "No track could be built at the pixel. ``stage`` names the step that said \
     so, ``reason`` is the sentence a person is shown, and ``diagnostics`` is \
     what was measured on the way."
);

/// What a track-at-pixel query reads beside the reconstruction and the
/// photographs, built once per capture and shared by every query.
///
/// Args:
///     edited: The reconstruction the clusters are indexed onto; only its
///         image names are read, so any version of one base will do.
///     forest: The SIFT index, whose corpus indexes the reconstruction's
///         images in the reconstruction's order.
///     keypoints: One ``(positions, affine_shapes)`` pair per image of the
///         reconstruction, in its order: ``(N, 2)`` and ``(N, 2, 2)`` float32
///         arrays, as :meth:`SiftReader.read_positions_and_shapes` returns them.
///     matches: A cluster-patches :class:`MatchesFile`. Its images are matched
///         to the reconstruction's by name.
#[pyclass(name = "TrackAtPixelSources", module = "sfmtool.bench", frozen)]
pub struct PyTrackAtPixelSources {
    forest: Py<PyLazyKdForest>,
    keypoints: Vec<ImageKeypoints>,
    clusters: MatchesClusters,
}

#[pymethods]
impl PyTrackAtPixelSources {
    #[new]
    fn new(
        edited: &PyEditedReconstruction,
        forest: Py<PyLazyKdForest>,
        keypoints: &Bound<'_, PyList>,
        matches: &PyMatchesFile,
    ) -> PyResult<Self> {
        let image_count = edited.inner.image_count();
        if keypoints.len() != image_count {
            return Err(PyValueError::new_err(format!(
                "keypoints has {} entries, but the reconstruction has {image_count} images",
                keypoints.len()
            )));
        }
        let keypoints = keypoints
            .iter()
            .map(|item| {
                let pair = item.cast::<PyTuple>()?;
                let positions: PyReadonlyArray2<'_, f32> = pair.get_item(0)?.extract()?;
                let shapes: PyReadonlyArray3<'_, f32> = pair.get_item(1)?.extract()?;
                read_keypoints(&positions, &shapes)
            })
            .collect::<PyResult<Vec<_>>>()?;
        let names: Vec<&str> = edited
            .inner
            .base
            .image_table
            .images
            .iter()
            .map(|im| im.name.as_str())
            .collect();
        let clusters = MatchesClusters::new(matches.data(), &names)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self {
            forest,
            keypoints,
            clusters,
        })
    }

    /// How many clusters the ``.matches`` file holds.
    #[getter]
    fn cluster_count(&self) -> usize {
        self.clusters.cluster_count()
    }

    fn __repr__(&self) -> String {
        format!(
            "TrackAtPixelSources({} images, {} clusters)",
            self.keypoints.len(),
            self.clusters.cluster_count()
        )
    }
}

/// Build a track-stage track centred on ``pixel`` in ``image``, or say why none
/// could be built.
///
/// The cascade tries its members in order -- ``clusters`` (the ``.matches``
/// clusters near the pixel), ``transfer`` (the neighbouring points' own
/// matched keypoints), ``sweep`` (a plane through the neighbours) and
/// ``constellation`` (the SIFT index's constellation query) -- and returns the
/// first track that passes its member's gates. Every member ends in the same
/// finish: anchored fits that keep the patch on the pixel, a tilt toward the
/// neighbours' normal, the geometry search, cleaning and the gates.
///
/// Args:
///     edited: The reconstruction, whose deleted points no query sees.
///     images: One decoded image per image of ``edited``, or a prebuilt
///         :class:`ImagePyramidSet`, as :func:`evaluate` takes them.
///     sources: A :class:`TrackAtPixelSources` built for this capture.
///     image: The queried image's index.
///     pixel: ``(x, y)`` in that image.
///     members: The members to try, in order; the default is all four in the
///         order above.
///
/// Returns:
///     ``(EditableTrack, report)``. The track is evaluated and its observation
///     0 is the one in ``image``. The report carries ``member`` (which member
///     built it), ``query_observation``, ``refusals`` (each earlier member's
///     ``member``, ``stage``, ``reason`` and ``diagnostics``) and one key per
///     step the member ran, with what it measured.
///
/// Raises:
///     TrackAtPixelError: a ``ValueError`` with ``stage``, ``reason`` and
///         ``diagnostics``. ``stage`` is ``"cascade"`` when every member
///         refused, and ``diagnostics["refusals"]`` then holds each one.
#[pyfunction]
#[pyo3(signature = (edited, images, sources, image, pixel, *, members = None))]
pub(super) fn build_track_at_pixel(
    py: Python<'_>,
    edited: &PyEditedReconstruction,
    images: &Bound<'_, PyAny>,
    sources: &PyTrackAtPixelSources,
    image: u32,
    pixel: [f64; 2],
    members: Option<Vec<String>>,
) -> PyResult<(PyEditableTrack, Py<PyDict>)> {
    let mut options = TrackAtPixelOptions::default();
    if let Some(members) = members {
        options.members = members
            .iter()
            .map(|m| m.parse::<CascadeMember>().map_err(PyValueError::new_err))
            .collect::<PyResult<_>>()?;
    }
    let posed = PosedViews::from_reconstruction(&edited.inner.base);
    let pyramids = resolve_pyramids(&posed, images)?;
    let views = views_of(&posed, &pyramids);
    let forest = sources.forest.bind(py).borrow();
    let core_sources = TrackAtPixelSources {
        sift_index: Some(SiftIndexSource {
            forest: forest.inner(),
            keypoints: &sources.keypoints,
        }),
        clusters: Some(&sources.clusters),
    };
    let result = py.detach(|| {
        core_build_track_at_pixel(
            &edited.inner,
            &views,
            &core_sources,
            image,
            pixel,
            &options,
            &Progress::none(),
        )
    });
    match result {
        Ok((track, report)) => {
            let d = PyDict::new(py);
            d.set_item("member", report.member.name())?;
            d.set_item("query_observation", report.query_observation)?;
            d.set_item("refusals", refusals_list(py, &report.refusals)?)?;
            stages_into(py, &d, &report.stages)?;
            Ok((
                PyEditableTrack {
                    inner: std::sync::Arc::new(track),
                },
                d.unbind(),
            ))
        }
        Err(e) => Err(to_py_error(py, &e)),
    }
}

/// The Python exception for a core refusal, with its attributes set.
fn to_py_error(py: Python<'_>, e: &CoreError) -> PyErr {
    let build = || -> PyResult<PyErr> {
        let reason = e.to_string();
        let err = TrackAtPixelError::new_err(format!("{}: {reason}", e.stage()));
        let value = err.value(py);
        let diagnostics = PyDict::new(py);
        if let CoreError::Refused { refusals } = e {
            diagnostics.set_item("refusals", refusals_list(py, refusals)?)?;
        }
        value.setattr("stage", e.stage())?;
        value.setattr("reason", reason)?;
        value.setattr("diagnostics", diagnostics)?;
        Ok(err)
    };
    build().unwrap_or_else(|err| err)
}

/// Each refusal as a dict of its member, stage, reason and diagnostics.
fn refusals_list<'py>(py: Python<'py>, refusals: &[MemberRefusal]) -> PyResult<Bound<'py, PyList>> {
    let out = PyList::empty(py);
    for r in refusals {
        let d = PyDict::new(py);
        d.set_item("member", r.member.name())?;
        d.set_item("stage", r.stage.name())?;
        d.set_item("reason", &r.reason)?;
        let diagnostics = PyDict::new(py);
        stages_into(py, &diagnostics, &r.stages)?;
        d.set_item("diagnostics", diagnostics)?;
        out.append(d)?;
    }
    Ok(out)
}

/// A tilt's record as a dict, or ``{"error": ...}``.
fn tilt_dict<'py>(
    py: Python<'py>,
    record: &Result<TiltRecord, String>,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    match record {
        Ok(t) => {
            if let Some(degrees) = t.degrees {
                d.set_item("degrees", degrees)?;
                d.set_item("stopped", t.stopped)?;
            }
            d.set_item("before", t.zncc_before)?;
            d.set_item("after", t.zncc_after)?;
            d.set_item("kept", t.kept)?;
        }
        Err(e) => d.set_item("error", e)?,
    }
    Ok(d)
}

/// A candidate's record as a dict.
fn candidate_dict<'py>(py: Python<'py>, c: &CandidateRecord) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    match &c.candidate {
        CandidateKind::Cluster {
            cluster,
            distance_px,
        } => {
            d.set_item("cluster", cluster)?;
            d.set_item("distance_px", distance_px)?;
        }
        CandidateKind::DepthMode { support, images } => {
            d.set_item("support", support)?;
            d.set_item("images", images)?;
        }
        CandidateKind::Hypothesis { views } => d.set_item("views", views)?,
    }
    d.set_item("sightings", c.sightings)?;
    if let Some(score) = c.score {
        d.set_item("score", score)?;
    }
    if let Some(error) = &c.error {
        d.set_item("error", error)?;
    }
    Ok(d)
}

/// The snake-case name of a classification reason.
fn reason_name(reason: ClassificationReason) -> &'static str {
    match reason {
        ClassificationReason::WellConditioned => "well_conditioned",
        ClassificationReason::DepthResolved => "depth_resolved",
        ClassificationReason::DepthUnresolved => "depth_unresolved",
        ClassificationReason::BaselineTooShort => "baseline_too_short",
        ClassificationReason::FiniteDoesNotExplainTheSightings => {
            "finite_does_not_explain_the_sightings"
        }
        ClassificationReason::BearingDoesNotExplainTheSightings => {
            "bearing_does_not_explain_the_sightings"
        }
    }
}

/// Write each stage record into `d` under the key the Python candidates use
/// for the same step.
fn stages_into(py: Python<'_>, d: &Bound<'_, PyDict>, stages: &[StageRecord]) -> PyResult<()> {
    for stage in stages {
        match stage {
            StageRecord::NearbyClusters { count } => d.set_item("clusters_near", count)?,
            StageRecord::DepthModes { sizes } => d.set_item("modes", sizes)?,
            StageRecord::Hypotheses { hypotheses } => {
                let list = PyList::empty(py);
                for h in hypotheses {
                    let e = PyDict::new(py);
                    e.set_item("support", h.support)?;
                    e.set_item("half_px", h.half_px)?;
                    e.set_item("nearest_px", h.nearest_px)?;
                    list.append(e)?;
                }
                d.set_item("hypotheses", list)?;
            }
            StageRecord::Candidates { tried } => {
                let list = PyList::empty(py);
                for c in tried {
                    list.append(candidate_dict(py, c)?)?;
                }
                d.set_item("tried", list)?;
            }
            StageRecord::LocalPrior { prior, radius_px } => {
                let e = PyDict::new(py);
                e.set_item("neighbours", prior.neighbours)?;
                let modes = PyList::empty(py);
                for (median, count) in &prior.depth_modes {
                    let m = PyDict::new(py);
                    m.set_item("median", median)?;
                    m.set_item("count", count)?;
                    modes.append(m)?;
                }
                e.set_item("depth_modes", modes)?;
                e.set_item("edge", prior.edge)?;
                if let Some(depth) = prior.depth {
                    e.set_item("depth", depth)?;
                }
                if let Some(half_px) = prior.half_px {
                    e.set_item("half_px", half_px)?;
                }
                if let Some(spread) = prior.normal_spread_deg {
                    e.set_item("normal_spread_deg", spread)?;
                }
                d.set_item("prior", e)?;
                d.set_item("radius_px", radius_px)?;
            }
            StageRecord::Constellation {
                search_radius_px,
                keypoints,
                images,
            } => {
                d.set_item("search_radius_px", search_radius_px)?;
                let e = PyDict::new(py);
                e.set_item("keypoints", keypoints)?;
                e.set_item("images", images)?;
                d.set_item("constellation", e)?;
            }
            StageRecord::Lateral { searches } => {
                let list = PyList::empty(py);
                for s in searches {
                    let e = PyDict::new(py);
                    e.set_item("from", s.from)?;
                    match &s.added {
                        Ok(added) => e.set_item("added", added)?,
                        Err(error) => e.set_item("error", error)?,
                    }
                    list.append(e)?;
                }
                d.set_item("lateral", list)?;
            }
            StageRecord::ClusterEvaluate {
                observations,
                in_views,
            } => {
                let e = PyDict::new(py);
                e.set_item("observations", observations)?;
                e.set_item("in", in_views)?;
                d.set_item("cluster", e)?;
            }
            StageRecord::Upgrade {
                at_infinity,
                reason,
                zncc_median,
            } => {
                let e = PyDict::new(py);
                e.set_item("at_infinity", at_infinity)?;
                e.set_item("reason", reason.map(reason_name))?;
                e.set_item("zncc_median", zncc_median)?;
                d.set_item("upgrade", e)?;
            }
            StageRecord::PriorTilt(record) => d.set_item("prior_tilt", tilt_dict(py, record)?)?,
            StageRecord::Anchor {
                in_views,
                zncc_median,
            } => {
                let e = PyDict::new(py);
                e.set_item("in", in_views)?;
                e.set_item("zncc", zncc_median)?;
                d.set_item("anchor", e)?;
            }
            StageRecord::NormalPrior(record) => {
                d.set_item("normal_prior", tilt_dict(py, record)?)?
            }
            StageRecord::GeometrySearch(record) => {
                let e = PyDict::new(py);
                match record {
                    Ok((added, self_agreement)) => {
                        e.set_item("added", added)?;
                        e.set_item("self_agreement", self_agreement)?;
                    }
                    Err(error) => e.set_item("error", error)?,
                }
                d.set_item("geometry_search", e)?;
            }
            StageRecord::Clean { removed_images } => {
                let e = PyDict::new(py);
                e.set_item("removed_images", removed_images)?;
                d.set_item("clean", e)?;
            }
            StageRecord::Final {
                in_views,
                zncc_median,
                query_offset_px,
                max_projection_offset_px,
            } => {
                let e = PyDict::new(py);
                e.set_item("in", in_views)?;
                e.set_item("zncc_median", zncc_median)?;
                e.set_item("query_offset_px", query_offset_px)?;
                if let Some(worst) = max_projection_offset_px {
                    e.set_item("max_projection_offset_px", worst)?;
                }
                d.set_item("final", e)?;
            }
        }
    }
    Ok(())
}

/// Register the track-at-pixel bindings on the `sfmtool.bench` submodule.
pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTrackAtPixelSources>()?;
    m.add_function(wrap_pyfunction!(build_track_at_pixel, m)?)?;
    m.add("TrackAtPixelError", m.py().get_type::<TrackAtPixelError>())?;
    Ok(())
}
