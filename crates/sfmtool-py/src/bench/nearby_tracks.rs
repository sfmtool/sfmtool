// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Python binding for finding the tracks near a pixel.
//!
//! [`find_nearby_tracks`] runs one query and gives back a dict: the tracks the
//! bench takes, in label order, with the new points they became when asked to
//! commit; and, for the track-at-pixel harness, every track found and every
//! layer as the dicts its anchor finder returned, and its per-source stages.

use std::sync::Arc;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use sfmtool_core::bench::{
    commit as core_commit, find_nearby_tracks as core_find_nearby_tracks, FarFieldTrigger,
    FarFieldWhen, LayerRankBy, NearbyFinding, NearbySource, NearbyTrackOptions, NearbyTrackSources,
    NearbyTracks, SiftIndexSource, SourceReport, StopRule,
};
use sfmtool_core::progress::Progress;

use super::far_field::{reading_dict, set_option as set_far_field_option};
use super::layers::{layer_dict, source_named};
use super::nearby::{
    candidate_dict, harness_name, set_clusters_option, set_constellation_option, set_guided_option,
    set_points_option, PyNearbyTrackSources,
};
use super::{refused, views_of, PyEditableTrack};
use crate::patches::views::{resolve_grey, resolve_pyramids, PosedViews};
use crate::reconstruction::edited::PyEditedReconstruction;

/// The matching source `word` names: its name (`points`, `clusters`,
/// `guided`, `constellation`) or the harness's (`tracks` for the points).
fn matching_source(word: &str) -> PyResult<NearbySource> {
    match word {
        "points" => Ok(NearbySource::Points),
        "far_field" | "farfield" => Err(PyValueError::new_err(
            "the far-field sweep is not a matching source; far_field_when runs it",
        )),
        other => source_named(other),
    }
}

/// Set one option of `options` from a Python value: a top-level field, or a
/// `"<section>.<field>"` one.
fn set_option(
    options: &mut NearbyTrackOptions,
    key: &str,
    value: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let known = match key.split_once('.') {
        None => {
            match key {
                "sources" => {
                    let words: Vec<String> = match value.extract::<String>() {
                        Ok(joined) => joined.split('+').map(str::to_string).collect(),
                        Err(_) => value.extract()?,
                    };
                    options.sources = words
                        .iter()
                        .filter(|w| !w.is_empty())
                        .map(|w| matching_source(w))
                        .collect::<PyResult<_>>()?;
                }
                "stop" => {
                    let word: String = value.extract()?;
                    options.stop = word.parse::<StopRule>().map_err(PyValueError::new_err)?;
                }
                "enough_count" => options.enough_count = value.extract()?,
                "enough_px" => options.enough_px = value.extract()?,
                "far_field_when" => {
                    let word: String = value.extract()?;
                    options.far_field_when = word
                        .parse::<FarFieldWhen>()
                        .map_err(PyValueError::new_err)?;
                }
                _ => return Err(PyValueError::new_err(format!("unknown option {key:?}"))),
            }
            true
        }
        Some(("points", field)) => set_points_option(&mut options.points, field, value)?,
        Some(("clusters", field)) => set_clusters_option(&mut options.clusters, field, value)?,
        Some(("guided", field)) => set_guided_option(&mut options.guided, field, value)?,
        Some(("constellation", field)) => {
            set_constellation_option(&mut options.constellation, field, value)?
        }
        Some(("range", field)) => {
            let r = &mut options.range;
            match field {
                "tolerance_px" => r.tolerance_px = value.extract()?,
                "max_span" => r.max_span = value.extract()?,
                "far_spread" => r.far_spread = value.extract()?,
                _ => return Err(PyValueError::new_err(format!("unknown option {key:?}"))),
            }
            true
        }
        Some(("far_field", field)) => {
            set_far_field_option(&mut options.far_field, field, value)
                .map_err(|_| PyValueError::new_err(format!("unknown option {key:?}")))?;
            true
        }
        Some(("layers", field)) => {
            let l = &mut options.layers;
            match field {
                "evidence" => l.evidence = value.extract()?,
                "rank_by" => {
                    let word: String = value.extract()?;
                    l.rank_by = word.parse::<LayerRankBy>().map_err(PyValueError::new_err)?;
                }
                "radius_px" => l.radius_px = value.extract()?,
                "samples" => l.samples = value.extract()?,
                _ => return Err(PyValueError::new_err(format!("unknown option {key:?}"))),
            }
            true
        }
        Some(("tracks", field)) => {
            let t = &mut options.tracks;
            match field {
                "build" => t.build = value.extract()?,
                "radius_px" => t.radius_px = value.extract()?,
                _ => return Err(PyValueError::new_err(format!("unknown option {key:?}"))),
            }
            true
        }
        Some(_) => false,
    };
    if known {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!("unknown option {key:?}")))
    }
}

/// Find the tracks near ``pixel`` in ``image``: 3D points that several of the
/// photographs agree on, grouped into depth layers ranked by how well the
/// pixel's own patch reads at each, each ready for the bench.
///
/// The matching sources run in order -- the reconstruction's own points, the
/// cluster-patches clusters, guided matching and the constellation query, each
/// skipped when ``sources`` lacks its input -- until two usable tracks lie
/// within 20 px of the pixel. The far-field sweep then runs when they leave
/// the pixel's distance open: no usable track, tracks on more than one layer,
/// or none within a pixel of the pixel. The usable tracks are grouped into
/// depth layers and ranked, and each is labelled
/// ``<stem>@<x>,<y> <rank><letter>``, the letter its order in its layer by
/// distance from the pixel, with `` pt <index>`` for an existing point.
///
/// Args:
///     edited: The reconstruction, whose deleted points no query sees.
///     images: One decoded image per image of ``edited``, or a prebuilt
///         :class:`ImagePyramidSet`, as :func:`far_field_sweep` takes them. A
///         set also keeps the grey images between calls.
///     sources: A :class:`NearbyTrackSources` built for this capture.
///     image: The queried image's index.
///     pixel: ``(x, y)`` in that image.
///     options: Overrides of the query's parameters, keyed by the field of the
///         Rust ``NearbyTrackOptions``: ``sources`` (a list of ``points``,
///         ``clusters``, ``guided``, ``constellation``, or those joined by
///         ``+``; ``tracks`` is the harness's name for ``points``), ``stop``
///         (``"enough"`` or ``"never"``), ``enough_count`` (2), ``enough_px``
///         (20), ``far_field_when`` (``"needed"``, ``"always"``, ``"never"``);
///         or ``"<section>.<field>"`` with the section one of ``points``,
///         ``clusters``, ``guided``, ``constellation``, ``range``,
///         ``far_field``, ``layers`` or ``tracks`` and the field as the
///         section's own binding takes it, e.g. ``{"tracks.build": False}``.
///         An unknown key is an error.
///     label: The group label every track's label starts with, in place of
///         ``<stem>@<x>,<y>``.
///     commit: Commit every track that is not an existing point and was
///         built, in label order, and return the version with them.
///
/// Returns:
///     A dict, or with ``commit`` ``(EditedReconstruction, dict)``. The dict
///     carries ``group_label``; ``tracks``, the usable tracks that are not
///     duplicates, in label order,
///     each with ``label``, ``source``, ``found`` (its index into ``found``),
///     ``layer``, ``rank``, ``confidence`` (its layer's, when ranked),
///     ``pixel`` (where it sits in the queried image), ``distance_px``,
///     ``range``, ``n_views``, ``point`` (the existing point it is, or with
///     ``commit`` the new point it became), ``track`` (the built
///     :class:`EditableTrack`, or ``None``) and, when building or committing
///     it failed, ``error``; ``found``, every track found, usable or not, as
///     the harness's anchor dicts with ``label`` (``None`` off the bench) and
///     ``duplicate_of`` (the index into ``found`` of the track whose built
///     track it repeats, which leaves it off the bench, or ``None``), and
///     ``layers``, as the harness's layer dicts; ``stages``, the
///     harness's per-source records (``source``, ``found``, ``seconds``,
///     ``range_seconds``, and ``skipped`` naming the missing input); and
///     ``report``: ``sources``, ``stopped_after``, ``far_field`` (``trigger``,
///     ``found``, ``dropped``, ``seconds``, or ``None`` when it did not run),
///     ``layers_seconds``, ``tracks_seconds`` and ``duplicates``.
///
/// Raises:
///     ValueError: the image or pixel names no place, an input does not
///         match the reconstruction, an option is unknown, or ``label`` is
///         empty, all whitespace or holds a control character.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (edited, images, sources, image, pixel, *, options = None, label = None,
                    commit = false))]
pub(super) fn find_nearby_tracks(
    py: Python<'_>,
    edited: &PyEditedReconstruction,
    images: &Bound<'_, PyAny>,
    sources: &PyNearbyTrackSources,
    image: u32,
    pixel: [f64; 2],
    options: Option<&Bound<'_, PyDict>>,
    label: Option<String>,
    commit: bool,
) -> PyResult<Py<PyAny>> {
    let overrides = options;
    let mut options = NearbyTrackOptions::default();
    if let Some(overrides) = overrides {
        for (key, value) in overrides.iter() {
            let key: String = key.extract()?;
            set_option(&mut options, &key, &value)?;
        }
    }
    options.label = label;
    let posed = PosedViews::from_reconstruction(&edited.inner.base);
    let pyramids = resolve_pyramids(&posed, images)?;
    let grey = resolve_grey(images, posed.cameras.len());
    let views = views_of(&posed, &pyramids);
    let sift = sources.sift_index(py);
    let core_sources = NearbyTrackSources {
        clusters: sources.clusters(),
        guided: sources.guided(),
        sift_index: sift.as_ref().map(|(forest, keypoints)| SiftIndexSource {
            forest: forest.inner(),
            keypoints,
        }),
    };
    let found = py
        .detach(|| {
            core_find_nearby_tracks(
                &edited.inner,
                &views,
                &grey,
                &core_sources,
                image,
                pixel,
                &options,
                &Progress::none(),
            )
        })
        .map_err(refused)?;

    let out = result_dict(py, &found, options.layers.evidence)?;
    if !commit {
        return Ok(out.into_any().unbind());
    }
    let mut version = edited.inner.clone();
    let tracks = out
        .get_item("tracks")?
        .expect("the result carries its tracks")
        .cast_into::<PyList>()?;
    for (row, k) in tracks.iter().zip(found.bench_order()) {
        let t = &found.tracks[k];
        if t.point.is_some() {
            continue;
        }
        let Some(Ok(track)) = &t.track else { continue };
        match core_commit(&version, track) {
            Ok((next, report)) => {
                version = next;
                row.set_item("point", report.point)?;
            }
            Err(e) => row.set_item("error", format!("commit: {e}"))?,
        }
    }
    Ok((PyEditedReconstruction { inner: version }, out)
        .into_pyobject(py)?
        .into_any()
        .unbind())
}

/// The result as the binding's dict.
fn result_dict<'py>(
    py: Python<'py>,
    found: &NearbyTracks,
    evidence: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("group_label", &found.group_label)?;

    let tracks = PyList::empty(py);
    for k in found.bench_order() {
        let t = &found.tracks[k];
        let layer = t.layer.expect("bench order holds tracks in a layer");
        let d = PyDict::new(py);
        d.set_item("label", &t.label)?;
        d.set_item("source", t.source().name())?;
        d.set_item("found", k)?;
        d.set_item("layer", layer)?;
        let ranking = found.layers[layer].ranking.as_ref();
        d.set_item("rank", ranking.map(|r| r.rank))?;
        d.set_item("confidence", ranking.map(|r| r.confidence))?;
        d.set_item("pixel", t.query_pixel())?;
        d.set_item("distance_px", t.distance_px())?;
        d.set_item("range", t.range.to_vec())?;
        d.set_item("n_views", t.n_views())?;
        d.set_item("point", t.point)?;
        match &t.track {
            Some(Ok(track)) => d.set_item(
                "track",
                PyEditableTrack {
                    inner: Arc::new(track.clone()),
                },
            )?,
            Some(Err(e)) => {
                d.set_item("track", py.None())?;
                d.set_item("error", e)?;
            }
            None => d.set_item("track", py.None())?,
        }
        tracks.append(d)?;
    }
    out.set_item("tracks", tracks)?;

    let all = PyList::empty(py);
    for t in &found.tracks {
        let d = match &t.finding {
            NearbyFinding::Candidate(c) => candidate_dict(py, c)?,
            NearbyFinding::FarField(r) => reading_dict(py, r)?,
        };
        d.set_item("distance", t.distance)?;
        d.set_item("range", t.range.to_vec())?;
        d.set_item("bounded", t.class.bounded)?;
        d.set_item("far", t.class.far)?;
        d.set_item("support", t.support)?;
        d.set_item("label", &t.label)?;
        d.set_item("duplicate_of", t.duplicate_of)?;
        all.append(d)?;
    }
    out.set_item("found", all)?;

    let layers = PyList::empty(py);
    for layer in &found.layers {
        layers.append(layer_dict(py, layer)?)?;
    }
    out.set_item("layers", layers)?;

    let r = &found.report;
    let stages = PyList::empty(py);
    let reports = PyList::empty(py);
    let stage = |s: &SourceReport| -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("source", harness_name(s.source))?;
        d.set_item("found", s.found)?;
        d.set_item("seconds", s.seconds)?;
        d.set_item("range_seconds", s.range_seconds)?;
        if let Some(missing) = s.skipped {
            d.set_item("skipped", missing)?;
        }
        Ok(d)
    };
    for s in &r.sources {
        stages.append(stage(s)?)?;
        let d = stage(s)?;
        d.set_item("source", s.source.name())?;
        reports.append(d)?;
    }
    let far_field = match &r.far_field {
        Some(run) => {
            stages.append(stage(&run.report)?)?;
            let d = PyDict::new(py);
            d.set_item(
                "trigger",
                match run.trigger {
                    FarFieldTrigger::Always => "always",
                    FarFieldTrigger::NoLayer => "no_layer",
                    FarFieldTrigger::SeveralLayers => "several_layers",
                    FarFieldTrigger::NoneAtPixel => "none_at_pixel",
                },
            )?;
            d.set_item("found", run.report.found)?;
            d.set_item("dropped", run.dropped)?;
            d.set_item("seconds", run.report.seconds)?;
            d.set_item("range_seconds", run.report.range_seconds)?;
            d.into_any()
        }
        None => py.None().into_bound(py),
    };
    if evidence && !found.layers.is_empty() {
        // The harness's record of the layers: support, grouping, evidence and
        // ranking together.
        let d = PyDict::new(py);
        d.set_item("source", "evidence")?;
        d.set_item("found", 0)?;
        d.set_item("seconds", r.layers_seconds)?;
        stages.append(d)?;
    }
    out.set_item("stages", stages)?;

    let report = PyDict::new(py);
    report.set_item("sources", reports)?;
    report.set_item("stopped_after", r.stopped_after.map(|s| s.name()))?;
    report.set_item("far_field", far_field)?;
    report.set_item("layers_seconds", r.layers_seconds)?;
    report.set_item("tracks_seconds", r.tracks_seconds)?;
    report.set_item("duplicates", r.duplicates)?;
    out.set_item("report", report)?;
    Ok(out)
}

/// Register the nearby-tracks binding on the `sfmtool.bench` submodule.
pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(find_nearby_tracks, m)?)?;
    Ok(())
}
