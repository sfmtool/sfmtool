// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Python binding for the depth layers.
//!
//! [`depth_layers`] takes the track-at-pixel harness's anchor dicts, with the
//! ranges and classes the harness gave them, and gives back each one's support
//! and the layers as dicts with the keys the harness's own layers carry, so
//! the harness can call it in place of its own grouping, evidence and ranking.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use sfmtool_core::bench::{
    depth_layers as core_depth_layers, DepthLayer, DepthLayerOptions, LayerCandidate, LayerRankBy,
    NearbySource, RangeClass,
};

use super::nearby::harness_name;
use super::{refused, views_of};
use crate::patches::views::{resolve_grey, resolve_pyramids, PosedViews};
use crate::reconstruction::edited::PyEditedReconstruction;

/// The source the harness's name `word` names.
fn source_named(word: &str) -> PyResult<NearbySource> {
    Ok(match word {
        "tracks" => NearbySource::Points,
        "clusters" => NearbySource::Clusters,
        "guided" => NearbySource::Guided,
        "constellation" => NearbySource::Constellation,
        "farfield" => NearbySource::FarField,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown anchor source {other:?} (expected \
                 tracks|clusters|guided|constellation|farfield)"
            )))
        }
    })
}

/// One anchor dict's fields, owned, from which a [`LayerCandidate`] borrows.
struct Anchor {
    source: NearbySource,
    sightings: Vec<(u32, [f64; 2])>,
    distance_px: f64,
    max_ray_angle_deg: f64,
    range: [f64; 2],
    class: RangeClass,
}

impl Anchor {
    fn from_dict(k: usize, d: &Bound<'_, PyAny>) -> PyResult<Self> {
        let field = |key: &str| {
            d.get_item(key)
                .map_err(|_| PyValueError::new_err(format!("anchor {k} has no {key:?}")))
        };
        let source: String = field("source")?.extract()?;
        let rows: Vec<Vec<f64>> = field("views")?.extract()?;
        let mut sightings = Vec::with_capacity(rows.len());
        for row in &rows {
            if row.len() != 3 || row[0] < 0.0 || row[0].fract() != 0.0 {
                return Err(PyValueError::new_err(format!(
                    "anchor {k}'s views must be [image, x, y] rows"
                )));
            }
            sightings.push((row[0] as u32, [row[1], row[2]]));
        }
        let n_views: usize = field("n_views")?.extract()?;
        if n_views != sightings.len() {
            return Err(PyValueError::new_err(format!(
                "anchor {k} has n_views {n_views} but {} views",
                sightings.len()
            )));
        }
        Ok(Self {
            source: source_named(&source)?,
            sightings,
            distance_px: field("distance_px")?.extract()?,
            max_ray_angle_deg: field("max_ray_angle_deg")?.extract()?,
            range: field("range")?.extract()?,
            class: RangeClass {
                bounded: field("bounded")?.extract()?,
                far: field("far")?.extract()?,
            },
        })
    }

    fn candidate(&self) -> LayerCandidate<'_> {
        LayerCandidate {
            source: self.source,
            sightings: &self.sightings,
            distance_px: self.distance_px,
            max_ray_angle_deg: self.max_ray_angle_deg,
            range: self.range,
            class: self.class,
        }
    }
}

/// Group the anchors near ``pixel`` in ``image`` into depth layers, read each
/// layer with the pixel's own patch, and rank them.
///
/// An anchor is usable when it is ``bounded`` or ``far``. The usable anchors
/// are taken nearest ``range`` first and each joins the last layer when its
/// near end is within that layer's range, or starts a new one, so the layers
/// are nearest first. An anchor's support counts the other usable anchors whose
/// ranges overlap its own and whose images are not all among its own, nor its
/// among theirs.
///
/// With ``evidence``, the pixel's patch (an 11 x 11 grid of ``radius_px``) is
/// read in every other image at ``samples`` distances across each layer's
/// range, even in inverse distance and from infinity in for a layer with no
/// far end, and each layer gets its evidence, a score from that reading, a key
/// that adds the middle of the patch and the anchors' nearness, a rank by the
/// key (or the score) and a confidence that the pixel is on it.
///
/// Args:
///     edited: The reconstruction whose cameras and poses are read.
///     images: One decoded image per image of ``edited``, or a prebuilt
///         :class:`ImagePyramidSet`, as :func:`far_field_sweep` takes them. A
///         set also keeps the grey images between calls.
///     image: The queried image's index.
///     pixel: ``(x, y)`` in that image.
///     anchors: The harness's anchor dicts. Each is read for ``source``
///         (``"tracks"``, ``"clusters"``, ``"guided"``, ``"constellation"`` or
///         ``"farfield"``), ``views`` (``[image, x, y]`` rows), ``n_views``,
///         ``distance_px``, ``max_ray_angle_deg``, ``range``, ``bounded`` and
///         ``far``; nothing else, and nothing is written to them.
///     options: Overrides keyed by the field of the Rust
///         ``DepthLayerOptions``: ``evidence`` (``True``), ``rank_by``
///         (``"evidence"`` or ``"score"``), ``radius_px`` (8) and ``samples``
///         (5). An unknown key is an error.
///
/// Returns:
///     A dict with ``support``, one count per anchor in the order given, and
///     ``layers``, nearest first, each a dict with ``range``, ``anchors`` (the
///     members' indexes), ``nearest_px`` and ``views`` (the most views of a
///     member), and with ``evidence`` also ``evidence`` (``n_anchors``,
///     ``n_independent``, ``n_images``, ``max_views``, ``max_ray_angle``,
///     ``nearest_px``, ``at_pixel``, ``sources``, ``support``, ``photo``,
///     ``votes``, ``votes_all``, ``photo_mid``, ``photo_both``), ``score``,
///     ``key``, ``rank`` and ``confidence``: the keys the harness's own
///     layers carry.
///
/// Raises:
///     ValueError: the image is not one of the reconstruction's, an anchor
///         lacks a field or names an unknown source, or ``images`` does not
///         match the reconstruction.
#[pyfunction]
#[pyo3(signature = (edited, images, image, pixel, anchors, *, options = None))]
pub(super) fn depth_layers(
    py: Python<'_>,
    edited: &PyEditedReconstruction,
    images: &Bound<'_, PyAny>,
    image: u32,
    pixel: [f64; 2],
    anchors: &Bound<'_, PyList>,
    options: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyDict>> {
    let overrides = options;
    let mut options = DepthLayerOptions::default();
    if let Some(overrides) = overrides {
        for (key, value) in overrides.iter() {
            let key: String = key.extract()?;
            match key.as_str() {
                "evidence" => options.evidence = value.extract()?,
                "rank_by" => {
                    let word: String = value.extract()?;
                    options.rank_by = word.parse::<LayerRankBy>().map_err(PyValueError::new_err)?;
                }
                "radius_px" => options.radius_px = value.extract()?,
                "samples" => options.samples = value.extract()?,
                _ => return Err(PyValueError::new_err(format!("unknown option {key:?}"))),
            }
        }
    }
    let owned = anchors
        .iter()
        .enumerate()
        .map(|(k, d)| Anchor::from_dict(k, &d))
        .collect::<PyResult<Vec<_>>>()?;
    let posed = PosedViews::from_reconstruction(&edited.inner.base);
    let pyramids = resolve_pyramids(&posed, images)?;
    let grey = resolve_grey(images, posed.cameras.len());
    let views = views_of(&posed, &pyramids);
    let found = py
        .detach(|| {
            let candidates: Vec<LayerCandidate<'_>> = owned.iter().map(Anchor::candidate).collect();
            core_depth_layers(&views, &grey, image, pixel, &candidates, &options)
        })
        .map_err(refused)?;
    let out = PyDict::new(py);
    out.set_item("support", &found.support)?;
    let layers = PyList::empty(py);
    for layer in &found.layers {
        layers.append(layer_dict(py, layer)?)?;
    }
    out.set_item("layers", layers)?;
    Ok(out.unbind())
}

/// One layer as the harness's layer dict.
fn layer_dict<'py>(py: Python<'py>, layer: &DepthLayer) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("range", layer.range.to_vec())?;
    d.set_item("anchors", &layer.members)?;
    d.set_item("nearest_px", layer.nearest_px)?;
    d.set_item("views", layer.max_views)?;
    let Some(r) = &layer.ranking else {
        return Ok(d);
    };
    let e = &r.evidence;
    let ev = PyDict::new(py);
    ev.set_item("n_anchors", e.n_candidates)?;
    ev.set_item("n_independent", e.n_independent)?;
    ev.set_item("n_images", e.n_images)?;
    ev.set_item("max_views", e.max_views)?;
    ev.set_item("max_ray_angle", e.max_ray_angle_deg)?;
    ev.set_item("nearest_px", e.nearest_px)?;
    ev.set_item("at_pixel", e.at_pixel)?;
    let mut sources: Vec<&str> = e.sources.iter().map(|&s| harness_name(s)).collect();
    sources.sort_unstable();
    ev.set_item("sources", sources)?;
    ev.set_item("support", e.weight)?;
    ev.set_item("photo", e.photo)?;
    ev.set_item("votes", e.votes)?;
    ev.set_item("votes_all", e.votes_all)?;
    ev.set_item("photo_mid", e.photo_middle)?;
    ev.set_item("photo_both", e.photo_both)?;
    d.set_item("evidence", ev)?;
    d.set_item("score", r.score)?;
    d.set_item("key", r.key)?;
    d.set_item("rank", r.rank)?;
    d.set_item("confidence", r.confidence)?;
    Ok(d)
}

/// Register the depth-layers binding on the `sfmtool.bench` submodule.
pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(depth_layers, m)?)?;
    Ok(())
}
