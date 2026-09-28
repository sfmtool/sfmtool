// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for the matching sources of finding the tracks near a
//! pixel.
//!
//! Each source runs one query and gives back its candidates as dicts with the
//! keys the track-at-pixel harness's anchors carry, so the harness can call it
//! in place of its own source.

use std::path::PathBuf;

use numpy::{PyReadonlyArray2, PyReadonlyArray3};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use rayon::prelude::*;
use sfmtool_sift_format::read_sift_features;

use sfmtool_core::bench::{
    constellation_seeds as core_constellation_seeds, guided_matches as core_guided_matches,
    nearby_cluster_tracks as core_nearby_cluster_tracks, nearby_points as core_nearby_points,
    ClusterMembers, ClusterTracksOptions, ConstellationAt, ConstellationSeedOptions, GuidedOptions,
    GuidedSource, ImageDescriptors, KeypointRays, MatchesClusters, NearbyCandidate, NearbySource,
    PointsOptions, SiftIndexSource,
};
use sfmtool_core::features::kdforest::ImageKeypoints;

use super::{read_keypoints, views_of};
use crate::io::matches_file::PyMatchesFile;
use crate::patches::views::{resolve_pyramids, PosedViews, PyramidSet};
use crate::reconstruction::edited::PyEditedReconstruction;
use crate::spatial::kdf::PyLazyKdForest;

/// What the matching sources read beside the reconstruction and the
/// photographs, built once per capture and shared by every query.
///
/// Every input is optional, and a source whose input is missing finds
/// nothing: the clusters source needs ``matches``, guided matching the
/// keypoints and ``sift``, and the constellation ``forest`` and the keypoints.
///
/// Args:
///     edited: The reconstruction the inputs are indexed onto; only its image
///         names and count are read, so any version of one base will do.
///     forest: The SIFT index, whose corpus indexes the reconstruction's
///         images in the reconstruction's order.
///     keypoints: One ``(positions, affine_shapes)`` pair per image of the
///         reconstruction, in its order, as :class:`TrackAtPixelSources` takes
///         them. Read from ``sift`` when left out and ``sift`` is given.
///     matches: A cluster-patches :class:`MatchesFile`. Its images are matched
///         to the reconstruction's by name.
///     sift: One ``.sift`` path per image of the reconstruction, in its order,
///         whose descriptors are read, row for row with the keypoints.
#[pyclass(name = "NearbyTrackSources", module = "sfmtool.bench", frozen)]
pub struct PyNearbyTrackSources {
    image_count: usize,
    forest: Option<Py<PyLazyKdForest>>,
    keypoints: Option<Vec<ImageKeypoints>>,
    clusters: Option<MatchesClusters>,
    descriptors: Option<Vec<ImageDescriptors>>,
    rays: KeypointRays,
}

#[pymethods]
impl PyNearbyTrackSources {
    #[new]
    #[pyo3(signature = (edited, *, forest = None, keypoints = None, matches = None, sift = None))]
    fn new(
        py: Python<'_>,
        edited: &PyEditedReconstruction,
        forest: Option<Py<PyLazyKdForest>>,
        keypoints: Option<&Bound<'_, PyList>>,
        matches: Option<&PyMatchesFile>,
        sift: Option<Vec<PathBuf>>,
    ) -> PyResult<Self> {
        let names: Vec<&str> = edited
            .inner
            .base
            .image_table
            .images
            .iter()
            .map(|im| im.name.as_str())
            .collect();
        let image_count = names.len();
        let count_of = |input: &str, got: usize| {
            if got == image_count {
                Ok(())
            } else {
                Err(PyValueError::new_err(format!(
                    "{input} has {got} entries, but the reconstruction has {image_count} images"
                )))
            }
        };
        let mut keypoints = keypoints
            .map(|list| {
                count_of("keypoints", list.len())?;
                list.iter()
                    .map(|item| {
                        let pair = item.cast::<PyTuple>()?;
                        let positions: PyReadonlyArray2<'_, f32> = pair.get_item(0)?.extract()?;
                        let shapes: PyReadonlyArray3<'_, f32> = pair.get_item(1)?.extract()?;
                        read_keypoints(&positions, &shapes)
                    })
                    .collect::<PyResult<Vec<_>>>()
            })
            .transpose()?;
        let clusters = matches
            .map(|m| MatchesClusters::new(m.data(), &names))
            .transpose()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let mut descriptors = None;
        if let Some(paths) = sift {
            count_of("sift", paths.len())?;
            let read = py
                .detach(|| {
                    paths
                        .par_iter()
                        .map(|path| {
                            read_sift_features(path).map_err(|e| format!("{}: {e}", path.display()))
                        })
                        .collect::<Result<Vec<_>, String>>()
                })
                .map_err(PyValueError::new_err)?;
            if keypoints.is_none() {
                keypoints = Some(
                    read.iter()
                        .map(|f| ImageKeypoints {
                            positions: f.positions_xy.clone(),
                            affine_shapes: f.affine_shapes.clone(),
                        })
                        .collect(),
                );
            }
            descriptors = Some(
                read.into_iter()
                    .map(|f| ImageDescriptors::new(f.descriptors))
                    .collect(),
            );
        }
        Ok(Self {
            image_count,
            forest,
            keypoints,
            clusters,
            descriptors,
            rays: KeypointRays::new(image_count),
        })
    }

    /// Whether the clusters source has its input.
    #[getter]
    fn has_clusters(&self) -> bool {
        self.clusters.is_some()
    }

    /// Whether guided matching has its inputs.
    #[getter]
    fn has_guided(&self) -> bool {
        self.guided().is_some()
    }

    /// Whether the constellation source has its inputs.
    #[getter]
    fn has_constellation(&self) -> bool {
        self.forest.is_some() && self.keypoints.is_some()
    }

    fn __repr__(&self) -> String {
        format!(
            "NearbyTrackSources({} images, clusters: {}, guided: {}, constellation: {})",
            self.image_count,
            self.clusters
                .as_ref()
                .map_or_else(|| "none".into(), |c| c.cluster_count().to_string()),
            self.has_guided(),
            self.has_constellation(),
        )
    }
}

impl PyNearbyTrackSources {
    /// What guided matching reads, when every part of it is here.
    fn guided(&self) -> Option<GuidedSource<'_>> {
        Some(GuidedSource {
            keypoints: self.keypoints.as_deref()?,
            descriptors: self.descriptors.as_deref()?,
            rays: &self.rays,
        })
    }
}

/// The posed views of `edited` over `images`, which a query reads.
fn posed_views(
    edited: &PyEditedReconstruction,
    images: &Bound<'_, PyAny>,
) -> PyResult<(PosedViews, PyramidSet)> {
    let posed = PosedViews::from_reconstruction(&edited.inner.base);
    let pyramids = resolve_pyramids(&posed, images)?;
    Ok((posed, pyramids))
}

/// Apply the `options` overrides through `set`, one key at a time.
fn with_overrides<T>(
    mut options: T,
    overrides: Option<&Bound<'_, PyDict>>,
    set: impl Fn(&mut T, &str, &Bound<'_, PyAny>) -> PyResult<bool>,
) -> PyResult<T> {
    if let Some(overrides) = overrides {
        for (key, value) in overrides.iter() {
            let key: String = key.extract()?;
            if !set(&mut options, &key, &value)? {
                return Err(PyValueError::new_err(format!("unknown option {key:?}")));
            }
        }
    }
    Ok(options)
}

/// The name the harness gives a source's anchors.
fn harness_name(source: NearbySource) -> &'static str {
    match source {
        // The harness calls the reconstruction's own points its tracks.
        NearbySource::Points => "tracks",
        NearbySource::Clusters => "clusters",
        NearbySource::Guided => "guided",
        NearbySource::Constellation => "constellation",
    }
}

/// One candidate as the harness's anchor dict.
fn candidate_dict<'py>(py: Python<'py>, c: &NearbyCandidate) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("source", harness_name(c.source))?;
    d.set_item("id", c.id)?;
    d.set_item("position", [c.position.x, c.position.y, c.position.z])?;
    // Each row a list `[image, x, y]`, as the harness writes them.
    let views = PyList::empty(py);
    for &(image, px) in &c.sightings {
        let row = PyList::empty(py);
        row.append(image)?;
        row.append(px[0])?;
        row.append(px[1])?;
        views.append(row)?;
    }
    d.set_item("views", views)?;
    d.set_item("query_pixel", c.query_pixel)?;
    d.set_item("distance_px", c.distance_px)?;
    d.set_item("n_views", c.n_views())?;
    d.set_item("max_reproj_px", c.max_reproj_px)?;
    d.set_item("max_ray_angle_deg", c.max_ray_angle_deg)?;
    d.set_item("depth", c.depth)?;
    Ok(d)
}

fn candidate_list(py: Python<'_>, found: &[NearbyCandidate]) -> PyResult<Py<PyList>> {
    let out = PyList::empty(py);
    for c in found {
        out.append(candidate_dict(py, c)?)?;
    }
    Ok(out.unbind())
}

/// The reconstruction's points observed near ``pixel`` in ``image``, nearest
/// first, as candidate tracks.
///
/// A point is kept when it is finite, has ``min_views`` or more observations
/// and every observation lies within ``max_reproj_px`` of where it projects;
/// at most ``max_points`` are kept.
///
/// Args:
///     edited: The reconstruction, whose deleted points are never found.
///     images: One decoded image per image of ``edited``, or a prebuilt
///         :class:`ImagePyramidSet`, as :func:`far_field_sweep` takes them.
///         Only the cameras are read.
///     image: The queried image's index.
///     pixel: ``(x, y)`` in that image.
///     options: Overrides keyed by the field of the Rust ``PointsOptions``:
///         ``radius_px`` (40), ``max_points`` (8), ``min_views`` (2) and
///         ``max_reproj_px`` (2). An unknown key is an error.
///
/// Returns:
///     A list of dicts with the harness's anchor keys: ``source``
///     (``"tracks"``), ``id`` (the point), ``position``, ``views``
///     (``[image, x, y]`` rows, the queried image first), ``query_pixel``,
///     ``distance_px``, ``n_views``, ``max_reproj_px``, ``max_ray_angle_deg``
///     and ``depth``.
///
/// Raises:
///     ValueError: the image or pixel names no place, or an input does not
///         match the reconstruction.
#[pyfunction]
#[pyo3(signature = (edited, images, image, pixel, *, options = None))]
pub(super) fn nearby_points(
    py: Python<'_>,
    edited: &PyEditedReconstruction,
    images: &Bound<'_, PyAny>,
    image: u32,
    pixel: [f64; 2],
    options: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyList>> {
    let options = with_overrides(PointsOptions::default(), options, |o, key, value| {
        match key {
            "radius_px" => o.radius_px = value.extract()?,
            "max_points" => o.max_points = value.extract()?,
            "min_views" => o.min_views = value.extract()?,
            "max_reproj_px" => o.max_reproj_px = value.extract()?,
            _ => return Ok(false),
        }
        Ok(true)
    })?;
    let (posed, pyramids) = posed_views(edited, images)?;
    let views = views_of(&posed, &pyramids);
    let found = py
        .detach(|| core_nearby_points(&edited.inner, &views, image, pixel, &options))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    candidate_list(py, &found)
}

/// The cluster-patches clusters with a member near ``pixel`` in ``image``,
/// nearest first, each vetted by triangulating its members, as candidate
/// tracks.
///
/// A cluster's member in ``image`` is its nearest there; elsewhere it
/// contributes one admitted member per image, the reference or a kept one
/// first, then the best-reading. The members are triangulated, dropping the
/// worst while three or more remain, until every one is within
/// ``max_reproj_px``; never the queried member.
///
/// Args:
///     edited: The reconstruction; only its cameras are read.
///     images: As :func:`nearby_points` takes them.
///     sources: A :class:`NearbyTrackSources`; with no ``matches`` the result
///         is empty.
///     image: The queried image's index.
///     pixel: ``(x, y)`` in that image.
///     options: Overrides keyed by the field of the Rust
///         ``ClusterTracksOptions``: ``radius_px`` (48), ``max_clusters``
///         (16), ``max_reproj_px`` (2) and ``members`` (``"any"`` or
///         ``"kept"``). An unknown key is an error.
///
/// Returns:
///     A list of the harness's anchor dicts, as :func:`nearby_points` returns
///     them, with ``source`` ``"clusters"`` and ``id`` the cluster.
#[pyfunction]
#[pyo3(signature = (edited, images, sources, image, pixel, *, options = None))]
pub(super) fn nearby_cluster_tracks(
    py: Python<'_>,
    edited: &PyEditedReconstruction,
    images: &Bound<'_, PyAny>,
    sources: &PyNearbyTrackSources,
    image: u32,
    pixel: [f64; 2],
    options: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyList>> {
    let options = with_overrides(ClusterTracksOptions::default(), options, |o, key, value| {
        match key {
            "radius_px" => o.radius_px = value.extract()?,
            "max_clusters" => o.max_clusters = value.extract()?,
            "max_reproj_px" => o.max_reproj_px = value.extract()?,
            "members" => {
                let word: String = value.extract()?;
                o.members = word
                    .parse::<ClusterMembers>()
                    .map_err(PyValueError::new_err)?;
            }
            _ => return Ok(false),
        }
        Ok(true)
    })?;
    let Some(clusters) = &sources.clusters else {
        return Ok(PyList::empty(py).unbind());
    };
    let (posed, pyramids) = posed_views(edited, images)?;
    let views = views_of(&posed, &pyramids);
    let found = py
        .detach(|| core_nearby_cluster_tracks(&views, clusters, image, pixel, &options))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    candidate_list(py, &found)
}

/// The keypoints near ``pixel`` in ``image``, each matched by descriptor along
/// the rays of every other image's keypoints, as candidate tracks.
///
/// A keypoint's match in another image is a keypoint whose ray passes within
/// ``epipolar_px`` of its own, whose descriptor is the nearest among those and
/// within ``ratio`` of the second nearest and ``max_distance``. Each keypoint's
/// matches are triangulated with it, dropping the worst while three or more
/// remain; a match past ``max_distance`` but within ``loose_distance`` is then
/// added, most distinct first, when the triangulation with it still meets
/// every sighting within ``max_reproj_px``.
///
/// Args:
///     edited: The reconstruction; only its cameras are read.
///     images: As :func:`nearby_points` takes them.
///     sources: A :class:`NearbyTrackSources`; without keypoints and ``sift``
///         the result is empty. The rays through its keypoints are built from
///         the cameras the first time each image is read and kept.
///     image: The queried image's index.
///     pixel: ``(x, y)`` in that image.
///     options: Overrides keyed by the field of the Rust ``GuidedOptions``:
///         ``radius_px`` (24), ``max_keypoints`` (8), ``skip_px`` (-1),
///         ``epipolar_px`` (2), ``ratio`` (0.8), ``max_distance`` (250),
///         ``loose_distance`` (400), ``min_views`` (2) and ``max_reproj_px``
///         (2). An unknown key is an error.
///
/// Returns:
///     A list of the harness's anchor dicts, as :func:`nearby_points` returns
///     them, with ``source`` ``"guided"`` and ``id`` the keypoint's row.
#[pyfunction]
#[pyo3(signature = (edited, images, sources, image, pixel, *, options = None))]
pub(super) fn guided_matches(
    py: Python<'_>,
    edited: &PyEditedReconstruction,
    images: &Bound<'_, PyAny>,
    sources: &PyNearbyTrackSources,
    image: u32,
    pixel: [f64; 2],
    options: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyList>> {
    let options = with_overrides(GuidedOptions::default(), options, |o, key, value| {
        match key {
            "radius_px" => o.radius_px = value.extract()?,
            "max_keypoints" => o.max_keypoints = value.extract()?,
            "skip_px" => o.skip_px = value.extract()?,
            "epipolar_px" => o.epipolar_px = value.extract()?,
            "ratio" => o.ratio = value.extract()?,
            "max_distance" => o.max_distance = value.extract()?,
            "loose_distance" => o.loose_distance = value.extract()?,
            "min_views" => o.min_views = value.extract()?,
            "max_reproj_px" => o.max_reproj_px = value.extract()?,
            _ => return Ok(false),
        }
        Ok(true)
    })?;
    let Some(source) = sources.guided() else {
        return Ok(PyList::empty(py).unbind());
    };
    let (posed, pyramids) = posed_views(edited, images)?;
    let views = views_of(&posed, &pyramids);
    let found = py
        .detach(|| core_guided_matches(&views, &source, image, pixel, &options))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    candidate_list(py, &found)
}

/// The SIFT index's constellation query from ``pixel`` in ``image``, and with
/// ``at="keypoints"`` from the keypoints near it, each as a candidate track.
///
/// Each other image whose matches agree on one affine warp carries the query's
/// position into its own frame; those positions and the query's own are
/// triangulated, dropping the worst while three or more remain, until every
/// one is within ``max_reproj_px``.
///
/// Args:
///     edited: The reconstruction; its cameras and the image's name are read.
///     images: As :func:`nearby_points` takes them.
///     sources: A :class:`NearbyTrackSources`; without ``forest`` and keypoints
///         the result is empty.
///     image: The queried image's index.
///     pixel: ``(x, y)`` in that image.
///     options: Overrides keyed by the field of the Rust
///         ``ConstellationSeedOptions``: ``target`` (50), ``min_inliers`` (6),
///         ``seed_radius_px`` (6), ``max_reproj_px`` (3), ``at`` (``"pixel"``
///         or ``"keypoints"``), ``lateral_max`` (4) and ``lateral_radius_px``
///         (24). An unknown key is an error.
///
/// Returns:
///     A list of the harness's anchor dicts, as :func:`nearby_points` returns
///     them, with ``source`` ``"constellation"`` and ``id`` ``None`` for the
///     pixel's query or the keypoint's row for a keypoint's.
#[pyfunction]
#[pyo3(signature = (edited, images, sources, image, pixel, *, options = None))]
pub(super) fn constellation_seeds(
    py: Python<'_>,
    edited: &PyEditedReconstruction,
    images: &Bound<'_, PyAny>,
    sources: &PyNearbyTrackSources,
    image: u32,
    pixel: [f64; 2],
    options: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyList>> {
    let options = with_overrides(
        ConstellationSeedOptions::default(),
        options,
        |o, key, value| {
            match key {
                "target" => o.target = value.extract()?,
                "min_inliers" => o.min_inliers = value.extract()?,
                "seed_radius_px" => o.seed_radius_px = value.extract()?,
                "max_reproj_px" => o.max_reproj_px = value.extract()?,
                "at" => {
                    let word: String = value.extract()?;
                    o.at = word
                        .parse::<ConstellationAt>()
                        .map_err(PyValueError::new_err)?;
                }
                "lateral_max" => o.lateral_max = value.extract()?,
                "lateral_radius_px" => o.lateral_radius_px = value.extract()?,
                _ => return Ok(false),
            }
            Ok(true)
        },
    )?;
    let (Some(forest), Some(keypoints)) = (&sources.forest, &sources.keypoints) else {
        return Ok(PyList::empty(py).unbind());
    };
    let forest = forest.bind(py).borrow();
    let index = SiftIndexSource {
        forest: forest.inner(),
        keypoints,
    };
    let (posed, pyramids) = posed_views(edited, images)?;
    let views = views_of(&posed, &pyramids);
    let found = py
        .detach(|| core_constellation_seeds(&edited.inner, &views, &index, image, pixel, &options))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    candidate_list(py, &found)
}

/// Register the matching-source bindings on the `sfmtool.bench` submodule.
pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyNearbyTrackSources>()?;
    m.add_function(wrap_pyfunction!(nearby_points, m)?)?;
    m.add_function(wrap_pyfunction!(nearby_cluster_tracks, m)?)?;
    m.add_function(wrap_pyfunction!(guided_matches, m)?)?;
    m.add_function(wrap_pyfunction!(constellation_seeds, m)?)?;
    Ok(())
}
