// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! `SfmrReconstruction.resolve_measurement_endpoint`,
//! `SfmrReconstruction.resolve_distance_measurements` and
//! `SfmrReconstruction.measurement_endpoint_views`: the 3D position of a
//! measurement endpoint, a pixel in one image, read from the observations
//! whose squares hold it, and the other images that show the same place (see
//! ``specs/core/reconstruction/measurement-endpoint.md``).

use std::sync::Arc;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use sfmtool_core::reconstruction::measurement_endpoint::{
    endpoint_views as core_endpoint_views, resolve_distance_measurements as core_resolve_all,
    resolve_measurement_endpoint as core_resolve, CandidateStanding, EndpointCandidate,
    EndpointCost, EndpointPosition, EndpointRefusal, EndpointResolution, EndpointView,
    MeasurementEndpointOptions,
};
use sfmtool_core::reconstruction::EditedReconstruction;

use super::sfmr_reconstruction::PySfmrReconstruction;

/// The options, each keyword left `None` taking its default.
#[allow(clippy::too_many_arguments)]
fn options(
    layer_depth_factor: Option<f64>,
    offset_weight: Option<f64>,
    placeholder_normal_factor: Option<f64>,
    angle_weight: Option<f64>,
    capped_radius_factor: Option<f64>,
    unread_radius: Option<f64>,
    agreement_band: Option<f64>,
    agreement_scale: Option<f64>,
) -> MeasurementEndpointOptions {
    let mut o = MeasurementEndpointOptions::default();
    let set = |field: &mut f64, value: Option<f64>| {
        if let Some(v) = value {
            *field = v;
        }
    };
    set(&mut o.layer_depth_factor, layer_depth_factor);
    set(&mut o.offset_weight, offset_weight);
    set(&mut o.placeholder_normal_factor, placeholder_normal_factor);
    let rule = &mut o.reference_view;
    set(&mut rule.angle_weight, angle_weight);
    set(&mut rule.capped_radius_factor, capped_radius_factor);
    set(&mut rule.unread_radius, unread_radius);
    set(&mut rule.agreement_band, agreement_band);
    set(&mut rule.agreement_scale, agreement_scale);
    o
}

fn cost_to_py<'py>(py: Python<'py>, c: &EndpointCost) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("localization", c.localization)?;
    d.set_item("localization_read", c.localization_read)?;
    d.set_item("capped_radius", c.capped_radius)?;
    d.set_item("offset", c.offset)?;
    d.set_item("angle", c.angle)?;
    d.set_item("agreement", c.agreement)?;
    d.set_item("cost", c.cost)?;
    Ok(d)
}

fn candidate_to_py<'py>(py: Python<'py>, c: &EndpointCandidate) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("point", c.point)?;
    d.set_item("observation", c.observation)?;
    d.set_item("patch_coords", (c.patch_coords[0], c.patch_coords[1]))?;
    match c.position {
        EndpointPosition::Finite { position } => {
            d.set_item("position", (position[0], position[1], position[2]))?;
            d.set_item("direction", py.None())?;
        }
        EndpointPosition::AtInfinity { direction } => {
            d.set_item("position", py.None())?;
            d.set_item("direction", (direction[0], direction[1], direction[2]))?;
        }
    }
    d.set_item("depth", c.depth)?;
    let (standing, exclusion, cost) = match &c.standing {
        CandidateStanding::Alone => ("alone", None, None),
        CandidateStanding::Ranked(cost) => ("ranked", None, Some(cost_to_py(py, cost)?)),
        CandidateStanding::Excluded(e) => ("excluded", Some(e.name()), None),
    };
    d.set_item("standing", standing)?;
    d.set_item("exclusion", exclusion)?;
    d.set_item("cost", cost)?;
    Ok(d)
}

fn resolution_to_py<'py>(py: Python<'py>, r: &EndpointResolution) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("chosen", r.chosen)?;
    let candidates = PyList::empty(py);
    for c in &r.candidates {
        candidates.append(candidate_to_py(py, c)?)?;
    }
    d.set_item("candidates", candidates)?;
    Ok(d)
}

fn view_to_py<'py>(py: Python<'py>, v: &EndpointView) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("point", v.point)?;
    d.set_item("observation", v.observation)?;
    d.set_item("image", v.image)?;
    d.set_item("pixel", (v.pixel[0], v.pixel[1]))?;
    d.set_item("patch_coords", (v.patch_coords[0], v.patch_coords[1]))?;
    d.set_item("localization", v.localization)?;
    d.set_item("viewing_angle_deg", v.viewing_angle_deg)?;
    d.set_item("resolves_to", v.resolves_to)?;
    d.set_item("resolves_elsewhere", v.resolves_elsewhere())?;
    Ok(d)
}

/// The value the core functions read, holding this reconstruction.
fn edited(recon: &PySfmrReconstruction) -> EditedReconstruction {
    EditedReconstruction::new(Arc::new(recon.inner().clone()))
}

#[pymethods]
impl PySfmrReconstruction {
    /// Resolve one measurement endpoint, a pixel in one image, to a texel of
    /// a point's patch and the 3D position it stands for.
    ///
    /// The candidates are the observations in the image whose square (the
    /// point's patch slid onto the observation's keypoint) holds the pixel.
    /// With one, it is the answer and no stored reading is read. With several,
    /// those seen from behind and those behind the front layer are excluded
    /// and the rest ranked by a cost read from each observation's own stored
    /// readings. The position is the texel's place on the point's own patch.
    /// See ``specs/core/reconstruction/measurement-endpoint.md``.
    ///
    /// Args:
    ///     image: The image index.
    ///     xy: The pixel ``(x, y)``, top-left origin, half-pixel centres.
    ///     layer_depth_factor, offset_weight, placeholder_normal_factor,
    ///         angle_weight, capped_radius_factor, unread_radius,
    ///         agreement_band, agreement_scale: The options; ``None`` takes
    ///         the default.
    ///
    /// Returns:
    ///     A dict with ``chosen`` (an index into ``candidates``) and
    ///     ``candidates``, each a dict with ``point``, ``observation``,
    ///     ``patch_coords`` ``(s, t)``, ``position`` (``None`` at infinity),
    ///     ``direction`` (``None`` for a finite point), ``depth``,
    ///     ``standing`` (``"alone"``, ``"ranked"`` or ``"excluded"``),
    ///     ``exclusion`` and ``cost`` (a dict of the cost's factors, for a
    ///     ranked candidate). Raises ``ValueError`` with a sentence naming the
    ///     remedy where the endpoint has no position.
    // A Python docstring, not Rust prose: its indented `Args:` paragraphs read
    // as Markdown code blocks, which rustdoc would parse as Rust.
    #[allow(rustdoc::invalid_rust_codeblocks)]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        image, xy, *, layer_depth_factor=None, offset_weight=None,
        placeholder_normal_factor=None, angle_weight=None, capped_radius_factor=None,
        unread_radius=None, agreement_band=None, agreement_scale=None,
    ))]
    fn resolve_measurement_endpoint<'py>(
        &self,
        py: Python<'py>,
        image: u32,
        xy: (f64, f64),
        layer_depth_factor: Option<f64>,
        offset_weight: Option<f64>,
        placeholder_normal_factor: Option<f64>,
        angle_weight: Option<f64>,
        capped_radius_factor: Option<f64>,
        unread_radius: Option<f64>,
        agreement_band: Option<f64>,
        agreement_scale: Option<f64>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let opts = options(
            layer_depth_factor,
            offset_weight,
            placeholder_normal_factor,
            angle_weight,
            capped_radius_factor,
            unread_radius,
            agreement_band,
            agreement_scale,
        );
        let value = edited(self);
        let resolved = py
            .detach(|| core_resolve(&value, image, [xy.0, xy.1], &opts))
            .map_err(|e: EndpointRefusal| PyValueError::new_err(e.to_string()))?;
        resolution_to_py(py, &resolved)
    }

    /// The other images that show the place a measurement endpoint names:
    /// for each candidate ``resolve_measurement_endpoint`` finds at ``(image,
    /// xy)``, chosen or not, every other observation of its point, at the
    /// pixel where that observation's square holds the candidate's texel.
    ///
    /// A view is left out where its texel lands outside its image or its
    /// square faces away from its camera. The chosen candidate's point comes
    /// first. See ``specs/core/reconstruction/measurement-endpoint.md``.
    ///
    /// Args:
    ///     image: The image index.
    ///     xy: The pixel ``(x, y)``, top-left origin, half-pixel centres.
    ///     layer_depth_factor, offset_weight, placeholder_normal_factor,
    ///         angle_weight, capped_radius_factor, unread_radius,
    ///         agreement_band, agreement_scale: As
    ///         ``resolve_measurement_endpoint`` takes them.
    ///
    /// Returns:
    ///     A list of dicts, each with ``point``, ``observation`` (its index in
    ///     the point's track), ``image``, ``pixel`` ``(x, y)``,
    ///     ``patch_coords`` ``(s, t)``, ``localization`` (the observation's
    ///     stored self-similarity length on the patch, ``None`` where it has
    ///     no reading), ``viewing_angle_deg``, ``resolves_to`` (the point the
    ///     endpoint resolves to at that pixel, ``None`` where it is refused
    ///     there) and ``resolves_elsewhere`` (whether that is not ``point``).
    ///     Raises ``ValueError`` with the resolver's sentence where the
    ///     endpoint has no position.
    // A Python docstring, not Rust prose; see `resolve_measurement_endpoint`.
    #[allow(rustdoc::invalid_rust_codeblocks)]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        image, xy, *, layer_depth_factor=None, offset_weight=None,
        placeholder_normal_factor=None, angle_weight=None, capped_radius_factor=None,
        unread_radius=None, agreement_band=None, agreement_scale=None,
    ))]
    fn measurement_endpoint_views<'py>(
        &self,
        py: Python<'py>,
        image: u32,
        xy: (f64, f64),
        layer_depth_factor: Option<f64>,
        offset_weight: Option<f64>,
        placeholder_normal_factor: Option<f64>,
        angle_weight: Option<f64>,
        capped_radius_factor: Option<f64>,
        unread_radius: Option<f64>,
        agreement_band: Option<f64>,
        agreement_scale: Option<f64>,
    ) -> PyResult<Bound<'py, PyList>> {
        let opts = options(
            layer_depth_factor,
            offset_weight,
            placeholder_normal_factor,
            angle_weight,
            capped_radius_factor,
            unread_radius,
            agreement_band,
            agreement_scale,
        );
        let value = edited(self);
        let views = py
            .detach(|| core_endpoint_views(&value, image, [xy.0, xy.1], &opts))
            .map_err(|e: EndpointRefusal| PyValueError::new_err(e.to_string()))?;
        let out = PyList::empty(py);
        for v in &views {
            out.append(view_to_py(py, v)?)?;
        }
        Ok(out)
    }

    /// Resolve both endpoints of every distance measurement the reconstruction
    /// stores (``measurements["distance"]``).
    ///
    /// Args:
    ///     layer_depth_factor, offset_weight, placeholder_normal_factor,
    ///         angle_weight, capped_radius_factor, unread_radius,
    ///         agreement_band, agreement_scale: As
    ///         ``resolve_measurement_endpoint`` takes them.
    ///
    /// Returns:
    ///     A list with one dict per distance measurement, in stored order:
    ///     ``endpoints``, two resolutions as ``resolve_measurement_endpoint``
    ///     returns them (``None`` for an endpoint that has no position);
    ///     ``refusals``, two sentences naming the remedy (``None`` for an
    ///     endpoint that resolved); and ``scene_distance``, the distance
    ///     between the two positions in scene units, ``None`` unless both
    ///     resolved to finite positions. Empty where the reconstruction stores
    ///     no distance measurements.
    // A Python docstring, not Rust prose; see `resolve_measurement_endpoint`.
    #[allow(rustdoc::invalid_rust_codeblocks)]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        *, layer_depth_factor=None, offset_weight=None, placeholder_normal_factor=None,
        angle_weight=None, capped_radius_factor=None, unread_radius=None,
        agreement_band=None, agreement_scale=None,
    ))]
    fn resolve_distance_measurements<'py>(
        &self,
        py: Python<'py>,
        layer_depth_factor: Option<f64>,
        offset_weight: Option<f64>,
        placeholder_normal_factor: Option<f64>,
        angle_weight: Option<f64>,
        capped_radius_factor: Option<f64>,
        unread_radius: Option<f64>,
        agreement_band: Option<f64>,
        agreement_scale: Option<f64>,
    ) -> PyResult<Bound<'py, PyList>> {
        let opts = options(
            layer_depth_factor,
            offset_weight,
            placeholder_normal_factor,
            angle_weight,
            capped_radius_factor,
            unread_radius,
            agreement_band,
            agreement_scale,
        );
        let value = edited(self);
        let resolved = py.detach(|| core_resolve_all(&value, &opts));
        let out = PyList::empty(py);
        for m in &resolved {
            let d = PyDict::new(py);
            let endpoints = PyList::empty(py);
            let refusals = PyList::empty(py);
            for e in &m.endpoints {
                match e {
                    Ok(r) => {
                        endpoints.append(resolution_to_py(py, r)?)?;
                        refusals.append(py.None())?;
                    }
                    Err(refusal) => {
                        endpoints.append(py.None())?;
                        refusals.append(refusal.to_string())?;
                    }
                }
            }
            d.set_item("endpoints", endpoints)?;
            d.set_item("refusals", refusals)?;
            d.set_item("scene_distance", m.scene_distance)?;
            out.append(d)?;
        }
        Ok(out)
    }
}
