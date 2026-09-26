// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! `switch_camera_model` on both reconstruction types: switching cameras to
//! another camera model as a fit (see
//! ``specs/core/reconstruction/switch-camera-model.md``).
//!
//! `EditedReconstruction` gets it as a bulk edit, the way the viewer runs it.
//! `SfmrReconstruction` gets the same call for `sfm xform`, which works on
//! plain reconstructions; both reach one core function with the same value.

use std::sync::Arc;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use sfmtool_core::camera::refit_intrinsics::{RefitOptions, RefitTarget};
use sfmtool_core::reconstruction::edited::EditedReconstruction;
use sfmtool_core::reconstruction::outermost_keypoint::{
    outermost_keypoints as core_outermost, KeypointReach, OutermostKeypoints,
};
use sfmtool_core::reconstruction::switch_camera_model::{
    switch_camera_model as core_switch, ErrorSummary, SwitchCameraModelReport,
};
use sfmtool_core::SfmrReconstruction;

use super::edited::{materialised, PyEditedReconstruction};
use super::sfmr_reconstruction::PySfmrReconstruction;
use crate::geometry::camera_intrinsics::refit_report_to_py;
use crate::PyCameraIntrinsics;

fn summary_to_py<'py>(py: Python<'py>, s: &ErrorSummary) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("median_px", s.median_px)?;
    d.set_item("p90_px", s.p90_px)?;
    d.set_item("max_px", s.max_px)?;
    Ok(d)
}

/// One keypoint's reach as a dict: ``radius_px``, ``theta_deg``, ``image``
/// and ``xy``; ``None`` for no keypoint.
pub(crate) fn reach_to_py(py: Python<'_>, r: Option<&KeypointReach>) -> PyResult<Py<PyAny>> {
    let Some(r) = r else {
        return Ok(py.None());
    };
    let d = PyDict::new(py);
    d.set_item("radius_px", r.radius_px)?;
    d.set_item("theta_deg", r.theta_deg)?;
    d.set_item("image", r.image)?;
    d.set_item("xy", (r.xy[0], r.xy[1]))?;
    Ok(d.into_any().unbind())
}

/// One camera's outermost keypoints as a dict: ``camera``, ``images``,
/// ``observed`` and ``detected`` (each a reach dict or ``None``) and
/// ``detected_images``.
fn outermost_to_py<'py>(py: Python<'py>, o: &OutermostKeypoints) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("camera", o.camera)?;
    d.set_item("images", o.images)?;
    d.set_item("observed", reach_to_py(py, o.observed.as_ref())?)?;
    d.set_item("detected", reach_to_py(py, o.detected.as_ref())?)?;
    d.set_item("detected_images", o.detected_images)?;
    Ok(d)
}

fn report_to_py<'py>(py: Python<'py>, r: &SwitchCameraModelReport) -> PyResult<Bound<'py, PyDict>> {
    let cameras = PyList::empty(py);
    for entry in &r.cameras {
        let c = PyDict::new(py);
        c.set_item("camera", entry.camera)?;
        c.set_item("images", entry.images)?;
        c.set_item(
            "source",
            PyCameraIntrinsics {
                inner: entry.source.clone(),
            },
        )?;
        c.set_item(
            "target",
            PyCameraIntrinsics {
                inner: entry.refit.camera.clone(),
            },
        )?;
        c.set_item("fit", refit_report_to_py(py, &entry.refit)?)?;
        let o = &entry.observations;
        let obs = PyDict::new(py);
        obs.set_item("observations", o.observations)?;
        obs.set_item("unmeasured", o.unmeasured)?;
        obs.set_item("max_theta_deg", o.max_theta_deg)?;
        obs.set_item("before", summary_to_py(py, &o.before)?)?;
        obs.set_item("after", summary_to_py(py, &o.after)?)?;
        obs.set_item("changed_over_1px", o.changed_over_1px)?;
        obs.set_item("trusted_deg", o.trusted_deg)?;
        obs.set_item("past_trusted", o.past_trusted)?;
        obs.set_item(
            "past_trusted_before",
            summary_to_py(py, &o.past_trusted_before)?,
        )?;
        obs.set_item(
            "past_trusted_after",
            summary_to_py(py, &o.past_trusted_after)?,
        )?;
        c.set_item("observations", obs)?;
        c.set_item("outermost", outermost_to_py(py, &entry.outermost)?)?;
        cameras.append(c)?;
    }
    let d = PyDict::new(py);
    d.set_item("cameras", cameras)?;
    Ok(d)
}

/// Resolve the Python arguments and run the core switch on `value`.
fn run(
    py: Python<'_>,
    value: &SfmrReconstruction,
    target: &str,
    cameras: Option<Vec<usize>>,
    coeff_count: Option<usize>,
    theta_fit_deg: Option<f64>,
    spline_domain_deg: Option<f64>,
) -> PyResult<(SfmrReconstruction, Py<PyDict>)> {
    let target = RefitTarget::from_name(target, coeff_count)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let options = RefitOptions {
        theta_fit_deg,
        spline_domain_deg,
    };
    let cameras = cameras.unwrap_or_else(|| (0..value.image_table.cameras.len()).collect());
    let (next, report) = py
        .detach(|| core_switch(value, &cameras, &target, &options))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((next, report_to_py(py, &report)?.unbind()))
}

#[pymethods]
impl PyEditedReconstruction {
    /// Switch cameras of this version to another camera model, and give back
    /// the answer as its successor.
    ///
    /// Each camera is replaced by a camera of ``target`` fitted to it over the
    /// angles where it is trusted (see
    /// ``specs/core/reconstruction/switch-camera-model.md``). Poses, points,
    /// keypoints, patches and tracks are unchanged; the stored errors of the
    /// points the switched cameras' images observe are recomputed. A **bulk**
    /// edit, so the value that comes back is a whole new base with an empty
    /// overlay, and this object is not changed.
    ///
    /// Args:
    ///     target: The model name, case-insensitive: ``SFMTOOL_FISHEYE``,
    ///         ``SFMTOOL_PINHOLE``, ``EQUIDISTANT_FISHEYE`` or a COLMAP lens
    ///         model.
    ///     cameras: Camera-table indexes to switch (default: every camera).
    ///     coeff_count: Spline coefficients for the two spline models (default
    ///         8). Refused for any other model.
    ///     theta_fit_deg: The largest incidence angle the fit samples.
    ///         Default: the camera's trusted bound, or, for a model with none,
    ///         the largest incidence angle among its images' observations.
    ///     spline_domain_deg: Where a spline target's domain ends, as an
    ///         incidence angle. Default: the far image corner.
    ///
    /// Returns:
    ///     ``(EditedReconstruction, report)``. The report's ``cameras`` holds
    ///     one dict per switched camera: ``camera``, ``images``, ``source`` and
    ///     ``target`` (the two ``CameraIntrinsics``), ``fit`` (the dict
    ///     ``CameraIntrinsics.refit`` reports) and ``observations``, the
    ///     comparison over one fixed set of observations: ``observations``,
    ///     ``unmeasured``, ``max_theta_deg``, ``before`` and ``after`` (each
    ///     ``median_px``, ``p90_px``, ``max_px``), ``changed_over_1px``,
    ///     ``trusted_deg``, ``past_trusted``, ``past_trusted_before`` and
    ///     ``past_trusted_after``; and ``outermost``, the camera's outermost
    ///     keypoint under the switched camera, as
    ///     ``SfmrReconstruction.outermost_keypoints`` reports it. Raises
    ///     ``ValueError`` naming the camera, the rule and the value when the
    ///     switch is refused.
    // This is a Python docstring (rendered by `help()`), not Rust prose: its
    // indented `Args:` / `Returns:` continuation paragraphs read as Markdown
    // indented code blocks, which rustdoc then tries to parse as Rust.
    #[allow(rustdoc::invalid_rust_codeblocks)]
    #[pyo3(signature = (target, *, cameras=None, coeff_count=None, theta_fit_deg=None, spline_domain_deg=None))]
    fn switch_camera_model(
        &self,
        py: Python<'_>,
        target: &str,
        cameras: Option<Vec<usize>>,
        coeff_count: Option<usize>,
        theta_fit_deg: Option<f64>,
        spline_domain_deg: Option<f64>,
    ) -> PyResult<(PyEditedReconstruction, Py<PyDict>)> {
        let value = materialised(&self.inner);
        let (next, report) = run(
            py,
            &value,
            target,
            cameras,
            coeff_count,
            theta_fit_deg,
            spline_domain_deg,
        )?;
        Ok((
            PyEditedReconstruction {
                inner: EditedReconstruction::new(Arc::new(next)),
            },
            report,
        ))
    }
}

#[pymethods]
impl PySfmrReconstruction {
    /// Switch cameras of this reconstruction to another camera model.
    ///
    /// The same call as ``EditedReconstruction.switch_camera_model``, on a
    /// plain reconstruction, for ``sfm xform --camera-model``. Returns
    /// ``(SfmrReconstruction, report)``; this object is not changed.
    #[pyo3(signature = (target, *, cameras=None, coeff_count=None, theta_fit_deg=None, spline_domain_deg=None))]
    fn switch_camera_model(
        &self,
        py: Python<'_>,
        target: &str,
        cameras: Option<Vec<usize>>,
        coeff_count: Option<usize>,
        theta_fit_deg: Option<f64>,
        spline_domain_deg: Option<f64>,
    ) -> PyResult<(PySfmrReconstruction, Py<PyDict>)> {
        let (next, report) = run(
            py,
            &self.inner,
            target,
            cameras,
            coeff_count,
            theta_fit_deg,
            spline_domain_deg,
        )?;
        Ok((PySfmrReconstruction { inner: next }, report))
    }

    /// The keypoint of each camera's images that lies furthest from its
    /// principal point, observed and detected.
    ///
    /// See ``specs/core/reconstruction/outermost-keypoint.md``. **Observed**
    /// is over this reconstruction's observations; **detected** is over every
    /// feature of the images' ``.sift`` files, read when ``read_sift_files``
    /// is set, skipping any file that cannot be read. Each keypoint's angle is
    /// the incidence angle this reconstruction's camera model gives its pixel.
    ///
    /// Args:
    ///     cameras: Camera-table indexes (default: every camera).
    ///     read_sift_files: Read the images' ``.sift`` files (default
    ///         ``True``). Without them a ``sift_files`` reconstruction has no
    ///         observed pixel either.
    ///
    /// Returns:
    ///     A list with one dict per camera, in ascending order: ``camera``,
    ///     ``images``, ``observed`` and ``detected`` (each ``None`` or a dict
    ///     of ``radius_px``, ``theta_deg``, ``image`` and ``xy``) and
    ///     ``detected_images``, the number of ``.sift`` files read.
    #[allow(rustdoc::invalid_rust_codeblocks)]
    #[pyo3(signature = (*, cameras=None, read_sift_files=true))]
    fn outermost_keypoints<'py>(
        &self,
        py: Python<'py>,
        cameras: Option<Vec<usize>>,
        read_sift_files: bool,
    ) -> PyResult<Bound<'py, PyList>> {
        let cameras =
            cameras.unwrap_or_else(|| (0..self.inner.image_table.cameras.len()).collect());
        let reach = py.detach(|| core_outermost(&self.inner, &cameras, read_sift_files));
        let out = PyList::empty(py);
        for entry in &reach {
            out.append(outermost_to_py(py, entry)?)?;
        }
        Ok(out)
    }
}
