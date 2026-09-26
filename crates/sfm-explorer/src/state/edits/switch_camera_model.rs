// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Switch one camera of a node to a camera model fitted to it, applied
//! directly as the node's next version.
//!
//! See `specs/gui/edits/switch-camera-model.md`. The mechanism is
//! `sfmtool_core::reconstruction::switch_camera_model` over one camera. A spline
//! camera switched to its own spline model is a refit of its spline to a new
//! coefficient count or domain end, which is what the Camera Intrinsics panel's
//! `Refit spline…` action asks for; the MCP tool `switch_camera_model` reaches
//! the same call with any target model.

use std::sync::Arc;
use std::time::Instant;

use sfmtool_core::camera::refit_intrinsics::{
    spline_domain_deg, MonotoneConstraint, RefitOptions, RefitTarget,
};
use sfmtool_core::reconstruction::switch_camera_model::{switch_camera_model, CameraSwitch};
use sfmtool_core::{EditedReconstruction, RowMap, SfmrReconstruction};

use crate::action_log::{version_step_text, Kind};
use crate::document::PointMap;
use crate::progress::Collector;
use crate::scene::ReconId;
use crate::state::AppState;

use super::version_before;

/// What a switch of one camera asks for. Every field but the camera has a
/// default, resolved against the camera as it is when the switch runs.
#[derive(Debug, Clone, PartialEq, Default)]
pub(crate) struct SwitchCameraModelRequest {
    /// The camera's index in the node's camera table.
    pub(crate) camera: usize,
    /// The target model's name, or `None` for the camera's own model.
    pub(crate) camera_model: Option<String>,
    /// The spline coefficient count, or `None` for the camera's own count when
    /// the target is its own spline model, and the core default otherwise.
    pub(crate) coeff_count: Option<usize>,
    /// Where a spline target's domain ends, in degrees, or `None` for the
    /// camera's own domain end in a refit of its spline, and the far image
    /// corner otherwise.
    pub(crate) spline_domain_deg: Option<f64>,
    /// The largest incidence angle the fit samples, or `None` for the core's
    /// default. Given, it makes even a refit of a spline an ordinary fit over
    /// that angle.
    pub(crate) theta_fit_deg: Option<f64>,
}

/// The monotonicity constraint's clause of a log row: empty when it did not
/// bind, and otherwise `; monotone constraint bound at 3 angles, 95.2°–118.7°`.
pub(crate) fn monotone_clause(constraint: &MonotoneConstraint) -> String {
    match constraint.range_deg {
        Some([from, to]) if constraint.active => {
            let count = constraint.active_angles;
            let plural = if count == 1 { "" } else { "s" };
            let (from, to) = (format!("{from:.1}"), format!("{to:.1}"));
            let range = if from == to {
                format!("{from}°")
            } else {
                format!("{from}°–{to}°")
            };
            format!("; monotone constraint bound at {count} angle{plural}, {range}")
        }
        _ => String::new(),
    }
}

/// Whether `target` is `camera`'s own spline model, which makes the switch a
/// refit of its spline rather than a change of model.
fn is_spline_refit(camera: &sfmtool_core::CameraIntrinsics, target: &str) -> bool {
    camera.model.radial_spline().is_some() && camera.model_name().eq_ignore_ascii_case(target)
}

/// The version label for one switch: what changed about the camera, in one
/// line.
///
/// A refit of a spline reads `Refit spline of camera 0 of kerry: 8 → 10
/// coefficients, domain 150.0° → 108.0°`, naming each of the two only as
/// "kept" where it did not move; a change of model reads `Switched camera 0 of
/// kerry from OPENCV_FISHEYE to SFMTOOL_FISHEYE`.
pub(crate) fn switch_label(label: &str, entry: &CameraSwitch) -> String {
    let source = &entry.source;
    let target = &entry.refit.camera;
    let c = entry.camera;
    if source.model_name() != target.model_name() || source.model.radial_spline().is_none() {
        return format!(
            "Switched camera {c} of {label} from {} to {}",
            source.model_name(),
            target.model_name()
        );
    }
    let count = |camera: &sfmtool_core::CameraIntrinsics| {
        camera.model.radial_spline().map_or(0, |(b, _, _)| b.len())
    };
    let (n0, n1) = (count(source), count(target));
    let coeffs = if n0 == n1 {
        format!("{n1} coefficients kept")
    } else {
        format!("{n0} → {n1} coefficients")
    };
    let (d0, d1) = (
        spline_domain_deg(source).unwrap_or(f64::NAN),
        spline_domain_deg(target).unwrap_or(f64::NAN),
    );
    let domain = if (d0 - d1).abs() <= 1e-9 {
        format!("domain {d1:.1}° kept")
    } else {
        format!("domain {d0:.1}° → {d1:.1}°")
    };
    format!("Refit spline of camera {c} of {label}: {coeffs}, {domain}")
}

/// The report half of the Action Log row: how far the new camera is from the
/// old over the fit, where the monotonicity constraint bound, and the median
/// reprojection error of the camera's observations before and after.
pub(crate) fn switch_report_text(entry: &CameraSwitch) -> String {
    let refit = &entry.refit;
    let o = &entry.observations;
    let median = if o.observations > 0 {
        format!(
            "; median error {:.3} → {:.3} px over {} observations",
            o.before.median_px, o.after.median_px, o.observations
        )
    } else {
        String::new()
    };
    format!(
        "fit rms {:.3} px, max {:.3} px over θ ≤ {:.1}° ({}){}{median}",
        refit.rms_px,
        refit.max_px,
        refit.theta_fit_deg,
        refit.theta_fit_source.as_str(),
        monotone_clause(&refit.monotone_constraint),
    )
}

impl AppState {
    /// Switch one camera of `id` to the model `request` names, fitted to it,
    /// and install the answer as the node's next version.
    ///
    /// A bulk edit: the camera table changes and the stored error of every
    /// point the camera's images observe is recomputed, so the next version is
    /// a whole new base; no pose, point, keypoint or track moves, and the row
    /// map read off the two values is the identity. Records its own outcome,
    /// success or refusal, as one Action Log entry, and returns the camera's
    /// entry of the core report for a caller that shows the numbers.
    pub(crate) fn switch_camera_model(
        &mut self,
        id: ReconId,
        request: &SwitchCameraModelRequest,
    ) -> Result<CameraSwitch, String> {
        let started = Instant::now();
        let collector = Collector::new(self.action_log.detailed_timing());
        let outcome = self.switch_camera_model_inner(id, request, &collector);
        match outcome {
            Ok((message, entry)) => {
                self.action_log
                    .record_done(Kind::Edit, started, message, collector.take());
                Ok(entry)
            }
            Err(message) => {
                self.action_log.fail(Kind::Edit, message.clone());
                Err(message)
            }
        }
    }

    /// The edit itself: `Ok` carries the Action Log's sentence and the
    /// camera's report entry, `Err` the refusal's sentence.
    fn switch_camera_model_inner(
        &mut self,
        id: ReconId,
        request: &SwitchCameraModelRequest,
        collector: &Collector,
    ) -> Result<(String, CameraSwitch), String> {
        if let Some(why) = self.busy_refusal(id) {
            return Err(why);
        }
        let index = self
            .scene
            .iter()
            .position(|n| n.id == id)
            .ok_or_else(|| "That reconstruction is no longer loaded.".to_string())?;
        let label = self.scene[index].label.clone();
        let c = request.camera;
        let refuse = |why: String| format!("Switch camera model of {label} refused: {why}");
        let cameras = &self.scene[index].recon().image_table.cameras;
        let Some(camera) = cameras.get(c) else {
            return Err(refuse(format!(
                "camera {c} does not exist; the reconstruction has {} camera(s)",
                cameras.len()
            )));
        };
        let camera_model = request
            .camera_model
            .clone()
            .unwrap_or_else(|| camera.model_name().to_string());
        let refit = is_spline_refit(camera, &camera_model);
        let coeff_count = match (request.coeff_count, refit) {
            (Some(n), _) => Some(n),
            (None, true) => camera.model.radial_spline().map(|(b, _, _)| b.len()),
            (None, false) => None,
        };
        let target = RefitTarget::from_name(&camera_model, coeff_count)
            .map_err(|e| refuse(format!("camera {c}: {e}")))?;
        let options = RefitOptions {
            theta_fit_deg: request.theta_fit_deg,
            spline_domain_deg: request.spline_domain_deg,
        };

        // Materialise only when there is an overlay to fold in; an empty one
        // materialises to its own base, which the switch can read directly.
        let edited = self.scene[index].history.current();
        let (materialised, mat_map) =
            if edited.deleted_points.is_empty() && edited.added.points.is_empty() {
                (None, None)
            } else {
                let _phase = collector.phase("materialise");
                let (value, map) = edited.materialize();
                (Some(value), Some(PointMap::Rows(map)))
            };
        let source: &SfmrReconstruction = match materialised.as_ref() {
            Some(value) => value,
            None => &self.scene[index].history.current().base,
        };

        let (switched, report) = {
            let _phase = collector.phase("switch camera model");
            switch_camera_model(source, &[c], &target, &options)
                .map_err(|e| refuse(e.to_string()))?
        };
        let entry = report
            .cameras
            .into_iter()
            .next()
            .expect("one camera was named");
        // The switch deletes and creates no points, so this scan is the
        // identity map -- read off the two values rather than asserted.
        let scan = {
            let _phase = collector.phase("row map");
            RowMap::by_scan(source, &switched, None).map_err(|e| refuse(e.to_string()))?
        };
        let mut steps = Vec::new();
        steps.extend(mat_map);
        steps.push(PointMap::Rows(scan));
        let map = PointMap::Chain(steps);

        let text = switch_label(&label, &entry);
        let node = &mut self.scene[index];
        let serial = {
            let _phase = collector.phase("push version");
            node.history.push(
                EditedReconstruction::new(Arc::new(switched)),
                map,
                text.clone(),
            )
        };
        let parent = version_before(node, serial);
        self.follow_selection_forward(id);
        let message = version_step_text(
            &format!("{text}: {}", switch_report_text(&entry)),
            parent,
            serial,
        );
        Ok((message, entry))
    }
}
