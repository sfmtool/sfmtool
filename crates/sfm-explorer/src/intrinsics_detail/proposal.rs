// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The camera model proposal's strip: the controls a proposal is fitted from,
//! the fit's one-line report, and Apply and Cancel.
//!
//! See `specs/gui/camera-intrinsics.md` § "The proposal strip". Every change of
//! a control goes through [`CameraModelProposal::set_inputs`], which refits at
//! once on the GUI thread; a change that leaves the inputs as they were keeps
//! the proposal's generation. A spline camera's own model, labelled as the
//! refit, carries the controls a refit takes: Coefficients and Spline domain,
//! each with the camera's own value beside it, and the outermost keypoint with
//! the button that sets the domain to its angle. Fit to θ is not shown for it,
//! because a given angle makes the core fit the camera like any other source.

use sfmtool_core::camera::refit_intrinsics::SPLINE_COEFF_COUNT_RANGE;

use crate::camera_model_proposal::{CameraModelProposal, ProposalInputs};

/// How the person ended the proposal from the strip.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ProposalAction {
    /// `Apply`: switch the camera and retriangulate its tracks, as one version.
    Apply,
    /// `Cancel`: drop the proposal.
    Cancel,
}

/// What the strip's Apply button is called.
pub(crate) const APPLY: &str = "Apply";

/// What the strip's Cancel button is called.
pub(crate) const CANCEL: &str = "Cancel";

/// Draw the strip for `proposal`, applying every change of a control to it,
/// and report Apply or Cancel.
pub(super) fn show_proposal(
    ui: &mut egui::Ui,
    proposal: &mut CameraModelProposal,
) -> Option<ProposalAction> {
    ui.label(egui::RichText::new("Proposal").strong());
    let mut inputs = proposal.inputs.clone();
    let refit = proposal.is_spline_refit(inputs.camera_model);

    // One wrapping row per control: at a narrow width what follows a control
    // (the camera's own value, the default and its button) moves to the next
    // line rather than past the panel's edge, so every control stays on
    // screen and can be reached.
    row(ui, "Model", |ui| {
        let mut chosen = inputs.camera_model;
        egui::ComboBox::from_id_salt("camera_model_proposal_model")
            .selected_text(proposal.target_label(chosen))
            .show_ui(ui, |ui| {
                for &target in &proposal.targets {
                    ui.selectable_value(&mut chosen, target, proposal.target_label(target));
                }
            });
        if chosen != inputs.camera_model {
            // A new target starts at its own defaults: the count it takes
            // from this camera, and the default angle and domain.
            inputs = proposal.default_inputs(chosen);
        }
    });

    if CameraModelProposal::has_spline(inputs.camera_model) {
        row(ui, "Coefficients", |ui| {
            let mut count = inputs.coeff_count.unwrap_or(8);
            // A value the wire gave outside the range is shown as it is, not
            // clamped: drawing the field must not change the request.
            ui.add(
                egui::DragValue::new(&mut count)
                    .range(SPLINE_COEFF_COUNT_RANGE)
                    .clamp_existing_to_range(false),
            )
            .on_hover_text(
                "The spline's coefficient count. The fit is the closest monotone curve to \
                 the current one; bundle adjustment with the lens distortion released then \
                 fits the coefficients to the observations.",
            );
            inputs.coeff_count = Some(count);
            if let Some(now) = proposal.source_coeff_count().filter(|_| refit) {
                part(ui, format!("now {now}"));
            }
        });
    }

    // A refit of a spline fits over its whole domain; a fit angle the wire
    // gave is shown so the proposal reads as it was opened. With no default
    // and none given there is no angle to show, and the field is left out
    // rather than showing a made-up one.
    if !refit || inputs.theta_fit_deg.is_some() {
        row(ui, "Fit to θ (°)", |ui| {
            let Some(shown_theta) = proposal.shown_theta_fit_deg() else {
                ui.weak("no default angle");
                return;
            };
            let upper = proposal
                .theta_fit_default
                .map_or(shown_theta, |(deg, _)| deg)
                .max(inputs.theta_fit_deg.unwrap_or(0.0));
            let mut theta = shown_theta;
            let before = theta;
            ui.add(
                egui::DragValue::new(&mut theta)
                    .range(0.1..=upper)
                    .clamp_existing_to_range(false)
                    .speed(0.1)
                    .fixed_decimals(1),
            )
            .on_hover_text(
                "The largest incidence angle the fit samples. It starts at the current model's \
                 trusted bound, or its observations' extent for a model without one, and can \
                 be lowered.",
            );
            if (theta - before).abs() > 1e-12 {
                inputs.theta_fit_deg = Some(theta);
            }
            if let Some((deg, source)) = proposal.theta_fit_default {
                part(ui, format!("default {deg:.1}° ({})", source.as_str()));
            }
            if inputs.theta_fit_deg.is_some() && ui.small_button(DEFAULT).clicked() {
                inputs.theta_fit_deg = None;
            }
        });
    }

    if CameraModelProposal::has_spline(inputs.camera_model) {
        row(ui, "Spline domain (°)", |ui| {
            let max = if inputs.camera_model == "SFMTOOL_PINHOLE" {
                89.9
            } else {
                180.0
            };
            let shown = proposal.shown_domain_deg().unwrap_or(max);
            let mut domain = inputs.spline_domain_deg.unwrap_or(shown);
            let before = domain;
            ui.add(
                egui::DragValue::new(&mut domain)
                    .range(0.1..=max)
                    .clamp_existing_to_range(false)
                    .speed(0.1)
                    .fixed_decimals(1),
            )
            .on_hover_text(
                "Where the spline's domain ends, as an incidence angle. Past it the model is a \
                 straight line bundle adjustment cannot bend.",
            );
            if (domain - before).abs() > 1e-12 {
                inputs.spline_domain_deg = Some(domain);
            }
            match proposal.source_domain_deg().filter(|_| refit) {
                Some(now) => part(ui, format!("now {now:.1}°")),
                None => part(ui, "default: the far image corner"),
            };
        });
    }

    // The outermost keypoint, beside the domain it should reach, with the
    // button that sets the domain to its angle.
    if CameraModelProposal::has_spline(inputs.camera_model) {
        if let Some(text) = proposal.keypoint_extent.describe() {
            ui.horizontal_wrapped(|ui| {
                ui.label(text);
                if let Some(deg) = proposal.keypoint_extent.suggested_domain_deg() {
                    if ui
                        .small_button(format!("Use {deg:.1}°"))
                        .on_hover_text(
                            "Set the domain to the outermost keypoint's angle: the detected \
                             one, or the observed one where no .sift file could be read.",
                        )
                        .clicked()
                    {
                        use_outermost_keypoint(proposal, &mut inputs);
                    }
                }
            });
        }
    }

    proposal.set_inputs(inputs);

    match &proposal.fit {
        Ok(_) => {
            ui.add(egui::Label::new(egui::RichText::new(proposal.fit_line()).monospace()).wrap())
        }
        Err(_) => ui.colored_label(ui.visuals().error_fg_color, proposal.fit_line()),
    };

    let mut action = None;
    ui.horizontal(|ui| {
        let apply = ui
            .add_enabled(proposal.fit.is_ok(), egui::Button::new(APPLY))
            .on_disabled_hover_text("The fit was refused, so there is nothing to apply.")
            .on_hover_text(
                "Switch the camera to the proposed model and retriangulate the tracks its \
                 images observe, as one version. Undo (Ctrl+Z) puts both back.",
            );
        if apply.clicked() {
            action = Some(ProposalAction::Apply);
        }
        if ui
            .button(CANCEL)
            .on_hover_text("Drop the proposal. Nothing changes.")
            .clicked()
        {
            action = Some(ProposalAction::Cancel);
        }
    });
    action
}

/// What the button that puts Fit to θ back to its default is called.
pub(crate) const DEFAULT: &str = "Default";

/// One row of the strip: its name, then `add`'s widgets, wrapping onto the
/// next line where the panel is too narrow for them.
fn row(ui: &mut egui::Ui, name: &str, add: impl FnOnce(&mut egui::Ui)) {
    ui.horizontal_wrapped(|ui| {
        part(ui, name);
        add(ui);
    });
}

/// A short piece of text kept on one line: the row wraps before it, not
/// inside it.
fn part(ui: &mut egui::Ui, text: impl Into<egui::WidgetText>) -> egui::Response {
    ui.add(egui::Label::new(text).wrap_mode(egui::TextWrapMode::Extend))
}

/// The domain row's button: the domain set to the outermost keypoint's angle,
/// detected where there is one and observed otherwise. Nothing happens with no
/// keypoint to take it from.
pub(crate) fn use_outermost_keypoint(proposal: &CameraModelProposal, inputs: &mut ProposalInputs) {
    if let Some(deg) = proposal.keypoint_extent.suggested_domain_deg() {
        inputs.spline_domain_deg = Some(deg);
    }
}
