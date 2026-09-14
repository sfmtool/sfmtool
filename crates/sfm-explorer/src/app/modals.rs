// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Dialog rendering and deferred close requests.

use super::{menu::forget_selected, save::save_dirty_before_closing, UiParts, UiRequests};
use crate::dock::Tab;

pub(super) fn show(root_ui: &mut egui::Ui, parts: &mut UiParts<'_>, requests: &mut UiRequests) {
    let UiParts {
        app_state,
        image_browser,
        image_detail,
        point_track_detail,
        intrinsics_detail,
        ..
    } = parts;
    let UiRequests {
        close_all_requested,
        quit_requested,
        quit_from_menu,
    } = requests;
    // The Bundle Adjust dialog, and the adjustment it asks for. The
    // solve starts here and runs on a worker; the version it produces
    // is installed by the poll at the top of a later frame, which is
    // also where the panel caches it invalidates are dropped.
    if let Some(answer) = app_state.bundle_adjust_prompt.show(root_ui.ctx()) {
        let options = sfmtool_core::BundleAdjustOptions {
            opt_f: answer.release_focal,
            ..sfmtool_core::BundleAdjustOptions::default()
        };
        // The refusal is already an Action Log row: the start writes it
        // itself, in the words the menu's own gate uses.
        let _ = app_state.start_bundle_adjust(answer.recon, &options);
    }

    // The close prompt, and the answer to whichever question it asked.
    // Drawn before the panels so it sits over them, and answered here so
    // the close it was standing in front of happens on the same frame.
    // What the menu asked for goes through the prompt when something is
    // dirty; what the prompt answered goes straight through, since the
    // question has already been put.
    let mut close_all_now = false;
    let dirty = app_state.dirty_labels();
    if let Some(answer) = app_state.close_prompt.show(root_ui.ctx(), &dirty) {
        let (pending, save_first) = match answer {
            crate::close_prompt::CloseAnswer::Save(pending) => (pending, true),
            crate::close_prompt::CloseAnswer::Discard(pending) => (pending, false),
        };
        if !save_first || save_dirty_before_closing(app_state, pending) {
            match pending {
                crate::close_prompt::PendingClose::Node(id) => {
                    forget_selected(
                        Some(id),
                        image_browser,
                        image_detail,
                        point_track_detail,
                        intrinsics_detail,
                    );
                    if let Err(message) = app_state.close_node(id) {
                        app_state
                            .action_log
                            .fail(crate::action_log::Kind::File, message);
                    }
                }
                crate::close_prompt::PendingClose::All => close_all_now = true,
                crate::close_prompt::PendingClose::Quit => *quit_requested = true,
            }
        }
    }

    if std::mem::take(close_all_requested) {
        if app_state.any_dirty() {
            app_state
                .close_prompt
                .ask(crate::close_prompt::PendingClose::All);
        } else {
            close_all_now = true;
        }
    }
    if close_all_now {
        for node in &app_state.scene {
            let id = node.id;
            image_browser.forget_recon(id);
            image_detail.forget_recon(id);
            point_track_detail.forget_recon(id);
            intrinsics_detail.forget_recon(id);
        }
        if let Err(message) = app_state.close_all() {
            app_state
                .action_log
                .fail(crate::action_log::Kind::File, message);
        }
    }
    if *quit_from_menu && app_state.any_dirty() {
        *quit_requested = false;
        app_state
            .close_prompt
            .ask(crate::close_prompt::PendingClose::Quit);
    }

    if app_state.show_demo_dialog {
        let mut open = true;
        let mut load_clicked = false;
        egui::Window::new("Load Demo Data")
            .open(&mut open)
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .show(root_ui.ctx(), |ui| {
                ui.horizontal(|ui| {
                    ui.label("Number of points:");
                    ui.add(
                        egui::DragValue::new(&mut app_state.demo_num_points)
                            .range(1..=100_000)
                            .speed(10.0),
                    );
                });
                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    if ui.button("Load").clicked() {
                        load_clicked = true;
                    }
                    if ui.button("Cancel").clicked() {
                        app_state.show_demo_dialog = false;
                    }
                });
            });
        if !open {
            app_state.show_demo_dialog = false;
        }
        if load_clicked {
            // Same node-creation path as File > Open, so the demo load
            // resets the caches and selection too.
            app_state.load_demo(app_state.demo_num_points);
            app_state.show_demo_dialog = false;
        }
    }

    // Go to Point. `select_point` also selects the owning
    // reconstruction, so a pasted ID naming a *different* loaded file
    // moves the whole session there — which is what makes an ID copied
    // out of one session usable in the next.
    if let Some(point) =
        app_state
            .goto_point
            .show(root_ui.ctx(), &app_state.scene, app_state.selected_recon)
    {
        app_state.select_point(point);
        // Raise the panel that answers "what is this point?", so the
        // jump has something to show for itself even when Point Track
        // is tabbed behind Image Detail (which is the default layout)
        // — or closed, which `show_panel` re-opens at its home rather
        // than silently finding nothing to raise.
        app_state.show_panel(Tab::PointTrackDetail);
    }
}
