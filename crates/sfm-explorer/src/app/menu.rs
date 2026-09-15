// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Menu bar, accelerators, and panel-local cache invalidation.

use super::{save, UiParts, UiRequests};
use std::path::Path;

use crate::action_log::Kind;
use crate::dock::Tab;
use crate::goto_point;
use crate::state::AppState;
use crate::window::WindowHost;

pub(super) fn show(
    root_ui: &mut egui::Ui,
    parts: &mut UiParts<'_>,
    requests: &mut UiRequests,
    window_host: &mut dyn WindowHost,
) {
    let UiParts {
        app_state,
        viewer_3d,
        image_browser,
        image_detail,
        point_track_detail,
        intrinsics_detail,
        track_edit,
    } = parts;
    let UiRequests {
        close_all_requested,
        quit_requested,
        quit_from_menu,
    } = requests;
    egui::Panel::top("menu_bar").show(root_ui, |ui| {
        egui::MenuBar::new().ui(ui, |ui| {
            ui.menu_button("File", |ui| {
                if ui.button("Open...").clicked() {
                    // Multi-select, and every chosen file *appends* a
                    // node, a path that is already open included: that
                    // opens it a second time, as a second node with a
                    // history of its own.
                    if let Some(paths) = rfd::FileDialog::new()
                        .add_filter("SfM Reconstruction", &["sfmr"])
                        .pick_files()
                    {
                        for path in paths {
                            // `load_file` returns its failure rather
                            // than logging it, so the menu writes the
                            // line in the vocabulary of the person who
                            // asked. See `AppState::load_file`.
                            if let Err(message) = app_state.load_file(&path) {
                                app_state
                                    .action_log
                                    .fail(crate::action_log::Kind::File, message);
                            }
                        }
                    }
                    ui.close();
                }
                ui.separator();
                // Both act on the selected node, which is what every
                // other node-scoped command in this menu bar does.
                let target = app_state.selected_recon;
                let has_path = target
                    .and_then(|id| app_state.node(id))
                    .is_some_and(|node| node.path.is_some());
                let save = ui
                    .add_enabled(
                        has_path,
                        egui::Button::new("Save")
                            .shortcut_text(ui.ctx().format_shortcut(&save::SAVE_SHORTCUT)),
                    )
                    .on_disabled_hover_text(
                        "The selected reconstruction came from no file — use Save As",
                    );
                if save.clicked() {
                    let outcome = target.map(|id| app_state.save_node(id));
                    save::save_outcome(app_state, outcome);
                    ui.close();
                }
                let save_as = ui
                    .add_enabled(
                        target.is_some(),
                        egui::Button::new("Save As...")
                            .shortcut_text(ui.ctx().format_shortcut(&save::SAVE_AS_SHORTCUT)),
                    )
                    .on_disabled_hover_text("Select a reconstruction to save it");
                if save_as.clicked() {
                    let outcome = target.map(|id| save::save_as_with_dialog(app_state, id));
                    save::save_outcome(app_state, outcome);
                    ui.close();
                }
                ui.separator();
                if ui
                    .add_enabled(!app_state.scene.is_empty(), egui::Button::new("Close All"))
                    .clicked()
                {
                    *close_all_requested = true;
                    ui.close();
                }
                ui.separator();
                if ui.button("Load Demo Data...").clicked() {
                    app_state.show_demo_dialog = true;
                    ui.close();
                }
                ui.separator();
                if ui.button("Quit").clicked() {
                    // Not `send_viewport_cmd(ViewportCommand::Close)`:
                    // this app drives its own winit loop and never
                    // reads `full_output.viewport_output`, so the
                    // command was silently dropped and Quit did
                    // nothing. The flag is read straight after the
                    // egui pass, where the event loop can act on it.
                    *quit_requested = true;
                    *quit_from_menu = true;
                    ui.close();
                }
            });
            ui.menu_button("Edit", |ui| {
                let target = app_state.selected_recon;
                let can_undo = target.is_some_and(|id| app_state.can_undo(id));
                let can_redo = target.is_some_and(|id| app_state.can_redo(id));
                let undo = ui
                    .add_enabled(
                        can_undo,
                        egui::Button::new("Undo")
                            .shortcut_text(ui.ctx().format_shortcut(&UNDO_SHORTCUT)),
                    )
                    .on_disabled_hover_text("The selected reconstruction has nothing to undo");
                if undo.clicked() {
                    let outcome = target.map(|id| app_state.undo(id));
                    edit_outcome(app_state, outcome);
                    forget_selected(
                        target,
                        image_browser,
                        image_detail,
                        point_track_detail,
                        intrinsics_detail,
                        track_edit,
                    );
                    crate::camera_lock::resnap_camera_view(viewer_3d, app_state);
                    ui.close();
                }
                let redo = ui
                    .add_enabled(
                        can_redo,
                        egui::Button::new("Redo")
                            .shortcut_text(ui.ctx().format_shortcut(&REDO_SHORTCUT)),
                    )
                    .on_disabled_hover_text("The selected reconstruction has nothing to redo");
                if redo.clicked() {
                    let outcome = target.map(|id| app_state.redo(id));
                    edit_outcome(app_state, outcome);
                    forget_selected(
                        target,
                        image_browser,
                        image_detail,
                        point_track_detail,
                        intrinsics_detail,
                        track_edit,
                    );
                    crate::camera_lock::resnap_camera_view(viewer_3d, app_state);
                    ui.close();
                }
                ui.separator();
                let point = app_state.selected_point;
                let delete_point = ui
                    .add_enabled(
                        point.is_some(),
                        egui::Button::new("Delete Point")
                            .shortcut_text(ui.ctx().format_shortcut(&DELETE_POINT_SHORTCUT)),
                    )
                    .on_disabled_hover_text("Select a 3D point to delete it");
                if delete_point.clicked() {
                    let outcome = app_state.delete_selected_point();
                    edit_outcome(app_state, Some(outcome));
                    ui.close();
                }
                let image = app_state.selected_image;
                let delete_image = ui
                    .add_enabled(image.is_some(), egui::Button::new("Delete Image"))
                    .on_disabled_hover_text("Select an image to delete it");
                if delete_image.clicked() {
                    let outcome = image.map(|image| app_state.delete_image(image));
                    edit_outcome(app_state, outcome);
                    forget_selected(
                        image.map(|i| i.recon),
                        image_browser,
                        image_detail,
                        point_track_detail,
                        intrinsics_detail,
                        track_edit,
                    );
                    ui.close();
                }
                ui.separator();
                // Move Camera, and the two entries a held lock turns it
                // into. The gate is the lock's own, so the entry and the
                // lock cannot disagree about when a camera can be taken
                // in hand.
                let locked = viewer_3d.camera_lock.is_some();
                let lock_refusal = crate::camera_lock::refusal(app_state, viewer_3d);
                let move_camera = ui
                    .add_enabled(
                        lock_refusal.is_none() || locked,
                        egui::Button::new(if locked {
                            "Commit Camera Move"
                        } else {
                            "Move Camera"
                        })
                        .shortcut_text("M"),
                    )
                    .on_disabled_hover_text(lock_refusal.unwrap_or_default())
                    .on_hover_text(
                        "Take the camera you are looking through in hand: every \
                                 navigation input moves it, and committing keeps the pose \
                                 as one version of the reconstruction.",
                    );
                if move_camera.clicked() {
                    if locked {
                        // `move_camera` writes its own lines, success or
                        // refusal, in the document model's vocabulary;
                        // what comes back is the node whose geometry
                        // moved, and whose panel caches describe one it
                        // no longer holds.
                        if let Ok(moved) = crate::camera_lock::commit(viewer_3d, app_state) {
                            forget_selected(
                                moved,
                                image_browser,
                                image_detail,
                                point_track_detail,
                                intrinsics_detail,
                                track_edit,
                            );
                        }
                    } else if let Err(message) = crate::camera_lock::enter(viewer_3d, app_state) {
                        app_state
                            .action_log
                            .fail(crate::action_log::Kind::View, message);
                    }
                    ui.close();
                }
                let cancel_move = ui
                    .add_enabled(locked, egui::Button::new("Cancel Camera Move"))
                    .on_disabled_hover_text("No camera is being moved");
                if cancel_move.clicked() {
                    crate::camera_lock::cancel(viewer_3d, app_state);
                    ui.close();
                }
                ui.separator();
                // The gate is the edit's own, so the entry and the edit
                // cannot disagree about when the adjustment can run.
                let refusal = match target.and_then(|id| app_state.node(id)) {
                    Some(node) => crate::bundle_adjust_prompt::refusal(node.edited()),
                    None => Some("Select a reconstruction to adjust it".to_string()),
                };
                let adjust = ui
                    .add_enabled(refusal.is_none(), egui::Button::new("Bundle Adjust..."))
                    .on_disabled_hover_text(refusal.unwrap_or_default())
                    .on_hover_text(
                        "Refine every pose and point of the selected reconstruction \
                                 against its observations, as one version of it.",
                    );
                if adjust.clicked() {
                    if let Some(id) = target {
                        app_state.open_bundle_adjust(id);
                    }
                    ui.close();
                }
            });
            ui.menu_button("Go", |ui| {
                if ui
                    .add(
                        egui::Button::new("Go to Point...")
                            .shortcut_text(ui.ctx().format_shortcut(&goto_point::SHORTCUT)),
                    )
                    .clicked()
                {
                    app_state.open_goto_point();
                    ui.close();
                }
            });
            // No View menu: the display controls it used to hold belong
            // to the 3D viewport's own HUD (`viewer_3d/hud.rs`), on the
            // principle that a panel owns its controls. What *is*
            // app-global about the window is which panels are in it,
            // and that is the Panels menu below rather than a View one.
            ui.menu_button("Panels", |ui| {
                panels_menu(ui, app_state, window_host);
            });
        });
    });
}

pub(super) fn shortcuts(root_ui: &mut egui::Ui, parts: &mut UiParts<'_>) {
    let UiParts {
        app_state,
        viewer_3d,
        image_browser,
        image_detail,
        point_track_detail,
        intrinsics_detail,
        track_edit,
    } = parts;
    // Ctrl/Cmd+G opens the same dialog from anywhere, gated on egui's
    // own keyboard arbitration so a HUD `DragValue` — or the dialog's
    // own text field — keeps the key while it is being typed into.
    // `open` is idempotent, so racing the menu item is harmless.
    if !root_ui.ctx().egui_wants_keyboard_input()
        && root_ui.input_mut(|i| i.consume_shortcut(&goto_point::SHORTCUT))
    {
        app_state.open_goto_point();
    }

    // The Edit menu's shortcuts, under the same keyboard arbitration:
    // Delete is a printable-looking key that a text field must keep,
    // and undo belongs to whatever field is being typed into.
    // The File menu's save shortcuts, under the same arbitration: Ctrl+S
    // belongs to whatever field is being typed into while it is.
    if !root_ui.ctx().egui_wants_keyboard_input() {
        let (save, save_as) = root_ui.input_mut(|i| {
            (
                i.consume_shortcut(&save::SAVE_SHORTCUT),
                i.consume_shortcut(&save::SAVE_AS_SHORTCUT),
            )
        });
        let target = app_state.selected_recon;
        if save || save_as {
            let outcome = target.map(|id| {
                if save_as {
                    save::save_as_with_dialog(app_state, id)
                } else {
                    app_state.save_node(id)
                }
            });
            save::save_outcome(app_state, outcome);
        }
    }

    if !root_ui.ctx().egui_wants_keyboard_input() {
        let (undo, redo, delete) = root_ui.input_mut(|i| {
            (
                i.consume_shortcut(&UNDO_SHORTCUT),
                i.consume_shortcut(&REDO_SHORTCUT) || i.consume_shortcut(&REDO_SHORTCUT_ALT),
                i.consume_shortcut(&DELETE_POINT_SHORTCUT),
            )
        });
        let target = app_state.selected_recon;
        if undo || redo {
            let outcome = target.map(|id| {
                if undo {
                    app_state.undo(id)
                } else {
                    app_state.redo(id)
                }
            });
            edit_outcome(app_state, outcome);
            forget_selected(
                target,
                image_browser,
                image_detail,
                point_track_detail,
                intrinsics_detail,
                track_edit,
            );
            // A step of the cursor can move the very pose the viewport
            // is looking through, and camera view follows the value.
            crate::camera_lock::resnap_camera_view(viewer_3d, app_state);
        }
        if delete && app_state.selected_point.is_some() {
            let outcome = app_state.delete_selected_point();
            edit_outcome(app_state, Some(outcome));
        }
    }
}

// ── The Edit menu's shortcuts and its two shared helpers ─────────────────

/// Undo the selected reconstruction's newest version.
pub(crate) const UNDO_SHORTCUT: egui::KeyboardShortcut =
    egui::KeyboardShortcut::new(egui::Modifiers::COMMAND, egui::Key::Z);

/// Redo, in the spelling the menu shows.
pub(crate) const REDO_SHORTCUT: egui::KeyboardShortcut =
    egui::KeyboardShortcut::new(egui::Modifiers::COMMAND, egui::Key::Y);

/// Redo, in the spelling the rest of the desktop also accepts. Both are live;
/// only [`REDO_SHORTCUT`] is written beside the menu item, because a menu that
/// lists two spellings of one action reads as two actions.
pub(crate) const REDO_SHORTCUT_ALT: egui::KeyboardShortcut = egui::KeyboardShortcut::new(
    egui::Modifiers {
        command: true,
        shift: true,
        ..egui::Modifiers::NONE
    },
    egui::Key::Z,
);

/// Delete the selected 3D point. Plain Delete, and so live only while no text
/// field has the keyboard.
pub(crate) const DELETE_POINT_SHORTCUT: egui::KeyboardShortcut =
    egui::KeyboardShortcut::new(egui::Modifiers::NONE, egui::Key::Delete);

/// Report an edit's outcome, if one was attempted.
///
/// Every edit method writes its own success line, in the vocabulary of the
/// document model, so this exists for the refusals: an edit that could not run
/// says why, once, wherever it was asked for.
fn edit_outcome(state: &mut crate::state::AppState, outcome: Option<Result<(), String>>) {
    if let Some(Err(message)) = outcome {
        state
            .action_log
            .fail(crate::action_log::Kind::Edit, message);
    }
}

/// Drop the panel-local caches of `id` after an edit that renumbered its
/// images.
///
/// The counterpart of `dock.rs`'s `forget_recon`, reachable from the menu bar,
/// where the panels are in scope but the dock is not. A node keeps its
/// [`crate::scene::ReconId`] across an edit, so nothing here becomes
/// unreachable on its own the way a closed node's does -- it has to be dropped.
pub(super) fn forget_selected(
    id: Option<crate::scene::ReconId>,
    image_browser: &mut crate::image_browser::ImageBrowser,
    image_detail: &mut crate::image_detail::ImageDetail,
    point_track_detail: &mut crate::point_track_detail::PointTrackDetail,
    intrinsics_detail: &mut crate::intrinsics_detail::IntrinsicsDetail,
    track_edit: &mut crate::track_edit::TrackEdit,
) {
    let Some(id) = id else { return };
    image_browser.forget_recon(id);
    image_detail.forget_recon(id);
    point_track_detail.forget_recon(id);
    intrinsics_detail.forget_recon(id);
    track_edit.forget_recon(id);
}

// ── The Panels menu ──────────────────────────────────────────────────────

/// The body of the **Panels** menu: a checkbox per panel, then the three
/// layout-wide items.
///
/// Split out of `app.rs` so it can be drawn — and read back — in a headless
/// frame. It takes the window host because Save and Load carry the window's
/// placement as well as the panels; the frame passes a clone of its
/// `Arc<Window>` and the headless test passes a fake. A menu load applies the
/// window change mid-frame rather than at the top of one, so the *next* frame
/// is the first laid out at the new size — right for a human click, and not
/// worth a deferral.
pub(crate) fn panels_menu(ui: &mut egui::Ui, state: &mut AppState, host: &mut dyn WindowHost) {
    for tab in Tab::ALL {
        let mut open = state.is_panel_open(tab);
        if ui.checkbox(&mut open, tab.title()).clicked() {
            if open {
                state.show_panel(tab);
            } else {
                state.hide_panel(tab);
            }
            ui.close();
        }
    }
    ui.separator();
    if ui
        .button("Reset Layout")
        .on_hover_text("Put every panel back in its default place")
        .clicked()
    {
        state.reset_layout();
        ui.close();
    }
    ui.separator();
    if ui.button("Save Layout...").clicked() {
        save_layout(state);
        ui.close();
    }
    if ui.button("Load Layout...").clicked() {
        load_layout(state, host);
        ui.close();
    }
}

/// Panels ▸ Save Layout…: a save dialog, then the file.
///
/// The dialog opens on the default file (§ "The default layout file"), so the
/// common case — "keep it like this" — is Save Layout…, Enter, and the viewer
/// comes up this way next time.
fn save_layout(state: &mut AppState) {
    let mut dialog = rfd::FileDialog::new()
        .add_filter("Layout", &["json"])
        .set_file_name(crate::layout::DEFAULT_LAYOUT_FILE_NAME);
    if let Some(directory) =
        crate::layout::default_layout_path().and_then(|path| path.parent().map(Path::to_owned))
    {
        dialog = dialog.set_directory(directory);
    }
    let Some(path) = dialog.save_file() else {
        return;
    };
    match std::fs::write(&path, state.window_layout().to_json()) {
        Ok(()) => state
            .action_log
            .record(Kind::Layout, format!("Saved layout to {}", path.display())),
        Err(error) => state.action_log.fail(
            Kind::Layout,
            format!("Save layout to {}: {error}", path.display()),
        ),
    }
}

/// Panels ▸ Load Layout…: an open dialog, then the file — or a refusal that
/// leaves the window and the arrangement exactly as they were.
fn load_layout(state: &mut AppState, host: &mut dyn WindowHost) {
    let mut dialog = rfd::FileDialog::new().add_filter("Layout", &["json"]);
    if let Some(directory) =
        crate::layout::default_layout_path().and_then(|path| path.parent().map(Path::to_owned))
    {
        dialog = dialog.set_directory(directory);
    }
    let Some(path) = dialog.pick_file() else {
        return;
    };
    state.load_layout_file(host, &path);
}
