// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! File-menu save operations and shortcuts.

// ── The File menu's save shortcuts and their helpers ─────────────────────

/// Write the selected reconstruction over its own file.
pub(crate) const SAVE_SHORTCUT: egui::KeyboardShortcut =
    egui::KeyboardShortcut::new(egui::Modifiers::COMMAND, egui::Key::S);

/// Write the selected reconstruction to a file chosen in the dialog.
pub(crate) const SAVE_AS_SHORTCUT: egui::KeyboardShortcut = egui::KeyboardShortcut::new(
    egui::Modifiers {
        command: true,
        shift: true,
        ..egui::Modifiers::NONE
    },
    egui::Key::S,
);

/// Ask for a path and write `id` to it, through the same native dialog
/// `File ▸ Open` uses.
///
/// `Ok(())` with nothing written when the dialog was dismissed: choosing not to
/// choose a file is not a failure, and a log line saying so would be noise.
pub(super) fn save_as_with_dialog(
    state: &mut crate::state::AppState,
    id: crate::scene::ReconId,
) -> Result<(), String> {
    let suggested = state
        .node(id)
        .map(|node| format!("{}.sfmr", node.label))
        .unwrap_or_else(|| "reconstruction.sfmr".to_string());
    let Some(path) = rfd::FileDialog::new()
        .add_filter("SfM Reconstruction", &["sfmr"])
        .set_file_name(suggested)
        .save_file()
    else {
        return Ok(());
    };
    // The dialog has no field for a workspace path, and measuring it from where
    // the file lands is what an interactive save wants.
    state.save_node_as(id, &path, None)
}

/// Write everything the close prompt was standing in front of, and say whether
/// the close may go ahead.
///
/// A node with no file goes through the Save As dialog, so *Save* on demo data
/// is a real offer rather than a refusal. A write that fails, or
/// a dialog that is dismissed, stops the close: the point of the prompt is that
/// nothing is lost without an answer, and neither of those is one.
pub(super) fn save_dirty_before_closing(
    state: &mut crate::state::AppState,
    pending: crate::close_prompt::PendingClose,
) -> bool {
    let ids = match pending {
        crate::close_prompt::PendingClose::Node(id) => vec![id],
        _ => state.dirty_ids(),
    };
    for id in ids {
        let has_path = state.node(id).is_some_and(|node| node.path.is_some());
        let outcome = if has_path {
            state.save_node(id)
        } else {
            save_as_with_dialog(state, id)
        };
        if let Err(message) = outcome {
            state
                .action_log
                .fail(crate::action_log::Kind::File, message);
            return false;
        }
        if state.is_dirty(id) {
            // Save As was dismissed, so nothing was written and the node is
            // still where it was.
            return false;
        }
    }
    true
}

/// Report a save's outcome, if one was attempted.
///
/// The counterpart of the Edit menu's outcome handler: a save writes its own success line
/// naming the path and the version, so this exists for the refusals.
pub(super) fn save_outcome(
    state: &mut crate::state::AppState,
    outcome: Option<Result<(), String>>,
) {
    if let Some(Err(message)) = outcome {
        state
            .action_log
            .fail(crate::action_log::Kind::File, message);
    }
}

/// Ask for a path for a minimal copy of `id`, through the same native dialog
/// Save As uses, and then ask what workspace path it should record.
///
/// The suggested name is the node's label with `-minimal`, so the dialog does
/// not open on the node's own file, which a minimal copy never replaces.
/// `Ok(())` with nothing written when the dialog was dismissed.
///
/// **This writes nothing itself.** The chosen file goes into
/// [`crate::save_minimal_prompt::SaveMinimalPrompt`], and the copy is written on
/// the frame that prompt is answered (`super::modals::show`). The refusals are
/// asked here, between the two dialogs, so nobody is made to confirm a workspace
/// path for a save that cannot happen.
pub(super) fn save_minimal_with_dialog(
    state: &mut crate::state::AppState,
    id: crate::scene::ReconId,
) -> Result<(), String> {
    let suggested = state
        .node(id)
        .map(|node| format!("{}-minimal.sfmr", node.label))
        .unwrap_or_else(|| "reconstruction-minimal.sfmr".to_string());
    let Some(path) = rfd::FileDialog::new()
        .add_filter("SfM Reconstruction", &["sfmr"])
        .set_file_name(suggested)
        .save_file()
    else {
        return Ok(());
    };
    if let Some(why) = state.minimal_copy_refusal(id, &path) {
        return Err(why);
    }
    let label = state
        .node(id)
        .map(|node| node.label.clone())
        .unwrap_or_else(|| "the reconstruction".to_string());
    let workspace_path = state.minimal_copy_workspace_path(id, &path);
    state
        .save_minimal_prompt
        .ask(id, label, path, workspace_path);
    Ok(())
}
