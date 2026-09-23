// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The workspace path `File > Save As Minimal...` asks about, once the save
//! dialog has named a file.
//!
//! See `specs/gui/saving.md`. A minimal copy records no absolute workspace
//! path, so `workspace.relative_path` is the only thing in it that leads a
//! reader back to the images and the features. The save measures that path from
//! the directory the copy is written to, which is right whenever the copy stays
//! there, and wrong for a copy that is staged somewhere and then moved: a ground
//! truth written next to a candidate and checked into its workspace records `.`
//! and not the path from wherever it was produced. Only the person doing the
//! saving knows which of the two this is, so the copy asks.
//!
//! The dialog owns no policy. It holds the file the save dialog chose and the
//! text of the field, and reports both back; the caller writes the copy. The
//! field is prefilled with the measurement, so accepting it is the same save the
//! viewer made before this prompt existed.

use std::path::PathBuf;

use crate::scene::ReconId;

#[cfg(test)]
mod tests;

/// What the user asked for once they pressed Save.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SaveMinimalAnswer {
    /// The node to write a minimal copy of.
    pub recon: ReconId,
    /// Where to write it, as the save dialog named it.
    pub path: PathBuf,
    /// The `workspace.relative_path` to record, as the field left it. Empty is
    /// an answer: it records no path, which is what the format reads as none.
    pub workspace_path: String,
}

/// The dialog, and what it remembers while it is up.
#[derive(Default)]
pub struct SaveMinimalPrompt {
    pending: Option<Pending>,
}

/// The question being asked: which node, where the copy goes, and the text of
/// the field.
struct Pending {
    recon: ReconId,
    label: String,
    path: PathBuf,
    workspace_path: String,
    /// Set when the dialog opens, consumed by the first frame that draws the
    /// field, which is the frame that can focus it.
    focus_pending: bool,
}

impl SaveMinimalPrompt {
    /// Ask where the copy of `recon` at `path` says its workspace is, starting
    /// the field at `workspace_path`.
    ///
    /// Idempotent while the dialog is already up, so a second trip through the
    /// menu cannot stack two of them and cannot throw away a half-typed path.
    /// The caller has already chosen the file and already checked that the copy
    /// can be written there, so there is nothing left for this to refuse.
    pub fn ask(&mut self, recon: ReconId, label: String, path: PathBuf, workspace_path: String) {
        if self.pending.is_none() {
            self.pending = Some(Pending {
                recon,
                label,
                path,
                workspace_path,
                focus_pending: true,
            });
        }
    }

    /// Draw one frame, returning the answer on the frame Save is pressed.
    ///
    /// Enter saves and Escape cancels, the vocabulary the viewer's other
    /// prompts use. Cancelling writes nothing: the file dialog named a path but
    /// the save has not happened yet, and backing out here is backing out of the
    /// whole save rather than accepting a path nobody confirmed.
    pub fn show(&mut self, ctx: &egui::Context) -> Option<SaveMinimalAnswer> {
        let pending = self.pending.as_mut()?;
        let mut save = false;
        let mut cancel = false;
        let mut still_open = true;

        egui::Window::new("Save As Minimal")
            .open(&mut still_open)
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .show(ctx, |ui| {
                ui.label(format!(
                    "Writing a minimal copy of {} to {}.",
                    pending.label,
                    pending.path.display()
                ));
                ui.add_space(8.0);
                ui.label("Workspace path recorded in the copy:");
                let edit = ui.add(
                    egui::TextEdit::singleline(&mut pending.workspace_path)
                        .id(egui::Id::new("save_minimal_workspace_path"))
                        .hint_text("(none recorded)")
                        .desired_width(320.0),
                );
                if std::mem::take(&mut pending.focus_pending) {
                    edit.request_focus();
                }
                ui.label(
                    egui::RichText::new(
                        "The path a reader walks from the copy's own directory to reach the \
                         workspace: \".\" for a copy that sits inside it. It starts at the \
                         path measured to where the copy is going, so change it only for a \
                         copy that will live somewhere else. Leave it empty to record no \
                         path, and the reader falls back to the workspace holding the file.",
                    )
                    .weak()
                    .small(),
                );
                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    if ui.button("Save").clicked() {
                        save = true;
                    }
                    if ui.button("Cancel").clicked() {
                        cancel = true;
                    }
                });
                // `lost_focus` with Enter is how the field reports a submission:
                // egui delivers the key on the frame the field gives focus up.
                save |= edit.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
                cancel |= ui.input(|i| i.key_pressed(egui::Key::Escape));
            });

        let answer = save.then(|| SaveMinimalAnswer {
            recon: pending.recon,
            path: pending.path.clone(),
            workspace_path: pending.workspace_path.trim().to_string(),
        });
        if answer.is_some() || cancel || !still_open {
            self.pending = None;
        }
        answer
    }
}
