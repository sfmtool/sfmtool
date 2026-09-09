// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The prompt that stands between an unsaved edit and losing it.
//!
//! See `specs/gui/saving.md`. Closing a node whose cursor is not at the version
//! its file holds, or closing the window while any such node is loaded, asks
//! first. The three answers are the usual three, and they are the whole
//! vocabulary: **Save** writes and then does the thing, **Don't Save** does the
//! thing, **Cancel** does neither.
//!
//! The dialog owns no policy. It holds what was asked for and reports back the
//! answer; the caller knows what closing means -- one node, every node, or the
//! window -- and carries the answer out. That keeps the same prompt in front of
//! all three without the prompt knowing about any of them.

use crate::scene::ReconId;

#[cfg(test)]
mod tests;

/// What the user was about to do when the prompt appeared.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PendingClose {
    /// Close one node.
    Node(ReconId),
    /// Close every loaded node.
    All,
    /// Close the window.
    Quit,
}

/// What the user answered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CloseAnswer {
    /// Write the dirty nodes first, then go ahead.
    Save(PendingClose),
    /// Go ahead and lose the edits.
    Discard(PendingClose),
}

/// The modal, and the one thing it remembers: what is waiting on it.
#[derive(Default)]
pub struct ClosePrompt {
    pending: Option<PendingClose>,
}

impl ClosePrompt {
    /// Ask about `pending`.
    ///
    /// Idempotent while the prompt is already up, so a menu item racing a
    /// shortcut cannot stack two of them; the first question is the one that
    /// gets answered.
    pub fn ask(&mut self, pending: PendingClose) {
        if self.pending.is_none() {
            self.pending = Some(pending);
        }
    }

    /// Draw one frame, returning an answer on the frame one is given.
    ///
    /// `dirty` is what the sentence names, so the user is told which
    /// reconstructions are at stake rather than that "there are unsaved
    /// changes".
    pub fn show(&mut self, ctx: &egui::Context, dirty: &[String]) -> Option<CloseAnswer> {
        let pending = self.pending?;
        let mut answer = None;
        let mut cancelled = false;
        let mut still_open = true;

        egui::Window::new("Unsaved changes")
            .open(&mut still_open)
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .show(ctx, |ui| {
                ui.label(match dirty {
                    [one] => format!("{one} has changes that are not on disk."),
                    many => format!(
                        "{} reconstructions have changes that are not on disk: {}.",
                        many.len(),
                        many.join(", ")
                    ),
                });
                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    if ui.button("Save").clicked() {
                        answer = Some(CloseAnswer::Save(pending));
                    }
                    if ui.button("Don't Save").clicked() {
                        answer = Some(CloseAnswer::Discard(pending));
                    }
                    if ui.button("Cancel").clicked() {
                        cancelled = true;
                    }
                });
                cancelled |= ui.input(|i| i.key_pressed(egui::Key::Escape));
            });

        if answer.is_some() || cancelled || !still_open {
            self.pending = None;
        }
        answer
    }
}
