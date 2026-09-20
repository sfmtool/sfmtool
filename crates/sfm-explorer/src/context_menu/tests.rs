// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The open rule, driven through whole egui frames: a primary contact held
//! past `max_click_duration` is what the Windows left mouse button looks like
//! to egui, so the long touch has to be produced for real rather than asserted
//! about.

use egui::{Event, PointerButton, Pos2, TouchDeviceId, TouchId, TouchPhase};

/// egui's default `max_click_duration`, past which a held primary contact is a
/// long touch.
const LONG_PRESS: f64 = 0.8;

/// Which builder a frame puts the menu up with.
///
/// `Egui` is not a thing this crate uses; it is the control that proves the
/// event sequence really does produce a long touch, so that
/// [`a_held_primary_contact_opens_nothing`] cannot pass by failing to press at
/// all.
#[derive(Clone, Copy)]
enum Builder {
    Ours,
    Egui,
}

/// A context with one clickable widget in it, and a menu built on that widget.
struct Harness {
    ctx: egui::Context,
    builder: Builder,
    time: f64,
    /// Where the widget was drawn on the last frame.
    target: egui::Rect,
    /// Whether the menu was shown on the last frame.
    open: bool,
}

impl Harness {
    /// A harness with one frame already laid out: egui resolves a press
    /// against the widget rects of the *previous* pass, so nothing can be
    /// pressed until something has been drawn.
    fn new(builder: Builder) -> Self {
        let mut harness = Self {
            ctx: egui::Context::default(),
            builder,
            time: 0.0,
            target: egui::Rect::ZERO,
            open: false,
        };
        harness.frame(Vec::new());
        harness
    }

    /// One frame at the current time, fed `events`.
    fn frame(&mut self, events: Vec<Event>) {
        let input = egui::RawInput {
            time: Some(self.time),
            screen_rect: Some(egui::Rect::from_min_size(
                Pos2::ZERO,
                egui::vec2(200.0, 100.0),
            )),
            events,
            ..Default::default()
        };
        let builder = self.builder;
        let mut target = egui::Rect::ZERO;
        let mut open = false;
        crate::test_support::run_frame_headless(&self.ctx, input, |ui| {
            egui::CentralPanel::default().show(ui, |ui| {
                let response = ui.add(egui::Button::new("target"));
                target = response.rect;
                let popup = match builder {
                    Builder::Ours => super::on_secondary_click(&response),
                    Builder::Egui => egui::Popup::context_menu(&response),
                };
                open = popup
                    .show(|ui| {
                        ui.label("entry");
                    })
                    .is_some();
            });
        });
        self.target = target;
        self.open = open;
    }

    fn pos(&self) -> Pos2 {
        self.target.center()
    }

    /// The events `egui-winit` emits for the start of a contact, which on
    /// Windows is what a left mouse button press arrives as.
    fn contact_started(&mut self) {
        let pos = self.pos();
        self.frame(vec![
            Event::Touch {
                device_id: TouchDeviceId(0),
                id: TouchId(0),
                phase: TouchPhase::Start,
                pos,
                force: None,
            },
            Event::PointerMoved(pos),
            pointer(pos, PointerButton::Primary, true),
        ]);
    }

    /// The events `egui-winit` emits for the end of that contact, minus the
    /// `PointerGone` the Windows layer already suppresses the effect of.
    fn contact_ended(&mut self) {
        let pos = self.pos();
        self.frame(vec![
            Event::Touch {
                device_id: TouchDeviceId(0),
                id: TouchId(0),
                phase: TouchPhase::End,
                pos,
                force: None,
            },
            pointer(pos, PointerButton::Primary, false),
        ]);
    }

    /// Press and release `button` over the widget, the two frames egui needs
    /// to call it a click.
    fn click(&mut self, button: PointerButton) {
        let pos = self.pos();
        self.frame(vec![Event::PointerMoved(pos), pointer(pos, button, true)]);
        self.frame(vec![pointer(pos, button, false)]);
    }

    /// Advance the clock and run a frame with no input, the way egui's own
    /// press-and-hold deadline wakes the window up.
    fn wait(&mut self, seconds: f64) {
        self.time += seconds;
        self.frame(Vec::new());
    }
}

fn pointer(pos: Pos2, button: PointerButton, pressed: bool) -> Event {
    Event::PointerButton {
        pos,
        button,
        pressed,
        modifiers: egui::Modifiers::default(),
    }
}

/// The bug this exists for: on Windows a left mouse press reaches egui as a
/// touch contact, so resting it for `max_click_duration` is a long touch, and
/// `Popup::context_menu` reads a long touch as a secondary click. Left-clicking
/// anywhere in the window put the right-click menu up after a delay.
#[test]
fn a_held_primary_contact_opens_nothing() {
    let mut harness = Harness::new(Builder::Ours);
    harness.contact_started();
    harness.wait(LONG_PRESS + 0.2);
    assert!(!harness.open, "the menu opened on the long-press frame");
    harness.wait(0.1);
    assert!(!harness.open, "the menu opened on the frame after");
    harness.contact_ended();
    assert!(!harness.open, "the menu opened when the contact ended");
}

/// The control for the test above: the same events through egui's own builder
/// do open the menu, so the long touch is really being produced.
#[test]
fn a_held_primary_contact_is_what_eguis_own_menu_opens_on() {
    let mut harness = Harness::new(Builder::Egui);
    harness.contact_started();
    harness.wait(LONG_PRESS + 0.2);
    assert!(
        harness.open,
        "egui did not read the held contact as a long touch, \
         so the test above proves nothing"
    );
}

#[test]
fn a_secondary_click_opens_the_menu() {
    let mut harness = Harness::new(Builder::Ours);
    harness.click(PointerButton::Secondary);
    assert!(harness.open, "a right click did not open the menu");
    // And it stays up on the frames that follow, with no input at all.
    harness.frame(Vec::new());
    assert!(harness.open, "the menu closed on its own");
}

/// A quick left click is not the defect: the release resets egui's press clock,
/// so nothing is ever held long enough. It must also not open the menu by any
/// other route.
#[test]
fn a_quick_primary_contact_opens_nothing() {
    let mut harness = Harness::new(Builder::Ours);
    harness.contact_started();
    harness.wait(0.05);
    harness.contact_ended();
    assert!(!harness.open, "a quick left click opened the menu");
    harness.wait(LONG_PRESS + 0.2);
    assert!(!harness.open, "the menu opened after the contact was over");
}

/// The other half of the open rule: a primary click on the widget the menu
/// belongs to takes it back down, which is also what disarms the long touch,
/// since egui reports one as an ordinary click too.
#[test]
fn a_primary_click_closes_an_open_menu() {
    let mut harness = Harness::new(Builder::Ours);
    harness.click(PointerButton::Secondary);
    assert!(harness.open, "a right click did not open the menu");
    harness.click(PointerButton::Primary);
    assert!(!harness.open, "a left click left the menu standing");
}
